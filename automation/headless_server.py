import os
import re
import sys
import shutil
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional

try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Add root directory to path
sys.path.append(os.getcwd())

# Import the workflow functions directly
# This avoids loading the entire Gradio UI stack
from main.app.tabs.automation.child.automation import (
    automation_workflow,
    train_workflow,
    convert_workflow,
    _pick_latest_model_file,
    _pick_index_file,
)

import threading
import uuid
import time
from typing import Dict, Any

app = FastAPI(title="RVC Headless Automation API")

# =====================================================================================
# BẢO MẬT: xác thực bằng API key qua header "X-API-Key"
# Đặt key qua biến môi trường API_KEY. Nếu không đặt -> mở (cảnh báo, dùng cho dev).
# Endpoint /health luôn mở để Docker healthcheck hoạt động.
# =====================================================================================
API_KEY = os.environ.get("API_KEY", "").strip()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

if not API_KEY:
    print("⚠️  CẢNH BÁO: Chưa đặt API_KEY -> API đang MỞ, ai cũng gọi được. "
          "Đặt biến môi trường API_KEY để bật bảo mật.")

def verify_api_key(key: str = Security(api_key_header)):
    """Chặn request nếu API_KEY được cấu hình mà header X-API-Key sai/thiếu."""
    if not API_KEY:
        return  # không cấu hình key -> không bảo vệ
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="API key sai hoặc thiếu (header X-API-Key).")

class AutomationRequest(BaseModel):
    training_files: List[str]
    target_song_path: str
    model_name: str
    epochs: int = 20
    pitch_shift: int = 0
    force_retrain: bool = False

# In-memory storage for tasks
# Structure: { task_id: { "status": str, "message": str, "result_path": str, "logs": str } }
TASKS: Dict[str, Dict[str, Any]] = {}

import queue

# Global Queue for sequential processing.
# Item: (task_id, kind, payload) — kind: "full" | "train" | "convert"
TASK_QUEUE = queue.Queue()

# =====================================================================================
# DB ĐĂNG KÝ MODEL THEO KHÁCH HÀNG (Postgres, cấu hình qua .env)
# =====================================================================================
DB_HOST = os.environ.get("DB_HOST", "")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "")
DB_USER = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
MODEL_TABLE = os.environ.get("MODEL_TABLE", "rvc_customer_models")

def _db_enabled() -> bool:
    return bool(DB_HOST and DB_NAME and psycopg2 is not None)

def _db_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, connect_timeout=5,
    )

def init_model_table():
    if not _db_enabled():
        print("⚠️  DB chưa cấu hình (DB_HOST/DB_NAME) -> đăng ký model chỉ dựa trên filesystem.")
        return
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {MODEL_TABLE} (
                    customer_id VARCHAR PRIMARY KEY,
                    model_name  VARCHAR NOT NULL,
                    model_file  VARCHAR,
                    index_file  VARCHAR,
                    epochs      INTEGER,
                    trained_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                )""")
        print(f"DB model registry sẵn sàng (bảng {MODEL_TABLE}).")
    except Exception as e:
        print(f"⚠️  Không khởi tạo được bảng model: {e}")

def save_customer_model(customer_id, model_name, model_file, index_file, epochs):
    """Upsert bản ghi model của khách hàng sau khi train xong."""
    if not _db_enabled():
        return
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {MODEL_TABLE} (customer_id, model_name, model_file, index_file, epochs, trained_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (customer_id) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    model_file = EXCLUDED.model_file,
                    index_file = EXCLUDED.index_file,
                    epochs     = EXCLUDED.epochs,
                    trained_at = NOW(),
                    updated_at = NOW()
                """, (customer_id, model_name, model_file, index_file, epochs))
        print(f"[DB] Đã lưu model của khách {customer_id}: {model_file}")
    except Exception as e:
        print(f"[DB] Lỗi lưu model khách {customer_id}: {e}")

def get_customer_model(customer_id):
    if not _db_enabled():
        return None
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT customer_id, model_name, model_file, index_file, epochs, trained_at, updated_at "
                f"FROM {MODEL_TABLE} WHERE customer_id = %s", (customer_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "customer_id": row[0], "model_name": row[1], "model_file": row[2],
                "index_file": row[3], "epochs": row[4],
                "trained_at": str(row[5]), "updated_at": str(row[6]),
            }
    except Exception as e:
        print(f"[DB] Lỗi đọc model khách {customer_id}: {e}")
        return None

def worker_loop():
    """Consumes tasks from the queue and processes them sequentially."""
    print("Worker thread started, waiting for tasks...")
    while True:
        try:
            # Block until a task is available
            task_id, kind, payload = TASK_QUEUE.get()
            print(f"[Worker] Picked up task {task_id} ({kind})")

            run_automation_task(task_id, kind, payload)

            TASK_QUEUE.task_done()
            print(f"[Worker] Finished task {task_id}, remaining in queue: {TASK_QUEUE.qsize()}")

        except Exception as e:
            print(f"[Worker] Critical Error in worker loop: {e}")
            import traceback
            traceback.print_exc()

def run_automation_task(task_id: str, kind: str, payload: dict):
    """Chạy 1 task theo loại: full (train+convert), train (chỉ train), convert (chỉ đổi giọng)."""
    print(f"[Task {task_id}] Processing '{kind}' workflow for model {payload.get('model_name')}")

    # Update status to running (it was 'queued')
    TASKS[task_id]["status"] = "running"
    TASKS[task_id]["message"] = "Starting workflow..."

    try:
        final_output = None

        if kind == "train":
            generator = train_workflow(
                training_files=payload["training_files"],
                model_name=payload["model_name"],
                epochs=payload["epochs"],
                force_retrain=payload["force_retrain"],
            )
        elif kind == "convert":
            generator = convert_workflow(
                target_song=payload["target_song"],
                model_name=payload["model_name"],
                pitch_shift=payload["pitch_shift"],
            )
        else:  # "full" — quy trình cũ: train + convert trong 1 lần
            generator = automation_workflow(
                training_files=payload["training_files"],
                target_song=payload["target_song"],
                model_name=payload["model_name"],
                epochs=payload["epochs"],
                pitch_shift=payload["pitch_shift"],
                force_retrain=payload["force_retrain"],
            )

        for output_path, log_msg in generator:
            # Update logs in real-time
            TASKS[task_id]["logs"] = log_msg[-3000:]
            TASKS[task_id]["message"] = "Processing..."

            if output_path:
                final_output = output_path
                print(f"[Task {task_id}] Got output: {output_path}")

        if final_output and os.path.exists(final_output):
            TASKS[task_id]["status"] = "completed"
            TASKS[task_id]["result_path"] = final_output
            TASKS[task_id]["message"] = "Success"

            # Train xong -> lưu đăng ký model theo khách hàng vào DB
            if kind == "train" and payload.get("customer_id"):
                save_customer_model(
                    customer_id=payload["customer_id"],
                    model_name=payload["model_name"],
                    model_file=os.path.basename(final_output),
                    index_file=os.path.basename(_pick_index_file(payload["model_name"]) or "") or None,
                    epochs=payload.get("epochs"),
                )
        else:
            TASKS[task_id]["status"] = "failed"
            TASKS[task_id]["message"] = "Workflow completed but no output file generated."

    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        print(f"[Task {task_id}] Error: {err_trace}")
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["message"] = str(e)
        TASKS[task_id]["logs"] += f"\n\nERROR:\n{err_trace}"

    finally:
        # Dọn file input đã upload để không đầy đĩa (chỉ áp dụng task từ /run_upload).
        # File kết quả nằm ở audios/ nên xóa thư mục upload không ảnh hưởng.
        # Tắt cơ chế này bằng biến môi trường CLEANUP_UPLOADS=0.
        upload_dir = TASKS[task_id].get("upload_dir")
        if upload_dir and os.environ.get("CLEANUP_UPLOADS", "1") != "0":
            try:
                shutil.rmtree(upload_dir, ignore_errors=True)
                print(f"[Task {task_id}] Đã dọn thư mục upload: {upload_dir}")
            except Exception as ce:
                print(f"[Task {task_id}] Không dọn được {upload_dir}: {ce}")

# =====================================================================================
# JANITOR: tự động xóa file cũ trong audios/ để không đầy đĩa
# =====================================================================================
AUDIOS_ROOT = os.environ.get("AUDIOS_ROOT", "audios")   # khớp thư mục output của workflow
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "10"))   # xóa file cũ hơn N ngày
JANITOR_INTERVAL_SEC = int(os.environ.get("JANITOR_INTERVAL_SEC", str(6 * 3600)))  # quét mỗi 6h

def cleanup_old_files():
    """Xóa file/thư mục trong audios/ có thời gian sửa đổi cũ hơn RETENTION_DAYS."""
    if RETENTION_DAYS <= 0 or not os.path.isdir(AUDIOS_ROOT):
        return
    cutoff = time.time() - RETENTION_DAYS * 86400
    for entry in os.listdir(AUDIOS_ROOT):
        path = os.path.join(AUDIOS_ROOT, entry)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            print(f"[Janitor] Đã xóa (cũ hơn {RETENTION_DAYS} ngày): {path}")
        except Exception as e:
            print(f"[Janitor] Lỗi khi xóa {path}: {e}")

def janitor_loop():
    print(f"Janitor thread started (xóa file audios/ cũ hơn {RETENTION_DAYS} ngày, quét mỗi {JANITOR_INTERVAL_SEC//3600}h).")
    while True:
        try:
            cleanup_old_files()
        except Exception as e:
            print(f"[Janitor] Lỗi vòng lặp: {e}")
        time.sleep(JANITOR_INTERVAL_SEC)

@app.on_event("startup")
def startup_event():
    """Start the worker + janitor threads on app startup."""
    init_model_table()
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=janitor_loop, daemon=True).start()

@app.post("/run", dependencies=[Depends(verify_api_key)])
def run_automation(request: AutomationRequest):
    """Enqueues an automation task."""
    print(f"Received request: {request}")
    
    # Basic Validation
    for f in request.training_files:
        if not os.path.exists(f):
            raise HTTPException(status_code=400, detail=f"Training file not found: {f}")
    if not os.path.exists(request.target_song_path):
        raise HTTPException(status_code=400, detail=f"Target song not found: {request.target_song_path}")

    # Generate Task ID
    task_id = str(uuid.uuid4())

    # Initialize Task State
    TASKS[task_id] = {
        "status": "queued", # New initial status
        "message": "Waiting in queue...",
        "result_path": None,
        "logs": "",
        "created_at": time.time(),
        "kind": "full"
    }

    # Add to Queue
    TASK_QUEUE.put((task_id, "full", {
        "training_files": request.training_files,
        "target_song": request.target_song_path,
        "model_name": request.model_name,
        "epochs": request.epochs,
        "pitch_shift": request.pitch_shift,
        "force_retrain": request.force_retrain,
    }))

    q_size = TASK_QUEUE.qsize()
    print(f"Task {task_id} queued. Queue size: {q_size}")
    
    return {
        "status": "queued",
        "task_id": task_id,
        "message": f"Task queued. Position in line: {q_size}",
        "queue_size": q_size
    }

@app.get("/status/{task_id}", dependencies=[Depends(verify_api_key)])
def get_task_status(task_id: str):
    """Returns the current status and logs of a task."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return {
        "task_id": task_id,
        **TASKS[task_id]
    }

from fastapi.responses import FileResponse

@app.get("/download/{task_id}", dependencies=[Depends(verify_api_key)])
def download_result(task_id: str):
    """Downloads the final result file if the task is completed."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    task = TASKS[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task['status']}")
        
    result_path = task.get("result_path")
    
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found on server.")
        
    return FileResponse(
        path=result_path, 
        filename=os.path.basename(result_path),
        media_type="audio/mpeg"
    )

# Nơi lưu file người dùng upload trực tiếp (nằm trong volume ./audios đã mount)
UPLOAD_DIR = "/app/audios/uploads"

# API lấy vị trí media của bài hát theo id (trả JSON có trường "video" = URL .mp4)
STREAM_INFO_URL = os.environ.get("STREAM_INFO_URL", "http://172.16.10.12:3004/stream/stream_info")
# Tên file audio (có giọng gốc) trong cùng thư mục bài hát. video.mp4 chỉ có video,
# nên phải lấy audio.mp3 (bản đầy đủ có giọng để tách & thay).
TARGET_AUDIO_FILENAME = os.environ.get("TARGET_AUDIO_FILENAME", "audio.mp3")

def fetch_target_by_id(song_id: str, dest_dir: str) -> str:
    """Gọi stream_info theo id -> suy ra URL audio.mp3 -> tải về làm target."""
    # 1) Hỏi vị trí media (trả về URL video.mp4)
    try:
        r = requests.get(STREAM_INFO_URL, params={"id": song_id, "sourceType": "LOCAL"}, timeout=15)
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không gọi được stream_info cho id={song_id}: {e}")

    video_url = info.get("video")
    if not video_url:
        raise HTTPException(status_code=404, detail=f"stream_info không có 'video' cho id={song_id}: {info}")

    # 2) Suy ra URL audio: đổi tên file cuối (video.mp4 -> audio.mp3) trong cùng thư mục
    audio_url = video_url.rsplit("/", 1)[0] + "/" + TARGET_AUDIO_FILENAME

    # 3) Tải audio về (đã là mp3, đưa thẳng vào pipeline)
    ext = os.path.splitext(TARGET_AUDIO_FILENAME)[1] or ".mp3"
    dest = os.path.join(dest_dir, f"target_{song_id}{ext}")
    try:
        with requests.get(audio_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không tải được audio {audio_url}: {e}")

    if os.path.getsize(dest) < 10_000:
        raise HTTPException(
            status_code=502,
            detail=f"Audio tải về quá nhỏ — có thể bài chưa xuất bản (version v/0) hoặc URL sai: {audio_url}",
        )

    print(f"[TargetById] id={song_id} -> {audio_url} -> {dest}")
    return dest

@app.post("/run_upload", dependencies=[Depends(verify_api_key)])
async def run_upload(
    model_name: str = Form(...),
    epochs: int = Form(20),
    pitch_shift: int = Form(0),
    force_retrain: bool = Form(False),
    target_song_id: Optional[str] = Form(None),
    target_song: Optional[UploadFile] = File(None),
    training_files: List[UploadFile] = File(...),
):
    """Nhận file huấn luyện (upload) + bài hát đích, rồi đưa vào hàng đợi.

    Bài hát đích có 2 cách cung cấp (chọn 1):
      - target_song_id: id bài hát -> server tự lấy file qua stream_info API.
      - target_song: upload file trực tiếp.
    """
    # Mỗi request lưu vào một thư mục con riêng để tránh trùng tên
    session_dir = os.path.join(UPLOAD_DIR, str(uuid.uuid4())[:8])
    os.makedirs(session_dir, exist_ok=True)

    def _save(upload: UploadFile) -> str:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Có file upload thiếu tên file.")
        dest = os.path.join(session_dir, os.path.basename(upload.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(upload.file, out)
        return dest

    # Xác định bài hát đích: ưu tiên id, nếu không thì file upload
    if target_song_id:
        target_path = fetch_target_by_id(target_song_id, session_dir)
    elif target_song is not None and target_song.filename:
        target_path = _save(target_song)
    else:
        raise HTTPException(status_code=400, detail="Cần cung cấp target_song_id hoặc target_song (file).")

    train_paths = [_save(f) for f in training_files]

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "message": "Waiting in queue...",
        "result_path": None,
        "logs": "",
        "created_at": time.time(),
        "kind": "full",
        "upload_dir": session_dir  # để worker dọn sau khi xử lý xong
    }
    TASK_QUEUE.put((task_id, "full", {
        "training_files": train_paths,
        "target_song": target_path,
        "model_name": model_name,
        "epochs": epochs,
        "pitch_shift": pitch_shift,
        "force_retrain": force_retrain,
    }))
    q_size = TASK_QUEUE.qsize()
    print(f"[Upload] Task {task_id} queued from {session_dir}. Queue size: {q_size}")

    return {
        "status": "queued",
        "task_id": task_id,
        "message": f"Đã nhận {len(train_paths)} file huấn luyện + 1 bài hát. Vào hàng đợi (vị trí {q_size}).",
        "queue_size": q_size,
        "saved_dir": session_dir,
    }

# =====================================================================================
# API TÁCH RIÊNG: /train (train model theo khách hàng) + /convert (đổi giọng bằng model đã có)
# =====================================================================================

def _model_name_for(customer_id: str) -> str:
    """Sinh tên model từ id khách hàng (chỉ giữ chữ/số/gạch)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(customer_id)).strip("_")
    if not safe:
        raise HTTPException(status_code=400, detail="customer_id không hợp lệ.")
    return f"cus_{safe}"

@app.post("/train", dependencies=[Depends(verify_api_key)])
async def train_customer_model(
    customer_id: str = Form(...),
    epochs: int = Form(150),
    force_retrain: bool = Form(False),
    training_files: List[UploadFile] = File(...),
):
    """API 1: Train model giọng từ file ghi âm của khách hàng.

    - Model được lưu theo customer_id (và ghi vào DB khi train xong).
    - Nếu khách đã có model và force_retrain=false -> trả về luôn, không train lại.
    - force_retrain=true -> xóa model cũ, train dữ liệu mới thay thế.
    """
    model_name = _model_name_for(customer_id)

    existing = _pick_latest_model_file(model_name)
    if existing and not force_retrain:
        return {
            "status": "exists",
            "customer_id": customer_id,
            "model_name": model_name,
            "model_file": existing,
            "message": "Khách hàng đã có model. Gửi force_retrain=true nếu muốn train dữ liệu mới thay thế.",
        }

    session_dir = os.path.join(UPLOAD_DIR, str(uuid.uuid4())[:8])
    os.makedirs(session_dir, exist_ok=True)

    def _save(upload: UploadFile) -> str:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Có file upload thiếu tên file.")
        dest = os.path.join(session_dir, os.path.basename(upload.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(upload.file, out)
        return dest

    train_paths = [_save(f) for f in training_files]

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "message": "Waiting in queue...",
        "result_path": None,
        "logs": "",
        "created_at": time.time(),
        "kind": "train",
        "customer_id": customer_id,
        "upload_dir": session_dir,
    }
    TASK_QUEUE.put((task_id, "train", {
        "training_files": train_paths,
        "model_name": model_name,
        "epochs": epochs,
        "force_retrain": force_retrain,
        "customer_id": customer_id,
    }))
    q_size = TASK_QUEUE.qsize()
    print(f"[Train] Task {task_id} queued for customer {customer_id}. Queue size: {q_size}")

    return {
        "status": "queued",
        "task_id": task_id,
        "customer_id": customer_id,
        "model_name": model_name,
        "message": f"Đã nhận {len(train_paths)} file ghi âm. Bắt đầu train model (vị trí hàng đợi: {q_size}).",
        "queue_size": q_size,
    }

@app.get("/model/{customer_id}", dependencies=[Depends(verify_api_key)])
def customer_model_info(customer_id: str):
    """Kiểm tra khách hàng đã có model train sẵn chưa."""
    model_name = _model_name_for(customer_id)
    model_file = _pick_latest_model_file(model_name)
    index_file = _pick_index_file(model_name)
    return {
        "customer_id": customer_id,
        "trained": bool(model_file),
        "model_name": model_name,
        "model_file": model_file,
        "index_file": os.path.basename(index_file) if index_file else None,
        "db_record": get_customer_model(customer_id),
    }

@app.post("/convert", dependencies=[Depends(verify_api_key)])
async def convert_with_customer_model(
    customer_id: str = Form(...),
    pitch_shift: int = Form(0),
    target_song_id: Optional[str] = Form(None),
    target_song: Optional[UploadFile] = File(None),
):
    """API 2: Đổi giọng bài hát bằng model đã train sẵn của khách hàng.

    Bài hát đích: gửi target_song_id (lấy từ hệ thống) HOẶC upload file target_song.
    """
    model_name = _model_name_for(customer_id)

    if not _pick_latest_model_file(model_name):
        raise HTTPException(
            status_code=404,
            detail=f"Khách hàng {customer_id} chưa có model. Hãy gọi /train trước.",
        )

    session_dir = os.path.join(UPLOAD_DIR, str(uuid.uuid4())[:8])
    os.makedirs(session_dir, exist_ok=True)

    if target_song_id:
        target_path = fetch_target_by_id(target_song_id, session_dir)
    elif target_song is not None and target_song.filename:
        target_path = os.path.join(session_dir, os.path.basename(target_song.filename))
        with open(target_path, "wb") as out:
            shutil.copyfileobj(target_song.file, out)
    else:
        raise HTTPException(status_code=400, detail="Cần cung cấp target_song_id hoặc target_song (file).")

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "message": "Waiting in queue...",
        "result_path": None,
        "logs": "",
        "created_at": time.time(),
        "kind": "convert",
        "customer_id": customer_id,
        "upload_dir": session_dir,
    }
    TASK_QUEUE.put((task_id, "convert", {
        "target_song": target_path,
        "model_name": model_name,
        "pitch_shift": pitch_shift,
        "customer_id": customer_id,
    }))
    q_size = TASK_QUEUE.qsize()
    print(f"[Convert] Task {task_id} queued for customer {customer_id}. Queue size: {q_size}")

    return {
        "status": "queued",
        "task_id": task_id,
        "customer_id": customer_id,
        "model_name": model_name,
        "message": f"Bắt đầu đổi giọng bằng model của khách (vị trí hàng đợi: {q_size}).",
        "queue_size": q_size,
    }

# =====================================================================================
# DANH SÁCH BÀI HÁT + KIỂM TRA BÀI CÓ SẴN
# =====================================================================================

@app.get("/songs", dependencies=[Depends(verify_api_key)])
def list_songs(q: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Danh sách bài hát (từ bảng ktv_song, chỉ các bài có media file_path).

    - q: từ khóa tìm theo tên (có dấu hoặc không dấu đều được)
    - limit/offset: phân trang (limit tối đa 200)
    Lưu ý: bài thuộc build mới nhất có thể chưa đồng bộ lên media server —
    dùng GET /check_song/{id} để xác nhận trước khi convert.
    """
    if not _db_enabled():
        raise HTTPException(status_code=503, detail="DB chưa được cấu hình trên server.")

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    where = "deleted_flag IS NOT TRUE AND file_path IS NOT NULL AND file_path <> ''"
    params = []
    if q:
        where += " AND (name ILIKE %s OR normalized_name ILIKE %s)"
        params += [f"%{q}%", f"%{q}%"]

    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM ktv_song WHERE {where}", params)
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT id, name, duration, version FROM ktv_song WHERE {where} "
                f"ORDER BY id DESC LIMIT %s OFFSET %s", params + [limit, offset])
            songs = [
                {"id": r[0], "name": r[1], "duration": r[2], "version": r[3]}
                for r in cur.fetchall()
            ]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lỗi truy vấn DB: {e}")

    return {"total": total, "limit": limit, "offset": offset, "count": len(songs), "songs": songs}

@app.get("/check_song/{song_id}", dependencies=[Depends(verify_api_key)])
def check_song(song_id: str):
    """Kiểm tra 1 bài hát có file audio sẵn trên media server không (trước khi /convert)."""
    try:
        r = requests.get(STREAM_INFO_URL, params={"id": song_id, "sourceType": "LOCAL"}, timeout=10)
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không gọi được stream_info cho id={song_id}: {e}")

    video_url = info.get("video")
    if not video_url:
        return {"song_id": song_id, "available": False, "reason": "stream_info không có media"}

    if "/media/v/0/" in video_url:
        return {"song_id": song_id, "available": False,
                "reason": "Bài chưa xuất bản/đồng bộ lên media server (version v/0)"}

    audio_url = video_url.rsplit("/", 1)[0] + "/" + TARGET_AUDIO_FILENAME
    try:
        h = requests.head(audio_url, timeout=10)
        ok = (h.status_code == 200)
    except Exception as e:
        return {"song_id": song_id, "available": False, "reason": f"Lỗi kiểm tra media: {e}"}

    size = h.headers.get("Content-Length")
    return {
        "song_id": song_id,
        "available": ok,
        "size_mb": round(int(size) / 1048576, 1) if (ok and size) else None,
        "reason": None if ok else f"{TARGET_AUDIO_FILENAME} trả về HTTP {h.status_code}",
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "headless-async", "active_tasks": len(TASKS)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
