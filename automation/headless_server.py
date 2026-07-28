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
            # Bản cũ chưa có cột object trên MinIO -> bổ sung
            cur.execute(f"ALTER TABLE {MODEL_TABLE} ADD COLUMN IF NOT EXISTS model_object VARCHAR")
            cur.execute(f"ALTER TABLE {MODEL_TABLE} ADD COLUMN IF NOT EXISTS index_object VARCHAR")
        print(f"DB model registry sẵn sàng (bảng {MODEL_TABLE}).")
    except Exception as e:
        print(f"⚠️  Không khởi tạo được bảng model: {e}")

def save_customer_model(customer_id, model_name, model_file, index_file, epochs,
                        model_object=None, index_object=None):
    """Upsert bản ghi model của khách hàng sau khi train xong."""
    if not _db_enabled():
        return
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {MODEL_TABLE} (customer_id, model_name, model_file, index_file, epochs,
                                           model_object, index_object, trained_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (customer_id) DO UPDATE SET
                    model_name   = EXCLUDED.model_name,
                    model_file   = EXCLUDED.model_file,
                    index_file   = EXCLUDED.index_file,
                    epochs       = EXCLUDED.epochs,
                    model_object = EXCLUDED.model_object,
                    index_object = EXCLUDED.index_object,
                    trained_at   = NOW(),
                    updated_at   = NOW()
                """, (customer_id, model_name, model_file, index_file, epochs,
                      model_object, index_object))
        print(f"[DB] Đã lưu model của khách {customer_id}: {model_file} (minio: {model_object})")
    except Exception as e:
        print(f"[DB] Lỗi lưu model khách {customer_id}: {e}")

def get_customer_model(customer_id):
    if not _db_enabled():
        return None
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT customer_id, model_name, model_file, index_file, epochs, trained_at, updated_at, "
                f"model_object, index_object "
                f"FROM {MODEL_TABLE} WHERE customer_id = %s", (customer_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "customer_id": row[0], "model_name": row[1], "model_file": row[2],
                "index_file": row[3], "epochs": row[4],
                "trained_at": str(row[5]), "updated_at": str(row[6]),
                "model_object": row[7], "index_object": row[8],
            }
    except Exception as e:
        print(f"[DB] Lỗi đọc model khách {customer_id}: {e}")
        return None

# =====================================================================================
# DB DANH SÁCH BÀI ĐÃ CONVERT (để khách nghe thử rồi chọn tải)
# =====================================================================================
CONVERT_TABLE = os.environ.get("CONVERT_TABLE", "rvc_converted_songs")

def init_convert_table():
    if not _db_enabled():
        return
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {CONVERT_TABLE} (
                    id          SERIAL PRIMARY KEY,
                    task_id     VARCHAR,
                    customer_id VARCHAR,
                    model_name  VARCHAR,
                    song_id     VARCHAR,
                    song_name   VARCHAR,
                    pitch_shift INTEGER,
                    result_path VARCHAR NOT NULL,
                    result_object VARCHAR,
                    file_size   BIGINT,
                    created_at  TIMESTAMP DEFAULT NOW()
                )""")
            # Bảng tạo từ bản cũ chưa có cột result_object -> bổ sung
            cur.execute(f"ALTER TABLE {CONVERT_TABLE} ADD COLUMN IF NOT EXISTS result_object VARCHAR")
        print(f"DB danh sách convert sẵn sàng (bảng {CONVERT_TABLE}).")
    except Exception as e:
        print(f"⚠️  Không khởi tạo được bảng convert: {e}")

def save_converted_song(task_id, customer_id, model_name, song_id, song_name, pitch_shift,
                        result_path, result_object=None):
    """Lưu 1 bản convert hoàn tất để khách nghe thử / tải lại sau."""
    if not _db_enabled():
        return
    try:
        size = os.path.getsize(result_path) if os.path.exists(result_path) else None
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {CONVERT_TABLE}
                    (task_id, customer_id, model_name, song_id, song_name, pitch_shift,
                     result_path, result_object, file_size)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (task_id, customer_id, model_name, song_id, song_name, pitch_shift,
                      result_path, result_object, size))
        print(f"[DB] Đã lưu bản convert của khách {customer_id}: {result_path} (minio: {result_object})")
    except Exception as e:
        print(f"[DB] Lỗi lưu bản convert khách {customer_id}: {e}")

def lookup_song_name(song_id):
    """Lấy tên bài hát từ bảng ktv_song (best-effort, lỗi thì trả None)."""
    if not (song_id and _db_enabled()):
        return None
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT name FROM ktv_song WHERE id = %s", (song_id,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
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

            # Train xong -> đẩy model lên MinIO (xóa bản cũ) + lưu đăng ký vào DB
            if kind == "train" and payload.get("customer_id"):
                model_object, index_object = upload_model_to_minio(payload["model_name"])
                save_customer_model(
                    customer_id=payload["customer_id"],
                    model_name=payload["model_name"],
                    model_file=os.path.basename(final_output),
                    index_file=os.path.basename(_pick_index_file(payload["model_name"]) or "") or None,
                    epochs=payload.get("epochs"),
                    model_object=model_object,
                    index_object=index_object,
                )

            # Convert xong -> đẩy kết quả lên MinIO + lưu vào danh sách để khách nghe thử rồi tải
            if kind == "convert":
                result_object = upload_result_to_minio(final_output, payload.get("customer_id"))
                save_converted_song(
                    task_id=task_id,
                    customer_id=payload.get("customer_id"),
                    model_name=payload.get("model_name"),
                    song_id=payload.get("song_id"),
                    song_name=payload.get("song_name"),
                    pitch_shift=payload.get("pitch_shift"),
                    result_path=final_output,
                    result_object=result_object,
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
        # Model trên đĩa chỉ là cache: làm mới hạn (MODEL_CACHE_DAYS tính từ LẦN DÙNG CUỐI)
        if kind == "convert" and payload.get("model_name"):
            _touch_model(payload["model_name"])

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
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "10"))   # file trung gian: xóa sau N ngày
# File KẾT QUẢ convert (audios/*_COVER_*.mp3) giữ lâu hơn để khách nghe lại/tải về
RESULT_RETENTION_DAYS = int(os.environ.get("RESULT_RETENTION_DAYS", "90"))
JANITOR_INTERVAL_SEC = int(os.environ.get("JANITOR_INTERVAL_SEC", str(6 * 3600)))  # quét mỗi 6h

def cleanup_old_files():
    """Dọn audios/ theo 2 mức: file kết quả (*_COVER_*) giữ RESULT_RETENTION_DAYS ngày,
    mọi thứ khác (file trung gian: tách beat/vocal, stub, target tải về...) giữ RETENTION_DAYS ngày."""
    if not os.path.isdir(AUDIOS_ROOT):
        return
    now = time.time()
    for entry in os.listdir(AUDIOS_ROOT):
        path = os.path.join(AUDIOS_ROOT, entry)
        is_result = os.path.isfile(path) and "_COVER_" in entry
        days = RESULT_RETENTION_DAYS if is_result else RETENTION_DAYS
        if days <= 0:
            continue
        try:
            if os.path.getmtime(path) >= now - days * 86400:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            print(f"[Janitor] Đã xóa ({'kết quả' if is_result else 'trung gian'}, cũ hơn {days} ngày): {path}")
        except Exception as e:
            print(f"[Janitor] Lỗi khi xóa {path}: {e}")

def janitor_loop():
    print(f"Janitor thread started (file kết quả giữ {RESULT_RETENTION_DAYS} ngày, "
          f"file trung gian {RETENTION_DAYS} ngày, quét mỗi {JANITOR_INTERVAL_SEC//3600}h).")
    while True:
        try:
            cleanup_old_files()
            cleanup_model_cache()
        except Exception as e:
            print(f"[Janitor] Lỗi vòng lặp: {e}")
        time.sleep(JANITOR_INTERVAL_SEC)

@app.on_event("startup")
def startup_event():
    """Start the worker + janitor threads on app startup."""
    init_model_table()
    init_record_table()
    init_convert_table()
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

from fastapi.responses import FileResponse, StreamingResponse

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
    record_ids: Optional[str] = Form(None),
    training_files: Optional[List[UploadFile]] = File(None),
):
    """API 1: Train model giọng từ file ghi âm của khách hàng.

    Nguồn file train — dùng 1 trong 2 (hoặc cả hai):
      - record_ids: danh sách record_id các bản thu khách đã upload (cách nhau dấu phẩy,
        vd "421,422,430") -> server tự lấy file từ MinIO, app KHÔNG phải tải về upload lại.
      - training_files: upload file trực tiếp.

    - Model được lưu theo customer_id (và ghi vào DB khi train xong).
    - Nếu khách đã có model và force_retrain=false -> trả về luôn, không train lại.
    - force_retrain=true -> xóa model cũ, train dữ liệu mới thay thế.
    """
    model_name = _model_name_for(customer_id)

    # Đã có model? — trên đĩa server hoặc bản lưu MinIO (server rebuild vẫn không train lại oan)
    existing = _pick_latest_model_file(model_name)
    if not existing:
        db_rec = get_customer_model(customer_id)
        if db_rec and db_rec.get("model_object"):
            existing = os.path.basename(db_rec["model_object"])
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

    train_paths = []
    n_uploaded = n_from_records = 0

    if training_files:
        for f in training_files:
            if f.filename:
                train_paths.append(_save(f))
                n_uploaded += 1

    # Lấy các bản thu đã upload sẵn trên MinIO theo record_id
    if record_ids:
        try:
            ids = [int(x) for x in re.split(r"[,\s]+", record_ids.strip()) if x]
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="record_ids không hợp lệ — gửi các số cách nhau dấu phẩy, vd: 421,422")
        client = _get_minio()
        for rid in ids:
            rec = _get_record(rid)  # 404 nếu không tồn tại
            dest = os.path.join(session_dir, f"record_{rid}_{os.path.basename(rec['audio_object'])}")
            try:
                client.fget_object(rec["bucket"], rec["audio_object"], dest)
            except Exception as e:
                raise HTTPException(status_code=502,
                                    detail=f"Không tải được bản thu record_id={rid} từ MinIO: {e}")
            train_paths.append(dest)
            n_from_records += 1

    if not train_paths:
        raise HTTPException(status_code=400,
                            detail="Cần gửi training_files (file) hoặc record_ids (id bản thu đã upload).")

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
        "message": f"Đã nhận {len(train_paths)} file ghi âm "
                   f"({n_uploaded} upload + {n_from_records} bản thu từ MinIO). "
                   f"Bắt đầu train model (vị trí hàng đợi: {q_size}).",
        "queue_size": q_size,
    }

@app.get("/model/{customer_id}", dependencies=[Depends(verify_api_key)])
def customer_model_info(customer_id: str):
    """Kiểm tra khách hàng đã có model train sẵn chưa (trên đĩa server hoặc MinIO)."""
    model_name = _model_name_for(customer_id)
    model_file = _pick_latest_model_file(model_name)
    index_file = _pick_index_file(model_name)
    db_record = get_customer_model(customer_id)
    on_minio = bool(db_record and db_record.get("model_object"))
    return {
        "customer_id": customer_id,
        "trained": bool(model_file) or on_minio,
        "storage": "local" if model_file else ("minio" if on_minio else None),
        "model_name": model_name,
        "model_file": model_file or (os.path.basename(db_record["model_object"]) if on_minio else None),
        "index_file": os.path.basename(index_file) if index_file else None,
        "db_record": db_record,
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

    # Model không có trên đĩa (vd server mới rebuild) -> tự khôi phục từ MinIO
    if not restore_model_from_minio(model_name):
        raise HTTPException(
            status_code=404,
            detail=f"Khách hàng {customer_id} chưa có model. Hãy gọi /train trước.",
        )

    session_dir = os.path.join(UPLOAD_DIR, str(uuid.uuid4())[:8])
    os.makedirs(session_dir, exist_ok=True)

    if target_song_id:
        target_path = fetch_target_by_id(target_song_id, session_dir)
        song_name = lookup_song_name(target_song_id)
    elif target_song is not None and target_song.filename:
        target_path = os.path.join(session_dir, os.path.basename(target_song.filename))
        with open(target_path, "wb") as out:
            shutil.copyfileobj(target_song.file, out)
        song_name = os.path.splitext(os.path.basename(target_song.filename))[0]
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
        "song_id": target_song_id,
        "song_name": song_name,
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

def _check_song_available(song):
    """Kiểm tra 1 bài có file audio thật trên media server không (dùng cho available=true).
    Trả về song (kèm size_mb) nếu convert được, None nếu không."""
    try:
        r = requests.get(STREAM_INFO_URL, params={"id": song["id"], "sourceType": "LOCAL"}, timeout=8)
        r.raise_for_status()
        video_url = r.json().get("video") or ""
        if not video_url or "/media/v/0/" in video_url:
            return None
        audio_url = video_url.rsplit("/", 1)[0] + "/" + TARGET_AUDIO_FILENAME
        h = requests.head(audio_url, timeout=8)
        if h.status_code != 200:
            return None
        size = h.headers.get("Content-Length")
        song["size_mb"] = round(int(size) / 1048576, 1) if size else None
        return song
    except Exception:
        return None

# Cache kết quả kiểm tra theo song_id để các lần gọi /songs?available=true sau nhanh
# (media server sync theo đợt nên kết quả ít thay đổi trong vài phút)
_SONG_AVAIL_CACHE = {}
SONG_CHECK_CACHE_SEC = int(os.environ.get("SONG_CHECK_CACHE_SEC", "600"))

def _check_song_available_cached(song):
    hit = _SONG_AVAIL_CACHE.get(song["id"])
    if hit and hit[0] > time.time():
        if hit[1] is None:
            return None
        return {**song, "size_mb": hit[1]}
    res = _check_song_available(song)
    _SONG_AVAIL_CACHE[song["id"]] = (time.time() + SONG_CHECK_CACHE_SEC,
                                     res["size_mb"] if res else None)
    return res

@app.get("/songs", dependencies=[Depends(verify_api_key)])
def list_songs(q: Optional[str] = None, limit: int = 50, offset: int = 0, available: bool = False):
    """Danh sách bài hát (từ bảng ktv_song, chỉ các bài có media file_path).

    - q: từ khóa tìm theo tên (có dấu hoặc không dấu đều được)
    - limit/offset: phân trang (limit tối đa 200)
    - available=true: CHỈ trả về bài convert được thật — lọc bài chưa xuất bản (version 0)
      và kiểm tra file audio tồn tại trên media server. Chậm hơn một chút (kiểm tra
      song song từng bài của trang hiện tại); count có thể nhỏ hơn limit.
    Không dùng available thì có thể xác nhận từng bài bằng GET /check_song/{id}.
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
    if available:
        # bài version 0 chắc chắn chưa đồng bộ lên media server -> loại ngay từ SQL
        where += " AND version IS NOT NULL AND version <> 0"

    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM ktv_song WHERE {where}", params)
            total = cur.fetchone()[0]

            if not available:
                cur.execute(
                    f"SELECT id, name, duration, version FROM ktv_song WHERE {where} "
                    f"ORDER BY id DESC LIMIT %s OFFSET %s", params + [limit, offset])
                songs = [
                    {"id": r[0], "name": r[1], "duration": r[2], "version": r[3]}
                    for r in cur.fetchall()
                ]
                return {"total": total, "limit": limit, "offset": offset, "count": len(songs),
                        "available_only": False, "songs": songs}

            # available=true: bài mới nhất thường CHƯA sync lên media server, nên phải
            # quét từng đợt từ mới -> cũ, kiểm tra thật (song song) và gom cho đủ `limit`.
            from concurrent.futures import ThreadPoolExecutor
            BATCH = 200
            MAX_SCAN = int(os.environ.get("SONGS_SCAN_LIMIT", "3000"))  # chặn trên để không quét cả DB
            songs, skipped, db_offset, scanned = [], 0, 0, 0
            with ThreadPoolExecutor(max_workers=20) as ex:
                while len(songs) < limit and db_offset < MAX_SCAN:
                    cur.execute(
                        f"SELECT id, name, duration, version FROM ktv_song WHERE {where} "
                        f"ORDER BY id DESC LIMIT %s OFFSET %s", params + [BATCH, db_offset])
                    rows = [{"id": r[0], "name": r[1], "duration": r[2], "version": r[3]}
                            for r in cur.fetchall()]
                    if not rows:
                        break
                    scanned += len(rows)
                    # ex.map giữ nguyên thứ tự -> phân trang ổn định
                    for s in ex.map(_check_song_available_cached, rows):
                        if s is None:
                            continue
                        if skipped < offset:   # offset tính trên danh sách BÀI CONVERT ĐƯỢC
                            skipped += 1
                            continue
                        if len(songs) < limit:
                            songs.append(s)
                    db_offset += BATCH
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lỗi truy vấn DB: {e}")

    return {"total": total, "limit": limit, "offset": offset, "count": len(songs),
            "available_only": True, "scanned": scanned, "songs": songs}

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

# =====================================================================================
# DANH SÁCH BÀI ĐÃ CONVERT — khách nghe thử rồi chọn tải
# =====================================================================================

def _flex_api_key(key: str = Security(api_key_header), api_key: Optional[str] = None):
    """Như verify_api_key nhưng nhận thêm query ?api_key= — để thẻ <audio src=...> phát được
    (trình duyệt không gửi được header tùy chỉnh trong thẻ audio/video)."""
    if not API_KEY:
        return
    if key == API_KEY or api_key == API_KEY:
        return
    raise HTTPException(status_code=401, detail="API key sai hoặc thiếu (header X-API-Key hoặc ?api_key=).")

_AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
                ".m4a": "audio/mp4", ".ogg": "audio/ogg"}

def _get_conversion(conversion_id: int):
    if not _db_enabled():
        raise HTTPException(status_code=503, detail="DB chưa được cấu hình trên server.")
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, customer_id, song_id, song_name, pitch_shift, result_path, "
                f"result_object, file_size, created_at "
                f"FROM {CONVERT_TABLE} WHERE id = %s", (conversion_id,))
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lỗi truy vấn DB: {e}")
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản convert.")
    return {
        "id": row[0], "customer_id": row[1], "song_id": row[2], "song_name": row[3],
        "pitch_shift": row[4], "result_path": row[5], "result_object": row[6],
        "file_size": row[7], "created_at": str(row[8]),
    }

def _serve_conversion(conv, as_download: bool):
    """Phục vụ file kết quả: ưu tiên file local (nhanh), hết local thì stream từ MinIO.
    Cả hai nơi đều hết (quá hạn lưu) -> 410."""
    path = conv["result_path"]
    ext = os.path.splitext(path or conv.get("result_object") or "")[1].lower() or ".mp3"
    media = _AUDIO_TYPES.get(ext, "audio/mpeg")
    filename = (conv["song_name"] or f"conversion_{conv['id']}") + ext

    # 1) File còn trên đĩa server
    if path and os.path.exists(path):
        if as_download:
            return FileResponse(path, media_type=media, filename=filename)
        return FileResponse(path, media_type=media)

    # 2) Bản lưu trên MinIO (bucket kết quả)
    obj_name = conv.get("result_object")
    if obj_name:
        try:
            client = _get_minio()
            obj = client.get_object(MINIO_RESULT_BUCKET, obj_name)
        except Exception as e:
            if "NoSuchKey" in str(e):
                raise HTTPException(
                    status_code=410,
                    detail=f"File đã hết hạn lưu trên MinIO ({RESULT_RETENTION_DAYS} ngày). Hãy gọi /convert lại.")
            raise HTTPException(status_code=502, detail=f"Không đọc được file từ MinIO: {e}")

        def _iter():
            try:
                for chunk in obj.stream(1 << 20):
                    yield chunk
            finally:
                obj.close()
                obj.release_conn()

        headers = {}
        if as_download:
            headers["Content-Disposition"] = "attachment; filename*=UTF-8''" + quote(filename)
        return StreamingResponse(_iter(), media_type=media, headers=headers)

    raise HTTPException(
        status_code=410,
        detail=f"File đã bị dọn (giữ tối đa {RESULT_RETENTION_DAYS} ngày). Hãy gọi /convert lại.")

@app.get("/conversions/{customer_id}", dependencies=[Depends(verify_api_key)])
def list_conversions(customer_id: str, limit: int = 50, offset: int = 0):
    """Danh sách các bản đã convert của khách (mới nhất trước) — nghe thử rồi chọn tải.

    `available=false` nghĩa là file kết quả đã bị dọn sau RESULT_RETENTION_DAYS ngày (convert lại nếu cần).
    """
    if not _db_enabled():
        raise HTTPException(status_code=503, detail="DB chưa được cấu hình trên server.")

    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))

    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {CONVERT_TABLE} WHERE customer_id = %s", (customer_id,))
            total = cur.fetchone()[0]
            cur.execute(
                f"SELECT id, song_id, song_name, pitch_shift, result_path, result_object, "
                f"file_size, created_at "
                f"FROM {CONVERT_TABLE} WHERE customer_id = %s ORDER BY id DESC LIMIT %s OFFSET %s",
                (customer_id, limit, offset))
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lỗi truy vấn DB: {e}")

    items = []
    for r in rows:
        on_disk = bool(r[4]) and os.path.exists(r[4])
        items.append({
            "conversion_id": r[0],
            "song_id": r[1],
            "song_name": r[2] or (os.path.basename(r[4]) if r[4] else None),
            "pitch_shift": r[3],
            "size_mb": round(r[6] / 1048576, 2) if r[6] else None,
            "created_at": str(r[7]),
            # còn nghe/tải được: file trên đĩa server hoặc bản lưu MinIO
            "available": on_disk or bool(r[5]),
            "storage": "local" if on_disk else ("minio" if r[5] else None),
            "stream_url": f"/conversions/{r[0]}/stream",
            "download_url": f"/conversions/{r[0]}/download",
        })
    return {"customer_id": customer_id, "total": total, "limit": limit, "offset": offset,
            "count": len(items), "conversions": items}

@app.get("/conversions/{conversion_id}/stream", dependencies=[Depends(_flex_api_key)])
def stream_conversion(conversion_id: int):
    """Nghe thử bản convert — phát trực tiếp (inline), dùng được cho <audio src="...?api_key=KEY">."""
    return _serve_conversion(_get_conversion(conversion_id), as_download=False)

@app.get("/conversions/{conversion_id}/download", dependencies=[Depends(_flex_api_key)])
def download_conversion(conversion_id: int):
    """Tải bản convert về (Content-Disposition: attachment, tên file = tên bài hát)."""
    return _serve_conversion(_get_conversion(conversion_id), as_download=True)

# =====================================================================================
# UPLOAD FILE GHI ÂM CỦA KHÁCH HÀNG LÊN NAS 25 (MinIO)
# Giống hệt API "api/files/upload/audio" của karaoke_system (cùng đường dẫn, cùng form
# field) để phòng karaoke trỏ recordingConfig.url vào server này là chạy được ngay.
# File lưu trên MinIO, thông tin bản ghi lưu vào Postgres (bảng RECORD_TABLE).
# Cấu hình qua biến môi trường: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
# MINIO_BUCKET, MINIO_SECURE (1 = https), RECORD_TABLE.
# =====================================================================================
try:
    from minio import Minio
except ImportError:
    Minio = None

from datetime import datetime
from urllib.parse import quote

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "172.16.20.12:9100")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "customer-records")
# Bucket riêng cho file KẾT QUẢ convert (khách nghe thử/tải) — tự xóa sau RESULT_RETENTION_DAYS
MINIO_RESULT_BUCKET = os.environ.get("MINIO_RESULT_BUCKET", "converted-songs")
# Bucket riêng cho MODEL đã train theo khách — giữ vĩnh viễn, train lại thì thay bản mới
MINIO_MODEL_BUCKET = os.environ.get("MINIO_MODEL_BUCKET", "customer-models")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "0") == "1"

_minio_client = None

def _get_minio():
    """Tạo (lazy) MinIO client + đảm bảo bucket tồn tại. Lỗi cấu hình/kết nối -> HTTP 503/502."""
    global _minio_client
    if Minio is None:
        raise HTTPException(status_code=503, detail="Thiếu thư viện 'minio' trên server (pip install minio).")
    if not (MINIO_ACCESS_KEY and MINIO_SECRET_KEY):
        raise HTTPException(status_code=503, detail="Chưa cấu hình MINIO_ACCESS_KEY/MINIO_SECRET_KEY trên server.")
    if _minio_client is None:
        client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                       secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
        try:
            if not client.bucket_exists(MINIO_BUCKET):
                client.make_bucket(MINIO_BUCKET)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Không kết nối được MinIO {MINIO_ENDPOINT}: {e}")
        _minio_client = client
    return _minio_client

def _safe_name(value: str) -> str:
    """Giữ chữ/số/gạch để làm tên thư mục/file trên MinIO."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value)).strip("_") or "unknown"

# Sau khi upload bản thu thành công -> gọi CMS báo metadata (để trống = tắt)
CMS_METADATA_URL = os.environ.get(
    "CMS_METADATA_URL", "https://cms-crm.icool.com.vn/api/files/recordings/metadata")

def notify_recording_metadata(payload):
    """Gửi thông tin bản thu vừa upload sang CMS. Best-effort: lỗi chỉ log +
    trả về trong response, KHÔNG làm upload fail (tránh client karaoke re-upload trùng)."""
    try:
        r = requests.post(CMS_METADATA_URL, json=payload, timeout=10)
        r.raise_for_status()
        print(f"[UploadAudio] Đã gửi metadata sang CMS (record_id={payload.get('record_id')}).")
        return True, None
    except Exception as e:
        print(f"[UploadAudio] Gửi metadata sang CMS thất bại: {e}")
        return False, str(e)

_result_bucket_ready = False

def _get_minio_result():
    """Client MinIO + đảm bảo bucket kết quả tồn tại, kèm lifecycle tự xóa sau RESULT_RETENTION_DAYS."""
    global _result_bucket_ready
    client = _get_minio()
    if not _result_bucket_ready:
        if not client.bucket_exists(MINIO_RESULT_BUCKET):
            client.make_bucket(MINIO_RESULT_BUCKET)
        try:
            from minio.lifecycleconfig import LifecycleConfig, Rule, Expiration
            from minio.commonconfig import Filter, ENABLED
            client.set_bucket_lifecycle(MINIO_RESULT_BUCKET, LifecycleConfig([
                Rule(ENABLED, rule_filter=Filter(prefix=""), rule_id="expire-results",
                     expiration=Expiration(days=RESULT_RETENTION_DAYS)),
            ]))
        except Exception as e:
            print(f"[MinIO] Không đặt được lifecycle cho {MINIO_RESULT_BUCKET} (object sẽ giữ vĩnh viễn): {e}")
        _result_bucket_ready = True
    return client

def upload_result_to_minio(path, customer_id):
    """Đẩy file kết quả convert lên MinIO, trả về object name (None nếu lỗi/chưa cấu hình).
    Chạy trong worker thread nên mọi lỗi chỉ log, không làm task fail."""
    try:
        client = _get_minio_result()
        object_name = "{}/{}/{}".format(
            _safe_name(customer_id or "unknown"),
            datetime.now().strftime("%Y-%m-%d"),
            _safe_name(os.path.basename(path)),
        )
        client.fput_object(MINIO_RESULT_BUCKET, object_name, path, content_type="audio/mpeg")
        print(f"[MinIO] Đã lưu kết quả convert: {MINIO_RESULT_BUCKET}/{object_name}")
        return object_name
    except Exception as e:
        print(f"[MinIO] Lỗi upload kết quả {path}: {e}")
        return None

# ------------------------- MODEL TRÊN MINIO (bucket customer-models) -------------------------
from main.app.variables import configs as _rvc_configs

def _weights_dir():
    return _rvc_configs.get("weights_path", os.path.join("assets", "weights"))

def _model_logs_dir(model_name):
    return os.path.join(_rvc_configs.get("logs_path", os.path.join("assets", "logs")), model_name)

_model_bucket_ready = False

def _get_minio_model():
    """Client MinIO + đảm bảo bucket model tồn tại (KHÔNG đặt lifecycle — model giữ vĩnh viễn)."""
    global _model_bucket_ready
    client = _get_minio()
    if not _model_bucket_ready:
        if not client.bucket_exists(MINIO_MODEL_BUCKET):
            client.make_bucket(MINIO_MODEL_BUCKET)
        _model_bucket_ready = True
    return client

def upload_model_to_minio(model_name):
    """Đẩy model vừa train (.pth + .index) lên MinIO dưới prefix {model_name}/.

    Xóa toàn bộ object cũ của model trước khi upload -> train lại là thay hẳn bản mới.
    Trả về (model_object, index_object); lỗi chỉ log (model vẫn còn trên đĩa server)."""
    try:
        client = _get_minio_model()

        # Xóa bản cũ trên MinIO
        for obj in client.list_objects(MINIO_MODEL_BUCKET, prefix=f"{model_name}/", recursive=True):
            client.remove_object(MINIO_MODEL_BUCKET, obj.object_name)

        model_object = index_object = None
        pth = _pick_latest_model_file(model_name)
        if pth:
            model_object = f"{model_name}/{pth}"
            client.fput_object(MINIO_MODEL_BUCKET, model_object,
                               os.path.join(_weights_dir(), pth))
        idx = _pick_index_file(model_name)
        if idx:
            index_object = f"{model_name}/{os.path.basename(idx)}"
            client.fput_object(MINIO_MODEL_BUCKET, index_object, idx)

        print(f"[MinIO] Đã lưu model {model_name}: {model_object} + {index_object}")
        return model_object, index_object
    except Exception as e:
        print(f"[MinIO] Lỗi upload model {model_name}: {e}")
        return None, None

# Bản model trên đĩa server chỉ là CACHE: quá N ngày không dùng thì janitor xóa
# (bản chính trên MinIO, cần lại sẽ tự tải về). Đặt 0 để giữ vĩnh viễn trên server.
MODEL_CACHE_DAYS = float(os.environ.get("MODEL_CACHE_DAYS", "1"))

def _model_on_minio(model_name):
    """Model có bản lưu trên MinIO không (điều kiện an toàn trước khi xóa bản local)."""
    try:
        client = _get_minio_model()
        for _ in client.list_objects(MINIO_MODEL_BUCKET, prefix=f"{model_name}/", recursive=True):
            return True
        return False
    except Exception:
        return False

def remove_model_from_server(model_name):
    """Xóa model khỏi đĩa server (.pth trong weights/ + thư mục logs/) — bản chính trên MinIO."""
    try:
        wd = _weights_dir()
        if os.path.isdir(wd):
            for f in os.listdir(wd):
                if f.startswith(model_name) and f.endswith(".pth"):
                    os.remove(os.path.join(wd, f))
        shutil.rmtree(_model_logs_dir(model_name), ignore_errors=True)
        print(f"[Model] Đã xóa cache model {model_name} trên đĩa server (bản chính trên MinIO).")
    except Exception as e:
        print(f"[Model] Lỗi xóa cache model {model_name}: {e}")

def _touch_model(model_name):
    """Làm mới mtime của model -> hạn cache tính từ LẦN DÙNG CUỐI."""
    try:
        wd = _weights_dir()
        for f in os.listdir(wd):
            if f.startswith(model_name) and f.endswith(".pth"):
                os.utime(os.path.join(wd, f), None)
    except Exception:
        pass

def cleanup_model_cache():
    """Janitor: xóa model cus_* trên đĩa server không dùng quá MODEL_CACHE_DAYS ngày.
    Chỉ xóa khi MinIO chắc chắn còn bản lưu."""
    wd = _weights_dir()
    if MODEL_CACHE_DAYS <= 0 or not os.path.isdir(wd):
        return
    cutoff = time.time() - MODEL_CACHE_DAYS * 86400
    for f in os.listdir(wd):
        if not (f.startswith("cus_") and f.endswith(".pth")):
            continue  # chỉ quản lý model tạo qua API (/train), không đụng model khác
        try:
            if os.path.getmtime(os.path.join(wd, f)) >= cutoff:
                continue
        except OSError:
            continue
        model_name = re.sub(r"_\d+e(_\d+s)?\.pth$", "", f)
        if _model_on_minio(model_name):
            remove_model_from_server(model_name)
        else:
            print(f"[Janitor] Giữ lại {f}: MinIO chưa có bản lưu (không xóa để tránh mất model).")

def restore_model_from_minio(model_name):
    """Tải model từ MinIO về đúng vị trí trên đĩa (weights/ + logs/) nếu local không có.

    Trả True khi model sẵn sàng dùng (đã có sẵn local hoặc khôi phục thành công)."""
    if _pick_latest_model_file(model_name):
        return True
    try:
        client = _get_minio_model()
        got_pth = False
        for obj in client.list_objects(MINIO_MODEL_BUCKET, prefix=f"{model_name}/", recursive=True):
            base = os.path.basename(obj.object_name)
            if base.endswith(".pth"):
                dest = os.path.join(_weights_dir(), base)
                got_pth = True
            elif base.endswith(".index"):
                dest = os.path.join(_model_logs_dir(model_name), base)
            else:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            client.fget_object(MINIO_MODEL_BUCKET, obj.object_name, dest)
            print(f"[MinIO] Khôi phục model: {obj.object_name} -> {dest}")
        return got_pth
    except Exception as e:
        print(f"[MinIO] Lỗi khôi phục model {model_name}: {e}")
        return False

# Bảng lưu thông tin bản ghi âm đã upload (Postgres, cùng DB với đăng ký model)
RECORD_TABLE = os.environ.get("RECORD_TABLE", "rvc_recorded_files")

def init_record_table():
    if not _db_enabled():
        return
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {RECORD_TABLE} (
                    id           SERIAL PRIMARY KEY,
                    media_id     VARCHAR,
                    name         VARCHAR,
                    cluster_id   VARCHAR,
                    room_code    VARCHAR,
                    singer_name  VARCHAR,
                    is_4k        BOOLEAN DEFAULT FALSE,
                    created_time VARCHAR,
                    bucket       VARCHAR,
                    audio_object VARCHAR NOT NULL,
                    image_object VARCHAR,
                    audio_size   BIGINT,
                    uploaded_at  TIMESTAMP DEFAULT NOW()
                )""")
        print(f"DB bảng ghi âm sẵn sàng (bảng {RECORD_TABLE}).")
    except Exception as e:
        print(f"⚠️  Không khởi tạo được bảng ghi âm: {e}")

def save_recorded_file(media_id, name, cluster_id, room_code, singer_name,
                       is_4k, created_time, audio_object, image_object, audio_size):
    """Ghi 1 dòng vào bảng RECORD_TABLE, trả về id của bản ghi."""
    with _db_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {RECORD_TABLE}
                (media_id, name, cluster_id, room_code, singer_name, is_4k,
                 created_time, bucket, audio_object, image_object, audio_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """, (media_id, name, cluster_id, room_code, singer_name, is_4k,
                  created_time, MINIO_BUCKET, audio_object, image_object, audio_size))
        return cur.fetchone()[0]

def _put_minio(client, upload: UploadFile, object_name: str, default_type: str, metadata: dict):
    """Stream 1 file upload lên MinIO, trả về kích thước file."""
    upload.file.seek(0, os.SEEK_END)
    size = upload.file.tell()
    upload.file.seek(0)
    if size <= 0:
        raise HTTPException(status_code=400, detail=f"File '{upload.filename}' rỗng.")
    try:
        client.put_object(
            MINIO_BUCKET, object_name, upload.file, size,
            content_type=upload.content_type or default_type,
            metadata=metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upload lên MinIO thất bại: {e}")
    return size

# Không yêu cầu X-API-Key: giữ nguyên contract của karaoke_system (client Retrofit
# không gửi header này) — chỉ cần đổi recordingConfig.url là dùng được, không sửa client.
@app.post("/api/files/upload/audio")
async def upload_audio(
    name: Optional[str] = Form(None),
    id: Optional[str] = Form(None),
    cluster_id: Optional[str] = Form(None),
    room_code: Optional[str] = Form(None),
    created_time: Optional[str] = Form(None),
    is_4k: Optional[str] = Form(None),
    singer_name: Optional[str] = Form(None),
    audio: UploadFile = File(...),
    image: Optional[UploadFile] = File(None),
):
    """Upload file ghi âm của khách hàng — giống API của karaoke_system.

    - Form field khớp với RecordingClient.kt: audio (file), image (file),
      name, id, cluster_id, room_code, created_time, is_4k, singer_name.
    - File audio + ảnh lưu lên MinIO (NAS 25), thông tin bản ghi lưu vào Postgres.
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="File audio thiếu tên file.")

    client = _get_minio()

    now = datetime.now()
    if not created_time:
        created_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Tổ chức: {cluster}/{phòng}/{ngày}/{giờ}_{uuid}_{id bài}_{tên file}
    base_dir = "{}/{}/{}".format(
        _safe_name(cluster_id or "unknown"),
        _safe_name(room_code or "unknown"),
        now.strftime("%Y-%m-%d"),
    )
    base_name = "{}_{}_{}".format(now.strftime("%H%M%S"), str(uuid.uuid4())[:8], _safe_name(id or "noid"))

    audio_ext = os.path.splitext(audio.filename)[1] or ".wav"
    audio_object = f"{base_dir}/{base_name}{audio_ext}"

    # Metadata header chỉ nhận ASCII -> quote các giá trị (tên bài tiếng Việt...)
    metadata = {
        f"x-amz-meta-{k}": quote(str(v))
        for k, v in {
            "media_id": id,
            "name": name,
            "cluster_id": cluster_id,
            "room_code": room_code,
            "singer_name": singer_name,
            "is_4k": is_4k,
            "created_time": created_time,
        }.items() if v
    }

    audio_size = _put_minio(client, audio, audio_object, "audio/mpeg", metadata)

    image_object = None
    if image is not None and image.filename:
        image_ext = os.path.splitext(image.filename)[1] or ".jpg"
        image_object = f"{base_dir}/{base_name}{image_ext}"
        _put_minio(client, image, image_object, "image/jpeg", metadata)

    # Lưu thông tin bản ghi vào DB (file đã nằm trên MinIO)
    record_id = None
    if _db_enabled():
        try:
            record_id = save_recorded_file(
                media_id=id, name=name, cluster_id=cluster_id, room_code=room_code,
                singer_name=singer_name, is_4k=str(is_4k).lower() == "true",
                created_time=created_time, audio_object=audio_object,
                image_object=image_object, audio_size=audio_size,
            )
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"File đã lên MinIO ({audio_object}) nhưng lưu DB thất bại: {e}")
    else:
        print("⚠️  [UploadAudio] DB chưa cấu hình -> chỉ lưu file lên MinIO, không có bản ghi DB.")

    # Báo metadata bản thu sang CMS (best-effort)
    metadata_sent, metadata_error = False, "CMS_METADATA_URL chưa cấu hình"
    if CMS_METADATA_URL:
        metadata_sent, metadata_error = notify_recording_metadata({
            "id": id,
            "name": name,
            "cluster_id": cluster_id,
            "room_code": room_code,
            "created_time": created_time,
            "is_4k": str(is_4k).lower() == "true",
            "singer_name": singer_name,
            "record_id": record_id,
        })

    print(f"[UploadAudio] {name or audio.filename} -> {MINIO_BUCKET}/{audio_object} "
          f"({round(audio_size/1048576, 2)} MB), db_id={record_id}, cms={metadata_sent}")
    return {
        "status": "uploaded",
        "record_id": record_id,
        "bucket": MINIO_BUCKET,
        "audio_object": audio_object,
        "image_object": image_object,
        "size_mb": round(audio_size / 1048576, 2),
        "created_time": created_time,
        "metadata_sent": metadata_sent,
        "metadata_error": metadata_error,
    }

# =====================================================================================
# NGHE LẠI / TẢI BẢN GHI ÂM ĐÃ UPLOAD (stream từ bucket customer-records)
# =====================================================================================

def _get_record(record_id: int):
    if not _db_enabled():
        raise HTTPException(status_code=503, detail="DB chưa được cấu hình trên server.")
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, name, singer_name, bucket, audio_object, image_object "
                f"FROM {RECORD_TABLE} WHERE id = %s",
                (record_id,))
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Lỗi truy vấn DB: {e}")
    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi âm.")
    return {"id": row[0], "name": row[1], "singer_name": row[2],
            "bucket": row[3] or MINIO_BUCKET, "audio_object": row[4], "image_object": row[5]}

def _stream_record(rec, as_download):
    obj_name = rec["audio_object"]
    ext = os.path.splitext(obj_name or "")[1].lower() or ".mp3"
    media = _AUDIO_TYPES.get(ext, "audio/mpeg")
    filename = (rec["name"] or f"record_{rec['id']}") + ext
    try:
        client = _get_minio()
        obj = client.get_object(rec["bucket"], obj_name)
    except Exception as e:
        if "NoSuchKey" in str(e):
            raise HTTPException(status_code=410, detail="File không còn trên MinIO.")
        raise HTTPException(status_code=502, detail=f"Không đọc được file từ MinIO: {e}")

    def _iter():
        try:
            for chunk in obj.stream(1 << 20):
                yield chunk
        finally:
            obj.close()
            obj.release_conn()

    headers = {}
    if as_download:
        headers["Content-Disposition"] = "attachment; filename*=UTF-8''" + quote(filename)
    return StreamingResponse(_iter(), media_type=media, headers=headers)

@app.get("/records/{record_id}/stream", dependencies=[Depends(_flex_api_key)])
def stream_record(record_id: int):
    """Nghe bản ghi âm đã upload — nhận key qua header hoặc ?api_key= (gắn được vào <audio src>)."""
    return _stream_record(_get_record(record_id), as_download=False)

@app.get("/records/{record_id}/download", dependencies=[Depends(_flex_api_key)])
def download_record(record_id: int):
    """Tải bản ghi âm đã upload về (attachment)."""
    return _stream_record(_get_record(record_id), as_download=True)

_IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp"}

# Tên file thumbnail trong thư mục bài hát trên media server (cùng chỗ với audio.mp3)
SONG_IMAGE_FILENAME = os.environ.get("SONG_IMAGE_FILENAME", "image.jpg")

@app.get("/songs/{song_id}/image", dependencies=[Depends(_flex_api_key)])
def song_image(song_id: str):
    """Thumbnail của bài hát (file image.jpg trên media server) — gắn được vào <img src="...?api_key=KEY">.

    Dùng id số của bảng ktv_song (id trong kết quả GET /songs)."""
    try:
        r = requests.get(STREAM_INFO_URL, params={"id": song_id, "sourceType": "LOCAL"}, timeout=10)
        r.raise_for_status()
        video_url = r.json().get("video") or ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không gọi được stream_info cho id={song_id}: {e}")
    if not video_url:
        raise HTTPException(status_code=404, detail="Bài hát không có media.")

    image_url = video_url.rsplit("/", 1)[0] + "/" + SONG_IMAGE_FILENAME
    try:
        resp = requests.get(image_url, stream=True, timeout=15)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không tải được thumbnail: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Bài này không có thumbnail (HTTP {resp.status_code}).")
    return StreamingResponse(resp.iter_content(1 << 16),
                             media_type=resp.headers.get("Content-Type", "image/jpeg"))

@app.get("/records/{record_id}/image", dependencies=[Depends(_flex_api_key)])
def record_image(record_id: int):
    """Ảnh bìa của bản ghi âm — gắn được vào <img src="...?api_key=KEY">."""
    rec = _get_record(record_id)
    obj_name = rec.get("image_object")
    if not obj_name:
        raise HTTPException(status_code=404, detail="Bản ghi này không có ảnh bìa.")
    media = _IMAGE_TYPES.get(os.path.splitext(obj_name)[1].lower(), "image/jpeg")
    try:
        client = _get_minio()
        obj = client.get_object(rec["bucket"], obj_name)
    except Exception as e:
        if "NoSuchKey" in str(e):
            raise HTTPException(status_code=410, detail="Ảnh không còn trên MinIO.")
        raise HTTPException(status_code=502, detail=f"Không đọc được ảnh từ MinIO: {e}")

    def _iter():
        try:
            for chunk in obj.stream(1 << 20):
                yield chunk
        finally:
            obj.close()
            obj.release_conn()

    return StreamingResponse(_iter(), media_type=media)

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "headless-async", "active_tasks": len(TASKS)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
