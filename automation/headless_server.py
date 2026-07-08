import os
import sys
import shutil
import subprocess
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional

# Add root directory to path
sys.path.append(os.getcwd())

# Import the workflow function directly
# This avoids loading the entire Gradio UI stack
from main.app.tabs.automation.child.automation import automation_workflow

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

# Global Queue for sequential processing
TASK_QUEUE = queue.Queue()

def worker_loop():
    """Consumes tasks from the queue and processes them sequentially."""
    print("Worker thread started, waiting for tasks...")
    while True:
        try:
            # Block until a task is available
            task_id, request = TASK_QUEUE.get()
            print(f"[Worker] Picked up task {task_id}")
            
            run_automation_task(task_id, request)
            
            TASK_QUEUE.task_done()
            print(f"[Worker] Finished task {task_id}, remaining in queue: {TASK_QUEUE.qsize()}")
            
        except Exception as e:
            print(f"[Worker] Critical Error in worker loop: {e}")
            import traceback
            traceback.print_exc()

def run_automation_task(task_id: str, request: AutomationRequest):
    """Actual logic to run the automation (formerly run_automation_thread)."""
    print(f"[Task {task_id}] Processing workflow for model {request.model_name}")
    
    # Update status to running (it was 'queued')
    TASKS[task_id]["status"] = "running"
    TASKS[task_id]["message"] = "Starting workflow..."
    
    try:
        final_output = None
        logs_accumulated = []

        generator = automation_workflow(
            training_files=request.training_files,
            target_song=request.target_song_path,
            model_name=request.model_name,
            epochs=request.epochs,
            pitch_shift=request.pitch_shift,
            force_retrain=request.force_retrain
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
        "created_at": time.time()
    }
    
    # Add to Queue
    TASK_QUEUE.put((task_id, request))
    
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

def fetch_target_by_id(song_id: str, dest_dir: str) -> str:
    """Gọi stream_info theo id -> tải media -> tách audio .wav để đưa vào pipeline."""
    # 1) Hỏi vị trí media
    try:
        r = requests.get(STREAM_INFO_URL, params={"id": song_id, "sourceType": "LOCAL"}, timeout=15)
        r.raise_for_status()
        info = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không gọi được stream_info cho id={song_id}: {e}")

    media_url = info.get("video") or info.get("audio")
    if not media_url:
        raise HTTPException(status_code=404, detail=f"stream_info không có media cho id={song_id}: {info}")

    # 2) Tải file media về
    raw_path = os.path.join(dest_dir, f"target_{song_id}_raw.mp4")
    try:
        with requests.get(media_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(raw_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không tải được media {media_url}: {e}")

    if os.path.getsize(raw_path) < 10_000:
        raise HTTPException(status_code=502, detail=f"File tải về quá nhỏ (URL lỗi?): {media_url}")

    # 3) Tách audio sang wav bằng ffmpeg (mp4 -> wav 44.1kHz)
    audio_path = os.path.join(dest_dir, f"target_{song_id}.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", audio_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg tách audio lỗi: {e.stderr.decode('utf-8', 'ignore')[-500:]}")

    os.remove(raw_path)  # bỏ file mp4, chỉ giữ wav
    print(f"[TargetById] id={song_id} -> {media_url} -> {audio_path}")
    return audio_path

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

    request = AutomationRequest(
        training_files=train_paths,
        target_song_path=target_path,
        model_name=model_name,
        epochs=epochs,
        pitch_shift=pitch_shift,
        force_retrain=force_retrain,
    )

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "message": "Waiting in queue...",
        "result_path": None,
        "logs": "",
        "created_at": time.time(),
        "upload_dir": session_dir  # để worker dọn sau khi xử lý xong
    }
    TASK_QUEUE.put((task_id, request))
    q_size = TASK_QUEUE.qsize()
    print(f"[Upload] Task {task_id} queued from {session_dir}. Queue size: {q_size}")

    return {
        "status": "queued",
        "task_id": task_id,
        "message": f"Đã nhận {len(train_paths)} file huấn luyện + 1 bài hát. Vào hàng đợi (vị trí {q_size}).",
        "queue_size": q_size,
        "saved_dir": session_dir,
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "headless-async", "active_tasks": len(TASKS)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
