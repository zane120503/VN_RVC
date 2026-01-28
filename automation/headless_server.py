import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

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

@app.on_event("startup")
def startup_event():
    """Start the worker thread on app startup."""
    thread = threading.Thread(target=worker_loop, daemon=True)
    thread.start()

@app.post("/run")
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

@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    """Returns the current status and logs of a task."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return {
        "task_id": task_id,
        **TASKS[task_id]
    }

from fastapi.responses import FileResponse

@app.get("/download/{task_id}")
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

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "headless-async", "active_tasks": len(TASKS)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
