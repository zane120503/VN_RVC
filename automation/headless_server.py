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

def run_automation_thread(task_id: str, request: AutomationRequest):
    """Background worker to run the heavy automation workflow."""
    print(f"[Task {task_id}] Started processing for model {request.model_name}")
    TASKS[task_id]["status"] = "running"
    TASKS[task_id]["message"] = "Starting workflow..."
    
    try:
        # Import inside thread to capture logs better if we redirect stdout suitable for this thread
        # (For now we just run it)
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
            TASKS[task_id]["logs"] = log_msg[-2000:] # Keep last 2000 chars to save memory
            TASKS[task_id]["message"] = "Processing..." # You could parse log_msg for better status
            
            if output_path:
                final_output = output_path
                print(f"[Task {task_id}] Got partial/final output: {output_path}")

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

@app.post("/run")
def run_automation(request: AutomationRequest):
    """Starts the automation workflow in the background and returns a task_id."""
    print(f"Received headless request: {request}")
    
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
        "status": "started",
        "message": "Initializing...",
        "result_path": None,
        "logs": ""
    }
    
    # Start Background Thread
    thread = threading.Thread(target=run_automation_thread, args=(task_id, request))
    thread.daemon = True # Daemon threads exit when the main program exits
    thread.start()
    
    return {
        "status": "started",
        "task_id": task_id,
        "message": "Automation started in background. Check status at /status/{task_id}"
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

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "headless-async", "active_tasks": len(TASKS)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
