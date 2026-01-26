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

app = FastAPI(title="RVC Headless Automation API")

class AutomationRequest(BaseModel):
    training_files: List[str]
    target_song_path: str
    model_name: str
    epochs: int = 20
    pitch_shift: int = 0
    force_retrain: bool = False

@app.post("/run")
def run_automation(request: AutomationRequest):
    print(f"Received headless request: {request}")
    
    # Validation
    for f in request.training_files:
        if not os.path.exists(f):
            raise HTTPException(status_code=400, detail=f"Training file not found: {f}")
    if not os.path.exists(request.target_song_path):
        raise HTTPException(status_code=400, detail=f"Target song not found: {request.target_song_path}")

    # Run the generator workflow
    # We iterate through the generator to execute all steps
    # The last yield result (usually) contains the return value or we check logs
    
    try:
        final_output = None
        logs = []
        
        generator = automation_workflow(
            training_files=request.training_files,
            target_song=request.target_song_path,
            model_name=request.model_name,
            epochs=request.epochs,
            pitch_shift=request.pitch_shift,
            force_retrain=request.force_retrain
        )
        
        for output_path, log_msg in generator:
            # log_msg is the full updated log, so we can just replace
            # But the workflow yields (None, full_log) or (result_path, full_log)
            if output_path:
                final_output = output_path
                print(f"Workflow produced output: {final_output}")
            
            # Print latest log line (approximated, since log_msg is full)
            # Actually let's just keep silent or print progress dots
            pass
            
        logs_str = log_msg if 'log_msg' in locals() else "No logs"
        
        if final_output and os.path.exists(final_output):
            return {
                "status": "success",
                "result_path": final_output,
                "logs": logs_str[-500:] # Return last 500 chars of logs
            }
        else:
             return {
                "status": "error",
                "message": "Workflow completed but no output file found.",
                "logs": logs_str[-1000:]
             }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(err)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "headless"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
