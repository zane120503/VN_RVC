import os
import sys
import uvicorn
import gradio as gr
from fastapi import FastAPI, HTTPException

# Add root directory to path
sys.path.append(os.getcwd())

# Import the RVC App (Gradio Blocks)
# Note: app.py code executes on import, so we need to be careful.
# It uses sys.argv to check for flags like --share, but we can ignore those for now or set them.
sys.argv.append("--client") # Force client mode triggers for compatibility check
from main.app.app import app as rvc_gradio_app

# Import our API routes (we'll define them here or import them)
# Re-implementing api_server logic here to avoid circular imports or complexity
from automation.client_wrapper import RVCAutomationClient
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="RVC Automation API + GUI")

# Mount Gradio App (This makes the GUI available at /, and API at /run)
# gradio_app = gr.routes.App.create_app(rvc_gradio_app)
# app.mount("/", gradio_app) 
# BETTER WAY:
app = gr.mount_gradio_app(app, rvc_gradio_app, path="/")

# Global client
client_wrapper: Optional[RVCAutomationClient] = None

class AutomationRequest(BaseModel):
    training_files: List[str]
    target_song_path: str
    model_name: str
    epochs: int = 150
    pitch_shift: int = 0
    force_retrain: bool = False

@app.on_event("startup")
def startup_event():
    global client_wrapper
    # When running in same process, Gradio is at 0.0.0.0:8000 (via FastAPI)
    # Gradio Client needs to connect to this.
    # Note: Gradio Client might need the server to be fully up before connecting?
    # Or we can lazy load.
    pass

@app.post("/run")
def run_automation(request: AutomationRequest):
    global client_wrapper
    
    # Lazy init client to ensure server is running
    if client_wrapper is None:
        try:
             # Connect to SELF (localhost:8000)
             client_wrapper = RVCAutomationClient(gradio_url="http://localhost:8000")
        except Exception as e:
             raise HTTPException(status_code=503, detail=f"Could not connect to local Gradio instance: {e}")

    print(f"Received request: {request}")
    try:
        result = client_wrapper.run_pipeline(
            training_files=request.training_files,
            target_song_path=request.target_song_path,
            model_name=request.model_name,
            epochs=request.epochs,
            pitch_shift=request.pitch_shift,
            force_retrain=request.force_retrain
        )
        if result["status"] == "error":
             raise HTTPException(status_code=500, detail=result.get("message", "Unknown error") + "\nLogs:\n" + result.get("logs", ""))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "bundled"}

if __name__ == "__main__":
    # Remove --open from args passed to uvicorn if any
    uvicorn.run(app, host="0.0.0.0", port=8000)
