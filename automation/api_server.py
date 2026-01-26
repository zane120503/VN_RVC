from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import sys

# Add parent directory to path to import client wrapper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.client_wrapper import RVCAutomationClient

app = FastAPI(title="RVC Automation API")

# Global client instance
client: Optional[RVCAutomationClient] = None
GRADIO_URL = os.environ.get("GRADIO_URL", "http://127.0.0.1:7860")

class AutomationRequest(BaseModel):
    training_files: List[str]
    target_song_path: str
    model_name: str
    epochs: int = 150
    pitch_shift: int = 0
    force_retrain: bool = False

@app.on_event("startup")
def startup_event():
    global client
    try:
        client = RVCAutomationClient(gradio_url=GRADIO_URL)
    except Exception as e:
        print(f"Failed to connect to Gradio on startup: {e}")
        print("Will attempt to reconnect on first request.")

@app.post("/run")
def run_automation(request: AutomationRequest):
    global client
    if client is None:
        try:
            client = RVCAutomationClient(gradio_url=GRADIO_URL)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Could not connect to Gradio backend: {str(e)}")

    print(f"Received request: {request}")
    
    # Run the pipeline
    try:
        result = client.run_pipeline(
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
    return {"status": "ok", "gradio_connected": client is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
