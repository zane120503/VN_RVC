import time
import os
from gradio_client import Client, handle_file

class RVCAutomationClient:
    def __init__(self, gradio_url="http://127.0.0.1:7860"):
        """
        Initialize the RVC Automation Client.
        
        Args:
            gradio_url (str): The URL of the running RVC Gradio instance.
        """
        print(f"Connecting to Gradio Client at {gradio_url}...")
        self.client = Client(gradio_url)
        print("Connected successfully.")

    def run_pipeline(
        self,
        training_files: list[str],
        target_song_path: str,
        model_name: str,
        epochs: int = 20,
        pitch_shift: int = 0,
        force_retrain: bool = False
    ):
        """
        Run the full automation pipeline via Gradio API.

        Args:
            training_files (list[str]): List of absolute paths to voice sample files.
            target_song_path (str): Absolute path to the target song (singer) to convert.
            model_name (str): Name of the model to train/use.
            epochs (int): Number of training epochs.
            pitch_shift (int): Pitch shift (semitones).
            force_retrain (bool): Whether to force retraining if model exists.

        Returns:
            dict: {"status": "success/error", "result_path": str, "logs": str}
        """
        # Validate inputs
        if not training_files or not isinstance(training_files, list):
            raise ValueError("training_files must be a non-empty list of file paths.")
        if not os.path.exists(target_song_path):
            raise ValueError(f"Target song not found: {target_song_path}")

        print(f"Starting pipeline for model: {model_name}")
        print(f"Training files: {len(training_files)}")
        print(f"Target song: {target_song_path}")

        # Prepare inputs for Gradio
        # Input 0: training_files (list of files)
        # Input 1: target_song (file)
        # Input 2: model_name (str)
        # Input 3: epochs (number)
        # Input 4: pitch_shift (number)
        # Input 5: force_retrain (bool)
        
        # Use handle_file for file uploads
        training_handles = [handle_file(f) for f in training_files]
        target_handle = handle_file(target_song_path)

        try:
            # Call the API
            # Returns: [output_audio, logs]
            result = self.client.predict(
                training_files=training_handles,
                target_song=target_handle,
                model_name=model_name,
                epochs=epochs,
                pitch_shift=pitch_shift,
                force_retrain=force_retrain,
                api_name="/run_automation"
            )
            
            output_audio, logs = result
            
            if output_audio:
                return {
                    "status": "success",
                    "result_path": output_audio,
                    "logs": logs
                }
            else:
                return {
                    "status": "error",
                    "message": "No output audio generated.",
                    "logs": logs
                }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
