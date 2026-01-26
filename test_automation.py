import requests
import os

# Configuration
API_URL = "http://localhost:8000/run"

# Example Paths - CHANGE THESE TO YOUR REAL FILES
# Note: Use absolute paths or correct relative paths
CURRENT_DIR = os.getcwd()

# Example: Using the included sample audio if available, or create dummy files for testing
# You should point these to valid files
TRAINING_FILES = [
    # r"D:\path\to\your\voice_sample1.wav", 
    # r"D:\path\to\your\voice_sample2.wav"
]

TARGET_SONG = r"D:\path\to\target_song.mp3"
MODEL_NAME = "test_automation_model"

def main():
    print(f"Calling Automation API at {API_URL}...")
    
    if not TRAINING_FILES or not os.path.exists(TARGET_SONG):
        print("Please edit this script to set valid TRAINING_FILES and TARGET_SONG paths first!")
        return

    payload = {
        "training_files": TRAINING_FILES,
        "target_song_path": TARGET_SONG,
        "model_name": MODEL_NAME,
        "epochs": 20, # Low epochs for testing
        "pitch_shift": 0,
        "force_retrain": False
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=3600) # Long timeout for full process
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(f"Result Audio Path: {result['result_path']}")
            print("-" * 20)
            print("Logs:")
            print(result['logs'])
        else:
            print(f"FAILED (Status {response.status_code})")
            try:
                print(response.json())
            except:
                print(response.text)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Is 'run_automation_api.bat' running?")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
