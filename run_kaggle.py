import os
import sys
import json
import subprocess
import time
from pathlib import Path

# Force UTF-8 encoding for all subprocess calls to prevent Windows charmap errors
os.environ["PYTHONIOENCODING"] = "utf-8"

# Config
KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME" # You need to replace this if the script asks!
NOTEBOOK_DIR = Path("notebooks")
NOTEBOOK_FILE = "x-phase-3.ipynb"
KERNEL_SLUG = "neural-memory-phase3" # The URL slug for your notebook

def get_kaggle_username():
    """Try to get username from kaggle.json"""
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        with open(kaggle_json_path, "r") as f:
            creds = json.load(f)
            return creds.get("username")
    return KAGGLE_USERNAME

def main():
    username = get_kaggle_username()
    if username == "YOUR_KAGGLE_USERNAME":
        print("Please edit run_kaggle.py to set your Kaggle username, or ensure ~/.kaggle/kaggle.json exists.")
        sys.exit(1)

    kernel_id = f"{username}/{KERNEL_SLUG}"
    
    # 1. Regenerate the notebook to make sure it's up to date
    print("Regenerating notebook...")
    subprocess.run([sys.executable, "generate_notebook.py"], check=True)
    
    # 2. Create kernel-metadata.json
    metadata = {
      "id": kernel_id,
      "title": "neural-memory-phase3",
      "code_file": NOTEBOOK_FILE,
      "language": "python",
      "kernel_type": "notebook",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": [],
      "competition_sources": [],
      "kernel_sources": [],
      "model_sources": []
    }
    
    with open(NOTEBOOK_DIR / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Created metadata for kernel: {kernel_id}")
    
    # 3. Push to Kaggle
    print("\nPushing notebook to Kaggle...")
    try:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(NOTEBOOK_DIR)], check=True)
    except subprocess.CalledProcessError:
        print("\nFailed to push. If it complains about the kernel not existing, make sure the ID is correct.")
        sys.exit(1)
        
    print("\nNotebook pushed successfully!")
    print(f"You can view it here: https://www.kaggle.com/{kernel_id}")
    
    # 4. Monitor Status
    print("\nWaiting for kernel to start running...")
    time.sleep(15) # Give it a moment to queue
    
    while True:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id], 
            capture_output=True, text=True
        )
        status = result.stdout.strip()
        print(f"Status: {status}")
        
        if "complete" in status.lower() or "error" in status.lower() or "fatal" in status.lower():
            break
            
        time.sleep(30)
        
    # 5. Download outputs
    print("\nRun finished! Downloading outputs...")
    output_dir = Path("kaggle_outputs")
    output_dir.mkdir(exist_ok=True)
    
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)])
    print(f"\nDone! Outputs downloaded to {output_dir.absolute()}")

if __name__ == "__main__":
    main()
