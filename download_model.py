# download_model.py
import os
from huggingface_hub import snapshot_download

model_id = "openthaigpt/openthaigpt1.5-7b-instruct"
local_dir = "./models/openthaigpt1.5-7b-instruct"

ignore_patterns = ["*.gguf", "*.bin"]

if __name__ == "__main__":
    print(f"--- Starting model download for '{model_id}' ---")
    
    if os.path.exists(local_dir):
        print(f"Directory '{local_dir}' already exists. Skipping download.")
        print("If you need to re-download, please delete the directory first.")
    else:
        print(f"Downloading model to '{local_dir}'...")
        print(f"Ignoring patterns: {ignore_patterns}")
        
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False, # แนะนำให้เป็น False บน Windows/WSL
                ignore_patterns=ignore_patterns
            )
            print("\n--- ✅ Model download completed successfully! ---")
        except Exception as e:
            print(f"\n--- ❌ An error occurred during download: {e} ---")