import shutil
import os

def setup():
    # Ensure we are in the right base directory
    # We expect to be in "c:\Users\parth\Downloads\Crop Yield\python_implementation" based on Cwd usually
    # But files are in "..\old data"
    
    current_dir = os.getcwd()
    print(f"Current working dir: {current_dir}")
    
    # Adjust paths based on where we are running
    if current_dir.endswith("python_implementation"):
        project_root = current_dir
        repo_root = os.path.dirname(current_dir)
    else:
        # Assuming we are in "Crop Yield"
        project_root = os.path.join(current_dir, "python_implementation")
        repo_root = current_dir
        
    src_dir = os.path.join(repo_root, "old data", "unimportant files")
    dst_dir = os.path.join(project_root, "data", "raw")
    
    print(f"Source: {src_dir}")
    print(f"Dest: {dst_dir}")
    
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    
    files = ["CropData.csv", "MaharashtrastateRainfall.csv", "District_ph.csv", "Crop_ph.csv", "CropRequiredTemperature.csv"]
    
    for f in files:
        s = os.path.join(src_dir, f)
        d = os.path.join(dst_dir, f)
        if os.path.exists(s):
            try:
                shutil.copy2(s, d)
                print(f"Successfully copied {f}")
            except Exception as e:
                print(f"Error copying {f}: {e}")
        else:
            print(f"Source file not found: {s}")

if __name__ == "__main__":
    setup()
