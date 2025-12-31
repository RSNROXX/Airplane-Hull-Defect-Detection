import os
import sys

# Define the pipeline steps (Files must exist in the root folder)
PIPELINE = [
    "train_custom.py",
    "train_transfer.py",
    "compare.py",
    "auto_test.py",
    "predict.py"
]

def run_pipeline():
    print(f"\n{'#'*50}")
    print(f"🚀 STARTING AIRPLANE HULL PROJECT PIPELINE")
    print(f"{'#'*50}\n")

    for script in PIPELINE:
        # Check if file exists
        if not os.path.exists(script):
            print(f"⚠️ SKIPPING {script}: File not found.")
            continue

        print(f"----  ▶️  RUNNING: {script}  ----")
        print("-" * 40)
        
        # Execute the script
        # sys.executable ensures we use the same Python environment
        exit_code = os.system(f"{sys.executable} {script}")
        
        # Check for failure (Exit code != 0)
        if exit_code != 0:
            print(f"\n❌ CRITICAL ERROR in {script}. Pipeline stopped.")
            sys.exit(1) # Stop the whole pipeline
            
        print("-" * 40)
        print(f"✅ {script} completed.\n")

    print(f"\n{'#'*50}")
    print(f"✨ PIPELINE EXECUTION FINISHED SUCCESSFULLY ✨")
    print(f"{'#'*50}")

if __name__ == "__main__":
    run_pipeline()