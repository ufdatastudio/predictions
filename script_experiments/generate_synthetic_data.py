import os
import time
import argparse
import papermill as pm
from pathlib import Path



def run_automation(notebook_path, num_batches, wait_time=60):
    """
    Executes a Jupyter notebook multiple times, handling rate limiting errors.
    """
    notebook_name = Path(notebook_path).stem
    print(f"Starting automation for: {notebook_name}")
    print(f"Target: {num_batches} batches")
    
    for i in range(num_batches):
        batch_num = i + 1
        print(f"\n--- [Batch {batch_num}/{num_batches}] ---")
        
        success = False
        attempt = 0
        max_attempts = 5
        
        while not success and attempt < max_attempts:
            try:
                print(f"Executing notebook (Attempt {attempt + 1})...")
                # Execute in-place by using the same path for input and output
                pm.execute_notebook(
                    notebook_path,
                    notebook_path,
                    cwd=os.path.dirname(os.path.abspath(notebook_path))
                )
                print(f"Batch {batch_num} completed successfully.")
                success = True
                
            except Exception as e:
                error_msg = str(e).lower()
                attempt += 1
                
                if "rate limit" in error_msg or "429" in error_msg:
                    print(f"Rate limit detected. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                elif "badrequesterror" in error_msg:
                    print(f"Bad Request Error (possibly model decommissioning). Stopping batch.")
                    print(f"Error details: {e}")
                    break
                else:
                    print(f"An unexpected error occurred: {e}")
                    if attempt < max_attempts:
                        print(f"Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        print("Max attempts reached. Skipping this batch.")
                        break

    print("\nAutomation task completed.")

if __name__ == "__main__":
    """
    python generate_synthetic_data.py --notebook /Users/justins/predictions-1/script_experiments/generate_synthetic_data.py --batches 1 --wait 60
    """
    parser = argparse.ArgumentParser(description="Automate synthetic data generation via notebooks.")
    parser.add_argument(
        "--notebook", 
        type=str, 
        required=True,
        help="Path to the .ipynb notebook to execute."
    )
    parser.add_argument(
        "--batches", 
        type=int, 
        default=1,
        help="Number of batches to generate (default: 1)."
    )
    parser.add_argument(
        "--wait", 
        type=int, 
        default=60,
        help="Seconds to wait on rate limit error (default: 60)."
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, '../' + args.notebook)
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file not found at {notebook_path}")
    else:
        run_automation(notebook_path, args.batches, args.wait)
