"""
This automation script uses `papermill` to execute data generation notebooks in isolated processes, ensuring that a single batch failure never crashes the entire multi-batch run. By clearing memory between each run and providing automatic checkpointing, it offers a robust and memory-efficient alternative to running loops directly within a notebook. This enables professional, large-scale synthetic data production directly from the command line while keeping the original notebook as a clean, reusable template.

This script can also be executed in parallel by opening multiple terminal windows and running different commands simultaneously (e.g., targeting different models or batches). This parallel approach maximizes data throughput and ensures that a rate limit on one model doesn't stall the generation process for others, significantly reducing the total time required to build large datasets.
"""
import os
import time
import argparse
import papermill as pm
from pathlib import Path



def run_automation(notebook_path, num_batches, wait_time=60, max_attempts=999999):
    """
    Executes a Jupyter notebook multiple times, handling rate limiting errors.
    """
    notebook_name = Path(notebook_path).stem
    print(f"Starting automation for: {notebook_name}")
    print(f"Target: {num_batches} batches")
    
    successful_batches = 0
    while successful_batches < num_batches:
        batch_num = successful_batches + 1
        print(f"\n--- [Batch {batch_num}/{num_batches}] ---")
        
        success = False
        attempt = 0
        
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
                successful_batches += 1
                
            except Exception as e:
                error_msg = str(e).lower()
                attempt += 1
                
                if "rate limit" in error_msg or "429" in error_msg:
                    print(f"Rate limit detected. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                elif "badrequesterror" in error_msg:
                    print(f"Bad Request Error (possibly model decommissioning). Stopping automation.")
                    print(f"Error details: {e}")
                    return
                else:
                    print(f"An unexpected error occurred: {e}")
                    if attempt < max_attempts:
                        print(f"Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        print(f"Max attempts ({max_attempts}) reached for Batch {batch_num}. Stopping automation to prevent data gaps.")
                        return

    print("\nAutomation task completed.")

if __name__ == "__main__":
    """
    ### Setup & Execution Guide ###
    1. Install uv
    2. Configure Environment (if 'uv' command not found)
    3. Add Dependencies (run from project root)
       uv add papermill
    4. Run Automation (from project root):
       - Windows (PowerShell): uv run python script_experiments\generate_synthetic_data.py --notebook pipelines\1-generate_predictions-all_domains.ipynb --batches 10 --wait 60 --retry 999999
       - macOS/Linux/Git Bash: uv run python script_experiments/generate_synthetic_data.py --notebook pipelines/1-generate_predictions-all_domains.ipynb --batches 10 --wait 60 --retry 999999
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
    parser.add_argument(
        "--retry", 
        type=int, 
        default=999999,
        help="Max retries per batch. Default is 999999 (effectively infinite)."
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, '../' + args.notebook)
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file not found at {notebook_path}")
    else:
        run_automation(notebook_path, args.batches, args.wait, args.retry)
