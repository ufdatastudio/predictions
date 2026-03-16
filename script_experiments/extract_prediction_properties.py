import os
import sys
import ast
import json
import re
import time
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List

# Add project modules to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))
from prompts import Prompts
from data_processing import DataProcessing
from prediction_properties import PredictionProperties
from text_generation_models import TextGenerationModelFactory

# Config for batch processing
BATCH_SIZE = 10

def load_dataset(base_data_path, dataset_name):
    """Load dataset from file path."""
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)
    
    data_path = os.path.join(base_data_path, dataset_name)
    print(f"Dataset path: {dataset_name}")
    
    # Load dataset
    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    
    # Ensure we have a stable index for resuming later
    # If the file doesn't have a unique ID, the pandas index is used.
    # We reset index to ensure it's a clean 0..N range for tracking.
    df = df.reset_index(drop=True)
    
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head()}\n")
    return df

def load_prompts_and_llms(model_names=None):
    """Initialize prompts and language models."""
    print("\n" + "="*50)
    print("LOAD PROMPTS & MODELS")
    print("="*50)
    
    prediction_properties = PredictionProperties.get_prediction_properties()
    system_identity_prompt, task, format_output = Prompts.extract_prediction_properties()
    base_prompt = f"""{system_identity_prompt} For each prediction, the format is based on: 
    {prediction_properties}
    """
    print("✓ Prompts loaded")
    
    tgmf = TextGenerationModelFactory()
    
    if model_names is None:
        model_names = ['llama-3.1-8b-instant']
    elif isinstance(model_names, str):
        model_names = [model_names]
    
    models = []
    for model_name in model_names:
        try:
            model = tgmf.create_instance(model_name=model_name)
            models.append(model)
            print(f"✓ Loaded: {model.__name__()}")
        except ValueError as e:
            print(f"✗ Failed to load {model_name}: {e}")
    
    if not models:
        raise ValueError("No models were successfully loaded")
    
    print(f"\n✓ Total models loaded: {len(models)}\n")
    return base_prompt, task, format_output, models

def get_remaining_data(df, results_path):
    """
    Identify which rows still need processing by checking specific indices 
    in the existing results file, rather than just counting lines.
    """
    if not os.path.exists(results_path):
        print("No existing results found. Starting from scratch.")
        return df

    try:
        # Read only the 'Input_Index' column to save memory
        # We assume the results file has an 'Input_Index' column (added in processing)
        existing_df = pd.read_csv(results_path, usecols=['Input_Index'])
        processed_indices = set(existing_df['Input_Index'].unique())
        
        print(f"Found {len(processed_indices)} unique processed sentences.")
        
        # Filter the original dataframe to exclude processed indices
        # This is robust against skipped rows or out-of-order saving
        df_remaining = df[~df.index.isin(processed_indices)]
        
        print(f"Resuming processing. {len(df_remaining)} sentences remaining.")
        return df_remaining
        
    except ValueError:
        # If 'Input_Index' doesn't exist (e.g., old file format), fallback to index slicing
        print("Warning: 'Input_Index' column not found in results. Falling back to row counting.")
        with open(results_path, 'r') as f:
            row_count = sum(1 for row in f) - 1 # minus header
        return df.iloc[max(0, row_count):]
    except Exception as e:
        print(f"Error reading existing file: {e}. Starting from scratch.")
        return df

def extract_json_from_text(text):
    """
    Robustly extract JSON dictionary from LLM response.
    """
    text = str(text).strip()
    
    # Remove markdown code blocks if present
    # This regex handles ```json ... ``` or just ``` ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback to finding the first outer curly braces
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        json_str = match.group(1) if match else text
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_str)
        except:
            return None

def llm_certifier(idx, sentence_to_classify, base_prompt, task, format_output, model, wait_time=360):
    """
    Extract prediction properties. Handles API failures by returning error dict instead of crashing.
    """
    prompt = f"""{base_prompt}
    
    Sentence to extract the prediction properties: '{sentence_to_classify}'
    {task}
    
    {format_output}
    """
    # Debug: show prompt for first few examples
    if idx < 2:
        print(f"\n--- Sample Prompt (idx={idx}) ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
    success = False
    attempt = 0
    max_attempts = 5
    
    while not success and attempt < max_attempts:
        try:
            input_prompt = model.user(prompt)
            raw_text_llm_generation = model.chat_completion([input_prompt])
            return raw_text_llm_generation # Success
            
        except Exception as e:
            error_msg = str(e).lower()
            attempt += 1
            print(f"Attempt {attempt}/{max_attempts} failed for index {idx}. Error: {e}")
            
            if "rate limit" in error_msg or "429" in error_msg:
                print(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif "badrequesterror" in error_msg:
                print(f"Bad Request. Stopping retry for this item.")
                break
            else:
                time.sleep(5) # Short wait for other errors
    
    # If we reach here, we failed. Return a structured error string.
    # We do NOT skip the row; we record the failure.
    return str({"0": ["ERROR_MAX_RETRIES"], "1": [], "2": [], "3": [], "4": []})

def process_single_result(input_index, text, raw_response, model_name) -> pd.DataFrame:
    """
    Process a single LLM result into a DataFrame row.
    Includes input_index for robust resuming.
    """
    # Create structure
    data = {
        'Input_Index': [input_index], # Crucial for row counting fix
        'Sentence': [text],
        'Raw Response': [raw_response],
        'Model Name': [model_name],
        'No Property': [''],
        'Source': [''],
        'Target': [''],
        'Date': [''],
        'Outcome': ['']
    }
    
    results_df = pd.DataFrame(data)
    
    # Use the robust extraction function
    parsed = extract_json_from_text(raw_response)
    
    if parsed and isinstance(parsed, dict):
        try:
            # Handle potential string/list variations in the JSON
            # Note: We cast to string in case the model returns non-string types
            results_df.at[0, 'No Property'] = str(parsed.get(0, []) or parsed.get("0", []) or "")
            results_df.at[0, 'Source'] = str(parsed.get(1, []) or parsed.get("1", []) or "")
            results_df.at[0, 'Target'] = str(parsed.get(2, []) or parsed.get("2", []) or "")
            results_df.at[0, 'Date'] = str(parsed.get(3, []) or parsed.get("3", []) or "")
            results_df.at[0, 'Outcome'] = str(parsed.get(4, []) or parsed.get("4", []) or "")
        except Exception as e:
            print(f"Error mapping JSON to columns: {e}")
    else:
        # Mark as parse error but keep the row
        results_df.at[0, 'No Property'] = "PARSE_ERROR"
        
    return results_df

def save_batch(batch_dfs, results_path):
    """Helper to save a list of dataframes to CSV."""
    if not batch_dfs:
        return
        
    batch_df = pd.concat(batch_dfs, ignore_index=True)
    
    # Determine if we need header (only if file doesn't exist)
    file_exists = os.path.exists(results_path)
    
    # Append mode
    batch_df.to_csv(results_path, mode='a', header=not file_exists, index=False)

def run_processing_pipeline(df, text_column, base_prompt, task, format_output, models, results_path, dataset_basename):
    """
    Process sentences with batch saving and robust error handling.
    """
    print("\n" + "="*50)
    print("PROCESSING SENTENCES")
    print("="*50)
    print(f"Sentences to process: {len(df)}")
    
    batch_results = []
    
    # Loop through the filtered dataframe
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row[text_column]
        
        for model in models:
            # Get raw response (handles retries internally)
            raw_response = llm_certifier(idx, text, base_prompt, task, format_output, model)
            
            # Process result
            single_df = process_single_result(idx, text, raw_response, model.__name__)
            single_df['Dataset Source'] = dataset_basename
            
            batch_results.append(single_df)
        
        # Check if batch is full
        if len(batch_results) >= BATCH_SIZE:
            save_batch(batch_results, results_path)
            batch_results = [] # Clear buffer

    # Save any remaining results in the buffer
    if batch_results:
        save_batch(batch_results, results_path)

    print(f"\n✓ Processing complete. Results saved to {results_path}\n")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("PREDICTION PROPERTY EXTRACTION")
    print("="*50)
    
    # 1. CONFIGURATION & ARGS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_dataset = DataProcessing.load_single_synthetic_data(
        script_dir, batch_idx=1, sep=',', return_as='path'
    )
    
    parser = argparse.ArgumentParser(description='Extract prediction properties from sentences using LLMs')
    parser.add_argument('--dataset', default=default_dataset, 
                       help='Path to dataset relative to base data directory.')
    parser.add_argument('--models', nargs='+', default='openai/gpt-oss-120b',
                       help='Model name(s) to use.')
    parser.add_argument('--text_column', type=str, default='Base Sentence',
                       help='Column name containing the text to analyze')
    args = parser.parse_args()
    
    # 2. LOAD DATASET
    df = load_dataset(base_data_path, args.dataset)
    
    if args.text_column not in df.columns:
        print(f"\n❌ ERROR: Text column '{args.text_column}' not found in dataset")
        # Fail gracefully instead of hanging on input() for batch jobs
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # 3. LOAD PROMPTS and LLMs
    base_prompt, task, format_output, models = load_prompts_and_llms(args.models)
    
    # 4. SETUP OUTPUT DIRECTORY
    dataset_basename = os.path.basename(args.dataset).split('.')[0]
    model_names_str = '-'.join([m.__name__() for m in models])
    
    output_dir = os.path.join(
        base_data_path, 
        "extract_prediction_properties", 
        dataset_basename,
        model_names_str
    )
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "results.csv")
    print(f"Output Directory: {output_dir}")
    print(f"Results File: {results_path}")

    # 4b. SAVE METADATA
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset": args.dataset,
        "dataset_basename": dataset_basename,
        "text_column": args.text_column,
        "models_used": [m.__name__() for m in models], 
        "prompts": {
            "base_prompt": base_prompt,
            "task": task, 
            "format_output": format_output
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")
    else:
        print(f"Metadata already exists at: {metadata_path}")

    # 5. CHECK PROGRESS & RESUME (ROBUST)
    # This filters the dataframe to only include rows that haven't been processed yet
    df_to_process = get_remaining_data(df, results_path)
    
    if df_to_process.empty:
        print("\nAll sentences have already been processed!")
    else:
        # 6. RUN PIPELINE
        run_processing_pipeline(
            df_to_process, 
            args.text_column, 
            base_prompt, 
            task, 
            format_output, 
            models, 
            results_path,
            dataset_basename
        )
    
    # 7. FINAL SUMMARY
    if os.path.exists(results_path):
        try:
            final_df = pd.read_csv(results_path)
            print("\n" + "="*50)
            print("FINAL RESULTS SUMMARY")
            print("="*50)
            
            summary = {
                "total_processed": len(final_df),
                "shape": final_df.shape,
                "columns": list(final_df.columns),
                "models_used": list(final_df['Model Name'].unique()),
                "sample_results": final_df[['Sentence', 'Source', 'Target', 'Date', 'Model Name']].head(3).to_dict('records') if not final_df.empty else []
            }
            
            print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"Could not print summary: {e}")
            
    print("\n✓ Pipeline complete!")