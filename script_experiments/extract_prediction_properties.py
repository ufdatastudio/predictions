import os
import sys
import ast
import json
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

def load_dataset(base_data_path, dataset_name):
    """
    Load dataset from file path.
    """
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)
    
    data_path = os.path.join(base_data_path, dataset_name)
    
    print(f"Dataset path: {dataset_name}")
    # Assuming DataProcessing handles file not found errors internally
    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head()}\n")
    
    return df

def load_prompts_and_llms(model_names=None):
    """
    Initialize prompts and language models.
    """
    print("\n" + "="*50)
    print("LOAD PROMPTS & MODELS")
    print("="*50)
    
    # Build base prompt
    prediction_properties = PredictionProperties.get_prediction_properties()
    system_identity_prompt, task, format_output = Prompts.extract_prediction_properties()
    base_prompt = f"""{system_identity_prompt} For each prediction, the format is based on: 
    {prediction_properties}
    """
    print("✓ Prompts loaded")
    
    # Initialize models
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

def get_progress(results_path, total_original_rows) -> int:
    """
    Check existing results file to determine where to resume.
    Returns the index to start from.
    """
    start_idx = 0
    if os.path.exists(results_path):
        try:
            # We only need to count rows. 
            # Note: This assumes 1 input row = 1 output row per model run.
            # If using multiple models in one run, this logic assumes 
            # results are saved sequentially per input sentence.
            with open(results_path, 'r') as f:
                row_count = sum(1 for row in f)
            # Subtract 1 for header if the file has content
            start_idx = max(0, row_count - 1)
            print(f"Found existing results. Resuming from index {start_idx}...")
        except Exception as e:
            print(f"Error reading existing file: {e}. Starting from 0.")
            start_idx = 0
    else:
        print("No existing results found. Starting from beginning.")
    
    if start_idx >= total_original_rows:
        print("\nAll sentences have already been processed!")
        sys.exit(0)
        
    print(f"Sentences remaining: {total_original_rows - start_idx}")
    return start_idx

def llm_certifier(idx, sentence_to_classify, base_prompt, task, format_output, model):
    """
    Extract prediction properties from a sentence using an LLM.
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
    
    input_prompt = model.user(prompt)
    raw_text_llm_generation = model.chat_completion([input_prompt])
    
    return raw_text_llm_generation

def process_single_result(text, raw_response, model_name) -> pd.DataFrame:
    """
    Process a single LLM result into a DataFrame row.
    Helper function used inside the loop.
    """
    # Create structure
    data = {
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
    
    try:
        parsed = ast.literal_eval(raw_response)
        # Handle dictionary parsing
        results_df.at[0, 'No Property'] = ', '.join(parsed.get(0, []) or parsed.get("0", []))
        results_df.at[0, 'Source'] = ', '.join(parsed.get(1, []) or parsed.get("1", []))
        results_df.at[0, 'Target'] = ', '.join(parsed.get(2, []) or parsed.get("2", []))
        results_df.at[0, 'Date'] = ', '.join(parsed.get(3, []) or parsed.get("3", []))
        results_df.at[0, 'Outcome'] = ', '.join(parsed.get(4, []) or parsed.get("4", []))
    except:
        pass  # Keep empty strings if parsing fails
        
    return results_df

def run_processing_pipeline(df, text_column, base_prompt, task, format_output, models, results_path, dataset_basename):
    """
    Process sentences, handle errors, and save incrementally.
    """
    print("\n" + "="*50)
    print("PROCESSING SENTENCES")
    print("="*50)
    print(f"Sentences to process: {len(df)}")
    print(f"Models per sentence: {len(models)}")
    
    # We loop through the dataframe
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        text = row[text_column]
        
        # Store results for this single sentence (across all models)
        sentence_results_dfs = []
        
        for model in models:
            try:
                raw_response = llm_certifier(idx, text, base_prompt, task, format_output, model)
                
                # Process immediately
                single_df = process_single_result(text, raw_response, model.__name__())
                sentence_results_dfs.append(single_df)
                
            except Exception as e:
                print(f"\n[!] Error processing index {idx} with {model.__name__()}: {e}")
                # Optional: Add sleep here if needed for rate limits
                continue
        
        # Save to CSV immediately if we have results
        if sentence_results_dfs:
            batch_df = pd.concat(sentence_results_dfs, ignore_index=True)
            batch_df['Dataset Source'] = dataset_basename
            
            # Determine if we need header (only if file doesn't exist)
            file_exists = os.path.exists(results_path)
            
            # Append mode
            batch_df.to_csv(results_path, mode='a', header=not file_exists, index=False)

    print(f"\n✓ Processing complete. Results saved to {results_path}\n")

if __name__ == "__main__":
    """Usage
    # Default models
    python3 extract_projection_properties.py
    # Single model
    python3 extract_projection_properties.py --models llama-3.1-70b-instruct
    """

    """NEEDS:
    + Change "Prompts.extract_projection_properties()" to "Prompts.extract_prediction_properties()"
    + read and save with my functions in DataProcessing
    + handle if prompt changes, but everything else same
    + get temperature and top_p for each llm and more metadata
    + be sure we are using llms in text_generation_models.py vs model.chat_completion() in llm_certifier
    """
        
    print("\n" + "="*50)
    print("PREDICTION PROPERTY EXTRACTION")
    print("="*50)
    
    # ============================================================
    # 1. CONFIGURATION & ARGS
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_dataset = DataProcessing.load_single_synthetic_data(
        script_dir, batch_idx=2, sep=',', return_as='path'
    )
    
    parser = argparse.ArgumentParser(description='Extract prediction properties from sentences using LLMs')
    parser.add_argument('--dataset', default=default_dataset, 
                       help='Path to dataset relative to base data directory.')
    parser.add_argument('--models', nargs='+', default='llama-3.1-8b-instant',
                       help='Model name(s) to use.')
    # Added arguments for flexible column names
    parser.add_argument('--text_column', type=str, default='Base Sentence',
                       help='Column name containing the text to analyze')
    args = parser.parse_args()
    
    # ============================================================
    # 2. LOAD DATASET & VALIDATE COLUMNS
    # ============================================================
    df = load_dataset(base_data_path, args.dataset)
    
    if args.text_column not in df.columns:
        print(f"\n❌ ERROR: Text column '{args.text_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}. Enter column of interests.")
        args.text_column = input("Enter column of interests: ")
        print(f"Text column '{args.text_column}'")

        # sys.exit(1)

    # ============================================================
    # 3. LOAD PROMPTS and LLMs
    # ============================================================
    base_prompt, task, format_output, models = load_prompts_and_llms(args.models)
    
    # ============================================================
    # 4. SETUP OUTPUT DIRECTORY
    # ============================================================
    dataset_basename = os.path.basename(args.dataset).split('.')[0]
    model_names_str = '-'.join([m.__name__() for m in models])
    
    # Structure: data/extract_prediction_properties/{dataset}/{model_name}/
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

    # ============================================================
    # 4b. SAVE METADATA (Crucial for Reproducibility)
    # ============================================================
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset": args.dataset,
        "dataset_basename": dataset_basename,
        "text_column": args.text_column,
        "models_used": [m.__name__() for m in models], 
        # saving the specific prompts is vital if you iterate on them later
        "prompts": {
            "base_prompt": base_prompt,
            "task": task, 
            "format_output": format_output
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Only write metadata if it doesn't exist (to preserve original run config)
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")
    else:
        print(f"Metadata already exists at: {metadata_path}")

    # ============================================================
    # 5. CHECK PROGRESS & RESUME
    # ============================================================
    # Determine start index based on existing results file
    start_idx = get_progress(results_path, len(df))
    
    # Slice the dataframe
    df_to_process = df.iloc[start_idx:]
    
    # ============================================================
    # 6. RUN PIPELINE
    # ============================================================
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
    
    # ============================================================
    # 7. FINAL SUMMARY
    # ============================================================
    if os.path.exists(results_path):
        final_df = pd.read_csv(results_path)
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        
        summary = {
            "total_processed": len(final_df),
            "shape": final_df.shape,
            "columns": list(final_df.columns),
            "models_used": list(final_df['Model Name'].unique()),
            "sample_results": final_df[['Sentence', 'Source', 'Target', 'Date', 'Model Name']].head(3).to_dict('records')
        }
        
        print(json.dumps(summary, indent=2))
        print("\n✓ Pipeline complete!")