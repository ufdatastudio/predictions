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
from prompts import EntityExtractionPrompt
from data_processing import DataProcessing
from prediction_properties import PredictionProperties
from text_generation_models import TextGenerationModelFactory

# How many sentence results to collect in memory before writing to disk
BATCH_SIZE = 10

# Stop after this many sentences (set to None to process all)
# STOP_AFTER = 10
STOP_AFTER = None

def load_dataset(base_data_path, dataset_name):
    """
    Load a dataset from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    base_data_path : str
        The root data directory path.
    dataset_name : str
        The relative path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset with a clean 0..N index.
    """
    print("\n" + "="*50)
    print("STEP: LOAD DATASET")
    print("="*50)

    data_path = os.path.join(base_data_path, dataset_name)
    print(f"Dataset path: {dataset_name}")

    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    # df = df.sample(n=7, random_state=42)

    # Reset index so we have a clean 0, 1, 2, ... row numbers.
    # This is important for the resume logic later — we track which
    # row numbers have already been processed.
    df = df.reset_index(drop=True)

    print(f"Shape: {df.shape}")
    print(f"\nFirst 7 rows:\n{df.head(7)}\n")
    print(f"\nLast 7 rows:\n{df.tail(7)}\n")
    return df

def load_prompts_and_llm(model_name=None):
    """
    Build the base prompt and load a single language model.

    The base prompt combines:
    - Who the model is (system identity)
    - What a prediction looks like (prediction properties)
    - Rules the model must follow (requirements)
    - Examples to guide the model (few-shot examples)

    Parameters
    ----------
    model_name : str, optional
        Model name to load. Defaults to 'llama-3.1-8b-instant'.

    Returns
    -------
    base_prompt : str
        The full prompt sent to the model before each sentence.
    task : str
        The labeling instruction for the model.
    format_output : str
        The expected JSON output format.
    model : object
        Loaded model instance.
    """
    print("\n" + "="*50)
    print("STEP: LOAD PROMPTS & MODEL")
    print("="*50)

    # Get prediction properties and requirements from PredictionProperties
    prediction_properties, prediction_requirements = PredictionProperties.get_prediction_properties_and_requirements()

    # Build the prompt using EntityExtractionPrompt with few-shot examples
    prompt = EntityExtractionPrompt()
    system_identity, task, format_output, examples = prompt.few_shot()

    # Combine everything into one base prompt that will be sent before each sentence
    base_prompt = f"""{system_identity}
    Prediction Properties:
    {prediction_properties}
    Requirements:
    {prediction_requirements}
    Examples:
    {examples}
    """

    print("\n--- Base Prompt ---")
    print(base_prompt)
    print("--- End Base Prompt ---\n")
    print("✓ Prompts loaded")

    # Load the single model
    if model_name is None:
        model_name = 'llama-3.1-8b-instant'

    tgmf = TextGenerationModelFactory()

    try:
        model = tgmf.create_instance(model_name=model_name)
        print(f"✓ Loaded: {model.__name__()}")
    except ValueError as e:
        raise ValueError(f"✗ Failed to load {model_name}: {e}")

    print(f"\n✓ Model loaded: {model.__name__()}\n")
    return base_prompt, task, format_output, model

def get_remaining_data(df, results_path):
    """
    Filter the dataset to only include sentences that have NOT been processed yet.

    This allows the pipeline to resume from where it left off if it was
    interrupted (e.g., SLURM job timeout, rate limit crash, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    results_path : str
        Path to the existing results CSV file (if it exists).

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing only unprocessed sentences.
    """
    if not os.path.exists(results_path):
        print("No existing results found. Starting from scratch.")
        return df

    try:
        # Load only the Input_Index column to save memory.
        # Input_Index tracks which row numbers have already been processed.
        existing_df = pd.read_csv(results_path, usecols=['Input_Index'])
        processed_indices = set(existing_df['Input_Index'].unique())
        print(f"Found {len(processed_indices)} already processed sentences.")

        # Keep only rows whose index is NOT in the processed set
        df_remaining = df[~df.index.isin(processed_indices)]
        print(f"Resuming. {len(df_remaining)} sentences remaining.")
        return df_remaining

    except ValueError:
        # If Input_Index column does not exist, fall back to row counting
        print("Warning: 'Input_Index' column not found. Falling back to row counting.")
        with open(results_path, 'r') as f:
            row_count = sum(1 for row in f) - 1  # subtract header row
        return df.iloc[max(0, row_count):]

    except Exception as e:
        print(f"Error reading existing results file: {e}. Starting from scratch.")
        return df

def join_property(values):
    """
    Convert a list of property values to a pipe-separated string.

    Parameters
    ----------
    values : list or str
        The extracted property values from the LLM response.

    Returns
    -------
    str
        A pipe-separated string of values.
        Example: ['stock price', 'remain stable'] -> 'stock price|remain stable'

    Examples
    --------
    >>> join_property(['stock price', 'remain stable'])
    'stock price|remain stable'
    >>> join_property('Analyst Michael Chen')
    'Analyst Michael Chen'
    >>> join_property([])
    ''
    """
    if isinstance(values, list):
        return '|'.join([str(v).strip() for v in values if v])
    if isinstance(values, str):
        return values.strip()
    return ''

def process_single_result(input_index, text, raw_response, model_name, seed) -> pd.DataFrame:
    """
    Convert a single LLM response into a structured one-row DataFrame.

    The LLM returns a raw string (hopefully JSON). This function parses
    that string and maps each key to the correct property column.

    Parameters
    ----------
    input_index : int
        The row number of this sentence in the original dataset.
        Stored so the resume logic can skip this row on future runs.
    text : str
        The original sentence that was sent to the model.
    raw_response : str
        The raw string response returned by the model.
    model_name : str
        The name of the model that generated the response.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with columns:
        Seed, Input_Index, Sentence, Raw Response, Model Name,
        Parse Status, No Property, Source, Target, Date, Outcome.
    """
    # Start with an empty structure — we will fill in the properties below
    data = {
        'Seed':         [seed],
        'Input_Index':  [input_index],
        'Sentence':     [text],
        'Raw Response': [raw_response],
        'Model Name':   [model_name],
        'Parse Status': [''],
        'No Property':  [''],
        'Source':       [''],
        'Target':       [''],
        'Date':         [''],
        'Outcome':      ['']
    }
    results_df = pd.DataFrame(data)

    # parse_slot_filling_response always returns a dict — never None
    # Keys are always strings "0" through "4", defaulting to [] on failure
    parsed = DataProcessing.parse_slot_filling_response(raw_response)

    try:
        results_df.at[0, 'No Property'] = join_property(parsed.get("0", []))
        results_df.at[0, 'Source']      = join_property(parsed.get("1", []))
        results_df.at[0, 'Target']      = join_property(parsed.get("2", []))
        results_df.at[0, 'Date']        = join_property(parsed.get("3", []))
        results_df.at[0, 'Outcome']     = join_property(parsed.get("4", []))

        # If all values came back empty, the parse silently failed
        all_empty = all(
            parsed.get(k, []) == [] for k in ["0", "1", "2", "3", "4"]
        )
        results_df.at[0, 'Parse Status'] = 'PARSE_ERROR' if all_empty else 'OK'

    except Exception as e:
        print(f"Error mapping JSON to columns for index {input_index}: {e}")
        results_df.at[0, 'Parse Status'] = 'PARSE_ERROR'

    return results_df

def save_batch(batch_dfs, results_path):
    """
    Write a list of single-row DataFrames to the results CSV file.

    Instead of writing to disk after every sentence (slow), we collect
    BATCH_SIZE rows in memory and flush them all at once here.

    Parameters
    ----------
    batch_dfs : list of pd.DataFrame
        A list of single-row DataFrames to save.
    results_path : str
        Path to the results CSV file.
    """
    if not batch_dfs:
        return

    batch_df = pd.concat(batch_dfs, ignore_index=True)

    results_dir = os.path.dirname(results_path)
    prefix = os.path.basename(results_path).split('.')[0]  # e.g., "extracted_properties"

    DataProcessing.save_to_file(
        data=batch_df,
        path=results_dir,
        prefix=prefix,
        save_file_type='csv',
        include_version=False,
        append=True
    )
    # print(f"✓ Saved metrics summary to: {results_dir}")

def extract_properties(
        df, 
        text_column, 
        base_prompt, 
        task, 
        format_output, 
        model, 
        results_path, 
        dataset_basename, 
        seed,
        stop_after=None):
    """
    Process sentences with batch saving and robust error handling.

    Parameters
    ----------
    df : pd.DataFrame
        The filtered DataFrame of sentences still needing processing.
    text_column : str
        The column name containing the sentences to extract properties from.
    base_prompt : str
        The full prompt sent before each sentence.
    task : str
        The labeling instruction for the model.
    format_output : str
        The expected JSON output format.
    model : object
        Loaded model instance.
    results_path : str
        Path to the results CSV file.
    dataset_basename : str
        The dataset name, stored in results so we know which dataset each row came from.
    stop_after : int or None
        If set, stop processing after this many sentences. Useful for testing.
        Default is None (process all sentences).
    """
    print("\n" + "="*50)
    print("STEP: EXTRACT PROPERTIES")
    print("="*50)
    print(f"Sentences to process: {len(df)}")

    if stop_after:
        print(f"⚠️  STOP_AFTER={stop_after}: Will stop after {stop_after} sentences for testing.")

    batch_results = []
    sentences_processed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):

        # Stop early if stop_after is set
        if stop_after is not None and sentences_processed >= stop_after:
            print(f"\n⚠️  Reached STOP_AFTER={stop_after}. Stopping early.")
            break

        text = row[text_column]

        prompt = f"""{base_prompt}
        Sentence to extract the prediction properties: '{text}'
        {task}
        {format_output}
        """

        if idx < 2:
            print(f"\n--- Sample Prompt (idx={idx}) ---")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

        input_prompt = model.user(prompt)
        raw_response = model.safe_chat_completion([input_prompt], idx=idx)

        # Proactive sleep to stay within Groq TPM limits
        # Groq recommends ~6.21s between requests for openai/gpt-oss-120b
        time.sleep(7)

        if raw_response is None:
            raw_response = str({"0": ["ERROR_MAX_RETRIES"], "1": [], "2": [], "3": [], "4": []})

        single_df = process_single_result(idx, text, raw_response, model.__name__(), seed)
        single_df['Dataset Source'] = dataset_basename
        batch_results.append(single_df)

        sentences_processed += 1

        if len(batch_results) >= BATCH_SIZE:
            save_batch(batch_results, results_path)
            batch_results = []

    if batch_results:
        save_batch(batch_results, results_path)

    print(f"\n✓ Processing complete. Results saved to {results_path}\n")

def create_properties_experiment_log(
    args, 
    model_name, 
    output_dir, 
    df_original_shape,
    df_sampled_shape,
    base_prompt,
    task,
    format_output
):
    """
    Generate and save a human-readable experiment log for property extraction.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    model_name : str
        Name of the LLM model used
    output_dir : str
        Directory where log will be saved
    df_original_shape : tuple
        Shape of dataset before sampling (rows, cols)
    df_sampled_shape : tuple
        Shape of dataset after sampling (rows, cols)
    base_prompt : str
        Base prompt template used
    task : str
        Task instruction string
    format_output : str
        Expected output format string
    """
    from datetime import datetime
    
    log_lines = []
    log_lines.append("="*60)
    log_lines.append("PROPERTY EXTRACTION EXPERIMENT LOG")
    log_lines.append("="*60)
    log_lines.append(f"Timestamp:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Task:              {args.task_name}")
    log_lines.append(f"Seed:              {args.seed}")
    log_lines.append("")
    
    log_lines.append("--- Data ---")
    log_lines.append(f"Dataset Source:    {args.dataset or args.dataset_path}")
    log_lines.append(f"Text Column:       {args.text_column}")
    log_lines.append(f"Original Shape:    {df_original_shape}")
    
    if args.sample_fraction:
        log_lines.append(f"Sample Fraction:   {args.sample_fraction}")
        log_lines.append(f"Stratify Cols:     {args.stratify_cols}")
        log_lines.append(f"Sampling Method:   {args.sampling_method}")
        log_lines.append(f"Sampled Shape:     {df_sampled_shape}")
    else:
        log_lines.append(f"Sampling:          None (full dataset)")
    
    log_lines.append("")
    log_lines.append("--- Model ---")
    log_lines.append(f"Model Name:        {model_name}")
    log_lines.append(f"Clean Model Name:  {args.model_name.replace('/', '_')}")
    log_lines.append("")
    
    log_lines.append("--- Prompts ---")
    log_lines.append("Base Prompt:")
    log_lines.append(base_prompt[:500] + "..." if len(base_prompt) > 500 else base_prompt)
    log_lines.append("")
    log_lines.append("Task Instruction:")
    log_lines.append(task)
    log_lines.append("")
    log_lines.append("Format Output:")
    log_lines.append(format_output)
    log_lines.append("")
    
    log_lines.append("--- Output ---")
    log_lines.append(f"Output Directory:  {output_dir}")
    log_lines.append(f"Results File:      {os.path.join(output_dir, 'extracted_properties.csv')}")
    log_lines.append(f"Metadata File:     {os.path.join(output_dir, 'extracted_properties_metadata.json')}")
    log_lines.append("")
    
    log_lines.append("--- Processing ---")
    log_lines.append(f"BATCH_SIZE:        {BATCH_SIZE}")
    log_lines.append(f"STOP_AFTER:        {STOP_AFTER if STOP_AFTER else 'None (process all)'}")
    log_lines.append("")
    
    log_lines.append("="*60)
    
    # Save to log file
    log_dir = os.path.join(output_dir, 'experiment_log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'experiment_log.txt')
    
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    
    print(f"✓ Experiment log saved to: {log_path}")

if __name__ == "__main__":
    """
    Usage:
        Be sure to run: source .venv_predictions/bin/activate

        # ============================================================
        # STEP 1: Create combined dataset (run once)
        # ============================================================
        python3 create_combined_dataset.py \
            --datasets synthetic financial_phrasebank chronicle2050 timebank yt news_api mf_climate \
            --predictions_only_datasets yt news_api mf_climate \
            --output_name properties_july_2026 \
            --no_version

        # ============================================================
        # STEP 2: Extract properties for ground truth (with 10% sampling)
        # ============================================================
        python3 llm-experiment.py \
            --dataset_path combined_datasets/properties_july_2026/properties_july_2026.csv \
            --model_name "llama-3.1-8b-instant" \
            --task_name ground_truth \
            --sample_fraction 0.1 \
            --seed 7

        # ============================================================
        # STEP 3: Test LLM extraction ability (classification task)
        # ============================================================
        python3 llm-experiment.py \
            --dataset_path combined_datasets/properties_july_2026/properties_july_2026.csv \
            --model_name "llama-3.1-8b-instant" \
            --task_name classification \
            --sample_fraction 0.1 \
            --seed 7

        # ============================================================
        # Alternative: Load via named loader (for quick testing)
        # ============================================================
        python3 llm-experiment.py \
            --dataset chronicle2050 \
            --model_name "llama-3.1-8b-instant" \
            --task_name ground_truth \
            --seed 7
    """
    print("\n" + "="*50)
    print("SENTENCE PROPERTY EXTRACTION")
    print("="*50)

    # ============================================================
    # 1. Configuration and Arguments
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    dataset_loader_map = {
        'synthetic':            DataProcessing.load_synthetic_dataset,
        'fin_phrasebank':       DataProcessing.load_financial_phrasebank_dataset,
        'chronicle2050':        DataProcessing.load_chronicle2050_dataset,
        'news_api':             DataProcessing.load_news_api_dataset,
        'yt':                   DataProcessing.load_yt_dataset,
        'timebank':             DataProcessing.load_timebank_dataset,
        'mf_climate':           DataProcessing.load_mf_climate_dataset,
        'clients_rivals_rouges': DataProcessing.load_clients_rivals_rouges_dataset,
        'forecast_bench':       DataProcessing.load_forecast_bench_dataset,
        'smart_hospitals':      DataProcessing.load_smart_hospitals_dataset
    }

    task_name_map = {
        'ground_truth':   'ground_truth',
        'classification': 'classification'
    }

    parser = argparse.ArgumentParser(description='Extract properties from sentences using LLMs.')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(dataset_loader_map.keys()),
        default=None,
        help='Named dataset to load via DataProcessing loader (for quick testing).'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to a pre-saved CSV file relative to base_data_path (primary workflow).'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='llama-3.1-8b-instant',
        help='LLM model name to use for extraction.'
    )
    parser.add_argument(
        '--text_column',
        type=str,
        default='Base Sentence',
        help='Column name containing the sentences to extract properties from.'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        choices=list(task_name_map.keys()),
        default='ground_truth',
        help='Either ground_truth for establishing labels or classification for testing extraction.'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=7, 
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--sample_fraction',
        type=float,
        default=None,
        help='Take stratified sample (e.g., 0.1 for 10%). Maintains dataset/label proportions.'
    )
    parser.add_argument(
        '--stratify_cols',
        nargs='+',
        default=['Ground Truth', 'Dataset Name'],
        help='Columns to stratify by when sampling. Default: Ground Truth, Dataset Name'
    )
    parser.add_argument(
        '--sampling_method',
        choices=['hierarchical', 'pair', 'simple'],
        default='hierarchical',
        help='Stratified sampling strategy. Default: hierarchical'
    )
    args = parser.parse_args()

    print(f"Task       : {args.task_name}")
    print(f"Model      : {args.model_name}")
    print(f"Dataset    : {args.dataset or args.dataset_path}")
    print(f"Seed       : {args.seed}")

    # ============================================================
    # 2. Load Prompts and Model
    # ============================================================
    base_prompt, task, format_output, model = load_prompts_and_llm(args.model_name)

    # ============================================================
    # 3. Setup Model Name for Output Directory
    # ============================================================
    clean_model_name = args.model_name.replace('/', '_')

    # ============================================================
    # 4. Load Dataset
    # ============================================================
    if args.dataset is not None and args.dataset_path is not None:
        print("❌ ERROR: Please specify either --dataset or --dataset_path, not both.")
        sys.exit(1)

    elif args.dataset is not None:
        loader = dataset_loader_map[args.dataset]
        df = loader(script_dir, visualize=False)
        dataset_basename = args.dataset

    elif args.dataset_path is not None:
        df = load_dataset(base_data_path, args.dataset_path)
        dataset_basename = os.path.basename(args.dataset_path).split('.')[0]

    else:
        print("❌ ERROR: Please specify either --dataset or --dataset_path.")
        sys.exit(1)

    if args.text_column not in df.columns:
        print(f"\n❌ ERROR: Text column '{args.text_column}' not found in dataset.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Track original shape for logging
    df_original_shape = (len(df), df.shape[1])

    # ============================================================
    # 5. Apply Sampling (Optional)
    # ============================================================
    if args.sample_fraction:
        df = DataProcessing.stratified_sample_dataset(
            df=df,
            sample_fraction=args.sample_fraction,
            stratify_cols=args.stratify_cols,
            random_state=args.seed,
            sampling_method=args.sampling_method
        )
        df_sampled_shape = (len(df), df.shape[1])
    else:
        df_sampled_shape = df_original_shape

    # ============================================================
    # 6. Setup Output Directory
    # ============================================================
    output_dir = os.path.join(
        base_data_path,
        "classification_results",
        dataset_basename,
        "extract_properties",
        args.task_name,
        f"seed{args.seed}",
        clean_model_name
    )
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "extracted_properties.csv")
    print(f"\nOutput Directory : {output_dir}")
    print(f"Results File     : {results_path}")

    # ============================================================
    # 7. Save Metadata
    # ============================================================
    metadata = {
        "timestamp":        pd.Timestamp.now().isoformat(),
        "dataset":          args.dataset or args.dataset_path,
        "dataset_basename": dataset_basename,
        "text_column":      args.text_column,
        "task_name":        args.task_name,
        "model_used":       args.model_name,
        "seed":             args.seed,
        "sample_fraction":  args.sample_fraction,
        "stratify_cols":    args.stratify_cols if args.sample_fraction else None,
        "sampling_method":  args.sampling_method if args.sample_fraction else None,
        "original_shape":   df_original_shape,
        "sampled_shape":    df_sampled_shape,
        "prompts": {
            "base_prompt":   base_prompt,
            "task":          task,
            "format_output": format_output
        }
    }

    metadata_path = os.path.join(output_dir, "extracted_properties_metadata.json")
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved to: {metadata_path}")
    else:
        print(f"Metadata already exists at: {metadata_path}")

    # ============================================================
    # 8. Create Experiment Log
    # ============================================================
    create_properties_experiment_log(
        args=args,
        model_name=args.model_name,
        output_dir=output_dir,
        df_original_shape=df_original_shape,
        df_sampled_shape=df_sampled_shape,
        base_prompt=base_prompt,
        task=task,
        format_output=format_output
    )

    # ============================================================
    # 9. Resume Check — Skip Already Processed Sentences
    # ============================================================
    df_to_process = get_remaining_data(df, results_path)

    # ============================================================
    # 10. Extract Properties
    # ============================================================
    if df_to_process.empty:
        print("\n✓ All sentences have already been processed!")
    else:
        extract_properties(
            df_to_process,
            args.text_column,
            base_prompt,
            task,
            format_output,
            model,
            results_path,
            dataset_basename,
            args.seed,
            stop_after=STOP_AFTER
        )

    # ============================================================
    # 11. Final Summary
    # ============================================================
    if os.path.exists(results_path):
        try:
            final_df = pd.read_csv(results_path)
            print("\n" + "="*50)
            print("FINAL RESULTS SUMMARY")
            print("="*50)

            summary = {
                "total_processed": len(final_df),
                "shape":           final_df.shape,
                "columns":         list(final_df.columns),
                "model_used":      list(final_df['Model Name'].unique()),
                "sample_results":  final_df[['Sentence', 'Source', 'Target', 'Date', 'Model Name']].head(3).to_dict('records') if not final_df.empty else []
            }

            print(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"Could not print summary: {e}")

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"✓ Experiment: {dataset_basename}")
    print(f"✓ Task: {args.task_name}")
    print(f"✓ Model: {args.model_name}")
    print(f"✓ Seed: {args.seed}")
    if args.sample_fraction:
        print(f"✓ Sample: {args.sample_fraction*100}% ({df_sampled_shape[0]} sentences)")
    print(f"✓ Results: {output_dir}")
    print()