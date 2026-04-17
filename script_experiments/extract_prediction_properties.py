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

    # Sample 300 rows for testing
    # df = df.sample(n=40, random_state=42).reset_index(drop=True)
    print(f"\n✓ Sampled dataset shape: {df.shape}")


    # Reset index so we have a clean 0, 1, 2, ... row numbers.
    # This is important for the resume logic later — we track which
    # row numbers have already been processed.
    df = df.reset_index(drop=True)

    print(f"Shape: {df.shape}")
    print(f"\nFirst 7 rows:\n{df.head(7)}\n")
    print(f"\nLast 7 rows:\n{df.tail(7)}\n")
    return df


def load_prompts_and_llms(model_names=None):
    """
    Build the base prompt and load the language model(s).

    The base prompt combines:
    - Who the model is (system identity)
    - What a prediction looks like (prediction properties)
    - Rules the model must follow (requirements)
    - Examples to guide the model (few-shot examples)

    Parameters
    ----------
    model_names : str or list of str, optional
        One or more model names to load. Defaults to 'llama-3.1-8b-instant'.

    Returns
    -------
    base_prompt : str
        The full prompt sent to the model before each sentence.
    task : str
        The labeling instruction for the model.
    format_output : str
        The expected JSON output format.
    models : list
        List of loaded model instances.
    """
    print("\n" + "="*50)
    print("STEP: LOAD PROMPTS & MODELS")
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

    # Load the model(s)
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
        raise ValueError("No models were successfully loaded.")

    print(f"\n✓ Total models loaded: {len(models)}\n")
    return base_prompt, task, format_output, models


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


def process_single_result(input_index, text, raw_response, model_name) -> pd.DataFrame:
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
        Input_Index, Sentence, Raw Response, Model Name,
        No Property, Source, Target, Date, Outcome.
    """
    # Start with an empty structure — we will fill in the properties below
    data = {
        'Input_Index': [input_index],
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

    # Try to parse the raw LLM response as a JSON dictionary
    parsed = DataProcessing.parse_llm_json_response(raw_response)

    if parsed and isinstance(parsed, dict):
        try:
            # The model returns keys 0-4 (sometimes as int, sometimes as string)
            # 0 = no property, 1 = source, 2 = target, 3 = date, 4 = outcome
            results_df.at[0, 'No Property'] = join_property(parsed.get(0, []) or parsed.get("0", []))
            results_df.at[0, 'Source']      = join_property(parsed.get(1, []) or parsed.get("1", []))
            results_df.at[0, 'Target']      = join_property(parsed.get(2, []) or parsed.get("2", []))
            results_df.at[0, 'Date']        = join_property(parsed.get(3, []) or parsed.get("3", []))
            results_df.at[0, 'Outcome']     = join_property(parsed.get(4, []) or parsed.get("4", []))
        except Exception as e:
            print(f"Error mapping JSON to columns for index {input_index}: {e}")
    else:
        # If we could not parse the response, mark it so we can review it later
        results_df.at[0, 'No Property'] = "PARSE_ERROR"

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
    prefix = os.path.basename(results_path).split('.')[0]  # e.g., "results"

    DataProcessing.save_to_file(
        data=batch_df,
        path=results_dir,
        prefix=prefix,
        save_file_type='csv',
        include_version=False,
        append=True
    )

def extract_properties(df, text_column, base_prompt, task, format_output, models, results_path, dataset_basename, stop_after=None):
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
    models : list
        List of loaded model instances.
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

        for model in models:
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
            time.sleep(7)

            if raw_response is None:
                raw_response = str({"0": ["ERROR_MAX_RETRIES"], "1": [], "2": [], "3": [], "4": []})

            single_df = process_single_result(idx, text, raw_response, model.__name__())
            single_df['Dataset Source'] = dataset_basename
            batch_results.append(single_df)

        sentences_processed += 1

        if len(batch_results) >= BATCH_SIZE:
            save_batch(batch_results, results_path)
            batch_results = []

    if batch_results:
        save_batch(batch_results, results_path)

    print(f"\n✓ Processing complete. Results saved to {results_path}\n")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("LABEL GROUND TRUTH — PREDICTION PROPERTY EXTRACTION")
    print("="*50)

    # ============================================================
    # 1. Configuration and Arguments
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    dataset_loader_map = {
        'predictions':          DataProcessing.load_predictions_dataset,
        'non_predictions':      DataProcessing.load_non_predictions_dataset,
        'financial_phrasebank': DataProcessing.load_financial_phrasebank_dataset,
        'chronicle2050':        DataProcessing.load_chronicle2050_dataset,
        'news_api':             DataProcessing.load_news_api_dataset,
        'yt':                   DataProcessing.load_yt_dataset
    }

    parser = argparse.ArgumentParser(description='Extract prediction properties from sentences using LLMs')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(dataset_loader_map.keys()),
        default='predictions',
        help='Dataset to process.'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['openai/gpt-oss-120b'],
        help='Model name(s) to use for extraction.'
    )
    parser.add_argument(
        '--text_column',
        type=str,
        default='Base Sentence',
        help='Column name containing the sentences to extract properties from.'
    )
    args = parser.parse_args()

    dataset_basename = args.dataset
    print(f"Selected dataset: {args.dataset}")

    # ============================================================
    # 2. Load Prompts and Models
    # ============================================================
    base_prompt, task, format_output, models = load_prompts_and_llms(args.models)

    # ============================================================
    # 3. Setup Output Directory and Metadata
    # ============================================================
    dataset_basename = os.path.basename(args.dataset).split('.')[0]
    cleaned_model_names = []
    for model in models:
        # Replace "/" with "_" to avoid creating nested folders
        # e.g., "openai/gpt-oss-120b" becomes "openai_gpt-oss-120b"
        clean_name = model.__name__().replace('/', '_')
        cleaned_model_names.append(clean_name)

    model_names_str = '-'.join(cleaned_model_names)

    # Output folder structure:
    # data/extract_prediction_properties/ground_truth/{dataset}/{model}/
    output_dir = os.path.join(
        base_data_path,
        "extract_prediction_properties",
        "ground_truth",
        dataset_basename,
        model_names_str
    )
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.csv")
    print(f"Output Directory: {output_dir}")
    print(f"Results File: {results_path}")

    # Save a metadata file so we can always trace back what settings were used
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
        print(f"✓ Metadata saved to: {metadata_path}")
    else:
        print(f"Metadata already exists at: {metadata_path}")

    # ============================================================
    # 4. Load Dataset
    # ============================================================
    loader = dataset_loader_map[args.dataset]
    df = loader(script_dir)
    dataset_basename = args.dataset

    # Sample 300 rows for testing
    # df = df.sample(n=300, random_state=42).reset_index(drop=True)
    print(f"\n✓ Sampled dataset shape: {df.shape}")

    if args.text_column not in df.columns:
        print(f"\n❌ ERROR: Text column '{args.text_column}' not found in dataset.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # ============================================================
    # 5. Resume Check — Skip Already Processed Sentences
    # ============================================================
    df_to_process = get_remaining_data(df, results_path)

    # ============================================================
    # 6. Extract Properties
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
            models,
            results_path,
            dataset_basename,
            stop_after=STOP_AFTER
        )

    # ============================================================
    # 7. Final Summary
    # ============================================================
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