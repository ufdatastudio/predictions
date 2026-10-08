# llm-experiment.py

import os
import sys
import ast
import json
import re
import time
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List

# Add project modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

from prompts import EntityExtractionPrompt
from data_processing import DataProcessing
from tolsa_properties import Tolsa
from text_generation_models import TextGenerationModelFactory


# How many sentence results to collect in memory before writing to disk
BATCH_SIZE = 10

# Stop after this many sentences (set to None to process all)
STOP_AFTER = None


def load_dataset(base_data_path, dataset_name):
    """
    Load a dataset from a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    base_data_path : str
        The root data directory path.
    dataset_name : str
        The relative path to the dataset CSV or Excel file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset with a clean 0..N index.
    """
    print("\n" + "=" * 50)
    print("STEP: LOAD DATASET")
    print("=" * 50)

    data_path = os.path.join(base_data_path, dataset_name)

    print(f"Dataset path: {dataset_name}")

    suffix = Path(data_path).suffix.lower()

    if suffix == ".csv":
        df = DataProcessing.load_from_file(
            data_path,
            file_type='csv'
        )

    elif suffix in [".xlsx", ".xls"]:
        raw_df = DataProcessing.load_from_file(
            data_path,
            file_type='xlsx',
            header=None
        )

        header_row = None

        for idx, row in raw_df.iterrows():
            if row.astype(str).str.strip().eq("Base Sentence").any():
                header_row = idx
                break

        if header_row is None:
            raise ValueError("Could not find 'Base Sentence' header row.")

        df = DataProcessing.load_from_file(
            data_path,
            file_type='xlsx',
            header=header_row
        )

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if "Base Sentence" not in df.columns:
        raise ValueError(
            f"'Base Sentence' column not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep Base Sentence plus metadata columns used for sampling and row-level dataset name
    keep_cols = [
        "Base Sentence"
    ] + [
        c for c in ["Dataset Name", "Ground Truth"]
        if c in df.columns
    ]

    df = df[keep_cols].copy()

    # Remove empty rows
    df = df.dropna(subset=["Base Sentence"])

    df = df.reset_index(drop=True)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 7 rows:\n{df.head(7)}\n")
    print(f"\nLast 7 rows:\n{df.tail(7)}\n")

    return df


def load_prompts_and_llm(model_name=None, prompt_type='few-shot'):
    """
    Build the base prompt and load a single language model.

    Parameters
    ----------
    model_name : str, optional
        Model name to load. Defaults to 'llama-3.1-8b-instant'.

    prompt_type : str, optional
        Prompting strategy: 'zero-shot', 'few-shot', or
        'chain-of-thought'. Default is 'few-shot'.

    Returns
    -------
    tuple
        Base prompt, task, output format, and model.
    """
    print("\n" + "=" * 50)
    print(f"STEP: LOAD PROMPTS & MODEL ({prompt_type})")
    print("=" * 50)

    tolsa_properties, linguistic_cues = (
        Tolsa.get_tolsa_properties_and_linguistic_cues()
    )

    prompt = EntityExtractionPrompt(
        prompt_type_name=prompt_type
    )

    if prompt_type == 'zero-shot':
        system_identity, task, format_output = prompt.zero_shot()
        examples_text = ""

    elif prompt_type == 'few-shot':
        system_identity, task, format_output, examples = prompt.few_shot()
        examples_text = f"Examples:\n{examples}"

    elif prompt_type == 'chain-of-thought':
        system_identity, task, format_output, steps = (
            prompt.chain_of_thought()
        )
        examples_text = f"Steps:\n{steps}"

    else:
        raise ValueError(
            f"Unknown prompt_type: '{prompt_type}'. "
            f"Choose from: 'zero-shot', 'few-shot', 'chain-of-thought'"
        )

    base_prompt = f"""{system_identity}

{tolsa_properties}

{linguistic_cues}

{examples_text}
"""

    print("\n--- Base Prompt ---")
    print(base_prompt)
    print("--- End Base Prompt ---\n")

    print("✓ Prompts loaded")

    if model_name is None:
        model_name = 'llama-3.1-8b-instant'

    tgmf = TextGenerationModelFactory()

    try:
        model = tgmf.create_instance(
            model_name=model_name
        )
        print(f"✓ Loaded: {model.__name__()}")

    except ValueError as e:
        raise ValueError(
            f"✗ Failed to load {model_name}: {e}"
        )

    print(f"\n✓ Model loaded: {model.__name__()}\n")

    return base_prompt, task, format_output, model


def get_remaining_data(df, results_path):
    """
    Filter the dataset to only include sentences that have NOT
    been processed yet.

    This allows the pipeline to resume from where it left off if
    it was interrupted.
    """
    if not os.path.exists(results_path):
        print("No existing results found. Starting from scratch.")
        return df

    try:
        # Load only the Input_Index column to save memory.
        # Input_Index tracks which row numbers have already been processed.
        existing_df = DataProcessing.load_from_file(
            results_path,
            file_type='csv',
            usecols=['Input_Index']
        )

        processed_indices = set(
            existing_df['Input_Index'].unique()
        )

        print(
            f"Found {len(processed_indices)} already processed sentences."
        )

        # Keep only rows whose index is NOT in the processed set
        df_remaining = df[
            ~df.index.isin(processed_indices)
        ]

        print(
            f"Resuming. {len(df_remaining)} sentences remaining."
        )

        return df_remaining

    except ValueError:
        # If Input_Index column does not exist, fall back to row counting
        print(
            "Warning: 'Input_Index' column not found. "
            "Falling back to row counting."
        )

        lines = DataProcessing.load_from_file(
            results_path,
            file_type='txt'
        )

        row_count = len(lines) - 1

        return df.iloc[max(0, row_count):]

    except Exception as e:
        print(
            f"Error reading existing results file: {e}. "
            "Starting from scratch."
        )

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
    """
    if isinstance(values, list):
        return '|'.join([str(v).strip() for v in values if v])

    if isinstance(values, str):
        return values.strip()

    return ''


def process_single_result(
        input_index,
        text,
        raw_response,
        model_name,
        seed,
        prompt_type,
        task_name
    ) -> pd.DataFrame:
    """
    Convert one LLM slot-filling response into a structured DataFrame row.

    Parse Status values:
    - OK: Complete JSON/dictionary parsed successfully.
    - PARTIAL_PARSE: Some slots recovered from malformed or truncated output.
    - PARSE_ERROR: No slots could be recovered.
    """
    data = {
        'Task Name':     [task_name],
        'Prompt Type':   [prompt_type],
        'Seed':          [seed],
        'Input_Index':   [input_index],
        'Base Sentence': [text],
        'Raw Response':  [raw_response],
        'Model Name':    [model_name],
        'Parse Status':  [''],
        'Source':        [''],
        'Target':        [''],
        'Date':          [''],
        'Outcome':       ['']
    }

    results_df = pd.DataFrame(data)

    parsed, parse_status = (
        DataProcessing.parse_slot_filling_response(
            raw_response
        )
    )

    try:
        results_df.at[0, 'Source'] = join_property(
            parsed.get("1", [])
        )

        results_df.at[0, 'Target'] = join_property(
            parsed.get("2", [])
        )

        results_df.at[0, 'Date'] = join_property(
            parsed.get("3", [])
        )

        results_df.at[0, 'Outcome'] = join_property(
            parsed.get("4", [])
        )

        results_df.at[0, 'Parse Status'] = parse_status

    except Exception as e:
        print(
            f"Error mapping JSON to columns for index "
            f"{input_index}: {e}"
        )

        results_df.at[0, 'Parse Status'] = 'PARSE_ERROR'

    return results_df


def save_batch(batch_dfs, results_path):
    """
    Write a list of single-row DataFrames to the results CSV file.

    Instead of writing to disk after every sentence, collect
    BATCH_SIZE rows in memory and flush them all at once.
    """
    if not batch_dfs:
        return

    batch_df = pd.concat(
        batch_dfs,
        ignore_index=True
    )

    results_dir = os.path.dirname(results_path)

    prefix = os.path.basename(
        results_path
    ).split('.')[0]

    DataProcessing.save_to_file(
        data=batch_df,
        path=results_dir,
        prefix=prefix,
        save_file_type='csv',
        include_version=False,
        append=True
    )


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
        prompt_type,
        task_name,
        sleep_seconds=7,
        stop_after=None):
    """
    Process sentences with batch saving and robust error handling.
    """
    print("\n" + "=" * 50)
    print("STEP: EXTRACT PROPERTIES")
    print("=" * 50)

    print(f"Sentences to process: {len(df)}")

    if stop_after:
        print(
            f"⚠️  STOP_AFTER={stop_after}: "
            f"Will stop after {stop_after} sentences for testing."
        )

    batch_results = []
    sentences_processed = 0

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing"
    ):
        # Stop early if stop_after is set
        if (
            stop_after is not None
            and sentences_processed >= stop_after
        ):
            print(
                f"\n⚠️  Reached STOP_AFTER={stop_after}. "
                "Stopping early."
            )
            break

        text = row[text_column]

        prompt = f"""{base_prompt}

<text_document>{text}</text_document>

{task}

{format_output}"""

        # Print the first prompt of the run once for a sanity check
        if sentences_processed == 0:
            print(
                f"\n--- Sample Prompt (idx={idx}) ---\n"
                f"{prompt}\n"
                f"--- End Sample Prompt ---\n"
            )

        input_prompt = model.user(prompt)

        raw_response = model.safe_chat_completion(
            [input_prompt],
            idx=idx
        )

        # Proactive sleep to stay within provider rate limits
        time.sleep(sleep_seconds)

        if raw_response is None:
            raw_response = "ERROR_MAX_RETRIES"

        single_df = process_single_result(
            idx,
            text,
            raw_response,
            model.__name__(),
            seed,
            prompt_type,
            task_name
        )

        # Preserve row-level dataset identity from the source dataframe.
        if 'Dataset Name' in row.index:
            single_df['Dataset Name'] = row['Dataset Name']
        else:
            single_df['Dataset Name'] = dataset_basename

        batch_results.append(single_df)

        sentences_processed += 1

        if len(batch_results) >= BATCH_SIZE:
            save_batch(
                batch_results,
                results_path
            )

            batch_results = []

    if batch_results:
        save_batch(
            batch_results,
            results_path
        )

    print(
        f"\n✓ Processing complete. "
        f"Results saved to {results_path}\n"
    )


if __name__ == "__main__":
    """
    Usage:

        Be sure to run: source .venv_predictions/bin/activate

        # ============================================================
        # STEP 1: Create combined dataset (run once)
        # ============================================================

        python3 create_combined_dataset.py \\
            --datasets synthetic financial_phrasebank chronicle2050 timebank yt news_api mf_climate \\
            --predictions_only_datasets yt news_api mf_climate \\
            --output_name properties_july_2026 \\
            --no_version

        # ============================================================
        # STEP 2: Extract properties for ground truth
        # ============================================================

        python3 llm-experiment.py \\
            --dataset_path combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \\
            --model_name "llama-3.1-8b-instant" \\
            --task_name ground_truth \\
            --prompt_type few-shot \\
            --sample_fraction 0.1 \\
            --seed 7

        # ============================================================
        # STEP 3: Test LLM extraction ability
        # ============================================================

        python3 llm-experiment.py \\
            --dataset_path combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \\
            --model_name "llama-3.1-8b-instant" \\
            --task_name extraction \\
            --prompt_type few-shot \\
            --sample_fraction 0.1 \\
            --seed 7

        # ============================================================
        # STEP 4: Zero-shot extraction
        # ============================================================

        python3 llm-experiment.py \\
            --dataset_path extract_tolsa_properties_results/naacl_2026_submission/ground_truth/extracted_properties-ground_truth_only.csv \\
            --model_name "openai/gpt-oss-20b" \\
            --task_name extraction \\
            --prompt_type zero-shot \\
            --seed 7

        # ============================================================
        # STEP 5: Full zero-shot experiment
        # ============================================================

        python3 llm-experiment.py \\
            --dataset_path extract_tolsa_properties_results/naacl_2026_submission/ground_truth/extracted_properties-ground_truth_only.csv \\
            --model_name "openai/gpt-oss-120b" \\
            --task_name extraction \\
            --prompt_type zero-shot \\
            --seed 3

        python3 llm-experiment.py \\
            --dataset_path extract_tolsa_properties_results/naacl_2026_submission/ground_truth/extracted_properties-ground_truth_only.csv \\
            --model_name "openai/gpt-oss-120b" \\
            --task_name extraction \\
            --prompt_type zero-shot \\
            --seed 7

        python3 llm-experiment.py \\
            --dataset_path extract_tolsa_properties_results/naacl_2026_submission/ground_truth/extracted_properties-ground_truth_only.csv \\
            --model_name "openai/gpt-oss-120b" \\
            --task_name extraction \\
            --prompt_type zero-shot \\
            --seed 33
    """

    print("\n" + "=" * 50)
    print("SENTENCE PROPERTY EXTRACTION")
    print("=" * 50)

    # ============================================================
    # 1. Configuration and Arguments
    # ============================================================

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    base_data_path = (
        DataProcessing.load_base_data_path(script_dir)
    )

    dataset_loader_map = {
        'synthetic':
            DataProcessing.load_synthetic_dataset,

        'fin_phrasebank':
            DataProcessing.load_financial_phrasebank_dataset,

        'chronicle2050':
            DataProcessing.load_chronicle2050_dataset,

        'news_api':
            DataProcessing.load_news_api_dataset,

        'yt':
            DataProcessing.load_yt_dataset,

        'timebank':
            DataProcessing.load_timebank_dataset,

        'mf_climate':
            DataProcessing.load_mf_climate_dataset,

        'clients_rivals_rouges':
            DataProcessing.load_clients_rivals_rouges_dataset,

        'forecast_bench':
            DataProcessing.load_forecast_bench_dataset,

        'smart_hospitals':
            DataProcessing.load_smart_hospitals_dataset
    }

    task_name_map = {
        'ground_truth': 'ground_truth',
        'extraction':   'extraction'
    }

    parser = argparse.ArgumentParser(
        description='Extract properties from sentences using LLMs.'
    )

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
        help='Either ground_truth for establishing labels or extraction for testing extraction.'
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

    parser.add_argument(
        '--prompt_type',
        type=str,
        choices=[
            'zero-shot',
            'few-shot',
            'chain-of-thought'
        ],
        default='few-shot',
        help='Prompting strategy for slot filling extraction. Default: few-shot'
    )

    parser.add_argument(
        '--sleep_seconds',
        type=float,
        default=7.0,
        help='Pause between API calls for rate limits. Use 0 if the provider has no TPM limit.'
    )

    args = parser.parse_args()

    print(f"Task       : {args.task_name}")
    print(f"Model      : {args.model_name}")
    print(f"Dataset    : {args.dataset or args.dataset_path}")
    print(f"Seed       : {args.seed}")
    print(f"Prompt Type: {args.prompt_type}")

    # ============================================================
    # 2. Load Prompts and Model
    # ============================================================

    base_prompt, task, format_output, model = (
        load_prompts_and_llm(
            model_name=args.model_name,
            prompt_type=args.prompt_type
        )
    )

    # ============================================================
    # 3. Setup Model Name for Output Directory
    # ============================================================

    clean_model_name = args.model_name.replace(
        '/',
        '_'
    )

    clean_prompt_type = re.sub(
        r'[^A-Za-z0-9_.-]+',
        '_',
        args.prompt_type
    )

    # ============================================================
    # 4. Load Dataset
    # ============================================================

    if (
        args.dataset is not None
        and args.dataset_path is not None
    ):
        print(
            "❌ ERROR: Please specify either "
            "--dataset or --dataset_path, not both."
        )
        sys.exit(1)

    elif args.dataset is not None:
        loader = dataset_loader_map[args.dataset]

        df = loader(
            script_dir,
            visualize=False
        )

        dataset_basename = args.dataset

    elif args.dataset_path is not None:
        df = load_dataset(
            base_data_path,
            args.dataset_path
        )

        dataset_basename = (
            os.path.basename(
                args.dataset_path
            ).split('.')[0]
        )

    else:
        print(
            "❌ ERROR: Please specify either "
            "--dataset or --dataset_path."
        )
        sys.exit(1)

    if args.text_column not in df.columns:
        print(
            f"\n❌ ERROR: Text column "
            f"'{args.text_column}' not found in dataset."
        )

        print(
            f"Available columns: {list(df.columns)}"
        )

        sys.exit(1)

    # Track original shape for logging
    df_original_shape = (
        len(df),
        df.shape[1]
    )

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

        df_sampled_shape = (
            len(df),
            df.shape[1]
        )

    else:
        df_sampled_shape = df_original_shape

    # ============================================================
    # 6. Setup Output Directory
    # ============================================================

    from datetime import datetime

    today_date = datetime.today().strftime(
        "%Y-%m-%d"
    )

    output_dir = os.path.join(
        base_data_path,
        "extract_tolsa_properties_results",
        "naacl_2026_submission",
        f"naacl_2026_results_{today_date}",
        f"seed{args.seed}",
        "in_domain",
        clean_model_name,
        clean_prompt_type
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    results_path = os.path.join(
        output_dir,
        "extracted_properties.csv"
    )

    print(
        f"\nOutput Directory : {output_dir}"
    )

    print(
        f"Results File     : {results_path}"
    )

    # ============================================================
    # 7. Save Metadata
    # ============================================================

    metadata = {
        "timestamp":
            pd.Timestamp.now().isoformat(),

        "dataset":
            args.dataset or args.dataset_path,

        "dataset_basename":
            dataset_basename,

        "text_column":
            args.text_column,

        "task_name":
            args.task_name,

        "model_used":
            args.model_name,

        "seed":
            args.seed,

        "sample_fraction":
            args.sample_fraction,

        "stratify_cols":
            args.stratify_cols
            if args.sample_fraction
            else None,

        "sampling_method":
            args.sampling_method
            if args.sample_fraction
            else None,

        "original_shape":
            df_original_shape,

        "sampled_shape":
            df_sampled_shape,

        "prompt_type":
            args.prompt_type,

        "batch_size":
            BATCH_SIZE,

        "stop_after":
            STOP_AFTER,

        "sleep_seconds":
            args.sleep_seconds,

        "prompts": {
            "base_prompt":
                base_prompt,

            "task":
                task,

            "format_output":
                format_output
        }
    }

    metadata_path = os.path.join(
        output_dir,
        "experiment_metadata.json"
    )

    if not os.path.exists(metadata_path):
        DataProcessing.save_to_file(
            data=metadata,
            path=output_dir,
            prefix="experiment_metadata",
            save_file_type='json',
            include_version=False
        )

        print(
            f"✓ Metadata saved to: {metadata_path}"
        )

    else:
        print(
            f"Metadata already exists at: {metadata_path}"
        )

    # ============================================================
    # 8. Resume Check — Skip Already Processed Sentences
    # ============================================================

    df_to_process = get_remaining_data(
        df,
        results_path
    )

    # ============================================================
    # 9. Extract Properties
    # ============================================================

    if df_to_process.empty:
        print(
            "\n✓ All sentences have already been processed!"
        )

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
            prompt_type=args.prompt_type,
            task_name=args.task_name,
            sleep_seconds=args.sleep_seconds,
            stop_after=STOP_AFTER
        )

    # ============================================================
    # 10. Final Summary
    # ============================================================

    if os.path.exists(results_path):
        try:
            final_df = DataProcessing.load_from_file(
                results_path,
                file_type='csv'
            )

            print("\n" + "=" * 50)
            print("FINAL RESULTS SUMMARY")
            print("=" * 50)

            parse_status = final_df.get(
                'Parse Status',
                pd.Series()
            )

            parse_error_count = int(
                (parse_status == 'PARSE_ERROR').sum()
            )

            parse_error_rate = round(
                parse_error_count /
                max(len(final_df), 1),
                4
            )

            summary = {
                "total_processed":
                    len(final_df),

                "shape":
                    final_df.shape,

                "columns":
                    list(final_df.columns),

                "model_used":
                    list(final_df['Model Name'].unique()),

                "prompt_type":
                    list(final_df['Prompt Type'].unique()),

                "parse_error_count":
                    parse_error_count,

                "parse_error_rate":
                    parse_error_rate,

                "sample_results":
                    final_df[
                        [
                            'Base Sentence',
                            'Source',
                            'Target',
                            'Date',
                            'Outcome',
                            'Model Name',
                            'Prompt Type'
                        ]
                    ].head(3).to_dict('records')
                    if not final_df.empty
                    else []
            }

            print(
                json.dumps(
                    summary,
                    indent=2
                )
            )

        except Exception as e:
            print(
                f"Could not print summary: {e}"
            )

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    print(
        f"✓ Experiment: {dataset_basename}"
    )

    print(
        f"✓ Task: {args.task_name}"
    )

    print(
        f"✓ Model: {args.model_name}"
    )

    print(
        f"✓ Seed: {args.seed}"
    )

    print(
        f"✓ Prompt Type: {args.prompt_type}"
    )

    if args.sample_fraction:
        print(
            f"✓ Sample: "
            f"{args.sample_fraction * 100}% "
            f"({df_sampled_shape[0]} sentences)"
        )

    print(
        f"✓ Results: {output_dir}"
    )

    print()