# Tense Classification
import os
import sys
import ast
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List
from datetime import datetime
from collections import defaultdict

# Add project modules to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction


def load_dataset(base_data_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    base_data_path : str
        Root path to all data.
    dataset_name : str
        Full path to the dataset file.

    Notes
    -----
    Expects a CSV with at least: Sentence, POS Label, Detailed POS Label,
    and a column named 'N Sentences' which holds the true sentence count.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)

    data_path = os.path.join(base_data_path, dataset_name)
    print(f"Data path: {data_path}")

    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head(10)}\n")

    # Validate that the expected column exists before doing anything
    if 'N Sentences' not in df.columns:
        raise ValueError("Input file must contain a column named 'N Sentences'.")

    return df


def classify_tense_per_sentence(df: pd.DataFrame, pos_feature: str = 'AUX', detailed_pos_feature: str = 'MD') -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Token-level POS features dataframe, one row per token, with empty
        rows separating each sentence.
    pos_feature : str, default 'AUX'
        The POS label to look for (e.g., 'AUX').
    detailed_pos_feature : str, default 'MD'
        The fine-grained POS tag to look for (e.g., 'MD' for modals).

    Notes
    -----
    The input data is token-level, but tense is a sentence-level label.
    We group tokens back into sentences using the empty-row boundaries
    that extract_pos_features() inserts after each sentence.
    Rule: if ANY token in the sentence has POS == pos_feature AND
    Detailed POS == detailed_pos_feature, classify as Future (1),
    otherwise Non-Future (0).

    Returns
    -------
    pd.DataFrame
        One row per sentence with columns: Sentence, Future Tense.
    """
    print("\n" + "="*50)
    print("CLASSIFYING TENSE PER WORD")
    print("Future: AUX and MD")
    print("="*50)

    results = []

    # -----------------------------------------------------------------
    # Group tokens back into sentence-level blocks using the empty-row
    # boundaries that extract_pos_features() inserts after each sentence.
    # An empty row is identified by an empty string in 'Sentence'.
    # -----------------------------------------------------------------
    current_sentence_tokens = []
    current_sentence_text = ""

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying words", unit="row"):

        # Empty row signals end of current sentence block
        if row['Sentence'] == "" or pd.isna(row['Sentence']):
            if current_sentence_tokens:
                # -------------------------------------------------------
                # Apply the classification rule at the sentence level:
                # future = 1 if any token matches AUX + MD, else 0
                # -------------------------------------------------------
                is_future = any(
                    t['POS Label'] == pos_feature and t['Detailed POS Label'] == detailed_pos_feature
                    for t in current_sentence_tokens
                )
                results.append({
                    "Sentence": current_sentence_text,
                    "Future Tense": 1 if is_future else 0,
                })
                # Reset for the next sentence
                current_sentence_tokens = []
                current_sentence_text = ""
        else:
            # Accumulate tokens belonging to the current sentence
            current_sentence_text = row['Sentence']  # Same for all tokens in sentence
            current_sentence_tokens.append({
                "POS Label": row['POS Label'],
                "Detailed POS Label": row['Detailed POS Label'],
            })

    # Handle any trailing sentence that didn't end with an empty row
    if current_sentence_tokens:
        is_future = any(
            t['POS Label'] == pos_feature and t['Detailed POS Label'] == detailed_pos_feature
            for t in current_sentence_tokens
        )
        results.append({
            "Sentence": current_sentence_text,
            "Future Tense": 1 if is_future else 0,
        })

    results_df = pd.DataFrame(results)

    print(f"Shape: {results_df.shape}")
    print(f"Future tense sentences: {results_df['Future Tense'].sum()}")
    print(f"Non-future sentences: {(results_df['Future Tense'] == 0).sum()}")
    print(f"\nPreview:\n{results_df.head(10)}\n")

    return results_df


def validate_sentence_count(results_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """
    Parameters
    ----------
    results_df : pd.DataFrame
        Sentence-level classification output.
    df : pd.DataFrame
        Original token-level input dataframe containing 'N Sentences'.

    Notes
    -----
    Compares the number of classified sentences against the expected
    sentence count stored in the 'N Sentences' column of the input file.
    Raises an error if they do not match so the issue is caught early.

    Returns
    -------
    None
    """
    # The 'N Sentences' column should be constant across all rows (it's a
    # property of the file, not a per-token value), so we take the first value.
    expected_n_sentences = int(df['N Sentences'].iloc[0])
    actual_n_sentences = len(results_df)

    print("\n" + "="*50)
    print("VALIDATING SENTENCE COUNT")
    print("="*50)
    print(f"Expected sentences (N Sentences column): {expected_n_sentences}")
    print(f"Actual sentences classified:             {actual_n_sentences}")

    if expected_n_sentences != actual_n_sentences:
        raise ValueError(
            f"Sentence count mismatch: expected {expected_n_sentences} "
            f"but classified {actual_n_sentences}. "
            f"Check that the input file is not corrupted or truncated."
        )

    print("✓ Sentence count matches.\n")


if __name__ == "__main__":
    """Usage
    python3 tense_classification.py \
        --dataset /path/to/pos_features-v1.csv \
        --dataset_name synthetic_batch_1 \
        --pos_feature AUX \
        --detailed_pos_feature MD
    """

    print("\n" + "="*50)
    print("TENSE CLASSIFICATION")
    print("="*50)

    # ============================================================
    # 1. CONFIGURATION & ARGS
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    # Default batch index – change as needed
    batch_idx = 1
    default_dataset = os.path.join(
        base_data_path,
        f'tense_extraction/synthetic_batch_{batch_idx}/',
        'pos_features-v1.csv'
    )

    parser = argparse.ArgumentParser(description='Classify sentence tense using rule-based POS features.')

    # Path to the POS features CSV (explicit – no auto-discovery of latest version)
    parser.add_argument('--dataset', default=default_dataset,
                        help='Full path to the POS features CSV file.')

    # Name used for organising the output folder
    parser.add_argument('--dataset_name', type=str, default=f"synthetic_batch_{batch_idx}",
                        help='Name of the dataset. Used for naming the output folder.')

    # POS feature to match (coarse tag) – explicit so every run is reproducible
    parser.add_argument('--pos_feature', type=str, default='AUX',
                        help='Coarse POS label to use for classification rule (e.g., AUX).')

    # Fine-grained POS tag to match – explicit for the same reason
    parser.add_argument('--detailed_pos_feature', type=str, default='MD',
                        help='Fine-grained POS tag to use for classification rule (e.g., MD).')

    args = parser.parse_args()

    # ============================================================
    # 2. LOAD DATASET
    # ============================================================
    df = load_dataset(base_data_path, args.dataset)

    # ============================================================
    # 3. CLASSIFY TENSE – sentence level
    # Rule: sentence contains AUX + MD → Future (1), else Non-Future (0)
    # ============================================================
    results_df = classify_tense_per_sentence(
        df,
        pos_feature=args.pos_feature,
        detailed_pos_feature=args.detailed_pos_feature,
    )

    # Tag each row with the source file so we can trace back later
    results_df['Source Dataset'] = args.dataset_name

    # ============================================================
    # 4. VALIDATE – classified sentence count must match N Sentences
    # ============================================================
    validate_sentence_count(results_df, df)

    # ============================================================
    # 5. SAVE TENSE CLASSIFICATION RESULTS
    # ============================================================
    print("\n" + "="*50)
    print("SAVING TENSE CLASSIFICATION RESULTS")
    print("="*50)

    # Save under rule_based method, organised by dataset name
    save_path = os.path.join(
        base_data_path,
        'tense_classification',
        'rule_based',
        f'{args.dataset_name}'
    )

    # Save sentence-level classifications
    DataProcessing.save_to_file(
        data=results_df,
        path=save_path,
        prefix='classifications',
        save_file_type='csv',
        include_version=True
    )

    # Save metadata so every run is fully reproducible
    classification_metadata = {
        "dataset_name": args.dataset_name,
        "source_features_path": args.dataset,
        "method": "rule_based",
        "classification_rule": f"POS=={args.pos_feature} AND Detailed POS=={args.detailed_pos_feature}",
        "pos_feature": args.pos_feature,
        "detailed_pos_feature": args.detailed_pos_feature,
        "total_token_rows_input": len(df),                      # Token-level rows in
        "total_sentences_classified": len(results_df),          # Sentence-level rows out
        "future_tense_count": int(results_df['Future Tense'].sum()),
        "non_future_tense_count": int((results_df['Future Tense'] == 0).sum()),
        "expected_sentence_count": int(df['N Sentences'].iloc[0]),
        "columns": list(results_df.columns),
        "classification_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    DataProcessing.save_to_file(
        data=classification_metadata,
        path=save_path,
        prefix='classification_metadata',
        save_file_type='json',
        include_version=False
    )

    print("\n✓ Tense classification complete!")
    print(f"Files saved to: {save_path}")
    print(f"Future tense sentences    : {classification_metadata['future_tense_count']}")
    print(f"Non-future tense sentences: {classification_metadata['non_future_tense_count']}")
    print(f"Total sentences           : {classification_metadata['total_sentences_classified']}")