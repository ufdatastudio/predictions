# Tense Classification
import os
import sys
import ast
import json
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from datetime import datetime
from collections import defaultdict

# Add project modules to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing
from data_visualizing import DataPlotting


def load_dataset(base_data_path, dataset_name):
    """
    Parameters
    ----------
    base_data_path : str
        Root path to all data.
    dataset_path : str
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
    print(f"Dataset path: {dataset_name}")

    # Assuming DataProcessing handles file not found errors internally
    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head()}\n")

    # Validate that the expected sentence count column exists
    if 'N Sentences' not in df.columns:
        raise ValueError("Input file must contain a column named 'N Sentences'.")

    return df


def classify_tense_per_sentence(
    df: pd.DataFrame,
    pos_feature: str = 'AUX',
    detailed_pos_feature: str = 'MD',
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Token-level POS features dataframe, one row per token, with empty
        rows separating each sentence.
    pos_feature : str, default 'AUX'
        The coarse POS label to look for.
    detailed_pos_feature : str, default 'MD'
        The fine-grained POS tag to look for.
    Notes
    -----
    Tense is a sentence-level label, but the input data is token-level.
    We group tokens back into sentences using the empty-row boundaries
    that extract_pos_features() inserts after each sentence.
    Rule: if ANY token in the sentence has POS == pos_feature AND
    Detailed POS == detailed_pos_feature → Prediction (1), else Non-Prediction (0).
    Returns
    -------
    pd.DataFrame
        One row per sentence with columns: Sentence, Future Tense.
    """
    print("\n" + "="*50)
    print("CLASSIFYING TENSE PER SENTENCE")
    print("="*50)

    results = []

    # Accumulate tokens per sentence, then classify at sentence boundary
    current_sentence_tokens = []
    current_sentence_text = ""

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying words", unit="row"):

        # Empty row signals end of the current sentence block
        if row['Sentence'] == "" or pd.isna(row['Sentence']):
            if current_sentence_tokens:
                # Apply rule: AUX + MD anywhere in sentence → Prediction
                is_prediction = any(
                    t['POS Label'] == pos_feature and t['Detailed POS Label'] == detailed_pos_feature
                    for t in current_sentence_tokens
                )
                results.append({
                    "Sentence": current_sentence_text,
                    "Future Tense": 1 if is_prediction else 0,
                })
                # Reset for the next sentence
                current_sentence_tokens = []
                current_sentence_text = ""
        else:
            # Accumulate tokens belonging to the current sentence
            current_sentence_text = row['Sentence']
            current_sentence_tokens.append({
                "POS Label": row['POS Label'],
                "Detailed POS Label": row['Detailed POS Label'],
            })

    # Handle any trailing sentence that didn't end with an empty row
    if current_sentence_tokens:
        is_prediction = any(
            t['POS Label'] == pos_feature and t['Detailed POS Label'] == detailed_pos_feature
            for t in current_sentence_tokens
        )
        results.append({
            "Sentence": current_sentence_text,
            "Future Tense": 1 if is_prediction else 0,
        })

    results_df = pd.DataFrame(results)

    print(f"Shape: {results_df.shape}")
    print(f"Prediction sentences    : {results_df['Future Tense'].sum()}")
    print(f"Non-Prediction sentences: {(results_df['Future Tense'] == 0).sum()}")
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
    Raises an error if the classified sentence count does not match
    the expected count stored in 'N Sentences'.
    Returns
    -------
    None
    """
    # 'N Sentences' is a file-level property – the same value on every row
    expected_n_sentences = int(df['N Sentences'].iloc[0])
    actual_n_sentences = len(results_df)

    print("\n" + "="*50)
    print("VALIDATING SENTENCE COUNT")
    print("="*50)
    print(f"Expected sentences (N Sentences column): {expected_n_sentences}")
    print(f"Actual sentences classified            : {actual_n_sentences}")

    if expected_n_sentences != actual_n_sentences:
        raise ValueError(
            f"Sentence count mismatch: expected {expected_n_sentences} "
            f"but classified {actual_n_sentences}. "
            f"Check that the input file is not corrupted or truncated."
        )

    print("✓ Sentence count matches.\n")


def visualize_tense_distribution(
    results_df: pd.DataFrame,
    save_path: str,
    dataset_name: str,
    show: bool = False,
) -> None:
    """
    Parameters
    ----------
    results_df : pd.DataFrame
        Sentence-level classification output with a 'Future Tense' column.
    save_path : str
        Directory path where the PNG will be saved.
    dataset_name : str
        Used in the chart title so it's clear which dataset is shown.
    show : bool, default False
        If True, display the interactive pop-up window.
        If False, only save the image (safe for bash/headless runs).
    Notes
    -----
    Always saves the image. Pop-up window is optional via --visualize flag.
    Uses the non-interactive 'Agg' backend when show=False to prevent
    the plot window from blocking the bash script.
    Returns
    -------
    None
    """
    print("\n" + "="*50)
    print("VISUALIZING TENSE DISTRIBUTION")
    print("="*50)

    # Switch to non-interactive backend when not showing the window –
    # this prevents the plot from blocking the bash script
    if not show:
        matplotlib.use('Agg')

    # Draw the bar chart
    DataPlotting.plot_class_distribution(
        df=results_df,
        label_col='Future Tense',
        class_names=['Non-Prediction', 'Prediction'],
        title=f'Tense Distribution – {dataset_name}',
        save_path=None,     # We handle saving below explicitly
    )

    # Always save the image regardless of show flag
    DataProcessing.save_to_file(
        data=None,
        path=save_path,
        prefix='tense_distribution',
        save_file_type='png',
        include_version=True,
        dpi=300,
        bbox_inches='tight',
    )

    # Only show the interactive window if explicitly requested
    if show:
        plt.show()
    else:
        plt.close()

    print(f"✓ Tense distribution plot saved to: {save_path}\n")


if __name__ == "__main__":
    """Usage
    # Save image only (default – safe for bash)
    python3 tense_classification.py \
        --dataset tense_extraction/synthetic_combined/pos_features-v1.csv \
        --dataset_name synthetic_combined \
        --pos_feature AUX \
        --detailed_pos_feature MD

    # Save image AND show pop-up window
    python3 tense_classification.py \
        --dataset tense_extraction/synthetic_combined/pos_features-v1.csv \
        --dataset_name synthetic_combined \
        --pos_feature AUX \
        --detailed_pos_feature MD \
        --visualize
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
        'pos_features-v1.csv',
    )

    parser = argparse.ArgumentParser(description='Classify sentence tense using rule-based POS features.')

    # Full path to the POS features CSV (explicit – no auto-discovery of latest version)
    parser.add_argument('--dataset', default=default_dataset,
                        help='Full path to the POS features CSV file.')

    # Name used for organising the output folder
    parser.add_argument('--dataset_name', type=str, default=f"synthetic_batch_{batch_idx}",
                        help='Name of the dataset. Used for naming the output folder.')

    # Coarse POS label for the classification rule
    parser.add_argument('--pos_feature', type=str, default='AUX',
                        help='Coarse POS label to use for classification rule (e.g., AUX).')

    # Fine-grained POS tag for the classification rule
    parser.add_argument('--detailed_pos_feature', type=str, default='MD',
                        help='Fine-grained POS tag to use for classification rule (e.g., MD).')

    # Flag to optionally show the pop-up window; image is always saved regardless
    parser.add_argument('--visualize', action='store_true',
                        help='If set, show the interactive plot window. Image is always saved.')

    args = parser.parse_args()

    # ============================================================
    # 2. LOAD DATASET
    # ============================================================
    df = load_dataset(base_data_path, args.dataset)

    # ============================================================
    # 3. CLASSIFY TENSE – sentence level
    # Rule: sentence contains AUX + MD → Prediction (1), else Non-Prediction (0)
    # ============================================================
    results_df = classify_tense_per_sentence(
        df,
        pos_feature=args.pos_feature,
        detailed_pos_feature=args.detailed_pos_feature,
    )

    # Tag each row with the source file so we can trace back later
    results_df['Source Dataset'] = args.dataset

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

    # All outputs go under rule_based/dataset_name/
    save_path = os.path.join(
        base_data_path,
        'tense_classification',
        'rule_based',
        f'{args.dataset_name}',
    )

    # Save sentence-level classifications
    DataProcessing.save_to_file(
        data=results_df,
        path=save_path,
        prefix='classifications',
        save_file_type='csv',
        include_version=True,
    )

    # Save metadata so every run is fully reproducible
    classification_metadata = {
        "dataset_name": args.dataset_name,
        "source_features_path": args.dataset,
        "method": "rule_based",
        "classification_rule": f"POS=={args.pos_feature} AND Detailed POS=={args.detailed_pos_feature}",
        "pos_feature": args.pos_feature,
        "detailed_pos_feature": args.detailed_pos_feature,
        "total_token_rows_input": len(df),
        "total_sentences_classified": len(results_df),
        "prediction_count": int(results_df['Future Tense'].sum()),
        "non_prediction_count": int((results_df['Future Tense'] == 0).sum()),
        "expected_sentence_count": int(df['N Sentences'].iloc[0]),
        "columns": list(results_df.columns),
        "classification_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    DataProcessing.save_to_file(
        data=classification_metadata,
        path=save_path,
        prefix='classification_metadata',
        save_file_type='json',
        include_version=False,
    )

    # ============================================================
    # 6. VISUALISATION – always saves image; pop-up only if --visualize
    # ============================================================
    visualize_tense_distribution(
        results_df=results_df,
        save_path=save_path,
        dataset_name=args.dataset_name,
        show=args.visualize,        # False by default → no blocking pop-up
    )

    print("\n✓ Tense classification complete!")
    print(f"Files saved to          : {save_path}")
    print(f"Prediction sentences    : {classification_metadata['prediction_count']}")
    print(f"Non-Prediction sentences: {classification_metadata['non_prediction_count']}")
    print(f"Total sentences         : {classification_metadata['total_sentences_classified']}")