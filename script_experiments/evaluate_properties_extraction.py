"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Property Extraction Evaluation
> y (ground truth) vs y_hat (LLM)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))
from metrics import EvaluationMetric
from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction


def embed_properties(y_df, y_hat_df, col_names):
    """
    Generate spaCy embeddings for each property column in both y and y_hat DataFrames.

    Parameters
    ----------
    y_df : pd.DataFrame
        Ground truth DataFrame.
    y_hat_df : pd.DataFrame
        Model prediction DataFrame.
    col_names : list of str
        Property columns to embed.

    Returns
    -------
    list of dict
        Each dict contains property_name, y_data, y_hat_data with embeddings attached.
    """
    property_results = []

    for col_name in col_names:
        print(f"Embeddings for {col_name}")

        y_spacy_fe = SpacyFeatureExtraction(y_df, col_name)
        embed_y_df = y_spacy_fe.sentence_embeddings_extraction(attach_to_df=True)

        y_hat_spacy_fe = SpacyFeatureExtraction(y_hat_df, col_name)
        embed_y_hat_df = y_hat_spacy_fe.sentence_embeddings_extraction(attach_to_df=True)

        np.testing.assert_equal(len(embed_y_df), len(embed_y_hat_df))

        property_results.append({
            'property_name': col_name,
            'y_data': embed_y_df,
            'y_hat_data': embed_y_hat_df
        })

    return property_results


def map_words_to_labels(y_df, y_hat_df, col_name):
    """
    Map word-level predictions to binary classification labels using cosine similarity.

    Compares ground truth to LLM predicted word for a given property column by computing
    cosine similarity between their embeddings. Each example is assigned a TP, FN, FP,
    or TN label based on the presence of embeddings and semantic similarity threshold.

    Parameters
    ----------
    y_df : pd.DataFrame
        Ground truth dataframe containing the property column and its embedding column.
    y_hat_df : pd.DataFrame
        Predicted dataframe containing the property column and its embedding column.
    col_name : str
        Name of the property column to evaluate (e.g., 'Source', 'Target', 'Date', 'Outcome').
        Expects a corresponding '{col_name} Embedding' column in both dataframes.

    Returns
    -------
    tps : list of dict
        True positives — y and y_hat are both present and cosine similarity >= 0.9.
    fns : list of dict
        False negatives — y is present but y_hat is absent or cosine similarity < 0.9.
    fps : list of dict
        False positives — y is absent but y_hat is present.
    tns : list of dict
        True negatives — both y and y_hat are absent.
    """
    tps = []
    fns = []
    fps = []
    tns = []

    for idx in range(len(y_df)):
        y_word      = y_df[f'{col_name}'].iloc[idx]
        y_embed     = y_df[f'{col_name} Embedding'].iloc[idx]
        y_hat_word  = y_hat_df[f'{col_name}'].iloc[idx]
        y_hat_embed = y_hat_df[f'{col_name} Embedding'].iloc[idx]

        if y_embed is not None:
            if y_hat_embed is not None:
                # Fallback for OOV/zero vectors
                if np.linalg.norm(y_embed) == 0 or np.linalg.norm(y_hat_embed) == 0:
                    cs = 1.0 if y_word == y_hat_word else 0.0
                else:
                    cs = EvaluationMetric.get_cosine_similarity(
                        y_embed, y_hat_embed, per_row=False, idx=idx
                    )

                if cs >= 0.9:
                    tps.append({'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': cs,   'y': 1, 'y_hat': 1})
                else:
                    fns.append({'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': cs,   'y': 1, 'y_hat': 0})
            else:
                fns.append(    {'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': None, 'y': 1, 'y_hat': 0})

        elif y_embed is None:
            if y_hat_embed is not None:
                fps.append(    {'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': None, 'y': 0, 'y_hat': 1})
            else:
                tns.append(    {'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': None, 'y': 0, 'y_hat': 0})

    return tps, fns, fps, tns


def evaluate_properties(property_results, seed, model_name):
    """
    Compute classification metrics per property column.

    Builds classification report and confusion matrix across
    Source, Target, Date, and Outcome.

    Parameters
    ----------
    property_results : list of dict
        Output from embed_properties(). Each dict contains
        property_name, y_data, y_hat_data.
    seed : int
        Random seed used for reproducibility tracking.
    model_name : str
        Name of the model being evaluated.

    Returns
    -------
    pd.DataFrame
        Metrics summary with one row per property.
    """
    metrics_summary = []

    for property_result in property_results:
        property_name = property_result['property_name']
        print(f"\nClassification Results from: {property_name}")

        y_df     = property_result['y_data']
        y_hat_df = property_result['y_hat_data']

        tps, fns, fps, tns = map_words_to_labels(y_df, y_hat_df, property_name)
        print(f"\t#TP: {len(tps)}")
        print(f"\t#FN: {len(fns)}")
        print(f"\t#FP: {len(fps)}")
        print(f"\t#TN: {len(tns)}")

        tps_df = pd.DataFrame(tps)
        fns_df = pd.DataFrame(fns)
        fps_df = pd.DataFrame(fps)
        tns_df = pd.DataFrame(tns)

        eval_report_df    = DataProcessing.concat_dfs([tps_df, fns_df, fps_df, tns_df])
        actual_labels     = eval_report_df['y']
        predicted_labels  = eval_report_df['y_hat']

        print(f"\tClassification Report")
        eval_report = EvaluationMetric.eval_classification_report(actual_labels, predicted_labels)

        confusion_mat, tn, fp, fn, tp = EvaluationMetric.get_confusion_matrix(
            actual_labels, predicted_labels, by_category=True
        )
        print(f"Confusion Matrix:\n{confusion_mat}\n")

        metrics_summary.append({
            'seed':               seed,
            'model':              model_name,
            'property':           property_name,
            'test_accuracy':      eval_report.get('accuracy', None),
            'precision_class_0':  eval_report.get('0', {}).get('precision', None),
            'precision_class_1':  eval_report.get('1', {}).get('precision', None),
            'recall_class_0':     eval_report.get('0', {}).get('recall', None),
            'recall_class_1':     eval_report.get('1', {}).get('recall', None),
            'f1_class_0':         eval_report.get('0', {}).get('f1-score', None),
            'f1_class_1':         eval_report.get('1', {}).get('f1-score', None),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })

    return pd.DataFrame(metrics_summary)


if __name__ == "__main__":
    """
    Usage:
        python3 evaluate_properties_extraction.py \
            --y_path extract_properties/ground_truth/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/llama-3.1-8b-instant/extracted_properties.csv \
            --y_hat_path extract_properties/classification/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/openai_gpt-oss-120b/extracted_properties.csv \
            --model_name "openai/gpt-oss-120b" \
            --seed 42
    """
    print("\n" + "="*50)
    print("PROPERTY EXTRACTION EVALUATION")
    print("="*50)

    # ============================================================
    # 1. Configuration and Arguments
    # ============================================================
    parser = argparse.ArgumentParser(description='Evaluate prediction property extraction.')
    parser.add_argument(
        '--y_path',
        type=str,
        required=True,
        help='Path to ground truth CSV relative to base_data_path.'
    )
    parser.add_argument(
        '--y_hat_path',
        type=str,
        required=True,
        help='Path to model predictions CSV relative to base_data_path.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='openai/gpt-oss-120b',
        help='Name of the model being evaluated.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility tracking.'
    )
    args = parser.parse_args()

    base_data_path = DataProcessing.load_base_data_path(script_dir)

    # ============================================================
    # 2. Load Data
    # ============================================================
    print("\n" + "="*50)
    print("STEP: LOAD DATA")
    print("="*50)

    y_path     = os.path.join(base_data_path, args.y_path)
    y_hat_path = os.path.join(base_data_path, args.y_hat_path)

    y_df     = DataProcessing.load_from_file(y_path)
    y_hat_df = DataProcessing.load_from_file(y_hat_path)

    print(f"Ground truth shape    : {y_df.shape}")
    print(f"Model prediction shape: {y_hat_df.shape}")

    # Property columns to evaluate (columns 4 through 8)
    col_names = y_df.loc[:, ["No Property", "Source", "Target", "Date", "Outcome"]].columns.tolist()
    print(f"Property columns: {col_names}")

    # ============================================================
    # 3. Embed Properties
    # ============================================================
    print("\n" + "="*50)
    print("STEP: EMBED PROPERTIES")
    print("="*50)

    property_results = embed_properties(y_df, y_hat_df, col_names)

    # ============================================================
    # 4. Evaluate Properties
    # ============================================================
    print("\n" + "="*50)
    print("STEP: EVALUATE PROPERTIES")
    print("="*50)

    metrics_summary_df = evaluate_properties(property_results, args.seed, args.model_name)
    print(f"\nMetrics Summary:\n{metrics_summary_df}\n")

    # ============================================================
    # 5. Save Results
    # ============================================================
    print("\n" + "="*50)
    print("STEP: SAVE RESULTS")
    print("="*50)

    # Extract dataset folder name from y_path
    # e.g., "extract_properties/ground_truth/synthetic-fpb-.../llama.../extracted_properties.csv"
    # -> "synthetic-fpb-..."
    path_parts = args.y_path.split('/')
    dataset_folder = path_parts[2]

    # Replace "/" with "_" in model name to avoid nested folder creation
    # e.g., "openai/gpt-oss-120b" -> "openai_gpt-oss-120b"
    clean_model_name = args.model_name.replace('/', '_')

    eval_save_path = os.path.join(
        base_data_path,
        'classification_results',
        dataset_folder,
        'properties'
    )
    os.makedirs(eval_save_path, exist_ok=True)

    DataProcessing.save_to_file(
        metrics_summary_df,
        path=eval_save_path,
        prefix=f'metrics_summary_{clean_model_name}',
        save_file_type='csv',
        include_version=True
    )

    print(f"✓ Saved metrics summary to: {eval_save_path}")
    print("\n✓ Evaluation complete!")