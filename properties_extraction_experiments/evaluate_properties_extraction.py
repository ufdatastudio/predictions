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
from feature_extraction import SpacyFeatureExtraction, SentenceTransformerFeatureExtraction

def embed_properties(y_df, y_hat_df, col_names, embedding_model_name='spacy_large'):
    """
    Generate SpaCy or Sentence‑Transformer embeddings for each property column
    in both the ground‑truth (y_df) and the LLM predictions (y_hat_df).

    Parameters
    ----------
    y_df : pd.DataFrame
        Ground‑truth dataframe.
    y_hat_df : pd.DataFrame
        LLM‑predicted dataframe.
    col_names : list[str]
        The property column names (e.g., ['Source', 'Target', ...]).
    embedding_model_name : str, optional
        The embedding model to use (default: 'spacy_large').

    Returns
    -------
    list[dict]
        One dictionary per property containing the embedded dataframes.
    """
    property_results = []

    # ------------------------------------------------------------------
    # Loop over every property and embed ground‑truth and predictions
    # ------------------------------------------------------------------
    for col_name in col_names:
        print(f"Embeddings for {col_name}")

        # ------------------------------------------------------------------
        # Ground‑truth embeddings
        # ------------------------------------------------------------------
        print(f"\tGround Truth")
        if embedding_model_name.startswith('st_'):
            # Sentence‑Transformer path
            y_fe = SentenceTransformerFeatureExtraction(
                y_df, col_name, embedding_model_name=embedding_model_name
            )
        else:
            # SpaCy path (default or any spacy_<size> model)
            y_fe = SpacyFeatureExtraction(
                y_df, col_name, embedding_model_name=embedding_model_name
            )
        embed_y_df = y_fe.sentence_embeddings_extraction(attach_to_df=True)

        # ------------------------------------------------------------------
        # LLM prediction embeddings
        # ------------------------------------------------------------------
        print(f"\tLLM Extraction")
        if embedding_model_name.startswith('st_'):
            y_hat_fe = SentenceTransformerFeatureExtraction(
                y_hat_df, col_name, embedding_model_name=embedding_model_name
            )
        else:
            y_hat_fe = SpacyFeatureExtraction(
                y_hat_df, col_name, embedding_model_name=embedding_model_name
            )
        embed_y_hat_df = y_hat_fe.sentence_embeddings_extraction(attach_to_df=True)

        # Sanity‑check that we have the same number of rows
        np.testing.assert_equal(len(embed_y_df), len(embed_y_hat_df))

        # Store the pair of dataframes so that downstream code can compare
        property_results.append({
            'property_name': col_name,
            'y_data': embed_y_df,
            'y_hat_data': embed_y_hat_df
        })

    return property_results

def map_words_to_labels(y_df, y_hat_df, col_name):
    """
    Map word-level predictions to binary classification labels using cosine similarity.
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
                        y_embed.reshape(1, -1),
                        y_hat_embed.reshape(1, -1),
                        per_row=False,
                        idx=0
                    )

                if cs >= 0.9:
                    tps.append({'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': cs,   'y': 1, 'y_hat': 1})
                else:
                    fns.append({'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': cs,   'y': 1, 'y_hat': 0})
            else:
                fns.append(    {'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': None, 'y': 1, 'y_hat': 0})

        elif y_embed is None:
            if y_hat_embed is not None:
                # Note: If y_hat_word == 'PARSE_ERROR', it counts as a False Positive here naturally.
                fps.append(    {'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': None, 'y': 0, 'y_hat': 1})
            else:
                tns.append(    {'y_word': y_word, 'y_hat_word': y_hat_word, 'cs': None, 'y': 0, 'y_hat': 0})

    return tps, fns, fps, tns

def evaluate_properties(property_results, model_name, seed, parse_error_count, parse_error_rate):
    """
    Compute classification metrics per property column.
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
            'parse_error_count':  parse_error_count,
            'parse_error_rate':   parse_error_rate,
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

    # Same file
    python3 evaluate_properties_extraction.py \
  --y_path extraction_results/naacl_2026_submission/classification/zero-shot/seed3/gemma-3-27b-it/extracted_properties.csv \
  --y_hat_path extraction_results/naacl_2026_submission/classification/zero-shot/seed3/gemma-3-27b-it/extracted_properties.csv \
  --model_name gemma-3-27b-it \
  --seed 3

    # Diff files
    python3 evaluate_properties_extraction.py \
  --y_path extraction_results/naacl_2026_submission/ground_truth/zero-shot/seed3/gpt-oss-120b/extracted_properties.csv \
  --y_hat_path extraction_results/naacl_2026_submission/classification/zero-shot/seed3/gemma-3-27b-it/extracted_properties.csv \
  --model_name 'gpt-oss-120b x gemma-3-27b-it' \
  --seed 3
    
    """
    print("\n" + "="*50)
    print("PROPERTY EXTRACTION EVALUATION")
    print("="*50)

    # ============================================================
    # 1. Configuration and Arguments
    # ============================================================
    parser = argparse.ArgumentParser(description='Evaluate prediction property extraction.')
    parser.add_argument('--y_path', type=str, required=True, help='Path to ground truth CSV.')
    parser.add_argument('--y_hat_path', type=str, required=True, help='Path to model predictions CSV.')
    parser.add_argument('--model_name', type=str, default='openai/gpt-oss-120b', help='Model name.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--embedding_model',
                    default='spacy_large',
                    choices=['spacy_small', 'spacy_medium', 'spacy_large', 'spacy_transformer',
                             'st_mpnet_base', 'st_distilroberta', 'st_minilm_l12', 'st_minilm_l6'],
                    help='SpaCy or Sentence‑Transformer model to use for sentence vectorization.')
    args = parser.parse_args()

    base_data_path = DataProcessing.load_base_data_path(script_dir)

    # ============================================================
    # 2. Load Data & Calculate Parse Errors
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

    # Calculate Parse Error Rate
    total_rows = len(y_hat_df)
    parse_error_count = (y_hat_df.get('Parse Status', pd.Series()) == 'PARSE_ERROR').sum()
    parse_error_rate = parse_error_count / total_rows if total_rows > 0 else 0.0

    print(f"\nParse Error Count: {parse_error_count} / {total_rows}")
    print(f"Parse Error Rate : {parse_error_rate:.2%}")

    col_names = y_df.loc[:, ["Source", "Target", "Date", "Outcome"]].columns.tolist()
    print(f"Property columns: {col_names}")

    # ============================================================
    # 3. Embed Properties
    # ============================================================
    print("\n" + "="*50)
    print("STEP: EMBED PROPERTIES")
    print("="*50)

    # 👉 New: supply the embedding model selected on the CLI
    property_results = embed_properties(
        y_df, y_hat_df, col_names, embedding_model_name=args.embedding_model
    )

    # ============================================================
    # 4. Evaluate Properties
    # ============================================================
    print("\n" + "="*50)
    print("STEP: EVALUATE PROPERTIES")
    print("="*50)
    metrics_summary_df = evaluate_properties(property_results, args.model_name, args.seed, parse_error_count, parse_error_rate)
    print(f"\nMetrics Summary:\n{metrics_summary_df}\n")

    # ============================================================
    # 5. Save Results
    # ============================================================
    print("\n" + "="*50)
    print("STEP: SAVE RESULTS")
    print("="*50)

    path_parts = args.y_hat_path.split('/')
    dataset_folders = path_parts[:-4]
    dataset_folder = '/'.join(dataset_folders)
    clean_model_name = args.model_name.replace('/', '_')

    eval_save_path = os.path.join(
        base_data_path, dataset_folder, f'seed{args.seed}', clean_model_name, args.embedding_model
    )
    os.makedirs(eval_save_path, exist_ok=True)

    DataProcessing.save_to_file(
        metrics_summary_df, path=eval_save_path, prefix=f'metrics_summary_{clean_model_name}_{args.embedding_model}',
        save_file_type='csv', include_version=True
    )
    print(f"✓ Saved metrics summary to: {eval_save_path}")
    print("\n✓ Evaluation complete!")