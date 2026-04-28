# evaluation_agreement.py
# Computes majority vote and Fleiss Kappa agreement metrics for ML and LLM classifiers.
#
# Usage:
#   python agreement.py \
#       --experiment synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-25

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

# Get the current working directory of the script
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))

from metrics import EvaluationMetric
from data_processing import DataProcessing


def load_seed_data(dataset_folder_path, dir_files):
    """
    Walk through all seed folders and load ml_classifiers_in_domain.csv,
    llm_classifiers_in_domain.csv, and x_y_test_set.csv.

    Parameters
    ----------
    dataset_folder_path : str
        Base path to the experiment folder containing seed subfolders.
    dir_files : list of str
        List of folder/file names inside dataset_folder_path.

    Returns
    -------
    seed_ml_classifiers_map : dict
        {seed_value: df} one DataFrame per seed for ml_classifiers_in_domain.csv.
    seed_llm_classifiers_map : dict
        {seed_value: df} one DataFrame per seed for llm_classifiers_in_domain.csv.
    x_test_dfs : list of pd.DataFrame
        All x_y_test_set DataFrames per seed.
    """
    x_test_dfs = []
    seed_ml_classifiers_map = {}
    seed_llm_classifiers_map = {}

    for dir_file in sorted(dir_files):
        if "seed" not in dir_file:
            continue

        seed_path = os.path.join(dataset_folder_path, dir_file, 'in_domain')

        if not os.path.exists(seed_path):
            print(f"⚠️  Skipping {dir_file} — in_domain folder not found.")
            continue

        seed_path_files = os.listdir(seed_path)
        seed_value = int(re.search(r'\d+', dir_file).group())

        print(f"\n{'='*40}")
        print(f"SEED: {seed_value}")
        print(f"{'='*40}")

        for seed_path_file in seed_path_files:

            if "ml_classifiers_in_domain" in seed_path_file:
                print(f"  ✓ Loading ML classifiers: {seed_path_file}")
                ml_path = os.path.join(seed_path, seed_path_file)
                df = DataProcessing.load_from_file(path=ml_path)
                seed_ml_classifiers_map[seed_value] = df

            elif "llm_classifiers_in_domain" in seed_path_file:
                print(f"  ✓ Loading LLM classifiers: {seed_path_file}")
                llm_path = os.path.join(seed_path, seed_path_file)
                df = DataProcessing.load_from_file(path=llm_path)
                seed_llm_classifiers_map[seed_value] = df

            if "x_y_test_set" in seed_path_file:
                print(f"  ✓ Loading test set: {seed_path_file}")
                x_test_path = os.path.join(seed_path, 'x_y_test_set.csv')
                x_test_df = DataProcessing.load_from_file(x_test_path)
                x_test_dfs.append(x_test_df)

    return seed_ml_classifiers_map, seed_llm_classifiers_map, x_test_dfs


def get_model_cols(df):
    """
    Extract model prediction columns from the classifier DataFrame.
    Skips all known metadata and agreement columns so only raw model
    prediction columns (0/1) are returned.

    Parameters
    ----------
    df : pd.DataFrame
        Classifier DataFrame.

    Returns
    -------
    list of str
        Model prediction column names.
    """
    meta_cols = [
        'seed', 'index', 'original_index', 'text', 'Base Sentence',
        'Ground Truth', 'Sentence Label', 'Dataset Name',
        'Base Sentence Embedding', 'ML Majority Vote', 'LLM Majority Vote',
        'All Models Majority Vote', 'Per Row Agreement', 'Per Row Entropy'
    ]
    return [col for col in df.columns if col not in meta_cols]


def compute_majority_vote(df, model_cols, label='Majority Vote'):
    """
    Compute majority vote across all model prediction columns.

    Parameters
    ----------
    df : pd.DataFrame
        Classifier DataFrame with one column per model.
    model_cols : list of str
        Model prediction column names.
    label : str
        Column name for the majority vote output. Default is 'Majority Vote'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a new majority vote column appended.
    """
    df = df.copy()
    df[label] = df[model_cols].mode(axis=1)[0].astype(int)
    return df


def compute_fleiss_kappa(df, model_cols):
    """
    Compute Fleiss Kappa across all model prediction columns,
    plus per-row agreement and per-row entropy.

    Parameters
    ----------
    df : pd.DataFrame
        Classifier DataFrame with one column per model.
    model_cols : list of str
        Model prediction column names.

    Returns
    -------
    kappa : float
        Overall Fleiss Kappa score.
    freq_table : pd.DataFrame
        Crosstab frequency table used for Fleiss Kappa.
    df : pd.DataFrame
        Input DataFrame with 'Per Row Agreement' and 'Per Row Entropy' appended.
    """
    kappa, freq_table, df = EvaluationMetric.get_fleiss_kappa(df, model_cols)
    return kappa, freq_table, df


def evaluate_agreement(seed_classifiers_map, classifier_type, dataset_folder_path):
    """
    Run majority vote and Fleiss Kappa for each seed and save results.

    Parameters
    ----------
    seed_classifiers_map : dict
        {seed_value: df} one DataFrame per seed.
    classifier_type : str
        'ml' or 'llm' — used for print labels and save file names.
    dataset_folder_path : str
        Base path used to construct the save path for each seed.

    Returns
    -------
    dict
        {seed_value: df} each DataFrame now includes Majority Vote,
        Per Row Agreement, and Per Row Entropy columns.
    """
    print(f"\n{'='*60}")
    print(f"AGREEMENT METRICS — {classifier_type.upper()}")
    print(f"{'='*60}")

    results = {}

    for seed_value in sorted(seed_classifiers_map.keys()):
        df = seed_classifiers_map[seed_value].copy()
        model_cols = get_model_cols(df)

        print(f"\n--- Seed {seed_value} | {classifier_type.upper()} ---")
        print(f"  Models: {model_cols}")
        print(f"  Shape: {df.shape}")

        # --------------------------------------------------------
        # 1. Majority Vote
        # --------------------------------------------------------
        mv_label = f'{classifier_type.upper()} Majority Vote'
        df = compute_majority_vote(df, model_cols, label=mv_label)
        print(f"\n  {mv_label} distribution:\n{df[mv_label].value_counts()}\n")

        # --------------------------------------------------------
        # 2. Fleiss Kappa (overall + per row)
        # --------------------------------------------------------
        kappa, freq_table, df = compute_fleiss_kappa(df, model_cols)
        print(f"  Fleiss Kappa: {kappa:.4f}")
        print(f"\n  Per Row Agreement (mean): {df['Per Row Agreement'].mean():.4f}")
        print(f"  Per Row Entropy (mean):    {df['Per Row Entropy'].mean():.4f}")

        # --------------------------------------------------------
        # 3. Save results
        # --------------------------------------------------------
        save_path = os.path.join(dataset_folder_path, f'seed{seed_value}', 'in_domain')
        DataProcessing.save_to_file(
            data=df,
            path=save_path,
            prefix=f'{classifier_type}_classifiers_agreement',
            save_file_type='csv',
            include_version=False
        )
        print(f"\n  ✓ Saved: {classifier_type}_classifiers_agreement.csv to {save_path}")

        results[seed_value] = df

    return results

def evaluate_combined_agreement(
    ml_agreement_results,
    llm_agreement_results,
    dataset_folder_path
):
    """
    Combine ML and LLM classifier DataFrames across seeds, compute a combined
    majority vote and Fleiss Kappa across all models together.

    Parameters
    ----------
    ml_agreement_results : dict
        {seed_value: df} ML agreement DataFrames with Per Row Agreement/Entropy.
    llm_agreement_results : dict
        {seed_value: df} LLM agreement DataFrames with Per Row Agreement/Entropy.
    dataset_folder_path : str
        Base path used to construct the save path for each seed.

    Returns
    -------
    dict
        {seed_value: df} Combined DataFrame per seed with All Models Majority Vote,
        Per Row Agreement, and Per Row Entropy across all ML and LLM models.
    """
    print(f"\n{'='*60}")
    print("AGREEMENT METRICS — ALL MODELS (ML + LLM COMBINED)")
    print(f"{'='*60}")

    combined_results = {}
    shared_seeds = sorted(set(ml_agreement_results.keys()) & set(llm_agreement_results.keys()))

    if not shared_seeds:
        print("⚠️  No shared seeds found between ML and LLM results. Skipping combined agreement.")
        return {}

    for seed_value in shared_seeds:
        ml_df = ml_agreement_results[seed_value].copy()
        llm_df = llm_agreement_results[seed_value].copy()

        # Get raw model prediction columns only — meta and agreement cols already excluded
        ml_model_cols = get_model_cols(ml_df)
        llm_model_cols = get_model_cols(llm_df)

        print(f"\n--- Seed {seed_value} | ALL MODELS ---")
        print(f"  ML models:  {ml_model_cols}")
        print(f"  LLM models: {llm_model_cols}")

        # Use whichever index column exists in the DataFrame
        index_col = 'original_index' if 'original_index' in ml_df.columns else \
                    'index' if 'index' in ml_df.columns else None

        if index_col is None:
            print(f"⚠️  No shared index column found. Aligning by positional index.")

            # ml_df uses 'Base Sentence' as text col, llm_df uses 'text'
            # Use llm_df as metadata base since it has seed, original_index, text
            ml_preds = ml_df[ml_model_cols].reset_index(drop=True)
            llm_preds = llm_df[llm_model_cols].reset_index(drop=True)
            all_preds = pd.concat([ml_preds, llm_preds], axis=1)

            combined_df = llm_df[['seed', 'original_index', 'text']].reset_index(drop=True).copy()
            combined_df = pd.concat([combined_df, all_preds], axis=1)

        else:
            ml_preds = ml_df.set_index(index_col)[ml_model_cols]
            llm_preds = llm_df.set_index(index_col)[llm_model_cols]
            all_preds = pd.concat([ml_preds, llm_preds], axis=1).reset_index()
            combined_df = ml_df[['seed', index_col, 'text']].copy()
            combined_df = combined_df.merge(all_preds, on=index_col, how='left')

        # <--- REMOVED stale lines that were here and overwriting the fix above

        all_model_cols = ml_model_cols + llm_model_cols
        print(f"  Total models combined: {len(all_model_cols)}")
        print(f"  Shape: {combined_df.shape}")

        # --------------------------------------------------------
        # 1. All Models Majority Vote
        # --------------------------------------------------------
        combined_df = compute_majority_vote(
            combined_df, all_model_cols, label='All Models Majority Vote'
        )
        print(f"\n  All Models Majority Vote distribution:\n{combined_df['All Models Majority Vote'].value_counts()}\n")

        # --------------------------------------------------------
        # 2. Fleiss Kappa across all models
        # --------------------------------------------------------
        kappa, freq_table, combined_df = compute_fleiss_kappa(combined_df, all_model_cols)
        print(f"  Fleiss Kappa (All Models): {kappa:.4f}")
        print(f"\n  Per Row Agreement (mean): {combined_df['Per Row Agreement'].mean():.4f}")
        print(f"  Per Row Entropy (mean):    {combined_df['Per Row Entropy'].mean():.4f}")

        # --------------------------------------------------------
        # 3. Save combined agreement results
        # --------------------------------------------------------
        save_path = os.path.join(dataset_folder_path, f'seed{seed_value}', 'in_domain')
        DataProcessing.save_to_file(
            data=combined_df,
            path=save_path,
            prefix='all_models_agreement',
            save_file_type='csv',
            include_version=False
        )
        print(f"\n  ✓ Saved: all_models_agreement.csv to {save_path}")

        combined_results[seed_value] = combined_df

    return combined_results

if __name__ == "__main__":
    """
    usage:
    python agreement.py \
        --experiment synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-25
    """
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_results_path = os.path.join(base_data_path, 'classification_results')

    parser = argparse.ArgumentParser(
        description='Compute majority vote and Fleiss Kappa for ML and LLM classifiers.'
    )

    parser.add_argument(
        '--experiment',
        required=True,
        help='Experiment folder name inside classification_results/. '
             'Example: synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-25'
    )

    parser.add_argument(
        '--results_path',
        default=default_results_path,
        help='Base path to classification_results/. Defaults to data/classification_results/'
    )

    args = parser.parse_args()

    # ============================================================
    # 2. SETUP
    # ============================================================
    dataset_folder_path = os.path.join(args.results_path, args.experiment)

    if not os.path.exists(dataset_folder_path):
        print(f"\n❌ Experiment folder not found: {dataset_folder_path}")
        sys.exit(1)

    dir_files = os.listdir(dataset_folder_path)

    print("\n" + "="*60)
    print("EVALUATION — AGREEMENT METRICS")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Path: {dataset_folder_path}")
    print(f"Seed folders found: {sorted([f for f in dir_files if 'seed' in f])}")

    # ============================================================
    # 3. LOAD SEED DATA
    # ============================================================
    seed_ml_classifiers_map, seed_llm_classifiers_map, x_test_dfs = load_seed_data(
        dataset_folder_path, dir_files
    )

    print(f"\nML seeds loaded:  {sorted(seed_ml_classifiers_map.keys())}")
    print(f"LLM seeds loaded: {sorted(seed_llm_classifiers_map.keys())}")

    # ============================================================
    # 4. EVALUATE ML AGREEMENT
    # ============================================================
    ml_agreement_results = evaluate_agreement(
        seed_classifiers_map=seed_ml_classifiers_map,
        classifier_type='ml',
        dataset_folder_path=dataset_folder_path
    )

    # ============================================================
    # 5. EVALUATE LLM AGREEMENT
    # ============================================================
    llm_agreement_results = evaluate_agreement(
        seed_classifiers_map=seed_llm_classifiers_map,
        classifier_type='llm',
        dataset_folder_path=dataset_folder_path
    )

    # ============================================================
    # 6. EVALUATE COMBINED AGREEMENT (ML + LLM)
    # ============================================================
    combined_agreement_results = evaluate_combined_agreement(
        ml_agreement_results=ml_agreement_results,
        llm_agreement_results=llm_agreement_results,
        dataset_folder_path=dataset_folder_path
    )

    # ============================================================
    # 7. COMPLETE
    # ============================================================
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"ML seeds processed:       {sorted(ml_agreement_results.keys())}")
    print(f"LLM seeds processed:      {sorted(llm_agreement_results.keys())}")
    print(f"Combined seeds processed: {sorted(combined_agreement_results.keys())}")
    print()