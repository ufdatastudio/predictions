# average_agreement_results.py
# Averages Fleiss Kappa, Per Row Agreement, Per Row Entropy, and Majority Vote
# distribution across seeds for ML, LLM, and All Models combined.
#
# Usage:
#   python average_agreement_results.py \
#       --experiment synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-25

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Get the current working directory of the script
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))

from data_processing import DataProcessing


def load_agreement_data(dataset_folder_path, dir_files):
    """
    Walk through all seed folders and load ml, llm, and all_models agreement CSVs.

    Parameters
    ----------
    dataset_folder_path : str
        Base path to the experiment folder containing seed subfolders.
    dir_files : list of str
        List of folder/file names inside dataset_folder_path.

    Returns
    -------
    ml_agreement_by_seed : dict
        {seed_value: df} ml_classifiers_agreement.csv per seed.
    llm_agreement_by_seed : dict
        {seed_value: df} llm_classifiers_agreement.csv per seed.
    all_models_agreement_by_seed : dict
        {seed_value: df} all_models_agreement.csv per seed.
    """
    ml_agreement_by_seed = {}
    llm_agreement_by_seed = {}
    all_models_agreement_by_seed = {}

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

        for seed_path_file in sorted(seed_path_files):

            if seed_path_file == 'ml_classifiers_agreement.csv':
                print(f"  ✓ Loading ML agreement: {seed_path_file}")
                path = os.path.join(seed_path, seed_path_file)
                ml_agreement_by_seed[seed_value] = DataProcessing.load_from_file(path=path)

            elif seed_path_file == 'llm_classifiers_agreement.csv':
                print(f"  ✓ Loading LLM agreement: {seed_path_file}")
                path = os.path.join(seed_path, seed_path_file)
                llm_agreement_by_seed[seed_value] = DataProcessing.load_from_file(path=path)

            elif seed_path_file == 'all_models_agreement.csv':
                print(f"  ✓ Loading All Models agreement: {seed_path_file}")
                path = os.path.join(seed_path, seed_path_file)
                all_models_agreement_by_seed[seed_value] = DataProcessing.load_from_file(path=path)

    return ml_agreement_by_seed, llm_agreement_by_seed, all_models_agreement_by_seed


def compute_per_seed_summary(agreement_by_seed, model_type, model_cols_to_skip):
    """
    For each seed, compute:
    - Fleiss Kappa (already computed per-row, stored as overall scalar — recompute from freq table)
    - Per Row Agreement mean
    - Per Row Entropy mean
    - Majority Vote class distribution (% predicted as 1)

    Parameters
    ----------
    agreement_by_seed : dict
        {seed_value: df} agreement DataFrame per seed.
    model_type : str
        'ml', 'llm', or 'all_models' — used for labeling.
    model_cols_to_skip : list of str
        Metadata and agreement columns to exclude when identifying model prediction cols.

    Returns
    -------
    per_seed_summary : dict
        {seed_value: {metric: value}} raw per-seed metrics.
    """
    per_seed_summary = {}

    for seed_value, df in sorted(agreement_by_seed.items()):
        mv_col = f'{model_type.upper()} Majority Vote' if model_type != 'all_models' \
                 else 'All Models Majority Vote'

        # Per Row Agreement and Entropy are already computed columns
        per_row_agreement_mean = df['Per Row Agreement'].mean() \
            if 'Per Row Agreement' in df.columns else None
        per_row_entropy_mean = df['Per Row Entropy'].mean() \
            if 'Per Row Entropy' in df.columns else None

        # Majority Vote class distribution
        mv_dist = None
        if mv_col in df.columns:
            mv_counts = df[mv_col].value_counts(normalize=True)
            mv_dist = {
                'pct_predicted_0': round(mv_counts.get(0, 0.0), 4),
                'pct_predicted_1': round(mv_counts.get(1, 0.0), 4),
                'n_predicted_0': int(df[mv_col].value_counts().get(0, 0)),
                'n_predicted_1': int(df[mv_col].value_counts().get(1, 0)),
            }

        per_seed_summary[seed_value] = {
            'per_row_agreement_mean': per_row_agreement_mean,
            'per_row_entropy_mean': per_row_entropy_mean,
            'majority_vote_distribution': mv_dist
        }

        print(f"\n  Seed {seed_value} [{model_type}]:")
        print(f"    Per Row Agreement (mean): {per_row_agreement_mean:.4f}")
        print(f"    Per Row Entropy (mean):   {per_row_entropy_mean:.4f}")
        if mv_dist:
            print(f"    Majority Vote → 0: {mv_dist['n_predicted_0']} ({mv_dist['pct_predicted_0']*100:.1f}%)")
            print(f"    Majority Vote → 1: {mv_dist['n_predicted_1']} ({mv_dist['pct_predicted_1']*100:.1f}%)")

    return per_seed_summary


def average_across_seeds(per_seed_summary, model_type):
    """
    Average Per Row Agreement, Per Row Entropy, and Majority Vote distribution
    across all seeds. Returns mean ± std for each metric.

    Parameters
    ----------
    per_seed_summary : dict
        {seed_value: {metric: value}} raw per-seed metrics.
    model_type : str
        'ml', 'llm', or 'all_models' — used for labeling.

    Returns
    -------
    averaged : dict
        {metric: {'mean': float, 'std': float}} averaged metrics.
    """
    agreements = [v['per_row_agreement_mean'] for v in per_seed_summary.values()
                  if v['per_row_agreement_mean'] is not None]
    entropies = [v['per_row_entropy_mean'] for v in per_seed_summary.values()
                 if v['per_row_entropy_mean'] is not None]
    pct_1s = [v['majority_vote_distribution']['pct_predicted_1']
              for v in per_seed_summary.values()
              if v['majority_vote_distribution'] is not None]
    pct_0s = [v['majority_vote_distribution']['pct_predicted_0']
              for v in per_seed_summary.values()
              if v['majority_vote_distribution'] is not None]

    averaged = {
        'model_type': model_type,
        'n_seeds': len(per_seed_summary),
        'seeds': sorted(per_seed_summary.keys()),
        'per_row_agreement': {
            'mean': round(float(np.mean(agreements)), 4),
            'std': round(float(np.std(agreements)), 4)
        },
        'per_row_entropy': {
            'mean': round(float(np.mean(entropies)), 4),
            'std': round(float(np.std(entropies)), 4)
        },
        'majority_vote_pct_predicted_1': {
            'mean': round(float(np.mean(pct_1s)), 4),
            'std': round(float(np.std(pct_1s)), 4)
        },
        'majority_vote_pct_predicted_0': {
            'mean': round(float(np.mean(pct_0s)), 4),
            'std': round(float(np.std(pct_0s)), 4)
        },
        'per_seed_details': per_seed_summary
    }

    print(f"\n{'='*50}")
    print(f"AVERAGED [{model_type.upper()}] across {len(per_seed_summary)} seeds")
    print(f"{'='*50}")
    print(f"  Per Row Agreement: {averaged['per_row_agreement']['mean']:.4f} ± {averaged['per_row_agreement']['std']:.4f}")
    print(f"  Per Row Entropy:   {averaged['per_row_entropy']['mean']:.4f} ± {averaged['per_row_entropy']['std']:.4f}")
    print(f"  MV → 1 (%):        {averaged['majority_vote_pct_predicted_1']['mean']:.4f} ± {averaged['majority_vote_pct_predicted_1']['std']:.4f}")
    print(f"  MV → 0 (%):        {averaged['majority_vote_pct_predicted_0']['mean']:.4f} ± {averaged['majority_vote_pct_predicted_0']['std']:.4f}")

    return averaged


def save_results(averaged_ml, averaged_llm, averaged_all, dataset_folder_path, experiment_name):
    """
    Save averaged agreement results as both CSV and JSON.

    Parameters
    ----------
    averaged_ml : dict
        Averaged ML agreement metrics.
    averaged_llm : dict
        Averaged LLM agreement metrics.
    averaged_all : dict
        Averaged All Models agreement metrics.
    dataset_folder_path : str
        Base experiment folder path.
    experiment_name : str
        Experiment name used in metadata.
    """
    save_path = os.path.join(dataset_folder_path, 'averaged', 'in_domain', 'agreement')
    os.makedirs(save_path, exist_ok=True)

    # --------------------------------------------------------
    # 1. Save combined summary CSV (one row per model type)
    # --------------------------------------------------------
    rows = []
    for averaged in [averaged_ml, averaged_llm, averaged_all]:
        rows.append({
            'model_type': averaged['model_type'],
            'n_seeds': averaged['n_seeds'],
            'per_row_agreement_mean': averaged['per_row_agreement']['mean'],
            'per_row_agreement_std': averaged['per_row_agreement']['std'],
            'per_row_entropy_mean': averaged['per_row_entropy']['mean'],
            'per_row_entropy_std': averaged['per_row_entropy']['std'],
            'mv_pct_predicted_1_mean': averaged['majority_vote_pct_predicted_1']['mean'],
            'mv_pct_predicted_1_std': averaged['majority_vote_pct_predicted_1']['std'],
            'mv_pct_predicted_0_mean': averaged['majority_vote_pct_predicted_0']['mean'],
            'mv_pct_predicted_0_std': averaged['majority_vote_pct_predicted_0']['std'],
        })

    summary_df = pd.DataFrame(rows)
    print(f"\n{'='*50}")
    print("AVERAGED AGREEMENT SUMMARY")
    print(f"{'='*50}")
    print(summary_df.to_string(index=False))

    DataProcessing.save_to_file(
        data=summary_df,
        path=save_path,
        prefix='agreement_summary',
        save_file_type='csv',
        include_version=False
    )
    print(f"\n✓ Saved: agreement_summary.csv to {save_path}")

    # --------------------------------------------------------
    # 2. Save full JSON with per-seed details for supplementary
    # --------------------------------------------------------
    full_json = {
        'experiment': experiment_name,
        'date_averaged': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ml': averaged_ml,
        'llm': averaged_llm,
        'all_models': averaged_all
    }

    DataProcessing.save_to_file(
        data=full_json,
        path=save_path,
        prefix='agreement_summary',
        save_file_type='json',
        include_version=False
    )
    print(f"✓ Saved: agreement_summary.json to {save_path}")

    return summary_df


if __name__ == "__main__":
    """
    usage:
    python average_agreement_results.py \
        --experiment synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-25
    """
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_results_path = os.path.join(base_data_path, 'classification_results')

    parser = argparse.ArgumentParser(
        description='Average agreement metrics across seeds for ML, LLM, and All Models.'
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
    print("AVERAGE AGREEMENT RESULTS")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Path: {dataset_folder_path}")
    print(f"Seed folders found: {sorted([f for f in dir_files if 'seed' in f])}")

    # ============================================================
    # 3. LOAD AGREEMENT DATA
    # ============================================================
    ml_agreement_by_seed, llm_agreement_by_seed, all_models_agreement_by_seed = load_agreement_data(
        dataset_folder_path, dir_files
    )

    print(f"\nML seeds loaded:         {sorted(ml_agreement_by_seed.keys())}")
    print(f"LLM seeds loaded:        {sorted(llm_agreement_by_seed.keys())}")
    print(f"All Models seeds loaded: {sorted(all_models_agreement_by_seed.keys())}")

    # Metadata and agreement columns to skip when identifying model prediction cols
    model_cols_to_skip = [
        'seed', 'index', 'original_index', 'text', 'Base Sentence',
        'Ground Truth', 'Sentence Label', 'Dataset Name',
        'Base Sentence Embedding', 'ML Majority Vote', 'LLM Majority Vote',
        'All Models Majority Vote', 'Per Row Agreement', 'Per Row Entropy'
    ]

    # ============================================================
    # 4. COMPUTE PER-SEED SUMMARIES
    # ============================================================
    print(f"\n{'='*60}")
    print("PER-SEED SUMMARIES")
    print(f"{'='*60}")

    ml_per_seed = compute_per_seed_summary(ml_agreement_by_seed, 'ml', model_cols_to_skip)
    llm_per_seed = compute_per_seed_summary(llm_agreement_by_seed, 'llm', model_cols_to_skip)
    all_per_seed = compute_per_seed_summary(all_models_agreement_by_seed, 'all_models', model_cols_to_skip)

    # ============================================================
    # 5. AVERAGE ACROSS SEEDS
    # ============================================================
    print(f"\n{'='*60}")
    print("AVERAGING ACROSS SEEDS")
    print(f"{'='*60}")

    averaged_ml = average_across_seeds(ml_per_seed, 'ml')
    averaged_llm = average_across_seeds(llm_per_seed, 'llm')
    averaged_all = average_across_seeds(all_per_seed, 'all_models')

    # ============================================================
    # 6. SAVE RESULTS
    # ============================================================
    summary_df = save_results(
        averaged_ml=averaged_ml,
        averaged_llm=averaged_llm,
        averaged_all=averaged_all,
        dataset_folder_path=dataset_folder_path,
        experiment_name=args.experiment
    )

    # ============================================================
    # 7. COMPLETE
    # ============================================================
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Results saved to: {os.path.join(dataset_folder_path, 'averaged', 'in_domain', 'agreement')}")
    print()