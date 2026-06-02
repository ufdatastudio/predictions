# combine_llm_classifiers_output.py
# Combines LLM classifier checkpoint outputs across models and seeds.
# Run after all LLM SLURM jobs have completed.
#
# Usage:
#   python combine_llm_classifiers_output.py \
#       --experiment synthetic-fpb-chronicle2050-yt-news-timebank_2026-04-22

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Get the current working directory of the script
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))

from data_processing import DataProcessing


def load_seed_data(dataset_folder_path, dir_files):
    """
    Walk through all seed folders and load checkpoint, metrics, and test set files.

    Parameters
    ----------
    dataset_folder_path : str
        Base path to the experiment folder containing seed subfolders.
    dir_files : list of str
        List of folder/file names inside dataset_folder_path.

    Returns
    -------
    seed_checkpoint_map : dict
        {seed_value: [df1, df2, ...]} one DataFrame per model checkpoint per seed.
    metrics_dfs_by_seed : dict
        {seed_value: [df1, df2, ...]} one DataFrame per model metrics per seed.
    x_test_dfs : list of pd.DataFrame
        All x_y_test_set DataFrames per seed.
    """
    metrics_dfs_by_seed = {}
    x_test_dfs = []
    seed_checkpoint_map = {}

    for dir_file in sorted(dir_files):
        if "seed" not in dir_file:
            continue

        seed_path = os.path.join(dataset_folder_path, dir_file, 'in_domain')

        if not os.path.exists(seed_path):
            print(f"⚠️  Skipping {dir_file} — in_domain folder not found at: {seed_path}")
            continue

        seed_path_files = os.listdir(seed_path)
        seed_value = int(re.search(r'\d+', dir_file).group())

        if seed_value not in seed_checkpoint_map:
            seed_checkpoint_map[seed_value] = []

        if seed_value not in metrics_dfs_by_seed:
            metrics_dfs_by_seed[seed_value] = []

        print(f"\n{'='*40}")
        print(f"SEED: {seed_value}")
        print(f"{'='*40}")

        for seed_path_file in sorted(seed_path_files):

            if "checkpoint" in seed_path_file:
                print(f"  ✓ Loading checkpoint: {seed_path_file}")
                checkpoint_path = os.path.join(seed_path, seed_path_file)
                df = DataProcessing.load_from_file(path=checkpoint_path)

                # Last column is the model prediction column (e.g. 'gpt-oss-120b')
                df[df.columns.to_list()[-1]] = df['llm_label'].values
                df.drop(columns=['llm_label', 'raw_response', 'llm_name'], inplace=True)
                seed_checkpoint_map[seed_value].append(df)

            if "metrics" in seed_path_file and "ml" not in seed_path_file and "llms" not in seed_path_file:
                print(f"  ✓ Loading metrics: {seed_path_file}")
                metrics_path = os.path.join(seed_path, seed_path_file)
                df = DataProcessing.load_from_file(path=metrics_path)
                metrics_dfs_by_seed[seed_value].append(df)

            if "x_y_test_set" in seed_path_file:
                print(f"  ✓ Loading test set: {seed_path_file}")
                x_test_path = os.path.join(seed_path, 'x_y_test_set.csv')
                x_test_df = DataProcessing.load_from_file(x_test_path)
                x_test_dfs.append(x_test_df)

    return seed_checkpoint_map, metrics_dfs_by_seed, x_test_dfs


def combine_checkpoints_per_seed(seed_checkpoint_map, dataset_folder_path):
    """
    Combine all model checkpoint DataFrames per seed into one wide DataFrame.
    Each combined DataFrame has: seed, original_index, text, <model_1>, <model_2>, ...

    Parameters
    ----------
    seed_checkpoint_map : dict
        {seed_value: [df1, df2, ...]} one DataFrame per model per seed.
    dataset_folder_path : str
        Base path used to construct the save path for each seed.

    Returns
    -------
    seeds_dfs : list of pd.DataFrame
        seeds_dfs[0] = seed3 combined, seeds_dfs[1] = seed7, seeds_dfs[2] = seed33.
    """
    print(f"\n{'='*40}")
    print("COMBINE CHECKPOINTS PER SEED")
    print(f"{'='*40}")

    seeds_dfs = []

    for seed_value in sorted(seed_checkpoint_map.keys()):
        seed_model_dfs = seed_checkpoint_map[seed_value]

        if not seed_model_dfs:
            print(f"⚠️  No checkpoint files found for seed {seed_value}. Skipping.")
            continue

        # Start with shared columns from first model df
        combined_df = seed_model_dfs[0][['seed', 'original_index', 'text']].copy()

        # Add one prediction column per model
        for df in seed_model_dfs:
            model_col = df.columns.to_list()[-1]
            combined_df[model_col] = df[model_col].values

        # Replace NaN with 0 and force int so labels are 0/1 not 0.0/1.0
        combined_df = combined_df.fillna(0)
        model_cols = [col for col in combined_df.columns if col not in ['seed', 'original_index', 'text']]
        combined_df[model_cols] = combined_df[model_cols].astype(int)

        print(f"\n✓ Seed {seed_value} combined shape: {combined_df.shape}")
        print(f"  Columns: {combined_df.columns.to_list()}")
        print(f"\n  Preview:\n{combined_df.head(3)}\n")

        # Save combined predictions for this seed alongside ML outputs
        save_path = os.path.join(dataset_folder_path, f'seed{seed_value}', 'in_domain')
        DataProcessing.save_to_file(
            data=combined_df,
            path=save_path,
            prefix='llm_classifiers_in_domain',
            save_file_type='csv',
            include_version=False
        )
        print(f"✓ Saved: {os.path.join(save_path, 'llm_classifiers_in_domain.csv')}")

        seeds_dfs.append(combined_df)

    print(f"\nTotal seeds combined: {len(seeds_dfs)}")
    if seeds_dfs:
        print(f"seeds_dfs[0] -> seed {sorted(seed_checkpoint_map.keys())[0]}")

    return seeds_dfs


def combine_and_save_metrics(metrics_dfs_by_seed, seed_checkpoint_map, dataset_folder_path):
    """
    Combine metrics from different LLM models within the same seed into one file.
    Mirrors the ML pipeline's metrics_summary_ml_models.csv structure where
    each seed has one file with one row per model.

    Parameters
    ----------
    metrics_dfs_by_seed : dict
        {seed_value: [df1, df2, ...]} one DataFrame per model per seed.
    seed_checkpoint_map : dict
        Used to get sorted seed values for saving to correct folders.
    dataset_folder_path : str
        Base path used to construct the save path for each seed.

    Returns
    -------
    dict
        {seed_value: combined_metrics_df} one combined DataFrame per seed.
    """
    print(f"\n{'='*40}")
    print("COMBINE METRICS PER SEED")
    print(f"{'='*40}")

    if not metrics_dfs_by_seed:
        print("⚠️  No metrics files found. Skipping metrics summary.")
        return None

    seed_metrics_combined = {}

    for seed_value in sorted(seed_checkpoint_map.keys()):
        seed_metrics = metrics_dfs_by_seed.get(seed_value, [])

        if not seed_metrics:
            print(f"⚠️  No metrics found for seed {seed_value}. Skipping.")
            continue

        # Stack all model metric rows for this seed into one DataFrame
        # One row per LLM model — mirrors metrics_summary_ml_models.csv
        combined_metrics_df = DataProcessing.concat_dfs(seed_metrics)

        print(f"\n✓ Seed {seed_value} metrics shape: {combined_metrics_df.shape}")
        print(f"\nPreview:\n{combined_metrics_df}\n")

        # Save alongside ML metrics for easy comparison
        save_metrics_path = os.path.join(dataset_folder_path, f'seed{seed_value}', 'in_domain')
        DataProcessing.save_to_file(
            data=combined_metrics_df,
            path=save_metrics_path,
            prefix='metrics_summary_llms',
            save_file_type='csv',
            include_version=False
        )
        print(f"✓ Saved metrics_summary_llms.csv to: {save_metrics_path}")

        seed_metrics_combined[seed_value] = combined_metrics_df

    return seed_metrics_combined


if __name__ == "__main__":
    """
    usage:
    python combine_llm_classifiers_output.py \
        --experiment synthetic-fpb-chronicle2050-yt-news-timebank_2026-04-22
    """
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_results_path = os.path.join(base_data_path, 'classification_results')

    parser = argparse.ArgumentParser(
        description='Combine LLM classifier checkpoint outputs across models and seeds.'
    )

    parser.add_argument(
        '--experiment',
        required=True,
        help='Experiment folder name inside classification_results/. '
             'Example: synthetic-fpb-chronicle2050-yt-news-timebank_2026-04-22'
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

    print("\n" + "="*40)
    print("COMBINE LLM CLASSIFIERS OUTPUT")
    print("="*40)
    print(f"Experiment: {args.experiment}")
    print(f"Path: {dataset_folder_path}")
    print(f"Seed folders found: {sorted([f for f in dir_files if 'seed' in f])}")

    # ============================================================
    # 3. LOAD SEED DATA
    # ============================================================
    seed_checkpoint_map, metrics_dfs_by_seed, x_test_dfs = load_seed_data(
        dataset_folder_path, dir_files
    )

    # ============================================================
    # 4. COMBINE CHECKPOINTS PER SEED
    # ============================================================
    seeds_dfs = combine_checkpoints_per_seed(seed_checkpoint_map, dataset_folder_path)

    # ============================================================
    # 5. COMBINE AND SAVE METRICS PER SEED
    # ============================================================
    seed_metrics_combined = combine_and_save_metrics(
        metrics_dfs_by_seed, seed_checkpoint_map, dataset_folder_path
    )

    # ============================================================
    # 6. COMPLETE
    # ============================================================
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("="*40)
    print(f"Experiment: {args.experiment}")
    print(f"Seeds processed: {sorted(seed_checkpoint_map.keys())}")
    print(f"seeds_dfs length: {len(seeds_dfs)}")
    if seed_metrics_combined:
        print(f"Metrics saved for seeds: {sorted(seed_metrics_combined.keys())}")
    print()