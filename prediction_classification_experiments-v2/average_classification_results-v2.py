# Before this, run python3 create_combined_dataset.py to create dataset
import os
import sys
import json
import joblib
import argparse
import matplotlib
matplotlib.use('Agg')  # Prevent GUI windows from opening
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime

# Get the current working directory of the script
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from data_visualizing import DataVisualizing
from feature_extraction import SpacyFeatureExtraction
from classification_models import SkLearnModelFactory
from explainability import Explainability


def get_latest_seed_version(experiment_dir, base_seed):
    """
    Find the latest version of a seed folder.
    
    Parameters
    ----------
    experiment_dir : str
        Path to experiment directory
    base_seed : int
        Base seed number (e.g., 42)
    
    Returns
    -------
    str or None
        Name of latest seed folder (e.g., 'seed42_v2' or 'seed42'), or None if not found
    
    Notes
    -----
    Handles versioned seeds: seed42_v2 > seed42_v1 > seed42
    Version number 0 is implicit for non-versioned folders.
    
    Examples
    --------
    >>> get_latest_seed_version('/path/to/experiment', 42)
    'seed42_v2'  # Returns latest version found
    """
    base_seed_str = str(base_seed)
    seed_pattern = f"seed{base_seed_str}"
    
    # Find all folders that start with this seed pattern
    all_folders = []
    items = os.listdir(experiment_dir)
    for item in items:
        item_path = os.path.join(experiment_dir, item)
        is_directory = os.path.isdir(item_path)
        starts_with_seed = item.startswith(seed_pattern)
        
        if is_directory and starts_with_seed:
            all_folders.append(item)
    
    # If no seed folders found, return None
    if len(all_folders) == 0:
        return None
    
    # Extract version numbers from each folder
    versioned_folders = []
    for folder in all_folders:
        if folder == seed_pattern:
            # No version suffix means version 0
            version = 0
            versioned_folders.append((version, folder))
        elif '_v' in folder:
            # Try to extract version number from seed42_v1 format
            try:
                version_part = folder.split('_v')[-1]
                version = int(version_part)
                versioned_folders.append((version, folder))
            except ValueError:
                # Skip if version number is not valid
                continue
    
    # If no valid versioned folders found, return None
    if len(versioned_folders) == 0:
        return None
    
    # Find the folder with highest version number
    max_version = -1
    latest_folder = None
    for version, folder_name in versioned_folders:
        if version > max_version:
            max_version = version
            latest_folder = folder_name
    
    return latest_folder


def collect_seed_results(seed_path, seed_folder_name, mode, external_dataset):
    """
    Collect in-domain and external test results for a single seed.
    
    Parameters
    ----------
    seed_path : str
        Full path to seed folder
    seed_folder_name : str
        Name of seed folder (e.g., 'seed42' or 'seed42_v1')
    mode : str
        Type of results to collect: 'in_domain', 'external', or 'both'
    external_dataset : str
        Filter for external datasets: 'fpb', 'chronicle2050', or 'all'
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'in_domain': pd.DataFrame or None (in-domain test metrics)
        - 'external': dict of {dataset_name: pd.DataFrame} (external test metrics)
        - 'folder_name': str (name of seed folder)
    
    Notes
    -----
    In-domain results are loaded from: seed_path/metrics_summary.csv
    External results are loaded from: seed_path/external_*/metrics_summary.csv
    
    The external_dataset filter determines which external folders to include.
    """
    # Initialize result structure
    results = {
        'in_domain': None,
        'external': {},
        'folder_name': seed_folder_name
    }
    
    # ============================================================
    # COLLECT IN-DOMAIN RESULTS
    # ============================================================
    mode_includes_in_domain = (mode == 'in_domain' or mode == 'both')
    
    if mode_includes_in_domain:
        in_domain_metrics_path = os.path.join(seed_path, 'metrics_summary.csv')
        file_exists = os.path.exists(in_domain_metrics_path)
        
        if file_exists:
            in_domain_df = DataProcessing.load_from_file(in_domain_metrics_path, 'csv', sep=',')
            results['in_domain'] = in_domain_df
            print(f"    ✓ Loaded in-domain: {seed_folder_name}/metrics_summary.csv")
            print(f"      → This is E7-style: standard train/val/test split on same dataset")
            print(f"      → OR training used --no_test_split but still tested on in-domain validation set")
            print()
        else:
            print(f"    ✗ Missing in-domain: {seed_folder_name}/metrics_summary.csv")
            print(f"      → This is expected for E1-E6: cross-domain evaluation only")
            print(f"      → Training used --no_test_split and tested ONLY on --test_datasets (external)")
            print(f"      → If you expected in-domain results: check training used --no_test_split incorrectly")
            print()
    
    # ============================================================
    # COLLECT EXTERNAL TEST RESULTS
    # ============================================================
    mode_includes_external = (mode == 'external' or mode == 'both')
    
    if mode_includes_external:
        # Find all external_* subfolders in this seed folder
        items = os.listdir(seed_path)
        external_folders = []
        
        for item in items:
            item_path = os.path.join(seed_path, item)
            is_directory = os.path.isdir(item_path)
            is_external = item.startswith('external_')
            
            if is_directory and is_external:
                external_folders.append(item)
        
        # Process each external folder
        for ext_folder in external_folders:
            # Extract dataset name (e.g., external_fpb-maya-binary... → fpb-maya-binary...)
            dataset_name = ext_folder.replace('external_', '')
            
            # Apply external_dataset filter
            should_include = False
            
            if external_dataset == 'all':
                should_include = True
            elif external_dataset == 'fpb':
                if dataset_name.startswith('fpb'):
                    should_include = True
            elif external_dataset == 'chronicle2050':
                if dataset_name.startswith('chronicle2050'):
                    should_include = True
            
            if not should_include:
                continue
            
            # Load external test metrics
            ext_metrics_path = os.path.join(seed_path, ext_folder, 'metrics_summary.csv')
            file_exists = os.path.exists(ext_metrics_path)
            
            if file_exists:
                ext_df = DataProcessing.load_from_file(ext_metrics_path, 'csv', sep=',')
                results['external'][dataset_name] = ext_df
                print(f"    ✓ Loaded external: {seed_folder_name}/{ext_folder}/metrics_summary.csv")
                print(f"      → Training used --test_datasets flag with external dataset: {dataset_name}")
                print(f"      → This is E1-E6-style: testing on held-out external dataset for cross-domain evaluation")
                print()
            else:
                print(f"    ✗ Missing external: {seed_folder_name}/{ext_folder}/metrics_summary.csv")
                print(f"      → This is expected for E7: standard train/val/test split (no external test)")
                print(f"      → Training did NOT use --test_datasets flag")
                print(f"      → If you expected external results: check training command included --test_datasets")
                print()
    
    return results


def collect_all_results(base_results_dir, experiments_filter, mode, external_dataset):
    """
    Collect all results from experiment folders based on filter criteria.
    
    Parameters
    ----------
    base_results_dir : str
        Path to classification_results directory
    experiments_filter : list of str or None
        List of specific experiment folder names to process, or None to process all
    mode : str
        Type of results to collect: 'in_domain', 'external', or 'both'
    external_dataset : str
        Filter for external datasets: 'fpb', 'chronicle2050', or 'all'
    
    Returns
    -------
    dict
        Nested dictionary structure:
        {
            experiment_name: {
                seed_num: {
                    'in_domain': pd.DataFrame or None,
                    'external': {dataset_name: pd.DataFrame},
                    'folder_name': str
                }
            }
        }
    
    Notes
    -----
    This function orchestrates the entire collection process:
    1. Determines which experiment folders to process
    2. Finds all seed folders within each experiment
    3. Collects in-domain and external results for each seed
    4. Returns organized results ready for aggregation
    """
    print("\n" + "="*40)
    print("COLLECTING RESULTS")
    print("="*40)
    
    # Structure to hold all results
    all_results = {}
    
    # ============================================================
    # DETERMINE WHICH EXPERIMENTS TO PROCESS
    # ============================================================
    if experiments_filter is not None:
        # Use the provided list of experiment names
        experiment_folders = experiments_filter
    else:
        # Get all experiment folders (exclude utility folders)
        all_items = os.listdir(base_results_dir)
        experiment_folders = []
        
        exclude_folders = ['averaged_results', 'cross_dataset_comparisons', 'cross_domain_analysis']
        
        for item in all_items:
            item_path = os.path.join(base_results_dir, item)
            is_directory = os.path.isdir(item_path)
            is_hidden = item.startswith('.')
            is_excluded = item in exclude_folders
            
            if is_directory and not is_hidden and not is_excluded:
                experiment_folders.append(item)
    
    num_experiments = len(experiment_folders)
    print(f"Found {num_experiments} experiment(s) to process\n")
    
    # ============================================================
    # PROCESS EACH EXPERIMENT
    # ============================================================
    for exp_folder in sorted(experiment_folders):
        exp_path = os.path.join(base_results_dir, exp_folder)
        
        if not os.path.isdir(exp_path):
            print(f"⚠️  Skipping {exp_folder} - not a directory")
            continue
        
        print(f"Processing: {exp_folder}")
        all_results[exp_folder] = {}
        
        # Find all seed folders in this experiment
        items = os.listdir(exp_path)
        seed_folders = []
        
        for item in items:
            item_path = os.path.join(exp_path, item)
            is_directory = os.path.isdir(item_path)
            is_seed_folder = item.startswith('seed')
            
            if is_directory and is_seed_folder:
                seed_folders.append(item)
        
        # Extract unique seed numbers (handle versioned seeds like seed42_v1)
        unique_seeds = set()
        for folder in seed_folders:
            parts = folder.split('_')
            base_seed_with_prefix = parts[0]
            base_seed_str = base_seed_with_prefix.replace('seed', '')
            
            try:
                seed_num = int(base_seed_str)
                unique_seeds.add(seed_num)
            except ValueError:
                continue
        
        # Convert set to sorted list
        unique_seeds_list = sorted(list(unique_seeds))
        print(f"  Found seeds: {unique_seeds_list} to average and get standard deviation.")
        
        # For each seed, get latest version and collect results
        for seed_num in unique_seeds_list:
            print(f"    Seed: {seed_num}")
            # Get latest version of this seed
            latest_seed_folder = get_latest_seed_version(exp_path, seed_num)
            
            if latest_seed_folder is None:
                continue
            
            seed_path = os.path.join(exp_path, latest_seed_folder)
            
            # Collect results for this seed
            seed_results = collect_seed_results(seed_path, latest_seed_folder, mode, external_dataset)
            all_results[exp_folder][seed_num] = seed_results
        
        print()
    
    # ============================================================
    # VALIDATE COLLECTED RESULTS
    # ============================================================
    print(f"\n{'='*40}")
    print("COLLECTION SUMMARY")
    print(f"{'='*40}")
    
    total_in_domain = 0
    total_external = 0
    external_datasets_found = set()
    
    for exp_name in all_results:
        seeds = all_results[exp_name]
        for seed_num in seeds:
            results = seeds[seed_num]
            
            if results['in_domain'] is not None:
                total_in_domain = total_in_domain + 1
            
            external_count = len(results['external'])
            total_external = total_external + external_count
            
            for dataset_name in results['external']:
                external_datasets_found.add(dataset_name)
    
    # Convert set to sorted list for display
    external_datasets_list = sorted(list(external_datasets_found))
    
    num_experiments_processed = len(all_results)
    print(f"Experiments processed: {num_experiments_processed}")
    print(f"In-domain results found: {total_in_domain}")
    print(f"External results found: {total_external}")
    
    num_external_datasets = len(external_datasets_found)
    if num_external_datasets > 0:
        print(f"External datasets found: {external_datasets_list}")
    print()
    
    # Exit if no results found
    if total_in_domain == 0 and total_external == 0:
        print("❌ No results found to aggregate. Exiting.")
        sys.exit(0)
    
    return all_results


def average_across_seeds(experiment_results):
    """
    Average metrics across seeds for one experiment.
    
    Parameters
    ----------
    experiment_results : dict
        Dictionary of seed results for one experiment:
        {seed_num: {'in_domain': df, 'external': {dataset: df}}}
    
    Returns
    -------
    tuple
        (mean_df, std_df, n_seeds) where:
        - mean_df: DataFrame with mean values across seeds
        - std_df: DataFrame with standard deviation across seeds
        - n_seeds: Number of seeds averaged
    
    Notes
    -----
    Combines all DataFrames from different seeds, groups by model,
    and calculates mean and std for numeric columns.
    """
    # Collect all DataFrames
    all_dfs = []
    seed_info = []
    
    for seed_num in experiment_results:
        seed_data = experiment_results[seed_num]
        df = seed_data  # This is a DataFrame
        
        if df is not None:
            all_dfs.append(df)
            seed_info.append(seed_num)
    
    # If no data found, return None
    if len(all_dfs) == 0:
        return None, None, 0
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Find numeric columns
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    numeric_cols_list = list(numeric_cols)
    
    # Group by model and calculate mean
    mean_df = combined_df.groupby('model')[numeric_cols_list].mean()
    
    # Group by model and calculate std
    std_df = combined_df.groupby('model')[numeric_cols_list].std()
    
    n_seeds = len(all_dfs)
    
    return mean_df, std_df, n_seeds


def aggregate_in_domain_results(all_results, base_results_dir):
    """
    Aggregate in-domain results across seeds for each experiment.
    
    Parameters
    ----------
    all_results : dict
        All collected results from collect_all_results()
    base_results_dir : str
        Base directory for saving results
    
    Returns
    -------
    dict
        {experiment_name: {'mean': df, 'std': df, 'n_seeds': int}}
    
    Notes
    -----
    For each experiment:
    1. Collects all in-domain results across seeds
    2. Averages metrics
    3. Saves to: experiment/averaged/in_domain/
    """
    print("\n" + "="*40)
    print("AGGREGATING IN-DOMAIN RESULTS")
    print("="*40)
    
    in_domain_summaries = {}
    
    for exp_name in all_results:
        exp_results = all_results[exp_name]
        
        # Collect in-domain results for this experiment
        in_domain_data = {}
        for seed_num in exp_results:
            seed_data = exp_results[seed_num]
            if seed_data['in_domain'] is not None:
                in_domain_data[seed_num] = seed_data['in_domain']
        
        # Skip if no in-domain results
        if len(in_domain_data) == 0:
            print(f"Skipping {exp_name}: no in-domain results found")
            continue
        
        print(f"\nAveraging {exp_name}:")
        print(f"  Seeds: {sorted(list(in_domain_data.keys()))}")
        
        # Average across seeds
        mean_df, std_df, n_seeds = average_across_seeds(in_domain_data)
        
        if mean_df is None:
            continue
        
        print(f"  Averaged {n_seeds} seed(s)")
        
        # Save results
        exp_path = os.path.join(base_results_dir, exp_name)
        save_dir = os.path.join(exp_path, 'averaged', 'in_domain')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save mean and std
        mean_df.to_csv(os.path.join(save_dir, 'mean.csv'))
        std_df.to_csv(os.path.join(save_dir, 'std.csv'))
        
        # Save combined mean ± std
        combined_df = mean_df.copy()
        combined_df = combined_df.reset_index()
        
        for col in combined_df.columns:
            if col != 'model':
                std_values = std_df.reset_index()[col]
                combined_df[col] = combined_df[col].apply(lambda x: f"{x:.4f}") + " ± " + std_values.apply(lambda x: f"{x:.4f}")
        
        combined_df.to_csv(os.path.join(save_dir, 'mean_std.csv'), index=False)
        
        print(f"  ✓ Saved to: {save_dir}/")
        
        # Store for later use
        in_domain_summaries[exp_name] = {
            'mean': mean_df,
            'std': std_df,
            'n_seeds': n_seeds
        }
    
    return in_domain_summaries


def aggregate_external_results(all_results, base_results_dir):
    """
    Aggregate external test results across seeds for each experiment.
    
    Parameters
    ----------
    all_results : dict
        All collected results from collect_all_results()
    base_results_dir : str
        Base directory for saving results
    
    Returns
    -------
    dict
        {experiment_name: {dataset_name: {'mean': df, 'std': df, 'n_seeds': int}}}
    
    Notes
    -----
    For each experiment and external dataset:
    1. Collects all external results across seeds
    2. Averages metrics
    3. Saves to: experiment/averaged/external_{dataset_name}/
    """
    print("\n" + "="*40)
    print("AGGREGATING EXTERNAL RESULTS")
    print("="*40)
    
    external_summaries = {}
    
    for exp_name in all_results:
        exp_results = all_results[exp_name]
        external_summaries[exp_name] = {}
        
        # Find all external datasets for this experiment
        all_external_datasets = set()
        for seed_num in exp_results:
            seed_data = exp_results[seed_num]
            for dataset_name in seed_data['external']:
                all_external_datasets.add(dataset_name)
        
        # Skip if no external results
        if len(all_external_datasets) == 0:
            print(f"Skipping {exp_name}: no external results found")
            continue
        
        print(f"\nProcessing {exp_name}:")
        
        # For each external dataset
        for dataset_name in sorted(all_external_datasets):
            # Collect external results for this dataset across seeds
            external_data = {}
            for seed_num in exp_results:
                seed_data = exp_results[seed_num]
                if dataset_name in seed_data['external']:
                    external_data[seed_num] = seed_data['external'][dataset_name]
            
            seeds_used = sorted(list(external_data.keys()))
            print(f"  {dataset_name}:")
            print(f"    Seeds: {seeds_used}")
            
            # Average across seeds
            mean_df, std_df, n_seeds = average_across_seeds(external_data)
            
            if mean_df is None:
                continue
            
            print(f"    Averaged {n_seeds} seed(s)")
            
            # Save results
            exp_path = os.path.join(base_results_dir, exp_name)
            save_dir = os.path.join(exp_path, 'averaged', f'external_{dataset_name}')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save mean and std
            mean_df.to_csv(os.path.join(save_dir, 'mean.csv'))
            std_df.to_csv(os.path.join(save_dir, 'std.csv'))
            
            # Save combined mean ± std
            combined_df = mean_df.copy()
            combined_df = combined_df.reset_index()
            
            for col in combined_df.columns:
                if col != 'model':
                    std_values = std_df.reset_index()[col]
                    combined_df[col] = combined_df[col].apply(lambda x: f"{x:.4f}") + " ± " + std_values.apply(lambda x: f"{x:.4f}")
            
            combined_df.to_csv(os.path.join(save_dir, 'mean_std.csv'), index=False)
            
            print(f"    ✓ Saved to: {save_dir}/")
            
            # Store for later use
            external_summaries[exp_name][dataset_name] = {
                'mean': mean_df,
                'std': std_df,
                'n_seeds': n_seeds
            }
    
    return external_summaries


def print_latex_tables(in_domain_summaries, external_summaries):
    """
    Print LaTeX-formatted tables for paper.
    
    Parameters
    ----------
    in_domain_summaries : dict
        Aggregated in-domain results
    external_summaries : dict
        Aggregated external results
    
    Notes
    -----
    Prints separate tables for:
    1. In-domain results (E7) per experiment
    2. External results (E1-E6) per experiment per external dataset
    """
    print("\n" + "="*40)
    print("LATEX TABLES")
    print("="*40)
    
    # ============================================================
    # IN-DOMAIN TABLES
    # ============================================================
    if len(in_domain_summaries) > 0:
        print("\n% IN-DOMAIN RESULTS (E7-style)\n")
        
        for exp_name in sorted(in_domain_summaries.keys()):
            summary = in_domain_summaries[exp_name]
            mean_df = summary['mean']
            std_df = summary['std']
            n_seeds = summary['n_seeds']
            
            print(f"% {exp_name}")
            print(f"% Seeds: {n_seeds}\n")
            
            # Create combined table
            latex_df = mean_df.copy()
            latex_df = latex_df.reset_index()
            
            # Select key columns for paper
            key_cols = ['model', 'test_accuracy', 'precision_class_1', 'recall_class_1', 'f1_class_1', 'roc_auc']
            available_cols = []
            for col in key_cols:
                if col in latex_df.columns:
                    available_cols.append(col)
            
            if len(available_cols) > 1:
                latex_df = latex_df[available_cols]
                
                # Format mean ± std
                for col in latex_df.columns:
                    if col != 'model':
                        std_values = std_df.reset_index()[col]
                        latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}") + " $\\pm$ " + std_values.apply(lambda x: f"{x:.4f}")
                
                print(latex_df.to_latex(index=False, escape=False))
            print()
    
    # ============================================================
    # EXTERNAL TABLES
    # ============================================================
    if len(external_summaries) > 0:
        print("\n% EXTERNAL RESULTS (E1-E6-style)\n")
        
        for exp_name in sorted(external_summaries.keys()):
            datasets = external_summaries[exp_name]
            
            for dataset_name in sorted(datasets.keys()):
                summary = datasets[dataset_name]
                mean_df = summary['mean']
                std_df = summary['std']
                n_seeds = summary['n_seeds']
                
                print(f"% {exp_name} → {dataset_name}")
                print(f"% Seeds: {n_seeds}\n")
                
                # Create combined table
                latex_df = mean_df.copy()
                latex_df = latex_df.reset_index()
                
                # Select key columns
                key_cols = ['model', 'test_accuracy', 'precision_class_1', 'recall_class_1', 'f1_class_1', 'roc_auc']
                available_cols = []
                for col in key_cols:
                    if col in latex_df.columns:
                        available_cols.append(col)
                
                if len(available_cols) > 1:
                    latex_df = latex_df[available_cols]
                    
                    # Format mean ± std
                    for col in latex_df.columns:
                        if col != 'model':
                            std_values = std_df.reset_index()[col]
                            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}") + " $\\pm$ " + std_values.apply(lambda x: f"{x:.4f}")
                    
                    print(latex_df.to_latex(index=False, escape=False))
                print()


if __name__ == "__main__":
    
    print("\n" + "="*40)
    print("AVERAGE CLASSIFICATION RESULTS - V2")
    print("="*40)
    
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    base_results_dir = os.path.join(base_data_path, 'classification_results/')
    
    parser = argparse.ArgumentParser(
        description='Average classification results across seed runs and optionally compare across datasets/domains.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # E7: Aggregate in-domain test results only (standard train/val/test split)
  python average_classification_results-v2.py --mode in_domain
  
  # E1-E6: Aggregate external test results only (cross-domain evaluation)
  python average_classification_results-v2.py --mode external --external_dataset fpb
  
  # Both E7 and E1-E6: Aggregate in-domain AND external test results
  python average_classification_results-v2.py --mode both
  
  # Process one specific experiment
  python average_classification_results-v2.py --experiment_folders combined-full_synthetic-v1_2026-03-17
  
  # Process multiple specific experiments
  python average_classification_results-v2.py --experiment_folders exp1_2026-03-17 exp2_2026-03-16
        """
    )
    
     # mode
    parser.add_argument(
        '--mode', 
        default='both', 
        choices=['in_domain', 'external', 'both'],
        help='Type of test results to aggregate. '
             'in_domain = E7-style: test set from standard train/val/test split on same dataset '
             '(note: val set used during training for hyperparameter tuning, not aggregated here). '
             'external = E1-E6-style: results from testing on held-out external datasets '
             '(specified during training via --test_datasets flag in ml_train.py). '
             'both = aggregate in-domain test results AND external test results (if both exist). '
             'Default: both'
    )
    
    # external_dataset
    parser.add_argument(
        '--external_dataset', 
        default='all', 
        choices=['fpb', 'chronicle2050', 'all'],
        help='Filter which external test dataset to aggregate (only applies when mode=external or both). '
             'fpb = only aggregate external_fpb-* folders (Financial PhraseBank results). '
             'chronicle2050 = only aggregate external_chronicle2050-* folders. '
             'all = aggregate all external_* folders found (recommended when adding new datasets). '
             'Note: If you add new external test datasets (e.g., reuters, semeval), update the choices list '
             'to enable filtering by name, or use "all" to automatically include them. '
             'Ignored when mode=in_domain since no external datasets involved. '
             'Default: all'
    )
    
    # experiment_folders
    parser.add_argument(
        '--experiment_folders', 
        nargs='+',
        default=None,
        help='Specify which experiment folder(s) to process. '
             'Single experiment: --experiment_folders combined-full_synthetic-v1_2026-03-17. '
             'Multiple experiments: --experiment_folders exp1_2026-03-17 exp2_2026-03-16 exp3_2026-03-15. '
             'If not specified: processes ALL experiment folders in classification_results/. '
             'Use this to filter specific experiments instead of processing everything.'
    )
    
    args = parser.parse_args()
    
    # ============================================================
    # 1b. VALIDATION
    # ============================================================
    if args.external_dataset != 'all' and args.mode == 'in_domain':
        print("\n⚠️  Warning: --external_dataset ignored when mode=in_domain (no external datasets to filter)")
    
    print("\n" + "="*40)
    print("CONFIGURATION")
    print("="*40)
    print(f"Mode: {args.mode}")
    if args.mode in ['external', 'both']:
        print(f"External dataset filter: {args.external_dataset}")
    if args.experiment_folders:
        print(f"Experiments to process: {len(args.experiment_folders)}")
        for exp in args.experiment_folders:
            print(f"  - {exp}")
    else:
        print(f"Experiments to process: ALL (no filter)")
    print()
    
    # ============================================================
    # 2. COLLECT RESULTS
    # ============================================================
    all_results = collect_all_results(
        base_results_dir, 
        args.experiment_folders,
        args.mode,
        args.external_dataset
    )
    
    # ============================================================
    # 3. FILTER BY MODE (already handled in collection)
    # ============================================================
    # Filtering is already done in collect_all_results() and collect_seed_results()
    # based on args.mode and args.external_dataset
    
    # ============================================================
    # 4. AGGREGATE IN-DOMAIN RESULTS (if applicable)
    # ============================================================
    in_domain_summaries = {}
    
    mode_includes_in_domain = (args.mode == 'in_domain' or args.mode == 'both')
    if mode_includes_in_domain:
        in_domain_summaries = aggregate_in_domain_results(all_results, base_results_dir)
    else:
        print("\n⚠️  Skipping in-domain aggregation (mode=external)")
    
    # ============================================================
    # 5. AGGREGATE EXTERNAL RESULTS (if applicable)
    # ============================================================
    external_summaries = {}
    
    mode_includes_external = (args.mode == 'external' or args.mode == 'both')
    if mode_includes_external:
        external_summaries = aggregate_external_results(all_results, base_results_dir)
    else:
        print("\n⚠️  Skipping external aggregation (mode=in_domain)")
    
    # ============================================================
    # 6. CROSS-DATASET COMPARISON (if multiple experiments)
    # ============================================================
    # TODO: Implement cross-dataset comparison
    # Compare same model across different training datasets
    # Example: "logistic_regression trained on synthetic vs fpb vs combined"
    # This would require multiple experiments with comparable model names
    # Save to: cross_dataset_comparisons/run_{timestamp}/
    
    num_experiments = len(all_results)
    if num_experiments >= 2:
        print("\n" + "="*40)
        print("CROSS-DATASET COMPARISON")
        print("="*40)
        print("⚠️  Cross-dataset comparison not yet implemented")
        print("    This feature will compare same models trained on different datasets")
        print(f"    Found {num_experiments} experiments that could be compared")
        print()
    
    # ============================================================
    # 7. CROSS-DOMAIN ANALYSIS (if external results exist)
    # ============================================================
    # TODO: Implement cross-domain transfer analysis
    # Analyze transfer: train on X, test on Y
    # Calculate domain transfer gaps (in-domain accuracy vs external accuracy)
    # Save to: cross_domain_analysis/run_{timestamp}/
    
    has_external_results = False
    for exp_name in external_summaries:
        if len(external_summaries[exp_name]) > 0:
            has_external_results = True
            break
    
    if has_external_results:
        print("\n" + "="*40)
        print("CROSS-DOMAIN ANALYSIS")
        print("="*40)
        print("⚠️  Cross-domain transfer analysis not yet implemented")
        print("    This feature will analyze performance gaps between in-domain and external datasets")
        print()
    
    # ============================================================
    # 8. PRINT LATEX TABLES
    # ============================================================
    print_latex_tables(in_domain_summaries, external_summaries)
    
    # ============================================================
    # 9. SUMMARY
    # ============================================================
    print("\n" + "="*40)
    print("AVERAGING COMPLETE")
    print("="*40)
    print(f"Mode: {args.mode}")
    
    num_experiments_processed = len(all_results)
    print(f"Total experiments processed: {num_experiments_processed}")
    
    if args.mode == 'in_domain' or args.mode == 'both':
        num_in_domain = len(in_domain_summaries)
        print(f"In-domain results averaged: {num_in_domain}")
    
    if args.mode == 'external' or args.mode == 'both':
        # Count total external datasets processed
        external_datasets_found = set()
        num_external_averaged = 0
        
        for exp_name in external_summaries:
            for dataset_name in external_summaries[exp_name]:
                external_datasets_found.add(dataset_name)
                num_external_averaged += 1
        
        external_datasets_list = sorted(list(external_datasets_found))
        print(f"External datasets found: {external_datasets_list}")
        print(f"External results averaged: {num_external_averaged}")
    
    print(f"\nResults saved to respective experiment folders:")
    print(f"  In-domain: experiment/averaged/in_domain/")
    print(f"  External: experiment/averaged/external_{{dataset}}/")
    
    if num_experiments_processed >= 2:
        print(f"\n⚠️  Cross-dataset comparison available but not yet implemented")
    
    if has_external_results:
        print(f"⚠️  Cross-domain transfer analysis available but not yet implemented")
    
    print()