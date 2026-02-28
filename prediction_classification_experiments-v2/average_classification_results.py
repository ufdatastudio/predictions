# average_classification_results.py
import os
import re
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing


def get_latest_seed_version(experiment_dir, base_seed):
    """
    Find the latest version of a seed folder.
    
    Parameters
    ----------
    experiment_dir : str
        Path to experiment directory
    base_seed : str or int
        Base seed number (e.g., 33 or 40)
    
    Returns
    -------
    str or None
        Name of latest seed folder (e.g., 'seed33_v2'), or None if not found
    """
    base_seed_str = str(base_seed)
    seed_pattern = f"seed{base_seed_str}"
    
    # Find all folders matching this seed
    all_folders = []
    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if os.path.isdir(item_path) and item.startswith(seed_pattern):
            all_folders.append(item)
    
    if not all_folders:
        return None
    
    # Extract version numbers
    versioned_folders = []
    for folder in all_folders:
        if folder == seed_pattern:
            versioned_folders.append((0, folder))
        elif '_v' in folder:
            try:
                version = int(folder.split('_v')[-1])
                versioned_folders.append((version, folder))
            except ValueError:
                continue
    
    if not versioned_folders:
        return None
    
    # Return folder with highest version
    latest_version, latest_folder = max(versioned_folders, key=lambda x: x[0])
    return latest_folder


def collect_results(results_dir):
    """
    Collect all metrics_summary.csv files and group by experiment.
    For each seed, uses the latest version.
    
    Parameters
    ----------
    results_dir : str
        Path to classification_results directory
    
    Returns
    -------
    dict
        {experiment_name: [{'seed': int, 'folder': str, 'data': pd.DataFrame}, ...]}
    """
    experiments = {}
    
    print(f"\n{'='*60}")
    print("COLLECTING RESULTS")
    print(f"{'='*60}\n")
    
    # Find all experiment directories
    experiment_dirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item != 'averaged_results' and not item.startswith('.'):
            if re.search(r'\d{4}-\d{2}-\d{2}', item):
                experiment_dirs.append(item)
    
    print(f"Found {len(experiment_dirs)} experiment directories\n")
    
    for exp_dir_name in sorted(experiment_dirs):
        exp_dir_path = os.path.join(results_dir, exp_dir_name)
        
        print(f"Processing: {exp_dir_name}")
        
        # Find all unique seeds
        seed_folders = os.listdir(exp_dir_path)
        unique_seeds = set()
        
        for folder in seed_folders:
            if folder.startswith('seed') and os.path.isdir(os.path.join(exp_dir_path, folder)):
                base_seed = folder.split('_')[0].replace('seed', '')
                try:
                    unique_seeds.add(int(base_seed))
                except ValueError:
                    continue
        
        print(f"  Found unique seeds: {sorted(unique_seeds)}")
        
        # For each unique seed, get the latest version
        for seed in sorted(unique_seeds):
            latest_folder = get_latest_seed_version(exp_dir_path, seed)
            
            if latest_folder:
                seed_folder_path = os.path.join(exp_dir_path, latest_folder)
                csv_file = os.path.join(seed_folder_path, 'metrics_summary.csv')
                
                if os.path.exists(csv_file):
                    if exp_dir_name not in experiments:
                        experiments[exp_dir_name] = []
                    
                    df = DataProcessing.load_from_file(csv_file, 'csv', sep=',')
                    experiments[exp_dir_name].append({
                        'seed': seed,
                        'folder': latest_folder,
                        'data': df
                    })
                    print(f"    ✓ Loaded: {latest_folder}/metrics_summary.csv")
                else:
                    print(f"    ✗ Missing: {latest_folder}/metrics_summary.csv")
        
        print()
    
    return experiments


def average_experiment_results(experiment_data):
    """
    Average metrics across seeds, grouped by model.
    
    Parameters
    ----------
    experiment_data : list
        List of dicts containing seed info and data
    
    Returns
    -------
    tuple
        (mean_df, std_df, n_seeds)
    """
    if len(experiment_data) == 0:
        return None, None, 0
    
    # Concatenate all dataframes
    all_dfs = [item['data'] for item in experiment_data]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Group by model and compute mean/std on numeric columns only
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    
    mean_df = combined_df.groupby('model')[numeric_cols].mean()
    mean_df.loc[-1] = mean_df.mean()
    mean_df = mean_df.rename(index={-1: 'mean_row'})
    
    std_df = combined_df.groupby('model')[numeric_cols].std()
    std_df.loc[-1] = std_df.std()
    std_df = std_df.rename(index={-1: 'std_row'})
    
    n_seeds = len(all_dfs)
    
    return mean_df, std_df, n_seeds


def compute_cross_dataset_margins(summaries):
    """
    Compute margins for same model across datasets
    and accuracy distribution per dataset.
    
    Parameters
    ----------
    summaries : list
        List of summary dicts from save_averaged_results()
        Each dict contains: experiment, version, n_seeds, seed_info, mean, std
    
    Returns
    -------
    tuple
        (model_margins_df, dataset_accuracy_df)
        - model_margins_df: mean ± std per model across all 3 datasets
        - dataset_accuracy_df: accuracy stats per dataset across all models
    
    Notes
    -----
    Identifies dataset type from experiment name:
    - 'imbalanced' in name → imbalanced dataset
    - 'oversampled' in name → oversampled dataset
    - 'undersampled' in name → undersampled dataset
    
    Example for Perceptron accuracy:
        Imbalanced:   0.8708
        Oversampled:  0.8276
        Undersampled: 0.7554
        Mean: 0.8179 ± 0.0493
    """
    print(f"\n{'='*50}")
    print("CROSS-DATASET MARGINS")
    print(f"{'='*50}\n")
    
    # ============================================================
    # ORGANIZE DATA BY DATASET TYPE
    # ============================================================
    dataset_means = {}
    for summary in summaries:
        exp_name = summary['experiment']
        mean_df = summary['mean']
        
        # Identify dataset type from experiment name
        if 'imbalanced' in exp_name:
            dataset_type = 'imbalanced'
        elif 'oversampled' in exp_name:
            dataset_type = 'oversampled'
        elif 'undersampled' in exp_name:
            dataset_type = 'undersampled'
        else:
            dataset_type = exp_name  # Fallback to full name
        
        dataset_means[dataset_type] = mean_df
    
    print(f"Datasets found: {list(dataset_means.keys())}\n")
    
    # ============================================================
    # SAME MODEL ACROSS DATASETS (MEAN ± STD)
    # ============================================================
    model_margins = []
    
    # Get all unique models (exclude mean_row)
    all_models = set()
    for mean_df in dataset_means.values():
        all_models.update(mean_df.index.tolist())
    all_models = sorted([m for m in all_models if m != 'mean_row'])
    
    for model in all_models:
        row = {'model': model}
        
        # Collect metric values per dataset for this model
        for dataset_type, mean_df in dataset_means.items():
            if model in mean_df.index:
                row[f'{dataset_type}_accuracy'] = mean_df.loc[model, 'accuracy']
                row[f'{dataset_type}_f1_class_1'] = mean_df.loc[model, 'f1_class_1']
                row[f'{dataset_type}_auc'] = mean_df.loc[model, 'auc']
        
        # Compute mean ± std across datasets for each metric
        # e.g., Perceptron accuracy: mean(0.8708, 0.8276, 0.7554) = 0.8179 ± 0.0493
        accuracy_vals = [row[f'{d}_accuracy'] for d in dataset_means.keys() if f'{d}_accuracy' in row]
        f1_vals = [row[f'{d}_f1_class_1'] for d in dataset_means.keys() if f'{d}_f1_class_1' in row]
        auc_vals = [row[f'{d}_auc'] for d in dataset_means.keys() if f'{d}_auc' in row]
        
        if accuracy_vals:
            row['accuracy_mean_across_datasets'] = np.mean(accuracy_vals)
            row['accuracy_std_across_datasets'] = np.std(accuracy_vals)
            row['accuracy_margin'] = max(accuracy_vals) - min(accuracy_vals)
        
        if f1_vals:
            row['f1_mean_across_datasets'] = np.mean(f1_vals)
            row['f1_std_across_datasets'] = np.std(f1_vals)
            row['f1_margin'] = max(f1_vals) - min(f1_vals)
        
        if auc_vals:
            row['auc_mean_across_datasets'] = np.mean(auc_vals)
            row['auc_std_across_datasets'] = np.std(auc_vals)
            row['auc_margin'] = max(auc_vals) - min(auc_vals)
        
        model_margins.append(row)
    
    model_margins_df = pd.DataFrame(model_margins)
    
    print("Model margins across datasets:")
    print(model_margins_df[['model', 'accuracy_mean_across_datasets', 'accuracy_std_across_datasets', 'accuracy_margin']])
    
    # ============================================================
    # ACCURACY PER DATASET (ACROSS ALL MODELS)
    # ============================================================
    # e.g., Imbalanced: mean(0.8708, 0.9213, 0.9168, ...) across all 8 models
    dataset_accuracy = []
    
    for dataset_type, mean_df in dataset_means.items():
        # Exclude mean_row
        model_only_df = mean_df[mean_df.index != 'mean_row']
        
        row = {
            'dataset': dataset_type,
            'accuracy_mean': model_only_df['accuracy'].mean(),
            'accuracy_std': model_only_df['accuracy'].std(),
            'accuracy_min': model_only_df['accuracy'].min(),
            'accuracy_max': model_only_df['accuracy'].max(),
            'accuracy_margin': model_only_df['accuracy'].max() - model_only_df['accuracy'].min(),
            'best_model': model_only_df['accuracy'].idxmax(),
            'worst_model': model_only_df['accuracy'].idxmin()
        }
        dataset_accuracy.append(row)
    
    dataset_accuracy_df = pd.DataFrame(dataset_accuracy)
    
    print("\nAccuracy per dataset (across all models):")
    print(dataset_accuracy_df)
    
    return model_margins_df, dataset_accuracy_df


def save_averaged_results(results_dir, experiments):
    """
    Save averaged results for each experiment with versioning and metadata.
    
    Parameters
    ----------
    results_dir : str
        Base results directory
    experiments : dict
        Collected experiment results
    
    Returns
    -------
    list
        Summary information for all experiments
    """
    summary_dir = os.path.join(results_dir, 'averaged_results')
    os.makedirs(summary_dir, exist_ok=True)
    
    all_summaries = []
    
    for exp_name, exp_data in experiments.items():
        print(f"\n{'='*50}")
        print(f"Averaging: {exp_name}")
        print(f"{'='*50}")
        
        mean_df, std_df, n_seeds = average_experiment_results(exp_data)
        
        if mean_df is not None:
            # Build seed details for metadata
            seed_details = []
            for item in exp_data:
                seed_num = item['seed']
                folder = item['folder']
                seed_details.append({
                    'seed': seed_num,
                    'folder': folder,
                    'version': folder.split('_v')[-1] if '_v' in folder else '0'
                })
            
            print(f"Seeds used: {n_seeds}")
            print(f"Seed details: {[s['folder'] for s in seed_details]}")
            print(f"\nMean metrics:\n{mean_df}")
            print(f"\nStd metrics:\n{std_df}")
            
            # Save mean and std with versioning
            DataProcessing.save_to_file(mean_df, summary_dir, f'{exp_name}_mean', 'csv', include_versioning=True)
            DataProcessing.save_to_file(std_df, summary_dir, f'{exp_name}_std', 'csv', include_versioning=True)
            
            # Create combined mean ± std format
            combined_df = mean_df.copy()
            # Reset index to make 'model' a column (if it's currently the index)
            if combined_df.index.name == 'model' or 'model' not in combined_df.columns:
                combined_df = combined_df.reset_index()
            for col in combined_df.columns:
                if col != 'model':  # Don't format the model name column
                    combined_df[col] = combined_df[col].apply(lambda x: f"{x:.4f}") + \
                                       " ± " + \
                                       std_df.reset_index()[col].apply(lambda x: f"{x:.4f}")
            DataProcessing.save_to_file(combined_df, summary_dir, f'{exp_name}_mean_std', 'csv', include_versioning=True)
            
            # Get the version number that was just used (from last save)
            next_version = DataProcessing.get_next_file_number(summary_dir, f'{exp_name}_mean') - 1
            
            # Save metadata with same version number
            metadata = {
                'experiment': exp_name,
                'version': next_version,
                'n_seeds': n_seeds,
                'seeds_used': seed_details,
                'date_averaged': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'files_generated': {
                    'mean': f'{exp_name}_mean-v{next_version}.csv',
                    'std': f'{exp_name}_std-v{next_version}.csv',
                    'mean_std': f'{exp_name}_mean_std-v{next_version}.csv'
                }
            }
            
            metadata_file = os.path.join(summary_dir, f'{exp_name}_metadata-v{next_version}.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Saved: {os.path.basename(metadata_file)}")
            
            all_summaries.append({
                'experiment': exp_name,
                'version': next_version,
                'n_seeds': n_seeds,
                'seed_info': seed_details,
                'mean': mean_df,
                'std': std_df
            })
    
    return all_summaries


if __name__ == "__main__":
    """
    Average classification results across multiple seed runs.
    
    Usage
    -----
    python3 average_classification_results.py
    
    Notes
    -----
    - Automatically finds all experiments in classification_results/
    - For each experiment, finds latest version of each seed
    - Averages metrics across seeds
    - Computes cross-dataset margins (same model across imbalanced/oversampled/undersampled)
    - Computes accuracy per dataset across all models
    - Saves mean, std, mean±std CSVs, metadata JSON, and margin CSVs
    """
    
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '../data/classification_results/')
    
    print("\n" + "="*50)
    print("AVERAGE CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Results directory: {results_dir}\n")
    
    # ============================================================
    # 2. COLLECT RESULTS
    # ============================================================
    experiments = collect_results(results_dir)
    
    if not experiments:
        print("\n❌ No experiments found to average.")
        print("Make sure you have run ml_classifiers.py with multiple seeds first.\n")
        sys.exit(0)
    
    print(f"\nFound {len(experiments)} experiment(s) to average:")
    for exp_name, exp_data in experiments.items():
        print(f"  - {exp_name}: {len(exp_data)} seed(s)")
    
    # ============================================================
    # 3. AVERAGE AND SAVE RESULTS
    # ============================================================
    summaries = save_averaged_results(results_dir, experiments)
    
    # ============================================================
    # 4. COMPUTE CROSS-DATASET MARGINS
    # ============================================================
    if len(summaries) >= 2:
        print("\n⚠️  Computing cross-dataset margins requires all 3 datasets.")
        print("Proceeding with available datasets...\n")
        
        model_margins_df, dataset_accuracy_df = compute_cross_dataset_margins(summaries)
        
        # Save cross-dataset results with versioning
        summary_dir = os.path.join(results_dir, 'averaged_results')
        DataProcessing.save_to_file(
            model_margins_df, summary_dir, 
            'cross_dataset_model_margins', 'csv', 
            include_versioning=True
        )
        DataProcessing.save_to_file(
            dataset_accuracy_df, summary_dir, 
            'cross_dataset_accuracy', 'csv', 
            include_versioning=True
        )
        print(f"\n✓ Saved: cross_dataset_model_margins-v{{}}.csv")
        print(f"✓ Saved: cross_dataset_accuracy-v{{}}.csv")
    else:
        print("\n⚠️  Need at least 2 experiments to compute cross-dataset margins.")
        print("Run experiments for imbalanced, oversampled, and undersampled datasets first.\n")
    
    # ============================================================
    # 5. PIPELINE COMPLETE
    # ============================================================
    print("\n" + "="*50)
    print("AVERAGING COMPLETE")
    print("="*50)
    print(f"Total experiments averaged: {len(summaries)}")
    print(f"Output location: {os.path.join(results_dir, 'averaged_results/')}")
    print("\nFiles created per experiment (versioned):")
    print("  - {{experiment}}_mean-v{{N}}.csv")
    print("  - {{experiment}}_std-v{{N}}.csv")
    print("  - {{experiment}}_mean_std-v{{N}}.csv")
    print("  - {{experiment}}_metadata-v{{N}}.json")
    print("\nCross-dataset files (versioned):")
    print("  - cross_dataset_model_margins-v{{N}}.csv")
    print("  - cross_dataset_accuracy-v{{N}}.csv\n")