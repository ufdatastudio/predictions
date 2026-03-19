# average_classification_results.py
import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
# Add project modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing

def get_latest_seed_version(experiment_dir, base_seed):
    """Find the latest version of a seed folder."""
    base_seed_str = str(base_seed)
    seed_pattern = f"seed{base_seed_str}"
    
    all_folders = []
    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)
        if os.path.isdir(item_path) and item.startswith(seed_pattern):
            all_folders.append(item)
    
    if not all_folders:
        return None
    
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
    
    latest_version, latest_folder = max(versioned_folders, key=lambda x: x[0])
    return latest_folder

def collect_results(results_dir, mode='cross_dataset', target_experiment=None, filter_experiments=None):
    """Collect all metrics_summary.csv files and group by experiment AND test set."""
    experiments = {}
    
    print(f"\n{'='*60}")
    print(f"COLLECTING RESULTS (mode={mode})")
    print(f"{'='*60}\n")
    
    if mode == 'single':
        if not target_experiment:
            raise ValueError("--experiment required for mode='single'")
        experiment_dirs = [target_experiment]
        print(f"Target experiment: {target_experiment}\n")
    else:  
        experiment_dirs = []
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and item not in ['averaged_results', 'cross_dataset_comparisons'] and not item.startswith('.'):
                if re.search(r'\d{4}-\d{2}-\d{2}', item):
                    if filter_experiments is None or item in filter_experiments:
                        experiment_dirs.append(item)
    
    for exp_dir_name in sorted(experiment_dirs):
        exp_dir_path = os.path.join(results_dir, exp_dir_name)
        seed_folders = [f for f in os.listdir(exp_dir_path) if f.startswith('seed')]
        
        for seed_folder in seed_folders:
            seed = int(re.search(r'\d+', seed_folder).group())
            seed_folder_path = os.path.join(exp_dir_path, seed_folder)
            
            # Walk through all directories inside the seed folder
            for root, dirs, files in os.walk(seed_folder_path):
                if 'ml_metrics_summary.csv' in files:
                    csv_path = os.path.join(root, 'ml_metrics_summary.csv')
                    
                    # Figure out if this is in_domain or external
                    rel_path = os.path.relpath(root, seed_folder_path)
                    
                    # Clean the path to group folds together! 
                    # If path is cross_domain_fold_1/external_dataset_A -> make it external_dataset_A
                    # If path is in_domain_fold_1 -> make it in_domain
                    if 'external_' in rel_path:
                        test_set_name = [p for p in rel_path.split(os.sep) if p.startswith('external_')][0]
                    elif 'in_domain' in rel_path:
                        test_set_name = 'in_domain'
                    else:
                        continue
                        
                    eval_key = f"{exp_dir_name}__TEST__{test_set_name}"
                    
                    if eval_key not in experiments:
                        experiments[eval_key] = []
                    
                    df = DataProcessing.load_from_file(csv_path, 'csv', sep=',')
                    experiments[eval_key].append({
                        'seed': seed,
                        'folder': rel_path,
                        'data': df
                    })
                    print(f"    ✓ Loaded: {seed_folder}/{rel_path}/ml_metrics_summary.csv")
    return experiments

def average_experiment_results(experiment_data):
    """Average metrics across seeds, grouped by model."""
    if len(experiment_data) == 0:
        return None, None, 0
    
    all_dfs = [item['data'] for item in experiment_data]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    
    mean_df = combined_df.groupby('model')[numeric_cols].mean()
    mean_df.loc['mean_across_models'] = mean_df.mean()
    
    std_df = combined_df.groupby('model')[numeric_cols].std()
    std_df.loc['std_across_models'] = std_df.std()
    
    n_seeds = len(all_dfs)
    
    return mean_df, std_df, n_seeds

def detect_dataset_type(experiment_name):
    """Auto-detect dataset type from experiment name."""
    name_lower = experiment_name.lower()
    
    if 'imbalanced' in name_lower:
        return 'imbalanced'
    elif 'oversampled' in name_lower or 'oversample' in name_lower:
        return 'oversampled'
    elif 'undersampled' in name_lower or 'undersample' in name_lower:
        return 'undersampled'
    else:
        return experiment_name

def compute_cross_dataset_margins(summaries):
    """Compute margins for same model across datasets."""
    print(f"\n{'='*50}")
    print("CROSS-DATASET MARGINS")
    print(f"{'='*50}\n")
    
    dataset_means = {}
    dataset_type_mapping = {}
    
    for summary in summaries:
        exp_name = summary['experiment']
        mean_df = summary['mean']
        
        dataset_type = detect_dataset_type(exp_name)
        dataset_type_mapping[exp_name] = dataset_type
        dataset_means[dataset_type] = mean_df
    
    print(f"Dataset types detected:")
    for exp_name, dataset_type in dataset_type_mapping.items():
        print(f"  {exp_name} → {dataset_type}")
    print()
    
    # ============================================================
    # SAME MODEL ACROSS DATASETS
    # ============================================================
    model_margins = []
    
    all_models = set()
    for mean_df in dataset_means.values():
        all_models.update(mean_df.index.tolist())
    all_models = sorted([m for m in all_models if not m.startswith('mean_') and not m.startswith('std_')])
    
    for model in all_models:
        row = {'model': model}
        
        # Collect all metric values per dataset for this model
        for dataset_type, mean_df in dataset_means.items():
            if model in mean_df.index:
                
                # List of all possible metrics from your dataframe
                metric_columns = [
                    'train_accuracy', 'val_accuracy', 'test_accuracy',
                    'precision_class_0', 'precision_class_1',
                    'recall_class_0', 'recall_class_1',
                    'f1_class_0', 'f1_class_1',
                    'tn', 'fp', 'fn', 'tp',
                    'roc_auc', 'pr_auc'
                ]
                
                # Dynamically extract and store any metric that exists for this model
                for metric in metric_columns:
                    if metric in mean_df.columns:
                        val = mean_df.loc[model, metric]
                        if pd.notna(val):
                            row[f'{dataset_type}_{metric}'] = val
                            
        # -------------------------------------------------------------
        # Compute mean ± std across datasets for all key metrics
        # -------------------------------------------------------------
        
        # Define the metrics we want to calculate cross-dataset statistics for
        metrics_to_summarize = [
            'train_accuracy', 'val_accuracy', 'test_accuracy',
            'precision_class_0', 'precision_class_1',
            'recall_class_0', 'recall_class_1',
            'f1_class_0', 'f1_class_1',
            'roc_auc', 'pr_auc'
        ]
        
        for metric in metrics_to_summarize:
            # Collect all values across datasets for this specific metric
            vals = [row[f'{d}_{metric}'] for d in dataset_means.keys() 
                    if f'{d}_{metric}' in row]
            
            # If values exist, compute stats
            if vals:
                row[f'{metric}_mean_across_datasets'] = np.mean(vals)
                row[f'{metric}_std_across_datasets'] = np.std(vals)
                row[f'{metric}_margin'] = max(vals) - min(vals)
                
        model_margins.append(row)
    
    model_margins_df = pd.DataFrame(model_margins)
    
    # Print summary
    print("Model margins across datasets:")
    if 'test_accuracy_mean_across_datasets' in model_margins_df.columns:
        summary_cols = ['model', 'test_accuracy_mean_across_datasets', 
                       'test_accuracy_std_across_datasets', 'test_accuracy_margin']
        print(model_margins_df[summary_cols].to_string(index=False))
    
    # ============================================================
    # ACCURACY PER DATASET
    # ============================================================
    dataset_accuracy = []
    
    for dataset_type, mean_df in dataset_means.items():
        model_only_df = mean_df[~mean_df.index.str.startswith('mean_') & ~mean_df.index.str.startswith('std_')]
        
        # Determine which accuracy column to use
        acc_col = 'test_accuracy' if 'test_accuracy' in model_only_df.columns else 'accuracy'
        
        if acc_col in model_only_df.columns:
            row = {
                'dataset': dataset_type,
                'accuracy_mean': model_only_df[acc_col].mean(),
                'accuracy_std': model_only_df[acc_col].std(),
                'accuracy_min': model_only_df[acc_col].min(),
                'accuracy_max': model_only_df[acc_col].max(),
                'accuracy_margin': model_only_df[acc_col].max() - model_only_df[acc_col].min(),
                'best_model': model_only_df[acc_col].idxmax(),
                'worst_model': model_only_df[acc_col].idxmin()
            }
            dataset_accuracy.append(row)
    
    dataset_accuracy_df = pd.DataFrame(dataset_accuracy)
    
    if not dataset_accuracy_df.empty:
        print("\nAccuracy per dataset (across all models):")
        print(dataset_accuracy_df.to_string(index=False))
    
    return model_margins_df, dataset_accuracy_df

def save_averaged_results(results_dir, experiments, mode='cross_dataset'):
    """Save averaged results for each experiment and test set."""
    all_summaries = []
    
    for raw_exp_name, exp_data in experiments.items():
        # Parse the Train -> Test relationship
        if "__TEST__" in raw_exp_name:
            base_exp_name, test_set_name = raw_exp_name.split("__TEST__")
            display_name = f"{base_exp_name} → {test_set_name}"
        else:
            base_exp_name = raw_exp_name
            test_set_name = ""
            display_name = base_exp_name

        print(f"\n{'='*50}")
        print(f"Averaging: {display_name}")
        print(f"{'='*50}")
        
        mean_df, std_df, n_seeds = average_experiment_results(exp_data)
        
        if mean_df is not None:
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
            
            # Determine save location based on mode
            if mode == 'single':
                # Save inside experiment folder: experiment/averaged/test_set_name
                if test_set_name:
                    save_dir = os.path.join(results_dir, base_exp_name, 'averaged', test_set_name)
                else:
                    save_dir = os.path.join(results_dir, base_exp_name, 'averaged')
            else:
                save_dir = None  # Handled by caller in cross_dataset mode
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
                # Save mean and std (no versioning needed)
                mean_df.to_csv(os.path.join(save_dir, 'mean.csv'))
                std_df.to_csv(os.path.join(save_dir, 'std.csv'))
                
                # Create combined mean ± std format
                combined_df = mean_df.copy()
                if combined_df.index.name == 'model' or 'model' not in combined_df.columns:
                    combined_df = combined_df.reset_index()
                    if 'index' in combined_df.columns:
                        combined_df = combined_df.rename(columns={'index': 'model'})
                
                for col in combined_df.columns:
                    if col != 'model':
                        combined_df[col] = combined_df[col].apply(lambda x: f"{x:.4f}") + \
                                           " ± " + \
                                           std_df.reset_index()[col].apply(lambda x: f"{x:.4f}")
                
                combined_df.to_csv(os.path.join(save_dir, 'mean_std.csv'), index=False)
                
                # Save metadata
                metadata = {
                    'experiment': display_name,
                    'n_seeds': n_seeds,
                    'seeds_used': seed_details,
                    'date_averaged': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'files_generated': {
                        'mean': 'mean.csv',
                        'std': 'std.csv',
                        'mean_std': 'mean_std.csv'
                    }
                }
                
                with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✓ Saved to: {save_dir}/")
            
            all_summaries.append({
                'experiment': display_name,  # Uses the clean "Train -> Test" string for LaTeX
                'n_seeds': n_seeds,
                'seed_info': seed_details,
                'mean': mean_df,
                'std': std_df
            })
    
    return all_summaries

def save_cross_dataset_results(results_dir, summaries, model_margins_df, dataset_accuracy_df):
    """
    Save cross-dataset comparison results in timestamped folder.
    
    Parameters
    ----------
    results_dir : str
        Base results directory
    summaries : list
        Summary information for all experiments
    model_margins_df : pd.DataFrame
        Cross-dataset model margins
    dataset_accuracy_df : pd.DataFrame
        Accuracy stats per dataset
    """
    # Create timestamped folder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    comparison_dir = os.path.join(results_dir, 'cross_dataset_comparisons', f'run_{timestamp}')
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"\nSaving cross-dataset comparison to: {comparison_dir}/")
    
    # Save model margins and dataset accuracy
    model_margins_df.to_csv(os.path.join(comparison_dir, 'cross_dataset_model_margins.csv'), index=False)
    dataset_accuracy_df.to_csv(os.path.join(comparison_dir, 'cross_dataset_accuracy.csv'), index=False)
    
    # Save metadata about which experiments were compared
    experiments_info = []
    for summary in summaries:
        experiments_info.append({
            'experiment': summary['experiment'],
            'n_seeds': summary['n_seeds'],
            'seeds_used': summary['seed_info']
        })
    
    comparison_metadata = {
        'timestamp': timestamp,
        'n_experiments': len(summaries),
        'experiments_compared': experiments_info,
        'files_generated': {
            'model_margins': 'cross_dataset_model_margins.csv',
            'dataset_accuracy': 'cross_dataset_accuracy.csv'
        }
    }
    
    with open(os.path.join(comparison_dir, 'experiments_compared.json'), 'w') as f:
        json.dump(comparison_metadata, f, indent=2)
    
    print(f"✓ Saved: cross_dataset_model_margins.csv")
    print(f"✓ Saved: cross_dataset_accuracy.csv")
    print(f"✓ Saved: experiments_compared.json")
    
    return comparison_dir

def print_latex_summary(summaries, model_margins_df=None):
    """Print LaTeX-formatted summary tables."""
    print(f"\n{'='*60}")
    print("LATEX OUTPUT (Mean ± Std)")
    print(f"{'='*60}\n")
    
    for summary in summaries:
        exp_name = summary['experiment']
        mean_df = summary['mean']
        std_df = summary['std']
        
        print(f"% {exp_name}")
        print(f"% Seeds: {summary['n_seeds']}\n")
        
        combined_df = mean_df.copy()
        
        if combined_df.index.name is not None or 'model' not in combined_df.columns:
            combined_df = combined_df.reset_index()
            if 'index' in combined_df.columns:
                combined_df = combined_df.rename(columns={'index': 'model'})
        
        # Order: Precision, Recall, F1, Test Acc, AUCs, Train Acc, Val Acc
        key_cols = ['precision_class_1', 'recall_class_1', 'f1_class_1', 
                   'test_accuracy', 'roc_auc', 'pr_auc', 'train_accuracy', 'val_accuracy']
        
        # Fallback to 'accuracy' if 'test_accuracy' doesn't exist
        if 'test_accuracy' not in combined_df.columns and 'accuracy' in combined_df.columns:
            key_cols = ['precision_class_1', 'recall_class_1', 'f1_class_1', 
                       'accuracy', 'roc_auc', 'pr_auc', 'train_accuracy', 'val_accuracy']
        
        # Only include columns that actually exist
        available_cols = ['model'] + [col for col in key_cols if col in combined_df.columns]
        
        if len(available_cols) > 1:
            latex_df = combined_df[available_cols].copy()
            
            for col in latex_df.columns:
                if col != 'model':
                    std_reset = std_df.reset_index()
                    if 'index' in std_reset.columns:
                        std_reset = std_reset.rename(columns={'index': 'model'})
                    
                    latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}") + \
                                    " $\\pm$ " + \
                                    std_reset[col].apply(lambda x: f"{x:.4f}")
            
            print(latex_df.to_latex(index=False, escape=False))
        else:
            for col in combined_df.columns:
                if col != 'model':
                    std_reset = std_df.reset_index()
                    if 'index' in std_reset.columns:
                        std_reset = std_reset.rename(columns={'index': 'model'})
                    
                    combined_df[col] = combined_df[col].apply(lambda x: f"{x:.4f}") + \
                                       " $\\pm$ " + \
                                       std_reset[col].apply(lambda x: f"{x:.4f}")
            
            print(combined_df.to_latex(index=False, escape=False))
        
        print()
    
    if model_margins_df is not None and not model_margins_df.empty:
        print("% Cross-Dataset Model Margins\n")
        
        margin_display = model_margins_df.copy()
        
        # Format mean ± std columns for test accuracy
        if 'test_accuracy_mean_across_datasets' in margin_display.columns:
            margin_display['Test Accuracy (Datasets)'] = \
                margin_display['test_accuracy_mean_across_datasets'].apply(lambda x: f"{x:.4f}") + \
                " $\\pm$ " + \
                margin_display['test_accuracy_std_across_datasets'].apply(lambda x: f"{x:.4f}")
        
        if 'f1_class_1_mean_across_datasets' in margin_display.columns:
            margin_display['F1 (Datasets)'] = \
                margin_display['f1_class_1_mean_across_datasets'].apply(lambda x: f"{x:.4f}") + \
                " $\\pm$ " + \
                margin_display['f1_class_1_std_across_datasets'].apply(lambda x: f"{x:.4f}")
        
        if 'roc_auc_mean_across_datasets' in margin_display.columns:
            margin_display['ROC AUC (Datasets)'] = \
                margin_display['roc_auc_mean_across_datasets'].apply(lambda x: f"{x:.4f}") + \
                " $\\pm$ " + \
                margin_display['roc_auc_std_across_datasets'].apply(lambda x: f"{x:.4f}")
                
        if 'pr_auc_mean_across_datasets' in margin_display.columns:
            margin_display['PR AUC (Datasets)'] = \
                margin_display['pr_auc_mean_across_datasets'].apply(lambda x: f"{x:.4f}") + \
                " $\\pm$ " + \
                margin_display['pr_auc_std_across_datasets'].apply(lambda x: f"{x:.4f}")
        
        display_cols = ['model', 'Test Accuracy (Datasets)', 'F1 (Datasets)', 'ROC AUC (Datasets)', 'PR AUC (Datasets)']
        available_display = [col for col in display_cols if col in margin_display.columns]
        
        if available_display:
            print(margin_display[available_display].to_latex(index=False, escape=False))
        else:
            print(margin_display.to_latex(index=False, escape=False, float_format="%.4f"))
        
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Average classification results across multiple seed runs'
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'cross_dataset'],
        default='cross_dataset',
        help='Mode: single or cross_dataset. Default: cross_dataset'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment folder name (required for mode=single). Example: combined-full_synthetic-v1_2026-03-07'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=None,
        help='Specific experiments to average (space-separated). Example: --experiments exp1_2026-03-07 exp2_2026-03-06'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single' and not args.experiment:
        parser.error("--experiment is required when --mode single")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '../data/classification_results/')
    
    print("\n" + "="*60)
    print("AVERAGE CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Mode: {args.mode}")
    if args.mode == 'single':
        print(f"Target experiment: {args.experiment}")
    elif args.experiments:
        print(f"Filtering experiments: {len(args.experiments)}")
    print(f"Results directory: {results_dir}\n")
    
    # Collect results
    experiments = collect_results(
        results_dir, 
        mode=args.mode, 
        target_experiment=args.experiment,
        filter_experiments=args.experiments
    )
    
    if not experiments:
        print("\n❌ No experiments found to average.")
        sys.exit(0)
    
    print(f"\nFound {len(experiments)} experiment(s) to average:")
    for exp_name, exp_data in experiments.items():
        print(f"  - {exp_name}: {len(exp_data)} seed(s)")
    
    # Save averaged results
    summaries = save_averaged_results(results_dir, experiments, mode=args.mode)
    
    # Cross-dataset comparison (if applicable)
    if args.mode == 'cross_dataset' and len(summaries) >= 2:
        print("\n⚠️  Computing cross-dataset margins...")
        
        model_margins_df, dataset_accuracy_df = compute_cross_dataset_margins(summaries)
        
        # Save to timestamped folder
        comparison_dir = save_cross_dataset_results(
            results_dir, summaries, model_margins_df, dataset_accuracy_df
        )
        
        print(f"\n✓ Cross-dataset comparison saved to: {comparison_dir}/")
    
    elif args.mode == 'cross_dataset' and len(summaries) < 2:
        print("\n⚠️  Need at least 2 experiments to compute cross-dataset margins.")
    
    # Print LaTeX
    model_margins_df = None
    if args.mode == 'cross_dataset' and len(summaries) >= 2:
        # Re-compute for printing (already computed above)
        model_margins_df, _ = compute_cross_dataset_margins(summaries)
    
    print_latex_summary(summaries, model_margins_df)
    
    # Summary
    print("\n" + "="*60)
    print("AVERAGING COMPLETE")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Total experiments averaged: {len(summaries)}")
    
    if args.mode == 'single':
        for exp_name in experiments.keys():
            print(f"\nResults saved to: {os.path.join(results_dir, exp_name, 'averaged/')}")
    else:
        if len(summaries) >= 2:
            print(f"\nCross-dataset comparison saved to: cross_dataset_comparisons/")
    
    print()