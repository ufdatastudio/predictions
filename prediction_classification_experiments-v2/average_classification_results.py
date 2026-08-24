# average_classification_results.py
import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

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

def collect_results(results_dir, mode='cross_dataset', target_experiment=None, filter_experiments=None, model_type='ml', embedding_model=None, prompting_strategy=None):
    """
    Collect all metrics_summary csv files and group by experiment AND test set.
    """
    experiments = {}
    if model_type == 'ml':
        target_files = ['metrics_summary_ml_models.csv']
    elif model_type == 'llm':
        target_files = ['metrics_summary_llms.csv']
    elif model_type == 'rnn':  # ← ADD THIS
        target_files = ['metrics_summary_rnn.csv']
    elif model_type == 'gru':  # ← ADD THIS
        target_files = ['metrics_summary_gru.csv']
    else:
        target_files = ['metrics_summary_ml_models.csv', 'metrics_summary_llms.csv', 'metrics_summary_rnn.csv', 'metrics_summary_gru.csv']  # ← UPDATE THIS
    
    print(f"\n{'='*60}")
    print(f"COLLECTING RESULTS (mode={mode}, model_type={model_type})")
    print(f"{'='*60}\n")
    print(f"Looking for: {target_files}")
    if embedding_model:
        print(f"Scoping to embedding model: {embedding_model}\n")
    
    if mode == 'single':
        experiment_dirs = [target_experiment]
        exp_dir_path = os.path.join(results_dir, target_experiment)
        all_seed_folders = []
        for f in os.listdir(exp_dir_path):
            if f.startswith('seed'):
                if filter_experiments is None or f in filter_experiments:
                    all_seed_folders.append(f)
        seed_folders = all_seed_folders
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
            
            if embedding_model:
                # walk_root = os.path.join(seed_folder_path, 'in_domain', embedding_model)
                walk_root = os.path.join(
                seed_folder_path,
                'in_domain',
                embedding_model
                )
                if prompting_strategy:
                    walk_root = os.path.join(
                        walk_root,
                        prompting_strategy
                    )
                if not os.path.exists(walk_root):
                    print(f"    ⚠️  Skipping {seed_folder}: no results for {embedding_model}")
                    continue
            else:
                walk_root = seed_folder_path
            
            for root, dirs, files in os.walk(walk_root):
                for target_file in target_files:
                    if target_file in files:
                        csv_path = os.path.join(root, target_file)
                        rel_path = os.path.relpath(root, seed_folder_path)
                        if 'ml_models' in target_file:
                            file_tag = 'ml'
                        elif 'llms' in target_file:
                            file_tag = 'llm'
                        elif 'rnn' in target_file:
                            # file_tag = 'rnn'
                            file_tag = os.path.basename(rel_path) 
                        elif 'gru' in target_file:
                            # file_tag = 'rnn'
                            file_tag = os.path.basename(rel_path) 
                        eval_key = (exp_dir_name, rel_path, file_tag)
                        if eval_key not in experiments:
                            experiments[eval_key] = []
                        df = DataProcessing.load_from_file(csv_path, 'csv', sep=',')
                        experiments[eval_key].append({
                            'seed': seed,
                            'folder': rel_path,
                            'data': df
                        })
                        print(f"    ✓ Loaded [{file_tag}]: {seed_folder}/{rel_path}/{target_file}")
    return experiments

def average_experiment_results(experiment_data):
    """Average metrics across seeds, grouped by model."""
    if len(experiment_data) == 0:
        return None, None, 0
    
    all_dfs = [item['data'] for item in experiment_data]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(combined_df.head(7))
    print(combined_df.tail(7))
    print(f"{'='*60}\n")
    
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    if combined_df.columns[0] == '':
        combined_df = combined_df.rename(columns={combined_df.columns[0]: 'model'})
    
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
    
    model_margins = []
    all_models = set()
    for mean_df in dataset_means.values():
        all_models.update(mean_df.index.tolist())
    all_models = sorted([m for m in all_models if not m.startswith('mean_') and not m.startswith('std_')])
    
    for model in all_models:
        row = {'model': model}
        for dataset_type, mean_df in dataset_means.items():
            if model in mean_df.index:
                metric_columns = [
                    'train_accuracy', 'val_accuracy', 'test_accuracy',
                    'precision_class_0', 'precision_class_1',
                    'recall_class_0', 'recall_class_1',
                    'f1_class_0', 'f1_class_1',
                    'tn', 'fp', 'fn', 'tp',
                    'roc_auc', 'pr_auc'
                ]
                for metric in metric_columns:
                    if metric in mean_df.columns:
                        val = mean_df.loc[model, metric]
                        if pd.notna(val):
                            row[f'{dataset_type}_{metric}'] = val
        
        metrics_to_summarize = [
            'train_accuracy', 'val_accuracy', 'test_accuracy',
            'precision_class_0', 'precision_class_1',
            'recall_class_0', 'recall_class_1',
            'f1_class_0', 'f1_class_1',
            'roc_auc', 'pr_auc'
        ]
        for metric in metrics_to_summarize:
            vals = [row[f'{d}_{metric}'] for d in dataset_means.keys()
                    if f'{d}_{metric}' in row]
            if vals:
                row[f'{metric}_mean_across_datasets'] = np.mean(vals)
                row[f'{metric}_std_across_datasets'] = np.std(vals)
                row[f'{metric}_margin'] = max(vals) - min(vals)
        model_margins.append(row)
    
    model_margins_df = pd.DataFrame(model_margins)
    print("Model margins across datasets:")
    if 'test_accuracy_mean_across_datasets' in model_margins_df.columns:
        summary_cols = ['model', 'test_accuracy_mean_across_datasets',
                        'test_accuracy_std_across_datasets', 'test_accuracy_margin']
        print(model_margins_df[summary_cols].to_string(index=False))
    
    dataset_accuracy = []
    for dataset_type, mean_df in dataset_means.items():
        model_only_df = mean_df[~mean_df.index.str.startswith('mean_') & ~mean_df.index.str.startswith('std_')]
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

def _format_mean_std(mean_df, std_df, key_cols=None):
    """Build a formatted mean ± std DataFrame with aligned positional indices."""
    mean_reset = mean_df.reset_index()
    std_reset = std_df.reset_index()
    
    for df in (mean_reset, std_reset):
        if 'index' in df.columns:
            df.rename(columns={'index': 'model'}, inplace=True)
    
    if key_cols is not None:
        available = ['model'] + [c for c in key_cols if c in mean_reset.columns]
    else:
        available = mean_reset.columns.tolist()
    
    mean_reset = mean_reset[available].reset_index(drop=True)
    std_reset = std_reset[available].reset_index(drop=True)
    formatted = mean_reset.copy()
    
    for col in formatted.columns:
        if col == 'model':
            continue
        formatted[col] = (
            mean_reset[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
            + " $\\pm$ "
            + std_reset[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
        )
    
    return formatted

def save_averaged_results(results_dir, experiments, mode='cross_dataset', embedding_model=None):
    """Save averaged results for each experiment and test set."""
    all_summaries = []
    for (base_exp_name, test_set_name, file_tag), exp_data in experiments.items():
        display_name = f"{base_exp_name} → {test_set_name} [{file_tag}]"
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
            
            if mode == 'single':
                averaged_base = os.path.join(results_dir, base_exp_name, 'averaged')
                if test_set_name:
                    save_dir = os.path.join(averaged_base, test_set_name, file_tag)
                else:
                    save_dir = os.path.join(averaged_base, file_tag)
            else:
                save_dir = None
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                mean_df.to_csv(os.path.join(save_dir, 'mean.csv'))
                std_df.to_csv(os.path.join(save_dir, 'std.csv'))
                combined_df = _format_mean_std(mean_df, std_df)
                combined_df.to_csv(os.path.join(save_dir, 'mean_std.csv'), index=False)
                
                metadata = {
                    'experiment': display_name,
                    'model_type': file_tag,
                    'embedding_model': embedding_model or 'N/A',
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
                'experiment': display_name,
                'model_type': file_tag,
                'n_seeds': n_seeds,
                'seed_info': seed_details,
                'mean': mean_df,
                'std': std_df
            })
    return all_summaries

def save_cross_dataset_results(results_dir, summaries, model_margins_df, dataset_accuracy_df):
    """Save cross-dataset comparison results in a timestamped folder."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    comparison_dir = os.path.join(results_dir, 'cross_dataset_comparisons', f'run_{timestamp}')
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"\nSaving cross-dataset comparison to: {comparison_dir}/")
    model_margins_df.to_csv(os.path.join(comparison_dir, 'cross_dataset_model_margins.csv'), index=False)
    dataset_accuracy_df.to_csv(os.path.join(comparison_dir, 'cross_dataset_accuracy.csv'), index=False)
    
    experiments_info = []
    for summary in summaries:
        experiments_info.append({
            'experiment': summary['experiment'],
            'model_type': summary.get('model_type', 'ml'),
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
    
    key_cols = [
        'precision_class_1', 'recall_class_1', 'f1_class_1',
        'test_accuracy', 'roc_auc', 'pr_auc',
        'train_accuracy', 'val_accuracy'
    ]
    key_cols_fallback = [
        'precision_class_1', 'recall_class_1', 'f1_class_1',
        'accuracy', 'roc_auc', 'pr_auc',
        'train_accuracy', 'val_accuracy'
    ]
    
    for summary in summaries:
        exp_name = summary['experiment']
        mean_df = summary['mean']
        std_df = summary['std']
        print(f"% {exp_name}")
        print(f"% Seeds: {summary['n_seeds']}\n")
        
        cols_to_use = key_cols if 'test_accuracy' in mean_df.columns else key_cols_fallback
        latex_df = _format_mean_std(mean_df, std_df, key_cols=cols_to_use)
        print(latex_df.to_latex(index=False, escape=False))
        print()
    
    if model_margins_df is not None and not model_margins_df.empty:
        print("% Cross-Dataset Model Margins\n")
        margin_display = model_margins_df.copy()
        
        def _margin_col(df, mean_col, std_col, label):
            if mean_col in df.columns and std_col in df.columns:
                df[label] = (
                    df[mean_col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
                    + " $\\pm$ "
                    + df[std_col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
                )
        
        _margin_col(margin_display,
                    'test_accuracy_mean_across_datasets', 'test_accuracy_std_across_datasets',
                    'Test Accuracy (Datasets)')
        _margin_col(margin_display,
                    'f1_class_1_mean_across_datasets', 'f1_class_1_std_across_datasets',
                    'F1 (Datasets)')
        _margin_col(margin_display,
                    'roc_auc_mean_across_datasets', 'roc_auc_std_across_datasets',
                    'ROC AUC (Datasets)')
        _margin_col(margin_display,
                    'pr_auc_mean_across_datasets', 'pr_auc_std_across_datasets',
                    'PR AUC (Datasets)')
        
        display_cols = ['model', 'Test Accuracy (Datasets)', 'F1 (Datasets)',
                        'ROC AUC (Datasets)', 'PR AUC (Datasets)']
        available_display = [c for c in display_cols if c in margin_display.columns]
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
        help='Experiment folder name (required for mode=single).'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=None,
        help='Specific seed folders to average (space-separated). Example: --experiments seed3 seed7 seed33'
    )
    parser.add_argument(
        '--model_type',
        choices=['ml', 'llm', 'rnn', 'gru', 'both'],
        default='ml',
        help='Which model type to average results for. Default: ml'
    )
    parser.add_argument(
        '--embedding_model',
        default=None,
        choices=['spacy_small', 'spacy_medium', 'spacy_large', 'spacy_transformer',
                'st_mpnet_base', 'st_distilroberta', 'st_minilm_l12', 'st_minilm_l6'],
        help='Scope results collection and averaging to a specific embedding model subfolder.'
    )
    parser.add_argument(
        '--prompting_strategy',
        default=None,
        help='Optional prompting strategy subfolder (e.g. chain-of-thought)'
    )
    args = parser.parse_args()
    
    if args.mode == 'single' and not args.experiment:
        parser.error("--experiment is required when --mode single")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '../data/classification_results/')
    
    print("\n" + "="*60)
    print("AVERAGE CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Mode:              {args.mode}")
    print(f"Model type:        {args.model_type}")
    print(f"Embedding model:   {args.embedding_model or 'all'}")
    if args.mode == 'single':
        print(f"Target experiment: {args.experiment}")
    elif args.experiments:
        print(f"Filtering to:      {args.experiments}")
    print(f"Results directory: {results_dir}\n")
    
    experiments = collect_results(
        results_dir,
        mode=args.mode,
        target_experiment=args.experiment,
        filter_experiments=args.experiments,
        model_type=args.model_type,
        embedding_model=args.embedding_model,
        prompting_strategy=args.prompting_strategy
    )
    
    if not experiments:
        print("\n❌ No experiments found to average.")
        sys.exit(0)
    
    print(f"\nFound {len(experiments)} experiment(s) to average:")
    for exp_name, exp_data in experiments.items():
        print(f"  - {exp_name}: {len(exp_data)} seed(s)")
    
    summaries = save_averaged_results(
        results_dir,
        experiments,
        mode=args.mode,
        embedding_model=args.embedding_model
    )
    
    model_margins_df = None
    if args.mode == 'cross_dataset' and len(summaries) >= 2:
        print("\n⚠️  Computing cross-dataset margins...")
        model_margins_df, dataset_accuracy_df = compute_cross_dataset_margins(summaries)
        comparison_dir = save_cross_dataset_results(
            results_dir, summaries, model_margins_df, dataset_accuracy_df
        )
        print(f"\n✓ Cross-dataset comparison saved to: {comparison_dir}/")
    elif args.mode == 'cross_dataset':
        print("\n⚠️  Need at least 2 experiments to compute cross-dataset margins.")
    
    print_latex_summary(summaries, model_margins_df)
    
    print("\n" + "="*60)
    print("AVERAGING COMPLETE")
    print("="*60)
    print(f"Mode:                      {args.mode}")
    print(f"Model type:                {args.model_type}")
    print(f"Embedding model:           {args.embedding_model or 'all'}")
    print(f"Total experiments averaged: {len(summaries)}")
    if args.mode == 'single':
        for (base_exp_name, test_set_name, file_tag) in experiments.keys():
            print(f"\nResults saved to: {os.path.join(results_dir, base_exp_name, 'averaged', test_set_name, file_tag)}")
    elif model_margins_df is not None:
        print(f"\nCross-dataset comparison saved to: cross_dataset_comparisons/")
    print()