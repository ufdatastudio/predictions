# average_results.py
import pandas as pd
import numpy as np
import os
import glob
import re

def extract_experiment_info(folder_name):
    """Extract base name and seed from folder name."""
    # Pattern: {base_name}_seed{number}[_optional_suffix]
    match = re.match(r'(.+?)_seed(\d+)(_.*)?$', folder_name)
    if match:
        base_name = match.group(1)
        seed = match.group(2)
        suffix = match.group(3) if match.group(3) else ''
        experiment_name = base_name + suffix
        return experiment_name, seed
    return None, None

def collect_results(results_dir):
    """Collect all ml_classifiers.csv files and group by experiment."""
    experiments = {}
    
    # Find all seed folders
    seed_folders = glob.glob(os.path.join(results_dir, '*_seed*'))
    
    for folder_path in seed_folders:
        folder_name = os.path.basename(folder_path)
        experiment_name, seed = extract_experiment_info(folder_name)
        
        if experiment_name and seed:
            csv_file = os.path.join(folder_path, 'ml_classifiers-v1.csv')
            
            if os.path.exists(csv_file):
                if experiment_name not in experiments:
                    experiments[experiment_name] = []
                
                df = pd.read_csv(csv_file, index_col=0)
                experiments[experiment_name].append({
                    'seed': seed,
                    'data': df
                })
                print(f"✓ Loaded: {folder_name}")
    
    return experiments

def average_experiment_results(experiment_data):
    """Average results across seeds for one experiment."""
    if len(experiment_data) == 0:
        return None, None, None
    
    # Stack all dataframes
    all_dfs = [item['data'] for item in experiment_data]
    
    # Select only numeric columns
    numeric_dfs = [df.select_dtypes(include=[np.number]) for df in all_dfs]
    
    # Compute mean and std on numeric data only
    mean_df = pd.concat(numeric_dfs).groupby(level=0).mean()
    std_df = pd.concat(numeric_dfs).groupby(level=0).std()
    
    # Add seed count info
    n_seeds = len(all_dfs)
    
    return mean_df, std_df, n_seeds

def save_averaged_results(results_dir, experiments):
    """Save averaged results for each experiment."""
    summary_dir = os.path.join(results_dir, 'averaged_results')
    os.makedirs(summary_dir, exist_ok=True)
    
    all_summaries = []
    
    for exp_name, exp_data in experiments.items():
        print(f"\n{'='*50}")
        print(f"Averaging: {exp_name}")
        print(f"{'='*50}")
        
        mean_df, std_df, n_seeds = average_experiment_results(exp_data)
        
        if mean_df is not None:
            print(f"Seeds used: {n_seeds}")
            print(f"\nMean metrics:\n{mean_df}")
            print(f"\nStd metrics:\n{std_df}")
            
            # Save mean
            mean_file = os.path.join(summary_dir, f'{exp_name}_mean.csv')
            mean_df.to_csv(mean_file)
            print(f"✓ Saved: {mean_file}")
            
            # Save std
            std_file = os.path.join(summary_dir, f'{exp_name}_std.csv')
            std_df.to_csv(std_file)
            print(f"✓ Saved: {std_file}")
            
            # Create combined mean ± std format
            combined_df = mean_df.copy()
            for col in combined_df.columns:
                combined_df[col] = combined_df[col].apply(lambda x: f"{x:.4f}") + \
                                   " ± " + \
                                   std_df[col].apply(lambda x: f"{x:.4f}")
            
            combined_file = os.path.join(summary_dir, f'{exp_name}_mean_std.csv')
            combined_df.to_csv(combined_file)
            print(f"✓ Saved: {combined_file}")
            
            all_summaries.append({
                'experiment': exp_name,
                'n_seeds': n_seeds,
                'mean': mean_df,
                'std': std_df
            })
    
    # Create master summary table
    if all_summaries:
        print(f"\n{'='*50}")
        print("MASTER SUMMARY")
        print(f"{'='*50}")
        
        for summary in all_summaries:
            print(f"\n{summary['experiment']} (n={summary['n_seeds']} seeds)")
            print(summary['mean'])
    
    return all_summaries

if __name__ == "__main__":
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '../data/classification_results/')
    
    print("\n" + "="*50)
    print("AVERAGE CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Results directory: {results_dir}\n")
    
    # Collect results
    experiments = collect_results(results_dir)
    
    print(f"\nFound {len(experiments)} experiments:")
    for exp_name, exp_data in experiments.items():
        print(f"  - {exp_name}: {len(exp_data)} seeds")
    
    # Average and save
    summaries = save_averaged_results(results_dir, experiments)
    
    print("\n" + "="*50)
    print("AVERAGING COMPLETE")
    print("="*50)
    print(f"Averaged results saved to: {os.path.join(results_dir, 'averaged_results/')}\n")