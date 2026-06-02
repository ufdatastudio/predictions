"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Average Properties Extraction Results
> Average metrics across seeds per model per property
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing


def collect_results(dataset_folder):
    """
    Collect all metrics_summary CSV files grouped by seed and model.

    Parameters
    ----------
    dataset_folder : str
        Path to the classification results folder containing seed subfolders.

    Returns
    -------
    list of dict
        Each dict contains seed, model_name, file name, file path, and loaded DataFrame.
    """
    print("\n" + "="*60)
    print("COLLECT RESULTS")
    print("="*60)

    model_files_to_avg = []

    seed_files = [
        f for f in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, f))
        and f.startswith('seed')
    ]

    print(f"Seeds found: {seed_files}\n")

    for seed_file in sorted(seed_files):
        print(f"Seed: {seed_file}")
        models_folder = os.path.join(dataset_folder, seed_file)
        model_files = os.listdir(models_folder)

        for model_file in model_files:
            print(f"  Model: {model_file}")
            model_path = os.path.join(models_folder, model_file)
            data_files = os.listdir(model_path)

            for data_file in data_files:
                if 'metrics_summary' in data_file and '.csv' in data_file:
                    print(f"    File: {data_file}")
                    metrics_summary_file_path = os.path.join(model_path, data_file)
                    df = DataProcessing.load_from_file(metrics_summary_file_path, 'csv', sep=',')

                    model_files_to_avg.append({
                        'seed':                      seed_file,
                        'model_name':                model_file,
                        'metrics_summary_file':      data_file,
                        'metrics_summary_file_path': metrics_summary_file_path,
                        'df':                        df
                    })
        print()

    print(f"✓ Total files collected: {len(model_files_to_avg)}")
    return model_files_to_avg


def average_results(model_files_to_avg):
    """
    Group collected results by model name and average metrics across seeds.

    Each property (Source, Target, Date, Outcome, No Property) is averaged
    separately so per-property performance is preserved.

    Parameters
    ----------
    model_files_to_avg : list of dict
        Output of collect_results().

    Returns
    -------
    dict
        { model_name: { 'mean': pd.DataFrame, 'std': pd.DataFrame } }
    """
    print("\n" + "="*60)
    print("AVERAGE RESULTS PER MODEL ACROSS SEEDS")
    print("="*60)

    # Group by model_name: { model_name: [df_seed3, df_seed7, df_seed33] }
    grouped_by_model = {}
    for model_data in model_files_to_avg:
        model_name = model_data['model_name']
        if model_name not in grouped_by_model:
            grouped_by_model[model_name] = []
        grouped_by_model[model_name].append(model_data['df'])

    averaged_results = {}

    for model_name, dfs in grouped_by_model.items():
        print(f"\nModel: {model_name} | Seeds: {len(dfs)}")

        # Stack all seed DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Average numeric columns grouped by property
        # so Source, Target, Date, Outcome are averaged separately
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()

        mean_df = combined_df.groupby('property')[numeric_cols].mean().reset_index()
        std_df  = combined_df.groupby('property')[numeric_cols].std().reset_index()

        mean_df.insert(0, 'model', model_name)
        std_df.insert(0,  'model', model_name)

        print(mean_df)

        averaged_results[model_name] = {
            'mean': mean_df,
            'std':  std_df
        }

    return averaged_results


def save_averaged_results(averaged_results, avg_save_path):
    """
    Save mean and std CSVs per model to disk.

    Parameters
    ----------
    averaged_results : dict
        Output of average_results().
    avg_save_path : str
        Directory path to save averaged results.
    """
    print("\n" + "="*60)
    print("SAVE AVERAGED RESULTS PER MODEL")
    print("="*60)

    os.makedirs(avg_save_path, exist_ok=True)

    for model_name, results in averaged_results.items():
        mean_df = results['mean']
        std_df  = results['std']

        # Replace "/" with "_" to avoid nested folders in filename
        # e.g., "openai/gpt-oss-120b" -> "openai_gpt-oss-120b"
        clean_model_name = model_name.replace('/', '_')

        DataProcessing.save_to_file(
            data=mean_df,
            path=avg_save_path,
            prefix=f'mean_{clean_model_name}',
            save_file_type='csv',
            include_version=False
        )

        DataProcessing.save_to_file(
            data=std_df,
            path=avg_save_path,
            prefix=f'std_{clean_model_name}',
            save_file_type='csv',
            include_version=False
        )

        print(f"✓ Saved averaged results for: {model_name}")

    print(f"\n✓ All averaged results saved to: {avg_save_path}")


def combine_and_save_all_models(averaged_results, avg_save_path):
    """
    Combine averaged results from all models into one unified CSV.

    Parameters
    ----------
    averaged_results : dict
        Output of average_results().
    avg_save_path : str
        Directory path to save the combined CSV.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all models and properties.
    """
    print("\n" + "="*60)
    print("COMBINED AVERAGED RESULTS — ALL MODELS")
    print("="*60)

    results_dfs = []
    for model_name, results in averaged_results.items():
        results_dfs.append(results['mean'])

    result_df = DataProcessing.concat_dfs(results_dfs)
    result_df = result_df.reset_index(drop=True)

    print(result_df)

    DataProcessing.save_to_file(
        data=result_df,
        path=avg_save_path,
        prefix='mean_all_models',
        save_file_type='csv',
        include_version=False
    )

    print(f"\n✓ Saved combined averaged results to: {avg_save_path}/mean_all_models.csv")
    return result_df


def export_to_latex(result_df, avg_save_path):
    """
    Export combined averaged results to a LaTeX table.

    Parameters
    ----------
    result_df : pd.DataFrame
        Combined averaged results from all models.
    avg_save_path : str
        Directory path to save the LaTeX file.
    """
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)

    # Select key columns for LaTeX table
    # Adjust based on what columns matter most for your paper
    latex_cols = [
        'model',
        'property',
        'test_accuracy',
        'precision_class_0',
        'precision_class_1',
        'recall_class_0',
        'recall_class_1',
        'f1_class_0',
        'f1_class_1',
        'tn', 'fp', 'fn', 'tp'
    ]

    # Only keep columns that exist
    available_latex_cols = [col for col in latex_cols if col in result_df.columns]
    latex_df = result_df[available_latex_cols].copy()

    # Round to 4 decimal places for cleaner LaTeX output
    numeric_cols = latex_df.select_dtypes(include=[np.number]).columns
    latex_df[numeric_cols] = latex_df[numeric_cols].round(4)

    latex_str = latex_df.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
        caption="Averaged Property Extraction Results Across Seeds",
        label="tab:property_extraction_results"
    )

    print(latex_str)

    latex_save_path = os.path.join(avg_save_path, 'mean_all_models.tex')
    with open(latex_save_path, 'w') as f:
        f.write(latex_str)

    print(f"✓ Saved LaTeX table to: {latex_save_path}")


if __name__ == "__main__":
    """
    Usage:
        python3 average_properties_extraction_results.py \
            --dataset synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate
    """
    print("\n" + "="*60)
    print("AVERAGE PROPERTIES EXTRACTION RESULTS")
    print("="*60)

    # ============================================================
    # 1. Configuration and Arguments
    # ============================================================
    parser = argparse.ArgumentParser(
        description='Average property extraction metrics across seeds per model.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate',
        help='Dataset folder name under classification_results.'
    )
    args = parser.parse_args()

    base_data_path = DataProcessing.load_base_data_path(script_dir)

    dataset_folder = os.path.join(
        base_data_path,
        'classification_results',
        args.dataset,
        'extract_properties',
        'classification'
    )

    avg_save_path = os.path.join(dataset_folder, 'average')

    print(f"Dataset       : {args.dataset}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Save path     : {avg_save_path}")

    # ============================================================
    # 2. Collect Results
    # ============================================================
    model_files_to_avg = collect_results(dataset_folder)

    if not model_files_to_avg:
        print("\n❌ No metrics_summary files found.")
        sys.exit(0)

    # ============================================================
    # 3. Average Results
    # ============================================================
    averaged_results = average_results(model_files_to_avg)

    # ============================================================
    # 4. Save Averaged Results Per Model
    # ============================================================
    save_averaged_results(averaged_results, avg_save_path)

    # ============================================================
    # 5. Combine All Models Into One Table
    # ============================================================
    result_df = combine_and_save_all_models(averaged_results, avg_save_path)

    # ============================================================
    # 6. Export to LaTeX
    # ============================================================
    export_to_latex(result_df, avg_save_path)

    # ============================================================
    # 7. Summary
    # ============================================================
    print("\n" + "="*60)
    print("AVERAGING COMPLETE")
    print("="*60)
    print(f"Dataset           : {args.dataset}")
    print(f"Models averaged   : {len(averaged_results)}")
    print(f"Results saved to  : {avg_save_path}")
    print()