import os
import sys
import argparse
import pandas as pd
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing
from data_visualizing import DataVisualizing


def filter_by_domain(df, domain_name):
    """
    Filter dataset by domain column.
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Domain' column.
    domain_name : str
        Domain to filter for.
    Returns
    -------
    pd.DataFrame
        Filtered dataset.
    """
    print(f"\nFiltering for domain: '{domain_name}'")
    if 'Domain' not in df.columns:
        print(f"⚠️  Warning: 'Domain' column not found. Skipping filter.")
        return df
    original_len = len(df)
    filtered_df = df[df['Domain'] == domain_name]
    filtered_len = len(filtered_df)
    print(f"  Original size: {original_len}")
    print(f"  Filtered size: {filtered_len}")
    print(f"  Kept {filtered_len/original_len*100:.1f}% of rows")
    return filtered_df


def combine_datasets(dataset_list, dataset_names):
    """
    Combine multiple datasets with standard columns.
    Parameters
    ----------
    dataset_list : list of pd.DataFrame
        List of datasets to combine.
    dataset_names : list of str
        Names of each dataset for logging.
    Returns
    -------
    pd.DataFrame
        Combined dataset.
    """
    print("\n" + "="*60)
    print("COMBINE DATASETS")
    print("="*60)
    print(f"Combining {len(dataset_list)} datasets:")
    for name, df in zip(dataset_names, dataset_list):
        print(f"  - {name}: {len(df)} rows")

    combined_df = DataProcessing.concat_dfs(dataset_list)
    print(f"\n✓ Combined dataset shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")

    required_cols = ['Base Sentence', 'Ground Truth']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        print(f"\n⚠️  Warning: Missing required columns: {missing_cols}")
    else:
        print(f"\n✓ All required columns present: {required_cols}")

    if 'Ground Truth' in combined_df.columns:
        print(f"\nGround Truth distribution:")
        print(combined_df['Ground Truth'].value_counts())

    print(f"\nPreview:\n{combined_df.head(3)}")
    print(f"\nTail:\n{combined_df.tail(3)}\n")
    return combined_df


def extract_standard_columns(combined_df, additional_cols=None):
    """
    Extract standard columns plus any additional specified columns.
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined dataset.
    additional_cols : list of str, optional
        Additional columns to keep beyond the standard set.
    Returns
    -------
    pd.DataFrame
        Dataset with only the selected columns.
    """
    standard_cols = ['Base Sentence', 'Ground Truth', 'Dataset Name']
    keep_cols = standard_cols + (additional_cols if additional_cols else [])
    filtered_keep_cols = [col for col in keep_cols if col in combined_df.columns]
    print(f"\nExtracting columns: {filtered_keep_cols}")
    return combined_df.loc[:, filtered_keep_cols]


def save_combined_dataset(df, save_path, output_name, include_version=True):
    """
    Save combined dataset to disk.
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to save.
    save_path : str
        Directory to save the dataset.
    output_name : str
        Output filename prefix.
    include_version : bool, optional
        If True, appends version number to filename. Default is True.
    """
    print("\n" + "="*60)
    print("SAVE COMBINED DATASET")
    print("="*60)
    combo_data_save_path = os.path.join(save_path, output_name)
    os.makedirs(combo_data_save_path, exist_ok=True)
    DataProcessing.save_to_file(
        df,
        combo_data_save_path,
        output_name,
        'csv',
        include_version=include_version
    )
    if include_version:
        existing_files = [f for f in os.listdir(combo_data_save_path) if f.startswith(output_name)]
        latest_file = sorted(existing_files)[-1] if existing_files else f"{output_name}.csv"
        full_path = os.path.join(combo_data_save_path, latest_file)
    else:
        full_path = os.path.join(combo_data_save_path, f"{output_name}.csv")

    print(f"\n✓ Saved dataset:")
    print(f"  Path: {full_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Size: {os.path.getsize(full_path) / 1024:.2f} KB\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMBINED DATASET CREATION PIPELINE")
    print("="*60)

    # ============================================================
    # 1. Configuration and Argument Parsing
    # ============================================================
    base_data_path = os.path.join(script_dir, '../data')
    default_save_path = os.path.join(base_data_path, 'combined_datasets/')

    dataset_loader_map = {
        'synthetic':            DataProcessing.load_synthetic_dataset,
        'financial_phrasebank': DataProcessing.load_financial_phrasebank_dataset,
        'chronicle2050':        DataProcessing.load_chronicle2050_dataset,
        'news_api':             DataProcessing.load_news_api_dataset,
        'yt':                   DataProcessing.load_yt_dataset,
        'timebank':             DataProcessing.load_timebank_dataset,
        'mf_climate':           DataProcessing.load_mf_climate_dataset
    }

    parser = argparse.ArgumentParser(
        description='Combine synthetic and real datasets for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Available Datasets:
            synthetic            - LLM-generated predictions + observations [Brinkley et al. (...)]
            financial_phrasebank - Real financial statements [Malo et al. (2014)]
            chronicle2050        - Real statements from Longbets, Horizons, etc. [Regev et al. (2024)]
            news_api             - Real news API annotated sentences
            yt                   - Real YouTube annotated sentences
            timebank             - TimeBank 1.2 annotated sentences
            mf_climate           - Real MF climate forecast predictions [B. Moe, 2024]

        Examples:
            python3 create_combined_dataset.py --datasets synthetic
            python3 create_combined_dataset.py --datasets synthetic financial_phrasebank
            python3 create_combined_dataset.py --datasets synthetic --filter_domain finance
            python3 create_combined_dataset.py --datasets synthetic financial_phrasebank chronicle2050 news_api yt timebank mf_climate --output_name all-combined
        """
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(dataset_loader_map.keys()),
        default=['synthetic'],
        help='One or more datasets to combine. Default: synthetic.'
    )
    parser.add_argument(
        '--predictions_only',
        action='store_true',
        default=False,
        help='Only load prediction-labeled rows (default: False).'
    )
    parser.add_argument(
        '--filter_domain',
        default=None,
        choices=['finance', 'weather', 'policy', 'health', 'sports', 'misc'],
        help='Filter synthetic datasets by domain. Default: None (all domains).'
    )
    parser.add_argument(
        '--save_path',
        default=default_save_path,
        help=f'Directory to save combined dataset. Default: {default_save_path}'
    )
    parser.add_argument(
        '--output_name',
        default='combined_dataset',
        help='Output filename (without extension). Default: combined_dataset.'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Skip saving to disk (dry run for testing).'
    )
    parser.add_argument(
        '--no_version',
        action='store_true',
        help='Overwrite existing file instead of creating versioned copy.'
    )
    parser.add_argument(
        '--keep_all_columns',
        action='store_true',
        help='Keep all columns from source datasets.'
    )
    parser.add_argument(
        '--additional_columns',
        nargs='+',
        default=None,
        help='Additional columns to keep (e.g., Domain Model).'
    )
    args = parser.parse_args()

    # ============================================================
    # 2. Display Configuration
    # ============================================================
    print(f"\nConfiguration:")
    print(f"  Datasets to combine : {', '.join(args.datasets)}")
    print(f"  Predictions only    : {args.predictions_only}")
    print(f"  Domain filter       : {args.filter_domain if args.filter_domain else 'None (all domains)'}")
    print(f"  Save path           : {args.save_path}")
    print(f"  Output name         : {args.output_name}")
    print(f"  Save to disk        : {'No (dry run)' if args.no_save else 'Yes'}")
    print(f"  Versioning          : {'Disabled' if args.no_version else 'Enabled'}")
    if args.additional_columns:
        print(f"  Additional columns  : {', '.join(args.additional_columns)}")
    print()

    # ============================================================
    # 3. Load Datasets
    # ============================================================
    domain_filterable = {'synthetic', 'news_api', 'yt'}
    datasets_to_combine = []
    dataset_names = []

    for dataset_key in args.datasets:
        loader = dataset_loader_map[dataset_key]
        df = loader(script_dir, predictions_only=False, visualize=False)  # always load all, never plot
        if args.filter_domain and dataset_key in domain_filterable:
            df = filter_by_domain(df, args.filter_domain)
        datasets_to_combine.append(df)
        dataset_names.append(dataset_key)
    # ============================================================
    # 4. Validate Dataset Selection
    # ============================================================
    if len(datasets_to_combine) == 0:
        print("\n❌ ERROR: No datasets were loaded.")
        print("Please specify at least one dataset using --datasets.")
        sys.exit(1)

    # ============================================================
    # 5. Combine Datasets
    # ============================================================
    combined_df = combine_datasets(datasets_to_combine, dataset_names)

    # ============================================================
    # 6. Extract Columns
    # ============================================================
    print("\n" + "="*60)
    print("COLUMN SELECTION")
    print("="*60)

    if args.keep_all_columns:
        print("Keeping all columns from source datasets.")
        final_df = combined_df
    else:
        final_df = extract_standard_columns(combined_df, args.additional_columns)

    print(f"\nFinal dataset:")
    print(f"  Shape  : {final_df.shape}")
    print(f"  Columns: {list(final_df.columns)}")
    print(f"\nPreview:\n{final_df.head(7)}\n")
    print(f"\nTail:\n{final_df.tail(7)}\n")

    # ============================================================
    # 7. Save Combined Dataset
    # ============================================================
    if args.no_save:
        print("\n" + "="*60)
        print("SKIPPING SAVE (DRY RUN)")
        print("="*60)
        print(f"Dataset would have been saved to: {os.path.join(args.save_path, args.output_name)}.csv")
    else:
        save_combined_dataset(
            final_df,
            args.save_path,
            args.output_name,
            include_version=not args.no_version
        )

    # ============================================================
    # 8. Pipeline Complete
    # ============================================================
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Datasets combined  : {len(datasets_to_combine)}")
    print(f"Final dataset shape: {final_df.shape}")
    if args.filter_domain:
        print(f"Domain filter applied: {args.filter_domain}")
    if not args.no_save:
        print(f"✓ Saved to: {args.save_path}")

    print("\nSummary statistics:")
    if 'Ground Truth' in final_df.columns:
        total = len(final_df)
        pred_count = (final_df['Ground Truth'] == 1).sum()
        non_pred_count = (final_df['Ground Truth'] == 0).sum()
        print(f"  Prediction Count     (Label=1): {pred_count} ({round(pred_count/total*100, 2)}%)")
        print(f"  Non-Prediction Count (Label=0): {non_pred_count} ({round(non_pred_count/total*100, 2)}%)")

    if args.predictions_only:
        final_df = final_df[final_df['Ground Truth'] == 1]
        print(f"Filtered to predictions only: {final_df.shape}")

    save_stacked_plot = os.path.join(args.save_path, args.output_name)
    if 'Dataset Name' in final_df.columns:
        print("Plotting stacked Dataset Name distribution...")
        DataVisualizing.plot_stacked_distribution(
            final_df,
            category_col='Dataset Name',
            label_col='Ground Truth',
            save_path=save_stacked_plot
        )
    print("\n" + "="*60 + "\n")