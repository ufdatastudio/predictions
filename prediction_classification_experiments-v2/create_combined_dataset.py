import os
import sys
import argparse
import pandas as pd
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

from data_processing import DataProcessing
from data_visualizing import DataVisualizing


# ==============================================================
# HELPER: Standardize columns
# ==============================================================
def standardize_columns(df, text_col, label_col, label_map=None):
    """
    Standardize column names and ordering for any dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded dataset
    text_col : str
        Name of the sentence/text column to map to 'Base Sentence'
    label_col : str
        Name of the label column to map to 'Ground Truth'
    label_map : dict or None
        Optional mapping to convert raw label values to integers
        before renaming to 'Ground Truth'. e.g. {'prediction': 1, 'not-prediction': 0}

    Returns
    -------
    pd.DataFrame
        Dataset with standardized column names and ordering
    """
    if label_map is not None:
        if label_col not in df.columns:
            raise ValueError(f"Expected label column '{label_col}' not found for label_map application.")
        df[label_col] = df[label_col].str.lower().map(label_map)

    rename_map = {}
    if text_col != 'Base Sentence' and text_col in df.columns:
        rename_map[text_col] = 'Base Sentence'
    if label_col != 'Ground Truth' and label_col in df.columns:
        rename_map[label_col] = 'Ground Truth'
    if rename_map:
        df = df.rename(columns=rename_map)

    if 'Base Sentence' not in df.columns:
        raise ValueError(f"Expected text column '{text_col}' not found in dataset.")
    if 'Ground Truth' not in df.columns:
        raise ValueError(f"Expected label column '{label_col}' not found in dataset.")

    priority_cols = ['Base Sentence', 'Ground Truth']
    remaining_cols = []
    for col in df.columns:
        if col not in priority_cols:
            remaining_cols.append(col)

    df = df[priority_cols + remaining_cols]
    return df


# ==============================================================
# LOAD PREDICTIONS
# ==============================================================
def load_predictions_dataset(script_dir, sep=','):
    """
    Load all synthetic prediction dataset batches.

    Parameters
    ----------
    script_dir : str
        Current script directory
    sep : str
        CSV separator character

    Returns
    -------
    pd.DataFrame
        Combined predictions dataset
    """
    print("\n" + "="*60)
    print("LOAD PREDICTIONS DATASET")
    print("="*60)

    predictions_df = DataProcessing.load_multiple_batches(
        script_dir,
        sep=sep,
        data_type='prediction'
    )

    print(f"Shape: {predictions_df.shape}")
    print(f"Columns: {list(predictions_df.columns)}")

    skip = ['Base Sentence']
    for col in predictions_df.columns:
        if col not in skip:
            print(f"\n=== {col} value counts ===")
            print(predictions_df[col].value_counts())

    predictions_df = standardize_columns(
        df=predictions_df,
        text_col='Base Sentence',
        label_col='Sentence Label'
    )
    predictions_df['Dataset Name'] = 'synthetic_predictions'
    print(f"\nPreview:\n{predictions_df.head(3)}\n")

    return predictions_df


# ==============================================================
# LOAD NON-PREDICTIONS
# ==============================================================
def load_non_predictions_dataset(script_dir, sep=','):
    """
    Load all synthetic non-prediction (observation) dataset batches.

    Parameters
    ----------
    script_dir : str
        Current script directory
    sep : str
        CSV separator character

    Returns
    -------
    pd.DataFrame
        Combined non-predictions dataset
    """
    print("\n" + "="*60)
    print("LOAD NON-PREDICTIONS DATASET")
    print("="*60)

    non_predictions_df = DataProcessing.load_multiple_batches(
        script_dir,
        sep=sep,
        data_type='observation'
    )

    print(f"Shape: {non_predictions_df.shape}")
    print(f"Columns: {list(non_predictions_df.columns)}")

    skip = ['Base Sentence']
    for col in non_predictions_df.columns:
        if col not in skip:
            print(f"\n=== {col} value counts ===")
            print(non_predictions_df[col].value_counts())

    non_predictions_df = standardize_columns(
        df=non_predictions_df,
        text_col='Base Sentence',
        label_col='Sentence Label'
    )
    non_predictions_df['Dataset Name'] = 'synthetic_non_predictions'
    print(f"\nPreview:\n{non_predictions_df.head(3)}\n")

    return non_predictions_df


# ==============================================================
# LOAD FINANCIAL PHRASEBANK
# ==============================================================
def load_financial_phrasebank_dataset(script_dir, sep=',', encoding='latin'):
    """
    Load and process financial_phrasebank dataset.

    Parameters
    ----------
    script_dir : str
        Current script directory
    sep : str
        CSV separator character
    encoding : str
        File encoding

    Returns
    -------
    pd.DataFrame
        Processed financial phrasebank dataset with binary labels
    """
    print("\n" + "="*60)
    print("LOAD FINANCIAL PHRASEBANK DATASET")
    print("="*60)

    base_data_path = DataProcessing.load_base_data_path(script_dir)
    fpb_path = os.path.join(
        base_data_path,
        'financial_phrase_bank/annotators/maya_annotations-financial_phrasebank_statements-v3-final.csv'
    )

    print(f"Loading from: {fpb_path}")

    fpb_df = DataProcessing.load_from_file(fpb_path, 'csv', sep=sep, encoding=encoding)
    print(f"Loaded shape: {fpb_df.shape}")

    original_len = len(fpb_df)
    fpb_df.dropna(inplace=True)
    dropped_count = original_len - len(fpb_df)
    if dropped_count > 0:
        print(f"✓ Dropped {dropped_count} rows without labels")

    print("\nConverting text labels to binary...")
    fpb_df = DataProcessing.match_text_label_to_int(
        fpb_df,
        text_label_col_name='maya_label',
        target_label='PREDICTION'
    )

    fpb_df = standardize_columns(
        df=fpb_df,
        text_col='statement',
        label_col='Binary Label'
    )

    print(f"Final shape: {fpb_df.shape}")
    print(f"Columns: {list(fpb_df.columns)}")
    print(f"\nGround Truth distribution:")
    print(fpb_df['Ground Truth'].value_counts())

    fpb_df['Dataset Name'] = 'fpb-imbalanced'
    print(f"\nPreview:\n{fpb_df.head(3)}\n")

    return fpb_df


# ==============================================================
# LOAD CHRONICLE2050
# ==============================================================
def load_chronicle2050_dataset(script_dir, sep=',', encoding='latin'):
    """
    Load and process chronicle2050 dataset.

    Processing steps:
    1. Load annotated Chronicle2050 CSV
    2. Drop rows without labels
    3. Standardize column names via standardize_columns() with label_map
    """
    print("\n" + "="*60)
    print("LOAD CHRONICLE2050 DATASET")
    print("="*60)

    base_data_path = DataProcessing.load_base_data_path(script_dir)
    chronicle2050_path = os.path.join(
        base_data_path,
        'chronicle2050',
        'annotators',
        'chronicle2050-shawnick-binary-v2.csv'
    )

    print(f"Loading from: {chronicle2050_path}")

    chronicle2050_df = DataProcessing.load_from_file(
        chronicle2050_path,
        'csv',
        sep=sep,
        encoding=encoding
    )
    print(f"Loaded shape: {chronicle2050_df.shape}")

    original_len = len(chronicle2050_df)
    chronicle2050_df.dropna(subset=['shawnick_labels'], inplace=True)
    dropped_count = original_len - len(chronicle2050_df)
    if dropped_count > 0:
        print(f"✓ Dropped {dropped_count} rows without labels")

    chronicle2050_df = standardize_columns(
        df=chronicle2050_df,
        text_col='sentence',
        label_col='shawnick_labels',
        label_map={'prediction': 1, 'not-prediction': 0}
    )

    print(f"Final shape: {chronicle2050_df.shape}")
    print(f"Columns: {list(chronicle2050_df.columns)}")
    print("\nGround Truth distribution:")
    print(chronicle2050_df['Ground Truth'].value_counts())

    chronicle2050_df['Dataset Name'] = 'chronicle2050'
    print(f"\nPreview:\n{chronicle2050_df.head(7)}\n")

    return chronicle2050_df


# ==============================================================
# LOAD NEWS API
# ==============================================================
def load_news_api_dataset(script_dir, sep=','):
    """
    Load and process NewsAPI annotated dataset.

    Processing steps:
    1. Load all NewsAPI annotation CSVs
    2. Keep only prediction rows (Human Annotation == 1)
    3. Standardize column names via standardize_columns()
    4. Attach dataset name
    """
    print("\n" + "="*60)
    print("LOAD NEWS API DATASET")
    print("="*60)

    base_data_path = DataProcessing.load_base_data_path(script_dir)
    news_api_path = os.path.join(base_data_path, "news_api", "annotators")
    print(f"Loading from: {news_api_path}")

    dfs = []
    for filename in os.listdir(news_api_path):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(news_api_path, filename)
        print(f"Loading: {filename}")
        df = DataProcessing.load_from_file(filepath, file_type="csv", sep=sep)
        dfs.append(df)

    if not dfs:
        print("⚠️ No NewsAPI CSVs found.")
        return pd.DataFrame()

    news_api_df = DataProcessing.concat_dfs(dfs)
    print(f"Loaded shape (all rows): {news_api_df.shape}")

    if 'Human Annotation' not in news_api_df.columns:
        raise ValueError("Expected 'Human Annotation' column in NewsAPI dataset")

    news_api_df = news_api_df[news_api_df['Human Annotation'] == 1]
    print(f"Filtered shape (predictions only): {news_api_df.shape}")

    news_api_df = standardize_columns(
        df=news_api_df,
        text_col='Base Sentence',
        label_col='Human Annotation'
    )

    print("\nGround Truth distribution:")
    print(news_api_df['Ground Truth'].value_counts())

    news_api_df['Dataset Name'] = 'news_api_predictions'
    print(f"\nPreview:\n{news_api_df.head(7)}\n")

    return news_api_df


# ==============================================================
# LOAD YT
# ==============================================================
def load_yt_dataset(script_dir, sep=','):
    """
    Load and process YT annotated dataset.

    Processing steps:
    1. Load all YT annotation CSVs
    2. Keep only prediction rows (Human Annotation == 1)
    3. Standardize column names via standardize_columns()
    4. Attach dataset name
    """
    print("\n" + "="*60)
    print("LOAD YT DATASET")
    print("="*60)

    base_data_path = DataProcessing.load_base_data_path(script_dir)
    yt_path = os.path.join(base_data_path, "yt", "annotators", "sports")
    print(f"Loading from: {yt_path}")

    dfs = []
    for filename in os.listdir(yt_path):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(yt_path, filename)
        print(f"Loading: {filename}")
        df = DataProcessing.load_from_file(filepath, file_type="csv", sep=sep)
        dfs.append(df)

    if not dfs:
        print("⚠️ No YT CSVs found.")
        return pd.DataFrame()

    yt_df = DataProcessing.concat_dfs(dfs)
    print(f"Loaded shape (all rows): {yt_df.shape}")

    if 'Human Annotation' not in yt_df.columns:
        raise ValueError("Expected 'Human Annotation' column in YT dataset")

    yt_df = yt_df[yt_df['Human Annotation'] == 1]
    print(f"Filtered shape (predictions only): {yt_df.shape}")

    yt_df = standardize_columns(
        df=yt_df,
        text_col='Base Sentence',
        label_col='Human Annotation'
    )

    print("\nGround Truth distribution:")
    print(yt_df['Ground Truth'].value_counts())

    yt_df['Dataset Name'] = 'yt_predictions'
    print(f"\nPreview:\n{yt_df.head(7)}\n")

    return yt_df


# ==============================================================
# FILTER BY DOMAIN
# ==============================================================
def filter_by_domain(df, domain_name):
    """
    Filter dataset by domain column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Domain' column
    domain_name : str
        Domain to filter for

    Returns
    -------
    pd.DataFrame
        Filtered dataset
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


# ==============================================================
# COMBINE DATASETS
# ==============================================================
def combine_datasets(dataset_list, dataset_names):
    """
    Combine multiple datasets with standard columns.

    Parameters
    ----------
    dataset_list : list of pd.DataFrame
    dataset_names : list of str

    Returns
    -------
    pd.DataFrame
        Combined dataset
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
    missing_cols = []
    for col in required_cols:
        if col not in combined_df.columns:
            missing_cols.append(col)

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


# ==============================================================
# EXTRACT STANDARD COLUMNS
# ==============================================================
def extract_standard_columns(combined_df, additional_cols=None):
    """
    Extract standard columns plus any additional specified columns.

    Parameters
    ----------
    combined_df : pd.DataFrame
    additional_cols : list of str, optional

    Returns
    -------
    pd.DataFrame
    """
    standard_cols = ['Base Sentence', 'Ground Truth', 'Dataset Name']

    if additional_cols:
        keep_cols = standard_cols + additional_cols
    else:
        keep_cols = standard_cols

    filtered_keep_cols = []
    for col in keep_cols:
        if col in combined_df.columns:
            filtered_keep_cols.append(col)

    print(f"\nExtracting columns: {filtered_keep_cols}")
    extracted_df = combined_df.loc[:, filtered_keep_cols]

    return extracted_df


# ==============================================================
# SAVE COMBINED DATASET
# ==============================================================
def save_combined_dataset(df, save_path, output_name, include_version=True):
    """
    Save combined dataset to disk.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : str
    output_name : str
    include_version : bool
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
        existing_files = []
        for f in os.listdir(combo_data_save_path):
            if f.startswith(output_name):
                existing_files.append(f)
        if existing_files:
            latest_file = sorted(existing_files)[-1]
            full_path = os.path.join(combo_data_save_path, latest_file)
        else:
            full_path = os.path.join(combo_data_save_path, f"{output_name}.csv")
    else:
        full_path = os.path.join(combo_data_save_path, f"{output_name}.csv")

    print(f"\n✓ Saved dataset:")
    print(f"  Path: {full_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Size: {os.path.getsize(full_path) / 1024:.2f} KB\n")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":

    print("\n" + "="*60)
    print("COMBINED DATASET CREATION PIPELINE")
    print("="*60)

    # ============================================================
    # 1. CONFIGURATION & ARGUMENT PARSING
    # ============================================================

    base_data_path = os.path.join(script_dir, '../data')
    default_save_path = os.path.join(base_data_path, 'combined_datasets/')

    parser = argparse.ArgumentParser(
        description='Combine synthetic and real datasets for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Available Datasets:
            predictions          - LLM-generated future-tense prediction sentences [Brinkley et al. (...)]
            non_predictions      - LLM-generated past-tense observation sentences  [Brinkley et al. (...)]
            financial_phrasebank - Real financial statements from calls/reports/news [Malo et al. (2014)]
            chronicle2050        - Real statements from Longbets, Horizons, synthetic (ChatGPT), news (New York Times) [Regev et al. (2024)]
            news_api
            yt

            Examples:
            # Combine all datasets (default)
            python3 create_combined_dataset.py --output_name synthetic_only

            # Combine specific datasets
            python3 create_combined_dataset.py --datasets predictions financial_phrasebank

            # Filter synthetic data to finance domain only
            python3 create_combined_dataset.py --filter_domain finance

            # Custom output location and name
            python3 create_combined_dataset.py --save_path ../results/ --output_name my_dataset

            # Combine all datasets
            python create_combined_dataset.py --datasets predictions non_predictions financial_phrasebank chronicle2050 news_api yt --output_name all-preds-non_preds-fpb-c2050-news-yt
        """
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=[
            'predictions',
            'non_predictions',
            'financial_phrasebank',
            'chronicle2050',
            'news_api',
            'yt'
        ],
        default=['predictions', 'non_predictions'],
        help='Datasets to combine. Default: all synthetic only'
    )

    parser.add_argument(
        '--filter_domain',
        default=None,
        choices=['finance', 'weather', 'policy', 'health', 'sports', 'misc'],
        help='Filter synthetic datasets by domain. Default: None (include all domains)'
    )

    parser.add_argument(
        '--save_path',
        default=default_save_path,
        help=f'Directory to save combined dataset. Default: {default_save_path}'
    )

    parser.add_argument(
        '--output_name',
        help='Output filename (without extension).'
    )

    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Skip saving to disk (dry run for testing)'
    )

    parser.add_argument(
        '--no_version',
        action='store_true',
        help='Overwrite existing file instead of creating versioned copy'
    )

    parser.add_argument(
        '--keep_all_columns',
        action='store_true',
        help='Keep all columns from source datasets (default: only standard columns)'
    )

    parser.add_argument(
        '--additional_columns',
        nargs='+',
        default=None,
        help='Specific additional columns to keep (e.g., Domain, Model)'
    )

    args = parser.parse_args()

    # ============================================================
    # 2. DISPLAY CONFIGURATION
    # ============================================================

    print(f"\nConfiguration:")
    print(f"  Datasets to combine: {', '.join(args.datasets)}")
    print(f"  Domain filter: {args.filter_domain if args.filter_domain else 'None (all domains)'}")
    print(f"  Save path: {args.save_path}")
    print(f"  Output name: {args.output_name}")
    print(f"  Save to disk: {'No (dry run)' if args.no_save else 'Yes'}")
    print(f"  Versioning: {'Disabled' if args.no_version else 'Enabled'}")
    if args.additional_columns:
        print(f"  Additional columns: {', '.join(args.additional_columns)}")
    print()

    # ============================================================
    # 3. LOAD DATASETS
    # ============================================================

    datasets_to_combine = []
    dataset_names = []

    if 'predictions' in args.datasets:
        predictions_df = load_predictions_dataset(script_dir)
        if args.filter_domain:
            predictions_df = filter_by_domain(predictions_df, args.filter_domain)
        datasets_to_combine.append(predictions_df)
        dataset_names.append("Predictions")

    if 'non_predictions' in args.datasets:
        non_predictions_df = load_non_predictions_dataset(script_dir)
        if args.filter_domain:
            non_predictions_df = filter_by_domain(non_predictions_df, args.filter_domain)
        datasets_to_combine.append(non_predictions_df)
        dataset_names.append("Non-Predictions")

    if 'financial_phrasebank' in args.datasets:
        fpb_df = load_financial_phrasebank_dataset(script_dir)
        datasets_to_combine.append(fpb_df)
        dataset_names.append("Financial Phrasebank")

    if 'chronicle2050' in args.datasets:
        chronicle2050_df = load_chronicle2050_dataset(script_dir)
        datasets_to_combine.append(chronicle2050_df)
        dataset_names.append("Chronicle2050")

    if 'news_api' in args.datasets:
        news_api_df = load_news_api_dataset(script_dir)
        if args.filter_domain:
            news_api_df = filter_by_domain(news_api_df, args.filter_domain)
        datasets_to_combine.append(news_api_df)
        dataset_names.append("NewsAPI")

    if 'yt' in args.datasets:
        yt_df = load_yt_dataset(script_dir)
        if args.filter_domain:
            yt_df = filter_by_domain(yt_df, args.filter_domain)
        datasets_to_combine.append(yt_df)
        dataset_names.append("YT")

    # ============================================================
    # 4. VALIDATE DATASET SELECTION
    # ============================================================

    if len(datasets_to_combine) == 0:
        print("\n❌ ERROR: No datasets selected for combination!")
        print("Please specify at least one dataset using --datasets")
        sys.exit(1)

    # ============================================================
    # 5. COMBINE DATASETS
    # ============================================================

    combined_df = combine_datasets(datasets_to_combine, dataset_names)

    # ============================================================
    # 6. EXTRACT COLUMNS
    # ============================================================

    print("\n" + "="*60)
    print("COLUMN SELECTION")
    print("="*60)

    if args.keep_all_columns:
        print("Keeping all columns from source datasets")
        final_df = combined_df
    else:
        final_df = extract_standard_columns(combined_df, args.additional_columns)

    print(f"\nFinal dataset:")
    print(f"  Shape: {final_df.shape}")
    print(f"  Columns: {list(final_df.columns)}")
    print(f"\nPreview:\n{final_df.head(7)}\n")
    print(f"\nTail:\n{final_df.tail(7)}\n")

    # ============================================================
    # 7. SAVE COMBINED DATASET
    # ============================================================

    if args.no_save:
        print("\n" + "="*60)
        print("SKIPPING SAVE (DRY RUN)")
        print("="*60)
        print("Dataset would have been saved to:")
        print(f"  {os.path.join(args.save_path, args.output_name)}.csv")
    else:
        save_combined_dataset(
            final_df,
            args.save_path,
            args.output_name,
            include_version=not args.no_version
        )

    # ============================================================
    # 8. PIPELINE COMPLETE
    # ============================================================

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Datasets combined: {len(datasets_to_combine)}")
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
        pred_ratio = round(pred_count / total * 100, 2)
        non_pred_ratio = round(non_pred_count / total * 100, 2)

        print(f"  Prediction Count     (Label=1): {pred_count}")
        print(f"  Prediction Ratio     (Label=1): {pred_ratio}%")
        print(f"  Non-Prediction Count (Label=0): {non_pred_count}")
        print(f"  Non-Prediction Ratio (Label=0): {non_pred_ratio}%")

    skip = ['Base Sentence']
    save_stacked_plot = os.path.join(args.save_path, args.output_name)
    for col in final_df.columns:
        if col not in skip:
            print(f"\n=== {col} value counts ===")
            class_values = final_df[col].value_counts()
            print(class_values)

            if col == "Dataset Name":
                print("Plotting stacked Dataset Name distribution...")
                DataVisualizing.plot_stacked_distribution(
                    final_df,
                    category_col='Dataset Name',
                    label_col='Ground Truth',
                    save_path=save_stacked_plot
                )

    print("\n" + "="*60 + "\n")