
import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import traceback
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(script_dir, '../'))

from data_processing import DataProcessing
from data_visualizing import DataVisualizing


class Tee:
    """Write output to both the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


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
        print("⚠️  Warning: 'Domain' column not found. Skipping filter.")
        return df

    original_len = len(df)
    filtered_df = df[df['Domain'] == domain_name]
    filtered_len = len(filtered_df)

    print(f"  Original size: {original_len}")
    print(f"  Filtered size: {filtered_len}")

    if original_len > 0:
        print(f"  Kept {filtered_len / original_len * 100:.1f}% of rows")
    else:
        print("  Kept 0.0% of rows")

    return filtered_df


def print_label_distribution(df, dataset_name, mode_name):
    """
    Print the label counts and percentages for a dataset.

    Always reports binary labels 0 and 1, including when either
    label has zero examples. Other observed labels and missing
    labels are also reported.
    """
    print("\n" + "-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {mode_name}")
    print(f"Total rows: {len(df)}")
    print("-" * 60)

    if 'Ground Truth' not in df.columns:
        print("⚠️  WARNING: 'Ground Truth' column not found.")
        return []

    total = len(df)
    label_counts = df['Ground Truth'].value_counts(dropna=False)
    distribution_rows = []

    # Always display both binary labels, even if a class is absent.
    for label in (0, 1):
        count = int((df['Ground Truth'] == label).sum())
        percentage = count / total * 100 if total > 0 else 0.0

        print(f"  Label {label}: {count} ({percentage:.2f}%)")

        distribution_rows.append({
            'Dataset': dataset_name,
            'Mode': mode_name,
            'Label': str(label),
            'Count': count,
            'Percentage': round(percentage, 2),
            'Total Rows': total
        })

    # Report any other observed labels, including missing values.
    for label, count in label_counts.items():
        if pd.isna(label):
            label_name = '<MISSING>'
        else:
            if label in (0, 1):
                continue
            label_name = str(label)

        count = int(count)
        percentage = count / total * 100 if total > 0 else 0.0

        print(f"  Label {label_name}: {count} ({percentage:.2f}%)")

        distribution_rows.append({
            'Dataset': dataset_name,
            'Mode': mode_name,
            'Label': label_name,
            'Count': count,
            'Percentage': round(percentage, 2),
            'Total Rows': total
        })

    return distribution_rows


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
    print("\n" + "=" * 60)
    print("COMBINE DATASETS")
    print("=" * 60)
    print(f"Combining {len(dataset_list)} datasets:")

    for name, df in zip(dataset_names, dataset_list):
        print(f"  - {name}: {len(df)} rows")

    combined_df = DataProcessing.concat_dfs(dataset_list)

    print(f"\n✓ Combined dataset shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")

    required_cols = ['Base Sentence', 'Ground Truth']
    missing_cols = [
        col for col in required_cols
        if col not in combined_df.columns
    ]

    if missing_cols:
        print(f"\n⚠️  Warning: Missing required columns: {missing_cols}")
    else:
        print(f"\n✓ All required columns present: {required_cols}")

    if 'Ground Truth' in combined_df.columns:
        print("\nGround Truth distribution:")
        print(combined_df['Ground Truth'].value_counts(dropna=False))

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
    keep_cols = standard_cols + (
        additional_cols if additional_cols else []
    )
    filtered_keep_cols = [
        col for col in keep_cols
        if col in combined_df.columns
    ]

    print(f"\nExtracting columns: {filtered_keep_cols}")

    return combined_df.loc[:, filtered_keep_cols]


def save_dataset_file(df, output_dir, file_stem, include_version=True):
    """Save a DataFrame as CSV, optionally using an incrementing version."""
    os.makedirs(output_dir, exist_ok=True)

    if include_version:
        existing_versions = []
        prefix = f"{file_stem}_v"
        for filename in os.listdir(output_dir):
            if filename.startswith(prefix) and filename.endswith('.csv'):
                version_text = filename[len(prefix):-4]
                if version_text.isdigit():
                    existing_versions.append(int(version_text))
        version = max(existing_versions, default=0) + 1
        filename = f"{file_stem}_v{version}.csv"
    else:
        filename = f"{file_stem}.csv"

    full_path = os.path.join(output_dir, filename)
    df.to_csv(full_path, index=False)

    print("\n✓ Saved dataset:")
    print(f"  Path: {full_path}")
    print(f"  Shape: {df.shape}")
    if os.path.exists(full_path):
        print(f"  Size: {os.path.getsize(full_path) / 1024:.2f} KB\n")
    return full_path


if __name__ == "__main__":

    # ============================================================
    # 1. Configuration and Argument Parsing
    # ============================================================

    base_data_path = os.path.join(script_dir, '../data')
    default_save_path = os.path.join(
        base_data_path,
        'combined_datasets/'
    )

    dataset_loader_map = {
        'synthetic':
            DataProcessing.load_synthetic_dataset,
        'financial_phrasebank':
            DataProcessing.load_financial_phrasebank_dataset,
        'chronicle2050':
            DataProcessing.load_chronicle2050_dataset,
        'timebank':
            DataProcessing.load_timebank_dataset,
        'yt':
            DataProcessing.load_yt_dataset,
        'news_api':
            DataProcessing.load_news_api_dataset,
        'mf_climate':
            DataProcessing.load_mf_climate_dataset,
        'clients_rivals_rouges':
            DataProcessing.load_clients_rivals_rouges_dataset,
        'forecast_bench':
            DataProcessing.load_forecast_bench_dataset,
        'smart_hospitals':
            DataProcessing.load_smart_hospitals_dataset
    }

    parser = argparse.ArgumentParser(
        description='Combine synthetic and real datasets for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Available Datasets:

            synthetic            - LLM-generated predictions + observations [Brinkley et al. (...)]
            financial_phrasebank - Real financial statements [Malo et al. (2014)]
            chronicle2050        - Real statements from Longbets, Horizons, etc. [Regev et al. (2024)]
            timebank             - TimeBank 1.2 annotated sentences
            yt                   - Real YouTube annotated sentences
            news_api             - Real news API annotated sentences
            mf_climate           - Real MF climate forecast predictions [B. Moe, 2024]
            clients_rivals_rouges
            forecast_bench
            smart_hospitals

        Examples:

            python3 create_combined_dataset.py --datasets synthetic

            python3 create_combined_dataset.py --datasets synthetic financial_phrasebank

            python3 create_combined_dataset.py --datasets synthetic --filter_domain finance

            python3 create_combined_dataset.py \
                --datasets synthetic financial_phrasebank chronicle2050 timebank yt news_api mf_climate clients_rivals_rouges forecast_bench smart_hospitals \
                --output_name tolsa_naacl_2026_2027 \
                --predictions_only_datasets yt news_api mf_climate clients_rivals_rouges forecast_bench
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
        '--predictions_only_datasets',
        nargs='+',
        choices=list(dataset_loader_map.keys()),
        help='Datasets to load in predictions-only mode. Default: yt.'
    )

    parser.add_argument(
        '--filter_domain',
        default=None,
        choices=['finance', 'weather', 'policy', 'health', 'sports', 'misc'],
        help='Filter datasets by domain where supported. Default: None.'
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
    # Initialize run log: mirror all terminal output to a timestamped file.
    # ============================================================
    run_started_at = datetime.now()
    run_start_clock = time.time()
    log_dir = os.path.join(args.save_path, args.output_name)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = (
        f"{args.output_name}_pipeline_{run_started_at.strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path = os.path.join(log_dir, log_filename)
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print("\n" + "=" * 60)
    print("COMBINED DATASET CREATION PIPELINE")
    print("=" * 60)
    print(f"Run started : {run_started_at.isoformat(timespec='seconds')}")
    print(f"Command     : {' '.join(sys.argv)}")
    print(f"Working dir : {os.getcwd()}")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Log file    : {log_path}")

    # ============================================================
    # 2. Display Configuration
    # ============================================================

    print("\nConfiguration:")
    print(f"  Datasets to combine       : {', '.join(args.datasets)}")
    print(f"  Predictions-only datasets : {args.predictions_only_datasets}")
    print(
        f"  Domain filter             : "
        f"{args.filter_domain if args.filter_domain else 'None (all domains)'}"
    )
    print(f"  Save path                 : {args.save_path}")
    print(f"  Output name               : {args.output_name}")
    print(f"  Save to disk              : {'No (dry run)' if args.no_save else 'Yes'}")
    print(f"  Versioning                : {'Disabled' if args.no_version else 'Enabled'}")

    if args.additional_columns:
        print(f"  Additional columns        : {', '.join(args.additional_columns)}")

    print()

    # ============================================================
    # 3. Load Datasets and Audit Distributions
    # ============================================================

    domain_filterable = {'synthetic', 'news_api', 'yt'}

    datasets_to_combine = []
    dataset_names = []
    distribution_rows = []

    for dataset_key in args.datasets:
        loader = dataset_loader_map[dataset_key]

        print("\n" + "=" * 70)
        print(f"AUDITING DATASET: {dataset_key}")
        print("=" * 70)

        # Inspect both loading modes for every selected dataset.
        for predictions_only in (False, True):
            mode_name = (
                'predictions_only'
                if predictions_only
                else 'all_rows'
            )

            audit_df = loader(
                script_dir,
                predictions_only=predictions_only,
                visualize=False
            )

            if args.filter_domain and dataset_key in domain_filterable:
                audit_df = filter_by_domain(
                    audit_df,
                    args.filter_domain
                )

            distribution_rows.extend(
                print_label_distribution(
                    audit_df,
                    dataset_key,
                    mode_name
                )
            )

        # Preserve the original CLI behavior for the combined dataset.
        use_predictions_only = (
            dataset_key in (args.predictions_only_datasets or [])
        )

        df = loader(
            script_dir,
            predictions_only=use_predictions_only,
            visualize=False
        )

        if args.filter_domain and dataset_key in domain_filterable:
            df = filter_by_domain(df, args.filter_domain)

        print(
            f"\nSelected for combination: {dataset_key} | "
            f"predictions_only={use_predictions_only} | "
            f"Rows: {len(df)}"
        )

        datasets_to_combine.append(df)
        dataset_names.append(dataset_key)

    # Save the distribution audit unless --no_save was specified.
    if distribution_rows and not args.no_save:
        audit_dir = os.path.join(args.save_path, args.output_name)
        os.makedirs(audit_dir, exist_ok=True)

        audit_path = os.path.join(
            audit_dir,
            f"{args.output_name}_distribution_audit.csv"
        )

        distribution_df = pd.DataFrame(distribution_rows)
        distribution_df.to_csv(audit_path, index=False)

        print("\n" + "=" * 60)
        print("DISTRIBUTION AUDIT SAVED")
        print("=" * 60)
        print(f"Path: {audit_path}")

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

    combined_df = combine_datasets(
        datasets_to_combine,
        dataset_names
    )

    # ============================================================
    # 6. Prepare the three requested outputs
    # ============================================================

    print("\n" + "=" * 60)
    print("PREPARE OUTPUT DATASETS")
    print("=" * 60)

    # Rows with a missing Base Sentence are retained separately for review.
    null_mask = combined_df['Base Sentence'].isnull()
    filtered_out_df = combined_df[null_mask].copy()
    all_columns_df = combined_df[~null_mask].copy()
    main_columns_df = extract_standard_columns(
        all_columns_df,
        args.additional_columns
    )

    print(f"Rows filtered out (null 'Base Sentence'): {len(filtered_out_df)}")
    print(f"All-columns dataset shape              : {all_columns_df.shape}")
    print(f"Main-columns dataset shape             : {main_columns_df.shape}")
    print(f"Filtered-out rows shape                : {filtered_out_df.shape}")

    print("\nAll-columns dataset columns:")
    print(list(all_columns_df.columns))
    print("\nMain-columns dataset columns:")
    print(list(main_columns_df.columns))

    print(f"\nMain-columns preview:\n{main_columns_df.head(7)}\n")

    # ============================================================
    # 7. Save all three CSV files in the same output directory
    # ============================================================

    output_dir = os.path.join(args.save_path, args.output_name)

    if args.no_save:
        print("\n" + "=" * 60)
        print("SKIPPING SAVE (DRY RUN)")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Would save: {args.output_name}_all_columns.csv")
        print(f"Would save: {args.output_name}_main_columns.csv")
        print(f"Would save: {args.output_name}_filtered_out_rows.csv")
    else:
        save_dataset_file(
            all_columns_df,
            output_dir,
            f"{args.output_name}_all_columns",
            include_version=not args.no_version
        )
        save_dataset_file(
            main_columns_df,
            output_dir,
            f"{args.output_name}_main_columns",
            include_version=not args.no_version
        )
        save_dataset_file(
            filtered_out_df,
            output_dir,
            f"{args.output_name}_filtered_out_rows",
            include_version=not args.no_version
        )

    # ============================================================
    # 8. Visualize Distribution
    # ============================================================

    if 'Dataset Name' in main_columns_df.columns and not args.no_save:
        print("\n" + "=" * 60)
        print("SAVE DISTRIBUTION PLOT")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"{args.output_name}_distribution.png"

        print("Plotting stacked Dataset Name distribution...")
        DataVisualizing.plot_stacked_distribution(
            main_columns_df,
            category_col='Dataset Name',
            label_col='Ground Truth',
            save_path=output_dir,
            filename=plot_filename
        )
        print(f"✓ Saved plot: {os.path.join(output_dir, plot_filename)}")

    # ============================================================
    # 9. Pipeline Complete
    # ============================================================

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Datasets combined        : {len(datasets_to_combine)}")
    print(f"All-columns dataset shape: {all_columns_df.shape}")
    print(f"Main-columns dataset shape: {main_columns_df.shape}")
    print(f"Filtered-out rows shape  : {filtered_out_df.shape}")

    if args.filter_domain:
        print(f"Domain filter applied: {args.filter_domain}")

    if not args.no_save:
        print(f"✓ Saved outputs under: {output_dir}")

    print("\nSummary statistics (main-columns dataset):")
    if 'Ground Truth' in main_columns_df.columns:
        total = len(main_columns_df)
        pred_count = (main_columns_df['Ground Truth'] == 1).sum()
        non_pred_count = (main_columns_df['Ground Truth'] == 0).sum()
        pred_pct = pred_count / total * 100 if total else 0.0
        non_pred_pct = non_pred_count / total * 100 if total else 0.0
        print(f"  Prediction Count     (Label=1): {pred_count} ({pred_pct:.2f}%)")
        print(f"  Non-Prediction Count (Label=0): {non_pred_count} ({non_pred_pct:.2f}%)")

    elapsed_seconds = time.time() - run_start_clock
    print(f"Run finished : {datetime.now().isoformat(timespec='seconds')}")
    print(f"Elapsed time : {elapsed_seconds:.2f} seconds")
    print(f"Full log     : {log_path}")
    print("\n" + "=" * 60 + "\n")
    log_file.flush()
