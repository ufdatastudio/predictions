"""
Concatenate all CSV files in a directory into a single CSV file.
================================================================
Usage:
    python concatenate_csvs.py --input_dir C:/path/to/csv_folder --output C:/path/to/output/combined.csv

Optional — drop duplicates and rows with missing values:
    python concatenate_csvs.py --input_dir C:/path/to/csv_folder --output C:/path/to/combined.csv --clean
"""

import os
import argparse
import pandas as pd

# ============================================================
# CONSTANTS — these match your RAW Time Banks CSV schema
# (File Name, Sentence, Label, Human Annotation, Human Reasoning)
# ============================================================
TEXT_COLUMN  = 'Sentence'
LABEL_COLUMN = 'Human Annotation'


def concatenate_csvs(input_dir, output_path, clean=False):
    print("\n" + "="*40)
    print("CONCATENATE CSV FILES")
    print("="*40)

    # ---- Find all CSV files (Mac-safe: skip directories/.DS_Store) ----
    csv_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.csv') and os.path.isfile(os.path.join(input_dir, f))
    ])

    if not csv_files:
        raise ValueError(f"No CSV files found in: {input_dir}")

    print(f"Found       : {len(csv_files)} CSV files")

    # ---- Load and concatenate ----
    dfs = []
    skipped = 0
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Skip files missing required columns
            if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
                print(f"  Skipping {os.path.basename(csv_file)} — missing required columns")
                skipped += 1
                continue

            dfs.append(df)

        except Exception as e:
            print(f"  Skipping {os.path.basename(csv_file)} — error: {e}")
            skipped += 1

    if not dfs:
        raise ValueError(
            f"No valid CSV files could be loaded.\n"
            f"Check that your CSVs contain '{TEXT_COLUMN}' and '{LABEL_COLUMN}' columns."
        )

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded      : {len(dfs)} files ({skipped} skipped)")
    print(f"Combined    : {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")

    # ---- Optional cleaning ----
    if clean:
        print("\n--- Cleaning ---")

        # Drop rows with missing values in key columns
        before = len(combined_df)
        combined_df = combined_df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
        dropped_na = before - len(combined_df)
        if dropped_na > 0:
            print(f"Dropped {dropped_na} rows with missing values")

        # Ensure label is integer
        combined_df[LABEL_COLUMN] = combined_df[LABEL_COLUMN].astype(int)

        # Remap unexpected labels to 0
        unexpected = set(combined_df[LABEL_COLUMN].unique()) - {0, 1}
        if unexpected:
            print(f"Unexpected labels found: {unexpected} — remapping to 0")
            combined_df[LABEL_COLUMN] = combined_df[LABEL_COLUMN].apply(
                lambda x: x if x in {0, 1} else 0
            )

        # Drop duplicate sentences
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
        dropped_dupes = before_dedup - len(combined_df)
        if dropped_dupes > 0:
            print(f"Dropped {dropped_dupes} duplicate sentences")

    # ---- Final summary ----
    print("\n--- Final Dataset ---")
    print(f"Shape       : {combined_df.shape}")
    print(f"Label distribution:\n{combined_df[LABEL_COLUMN].value_counts()}")

    # ---- Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved to  : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate all CSV files in a directory into a single CSV.'
    )
    parser.add_argument(
        '--input_dir', required=True,
        help='Directory containing CSV files to concatenate.'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output path for the combined CSV file. Example: C:/path/to/combined.csv'
    )
    parser.add_argument(
        '--clean', action='store_true',
        help='If set, drops missing values, deduplicates sentences, and remaps unexpected labels.'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    concatenate_csvs(args.input_dir, args.output, clean=args.clean)

    print("\n" + "="*40)
    print("DONE")
    print("="*40)


if __name__ == '__main__':
    main()