import os
import argparse
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================
# Final unified column names (output)
FINAL_TEXT_COLUMN      = 'sentence'
FINAL_LABEL_COLUMN     = 'labels'
FINAL_REASONING_COLUMN = 'reasoning'
FINAL_COLUMNS = [FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN, FINAL_REASONING_COLUMN]

# Time Banks column names (RAW schema — output of concatenate_csvs.py)
# Raw columns: File Name, Sentence, Label, Human Annotation, Human Reasoning
TIMEBANKS_TEXT_COLUMN      = 'Sentence'
TIMEBANKS_LABEL_COLUMN     = 'Human Annotation'
TIMEBANKS_REASONING_COLUMN = 'Human Reasoning'

# Chronicle column names (input)
CHRONICLE_TEXT_COLUMN      = 'sentence'
CHRONICLE_LABEL_COLUMN     = 'labels'
CHRONICLE_REASONING_COLUMN = 'reasoning'

# Chronicle label mapping (string -> int)
CHRONICLE_LABEL_MAP = {
    'prediction':     1,
    'not-prediction': 0,
}

# Adriano column names (input)
ADRIANO_TEXT_COLUMN  = 'Base Sentence'
ADRIANO_LABEL_COLUMN = 'Sentence Label'

# Adriano folder structure:
#   prediction_adriano/
#       batch_3-prediction/
#           batch_3-from_df.csv
#       batch_4-prediction/   <- may be missing entirely
#       ...
#       batch_67-prediction/
#           batch_67-from_df.csv
ADRIANO_SUBFOLDER_PATTERN = 'batch_'   # subfolders look like batch_16-prediction
ADRIANO_FILE_PATTERN      = 'batch_'   # files look like batch_16-from_df.csv


# ============================================================
# SOURCE 1 — TIME BANKS
# ============================================================
def load_timebanks_dataset(path):
    """
    Load our combined Time Banks dataset (raw output of concatenate_csvs.py).
    Raw schema: File Name, Sentence, Label, Human Annotation, Human Reasoning
    (extra columns like 'Unnamed: 0' from stray CSV indices are dropped).

    Renames to final unified schema:
        Sentence          -> sentence
        Human Annotation  -> labels
        Human Reasoning   -> reasoning
    """
    print("\n" + "="*40)
    print("LOAD TIME BANKS DATASET")
    print("="*40)
    print(f"Path: {path}")

    df = pd.read_csv(path)

    # Drop stray index columns that sometimes appear (e.g. 'Unnamed: 0')
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"WARNING: Dropped stray index column(s): {unnamed_cols}")

    missing = [c for c in [TIMEBANKS_TEXT_COLUMN, TIMEBANKS_LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in Time Banks dataset: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if TIMEBANKS_REASONING_COLUMN not in df.columns:
        print(f"WARNING: '{TIMEBANKS_REASONING_COLUMN}' column not found - filling with N/A")
        df[TIMEBANKS_REASONING_COLUMN] = 'N/A'

    df = df.rename(columns={
        TIMEBANKS_TEXT_COLUMN:      FINAL_TEXT_COLUMN,
        TIMEBANKS_LABEL_COLUMN:     FINAL_LABEL_COLUMN,
        TIMEBANKS_REASONING_COLUMN: FINAL_REASONING_COLUMN,
    })
    df = df[FINAL_COLUMNS]

    # Drop rows with missing text/label
    before = len(df)
    df = df.dropna(subset=[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"WARNING: Dropped {dropped} rows with missing values")

    df[FINAL_LABEL_COLUMN] = df[FINAL_LABEL_COLUMN].astype(int)

    # Guard against label typos (e.g. '9' instead of '0')
    unexpected_labels = set(df[FINAL_LABEL_COLUMN].unique()) - {0, 1}
    if unexpected_labels:
        print(f"WARNING: Unexpected labels found: {unexpected_labels} - remapping to 0")
        df[FINAL_LABEL_COLUMN] = df[FINAL_LABEL_COLUMN].apply(lambda x: x if x in {0, 1} else 0)

    print(f"Shape              : {df.shape}")
    print(f"Label distribution :")
    print(df[FINAL_LABEL_COLUMN].value_counts().to_string())

    return df


# ============================================================
# SOURCE 2 — CHRONICLE (teammate's file)
# ============================================================
def load_chronicle_dataset(path):
    """
    Load teammate's Chronicle dataset.
    Maps string labels 'prediction'/'not-prediction' -> 1/0.
    """
    print("\n" + "="*40)
    print("LOAD CHRONICLE DATASET")
    print("="*40)
    print(f"Path: {path}")

    df = pd.read_csv(path)

    missing = [c for c in [CHRONICLE_TEXT_COLUMN, CHRONICLE_LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in Chronicle dataset: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if CHRONICLE_REASONING_COLUMN not in df.columns:
        print(f"WARNING: '{CHRONICLE_REASONING_COLUMN}' column not found - filling with N/A")
        df[CHRONICLE_REASONING_COLUMN] = 'N/A'

    print(f"Original shape     : {df.shape}")
    print(f"Original labels    :\n{df[CHRONICLE_LABEL_COLUMN].value_counts().to_string()}")

    # ---- Map string labels to integers ----
    unexpected_labels = set(df[CHRONICLE_LABEL_COLUMN].unique()) - set(CHRONICLE_LABEL_MAP.keys())
    if unexpected_labels:
        print(f"WARNING: Unexpected labels found: {unexpected_labels} - will be dropped")
        df = df[df[CHRONICLE_LABEL_COLUMN].isin(CHRONICLE_LABEL_MAP.keys())].reset_index(drop=True)

    df[CHRONICLE_LABEL_COLUMN] = df[CHRONICLE_LABEL_COLUMN].map(CHRONICLE_LABEL_MAP)

    df = df.rename(columns={
        CHRONICLE_TEXT_COLUMN:      FINAL_TEXT_COLUMN,
        CHRONICLE_LABEL_COLUMN:     FINAL_LABEL_COLUMN,
        CHRONICLE_REASONING_COLUMN: FINAL_REASONING_COLUMN,
    })
    df = df[FINAL_COLUMNS]

    before = len(df)
    df = df.dropna(subset=[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"WARNING: Dropped {dropped} rows with missing values")

    df[FINAL_LABEL_COLUMN] = df[FINAL_LABEL_COLUMN].astype(int)

    print(f"\nRestructured shape    : {df.shape}")
    print(f"Restructured labels   :")
    print(df[FINAL_LABEL_COLUMN].value_counts().to_string())

    return df


# ============================================================
# SOURCE 3 — ADRIANO (65 subfolders of batch_X-from_df.csv)
# ============================================================
def load_adriano_dataset(root_dir):
    """
    Recursively scan all subfolders under root_dir for files matching
    'batch_*-from_df.csv', load each, and combine.

    Maps:
        Base Sentence  -> sentence
        Sentence Label -> labels
        reasoning      -> 'N/A' (Adriano's data has no reasoning column;
                                   it is LLM-generated, 100% class 1)

    Drops all other metadata columns (Domain, Model Name, API Name,
    Batch ID, Temperature, Top P, Generated At, Prompt Used, Template Number).
    """
    print("\n" + "="*40)
    print("LOAD ADRIANO DATASET")
    print("="*40)
    print(f"Root directory: {root_dir}")

    # ---- Step 1: list expected batch subfolders (batch_X-prediction) ----
    subfolders = sorted([
        d for d in os.listdir(root_dir)
        if d.startswith(ADRIANO_SUBFOLDER_PATTERN)
        and os.path.isdir(os.path.join(root_dir, d))
    ])
    print(f"Subfolders found : {len(subfolders)}")

    # ---- Step 2: for each subfolder, find its batch_X-from_df.csv file ----
    batch_files = []
    missing_files_in = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        matched = [
            f for f in os.listdir(subfolder_path)
            if f.startswith(ADRIANO_FILE_PATTERN)
            and f.endswith('.csv')
            and os.path.isfile(os.path.join(subfolder_path, f))
        ]
        if matched:
            # If multiple matches somehow exist, take all of them
            for f in matched:
                batch_files.append(os.path.join(subfolder_path, f))
        else:
            missing_files_in.append(subfolder)

    batch_files = sorted(batch_files)

    if missing_files_in:
        print(f"WARNING: {len(missing_files_in)} subfolder(s) had no matching CSV file:")
        for sf in missing_files_in:
            print(f"    - {sf}")

    if not batch_files:
        raise ValueError(f"No 'batch_*-from_df.csv' files found under: {root_dir}")

    print(f"Found       : {len(batch_files)} batch files")

    dfs = []
    skipped = 0
    for batch_file in batch_files:
        try:
            df = pd.read_csv(batch_file)

            if ADRIANO_TEXT_COLUMN not in df.columns or ADRIANO_LABEL_COLUMN not in df.columns:
                print(f"  WARNING: Skipping {os.path.basename(batch_file)} - missing required columns")
                skipped += 1
                continue

            df = df[[ADRIANO_TEXT_COLUMN, ADRIANO_LABEL_COLUMN]].copy()
            dfs.append(df)

        except Exception as e:
            print(f"  WARNING: Skipping {os.path.basename(batch_file)} - error: {e}")
            skipped += 1

    if not dfs:
        raise ValueError(
            f"No valid batch files could be loaded from: {root_dir}\n"
            f"All files were skipped. Check that they contain "
            f"'{ADRIANO_TEXT_COLUMN}' and '{ADRIANO_LABEL_COLUMN}' columns."
        )

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded      : {len(dfs)} files ({skipped} skipped)")
    print(f"Combined shape (before restructure): {combined.shape}")

    # ---- Rename to final unified schema ----
    combined = combined.rename(columns={
        ADRIANO_TEXT_COLUMN:  FINAL_TEXT_COLUMN,
        ADRIANO_LABEL_COLUMN: FINAL_LABEL_COLUMN,
    })

    # ---- Reasoning is not available for Adriano's data ----
    combined[FINAL_REASONING_COLUMN] = 'N/A'
    combined = combined[FINAL_COLUMNS]

    # ---- Drop rows with missing values ----
    before = len(combined)
    combined = combined.dropna(subset=[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN]).reset_index(drop=True)
    dropped = before - len(combined)
    if dropped > 0:
        print(f"WARNING: Dropped {dropped} rows with missing values")

    combined[FINAL_LABEL_COLUMN] = combined[FINAL_LABEL_COLUMN].astype(int)

    print(f"\nRestructured shape    : {combined.shape}")
    print(f"Restructured labels   :")
    print(combined[FINAL_LABEL_COLUMN].value_counts().to_string())

    return combined


# ============================================================
# CONCATENATE & SAVE
# ============================================================
def concatenate_and_save(dfs_with_names, output_path):
    """Concatenate all source datasets, deduplicate, and save."""
    print("\n" + "="*40)
    print("CONCATENATE ALL DATASETS")
    print("="*40)

    for name, df in dfs_with_names:
        print(f"{name:<12}: {df.shape[0]} rows")

    combined_df = pd.concat([df for _, df in dfs_with_names], ignore_index=True)
    print(f"\nCombined shape (before dedup): {combined_df.shape}")

    # ---- Deduplicate on sentence ----
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
    dropped_dupes = before_dedup - len(combined_df)
    if dropped_dupes > 0:
        print(f"WARNING: Dropped {dropped_dupes} duplicate sentences")

    # ---- Final summary ----
    print("\n" + "="*40)
    print("FINAL DATASET SUMMARY")
    print("="*40)
    print(f"Final shape        : {combined_df.shape}")
    print(f"\nLabel distribution :")
    label_counts = combined_df[FINAL_LABEL_COLUMN].value_counts()
    print(label_counts.to_string())

    if 0 in label_counts.index and 1 in label_counts.index:
        print(f"\nClass ratio (0:1)  : {label_counts[0]}:{label_counts[1]} "
              f"({label_counts[0]/label_counts[1]:.2f}:1)")
        print(f"Class 0 percentage : {label_counts[0]/len(combined_df)*100:.1f}%")
        print(f"Class 1 percentage : {label_counts[1]/len(combined_df)*100:.1f}%")

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved to           : {output_path}")

    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Merge Time Banks, Chronicle, and Adriano datasets into one final training CSV.'
    )
    parser.add_argument(
        '--timebanks_dataset', required=True,
        help='Path to our combined Time Banks CSV (schema: sentence, labels, reasoning).'
    )
    parser.add_argument(
        '--chronicle_dataset', required=True,
        help='Path to Chronicle CSV (schema: sentence, labels [string], reasoning).'
    )
    parser.add_argument(
        '--adriano_dataset', required=True,
        help=(
            "Path to the 'prediction_adriano' root folder. "
            "Expects subfolders named 'batch_X-prediction', each containing "
            "one 'batch_X-from_df.csv' file."
        )
    )
    parser.add_argument(
        '--output', required=True,
        help='Output path for the final combined CSV. Example: data/final_combined.csv'
    )
    args = parser.parse_args()

    if not args.output.endswith('.csv'):
        raise ValueError(
            f"--output must be a full file path ending in .csv\n"
            f"Got: {args.output}\n"
            f"Example: data/final_combined.csv"
        )

    # ---- Run pipeline ----
    timebanks_df = load_timebanks_dataset(args.timebanks_dataset)
    chronicle_df = load_chronicle_dataset(args.chronicle_dataset)
    adriano_df   = load_adriano_dataset(args.adriano_dataset)

    combined_df = concatenate_and_save(
        [
            ('Time Banks', timebanks_df),
            ('Chronicle',  chronicle_df),
            ('Adriano',    adriano_df),
        ],
        args.output
    )

    print("\n" + "="*40)
    print("DONE")
    print("="*40)


if __name__ == '__main__':
    main()