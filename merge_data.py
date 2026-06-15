import os
import argparse
import pandas as pd

# CONSTANTS

#Our Schema
OUR_TEXT_COLUMN      = 'Sentence'
OUR_LABEL_COLUMN     = 'Human Annotation'
OUR_REASONING_COLUMN = 'Human Reasoning'

#Their schema
THEIR_TEXT_COLUMN      = 'sentence'
THEIR_LABEL_COLUMN     = 'labels'
THEIR_REASONING_COLUMN = 'reasoning'

#Define the final file schema:
FINAL_TEXT_COLUMN      = 'sentence'
FINAL_LABEL_COLUMN     = 'labels'
FINAL_REASONING_COLUMN = 'reasoning'

# Label mapping 
LABEL_MAP = {
    'prediction':     1,
    'not-prediction': 0,
}


def load_our_dataset(path):
    """
    Load and validate our timebanks combined dataset.
    Renames columns to final unified schema:
        Sentence       -> sentence
        Human Annotation -> labels
        Human Reasoning  -> reasoning
    """
    print("\n" + "="*40)
    print("LOAD OUR DATASET")
    print("="*40)
    print(f"Path: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    missing = [c for c in [OUR_TEXT_COLUMN, OUR_LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in our dataset: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Add reasoning column if missing
    if OUR_REASONING_COLUMN not in df.columns:
        print(f"'{OUR_REASONING_COLUMN}' column not found — filling with NaN")
        df[OUR_REASONING_COLUMN] = None

    # Rename to final unified schema
    df = df.rename(columns={
        OUR_TEXT_COLUMN:      FINAL_TEXT_COLUMN,
        OUR_LABEL_COLUMN:     FINAL_LABEL_COLUMN,
        OUR_REASONING_COLUMN: FINAL_REASONING_COLUMN,
    })

    # Keep only final columns
    df = df[[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN, FINAL_REASONING_COLUMN]]

    # Ensure label is integer
    df[FINAL_LABEL_COLUMN] = df[FINAL_LABEL_COLUMN].astype(int)

    print(f"Shape              : {df.shape}")
    print(f"Columns            : {list(df.columns)}")
    print(f"Label distribution :")
    print(df[FINAL_LABEL_COLUMN].value_counts().to_string())

    return df


def load_and_restructure_their_dataset(path):
    """
    Load teammate's dataset and restructure it to match final unified schema.
    - Maps string labels: 'prediction' -> 1, 'not-prediction' -> 0
    - Keeps reasoning column
    - Renames to final unified schema (already matches: sentence, labels, reasoning)
    """
    print("\n" + "="*40)
    print("LOAD & RESTRUCTURE TEAMMATE'S DATASET")
    print("="*40)
    print(f"Path: {path}")

    df = pd.read_csv(path)

    print(f"Original shape     : {df.shape}")
    print(f"Original columns   : {list(df.columns)}")
    print(f"Original labels    :\n{df[THEIR_LABEL_COLUMN].value_counts().to_string()}")

    # ---- Validate required columns ----
    missing = [c for c in [THEIR_TEXT_COLUMN, THEIR_LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in teammate's dataset: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ---- Add reasoning column if missing ----
    if THEIR_REASONING_COLUMN not in df.columns:
        print(f"'{THEIR_REASONING_COLUMN}' column not found — filling with NaN")
        df[THEIR_REASONING_COLUMN] = None

    # ---- Map string labels to integers ----
    unexpected_labels = set(df[THEIR_LABEL_COLUMN].unique()) - set(LABEL_MAP.keys())
    if unexpected_labels:
        print(f"Unexpected labels found: {unexpected_labels} — will be dropped")
        df = df[df[THEIR_LABEL_COLUMN].isin(LABEL_MAP.keys())].reset_index(drop=True)

    df[THEIR_LABEL_COLUMN] = df[THEIR_LABEL_COLUMN].map(LABEL_MAP)

    # ---- Rename to final unified schema ----
    df = df.rename(columns={
        THEIR_TEXT_COLUMN:      FINAL_TEXT_COLUMN,
        THEIR_LABEL_COLUMN:     FINAL_LABEL_COLUMN,
        THEIR_REASONING_COLUMN: FINAL_REASONING_COLUMN,
    })

    # ---- Keep only final columns ----
    df = df[[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN, FINAL_REASONING_COLUMN]]

    # ---- Drop rows with missing values in key columns ----
    before = len(df)
    df = df.dropna(subset=[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values")

    # Ensure label is integer
    df[FINAL_LABEL_COLUMN] = df[FINAL_LABEL_COLUMN].astype(int)

    print(f"\nRestructured shape    : {df.shape}")
    print(f"Restructured columns  : {list(df.columns)}")
    print(f"Restructured labels   :")
    print(df[FINAL_LABEL_COLUMN].value_counts().to_string())

    return df


def concatenate_and_save(our_df, their_df, output_path):
    """Concatenate both datasets, deduplicate, and save."""
    print("\n" + "="*40)
    print("CONCATENATE DATASETS")
    print("="*40)

    # ---- Align columns — use final unified schema ----
    final_cols = [FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN, FINAL_REASONING_COLUMN]
    print(f"Final columns      : {final_cols}")

    our_df   = our_df[final_cols]
    their_df = their_df[final_cols]

    # ---- Concatenate ----
    combined_df = pd.concat([our_df, their_df], ignore_index=True)
    print(f"\nCombined shape     : {combined_df.shape}")

    # ---- Deduplicate on Sentence ----
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
    dropped_dupes = before_dedup - len(combined_df)
    if dropped_dupes > 0:
        print(f"Dropped {dropped_dupes} duplicate sentences")

    # ---- Final summary ----
    print("\n" + "="*40)
    print("FINAL DATASET SUMMARY")
    print("="*40)
    print(f"Final shape        : {combined_df.shape}")
    print(f"\nLabel distribution :")
    label_counts = combined_df[FINAL_LABEL_COLUMN].value_counts()
    print(label_counts.to_string())
    print(f"\nClass ratio (0:1)  : {label_counts[0]}:{label_counts[1]} "
          f"({label_counts[0]/label_counts[1]:.2f}:1)")
    print(f"Class 0 percentage : {label_counts[0]/len(combined_df)*100:.1f}%")
    print(f"Class 1 percentage : {label_counts[1]/len(combined_df)*100:.1f}%")

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved to         : {output_path}")

    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Restructure and merge teammate dataset with our timebanks dataset.'
    )
    parser.add_argument(
        '--our_dataset', required=True,
        help='Path to our combined timebanks CSV. Must have "Sentence" and "Human Annotation" columns.'
    )
    parser.add_argument(
        '--their_dataset', required=True,
        help='Path to teammate\'s chronicle2050 CSV. Must have "sentence" and "labels" columns.'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output path for the final combined CSV. Example: data/final_combined.csv'
    )
    args = parser.parse_args()

    # Validate output path
    if not args.output.endswith('.csv'):
        raise ValueError(
            f"--output must be a full file path ending in .csv\n"
            f"Got: {args.output}\n"
            f"Example: data/final_combined.csv"
        )

    # ---- Run pipeline ----
    our_df   = load_our_dataset(args.our_dataset)
    their_df = load_and_restructure_their_dataset(args.their_dataset)
    combined_df = concatenate_and_save(our_df, their_df, args.output)

    print("\n" + "="*40)
    print("DONE")
    print("="*40)


if __name__ == '__main__':
    main()