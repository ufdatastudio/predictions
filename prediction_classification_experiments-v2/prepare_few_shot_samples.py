
#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

EXPERIMENT = "tolsa_naacl_2026_2027-final"

COMBINED_DATASET = Path(
    "../data/combined_datasets/"
    "tolsa_naacl_2026_2027-final/"
    "tolsa_naacl_2026_2027_main_columns_v1.csv"
)

RESULTS_DIR = Path(
    "../data/classification_results/"
    f"naacl_2026_submission/{EXPERIMENT}"
)

REQUIRED_COLUMNS = ["Dataset Name", "Base Sentence", "Ground Truth"]
MATCH_COLUMNS = ["Base Sentence", "Ground Truth"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare seed-specific few-shot examples."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=[3, 7, 33],
        help="Random seed whose existing data splits will be checked.",
    )
    parser.add_argument(
        "--combined_dataset",
        type=Path,
        default=COMBINED_DATASET,
        help="Path to the combined dataset.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTS_DIR,
        help="Root directory containing the existing seed splits.",
    )
    return parser.parse_args()


def load_csv(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, dtype="string")

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    return df


def get_sentence_label_pairs(df):
    """Return valid (Base Sentence, Ground Truth) pairs."""
    pairs = set()

    for sentence, label in df[MATCH_COLUMNS].itertuples(
        index=False, name=None
    ):
        if pd.isna(sentence) or pd.isna(label):
            continue

        if not sentence.strip() or not label.strip():
            continue

        # Preserve the original sentence and label for exact matching.
        pairs.add((sentence, label))

    return pairs


def main():
    args = parse_args()
    seed = args.seed

    seed_dir = args.results_dir / f"seed{seed}" / "in_domain"
    split_dir = seed_dir / "splits"

    train_path = split_dir / "x_y_train_set.csv"
    val_path = split_dir / "x_y_val_set.csv"
    test_path = split_dir / "x_y_test_set.csv"

    output_path = seed_dir / "llm" / "few-shot.csv"

    print("=" * 65)
    print("Preparing few-shot samples")
    print(f"Seed: {seed}")
    print(f"Combined dataset: {args.combined_dataset}")
    print(f"Output: {output_path}")
    print("=" * 65)

    combined = load_csv(
        args.combined_dataset, REQUIRED_COLUMNS
    )
    train = load_csv(train_path, MATCH_COLUMNS)
    val = load_csv(val_path, MATCH_COLUMNS)
    test = load_csv(test_path, MATCH_COLUMNS)

    train_pairs = get_sentence_label_pairs(train)
    val_pairs = get_sentence_label_pairs(val)
    test_pairs = get_sentence_label_pairs(test)

    print(f"Combined dataset rows: {len(combined):,}")
    print(f"Unique train pairs: {len(train_pairs):,}")
    print(f"Unique validation pairs: {len(val_pairs):,}")
    print(f"Unique test pairs: {len(test_pairs):,}")

    # Only accept pairs found in train and absent from both
    # validation and test.
    eligible_pairs = train_pairs - val_pairs - test_pairs

    selected = []
    missing_datasets = []

    # Preserve the dataset order and row order of the combined CSV.
    for dataset_name, group in combined.groupby(
        "Dataset Name", sort=False, dropna=False
    ):
        found = False

        for _, row in group.iterrows():
            sentence = row["Base Sentence"]
            label = row["Ground Truth"]

            if pd.isna(sentence) or pd.isna(label):
                continue

            if not sentence.strip() or not label.strip():
                continue

            pair = (sentence, label)

            if pair in eligible_pairs:
                selected.append({
                    "Dataset Name": dataset_name,
                    "Base Sentence": sentence,
                    "Ground Truth": label,
                })
                found = True
                break

        if not found:
            missing_datasets.append(
                "<missing Dataset Name>"
                if pd.isna(dataset_name)
                else dataset_name
            )

    output_columns = REQUIRED_COLUMNS
    output = pd.DataFrame(selected, columns=output_columns)

    if output.empty:
        raise RuntimeError(
            f"No eligible few-shot examples found for seed {seed}. "
            "No output file was written."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print(f"Seed: {seed}")
    print(f"Datasets in combined data: {combined['Dataset Name'].nunique(dropna=False)}")
    print(f"Few-shot examples selected: {len(output)}")
    print(f"Datasets without eligible examples: {len(missing_datasets)}")
    print(f"Saved: {output_path}")

    if missing_datasets:
        print("\nDatasets without eligible examples:")
        for name in missing_datasets:
            print(f"  - {name}")

    print("\nSelected examples per dataset:")
    print(output[["Dataset Name", "Ground Truth"]].to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()