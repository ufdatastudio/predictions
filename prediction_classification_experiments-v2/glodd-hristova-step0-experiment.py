# =============================================================================
# Glodd & Hristova (2023)
# Step 0: Rule-Based Forward Looking Statement (FLS) Extraction
#
# Reproduces ONLY Step 0 from:
# "Extraction of Forward-looking Financial Information for Stock Price Prediction
# from Annual Reports Using NLP Techniques"
#
# DOES:
#   - Keyword/Rule-Based FLS Identification
#
# DOES NOT:
#   - FinBERT Sentiment Analysis (Step 1/2)
#   - Stock Price Prediction (Step 3)
# =============================================================================

import os
import re
import sys
import argparse
import pandas as pd

from datetime import datetime

# =============================================================================
# PROJECT IMPORTS
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

if script_dir not in sys.path:
    sys.path.append(script_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)

from data_processing import DataProcessing


# =============================================================================
# FLS RULES FROM PAPER
# =============================================================================

TIME_WORDS = [
    "next",
    "subsequent",
    "following",
    "upcoming",
    "incoming",
    "coming",
    "succeeding",
    "carryforward"
]

TIME_PERIODS = [
    "month",
    "quarter",
    "year",
    "fiscal",
    "taxable",
    "period"
]

PREDICTIVE_VERBS = [
    "aim",
    "anticipate",
    "assume",
    "commit",
    "estimate",
    "expect",
    "forecast",
    "foresee",
    "hope",
    "intend",
    "plan",
    "predict",
    "project",
    "seek",
    "target"
]

UNCERTAIN_TERMS = [
    "may",
    "might",
    "could",
    "possibly",
    "perhaps"
]


# =============================================================================
# RULE FUNCTIONS
# =============================================================================

def contains_time_period_phrase(text):

    text = text.lower()

    for time_word in TIME_WORDS:
        for period in TIME_PERIODS:

            pattern = rf"\b{time_word}\s+{period}s?\b"

            if re.search(pattern, text):
                return True

    return False


def contains_predictive_verb(text):

    text = text.lower()

    for word in PREDICTIVE_VERBS:

        pattern = rf"\b{word}(s|ed|ing)?\b"

        if re.search(pattern, text):
            return True

    return False


def contains_future_year(text, publication_year=2023):

    years = re.findall(r"\b[12]\d{3}\b", text)

    years = [int(x) for x in years]

    for year in years:

        if year > publication_year:
            return True

    return False


def contains_uncertainty(text):

    text = text.lower()

    for token in UNCERTAIN_TERMS:

        if token in text:
            return True

    return False


def identify_fls(sentence, publication_year):

    rule_time = contains_time_period_phrase(sentence)

    rule_predictive = contains_predictive_verb(sentence)

    rule_year = contains_future_year(
        sentence,
        publication_year
    )

    rule_uncertain = contains_uncertainty(sentence)

    positive_signal = (
        rule_time
        or rule_predictive
        or rule_year
    )

    fls_prediction = int(
        positive_signal
        and not rule_uncertain
    )

    return pd.Series({
        "Rule_TimePeriod": int(rule_time),
        "Rule_PredictiveVerb": int(rule_predictive),
        "Rule_FutureYear": int(rule_year),
        "Rule_Uncertain": int(rule_uncertain),
        "GloddHristova_FLS": fls_prediction
    })


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

def create_output_directory(args):

    current_date = datetime.now().strftime("%Y-%m-%d")

    dataset_filename = os.path.basename(args.dataset)

    dataset_base = os.path.splitext(dataset_filename)[0]

    experiment_name = f"{dataset_base}_{current_date}"

    seed_dir = os.path.join(
        args.save_path,
        experiment_name,
        f"seed{args.seed}",
        "in_domain",
        "glodd_hristova_step0"
    )

    os.makedirs(
        seed_dir,
        exist_ok=True
    )

    print(f"\n✓ Output Directory:\n{seed_dir}")

    return seed_dir


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset(base_data_path, dataset_path):

    print("\n" + "=" * 40)
    print("LOAD DATASET")
    print("=" * 40)

    if os.path.isabs(dataset_path):
        data_path = dataset_path
    else:
        data_path = os.path.join(
            base_data_path,
            dataset_path
        )

    print(f"Dataset: {data_path}")

    df = DataProcessing.load_from_file(
        data_path,
        "csv",
        sep=","
    )

    print(f"Shape: {df.shape}")

    return df


# =============================================================================
# LOG
# =============================================================================

def save_experiment_log(
    output_dir,
    args,
    summary_df
):

    log_path = os.path.join(
        output_dir,
        "experiment_log.txt"
    )

    with open(log_path, "w") as f:

        f.write("=" * 40 + "\n")
        f.write("GLODD & HRISTOVA STEP 0\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Seed: {args.seed}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(
            f"Publication Year: {args.publication_year}\n\n"
        )

        f.write(
            summary_df.to_string(index=False)
        )

    print(f"✓ Saved log: {log_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Glodd & Hristova Step 0 FLS Extraction"
    )

    base_data_path = DataProcessing.load_base_data_path(
        script_dir
    )

    parser.add_argument(
        "--dataset",
        default=os.path.join(
            base_data_path,
            "combined_datasets/naacl_2026_submission/naacl_2026_submission.csv"
        )
    )

    parser.add_argument(
        "--text_column",
        default="Base Sentence"
    )

    parser.add_argument(
        "--label_column",
        default="Ground Truth"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=300
    )

    parser.add_argument(
        "--save_path",
        default=os.path.join(
            base_data_path,
            "classification_results"
        )
    )

    parser.add_argument(
        "--publication_year",
        type=int,
        default=2023
    )

    args = parser.parse_args()

    output_dir = create_output_directory(args)

    df = load_dataset(
        base_data_path,
        args.dataset
    )

    print("\nApplying Step 0 rules...")

    rule_results = df[
        args.text_column
    ].astype(str).apply(
        lambda sentence:
        identify_fls(
            sentence,
            args.publication_year
        )
    )

    df = pd.concat(
        [df, rule_results],
        axis=1
    )

    summary_df = pd.DataFrame({
        "Metric": [
            "Potential FLS",
            "Not Potential FLS",
            "Time Rules",
            "Predictive Verb Rules",
            "Future Year Rules",
            "Uncertain Rules",
        ],
        "Count": [
            df["GloddHristova_FLS"].sum(),
            len(df) - df["GloddHristova_FLS"].sum(),
            df["Rule_TimePeriod"].sum(),
            df["Rule_PredictiveVerb"].sum(),
            df["Rule_FutureYear"].sum(),
            df["Rule_Uncertain"].sum()
        ]
    })

    print("\nSummary:")
    print(summary_df)

    DataProcessing.save_to_file(
        df,
        path=output_dir,
        prefix="potential_fls",
        save_file_type="csv",
        include_version=False
    )

    DataProcessing.save_to_file(
        summary_df,
        path=output_dir,
        prefix="rule_summary",
        save_file_type="csv",
        include_version=False
    )

    save_experiment_log(
        output_dir,
        args,
        summary_df
    )

    print("\n✓ Step 0 Complete")
    print("✓ No FinBERT Sentiment Analysis Performed")
    print("✓ No Stock Prediction Performed")