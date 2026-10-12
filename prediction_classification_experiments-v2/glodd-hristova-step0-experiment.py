# =============================================================================
# Glodd & Hristova (2023)
# Step 0: Rule-Based Forward-Looking Statement Candidate Baseline
#
# This implementation stops at the rule-based identification of potential FLS.
#
# Pipeline:
#   1. Load the pre-split validation and test files (no splitting happens here)
#   2. Load the spaCy pipeline once
#   3. Apply the Glodd & Hristova rule-based patterns to validation
#   4. Identify and save potential FLS from validation
#   5. Apply the Glodd & Hristova rule-based patterns to test
#   6. Identify and save potential FLS from test
#   7. Evaluate validation predictions against Ground Truth
#   8. Evaluate test predictions against Ground Truth
#   9. Save distributions (all, TOLSA-M, non-TOLSA-M) on combined val+test
#  10. Save metrics and stop
#
# This script does NOT reproduce the paper's later classifier stages:
#   - manual FLS/non-FLS labeling
#   - classifier training
#   - classifier validation
#   - applying trained classifiers to potential FLS
#   - separate classifier-specific FLS datasets
#
# The --embedding_model argument names the spaCy pipeline used for the
# linguistic rule application. It is NOT a classifier-selection mechanism.
# Run one spaCy model per invocation, consistent with the ML/BERT pipeline.
# The same loaded spaCy model is applied to the val and test files separately.
# =============================================================================

import argparse
import gc
import json
import os
import sys
from datetime import datetime

import pandas as pd
import spacy
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

if script_dir not in sys.path:
    sys.path.append(script_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)

from data_processing import DataProcessing


# =============================================================================
# RULE CONFIGURATION
# =============================================================================

SPACY_MODELS = {
    "spacy_small": "en_core_web_sm",
    "spacy_medium": "en_core_web_md",
    "spacy_large": "en_core_web_lg",
    "spacy_transformer": "en_core_web_trf",
}

TIME_WORDS = {
    "next",
    "subsequent",
    "following",
    "upcoming",
    "incoming",
    "coming",
    "succeeding",
    "carryforward",
}

TIME_PERIODS = {
    "month",
    "months",
    "quarter",
    "quarters",
    "year",
    "years",
    "fiscal",
    "taxable",
    "period",
    "periods",
}

PREDICTIVE_LEMMAS = {
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
    "target",
}

UNCERTAIN_TERMS = {
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
}

CANDIDATE_COLUMN = "RuleBased_FLS_Candidate"

RULE_COLUMNS = {
    "time_period": "Rule_TimePeriod",
    "predictive_keyword_any_pos": "Rule_PredictiveKeyword_AnyPOS",
    "predictive_verb": "Rule_PredictiveVerb",
    "candidate_union": CANDIDATE_COLUMN,
}


# =============================================================================
# PATH / DATA HELPERS
# =============================================================================

def resolve_path(base_data_path, file_path):
    """Return an absolute path for an absolute or data-relative input path."""
    if os.path.isabs(file_path):
        return file_path

    return os.path.join(base_data_path, file_path)


def load_dataset(
    base_data_path,
    dataset_path,
    text_column,
    label_column,
    split_name,
):
    """Load and clean one pre-split file (validation or test)."""
    print("\n" + "=" * 40)
    print(f"LOAD {split_name.upper()} SET")
    print("=" * 40)

    data_path = resolve_path(base_data_path, dataset_path)
    print(f"Dataset path: {data_path}")

    df = DataProcessing.load_from_file(
        data_path,
        "csv",
        sep=",",
    )

    if df.empty:
        raise ValueError(f"The {split_name} dataset is empty.")

    required_columns = [text_column, label_column]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the {split_name} set: "
            f"{missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    original_count = len(df)

    df = df.dropna(subset=[text_column]).copy()

    df[text_column] = (
        df[text_column]
        .astype(str)
        .str.strip()
    )

    df = df[df[text_column] != ""].copy()

    labels = pd.to_numeric(
        df[label_column].astype(str).str.strip(),
        errors="coerce",
    )

    invalid_labels = ~labels.isin([0, 1])

    if invalid_labels.any():
        examples = (
            df.loc[invalid_labels, label_column]
            .head(5)
            .tolist()
        )

        raise ValueError(
            f"Every row in the {split_name} set needs a binary label "
            f"(0 or 1). Invalid examples: {examples}"
        )

    df[label_column] = labels.astype(int)

    print(f"Original rows: {original_count}")
    print(f"Usable rows:   {len(df)}")
    print(f"Shape:         {df.shape}")

    print("\nClass distribution:")
    print(df[label_column].value_counts())

    print(
        f"\nPreview:\n"
        f"{df[[text_column, label_column]].head(5)}\n"
    )

    return df.reset_index(drop=True)


def save_csv(df, output_dir, prefix):
    """Save a CSV using the project's existing DataProcessing utility."""
    DataProcessing.save_to_file(
        df,
        path=output_dir,
        prefix=prefix,
        save_file_type="csv",
        include_version=False,
    )


# =============================================================================
# RULE MATCHING
# =============================================================================

def matched_time_pairs(doc):
    """Match a time word followed by a time-period indicator."""
    words = [
        token.lower_
        for token in doc
        if token.is_alpha
    ]

    matches = set()

    for index, word in enumerate(words):
        if word not in TIME_WORDS:
            continue

        for following in words[index + 1:index + 3]:
            if following in TIME_PERIODS:
                matches.add(f"{word} ... {following}")

    return sorted(matches)


def matched_predictive_keywords(doc):
    """Find predictive vocabulary regardless of part of speech."""
    return [
        {
            "text": token.text,
            "lemma": token.lemma_.lower(),
            "pos": token.pos_,
            "tag": token.tag_,
        }
        for token in doc
        if token.lemma_.lower() in PREDICTIVE_LEMMAS
    ]


def matched_predictive_verbs(doc):
    """Find predictive words when spaCy identifies their use as a VERB."""
    matches = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        lemma = token.lemma_.lower()

        if lemma not in PREDICTIVE_LEMMAS:
            continue

        tense_values = token.morph.get("Tense")

        matches.append(
            {
                "text": token.text,
                "lemma": lemma,
                "tag": token.tag_,
                "pos": token.pos_,
                "tense": (
                    "|".join(tense_values)
                    if tense_values
                    else "NotAnnotated"
                ),
            }
        )

    return matches


def matched_uncertain_terms(doc):
    """Record uncertainty terms as diagnostics."""
    return sorted(
        {
            token.lower_
            for token in doc
            if token.lower_ in UNCERTAIN_TERMS
        }
    )


def extract_rule_record(doc):
    """
    Apply the rule-based signals to one spaCy Doc.

    Potential FLS = time-period rule OR predictive-verb rule.
    """
    time_pairs = matched_time_pairs(doc)
    predictive_keywords = matched_predictive_keywords(doc)
    predictive_verbs = matched_predictive_verbs(doc)
    uncertain_terms = matched_uncertain_terms(doc)

    excluded_nouns = [
        match
        for match in predictive_keywords
        if match["pos"] in {"NOUN", "PROPN"}
    ]

    time_hit = bool(time_pairs)
    verb_hit = bool(predictive_verbs)

    return {
        "Rule_TimePeriod": int(time_hit),

        "Rule_PredictiveKeyword_AnyPOS": int(
            bool(predictive_keywords)
        ),

        "Rule_PredictiveVerb": int(verb_hit),

        "Rule_UncertainTermPresent": int(
            bool(uncertain_terms)
        ),

        "Matched_TimePairs": "; ".join(time_pairs),

        "Matched_PredictiveKeywords_AnyPOS": "; ".join(
            (
                f"{match['text']}"
                f"[lemma={match['lemma']},"
                f"pos={match['pos']}]"
            )
            for match in predictive_keywords
        ),

        "Matched_PredictiveVerbs": "; ".join(
            (
                f"{match['text']}"
                f"[lemma={match['lemma']},"
                f"pos={match['pos']},"
                f"tag={match['tag']},"
                f"tense={match['tense']}]"
            )
            for match in predictive_verbs
        ),

        "Matched_PredictiveNouns_Excluded": "; ".join(
            (
                f"{match['text']}"
                f"[pos={match['pos']}]"
            )
            for match in excluded_nouns
        ),

        "Matched_UncertainTerms": "; ".join(uncertain_terms),

        CANDIDATE_COLUMN: int(time_hit or verb_hit),
    }


# =============================================================================
# SPACY: LOAD + APPLY RULES
# =============================================================================

def load_spacy_pipeline(model_key):
    """Load the requested spaCy pipeline with POS information available."""
    package_name = SPACY_MODELS[model_key]

    print(f"\nLoading {model_key}: {package_name}")

    try:
        nlp = spacy.load(package_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {package_name}. "
            "Install that spaCy model and its dependencies."
        ) from exc

    if not any(
        component in nlp.pipe_names
        for component in ("tagger", "morphologizer")
    ):
        raise RuntimeError(
            f"{package_name} has no POS tagging component. "
            "A POS-aware spaCy pipeline is required for the "
            "predictive-verb rule."
        )

    for component in (
        "ner",
        "entity_ruler",
        "textcat",
        "textcat_multilabel",
    ):
        if component in nlp.pipe_names:
            nlp.disable_pipe(component)

    return nlp


def apply_glodd_rules(df, model_key, nlp, text_column, split_name):
    """Apply the Glodd rule-based extraction to the supplied split only."""
    print("\n" + "=" * 40)
    print(f"APPLY GLODD & HRISTOVA RULES ({split_name.upper()} SET)")
    print("=" * 40)
    print(f"spaCy model: {model_key}")
    print(f"Sentences:   {len(df)}")

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    records = [
        extract_rule_record(doc)
        for doc in nlp.pipe(texts)
    ]

    if len(records) != len(df):
        raise RuntimeError(
            "spaCy output did not align with input sentences."
        )

    rule_df = pd.DataFrame(
        records,
        index=df.index,
    )

    conflicting_columns = [
        column
        for column in rule_df.columns
        if column in df.columns
    ]

    if conflicting_columns:
        raise ValueError(
            f"The {split_name} input already contains rule-output columns: "
            f"{conflicting_columns}"
        )

    annotated_df = pd.concat(
        [df.copy(), rule_df],
        axis=1,
    )

    annotated_df["Spacy_Model"] = model_key

    return annotated_df


# =============================================================================
# DISTRIBUTIONS
# =============================================================================

def create_distribution(df):
    """
    Count rule signal hits across the supplied rows.

    Mirrors the CoNLL distribution function. Applied to the combined
    val + test annotated rows only.
    """
    return pd.DataFrame({
        "Category": [
            "FLS Candidate",
            "Not FLS Candidate",
            "Time Period",
            "Predictive Keyword (Any POS)",
            "Predictive Verb",
            "Uncertain Term Present",
        ],
        "Count": [
            int(df[CANDIDATE_COLUMN].sum()),
            int(len(df) - df[CANDIDATE_COLUMN].sum()),
            int(df["Rule_TimePeriod"].sum()),
            int(df["Rule_PredictiveKeyword_AnyPOS"].sum()),
            int(df["Rule_PredictiveVerb"].sum()),
            int(df["Rule_UncertainTermPresent"].sum()),
        ],
    })


# =============================================================================
# OPTIONAL BENCHMARK METRICS
# =============================================================================

def evaluate_candidates(df, label_column):
    """Compare Glodd's rule-based candidates against human ground truth."""
    actual = df[label_column].astype(int).to_numpy()
    predicted = df[CANDIDATE_COLUMN].astype(int).to_numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1],
        zero_division=0,
    )

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    return {
        "precision_class_0": float(precision[0]),
        "precision_class_1": float(precision[1]),
        "recall_class_0": float(recall[0]),
        "recall_class_1": float(recall[1]),
        "f1_class_0": float(f1[0]),
        "f1_class_1": float(f1[1]),
        "accuracy": float(accuracy_score(actual, predicted)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def save_metrics_summary(
    val_df,
    test_df,
    model_key,
    label_column,
    seed,
    output_dir,
):
    """
    Evaluate only the validation and test splits against Ground Truth.

    A single row is saved with separate validation and test metric columns.
    """
    if label_column not in val_df.columns or test_df is None:
        return None

    val_metrics = evaluate_candidates(val_df, label_column)
    test_metrics = evaluate_candidates(test_df, label_column)

    metrics_row = {
        "seed": seed,
        "model": model_key,
        "train_accuracy": None,
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "val_precision_class_0": val_metrics["precision_class_0"],
        "val_precision_class_1": val_metrics["precision_class_1"],
        "val_recall_class_0": val_metrics["recall_class_0"],
        "val_recall_class_1": val_metrics["recall_class_1"],
        "val_f1_class_0": val_metrics["f1_class_0"],
        "val_f1_class_1": val_metrics["f1_class_1"],
        "val_tn": val_metrics["tn"],
        "val_fp": val_metrics["fp"],
        "val_fn": val_metrics["fn"],
        "val_tp": val_metrics["tp"],
        "test_precision_class_0": test_metrics["precision_class_0"],
        "test_precision_class_1": test_metrics["precision_class_1"],
        "test_recall_class_0": test_metrics["recall_class_0"],
        "test_recall_class_1": test_metrics["recall_class_1"],
        "test_f1_class_0": test_metrics["f1_class_0"],
        "test_f1_class_1": test_metrics["f1_class_1"],
        "test_tn": test_metrics["tn"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "test_tp": test_metrics["tp"],
    }

    metrics_df = pd.DataFrame([metrics_row])

    save_csv(metrics_df, output_dir, "metrics_summary_glodd_step0")

    print("\n" + "=" * 40)
    print("VALIDATION BENCHMARK")
    print("=" * 40)
    print(pd.DataFrame([val_metrics]).to_string(index=False))

    print("\n" + "=" * 40)
    print("TEST BENCHMARK")
    print("=" * 40)
    print(pd.DataFrame([test_metrics]).to_string(index=False))

    return metrics_df


# =============================================================================
# EXPERIMENT METADATA
# =============================================================================

def save_experiment_metadata(
    args,
    output_dir,
    val_rows,
    test_rows,
    val_candidate_rows,
    test_candidate_rows,
):
    """Save a compact record of the rule-based run."""
    metadata = {
        "val_dataset": args.val_dataset,
        "test_dataset": args.test_dataset,
        "experiment": args.experiment,
        "seed": args.seed,
        "validation_rows": int(val_rows),
        "test_rows": int(test_rows),
        "validation_potential_fls_candidates": int(val_candidate_rows),
        "test_potential_fls_candidates": int(test_candidate_rows),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "spacy_model": args.embedding_model,
        "spacy_package": SPACY_MODELS[args.embedding_model],
        "rule_based_signals": [
            "time-word + time-period combination",
            "predictive word used as a verb",
        ],
        "candidate_definition": (
            "Rule_TimePeriod OR Rule_PredictiveVerb"
        ),
        "distribution_scope": (
            "Combined validation and test annotated rows only."
        ),
        "processing_scope": (
            "Pre-split validation and test files are supplied directly; "
            "no splitting is done in this script. The same spaCy pipeline "
            "is loaded once and applied separately to the validation and "
            "test files. Benchmark metrics compare each held-out split "
            "against the supplied Ground Truth labels. No manual labeling, "
            "classifier training, classifier selection, classifier "
            "application, or classifier-specific FLS datasets are produced."
        ),
        "completed_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    metadata_path = os.path.join(
        output_dir,
        "experiment_metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"✓ Saved metadata: {metadata_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Example:

    python glodd-hristova-step0-experiment.py \
        --val_dataset ../data/classification_results/naacl_2026_submission/tolsa_naacl_2026_2027-final/seed3/in_domain/splits/x_y_val_set.csv \
        --test_dataset ../data/classification_results/naacl_2026_submission/tolsa_naacl_2026_2027-final/seed3/in_domain/splits/x_y_test_set.csv \
        --experiment tolsa_naacl_2026_2027-final \
        --save_path ../data/classification_results/naacl_2026_submission \
        --text_column "Base Sentence" \
        --label_column "Ground Truth" \
        --embedding_model spacy_small \
        --seed 3
    """
    print("\n" + "=" * 40)
    print("GLODD & HRISTOVA STEP 0 CANDIDATE BASELINE")
    print("=" * 40)

    base_data_path = DataProcessing.load_base_data_path(
        script_dir
    )

    parser = argparse.ArgumentParser(
        description=(
            "Apply the Glodd & Hristova rule-based potential-FLS extraction "
            "separately to pre-split validation and test files."
        )
    )

    parser.add_argument(
        "--val_dataset",
        required=True,
        help="Path to the pre-split validation CSV",
    )

    parser.add_argument(
        "--test_dataset",
        required=True,
        help="Path to the pre-split test CSV",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment folder name under save_path "
             "(e.g. tolsa_naacl_2026_2027-final)",
    )

    parser.add_argument(
        "--text_column",
        default="Base Sentence",
    )

    parser.add_argument(
        "--label_column",
        default="Ground Truth",
    )

    parser.add_argument(
        "--embedding_model",
        required=True,
        choices=list(SPACY_MODELS.keys()),
    )

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument(
        "--save_path",
        default=os.path.join(
            base_data_path,
            "classification_results/naacl_2026_submission"
        )
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # EXPERIMENT SETUP
    # -------------------------------------------------------------------------

    output_dir = os.path.join(
        args.save_path,
        args.experiment,
        f"seed{args.seed}",
        "in_domain",
        "glodd_hristova_step0",
        args.embedding_model,
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment name:  {args.experiment}")
    print(f"spaCy model:      {args.embedding_model}")
    print(f"Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # LOAD PRE-SPLIT VALIDATION AND TEST FILES
    # -------------------------------------------------------------------------

    val_df = load_dataset(
        base_data_path=base_data_path,
        dataset_path=args.val_dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        split_name="val",
    )

    test_df = load_dataset(
        base_data_path=base_data_path,
        dataset_path=args.test_dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        split_name="test",
    )

    # -------------------------------------------------------------------------
    # LOAD SPACY (once; applied to val and test separately below)
    # -------------------------------------------------------------------------

    nlp = load_spacy_pipeline(args.embedding_model)

    try:
        # -------------------------------------------------------------
        # APPLY GLODD RULES: VALIDATION ONLY
        # -------------------------------------------------------------
        val_annotated_df = apply_glodd_rules(
            df=val_df,
            model_key=args.embedding_model,
            nlp=nlp,
            text_column=args.text_column,
            split_name="val",
        )

        val_potential_fls_df = val_annotated_df.loc[
            val_annotated_df[CANDIDATE_COLUMN].eq(1)
        ].copy()

        save_csv(val_annotated_df, output_dir, "rule_annotations_val")
        save_csv(val_potential_fls_df, output_dir, "fls_candidates_val")

        # -------------------------------------------------------------
        # APPLY GLODD RULES: TEST ONLY
        # -------------------------------------------------------------
        test_annotated_df = apply_glodd_rules(
            df=test_df,
            model_key=args.embedding_model,
            nlp=nlp,
            text_column=args.text_column,
            split_name="test",
        )

        test_potential_fls_df = test_annotated_df.loc[
            test_annotated_df[CANDIDATE_COLUMN].eq(1)
        ].copy()

        save_csv(test_annotated_df, output_dir, "rule_annotations_test")
        save_csv(test_potential_fls_df, output_dir, "fls_candidates_test")

    finally:
        del nlp
        gc.collect()

    # -------------------------------------------------------------------------
    # DISTRIBUTIONS (combined val + test annotated rows only)
    # -------------------------------------------------------------------------

    combined_annotated_df = pd.concat(
        [val_annotated_df, test_annotated_df],
        ignore_index=True,
    )

    all_distribution_df = create_distribution(combined_annotated_df)

    tolsam_distribution_df = create_distribution(
        combined_annotated_df.loc[
            combined_annotated_df[args.label_column].eq(1)
        ]
    )

    non_tolsam_distribution_df = create_distribution(
        combined_annotated_df.loc[
            combined_annotated_df[args.label_column].eq(0)
        ]
    )

    print("\n" + "=" * 40)
    print("DISTRIBUTIONS (VAL + TEST)")
    print("=" * 40)
    print("\nAll rows:")
    print(all_distribution_df.to_string(index=False))
    print("\nTOLSA-M rows:")
    print(tolsam_distribution_df.to_string(index=False))
    print("\nNon-TOLSA-M rows:")
    print(non_tolsam_distribution_df.to_string(index=False))

    save_csv(all_distribution_df, output_dir, "distribution_all")
    save_csv(tolsam_distribution_df, output_dir, "distribution_tolsam")
    save_csv(non_tolsam_distribution_df, output_dir, "distribution_non_tolsam")

    # -------------------------------------------------------------------------
    # EVALUATE: VALIDATION AND TEST AGAINST GROUND TRUTH
    # -------------------------------------------------------------------------

    save_metrics_summary(
        val_df=val_annotated_df,
        test_df=test_annotated_df,
        model_key=args.embedding_model,
        label_column=args.label_column,
        seed=args.seed,
        output_dir=output_dir,
    )

    save_experiment_metadata(
        args=args,
        output_dir=output_dir,
        val_rows=len(val_df),
        test_rows=len(test_df),
        val_candidate_rows=len(val_potential_fls_df),
        test_candidate_rows=len(test_potential_fls_df),
    )

    # -------------------------------------------------------------------------
    # STOP
    # -------------------------------------------------------------------------

    print("\n" + "=" * 40)
    print("GLODD & HRISTOVA STEP 0 COMPLETE")
    print("=" * 40)
    print("✓ Pre-split val and test files loaded (no splitting)")
    print("✓ Glodd rules applied to validation")
    print("✓ Glodd rules applied to test")
    print("✓ Distributions saved (all, TOLSA-M, non-TOLSA-M)")
    print("✓ Validation evaluated against Ground Truth")
    print("✓ Test evaluated against Ground Truth")
    print("✓ No classifier stages were run")


if __name__ == "__main__":
    main()