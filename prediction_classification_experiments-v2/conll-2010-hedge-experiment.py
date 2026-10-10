# =============================================================================
# CoNLL-2010 Shared Task Rule-Based Baseline (Farkas et al., 2010)
# Hedge / Weasel Cue Detection
#
# Binary Classification
# ---------------------
# 1 = Uncertain (contains >= 1 hedge/weasel cue)
# 0 = Certain (contains no hedge/weasel cue)
#
# Additionally records:
# - Which hedge category fired
# - Which cue(s) matched
# - Distribution of hedge categories
# - Distribution within TOLSA-M
# - Distribution within NON-TOLSA-M
#
# PURPOSE:
# Evaluate whether uncertainty / hedge detection aligns
# with TOLSA-M predictive statement identification.
# =============================================================================

import os
import re
import sys
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# =============================================================================
# PROJECT IMPORTS
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

if script_dir not in sys.path:
    sys.path.append(script_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)

from metrics import EvaluationMetric
from data_processing import DataProcessing

# =============================================================================
# HEDGE CATEGORIES (Based directly on CoNLL-2010 Guidelines)
# =============================================================================

AUXILIARIES = [
    "may", "might", "can", "would", "should", "could"
]

VERBS = [
    "suggest", "suggests", 
    "question", "questions",
    "presume", "presumes", 
    "suspect", "suspects",
    "indicate", "indicates", 
    "suppose", "supposes",
    "seem", "seems", 
    "appear", "appears", 
    "favor", "favors"
]

UNCERTAINTY = [
    "probable", "likely", "possible", "unsure", "often", 
    "possibly", "allegedly", "apparently", "perhaps"
]

CONJUNCTIONS = [
    "or", "and/or", "either or"
]

GENERALIZATION = [
    "widely", "traditionally", "generally", "broadly-accepted", "widespread"
]

QUALIFIERS = [
    "global", "superior", "excellent", "immensely", "legendary", 
    "best", "largest", "one of the largest", "most prominent"
]

OBVIOUSNESS = [
    "clearly", "obviously", "arguably"
]

COMPLEX_PHRASES = [
    "raises the question of",
    "it is claimed that", "it has been mentioned", "it is known",
    "there is evidence", "there is concern", "there is indication"
]

QUANTIFIERS = [
    "certain", "numerous", "many", "most", "some", "much", 
    "everyone", "few", "various", "one group of", 
    "experts say", "some people think", "more than 60% percent"
]

NOUNS = [
    "speculation", "proposal", "consideration", 
    "rumour has it that", "common sense insists that"
]

# =============================================================================
# RULE FUNCTIONS
# =============================================================================

def find_matches(text, terms):
    text = str(text).lower()
    matches = []
    for term in terms:
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, text):
            matches.append(term)
    return matches

def identify_hedge(sentence):
    """
    Identifies whether a sentence contains hedging language by matching
    against multiple lexical categories.

    Returns a dict with an 'uncertain' boolean and matched terms per category.

    Examples
    --------
    Modal verb hit:
        "This may indicate a systemic infection."
        -> AUXILIARIES: ["may"], uncertain=True

    Verb hit:
        "We suggest that further testing is needed."
        -> VERBS: ["suggest"], uncertain=True

    Uncertainty hit:
        "The cause of the outbreak remains unclear."
        -> UNCERTAINTY: ["unclear"], uncertain=True

    Generalization + verb hit (multiple categories):
        "In general, patients tend to recover within two weeks."
        -> GENERALIZATION: ["in general"], VERBS: ["tend"], uncertain=True

    No hedge — clean factual sentence:
        "The patient was discharged on March 3rd."
        -> all categories: [], uncertain=False

    Complex phrase hit:
        "It is thought to be caused by a viral pathogen."
        -> COMPLEX_PHRASES: ["it is thought to be"], uncertain=True

    Four categories fire simultaneously:
        "Some researchers believe this could potentially explain the findings."
        -> QUANTIFIERS: ["some"], VERBS: ["believe"],
           AUXILIARIES: ["could"], QUALIFIERS: ["potentially"], uncertain=True
    """
    modal_matches = find_matches(sentence, AUXILIARIES)
    verb_matches = find_matches(sentence, VERBS)
    uncertainty_matches = find_matches(sentence, UNCERTAINTY)
    generalization_matches = find_matches(sentence, GENERALIZATION)
    qualifier_matches = find_matches(sentence, QUALIFIERS)
    quantifier_matches = find_matches(sentence, QUANTIFIERS)
    noun_matches = find_matches(sentence, NOUNS)
    conjunction_matches = find_matches(sentence, CONJUNCTIONS)
    obviousness_matches = find_matches(sentence, OBVIOUSNESS)
    complex_matches = find_matches(sentence, COMPLEX_PHRASES)

    uncertain = any([
        modal_matches, verb_matches, uncertainty_matches,
        generalization_matches, qualifier_matches, quantifier_matches,
        noun_matches, conjunction_matches, obviousness_matches, complex_matches
    ])

    return pd.Series({
        "Rule_ModalVerb": int(len(modal_matches) > 0),
        "Rule_Verb": int(len(verb_matches) > 0),
        "Rule_Uncertainty": int(len(uncertainty_matches) > 0),
        "Rule_Generalization": int(len(generalization_matches) > 0),
        "Rule_Qualifier": int(len(qualifier_matches) > 0),
        "Rule_Quantifier": int(len(quantifier_matches) > 0),
        "Rule_Noun": int(len(noun_matches) > 0),
        "Rule_Conjunction": int(len(conjunction_matches) > 0),
        "Rule_Obviousness": int(len(obviousness_matches) > 0),
        "Rule_ComplexPhrase": int(len(complex_matches) > 0),
        "Matched_ModalVerb": "; ".join(modal_matches),
        "Matched_Verb": "; ".join(verb_matches),
        "Matched_Uncertainty": "; ".join(uncertainty_matches),
        "Matched_Generalization": "; ".join(generalization_matches),
        "Matched_Qualifier": "; ".join(qualifier_matches),
        "Matched_Quantifier": "; ".join(quantifier_matches),
        "Matched_Noun": "; ".join(noun_matches),
        "Matched_Conjunction": "; ".join(conjunction_matches),
        "Matched_Obviousness": "; ".join(obviousness_matches),
        "Matched_ComplexPhrase": "; ".join(complex_matches),
        "CoNLL_Uncertain": int(uncertain)
    })

# =============================================================================
# DATA SPLITS
# =============================================================================

def split_dataset(df, label_id_column, test_size, val_size, seed):
    """Create stratified train, validation, and optional test splits."""
    print("\n" + "=" * 40)
    print("CREATE DATA SPLITS")
    print("=" * 40)

    if not 0 <= test_size < 1:
        raise ValueError("test_size must be at least 0 and less than 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be greater than 0 and less than 1.")

    if test_size > 0:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[label_id_column],
        )
        adjusted_val_size = val_size / (1 - test_size)
    else:
        train_val_df = df.copy()
        test_df = None
        adjusted_val_size = val_size

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val_df[label_id_column],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if test_df is not None:
        test_df = test_df.reset_index(drop=True)

    print(f"Train size:      {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size:       {len(test_df) if test_df is not None else 0}")

    return train_df, val_df, test_df

def save_data_splits(train_df, val_df, test_df, output_dir):
    """Save the exact train, validation, and test splits used."""
    split_dir = os.path.join(output_dir, "data_splits")
    os.makedirs(split_dir, exist_ok=True)

    DataProcessing.save_to_file(train_df, path=split_dir, prefix="x_y_train_set", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(val_df, path=split_dir, prefix="x_y_val_set", save_file_type="csv", include_version=False)
    
    if test_df is not None:
        DataProcessing.save_to_file(test_df, path=split_dir, prefix="x_y_test_set", save_file_type="csv", include_version=False)
    
    print(f"✓ Saved data splits to: {split_dir}")

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

def create_output_directory(args):
    current_date = datetime.now().strftime("%Y-%m-%d")
    dataset_base = os.path.splitext(os.path.basename(args.dataset))[0]
    experiment_name = f"{dataset_base}_{current_date}"
    output_dir = os.path.join(args.save_path, experiment_name, f"seed{args.seed}", "in_domain", "conll_2010_hedge_rule")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output Directory:\n{output_dir}")
    return output_dir

# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset(base_data_path, dataset_path):
    print("\n" + "=" * 40)
    print("LOAD DATASET")
    print("=" * 40)
    data_path = dataset_path if os.path.isabs(dataset_path) else os.path.join(base_data_path, dataset_path)
    print(f"Dataset: {data_path}")
    df = DataProcessing.load_from_file(data_path, "csv", sep=",")
    print(f"Shape: {df.shape}")
    return df

# =============================================================================
# DISTRIBUTIONS
# =============================================================================

def create_distribution(df):
    return pd.DataFrame({
        "Category": [
            "Modal Verb", "Verb", "Uncertainty", "Generalization", "Qualifier",
            "Quantifier", "Noun", "Conjunction", "Obviousness", "Complex Phrase"
        ],
        "Count": [
            df["Rule_ModalVerb"].sum(), df["Rule_Verb"].sum(), df["Rule_Uncertainty"].sum(),
            df["Rule_Generalization"].sum(), df["Rule_Qualifier"].sum(), df["Rule_Quantifier"].sum(),
            df["Rule_Noun"].sum(), df["Rule_Conjunction"].sum(), df["Rule_Obviousness"].sum(),
            df["Rule_ComplexPhrase"].sum()
        ]
    })

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(df, label_name, seed, split_name):
    print("\n" + "=" * 40)
    print(f"EVALUATION RESULTS ({split_name.upper()} SET)")
    print("=" * 40)

    y_true = df[label_name].astype(int).values
    y_pred = df["CoNLL_Uncertain"].astype(int).values

    eval_report = EvaluationMetric.eval_classification_report(y_true, y_pred)
    confusion_mat, tn, fp, fn, tp = EvaluationMetric.get_confusion_matrix(y_true, y_pred, by_category=True)
    roc_auc_score = EvaluationMetric.get_roc_auc(y_true, y_pred)
    pr_auc_score = EvaluationMetric.get_pr_auc(y_true, y_pred)

    metrics_summary_df = pd.DataFrame([{
        "seed": seed,
        "model": "conll_2010_hedge_rule",
        "split": split_name,
        "train_accuracy": None,
        "val_accuracy": None if split_name == "test" else eval_report.get("accuracy", None),
        "test_accuracy": None if split_name == "val" else eval_report.get("accuracy", None),
        "precision_class_0": eval_report.get("0", {}).get("precision", None),
        "precision_class_1": eval_report.get("1", {}).get("precision", None),
        "recall_class_0": eval_report.get("0", {}).get("recall", None),
        "recall_class_1": eval_report.get("1", {}).get("recall", None),
        "f1_class_0": eval_report.get("0", {}).get("f1-score", None),
        "f1_class_1": eval_report.get("1", {}).get("f1-score", None),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "roc_auc": roc_auc_score,
        "pr_auc": pr_auc_score
    }])

    return metrics_summary_df, confusion_mat

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    """
    python conll-2010-hedge-experiment.py \
    --dataset "../data/combined_datasets/july_2026_results/july_2026_results.csv" \
    --text_column "Base Sentence" \
    --label_column "Ground Truth" \
    --seed 300 \
    --val_size 0.2 \
    --test_size 0.2 \
    --save_path "../data/classification_results/naacl_2026_submission"
    
    """

    parser = argparse.ArgumentParser(description="CoNLL-2010 Hedge and Weasel Baseline")
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    parser.add_argument("--dataset", default=os.path.join(base_data_path, "combined_datasets/july_2026_results/july_2026_results.csv"))
    parser.add_argument("--text_column", default="Base Sentence")
    parser.add_argument("--label_column", default="Ground Truth")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--save_path", default=os.path.join(base_data_path, "classification_results/naacl_2026_submission"))

    args = parser.parse_args()
    output_dir = create_output_directory(args)
    df = load_dataset(base_data_path, args.dataset)

    print("\nApplying Hedge Rules...")
    rule_results = df[args.text_column].astype(str).apply(identify_hedge)
    df = pd.concat([df, rule_results], axis=1)

    # ============================================================
    # DATA SPLITS: CREATE + SAVE
    # ============================================================
    train_df, val_df, test_df = split_dataset(
        df=df,
        label_id_column=args.label_column,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    save_data_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=output_dir,
    )

    # =========================================================
    # DISTRIBUTIONS (Calculated on Full Dataset)
    # =========================================================
    all_distribution_df = create_distribution(df)
    tolsam_distribution_df = create_distribution(df[df[args.label_column] == 1])
    non_tolsam_distribution_df = create_distribution(df[df[args.label_column] == 0])

    # =========================================================
    # METRICS (Evaluated on Val and Test Splits)
    # =========================================================
    val_metrics_df, val_conf_mat = evaluate_model(
        df=val_df,
        label_name=args.label_column,
        seed=args.seed,
        split_name="val"
    )

    test_metrics_df, test_conf_mat = evaluate_model(
        df=test_df,
        label_name=args.label_column,
        seed=args.seed,
        split_name="test"
    )
    
    # Combine metrics into one summary file
    metrics_summary_df = pd.concat([val_metrics_df, test_metrics_df], ignore_index=True)

    val_confusion_matrix_df = pd.DataFrame(
        val_conf_mat,
        index=["Actual_NON_TOLSA-M", "Actual_TOLSA-M"],
        columns=["Predicted_Certain", "Predicted_Uncertain"]
    )
    
    test_confusion_matrix_df = pd.DataFrame(
        test_conf_mat,
        index=["Actual_NON_TOLSA-M", "Actual_TOLSA-M"],
        columns=["Predicted_Certain", "Predicted_Uncertain"]
    )

    summary_df = pd.DataFrame({
        "Metric": [
            "Uncertain", "Certain", "Modal Verb", "Verb", "Uncertainty",
            "Generalization", "Qualifier", "Quantifier", "Noun",
            "Conjunction", "Obviousness", "Complex Phrase"
        ],
        "Count": [
            df["CoNLL_Uncertain"].sum(),
            len(df) - df["CoNLL_Uncertain"].sum(),
            df["Rule_ModalVerb"].sum(), df["Rule_Verb"].sum(), df["Rule_Uncertainty"].sum(),
            df["Rule_Generalization"].sum(), df["Rule_Qualifier"].sum(), df["Rule_Quantifier"].sum(),
            df["Rule_Noun"].sum(), df["Rule_Conjunction"].sum(), df["Rule_Obviousness"].sum(),
            df["Rule_ComplexPhrase"].sum()
        ]
    })

    # =========================================================
    # SAVE
    # =========================================================
    DataProcessing.save_to_file(df, path=output_dir, prefix="sentence_level_hedges", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(metrics_summary_df, path=output_dir, prefix="metrics_summary_conll", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(val_confusion_matrix_df, path=output_dir, prefix="confusion_matrix_val", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(test_confusion_matrix_df, path=output_dir, prefix="confusion_matrix_test", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(summary_df, path=output_dir, prefix="rule_summary", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(all_distribution_df, path=output_dir, prefix="distribution_all", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(tolsam_distribution_df, path=output_dir, prefix="distribution_tolsam", save_file_type="csv", include_version=False)
    DataProcessing.save_to_file(non_tolsam_distribution_df, path=output_dir, prefix="distribution_non_tolsam", save_file_type="csv", include_version=False)

    print("\n✓ Hedge Baseline Complete")
    print("✓ Sentence-level Certain vs Uncertain Classification")
    print("✓ Category-level Hedge Tracking")
    print("✓ TOLSA-M Category Distributions Saved")