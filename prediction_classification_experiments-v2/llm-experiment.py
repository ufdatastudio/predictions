import os
import re
import sys
import json
import time
import pprint
import argparse
import matplotlib

matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Get the current working directory of the script.
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, "../"))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from data_visualizing import DataVisualizing
from prompts import SentenceClassificationPrompt
from text_generation_models import TextGenerationModelFactory


# Number of results buffered before writing to disk.
BATCH_SIZE = 50

# Process the complete dataset.
STOP_AFTER = None


def load_dataset(
    dataset_path,
    text_column="Base Sentence",
    label_column="Sentence Label",
):
    """Load a fixed dataset split without resampling."""
    print("\n" + "=" * 40)
    print("LOAD DATASET")
    print("=" * 40)
    print(f"Dataset path: {dataset_path}")

    df = DataProcessing.load_from_file(
        dataset_path, "csv", sep=",", encoding="utf-8"
    )

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head(7)}\n")
    print(f"\nPreview:\n{df.tail(7)}\n")
    print(f"Label distribution:\n{df[label_column].value_counts()}\n")

    return df


def build_model(model_name):
    """Initialize the requested LLM."""
    tgmf = TextGenerationModelFactory()

    print("\n" + "=" * 40)
    print("LLM TO LOAD")
    print("=" * 40)

    pprint.PrettyPrinter(indent=3).pprint(model_name)

    return tgmf.create_instance(model_name)



def load_prompts(
    prompt_type,
    few_shot_samples_path=None,
    seed=42,
):
    """Build the classification prompt using prepared few-shot samples."""
    print("\n" + "=" * 50)
    print(f"LOAD PROMPT: {prompt_type}")
    print("=" * 50)

    prompt = SentenceClassificationPrompt(prompt_type_name=prompt_type)
    tolsa_properties, linguistic_cues = prompt.properties_and_cues()

    if prompt_type == "zero-shot":
        system_identity, task, format_output = prompt.zero_shot()
        extra_text = ""

    elif prompt_type == "few-shot":
        if not few_shot_samples_path:
            raise ValueError(
                "--few_shot_samples is required for few-shot prompts."
            )

        if not os.path.isfile(few_shot_samples_path):
            raise FileNotFoundError(
                f"Few-shot samples not found: {few_shot_samples_path}"
            )

        samples_df = pd.read_csv(few_shot_samples_path)

        required_columns = {
            "Dataset Name",
            "Base Sentence",
            "Ground Truth",
        }
        missing_columns = required_columns - set(samples_df.columns)

        if missing_columns:
            raise ValueError(
                f"Few-shot CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if samples_df.empty:
            raise ValueError(
                f"Few-shot samples file is empty: "
                f"{few_shot_samples_path}"
            )

        if samples_df[
            ["Base Sentence", "Ground Truth"]
        ].isna().any().any():
            raise ValueError(
                "Few-shot samples contain missing sentences or labels."
            )

        if not samples_df["Ground Truth"].astype(str).isin(
            ["0", "1"]
        ).all():
            raise ValueError(
                "Few-shot Ground Truth values must be binary labels 0 or 1."
            )

        print(f"Few-shot samples: {len(samples_df)}")
        print(f"Samples file: {few_shot_samples_path}")
        print(
            "Examples per dataset:\n"
            f"{samples_df['Dataset Name'].value_counts(dropna=False)}"
        )

        # Pass the prepared file to the existing prompt formatter.
        # The prompt formatter must use all rows as provided, without
        # performing another sampling operation.
        system_identity, task, format_output, examples = prompt.few_shot(
            dataset_path=few_shot_samples_path,
            stratify_columns=None,
            seed=seed,
        )
        extra_text = f"Examples:\n{examples}"

    elif prompt_type == "chain-of-thought":
        system_identity, task, format_output, steps = (
            prompt.chain_of_thought()
        )
        extra_text = f"Steps:\n{steps}"

    else:
        raise ValueError(
            f"Unknown prompt_type: '{prompt_type}'. "
            "Choose from: zero-shot, few-shot, chain-of-thought"
        )

    base_prompt = f"""{system_identity}

{tolsa_properties}

{linguistic_cues}

{extra_text}

"""

    return base_prompt, task, format_output


def _llm_classifier(
    sentence_to_classify,
    base_prompt,
    model,
    task,
    format_output,
    is_first,
):
    """Classify one sentence and return its raw response and label."""
    prompt = f"""{base_prompt}

<text_document>{sentence_to_classify}</text_document>

{task}
{format_output}"""

    if is_first:
        print(f"\n\tPrompt: {prompt}\n")

    input_prompt = model.user(prompt)
    raw_response = model.safe_chat_completion([input_prompt])

    if raw_response is None:
        return None, None

    parsed = DataProcessing.parse_llm_json_response(raw_response)
    label = parsed.get("y_hat") if parsed else None

    # Accept only actual binary labels.
    try:
        numeric_label = float(label)
        if numeric_label not in (0.0, 1.0):
            numeric_label = None
    except (TypeError, ValueError):
        numeric_label = None

    return raw_response, numeric_label


def llm_classifer(
    model_name,
    model,
    test_df,
    base_prompt,
    sentence_label_task,
    sentence_label_format_output,
    save_directory,
    seed,
    prompt_type,
):
    """
    Run LLM inference with resumable checkpointing.

    Successful binary classifications are preserved. Rows with missing
    or invalid labels are retried on the next run.
    """
    print(f"Shape: {test_df.shape}")
    print(f"\nPreview:\n{test_df.head(7)}\n")
    print(f"\nPreview:\n{test_df.tail(7)}\n")

    os.makedirs(save_directory, exist_ok=True)

    safe_model_name = model_name.replace("/", "_")
    checkpoint_file = os.path.join(
        save_directory,
        f"checkpoint_{safe_model_name}_{prompt_type}.csv",
    )

    checkpoint_columns = [
        "seed",
        "original_index",
        "text",
        "raw_response",
        "llm_label",
        "llm_name",
    ]

    # --------------------------------------------------------
    # Resume from checkpoint
    # --------------------------------------------------------
    if os.path.exists(checkpoint_file):
        print(
            f"\n[!] Found checkpoint file. "
            f"Resuming from: {checkpoint_file}"
        )

        processed_df = pd.read_csv(checkpoint_file)

        required_columns = set(checkpoint_columns)
        missing_columns = required_columns - set(processed_df.columns)

        if missing_columns:
            raise ValueError(
                "Checkpoint is missing required columns: "
                f"{sorted(missing_columns)}. "
                "The checkpoint was not modified."
            )

        labels = pd.to_numeric(
            processed_df["llm_label"], errors="coerce"
        )

        indices = pd.to_numeric(
            processed_df["original_index"], errors="coerce"
        )

        # Keep rows with a valid binary label and valid original index.
        valid_mask = labels.isin([0, 1]) & indices.notna()

        valid_df = processed_df.loc[valid_mask].copy()
        valid_df["llm_label"] = labels.loc[valid_mask].astype(int)
        valid_df["original_index"] = (
            indices.loc[valid_mask].astype(int)
        )

        # Discard unintended columns from older checkpoint formats.
        valid_df = valid_df.reindex(columns=checkpoint_columns)

        # Deduplicate checkpoint entries by original index, keeping the
        # most recent successful result.
        valid_df = valid_df.drop_duplicates(
            subset=["original_index"], keep="last"
        )

        # Only reuse results whose original indices exist in this split.
        test_indices = set(test_df.index)
        valid_df = valid_df[
            valid_df["original_index"].isin(test_indices)
        ].copy()

        processed_indices = set(valid_df["original_index"].tolist())

        remaining_df = test_df[
            ~test_df.index.isin(processed_indices)
        ]

        results = valid_df.to_dict("records")

        # Rewrite the checkpoint with clean, consistent columns.
        # The backup should be retained until the run is verified.
        valid_df.to_csv(checkpoint_file, index=False)

        print(f"Successful checkpoint rows: {len(valid_df)}")
        print(f"Rows remaining: {len(remaining_df)}")

    else:
        remaining_df = test_df.copy()
        results = []

        # Initialize a checkpoint with the correct header.
        pd.DataFrame(columns=checkpoint_columns).to_csv(
            checkpoint_file, index=False
        )

    # --------------------------------------------------------
    # Optional testing limit
    # --------------------------------------------------------
    if STOP_AFTER is not None:
        remaining_df = remaining_df.head(STOP_AFTER)

    print(f"Rows to process in this run: {len(remaining_df)}\n")
    print(model_name, model)

    batch_buffer = []

    # --------------------------------------------------------
    # Inference and checkpoint writing
    # --------------------------------------------------------
    for loop_idx, (idx, row) in enumerate(
        tqdm(
            remaining_df.iterrows(),
            total=len(remaining_df),
            desc="Processing",
        )
    ):
        text = row["Base Sentence"]

        if loop_idx < 3:
            print(
                "Classify text document as either TOLSA (1) "
                "or non-TOLSA (0)"
            )
            print(f"    {idx} --- Sentence: {text}")

        raw_response, llm_label = _llm_classifier(
            text,
            base_prompt,
            model,
            sentence_label_task,
            sentence_label_format_output,
            loop_idx == 0,
        )

        result_dict = {
            "seed": seed,
            "original_index": idx,
            "text": text,
            "raw_response": raw_response,
            "llm_label": llm_label,
            "llm_name": model_name,
        }

        batch_buffer.append(result_dict)
        results.append(result_dict)

        if loop_idx < 3:
            print(f"\tLabel: {llm_label} via Model: {model_name}")

        # Append using exactly the same columns as the checkpoint header.
        if len(batch_buffer) >= BATCH_SIZE:
            pd.DataFrame(
                batch_buffer, columns=checkpoint_columns
            ).to_csv(
                checkpoint_file,
                mode="a",
                header=False,
                index=False,
            )

            print(
                f"Checkpoint saved: {len(batch_buffer)} rows flushed."
            )
            batch_buffer = []

    # Flush the final partial batch.
    if batch_buffer:
        pd.DataFrame(
            batch_buffer, columns=checkpoint_columns
        ).to_csv(
            checkpoint_file,
            mode="a",
            header=False,
            index=False,
        )

        print(
            f"Final checkpoint saved: {len(batch_buffer)} rows flushed."
        )

    # --------------------------------------------------------
    # Reconstruct results in original test-set order
    # --------------------------------------------------------
    results_df = pd.DataFrame(results, columns=checkpoint_columns)

    if not results_df.empty:
        results_df = results_df.drop_duplicates(
            subset=["original_index"], keep="last"
        )

        results_df = (
            results_df.set_index("original_index")
            .reindex(test_df.index)
            .rename_axis("original_index")
            .reset_index()
        )

    print(f"Shape: {results_df.shape}")
    print(f"\nPreview:\n{results_df.head(10)}\n")

    return results_df


def create_results_dataframe(X_test_df, y_hat_df, model_name, seed):
    """Combine the test data with model predictions."""
    print("\n" + "=" * 40)
    print("CREATE RESULTS DATAFRAME")
    print("=" * 40)

    results_df = X_test_df.copy()
    results_df["seed"] = seed

    results_df[model_name] = y_hat_df["llm_label"].to_list()

    print(f"Added predictions: {model_name}")
    print(f"\nFinal shape: {results_df.shape}")
    print(f"\nPreview:\n{results_df.head(3)}\n")

    return results_df


def evaluate_models(
    results_df,
    model_name,
    label_col_name,
    save_path,
    seed,
    prompt_type,
):
    """Evaluate predictions and save metrics and the confusion matrix."""
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Label: {label_col_name}\n")

    actual_labels = results_df[label_col_name].values
    predictions = results_df[model_name].values

    # Preserve the existing evaluation convention for failed predictions.
    predictions = np.nan_to_num(
        np.asarray(predictions, dtype=float), nan=0.0
    )

    X_test_list = results_df["Base Sentence"].to_list()
    y_test_df = results_df[[label_col_name]]

    eval_report = EvaluationMetric.eval_classification_report(
        actual_labels, predictions
    )

    confusion_mat, tn, fp, fn, tp = (
        EvaluationMetric.get_confusion_matrix(
            actual_labels, predictions, by_category=True
        )
    )

    print(f"Confusion Matrix:\n{confusion_mat}\n")

    DataVisualizing.confusion_matrix(
        f"{model_name}_{prompt_type}",
        confusion_mat,
        save_path,
        include_version=False,
    )

    print(
        f"Saved confusion matrix: "
        f"confusion_matrix_{model_name}_{prompt_type}.png\n"
    )

    roc_auc_score = EvaluationMetric.get_roc_auc(
        actual_labels, predictions
    )
    print(f"ROC-AUC Score: {roc_auc_score:.4f}\n")

    pr_auc_score = EvaluationMetric.get_pr_auc(
        actual_labels, predictions
    )
    print(f"PR-AUC Score: {pr_auc_score:.4f}\n")

    # Support both string and integer class keys.
    def get_metric(class_id, metric):
        class_report = eval_report.get(
            str(class_id), eval_report.get(class_id, {})
        )
        return class_report.get(metric)

    precision_0 = get_metric(0, "precision")
    precision_1 = get_metric(1, "precision")
    recall_0 = get_metric(0, "recall")
    recall_1 = get_metric(1, "recall")
    f1_0 = get_metric(0, "f1-score")
    f1_1 = get_metric(1, "f1-score")

    # Calculate metrics from the confusion matrix when absent.
    if any(
        value is None
        for value in (
            precision_0,
            precision_1,
            recall_0,
            recall_1,
            f1_0,
            f1_1,
        )
    ):
        precision_0 = tn / (tn + fn) if (tn + fn) else 0.0
        precision_1 = tp / (tp + fp) if (tp + fp) else 0.0
        recall_0 = tn / (tn + fp) if (tn + fp) else 0.0
        recall_1 = tp / (tp + fn) if (tp + fn) else 0.0

        f1_0 = (
            2 * precision_0 * recall_0 / (precision_0 + recall_0)
            if (precision_0 + recall_0)
            else 0.0
        )
        f1_1 = (
            2 * precision_1 * recall_1 / (precision_1 + recall_1)
            if (precision_1 + recall_1)
            else 0.0
        )

        print("Calculated missing metrics from the confusion matrix.")
        print(
            f"Precision - Class 0: {precision_0:.4f}, "
            f"Class 1: {precision_1:.4f}"
        )
        print(
            f"Recall - Class 0: {recall_0:.4f}, "
            f"Class 1: {recall_1:.4f}"
        )
        print(
            f"F1-Score - Class 0: {f1_0:.4f}, "
            f"Class 1: {f1_1:.4f}\n"
        )

    metrics_row = {
        "seed": seed,
        "model": model_name,
        "train_accuracy": None,
        "val_accuracy": None,
        "test_accuracy": eval_report.get("accuracy"),
        "precision_class_0": precision_0,
        "precision_class_1": precision_1,
        "recall_class_0": recall_0,
        "recall_class_1": recall_1,
        "f1_class_0": f1_0,
        "f1_class_1": f1_1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_auc": roc_auc_score,
        "pr_auc": pr_auc_score,
    }

    metrics_summary_df = pd.DataFrame([metrics_row])

    print("\n" + "=" * 40)
    print(f"METRICS SUMMARY WITH SEED {seed}")
    print("=" * 40)
    print(metrics_summary_df)
    print()

    DataProcessing.save_to_file(
        data=metrics_summary_df,
        path=save_path,
        prefix=f"metrics_summary_{model_name}_{prompt_type}",
        save_file_type="csv",
        include_version=False,
    )

    print(
        f"Saved metrics summary: "
        f"metrics_summary_{model_name}_{prompt_type}.csv\n"
    )

    return metrics_summary_df


if __name__ == "__main__":
    # --------------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------------
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    default_load_save_path = os.path.join(
        base_data_path,
        "classification_results/train_synthetic-v1_2026-03-23/"
        "seed3/external_fpb-maya-binary-imbalanced-96d-v1",
    )
    default_dataset_path = os.path.join(
        default_load_save_path, "x_y_test_set.csv"
    )

    parser = argparse.ArgumentParser(
        description=(
            "LLM sentence classification using a saved dataset split."
        )
    )

    parser.add_argument(
        "--test_dataset",
        default=default_dataset_path,
        help="Path to x_y_val_set.csv or x_y_test_set.csv.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="The model used for sentence classification.",
    )
    parser.add_argument(
        "--label_column",
        default="Ground Truth",
        help="Column containing the true labels.",
    )
    parser.add_argument(
        "--prompt_type",
        required=True,
        choices=["zero-shot", "few-shot", "chain-of-thought"],
        help="Prompt strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Experiment seed, e.g. 3, 7, or 33.",
    )
    parser.add_argument(
        "--few_shot_samples",
        default=None,
        help="Path to the prepared few-shot samples CSV.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 2. Experiment paths
    # --------------------------------------------------------
    splits_dir = os.path.dirname(os.path.abspath(args.test_dataset))
    in_domain_dir = os.path.dirname(splits_dir)

    split_filename = os.path.basename(args.test_dataset)

    split_name = (
        "val" if split_filename == "x_y_val_set.csv"
        else "test" if split_filename == "x_y_test_set.csv"
        else "other"
    )

    safe_model_name = args.model_name.replace("/", "_")

    save_directory = os.path.join(
        in_domain_dir,
        "llm",
        safe_model_name,
        args.prompt_type,
        split_name,
    )
    os.makedirs(save_directory, exist_ok=True)

    print(f"\nSave directory: {save_directory}")


    # --------------------------------------------------------
    # 3. Load data and prompts
    # --------------------------------------------------------
    test_df = load_dataset(
        args.test_dataset,
        label_column=args.label_column,
    )

    if args.prompt_type == "few-shot" and not args.few_shot_samples:
        parser.error(
            "--few_shot_samples is required when --prompt_type=few-shot."
        )

    model = build_model(args.model_name)

    base_prompt, sentence_label_task, sentence_label_format_output = (
        load_prompts(
            prompt_type=args.prompt_type,
            few_shot_samples_path=(
                args.few_shot_samples
                if args.prompt_type == "few-shot"
                else None
            ),
            seed=args.seed,
        )
    )

    # --------------------------------------------------------
    # 4. Run inference
    # --------------------------------------------------------
    y_hat_df = llm_classifer(
        model_name=args.model_name,
        model=model,
        test_df=test_df,
        base_prompt=base_prompt,
        sentence_label_task=sentence_label_task,
        sentence_label_format_output=sentence_label_format_output,
        save_directory=save_directory,
        seed=args.seed,
        prompt_type=args.prompt_type,
    )

    # --------------------------------------------------------
    # 5. Create results dataframe
    # --------------------------------------------------------
    results_df = create_results_dataframe(
        test_df, y_hat_df, args.model_name, args.seed
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------
    evaluate_models(
        results_df=results_df,
        model_name=args.model_name,
        label_col_name=args.label_column,
        save_path=save_directory,
        seed=args.seed,
        prompt_type=args.prompt_type,
    )

    # --------------------------------------------------------
    # 7. Complete
    # --------------------------------------------------------
    print("\n" + "=" * 40)
    print("PIPELINE COMPLETE")
    print("=" * 40)
    print(f"Model: {args.model_name}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Results saved to: {save_directory}")
