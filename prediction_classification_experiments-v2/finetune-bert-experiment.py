# Fine-tune a pretrained BERT model for binary single-sentence classification.

import os
import sys
import torch
import random
import argparse

import numpy as np
import pandas as pd

from datasets import Dataset
from functools import partial
from datetime import datetime
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

if script_dir not in sys.path:
    sys.path.append(script_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)

from data_processing import DataProcessing
from metrics import EvaluationMetric


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def set_all_seeds(seed):
    """Set random seeds for reproducible splits and BERT fine-tuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def resolve_path(base_data_path, file_path):
    """Return an absolute path for either absolute or data-relative input."""
    if os.path.isabs(file_path):
        return file_path

    return os.path.join(base_data_path, file_path)


# ============================================================
# DATA: LOAD + TRANSFORM LABELS
# ============================================================

def load_dataset(
    base_data_path,
    dataset_path,
    text_column,
    label_column,
    sample_size=None,
    seed=300,
):
    """Load, validate, clean, and optionally sample a classification dataset."""
    print("\n" + "=" * 40)
    print("LOAD DATASET")
    print("=" * 40)

    data_path = resolve_path(base_data_path, dataset_path)
    print(f"Dataset path: {data_path}")

    df = DataProcessing.load_from_file(
        data_path,
        "csv",
        sep=",",
    )

    required_columns = [text_column, label_column]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    original_count = len(df)

    df = df.dropna(subset=[text_column, label_column]).copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""].copy()
    df[label_column] = df[label_column].astype(str).str.strip()

    if sample_size is not None:
        if sample_size > len(df):
            raise ValueError(
                f"sample_size={sample_size} exceeds usable rows={len(df)}."
            )

        df = df.sample(
            n=sample_size,
            random_state=seed,
        ).reset_index(drop=True)

        print(f"Sample size: {sample_size}")

    print(f"Original rows: {original_count}")
    print(f"Usable rows:   {len(df)}")
    print(f"Shape:         {df.shape}")

    print("\nClass distribution:")
    print(df[label_column].value_counts())

    print(f"\nPreview:\n{df[[text_column, label_column]].head(5)}\n")

    return df.reset_index(drop=True)


def create_label_mapping(df, label_column):
    """Create BERT-compatible integer IDs for exactly two labels."""
    unique_labels = sorted(df[label_column].unique().tolist())

    if len(unique_labels) != 2:
        raise ValueError(
            "This pipeline currently supports binary classification only. "
            f"Found {len(unique_labels)} labels: {unique_labels}"
        )

    label_to_id = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    print("\nLabel mapping:")
    for label, label_id in label_to_id.items():
        print(f"  {label_id} -> {label}")

    return label_to_id, id_to_label


# ============================================================
# DATA SPLITS: CREATE + SAVE
# ============================================================

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

    DataProcessing.save_to_file(
        train_df,
        path=split_dir,
        prefix="x_y_train_set",
        save_file_type="csv",
        include_version=False,
    )

    DataProcessing.save_to_file(
        val_df,
        path=split_dir,
        prefix="x_y_val_set",
        save_file_type="csv",
        include_version=False,
    )

    if test_df is not None:
        DataProcessing.save_to_file(
            test_df,
            path=split_dir,
            prefix="x_y_test_set",
            save_file_type="csv",
            include_version=False,
        )

    print(f"✓ Saved data splits to: {split_dir}")


# ============================================================
# HUGGING FACE DATASET PREPARATION
# ============================================================

def dataframe_to_hf_dataset(df, text_column, label_id_column):
    """Convert a pandas DataFrame to a Hugging Face Dataset."""
    hf_df = df[[text_column, label_id_column]].copy()
    hf_df = hf_df.rename(
        columns={label_id_column: "labels"}
    )

    return Dataset.from_pandas(
        hf_df,
        preserve_index=False,
    )


# ============================================================
# BERT TOKENIZATION
# ============================================================

def tokenize_batch(examples, tokenizer, text_column, max_length):
    """Tokenize a batch of sentences using a pretrained tokenizer."""
    return tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_length,
    )


def tokenize_dataset(
    dataset,
    tokenizer,
    text_column,
    max_length,
    split_name="dataset",
    original_df=None,
    label_column=None,
    show_examples=False,
):
    """Tokenize all sentences in a Hugging Face dataset."""
    tokenized_dataset = dataset.map(
        partial(
            tokenize_batch,
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=[text_column],
    )

    print(f"✓ Tokenized {split_name} sentences: {len(tokenized_dataset)}")

    if show_examples and original_df is not None and label_column is not None:
        print("\n" + "=" * 40)
        print(f"TOKENIZATION MINI EXAMPLE — 3 {split_name} samples")
        print("=" * 40)

        num_examples = min(3, len(tokenized_dataset))

        for i in range(num_examples):
            example = tokenized_dataset[i]
            original_text = original_df.iloc[i][text_column]
            label = original_df.iloc[i][label_column]

            tokens = tokenizer.convert_ids_to_tokens(
                example["input_ids"]
            )

            num_real_tokens = sum(example["attention_mask"])
            num_pad_tokens = (
                len(example["attention_mask"]) - num_real_tokens
            )

            print(f"\n--- Example {i + 1} ---")
            print(f"  Original sentence : {original_text}")
            print(f"  Label ID          : {label}")
            print(f"  Tokens            : {tokens[:num_real_tokens]}")
            print(
                f"  input_ids         : "
                f"{example['input_ids'][:num_real_tokens]}"
            )
            print(
                f"  attention_mask    : "
                f"{example['attention_mask'][:num_real_tokens]} "
                f"(+ {num_pad_tokens} zeros for padding)"
            )
            print(
                f"  Total length      : {len(example['input_ids'])} "
                f"(max_length={max_length})"
            )

    return tokenized_dataset


# ============================================================
# BERT: LOAD + CONFIGURE + FINETUNE
# ============================================================

def load_pretrained_bert_classifier(
    model_name,
    label_to_id,
    id_to_label,
):
    """Load pretrained BERT and add a randomly initialized classifier."""
    print(f"\n[1/5] Loading pretrained BERT checkpoint: {model_name}")
    print("      num_labels  : 2 (binary classification)")
    print(f"      label2id    : {label_to_id}")
    print(f"      id2label    : {id_to_label}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        label2id=label_to_id,
        id2label=id_to_label,
    )

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"      Total parameters     : {total_params:,}")
    print(f"      Trainable parameters : {trainable_params:,}")

    return model


def configure_finetuning_arguments(
    output_dir,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    epochs,
    weight_decay,
    logging_steps,
    seed,
):
    """Configure Hugging Face settings for BERT fine-tuning."""
    checkpoint_dir = os.path.join(
        output_dir,
        "training_checkpoints",
    )

    use_fp16 = torch.cuda.is_available()

    print("\n[2/5] Configuring TrainingArguments")
    print(f"      Output dir          : {checkpoint_dir}")
    print(f"      Learning rate       : {learning_rate}")
    print(f"      Train batch size    : {train_batch_size}")
    print(f"      Eval batch size     : {eval_batch_size}")
    print(f"      Epochs              : {epochs}")
    print(f"      Weight decay        : {weight_decay}")
    print("      Eval strategy       : epoch")
    print("      Save strategy       : epoch")
    print(f"      Logging steps       : {logging_steps}")
    print("      Save total limit    : 1")
    print(f"      FP16                : {use_fp16}")
    print(f"      Seed                : {seed}")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_total_limit=1,
        fp16=use_fp16,
        report_to=[],
        seed=seed,
    )

    return training_args


def bert_trainer_metrics(eval_pred):
    """Compute validation metrics from Hugging Face model outputs."""
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)

    return EvaluationMetric.custom_evaluation_metrics(
        y_true=y_true,
        y_prediction=y_pred,
    )


def build_bert_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
    tokenizer,
):
    """Combine BERT model, datasets, tokenizer, and metrics."""
    print("\n[3/5] Building Trainer")
    print(f"      Train examples      : {len(train_dataset)}")
    print(f"      Validation examples : {len(val_dataset)}")
    print("      Data collator       : dynamic batch padding")
    print("      Compute metrics     : Accuracy, Precision, Recall, F1 Score")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=bert_trainer_metrics,
    )

    return trainer


def run_bert_finetuning(trainer, logging_steps):
    """Run supervised fine-tuning on labeled training sentences."""
    print("\n[4/5] Starting fine-tuning...")
    print(f"      Logging frequency   : every {logging_steps} steps")
    print("      Training logs       : loss, grad_norm, learning_rate, epoch")
    print("      Validation logs     : eval_loss, Accuracy, Precision, Recall, F1 Score")

    trainer.train()

    return trainer


def save_finetuned_bert(
    finetuned_model_name,
    trainer,
    tokenizer,
    output_dir,
):
    """Save fine-tuned BERT weights, configuration, and tokenizer."""
    final_model_dir = os.path.join(
        output_dir,
        finetuned_model_name,
    )

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print("\n[5/5] Fine-tuning complete.")
    print(f"      ✓ Saved fine-tuned BERT model to : {final_model_dir}")
    print(f"      ✓ Saved tokenizer to             : {final_model_dir}")

    return final_model_dir


# ============================================================
# EVALUATION: HELPER FUNCTIONS
# ============================================================

def get_finetuned_bert_predictions(trainer, tokenized_dataset):
    """Generate BERT predicted labels and probability vectors."""
    print("\n" + "=" * 40)
    print("BERT PREDICTIONS")
    print("=" * 40)

    prediction_output = trainer.predict(tokenized_dataset)

    logits = prediction_output.predictions
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(shifted_logits)

    probabilities = exp_logits / np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    print(f"Number of BERT predictions : {len(y_pred)}")
    print(f"First 5 predicted labels   : {y_pred[:5]}")
    print(f"Probability matrix shape   : {probabilities.shape}")
    print(f"First probability vector   : {probabilities[0]}")

    return y_pred, probabilities


def evaluate_bert_predictions(
    y_true,
    y_pred,
    probabilities,
    split_name,
    seed,
    model_name,
):
    """Evaluate BERT predictions using shared classification metrics."""
    print("\n" + "=" * 40)
    print(f"EVALUATE BERT: {split_name.upper()}")
    print("=" * 40)

    eval_report = EvaluationMetric.eval_classification_report(
        y_true,
        y_pred,
    )

    cm, tn, fp, fn, tp = EvaluationMetric.get_confusion_matrix(
        y_true,
        y_pred,
        by_category=True,
    )

    positive_class_probabilities = probabilities[:, 1]

    roc_auc_score = EvaluationMetric.get_roc_auc(
        y_true,
        positive_class_probabilities,
    )

    pr_auc_score = EvaluationMetric.get_pr_auc(
        y_true,
        positive_class_probabilities,
    )

    print(f"Confusion Matrix:\n{cm}\n")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n")
    print(f"ROC-AUC Score: {roc_auc_score:.4f}")
    print(f"PR-AUC Score: {pr_auc_score:.4f}\n")

    metrics_df = pd.DataFrame([{
        "seed": seed,
        "model": model_name,
        "split": split_name,
        "train_accuracy": None,
        "val_accuracy": None,
        "test_accuracy": eval_report.get("accuracy", None),
        "precision_class_0": eval_report.get("0", {}).get("precision", None),
        "precision_class_1": eval_report.get("1", {}).get("precision", None),
        "recall_class_0": eval_report.get("0", {}).get("recall", None),
        "recall_class_1": eval_report.get("1", {}).get("recall", None),
        "f1_class_0": eval_report.get("0", {}).get("f1-score", None),
        "f1_class_1": eval_report.get("1", {}).get("f1-score", None),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_auc": roc_auc_score,
        "pr_auc": pr_auc_score,
    }])

    print("\n" + "=" * 40)
    print(f"METRICS SUMMARY WITH SEED {seed}")
    print("=" * 40)
    print(metrics_df)
    print()

    return metrics_df, cm, eval_report


def create_bert_results_dataframe(
    source_df,
    y_pred,
    probabilities,
    id_to_label,
):
    """Attach BERT predictions and probabilities to source data."""
    print("\n" + "=" * 40)
    print("CREATE BERT RESULTS DATAFRAME")
    print("=" * 40)

    results_df = source_df.copy()

    if "Ground Truth" in results_df.columns:
        results_df["Ground Truth"] = results_df[
            "Ground Truth"
        ].astype(int)

    results_df["BERT Predicted Label ID"] = y_pred

    results_df["BERT Predicted Label"] = [
        id_to_label[prediction]
        for prediction in y_pred
    ]

    results_df["BERT Probability Class 0"] = probabilities[:, 0]
    results_df["BERT Probability Class 1"] = probabilities[:, 1]

    print("✓ Added BERT predicted labels and probabilities")
    print(f"\nFinal shape: {results_df.shape}")
    print(f"\nPreview:\n{results_df.head(3)}\n")

    return results_df


def create_confusion_matrix_dataframe(cm, id_to_label):
    """Add readable actual and predicted labels to a confusion matrix."""
    return pd.DataFrame(
        cm,
        index=[
            f"Actual: {id_to_label[0]}",
            f"Actual: {id_to_label[1]}",
        ],
        columns=[
            f"Predicted: {id_to_label[0]}",
            f"Predicted: {id_to_label[1]}",
        ],
    )


# ============================================================
# EVALUATION: VALIDATION + TEST + EXTERNAL DATASETS
# ============================================================

def evaluate_and_save_bert_results(
    trainer,
    tokenized_dataset,
    original_df,
    label_id_column,
    id_to_label,
    split_name,
    seed,
    output_dir,
    model_name,
    csv_prefix="bert_predictions",
):
    """Run full BERT prediction, evaluation, and output-saving pipeline."""
    print("\n" + "=" * 40)
    print(f"BERT EVALUATE AND SAVE: {split_name.upper()}")
    print("=" * 40)

    split_dir = os.path.join(
        output_dir,
        split_name,
    )

    os.makedirs(split_dir, exist_ok=True)
    print(f"✓ Output directory: {split_dir}")

    y_pred, probabilities = get_finetuned_bert_predictions(
        trainer=trainer,
        tokenized_dataset=tokenized_dataset,
    )

    y_true = original_df[label_id_column].to_numpy()

    print(f"\nNumber of ground-truth labels : {len(y_true)}")
    print(f"First 5 ground-truth labels   : {y_true[:5]}")

    predictions_df = create_bert_results_dataframe(
        source_df=original_df,
        y_pred=y_pred,
        probabilities=probabilities,
        id_to_label=id_to_label,
    )

    DataProcessing.save_to_file(
        predictions_df,
        path=split_dir,
        prefix=csv_prefix,
        save_file_type="csv",
        include_version=False,
    )

    print(
        f"✓ Saved predictions CSV to: "
        f"{os.path.join(split_dir, f'{csv_prefix}.csv')}"
    )

    metrics_df, cm, eval_report = evaluate_bert_predictions(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        split_name=split_name,
        seed=seed,
        model_name=model_name,
    )

    DataProcessing.save_to_file(
        metrics_df,
        path=split_dir,
        prefix="bert_metrics_summary",
        save_file_type="csv",
        include_version=False,
    )

    print(
        f"✓ Saved metrics summary to: "
        f"{os.path.join(split_dir, 'bert_metrics_summary.csv')}"
    )

    confusion_matrix_df = create_confusion_matrix_dataframe(
        cm=cm,
        id_to_label=id_to_label,
    )

    print(f"\nConfusion Matrix:\n{confusion_matrix_df}\n")

    DataProcessing.save_to_file(
        confusion_matrix_df,
        path=split_dir,
        prefix="bert_confusion_matrix",
        save_file_type="csv",
        include_version=False,
    )

    print(
        f"✓ Saved confusion matrix to: "
        f"{os.path.join(split_dir, 'bert_confusion_matrix.csv')}"
    )

    return metrics_df, cm, eval_report


def evaluate_external_datasets(
    external_dataset_paths,
    base_data_path,
    trainer,
    tokenizer,
    text_column,
    label_column,
    label_to_id,
    id_to_label,
    output_dir,
    max_length,
    seed,
    model_name,
    sample_size=None,
):
    """Evaluate a fine-tuned BERT classifier on external labeled datasets."""
    if not external_dataset_paths:
        print("\nNo external datasets provided.")
        return

    print("\n" + "=" * 40)
    print("CROSS-DOMAIN EVALUATION")
    print("=" * 40)

    label_id_column = "_label_id"

    for dataset_path in external_dataset_paths:
        dataset_name = os.path.splitext(
            os.path.basename(dataset_path)
        )[0]

        print("\n" + "=" * 40)
        print(f"EXTERNAL DATASET: {dataset_name}")
        print("=" * 40)

        external_df = load_dataset(
            base_data_path=base_data_path,
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column,
            sample_size=sample_size,
            seed=seed,
        )

        external_df[label_id_column] = external_df[
            label_column
        ].map(label_to_id)

        unknown_labels = external_df[
            external_df[label_id_column].isna()
        ][label_column].unique()

        if len(unknown_labels) > 0:
            print(
                f"⚠️ Skipping {dataset_name}. "
                f"Unknown labels: {unknown_labels.tolist()}"
            )
            continue

        external_df[label_id_column] = external_df[
            label_id_column
        ].astype(int)

        external_dataset = dataframe_to_hf_dataset(
            external_df,
            text_column=text_column,
            label_id_column=label_id_column,
        )

        tokenized_external_dataset = tokenize_dataset(
            dataset=external_dataset,
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=max_length,
            split_name=f"external_{dataset_name}",
        )

        evaluate_and_save_bert_results(
            trainer=trainer,
            tokenized_dataset=tokenized_external_dataset,
            original_df=external_df,
            label_id_column=label_id_column,
            id_to_label=id_to_label,
            split_name=f"external_{dataset_name}",
            seed=seed,
            output_dir=output_dir,
            model_name=model_name,
            csv_prefix=f"bert_predictions_{dataset_name}",
        )


# ============================================================
# EXPERIMENT LOG
# ============================================================

def create_bert_experiment_log(
    args,
    experiment_name,
    output_dir,
    train_df,
    val_df,
    test_df,
    label_to_id,
    final_model_dir,
):
    """Save a readable record of the BERT experiment configuration."""
    log_lines = [
        "=" * 40,
        "BERT SENTENCE CLASSIFICATION EXPERIMENT LOG",
        "=" * 40,
        f"Timestamp:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiment:             {experiment_name}",
        f"Seed:                   {args.seed}",
        "",
        "--- Data ---",
        f"Dataset:                {args.dataset}",
        f"Text Column:            {args.text_column}",
        f"Label Column:           {args.label_column}",
        f"Sample Size:            {args.sample_size}",
        "",
        "--- Splits ---",
        f"Validation Size:        {args.val_size}",
        f"Test Size:              {args.test_size}",
        f"Train Rows:             {len(train_df)}",
        f"Validation Rows:        {len(val_df) if val_df is not None else 'N/A'}",
        f"Test Rows:              {len(test_df) if test_df is not None else 'N/A'}",
        "",
        "--- BERT Model ---",
        f"Pretrained Checkpoint:  {args.model_name}",
        f"Finetuned Model Name:   {args.finetuned_model_name}",
        f"Finetuned Model Path:   {final_model_dir}",
        f"Number of Labels:       {len(label_to_id)}",
        "",
        "--- Training Arguments ---",
        f"Epochs:                 {args.epochs}",
        f"Learning Rate:          {args.learning_rate}",
        f"Weight Decay:           {args.weight_decay}",
        f"Train Batch Size:       {args.train_batch_size}",
        f"Evaluation Batch Size:  {args.eval_batch_size}",
        f"Maximum Token Length:   {args.max_length}",
        f"Logging Steps:          {args.logging_steps}",
        f"FP16:                   {torch.cuda.is_available()}",
        "",
        "--- Label Mapping ---",
    ]

    for label, label_id in label_to_id.items():
        log_lines.append(f"  {label_id} -> {label}")

    log_lines.append("")
    log_lines.append("--- External Test Datasets ---")

    if args.test_datasets:
        for dataset_path in args.test_datasets:
            log_lines.append(f"  - {dataset_path}")
    else:
        log_lines.append("  None")

    log_lines.extend([
        "",
        "--- Output ---",
        f"Output Directory:       {output_dir}",
        "=" * 40,
    ])

    log_dir = os.path.join(
        output_dir,
        "experiment_log",
    )

    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        "experiment_log.txt",
    )

    with open(log_path, "w", encoding="utf-8") as file:
        file.write("\n".join(log_lines))

    print(f"✓ Experiment log saved to: {log_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    """
    In-domain train / validation / test experiment:

    python finetune-bert-experiment.py \
        --dataset ../data/combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \
        --save_path ../data/classification_results/naacl_2026_submission \
        --text_column "Base Sentence" \
        --label_column "Ground Truth" \
        --model_name bert-base-cased \
        --seed 200 \
        --val_size 0.2 \
        --test_size 0.2 \
        --max_length 512 \
        --epochs 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --logging_steps 25 \
        --finetuned_model_name tolsa-bert \
        --sample_size 1000 \
        --test_datasets

    Train on one dataset and evaluate external datasets:

    python finetune-bert-experiment.py \
        --dataset ../data/combined_datasets/july_2026_results/july_2026_results.csv \
        --save_path ../data/classification_results/july_2026_results \
        --text_column "Base Sentence" \
        --label_column "Ground Truth" \
        --model_name bert-base-cased \
        --seed 300 \
        --val_size 0.2 \
        --test_size 0 \
        --max_length 512 \
        --epochs 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --logging_steps 25 \
        --finetuned_model_name tolsa-bert \
        --sample_size 1000 \
        --test_datasets \
            financial_phrase_bank/fpb.csv \
            chronicle2050/data.csv
    """

    print("\n" + "=" * 40)
    print("BERT SENTENCE CLASSIFICATION PIPELINE")
    print("=" * 40)

    # ============================================================
    # CONFIGURATION
    # ============================================================

    base_data_path = DataProcessing.load_base_data_path(
        script_dir
    )

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained BERT model for "
            "binary single-sentence classification."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset path relative to the project data directory "
            "or an absolute file path."
        ),
    )

    parser.add_argument(
        "--save_path",
        required=True,
        help="Directory in which experiment outputs are saved.",
    )

    parser.add_argument(
        "--text_column",
        default="Base Sentence",
        help="Name of the sentence-text column.",
    )

    parser.add_argument(
        "--label_column",
        default="Ground Truth",
        help="Name of the classification-label column.",
    )

    parser.add_argument(
        "--model_name",
        default="bert-base-cased",
        help="Hugging Face pretrained BERT checkpoint.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=300,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of the overall data used for validation.",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help=(
            "Proportion of the overall data used for in-domain testing. "
            "Set to 0 for no internal test split."
        ),
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum number of BERT tokenizer output tokens.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1,
        help="Number of fine-tuning epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate for fine-tuning.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay regularization.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device.",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Validation and test batch size per device.",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=25,
        help="Number of steps between Trainer progress logs.",
    )

    parser.add_argument(
        "--finetuned_model_name",
        default="tolsa-bert",
        help="Directory name for the saved fine-tuned model.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help=(
            "Optional number of rows to randomly sample for debugging. "
            "Default: use all usable rows."
        ),
    )

    parser.add_argument(
        "--test_datasets",
        nargs="*",
        default=None,
        help="Optional labeled datasets for external evaluation.",
    )

    args = parser.parse_args()

    set_all_seeds(args.seed)

    # ============================================================
    # EXPERIMENT SETUP
    # ============================================================

    current_date = datetime.now().strftime("%Y-%m-%d")

    dataset_filename = os.path.basename(args.dataset)
    dataset_base = os.path.splitext(dataset_filename)[0]

    experiment_name = f"{dataset_base}_{current_date}"

    output_dir = os.path.join(
        args.save_path,
        experiment_name,
        f"seed{args.seed}",
        "in_domain",
        args.model_name,
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset base: {dataset_base}")
    print(f"Experiment name: {experiment_name}")
    print(f"Output directory: {output_dir}")

    # ============================================================
    # DATA: LOAD + TRANSFORM LABELS
    # ============================================================

    df = load_dataset(
        base_data_path=base_data_path,
        dataset_path=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    label_to_id, id_to_label = create_label_mapping(
        df=df,
        label_column=args.label_column,
    )

    label_id_column = "_label_id"

    df[label_id_column] = df[
        args.label_column
    ].map(label_to_id).astype(int)

    num_labels = len(label_to_id)

    print(f"Usable rows:   {len(df)}")
    print(f"Shape:         {df.shape}")

    print("\nClass distribution:")
    print(df[label_id_column].value_counts())

    # ============================================================
    # DATA SPLITS: CREATE + SAVE
    # ============================================================

    train_df, val_df, test_df = split_dataset(
        df=df,
        label_id_column=label_id_column,
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

    # ============================================================
    # HUGGING FACE DATASET PREPARATION
    # ============================================================

    train_dataset = dataframe_to_hf_dataset(
        train_df,
        text_column=args.text_column,
        label_id_column=label_id_column,
    )

    val_dataset = dataframe_to_hf_dataset(
        val_df,
        text_column=args.text_column,
        label_id_column=label_id_column,
    )

    if test_df is not None:
        test_dataset = dataframe_to_hf_dataset(
            test_df,
            text_column=args.text_column,
            label_id_column=label_id_column,
        )
    else:
        test_dataset = None

    # ============================================================
    # BERT TOKENIZE
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    tokenized_train_dataset = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_length=args.max_length,
        split_name="training",
        original_df=train_df,
        label_column=label_id_column,
        show_examples=True,
    )

    tokenized_val_dataset = tokenize_dataset(
        dataset=val_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_length=args.max_length,
        split_name="validation",
    )

    if test_dataset is not None:
        tokenized_test_dataset = tokenize_dataset(
            dataset=test_dataset,
            tokenizer=tokenizer,
            text_column=args.text_column,
            max_length=args.max_length,
            split_name="in_domain_test",
        )
    else:
        tokenized_test_dataset = None
        print("✓ Tokenized test sentences: 0")

    # ============================================================
    # BERT FINETUNE: LOAD PRETRAINED MODEL
    # ============================================================

    model = load_pretrained_bert_classifier(
        model_name=args.model_name,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )

    # ============================================================
    # BERT FINETUNE: CONFIGURE TRAINING ARGUMENTS
    # ============================================================

    training_args = configure_finetuning_arguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        seed=args.seed,
    )

    # ============================================================
    # BERT FINETUNE: BUILD TRAINER
    # ============================================================

    trainer = build_bert_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_train_dataset,
        val_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
    )

    # ============================================================
    # BERT FINETUNE: TRAIN
    # ============================================================

    trainer = run_bert_finetuning(
        trainer=trainer,
        logging_steps=args.logging_steps,
    )

    # ============================================================
    # BERT FINETUNE: SAVE MODEL + TOKENIZER
    # ============================================================

    final_model_dir = save_finetuned_bert(
        finetuned_model_name=args.finetuned_model_name,
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    # ============================================================
    # EVALUATION: VALIDATION
    # ============================================================

    val_metrics_df, val_cm, val_report = evaluate_and_save_bert_results(
        trainer=trainer,
        tokenized_dataset=tokenized_val_dataset,
        original_df=val_df,
        label_id_column=label_id_column,
        id_to_label=id_to_label,
        split_name="validation",
        seed=args.seed,
        output_dir=output_dir,
        model_name=args.finetuned_model_name,
        csv_prefix="bert_predictions_val",
    )

    # ============================================================
    # EVALUATION: IN-DOMAIN TEST
    # ============================================================

    if test_df is not None and tokenized_test_dataset is not None:
        test_metrics_df, test_cm, test_report = (
            evaluate_and_save_bert_results(
                trainer=trainer,
                tokenized_dataset=tokenized_test_dataset,
                original_df=test_df,
                label_id_column=label_id_column,
                id_to_label=id_to_label,
                split_name="in_domain_test",
                seed=args.seed,
                output_dir=output_dir,
                model_name=args.finetuned_model_name,
                csv_prefix="bert_predictions_test",
            )
        )

    # ============================================================
    # EVALUATION: EXTERNAL DATASETS
    # ============================================================

    evaluate_external_datasets(
        external_dataset_paths=args.test_datasets,
        base_data_path=base_data_path,
        trainer=trainer,
        tokenizer=tokenizer,
        text_column=args.text_column,
        label_column=args.label_column,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        output_dir=output_dir,
        max_length=args.max_length,
        seed=args.seed,
        model_name=args.finetuned_model_name,
        sample_size=args.sample_size,
    )

    # ============================================================
    # EXPERIMENT LOG
    # ============================================================

    create_bert_experiment_log(
        args=args,
        experiment_name=experiment_name,
        output_dir=output_dir,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_to_id=label_to_id,
        final_model_dir=final_model_dir,
    )

    # ============================================================
    # COMPLETE
    # ============================================================

    print("\n" + "=" * 40)
    print("BERT PIPELINE COMPLETE")
    print("=" * 40)
    print(f"Experiment:          {experiment_name}")
    print(f"Pretrained model:    {args.model_name}")
    print(f"Fine-tuned model:    {args.finetuned_model_name}")
    print(f"Number of labels:    {num_labels}")
    print(f"\n✓ All outputs saved to: {output_dir}\n")