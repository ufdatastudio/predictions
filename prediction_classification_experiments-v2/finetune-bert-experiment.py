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

def load_dataset(base_data_path, dataset_path, text_column, label_column):  
    """Load, validate, and clean a sentence-classification dataset."""  
    print("\n" + "=" * 40)  
    print("LOAD DATASET")  
    print("=" * 40)  
  
    data_path = resolve_path(base_data_path, dataset_path)  
    print(f"Dataset path: {data_path}")  
  
    # Uses your custom DataProcessing class to load the file
    df = DataProcessing.load_from_file(data_path, "csv", sep=",")

    df = df.sample(n=1000, random_state=140).reset_index(drop=True)
  
    required_columns = [text_column, label_column]  
    missing_columns = [column for column in required_columns if column not in df.columns]  
  
    if missing_columns:  
        raise ValueError(  
            f"Required columns are missing: {missing_columns}\n"  
            f"Available columns: {list(df.columns)}"  
        )  
  
    original_count = len(df)  
  
    # Remove missing text and missing labels.  
    df = df.dropna(subset=[text_column, label_column]).copy()  
  
    # Ensure all text values are strings.  
    df[text_column] = df[text_column].astype(str).str.strip()  
  
    # Remove blank sentences.  
    df = df[df[text_column] != ""].copy()  
  
    # Standardize labels as strings before creating integer label IDs.  
    df[label_column] = df[label_column].astype(str).str.strip()  
  
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

def split_dataset(df, label_id_column, test_size, val_size, seed):
    """Create stratified train, validation, and optional test data splits."""
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

def dataframe_to_hf_dataset(df, text_column, label_id_column):
    """Convert a pandas DataFrame to a Hugging Face Dataset."""
    hf_df = df[[text_column, label_id_column]].copy()
    hf_df = hf_df.rename(columns={label_id_column: "labels"})

    return Dataset.from_pandas(hf_df, preserve_index=False)

def tokenize_batch(examples, tokenizer, text_column, max_length):
    """Tokenize a batch of sentences using a pretrained tokenizer.
    Apply truncation and maximum sequence length constraints.
    Return tokenized model inputs (e.g., input IDs and attention masks).
    """
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
    show_examples=False
):
    """Tokenize all sentences in a Hugging Face dataset.
    Apply preprocessing and convert text into BERT-compatible inputs.
    Optionally display tokenization examples for inspection.
    """
    # Use functools.partial to pass the extra arguments without nesting
    tokenized_dataset = dataset.map(
        partial(
            tokenize_batch, 
            tokenizer=tokenizer, 
            text_column=text_column, 
            max_length=max_length
        ),
        batched=True,
        remove_columns=[text_column],
    )

    # Print tokenized count
    print(f"✓ Tokenized {split_name} sentences: {len(tokenized_dataset)}")

    # Print mini-example if requested
    if show_examples and original_df is not None and label_column is not None:
        print("\n" + "=" * 40)
        print(f"TOKENIZATION MINI EXAMPLE — 3 {split_name} samples")
        print("=" * 40)

        # Ensure we don't try to print more examples than exist
        num_examples = min(3, len(tokenized_dataset))
        
        for i in range(num_examples):
            example = tokenized_dataset[i]
            original_text = original_df.iloc[i][text_column]
            label = original_df.iloc[i][label_column]

            tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
            num_real_tokens = sum(example["attention_mask"])
            num_pad_tokens = len(example["attention_mask"]) - num_real_tokens

            print(f"\n--- Example {i + 1} ---")
            print(f"  Original sentence : {original_text}")
            print(f"  Label ID          : {label}")
            print(f"  Tokens            : {tokens[:num_real_tokens]}")
            print(f"  input_ids         : {example['input_ids'][:num_real_tokens]}")
            print(f"  attention_mask    : {example['attention_mask'][:num_real_tokens]} "
                  f"(+ {num_pad_tokens} zeros for padding)")
            print(f"  Total length      : {len(example['input_ids'])} (max_length={max_length})")

    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute validation metrics from Hugging Face model outputs."""
    logits, y_true = eval_pred
    # logits: raw output scores for each class, shape (N, 2).
    # y_true: ground-truth label IDs from the validation dataset, shape (N,).
    y_pred = np.argmax(logits, axis=-1)

    return EvaluationMetric.custom_evaluation_metrics(
        y_true=y_true,
        y_prediction=y_pred,
    )

def load_pretrained_bert_classifier(model_name, label_to_id, id_to_label):
    """Load pretrained BERT and add a randomly initialized binary classifier."""
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

    total_params = sum(parameter.numel() for parameter in model.parameters())
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
    checkpoint_dir = os.path.join(output_dir, "training_checkpoints")
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
    print(f"      FP16               : {use_fp16}")
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

def build_bert_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
    tokenizer,
):
    """Combine model, datasets, tokenizer, metrics, and configuration."""
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
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    return trainer

def run_bert_finetuning(trainer, logging_steps):
    """Run supervised fine-tuning on the labeled training sentences."""
    print("\n[4/5] Starting fine-tuning...")
    print(f"      Logging frequency   : every {logging_steps} steps")
    print("      Training logs       : loss, grad_norm, learning_rate, epoch")
    print("      Validation logs     : eval_loss, Accuracy, Precision, Recall, F1 Score")

    trainer.train()

    return trainer

def save_finetuned_bert(trainer, tokenizer, output_dir):
    """Save the fine-tuned model weights, configuration, and tokenizer."""
    final_model_dir = os.path.join(output_dir, "final_model")

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print("\n[5/5] Fine-tuning complete.")
    print(f"      ✓ Saved fine-tuned BERT model to : {final_model_dir}")
    print(f"      ✓ Saved tokenizer to             : {final_model_dir}")

    return final_model_dir

def softmax(logits):
    """Convert BERT classification logits into class probabilities."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def predict_dataset(trainer, tokenized_dataset):
    """Generate predicted labels and probabilities for all examples."""
    prediction_output = trainer.predict(tokenized_dataset)

    logits = prediction_output.predictions
    probabilities = softmax(logits)
    y_pred = np.argmax(probabilities, axis=1)

    return y_pred, probabilities

def get_true_labels(source_df, label_id_column):
    """Extract integer ground-truth labels from the source DataFrame."""
    return source_df[label_id_column].to_numpy()

def get_bert_predictions(trainer, tokenized_dataset):
    """Generate BERT predicted labels and probability vectors."""
    return predict_dataset(
        trainer=trainer,
        tokenized_dataset=tokenized_dataset,
    )

def evaluate_predictions(y_true, y_pred, probabilities, split_name, seed):
    """Use EvaluationMetric to evaluate binary BERT predictions."""
    print("\n" + "=" * 40)
    print(f"EVALUATE: {split_name.upper()}")
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

    metrics_df = pd.DataFrame([{
        "seed": seed,
        "split": split_name,
        "accuracy": eval_report.get("accuracy"),
        "precision_class_0": eval_report.get("0", {}).get("precision"),
        "precision_class_1": eval_report.get("1", {}).get("precision"),
        "recall_class_0": eval_report.get("0", {}).get("recall"),
        "recall_class_1": eval_report.get("1", {}).get("recall"),
        "f1_class_0": eval_report.get("0", {}).get("f1-score"),
        "f1_class_1": eval_report.get("1", {}).get("f1-score"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_auc": EvaluationMetric.get_roc_auc(
            y_true,
            positive_class_probabilities,
        ),
        "pr_auc": EvaluationMetric.get_pr_auc(
            y_true,
            positive_class_probabilities,
        ),
    }])

    print(f"Confusion Matrix:\n{cm}\n")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    return metrics_df, cm, eval_report

def create_bert_results_dataframe(source_df, y_pred, probabilities, id_to_label):
    """Attach BERT predictions and probabilities to source data."""
    results_df = source_df.copy()

    results_df["BERT Predicted Label ID"] = y_pred
    results_df["BERT Predicted Label"] = [
        id_to_label[prediction]
        for prediction in y_pred
    ]

    results_df["BERT Probability Class 0"] = probabilities[:, 0]
    results_df["BERT Probability Class 1"] = probabilities[:, 1]

    return results_df

def create_confusion_matrix_dataframe(cm, id_to_label):
    """Add readable actual and predicted labels to the confusion matrix."""
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

def save_split_outputs(
    predictions_df,
    metrics_df,
    confusion_matrix_df,
    output_dir,
    split_name,
):
    """Save predictions, metrics, and confusion matrix for one split."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    DataProcessing.save_to_file(
        predictions_df,
        path=split_dir,
        prefix="bert_predictions",
        save_file_type="csv",
        include_version=False,
    )

    DataProcessing.save_to_file(
        metrics_df,
        path=split_dir,
        prefix="bert_metrics_summary",
        save_file_type="csv",
        include_version=False,
    )

    DataProcessing.save_to_file(
        confusion_matrix_df,
        path=split_dir,
        prefix="bert_confusion_matrix",
        save_file_type="csv",
        include_version=False,
    )

    print(f"✓ Saved {split_name} outputs to: {split_dir}")

    return split_dir

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
):
    """Evaluate the trained BERT classifier on each external labeled dataset."""
    if not external_dataset_paths:
        return

    print("\n" + "=" * 40)
    print("CROSS-DOMAIN EVALUATION")
    print("=" * 40)

    label_id_column = "_label_id"

    for dataset_path in external_dataset_paths:
        external_df = load_dataset(
            base_data_path=base_data_path,
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column,
        )

        external_df[label_id_column] = external_df[label_column].map(label_to_id)

        unknown_labels = external_df[external_df[label_id_column].isna()][label_column].unique()

        if len(unknown_labels) > 0:
            print(
                f"⚠️ Skipping {dataset_path}. "
                f"Unknown labels not seen in training data: {unknown_labels.tolist()}"
            )
            continue

        external_df[label_id_column] = external_df[label_id_column].astype(int)

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
        )

        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

        y_true = get_true_labels(
            source_df=external_df,
            label_id_column=label_id_column,
        )

        y_pred, probabilities = get_bert_predictions(
            trainer=trainer,
            tokenized_dataset=tokenized_external_dataset,
        )

        metrics_df, cm, _ = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            split_name=f"external_{dataset_name}",
            seed=seed,
        )

        predictions_df = create_bert_results_dataframe(
            source_df=external_df,
            y_pred=y_pred,
            probabilities=probabilities,
            id_to_label=id_to_label,
        )

        confusion_matrix_df = create_confusion_matrix_dataframe(
            cm=cm,
            id_to_label=id_to_label,
        )

        save_split_outputs(
            predictions_df=predictions_df,
            metrics_df=metrics_df,
            confusion_matrix_df=confusion_matrix_df,
            output_dir=output_dir,
            split_name=f"external_{dataset_name}",
        )

def create_experiment_log(
    args,
    experiment_name,
    output_dir,
    train_df,
    val_df,
    test_df,
    label_to_id,
):
    """Save a readable record of the experiment configuration."""
    log_lines = [
        "=" * 40,
        "BERT SENTENCE CLASSIFICATION EXPERIMENT LOG",
        "=" * 40,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiment: {experiment_name}",
        f"Seed: {args.seed}",
        "",
        "--- Data ---",
        f"Dataset: {args.dataset}",
        f"Text column: {args.text_column}",
        f"Label column: {args.label_column}",
        f"Train size: {len(train_df)}",
        f"Validation size: {len(val_df)}",
        f"Test size: {len(test_df) if test_df is not None else 0}",
        "",
        "--- BERT Configuration ---",
        f"Pretrained checkpoint: {args.model_name}",
        f"Maximum token length: {args.max_length}",
        f"Epochs: {args.epochs}",
        f"Learning rate: {args.learning_rate}",
        f"Weight decay: {args.weight_decay}",
        f"Train batch size: {args.train_batch_size}",
        f"Evaluation batch size: {args.eval_batch_size}",
        "",
        "--- Label Mapping ---",
    ]

    for label, label_id in label_to_id.items():
        log_lines.append(f"{label_id}: {label}")

    log_lines.append("")
    log_lines.append("--- External Test Datasets ---")

    if args.test_datasets:
        for dataset_path in args.test_datasets:
            log_lines.append(f"- {dataset_path}")
    else:
        log_lines.append("None")

    log_path = os.path.join(output_dir, "experiment_log.txt")

    with open(log_path, "w", encoding="utf-8") as file:
        file.write("\n".join(log_lines))

    print(f"✓ Saved experiment log to: {log_path}")

if __name__ == "__main__":
    """
    Example: in-domain train / validation / test experiment
    python finetune-bert-experiment.py \
        --dataset ../data/combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \
        --save_path ../data/classification_results/naacl_2026_submission \
        --text_column "Base Sentence" \
        --label_column "Ground Truth" \
        --val_size 0.2 \
        --test_size 0.2 \
        --model_name bert-base-uncased \
        --seed 140

    Example: train on one dataset and evaluate only on external datasets
    python finetune-bert-experiment.py \
        --dataset ../data/combined_datasets/july_2026_results/july_2026_results.csv \
        --text_column "Base Sentence" \
        --label_column "Ground Truth" \
        --test_size 0 \
        --val_size 0.2 \
        --test_datasets financial_phrase_bank/fpb.csv chronicle2050/data.csv
    """

    print("\n" + "=" * 40)
    print("BERT SENTENCE CLASSIFICATION PIPELINE")
    print("=" * 40)

    # ============================================================
    # 1. CONFIGURATION
    # ============================================================

    base_data_path = DataProcessing.load_base_data_path(script_dir)

    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained BERT model for sentence classification."
    )

    parser.add_argument("--dataset",
        required=True,
        help="Dataset path relative to the project's data directory, or an absolute path.",
    )

    parser.add_argument("--save_path",
        default="../data/classification_results/naacl_2026_submission",
        help="Directory in which experiment outputs are saved.",
    )

    parser.add_argument("--text_column",
        default="Base Sentence",
        help="Name of the sentence-text column.",
    )

    parser.add_argument("--label_column",
        default="Ground Truth",
        help="Name of the classification-label column.",
    )

    parser.add_argument("--model_name",
        default="bert-base-cased",
        help="Hugging Face pretrained checkpoint for tokenizer and model.",
    )

    parser.add_argument("--seed",
        type=int,
        default=300,
        help="Random seed for reproducibility.",
    )

    parser.add_argument("--val_size",
        type=float,
        default=0.2,
        help="Proportion of the full dataset reserved for validation.",
    )

    parser.add_argument("--test_size",
        type=float,
        default=0.2,
        help="Proportion of the in-domain dataset reserved for final testing. Use 0 for no internal test split.",
    )

    parser.add_argument("--max_length",
        type=int,
        default=512,
        help="Maximum number of tokenizer output tokens per sentence.",
    )

    parser.add_argument("--epochs",
        type=float,
        default=3,
        help="Number of fine-tuning epochs.",
    )

    parser.add_argument("--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate for BERT fine-tuning.",
    )

    parser.add_argument("--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay regularization.",
    )

    parser.add_argument("--train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device.",
    )

    parser.add_argument("--eval_batch_size",
        type=int,
        default=32,
        help="Validation and test batch size per device.",
    )

    parser.add_argument("--logging_steps",
        type=int,
        default=25,
        help="How frequently Trainer logs training progress.",
    )

    parser.add_argument("--test_datasets",
        nargs="*",
        default=None,
        help="Optional labeled external datasets for cross-domain evaluation.",
    )

    args = parser.parse_args()

    set_all_seeds(args.seed)

    # ============================================================
    # 2. EXPERIMENT SETUP
    # ============================================================

    current_date = datetime.now().strftime("%Y-%m-%d")
    dataset_filename = os.path.basename(args.dataset)
    dataset_base = os.path.splitext(dataset_filename)[0]
    print(f"Dataset base: {dataset_base}")
    experiment_name = f"{dataset_base}_{current_date}"
    print(f"Experiment name: {experiment_name}")

    output_dir = os.path.join(
        "../data/classification_results",
        experiment_name,
        f"seed{args.seed}",
        "in_domain",
        args.model_name,
    )
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 3. DATA: LOAD + TRANSFORM TO HF
    # ============================================================

    df = load_dataset(
        base_data_path=base_data_path,
        dataset_path=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    label_to_id, id_to_label = create_label_mapping(
        df=df,
        label_column=args.label_column,
    )

    num_labels = len(label_to_id)

    if num_labels < 2:
        raise ValueError(
            "Sentence classification requires at least two unique labels."
        )

    label_id_column = "_label_id"
    df[label_id_column] = df[args.label_column].map(label_to_id).astype(int)

    print(f"Usable rows:   {len(df)}")
    print(f"Shape:         {df.shape}")

    print("\nClass distribution:")
    print(df[label_id_column].value_counts())

    print(f"\nPreview:\n{df[[args.text_column, label_id_column]].head(5)}\n")

    # ============================================================
    # 4. CREATE AND SAVE DATA SPLITS (TRAIN/VAL/TEST)
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
    # 5. HF PREPAPRE DATASET FOR TOKENIZATION
    # ============================================================

    train_dataset = dataframe_to_hf_dataset(
        train_df,
        text_column=args.text_column,
        label_id_column=label_id_column,
    )

    if val_df is not None:
        val_dataset = dataframe_to_hf_dataset(
            val_df,
            text_column=args.text_column,
            label_id_column=label_id_column
        )

    if test_df is not None:
        test_dataset = dataframe_to_hf_dataset(
            test_df,
            text_column=args.text_column,
            label_id_column=label_id_column
        )

    # ============================================================
    # 6. BERT TOKENIZE
    # ============================================================
    # YOUR pipeline is a SINGLE-sentence task.
    # Each sentence is represented as:
    #   Input:    [CLS]  The  stock  rose  [SEP]  [PAD] ... [PAD]
    #   Segment:  e_A    e_A   e_A   e_A   e_A    e_A  ...  e_A
    #   Position: e_0    e_1   e_2   e_3   e_4    e_5  ...  e_127
    #
    # For single-sentence tasks, segment IDs are all 0s — only
    # segment A is used, because there is no second sentence [1].

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_train_dataset = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_length=args.max_length,
        split_name="training",
        original_df=train_df,
        label_column=label_id_column,
        show_examples=True  # This triggers the mini-example printout
    )

    if val_df is not None and val_dataset:
        tokenized_val_dataset = tokenize_dataset(
            dataset=val_dataset,
            tokenizer=tokenizer,
            text_column=args.text_column,
            max_length=args.max_length,
            split_name="validation"
        )
    else:
        tokenized_val_dataset = None
        print("✓ Tokenized validation sentences: 0")

    if test_df is not None and test_dataset:
        tokenized_test_dataset = tokenize_dataset(
            dataset=test_dataset,
            tokenizer=tokenizer,
            text_column=args.text_column,
            max_length=args.max_length,
            split_name="test"
        )
    else:
        tokenized_test_dataset = None
        print("✓ Tokenized test sentences:       0")
        
    # ============================================================
    # 6. STEP 1 — LOAD PRETRAINED BERT
    # ============================================================

    model = load_pretrained_bert_classifier(
        model_name=args.model_name,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )

    # ============================================================
    # 7. STEP 2 — CONFIGURE TRAINING ARGUMENTS
    # ============================================================

    training_args = configure_finetuning_arguments(
        output_dir=seed_dir,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        seed=args.seed,
    )

    # ============================================================
    # 8. STEP 3 — BUILD TRAINER
    # ============================================================

    trainer = build_bert_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_train_dataset,
        val_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
    )

    # ============================================================
    # 9. STEP 4 — FINE-TUNE BERT
    # ============================================================

    trainer = run_bert_finetuning(
        trainer=trainer,
        logging_steps=args.logging_steps,
    )

    # ============================================================
    # 10. STEP 5 — SAVE FINE-TUNED MODEL AND TOKENIZER
    # ============================================================

    save_finetuned_bert(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=seed_dir,
    )

    # ============================================================
    # 11. GET GROUND-TRUTH LABELS — VALIDATION
    # ============================================================

    y_true_val = get_true_labels(
        source_df=val_df,
        label_id_column=label_id_column,
    )

    print(f"Validation ground-truth labels: {len(y_true_val)}")

    # ============================================================
    # 12. GET BERT PREDICTIONS AND PROBABILITIES — VALIDATION
    # ============================================================

    y_pred_val, probabilities_val = get_bert_predictions(
        trainer=trainer,
        tokenized_dataset=tokenized_val_dataset,
    )

    print(f"Validation predictions: {len(y_pred_val)}")

    # ============================================================
    # 13. EVALUATE BERT PREDICTIONS — VALIDATION
    # ============================================================

    metrics_df_val, cm_val, report_dict_val = evaluate_predictions(
        y_true=y_true_val,
        y_pred=y_pred_val,
        probabilities=probabilities_val,
        split_name="validation",
        seed=args.seed,
    )

    # ============================================================
    # 14. SAVE VALIDATION OUTPUTS
    # ============================================================

    predictions_df_val = create_bert_results_dataframe(
        source_df=val_df,
        y_pred=y_pred_val,
        probabilities=probabilities_val,
        id_to_label=id_to_label,
    )

    confusion_matrix_df_val = create_confusion_matrix_dataframe(
        cm=cm_val,
        id_to_label=id_to_label,
    )

    save_split_outputs(
        predictions_df=predictions_df_val,
        metrics_df=metrics_df_val,
        confusion_matrix_df=confusion_matrix_df_val,
        output_dir=seed_dir,
        split_name="validation",
    )

    # ============================================================
    # 15. GET GROUND-TRUTH LABELS — IN-DOMAIN TEST
    # ============================================================

    if test_df is not None and tokenized_test_dataset is not None:
        y_true_test = get_true_labels(
            source_df=test_df,
            label_id_column=label_id_column,
        )

        print(f"Test ground-truth labels: {len(y_true_test)}")

        # ============================================================
        # 16. GET BERT PREDICTIONS AND PROBABILITIES — IN-DOMAIN TEST
        # ============================================================

        y_pred_test, probabilities_test = get_bert_predictions(
            trainer=trainer,
            tokenized_dataset=tokenized_test_dataset,
        )

        print(f"Test predictions: {len(y_pred_test)}")

        # ============================================================
        # 17. EVALUATE BERT PREDICTIONS — IN-DOMAIN TEST
        # ============================================================

        metrics_df_test, cm_test, report_dict_test = evaluate_predictions(
            y_true=y_true_test,
            y_pred=y_pred_test,
            probabilities=probabilities_test,
            split_name="in_domain_test",
            seed=args.seed,
        )

        # ============================================================
        # 18. SAVE IN-DOMAIN TEST OUTPUTS
        # ============================================================

        predictions_df_test = create_bert_results_dataframe(
            source_df=test_df,
            y_pred=y_pred_test,
            probabilities=probabilities_test,
            id_to_label=id_to_label,
        )

        confusion_matrix_df_test = create_confusion_matrix_dataframe(
            cm=cm_test,
            id_to_label=id_to_label,
        )

        save_split_outputs(
            predictions_df=predictions_df_test,
            metrics_df=metrics_df_test,
            confusion_matrix_df=confusion_matrix_df_test,
            output_dir=seed_dir,
            split_name="in_domain_test",
        )

    # ============================================================
    # 19. EVALUATE ON EXTERNAL DATASETS
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
        output_dir=seed_dir,
        max_length=args.max_length,
        seed=args.seed,
    )

    # ============================================================
    # 20. SAVE EXPERIMENT LOG
    # ============================================================

    create_experiment_log(
        args=args,
        experiment_name=experiment_name,
        output_dir=seed_dir,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_to_id=label_to_id,
    )

    # ============================================================
    # 21. COMPLETE
    # ============================================================

    print("\n" + "=" * 40)
    print("BERT PIPELINE COMPLETE")
    print("=" * 40)
    print(f"Experiment: {experiment_name}")
    print(f"Fine-tuned model: {args.model_name}")
    print(f"Number of labels: {num_labels}")
    print(f"✓ All outputs saved to: {experiment_dir}\n")