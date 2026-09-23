# Fine-tune a pretrained BERT model for binary single-sentence classification.
#
# Example:
# python finetune_bert_sentence_classifier.py \
#     --dataset combined_datasets/combined-full_synthetic-v1.csv \
#     --text_column "Base Sentence" \
#     --label_column "Ground Truth" \
#     --val_size 0.2 \
#     --test_size 0.2 \
#     --model_name bert-base-cased

import os
import sys
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ============================================================
# PROJECT IMPORTS
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

if script_dir not in sys.path:
    sys.path.append(script_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)

from data_processing import DataProcessing
from metrics import EvaluationMetric


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_all_seeds(seed):
    """Set random seeds for reproducible splits and BERT fine-tuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ============================================================
# PATHS AND OUTPUTS
# ============================================================

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


def create_output_directory(args, experiment_name):
    """Create output directory matching ML pipeline structure."""
    # classification_results/naacl_2026_submission/seed#/in_domain/bert-base-cased/experiment_name
    seed_dir = os.path.join(
        args.save_path,
        experiment_name,
        f"seed{args.seed}",
        "in_domain",
        args.model_name.replace("/", "_"),  # e.g. bert-base-cased
    )

    os.makedirs(seed_dir, exist_ok=True)

    experiment_dir = os.path.join(args.save_path, experiment_name)

    print(f"\n✓ Experiment directory: {experiment_dir}")
    print(f"✓ Seed directory: {seed_dir}")

    return experiment_dir, seed_dir


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def load_sentence_dataset(base_data_path, dataset_path, text_column, label_column):
    """Load and validate a labeled sentence-classification CSV dataset."""
    print("\n" + "=" * 40)
    print("LOAD DATASET")
    print("=" * 40)

    full_dataset_path = resolve_path(base_data_path, dataset_path)
    print(f"Dataset path: {full_dataset_path}")

    df = DataProcessing.load_from_file(
        full_dataset_path,
        file_type="csv",
        sep=",",
    )

    required_columns = [text_column, label_column]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    original_count = len(df)

    df = df.dropna(subset=[text_column, label_column]).copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""].copy()
    df[label_column] = df[label_column].astype(str).str.strip()

    print(f"Original rows: {original_count}")
    print(f"Usable rows: {len(df)}")
    print(f"\nClass distribution:\n{df[label_column].value_counts()}")
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
# TRAIN / VALIDATION / TEST SPLITS
# ============================================================

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

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df) if test_df is not None else 0}")

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
    hf_df = hf_df.rename(columns={label_id_column: "labels"})

    return Dataset.from_pandas(hf_df, preserve_index=False)


def tokenize_dataset(dataset, tokenizer, text_column, max_length):
    """Tokenize single-sentence BERT inputs."""
    def tokenize_batch(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[text_column],
    )


# ============================================================
# TRAINER METRICS
# ============================================================

def build_compute_metrics():
    """Use EvaluationMetric metrics during Hugging Face validation."""
    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)

        return EvaluationMetric.custom_evaluation_metrics(
            y_true=y_true,
            y_prediction=y_pred,
        )

    return compute_metrics


# ============================================================
# BERT FINE-TUNING
# ============================================================

def train_bert_model(
    train_dataset,
    val_dataset,
    model_name,
    label_to_id,
    id_to_label,
    tokenizer,
    output_dir,
    args,
):
    """Fine-tune pretrained BERT and save its final model weights."""
    print("\n" + "=" * 40)
    print("FINE-TUNE BERT")
    print("=" * 40)

    # --------------------------------------------------------
    # STEP 1: LOAD PRETRAINED BERT
    # --------------------------------------------------------
    # From the D2L tutorial [1]: BERT is pretrained on BookCorpus +
    # English Wikipedia (800M + 2.5B words) using MLM and NSP tasks.
    # Here we are NOT pretraining — we are loading those pretrained
    # weights and attaching a new classification head on top.
    # The warning "Some weights not initialized" is EXPECTED — it refers
    # to the new classifier.weight and classifier.bias which are
    # randomly initialized and will be learned during fine-tuning.
    # From the HuggingFace tutorial: this is called transfer learning.
    print(f"\n[1/5] Loading pretrained BERT checkpoint: {model_name}")
    print(f"      num_labels  : 2 (binary classification)")
    print(f"      label2id    : {label_to_id}")
    print(f"      id2label    : {id_to_label}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        label2id=label_to_id,
        id2label=id_to_label,
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total parameters     : {total_params:,}")
    print(f"      Trainable parameters : {trainable_params:,}")

    # --------------------------------------------------------
    # STEP 2: CONFIGURE TRAINING ARGUMENTS
    # --------------------------------------------------------
    # From the HuggingFace tutorial: TrainingArguments controls
    # everything about HOW the Trainer will train the model —
    # learning rate, batch sizes, epochs, evaluation frequency, etc.
    # From D2L [1]: BERT uses Adam optimizer with weight decay.
    print(f"\n[2/5] Configuring TrainingArguments")
    print(f"      Output dir          : {os.path.join(output_dir, 'training_checkpoints')}")
    print(f"      Learning rate       : {args.learning_rate}")
    print(f"      Train batch size    : {args.train_batch_size}")
    print(f"      Eval batch size     : {args.eval_batch_size}")
    print(f"      Epochs              : {args.epochs}")
    print(f"      Weight decay        : {args.weight_decay}")
    print(f"      Eval strategy       : epoch (evaluate after every full pass through train data)")
    print(f"      Save strategy       : epoch (save checkpoint after every epoch)")
    print(f"      Logging steps       : {args.logging_steps} (log loss every N steps)")
    print(f"      Save total limit    : 1 (only keep the best checkpoint on disk)")
    print(f"      FP16 (mixed prec.)  : {torch.cuda.is_available()} (True only if GPU available)")
    print(f"      Seed                : {args.seed}")

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "training_checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    # --------------------------------------------------------
    # STEP 3: BUILD THE TRAINER
    # --------------------------------------------------------
    # From the HuggingFace tutorial: the Trainer wraps the model,
    # data, and training configuration into a single object.
    # DataCollatorWithPadding dynamically pads each batch to the
    # length of the longest sentence in that batch — more efficient
    # than padding everything to max_length upfront.
    # compute_metrics tells the Trainer to call EvaluationMetric
    # after each epoch to report Accuracy, Precision, Recall, F1.
    print(f"\n[3/5] Building Trainer")
    print(f"      Train examples      : {len(train_dataset)}")
    print(f"      Validation examples : {len(val_dataset)}")
    print(f"      Data collator       : DataCollatorWithPadding (dynamic per-batch padding)")
    print(f"      Compute metrics     : Accuracy, Precision, Recall, F1 Score")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
    )

    # --------------------------------------------------------
    # STEP 4: FINE-TUNE
    # --------------------------------------------------------
    # From D2L [1]: fine-tuning updates ALL pretrained Transformer
    # encoder parameters (unlike ELMo which freezes them) while
    # simultaneously training the new classification head.
    # The [CLS] token representation is what flows into the
    # classification head — the same [CLS] token used in BERT's
    # pretraining NSP task [1].
    # During training the Hugging Face Trainer will print a log
    # line every args.logging_steps steps showing:
    #   loss       → cross-entropy loss on the current training batch
    #   grad_norm  → magnitude of the gradient (spikes = instability)
    #   learning_rate → current LR after linear warmup + decay
    #   epoch      → fractional progress through the dataset
    # At the end of each epoch it will print eval metrics.
    print(f"\n[4/5] Starting fine-tuning...")
    print(f"      Every {args.logging_steps} steps you will see:")
    print(f"        loss         → cross-entropy loss on that training batch")
    print(f"        grad_norm    → gradient magnitude (large spikes = instability)")
    print(f"        learning_rate → current LR value (decays toward 0)")
    print(f"        epoch        → fractional progress (e.g. 1.5 = halfway through epoch 2)")
    print(f"      At the end of each epoch you will see eval metrics:")
    print(f"        eval_loss, eval_Accuracy, eval_Precision, eval_Recall, eval_F1 Score\n")

    trainer.train()

    # --------------------------------------------------------
    # STEP 5: SAVE THE FINE-TUNED MODEL
    # --------------------------------------------------------
    # Saves the full fine-tuned BERT model weights + tokenizer so
    # you can reload them later for inference without retraining.
    final_model_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"\n[5/5] Fine-tuning complete.")
    print(f"      ✓ Saved fine-tuned BERT model to : {final_model_dir}")
    print(f"      ✓ Saved tokenizer to             : {final_model_dir}")
    print(f"      To reload later: AutoModelForSequenceClassification.from_pretrained('{final_model_dir}')")

    return trainer


# ============================================================
# PREDICTION AND EVALUATION
# ============================================================

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


def create_predictions_dataframe(
    source_df,
    y_pred,
    probabilities,
    id_to_label,
):
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


def evaluate_and_save_split(
    trainer,
    source_df,
    tokenized_dataset,
    label_id_column,
    id_to_label,
    output_dir,
    split_name,
    seed,
):
    """Predict, evaluate, and save results for validation, test, or external data."""
    y_true = source_df[label_id_column].to_numpy()

    y_pred, probabilities = predict_dataset(
        trainer,
        tokenized_dataset,
    )

    metrics_df, cm, report_dict = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        split_name=split_name,
        seed=seed,
    )

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    predictions_df = create_predictions_dataframe(
        source_df=source_df,
        y_pred=y_pred,
        probabilities=probabilities,
        id_to_label=id_to_label,
    )

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

    confusion_matrix_df = pd.DataFrame(
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

    DataProcessing.save_to_file(
        confusion_matrix_df,
        path=split_dir,
        prefix="bert_confusion_matrix",
        save_file_type="csv",
        include_version=False,
    )

    print(f"✓ Saved {split_name} outputs to: {split_dir}")

    return metrics_df, predictions_df, report_dict

# ============================================================
# EXTERNAL / CROSS-DOMAIN EVALUATION
# ============================================================

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
        # Load external dataset using your custom DataProcessing logic
        external_df = load_dataset(
            base_data_path=base_data_path,
            dataset_path=dataset_path,
            text_column=text_column,
            label_column=label_column,
        )

        # Map external labels to the IDs used during training
        external_df[label_id_column] = external_df[label_column].map(label_to_id)

        # Check for unseen labels in the external dataset
        unknown_labels = external_df[external_df[label_id_column].isna()][label_column].unique()

        if len(unknown_labels) > 0:
            print(
                f"⚠️ Skipping {dataset_path}. "
                f"Unknown labels not seen in training data: {unknown_labels.tolist()}"
            )
            continue

        external_df[label_id_column] = external_df[label_id_column].astype(int)

        # Convert to Hugging Face dataset
        external_dataset = dataframe_to_hf_dataset(
            external_df,
            text_column=text_column,
            label_id_column=label_id_column,
        )

        # Tokenize
        tokenized_external_dataset = tokenize_dataset(
            dataset=external_dataset,
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=max_length,
        )

        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

        # Predict, evaluate, and save artifacts
        evaluate_and_save_split(
            trainer=trainer,
            source_df=external_df,
            tokenized_dataset=tokenized_external_dataset,
            label_id_column=label_id_column,
            id_to_label=id_to_label,
            output_dir=output_dir,
            split_name=f"external_{dataset_name}",
            seed=seed,
        )


# ============================================================
# EXPERIMENT LOG
# ============================================================

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


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    """
    Example: in-domain train / validation / test experiment
    python finetune-bert-experiment.py \
        --dataset ../data/combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \
        --text_column "Base Sentence" \
        --label_column "Ground Truth" \
        --val_size 0.2 \
        --test_size 0.2 \
        --model_name bert-base-cased \
        --seed 140

    
    Example: train on one dataset and evaluate only on external datasets
    python fintune-bert-experiment.py.py \
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

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset path relative to the project's data directory, or an absolute path.",
    )

    parser.add_argument(
        "--save_path",
        default=os.path.join(base_data_path, "classification_results"),  # same root as ML pipeline
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
        help="Hugging Face pretrained checkpoint for tokenizer and model.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the in-domain dataset reserved for final testing. Use 0 for no internal test split.",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of the full dataset reserved for validation.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum number of tokenizer output tokens per sentence.",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=3,
        help="Number of fine-tuning epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate for BERT fine-tuning.",
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
        help="How frequently Trainer logs training progress.",
    )

    parser.add_argument(
        "--test_datasets",
        nargs="*",
        default=None,
        help="Optional labeled external datasets for cross-domain evaluation.",
    )

    parser.add_argument(
        "--experiment_suffix",
        default="",
        help="Optional suffix appended to the experiment directory name.",
    )

    args = parser.parse_args()

    set_all_seeds(args.seed)

    # ============================================================
    # 2. EXPERIMENT SETUP
    # ============================================================
    current_date = datetime.now().strftime("%Y-%m-%d")
    dataset_filename = os.path.basename(args.dataset)
    dataset_base = os.path.splitext(dataset_filename)[0]

    experiment_base = dataset_base + args.experiment_suffix
    experiment_name = f"{experiment_base}_{current_date}"

    print(experiment_name)

    experiment_dir, seed_dir = create_output_directory(
        args=args,
        experiment_name=experiment_name,
    )

    print(f"\nExperiment: {experiment_name}")
    print(f"Model: {args.model_name}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {seed_dir}")

    # ============================================================
    # 3. LOAD, CLEAN, AND LABEL THE DATA
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
    # 4. CREATE AND SAVE DATA SPLITS
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
        output_dir=seed_dir,
    )

    # ============================================================
    # TOKENIZATION EXAMPLE — what each sentence looks like after tokenization
    # ============================================================
    #
    # The book image shows a MULTIPLE input sequence:
    #   Input:    [CLS]  this  movie  is  great  [SEP]  i  like  it  [SEP]
    #   Segment:  e_A    e_A   e_A    e_A  e_A   e_A   e_B e_B  e_B  e_B
    #   Position: e_0    e_1   e_2    e_3  e_4   e_5   e_6 e_7  e_8  e_9
    #
    # YOUR pipeline is a SINGLE-sentence task (sentiment / prediction classification).
    # So each sentence looks like:
    #   Input:    [CLS]  The  stock  rose  sharply  [SEP]  [PAD] ... [PAD]
    #   Segment:  e_A    e_A   e_A   e_A    e_A     e_A    e_A  ...  e_A
    #   Position: e_0    e_1   e_2   e_3    e_4     e_5    e_6  ...  e_127
    #
    # The Hugging Face tokenizer handles ALL of this automatically:
    #   - Inserts [CLS] at position 0
    #   - Inserts [SEP] after the last real token
    #   - Pads to max_length with [PAD] tokens
    #   - Builds attention_mask: 1 for real tokens, 0 for [PAD]
    #   - input_ids:       integer ID for every token in the vocabulary
    #   - attention_mask:  tells BERT which tokens to attend to vs. ignore
    #
    # NOTE: For single-sentence tasks, segment IDs (token_type_ids) are all 0s
    # — only segment A (e_A) is used, because there is no second sentence [1].
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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

    tokenized_train_dataset = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_length=args.max_length,
    )

    tokenized_val_dataset = tokenize_dataset(
        dataset=val_dataset,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_length=args.max_length,
    )

    if test_df is not None:
        test_dataset = dataframe_to_hf_dataset(
            test_df,
            text_column=args.text_column,
            label_id_column=label_id_column,
        )

        tokenized_test_dataset = tokenize_dataset(
            dataset=test_dataset,
            tokenizer=tokenizer,
            text_column=args.text_column,
            max_length=args.max_length,
        )
    else:
        tokenized_test_dataset = None

    print(f"✓ Tokenized training sentences: {len(tokenized_train_dataset)}")
    print(f"✓ Tokenized validation sentences: {len(tokenized_val_dataset)}")
    print(
        f"✓ Tokenized test sentences: "
        f"{len(tokenized_test_dataset) if tokenized_test_dataset is not None else 0}"
    )

    
    print("\n" + "=" * 40)
    print("TOKENIZATION MINI EXAMPLE — 3 train samples")
    print("=" * 40)

    for i in range(3):
        example = tokenized_train_dataset[i]
        original_text = train_df.iloc[i][args.text_column]
        label = train_df.iloc[i][label_id_column]

        # Decode token IDs back to human-readable tokens including special tokens
        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])

        # Count real tokens (attention_mask == 1) vs padding (attention_mask == 0)
        num_real_tokens = sum(example["attention_mask"])
        num_pad_tokens  = len(example["attention_mask"]) - num_real_tokens

        print(f"\n--- Example {i + 1} ---")
        print(f"  Original sentence : {original_text}")
        print(f"  Label ID          : {label}")
        print(f"  Tokens            : {tokens[:num_real_tokens]}")
        print(f"  input_ids         : {example['input_ids'][:num_real_tokens]}")
        print(f"  attention_mask    : {example['attention_mask'][:num_real_tokens]} "
              f"(+ {num_pad_tokens} zeros for padding)")
        print(f"  Total length      : {len(example['input_ids'])} (max_length={args.max_length})")

    # ============================================================
    # 6. FINE-TUNE PRETRAINED BERT
    # ============================================================

    trainer = train_bert_model(
        train_dataset=tokenized_train_dataset,
        val_dataset=tokenized_val_dataset,
        model_name=args.model_name,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        tokenizer=tokenizer,
        output_dir=seed_dir,
        args=args,
    )

    quit()
    # ============================================================
    # 7. EVALUATE ON VALIDATION DATA
    # ============================================================

    evaluate_and_save_split(
        trainer=trainer,
        source_df=val_df,
        tokenized_dataset=tokenized_val_dataset,
        label_id_column=label_id_column,
        id_to_label=id_to_label,
        output_dir=seed_dir,
        split_name="validation",
        seed=args.seed,
    )

    # ============================================================
    # 8. EVALUATE ON FINAL IN-DOMAIN TEST DATA
    # ============================================================

    if test_df is not None and tokenized_test_dataset is not None:
        evaluate_and_save_split(
            trainer=trainer,
            source_df=test_df,
            tokenized_dataset=tokenized_test_dataset,
            label_id_column=label_id_column,
            id_to_label=id_to_label,
            output_dir=seed_dir,
            split_name="in_domain_test",
            seed=args.seed,
        )

    # ============================================================
    # 9. EVALUATE ON EXTERNAL DATASETS
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
    # 10. SAVE EXPERIMENT LOG
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
    # 11. COMPLETE
    # ============================================================

    print("\n" + "=" * 40)
    print("BERT PIPELINE COMPLETE")
    print("=" * 40)
    print(f"Experiment: {experiment_name}")
    print(f"Fine-tuned model: {args.model_name}")
    print(f"Number of labels: {num_labels}")
    print(f"✓ All outputs saved to: {experiment_dir}\n")