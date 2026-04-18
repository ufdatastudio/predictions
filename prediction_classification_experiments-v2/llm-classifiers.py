# Before running, be sure ml-train.py has produced a test file via evaluate_external_datasets
# or the in-domain split saved by split_train_test.
import os
import re
import sys
import json
import time
import pprint
import argparse
import matplotlib
matplotlib.use('Agg')  # Prevent GUI windows from opening
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Get the current working directory of the script
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from data_visualizing import DataVisualizing
from prompts import SentenceClassificationPrompt
from prediction_properties import PredictionProperties
from text_generation_models import TextGenerationModelFactory

# How many sentence results to collect in memory before writing to disk
BATCH_SIZE = 50
# Stop after this many sentences (set to None to process all)
# STOP_AFTER = 10
STOP_AFTER = None


def load_dataset(dataset_path, text_column='Base Sentence', label_column='Sentence Label'):
    """
    Load the test split saved by ml-train.py.
    Does NOT resample — the split is already fixed by the ML pipeline
    so LLM and ML models evaluate on the exact same sentences.

    Parameters
    ----------
    dataset_path : str
        Path to the x_y_test_set.csv saved by ml-train.py.
    text_column : str
        Column name containing the sentences. Default is 'Base Sentence'.
    label_column : str
        Column name containing the true labels. Default is 'Sentence Label'.
    """
    print("\n" + "="*40)
    print("LOAD DATASET")
    print("="*40)

    print(f"Dataset path: {dataset_path}")
    df = DataProcessing.load_from_file(dataset_path, 'csv', sep=',', encoding='latin-1')
    # Strip all non-ASCII characters entirely from string columns
    # This is the safest fix when source data has deep unicode corruption
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(
            lambda x: x.encode('ascii', errors='ignore').decode('ascii') if isinstance(x, str) else x
        )

    # Validate required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")

    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head(7)}\n")
    print(f"\nPreview:\n{df.tail(7)}\n")
    print(f"Label distribution:\n{df[label_column].value_counts()}\n")

    return df


def build_model(model_name):
    """Initialize single LLM from factory for this SLURM job."""
    tgmf = TextGenerationModelFactory()

    print("\n" + "="*40)
    print("LLM TO LOAD")
    print("="*40)
    pp = pprint.PrettyPrinter(indent=3)
    pp.pprint(model_name)

    model = tgmf.create_instance(model_name)
    return model


def load_prompts(model_name):
    """
    Build the base prompt using SentenceClassificationPrompt with few-shot examples.
    Combines system identity, prediction properties, requirements, and examples.

    Parameters
    ----------
    model_name : str
        The name of the model being loaded. Used for logging only.

    Returns
    -------
    base_prompt : str
        The full prompt sent to the model before each sentence.
    task : str
        The labeling instruction for the model.
    format_output : str
        The expected JSON output format.
    """
    print("\n" + "="*50)
    print("LOAD PROMPTS")
    print("="*50)

    # Get prediction properties and requirements
    prediction_properties, prediction_requirements = PredictionProperties.get_prediction_properties_and_requirements()

    # Build the prompt using SentenceClassificationPrompt with few-shot examples
    prompt = SentenceClassificationPrompt()
    system_identity, task, format_output, examples = prompt.few_shot()

    # Combine into one base prompt sent before each sentence
    base_prompt = f"""{system_identity}
    Prediction Properties:
    {prediction_properties}
    Requirements:
    {prediction_requirements}
    Examples:
    {examples}
    """

    print(f"\nSystem Identity: {system_identity}")
    print(f"\nTask: {task}")
    print(f"\nFormat Output: {format_output}")
    print(f"\nBase Prompt: {base_prompt}")
    print(f"\n✓ Prompts loaded for model: {model_name}")

    return base_prompt, task, format_output


def _llm_classifier(
        sentence_to_classify,
        base_prompt,
        model,
        task,
        format_output,
        is_first):
    """
    Send a single sentence to the LLM and return the raw response and parsed label.

    Uses model.safe_chat_completion which already handles rate limiting,
    progressive retries, and bad request errors internally.

    Uses DataProcessing.parse_llm_json_response for robust JSON parsing which
    handles markdown code blocks, single quotes, and conversational filler text.

    Parameters
    ----------
    sentence_to_classify : str
        The sentence to classify.
    base_prompt : str
        The full prompt sent before each sentence.
    model : TextGenerationModelFactory
        The loaded LLM instance.
    task : str
        The labeling instruction for the model.
    format_output : str
        The expected JSON output format.
    is_first : bool
        If True, print the full prompt to terminal for debugging.

    Returns
    -------
    tuple
        (raw_text_llm_generation, label) or (None, None) on unrecoverable failure.
    """
    # Sanitize sentence to prevent Unicode encoding errors crashing the API call
    sentence_to_classify = sentence_to_classify.encode('ascii', errors='ignore').decode('ascii')

    prompt = f"""{base_prompt}

    Sentence to label: '{sentence_to_classify}'
    {task}

    {format_output}
    """

    if is_first:
        print(f"\tPrompt: {prompt}")

    input_prompt = model.user(prompt)

    # safe_chat_completion handles rate limiting, retries, and bad request errors internally
    # Returns None if all attempts fail
    raw_text_llm_generation = model.safe_chat_completion([input_prompt])

    if raw_text_llm_generation is None:
        return None, None

    # Use DataProcessing.parse_llm_json_response for robust JSON parsing
    # Handles markdown code blocks, single quotes, and conversational filler
    parsed = DataProcessing.parse_llm_json_response(raw_text_llm_generation)
    label = parsed.get('predicted_sentence_label') if parsed else None

    return raw_text_llm_generation, label


def llm_classifer(model_name, model, test_df, base_prompt, sentence_label_task, sentence_label_format_output, save_directory):
    """
    Run inference over all sentences in test_df with incremental checkpointing.
    Automatically resumes from checkpoint if the job was previously interrupted.

    Parameters
    ----------
    model_name : str
        Name of the LLM being used.
    model : TextGenerationModelFactory
        The loaded LLM instance.
    test_df : pd.DataFrame
        The test dataset loaded from ml-train.py's saved test split.
    base_prompt : str
        The full prompt sent before each sentence.
    sentence_label_task : str
        The labeling instruction for the model.
    sentence_label_format_output : str
        The expected JSON output format.
    save_directory : str
        Directory where checkpoint and results CSVs are saved.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: original_index, text, raw_response, llm_label, llm_name.
    """
    print(f"Shape: {test_df.shape}")
    print(f"\nPreview:\n{test_df.head(7)}\n")
    print(f"\nPreview:\n{test_df.tail(7)}\n")

    # Create safe filename for checkpoint (handles model names with slashes e.g. openai/gpt)
    safe_model_name = model_name.replace("/", "_")
    checkpoint_file = os.path.join(save_directory, f"checkpoint_{safe_model_name}.csv")

    # --------------------------------------------------------
    # Resume from checkpoint if it exists
    # --------------------------------------------------------
    if os.path.exists(checkpoint_file):
        print(f"\n[!] Found checkpoint file. Resuming from: {checkpoint_file}")
        processed_df = pd.read_csv(checkpoint_file)
        processed_indices = set(processed_df['original_index'].tolist())
        remaining_df = test_df[~test_df.index.isin(processed_indices)]
        results = processed_df.to_dict('records')
    else:
        remaining_df = test_df.copy()
        results = []
        # Initialize empty CSV with headers so append mode works correctly later
        pd.DataFrame(columns=['original_index', 'text', 'raw_response', 'llm_label', 'llm_name']).to_csv(checkpoint_file, index=False)

    print(f"Rows remaining to process: {len(remaining_df)}\n")
    print(model_name, model)

    batch_buffer = []

    for loop_idx, (idx, row) in enumerate(tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Processing")):
        print(idx, row)
        print()
        text = row['Base Sentence']

        if loop_idx < 3:
            print("Classify sentence as either prediction (1) or non-prediction (0)")
            print(f"    {idx} --- Sentence: {text}")

        is_first = (loop_idx == 0)
        raw_response, llm_label = _llm_classifier(
            text, base_prompt, model,
            sentence_label_task, sentence_label_format_output, is_first
        )

        result_dict = {
            'original_index': idx,
            'text': text,
            'raw_response': raw_response,
            'llm_label': llm_label,
            'llm_name': model_name
        }
        batch_buffer.append(result_dict)
        results.append(result_dict)

        if loop_idx < 3:
            print(f"\tLabel: {llm_label} via Model: {model_name}")

        # Checkpoint save every BATCH_SIZE rows so we don't lose progress on timeout
        if len(batch_buffer) >= BATCH_SIZE:
            pd.DataFrame(batch_buffer).to_csv(checkpoint_file, mode='a', header=False, index=False)
            print(f"✓ Checkpoint saved: {len(batch_buffer)} rows flushed to disk.")
            batch_buffer = []  # Clear buffer after flush

    # Save any remaining rows that didn't fill a full batch
    if batch_buffer:
        pd.DataFrame(batch_buffer).to_csv(checkpoint_file, mode='a', header=False, index=False)
        print(f"✓ Final checkpoint saved: {len(batch_buffer)} remaining rows flushed to disk.")

    # Reconstruct full DataFrame in the EXACT original order for metric mapping
    results_with_llm_label_df = pd.DataFrame(results)
    if not results_with_llm_label_df.empty:
        results_with_llm_label_df = (
            results_with_llm_label_df
            .set_index('original_index')
            .reindex(test_df.index)
            .reset_index()
        )

    print(f"Shape: {results_with_llm_label_df.shape}")
    print(f"\nPreview:\n{results_with_llm_label_df}\n")

    return results_with_llm_label_df


def create_results_dataframe(X_test_df, y_hat_df, model_name):
    """
    Combine test data with model predictions.

    Parameters
    ----------
    X_test_df : pd.DataFrame
        The original test DataFrame.
    y_hat_df : pd.DataFrame
        The LLM inference results containing llm_label column.
    model_name : str
        The model name used as the prediction column name.

    Returns
    -------
    pd.DataFrame
        Test DataFrame with model predictions appended as a new column.
    """
    print("\n" + "="*40)
    print("CREATE RESULTS DATAFRAME")
    print("="*40)

    results_df = X_test_df.copy()

    # Map predictions back into the original dataframe
    # y_hat_df is guaranteed to be in the same order as X_test_df (reindexed in llm_classifer)
    results_df[model_name] = y_hat_df['llm_label'].to_list()
    print(f"✓ Added predictions: {model_name}")

    print(f"\nFinal shape: {results_df.shape}")
    print(f"\nPreview:\n{results_df.head(3)}\n")

    return results_df


def evaluate_models(
    results_df,
    model_name,
    label_col_name,
    save_path,
    seed
    # X_val_df=None,
    # y_val_df=None
):
    """
    Evaluate LLM predictions and save unified metrics to disk.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing both ground truth labels and LLM predictions.
    model_name : str
        The model name used as the prediction column name.
    label_col_name : str
        The column name containing true labels.
    save_path : str
        Directory path to save metrics and visualizations.
    seed : int
        Random seed used in this run, stored in printed summary for tracking.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame of all evaluation metrics for this model.
    """
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Label: {label_col_name}\n")

    actual_labels = results_df[label_col_name].values
    predictions = results_df[model_name].values

    # Recreate X_test_list and y_test_df for DataVisualizing methods
    X_test_list = results_df['Base Sentence'].to_list()
    y_test_df = results_df[[label_col_name]]

    # Fill NaN predictions with 0 if LLM failed to parse a valid JSON label
    predictions = np.nan_to_num(np.array(predictions, dtype=float), nan=0.0)

    metrics_summary = []

    print(f"### Model: {model_name} ###")

    # Train/val metrics are N/A for zero-shot LLMs, keeping placeholders
    # val_acc = None
    # if y_val_df is not None:
    #     val_predictions = ...
    #     val_acc = EvaluationMetric.get_score(actual_val_labels, val_predictions)

    # For LLMs, we don't have predict_proba so we fallback to binary predictions for AUC
    continuous_scores = predictions

    # Classification report: precision, recall, f1, test accuracy
    eval_report = EvaluationMetric.eval_classification_report(actual_labels, predictions)

    # Confusion matrix
    confusion_mat, tn, fp, fn, tp = EvaluationMetric.get_confusion_matrix(actual_labels, predictions, by_category=True)
    print(f"Confusion Matrix:\n{confusion_mat}\n")

    # Save confusion matrix visualization
    DataVisualizing.confusion_matrix(
        model_name,
        confusion_mat,
        save_path,
        include_version=False
    )
    print(f"✓ Saved confusion matrix: confusion_matrix_{model_name}.png\n")

    # ROC-AUC score
    roc_auc_score = EvaluationMetric.get_roc_auc(actual_labels, continuous_scores)
    print(f"ROC-AUC Score: {roc_auc_score:.4f}\n")

    # ROC Curve
    # DataVisualizing.roc_curve(
    #     model_name,
    #     None, # No scikit-learn model object to pass
    #     X_test_list,
    #     y_test_df,
    #     save_path,
    #     include_version=False
    # )
    # print(f"✓ Saved ROC-CURVE: roc_curve_{model_name}.png\n")

    # PR-AUC
    pr_auc_score = EvaluationMetric.get_pr_auc(actual_labels, continuous_scores)
    print(f"PR-AUC Score: {pr_auc_score:.4f}\n")

    # PR Curve
    # DataVisualizing.pr_curve(
    #     model_name,
    #     None, # No scikit-learn model object to pass
    #     X_test_list,
    #     y_test_df,
    #     save_path,
    #     include_version=False
    # )
    # print(f"✓ Saved PR-CURVE: pr_curve_{model_name}.png\n")

    # Build unified metrics row matching ML pipeline's metrics_summary_ml_models.csv format
    metrics_row = {
        'model': model_name,
        # Train & val metrics are N/A for zero-shot LLMs
        'train_accuracy': None,
        # 'val_accuracy': val_acc,
        'val_accuracy': None,

        # Classification report metrics
        'test_accuracy': eval_report.get('accuracy', None),
        'precision_class_0': eval_report.get('0', {}).get('precision', None),
        'precision_class_1': eval_report.get('1', {}).get('precision', None),
        'recall_class_0': eval_report.get('0', {}).get('recall', None),
        'recall_class_1': eval_report.get('1', {}).get('recall', None),
        'f1_class_0': eval_report.get('0', {}).get('f1-score', None),
        'f1_class_1': eval_report.get('1', {}).get('f1-score', None),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,

        # AUC metrics
        'roc_auc': roc_auc_score,
        'pr_auc': pr_auc_score
    }
    metrics_summary.append(metrics_row)

    # Save unified metrics summary
    metrics_summary_df = pd.DataFrame(metrics_summary)
    print("\n" + "="*40)
    print(f"METRICS SUMMARY WITH SEED {seed}")
    print("="*40)
    print(metrics_summary_df)
    print()

    # Save metrics CSV alongside ML model metrics for easy comparison
    DataProcessing.save_to_file(
        data=metrics_summary_df,
        path=save_path,
        prefix=f"metrics_summary_{model_name}",
        save_file_type='csv',
        include_version=False
    )
    print(f"✓ Saved metrics summary: metrics_summary_{model_name}.csv\n")

    return metrics_summary_df


if __name__ == "__main__":
    """
    usage:
    # Run single LLM job on HiPerGator, loading test split saved by ml-train.py

    # In-domain test set (saved by ml-train.py at seed_dir/in_domain/x_y_test_set.csv):
    python llm-classifiers.py \\
        --model_name gpt-oss-120b \\
        --test_dataset ../data/classification_results/synthetic-fpb-c2050-yt-news-timebank_2026-04-17/seed3/in_domain/x_y_test_set.csv \\
        --label_column 'Ground Truth'

    # External cross-domain test set (saved by ml-train.py at seed_dir/external_*/x_y_test_set.csv):
    python llm-classifiers.py \\
        --model_name gpt-oss-120b \\
        --test_dataset ../data/classification_results/synthetic-fpb-c2050-yt-news-timebank_2026-04-17/seed3/external_fpb-maya-binary-imbalanced-96d-v1/x_y_test_set.csv \\
        --label_column 'Sentence Label'
    """
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    # Default points to the external test set saved by ml-train.py
    default_load_save_path = os.path.join(
        base_data_path,
        'classification_results/train_synthetic-v1_2026-03-23/seed3/external_fpb-maya-binary-imbalanced-96d-v1'
    )
    default_dataset_path = os.path.join(default_load_save_path, 'x_y_test_set.csv')

    parser = argparse.ArgumentParser(
        description='LLM zero-shot sentence classification. Loads test split produced by ml-train.py.'
    )

    parser.add_argument(
        '--test_dataset',
        default=default_dataset_path,
        help='Path to x_y_test_set.csv saved by ml-train.py. '
             'Can be an in_domain or external_* subfolder path.'
    )

    parser.add_argument(
        '--model_name',
        required=True,
        help='The specific single LLM to use as the classifier for this SLURM job.'
    )

    parser.add_argument(
        '--label_column',
        default='Sentence Label',
        help='The column name in the dataset containing the true labels.'
    )

    # Placeholder for val datasets arguments
    # parser.add_argument('--val_datasets', default=None, help='Paths to validation datasets.')

    args = parser.parse_args()

    # ============================================================
    # 2. EXPERIMENT SETUP
    # ============================================================
    current_date = datetime.now().strftime('%Y-%m-%d')
    seed_value = 42  # Default random seed for tracking

    # Save results alongside the test file so ML and LLM outputs sit in the same folder
    save_directory = os.path.dirname(args.test_dataset)
    print(f"\nSave directory: {save_directory}")

    # ============================================================
    # 3. LOAD & PREPARE DATA
    # ============================================================
    # Load the exact same test split that ml-train.py used
    test_df = load_dataset(args.test_dataset, label_column=args.label_column)

    # Placeholder for loading validation data
    # val_df = load_dataset(args.val_datasets) if args.val_datasets else None

    # Build the single LLM for this SLURM job
    model = build_model(args.model_name)

    # Load the base prompt, task instruction, and output format
    base_prompt, sentence_label_task, sentence_label_format_output = load_prompts(args.model_name)

    # ============================================================
    # 4. RUN INFERENCE
    # ============================================================
    y_hat_df = llm_classifer(
        model_name=args.model_name,
        model=model,
        test_df=test_df,
        base_prompt=base_prompt,
        sentence_label_task=sentence_label_task,
        sentence_label_format_output=sentence_label_format_output,
        save_directory=save_directory
    )

    # ============================================================
    # 5. CREATE RESULTS DATAFRAME
    # ============================================================
    results_df = create_results_dataframe(test_df, y_hat_df, args.model_name)

    # ============================================================
    # 6. EVALUATE MODELS
    # ============================================================
    metrics_summary_df = evaluate_models(
        results_df=results_df,
        model_name=args.model_name,
        label_col_name=args.label_column,
        save_path=save_directory,
        seed=seed_value
        # X_val_df=val_df,
        # y_val_df=val_df[[args.label_column]] if val_df is not None else None
    )

    # ============================================================
    # 9. COMPLETE
    # ============================================================
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("="*40)
    print(f"Model: {args.model_name}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Results saved to: {save_directory}")