import os
import re
import sys
import json
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
from prediction_properties import PredictionProperties
from text_generation_models import TextGenerationModelFactory

# If you have a DataVisualizing module, uncomment this:
# from visualizations import DataVisualizing 

def load_dataset(dataset_path):
    """Load dataset from file path."""
    print("\n" + "="*40)
    print("LOAD DATASET")
    print("="*40)
    
    print(f"Dataset path: {dataset_path}")
    df = DataProcessing.load_from_file(dataset_path, 'csv', sep=',')
    
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head(7)}\n")
    print(f"\nPreview:\n{df.tail(7)}\n")
    df = df.sample(n=40, random_state=42) # Added random_state for reproducibility
    
    return df

def build_model(model_name):
    """Initialize single ML model from factory for parallel processing."""
    tgmf = TextGenerationModelFactory()
    
    print("\n" + "="*40)
    print("LLM TO LOAD")
    print("="*40)
    pp = pprint.PrettyPrinter(indent=3)
    pp.pprint(model_name)
    
    models = tgmf.create_instance(model_name)
    return models

def load_prompt():
    system_identity_prompt = "You are an expert at identifying specific types of sentences called prediction."
    prediction_requirements = PredictionProperties.get_requirements()
    prediction_properties = PredictionProperties.get_prediction_properties()
    prediction_properties_base_prompt = f"""{system_identity_prompt} For each prediction, the format is based on: {prediction_properties} \nEnforce the {prediction_requirements}"""
    sentence_label_task = """Classify the sentence "label" as either a "non-prediction": 0, "prediction": 1."""
    sentence_label_format_output = """Respond ONLY with valid JSON in this exact format: {"predicted_sentence_label": 0}. Do NOT reason or provide anything other than {"predicted_sentence_label": 0}. """
    
    print("\n" + "="*40)
    print("PROMPT")
    print("="*40)
    print(f"\nTask: {sentence_label_task}")
    print(f"\nFormat Output: {sentence_label_format_output}")
    print(f"\nBase Prompt: {prediction_properties_base_prompt}")
    
    return (sentence_label_task, sentence_label_format_output, prediction_properties_base_prompt)

def parse_json_response(response, reasoning=False):
    """Parse JSON response from LLM to extract label and reasoning"""
    try:
        # Extract JSON if there's extra text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if reasoning:
                return data.get('predicted_sentence_label'), data.get('reasoning')
            else:
                return data.get('predicted_sentence_label')  # Return single value, not tuple
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        if reasoning:
            return None, None
        else:
            return None  # Return single None when reasoning=False

def _llm_classifier(sentence_to_classify: str, base_prompt: str, model, task, format_output: str, is_first: bool):
      prompt = f"""{base_prompt}
      
      Sentence to label: '{sentence_to_classify}'
      {task}
      
      {format_output}
      """
      
      if is_first:
            print(f"\tPrompt: {prompt}")
            
      input_prompt = model.user(prompt)
      raw_text_llm_generation = model.chat_completion([input_prompt])
      # print(f"Raw response: {raw_text_llm_generation}")
      # Parse the JSON response
      label = parse_json_response(raw_text_llm_generation, reasoning=False)
      
      return raw_text_llm_generation, label

def llm_classifer(model_name, model, test_df, base_prompt, sentence_label_task, sentence_label_format_output):
    print(f"Shape: {test_df.shape}")
    print(f"\nPreview:\n{test_df.head(7)}\n")
    print(f"\nPreview:\n{test_df.tail(7)}\n")
    
    results = []
    print(model_name, model)
    
    for loop_idx, (idx, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="Processing")):
        print(idx, row)
        print()
        text = row['Base Sentence']
        
        if loop_idx < 3:
            print("Classify sentence as either prediction (1) or non-prediction (0)")
            print(f"    {idx} --- Sentence: {text}")
            
        is_first = (loop_idx == 0)
        raw_response, llm_label = _llm_classifier(text, base_prompt, model, sentence_label_task, sentence_label_format_output, is_first)
        result = (text, raw_response, llm_label, model_name)
        results.append(result)
        
        if loop_idx < 3:
            print(f"\tLabel: {llm_label} via Model: {model_name}")
            
    results_with_llm_label_df = pd.DataFrame(results, columns=['text', 'raw_response', 'llm_label', 'llm_name'])
    print(f"Shape: {results_with_llm_label_df.shape}")
    print(f"\nPreview:\n{results_with_llm_label_df}\n")
    print(f"\nPreview:\n{results_with_llm_label_df}\n")
    return results_with_llm_label_df

def create_results_dataframe(X_test_df, y_hat_df, model_name):
    """Combine test data with model predictions."""
    print("\n" + "="*40)
    print("CREATE RESULTS DATAFRAME")
    print("="*40)
    
    results_df = X_test_df.copy()
    
    # We map the predictions back into the original dataframe
    # Assuming y_hat_df is ordered the same as X_test_df
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
    Evaluate all model predictions and save unified metrics.
    """
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Label: {label_col_name}\n")
    
    actual_labels = results_df[label_col_name].values
    predictions = results_df[model_name].values
    
    # Fill NaN predictions with a default (e.g., 0) if LLM failed to parse
    predictions = np.nan_to_num(np.array(predictions, dtype=float), nan=0.0)
    
    metrics_summary = []
    
    print(f"### Model: {model_name} ###")
    
    # Train/val metrics are N/A for zero-shot LLMs, keeping placeholders
    # val_acc = None
    # if y_val_df is not None:
    #     val_predictions = ... 
    #     val_acc = EvaluationMetric.get_score(actual_val_labels, val_predictions)
    
    # --- GET CONTINUOUS SCORES FOR AUC MATCHING ---
    # For LLMs, we don't have predict_proba by default, so we fallback to binary predictions
    continuous_scores = predictions 
    
    # Classification report: precision, recall, f1, test accuracy
    eval_report = EvaluationMetric.eval_classification_report(actual_labels, predictions)
    
    # Confusion matrix
    confusion_mat, tn, fp, fn, tp = EvaluationMetric.get_confusion_matrix(actual_labels, predictions, by_category=True)
    print(f"Confusion Matrix:\n{confusion_mat}\n")
    
    # Save confusion matrix visualization
    # DataVisualizing.confusion_matrix(
    #     model_name,
    #     confusion_mat, 
    #     save_path, 
    #     include_version=False
    # )
    print(f"✓ Saved confusion matrix: confusion_matrix_{model_name}.png\n")
    
    # ROC-AUC score
    roc_auc_score = EvaluationMetric.get_roc_auc(actual_labels, continuous_scores)
    print(f"ROC-AUC Score: {roc_auc_score:.4f}\n")
    
    # POC Curve
    # DataVisualizing.roc_curve(
    #     model_name,
    #     None, # No scikit-learn model object to pass 
    #     X_test_list, 
    #     y_test_df,
    #     save_path, 
    #     include_version=False
    # )
    print(f"✓ Saved POC-CURVE: pos_curve{model_name}.png\n")
    
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
    print(f"✓ Saved PR-CURVE: pr_curve{model_name}.png\n")
            
    # Build unified metrics row
    metrics_row = {
        'model': model_name,
        # Train & val metrics placeholder
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
    
    # metrics_summary_df.to_csv(os.path.join(save_path, f"metrics_summary_{model_name}.csv"), index=False)
    
    return metrics_summary_df
    
if __name__ == "__main__":
    """
    usage: 
    # E1: Run single model job for SLURM on HiPerGator
    python llm-classifiers.py \
        --model_name llama-3.1-8b-instant \
        --test_datasets financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
        --label_column 'Sentence Label'
    """
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    
    default_load_save_path = os.path.join(base_data_path, 'classification_results/train_synthetic-v1_2026-03-23/seed3/external_fpb-maya-binary-imbalanced-96d-v1')
    default_dataset_path = os.path.join(default_load_save_path, 'x_y_test_set.csv')
    print(default_dataset_path)
    
    parser = argparse.ArgumentParser(
        description='Train ML classifiers for prediction sentence classification'
    )
    
    parser.add_argument(
        '--test_datasets', 
        default=default_dataset_path,
        help='Paths to external test datasets for cross-domain evaluation.'
    )
    
    parser.add_argument(
        '--model_name', 
        required=True,
        help='The specific single LLM to use as the classifier for this run.'
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
    seed_value = 42 # Default random seed for tracking
    save_directory = os.path.dirname(args.test_datasets)
    
    # ============================================================
    # 3. LOAD & PREPARE DATA
    # ============================================================
    test_df = load_dataset(args.test_datasets)
    
    # Placeholder for loading validation data
    # val_df = load_dataset(args.val_datasets) if args.val_datasets else None
    
    model = build_model(args.model_name)
    sentence_label_task, sentence_label_format_output, prediction_properties_base_prompt = load_prompt()
    
    # ============================================================
    # 4. RUN INFERENCE
    # ============================================================
    y_hat_df = llm_classifer(
        args.model_name,
        model,
        test_df,
        prediction_properties_base_prompt,
        sentence_label_task,
        sentence_label_format_output
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