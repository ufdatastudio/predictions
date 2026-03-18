# Before this, run python3 create_combined_dataset.py to create dataset
import os
import sys
import joblib
import argparse
import matplotlib
matplotlib.use('Agg')  # Prevent GUI windows from opening
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime
# Get the current working directory of the script
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))
from metrics import EvaluationMetric
from data_processing import DataProcessing
from data_visualizing import DataVisualizing
from feature_extraction import SpacyFeatureExtraction
from classification_models import SkLearnModelFactory
from explainability import Explainability

def create_output_directory(args, experiment_name):
    """Create unique output directory with date and seed."""
    seed_number = f"seed{args.seed}"
    
    experiment_dir = os.path.join(args.save_path, experiment_name)
    seed_dir = os.path.join(experiment_dir, seed_number)
    
    # This single call creates both the experiment_dir and the seed_dir inside it
    os.makedirs(seed_dir, exist_ok=True)
    
    print(f"\n✓ Experiment directory: {experiment_dir}")
    print(f"✓ Seed directory: {seed_dir}")
    
    return experiment_dir, seed_dir
def load_dataset(script_dir, dataset_path):
    """Load dataset from file path."""
    print("\n" + "="*40)
    print("LOAD DATASET")
    print("="*40)
    
    if not os.path.isabs(dataset_path):
        data_path = os.path.join(script_dir, dataset_path)
    else:
        data_path = dataset_path
    
    print(f"Dataset path: {data_path}")
    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head(7)}\n")
    
    return df
def get_which_dataset(df, dataset_name):
    """Filter combined dataset based on author type."""
    print("\n" + "="*40)
    print("FILTER COMBINED DATASET")
    print("="*40)
    print(f"Dataset selection: {dataset_name}")
    
    if 'Author Type' not in df.columns:
        raise ValueError(
            f"'Author Type' column required for dataset filtering but not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    print(f"\nAuthor Type distribution (before filtering):")
    print(df['Author Type'].value_counts())
    
    if dataset_name == "synthetic_fin_phrasebank":
        result_df = df
        print("\nNo filtering applied - using full combined dataset")
    elif dataset_name == "synthetic":
        result_df = df[df['Author Type'] == 0]
        print("\nFiltered to synthetic only (Author Type == 0)")
    elif dataset_name == "fin_phrasebank":
        result_df = df[df['Author Type'] == 1]
        print("\nFiltered to financial phrasebank only (Author Type == 1)")
    else:
        raise ValueError(
            f"Unknown dataset name: '{dataset_name}'\n"
            f"Valid options: 'synthetic_fin_phrasebank', 'synthetic', 'fin_phrasebank'"
        )
    
    print(f"\nFinal dataset shape: {result_df.shape}")
    if len(result_df) > 0:
        print(f"Author Type distribution (after filtering):")
        print(result_df['Author Type'].value_counts())
    print()
    
    return result_df
def shuffle_dataset(df, seed):
    """Shuffle dataset rows."""
    print("\n" + "="*40)
    print("SHUFFLE DATASET")
    print("="*40)
    
    shuffled_df = DataProcessing.shuffle_df(df, random_state=seed)
    print(f"Shape: {shuffled_df.shape}")
    print(f"\nPreview:\n{shuffled_df.head(7)}\n")
    
    return shuffled_df
def extract_sentence_embeddings(df, text_column='Base Sentence'):
    """Extract sentence embeddings using SpaCy."""
    print("\n" + "="*40)
    print("EXTRACT SENTENCE EMBEDDINGS (SpaCy)")
    print("="*40)
    print(f"Using text column: '{text_column}'")
    
    spacy_fe = SpacyFeatureExtraction(df, text_column)
    embeddings_df = spacy_fe.sentence_embeddings_extraction(attach_to_df=True)
    
    embeddings_col_name = f'{text_column} Embedding'
    
    # Replaces your current loop in extract_sentence_embeddings:
    for i, (idx, row) in enumerate(embeddings_df.head(3).iterrows()):
        text = row[text_column]
        embedding = row[embeddings_col_name]
        
        # Force into a numpy array so .shape is guaranteed to work
        emb_array = np.array(embedding)
        
        print(f"\nSample {i}:")
        print(f"  Sentence [:100]: {str(text)[:100]}...")
        print(f"  Embedding shape: {emb_array.shape}")
    
    return embeddings_df, embeddings_col_name
def split_train_test(df, seed, val_size=None, stratify_by='Sentence Label'):
    """Split dataset into train/test or train/val/test sets."""
    cols_with_labels = df.loc[:, [stratify_by]]
    
    if val_size is None:
        print("\n" + "="*40)
        print("SPLIT TRAIN/TEST DATA")
        print("="*40)
        print(f"Stratifying by: {stratify_by}")
        
        X_train_df, X_test_df, y_train_df, y_test_df = DataProcessing.split_data(
            df, cols_with_labels, 
            test_size=0.2,
            val_size=None,
            random_state=seed, 
            stratify_by=stratify_by
        )
        
        print("\n{:<25} {:>10}".format("Dataset", "Count"))
        print("-" * 37)
        print("{:<25} {:>10}".format("X_train", len(X_train_df)))
        print("{:<25} {:>10}".format("X_test", len(X_test_df)))
        print("{:<25} {:>10}".format("y_train", len(y_train_df)))
        print("{:<25} {:>10}".format("y_test", len(y_test_df)))
        print()
        
        return X_train_df, X_test_df, y_train_df, y_test_df
    
    else:
        print("\n" + "="*40)
        print("SPLIT TRAIN/VAL/TEST DATA")
        print("="*40)
        print(f"Stratifying by: {stratify_by}")
        
        X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df = DataProcessing.split_data(
            df, cols_with_labels,
            test_size=0.2,
            val_size=val_size,
            random_state=seed,
            stratify_by=stratify_by
        )
        
        print("\n{:<25} {:>10}".format("Dataset", "Count"))
        print("-" * 37)
        print("{:<25} {:>10}".format("X_train", len(X_train_df)))
        print("{:<25} {:>10}".format("X_val", len(X_val_df)))
        print("{:<25} {:>10}".format("X_test", len(X_test_df)))
        print("{:<25} {:>10}".format("y_train", len(y_train_df)))
        print("{:<25} {:>10}".format("y_val", len(y_val_df)))
        print("{:<25} {:>10}".format("y_test", len(y_test_df)))
        print()
        
        return X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df
def build_models(factory, model_names, seed):
    """Initialize ML models from factory."""
    models = {}
    for name in model_names:
        models[name] = factory.select_model(name, random_state=seed)
    return models
def train_and_predict_models(
    ml_model_names,
    X_train_df,
    y_train_df,
    X_test_df,
    embeddings_col_name,
    label_name,
    model_checkpoint_path,
    seed,
    X_val_df=None,
    y_val_df=None
):
    """
    Train models and generate predictions.
    
    Returns
    -------
    tuple
        (predictions_dict, train_val_metrics_dict, trained_models_dict)
        - predictions_dict: {model_name: predictions_array}
        - train_val_metrics_dict: {model_name: {'train_accuracy': float, 'val_accuracy': float}}
        - trained_models_dict: {model_name: model_instance}
    """
    print("\n" + "="*40)
    print("TRAIN & PREDICT MODELS")
    print("="*40)
    print(f"Label: {label_name}")
    print(f"Models: {len(ml_model_names)}")
    trained_models_with_predictions = {}
    train_val_metrics = {}
    
    models = build_models(SkLearnModelFactory, ml_model_names, seed=seed)
    # Prepare train data
    X_train_list = X_train_df[embeddings_col_name].to_list()
    y_train_list = y_train_df.values.ravel()
    # Prepare validation data if provided
    # No in-domain test set, just train models without predictions
    # We'll generate predictions later on external datasets
    # Ex: synthetic only, so split with train/val and no test.
    if X_val_df is not None and y_val_df is not None:
        X_val_list = X_val_df[embeddings_col_name].to_list()
        y_val_list = y_val_df.values.ravel()
    else:
        X_val_list = None
        y_val_list = None
        val_acc = None
    # Prepare test data if provided
    # Train models (predictions on in-domain test if exists, otherwise None)
    # Ex: synthetic + fpb + chronicle2050, we train/test or train/val/test
    # Ex: train only on synthetic, test on fpb + chronicle2050
    if X_test_df is not None:    
        X_test_list = X_test_df[embeddings_col_name].to_list()
    else:
        X_test_list = None
        
    print(f"\nTrain size: {len(X_train_list)}")
    print(f"Validation set: {len(X_val_list) if X_val_list is not None else 0}")
    print(f"Test size: {len(X_test_list) if X_test_list is not None else 0}\n")
    
    # Training
    for model_name, model in models.items():
        print(f"Training {model.get_model_name()}...")
    
        # Train model
        trained_model = model.train_model(X_train_list, y_train_list)
        
        # Compute train accuracy
        train_acc = trained_model.get_score(X_train_list, y_train_list)
        
        # Compute validation accuracy if data is provided
        if X_val_list is not None:
            val_acc = trained_model.get_score(X_val_list, y_val_list)
        
        # Store metrics
        train_val_metrics[model_name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
        print(f"Accuracy: {train_val_metrics[model_name]}")
        
        if X_test_list is not None:
            # Generate predictions
            model_predictions = trained_model.predict(X_test_list)
            trained_models_with_predictions[model_name] = (trained_model, model_predictions)
        else:
            # Still save the model so external datasets can evaluate it later
            trained_models_with_predictions[model_name] = trained_model
            
        checkpoint_file = f"model_checkpoint-{model_name}-{label_name}.pkl"
        checkpoint_path = os.path.join(model_checkpoint_path, checkpoint_file)
        joblib.dump(trained_model, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_file}")
        
    return trained_models_with_predictions, train_val_metrics
def create_results_dataframe(X_test_df, trained_models_with_predictions_dict):
    """Combine test data with model predictions."""
    print("\n" + "="*40)
    print("CREATE RESULTS DATAFRAME")
    print("="*40)
    
    results_df = X_test_df.copy()
    
    for model_name, models_and_predictions in trained_models_with_predictions_dict.items():
        _, predictions = models_and_predictions
        results_df[model_name] = predictions.to_list()
        print(f"✓ Added predictions: {model_name}")
    
    print(f"\nFinal shape: {results_df.shape}")
    print(f"\nPreview:\n{results_df.head(3)}\n")
    
    return results_df
def evaluate_models(
    trained_models_with_predictions_dict,
    X_test_df,
    y_test_df,
    embeddings_col_name,
    label_name,
    save_path,
    train_val_metrics_dict,
    seed
):
    """
    Evaluate all model predictions and save unified metrics.
    
    Parameters
    ----------
    predictions_dict : dict
        {model_name: (model, predictions_array)}
    y_test_df : pd.DataFrame
        True test labels
    label_name : str
        Name of label column
    save_path : str
        Directory path to save visualizations
    train_val_metrics_dict : dict
        {model_name: {'train_accuracy': float, 'val_accuracy': float}}
    
    Returns
    -------
    tuple
        (eval_reports_df, confusion_matrices, auc_scores)
    """
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Label: {label_name}\n")
    
    X_test_list = X_test_df[embeddings_col_name].to_list()
    actual_labels = y_test_df.values
    metrics_summary = []
    
    for model_name, model_and_predictions in trained_models_with_predictions_dict.items():
        print(f"### Model: {model_name} ###")
        
        # Get train/val metrics
        train_val_data = train_val_metrics_dict.get(model_name, {})   
        
        model, predictions = model_and_predictions
        # --- GET CONTINUOUS SCORES FOR AUC MATCHING ---
        if hasattr(model.classifer, "predict_proba"):
            continuous_scores = model.classifer.predict_proba(X_test_list)[:, 1]
        elif hasattr(model.classifer, "decision_function"):
            continuous_scores = model.classifer.decision_function(X_test_list)
        else:
            continuous_scores = predictions  # Fallback just in case
        # Classification report: precision, recall, f1, test accuracy
        eval_report = EvaluationMetric.eval_classification_report(actual_labels, predictions)
        # eval_reports[f"{label_name}-{model_name}"] = eval_report
        
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
        # POC Curve
        DataVisualizing.roc_curve(
            model_name,
            model,
            X_test_list, 
            y_test_df,
            save_path, 
            include_version=False
        )
        print(f"✓ Saved POC-CURVE: pos_curve{model_name}.png\n")
        # PR-AUC
        pr_auc_score = EvaluationMetric.get_pr_auc(actual_labels, continuous_scores)
        print(f"PR-AUC Score: {pr_auc_score:.4f}\n")
        # PR Curve
        DataVisualizing.pr_curve(
            model_name,
            model,
            X_test_list, 
            y_test_df,
            save_path, 
            include_version=False
        )
        print(f"✓ Saved PR-CURVE: pr_curve{model_name}.png\n")
             
        # Build unified metrics row
        metrics_row = {
            'model': model_name,
            # Get train & val metrics
            'train_accuracy': train_val_data.get('train_accuracy', None),
            'val_accuracy': train_val_data.get('val_accuracy', None),
            # Get classification report metrics
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
            # Get auc metrics
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
    
    return metrics_summary_df
def evaluate_and_save_results(
    trained_models_with_predictions_dict, 
    X_test_df, 
    y_test_df, 
    embeddings_col_name, 
    label_column, 
    output_dir, 
    metrics_folder_name,
    csv_prefix,
    train_val_metrics,
    seed
):
    """Run full evaluation pipeline and save results to disk."""
    print("\n" + "="*40)
    print(f"EVALUATION: {metrics_folder_name.upper()}")
    print("="*40)
    
    # Create save directory for this specific evaluation
    eval_save_path = os.path.join(output_dir, metrics_folder_name)
    os.makedirs(eval_save_path, exist_ok=True)
    
    # Save predictions dataframe
    models_predictions_df = create_results_dataframe(X_test_df, trained_models_with_predictions_dict)
    DataProcessing.save_to_file(
        models_predictions_df, 
        path=eval_save_path, 
        prefix=csv_prefix, 
        save_file_type='csv', 
        include_version=False,
        )
    print(f"✓ Saved results CSV to: {os.path.join(eval_save_path, f'{csv_prefix}.csv')}")
    
    # Generate and save metrics/visualizations
    models_metrics_df = evaluate_models(
        trained_models_with_predictions_dict, X_test_df, y_test_df, 
        embeddings_col_name, label_column, 
        eval_save_path, train_val_metrics, seed
    )
    DataProcessing.save_to_file(
        models_metrics_df, 
        path=eval_save_path, 
        prefix='ml_metrics_summary', 
        save_file_type='csv', 
        include_version=False,
        )
    print(f"✓ Saved metrics summary to: {os.path.join(eval_save_path, 'ml_metrics_summary.csv')}")
def evaluate_external_datasets(
    test_dataset_paths,
    trained_models_with_predictions_dict,
    text_column,
    label_column,
    output_dir,
    train_val_metrics,
    seed,
    script_dir
):
    """Handle loading, extracting, and evaluating all external cross-domain datasets."""
    print("\n" + "="*40)
    print("CROSS-DOMAIN TEST EVALUATION")
    print("="*40)
    print(f"External test datasets: {len(test_dataset_paths)}")
    
    for test_dataset_path in test_dataset_paths:
        # Extract dataset name from path
        test_dataset_name = os.path.splitext(os.path.basename(test_dataset_path))[0]
        metrics_folder_name = f'external_{test_dataset_name}'
        
        print(f"\n{'='*40}")
        print(f"Testing on: {test_dataset_name}")
        print(f"{'='*40}")
        
        # Load external test dataset
        external_test_df = load_dataset(script_dir, test_dataset_path)
        
        # Validate required columns
        if text_column not in external_test_df.columns:
            print(f"⚠️  Skipping {test_dataset_name}: missing '{text_column}' column")
            continue
        if label_column not in external_test_df.columns:
            print(f"⚠️  Skipping {test_dataset_name}: missing '{label_column}' column")
            continue
        
        # Extract embeddings for external test set
        external_embeddings_df, external_embeddings_col = extract_sentence_embeddings(
            external_test_df, text_column=text_column
        )
        
        # Prepare test data
        X_external_test_df = external_embeddings_df
        y_external_test = external_embeddings_df[[label_column]]
        
        # Generate predictions from all trained models
        external_predictions_dict = {}
        for model_name, model_or_tuple in trained_models_with_predictions_dict.items():
            # Handle both cases: model only or (model, predictions) tuple
            if isinstance(model_or_tuple, tuple):
                ml_model, _ = model_or_tuple
            else:
                ml_model = model_or_tuple
            
            X_test_list = X_external_test_df[external_embeddings_col].to_list()
            model_predictions = ml_model.predict(X_test_list)
            external_predictions_dict[model_name] = (ml_model, model_predictions)
        
        # Create output directory for this test dataset and save sets
        test_output_dir = os.path.join(output_dir, metrics_folder_name)
        os.makedirs(test_output_dir, exist_ok=True)
        DataProcessing.save_to_file(X_external_test_df, test_output_dir, 'x_y_test_set', 'csv', include_version=False)
        print(f"✓ Saved X_external_test_df to: {os.path.join(test_output_dir, 'x_y_test_set.csv')}")
        
        # Evaluate models on external test set
        evaluate_and_save_results(
            trained_models_with_predictions_dict=external_predictions_dict, 
            X_test_df=X_external_test_df, 
            y_test_df=y_external_test, 
            embeddings_col_name=external_embeddings_col, 
            label_column=label_column, 
            output_dir=output_dir, 
            metrics_folder_name=metrics_folder_name, 
            csv_prefix=f'ml_classifiers_{test_dataset_name}', 
            train_val_metrics=train_val_metrics,
            seed=seed
        )
def generate_all_explanations(
    trained_models_with_predictions_dict,
    X_train_df,
    embeddings_col_name,
    text_col_name,
    save_path
):
    """
    Generate SHAP and LIME explanations for trained models.
    """
    print("\n" + "="*40)
    print("MODEL EXPLAINABILITY")
    print("="*40)
    
    # Remove the old comparison file if it exists so we don't append to a previous run
    comparison_file = os.path.join(save_path, 'lime_comparison_all_models.html')
    if os.path.exists(comparison_file):
        os.remove(comparison_file)
    for model_name, models_and_predictions in trained_models_with_predictions_dict.items():
        print(f"\nExplaining {model_name}...")
        
        ml_model, _ = models_and_predictions
        # SHAP explainability
        Explainability.explain_model(
            X_train_df=X_train_df,
            embeddings_col_name=embeddings_col_name,
            ml_model=ml_model,
            model_name=model_name,
            save_path=save_path,
            include_version=False
        )
        
        # LIME explainability
        Explainability.explain_text_with_lime(
            X_train_df=X_train_df,
            text_col_name=text_col_name,
            embeddings_col_name=embeddings_col_name,
            ml_model=ml_model,
            model_name=model_name,
            save_path=save_path,
            num_samples=3,
            num_features=10 
        )
        
        # LIME explainability comparison (appending one by one)
        Explainability.add_to_lime_comparison(
            X_train_df=X_train_df,
            text_col_name=text_col_name,
            ml_model=ml_model,
            model_name=model_name,
            save_path=save_path,
            sample_idx=0,
            num_features=10,
            num_samples=50
        )
        
        print(f"  ✓ Explanations saved for {model_name}")
        
    print("\n✓ All model explanations complete\n")
if __name__ == "__main__":
    """
    usage: 
    # E1: Train on synthetic, test on FPB + C2050
    python train.py --dataset combined_datasets/combined-full_synthetic-v1.csv --no_test_split --val_size 0.2 \
                --test_datasets financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv chronicle2050/data.csv
    
    # E7: Standard split on combined dataset
    python train.py --dataset combined.csv --val_size 0.2
    
    # E1 with no validation set (use all synthetic for training)
    python train.py --dataset synthetic.csv --no_test_split \
                --test_datasets fpb.csv c2050.csv
    """

    print("\n" + "="*40)
    print("ML CLASSIFIER PIPELINE")
    print("="*40)
    
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    
    default_dataset = os.path.join(base_data_path, 'combined_datasets/combined-full_synthetic-v1.csv')
    default_save_path = os.path.join(base_data_path, 'classification_results/')
    
    parser = argparse.ArgumentParser(
        description='Train ML classifiers for prediction sentence classification'
    )
    
    parser.add_argument('--dataset', default=default_dataset, help='Path to dataset file')
    parser.add_argument('--save_path', default=default_save_path, help='Directory to save results')
    parser.add_argument('--dataset_type', default=None, 
                       choices=['synthetic_fin_phrasebank', 'synthetic', 'fin_phrasebank'],
                       help='Filter combined dataset by source')
    parser.add_argument('--text_column', default='Base Sentence', help='Text column name')
    parser.add_argument('--label_column', default='Sentence Label', help='Label column name')
    parser.add_argument('--explainability', action='store_true', help='Generate SHAP/LIME explanations')
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--val_size', type=float, default=None, help='Validation set size (0-1)')
    parser.add_argument('--no_test_split', action='store_true', 
                        help='Skip test set split from training data (for cross-domain evaluation).'
                        'Use with --test_datasets to evaluate only on external datasets that have ground truth.'
                        )
    parser.add_argument('--test_datasets', nargs='+', default=None,
                        help='Paths to external test datasets for cross-domain evaluation.'
                        'Models will be evaluated on each dataset separately.'
                        )
    
    args = parser.parse_args()
    
    # ============================================================
    # 2. EXPERIMENT SETUP
    # ============================================================
    current_date = datetime.now().strftime('%Y-%m-%d')
    dataset_filename = os.path.basename(args.dataset)
    dataset_base = os.path.splitext(dataset_filename)[0]
    
    if args.dataset_type and args.dataset_type != 'synthetic_fin_phrasebank':
        experiment_base = args.dataset_type
    else:
        experiment_base = dataset_base
    
    experiment_name = f"{experiment_base}_{current_date}"
    
    # Unpack the tuple directly into two variables
    experiment_dir, seed_dir = create_output_directory(args, experiment_name)
    
    print(f"\nExperiment: {experiment_name}")
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {seed_dir}\n")
    
    ml_model_names = [
        'perceptron',
        'sgd_classifier',
        'logistic_regression',
        'ridge_classifier',
        'decision_tree_classifier',
        'random_forest_classifier',
        'gradient_boosting_classifier',
        'support_vector_machine_classifier',
        'x_gradient_boosting_classifier'
    ]
    
    # ============================================================
    # 3. LOAD & PREPARE DATA
    # ============================================================
    df = load_dataset(script_dir, args.dataset)
    
    if args.text_column not in df.columns or args.label_column not in df.columns:
        print(f"\n❌ ERROR: Required columns not found")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    if args.dataset_type in ['synthetic_fin_phrasebank', 'synthetic', 'fin_phrasebank']:
        df = get_which_dataset(df, args.dataset_type)
    
    shuffled_df = shuffle_dataset(df, seed=args.seed)
    embeddings_df, embeddings_col_name = extract_sentence_embeddings(
        shuffled_df, text_column=args.text_column
    )
    # ============================================================
    # 4. SPLIT DATA
    # ============================================================
    # Determine split strategy
    if args.no_test_split:
        # For cross-domain evaluation (E1-E6): only train/val split
        print("\n" + "="*40)
        print("TRAIN/VAL SPLIT (No Test Set)")
        print("="*40)
        print("Using full dataset for training (will test on external datasets)")
        
        if args.val_size is not None:
            # Train/val split with specified val_size
            X_train_df, X_val_df, y_train_df, y_val_df = DataProcessing.split_data(
                embeddings_df, 
                embeddings_df[[args.label_column]],
                test_size=args.val_size,  # val_size used as test_size in this context
                val_size=None,
                random_state=args.seed,
                stratify_by=args.label_column
            )
        else:
            # No validation set: use all data for training
            X_train_df = embeddings_df
            y_train_df = embeddings_df[[args.label_column]]
            X_val_df = None
            y_val_df = None
        
        X_test_df = None
        y_test_df = None
        
        print(f"\nTrain size: {len(X_train_df)}")
        if X_val_df is not None:
            print(f"Val size: {len(X_val_df)}")
        print("Test size: 0 (will use external datasets)\n")
    else:
        # Standard split with test set (E7)
        if args.val_size is not None:
            # Train/val/test split
            X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df = split_train_test(
                embeddings_df, seed=args.seed, val_size=args.val_size, stratify_by=args.label_column
            )
        else:
            # Train/test split (no validation)
            X_train_df, X_test_df, y_train_df, y_test_df = split_train_test(
                embeddings_df, val_size=None, seed=args.seed, stratify_by=args.label_column
            )
            X_val_df = None
            y_val_df = None
    # Save test sets if they exist (for reproducibility)
    if X_test_df is not None and y_test_df is not None:
        print(X_test_df)
        DataProcessing.save_to_file(X_test_df, seed_dir, 'x_test_set', 'csv', include_version=False)
        print(f"✓ Saved X_test to: {os.path.join(seed_dir, 'x_test_set.csv')}")
        DataProcessing.save_to_file(y_test_df, seed_dir, 'y_test_set', 'csv', include_version=False)
        print(f"✓ Saved y_test to: {os.path.join(seed_dir, 'y_test_set.csv')}")
    # ============================================================
    # 4.5 SAVE SPLIT SIZES
    # ============================================================
    split_sizes = {
        'Split': ['Train', 'Validation', 'Test'],
        'Size': [
            len(X_train_df) if X_train_df is not None else 0,
            len(X_val_df) if X_val_df is not None else 0,
            len(X_test_df) if X_test_df is not None else 0
        ]
    }
    split_sizes_df = pd.DataFrame(split_sizes)
    
    print("Saving dataset split sizes...")
    DataProcessing.save_to_file(
        data=split_sizes_df,
        path=seed_dir,
        prefix='dataset_split_sizes',
        save_file_type='csv',
        include_version=False
    )
    # ============================================================
    # 5. TRAIN MODELS
    # ============================================================
    model_checkpoint_path = os.path.join(seed_dir, 'model_checkpoints')
    os.makedirs(model_checkpoint_path, exist_ok=True)
    # NOTE: within below, we check if val_df and test_df is None
    trained_models_with_predictions_dict, train_val_metrics = train_and_predict_models(
        ml_model_names, X_train_df, y_train_df, X_test_df,
        embeddings_col_name, args.label_column, model_checkpoint_path,
        seed=args.seed, X_val_df=X_val_df, y_val_df=y_val_df
    )
    # ============================================================
    # 6. EVALUATE ON IN-DOMAIN TEST SET (if exists)
    # ============================================================
    if X_test_df is not None and y_test_df is not None:
        evaluate_and_save_results(
            trained_models_with_predictions_dict=trained_models_with_predictions_dict, 
            X_test_df=X_test_df, 
            y_test_df=y_test_df, 
            embeddings_col_name=embeddings_col_name, 
            label_column=args.label_column, 
            output_dir=seed_dir, 
            metrics_folder_name='in_domain',
            csv_prefix='ml_classifiers_in_domain', 
            train_val_metrics=train_val_metrics,
            seed=args.seed
        )
    # ============================================================
    # 7. EVALUATE ON CROSS-DOMAIN TEST SETS (if provided)
    # ============================================================
    if args.test_datasets:
        evaluate_external_datasets(
            test_dataset_paths=args.test_datasets,
            trained_models_with_predictions_dict=trained_models_with_predictions_dict,
            text_column=args.text_column,
            label_column=args.label_column,
            output_dir=seed_dir,
            train_val_metrics=train_val_metrics,
            seed=args.seed,
            script_dir=script_dir
        )
    # ============================================================
    # 8. EXPLAINABILITY (OPTIONAL)
    # ============================================================
    if args.explainability:
        # Use in-domain test set if exists, otherwise use validation set
        explain_df = X_test_df if X_test_df is not None else X_val_df
        
        if explain_df is not None:
            generate_all_explanations(
                trained_models_with_predictions_dict=trained_models_with_predictions_dict,
                X_train_df=X_train_df,
                embeddings_col_name=embeddings_col_name,
                text_col_name=args.text_column,
                save_path=seed_dir
            )
    # ============================================================
    # 9. COMPLETE
    # ============================================================
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("="*40)
    print(f"Experiment: {experiment_name}")
    print(f"Training data: {experiment_base}")
    if args.test_datasets:
        print(f"External test sets: {len(args.test_datasets)}")
    print(f"\n✓ All outputs saved to: {experiment_dir}\n")