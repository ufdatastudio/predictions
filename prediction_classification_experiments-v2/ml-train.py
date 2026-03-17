# Before this, run python3 create_combined_dataset.py to create dataset
import os
import sys
import joblib
import argparse
import matplotlib
matplotlib.use('Agg')  # Prevent GUI windows from opening
import warnings
warnings.filterwarnings("ignore")

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
    """Create unique output directory with timestamp."""
    timestamp = datetime.now().strftime('%H-%M-%S') # get seconds so we can distinguish runs and seed #s
    experiment_folder = f"{experiment_name}_{timestamp}"
    seed_folder = f"seed{args.seed}"
    output_dir = os.path.join(args.save_path, experiment_folder, seed_folder)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n✓ Output directory: {output_dir}")
    return output_dir

def load_dataset(script_dir, dataset_path):
    """Load dataset from file path."""
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)
    
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
    print("\n" + "="*50)
    print("FILTER COMBINED DATASET")
    print("="*50)
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
    print("\n" + "="*50)
    print("SHUFFLE DATASET")
    print("="*50)
    
    shuffled_df = DataProcessing.shuffle_df(df, random_state=seed)
    print(f"Shape: {shuffled_df.shape}")
    print(f"\nPreview:\n{shuffled_df.head(7)}\n")
    
    return shuffled_df

def extract_sentence_embeddings(df, text_column='Base Sentence'):
    """Extract sentence embeddings using SpaCy."""
    print("\n" + "="*50)
    print("EXTRACT SENTENCE EMBEDDINGS (SpaCy)")
    print("="*50)
    print(f"Using text column: '{text_column}'")
    
    spacy_fe = SpacyFeatureExtraction(df, text_column)
    embeddings_df = spacy_fe.sentence_embeddings_extraction(attach_to_df=True)
    
    embeddings_col_name = f'{text_column} Embedding'
    
    for idx, row in embeddings_df.iterrows():
        if idx < 3:
            text = row[text_column]
            embedding = row[embeddings_col_name]
            print(f"\nSample {idx}:")
            print(f"  Sentence [:100]: {text[:100]}...")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding subset [:6]: {embedding[:6]}")
    
    print(f"\n✓ Embeddings extracted: {embeddings_df.shape}\n")
    
    return embeddings_df, embeddings_col_name

def split_train_test(df, seed, val_size=None, stratify_by='Sentence Label'):
    """Split dataset into train/test or train/val/test sets."""
    cols_with_labels = df.loc[:, [stratify_by]]
    
    if val_size is None:
        print("\n" + "="*50)
        print("SPLIT TRAIN/TEST DATA")
        print("="*50)
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
        print("\n" + "="*50)
        print("SPLIT TRAIN/VAL/TEST DATA")
        print("="*50)
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

def save_test_sets(X_test_df, y_test_df, save_path):
    """Save test sets to disk for later use."""
    print("\n" + "="*50)
    print("SAVE TEST SETS")
    print("="*50)
    
    DataProcessing.save_to_file(X_test_df, save_path, 'x_test_set', 'csv', include_version=False)
    DataProcessing.save_to_file(y_test_df, save_path, 'y_sentence_test_df', 'csv', include_version=False)
    
    print(f"✓ Saved X_test to: {os.path.join(save_path, 'x_test_set.csv')}")
    print(f"✓ Saved y_test to: {os.path.join(save_path, 'y_sentence_test_df.csv')}\n")

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
    print("\n" + "="*50)
    print("TRAIN & PREDICT MODELS")
    print("="*50)
    print(f"Label: {label_name}")
    print(f"Models: {len(ml_model_names)}")
    
    ml_models = build_models(SkLearnModelFactory, ml_model_names, seed=seed)
    
    X_train_list = X_train_df[embeddings_col_name].to_list()    
    y_train_list = y_train_df.values.ravel()
    X_test_list = X_test_df[embeddings_col_name].to_list()
    
    # Prepare validation data if provided
    X_val_list = None
    y_val_list = None
    if X_val_df is not None and y_val_df is not None:
        X_val_list = X_val_df[embeddings_col_name].to_list()
        y_val_list = y_val_df.values.ravel()
        print(f"Validation set: {len(X_val_list)} samples")
    
    print(f"\nTrain size: {len(X_train_list)}")
    print(f"Test size: {len(X_test_list)}\n")
    
    predictions = {}
    train_val_metrics = {}
    
    for model_name, ml_model in ml_models.items():
        print(f"Training {ml_model.get_model_name()}...")
        
        # Train model
        ml_model.train_model(X_train_list, y_train_list)
        
        # Compute train/val accuracy
        train_acc = ml_model.get_score(X_train_list, y_train_list)
        val_acc = None
        
        if X_val_list is not None:
            val_acc = ml_model.get_score(X_val_list, y_val_list)
            print(f"  Train accuracy: {train_acc:.4f}")
            print(f"  Val accuracy: {val_acc:.4f}")
        else:
            print(f"  Train accuracy: {train_acc:.4f}")
        
        # Store metrics
        train_val_metrics[model_name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
        
        # Generate predictions
        ml_model_predictions = ml_model.predict(X_test_list)
        predictions[model_name] = (ml_model, ml_model_predictions)
        
        # Save model checkpoint
        checkpoint_file = f"model_checkpoint-{model_name}-{label_name}.pkl"
        checkpoint_path = os.path.join(model_checkpoint_path, checkpoint_file)
        joblib.dump(ml_model, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_file}")
    
    print()
    return predictions, train_val_metrics

def create_results_dataframe(X_test_df, trained_models_with_predictions_dict):
    """Combine test data with model predictions."""
    print("\n" + "="*50)
    print("CREATE RESULTS DATAFRAME")
    print("="*50)
    
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
    train_val_metrics_dict
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
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Label: {label_name}\n")
    
    # x_test_labels = X_test_df[label_name]
    actual_labels = y_test_df.values
    # print(X_test_df[embeddings_col_name].to_list())


    eval_reports = {}
    confusion_matrices = {}
    roc_auc_scores = {}
    pr_auc_scores = {}
    metrics_summary = []
    
    for model_name, model_and_predictions in trained_models_with_predictions_dict.items():
        print(f"### Model: {model_name} ###")
        
        model, predictions = model_and_predictions
        # Classification report
        eval_report = EvaluationMetric.eval_classification_report(actual_labels, predictions)
        eval_reports[f"{label_name}-{model_name}"] = eval_report
        
        # Confusion matrix
        confusion_mat = EvaluationMetric.get_confusion_matrix(actual_labels, predictions)
        confusion_matrices[model_name] = confusion_mat
        print(f"Confusion Matrix:\n{confusion_mat}\n")
        
        # ROC-AUC score
        roc_auc_score = EvaluationMetric.get_roc_auc(actual_labels, predictions)
        roc_auc_scores[model_name] = roc_auc_score
        print(f"ROC-AUC Score: {roc_auc_score:.4f}\n")

        # PR-AUC
        pr_auc_score = EvaluationMetric.get_pr_auc(actual_labels, predictions)
        pr_auc_scores[model_name] = pr_auc_score
        print(f"PR-AUC Score: {pr_auc_score:.4f}\n")
        
        # Get train/val metrics
        train_val_data = train_val_metrics_dict.get(model_name, {})
        train_acc = train_val_data.get('train_accuracy', None)
        val_acc = train_val_data.get('val_accuracy', None)
        
        # Build unified metrics row
        metrics_row = {
            'model': model_name,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': eval_report.get('accuracy', None),
            'precision_class_0': eval_report.get('0', {}).get('precision', None),
            'precision_class_1': eval_report.get('1', {}).get('precision', None),
            'recall_class_0': eval_report.get('0', {}).get('recall', None),
            'recall_class_1': eval_report.get('1', {}).get('recall', None),
            'f1_class_0': eval_report.get('0', {}).get('f1-score', None),
            'f1_class_1': eval_report.get('1', {}).get('f1-score', None),
            'roc_auc': roc_auc_score,
            'pr_auc': pr_auc_score
        }
        metrics_summary.append(metrics_row)
        
        # Save confusion matrix visualization
        DataVisualizing.confusion_matrix(
            model_name,
            confusion_mat, 
            save_path, 
            include_version=False
        )
        print(f"✓ Saved confusion matrix: confusion_matrix_{model_name}.png\n")

        # POC Curve
        DataVisualizing.roc_curve(
            model_name,
            model,
            X_test_df[embeddings_col_name].to_list(), 
            y_test_df,
            save_path, 
            include_version=False
        )
        print(f"✓ Saved POC-CURVE: pos_curve{model_name}.png\n")
        
        # PR Curve
        DataVisualizing.pr_curve(
            model_name,
            model,
            X_test_df[embeddings_col_name].to_list(), 
            y_test_df,
            save_path, 
            include_version=False
        )
        print(f"✓ Saved PR-CURVE: pr_curve{model_name}.png\n")

    eval_reports_df = pd.DataFrame(eval_reports)
    
    # Save unified metrics summary
    metrics_summary_df = pd.DataFrame(metrics_summary)
    metrics_file = os.path.join(save_path, 'metrics_summary.csv')
    metrics_summary_df.to_csv(metrics_file, index=False)
    print(f"✓ Saved metrics summary to: {metrics_file}")
    
    print("\n" + "="*50)
    print("METRICS SUMMARY (LaTeX)")
    print("="*50)
    print(eval_reports_df.to_latex())
    print()
    
    return eval_reports_df, confusion_matrices, roc_auc_scores, pr_auc_scores

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
    print("\n" + "="*50)
    print("MODEL EXPLAINABILITY")
    print("="*50)
    
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
    # E1: Train on synthetic, test on FPB + C2050
    python train.py --dataset combined_datasets/combined-full_synthetic-v1.csv --no_test_split --val_size 0.2 \
                --test_datasets financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv chronicle2050/data.csv
    
    # E7: Standard split on combined dataset
    python train.py --dataset combined.csv --val_size 0.2
    
    # E1 with no validation set (use all synthetic for training)
    python train.py --dataset synthetic.csv --no_test_split \
                --test_datasets fpb.csv c2050.csv
    """


    print("\n" + "="*50)
    print("ML CLASSIFIER PIPELINE")
    print("="*50)
    
    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, '../data')
    
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
    output_dir = create_output_directory(args, experiment_name)
    
    print(f"\nExperiment: {experiment_name}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {output_dir}\n")
    
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
        print("\n" + "="*50)
        print("TRAIN/VAL SPLIT (No Test Set)")
        print("="*50)
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
        save_test_sets(X_test_df, y_test_df, output_dir)

    # ============================================================
    # 5. TRAIN MODELS
    # ============================================================
    model_checkpoint_path = os.path.join(output_dir, 'model_checkpoints')
    os.makedirs(model_checkpoint_path, exist_ok=True)

    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)

    # Train models (predictions on in-domain test if exists, otherwise None)
    if X_test_df is not None:
        trained_models_with_predictions_dict, train_val_metrics = train_and_predict_models(
            ml_model_names, X_train_df, y_train_df, X_test_df,
            embeddings_col_name, args.label_column, model_checkpoint_path,
            seed=args.seed, X_val_df=X_val_df, y_val_df=y_val_df
        )
    else:
        # No in-domain test set, just train models without predictions
        # We'll generate predictions later on external datasets
        trained_models_dict = {}
        train_val_metrics = {}
        
        ml_models = build_models(SkLearnModelFactory, ml_model_names, seed=args.seed)
        X_train_list = X_train_df[embeddings_col_name].to_list()
        y_train_list = y_train_df.values.ravel()
        
        if X_val_df is not None:
            X_val_list = X_val_df[embeddings_col_name].to_list()
            y_val_list = y_val_df.values.ravel()
        
        for model_name, ml_model in ml_models.items():
            print(f"Training {ml_model.get_model_name()}...")
            ml_model.train_model(X_train_list, y_train_list)
            
            train_acc = ml_model.get_score(X_train_list, y_train_list)
            val_acc = None
            if X_val_df is not None:
                val_acc = ml_model.get_score(X_val_list, y_val_list)
                print(f"  Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}")
            else:
                print(f"  Train accuracy: {train_acc:.4f}")
            
            train_val_metrics[model_name] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            }
            
            trained_models_dict[model_name] = ml_model
            
            # Save checkpoint
            checkpoint_file = f"model_checkpoint-{model_name}-{args.label_column}.pkl"
            checkpoint_path = os.path.join(model_checkpoint_path, checkpoint_file)
            joblib.dump(ml_model, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_file}")
        
        trained_models_with_predictions_dict = trained_models_dict

    # ============================================================
    # 6. EVALUATE ON IN-DOMAIN TEST SET (if exists)
    # ============================================================
    if X_test_df is not None and y_test_df is not None:
        print("\n" + "="*50)
        print("IN-DOMAIN TEST EVALUATION")
        print("="*50)
        
        results_df = create_results_dataframe(X_test_df, trained_models_with_predictions_dict)
        results_file = os.path.join(output_dir, 'ml_classifiers_in_domain.csv')
        results_df.to_csv(results_file, index=False)
        print(f"✓ Saved in-domain results to: {results_file}")
        
        eval_df, confusion_matrices, roc_auc_scores, pr_auc_scores = evaluate_models(
            trained_models_with_predictions_dict, X_test_df, y_test_df, 
            embeddings_col_name, args.label_column, 
            os.path.join(output_dir, 'in_domain'), train_val_metrics
        )

    # ============================================================
    # 7. EVALUATE ON CROSS-DOMAIN TEST SETS (if provided)
    # ============================================================
    if args.test_datasets:
        print("\n" + "="*50)
        print("CROSS-DOMAIN TEST EVALUATION")
        print("="*50)
        print(f"External test datasets: {len(args.test_datasets)}")
        
        for test_dataset_path in args.test_datasets:
            # Extract dataset name from path
            test_dataset_name = os.path.splitext(os.path.basename(test_dataset_path))[0]
            
            print(f"\n{'='*50}")
            print(f"Testing on: {test_dataset_name}")
            print(f"{'='*50}")
            
            # Load external test dataset
            external_test_df = load_dataset(script_dir, test_dataset_path)
            
            # Validate required columns
            if args.text_column not in external_test_df.columns:
                print(f"⚠️  Skipping {test_dataset_name}: missing '{args.text_column}' column")
                continue
            if args.label_column not in external_test_df.columns:
                print(f"⚠️  Skipping {test_dataset_name}: missing '{args.label_column}' column")
                continue
            
            # Extract embeddings for external test set
            external_embeddings_df, external_embeddings_col = extract_sentence_embeddings(
                external_test_df, text_column=args.text_column
            )
            
            # Prepare test data
            X_external_test = external_embeddings_df
            y_external_test = external_embeddings_df[[args.label_column]]
            
            # Generate predictions from all trained models
            external_predictions_dict = {}
            
            for model_name, model_or_tuple in trained_models_with_predictions_dict.items():
                # Handle both cases: model only or (model, predictions) tuple
                if isinstance(model_or_tuple, tuple):
                    ml_model, _ = model_or_tuple
                else:
                    ml_model = model_or_tuple
                
                X_test_list = X_external_test[external_embeddings_col].to_list()
                predictions = ml_model.predict(X_test_list)
                external_predictions_dict[model_name] = (ml_model, predictions)
            
            # Create output directory for this test dataset
            test_output_dir = os.path.join(output_dir, f'external_{test_dataset_name}')
            os.makedirs(test_output_dir, exist_ok=True)
            
            # Save external test sets
            save_test_sets(X_external_test, y_external_test, test_output_dir)
            
            # Evaluate models on external test set
            results_df = create_results_dataframe(X_external_test, external_predictions_dict)
            results_file = os.path.join(test_output_dir, f'ml_classifiers_{test_dataset_name}.csv')
            results_df.to_csv(results_file, index=False)
            print(f"✓ Saved {test_dataset_name} results to: {results_file}")
            
            eval_df, confusion_matrices, roc_auc_scores, pr_auc_scores = evaluate_models(
                external_predictions_dict, X_external_test, y_external_test,
                external_embeddings_col, args.label_column, test_output_dir, 
                train_val_metrics  # Use same train/val metrics for context
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
                save_path=output_dir
            )

    # ============================================================
    # 9. COMPLETE
    # ============================================================
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"Experiment: {experiment_name}")
    print(f"Training data: {experiment_base}")
    if args.test_datasets:
        print(f"External test sets: {len(args.test_datasets)}")
    print(f"\n✓ All outputs saved to: {output_dir}\n")