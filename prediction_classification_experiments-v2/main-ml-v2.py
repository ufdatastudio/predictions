import os
import sys
import warnings
import argparse
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Get the current working directory of the script
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from data_visualizing import DataVisualizing
from feature_extraction import SpacyFeatureExtraction
from classification_models import SkLearnModelFactory


pd.set_option('max_colwidth', 800)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')

def load_dataset(script_dir, dataset_path):
    """
    Load dataset from file path.
    
    Parameters
    ----------
    script_dir : str
        Current script directory
    dataset_path : str
        Relative or absolute path to dataset file
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    
    Notes
    -----
    Prints dataset shape and preview to terminal for verification.
    """
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)
    
    # Handle relative vs absolute paths
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
    """
    Filter combined dataset based on author type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'Author Type' column
    dataset_name : str
        One of: 'synthetic_fin_phrasebank', 'synthetic', 'fin_phrasebank'
    
    Returns
    -------
    pd.DataFrame
        Filtered dataset
    
    Notes
    -----
    This function is specifically for the combined synthetic + fin_phrasebank dataset.
    - 'synthetic_fin_phrasebank': Returns full dataset (both sources)
    - 'synthetic': Filters to Author Type == 0
    - 'fin_phrasebank': Filters to Author Type == 1
    
    Raises error if 'Author Type' column doesn't exist, as it's required for filtering.
    """
    print("\n" + "="*50)
    print("FILTER COMBINED DATASET")
    print("="*50)
    print(f"Dataset selection: {dataset_name}")
    
    # Check if 'Author Type' column exists
    if 'Author Type' not in df.columns:
        raise ValueError(
            f"'Author Type' column required for dataset filtering but not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"This filter only works with the combined synthetic + fin_phrasebank dataset."
        )
    
    # Show distribution before filtering
    print(f"\nAuthor Type distribution (before filtering):")
    print(df['Author Type'].value_counts())
    
    # Apply filtering based on dataset_name
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
    """
    Shuffle dataset rows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    
    Returns
    -------
    pd.DataFrame
        Shuffled dataset
    """
    print("\n" + "="*50)
    print("SHUFFLE DATASET")
    print("="*50)
    
    shuffled_df = DataProcessing.shuffle_df(df, random_state=seed)
    print(f"Shape: {shuffled_df.shape}")
    print(f"\nPreview:\n{shuffled_df.head(7)}\n")
    
    return shuffled_df

def extract_sentence_embeddings(df, text_column='Base Sentence'):
    """
    Extract sentence embeddings using SpaCy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with text column
    text_column : str
        Name of column containing sentences
    
    Returns
    -------
    pd.DataFrame
        Dataset with added embedding column
    str
        Name of embeddings column
    
    Notes
    -----
    Adds column '{text_column} Embedding' to dataframe
    """
    print("\n" + "="*50)
    print("EXTRACT SENTENCE EMBEDDINGS (SpaCy)")
    print("="*50)
    print(f"Using text column: '{text_column}'")
    
    spacy_fe = SpacyFeatureExtraction(df, text_column)
    embeddings_df = spacy_fe.sentence_embeddings_extraction(attach_to_df=True)
    
    embeddings_col_name = f'{text_column} Embedding'
    
    # Show sample embeddings
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

def split_train_test(df, embeddings_col_name, seed=42, stratify_by='Sentence Label'):
    """
    Split dataset into train and test sets with stratification.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with embeddings and labels
    embeddings_col_name : str
        Name of column containing embeddings
    stratify_by : str
        Column name to stratify split on
    
    Returns
    -------
    tuple
        (X_train_df, X_test_df, y_train_df, y_test_df)
    
    Notes
    -----
    Stratification preserves the original dataset ratio when splitting.
    Example: If 920/1000 are non-predictions, train/test maintain ~92% non-predictions.
    """
    print("\n" + "="*50)
    print("SPLIT TRAIN/TEST DATA")
    print("="*50)
    print(f"Stratifying by: {stratify_by}")
    
    cols_with_labels = df.loc[:, [stratify_by]]
    data_splits = DataProcessing.split_data(
        df, cols_with_labels, stratify=True, random_state=seed, stratify_by=stratify_by
    )
    
    X_train_df, X_test_df, y_train_df, y_test_df = data_splits
    
    # Print split statistics
    print("\n{:<25} {:>10}".format("Dataset", "Count"))
    print("-" * 37)
    print("{:<25} {:>10}".format("X_train", len(X_train_df)))
    print("{:<25} {:>10}".format("X_test", len(X_test_df)))
    print("{:<25} {:>10}".format("y_train", len(y_train_df)))
    print("{:<25} {:>10}".format("y_test", len(y_test_df)))
    print()
    
    return X_train_df, X_test_df, y_train_df, y_test_df

def save_test_sets(X_test_df, y_test_df, save_path):
    """
    Save test sets to disk for later use with LLMs.
    
    Parameters
    ----------
    X_test_df : pd.DataFrame
        Test features
    y_test_df : pd.DataFrame
        Test labels
    save_path : str
        Directory path to save files
    """
    print("\n" + "="*50)
    print("SAVE TEST SETS")
    print("="*50)
    
    DataProcessing.save_to_file(X_test_df, save_path, 'x_test_set', 'csv')
    DataProcessing.save_to_file(y_test_df, save_path, 'y_sentence_test_df', 'csv')
    
    print(f"✓ Saved X_test to: {os.path.join(save_path, 'x_test_set.csv')}")
    print(f"✓ Saved y_test to: {os.path.join(save_path, 'y_sentence_test_df.csv')}\n")

def build_models(factory, model_names, seed=42):
    """
    Initialize ML models from factory.
    
    Parameters
    ----------
    factory : class
        Model factory class (e.g., SkLearnModelFactory)
    model_names : list of str
        List of model names to instantiate
    
    Returns
    -------
    dict
        {model_name: model_instance}
    """
    models = {}
    for name in model_names:
        models[name] = factory.select_model(name, random_state=seed)
    return models

def train_and_predict_models(
    ml_model_names, X_train_df, y_train_df, X_test_df, 
    embeddings_col_name, label_name, model_checkpoint_path, seed
):
    """
    Train multiple ML models and generate predictions on test set.
    
    Parameters
    ----------
    ml_model_names : list of str
        Names of models to train
    X_train_df : pd.DataFrame
        Training features with embeddings
    y_train_df : pd.DataFrame
        Training labels
    X_test_df : pd.DataFrame
        Test features with embeddings
    embeddings_col_name : str
        Name of embeddings column
    label_name : str
        Name of label column being predicted
    model_checkpoint_path : str
        Directory to save trained models
    
    Returns
    -------
    dict
        {model_name: predictions_array}
    
    Notes
    -----
    Saves each trained model to disk as: model_checkpoint-{model_name}-{label_name}.pkl
    """
    print("\n" + "="*50)
    print("TRAIN & PREDICT MODELS")
    print("="*50)
    print(f"Label: {label_name}")
    print(f"Models: {len(ml_model_names)}")
    
    ml_models = build_models(SkLearnModelFactory, ml_model_names, seed=seed)
    
    X_train_list = X_train_df[embeddings_col_name].to_list()
    y_train_list = y_train_df.to_list()
    X_test_list = X_test_df[embeddings_col_name].to_list()
    
    print(f"\nTrain size: {len(X_train_list)}")
    print(f"Test size: {len(X_test_list)}\n")
    
    predictions = {}
    
    for model_name, ml_model in ml_models.items():
        print(f"Training {ml_model.get_model_name()}...")
        ml_model.train_model(X_train_list, y_train_list)
        
        ml_model_predictions = ml_model.predict(X_test_list)
        predictions[model_name] = ml_model_predictions
        
        # Save model checkpoint
        checkpoint_file = f"model_checkpoint-{model_name}-{label_name}.pkl"
        checkpoint_path = os.path.join(model_checkpoint_path, checkpoint_file)
        joblib.dump(ml_model, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_file}")
    
    print()
    return predictions

def create_results_dataframe(X_test_df, predictions_dict):
    """
    Combine test data with model predictions.
    
    Parameters
    ----------
    X_test_df : pd.DataFrame
        Test features
    predictions_dict : dict
        {model_name: predictions_array}
    
    Returns
    -------
    pd.DataFrame
        Test data with added prediction columns for each model
    """
    print("\n" + "="*50)
    print("CREATE RESULTS DATAFRAME")
    print("="*50)
    
    results_df = X_test_df.copy()
    
    for model_name, predictions in predictions_dict.items():
        results_df[model_name] = predictions.to_list()
        print(f"✓ Added predictions: {model_name}")
    
    print(f"\nFinal shape: {results_df.shape}")
    print(f"\nPreview:\n{results_df.head(3)}\n")
    
    return results_df

def evaluate_models(predictions_dict: dict, y_test_df: pd.DataFrame, label_name: str, save_path: str):
    """
    Evaluate all model predictions and generate classification reports.
    
    Parameters
    ----------
    predictions_dict : dict
        {model_name: predictions_array}
    y_test_df : pd.DataFrame
        True test labels
    label_name : str
        Name of label column
    save_path : str
        Directory path to save visualizations
    
    Returns
    -------
    tuple
        (eval_reports_df, confusion_matrices, auc_scores)
        - eval_reports_df: DataFrame with classification metrics
        - confusion_matrices: dict of {model_name: confusion_matrix_array}
        - auc_scores: dict of {model_name: auc_score}
    
    Notes
    -----
    Prints classification report for each model and saves confusion matrix visualizations.
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Label: {label_name}\n")
    
    get_metrics = EvaluationMetric()
    actual_labels = y_test_df.values
    
    eval_reports = {}
    confusion_matrices = {}
    auc_scores = {}
    
    for model_name, predictions in predictions_dict.items():
        print(f"### Model: {model_name} ###")
        
        # Classification report
        eval_report = get_metrics.eval_classification_report(actual_labels, predictions)
        eval_reports[f"{label_name}-{model_name}"] = eval_report
        
        # Confusion matrix
        confusion_mat = get_metrics.get_confusion_matrix(actual_labels, predictions)
        confusion_matrices[model_name] = confusion_mat
        print(f"Confusion Matrix:\n{confusion_mat}\n")
        
        # AUC score
        auc_score = get_metrics.get_auc(actual_labels, predictions)
        auc_scores[model_name] = auc_score
        print(f"AUC Score: {auc_score:.4f}\n")
        
        # Save confusion matrix visualization
        DataVisualizing.visualize_confusion_matrix(confusion_mat, model_name, save_path)
        print(f"✓ Saved confusion matrix visualization: confusion_matrix_{model_name}.png\n")
    
    eval_reports_df = pd.DataFrame(eval_reports)
    
    print("\n" + "="*50)
    print("METRICS SUMMARY (LaTeX)")
    print("="*50)
    print(eval_reports_df.to_latex())
    print()
    
    return eval_reports_df, confusion_matrices, auc_scores

if __name__ == "__main__":
    """
    Usage Examples:
    
    # Default dataset (combined synthetic + financial phrasebank)
    python3 ml_classifiers.py
    
    # Custom single file
    python3 ml_classifiers.py --dataset ../data/my_data.csv
    
    # Filter to synthetic only
    python3 ml_classifiers.py --dataset_type synthetic
    
    # Specify custom column names
    python3 ml_classifiers.py --text_column "Sentence" --label_column "Label"
    
    # Full example with custom dataset
    python3 ml_classifiers.py \
        --dataset ../data/my_data.csv \
        --text_column "Text" \
        --label_column "Class" \
        --save_path ../data/results/
    """
    
    print("\n" + "="*50)
    print("ML CLASSIFIER PIPELINE")
    print("="*50)
    
    # Parse arguments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, '../data')
    
    default_dataset = os.path.join(base_data_path, 'classification_results/combined-synthetic-fin_phrase_bank-v5.csv')
    default_save_path = os.path.join(base_data_path, 'classification_results/')
    
    parser = argparse.ArgumentParser(
        description='Train ML classifiers for prediction sentence classification'
    )
    
    parser.add_argument(
        '--dataset',
        default=default_dataset,
        help='Path to dataset file. Default: combined-synthetic-fin_phrase_bank-v5.csv'
    )
    parser.add_argument(
        '--save_path',
        default=default_save_path,
        help='Directory to save results and checkpoints. Default: ../data/classification_results/'
    )
    parser.add_argument(
        '--dataset_type',
        default=None,
        choices=['synthetic_fin_phrasebank', 'synthetic', 'fin_phrasebank'],
        help='Filter combined dataset by source. Only applies to combined synthetic + fin_phrasebank dataset. '
            'Options: synthetic_fin_phrasebank (both), synthetic (only synthetic), fin_phrasebank (only fin_phrasebank). '
            'Default: None (no filtering, use dataset as-is)'
    )
    parser.add_argument(
        '--text_column',
        default='Base Sentence',
        help='Name of column containing text to classify. Default: "Base Sentence"'
    )
    parser.add_argument(
        '--label_column',
        default='Sentence Label',
        help='Name of column containing classification labels. Default: "Sentence Label"'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility. Default: 42'
    )
    
    args = parser.parse_args()
    
    # Extract dataset name for organized output directory
    dataset_filename = os.path.basename(args.dataset)
    dataset_name = os.path.splitext(dataset_filename)[0]
    dataset_name = f"{dataset_name}_seed{args.seed}"
    
    # Add filter suffix to dataset name if filtering applied
    if args.dataset_type and args.dataset_type != 'synthetic_fin_phrasebank':
        dataset_name = f"{dataset_name}_{args.dataset_type}"
    
    # Create organized output directory
    output_dir = os.path.join(args.save_path, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}\n")
    
    # Load dataset
    print("="*50)
    print("DATASET LOADING")
    print("="*50)
    print(f"Loading dataset: {args.dataset}")
    
    df = load_dataset(script_dir, args.dataset)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    
    print(f"Dataset type filter: {args.dataset_type}")
    print(f"Text column: '{args.text_column}'")
    print(f"Label column: '{args.label_column}'")
    print(f"Save path: {output_dir}\n")
    
    # Define model names
    ml_model_names = [
        'perceptron',
        'sgd_classifier',
        'logistic_regression',
        'ridge_classifier',
        'decision_tree_classifier',
        'random_forest_classifier',
        'gradient_boosting_classifier',
        'x_gradient_boosting_classifier'
    ]
    
    # Validate columns exist
    if args.text_column not in df.columns:
        print(f"\n❌ ERROR: Text column '{args.text_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    if args.label_column not in df.columns:
        print(f"\n❌ ERROR: Label column '{args.label_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Execute pipeline
    # Only apply filtering if dataset_type is specified AND is one of the known options
    if args.dataset_type in ['synthetic_fin_phrasebank', 'synthetic', 'fin_phrasebank']:
        print(f"Applying filter: {args.dataset_type}")
        df = get_which_dataset(df, args.dataset_type)
    else:
        print("No dataset filtering applied - using loaded dataset as-is")
        print(f"Dataset shape: {df.shape}\n")
    
    shuffled_df = shuffle_dataset(df, seed=args.seed)
    
    embeddings_df, embeddings_col_name = extract_sentence_embeddings(
        shuffled_df, text_column=args.text_column
    )
    
    X_train_df, X_test_df, y_train_df, y_test_df = split_train_test(
        embeddings_df, embeddings_col_name, seed=args.seed, stratify_by=args.label_column
    )
    
    save_test_sets(X_test_df, y_test_df, output_dir)
    
    # Create model checkpoint directory inside output directory
    model_checkpoint_path = os.path.join(output_dir, 'model_checkpoints')
    os.makedirs(model_checkpoint_path, exist_ok=True)
    
    predictions = train_and_predict_models(
        ml_model_names, X_train_df, y_train_df, X_test_df,
        embeddings_col_name, args.label_column, model_checkpoint_path, seed=args.seed
    )
    
    results_df = create_results_dataframe(X_test_df, predictions)
    
    DataProcessing.save_to_file(results_df, output_dir, 'ml_classifiers', '.csv')
    print(f"✓ Saved results to: {os.path.join(output_dir, 'ml_classifiers.csv')}")
    
    # Evaluate models and save confusion matrices
    eval_df, confusion_matrices, auc_scores = evaluate_models(
        predictions, y_test_df, args.label_column, output_dir
    )
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"Results shape: {results_df.shape}")
    print(f"Models evaluated: {len(predictions)}")
    print(f"\n✓ All outputs saved to: {output_dir}\n")