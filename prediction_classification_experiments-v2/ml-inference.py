# inference.py
import os
import sys
import joblib
import argparse
import pandas as pd
from datetime import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction

def validate_paths(dataset_path, experiment_path, model_checkpoint_path):
    """Check all paths exist before processing."""
    print("\n" + "="*50)
    print("VALIDATING PATHS")
    print("="*50)
    errors = []
    if not os.path.exists(dataset_path):
        errors.append(f"Dataset not found: {dataset_path}")
    else:
        print(f"✓ Dataset found: {dataset_path}")
    if not os.path.exists(experiment_path):
        errors.append(f"Experiment folder not found: {experiment_path}")
    else:
        print(f"✓ Experiment found: {experiment_path}")
    if not os.path.exists(model_checkpoint_path):
        errors.append(f"Model checkpoints not found: {model_checkpoint_path}")
    else:
        pkl_files = [f for f in os.listdir(model_checkpoint_path) if f.endswith('.pkl')]
        if not pkl_files:
            errors.append(f"No model checkpoints (.pkl) found in: {model_checkpoint_path}")
        else:
            print(f"✓ Model checkpoints found: {len(pkl_files)} models")
    if errors:
        print("\n❌ VALIDATION FAILED:\n")
        for error in errors:
            print(f"  - {error}")
        print()
        sys.exit(1)
    print("✓ All paths validated\n")

def load_dataset(dataset_path):
    """Load dataset from file path."""
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)
    print(f"Dataset path: {dataset_path}")
    is_sentiment140 = 'sentiment140' in dataset_path.lower()
    if is_sentiment140:
        print("Detected sentiment140 dataset - applying column names...")
        df = DataProcessing.load_from_file(
            dataset_path,
            'csv',
            sep=',',
            encoding='latin-1',
            header=None,
            names=['target', 'ids', 'date', 'query', 'user', 'text']
        )
    else:
        try:
            df = DataProcessing.load_from_file(dataset_path, 'csv', sep=',')
        except UnicodeDecodeError:
            print("⚠️  UTF-8 failed, trying latin-1 encoding...")
            df = DataProcessing.load_from_file(dataset_path, 'csv', sep=',', encoding='latin-1')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nPreview:\n{df.head(3)}\n")
    return df

def validate_text_column(df, text_column):
    """Validate text column exists, prompt user if not."""
    if text_column in df.columns:
        return text_column
    print(f"\n⚠️  WARNING: Text column '{text_column}' not found")
    print(f"Available columns: {list(df.columns)}")
    while True:
        user_input = input("\nEnter the correct text column name (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            print("Exiting...")
            sys.exit(0)
        if user_input in df.columns:
            print(f"✓ Using column: '{user_input}'")
            return user_input
        else:
            print(f"❌ Column '{user_input}' not found. Try again.")

def extract_sentence_embeddings(df, text_column):
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
    print(f"\n✓ Embeddings extracted: {embeddings_df.shape}\n")
    return embeddings_df, embeddings_col_name

def load_models(model_checkpoint_path):
    """Load all trained models from checkpoint directory."""
    print("\n" + "="*50)
    print("LOAD MODELS")
    print("="*50)
    print(f"Checkpoint path: {model_checkpoint_path}\n")
    ml_models = {}
    for filename in sorted(os.listdir(model_checkpoint_path)):
        if filename.endswith('.pkl') and filename.startswith('model_checkpoint'):
            filepath = os.path.join(model_checkpoint_path, filename)
            parts = filename.replace('model_checkpoint-', '').replace('.pkl', '').split('-')
            model_name = parts[0] if parts else filename
            print(f"Loading: {model_name}")
            ml_model = joblib.load(filepath)
            ml_models[model_name] = ml_model
    print(f"\n✓ Loaded {len(ml_models)} models\n")
    return ml_models

def run_inference_with_checkpoints(ml_models, embeddings_df, embeddings_col_name, save_dir, output_name):
    """Run inference with checkpointing to resume if interrupted."""
    print("\n" + "="*50)
    print("RUN INFERENCE (with checkpoints)")
    print("="*50)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file = os.path.join(save_dir, f'checkpoint_{output_name}.csv')
    embeddings_checkpoint = os.path.join(save_dir, f'embeddings_checkpoint_{output_name}.pkl')
    start_model_idx = 0
    results_df = None
    if os.path.exists(checkpoint_file):
        print(f"\n⚠️  Found checkpoint file: {checkpoint_file}")
        resume = input("Resume from checkpoint? (y/n): ").strip().lower()
        if resume == 'y':
            print("Loading checkpoint...")
            results_df = pd.read_csv(checkpoint_file)
            completed_models = [col for col in results_df.columns if col in ml_models.keys()]
            start_model_idx = len(completed_models)
            print(f"✓ Resuming from model index {start_model_idx}")
            print(f"  Completed models: {completed_models}")
        else:
            print("Starting fresh...")
            if os.path.exists(embeddings_checkpoint):
                os.remove(embeddings_checkpoint)
    if results_df is None:
        results_df = embeddings_df.copy()
    X_embeddings = embeddings_df[embeddings_col_name].to_list()
    model_names = list(ml_models.keys())
    total_models = len(model_names)
    for idx, model_name in enumerate(model_names):
        if idx < start_model_idx:
            print(f"Skipping {model_name} (already completed)")
            continue
        print(f"\n[{idx+1}/{total_models}] Predicting: {model_name}")
        ml_model = ml_models[model_name]
        predictions = ml_model.predict(X_embeddings)
        results_df[model_name] = predictions.to_list()
        results_df.to_csv(checkpoint_file, index=False)
        print(f"  ✓ Checkpoint saved ({idx+1}/{total_models} models complete)")
    print(f"\n✓ Inference complete for all {total_models} models\n")
    return results_df, checkpoint_file

if __name__ == "__main__":
    print("\n" + "="*50)
    print("ML INFERENCE PIPELINE")
    print("="*50)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, '../data')
    parser = argparse.ArgumentParser(
        description='Run inference with trained ML classifiers'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset filename (e.g., sentiment140/data.csv) - relative to data/ folder'
    )
    parser.add_argument(
        '--experiment',
        required=True,
        help='Experiment folder name (e.g., combined-full_synthetic-v1_2026-03-07)'
    )
    parser.add_argument(
        '--seed',
        required=True,
        help='Seed number to load models from (e.g., 40)'
    )
    parser.add_argument(
        '--text_column',
        default='Base Sentence',
        help='Text column name (default: Base Sentence)'
    )
    parser.add_argument(
        '--output_name',
        required=True,
        help='Name for output file (e.g., chronicle2050, sentiment140, fpb_imbalanced)'
    )
    args = parser.parse_args()
    dataset_path = os.path.join(base_data_path, args.dataset)
    results_dir = os.path.join(base_data_path, 'classification_results')
    experiment_path = os.path.join(results_dir, args.experiment)
    seed_path = os.path.join(experiment_path, f'seed{args.seed}')
    model_checkpoint_path = os.path.join(seed_path, 'model_checkpoints')
    save_dir = os.path.join(seed_path, 'inference')

    print(f"\nExperiment: {args.experiment}")
    print(f"Seed: {args.seed}")
    print(f"Dataset: {args.dataset}")
    print(f"Output name: {args.output_name}")
    print(f"Models path: {model_checkpoint_path}")
    print(f"Save path: {save_dir}\n")

    validate_paths(dataset_path, experiment_path, model_checkpoint_path)
    
    df = load_dataset(dataset_path)
    
    text_column = validate_text_column(df, args.text_column)
    
    embeddings_df, embeddings_col_name = extract_sentence_embeddings(df, text_column)
    
    ml_models = load_models(model_checkpoint_path)
    
    results_df, checkpoint_file = run_inference_with_checkpoints(
        ml_models, embeddings_df, embeddings_col_name, save_dir, args.output_name
    )
    
    results_df['Dataset'] = args.output_name
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(save_dir, f'inference_{args.output_name}_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Removed checkpoint file")
    
    print(f"\n✓ Saved final results to: {results_file}")
    print(f"  Shape: {results_df.shape}")
    print(f"  Columns: {list(results_df.columns)}\n")