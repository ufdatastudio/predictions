import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

# Get script directory and add parent to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction

EMBEDDING_SIZES = {
    'spacy_small': 96,
    'spacy_medium': 300,
    'spacy_large': 300,
    'spacy_transformer': 768,
}

class RNN_Linear(nn.Module):
    """RNN using nn.Linear layers for sentence classification."""

    def __init__(self, input_embedding_size, hidden_size, output_size):
        super(RNN_Linear, self).__init__()
        self.input_embedding_size = input_embedding_size
        input_size = self.input_embedding_size.size()[0]
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.input_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, input_tensor, hidden_tensor):
        """
        Forward pass for one time step (one word).
        
        Parameters
        ----------
        input_tensor : torch.Tensor
            Current word embedding
        hidden_tensor : torch.Tensor
            Previous hidden state
        
        Returns
        -------
        tuple
            (hidden_t, output) - Updated hidden state and classification output
        """
        x_t = input_tensor
        h_t_1 = hidden_tensor
        i_h = torch.cat((x_t, h_t_1), dim=1)
        hidden_t = torch.tanh(self.input_to_hidden(i_h))
        y_hat = self.hidden_to_output(hidden_t)
        output = self.sigmoid(y_hat)
        return hidden_t, output

    def resize_hidden(self):
        """Initialize hidden state to zeros."""
        return torch.zeros(1, self.hidden_size)


def create_output_directory(args, experiment_name):
    """
    Create unique output directory with date and seed.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    experiment_name : str
        Name of experiment
    
    Returns
    -------
    tuple
        (experiment_dir, seed_dir) - Paths to experiment and seed directories
    """
    seed_number = f"seed{args.seed}"
    experiment_dir = os.path.join(args.save_path, experiment_name)
    seed_dir = os.path.join(experiment_dir, seed_number)
    os.makedirs(seed_dir, exist_ok=True)
    print(f"\n✓ Experiment directory: {experiment_dir}")
    print(f"✓ Seed directory: {seed_dir}")
    return experiment_dir, seed_dir


def load_dataset(script_dir, dataset_path):
    """
    Load dataset from file path.
    
    Parameters
    ----------
    script_dir : str
        Script directory
    dataset_path : str
        Relative path to dataset
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    print("\n" + "="*40)
    print("LOAD DATASET")
    print("="*40)
    
    if not os.path.isabs(dataset_path):
        data_path = os.path.join(script_dir, dataset_path)
    else:
        data_path = dataset_path
    
    print(f"Dataset path: {data_path}")
    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    
    return df


def load_and_preprocess_data(train_rel_path, test_rel_path, base_data_path=None, 
                             sample_size=None, embedding_model_name='spacy_large',
                             val_rel_path=None):
    """
    Load datasets and extract word embeddings.
    
    Parameters
    ----------
    train_rel_path : str
        Relative path from base_data_path to training CSV
    test_rel_path : str
        Relative path from base_data_path to test CSV
    base_data_path : str, optional
        Base data directory (default: script_dir/../data)
    train_sample_size : int, optional
        Number of training samples (None = use all)
    test_sample_size : int, optional
        Number of test samples (None = use all)
    embedding_model_name : str
        SpaCy embedding model name
    
    Returns
    -------
    tuple
        (train_embeddings_df, test_embeddings_df, train_df, test_df)
    """
    if base_data_path is None:
        base_data_path = DataProcessing.load_base_data_path(script_dir)
    
    print(f"Base data path: {base_data_path}")
    train_path = os.path.join(base_data_path, train_rel_path)
    test_path = os.path.join(base_data_path, test_rel_path)
    
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")
    print("\nLoading datasets...")
    
    train_df = DataProcessing.load_from_file(train_path)
    train_df['Ground Truth'] = train_df['Ground Truth'].astype(int)

    test_df = DataProcessing.load_from_file(test_path)
    test_df['Ground Truth'] = test_df['Ground Truth'].astype(int)
    
    if sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} training sentences")

        test_df = test_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} test sentences")
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
    print("\nTokenizing and embedding training data...")
    train_sfe = SpacyFeatureExtraction(train_df, 'Base Sentence', embedding_model_name=embedding_model_name)
    train_tokenized_df = train_sfe.split_words_in_sentence()
    train_embeddings_df = train_sfe.word_embeddings_extraction(
        tokenized_words_with_metadata_df=train_tokenized_df,
        reorder_cols=["Base Sentence", "Word", "Word Embedding", "Ground Truth"]
    )
    print(f"Training: {len(train_embeddings_df)} word embeddings extracted")
    
    print("\nTokenizing and embedding test data...")
    test_sfe = SpacyFeatureExtraction(test_df, 'Base Sentence', embedding_model_name=embedding_model_name)
    test_tokenized_df = test_sfe.split_words_in_sentence()
    test_embeddings_df = test_sfe.word_embeddings_extraction(
        tokenized_words_with_metadata_df=test_tokenized_df,
        reorder_cols=["Base Sentence", "Word", "Word Embedding", "Ground Truth"]
    )
    print(f"Test: {len(test_embeddings_df)} word embeddings extracted")

    val_embeddings_df, val_df = None, None
    if val_rel_path is not None:
        val_path = os.path.join(base_data_path, val_rel_path)
        print(f"Val path: {val_path}")
        val_df = DataProcessing.load_from_file(val_path)
        val_df['Ground Truth'] = val_df['Ground Truth'].astype(int)
        print(f"\nTokenizing and embedding val data...")
        val_sfe = SpacyFeatureExtraction(val_df, 'Base Sentence', embedding_model_name=embedding_model_name)
        val_tokenized_df = val_sfe.split_words_in_sentence()
        val_embeddings_df = val_sfe.word_embeddings_extraction(
            tokenized_words_with_metadata_df=val_tokenized_df,
            reorder_cols=["Base Sentence", "Word", "Word Embedding", "Ground Truth"]
        )
        print(f"Val: {len(val_embeddings_df)} word embeddings extracted")
    
    return train_embeddings_df, test_embeddings_df, val_embeddings_df, train_df, test_df, val_df

def train_step(classifier, sequence_of_embeddings, y, criterion, optimizer):
    """
    Train the RNN on a single sentence sequence.
    
    Parameters
    ----------
    classifier : nn.Module
        RNN model
    sequence_of_embeddings : list
        List of word embeddings for one sentence
    y : torch.Tensor
        Ground truth label (0 or 1)
    criterion : nn.Module
        Loss function (BCELoss)
    optimizer : torch.optim.Optimizer
        Optimizer (Adam or SGD)
    
    Returns
    -------
    tuple
        (final_output, loss_value) - Model prediction and loss
    """
    hidden = classifier.resize_hidden()
    
    for input_embedding_t in sequence_of_embeddings:
        x_embedding_t_reshaped = torch.tensor(input_embedding_t, dtype=torch.float32).unsqueeze(0)
        hidden, y_hat = classifier.forward(x_embedding_t_reshaped, hidden)
    
    final_output = y_hat
    loss = criterion(final_output, y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
    optimizer.step()
    
    return final_output, loss.item()


def train_model(train_embeddings_df, classifier, n_epochs, learning_rate, optimizer_name):
    print(f"\nTraining with {optimizer_name.upper()}, lr={learning_rate}, epochs={n_epochs}")
    
    criterion = nn.BCELoss()
    
    if optimizer_name.lower() == 'adam':
        # Adding weight decay (L2 Regularization) to prevent weights from exploding
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
        
    # NEW: Add a Learning Rate Scheduler. 
    # If the loss doesn't improve for 3 epochs, cut the learning rate in half.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    loss_history = []
    accumulation_steps = 32 # NEW: Simulate a batch size of 32
    
    for n_iter in range(n_epochs):
        text_document_sequences = []
        previous_text_document = None
        current_y_tensor = None
        
        current_loss = 0
        sentences_per_iteration = 0
        batch_loss_sum = 0 # NEW: Tracker for accumulated loss
        
        # Ensure gradients are zeroed at start of epoch
        optimizer.zero_grad() 
        
        for row in tqdm(train_embeddings_df.itertuples(index=False), 
                       total=len(train_embeddings_df),
                       desc=f"Epoch {n_iter+1}/{n_epochs}"):
            
            text_document = row._0  
            word_embedding = row._2 
            y = row._3  
            y_tensor = torch.as_tensor([[y]], dtype=torch.float)
            
            if text_document == previous_text_document:
                text_document_sequences.append(word_embedding)
            else:
                if len(text_document_sequences) > 0:
                    
                    # 1. Forward pass only
                    hidden = classifier.resize_hidden()
                    for input_embedding_t in text_document_sequences:
                        x = torch.tensor(input_embedding_t, dtype=torch.float32).unsqueeze(0)
                        hidden, y_hat = classifier.forward(x, hidden)
                    
                    # 2. Compute loss and scale it by accumulation steps
                    loss = criterion(y_hat, current_y_tensor)
                    scaled_loss = loss / accumulation_steps 
                    
                    # 3. Backward pass (accumulates gradients, but does NOT step yet)
                    scaled_loss.backward()
                    
                    current_loss += loss.item()
                    sentences_per_iteration += 1
                    
                    # 4. Step optimizer ONLY after 32 sentences
                    if sentences_per_iteration % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                text_document_sequences = [word_embedding]
                previous_text_document = text_document
                current_y_tensor = y_tensor
        
        # Process very last sequence in the dataset
        if len(text_document_sequences) > 0:
            hidden = classifier.resize_hidden()
            for input_embedding_t in text_document_sequences:
                x = torch.tensor(input_embedding_t, dtype=torch.float32).unsqueeze(0)
                hidden, y_hat = classifier.forward(x, hidden)
            
            loss = criterion(y_hat, current_y_tensor)
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            current_loss += loss.item()
            sentences_per_iteration += 1
            
            # Final step for any remaining accumulated gradients
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        if sentences_per_iteration > 0:
            avg_loss = current_loss / sentences_per_iteration
            loss_history.append(avg_loss)
            
            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {n_iter+1}/{n_epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            # NEW: Step the scheduler based on the epoch's average loss
            scheduler.step(avg_loss)
    
    return classifier, loss_history


def evaluate_model(embeddings_df, original_df, classifier, dataset_name="test"):
    """
    Evaluate trained RNN on data.
    
    Parameters
    ----------
    embeddings_df : pd.DataFrame
        Word-level data
    original_df : pd.DataFrame
        Original sentence-level data
    classifier : nn.Module
        Trained RNN model
    dataset_name : str
        Name of dataset for logging (e.g., 'test' or 'train')
        
    Returns
    -------
    tuple
        (final_df, y_preds, y_probs) - Dataframe with predictions and raw probabilities
    """
    print(f"\nEvaluating on {dataset_name} set...")
    
    text_document_sequences = []
    y_preds = []
    y_probs = []
    previous_text_document = None
    
    for row_idx in range(len(embeddings_df)):
        row = embeddings_df.iloc[row_idx]
        text_document = row['Base Sentence']
        word_embedding = row['Word Embedding']
        
        if text_document == previous_text_document:
            text_document_sequences.append(word_embedding)
        else:
            if len(text_document_sequences) > 0:
                with torch.no_grad():
                    hidden = classifier.resize_hidden()
                    for input_embedding_t in text_document_sequences:
                        x = torch.tensor(input_embedding_t, dtype=torch.float32).unsqueeze(0)
                        hidden, y_hat = classifier.forward(x, hidden)
                    
                    prob = y_hat.squeeze().item()
                    y_probs.append(prob)
                    y_preds.append(1 if prob > 0.5 else 0)
            
            text_document_sequences = [word_embedding]
            previous_text_document = text_document
    
    if len(text_document_sequences) > 0:
        with torch.no_grad():
            hidden = classifier.resize_hidden()
            for input_embedding_t in text_document_sequences:
                x = torch.tensor(input_embedding_t, dtype=torch.float32).unsqueeze(0)
                hidden, y_hat = classifier.forward(x, hidden)
            
            prob = y_hat.squeeze().item()
            y_probs.append(prob)
            y_preds.append(1 if prob > 0.5 else 0)
    
    print(f"Generated {len(y_preds)} predictions for {embeddings_df['Base Sentence'].nunique()} unique sentences")
    
    # Map predictions and probabilities to original df (handles duplicates)
    results_df = embeddings_df.groupby('Base Sentence', sort=False).first().reset_index()
    
    sentence_to_pred = dict(zip(results_df['Base Sentence'], y_preds))
    sentence_to_prob = dict(zip(results_df['Base Sentence'], y_probs))
    
    final_df = original_df.copy()
    final_df['RNN'] = final_df['Base Sentence'].map(sentence_to_pred)
    final_df['RNN_prob'] = final_df['Base Sentence'].map(sentence_to_prob)
    
    num_duplicates = len(original_df) - len(y_preds)
    print(f"Evaluating on {len(final_df)} sentences (including {num_duplicates} duplicates)")
    
    return final_df, y_preds, y_probs


def compute_metrics(test_final_df, loss_history, seed, train_accuracy=None, val_accuracy=None):
    """
    Compute classification metrics.
    """
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    
    y = test_final_df['Ground Truth'].values
    y_hat = test_final_df['RNN'].values
    y_prob = test_final_df['RNN_prob'].values  # Get continuous probabilities for AUC
    
    eval_report = EvaluationMetric.eval_classification_report(y, y_hat)
    confusion_mat, tn, fp, fn, tp = EvaluationMetric.get_confusion_matrix(y, y_hat, by_category=True)
    
    # Compute AUCs using continuous probabilities
    try:
        roc_auc = EvaluationMetric.get_roc_auc(y, y_prob)
        pr_auc = EvaluationMetric.get_pr_auc(y, y_prob)
    except Exception as e:
        print(f"Warning: Could not compute AUCs. {e}")
        roc_auc, pr_auc = np.nan, np.nan
    
    print(f"\nConfusion Matrix:\n{confusion_mat}\n")
    print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    
    metrics_row = {
        'seed': seed,
        'model': args.run_name,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'final_train_loss': loss_history[-1] if loss_history else None,
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
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    metrics_df = pd.DataFrame([metrics_row])
    print(f"\nMetrics Summary:\n{metrics_df}\n")
    
    return metrics_df


def create_experiment_log(args, experiment_name, seed_dir, loss_history):
    """
    Generate and save experiment log.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    experiment_name : str
        Experiment name
    seed_dir : str
        Seed directory path
    loss_history : list
        Training loss history
    """
    log_lines = []
    log_lines.append("="*40)
    log_lines.append("RNN EXPERIMENT LOG")
    log_lines.append("="*40)
    log_lines.append(f"Timestamp:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Experiment:        {experiment_name}")
    log_lines.append(f"Seed:              {args.seed}")
    log_lines.append("")
    log_lines.append("--- Model ---")
    log_lines.append(f"Architecture:      {args.run_name}")
    log_lines.append(f"Hidden Size:       {args.hidden_size}")
    log_lines.append(f"Epochs:            {args.n_epochs}")
    log_lines.append(f"Learning Rate:     {args.learning_rate}")
    log_lines.append(f"Optimizer:         {args.optimizer.upper()}")
    log_lines.append(f"Embedding Model:   {args.embedding_model}")
    log_lines.append("")
    log_lines.append("--- Data ---")
    log_lines.append(f"Train Path:        {args.train_path}")
    log_lines.append(f"val Path:          {args.val_path}")
    log_lines.append(f"Test Path:         {args.test_path}")
    log_lines.append(f"Sample Size:       {args.sample or 'All'}")
    log_lines.append("")
    log_lines.append("--- Training ---")
    if loss_history:
        log_lines.append(f"Initial Loss:      {loss_history[0]:.4f}")
        log_lines.append(f"Final Loss:        {loss_history[-1]:.4f}")
    log_lines.append("="*40)
    
    # log_dir = os.path.join(seed_dir, 'in_domain', 'experiment_log')
    log_dir = os.path.join(seed_dir, 'in_domain', args.embedding_model, args.run_name, 'experiment_log')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'experiment_log.txt')
    
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    
    print(f"✓ Experiment log saved to: {log_path}")


if __name__ == "__main__":
    print("\n" + "="*40)
    print("RNN CLASSIFIER PIPELINE")
    print("="*40)

    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_save_path = os.path.join(base_data_path, 'classification_results/')

    parser = argparse.ArgumentParser(description='Train RNN for sentence classification')

    # Data arguments
    parser.add_argument('--train_path', type=str,
                       default='classification_results/eacl_2026_results_2026-06-12/seed7/in_domain/spacy_small/x_y_train_set.csv',
                       help='Relative path from data/ to training CSV')
    parser.add_argument('--val_path', type=str, default=None,
                       help='Relative path from data/ to validation CSV')
    parser.add_argument('--test_path', type=str,
                       default='classification_results/eacl_2026_results_2026-06-12/seed7/in_domain/spacy_small/x_y_test_set.csv',
                       help='Relative path from data/ to test CSV')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of samples for train/val/test samples (default: use all)')
    parser.add_argument('--save_path', default=default_save_path,
                       help='Directory to save results')
    parser.add_argument('--experiment_name', default='eacl_2026_results_2026-06-07',
                       help='Existing experiment directory name to save results into')
    parser.add_argument('--run_name', type=str, default='rnn',
                       help='Subfolder name for this specific run to prevent overwriting')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden layer size (default: 128)')
    parser.add_argument('--embedding_model', default='spacy_large',
                       choices=['spacy_small', 'spacy_medium', 'spacy_large', 'spacy_transformer'],
                       help='SpaCy embedding model (default: spacy_large)')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer (default: adam)')
    parser.add_argument('--seed', type=int, default=3,
                       help='Random seed (default: 3)')

    args = parser.parse_args()

    # ============================================================
    # 1. CREATE OUTPUT DIRECTORY
    # ============================================================
    experiment_dir, seed_dir = create_output_directory(args, args.experiment_name)

    print(f"\nExperiment:        {experiment_dir}")
    print(f"Seed:              {args.seed}")
    print(f"Run Name:          {args.run_name}")
    print(f"Output directory:  {seed_dir}\n")

    # ============================================================
    # 2. LOAD & PREPROCESS DATA
    # ============================================================
    train_embeddings_df, test_embeddings_df, val_embeddings_df, train_df, test_df, val_df = load_and_preprocess_data(
        args.train_path, args.test_path, None,
        args.sample,
        args.embedding_model,
        val_rel_path=args.val_path
    )

    # ============================================================
    # 3. INITIALIZE MODEL
    # ============================================================
    print("\nInitializing RNN model...")
    input_embedding = torch.tensor(train_embeddings_df['Word Embedding'].iloc[0])
    rnn_classifier = RNN_Linear(input_embedding, args.hidden_size, output_size=1)
    print(f"Model: RNN_Linear(input_size={input_embedding.size()[0]}, hidden_size={args.hidden_size}, output_size=1)")

    # ============================================================
    # 4. TRAIN MODEL
    # ============================================================
    trained_model, loss_history = train_model(
        train_embeddings_df, rnn_classifier, args.n_epochs, args.learning_rate, args.optimizer
    )

    # ============================================================
    # 5. EVALUATE MODEL
    # ============================================================
    # 5a. Train accuracy
    train_final_df, _, _ = evaluate_model(train_embeddings_df, train_df, trained_model, dataset_name="train")
    y_train_true = train_final_df['Ground Truth'].values
    y_train_pred = train_final_df['RNN'].values
    train_eval_report = EvaluationMetric.eval_classification_report(y_train_true, y_train_pred)
    train_acc = train_eval_report.get('accuracy', None)

    # 5b. Validation accuracy (if val set provided)
    val_acc = np.nan
    if val_embeddings_df is not None and val_df is not None:
        val_final_df, _, _ = evaluate_model(val_embeddings_df, val_df, trained_model, dataset_name="val")
        y_val_true = val_final_df['Ground Truth'].values
        y_val_pred = val_final_df['RNN'].values
        val_eval_report = EvaluationMetric.eval_classification_report(y_val_true, y_val_pred)
        val_acc = val_eval_report.get('accuracy', None)

    # 5c. Test accuracy + AUC metrics
    test_final_df, y_hats, _ = evaluate_model(test_embeddings_df, test_df, trained_model, dataset_name="test")

    # ============================================================
    # 6. COMPUTE METRICS
    # ============================================================
    metrics_df = compute_metrics(
        test_final_df, loss_history, args.seed,
        train_accuracy=train_acc,
        val_accuracy=val_acc
    )

    # ============================================================
    # 7. SAVE RESULTS
    # ============================================================
    in_domain_dir = os.path.join(seed_dir, 'in_domain', args.embedding_model, args.run_name)
    os.makedirs(in_domain_dir, exist_ok=True)

    # Save predictions
    DataProcessing.save_to_file(
        test_final_df, in_domain_dir, 'rnn_predictions', 'csv', include_version=False
    )
    print(f"✓ Saved predictions to: {os.path.join(in_domain_dir, 'rnn_predictions.csv')}")

    # Save metrics
    DataProcessing.save_to_file(
        metrics_df, in_domain_dir, 'metrics_summary_rnn', 'csv', include_version=False
    )
    print(f"✓ Saved metrics to: {os.path.join(in_domain_dir, 'metrics_summary_rnn.csv')}")

    # Save loss history
    loss_df = pd.DataFrame({'epoch': range(1, len(loss_history) + 1), 'loss': loss_history})
    DataProcessing.save_to_file(
        loss_df, in_domain_dir, 'training_losses', 'csv', include_version=False
    )
    print(f"✓ Saved training losses to: {os.path.join(in_domain_dir, 'training_losses.csv')}")

    # ============================================================
    # 8. SAVE EXPERIMENT LOG
    # ============================================================
    create_experiment_log(args, args.experiment_name, seed_dir, loss_history)

    # ============================================================
    # 9. COMPLETE
    # ============================================================
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("="*40)
    print(f"✓ All outputs saved to: {experiment_dir}\n")