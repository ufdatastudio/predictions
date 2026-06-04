# BEFORE: run python3 ml-train.py to generate splits

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Setup paths
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))

from metrics import EvaluationMetric
from data_processing import DataProcessing
from dl_models import PyTorchRNN


def load_dataset(script_dir, dataset_path):
    print("\n" + "=" * 40)
    print("LOAD DATASET")
    print("=" * 40)

    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(script_dir, dataset_path)

    df = DataProcessing.load_from_file(dataset_path, "csv", sep=",")

    print(f"Shape: {df.shape}")
    print(df.head(3))
    return df

def convert_embedding_column(df, col='Base Sentence Embedding'):
    cleaned_embeddings = []

    for i, x in enumerate(df[col]):
        try:
            if isinstance(x, str):
                x = np.fromstring(x.strip('[]'), sep=' ')
            else:
                x = np.array(x)

            if len(x) == 0:
                continue

            cleaned_embeddings.append(x.astype(np.float32))

        except Exception:
            continue

    # ✅ enforce consistent size
    first_dim = len(cleaned_embeddings[0])

    final_embeddings = []
    valid_indices = []

    for i, emb in enumerate(cleaned_embeddings):
        if len(emb) == first_dim:
            final_embeddings.append(emb)
            valid_indices.append(i)

    df = df.iloc[valid_indices].reset_index(drop=True)
    df[col] = final_embeddings

    return df


# =============================
# VALIDATE SEED
# =============================
def validate_seed_and_dir(train_path, test_path, args_seed):
    train_seed, test_seed, seed_dir = None, None, None

    current_path = ""
    for part in train_path.split(os.sep):
        current_path = os.path.join(current_path, part)
        if part.startswith("seed"):
            train_seed = int(part.replace("seed", ""))
            seed_dir = current_path
            break

    for part in test_path.split(os.sep):
        if part.startswith("seed"):
            test_seed = int(part.replace("seed", ""))
            break

    if train_seed != test_seed:
        raise ValueError(f"Seed mismatch: {train_seed} vs {test_seed}")

    if args_seed is not None and train_seed != args_seed:
        raise ValueError(f"Seed mismatch with args")

    return train_seed, seed_dir


if __name__ == "__main__":

    print("\n" + "="*40)
    print("DL CLASSIFIER PIPELINE")
    print("="*40)

    # ============================================================
    # 1. CONFIGURATION
    # ============================================================
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--test_dataset", required=True)

    parser.add_argument("--embedding_column", default="Base Sentence Embedding")
    parser.add_argument("--label_column", default="Ground Truth")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # ============================================================
    # 2. LOAD & PREPARE DATA
    # ============================================================
    train_df = load_dataset(script_dir, args.train_dataset)
    test_df  = load_dataset(script_dir, args.test_dataset)

    train_df = convert_embedding_column(train_df)
    test_df  = convert_embedding_column(test_df)

    # ✅ Extract clean arrays
    X_train_np = np.stack(train_df[args.embedding_column].values)
    y_train_np = train_df[args.label_column].values

    X_test_np = np.stack(test_df[args.embedding_column].values)
    y_test_np = test_df[args.label_column].values

    # ✅ Convert to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)

    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)

    # ============================================================
    # 3. MODEL
    # ============================================================
    input_tensor = X_train[0]
    model = PyTorchRNN(input_tensor, hidden_size=128, output_size=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # ============================================================
    # 4. TRAIN
    # ============================================================
    for epoch in range(args.epochs):
        total_loss = 0

        for i in range(X_train.size(0)):
            x = X_train[i].unsqueeze(0)
            h = model.resize_hidden().to(device)

            _, out = model(x, h)

            loss = criterion(out, y_train[i].unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ============================================================
    # 5. TEST
    # ============================================================
    predictions = []

    with torch.no_grad():
        for i in range(X_test.size(0)):
            x = X_test[i].unsqueeze(0)
            h = model.resize_hidden().to(device)
            
            _, out = model(x, h)

            pred = 1 if out.item() >= 0.5 else 0

            predictions.append(pred)


    # ============================================================
    # 6. EVALUATION
    # ============================================================
    print("\n=== EVALUATION ===")
    EvaluationMetric.eval_classification_report(y_test_np, predictions)

    print("\n✅ PIPELINE COMPLETE")