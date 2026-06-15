import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import hashlib
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
    classification_report
)


script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../'))

from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction
from classification_models import SkLearnModelFactory


TEXT_COLUMN  = 'Sentence'
LABEL_COLUMN = 'Human Annotation'

# Models that do NOT support predict_proba or decision_function
# — AUC scores will be set to N/A for these
NO_AUC_MODELS = {
    'perceptron',
    'ridge_classifier',
    'linear_regression',
    'elastic_net'
}

# Models that do NOT accept random_state
NO_SEED_MODELS = {
    'linear_regression'
}


PARAM_GRIDS = {
    'perceptron': {
        'penalty':   [None, 'l2', 'l1', 'elasticnet'],
        'alpha':     [0.0001, 0.001, 0.01],
        'max_iter':  [500, 1000, 2000],
        'tol':       [1e-3, 1e-4],
    },
    'sgd_classifier': {
        'alpha':        [0.0001, 0.001, 0.01],
        'max_iter':     [500, 1000, 2000],
        'tol':          [1e-3, 1e-4],
        'penalty':      ['l2', 'l1', 'elasticnet'],
        'learning_rate':['optimal', 'adaptive'],
    },
    'logistic_regression': {
        'C':        [0.01, 0.1, 1, 10, 100],
        'penalty':  ['l1', 'l2'],
        'solver':   ['liblinear', 'saga'],
        'max_iter': [500, 1000, 2000],
    },
    'ridge_classifier': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'tol':   [1e-3, 1e-4],
    },
    'linear_regression': {
        'fit_intercept': [True, False],
        'positive':      [True, False],
    },
    'elastic_net': {
        'alpha':    [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [500, 1000, 2000],
    },
    #Lots of overfitting for decision trees
    #Come back to increase more regularization
    'decision_tree_classifier': {
        'max_depth':        [3, 5, 7, 10],
        'min_samples_split':[10, 20, 50],
        'min_samples_leaf': [4, 8, 16],
        #Try: Put more weight on label 1
        'class_weight':      [None, 'balanced'],
        'criterion':        ['gini', 'entropy'],
    },
    'random_forest_classifier': {
        'n_estimators':     [50, 100, 200],
        'max_depth':        [5, 10, 20],
        'min_samples_split':[10, 20,50],
        'min_samples_leaf': [4, 8,16],
        'class_weight':     [None, 'balanced'],
        'max_features':     ['sqrt', 'log2'],
    },
    #Take too long to run
    'gradient_boosting_classifier': {
        'n_estimators':  [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth':     [3, 5, 7],
        'subsample':     [0.8, 1.0],
    },
    'support_vector_machine_classifier': {
        'C':      [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma':  ['scale', 'auto'],
    },
    'x_gradient_boosting_classifier': {
        'n_estimators':  [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth':     [3, 5, 7],
        'subsample':     [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    },
}

ALL_MODEL_NAMES = list(PARAM_GRIDS.keys())


def load_and_prepare_data(dataset_path):
    """
    Load dataset from either:
      - A single CSV file path, or
      - A directory containing multiple CSV files (all combined into one DataFrame)

    Validates required columns, drops rows with missing labels, deduplicates
    sentences, and ensures the label column is integer typed.

    REQUIREMENT — data_processing.py:
        DataProcessing.load_from_file(path, file_type, sep) -> pd.DataFrame
    """
    print("\n" + "="*40)
    print("LOAD DATASET")
    print("="*40)

    # ---- Determine if input is a directory or a single file ----
    if os.path.isdir(dataset_path):
        csv_files = sorted([
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.endswith('.csv')
        ])

        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {dataset_path}")

        print(f"Directory mode  : found {len(csv_files)} CSV files")

        dfs = []
        skipped = 0
        for csv_file in csv_files:
            try:
                file_df = DataProcessing.load_from_file(csv_file, 'csv', sep=',')

                # Only keep files that have both required columns
                if TEXT_COLUMN not in file_df.columns or LABEL_COLUMN not in file_df.columns:
                    print(f"Skipping {os.path.basename(csv_file)} — missing required columns")
                    skipped += 1
                    continue

                dfs.append(file_df)
            except Exception as e:
                print(f"Skipping {os.path.basename(csv_file)} — error: {e}")
                skipped += 1

        if not dfs:
            raise ValueError(
                f"No valid CSV files could be loaded from: {dataset_path}\n"
                f"All files were skipped. Check that your CSVs contain "
                f"columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}'."
            )

        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded       : {len(dfs)} files ({skipped} skipped)")
        print(f"Combined shape: {df.shape}")

    else:
        # Single file mode
        print(f"Single file mode: {dataset_path}")
        df = DataProcessing.load_from_file(dataset_path, 'csv', sep=',')

    # ---- Validate required columns ----
    missing = [c for c in [TEXT_COLUMN, LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"HINT: Check TEXT_COLUMN='{TEXT_COLUMN}' and "
            f"LABEL_COLUMN='{LABEL_COLUMN}' at the top of this script."
        )

    # ---- Drop rows with missing values ----
    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing values.")

    # ---- Ensure label is integer ----
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    # Remap any label typos → 9 is a known typo for 0
    unexpected_labels = set(df[LABEL_COLUMN].unique()) - {0, 1}
    if unexpected_labels:
        print(f"Unexpected labels found: {unexpected_labels} — remapping to 0")
        df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(lambda x: x if x in {0, 1} else 0)

    # ---- Deduplicate sentences ----
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
    dupes = before_dedup - len(df)
    if dupes > 0:
        print(f"Dropped {dupes} duplicate sentences.")

    print(f"\nFinal shape     : {df.shape}")
    print(f"Label distribution:\n{df[LABEL_COLUMN].value_counts()}\n")

    return df


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================
def extract_embeddings(df, text_column=TEXT_COLUMN):
    """
    Extract SpaCy sentence embeddings.

    REQUIREMENT — feature_extraction.py:
        SpacyFeatureExtraction(df, text_column)
            .sentence_embeddings_extraction(attach_to_df=True)
            → returns DataFrame with column '{text_column} Embedding'
    """
    print("\n" + "="*40)
    print("EXTRACT EMBEDDINGS (SpaCy)")
    print("="*40)

    fe = SpacyFeatureExtraction(df, text_column)
    embeddings_df = fe.sentence_embeddings_extraction(attach_to_df=True)
    embeddings_col = f'{text_column} Embedding'

    print(f"Embedding column: '{embeddings_col}'")
    print(f"Shape after embedding: {embeddings_df.shape}\n")

    return embeddings_df, embeddings_col



# TRAIN / TEST SPLIT

def split_data(embeddings_df, embeddings_col, seed):
    """
    Split into 80% train+val (used inside GridSearchCV) and 20% held-out test.
    Stratified by label to preserve class balance.
    """
    print("\n" + "="*40)
    print(f"SPLIT DATA (seed={seed})")
    print("="*40)

    X = np.vstack(embeddings_df[embeddings_col].to_list())
    y = embeddings_df[LABEL_COLUMN].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    print(f"Train+Val size : {len(X_trainval)}")
    print(f"Test size      : {len(X_test)}")
    print(f"Train+Val label dist: {dict(zip(*np.unique(y_trainval, return_counts=True)))}")
    print(f"Test label dist     : {dict(zip(*np.unique(y_test, return_counts=True)))}\n")

    return X_trainval, X_test, y_trainval, y_test


# METRIC HELPERS

def compute_auc_scores(model_name, clf, X_test, y_test):
    """
    Compute ROC AUC and PR AUC.
    Returns 'N/A' strings for models that don't support probability output.
    """
    if model_name in NO_AUC_MODELS:
        return 'N/A', 'N/A'

    try:
        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_test)
        else:
            return 'N/A', 'N/A'

        roc_auc = roc_auc_score(y_test, scores)
        pr_auc  = average_precision_score(y_test, scores)
        return round(roc_auc, 4), round(pr_auc, 4)

    except Exception as e:
        print(f"AUC computation failed: {e}")
        return 'N/A', 'N/A'


def compute_all_metrics(model_name, clf, X_train, y_train, X_test, y_test):
    """
    Compute full metric suite after GridSearch refit.
    """
    # Regression models predict continuous values — round to nearest int for classification metrics, used for models that are not designed for classification
    if model_name in {'linear_regression', 'elastic_net'}:
        y_pred_train = np.round(clf.predict(X_train)).astype(int).clip(0, 1)
        y_pred_test  = np.round(clf.predict(X_test)).astype(int).clip(0, 1)
    else:
        y_pred_train = clf.predict(X_train)
        y_pred_test  = clf.predict(X_test)

    train_acc  = round(accuracy_score(y_train, y_pred_train), 4)
    test_acc   = round(accuracy_score(y_test,  y_pred_test),  4)
    f1_0       = round(f1_score(y_test, y_pred_test, pos_label=0, zero_division=0), 4)
    f1_1       = round(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0), 4)
    prec_0     = round(precision_score(y_test, y_pred_test, pos_label=0, zero_division=0), 4)
    prec_1     = round(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0), 4)
    rec_0      = round(recall_score(y_test, y_pred_test, pos_label=0, zero_division=0), 4)
    rec_1      = round(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0), 4)
    roc_auc, pr_auc = compute_auc_scores(model_name, clf, X_test, y_test)

    return {
        'train_accuracy':    train_acc,
        'test_accuracy':     test_acc,
        'f1_class_0':        f1_0,
        'f1_class_1':        f1_1,
        'precision_class_0': prec_0,
        'precision_class_1': prec_1,
        'recall_class_0':    rec_0,
        'recall_class_1':    rec_1,
        'roc_auc':           roc_auc,
        'pr_auc':            pr_auc,
    }


# GRID SEARCH — SINGLE MODEL

def run_grid_search(model_name, X_trainval, y_trainval, X_test, y_test, seed):
    """
    Run GridSearchCV for one model and return a results row dict.
    """
    print(f"\n{'='*40}")
    print(f"GRID SEARCH: {model_name.upper()} | seed={seed}")
    print(f"{'='*40}")

    # Build base model — Linear Regression has no random_state
    rs = None if model_name in NO_SEED_MODELS else seed
    wrapper = SkLearnModelFactory.select_model(model_name, random_state=rs)
    base_clf = wrapper.classifer if wrapper.classifer is not None else wrapper.train_model(
        X_trainval[:2], y_trainval[:2]
    ).classifer

    # Re-instantiate fresh unfitted estimator from the wrapper
    # We need the raw sklearn estimator for GridSearchCV
    wrapper_fresh = SkLearnModelFactory.select_model(model_name, random_state=rs)
    # Train on a tiny slice just to instantiate .classifer
    wrapper_fresh.train_model(X_trainval[:2], y_trainval[:2])
    estimator = wrapper_fresh.classifer.__class__(
        **{k: v for k, v in wrapper_fresh.classifer.get_params().items()}
    )

    param_grid = PARAM_GRIDS[model_name]

    # 7-fold stratified CV on the 80% train+val portion
    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,        # Refit best model on full train+val after search
        error_score=0.0    # Don't crash on bad param combos — score 0 instead
    )

    print(f"Fitting GridSearchCV ({cv.n_splits} folds)...")
    grid_search.fit(X_trainval, y_trainval)
    if hasattr(grid_search.best_estimator_, 'classes_'):
        print(f"Classes seen by model: {grid_search.best_estimator_.classes_}")
    else:
        print(f"Classes seen by model: N/A (regression model)")
    
    
    best_clf    = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1  = round(grid_search.best_score_, 4)

    print(f"Best CV F1   : {best_cv_f1}")
    print(f"Best Params  : {best_params}")

    # Full metric suite on test set
    metrics = compute_all_metrics(model_name, best_clf, X_trainval, y_trainval, X_test, y_test)

    row = {
        'seed':        seed,
        'model':       model_name,
        'best_cv_f1':  best_cv_f1,
        'best_params': str(best_params),
        **metrics
    }

    print(f"\nMetrics on held-out test set:")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")

    return row


# ============================================================
# AGGREGATE RESULTS ACROSS SEEDS
# ============================================================
def aggregate_results(all_results_df):
    """
    Compute mean ± std across seeds for numeric metrics.
    N/A values are excluded from aggregation.
    """
    numeric_cols = [
        'best_cv_f1', 'train_accuracy', 'test_accuracy',
        'f1_class_0', 'f1_class_1',
        'precision_class_0', 'precision_class_1',
        'recall_class_0', 'recall_class_1',
    ]
    auc_cols = ['roc_auc', 'pr_auc']

    agg_rows = []
    for model_name, group in all_results_df.groupby('model'):
        row = {'model': model_name}

        for col in numeric_cols:
            vals = pd.to_numeric(group[col], errors='coerce').dropna()
            row[f'{col}_mean'] = round(vals.mean(), 4) if len(vals) > 0 else 'N/A'
            row[f'{col}_std']  = round(vals.std(),  4) if len(vals) > 1 else 0.0

        for col in auc_cols:
            vals = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(vals) == 0:
                row[f'{col}_mean'] = 'N/A'
                row[f'{col}_std']  = 'N/A'
            else:
                row[f'{col}_mean'] = round(vals.mean(), 4)
                row[f'{col}_std']  = round(vals.std(),  4) if len(vals) > 1 else 0.0

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


# SAVE RESULTS
def save_results(results_df, agg_df, output_dir, model_names_tag):
    """
    Save outputs:
      - One CSV per seed   -> grid_search_seed_{seed}_{model_names_tag}.csv
      - One aggregated CSV -> grid_search_aggregated_{model_names_tag}.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Per-seed CSVs ---
    for seed, seed_group in results_df.groupby('seed'):
        seed_path = os.path.join(output_dir, f'grid_search_seed_{seed}_{model_names_tag}.csv')
        seed_group.to_csv(seed_path, index=False)
        print(f"Seed {seed} results saved : {seed_path}")

    # --- Aggregated CSV ---
    agg_path = os.path.join(output_dir, f'grid_search_aggregated_{model_names_tag}.csv')
    agg_df.to_csv(agg_path, index=False)
    print(f"Aggregated summary saved  : {agg_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Grid Search fine-tuning for prediction sentence classification'
    )
    parser.add_argument(
        '--dataset', required=True,
        help='Path to a single CSV file OR a directory of CSV files to combine. Must contain "Sentence" and "Human Annotation" columns.'
    )
    parser.add_argument(
        '--model', nargs='+', default=['all'],
        choices=ALL_MODEL_NAMES + ['all'],
        help=(
            'Model(s) to run grid search on. '
            'Pass one or more model names, or "all" to run every model. '
            'Example: --model logistic_regression random_forest_classifier'
        )
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=None,
        help='One or more random seeds. Example: --seeds 7 42 123'
    )
    parser.add_argument(
        '--save_path', default=None,
        help='Directory to save results. Defaults to grid_search_results/ next to this script.'
    )
    args = parser.parse_args()

    # Resolve models to run
    model_names = ALL_MODEL_NAMES if 'all' in args.model else args.model
    model_names_tag = 'all' if 'all' in args.model else '_'.join(args.model)

    # Resolve output directory
    output_dir = args.save_path or os.path.join(script_dir, 'grid_search_results')

    print("\n" + "="*40)
    print("GRID SEARCH PIPELINE")
    print("="*40)
    dataset_mode = 'directory' if os.path.isdir(args.dataset) else 'single file'
    print(f"Dataset    : {args.dataset} ({dataset_mode})")
    print(f"Models     : {model_names}")
    print(f"Seeds      : {args.seeds}")
    print(f"Output dir : {output_dir}")

    # ---- Load & embed once (same for all seeds) ----
    df = load_and_prepare_data(args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    #embeddings_df, embeddings_col = extract_embeddings(df)
    #Cache the Scapy so won't have to re-run everytime we run the script:
    # Cache embeddings to avoid re-running SpaCy on every execution
    cache_key = hashlib.md5(f"{args.dataset}_{df.shape}".encode()).hexdigest()[:8]
    cache_path = os.path.join(output_dir, f'embeddings_cache_{cache_key}.pkl')

    if os.path.exists(cache_path):
        print(f"\nLoading cached embeddings: {cache_path}")
        embeddings_df, embeddings_col = joblib.load(cache_path)
    else:
        print(f"\nNo cache found — running SpaCy embedding...")
        embeddings_df, embeddings_col = extract_embeddings(df)
        joblib.dump((embeddings_df, embeddings_col), cache_path)
        print(f"Embeddings cached to: {cache_path}")

    all_results = []

    for seed in args.seeds:
        print(f"\n{'#'*40}")
        print(f"# SEED {seed}")
        print(f"{'#'*40}")

        X_trainval, X_test, y_trainval, y_test = split_data(embeddings_df, embeddings_col, seed)

        for model_name in model_names:
            try:
                row = run_grid_search(model_name, X_trainval, y_trainval, X_test, y_test, seed)
                all_results.append(row)
            except Exception as e:
                print(f"\nERROR running {model_name} (seed={seed}): {e}")
                all_results.append({
                    'seed': seed, 'model': model_name,
                    'best_cv_f1': 'ERROR', 'best_params': str(e),
                    'train_accuracy': 'ERROR', 'test_accuracy': 'ERROR',
                    'f1_class_0': 'ERROR', 'f1_class_1': 'ERROR',
                    'precision_class_0': 'ERROR', 'precision_class_1': 'ERROR',
                    'recall_class_0': 'ERROR', 'recall_class_1': 'ERROR',
                    'roc_auc': 'N/A', 'pr_auc': 'N/A'
                })

    # ---- Compile & save ----
    all_results_df = pd.DataFrame(all_results)
    agg_df = aggregate_results(all_results_df)

    print("\n" + "="*40)
    print("AGGREGATED RESULTS (mean across seeds)")
    print("="*40)
    print(agg_df.to_string(index=False))

    save_results(all_results_df, agg_df, output_dir, model_names_tag)

    print("\n" + "="*40)
    print("GRID SEARCH COMPLETE")
    print("="*40)
    print(f"Models run : {len(model_names)}")
    print(f"Seeds run  : {args.seeds}")
    print(f"Results    : {output_dir}\n")


if __name__ == '__main__':
    main()