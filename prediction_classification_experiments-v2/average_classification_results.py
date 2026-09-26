# average_classification_results.py

import os
import re
import sys
import json
import argparse

import numpy as np
import pandas as pd

from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../"))

from data_processing import DataProcessing


# ============================================================
# RESULTS COLLECTION
# ============================================================

def get_latest_seed_version(experiment_dir, base_seed):
    """Find the latest version of a seed folder."""
    seed_pattern = f"seed{base_seed}"
    versioned_folders = []

    for item in os.listdir(experiment_dir):
        item_path = os.path.join(experiment_dir, item)

        if not os.path.isdir(item_path):
            continue

        if item == seed_pattern:
            versioned_folders.append((0, item))

        elif item.startswith(f"{seed_pattern}_v"):
            try:
                version = int(item.split("_v")[-1])
                versioned_folders.append((version, item))
            except ValueError:
                continue

    if not versioned_folders:
        return None

    _, latest_folder = max(
        versioned_folders,
        key=lambda item: item[0],
    )

    return latest_folder


def get_target_files(model_type):
    """Return result files associated with a requested model type."""
    model_files = {
        "ml": ["metrics_summary_ml_models.csv"],
        "llm": ["metrics_summary_llms.csv"],
        "rnn": ["metrics_summary_rnn.csv"],
        "gru": ["metrics_summary_gru.csv"],
        "bert": ["metrics_summary_bert.csv"],
    }

    if model_type == "all":
        return [
            "metrics_summary_ml_models.csv",
            "metrics_summary_llms.csv",
            "metrics_summary_rnn.csv",
            "metrics_summary_gru.csv",
            "metrics_summary_bert.csv",
        ]

    return model_files[model_type]


def get_file_tag(target_file):
    """Assign a concise label to a result-file type."""
    if target_file == "metrics_summary_ml_models.csv":
        return "ml"

    if target_file == "metrics_summary_llms.csv":
        return "llm"

    if target_file == "metrics_summary_rnn.csv":
        return "rnn"

    if target_file == "metrics_summary_gru.csv":
        return "gru"

    if target_file == "metrics_summary_bert.csv":
        return "bert"

    return "unknown"


def get_seed_folders(
    experiment_dir,
    filter_experiments=None,
):
    """Return selected seed folders from one experiment directory."""
    seed_folders = []

    for folder in os.listdir(experiment_dir):
        folder_path = os.path.join(experiment_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        if not folder.startswith("seed"):
            continue

        if filter_experiments is None or folder in filter_experiments:
            seed_folders.append(folder)

    return sorted(seed_folders)


def get_walk_root(
    seed_folder_path,
    model_type,
    embedding_model=None,
    prompting_strategy=None,
    model_name=None,
):
    """Return the directory within a seed folder to search for metrics files."""
    if model_type == "bert":
        if not model_name:
            raise ValueError(
                "--model_name is required when --model_type bert. "
                "Example: --model_name bert-large-cased"
            )

        return os.path.join(
            seed_folder_path,
            "in_domain",
            model_name,
        )

    if embedding_model:
        walk_root = os.path.join(
            seed_folder_path,
            "in_domain",
            embedding_model,
        )

        if prompting_strategy:
            walk_root = os.path.join(
                walk_root,
                prompting_strategy,
            )

        return walk_root

    return seed_folder_path


def collect_results(
    results_dir,
    mode="cross_dataset",
    target_experiment=None,
    filter_experiments=None,
    model_type="ml",
    embedding_model=None,
    prompting_strategy=None,
    model_name=None,
):
    """
    Collect metrics CSV files across seed folders.

    Results are grouped by:
    (experiment name, relative evaluation path, model type).
    """
    experiments = {}
    target_files = get_target_files(model_type)

    print("\n" + "=" * 60)
    print(f"COLLECTING RESULTS (mode={mode}, model_type={model_type})")
    print("=" * 60)
    print(f"Looking for: {target_files}")

    if embedding_model:
        print(f"Embedding model: {embedding_model}")

    if model_name:
        print(f"Model name: {model_name}")

    print(f"Results directory: {results_dir}\n")

    if mode == "single":
        if not target_experiment:
            raise ValueError(
                "target_experiment is required when mode='single'."
            )

        experiment_dirs = [target_experiment]

    else:
        experiment_dirs = []

        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)

            if not os.path.isdir(item_path):
                continue

            if item.startswith("."):
                continue

            if item in [
                "averaged_results",
                "cross_dataset_comparisons",
            ]:
                continue

            if re.search(r"\d{4}-\d{2}-\d{2}", item):
                if (
                    filter_experiments is None
                    or item in filter_experiments
                ):
                    experiment_dirs.append(item)

    for exp_dir_name in sorted(experiment_dirs):
        exp_dir_path = os.path.join(
            results_dir,
            exp_dir_name,
        )

        if not os.path.exists(exp_dir_path):
            print(f"⚠️ Experiment directory not found: {exp_dir_path}")
            continue

        seed_folders = get_seed_folders(
            experiment_dir=exp_dir_path,
            filter_experiments=filter_experiments,
        )

        if not seed_folders:
            print(f"⚠️ No matching seed folders found in: {exp_dir_path}")
            continue

        for seed_folder in seed_folders:
            seed_match = re.search(r"\d+", seed_folder)

            if seed_match is None:
                print(f"⚠️ Skipping malformed seed folder: {seed_folder}")
                continue

            seed = int(seed_match.group())
            seed_folder_path = os.path.join(
                exp_dir_path,
                seed_folder,
            )

            walk_root = get_walk_root(
                seed_folder_path=seed_folder_path,
                model_type=model_type,
                embedding_model=embedding_model,
                prompting_strategy=prompting_strategy,
                model_name=model_name,
            )

            if not os.path.exists(walk_root):
                print(
                    f"⚠️ Skipping {seed_folder}: "
                    f"results directory not found: {walk_root}"
                )
                continue

            for root, _, files in os.walk(walk_root):
                for target_file in target_files:
                    if target_file not in files:
                        continue

                    csv_path = os.path.join(
                        root,
                        target_file,
                    )

                    rel_path = os.path.relpath(
                        root,
                        seed_folder_path,
                    )

                    file_tag = get_file_tag(target_file)

                    eval_key = (
                        exp_dir_name,
                        rel_path,
                        file_tag,
                    )

                    if eval_key not in experiments:
                        experiments[eval_key] = []

                    df = DataProcessing.load_from_file(
                        csv_path,
                        "csv",
                        sep=",",
                    )

                    experiments[eval_key].append({
                        "seed": seed,
                        "folder": rel_path,
                        "data": df,
                    })

                    print(
                        f"✓ Loaded [{file_tag}]: "
                        f"{seed_folder}/{rel_path}/{target_file}"
                    )

    return experiments


# ============================================================
# AVERAGING
# ============================================================

def average_experiment_results(experiment_data):
    """Average numeric metrics across seeds, grouped by model."""
    if not experiment_data:
        return None, None, 0

    all_dfs = [
        item["data"]
        for item in experiment_data
    ]

    combined_df = pd.concat(
        all_dfs,
        ignore_index=True,
    )

    if combined_df.columns[0] == "":
        combined_df = combined_df.rename(
            columns={combined_df.columns[0]: "model"}
        )

    if "model" not in combined_df.columns:
        raise ValueError(
            "Expected a 'model' column in metrics files. "
            f"Found columns: {combined_df.columns.tolist()}"
        )

    numeric_cols = combined_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    if "seed" in numeric_cols:
        numeric_cols.remove("seed")

    mean_df = combined_df.groupby("model")[numeric_cols].mean()
    std_df = combined_df.groupby("model")[numeric_cols].std()

    if len(mean_df) > 1:
        mean_df.loc["mean_across_models"] = mean_df.mean()
        std_df.loc["std_across_models"] = std_df.mean()

    n_seeds = len(experiment_data)

    return mean_df, std_df, n_seeds


def format_mean_std(mean_df, std_df, key_cols=None):
    """Create a DataFrame formatted as mean ± standard deviation."""
    mean_reset = mean_df.reset_index()
    std_reset = std_df.reset_index()

    if "index" in mean_reset.columns:
        mean_reset = mean_reset.rename(
            columns={"index": "model"}
        )

    if "index" in std_reset.columns:
        std_reset = std_reset.rename(
            columns={"index": "model"}
        )

    if key_cols is None:
        available_columns = mean_reset.columns.tolist()

    else:
        available_columns = [
            "model",
            *[
                column
                for column in key_cols
                if column in mean_reset.columns
            ],
        ]

    mean_reset = mean_reset[
        available_columns
    ].reset_index(drop=True)

    std_reset = std_reset[
        available_columns
    ].reset_index(drop=True)

    formatted_df = mean_reset.copy()

    for column in formatted_df.columns:
        if column == "model":
            continue

        formatted_df[column] = (
            mean_reset[column].apply(
                lambda value: (
                    f"{value:.4f}"
                    if pd.notna(value)
                    else "nan"
                )
            )
            + " $\\pm$ "
            + std_reset[column].apply(
                lambda value: (
                    f"{value:.4f}"
                    if pd.notna(value)
                    else "nan"
                )
            )
        )

    return formatted_df


# ============================================================
# SAVING
# ============================================================

def save_averaged_results(
    results_dir,
    experiments,
    mode="cross_dataset",
    embedding_model=None,
    model_name=None,
):
    """Save mean, standard deviation, formatted, and metadata outputs."""
    all_summaries = []

    for (
        base_exp_name,
        evaluation_path,
        file_tag,
    ), exp_data in experiments.items():

        display_name = (
            f"{base_exp_name} → {evaluation_path} [{file_tag}]"
        )

        print("\n" + "=" * 60)
        print(f"AVERAGING: {display_name}")
        print("=" * 60)

        mean_df, std_df, n_seeds = average_experiment_results(
            exp_data
        )

        if mean_df is None:
            continue

        print(f"Seeds used: {n_seeds}")

        seed_details = []

        for item in exp_data:
            seed_details.append({
                "seed": item["seed"],
                "folder": item["folder"],
            })

        if mode == "single":
            averaged_base = os.path.join(
                results_dir,
                base_exp_name,
                "averaged",
            )

            save_dir = os.path.join(
                averaged_base,
                evaluation_path,
                file_tag,
            )

        else:
            save_dir = None

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            mean_df.to_csv(
                os.path.join(save_dir, "mean.csv")
            )

            std_df.to_csv(
                os.path.join(save_dir, "std.csv")
            )

            mean_std_df = format_mean_std(
                mean_df,
                std_df,
            )

            mean_std_df.to_csv(
                os.path.join(save_dir, "mean_std.csv"),
                index=False,
            )

            metadata = {
                "experiment": display_name,
                "model_type": file_tag,
                "embedding_model": embedding_model or "N/A",
                "model_name": model_name or "N/A",
                "n_seeds": n_seeds,
                "seeds_used": seed_details,
                "date_averaged": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "files_generated": {
                    "mean": "mean.csv",
                    "std": "std.csv",
                    "mean_std": "mean_std.csv",
                },
            }

            metadata_path = os.path.join(
                save_dir,
                "metadata.json",
            )

            with open(metadata_path, "w") as file:
                json.dump(
                    metadata,
                    file,
                    indent=2,
                )

            print(f"✓ Saved averaged results to: {save_dir}")

        all_summaries.append({
            "experiment": display_name,
            "model_type": file_tag,
            "n_seeds": n_seeds,
            "seed_info": seed_details,
            "mean": mean_df,
            "std": std_df,
        })

    return all_summaries


# ============================================================
# CROSS-DATASET COMPARISONS
# ============================================================

def detect_dataset_type(experiment_name):
    """Auto-detect dataset type from experiment name."""
    name_lower = experiment_name.lower()

    if "imbalanced" in name_lower:
        return "imbalanced"

    if "oversampled" in name_lower or "oversample" in name_lower:
        return "oversampled"

    if "undersampled" in name_lower or "undersample" in name_lower:
        return "undersampled"

    return experiment_name


def compute_cross_dataset_margins(summaries):
    """Compute same-model metric variation across datasets."""
    print("\n" + "=" * 50)
    print("CROSS-DATASET MARGINS")
    print("=" * 50 + "\n")

    dataset_means = {}
    dataset_type_mapping = {}

    for summary in summaries:
        experiment_name = summary["experiment"]
        mean_df = summary["mean"]

        dataset_type = detect_dataset_type(
            experiment_name
        )

        dataset_type_mapping[
            experiment_name
        ] = dataset_type

        dataset_means[dataset_type] = mean_df

    print("Dataset types detected:")

    for experiment_name, dataset_type in (
        dataset_type_mapping.items()
    ):
        print(f"  {experiment_name} → {dataset_type}")

    model_margins = []
    all_models = set()

    for mean_df in dataset_means.values():
        all_models.update(mean_df.index.tolist())

    all_models = sorted([
        model
        for model in all_models
        if not model.startswith("mean_")
        and not model.startswith("std_")
    ])

    metric_columns = [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "precision_class_0",
        "precision_class_1",
        "recall_class_0",
        "recall_class_1",
        "f1_class_0",
        "f1_class_1",
        "train_f1_class_0",
        "train_f1_class_1",
        "val_f1_class_0",
        "val_f1_class_1",
        "test_f1_class_0",
        "test_f1_class_1",
        "roc_auc",
        "pr_auc",
        "train_roc_auc",
        "val_roc_auc",
        "test_roc_auc",
        "train_pr_auc",
        "val_pr_auc",
        "test_pr_auc",
    ]

    for model in all_models:
        row = {"model": model}

        for dataset_type, mean_df in dataset_means.items():
            if model not in mean_df.index:
                continue

            for metric in metric_columns:
                if metric not in mean_df.columns:
                    continue

                value = mean_df.loc[model, metric]

                if pd.notna(value):
                    row[f"{dataset_type}_{metric}"] = value

        for metric in metric_columns:
            values = [
                row[f"{dataset_type}_{metric}"]
                for dataset_type in dataset_means
                if f"{dataset_type}_{metric}" in row
            ]

            if values:
                row[f"{metric}_mean_across_datasets"] = np.mean(
                    values
                )
                row[f"{metric}_std_across_datasets"] = np.std(
                    values
                )
                row[f"{metric}_margin"] = (
                    max(values) - min(values)
                )

        model_margins.append(row)

    model_margins_df = pd.DataFrame(model_margins)

    dataset_accuracy_rows = []

    for dataset_type, mean_df in dataset_means.items():
        model_only_df = mean_df[
            ~mean_df.index.str.startswith("mean_")
            & ~mean_df.index.str.startswith("std_")
        ]

        accuracy_column = (
            "test_accuracy"
            if "test_accuracy" in model_only_df.columns
            else "accuracy"
        )

        if accuracy_column not in model_only_df.columns:
            continue

        dataset_accuracy_rows.append({
            "dataset": dataset_type,
            "accuracy_mean": model_only_df[
                accuracy_column
            ].mean(),
            "accuracy_std": model_only_df[
                accuracy_column
            ].std(),
            "accuracy_min": model_only_df[
                accuracy_column
            ].min(),
            "accuracy_max": model_only_df[
                accuracy_column
            ].max(),
            "accuracy_margin": (
                model_only_df[accuracy_column].max()
                - model_only_df[accuracy_column].min()
            ),
            "best_model": model_only_df[
                accuracy_column
            ].idxmax(),
            "worst_model": model_only_df[
                accuracy_column
            ].idxmin(),
        })

    dataset_accuracy_df = pd.DataFrame(
        dataset_accuracy_rows
    )

    return model_margins_df, dataset_accuracy_df


def save_cross_dataset_results(
    results_dir,
    summaries,
    model_margins_df,
    dataset_accuracy_df,
):
    """Save cross-dataset comparison outputs."""
    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H%M%S"
    )

    comparison_dir = os.path.join(
        results_dir,
        "cross_dataset_comparisons",
        f"run_{timestamp}",
    )

    os.makedirs(comparison_dir, exist_ok=True)

    model_margins_df.to_csv(
        os.path.join(
            comparison_dir,
            "cross_dataset_model_margins.csv",
        ),
        index=False,
    )

    dataset_accuracy_df.to_csv(
        os.path.join(
            comparison_dir,
            "cross_dataset_accuracy.csv",
        ),
        index=False,
    )

    metadata = {
        "timestamp": timestamp,
        "n_experiments": len(summaries),
        "experiments_compared": [
            {
                "experiment": summary["experiment"],
                "model_type": summary["model_type"],
                "n_seeds": summary["n_seeds"],
                "seeds_used": summary["seed_info"],
            }
            for summary in summaries
        ],
    }

    with open(
        os.path.join(
            comparison_dir,
            "experiments_compared.json",
        ),
        "w",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"✓ Saved cross-dataset comparison to: "
        f"{comparison_dir}"
    )

    return comparison_dir


# ============================================================
# LATEX OUTPUT
# ============================================================

def print_latex_summary(summaries, model_margins_df=None):
    """Print LaTeX-formatted mean ± standard deviation tables."""
    print("\n" + "=" * 60)
    print("LATEX OUTPUT (MEAN ± STD)")
    print("=" * 60 + "\n")

    ml_key_columns = [
        "precision_class_1",
        "recall_class_1",
        "f1_class_1",
        "test_accuracy",
        "roc_auc",
        "pr_auc",
        "train_accuracy",
        "val_accuracy",
    ]

    fallback_key_columns = [
        "precision_class_1",
        "recall_class_1",
        "f1_class_1",
        "accuracy",
        "roc_auc",
        "pr_auc",
        "train_accuracy",
        "val_accuracy",
    ]

    bert_key_columns = [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "train_f1_class_1",
        "val_f1_class_1",
        "test_f1_class_1",
        "train_roc_auc",
        "val_roc_auc",
        "test_roc_auc",
        "train_pr_auc",
        "val_pr_auc",
        "test_pr_auc",
    ]

    for summary in summaries:
        experiment_name = summary["experiment"]
        mean_df = summary["mean"]
        std_df = summary["std"]

        print(f"% {experiment_name}")
        print(f"% Seeds: {summary['n_seeds']}\n")

        if "test_f1_class_1" in mean_df.columns:
            key_columns = bert_key_columns

        elif "test_accuracy" in mean_df.columns:
            key_columns = ml_key_columns

        else:
            key_columns = fallback_key_columns

        latex_df = format_mean_std(
            mean_df,
            std_df,
            key_cols=key_columns,
        )

        print(
            latex_df.to_latex(
                index=False,
                escape=False,
            )
        )

        print()

    if (
        model_margins_df is not None
        and not model_margins_df.empty
    ):
        print("% Cross-Dataset Model Margins\n")
        print(
            model_margins_df.to_latex(
                index=False,
                escape=False,
                float_format="%.4f",
            )
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Average classification results across "
            "multiple seed runs."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["single", "cross_dataset"],
        default="cross_dataset",
        help="Averaging mode: single or cross_dataset.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment folder name. Required for --mode single.",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Specific seed folders to average. "
            "Example: --experiments seed3 seed7 seed33"
        ),
    )

    parser.add_argument(
        "--model_type",
        choices=["ml", "llm", "rnn", "gru", "bert", "all"],
        default="ml",
        help="Model family whose result files should be averaged.",
    )

    parser.add_argument(
        "--embedding_model",
        default=None,
        choices=[
            "spacy_small",
            "spacy_medium",
            "spacy_large",
            "spacy_transformer",
            "st_mpnet_base",
            "st_distilroberta",
            "st_minilm_l12",
            "st_minilm_l6",
        ],
        help=(
            "Embedding-model directory used by ML, RNN, "
            "or GRU experiments."
        ),
    )

    parser.add_argument(
        "--model_name",
        default=None,
        help=(
            "Model directory under seed/in_domain/. "
            "Required for BERT. Example: bert-large-cased"
        ),
    )

    parser.add_argument(
        "--prompting_strategy",
        default=None,
        help=(
            "Optional prompting-strategy subdirectory "
            "for LLM experiments."
        ),
    )

    parser.add_argument(
        "--results_dir",
        default=None,
        help=(
            "Root directory containing experiment folders. "
            "Default: ../data/classification_results/"
        ),
    )

    args = parser.parse_args()

    if args.mode == "single" and not args.experiment:
        parser.error(
            "--experiment is required when --mode single."
        )

    if args.model_type == "bert" and not args.model_name:
        parser.error(
            "--model_name is required when --model_type bert."
        )

    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(
            script_dir,
            "../data/classification_results/",
        )

    print("\n" + "=" * 60)
    print("AVERAGE CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Mode:              {args.mode}")
    print(f"Model type:        {args.model_type}")
    print(
        f"Embedding model:   "
        f"{args.embedding_model or 'N/A'}"
    )
    print(f"Model name:        {args.model_name or 'N/A'}")
    print(f"Results directory: {results_dir}")

    if args.mode == "single":
        print(f"Target experiment: {args.experiment}")

    if args.experiments:
        print(f"Seed folders:      {args.experiments}")

    print()

    experiments = collect_results(
        results_dir=results_dir,
        mode=args.mode,
        target_experiment=args.experiment,
        filter_experiments=args.experiments,
        model_type=args.model_type,
        embedding_model=args.embedding_model,
        prompting_strategy=args.prompting_strategy,
        model_name=args.model_name,
    )

    if not experiments:
        print("\n❌ No experiment results found to average.")
        sys.exit(0)

    print(f"\nFound {len(experiments)} evaluation group(s):")

    for evaluation_key, evaluation_data in experiments.items():
        print(
            f"  - {evaluation_key}: "
            f"{len(evaluation_data)} seed(s)"
        )

    summaries = save_averaged_results(
        results_dir=results_dir,
        experiments=experiments,
        mode=args.mode,
        embedding_model=args.embedding_model,
        model_name=args.model_name,
    )

    model_margins_df = None

    if args.mode == "cross_dataset" and len(summaries) >= 2:
        model_margins_df, dataset_accuracy_df = (
            compute_cross_dataset_margins(summaries)
        )

        save_cross_dataset_results(
            results_dir=results_dir,
            summaries=summaries,
            model_margins_df=model_margins_df,
            dataset_accuracy_df=dataset_accuracy_df,
        )

    elif args.mode == "cross_dataset":
        print(
            "\n⚠️ At least two experiment summaries are required "
            "for cross-dataset margins."
        )

    print_latex_summary(
        summaries=summaries,
        model_margins_df=model_margins_df,
    )

    print("\n" + "=" * 60)
    print("AVERAGING COMPLETE")
    print("=" * 60)
    print(f"Mode:                      {args.mode}")
    print(f"Model type:                {args.model_type}")
    print(f"Total evaluations averaged: {len(summaries)}")

    if args.mode == "single":
        print("\nAveraged results are under:")

        for (
            experiment_name,
            evaluation_path,
            file_tag,
        ) in experiments.keys():

            print(
                os.path.join(
                    results_dir,
                    experiment_name,
                    "averaged",
                    evaluation_path,
                    file_tag,
                )
            )

    print()