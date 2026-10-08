# average_properties_extracation-v2.py

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
from bert_score import score as bert_score

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(
    os.path.join(script_dir, '../')
)

from data_processing import DataProcessing


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_SEEDS = [3, 7, 33]

METRICS = [
    'strict',
    'chrf',
    'bleu',
    'rouge',
    'bert_score'
]

PROPERTIES = [
    'Source',
    'Target',
    'Date',
    'Outcome'
]

GROUP_COLUMNS = [
    'model',
    'prompt_type',
    'property',
    'metric'
]

BERT_SCORE_MODEL = 'roberta-large'

# Paths:
#   <base_data_path>/extract_tolsa_properties_results/naacl_2026_submission/
#       naacl_2026_results_<YYYY-MM-DD>/
#           seed<S>/in_domain/<MODEL>/<PROMPT>/extracted_properties.csv

RESULTS_ROOT = 'extract_tolsa_properties_results'

SUBMISSION_DIR = 'naacl_2026_submission'

RESULTS_DIR_PREFIX = 'naacl_2026_results_'

DOMAIN_DIR = 'in_domain'

EXTRACTION_FILENAME = 'extracted_properties.csv'

GROUND_TRUTH_RELATIVE_PATH = os.path.join(
    RESULTS_ROOT,
    SUBMISSION_DIR,
    'ground_truth',
    'extracted_properties-ground_truth_only.csv'
)


# ============================================================
# PATH AND NAME HELPERS
# ============================================================

def clean_name(value):
    """
    Convert a model or prompt name into a filesystem-safe name.
    """
    return re.sub(
        r'[^A-Za-z0-9_.-]+',
        '_',
        str(value)
    )


def get_seed_folders(experiment_dir, expected_seeds):
    """
    Find the requested seed directories directly under a results directory.
    """
    seed_folders = {}

    if not os.path.isdir(experiment_dir):
        return seed_folders

    for item in os.listdir(experiment_dir):
        item_path = os.path.join(
            experiment_dir,
            item
        )

        if not os.path.isdir(item_path):
            continue

        match = re.fullmatch(
            r'seed(\d+)',
            item
        )

        if not match:
            continue

        seed = int(match.group(1))

        if seed not in expected_seeds:
            continue

        seed_folders[seed] = item_path

    return dict(
        sorted(seed_folders.items())
    )


def find_results_dirs(path, expected_seeds):
    """
    Return the results folder(s) that directly contain seed<S> folders.

    If path itself holds seed folders, use it.

    Otherwise treat path as the submission folder and use every
    naacl_2026_results_<date> folder under it that holds seed folders.

    Oldest first, so a newer run overrides an older one.
    """
    if get_seed_folders(
        path,
        expected_seeds
    ):
        return [path]

    results_dirs = []

    for item in sorted(os.listdir(path)):
        item_path = os.path.join(
            path,
            item
        )

        if not item.startswith(
            RESULTS_DIR_PREFIX
        ):
            continue

        if not os.path.isdir(item_path):
            continue

        if get_seed_folders(
            item_path,
            expected_seeds
        ):
            results_dirs.append(item_path)

    return results_dirs


def find_extraction_files(seed_dir):
    """
    Find extracted_properties.csv files under a seed directory.
    """
    extraction_files = []

    for root, _, files in os.walk(seed_dir):
        for filename in files:
            if filename != EXTRACTION_FILENAME:
                continue

            extraction_files.append(
                os.path.join(
                    root,
                    filename
                )
            )

    return sorted(extraction_files)


def infer_model_and_prompt(file_path, seed_dir):
    """
    Infer model name and prompt type from the extraction file path.

    Expected structure:

        seedX/
            in_domain/
                MODEL/
                    PROMPT/
                        extracted_properties.csv

    Older result files saved as:

        seedX/in_domain/MODEL/extracted_properties.csv

    return an empty prompt_type.
    """
    relative_path = os.path.relpath(
        file_path,
        seed_dir
    )

    parts = relative_path.split(
        os.sep
    )[:-1]

    model = ''
    prompt_type = ''

    if DOMAIN_DIR in parts:
        domain_index = parts.index(
            DOMAIN_DIR
        )

        if len(parts) > domain_index + 1:
            model = parts[
                domain_index + 1
            ]

        if len(parts) > domain_index + 2:
            prompt_type = parts[
                domain_index + 2
            ]

    return model, prompt_type


def read_prompt_type(file_path):
    """
    Fallback for result files saved without a PROMPT folder.

    Read the 'Prompt Type' column the extraction script writes.
    Returns '' if the column is missing or mixed.
    """
    try:
        values = (
            DataProcessing.load_from_file(
                file_path,
                file_type='csv',
                usecols=['Prompt Type']
            )['Prompt Type']
            .dropna()
            .unique()
        )

    except (
        ValueError,
        FileNotFoundError
    ):
        return ''

    if len(values) == 1:
        return clean_name(values[0])

    return ''


# ============================================================
# DATA LOADING
# ============================================================

def load_ground_truth(y_path):
    """
    Load the ground-truth property extraction file.
    """
    suffix = os.path.splitext(
        y_path
    )[1].lower()

    if suffix == '.xlsx':
        df = DataProcessing.load_from_file(
            y_path,
            file_type='xlsx'
        )

    else:
        df = DataProcessing.load_from_file(
            y_path,
            file_type='csv'
        )

    return df


def prepare_ground_truth(df):
    """
    Mirror how the extraction script reads this same file.

    Drop rows with an empty 'Base Sentence', then reset to a
    0..N-1 index so that ground-truth row i is the row the
    prediction files call Input_Index i.
    """
    if 'Base Sentence' in df.columns:
        df = df.dropna(
            subset=['Base Sentence']
        )

    return df.reset_index(
        drop=True
    )


def load_predictions(file_path):
    """
    Load one seed's extracted property results.
    """
    suffix = os.path.splitext(
        file_path
    )[1].lower()

    if suffix == '.xlsx':
        return DataProcessing.load_from_file(
            file_path,
            file_type='xlsx'
        )

    return DataProcessing.load_from_file(
        file_path,
        file_type='csv'
    )


def find_index_column(df):
    """
    Find the input/example index column used to align
    predictions with ground truth.
    """
    candidates = [
        'Input_Index',
        'input_index',
        'Index',
        'index'
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def normalize_text(value):
    """
    Collapse whitespace so 'Base Sentence' values can be
    compared after a CSV round trip.
    """
    return re.sub(
        r'\s+',
        ' ',
        str(value)
    ).strip()


def align_ground_truth_and_predictions(
        ground_truth_df,
        prediction_df):
    """
    Row-align ground truth and predictions.

    The prediction files' Input_Index is the row position in
    the ground-truth file. Ground-truth row i is matched to
    Input_Index i.

    If the predictions have no index column, rows are matched
    by order.
    """
    gt = (
        ground_truth_df
        .reset_index(drop=True)
        .copy()
    )

    prediction_index = find_index_column(
        prediction_df
    )

    if prediction_index is None:
        print(
            "    WARNING: no Input_Index in predictions; "
            "aligning by row order."
        )

        n_rows = min(
            len(gt),
            len(prediction_df)
        )

        return (
            gt.iloc[:n_rows].reset_index(drop=True),
            prediction_df.iloc[:n_rows].reset_index(drop=True)
        )

    pred = prediction_df.copy()

    pred['_alignment_index'] = pd.to_numeric(
        pred[prediction_index],
        errors='coerce'
    )

    pred = pred.dropna(
        subset=['_alignment_index']
    )

    pred['_alignment_index'] = (
        pred['_alignment_index']
        .astype(int)
    )

    pred = pred.drop_duplicates(
        subset='_alignment_index',
        keep='last'
    )

    pred = pred[
        pred['_alignment_index'].isin(
            gt.index
        )
    ]

    pred = (
        pred
        .sort_values('_alignment_index')
        .reset_index(drop=True)
    )

    gt = (
        gt.loc[
            pred['_alignment_index']
        ]
        .reset_index(drop=True)
    )

    return gt, pred


def count_text_mismatches(
        gt_df,
        pred_df):
    """
    Count aligned rows whose 'Base Sentence' differs
    between ground truth and predictions.
    """
    if (
        'Base Sentence' not in gt_df.columns
        or
        'Base Sentence' not in pred_df.columns
    ):
        return 0

    gt_text = gt_df[
        'Base Sentence'
    ].map(normalize_text)

    pred_text = pred_df[
        'Base Sentence'
    ].map(normalize_text)

    return int(
        (gt_text != pred_text).sum()
    )


# ============================================================
# PROPERTY VALUE HELPERS
# ============================================================

def get_property_value(
        row,
        property_name,
        suffix=''):
    """
    Safely retrieve a property value from a row.
    """
    column_name = (
        property_name + suffix
    )

    if column_name not in row.index:
        return ''

    value = row[column_name]

    if pd.isna(value):
        return ''

    return str(value).strip()


def normalize_strict(value):
    """
    Normalize whitespace and case for strict comparison.
    """
    value = str(value)

    value = re.sub(
        r'\s+',
        ' ',
        value
    )

    value = re.sub(
        r'\s*\|\s*',
        '|',
        value
    )

    return value.strip().lower()


def tokenize(value):
    """
    Tokenize a property value for BLEU.
    """
    value = str(value).strip()

    if not value:
        return []

    return value.split()

# ============================================================
# METRIC CALCULATIONS
# ============================================================

def compare_strict(
        reference,
        candidate):
    """
    Calculate strict normalized string matching.

    Returns 1.0 for an exact normalized match and 0.0 otherwise.
    """
    reference = normalize_strict(
        reference
    )

    candidate = normalize_strict(
        candidate
    )

    return float(
        reference == candidate
    )


def compare_chrf(
        reference,
        candidate,
        chrf_metric):
    """
    Calculate CHRF similarity.
    """
    reference = str(reference)
    candidate = str(candidate)

    if not reference and not candidate:
        return 1.0

    if not reference or not candidate:
        return 0.0

    result = chrf_metric.sentence_score(
        candidate,
        [reference]
    )

    return float(
        result.score / 100.0
    )


def compare_bleu(reference, candidate):
    """
    Calculate sentence-level BLEU.
    """
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    # Scale the n-gram weights to the span length so short exact matches score 1.0
    n = min(4, len(reference_tokens), len(candidate_tokens))
    score = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=tuple(1.0 / n for _ in range(n)),
        smoothing_function=smoothing
    )
    return float(score)


def compare_rouge(
        reference,
        candidate,
        rouge_metric):
    """
    Calculate ROUGE-L F1.
    """
    reference = str(reference)
    candidate = str(candidate)

    if not reference and not candidate:
        return 1.0

    if not reference or not candidate:
        return 0.0

    scores = rouge_metric.score(
        reference,
        candidate
    )

    return float(
        scores['rougeL'].fmeasure
    )


def calculate_bert_scores(
        references,
        candidates):
    """
    Calculate genuine BERTScore F1 values.

    Empty/empty pairs receive 1.0.
    Empty/non-empty pairs receive 0.0.
    BERTScore is calculated only for pairs where both
    strings are non-empty.
    """
    scores = [
        0.0
        for _ in references
    ]

    valid_indices = []
    valid_references = []
    valid_candidates = []

    for index, (
        reference,
        candidate
    ) in enumerate(
        zip(
            references,
            candidates
        )
    ):
        reference = str(
            reference
        ).strip()

        candidate = str(
            candidate
        ).strip()

        if not reference and not candidate:
            scores[index] = 1.0
            continue

        if not reference or not candidate:
            scores[index] = 0.0
            continue

        valid_indices.append(index)
        valid_references.append(reference)
        valid_candidates.append(candidate)

    if not valid_indices:
        return scores

    _, _, f1 = bert_score(
        valid_candidates,
        valid_references,
        model_type=BERT_SCORE_MODEL,
        lang='en',
        verbose=False
    )

    f1_scores = (
        f1
        .detach()
        .cpu()
        .numpy()
    )

    for index, score in zip(
        valid_indices,
        f1_scores
    ):
        scores[index] = float(score)

    return scores


# ============================================================
# EVALUATE ONE EXTRACTION FILE
# ============================================================

def evaluate_extraction_file(
        ground_truth_df,
        prediction_df):
    """
    Calculate all requested metrics for all TOLSA properties.

    Returns an empty DataFrame if the file cannot be aligned
    with the ground truth.
    """
    chrf_metric = CHRF(
        word_order=2
    )

    rouge_metric = (
        rouge_scorer.RougeScorer(
            ['rougeL'],
            use_stemmer=True
        )
    )

    rows = []

    gt_df, pred_df = (
        align_ground_truth_and_predictions(
            ground_truth_df,
            prediction_df
        )
    )

    print(
        f"    Aligned {len(pred_df)} of "
        f"{len(ground_truth_df)} ground-truth rows"
    )

    if pred_df.empty:
        print(
            "    WARNING: no prediction rows line up "
            "with the ground truth."
        )

        return pd.DataFrame(rows)

    n_mismatch = count_text_mismatches(
        gt_df,
        pred_df
    )

    if n_mismatch:
        bad = (
            gt_df['Base Sentence'].map(
                normalize_text
            )
            !=
            pred_df['Base Sentence'].map(
                normalize_text
            )
        )

        print(
            f"    ERROR: {n_mismatch} aligned row(s) "
            "have a different 'Base Sentence' than "
            "the ground truth. Skipping this file."
        )

        for gt_text, pred_text in zip(
            gt_df.loc[
                bad,
                'Base Sentence'
            ].head(3),

            pred_df.loc[
                bad,
                'Base Sentence'
            ].head(3)
        ):
            print(
                f"      ground truth: "
                f"{str(gt_text)[:80]}"
            )

            print(
                f"      prediction  : "
                f"{str(pred_text)[:80]}"
            )

        return pd.DataFrame(rows)

    gt_rows = [
        pd.Series(r)
        for r in gt_df.to_dict(
            'records'
        )
    ]

    pred_rows = [
        pd.Series(r)
        for r in pred_df.to_dict(
            'records'
        )
    ]

    for property_name in PROPERTIES:
        if (
            property_name not in gt_df.columns
            or
            property_name not in pred_df.columns
        ):
            print(
                f"    WARNING: column "
                f"'{property_name}' missing from "
                "ground truth or predictions. "
                "Skipping property."
            )
            continue

        references = [
            get_property_value(
                row,
                property_name
            )
            for row in gt_rows
        ]

        candidates = [
            get_property_value(
                row,
                property_name
            )
            for row in pred_rows
        ]

        bert_scores = calculate_bert_scores(
            references,
            candidates
        )

        strict_scores = []
        chrf_scores = []
        bleu_scores = []
        rouge_scores = []

        for reference, candidate in zip(
            references,
            candidates
        ):
            strict_scores.append(
                compare_strict(
                    reference,
                    candidate
                )
            )

            chrf_scores.append(
                compare_chrf(
                    reference,
                    candidate,
                    chrf_metric
                )
            )

            bleu_scores.append(
                compare_bleu(
                    reference,
                    candidate
                )
            )

            rouge_scores.append(
                compare_rouge(
                    reference,
                    candidate,
                    rouge_metric
                )
            )

        metric_scores = {
            'strict': strict_scores,
            'chrf': chrf_scores,
            'bleu': bleu_scores,
            'rouge': rouge_scores,
            'bert_score': bert_scores
        }

        for metric in METRICS:
            scores = metric_scores[metric]

            if scores:
                score = float(
                    np.mean(scores)
                )
            else:
                score = 0.0

            rows.append({
                'property': property_name,
                'metric': metric,
                'score': score,
                'n_examples': len(scores)
            })

    return pd.DataFrame(rows)


# ============================================================
# COLLECT PER-SEED RESULTS
# ============================================================

def collect_seed_metrics(
        experiment_dirs,
        ground_truth_df,
        expected_seeds):
    """
    Find extracted_properties.csv for each requested seed
    and evaluate them.

    experiment_dirs are searched oldest first. If the same
    seed/model/prompt appears in more than one folder, the
    newest one is used.
    """
    selected = {}

    for experiment_dir in experiment_dirs:
        print(
            f"\nResults folder: {experiment_dir}"
        )

        seed_folders = get_seed_folders(
            experiment_dir,
            expected_seeds
        )

        for seed, seed_dir in seed_folders.items():
            print(
                f"  seed{seed}: {seed_dir}"
            )

            for file_path in find_extraction_files(
                seed_dir
            ):
                model, prompt_type = (
                    infer_model_and_prompt(
                        file_path,
                        seed_dir
                    )
                )

                if not model:
                    model = 'unknown_model'

                if not prompt_type:
                    prompt_type = read_prompt_type(
                        file_path
                    )

                if not prompt_type:
                    print(
                        f"    WARNING: no PROMPT folder "
                        f"or 'Prompt Type' column for "
                        f"{file_path}; labeling it "
                        "'unknown_prompt'."
                    )

                    prompt_type = 'unknown_prompt'

                key = (
                    seed,
                    model,
                    prompt_type
                )

                if key in selected:
                    print(
                        f"    NOTE: seed{seed} / "
                        f"{model} / {prompt_type} was "
                        "also found in an older folder; "
                        "using the newer one."
                    )

                selected[key] = file_path

    found_seeds = {
        key[0]
        for key in selected
    }

    for seed in expected_seeds:
        if seed not in found_seeds:
            print(
                f"  seed{seed}: NOT FOUND"
            )

    result_columns = [
        'seed',
        'model',
        'prompt_type',
        'property',
        'metric',
        'score',
        'n_examples',
        'source_file'
    ]

    all_results = []

    for (
        seed,
        model,
        prompt_type
    ), file_path in sorted(
        selected.items()
    ):
        print(
            f"\nEvaluating seed {seed}: "
            f"{file_path}"
        )

        print(
            f"    Model: {model}"
        )

        print(
            f"    Prompt type: {prompt_type}"
        )

        prediction_df = load_predictions(
            file_path
        )

        evaluation_df = evaluate_extraction_file(
            ground_truth_df,
            prediction_df
        )

        if evaluation_df.empty:
            print(
                "    WARNING: No evaluation "
                "results generated."
            )
            continue

        evaluation_df['seed'] = seed
        evaluation_df['model'] = model
        evaluation_df['prompt_type'] = prompt_type
        evaluation_df['source_file'] = file_path

        all_results.append(
            evaluation_df[result_columns]
        )

    if not all_results:
        return pd.DataFrame(
            columns=result_columns
        )

    return pd.concat(
        all_results,
        ignore_index=True
    )


# ============================================================
# REMOVE DUPLICATE RESULTS
# ============================================================

def remove_duplicate_seed_rows(df):
    """
    Remove duplicate seed/model/prompt/property/metric results.
    """
    if df.empty:
        return df

    key_columns = [
        'seed',
        'model',
        'prompt_type',
        'property',
        'metric'
    ]

    df = df.sort_values(
        key_columns
    )

    df = df.drop_duplicates(
        subset=key_columns,
        keep='first'
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# IDENTIFY COMPLETE SEED GROUPS
# ============================================================

def get_complete_groups(
        df,
        expected_seeds):
    """
    Return groups containing every requested seed.

    A group is defined by:
    model, prompt_type, property, metric.
    """
    if df.empty:
        return pd.DataFrame(
            columns=GROUP_COLUMNS
        )

    complete_groups = []

    for (
        group_values,
        group_df
    ) in df.groupby(
        GROUP_COLUMNS,
        dropna=False
    ):
        observed_seeds = set(
            group_df['seed'].astype(int)
        )

        if observed_seeds == set(
            expected_seeds
        ):
            complete_groups.append(
                dict(
                    zip(
                        GROUP_COLUMNS,
                        group_values
                    )
                )
            )

    return pd.DataFrame(
        complete_groups
    )


# ============================================================
# AVERAGE COMPLETE GROUPS
# ============================================================

def average_complete_groups(
        df,
        complete_groups):
    """
    Average metrics across complete seed groups only.
    """
    if (
        df.empty
        or complete_groups.empty
    ):
        return pd.DataFrame(
            columns=[
                'model',
                'prompt_type',
                'property',
                'metric',
                'mean_score',
                'std_score',
                'n_seeds'
            ]
        )

    averaged_rows = []

    for _, group in complete_groups.iterrows():
        mask = pd.Series(
            True,
            index=df.index
        )

        for column in GROUP_COLUMNS:
            mask &= (
                df[column]
                == group[column]
            )

        group_df = df.loc[mask]

        scores = (
            group_df['score']
            .astype(float)
            .values
        )

        averaged_rows.append({
            'model':
                group['model'],

            'prompt_type':
                group['prompt_type'],

            'property':
                group['property'],

            'metric':
                group['metric'],

            'mean_score':
                float(np.mean(scores)),

            'std_score':
                float(
                    np.std(
                        scores,
                        ddof=1
                    )
                )
                if len(scores) > 1
                else 0.0,

            'n_seeds':
                len(scores)
        })

    return pd.DataFrame(
        averaged_rows
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
        experiment_dir,
        seed_df,
        averaged_df):
    """
    Save per-seed and averaged metric results.

    Output structure:

        averaged/
            in_domain/
                MODEL/
                    PROMPT/
                        seed_metrics_MODEL_PROMPT.csv
                        averaged_metrics_MODEL_PROMPT.csv
    """
    if seed_df.empty:
        return

    averaged_root = os.path.join(
        experiment_dir,
        'averaged'
    )

    os.makedirs(
        averaged_root,
        exist_ok=True
    )

    model_prompt_groups = (
        seed_df[
            [
                'model',
                'prompt_type'
            ]
        ]
        .drop_duplicates()
    )

    for _, model_prompt in (
        model_prompt_groups.iterrows()
    ):
        model = model_prompt['model']
        prompt_type = model_prompt[
            'prompt_type'
        ]

        clean_model = clean_name(
            model
        )

        clean_prompt = clean_name(
            prompt_type
        )

        output_dir = os.path.join(
            averaged_root,
            DOMAIN_DIR,
            clean_model,
            clean_prompt
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        seed_mask = (
            (seed_df['model'] == model)
            &
            (
                seed_df['prompt_type']
                == prompt_type
            )
        )

        model_prompt_seed_df = (
            seed_df.loc[
                seed_mask
            ].copy()
        )

        averaged_mask = (
            (averaged_df['model'] == model)
            &
            (
                averaged_df['prompt_type']
                == prompt_type
            )
        )

        model_prompt_averaged_df = (
            averaged_df.loc[
                averaged_mask
            ].copy()
        )

        seed_prefix = (
            f"seed_metrics_"
            f"{clean_model}_"
            f"{clean_prompt}"
        )

        averaged_prefix = (
            f"averaged_metrics_"
            f"{clean_model}_"
            f"{clean_prompt}"
        )

        DataProcessing.save_to_file(
            data=model_prompt_seed_df,
            path=output_dir,
            prefix=seed_prefix,
            save_file_type='csv',
            include_version=False
        )

        DataProcessing.save_to_file(
            data=model_prompt_averaged_df,
            path=output_dir,
            prefix=averaged_prefix,
            save_file_type='csv',
            include_version=False
        )

        seed_output_path = os.path.join(
            output_dir,
            f"{seed_prefix}.csv"
        )

        averaged_output_path = os.path.join(
            output_dir,
            f"{averaged_prefix}.csv"
        )

        print("\nSaved seed metrics:")
        print(
            f"  {seed_output_path}"
        )

        print("Saved averaged metrics:")
        print(
            f"  {averaged_output_path}"
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
        seed_df,
        averaged_df):
    """
    Print a compact summary of the evaluation results.
    """
    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)

    if seed_df.empty:
        print(
            "\nNo seed-level results were generated."
        )
        return

    print(
        f"\nPer-seed result rows: "
        f"{len(seed_df)}"
    )

    print(
        f"Complete averaged groups: "
        f"{len(averaged_df)}"
    )

    print(
        f"Seeds evaluated: "
        f"{sorted(seed_df['seed'].unique().tolist())}"
    )

    print(
        f"Models: "
        f"{sorted(seed_df['model'].unique().tolist())}"
    )

    print(
        f"Prompt types: "
        f"{sorted(seed_df['prompt_type'].unique().tolist())}"
    )

    print(
        f"Properties: "
        f"{sorted(seed_df['property'].unique().tolist())}"
    )

    print(
        f"Metrics: "
        f"{sorted(seed_df['metric'].unique().tolist())}"
    )

    if not averaged_df.empty:
        print("\nAveraged results:")

        display_columns = [
            'model',
            'prompt_type',
            'property',
            'metric',
            'mean_score',
            'std_score',
            'n_seeds'
        ]

        print(
            averaged_df[
                display_columns
            ].to_string(index=False)
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    """
    usage:

    # Default: ground truth + every naacl_2026_results_<date> folder under
    # extract_tolsa_properties_results/naacl_2026_submission
    # (newest run wins per seed/model/prompt)

    python average_properties_extracation-v2.py --seeds 3 7 33

    # Or pin one dated results folder:

    python average_properties_extracation-v2.py \\
        --y_path extract_tolsa_properties_results/naacl_2026_submission/ground_truth/extracted_properties-ground_truth_only.csv \\
        --y_hat_path extract_tolsa_properties_results/naacl_2026_submission/naacl_2026_results_2026-10-07 \\
        --seeds 3 7 33
    """

    print("\n" + "=" * 40)
    print("TOLSA PROPERTY EXTRACTION METRIC AVERAGING")
    print("=" * 40)

    # ============================================================
    # 1. CONFIGURATION
    # ============================================================

    base_data_path = (
        DataProcessing.load_base_data_path(
            script_dir
        )
    )

    default_y_path = os.path.join(
        base_data_path,
        GROUND_TRUTH_RELATIVE_PATH
    )

    default_y_hat_path = os.path.join(
        base_data_path,
        RESULTS_ROOT,
        SUBMISSION_DIR
    )

    parser = argparse.ArgumentParser(
        description=(
            'Average TOLSA property-extraction '
            'evaluation results across seeds'
        )
    )

    parser.add_argument(
        '--y_path',
        default=default_y_path,
        help=(
            'Path to ground-truth extraction file '
            '(absolute, or relative to base_data_path)'
        )
    )

    parser.add_argument(
        '--y_hat_path',
        default=default_y_hat_path,
        help=(
            'Either one naacl_2026_results_<date> '
            'folder, or the naacl_2026_submission '
            'folder (then every dated results folder '
            'under it is used)'
        )
    )

    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=EXPECTED_SEEDS,
        help='Seeds to include in averaging'
    )

    args = parser.parse_args()

    y_path = args.y_path

    if not os.path.isabs(y_path):
        y_path = os.path.join(
            base_data_path,
            y_path
        )

    y_hat_path = args.y_hat_path

    if not os.path.isabs(y_hat_path):
        y_hat_path = os.path.join(
            base_data_path,
            y_hat_path
        )

    expected_seeds = sorted(
        set(args.seeds)
    )

    print(
        f"Ground truth path: {y_path}"
    )

    print(
        f"Prediction path:   {y_hat_path}"
    )

    print(
        f"Expected seeds:    {expected_seeds}"
    )

    print(
        f"Metrics:           {METRICS}"
    )

    print(
        f"Grouping:          {GROUP_COLUMNS}"
    )

    # ============================================================
    # 2. VALIDATE INPUT PATHS
    # ============================================================

    if not os.path.isfile(y_path):
        print(
            "\nERROR: Ground-truth file does not exist:"
        )

        print(
            f"  {y_path}"
        )

        sys.exit(1)

    if not os.path.isdir(y_hat_path):
        print(
            "\nERROR: Prediction directory does not exist:"
        )

        print(
            f"  {y_hat_path}"
        )

        sys.exit(1)

    experiment_dirs = find_results_dirs(
        y_hat_path,
        expected_seeds
    )

    if not experiment_dirs:
        print(
            f"\nERROR: No seed folders found under: "
            f"{y_hat_path}"
        )

        print(
            "\nExpected prediction files:"
        )

        print(
            f"  {RESULTS_DIR_PREFIX}"
            f"<date>/seedX/{DOMAIN_DIR}/"
            f"MODEL/PROMPT/{EXTRACTION_FILENAME}"
        )

        sys.exit(1)

    print(
        f"Results folder(s): "
        f"{len(experiment_dirs)}"
    )

    for experiment_dir in experiment_dirs:
        print(
            f"  {experiment_dir}"
        )

    # ============================================================
    # 3. LOAD GROUND TRUTH
    # ============================================================

    print("\n" + "=" * 40)
    print("LOAD GROUND TRUTH")
    print("=" * 40)

    ground_truth_df = prepare_ground_truth(
        load_ground_truth(y_path)
    )

    print(
        f"Ground-truth rows: "
        f"{len(ground_truth_df)}"
    )

    # ============================================================
    # 4. COLLECT PER-SEED RESULTS
    # ============================================================

    print("\n" + "=" * 40)
    print("COLLECT PER-SEED RESULTS")
    print("=" * 40)

    seed_df = collect_seed_metrics(
        experiment_dirs=experiment_dirs,
        ground_truth_df=ground_truth_df,
        expected_seeds=expected_seeds
    )

    if seed_df.empty:
        print(
            "\nERROR: No per-seed metric results "
            "were generated."
        )

        print(
            "\nExpected prediction files:"
        )

        print(
            f"  seedX/{DOMAIN_DIR}/"
            f"MODEL/PROMPT/{EXTRACTION_FILENAME}"
        )

        sys.exit(1)

    # ============================================================
    # 5. REMOVE DUPLICATE RESULTS
    # ============================================================

    seed_df = remove_duplicate_seed_rows(
        seed_df
    )

    # ============================================================
    # 6. IDENTIFY COMPLETE SEED GROUPS
    # ============================================================

    complete_groups = get_complete_groups(
        df=seed_df,
        expected_seeds=expected_seeds
    )

    print(
        f"\nComplete groups found: "
        f"{len(complete_groups)}"
    )

    if not complete_groups.empty:
        print(
            f"Only groups containing all requested "
            f"seeds {expected_seeds} will be averaged."
        )

    # ============================================================
    # 7. AVERAGE COMPLETE GROUPS
    # ============================================================

    averaged_df = average_complete_groups(
        df=seed_df,
        complete_groups=complete_groups
    )

    # ============================================================
    # 8. SAVE RESULTS
    # ============================================================

    save_results(
        experiment_dir=y_hat_path,
        seed_df=seed_df,
        averaged_df=averaged_df
    )

    # ============================================================
    # 9. PRINT SUMMARY
    # ============================================================

    print_summary(
        seed_df=seed_df,
        averaged_df=averaged_df
    )

    # ============================================================
    # 10. COMPLETE
    # ============================================================

    print("\n" + "=" * 40)
    print("AVERAGING COMPLETE")
    print("=" * 40)