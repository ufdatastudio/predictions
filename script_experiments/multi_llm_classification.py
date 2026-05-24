"""
Multi-LLM prediction classification.

Three NaviGator models classify each sentence independently (with reasoning).
A fourth NaviGator model aggregates those three opinions into a final label.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../"))

from data_processing import DataProcessing
from metrics import EvaluationMetric
from text_generation_models import TextGenerationModelFactory, parse_json_response

DEFAULT_DATASET = "chronicle2050/chronicle2050.csv"
DEFAULT_PANEL_MODELS = [
    "llama-3.1-70b-instruct",
    "mistral-7b-instruct",
    "granite-3.3-8b-instruct",
]
DEFAULT_AGGREGATOR_MODEL = "llama-3.3-70b-instruct"

BASE_PROMPT = """
Role:
You are a linguist expert acting as a prediction detector. Your task is to identify
if a given sentence is a prediction (projection) about the future.

Background:
A prediction is a statement about what someone thinks will happen in the future.
Examples of predictions:
- "It will rain tomorrow." (Yes)
- "The stock market is expected to rise next quarter." (Yes)
- "I am going to the store." (No)
- "Lakers will win the championship." (Yes)

A prediction may contain: source, target, date, outcome.
""".strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify sentences using three panel LLMs and one aggregator LLM."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path relative to data/ (default: chronicle2050/chronicle2050.csv)",
    )
    parser.add_argument(
        "--panel-models",
        nargs=3,
        default=DEFAULT_PANEL_MODELS,
        metavar="MODEL",
        help="Three NaviGator model names for the panel stage",
    )
    parser.add_argument(
        "--aggregator-model",
        default=DEFAULT_AGGREGATOR_MODEL,
        help="NaviGator model name for the final aggregation stage",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sentences (useful for testing)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for results (default: script_experiments/output/multi_llm_<timestamp>.csv)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Print classification metrics against label when available",
    )
    return parser.parse_args()


def build_panel_prompt(sentence: str) -> str:
    return f"""{BASE_PROMPT}

Given the sentence below, analyze it and determine if it is a prediction (projection).
Use label 1 for prediction and label 0 for non-prediction.

Sentence: "{sentence}"

Respond ONLY with valid JSON in this exact format:
{{"label": 0, "reasoning": "your explanation here"}}
"""


def build_aggregator_prompt(
    sentence: str,
    panel_outputs: List[Dict[str, Optional[str]]],
) -> str:
    panel_summaries = []
    for idx, output in enumerate(panel_outputs, start=1):
        panel_summaries.append(
            f"""Panel model {idx} ({output["model_name"]}):
- label: {output["label"]}
- reasoning: {output["reasoning"]}
- raw_response: {output["raw_response"]}"""
        )

    joined_panels = "\n\n".join(panel_summaries)

    return f"""{BASE_PROMPT}

You are the final judge. Three independent models have already classified the same
sentence. Review their labels and reasoning, then decide the final classification.

Sentence: "{sentence}"

Previous model outputs:
{joined_panels}

Weigh agreement, disagreement, and the quality of each model's reasoning.
Respond ONLY with valid JSON in this exact format:
{{"label": 0, "reasoning": "your final explanation here"}}
"""


def normalize_label(label) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return int(label)
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "prediction", "predictions", "yes", "true"}:
            return 1
        if normalized in {"0", "not-prediction", "non-prediction", "no", "false"}:
            return 0
    return None


def label_to_text(label: Optional[int]) -> Optional[str]:
    if label is None:
        return None
    return "prediction" if label == 1 else "not-prediction"


def ground_truth_to_int(label: str) -> Optional[int]:
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower()
    if normalized == "prediction":
        return 1
    if normalized == "not-prediction":
        return 0
    return None


def classify_sentence(
    sentence: str,
    panel_models: List,
    aggregator_model,
) -> Dict:
    panel_outputs = []
    panel_prompt = build_panel_prompt(sentence)

    for model in panel_models:
        raw_response = model.chat_completion([model.user(panel_prompt)])
        label, reasoning = parse_json_response(raw_response)
        panel_outputs.append(
            {
                "model_name": model.__name__(),
                "raw_response": raw_response,
                "label": normalize_label(label),
                "reasoning": reasoning,
            }
        )

    aggregator_prompt = build_aggregator_prompt(sentence, panel_outputs)
    final_raw_response = aggregator_model.chat_completion([aggregator_model.user(aggregator_prompt)])
    final_label, final_reasoning = parse_json_response(final_raw_response)
    final_label = normalize_label(final_label)

    result = {
        "sentence": sentence,
        "final_label": final_label,
        "final_label_text": label_to_text(final_label),
        "final_reasoning": final_reasoning,
        "final_raw_response": final_raw_response,
        "aggregator_model": aggregator_model.__name__(),
    }

    for idx, panel_output in enumerate(panel_outputs, start=1):
        result[f"panel_{idx}_model"] = panel_output["model_name"]
        result[f"panel_{idx}_label"] = panel_output["label"]
        result[f"panel_{idx}_label_text"] = label_to_text(panel_output["label"])
        result[f"panel_{idx}_reasoning"] = panel_output["reasoning"]
        result[f"panel_{idx}_raw_response"] = panel_output["raw_response"]

    return result


def evaluate_results(results_df: pd.DataFrame) -> None:
    if "label" not in results_df.columns:
        print("Ground-truth column 'label' not found; skipping evaluation.")
        return

    y_true = results_df["label"].map(ground_truth_to_int)
    valid_mask = y_true.notna() & results_df["final_label"].notna()
    if not valid_mask.any():
        print("No rows with both ground truth and final labels; skipping evaluation.")
        return

    metrics = EvaluationMetric()
    print("\nFinal aggregator classification report:")
    metrics.eval_classification_report(
        y_true[valid_mask].astype(int).values,
        results_df.loc[valid_mask, "final_label"].astype(int).values,
    )

    for idx in range(1, 4):
        panel_col = f"panel_{idx}_label"
        panel_mask = valid_mask & results_df[panel_col].notna()
        if not panel_mask.any():
            continue
        print(f"\nPanel model {idx} classification report:")
        metrics.eval_classification_report(
            y_true[panel_mask].astype(int).values,
            results_df.loc[panel_mask, panel_col].astype(int).values,
        )


def main() -> None:
    args = parse_arguments()

    data_path = os.path.join(script_dir, "../data", args.dataset)
    print(f"Loading dataset: {data_path}")
    df = DataProcessing.load_from_file(data_path, "csv", sep=",")
    print(f"Shape: {df.shape}")

    if "sentence" not in df.columns:
        raise ValueError("Dataset must contain a 'sentence' column.")

    if args.limit is not None:
        df = df.head(args.limit)

    factory = TextGenerationModelFactory()
    panel_models = list(factory.create_instances(list(args.panel_models)).values())
    aggregator_model = factory.create_instance(args.aggregator_model)

    if len(panel_models) != 3:
        raise ValueError("Exactly three panel models are required.")

    print("\nPanel models:")
    for model in panel_models:
        print(f"  - {model.__name__()}")
    print(f"Aggregator model: {aggregator_model.__name__()}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying sentences"):
        sentence = row["sentence"]
        result = classify_sentence(sentence, panel_models, aggregator_model)

        for column in df.columns:
            if column != "sentence":
                result[column] = row[column]

        results.append(result)

    results_df = pd.DataFrame(results)

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"multi_llm_classification_{timestamp}.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    if args.evaluate:
        evaluate_results(results_df)


if __name__ == "__main__":
    main()
