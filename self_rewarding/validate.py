"""Validation experiments for the Judge.

1. Gold-agreement: run the Judge over the human-verified gold examples and
   report the score distribution. If the judge cannot recognise human gold as
   high quality, it cannot be trusted on new data. This is the TOLSA analogue
   of the paper's "judging accuracy" result.

2. Cross-model bias check (optional): judge examples produced by a DIFFERENT
   model (an existing Task-2 run file) to measure self-enhancement bias -
   the judge should not systematically favour its own generations.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .backends import LLMBackend
from .checks import parse_judge_score
from .pipeline import (_append_jsonl, _load_jsonl, load_gold_csv, row_to_spans)
from .prompts import JudgePrompt


def judge_examples(backend: LLMBackend, examples: list[dict],
                   judge_runs: int, out_path: Path,
                   judge_temperature: float = 0.2,
                   seed: int = 3) -> list[dict]:
    """Judge a list of {'id', 'sentence', 'spans'} records, resume-safe."""
    done = {r["id"]: r for r in _load_jsonl(out_path)}
    results = []
    for ex in examples:
        if ex["id"] in done:
            results.append(done[ex["id"]])
            continue
        scores = []
        for j in range(judge_runs):
            raw = backend.generate(
                JudgePrompt.system_identity(),
                JudgePrompt.build(ex["sentence"], ex["spans"]),
                temperature=judge_temperature,
                seed=seed + 1000 * j + hash(str(ex["id"])) % 10000)
            score, err = parse_judge_score(raw)
            if score is not None:
                scores.append(score)
            else:
                print(f"[validate] id={ex['id']} run {j}: {err}")
        rec = {**ex, "judge_scores": scores,
               "judge_avg": sum(scores) / len(scores) if scores else 0.0}
        _append_jsonl(out_path, rec)
        done[ex["id"]] = rec
        results.append(rec)
    return results


def validate_gold(backend: LLMBackend, gold_path: str | Path,
                  out_dir: str | Path, judge_runs: int = 3,
                  threshold: float = 4.5, seed: int = 3,
                  limit: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = load_gold_csv(gold_path)
    if limit:
        gold = gold[:limit]
    examples = [{
        "id": row.get("Input_Index", i),
        "sentence": row["Base Sentence"],
        "spans": row_to_spans(row),
    } for i, row in enumerate(gold)]

    results = judge_examples(
        backend, examples, judge_runs,
        out_dir / "gold_judged.jsonl", seed=seed)

    avgs = [r["judge_avg"] for r in results if r["judge_scores"]]
    pass_rate = (sum(1 for a in avgs if a >= threshold) / len(avgs)
                 if avgs else 0.0)
    report = {
        "backend": backend.name,
        "n_gold": len(examples),
        "n_scored": len(avgs),
        "judge_runs": judge_runs,
        "threshold": threshold,
        "gold_pass_rate": round(pass_rate, 4),
        "mean_score": round(statistics.mean(avgs), 4) if avgs else None,
        "median_score": (round(statistics.median(avgs), 4) if avgs else None),
        "score_distribution": {
            str(s): sum(1 for a in avgs if round(a, 2) == s)
            for s in sorted(set(round(a, 2) for a in avgs))
        },
        "per_example": [
            {"id": r["id"], "judge_avg": r["judge_avg"],
             "judge_scores": r["judge_scores"]}
            for r in results
        ],
    }
    (out_dir / "gold_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_cross_model(backend: LLMBackend, run_file: str | Path,
                         out_dir: str | Path, judge_runs: int = 3,
                         threshold: float = 4.5, seed: int = 3,
                         limit: int | None = None) -> dict:
    """Judge another model's Task-2 extractions (11-column run file).

    The run file's Source/Target/Date/Outcome columns are '|'-joined, the same
    layout as the gold CSV, so row_to_spans works on it directly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_gold_csv(run_file)  # same column layout for the 4 properties
    if limit:
        rows = rows[:limit]
    examples = [{
        "id": row.get("Input_Index", i),
        "sentence": row["Base Sentence"],
        "spans": row_to_spans(row),
    } for i, row in enumerate(rows)]

    results = judge_examples(
        backend, examples, judge_runs,
        out_dir / "cross_model_judged.jsonl", seed=seed)

    avgs = [r["judge_avg"] for r in results if r["judge_scores"]]
    report = {
        "backend": backend.name,
        "run_file": str(run_file),
        "n_examples": len(examples),
        "judge_runs": judge_runs,
        "threshold": threshold,
        "mean_score": round(statistics.mean(avgs), 4) if avgs else None,
        "pass_rate": (round(sum(1 for a in avgs if a >= threshold) / len(avgs),
                            4) if avgs else None),
    }
    (out_dir / "cross_model_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
