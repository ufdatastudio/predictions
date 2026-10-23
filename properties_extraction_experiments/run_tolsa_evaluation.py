"""
TOLSA Property-Extraction Evaluation Pipeline
=============================================

Compares human-annotated TOLSA properties (ground truth, y) against
LLM-extracted properties (y_hat) across experiment configurations and
computes strict, n-gram, and semantic similarity metrics plus parse
error rates.

Expected inputs
---------------
Ground truth CSV (exported from the Google Spreadsheet):
    Must contain `Base Sentence` plus the four property columns
    `Source`, `Target`, `Date`, `Outcome` (human annotations).
    An optional `Input_Index` column is used for alignment when present.

Results root (the Archive.zip Detravious sent):
    seed<S>/in_domain/<MODEL>/<PROMPT>/extracted_properties.csv
    A single extracted_properties.csv file also works.

Usage
-----
    # Full evaluation (ground truth available):
    python run_tolsa_evaluation.py \
        --ground-truth extracted_properties-ground_truth_only.csv \
        --results-root data/archive \
        --out results

    # Plumbing smoke test without ground truth (synthetic demo GT):
    python run_tolsa_evaluation.py --results-root data/archive --out results_demo --demo

Outputs (under --out)
---------------------
per_pair_scores.csv                 one row per (run, row, property, metric)
summary_config_property_metric.csv  mean/std across seeds, per model x prompt x property x metric
summary_model_metric.csv            mean/std across seeds+prompts+properties, per model x metric
summary_grand.csv                   headline: metric rows x model columns, one averaged number per cell
parse_error_summary.csv             parse error rate per run and per model x prompt
fig_metrics_by_model.png            bar chart with error bars (across seeds), paper-ready
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from tolsa_metrics import TolsaExtractionMetric, normalize_text

PROPERTIES = ["source", "target", "date", "outcome"]
PROP_COL = {p: p.capitalize() for p in PROPERTIES}  # source -> Source, etc.

RUN_PATTERN = re.compile(
    r"seed(\d+)[/\\](?:[a-z_]+)[/\\]([^/\\]+)[/\\]([^/\\]+)[/\\]extracted_properties\.csv$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Loading and alignment
# --------------------------------------------------------------------------
def find_run_files(results_root):
    """Return list of (seed, model, prompt, path) for every run CSV."""
    results_root = os.path.normpath(results_root)
    runs = []
    if os.path.isfile(results_root):
        meta = pd.read_csv(results_root, nrows=1)
        seed = str(meta["Seed"].iloc[0]) if "Seed" in meta else "na"
        model = str(meta["Model Name"].iloc[0]) if "Model Name" in meta else "na"
        prompt = str(meta["Prompt Type"].iloc[0]) if "Prompt Type" in meta else "na"
        runs.append((seed, model, prompt, results_root))
        return runs
    for dirpath, _dirnames, filenames in os.walk(results_root):
        if "extracted_properties.csv" not in filenames:
            continue
        path = os.path.join(dirpath, "extracted_properties.csv")
        rel = os.path.relpath(path, results_root)
        m = RUN_PATTERN.search(rel.replace(os.sep, "/"))
        if m:
            seed, model, prompt = m.group(1), m.group(2), m.group(3)
        else:
            parts = rel.split(os.sep)
            seed = next((p for p in parts if p.startswith("seed")), "na").replace("seed", "")
            model = parts[-3] if len(parts) >= 3 else "na"
            prompt = parts[-2] if len(parts) >= 2 else "na"
        runs.append((seed, model, prompt, path))
    return sorted(runs)


def load_ground_truth(path):
    """Load and normalize the human-annotated ground truth file (CSV or XLSX)."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, header=None)
        header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip()
                               .str.lower() == "base sentence"]
        if len(header_row) == 0:
            raise ValueError("Could not find the 'Base Sentence' header row "
                             f"in {path}")
        gt = pd.read_excel(path, header=int(header_row[0]))
    else:
        gt = pd.read_csv(path)
    gt.columns = [c.strip() for c in gt.columns]
    required = ["Base Sentence"] + [PROP_COL[p] for p in PROPERTIES]
    missing = [c for c in required if c not in gt.columns]
    if missing:
        raise ValueError(f"Ground truth CSV is missing columns: {missing}")
    if "Input_Index" not in gt.columns:
        gt["Input_Index"] = range(len(gt))
    for p in PROPERTIES:
        gt[PROP_COL[p]] = gt[PROP_COL[p]].fillna("").astype(str)
    gt["_norm_sent"] = gt["Base Sentence"].map(normalize_text)
    return gt


def align_run(run_df, gt):
    """Attach ground-truth property columns to a run's predictions.

    Primary key: normalized Base Sentence text (the sheet's Input_Index
    column is unreliable — it disagrees with the sentence text on most
    rows). Input_Index is kept for reporting only.
    """
    run_df = run_df.copy()
    run_df["Input_Index"] = run_df["Input_Index"].astype(int)
    gt_by_sent = {}
    for _, r in gt.iterrows():
        gt_by_sent.setdefault(normalize_text(r["Base Sentence"]), r)
    matched, gt_rows = [], []
    for _, row in run_df.iterrows():
        key = normalize_text(row["Base Sentence"])
        if key in gt_by_sent:
            matched.append(True)
            gt_rows.append(gt_by_sent[key])
        else:
            matched.append(False)
            gt_rows.append(None)
    run_df["_matched"] = matched
    aligned = run_df[run_df["_matched"]].copy()
    for p in PROPERTIES:
        aligned[f"y_{p}"] = [r[PROP_COL[p]] for r in gt_rows if r is not None]
    n_missing = int((~run_df["_matched"]).sum())
    if n_missing:
        print(f"  WARNING: {n_missing} run rows have no ground-truth sentence match.")
    return aligned


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_run(aligned, seed, model, prompt, include_neural=True):
    """Compute all metrics for one aligned run. Returns long-format rows."""
    records = []
    for p in PROPERTIES:
        y = aligned[f"y_{p}"].tolist()
        y_hat = aligned[PROP_COL[p]].fillna("").astype(str).tolist()
        scores = {
            "strict": [TolsaExtractionMetric.strict(a, b) for a, b in zip(y, y_hat)],
            "strict_set": [TolsaExtractionMetric.strict_set(a, b) for a, b in zip(y, y_hat)],
            "bleu": [TolsaExtractionMetric.bleu(a, b) for a, b in zip(y, y_hat)],
            "rouge_l": [TolsaExtractionMetric.rouge_l(a, b) for a, b in zip(y, y_hat)],
            "chrf": [TolsaExtractionMetric.chrf(a, b) for a, b in zip(y, y_hat)],
        }
        if include_neural:
            scores["sbert"] = TolsaExtractionMetric.sbert_batch(y, y_hat)
            scores["bertscore_f1"] = TolsaExtractionMetric.bertscore_f1_batch(y, y_hat)
        for metric, vals in scores.items():
            for idx, v in zip(aligned["Input_Index"], vals):
                records.append({
                    "seed": seed, "model": model, "prompt": prompt,
                    "property": p, "metric": metric,
                    "input_index": int(idx), "score": float(v),
                })
    return records


def parse_error_rate(run_df):
    status = run_df["Parse Status"].fillna("").astype(str).str.strip().str.upper()
    return float((status != "OK").mean())


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def aggregate(per_pair, parse_rates):
    """Build the summary tables from long-format per-pair scores."""
    df = pd.DataFrame(per_pair)

    # 1) model x prompt x property x metric: mean across rows, std across seeds
    s1 = (df.groupby(["model", "prompt", "property", "metric", "seed"])["score"]
            .mean().reset_index())
    by_config = (s1.groupby(["model", "prompt", "property", "metric"])
                 .agg(mean=("score", "mean"), std=("score", "std"),
                      n_seeds=("seed", "nunique"))
                 .reset_index())

    # 2) model x metric: collapse properties AND prompts, keep seed spread
    s2 = (df.groupby(["model", "metric", "seed"])["score"].mean().reset_index())
    by_model = (s2.groupby(["model", "metric"])
                .agg(mean=("score", "mean"), std=("score", "std"),
                     n_seeds=("seed", "nunique"))
                .reset_index())

    # 3) grand summary: metric rows x model columns (averaged over seeds, prompts, properties)
    grand = by_model.pivot(index="metric", columns="model", values="mean")
    grand["macro_avg"] = grand.mean(axis=1)

    # 4) parse error summary
    pe = pd.DataFrame(parse_rates)
    pe_summary = None
    if not pe.empty:
        pe_summary = (pe.groupby(["model", "prompt"])
                      .agg(parse_error_rate=("parse_error_rate", "mean"),
                           n_seeds=("seed", "nunique"))
                      .reset_index())
    return by_config, by_model, grand, pe_summary


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def make_figure(by_model, out_path):
    import matplotlib.pyplot as plt
    models = sorted(by_model["model"].unique())
    metrics = ["strict", "bleu", "rouge_l", "chrf", "sbert", "bertscore_f1"]
    metrics = [m for m in metrics if m in set(by_model["metric"])]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, model in enumerate(models):
        sub = by_model[by_model["model"] == model].set_index("metric")
        means = [sub.loc[m, "mean"] if m in sub.index else 0 for m in metrics]
        stds = [sub.loc[m, "std"] if m in sub.index else 0 for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace("_", "-") for m in metrics])
    ax.set_ylabel("Score (mean across seeds, prompts, properties)")
    ax.set_ylim(0, 1.05)
    ax.set_title("TOLSA Property Extraction: y vs. ŷ Similarity by Metric")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Demo ground truth (plumbing test only — NOT real annotations)
# --------------------------------------------------------------------------
def make_demo_ground_truth(run_files):
    """Build a synthetic 'ground truth' by lightly perturbing one run's outputs.

    Only used to validate pipeline plumbing when the real Google-Spreadsheet
    export is not available. Scores against it are meaningless.
    """
    seed, model, prompt, path = run_files[0]
    df = pd.read_csv(path)
    print(f"[demo] using {path} as ground-truth scaffold")
    gt = pd.DataFrame({
        "Input_Index": df["Input_Index"],
        "Base Sentence": df["Base Sentence"],
    })
    rng = np.random.default_rng(42)
    for p in PROPERTIES:
        col = PROP_COL[p]
        vals = df[col].fillna("").astype(str).tolist()
        perturbed = []
        for v in vals:
            if v and rng.random() < 0.5:
                spans = v.split("|")
                spans = [s.strip().upper() if rng.random() < 0.5 else s for s in spans]
                v = " | ".join(spans)
            perturbed.append(v)
        gt[col] = perturbed
    return gt


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TOLSA extraction evaluation")
    ap.add_argument("--ground-truth", help="human-annotated properties CSV")
    ap.add_argument("--results-root", required=True,
                    help="archive folder (seed*/in_domain/MODEL/PROMPT/) or a single run CSV")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-neural", action="store_true",
                    help="skip sbert/bertscore (no torch installed)")
    ap.add_argument("--filter", default=None,
                    help="only run configs whose seed/model/prompt contains this string "
                         "(e.g. 'seed3' or 'gpt-oss-120b/zero'); enables chunked scoring")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic ground truth for plumbing tests only")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    include_neural = not args.no_neural

    run_files = find_run_files(args.results_root)
    if args.filter:
        f = args.filter.lower().replace("\\", "/")
        run_files = [r for r in run_files
                     if f in f"seed{r[0]}/{r[1]}/{r[2]}".lower()]
    print(f"Found {len(run_files)} run file(s).")
    if not run_files:
        sys.exit("No extracted_properties.csv files found under results root.")

    if args.ground_truth:
        gt = load_ground_truth(args.ground_truth)
        print(f"Ground truth: {len(gt)} rows from {args.ground_truth}")
    elif args.demo:
        gt = make_demo_ground_truth(run_files)
        gt_path = os.path.join(args.out, "demo_ground_truth_NOT_REAL.csv")
        gt.to_csv(gt_path, index=False)
        print(f"[demo] synthetic ground truth written to {gt_path}")
    else:
        sys.exit("Provide --ground-truth CSV, or pass --demo for a plumbing test.")

    if include_neural:
        from tolsa_metrics import TolsaExtractionMetric
        TolsaExtractionMetric.enable_neural_cache(
            os.path.join(args.out, "neural_cache.json"))
        print(f"Neural pair cache: {os.path.join(args.out, 'neural_cache.json')}")

    per_pair, parse_rates = [], []
    pair_path = os.path.join(args.out, "per_pair_scores.csv")
    pe_path = os.path.join(args.out, "parse_error_summary.csv")
    done = set()
    if os.path.exists(pair_path):
        existing = pd.read_csv(pair_path)
        done = set(zip(existing["seed"].astype(str),
                       existing["model"].astype(str),
                       existing["prompt"].astype(str)))
        per_pair = existing.to_dict("records")
        print(f"Resuming: {len(done)} run(s) already scored in {pair_path}")
    if os.path.exists(pe_path):
        parse_rates = pd.read_csv(pe_path).to_dict("records")

    for seed, model, prompt, path in run_files:
        if (str(seed), model, prompt) in done:
            print(f"seed={seed} model={model} prompt={prompt}: already scored, skipping")
            continue
        run_df = pd.read_csv(path)
        aligned = align_run(run_df, gt)
        print(f"seed={seed} model={model} prompt={prompt}: "
              f"{len(aligned)}/{len(run_df)} rows matched", flush=True)
        if len(aligned) == 0:
            continue
        records = score_run(aligned, seed, model, prompt,
                            include_neural=include_neural)
        per_pair.extend(records)
        # incremental write: a timeout never loses finished runs
        pd.DataFrame(per_pair).to_csv(pair_path, index=False)
        parse_rates = [r for r in parse_rates
                       if not (str(r["seed"]) == str(seed) and r["model"] == model
                               and r["prompt"] == prompt)]
        parse_rates.append({"seed": seed, "model": model, "prompt": prompt,
                            "parse_error_rate": parse_error_rate(run_df)})
        pd.DataFrame(parse_rates).to_csv(pe_path, index=False)
        print(f"  -> appended {len(records)} scores ({len(per_pair)} total)", flush=True)

    pair_df = pd.DataFrame(per_pair)
    print(f"Wrote {pair_path} ({len(pair_df)} rows)")

    by_config, by_model, grand, pe_summary = aggregate(per_pair, parse_rates)
    by_config.to_csv(os.path.join(args.out, "summary_config_property_metric.csv"), index=False)
    by_model.to_csv(os.path.join(args.out, "summary_model_metric.csv"), index=False)
    grand.to_csv(os.path.join(args.out, "summary_grand.csv"))
    print("Wrote summary_config_property_metric.csv, summary_model_metric.csv, "
          "summary_grand.csv, parse_error_summary.csv")

    fig_path = os.path.join(args.out, "fig_metrics_by_model.png")
    make_figure(by_model, fig_path)
    print(f"Wrote {fig_path}")

    print("\n=== Grand summary (mean across seeds, prompts, properties) ===")
    print(grand.round(3).to_string())


if __name__ == "__main__":
    main()
