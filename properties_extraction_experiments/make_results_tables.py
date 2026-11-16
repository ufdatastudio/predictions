"""
Paper Results-Table Generator
=============================

Reads `per_pair_scores.csv` produced by run_tolsa_evaluation.py and emits
`results_tables.tex`: booktabs tables sized for an 8-page NAACL/EACL-style
paper (results in one column block, no wasted width).

Tables
------
T1  Main results: model x prompt-style grid. Columns = metrics averaged over
    the four properties (strict, BLEU, ROUGE-L, chrF, BERTScore-F1),
    values = mean +/- std across the 3 seeds; last column = parse error rate.
T2  Property-level breakdown: property x metric, macro-averaged over all
    configurations (or restrict with --model / --prompt).
T3  Headline comparison: model x metric, everything else collapsed —
    the "average across seeds, prompts, and properties" number Detravious
    asked for, one row per model.

Usage
-----
    python make_results_tables.py --per-pair results/per_pair_scores.csv \
        --parse results/parse_error_summary.csv --out results_tables.tex

    # Restrict T2 to one configuration:
    python make_results_tables.py ... --model gpt-oss-120b --prompt zero-shot
"""

import argparse

import numpy as np
import pandas as pd

METRIC_ORDER = ["strict", "bleu", "rouge_l", "chrf", "bertscore_f1", "sbert"]
METRIC_LABEL = {
    "strict": "Strict",
    "bleu": "BLEU",
    "rouge_l": "ROUGE-L",
    "chrf": "chrF",
    "bertscore_f1": r"BERTScore$_F$",
    "sbert": "SBERT",
}
PROMPT_ORDER = ["zero-shot", "few-shot", "chain-of-thought"]
PROMPT_LABEL = {
    "zero-shot": "Zero-Shot",
    "few-shot": "Few-Shot",
    "chain-of-thought": "CoT",
}
PROP_LABEL = {"source": "Source", "target": "Target",
              "date": "Date", "outcome": "Outcome"}


def ms(mean, std, digits=2):
    """Format mean +/- std the way Dr. Grant's error bars read in table form."""
    return f"{mean:.{digits}f}$\\tiny{{\\raisebox{{-0.5ex}}{{\\scriptsize$\\pm${std:.{digits}f}}}}}$"


def tex_escape_model(name):
    return name.replace("_", r"\_").replace("-", "-")


# --------------------------------------------------------------------------
def table_main(per_pair, parse_df):
    """T1: rows = model x prompt; columns = metrics over properties, +- std across seeds."""
    # per (model, prompt, metric, seed): mean over rows+properties
    s = (per_pair.groupby(["model", "prompt", "metric", "seed"])["score"]
         .mean().reset_index())
    agg = (s.groupby(["model", "prompt", "metric"])
           .agg(mean=("score", "mean"), std=("score", "std"))
           .reset_index())

    metrics = [m for m in METRIC_ORDER if m in set(agg["metric"])]
    models = sorted(agg["model"].unique())
    prompts = [p for p in PROMPT_ORDER if p in set(agg["prompt"])]

    colspec = "l" + "c" * len(metrics) + "c"
    lines = [
        r"% ---------- T1: main results (model x prompt) ----------",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4.5pt}",
        r"  \begin{tabular}{" + colspec + "}",
        r"    \toprule",
        "    Model & Prompt & " + " & ".join(METRIC_LABEL[m] for m in metrics)
        + r" & Parse Err.\ \\",
        r"    \midrule",
    ]
    for mi, model in enumerate(models):
        for prompt in prompts:
            row = [tex_escape_model(model) if prompt == prompts[0] else "", PROMPT_LABEL.get(prompt, prompt)]
            for m in metrics:
                sub = agg[(agg.model == model) & (agg.prompt == prompt) & (agg.metric == m)]
                if len(sub):
                    row.append(ms(sub["mean"].iloc[0], sub["std"].iloc[0]))
                else:
                    row.append("--")
            pe = "--"
            if parse_df is not None and len(parse_df):
                psub = parse_df[(parse_df.model == model) & (parse_df.prompt == prompt)]
                if len(psub):
                    pe = f"{psub['parse_error_rate'].mean()*100:.1f}\\%"
            row.append(pe)
            lines.append("    " + " & ".join(row) + r" \\")
        if mi < len(models) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}", ""]
    lines.append(r"  \caption{TOLSA property extraction: similarity between human annotations "
                 r"(y) and LLM extractions ($\hat{y}$), averaged over the four properties "
                 r"(source, target, date, outcome). Values are mean $\pm$ std across seeds. "
                 r"Parse Err.\ is the rate of unparseable model responses.}")
    lines.append(r"  \label{tab:tolsa-main}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def table_by_property(per_pair, model=None, prompt=None):
    """T2: rows = property; columns = metrics; macro over remaining dimensions."""
    df = per_pair
    if model:
        df = df[df.model == model]
    if prompt:
        df = df[df.prompt == prompt]
    s = (df.groupby(["property", "metric", "seed"])["score"]
         .mean().reset_index())
    agg = (s.groupby(["property", "metric"])
           .agg(mean=("score", "mean"), std=("score", "std"))
           .reset_index())
    metrics = [m for m in METRIC_ORDER if m in set(agg["metric"])]
    props = [p for p in ["source", "target", "date", "outcome"]
             if p in set(agg["property"])]

    scope = ""
    if model or prompt:
        scope = f" for {model or 'all models'}, {prompt or 'all prompt styles'}"
    lines = [
        r"% ---------- T2: property-level breakdown ----------",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "c" * len(metrics) + "}",
        r"    \toprule",
        "    Property & " + " & ".join(METRIC_LABEL[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    for p in props:
        row = [PROP_LABEL[p]]
        for m in metrics:
            sub = agg[(agg.property == p) & (agg.metric == m)]
            row.append(ms(sub["mean"].iloc[0], sub["std"].iloc[0]) if len(sub) else "--")
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", ""]
    lines.append(r"  \caption{Per-property similarity" + scope + r", averaged over "
                 r"remaining dimensions; mean $\pm$ std across seeds.}")
    lines.append(r"  \label{tab:tolsa-property}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def table_headline(per_pair):
    """T3: rows = model; columns = metrics; everything collapsed."""
    s = (per_pair.groupby(["model", "metric", "seed"])["score"]
         .mean().reset_index())
    agg = (s.groupby(["model", "metric"])
           .agg(mean=("score", "mean"), std=("score", "std"))
           .reset_index())
    metrics = [m for m in METRIC_ORDER if m in set(agg["metric"])]
    models = sorted(agg["model"].unique())

    lines = [
        r"% ---------- T3: headline model comparison ----------",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{l" + "c" * len(metrics) + "}",
        r"    \toprule",
        "    Model & " + " & ".join(METRIC_LABEL[m] for m in metrics) + r" \\",
        r"    \midrule",
    ]
    for model in models:
        row = [tex_escape_model(model)]
        for m in metrics:
            sub = agg[(agg.model == model) & (agg.metric == m)]
            row.append(ms(sub["mean"].iloc[0], sub["std"].iloc[0]) if len(sub) else "--")
        lines.append("    " + " & ".join(row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", ""]
    lines.append(r"  \caption{Headline comparison: each cell averages over seeds, prompt "
                 r"styles, and properties. Best semantic metric "
                 r"(BERTScore$_F$) shown alongside n-gram baselines to demonstrate why "
                 r"token-overlap metrics under-score paraphrase-level agreement.}")
    lines.append(r"  \label{tab:tolsa-headline}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pair", required=True, help="per_pair_scores.csv")
    ap.add_argument("--parse", help="parse_error_summary.csv")
    ap.add_argument("--model", help="restrict T2 to this model")
    ap.add_argument("--prompt", help="restrict T2 to this prompt style")
    ap.add_argument("--out", default="results_tables.tex")
    args = ap.parse_args()

    per_pair = pd.read_csv(args.per_pair)
    parse_df = pd.read_csv(args.parse) if args.parse else None

    header = (
        "% Auto-generated by make_results_tables.py -- edit the script, not this file.\n"
        "% Requires: \\usepackage{booktabs}\n"
        "% Place T1 (table*) at the top of a results page; T2/T3 are single-column.\n\n"
    )
    body = "\n\n".join([
        table_main(per_pair, parse_df),
        table_by_property(per_pair, args.model, args.prompt),
        table_headline(per_pair),
    ])
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + body + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
