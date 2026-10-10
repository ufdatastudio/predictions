#!/usr/bin/env python3

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Average per-model LLM metrics across seeds."
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 33])
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of model directories to average. Defaults to all models.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.results_dir) / args.experiment
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment not found: {experiment_dir}")

    groups = {}

    print(f"Experiment: {experiment_dir}")
    print(f"Expected seeds: {args.seeds}")

    for seed in args.seeds:
        seed_dir = experiment_dir / f"seed{seed}" / "in_domain" / "llm"

        if not seed_dir.is_dir():
            print(f"WARNING: Missing seed directory: {seed_dir}")
            continue

        for csv_path in sorted(seed_dir.glob(
            "*/*/*/metrics_summary_*.csv"
        )):
            # Expected: model / prompt / split / metrics_summary_*.csv
            model, prompt, split = csv_path.relative_to(seed_dir).parts[:3]
            key = (model, prompt, split)

            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"WARNING: Empty metrics file: {csv_path}")
                continue

            # These files should contain one summary row per seed.
            if len(df) != 1:
                raise ValueError(
                    f"Expected one metrics row, found {len(df)}: {csv_path}"
                )

            row = df.iloc[0].to_dict()
            row["seed"] = seed

            groups.setdefault(key, []).append({
                "seed": seed,
                "path": str(csv_path),
                "row": row,
            })

    if not groups:
        raise RuntimeError(
            "No per-model LLM metrics found. Check results_dir and experiment."
        )

    output_root = experiment_dir / "averaged" / "in_domain" / "llm"
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    
    for (model, prompt, split), records in sorted(groups.items()):
        # Filter by the model name in the group key, not by fields inside records.
        if args.models and model not in args.models:
            continue

        found_seeds = sorted({r["seed"] for r in records})
        missing_seeds = sorted(set(args.seeds) - set(found_seeds))

        print(
            f"\n{model} | {prompt} | {split}: "
            f"seeds={found_seeds}"
        )
        if missing_seeds:
            print(f"WARNING: Missing seeds: {missing_seeds}")

        df = pd.DataFrame([r["row"] for r in records])

        # Average numeric metrics only; never average the seed identifier.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "seed"]

        if not numeric_cols:
            print("WARNING: No numeric metrics found; skipping.")
            continue

        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std(ddof=1)

        save_dir = output_root / model / prompt / split
        save_dir.mkdir(parents=True, exist_ok=True)

        mean_df = means.to_frame(name="mean")
        std_df = stds.to_frame(name="std")
        mean_std_df = pd.DataFrame({
            "metric": numeric_cols,
            "mean": [means[c] for c in numeric_cols],
            "std": [stds[c] for c in numeric_cols],
            "mean_pm_std": [
                f"{means[c]:.4f} ± {stds[c]:.4f}"
                if pd.notna(stds[c])
                else f"{means[c]:.4f} ± nan"
                for c in numeric_cols
            ],
        })

        mean_df.to_csv(save_dir / "mean.csv")
        std_df.to_csv(save_dir / "std.csv")
        mean_std_df.to_csv(save_dir / "mean_std.csv", index=False)

        metadata = {
            "model": model,
            "prompt_type": prompt,
            "split": split,
            "expected_seeds": args.seeds,
            "seeds_used": found_seeds,
            "missing_seeds": missing_seeds,
            "n_seeds": len(found_seeds),
            "input_files": [r["path"] for r in records],
            "date_averaged": datetime.now().isoformat(timespec="seconds"),
            "note": (
                "Smoke-test results; verify full runs before final reporting."
                if "smoke" in str(experiment_dir).lower()
                else "Verify run completeness before final reporting."
            ),
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for metric in numeric_cols:
            summary_rows.append({
                "model": model,
                "prompt_type": prompt,
                "split": split,
                "metric": metric,
                "mean": means[metric],
                "std": stds[metric],
                "n_seeds": len(found_seeds),
                "seeds_used": ",".join(map(str, found_seeds)),
                "missing_seeds": ",".join(map(str, missing_seeds)),
            })

        print(f"Saved: {save_dir}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_root / "averaging_summary.csv", index=False
        )

    print(f"\nAveraging complete. Output: {output_root}")


if __name__ == "__main__":
    main()