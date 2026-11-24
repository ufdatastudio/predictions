"""CLI entry point for the TOLSA self-rewarding data factory.

Usage (from the repo root):

  # 1. Validate the judge on the human gold set FIRST:
  python -m self_rewarding.run_factory validate-gold \
      --backend groq:openai/gpt-oss-120b \
      --gold data/ground_truth_annotations/ground_truth_final_200.csv \
      --out runs/validation_gold

  # 2. Optional: cross-model self-enhancement-bias check on another model's
  #    run file (e.g. mistral):
  python -m self_rewarding.run_factory validate-cross \
      --backend groq:openai/gpt-oss-120b \
      --run-file data/extraction_results/.../mistral-small-3.1/extracted_properties.csv \
      --out runs/validation_cross_mistral

  # 3. Generate ~1200 new judge-approved examples (resume-safe; re-run the
  #    same command after interruption to continue where it left off):
  python -m self_rewarding.run_factory generate \
      --backend groq:openai/gpt-oss-120b \
      --gold data/ground_truth_annotations/ground_truth_final_200.csv \
      --out runs/factory_gpt_oss_120b_seed3 \
      --target 1200 --threshold 4.5 --judge-runs 3 --seed 3
"""

from __future__ import annotations

import argparse
import json
import sys

from .backends import make_backend
from .pipeline import SelfRewardingFactory
from .validate import validate_cross_model, validate_gold


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="self_rewarding.run_factory",
        description="Self-rewarding data factory for TOLSA Task 2 "
                    "ground-truth augmentation.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--backend", default="mock",
                       help="Backend spec, e.g. groq:openai/gpt-oss-120b, "
                            "openai-compat:<model>@<base_url>, hf:<id>, mock")
        p.add_argument("--judge-runs", type=int, default=3,
                       help="Judge repetitions per candidate; scores are "
                            "averaged (Yuan et al. 2024 used 3).")
        p.add_argument("--threshold", type=float, default=4.5,
                       help="Minimum average judge score to accept (4.5 or "
                            "5.0 recommended).")
        p.add_argument("--seed", type=int, default=3)
        p.add_argument("--out", required=True, help="Output run directory.")

    g = sub.add_parser("generate", help="Generate judge-approved examples.")
    add_common(g)
    g.add_argument("--gold", required=True,
                   help="Path to gold CSV (e.g. ground_truth_final_200.csv); "
                        "used for few-shot seeds and dedupe.")
    g.add_argument("--target", type=int, default=1200,
                   help="Number of accepted examples to produce.")
    g.add_argument("--batch-size", type=int, default=10,
                   help="Candidates requested per Writer call.")
    g.add_argument("--writer-temperature", type=float, default=0.9)
    g.add_argument("--judge-temperature", type=float, default=0.2)
    g.add_argument("--few-shot-k", type=int, default=3,
                   help="Gold few-shot examples per Writer call.")
    g.add_argument("--near-dup-ratio", type=float, default=0.85)
    g.add_argument("--review-fraction", type=float, default=0.10,
                   help="Fraction of accepted examples exported for human "
                        "review (default 10%%).")
    g.add_argument("--request-pause-s", type=float, default=0.0,
                   help="Pause between judge calls (rate-limit safety).")

    vg = sub.add_parser("validate-gold",
                        help="Judge the human gold set; reports pass rate.")
    add_common(vg)
    vg.add_argument("--gold", required=True)
    vg.add_argument("--limit", type=int, default=None,
                    help="Judge only the first N gold rows (smoke test).")

    vc = sub.add_parser("validate-cross",
                        help="Judge another model's run file "
                             "(self-enhancement-bias check).")
    add_common(vc)
    vc.add_argument("--run-file", required=True,
                    help="An extracted_properties.csv run file from a "
                         "different model.")
    vc.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    backend = make_backend(args.backend)
    print(f"Backend: {backend.name}")

    if args.command == "generate":
        factory = SelfRewardingFactory(
            backend=backend,
            gold_path=args.gold,
            out_dir=args.out,
            target_count=args.target,
            batch_size=args.batch_size,
            judge_runs=args.judge_runs,
            threshold=args.threshold,
            seed=args.seed,
            few_shot_k=args.few_shot_k,
            writer_temperature=args.writer_temperature,
            judge_temperature=args.judge_temperature,
            near_dup_ratio=args.near_dup_ratio,
            review_fraction=args.review_fraction,
            request_pause_s=args.request_pause_s,
        )
        summary = factory.run()
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "validate-gold":
        report = validate_gold(
            backend, args.gold, args.out, judge_runs=args.judge_runs,
            threshold=args.threshold, seed=args.seed, limit=args.limit)
        report = {k: v for k, v in report.items() if k != "per_example"}
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "validate-cross":
        report = validate_cross_model(
            backend, args.run_file, args.out, judge_runs=args.judge_runs,
            threshold=args.threshold, seed=args.seed, limit=args.limit)
        print(json.dumps(report, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
