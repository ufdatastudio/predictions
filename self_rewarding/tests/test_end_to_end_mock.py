"""End-to-end smoke test of the factory and validation with MockBackend.

Run from the workspace/repo root:
    python self_rewarding/tests/test_end_to_end_mock.py

Verifies: generation, hard-check rejection, judge scoring, threshold
filtering, dedupe, resume after interruption, CSV exports, gold validation.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from self_rewarding.backends import MockBackend
from self_rewarding.pipeline import SelfRewardingFactory
from self_rewarding.validate import validate_gold

# Gold CSV resolution order:
#   1. GOLD_CSV environment variable
#   2. repo location (data/ground_truth_annotations/, not committed -
#      Detravious distributes it separately)
# If neither exists, the test skips cleanly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD = Path(os.environ["GOLD_CSV"]) if os.environ.get("GOLD_CSV") else \
    _REPO_ROOT / "data" / "ground_truth_annotations" / \
    "ground_truth_final_200.csv"


def main():
    if not GOLD.exists():
        print(f"SKIPPED: gold CSV not found at {GOLD}\n"
              f"Set GOLD_CSV=/path/to/ground_truth_final_200.csv or place it "
              f"under data/ground_truth_annotations/.")
        return
    tmp = Path(tempfile.mkdtemp(prefix="sr_test_"))
    out = tmp / "run1"
    backend = MockBackend(per_call=5)

    factory = SelfRewardingFactory(
        backend=backend, gold_path=GOLD, out_dir=out,
        target_count=12, batch_size=5, judge_runs=3, threshold=4.5, seed=3)
    summary = factory.run()

    assert summary["accepted"] >= 12, summary
    print("summary:", json.dumps({k: v for k, v in summary.items()
                                  if not k.endswith("path")}, indent=2))

    # Export files exist and accepted.csv has the gold column layout first.
    accepted = (out / "accepted.csv").read_text(encoding="utf-8").splitlines()
    header = accepted[0].split(",")
    assert header[:6] == ["Input_Index", "Base Sentence", "Source", "Target",
                          "Date", "Outcome"], header
    assert (out / "review_sample.csv").exists()
    assert (out / "rejected.csv").exists()
    assert (out / "summary.json").exists()

    # Mock: every 4th candidate has a non-verbatim span -> hard-check rejects.
    rejected_txt = (out / "rejected.csv").read_text(encoding="utf-8")
    assert "not verbatim" in rejected_txt

    # Resume: a second factory over the same out_dir must reload judged work
    # and accept 0 additional when target is already met.
    factory2 = SelfRewardingFactory(
        backend=MockBackend(per_call=5), gold_path=GOLD, out_dir=out,
        target_count=12, batch_size=5, judge_runs=3, threshold=4.5, seed=3)
    assert factory2._accepted_count() >= 12
    summary2 = factory2.run()
    assert summary2["accepted"] == summary["accepted"], (summary2, summary)

    # Dedupe: gold sentences must never appear in accepted.csv.
    gold_sentences = set()
    import csv
    with open(GOLD, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold_sentences.add(row["Base Sentence"].strip())
    with open(out / "accepted.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert row["Base Sentence"].strip() not in gold_sentences

    # Validation on a small gold slice.
    vdir = tmp / "val"
    report = validate_gold(MockBackend(), GOLD, vdir, judge_runs=3,
                           threshold=4.5, limit=10)
    assert report["n_gold"] == 10
    assert report["n_scored"] == 10
    print("gold validation (mock):", {k: v for k, v in report.items()
                                      if k != "per_example"})

    shutil.rmtree(tmp)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
