"""The self-rewarding data factory loop.

    generate N candidates (Writer)  ->
    hard programmatic checks         ->
    judge 3x, average score (Judge) ->
    keep if avg >= threshold         ->
    dedupe vs gold + accepted pool   ->
    resume-safe JSONL checkpoints

Every stage appends to JSONL files, so a run can be interrupted (SLURM time
limit, crash) and resumed with the same command: completed work is skipped.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

from .backends import LLMBackend
from .checks import (check_candidate, parse_judge_score,
                     parse_writer_output)
from .dedupe import DedupeFilter, normalize
from .prompts import DOMAINS, JudgePrompt, WriterPrompt

SPAN_KEYS = ["1", "2", "3", "4"]  # 1=Source, 2=Target, 3=Date, 4=Outcome
GOLD_COLUMNS = ["Input_Index", "Base Sentence", "Source", "Target", "Date",
                "Outcome"]


def spans_to_row(spans: dict) -> dict:
    """Map {"1": [...], ...} to the gold CSV column layout ('|'-joined)."""
    return {
        "Source": "|".join(spans.get("1", [])),
        "Target": "|".join(spans.get("2", [])),
        "Date": "|".join(spans.get("3", [])),
        "Outcome": "|".join(spans.get("4", [])),
    }


def row_to_spans(row: dict) -> dict:
    """Inverse of spans_to_row for the gold CSV format."""
    def split(v):
        v = (v or "").strip()
        return [s for s in v.split("|") if s] if v else []
    return {"1": split(row.get("Source")), "2": split(row.get("Target")),
            "3": split(row.get("Date")), "4": split(row.get("Outcome"))}


def load_gold_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class SelfRewardingFactory:
    def __init__(
        self,
        backend: LLMBackend,
        gold_path: str | Path,
        out_dir: str | Path,
        target_count: int = 1200,
        batch_size: int = 10,
        judge_runs: int = 3,
        threshold: float = 4.5,
        seed: int = 3,
        few_shot_k: int = 3,
        writer_temperature: float = 0.9,
        judge_temperature: float = 0.2,
        near_dup_ratio: float = 0.85,
        max_batches: int = 10000,
        review_fraction: float = 0.10,
        request_pause_s: float = 0.0,
    ):
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_count = target_count
        self.batch_size = batch_size
        self.judge_runs = judge_runs
        self.threshold = threshold
        self.seed = seed
        self.few_shot_k = few_shot_k
        self.writer_temperature = writer_temperature
        self.judge_temperature = judge_temperature
        self.max_batches = max_batches
        self.review_fraction = review_fraction
        self.request_pause_s = request_pause_s

        self.gold = load_gold_csv(gold_path)
        self.rng = random.Random(seed)

        self.dedupe = DedupeFilter(near_dup_ratio=near_dup_ratio)
        for row in self.gold:
            self.dedupe.add(row["Base Sentence"])

        # Checkpoint files
        self.candidates_path = self.out_dir / "candidates.jsonl"
        self.judged_path = self.out_dir / "judged.jsonl"
        self.state_path = self.out_dir / "state.json"

        # Resume: reload already-judged candidates, rebuild accepted pool.
        self._judged = {r["candidate_id"]: r for r in
                        _load_jsonl(self.judged_path)}
        self._next_candidate_id = 0
        for r in _load_jsonl(self.candidates_path):
            self._next_candidate_id = max(self._next_candidate_id,
                                          r["candidate_id"] + 1)
        for r in self._judged.values():
            if r["accepted"]:
                self.dedupe.add(r["sentence"])
        if self._judged:
            print(f"[resume] {len(self._judged)} judged candidates reloaded, "
                  f"{sum(1 for r in self._judged.values() if r['accepted'])} "
                  f"already accepted")

    # ---------------------------------------------------------------
    def _accepted_count(self) -> int:
        return sum(1 for r in self._judged.values() if r["accepted"])

    def _sample_few_shot(self) -> list:
        rows = self.rng.sample(self.gold,
                               k=min(self.few_shot_k, len(self.gold)))
        return [(r["Base Sentence"], row_to_spans(r)) for r in rows]

    def _generate_batch(self, batch_idx: int) -> int:
        """One Writer call -> hard checks -> candidates.jsonl. Returns the
        number of candidates that passed hard checks."""
        domain = DOMAINS[batch_idx % len(DOMAINS)]
        few_shot = self._sample_few_shot()
        prompt = WriterPrompt.build(self.batch_size, few_shot, domain)
        raw = self.backend.generate(
            WriterPrompt.system_identity(), prompt,
            temperature=self.writer_temperature, seed=self.seed + batch_idx)
        items, err = parse_writer_output(raw)
        if err:
            print(f"[batch {batch_idx}] writer parse failure: {err}")
            return 0
        kept = 0
        for item in items:
            cid = self._next_candidate_id
            self._next_candidate_id += 1
            sentence = str(item.get("sentence", ""))
            spans = item.get("spans", {})
            errors = check_candidate(sentence, spans)
            rec = {
                "candidate_id": cid, "batch": batch_idx, "domain": domain,
                "sentence": sentence,
                "spans": {k: spans.get(k, []) for k in SPAN_KEYS},
                "is_tolsa": bool(item.get("is_tolsa", True)),
                "hard_check_errors": errors,
                "raw_writer_item": item,
            }
            _append_jsonl(self.candidates_path, rec)
            if not errors:
                kept += 1
        return kept

    def _judge_candidate(self, rec: dict) -> dict:
        scores, raw_judges = [], []
        for j in range(self.judge_runs):
            prompt = JudgePrompt.build(rec["sentence"], rec["spans"])
            raw = self.backend.generate(
                JudgePrompt.system_identity(), prompt,
                temperature=self.judge_temperature,
                seed=self.seed + 1000 * j + rec["candidate_id"])
            raw_judges.append(raw)
            score, err = parse_judge_score(raw)
            if score is not None:
                scores.append(score)
            else:
                print(f"[judge] candidate {rec['candidate_id']} run {j}: {err}")
            if self.request_pause_s:
                time.sleep(self.request_pause_s)
        avg = sum(scores) / len(scores) if scores else 0.0
        is_dup, dup_reason = self.dedupe.is_duplicate(rec["sentence"])
        accepted = (scores and avg >= self.threshold and not is_dup)
        result = {
            **rec,
            "judge_scores": scores,
            "judge_avg": avg,
            "judge_raw": raw_judges,
            "duplicate": is_dup,
            "duplicate_reason": dup_reason,
            "accepted": bool(accepted),
        }
        if accepted:
            self.dedupe.add(rec["sentence"])
        return result

    # ---------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        pending = [r for r in _load_jsonl(self.candidates_path)
                   if r["candidate_id"] not in self._judged
                   and not r["hard_check_errors"]]
        for rec in pending:
            if self._accepted_count() >= self.target_count:
                break
            judged = self._judge_candidate(rec)
            _append_jsonl(self.judged_path, judged)
            self._judged[judged["candidate_id"]] = judged

        batch_idx = len({r["batch"] for r in self._judged.values()}
                        or {0})
        while (self._accepted_count() < self.target_count
               and batch_idx < self.max_batches):
            self._generate_batch(batch_idx)
            new_recs = [r for r in _load_jsonl(self.candidates_path)
                        if r["candidate_id"] not in self._judged
                        and not r["hard_check_errors"]]
            for rec in new_recs:
                if self._accepted_count() >= self.target_count:
                    break
                judged = self._judge_candidate(rec)
                _append_jsonl(self.judged_path, judged)
                self._judged[judged["candidate_id"]] = judged
            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"[progress] batch {batch_idx}: "
                      f"{self._accepted_count()}/{self.target_count} accepted")
            self._write_state(batch_idx)

        summary = self.export()
        summary["wall_time_s"] = round(time.time() - t0, 1)
        _append_jsonl(self.out_dir / "run_history.jsonl",
                      {"event": "run_finished", **summary})
        return summary

    # ---------------------------------------------------------------
    def _write_state(self, batch_idx: int) -> None:
        self.state_path.write_text(json.dumps({
            "batch_idx": batch_idx,
            "candidates": len(_load_jsonl(self.candidates_path)),
            "judged": len(self._judged),
            "accepted": self._accepted_count(),
            "target_count": self.target_count,
        }, indent=2), encoding="utf-8")

    def export(self) -> dict:
        accepted = [r for r in self._judged.values() if r["accepted"]]
        rejected = [r for r in self._judged.values() if not r["accepted"]]
        accepted.sort(key=lambda r: r["candidate_id"])

        def to_row(r, idx):
            row = {"Input_Index": idx, "Base Sentence": r["sentence"]}
            row.update(spans_to_row(r["spans"]))
            row.update({
                "Judge_Avg": round(r["judge_avg"], 3),
                "Judge_Scores": "|".join(map(str, r["judge_scores"])),
                "Is_TOLSA": r["is_tolsa"],
                "Domain": r["domain"],
                "Judge_Reasoning_Last_Run": (r["judge_raw"][-1] if
                                             r["judge_raw"] else ""),
            })
            return row

        cols = GOLD_COLUMNS + ["Judge_Avg", "Judge_Scores", "Is_TOLSA",
                               "Domain", "Judge_Reasoning_Last_Run"]

        accepted_path = self.out_dir / "accepted.csv"
        with open(accepted_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i, r in enumerate(accepted):
                w.writerow(to_row(r, i))

        # 10% human-review sample (stratified across the run).
        review_path = self.out_dir / "review_sample.csv"
        n_review = max(1, int(len(accepted) * self.review_fraction))
        review_idx = set(self.rng.sample(range(len(accepted)),
                                         k=min(n_review, len(accepted))))
        with open(review_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols + ["Human_Verdict"])
            w.writeheader()
            for i, r in enumerate(accepted):
                if i in review_idx:
                    row = to_row(r, i)
                    row["Human_Verdict"] = ""  # fill in during review
                    w.writerow(row)

        # Rejected = judged-but-not-accepted PLUS hard-check failures (which
        # were never sent to the judge).
        hard_failed = [r for r in _load_jsonl(self.candidates_path)
                       if r["hard_check_errors"]]
        all_rejected = list(rejected) + [
            {**r, "judge_scores": [], "judge_avg": 0.0, "judge_raw": [],
             "duplicate": False, "duplicate_reason": None}
            for r in hard_failed]
        rejected_path = self.out_dir / "rejected.csv"
        with open(rejected_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols + ["Reject_Reason"])
            w.writeheader()
            for i, r in enumerate(all_rejected):
                row = to_row(r, i)
                reasons = []
                if r["hard_check_errors"]:
                    reasons += r["hard_check_errors"]
                if r["duplicate"]:
                    reasons.append(r["duplicate_reason"] or "duplicate")
                if r["judge_scores"] and r["judge_avg"] < self.threshold:
                    reasons.append(f"judge_avg {r['judge_avg']:.2f} < "
                                   f"{self.threshold}")
                if not r["judge_scores"]:
                    reasons.append("no parseable judge score")
                row["Reject_Reason"] = " ; ".join(reasons)
                w.writerow(row)

        summary = {
            "backend": self.backend.name,
            "target_count": self.target_count,
            "accepted": len(accepted),
            "rejected": len(all_rejected),
            "hard_check_failures": len(hard_failed),
            "threshold": self.threshold,
            "judge_runs": self.judge_runs,
            "accepted_path": str(accepted_path),
            "review_path": str(review_path),
            "rejected_path": str(rejected_path),
        }
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        return summary
