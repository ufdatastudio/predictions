# Self-Rewarding Data Factory for TOLSA (Task 2 Ground-Truth Augmentation)

Adapts **Self-Rewarding Language Models** (Yuan et al., ICML 2024, arXiv:2401.10020)
to the TOLSA framework: **one LLM wears two hats** —

- **Writer** — generates new candidate Task-2 examples: a sentence plus its
  TOLSA property annotation in the project's `{"1": source, "2": target,
  "3": date, "4": outcome}` schema.
- **Judge** — the *same* model scores each candidate with the paper's
  **additive 1–5 rubric**, adapted to the TOLSA schema and the project's human
  annotation guidelines. Each candidate is judged `--judge-runs` times
  (default 3) and the scores are averaged.

Only candidates with **average score ≥ threshold** (default **4.5**, use 5.0
for strict mode) that also pass deterministic checks and dedupe are kept.

## The loop

```
Writer call (batch of N candidates, seeded with gold few-shot examples)
  -> HARD CHECKS (no LLM): valid JSON, schema keys 1-4, every span a verbatim
     substring of the sentence, spans in sentence order, no "|" joins
  -> JUDGE x 3, averaged (additive rubric, per-criterion reasoning)
  -> DEDUPE: exact + near-duplicate (SequenceMatcher >= 0.85) against the
     200 gold sentences AND all previously accepted examples
  -> keep if judge_avg >= threshold
```

Resume-safe: every stage appends to `candidates.jsonl` / `judged.jsonl`.
If the run is interrupted (SLURM time limit, crash), **re-run the exact same
command** — completed candidates are skipped.

## Installation

Same environment as the repo (`requirements.txt`), plus one of:

```bash
pip install groq          # for groq: backends (how gpt-oss-120b is run today)
pip install openai        # for openai-compat: backends (e.g. vLLM on HiPerGator)
pip install transformers torch   # for hf: local backends
```

## Backends (`--backend`)

| Spec | Use |
|---|---|
| `groq:openai/gpt-oss-120b` | Groq Cloud; needs `GROQ_CLOUD_API_KEY` (or `GROQ_API_KEY`) env var |
| `openai-compat:<model>@<base_url>` | Any OpenAI-compatible server, e.g. vLLM on HiPerGator: `openai-compat:openai/gpt-oss-120b@http://127.0.0.1:8000/v1` |
| `hf:<model-id>` | Local Hugging Face model |
| `mock` | Deterministic fake backend for pipeline testing (no GPU/API needed) |

## Commands (run from repo root)

### Step 0 — smoke test (no model needed)

```bash
python self_rewarding/tests/test_end_to_end_mock.py
```

### Step 1 — validate the Judge on human gold (DO THIS FIRST)

If the judge cannot recognise the 200 human-reviewed gold examples as high
quality, it cannot be trusted on new data:

```bash
python -m self_rewarding.run_factory validate-gold \
    --backend groq:openai/gpt-oss-120b \
    --gold data/ground_truth_annotations/ground_truth_final_200.csv \
    --out runs/validation_gold_gpt_oss_120b
# add --limit 10 for a quick smoke test first
```

Report: `gold_validation_report.json` — `gold_pass_rate`, mean/median score,
score distribution, per-example scores. **Expect pass rate ≥ ~0.9** before
proceeding.

### Step 2 — self-enhancement-bias check (recommended, for the paper)

Judge a *different* model's extractions (any 11-column run file, e.g.
mistral's), to show the judge does not favour its own generations:

```bash
python -m self_rewarding.run_factory validate-cross \
    --backend groq:openai/gpt-oss-120b \
    --run-file data/extraction_results/naacl_2026_submission/classification/zero-shot/seed3/mistral-small-3.1/extracted_properties.csv \
    --out runs/validation_cross_mistral
```

### Step 3 — generate ~1200 judge-approved examples

```bash
python -m self_rewarding.run_factory generate \
    --backend groq:openai/gpt-oss-120b \
    --gold data/ground_truth_annotations/ground_truth_final_200.csv \
    --out runs/factory_gpt_oss_120b_seed3 \
    --target 1200 --threshold 4.5 --judge-runs 3 --seed 3
```

Key flags: `--batch-size` (candidates per Writer call, default 10),
`--writer-temperature` (0.9), `--judge-temperature` (0.2),
`--near-dup-ratio` (0.85), `--review-fraction` (0.10 = 10% human-review
sample), `--request-pause-s` (rate-limit safety).

### Outputs (in `--out` directory)

| File | Contents |
|---|---|
| `accepted.csv` | New candidate ground truth. **First 6 columns are exactly the gold layout** (`Input_Index, Base Sentence, Source, Target, Date, Outcome`, spans `\|`-joined) → drop-in compatible; extra columns carry judge scores, domain, reasoning |
| `review_sample.csv` | 10% of accepted examples + empty `Human_Verdict` column for hand review (paste into the Google Sheet workflow) |
| `rejected.csv` | All rejects with `Reject_Reason` (hard check / judge score / duplicate) |
| `candidates.jsonl`, `judged.jsonl` | Full audit trail (raw writer items, all judge responses) |
| `summary.json`, `state.json`, `run_history.jsonl` | Run statistics and resume state |

## Generation mixture (Writer prompt)

Per batch: ~70% full TOLSA (≥ Target + Outcome), ~15% with an explicit named
Source, ~15% **partial** examples that intentionally lack Target and/or
Outcome — these preserve the "extract nothing" test behaviour from the human
annotation protocol. Domains rotate over 12 topic areas; 3 gold few-shot
examples are sampled per call (`--seed` reproducible).

## Judge rubric (additive, Yuan et al. 2024 style)

Start at 1; +1 each for: (1) sentence quality / genuine TOLSA fit, (2) Target
correctness (entity/event/location, noun POS, verbatim), (3) Outcome
correctness (contains attribute/metric/slope, verbatim), (4) Source & Date
correctness (titles kept, explicit/event-relative dates, correct empties),
(5) schema & formatting compliance. The judge reasons through each criterion
before printing `Score: N`.

## HiPerGator

See `hpg/self_rewarding_gpt_oss_120b.sbatch` for a SLURM template that mirrors
the existing `run_llm_gpt_oss_120b-pipeline-ground_truth.sbatch` layout
(venv activation, log dirs). Because the factory is resume-safe, a 24 h
SLURM job that hits the time limit can simply be resubmitted.

## Known limitations (for the paper's Limitations section)

- The judge scores its *own* generations → self-enhancement / reward-hacking
  risk (Step 2 quantifies this cross-model).
- Judge-approved examples are **candidate** ground truth until the human
  review of `review_sample.csv` confirms the agreement rate.
- The Writer can drift toward templated sentences; near-dup filtering (0.85)
  mitigates but does not eliminate this — check domain/structure diversity in
  the accepted set.
