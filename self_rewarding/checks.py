"""Programmatic (non-LLM) checks applied to every Writer candidate before the
LLM Judge is consulted. Hard failures here reject the candidate immediately,
saving judge calls and catching structural errors deterministically.
"""

from __future__ import annotations

import json
import re

VALID_KEYS = {"1", "2", "3", "4"}


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_writer_output(raw: str) -> tuple[list[dict], str | None]:
    """Parse the writer's JSON array. Returns (items, error)."""
    raw = strip_code_fences(raw)
    # Tolerate prose around the JSON array.
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return [], "no JSON array found in writer output"
    try:
        items = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        return [], f"JSON decode error: {e}"
    if not isinstance(items, list):
        return [], "writer output is not a JSON array"
    return [it for it in items if isinstance(it, dict)], None


def check_candidate(sentence: str, spans: dict) -> list[str]:
    """Return a list of hard-check failure reasons (empty = pass)."""
    errors = []
    if not isinstance(sentence, str) or not sentence.strip():
        return ["empty sentence"]
    if len(sentence) < 20:
        errors.append("sentence suspiciously short (<20 chars)")
    if len(sentence) > 600:
        errors.append("sentence too long (>600 chars)")
    if not isinstance(spans, dict):
        return ["spans is not a dict"]
    bad_keys = set(spans) - VALID_KEYS
    if bad_keys:
        errors.append(f"unexpected span keys: {sorted(bad_keys)}")
    for key in VALID_KEYS:
        values = spans.get(key, [])
        if not isinstance(values, list):
            errors.append(f"spans[{key!r}] is not a list")
            continue
        for span in values:
            if not isinstance(span, str) or not span.strip():
                errors.append(f"empty or non-string span in key {key!r}")
                continue
            # Verbatim check: exact wording incl. punctuation/capitalization.
            if span not in sentence:
                errors.append(f"span not verbatim in sentence: {span!r}")
            if "|" in span:
                errors.append(f"span contains '|' (should be separate items): "
                              f"{span!r}")
    # Order check: spans within each list must appear in sentence order.
    for key in VALID_KEYS:
        values = spans.get(key, [])
        if not isinstance(values, list):
            continue
        positions = [sentence.find(s) for s in values
                     if isinstance(s, str) and s in sentence]
        if positions != sorted(positions):
            errors.append(f"spans in key {key!r} are not in sentence order")
    return errors


_SCORE_RE = re.compile(r"Score:\s*([1-5])(?:\s*/\s*5)?\b", re.IGNORECASE)


def parse_judge_score(raw: str) -> tuple[int | None, str | None]:
    """Extract 'Score: N' from the judge response. Returns (score, error)."""
    matches = _SCORE_RE.findall(raw)
    if not matches:
        return None, "no 'Score: N' line found in judge output"
    return int(matches[-1]), None  # last occurrence = final score line
