"""Duplicate and near-duplicate detection.

New candidates are checked against (a) the existing gold dataset and (b) the
pool of already-accepted synthetic examples, so the augmented set stays
distinct from human ground truth and from itself.
"""

from __future__ import annotations

import re
import string
from difflib import SequenceMatcher

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(sentence: str) -> str:
    return re.sub(r"\s+", " ",
                  sentence.lower().translate(_PUNCT_TABLE)).strip()


class DedupeFilter:
    def __init__(self, near_dup_ratio: float = 0.85):
        self.near_dup_ratio = near_dup_ratio
        self._exact: set[str] = set()
        self._normalized: list[str] = []

    def add(self, sentence: str) -> None:
        norm = normalize(sentence)
        self._exact.add(norm)
        self._normalized.append(norm)

    def is_duplicate(self, sentence: str) -> tuple[bool, str | None]:
        norm = normalize(sentence)
        if norm in self._exact:
            return True, "exact duplicate (normalized)"
        for existing in self._normalized:
            # Cheap length pre-filter before the quadratic SequenceMatcher.
            if abs(len(existing) - len(norm)) > 0.3 * max(len(existing),
                                                          len(norm), 1):
                continue
            if SequenceMatcher(None, norm, existing).ratio() >= \
                    self.near_dup_ratio:
                return True, f"near-duplicate (ratio>={self.near_dup_ratio})"
        return False, None
