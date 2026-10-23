"""
TOLSA Property-Extraction Evaluation Metrics
=============================================

String- and semantic-similarity metrics for comparing human-annotated
TOLSA properties (y) against LLM-extracted properties (y_hat).

Follows the static-method conventions of the `EvaluationMetric` class in
the ufdatastudio/predictions `metrics.py`.

Metrics
-------
strict        Exact match after normalization (0/1). Primary "strict" metric.
strict_set    Exact match of the *set* of spans (order-insensitive, 0/1).
bleu          sacrebleu sentence BLEU (0-1).
rouge_l       ROUGE-L F1 with stemming (0-1).
chrf          sacrebleu sentence chrF (0-1).
sbert         Sentence-BERT cosine similarity (0-1), lazy backend.
bertscore_f1  BERTScore F1 (0-1), lazy backend.

Design decisions
----------------
* A property value may hold several spans, joined with "|" in the CSVs.
  All string/semantic metrics score the whole property value as one text
  (spans joined by " | "), matching how the POC notebook scored them.
  `strict` is order-sensitive on the joined string; `strict_set` compares
  the unordered span sets, so "A|B" == "B|A".
* Empty-vs-empty is defined as a perfect match (score 1.0); empty-vs-nonempty
  is 0.0. This treats "both agree nothing is extractable" as agreement.
* Neural backends (sbert, bertscore) import lazily and are cached on the
  class so the n-gram metrics work even without torch installed.
"""

from functools import lru_cache
import os

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF


def normalize_text(text):
    """Lowercase, collapse whitespace, strip."""
    return " ".join(str(text).strip().lower().split())


def split_spans(text):
    """Split a multi-span property value on '|' and normalize each span."""
    return [normalize_text(s) for s in str(text).split("|") if normalize_text(s)]


class TolsaExtractionMetric:
    """Static metric functions for TOLSA property extraction evaluation."""

    # ------------------------------------------------------------- strict
    @staticmethod
    def strict(y, y_hat):
        """1.0 if normalized strings match exactly, else 0.0."""
        return 1.0 if normalize_text(y) == normalize_text(y_hat) else 0.0

    @staticmethod
    def strict_set(y, y_hat):
        """1.0 if the unordered span sets are equal, else 0.0."""
        return 1.0 if set(split_spans(y)) == set(split_spans(y_hat)) else 0.0

    # ----------------------------------------------------- empty handling
    @staticmethod
    def _empty_aware(y, y_hat, scorer):
        y_n, y_h = str(y).strip(), str(y_hat).strip()
        if y_n == "" and y_h == "":
            return 1.0
        if y_n == "" or y_h == "":
            return 0.0
        return scorer(y_n, y_h)

    # -------------------------------------------------------- n-gram metrics
    _bleu = BLEU(effective_order=True)
    _chrf = CHRF()
    _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    @staticmethod
    def bleu(y, y_hat):
        """sacrebleu sentence BLEU, normalized to 0-1 (y_hat vs reference y)."""
        return TolsaExtractionMetric._empty_aware(
            y, y_hat,
            lambda p, r: TolsaExtractionMetric._bleu.sentence_score(p, [r]).score / 100.0
        )

    @staticmethod
    def chrf(y, y_hat):
        """sacrebleu sentence chrF, normalized to 0-1 (y_hat vs reference y)."""
        return TolsaExtractionMetric._empty_aware(
            y, y_hat,
            lambda p, r: TolsaExtractionMetric._chrf.sentence_score(p, [r]).score / 100.0
        )

    @staticmethod
    def rouge_l(y, y_hat):
        """ROUGE-L F1 with Porter stemming, 0-1."""
        return TolsaExtractionMetric._empty_aware(
            y, y_hat,
            lambda p, r: TolsaExtractionMetric._rouge.score(r, p)["rougeL"].fmeasure
        )

    # ------------------------------------------------------- neural backends
    _sbert_model = None
    _bertscore_fn = None
    sbert_model_name = "all-MiniLM-L6-v2"
    bertscore_model_name = "roberta-large"  # BERTScore's recommended default
    _neural_cache = None                    # (y, y_hat) -> {"sbert": v, "bertscore_f1": v}
    _neural_cache_path = None

    @classmethod
    def enable_neural_cache(cls, path):
        """Persist neural scores per (y, y_hat) pair across process restarts."""
        import json
        cls._neural_cache_path = path
        cls._neural_cache = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cls._neural_cache = {tuple(k.split("\x1f")): v
                                     for k, v in json.load(f).items()}

    @classmethod
    def _save_neural_cache(cls):
        import json
        if cls._neural_cache_path and cls._neural_cache is not None:
            with open(cls._neural_cache_path, "w", encoding="utf-8") as f:
                json.dump({"\x1f".join(k): v for k, v in cls._neural_cache.items()}, f)

    @classmethod
    def _cached_neural(cls, pairs, metric):
        """Split pairs into (cached values, indices needing computation)."""
        if cls._neural_cache is None:
            return [None] * len(pairs), list(range(len(pairs)))
        vals, todo = [], []
        for i, pair in enumerate(pairs):
            hit = cls._neural_cache.get(pair)
            vals.append(hit.get(metric) if hit else None)
            if hit is None or metric not in hit:
                todo.append(i)
        return vals, todo

    @classmethod
    def _load_sbert(cls):
        if cls._sbert_model is None:
            from sentence_transformers import SentenceTransformer
            cls._sbert_model = SentenceTransformer(cls.sbert_model_name)
        return cls._sbert_model

    @classmethod
    def _load_bertscore(cls):
        if cls._bertscore_fn is None:
            import os
            import torch
            torch.set_num_threads(max(torch.get_num_threads(),
                                      os.cpu_count() or 4))
            from bert_score import score as bert_score_fn
            cls._bertscore_fn = bert_score_fn
        return cls._bertscore_fn

    @staticmethod
    def _empty_mask(y_list, y_hat_list):
        """Return (out_array, compute_idx): 1.0 for both-empty, 0.0 for one-empty."""
        y_n = [str(t).strip() for t in y_list]
        y_h = [str(t).strip() for t in y_hat_list]
        out = np.zeros(len(y_n))
        idx = []
        for i, (a, b) in enumerate(zip(y_n, y_h)):
            if a == "" and b == "":
                out[i] = 1.0
            elif a == "" or b == "":
                out[i] = 0.0
            else:
                idx.append(i)
        return out, idx, y_n, y_h

    @classmethod
    def sbert_batch(cls, y_list, y_hat_list):
        """Sentence-BERT cosine similarity per row (batched, pair-cached)."""
        out, idx, y_n, y_h = cls._empty_mask(y_list, y_hat_list)
        if idx:
            pairs = [(y_n[i], y_h[i]) for i in idx]
            cached, todo = cls._cached_neural(pairs, "sbert")
            if todo:
                model = cls._load_sbert()
                t_idx = [idx[t] for t in todo]
                emb = model.encode([y_n[i] for i in t_idx] + [y_h[i] for i in t_idx],
                                   convert_to_numpy=True)
                n = len(t_idx)
                a, b = emb[:n], emb[n:]
                num = (a * b).sum(axis=1)
                den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                sims = num / np.where(den == 0, 1.0, den)
                for k, t in enumerate(todo):
                    cached[t] = float(sims[k])
                    if cls._neural_cache is not None:
                        cls._neural_cache.setdefault(pairs[t], {})["sbert"] = cached[t]
                cls._save_neural_cache()
            for i, v in zip(idx, cached):
                out[i] = v
        return list(out)

    @classmethod
    def bertscore_f1_batch(cls, y_list, y_hat_list):
        """BERTScore F1 per row (batched, pair-cached)."""
        out, idx, y_n, y_h = cls._empty_mask(y_list, y_hat_list)
        if idx:
            pairs = [(y_n[i], y_h[i]) for i in idx]
            cached, todo = cls._cached_neural(pairs, "bertscore_f1")
            if todo:
                fn = cls._load_bertscore()
                t_idx = [idx[t] for t in todo]
                _, _, f1 = fn([y_h[i] for i in t_idx], [y_n[i] for i in t_idx],
                              lang="en", verbose=False)
                f1 = f1.cpu().numpy()
                for k, t in enumerate(todo):
                    cached[t] = float(f1[k])
                    if cls._neural_cache is not None:
                        cls._neural_cache.setdefault(pairs[t], {})["bertscore_f1"] = cached[t]
                cls._save_neural_cache()
            for i, v in zip(idx, cached):
                out[i] = v
        return list(out)

    # ------------------------------------------------------------ registry
    @staticmethod
    @lru_cache(maxsize=1)
    def ngram_metric_names():
        return ("strict", "strict_set", "bleu", "rouge_l", "chrf")

    @staticmethod
    @lru_cache(maxsize=1)
    def neural_metric_names():
        return ("sbert", "bertscore_f1")

    @staticmethod
    def all_metric_names(include_neural=True):
        names = list(TolsaExtractionMetric.ngram_metric_names())
        if include_neural:
            names += list(TolsaExtractionMetric.neural_metric_names())
        return names

    @staticmethod
    def compute_all(y, y_hat, include_neural=True):
        """Score a single (y, y_hat) pair on every available metric."""
        scores = {
            "strict": TolsaExtractionMetric.strict(y, y_hat),
            "strict_set": TolsaExtractionMetric.strict_set(y, y_hat),
            "bleu": TolsaExtractionMetric.bleu(y, y_hat),
            "rouge_l": TolsaExtractionMetric.rouge_l(y, y_hat),
            "chrf": TolsaExtractionMetric.chrf(y, y_hat),
        }
        if include_neural:
            scores["sbert"] = TolsaExtractionMetric.sbert_batch([y], [y_hat])[0]
            scores["bertscore_f1"] = TolsaExtractionMetric.bertscore_f1_batch([y], [y_hat])[0]
        return scores

    # ------------------------------------------------------- parse error rate
    @staticmethod
    def parse_error_rate(parse_status_series):
        """Fraction of rows whose Parse Status is not OK."""
        status = parse_status_series.astype(str).str.strip().str.upper()
        return float((status != "OK").mean())


# ---------------------------------------------------------------------------
# Convenience alias matching repo naming style
# ---------------------------------------------------------------------------
EvaluationMetricTOLSA = TolsaExtractionMetric
