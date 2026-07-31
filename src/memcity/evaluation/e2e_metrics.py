"""End-to-end evaluation metrics for the reader layer."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalise(pred) == _normalise(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalise(pred).split()
    gold_tokens = _normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = sum(1 for t in pred_tokens if t in common) / len(pred_tokens)
    rec = sum(1 for t in gold_tokens if t in common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def compute_e2e_metrics(
    results: list[dict],
    samples: list[Any],
) -> dict[str, float]:
    """Compute all end-to-end metrics across a set of reader results.

    Args:
        results: list of serialised ReaderResult dicts.
        samples: list of QASample (or dicts with .answer, .category, .question_category,
                 .evidence_episode_ids, .expected_current_fact).

    Returns:
        dict of metric_name → float value.
    """
    # Build lookup by sample_id
    sample_map: dict[str, Any] = {}
    for s in samples:
        sid = s.sample_id if hasattr(s, "sample_id") else s.get("sample_id", "")
        sample_map[sid] = s

    em_vals, f1_vals = [], []
    abstain_tp = abstain_fp = abstain_fn = 0
    cite_prec_vals, cite_rec_vals = [], []
    grounded = unsupported = schema_ok = 0
    schema_total = 0
    latency_vals: list[float] = []
    prompt_tokens_total = completion_tokens_total = 0
    n = len(results)

    by_category: dict[str, list[float]] = defaultdict(list)

    for res in results:
        sample_id = res.get("sample_id", "")
        sample = sample_map.get(sample_id)
        if sample is None:
            continue

        def _getfield(obj: Any, attr: str, default: Any = "") -> Any:
            """Get a field from either a Pydantic model or a dict."""
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        gold_answer = _getfield(sample, "answer", "")
        gold_ids = _getfield(sample, "evidence_episode_ids", [])
        cat_obj = _getfield(sample, "category", None)
        if cat_obj is None:
            cat_obj = _getfield(sample, "question_category", None)
        if cat_obj is not None and hasattr(cat_obj, "value"):
            category = cat_obj.value
        elif isinstance(cat_obj, str):
            category = cat_obj
        else:
            category = ""
        is_abstention = category == "abstention"

        schema_total += 1
        answer_obj = res.get("answer")
        if res.get("schema_ok"):
            schema_ok += 1

        latency_vals.append(res.get("latency_ms", 0.0))
        prompt_tokens_total += res.get("prompt_tokens", 0)
        completion_tokens_total += res.get("completion_tokens", 0)

        if not answer_obj:
            em_vals.append(0.0)
            f1_vals.append(0.0)
            if is_abstention:
                abstain_fn += 1
            unsupported += 1
            continue

        pred_answer = answer_obj.get("answer", "") if isinstance(answer_obj, dict) else answer_obj.answer
        pred_abstained = answer_obj.get("abstained", False) if isinstance(answer_obj, dict) else answer_obj.abstained
        pred_ids = answer_obj.get("evidence_ids", []) if isinstance(answer_obj, dict) else answer_obj.evidence_ids

        # Exact match / token F1
        if pred_abstained:
            em_v = 1.0 if is_abstention else 0.0
            f1_v = em_v
        else:
            em_v = exact_match(pred_answer, gold_answer)
            f1_v = token_f1(pred_answer, gold_answer)

        em_vals.append(em_v)
        f1_vals.append(f1_v)
        by_category[category].append(em_v)

        # Abstention metrics
        if is_abstention:
            if pred_abstained:
                abstain_tp += 1
            else:
                abstain_fn += 1
        else:
            if pred_abstained:
                abstain_fp += 1

        # Citation metrics
        if gold_ids:
            cited = set(pred_ids)
            relevant = set(gold_ids)
            tp_cite = len(cited & relevant)
            cite_prec = tp_cite / len(cited) if cited else 0.0
            cite_rec = tp_cite / len(relevant) if relevant else 0.0
            cite_prec_vals.append(cite_prec)
            cite_rec_vals.append(cite_rec)

        # Groundedness
        if not pred_abstained:
            if pred_ids:
                grounded += 1
            else:
                unsupported += 1

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _pct(num: int, den: int) -> float:
        return num / den if den else 0.0

    non_abstain_count = sum(1 for r in results if not (
        r.get("answer") and (
            r["answer"].get("abstained", False)
            if isinstance(r["answer"], dict)
            else getattr(r["answer"], "abstained", False)
        )
    ))

    abstain_prec = _pct(abstain_tp, abstain_tp + abstain_fp)
    abstain_rec = _pct(abstain_tp, abstain_tp + abstain_fn)
    abstain_f1 = (
        2 * abstain_prec * abstain_rec / (abstain_prec + abstain_rec)
        if (abstain_prec + abstain_rec) else 0.0
    )

    latency_vals_sorted = sorted(latency_vals)

    def _pct_rank(vals: list[float], pct: float) -> float:
        if not vals:
            return 0.0
        idx = max(0, int(pct * len(vals)) - 1)
        return vals[idx]

    metrics: dict[str, float] = {
        "exact_match": _avg(em_vals),
        "token_f1": _avg(f1_vals),
        "abstention_precision": abstain_prec,
        "abstention_recall": abstain_rec,
        "abstention_f1": abstain_f1,
        "evidence_citation_precision": _avg(cite_prec_vals),
        "evidence_citation_recall": _avg(cite_rec_vals),
        "grounded_answer_rate": _pct(grounded, non_abstain_count),
        "unsupported_answer_rate": _pct(unsupported, non_abstain_count),
        "json_schema_success_rate": _pct(schema_ok, schema_total),
        "reader_latency_p50_ms": _pct_rank(latency_vals_sorted, 0.5),
        "reader_latency_p95_ms": _pct_rank(latency_vals_sorted, 0.95),
        "prompt_tokens_total": float(prompt_tokens_total),
        "completion_tokens_total": float(completion_tokens_total),
        "n_samples": float(n),
    }

    # Per-category exact match
    for cat, vals in by_category.items():
        metrics[f"em_{cat}"] = _avg(vals)

    return metrics
