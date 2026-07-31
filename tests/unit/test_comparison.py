"""Paired bootstrap run-comparison tests."""

from __future__ import annotations

import json

import pytest

from memcity.reporting.comparison import compare_runs


def _make_run(tmp_path, name, rows, *, method="method", query_ids=None):
    run_dir = tmp_path / name
    run_dir.mkdir()
    metadata = {
        "dataset": "fixture",
        "dataset_hash": "same-hash",
        "top_ks": [1, 10],
        "evaluation_protocol": "retrieval-episode-id-v1",
        "method": method,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metadata), encoding="utf-8")
    ids = query_ids or [f"q{index}" for index in range(len(rows))]
    with (run_dir / "retrieval_results_42.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id, retrieved in zip(ids, rows, strict=True):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "ground_truth": [f"gold-{sample_id}"],
                        "retrieved": retrieved(sample_id),
                    }
                )
                + "\n"
            )
    (run_dir / "failures_42.jsonl").write_text("", encoding="utf-8")
    return run_dir


def _hit(sample_id):
    return [f"gold-{sample_id}"]


def _miss(sample_id):
    return [f"wrong-{sample_id}"]


def test_identical_runs_have_zero_delta(tmp_path):
    first = _make_run(tmp_path, "first", [_hit, _miss, _hit])
    second = _make_run(tmp_path, "second", [_hit, _miss, _hit])

    result = compare_runs(first, second, bootstrap_resamples=300, seed=7)

    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["tie_count"] == 3


def test_second_run_always_better(tmp_path):
    first = _make_run(tmp_path, "weak", [_miss, _miss, _miss], method="weak")
    second = _make_run(tmp_path, "strong", [_hit, _hit, _hit], method="strong")

    result = compare_runs(first, second, bootstrap_resamples=300, seed=42)

    assert result["mean_delta"] == pytest.approx(1.0)
    assert result["win_count"] == 3
    assert result["loss_count"] == 0


def test_query_id_mismatch_is_rejected(tmp_path):
    first = _make_run(tmp_path, "first", [_hit], query_ids=["q1"])
    second = _make_run(tmp_path, "second", [_hit], query_ids=["q2"])

    with pytest.raises(ValueError, match="Query IDs do not match"):
        compare_runs(first, second, bootstrap_resamples=100)


def test_comparison_is_reproducible_with_same_seed(tmp_path):
    first = _make_run(tmp_path, "first", [_hit, _miss, _miss, _hit])
    second = _make_run(tmp_path, "second", [_hit, _hit, _miss, _miss])

    one = compare_runs(first, second, bootstrap_resamples=500, seed=99)
    two = compare_runs(first, second, bootstrap_resamples=500, seed=99)

    assert one == two
