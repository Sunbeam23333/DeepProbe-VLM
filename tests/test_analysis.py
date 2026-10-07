import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("analysis_summary", Path(__file__).parents[1] / "scripts/analysis/summarize.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def row(i, source, correct):
    return {"sample_id": str(i), "source_video_id": source, "correct": correct,
            "answer": "A", "metric": "accuracy", "split": "test"}


def test_source_cluster_bootstrap_is_deterministic():
    rows = [row(1, "a", 1), row(2, "a", 0), row(3, "b", 1)]
    a = summary.bootstrap(rows, repeats=100)
    assert a == summary.bootstrap(rows, repeats=100)
    assert a["estimate"] == pytest.approx(2 / 3)
    assert a["sources"] == 2


def test_paired_delta_and_missing_sample_rejection():
    now = [row(1, "a", 1), row(2, "b", 1)]
    old = [row(1, "a", 0), row(2, "b", 1)]
    result = summary.bootstrap(summary.paired_rows(now, old), "delta", repeats=100)
    assert result["estimate"] == 0.5
    with pytest.raises(ValueError, match="same sample"):
        summary.paired_rows(now, old[:1])


def test_single_source_no_false_ci():
    assert summary.bootstrap([row(1, "a", 1)])["ci95"] is None
