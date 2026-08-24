"""Red tests for R2 Bible-QA pilot analysis tooling (``humanbrain.qa_analysis``).

Covers:

- Wilson score intervals (shared z with run_pilot; preregistered baseline
  proportions reproduce their intervals);
- Newcombe method-10 hybrid-score intervals on proportion deltas (point
  containment, degenerate limits, separation behaviour);
- degeneracy-guard verdicts incl. majority_floor failing by construction;
- end-to-end analysis over a deterministic synthetic pilot record against the
  REAL committed baselines, plus FINDINGS markdown rendering;
- fixture-synthesis determinism and schema completeness;
- the CLI wrapper (--synthesize-fixture / full run).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from humanbrain import qa_analysis

ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = ROOT / "experiments" / "qa_baselines_bible_v1"


# ---------------------------------------------------------------------------
# Wilson intervals
# ---------------------------------------------------------------------------


def _reference_wilson(correct: int, n: int, z: float = qa_analysis.Z_95) -> tuple[float, float]:
    """Independent restatement of the Wilson formula (drift tripwire)."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@pytest.mark.parametrize(
    ("correct", "n"), [(29, 639), (104, 639), (132, 639), (0, 60), (60, 60)]
)
def test_wilson_matches_reference_formula(correct: int, n: int) -> None:
    got = qa_analysis.wilson(correct, n)
    low, high = _reference_wilson(correct, n)
    assert got["low"] == pytest.approx(low, abs=1e-12)
    assert got["high"] == pytest.approx(high, abs=1e-12)


def test_wilson_preregistered_baseline_accuracy_is_inside_its_interval() -> None:
    # Preregistered comparators: unigram 4.5383%, trigram 16.2754% span-acc.
    unigram = qa_analysis.wilson(29, 639)
    trigram = qa_analysis.wilson(104, 639)
    assert unigram["low"] <= 29 / 639 <= unigram["high"]
    assert trigram["low"] <= 104 / 639 <= trigram["high"]
    assert 0.031 <= unigram["low"] and unigram["high"] <= 0.066
    assert 0.135 <= trigram["low"] and trigram["high"] <= 0.195


def test_wilson_intervals_shrink_with_sample_size() -> None:
    small = qa_analysis.wilson(5, 50)
    large = qa_analysis.wilson(50, 500)
    assert (large["high"] - large["low"]) < (small["high"] - small["low"])


def test_wilson_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        qa_analysis.wilson(7, 5)
    with pytest.raises(ValueError):
        qa_analysis.wilson(-1, 10)


# ---------------------------------------------------------------------------
# Newcombe method-10 delta intervals
# ---------------------------------------------------------------------------


def test_newcombe_point_contained_and_ordered() -> None:
    ci = qa_analysis.newcombe_delta_ci(132, 639, 104, 639)
    assert ci["low"] <= ci["point"] <= ci["high"]
    assert -1.0 <= ci["low"] and ci["high"] <= 1.0
    assert ci["point"] == pytest.approx(132 / 639 - 104 / 639)


def test_newcombe_identical_proportions_cover_zero() -> None:
    ci = qa_analysis.newcombe_delta_ci(100, 400, 100, 400)
    assert ci["point"] == 0.0
    assert ci["low"] < 0.0 < ci["high"]


def test_newcombe_separated_proportions_exclude_zero_above() -> None:
    ci = qa_analysis.newcombe_delta_ci(60, 100, 30, 100)
    assert ci["low"] > 0.0


def test_newcombe_zero_against_half_excludes_zero_below() -> None:
    ci = qa_analysis.newcombe_delta_ci(0, 50, 25, 50)
    assert ci["high"] < 0.0


def test_newcombe_width_close_to_published_method10_behaviour() -> None:
    # For 56/70 vs 48/80 (Newcombe 1998 worked example proportions),
    # method 10 gives roughly [0.07, 0.33] for d = 0.8 - 0.6.
    ci = qa_analysis.newcombe_delta_ci(56, 70, 48, 80)
    assert 0.05 <= ci["low"] <= 0.095
    assert 0.30 <= ci["high"] <= 0.35


def test_newcombe_rejects_empty_samples() -> None:
    with pytest.raises(ValueError):
        qa_analysis.newcombe_delta_ci(0, 0, 1, 10)


# ---------------------------------------------------------------------------
# Degeneracy-guard verdicts
# ---------------------------------------------------------------------------


def _degeneracy(distinct: int, modal_share: float, guard_pass: bool | None = None) -> dict:
    payload = {
        "evaluated": 60,
        "distinct_answers": distinct,
        "modal_answer": "and the lord said",
        "modal_share": modal_share,
        "guard": {"distinct_answers_min": 3, "modal_share_max": 0.9},
    }
    if guard_pass is not None:
        payload["guard_pass"] = guard_pass
    return payload


def test_guard_verdict_pass() -> None:
    verdict = qa_analysis.guard_verdict(_degeneracy(31, 0.3, guard_pass=True), "arm")
    assert verdict["verdict"] == "PASS"
    assert verdict["reasons"] == []


def test_guard_verdict_fails_on_distinct_floor() -> None:
    verdict = qa_analysis.guard_verdict(_degeneracy(2, 0.3, guard_pass=False), "arm")
    assert verdict["verdict"] == "FAIL"
    assert any("distinct" in reason for reason in verdict["reasons"])


def test_guard_verdict_fails_on_modal_ceiling() -> None:
    verdict = qa_analysis.guard_verdict(_degeneracy(20, 0.95, guard_pass=False), "arm")
    assert verdict["verdict"] == "FAIL"
    assert any("modal_share" in reason for reason in verdict["reasons"])


def test_guard_verdict_flags_disagreement_with_recorded_flag() -> None:
    verdict = qa_analysis.guard_verdict(_degeneracy(2, 0.99, guard_pass=True), "arm")
    assert verdict["verdict"] == "FAIL"
    assert any("disagrees" in reason for reason in verdict["reasons"])


def test_majority_floor_baseline_fails_guard_by_construction() -> None:
    baseline = json.loads((BASELINES_DIR / "majority_floor.json").read_text(encoding="utf-8"))
    verdict = qa_analysis.guard_verdict(baseline["metrics"]["degeneracy"], "majority_floor")
    assert verdict["verdict"] == "FAIL"
    assert len(verdict["reasons"]) == 2


# ---------------------------------------------------------------------------
# Fixture synthesis
# ---------------------------------------------------------------------------


def test_fixture_synthesis_is_deterministic_and_seed_sensitive() -> None:
    first = qa_analysis.canonical_json_bytes(qa_analysis.synthesize_fixture_record(seed=3))
    second = qa_analysis.canonical_json_bytes(qa_analysis.synthesize_fixture_record(seed=3))
    other = qa_analysis.canonical_json_bytes(qa_analysis.synthesize_fixture_record(seed=4))
    assert first == second
    assert first != other


def test_fixture_record_has_full_pilot_schema_shape() -> None:
    record = qa_analysis.synthesize_fixture_record(seed=0)
    assert record["schema_version"] == qa_analysis.PILOT_RECORD_SCHEMA
    metrics = record["metrics"]
    assert metrics["answer_span"]["tasks"] == 60
    assert metrics["answer_span"]["predictions"] == 639
    assert set(metrics["per_family"]) == set(qa_analysis.FAMILIES)
    assert all(row["tasks"] == 20 for row in metrics["per_family"].values())
    assert metrics["degeneracy"]["guard_pass"] is True
    assert "synthetic_fixture" in record


# ---------------------------------------------------------------------------
# End-to-end analysis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_baselines() -> dict[str, dict]:
    return qa_analysis.load_baseline_records(BASELINES_DIR)


def test_load_baseline_records_rejects_foreign_schema(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(json.dumps({"schema_version": "nope", "arm": "x"}))
    with pytest.raises(ValueError):
        qa_analysis.load_baseline_records(tmp_path)


def test_analyze_end_to_end_on_fixture_vs_real_baselines(real_baselines: dict) -> None:
    record = qa_analysis.synthesize_fixture_record(seed=0)
    analysis = qa_analysis.analyze_pilot_record(record, real_baselines)

    assert analysis["schema_version"] == qa_analysis.ANALYSIS_SCHEMA_VERSION
    headline = analysis["headline"]
    assert headline["span_correct"] == record["metrics"]["answer_span"]["correct"]
    assert headline["span_predictions"] == 639

    by_arm = {comp["arm"]: comp for comp in analysis["comparisons"]}
    assert set(by_arm) == {"unigram_smoothed", "trigram_smoothed", "majority_floor"}

    # Delta arithmetic must reproduce newcombe on the exact counts.
    pilot_correct = headline["span_correct"]
    tri = by_arm["trigram_smoothed"]
    manual = qa_analysis.newcombe_delta_ci(pilot_correct, 639, 104, 639)
    assert tri["delta_point"] == pytest.approx(manual["point"])
    assert tri["delta_ci95"]["low"] == pytest.approx(manual["low"])
    assert tri["delta_ci95"]["high"] == pytest.approx(manual["high"])
    assert tri["verdict"] in {"PILOT_BEATS", "PILOT_LOSES", "STATISTICAL_TIE"}
    assert tri["decision_grade"] == (tri["ci_excludes_zero"] is True)

    uni = by_arm["unigram_smoothed"]
    assert uni["baseline_accuracy"] == pytest.approx(29 / 639)
    assert uni["delta_point"] == pytest.approx(pilot_correct / 639 - 29 / 639)

    # majority_floor has zero teacher-forced predictions: comparison N/A.
    floor = by_arm["majority_floor"]
    assert floor["applicable"] is False
    assert floor["verdict"] == "NOT_APPLICABLE"

    assert set(analysis["per_family"]) == set(qa_analysis.FAMILIES)
    for family, row in analysis["per_family"].items():
        assert row["span_predictions"] > 0
        lo, hi = row["span_accuracy_wilson95"].values()
        assert 0.0 <= lo <= row["span_accuracy"] <= hi <= 1.0
        assert family in qa_analysis.FAMILIES

    rules = {rule["rule"]: rule for rule in analysis["mechanical_rules"]}
    assert set(rules) == {
        "beat_trigram_smoothed",
        "beat_unigram_smoothed",
        "degeneracy_guard_clean",
    }
    assert rules["beat_unigram_smoothed"]["verdict"] == "PILOT_BEATS"

    # Round-trips through canonical JSON cleanly.
    encoded = json.loads(qa_analysis.canonical_json_bytes(analysis).decode("utf-8"))
    assert encoded["headline"] == headline


def test_analyze_requires_matching_schemas(real_baselines: dict) -> None:
    bad = qa_analysis.synthesize_fixture_record(seed=1)
    bad["schema_version"] = "humanbrain/some-other-thing/v9"
    with pytest.raises(ValueError):
        qa_analysis.analyze_pilot_record(bad, real_baselines)


def test_findings_markdown_renders_sections_and_is_stable(real_baselines: dict) -> None:
    record = qa_analysis.synthesize_fixture_record(seed=0)
    analysis = qa_analysis.analyze_pilot_record(record, real_baselines)
    analysis["baseline_guards"] = [
        qa_analysis.guard_verdict(b["metrics"]["degeneracy"], arm)
        for arm, b in sorted(real_baselines.items())
    ]
    first = qa_analysis.render_findings_markdown(analysis)
    second = qa_analysis.render_findings_markdown(analysis)
    assert first == second
    for needle in (
        "# R2 Bible-QA pilot",
        "## Headline",
        "Newcombe 95%",
        "| trigram_smoothed |",
        "## Degeneracy guard",
        "majority_floor",
        "## Per-family breakdown",
        "| who_said |",
        "Mechanical rule evaluation",
        "beat_unigram_smoothed",
    ):
        assert needle in first, f"markdown missing {needle!r}"


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    tool = ROOT / "tools" / "analyze_bible_qa.py"
    return subprocess.run(
        [sys.executable, str(tool), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=180,
    )


def test_cli_synthesizes_then_analyzes_end_to_end(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture_record.json"
    synth = _run_cli("--synthesize-fixture", str(fixture), "--fixture-seed", "7")
    assert synth.returncode == 0, synth.stderr
    assert fixture.exists()

    out_json = tmp_path / "analysis.json"
    out_md = tmp_path / "FINDINGS.md"
    analyzed = _run_cli(
        "--record", str(fixture),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    )
    assert analyzed.returncode == 0, analyzed.stderr

    analysis = json.loads(out_json.read_text(encoding="utf-8"))
    assert analysis["inputs"]["pilot_seed"] == 7
    markdown = out_md.read_text(encoding="utf-8")
    assert "# R2 Bible-QA pilot" in markdown
