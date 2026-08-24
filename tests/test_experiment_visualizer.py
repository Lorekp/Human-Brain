from __future__ import annotations

import json
from pathlib import Path

import pytest

from humanbrain.visualizer import ExperimentFormatError, load_catalog, load_run
from humanbrain.visualizer.cli import main
from humanbrain.visualizer.model import ProportionEstimate


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "experiments" / "visualizer_demo"


def test_demo_catalog_loads_through_the_canonical_result_path() -> None:
    catalog = load_catalog(ROOT / "experiments")
    demo = catalog.for_experiment("visualizer_demo")

    assert [run.arm for run in demo] == ["demo-ridge-512", "demo-similarity-512"]
    assert all(run.population.model_family == "GLIF5" for run in demo)
    assert all(run.outcome.calibration.fraction == 0.0 for run in demo)


def test_population_summary_reports_truthful_structural_estimates() -> None:
    run = load_run(DEMO / "demo-ridge-512.json")

    assert run.population.configured_neurons == 512
    assert run.population.enabled_neurons == 512
    assert run.population.excitatory_neurons == 410
    assert run.population.inhibitory_neurons == 102
    assert run.population.concurrent_activity_reference_neurons == 11
    assert run.population.input_contacts == 277 * 16
    assert run.population.recurrent_contacts == 512 * 8
    assert run.population.readout_contacts == 277 * 64
    assert run.population.total_contacts == 26_256
    assert run.population.recurrent_connection_density == pytest.approx(8 / 512)


def test_wilson_interval_remains_in_probability_bounds() -> None:
    estimate = ProportionEstimate.from_counts(8, 54)

    assert estimate.fraction == pytest.approx(8 / 54)
    assert 0.0 <= estimate.low < estimate.fraction < estimate.high <= 1.0


def test_malformed_current_schema_is_reported_without_guessing(tmp_path: Path) -> None:
    experiment = tmp_path / "broken"
    experiment.mkdir()
    path = experiment / "arm.json"
    path.write_text(json.dumps({"experiment_id": "broken", "arm": "arm"}), encoding="utf-8")

    with pytest.raises(ExperimentFormatError, match="missing ladder fields"):
        load_run(path)


def test_catalog_keeps_unsupported_json_visible_as_an_issue(tmp_path: Path) -> None:
    experiment = tmp_path / "scale"
    experiment.mkdir()
    path = experiment / "smoke.json"
    path.write_text(json.dumps({"cells": 1_000_000, "backend": "cuda"}), encoding="utf-8")

    catalog = load_catalog(tmp_path)

    assert catalog.runs == ()
    assert len(catalog.issues) == 1
    assert catalog.issues[0].unsupported is True


def test_cli_rejects_an_unknown_experiment_before_opening_the_gui() -> None:
    with pytest.raises(SystemExit, match="2"):
        main([str(ROOT / "experiments"), "--experiment", "not-a-real-experiment"])
