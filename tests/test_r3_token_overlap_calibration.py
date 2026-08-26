from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "r3_pilot" / "calibrate_token_overlap.py"
ARTIFACT = ROOT / "experiments" / "r3_pilot" / "token_overlap_calibration.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("r3_token_overlap_calibration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_artifact_is_a_fresh_deterministic_rebuild() -> None:
    module = _load_module()
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert committed == module.build_artifact()
    assert committed["pilot_outcome_data_used"] is False
    assert committed["inputs_limited_to_frozen_pre_pilot_sources"] is True
    assert committed["status"] == "pre-replay diagnostic; non-gating"
    assert all(
        "max_float_pmf_sum_error" not in validation
        for validation in committed["self_validation"].values()
    )


def test_exact_hypergeometric_reference_values() -> None:
    module = _load_module()
    assert module.hypergeometric_tail(74, 20, 20, 10) == Fraction(
        1541612473433, 166287370288628
    )
    assert module.hypergeometric_tail(74, 20, 20, 13) == Fraction(
        17307363779, 689683683328244
    )
    assert module.hypergeometric_tail(439, 20, 20, 10) == Fraction(
        338789308259498892347837,
        814006712127603554985416457224115,
    )
    assert module.hypergeometric_tail(439, 20, 20, 13) == Fraction(
        484661482946615927,
        271335570709201184995138819074705,
    )


def test_contact_budget_curve_respects_frozen_thresholds() -> None:
    module = _load_module()
    artifact = module.build_artifact()
    for regime in artifact["regimes"]:
        rows = regime["contact_curve_0_to_20"]
        assert len(rows) == 21
        for row in rows:
            contacts = row["contacts"]
            p_match = row["p_ge_theta_m_10"]["decimal"]
            p_active = row["p_ge_theta_a_13"]["decimal"]
            assert p_active <= p_match
            if contacts < 10:
                assert p_match == 0.0
            if contacts < 13:
                assert p_active == 0.0
