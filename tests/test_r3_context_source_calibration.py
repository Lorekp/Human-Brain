from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "r3_pilot" / "calibrate_context_source.py"
ARTIFACT = ROOT / "experiments" / "r3_pilot" / "context_source_calibration.json"


def test_committed_context_source_artifact_passes_fast_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "context-source artifact check: PASS" in completed.stdout


def test_artifact_freezes_the_train_only_rejection_evidence() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["scope"]["held_out_books_decoded"] == 0
    assert artifact["scope"]["held_out_books_skipped"] == 17
    assert artifact["scope"]["first_200_token_ids_sha256"] == (
        "de5d527b12968762d6bd614c3a77c705e053e272361a72d20f3c0a21b272ef3c"
    )
    assert artifact["selectors"]["earliest_refractory"]["by_k"]["20"] == {
        "auc_same_vs_different_previous": 0.503477,
        "different_previous_mean_overlap": 0.150177,
        "paired_occurrences": 1132,
        "same_previous_mean_overlap": 0.132509,
        "same_previous_median_overlap": 0,
        "same_previous_theta10_probability": 0.0,
        "unrelated_theta10_probability": 0.004417,
    }
    assert artifact["selectors"]["earliest_refractory"]["by_k"]["64"][
        "paired_occurrences"
    ] == 1128
    assert artifact["deep_state"]["checkpoints"][0] == {
        "all_finite": True,
        "step_count": 512,
        "threshold_voltage_max_abs_order10": 96,
        "voltage_max_abs_order10": 100,
    }
    assert artifact["deep_state"]["first_nonfinite"]["step_index_zero_based"] == 1578
    assert artifact["verdict"] == {
        "implement_native_first_k": False,
        "selected_k": None,
        "status": "REJECT_CURRENT_GLIF5_CONTEXT_SOURCE",
    }
