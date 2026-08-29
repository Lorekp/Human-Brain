from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import run_cn2_dev_4096 as gate


def _step(
    token: int,
    winners: tuple[int, ...],
    *,
    event: int,
    allocated: int,
    predicted_columns: int = 0,
) -> object:
    return SimpleNamespace(
        token_id=token,
        event_count=event,
        winner_spikes=tuple(SimpleNamespace(cell_id=winner) for winner in winners),
        allocated_segments=allocated,
        punished_segments=0,
        predicted_columns=predicted_columns,
        active_segments=0,
        matching_segments=0,
        candidate_segments=0,
        postings_visited=0,
    )


def test_factorized_ledger_accepts_one_shared_column_then_exact_replay() -> None:
    source_columns = tuple(range(11))
    a_columns = tuple(range(11, 22))
    b_columns = tuple(range(21, 32))
    source = tuple(range(100, 111))
    a = tuple(range(200, 211))
    b = (a[-1], *range(300, 310))
    ledger = gate.FactorizedIdentityLedger()

    ledger.observe(
        _step(0, source, event=1, allocated=0), columns=source_columns, reset_before=True
    )
    ledger.observe(_step(1, a, event=2, allocated=11), columns=a_columns, reset_before=False)
    ledger.observe(
        _step(0, source, event=1, allocated=0), columns=source_columns, reset_before=True
    )
    ledger.observe(
        _step(2, b, event=2, allocated=10, predicted_columns=1),
        columns=b_columns,
        reset_before=False,
    )
    ledger.observe(
        _step(0, source, event=1, allocated=0), columns=source_columns, reset_before=True
    )
    ledger.observe(
        _step(2, b, event=2, allocated=0, predicted_columns=11),
        columns=b_columns,
        reset_before=False,
    )

    assert b != a
    assert set(a) & set(b) == {a[-1]}
    assert ledger.complete_keys[(source, 2)] == b
    assert ledger.column_owners[(source, 21)] == a[-1]
    assert ledger.novel_keys == 2
    assert ledger.repeated_keys == 1
    assert ledger.actual_allocations == 21
    assert ledger.partial_novel_allocations == 1


def test_factorized_ledger_stops_shared_column_owner_drift() -> None:
    source_columns = tuple(range(11))
    a_columns = tuple(range(11, 22))
    b_columns = tuple(range(21, 32))
    source = tuple(range(100, 111))
    a = tuple(range(200, 211))
    drifted_b = tuple(range(300, 311))
    ledger = gate.FactorizedIdentityLedger()

    ledger.observe(
        _step(0, source, event=1, allocated=0), columns=source_columns, reset_before=True
    )
    ledger.observe(_step(1, a, event=2, allocated=11), columns=a_columns, reset_before=False)
    ledger.observe(
        _step(0, source, event=1, allocated=0), columns=source_columns, reset_before=True
    )
    with pytest.raises(gate.GateStop) as caught:
        ledger.observe(
            _step(2, drifted_b, event=2, allocated=10),
            columns=b_columns,
            reset_before=False,
        )
    assert caught.value.kind == "EXACT_SOURCE_COLUMN_OWNER_DRIFT"
    assert caught.value.details["column"] == 21


def test_factorized_ledger_stops_repeated_complete_tuple_drift() -> None:
    columns = tuple(range(11))
    source = tuple(range(100, 111))
    owners = tuple(range(200, 211))
    ledger = gate.FactorizedIdentityLedger()

    ledger.observe(_step(0, source, event=1, allocated=0), columns=columns, reset_before=True)
    ledger.observe(_step(1, owners, event=2, allocated=11), columns=columns, reset_before=False)
    ledger.observe(_step(0, source, event=1, allocated=0), columns=columns, reset_before=True)
    with pytest.raises(gate.GateStop) as caught:
        ledger.observe(
            _step(1, tuple(range(300, 311)), event=2, allocated=0),
            columns=columns,
            reset_before=False,
        )
    assert caught.value.kind == "REPEATED_COMPLETE_KEY_DRIFT"


def test_cost_fit_is_nonnegative_and_contract_is_development_only() -> None:
    fit = gate._log_log_fit(((2_000.0, 10_000.0), (16_000.0, 20_000.0), (40_000.0, 30_000.0)))
    assert fit["nonnegative_exponent"] > 0
    assert gate.WORST_CASE_SEGMENT_CAPACITY == 23_290_993
    assert gate.TOTAL_C1_SCHEDULE_STEPS == 33_135_753
    assert gate.GATE_CONTRACT["claim_boundary"] == {
        "authorizes_c1": False,
        "capability_evidence": False,
        "held_out_data_accessed": False,
        "threshold_selection": False,
        "transfer_evidence": False,
    }


def test_banked_public_train_gate_keeps_identity_pass_separate_from_cost_stop() -> None:
    payload = json.loads(gate.OUTPUT_PATH.read_bytes())
    assert payload["status"] == "STOP_CN2_DEV_COST_PROJECTION"
    assert payload["input"]["partition"] == "public TRAIN"
    assert payload["input"]["held_out_books_materialized"] == 0
    assert payload["learn"]["identity"]["actual_allocated_segments"] == 45_045
    assert payload["learn"]["terminal_checkpoint_identity"]["segment_count"] == 45_045
    assert all(
        payload["read_only"][key] is True
        for key in (
            "exact_winner_trace",
            "zero_allocations",
            "zero_punishments",
            "terminal_identity_exact",
            "checkpoint_bytes_exact",
        )
    )
    projection = payload["developmental_cost_projection"]
    assert projection["authorizes_c1"] is False
    assert projection["passes_development_cap"] is False
    assert projection["projected_total_hours"] > projection["development_cap_hours"]
    runner_path = Path(gate.__file__).resolve()
    runner_key = runner_path.relative_to(gate.REPO_ROOT).as_posix()
    assert payload["source_binding"]["files"][runner_key] == gate._canonical_lf_sha256(runner_path)


def test_timing_contamination_interpretation_preserves_raw_identity_evidence() -> None:
    interpretation_path = gate.OUTPUT_PATH.with_name("cn2_dev_4096.interpretation.json")
    interpretation = json.loads(interpretation_path.read_bytes())
    assert interpretation["classification"] == "STOP_COST_TIMING_CONTAMINATED"
    assert interpretation["cost_interpretation"]["authoritative_cap_failure"] is False
    assert interpretation["cost_interpretation"]["replacement_measurement"] == (
        "P2-COST-DENSE-TRAIN49-v1"
    )
    assert interpretation["identity_interpretation"]["verdict"] == (
        "PASS_CN2_DEV_IDENTITY_DURABILITY"
    )
    raw = gate.OUTPUT_PATH.read_bytes()
    assert len(raw) == interpretation["raw_artifact"]["bytes"]
    assert hashlib.sha256(raw).hexdigest() == interpretation["raw_artifact"]["sha256"]
    assert interpretation["raw_artifact"]["raw_status"] == ("STOP_CN2_DEV_COST_PROJECTION")
