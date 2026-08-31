from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v3_bound_execution as engine,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v3_protocol as protocol,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v3_runner as runner,
)


def test_v3_runner_zero_work_prepare_and_fake_stop_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def digest(label: str) -> str:
        return sha256(label.encode("ascii")).hexdigest()

    def outcome(
        row_ordinal: int,
        *,
        selection_mode: str | None = None,
    ) -> engine.RowTreatmentOutcome:
        geometry = protocol.CANDIDATE_GEOMETRY_RECEIPTS[row_ordinal]
        selected_count = 1 if row_ordinal == 0 else 3
        static_on_ready = row_ordinal == 1
        static_on_status = (
            engine.occurrence_static.READY
            if static_on_ready
            else engine.occurrence_static.STOP_ZERO
        )
        static_on_queries = (
            selected_count
            + 6 * (selected_count * selected_count + 18 * selected_count)
            if static_on_ready
            else selected_count
        )
        static_queries = selected_count + static_on_queries
        ready = static_on_ready
        normal = selection_mode == engine.MODE_NORMAL
        wrong = selection_mode == engine.MODE_WRONG_DESIRED
        build_status = engine.READY if ready else static_on_status
        if normal:
            public_status = selection_status = engine.SELECTED
            proof_count = witness_count = 1
            forest_sha = digest(f"forest:{row_ordinal}")
            relation, predictions, races = 120, 21, 2
        elif wrong:
            public_status = protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
            selection_status = engine.STOP_AUTHORIZED_ECLASS_MISMATCH
            proof_count = witness_count = 0
            forest_sha = None
            relation, predictions, races = 100, 20, 1
        else:
            public_status = build_status
            selection_status = None
            proof_count = witness_count = 0
            forest_sha = None
            relation = predictions = races = 0
        roots = lambda label: (  # noqa: E731 - compact fixture framing
            digest(f"{label}:0:{row_ordinal}"),
            digest(f"{label}:1:{row_ordinal}"),
        )
        values = {
            "row_ordinal": row_ordinal,
            "build_status": build_status,
            "public_status": public_status,
            "selection_mode": selection_mode,
            "selection_status": selection_status,
            "source_gain0_status": engine.source_static.SOURCE_ZERO,
            "source_gain1_status": engine.source_static.SOURCE_SELECTED,
            "static_off_status": engine.occurrence_static.STOP_ZERO,
            "static_on_status": static_on_status,
            "source_candidate_event_count": geometry.total_candidate_count,
            "source_selected_event_count": selected_count,
            "selector_pass_count": 2,
            "source_query_count": 2 * geometry.total_candidate_count,
            "source_fitted_glif_step_count": 2
            * geometry.total_candidate_count
            * 30,
            "source_compiled_kernel_call_count": 2,
            "static_pass_count": 2,
            "static_off_query_count": selected_count,
            "static_on_query_count": static_on_queries,
            "static_query_count": static_queries,
            "static_fitted_glif_step_count": static_queries * 30,
            "converter_control_status": (
                engine.STOP_SOURCE_CERTIFICATE if ready else None
            ),
            "converter_control_plan_build_attempts": 0,
            "converter_build_attempts": 1 if ready else 0,
            "plan_build_attempts": 1 if ready else 0,
            "runtime_desired_reads": (
                protocol.VISIBLE_CASE_COUNT if selection_mode is not None else 0
            ),
            "exact_proof_count": proof_count,
            "witness_count": witness_count,
            "selected_proof_forest_sha256": forest_sha,
            "relation_completed_units": relation,
            "relation_attempted_units": relation,
            "relation_limit": 30_000 if ready else 0,
            "physical_predictions": predictions,
            "physical_races": races,
            "dependency_sha256": digest(f"dependency:{row_ordinal}"),
            "payload_sha256": digest(f"payload:{row_ordinal}"),
            "geometry_sha256": geometry.sha256,
            "source_certificate_sha256s": roots("source-certificate"),
            "source_request_sha256s": roots("source-request"),
            "source_candidate_event_root_sha256s": roots("candidate-root"),
            "source_selected_event_root_sha256s": roots("selected-root"),
            "source_work_sha256s": roots("source-work"),
            "static_certificate_sha256s": roots("static-certificate"),
            "static_request_sha256s": (
                digest(f"static-request:{row_ordinal}"),
                digest(f"static-request:{row_ordinal}"),
            ),
            "static_view_sha256s": (
                None,
                digest(f"static-view:{row_ordinal}") if ready else None,
            ),
            "static_work_sha256": digest(f"static-work:{row_ordinal}"),
            "converter_control_frontier_sha256": (
                digest(f"converter-control:{row_ordinal}") if ready else None
            ),
            "frontier_sha256": digest(f"frontier:{row_ordinal}") if ready else None,
            "binding_sha256": digest(f"binding:{row_ordinal}"),
            "frontier_work_sha256": (
                digest(f"work:{selection_mode}:{row_ordinal}") if ready else None
            ),
            "sha256": digest(f"outcome:{selection_mode}:{row_ordinal}"),
        }
        return engine.RowTreatmentOutcome(**values)

    def forbid_real_bind(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("focused V3 runner test touched real engine.bind_row")

    monkeypatch.setattr(engine, "bind_row", forbid_real_bind)
    row_calls: list[int] = []
    selection_order: list[str] = []
    marker_seen_before_call: list[bool] = []

    def fake_call(
        preflight: protocol.FrozenPairingPreflight,
        row: protocol.FrozenPublicTrainJointRow,
        row_ordinal: int,
    ) -> tuple[engine.RowTreatmentOutcome, ...]:
        assert preflight.status == protocol.PASS
        assert row is protocol.ROWS[row_ordinal]
        row_calls.append(row_ordinal)
        marker_seen_before_call.append(
            runner.SPENT_PATH.is_file() and not runner.RESULT_PATH.exists()
        )
        if row_ordinal == 0:
            return (outcome(row_ordinal),)
        selection_order.extend(engine.SELECTION_ORDER)
        return (
            outcome(row_ordinal, selection_mode=engine.MODE_WRONG_DESIRED),
            outcome(row_ordinal, selection_mode=engine.MODE_NORMAL),
        )

    fake = runner._RowExecutor(
        "focused-v3-fake", False, fake_call, lambda value: value
    )

    # Preparing and rejecting a fake on either canonical path performs zero work.
    prepared = runner._prepare_pre_spend()
    prepared_receipt = runner._lookup_prepared(prepared)
    assert runner._zero_preflight(prepared_receipt.exact_preflight)
    with pytest.raises(runner.RunnerStop, match="may not target either canonical"):
        runner._create_spend_authority(prepared, fake)
    assert not runner.CANONICAL_RESULT_PATH.exists()
    assert not runner.CANONICAL_SPENT_PATH.exists()
    assert row_calls == []

    result_path = tmp_path / "v3-focused.result.json"
    spent_path = tmp_path / "v3-focused.spent.json"
    monkeypatch.setattr(runner, "RESULT_PATH", result_path)
    monkeypatch.setattr(runner, "SPENT_PATH", spent_path)
    artifact = runner._run_with_executor(fake)
    execution = artifact["execution"]

    assert type(execution) is runner.RunnerExecution
    assert execution.verdict == protocol.ONE_SHOT_STOP
    assert "row 0 static-on stopped" in (execution.stop_reason or "")
    assert row_calls == [0, 1]
    assert selection_order == list(engine.SELECTION_ORDER)
    assert all(marker_seen_before_call)
    assert tuple(len(rows) for rows in execution.row_outcomes) == (1, 2)
    assert execution.work.source_query_count == protocol.WORK_FORMULA.source_query_cap
    assert execution.work.selected_source_counts == (1, 3)
    assert execution.work.static_query_count == 386
    assert execution.work.static_query_dynamic_cap == 992
    assert execution.work.plan_build_count == 1
    assert execution.work.relation_completed_units == 120
    assert execution.work.physical_predictions == 21
    assert execution.work.physical_races == 2
    assert execution.work.proof_materialization_count == 1
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(stored["execution"]["row_outcomes"]) == 2
    assert stored["spend_marker"]["exclusive_create"] is True
    assert stored["preflights"]["electrical_execution_count"] == 0

    result_path.unlink()
    spent_path.unlink()
    assert not result_path.exists()
    assert not spent_path.exists()
    assert not runner.CANONICAL_RESULT_PATH.exists()
    assert not runner.CANONICAL_SPENT_PATH.exists()
