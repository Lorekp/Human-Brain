from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v2_bound_execution as engine,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v2_protocol as protocol,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v2_runner as runner,
)


def test_v2_runner_binds_fresh_paths_and_executes_both_normal_rows_with_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_source = Path(engine.__file__).read_text(encoding="utf-8")
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    combined_source = engine_source + runner_source
    assert "canonical_joint_v1_protocol" not in combined_source
    assert "canonical_joint_v1_runner" not in combined_source
    assert (
        "typed_semantic_packed_public_train_fit_prompt_record_authorized_plan_frontier_v1"
        not in combined_source
    )
    assert "produce_source_event_bundle_batch(" in engine_source
    assert "cell_gain=0.0" in engine_source
    assert protocol.PROTOCOL.protocol_sha256 == runner.PROTOCOL_ROOT_SHA256
    assert protocol.PROTOCOL.dependency_sha256 == runner.PROTOCOL_DEPENDENCY_SHA256

    result_path = tmp_path / "v2-focused.result.json"
    spent_path = tmp_path / "v2-focused.spent.json"
    monkeypatch.setattr(runner, "RESULT_PATH", result_path)
    monkeypatch.setattr(runner, "SPENT_PATH", spent_path)

    def no_real_execution(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("focused runner test touched real physical execution")

    monkeypatch.setattr(engine, "bind_treatment", no_real_execution)

    def digest(label: str) -> str:
        return sha256(label.encode("ascii")).hexdigest()

    call_order: list[tuple[str, int]] = []
    marker_seen_before_call: list[bool] = []
    source_counts = (8, 9)

    def fake_call(
        preflight: protocol.FrozenPairingPreflight,
        row: protocol.FrozenPublicTrainJointRow,
        row_ordinal: int,
        mode: str,
    ) -> engine.TreatmentOutcome:
        assert preflight.status == protocol.PASS
        assert row is protocol.ROWS[row_ordinal]
        marker_seen_before_call.append(
            runner.SPENT_PATH.is_file() and not runner.RESULT_PATH.exists()
        )
        marker = json.loads(runner.SPENT_PATH.read_text(encoding="utf-8"))
        assert marker["state"] == "PUBLIC_TRAIN_JOINT_V2_ATTEMPT_SPENT"
        assert marker["no_retry"] is True
        assert marker["real_electrical_executor"] is False
        call_order.append((mode, row_ordinal))

        source_count = 0 if mode == engine.MODE_PRODUCER_OFF else source_counts[row_ordinal]
        occurrence_count = (
            source_count
            if mode
            in (
                engine.MODE_CONVERTER_OFF,
                engine.MODE_WRONG_DESIRED,
                engine.MODE_NORMAL,
            )
            else 0
        )
        if mode == engine.MODE_PRODUCER_OFF:
            build_status = public_status = engine.ZERO_SELECTED_SOURCE_EVENTS
            source_status = engine.prompt_producer.STOP_ZERO
            occurrence_status = selection_status = None
            occurrence_attempts = plan_attempts = desired_reads = static_queries = 0
            proof_count = witness_count = relation = predictions = races = 0
            forest_sha = work_sha = frontier_sha = bundle_sha = None
            static_shas: tuple[str, ...] = ()
        elif mode == engine.MODE_STATIC_OFF:
            build_status = public_status = engine.occurrence_static.STOP_ZERO
            source_status = engine.prompt_producer.STOP_UNSUPPORTED
            occurrence_status = engine.occurrence_static.STOP_ZERO
            selection_status = None
            occurrence_attempts = 1
            plan_attempts = desired_reads = 0
            static_queries = source_count
            proof_count = witness_count = relation = predictions = races = 0
            forest_sha = work_sha = frontier_sha = None
            bundle_sha = digest(f"bundle:{row_ordinal}")
            static_shas = (digest(f"static-off:{row_ordinal}"),)
        else:
            source_status = engine.prompt_producer.READY
            occurrence_status = engine.occurrence_static.READY
            occurrence_attempts = 2
            static_queries = 2 * (
                source_count * source_count + 18 * source_count
            )
            bundle_sha = digest(f"bundle:{row_ordinal}")
            static_shas = (
                digest(f"static-first:{mode}:{row_ordinal}"),
                digest(f"static-second:{mode}:{row_ordinal}"),
            )
            frontier_sha = digest(f"frontier:{mode}:{row_ordinal}")
            if mode == engine.MODE_CONVERTER_OFF:
                build_status = public_status = engine.STOP_SOURCE_CERTIFICATE
                selection_status = None
                plan_attempts = desired_reads = 0
                proof_count = witness_count = relation = predictions = races = 0
                forest_sha = work_sha = None
            else:
                build_status = engine.READY
                plan_attempts = 1
                desired_reads = protocol.VISIBLE_CASE_COUNT
                relation, predictions, races = (120, 21, 6)
                work_sha = digest(f"work:{mode}:{row_ordinal}")
                if mode == engine.MODE_WRONG_DESIRED:
                    selection_status = (
                        engine.STOP_AUTHORIZED_ECLASS_MISMATCH
                        if row_ordinal == 0
                        else engine.STOP_NO_EXACT
                    )
                    public_status = (
                        protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
                        if row_ordinal == 0
                        else engine.STOP_NO_EXACT
                    )
                    proof_count = witness_count = 0
                    forest_sha = None
                elif row_ordinal == 0:
                    selection_status = public_status = engine.STOP_NO_EXACT
                    proof_count = witness_count = 0
                    forest_sha = None
                else:
                    selection_status = public_status = engine.SELECTED
                    proof_count = witness_count = 1
                    forest_sha = digest("normal-row-1-forest")

        geometry = protocol.CANDIDATE_GEOMETRY_RECEIPTS[row_ordinal]
        values = {
            "mode": mode,
            "row_ordinal": row_ordinal,
            "build_status": build_status,
            "source_event_status": source_status,
            "occurrence_status": occurrence_status,
            "public_status": public_status,
            "selection_status": selection_status,
            "source_candidate_event_count": geometry.total_candidate_count,
            "source_event_count": source_count,
            "occurrence_count": occurrence_count,
            "producer_build_attempts": 1,
            "occurrence_inference_attempts": occurrence_attempts,
            "plan_build_attempts": plan_attempts,
            "runtime_desired_reads": desired_reads,
            "exact_proof_count": proof_count,
            "witness_count": witness_count,
            "selected_proof_forest_sha256": forest_sha,
            "relation_work_units": relation,
            "physical_predictions": predictions,
            "physical_races": races,
            "static_query_count": static_queries,
            "payload_sha256": digest(f"payload:{mode}:{row_ordinal}"),
            "geometry_sha256": geometry.sha256,
            "source_certificate_sha256": digest(f"source:{mode}:{row_ordinal}"),
            "source_bundle_sha256": bundle_sha,
            "static_certificate_sha256s": static_shas,
            "frontier_sha256": frontier_sha,
            "binding_sha256": digest(f"binding:{mode}:{row_ordinal}"),
            "work_sha256": work_sha,
            "sha256": digest(f"outcome:{mode}:{row_ordinal}"),
        }
        return engine.TreatmentOutcome(**values)

    def fake_authenticate(outcome: engine.TreatmentOutcome) -> object:
        assert type(outcome) is engine.TreatmentOutcome
        return outcome

    fake_executor = runner._TreatmentExecutor(
        "focused-no-electrical-executor", False, fake_call, fake_authenticate
    )
    misplaced_real = fake_executor._replace(
        name="misplaced-real-executor", real=True
    )
    zero_work_prepared = runner._prepare_pre_spend()
    with pytest.raises(runner.RunnerStop) as misplaced:
        runner._create_spend_authority(zero_work_prepared, misplaced_real)
    assert misplaced.value.status == runner.STOP_PRESPEND
    assert "exact canonical" in str(misplaced.value)
    assert not result_path.exists()
    assert not spent_path.exists()

    invalid_result_path = tmp_path / "v2-invalid-control.result.json"
    invalid_spent_path = tmp_path / "v2-invalid-control.spent.json"
    monkeypatch.setattr(runner, "RESULT_PATH", invalid_result_path)
    monkeypatch.setattr(runner, "SPENT_PATH", invalid_spent_path)

    def invalid_control_call(
        preflight: protocol.FrozenPairingPreflight,
        row: protocol.FrozenPublicTrainJointRow,
        row_ordinal: int,
        mode: str,
    ) -> engine.TreatmentOutcome:
        outcome = fake_call(preflight, row, row_ordinal, mode)
        return outcome._replace(
            source_candidate_event_count=outcome.source_candidate_event_count - 1,
            sha256=digest("invalid-control-outcome"),
        )

    invalid_executor = fake_executor._replace(
        name="focused-invalid-control-executor", call=invalid_control_call
    )
    invalid_artifact = runner._run_with_executor(invalid_executor)
    invalid_execution = invalid_artifact["execution"]
    assert type(invalid_execution) is runner.RunnerExecution
    assert invalid_execution.verdict == protocol.ONE_SHOT_STOP
    assert len(invalid_execution.treatment_outcomes) == 1
    assert call_order == [(engine.MODE_PRODUCER_OFF, 0)]

    call_order.clear()
    marker_seen_before_call.clear()
    monkeypatch.setattr(runner, "RESULT_PATH", result_path)
    monkeypatch.setattr(runner, "SPENT_PATH", spent_path)
    artifact = runner._run_with_executor(fake_executor)
    execution = artifact["execution"]
    assert type(execution) is runner.RunnerExecution
    assert execution.verdict == protocol.ONE_SHOT_STOP
    assert execution.stop_reason == "normal row 0 did not select an exact proof"
    assert call_order == [
        (mode, row_ordinal)
        for mode in engine.TREATMENT_ORDER
        for row_ordinal in (0, 1)
    ]
    assert all(marker_seen_before_call)
    assert call_order[-2:] == [
        (engine.MODE_NORMAL, 0),
        (engine.MODE_NORMAL, 1),
    ]
    assert execution.work.producer_query_count == 5_540
    assert execution.work.static_query_count == (
        8 + 6 * (8 * 8 + 18 * 8) + 9 + 6 * (9 * 9 + 18 * 9)
    )
    assert execution.work.fitted_glif_step_count == (
        execution.work.producer_query_count + execution.work.static_query_count
    ) * 30
    assert execution.work.static_query_count < protocol.WORK_FORMULA.static_query_cap
    assert execution.work.plan_build_count == 4
    assert execution.proof_materialization_count == 1
    assert result_path.is_file()
    assert spent_path.is_file()
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    assert stored["verdict"] == protocol.ONE_SHOT_STOP
    assert len(stored["execution"]["treatment_outcomes"]) == 10
    assert stored["preflights"]["electrical_execution_count"] == 0
    assert stored["scope"]["retries"] == 0

    with pytest.raises(runner.RunnerStop) as repeated:
        runner._run_with_executor(fake_executor)
    assert repeated.value.status == runner.STOP_ALREADY_SPENT
    assert len(call_order) == 10
