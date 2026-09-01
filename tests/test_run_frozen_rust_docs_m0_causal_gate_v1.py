from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import frozen_rust_docs_m0_causal_gate_v1_protocol as protocol
from experiments.r3_pilot import run_frozen_rust_docs_m0_causal_gate_v1 as runner

ROOT = "a" * 64
TASK_ROOTS = {
    name: sha256(f"synthetic-task:{name}".encode()).hexdigest()
    for name, _text in protocol.TASK_SPECS
}


def _synthetic_outputs() -> tuple[tuple[int, ...], ...]:
    return tuple((index,) for index, _case in enumerate(protocol.FROZEN_CASES))


def _zero_work() -> dict[str, int]:
    return {
        "recall_count": 0,
        "selected_source_materialization_count": 0,
        "selected_executor_call_count": 0,
        "authenticated_rustc_compile_receipt_count": 0,
        "authenticated_binary_run_receipt_count": 0,
        "oracle_read_count": 0,
    }


def _rustc_identity() -> protocol.rust_selected_action_executor.RustcIdentityV1:
    executor = protocol.rust_selected_action_executor
    toolchain = Path(__file__).resolve().parent / "synthetic-rust-toolchain"
    link_search_paths = (
        tuple(str(toolchain / f"search-{index}") for index in range(3)) if os.name == "nt" else ()
    )
    payload = {
        "path": str(toolchain / "rustc"),
        "version_verbose": "rustc synthetic-test\nhost: synthetic-test",
        "host_triple": "synthetic-test",
        "executable_sha256": ROOT,
        "linker_path": str(toolchain / "rust-lld"),
        "linker_executable_sha256": ROOT,
        "target_libdir_path": str(toolchain / "target-libdir"),
        "target_libdir_manifest_sha256": ROOT,
        "target_libdir_file_count": 1,
        "target_libdir_total_bytes": 1,
        "link_search_paths": link_search_paths,
        "link_search_manifest_sha256s": tuple(ROOT for _path in link_search_paths),
        "link_search_file_counts": tuple(1 for _path in link_search_paths),
        "link_search_total_bytes": tuple(1 for _path in link_search_paths),
    }
    identity = executor.RustcIdentityV1(
        **payload,
        sha256=executor._digest(executor._IDENTITY_NAMESPACE, payload),
    )
    identity.authenticate_structure()
    return identity


def _executor_preparation(
    identity: protocol.rust_selected_action_executor.RustcIdentityV1,
) -> protocol.rust_selected_action_executor.RustExecutorPreparationV1:
    executor = protocol.rust_selected_action_executor
    run_environment = tuple(
        sorted(
            {
                "RUST_BACKTRACE": "0",
                "TEMP": executor._TEMPORARY_ROOT_PLACEHOLDER,
                "TMP": executor._TEMPORARY_ROOT_PLACEHOLDER,
            }.items()
        )
    )
    compile_values = dict(run_environment)
    if os.name == "nt":
        compile_values["LIB"] = os.pathsep.join(identity.link_search_paths)
    compile_environment = tuple(sorted(compile_values.items()))
    payload = {
        "schema": executor.SCHEMA,
        "claim_boundary": executor.CLAIM_BOUNDARY,
        "compiler_policy_sha256": executor.COMPILER_POLICY_SHA256,
        "rustc_identity_sha256": identity.sha256,
        "expected_registry_sha256": ROOT,
        "harness_sha256": executor.HARNESS_SHA256,
        "batch_protocol_sha256": executor.BATCH_PROTOCOL_SHA256,
        "batch_sha256": executor._digest(executor._BATCH_NAMESPACE, protocol.FROZEN_CASES),
        "stdin_sha256": sha256(executor._encode_batch(protocol.FROZEN_CASES)).hexdigest(),
        "compile_environment": compile_environment,
        "compile_environment_sha256": executor._environment_sha256("compile", compile_environment),
        "run_environment": run_environment,
        "run_environment_sha256": executor._environment_sha256("run", run_environment),
        "case_count": len(protocol.FROZEN_CASES),
        "compile_limits_sha256": executor.COMPILE_LIMITS.sha256,
        "run_limits_sha256": executor.RUN_LIMITS.sha256,
    }
    preparation = executor.RustExecutorPreparationV1(
        **payload,
        sha256=executor._digest(executor._PREPARATION_NAMESPACE, payload),
    )
    preparation.authenticate_structure()
    return preparation


def _prepared_record() -> dict[str, object]:
    identity = _rustc_identity()
    preparation = _executor_preparation(identity)
    payload = {
        name: ROOT
        for name in protocol.ReadyUnspentReceiptV1.__dataclass_fields__
        if name not in {"schema", "status", "claim_boundary", "work", "sha256"}
    }
    payload.update(
        {
            "schema": protocol.MANIFEST_SCHEMA,
            "status": protocol.READY_UNSPENT,
            "claim_boundary": protocol.CLAIM_BOUNDARY,
            "candidate_source_set_sha256": protocol.CANDIDATE_SOURCE_SET_SHA256,
            "control_roster_sha256": protocol.CONTROL_ROSTER_SHA256,
            "batch_payload_sha256": protocol.BATCH_PAYLOAD_SHA256,
            "oracle_payload_sha256": protocol.ORACLE_PAYLOAD_SHA256,
            "executor_harness_sha256": protocol.rust_selected_action_executor.HARNESS_SHA256,
            "executor_batch_protocol_sha256": (
                protocol.rust_selected_action_executor.BATCH_PROTOCOL_SHA256
            ),
            "channel_action_handle": f"ra_{'1' * 64}",
            "channel_structural_sha256": "e" * 64,
            "mutex_action_handle": f"ra_{'2' * 64}",
            "task_projection_sha256s": TASK_ROOTS,
            "rustc_identity_sha256": identity.sha256,
            "executor_preparation_sha256": preparation.sha256,
            "executor_batch_sha256": preparation.batch_sha256,
            "executor_stdin_sha256": preparation.stdin_sha256,
            "work": protocol.ZERO_EVALUATION_WORK,
        }
    )
    receipt = protocol.ReadyUnspentReceiptV1(
        **payload,
        sha256=protocol._digest(
            protocol._READY_NAMESPACE,
            {**payload, "work": asdict(protocol.ZERO_EVALUATION_WORK)},
        ),
    )
    receipt.authenticate()
    return receipt.record()


def _recall_record(
    status: str,
    *,
    selected_candidate_id: str | None = None,
    scores: tuple[int, int] = (0, 0),
    checkpoint_sha256: str = ROOT,
    task_sha256: str = TASK_ROOTS["primary"],
) -> dict[str, object]:
    memory = protocol.knowledge_action
    symbols = ("std.sync.channel", "std.sync.mutex")
    raw = [[0 for _ in range(memory.OUTPUT_RADIX)] for _ in range(memory.OUTPUT_LANES)]
    if any(scores):
        for row in raw[:-1]:
            row[0] = memory.common_events.STOP_SCORE + 1
    for index, score in enumerate(scores):
        if score:
            raw[-1][index] = memory.common_events.STOP_SCORE + score
    physical_raw_scores = tuple(tuple(row) for row in raw)
    output = memory._CELLS.race_output(physical_raw_scores)
    selected_index = scores.index(max(scores)) if selected_candidate_id is not None else None
    selected_symbol = None if selected_index is None else symbols[selected_index]
    work = memory.KnowledgeActionRecallWorkV1(0, 0, 0, 0, 0, 0, 0, 0)
    payload = {
        "status": status,
        "selected_symbol": selected_symbol,
        "selected_candidate_id": selected_candidate_id,
        "matched_factor_count": 0 if status == memory.STOP_NO_MATCHED_FACTOR else 1,
        "unknown_factor_count": 1 if status == memory.STOP_NO_MATCHED_FACTOR else 0,
        "query_bucket_collision_count": 0,
        "candidate_scores": tuple(zip(symbols, scores, strict=True)),
        "physical_raw_scores": physical_raw_scores,
        "physical_output_record": output.record(),
        "work": asdict(work),
        "checkpoint_sha256": checkpoint_sha256,
        "task_sha256": task_sha256,
        "candidate_set_sha256": ROOT,
        "selected_candidate_sha256": ROOT if selected_candidate_id is not None else None,
    }
    recall = memory.KnowledgeActionRecallV1(
        **{**payload, "work": work},
        sha256=memory._digest(memory._RECALL_NAMESPACE, payload),
    )
    recall.authenticate()
    return protocol._plain(recall.record())


def _controls_record() -> dict[str, object]:
    prepared = _prepared_record()
    work = _zero_work()
    work["recall_count"] = 9
    channel_id = prepared["channel_action_handle"]
    mutex_id = prepared["mutex_action_handle"]
    primary = _recall_record(
        protocol.knowledge_action.RECALLED,
        selected_candidate_id=channel_id,
        scores=(2, 0),
    )
    items = [
        {"name": "primary", "kind": "physical_recall", "recall": primary},
        {
            "name": "contrast",
            "kind": "physical_recall",
            "recall": _recall_record(
                protocol.knowledge_action.RECALLED,
                selected_candidate_id=mutex_id,
                scores=(0, 2),
                task_sha256=TASK_ROOTS["contrast"],
            ),
        },
        {
            "name": "source_order_reversal",
            "kind": "physical_recall",
            "recall": primary,
        },
        {
            "name": "cell_off",
            "kind": "physical_recall",
            "recall": _recall_record(
                protocol.knowledge_action.STOP_CELL_OFF,
                scores=(2, 0),
            ),
        },
        {
            "name": "plasticity_off",
            "kind": "physical_recall",
            "recall": _recall_record(
                protocol.knowledge_action.STOP_PHYSICAL_RACE,
                checkpoint_sha256=prepared["no_plasticity_checkpoint_sha256"],
            ),
        },
        {
            "name": "bound_memory_splice",
            "kind": "authority_control",
            "status": "STOP_BOUND_MEMORY_SPLICE",
            "physical_recall_count": 0,
        },
        {
            "name": "shared_factor_thread",
            "kind": "physical_recall",
            "recall": _recall_record(
                protocol.knowledge_action.STOP_PHYSICAL_RACE,
                scores=(1, 1),
                task_sha256=TASK_ROOTS["shared_factor_thread"],
            ),
        },
        {
            "name": "unknown_factor",
            "kind": "physical_recall",
            "recall": _recall_record(
                protocol.knowledge_action.STOP_NO_MATCHED_FACTOR,
                task_sha256=TASK_ROOTS["unknown_factor"],
            ),
        },
        {
            "name": "single_factor_channel",
            "kind": "physical_recall",
            "recall": _recall_record(
                protocol.knowledge_action.RECALLED,
                selected_candidate_id=channel_id,
                scores=(1, 0),
                task_sha256=TASK_ROOTS["single_factor_channel"],
            ),
        },
        {
            "name": "single_factor_mutex",
            "kind": "physical_recall",
            "recall": _recall_record(
                protocol.knowledge_action.RECALLED,
                selected_candidate_id=mutex_id,
                scores=(0, 1),
                task_sha256=TASK_ROOTS["single_factor_mutex"],
            ),
        },
    ]
    payload = {
        "schema": protocol.CONTROL_SCHEMA,
        "ready_sha256": prepared["sha256"],
        "roster_sha256": protocol.CONTROL_ROSTER_SHA256,
        "items": items,
        "control_count": len(protocol.CONTROL_ROSTER),
        "physical_recall_count": 9,
        "work": work,
    }
    return {
        **payload,
        "sha256": protocol._digest(protocol._CONTROL_NAMESPACE, payload),
    }


def _execution_record() -> dict[str, object]:
    prepared = _prepared_record()
    controls = _controls_record()
    registry = protocol.rust_action_registry
    executor = protocol.rust_selected_action_executor
    identity = _rustc_identity()
    preparation = _executor_preparation(identity)
    materialization_payload = {
        "registry_sha256": prepared["registry_sha256"],
        "recall_sha256": controls["items"][0]["recall"]["sha256"],
        "action_handle": prepared["channel_action_handle"],
        "source_sha256": protocol._source_sha256(protocol.FROZEN_CHANNEL_SOURCE),
        "structural_sha256": prepared["channel_structural_sha256"],
        "source": protocol.FROZEN_CHANNEL_SOURCE,
    }
    materialization = registry.RustActionMaterializationV1(
        **materialization_payload,
        sha256=registry._digest(registry._MATERIALIZATION_NAMESPACE, materialization_payload),
    )
    dependency_payload = {
        "schema": executor.SCHEMA,
        "materialization_sha256": materialization.sha256,
        "harness_sha256": executor.HARNESS_SHA256,
        "selected_source_sha256": materialization.source_sha256,
        "dep_info_sha256": ROOT,
        "dependency_paths": (executor.HARNESS_FILENAME, executor.SELECTED_FILENAME),
        "dependency_content_sha256s": (
            (executor.HARNESS_FILENAME, executor.HARNESS_SHA256),
            (executor.SELECTED_FILENAME, materialization.source_sha256),
        ),
    }
    dependency = executor.RustDependencyReceiptV1(
        **dependency_payload,
        sha256=executor._digest(executor._DEPENDENCY_NAMESPACE, dependency_payload),
    )
    compile_payload = {
        "schema": executor.SCHEMA,
        "compiler_policy_sha256": executor.COMPILER_POLICY_SHA256,
        "rustc_identity_sha256": identity.sha256,
        "preparation_sha256": preparation.sha256,
        "materialization_sha256": materialization.sha256,
        "selected_source_sha256": materialization.source_sha256,
        "harness_sha256": executor.HARNESS_SHA256,
        "edition": executor.RUST_EDITION,
        "arguments": executor._compile_arguments(),
        "compile_limits_sha256": executor.COMPILE_LIMITS.sha256,
        "compile_environment_sha256": preparation.compile_environment_sha256,
        "compile_invocations": 1,
        "dependency_receipt_sha256": dependency.sha256,
        "binary_sha256": "b" * 64,
        "binary_size_bytes": 1,
        "stdout_sha256": executor._EMPTY_SHA256,
        "stderr_sha256": executor._EMPTY_SHA256,
    }
    compile_receipt = executor.RustCompileReceiptV1(
        **compile_payload,
        sha256=executor._digest(executor._COMPILE_NAMESPACE, compile_payload),
    )
    outputs = _synthetic_outputs()
    run_payload = {
        "schema": executor.SCHEMA,
        "compiler_policy_sha256": executor.COMPILER_POLICY_SHA256,
        "preparation_sha256": preparation.sha256,
        "compile_receipt_sha256": compile_receipt.sha256,
        "materialization_sha256": materialization.sha256,
        "batch_protocol_sha256": executor.BATCH_PROTOCOL_SHA256,
        "batch_sha256": preparation.batch_sha256,
        "stdin_sha256": preparation.stdin_sha256,
        "stdout_sha256": sha256(executor._encode_batch(outputs)).hexdigest(),
        "stderr_sha256": executor._EMPTY_SHA256,
        "output_sha256": executor._digest(executor._BATCH_NAMESPACE, outputs),
        "run_limits_sha256": executor.RUN_LIMITS.sha256,
        "run_environment_sha256": preparation.run_environment_sha256,
        "run_invocations": 1,
        "case_count": len(outputs),
        "returncode": 0,
    }
    run_receipt = executor.RustRunReceiptV1(
        **run_payload,
        sha256=executor._digest(executor._RUN_NAMESPACE, run_payload),
    )
    execution_payload = {
        "materialization_sha256": materialization.sha256,
        "rustc_identity_sha256": identity.sha256,
        "preparation_sha256": preparation.sha256,
        "dependency_receipt_sha256": dependency.sha256,
        "compile_receipt_sha256": compile_receipt.sha256,
        "run_receipt_sha256": run_receipt.sha256,
        "outputs": outputs,
    }
    selected = executor.RustSelectedActionExecutionV1(
        materialization=materialization,
        rustc_identity=identity,
        preparation=preparation,
        dependency_receipt=dependency,
        compile_receipt=compile_receipt,
        run_receipt=run_receipt,
        outputs=outputs,
        sha256=executor._digest(executor._EXECUTION_NAMESPACE, execution_payload),
    )
    selected.authenticate_structure()
    work = _zero_work()
    work.update(
        {
            "recall_count": 9,
            "selected_source_materialization_count": 1,
            "selected_executor_call_count": 1,
            "authenticated_rustc_compile_receipt_count": 1,
            "authenticated_binary_run_receipt_count": 1,
        }
    )
    payload = {
        "schema": protocol.EXECUTION_SCHEMA,
        "control_bundle_sha256": controls["sha256"],
        "execution": asdict(selected),
        "work": work,
    }
    return protocol._plain(
        {
            **payload,
            "sha256": protocol._digest(protocol._EXECUTION_NAMESPACE, payload),
        }
    )


def _resign_execution_tree(record: dict[str, object]) -> dict[str, object]:
    registry = protocol.rust_action_registry
    executor = protocol.rust_selected_action_executor
    copied = protocol._plain(record)
    selected = copied["execution"]
    materialization = selected["materialization"]
    materialization["sha256"] = registry._digest(
        registry._MATERIALIZATION_NAMESPACE,
        {key: value for key, value in materialization.items() if key != "sha256"},
    )
    dependency = selected["dependency_receipt"]
    dependency["materialization_sha256"] = materialization["sha256"]
    dependency["sha256"] = executor._digest(
        executor._DEPENDENCY_NAMESPACE,
        {key: value for key, value in dependency.items() if key != "sha256"},
    )
    compile_receipt = selected["compile_receipt"]
    compile_receipt["materialization_sha256"] = materialization["sha256"]
    compile_receipt["dependency_receipt_sha256"] = dependency["sha256"]
    compile_receipt["sha256"] = executor._digest(
        executor._COMPILE_NAMESPACE,
        {key: value for key, value in compile_receipt.items() if key != "sha256"},
    )
    run_receipt = selected["run_receipt"]
    run_receipt["materialization_sha256"] = materialization["sha256"]
    run_receipt["compile_receipt_sha256"] = compile_receipt["sha256"]
    run_receipt["sha256"] = executor._digest(
        executor._RUN_NAMESPACE,
        {key: value for key, value in run_receipt.items() if key != "sha256"},
    )
    selected["sha256"] = executor._digest(
        executor._EXECUTION_NAMESPACE,
        {
            "materialization_sha256": materialization["sha256"],
            "rustc_identity_sha256": selected["rustc_identity"]["sha256"],
            "preparation_sha256": selected["preparation"]["sha256"],
            "dependency_receipt_sha256": dependency["sha256"],
            "compile_receipt_sha256": compile_receipt["sha256"],
            "run_receipt_sha256": run_receipt["sha256"],
            "outputs": selected["outputs"],
        },
    )
    copied["sha256"] = protocol._digest(
        protocol._EXECUTION_NAMESPACE,
        {key: value for key, value in copied.items() if key != "sha256"},
    )
    return copied


def _oracle_record(*, matches: bool | None = None) -> dict[str, object]:
    execution = _execution_record()
    outputs = tuple(tuple(row) for row in execution["execution"]["outputs"])
    actual_match = (
        protocol._digest(protocol._ORACLE_NAMESPACE, outputs) == protocol.ORACLE_PAYLOAD_SHA256
    )
    serialized_match = actual_match if matches is None else matches
    work = _zero_work()
    work.update(
        {
            "recall_count": 9,
            "selected_source_materialization_count": 1,
            "selected_executor_call_count": 1,
            "authenticated_rustc_compile_receipt_count": 1,
            "authenticated_binary_run_receipt_count": 1,
            "oracle_read_count": 1,
        }
    )
    payload = {
        "schema": protocol.ADJUDICATION_SCHEMA,
        "status": (protocol.TERMINAL_PASS if serialized_match else protocol.TERMINAL_STOP),
        "oracle_file_sha256": protocol.EXPECTED_ORACLE_FILE_SHA256,
        "oracle_payload_sha256": protocol.ORACLE_PAYLOAD_SHA256,
        "batch_payload_sha256": protocol.BATCH_PAYLOAD_SHA256,
        "execution_sha256": execution["sha256"],
        "outputs_match": serialized_match,
        "work": work,
    }
    return {
        **payload,
        "sha256": protocol._digest(protocol._ADJUDICATION_NAMESPACE, payload),
    }


def _backend(
    events: list[str],
    paths: runner._RunPaths,
    *,
    controls: Callable[[], object] | None = None,
    execute: Callable[[], object] | None = None,
    oracle: Callable[[], object] | None = None,
) -> runner._GateBackend:
    prepared = object()
    control_bundle = object()
    execution = object()

    def prepare() -> object:
        events.append("prepare")
        assert not paths.spent.exists()
        return prepared

    def run_controls(value: object) -> object:
        assert value is prepared
        assert paths.spent.is_file()
        events.append("controls")
        if controls is not None:
            return controls()
        return control_bundle

    def execute_selected(value: object, supplied_controls: object) -> object:
        assert value is prepared
        assert supplied_controls is control_bundle
        assert events[-1] == "controls"
        events.append("execute")
        if execute is not None:
            return execute()
        return execution

    def open_oracle(
        value: object,
        supplied_controls: object,
        supplied_execution: object,
    ) -> object:
        assert value is prepared
        assert supplied_controls is control_bundle
        assert supplied_execution is execution
        assert events[-1] == "execute"
        events.append("oracle")
        if oracle is not None:
            return oracle()
        return _oracle_record()

    return runner._GateBackend(
        name="synthetic-rust-gate",
        real=False,
        prepare=prepare,
        prepared_record=lambda value: _prepared_record(),
        manifest_sha256=lambda value: "8" * 64,
        run_controls=run_controls,
        controls_record=lambda value: _controls_record(),
        execute_selected=execute_selected,
        execution_record=lambda value: _execution_record(),
        open_oracle=open_oracle,
    )


def _paths(tmp_path: Path) -> runner._RunPaths:
    return runner._RunPaths(tmp_path / "synthetic.spent.json", tmp_path / "synthetic.result.json")


def test_synthetic_runner_enforces_marker_controls_execution_oracle_chronology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        protocol,
        "ORACLE_PAYLOAD_SHA256",
        protocol._digest(protocol._ORACLE_NAMESPACE, _synthetic_outputs()),
    )

    artifact = runner._run_with_backend(_backend(events, paths), paths)

    assert artifact["verdict"] == runner.PASS
    assert events == ["prepare", "controls", "execute", "oracle"]
    assert type(artifact["controls"]["items"]) is list
    assert type(artifact["execution"]["execution"]["outputs"]) is list
    assert paths.spent.is_file()
    assert paths.result.is_file()
    assert artifact["runner_calls"] == {
        "control_phase_calls": 1,
        "selected_execution_calls": 1,
        "oracle_open_calls": 1,
    }
    assert runner._load_terminal_result(paths) == artifact


def test_fully_resigned_terminal_contradiction_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    artifact = runner._run_with_backend(_backend([], paths), paths)
    payload = dict(artifact)
    payload.pop("sha256")
    payload["state"] = (
        protocol.TERMINAL_PASS
        if artifact["state"] == protocol.TERMINAL_STOP
        else protocol.TERMINAL_STOP
    )
    forged = runner._signed(runner._RESULT_NAMESPACE, payload)
    paths.result.write_bytes(runner._canonical_document(forged))

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="terminal result semantics",
    ):
        runner._load_terminal_result(paths)


def test_terminal_runner_call_mapping_requires_exact_int_types(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    artifact = runner._run_with_backend(_backend([], paths), paths)
    payload = protocol._plain(artifact)
    payload.pop("sha256")
    payload["runner_calls"]["oracle_open_calls"] = True
    forged = runner._signed(runner._RESULT_NAMESPACE, payload)
    paths.result.write_bytes(runner._canonical_document(forged))

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="terminal result semantics",
    ):
        runner._load_terminal_result(paths)


def test_fully_redigested_primary_recall_cross_bind_mismatch_is_rejected() -> None:
    prepared = _prepared_record()
    controls = _controls_record()
    execution = _execution_record()
    execution["execution"]["materialization"]["recall_sha256"] = "9" * 64
    forged = _resign_execution_tree(execution)

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="selected execution receipt",
    ):
        runner._authenticate_execution_record(forged, prepared, controls)


def test_fully_redigested_control_task_mismatch_is_rejected() -> None:
    memory = protocol.knowledge_action
    prepared = _prepared_record()
    forged = protocol._plain(_controls_record())
    recall = forged["items"][0]["recall"]
    recall["task_sha256"] = TASK_ROOTS["contrast"]
    recall["sha256"] = memory._digest(
        memory._RECALL_NAMESPACE,
        {key: value for key, value in recall.items() if key != "sha256"},
    )
    forged["items"][2]["recall"] = protocol._plain(recall)
    forged["sha256"] = protocol._digest(
        protocol._CONTROL_NAMESPACE,
        {key: value for key, value in forged.items() if key != "sha256"},
    )

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="serialized causal-control outcome",
    ):
        runner._authenticate_controls_record(forged, prepared)


@pytest.mark.parametrize(
    "field",
    ("control_count", "physical_recall_count", "work", "splice"),
)
def test_control_integer_counters_require_exact_int_type(field: str) -> None:
    prepared = _prepared_record()
    forged = protocol._plain(_controls_record())
    if field == "control_count":
        forged["control_count"] = float(forged["control_count"])
    elif field == "physical_recall_count":
        forged["physical_recall_count"] = float(forged["physical_recall_count"])
    elif field == "work":
        forged["work"]["recall_count"] = float(forged["work"]["recall_count"])
    else:
        forged["items"][5]["physical_recall_count"] = False
    forged["sha256"] = protocol._digest(
        protocol._CONTROL_NAMESPACE,
        {key: value for key, value in forged.items() if key != "sha256"},
    )

    with pytest.raises(runner.FrozenRustDocsM0RunnerError):
        runner._authenticate_controls_record(forged, prepared)


def test_recall_physical_output_integer_mapping_requires_exact_types() -> None:
    memory = protocol.knowledge_action
    prepared = _prepared_record()
    forged = protocol._plain(_controls_record())
    recall = forged["items"][0]["recall"]
    output = recall["physical_output_record"]
    output["physical_cell_steps"] = float(output["physical_cell_steps"])
    recall["sha256"] = memory._digest(
        memory._RECALL_NAMESPACE,
        {key: value for key, value in recall.items() if key != "sha256"},
    )
    forged["items"][2]["recall"] = protocol._plain(recall)
    forged["sha256"] = protocol._digest(
        protocol._CONTROL_NAMESPACE,
        {key: value for key, value in forged.items() if key != "sha256"},
    )

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="serialized recall authentication",
    ):
        runner._authenticate_controls_record(forged, prepared)


def test_fully_redigested_nested_compile_contradiction_is_rejected() -> None:
    prepared = _prepared_record()
    controls = _controls_record()
    execution = _execution_record()
    execution["execution"]["compile_receipt"]["compile_invocations"] = 2
    forged = _resign_execution_tree(execution)

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="serialized executor receipt authentication",
    ):
        runner._authenticate_execution_record(forged, prepared, controls)


def test_nested_executor_receipt_round_trips_exactly() -> None:
    prepared = _prepared_record()
    controls = _controls_record()
    record = _execution_record()

    assert runner._authenticate_execution_record(record, prepared, controls) == record
    reconstructed = runner._execution_from_record(record["execution"])
    assert protocol._plain(asdict(reconstructed)) == record["execution"]


def test_fully_redigested_arbitrary_output_pass_is_rejected() -> None:
    execution = _execution_record()
    synthetic_output_root = protocol._digest(
        protocol._ORACLE_NAMESPACE,
        _synthetic_outputs(),
    )
    assert synthetic_output_root != protocol.ORACLE_PAYLOAD_SHA256
    forged = _oracle_record(matches=True)

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="oracle adjudication receipt changed",
    ):
        runner._authenticate_oracle_record(forged, execution)


def test_execution_and_oracle_work_require_exact_int_types() -> None:
    prepared = _prepared_record()
    controls = _controls_record()
    execution = _execution_record()
    forged_execution = protocol._plain(execution)
    forged_execution["work"]["selected_executor_call_count"] = True
    forged_execution["sha256"] = protocol._digest(
        protocol._EXECUTION_NAMESPACE,
        {key: value for key, value in forged_execution.items() if key != "sha256"},
    )
    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="selected execution wrapper changed",
    ):
        runner._authenticate_execution_record(forged_execution, prepared, controls)

    forged_oracle = protocol._plain(_oracle_record())
    forged_oracle["work"]["oracle_read_count"] = True
    forged_oracle["sha256"] = protocol._digest(
        protocol._ADJUDICATION_NAMESPACE,
        {key: value for key, value in forged_oracle.items() if key != "sha256"},
    )
    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="oracle adjudication receipt changed",
    ):
        runner._authenticate_oracle_record(forged_oracle, execution)


def test_control_failure_seals_stop_without_execution_or_oracle(tmp_path: Path) -> None:
    events: list[str] = []
    paths = _paths(tmp_path)

    def fail_controls() -> object:
        raise RuntimeError("synthetic control stop")

    artifact = runner._run_with_backend(
        _backend(events, paths, controls=fail_controls),
        paths,
    )

    assert artifact["verdict"] == runner.STOP
    assert artifact["stop_code"] == runner.STOP_CONTROL
    assert artifact["controls"] is None
    assert artifact["execution"] is None
    assert artifact["oracle"] is None
    assert events == ["prepare", "controls"]


def test_execution_failure_seals_stop_and_never_opens_oracle(tmp_path: Path) -> None:
    events: list[str] = []
    paths = _paths(tmp_path)

    def fail_execution() -> object:
        raise RuntimeError("synthetic executor stop")

    artifact = runner._run_with_backend(
        _backend(events, paths, execute=fail_execution),
        paths,
    )

    assert artifact["verdict"] == runner.STOP
    assert artifact["stop_code"] == runner.STOP_EXECUTION
    assert artifact["controls"] is not None
    assert artifact["execution"] is None
    assert artifact["oracle"] is None
    assert artifact["runner_calls"]["selected_execution_calls"] == 1
    assert artifact["runner_calls"]["oracle_open_calls"] == 0
    assert events == ["prepare", "controls", "execute"]


def test_fake_backend_is_rejected_from_canonical_path_before_prepare(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    paths = runner._RunPaths(runner.CANONICAL_SPENT_PATH, tmp_path / "result.json")
    backend = _backend(events, paths)

    with pytest.raises(
        runner.FrozenRustDocsM0RunnerError,
        match="synthetic backend",
    ):
        runner._run_with_backend(backend, paths)

    assert events == []


def test_terminal_stop_is_immutable_and_second_attempt_does_not_prepare(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    first_events: list[str] = []

    def fail_controls() -> object:
        raise RuntimeError("synthetic stop")

    first = runner._run_with_backend(
        _backend(first_events, paths, controls=fail_controls),
        paths,
    )
    second_events: list[str] = []

    with pytest.raises(runner.GateAlreadyTerminalError):
        runner._run_with_backend(_backend(second_events, paths), paths)

    assert first["state"] == protocol.TERMINAL_STOP
    assert second_events == []
    assert runner._load_terminal_result(paths) == first


def test_crash_sealer_performs_no_backend_or_oracle_work(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    events: list[str] = []
    backend = _backend(events, paths)
    ready = _prepared_record()
    marker, marker_file_sha256 = runner._create_spend_marker(
        backend,
        paths,
        ready,
        "8" * 64,
    )

    artifact = runner._seal_interrupted_spend(paths)

    assert events == []
    assert artifact["verdict"] == runner.STOP
    assert artifact["stop_code"] == runner.STOP_INTERRUPTED
    assert artifact["interrupted_prior_work_unknown"] is True
    assert artifact["controls"] is None
    assert artifact["execution"] is None
    assert artifact["oracle"] is None
    assert artifact["spend_marker_sha256"] == marker["sha256"]
    assert artifact["spend_marker_file_sha256"] == marker_file_sha256
    assert artifact["runner_calls"] == {
        "control_phase_calls": 0,
        "selected_execution_calls": 0,
        "oracle_open_calls": 0,
    }


def test_active_attempt_owner_lock_prevents_crash_sealer(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    marker_backend = _backend([], paths)
    runner._create_spend_marker(
        marker_backend,
        paths,
        _prepared_record(),
        "8" * 64,
    )

    with (
        runner._AttemptOwnerLock.acquire(paths.owner_lock),
        pytest.raises(runner.GateAttemptActiveError, match="still active"),
    ):
        runner._seal_interrupted_spend(paths)

    assert not paths.result.exists()


def test_public_terminal_load_holds_owner_lock_before_private_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spent = tmp_path / "show.spent.json"
    result = tmp_path / "show.result.json"
    owner = tmp_path / "show.owner.lock"
    monkeypatch.setattr(runner, "CANONICAL_SPENT_PATH", spent)
    monkeypatch.setattr(runner, "CANONICAL_RESULT_PATH", result)
    monkeypatch.setattr(runner, "CANONICAL_OWNER_LOCK_PATH", owner)

    def observe_owned_load(paths: runner._RunPaths) -> dict[str, object]:
        assert paths == runner._RunPaths(spent, result)
        with pytest.raises(runner.GateAttemptActiveError, match="still active"):
            runner._AttemptOwnerLock.acquire(paths.owner_lock)
        return {"verdict": runner.STOP}

    monkeypatch.setattr(runner, "_load_terminal_result", observe_owned_load)

    assert runner.load_terminal_result() == {"verdict": runner.STOP}


@pytest.mark.parametrize(
    ("writer", "publisher_owner"),
    (
        (runner._write_json_exclusive, runner),
        (protocol._write_json_exclusive, protocol),
    ),
)
def test_publish_failure_never_exposes_final_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: Callable[[Path, dict[str, object]], object],
    publisher_owner: object,
) -> None:
    final = tmp_path / f"{writer.__module__.rsplit('.', 1)[-1]}.json"

    def fail_publication(_source: Path, _target: Path) -> None:
        raise OSError("synthetic exclusive publication failure")

    monkeypatch.setattr(publisher_owner, "_publish_pending_exclusive", fail_publication)

    with pytest.raises(OSError, match="synthetic exclusive"):
        writer(final, {"complete": True})

    assert not final.exists()
    assert not tuple(tmp_path.glob(f".{final.name}.*.pending"))


@pytest.mark.skipif(os.name != "nt", reason="MoveFileExW behavior is Windows-specific")
@pytest.mark.parametrize(
    ("publisher", "collision_error"),
    (
        (runner._publish_pending_exclusive, runner.GateAlreadySpentError),
        (protocol._publish_pending_exclusive, protocol.FrozenRustDocsM0GateError),
    ),
)
def test_windows_write_through_publication_is_exclusive(
    tmp_path: Path,
    publisher: Callable[[Path, Path], None],
    collision_error: type[Exception],
) -> None:
    final = tmp_path / f"{publisher.__module__.rsplit('.', 1)[-1]}.final"
    pending = tmp_path / f"{final.name}.first.pending"
    pending.write_bytes(b"complete-first")

    publisher(pending, final)

    assert not pending.exists()
    assert final.read_bytes() == b"complete-first"
    collision = tmp_path / f"{final.name}.second.pending"
    collision.write_bytes(b"complete-second")
    with pytest.raises(collision_error, match="refusing to overwrite"):
        publisher(collision, final)
    assert collision.read_bytes() == b"complete-second"
    assert final.read_bytes() == b"complete-first"
