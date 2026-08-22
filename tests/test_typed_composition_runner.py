from __future__ import annotations

import base64
import inspect
import json
import os
import platform
import struct
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from humanbrain import native as native_runtime
from humanbrain import typed_program
from humanbrain import typed_composition_runner as runner
from humanbrain.typed_composition import CompletionRequest, load_composition_protocol
from humanbrain.typed_composition_runner import (
    A2EligibilityMetrics,
    EvidenceExistsError,
    ProtocolConfirmationError,
    _execute_protocol,
    _execute_protocol_synthetic,
    _race_integrity,
    _require_cpu_runtime_backend,
    _validate_checkpoint_accounting,
    a2_is_eligible,
    build_execution_plan,
    create_evidence_run,
    execute_protocol,
    score_sealed_completions,
    seal_all_lane_completions,
    validate_execution_confirmation,
    validate_source_bundle_confirmation,
)
from humanbrain.typed_program import Completion, Program, ProgramRaceAudit

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "typed_composition_protocol_v1.json"


def test_plan_binds_exact_protocol_and_complete_source_bundle_without_execution() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    plan = build_execution_plan(ROOT, PROTOCOL_PATH)

    assert plan["mode"] == "plan"
    assert plan["material_execution_authorized"] is False
    assert plan["protocol_sha256"] == protocol.raw_sha256
    assert len(plan["source_bundle_sha256"]) == 64
    assert tuple(item["path"] for item in plan["source_bundle"]) == (protocol.source_bundle_paths)
    assert all(len(item["sha256"]) == 64 for item in plan["source_bundle"])
    assert plan["source_bundle_limits"] == {
        "maximum_file_bytes": 1024 * 1024,
        "maximum_aggregate_bytes": 8 * 1024 * 1024,
        "actual_aggregate_bytes": sum(
            item["byte_count"] for item in plan["source_bundle"]
        ),
    }
    expected_native_inputs = {
        f"src/humanbrain/_native_src/{relative_path}"
        for relative_path, _path in native_runtime._native_build_input_paths()
    }
    assert expected_native_inputs <= set(protocol.source_bundle_paths)
    assert ".gitattributes" in protocol.source_bundle_paths
    assert plan["execution_requirements"] == {
        "protocol_confirmation": "exact raw protocol SHA-256",
        "source_bundle_confirmation": "exact reviewed source-bundle SHA-256",
        "evidence": "exclusive nonoverwrite creation",
        "resume": False,
        "rerun_same_run_id": False,
        "preserve_failure": True,
    }
    assert plan["run_hold"].startswith("No material TRAIN")
    assert plan["future_cohort"]["seed_exists"] is False
    assert plan["future_cohort"]["generated"] is False


def test_plan_fails_if_any_discovered_native_cargo_input_is_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = native_runtime._native_build_input_paths()
    monkeypatch.setattr(
        native_runtime,
        "_native_build_input_paths",
        lambda: (*paths, ("src/new_kernel.rs", Path("synthetic"))),
    )

    with pytest.raises(ValueError, match="omits a native Cargo build input"):
        build_execution_plan(ROOT, PROTOCOL_PATH)


def test_plan_discovers_native_paths_without_reading_native_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_manifest(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("planning must not read native build payloads twice")

    monkeypatch.setattr(native_runtime, "_native_build_input_manifest", forbidden_manifest)

    plan = build_execution_plan(ROOT, PROTOCOL_PATH)

    assert plan["material_execution_authorized"] is False


def test_plan_binds_every_transitive_local_runtime_module() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)

    required = runner._required_local_source_paths(ROOT)

    assert "src/humanbrain/__init__.py" in required
    assert set(required) <= set(protocol.source_bundle_paths)


def test_plan_fails_if_a_transitive_local_runtime_module_is_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_module = SimpleNamespace(
        __name__="humanbrain.procedural_runtime",
        __file__=str(ROOT / "src" / "humanbrain" / "procedural_runtime.py"),
    )
    monkeypatch.setattr(
        runner,
        "_LOCAL_RUNTIME_MODULES",
        (*runner._LOCAL_RUNTIME_MODULES, synthetic_module),
    )

    with pytest.raises(ValueError, match="transitive local runtime source"):
        build_execution_plan(ROOT, PROTOCOL_PATH)


def test_source_bundle_reads_are_bounded_before_evidence_or_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "oversized.py"
    oversized.write_bytes(b"x" * (runner._MAX_SOURCE_FILE_BYTES + 1))
    protocol = load_composition_protocol(PROTOCOL_PATH)
    grow_called = False

    def forbidden_grow(_protocol: object, _examples: object) -> object:
        nonlocal grow_called
        grow_called = True
        raise AssertionError("grow must not be reached")

    monkeypatch.setattr(runner, "_resolved_source", lambda _root, _relative: oversized)

    with pytest.raises(ValueError, match="source bundle file exceeds"):
        _execute_protocol_synthetic(
            ROOT,
            PROTOCOL_PATH,
            evidence_root=tmp_path / "evidence",
            run_id="oversized-source",
            confirmation=protocol.raw_sha256,
            source_bundle_confirmation="not-reached",
            grow_fn=forbidden_grow,
        )

    assert grow_called is False
    assert not (tmp_path / "evidence").exists()


def test_oversized_native_source_is_refused_before_evidence_model_or_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "oversized-native.rs"
    oversized.write_bytes(b"x" * (runner._MAX_SOURCE_FILE_BYTES + 1))
    protocol = load_composition_protocol(PROTOCOL_PATH)
    native_relative = "src/humanbrain/_native_src/src/lib.rs"
    original_resolved_source = runner._resolved_source

    def resolved_source(root: Path, relative_path: str) -> Path:
        if relative_path == native_relative:
            return oversized
        return original_resolved_source(root, relative_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized native input must fail before execution")

    monkeypatch.setattr(runner, "_resolved_source", resolved_source)
    monkeypatch.setattr(native_runtime, "build_native", forbidden)

    with pytest.raises(ValueError, match="source bundle file exceeds"):
        _execute_protocol_synthetic(
            ROOT,
            PROTOCOL_PATH,
            evidence_root=tmp_path / "evidence",
            run_id="oversized-native-source",
            confirmation=protocol.raw_sha256,
            source_bundle_confirmation="not-reached",
            grow_fn=forbidden,
        )

    assert not (tmp_path / "evidence").exists()


def test_source_bundle_aggregate_read_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_MAX_SOURCE_BUNDLE_BYTES", 1)

    with pytest.raises(ValueError, match="aggregate byte limit"):
        build_execution_plan(ROOT, PROTOCOL_PATH)


def test_execution_needs_the_exact_raw_protocol_sha_and_a_new_evidence_target(
    tmp_path: Path,
) -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)

    for confirmation in (None, "0" * 64, protocol.raw_sha256.upper()):
        with pytest.raises(ProtocolConfirmationError, match="exact raw protocol SHA-256"):
            validate_execution_confirmation(protocol, confirmation)
    validate_execution_confirmation(protocol, protocol.raw_sha256)

    plan = build_execution_plan(ROOT, PROTOCOL_PATH)
    for confirmation in (None, "0" * 64, str(plan["source_bundle_sha256"]).upper()):
        with pytest.raises(
            ProtocolConfirmationError,
            match="exact reviewed source-bundle SHA-256",
        ):
            validate_source_bundle_confirmation(plan, confirmation)
    validate_source_bundle_confirmation(plan, str(plan["source_bundle_sha256"]))

    evidence = create_evidence_run(tmp_path, "synthetic-boundary-test", b"{}")
    assert evidence.is_dir()
    assert (evidence / "000-plan.json").read_bytes() == b"{}"
    with pytest.raises(EvidenceExistsError, match="already exists"):
        create_evidence_run(tmp_path, "synthetic-boundary-test", b"{}")


def test_evidence_publication_handles_short_writes_and_is_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "record.json"
    original_write = os.write

    def short_write(fd: int, value: bytes | memoryview) -> int:
        return original_write(fd, bytes(value[:2]))

    monkeypatch.setattr(os, "write", short_write)
    runner._write_new(target, b"complete-record")
    assert target.read_bytes() == b"complete-record"
    assert not tuple(tmp_path.glob(".*.tmp"))
    with pytest.raises(EvidenceExistsError, match="already exists"):
        runner._write_new(target, b"replacement")
    assert target.read_bytes() == b"complete-record"
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_missing_parents_run_directory_and_publications_use_durable_primitives_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "artifacts" / "typed" / "evidence"
    calls: list[tuple[str, Path]] = []
    original_create = runner._create_new_directory
    original_publish = runner._publish_new

    def record_create(path: Path) -> None:
        calls.append(("directory", path.resolve()))
        original_create(path)

    def record_publish(temporary: Path, target: Path) -> None:
        calls.append(("publish", target.resolve()))
        original_publish(temporary, target)

    monkeypatch.setattr(runner, "_create_new_directory", record_create)
    monkeypatch.setattr(runner, "_publish_new", record_publish)

    run_directory = create_evidence_run(evidence_root, "durable-order", b"{}")
    runner._write_new(run_directory / "010-record.json", b"record")

    assert calls == [
        ("directory", (tmp_path / "artifacts").resolve()),
        ("directory", (tmp_path / "artifacts" / "typed").resolve()),
        ("directory", evidence_root.resolve()),
        ("directory", run_directory.resolve()),
        ("publish", (run_directory / "000-plan.json").resolve()),
        ("publish", (run_directory / "010-record.json").resolve()),
    ]


def test_missing_evidence_component_race_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "raced-parent" / "evidence"
    original_create = runner._create_new_directory

    def lose_first_creation_race(path: Path) -> None:
        if path.name == "raced-parent":
            path.mkdir()
        original_create(path)

    monkeypatch.setattr(runner, "_create_new_directory", lose_first_creation_race)

    with pytest.raises(EvidenceExistsError, match="already exists"):
        runner._durably_create_missing_directories(evidence_root)

    assert (tmp_path / "raced-parent").is_dir()
    assert not evidence_root.exists()


def test_publication_durability_failure_is_fail_closed_and_never_claims_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "record.json"

    original_publish = runner._publish_new

    def fail_after_publish(temporary: Path, published: Path) -> None:
        original_publish(temporary, published)
        raise OSError("synthetic directory flush failure")

    monkeypatch.setattr(runner, "_publish_new", fail_after_publish)

    with pytest.raises(OSError, match="synthetic directory flush failure"):
        runner._write_new(target, b"complete-record")

    # The exclusive publication is immutable partial evidence, but the failed call
    # cannot be mistaken for a durable success claim.
    assert target.read_bytes() == b"complete-record"
    assert not tuple(tmp_path.glob(".*.tmp"))
    monkeypatch.setattr(runner, "_publish_new", original_publish)
    with pytest.raises(EvidenceExistsError, match="already exists"):
        runner._write_new(target, b"retry")


def test_run_directory_durability_failure_preserves_the_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_create = runner._create_new_directory

    def fail_after_create(path: Path) -> None:
        nonlocal calls
        calls += 1
        original_create(path)
        if path.name == "directory-flush-failure":
            raise OSError("original run-directory flush failure")

    monkeypatch.setattr(runner, "_create_new_directory", fail_after_create)

    with pytest.raises(OSError, match="original run-directory flush failure"):
        create_evidence_run(tmp_path, "directory-flush-failure", b"{}")

    assert calls >= 1
    assert not (tmp_path / "directory-flush-failure" / "000-plan.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows MoveFileExW contract")
def test_windows_publication_uses_write_through_without_replace_and_maps_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path, int]] = []

    def fake_move(source: Path, target: Path, flags: int) -> int:
        calls.append((source, target, flags))
        return 183

    source = tmp_path / "source.tmp"
    target = tmp_path / "target.json"
    source.write_bytes(b"record")
    monkeypatch.setattr(runner, "_move_file_ex", fake_move)

    with pytest.raises(EvidenceExistsError, match="already exists"):
        runner._windows_move_new(source, target)

    assert calls == [(source, target, runner._MOVEFILE_WRITE_THROUGH)]
    assert runner._MOVEFILE_WRITE_THROUGH == 0x8

    monkeypatch.setattr(runner, "_move_file_ex", lambda *_args: 5)
    with pytest.raises(OSError, match="WinError 5"):
        runner._windows_move_new(source, target)


def test_windows_publication_does_not_resolve_the_final_target_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    target = tmp_path / "existing-reparse-point"
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == target:
            raise AssertionError("exclusive publication followed the final target")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    monkeypatch.setattr(runner, "_move_file_ex", lambda *_args: 183)

    with pytest.raises(EvidenceExistsError, match="already exists"):
        runner._windows_move_new(source, target)


def test_posix_publication_unlink_failure_syncs_parent_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.json"
    source.write_bytes(b"immutable record")
    calls: list[str] = []
    original_unlink = Path.unlink

    def fail_source_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == source:
            calls.append("unlink")
            raise OSError("synthetic unlink failure")
        original_unlink(path, *args, **kwargs)

    def record_sync(directory: Path) -> None:
        assert directory == tmp_path
        calls.append("sync")

    monkeypatch.setattr(Path, "unlink", fail_source_unlink)
    monkeypatch.setattr(runner, "_sync_posix_directory", record_sync)

    with pytest.raises(OSError, match="synthetic unlink failure"):
        runner._publish_posix_new(source, target)

    assert calls == ["unlink", "sync"]
    assert source.read_bytes() == b"immutable record"
    assert target.read_bytes() == b"immutable record"


def test_interrupted_evidence_temp_never_publishes_a_partial_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "record.json"

    def interrupted(fd: int, value: bytes) -> None:
        os.write(fd, value[:3])
        os.fsync(fd)
        raise OSError("synthetic interruption")

    monkeypatch.setattr(runner, "_write_temp_fully", interrupted)
    with pytest.raises(OSError, match="synthetic interruption"):
        runner._write_new(target, b"complete-record")

    assert not target.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_initial_plan_write_failure_preserves_a_terminal_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = runner._publish_new

    def fail_initial_plan(source: Path, target: Path) -> None:
        if Path(target).name == "000-plan.json":
            raise OSError("synthetic initial plan write failure")
        original_publish(source, target)

    monkeypatch.setattr(runner, "_publish_new", fail_initial_plan)

    with pytest.raises(OSError, match="synthetic initial plan write failure"):
        create_evidence_run(tmp_path, "plan-write-failure", b"{}")

    terminal = tmp_path / "plan-write-failure" / "999-terminal-failure.json"
    record = json.loads(terminal.read_text(encoding="utf-8"))
    assert record["error_type"] == "OSError"
    assert record["partial_evidence_preserved"] is True
    assert record["resume_allowed"] is False


def test_terminal_record_write_failure_never_masks_the_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_both_records(_source: Path, target: Path) -> None:
        if Path(target).name == "000-plan.json":
            raise OSError("original plan failure")
        if Path(target).name == "999-terminal-failure.json":
            raise PermissionError("secondary terminal failure")
        raise AssertionError("unexpected evidence target")

    monkeypatch.setattr(runner, "_publish_new", fail_both_records)

    with pytest.raises(OSError, match="original plan failure"):
        create_evidence_run(tmp_path, "double-write-failure", b"{}")


def test_public_executor_has_no_injectable_learner_or_decoder() -> None:
    parameters = inspect.signature(execute_protocol).parameters

    assert "grow_fn" not in parameters
    assert "complete_fn" not in parameters
    assert "root" not in parameters
    assert "source_bundle_confirmation" in parameters


def test_public_executor_rejects_a_protocol_from_another_checkout_before_evidence(
    tmp_path: Path,
) -> None:
    copied_protocol = tmp_path / PROTOCOL_PATH.name
    copied_protocol.write_bytes(PROTOCOL_PATH.read_bytes())

    with pytest.raises(ValueError, match="imported repository checkout"):
        execute_protocol(
            copied_protocol,
            evidence_root=tmp_path / "evidence",
            run_id="wrong-checkout",
            confirmation="not-reached",
            source_bundle_confirmation="not-reached",
        )

    assert not (tmp_path / "evidence").exists()


def test_private_production_executor_also_rejects_a_foreign_root_before_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="imported repository checkout"):
        _execute_protocol(
            tmp_path,
            PROTOCOL_PATH,
            evidence_root=tmp_path / "evidence",
            run_id="wrong-private-root",
            confirmation="not-reached",
            source_bundle_confirmation="not-reached",
            grow_fn=runner.grow,
            complete_fn=runner.complete,
            synthetic_test_only=False,
        )

    assert not (tmp_path / "evidence").exists()


def test_runtime_backend_is_frozen_to_the_same_native_cpu_path_for_all_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HUMANBRAIN_DEVICE", raising=False)
    assert _require_cpu_runtime_backend() == {
        "requested_device": "auto",
        "resolved_backend": "native-cpu",
        "reservoir_cells": 128,
        "cuda_minimum_cells": 100_000,
    }

    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="native CPU backend"):
        _require_cpu_runtime_backend()


def test_loaded_native_identity_is_portable_and_rehashes_the_attested_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_digest = "0" * 64
    library_path = (
        tmp_path
        / source_digest
        / "x86_64-pc-windows-msvc"
        / "release"
        / "humanbrain_kernel.dll"
    )
    library_path.parent.mkdir(parents=True)
    binary = b"synthetic-native-library"
    library_path.write_bytes(binary)
    identity_record = {
        "binary_sha256": sha256(binary).hexdigest(),
        "binary_size": len(binary),
        "source_digest": source_digest,
    }
    manifest_path = library_path.with_name(library_path.name + ".identity.json")
    manifest_path.write_text(
        json.dumps(identity_record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(native_runtime, "_LIBRARY", object())
    monkeypatch.setattr(native_runtime, "_LIBRARY_PATH", library_path)
    monkeypatch.setattr(
        native_runtime,
        "_LOADED_BUILD_PLAN",
        replace(native_runtime._native_build_plan(), source_digest=source_digest),
    )

    identity = native_runtime.loaded_library_identity()

    assert identity.to_record() == {
        "binary_sha256": sha256(binary).hexdigest(),
        "binary_size": 24,
        "source_digest": source_digest,
    }
    assert "path" not in identity.to_record()

    library_path.write_bytes(b"corrupted-native-library")
    with pytest.raises(native_runtime.NativeBuildError, match="integrity"):
        native_runtime.loaded_library_identity()

    manifest_path.unlink()
    with pytest.raises(native_runtime.NativeBuildError, match="no readable integrity manifest"):
        native_runtime.loaded_library_identity()


def test_internal_executor_cannot_mark_injected_dependencies_as_production(
    tmp_path: Path,
) -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)

    with pytest.raises(TypeError, match="fixed public grow and complete"):
        _execute_protocol(
            ROOT,
            PROTOCOL_PATH,
            evidence_root=tmp_path / "evidence",
            run_id="injected-production",
            confirmation=protocol.raw_sha256,
            source_bundle_confirmation="not-consulted-before-fixed-dependency-check",
            grow_fn=lambda _protocol, _examples: object(),
            complete_fn=lambda _checkpoint, _prompt, _signature: object(),
            synthetic_test_only=False,
        )

    assert not (tmp_path / "evidence").exists()


def test_all_lane_completions_are_sealed_before_labels_or_cases_are_needed() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    task_by_request = {
        (task.prompt, task.signature): task
        for task in (*protocol.train_tasks, *protocol.calibration_tasks)
    }
    calls: list[tuple[str, str, object]] = []

    class Checkpoint:
        def __init__(self, lane: str) -> None:
            self.lane = lane

    def fake_complete(
        checkpoint: Checkpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        # The test oracle remains outside the call boundary. Production passes
        # exactly these three values and never a task, teacher, case, or family.
        calls.append((checkpoint.lane, prompt, signature))
        task = task_by_request[(prompt, signature)]
        marker = checkpoint.lane.encode("ascii").hex().ljust(64, "0")
        return Completion(
            program=Program(task.teacher_tokens, task.signature),
            races=(),
            prompt_sha256=marker,
            model_input_sha256=marker,
            checkpoint_sha256=marker,
            checkpoint_cell_synapse_before=marker,
            checkpoint_cell_synapse_after=marker,
        )

    checkpoints = {lane.lane: Checkpoint(lane.lane) for lane in protocol.lanes}
    sealed = seal_all_lane_completions(
        protocol,
        checkpoints,
        complete_fn=fake_complete,
    )

    assert len(calls) == 3 * (42 + 18)
    assert len(sealed) == len(calls)
    assert tuple(item.sequence_number for item in sealed) == tuple(range(1, 181))
    assert all(item.record_bytes.startswith(b"{") for item in sealed)
    assert all(len(item.sha256) == 64 for item in sealed)
    assert tuple((item.lane, item.cohort) for item in sealed[:2]) == (
        ("A0", "TRAIN"),
        ("A0", "TRAIN"),
    )


def test_trusted_task_metadata_is_attached_only_after_every_raw_completion_seal() -> None:
    signature = typed_program.TypedSignature(
        entry_point="q_bcdfghjkl",
        parameters=(typed_program.ProgramParameter("datum", typed_program.ProgramType.Int()),),
        result=typed_program.ProgramType.Int(),
    )
    calls: list[str] = []
    expected_calls = 6

    class Task:
        def __init__(self, name: str, cohort: str) -> None:
            self.name = name
            self._cohort = cohort

        def completion_request(self) -> CompletionRequest:
            return CompletionRequest(prompt=self.name, signature=signature)

        @property
        def cohort(self) -> str:
            assert len(calls) == expected_calls
            return self._cohort

        @property
        def task_id(self) -> str:
            assert len(calls) == expected_calls
            return self.name

    lanes = tuple(SimpleNamespace(lane=lane) for lane in ("A0", "A1", "A2"))
    protocol = SimpleNamespace(
        lanes=lanes,
        train_tasks=(Task("train-one", "TRAIN"),),
        calibration_tasks=(Task("calibration-one", "CALIBRATION"),),
    )
    checkpoints = {lane.lane: SimpleNamespace(lane=lane.lane) for lane in lanes}

    def fake_complete(checkpoint: object, prompt: str, typed_signature: object) -> Completion:
        calls.append(prompt)
        marker = sha256(checkpoint.lane.encode("ascii")).hexdigest()
        return Completion(
            program=Program(("variable-0",), typed_signature),
            races=(),
            prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
            model_input_sha256=marker,
            checkpoint_sha256=marker,
            checkpoint_cell_synapse_before=marker,
            checkpoint_cell_synapse_after=marker,
        )

    sealed = seal_all_lane_completions(
        protocol,
        checkpoints,
        complete_fn=fake_complete,
    )

    assert len(calls) == expected_calls
    assert (
        tuple((item.cohort, item.task_id) for item in sealed)
        == (
            ("TRAIN", "train-one"),
            ("CALIBRATION", "calibration-one"),
        )
        * 3
    )


def test_a2_is_the_only_lane_and_only_exact_capability_gate_can_make_it_eligible() -> None:
    passing = A2EligibilityMetrics(
        train_replay_tasks=42,
        train_cases=126,
        calibration_replay_tasks=18,
        calibration_cases=54,
        common_integrity_gates_passed=True,
        decision_collision_count=0,
        competitive_train_margin_count=1,
        minimum_train_target_margin_pa=0.001,
    )
    assert a2_is_eligible("A0", passing) is False
    assert a2_is_eligible("A1", passing) is False
    assert a2_is_eligible("A2", passing) is True

    failing_variants = (
        replace(passing, train_replay_tasks=41),
        replace(passing, train_cases=125),
        replace(passing, calibration_replay_tasks=17),
        replace(passing, calibration_cases=53),
        replace(passing, common_integrity_gates_passed=False),
        replace(passing, decision_collision_count=1),
        replace(passing, competitive_train_margin_count=0),
        replace(passing, minimum_train_target_margin_pa=0.0),
        replace(passing, minimum_train_target_margin_pa=float("nan")),
    )
    assert all(not a2_is_eligible("A2", metrics) for metrics in failing_variants)


def test_race_integrity_binds_typed_state_winner_prefix_and_physical_values() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    task = next(task for task in protocol.train_tasks if len(task.teacher_tokens) == 1)
    state = typed_program._TypedHoleState(task.signature)
    allowed = tuple(typed_program._TOKEN_TO_ID[spec.token] for spec in state.allowed())
    winner = typed_program._TOKEN_TO_ID[task.teacher_tokens[0]]
    eligibility = struct.pack("<128d", *([1.0] * 128))
    race = {
        "accepted_token": task.teacher_tokens[0],
        "winner_cell_id": winner,
        "spike_time_ms": 1.0,
        "simulated_steps": 1,
        "prefix_sha256": sha256(b"[]").hexdigest(),
        "allowed_cell_ids": list(allowed),
        "allowed_output_currents_pa": [2.0 if cell_id == winner else 1.0 for cell_id in allowed],
        "excluded_cell_ids": [],
        "output_current_sha256": "7" * 64,
        "eligibility_f64le_base64": base64.b64encode(eligibility).decode("ascii"),
        "eligibility_sha256": sha256(eligibility).hexdigest(),
    }

    assert _race_integrity([race], task.teacher_tokens, task.signature)[0] is True

    invalid_records: list[dict[str, object]] = []
    wrong_winner = dict(race)
    wrong_winner["winner_cell_id"] = next(cell for cell in allowed if cell != winner)
    invalid_records.append(wrong_winner)
    wrong_prefix = dict(race)
    wrong_prefix["prefix_sha256"] = "0" * 64
    invalid_records.append(wrong_prefix)
    wrong_allowed = dict(race)
    wrong_allowed["allowed_cell_ids"] = list(reversed(allowed))
    invalid_records.append(wrong_allowed)
    no_spike = dict(race)
    no_spike["spike_time_ms"] = 0.0
    invalid_records.append(no_spike)
    no_steps = dict(race)
    no_steps["simulated_steps"] = 0
    invalid_records.append(no_steps)
    bad_current_digest = dict(race)
    bad_current_digest["output_current_sha256"] = "G" * 64
    invalid_records.append(bad_current_digest)
    nonfinite_eligibility = struct.pack("<128d", float("nan"), *([1.0] * 127))
    bad_eligibility = dict(race)
    bad_eligibility["eligibility_f64le_base64"] = base64.b64encode(nonfinite_eligibility).decode(
        "ascii"
    )
    bad_eligibility["eligibility_sha256"] = sha256(nonfinite_eligibility).hexdigest()
    invalid_records.append(bad_eligibility)

    assert all(
        not _race_integrity([invalid], task.teacher_tokens, task.signature)[0]
        for invalid in invalid_records
    )


def test_synthetic_execution_failure_is_preserved_and_cannot_resume(
    tmp_path: Path,
) -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    source_bundle_sha256 = build_execution_plan(ROOT, PROTOCOL_PATH)["source_bundle_sha256"]

    class FakeCheckpoint:
        def __init__(self, layer_protocol: object, examples: tuple[object, ...]) -> None:
            self.lane = layer_protocol.lane
            digest = sha256()
            input_hashes: list[str] = []
            sequence_counts: list[int] = []
            teacher_counts: list[int] = []
            for example in examples:
                input_ids = typed_program._model_input(
                    example.prompt,
                    example.signature,
                    framing_order=layer_protocol.framing_order,
                )
                teacher_ids = typed_program._program_token_ids(example.teacher_tokens)
                input_bytes = typed_program._token_stream_bytes(input_ids)
                input_hashes.append(sha256(input_bytes).hexdigest())
                sequence_counts.append(len(input_ids) + len(teacher_ids))
                teacher_counts.append(len(teacher_ids))
                digest.update(input_bytes)
                digest.update(typed_program._token_stream_bytes(teacher_ids))
            self.training_digest = digest.hexdigest()
            self.training_input_sha256 = tuple(input_hashes)
            self.training_sequence_token_counts = tuple(sequence_counts)
            self.training_teacher_token_counts = tuple(teacher_counts)
            self.protocol_sha256 = sha256(typed_program._protocol_json(layer_protocol)).hexdigest()
            self.cell_synapse_sha256 = sha256(("cells-" + self.lane).encode("ascii")).hexdigest()
            self.sha256 = sha256(self.to_bytes()).hexdigest()

        def to_bytes(self) -> bytes:
            return ("checkpoint-" + self.lane).encode("ascii")

    def failing_grow(layer_protocol: object, examples: object) -> FakeCheckpoint:
        lane = layer_protocol.lane
        if lane == "A1":
            raise RuntimeError("synthetic grow failure")
        return FakeCheckpoint(layer_protocol, tuple(examples))

    evidence_root = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="synthetic grow failure"):
        _execute_protocol_synthetic(
            ROOT,
            PROTOCOL_PATH,
            evidence_root=evidence_root,
            run_id="synthetic-failure",
            confirmation=protocol.raw_sha256,
            source_bundle_confirmation=source_bundle_sha256,
            grow_fn=failing_grow,
        )

    run_directory = evidence_root / "synthetic-failure"
    assert (run_directory / "000-plan.json").is_file()
    assert (run_directory / "010-checkpoint-A0.json").is_file()
    assert (run_directory / "999-terminal-failure.json").is_file()
    with pytest.raises(EvidenceExistsError, match="already exists"):
        _execute_protocol_synthetic(
            ROOT,
            PROTOCOL_PATH,
            evidence_root=evidence_root,
            run_id="synthetic-failure",
            confirmation=protocol.raw_sha256,
            source_bundle_confirmation=source_bundle_sha256,
            grow_fn=failing_grow,
        )


def test_synthetic_execution_records_non_gating_lane_costs_and_runtime_provenance(
    tmp_path: Path,
) -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    source_bundle_sha256 = build_execution_plan(ROOT, PROTOCOL_PATH)["source_bundle_sha256"]
    task_by_request = {
        (task.prompt, task.signature): task
        for task in (*protocol.train_tasks, *protocol.calibration_tasks)
    }

    class StepClock:
        def __init__(self) -> None:
            self.wall_value = 0
            self.process_value = 0

        def wall_ns(self) -> int:
            value = self.wall_value
            self.wall_value += 10
            return value

        def process_ns(self) -> int:
            value = self.process_value
            self.process_value += 3
            return value

    class Checkpoint:
        def __init__(self, layer_protocol: object, examples: tuple[object, ...]) -> None:
            self.lane = layer_protocol.lane
            digest = sha256()
            input_hashes: list[str] = []
            sequence_counts: list[int] = []
            teacher_counts: list[int] = []
            for example in examples:
                input_ids = typed_program._model_input(
                    example.prompt,
                    example.signature,
                    framing_order=layer_protocol.framing_order,
                )
                teacher_ids = typed_program._program_token_ids(example.teacher_tokens)
                input_bytes = typed_program._token_stream_bytes(input_ids)
                input_hashes.append(sha256(input_bytes).hexdigest())
                sequence_counts.append(len(input_ids) + len(teacher_ids))
                teacher_counts.append(len(teacher_ids))
                digest.update(input_bytes)
                digest.update(typed_program._token_stream_bytes(teacher_ids))
            self.training_digest = digest.hexdigest()
            self.training_input_sha256 = tuple(input_hashes)
            self.training_sequence_token_counts = tuple(sequence_counts)
            self.training_teacher_token_counts = tuple(teacher_counts)
            self.protocol_sha256 = sha256(
                typed_program._protocol_json(layer_protocol)
            ).hexdigest()
            self.cell_synapse_sha256 = sha256(
                ("cells-" + self.lane).encode("ascii")
            ).hexdigest()
            self.sha256 = sha256(self.to_bytes()).hexdigest()

        def to_bytes(self) -> bytes:
            return ("checkpoint-" + self.lane).encode("ascii")

    def synthetic_complete(
        checkpoint: Checkpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        task = task_by_request[(prompt, signature)]
        return Completion(
            program=Program(task.teacher_tokens, task.signature),
            races=(),
            prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
            model_input_sha256=sha256(
                typed_program._token_stream_bytes(
                    typed_program._model_input(
                        prompt,
                        task.signature,
                        framing_order=typed_program.Layer1Protocol.for_lane(
                            checkpoint.lane
                        ).framing_order,
                    )
                )
            ).hexdigest(),
            checkpoint_sha256=checkpoint.sha256,
            checkpoint_cell_synapse_before=checkpoint.cell_synapse_sha256,
            checkpoint_cell_synapse_after=checkpoint.cell_synapse_sha256,
        )

    evaluation = _execute_protocol_synthetic(
        ROOT,
        PROTOCOL_PATH,
        evidence_root=tmp_path / "evidence",
        run_id="synthetic-measurements",
        confirmation=protocol.raw_sha256,
        source_bundle_confirmation=source_bundle_sha256,
        grow_fn=lambda layer, examples: Checkpoint(layer, examples),
        complete_fn=synthetic_complete,
        clock=runner._ExecutionClock(
            wall_ns=StepClock().wall_ns,
            process_ns=StepClock().process_ns,
        ),
    )

    measurements = evaluation.record["performance_measurements"]
    assert measurements["capability_gating"] is False
    for lane in ("A0", "A1", "A2"):
        lane_measurement = measurements["lanes"][lane]
        assert lane_measurement == {
            "grow_process_ns": 3,
            "grow_wall_ns": 10,
            "checkpoint_bytes": len(("checkpoint-" + lane).encode("ascii")),
            "completion_sealing_process_ns": 3,
            "completion_sealing_wall_ns": 10,
            "completion_count": 60,
            "race_count": 0,
        }
    provenance = evaluation.record["runtime_provenance"]
    assert provenance["python_implementation"]
    assert provenance["python_version"]
    assert provenance["numpy_version"]
    assert provenance["os"] == {
        "machine": platform.machine(),
        "release": platform.release(),
        "system": platform.system(),
    }
    assert evaluation.a2_eligible is False
    persisted = json.loads(
        (
            tmp_path
            / "evidence"
            / "synthetic-measurements"
            / "030-evaluation.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["performance_measurements"] == measurements
    assert persisted["runtime_provenance"] == provenance


def test_checkpoint_accounting_binds_exact_inputs_observations_and_strengthens() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    lane = typed_program.Layer1Protocol.for_lane("A2")
    digest = sha256()
    input_hashes: list[str] = []
    sequence_counts: list[int] = []
    teacher_counts: list[int] = []
    for task in protocol.train_tasks:
        input_ids = typed_program._model_input(
            task.prompt,
            task.signature,
            framing_order=lane.framing_order,
        )
        teacher_ids = typed_program._program_token_ids(task.teacher_tokens)
        input_bytes = typed_program._token_stream_bytes(input_ids)
        input_hashes.append(sha256(input_bytes).hexdigest())
        sequence_counts.append(len(input_ids) + len(teacher_ids))
        teacher_counts.append(len(teacher_ids))
        digest.update(input_bytes)
        digest.update(typed_program._token_stream_bytes(teacher_ids))
    checkpoint = SimpleNamespace(
        lane="A2",
        training_digest=digest.hexdigest(),
        training_input_sha256=tuple(input_hashes),
        training_sequence_token_counts=tuple(sequence_counts),
        training_teacher_token_counts=tuple(teacher_counts),
        protocol_sha256=sha256(typed_program._protocol_json(lane)).hexdigest(),
    )

    accounting = _validate_checkpoint_accounting(protocol, "A2", checkpoint)

    assert accounting == {
        "training_sequences": 42,
        "teacher_targets_per_pass": 272,
        "framing_tokens_per_pass": 8_484,
        "observations_per_pass": 8_714,
        "corpus_passes": 3,
        "observation_calls": 26_142,
        "strengthen_calls": 816,
    }

    checkpoint.training_sequence_token_counts = tuple(sequence_counts[:-1])
    with pytest.raises(ValueError, match="accounting"):
        _validate_checkpoint_accounting(protocol, "A2", checkpoint)


def test_synthetic_perfect_seals_pass_exact_a2_gate_without_material_training() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    task_by_request = {
        (task.prompt, task.signature): task
        for task in (*protocol.train_tasks, *protocol.calibration_tasks)
    }

    class Checkpoint:
        def __init__(self, lane: str) -> None:
            self.lane = lane
            self.sha256 = sha256(lane.encode("ascii")).hexdigest()
            self.cell_synapse_sha256 = sha256((lane + "-cells").encode()).hexdigest()

    def perfect_completion(
        checkpoint: Checkpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        task = task_by_request[(prompt, signature)]
        state = typed_program._TypedHoleState(task.signature)
        prefix: list[str] = []
        races: list[ProgramRaceAudit] = []
        lane_offset = {"A0": 0, "A1": 10_000, "A2": 20_000}[checkpoint.lane]
        for position, token in enumerate(task.teacher_tokens):
            allowed = tuple(typed_program._TOKEN_TO_ID[spec.token] for spec in state.allowed())
            winner = typed_program._TOKEN_TO_ID[token]
            currents = tuple(2.0 if cell == winner else 1.0 for cell in allowed)
            value = float(lane_offset + task.opaque_ordinal * 100 + position + 1)
            races.append(
                ProgramRaceAudit(
                    accepted_token=token,
                    winner_cell_id=winner,
                    spike_time_ms=1.0,
                    simulated_steps=1,
                    prefix_sha256=sha256(
                        json.dumps(prefix, separators=(",", ":")).encode()
                    ).hexdigest(),
                    allowed_cell_ids=allowed,
                    allowed_output_currents_pa=currents,
                    excluded_cell_ids=(),
                    output_current_sha256="7" * 64,
                    eligibility_bytes=struct.pack("<128d", *([value] * 128)),
                )
            )
            state.consume(typed_program._SPEC_BY_TOKEN[token])
            prefix.append(token)
        return Completion(
            program=Program(task.teacher_tokens, task.signature),
            races=tuple(races),
            prompt_sha256=sha256(prompt.encode()).hexdigest(),
            model_input_sha256=sha256(
                typed_program._token_stream_bytes(
                    typed_program._model_input(
                        prompt,
                        task.signature,
                        framing_order=typed_program.Layer1Protocol.for_lane(
                            checkpoint.lane
                        ).framing_order,
                    )
                )
            ).hexdigest(),
            checkpoint_sha256=checkpoint.sha256,
            checkpoint_cell_synapse_before=checkpoint.cell_synapse_sha256,
            checkpoint_cell_synapse_after=checkpoint.cell_synapse_sha256,
        )

    checkpoints = {lane.lane: Checkpoint(lane.lane) for lane in protocol.lanes}
    sealed = seal_all_lane_completions(
        protocol,
        checkpoints,
        complete_fn=perfect_completion,
    )
    checkpoint_hashes = {lane: checkpoint.sha256 for lane, checkpoint in checkpoints.items()}
    checkpoint_cell_hashes = {
        lane: checkpoint.cell_synapse_sha256 for lane, checkpoint in checkpoints.items()
    }
    evaluation = score_sealed_completions(
        protocol,
        sealed,
        checkpoint_sha256s=checkpoint_hashes,
        checkpoint_cell_synapse_sha256s=checkpoint_cell_hashes,
    )

    assert evaluation.a2_eligible is True
    assert evaluation.a2_metrics.train_replay_tasks == 42
    assert evaluation.a2_metrics.train_cases == 126
    assert evaluation.a2_metrics.calibration_replay_tasks == 18
    assert evaluation.a2_metrics.calibration_cases == 54
    assert evaluation.a2_metrics.decision_collision_count == 0
    assert evaluation.a2_metrics.competitive_train_margin_count > 0
    assert evaluation.a2_metrics.minimum_train_target_margin_pa == 1.0

    def replace_record(
        item_index: int,
        mutate: object,
    ) -> tuple[object, ...]:
        items = list(sealed)
        record = json.loads(items[item_index].record_bytes.decode("utf-8"))
        mutate(record)
        raw = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        items[item_index] = replace(
            items[item_index],
            record_bytes=raw,
            sha256=sha256(raw).hexdigest(),
        )
        return tuple(items)

    bad_model_input = replace_record(
        0,
        lambda record: record.__setitem__("model_input_sha256", "0" * 64),
    )
    bad_program_digest = replace_record(
        0,
        lambda record: record["program"].__setitem__("sha256", "0" * 64),
    )
    bad_outer_cohort = list(sealed)
    bad_outer_cohort[0] = replace(bad_outer_cohort[0], cohort="CALIBRATION")

    bad_expected_cell_hashes = dict(checkpoint_cell_hashes)
    bad_expected_cell_hashes["A0"] = "0" * 64
    common_arguments = {
        "checkpoint_sha256s": checkpoint_hashes,
        "checkpoint_cell_synapse_sha256s": checkpoint_cell_hashes,
    }
    assert (
        score_sealed_completions(protocol, bad_model_input, **common_arguments).a2_eligible is False
    )
    assert (
        score_sealed_completions(protocol, bad_program_digest, **common_arguments).a2_eligible
        is False
    )
    assert (
        score_sealed_completions(protocol, tuple(bad_outer_cohort), **common_arguments).a2_eligible
        is False
    )
    assert (
        score_sealed_completions(
            protocol,
            sealed,
            checkpoint_sha256s=checkpoint_hashes,
            checkpoint_cell_synapse_sha256s=bad_expected_cell_hashes,
        ).a2_eligible
        is False
    )
