"""Capability tests for the authority-gated native temporal runtime."""

from __future__ import annotations

import ctypes
import copy
import hashlib
import importlib
import inspect
import math
import os
import platform
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import humanbrain.native_temporal_runtime as runtime_adapter
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    NativeTemporalRuntimeStatus,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)
from humanbrain.native import NativeBuildError, NativeLibraryIdentity
from experiments.r3_pilot.calibration_v3_protocol import ProtocolError


def _tiny_definition() -> NativeTemporalCircuitDefinition:
    config = NativeTemporalCircuitConfig(
        column_count=11,
        max_pyramidal_cells=11,
        segment_capacity=1,
        construction_seed=0x1020_3040,
        connected_permanence_ticks=250,
        initial_permanence_ticks=245,
        permanence_increment_ticks=17,
        negative_decrement_ticks=9,
        base_cells_per_column=1,
        novel_race_cap=1,
        prediction_quorum=11,
        max_segments_per_cell=2,
        theta_match=18,
        theta_active=20,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.GROWTH_OFF,
    )
    return NativeTemporalCircuitDefinition(
        config,
        {7: tuple(reversed(range(11)))},
        (0.0,) * 55,
    )


class _FakeFunction:
    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self, *args: object) -> int | None:
        return self.callback(*args)  # type: ignore[operator,no-any-return]


def _pointed(pointer: object, record: type[ctypes.Structure]) -> ctypes.Structure:
    return ctypes.cast(pointer, ctypes.POINTER(record)).contents


class _FakeTemporalLibrary:
    def __init__(self) -> None:
        self.checkpoint_bytes = b"CPV1" + bytes(range(252))
        self.destroyed: list[int] = []
        self.next_handle = 0x1234
        self.create_calls = 0
        self.import_calls = 0
        self.size_calls = 0
        self.export_calls = 0
        self.advance_override: object | None = None
        self.humanbrain_temporal_circuit_runtime_create_v1 = _FakeFunction(self._create)
        self.humanbrain_temporal_circuit_runtime_import_checkpoint_v1 = _FakeFunction(
            self._import
        )
        self.humanbrain_temporal_circuit_runtime_advance_block_v1 = _FakeFunction(
            self._advance
        )
        self.humanbrain_temporal_circuit_runtime_checkpoint_size_v1 = _FakeFunction(
            self._size
        )
        self.humanbrain_temporal_circuit_runtime_export_checkpoint_v1 = _FakeFunction(
            self._export
        )
        self.humanbrain_temporal_circuit_runtime_destroy = _FakeFunction(self._destroy)

    @staticmethod
    def _report(pointer: object) -> runtime_adapter._NativeReport:
        return ctypes.cast(
            pointer, ctypes.POINTER(runtime_adapter._NativeReport)
        ).contents

    def _publish_handle(self, handle_out: object, report_out: object) -> int:
        ctypes.cast(handle_out, ctypes.POINTER(ctypes.c_void_p))[0] = self.next_handle
        self.next_handle += 1
        self._report(report_out).status = 0
        return 0

    def _create(self, definition: object, handle_out: object, report_out: object) -> int:
        self.create_calls += 1
        native = _pointed(definition, runtime_adapter._NativeDefinition)
        assert native.token_code_count == 1
        assert native.token_codes[0].token_id == 7
        assert tuple(native.token_codes[0].columns) == tuple(range(11))
        return self._publish_handle(handle_out, report_out)

    def _import(self, request: object, handle_out: object, report_out: object) -> int:
        self.import_calls += 1
        native = _pointed(request, runtime_adapter._NativeImportRequest)
        assert ctypes.string_at(native.checkpoint_bytes, native.checkpoint_len) == (
            self.checkpoint_bytes
        )
        return self._publish_handle(handle_out, report_out)

    def _advance(self, handle: object, request: object, report_out: object) -> int:
        if self.advance_override is not None:
            return self.advance_override(  # type: ignore[operator,no-any-return]
                handle, request, report_out
            )
        native = _pointed(request, runtime_adapter._NativeAdvanceRequest)
        report = self._report(report_out)
        for index in range(native.token_count):
            step = native.output[index]
            step.token_id = native.token_ids[index]
            step.punished_segments = index + 3
            step.event_count = index + 101
            step.predicted = index % 2
            step.predicted_columns = index + 1
            step.allocated_segments = index + 2
            for column in range(11):
                step.inhibition_onset_ticks[column] = 10 + column
                step.column_simulated_ticks[column] = 30 + column
                for group in (
                    step.winner_spikes,
                    step.basket_spikes,
                    step.proximal_spikes,
                    step.proximal_relay_spikes,
                    step.winner_relay_spikes,
                ):
                    group[column].cell_id = 1000 + 100 * index + column
                    group[column].tick = 50 + column
            for context_index in range(22):
                step.winner_context[context_index] = 2000 + context_index
                step.proximal_context[context_index] = 3000 + context_index
                step.context[context_index] = 4000 + context_index
        report.completed_prefix = native.token_count
        return 0

    def _size(self, handle: object, authority: object, report_out: object) -> int:
        self.size_calls += 1
        del handle, authority
        report = self._report(report_out)
        report.required = len(self.checkpoint_bytes)
        return 0

    def _export(
        self,
        handle: object,
        authority: object,
        output: object,
        output_capacity: int,
        report_out: object,
    ) -> int:
        self.export_calls += 1
        del handle, authority
        assert output_capacity == len(self.checkpoint_bytes)
        ctypes.memmove(output, self.checkpoint_bytes, len(self.checkpoint_bytes))
        report = self._report(report_out)
        report.required = len(self.checkpoint_bytes)
        report.observed = len(self.checkpoint_bytes)
        return 0

    def _destroy(self, handle: object) -> None:
        value = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)
        self.destroyed.append(value)


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeTemporalLibrary, NativeLibraryIdentity]:
    library = _FakeTemporalLibrary()
    identity = NativeLibraryIdentity("cd" * 32, 123, "ab" * 32)
    monkeypatch.setattr(
        runtime_adapter,
        "_fresh_authority",
        lambda: runtime_adapter._FreshAuthority(b"\x11" * 32, b"\x22" * 32),
    )
    monkeypatch.setattr(runtime_adapter, "_load_library", lambda: library)
    monkeypatch.setattr(runtime_adapter, "_loaded_library_identity", lambda: identity)
    monkeypatch.setattr(
        runtime_adapter,
        "_current_native_source_digest",
        lambda: identity.source_digest,
    )
    return library, identity


def test_definition_canonicalizes_owned_inputs_without_policy_defaults() -> None:
    definition = _tiny_definition()

    assert definition.token_codes == ((7, tuple(range(11))),)
    assert definition.threshold_offsets_mv == (0.0,) * 55
    with pytest.raises(TypeError, match="context_policy"):
        NativeTemporalCircuitConfig(  # type: ignore[call-arg]
            column_count=11,
            max_pyramidal_cells=11,
            segment_capacity=1,
            construction_seed=1,
            connected_permanence_ticks=250,
            initial_permanence_ticks=250,
            permanence_increment_ticks=1,
            negative_decrement_ticks=1,
            base_cells_per_column=1,
            novel_race_cap=1,
            prediction_quorum=11,
            max_segments_per_cell=1,
            theta_match=18,
            theta_active=20,
            plasticity_policy=TemporalPlasticityPolicy.GROWTH_OFF,
        )
    with pytest.raises(ValueError, match="finite exact f64"):
        NativeTemporalCircuitDefinition(
            definition.config,
            definition.token_codes,
            (0.0,) * 54 + (math.nan,),
        )

    mutable_columns = list(reversed(range(11)))
    mutable_codes = {7: mutable_columns}
    mutable_thresholds = [0.0] * 55
    owned = NativeTemporalCircuitDefinition(
        definition.config, mutable_codes, mutable_thresholds
    )
    mutable_columns[0] = 999
    mutable_codes[8] = list(range(11))
    mutable_thresholds[0] = 17.0
    assert owned.token_codes == ((7, tuple(range(11))),)
    assert owned.threshold_offsets_mv == (0.0,) * 55


def test_import_defines_transport_without_loading_native_code(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import humanbrain.native as n; "
                "assert n._LIBRARY is None; "
                "import humanbrain.native_temporal_runtime; "
                "assert n._LIBRARY is None; "
                "import sys; "
                "assert 'experiments.r3_pilot.calibration_v3_protocol' not in sys.modules; "
                "assert 'experiments.r3_pilot.run_c1_cohort' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_ctypes_transport_matches_frozen_x64_layout_and_independent_oracles() -> None:
    expected_layouts = {
        runtime_adapter._NativeAuthority: (64, 1, (0, 32)),
        runtime_adapter._NativeConfig: (
            32,
            4,
            tuple(range(0, 16, 4)) + tuple(range(16, 24, 2)) + tuple(range(24, 32)),
        ),
        runtime_adapter._NativeTokenCode: (48, 4, (0, 4)),
        runtime_adapter._NativeDefinition: (128, 8, (0, 32, 40, 48, 56, 64)),
        runtime_adapter._NativeImportRequest: (144, 8, (0, 128, 136)),
        runtime_adapter._NativeSpike: (8, 4, (0, 4, 6)),
        runtime_adapter._NativeStep: (
            768,
            8,
            (0, 4, 8, 16, 17, 18, 19, 20, 42, 64, 152, 240, 328, 416, 504, 592, 680),
        ),
        runtime_adapter._NativeAdvanceRequest: (40, 8, (0, 8, 16, 24, 32, 33, 34)),
        runtime_adapter._NativeReport: (512, 8, (0, 4, 8, 16, 24, 32, 40, 42, 44, 48)),
    }
    for record, (size, alignment, offsets) in expected_layouts.items():
        assert ctypes.sizeof(record) == size
        assert ctypes.alignment(record) == alignment
        assert tuple(getattr(record, name).offset for name, _ctype in record._fields_) == offsets

    message = "oracle/R1: snowman=☃; café; little-endian; stop@column-race".encode()
    report_bytes = bytearray(512)
    struct.pack_into(
        "<IIQQQQHHI",
        report_bytes,
        0,
        8,
        len(message),
        0x0102_0304_0506_0708,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
        7,
        3,
        0,
    )
    report_bytes[48 : 48 + len(message)] = message
    assert hashlib.sha256(report_bytes).hexdigest() == (
        "5d7e11f6a153082ae9e39ef176f50e11e7a56067b023a6e6213efc4b0d97bda4"
    )

    report = runtime_adapter._NativeReport()
    (
        report.status,
        report.message_len,
        report.completed_prefix,
        report.required,
        report.observed,
        report.location,
        report.poison_code,
        report.stage_code,
    ) = struct.unpack_from("<IIQQQQHH", report_bytes)
    report.message_utf8[: len(message)] = message
    assert bytes(report) == report_bytes

    step_bytes = bytearray(768)
    struct.pack_into(
        "<IIQBBBB",
        step_bytes,
        0,
        0x0A1B_2C3D,
        0x1E2F_4051,
        0x6273_8495_A6B7_C8D9,
        1,
        7,
        5,
        0,
    )
    struct.pack_into("<11H", step_bytes, 20, *(101 + 2 * index for index in range(11)))
    struct.pack_into("<11H", step_bytes, 42, *(102 + 2 * index for index in range(11)))
    for index in range(55):
        struct.pack_into(
            "<IHH",
            step_bytes,
            64 + 8 * index,
            0x1020_3040 + index * 0x0001_0101,
            201 + index,
            0,
        )
    for index in range(66):
        struct.pack_into(
            "<I",
            step_bytes,
            504 + 4 * index,
            0x5060_7080 + index * 0x0001_0003,
        )
    assert step_bytes[42:64] == bytes.fromhex(
        "660068006a006c006e00700072007400760078007a00"
    )
    assert hashlib.sha256(step_bytes).hexdigest() == (
        "16b918d572aa21cc6323d16545f5567e67f8dde07f6599af911b1180a4825bda"
    )

    step = runtime_adapter._NativeStep.from_buffer_copy(step_bytes)
    assert bytes(step) == step_bytes


def test_frozen_abi_rejects_little_endian_arm64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")

    with pytest.raises(RuntimeError, match="little-endian x64"):
        runtime_adapter._assert_frozen_abi()


def test_fresh_authority_validates_protocol_commit_and_live_bytes_before_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    binding = {"files": {}}
    protocol = {"source_binding": binding}
    governing = importlib.import_module(
        "experiments.r3_pilot.calibration_v3_protocol"
    )
    monkeypatch.setattr(runtime_adapter, "_governing_protocol", lambda: governing)

    monkeypatch.setattr(
        governing,
        "build_protocol",
        lambda: calls.append("build") or protocol,
    )
    monkeypatch.setattr(
        governing,
        "validate_protocol",
        lambda value, *, require_runnable: calls.append(
            f"protocol:{value is protocol}:{require_runnable}"
        ),
    )
    monkeypatch.setattr(
        governing,
        "validate_committed_source_blobs",
        lambda root, value: calls.append(f"committed:{value is binding}"),
    )
    monkeypatch.setattr(
        governing,
        "validate_source_bytes",
        lambda root, value: calls.append(f"live:{value is binding}"),
    )
    monkeypatch.setattr(
        governing,
        "protocol_sha256",
        lambda: calls.append("protocol-digest") or "11" * 32,
    )
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_bundle_sha256",
        lambda root: calls.append("source-digest") or "22" * 32,
    )

    authority = runtime_adapter._fresh_authority()

    assert calls == [
        "build",
        "protocol:True:True",
        "committed:True",
        "live:True",
        "protocol-digest",
        "source-digest",
    ]
    assert authority.protocol_identity_sha256 == b"\x11" * 32
    assert authority.source_bundle_identity_sha256 == b"\x22" * 32


def test_current_no_go_fails_before_native_load(monkeypatch: pytest.MonkeyPatch) -> None:
    reached_loader = False

    def forbidden_loader() -> object:
        nonlocal reached_loader
        reached_loader = True
        raise AssertionError("native loader must not run for a non-runnable protocol")

    monkeypatch.setattr(runtime_adapter, "_load_library", forbidden_loader)

    with pytest.raises(ProtocolError, match="protocol is not runnable"):
        NativeTemporalCircuit.fresh(_tiny_definition())

    assert not reached_loader


def test_missing_governing_protocol_fails_before_native_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached_loader = False

    def unavailable(name: str) -> object:
        assert name == "experiments.r3_pilot.calibration_v3_protocol"
        raise ModuleNotFoundError("No module named 'experiments'")

    def forbidden_loader() -> object:
        nonlocal reached_loader
        reached_loader = True
        raise AssertionError("native loader must not run without governing protocol")

    monkeypatch.setattr(importlib, "import_module", unavailable)
    monkeypatch.setattr(runtime_adapter, "_load_library", forbidden_loader)

    with pytest.raises(RuntimeError, match="governing calibration-v3 protocol is unavailable"):
        NativeTemporalCircuit.fresh(_tiny_definition())

    assert not reached_loader


def test_public_capability_owns_handle_and_round_trips_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    definition = _tiny_definition()

    with NativeTemporalCircuit.fresh(definition) as circuit:
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(circuit)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(circuit)

        steps = circuit.advance_block([7, 7], mode=TemporalStepMode.READ_ONLY)
        assert len(steps) == 2
        assert steps[0].token_id == 7
        assert steps[0].predicted is False
        assert steps[1].predicted is True
        assert steps[1].winner_spikes[3].cell_id == 1103
        assert steps[1].winner_spikes[3].tick == 53
        assert steps[0].inhibition_onset_ticks == tuple(range(10, 21))
        assert steps[0].column_simulated_ticks == tuple(range(30, 41))
        assert steps[0].winner_context == tuple(range(2000, 2022))
        checkpoint = circuit.checkpoint()
        assert checkpoint == library.checkpoint_bytes

    circuit.close()
    with pytest.raises(RuntimeError, match="closed"):
        circuit.advance_block([7], mode=TemporalStepMode.READ_ONLY)

    with NativeTemporalCircuit.from_checkpoint(
        definition, checkpoint
    ) as restored:
        assert restored.checkpoint() == checkpoint

    assert library.destroyed == [0x1234, 0x1235]
    assert set(NativeTemporalRuntimeStatus) == {
        NativeTemporalRuntimeStatus(code) for code in range(11)
    }
    assert set(runtime_adapter.__all__) == {
        "NativeTemporalCircuit",
        "NativeTemporalCircuitConfig",
        "NativeTemporalCircuitDefinition",
        "NativeTemporalRuntimeError",
        "NativeTemporalRuntimeStatus",
        "TemporalCircuitStep",
        "TemporalContextPolicy",
        "TemporalPlasticityPolicy",
        "TemporalSpike",
        "TemporalStepMode",
    }
    for method in (
        NativeTemporalCircuit.fresh,
        NativeTemporalCircuit.from_checkpoint,
        NativeTemporalCircuit.advance_block,
        NativeTemporalCircuit.checkpoint,
    ):
        parameters = set(inspect.signature(method).parameters)
        assert not parameters.intersection(
            {
                "authority",
                "path",
                "current",
                "currents",
                "winner",
                "winners",
                "context",
                "contexts",
            }
        )

    create = library.humanbrain_temporal_circuit_runtime_create_v1
    imported = library.humanbrain_temporal_circuit_runtime_import_checkpoint_v1
    advance = library.humanbrain_temporal_circuit_runtime_advance_block_v1
    size = library.humanbrain_temporal_circuit_runtime_checkpoint_size_v1
    export = library.humanbrain_temporal_circuit_runtime_export_checkpoint_v1
    destroy = library.humanbrain_temporal_circuit_runtime_destroy
    functions = (create, imported, advance, size, export, destroy)
    assert [len(function.argtypes or ()) for function in functions] == [3, 3, 3, 3, 5, 1]
    assert all(
        function.restype is ctypes.c_uint32
        for function in (create, imported, advance, size, export)
    )
    assert destroy.restype is None


def test_advance_error_retains_only_the_committed_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    def fail_third_token(handle: object, request: object, report_out: object) -> int:
        del handle
        native = _pointed(request, runtime_adapter._NativeAdvanceRequest)
        for index in range(2):
            native.output[index].token_id = native.token_ids[index]
            native.output[index].event_count = 50 + index
        report = library._report(report_out)
        report.status = NativeTemporalRuntimeStatus.UNKNOWN_TOKEN
        report.completed_prefix = 2
        report.observed = native.token_ids[2]
        message = b"unknown token ID"
        report.message_len = len(message)
        report.message_utf8[: len(message)] = message
        return int(NativeTemporalRuntimeStatus.UNKNOWN_TOKEN)

    library.advance_override = fail_third_token
    with NativeTemporalCircuit.fresh(_tiny_definition()) as circuit:
        with pytest.raises(NativeTemporalRuntimeError) as stopped:
            circuit.advance_block(
                [7, 7, 99],
                mode=TemporalStepMode.LEARN,
                reset_before=True,
            )

    error = stopped.value
    assert error.status is NativeTemporalRuntimeStatus.UNKNOWN_TOKEN
    assert error.message == "unknown token ID"
    assert error.completed_prefix == 2
    assert tuple(step.token_id for step in error.completed_steps) == (7, 7)
    assert tuple(step.event_count for step in error.completed_steps) == (50, 51)
    assert error.observed == 99


@pytest.mark.parametrize(
    "defect", ("status-mismatch", "message-tail", "message-utf8", "step-flag")
)
def test_fake_native_transport_defects_fail_as_internal_invariants(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    def malformed(handle: object, request: object, report_out: object) -> int:
        del handle
        native = _pointed(request, runtime_adapter._NativeAdvanceRequest)
        report = library._report(report_out)
        native.output[0].token_id = native.token_ids[0]
        native.output[0].predicted = 2 if defect == "step-flag" else 0
        report.completed_prefix = 1
        if defect == "status-mismatch":
            report.status = NativeTemporalRuntimeStatus.UNKNOWN_TOKEN
            message = b"unknown token ID"
            report.message_len = len(message)
            report.message_utf8[: len(message)] = message
            return int(NativeTemporalRuntimeStatus.OK)
        if defect == "message-tail":
            report.message_utf8[17] = 1
        if defect == "message-utf8":
            report.message_len = 1
            report.message_utf8[0] = 0xFF
        return int(NativeTemporalRuntimeStatus.OK)

    library.advance_override = malformed
    with NativeTemporalCircuit.fresh(_tiny_definition()) as circuit:
        with pytest.raises(NativeTemporalRuntimeError) as rejected:
            circuit.advance_block([7], mode=TemporalStepMode.READ_ONLY)

    assert rejected.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
    assert "native ABI invariant failed" in str(rejected.value)
    assert rejected.value.completed_prefix == 1
    if defect == "step-flag":
        assert rejected.value.completed_steps == ()
    else:
        assert tuple(step.token_id for step in rejected.value.completed_steps) == (7,)


def test_create_revalidates_authority_after_native_load_before_publishing_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    authorities = iter(
        (
            runtime_adapter._FreshAuthority(b"\x11" * 32, b"\x22" * 32),
            runtime_adapter._FreshAuthority(b"\x33" * 32, b"\x22" * 32),
        )
    )
    monkeypatch.setattr(runtime_adapter, "_fresh_authority", lambda: next(authorities))

    with pytest.raises(NativeTemporalRuntimeError) as drifted:
        NativeTemporalCircuit.fresh(_tiny_definition())

    assert drifted.value.status is NativeTemporalRuntimeStatus.INVALID_DEFINITION
    assert "changed while loading" in drifted.value.message
    assert library.next_handle == 0x1234
    assert library.destroyed == []


def test_create_stops_before_native_call_when_loaded_source_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    monkeypatch.setattr(
        runtime_adapter,
        "_current_native_source_digest",
        lambda: "ef" * 32,
    )

    with pytest.raises(NativeBuildError, match="current native build inputs"):
        NativeTemporalCircuit.fresh(_tiny_definition())

    assert library.create_calls == 0
    assert library.next_handle == 0x1234
    assert library.destroyed == []


def test_checkpoint_stops_before_native_export_when_authority_or_binary_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, identity = _install_fake_runtime(monkeypatch)
    circuit = NativeTemporalCircuit.fresh(_tiny_definition())
    try:
        assert library.size_calls == 0
        assert library.export_calls == 0

        monkeypatch.setattr(
            runtime_adapter,
            "_current_native_source_digest",
            lambda: "ef" * 32,
        )
        with pytest.raises(NativeBuildError, match="current native build inputs"):
            circuit.checkpoint()
        assert library.size_calls == 0
        assert library.export_calls == 0

        monkeypatch.setattr(
            runtime_adapter,
            "_current_native_source_digest",
            lambda: identity.source_digest,
        )
        monkeypatch.setattr(
            runtime_adapter,
            "_fresh_authority",
            lambda: runtime_adapter._FreshAuthority(b"\x33" * 32, b"\x22" * 32),
        )
        with pytest.raises(NativeTemporalRuntimeError) as authority_error:
            circuit.checkpoint()
        assert authority_error.value.status is NativeTemporalRuntimeStatus.INVALID_DEFINITION
        assert library.size_calls == 0
        assert library.export_calls == 0

        monkeypatch.setattr(
            runtime_adapter,
            "_fresh_authority",
            lambda: runtime_adapter._FreshAuthority(b"\x11" * 32, b"\x22" * 32),
        )
        monkeypatch.setattr(
            runtime_adapter,
            "_loaded_library_identity",
            lambda: NativeLibraryIdentity(
                identity.source_digest,
                identity.binary_size,
                "cd" * 32,
            ),
        )
        with pytest.raises(NativeTemporalRuntimeError) as binary_error:
            circuit.checkpoint()
        assert binary_error.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
        assert "library identity changed" in binary_error.value.message
        assert library.size_calls == 0
        assert library.export_calls == 0
    finally:
        circuit.close()


def test_close_waits_for_inflight_advance_before_destroying_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    circuit = NativeTemporalCircuit.fresh(_tiny_definition())
    entered_native = threading.Event()
    release_native = threading.Event()
    close_started = threading.Event()

    def blocked_advance(handle: object, request: object, report_out: object) -> int:
        del handle
        entered_native.set()
        if not release_native.wait(timeout=5.0):
            raise AssertionError("test did not release native advance")
        native = _pointed(request, runtime_adapter._NativeAdvanceRequest)
        native.output[0].token_id = native.token_ids[0]
        library._report(report_out).completed_prefix = 1
        return 0

    library.advance_override = blocked_advance

    def close_circuit() -> None:
        close_started.set()
        circuit.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        advance_future = executor.submit(
            circuit.advance_block,
            [7],
            mode=TemporalStepMode.READ_ONLY,
        )
        assert entered_native.wait(timeout=5.0)
        close_future = executor.submit(close_circuit)
        assert close_started.wait(timeout=5.0)
        try:
            assert not close_future.done()
            assert library.destroyed == []
        finally:
            release_native.set()
        assert advance_future.result(timeout=5.0)[0].token_id == 7
        close_future.result(timeout=5.0)

    assert library.destroyed == [0x1234]
    with pytest.raises(RuntimeError, match="closed"):
        circuit.checkpoint()


def test_actual_native_fresh_checkpoint_import_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = runtime_adapter._FreshAuthority(b"\x41" * 32, b"\x92" * 32)
    monkeypatch.setattr(runtime_adapter, "_fresh_authority", lambda: authority)
    definition = _tiny_definition()

    with NativeTemporalCircuit.fresh(definition) as fresh:
        assert len(fresh.checkpoint()) == 256
        steps = fresh.advance_block([7], mode=TemporalStepMode.READ_ONLY)
        assert len(steps) == 1
        assert steps[0].token_id == 7
        checkpoint = fresh.checkpoint()
    assert len(checkpoint) > 256

    with NativeTemporalCircuit.from_checkpoint(definition, checkpoint) as restored:
        assert restored.checkpoint() == checkpoint
