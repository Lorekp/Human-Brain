"""Capability tests for the authority-gated native temporal runtime."""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import mmap
import os
import platform
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import humanbrain.native_temporal_runtime as runtime_adapter
import pytest
from experiments.r3_pilot.calibration_v3_protocol import (
    ProtocolError,
    canonical_source_bytes,
)
from humanbrain.native import NativeBuildError, NativeLibraryIdentity
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    NativeTemporalRuntimeStatus,
    TemporalCheckpointCleanup,
    TemporalCheckpointErrorDomain,
    TemporalCheckpointErrorStage,
    TemporalCheckpointIdentity,
    TemporalCheckpointPublication,
    TemporalCheckpointPublicationError,
    TemporalCheckpointPublicationOutcome,
    TemporalContextPolicy,
    TemporalGeneration,
    TemporalGenerationPath,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)

GATE3_PROTOCOL_PATH = "benchmarks/procedural_runtime_gate3_slice_protocol_v1.json"
GATE3_SOURCE_FRAMING = (
    "humanbrain-procedural-runtime-source-v1: ordered UTF-8 path length/path "
    "plus uint64 byte length/raw bytes"
)
SYNTHETIC_SOURCE_MANIFEST = ({"path": "src/synthetic.rs", "size": 3, "sha256": "44" * 32},)


def _synthetic_authority_governing(
    calls: list[str],
    repository_root: Path,
    *,
    declared_bundle_sha256: str = "22" * 32,
) -> SimpleNamespace:
    gate3_path = repository_root.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    gate3_path.parent.mkdir(parents=True)
    gate3_bytes = json.dumps(
        {
            "source_binding": {
                "framing": GATE3_SOURCE_FRAMING,
                "files": list(SYNTHETIC_SOURCE_MANIFEST),
                "bundle_sha256": declared_bundle_sha256,
                "limits": {
                    "maximum_file_bytes": 1 << 20,
                    "maximum_aggregate_bytes": 32 << 20,
                    "actual_aggregate_bytes": 3,
                },
            }
        }
    ).encode("utf-8")
    gate3_path.write_bytes(gate3_bytes)
    binding = {
        "files": {
            GATE3_PROTOCOL_PATH: {
                "commit": "aa" * 20,
                "sha256": hashlib.sha256(canonical_source_bytes(gate3_bytes)).hexdigest(),
            }
        },
        "governing_gate3_path": GATE3_PROTOCOL_PATH,
        "runnable": True,
    }
    protocol = {"source_binding": binding}

    return SimpleNamespace(
        REPO_ROOT=repository_root,
        build_protocol=lambda: calls.append("build") or protocol,
        validate_protocol=lambda value, *, require_runnable: calls.append(
            f"protocol:{value is protocol}:{require_runnable}"
        ),
        validate_committed_source_blobs=lambda _root, value: calls.append(
            f"committed:{value is binding}"
        ),
        validate_source_bytes=lambda _root, value: calls.append(f"live:{value is binding}"),
        protocol_sha256=lambda: calls.append("protocol-digest") or "11" * 32,
        source_binding=binding,
    )


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


def _basic_checkpoint_v1(size: int = 256) -> bytes:
    checkpoint = bytearray(size)
    checkpoint[:8] = b"R3NTCP1\x00"
    struct.pack_into("<HH", checkpoint, 8, 1, 256)
    struct.pack_into("<QQ", checkpoint, 16, size, size - 256)
    checkpoint[32:64] = b"\x11" * 32
    checkpoint[64:96] = b"\x22" * 32
    checkpoint[96:128] = hashlib.sha256(b"fake-circuit").digest()
    checkpoint[128:160] = hashlib.sha256(checkpoint[256:]).digest()
    checkpoint[160:192] = hashlib.sha256(b"").digest()
    struct.pack_into("<IIIQQBBH", checkpoint, 192, 0, 0, 11, 17, 19, 0, 0, 0)
    checkpoint[224:256] = hashlib.sha256(
        b"humanbrain-r3-checkpoint-logical-state-v1\x00" + checkpoint[128:160] + checkpoint[192:224]
    ).digest()
    return bytes(checkpoint)


class _FakeTemporalLibrary:
    def __init__(self) -> None:
        self.checkpoint_bytes = _basic_checkpoint_v1()
        self.destroyed: list[int] = []
        self.next_handle = 0x1234
        self.definition_identity_calls = 0
        self.create_calls = 0
        self.import_calls = 0
        self.generation_calls = 0
        self.size_calls = 0
        self.export_calls = 0
        self.identity_calls = 0
        self.publish_calls = 0
        self.immutable_publish_calls = 0
        self.reconcile_calls = 0
        self.prepare_directory_calls = 0
        self.published_paths: list[tuple[bytes, bytes]] = []
        self.published_payloads: list[tuple[bytes, bytes, bytes]] = []
        self.published_payload_addresses: list[int] = []
        self.reconciled_payloads: list[tuple[bytes, bytes, bytes]] = []
        self.prepared_directories: list[tuple[bytes, bytes]] = []
        self.advance_override: object | None = None
        self.import_override: object | None = None
        self.identity_override: object | None = None
        self.publish_override: object | None = None
        self.immutable_publish_override: object | None = None
        self.reconcile_override: object | None = None
        self.prepare_directory_override: object | None = None
        self.humanbrain_temporal_circuit_definition_identity_v1 = _FakeFunction(
            self._definition_identity
        )
        self.humanbrain_temporal_circuit_runtime_create_v1 = _FakeFunction(self._create)
        self.humanbrain_temporal_circuit_runtime_import_checkpoint_v1 = _FakeFunction(self._import)
        self.humanbrain_temporal_circuit_runtime_advance_block_v2 = _FakeFunction(self._advance)
        self.humanbrain_temporal_circuit_runtime_generate_v1 = _FakeFunction(
            self._generate
        )
        self.humanbrain_temporal_circuit_runtime_checkpoint_size_v1 = _FakeFunction(self._size)
        self.humanbrain_temporal_circuit_runtime_checkpoint_identity_v1 = _FakeFunction(
            self._identity
        )
        self.humanbrain_temporal_circuit_runtime_export_checkpoint_v1 = _FakeFunction(self._export)
        self.humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1 = _FakeFunction(
            self._publish
        )
        self.humanbrain_temporal_publish_immutable_bytes_path_v1 = _FakeFunction(
            self._publish_immutable_bytes
        )
        self.humanbrain_temporal_reconcile_existing_immutable_bytes_path_v1 = _FakeFunction(
            self._reconcile_existing_immutable_bytes
        )
        self.humanbrain_temporal_prepare_durable_directory_path_v1 = _FakeFunction(
            self._prepare_durable_directory
        )
        self.humanbrain_temporal_circuit_runtime_destroy = _FakeFunction(self._destroy)

    @staticmethod
    def _report(pointer: object) -> runtime_adapter._NativeReport:
        return ctypes.cast(pointer, ctypes.POINTER(runtime_adapter._NativeReport)).contents

    def _publish_handle(self, handle_out: object, report_out: object) -> int:
        ctypes.cast(handle_out, ctypes.POINTER(ctypes.c_void_p))[0] = self.next_handle
        self.next_handle += 1
        self._report(report_out).status = 0
        return 0

    def _definition_identity(self, request: object, output: object, report_out: object) -> int:
        self.definition_identity_calls += 1
        native = _pointed(request, runtime_adapter._NativeDefinitionIdentityRequest)
        assert native.token_code_count == 1
        assert native.token_codes[0].token_id == 7
        assert tuple(native.token_codes[0].columns) == tuple(range(11))
        ctypes.memmove(output, self.checkpoint_bytes[96:128], 32)
        report = self._report(report_out)
        report.required = 32
        report.observed = 32
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
        if self.import_override is not None:
            return self.import_override(  # type: ignore[operator,no-any-return]
                request, handle_out, report_out
            )
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
        native = _pointed(request, runtime_adapter._NativeAdvanceRequestV2)
        report = self._report(report_out)
        for index in range(native.token_count):
            step = native.output[index]
            step.token_id = native.token_ids[index]
            step.punished_segments = index + 3
            step.event_count = index + 101
            step.candidate_segments = index + 13
            step.matching_segments = index + 11
            step.active_segments = index + 7
            step.postings_visited = index + 1013
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

    def _generate(self, handle: object, output: object, report_out: object) -> int:
        del handle, report_out
        self.generation_calls += 1
        native = _pointed(output, runtime_adapter._NativeGeneration)
        native.surface_token_id = 7
        native.surface_output_cell_id = 55
        native.generation_index = self.generation_calls
        native.temporal_event_count = 102
        native.activated_columns = 11
        native.candidate_tokens = 1
        native.control_token_id = (1 << 32) - 1
        native.control_output_cell_id = (1 << 32) - 1
        native.surface_winner_tick = 180
        native.surface_basket_tick = 181
        native.surface_inhibition_onset_tick = 182
        native.total_simulated_ticks = 181
        native.path = TemporalGenerationPath.TEMPORAL
        native.winner_input_columns = 11
        native.feedback_predicted_columns = 11
        native.feedback_predicted = 1
        native.copy_place_ordinal = (1 << 8) - 1
        native.physical_election_count = 1
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

    def _identity(
        self,
        handle: object,
        authority: object,
        output: object,
        report_out: object,
    ) -> int:
        self.identity_calls += 1
        if self.identity_override is not None:
            return self.identity_override(  # type: ignore[operator,no-any-return]
                handle, authority, output, report_out
            )
        del handle, authority
        ctypes.memmove(output, self.checkpoint_bytes, 256)
        report = self._report(report_out)
        report.required = 256
        report.observed = 256
        return 0

    def _publish(
        self,
        handle: object,
        authority: object,
        request: object,
        publication_out: object,
        report_out: object,
    ) -> int:
        self.publish_calls += 1
        native_request = _pointed(request, runtime_adapter._NativeCheckpointPathRequest)
        self.published_paths.append(
            (
                ctypes.string_at(native_request.parent_utf8, native_request.parent_len),
                ctypes.string_at(
                    native_request.final_basename_utf8,
                    native_request.final_basename_len,
                ),
            )
        )
        if self.publish_override is not None:
            return self.publish_override(  # type: ignore[operator,no-any-return]
                handle, authority, request, publication_out, report_out
            )
        del handle, authority
        publication = _pointed(publication_out, runtime_adapter._NativeCheckpointPublicationReport)
        publication.outcome = TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED
        publication.temp_cleanup = TemporalCheckpointCleanup.CLEAN
        publication.owned_temp_pid = 123
        publication.owned_temp_nonce = 456
        publication.checkpoint_bytes = len(self.checkpoint_bytes)
        publication.encoded_bytes = len(self.checkpoint_bytes)
        report = self._report(report_out)
        report.required = len(self.checkpoint_bytes)
        report.observed = len(self.checkpoint_bytes)
        return 0

    def _destroy(self, handle: object) -> None:
        value = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)
        self.destroyed.append(value)

    def _publish_immutable_bytes(
        self,
        request: object,
        publication_out: object,
        report_out: object,
    ) -> int:
        self.immutable_publish_calls += 1
        native_request = _pointed(request, runtime_adapter._NativeImmutableBytesPathRequest)
        payload = ctypes.string_at(native_request.payload, native_request.payload_len)
        self.published_payload_addresses.append(
            int(ctypes.cast(native_request.payload, ctypes.c_void_p).value or 0)
        )
        self.published_payloads.append(
            (
                ctypes.string_at(native_request.parent_utf8, native_request.parent_len),
                ctypes.string_at(
                    native_request.final_basename_utf8,
                    native_request.final_basename_len,
                ),
                payload,
            )
        )
        if self.immutable_publish_override is not None:
            return self.immutable_publish_override(  # type: ignore[operator,no-any-return]
                request, publication_out, report_out
            )
        publication = _pointed(publication_out, runtime_adapter._NativeCheckpointPublicationReport)
        publication.outcome = runtime_adapter.TemporalPublicationOutcome.DURABLE_PUBLISHED
        publication.temp_cleanup = runtime_adapter.TemporalPublicationCleanup.CLEAN
        publication.owned_temp_pid = 123
        publication.owned_temp_nonce = 456
        publication.checkpoint_bytes = len(payload)
        publication.encoded_bytes = len(payload)
        report = self._report(report_out)
        report.required = len(payload)
        report.observed = len(payload)
        return 0

    def _reconcile_existing_immutable_bytes(
        self,
        request: object,
        durability_out: object,
        report_out: object,
    ) -> int:
        self.reconcile_calls += 1
        native_request = _pointed(request, runtime_adapter._NativeImmutableBytesPathRequest)
        payload = ctypes.string_at(native_request.payload, native_request.payload_len)
        self.reconciled_payloads.append(
            (
                ctypes.string_at(native_request.parent_utf8, native_request.parent_len),
                ctypes.string_at(
                    native_request.final_basename_utf8,
                    native_request.final_basename_len,
                ),
                payload,
            )
        )
        if self.reconcile_override is not None:
            return self.reconcile_override(  # type: ignore[operator,no-any-return]
                request, durability_out, report_out
            )
        durability = _pointed(
            durability_out,
            runtime_adapter._NativeImmutableBytesReconciliationReport,
        )
        durability.outcome = runtime_adapter.TemporalDurabilityOutcome.DURABLE
        durability.required_bytes = len(payload)
        durability.verified_bytes = len(payload)
        report = self._report(report_out)
        report.required = len(payload)
        report.observed = len(payload)
        return 0

    def _prepare_durable_directory(
        self,
        request: object,
        durability_out: object,
        report_out: object,
    ) -> int:
        self.prepare_directory_calls += 1
        native_request = _pointed(request, runtime_adapter._NativeDurableDirectoryPathRequest)
        root = ctypes.string_at(native_request.root_utf8, native_request.root_len)
        relative = ctypes.string_at(
            native_request.relative_directory_utf8,
            native_request.relative_directory_len,
        )
        self.prepared_directories.append((root, relative))
        if self.prepare_directory_override is not None:
            return self.prepare_directory_override(  # type: ignore[operator,no-any-return]
                request, durability_out, report_out
            )
        component_count = relative.count(b"/") + 1
        durability = _pointed(
            durability_out, runtime_adapter._NativeDurableDirectoryPreparationReport
        )
        durability.outcome = runtime_adapter.TemporalDurabilityOutcome.DURABLE
        durability.required_component_count = component_count
        durability.confirmed_component_count = component_count
        durability.created_component_count = component_count
        durability.failed_component_index = (1 << 32) - 1
        report = self._report(report_out)
        report.required = component_count
        report.observed = component_count
        return 0


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


def _set_publication_error(
    error: runtime_adapter._NativeCheckpointPublicationError,
    stage: TemporalCheckpointErrorStage,
    domain: TemporalCheckpointErrorDomain,
    code: int,
) -> None:
    error.stage = stage
    error.domain = domain
    error.code = code


def _completion_override(
    library: _FakeTemporalLibrary,
    outcome: TemporalCheckpointPublicationOutcome,
    *,
    cleanup: TemporalCheckpointCleanup,
    primary: tuple[TemporalCheckpointErrorStage, TemporalCheckpointErrorDomain, int] | None = None,
    cleanup_errors: tuple[
        tuple[TemporalCheckpointErrorStage, TemporalCheckpointErrorDomain, int],
        ...,
    ] = (),
) -> object:
    def complete(
        handle: object,
        authority: object,
        request: object,
        publication_out: object,
        report_out: object,
    ) -> int:
        del handle, authority, request
        publication = _pointed(publication_out, runtime_adapter._NativeCheckpointPublicationReport)
        publication.outcome = outcome
        publication.temp_cleanup = cleanup
        publication.cleanup_error_count = len(cleanup_errors)
        if primary is not None:
            _set_publication_error(publication.primary_error, *primary)
        for index, error in enumerate(cleanup_errors):
            _set_publication_error(publication.cleanup_errors[index], *error)
        if cleanup is not TemporalCheckpointCleanup.NOT_APPLICABLE:
            publication.owned_temp_pid = 321
            publication.owned_temp_nonce = 654
        publication.checkpoint_bytes = len(library.checkpoint_bytes)
        publication.encoded_bytes = (
            0
            if cleanup is TemporalCheckpointCleanup.NOT_APPLICABLE
            else len(library.checkpoint_bytes)
        )
        report = library._report(report_out)
        report.required = publication.checkpoint_bytes
        report.observed = publication.encoded_bytes
        return 0

    return complete


def _immutable_completion_override(
    outcome: runtime_adapter.TemporalPublicationOutcome,
    *,
    cleanup: runtime_adapter.TemporalPublicationCleanup,
    primary: tuple[
        runtime_adapter.TemporalPublicationErrorStage,
        runtime_adapter.TemporalPublicationErrorDomain,
        int,
    ]
    | None = None,
    cleanup_errors: tuple[
        tuple[
            runtime_adapter.TemporalPublicationErrorStage,
            runtime_adapter.TemporalPublicationErrorDomain,
            int,
        ],
        ...,
    ] = (),
) -> object:
    def complete(
        request: object,
        publication_out: object,
        report_out: object,
    ) -> int:
        native_request = _pointed(request, runtime_adapter._NativeImmutableBytesPathRequest)
        publication = _pointed(publication_out, runtime_adapter._NativeCheckpointPublicationReport)
        publication.outcome = outcome
        publication.temp_cleanup = cleanup
        publication.cleanup_error_count = len(cleanup_errors)
        if primary is not None:
            publication.primary_error.stage = primary[0]
            publication.primary_error.domain = primary[1]
            publication.primary_error.code = primary[2]
        for index, error in enumerate(cleanup_errors):
            publication.cleanup_errors[index].stage = error[0]
            publication.cleanup_errors[index].domain = error[1]
            publication.cleanup_errors[index].code = error[2]
        if cleanup is not runtime_adapter.TemporalPublicationCleanup.NOT_APPLICABLE:
            publication.owned_temp_pid = 321
            publication.owned_temp_nonce = 654
        publication.checkpoint_bytes = native_request.payload_len
        publication.encoded_bytes = (
            0
            if cleanup is runtime_adapter.TemporalPublicationCleanup.NOT_APPLICABLE
            else native_request.payload_len
        )
        report = _pointed(report_out, runtime_adapter._NativeReport)
        report.required = publication.checkpoint_bytes
        report.observed = publication.encoded_bytes
        return 0

    return complete


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
    owned = NativeTemporalCircuitDefinition(definition.config, mutable_codes, mutable_thresholds)
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
        runtime_adapter._NativeCheckpointPathRequest: (40, 8, (0, 8, 16, 24, 32)),
        runtime_adapter._NativeImmutableBytesPathRequest: (
            56,
            8,
            (0, 8, 16, 24, 32, 40, 48),
        ),
        runtime_adapter._NativeDurableDirectoryPathRequest: (
            40,
            8,
            (0, 8, 16, 24, 32),
        ),
        runtime_adapter._NativeCheckpointPublicationError: (8, 4, (0, 2, 4)),
        runtime_adapter._NativeCheckpointPublicationReport: (
            80,
            8,
            (0, 2, 3, 4, 12, 44, 48, 56, 64, 72),
        ),
        runtime_adapter._NativeImmutableBytesReconciliationReport: (
            64,
            8,
            (0, 2, 3, 4, 12, 28, 32, 40, 48),
        ),
        runtime_adapter._NativeDurableDirectoryPreparationReport: (
            64,
            4,
            (0, 2, 3, 4, 12, 28, 32, 36, 40, 44, 48),
        ),
        runtime_adapter._PyBuffer: (
            80,
            8,
            (0, 8, 16, 24, 32, 36, 40, 48, 56, 64, 72),
        ),
        runtime_adapter._NativeSpike: (8, 4, (0, 4, 6)),
        runtime_adapter._NativeStepV2: (
            800,
            8,
            (
                0,
                4,
                8,
                16,
                24,
                32,
                40,
                48,
                49,
                50,
                51,
                52,
                74,
                96,
                184,
                272,
                360,
                448,
                536,
                624,
                712,
            ),
        ),
        runtime_adapter._NativeGeneration: (
            64,
            8,
            (0, 4, 8, 16, 24, 28, 32, 36, 40, 42, 44, 46, 48, 50, 52, 54, 55, 56, 57, 58, 59, 60),
        ),
        runtime_adapter._NativeAdvanceRequestV2: (40, 8, (0, 8, 16, 24, 32, 33, 34)),
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

    step_bytes = bytearray(800)
    struct.pack_into(
        "<IIQQQQQBBBB",
        step_bytes,
        0,
        0x0A1B_2C3D,
        0x1E2F_4051,
        0x6273_8495_A6B7_C8D9,
        0x0102_0304_0506_0708,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
        1,
        7,
        5,
        0,
    )
    struct.pack_into("<11H", step_bytes, 52, *(101 + 2 * index for index in range(11)))
    struct.pack_into("<11H", step_bytes, 74, *(102 + 2 * index for index in range(11)))
    for index in range(55):
        struct.pack_into(
            "<IHH",
            step_bytes,
            96 + 8 * index,
            0x1020_3040 + index * 0x0001_0101,
            201 + index,
            0,
        )
    for index in range(66):
        struct.pack_into(
            "<I",
            step_bytes,
            536 + 4 * index,
            0x5060_7080 + index * 0x0001_0003,
        )
    assert step_bytes[74:96] == bytes.fromhex("660068006a006c006e00700072007400760078007a00")
    assert hashlib.sha256(step_bytes).hexdigest() == (
        "375b3a3e8996a37633df078068fbf8fb0e5d7863cb8a2276df1e552cbab33f92"
    )

    step = runtime_adapter._NativeStepV2.from_buffer_copy(step_bytes)
    assert bytes(step) == step_bytes

    publication = runtime_adapter._NativeCheckpointPublicationReport()
    publication.outcome = 4
    publication.temp_cleanup = 2
    publication.cleanup_error_count = 4
    publication.primary_error.stage = 9
    publication.primary_error.domain = 2
    publication.primary_error.code = 0x1122_3344
    publication.cleanup_errors[0].stage = 10
    publication.cleanup_errors[0].domain = 3
    publication.cleanup_errors[0].code = 0x0102_0304
    publication.cleanup_errors[1].stage = 11
    publication.cleanup_errors[1].domain = 1
    publication.cleanup_errors[1].code = 0x5566_7788
    publication.cleanup_errors[2].stage = 7
    publication.cleanup_errors[2].domain = 2
    publication.cleanup_errors[2].code = 0x99AA_BBCC
    publication.cleanup_errors[3].stage = 8
    publication.cleanup_errors[3].domain = 4
    publication.cleanup_errors[3].code = 0xDDEE_FF00
    publication.owned_temp_pid = 0xA1B2_C3D4
    publication.owned_temp_nonce = 0x0102_0304_0506_0708
    publication.checkpoint_bytes = 0x1112_1314_1516_1718
    publication.encoded_bytes = 0x2122_2324_2526_2728
    assert bytes(publication) == bytes.fromhex(
        "0400020409000200443322110a000300040302010b00010088776655"
        "07000200ccbbaa990800040000ffeeddd4c3b2a10807060504030201"
        "181716151413121128272625242322210000000000000000"
    )


def test_frozen_abi_rejects_little_endian_arm64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")

    with pytest.raises(RuntimeError, match="little-endian x64"):
        runtime_adapter._assert_frozen_abi()


def test_fresh_authority_validates_protocol_commit_and_live_bytes_before_digests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_authority_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    original_open = Path.open

    def observed_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == gate3_path:
            calls.append("gate3-read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    monkeypatch.setattr(runtime_adapter, "_governing_protocol", lambda: governing)
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_manifest",
        lambda root: calls.append("source-manifest") or SYNTHETIC_SOURCE_MANIFEST,
        raising=False,
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
        "gate3-read",
        "source-manifest",
        "source-digest",
    ]
    assert authority.protocol_identity_sha256 == b"\x11" * 32
    assert authority.source_bundle_identity_sha256 == b"\x22" * 32


def test_fresh_authority_rejects_live_bundle_not_approved_by_nested_gate3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_authority_governing(
        calls,
        tmp_path,
        declared_bundle_sha256="33" * 32,
    )
    monkeypatch.setattr(runtime_adapter, "_governing_protocol", lambda: governing)
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_manifest",
        lambda _root: SYNTHETIC_SOURCE_MANIFEST,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_bundle_sha256",
        lambda _root: "22" * 32,
    )

    with pytest.raises(RuntimeError, match=r"Gate3.*live source-bundle"):
        runtime_adapter._fresh_authority()


def test_fresh_authority_rejects_live_manifest_before_computing_bundle_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_authority_governing(calls, tmp_path)
    mismatched_manifest = ({"path": "src/synthetic.rs", "size": 3, "sha256": "55" * 32},)
    monkeypatch.setattr(runtime_adapter, "_governing_protocol", lambda: governing)
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_manifest",
        lambda _root: mismatched_manifest,
    )
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_bundle_sha256",
        lambda _root: pytest.fail("bundle digest must not run after a manifest mismatch"),
    )

    with pytest.raises(RuntimeError, match="Gate3 source manifest"):
        runtime_adapter._fresh_authority()


def test_fresh_authority_surfaces_duplicate_gate3_json_before_live_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_authority_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    manifest_json = json.dumps(list(SYNTHETIC_SOURCE_MANIFEST), separators=(",", ":"))
    gate3_path.write_text(
        "{"
        '"source_binding":{'
        f'"framing":"{GATE3_SOURCE_FRAMING}",'
        f'"files":{manifest_json},'
        f'"bundle_sha256":"{"22" * 32}",'
        f'"bundle_sha256":"{"22" * 32}",'
        '"limits":{'
        '"maximum_file_bytes":1048576,'
        '"maximum_aggregate_bytes":33554432,'
        '"actual_aggregate_bytes":3'
        "}}}",
        encoding="utf-8",
    )
    governing.source_binding["files"][GATE3_PROTOCOL_PATH]["sha256"] = hashlib.sha256(
        canonical_source_bytes(gate3_path.read_bytes())
    ).hexdigest()
    monkeypatch.setattr(runtime_adapter, "_governing_protocol", lambda: governing)
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_manifest",
        lambda _root: pytest.fail("live source manifest must not run after duplicate Gate3 JSON"),
    )
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_bundle_sha256",
        lambda _root: pytest.fail("live source digest must not run after duplicate Gate3 JSON"),
    )

    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        runtime_adapter._fresh_authority()


def test_fresh_authority_rejects_oversized_gate3_before_live_source_computation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_authority_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    gate3_path.write_bytes(b" " * ((1 << 20) + 1))
    monkeypatch.setattr(runtime_adapter, "_governing_protocol", lambda: governing)
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_manifest",
        lambda _root: pytest.fail(
            "live source manifest must not run after an oversized Gate3 file"
        ),
    )
    monkeypatch.setattr(
        runtime_adapter,
        "_procedural_runtime_source_bundle_sha256",
        lambda _root: pytest.fail("live source digest must not run after an oversized Gate3 file"),
    )

    with pytest.raises(RuntimeError, match="Gate3 file size is outside"):
        runtime_adapter._fresh_authority()


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


def test_checkpoint_identity_symbol_is_mandatory_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    del library.humanbrain_temporal_circuit_runtime_checkpoint_identity_v1

    with pytest.raises(NativeBuildError, match="ABI symbols are unavailable"):
        NativeTemporalCircuit.fresh(_tiny_definition())

    assert library.create_calls == 0
    assert library.size_calls == 0
    assert library.export_calls == 0


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
        generation = circuit.generate()
        assert generation == TemporalGeneration(
            surface_token_id=7,
            surface_output_cell_id=55,
            generation_index=1,
            temporal_event_count=102,
            activated_columns=11,
            candidate_tokens=1,
            control_token_id=None,
            control_output_cell_id=None,
            winner_input_columns=11,
            surface_winner_tick=180,
            surface_basket_tick=181,
            surface_inhibition_onset_tick=182,
            total_simulated_ticks=181,
            control_winner_tick=None,
            control_basket_tick=None,
            control_simulated_end_tick=None,
            path=TemporalGenerationPath.TEMPORAL,
            feedback_predicted_columns=11,
            feedback_predicted=True,
            copy_place_ordinal=None,
            physical_election_count=1,
        )
        assert library.generation_calls == 1
        checkpoint = circuit.checkpoint()
        assert checkpoint == library.checkpoint_bytes

    circuit.close()
    with pytest.raises(RuntimeError, match="closed"):
        circuit.advance_block([7], mode=TemporalStepMode.READ_ONLY)

    with NativeTemporalCircuit.from_checkpoint(definition, checkpoint) as restored:
        assert restored.checkpoint() == checkpoint

    assert library.destroyed == [0x1234, 0x1235]
    assert set(NativeTemporalRuntimeStatus) == {
        NativeTemporalRuntimeStatus(code) for code in range(11)
    }
    assert {member.name: member.value for member in TemporalCheckpointPublicationOutcome} == {
        "DURABLE_PUBLISHED": 1,
        "COLLISION": 2,
        "NOT_PUBLISHED": 3,
        "PUBLISHED_BUT_NOT_CONFIRMED_DURABLE": 4,
    }
    assert {member.name: member.value for member in TemporalCheckpointCleanup} == {
        "NOT_APPLICABLE": 0,
        "CLEAN": 1,
        "ORPHANED": 2,
    }
    assert {member.value for member in TemporalCheckpointErrorDomain} == set(range(6))
    assert {member.value for member in TemporalCheckpointErrorStage} == set(range(12))
    assert {member.name: member.value for member in runtime_adapter.TemporalPublicationOutcome} == {
        "DURABLE_PUBLISHED": 1,
        "COLLISION": 2,
        "NOT_PUBLISHED": 3,
        "PUBLISHED_BUT_NOT_CONFIRMED_DURABLE": 4,
    }
    assert {member.name: member.value for member in runtime_adapter.TemporalPublicationCleanup} == {
        "NOT_APPLICABLE": 0,
        "CLEAN": 1,
        "ORPHANED": 2,
    }
    assert {
        member.name: member.value for member in runtime_adapter.TemporalPublicationErrorDomain
    } == {
        "NONE": 0,
        "POSIX": 1,
        "WIN32": 2,
        "POLICY": 3,
        "CHECKPOINT_ENCODER_RESERVED": 4,
        "IO": 5,
    }
    assert {
        member.name: member.value for member in runtime_adapter.TemporalPublicationErrorStage
    } == {
        "NONE": 0,
        "PARENT_OPEN": 1,
        "PARENT_IDENTITY": 2,
        "FILESYSTEM_POLICY": 3,
        "TEMP_CREATE": 4,
        "WRITE": 5,
        "FILE_SYNC": 6,
        "FILE_CLOSE": 7,
        "TEMP_IDENTITY": 8,
        "EXCLUSIVE_PUBLISH": 9,
        "TEMP_CLEANUP": 10,
        "PARENT_SYNC": 11,
    }
    assert set(runtime_adapter.__all__) == {
        "NativeTemporalCircuit",
        "NativeTemporalCircuitConfig",
        "NativeTemporalCircuitDefinition",
        "NativeTemporalRuntimeError",
        "NativeTemporalRuntimeStatus",
        "TemporalCheckpointCleanup",
        "TemporalCheckpointErrorDomain",
        "TemporalCheckpointErrorStage",
        "TemporalCheckpointIdentity",
        "TemporalCheckpointPublication",
        "TemporalCheckpointPublicationError",
        "TemporalCheckpointPublicationOutcome",
        "TemporalImmutableBytesPublication",
        "TemporalImmutableBytesReconciliation",
        "TemporalDurableDirectoryPreparation",
        "TemporalDurabilityError",
        "TemporalDurabilityErrorDomain",
        "TemporalDurabilityErrorStage",
        "TemporalDurabilityOutcome",
        "TemporalPublicationCleanup",
        "TemporalPublicationError",
        "TemporalPublicationErrorDomain",
        "TemporalPublicationErrorStage",
        "TemporalPublicationOutcome",
        "TemporalCircuitStep",
        "TemporalContextPolicy",
        "TemporalGeneration",
        "TemporalGenerationPath",
        "TemporalPlasticityPolicy",
        "TemporalSpike",
        "TemporalStepMode",
        "IDENTIFIER_START_TOKEN_ID",
        "IDENTIFIER_END_TOKEN_ID",
        "COPY_ID_TOKEN_ID",
        "COPY_END_TOKEN_ID",
        "publish_immutable_bytes",
        "prepare_durable_directory",
        "reconcile_existing_immutable_bytes",
    }
    for method in (
        NativeTemporalCircuitDefinition.circuit_identity_sha256,
        NativeTemporalCircuit.fresh,
        NativeTemporalCircuit.from_checkpoint,
        NativeTemporalCircuit.from_checkpoint_path,
        NativeTemporalCircuit.advance_block,
        NativeTemporalCircuit.generate,
        NativeTemporalCircuit.publish_checkpoint,
        NativeTemporalCircuit.checkpoint,
        NativeTemporalCircuit.checkpoint_identity,
    ):
        parameters = set(inspect.signature(method).parameters)
        assert not parameters.intersection(
            {
                "authority",
                "current",
                "currents",
                "winner",
                "winners",
                "context",
                "contexts",
            }
        )
    for method in (
        NativeTemporalCircuitDefinition.circuit_identity_sha256,
        NativeTemporalCircuit.fresh,
        NativeTemporalCircuit.from_checkpoint,
        NativeTemporalCircuit.advance_block,
        NativeTemporalCircuit.generate,
        NativeTemporalCircuit.checkpoint,
        NativeTemporalCircuit.checkpoint_identity,
    ):
        assert "path" not in inspect.signature(method).parameters
    assert set(inspect.signature(NativeTemporalCircuit.from_checkpoint_path).parameters) == {
        "definition",
        "path",
    }
    assert set(inspect.signature(NativeTemporalCircuit.publish_checkpoint).parameters) == {
        "self",
        "destination",
    }
    assert set(inspect.signature(NativeTemporalCircuit.checkpoint_identity).parameters) == {"self"}
    assert set(inspect.signature(NativeTemporalCircuit.generate).parameters) == {"self"}
    assert set(
        inspect.signature(NativeTemporalCircuitDefinition.circuit_identity_sha256).parameters
    ) == {"self"}
    assert set(inspect.signature(runtime_adapter.publish_immutable_bytes).parameters) == {
        "destination",
        "payload",
    }
    assert set(
        inspect.signature(runtime_adapter.reconcile_existing_immutable_bytes).parameters
    ) == {"destination", "expected_payload"}
    assert set(inspect.signature(runtime_adapter.prepare_durable_directory).parameters) == {
        "root",
        "relative_directory",
    }

    definition_identity = library.humanbrain_temporal_circuit_definition_identity_v1
    create = library.humanbrain_temporal_circuit_runtime_create_v1
    imported = library.humanbrain_temporal_circuit_runtime_import_checkpoint_v1
    advance = library.humanbrain_temporal_circuit_runtime_advance_block_v2
    generate = library.humanbrain_temporal_circuit_runtime_generate_v1
    size = library.humanbrain_temporal_circuit_runtime_checkpoint_size_v1
    checkpoint_identity = library.humanbrain_temporal_circuit_runtime_checkpoint_identity_v1
    export = library.humanbrain_temporal_circuit_runtime_export_checkpoint_v1
    publish = library.humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1
    publish_bytes = library.humanbrain_temporal_publish_immutable_bytes_path_v1
    reconcile_bytes = library.humanbrain_temporal_reconcile_existing_immutable_bytes_path_v1
    prepare_directory = library.humanbrain_temporal_prepare_durable_directory_path_v1
    destroy = library.humanbrain_temporal_circuit_runtime_destroy
    functions = (
        definition_identity,
        create,
        imported,
        advance,
        generate,
        size,
        checkpoint_identity,
        export,
        publish,
        publish_bytes,
        reconcile_bytes,
        prepare_directory,
        destroy,
    )
    assert [len(function.argtypes or ()) for function in functions] == [
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        5,
        5,
        3,
        3,
        3,
        1,
    ]
    assert all(
        function.restype is ctypes.c_uint32
        for function in (
            definition_identity,
            create,
            imported,
            advance,
            generate,
            size,
            checkpoint_identity,
            export,
            publish,
            publish_bytes,
            reconcile_bytes,
            prepare_directory,
        )
    )
    assert destroy.restype is None


def test_checkpoint_identity_decodes_the_complete_frozen_header_without_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    with NativeTemporalCircuit.fresh(_tiny_definition()) as circuit:
        identity = circuit.checkpoint_identity()

    header = library.checkpoint_bytes[:256]
    assert identity == runtime_adapter.TemporalCheckpointIdentity(
        format_version=1,
        header_bytes=256,
        checkpoint_bytes=256,
        body_bytes=0,
        protocol_identity_sha256="11" * 32,
        source_bundle_identity_sha256="22" * 32,
        circuit_identity_sha256=header[96:128].hex(),
        body_sha256=header[128:160].hex(),
        semantic_index_sha256=header[160:192].hex(),
        segment_count=0,
        cohort_count=0,
        next_reserve_cell=11,
        event_count=17,
        learning_step=19,
        prior_context_length=0,
        winner_context_length=0,
        poison_code=0,
        logical_state_sha256=header[224:256].hex(),
    )
    assert library.identity_calls == 1
    assert library.size_calls == 0
    assert library.export_calls == 0
    assert tuple(field.name for field in dataclasses.fields(identity)) == (
        "format_version",
        "header_bytes",
        "checkpoint_bytes",
        "body_bytes",
        "protocol_identity_sha256",
        "source_bundle_identity_sha256",
        "circuit_identity_sha256",
        "body_sha256",
        "semantic_index_sha256",
        "segment_count",
        "cohort_count",
        "next_reserve_cell",
        "event_count",
        "learning_step",
        "prior_context_length",
        "winner_context_length",
        "poison_code",
        "logical_state_sha256",
    )
    assert identity == TemporalCheckpointIdentity(**dataclasses.asdict(identity))
    assert identity != dataclasses.asdict(identity)
    assert not hasattr(identity, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.event_count = 18  # type: ignore[misc]


@pytest.mark.parametrize(
    "defect",
    (
        "magic",
        "version",
        "header-bytes",
        "reserved",
        "framing",
        "protocol-authority",
        "source-authority",
        "prior-context-length",
        "winner-context-length",
        "poison-code",
        "scalar-body-length",
        "logical-state-digest",
        "report-required",
        "report-observed",
        "report-location",
    ),
)
def test_checkpoint_identity_rejects_malformed_headers_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    header = bytearray(library.checkpoint_bytes[:256])

    def refresh_logical_state() -> None:
        header[224:256] = hashlib.sha256(
            b"humanbrain-r3-checkpoint-logical-state-v1\x00" + header[128:160] + header[192:224]
        ).digest()

    if defect == "magic":
        header[0] ^= 1
    elif defect == "version":
        struct.pack_into("<H", header, 8, 2)
    elif defect == "header-bytes":
        struct.pack_into("<H", header, 10, 255)
    elif defect == "reserved":
        header[12] = 1
    elif defect == "framing":
        struct.pack_into("<Q", header, 16, 257)
    elif defect == "protocol-authority":
        header[32] ^= 1
    elif defect == "source-authority":
        header[64] ^= 1
    elif defect == "prior-context-length":
        header[220] = 23
        refresh_logical_state()
    elif defect == "winner-context-length":
        header[221] = 23
        refresh_logical_state()
    elif defect == "poison-code":
        struct.pack_into("<H", header, 222, 16)
        refresh_logical_state()
    elif defect == "scalar-body-length":
        struct.pack_into("<I", header, 192, 1)
        refresh_logical_state()
    elif defect == "logical-state-digest":
        header[224] ^= 1

    def malformed_identity(
        handle: object,
        authority: object,
        output: object,
        report_out: object,
    ) -> int:
        del handle, authority
        ctypes.memmove(output, bytes(header), len(header))
        report = library._report(report_out)
        report.required = 255 if defect == "report-required" else 256
        report.observed = 255 if defect == "report-observed" else 256
        report.location = 1 if defect == "report-location" else 0
        return 0

    library.identity_override = malformed_identity
    with NativeTemporalCircuit.fresh(_tiny_definition()) as circuit:
        with pytest.raises(NativeTemporalRuntimeError) as rejected:
            circuit.checkpoint_identity()

    assert rejected.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
    assert "native ABI invariant failed" in rejected.value.message
    assert library.identity_calls == 1


def test_checkpoint_identity_propagates_failure_only_with_zero_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    def authority_failure(
        handle: object,
        authority: object,
        output: object,
        report_out: object,
    ) -> int:
        del handle, authority, output
        report = library._report(report_out)
        report.status = NativeTemporalRuntimeStatus.INVALID_DEFINITION
        message = b"fresh authority does not match handle creation authority"
        report.message_len = len(message)
        report.message_utf8[: len(message)] = message
        return int(NativeTemporalRuntimeStatus.INVALID_DEFINITION)

    library.identity_override = authority_failure
    circuit = NativeTemporalCircuit.fresh(_tiny_definition())
    try:
        with pytest.raises(NativeTemporalRuntimeError) as rejected:
            circuit.checkpoint_identity()
        assert rejected.value.status is NativeTemporalRuntimeStatus.INVALID_DEFINITION
        assert "handle creation authority" in rejected.value.message

        def nonzero_failure(
            handle: object,
            authority: object,
            output: object,
            report_out: object,
        ) -> int:
            status = authority_failure(handle, authority, output, report_out)
            ctypes.cast(
                output, ctypes.POINTER(runtime_adapter._NativeCheckpointIdentity)
            ).contents.header[0] = 1
            return status

        library.identity_override = nonzero_failure
        with pytest.raises(NativeTemporalRuntimeError) as malformed:
            circuit.checkpoint_identity()
        assert malformed.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
        assert "nonzero output" in malformed.value.message

        def diagnostic_failure(
            handle: object,
            authority: object,
            output: object,
            report_out: object,
        ) -> int:
            status = authority_failure(handle, authority, output, report_out)
            library._report(report_out).poison_code = 1
            return status

        library.identity_override = diagnostic_failure
        with pytest.raises(NativeTemporalRuntimeError) as diagnostic:
            circuit.checkpoint_identity()
        assert diagnostic.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
        assert "scalar diagnostics" in diagnostic.value.message
    finally:
        circuit.close()


@pytest.mark.parametrize(
    ("outcome", "cleanup", "primary", "resumable"),
    (
        (
            TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED,
            TemporalCheckpointCleanup.CLEAN,
            None,
            True,
        ),
        (
            TemporalCheckpointPublicationOutcome.COLLISION,
            TemporalCheckpointCleanup.CLEAN,
            None,
            False,
        ),
        (
            TemporalCheckpointPublicationOutcome.NOT_PUBLISHED,
            TemporalCheckpointCleanup.NOT_APPLICABLE,
            (
                TemporalCheckpointErrorStage.PARENT_OPEN,
                TemporalCheckpointErrorDomain.POLICY,
                1,
            ),
            False,
        ),
        (
            TemporalCheckpointPublicationOutcome.PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
            TemporalCheckpointCleanup.CLEAN,
            (
                TemporalCheckpointErrorStage.PARENT_SYNC,
                TemporalCheckpointErrorDomain.POSIX,
                5,
            ),
            False,
        ),
    ),
)
def test_publish_checkpoint_returns_every_completion_outcome_and_only_durable_resumes(
    monkeypatch: pytest.MonkeyPatch,
    outcome: TemporalCheckpointPublicationOutcome,
    cleanup: TemporalCheckpointCleanup,
    primary: tuple[TemporalCheckpointErrorStage, TemporalCheckpointErrorDomain, int] | None,
    resumable: bool,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    library.publish_override = _completion_override(
        library, outcome, cleanup=cleanup, primary=primary
    )

    with NativeTemporalCircuit.fresh(_tiny_definition()) as circuit:
        completion = circuit.publish_checkpoint(Path("folder") / ".." / "state.chk")

    assert completion.outcome is outcome
    assert completion.temp_cleanup is cleanup
    assert completion.semantically_resumable is resumable
    assert completion.primary_error == (
        None if primary is None else TemporalCheckpointPublicationError(*primary)
    )
    assert library.publish_calls == 1
    with pytest.raises(AttributeError):
        completion.outcome = TemporalCheckpointPublicationOutcome.COLLISION  # type: ignore[misc]


def test_publish_immutable_bytes_is_a_standalone_bit_exact_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    payload = b'{"kind":"resume-manifest","generation":7}\x00\xff'
    get_buffer, release_buffer = runtime_adapter._readonly_buffer_api("immutable-byte test")
    view = runtime_adapter._PyBuffer()
    assert int(get_buffer(payload, ctypes.byref(view), 0)) == 0
    expected_payload_address = int(view.buf)
    release_buffer(ctypes.byref(view))

    completion = runtime_adapter.publish_immutable_bytes(Path("folder") / "result.json", payload)

    assert completion.payload_bytes == len(payload)
    assert completion.written_bytes == len(payload)
    assert completion.durably_published
    assert library.published_payloads == [(b"folder", b"result.json", payload)]
    assert library.published_payload_addresses == [expected_payload_address]


def test_reconcile_existing_immutable_bytes_is_a_bit_exact_durability_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    payload = b'{"kind":"resume-manifest","generation":7}\x00\xff'

    receipt = runtime_adapter.reconcile_existing_immutable_bytes(
        Path("folder") / "result.json", payload
    )

    assert receipt.outcome is runtime_adapter.TemporalDurabilityOutcome.DURABLE
    assert receipt.durably_reconciled
    assert receipt.primary_error is None
    assert receipt.expected_bytes == len(payload)
    assert receipt.verified_bytes == len(payload)
    assert library.reconciled_payloads == [(b"folder", b"result.json", payload)]
    assert library.reconcile_calls == 1
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        receipt.outcome = runtime_adapter.TemporalDurabilityOutcome.NOT_DURABLE


def test_prepare_durable_directory_reports_the_complete_relative_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    root = Path.cwd()

    receipt = runtime_adapter.prepare_durable_directory(
        root, Path("artifacts") / "resume" / "generation-7"
    )

    assert receipt.outcome is runtime_adapter.TemporalDurabilityOutcome.DURABLE
    assert receipt.durably_prepared
    assert receipt.required_component_count == 3
    assert receipt.confirmed_component_count == 3
    assert receipt.created_component_count == 3
    assert receipt.failed_component_index is None
    assert library.prepared_directories == [
        (
            str(root).encode("utf-8"),
            b"artifacts/resume/generation-7",
        )
    ]
    assert library.prepare_directory_calls == 1


def test_publish_immutable_bytes_releases_buffer_when_native_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    real_get_buffer, real_release_buffer = runtime_adapter._readonly_buffer_api(
        "immutable-byte test"
    )
    events: list[str] = []

    def tracking_get_buffer(owner: object, view: object, flags: int) -> int:
        events.append("get")
        return int(real_get_buffer(owner, view, flags))

    def tracking_release_buffer(view: object) -> None:
        events.append("release")
        real_release_buffer(view)

    monkeypatch.setattr(
        runtime_adapter,
        "_readonly_buffer_api",
        lambda purpose="checkpoint path import": (
            tracking_get_buffer,
            tracking_release_buffer,
        ),
    )

    def fail_native(*args: object) -> int:
        del args
        events.append("native")
        raise RuntimeError("native sentinel")

    library.immutable_publish_override = fail_native
    with pytest.raises(RuntimeError, match="native sentinel"):
        runtime_adapter.publish_immutable_bytes(Path("result.json"), b"opaque")

    assert events == ["get", "native", "release"]


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        (bytearray(b"x"), TypeError),
        (memoryview(b"x"), TypeError),
        (b"", ValueError),
        (b"x" * ((1 << 20) + 1), ValueError),
    ),
    ids=("bytearray", "memoryview", "empty", "oversized"),
)
def test_publish_immutable_bytes_rejects_payload_before_authority_or_native_load(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    error: type[Exception],
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    def forbidden_authority() -> object:
        raise AssertionError("invalid inputs must precede authority validation")

    monkeypatch.setattr(runtime_adapter, "_fresh_authority", forbidden_authority)
    with pytest.raises(error):
        runtime_adapter.publish_immutable_bytes(Path("result.json"), payload)  # type: ignore[arg-type]

    assert library.immutable_publish_calls == 0


def test_publish_immutable_bytes_rejects_path_type_before_authority_or_native_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    def forbidden_authority() -> object:
        raise AssertionError("path type validation must precede authority validation")

    monkeypatch.setattr(runtime_adapter, "_fresh_authority", forbidden_authority)
    with pytest.raises(TypeError, match="pathlib.Path"):
        runtime_adapter.publish_immutable_bytes("result.json", b"x")  # type: ignore[arg-type]

    assert library.immutable_publish_calls == 0


def test_publish_immutable_bytes_revalidates_authority_and_binary_before_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, identity = _install_fake_runtime(monkeypatch)
    path_encoding_calls = 0

    def forbidden_path_encoding(destination: Path) -> tuple[bytes, bytes]:
        del destination
        nonlocal path_encoding_calls
        path_encoding_calls += 1
        raise AssertionError("authority and binary attestation must precede path encoding")

    monkeypatch.setattr(runtime_adapter, "_lexical_publication_path_parts", forbidden_path_encoding)
    authorities = iter(
        (
            runtime_adapter._FreshAuthority(b"\x11" * 32, b"\x22" * 32),
            runtime_adapter._FreshAuthority(b"\x33" * 32, b"\x22" * 32),
        )
    )
    monkeypatch.setattr(runtime_adapter, "_fresh_authority", lambda: next(authorities))

    with pytest.raises(NativeTemporalRuntimeError) as drifted:
        runtime_adapter.publish_immutable_bytes(Path("result.json"), b"{}")

    assert drifted.value.status is NativeTemporalRuntimeStatus.INVALID_DEFINITION
    assert "changed while preparing" in drifted.value.message
    assert library.immutable_publish_calls == 0

    stable = runtime_adapter._FreshAuthority(b"\x11" * 32, b"\x22" * 32)
    monkeypatch.setattr(runtime_adapter, "_fresh_authority", lambda: stable)
    monkeypatch.setattr(
        runtime_adapter,
        "_current_native_source_digest",
        lambda: "ef" * 32,
    )
    with pytest.raises(NativeBuildError, match="current native build inputs"):
        runtime_adapter.publish_immutable_bytes(Path("result.json"), b"{}")

    assert identity.source_digest != "ef" * 32
    assert library.immutable_publish_calls == 0
    assert path_encoding_calls == 0


@pytest.mark.parametrize(
    ("outcome", "cleanup", "primary", "durable"),
    (
        (
            runtime_adapter.TemporalPublicationOutcome.DURABLE_PUBLISHED,
            runtime_adapter.TemporalPublicationCleanup.CLEAN,
            None,
            True,
        ),
        (
            runtime_adapter.TemporalPublicationOutcome.COLLISION,
            runtime_adapter.TemporalPublicationCleanup.CLEAN,
            None,
            False,
        ),
        (
            runtime_adapter.TemporalPublicationOutcome.NOT_PUBLISHED,
            runtime_adapter.TemporalPublicationCleanup.NOT_APPLICABLE,
            (
                runtime_adapter.TemporalPublicationErrorStage.PARENT_OPEN,
                runtime_adapter.TemporalPublicationErrorDomain.POLICY,
                1,
            ),
            False,
        ),
        (
            runtime_adapter.TemporalPublicationOutcome.PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
            runtime_adapter.TemporalPublicationCleanup.CLEAN,
            (
                runtime_adapter.TemporalPublicationErrorStage.PARENT_SYNC,
                runtime_adapter.TemporalPublicationErrorDomain.POSIX,
                5,
            ),
            False,
        ),
    ),
)
def test_publish_immutable_bytes_returns_all_outcomes_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    outcome: runtime_adapter.TemporalPublicationOutcome,
    cleanup: runtime_adapter.TemporalPublicationCleanup,
    primary: tuple[
        runtime_adapter.TemporalPublicationErrorStage,
        runtime_adapter.TemporalPublicationErrorDomain,
        int,
    ]
    | None,
    durable: bool,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    library.immutable_publish_override = _immutable_completion_override(
        outcome, cleanup=cleanup, primary=primary
    )

    completion = runtime_adapter.publish_immutable_bytes(Path("result.json"), b'{"ok":true}')

    assert completion.outcome is outcome
    assert completion.temp_cleanup is cleanup
    assert completion.primary_error == (
        None if primary is None else runtime_adapter.TemporalPublicationError(*primary)
    )
    assert completion.durably_published is durable
    assert library.immutable_publish_calls == 1
    with pytest.raises(AttributeError):
        completion.outcome = runtime_adapter.TemporalPublicationOutcome.COLLISION  # type: ignore[misc]


def test_publish_immutable_bytes_preserves_cleanup_error_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    cleanup_errors = (
        (
            runtime_adapter.TemporalPublicationErrorStage.FILE_CLOSE,
            runtime_adapter.TemporalPublicationErrorDomain.WIN32,
            91,
        ),
        (
            runtime_adapter.TemporalPublicationErrorStage.TEMP_CLEANUP,
            runtime_adapter.TemporalPublicationErrorDomain.POLICY,
            92,
        ),
    )
    library.immutable_publish_override = _immutable_completion_override(
        runtime_adapter.TemporalPublicationOutcome.NOT_PUBLISHED,
        cleanup=runtime_adapter.TemporalPublicationCleanup.ORPHANED,
        primary=(
            runtime_adapter.TemporalPublicationErrorStage.WRITE,
            runtime_adapter.TemporalPublicationErrorDomain.IO,
            6,
        ),
        cleanup_errors=cleanup_errors,
    )

    completion = runtime_adapter.publish_immutable_bytes(Path("result.json"), b"x")

    assert completion.cleanup_errors == tuple(
        runtime_adapter.TemporalPublicationError(*error) for error in cleanup_errors
    )


@pytest.mark.parametrize(
    "defect",
    (
        "reserved",
        "outcome",
        "cleanup-tag",
        "cleanup-count",
        "active-cleanup-zero",
        "unused-cleanup-active",
        "malformed-primary",
        "payload-length",
        "input-length-mismatch",
        "written-overflow",
        "runtime-length-mismatch",
        "durable-primary",
        "clean-with-error",
        "orphan-without-error",
        "not-applicable-temp-stage",
        "temp-backed-pre-temp-stage",
        "unconfirmed-wrong-stage",
        "checkpoint-encoder-domain",
        "failure-changed-publication",
    ),
)
def test_immutable_bytes_publication_decoder_rejects_malformed_reports(
    defect: str,
) -> None:
    publication = runtime_adapter._NativeCheckpointPublicationReport()
    publication.outcome = runtime_adapter.TemporalPublicationOutcome.DURABLE_PUBLISHED
    publication.temp_cleanup = runtime_adapter.TemporalPublicationCleanup.CLEAN
    publication.owned_temp_pid = 123
    publication.owned_temp_nonce = 456
    publication.checkpoint_bytes = 1
    publication.encoded_bytes = 1
    report = runtime_adapter._NativeReport()
    report.required = 1
    report.observed = 1
    returned: int = NativeTemporalRuntimeStatus.OK
    expected_payload_bytes = 1

    if defect == "reserved":
        publication.reserved_zero[7] = 1
    elif defect == "outcome":
        publication.outcome = 0
    elif defect == "cleanup-tag":
        publication.temp_cleanup = 3
    elif defect == "cleanup-count":
        publication.cleanup_error_count = 5
    elif defect == "active-cleanup-zero":
        publication.cleanup_error_count = 1
    elif defect == "unused-cleanup-active":
        publication.cleanup_errors[
            3
        ].stage = runtime_adapter.TemporalPublicationErrorStage.TEMP_CLEANUP
        publication.cleanup_errors[3].domain = runtime_adapter.TemporalPublicationErrorDomain.POLICY
        publication.cleanup_errors[3].code = 1
    elif defect == "malformed-primary":
        publication.primary_error.domain = runtime_adapter.TemporalPublicationErrorDomain.POLICY
    elif defect == "payload-length":
        publication.checkpoint_bytes = 0
        publication.encoded_bytes = 0
        report.required = 0
        report.observed = 0
    elif defect == "input-length-mismatch":
        expected_payload_bytes = 2
    elif defect == "written-overflow":
        publication.encoded_bytes = 2
        report.observed = 2
    elif defect == "runtime-length-mismatch":
        report.required = 2
    elif defect == "durable-primary":
        publication.primary_error.stage = runtime_adapter.TemporalPublicationErrorStage.FILE_SYNC
        publication.primary_error.domain = runtime_adapter.TemporalPublicationErrorDomain.POSIX
        publication.primary_error.code = 5
    elif defect == "clean-with-error":
        publication.cleanup_error_count = 1
        publication.cleanup_errors[
            0
        ].stage = runtime_adapter.TemporalPublicationErrorStage.TEMP_CLEANUP
        publication.cleanup_errors[0].domain = runtime_adapter.TemporalPublicationErrorDomain.POLICY
        publication.cleanup_errors[0].code = 1
    elif defect == "orphan-without-error":
        publication.temp_cleanup = runtime_adapter.TemporalPublicationCleanup.ORPHANED
    elif defect == "not-applicable-temp-stage":
        publication.outcome = runtime_adapter.TemporalPublicationOutcome.NOT_PUBLISHED
        publication.temp_cleanup = runtime_adapter.TemporalPublicationCleanup.NOT_APPLICABLE
        publication.owned_temp_pid = 0
        publication.owned_temp_nonce = 0
        publication.encoded_bytes = 0
        report.observed = 0
        publication.primary_error.stage = runtime_adapter.TemporalPublicationErrorStage.WRITE
        publication.primary_error.domain = runtime_adapter.TemporalPublicationErrorDomain.IO
        publication.primary_error.code = 1
    elif defect == "temp-backed-pre-temp-stage":
        publication.outcome = runtime_adapter.TemporalPublicationOutcome.NOT_PUBLISHED
        publication.primary_error.stage = runtime_adapter.TemporalPublicationErrorStage.PARENT_OPEN
        publication.primary_error.domain = runtime_adapter.TemporalPublicationErrorDomain.POLICY
        publication.primary_error.code = 1
    elif defect == "unconfirmed-wrong-stage":
        publication.outcome = (
            runtime_adapter.TemporalPublicationOutcome.PUBLISHED_BUT_NOT_CONFIRMED_DURABLE
        )
        publication.primary_error.stage = runtime_adapter.TemporalPublicationErrorStage.WRITE
        publication.primary_error.domain = runtime_adapter.TemporalPublicationErrorDomain.IO
        publication.primary_error.code = 1
    elif defect == "checkpoint-encoder-domain":
        publication.outcome = runtime_adapter.TemporalPublicationOutcome.NOT_PUBLISHED
        publication.primary_error.stage = runtime_adapter.TemporalPublicationErrorStage.WRITE
        publication.primary_error.domain = 4
        publication.primary_error.code = 1
    else:
        returned = NativeTemporalRuntimeStatus.INVALID_ARGUMENT
        report.status = NativeTemporalRuntimeStatus.INVALID_ARGUMENT
        message = b"bad request"
        report.message_len = len(message)
        report.message_utf8[: len(message)] = message

    with pytest.raises(NativeTemporalRuntimeError) as rejected:
        runtime_adapter._decode_immutable_bytes_publication(
            returned,
            publication,
            report,
            expected_payload_bytes=expected_payload_bytes,
        )

    assert rejected.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT


def test_publish_checkpoint_preserves_cleanup_error_order_and_uses_path_lexically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    cleanup_errors = (
        (
            TemporalCheckpointErrorStage.FILE_CLOSE,
            TemporalCheckpointErrorDomain.WIN32,
            91,
        ),
        (
            TemporalCheckpointErrorStage.TEMP_CLEANUP,
            TemporalCheckpointErrorDomain.POLICY,
            92,
        ),
    )
    library.publish_override = _completion_override(
        library,
        TemporalCheckpointPublicationOutcome.NOT_PUBLISHED,
        cleanup=TemporalCheckpointCleanup.ORPHANED,
        primary=(
            TemporalCheckpointErrorStage.ENCODE,
            TemporalCheckpointErrorDomain.ENCODER,
            7,
        ),
        cleanup_errors=cleanup_errors,
    )
    filesystem_calls = 0

    def forbidden_filesystem(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal filesystem_calls
        filesystem_calls += 1
        raise AssertionError("path publisher wrapper must not touch the filesystem")

    with NativeTemporalCircuit.fresh(_tiny_definition()) as circuit:
        monkeypatch.setattr(runtime_adapter.os, "lstat", forbidden_filesystem)
        monkeypatch.setattr(runtime_adapter.os, "open", forbidden_filesystem)
        destination = Path("lexical-☃") / "nested" / ".." / "final-☃.chk"
        completion = circuit.publish_checkpoint(destination)

    assert filesystem_calls == 0
    assert library.published_paths == [
        (
            (
                b"lexical-\xe2\x98\x83\\nested\\.."
                if os.name == "nt"
                else b"lexical-\xe2\x98\x83/nested/.."
            ),
            b"final-\xe2\x98\x83.chk",
        )
    ]
    assert completion.cleanup_errors == tuple(
        TemporalCheckpointPublicationError(*error) for error in cleanup_errors
    )


@pytest.mark.parametrize(
    "defect",
    (
        "reserved",
        "outcome",
        "cleanup-tag",
        "cleanup-count",
        "active-cleanup-zero",
        "unused-cleanup-active",
        "malformed-primary",
        "durable-primary",
        "clean-with-error",
        "encoded-overflow",
        "runtime-length-mismatch",
        "not-applicable-temp-stage",
        "temp-backed-pre-temp-stage",
        "unconfirmed-wrong-stage",
        "failure-changed-publication",
    ),
)
def test_checkpoint_publication_decoder_rejects_every_malformed_field_cross_product(
    defect: str,
) -> None:
    publication = runtime_adapter._NativeCheckpointPublicationReport()
    publication.outcome = TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED
    publication.temp_cleanup = TemporalCheckpointCleanup.CLEAN
    publication.owned_temp_pid = 123
    publication.owned_temp_nonce = 456
    publication.checkpoint_bytes = 256
    publication.encoded_bytes = 256
    report = runtime_adapter._NativeReport()
    report.required = 256
    report.observed = 256
    returned: int = NativeTemporalRuntimeStatus.OK

    if defect == "reserved":
        publication.reserved_zero[7] = 1
    elif defect == "outcome":
        publication.outcome = 0
    elif defect == "cleanup-tag":
        publication.temp_cleanup = 3
    elif defect == "cleanup-count":
        publication.cleanup_error_count = 5
    elif defect == "active-cleanup-zero":
        publication.cleanup_error_count = 1
    elif defect == "unused-cleanup-active":
        _set_publication_error(
            publication.cleanup_errors[3],
            TemporalCheckpointErrorStage.TEMP_CLEANUP,
            TemporalCheckpointErrorDomain.POLICY,
            1,
        )
    elif defect == "malformed-primary":
        publication.primary_error.domain = TemporalCheckpointErrorDomain.POLICY
    elif defect == "durable-primary":
        _set_publication_error(
            publication.primary_error,
            TemporalCheckpointErrorStage.FILE_SYNC,
            TemporalCheckpointErrorDomain.POSIX,
            5,
        )
    elif defect == "clean-with-error":
        publication.cleanup_error_count = 1
        _set_publication_error(
            publication.cleanup_errors[0],
            TemporalCheckpointErrorStage.TEMP_CLEANUP,
            TemporalCheckpointErrorDomain.POLICY,
            1,
        )
    elif defect == "encoded-overflow":
        publication.encoded_bytes = 257
        report.observed = 257
    elif defect == "runtime-length-mismatch":
        report.required = 257
    elif defect == "not-applicable-temp-stage":
        publication.outcome = TemporalCheckpointPublicationOutcome.NOT_PUBLISHED
        publication.temp_cleanup = TemporalCheckpointCleanup.NOT_APPLICABLE
        publication.owned_temp_pid = 0
        publication.owned_temp_nonce = 0
        publication.encoded_bytes = 0
        report.observed = 0
        _set_publication_error(
            publication.primary_error,
            TemporalCheckpointErrorStage.ENCODE,
            TemporalCheckpointErrorDomain.ENCODER,
            1,
        )
    elif defect == "temp-backed-pre-temp-stage":
        publication.outcome = TemporalCheckpointPublicationOutcome.NOT_PUBLISHED
        _set_publication_error(
            publication.primary_error,
            TemporalCheckpointErrorStage.PARENT_OPEN,
            TemporalCheckpointErrorDomain.POLICY,
            1,
        )
    elif defect == "unconfirmed-wrong-stage":
        publication.outcome = (
            TemporalCheckpointPublicationOutcome.PUBLISHED_BUT_NOT_CONFIRMED_DURABLE
        )
        _set_publication_error(
            publication.primary_error,
            TemporalCheckpointErrorStage.ENCODE,
            TemporalCheckpointErrorDomain.ENCODER,
            1,
        )
    else:
        returned = NativeTemporalRuntimeStatus.INVALID_ARGUMENT
        report.status = NativeTemporalRuntimeStatus.INVALID_ARGUMENT
        message = b"bad request"
        report.message_len = len(message)
        report.message_utf8[: len(message)] = message

    with pytest.raises(NativeTemporalRuntimeError) as rejected:
        runtime_adapter._decode_publication(returned, publication, report)
    assert rejected.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT


def test_publish_checkpoint_revalidates_authority_and_binary_before_native_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, identity = _install_fake_runtime(monkeypatch)
    circuit = NativeTemporalCircuit.fresh(_tiny_definition())
    try:
        monkeypatch.setattr(
            runtime_adapter,
            "_fresh_authority",
            lambda: runtime_adapter._FreshAuthority(b"\x33" * 32, b"\x22" * 32),
        )
        with pytest.raises(NativeTemporalRuntimeError) as authority_error:
            circuit.publish_checkpoint(Path("unused.chk"))
        assert authority_error.value.status is NativeTemporalRuntimeStatus.INVALID_DEFINITION
        assert library.publish_calls == 0

        monkeypatch.setattr(
            runtime_adapter,
            "_fresh_authority",
            lambda: runtime_adapter._FreshAuthority(b"\x11" * 32, b"\x22" * 32),
        )
        monkeypatch.setattr(
            runtime_adapter,
            "_loaded_library_identity",
            lambda: NativeLibraryIdentity(identity.source_digest, identity.binary_size, "ef" * 32),
        )
        with pytest.raises(NativeTemporalRuntimeError) as binary_error:
            circuit.publish_checkpoint(Path("unused.chk"))
        assert binary_error.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
        assert library.publish_calls == 0
    finally:
        circuit.close()


@pytest.mark.parametrize("gate", ("authority", "binary"))
def test_checkpoint_path_import_completes_all_authority_checks_before_file_io(
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    if gate == "authority":
        authorities = iter(
            (
                runtime_adapter._FreshAuthority(b"\x11" * 32, b"\x22" * 32),
                runtime_adapter._FreshAuthority(b"\x33" * 32, b"\x22" * 32),
            )
        )
        monkeypatch.setattr(runtime_adapter, "_fresh_authority", lambda: next(authorities))
    else:
        monkeypatch.setattr(
            runtime_adapter,
            "_current_native_source_digest",
            lambda: "ef" * 32,
        )
    filesystem_calls = {"lstat": 0, "open": 0}

    def forbidden_lstat(path: object) -> object:
        del path
        filesystem_calls["lstat"] += 1
        raise AssertionError("authority gate must precede checkpoint lstat")

    def forbidden_open(path: object, flags: int) -> object:
        del path, flags
        filesystem_calls["open"] += 1
        raise AssertionError("authority gate must precede checkpoint open")

    monkeypatch.setattr(runtime_adapter.os, "lstat", forbidden_lstat)
    monkeypatch.setattr(runtime_adapter.os, "open", forbidden_open)

    expected_error = NativeTemporalRuntimeError if gate == "authority" else NativeBuildError
    with pytest.raises(expected_error):
        NativeTemporalCircuit.from_checkpoint_path(
            _tiny_definition(), Path("must-not-be-opened.chk")
        )

    assert filesystem_calls == {"lstat": 0, "open": 0}
    assert library.import_calls == 0
    assert library.destroyed == []


@pytest.mark.parametrize("unsupported", ("vm", "version", "symbols"))
def test_checkpoint_path_import_rejects_unsupported_buffer_abi_before_file_io(
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    if unsupported == "vm":
        monkeypatch.setattr(platform, "python_implementation", lambda: "PyPy")
    elif unsupported == "version":
        monkeypatch.setattr(runtime_adapter.sys, "version_info", (3, 10, 99))
    else:
        monkeypatch.setattr(runtime_adapter, "_python_buffer_api", lambda: object())
    filesystem_calls = {"lstat": 0, "open": 0}

    def forbidden_lstat(path: object) -> object:
        del path
        filesystem_calls["lstat"] += 1
        raise AssertionError("buffer ABI validation must precede checkpoint lstat")

    def forbidden_open(path: object, flags: int) -> object:
        del path, flags
        filesystem_calls["open"] += 1
        raise AssertionError("buffer ABI validation must precede checkpoint open")

    monkeypatch.setattr(runtime_adapter.os, "lstat", forbidden_lstat)
    monkeypatch.setattr(runtime_adapter.os, "open", forbidden_open)

    with pytest.raises(RuntimeError, match="checkpoint path import requires"):
        NativeTemporalCircuit.from_checkpoint_path(
            _tiny_definition(), Path("must-not-be-opened.chk")
        )
    assert filesystem_calls == {"lstat": 0, "open": 0}
    assert library.import_calls == 0
    assert library.destroyed == []


@pytest.mark.parametrize(
    "defect",
    (
        "acquisition-failed",
        "null-owner",
        "null-address",
        "length",
        "writable",
        "itemsize",
        "dimensions",
        "format",
        "shape",
        "strides",
        "suboffsets",
        "internal",
    ),
)
def test_checkpoint_path_import_rejects_malformed_stable_abi_buffers_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    defect: str,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    checkpoint_path = tmp_path / f"malformed-buffer-{defect}.chk"
    checkpoint_path.write_bytes(library.checkpoint_bytes)
    backing = ctypes.create_string_buffer(library.checkpoint_bytes)
    get_calls = 0
    release_calls = 0

    def malformed_get_buffer(owner: object, pointer: object, flags: int) -> int:
        nonlocal get_calls
        get_calls += 1
        assert flags == 0
        if defect == "acquisition-failed":
            return -1
        view = ctypes.cast(pointer, ctypes.POINTER(runtime_adapter._PyBuffer)).contents
        view.obj = id(owner)
        view.buf = ctypes.addressof(backing)
        view.len = len(library.checkpoint_bytes)
        view.itemsize = 1
        view.readonly = 1
        view.ndim = 1
        if defect == "null-owner":
            view.obj = None
        elif defect == "null-address":
            view.buf = None
        elif defect == "length":
            view.len += 1
        elif defect == "writable":
            view.readonly = 0
        elif defect == "itemsize":
            view.itemsize = 2
        elif defect == "dimensions":
            view.ndim = 2
        else:
            setattr(view, defect, 1)
        return 0

    def tracking_release_buffer(pointer: object) -> None:
        del pointer
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(
        runtime_adapter,
        "_readonly_buffer_api",
        lambda: (malformed_get_buffer, tracking_release_buffer),
    )

    with pytest.raises(NativeTemporalRuntimeError) as rejected:
        NativeTemporalCircuit.from_checkpoint_path(_tiny_definition(), checkpoint_path)
    assert rejected.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
    assert get_calls == 1
    assert release_calls == (0 if defect == "acquisition-failed" else 1)
    assert library.import_calls == 0
    assert library.destroyed == []
    checkpoint_path.unlink()


def test_checkpoint_path_import_holds_readonly_buffer_and_releases_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    checkpoint_path = tmp_path / "state.chk"
    checkpoint_path.write_bytes(library.checkpoint_bytes)
    events: list[str] = []
    real_mmap = mmap.mmap
    real_fdopen = os.fdopen
    real_get_buffer, real_release_buffer = runtime_adapter._readonly_buffer_api()

    class TrackingMmap(real_mmap):
        def __new__(cls, *args: object, **kwargs: object) -> TrackingMmap:
            assert kwargs["access"] == mmap.ACCESS_READ
            events.append("mmap-read")
            return super().__new__(cls, *args, **kwargs)

        def close(self) -> None:
            events.append("mmap-close")
            super().close()

    class TrackingFile:
        def __init__(self, descriptor: int, mode: str, buffering: int) -> None:
            self._file = real_fdopen(descriptor, mode, buffering)

        def fileno(self) -> int:
            return self._file.fileno()

        def read(self, size: int) -> bytes:
            return self._file.read(size)

        def close(self) -> None:
            events.append("file-close")
            self._file.close()

    def tracking_fdopen(descriptor: int, mode: str, *, buffering: int) -> TrackingFile:
        return TrackingFile(descriptor, mode, buffering)

    def tracking_get_buffer(owner: object, view: object, flags: int) -> int:
        assert flags == 0
        events.append("buffer-get")
        return real_get_buffer(owner, view, flags)

    def tracking_release_buffer(view: object) -> None:
        events.append("buffer-release")
        real_release_buffer(view)

    def import_while_buffer_is_live(request: object, handle_out: object, report_out: object) -> int:
        events.append("native-import")
        native = _pointed(request, runtime_adapter._NativeImportRequest)
        assert ctypes.string_at(native.checkpoint_bytes, native.checkpoint_len) == (
            library.checkpoint_bytes
        )
        return library._publish_handle(handle_out, report_out)

    monkeypatch.setattr(runtime_adapter.mmap, "mmap", TrackingMmap)
    monkeypatch.setattr(runtime_adapter.os, "fdopen", tracking_fdopen)
    monkeypatch.setattr(
        runtime_adapter,
        "_readonly_buffer_api",
        lambda: (tracking_get_buffer, tracking_release_buffer),
    )
    library.import_override = import_while_buffer_is_live

    with NativeTemporalCircuit.from_checkpoint_path(
        _tiny_definition(), checkpoint_path
    ) as restored:
        assert library.import_calls == 1
        assert restored.checkpoint() == library.checkpoint_bytes

    assert events == [
        "mmap-read",
        "buffer-get",
        "native-import",
        "buffer-release",
        "mmap-close",
        "file-close",
    ]
    checkpoint_path.unlink()
    assert not checkpoint_path.exists()
    assert library.destroyed == [0x1234]


@pytest.mark.parametrize(
    "defect", ("short", "magic", "version", "header-size", "reserved", "total")
)
def test_checkpoint_path_import_rejects_bad_basic_v1_headers_before_native_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    defect: str,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    checkpoint = bytearray(_basic_checkpoint_v1())
    if defect == "short":
        checkpoint = checkpoint[:255]
    elif defect == "magic":
        checkpoint[0] ^= 1
    elif defect == "version":
        struct.pack_into("<H", checkpoint, 8, 2)
    elif defect == "header-size":
        struct.pack_into("<H", checkpoint, 10, 255)
    elif defect == "reserved":
        checkpoint[15] = 1
    else:
        struct.pack_into("<Q", checkpoint, 16, len(checkpoint) + 1)
    checkpoint_path = tmp_path / f"bad-{defect}.chk"
    checkpoint_path.write_bytes(checkpoint)

    with pytest.raises(NativeTemporalRuntimeError) as rejected:
        NativeTemporalCircuit.from_checkpoint_path(_tiny_definition(), checkpoint_path)

    assert rejected.value.status is NativeTemporalRuntimeStatus.CHECKPOINT_REJECTED
    assert library.import_calls == 0
    assert library.destroyed == []
    checkpoint_path.unlink()


def test_checkpoint_path_import_rejects_final_symlink_and_nonregular_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    with pytest.raises(NativeTemporalRuntimeError) as directory_error:
        NativeTemporalCircuit.from_checkpoint_path(_tiny_definition(), tmp_path)
    assert directory_error.value.status is NativeTemporalRuntimeStatus.CHECKPOINT_REJECTED

    target = tmp_path / "target.chk"
    target.write_bytes(library.checkpoint_bytes)
    link = tmp_path / "link.chk"
    try:
        link.symlink_to(target)
    except OSError as exc:
        del exc
        symlink_metadata = os.stat_result(
            (0o120777, 17, 23, 1, 0, 0, len(library.checkpoint_bytes), 0, 0, 0)
        )
        monkeypatch.setattr(runtime_adapter.os, "lstat", lambda _path: symlink_metadata)
        reached_open = False

        def forbidden_open(path: object, flags: int) -> object:
            del path, flags
            nonlocal reached_open
            reached_open = True
            raise AssertionError("symlink rejection must precede descriptor open")

        monkeypatch.setattr(runtime_adapter.os, "open", forbidden_open)
    with pytest.raises(NativeTemporalRuntimeError) as symlink_error:
        NativeTemporalCircuit.from_checkpoint_path(_tiny_definition(), link)
    assert symlink_error.value.status is NativeTemporalRuntimeStatus.CHECKPOINT_REJECTED
    assert library.import_calls == 0
    if link.exists() or link.is_symlink():
        link.unlink()
    else:
        assert reached_open is False
    target.unlink()


def test_checkpoint_path_import_destroys_native_handle_and_releases_mapping_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    checkpoint_path = tmp_path / "malformed-native-report.chk"
    checkpoint_path.write_bytes(library.checkpoint_bytes)

    def malformed_import(request: object, handle_out: object, report_out: object) -> int:
        native = _pointed(request, runtime_adapter._NativeImportRequest)
        assert ctypes.string_at(native.checkpoint_bytes, native.checkpoint_len) == (
            library.checkpoint_bytes
        )
        returned = library._publish_handle(handle_out, report_out)
        library._report(report_out).required = 1
        return returned

    library.import_override = malformed_import
    with pytest.raises(NativeTemporalRuntimeError) as rejected:
        NativeTemporalCircuit.from_checkpoint_path(_tiny_definition(), checkpoint_path)
    assert rejected.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
    assert library.destroyed == [0x1234]
    checkpoint_path.unlink()
    assert not checkpoint_path.exists()


def test_checkpoint_path_import_revalidates_descriptor_identity_and_size_after_native(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    checkpoint_path = tmp_path / "changed-during-import.chk"
    checkpoint_path.write_bytes(library.checkpoint_bytes)

    def mutate_during_import(request: object, handle_out: object, report_out: object) -> int:
        del request
        with checkpoint_path.open("ab") as stream:
            stream.write(b"x")
        return library._publish_handle(handle_out, report_out)

    library.import_override = mutate_during_import
    with pytest.raises(NativeTemporalRuntimeError) as rejected:
        NativeTemporalCircuit.from_checkpoint_path(_tiny_definition(), checkpoint_path)
    assert rejected.value.status is NativeTemporalRuntimeStatus.CHECKPOINT_REJECTED
    assert library.destroyed == [0x1234]
    checkpoint_path.unlink()


def test_advance_error_retains_only_the_committed_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    def fail_third_token(handle: object, request: object, report_out: object) -> int:
        del handle
        native = _pointed(request, runtime_adapter._NativeAdvanceRequestV2)
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


@pytest.mark.parametrize("defect", ("status-mismatch", "message-tail", "message-utf8", "step-flag"))
def test_fake_native_transport_defects_fail_as_internal_invariants(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)

    def malformed(handle: object, request: object, report_out: object) -> int:
        del handle
        native = _pointed(request, runtime_adapter._NativeAdvanceRequestV2)
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


def test_checkpoint_identity_stops_before_native_when_authority_or_binary_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, identity = _install_fake_runtime(monkeypatch)
    circuit = NativeTemporalCircuit.fresh(_tiny_definition())
    try:
        monkeypatch.setattr(
            runtime_adapter,
            "_current_native_source_digest",
            lambda: "ef" * 32,
        )
        with pytest.raises(NativeBuildError, match="current native build inputs"):
            circuit.checkpoint_identity()
        assert library.identity_calls == 0

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
            circuit.checkpoint_identity()
        assert authority_error.value.status is NativeTemporalRuntimeStatus.INVALID_DEFINITION
        assert library.identity_calls == 0

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
            circuit.checkpoint_identity()
        assert binary_error.value.status is NativeTemporalRuntimeStatus.INTERNAL_INVARIANT
        assert "library identity changed" in binary_error.value.message
        assert library.identity_calls == 0
    finally:
        circuit.close()

    with pytest.raises(RuntimeError, match="closed"):
        circuit.checkpoint_identity()
    assert library.identity_calls == 0


def test_checkpoint_identity_orders_open_authority_attestation_native_and_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, library_identity = _install_fake_runtime(monkeypatch)
    live = NativeTemporalCircuit.fresh(_tiny_definition())
    closed = NativeTemporalCircuit.fresh(_tiny_definition())
    closed.close()
    trace: list[str] = []

    def fresh_authority(
        protocol_byte: int = 0x11,
    ) -> runtime_adapter._FreshAuthority:
        trace.append("fresh-authority")
        return runtime_adapter._FreshAuthority(
            bytes([protocol_byte]) * 32,
            b"\x22" * 32,
        )

    monkeypatch.setattr(runtime_adapter, "_fresh_authority", fresh_authority)
    with pytest.raises(RuntimeError, match="closed"):
        closed.checkpoint_identity()
    assert trace == []
    assert library.identity_calls == 0

    monkeypatch.setattr(
        runtime_adapter,
        "_fresh_authority",
        lambda: fresh_authority(0x33),
    )

    def forbidden_attestation() -> NativeLibraryIdentity:
        raise AssertionError("checkpoint identity must reject authority before binary attestation")

    monkeypatch.setattr(
        runtime_adapter,
        "_attested_current_library_identity",
        forbidden_attestation,
    )
    with pytest.raises(NativeTemporalRuntimeError) as authority_error:
        live.checkpoint_identity()
    assert authority_error.value.status is NativeTemporalRuntimeStatus.INVALID_DEFINITION
    assert trace == ["fresh-authority"]
    assert library.identity_calls == 0

    trace.clear()
    monkeypatch.setattr(runtime_adapter, "_fresh_authority", fresh_authority)

    def attest() -> NativeLibraryIdentity:
        trace.append("binary-attestation")
        return library_identity

    monkeypatch.setattr(runtime_adapter, "_attested_current_library_identity", attest)
    strict_decode = runtime_adapter._decode_checkpoint_identity

    def decode(
        returned_status: object,
        native: runtime_adapter._NativeCheckpointIdentity,
        report: runtime_adapter._NativeReport,
        authority: runtime_adapter._FreshAuthority,
    ) -> TemporalCheckpointIdentity:
        trace.append("strict-decode")
        return strict_decode(returned_status, native, report, authority)

    monkeypatch.setattr(runtime_adapter, "_decode_checkpoint_identity", decode)

    def native_identity(
        handle: object,
        authority: object,
        output: object,
        report_out: object,
    ) -> int:
        del handle, authority
        trace.append("native-identity")
        ctypes.memmove(output, library.checkpoint_bytes, 256)
        report = library._report(report_out)
        report.required = 256
        report.observed = 256
        return 0

    library.identity_override = native_identity
    try:
        decoded = live.checkpoint_identity()
    finally:
        live.close()

    assert trace == [
        "fresh-authority",
        "binary-attestation",
        "native-identity",
        "strict-decode",
    ]
    assert decoded.logical_state_sha256 == library.checkpoint_bytes[224:256].hex()
    assert library.identity_calls == 1


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
        native = _pointed(request, runtime_adapter._NativeAdvanceRequestV2)
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


def test_close_waits_for_inflight_checkpoint_identity_before_destroying_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, _identity = _install_fake_runtime(monkeypatch)
    circuit = NativeTemporalCircuit.fresh(_tiny_definition())
    entered_native = threading.Event()
    release_native = threading.Event()
    close_started = threading.Event()

    def blocked_identity(
        handle: object,
        authority: object,
        output: object,
        report_out: object,
    ) -> int:
        del handle, authority
        entered_native.set()
        if not release_native.wait(timeout=5.0):
            raise AssertionError("test did not release native checkpoint identity")
        ctypes.memmove(output, library.checkpoint_bytes, 256)
        report = library._report(report_out)
        report.required = 256
        report.observed = 256
        return 0

    library.identity_override = blocked_identity

    def close_circuit() -> None:
        close_started.set()
        circuit.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        identity_future = executor.submit(circuit.checkpoint_identity)
        assert entered_native.wait(timeout=5.0)
        close_future = executor.submit(close_circuit)
        assert close_started.wait(timeout=5.0)
        try:
            assert not close_future.done()
            assert library.destroyed == []
        finally:
            release_native.set()
        identity = identity_future.result(timeout=5.0)
        close_future.result(timeout=5.0)

    assert identity.logical_state_sha256 == library.checkpoint_bytes[224:256].hex()
    assert library.identity_calls == 1
    assert library.destroyed == [0x1234]
    circuit.close()
    assert library.destroyed == [0x1234]


def test_actual_native_fresh_checkpoint_import_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = runtime_adapter._FreshAuthority(b"\x41" * 32, b"\x92" * 32)
    authority_calls: list[None] = []

    def fresh_authority() -> runtime_adapter._FreshAuthority:
        authority_calls.append(None)
        return authority

    monkeypatch.setattr(runtime_adapter, "_fresh_authority", fresh_authority)
    definition = _tiny_definition()
    definition_identity = definition.circuit_identity_sha256()
    assert authority_calls == []

    with NativeTemporalCircuit.fresh(definition) as fresh:
        empty_identity = fresh.checkpoint_identity()
        assert empty_identity.checkpoint_bytes == len(fresh.checkpoint()) == 256
        steps = fresh.advance_block([7], mode=TemporalStepMode.READ_ONLY)
        assert len(steps) == 1
        assert steps[0].token_id == 7
        identity = fresh.checkpoint_identity()
        checkpoint = fresh.checkpoint()
        header = checkpoint[:256]
        assert identity != empty_identity
        assert identity.format_version == int.from_bytes(header[8:10], "little")
        assert identity.header_bytes == int.from_bytes(header[10:12], "little")
        assert identity.checkpoint_bytes == int.from_bytes(header[16:24], "little")
        assert identity.body_bytes == int.from_bytes(header[24:32], "little")
        assert identity.protocol_identity_sha256 == header[32:64].hex()
        assert identity.source_bundle_identity_sha256 == header[64:96].hex()
        assert identity.circuit_identity_sha256 == header[96:128].hex()
        assert identity.circuit_identity_sha256 == definition_identity
        assert identity.body_sha256 == header[128:160].hex()
        assert identity.semantic_index_sha256 == header[160:192].hex()
        assert (
            identity.segment_count,
            identity.cohort_count,
            identity.next_reserve_cell,
            identity.event_count,
            identity.learning_step,
            identity.prior_context_length,
            identity.winner_context_length,
            identity.poison_code,
        ) == struct.unpack("<IIIQQBBH", header[192:224])
        assert identity.logical_state_sha256 == header[224:256].hex()
        checkpoint_path = tmp_path / "published.chk"
        publication = fresh.publish_checkpoint(checkpoint_path)
        assert publication.outcome is TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED
        assert publication.semantically_resumable is True
        assert publication.checkpoint_bytes == len(checkpoint)
        assert publication.encoded_bytes == len(checkpoint)
        assert checkpoint_path.read_bytes() == checkpoint
        collision = fresh.publish_checkpoint(checkpoint_path)
        assert collision.outcome is TemporalCheckpointPublicationOutcome.COLLISION
        assert collision.semantically_resumable is False
    assert len(checkpoint) > 256

    with NativeTemporalCircuit.from_checkpoint_path(definition, checkpoint_path) as mapped:
        assert mapped.checkpoint() == checkpoint
        assert mapped.checkpoint_identity() == identity
    with NativeTemporalCircuit.from_checkpoint(definition, checkpoint) as restored:
        assert restored.checkpoint() == checkpoint
        assert restored.checkpoint_identity() == identity


def test_actual_native_immutable_bytes_publication_is_exclusive_and_bit_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authority = runtime_adapter._FreshAuthority(b"\x41" * 32, b"\x92" * 32)
    monkeypatch.setattr(runtime_adapter, "_fresh_authority", lambda: authority)
    destination = tmp_path / "result.json"
    payload = b'{"status":"complete","checkpoint":"generation-7"}\x00\xff'

    published = runtime_adapter.publish_immutable_bytes(destination, payload)

    assert published.outcome is runtime_adapter.TemporalPublicationOutcome.DURABLE_PUBLISHED
    assert published.durably_published
    assert published.payload_bytes == len(payload)
    assert published.written_bytes == len(payload)
    assert destination.read_bytes() == payload

    collision = runtime_adapter.publish_immutable_bytes(destination, b"different")
    assert collision.outcome is runtime_adapter.TemporalPublicationOutcome.COLLISION
    assert not collision.durably_published
    assert destination.read_bytes() == payload

    reconciled = runtime_adapter.reconcile_existing_immutable_bytes(destination, payload)
    assert reconciled.durably_reconciled
    assert reconciled.verified_bytes == len(payload)
    mismatch = runtime_adapter.reconcile_existing_immutable_bytes(destination, b"different")
    assert mismatch.outcome is runtime_adapter.TemporalDurabilityOutcome.NOT_DURABLE
    assert destination.read_bytes() == payload

    prepared = runtime_adapter.prepare_durable_directory(tmp_path, Path("durable") / "generation-7")
    assert prepared.durably_prepared
    assert prepared.required_component_count == 2
    assert prepared.confirmed_component_count == 2
    assert (tmp_path / "durable" / "generation-7").is_dir()
    repeated = runtime_adapter.prepare_durable_directory(tmp_path, Path("durable") / "generation-7")
    assert repeated.durably_prepared
    assert repeated.created_component_count == 0
