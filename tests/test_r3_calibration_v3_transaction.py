from __future__ import annotations

import hashlib
import importlib
import json
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from experiments.r3_pilot import calibration_v3_persistence as persistence
from experiments.r3_pilot import calibration_v3_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = "experiments/r3_pilot/calibration_v3_artifacts"
ARM_SLUG = "theta-m-10_theta-a-13"
RESULT_PATH = (
    f"{ARTIFACT_ROOT}/r3-f0/deterministic-confirmation/{ARM_SLUG}/construction-seed-0000000017.json"
)
MANIFEST_PATH = (
    f"{ARTIFACT_ROOT}/r3-f0/deterministic-confirmation/{ARM_SLUG}/resume/"
    "transition-0000529340.resume.json"
)
SECOND_MANIFEST_PATH = (
    f"{ARTIFACT_ROOT}/r3-f0/deterministic-confirmation/{ARM_SLUG}/resume/"
    "transition-0001058681.resume.json"
)
CHECKPOINT_PATH = MANIFEST_PATH.removesuffix(".resume.json") + ".r3ntcp1"
SOURCE_BUNDLE_SHA256 = "3" * 64
ARM_CIRCUIT_IDENTITY_SHA256 = "be6d5470a8cab36cd8a8d52b456c2d7601012a71d4b689999233b6c37e84874a"
REFERENCE_TRAJECTORY_ID = (
    "r3-f0/deterministic-confirmation/theta-m-10_theta-a-13/construction-seed-0000000017/fresh"
)

IDENTITY_FIELDS = (
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
STEP_FIELDS = (
    "token_id",
    "punished_segments",
    "event_count",
    "candidate_segments",
    "matching_segments",
    "active_segments",
    "postings_visited",
    "predicted",
    "predicted_columns",
    "allocated_segments",
    "inhibition_onset_ticks",
    "column_simulated_ticks",
    "winner_spikes",
    "basket_spikes",
    "proximal_spikes",
    "proximal_relay_spikes",
    "winner_relay_spikes",
    "winner_context",
    "proximal_context",
    "context",
)


def _production_modules() -> tuple[ModuleType, ModuleType]:
    authority = importlib.import_module("experiments.r3_pilot.calibration_v3_authority")
    transaction = importlib.import_module("experiments.r3_pilot.calibration_v3_transaction")
    return authority, transaction


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _runnable_protocol() -> dict[str, Any]:
    frozen = json.loads(json.dumps(protocol.build_protocol()))
    frozen["source_binding"]["runnable"] = True
    frozen["source_binding"]["blockers"] = []
    return frozen


def _protocol_sha256(frozen: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(frozen)).hexdigest()


def _absolute(root: Path, relative_path: str) -> Path:
    return root.joinpath(*relative_path.split("/"))


@dataclass(frozen=True, slots=True)
class _DefinitionConfig:
    column_count: int = 4_096
    max_pyramidal_cells: int = 131_072
    segment_capacity: int = 23_286_406
    construction_seed: int = 17
    base_cells_per_column: int = 8
    novel_race_cap: int = 8
    max_segments_per_cell: int = 255
    theta_match: int = 10
    theta_active: int = 13


@dataclass(frozen=True, slots=True)
class _Definition:
    config: _DefinitionConfig = _DefinitionConfig()
    token_codes: tuple[object, ...] = (None,) * 768
    threshold_offsets_mv: tuple[float, ...] = (0.0,) * 274_432


class _Clock:
    def __init__(
        self,
        events: list[str],
        now_ns: int = 100,
        *,
        scripted_ns: list[int] | None = None,
    ) -> None:
        self.events = events
        self.now_ns = now_ns
        self.on_next: Callable[[], None] | None = None
        self.scripted_ns = scripted_ns

    def __call__(self) -> int:
        self.events.append("clock")
        if self.on_next is not None:
            callback = self.on_next
            self.on_next = None
            callback()
        if self.scripted_ns is not None:
            if not self.scripted_ns:
                raise AssertionError("scripted clock was read more times than expected")
            return self.scripted_ns.pop(0)
        return self.now_ns


def _identity(
    native: ModuleType,
    protocol_sha256: str,
    circuit_identity_sha256: str = ARM_CIRCUIT_IDENTITY_SHA256,
) -> object:
    segment_count = 56
    cohort_count = 20
    next_reserve_cell = 81_632
    event_count = 42_124
    learning_step = 529_346
    prior_context_length = 22
    winner_context_length = 22
    poison_code = 0
    body_sha256 = "5" * 64
    logical_state_sha256 = hashlib.sha256(
        b"humanbrain-r3-checkpoint-logical-state-v1\x00"
        + bytes.fromhex(body_sha256)
        + struct.pack(
            "<IIIQQBBH",
            segment_count,
            cohort_count,
            next_reserve_cell,
            event_count,
            learning_step,
            prior_context_length,
            winner_context_length,
            poison_code,
        )
    ).hexdigest()
    return native.TemporalCheckpointIdentity(
        format_version=1,
        header_bytes=256,
        checkpoint_bytes=4_096,
        body_bytes=3_840,
        protocol_identity_sha256=protocol_sha256,
        source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
        circuit_identity_sha256=circuit_identity_sha256,
        body_sha256=body_sha256,
        semantic_index_sha256="6" * 64,
        segment_count=segment_count,
        cohort_count=cohort_count,
        next_reserve_cell=next_reserve_cell,
        event_count=event_count,
        learning_step=learning_step,
        prior_context_length=prior_context_length,
        winner_context_length=winner_context_length,
        poison_code=poison_code,
        logical_state_sha256=logical_state_sha256,
    )


def _identity_payload(identity: object) -> dict[str, object]:
    return {
        "format_version": identity.format_version,
        "header_bytes": identity.header_bytes,
        "checkpoint_bytes": identity.checkpoint_bytes,
        "body_bytes": identity.body_bytes,
        "protocol_identity_sha256": identity.protocol_identity_sha256,
        "source_bundle_identity_sha256": (identity.source_bundle_identity_sha256),
        "circuit_identity_sha256": identity.circuit_identity_sha256,
        "body_sha256": identity.body_sha256,
        "semantic_index_sha256": identity.semantic_index_sha256,
        "segment_count": identity.segment_count,
        "cohort_count": identity.cohort_count,
        "next_reserve_cell": identity.next_reserve_cell,
        "event_count": identity.event_count,
        "learning_step": identity.learning_step,
        "prior_context_length": identity.prior_context_length,
        "winner_context_length": identity.winner_context_length,
        "poison_code": identity.poison_code,
        "logical_state_sha256": identity.logical_state_sha256,
    }


def _logical_state_sha256(identity: object) -> str:
    return hashlib.sha256(
        b"humanbrain-r3-checkpoint-logical-state-v1\x00"
        + bytes.fromhex(identity.body_sha256)
        + struct.pack(
            "<IIIQQBBH",
            identity.segment_count,
            identity.cohort_count,
            identity.next_reserve_cell,
            identity.event_count,
            identity.learning_step,
            identity.prior_context_length,
            identity.winner_context_length,
            identity.poison_code,
        )
    ).hexdigest()


def _different_identity(identity: object, field: str) -> object:
    value = getattr(identity, field)
    replacement = ("0" * 64 if value != "0" * 64 else "1" * 64) if type(value) is str else value + 1
    return replace(identity, **{field: replacement})


def _spikes(base: int) -> tuple[persistence.TemporalSpikeFrame, ...]:
    return tuple(
        persistence.TemporalSpikeFrame(cell_id=base + offset, tick=100 + offset)
        for offset in range(11)
    )


def _last_step() -> persistence.TemporalStepFrame:
    return persistence.TemporalStepFrame(
        token_id=37,
        punished_segments=2,
        event_count=42_124,
        candidate_segments=17,
        matching_segments=15,
        active_segments=13,
        postings_visited=21,
        predicted=True,
        predicted_columns=9,
        allocated_segments=2,
        inhibition_onset_ticks=tuple(range(11)),
        column_simulated_ticks=tuple(range(20, 31)),
        winner_spikes=_spikes(1_000),
        basket_spikes=_spikes(2_000),
        proximal_spikes=_spikes(3_000),
        proximal_relay_spikes=_spikes(4_000),
        winner_relay_spikes=_spikes(5_000),
        winner_context=tuple(range(6_000, 6_022)),
        proximal_context=tuple(range(7_000, 7_022)),
        context=tuple(range(8_000, 8_022)),
    )


def _step_at(token_id: int, event_count: int) -> persistence.TemporalStepFrame:
    return replace(_last_step(), token_id=token_id, event_count=event_count)


def _native_step(native: ModuleType, frame: persistence.TemporalStepFrame) -> object:
    def spikes(
        values: tuple[persistence.TemporalSpikeFrame, ...],
    ) -> tuple[object, ...]:
        return tuple(native.TemporalSpike(value.cell_id, value.tick) for value in values)

    return native.TemporalCircuitStep(
        token_id=frame.token_id,
        punished_segments=frame.punished_segments,
        event_count=frame.event_count,
        candidate_segments=frame.candidate_segments,
        matching_segments=frame.matching_segments,
        active_segments=frame.active_segments,
        postings_visited=frame.postings_visited,
        predicted=frame.predicted,
        predicted_columns=frame.predicted_columns,
        allocated_segments=frame.allocated_segments,
        inhibition_onset_ticks=frame.inhibition_onset_ticks,
        column_simulated_ticks=frame.column_simulated_ticks,
        winner_spikes=spikes(frame.winner_spikes),
        basket_spikes=spikes(frame.basket_spikes),
        proximal_spikes=spikes(frame.proximal_spikes),
        proximal_relay_spikes=spikes(frame.proximal_relay_spikes),
        winner_relay_spikes=spikes(frame.winner_relay_spikes),
        winner_context=frame.winner_context,
        proximal_context=frame.proximal_context,
        context=frame.context,
    )


def _progress(transaction: ModuleType) -> object:
    return transaction.FrozenResumeProgress(
        last_step=_last_step(),
        winner_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.WINNER,
            count=529_340,
            digest=b"\x11" * 32,
        ),
        step_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.STEP,
            count=529_346,
            digest=b"\x22" * 32,
        ),
    )


def _progress_for_cursor(transaction: ModuleType, cursor: dict[str, Any]) -> object:
    step = replace(
        _last_step(),
        event_count=cursor["last_token_index_zero_based"] + 1,
    )
    return transaction.FrozenResumeProgress(
        last_step=step,
        winner_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.WINNER,
            count=cursor["after_transition_event"],
            digest=b"\x33" * 32,
        ),
        step_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.STEP,
            count=cursor["event_count"],
            digest=b"\x44" * 32,
        ),
    )


def _seed_progress(session: object, progress: object) -> object:
    """Trusted test setup for coordinator-owned progress at a distant milestone."""

    session._owner._progress = progress
    return session


def _commit_with_progress(
    coordinator: object,
    session: object,
    progress: object,
) -> object:
    _seed_progress(session, progress)
    return coordinator.commit_resume_checkpoint(session)


def _step_payload(step: persistence.TemporalStepFrame) -> dict[str, object]:
    def spikes(values: tuple[persistence.TemporalSpikeFrame, ...]) -> list[dict[str, int]]:
        return [{"cell_id": spike.cell_id, "tick": spike.tick} for spike in values]

    return {
        "token_id": step.token_id,
        "punished_segments": step.punished_segments,
        "event_count": step.event_count,
        "candidate_segments": step.candidate_segments,
        "matching_segments": step.matching_segments,
        "active_segments": step.active_segments,
        "postings_visited": step.postings_visited,
        "predicted": step.predicted,
        "predicted_columns": step.predicted_columns,
        "allocated_segments": step.allocated_segments,
        "inhibition_onset_ticks": list(step.inhibition_onset_ticks),
        "column_simulated_ticks": list(step.column_simulated_ticks),
        "winner_spikes": spikes(step.winner_spikes),
        "basket_spikes": spikes(step.basket_spikes),
        "proximal_spikes": spikes(step.proximal_spikes),
        "proximal_relay_spikes": spikes(step.proximal_relay_spikes),
        "winner_relay_spikes": spikes(step.winner_relay_spikes),
        "winner_context": list(step.winner_context),
        "proximal_context": list(step.proximal_context),
        "context": list(step.context),
    }


def _r3_result() -> dict[str, object]:
    return {
        "arm_slug": ARM_SLUG,
        "comparisons": {},
        "kind": "r3_f0_train",
        "reference": {},
        "resumes": [],
        "structural_metrics": {"step_count": 529_346},
        "train_input": {"held_out_access": "forbidden"},
        "verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
    }


def _error_payload(error: object) -> dict[str, object]:
    return {
        "stage": error.stage.name,
        "domain": error.domain.name,
        "code": error.code,
    }


def _publication_payload(publication: object) -> dict[str, object]:
    return {
        "outcome": publication.outcome.name,
        "temp_cleanup": publication.temp_cleanup.name,
        "primary_error": (
            None if publication.primary_error is None else _error_payload(publication.primary_error)
        ),
        "cleanup_errors": [_error_payload(error) for error in publication.cleanup_errors],
        "owned_temp_pid": publication.owned_temp_pid,
        "owned_temp_nonce": publication.owned_temp_nonce,
        "checkpoint_bytes": publication.checkpoint_bytes,
        "encoded_bytes": publication.encoded_bytes,
    }


class _NativeHarness:
    def __init__(
        self,
        *,
        events: list[str],
        live_identity: object,
        imported_identity: object,
        checkpoint_publication: object,
        after_checkpoint: Callable[[], None] | None = None,
        during_import: Callable[[Path], None] | None = None,
    ) -> None:
        self.events = events
        self.live_identity = live_identity
        self.imported_identity = imported_identity
        self.checkpoint_publication = checkpoint_publication
        self.after_checkpoint = after_checkpoint
        self.during_import = during_import
        self.circuits: list[Any] = []
        self.checkpoint_attempts = 0
        self.manifest_attempts = 0
        self.result_attempts = 0
        self.close_error_kinds: set[str] = set()
        self.close_errors: dict[str, BaseException] = {}
        self.close_attempts: list[tuple[str, int]] = []
        self.advance_error: BaseException | None = None
        self.advance_results: list[tuple[object, ...]] = []
        self.directory_preparations: list[Path] = []
        self.directory_preparation_failure_at: int | None = None
        self.manifest_reconciliations: list[tuple[Path, bytes]] = []
        self.manifest_reconciliation_durable = True

    def set_active_identity(self, identity: object) -> None:
        self.live_identity = identity
        for circuit in self.circuits:
            if not circuit.closed:
                circuit.identity = identity

    def install(self, monkeypatch: pytest.MonkeyPatch, native: ModuleType) -> None:
        harness = self

        class FakeCircuit:
            def __init__(
                self,
                *,
                kind: str,
                definition: object,
                checkpoint_identity: object,
            ) -> None:
                self.kind = kind
                self.definition = definition
                self.identity = checkpoint_identity
                self.closed = False
                self.serial = len(harness.circuits)
                harness.circuits.append(self)

            @classmethod
            def fresh(cls, definition: object) -> FakeCircuit:
                harness.events.append("fresh")
                return cls(
                    kind="fresh",
                    definition=definition,
                    checkpoint_identity=harness.live_identity,
                )

            @classmethod
            def from_checkpoint_path(
                cls,
                definition: object,
                path: Path,
            ) -> FakeCircuit:
                harness.events.append("import")
                assert path.is_file()
                if harness.during_import is not None:
                    harness.during_import(path)
                return cls(
                    kind="import",
                    definition=definition,
                    checkpoint_identity=harness.imported_identity,
                )

            def checkpoint_identity(self) -> object:
                assert not self.closed
                harness.events.append(f"identity:{self.kind}:{self.serial}")
                return self.identity

            def advance_block(
                self,
                token_ids: object,
                *,
                mode: object,
                reset_before: bool,
            ) -> tuple[object, ...]:
                tuple(token_ids)
                del mode, reset_before
                assert not self.closed
                harness.events.append(f"advance:{self.serial}")
                if harness.advance_error is not None:
                    raise harness.advance_error
                if harness.advance_results:
                    return harness.advance_results.pop(0)
                return ()

            def publish_checkpoint(self, destination: Path) -> object:
                assert not self.closed
                harness.events.append(f"checkpoint:{self.serial}")
                harness.checkpoint_attempts += 1
                report = harness.checkpoint_publication
                if report.outcome in {
                    native.TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED,
                    native.TemporalCheckpointPublicationOutcome.PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
                }:
                    assert destination.parent.is_dir()
                    destination.write_bytes(b"C" * report.checkpoint_bytes)
                if harness.after_checkpoint is not None:
                    harness.after_checkpoint()
                return report

            def close(self) -> None:
                if not self.closed:
                    self.closed = True
                    harness.events.append(f"close:{self.kind}:{self.serial}")
                    harness.close_attempts.append((self.kind, self.serial))
                    if self.kind in harness.close_error_kinds:
                        raise RuntimeError(f"synthetic {self.kind} close failure")
                    if self.kind in harness.close_errors:
                        raise harness.close_errors[self.kind]

        def publish_immutable_bytes(destination: Path, payload: bytes) -> object:
            if destination.name.endswith(".resume.json"):
                harness.events.append("manifest_publish")
                harness.manifest_attempts += 1
            else:
                harness.events.append("result_publish")
                harness.result_attempts += 1
            assert type(payload) is bytes
            assert destination.parent.is_dir()
            if destination.exists():
                return native.TemporalImmutableBytesPublication(
                    outcome=native.TemporalPublicationOutcome.COLLISION,
                    temp_cleanup=native.TemporalPublicationCleanup.CLEAN,
                    primary_error=None,
                    cleanup_errors=(),
                    owned_temp_pid=0,
                    owned_temp_nonce=0,
                    payload_bytes=len(payload),
                    written_bytes=0,
                )
            destination.write_bytes(payload)
            return native.TemporalImmutableBytesPublication(
                outcome=native.TemporalPublicationOutcome.DURABLE_PUBLISHED,
                temp_cleanup=native.TemporalPublicationCleanup.CLEAN,
                primary_error=None,
                cleanup_errors=(),
                owned_temp_pid=41,
                owned_temp_nonce=42,
                payload_bytes=len(payload),
                written_bytes=len(payload),
            )

        def prepare_durable_directory(
            root: Path,
            relative_directory: Path,
        ) -> object:
            harness.events.append(f"directory:{relative_directory.as_posix()}")
            harness.directory_preparations.append(relative_directory)
            required = len(relative_directory.parts)
            call_index = len(harness.directory_preparations) - 1
            if harness.directory_preparation_failure_at == call_index:
                return native.TemporalDurableDirectoryPreparation(
                    outcome=native.TemporalDurabilityOutcome.NOT_DURABLE,
                    primary_error=native.TemporalDurabilityError(
                        stage=native.TemporalDurabilityErrorStage.DIRECTORY_SYNC,
                        domain=native.TemporalDurabilityErrorDomain.IO,
                        code=5,
                    ),
                    cleanup_errors=(),
                    required_component_count=required,
                    confirmed_component_count=max(0, required - 1),
                    created_component_count=0,
                    failed_component_index=max(0, required - 1),
                )
            root.joinpath(*relative_directory.parts).mkdir(parents=True, exist_ok=True)
            return native.TemporalDurableDirectoryPreparation(
                outcome=native.TemporalDurabilityOutcome.DURABLE,
                primary_error=None,
                cleanup_errors=(),
                required_component_count=required,
                confirmed_component_count=required,
                created_component_count=0,
                failed_component_index=None,
            )

        def reconcile_existing_immutable_bytes(
            destination: Path,
            expected_payload: bytes,
        ) -> object:
            harness.events.append("manifest_reconcile")
            harness.manifest_reconciliations.append((destination, expected_payload))
            if not harness.manifest_reconciliation_durable:
                return native.TemporalImmutableBytesReconciliation(
                    outcome=native.TemporalDurabilityOutcome.NOT_DURABLE,
                    primary_error=native.TemporalDurabilityError(
                        stage=native.TemporalDurabilityErrorStage.FILE_SYNC,
                        domain=native.TemporalDurabilityErrorDomain.IO,
                        code=5,
                    ),
                    cleanup_errors=(),
                    expected_bytes=len(expected_payload),
                    verified_bytes=len(expected_payload),
                )
            assert destination.read_bytes() == expected_payload
            return native.TemporalImmutableBytesReconciliation(
                outcome=native.TemporalDurabilityOutcome.DURABLE,
                primary_error=None,
                cleanup_errors=(),
                expected_bytes=len(expected_payload),
                verified_bytes=len(expected_payload),
            )

        monkeypatch.setattr(native, "NativeTemporalCircuit", FakeCircuit)
        monkeypatch.setattr(native, "NativeTemporalCircuitDefinition", _Definition)
        monkeypatch.setattr(native, "publish_immutable_bytes", publish_immutable_bytes)
        monkeypatch.setattr(
            native,
            "prepare_durable_directory",
            prepare_durable_directory,
        )
        monkeypatch.setattr(
            native,
            "reconcile_existing_immutable_bytes",
            reconcile_existing_immutable_bytes,
        )


def _checkpoint_publication(native: ModuleType, mode: str) -> object:
    if mode == "durable_clean":
        outcome = native.TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED
        cleanup = native.TemporalCheckpointCleanup.CLEAN
        primary = None
        cleanup_errors: tuple[object, ...] = ()
        owned_temp_pid = 31
        owned_temp_nonce = 0
        encoded_bytes = 4_096
    elif mode == "durable_orphaned":
        outcome = native.TemporalCheckpointPublicationOutcome.DURABLE_PUBLISHED
        cleanup = native.TemporalCheckpointCleanup.ORPHANED
        primary = None
        cleanup_errors = (
            native.TemporalCheckpointPublicationError(
                stage=native.TemporalCheckpointErrorStage.TEMP_CLEANUP,
                domain=native.TemporalCheckpointErrorDomain.POLICY,
                code=91,
            ),
        )
        owned_temp_pid = 31
        owned_temp_nonce = 0
        encoded_bytes = 4_096
    elif mode == "collision":
        outcome = native.TemporalCheckpointPublicationOutcome.COLLISION
        cleanup = native.TemporalCheckpointCleanup.CLEAN
        primary = None
        cleanup_errors = ()
        owned_temp_pid = 31
        owned_temp_nonce = 0
        encoded_bytes = 4_096
    elif mode == "not_published":
        outcome = native.TemporalCheckpointPublicationOutcome.NOT_PUBLISHED
        cleanup = native.TemporalCheckpointCleanup.NOT_APPLICABLE
        primary = native.TemporalCheckpointPublicationError(
            stage=native.TemporalCheckpointErrorStage.PARENT_OPEN,
            domain=native.TemporalCheckpointErrorDomain.POLICY,
            code=7,
        )
        cleanup_errors = ()
        owned_temp_pid = 0
        owned_temp_nonce = 0
        encoded_bytes = 0
    elif mode == "unconfirmed":
        outcome = native.TemporalCheckpointPublicationOutcome.PUBLISHED_BUT_NOT_CONFIRMED_DURABLE
        cleanup = native.TemporalCheckpointCleanup.CLEAN
        primary = native.TemporalCheckpointPublicationError(
            stage=native.TemporalCheckpointErrorStage.PARENT_SYNC,
            domain=native.TemporalCheckpointErrorDomain.IO,
            code=8,
        )
        cleanup_errors = ()
        owned_temp_pid = 31
        owned_temp_nonce = 0
        encoded_bytes = 4_096
    else:  # pragma: no cover - test construction guard.
        raise AssertionError(f"unknown publication mode: {mode}")
    return native.TemporalCheckpointPublication(
        outcome=outcome,
        temp_cleanup=cleanup,
        primary_error=primary,
        cleanup_errors=cleanup_errors,
        owned_temp_pid=owned_temp_pid,
        owned_temp_nonce=owned_temp_nonce,
        checkpoint_bytes=4_096,
        encoded_bytes=encoded_bytes,
    )


def _manifest_publication(native: ModuleType, mode: str, payload_bytes: int) -> object:
    if mode == "durable_clean":
        outcome = native.TemporalPublicationOutcome.DURABLE_PUBLISHED
        cleanup = native.TemporalPublicationCleanup.CLEAN
        primary = None
        cleanup_errors: tuple[object, ...] = ()
        owned_temp_pid = 41
        owned_temp_nonce = 0
        written_bytes = payload_bytes
    elif mode == "durable_orphaned":
        outcome = native.TemporalPublicationOutcome.DURABLE_PUBLISHED
        cleanup = native.TemporalPublicationCleanup.ORPHANED
        primary = None
        cleanup_errors = (
            native.TemporalPublicationError(
                stage=native.TemporalPublicationErrorStage.TEMP_CLEANUP,
                domain=native.TemporalPublicationErrorDomain.POLICY,
                code=91,
            ),
        )
        owned_temp_pid = 41
        owned_temp_nonce = 0
        written_bytes = payload_bytes
    elif mode == "collision":
        outcome = native.TemporalPublicationOutcome.COLLISION
        cleanup = native.TemporalPublicationCleanup.CLEAN
        primary = None
        cleanup_errors = ()
        owned_temp_pid = 41
        owned_temp_nonce = 0
        written_bytes = payload_bytes
    elif mode == "not_published":
        outcome = native.TemporalPublicationOutcome.NOT_PUBLISHED
        cleanup = native.TemporalPublicationCleanup.NOT_APPLICABLE
        primary = native.TemporalPublicationError(
            stage=native.TemporalPublicationErrorStage.PARENT_OPEN,
            domain=native.TemporalPublicationErrorDomain.POLICY,
            code=7,
        )
        cleanup_errors = ()
        owned_temp_pid = 0
        owned_temp_nonce = 0
        written_bytes = 0
    elif mode == "unconfirmed":
        outcome = native.TemporalPublicationOutcome.PUBLISHED_BUT_NOT_CONFIRMED_DURABLE
        cleanup = native.TemporalPublicationCleanup.CLEAN
        primary = native.TemporalPublicationError(
            stage=native.TemporalPublicationErrorStage.PARENT_SYNC,
            domain=native.TemporalPublicationErrorDomain.IO,
            code=8,
        )
        cleanup_errors = ()
        owned_temp_pid = 41
        owned_temp_nonce = 0
        written_bytes = payload_bytes
    else:  # pragma: no cover - test construction guard.
        raise AssertionError(f"unknown manifest publication mode: {mode}")
    return native.TemporalImmutableBytesPublication(
        outcome=outcome,
        temp_cleanup=cleanup,
        primary_error=primary,
        cleanup_errors=cleanup_errors,
        owned_temp_pid=owned_temp_pid,
        owned_temp_nonce=owned_temp_nonce,
        payload_bytes=payload_bytes,
        written_bytes=written_bytes,
    )


@dataclass(slots=True)
class _Environment:
    transaction: ModuleType
    native: ModuleType
    authority: object
    frozen: dict[str, Any]
    clock: _Clock
    harness: _NativeHarness
    coordinator: object
    identity: object


def _environment(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    publication_mode: str = "durable_clean",
    imported_identity_mutator: Callable[[object], object] | None = None,
    after_checkpoint: Callable[[], None] | None = None,
    during_import: Callable[[Path], None] | None = None,
    clock: _Clock | None = None,
    live_circuit_identity_sha256: str = ARM_CIRCUIT_IDENTITY_SHA256,
) -> _Environment:
    authority_module, transaction = _production_modules()
    native = importlib.import_module("humanbrain.native_temporal_runtime")
    frozen = _runnable_protocol()
    digest = _protocol_sha256(frozen)
    identity = _identity(native, digest, live_circuit_identity_sha256)
    imported_identity = (
        identity if imported_identity_mutator is None else imported_identity_mutator(identity)
    )
    events = [] if clock is None else clock.events
    exact_clock = _Clock(events) if clock is None else clock
    harness = _NativeHarness(
        events=events,
        live_identity=identity,
        imported_identity=imported_identity,
        checkpoint_publication=_checkpoint_publication(native, publication_mode),
        after_checkpoint=after_checkpoint,
        during_import=during_import,
    )
    harness.install(monkeypatch, native)
    authority = authority_module.FrozenPlanAuthority(
        repository_root=root,
        protocol=frozen,
        protocol_sha256=digest,
        source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )
    coordinator = transaction.CalibrationV3Transaction._from_authority_for_test(
        authority,
        clock=exact_clock,
    )
    return _Environment(
        transaction=transaction,
        native=native,
        authority=authority,
        frozen=frozen,
        clock=exact_clock,
        harness=harness,
        coordinator=coordinator,
        identity=identity,
    )


def _read_manifest(environment: _Environment, root: Path) -> dict[str, Any]:
    store = persistence.CalibrationV3ArtifactStore.open(
        repository_root=root,
        validated_protocol=environment.frozen,
        expected_source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )
    return store.read(MANIFEST_PATH).payload


def _new_coordinator(environment: _Environment) -> object:
    return environment.transaction.CalibrationV3Transaction._from_authority_for_test(
        environment.authority,
        clock=environment.clock,
    )


def _assert_durable_namespace_precedes_clock(events: list[str]) -> None:
    directory_indices = [
        index for index, event in enumerate(events) if event.startswith("directory:")
    ]
    assert len(directory_indices) == 54
    assert directory_indices == list(range(54))
    assert events[54] == "clock"


def test_transaction_import_is_transport_only_and_never_imports_the_runner() -> None:
    script = "\n".join(
        (
            "import sys",
            "runner = 'experiments.r3_pilot.run_calibration_v3_fixtures'",
            "native = 'humanbrain.native_temporal_runtime'",
            "assert runner not in sys.modules",
            "assert native not in sys.modules",
            "import experiments.r3_pilot.calibration_v3_authority",
            "import experiments.r3_pilot.calibration_v3_transaction",
            "assert runner not in sys.modules",
            "assert native not in sys.modules",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_constructor_rejects_clock_injection_and_owns_perf_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    calls: list[int] = []

    def production_clock() -> int:
        calls.append(len(calls))
        return 1_000 + len(calls)

    monkeypatch.setattr(environment.transaction.time, "perf_counter_ns", production_clock)
    with pytest.raises(TypeError, match="clock"):
        environment.transaction.CalibrationV3Transaction.from_authority(
            environment.authority,
            clock=lambda: 0,
        )
    assert calls == []
    assert environment.harness.circuits == []

    coordinator = environment.transaction.CalibrationV3Transaction.from_authority(
        environment.authority
    )
    session = coordinator.start_fresh(RESULT_PATH, _Definition)
    assert len(calls) >= 2
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        session.close()


def test_nonrunnable_authority_stops_before_clock_store_native_or_filesystem(
    tmp_path: Path,
) -> None:
    authority_module, transaction = _production_modules()
    frozen = protocol.build_protocol()
    untouched = tmp_path / "must-not-exist"
    calls: list[str] = []
    authority = authority_module.FrozenPlanAuthority(
        repository_root=untouched,
        protocol=frozen,
        protocol_sha256=hashlib.sha256(protocol.canonical_json_bytes(frozen)).hexdigest(),
        source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )

    def forbidden_clock() -> int:
        calls.append("clock")
        pytest.fail("clock must not be read for non-runnable authority")

    with pytest.raises(
        transaction.CalibrationV3TransactionStop,
        match="runnable",
    ):
        transaction.CalibrationV3Transaction._from_authority_for_test(
            authority,
            clock=forbidden_clock,
        )

    assert calls == []
    assert not untouched.exists()
    assert list(tmp_path.iterdir()) == []


def test_malformed_runnable_authority_stops_before_filesystem_clock_or_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_module, transaction = _production_modules()
    frozen = _runnable_protocol()
    frozen["artifact_contract"] = None
    authority = authority_module.FrozenPlanAuthority(
        repository_root=tmp_path,
        protocol=frozen,
        protocol_sha256=_protocol_sha256(frozen),
        source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )
    calls: list[str] = []

    def forbidden_lstat(path: object) -> object:
        del path
        calls.append("lstat")
        raise AssertionError("malformed authority must stop before filesystem access")

    def forbidden_clock() -> int:
        calls.append("clock")
        raise AssertionError("malformed authority must stop before the clock")

    def forbidden_import(name: str, package: str | None = None) -> ModuleType:
        del name, package
        calls.append("native")
        raise AssertionError("malformed authority must stop before native import")

    monkeypatch.setattr(persistence.os, "lstat", forbidden_lstat)
    monkeypatch.setattr(transaction.importlib, "import_module", forbidden_import)

    with pytest.raises(
        transaction.CalibrationV3TransactionStop,
        match="artifact contract",
    ):
        transaction.CalibrationV3Transaction._from_authority_for_test(
            authority,
            clock=forbidden_clock,
        )

    assert calls == []


def test_runnable_authority_requires_matching_native_circuit_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_module, transaction = _production_modules()
    missing = _runnable_protocol()
    del missing["screens"]["r3_f0_train"]["native_circuit_identity_sha256_by_arm_and_seed"]
    missing_authority = authority_module.FrozenPlanAuthority(
        repository_root=tmp_path,
        protocol=missing,
        protocol_sha256=_protocol_sha256(missing),
        source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )

    with pytest.raises(
        transaction.CalibrationV3TransactionStop,
        match=r"circuit.*identity|identity.*circuit",
    ):
        transaction.CalibrationV3Transaction._from_authority_for_test(
            missing_authority,
            clock=lambda: pytest.fail(
                "missing circuit authority must stop before the active clock"
            ),
        )

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    environment = _environment(
        mismatch_root,
        monkeypatch,
        live_circuit_identity_sha256="9" * 64,
    )
    with pytest.raises(
        transaction.CalibrationV3TransactionStop,
        match=r"circuit.*identity|identity.*circuit",
    ):
        session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(transaction),
        )

    assert environment.harness.checkpoint_attempts == 0
    assert all(circuit.closed for circuit in environment.harness.circuits)


def test_fresh_definition_factory_runs_after_active_clock_and_before_native_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)

    def definition_factory() -> _Definition:
        environment.harness.events.append("definition_factory")
        return _Definition()

    session = environment.coordinator.start_fresh(RESULT_PATH, definition_factory)

    _assert_durable_namespace_precedes_clock(environment.harness.events)
    assert environment.harness.events.index("clock") < environment.harness.events.index(
        "definition_factory"
    )
    assert environment.harness.events.index(
        "definition_factory"
    ) < environment.harness.events.index("fresh")
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        session.close()


def test_fresh_session_owns_clocked_circuit_and_commits_exact_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    progress = _progress(environment.transaction)

    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    _assert_durable_namespace_precedes_clock(environment.harness.events)
    assert environment.harness.events[55] == "fresh"
    session.check_cap()
    commit = _commit_with_progress(environment.coordinator, session, progress)
    assert not hasattr(session, "circuit")
    with pytest.raises(AttributeError):
        commit.timing = persistence.CumulativeActiveWallTiming.complete_fresh_session(0)
    with pytest.raises(AttributeError):
        commit._timing = persistence.CumulativeActiveWallTiming.complete_fresh_session(0)

    events = environment.harness.events
    publish_index = events.index("checkpoint:0")
    import_index = events.index("import")
    verifier_close_index = events.index("close:import:1")
    manifest_index = events.index("manifest_publish")
    assert events.index("identity:fresh:0") < publish_index
    assert publish_index < import_index < verifier_close_index
    assert verifier_close_index < max(
        index for index, event in enumerate(events) if event == "clock"
    )
    assert max(index for index, event in enumerate(events) if event == "clock") < (manifest_index)

    payload = _read_manifest(environment, tmp_path)
    assert set(payload["checkpoint"]) == set(IDENTITY_FIELDS)
    assert payload["checkpoint"] == _identity_payload(environment.identity)
    assert payload["checkpoint_publication"] == _publication_payload(
        environment.harness.checkpoint_publication
    )
    assert payload["circuit_contract"] == environment.frozen["circuit_contract"]
    assert payload["train_input"] == environment.frozen["screens"]["r3_f0_train"]["input"]
    assert set(payload["previous_state"]) == set(STEP_FIELDS)
    assert payload["previous_state"] == _step_payload(progress.last_step)
    assert payload["stream_prefixes"] == {
        "winner": {"count": 529_340, "sha256": "11" * 32},
        "step": {"count": 529_346, "sha256": "22" * 32},
    }
    assert payload["reference_trajectory_id"] == REFERENCE_TRAJECTORY_ID
    assert (
        payload["cursor"]
        == environment.frozen["screens"]["r3_f0_train"]["determinism"]["resume_milestones"][0]
    )
    assert payload["timing"] == {
        "prior_cumulative_ns": 0,
        "session_active_ns": 0,
        "cumulative_active_ns": 0,
    }

    continued = environment.coordinator.continue_after(commit)
    assert len(environment.harness.circuits) == 2
    assert environment.harness.circuits[0].closed is False
    assert environment.harness.circuits[1].closed is True
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        continued.close()
    assert environment.harness.circuits[0].closed is True


def test_session_builds_exact_progress_chains_from_native_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    frames = tuple(
        _step_at(token_id, event_count)
        for token_id, event_count in zip(
            (11, 12, 13, 14, 15),
            (1, 2, 3, 4, 1),
            strict=True,
        )
    )
    environment.harness.advance_results = [
        tuple(_native_step(environment.native, frame) for frame in frames[:2]),
        tuple(_native_step(environment.native, frame) for frame in frames[2:4]),
        (_native_step(environment.native, frames[4]),),
    ]

    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    assert (
        session.advance_block(
            (11, 12),
            mode=environment.native.TemporalStepMode.LEARN,
            reset_before=True,
        )
        == frames[:2]
    )
    assert (
        session.advance_block(
            (13, 14),
            mode=environment.native.TemporalStepMode.LEARN,
        )
        == frames[2:4]
    )
    assert (
        session.advance_block(
            (15,),
            mode=environment.native.TemporalStepMode.LEARN,
            reset_before=True,
        )
        == frames[4:]
    )

    expected_step = persistence.ResumableHashChain.start(persistence.HashChainKind.STEP)
    for frame in frames:
        expected_step = expected_step.append_step(frame)
    expected_winner = persistence.ResumableHashChain.start(persistence.HashChainKind.WINNER)
    for frame in frames[1:4]:
        expected_winner = expected_winner.append_winner(
            persistence.WinnerChainFrame(
                token_id=frame.token_id,
                event_count=frame.event_count,
                winner_context=frame.winner_context[:11],
            )
        )

    assert session.progress == environment.transaction.FrozenResumeProgress(
        last_step=frames[-1],
        winner_chain=expected_winner,
        step_chain=expected_step,
    )
    assert expected_step.count == 5
    assert expected_winner.count == 3
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        session.close()


def test_commit_rejects_caller_progress_without_consuming_the_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    progress = _progress(environment.transaction)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(TypeError, match="progress"):
        environment.coordinator.commit_resume_checkpoint(
            session,
            progress=progress,
        )

    assert environment.harness.checkpoint_attempts == 0
    assert environment.harness.circuits[0].closed is False
    commit = _commit_with_progress(environment.coordinator, session, progress)
    commit.close()


@pytest.mark.parametrize("invalid_event_count", (8, 10))
def test_progress_continuity_failure_preserves_last_complete_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_event_count: int,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    seeded = environment.transaction.FrozenResumeProgress(
        last_step=_step_at(7, 7),
        winner_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.WINNER,
            count=5,
            digest=b"\x51" * 32,
        ),
        step_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.STEP,
            count=7,
            digest=b"\x52" * 32,
        ),
    )
    _seed_progress(session, seeded)
    accepted = _step_at(8, 8)
    environment.harness.advance_results = [
        (_native_step(environment.native, accepted),),
        (
            _native_step(
                environment.native,
                _step_at(9, invalid_event_count),
            ),
        ),
    ]

    session.advance_block(
        (8,),
        mode=environment.native.TemporalStepMode.LEARN,
    )
    complete_prefix = session.progress
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="next stream position",
    ) as stopped:
        session.advance_block(
            (9,),
            mode=environment.native.TemporalStepMode.LEARN,
        )

    assert complete_prefix is not None
    assert environment.coordinator._progress is complete_prefix
    assert complete_prefix.step_chain.count == 8
    assert complete_prefix.winner_chain.count == 6
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )
    assert environment.harness.circuits[0].closed is True
    assert not _absolute(tmp_path, CHECKPOINT_PATH).exists()
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()


@pytest.mark.parametrize("failure", ("schema", "gap"))
def test_multistep_validation_is_atomic_before_progress_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    first = _native_step(environment.native, _step_at(21, 1))
    second = _native_step(
        environment.native,
        _step_at(22, 3 if failure == "gap" else 2),
    )
    if failure == "schema":
        second = replace(second, winner_context=(1,))
    environment.harness.advance_results = [(first, second)]
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        session.advance_block(
            (21, 22),
            mode=environment.native.TemporalStepMode.LEARN,
            reset_before=True,
        )

    assert environment.coordinator._progress is None
    assert environment.harness.circuits[0].closed is True
    assert not _absolute(tmp_path, CHECKPOINT_PATH).exists()
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()


def test_post_confirmation_capability_failure_retains_clean_timing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    def fail_capability_construction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic sealed capability construction failure")

    monkeypatch.setattr(
        environment.transaction,
        "CommittedResumeCheckpoint",
        fail_capability_construction,
    )

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="synthetic sealed capability construction failure",
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert stopped.value.timing_state is (environment.transaction.TransactionTimingState.WITHIN_CAP)
    assert stopped.value.timing is not None
    assert stopped.value.timing.state is persistence.TimingState.WITHIN_CAP
    assert stopped.value.checkpoint_publication is environment.harness.checkpoint_publication
    assert stopped.value.manifest_publication is not None
    assert _absolute(tmp_path, CHECKPOINT_PATH).is_file()
    assert _absolute(tmp_path, MANIFEST_PATH).is_file()
    assert environment.harness.circuits[0].closed is True


def test_forged_exact_commit_object_cannot_supply_authoritative_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    forged = object.__new__(environment.transaction.CommittedResumeCheckpoint)
    object.__setattr__(forged, "_owner", environment.coordinator)
    object.__setattr__(forged, "_token", commit._token)
    object.__setattr__(
        forged,
        "_timing",
        persistence.CumulativeActiveWallTiming.complete_fresh_session(
            persistence.ACTIVE_WALL_CAP_NS + 1
        ),
    )
    object.__setattr__(forged, "_checkpoint_publication", commit.checkpoint_publication)
    object.__setattr__(forged, "_manifest_publication", commit.manifest_publication)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"consumed|unavailable",
    ):
        environment.coordinator.continue_after(forged)

    continued = environment.coordinator.continue_after(commit)
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        continued.close()


def test_only_durable_checkpoint_outcomes_commit_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        ("durable_clean", True),
        ("durable_orphaned", True),
        ("collision", False),
        ("not_published", False),
        ("unconfirmed", False),
    )
    for mode, commits in cases:
        root = tmp_path / mode
        root.mkdir()
        environment = _environment(
            root,
            monkeypatch,
            publication_mode=mode,
        )
        session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
        if commits:
            commit = _commit_with_progress(
                environment.coordinator,
                session,
                _progress(environment.transaction),
            )
            payload = _read_manifest(environment, root)
            assert payload["checkpoint_publication"] == _publication_payload(
                environment.harness.checkpoint_publication
            )
            commit.close()
        else:
            with pytest.raises(
                environment.transaction.CalibrationV3TransactionStop,
                match=r"checkpoint|durable|published",
            ):
                _commit_with_progress(
                    environment.coordinator,
                    session,
                    _progress(environment.transaction),
                )
            assert not _absolute(root, MANIFEST_PATH).exists()
            assert all(circuit.closed for circuit in environment.harness.circuits)
        assert environment.harness.checkpoint_attempts == 1


def test_reachable_checkpoint_temp_identity_orphan_is_accepted_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    report = _checkpoint_publication(environment.native, "durable_orphaned")
    environment.harness.checkpoint_publication = replace(
        report,
        cleanup_errors=(
            replace(
                report.cleanup_errors[0],
                stage=environment.native.TemporalCheckpointErrorStage.TEMP_IDENTITY,
            ),
        ),
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    assert commit.checkpoint_publication is environment.harness.checkpoint_publication
    commit.close()

    resumed = _new_coordinator(environment).open_resume(MANIFEST_PATH, _Definition())
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        resumed.close()


@pytest.mark.parametrize(
    "defect",
    (
        "cleanup-list",
        "cleanup-capacity",
        "cleanup-duplicate",
        "cleanup-error-type",
        "cleanup-stage",
        "cleanup-error-code-width",
        "pid-bool",
        "pid-width",
        "nonce-width",
    ),
)
def test_checkpoint_publication_reports_must_match_the_exact_native_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    report = _checkpoint_publication(environment.native, "durable_clean")
    cleanup_error = environment.native.TemporalCheckpointPublicationError(
        stage=environment.native.TemporalCheckpointErrorStage.TEMP_CLEANUP,
        domain=environment.native.TemporalCheckpointErrorDomain.POLICY,
        code=91,
    )
    if defect == "cleanup-list":
        report = replace(report, cleanup_errors=[])
    elif defect == "cleanup-capacity":
        report = replace(
            report,
            temp_cleanup=environment.native.TemporalCheckpointCleanup.ORPHANED,
            cleanup_errors=(cleanup_error,) * 5,
        )
    elif defect == "cleanup-duplicate":
        report = replace(
            report,
            temp_cleanup=environment.native.TemporalCheckpointCleanup.ORPHANED,
            cleanup_errors=(cleanup_error,) * 2,
        )
    elif defect == "cleanup-error-type":
        report = replace(
            report,
            temp_cleanup=environment.native.TemporalCheckpointCleanup.ORPHANED,
            cleanup_errors=(object(),),
        )
    elif defect == "cleanup-stage":
        report = replace(
            report,
            temp_cleanup=environment.native.TemporalCheckpointCleanup.ORPHANED,
            cleanup_errors=(
                replace(
                    cleanup_error,
                    stage=environment.native.TemporalCheckpointErrorStage.FILE_CLOSE,
                ),
            ),
        )
    elif defect == "cleanup-error-code-width":
        report = replace(
            report,
            temp_cleanup=environment.native.TemporalCheckpointCleanup.ORPHANED,
            cleanup_errors=(replace(cleanup_error, code=1 << 32),),
        )
    elif defect == "pid-bool":
        report = replace(report, owned_temp_pid=True)
    elif defect == "pid-width":
        report = replace(report, owned_temp_pid=1 << 32)
    elif defect == "nonce-width":
        report = replace(report, owned_temp_nonce=1 << 64)
    environment.harness.checkpoint_publication = report
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(environment.transaction.CalibrationV3TransactionStop) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert stopped.value.checkpoint_publication is report
    assert environment.harness.checkpoint_attempts == 1
    assert environment.harness.manifest_attempts == 0
    assert environment.harness.circuits[0].closed is True


def test_only_durable_manifest_outcomes_commit_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        ("durable_clean", True),
        ("durable_orphaned", True),
        ("collision", False),
        ("not_published", False),
        ("unconfirmed", False),
    )
    for mode, commits in cases:
        root = tmp_path / mode
        root.mkdir()
        environment = _environment(root, monkeypatch)
        reports: list[object] = []

        def publish_manifest(
            destination: Path,
            payload: bytes,
            *,
            mode: str = mode,
            environment: _Environment = environment,
            reports: list[object] = reports,
        ) -> object:
            report = _manifest_publication(environment.native, mode, len(payload))
            reports.append(report)
            if mode in {"durable_clean", "durable_orphaned", "unconfirmed"}:
                destination.write_bytes(payload)
            return report

        monkeypatch.setattr(environment.native, "publish_immutable_bytes", publish_manifest)
        session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
        if commits:
            commit = _commit_with_progress(
                environment.coordinator,
                session,
                _progress(environment.transaction),
            )
            assert commit.manifest_publication is reports[0]
            commit.close()
        else:
            with pytest.raises(
                environment.transaction.CalibrationV3TransactionStop,
                match=r"manifest|durable|published",
            ) as stopped:
                _commit_with_progress(
                    environment.coordinator,
                    session,
                    _progress(environment.transaction),
                )
            assert stopped.value.manifest_publication is reports[0]
            assert all(circuit.closed for circuit in environment.harness.circuits)
        assert len(reports) == 1


def test_reachable_manifest_temp_identity_orphan_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    reports: list[object] = []

    def publish_manifest(destination: Path, payload: bytes) -> object:
        report = _manifest_publication(environment.native, "durable_orphaned", len(payload))
        report = replace(
            report,
            cleanup_errors=(
                replace(
                    report.cleanup_errors[0],
                    stage=environment.native.TemporalPublicationErrorStage.TEMP_IDENTITY,
                ),
            ),
        )
        reports.append(report)
        destination.write_bytes(payload)
        return report

    monkeypatch.setattr(environment.native, "publish_immutable_bytes", publish_manifest)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    assert commit.manifest_publication is reports[0]
    commit.close()


@pytest.mark.parametrize(
    "defect",
    (
        "cleanup-list",
        "cleanup-capacity",
        "cleanup-duplicate",
        "cleanup-error-type",
        "cleanup-stage",
        "cleanup-error-domain",
        "cleanup-error-code-width",
        "pid-bool",
        "pid-width",
        "nonce-width",
        "payload-bool",
    ),
)
def test_manifest_publication_reports_must_match_the_exact_native_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    reports: list[object] = []

    def publish_manifest(destination: Path, payload: bytes) -> object:
        report = _manifest_publication(environment.native, "durable_clean", len(payload))
        cleanup_error = environment.native.TemporalPublicationError(
            stage=environment.native.TemporalPublicationErrorStage.TEMP_CLEANUP,
            domain=environment.native.TemporalPublicationErrorDomain.POLICY,
            code=91,
        )
        if defect == "cleanup-list":
            report = replace(report, cleanup_errors=[])
        elif defect == "cleanup-capacity":
            report = replace(
                report,
                temp_cleanup=environment.native.TemporalPublicationCleanup.ORPHANED,
                cleanup_errors=(cleanup_error,) * 5,
            )
        elif defect == "cleanup-duplicate":
            report = replace(
                report,
                temp_cleanup=environment.native.TemporalPublicationCleanup.ORPHANED,
                cleanup_errors=(cleanup_error,) * 2,
            )
        elif defect == "cleanup-error-type":
            report = replace(
                report,
                temp_cleanup=environment.native.TemporalPublicationCleanup.ORPHANED,
                cleanup_errors=(object(),),
            )
        elif defect == "cleanup-stage":
            report = replace(
                report,
                temp_cleanup=environment.native.TemporalPublicationCleanup.ORPHANED,
                cleanup_errors=(
                    replace(
                        cleanup_error,
                        stage=environment.native.TemporalPublicationErrorStage.FILE_CLOSE,
                    ),
                ),
            )
        elif defect == "cleanup-error-domain":
            report = replace(
                report,
                temp_cleanup=environment.native.TemporalPublicationCleanup.ORPHANED,
                cleanup_errors=(
                    replace(
                        cleanup_error,
                        domain=(
                            environment.native.TemporalPublicationErrorDomain.CHECKPOINT_ENCODER_RESERVED
                        ),
                    ),
                ),
            )
        elif defect == "cleanup-error-code-width":
            report = replace(
                report,
                temp_cleanup=environment.native.TemporalPublicationCleanup.ORPHANED,
                cleanup_errors=(replace(cleanup_error, code=1 << 32),),
            )
        elif defect == "pid-bool":
            report = replace(report, owned_temp_pid=True)
        elif defect == "pid-width":
            report = replace(report, owned_temp_pid=1 << 32)
        elif defect == "nonce-width":
            report = replace(report, owned_temp_nonce=1 << 64)
        elif defect == "payload-bool":
            report = replace(report, payload_bytes=True)
        reports.append(report)
        destination.write_bytes(payload)
        return report

    monkeypatch.setattr(environment.native, "publish_immutable_bytes", publish_manifest)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(environment.transaction.CalibrationV3TransactionStop) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert stopped.value.manifest_publication is reports[0]
    assert environment.harness.checkpoint_attempts == 1
    assert len(reports) == 1
    assert environment.harness.circuits[0].closed is True


def test_checkpoint_publication_bytes_must_equal_the_exact_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    environment.harness.checkpoint_publication = replace(
        environment.harness.checkpoint_publication,
        checkpoint_bytes=8_192,
        encoded_bytes=8_192,
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="byte counts differ from checkpoint identity",
    ):
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert environment.harness.checkpoint_attempts == 1
    assert environment.harness.manifest_attempts == 0
    assert environment.harness.circuits[0].closed is True


def test_checkpoint_import_must_match_every_one_of_the_18_identity_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(IDENTITY_FIELDS) == 18
    for field in IDENTITY_FIELDS:
        root = tmp_path / field
        root.mkdir()
        environment = _environment(
            root,
            monkeypatch,
            imported_identity_mutator=lambda identity, field=field: _different_identity(
                identity, field
            ),
        )
        session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

        with pytest.raises(
            environment.transaction.CalibrationV3TransactionStop,
            match=r"identity|parity",
        ):
            _commit_with_progress(
                environment.coordinator,
                session,
                _progress(environment.transaction),
            )

        assert environment.harness.checkpoint_attempts == 1
        assert not _absolute(root, MANIFEST_PATH).exists()
        assert all(circuit.closed for circuit in environment.harness.circuits)


def test_checkpoint_replacement_during_native_import_breaks_the_final_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def replace_checkpoint(path: Path) -> None:
        replacement = path.with_name(f"{path.name}.replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)

    environment = _environment(
        tmp_path,
        monkeypatch,
        during_import=replace_checkpoint,
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="checkpoint final identity changed",
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )
    assert stopped.value.checkpoint_publication is environment.harness.checkpoint_publication
    assert stopped.value.manifest_publication is None
    assert environment.harness.manifest_attempts == 0
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()
    assert environment.harness.circuits[0].closed is True


def test_open_resume_rejects_checkpoint_replacement_during_native_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    commit.close()

    def replace_checkpoint(path: Path) -> None:
        replacement = path.with_name(f"{path.name}.resume-replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)

    environment.harness.during_import = replace_checkpoint
    resumed = _new_coordinator(environment)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="checkpoint final identity changed",
    ):
        resumed.open_resume(MANIFEST_PATH, _Definition())

    assert all(circuit.closed for circuit in environment.harness.circuits)


def test_open_resume_rejects_manifest_replacement_during_native_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    commit.close()
    manifest = _absolute(tmp_path, MANIFEST_PATH)

    def replace_manifest(_checkpoint: Path) -> None:
        replacement = manifest.with_name(f"{manifest.name}.resume-replacement")
        replacement.write_bytes(manifest.read_bytes())
        replacement.replace(manifest)

    environment.harness.during_import = replace_manifest
    resumed = _new_coordinator(environment)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="manifest final identity changed",
    ):
        resumed.open_resume(MANIFEST_PATH, _Definition())

    assert all(circuit.closed for circuit in environment.harness.circuits)


def test_cap_crossing_commits_manifest_last_then_raises_terminal_over_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock = _Clock(events, now_ns=100)

    def cross_cap() -> None:
        clock.now_ns = 100 + persistence.ACTIVE_WALL_CAP_NS + 1

    environment = _environment(
        tmp_path,
        monkeypatch,
        clock=clock,
        after_checkpoint=cross_cap,
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    session.check_cap()
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="cap",
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert _read_manifest(environment, tmp_path)["timing"] == {
        "prior_cumulative_ns": 0,
        "session_active_ns": persistence.ACTIVE_WALL_CAP_NS + 1,
        "cumulative_active_ns": persistence.ACTIVE_WALL_CAP_NS + 1,
    }
    assert environment.harness.manifest_attempts == 1
    assert stopped.value.timing_state is environment.transaction.TransactionTimingState.OVER_CAP
    assert stopped.value.timing is not None
    assert stopped.value.checkpoint_publication is environment.harness.checkpoint_publication
    assert stopped.value.manifest_publication is not None
    assert environment.harness.circuits[0].closed is True


def test_resume_clocks_before_manifest_and_rejects_tamper_before_native_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    commit.close()
    manifest = _absolute(tmp_path, MANIFEST_PATH)
    environment.harness.events.clear()

    def corrupt_manifest() -> None:
        manifest.write_bytes(manifest.read_bytes() + b"\n")

    environment.clock.on_next = corrupt_manifest
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"manifest|resume|canonical|envelope",
    ):
        _new_coordinator(environment).open_resume(MANIFEST_PATH, _Definition())

    _assert_durable_namespace_precedes_clock(environment.harness.events)
    assert "import" not in environment.harness.events


def test_session_and_commit_capabilities_are_transaction_bound_and_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    second = environment.transaction.CalibrationV3Transaction._from_authority_for_test(
        environment.authority,
        clock=environment.clock,
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    progress = _progress(environment.transaction)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"transaction|owner|bound",
    ):
        _commit_with_progress(second, session, progress)
    assert environment.harness.checkpoint_attempts == 0

    commit = _commit_with_progress(environment.coordinator, session, progress)
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"consumed|active|session",
    ):
        environment.coordinator.commit_resume_checkpoint(session)
    assert environment.harness.checkpoint_attempts == 1

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"transaction|owner|bound",
    ):
        second.continue_after(commit)
    continued = environment.coordinator.continue_after(commit)
    assert len(environment.harness.circuits) == 2
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"consumed|commit",
    ):
        environment.coordinator.continue_after(commit)
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
    ) as stopped:
        continued.close()
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )


def test_failed_continuation_clock_leaves_the_sealed_commit_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )

    def fail_clock() -> None:
        raise RuntimeError("synthetic continuation clock failure")

    environment.clock.on_next = fail_clock
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="clock",
    ) as stopped:
        environment.coordinator.continue_after(commit)

    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.NOT_STARTED
    )
    assert environment.harness.circuits[0].closed is False

    environment.clock.now_ns = 111
    continued = environment.coordinator.continue_after(commit)
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        continued.close()
    assert environment.harness.circuits[0].closed is True


def test_continuation_clock_cannot_regress_before_starting_the_next_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    environment.clock.now_ns = 111
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )

    environment.clock.now_ns = 110
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="backward",
    ):
        environment.coordinator.continue_after(commit)
    assert environment.harness.circuits[0].closed is False

    environment.clock.now_ns = 112
    continued = environment.coordinator.continue_after(commit)
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        continued.close()


def test_prepublication_cap_crossing_writes_nothing_and_closes_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock = _Clock(events, now_ns=100)
    environment = _environment(tmp_path, monkeypatch, clock=clock)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    clock.now_ns = 100 + persistence.ACTIVE_WALL_CAP_NS + 1

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="cap",
    ):
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert environment.harness.checkpoint_attempts == 0
    assert not _absolute(tmp_path, CHECKPOINT_PATH).exists()
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()
    assert environment.harness.circuits[0].closed is True


def test_invalid_typed_progress_stops_before_checkpoint_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    valid = _progress(environment.transaction)
    invalid = environment.transaction.FrozenResumeProgress(
        last_step=valid.last_step,
        winner_chain=replace(valid.winner_chain, count=valid.winner_chain.count + 1),
        step_chain=valid.step_chain,
    )

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"winner|progress|cursor",
    ):
        _commit_with_progress(environment.coordinator, session, invalid)

    assert environment.harness.checkpoint_attempts == 0
    assert not _absolute(tmp_path, CHECKPOINT_PATH).exists()
    assert environment.harness.circuits[0].closed is True


def test_manifest_nonpublication_never_retries_or_adopts_checkpoint_only_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    attempts = 0

    def reject_manifest(destination: Path, payload: bytes) -> object:
        nonlocal attempts
        attempts += 1
        assert not destination.exists()
        return environment.native.TemporalImmutableBytesPublication(
            outcome=environment.native.TemporalPublicationOutcome.NOT_PUBLISHED,
            temp_cleanup=environment.native.TemporalPublicationCleanup.NOT_APPLICABLE,
            primary_error=environment.native.TemporalPublicationError(
                stage=environment.native.TemporalPublicationErrorStage.PARENT_OPEN,
                domain=environment.native.TemporalPublicationErrorDomain.POLICY,
                code=7,
            ),
            cleanup_errors=(),
            owned_temp_pid=0,
            owned_temp_nonce=0,
            payload_bytes=len(payload),
            written_bytes=0,
        )

    monkeypatch.setattr(environment.native, "publish_immutable_bytes", reject_manifest)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"manifest|durable|published",
    ):
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert attempts == 1
    assert _absolute(tmp_path, CHECKPOINT_PATH).is_file()
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()
    assert all(circuit.closed for circuit in environment.harness.circuits)


def test_checkpoint_deletion_during_manifest_publication_never_commits_a_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    reports: list[object] = []

    def publish_then_delete_checkpoint(destination: Path, payload: bytes) -> object:
        destination.write_bytes(payload)
        _absolute(tmp_path, CHECKPOINT_PATH).unlink()
        report = _manifest_publication(
            environment.native,
            "durable_clean",
            len(payload),
        )
        reports.append(report)
        return report

    monkeypatch.setattr(
        environment.native,
        "publish_immutable_bytes",
        publish_then_delete_checkpoint,
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"MANIFEST_ONLY|manifest|checkpoint",
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )
    assert stopped.value.manifest_publication is reports[0]
    assert _absolute(tmp_path, MANIFEST_PATH).is_file()
    assert not _absolute(tmp_path, CHECKPOINT_PATH).exists()
    assert environment.harness.circuits[0].closed is True


def test_successful_resume_restores_exact_progress_and_active_abort_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    progress = _progress(environment.transaction)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(environment.coordinator, session, progress)
    commit.close()
    environment.harness.events.clear()

    resumed = _new_coordinator(environment).open_resume(MANIFEST_PATH, _Definition())

    _assert_durable_namespace_precedes_clock(environment.harness.events)
    assert "import" in environment.harness.events
    assert resumed.progress == progress
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop) as stopped:
        resumed.close()
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )


def test_resumed_branch_finishes_at_exact_train_boundary_without_result_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    commit.close()

    coordinator = _new_coordinator(environment)
    resumed = coordinator.open_resume(MANIFEST_PATH, _Definition())
    train_input = environment.frozen["screens"]["r3_f0_train"]["input"]
    terminal_step = replace(_last_step(), event_count=7)
    terminal_progress = environment.transaction.FrozenResumeProgress(
        last_step=terminal_step,
        winner_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.WINNER,
            count=train_input["transition_events"],
            digest=b"\x77" * 32,
        ),
        step_chain=persistence.ResumableHashChain(
            kind=persistence.HashChainKind.STEP,
            count=train_input["token_count"],
            digest=b"\x88" * 32,
        ),
    )
    _seed_progress(resumed, terminal_progress)
    terminal_identity = replace(
        environment.identity,
        event_count=terminal_step.event_count,
        learning_step=train_input["token_count"],
        prior_context_length=len(terminal_step.context),
        winner_context_length=len(terminal_step.winner_context),
    )
    terminal_identity = replace(
        terminal_identity,
        logical_state_sha256=_logical_state_sha256(terminal_identity),
    )
    environment.harness.set_active_identity(terminal_identity)
    environment.clock.now_ns += 19
    reconciliation = resumed.manifest_reconciliation

    receipt = coordinator.finish_resumed_branch(resumed)

    assert receipt.progress == terminal_progress
    assert dict(receipt.checkpoint_identity) == _identity_payload(terminal_identity)
    assert receipt.timing.state is persistence.TimingState.WITHIN_CAP
    assert receipt.manifest_reconciliation is reconciliation
    assert environment.harness.circuits[-1].closed is True
    assert not _absolute(tmp_path, RESULT_PATH).exists()


@pytest.mark.parametrize("recovery_mode", ["continue", "reopen"])
def test_frozen_resume_milestones_advance_in_order_after_each_recovery_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_mode: str,
) -> None:
    root = tmp_path / recovery_mode
    root.mkdir()
    environment = _environment(root, monkeypatch)
    first_session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    environment.clock.now_ns = 111
    first_commit = _commit_with_progress(
        environment.coordinator,
        first_session,
        _progress(environment.transaction),
    )
    first_cumulative_ns = first_commit.timing.cumulative_active_ns
    assert first_cumulative_ns == 11

    if recovery_mode == "continue":
        coordinator = environment.coordinator
        recovered = environment.coordinator.continue_after(first_commit)
    else:
        first_commit.close()
        coordinator = environment.transaction.CalibrationV3Transaction._from_authority_for_test(
            environment.authority,
            clock=environment.clock,
        )
        recovered = coordinator.open_resume(MANIFEST_PATH, _Definition())

    environment.clock.now_ns = 128

    second_cursor = environment.frozen["screens"]["r3_f0_train"]["determinism"][
        "resume_milestones"
    ][1]
    second_progress = _progress_for_cursor(environment.transaction, second_cursor)
    second_identity = replace(
        environment.identity,
        event_count=second_cursor["last_token_index_zero_based"] + 1,
        learning_step=second_cursor["learning_step"],
        prior_context_length=len(second_progress.last_step.context),
        winner_context_length=len(second_progress.last_step.winner_context),
    )
    second_identity = replace(
        second_identity,
        logical_state_sha256=_logical_state_sha256(second_identity),
    )
    environment.harness.set_active_identity(second_identity)
    environment.harness.imported_identity = second_identity

    second_commit = _commit_with_progress(coordinator, recovered, second_progress)

    assert environment.harness.checkpoint_attempts == 2
    assert environment.harness.manifest_attempts == 2
    assert second_commit.timing.prior_cumulative_ns == 11
    assert second_commit.timing.session_active_ns == 17
    assert second_commit.timing.cumulative_active_ns == 28
    assert _absolute(root, SECOND_MANIFEST_PATH).is_file()
    second_commit.close()


def test_resume_rejects_non_durable_recorded_publication_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    commit.close()
    store = persistence.CalibrationV3ArtifactStore.open(
        repository_root=tmp_path,
        validated_protocol=environment.frozen,
        expected_source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )
    payload = store.read(MANIFEST_PATH).payload
    payload["checkpoint_publication"] = {
        "outcome": "NOT_PUBLISHED",
        "temp_cleanup": "NOT_APPLICABLE",
        "primary_error": {"stage": "PARENT_OPEN", "domain": "POLICY", "code": 7},
        "cleanup_errors": [],
        "owned_temp_pid": 0,
        "owned_temp_nonce": 0,
        "checkpoint_bytes": 4_096,
        "encoded_bytes": 0,
    }
    _absolute(tmp_path, MANIFEST_PATH).write_bytes(store.encode_resume(payload))
    environment.harness.events.clear()

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"checkpoint|durable|published",
    ):
        _new_coordinator(environment).open_resume(MANIFEST_PATH, _Definition())

    _assert_durable_namespace_precedes_clock(environment.harness.events)
    assert "import" not in environment.harness.events


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("format_version", 2),
        ("header_bytes", 257),
        ("checkpoint_bytes", 1 << 63),
        ("segment_count", 1 << 32),
        ("event_count", 1 << 64),
        ("prior_context_length", 23),
        ("logical_state_sha256", "0" * 64),
    ),
)
def test_resume_rejects_checkpoint_identity_outside_the_native_abi_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    commit = _commit_with_progress(
        environment.coordinator,
        session,
        _progress(environment.transaction),
    )
    commit.close()
    store = persistence.CalibrationV3ArtifactStore.open(
        repository_root=tmp_path,
        validated_protocol=environment.frozen,
        expected_source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )
    payload = store.read(MANIFEST_PATH).payload
    payload["checkpoint"][field] = value
    _absolute(tmp_path, MANIFEST_PATH).write_bytes(store.encode_resume(payload))
    environment.harness.events.clear()

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"checkpoint|identity|header|format",
    ):
        _new_coordinator(environment).open_resume(MANIFEST_PATH, _Definition())

    _assert_durable_namespace_precedes_clock(environment.harness.events)
    assert "import" not in environment.harness.events


def test_backward_or_failing_clock_invalidates_and_closes_active_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    environment.clock.now_ns = 99
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="backward",
    ):
        session.check_cap()
    assert environment.harness.circuits[0].closed is True

    second_root = tmp_path / "clock-failure"
    second_root.mkdir()
    second = _environment(second_root, monkeypatch)
    second_session = second.coordinator.start_fresh(RESULT_PATH, _Definition)

    def fail_clock() -> None:
        raise RuntimeError("synthetic clock failure")

    second.clock.on_next = fail_clock
    with pytest.raises(
        second.transaction.CalibrationV3TransactionStop,
        match="clock",
    ):
        second_session.check_cap()
    assert second.harness.circuits[0].closed is True


def test_clock_must_not_regress_after_an_over_cap_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock = _Clock(
        events,
        scripted_ns=[
            100,
            100,
            100,
            100,
            100,
            100 + persistence.ACTIVE_WALL_CAP_NS + 1,
            101,
        ],
    )
    environment = _environment(tmp_path, monkeypatch, clock=clock)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="backward",
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert stopped.value.timing_state is (environment.transaction.TransactionTimingState.OVER_CAP)
    assert stopped.value.timing is not None
    assert stopped.value.timing.state is persistence.TimingState.OVER_CAP
    assert stopped.value.checkpoint_publication is environment.harness.checkpoint_publication
    assert _absolute(tmp_path, CHECKPOINT_PATH).is_file()
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()
    assert environment.harness.circuits[0].closed is True


def test_over_cap_observation_survives_later_checkpoint_import_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock([], now_ns=100)

    def cross_cap() -> None:
        clock.now_ns = 100 + persistence.ACTIVE_WALL_CAP_NS + 1

    environment = _environment(
        tmp_path,
        monkeypatch,
        clock=clock,
        after_checkpoint=cross_cap,
        imported_identity_mutator=lambda identity: replace(
            identity,
            event_count=identity.event_count + 1,
        ),
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match=r"identity|parity",
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert stopped.value.timing_state is (environment.transaction.TransactionTimingState.OVER_CAP)
    assert stopped.value.timing is not None
    assert stopped.value.timing.state is persistence.TimingState.OVER_CAP
    assert _absolute(tmp_path, CHECKPOINT_PATH).is_file()
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
    ) as repeated:
        session.check_cap()
    assert repeated.value.timing_state is (environment.transaction.TransactionTimingState.OVER_CAP)


def test_native_runtime_failure_is_not_started_and_leaves_coordinator_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    original_import_module = environment.transaction.importlib.import_module

    def fail_native_import(name: str, package: str | None = None) -> ModuleType:
        if name == "humanbrain.native_temporal_runtime":
            raise RuntimeError("synthetic native loader failure")
        return original_import_module(name, package)

    monkeypatch.setattr(
        environment.transaction.importlib,
        "import_module",
        fail_native_import,
    )
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="native temporal runtime is unavailable",
    ) as stopped:
        environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.NOT_STARTED
    )
    assert environment.harness.events == []
    assert not _absolute(tmp_path, RESULT_PATH).parent.exists()

    monkeypatch.setattr(
        environment.transaction.importlib,
        "import_module",
        original_import_module,
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        session.close()


def test_post_clock_failure_is_irreversible_and_restart_has_no_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)

    def fail_fresh(cls: type[object], definition: object) -> object:
        del cls, definition
        environment.harness.events.append("fresh_failure")
        raise RuntimeError("synthetic construction failure")

    monkeypatch.setattr(
        environment.native.NativeTemporalCircuit,
        "fresh",
        classmethod(fail_fresh),
    )
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="synthetic construction failure",
    ) as stopped:
        environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )

    observed_events = list(environment.harness.events)
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="no longer ready",
    ) as repeated:
        environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    assert repeated.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )
    assert environment.harness.events == observed_events


def test_terminal_stop_receipt_cannot_be_mutated_through_the_caught_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
    ) as caught:
        session.close()
    stopped = caught.value

    with pytest.raises(AttributeError, match="immutable"):
        stopped._receipt = object()
    with pytest.raises(AttributeError, match="immutable"):
        stopped._timing_state = environment.transaction.TransactionTimingState.OVER_CAP

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
    ) as repeated:
        session.check_cap()
    assert repeated.value is not stopped
    assert repeated.value.timing_state is stopped.timing_state
    assert repeated.value.timing == stopped.timing
    assert repeated.value.cleanup_failures == stopped.cleanup_failures


def test_caller_thrown_transaction_stop_cannot_forge_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    forged = environment.transaction.CalibrationV3TransactionStop(
        "caller-forged receipt",
        timing_state=environment.transaction.TransactionTimingState.OVER_CAP,
        timing=persistence.CumulativeActiveWallTiming.complete_fresh_session(
            persistence.ACTIVE_WALL_CAP_NS + 1
        ),
        checkpoint_publication=object(),
        manifest_publication=object(),
        cleanup_failures=(
            environment.transaction.CalibrationV3CleanupFailure(
                stage="forged",
                error_type="forged",
            ),
        ),
    )

    class RaisingTokens:
        def __iter__(self) -> object:
            raise forged

    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="caller-forged receipt",
    ) as stopped:
        session.advance_block(
            RaisingTokens(),
            mode=environment.native.TemporalStepMode.LEARN,
        )

    assert stopped.value is not forged
    assert stopped.value.__cause__ is forged
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )
    assert stopped.value.timing is None
    assert stopped.value.checkpoint_publication is None
    assert stopped.value.manifest_publication is None
    assert stopped.value.cleanup_failures == ()
    assert environment.harness.circuits[0].closed is True
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop) as repeated:
        session.check_cap()
    assert repeated.value.timing_state is stopped.value.timing_state
    assert repeated.value.timing is None
    assert repeated.value.checkpoint_publication is None
    assert repeated.value.manifest_publication is None
    assert repeated.value.cleanup_failures == ()


def test_hostile_exception_string_is_never_called_during_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)

    class HostileError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt("hostile __str__ ran")

    primary = HostileError("safe exception argument")

    class RaisingTokens:
        def __iter__(self) -> object:
            raise primary

    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="safe exception argument",
    ) as stopped:
        session.advance_block(
            RaisingTokens(),
            mode=environment.native.TemporalStepMode.LEARN,
        )

    assert stopped.value.__cause__ is primary
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )
    assert environment.harness.circuits[0].closed is True
    with pytest.raises(environment.transaction.CalibrationV3TransactionStop) as repeated:
        session.check_cap()
    assert repeated.value.timing_state is stopped.value.timing_state
    assert repeated.value.timing is None


def test_keyboard_interrupt_after_clock_cleans_up_then_preserves_the_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    environment.harness.advance_error = KeyboardInterrupt("synthetic interrupt")
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        session.advance_block((), mode="training")

    assert environment.harness.circuits[0].closed is True
    observed_events = list(environment.harness.events)
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
    ) as stopped:
        session.check_cap()
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )
    assert environment.harness.events == observed_events


def test_keyboard_interrupt_from_cleanup_never_masks_the_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    environment.harness.advance_error = RuntimeError("synthetic advance failure")
    environment.harness.close_errors["fresh"] = KeyboardInterrupt("synthetic close interrupt")
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="synthetic advance failure",
    ) as stopped:
        session.advance_block((), mode="training")

    assert stopped.value.cleanup_failures == (
        environment.transaction.CalibrationV3CleanupFailure(
            stage="original_circuit_close",
            error_type="KeyboardInterrupt",
        ),
    )
    assert environment.harness.close_attempts == [("fresh", 0)]


def test_keyboard_interrupt_during_checkpoint_publication_terminalizes_the_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt_after_checkpoint() -> None:
        raise KeyboardInterrupt("synthetic checkpoint interrupt")

    environment = _environment(
        tmp_path,
        monkeypatch,
        after_checkpoint=interrupt_after_checkpoint,
    )
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(KeyboardInterrupt, match="synthetic checkpoint interrupt"):
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert _absolute(tmp_path, CHECKPOINT_PATH).is_file()
    assert not _absolute(tmp_path, MANIFEST_PATH).exists()
    assert environment.harness.circuits[0].closed is True
    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )


def test_verifier_and_original_close_failures_are_both_preserved_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    environment.harness.close_error_kinds = {"fresh", "import"}
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)

    with pytest.raises(
        environment.transaction.CalibrationV3TransactionStop,
        match="checkpoint verifier close failed",
    ) as stopped:
        _commit_with_progress(
            environment.coordinator,
            session,
            _progress(environment.transaction),
        )

    assert environment.harness.close_attempts == [("import", 1), ("fresh", 0)]
    assert stopped.value.cleanup_failures == (
        environment.transaction.CalibrationV3CleanupFailure(
            stage="verifier_close",
            error_type="RuntimeError",
        ),
        environment.transaction.CalibrationV3CleanupFailure(
            stage="original_circuit_close",
            error_type="RuntimeError",
        ),
    )
    assert stopped.value.timing_state is (
        environment.transaction.TransactionTimingState.TIMING_INDETERMINATE
    )


def test_terminal_result_closes_and_stops_clock_before_one_shot_durable_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    session = environment.coordinator.start_fresh(RESULT_PATH, _Definition)
    _seed_progress(session, _progress(environment.transaction))

    receipt = environment.coordinator.commit_r3_result(session, _r3_result())

    events = environment.harness.events
    close_index = events.index("close:fresh:0")
    final_clock_index = next(
        index for index in range(close_index + 1, len(events)) if events[index] == "clock"
    )
    result_publish_index = events.index("result_publish")
    assert close_index < final_clock_index < result_publish_index
    assert environment.harness.result_attempts == 1
    assert environment.harness.manifest_attempts == 0
    assert receipt.artifact_path == RESULT_PATH
    assert receipt.timing.state is persistence.TimingState.WITHIN_CAP
    assert receipt.publication.outcome is (
        environment.native.TemporalPublicationOutcome.DURABLE_PUBLISHED
    )

    store = persistence.CalibrationV3ArtifactStore.open(
        repository_root=tmp_path,
        validated_protocol=environment.frozen,
        expected_source_bundle_identity_sha256=SOURCE_BUNDLE_SHA256,
    )
    envelope = store.read(RESULT_PATH)
    expected_timing = {
        "contract_id": receipt.timing.contract_id,
        "cap_ns": receipt.timing.cap_ns,
        "prior_cumulative_ns": receipt.timing.prior_cumulative_ns,
        "session_active_ns": receipt.timing.session_active_ns,
        "cumulative_active_ns": receipt.timing.cumulative_active_ns,
        "state": receipt.timing.state.value,
    }
    expected_result = _r3_result()
    expected_result["structural_metrics"]["active_wall_timing"] = expected_timing
    assert envelope.payload["result"] == expected_result
    assert envelope.payload["result"]["structural_metrics"]["active_wall_timing"] == (
        expected_timing
    )
    assert envelope.payload_sha256 == receipt.payload_sha256
    assert environment.harness.circuits[0].closed is True

    with pytest.raises(environment.transaction.CalibrationV3TransactionStop):
        environment.coordinator.commit_r3_result(session, _r3_result())
    assert environment.harness.result_attempts == 1
