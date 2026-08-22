from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from humanbrain.simple_swe.artifacts import (
    ArtifactEvent,
    ArtifactLedger,
    atomic_create_bytes,
)
from humanbrain.simple_swe.freeze import (
    FreezeIntegrityError,
    FreezeManifest,
    verify_freeze_manifest,
)
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.tasks import (
    GeneratedTask,
    HiddenCase,
    PrivateTask,
    PublicTask,
    _generate_partition,
    canonical_json_bytes,
    canonical_sha256,
)
from humanbrain.typed_program import TypedSignature

BUNDLE_SCHEMA = "humanbrain.simple-swe-private-eval-bundle.v3"
SEED_SCHEMA = "humanbrain.simple-swe-private-seed-reservation.v1"


@dataclass(frozen=True)
class EvaluationPaths:
    run_root: Path
    ledger_root: Path
    seed_reservation_path: Path
    bundle_path: Path


@dataclass(frozen=True)
class EvaluationBundle:
    protocol_id: str
    protocol_sha256: str
    freeze_manifest_sha256: str
    checkpoint_sha256: str
    checkpoint_state_sha256: str
    source_head: str
    eval_seed_sha256: str
    eval_seed: bytes = field(repr=False)
    tasks: tuple[GeneratedTask, ...]
    bundle_sha256: str
    path: Path


@dataclass(frozen=True)
class EvaluationSeedReservation:
    paths: EvaluationPaths
    eval_seed_sha256: str
    run_start_event_sha256: str
    seed_event_sha256: str
    anchor_payload: dict[str, str]
    anchor_payload_sha256: str


@dataclass(frozen=True)
class EvaluationReveal:
    bundle_path: Path
    ledger_root: Path
    bundle_sha256: str
    eval_seed_sha256: str
    task_count: int
    run_start_event_sha256: str
    seed_event_sha256: str
    external_anchor_event_sha256: str
    reveal_event_sha256: str


@dataclass(frozen=True)
class _PrivateSeedReservation:
    seed: bytes = field(repr=False)
    seed_sha256: str
    file_sha256: str


def canonical_evaluation_paths(
    manifest_path: str | Path, manifest: FreezeManifest
) -> EvaluationPaths:
    resolved_manifest = Path(manifest_path).resolve()
    expected_manifest = Path(manifest.evaluation_root).resolve() / "freeze.json"
    if resolved_manifest != expected_manifest:
        raise FreezeIntegrityError("freeze manifest moved from canonical evidence root")
    manifest_sha256 = sha256(resolved_manifest.read_bytes()).hexdigest()
    run_root = expected_manifest.parent / "runs" / manifest_sha256
    return EvaluationPaths(
        run_root=run_root,
        ledger_root=run_root / "public-ledger",
        seed_reservation_path=run_root / "private" / "seed-reservation.json",
        bundle_path=run_root / "private" / "eval-bundle.json",
    )


def _case(value: dict[str, Any]) -> HiddenCase:
    return HiddenCase(
        args=tuple(value["args"]),
        expected=value["expected"],
        distribution=str(value["distribution"]),
    )


def _task(value: dict[str, Any]) -> GeneratedTask:
    public_value = dict(value["public"])
    try:
        public_value["signature"] = TypedSignature.from_record(
            public_value["signature"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FreezeIntegrityError("public task signature is invalid") from error
    public = PublicTask(**public_value)
    private_value = value["private"]
    private = PrivateTask(
        task_id=str(private_value["task_id"]),
        public_task_sha256=str(private_value["public_task_sha256"]),
        regression_cases=tuple(_case(case) for case in private_value["regression_cases"]),
        property_cases=tuple(_case(case) for case in private_value["property_cases"]),
        test_sha256=str(private_value["test_sha256"]),
    )
    private_record = {
        "task_id": private.task_id,
        "signature": public.signature.to_record(),
        "regression_cases": [asdict(case) for case in private.regression_cases],
        "property_cases": [asdict(case) for case in private.property_cases],
    }
    if canonical_sha256(private_record) != private.test_sha256:
        raise FreezeIntegrityError("private hidden-test commitment is invalid")
    public_record = {
        "task_id": public.task_id,
        "partition": public.partition,
        "tier": public.tier,
        "family": public.family,
        "ordinal": public.ordinal,
        "signature": public.signature.to_record(),
        "prompt": public.prompt,
        "hidden_test_sha256": public.hidden_test_sha256,
    }
    if canonical_sha256(public_record) != public.task_sha256:
        raise FreezeIntegrityError("public task commitment is invalid")
    if (
        public.hidden_test_sha256 != private.test_sha256
        or private.public_task_sha256 != public.task_sha256
        or private.task_id != public.task_id
    ):
        raise FreezeIntegrityError("public/private task commitments are not linked")
    return GeneratedTask(public=public, private=private)


def load_evaluation_bundle(path: str | Path) -> EvaluationBundle:
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FreezeIntegrityError("private eval bundle is not JSON") from error
    if raw != canonical_json_bytes(payload):
        raise FreezeIntegrityError("private eval bundle is not canonical")
    if not isinstance(payload, dict) or payload.get("schema") != BUNDLE_SCHEMA:
        raise FreezeIntegrityError("private eval bundle schema differs")
    seed_hex = payload.get("eval_seed_hex")
    if not isinstance(seed_hex, str) or len(seed_hex) != 64:
        raise FreezeIntegrityError("private eval seed is invalid")
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as error:
        raise FreezeIntegrityError("private eval seed is invalid") from error
    seed_sha256 = sha256(seed).hexdigest()
    if payload.get("eval_seed_sha256") != seed_sha256:
        raise FreezeIntegrityError("private eval seed commitment differs")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise FreezeIntegrityError("private eval tasks are missing")
    tasks = tuple(_task(value) for value in raw_tasks)
    if len({task.public.task_id for task in tasks}) != len(tasks):
        raise FreezeIntegrityError("private eval task ids are not unique")
    return EvaluationBundle(
        protocol_id=str(payload["protocol_id"]),
        protocol_sha256=str(payload["protocol_sha256"]),
        freeze_manifest_sha256=str(payload["freeze_manifest_sha256"]),
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        checkpoint_state_sha256=str(payload["checkpoint_state_sha256"]),
        source_head=str(payload["source_head"]),
        eval_seed_sha256=seed_sha256,
        eval_seed=seed,
        tasks=tasks,
        bundle_sha256=sha256(raw).hexdigest(),
        path=resolved,
    )


def _bundle_payload(
    protocol: ProtocolSpec,
    manifest: FreezeManifest,
    manifest_sha256: str,
    seed: bytes,
    tasks: tuple[GeneratedTask, ...],
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "freeze_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "checkpoint_state_sha256": manifest.checkpoint_state_sha256,
        "source_head": manifest.source_head,
        "eval_seed_hex": seed.hex(),
        "eval_seed_sha256": sha256(seed).hexdigest(),
        "tasks": [
            {"public": asdict(task.public), "private": asdict(task.private)}
            for task in tasks
        ],
    }


def validate_evaluation_bundle(
    bundle: EvaluationBundle,
    protocol: ProtocolSpec,
    manifest: FreezeManifest,
    manifest_sha256: str,
) -> None:
    if (
        bundle.protocol_id != protocol.protocol_id
        or bundle.protocol_sha256 != protocol.sha256
        or bundle.freeze_manifest_sha256 != manifest_sha256
        or bundle.checkpoint_sha256 != manifest.checkpoint_sha256
        or bundle.checkpoint_state_sha256 != manifest.checkpoint_state_sha256
        or bundle.source_head != manifest.source_head
        or len(bundle.tasks) != protocol.partitions["eval"].task_count
    ):
        raise FreezeIntegrityError("private eval bundle differs from its freeze")
    observed: dict[str, int] = {}
    for task in bundle.tasks:
        observed[task.public.family] = observed.get(task.public.family, 0) + 1
        if task.public.partition != "eval":
            raise FreezeIntegrityError("private eval bundle contains another partition")
    if observed != dict(protocol.partitions["eval"].family_quotas):
        raise FreezeIntegrityError("private eval bundle changed frozen family quotas")
    expected_tasks = _generate_partition(
        protocol,
        "eval",
        int.from_bytes(bundle.eval_seed, byteorder="big", signed=False),
    )
    observed_task_bytes = canonical_json_bytes(
        [
            {"public": asdict(task.public), "private": asdict(task.private)}
            for task in bundle.tasks
        ]
    )
    expected_task_bytes = canonical_json_bytes(
        [
            {"public": asdict(task.public), "private": asdict(task.private)}
            for task in expected_tasks
        ]
    )
    if observed_task_bytes != expected_task_bytes:
        raise FreezeIntegrityError(
            "private eval tasks differ from the deterministic eval seed"
        )


def run_start_commitment(
    manifest: FreezeManifest, manifest_sha256: str
) -> dict[str, Any]:
    return {
        "freeze_manifest_sha256": manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "checkpoint_state_sha256": manifest.checkpoint_state_sha256,
        "source_head": manifest.source_head,
        "source_tree_sha256": manifest.source_tree_sha256,
        "inference_config_sha256": manifest.inference_config_sha256,
        "completion_adapter_sha256": manifest.completion_adapter_sha256,
        "compiler_id": manifest.compiler_id,
        "compiler_sha256": manifest.compiler_sha256,
        "compiler_mode": manifest.compiler_mode,
        "compiler_algorithm_source": manifest.compiler_algorithm_source,
        "canonical_runner_id": manifest.canonical_runner_id,
        "canonical_runner_sha256": manifest.canonical_runner_sha256,
        "sandbox_executor_id": manifest.sandbox_executor_id,
        "sandbox_executor_sha256": manifest.sandbox_executor_sha256,
        "sandbox_worker_sha256": manifest.sandbox_worker_sha256,
        "sandbox_python_executable": manifest.sandbox_python_executable,
        "sandbox_python_implementation": manifest.sandbox_python_implementation,
        "sandbox_python_version": manifest.sandbox_python_version,
        "sandbox_python_executable_sha256": manifest.sandbox_python_executable_sha256,
        "sandbox_python_flags": list(manifest.sandbox_python_flags),
        "simulation_backend": manifest.simulation_backend,
        "simulation_device": manifest.simulation_device,
        "simulation_source_digest": manifest.simulation_source_digest,
        "simulation_artifact_sha256": manifest.simulation_artifact_sha256,
        "production_dependencies_injected": False,
    }


def evaluation_reveal_commitment(bundle: EvaluationBundle) -> dict[str, Any]:
    return {
        "bundle_sha256": bundle.bundle_sha256,
        "eval_seed_sha256": bundle.eval_seed_sha256,
        "task_count": len(bundle.tasks),
        "task_commitments": [
            {
                "task_id": task.public.task_id,
                "task_sha256": task.public.task_sha256,
                "test_sha256": task.private.test_sha256,
                "tier": task.public.tier,
                "family": task.public.family,
            }
            for task in bundle.tasks
        ],
    }


def _ensure_run_start(
    ledger: ArtifactLedger,
    manifest: FreezeManifest,
    manifest_sha256: str,
) -> ArtifactEvent:
    events = ledger.verify()
    body = run_start_commitment(manifest, manifest_sha256)
    if not events:
        ledger.append("run-start", body)
        events = ledger.verify()
    if events[0].kind != "run-start" or events[0].body != body:
        raise FreezeIntegrityError("artifact ledger run-start differs from eval freeze")
    return events[0]


def _seed_payload(
    protocol: ProtocolSpec,
    manifest: FreezeManifest,
    manifest_sha256: str,
    seed: bytes,
) -> dict[str, str]:
    return {
        "schema": SEED_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "freeze_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "checkpoint_state_sha256": manifest.checkpoint_state_sha256,
        "source_head": manifest.source_head,
        "eval_seed_hex": seed.hex(),
        "eval_seed_sha256": sha256(seed).hexdigest(),
    }


def _load_seed_reservation(
    path: Path,
    protocol: ProtocolSpec,
    manifest: FreezeManifest,
    manifest_sha256: str,
) -> _PrivateSeedReservation:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FreezeIntegrityError("private eval seed reservation is not JSON") from error
    if raw != canonical_json_bytes(payload) or not isinstance(payload, dict):
        raise FreezeIntegrityError("private eval seed reservation is not canonical")
    seed_hex = payload.get("eval_seed_hex")
    if not isinstance(seed_hex, str) or len(seed_hex) != 64:
        raise FreezeIntegrityError("private eval seed reservation is invalid")
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as error:
        raise FreezeIntegrityError("private eval seed reservation is invalid") from error
    if payload != _seed_payload(protocol, manifest, manifest_sha256, seed):
        raise FreezeIntegrityError("private eval seed reservation differs from freeze")
    return _PrivateSeedReservation(
        seed=seed,
        seed_sha256=sha256(seed).hexdigest(),
        file_sha256=sha256(raw).hexdigest(),
    )


def _seed_event_body(seed: _PrivateSeedReservation) -> dict[str, Any]:
    return {
        "eval_seed_sha256": seed.seed_sha256,
        "seed_reservation_sha256": seed.file_sha256,
        "seed_persisted_before_task_generation": True,
        "evaluation_tasks_generated": False,
    }


def _ensure_seed_event(
    ledger: ArtifactLedger, seed: _PrivateSeedReservation
) -> ArtifactEvent:
    events = ledger.verify()
    body = _seed_event_body(seed)
    if len(events) == 1:
        ledger.append("evaluation-seed-reserved", body)
        events = ledger.verify()
    if len(events) < 2 or events[1].kind != "evaluation-seed-reserved":
        raise FreezeIntegrityError("artifact ledger lacks eval seed reservation")
    if events[1].body != body:
        raise FreezeIntegrityError("artifact ledger eval seed reservation differs")
    return events[1]


def _anchor_payload(
    manifest_sha256: str,
    protocol_sha256: str,
    run_start: ArtifactEvent,
    seed_event: ArtifactEvent,
    seed: _PrivateSeedReservation,
) -> dict[str, str]:
    return {
        "freeze_manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "run_start_event_sha256": run_start.file_sha256,
        "seed_event_sha256": seed_event.file_sha256,
        "eval_seed_sha256": seed.seed_sha256,
        "seed_reservation_sha256": seed.file_sha256,
    }


def external_anchor_payload_sha256(
    *,
    manifest_sha256: str,
    protocol_sha256: str,
    run_start_event_sha256: str,
    seed_event_sha256: str,
    eval_seed_sha256: str,
    seed_reservation_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "freeze_manifest_sha256": manifest_sha256,
            "protocol_sha256": protocol_sha256,
            "run_start_event_sha256": run_start_event_sha256,
            "seed_event_sha256": seed_event_sha256,
            "eval_seed_sha256": eval_seed_sha256,
            "seed_reservation_sha256": seed_reservation_sha256,
        }
    )


def _is_public_https_uri(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.netloc)


def validate_external_anchor_body(
    body: dict[str, Any], expected_payload_sha256: str
) -> None:
    expected_fields = {
        "anchor_payload_sha256",
        "public_uri",
        "receipt_sha256",
        "operator_supplied_receipt",
        "harness_verified_external_publication",
    }
    public_uri = body.get("public_uri")
    receipt_sha256 = body.get("receipt_sha256")
    if set(body) != expected_fields:
        raise FreezeIntegrityError("external anchor fields differ")
    if body.get("anchor_payload_sha256") != expected_payload_sha256:
        raise FreezeIntegrityError("external anchor does not commit this eval seed")
    if not _is_public_https_uri(public_uri):
        raise FreezeIntegrityError("external anchor lacks a public HTTPS URI")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in receipt_sha256)
    ):
        raise FreezeIntegrityError("external anchor receipt is not lowercase SHA-256")
    if (
        body.get("operator_supplied_receipt") is not True
        or body.get("harness_verified_external_publication") is not False
    ):
        raise FreezeIntegrityError("external anchor trust boundary differs")


def reserve_evaluation_seed(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
) -> EvaluationSeedReservation:
    """Persist one private seed and public hash, then stop before task generation."""
    protocol = load_protocol(protocol_path)
    manifest = verify_freeze_manifest(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        checkpoint_path=checkpoint_path,
        source_root=source_root,
    )
    manifest_sha256 = sha256(Path(manifest_path).read_bytes()).hexdigest()
    paths = canonical_evaluation_paths(manifest_path, manifest)
    ledger = ArtifactLedger(paths.ledger_root, protocol_sha256=protocol.sha256)
    events = ledger.verify()
    if not events and paths.run_root.exists():
        raise FreezeIntegrityError("canonical run registry predates run-start")
    run_start = _ensure_run_start(ledger, manifest, manifest_sha256)
    if paths.bundle_path.exists():
        raise FreezeIntegrityError("private eval bundle predates external anchor")
    if not paths.seed_reservation_path.exists():
        if len(ledger.verify()) != 1:
            raise FreezeIntegrityError("eval ledger advanced without its private seed")
        seed = secrets.token_bytes(32)
        atomic_create_bytes(
            paths.seed_reservation_path,
            canonical_json_bytes(_seed_payload(protocol, manifest, manifest_sha256, seed)),
        )
    seed = _load_seed_reservation(
        paths.seed_reservation_path, protocol, manifest, manifest_sha256
    )
    seed_event = _ensure_seed_event(ledger, seed)
    anchor_payload = _anchor_payload(
        manifest_sha256, protocol.sha256, run_start, seed_event, seed
    )
    return EvaluationSeedReservation(
        paths=paths,
        eval_seed_sha256=seed.seed_sha256,
        run_start_event_sha256=run_start.file_sha256,
        seed_event_sha256=seed_event.file_sha256,
        anchor_payload=anchor_payload,
        anchor_payload_sha256=canonical_sha256(anchor_payload),
    )


def record_external_anchor(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
    public_uri: str,
    receipt_sha256: str,
) -> ArtifactEvent:
    """Record an operator-supplied public anchor receipt; perform no network I/O."""
    if not _is_public_https_uri(public_uri):
        raise ValueError("external anchor must have a public HTTPS URI")
    if len(receipt_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in receipt_sha256
    ):
        raise ValueError("external anchor receipt must be lowercase SHA-256")
    reservation = reserve_evaluation_seed(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        checkpoint_path=checkpoint_path,
        source_root=source_root,
    )
    protocol = load_protocol(protocol_path)
    ledger = ArtifactLedger(
        reservation.paths.ledger_root, protocol_sha256=protocol.sha256
    )
    body = {
        "anchor_payload_sha256": reservation.anchor_payload_sha256,
        "public_uri": public_uri,
        "receipt_sha256": receipt_sha256,
        "operator_supplied_receipt": True,
        "harness_verified_external_publication": False,
    }
    events = ledger.verify()
    if len(events) == 2:
        ledger.append("external-anchor-recorded", body)
        events = ledger.verify()
    if len(events) < 3 or events[2].kind != "external-anchor-recorded":
        raise FreezeIntegrityError("artifact ledger lacks an external anchor")
    validate_external_anchor_body(events[2].body, reservation.anchor_payload_sha256)
    if events[2].body != body:
        raise FreezeIntegrityError("external anchor differs from existing receipt")
    return events[2]


def _ensure_reveal(ledger: ArtifactLedger, bundle: EvaluationBundle) -> ArtifactEvent:
    events = ledger.verify()
    body = evaluation_reveal_commitment(bundle)
    if len(events) == 3:
        ledger.append("evaluation-revealed", body)
        events = ledger.verify()
    if len(events) < 4 or events[3].kind != "evaluation-revealed":
        raise FreezeIntegrityError("artifact ledger has no evaluation reveal")
    if events[3].body != body:
        raise FreezeIntegrityError("artifact ledger reveal differs from private bundle")
    return events[3]


def reveal_evaluation(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
) -> EvaluationReveal:
    """Generate tasks only from an already anchored immutable seed reservation."""
    protocol = load_protocol(protocol_path)
    manifest = verify_freeze_manifest(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        checkpoint_path=checkpoint_path,
        source_root=source_root,
    )
    manifest_sha256 = sha256(Path(manifest_path).read_bytes()).hexdigest()
    paths = canonical_evaluation_paths(manifest_path, manifest)
    ledger = ArtifactLedger(paths.ledger_root, protocol_sha256=protocol.sha256)
    events = ledger.verify()
    if not paths.seed_reservation_path.exists() or len(events) < 2:
        raise FreezeIntegrityError("reserve and externally anchor eval seed first")
    seed = _load_seed_reservation(
        paths.seed_reservation_path, protocol, manifest, manifest_sha256
    )
    run_start = _ensure_run_start(ledger, manifest, manifest_sha256)
    seed_event = _ensure_seed_event(ledger, seed)
    if len(events) < 3 or events[2].kind != "external-anchor-recorded":
        raise FreezeIntegrityError("externally anchor eval seed before task generation")
    anchor = events[2]
    expected_anchor_sha256 = external_anchor_payload_sha256(
        manifest_sha256=manifest_sha256,
        protocol_sha256=protocol.sha256,
        run_start_event_sha256=run_start.file_sha256,
        seed_event_sha256=seed_event.file_sha256,
        eval_seed_sha256=seed.seed_sha256,
        seed_reservation_sha256=seed.file_sha256,
    )
    validate_external_anchor_body(anchor.body, expected_anchor_sha256)
    if paths.bundle_path.exists():
        bundle = load_evaluation_bundle(paths.bundle_path)
    else:
        if len(ledger.verify()) != 3:
            raise FreezeIntegrityError("eval ledger advanced without its private bundle")
        tasks = _generate_partition(
            protocol, "eval", int.from_bytes(seed.seed, byteorder="big", signed=False)
        )
        atomic_create_bytes(
            paths.bundle_path,
            canonical_json_bytes(
                _bundle_payload(
                    protocol, manifest, manifest_sha256, seed.seed, tasks
                )
            ),
        )
        bundle = load_evaluation_bundle(paths.bundle_path)
    validate_evaluation_bundle(bundle, protocol, manifest, manifest_sha256)
    reveal_event = _ensure_reveal(ledger, bundle)
    return EvaluationReveal(
        bundle_path=bundle.path,
        ledger_root=paths.ledger_root,
        bundle_sha256=bundle.bundle_sha256,
        eval_seed_sha256=bundle.eval_seed_sha256,
        task_count=len(bundle.tasks),
        run_start_event_sha256=run_start.file_sha256,
        seed_event_sha256=seed_event.file_sha256,
        external_anchor_event_sha256=anchor.file_sha256,
        reveal_event_sha256=reveal_event.file_sha256,
    )


__all__ = [
    "EvaluationBundle",
    "EvaluationPaths",
    "EvaluationReveal",
    "EvaluationSeedReservation",
    "canonical_evaluation_paths",
    "evaluation_reveal_commitment",
    "external_anchor_payload_sha256",
    "load_evaluation_bundle",
    "record_external_anchor",
    "reserve_evaluation_seed",
    "reveal_evaluation",
    "run_start_commitment",
    "validate_evaluation_bundle",
    "validate_external_anchor_body",
]
