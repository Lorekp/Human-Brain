from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from humanbrain.simple_swe.artifacts import ArtifactLedger, atomic_create_bytes
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


SCHEMA = "humanbrain.simple-swe-private-eval-bundle.v1"


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
class EvaluationReveal:
    bundle_path: Path
    bundle_sha256: str
    eval_seed_sha256: str
    task_count: int
    run_start_event_sha256: str
    reveal_event_sha256: str


def _outside(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return
    raise FreezeIntegrityError(f"{label} must remain outside the model-visible source tree")


def _case(value: dict[str, Any]) -> HiddenCase:
    return HiddenCase(
        args=tuple(value["args"]),
        expected=value["expected"],
        distribution=str(value["distribution"]),
    )


def _task(value: dict[str, Any]) -> GeneratedTask:
    public_value = dict(value["public"])
    public_value["parameter_names"] = tuple(public_value["parameter_names"])
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
        "entry_point": public.entry_point,
        "parameter_names": list(public.parameter_names),
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
        "entry_point": public.entry_point,
        "parameter_names": list(public.parameter_names),
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
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
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
        "schema": SCHEMA,
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


def _validate_bundle_against_freeze(
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
    observed = {}
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
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "checkpoint_state_sha256": manifest.checkpoint_state_sha256,
        "source_head": manifest.source_head,
        "completion_adapter_sha256": manifest.completion_adapter_sha256,
        "compiler_sha256": manifest.compiler_sha256,
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
) -> str:
    events = ledger.verify()
    run_start_body = run_start_commitment(manifest, manifest_sha256)
    if not events:
        ledger.append("run-start", run_start_body)
        events = ledger.verify()
    if events[0].kind != "run-start" or events[0].body != run_start_body:
        raise FreezeIntegrityError("artifact ledger run-start differs from eval freeze")
    return events[0].file_sha256


def _ensure_reveal(ledger: ArtifactLedger, bundle: EvaluationBundle) -> str:
    events = ledger.verify()
    reveal_body = evaluation_reveal_commitment(bundle)
    if len(events) == 1:
        ledger.append("evaluation-revealed", reveal_body)
        events = ledger.verify()
    if len(events) < 2:
        raise FreezeIntegrityError("artifact ledger has no evaluation reveal")
    if events[1].kind != "evaluation-revealed" or events[1].body != reveal_body:
        raise FreezeIntegrityError("artifact ledger reveal differs from private bundle")
    return events[1].file_sha256


def reveal_evaluation(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
    private_bundle_path: str | Path,
    ledger: ArtifactLedger,
    compiler_identity: object | None = None,
) -> EvaluationReveal:
    """Reveal all frozen eval tasks once, after every freeze check succeeds."""
    root = Path(source_root).resolve()
    private_path = Path(private_bundle_path).resolve()
    _outside(private_path, root, "private eval bundle")
    _outside(ledger.root, root, "artifact ledger")
    try:
        private_path.relative_to(ledger.root)
    except ValueError:
        pass
    else:
        raise FreezeIntegrityError("private eval bundle must not be stored in public ledger")
    protocol = load_protocol(protocol_path)
    if ledger.protocol_sha256 != protocol.sha256:
        raise FreezeIntegrityError("artifact ledger protocol hash differs")
    manifest = verify_freeze_manifest(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        checkpoint_path=checkpoint_path,
        source_root=root,
        compiler_identity=compiler_identity,
    )
    manifest_sha256 = sha256(Path(manifest_path).read_bytes()).hexdigest()
    events_before_run_start = ledger.verify()
    if private_path.exists() and not events_before_run_start:
        raise FreezeIntegrityError(
            "private eval bundle predates the public run-start commitment"
        )
    run_start_sha256 = _ensure_run_start(ledger, manifest, manifest_sha256)
    if private_path.exists():
        bundle = load_evaluation_bundle(private_path)
    else:
        if len(ledger.verify()) != 1:
            raise FreezeIntegrityError(
                "artifact ledger advanced without its private eval bundle"
            )
        seed = secrets.token_bytes(32)
        tasks = _generate_partition(
            protocol, "eval", int.from_bytes(seed, byteorder="big", signed=False)
        )
        atomic_create_bytes(
            private_path,
            canonical_json_bytes(
                _bundle_payload(protocol, manifest, manifest_sha256, seed, tasks)
            ),
        )
        bundle = load_evaluation_bundle(private_path)
    _validate_bundle_against_freeze(bundle, protocol, manifest, manifest_sha256)
    reveal_sha256 = _ensure_reveal(ledger, bundle)
    return EvaluationReveal(
        bundle_path=bundle.path,
        bundle_sha256=bundle.bundle_sha256,
        eval_seed_sha256=bundle.eval_seed_sha256,
        task_count=len(bundle.tasks),
        run_start_event_sha256=run_start_sha256,
        reveal_event_sha256=reveal_sha256,
    )


__all__ = [
    "EvaluationBundle",
    "EvaluationReveal",
    "evaluation_reveal_commitment",
    "load_evaluation_bundle",
    "reveal_evaluation",
    "run_start_commitment",
]
