"""One-shot V3 public gate for factorized H=2 created-child planning.

V3 preserves V1/V2's two visible Int tasks, declared targets, order, literal
bank, and fail-fast semantics.  Its only mechanism change is the generic
compositional sequence-prompt projection committed at ``3f2a3cb``.  Importing
this module spends nothing; ``main`` creates an exclusive durable marker
immediately before evaluating the first task.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from experiments.r3_pilot import (
    public_typed_factor_horizon2_int_created_child_gate as v1,
)
from experiments.r3_pilot import public_typed_prompt_factors as prompt_factors
from experiments.r3_pilot import public_typed_structure_successor as structure
from humanbrain import typed_program as tp

SCHEMA = "public-typed-factor-horizon2-int-created-child-v3"
PASS_VERDICT = "PASS_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V3"
STOP_VERDICT = "STOP_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V3"
MECHANISM_COMMIT = "3f2a3cbe97f4cd935137e4f97970957271b9384a"
PUBLIC_LITERAL_BANK = v1.PUBLIC_LITERAL_BANK
PUBLIC_TASKS = v1.PUBLIC_TASKS
PUBLIC_TARGETS = v1.PUBLIC_TARGETS
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_SPEND_MARKER_PATH = Path(__file__).with_suffix(".spend.json")

_EXPECTED_PROTOCOL_SHA256 = (
    "8391878e95b6da435308654ed4998f7e3b89af681728f9ecfbe1bbd36c34adfd"
)
_EXPECTED_MODEL_CHECKPOINT = (
    "7820c258c0d7d243008c5ec95b7ed19ceb91b77729396877b0d474af5a272f85"
)
_EXPECTED_WORLD_CHECKPOINT = (
    "37b5e25bbb532f7398f984defeb52b937a6230535fad76069aefc359a5f5b666"
)
_EXPECTED_STRUCTURE_CHECKPOINT = (
    "95a35644cb0689ddd30dc5994ac4c9f15b61ae019ce377681051845c68d96c8e"
)
_EXPECTED_SOURCE_HASHES = {
    "protocol_source_lf_sha256": (
        "1a7ec5743430b2c1368f2b0c4c3509f248150bb00626e296826d250c9e662d15"
    ),
    "horizon2_source_lf_sha256": (
        "694559e170f00088ee11c370d3ce7bfc708c8e20b693502501855fb042eeca7d"
    ),
    "prompt_factors_source_lf_sha256": (
        "d740f14731affe5561801e8d0eea3bd37cdf1ddc088e57839d857b5340bcdd5d"
    ),
    "structure_successor_source_lf_sha256": (
        "d1e7670e8f218aa719f8ccd715f404a75a3200ae7f79744dd46a02005b2909bf"
    ),
    "typed_program_source_lf_sha256": (
        "339e55b69c177c7f529a18843e6bdc2bba96d09c251b97306abcfdb86e1d22a2"
    ),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _lf_sha256(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _module_path(module: object) -> Path:
    source = getattr(module, "__file__", None)
    if source is None:
        raise RuntimeError(f"module has no source path: {module!r}")
    return Path(source)


def _protocol_record() -> dict[str, object]:
    return {
        "literal_bank": PUBLIC_LITERAL_BANK,
        "tasks": tuple(asdict(task) for task in PUBLIC_TASKS),
        "targets": tuple(asdict(target) for target in PUBLIC_TARGETS),
    }


def public_protocol_sha256() -> str:
    return sha256(_canonical_json(_protocol_record())).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "runner_source_lf_sha256": _lf_sha256(Path(__file__)),
        "protocol_source_lf_sha256": _lf_sha256(_module_path(v1)),
        "horizon2_source_lf_sha256": _lf_sha256(_module_path(h2)),
        "prompt_factors_source_lf_sha256": _lf_sha256(
            _module_path(prompt_factors)
        ),
        "structure_successor_source_lf_sha256": _lf_sha256(
            _module_path(structure)
        ),
        "typed_program_source_lf_sha256": _lf_sha256(_module_path(tp)),
    }


class _DecisionRecordingModel:
    """Transparent planner wrapper retaining decisions for provenance only."""

    def __init__(self, model: h2.FrozenFactorizedHorizon2Int) -> None:
        self.model = model
        self.decisions: list[h2.FactorizedH2Decision] = []

    @property
    def world(self):
        return self.model.world

    @property
    def checkpoint_sha256(self) -> str:
        return self.model.checkpoint_sha256

    @property
    def live_checkpoint_sha256(self) -> str:
        return self.model.live_checkpoint_sha256

    def observe(self, *args, **kwargs):
        return self.model.observe(*args, **kwargs)

    def plan(self, *args, **kwargs):
        decision = self.model.plan(*args, **kwargs)
        self.decisions.append(decision)
        return decision

    def actuate_and_reobserve(self, *args, **kwargs):
        return self.model.actuate_and_reobserve(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class PromptSequenceCheckpointTrace:
    task_name: str
    statuses: tuple[str, ...]
    projection_sha256s: tuple[str, ...]
    checkpoint_before_sha256s: tuple[str, ...]
    checkpoint_after_sha256s: tuple[str, ...]
    immutable: bool


def _prompt_trace(
    task_name: str,
    decisions: tuple[h2.FactorizedH2Decision, ...],
) -> PromptSequenceCheckpointTrace:
    prompt_decisions = tuple(
        attempt.prompt_decision
        for decision in decisions
        for attempt in decision.terminal_attempts
        if attempt.prompt_decision is not None
    )
    return PromptSequenceCheckpointTrace(
        task_name=task_name,
        statuses=tuple(item.status for item in prompt_decisions),
        projection_sha256s=tuple(
            item.projection_sha256 for item in prompt_decisions
        ),
        checkpoint_before_sha256s=tuple(
            item.checkpoint_before_sha256 for item in prompt_decisions
        ),
        checkpoint_after_sha256s=tuple(
            item.checkpoint_after_sha256 for item in prompt_decisions
        ),
        immutable=all(
            item.projection_sha256
            == item.checkpoint_before_sha256
            == item.checkpoint_after_sha256
            for item in prompt_decisions
        ),
    )


@dataclass(frozen=True, slots=True)
class PublicCreatedChildV3MatrixResult:
    schema: str
    verdict: str
    passed_tasks: int
    attempted_tasks: int
    total_tasks: int
    wall_time_seconds: float
    literal_bank: tuple[int, ...]
    mechanism_commit: str
    public_protocol_sha256: str
    model_checkpoint_sha256: str
    world_checkpoint_sha256: str
    structure_successor_checkpoint_sha256: str
    runner_source_lf_sha256: str
    protocol_source_lf_sha256: str
    horizon2_source_lf_sha256: str
    prompt_factors_source_lf_sha256: str
    structure_successor_source_lf_sha256: str
    typed_program_source_lf_sha256: str
    spend_marker_lf_sha256: str
    prompt_sequence_traces: tuple[PromptSequenceCheckpointTrace, ...]
    tasks: tuple[v1.PublicCreatedChildTaskResult, ...]
    retry_policy: str
    claim_boundary: str


def run_public_matrix(
    model: h2.FrozenFactorizedHorizon2Int,
    *,
    spend_marker_lf_sha256: str,
) -> PublicCreatedChildV3MatrixResult:
    """Run the unchanged two-task public protocol once, fail-fast in V1 order."""

    started = perf_counter()
    results: list[v1.PublicCreatedChildTaskResult] = []
    traces: list[PromptSequenceCheckpointTrace] = []
    for task in PUBLIC_TASKS:
        recording_model = _DecisionRecordingModel(model)
        try:
            result = v1.run_public_task(recording_model, task)
        except Exception as error:
            result = v1._exception_result(task, error, recording_model)
        results.append(result)
        traces.append(_prompt_trace(task.name, tuple(recording_model.decisions)))
        if not result.passed:
            break
    passed = sum(item.passed for item in results)
    prompt_bound = all(
        trace.projection_sha256s and trace.immutable for trace in traces
    )
    verdict = (
        PASS_VERDICT
        if passed == len(PUBLIC_TASKS)
        and len(results) == len(PUBLIC_TASKS)
        and prompt_bound
        else STOP_VERDICT
    )
    hashes = _source_hashes()
    return PublicCreatedChildV3MatrixResult(
        schema=SCHEMA,
        verdict=verdict,
        passed_tasks=passed,
        attempted_tasks=len(results),
        total_tasks=len(PUBLIC_TASKS),
        wall_time_seconds=perf_counter() - started,
        literal_bank=PUBLIC_LITERAL_BANK,
        mechanism_commit=MECHANISM_COMMIT,
        public_protocol_sha256=public_protocol_sha256(),
        model_checkpoint_sha256=model.checkpoint_sha256,
        world_checkpoint_sha256=model.world.checkpoint_sha256,
        structure_successor_checkpoint_sha256=(
            model.world.structure_successor.checkpoint_sha256
        ),
        spend_marker_lf_sha256=spend_marker_lf_sha256,
        prompt_sequence_traces=tuple(traces),
        tasks=tuple(results),
        retry_policy=(
            "This V3 public-development artifact and spend marker are exclusive. "
            "No prompt, case, current, threshold, literal-bank, candidate-order, "
            "seed, fixture, structure, or mechanism tuning is permitted after "
            "this spend."
        ),
        claim_boundary=(
            "Two visible public Int tasks with generic compositional sequence-"
            "prompt evidence after exact value and supported-structure tiers, "
            "genuine learned-cell H=2 lookahead over created child sites, one "
            "real edit per replan, and terminal DONE. Not Bool/List, Simple-SWE, "
            "M0, DEV/eval/held-out, protected data, C1, or broad coding evidence."
        ),
        **hashes,
    )


def write_public_artifact(
    result: PublicCreatedChildV3MatrixResult,
    path: Path = _ARTIFACT_PATH,
) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_pretty_json(asdict(result)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            f"public created-child H=2 V3 artifact already exists: {path}"
        ) from error


def _validate_predeclared_boundary(
    model: h2.FrozenFactorizedHorizon2Int,
) -> None:
    if PUBLIC_LITERAL_BANK != tuple(range(-4, 5)):
        raise RuntimeError("public created-child V3 literal bank changed")
    if PUBLIC_TASKS is not v1.PUBLIC_TASKS or PUBLIC_TARGETS is not v1.PUBLIC_TARGETS:
        raise RuntimeError("public created-child V3 protocol diverged from V1")
    if tuple(task.name for task in PUBLIC_TASKS) != (
        "left-created-child",
        "right-created-child",
    ):
        raise RuntimeError("public created-child V3 task order changed")
    if public_protocol_sha256() != _EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("public created-child V3 protocol hash changed")
    sources = _source_hashes()
    for name, expected in _EXPECTED_SOURCE_HASHES.items():
        if sources[name] != expected:
            raise RuntimeError(f"public created-child V3 source changed: {name}")
    if (
        model.checkpoint_sha256 != _EXPECTED_MODEL_CHECKPOINT
        or model.live_checkpoint_sha256 != _EXPECTED_MODEL_CHECKPOINT
        or model.world.checkpoint_sha256 != _EXPECTED_WORLD_CHECKPOINT
        or model.world.structure_successor.checkpoint_sha256
        != _EXPECTED_STRUCTURE_CHECKPOINT
    ):
        raise RuntimeError("public created-child V3 checkpoint changed")


def _create_spend_marker(
    model: h2.FrozenFactorizedHorizon2Int,
    path: Path = _SPEND_MARKER_PATH,
) -> str:
    if _ARTIFACT_PATH.exists():
        raise RuntimeError("V3 artifact appeared before the spend marker")
    marker = {
        "schema": f"{SCHEMA}-spend",
        "mechanism_commit": MECHANISM_COMMIT,
        "public_protocol_sha256": public_protocol_sha256(),
        "task_order": tuple(task.name for task in PUBLIC_TASKS),
        "literal_bank": PUBLIC_LITERAL_BANK,
        "model_checkpoint_sha256": model.checkpoint_sha256,
        "world_checkpoint_sha256": model.world.checkpoint_sha256,
        "structure_successor_checkpoint_sha256": (
            model.world.structure_successor.checkpoint_sha256
        ),
        **_source_hashes(),
        "policy": (
            "Durably spent immediately before the first V3 public task; no "
            "same-name retry or post-result prompt/current/case/structure tuning."
        ),
    }
    try:
        with path.open("xb") as handle:
            handle.write(_pretty_json(marker))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing spent V3 gate: {path}") from error
    if _ARTIFACT_PATH.exists():
        raise RuntimeError("V3 artifact appeared while creating the spend marker")
    return _lf_sha256(path)


def main() -> int:
    if _ARTIFACT_PATH.exists():
        prior = json.loads(_ARTIFACT_PATH.read_bytes())
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "verdict": prior.get("verdict"),
                    "guard": "existing artifact; V3 public matrix not rerun",
                },
                sort_keys=True,
            )
        )
        return 0
    if _SPEND_MARKER_PATH.exists():
        raise RuntimeError(
            "V3 spend marker exists without an artifact; same-name rerun refused"
        )
    model = h2.build_public_factorized_horizon2_int(literals=PUBLIC_LITERAL_BANK)
    _validate_predeclared_boundary(model)
    marker_sha = _create_spend_marker(model)
    result = run_public_matrix(model, spend_marker_lf_sha256=marker_sha)
    if _lf_sha256(_SPEND_MARKER_PATH) != marker_sha:
        raise RuntimeError("V3 spend marker changed during the public matrix")
    write_public_artifact(result)
    print(
        json.dumps(
            {
                "artifact": str(_ARTIFACT_PATH),
                "verdict": result.verdict,
                "passed_tasks": result.passed_tasks,
                "attempted_tasks": result.attempted_tasks,
                "total_tasks": result.total_tasks,
                "wall_time_seconds": result.wall_time_seconds,
            },
            sort_keys=True,
        )
    )
    return 0 if result.verdict == PASS_VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MECHANISM_COMMIT",
    "PASS_VERDICT",
    "PUBLIC_LITERAL_BANK",
    "PUBLIC_TARGETS",
    "PUBLIC_TASKS",
    "SCHEMA",
    "STOP_VERDICT",
    "PromptSequenceCheckpointTrace",
    "PublicCreatedChildV3MatrixResult",
    "public_protocol_sha256",
    "run_public_matrix",
    "write_public_artifact",
]
