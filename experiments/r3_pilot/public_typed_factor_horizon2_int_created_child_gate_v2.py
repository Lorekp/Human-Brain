"""One-shot V2 public gate for factorized H=2 created-child planning.

V2 preserves the two visible Int tasks, declared targets, order, literal bank,
and fail-fast semantics of V1.  Its only mechanism change is the target-free,
compiler-generated structural-successor curriculum committed at ``a31f647``.
Importing this module spends nothing; ``main`` writes an exclusive durable
spend marker immediately before the first task is evaluated.
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
from experiments.r3_pilot import public_typed_structure_successor as structure
from humanbrain import typed_program as tp

SCHEMA = "public-typed-factor-horizon2-int-created-child-v2"
PASS_VERDICT = "PASS_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V2"
STOP_VERDICT = "STOP_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V2"
MECHANISM_COMMIT = "a31f64751b6025d99b06087ed4ec2c36bb8b4718"
PUBLIC_LITERAL_BANK = v1.PUBLIC_LITERAL_BANK
PUBLIC_TASKS = v1.PUBLIC_TASKS
PUBLIC_TARGETS = v1.PUBLIC_TARGETS
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_SPEND_MARKER_PATH = Path(__file__).with_suffix(".spend.json")


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
        "structure_successor_source_lf_sha256": _lf_sha256(
            _module_path(structure)
        ),
        "typed_program_source_lf_sha256": _lf_sha256(_module_path(tp)),
    }


@dataclass(frozen=True, slots=True)
class PublicCreatedChildV2MatrixResult:
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
    structure_successor_source_lf_sha256: str
    typed_program_source_lf_sha256: str
    spend_marker_lf_sha256: str
    tasks: tuple[v1.PublicCreatedChildTaskResult, ...]
    retry_policy: str
    claim_boundary: str


def run_public_matrix(
    *,
    spend_marker_lf_sha256: str,
) -> PublicCreatedChildV2MatrixResult:
    """Run the unchanged two-task public protocol once, fail-fast in V1 order."""

    started = perf_counter()
    model = h2.build_public_factorized_horizon2_int(literals=PUBLIC_LITERAL_BANK)
    results: list[v1.PublicCreatedChildTaskResult] = []
    for task in PUBLIC_TASKS:
        try:
            result = v1.run_public_task(model, task)
        except Exception as error:
            # Preserve V1's frozen exception record and STOP behavior.
            result = v1._exception_result(task, error, model)
        results.append(result)
        if not result.passed:
            break
    passed = sum(item.passed for item in results)
    verdict = (
        PASS_VERDICT
        if passed == len(PUBLIC_TASKS) and len(results) == len(PUBLIC_TASKS)
        else STOP_VERDICT
    )
    hashes = _source_hashes()
    return PublicCreatedChildV2MatrixResult(
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
        tasks=tuple(results),
        retry_policy=(
            "This V2 public-development artifact and spend marker are exclusive. "
            "No prompt, case, current, threshold, literal-bank, candidate-order, "
            "seed, fixture, or mechanism tuning is permitted after this spend."
        ),
        claim_boundary=(
            "Two visible public Int tasks with compiler-generated structural "
            "successor coverage, genuine learned-cell H=2 lookahead over created "
            "child sites, one real edit per replan, and terminal DONE. Not Bool/"
            "List, Simple-SWE, M0, DEV/eval/held-out, protected data, C1, or "
            "broad coding evidence."
        ),
        **hashes,
    )


def write_public_artifact(
    result: PublicCreatedChildV2MatrixResult,
    path: Path = _ARTIFACT_PATH,
) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(_pretty_json(asdict(result)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            f"public created-child H=2 V2 artifact already exists: {path}"
        ) from error


def _validate_predeclared_boundary() -> None:
    if PUBLIC_LITERAL_BANK != tuple(range(-4, 5)):
        raise RuntimeError("public created-child V2 literal bank changed")
    if PUBLIC_TASKS != v1.PUBLIC_TASKS or PUBLIC_TARGETS != v1.PUBLIC_TARGETS:
        raise RuntimeError("public created-child V2 protocol diverged from V1")
    if len(PUBLIC_TASKS) != 2:
        raise RuntimeError("public created-child V2 gate must contain two tasks")
    if tuple(task.name for task in PUBLIC_TASKS) != (
        "left-created-child",
        "right-created-child",
    ):
        raise RuntimeError("public created-child V2 task order changed")


def _create_spend_marker(path: Path = _SPEND_MARKER_PATH) -> str:
    if _ARTIFACT_PATH.exists():
        raise RuntimeError("V2 artifact appeared before the spend marker")
    marker = {
        "schema": f"{SCHEMA}-spend",
        "mechanism_commit": MECHANISM_COMMIT,
        "public_protocol_sha256": public_protocol_sha256(),
        "task_order": tuple(task.name for task in PUBLIC_TASKS),
        "literal_bank": PUBLIC_LITERAL_BANK,
        **_source_hashes(),
        "policy": (
            "Durably spent immediately before the first V2 public task; no "
            "same-name retry or post-result mechanism/fixture tuning."
        ),
    }
    try:
        with path.open("xb") as handle:
            handle.write(_pretty_json(marker))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing spent V2 gate: {path}") from error
    if _ARTIFACT_PATH.exists():
        raise RuntimeError("V2 artifact appeared while creating the spend marker")
    return _lf_sha256(path)


def main() -> int:
    if _ARTIFACT_PATH.exists():
        prior = json.loads(_ARTIFACT_PATH.read_bytes())
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "verdict": prior.get("verdict"),
                    "guard": "existing artifact; V2 public matrix not rerun",
                },
                sort_keys=True,
            )
        )
        return 0
    if _SPEND_MARKER_PATH.exists():
        raise RuntimeError(
            "V2 spend marker exists without an artifact; same-name rerun refused"
        )
    _validate_predeclared_boundary()
    marker_sha = _create_spend_marker()
    result = run_public_matrix(spend_marker_lf_sha256=marker_sha)
    if _lf_sha256(_SPEND_MARKER_PATH) != marker_sha:
        raise RuntimeError("V2 spend marker changed during the public matrix")
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
    "PublicCreatedChildV2MatrixResult",
    "public_protocol_sha256",
    "run_public_matrix",
    "write_public_artifact",
]
