"""Run the post-freeze, self-posed fresh coding challenge exactly once.

The generic cell checkpoint predates this file and all four tasks below.  Each
arm sees the same public task and visible cases, receives only the worker's raw
case envelope, and may make an initial attempt plus two repairs.  Final
completions are sealed before hidden cases are constructed or executed.

This is intentionally a small capability experiment, not a new mechanism
tuning loop and not a canonical Simple-SWE/M0 evaluation.
"""

from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import fresh_coding_checkpoint as checkpoint_spec  # noqa: E402
import numpy as np  # noqa: E402
from humanbrain.language import CellLanguageModel  # noqa: E402
from humanbrain.simple_swe import sandbox  # noqa: E402
from humanbrain.simple_swe.protocol import ExecutionLimits  # noqa: E402

FORMAT = "humanbrain.fresh-coding-challenge.v1"
SEAL_FORMAT = "humanbrain.fresh-coding-completion-seal.v1"
SPENT_FORMAT = "humanbrain.fresh-coding-spent.v1"

CHECKPOINT_SOURCE_COMMIT = "34ba9625b943d6ae4714a801fd4811915a706969"
CHECKPOINT_ARTIFACT_COMMIT = "d6cca59874ee112b43867407898bac890147ce2c"
CHECKPOINT_SHA256 = "6d972946012f34884a7bf012223eda52aa8ac252f12d2356a12b7bc389077fe9"
CHECKPOINT_SOURCE_SHA256 = "40492212e78f32baf976291f32ead91c48d8e83568faee0e39d7675bddbe815f"
CORPUS_SHA256 = "b35a2edeabfb9e78755c4eb7bf3c77babacdddeaf2c0b4164af0c934e54ea56c"
MANIFEST_SHA256 = "0942545cd03e21ac94c047db90c15e59b9327c5171db74afd3963d3cbeef7362"
WORKER_SHA256 = "3a731267aa59799693e9eeb398c86ba9d8e261a2df86856da201430a65b7c223"

CHECKPOINT_PATH = Path(__file__).with_name("fresh_coding_checkpoint.npz")
CHECKPOINT_SOURCE_PATH = Path(__file__).with_name("fresh_coding_checkpoint.py")
CORPUS_PATH = Path(__file__).with_name("fresh_coding_generic_corpus.json")
MANIFEST_PATH = Path(__file__).with_name("fresh_coding_checkpoint.manifest.json")
WORKER_PATH = Path(__file__).with_name("fresh_coding_visible_worker.py")
RESULT_PATH = Path(__file__).with_name("fresh_coding_challenge.json")
SEAL_PATH = Path(__file__).with_name("fresh_coding_challenge.seal.json")
SPENT_PATH = Path(__file__).with_name("fresh_coding_challenge.spent.json")

FROZEN_RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.11.15",
    "numpy_version": "2.4.6",
}

ENTRY_POINT = "solve"
PARAMETERS = ("arg0", "arg1", "arg2")
ARMS = ("cell", "cell_off", "b_random")
ATTEMPTS = (0, 1, 2)
MAX_NEW_TOKENS = 192
RANDOM_BASELINE_SEED = 20260828
REGRESSION_CASES_PER_TASK = 4
PROPERTY_CASES_PER_TASK = 32

EXECUTION_LIMITS = ExecutionLimits(
    wall_seconds=3.0,
    cpu_seconds=2,
    memory_mib=256,
    processes=1,
    output_bytes=1_048_576,
    max_completion_bytes=16_384,
    max_ast_nodes=1_024,
    python_hash_seed=0,
)

_POLICY_TASK = SimpleNamespace(
    signature=SimpleNamespace(
        entry_point=ENTRY_POINT,
        parameters=tuple(SimpleNamespace(name=name) for name in PARAMETERS),
    )
)


class CompletionPreflightError(ValueError):
    """A raw canonical sandbox preflight rejection."""


@dataclass(frozen=True)
class Case:
    args: tuple[Any, Any, Any]
    expected: Any
    kind: str


@dataclass(frozen=True)
class ChallengeTask:
    task_id: str
    text: str
    oracle: Callable[[Any, Any, Any], Any]
    visible_cases: tuple[Case, ...]


@dataclass(frozen=True)
class InventorySource:
    source_id: str
    task: str
    source: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _oracle_affine_list(arg0: list[int], arg1: int, arg2: int) -> list[int]:
    return [item * arg1 + arg2 for item in arg0]


def _oracle_nearest_prefix(arg0: list[int], arg1: int, arg2: int) -> int:
    best = arg0[0]
    best_distance = abs(best - arg1)
    for item in arg0[1:arg2]:
        distance = abs(item - arg1)
        if distance < best_distance:
            best = item
            best_distance = distance
    return best


def _oracle_running_clamped_maxima(
    arg0: list[int], arg1: int, arg2: int
) -> list[int]:
    current = arg1
    result: list[int] = []
    for item in arg0:
        current = max(current, item)
        current = min(arg2, current)
        result.append(current)
    return result


def _oracle_cyclic_rotation_prefix(
    arg0: list[int], arg1: int, arg2: int
) -> list[int]:
    if not arg0 or arg2 <= 0:
        return []
    offset = arg1 % len(arg0)
    rotated = arg0[offset:] + arg0[:offset]
    return [rotated[index % len(rotated)] for index in range(arg2)]


def _case(
    args: tuple[Any, Any, Any], expected: Any, kind: str = "visible"
) -> Case:
    return Case(args=args, expected=expected, kind=kind)


def task_specs() -> tuple[ChallengeTask, ...]:
    """Return the four post-freeze tasks in their immutable execution order."""

    return (
        ChallengeTask(
            task_id="fresh_01_affine_list",
            text=(
                "Given integer list arg0 and integers arg1 and arg2, return a new "
                "list whose item for each input x is x multiplied by arg1 and then "
                "increased by arg2. Preserve order."
            ),
            oracle=_oracle_affine_list,
            visible_cases=(
                _case(([], 2, 3), []),
                _case(([1, -2, 0], 3, 4), [7, -2, 4]),
                _case(([-3, 5], -2, 1), [7, -9]),
            ),
        ),
        ChallengeTask(
            task_id="fresh_02_nearest_prefix",
            text=(
                "arg0 is a nonempty integer list. Inspect only its first arg2 items, "
                "where 1 <= arg2 <= len(arg0), and return the item closest to integer "
                "arg1. If distances tie, return the leftmost tied item."
            ),
            oracle=_oracle_nearest_prefix,
            visible_cases=(
                _case(([8, 3, 5, 4], 4, 3), 3),
                _case(([-5, 10, 2], 0, 2), -5),
                _case(([7, 1, 9], 8, 1), 7),
            ),
        ),
        ChallengeTask(
            task_id="fresh_03_running_clamped_maxima",
            text=(
                "Starting with current = arg1, process integer list arg0 from left to "
                "right. For each item, replace current by max(current, item), then by "
                "min(arg2, current), and append current. Return the appended list. "
                "arg1 is never greater than arg2."
            ),
            oracle=_oracle_running_clamped_maxima,
            visible_cases=(
                _case(([1, 7, 3, 12], 0, 9), [1, 7, 7, 9]),
                _case(([], 2, 5), []),
                _case(([-5, -1, 0], -3, 2), [-3, -1, 0]),
                _case(([10, -4], 5, 5), [5, 5]),
            ),
        ),
        ChallengeTask(
            task_id="fresh_04_cyclic_rotation_prefix",
            text=(
                "Rotate list arg0 left by integer arg1 positions (arg1 may be "
                "negative), repeat the rotated list cyclically, and return its first "
                "arg2 values. Return [] when arg0 is empty or arg2 <= 0."
            ),
            oracle=_oracle_cyclic_rotation_prefix,
            visible_cases=(
                _case(([1, 2, 3], 1, 5), [2, 3, 1, 2, 3]),
                _case(([1, 2, 3, 4], -1, 3), [4, 1, 2]),
                _case(([], 7, 4), []),
                _case(([9, 8], 3, 0), []),
            ),
        ),
    )


def _hidden_cases(task_id: str) -> tuple[Case, ...]:
    """Construct a task's hidden cases; official execution calls this post-seal."""

    if task_id == "fresh_01_affine_list":
        oracle = _oracle_affine_list
        regression_args = (
            ([0], 0, 0),
            ([2, -2], 0, 7),
            ([4, 1, -3], -4, -2),
            ([-9, 9], 1, 0),
        )
        rng = random.Random(11_701)
        property_args = tuple(
            (
                [rng.randint(-20, 20) for _ in range(rng.randint(0, 8))],
                rng.randint(-5, 5),
                rng.randint(-9, 9),
            )
            for _ in range(PROPERTY_CASES_PER_TASK)
        )
    elif task_id == "fresh_02_nearest_prefix":
        oracle = _oracle_nearest_prefix
        regression_args = (
            ([4], 100, 1),
            ([1, 9, 5], 7, 3),
            ([-10, -2, 6, 1], 0, 3),
            ([8, 2, 3], 3, 2),
        )
        rng = random.Random(22_702)
        generated: list[tuple[list[int], int, int]] = []
        for _ in range(PROPERTY_CASES_PER_TASK):
            values = [rng.randint(-30, 30) for _ in range(rng.randint(1, 9))]
            generated.append((values, rng.randint(-35, 35), rng.randint(1, len(values))))
        property_args = tuple(generated)
    elif task_id == "fresh_03_running_clamped_maxima":
        oracle = _oracle_running_clamped_maxima
        regression_args = (
            ([], -2, -2),
            ([10, 9, 8], 0, 5),
            ([-8, -3, -7], -5, 4),
            ([3, 1, 4, 2], 2, 9),
        )
        rng = random.Random(33_703)
        generated = []
        for _ in range(PROPERTY_CASES_PER_TASK):
            floor = rng.randint(-10, 10)
            cap = rng.randint(floor, floor + 20)
            values = [rng.randint(-20, 25) for _ in range(rng.randint(0, 9))]
            generated.append((values, floor, cap))
        property_args = tuple(generated)
    elif task_id == "fresh_04_cyclic_rotation_prefix":
        oracle = _oracle_cyclic_rotation_prefix
        regression_args = (
            ([], -99, 8),
            ([5], -7, 4),
            ([1, 2, 3], 7, 2),
            ([4, 6, 8, 10], -6, 9),
        )
        rng = random.Random(44_704)
        property_args = tuple(
            (
                [rng.randint(-15, 15) for _ in range(rng.randint(0, 8))],
                rng.randint(-20, 20),
                rng.randint(-3, 20),
            )
            for _ in range(PROPERTY_CASES_PER_TASK)
        )
    else:
        raise KeyError(f"unknown fresh task: {task_id}")

    regression = tuple(
        _case(tuple(args), oracle(*args), "regression") for args in regression_args
    )
    properties = tuple(
        _case(tuple(args), oracle(*args), "property") for args in property_args
    )
    if len(regression) != REGRESSION_CASES_PER_TASK:
        raise AssertionError("hidden regression count drifted")
    if len(properties) != PROPERTY_CASES_PER_TASK:
        raise AssertionError("hidden property count drifted")
    return regression + properties


def _case_payload(case: Case) -> dict[str, Any]:
    return {
        "args": list(case.args),
        "expected": case.expected,
    }


def _same_json_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in right
        )
    return left == right


def _validate_task_specs(tasks: Sequence[ChallengeTask]) -> None:
    if tuple(task.task_id for task in tasks) != (
        "fresh_01_affine_list",
        "fresh_02_nearest_prefix",
        "fresh_03_running_clamped_maxima",
        "fresh_04_cyclic_rotation_prefix",
    ):
        raise RuntimeError("fresh task identity or order drifted")
    for task in tasks:
        if not task.visible_cases:
            raise RuntimeError(f"{task.task_id} needs visible cases")
        for case in task.visible_cases:
            if len(case.args) != len(PARAMETERS) or case.kind != "visible":
                raise RuntimeError(f"{task.task_id} visible case contract drifted")
            expected = task.oracle(*case.args)
            if not _same_json_value(expected, case.expected):
                raise RuntimeError(f"{task.task_id} visible oracle mismatch")


def _run_git(arguments: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise RuntimeError(detail)
    return completed.stdout.strip()


def _assert_hash(path: Path, expected: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"frozen hash mismatch for {_repo_relative(path)}: {actual}")


def _creation_commit(path: Path) -> str:
    commits = _run_git(
        ["log", "--diff-filter=A", "--format=%H", "--", _repo_relative(path)]
    ).splitlines()
    if len(commits) != 1:
        raise RuntimeError(f"expected one creation commit for {_repo_relative(path)}")
    return commits[0]


def _latest_source_commit(path: Path) -> str:
    commit = _run_git(["log", "-1", "--format=%H", "--", _repo_relative(path)])
    if not commit:
        raise RuntimeError(f"no source commit for {_repo_relative(path)}")
    return commit


def _assert_ancestor(ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{ancestor} is not an ancestor of {descendant}")


def _assert_tracked_clean(paths: Sequence[Path]) -> None:
    relative = [_repo_relative(path) for path in paths]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError("challenge, worker, checkpoint, and corpus must be committed")
    dirty = _run_git(["status", "--porcelain", "--", *relative])
    if dirty:
        raise RuntimeError(f"frozen challenge inputs are dirty:\n{dirty}")


def _preflight() -> dict[str, Any]:
    if _runtime_identity() != FROZEN_RUNTIME:
        raise RuntimeError(
            f"frozen runtime mismatch: expected {FROZEN_RUNTIME}, got {_runtime_identity()}"
        )
    for output in (RESULT_PATH, SEAL_PATH, SPENT_PATH):
        if output.exists():
            raise FileExistsError(f"refusing to repeat spent challenge: {output}")

    sources = (
        Path(__file__),
        WORKER_PATH,
        CHECKPOINT_PATH,
        MANIFEST_PATH,
        CHECKPOINT_SOURCE_PATH,
        CORPUS_PATH,
    )
    _assert_tracked_clean(sources)
    _assert_hash(WORKER_PATH, WORKER_SHA256)
    _assert_hash(CHECKPOINT_PATH, CHECKPOINT_SHA256)
    _assert_hash(MANIFEST_PATH, MANIFEST_SHA256)
    _assert_hash(CHECKPOINT_SOURCE_PATH, CHECKPOINT_SOURCE_SHA256)
    _assert_hash(CORPUS_PATH, CORPUS_SHA256)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["freeze_source_commit"] != CHECKPOINT_SOURCE_COMMIT:
        raise RuntimeError("checkpoint source commit drifted")
    if manifest["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint manifest hash drifted")
    if manifest["corpus"]["sha256"] != CORPUS_SHA256:
        raise RuntimeError("corpus manifest hash drifted")
    if manifest["scope"]["target_tasks_present_at_freeze"] is not False:
        raise RuntimeError("checkpoint no longer proves pre-task freeze")

    challenge_creation = _creation_commit(Path(__file__))
    worker_creation = _creation_commit(WORKER_PATH)
    if challenge_creation == CHECKPOINT_ARTIFACT_COMMIT:
        raise RuntimeError("challenge must be created after the checkpoint artifact")
    _assert_ancestor(CHECKPOINT_SOURCE_COMMIT, CHECKPOINT_ARTIFACT_COMMIT)
    _assert_ancestor(CHECKPOINT_ARTIFACT_COMMIT, challenge_creation)
    _assert_ancestor(CHECKPOINT_ARTIFACT_COMMIT, worker_creation)

    challenge_source_commit = _latest_source_commit(Path(__file__))
    worker_source_commit = _latest_source_commit(WORKER_PATH)
    head = _run_git(["rev-parse", "HEAD"])
    _assert_ancestor(challenge_source_commit, head)
    _assert_ancestor(worker_source_commit, head)
    return {
        "head_commit": head,
        "checkpoint_source_commit": CHECKPOINT_SOURCE_COMMIT,
        "checkpoint_artifact_commit": CHECKPOINT_ARTIFACT_COMMIT,
        "challenge_creation_commit": challenge_creation,
        "challenge_source_commit": challenge_source_commit,
        "challenge_source_sha256": _sha256_file(Path(__file__)),
        "worker_creation_commit": worker_creation,
        "worker_source_commit": worker_source_commit,
        "worker_sha256": WORKER_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "corpus_sha256": CORPUS_SHA256,
    }


def _source_inventory(corpus: dict[str, Any]) -> tuple[InventorySource, ...]:
    inventory: list[InventorySource] = []
    for example in corpus["examples"]:
        for version in ("revision", "initial"):
            inventory.append(
                InventorySource(
                    source_id=f"{example['id']}:{version}",
                    task=example["task"],
                    source=example[f"{version}_source"],
                )
            )
    if not inventory:
        raise RuntimeError("frozen source inventory is empty")
    return tuple(inventory)


def _inventory_sha256(inventory: Sequence[InventorySource]) -> str:
    payload = [
        {"source_id": item.source_id, "task": item.task, "source": item.source}
        for item in inventory
    ]
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _task_prompt(task: ChallengeTask) -> str:
    return (
        f"{checkpoint_spec.TASK_OPEN}\nReturn only Python source.\n"
        f"signature: solve(arg0, arg1, arg2)\n"
        f"task: {task.text.strip()}\n"
    )


def _direct_prompt(task: ChallengeTask) -> str:
    return f"{_task_prompt(task)}{checkpoint_spec.SOURCE_OPEN}\n"


def _repair_prompt(task: ChallengeTask, candidate: str, raw_feedback: str) -> str:
    return (
        f"{_task_prompt(task)}{checkpoint_spec.CANDIDATE_OPEN}\n"
        f"{candidate}{checkpoint_spec.SOURCE_END}\n"
        f"{checkpoint_spec.FEEDBACK_OPEN}\n{raw_feedback}\n"
        f"{checkpoint_spec.REVISION_OPEN}\n"
    )


def _extract_source(completion: str) -> tuple[str, bool]:
    """Remove only an exact terminal protocol frame; preserve every other byte."""

    framed = completion.endswith(checkpoint_spec.SOURCE_END)
    if framed:
        return completion[: -len(checkpoint_spec.SOURCE_END)], True
    return completion, False


def _validate_source_policy(source: str) -> None:
    outcome, detail = sandbox._preflight(source, _POLICY_TASK, EXECUTION_LIMITS)
    if outcome is not None:
        raise CompletionPreflightError(f"{outcome}: {detail}")


def _is_parseable_solve(source: str) -> bool:
    try:
        _validate_source_policy(source)
    except (CompletionPreflightError, UnicodeError):
        return False
    return True


def _lexemes(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _cell_off_source(
    task: ChallengeTask, attempt: int, inventory: Sequence[InventorySource]
) -> InventorySource:
    task_words = _lexemes(task.text)
    ranked = sorted(
        inventory,
        key=lambda item: (
            -len(task_words.intersection(_lexemes(item.task))),
            item.source_id,
        ),
    )
    return ranked[attempt % len(ranked)]


def _random_source(
    task: ChallengeTask, attempt: int, inventory: Sequence[InventorySource]
) -> InventorySource:
    seed_material = f"{RANDOM_BASELINE_SEED}:{task.task_id}:{attempt}".encode()
    seed = int.from_bytes(sha256(seed_material).digest()[:8], "big")
    return inventory[random.Random(seed).randrange(len(inventory))]


def _decode_worker_result(
    stdout: bytes, cases: Sequence[Case]
) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
        result = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("worker emitted invalid JSON") from error
    if not isinstance(result, dict) or set(result) != {"cases", "passed"}:
        raise RuntimeError("worker result envelope drifted")
    rows = result["cases"]
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise RuntimeError("worker case count drifted")
    if not isinstance(result["passed"], bool):
        raise RuntimeError("worker passed flag must be bool")
    for row, case in zip(rows, cases, strict=True):
        if not isinstance(row, dict):
            raise RuntimeError("worker case row must be an object")
        observation = {"actual", "exception"}.intersection(row)
        required = {"args", "expected", "passed"}
        if set(row) != required | observation or len(observation) != 1:
            raise RuntimeError("worker case schema drifted")
        if not isinstance(row["passed"], bool):
            raise RuntimeError("worker case passed flag must be bool")
        if not _same_json_value(row["args"], list(case.args)):
            raise RuntimeError("worker changed case args")
        if not _same_json_value(row["expected"], case.expected):
            raise RuntimeError("worker changed expected value")
        if "exception" in row:
            if not isinstance(row["exception"], str) or not row["exception"]:
                raise RuntimeError("worker exception must be non-empty text")
            if row["passed"] is not False:
                raise RuntimeError("exception case cannot pass")
        else:
            expected_pass = _same_json_value(row["actual"], row["expected"])
            if row["passed"] is not expected_pass:
                raise RuntimeError("worker case comparison drifted")
    if result["passed"] is not (bool(rows) and all(row["passed"] for row in rows)):
        raise RuntimeError("worker aggregate flag drifted")
    if text != _canonical_json(result):
        raise RuntimeError("worker JSON is not canonical")
    return result


def _raw_policy_failure(cases: Sequence[Case], error: BaseException) -> dict[str, Any]:
    exception = f"{type(error).__name__}: {error}"
    return {
        "cases": [
            {
                "args": list(case.args),
                "expected": case.expected,
                "exception": exception,
                "passed": False,
            }
            for case in cases
        ],
        "passed": False,
    }


def _execute_cases(
    source: str,
    cases: Sequence[Case],
    worker_bytes: bytes,
    executable: Path,
) -> dict[str, Any]:
    try:
        _validate_source_policy(source)
    except (SyntaxError, ValueError, TypeError, UnicodeError) as error:
        return _raw_policy_failure(cases, error)
    payload = {
        "source": source,
        "entry_point": ENTRY_POINT,
        "cases": [_case_payload(case) for case in cases],
        "limits": {
            "cpu_seconds": EXECUTION_LIMITS.cpu_seconds,
            "memory_bytes": EXECUTION_LIMITS.memory_mib * 1024 * 1024,
            "output_bytes": EXECUTION_LIMITS.output_bytes,
            "processes": EXECUTION_LIMITS.processes,
        },
    }
    observation = sandbox._run_worker_process(
        executable,
        worker_bytes,
        _canonical_json(payload).encode("utf-8"),
        EXECUTION_LIMITS,
    )
    if len(observation.stdout) > EXECUTION_LIMITS.output_bytes:
        return _raw_policy_failure(cases, RuntimeError("sandbox stdout limit exceeded"))
    if len(observation.stderr) > EXECUTION_LIMITS.output_bytes:
        return _raw_policy_failure(cases, RuntimeError("sandbox stderr limit exceeded"))
    if observation.timed_out:
        return _raw_policy_failure(cases, TimeoutError("sandbox worker timed out"))
    if observation.returncode != 0:
        return _raw_policy_failure(
            cases, RuntimeError(f"sandbox worker returned {observation.returncode}")
        )
    if not observation.stdout:
        return _raw_policy_failure(cases, RuntimeError("sandbox worker emitted empty stdout"))
    if observation.stderr:
        return _raw_policy_failure(cases, RuntimeError("sandbox worker emitted stderr"))
    try:
        return _decode_worker_result(observation.stdout, cases)
    except RuntimeError:
        return _raw_policy_failure(cases, RuntimeError("sandbox worker result was invalid"))


def _smoke_worker(worker_bytes: bytes, executable: Path) -> None:
    source = "def solve(arg0, arg1, arg2):\n    return arg0 + arg1 + arg2\n"
    smoke_case = _case((2, 3, 4), 9, "smoke")
    result = _execute_cases(source, (smoke_case,), worker_bytes, executable)
    if result["passed"] is not True:
        raise RuntimeError("known-source worker smoke failed")


def _candidate_for_arm(
    *,
    arm: str,
    task: ChallengeTask,
    attempt: int,
    model: CellLanguageModel,
    inventory: Sequence[InventorySource],
    previous_source: str | None,
    raw_feedback: str | None,
) -> tuple[str, str | None, bool | None]:
    if arm == "cell":
        if attempt == 0:
            prompt = _direct_prompt(task)
        else:
            if previous_source is None or raw_feedback is None:
                raise RuntimeError("cell repair requires its preceding raw observation")
            prompt = _repair_prompt(task, previous_source, raw_feedback)
        attempt_model = CellLanguageModel.load(CHECKPOINT_PATH)
        if attempt_model.training_digest != model.training_digest:
            raise RuntimeError("per-attempt checkpoint reload drifted")
        completion = attempt_model.complete(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            stop_text=checkpoint_spec.SOURCE_END,
        )
        source, terminal_frame = _extract_source(completion)
        return source, None, terminal_frame
    if arm == "cell_off":
        selected = _cell_off_source(task, attempt, inventory)
        return selected.source, selected.source_id, None
    if arm == "b_random":
        selected = _random_source(task, attempt, inventory)
        return selected.source, selected.source_id, None
    raise KeyError(f"unknown arm: {arm}")


def _run_arm(
    *,
    arm: str,
    task: ChallengeTask,
    model: CellLanguageModel,
    inventory: Sequence[InventorySource],
    worker_bytes: bytes,
    executable: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    previous_source: str | None = None
    raw_feedback: str | None = None
    for attempt in ATTEMPTS:
        source, inventory_source_id, terminal_frame = _candidate_for_arm(
            arm=arm,
            task=task,
            attempt=attempt,
            model=model,
            inventory=inventory,
            previous_source=previous_source,
            raw_feedback=raw_feedback,
        )
        result = _execute_cases(source, task.visible_cases, worker_bytes, executable)
        raw_feedback = _canonical_json(result)
        attempts.append(
            {
                "attempt": attempt,
                "source": source,
                "source_sha256": _sha256_bytes(source.encode("utf-8")),
                "inventory_source_id": inventory_source_id,
                "terminal_source_end_emitted": terminal_frame,
                "parseable_solve": _is_parseable_solve(source),
                "visible_passed_cases": sum(row["passed"] for row in result["cases"]),
                "visible_case_count": len(result["cases"]),
                "visible_passed": result["passed"],
                "raw_feedback": raw_feedback,
            }
        )
        previous_source = source
        if result["passed"]:
            break

    final = attempts[-1]
    initial_passed = attempts[0]["visible_passed_cases"]
    improvement = any(
        attempt["visible_passed_cases"] > initial_passed for attempt in attempts[1:]
    )
    has_parseable = any(attempt["parseable_solve"] for attempt in attempts)
    productive = has_parseable and (bool(final["visible_passed"]) or improvement)
    return {
        "attempts": attempts,
        "final_attempt": final["attempt"],
        "final_source": final["source"],
        "final_source_sha256": final["source_sha256"],
        "has_parseable_source": has_parseable,
        "visible_improvement": improvement,
        "visible_passed": final["visible_passed"],
        "productive_visible_signal": productive,
        "selection_rule": "last_attempted_candidate_not_best_of",
    }


def _run_task(
    task: ChallengeTask,
    model: CellLanguageModel,
    inventory: Sequence[InventorySource],
    worker_bytes: bytes,
    executable: Path,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task": task.text,
        "signature": "solve(arg0, arg1, arg2)",
        "visible_cases": [_case_payload(case) for case in task.visible_cases],
        "arms": {
            arm: _run_arm(
                arm=arm,
                task=task,
                model=model,
                inventory=inventory,
                worker_bytes=worker_bytes,
                executable=executable,
            )
            for arm in ARMS
        },
    }


def _fail_fast_status(task_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first_two = task_records[:2]
    signals = {
        record["task_id"]: bool(record["arms"]["cell"]["productive_visible_signal"])
        for record in first_two
    }
    return {
        "rule": "stop_after_task_2_if_both_cell_tasks_have_no_parseable_progress_signal",
        "evaluated_task_ids": [record["task_id"] for record in first_two],
        "cell_productive_visible_signal": signals,
        "triggered": len(first_two) == 2 and not any(signals.values()),
    }


def _completion_seal_payload(
    task_records: Sequence[dict[str, Any]], preflight: dict[str, Any]
) -> dict[str, Any]:
    completions: list[dict[str, Any]] = []
    for record in task_records:
        for arm in ARMS:
            arm_record = record["arms"][arm]
            completions.append(
                {
                    "task_id": record["task_id"],
                    "arm": arm,
                    "final_attempt": arm_record["final_attempt"],
                    "source_sha256": arm_record["final_source_sha256"],
                }
            )
    return {
        "format": SEAL_FORMAT,
        "challenge_source_commit": preflight["challenge_source_commit"],
        "challenge_source_sha256": preflight["challenge_source_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "worker_sha256": WORKER_SHA256,
        "final_completions": completions,
    }


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(data)


def _score_hidden(
    *,
    task: ChallengeTask,
    task_record: dict[str, Any],
    worker_bytes: bytes,
    executable: Path,
    expected_seal_sha256: str,
    expected_seal_payload: dict[str, Any],
) -> dict[str, Any]:
    if not SEAL_PATH.exists() or _sha256_file(SEAL_PATH) != expected_seal_sha256:
        raise RuntimeError("hidden scoring requires the exact durable completion seal")
    stored_seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    if stored_seal != expected_seal_payload:
        raise RuntimeError("stored completion seal does not match the frozen payload")
    sealed_sources = {
        (row["task_id"], row["arm"]): row["source_sha256"]
        for row in stored_seal["final_completions"]
    }
    cases = _hidden_cases(task.task_id)
    commitment = _sha256_bytes(
        _canonical_json([_case_payload(case) for case in cases]).encode("utf-8")
    )
    arms: dict[str, Any] = {}
    for arm in ARMS:
        final_source = task_record["arms"][arm]["final_source"]
        actual_source_sha256 = _sha256_bytes(final_source.encode("utf-8"))
        if sealed_sources.get((task.task_id, arm)) != actual_source_sha256:
            raise RuntimeError(f"final source no longer matches seal for {task.task_id}/{arm}")
        if task_record["arms"][arm]["final_source_sha256"] != actual_source_sha256:
            raise RuntimeError(f"task record source hash drifted for {task.task_id}/{arm}")
        result = _execute_cases(final_source, cases, worker_bytes, executable)
        regression_rows = result["cases"][:REGRESSION_CASES_PER_TASK]
        property_rows = result["cases"][REGRESSION_CASES_PER_TASK:]
        arms[arm] = {
            "passed": result["passed"],
            "passed_cases": sum(row["passed"] for row in result["cases"]),
            "case_count": len(result["cases"]),
            "regression_passed_cases": sum(row["passed"] for row in regression_rows),
            "regression_case_count": len(regression_rows),
            "property_passed_cases": sum(row["passed"] for row in property_rows),
            "property_case_count": len(property_rows),
        }
    return {
        "case_commitment_sha256": commitment,
        "regression_case_count": REGRESSION_CASES_PER_TASK,
        "property_case_count": PROPERTY_CASES_PER_TASK,
        "arms": arms,
    }


def _promotion(hidden: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cell_passes = [
        task_id
        for task_id, score in hidden.items()
        if score["arms"]["cell"]["passed"]
    ]
    cell_off_passes = {
        task_id
        for task_id, score in hidden.items()
        if score["arms"]["cell_off"]["passed"]
    }
    distinct_from_cell_off = [
        task_id for task_id in cell_passes if task_id not in cell_off_passes
    ]
    new_shape_passes = [
        task_id
        for task_id in cell_passes
        if task_id
        in {"fresh_03_running_clamped_maxima", "fresh_04_cyclic_rotation_prefix"}
    ]
    gates = {
        "at_least_two_of_four": len(cell_passes) >= 2,
        "includes_task_3_or_4": bool(new_shape_passes),
        "at_least_one_not_matched_by_cell_off": bool(distinct_from_cell_off),
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "cell_passed_task_ids": cell_passes,
        "new_shape_passed_task_ids": new_shape_passes,
        "cell_passes_not_matched_by_cell_off": distinct_from_cell_off,
    }


def run_challenge() -> dict[str, Any]:
    """Run the one-shot cohort and publish its immutable result."""

    preflight = _preflight()
    tasks = task_specs()
    _validate_task_specs(tasks)
    worker_bytes = WORKER_PATH.read_bytes()
    if _sha256_bytes(worker_bytes) != WORKER_SHA256:
        raise RuntimeError("worker bytes changed after preflight")

    corpus = checkpoint_spec.load_corpus(CORPUS_PATH)
    inventory = _source_inventory(corpus)
    model = CellLanguageModel.load(CHECKPOINT_PATH)
    executable = sandbox.resolve_python_executable(None)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if model.training_digest != manifest["model"]["training_digest"]:
        raise RuntimeError("loaded cell checkpoint training digest drifted")

    _smoke_worker(worker_bytes, executable)
    spent = {
        "format": SPENT_FORMAT,
        "state": "started",
        "challenge_source_commit": preflight["challenge_source_commit"],
        "challenge_source_sha256": preflight["challenge_source_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "worker_sha256": WORKER_SHA256,
    }
    spent_sha256 = _write_json_exclusive(SPENT_PATH, spent)

    task_records: list[dict[str, Any]] = []
    for task in tasks[:2]:
        task_records.append(_run_task(task, model, inventory, worker_bytes, executable))
    fail_fast = _fail_fast_status(task_records)
    if not fail_fast["triggered"]:
        for task in tasks[2:]:
            task_records.append(_run_task(task, model, inventory, worker_bytes, executable))

    seal_payload = _completion_seal_payload(task_records, preflight)
    seal_sha256 = _write_json_exclusive(SEAL_PATH, seal_payload)

    hidden_scores: dict[str, dict[str, Any]] = {}
    if fail_fast["triggered"]:
        promotion = {
            "gates": {},
            "passed": False,
            "cell_passed_task_ids": [],
            "new_shape_passed_task_ids": [],
            "cell_passes_not_matched_by_cell_off": [],
        }
        verdict = "STOP_FRESH_CODING_NO_VISIBLE_PROGRESS"
        hidden_opened = False
    else:
        by_id = {record["task_id"]: record for record in task_records}
        for task in tasks:
            hidden_scores[task.task_id] = _score_hidden(
                task=task,
                task_record=by_id[task.task_id],
                worker_bytes=worker_bytes,
                executable=executable,
                expected_seal_sha256=seal_sha256,
                expected_seal_payload=seal_payload,
            )
        promotion = _promotion(hidden_scores)
        verdict = (
            "PASS_FRESH_CODING_CHALLENGE"
            if promotion["passed"]
            else "STOP_FRESH_CODING_CHALLENGE"
        )
        hidden_opened = True

    artifact: dict[str, Any] = {
        "format": FORMAT,
        "verdict": verdict,
        "scope": {
            "tasks": "self_posed_noncanonical_post_freeze",
            "canonical_simple_swe_opened": False,
            "m0_opened": False,
            "claim": "fresh_micro_coding_capability_only",
        },
        "preflight": preflight,
        "protocol": {
            "entry_point": ENTRY_POINT,
            "parameters": list(PARAMETERS),
            "arms": list(ARMS),
            "attempts": list(ATTEMPTS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "candidate_selection": "last_attempted_candidate_not_best_of",
            "feedback": "raw_visible_worker_json_only",
            "completion_sealed_before_hidden": True,
            "hidden_regression_cases_per_task": REGRESSION_CASES_PER_TASK,
            "hidden_property_cases_per_task": PROPERTY_CASES_PER_TASK,
            "source_inventory_sha256": _inventory_sha256(inventory),
            "source_inventory_count": len(inventory),
        },
        "completion_seal": {
            "path": _repo_relative(SEAL_PATH),
            "sha256": seal_sha256,
            "completion_count": len(seal_payload["final_completions"]),
        },
        "spend_marker": {
            "path": _repo_relative(SPENT_PATH),
            "sha256": spent_sha256,
            "state": "started_before_first_generation",
        },
        "fail_fast": fail_fast,
        "hidden_scoring_opened": hidden_opened,
        "promotion": promotion,
        "tasks": task_records,
        "hidden_scores": hidden_scores,
    }
    _write_json_exclusive(RESULT_PATH, artifact)
    return artifact


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("this one-shot challenge accepts no options")
    print(json.dumps(run_challenge(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
