"""Lean sealed V9 session for persistent neuronal repository patch learning.

The learner sees only a public issue, every file in a detached public workspace,
an optional public receipt from its preceding action, and one ``submit_patch``
JSON-Schema affordance.  Private expected actions are imported only by
``_posthoc_score`` after every arm has finished its public stream.

Importing this module never grows a model, opens private task data, executes a
patch, or reserves the result artifact.  Running the expensive session requires
an explicit ``--run`` flag.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Protocol

from humanbrain.persistent_tool_learning import (
    EpisodeResult,
    ExecutionReceipt,
    ObservationBlock,
    ObservationFrame,
    PersistentToolLearner,
    StopDecision,
    ToolAction,
    ToolAffordance,
    prompt_for_frame,
)

SCHEMA = "continual-coding-session-v9"
RESERVATION_SCHEMA = "continual-coding-session-v9-reservation"
SUBMIT_PATCH = "submit_patch"
PUBLIC_PRIOR_CASES = 8
MAX_NEW_TOKENS = 8192
MAX_LEXICAL_ROLES = 4096
MAX_OBSERVATION_FEATURES = 4096
MODEL_STOP_TEXT = ("</tool>", "</stop>")
_ASCII_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "r4_integrated_continual_coding_session_v9.json"
RUNNER_PATH = Path(__file__).resolve()
FIXTURE_PATH = REPO_ROOT / "experiments" / "r4_integrated" / "continual_toy_swe_tasks_v9.py"
FIXTURE_MANIFEST_PATH = (
    REPO_ROOT / "experiments" / "r4_integrated" / "continual_toy_swe_tasks_v9_manifest.json"
)
MODEL_PATH = REPO_ROOT / "src" / "humanbrain" / "neuronal_tool_model.py"
TOOL_CORE_PATH = REPO_ROOT / "src" / "humanbrain" / "persistent_tool_learning.py"
SEQUENCE_MODEL_PATH = REPO_ROOT / "src" / "humanbrain" / "causal_sequence_pointer.py"
PRIOR_PATH = REPO_ROOT / "experiments" / "r4_integrated" / "public_coding_prior_v1.py"
SANDBOX_WORKER_PATH = REPO_ROOT / "src" / "humanbrain" / "simple_swe" / "_repository_worker.py"
SANDBOX_PATH = REPO_ROOT / "src" / "humanbrain" / "simple_swe" / "sandbox.py"

# Filled exactly once by the orchestrator after the converging fixture, worker,
# and neuronal bridge commits land.  Until then public preflight fails closed;
# merely recording whatever files happen to be live would not freeze evidence.
EXPECTED_DEPENDENCY_SHA256: dict[str, str | None] = {
    "experiments/r4_integrated/continual_toy_swe_tasks_v9.py": (
        "2ff864ec3819e79dca5ef11dbd7ebe2a0b1dee2fe50bc47c65d424d3c8b5ef5d"
    ),
    "experiments/r4_integrated/continual_toy_swe_tasks_v9_manifest.json": (
        "b01d00a2ba26c3245ca39251a024db6764679533da15c8ab3bc00d17d8a06ddc"
    ),
    "experiments/r4_integrated/public_coding_prior_v1.py": (
        "95b0bbae62cc1fc08dd94e0f25e32116c2bbe671c7d5ceb10b50a4617ff0ac5b"
    ),
    "src/humanbrain/causal_sequence_pointer.py": (
        "e8d1ff323755209c0f71fb6e6617d7c677e8a2355454bc2e7658322cc3cd2495"
    ),
    "src/humanbrain/neuronal_tool_model.py": (
        "15c6cb6e96f0b18f589cd2f38d1b23332d82b9ae4254d37bacfa8d5fa5699049"
    ),
    "src/humanbrain/persistent_tool_learning.py": (
        "8c0d3bc3a105c9a15a00f5f0eb9eb418f932f69d6fb8803649e2d3be2d5bd80c"
    ),
    "src/humanbrain/simple_swe/_repository_worker.py": (
        "0a130dd6123704fa7905bd79b6f5d94d4cea8d3f1263d2a565c99c8319a5ac9f"
    ),
    "src/humanbrain/simple_swe/sandbox.py": (
        "37cbf4667d51f43d343406735d0252d8672da2ab773ea0c914c14ba216ab370b"
    ),
}


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(domain: str, value: object) -> str:
    return sha256(domain.encode("ascii") + b"\0" + _json(value).encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _lexical_inventory_cardinality(value: str) -> int:
    """Count the documented first-occurrence ASCII lexical copy inventory."""

    return len(dict.fromkeys(_ASCII_IDENTIFIER.findall(value)))


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _audit_value(value: object) -> object:
    """Return stable JSON data for source/config custody, not learner input."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _audit_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _audit_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_audit_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported audit value: {type(value).__name__}")


def _canonical_workspace_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("workspace path must be a non-empty canonical POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("workspace path must stay inside the detached repository")
    canonical = path.as_posix()
    if canonical != value:
        raise ValueError("workspace path is not canonical")
    return canonical


def _workspace_text(path: str, content: bytes) -> str:
    canonical = _canonical_workspace_path(path)
    if not isinstance(content, bytes):
        raise TypeError("learner workspace contents must be detached bytes")
    return _json({"content_utf8": content.decode("utf-8"), "path": canonical})


def _submit_patch_affordance(contract: Mapping[str, object]) -> ToolAffordance:
    """Authenticate the one public tool contract used by training and runtime."""

    if not isinstance(contract, Mapping):
        raise TypeError("V9 global tool contract must be a mapping")
    external = contract.get("external_mutating_affordances")
    if not isinstance(external, list) or len(external) != 1:
        raise ValueError("V9 must expose one external mutating affordance")
    action = external[0]
    if (
        not isinstance(action, Mapping)
        or set(action) != {"name", "payload"}
        or action["name"] != SUBMIT_PATCH
        or not isinstance(action["payload"], Mapping)
    ):
        raise ValueError("V9 must expose exactly the submit_patch JSON-Schema affordance")
    schema = dict(action["payload"])
    expected = {
        "additionalProperties": False,
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
        "type": "object",
    }
    if schema != expected:
        raise ValueError("submit_patch schema is not the frozen strict patch schema")
    return ToolAffordance.seal(SUBMIT_PATCH, schema)


def _frame(
    *,
    issue: str,
    workspace_files: Mapping[str, bytes],
    affordance: ToolAffordance,
    prior_public_receipt: Mapping[str, object] | None = None,
) -> ObservationFrame:
    """Build the whole learner-visible state with no task-specific routing fields."""

    if not isinstance(issue, str) or not issue:
        raise ValueError("public issue must be non-empty text")
    if not isinstance(workspace_files, Mapping) or not workspace_files:
        raise ValueError("learner workspace must contain at least one file")
    canonical_files: list[tuple[str, bytes]] = []
    for raw_path, content in workspace_files.items():
        canonical_files.append((_canonical_workspace_path(raw_path), content))
    canonical_files.sort(key=lambda item: item[0])
    if len({path for path, _ in canonical_files}) != len(canonical_files):
        raise ValueError("learner workspace contains duplicate canonical paths")

    blocks = [ObservationBlock.seal("issue", issue)]
    blocks.extend(
        ObservationBlock.seal("workspace-file", _workspace_text(path, content))
        for path, content in canonical_files
    )
    if prior_public_receipt is not None:
        if not isinstance(prior_public_receipt, Mapping):
            raise TypeError("prior receipt projection must be a mapping")
        blocks.append(
            ObservationBlock.seal(
                "prior-public-receipt",
                _json(dict(prior_public_receipt)),
            )
        )
    return ObservationFrame.seal(blocks, (affordance,))


def _one_file_patch(path: str, before: str, after: str) -> str:
    """Create one strict git-style unified diff from a public supervision pair."""

    canonical = _canonical_workspace_path(path)
    if not isinstance(before, str) or not isinstance(after, str) or before == after:
        raise ValueError("public prior must provide two distinct source strings")
    if not before.endswith("\n") or not after.endswith("\n"):
        raise ValueError("public prior sources must end in a newline")
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{canonical}",
            tofile=f"b/{canonical}",
            lineterm="\n",
        )
    )
    if not body.startswith(f"--- a/{canonical}\n+++ b/{canonical}\n"):
        raise RuntimeError("difflib did not produce a one-file unified diff")
    return f"diff --git a/{canonical} b/{canonical}\n{body}"


def _bootstrap_examples(
    example_type: type,
    affordance: ToolAffordance,
) -> tuple[object, ...]:
    """Translate the frozen public prior without importing any V9 target or gold."""

    from experiments.r4_integrated.public_coding_prior_v1 import (
        PUBLIC_CODING_PRIOR_V1_SHA256,
        public_coding_prior_v1,
    )

    prior = public_coding_prior_v1()
    if len(prior) != PUBLIC_PRIOR_CASES:
        raise RuntimeError("public coding prior must contain exactly eight cases")
    examples: list[object] = []
    for index, case in enumerate(prior):
        training_path = "solution.py"
        training_frame = _frame(
            issue=case.public_prompt,
            workspace_files={training_path: case.starter_source.encode("utf-8")},
            affordance=affordance,
        )
        patch = _one_file_patch(training_path, case.starter_source, case.successful_source)
        action = {"arguments": {"patch": patch}, "name": SUBMIT_PATCH}
        examples.append(
            example_type(
                prompt=prompt_for_frame(training_frame),
                output=f"<tool>{_json(action)}</tool>",
                provenance_id=(
                    f"public-coding-prior-v1:{PUBLIC_CODING_PRIOR_V1_SHA256}:"
                    f"{index}:{_text_sha256(case.public_prompt)}"
                ),
            )
        )
    return tuple(examples)


def _bootstrap_root(examples: Sequence[object]) -> str:
    records = []
    for example in examples:
        records.append(
            {
                "output_sha256": _text_sha256(example.output),
                "prompt_sha256": _text_sha256(example.prompt),
                "provenance_id": example.provenance_id,
            }
        )
    return _digest("continual-coding-v9-bootstrap-root-v1", records)


@dataclass(frozen=True, slots=True)
class _ArmPolicy:
    name: str
    plasticity: bool
    ablation: str


_ARM_POLICIES = (
    _ArmPolicy("treatment", True, "none"),
    _ArmPolicy("no_update", False, "none"),
    _ArmPolicy("contrast_off", True, "contrast-off"),
)


def _grow_arms(
    affordance: ToolAffordance,
) -> tuple[
    dict[str, object],
    dict[str, object],
    str,
    tuple[object, ...],
]:
    """Independently grow all causal arms from one exact public bootstrap root."""

    from humanbrain.causal_sequence_pointer import (
        CausalSequenceAblation,
        CausalSequenceConfig,
    )
    from humanbrain.neuronal_tool_model import (
        NeuronalToolModel,
        NeuronalToolTrainingExample,
    )

    examples = _bootstrap_examples(NeuronalToolTrainingExample, affordance)
    bootstrap_root = _bootstrap_root(examples)
    base = CausalSequenceConfig(
        max_contrast_output_bytes=MAX_NEW_TOKENS,
        max_observation_features=MAX_OBSERVATION_FEATURES,
        max_roles=MAX_LEXICAL_ROLES,
    )
    enum_by_value = {item.value: item for item in CausalSequenceAblation}
    configs = {
        policy.name: replace(
            base,
            plasticity=policy.plasticity,
            ablation=enum_by_value[policy.ablation],
        )
        for policy in _ARM_POLICIES
    }
    models = {
        name: NeuronalToolModel.grow(examples, config=config) for name, config in configs.items()
    }
    if set(models) != {policy.name for policy in _ARM_POLICIES}:
        raise RuntimeError("V9 did not grow every required independent causal arm")
    config_records = {name: _audit_value(config) for name, config in configs.items()}
    return models, config_records, bootstrap_root, examples


class _FixtureReceipt(Protocol):
    task_commitment_sha256: str
    sequence: int
    prior_receipt_sha256: str
    action: str
    action_sha256: str
    evaluator_sha256: str
    conclusion: str
    feedback: str
    sandbox_evaluator_reached: bool
    receipt_sha256: str

    def record(self) -> dict[str, object]: ...

    def authenticate(self) -> None: ...


class _SubmitPatch(Protocol):
    def __call__(
        self,
        task_commitment_sha256: str,
        action: Mapping[str, object],
        prior_receipt: _FixtureReceipt | None = None,
    ) -> _FixtureReceipt: ...


class V9ExecutionAdapter:
    """Bind one neuronal action to one task/frame/receipt transition exactly once."""

    def __init__(
        self,
        *,
        task_commitment_sha256: str,
        frame_sha256: str,
        prior_fixture_receipt: _FixtureReceipt | None,
        submit_patch: _SubmitPatch,
        learner_receipt_record: Callable[[_FixtureReceipt], Mapping[str, object]],
        candidate_rejection: type[Exception],
    ) -> None:
        if not isinstance(candidate_rejection, type) or not issubclass(
            candidate_rejection, Exception
        ):
            raise TypeError("candidate_rejection must be an exception type")
        self._task_commitment_sha256 = task_commitment_sha256
        self._frame_sha256 = frame_sha256
        self._prior_fixture_receipt = prior_fixture_receipt
        self._submit_patch = submit_patch
        self._learner_receipt_record = learner_receipt_record
        self._candidate_rejection = candidate_rejection
        self._executed = False
        self._patch_validator_reached = False
        self.fixture_receipt: _FixtureReceipt | None = None
        self.submitted_patch: str | None = None

    @property
    def executed(self) -> bool:
        return self._executed

    @property
    def patch_validator_reached(self) -> bool:
        return self._patch_validator_reached

    def execute(self, action: ToolAction) -> ExecutionReceipt:
        if self._executed:
            raise RuntimeError("V9 execution adapter is exact-once")
        self._executed = True
        if action.frame_sha256 != self._frame_sha256:
            raise ValueError("tool action is not bound to the current V9 frame")
        if action.tool_name != SUBMIT_PATCH:
            raise ValueError("V9 adapter received an unavailable tool")
        arguments = json.loads(action.arguments_json)
        if set(arguments) != {"patch"} or not isinstance(arguments["patch"], str):
            return self._local_failure(action, "submit_patch requires exactly one string patch")
        patch = arguments["patch"]
        if not patch:
            return self._local_failure(action, "submit_patch patch must not be empty")

        self.submitted_patch = patch
        self._patch_validator_reached = True
        try:
            fixture_receipt = self._submit_patch(
                self._task_commitment_sha256,
                {"patch": patch},
                prior_receipt=self._prior_fixture_receipt,
            )
        except self._candidate_rejection:
            return self._local_failure(action, "candidate patch was rejected")
        fixture_receipt.authenticate()
        if fixture_receipt.task_commitment_sha256 != self._task_commitment_sha256:
            raise ValueError("fixture receipt is bound to another V9 task")
        expected_sequence = (
            1 if self._prior_fixture_receipt is None else self._prior_fixture_receipt.sequence + 1
        )
        if fixture_receipt.sequence != expected_sequence:
            raise ValueError("fixture receipt sequence is not the next action")
        if (
            self._prior_fixture_receipt is not None
            and fixture_receipt.prior_receipt_sha256 != self._prior_fixture_receipt.receipt_sha256
        ):
            raise ValueError("fixture receipt is not bound to the prior revision")
        if fixture_receipt.action != SUBMIT_PATCH:
            raise ValueError("fixture receipt is bound to another action")
        if type(fixture_receipt.sandbox_evaluator_reached) is not bool:
            raise ValueError("fixture receipt has invalid evaluator-reach telemetry")
        public = dict(self._learner_receipt_record(fixture_receipt))
        conclusion = fixture_receipt.conclusion
        if conclusion not in {"success", "failure", "action-required"}:
            raise ValueError("fixture returned an unknown public conclusion")
        if (
            set(public) != {"conclusion", "feedback"}
            or public.get("conclusion") != conclusion
            or public.get("feedback") != fixture_receipt.feedback
        ):
            raise ValueError("fixture public receipt projection diverged")
        self.fixture_receipt = fixture_receipt
        observations = (
            ObservationBlock.seal("public-conclusion", conclusion),
            ObservationBlock.seal("public-feedback", fixture_receipt.feedback),
        )
        return ExecutionReceipt.seal(
            action,
            accepted=True,
            succeeded=conclusion in {"success", "action-required"},
            observations=observations,
        )

    @staticmethod
    def _local_failure(action: ToolAction, feedback: str) -> ExecutionReceipt:
        return ExecutionReceipt.seal(
            action,
            accepted=False,
            succeeded=False,
            observations=(
                ObservationBlock.seal("public-conclusion", "failure"),
                ObservationBlock.seal("public-feedback", feedback),
            ),
        )


@dataclass(slots=True)
class _TaskRun:
    arm: str
    task_commitment_sha256: str
    instance_id: str
    issue_sha256: str
    attempts: list[dict[str, object]]
    submitted_patches: list[str]
    fixture_receipts: list[_FixtureReceipt]
    episode_receipts: list[_FixtureReceipt | None]
    terminal_action: str
    terminal_conclusion: str
    budget_exhausted: bool
    posthoc_stage: str | None = None
    posthoc_passed: bool | None = None
    posthoc_expected_action_matches: tuple[bool, ...] | None = None
    posthoc_winning_attempt: int | None = None

    def public_record(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "attempts": self.attempts,
            "budget_exhausted": self.budget_exhausted,
            "instance_id": self.instance_id,
            "issue_sha256": self.issue_sha256,
            "posthoc_passed": self.posthoc_passed,
            "posthoc_expected_action_matches": self.posthoc_expected_action_matches,
            "posthoc_stage": self.posthoc_stage,
            "posthoc_winning_attempt": self.posthoc_winning_attempt,
            "task_commitment_sha256": self.task_commitment_sha256,
            "terminal_action_sha256": _text_sha256(self.terminal_action),
            "terminal_conclusion": self.terminal_conclusion,
        }


def _compact_model_trace(model: object) -> dict[str, object] | None:
    trace = getattr(model, "last_trace", None)
    if trace is None:
        return None
    return {
        "inventory_cardinality": trace.inventory_cardinality,
        "inventory_sha256": trace.inventory_sha256,
        "materialization_error": trace.materialization_error,
        "materialized_output_sha256": trace.materialized_output_sha256,
        "pointer_output_sha256": trace.pointer_output_sha256,
        "query_sha256": trace.query_sha256,
        "selected_stop": trace.selected_stop,
        "sha256": trace.sha256,
    }


def _compact_model_update(model: object) -> dict[str, object] | None:
    update = getattr(model, "last_update", None)
    if update is None:
        return None
    return {
        "consequence_sha256": update.consequence_sha256,
        "polarity": update.polarity,
        "prior_training_digest": update.prior_training_digest,
        "provenance_sha256": update.provenance_sha256,
        "query_sha256": update.query_sha256,
        "training_digest": update.training_digest,
    }


def _attempt_record(
    *,
    arm: str,
    attempt: int,
    frame: ObservationFrame,
    result: EpisodeResult,
    adapter: V9ExecutionAdapter,
    model: object,
) -> dict[str, object]:
    trace = result.generation_trace
    raw_output = trace.raw_output
    model_trace = _compact_model_trace(model)
    decision: dict[str, object]
    if isinstance(result.decision, StopDecision):
        action_kind = "neutral-stop" if result.resolution == "neutral" else "invalid-generation"
        decision = {
            "action_kind": action_kind,
            "kind": "stop",
            "reason_sha256": _text_sha256(result.decision.reason),
            "reason_utf8_bytes": len(result.decision.reason.encode("utf-8")),
            "sha256": result.decision.sha256,
        }
    else:
        arguments = json.loads(result.decision.arguments_json)
        patch = arguments.get("patch")
        decision = {
            "action_kind": "submit_patch",
            "arguments_sha256": _text_sha256(result.decision.arguments_json),
            "kind": "tool",
            "patch_utf8_bytes": len(patch.encode("utf-8")) if isinstance(patch, str) else None,
            "sha256": result.decision.sha256,
            "tool_name": result.decision.tool_name,
        }
    fixture = adapter.fixture_receipt
    physical_non_stop = trace.selected_stop is False
    pointer_materialized = (
        physical_non_stop
        and model_trace is not None
        and model_trace["materialization_error"] is None
        and model_trace["materialized_output_sha256"] is not None
    )
    canonical_tool_action = isinstance(result.decision, ToolAction)
    return {
        "arm": arm,
        "attempt": attempt,
        "decision": decision,
        "episode_sha256": result.sha256,
        "fixture_receipt_sha256": None if fixture is None else fixture.receipt_sha256,
        "frame_sha256": frame.sha256,
        "generation_trace": {
            "max_new_tokens": trace.max_new_tokens,
            "model_query_sha256": trace.model_query_sha256,
            "prompt_sha256": trace.prompt_sha256,
            "raw_output_sha256": _text_sha256(raw_output),
            "raw_output_utf8_bytes": len(raw_output.encode("utf-8")),
            "selected_stop": trace.selected_stop,
            "sha256": trace.sha256,
        },
        "capability": {
            "canonical_tool_action_admitted": canonical_tool_action,
            "fixture_receipt_reached": fixture is not None,
            "patch_validator_reached": adapter.patch_validator_reached,
            "physical_non_stop": physical_non_stop,
            "pointer_materialized": pointer_materialized,
            "sandbox_evaluator_reached": (
                False if fixture is None else fixture.sandbox_evaluator_reached
            ),
        },
        "learning_applied": result.learning_applied,
        "model_digest": result.model_digest,
        "model_trace": model_trace,
        "model_update": _compact_model_update(model),
        "prior_model_digest": result.prior_model_digest,
        "resolution": result.resolution,
    }


def _failure_distribution(runs: Sequence[_TaskRun]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for policy in _ARM_POLICIES:
        arm_runs = [run for run in runs if run.arm == policy.name]
        terminals = Counter(run.terminal_conclusion for run in arm_runs)
        resolutions = Counter(
            str(attempt["resolution"]) for run in arm_runs for attempt in run.attempts
        )
        decisions = Counter(
            str(attempt["decision"]["action_kind"]) for run in arm_runs for attempt in run.attempts
        )
        materialization_errors = Counter(
            str(attempt["model_trace"]["materialization_error"])
            for run in arm_runs
            for attempt in run.attempts
            if attempt["model_trace"] is not None
            and attempt["model_trace"]["materialization_error"] is not None
            and attempt["model_trace"]["selected_stop"] is False
        )
        result[policy.name] = {
            "attempt_decisions": dict(sorted(decisions.items())),
            "attempt_resolutions": dict(sorted(resolutions.items())),
            "materialization_errors": dict(sorted(materialization_errors.items())),
            "terminal_conclusions": dict(sorted(terminals.items())),
            "total_attempts": sum(len(run.attempts) for run in arm_runs),
            "total_tasks": len(arm_runs),
        }
    return result


def _source_custody() -> dict[str, object]:
    paths = (
        RUNNER_PATH,
        FIXTURE_PATH,
        FIXTURE_MANIFEST_PATH,
        MODEL_PATH,
        TOOL_CORE_PATH,
        SEQUENCE_MODEL_PATH,
        PRIOR_PATH,
        SANDBOX_WORKER_PATH,
        SANDBOX_PATH,
    )
    records = {path.relative_to(REPO_ROOT).as_posix(): _file_sha256(path) for path in paths}
    missing = sorted(
        path for path, expected in EXPECTED_DEPENDENCY_SHA256.items() if expected is None
    )
    if missing:
        raise RuntimeError("V9 dependency freeze hashes are not installed: " + ", ".join(missing))
    mismatches = {
        path: {"actual": records[path], "expected": expected}
        for path, expected in EXPECTED_DEPENDENCY_SHA256.items()
        if records.get(path) != expected
    }
    if mismatches:
        raise RuntimeError("V9 dependency source custody changed: " + _json(mismatches))
    return {
        "files": records,
        "sha256": _digest("continual-coding-v9-source-custody-v1", records),
    }


def _write_exclusive(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_terminal(path: Path, record: Mapping[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            prefix=f"{path.name}.",
            suffix=".tmp",
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="reserve the one V9 artifact and run all three arms",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="absent output artifact path (default: frozen benchmark path)",
    )
    return parser.parse_args(argv)


# Fixture-specific orchestration is kept below this line so public task naming,
# receipt projection, and the single posthoc scorer cannot seep into cognition.


@dataclass(frozen=True, slots=True)
class _PublicSetup:
    tasks: tuple[object, ...]
    affordance: ToolAffordance
    max_patch_submissions: int
    preflight_record: dict[str, object]


@dataclass(frozen=True, slots=True)
class _PromptExample:
    prompt: str
    output: str
    provenance_id: str


def _learner_receipt_record(receipt: _FixtureReceipt) -> dict[str, object]:
    """Project only ordinary public consequence text into neuronal cognition."""

    receipt.authenticate()
    return {
        "conclusion": receipt.conclusion,
        "feedback": receipt.feedback,
    }


def _authenticate_execution_substrate_record(
    record: Mapping[str, object],
    expected: Mapping[str, str],
) -> dict[str, str]:
    required = {
        "python_executable_sha256",
        "repository_worker_sha256",
        "simple_swe_sandbox_sha256",
    }
    if (
        not isinstance(record, Mapping)
        or not isinstance(expected, Mapping)
        or set(record) != required
        or set(expected) != required
        or any(record[key] != expected[key] for key in required)
    ):
        raise RuntimeError("V9 execution substrate custody changed")
    return {key: expected[key] for key in sorted(required)}


def _load_public_setup(output_path: Path) -> _PublicSetup:
    """Authenticate only public V9 material; never import a model or private scorer."""

    from experiments.r4_integrated.continual_toy_swe_tasks_v9 import (
        GLOBAL_ACTIONS,
        MAX_PATCH_SUBMISSIONS,
        PYTHON_EXECUTABLE_SHA256,
        REPOSITORY_WORKER_SHA256,
        SIMPLE_SWE_SANDBOX_SHA256,
        SOURCE_PRIOR_CASE_COUNT,
        SOURCE_PRIOR_CORPUS_SHA256,
        TASK_COUNT,
        TOOL_CONTRACT_SHA256,
        TRAIN_STREAM_CUSTODY_SHA256,
        TRAIN_STREAM_SHA256,
        global_tool_contract_v9,
        learner_workspace_files_v9,
        preflight_execution_substrate_v9,
        public_train_tasks_v9,
        runner_mount_allowlist_v9,
        runner_writable_source_allowlist_v9,
        train_stream_custody_v9,
        train_stream_sha256_v9,
    )
    from experiments.r4_integrated.continual_toy_swe_tasks_v9 import (
        SUBMIT_PATCH as FIXTURE_SUBMIT_PATCH,
    )
    from experiments.r4_integrated.public_coding_prior_v1 import (
        PUBLIC_CODING_PRIOR_V1_SHA256,
    )

    if output_path.exists():
        raise FileExistsError(f"V9 artifact already exists: {output_path}")
    if (
        FIXTURE_SUBMIT_PATCH != SUBMIT_PATCH
        or tuple(GLOBAL_ACTIONS) != (SUBMIT_PATCH,)
        or MAX_PATCH_SUBMISSIONS != 3
    ):
        raise RuntimeError("V9 action surface or retry budget changed")
    if (
        SOURCE_PRIOR_CASE_COUNT != PUBLIC_PRIOR_CASES
        or SOURCE_PRIOR_CORPUS_SHA256 != PUBLIC_CODING_PRIOR_V1_SHA256
    ):
        raise RuntimeError("V9 fixture and frozen public coding prior diverged")

    execution_substrate = _authenticate_execution_substrate_record(
        preflight_execution_substrate_v9(),
        {
            "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
            "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
            "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        },
    )

    contract = global_tool_contract_v9()
    affordance = _submit_patch_affordance(contract)
    tasks = tuple(public_train_tasks_v9())
    if len(tasks) != TASK_COUNT or not tasks:
        raise RuntimeError("V9 public TRAIN stream size changed")
    if train_stream_sha256_v9() != TRAIN_STREAM_SHA256:
        raise RuntimeError("V9 public TRAIN stream custody changed")
    stream_custody = train_stream_custody_v9()
    if stream_custody.get("train_stream_custody_sha256") != TRAIN_STREAM_CUSTODY_SHA256:
        raise RuntimeError("V9 TRAIN custody record changed")

    prior_examples = _bootstrap_examples(_PromptExample, affordance)
    prior_cardinalities = [
        _lexical_inventory_cardinality(example.prompt) for example in prior_examples
    ]

    task_records: list[dict[str, object]] = []
    task_cardinalities: list[int] = []
    for task in tasks:
        task.authenticate()
        if task.tool_contract_sha256 != TOOL_CONTRACT_SHA256:
            raise RuntimeError("V9 task is bound to another tool contract")
        mounted = tuple(runner_mount_allowlist_v9(task.task_commitment_sha256))
        writable = tuple(runner_writable_source_allowlist_v9(task.task_commitment_sha256))
        workspace = learner_workspace_files_v9(task.task_commitment_sha256)
        if tuple(workspace) != mounted or not set(writable).issubset(mounted):
            raise RuntimeError("V9 detached workspace exceeded its public allowlists")
        if any(path.startswith("tests/") for path in writable):
            raise RuntimeError("V9 public tests must remain immutable")
        frame = _frame(
            issue=task.issue,
            workspace_files=workspace,
            affordance=affordance,
        )
        lexical_cardinality = _lexical_inventory_cardinality(prompt_for_frame(frame))
        task_cardinalities.append(lexical_cardinality)
        task_records.append(
            {
                "frame_sha256": frame.sha256,
                "instance_id": task.instance_id,
                "issue_sha256": _text_sha256(task.issue),
                "lexical_inventory_cardinality": lexical_cardinality,
                "mount_allowlist": mounted,
                "repository_snapshot_sha256": task.repository_snapshot_sha256,
                "task_commitment_sha256": task.task_commitment_sha256,
                "writable_source_allowlist": writable,
                "workspace_files": {
                    path: sha256(content).hexdigest() for path, content in workspace.items()
                },
            }
        )

    max_observed_cardinality = max((*prior_cardinalities, *task_cardinalities))
    if max_observed_cardinality > MAX_LEXICAL_ROLES:
        raise RuntimeError("V9 public prompts exceed the frozen lexical inventory capacity")
    source_custody = _source_custody()
    record = {
        "artifact_path": str(output_path.resolve()),
        "execution_substrate": execution_substrate,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_observation_features": MAX_OBSERVATION_FEATURES,
        "max_patch_submissions": MAX_PATCH_SUBMISSIONS,
        "public_prior_cases": PUBLIC_PRIOR_CASES,
        "public_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "lexical_inventory_capacity": {
            "capacity": MAX_LEXICAL_ROLES,
            "headroom": MAX_LEXICAL_ROLES - max_observed_cardinality,
            "max_observed": max_observed_cardinality,
            "max_public_prior": max(prior_cardinalities),
            "max_public_task_initial_frame": max(task_cardinalities),
            "policy": "first-occurrence-ascii-identifier-lexemes",
        },
        "schema": f"{SCHEMA}-preflight",
        "source_custody": source_custody,
        "task_count": len(tasks),
        "task_public_custody": task_records,
        "tool_contract": contract,
        "tool_contract_sha256": TOOL_CONTRACT_SHA256,
        "train_stream_custody": stream_custody,
        "train_stream_custody_sha256": TRAIN_STREAM_CUSTODY_SHA256,
        "train_stream_sha256": TRAIN_STREAM_SHA256,
    }
    record["sha256"] = _digest("continual-coding-v9-preflight-v1", record)
    return _PublicSetup(tasks, affordance, MAX_PATCH_SUBMISSIONS, record)


def preflight(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    """Validate public/source custody without model construction or execution."""

    return _load_public_setup(Path(output_path)).preflight_record


def _run_task(
    *,
    arm: str,
    task: object,
    model: object,
    affordance: ToolAffordance,
    max_patch_submissions: int,
    workspace_files: Callable[[str], Mapping[str, bytes]],
    submit_patch: _SubmitPatch,
    candidate_rejection: type[Exception],
) -> _TaskRun:
    prior: _FixtureReceipt | None = None
    attempts: list[dict[str, object]] = []
    patches: list[str] = []
    receipts: list[_FixtureReceipt] = []
    episode_receipts: list[_FixtureReceipt | None] = []
    terminal_action = "<stop>unattempted</stop>"
    terminal_conclusion = "unattempted"

    for attempt in range(1, max_patch_submissions + 1):
        prior_public = None if prior is None else _learner_receipt_record(prior)
        frame = _frame(
            issue=task.issue,
            workspace_files=workspace_files(task.task_commitment_sha256),
            affordance=affordance,
            prior_public_receipt=prior_public,
        )
        adapter = V9ExecutionAdapter(
            task_commitment_sha256=task.task_commitment_sha256,
            frame_sha256=frame.sha256,
            prior_fixture_receipt=prior,
            submit_patch=submit_patch,
            learner_receipt_record=_learner_receipt_record,
            candidate_rejection=candidate_rejection,
        )
        learner = PersistentToolLearner(
            model,
            adapter,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        result = learner.episode(frame)
        attempts.append(
            _attempt_record(
                arm=arm,
                attempt=attempt,
                frame=frame,
                result=result,
                adapter=adapter,
                model=model,
            )
        )
        episode_receipts.append(adapter.fixture_receipt)

        if isinstance(result.decision, StopDecision):
            if adapter.executed:
                raise RuntimeError("STOP or generation failure crossed the V9 executor")
            if result.resolution == "neutral":
                terminal_action = "<stop>"
                terminal_conclusion = "stop"
                break
            terminal_action = f"<generation-failure>{result.decision.reason}</generation-failure>"
            terminal_conclusion = "generation-failure"
            continue
        if adapter.submitted_patch is not None:
            patches.append(adapter.submitted_patch)
            terminal_action = adapter.submitted_patch
        if adapter.fixture_receipt is None:
            terminal_conclusion = "failure"
            continue
        prior = adapter.fixture_receipt
        receipts.append(prior)
        terminal_conclusion = prior.conclusion
        if prior.conclusion == "success":
            break
        # Both an action-required review and an ordinary failure become public
        # causal evidence.  The next complete patch is evaluated against the
        # original snapshot and authenticated by this receipt chain.

    return _TaskRun(
        arm=arm,
        task_commitment_sha256=task.task_commitment_sha256,
        instance_id=task.instance_id,
        issue_sha256=_text_sha256(task.issue),
        attempts=attempts,
        submitted_patches=patches,
        fixture_receipts=receipts,
        episode_receipts=episode_receipts,
        terminal_action=terminal_action,
        terminal_conclusion=terminal_conclusion,
        budget_exhausted=(
            len(attempts) == max_patch_submissions
            and terminal_conclusion not in {"success", "stop"}
        ),
    )


def _rate(successes: int, total: int) -> float | None:
    return None if total == 0 else successes / total


def _public_prior_retention_probe(
    model: object,
    examples: Sequence[object],
) -> dict[str, object]:
    """Measure exact public-prior retention with inference + neutral closure only."""

    checkpoint = model.training_digest
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        before = model.training_digest
        emission = model.infer(
            example.prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            stop_text=MODEL_STOP_TEXT,
        )
        if model.training_digest != before:
            raise RuntimeError("public retention inference mutated the neuronal model")
        trace_record = _compact_model_trace(model)
        model.resolve_neutral(emission.query_sha256)
        if model.training_digest != before:
            raise RuntimeError("neutral public retention closure mutated the neuronal model")
        expected_sha256 = _text_sha256(example.output)
        observed_sha256 = _text_sha256(emission.raw_output)
        rows.append(
            {
                "expected_output_sha256": expected_sha256,
                "index": index,
                "matched": not emission.selected_stop and observed_sha256 == expected_sha256,
                "observed_output_sha256": observed_sha256,
                "query_sha256": emission.query_sha256,
                "selected_stop": emission.selected_stop,
                "trace": trace_record,
            }
        )
    if model.training_digest != checkpoint:
        raise RuntimeError("public retention probe changed the arm checkpoint")
    passed = sum(row["matched"] is True for row in rows)
    return {
        "checkpoint_digest": checkpoint,
        "passed": passed,
        "pass_rate": _rate(passed, len(rows)),
        "probes": rows,
        "total": len(rows),
    }


_EXPECTED_POSTHOC_STAGE_SEQUENCE = (
    "direct-repository-relation",
    "issue-doc-test-implementation-composition",
    "direct-repository-relation",
    "issue-doc-test-implementation-composition",
    "consequence-guided-revision",
    "direct-repository-relation",
    "issue-doc-test-implementation-composition",
    "genuinely-ambiguous-stop",
    "direct-repository-relation",
    "consequence-guided-revision",
    "issue-doc-test-implementation-composition",
    "genuinely-ambiguous-stop",
)
_CAPABILITY_KEYS = (
    "physical_non_stop",
    "pointer_materialized",
    "canonical_tool_action_admitted",
    "patch_validator_reached",
    "fixture_receipt_reached",
    "sandbox_evaluator_reached",
)


def _changed_action_recovery(run: _TaskRun) -> bool:
    winning_attempt = run.posthoc_winning_attempt
    if run.posthoc_passed is not True or winning_attempt is None or winning_attempt <= 1:
        return False
    winner = run.attempts[winning_attempt - 1]["decision"]
    if winner["action_kind"] != "submit_patch":
        return False
    winner_arguments = winner.get("arguments_sha256")
    earlier_arguments = [
        attempt["decision"].get("arguments_sha256")
        for attempt in run.attempts[: winning_attempt - 1]
        if attempt["decision"]["action_kind"] == "submit_patch"
    ]
    return bool(
        earlier_arguments
        and winner_arguments is not None
        and any(item != winner_arguments for item in earlier_arguments)
    )


def _capability_counts(runs: Sequence[_TaskRun]) -> dict[str, object]:
    attempts = [attempt for run in runs for attempt in run.attempts]
    expected_stop = [run.posthoc_stage == "genuinely-ambiguous-stop" for run in runs]
    chose_stop = [run.terminal_conclusion == "stop" for run in runs]
    correct_stop = sum(
        expected == chosen for expected, chosen in zip(expected_stop, chose_stop, strict=True)
    )
    return {
        **{
            key: sum(attempt["capability"][key] is True for attempt in attempts)
            for key in _CAPABILITY_KEYS
        },
        "attempts": len(attempts),
        "changed_action_recovery": sum(_changed_action_recovery(run) for run in runs),
        "semantic_pass": sum(
            run.posthoc_passed is True and run.posthoc_stage != "genuinely-ambiguous-stop"
            for run in runs
        ),
        "stop_accuracy": {
            "accuracy": _rate(correct_stop, len(runs)),
            "correct": correct_stop,
            "expected_stop": sum(expected_stop),
            "false_negative": sum(
                expected and not chosen
                for expected, chosen in zip(expected_stop, chose_stop, strict=True)
            ),
            "false_positive": sum(
                chosen and not expected
                for expected, chosen in zip(expected_stop, chose_stop, strict=True)
            ),
            "selected_stop": sum(chose_stop),
            "tasks": len(runs),
        },
        "tasks": len(runs),
    }


def _retention_summary(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    return {
        "expected_exact": PUBLIC_PRIOR_CASES,
        "post": {
            "pass_rate": after["pass_rate"],
            "passed": after["passed"],
            "total": after["total"],
        },
        "post_minus_pre": int(after["passed"]) - int(before["passed"]),
        "pre": {
            "pass_rate": before["pass_rate"],
            "passed": before["passed"],
            "total": before["total"],
        },
    }


def _summary_metrics(
    runs: Sequence[_TaskRun],
    pre_retention: Mapping[str, Mapping[str, object]],
    post_retention: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if any(run.posthoc_passed is None or run.posthoc_stage is None for run in runs):
        raise RuntimeError("posthoc scoring must finish before metric aggregation")

    by_arm: dict[str, dict[str, object]] = {}
    for policy in _ARM_POLICIES:
        arm_runs = [run for run in runs if run.arm == policy.name]
        stage_sequence = tuple(str(run.posthoc_stage) for run in arm_runs)
        if stage_sequence != _EXPECTED_POSTHOC_STAGE_SEQUENCE:
            raise RuntimeError("V9 posthoc stage order or cardinality changed")
        stage_curve = {
            stage: {
                **_capability_counts([run for run in arm_runs if run.posthoc_stage == stage]),
                "pass_rate": _rate(
                    sum(
                        run.posthoc_passed is True for run in arm_runs if run.posthoc_stage == stage
                    ),
                    sum(run.posthoc_stage == stage for run in arm_runs),
                ),
                "passed": sum(
                    run.posthoc_passed is True for run in arm_runs if run.posthoc_stage == stage
                ),
            }
            for stage in dict.fromkeys(_EXPECTED_POSTHOC_STAGE_SEQUENCE)
        }
        passed = sum(run.posthoc_passed is True for run in arm_runs)
        by_arm[policy.name] = {
            "capability_curve": _capability_counts(arm_runs),
            "pass_rate": _rate(passed, len(arm_runs)),
            "passed": passed,
            "public_prior_retention": _retention_summary(
                pre_retention[policy.name], post_retention[policy.name]
            ),
            "stage_curve": stage_curve,
            "tasks": len(arm_runs),
        }

    treatment = by_arm["treatment"]["pass_rate"]
    no_update = by_arm["no_update"]["pass_rate"]
    contrast_off = by_arm["contrast_off"]["pass_rate"]
    causal = {
        "treatment_minus_contrast_off": (
            None if treatment is None or contrast_off is None else treatment - contrast_off
        ),
        "treatment_minus_no_update": (
            None if treatment is None or no_update is None else treatment - no_update
        ),
    }
    return {"arms": by_arm, "causal_deltas": causal}


def _posthoc_score(runs: Sequence[_TaskRun]) -> None:
    """Open the narrow private scorer only after all public arm streams finish."""

    required_arms = {policy.name for policy in _ARM_POLICIES}
    seen_arms = {run.arm for run in runs}
    counts = Counter(run.arm for run in runs)
    arm_streams = {
        arm: tuple(run.task_commitment_sha256 for run in runs if run.arm == arm)
        for arm in required_arms
    }
    if (
        seen_arms != required_arms
        or len(set(counts.values())) != 1
        or not next(iter(arm_streams.values()), ())
        or len(set(arm_streams.values())) != 1
    ):
        raise RuntimeError("private scoring phase opened before every arm stream completed")

    # The exact scorer call is intentionally the only import/use of evaluator
    # knowledge in this runner.  It returns labels and booleans, never expected
    # patches or evaluator bodies.
    from experiments.r4_integrated.continual_toy_swe_tasks_v9 import (
        INVALID_ACTION,
        STOP,
        posthoc_score_terminal_actions_v9,
        seal_action_summary_v9,
    )
    from experiments.r4_integrated.continual_toy_swe_tasks_v9 import (
        SUBMIT_PATCH as FIXTURE_SUBMIT_PATCH,
    )

    for run in runs:
        if len(run.attempts) != len(run.episode_receipts):
            raise RuntimeError("V9 episode/receipt scoring custody diverged")
        completed: list[object] = []
        for attempt, receipt in zip(run.attempts, run.episode_receipts, strict=True):
            action_kind = attempt["decision"]["action_kind"]
            if action_kind == "submit_patch":
                action = FIXTURE_SUBMIT_PATCH
            elif action_kind == "neutral-stop":
                action = STOP
            elif action_kind == "invalid-generation":
                action = INVALID_ACTION
            else:
                raise RuntimeError("unknown V9 episode action kind")
            if action_kind != "submit_patch" and receipt is not None:
                raise RuntimeError("non-patch V9 episode unexpectedly owns a receipt")
            completed.append(
                seal_action_summary_v9(
                    run.task_commitment_sha256,
                    int(attempt["attempt"]),
                    action,
                    receipt,
                )
            )
        completed_actions = tuple(completed)
        score = posthoc_score_terminal_actions_v9(
            run.task_commitment_sha256,
            completed_actions,
            all_arm_streams_finished=True,
        )
        expected_keys = {
            "expected_action_matches",
            "stage",
            "terminal_pass",
            "winning_attempt",
        }
        if not isinstance(score, Mapping) or set(score) != expected_keys:
            raise RuntimeError("V9 posthoc scorer returned an unsafe or incomplete projection")
        matches = score["expected_action_matches"]
        winning_attempt = score["winning_attempt"]
        if (
            type(score["terminal_pass"]) is not bool
            or not isinstance(score["stage"], str)
            or not isinstance(matches, tuple)
            or not all(type(item) is bool for item in matches)
            or len(matches) != len(run.attempts)
            or (
                winning_attempt is not None
                and (type(winning_attempt) is not int or not 1 <= winning_attempt <= len(matches))
            )
        ):
            raise RuntimeError("V9 posthoc score fields are invalid")
        run.posthoc_passed = score["terminal_pass"]
        run.posthoc_stage = score["stage"]
        run.posthoc_expected_action_matches = matches
        run.posthoc_winning_attempt = winning_attempt


def run_session(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    """Reserve and run the one public V9 cohort; never called during import/preflight."""

    from experiments.r4_integrated.continual_toy_swe_tasks_v9 import (
        CandidatePatchRejectedV9,
        learner_workspace_files_v9,
        submit_patch_v9,
    )

    output_path = Path(output_path)
    setup = _load_public_setup(output_path)
    models, model_configs, bootstrap_root, bootstrap_examples = _grow_arms(setup.affordance)
    initial_models = {name: model.training_digest for name, model in models.items()}
    if len(set(initial_models.values())) != 1:
        raise RuntimeError("V9 causal arms did not start from one equal learned checkpoint")
    reservation = {
        "artifact_path": str(output_path.resolve()),
        "bootstrap_root_sha256": bootstrap_root,
        "initial_models": initial_models,
        "model_configs": model_configs,
        "preflight_sha256": setup.preflight_record["sha256"],
        "reservation_policy": "absent-only-exclusive-create-before-first-episode",
        "schema": RESERVATION_SCHEMA,
        "status": "RUNNING",
    }
    _write_exclusive(output_path, reservation)

    try:
        pre_retention = {
            policy.name: _public_prior_retention_probe(models[policy.name], bootstrap_examples)
            for policy in _ARM_POLICIES
        }
        if any(
            record["total"] != PUBLIC_PRIOR_CASES or record["passed"] != PUBLIC_PRIOR_CASES
            for record in pre_retention.values()
        ):
            raise RuntimeError("frozen V9 bootstrap no longer replays its eight public priors")

        runs: list[_TaskRun] = []
        for policy in _ARM_POLICIES:
            model = models[policy.name]
            initial_digest = model.training_digest
            for task in setup.tasks:
                runs.append(
                    _run_task(
                        arm=policy.name,
                        task=task,
                        model=model,
                        affordance=setup.affordance,
                        max_patch_submissions=setup.max_patch_submissions,
                        workspace_files=learner_workspace_files_v9,
                        submit_patch=submit_patch_v9,
                        candidate_rejection=CandidatePatchRejectedV9,
                    )
                )
                if not policy.plasticity and model.training_digest != initial_digest:
                    raise RuntimeError("nonplastic V9 arm changed its learned checkpoint")

        post_retention = {
            policy.name: _public_prior_retention_probe(models[policy.name], bootstrap_examples)
            for policy in _ARM_POLICIES
        }
        _posthoc_score(runs)
        result: dict[str, object] = {
            "arms": [
                {
                    "final_model_digest": models[policy.name].training_digest,
                    "initial_model_digest": initial_models[policy.name],
                    "name": policy.name,
                    "plasticity": policy.plasticity,
                    "ablation": policy.ablation,
                }
                for policy in _ARM_POLICIES
            ],
            "bootstrap_root_sha256": bootstrap_root,
            "failure_distribution": _failure_distribution(runs),
            "metrics": _summary_metrics(runs, pre_retention, post_retention),
            "preflight": setup.preflight_record,
            "public_prior_retention": {
                "post": post_retention,
                "pre": pre_retention,
            },
            "runs": [run.public_record() for run in runs],
            "schema": SCHEMA,
            "status": "COMPLETE",
        }
        result["sha256"] = _digest("continual-coding-session-v9-result-v1", result)
        _write_terminal(output_path, result)
        return result
    except Exception as error:
        failed = {
            **reservation,
            "error": {"message": str(error), "type": type(error).__name__},
            "status": "FAILED",
        }
        failed["sha256"] = _digest("continual-coding-session-v9-failure-v1", failed)
        _write_terminal(output_path, failed)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    record = run_session(args.output) if args.run else preflight(args.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
