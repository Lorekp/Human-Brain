"""Unspent variability-aware multi-source Python repair cohort.

The frozen controller grows from exactly two independent successful public
transitions.  A single controller then sees three fresh complete issue/workspace
observations whose target files occupy the first, middle, and last sorted
positions, followed by one genuinely ambiguous observation.  Only after each
one-shot response and neutral consequence resolution may the evaluator reveal
the intended path, exact correction, or executable behavioral check.

The motor remains narrow: emit a live path and the live prefix of one source
while dropping its trailing crashing line.  This cohort tests whether learned
variable event propagation composes with boundary grounding and a neuronal
motor.  It is not evidence of arbitrary Python synthesis or broad language
understanding.

Importing this module is inert.  The sole benchmark entry point requires an
explicit ``--write-artifact`` flag, writes one fixed artifact with exclusive
creation, and refuses to run when that artifact already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueEventTapeError,
    OpaqueWorkspaceFile,
)
from humanbrain.source_relative_action_tape import TapeValence
from humanbrain.variability_multiscale_source_motor_controller import (
    SuccessfulVariabilityMultiscaleSourceMotorExperience,
    VariabilityMultiscaleSourceMotorAblation,
    VariabilityMultiscaleSourceMotorController,
)

ARCHITECTURE_COMMIT = "101edfb86470f0124f0859f782f7f952671f5115"
EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256 = (
    "79344744770ac5805fb94ee8f61a9bc475856307809e35d6168675329e57dfb0"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = Path(__file__).resolve()
_ARTIFACT_PATH = (
    _REPO_ROOT
    / "benchmarks"
    / "r4_integrated_variability_multiscale_multi_source_python_edit_v3.json"
)
_RESULT_NAMESPACE = b"humanbrain.variability-multiscale-python-edit-v3\x00"
_COHORT_NAMESPACE = b"humanbrain.variability-multiscale-python-edit-v3.cohort\x00"
_CUSTODY_NAMESPACE = b"humanbrain.variability-multiscale-python-edit-v3.custody\x00"
_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_SOURCE_CUSTODY_PATTERN = re.compile(
    rb'(EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256\s*=\s*(?:\(\s*)?")'
    rb'[0-9a-f]{64}(")'
)

_EXPECTED_DEPENDENCY_SHA256 = {
    "src/humanbrain/cells.py": (
        "5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6"
    ),
    "src/humanbrain/multiscale_opaque_event_tape.py": (
        "0e608ea64133dc783bb715f6dcedd471da134897a540b440f35283e08dd8899a"
    ),
    "src/humanbrain/resting_glif_race.py": (
        "b78a14ad6f601800c170bbd9e53d6204e992842954f22ea646d5f9ec04accca0"
    ),
    "src/humanbrain/source_relative_action_tape.py": (
        "8bb71cb7bf5ae8e74e6d568a9fd9ccf7ecd2bba0fc275e8a877b9f7b70351331"
    ),
    "src/humanbrain/source_relative_training_compiler.py": (
        "d60acb21fd4b1829569f973a4c37fb4eb2c6facd3322a6eb34431f8afa8b2e0f"
    ),
    "src/humanbrain/variability_aware_segment_binder.py": (
        "03788ed36e7f2c95c5de4dae4f4bb13d610d898b206f86cc397d62a8a48f09e5"
    ),
    "src/humanbrain/variability_multiscale_source_motor_controller.py": (
        "eb7edc7fb7d79989374de08d5ff4e74e2055351f20ec38ad16c8495d32dac0e0"
    ),
}


@dataclass(frozen=True, slots=True)
class PublicVariabilityMultiSourceTask:
    """The complete answer-free inference surface."""

    task_id: str
    issue: bytes
    files: tuple[OpaqueWorkspaceFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be non-empty text")
        if not isinstance(self.issue, bytes) or not self.issue:
            raise ValueError("issue must be non-empty bytes")
        MultiscaleOpaqueEventTape.observe(self.issue, self.files)


@dataclass(frozen=True, slots=True)
class _PrivatePythonOracle:
    intended_path: str | None
    corrected_content: bytes | None
    expected_stdout: bytes | None

    def __post_init__(self) -> None:
        absent = (
            self.intended_path is None,
            self.corrected_content is None,
            self.expected_stdout is None,
        )
        if len(set(absent)) != 1:
            raise ValueError("private oracle fields must be jointly present or absent")


def _json_bytes(record: object) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _digest(namespace: bytes, record: object) -> str:
    return hashlib.sha256(namespace + _json_bytes(record)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_source_sha256() -> str:
    """Return the exact current runner bytes hash without executing it."""

    return _file_sha256(_RUNNER_PATH)


def experiment_source_custody_sha256() -> str:
    """Hash source after masking only its embedded custody hash."""

    source = _RUNNER_PATH.read_bytes()
    normalized, replacement_count = _SOURCE_CUSTODY_PATTERN.subn(
        lambda match: match.group(1) + b"0" * 64 + match.group(2), source
    )
    if replacement_count != 1:
        raise RuntimeError("experiment source custody marker changed")
    return hashlib.sha256(normalized).hexdigest()


def dependency_custody() -> dict[str, object]:
    """Authenticate every frozen neuronal dependency without running a model."""

    actual = {
        relative: _file_sha256(_REPO_ROOT / PurePosixPath(relative))
        for relative in sorted(_EXPECTED_DEPENDENCY_SHA256)
    }
    mismatches = {
        relative: {"actual": actual[relative], "expected": expected}
        for relative, expected in _EXPECTED_DEPENDENCY_SHA256.items()
        if actual[relative] != expected
    }
    if mismatches:
        raise RuntimeError(
            "variability V3 dependency custody changed: "
            + _json_bytes(mismatches).decode("ascii")
        )
    return {
        "architecture_commit": ARCHITECTURE_COMMIT,
        "files": actual,
        "sha256": _digest(_CUSTODY_NAMESPACE, actual),
    }


def _assert_source_custody() -> dict[str, object]:
    custody_sha256 = experiment_source_custody_sha256()
    if custody_sha256 != EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256:
        raise RuntimeError(
            "variability V3 experiment source custody changed: "
            f"expected {EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256}, "
            f"observed {custody_sha256}"
        )
    return {
        "custody_sha256": custody_sha256,
        "source_sha256": experiment_source_sha256(),
    }


def _buggy_source(function_name: str, value: str) -> bytes:
    return (
        f"def {function_name}():\n"
        f'    return "{value}"\n'
        f'print({function_name}(), end="")\n'
        'raise RuntimeError("obsolete execution path")\n'
    ).encode("ascii")


def _correct_source(function_name: str, value: str) -> bytes:
    return (
        f"def {function_name}():\n"
        f'    return "{value}"\n'
        f'print({function_name}(), end="")\n'
    ).encode("ascii")


def _sorted_workspace(
    rows: tuple[tuple[str, bytes], ...],
) -> tuple[OpaqueWorkspaceFile, ...]:
    return tuple(
        OpaqueWorkspaceFile(path, content)
        for path, content in sorted(rows, key=lambda item: item[0])
    )


def _target_workspace(
    *,
    target_path: str,
    function_name: str,
    value: str,
    decoys: tuple[tuple[str, bytes], ...],
) -> tuple[OpaqueWorkspaceFile, ...]:
    return _sorted_workspace(
        (*decoys, (target_path, _buggy_source(function_name, value)))
    )


def _successful_experience(
    *,
    provenance_id: str,
    query: bytes,
    target_path: str,
    function_name: str,
    value: str,
    decoys: tuple[tuple[str, bytes], ...],
) -> SuccessfulVariabilityMultiscaleSourceMotorExperience:
    before = _target_workspace(
        target_path=target_path,
        function_name=function_name,
        value=value,
        decoys=decoys,
    )
    after = tuple(
        OpaqueWorkspaceFile(
            item.path,
            (
                _correct_source(function_name, value)
                if item.path == target_path
                else item.content
            ),
        )
        for item in before
    )
    return SuccessfulVariabilityMultiscaleSourceMotorExperience(
        query=query,
        before_files=before,
        after_files=after,
        provenance_id=provenance_id,
    )


def public_training_experiences() -> tuple[
    SuccessfulVariabilityMultiscaleSourceMotorExperience, ...
]:
    """Exactly two independent successful full-stream transitions."""

    return (
        _successful_experience(
            provenance_id="public-train-heliodor-v3",
            query=b"repair heliodor_worker using existing bytes",
            target_path="citrine/heliodor_worker.py",
            function_name="heliodor_cycle",
            value="settled_heliodor",
            decoys=(
                ("amber/fern_notes.md", b"# fern ledger\nquiet retained prose\n"),
                ("quartz/mica_data.json", b'{"mica": 19, "ready": true}\n'),
                ("zinc/otter_helper.py", b"OTTER_LEVEL = 7\n"),
            ),
        ),
        _successful_experience(
            provenance_id="public-train-selkie-v3",
            query=b"repair selkie_worker using existing bytes",
            target_path="nimbus/selkie_worker.py",
            function_name="selkie_cycle",
            value="settled_selkie",
            decoys=(
                ("brass/iris_config.toml", b"enabled = true\nlimit = 29\n"),
                ("copper/kestrel_helper.py", b"KESTREL_SCALE = 13\n"),
                ("willow/pollen_notes.txt", b"pollen record\nsecond retained row\n"),
            ),
        ),
    )


def public_fresh_tasks() -> tuple[PublicVariabilityMultiSourceTask, ...]:
    """Three positional repair tasks and one equality-ambiguous task."""

    return (
        PublicVariabilityMultiSourceTask(
            task_id="fresh-saffron-first-v3",
            issue=b"repair saffron_worker using existing bytes",
            files=_target_workspace(
                target_path="aurora/saffron_worker.py",
                function_name="saffron_cycle",
                value="luminous_saffron",
                decoys=(
                    ("cedar/velvet_notes.md", b"# velvet index\nretained paragraph\n"),
                    ("pyrope/badger_data.json", b'{"badger": [3, 5, 8]}\n'),
                    ("zephyr/coral_helper.py", b"def coral_value():\n    return 37\n"),
                ),
            ),
        ),
        PublicVariabilityMultiSourceTask(
            task_id="fresh-vireo-middle-v3",
            issue=b"repair vireo_worker using existing bytes",
            files=_target_workspace(
                target_path="opal/vireo_worker.py",
                function_name="vireo_cycle",
                value="luminous_vireo",
                decoys=(
                    ("birch/indigo_notes.txt", b"indigo one\nindigo two\n"),
                    ("frost/marten_config.toml", b'name = "marten"\nweight = 0.625\n'),
                    ("yarrow/paprika_helper.py", b"PAPRIKA_READY = False\n"),
                ),
            ),
        ),
        PublicVariabilityMultiSourceTask(
            task_id="fresh-tamarind-last-v3",
            issue=b"repair tamarind_worker using existing bytes",
            files=_target_workspace(
                target_path="zenith/tamarind_worker.py",
                function_name="tamarind_cycle",
                value="luminous_tamarind",
                decoys=(
                    ("atlas/sorrel_notes.md", b"# sorrel\nlong retained reference\n"),
                    ("cobalt/lynx_config.json", b'{"lynx": "cobalt", "count": 17}\n'),
                    ("lumen/thistle_helper.py", b"THISTLE_SCALE = (2, 6, 10)\n"),
                ),
            ),
        ),
        PublicVariabilityMultiSourceTask(
            task_id="fresh-ambiguous-marble-river-v3",
            issue=b"repair marble_worker or river_worker using existing bytes",
            files=_sorted_workspace(
                (
                    (
                        "dahlia/marble_worker.py",
                        _buggy_source("marble_cycle", "luminous_marble"),
                    ),
                    ("granite/wren_notes.md", b"# wren\nunaltered ambiguity decoy\n"),
                    ("tundra/lotus_config.json", b'{"lotus": true, "depth": 12}\n'),
                    (
                        "xenia/river_worker.py",
                        _buggy_source("river_cycle", "luminous_river"),
                    ),
                )
            ),
        ),
    )


_PRIVATE_ORACLES = {
    "fresh-saffron-first-v3": _PrivatePythonOracle(
        intended_path="aurora/saffron_worker.py",
        corrected_content=_correct_source("saffron_cycle", "luminous_saffron"),
        expected_stdout=b"luminous_saffron",
    ),
    "fresh-vireo-middle-v3": _PrivatePythonOracle(
        intended_path="opal/vireo_worker.py",
        corrected_content=_correct_source("vireo_cycle", "luminous_vireo"),
        expected_stdout=b"luminous_vireo",
    ),
    "fresh-tamarind-last-v3": _PrivatePythonOracle(
        intended_path="zenith/tamarind_worker.py",
        corrected_content=_correct_source("tamarind_cycle", "luminous_tamarind"),
        expected_stdout=b"luminous_tamarind",
    ),
    "fresh-ambiguous-marble-river-v3": _PrivatePythonOracle(
        intended_path=None,
        corrected_content=None,
        expected_stdout=None,
    ),
}


def public_inference_schema() -> tuple[str, ...]:
    """Expose the exact answer-free task fields."""

    return tuple(item.name for item in fields(PublicVariabilityMultiSourceTask))


def benchmark_artifact_path() -> Path:
    """Return the one immutable destination without creating it."""

    return _ARTIFACT_PATH


def _workspace_record(
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {"content_hex": item.content.hex(), "path": item.path} for item in files
    )


def _cohort_record() -> dict[str, object]:
    training = public_training_experiences()
    tasks = public_fresh_tasks()
    return {
        "private_oracles": tuple(
            {
                "corrected_content_hex": (
                    None
                    if oracle.corrected_content is None
                    else oracle.corrected_content.hex()
                ),
                "expected_stdout_hex": (
                    None
                    if oracle.expected_stdout is None
                    else oracle.expected_stdout.hex()
                ),
                "intended_path": oracle.intended_path,
                "task_id": task_id,
            }
            for task_id, oracle in sorted(_PRIVATE_ORACLES.items())
        ),
        "public_fresh": tuple(
            {
                "files": _workspace_record(task.files),
                "issue_hex": task.issue.hex(),
                "task_id": task.task_id,
            }
            for task in tasks
        ),
        "public_training": tuple(
            {
                "after_files": _workspace_record(row.after_files),
                "before_files": _workspace_record(row.before_files),
                "provenance_id": row.provenance_id,
                "query_hex": row.query.hex(),
            }
            for row in training
        ),
    }


def cohort_sha256() -> str:
    """Commit to public streams and evaluator-private answers without running."""

    return _digest(_COHORT_NAMESPACE, _cohort_record())


def _private_oracle(task_id: str) -> _PrivatePythonOracle:
    try:
        return _PRIVATE_ORACLES[task_id]
    except KeyError as error:
        raise RuntimeError("fresh task lost its sealed evaluator oracle") from error


def _materialize_workspace(
    files: tuple[OpaqueWorkspaceFile, ...],
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in files:
        destination = root.joinpath(*PurePosixPath(item.path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.content)


def _run_python_file(
    files: tuple[OpaqueWorkspaceFile, ...],
    path: str,
    expected_stdout: bytes,
    root: Path,
) -> bool:
    if sum(item.path == path for item in files) != 1:
        return False
    _materialize_workspace(files, root)
    execution = subprocess.run(
        [sys.executable, str(root.joinpath(*PurePosixPath(path).parts))],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return execution.returncode == 0 and execution.stdout == expected_stdout


def _changed_paths(
    before: tuple[OpaqueWorkspaceFile, ...],
    after: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[str, ...]:
    before_by_path = {item.path: item.content for item in before}
    after_by_path = {item.path: item.content for item in after}
    if before_by_path.keys() != after_by_path.keys():
        return tuple(sorted(before_by_path.keys() | after_by_path.keys()))
    return tuple(
        path
        for path in sorted(before_by_path)
        if before_by_path[path] != after_by_path[path]
    )


def _decoys_preserved(
    before: tuple[OpaqueWorkspaceFile, ...],
    after: tuple[OpaqueWorkspaceFile, ...],
    intended_path: str,
) -> bool:
    before_by_path = {item.path: item.content for item in before}
    after_by_path = {item.path: item.content for item in after}
    return before_by_path.keys() == after_by_path.keys() and all(
        after_by_path[path] == content
        for path, content in before_by_path.items()
        if path != intended_path
    )


def _physical_stop_observed(decision: object) -> bool:
    bindings = decision.bindings
    inventory_races = decision.inventory_races
    action = decision.action
    return (
        any(race.stopped for binding in bindings for race in binding.races)
        or any(race.stopped for race in inventory_races)
        or (action is not None and any(race.stopped for race in action.races))
    )


def _run_task(
    controller: VariabilityMultiscaleSourceMotorController,
    task: PublicVariabilityMultiSourceTask,
    *,
    ablation: VariabilityMultiscaleSourceMotorAblation,
    root: Path,
    score_behavior: bool,
) -> dict[str, object]:
    # This is the sole model call.  No evaluator oracle is read before it.
    decision = controller.respond(task.issue, task.files, ablation=ablation)

    observation = MultiscaleOpaqueEventTape.observe(task.issue, task.files)
    before = observation.reconstruct_workspace()
    available = Counter(observation.motor_source_tokens)
    emitted = Counter(decision.emissions)
    emissions_from_current_tape = all(
        count <= available[event] for event, count in emitted.items()
    )
    updated = before
    materialization_valid: bool | None = None
    materialization_error: str | None = None
    if decision.decision == "EMIT":
        if emissions_from_current_tape:
            try:
                updated = observation.materialize_motor_emission(decision.emissions)
                materialization_valid = True
            except (OpaqueEventTapeError, TypeError, ValueError) as error:
                materialization_valid = False
                materialization_error = type(error).__name__
        else:
            materialization_valid = False
            materialization_error = "non_live_emission"
        consequence = controller.resolve(TapeValence.NEUTRAL)
        neutral_checkpoint_retained = (
            consequence.prior_training_sha256 == consequence.training_sha256
            and controller.checkpoint_sha256 == decision.checkpoint_sha256
        )
    else:
        neutral_checkpoint_retained = True

    # Evaluator-only path, correction, and execution expectations begin here.
    oracle = _private_oracle(task.task_id)
    ambiguous = oracle.intended_path is None
    intended_path = oracle.intended_path
    sorted_paths = tuple(item.path for item in before)
    intended_position = (
        None if intended_path is None else sorted_paths.index(intended_path)
    )
    changed_paths = _changed_paths(before, updated)
    exact_correction: bool | None = None
    before_behavior_passed: bool | None = None
    after_behavior_passed: bool | None = None
    decoys_preserved: bool | None = None
    execution_count = 0
    if intended_path is not None and score_behavior:
        if oracle.corrected_content is None or oracle.expected_stdout is None:
            raise RuntimeError("action task lost its private executable oracle")
        after_by_path = {item.path: item.content for item in updated}
        exact_correction = after_by_path.get(intended_path) == oracle.corrected_content
        before_behavior_passed = _run_python_file(
            before,
            intended_path,
            oracle.expected_stdout,
            root / "before",
        )
        after_behavior_passed = _run_python_file(
            updated,
            intended_path,
            oracle.expected_stdout,
            root / "after",
        )
        execution_count = 2
        decoys_preserved = _decoys_preserved(before, updated, intended_path)

    physical_stop = decision.decision == "STOP" and _physical_stop_observed(decision)
    binding_cell_steps = sum(item.physical_cell_steps for item in decision.bindings)
    inventory_cell_steps = sum(
        item.physical_cell_steps for item in decision.inventory_races
    )
    action_cell_steps = (
        0 if decision.action is None else decision.action.physical_cell_steps
    )
    cell_step_accounting_exact = decision.physical_cell_steps == (
        binding_cell_steps + inventory_cell_steps + action_cell_steps
    )

    if ambiguous or not score_behavior:
        capability_success = (
            decision.decision == "STOP" and physical_stop and not changed_paths
        )
    else:
        capability_success = (
            decision.decision == "EMIT"
            and materialization_valid is True
            and exact_correction is True
            and before_behavior_passed is False
            and after_behavior_passed is True
            and changed_paths == (intended_path,)
            and decoys_preserved is True
            and emissions_from_current_tape
            and execution_count == 2
        )

    if capability_success:
        outcome = (
            "correct_ambiguity_stop"
            if ambiguous
            else ("physical_control_stop" if not score_behavior else "correct_edit")
        )
    elif decision.decision == "STOP":
        outcome = "nonphysical_or_unexpected_stop"
    elif materialization_valid is not True:
        outcome = "invalid_materialization"
    elif intended_path not in changed_paths:
        outcome = "wrong_file_or_no_change"
    elif exact_correction is not True:
        outcome = "inexact_correction"
    else:
        outcome = "behavior_failure"

    return {
        "action_cell_steps": action_cell_steps,
        "after_behavior_passed": after_behavior_passed,
        "ambiguous": ambiguous,
        "before_behavior_passed": before_behavior_passed,
        "binding_cell_steps": binding_cell_steps,
        "boundary_match_counts": tuple(
            binding.boundary_match_count for binding in decision.bindings
        ),
        "capability_success": capability_success,
        "cell_step_accounting_exact": cell_step_accounting_exact,
        "changed_paths": changed_paths,
        "decision": decision.decision,
        "decoys_preserved": decoys_preserved,
        "emission_count": len(decision.emissions),
        "emissions_from_current_tape": emissions_from_current_tape,
        "exact_correction": exact_correction,
        "execution_count": execution_count,
        "intended_position": intended_position,
        "inventory_cell_steps": inventory_cell_steps,
        "inventory_race_count": len(decision.inventory_races),
        "materialization_error": materialization_error,
        "materialization_valid": materialization_valid,
        "neutral_checkpoint_retained": neutral_checkpoint_retained,
        "outcome": outcome,
        "physical_cell_steps": decision.physical_cell_steps,
        "physical_stop_observed": physical_stop,
        "query_count": 1,
        "query_sha256": decision.query_sha256,
        "stop_reason": decision.stop_reason,
        "stop_stage": decision.stop_stage,
        "task_id": task.task_id,
        "workspace_lossless": observation.reconstruct_workspace() == before,
    }


def _run_arm(
    *,
    ablation: VariabilityMultiscaleSourceMotorAblation,
    tasks: tuple[PublicVariabilityMultiSourceTask, ...],
    root: Path,
    score_behavior: bool,
) -> dict[str, object]:
    training = public_training_experiences()
    if len(training) != 2:
        raise RuntimeError("V3 training cohort must contain exactly two successes")
    controller = VariabilityMultiscaleSourceMotorController.grow(training)
    initial_checkpoint = controller.checkpoint_sha256
    task_records = tuple(
        _run_task(
            controller,
            task,
            ablation=ablation,
            root=root / task.task_id,
            score_behavior=score_behavior,
        )
        for task in tasks
    )
    failures = Counter(
        str(row["outcome"])
        for row in task_records
        if not bool(row["capability_success"])
    )
    stop_reasons = Counter(
        str(row["stop_reason"])
        for row in task_records
        if row["stop_reason"] is not None
    )
    action_records = tuple(row for row in task_records if not bool(row["ambiguous"]))
    ambiguity_records = tuple(row for row in task_records if bool(row["ambiguous"]))
    return {
        "ablation": ablation.value,
        "action_required_success_count": sum(
            bool(row["capability_success"]) for row in action_records
        ),
        "ambiguity_stop_success_count": sum(
            bool(row["capability_success"]) for row in ambiguity_records
        ),
        "capability_success_count": sum(
            bool(row["capability_success"]) for row in task_records
        ),
        "cell_step_accounting_exact": all(
            bool(row["cell_step_accounting_exact"]) for row in task_records
        ),
        "checkpoint_retained": controller.checkpoint_sha256 == initial_checkpoint,
        "emit_count": sum(row["decision"] == "EMIT" for row in task_records),
        "failure_distribution": dict(sorted(failures.items())),
        "physical_stop_count": sum(
            bool(row["physical_stop_observed"]) for row in task_records
        ),
        "stop_count": sum(row["decision"] == "STOP" for row in task_records),
        "stop_reason_distribution": dict(sorted(stop_reasons.items())),
        "tasks": task_records,
        "total_physical_cell_steps": sum(
            int(row["physical_cell_steps"]) for row in task_records
        ),
        "training_summary": asdict(controller.training_summary),
    }


def _run_experiment_at(root: Path) -> dict[str, object]:
    dependencies = dependency_custody()
    source = _assert_source_custody()
    tasks = public_fresh_tasks()
    action_tasks = tasks[:3]
    treatment = _run_arm(
        ablation=VariabilityMultiscaleSourceMotorAblation.NONE,
        tasks=tasks,
        root=root / "none",
        score_behavior=True,
    )
    controls = {
        ablation.value: _run_arm(
            ablation=ablation,
            tasks=action_tasks,
            root=root / ablation.value,
            score_behavior=False,
        )
        for ablation in (
            VariabilityMultiscaleSourceMotorAblation.EQUALITY_OFF,
            VariabilityMultiscaleSourceMotorAblation.PROPAGATION_OFF,
            VariabilityMultiscaleSourceMotorAblation.BOUNDARY_OFF,
        )
    }
    treatment_actions = treatment["tasks"][:3]
    treatment_ambiguity = treatment["tasks"][3]
    positions = tuple(row["intended_position"] for row in treatment_actions)
    integrity = {
        "all_cell_step_accounting_exact": treatment["cell_step_accounting_exact"]
        and all(control["cell_step_accounting_exact"] for control in controls.values()),
        "all_emissions_live": all(
            bool(row["emissions_from_current_tape"]) for row in treatment["tasks"]
        ),
        "all_neutral_checkpoints_retained": treatment["checkpoint_retained"]
        and all(control["checkpoint_retained"] for control in controls.values())
        and all(bool(row["neutral_checkpoint_retained"]) for row in treatment["tasks"]),
        "all_queries_single_shot": all(
            row["query_count"] == 1
            for arm in (treatment, *controls.values())
            for row in arm["tasks"]
        ),
        "all_workspaces_lossless": all(
            bool(row["workspace_lossless"])
            for arm in (treatment, *controls.values())
            for row in arm["tasks"]
        ),
        "fresh_action_target_positions": positions,
        "fresh_task_fields": public_inference_schema(),
        "public_training_experience_count": len(public_training_experiences()),
    }
    gate_passed = (
        treatment["capability_success_count"] == 4
        and treatment["action_required_success_count"] == 3
        and treatment["ambiguity_stop_success_count"] == 1
        and bool(treatment_ambiguity["capability_success"])
        and treatment["emit_count"] == 3
        and treatment["stop_count"] == 1
        and all(control["emit_count"] == 0 for control in controls.values())
        and all(control["stop_count"] == 3 for control in controls.values())
        and all(control["physical_stop_count"] == 3 for control in controls.values())
        and integrity["fresh_task_fields"] == ("task_id", "issue", "files")
        and integrity["public_training_experience_count"] == 2
        and integrity["fresh_action_target_positions"] == (0, 2, 3)
        and integrity["all_queries_single_shot"] is True
        and integrity["all_workspaces_lossless"] is True
        and integrity["all_emissions_live"] is True
        and integrity["all_neutral_checkpoints_retained"] is True
        and integrity["all_cell_step_accounting_exact"] is True
    )
    failure_distribution = {
        "none": treatment["failure_distribution"],
        **{name: control["failure_distribution"] for name, control in controls.items()},
    }
    result: dict[str, object] = {
        "architecture_commit": ARCHITECTURE_COMMIT,
        "arms": {"none": treatment, **controls},
        "claim": (
            "Fresh variable-intermediate, boundary-invariant issue-to-path grounding "
            "composed with one learned live-token prefix-truncation motor."
        ),
        "cohort_sha256": cohort_sha256(),
        "failure_distribution": failure_distribution,
        "fresh_action_task_count": 3,
        "fresh_ambiguity_task_count": 1,
        "gate_passed": gate_passed,
        "integrity": integrity,
        "limitations": (
            "Exact opaque identifier equality supplies the query-to-workspace seed.",
            "The learned motor reuses one observed physical-line action shape.",
            "Fresh tasks do not update the learner; this is not continual learning.",
            "This is not arbitrary language understanding, debugging, or synthesis.",
            "The cohort is public TRAIN-only and does not open protected R5/R6 evidence.",
        ),
        "public_api_schema": {
            "controller_respond": ("self", "query", "files", "ablation"),
            "task_fields": public_inference_schema(),
        },
        "source_custody": {
            "dependencies": dependencies,
            "experiment": source,
        },
    }
    result["result_sha256"] = _digest(_RESULT_NAMESPACE, result)
    return result


def run_and_write_artifact() -> dict[str, object]:
    """Spend the cohort once and exclusively create its fixed result artifact."""

    if _ARTIFACT_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite spent V3 artifact: {_ARTIFACT_PATH}"
        )
    with tempfile.TemporaryDirectory(
        prefix="variability-multiscale-python-edit-v3-"
    ) as directory:
        result = _run_experiment_at(Path(directory))
    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _ARTIFACT_PATH.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main(argv: tuple[str, ...] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="spend the frozen cohort once and exclusively create its artifact",
    )
    arguments = parser.parse_args(argv)
    if not arguments.write_artifact:
        parser.error("explicit --write-artifact is required")
    result = run_and_write_artifact()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE_COMMIT",
    "EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256",
    "PublicVariabilityMultiSourceTask",
    "benchmark_artifact_path",
    "cohort_sha256",
    "dependency_custody",
    "experiment_source_custody_sha256",
    "experiment_source_sha256",
    "public_fresh_tasks",
    "public_inference_schema",
    "public_training_experiences",
    "run_and_write_artifact",
]
