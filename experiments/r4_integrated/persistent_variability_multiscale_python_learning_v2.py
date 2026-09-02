"""Unspent persistent variability-aware Python learning cohort.

Two independently seeded neuronal stacks traverse the same public TRAIN stream.
The treatment stack may append one shard from exactly two independently
evaluator-authenticated current successful transitions; the frozen control is
offered the same transitions under ``NO_UPDATE``.  Fresh transfer, delayed
retention, uncertainty, and causal ablations are measured only after that
intervention.

Every frozen response receives only issue bytes and a complete lossless
workspace.  The evaluator never supplies a path, span, patch, candidate set, or
repaired-source lookup to inference.  Emitted actions must be occurrences from
the live sensory tape, and Python behavior is evaluated only after the one-shot
response.  This is a narrow public-TRAIN test of replay-free structural growth,
not evidence of arbitrary Python synthesis or general language understanding.

Importing this module is inert.  The only spending entry point requires an
explicit ``--write-artifact`` flag and exclusively creates one fixed artifact.
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
from humanbrain.variability_multiscale_source_motor_controller import (
    SuccessfulVariabilityMultiscaleSourceMotorExperience,
    VariabilityMultiscaleSourceMotorAblation,
    VariabilityMultiscaleSourceMotorController,
)
from humanbrain.variability_multiscale_source_motor_online_growth import (
    AppendOnlyVariabilityMultiscaleSourceMotorStack,
    MultiscaleSourceMotorExecutionAuthority,
    MultiscaleSourceMotorSuccessEvidence,
    OnlineMultiscaleGrowthAblation,
)

ARCHITECTURE_COMMIT = "fd38ef6981f3b0c00d5c76557527b3aed69b9d1b"
EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256 = (
    "d40296a971fedffcea6d2310403f826dc1eaebed751b7e173ade2e3d1dec06f2"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = Path(__file__).resolve()
_ARTIFACT_PATH = (
    _REPO_ROOT
    / "benchmarks"
    / "r4_integrated_persistent_variability_multiscale_python_learning_v2.json"
)
_RESULT_NAMESPACE = b"humanbrain.persistent-variability-python-v2.result\x00"
_COHORT_NAMESPACE = b"humanbrain.persistent-variability-python-v2.cohort\x00"
_CUSTODY_NAMESPACE = b"humanbrain.persistent-variability-python-v2.custody\x00"
_EVALUATOR_SHA256 = hashlib.sha256(
    b"persistent-variability-python-v2.behavioral-transition-evaluator"
).hexdigest()
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
    "src/humanbrain/multiscale_source_motor_online_growth.py": (
        "fa0561f24006df0750ccd5ba640bdb53bba1f871fa7d3082e5f6453f658a57d9"
    ),
    "src/humanbrain/variability_multiscale_source_motor_online_growth.py": (
        "47a13e0eb493005af2aed6ec6422283a51b077a68eff084fa51377effe177f26"
    ),
}


@dataclass(frozen=True, slots=True)
class PublicPersistentVariabilityTask:
    """Complete answer-free model surface for one stream event."""

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
class _PreparedTask:
    public: PublicPersistentVariabilityTask
    successful_after: tuple[OpaqueWorkspaceFile, ...] | None


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
    """Return the exact current runner hash without spending the cohort."""

    return _file_sha256(_RUNNER_PATH)


def experiment_source_custody_sha256() -> str:
    """Hash runner bytes after masking only the embedded custody hash."""

    source = _RUNNER_PATH.read_bytes()
    normalized, replacement_count = _SOURCE_CUSTODY_PATTERN.subn(
        lambda match: match.group(1) + b"0" * 64 + match.group(2), source
    )
    if replacement_count != 1:
        raise RuntimeError("experiment source custody marker changed")
    return hashlib.sha256(normalized).hexdigest()


def dependency_custody() -> dict[str, object]:
    """Authenticate frozen neuronal dependencies without running a model."""

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
            "persistent variability V2 dependency custody changed: "
            + _json_bytes(mismatches).decode("ascii")
        )
    return {
        "architecture_commit": ARCHITECTURE_COMMIT,
        "files": actual,
        "sha256": _digest(_CUSTODY_NAMESPACE, actual),
    }


def _assert_source_custody() -> dict[str, str]:
    custody_sha256 = experiment_source_custody_sha256()
    if custody_sha256 != EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256:
        raise RuntimeError(
            "persistent variability V2 experiment source custody changed: "
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


def _opaque_decoys(paths: tuple[str, ...]) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (
            path,
            b"retained opaque bytes: "
            + path.replace("/", "_").replace(".", "_").encode("ascii")
            + b"\n",
        )
        for path in paths
    )


def _sorted_workspace(
    rows: tuple[tuple[str, bytes], ...],
) -> tuple[OpaqueWorkspaceFile, ...]:
    return tuple(
        OpaqueWorkspaceFile(path, content)
        for path, content in sorted(rows, key=lambda item: item[0])
    )


def _prepare_action_task(
    *,
    task_id: str,
    changed_path: str,
    function_name: str,
    value: str,
    decoy_paths: tuple[str, ...],
) -> _PreparedTask:
    stem = PurePosixPath(changed_path).stem
    before = _sorted_workspace(
        (
            *_opaque_decoys(decoy_paths),
            (changed_path, _buggy_source(function_name, value)),
        )
    )
    after = tuple(
        OpaqueWorkspaceFile(
            item.path,
            _correct_source(function_name, value)
            if item.path == changed_path
            else item.content,
        )
        for item in before
    )
    return _PreparedTask(
        public=PublicPersistentVariabilityTask(
            task_id=task_id,
            issue=(
                b"remove the obsolete crash in "
                + stem.encode("ascii")
                + b" using current workspace bytes"
            ),
            files=before,
        ),
        successful_after=after,
    )


def _prepare_ambiguity_task() -> _PreparedTask:
    before = _sorted_workspace(
        (
            *_opaque_decoys(
                (
                    "coral/ledger_note.txt",
                    "maple/fossil_data.json",
                    "vapor/harbor_table.csv",
                )
            ),
            (
                "dahlia/inner/marble_worker.py",
                _buggy_source("marble_cycle", "marble"),
            ),
            (
                "xenia/outer/river_worker.py",
                _buggy_source("river_cycle", "river"),
            ),
        )
    )
    return _PreparedTask(
        public=PublicPersistentVariabilityTask(
            task_id="persistent-v2-ambiguity-dual",
            issue=(
                b"remove the obsolete crash in marble_worker or river_worker "
                b"using current workspace bytes"
            ),
            files=before,
        ),
        successful_after=None,
    )


def public_seed_training_experiences() -> tuple[
    SuccessfulVariabilityMultiscaleSourceMotorExperience, ...
]:
    """Two independent public shallow successes used to seed each stack."""

    prepared = (
        _prepare_action_task(
            task_id="persistent-v2-seed-amber",
            changed_path="boreal/amber_seed.py",
            function_name="amber_seed_cycle",
            value="amber",
            decoy_paths=(
                "cobalt/brass_note.md",
                "linden/cedar_data.json",
                "quartz/delta_helper.py",
                "zephyr/ember_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-seed-violet",
            changed_path="morrow/violet_seed.py",
            function_name="violet_seed_cycle",
            value="violet",
            decoy_paths=(
                "acacia/fjord_note.md",
                "cedar/grove_data.json",
                "sable/willow_helper.py",
                "xenon/yarrow_table.csv",
            ),
        ),
    )
    experiences = []
    for row in prepared:
        if row.successful_after is None:
            raise RuntimeError("seed row lost its successful transition")
        experiences.append(
            SuccessfulVariabilityMultiscaleSourceMotorExperience(
                query=row.public.issue,
                before_files=row.public.files,
                after_files=row.successful_after,
                provenance_id=row.public.task_id,
            )
        )
    return tuple(experiences)


def _prepared_stream() -> tuple[_PreparedTask, ...]:
    return (
        _prepare_action_task(
            task_id="persistent-v2-shallow-garnet",
            changed_path="garnet/sable_probe.py",
            function_name="sable_probe_cycle",
            value="garnet",
            decoy_paths=(
                "alder/canvas_note.md",
                "dune/ivory_data.json",
                "juniper/linen_helper.py",
                "willow/pearl_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-shallow-topaz",
            changed_path="umber/topaz_probe.py",
            function_name="topaz_probe_cycle",
            value="umber",
            decoy_paths=(
                "bramble/ruby_note.md",
                "elm/silver_data.json",
                "river/tawny_helper.py",
                "zinnia/velvet_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-current-cinder",
            changed_path="adobe/birch/cinder_growth.py",
            function_name="cinder_growth_cycle",
            value="cinder",
            decoy_paths=(
                "bamboo/copper_note.md",
                "frost/indigo_data.json",
                "stone/jasper_helper.py",
                "yarrow/kestrel_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-current-umber",
            changed_path="zodiac/yarrow/umber_growth.py",
            function_name="umber_growth_cycle",
            value="zodiac",
            decoy_paths=(
                "canyon/lilac_note.md",
                "hemlock/nacre_data.json",
                "prairie/opal_helper.py",
                "tulip/quill_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-transfer-lattice",
            changed_path="nimbus/opal/lattice_transfer.py",
            function_name="lattice_transfer_cycle",
            value="nimbus",
            decoy_paths=(
                "charcoal/mica_note.md",
                "lagoon/ochre_data.json",
                "raven/sienna_helper.py",
                "velvet/topaz_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-retention-citrine",
            changed_path="thistle/citrine_retention.py",
            function_name="citrine_retention_cycle",
            value="thistle",
            decoy_paths=(
                "flint/umber_note.md",
                "kelp/violet_data.json",
                "orchid/white_helper.py",
                "wren/yellow_table.csv",
            ),
        ),
        _prepare_ambiguity_task(),
        _prepare_action_task(
            task_id="persistent-v2-unsupported-prism",
            changed_path="sierra/ember/core/prism_unsupported.py",
            function_name="prism_unsupported_cycle",
            value="sierra",
            decoy_paths=(
                "cloud/auburn_note.md",
                "hazel/bisque_data.json",
                "tundra/cream_helper.py",
                "yucca/denim_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-ablation-equinox",
            changed_path="iris/jade/equinox_probe.py",
            function_name="equinox_probe_cycle",
            value="iris",
            decoy_paths=(
                "copper/egret_note.md",
                "lilac/fox_data.json",
                "reed/gull_helper.py",
                "winter/heron_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-ablation-pulse",
            changed_path="monsoon/ochre/pulse_probe.py",
            function_name="pulse_probe_cycle",
            value="monsoon",
            decoy_paths=(
                "ginkgo/ibis_note.md",
                "nettle/jay_data.json",
                "spruce/koala_helper.py",
                "walnut/lemur_table.csv",
            ),
        ),
        _prepare_action_task(
            task_id="persistent-v2-ablation-keystone",
            changed_path="violet/umber/keystone_probe.py",
            function_name="keystone_probe_cycle",
            value="violet",
            decoy_paths=(
                "fern/marten_note.md",
                "oak/newt_data.json",
                "truffle/otter_helper.py",
                "zircon/panda_table.csv",
            ),
        ),
    )


def public_stream_tasks() -> tuple[PublicPersistentVariabilityTask, ...]:
    """Return the immutable answer-free public TRAIN stream."""

    return tuple(row.public for row in _prepared_stream())


def public_inference_schema() -> tuple[str, ...]:
    return tuple(field.name for field in fields(PublicPersistentVariabilityTask))


def public_cohort_manifest() -> dict[str, object]:
    """Describe the one-shot protocol without revealing successful actions."""

    tasks = public_stream_tasks()
    return {
        "architecture_commit": ARCHITECTURE_COMMIT,
        "causal_ablation_task_ids": tuple(task.task_id for task in tasks[8:]),
        "current_success_task_ids": tuple(task.task_id for task in tasks[2:4]),
        "mode": "public_train_one_shot",
        "protected_splits_opened": (),
        "seed_success_count": len(public_seed_training_experiences()),
        "stream_task_ids": tuple(task.task_id for task in tasks),
        "treatment_stack_count": 1,
        "frozen_control_stack_count": 1,
    }


def _workspace_record(
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {"content_hex": item.content.hex(), "path": item.path} for item in files
    )


def _cohort_record() -> dict[str, object]:
    return {
        "evaluator_sha256": _EVALUATOR_SHA256,
        "manifest": public_cohort_manifest(),
        "seed_training": tuple(
            {
                "after_files": _workspace_record(row.after_files),
                "before_files": _workspace_record(row.before_files),
                "provenance_id": row.provenance_id,
                "query_hex": row.query.hex(),
            }
            for row in public_seed_training_experiences()
        ),
        "stream": tuple(
            {
                "evaluator_successful_after": (
                    None
                    if row.successful_after is None
                    else _workspace_record(row.successful_after)
                ),
                "files": _workspace_record(row.public.files),
                "issue_hex": row.public.issue.hex(),
                "task_id": row.public.task_id,
            }
            for row in _prepared_stream()
        ),
    }


def cohort_sha256() -> str:
    """Commit to the public stream and evaluator transitions without running."""

    return _digest(_COHORT_NAMESPACE, _cohort_record())


def benchmark_artifact_path() -> Path:
    return _ARTIFACT_PATH


def _materialize_workspace(
    files: tuple[OpaqueWorkspaceFile, ...],
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in files:
        destination = root.joinpath(*PurePosixPath(item.path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.content)


def _execute_python_file(
    files: tuple[OpaqueWorkspaceFile, ...],
    path: str,
    root: Path,
) -> dict[str, object]:
    if sum(item.path == path for item in files) != 1:
        return {
            "executed": False,
            "returncode": None,
            "stderr_sha256": None,
            "stdout_hex": None,
        }
    _materialize_workspace(files, root)
    execution = subprocess.run(
        [sys.executable, str(root.joinpath(*PurePosixPath(path).parts))],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return {
        "executed": True,
        "returncode": execution.returncode,
        "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
        "stdout_hex": execution.stdout.hex(),
    }


def _issue_related_paths(
    issue: bytes,
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[str, ...]:
    identities = frozenset(_IDENTIFIER.findall(issue))
    return tuple(
        item.path
        for item in files
        if PurePosixPath(item.path).stem.encode("utf-8") in identities
    )


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
    changed_path: str,
) -> bool:
    before_by_path = {item.path: item.content for item in before}
    after_by_path = {item.path: item.content for item in after}
    return before_by_path.keys() == after_by_path.keys() and all(
        after_by_path[path] == content
        for path, content in before_by_path.items()
        if path != changed_path
    )


def _evaluate_successful_transition(
    *,
    query: bytes,
    before_files: tuple[OpaqueWorkspaceFile, ...],
    after_files: tuple[OpaqueWorkspaceFile, ...],
    root: Path,
) -> dict[str, object]:
    """Behaviorally evaluate one transition without a repaired-source lookup."""

    related = _issue_related_paths(query, before_files)
    changed = _changed_paths(before_files, after_files)
    before_execution: dict[str, object] | None = None
    after_execution: dict[str, object] | None = None
    decoys_preserved = False
    if len(related) == 1:
        path = related[0]
        before_execution = _execute_python_file(before_files, path, root / "before")
        after_execution = _execute_python_file(after_files, path, root / "after")
        decoys_preserved = _decoys_preserved(before_files, after_files, path)
    passed = (
        len(related) == 1
        and changed == related
        and decoys_preserved
        and before_execution is not None
        and before_execution["executed"] is True
        and before_execution["returncode"] != 0
        and after_execution is not None
        and after_execution["executed"] is True
        and after_execution["returncode"] == 0
        and before_execution["stdout_hex"] == after_execution["stdout_hex"]
        and before_execution["stdout_hex"] not in (None, "")
    )
    return {
        "after_execution": after_execution,
        "before_execution": before_execution,
        "changed_paths": changed,
        "decoys_preserved": decoys_preserved,
        "issue_related_paths": related,
        "passed": passed,
    }


def _model_attempt(
    *,
    learner: AppendOnlyVariabilityMultiscaleSourceMotorStack,
    prepared: _PreparedTask,
    arm: str,
    phase: str,
    expected_decision: str,
    ablation: VariabilityMultiscaleSourceMotorAblation,
    root: Path,
    sequence_index: int,
) -> dict[str, object]:
    public = prepared.public
    checkpoint_before = learner.checkpoint_sha256

    # This is the sole model call.  No successful transition is read before it.
    decision = learner.respond(public.issue, public.files, ablation=ablation)

    observation = MultiscaleOpaqueEventTape.observe(public.issue, public.files)
    before = observation.reconstruct_workspace()
    available = Counter(observation.motor_source_tokens)
    emitted = Counter(decision.emissions)
    emissions_live = all(
        count <= available[event] for event, count in emitted.items()
    )
    updated = before
    materialization_valid: bool | None = None
    materialization_error: str | None = None
    if decision.decision == "EMIT":
        if emissions_live:
            try:
                updated = observation.materialize_motor_emission(decision.emissions)
                materialization_valid = True
            except (OpaqueEventTapeError, TypeError, ValueError) as error:
                materialization_valid = False
                materialization_error = type(error).__name__
        else:
            materialization_valid = False
            materialization_error = "non_live_emission"

    # Evaluator-side relation and Python execution begin only after inference.
    related = _issue_related_paths(public.issue, before)
    changed = _changed_paths(before, updated)
    before_execution: dict[str, object] | None = None
    after_execution: dict[str, object] | None = None
    correct_file = False
    decoys_preserved = False
    behavior_passed = False
    if len(related) == 1:
        path = related[0]
        before_execution = _execute_python_file(before, path, root / "before")
        correct_file = changed == related
        decoys_preserved = _decoys_preserved(before, updated, path)
        if decision.decision == "EMIT" and materialization_valid is True:
            after_execution = _execute_python_file(updated, path, root / "after")
            behavior_passed = (
                before_execution["executed"] is True
                and before_execution["returncode"] != 0
                and after_execution["executed"] is True
                and after_execution["returncode"] == 0
                and before_execution["stdout_hex"] == after_execution["stdout_hex"]
                and before_execution["stdout_hex"] not in (None, "")
            )

    edit_success = (
        decision.decision == "EMIT"
        and materialization_valid is True
        and emissions_live
        and correct_file
        and decoys_preserved
        and behavior_passed
    )
    stop_success = (
        decision.decision == "STOP"
        and decision.emissions == ()
        and decision.race.stopped
        and changed == ()
    )
    if expected_decision == "EDIT":
        expectation_met = edit_success
    elif expected_decision == "STOP":
        expectation_met = stop_success
    else:
        raise ValueError("expected_decision must be EDIT or STOP")

    if edit_success:
        outcome = "correct_edit"
    elif stop_success:
        outcome = "correct_physical_stop"
    elif decision.decision == "STOP":
        outcome = "nonphysical_or_mutating_stop"
    elif materialization_valid is not True:
        outcome = "invalid_materialization"
    elif not correct_file:
        outcome = "wrong_file"
    elif not decoys_preserved:
        outcome = "decoy_changed"
    else:
        outcome = "python_behavior_failure"

    checkpoint_after = learner.checkpoint_sha256
    return {
        "ablation": ablation.value,
        "after_execution": after_execution,
        "arm": arm,
        "before_execution": before_execution,
        "behavior_passed": behavior_passed,
        "changed_paths": changed,
        "checkpoint_after": checkpoint_after,
        "checkpoint_before": checkpoint_before,
        "checkpoint_retained": checkpoint_after == checkpoint_before,
        "component_decision_sha256s": decision.component_decision_sha256s,
        "component_physical_cell_steps": (
            decision.physical_cell_steps - decision.race.physical_cell_steps
        ),
        "component_stop_reasons": decision.component_stop_reasons,
        "correct_file": correct_file,
        "decision": decision.decision,
        "decision_sha256": decision.sha256,
        "decoys_preserved": decoys_preserved,
        "emission_count": len(decision.emissions),
        "emissions_hex": tuple(event.hex() for event in decision.emissions),
        "emissions_live": emissions_live,
        "expected_decision": expected_decision,
        "expectation_met": expectation_met,
        "issue_related_paths": related,
        "materialization_error": materialization_error,
        "materialization_valid": materialization_valid,
        "model_query_count": 1,
        "outcome": outcome,
        "outer_race": asdict(decision.race),
        "phase": phase,
        "physical_cell_steps": decision.physical_cell_steps,
        "physical_stop": decision.decision == "STOP" and decision.race.stopped,
        "query_sha256": decision.query_sha256,
        "sequence_index": sequence_index,
        "stop_reason": decision.stop_reason,
        "task_id": public.task_id,
        "winning_shard_ordinal": decision.winning_shard_ordinal,
        "workspace_lossless": (
            observation.reconstruct_query() == public.issue
            and observation.reconstruct_workspace() == before
        ),
    }


def _authenticate_current_success(
    *,
    learner: AppendOnlyVariabilityMultiscaleSourceMotorStack,
    prepared: _PreparedTask,
    arm: str,
    root: Path,
) -> tuple[MultiscaleSourceMotorSuccessEvidence, dict[str, object]]:
    after = prepared.successful_after
    if after is None:
        raise RuntimeError("current success row lost its complete transition")
    evaluator_audits: list[dict[str, object]] = []

    def evaluator(**observation: object) -> bool:
        audit = _evaluate_successful_transition(
            query=observation["query"],  # type: ignore[arg-type]
            before_files=observation["before_files"],  # type: ignore[arg-type]
            after_files=observation["after_files"],  # type: ignore[arg-type]
            root=root,
        )
        evaluator_audits.append(audit)
        return bool(audit["passed"])

    authority = MultiscaleSourceMotorExecutionAuthority(
        evaluator=evaluator,
        evaluator_sha256=_EVALUATOR_SHA256,
    )
    receipt = authority.execute_and_issue(
        learner_checkpoint_sha256=learner.checkpoint_sha256,
        query=prepared.public.issue,
        before_files=prepared.public.files,
        after_files=after,
    )
    evidence = MultiscaleSourceMotorSuccessEvidence.seal(
        query=prepared.public.issue,
        before_files=prepared.public.files,
        after_files=after,
        execution_receipt=receipt,
        provenance_id=f"{arm}:{prepared.public.task_id}",
    )
    if len(evaluator_audits) != 1:
        raise RuntimeError("current success evaluator call count changed")
    return evidence, {
        "arm": arm,
        "audit": evaluator_audits[0],
        "evidence_sha256": evidence.sha256,
        "evaluator_call_count": 1,
        "evaluator_sha256": _EVALUATOR_SHA256,
        "execution_receipt": asdict(receipt),
        "learner_checkpoint_sha256": learner.checkpoint_sha256,
        "task_id": prepared.public.task_id,
    }


def _new_seed_stack() -> AppendOnlyVariabilityMultiscaleSourceMotorStack:
    controller = VariabilityMultiscaleSourceMotorController.grow(
        public_seed_training_experiences()
    )
    return AppendOnlyVariabilityMultiscaleSourceMotorStack.seed(controller)


def _distribution(
    episodes: list[dict[str, object]],
    field: str,
    *,
    failures_only: bool = False,
) -> dict[str, int]:
    rows = (
        row
        for row in episodes
        if not failures_only or not bool(row["expectation_met"])
    )
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _run_experiment_at(root: Path) -> dict[str, object]:
    dependencies = dependency_custody()
    source = _assert_source_custody()
    prepared = {row.public.task_id: row for row in _prepared_stream()}
    task_ids = tuple(prepared)
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("stream task identifiers must be unique")

    treatment = _new_seed_stack()
    frozen_control = _new_seed_stack()
    stacks_independently_seeded = treatment is not frozen_control
    treatment_initial_checkpoint = treatment.checkpoint_sha256
    control_initial_checkpoint = frozen_control.checkpoint_sha256
    episodes: list[dict[str, object]] = []

    def append_attempt(
        *,
        learner: AppendOnlyVariabilityMultiscaleSourceMotorStack,
        task_id: str,
        arm: str,
        phase: str,
        expected_decision: str,
        ablation: VariabilityMultiscaleSourceMotorAblation = (
            VariabilityMultiscaleSourceMotorAblation.NONE
        ),
    ) -> None:
        sequence_index = len(episodes)
        episodes.append(
            _model_attempt(
                learner=learner,
                prepared=prepared[task_id],
                arm=arm,
                phase=phase,
                expected_decision=expected_decision,
                ablation=ablation,
                root=root / f"{sequence_index:02d}-{arm}-{phase}",
                sequence_index=sequence_index,
            )
        )

    for task_id, suffix in (
        ("persistent-v2-shallow-garnet", "garnet"),
        ("persistent-v2-shallow-topaz", "topaz"),
    ):
        append_attempt(
            learner=treatment,
            task_id=task_id,
            arm="treatment",
            phase=f"prior_shallow_{suffix}",
            expected_decision="EDIT",
        )
        append_attempt(
            learner=frozen_control,
            task_id=task_id,
            arm="frozen_control",
            phase=f"prior_shallow_{suffix}",
            expected_decision="EDIT",
        )

    treatment_evidence: list[MultiscaleSourceMotorSuccessEvidence] = []
    control_evidence: list[MultiscaleSourceMotorSuccessEvidence] = []
    current_success_audits: list[dict[str, object]] = []
    for task_id, suffix in (
        ("persistent-v2-current-cinder", "cinder"),
        ("persistent-v2-current-umber", "umber"),
    ):
        append_attempt(
            learner=treatment,
            task_id=task_id,
            arm="treatment",
            phase=f"current_before_learning_{suffix}",
            expected_decision="STOP",
        )
        append_attempt(
            learner=frozen_control,
            task_id=task_id,
            arm="frozen_control",
            phase=f"current_before_learning_{suffix}",
            expected_decision="STOP",
        )
        treatment_row, treatment_audit = _authenticate_current_success(
            learner=treatment,
            prepared=prepared[task_id],
            arm="treatment",
            root=root / "current-success" / "treatment" / suffix,
        )
        control_row, control_audit = _authenticate_current_success(
            learner=frozen_control,
            prepared=prepared[task_id],
            arm="frozen_control",
            root=root / "current-success" / "frozen-control" / suffix,
        )
        treatment_evidence.append(treatment_row)
        control_evidence.append(control_row)
        current_success_audits.extend((treatment_audit, control_audit))

    treatment_parent = treatment
    control_parent = frozen_control
    treatment_growth = treatment_parent.extend_successes(tuple(treatment_evidence))
    control_growth = control_parent.extend_successes(
        tuple(control_evidence),
        ablation=OnlineMultiscaleGrowthAblation.NO_UPDATE,
    )
    treatment = treatment_growth.successor
    frozen_control = control_growth.successor

    append_attempt(
        learner=treatment,
        task_id="persistent-v2-transfer-lattice",
        arm="treatment",
        phase="fresh_deeper_transfer",
        expected_decision="EDIT",
    )
    append_attempt(
        learner=frozen_control,
        task_id="persistent-v2-transfer-lattice",
        arm="frozen_control",
        phase="fresh_deeper_transfer",
        expected_decision="STOP",
    )
    append_attempt(
        learner=treatment,
        task_id="persistent-v2-retention-citrine",
        arm="treatment",
        phase="delayed_shallow_retention",
        expected_decision="EDIT",
    )
    append_attempt(
        learner=frozen_control,
        task_id="persistent-v2-retention-citrine",
        arm="frozen_control",
        phase="delayed_shallow_retention",
        expected_decision="EDIT",
    )
    for task_id, phase in (
        ("persistent-v2-ambiguity-dual", "ambiguity"),
        ("persistent-v2-unsupported-prism", "unsupported_deeper_structure"),
    ):
        append_attempt(
            learner=treatment,
            task_id=task_id,
            arm="treatment",
            phase=phase,
            expected_decision="STOP",
        )
        append_attempt(
            learner=frozen_control,
            task_id=task_id,
            arm="frozen_control",
            phase=phase,
            expected_decision="STOP",
        )

    causal_tasks = (
        (
            "persistent-v2-ablation-equinox",
            VariabilityMultiscaleSourceMotorAblation.EQUALITY_OFF,
        ),
        (
            "persistent-v2-ablation-pulse",
            VariabilityMultiscaleSourceMotorAblation.PROPAGATION_OFF,
        ),
        (
            "persistent-v2-ablation-keystone",
            VariabilityMultiscaleSourceMotorAblation.BOUNDARY_OFF,
        ),
    )
    for task_id, ablation in causal_tasks:
        append_attempt(
            learner=treatment,
            task_id=task_id,
            arm="treatment",
            phase=f"causal_reference_{ablation.value}",
            expected_decision="EDIT",
        )
        append_attempt(
            learner=treatment,
            task_id=task_id,
            arm=f"ablation_{ablation.value}",
            phase=f"causal_intervention_{ablation.value}",
            expected_decision="STOP",
            ablation=ablation,
        )

    by_arm_phase = {
        (str(row["arm"]), str(row["phase"])): row for row in episodes
    }
    transfer_treatment = by_arm_phase[("treatment", "fresh_deeper_transfer")]
    transfer_control = by_arm_phase[("frozen_control", "fresh_deeper_transfer")]
    delayed_treatment = by_arm_phase[("treatment", "delayed_shallow_retention")]
    delayed_control = by_arm_phase[("frozen_control", "delayed_shallow_retention")]
    causal_pairs = tuple(
        {
            "ablation": ablation.value,
            "intervention": by_arm_phase[
                (f"ablation_{ablation.value}", f"causal_intervention_{ablation.value}")
            ],
            "reference": by_arm_phase[
                ("treatment", f"causal_reference_{ablation.value}")
            ],
            "task_id": task_id,
        }
        for task_id, ablation in causal_tasks
    )

    attempt_keys = tuple(
        (str(row["arm"]), str(row["task_id"])) for row in episodes
    )
    growth_integrity = {
        "control_no_update": control_growth.receipt.status == "NO_UPDATE",
        "control_object_unchanged": frozen_control is control_parent,
        "control_checkpoint_unchanged": (
            frozen_control.checkpoint_sha256 == control_initial_checkpoint
            == control_growth.receipt.parent_checkpoint_sha256
            == control_growth.receipt.successor_checkpoint_sha256
        ),
        "control_raw_rows_not_retained": (
            control_growth.receipt.raw_training_sequences_retained == 0
        ),
        "current_success_count_is_two": (
            treatment_growth.receipt.current_success_count == 2
        ),
        "parent_experiences_not_replayed": (
            treatment_growth.receipt.parent_experience_replay_count == 0
        ),
        "raw_training_sequences_not_retained": (
            treatment_growth.receipt.raw_training_sequences_retained == 0
        ),
        "treatment_appended": treatment_growth.receipt.status == "APPENDED",
        "treatment_parent_immutable": (
            treatment_parent.checkpoint_sha256 == treatment_initial_checkpoint
        ),
        "treatment_successor_changed": (
            treatment.checkpoint_sha256 != treatment_initial_checkpoint
        ),
    }
    integrity = {
        "all_attempt_checkpoints_retained": all(
            bool(row["checkpoint_retained"]) for row in episodes
        ),
        "all_current_successes_authenticated_once": (
            len(current_success_audits) == 4
            and all(row["evaluator_call_count"] == 1 for row in current_success_audits)
            and all(bool(row["audit"]["passed"]) for row in current_success_audits)
        ),
        "all_emissions_live": all(bool(row["emissions_live"]) for row in episodes),
        "all_model_calls_one_shot": (
            all(row["model_query_count"] == 1 for row in episodes)
            and len(attempt_keys) == len(set(attempt_keys))
        ),
        "all_workspaces_lossless": all(
            bool(row["workspace_lossless"]) for row in episodes
        ),
        "causal_ablations": all(
            pair["reference"]["outcome"] == "correct_edit"
            and pair["intervention"]["outcome"] == "correct_physical_stop"
            for pair in causal_pairs
        ),
        "correct_file_and_decoys_preserved": all(
            bool(row["correct_file"]) and bool(row["decoys_preserved"])
            for row in episodes
            if row["expected_decision"] == "EDIT"
        ),
        "independently_seeded_stack_objects": stacks_independently_seeded,
        "public_inference_is_answer_free": (
            public_inference_schema() == ("task_id", "issue", "files")
        ),
        "protected_splits_remain_sealed": (
            public_cohort_manifest()["protected_splits_opened"] == ()
        ),
        "real_before_after_execution_for_edits": all(
            row["before_execution"] is not None
            and row["after_execution"] is not None
            and bool(row["behavior_passed"])
            for row in episodes
            if row["expected_decision"] == "EDIT"
        ),
    }
    gate_passed = (
        all(bool(row["expectation_met"]) for row in episodes)
        and all(bool(value) for value in growth_integrity.values())
        and all(bool(value) for value in integrity.values())
        and transfer_treatment["outcome"] == "correct_edit"
        and transfer_treatment["winning_shard_ordinal"] == 1
        and transfer_control["outcome"] == "correct_physical_stop"
        and delayed_treatment["winning_shard_ordinal"] == 0
        and delayed_control["winning_shard_ordinal"] == 0
    )

    result: dict[str, object] = {
        "architecture_commit": ARCHITECTURE_COMMIT,
        "causal_ablation_results": causal_pairs,
        "checkpoint_evolution": {
            "control": (
                control_initial_checkpoint,
                control_growth.receipt.successor_checkpoint_sha256,
                frozen_control.checkpoint_sha256,
            ),
            "treatment": (
                treatment_initial_checkpoint,
                treatment_growth.receipt.successor_checkpoint_sha256,
                treatment.checkpoint_sha256,
            ),
        },
        "claim": (
            "One replay-free neuronal shard learned a fresh opaque deeper path "
            "structure from two evaluator-authenticated current transitions, "
            "transferred it once, and retained the prior shallow structure."
        ),
        "cohort_sha256": cohort_sha256(),
        "current_success_audits": tuple(current_success_audits),
        "episodes": tuple(episodes),
        "failure_distribution": _distribution(
            episodes, "outcome", failures_only=True
        ),
        "gate_passed": gate_passed,
        "growth": {
            "frozen_control": asdict(control_growth.receipt),
            "treatment": asdict(treatment_growth.receipt),
        },
        "growth_integrity": growth_integrity,
        "integrity": integrity,
        "limitations": (
            "Exact opaque identifier equality seeds issue-to-workspace grounding.",
            "The neuronal motor reuses one learned live-prefix truncation shape.",
            "Growth appends one depth-specific shard rather than consolidating all structure.",
            "The cohort does not test arbitrary language, debugging, or synthesis.",
            "This public TRAIN cohort does not open protected DEV, R5, or R6 evidence.",
        ),
        "outcome_distribution": _distribution(episodes, "outcome"),
        "public_api_schema": {
            "respond": ("self", "query", "files", "ablation"),
            "task_fields": public_inference_schema(),
        },
        "public_cohort_manifest": public_cohort_manifest(),
        "source_custody": {
            "dependencies": dependencies,
            "experiment": source,
        },
        "stop_reason_distribution": _distribution(episodes, "stop_reason"),
        "summaries": {
            "frozen_control": asdict(frozen_control.training_summary),
            "treatment": asdict(treatment.training_summary),
        },
        "total_physical_cell_steps": sum(
            int(row["physical_cell_steps"]) for row in episodes
        ),
    }
    result["result_sha256"] = _digest(_RESULT_NAMESPACE, result)
    return result


def run_and_write_artifact() -> dict[str, object]:
    """Spend the frozen cohort once and exclusively create its result artifact."""

    if _ARTIFACT_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite spent persistent V2 artifact: {_ARTIFACT_PATH}"
        )
    with tempfile.TemporaryDirectory(
        prefix="persistent-variability-python-v2-"
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
    "PublicPersistentVariabilityTask",
    "benchmark_artifact_path",
    "cohort_sha256",
    "dependency_custody",
    "experiment_source_custody_sha256",
    "experiment_source_sha256",
    "public_cohort_manifest",
    "public_inference_schema",
    "public_seed_training_experiences",
    "public_stream_tasks",
    "run_and_write_artifact",
]
