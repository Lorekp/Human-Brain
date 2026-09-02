"""Fresh replay-free persistent multi-source Python repair cohort.

One boundary-aware neuronal motor is seeded from two flat-path public TRAIN
successes.  The same immutable append-only learner then encounters two new
structural regimes (one and two path directories), admits exactly one
evaluator-authenticated successful transition for each regime, and is measured
for transfer and retention without replaying either parent experience.

Every inference call receives only raw issue bytes and the complete lossless
workspace.  Corrected workspaces, target relations, execution outcomes, and
receipts stay on the evaluator side of the model boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

from humanbrain.boundary_multiscale_source_motor_controller import (
    BoundaryMultiscaleSourceMotorAblation,
    BoundaryMultiscaleSourceMotorController,
    SuccessfulBoundaryMultiscaleSourceMotorExperience,
)
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueEventTapeError,
    OpaqueWorkspaceFile,
)
from humanbrain.multiscale_source_motor_online_growth import (
    AppendOnlyMultiscaleSourceMotorStack,
    MultiscaleSourceMotorExecutionAuthority,
    MultiscaleSourceMotorSuccessEvidence,
    OnlineMultiscaleGrowthAblation,
)

_EXPECTED_STDOUT = b"coherent"
_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_RESULT_NAMESPACE = b"humanbrain.persistent-multiscale-python-learning-v1\x00"
_COMMITMENT_NAMESPACE = b"humanbrain.persistent-multiscale-python-transition-v1\x00"
_EVALUATOR_SHA256 = hashlib.sha256(
    b"persistent-multiscale-python-execution-evaluator-v1"
).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicPersistentMultiSourceTask:
    """Complete public task; no target, patch, candidate, or expected output."""

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
    public: PublicPersistentMultiSourceTask
    correct_after: tuple[OpaqueWorkspaceFile, ...]


def _buggy_source(function_name: str) -> bytes:
    return (
        f"def {function_name}():\n"
        '    return "coherent"\n'
        f'print({function_name}(), end="")\n'
        'raise RuntimeError("obsolete")\n'
    ).encode()


def _correct_source(function_name: str) -> bytes:
    return (
        f'def {function_name}():\n    return "coherent"\n'
        f'print({function_name}(), end="")\n'
    ).encode()


def _sorted_workspace(
    rows: tuple[tuple[str, bytes], ...],
) -> tuple[OpaqueWorkspaceFile, ...]:
    return tuple(
        OpaqueWorkspaceFile(path, content)
        for path, content in sorted(rows, key=lambda item: item[0])
    )


def _prepare_task(
    *,
    task_id: str,
    target_path: str,
    function_name: str,
    decoys: tuple[tuple[str, bytes], ...],
) -> _PreparedTask:
    stem = PurePosixPath(target_path).stem
    before = _sorted_workspace((*decoys, (target_path, _buggy_source(function_name))))
    after = tuple(
        OpaqueWorkspaceFile(
            item.path,
            _correct_source(function_name) if item.path == target_path else item.content,
        )
        for item in before
    )
    return _PreparedTask(
        public=PublicPersistentMultiSourceTask(
            task_id=task_id,
            issue=f"repair {stem} using existing bytes".encode(),
            files=before,
        ),
        correct_after=after,
    )


def _base_experience(
    *,
    provenance_id: str,
    target_path: str,
    function_name: str,
    decoys: tuple[tuple[str, bytes], ...],
) -> SuccessfulBoundaryMultiscaleSourceMotorExperience:
    prepared = _prepare_task(
        task_id=provenance_id,
        target_path=target_path,
        function_name=function_name,
        decoys=decoys,
    )
    return SuccessfulBoundaryMultiscaleSourceMotorExperience(
        query=prepared.public.issue,
        before_files=prepared.public.files,
        after_files=prepared.correct_after,
        provenance_id=provenance_id,
    )


def public_base_training_experiences() -> tuple[
    SuccessfulBoundaryMultiscaleSourceMotorExperience, ...
]:
    """Two public flat-path successes used once to grow the frozen base."""

    return (
        _base_experience(
            provenance_id="persistent-train-marzipan",
            target_path="marzipan_worker.py",
            function_name="marzipan_cycle",
            decoys=(
                ("cedar_notes.md", b"# cedar atlas\nretained first note\n"),
                ("quartz_data.json", b'{"quartz": 29, "open": true}\n'),
                ("willow_helper.py", b"WILLOW_LIMIT = 13\n"),
            ),
        ),
        _base_experience(
            provenance_id="persistent-train-halyard",
            target_path="halyard_worker.py",
            function_name="halyard_cycle",
            decoys=(
                ("amber_config.toml", b"enabled = true\nlevel = 7\n"),
                ("brass_notes.txt", b"brass ledger\nretained second note\n"),
                ("zinc_helper.py", b"def zinc_value():\n    return 41\n"),
            ),
        ),
    )


def _cohort_tasks() -> dict[str, _PreparedTask]:
    """Prepare the fresh tasks without exposing their corrections to inference."""

    return {
        "base_before": _prepare_task(
            task_id="persistent-base-opaline",
            target_path="opaline_worker.py",
            function_name="opaline_cycle",
            decoys=(
                ("cedar_map.md", b"# cedar map\nflat transfer decoy\n"),
                ("quartz_state.json", b'{"quartz": [3, 8, 13]}\n'),
                ("willow_helper.py", b"WILLOW_READY = False\n"),
            ),
        ),
        "one_acquire": _prepare_task(
            task_id="persistent-acquire-sirocco",
            target_path="orchard/sirocco_worker.py",
            function_name="sirocco_cycle",
            decoys=(
                ("amber_notes.md", b"# amber orchard\nretained context\n"),
                ("honey/config.toml", b'name = "honey"\nweight = 0.5\n'),
                ("zephyr_helper.py", b"ZEPHYR_SCALE = (2, 5, 8)\n"),
            ),
        ),
        "one_transfer": _prepare_task(
            task_id="persistent-transfer-votive",
            target_path="archive/votive_worker.py",
            function_name="votive_cycle",
            decoys=(
                ("basalt_notes.txt", b"basalt one\nbasalt two\nbasalt three\n"),
                ("mica/config.json", b'{"mica": true, "depth": 6}\n'),
                ("tundra_helper.py", b"TUNDRA_LEVEL = 17\n"),
                ("xenon_table.csv", b"name,value\nxenon,23\n"),
            ),
        ),
        "two_acquire": _prepare_task(
            task_id="persistent-acquire-kestrel",
            target_path="packages/engine/kestrel_worker.py",
            function_name="kestrel_cycle",
            decoys=(
                ("acorn_notes.md", b"# acorn\ntwo-level acquisition\n"),
                ("frost/config.toml", b"limit = 31\nactive = true\n"),
                ("lumen_helper.py", b"LUMEN_BAND = [1, 4, 9]\n"),
                ("zephyr/data.json", b'{"zephyr": "retained"}\n'),
            ),
        ),
        "flat_after": _prepare_task(
            task_id="persistent-retain-umbriel",
            target_path="umbriel_worker.py",
            function_name="umbriel_cycle",
            decoys=(
                ("azure_notes.md", b"# azure\nfresh flat retention\n"),
                ("birch_config.toml", b"enabled = false\nlimit = 19\n"),
                ("morrow_helper.py", b"MORROW_VALUE = 37\n"),
                ("zinnia_data.json", b'{"zinnia": 43}\n'),
            ),
        ),
        "one_after": _prepare_task(
            task_id="persistent-retain-citrine",
            target_path="garden/citrine_worker.py",
            function_name="citrine_cycle",
            decoys=(
                ("cobalt_notes.txt", b"cobalt retained row\n"),
                ("juniper/config.json", b'{"juniper": [5, 12]}\n'),
                ("opal_helper.py", b"OPAL_ACTIVE = True\n"),
                ("willow/table.csv", b"kind,count\nwillow,11\n"),
            ),
        ),
        "two_after": _prepare_task(
            task_id="persistent-transfer-palimpsest",
            target_path="services/core/palimpsest_worker.py",
            function_name="palimpsest_cycle",
            decoys=(
                ("atlas_notes.md", b"# atlas\nfresh two-level transfer\n"),
                ("cinder/config.toml", b"weight = 0.75\n"),
                ("nacre_helper.py", b"NACRE_INDEX = 47\n"),
                ("xylem/data.json", b'{"xylem": false}\n'),
            ),
        ),
        "no_update": _prepare_task(
            task_id="persistent-no-update-quorum",
            target_path="src/domain/tools/quorum_worker.py",
            function_name="quorum_cycle",
            decoys=(
                ("amber/readme.md", b"# amber\nthree-level no-update control\n"),
                ("config/forest.toml", b"threshold = 53\n"),
                ("tests/keel_data.json", b'{"keel": "unchanged"}\n'),
            ),
        ),
    }


def _ambiguity_task() -> PublicPersistentMultiSourceTask:
    return PublicPersistentMultiSourceTask(
        task_id="persistent-ambiguity-pomelo-yarrow",
        issue=b"repair pomelo_worker or yarrow_worker using existing bytes",
        files=_sorted_workspace(
            (
                ("amber_notes.md", b"# amber ambiguity\n"),
                ("modules/alpha/pomelo_worker.py", _buggy_source("pomelo_cycle")),
                ("quartz/config.json", b'{"quartz": 59}\n'),
                ("services/beta/yarrow_worker.py", _buggy_source("yarrow_cycle")),
                ("zinc_helper.py", b"ZINC_READY = True\n"),
            )
        ),
    )


def public_inference_schema() -> tuple[str, ...]:
    return tuple(item.name for item in fields(PublicPersistentMultiSourceTask))


def _issue_related_paths(
    issue: bytes,
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[str, ...]:
    """Evaluator-only lexical equality relation; called only after response."""

    identities = frozenset(_IDENTIFIER.findall(issue))
    return tuple(
        item.path
        for item in files
        if PurePosixPath(item.path).stem.encode("utf-8") in identities
    )


def _workspace_record(
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[dict[str, str], ...]:
    return tuple({"path": item.path, "content_hex": item.content.hex()} for item in files)


def _record_digest(namespace: bytes, record: object) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(namespace + payload).hexdigest()


def _transition_commitment(
    learner_checkpoint_sha256: str,
    prepared: _PreparedTask,
) -> str:
    return _record_digest(
        _COMMITMENT_NAMESPACE,
        {
            "after_files": _workspace_record(prepared.correct_after),
            "before_files": _workspace_record(prepared.public.files),
            "evaluator_sha256": _EVALUATOR_SHA256,
            "learner_checkpoint_sha256": learner_checkpoint_sha256,
            "query_hex": prepared.public.issue.hex(),
            "task_id": prepared.public.task_id,
        },
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
        path for path in sorted(before_by_path) if before_by_path[path] != after_by_path[path]
    )


def _materialize_workspace(
    files: tuple[OpaqueWorkspaceFile, ...],
    root: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for item in files:
        destination = root.joinpath(*PurePosixPath(item.path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.content)


def _execute_python(
    files: tuple[OpaqueWorkspaceFile, ...],
    path: str,
    root: Path,
) -> dict[str, object]:
    if sum(item.path == path for item in files) != 1:
        return {
            "passed": False,
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
        "passed": execution.returncode == 0 and execution.stdout == _EXPECTED_STDOUT,
        "returncode": execution.returncode,
        "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
        "stdout_hex": execution.stdout.hex(),
    }


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


def _model_attempt(
    learner: AppendOnlyMultiscaleSourceMotorStack,
    task: PublicPersistentMultiSourceTask,
    *,
    correct_after: tuple[OpaqueWorkspaceFile, ...] | None,
    ablation: BoundaryMultiscaleSourceMotorAblation,
    expected: str,
    phase: str,
    transfer_scored: bool,
    sanity_only: bool,
    root: Path,
) -> dict[str, object]:
    observation = MultiscaleOpaqueEventTape.observe(task.issue, task.files)
    before = observation.reconstruct_workspace()
    checkpoint_before = learner.checkpoint_sha256

    # This is the entire answer-free model surface.  All scoring starts below.
    decision = learner.respond(task.issue, task.files, ablation=ablation)

    source_occurrences = Counter(observation.motor_source_tokens)
    emitted_occurrences = Counter(decision.emissions)
    emissions_live = all(
        count <= source_occurrences[event] for event, count in emitted_occurrences.items()
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

    # Evaluator-only target relation and execution begin after the model decision.
    related_paths = _issue_related_paths(task.issue, task.files)
    ambiguous = len(related_paths) != 1
    intended_path = None if ambiguous else related_paths[0]
    changed_paths = _changed_paths(before, updated)
    before_execution: dict[str, object] | None = None
    after_execution: dict[str, object] | None = None
    target_position: int | None = None
    target_depth: int | None = None
    decoys_preserved: bool | None = None
    exact_expected_transition: bool | None = None
    if intended_path is not None:
        target_position = tuple(item.path for item in before).index(intended_path)
        target_depth = len(PurePosixPath(intended_path).parts) - 1
        before_execution = _execute_python(before, intended_path, root / "before")
        if decision.decision == "EMIT" and materialization_valid is True:
            after_execution = _execute_python(updated, intended_path, root / "after")
        decoys_preserved = _decoys_preserved(before, updated, intended_path)
        exact_expected_transition = correct_after is not None and updated == correct_after

    correct_edit = (
        decision.decision == "EMIT"
        and materialization_valid is True
        and intended_path is not None
        and before_execution is not None
        and before_execution["passed"] is False
        and after_execution is not None
        and after_execution["passed"] is True
        and changed_paths == (intended_path,)
        and decoys_preserved is True
        and exact_expected_transition is True
        and emissions_live
    )
    physical_stop = (
        decision.decision == "STOP"
        and decision.race.stopped
        and decision.emissions == ()
        and changed_paths == ()
    )
    expectation_met = correct_edit if expected == "EDIT" else physical_stop
    if correct_edit:
        outcome = "correct_edit"
    elif ambiguous and physical_stop:
        outcome = "correct_ambiguity_stop"
    elif physical_stop:
        outcome = "physical_stop"
    elif materialization_valid is not True:
        outcome = "invalid_or_absent_materialization"
    elif intended_path not in changed_paths:
        outcome = "wrong_file_or_no_change"
    else:
        outcome = "behavior_failure"

    return {
        "ablation": ablation.value,
        "after_execution": after_execution,
        "ambiguous": ambiguous,
        "before_execution": before_execution,
        "checkpoint_before": checkpoint_before,
        "checkpoint_retained": learner.checkpoint_sha256 == checkpoint_before,
        "component_count": len(decision.component_decision_sha256s),
        "correct_edit": correct_edit,
        "decoys_preserved": decoys_preserved,
        "decision": decision.decision,
        "decision_sha256": decision.sha256,
        "emission_count": len(decision.emissions),
        "emissions_live": emissions_live,
        "exact_expected_transition": exact_expected_transition,
        "expectation_met": expectation_met,
        "expected": expected,
        "materialization_error": materialization_error,
        "materialization_valid": materialization_valid,
        "model_query_fields": ("issue", "files"),
        "outcome": outcome,
        "phase": phase,
        "physical_cell_steps": decision.physical_cell_steps,
        "physical_stop": physical_stop,
        "query_count": 1,
        "query_sha256": decision.query_sha256,
        "related_path_count": len(related_paths),
        "sanity_only": sanity_only,
        "stop_reason": decision.stop_reason,
        "target_depth": target_depth,
        "target_position": target_position,
        "task_id": task.task_id,
        "transfer_scored": transfer_scored,
        "winning_segment_ordinal": decision.winning_segment_ordinal,
        "winning_selector_cell_id": decision.winning_selector_cell_id,
        "workspace_file_count": len(before),
        "workspace_lossless": before == task.files,
    }


def _run_experiment_at(root: Path) -> dict[str, object]:
    prepared = _cohort_tasks()
    base_controller = BoundaryMultiscaleSourceMotorController.grow(
        public_base_training_experiences()
    )
    learner = AppendOnlyMultiscaleSourceMotorStack.seed(base_controller)
    seed_checkpoint = learner.checkpoint_sha256
    seed_summary = asdict(learner.training_summary)
    episodes: list[dict[str, object]] = []
    execution_audits: list[dict[str, object]] = []
    checkpoint_lineage: list[dict[str, object]] = [
        {
            "event": "seed",
            "checkpoint_sha256": seed_checkpoint,
            "segment_count": learner.training_summary.segment_count,
            "summary": seed_summary,
        }
    ]

    def execution_evaluator(
        *,
        query: bytes,
        before_files: tuple[OpaqueWorkspaceFile, ...],
        after_files: tuple[OpaqueWorkspaceFile, ...],
    ) -> bool:
        audit_ordinal = len(execution_audits)
        related = _issue_related_paths(query, before_files)
        intended = related[0] if len(related) == 1 else None
        changed = _changed_paths(before_files, after_files)
        before_run = (
            None
            if intended is None
            else _execute_python(
                before_files,
                intended,
                root / "authority" / str(audit_ordinal) / "before",
            )
        )
        after_run = (
            None
            if intended is None
            else _execute_python(
                after_files,
                intended,
                root / "authority" / str(audit_ordinal) / "after",
            )
        )
        passed = (
            intended is not None
            and before_run is not None
            and before_run["passed"] is False
            and after_run is not None
            and after_run["passed"] is True
            and changed == (intended,)
            and _decoys_preserved(before_files, after_files, intended)
        )
        execution_audits.append(
            {
                "after_execution": after_run,
                "before_execution": before_run,
                "changed_paths": changed,
                "decoys_preserved": (
                    False
                    if intended is None
                    else _decoys_preserved(before_files, after_files, intended)
                ),
                "intended_path": intended,
                "passed": passed,
                "query_sha256": hashlib.sha256(query).hexdigest(),
            }
        )
        return passed

    authority = MultiscaleSourceMotorExecutionAuthority(
        evaluator=execution_evaluator,
        evaluator_sha256=_EVALUATOR_SHA256,
    )

    def attempt(
        key: str,
        *,
        phase: str,
        expected: str,
        transfer_scored: bool = False,
        sanity_only: bool = False,
        ablation: BoundaryMultiscaleSourceMotorAblation = (
            BoundaryMultiscaleSourceMotorAblation.NONE
        ),
    ) -> dict[str, object]:
        item = prepared[key]
        row = _model_attempt(
            learner,
            item.public,
            correct_after=item.correct_after,
            ablation=ablation,
            expected=expected,
            phase=phase,
            transfer_scored=transfer_scored,
            sanity_only=sanity_only,
            root=root / "episodes" / phase,
        )
        episodes.append(row)
        return row

    # Fresh flat transfer must work before online structural growth.
    base_transfer = attempt(
        "base_before",
        phase="base_transfer_before_updates",
        expected="EDIT",
        transfer_scored=True,
    )

    # The correct one-directory transition is frozen before the first query.
    one = prepared["one_acquire"]
    one_parent = learner.checkpoint_sha256
    one_commitment = _transition_commitment(one_parent, one)
    one_initial = attempt(
        "one_acquire",
        phase="one_directory_initial",
        expected="STOP",
    )
    one_execution_receipt = authority.execute_and_issue(
        learner_checkpoint_sha256=one_parent,
        query=one.public.issue,
        before_files=one.public.files,
        after_files=one.correct_after,
    )
    one_evidence = MultiscaleSourceMotorSuccessEvidence.seal(
        query=one.public.issue,
        before_files=one.public.files,
        after_files=one.correct_after,
        execution_receipt=one_execution_receipt,
        provenance_id="persistent-success-sirocco",
    )
    one_transition = learner.extend_success(one_evidence)
    one_receipt = one_transition.receipt
    learner = one_transition.successor
    checkpoint_lineage.append(
        {
            "event": "append_one_directory",
            "precommit_sha256": one_commitment,
            "receipt": asdict(one_receipt),
            "summary": asdict(learner.training_summary),
        }
    )
    one_retry = attempt(
        "one_acquire",
        phase="one_directory_exact_retry_sanity",
        expected="EDIT",
        sanity_only=True,
    )
    one_transfer = attempt(
        "one_transfer",
        phase="one_directory_fresh_transfer",
        expected="EDIT",
        transfer_scored=True,
    )

    # A deeper path is novel to both retained segments; freeze its correction first.
    two = prepared["two_acquire"]
    two_parent = learner.checkpoint_sha256
    two_commitment = _transition_commitment(two_parent, two)
    two_initial = attempt(
        "two_acquire",
        phase="two_directory_initial",
        expected="STOP",
    )
    two_execution_receipt = authority.execute_and_issue(
        learner_checkpoint_sha256=two_parent,
        query=two.public.issue,
        before_files=two.public.files,
        after_files=two.correct_after,
    )
    two_evidence = MultiscaleSourceMotorSuccessEvidence.seal(
        query=two.public.issue,
        before_files=two.public.files,
        after_files=two.correct_after,
        execution_receipt=two_execution_receipt,
        provenance_id="persistent-success-kestrel",
    )
    two_transition = learner.extend_success(two_evidence)
    two_receipt = two_transition.receipt
    learner = two_transition.successor
    checkpoint_lineage.append(
        {
            "event": "append_two_directory",
            "precommit_sha256": two_commitment,
            "receipt": asdict(two_receipt),
            "summary": asdict(learner.training_summary),
        }
    )
    two_retry = attempt(
        "two_acquire",
        phase="two_directory_exact_retry_sanity",
        expected="EDIT",
        sanity_only=True,
    )

    # Distinct fresh tasks probe all three retained path-depth regimes.
    flat_after = attempt(
        "flat_after",
        phase="flat_base_retention_after_update_two",
        expected="EDIT",
        transfer_scored=True,
    )
    one_after = attempt(
        "one_after",
        phase="one_directory_retention_transfer_after_update_two",
        expected="EDIT",
        transfer_scored=True,
    )
    two_after = attempt(
        "two_after",
        phase="two_directory_fresh_transfer_after_update_two",
        expected="EDIT",
        transfer_scored=True,
    )

    ambiguity = _model_attempt(
        learner,
        _ambiguity_task(),
        correct_after=None,
        ablation=BoundaryMultiscaleSourceMotorAblation.NONE,
        expected="STOP",
        phase="genuine_ambiguity",
        transfer_scored=False,
        sanity_only=False,
        root=root / "episodes" / "genuine_ambiguity",
    )
    episodes.append(ambiguity)

    # The NO_UPDATE branch authenticates the same kind of success but commits zero state.
    no_update_item = prepared["no_update"]
    no_update_parent = learner
    no_update_parent_root = no_update_parent.checkpoint_sha256
    no_update_summary_before = asdict(no_update_parent.training_summary)
    no_update_commitment = _transition_commitment(no_update_parent_root, no_update_item)
    no_update_initial = attempt(
        "no_update",
        phase="no_update_initial",
        expected="STOP",
    )
    no_update_execution_receipt = authority.execute_and_issue(
        learner_checkpoint_sha256=no_update_parent_root,
        query=no_update_item.public.issue,
        before_files=no_update_item.public.files,
        after_files=no_update_item.correct_after,
    )
    no_update_evidence = MultiscaleSourceMotorSuccessEvidence.seal(
        query=no_update_item.public.issue,
        before_files=no_update_item.public.files,
        after_files=no_update_item.correct_after,
        execution_receipt=no_update_execution_receipt,
        provenance_id="persistent-no-update-quorum",
    )
    no_update_transition = learner.extend_success(
        no_update_evidence,
        ablation=OnlineMultiscaleGrowthAblation.NO_UPDATE,
    )
    no_update_receipt = no_update_transition.receipt
    learner = no_update_transition.successor
    no_update_retry = attempt(
        "no_update",
        phase="no_update_exact_retry",
        expected="STOP",
        sanity_only=True,
    )

    ablation_rows = tuple(
        attempt(
            "two_after",
            phase=f"final_{ablation.value}",
            expected="STOP",
            ablation=ablation,
        )
        for ablation in (
            BoundaryMultiscaleSourceMotorAblation.EQUALITY_OFF,
            BoundaryMultiscaleSourceMotorAblation.PROPAGATION_OFF,
            BoundaryMultiscaleSourceMotorAblation.BOUNDARY_OFF,
        )
    )

    transfer_rows = tuple(row for row in episodes if row["transfer_scored"])
    transfer_passed = sum(bool(row["expectation_met"]) for row in transfer_rows)
    non_sanity_rows = tuple(row for row in episodes if not row["sanity_only"])
    failure_distribution = dict(sorted(Counter(row["outcome"] for row in episodes).items()))
    unmet_distribution = dict(
        sorted(
            Counter(row["phase"] for row in non_sanity_rows if not row["expectation_met"]).items()
        )
    )
    final_summary = asdict(learner.training_summary)
    append_receipts = (one_receipt, two_receipt)
    append_accounting_valid = (
        all(receipt.status == "APPENDED" for receipt in append_receipts)
        and all(receipt.parent_experience_replay_count == 0 for receipt in append_receipts)
        and all(receipt.raw_training_sequences_retained == 0 for receipt in append_receipts)
        and all(receipt.source_experience_count == 1 for receipt in append_receipts)
        and all(receipt.current_experience_training_rows == 2 for receipt in append_receipts)
        and all(receipt.committed_evidence_tick_phases == 2 for receipt in append_receipts)
        and all(
            receipt.excitatory_support_tick_estimate_added
            == receipt.coactivation_support_tick_estimate_added
            + receipt.positive_success_support_tick_estimate_added
            for receipt in append_receipts
        )
        and two_receipt.reserved_cell_address_start
        == one_receipt.reserved_cell_address_start
        + one_receipt.reserved_cell_capacity_estimate_added
        and two_receipt.reserved_synapse_address_start
        == one_receipt.reserved_synapse_address_start
        + one_receipt.reserved_synapse_capacity_estimate_added
        and final_summary["segment_count"] == 3
    )
    no_update_zero_state = (
        no_update_transition.successor is no_update_parent
        and no_update_receipt.status == "NO_UPDATE"
        and no_update_receipt.parent_checkpoint_sha256 == no_update_parent_root
        and no_update_receipt.successor_checkpoint_sha256 == no_update_parent_root
        and no_update_receipt.successor_provenance_sha256
        == no_update_receipt.prior_provenance_sha256
        and no_update_receipt.reserved_cell_capacity_estimate_added == 0
        and no_update_receipt.reserved_synapse_capacity_estimate_added == 0
        and no_update_receipt.learned_event_bytes_retained_estimate_added == 0
        and no_update_receipt.excitatory_support_tick_estimate_added == 0
        and no_update_receipt.committed_evidence_tick_phases == 0
        and asdict(learner.training_summary) == no_update_summary_before
    )
    integrity = {
        "all_checkpoints_read_only_during_inference": all(
            row["checkpoint_retained"] for row in episodes
        ),
        "all_emissions_live": all(row["emissions_live"] for row in episodes),
        "all_model_query_fields": all(
            row["model_query_fields"] == ("issue", "files") for row in episodes
        ),
        "all_queries_single_shot": all(row["query_count"] == 1 for row in episodes),
        "all_workspaces_four_or_five_files": all(
            row["workspace_file_count"] in (4, 5) for row in episodes
        ),
        "all_workspaces_lossless": all(row["workspace_lossless"] for row in episodes),
        "append_accounting_valid": append_accounting_valid,
        "execution_audits_all_passed": all(row["passed"] for row in execution_audits),
        "no_update_zero_state": no_update_zero_state,
        "public_task_fields": public_inference_schema(),
    }
    gate_passed = (
        base_transfer["expectation_met"] is True
        and one_initial["expectation_met"] is True
        and one_receipt.status == "APPENDED"
        and one_retry["expectation_met"] is True
        and one_transfer["expectation_met"] is True
        and two_initial["expectation_met"] is True
        and two_receipt.status == "APPENDED"
        and two_retry["expectation_met"] is True
        and flat_after["expectation_met"] is True
        and one_after["expectation_met"] is True
        and two_after["expectation_met"] is True
        and ambiguity["outcome"] == "correct_ambiguity_stop"
        and no_update_initial["expectation_met"] is True
        and no_update_retry["expectation_met"] is True
        and all(row["expectation_met"] for row in ablation_rows)
        and transfer_passed == len(transfer_rows) == 5
        and len(execution_audits) == 3
        and all(integrity.values())
        and integrity["public_task_fields"] == ("task_id", "issue", "files")
    )
    result: dict[str, object] = {
        "checkpoint_lineage": checkpoint_lineage,
        "claim": (
            "One persistent append-only neuronal motor retained flat-path transfer "
            "while acquiring and transferring one- and two-directory path envelopes "
            "from two evaluator-authenticated successes without parent replay."
        ),
        "episodes": episodes,
        "execution_audits": execution_audits,
        "failure_distribution": failure_distribution,
        "final_checkpoint_sha256": learner.checkpoint_sha256,
        "final_summary": final_summary,
        "gate_passed": gate_passed,
        "integrity": integrity,
        "limitations": (
            "Exact opaque identifier equality supplies issue-to-path grounding.",
            "Each append stores a separate path-depth motor shard rather than consolidating one schema.",
            "The motor reuses one demonstrated contiguous prefix-copy edit program.",
            "Acquisition retries are sanity checks and are excluded from transfer scoring.",
            "This is a small public TRAIN cohort, not broad language, repository, or SWE evidence.",
        ),
        "no_update": {
            "precommit_sha256": no_update_commitment,
            "receipt": asdict(no_update_receipt),
            "summary_before": no_update_summary_before,
            "zero_state_committed": no_update_zero_state,
        },
        "seed_checkpoint_sha256": seed_checkpoint,
        "seed_summary": seed_summary,
        "transfer_score": {
            "attempted": len(transfer_rows),
            "passed": transfer_passed,
            "sanity_retries_excluded": 2,
        },
        "unmet_expectation_distribution": unmet_distribution,
    }
    result["result_sha256"] = _record_digest(_RESULT_NAMESPACE, result)
    return result


def run_experiment(root: Path | None = None) -> dict[str, object]:
    """Run this one fresh persistent public-TRAIN cohort without an artifact."""

    if root is not None:
        return _run_experiment_at(root)
    with tempfile.TemporaryDirectory(prefix="persistent-multiscale-python-v1-") as directory:
        return _run_experiment_at(Path(directory))


def main() -> None:
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PublicPersistentMultiSourceTask",
    "public_base_training_experiences",
    "public_inference_schema",
    "run_experiment",
]
