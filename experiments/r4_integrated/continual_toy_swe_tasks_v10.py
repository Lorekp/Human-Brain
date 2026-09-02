"""Fresh public-TRAIN repository tasks for integrated continual coding V10.

The twelve tasks exercise direct repository repair, consequence-driven revision,
history-required renamed transfer, and genuine ambiguity.  Learners receive only
one public issue, an authenticated repository commitment, and the uniform patch
tool contract.  Family, stage, lineage, witnesses, review plans, and ambiguity
certificates remain evaluator-private.

This fixture imports or runs no learner/model and creates no benchmark artifact.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from textwrap import dedent

from humanbrain.simple_swe.protocol import ExecutionLimits
from humanbrain.simple_swe.sandbox import execute_frozen_worker, resolve_python_executable

SCHEMA = "continual-repository-swe-tasks-v10"
MANIFEST_SCHEMA = "continual-repository-swe-custody-manifest-v10"
PARTITION = "train"
TASK_COUNT = 12
SOURCE_PRIOR_CORPUS_SHA256 = "81a60454b9af6b8592ea8b422c25bdd98115c101e5580c8cd571555fdf0abfb1"
SOURCE_PRIOR_CASE_COUNT = 8
MANIFEST_FILENAME = "continual_toy_swe_tasks_v10_manifest.json"

SUBMIT_PATCH = "submit_patch"
STOP = "<stop>"
INVALID_ACTION = "invalid"
GLOBAL_ACTIONS = (SUBMIT_PATCH,)
MAX_PATCH_SUBMISSIONS = 3

_DIRECT = "direct-repository-relation"
_TRANSFER = "history-required-transfer"
_REVISION = "consequence-guided-revision"
_INSUFFICIENT = "genuinely-ambiguous-stop"
_STAGES = (_DIRECT, _TRANSFER, _REVISION, _INSUFFICIENT)

_JOURNAL = "committed-mutation-journal"
_GRAPH = "dependency-postorder"
_OVERLAY = "sparse-overlay-policy"
_SCANNER = "escape-aware-field-scan"
_ENDIAN = "unresolved-byte-order"
_TREE = "typed-scalar-walk"
_SCHEDULER = "tenant-scheduling-policy"
_TIMEZONE = "unresolved-timezone-policy"
_FAMILIES = (_JOURNAL, _GRAPH, _OVERLAY, _SCANNER, _ENDIAN, _TREE, _SCHEDULER, _TIMEZONE)

_SUCCESS = "success"
_FAILURE = "failure"
_ACTION_REQUIRED = "action-required"
_NEUTRAL = "neutral"
_RECEIPT_CONCLUSIONS = (_SUCCESS, _FAILURE, _ACTION_REQUIRED)

_RELATION_JOURNAL = sha256(b"v10:committed-mutations-only").hexdigest()
_RELATION_GRAPH = sha256(b"v10:dependency-postorder-dedup-cycle-tolerant").hexdigest()
_RELATION_SCANNER = sha256(b"v10:escape-consumes-next-terminal-retained").hexdigest()
_RELATION_TREE = sha256(b"v10:typed-path-stable-recursive-scalars").hexdigest()
_RECEIPT_GENESIS_SHA256 = sha256(b"v10:action-receipt-genesis").hexdigest()

_REPOSITORY_WORKER_SCHEMA = "humanbrain.simple-swe-repository-worker.v1"
_REPOSITORY_RESULT_SCHEMA = "humanbrain.simple-swe-repository-result.v1"
_REPOSITORY_WORKER_PATH = (
    Path(__file__).parents[2] / "src" / "humanbrain" / "simple_swe" / "_repository_worker.py"
)
_SANDBOX_SOURCE_PATH = (
    Path(__file__).parents[2] / "src" / "humanbrain" / "simple_swe" / "sandbox.py"
)
REPOSITORY_WORKER_SHA256 = "0a130dd6123704fa7905bd79b6f5d94d4cea8d3f1263d2a565c99c8319a5ac9f"
SIMPLE_SWE_SANDBOX_SHA256 = "37cbf4667d51f43d343406735d0252d8672da2ab773ea0c914c14ba216ab370b"
PYTHON_EXECUTABLE_SHA256 = "6765c6b1685c86877fabe14d82240f3fab2913a617a85e8d09e61dc40798013b"
_EXECUTION_LIMITS = ExecutionLimits(
    wall_seconds=5.0,
    cpu_seconds=2,
    memory_mib=128,
    processes=1,
    output_bytes=4096,
    max_completion_bytes=65536,
    max_ast_nodes=4096,
    python_hash_seed=0,
)


class ContinualToySweTaskV10Error(ValueError):
    """A V10 task, repository, receipt, or custody record is unauthenticated."""


class CandidatePatchRejectedV10(ContinualToySweTaskV10Error):
    """A candidate diff is invalid; evaluator/custody faults use the base type."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source(text: str) -> str:
    return dedent(text).strip("\n") + "\n"


def _frozen_json(raw: str) -> bytes:
    return _canonical_json(json.loads(raw))


def _tool_contract_record() -> dict[str, object]:
    return {
        "schema": "continual-repository-swe-tool-contract-v10",
        "external_mutating_affordances": [
            {
                "name": SUBMIT_PATCH,
                "payload": {
                    "type": "object",
                    "required": ["patch"],
                    "additionalProperties": False,
                    "properties": {"patch": {"type": "string"}},
                },
            }
        ],
        "max_patch_submissions": MAX_PATCH_SUBMISSIONS,
        "instructions": (
            "Inspect the learner-readable repository. Submit one complete unified diff when "
            "public evidence determines a change. The neuronal <stop> choice is terminal and "
            "neutral. Acceptance returns only a conclusion unless a reviewer provides new "
            "public text."
        ),
        "learner_readable_files": "the exact task mount allowlist, including public tests",
        "writable_files": "existing Python files inside the repository package",
        "patch_paths": (
            "canonical repository-relative POSIX paths from the writable source allowlist; "
            "tests, configuration, additions, deletions, renames, modes, binaries, absolute "
            "paths, and traversal are rejected"
        ),
        "patch_hashing": "the runner computes sha256 from the submitted UTF-8 diff",
        "terminal_neutral_choice": STOP,
    }


TOOL_CONTRACT_SHA256 = _digest(
    b"continual-repository-swe-tool-contract-v10", _tool_contract_record()
)


@dataclass(frozen=True, slots=True)
class RepositoryFileV10:
    relative_path: str
    content_utf8: bytes
    file_sha256: str

    def record(self) -> dict[str, str]:
        return {"relative_path": self.relative_path, "file_sha256": self.file_sha256}

    def authenticate(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
        ):
            raise ContinualToySweTaskV10Error("repository path is not canonical")
        if type(self.content_utf8) is not bytes:
            raise TypeError("repository content must be immutable bytes")
        if sha256(self.content_utf8).hexdigest() != self.file_sha256:
            raise ContinualToySweTaskV10Error("repository file changed")


@dataclass(frozen=True, slots=True)
class RepositorySnapshotV10:
    files: tuple[RepositoryFileV10, ...]
    snapshot_sha256: str

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "files": tuple(file.record() for file in self.files)}

    def authenticate(self) -> None:
        if type(self.files) is not tuple or not self.files:
            raise ContinualToySweTaskV10Error("repository snapshot is empty or mutable")
        for file in self.files:
            if type(file) is not RepositoryFileV10:
                raise TypeError("repository snapshot contains a foreign file type")
            file.authenticate()
        paths = tuple(file.relative_path for file in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ContinualToySweTaskV10Error("repository paths changed")
        if self.snapshot_sha256 != _digest(b"continual-repository-snapshot-v10", self.record()):
            raise ContinualToySweTaskV10Error("repository snapshot changed")


@dataclass(frozen=True, slots=True)
class LearnerVisibleToySweObservationV10:
    """The complete per-task record allowed into learner input."""

    task_commitment_sha256: str
    instance_id: str
    issue: str
    repository_snapshot_sha256: str
    tool_contract_sha256: str
    observation_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_commitment_sha256": self.task_commitment_sha256,
            "instance_id": self.instance_id,
            "issue": self.issue,
            "repository_snapshot_sha256": self.repository_snapshot_sha256,
            "tool_contract_sha256": self.tool_contract_sha256,
        }

    def authenticate(self) -> None:
        if not all(
            _is_sha256(value)
            for value in (
                self.task_commitment_sha256,
                self.repository_snapshot_sha256,
                self.tool_contract_sha256,
                self.observation_sha256,
            )
        ):
            raise ContinualToySweTaskV10Error("learner observation commitment changed")
        if not self.instance_id or not self.issue:
            raise ContinualToySweTaskV10Error("learner observation is incomplete")
        if self.tool_contract_sha256 != TOOL_CONTRACT_SHA256:
            raise ContinualToySweTaskV10Error("task-specific action contract is forbidden")
        expected = _digest(b"continual-repository-swe-observation-v10", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskV10Error("learner observation changed")


@dataclass(frozen=True, slots=True)
class FrozenWitnessV10:
    case_id: str
    request_json: bytes
    expected_json: bytes
    witness_sha256: str

    def record(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "request_json": self.request_json.decode("ascii"),
            "expected_json": self.expected_json.decode("ascii"),
        }

    def authenticate(self) -> None:
        if type(self.request_json) is not bytes or type(self.expected_json) is not bytes:
            raise TypeError("witness JSON must be immutable bytes")
        if _frozen_json(self.request_json.decode("ascii")) != self.request_json:
            raise ContinualToySweTaskV10Error("witness request is not canonical")
        if _frozen_json(self.expected_json.decode("ascii")) != self.expected_json:
            raise ContinualToySweTaskV10Error("witness expectation is not canonical")
        if self.witness_sha256 != _digest(b"continual-repository-swe-witness-v10", self.record()):
            raise ContinualToySweTaskV10Error("acceptance witness changed")


@dataclass(frozen=True, slots=True)
class FrozenEvaluatorV10:
    entry_module: str
    entry_point: str
    initial_cases: tuple[FrozenWitnessV10, ...]
    revised_cases: tuple[FrozenWitnessV10, ...] | None
    evaluator_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "worker_sha256": REPOSITORY_WORKER_SHA256,
            "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
            "entry_module": self.entry_module,
            "entry_point": self.entry_point,
            "initial_case_sha256s": tuple(case.witness_sha256 for case in self.initial_cases),
            "revised_case_sha256s": (
                None
                if self.revised_cases is None
                else tuple(case.witness_sha256 for case in self.revised_cases)
            ),
        }

    def phase_sha256(self, *, revised: bool) -> str:
        cases = self.revised_cases if revised else self.initial_cases
        if cases is None:
            raise ContinualToySweTaskV10Error("revised evaluator cases are unavailable")
        return _digest(
            b"continual-repository-swe-evaluator-phase-v10",
            {
                "worker_sha256": REPOSITORY_WORKER_SHA256,
                "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
                "entry_module": self.entry_module,
                "entry_point": self.entry_point,
                "case_sha256s": tuple(case.witness_sha256 for case in cases),
            },
        )

    def authenticate(self) -> None:
        if (
            not self.entry_module
            or any(
                not part.isidentifier() or part.startswith("__")
                for part in self.entry_module.split(".")
            )
            or not self.entry_point.isidentifier()
            or self.entry_point.startswith("__")
        ):
            raise ContinualToySweTaskV10Error("private evaluator entry point changed")
        for cases in (self.initial_cases, self.revised_cases):
            if cases is not None:
                if type(cases) is not tuple or len(cases) < 2:
                    raise ContinualToySweTaskV10Error("private evaluator cases changed")
                for case in cases:
                    if type(case) is not FrozenWitnessV10:
                        raise TypeError("private evaluator case type changed")
                    case.authenticate()
        if self.evaluator_sha256 != _digest(
            b"continual-repository-swe-evaluator-v10", self.record()
        ):
            raise ContinualToySweTaskV10Error("private evaluator changed")


@dataclass(frozen=True, slots=True)
class ReviewPlanV10:
    review_text: str
    revised_witnesses: tuple[FrozenWitnessV10, ...]
    plan_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "review_text": self.review_text,
            "revised_witness_sha256s": tuple(
                witness.witness_sha256 for witness in self.revised_witnesses
            ),
        }

    def authenticate(self, initial_witnesses: tuple[FrozenWitnessV10, ...]) -> None:
        if not self.review_text or len(self.revised_witnesses) < 2:
            raise ContinualToySweTaskV10Error("review plan lacks public evidence")
        for witness in self.revised_witnesses:
            witness.authenticate()
        initial = {row.request_json: row.expected_json for row in initial_witnesses}
        revised = {row.request_json: row.expected_json for row in self.revised_witnesses}
        shared = initial.keys() & revised.keys()
        if not shared or not any(initial[key] != revised[key] for key in shared):
            raise ContinualToySweTaskV10Error("review does not change shared behavior")
        if self.plan_sha256 != _digest(b"continual-repository-swe-review-plan-v10", self.record()):
            raise ContinualToySweTaskV10Error("review plan changed")


@dataclass(frozen=True, slots=True)
class AmbiguityCertificateV10:
    evidence_fingerprint: str
    witness_request_json: bytes
    compatible_outcome_jsons: tuple[bytes, bytes]
    certificate_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "evidence_fingerprint": self.evidence_fingerprint,
            "witness_request_json": self.witness_request_json.decode("ascii"),
            "compatible_outcome_jsons": tuple(
                outcome.decode("ascii") for outcome in self.compatible_outcome_jsons
            ),
        }

    def authenticate(self) -> None:
        if not self.evidence_fingerprint:
            raise ContinualToySweTaskV10Error("ambiguity evidence is empty")
        if _frozen_json(self.witness_request_json.decode("ascii")) != self.witness_request_json:
            raise ContinualToySweTaskV10Error("ambiguity request is not canonical")
        if len(set(self.compatible_outcome_jsons)) != 2:
            raise ContinualToySweTaskV10Error("ambiguity outcomes do not disagree")
        for outcome in self.compatible_outcome_jsons:
            if _frozen_json(outcome.decode("ascii")) != outcome:
                raise ContinualToySweTaskV10Error("ambiguity outcome is not canonical")
        if self.certificate_sha256 != _digest(
            b"continual-repository-swe-ambiguity-v10", self.record()
        ):
            raise ContinualToySweTaskV10Error("ambiguity certificate changed")


@dataclass(frozen=True, slots=True)
class PrivateAcceptanceV10:
    family_id: str
    stage: str
    expected_action: str
    retention_anchor_ordinal: int | None
    transfer_relation_sha256: str | None
    witnesses: tuple[FrozenWitnessV10, ...]
    evaluator: FrozenEvaluatorV10 | None
    review_plan: ReviewPlanV10 | None
    ambiguity_certificate: AmbiguityCertificateV10 | None
    bundle_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "family_id": self.family_id,
            "stage": self.stage,
            "expected_action": self.expected_action,
            "retention_anchor_ordinal": self.retention_anchor_ordinal,
            "transfer_relation_sha256": self.transfer_relation_sha256,
            "witness_sha256s": tuple(witness.witness_sha256 for witness in self.witnesses),
            "evaluator_sha256": (
                None if self.evaluator is None else self.evaluator.evaluator_sha256
            ),
            "review_plan_sha256": (
                None if self.review_plan is None else self.review_plan.plan_sha256
            ),
            "ambiguity_certificate_sha256": (
                None
                if self.ambiguity_certificate is None
                else self.ambiguity_certificate.certificate_sha256
            ),
        }

    def authenticate(self, ordinal: int) -> None:
        if self.family_id not in _FAMILIES or self.stage not in _STAGES:
            raise ContinualToySweTaskV10Error("private family or stage changed")
        if self.expected_action not in (SUBMIT_PATCH, STOP):
            raise ContinualToySweTaskV10Error("private expected action changed")
        for witness in self.witnesses:
            witness.authenticate()
        if self.expected_action == STOP:
            if (
                self.stage != _INSUFFICIENT
                or self.witnesses
                or self.evaluator is not None
                or self.review_plan is not None
                or self.ambiguity_certificate is None
            ):
                raise ContinualToySweTaskV10Error("STOP task is not evidence-limited")
            self.ambiguity_certificate.authenticate()
        else:
            if (
                not self.witnesses
                or self.evaluator is None
                or self.ambiguity_certificate is not None
            ):
                raise ContinualToySweTaskV10Error("executable acceptance changed")
            self.evaluator.authenticate()
            if self.evaluator.initial_cases != self.witnesses:
                raise ContinualToySweTaskV10Error("initial evaluator cases changed")
            if self.stage == _REVISION:
                if self.review_plan is None or self.evaluator.revised_cases is None:
                    raise ContinualToySweTaskV10Error("revision task lost review evidence")
                self.review_plan.authenticate(self.witnesses)
                if self.evaluator.revised_cases != self.review_plan.revised_witnesses:
                    raise ContinualToySweTaskV10Error("revised evaluator cases changed")
            elif self.review_plan is not None or self.evaluator.revised_cases is not None:
                raise ContinualToySweTaskV10Error("non-revision task gained review evidence")
        if self.stage == _TRANSFER:
            if (
                type(self.retention_anchor_ordinal) is not int
                or not 0 <= self.retention_anchor_ordinal < ordinal
                or not _is_sha256(self.transfer_relation_sha256)
            ):
                raise ContinualToySweTaskV10Error("history-transfer lineage changed")
        elif self.retention_anchor_ordinal is not None:
            raise ContinualToySweTaskV10Error("non-transfer task gained an anchor")
        if self.expected_action == STOP and self.transfer_relation_sha256 is not None:
            raise ContinualToySweTaskV10Error("STOP task gained a transfer relation")
        if self.bundle_sha256 != _digest(b"continual-repository-swe-private-v10", self.record()):
            raise ContinualToySweTaskV10Error("private acceptance bundle changed")


@dataclass(frozen=True, slots=True)
class _EvaluatorTaskV10:
    ordinal: int
    public: LearnerVisibleToySweObservationV10
    repository: RepositorySnapshotV10
    private: PrivateAcceptanceV10

    def authenticate(self) -> None:
        self.public.authenticate()
        self.repository.authenticate()
        self.private.authenticate(self.ordinal)
        expected = _digest(
            b"continual-repository-swe-task-v10",
            {
                "schema": SCHEMA,
                "partition": PARTITION,
                "instance_id": self.public.instance_id,
                "issue": self.public.issue,
                "repository_snapshot_sha256": self.repository.snapshot_sha256,
                "tool_contract_sha256": TOOL_CONTRACT_SHA256,
                "private_bundle_sha256": self.private.bundle_sha256,
            },
        )
        if self.public.task_commitment_sha256 != expected:
            raise ContinualToySweTaskV10Error("public task is not bound to private acceptance")
        if self.public.repository_snapshot_sha256 != self.repository.snapshot_sha256:
            raise ContinualToySweTaskV10Error("public task is not bound to its repository")


@dataclass(frozen=True, slots=True)
class ActionReceiptV10:
    task_commitment_sha256: str
    sequence: int
    prior_receipt_sha256: str
    action: str
    action_sha256: str
    evaluator_sha256: str
    sandbox_evaluator_reached: bool
    conclusion: str
    feedback: str
    receipt_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": "continual-repository-swe-action-receipt-v10",
            "task_commitment_sha256": self.task_commitment_sha256,
            "sequence": self.sequence,
            "prior_receipt_sha256": self.prior_receipt_sha256,
            "action": self.action,
            "action_sha256": self.action_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "sandbox_evaluator_reached": self.sandbox_evaluator_reached,
            "conclusion": self.conclusion,
            "feedback": self.feedback,
        }

    def authenticate(self) -> None:
        if (
            not all(
                _is_sha256(value)
                for value in (
                    self.task_commitment_sha256,
                    self.prior_receipt_sha256,
                    self.action_sha256,
                    self.evaluator_sha256,
                )
            )
            or type(self.sequence) is not int
            or not 1 <= self.sequence <= MAX_PATCH_SUBMISSIONS
            or type(self.sandbox_evaluator_reached) is not bool
            or self.action != SUBMIT_PATCH
            or self.conclusion not in _RECEIPT_CONCLUSIONS
            or not self.feedback
        ):
            raise ContinualToySweTaskV10Error("action receipt fields changed")
        if self.receipt_sha256 != _digest(
            b"continual-repository-swe-action-receipt-v10", self.record()
        ):
            raise ContinualToySweTaskV10Error("action receipt changed")


@dataclass(frozen=True, slots=True)
class SealedActionSummaryV10:
    task_commitment_sha256: str
    attempt: int
    action: str
    conclusion: str
    receipt: ActionReceiptV10 | None
    summary_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": "continual-repository-swe-sealed-action-v10",
            "task_commitment_sha256": self.task_commitment_sha256,
            "attempt": self.attempt,
            "action": self.action,
            "conclusion": self.conclusion,
            "receipt_sha256": None if self.receipt is None else self.receipt.receipt_sha256,
        }

    def authenticate(self) -> None:
        if (
            not _is_sha256(self.task_commitment_sha256)
            or type(self.attempt) is not int
            or not 1 <= self.attempt <= MAX_PATCH_SUBMISSIONS
            or self.action not in (SUBMIT_PATCH, STOP, INVALID_ACTION)
        ):
            raise ContinualToySweTaskV10Error("sealed action fields changed")
        if self.receipt is None:
            expected_conclusion = _NEUTRAL if self.action == STOP else _FAILURE
            if self.conclusion != expected_conclusion:
                raise ContinualToySweTaskV10Error("unreceipted action conclusion changed")
        else:
            if type(self.receipt) is not ActionReceiptV10 or self.action != SUBMIT_PATCH:
                raise ContinualToySweTaskV10Error("sealed receipt action changed")
            self.receipt.authenticate()
            if (
                self.receipt.task_commitment_sha256 != self.task_commitment_sha256
                or self.receipt.conclusion != self.conclusion
            ):
                raise ContinualToySweTaskV10Error("sealed receipt conclusion changed")
        if self.summary_sha256 != _digest(
            b"continual-repository-swe-sealed-action-v10", self.record()
        ):
            raise ContinualToySweTaskV10Error("sealed action summary changed")


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    instance_id: str
    issue: str
    family_id: str
    stage: str
    files: tuple[tuple[str, str], ...]
    expected_action: str
    witnesses: tuple[tuple[str, str, str], ...]
    entry_module: str | None
    entry_point: str | None
    review_text: str | None = None
    revised_witnesses: tuple[tuple[str, str, str], ...] = ()
    retention_anchor_ordinal: int | None = None
    transfer_relation_sha256: str | None = None
    ambiguity: tuple[str, str, str, str] | None = None


_PYPROJECT = _source(
    """
    [project]
    name = "fresh-public-train-repository"
    version = "0.0.0"
    requires-python = ">=3.11"

    [tool.pytest.ini_options]
    pythonpath = ["."]
    testpaths = ["tests"]
    """
)


def _case(case_id: str, request: object, expected: object) -> tuple[str, str, str]:
    return (
        case_id,
        _canonical_json(request).decode("ascii"),
        _canonical_json(expected).decode("ascii"),
    )


def _repository(
    package: str,
    readme: str,
    modules: tuple[tuple[str, str], ...],
    public_test_name: str,
    public_test: str,
    *,
    documents: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    files = [
        ("README.md", _source(readme)),
        ("pyproject.toml", _PYPROJECT),
        (f"{package}/__init__.py", ""),
        *((f"{package}/{name}", _source(source)) for name, source in modules),
        *((f"docs/{name}", _source(source)) for name, source in documents),
        (f"tests/{public_test_name}", _source(public_test)),
    ]
    return tuple(sorted(files))


def _ember_journal_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "emberjournal",
        """
        # Ember mutation journal

        Mutation records arrive in named batches. The public contract distinguishes
        staged writes, committed batches, and discarded batches.
        """,
        (
            (
                "replay.py",
                """
                def materialize(entries: list[dict[str, object]]) -> dict[str, object]:
                    state: dict[str, object] = {}
                    staged: dict[str, object] = {}
                    for entry in entries:
                        kind = entry["kind"]
                        if kind == "begin":
                            staged = {}
                        elif kind == "write":
                            state[entry["key"]] = entry["value"]
                        elif kind == "commit":
                            for key, value in staged.items():
                                state[key] = value
                            staged = {}
                        elif kind == "abort":
                            staged = {}
                    return state
                """,
            ),
            (
                "counts.py",
                """
                def committed_batches(entries: list[dict[str, object]]) -> int:
                    count = 0
                    for entry in entries:
                        if entry["kind"] == "commit":
                            count += 1
                    return len(entries)
                """,
            ),
            (
                "api.py",
                """
                from emberjournal.counts import committed_batches
                from emberjournal.replay import materialize


                def journal_result(entries: list[dict[str, object]]) -> dict[str, object]:
                    return {
                        "committed": committed_batches(entries),
                        "state": materialize(entries),
                    }
                """,
            ),
            (
                "labels.py",
                """
                def batch_label(number: int) -> str:
                    return f"ember-{number}"
                """,
            ),
        ),
        "test_ember_contract.py",
        """
        from emberjournal.api import journal_result


        def test_aborted_write_is_not_materialized() -> None:
            entries = [
                {"kind": "begin"},
                {"kind": "write", "key": "heat", "value": 7},
                {"kind": "abort"},
            ]
            assert journal_result(entries) == {"committed": 0, "state": {}}
        """,
    )


def _cairn_graph_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "cairnroute",
        """
        # Cairn prerequisite routes

        A build target may depend on a graph of earlier stones. Plans are stable,
        duplicate-free, and tolerate a back-edge without repeating a stone.
        """,
        (
            (
                "traverse.py",
                """
                def prerequisites(graph: dict[str, list[str]], target: str) -> list[str]:
                    ordered: list[str] = []
                    seen: list[str] = []

                    def visit(node: str) -> None:
                        if node in seen:
                            return
                        seen.append(node)
                        if node != target:
                            ordered.append(node)
                        for dependency in sorted(graph.get(node, [])):
                            visit(dependency)

                    visit(target)
                    return ordered
                """,
            ),
            (
                "render.py",
                """
                def plan_line(nodes: list[str]) -> str:
                    return " > ".join(nodes)
                """,
            ),
            (
                "edges.py",
                """
                def edge_count(graph: dict[str, list[str]]) -> int:
                    return sum(len(values) for values in graph.values())
                """,
            ),
            (
                "roots.py",
                """
                def named_root(target: str) -> str:
                    return f"root:{target}"
                """,
            ),
        ),
        "test_cairn_plan.py",
        """
        from cairnroute.traverse import prerequisites


        def test_dependencies_precede_their_consumers() -> None:
            graph = {"tower": ["arch"], "arch": ["footing"], "footing": []}
            assert prerequisites(graph, "tower") == ["footing", "arch"]
        """,
    )


def _mosaic_overlay_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "mosaicoverlay",
        """
        # Mosaic sparse overlay

        Overlay entries update a base map. An omitted key retains its base value;
        the current public policy treats an explicit null as an ordinary value.
        """,
        (
            (
                "merge.py",
                """
                def overlay(
                    base: dict[str, object],
                    sparse: dict[str, object],
                ) -> dict[str, object]:
                    merged = dict(base)
                    for key, value in sparse.items():
                        if value is not None:
                            merged[key] = value
                    return merged
                """,
            ),
            (
                "presence.py",
                """
                def supplied(sparse: dict[str, object], key: str) -> bool:
                    return key in sparse
                """,
            ),
            (
                "display.py",
                """
                def sorted_keys(values: dict[str, object]) -> list[str]:
                    return sorted(values)
                """,
            ),
            (
                "defaults.py",
                """
                def empty_base() -> dict[str, object]:
                    return {}
                """,
            ),
        ),
        "test_mosaic_overlay.py",
        """
        from mosaicoverlay.merge import overlay


        def test_explicit_null_is_currently_a_value() -> None:
            assert overlay({"ink": "blue", "size": 4}, {"ink": None}) == {
                "ink": None,
                "size": 4,
            }
        """,
    )


def _glyph_scanner_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "glyphfields",
        """
        # Glyph field scanner

        Fields use a configurable separator and escape glyph. Escaping consumes the
        following glyph, empty fields are retained, and a terminal escape stays literal.
        """,
        (
            (
                "scan.py",
                """
                def fields(text: str, separator: str, escape: str) -> list[str]:
                    return text.split(separator)
                """,
            ),
            (
                "contract.py",
                """
                def valid_markers(separator: str, escape: str) -> bool:
                    return len(separator) == 1 and len(escape) == 1 and separator != escape
                """,
            ),
            (
                "labels.py",
                """
                def field_label(index: int) -> str:
                    return f"glyph-field-{index}"
                """,
            ),
            (
                "joining.py",
                """
                def visible_join(values: list[str]) -> str:
                    return " / ".join(values)
                """,
            ),
        ),
        "test_glyph_fields.py",
        r"""
        from glyphfields.scan import fields


        def test_escaped_separator_and_empty_field() -> None:
            assert fields(r"ash\|birch||cedar", "|", "\\") == ["ash|birch", "", "cedar"]
        """,
    )


def _packet_order_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "packetmeasure",
        """
        # Packet measure

        A legacy header stores a length in two unsigned bytes. No version marker,
        byte-order convention, or external protocol identifier is available.
        """,
        (
            (
                "decode.py",
                """
                def payload_length(first: int, second: int) -> int:
                    return first + second * 256
                """,
            ),
            (
                "bounds.py",
                """
                def valid_byte(value: int) -> bool:
                    return 0 <= value <= 255
                """,
            ),
            (
                "header.py",
                """
                def header_pair(first: int, second: int) -> list[int]:
                    return [first, second]
                """,
            ),
            (
                "display.py",
                """
                def byte_text(first: int, second: int) -> str:
                    return f"{first}:{second}"
                """,
            ),
        ),
        "test_packet_measure.py",
        """
        from packetmeasure.decode import payload_length


        def test_symmetric_header_has_one_value() -> None:
            assert payload_length(3, 3) == 771
        """,
    )


def _press_batch_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "pressbatch",
        """
        # Press publication batches

        Publication records use renamed operations. Their state and accepted-batch count
        follow the earlier Ember mutation-journal convention.
        """,
        (
            (
                "publish.py",
                """
                def published_state(records: list[dict[str, object]]) -> dict[str, object]:
                    published: dict[str, object] = {}
                    pending: dict[str, object] = {}
                    for record in records:
                        operation = record["operation"]
                        if operation == "open":
                            pending = {}
                        elif operation == "revise":
                            published[record["slug"]] = record["body"]
                        elif operation == "release":
                            for slug, body in pending.items():
                                published[slug] = body
                            pending = {}
                        elif operation == "discard":
                            pending = {}
                    return published
                """,
            ),
            (
                "accepted.py",
                """
                def accepted_batches(records: list[dict[str, object]]) -> int:
                    accepted = 0
                    for record in records:
                        if record["operation"] == "release":
                            accepted += 1
                    return len(records)
                """,
            ),
            (
                "result.py",
                """
                from pressbatch.accepted import accepted_batches
                from pressbatch.publish import published_state


                def publication_result(records: list[dict[str, object]]) -> dict[str, object]:
                    return {
                        "accepted": accepted_batches(records),
                        "published": published_state(records),
                    }
                """,
            ),
            (
                "slugs.py",
                """
                def clean_slug(value: str) -> str:
                    return value.strip().lower()
                """,
            ),
        ),
        "test_press_batch.py",
        """
        from pressbatch.result import publication_result


        def test_empty_record_stream_is_empty() -> None:
            assert publication_result([]) == {"accepted": 0, "published": {}}
        """,
        documents=(
            (
                "lineage.md",
                """
                Apply the earlier Ember mutation-journal convention to these renamed
                publication records. Local presentation rules do not redefine it.
                """,
            ),
        ),
    )


def _nested_scalar_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "nestedscalar",
        """
        # Nested scalar paths

        Walk tagged map/list trees recursively. Map keys are visited in sorted order and
        list items in index order. Each path segment records whether it is a key or index.
        """,
        (
            (
                "walk.py",
                """
                def scalar_paths(node: dict[str, object]) -> list[list[object]]:
                    if node["kind"] == "scalar":
                        return [[[], node["value"]]]
                    return []
                """,
            ),
            (
                "segments.py",
                """
                def key_segment(value: str) -> list[object]:
                    return ["key", value]


                def index_segment(value: int) -> list[object]:
                    return ["index", value]
                """,
            ),
            (
                "labels.py",
                """
                def leaf_count(rows: list[list[object]]) -> int:
                    return len(rows)
                """,
            ),
            (
                "empty.py",
                """
                def empty_rows() -> list[list[object]]:
                    return []
                """,
            ),
        ),
        "test_nested_scalar.py",
        """
        from nestedscalar.walk import scalar_paths


        def test_map_and_list_paths_are_typed() -> None:
            node = {
                "kind": "map",
                "children": {
                    "a": {
                        "kind": "list",
                        "children": [{"kind": "scalar", "value": 5}],
                    }
                },
            }
            assert scalar_paths(node) == [[[ ["key", "a"], ["index", 0] ], 5]]
        """,
    )


def _turn_scheduler_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "turnscheduler",
        """
        # Turn scheduler

        The current queue contract dispatches jobs in global submission order.
        Tenant labels and reporting are separate from dispatch policy.
        """,
        (
            (
                "dispatch.py",
                """
                def dispatch(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
                    return list(reversed(jobs))
                """,
            ),
            (
                "tenants.py",
                """
                def tenant_names(jobs: list[dict[str, object]]) -> list[str]:
                    names: list[str] = []
                    for job in jobs:
                        tenant = job["tenant"]
                        if tenant not in names:
                            names.append(tenant)
                    return names
                """,
            ),
            (
                "display.py",
                """
                def job_ids(jobs: list[dict[str, object]]) -> list[str]:
                    return [job["id"] for job in jobs]
                """,
            ),
            (
                "audit.py",
                """
                def audit_size(jobs: list[dict[str, object]]) -> int:
                    return len(jobs)
                """,
            ),
        ),
        "test_turn_scheduler.py",
        """
        from turnscheduler.dispatch import dispatch


        def test_submission_order_is_preserved() -> None:
            jobs = [{"id": "a", "tenant": "red"}, {"id": "b", "tenant": "blue"}]
            assert dispatch(jobs) == jobs
        """,
    )


def _lab_graph_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "labweave",
        """
        # Laboratory protocol weave

        Protocol prerequisites use renamed records. Their ordering follows the earlier
        Cairn prerequisite-route convention; local labels do not restate that convention.
        """,
        (
            (
                "sequence.py",
                """
                def protocol_order(links: dict[str, list[str]], assay: str) -> list[str]:
                    plan: list[str] = []
                    visited: list[str] = []

                    def collect(protocol: str) -> None:
                        if protocol in visited:
                            return
                        visited.append(protocol)
                        if protocol != assay:
                            plan.append(protocol)
                        for prerequisite in sorted(links.get(protocol, [])):
                            collect(prerequisite)

                    collect(assay)
                    return plan
                """,
            ),
            (
                "names.py",
                """
                def protocol_name(value: str) -> str:
                    return value.strip()
                """,
            ),
            (
                "counts.py",
                """
                def link_count(links: dict[str, list[str]]) -> int:
                    return sum(len(values) for values in links.values())
                """,
            ),
            (
                "report.py",
                """
                def report_line(values: list[str]) -> str:
                    return " | ".join(values)
                """,
            ),
        ),
        "test_lab_weave.py",
        """
        from labweave.sequence import protocol_order


        def test_one_leaf_prerequisite_is_listed() -> None:
            assert protocol_order({"assay": ["wash"], "wash": []}, "assay") == ["wash"]
        """,
        documents=(
            (
                "precedent.md",
                """
                Use the earlier Cairn prerequisite-route convention for ordering and
                revisits. Protocol naming and report punctuation are independent.
                """,
            ),
        ),
    )


def _command_fragment_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "commandfragments",
        """
        # Command fragments

        Command text uses a divider and quoting marker whose behavior follows the earlier
        Glyph field-scanner convention under renamed symbols.
        """,
        (
            (
                "fragments.py",
                """
                def fragments(text: str, divider: str, quote: str) -> list[str]:
                    return text.split(divider)
                """,
            ),
            (
                "markers.py",
                """
                def distinct(divider: str, quote: str) -> bool:
                    return divider != quote
                """,
            ),
            (
                "display.py",
                """
                def display(values: list[str]) -> str:
                    return " + ".join(values)
                """,
            ),
            (
                "labels.py",
                """
                def fragment_label(index: int) -> str:
                    return f"fragment-{index}"
                """,
            ),
        ),
        "test_command_fragments.py",
        """
        from commandfragments.fragments import fragments


        def test_plain_divider_splits_text() -> None:
            assert fragments("north:south", ":", "~") == ["north", "south"]
        """,
        documents=(
            (
                "precedent.md",
                """
                Apply the earlier Glyph field-scanner convention to the divider and quoting
                marker. Command display and labels do not define the parsing relation.
                """,
            ),
        ),
    )


def _local_clock_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "localclock",
        """
        # Local clock normalization

        Imported readings contain naive local hours and site names. No timezone database,
        UTC offset, daylight-saving rule, or capture date is supplied.
        """,
        (
            (
                "normalize.py",
                """
                def normalized_hour(site: str, local_hour: int) -> int:
                    return local_hour
                """,
            ),
            (
                "sites.py",
                """
                def site_label(site: str) -> str:
                    return site.strip().upper()
                """,
            ),
            (
                "hours.py",
                """
                def valid_hour(value: int) -> bool:
                    return 0 <= value < 24
                """,
            ),
            (
                "display.py",
                """
                def hour_text(value: int) -> str:
                    return f"{value:02d}"
                """,
            ),
        ),
        "test_local_clock.py",
        """
        from localclock.normalize import normalized_hour


        def test_reference_site_is_unchanged() -> None:
            assert normalized_hour("reference", 11) == 11
        """,
    )


def _sample_tree_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "samplecanopy",
        """
        # Sample canopy measurements

        Tagged specimen trees use the earlier Nested scalar-path convention. Local sample
        names do not redefine traversal order, leaf selection, or path-segment typing.
        """,
        (
            (
                "measure.py",
                """
                def measurements(specimen: dict[str, object]) -> list[list[object]]:
                    if specimen["type"] == "reading":
                        return [[[], specimen["reading"]]]
                    return []
                """,
            ),
            (
                "segments.py",
                """
                def branch_segment(name: str) -> list[object]:
                    return ["key", name]


                def sample_segment(index: int) -> list[object]:
                    return ["index", index]
                """,
            ),
            (
                "summary.py",
                """
                def measurement_count(values: list[list[object]]) -> int:
                    return len(values)
                """,
            ),
            (
                "labels.py",
                """
                def specimen_label(name: str) -> str:
                    return f"specimen:{name}"
                """,
            ),
        ),
        "test_sample_canopy.py",
        """
        from samplecanopy.measure import measurements


        def test_top_level_reading_has_empty_path() -> None:
            specimen = {"type": "reading", "reading": 8}
            assert measurements(specimen) == [[[], 8]]
        """,
        documents=(
            (
                "precedent.md",
                """
                Apply the earlier Nested scalar-path convention to these renamed specimen
                nodes. Summary counts and labels are not traversal evidence.
                """,
            ),
        ),
    )


_SPECS = (
    _TaskSpec(
        "train-v10-ember-journal-31",
        (
            "Journal writes are provisional until their batch is committed. An abort discards "
            "that batch, and the result must report the number of committed batches rather "
            "than the number of input records."
        ),
        _JOURNAL,
        _DIRECT,
        _ember_journal_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "ember-hidden-commit",
                {
                    "entries": [
                        {"kind": "begin"},
                        {"kind": "write", "key": "heat", "value": 4},
                        {"kind": "commit"},
                    ]
                },
                {"committed": 1, "state": {"heat": 4}},
            ),
            _case(
                "ember-hidden-abort",
                {
                    "entries": [
                        {"kind": "begin"},
                        {"kind": "write", "key": "tone", "value": "red"},
                        {"kind": "abort"},
                    ]
                },
                {"committed": 0, "state": {}},
            ),
            _case(
                "ember-hidden-two-batches",
                {
                    "entries": [
                        {"kind": "begin"},
                        {"kind": "write", "key": "a", "value": 1},
                        {"kind": "commit"},
                        {"kind": "begin"},
                        {"kind": "write", "key": "b", "value": 2},
                        {"kind": "commit"},
                    ]
                },
                {"committed": 2, "state": {"a": 1, "b": 2}},
            ),
        ),
        "emberjournal.api",
        "journal_result",
        transfer_relation_sha256=_RELATION_JOURNAL,
    ),
    _TaskSpec(
        "train-v10-cairn-prerequisites-74",
        (
            "Produce a deterministic prerequisite plan: recursively visit sorted dependencies "
            "before each consumer, omit the requested target, emit each prerequisite once, "
            "and tolerate cycles."
        ),
        _GRAPH,
        _DIRECT,
        _cairn_graph_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "cairn-hidden-chain",
                {
                    "graph": {"spire": ["arch"], "arch": ["base"], "base": []},
                    "target": "spire",
                },
                ["base", "arch"],
            ),
            _case(
                "cairn-hidden-diamond",
                {
                    "graph": {
                        "summit": ["west", "east"],
                        "east": ["root"],
                        "west": ["root"],
                        "root": [],
                    },
                    "target": "summit",
                },
                ["root", "east", "west"],
            ),
            _case(
                "cairn-hidden-cycle",
                {"graph": {"top": ["ring"], "ring": ["top"]}, "target": "top"},
                ["ring"],
            ),
        ),
        "cairnroute.traverse",
        "prerequisites",
        transfer_relation_sha256=_RELATION_GRAPH,
    ),
    _TaskSpec(
        "train-v10-mosaic-null-58",
        (
            "A sparse overlay changes only supplied keys. Omitted keys retain their base "
            "values, and under the current published policy an explicit null is stored as "
            "an ordinary value."
        ),
        _OVERLAY,
        _REVISION,
        _mosaic_overlay_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "mosaic-initial-null",
                {"base": {"ink": "blue", "size": 4}, "sparse": {"ink": None}},
                {"ink": None, "size": 4},
            ),
            _case(
                "mosaic-initial-omitted",
                {"base": {"ink": "blue", "size": 4}, "sparse": {"size": 9}},
                {"ink": "blue", "size": 9},
            ),
            _case(
                "mosaic-initial-empty",
                {"base": {"ink": "blue"}, "sparse": {}},
                {"ink": "blue"},
            ),
        ),
        "mosaicoverlay.merge",
        "overlay",
        review_text=(
            "A superseding public policy now defines explicit null as a tombstone: remove "
            "that key when present. Omission still retains the base value."
        ),
        revised_witnesses=(
            _case(
                "mosaic-revised-null",
                {"base": {"ink": "blue", "size": 4}, "sparse": {"ink": None}},
                {"size": 4},
            ),
            _case(
                "mosaic-revised-omitted",
                {"base": {"ink": "blue", "size": 4}, "sparse": {"size": 9}},
                {"ink": "blue", "size": 9},
            ),
            _case(
                "mosaic-revised-absent-delete",
                {"base": {"ink": "blue"}, "sparse": {"missing": None}},
                {"ink": "blue"},
            ),
        ),
    ),
    _TaskSpec(
        "train-v10-glyph-escape-46",
        (
            "Implement the documented escape-aware field scan for configurable one-character "
            "markers, including empty fields and a terminal escape glyph."
        ),
        _SCANNER,
        _DIRECT,
        _glyph_scanner_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "glyph-hidden-escaped",
                {"text": "red\\|blue|green", "separator": "|", "escape": "\\"},
                ["red|blue", "green"],
            ),
            _case(
                "glyph-hidden-empty",
                {"text": "|a||", "separator": "|", "escape": "\\"},
                ["", "a", "", ""],
            ),
            _case(
                "glyph-hidden-terminal",
                {"text": "tail\\", "separator": "|", "escape": "\\"},
                ["tail\\"],
            ),
        ),
        "glyphfields.scan",
        "fields",
        transfer_relation_sha256=_RELATION_SCANNER,
    ),
    _TaskSpec(
        "train-v10-packet-byte-order-19",
        (
            "Decode the two-byte payload length for this legacy packet family. The available "
            "repository does not identify the byte order or a protocol version."
        ),
        _ENDIAN,
        _INSUFFICIENT,
        _packet_order_repository(),
        STOP,
        (),
        None,
        None,
        ambiguity=(
            "two-byte-orderings-remain-compatible-v10",
            _canonical_json({"first": 1, "second": 2}).decode("ascii"),
            _canonical_json(513).decode("ascii"),
            _canonical_json(258).decode("ascii"),
        ),
    ),
    _TaskSpec(
        "train-v10-press-publication-83",
        (
            "Apply the earlier Ember mutation-journal behavior to these renamed publication "
            "records and their accepted-batch count. Preserve local slug presentation."
        ),
        _JOURNAL,
        _TRANSFER,
        _press_batch_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "press-hidden-release",
                {
                    "records": [
                        {"operation": "open"},
                        {"operation": "revise", "slug": "news", "body": "ready"},
                        {"operation": "release"},
                    ]
                },
                {"accepted": 1, "published": {"news": "ready"}},
            ),
            _case(
                "press-hidden-discard",
                {
                    "records": [
                        {"operation": "open"},
                        {"operation": "revise", "slug": "draft", "body": "x"},
                        {"operation": "discard"},
                    ]
                },
                {"accepted": 0, "published": {}},
            ),
            _case(
                "press-hidden-mixed",
                {
                    "records": [
                        {"operation": "open"},
                        {"operation": "revise", "slug": "a", "body": 1},
                        {"operation": "release"},
                        {"operation": "open"},
                        {"operation": "revise", "slug": "b", "body": 2},
                        {"operation": "discard"},
                    ]
                },
                {"accepted": 1, "published": {"a": 1}},
            ),
        ),
        "pressbatch.result",
        "publication_result",
        retention_anchor_ordinal=0,
        transfer_relation_sha256=_RELATION_JOURNAL,
    ),
    _TaskSpec(
        "train-v10-nested-scalars-65",
        (
            "Recursively enumerate every scalar in the tagged map/list tree. Sort map keys, "
            "preserve list index order, and represent each path segment as [key, name] or "
            "[index, number]."
        ),
        _TREE,
        _DIRECT,
        _nested_scalar_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "nested-hidden-map",
                {
                    "node": {
                        "kind": "map",
                        "children": {
                            "b": {"kind": "scalar", "value": 2},
                            "a": {"kind": "scalar", "value": 1},
                        },
                    }
                },
                [
                    [[["key", "a"]], 1],
                    [[["key", "b"]], 2],
                ],
            ),
            _case(
                "nested-hidden-list",
                {
                    "node": {
                        "kind": "list",
                        "children": [
                            {"kind": "scalar", "value": "x"},
                            {"kind": "scalar", "value": None},
                        ],
                    }
                },
                [
                    [[["index", 0]], "x"],
                    [[["index", 1]], None],
                ],
            ),
            _case(
                "nested-hidden-mixed",
                {
                    "node": {
                        "kind": "map",
                        "children": {
                            "samples": {
                                "kind": "list",
                                "children": [{"kind": "scalar", "value": 7}],
                            }
                        },
                    }
                },
                [[[["key", "samples"], ["index", 0]], 7]],
            ),
        ),
        "nestedscalar.walk",
        "scalar_paths",
        transfer_relation_sha256=_RELATION_TREE,
    ),
    _TaskSpec(
        "train-v10-turn-scheduler-27",
        (
            "The published scheduler currently dispatches every job in global submission "
            "order. Align the implementation without changing tenant or display helpers."
        ),
        _SCHEDULER,
        _REVISION,
        _turn_scheduler_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "turn-initial-three",
                {
                    "jobs": [
                        {"id": "a", "tenant": "red"},
                        {"id": "b", "tenant": "red"},
                        {"id": "c", "tenant": "blue"},
                    ]
                },
                [
                    {"id": "a", "tenant": "red"},
                    {"id": "b", "tenant": "red"},
                    {"id": "c", "tenant": "blue"},
                ],
            ),
            _case("turn-initial-empty", {"jobs": []}, []),
            _case(
                "turn-initial-one",
                {"jobs": [{"id": "q", "tenant": "green"}]},
                [{"id": "q", "tenant": "green"}],
            ),
        ),
        "turnscheduler.dispatch",
        "dispatch",
        review_text=(
            "A superseding public fairness policy requires tenant round-robin dispatch. "
            "Preserve each tenant's local submission order and the order tenants first appear."
        ),
        revised_witnesses=(
            _case(
                "turn-revised-three",
                {
                    "jobs": [
                        {"id": "a", "tenant": "red"},
                        {"id": "b", "tenant": "red"},
                        {"id": "c", "tenant": "blue"},
                    ]
                },
                [
                    {"id": "a", "tenant": "red"},
                    {"id": "c", "tenant": "blue"},
                    {"id": "b", "tenant": "red"},
                ],
            ),
            _case(
                "turn-revised-four",
                {
                    "jobs": [
                        {"id": "a1", "tenant": "amber"},
                        {"id": "a2", "tenant": "amber"},
                        {"id": "b1", "tenant": "blue"},
                        {"id": "b2", "tenant": "blue"},
                    ]
                },
                [
                    {"id": "a1", "tenant": "amber"},
                    {"id": "b1", "tenant": "blue"},
                    {"id": "a2", "tenant": "amber"},
                    {"id": "b2", "tenant": "blue"},
                ],
            ),
            _case("turn-revised-empty", {"jobs": []}, []),
        ),
    ),
    _TaskSpec(
        "train-v10-lab-protocols-92",
        (
            "Apply the earlier Cairn prerequisite-route convention to these renamed laboratory "
            "protocol links. Preserve local naming and report formatting."
        ),
        _GRAPH,
        _TRANSFER,
        _lab_graph_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "lab-hidden-chain",
                {
                    "links": {"assay": ["stain"], "stain": ["wash"], "wash": []},
                    "assay": "assay",
                },
                ["wash", "stain"],
            ),
            _case(
                "lab-hidden-shared",
                {
                    "links": {
                        "report": ["beta", "alpha"],
                        "alpha": ["prep"],
                        "beta": ["prep"],
                        "prep": [],
                    },
                    "assay": "report",
                },
                ["prep", "alpha", "beta"],
            ),
            _case(
                "lab-hidden-cycle",
                {"links": {"final": ["loop"], "loop": ["final"]}, "assay": "final"},
                ["loop"],
            ),
        ),
        "labweave.sequence",
        "protocol_order",
        retention_anchor_ordinal=1,
        transfer_relation_sha256=_RELATION_GRAPH,
    ),
    _TaskSpec(
        "train-v10-command-fragments-39",
        (
            "Apply the earlier Glyph field-scanner convention to this command divider and "
            "quoting marker. Preserve local fragment labels and display formatting."
        ),
        _SCANNER,
        _TRANSFER,
        _command_fragment_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "command-hidden-quoted",
                {"text": "red~:blue:green", "divider": ":", "quote": "~"},
                ["red:blue", "green"],
            ),
            _case(
                "command-hidden-empty",
                {"text": ":a::", "divider": ":", "quote": "~"},
                ["", "a", "", ""],
            ),
            _case(
                "command-hidden-terminal",
                {"text": "tail~", "divider": ":", "quote": "~"},
                ["tail~"],
            ),
        ),
        "commandfragments.fragments",
        "fragments",
        retention_anchor_ordinal=3,
        transfer_relation_sha256=_RELATION_SCANNER,
    ),
    _TaskSpec(
        "train-v10-local-timezone-51",
        (
            "Normalize this site's naive local hour for cross-site comparison. No timezone, "
            "offset, date, or daylight-saving convention is available."
        ),
        _TIMEZONE,
        _INSUFFICIENT,
        _local_clock_repository(),
        STOP,
        (),
        None,
        None,
        ambiguity=(
            "naive-hour-offset-is-unpublished-v10",
            _canonical_json({"site": "harbor", "local_hour": 9}).decode("ascii"),
            _canonical_json(9).decode("ascii"),
            _canonical_json(8).decode("ascii"),
        ),
    ),
    _TaskSpec(
        "train-v10-sample-canopy-88",
        (
            "Apply the earlier Nested scalar-path convention to these renamed specimen trees. "
            "Preserve local measurement summaries and specimen labels."
        ),
        _TREE,
        _TRANSFER,
        _sample_tree_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "sample-hidden-group",
                {
                    "specimen": {
                        "type": "group",
                        "branches": {
                            "z": {"type": "reading", "reading": 9},
                            "a": {"type": "reading", "reading": 2},
                        },
                    }
                },
                [
                    [[["key", "a"]], 2],
                    [[["key", "z"]], 9],
                ],
            ),
            _case(
                "sample-hidden-series",
                {
                    "specimen": {
                        "type": "series",
                        "samples": [
                            {"type": "reading", "reading": "dry"},
                            {"type": "reading", "reading": 4},
                        ],
                    }
                },
                [
                    [[["index", 0]], "dry"],
                    [[["index", 1]], 4],
                ],
            ),
            _case(
                "sample-hidden-mixed",
                {
                    "specimen": {
                        "type": "group",
                        "branches": {
                            "tray": {
                                "type": "series",
                                "samples": [{"type": "reading", "reading": 6}],
                            }
                        },
                    }
                },
                [[[["key", "tray"], ["index", 0]], 6]],
            ),
        ),
        "samplecanopy.measure",
        "measurements",
        retention_anchor_ordinal=6,
        transfer_relation_sha256=_RELATION_TREE,
    ),
)


def _build_repository(files: tuple[tuple[str, str], ...]) -> RepositorySnapshotV10:
    repository_files = tuple(
        RepositoryFileV10(
            relative_path=path,
            content_utf8=content.encode("utf-8"),
            file_sha256=sha256(content.encode("utf-8")).hexdigest(),
        )
        for path, content in files
    )
    provisional = RepositorySnapshotV10(repository_files, "")
    return RepositorySnapshotV10(
        repository_files,
        _digest(b"continual-repository-snapshot-v10", provisional.record()),
    )


def _build_witnesses(
    records: tuple[tuple[str, str, str], ...],
) -> tuple[FrozenWitnessV10, ...]:
    witnesses = []
    for case_id, request_json, expected_json in records:
        request = _frozen_json(request_json)
        expected = _frozen_json(expected_json)
        provisional = FrozenWitnessV10(case_id, request, expected, "")
        witnesses.append(
            FrozenWitnessV10(
                case_id,
                request,
                expected,
                _digest(b"continual-repository-swe-witness-v10", provisional.record()),
            )
        )
    return tuple(witnesses)


def _build_review_plan(spec: _TaskSpec) -> ReviewPlanV10 | None:
    if spec.review_text is None:
        return None
    revised = _build_witnesses(spec.revised_witnesses)
    provisional = ReviewPlanV10(spec.review_text, revised, "")
    return ReviewPlanV10(
        spec.review_text,
        revised,
        _digest(b"continual-repository-swe-review-plan-v10", provisional.record()),
    )


def _build_ambiguity(spec: _TaskSpec) -> AmbiguityCertificateV10 | None:
    if spec.ambiguity is None:
        return None
    fingerprint, request_json, first_json, second_json = spec.ambiguity
    request = _frozen_json(request_json)
    outcomes = (_frozen_json(first_json), _frozen_json(second_json))
    provisional = AmbiguityCertificateV10(fingerprint, request, outcomes, "")
    return AmbiguityCertificateV10(
        fingerprint,
        request,
        outcomes,
        _digest(b"continual-repository-swe-ambiguity-v10", provisional.record()),
    )


def _build_evaluator(
    spec: _TaskSpec,
    witnesses: tuple[FrozenWitnessV10, ...],
    review_plan: ReviewPlanV10 | None,
) -> FrozenEvaluatorV10 | None:
    if spec.entry_module is None or spec.entry_point is None:
        if spec.entry_module is not None or spec.entry_point is not None or witnesses:
            raise ContinualToySweTaskV10Error("STOP task gained an evaluator")
        return None
    revised = None if review_plan is None else review_plan.revised_witnesses
    provisional = FrozenEvaluatorV10(
        spec.entry_module,
        spec.entry_point,
        witnesses,
        revised,
        "",
    )
    return FrozenEvaluatorV10(
        spec.entry_module,
        spec.entry_point,
        witnesses,
        revised,
        _digest(b"continual-repository-swe-evaluator-v10", provisional.record()),
    )


def _build_task(ordinal: int, spec: _TaskSpec) -> _EvaluatorTaskV10:
    repository = _build_repository(spec.files)
    witnesses = _build_witnesses(spec.witnesses)
    review_plan = _build_review_plan(spec)
    ambiguity = _build_ambiguity(spec)
    evaluator = _build_evaluator(spec, witnesses, review_plan)
    provisional_private = PrivateAcceptanceV10(
        spec.family_id,
        spec.stage,
        spec.expected_action,
        spec.retention_anchor_ordinal,
        spec.transfer_relation_sha256,
        witnesses,
        evaluator,
        review_plan,
        ambiguity,
        "",
    )
    private = PrivateAcceptanceV10(
        spec.family_id,
        spec.stage,
        spec.expected_action,
        spec.retention_anchor_ordinal,
        spec.transfer_relation_sha256,
        witnesses,
        evaluator,
        review_plan,
        ambiguity,
        _digest(b"continual-repository-swe-private-v10", provisional_private.record()),
    )
    commitment = _digest(
        b"continual-repository-swe-task-v10",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "instance_id": spec.instance_id,
            "issue": spec.issue,
            "repository_snapshot_sha256": repository.snapshot_sha256,
            "tool_contract_sha256": TOOL_CONTRACT_SHA256,
            "private_bundle_sha256": private.bundle_sha256,
        },
    )
    provisional_public = LearnerVisibleToySweObservationV10(
        commitment,
        spec.instance_id,
        spec.issue,
        repository.snapshot_sha256,
        TOOL_CONTRACT_SHA256,
        "",
    )
    public = LearnerVisibleToySweObservationV10(
        commitment,
        spec.instance_id,
        spec.issue,
        repository.snapshot_sha256,
        TOOL_CONTRACT_SHA256,
        _digest(b"continual-repository-swe-observation-v10", provisional_public.record()),
    )
    return _EvaluatorTaskV10(ordinal, public, repository, private)


_EVALUATOR_STREAM = tuple(_build_task(ordinal, spec) for ordinal, spec in enumerate(_SPECS))
_PUBLIC_STREAM = tuple(task.public for task in _EVALUATOR_STREAM)

_EXPECTED_STAGES = (
    _DIRECT,
    _DIRECT,
    _REVISION,
    _DIRECT,
    _INSUFFICIENT,
    _TRANSFER,
    _DIRECT,
    _REVISION,
    _TRANSFER,
    _TRANSFER,
    _INSUFFICIENT,
    _TRANSFER,
)
_EXPECTED_FAMILIES = (
    _JOURNAL,
    _GRAPH,
    _OVERLAY,
    _SCANNER,
    _ENDIAN,
    _JOURNAL,
    _TREE,
    _SCHEDULER,
    _GRAPH,
    _SCANNER,
    _TIMEZONE,
    _TREE,
)


def _authenticate_evaluator_stream(stream: tuple[_EvaluatorTaskV10, ...]) -> None:
    if type(stream) is not tuple or len(stream) != TASK_COUNT:
        raise ContinualToySweTaskV10Error("evaluator stream size changed")
    if tuple(task.ordinal for task in stream) != tuple(range(TASK_COUNT)):
        raise ContinualToySweTaskV10Error("task order changed")
    if tuple(task.private.stage for task in stream) != _EXPECTED_STAGES:
        raise ContinualToySweTaskV10Error("stage progression changed")
    if tuple(task.private.family_id for task in stream) != _EXPECTED_FAMILIES:
        raise ContinualToySweTaskV10Error("family lineage changed")
    if Counter(task.private.stage for task in stream) != Counter(
        {_DIRECT: 4, _TRANSFER: 4, _REVISION: 2, _INSUFFICIENT: 2}
    ):
        raise ContinualToySweTaskV10Error("task-kind balance changed")
    anchors = tuple(
        (task.ordinal, task.private.retention_anchor_ordinal)
        for task in stream
        if task.private.stage == _TRANSFER
    )
    if anchors != ((5, 0), (8, 1), (9, 3), (11, 6)):
        raise ContinualToySweTaskV10Error("history-transfer anchors changed")
    for ordinal, anchor in anchors:
        if (
            stream[ordinal].private.transfer_relation_sha256
            != stream[anchor].private.transfer_relation_sha256
        ):
            raise ContinualToySweTaskV10Error("history-transfer relation changed")
    if tuple(task.ordinal for task in stream if task.private.review_plan is not None) != (2, 7):
        raise ContinualToySweTaskV10Error("revision placement changed")
    if tuple(task.ordinal for task in stream if task.private.expected_action == STOP) != (4, 10):
        raise ContinualToySweTaskV10Error("ambiguity placement changed")
    instance_ids = tuple(task.public.instance_id for task in stream)
    commitments = tuple(task.public.task_commitment_sha256 for task in stream)
    snapshots = tuple(task.repository.snapshot_sha256 for task in stream)
    if (
        len(set(instance_ids)) != TASK_COUNT
        or len(set(commitments)) != TASK_COUNT
        or len(set(snapshots)) != TASK_COUNT
    ):
        raise ContinualToySweTaskV10Error("tasks are not unique")
    for task in stream:
        task.authenticate()
        paths = tuple(file.relative_path for file in task.repository.files)
        tests = tuple(path for path in paths if path.startswith("tests/") and path.endswith(".py"))
        writable = tuple(
            path
            for path in paths
            if path.endswith(".py") and "/" in path and not path.startswith("tests/")
        )
        if (
            not 7 <= len(paths) <= 11
            or "README.md" not in paths
            or "pyproject.toml" not in paths
            or len(tests) != 1
            or len(writable) < 4
        ):
            raise ContinualToySweTaskV10Error("repository evidence shape changed")
        if task.private.stage == _TRANSFER and not any(path.startswith("docs/") for path in paths):
            raise ContinualToySweTaskV10Error("history transfer lost its public precedent cue")


def _authenticate_public_stream(
    stream: tuple[LearnerVisibleToySweObservationV10, ...],
) -> None:
    if type(stream) is not tuple or len(stream) != TASK_COUNT:
        raise ContinualToySweTaskV10Error("public stream size changed")
    for public, private_task in zip(stream, _EVALUATOR_STREAM, strict=True):
        if type(public) is not LearnerVisibleToySweObservationV10:
            raise TypeError("public stream contains a private task type")
        public.authenticate()
        if public != private_task.public:
            raise ContinualToySweTaskV10Error("public stream order changed")


def _stream_root(stream: tuple[LearnerVisibleToySweObservationV10, ...]) -> str:
    _authenticate_public_stream(stream)
    return _digest(
        b"continual-repository-swe-stream-v10",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_commitment_sha256s": tuple(task.task_commitment_sha256 for task in stream),
        },
    )


_authenticate_evaluator_stream(_EVALUATOR_STREAM)
_authenticate_public_stream(_PUBLIC_STREAM)
TRAIN_STREAM_SHA256 = _stream_root(_PUBLIC_STREAM)
TASK_SOURCE_SHA256 = sha256(Path(__file__).read_bytes()).hexdigest()


def _custody_record() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "partition": PARTITION,
        "task_count": TASK_COUNT,
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "task_source_sha256": TASK_SOURCE_SHA256,
        "source_prior_corpus_sha256": SOURCE_PRIOR_CORPUS_SHA256,
        "source_prior_case_count": SOURCE_PRIOR_CASE_COUNT,
        "tool_contract_sha256": TOOL_CONTRACT_SHA256,
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }


TRAIN_STREAM_CUSTODY_SHA256 = _digest(b"continual-repository-swe-custody-v10", _custody_record())


def _computed_manifest_record() -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "partition": PARTITION,
        "task_count": TASK_COUNT,
        "raw_source_sha256": TASK_SOURCE_SHA256,
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "train_stream_custody_sha256": TRAIN_STREAM_CUSTODY_SHA256,
        "tool_contract_sha256": TOOL_CONTRACT_SHA256,
        "source_prior_corpus_sha256": SOURCE_PRIOR_CORPUS_SHA256,
        "source_prior_case_count": SOURCE_PRIOR_CASE_COUNT,
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }


def _authenticate_external_manifest() -> None:
    path = Path(__file__).with_name(MANIFEST_FILENAME)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContinualToySweTaskV10Error("external custody manifest is unavailable") from error
    if manifest != _computed_manifest_record():
        raise ContinualToySweTaskV10Error("external custody manifest does not pin this source")


def _authenticate_execution_substrate() -> tuple[bytes, Path]:
    try:
        worker_source = _REPOSITORY_WORKER_PATH.read_bytes()
        sandbox_source = _SANDBOX_SOURCE_PATH.read_bytes()
    except OSError as error:
        raise ContinualToySweTaskV10Error("frozen evaluator substrate is unavailable") from error
    if sha256(worker_source).hexdigest() != REPOSITORY_WORKER_SHA256:
        raise ContinualToySweTaskV10Error("repository worker differs from its freeze")
    if sha256(sandbox_source).hexdigest() != SIMPLE_SWE_SANDBOX_SHA256:
        raise ContinualToySweTaskV10Error("sandbox launcher differs from its freeze")
    executable = resolve_python_executable()
    if sha256(executable.read_bytes()).hexdigest() != PYTHON_EXECUTABLE_SHA256:
        raise ContinualToySweTaskV10Error("sandbox Python executable differs from its freeze")
    return worker_source, executable


def preflight_execution_substrate_v10() -> dict[str, str]:
    """Authenticate worker, launcher, and interpreter before cohort reservation."""

    _authenticate_external_manifest()
    _authenticate_execution_substrate()
    return {
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }


def public_train_tasks_v10() -> tuple[LearnerVisibleToySweObservationV10, ...]:
    """Return the authenticated evaluator-free public TRAIN stream."""

    _authenticate_external_manifest()
    _authenticate_public_stream(_PUBLIC_STREAM)
    return tuple(_PUBLIC_STREAM)


def global_tool_contract_v10() -> dict[str, object]:
    """Return a detached copy of the one uniform action contract."""

    return json.loads(_canonical_json(_tool_contract_record()).decode("ascii"))


def _evaluator_task(task_commitment_sha256: str) -> _EvaluatorTaskV10:
    matches = tuple(
        task
        for task in _EVALUATOR_STREAM
        if task.public.task_commitment_sha256 == task_commitment_sha256
    )
    if len(matches) != 1:
        raise ContinualToySweTaskV10Error("unknown task commitment")
    task = matches[0]
    task.authenticate()
    return task


def runner_mount_allowlist_v10(task_commitment_sha256: str) -> tuple[str, ...]:
    """Return every learner-readable public repository path."""

    task = _evaluator_task(task_commitment_sha256)
    return tuple(file.relative_path for file in task.repository.files)


def runner_writable_source_allowlist_v10(task_commitment_sha256: str) -> tuple[str, ...]:
    """Return existing package sources writable through ``submit_patch``."""

    readable = runner_mount_allowlist_v10(task_commitment_sha256)
    writable = tuple(
        path
        for path in readable
        if path.endswith(".py") and "/" in path and not path.startswith("tests/")
    )
    if not writable:
        raise ContinualToySweTaskV10Error("repository has no writable package source")
    return writable


def learner_workspace_files_v10(task_commitment_sha256: str) -> dict[str, bytes]:
    """Materialize detached learner-readable files and no evaluator material."""

    task = _evaluator_task(task_commitment_sha256)
    allowlist = runner_mount_allowlist_v10(task_commitment_sha256)
    workspace = {file.relative_path: bytes(file.content_utf8) for file in task.repository.files}
    if tuple(workspace) != allowlist:
        raise ContinualToySweTaskV10Error("runner mount exceeded its allowlist")
    return workspace


def _validated_patch(task_commitment_sha256: str, payload: dict[str, str]) -> tuple[str, str]:
    if type(payload) is not dict or set(payload) != {"patch"}:
        raise ContinualToySweTaskV10Error(
            'submit_patch payload must be exactly {"patch": <unified diff>}'
        )
    patch = payload["patch"]
    if type(patch) is not str:
        raise ContinualToySweTaskV10Error("submit_patch.patch must be a string")
    if not patch or "\r" in patch or not patch.endswith("\n"):
        raise CandidatePatchRejectedV10("patch must be a canonical LF-terminated string")
    try:
        encoded = patch.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CandidatePatchRejectedV10("patch is not valid UTF-8") from error
    if len(encoded) > _EXECUTION_LIMITS.max_completion_bytes:
        raise CandidatePatchRejectedV10("patch exceeds the global byte budget")

    writable = set(runner_writable_source_allowlist_v10(task_commitment_sha256))
    lines = patch.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts or starts[0] != 0:
        raise CandidatePatchRejectedV10("patch is not a unified Git diff")
    touched: list[str] = []
    forbidden_headers = (
        "new file mode ",
        "deleted file mode ",
        "old mode ",
        "new mode ",
        "similarity index ",
        "dissimilarity index ",
        "rename from ",
        "rename to ",
        "copy from ",
        "copy to ",
        "Binary files ",
        "GIT binary patch",
    )
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]
        header = block[0].split(" ")
        if len(header) != 4 or header[:2] != ["diff", "--git"]:
            raise CandidatePatchRejectedV10("patch path header is not canonical")
        old_name, new_name = header[2], header[3]
        if not old_name.startswith("a/") or not new_name.startswith("b/"):
            raise CandidatePatchRejectedV10("patch paths must use a/ and b/ prefixes")
        path = old_name[2:]
        if path != new_name[2:] or path not in writable or path in touched:
            raise CandidatePatchRejectedV10("patch path is outside the writable allowlist")
        canonical = PurePosixPath(path)
        if canonical.is_absolute() or ".." in canonical.parts or canonical.as_posix() != path:
            raise CandidatePatchRejectedV10("patch path is not canonical")
        if any(line.startswith(forbidden_headers) for line in block):
            raise CandidatePatchRejectedV10("patch changes repository shape or file mode")
        if block.count(f"--- a/{path}") != 1 or block.count(f"+++ b/{path}") != 1:
            raise CandidatePatchRejectedV10("patch file headers changed")
        if not any(line.startswith("@@ ") for line in block):
            raise CandidatePatchRejectedV10("patch has no textual hunk")
        touched.append(path)
    return patch, sha256(encoded).hexdigest()


def _module_name(relative_path: str) -> str:
    parts = PurePosixPath(relative_path).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _run_private_evaluator(
    task: _EvaluatorTaskV10,
    patch: str,
    patch_sha256: str,
    *,
    revised: bool,
) -> bool:
    workspace = learner_workspace_files_v10(task.public.task_commitment_sha256)
    with tempfile.TemporaryDirectory(prefix="continual-swe-v10-") as root_name:
        root = Path(root_name)
        for relative_path, content in workspace.items():
            target = root / PurePosixPath(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content.decode("utf-8"), encoding="utf-8")

        apply_command = (
            "git",
            "-c",
            "core.autocrlf=false",
            "apply",
            "--whitespace=error-all",
            "-",
        )
        checked = subprocess.run(
            (*apply_command[:4], "--check", *apply_command[4:]),
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if checked.returncode != 0:
            detail = checked.stderr.strip() or "git apply rejected the diff"
            raise CandidatePatchRejectedV10(f"submitted patch does not apply cleanly: {detail}")
        applied = subprocess.run(
            apply_command,
            cwd=root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if applied.returncode != 0:
            raise CandidatePatchRejectedV10("submitted patch could not be applied")

        evaluator = task.private.evaluator
        if evaluator is None:
            raise ContinualToySweTaskV10Error("task has no executable evaluator")
        cases = evaluator.revised_cases if revised else evaluator.initial_cases
        if cases is None:
            raise ContinualToySweTaskV10Error("revised evaluator cases are unavailable")
        writable = runner_writable_source_allowlist_v10(task.public.task_commitment_sha256)
        modules = [
            {
                "name": _module_name(path),
                "source": (root / PurePosixPath(path)).read_text(encoding="utf-8"),
            }
            for path in writable
        ]
        phase_sha256 = evaluator.phase_sha256(revised=revised)
        payload = {
            "schema": _REPOSITORY_WORKER_SCHEMA,
            "task_commitment_sha256": task.public.task_commitment_sha256,
            "action_sha256": patch_sha256,
            "evaluator_sha256": phase_sha256,
            "modules": modules,
            "entry_module": evaluator.entry_module,
            "entry_point": evaluator.entry_point,
            "cases": [
                {
                    "request": json.loads(case.request_json),
                    "expected": json.loads(case.expected_json),
                }
                for case in cases
            ],
            "expected_case_count": len(cases),
            "max_ast_nodes": _EXECUTION_LIMITS.max_ast_nodes,
            "limits": {
                "cpu_seconds": _EXECUTION_LIMITS.cpu_seconds,
                "memory_bytes": _EXECUTION_LIMITS.memory_mib * 1024 * 1024,
                "output_bytes": _EXECUTION_LIMITS.output_bytes,
                "processes": _EXECUTION_LIMITS.processes,
            },
        }
        worker_source, executable = _authenticate_execution_substrate()
        execution = execute_frozen_worker(
            _canonical_json(payload),
            _EXECUTION_LIMITS,
            python_executable=executable,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            worker_source=worker_source,
            worker_sha256=REPOSITORY_WORKER_SHA256,
        )
        if not execution.sandboxed:
            raise ContinualToySweTaskV10Error(
                execution.detail or "frozen sandbox execution was unavailable"
            )
        if execution.detail or execution.stderr or execution.returncode != 0:
            return False
        try:
            result = json.loads(execution.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(result, dict) or set(result) != {
            "schema",
            "task_commitment_sha256",
            "action_sha256",
            "evaluator_sha256",
            "outcome",
            "completed_case_count",
            "expected_case_count",
            "detail",
        }:
            return False
        return (
            result["schema"] == _REPOSITORY_RESULT_SCHEMA
            and result["task_commitment_sha256"] == task.public.task_commitment_sha256
            and result["action_sha256"] == patch_sha256
            and result["evaluator_sha256"] == phase_sha256
            and result["outcome"] == "passed"
            and result["completed_case_count"] == len(cases)
            and result["expected_case_count"] == len(cases)
            and result["detail"] == ""
        )


def _make_receipt(
    *,
    task_commitment_sha256: str,
    sequence: int,
    prior_receipt_sha256: str,
    action_sha256: str,
    evaluator_sha256: str,
    sandbox_evaluator_reached: bool,
    conclusion: str,
    feedback: str,
) -> ActionReceiptV10:
    provisional = ActionReceiptV10(
        task_commitment_sha256,
        sequence,
        prior_receipt_sha256,
        SUBMIT_PATCH,
        action_sha256,
        evaluator_sha256,
        sandbox_evaluator_reached,
        conclusion,
        feedback,
        "",
    )
    receipt = ActionReceiptV10(
        task_commitment_sha256,
        sequence,
        prior_receipt_sha256,
        SUBMIT_PATCH,
        action_sha256,
        evaluator_sha256,
        sandbox_evaluator_reached,
        conclusion,
        feedback,
        _digest(b"continual-repository-swe-action-receipt-v10", provisional.record()),
    )
    receipt.authenticate()
    return receipt


def submit_patch_v10(
    task_commitment_sha256: str,
    payload: dict[str, str],
    prior_receipt: ActionReceiptV10 | None = None,
) -> ActionReceiptV10:
    """Apply one diff and run the committed private cases."""

    task = _evaluator_task(task_commitment_sha256)
    patch, patch_sha256 = _validated_patch(task_commitment_sha256, payload)
    evaluator = task.private.evaluator
    if task.private.expected_action == STOP:
        certificate = task.private.ambiguity_certificate
        if certificate is None:
            raise ContinualToySweTaskV10Error("STOP certificate is unavailable")
        evaluator_sha256 = _digest(
            b"continual-repository-swe-stop-acceptance-v10",
            {"ambiguity_certificate_sha256": certificate.certificate_sha256},
        )
        if prior_receipt is None:
            sequence = 1
            prior_sha256 = _RECEIPT_GENESIS_SHA256
        else:
            prior_receipt.authenticate()
            if (
                prior_receipt.task_commitment_sha256 != task_commitment_sha256
                or prior_receipt.conclusion != _FAILURE
                or prior_receipt.evaluator_sha256 != evaluator_sha256
                or prior_receipt.sequence >= MAX_PATCH_SUBMISSIONS
            ):
                raise ContinualToySweTaskV10Error("prior receipt cannot authorize submission")
            sequence = prior_receipt.sequence + 1
            prior_sha256 = prior_receipt.receipt_sha256
        return _make_receipt(
            task_commitment_sha256=task_commitment_sha256,
            sequence=sequence,
            prior_receipt_sha256=prior_sha256,
            action_sha256=patch_sha256,
            evaluator_sha256=evaluator_sha256,
            sandbox_evaluator_reached=False,
            conclusion=_FAILURE,
            feedback="Patch rejected by the committed acceptance protocol.",
        )
    if evaluator is None:
        raise ContinualToySweTaskV10Error("task has no executable evaluator")

    if prior_receipt is None:
        sequence = 1
        prior_sha256 = _RECEIPT_GENESIS_SHA256
        revised = False
    else:
        prior_receipt.authenticate()
        if (
            prior_receipt.task_commitment_sha256 != task_commitment_sha256
            or prior_receipt.conclusion == _SUCCESS
            or prior_receipt.sequence >= MAX_PATCH_SUBMISSIONS
        ):
            raise ContinualToySweTaskV10Error("prior receipt cannot authorize submission")
        initial_sha256 = evaluator.phase_sha256(revised=False)
        revised_sha256 = (
            None if evaluator.revised_cases is None else evaluator.phase_sha256(revised=True)
        )
        if prior_receipt.evaluator_sha256 not in {initial_sha256, revised_sha256}:
            raise ContinualToySweTaskV10Error("prior receipt evaluator changed")
        if prior_receipt.conclusion == _ACTION_REQUIRED:
            plan = task.private.review_plan
            if (
                plan is None
                or revised_sha256 is None
                or prior_receipt.evaluator_sha256 != initial_sha256
                or prior_receipt.feedback != plan.review_text
                or prior_receipt.action_sha256 == patch_sha256
            ):
                raise ContinualToySweTaskV10Error("receipt does not authorize a distinct revision")
            revised = True
        else:
            revised = prior_receipt.evaluator_sha256 == revised_sha256
        sequence = prior_receipt.sequence + 1
        prior_sha256 = prior_receipt.receipt_sha256

    evaluator_sha256 = evaluator.phase_sha256(revised=revised)
    passed = _run_private_evaluator(task, patch, patch_sha256, revised=revised)
    if not passed:
        conclusion = _FAILURE
        feedback = "Patch rejected by the committed acceptance protocol."
    elif not revised and task.private.review_plan is not None:
        conclusion = _ACTION_REQUIRED
        feedback = task.private.review_plan.review_text
    else:
        conclusion = _SUCCESS
        feedback = "Patch accepted by the committed acceptance protocol."
    return _make_receipt(
        task_commitment_sha256=task_commitment_sha256,
        sequence=sequence,
        prior_receipt_sha256=prior_sha256,
        action_sha256=patch_sha256,
        evaluator_sha256=evaluator_sha256,
        sandbox_evaluator_reached=True,
        conclusion=conclusion,
        feedback=feedback,
    )


def seal_action_summary_v10(
    task_commitment_sha256: str,
    attempt: int,
    action: str,
    receipt: ActionReceiptV10 | None = None,
) -> SealedActionSummaryV10:
    conclusion = (
        receipt.conclusion if receipt is not None else (_NEUTRAL if action == STOP else _FAILURE)
    )
    provisional = SealedActionSummaryV10(
        task_commitment_sha256,
        attempt,
        action,
        conclusion,
        receipt,
        "",
    )
    summary = SealedActionSummaryV10(
        task_commitment_sha256,
        attempt,
        action,
        conclusion,
        receipt,
        _digest(b"continual-repository-swe-sealed-action-v10", provisional.record()),
    )
    summary.authenticate()
    return summary


def train_stream_sha256_v10() -> str:
    _authenticate_external_manifest()
    return _stream_root(_PUBLIC_STREAM)


def train_stream_custody_v10() -> dict[str, object]:
    _authenticate_external_manifest()
    record = _custody_record()
    record["train_stream_custody_sha256"] = TRAIN_STREAM_CUSTODY_SHA256
    return json.loads(_canonical_json(record).decode("ascii"))


def posthoc_score_terminal_actions_v10(
    task_commitment_sha256: str,
    completed_actions: tuple[SealedActionSummaryV10, ...],
    *,
    all_arm_streams_finished: bool,
) -> dict[str, object]:
    """Score sealed terminal actions only after all experimental streams finish."""

    if all_arm_streams_finished is not True:
        raise ContinualToySweTaskV10Error("posthoc scoring is forbidden before arm completion")
    if type(completed_actions) is not tuple or len(completed_actions) > MAX_PATCH_SUBMISSIONS:
        raise ContinualToySweTaskV10Error("sealed action sequence changed")
    task = _evaluator_task(task_commitment_sha256)
    action_names: list[str] = []
    prior_receipt_sha256 = _RECEIPT_GENESIS_SHA256
    expected_sequence = 1
    terminal_seen = False
    for attempt, summary in enumerate(completed_actions, start=1):
        if type(summary) is not SealedActionSummaryV10 or terminal_seen:
            raise ContinualToySweTaskV10Error("sealed action sequence changed")
        summary.authenticate()
        if summary.task_commitment_sha256 != task_commitment_sha256 or summary.attempt != attempt:
            raise ContinualToySweTaskV10Error("sealed action order changed")
        action_names.append(summary.action)
        if summary.receipt is not None:
            receipt = summary.receipt
            if (
                receipt.sequence != expected_sequence
                or receipt.prior_receipt_sha256 != prior_receipt_sha256
            ):
                raise ContinualToySweTaskV10Error("sealed receipt chain changed")
            expected_sequence += 1
            prior_receipt_sha256 = receipt.receipt_sha256
            terminal_seen = receipt.conclusion == _SUCCESS
        else:
            terminal_seen = summary.action == STOP

    expected_matches = tuple(
        action_name == task.private.expected_action for action_name in action_names
    )
    winning_attempt: int | None = None
    if task.private.expected_action == STOP:
        if expected_matches and expected_matches[-1]:
            winning_attempt = len(expected_matches)
    else:
        for attempt, summary in enumerate(completed_actions, start=1):
            if summary.receipt is not None and summary.receipt.conclusion == _SUCCESS:
                winning_attempt = attempt
                break
    return {
        "stage": task.private.stage,
        "expected_action_matches": expected_matches,
        "terminal_pass": winning_attempt is not None,
        "winning_attempt": winning_attempt,
    }


def _materialized_witnesses_v10(
    task_commitment_sha256: str,
    *,
    revised: bool = False,
) -> tuple[dict[str, object], ...]:
    """Evaluator-only detached witnesses used by focused fixture tests."""

    task = _evaluator_task(task_commitment_sha256)
    if revised:
        if task.private.review_plan is None:
            raise ContinualToySweTaskV10Error("task has no revised witness set")
        witnesses = task.private.review_plan.revised_witnesses
    else:
        witnesses = task.private.witnesses
    return tuple(
        {
            "case_id": witness.case_id,
            "request": json.loads(witness.request_json),
            "expected": json.loads(witness.expected_json),
        }
        for witness in witnesses
    )


__all__ = [
    "GLOBAL_ACTIONS",
    "INVALID_ACTION",
    "MANIFEST_FILENAME",
    "MAX_PATCH_SUBMISSIONS",
    "PARTITION",
    "PYTHON_EXECUTABLE_SHA256",
    "REPOSITORY_WORKER_SHA256",
    "SCHEMA",
    "SIMPLE_SWE_SANDBOX_SHA256",
    "SOURCE_PRIOR_CASE_COUNT",
    "SOURCE_PRIOR_CORPUS_SHA256",
    "STOP",
    "SUBMIT_PATCH",
    "TASK_COUNT",
    "TASK_SOURCE_SHA256",
    "TOOL_CONTRACT_SHA256",
    "TRAIN_STREAM_CUSTODY_SHA256",
    "TRAIN_STREAM_SHA256",
    "ActionReceiptV10",
    "CandidatePatchRejectedV10",
    "ContinualToySweTaskV10Error",
    "LearnerVisibleToySweObservationV10",
    "SealedActionSummaryV10",
    "global_tool_contract_v10",
    "learner_workspace_files_v10",
    "posthoc_score_terminal_actions_v10",
    "preflight_execution_substrate_v10",
    "public_train_tasks_v10",
    "runner_mount_allowlist_v10",
    "runner_writable_source_allowlist_v10",
    "seal_action_summary_v10",
    "submit_patch_v10",
    "train_stream_custody_v10",
    "train_stream_sha256_v10",
]
