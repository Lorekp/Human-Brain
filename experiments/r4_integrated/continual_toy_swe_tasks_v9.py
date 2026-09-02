"""Fresh public-TRAIN repository tasks for integrated continual coding V9.

V9 contains twelve deterministic repository snapshots spanning direct repair,
cross-source composition, consequence-driven revision, delayed renamed transfer,
and genuine ambiguity. Learners receive only an opaque task commitment, an issue, a snapshot
commitment, and one global action-contract commitment. The mounted repository is
supplied separately through an allowlist; target files, signatures, spans,
family/stage labels, retry metadata, hidden witnesses, and future review text are
not part of the observation.

This module is a fixture and evaluator protocol, not a learner or session runner.
It stores no successful or candidate patch. It imports and runs no learner or
model, creates no benchmark artifact, and touches no protected split.
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

SCHEMA = "continual-repository-swe-tasks-v9"
MANIFEST_SCHEMA = "continual-repository-swe-custody-manifest-v9"
PARTITION = "train"
TASK_COUNT = 12
# Immutable logical identity of the public_coding_prior_v1 bootstrap corpus.
# It is source-prior custody, not a prompt transform, bootstrap, or runner API hash.
SOURCE_PRIOR_CORPUS_SHA256 = "81a60454b9af6b8592ea8b422c25bdd98115c101e5580c8cd571555fdf0abfb1"
SOURCE_PRIOR_CASE_COUNT = 8
MANIFEST_FILENAME = "continual_toy_swe_tasks_v9_manifest.json"

SUBMIT_PATCH = "submit_patch"
STOP = "<stop>"
INVALID_ACTION = "invalid"
GLOBAL_ACTIONS = (SUBMIT_PATCH,)
MAX_PATCH_SUBMISSIONS = 3

_DIRECT = "direct-repository-relation"
_COMPOSITION = "issue-doc-test-implementation-composition"
_REVISION = "consequence-guided-revision"
_INSUFFICIENT = "genuinely-ambiguous-stop"
_STAGES = (_DIRECT, _COMPOSITION, _REVISION, _INSUFFICIENT)

_CREDIT_ARITHMETIC = "credit-arithmetic"
_NORMALIZED_UNIQUENESS = "normalized-uniqueness"
_INCLUSIVE_BOUNDARY = "inclusive-boundary"
_LATENCY_SELECTION = "latency-selection"
_MOISTURE_CALIBRATION = "moisture-calibration"
_TAIL_PRESERVATION = "tail-preservation"
_TIE_POLICY = "unresolved-tie-policy"
_AGE_POLICY = "age-threshold-revision"
_FALLBACK_POLICY = "unresolved-fallback-policy"
_FAMILIES = (
    _CREDIT_ARITHMETIC,
    _NORMALIZED_UNIQUENESS,
    _INCLUSIVE_BOUNDARY,
    _LATENCY_SELECTION,
    _MOISTURE_CALIBRATION,
    _TAIL_PRESERVATION,
    _TIE_POLICY,
    _AGE_POLICY,
    _FALLBACK_POLICY,
)

_SUCCESS = "success"
_FAILURE = "failure"
_ACTION_REQUIRED = "action-required"
_NEUTRAL = "neutral"
_RECEIPT_CONCLUSIONS = (_SUCCESS, _FAILURE, _ACTION_REQUIRED)

_RELATION_CREDIT = sha256(b"v9:credit-reduces-base").hexdigest()
_RELATION_NORMALIZE_UNIQUE = sha256(b"v9:normalize-before-stable-unique").hexdigest()
_RELATION_INCLUSIVE_END = sha256(b"v9:inclusive-upper-bound").hexdigest()
_RECEIPT_GENESIS_SHA256 = sha256(b"v9:action-receipt-genesis").hexdigest()
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


class ContinualToySweTaskV9Error(ValueError):
    """A V9 task, receipt, repository, or custody record is unauthenticated."""


class CandidatePatchRejectedV9(ContinualToySweTaskV9Error):
    """A candidate diff is invalid; evaluator/custody failures use the base type."""


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
        "schema": "continual-repository-swe-tool-contract-v9",
        "external_mutating_affordances": (
            {
                "name": SUBMIT_PATCH,
                "payload": {
                    "type": "object",
                    "required": ("patch",),
                    "additionalProperties": False,
                    "properties": {"patch": {"type": "string"}},
                },
            },
        ),
        "max_patch_submissions": MAX_PATCH_SUBMISSIONS,
        "instructions": (
            "Inspect and test the learner-readable repository. Submit exactly one complete "
            "unified diff when the available evidence determines a correct change. The "
            "neuronal <stop> choice is terminal and neutral: it never crosses the external "
            "executor and never produces an action receipt. Evaluator checks return only a "
            "conclusion unless a reviewer supplies new public text."
        ),
        "learner_readable_files": (
            "the exact task mount allowlist, including public tests as read-only evidence"
        ),
        "writable_files": "existing Python files under the repository package directory",
        "patch_paths": (
            "repository-relative canonical POSIX paths from the writable source allowlist; "
            "tests, configuration, README, additions, deletions, renames, mode changes, "
            "binary patches, absolute paths, and traversal are rejected"
        ),
        "patch_hashing": "the runner computes sha256 from the submitted UTF-8 diff",
        "terminal_neutral_choice": STOP,
    }


TOOL_CONTRACT_SHA256 = _digest(
    b"continual-repository-swe-tool-contract-v9", _tool_contract_record()
)


@dataclass(frozen=True, slots=True)
class RepositoryFileV9:
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
            or self.relative_path != path.as_posix()
        ):
            raise ContinualToySweTaskV9Error("repository path is not canonical")
        if type(self.content_utf8) is not bytes:
            raise TypeError("repository content must be immutable bytes")
        if sha256(self.content_utf8).hexdigest() != self.file_sha256:
            raise ContinualToySweTaskV9Error("repository file changed")


@dataclass(frozen=True, slots=True)
class RepositorySnapshotV9:
    files: tuple[RepositoryFileV9, ...]
    snapshot_sha256: str

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "files": tuple(file.record() for file in self.files)}

    def authenticate(self) -> None:
        if not isinstance(self.files, tuple) or not self.files:
            raise ContinualToySweTaskV9Error("repository snapshot is empty or mutable")
        for file in self.files:
            if type(file) is not RepositoryFileV9:
                raise TypeError("repository snapshot must contain exact immutable files")
            file.authenticate()
        paths = tuple(file.relative_path for file in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ContinualToySweTaskV9Error("repository paths changed")
        expected = _digest(b"continual-repository-snapshot-v9", self.record())
        if self.snapshot_sha256 != expected:
            raise ContinualToySweTaskV9Error("repository snapshot changed")


@dataclass(frozen=True, slots=True)
class LearnerVisibleToySweObservationV9:
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
            raise ContinualToySweTaskV9Error("learner observation has an invalid commitment")
        if not self.instance_id or not self.issue:
            raise ContinualToySweTaskV9Error("learner observation is incomplete")
        if self.tool_contract_sha256 != TOOL_CONTRACT_SHA256:
            raise ContinualToySweTaskV9Error("task-specific action contract is forbidden")
        expected = _digest(b"continual-repository-swe-observation-v9", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskV9Error("learner observation changed")


@dataclass(frozen=True, slots=True)
class FrozenWitnessV9:
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
            raise ContinualToySweTaskV9Error("witness request is not canonical JSON")
        if _frozen_json(self.expected_json.decode("ascii")) != self.expected_json:
            raise ContinualToySweTaskV9Error("witness expectation is not canonical JSON")
        expected = _digest(b"continual-repository-swe-witness-v9", self.record())
        if self.witness_sha256 != expected:
            raise ContinualToySweTaskV9Error("acceptance witness changed")


@dataclass(frozen=True, slots=True)
class FrozenEvaluatorV9:
    entry_module: str
    entry_point: str
    initial_cases: tuple[FrozenWitnessV9, ...]
    revised_cases: tuple[FrozenWitnessV9, ...] | None
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
            raise ContinualToySweTaskV9Error("revised evaluator cases are unavailable")
        return _digest(
            b"continual-repository-swe-evaluator-phase-v9",
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
            raise ContinualToySweTaskV9Error("private evaluator entry point changed")
        if type(self.initial_cases) is not tuple or len(self.initial_cases) < 2:
            raise ContinualToySweTaskV9Error("private evaluator lacks frozen cases")
        for cases in (self.initial_cases, self.revised_cases):
            if cases is not None:
                if type(cases) is not tuple or len(cases) < 2:
                    raise ContinualToySweTaskV9Error("private evaluator cases changed")
                for case in cases:
                    if type(case) is not FrozenWitnessV9:
                        raise TypeError("private evaluator case type changed")
                    case.authenticate()
        expected = _digest(b"continual-repository-swe-evaluator-v9", self.record())
        if self.evaluator_sha256 != expected:
            raise ContinualToySweTaskV9Error("private evaluator changed")


@dataclass(frozen=True, slots=True)
class ReviewPlanV9:
    review_text: str
    revised_witnesses: tuple[FrozenWitnessV9, ...]
    plan_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "review_text": self.review_text,
            "revised_witness_sha256s": tuple(
                witness.witness_sha256 for witness in self.revised_witnesses
            ),
        }

    def authenticate(self, initial_witnesses: tuple[FrozenWitnessV9, ...]) -> None:
        if not self.review_text or len(self.revised_witnesses) < 2:
            raise ContinualToySweTaskV9Error("review plan lacks public evidence")
        for witness in self.revised_witnesses:
            witness.authenticate()
        initial = {witness.request_json: witness.expected_json for witness in initial_witnesses}
        revised = {
            witness.request_json: witness.expected_json for witness in self.revised_witnesses
        }
        shared = initial.keys() & revised.keys()
        if not shared or not any(initial[key] != revised[key] for key in shared):
            raise ContinualToySweTaskV9Error("review does not require a behavioral revision")
        expected = _digest(b"continual-repository-swe-review-plan-v9", self.record())
        if self.plan_sha256 != expected:
            raise ContinualToySweTaskV9Error("review plan changed")


@dataclass(frozen=True, slots=True)
class AmbiguityCertificateV9:
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
            raise ContinualToySweTaskV9Error("ambiguity evidence is empty")
        if _frozen_json(self.witness_request_json.decode("ascii")) != self.witness_request_json:
            raise ContinualToySweTaskV9Error("ambiguity request is not canonical")
        if len(set(self.compatible_outcome_jsons)) != 2:
            raise ContinualToySweTaskV9Error("ambiguity outcomes do not disagree")
        for outcome in self.compatible_outcome_jsons:
            if _frozen_json(outcome.decode("ascii")) != outcome:
                raise ContinualToySweTaskV9Error("ambiguity outcome is not canonical")
        expected = _digest(b"continual-repository-swe-ambiguity-v9", self.record())
        if self.certificate_sha256 != expected:
            raise ContinualToySweTaskV9Error("ambiguity certificate changed")


@dataclass(frozen=True, slots=True)
class PrivateAcceptanceV9:
    family_id: str
    stage: str
    expected_action: str
    retention_anchor_ordinal: int | None
    transfer_relation_sha256: str | None
    witnesses: tuple[FrozenWitnessV9, ...]
    evaluator: FrozenEvaluatorV9 | None
    review_plan: ReviewPlanV9 | None
    ambiguity_certificate: AmbiguityCertificateV9 | None
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
            "review_plan_sha256": None
            if self.review_plan is None
            else self.review_plan.plan_sha256,
            "ambiguity_certificate_sha256": (
                None
                if self.ambiguity_certificate is None
                else self.ambiguity_certificate.certificate_sha256
            ),
        }

    def authenticate(self, ordinal: int) -> None:
        if self.family_id not in _FAMILIES or self.stage not in _STAGES:
            raise ContinualToySweTaskV9Error("private family or stage changed")
        if self.expected_action not in (SUBMIT_PATCH, STOP):
            raise ContinualToySweTaskV9Error("private terminal action changed")
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
                raise ContinualToySweTaskV9Error("STOP task is not evidence-limited")
            self.ambiguity_certificate.authenticate()
        else:
            if (
                not self.witnesses
                or self.evaluator is None
                or self.ambiguity_certificate is not None
            ):
                raise ContinualToySweTaskV9Error("executable task acceptance changed")
            self.evaluator.authenticate()
            if self.evaluator.initial_cases != self.witnesses:
                raise ContinualToySweTaskV9Error("initial evaluator cases changed")
            if self.stage == _REVISION:
                if self.review_plan is None or self.evaluator.revised_cases is None:
                    raise ContinualToySweTaskV9Error("revision task lost its review plan")
                self.review_plan.authenticate(self.witnesses)
                if self.evaluator.revised_cases != self.review_plan.revised_witnesses:
                    raise ContinualToySweTaskV9Error("revised evaluator cases changed")
            elif self.review_plan is not None or self.evaluator.revised_cases is not None:
                raise ContinualToySweTaskV9Error("non-revision task gained future review text")
        if self.retention_anchor_ordinal is not None:
            if (
                self.expected_action == STOP
                or type(self.retention_anchor_ordinal) is not int
                or not 0 <= self.retention_anchor_ordinal < ordinal
                or not _is_sha256(self.transfer_relation_sha256)
            ):
                raise ContinualToySweTaskV9Error("delayed-transfer lineage changed")
        elif self.transfer_relation_sha256 is not None and self.expected_action == STOP:
            raise ContinualToySweTaskV9Error("STOP task gained a transfer relation")
        expected = _digest(b"continual-repository-swe-private-v9", self.record())
        if self.bundle_sha256 != expected:
            raise ContinualToySweTaskV9Error("private acceptance bundle changed")


@dataclass(frozen=True, slots=True)
class _EvaluatorTaskV9:
    ordinal: int
    public: LearnerVisibleToySweObservationV9
    repository: RepositorySnapshotV9
    private: PrivateAcceptanceV9

    def authenticate(self) -> None:
        self.public.authenticate()
        self.repository.authenticate()
        self.private.authenticate(self.ordinal)
        expected_commitment = _digest(
            b"continual-repository-swe-task-v9",
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
        if self.public.task_commitment_sha256 != expected_commitment:
            raise ContinualToySweTaskV9Error("public task is not bound to private acceptance")
        if self.public.repository_snapshot_sha256 != self.repository.snapshot_sha256:
            raise ContinualToySweTaskV9Error("public task is not bound to its repository")


@dataclass(frozen=True, slots=True)
class ActionReceiptV9:
    """Action-bound public feedback produced after a learner action."""

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
            "schema": "continual-repository-swe-action-receipt-v9",
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
            not _is_sha256(self.task_commitment_sha256)
            or not _is_sha256(self.prior_receipt_sha256)
            or not _is_sha256(self.action_sha256)
            or not _is_sha256(self.evaluator_sha256)
            or type(self.sandbox_evaluator_reached) is not bool
            or self.action != SUBMIT_PATCH
            or self.conclusion not in _RECEIPT_CONCLUSIONS
            or type(self.sequence) is not int
            or not 1 <= self.sequence <= MAX_PATCH_SUBMISSIONS
            or not self.feedback
        ):
            raise ContinualToySweTaskV9Error("action receipt fields changed")
        expected = _digest(b"continual-repository-swe-action-receipt-v9", self.record())
        if self.receipt_sha256 != expected:
            raise ContinualToySweTaskV9Error("action receipt changed")


@dataclass(frozen=True, slots=True)
class SealedActionSummaryV9:
    """Immutable action-kind summary for posthoc scoring after all arms finish."""

    task_commitment_sha256: str
    attempt: int
    action: str
    conclusion: str
    receipt: ActionReceiptV9 | None
    summary_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": "continual-repository-swe-sealed-action-v9",
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
            raise ContinualToySweTaskV9Error("sealed action fields changed")
        if self.receipt is None:
            expected_conclusion = _NEUTRAL if self.action == STOP else _FAILURE
            if self.conclusion != expected_conclusion:
                raise ContinualToySweTaskV9Error("unreceipted action conclusion changed")
        else:
            if type(self.receipt) is not ActionReceiptV9 or self.action != SUBMIT_PATCH:
                raise ContinualToySweTaskV9Error("sealed receipt action changed")
            self.receipt.authenticate()
            if (
                self.receipt.task_commitment_sha256 != self.task_commitment_sha256
                or self.conclusion != self.receipt.conclusion
            ):
                raise ContinualToySweTaskV9Error("sealed receipt conclusion changed")
        expected = _digest(b"continual-repository-swe-sealed-action-v9", self.record())
        if self.summary_sha256 != expected:
            raise ContinualToySweTaskV9Error("sealed action summary changed")


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
    name = "public-train-repository"
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


def _quay_ledger_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "quayledger",
        """
        # Quay ledger

        The settlement service combines a validated cargo amount with a voucher.
        Forecast projections are informational and do not affect a settlement.
        """,
        (
            (
                "amounts.py",
                """
                def gross(units: int, unit_price: int) -> int:
                    return units * unit_price


                def voucher_credit(voucher: int) -> int:
                    return max(voucher, 0)
                """,
            ),
            (
                "service.py",
                """
                from quayledger.amounts import gross, voucher_credit


                def settle(units: int, unit_price: int, voucher: int) -> int:
                    amount = gross(units, unit_price)
                    return amount + voucher_credit(voucher)
                """,
            ),
            (
                "forecast.py",
                """
                def projected(units: int, unit_price: int, drift: int) -> int:
                    return units * unit_price + drift
                """,
            ),
        ),
        "test_quay_contract.py",
        """
        from quayledger.service import settle


        def test_voucher_reduces_the_amount() -> None:
            assert settle(4, 6, 5) == 19
        """,
    )


def _folio_caption_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "folioleaf",
        """
        # Folio leaf captions

        Captions keep the cleaned title and a stable sequence of visible labels.
        The detailed label procedure is part of the public handbook.
        """,
        (
            (
                "text.py",
                """
                def clean(value: str) -> str:
                    return value.strip().lower()


                def stable_unique(values: list[str]) -> list[str]:
                    result: list[str] = []
                    for value in values:
                        if value not in result:
                            result.append(value)
                    return result
                """,
            ),
            (
                "caption.py",
                """
                from folioleaf.text import clean, stable_unique


                def caption(title: str, labels: list[str]) -> str:
                    cleaned = [clean(item) for item in stable_unique(labels)]
                    return f"{title.strip()} :: {' | '.join(cleaned)}"
                """,
            ),
            (
                "palette.py",
                """
                def swatch_names(values: list[str]) -> list[str]:
                    return [value.strip() for value in values]
                """,
            ),
        ),
        "test_folio_caption.py",
        """
        from folioleaf.caption import caption


        def test_equivalent_labels_collapse_after_cleaning() -> None:
            assert caption(" Dawn ", [" Blue ", "blue", "GOLD"]) == "Dawn :: blue | gold"
        """,
        documents=(
            (
                "label_handbook.md",
                """
                A label is stripped and folded to lowercase before duplicate labels are
                removed. The first occurrence of each cleaned label wins. Captions place
                the cleaned labels after the title, separated by vertical bars.
                """,
            ),
        ),
    )


def _fen_window_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "fenwatch",
        """
        # Fen observation window

        Window checks are shared by field observations. Calendar display and naming
        helpers are separate concerns.
        """,
        (
            (
                "bounds.py",
                """
                def is_open(day: int, first_day: int, final_day: int) -> bool:
                    return first_day <= day < final_day
                """,
            ),
            (
                "calendar.py",
                """
                def month_slot(day: int) -> int:
                    return (day - 1) // 30 + 1
                """,
            ),
            (
                "labels.py",
                """
                def window_label(first_day: int, final_day: int) -> str:
                    return f"{first_day}-{final_day}"
                """,
            ),
        ),
        "test_fen_window.py",
        """
        from fenwatch.bounds import is_open


        def test_first_day_is_open() -> None:
            assert is_open(3, 3, 8)
        """,
    )


def _swift_selection_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "swiftpick",
        """
        # Swift signal selection

        A report identifies one enabled signal. Eligibility, ranking, and display are
        separate modules so each public rule can be inspected independently.
        """,
        (
            (
                "candidates.py",
                """
                def enabled(rows: list[dict[str, object]]) -> list[dict[str, object]]:
                    return [row for row in rows if row["enabled"] is True]
                """,
            ),
            (
                "ranking.py",
                """
                from swiftpick.candidates import enabled


                def rank(row: dict[str, object]) -> tuple[object, object]:
                    return (row["latency"], row["code"])


                def choose(rows: list[dict[str, object]]) -> dict[str, object]:
                    choices = enabled(rows)
                    return max(choices, key=rank)
                """,
            ),
            (
                "report.py",
                """
                from swiftpick.ranking import choose


                def best_signal(rows: list[dict[str, object]]) -> str:
                    selected = choose(rows)
                    return f"{selected['latency']}@{selected['code']}"
                """,
            ),
            (
                "fallback.py",
                """
                def fallback_code(rows: list[dict[str, object]]) -> str:
                    return str(rows[0]["code"]) if rows else "none"
                """,
            ),
        ),
        "test_swift_report.py",
        """
        from swiftpick.report import best_signal


        def test_enabled_low_latency_signal_is_reported() -> None:
            rows = [
                {"code": "iris", "enabled": True, "latency": 8},
                {"code": "wren", "enabled": True, "latency": 3},
                {"code": "ash", "enabled": False, "latency": 1},
            ]
            assert best_signal(rows) == "wren@3"
        """,
        documents=(
            (
                "selection_guide.md",
                """
                Ignore disabled signals. Of the remaining signals, choose the smallest
                latency; equal latency is ordered by code. Render the chosen code first,
                followed by an at sign and its latency.
                """,
            ),
        ),
    )


def _loam_calibration_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "loammark",
        """
        # Loam moisture marks

        The service exposes a compact public moisture classification. Sampling and
        display helpers must not redefine the calibrated boundary.
        """,
        (
            (
                "bands.py",
                """
                def moisture_band(moisture: int) -> str:
                    return "dry" if moisture <= 2 else "ready"
                """,
            ),
            (
                "service.py",
                """
                from loammark.bands import moisture_band


                def describe(moisture: int) -> str:
                    return moisture_band(moisture)
                """,
            ),
            (
                "samples.py",
                """
                def average(first: int, second: int) -> float:
                    return (first + second) / 2
                """,
            ),
        ),
        "test_loammark.py",
        """
        from loammark.service import describe


        def test_low_reading_is_dry() -> None:
            assert describe(1) == "dry"
        """,
    )


def _reed_batch_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "reedpack",
        """
        # Reed batches

        The batching API preserves input order. Width validation, batch construction,
        and decorative labels live in separate modules.
        """,
        (
            (
                "validation.py",
                """
                def checked_width(width: int) -> int:
                    if width < 1:
                        raise ValueError("width must be positive")
                    return width
                """,
            ),
            (
                "batches.py",
                """
                from reedpack.validation import checked_width


                def partition(items: list[str], width: int) -> list[list[str]]:
                    size = checked_width(width)
                    groups: list[list[str]] = []
                    current: list[str] = []
                    for item in items:
                        current.append(item)
                        if len(current) == size:
                            groups.append(current)
                            current = []
                    return groups
                """,
            ),
            (
                "labels.py",
                """
                def batch_label(index: int) -> str:
                    return f"batch-{index}"
                """,
            ),
        ),
        "test_reed_batches.py",
        """
        from reedpack.batches import partition


        def test_complete_batches_keep_order() -> None:
            assert partition(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]
        """,
    )


def _canal_settlement_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "canalsettle",
        """
        # Canal settlement

        The public service combines parcel cost, an accepted rebate, and handling.
        Audit formatting and forecasts are unrelated to the settlement relation.
        """,
        (
            (
                "formula.py",
                """
                def parcel_gross(parcels: int, unit_cost: int) -> int:
                    return parcels * unit_cost


                def apply_rebate(amount: int, rebate: int) -> int:
                    return amount + max(rebate, 0)
                """,
            ),
            (
                "service.py",
                """
                from canalsettle.formula import apply_rebate, parcel_gross


                def finalize(parcels: int, unit_cost: int, rebate: int, handling: int) -> int:
                    adjusted = apply_rebate(parcel_gross(parcels, unit_cost), rebate)
                    return adjusted + handling
                """,
            ),
            (
                "audit.py",
                """
                def audit_line(parcels: int, handling: int) -> str:
                    return f"parcels={parcels};handling={handling}"
                """,
            ),
            (
                "forecast.py",
                """
                def forecast(parcels: int, unit_cost: int, growth: int) -> int:
                    return parcels * unit_cost + growth
                """,
            ),
        ),
        "test_canal_settlement.py",
        """
        from canalsettle.service import finalize


        def test_zero_rebate_preserves_base_and_handling() -> None:
            assert finalize(3, 10, 0, 2) == 32
        """,
        documents=(
            (
                "settlement_charter.md",
                """
                Parcel gross, rebate adjustment, and handling remain separate named steps.
                The rebate adjustment follows the earlier Quay voucher settlement relation.
                """,
            ),
        ),
    )


def _cloud_tie_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "cloudvote",
        """
        # Cloud marker choice

        Higher confidence wins. The owner has not selected a side for equal confidence;
        both historical choices remain compatible with the published contract.
        """,
        (
            (
                "choice.py",
                """
                def choose(left: str, right: str, left_score: int, right_score: int) -> str:
                    return left if left_score >= right_score else right
                """,
            ),
            (
                "scores.py",
                """
                def confidence(value: int) -> int:
                    return max(value, 0)
                """,
            ),
            (
                "history.py",
                """
                def prior_choices() -> tuple[str, str]:
                    return ("left", "right")
                """,
            ),
        ),
        "test_cloud_choice.py",
        """
        from cloudvote.choice import choose


        def test_higher_confidence_wins() -> None:
            assert choose("cirrus", "nimbus", 9, 4) == "cirrus"
        """,
    )


def _ridge_permit_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "ridgepermit",
        """
        # Ridge permits

        A permit decision uses a configured floor and ceiling. Labels and advisory
        projections do not alter the decision interval.
        """,
        (
            (
                "span.py",
                """
                def admitted(level: int, floor: int, ceiling: int) -> bool:
                    above_floor = floor <= level
                    below_ceiling = level < ceiling
                    return above_floor and below_ceiling
                """,
            ),
            (
                "labels.py",
                """
                def permit_label(floor: int, ceiling: int) -> str:
                    return f"permit:{floor}:{ceiling}"
                """,
            ),
            (
                "advisory.py",
                """
                def projected(level: int, change: int) -> int:
                    return level + change
                """,
            ),
        ),
        "test_ridge_permit.py",
        """
        from ridgepermit.span import admitted


        def test_floor_is_admitted() -> None:
            assert admitted(4, 4, 9)
        """,
    )


def _prism_queue_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "prismdesk",
        """
        # Prism queue desk

        Queue age maps to one of two public priority labels. Display and audit modules
        consume the classification but do not set its threshold.
        """,
        (
            (
                "rule.py",
                """
                def priority(age: int) -> str:
                    return "expedite" if age > 7 else "standard"
                """,
            ),
            (
                "service.py",
                """
                from prismdesk.rule import priority


                def queue_label(age: int) -> str:
                    return priority(age)
                """,
            ),
            (
                "display.py",
                """
                def decorated(label: str) -> str:
                    return f"[{label}]"
                """,
            ),
            (
                "audit.py",
                """
                def audit_age(age: int) -> str:
                    return f"age={age}"
                """,
            ),
        ),
        "test_prism_queue.py",
        """
        from prismdesk.service import queue_label


        def test_old_item_is_expedited() -> None:
            assert queue_label(12) == "expedite"
        """,
    )


def _quilt_legend_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "quiltlegend",
        """
        # Quilt legends

        A legend keeps a title plus a list of visible swatch names. Identity and ordering
        follow a convention established earlier in the public training stream.
        """,
        (
            (
                "textops.py",
                """
                def canonical(value: str) -> str:
                    return value.strip().lower()


                def first_seen(values: list[str]) -> list[str]:
                    result: list[str] = []
                    for value in values:
                        if value not in result:
                            result.append(value)
                    return result
                """,
            ),
            (
                "legend.py",
                """
                from quiltlegend.textops import canonical, first_seen


                def render(name: str, swatches: list[str]) -> str:
                    unique = first_seen(swatches)
                    cleaned = [canonical(value) for value in unique]
                    return f"{name.strip()} => {','.join(cleaned)}"
                """,
            ),
            (
                "colors.py",
                """
                def color_count(swatches: list[str]) -> int:
                    return len(swatches)
                """,
            ),
            (
                "archive.py",
                """
                def archive_name(name: str) -> str:
                    return name.strip().replace(" ", "-")
                """,
            ),
        ),
        "test_quilt_legend.py",
        """
        from quiltlegend.legend import render


        def test_distinct_clean_swatches_render_in_order() -> None:
            assert render(" Sky ", ["teal", "ochre"]) == "Sky => teal,ochre"
        """,
        documents=(
            (
                "craft_note.md",
                """
                Swatch identity and ordering follow the earlier Folio label convention.
                Keep the local legend rendering unchanged.
                """,
            ),
            (
                "separator_note.md",
                """
                Legend titles precede a right-facing pair and visible swatches use commas.
                """,
            ),
        ),
    )


def _seed_fallback_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "seedreserve",
        """
        # Seed reserve choice

        If exactly one store is ready, use it. The public material does not state which
        store wins when both are ready; records contain examples of both choices.
        """,
        (
            (
                "choice.py",
                """
                def reserve(
                    first: str,
                    second: str,
                    first_ready: bool,
                    second_ready: bool,
                ) -> str:
                    if first_ready:
                        return first
                    if second_ready:
                        return second
                    return "none"
                """,
            ),
            (
                "readiness.py",
                """
                def exactly_one(first_ready: bool, second_ready: bool) -> bool:
                    return first_ready != second_ready
                """,
            ),
            (
                "records.py",
                """
                def historic_labels() -> tuple[str, str]:
                    return ("first", "second")
                """,
            ),
        ),
        "test_seed_reserve.py",
        """
        from seedreserve.choice import reserve


        def test_only_ready_store_is_used() -> None:
            assert reserve("oak", "birch", False, True) == "birch"
        """,
    )


_SPECS = (
    _TaskSpec(
        "train-v9-quay-voucher-41",
        (
            "A cargo voucher is a credit, never a surcharge. Settlement must subtract the "
            "validated voucher from the cargo amount while leaving validation and forecasts "
            "unchanged."
        ),
        _CREDIT_ARITHMETIC,
        _DIRECT,
        _quay_ledger_repository(),
        SUBMIT_PATCH,
        (
            _case("quay-hidden-elm", {"units": 2, "unit_price": 9, "voucher": 3}, 15),
            _case("quay-hidden-pine", {"units": 5, "unit_price": 4, "voucher": 0}, 20),
            _case("quay-hidden-ash", {"units": 1, "unit_price": 7, "voucher": -2}, 7),
        ),
        "quayledger.service",
        "settle",
        transfer_relation_sha256=_RELATION_CREDIT,
    ),
    _TaskSpec(
        "train-v9-folio-labels-73",
        (
            "Folio captions disagree with the handbook and the public duplicate-label example. "
            "Bring the implementation into agreement without changing palette reporting."
        ),
        _NORMALIZED_UNIQUENESS,
        _COMPOSITION,
        _folio_caption_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "folio-hidden-river",
                {"title": " Noon ", "labels": [" RED", "red ", "Blue"]},
                "Noon :: red | blue",
            ),
            _case(
                "folio-hidden-stone",
                {"title": "Map", "labels": ["A", " b ", "a", "B"]},
                "Map :: a | b",
            ),
            _case("folio-hidden-bare", {"title": " Solo ", "labels": []}, "Solo :: "),
        ),
        "folioleaf.caption",
        "caption",
        transfer_relation_sha256=_RELATION_NORMALIZE_UNIQUE,
    ),
    _TaskSpec(
        "train-v9-fen-final-day-28",
        (
            "Field windows include both named boundary days. A reading on the configured "
            "final day must remain open; do not alter calendar or label helpers."
        ),
        _INCLUSIVE_BOUNDARY,
        _DIRECT,
        _fen_window_repository(),
        SUBMIT_PATCH,
        (
            _case("fen-hidden-final", {"day": 12, "first_day": 5, "final_day": 12}, True),
            _case("fen-hidden-after", {"day": 13, "first_day": 5, "final_day": 12}, False),
            _case("fen-hidden-before", {"day": 4, "first_day": 5, "final_day": 12}, False),
        ),
        "fenwatch.bounds",
        "is_open",
        transfer_relation_sha256=_RELATION_INCLUSIVE_END,
    ),
    _TaskSpec(
        "train-v9-swift-signal-56",
        (
            "The selected-signal report conflicts with the operations guide and public example. "
            "Reconcile eligibility, ordering, and rendered field order across the implementation."
        ),
        _LATENCY_SELECTION,
        _COMPOSITION,
        _swift_selection_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "swift-hidden-low",
                {
                    "rows": [
                        {"code": "kite", "enabled": True, "latency": 9},
                        {"code": "moss", "enabled": True, "latency": 2},
                    ]
                },
                "moss@2",
            ),
            _case(
                "swift-hidden-disabled",
                {
                    "rows": [
                        {"code": "alder", "enabled": False, "latency": 1},
                        {"code": "cedar", "enabled": True, "latency": 6},
                    ]
                },
                "cedar@6",
            ),
            _case(
                "swift-hidden-tie",
                {
                    "rows": [
                        {"code": "zinc", "enabled": True, "latency": 4},
                        {"code": "amber", "enabled": True, "latency": 4},
                    ]
                },
                "amber@4",
            ),
        ),
        "swiftpick.report",
        "best_signal",
    ),
    _TaskSpec(
        "train-v9-loam-calibration-64",
        (
            "The current field specification classifies moisture readings through three as "
            "dry, inclusively. Update the behavior while preserving the two public labels."
        ),
        _MOISTURE_CALIBRATION,
        _REVISION,
        _loam_calibration_repository(),
        SUBMIT_PATCH,
        (
            _case("loam-initial-two", {"moisture": 2}, "dry"),
            _case("loam-initial-three", {"moisture": 3}, "dry"),
            _case("loam-initial-four", {"moisture": 4}, "ready"),
        ),
        "loammark.service",
        "describe",
        review_text=(
            "A later instrument calibration is now public: readings through four are dry, "
            "inclusively. Revise the submitted behavior."
        ),
        revised_witnesses=(
            _case("loam-revised-three", {"moisture": 3}, "dry"),
            _case("loam-revised-four", {"moisture": 4}, "dry"),
            _case("loam-revised-five", {"moisture": 5}, "ready"),
        ),
    ),
    _TaskSpec(
        "train-v9-reed-tail-35",
        (
            "Batching must retain the final partial batch instead of dropping it. Preserve "
            "item order and the existing positive-width check."
        ),
        _TAIL_PRESERVATION,
        _DIRECT,
        _reed_batch_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "reed-hidden-tail",
                {"items": ["a", "b", "c", "d", "e"], "width": 2},
                [["a", "b"], ["c", "d"], ["e"]],
            ),
            _case("reed-hidden-short", {"items": ["fern"], "width": 3}, [["fern"]]),
            _case("reed-hidden-empty", {"items": [], "width": 4}, []),
        ),
        "reedpack.batches",
        "partition",
    ),
    _TaskSpec(
        "train-v9-canal-rebate-82",
        (
            "Apply the earlier Quay voucher settlement relation to Canal's renamed rebate "
            "adjustment while preserving the existing handling step."
        ),
        _CREDIT_ARITHMETIC,
        _COMPOSITION,
        _canal_settlement_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "canal-hidden-rebate",
                {"parcels": 6, "unit_cost": 7, "rebate": 5, "handling": 3},
                40,
            ),
            _case(
                "canal-hidden-zero",
                {"parcels": 2, "unit_cost": 11, "rebate": 0, "handling": 4},
                26,
            ),
            _case(
                "canal-hidden-negative",
                {"parcels": 3, "unit_cost": 4, "rebate": -6, "handling": 1},
                13,
            ),
        ),
        "canalsettle.service",
        "finalize",
        retention_anchor_ordinal=0,
        transfer_relation_sha256=_RELATION_CREDIT,
    ),
    _TaskSpec(
        "train-v9-cloud-tie-17",
        (
            "Choose the approved marker when two cloud readings have equal confidence. "
            "The product request omitted which side is approved."
        ),
        _TIE_POLICY,
        _INSUFFICIENT,
        _cloud_tie_repository(),
        STOP,
        (),
        None,
        None,
        ambiguity=(
            "equal-confidence-owner-choice-is-absent-v9",
            _canonical_json(
                {"left": "cirrus", "right": "nimbus", "left_score": 5, "right_score": 5}
            ).decode("ascii"),
            _canonical_json("cirrus").decode("ascii"),
            _canonical_json("nimbus").decode("ascii"),
        ),
    ),
    _TaskSpec(
        "train-v9-ridge-ceiling-90",
        (
            "Apply the earlier Fen observation-window endpoint convention to the Ridge permit "
            "ceiling without changing floor handling or advisory projection."
        ),
        _INCLUSIVE_BOUNDARY,
        _DIRECT,
        _ridge_permit_repository(),
        SUBMIT_PATCH,
        (
            _case("ridge-hidden-ceiling", {"level": 14, "floor": 8, "ceiling": 14}, True),
            _case("ridge-hidden-above", {"level": 15, "floor": 8, "ceiling": 14}, False),
            _case("ridge-hidden-floor", {"level": 8, "floor": 8, "ceiling": 14}, True),
        ),
        "ridgepermit.span",
        "admitted",
        retention_anchor_ordinal=2,
        transfer_relation_sha256=_RELATION_INCLUSIVE_END,
    ),
    _TaskSpec(
        "train-v9-prism-age-49",
        (
            "Queue desk policy currently says an age of seven or more is expedited. Align "
            "the classification while leaving display and audit output alone."
        ),
        _AGE_POLICY,
        _REVISION,
        _prism_queue_repository(),
        SUBMIT_PATCH,
        (
            _case("prism-initial-six", {"age": 6}, "standard"),
            _case("prism-initial-seven", {"age": 7}, "expedite"),
            _case("prism-initial-ten", {"age": 10}, "expedite"),
        ),
        "prismdesk.service",
        "queue_label",
        review_text=(
            "A superseding desk bulletin is now public: expedited handling begins at age "
            "nine, inclusively. Revise the submitted behavior."
        ),
        revised_witnesses=(
            _case("prism-revised-seven", {"age": 7}, "standard"),
            _case("prism-revised-nine", {"age": 9}, "expedite"),
            _case("prism-revised-twelve", {"age": 12}, "expedite"),
        ),
    ),
    _TaskSpec(
        "train-v9-quilt-swatches-68",
        (
            "Apply the earlier Folio label identity and ordering convention to Quilt swatches "
            "while preserving the local legend rendering."
        ),
        _NORMALIZED_UNIQUENESS,
        _COMPOSITION,
        _quilt_legend_repository(),
        SUBMIT_PATCH,
        (
            _case(
                "quilt-hidden-earth",
                {"name": " Earth ", "swatches": ["CLAY", " clay ", "MOSS"]},
                "Earth => clay,moss",
            ),
            _case(
                "quilt-hidden-night",
                {"name": "Night", "swatches": [" Ink ", "GOLD", "ink"]},
                "Night => ink,gold",
            ),
            _case("quilt-hidden-empty", {"name": " Blank ", "swatches": []}, "Blank => "),
        ),
        "quiltlegend.legend",
        "render",
        retention_anchor_ordinal=1,
        transfer_relation_sha256=_RELATION_NORMALIZE_UNIQUE,
    ),
    _TaskSpec(
        "train-v9-seed-both-ready-22",
        (
            "Select the approved reserve when both seed stores are ready. The request and "
            "published records omit whether the first or second store has priority."
        ),
        _FALLBACK_POLICY,
        _INSUFFICIENT,
        _seed_fallback_repository(),
        STOP,
        (),
        None,
        None,
        ambiguity=(
            "both-ready-priority-is-unpublished-v9",
            _canonical_json(
                {
                    "first": "oak",
                    "second": "birch",
                    "first_ready": True,
                    "second_ready": True,
                }
            ).decode("ascii"),
            _canonical_json("oak").decode("ascii"),
            _canonical_json("birch").decode("ascii"),
        ),
    ),
)


def _build_repository(files: tuple[tuple[str, str], ...]) -> RepositorySnapshotV9:
    repository_files = tuple(
        RepositoryFileV9(
            relative_path=relative_path,
            content_utf8=content.encode("utf-8"),
            file_sha256=sha256(content.encode("utf-8")).hexdigest(),
        )
        for relative_path, content in files
    )
    provisional = RepositorySnapshotV9(files=repository_files, snapshot_sha256="")
    return RepositorySnapshotV9(
        files=repository_files,
        snapshot_sha256=_digest(b"continual-repository-snapshot-v9", provisional.record()),
    )


def _build_witnesses(
    records: tuple[tuple[str, str, str], ...],
) -> tuple[FrozenWitnessV9, ...]:
    witnesses = []
    for case_id, request_json, expected_json in records:
        request = _frozen_json(request_json)
        expected = _frozen_json(expected_json)
        provisional = FrozenWitnessV9(case_id, request, expected, "")
        witnesses.append(
            FrozenWitnessV9(
                case_id,
                request,
                expected,
                _digest(b"continual-repository-swe-witness-v9", provisional.record()),
            )
        )
    return tuple(witnesses)


def _build_review_plan(spec: _TaskSpec) -> ReviewPlanV9 | None:
    if spec.review_text is None:
        return None
    revised = _build_witnesses(spec.revised_witnesses)
    provisional = ReviewPlanV9(spec.review_text, revised, "")
    return ReviewPlanV9(
        spec.review_text,
        revised,
        _digest(b"continual-repository-swe-review-plan-v9", provisional.record()),
    )


def _build_ambiguity(spec: _TaskSpec) -> AmbiguityCertificateV9 | None:
    if spec.ambiguity is None:
        return None
    fingerprint, request_json, first_json, second_json = spec.ambiguity
    request = _frozen_json(request_json)
    outcomes = (_frozen_json(first_json), _frozen_json(second_json))
    provisional = AmbiguityCertificateV9(fingerprint, request, outcomes, "")
    return AmbiguityCertificateV9(
        fingerprint,
        request,
        outcomes,
        _digest(b"continual-repository-swe-ambiguity-v9", provisional.record()),
    )


def _build_evaluator(
    spec: _TaskSpec,
    witnesses: tuple[FrozenWitnessV9, ...],
    review_plan: ReviewPlanV9 | None,
) -> FrozenEvaluatorV9 | None:
    if spec.entry_module is None or spec.entry_point is None:
        if spec.entry_module is not None or spec.entry_point is not None or witnesses:
            raise ContinualToySweTaskV9Error("STOP task gained an evaluator")
        return None
    revised = None if review_plan is None else review_plan.revised_witnesses
    provisional = FrozenEvaluatorV9(
        spec.entry_module,
        spec.entry_point,
        witnesses,
        revised,
        "",
    )
    return FrozenEvaluatorV9(
        spec.entry_module,
        spec.entry_point,
        witnesses,
        revised,
        _digest(b"continual-repository-swe-evaluator-v9", provisional.record()),
    )


def _build_task(ordinal: int, spec: _TaskSpec) -> _EvaluatorTaskV9:
    repository = _build_repository(spec.files)
    witnesses = _build_witnesses(spec.witnesses)
    review_plan = _build_review_plan(spec)
    ambiguity = _build_ambiguity(spec)
    evaluator = _build_evaluator(spec, witnesses, review_plan)
    provisional_private = PrivateAcceptanceV9(
        family_id=spec.family_id,
        stage=spec.stage,
        expected_action=spec.expected_action,
        retention_anchor_ordinal=spec.retention_anchor_ordinal,
        transfer_relation_sha256=spec.transfer_relation_sha256,
        witnesses=witnesses,
        evaluator=evaluator,
        review_plan=review_plan,
        ambiguity_certificate=ambiguity,
        bundle_sha256="",
    )
    private = PrivateAcceptanceV9(
        family_id=spec.family_id,
        stage=spec.stage,
        expected_action=spec.expected_action,
        retention_anchor_ordinal=spec.retention_anchor_ordinal,
        transfer_relation_sha256=spec.transfer_relation_sha256,
        witnesses=witnesses,
        evaluator=evaluator,
        review_plan=review_plan,
        ambiguity_certificate=ambiguity,
        bundle_sha256=_digest(b"continual-repository-swe-private-v9", provisional_private.record()),
    )
    commitment = _digest(
        b"continual-repository-swe-task-v9",
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
    provisional_public = LearnerVisibleToySweObservationV9(
        task_commitment_sha256=commitment,
        instance_id=spec.instance_id,
        issue=spec.issue,
        repository_snapshot_sha256=repository.snapshot_sha256,
        tool_contract_sha256=TOOL_CONTRACT_SHA256,
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV9(
        task_commitment_sha256=commitment,
        instance_id=spec.instance_id,
        issue=spec.issue,
        repository_snapshot_sha256=repository.snapshot_sha256,
        tool_contract_sha256=TOOL_CONTRACT_SHA256,
        observation_sha256=_digest(
            b"continual-repository-swe-observation-v9", provisional_public.record()
        ),
    )
    return _EvaluatorTaskV9(ordinal, public, repository, private)


_EVALUATOR_STREAM = tuple(_build_task(ordinal, spec) for ordinal, spec in enumerate(_SPECS))
_PUBLIC_STREAM = tuple(task.public for task in _EVALUATOR_STREAM)

_EXPECTED_STAGES = (
    _DIRECT,
    _COMPOSITION,
    _DIRECT,
    _COMPOSITION,
    _REVISION,
    _DIRECT,
    _COMPOSITION,
    _INSUFFICIENT,
    _DIRECT,
    _REVISION,
    _COMPOSITION,
    _INSUFFICIENT,
)
_EXPECTED_FAMILIES = (
    _CREDIT_ARITHMETIC,
    _NORMALIZED_UNIQUENESS,
    _INCLUSIVE_BOUNDARY,
    _LATENCY_SELECTION,
    _MOISTURE_CALIBRATION,
    _TAIL_PRESERVATION,
    _CREDIT_ARITHMETIC,
    _TIE_POLICY,
    _INCLUSIVE_BOUNDARY,
    _AGE_POLICY,
    _NORMALIZED_UNIQUENESS,
    _FALLBACK_POLICY,
)


def _authenticate_evaluator_stream(stream: tuple[_EvaluatorTaskV9, ...]) -> None:
    if type(stream) is not tuple or len(stream) != TASK_COUNT:
        raise ContinualToySweTaskV9Error("evaluator stream size changed")
    if tuple(task.ordinal for task in stream) != tuple(range(TASK_COUNT)):
        raise ContinualToySweTaskV9Error("task order changed")
    if tuple(task.private.stage for task in stream) != _EXPECTED_STAGES:
        raise ContinualToySweTaskV9Error("stage progression changed")
    if tuple(task.private.family_id for task in stream) != _EXPECTED_FAMILIES:
        raise ContinualToySweTaskV9Error("repository family lineage changed")
    if Counter(task.private.stage for task in stream) != Counter(
        {_DIRECT: 4, _COMPOSITION: 4, _REVISION: 2, _INSUFFICIENT: 2}
    ):
        raise ContinualToySweTaskV9Error("task-kind balance changed")
    if tuple(
        (task.ordinal, task.private.retention_anchor_ordinal)
        for task in stream
        if task.private.retention_anchor_ordinal is not None
    ) != ((6, 0), (8, 2), (10, 1)):
        raise ContinualToySweTaskV9Error("delayed-transfer lineage changed")
    for task in stream:
        anchor = task.private.retention_anchor_ordinal
        if (
            anchor is not None
            and task.private.transfer_relation_sha256
            != stream[anchor].private.transfer_relation_sha256
        ):
            raise ContinualToySweTaskV9Error("delayed-transfer relation changed")
    if tuple(task.ordinal for task in stream if task.private.review_plan is not None) != (4, 9):
        raise ContinualToySweTaskV9Error("review-revision placement changed")
    if tuple(task.ordinal for task in stream if task.private.expected_action == STOP) != (7, 11):
        raise ContinualToySweTaskV9Error("evidence-limited pair changed")
    instance_ids = tuple(task.public.instance_id for task in stream)
    commitments = tuple(task.public.task_commitment_sha256 for task in stream)
    snapshots = tuple(task.repository.snapshot_sha256 for task in stream)
    if (
        len(set(instance_ids)) != TASK_COUNT
        or len(set(commitments)) != TASK_COUNT
        or len(set(snapshots)) != TASK_COUNT
    ):
        raise ContinualToySweTaskV9Error("tasks are not unique")
    for task in stream:
        task.authenticate()
        paths = tuple(file.relative_path for file in task.repository.files)
        public_tests = tuple(
            path for path in paths if path.startswith("tests/") and path.endswith(".py")
        )
        writable = tuple(
            path
            for path in paths
            if path.endswith(".py") and "/" in path and not path.startswith("tests/")
        )
        if not 7 <= len(paths) <= 11 or "README.md" not in paths or "pyproject.toml" not in paths:
            raise ContinualToySweTaskV9Error("repository shape changed")
        if len(public_tests) != 1 or len(writable) < 3:
            raise ContinualToySweTaskV9Error("repository evidence or source roles changed")
        if task.private.stage == _COMPOSITION and not any(
            path.startswith("docs/") for path in paths
        ):
            raise ContinualToySweTaskV9Error("composition task lost documentation evidence")


def _authenticate_public_stream(
    stream: tuple[LearnerVisibleToySweObservationV9, ...],
) -> None:
    if type(stream) is not tuple or len(stream) != TASK_COUNT:
        raise ContinualToySweTaskV9Error("public stream size changed")
    for public, evaluator_task in zip(stream, _EVALUATOR_STREAM, strict=True):
        if type(public) is not LearnerVisibleToySweObservationV9:
            raise TypeError("public stream contains a private task type")
        public.authenticate()
        if public != evaluator_task.public:
            raise ContinualToySweTaskV9Error("public stream order changed")


def _stream_root(stream: tuple[LearnerVisibleToySweObservationV9, ...]) -> str:
    _authenticate_public_stream(stream)
    return _digest(
        b"continual-repository-swe-stream-v9",
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


TRAIN_STREAM_CUSTODY_SHA256 = _digest(b"continual-repository-swe-custody-v9", _custody_record())


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
        raise ContinualToySweTaskV9Error("external custody manifest is unavailable") from error
    if manifest != _computed_manifest_record():
        raise ContinualToySweTaskV9Error("external custody manifest does not pin this source")


def preflight_execution_substrate_v9() -> dict[str, str]:
    """Authenticate frozen worker, launcher, and interpreter before reserving a cohort."""

    _authenticate_external_manifest()
    _authenticate_execution_substrate()
    return {
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }


def public_train_tasks_v9() -> tuple[LearnerVisibleToySweObservationV9, ...]:
    """Return the authenticated public TRAIN stream without evaluator material."""

    _authenticate_external_manifest()
    _authenticate_public_stream(_PUBLIC_STREAM)
    return tuple(_PUBLIC_STREAM)


def global_tool_contract_v9() -> dict[str, object]:
    """Return a detached copy of the one uniform action contract."""

    return json.loads(_canonical_json(_tool_contract_record()).decode("ascii"))


def _evaluator_task(task_commitment_sha256: str) -> _EvaluatorTaskV9:
    matches = tuple(
        task
        for task in _EVALUATOR_STREAM
        if task.public.task_commitment_sha256 == task_commitment_sha256
    )
    if len(matches) != 1:
        raise ContinualToySweTaskV9Error("unknown task commitment")
    task = matches[0]
    task.authenticate()
    return task


def runner_mount_allowlist_v9(task_commitment_sha256: str) -> tuple[str, ...]:
    """Return every learner-readable public repository path."""

    task = _evaluator_task(task_commitment_sha256)
    return tuple(file.relative_path for file in task.repository.files)


def runner_writable_source_allowlist_v9(task_commitment_sha256: str) -> tuple[str, ...]:
    """Return existing package sources writable through ``submit_patch``."""

    readable = runner_mount_allowlist_v9(task_commitment_sha256)
    writable = tuple(
        path
        for path in readable
        if path.endswith(".py") and "/" in path and not path.startswith("tests/")
    )
    if not writable:
        raise ContinualToySweTaskV9Error("repository has no writable package source")
    return writable


def learner_workspace_files_v9(task_commitment_sha256: str) -> dict[str, bytes]:
    """Materialize detached public workspace bytes and nothing evaluator-private."""

    task = _evaluator_task(task_commitment_sha256)
    allowlist = runner_mount_allowlist_v9(task_commitment_sha256)
    workspace = {file.relative_path: bytes(file.content_utf8) for file in task.repository.files}
    if tuple(workspace) != allowlist:
        raise ContinualToySweTaskV9Error("runner mount exceeded its allowlist")
    return workspace


def _validated_patch(task_commitment_sha256: str, payload: dict[str, str]) -> tuple[str, str]:
    if type(payload) is not dict or set(payload) != {"patch"}:
        raise ContinualToySweTaskV9Error(
            'submit_patch payload must be exactly {"patch": <unified diff>}'
        )
    patch = payload["patch"]
    if type(patch) is not str:
        raise ContinualToySweTaskV9Error("submit_patch.patch must be a string")
    if not patch or "\r" in patch or not patch.endswith("\n"):
        raise CandidatePatchRejectedV9("patch must be a canonical LF-terminated string")
    try:
        encoded = patch.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CandidatePatchRejectedV9("patch is not valid UTF-8") from error
    if len(encoded) > _EXECUTION_LIMITS.max_completion_bytes:
        raise CandidatePatchRejectedV9("patch exceeds the global byte budget")

    writable = set(runner_writable_source_allowlist_v9(task_commitment_sha256))
    lines = patch.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts or starts[0] != 0:
        raise CandidatePatchRejectedV9("patch is not a unified Git diff")
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
            raise CandidatePatchRejectedV9("patch path header is not canonical")
        old_name, new_name = header[2], header[3]
        if not old_name.startswith("a/") or not new_name.startswith("b/"):
            raise CandidatePatchRejectedV9("patch paths must use a/ and b/ prefixes")
        path = old_name[2:]
        if path != new_name[2:] or path not in writable or path in touched:
            raise CandidatePatchRejectedV9("patch path is outside the writable allowlist")
        canonical = PurePosixPath(path)
        if canonical.is_absolute() or ".." in canonical.parts or canonical.as_posix() != path:
            raise CandidatePatchRejectedV9("patch path is not canonical")
        if any(line.startswith(forbidden_headers) for line in block):
            raise CandidatePatchRejectedV9("patch changes repository shape or file mode")
        if block.count(f"--- a/{path}") != 1 or block.count(f"+++ b/{path}") != 1:
            raise CandidatePatchRejectedV9("patch file headers changed")
        if not any(line.startswith("@@ ") for line in block):
            raise CandidatePatchRejectedV9("patch has no textual hunk")
        touched.append(path)
    return patch, sha256(encoded).hexdigest()


def _authenticate_execution_substrate() -> tuple[bytes, Path]:
    try:
        worker_source = _REPOSITORY_WORKER_PATH.read_bytes()
        sandbox_source = _SANDBOX_SOURCE_PATH.read_bytes()
    except OSError as error:
        raise ContinualToySweTaskV9Error("frozen evaluator substrate is unavailable") from error
    if sha256(worker_source).hexdigest() != REPOSITORY_WORKER_SHA256:
        raise ContinualToySweTaskV9Error("repository worker differs from its freeze")
    if sha256(sandbox_source).hexdigest() != SIMPLE_SWE_SANDBOX_SHA256:
        raise ContinualToySweTaskV9Error("sandbox launcher differs from its freeze")
    executable = resolve_python_executable()
    if sha256(executable.read_bytes()).hexdigest() != PYTHON_EXECUTABLE_SHA256:
        raise ContinualToySweTaskV9Error("sandbox Python executable differs from its freeze")
    return worker_source, executable


def _module_name(relative_path: str) -> str:
    parts = PurePosixPath(relative_path).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _run_private_evaluator(
    task: _EvaluatorTaskV9,
    patch: str,
    patch_sha256: str,
    *,
    revised: bool,
) -> bool:
    workspace = learner_workspace_files_v9(task.public.task_commitment_sha256)
    with tempfile.TemporaryDirectory(prefix="continual-swe-v9-") as root_name:
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
            raise CandidatePatchRejectedV9(f"submitted patch does not apply cleanly: {detail}")
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
            raise CandidatePatchRejectedV9("submitted patch could not be applied")

        evaluator = task.private.evaluator
        if evaluator is None:
            raise ContinualToySweTaskV9Error("task has no executable evaluator")
        cases = evaluator.revised_cases if revised else evaluator.initial_cases
        if cases is None:
            raise ContinualToySweTaskV9Error("revised evaluator cases are unavailable")
        writable = runner_writable_source_allowlist_v9(task.public.task_commitment_sha256)
        modules = [
            {
                "name": _module_name(relative_path),
                "source": (root / PurePosixPath(relative_path)).read_text(encoding="utf-8"),
            }
            for relative_path in writable
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
            raise ContinualToySweTaskV9Error(
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
    action: str,
    action_sha256: str,
    evaluator_sha256: str,
    sandbox_evaluator_reached: bool,
    conclusion: str,
    feedback: str,
) -> ActionReceiptV9:
    provisional = ActionReceiptV9(
        task_commitment_sha256,
        sequence,
        prior_receipt_sha256,
        action,
        action_sha256,
        evaluator_sha256,
        sandbox_evaluator_reached,
        conclusion,
        feedback,
        "",
    )
    receipt = ActionReceiptV9(
        task_commitment_sha256,
        sequence,
        prior_receipt_sha256,
        action,
        action_sha256,
        evaluator_sha256,
        sandbox_evaluator_reached,
        conclusion,
        feedback,
        _digest(b"continual-repository-swe-action-receipt-v9", provisional.record()),
    )
    receipt.authenticate()
    return receipt


def seal_action_summary_v9(
    task_commitment_sha256: str,
    attempt: int,
    action: str,
    receipt: ActionReceiptV9 | None = None,
) -> SealedActionSummaryV9:
    """Seal one receipt, neutral STOP, or unreceipted invalid/local rejection."""

    if receipt is not None:
        conclusion = receipt.conclusion
    elif action == STOP:
        conclusion = _NEUTRAL
    else:
        conclusion = _FAILURE
    provisional = SealedActionSummaryV9(
        task_commitment_sha256,
        attempt,
        action,
        conclusion,
        receipt,
        "",
    )
    summary = SealedActionSummaryV9(
        task_commitment_sha256,
        attempt,
        action,
        conclusion,
        receipt,
        _digest(b"continual-repository-swe-sealed-action-v9", provisional.record()),
    )
    summary.authenticate()
    return summary


def submit_patch_v9(
    task_commitment_sha256: str,
    payload: dict[str, str],
    prior_receipt: ActionReceiptV9 | None = None,
) -> ActionReceiptV9:
    """Apply one complete diff and run hidden cases in the frozen repository worker."""

    task = _evaluator_task(task_commitment_sha256)
    patch, patch_sha256 = _validated_patch(task_commitment_sha256, payload)
    evaluator = task.private.evaluator
    if task.private.expected_action == STOP:
        certificate = task.private.ambiguity_certificate
        if certificate is None:
            raise ContinualToySweTaskV9Error("STOP acceptance certificate is unavailable")
        stop_evaluator_sha256 = _digest(
            b"continual-repository-swe-stop-acceptance-v9",
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
                or prior_receipt.evaluator_sha256 != stop_evaluator_sha256
                or prior_receipt.sequence >= MAX_PATCH_SUBMISSIONS
            ):
                raise ContinualToySweTaskV9Error("prior receipt cannot authorize this submission")
            sequence = prior_receipt.sequence + 1
            prior_sha256 = prior_receipt.receipt_sha256
        return _make_receipt(
            task_commitment_sha256=task_commitment_sha256,
            sequence=sequence,
            prior_receipt_sha256=prior_sha256,
            action=SUBMIT_PATCH,
            action_sha256=patch_sha256,
            evaluator_sha256=stop_evaluator_sha256,
            sandbox_evaluator_reached=False,
            conclusion=_FAILURE,
            feedback="Patch rejected by the committed acceptance protocol.",
        )
    if evaluator is None:
        raise ContinualToySweTaskV9Error("task has no executable evaluator")

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
            raise ContinualToySweTaskV9Error("prior receipt cannot authorize this submission")
        initial_sha256 = evaluator.phase_sha256(revised=False)
        revised_sha256 = (
            None if evaluator.revised_cases is None else evaluator.phase_sha256(revised=True)
        )
        if prior_receipt.evaluator_sha256 not in {initial_sha256, revised_sha256}:
            raise ContinualToySweTaskV9Error("prior receipt evaluator changed")
        if prior_receipt.conclusion == _ACTION_REQUIRED:
            plan = task.private.review_plan
            if (
                plan is None
                or revised_sha256 is None
                or prior_receipt.evaluator_sha256 != initial_sha256
                or prior_receipt.feedback != plan.review_text
                or prior_receipt.action_sha256 == patch_sha256
            ):
                raise ContinualToySweTaskV9Error("receipt does not authorize a distinct revision")
            revised = True
        else:
            revised = prior_receipt.evaluator_sha256 == revised_sha256
        sequence = prior_receipt.sequence + 1
        prior_sha256 = prior_receipt.receipt_sha256

    evaluator_sha256 = evaluator.phase_sha256(revised=revised)
    passed = _run_private_evaluator(
        task,
        patch,
        patch_sha256,
        revised=revised,
    )
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
        action=SUBMIT_PATCH,
        action_sha256=patch_sha256,
        evaluator_sha256=evaluator_sha256,
        sandbox_evaluator_reached=True,
        conclusion=conclusion,
        feedback=feedback,
    )


def train_stream_sha256_v9() -> str:
    """Return the authenticated logical TRAIN stream commitment."""

    _authenticate_external_manifest()
    return _stream_root(_PUBLIC_STREAM)


def train_stream_custody_v9() -> dict[str, object]:
    """Return detached source, prior, contract, and stream custody commitments."""

    _authenticate_external_manifest()
    record = _custody_record()
    record["train_stream_custody_sha256"] = TRAIN_STREAM_CUSTODY_SHA256
    return json.loads(_canonical_json(record).decode("ascii"))


def posthoc_score_terminal_actions_v9(
    task_commitment_sha256: str,
    completed_actions: tuple[SealedActionSummaryV9, ...],
    *,
    all_arm_streams_finished: bool,
) -> dict[str, object]:
    """Score sealed actions only after every experimental arm stream has finished."""

    if all_arm_streams_finished is not True:
        raise ContinualToySweTaskV9Error("posthoc scoring is forbidden before arm completion")
    if type(completed_actions) is not tuple:
        raise ContinualToySweTaskV9Error("posthoc scoring requires a sealed action tuple")
    if len(completed_actions) > MAX_PATCH_SUBMISSIONS:
        raise ContinualToySweTaskV9Error("sealed action sequence exceeds the global budget")
    task = _evaluator_task(task_commitment_sha256)
    action_names: list[str] = []
    prior_receipt_sha256 = _RECEIPT_GENESIS_SHA256
    expected_receipt_sequence = 1
    terminal_seen = False
    for attempt, summary in enumerate(completed_actions, start=1):
        if type(summary) is not SealedActionSummaryV9 or terminal_seen:
            raise ContinualToySweTaskV9Error("sealed action sequence changed")
        summary.authenticate()
        if summary.task_commitment_sha256 != task_commitment_sha256 or summary.attempt != attempt:
            raise ContinualToySweTaskV9Error("sealed action order changed")
        action_names.append(summary.action)
        if summary.receipt is not None:
            receipt = summary.receipt
            if (
                receipt.sequence != expected_receipt_sequence
                or receipt.prior_receipt_sha256 != prior_receipt_sha256
            ):
                raise ContinualToySweTaskV9Error("sealed receipt chain changed")
            prior_receipt_sha256 = receipt.receipt_sha256
            expected_receipt_sequence += 1
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


def _materialized_witnesses_v9(
    task_commitment_sha256: str, *, revised: bool = False
) -> tuple[dict[str, object], ...]:
    """Evaluator-only detached witness material used by fixture tests."""

    task = _evaluator_task(task_commitment_sha256)
    if revised:
        if task.private.review_plan is None:
            raise ContinualToySweTaskV9Error("task has no revised witness set")
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
    "ActionReceiptV9",
    "CandidatePatchRejectedV9",
    "ContinualToySweTaskV9Error",
    "LearnerVisibleToySweObservationV9",
    "SealedActionSummaryV9",
    "global_tool_contract_v9",
    "learner_workspace_files_v9",
    "posthoc_score_terminal_actions_v9",
    "preflight_execution_substrate_v9",
    "public_train_tasks_v9",
    "runner_mount_allowlist_v9",
    "runner_writable_source_allowlist_v9",
    "seal_action_summary_v9",
    "submit_patch_v9",
    "train_stream_custody_v9",
    "train_stream_sha256_v9",
]
