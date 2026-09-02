"""Unspent public-TRAIN repository tasks for integrated continual coding V8.

V8 contains twelve deterministic repository snapshots across three repeated
families. Learners receive only an opaque task commitment, an issue, a snapshot
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

SCHEMA = "continual-repository-swe-tasks-v8"
MANIFEST_SCHEMA = "continual-repository-swe-custody-manifest-v8"
PARTITION = "train"
TASK_COUNT = 12
# Immutable logical identity of the public_coding_prior_v1 corpus used by V1-V7.
# It is source-prior custody, not a prompt transform, bootstrap, or runner API hash.
SOURCE_PRIOR_CORPUS_SHA256 = "81a60454b9af6b8592ea8b422c25bdd98115c101e5580c8cd571555fdf0abfb1"
SOURCE_PRIOR_CASE_COUNT = 8
MANIFEST_FILENAME = "continual_toy_swe_tasks_v8_manifest.json"

SUBMIT_PATCH = "submit_patch"
STOP = "<stop>"
INVALID_ACTION = "invalid"
GLOBAL_ACTIONS = (SUBMIT_PATCH,)
MAX_PATCH_SUBMISSIONS = 3

_ACQUISITION = "repository-family-acquisition"
_COMPOSITION = "cross-module-composition"
_REVISION = "action-bound-review-revision"
_RETENTION = "delayed-structural-transfer"
_INSUFFICIENT = "evidence-limited-stop"
_STAGES = (_ACQUISITION, _COMPOSITION, _REVISION, _RETENTION, _INSUFFICIENT)

_CATALOG = "catalog-normalization"
_ROUTING = "route-registry"
_TELEMETRY = "telemetry-calibration"
_FAMILIES = (_CATALOG, _ROUTING, _TELEMETRY)

_SUCCESS = "success"
_FAILURE = "failure"
_ACTION_REQUIRED = "action-required"
_NEUTRAL = "neutral"
_RECEIPT_CONCLUSIONS = (_SUCCESS, _FAILURE, _ACTION_REQUIRED)

_RELATION_CATALOG = "89b641979f7c779038da876e36cad20b8a763016b97f9b112a1794f3250eff46"
_RELATION_TELEMETRY = "087863950cefe0ec8ca3e27ed25d50bd34eef9085174bf629c93eb6be6b2077c"
_RECEIPT_GENESIS_SHA256 = "76c11ebc3d256187bdf08fb83db572a8fb44689250f687a01f41ec1d4871979e"
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


class ContinualToySweTaskV8Error(ValueError):
    """A V8 task, receipt, repository, or custody record is unauthenticated."""


class CandidatePatchRejectedV8(ContinualToySweTaskV8Error):
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
        "schema": "continual-repository-swe-tool-contract-v8",
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
    b"continual-repository-swe-tool-contract-v8", _tool_contract_record()
)


@dataclass(frozen=True, slots=True)
class RepositoryFileV8:
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
            raise ContinualToySweTaskV8Error("repository path is not canonical")
        if type(self.content_utf8) is not bytes:
            raise TypeError("repository content must be immutable bytes")
        if sha256(self.content_utf8).hexdigest() != self.file_sha256:
            raise ContinualToySweTaskV8Error("repository file changed")


@dataclass(frozen=True, slots=True)
class RepositorySnapshotV8:
    files: tuple[RepositoryFileV8, ...]
    snapshot_sha256: str

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "files": tuple(file.record() for file in self.files)}

    def authenticate(self) -> None:
        if not isinstance(self.files, tuple) or not self.files:
            raise ContinualToySweTaskV8Error("repository snapshot is empty or mutable")
        for file in self.files:
            if type(file) is not RepositoryFileV8:
                raise TypeError("repository snapshot must contain exact immutable files")
            file.authenticate()
        paths = tuple(file.relative_path for file in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ContinualToySweTaskV8Error("repository paths changed")
        expected = _digest(b"continual-repository-snapshot-v8", self.record())
        if self.snapshot_sha256 != expected:
            raise ContinualToySweTaskV8Error("repository snapshot changed")


@dataclass(frozen=True, slots=True)
class LearnerVisibleToySweObservationV8:
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
            raise ContinualToySweTaskV8Error("learner observation has an invalid commitment")
        if not self.instance_id or not self.issue:
            raise ContinualToySweTaskV8Error("learner observation is incomplete")
        if self.tool_contract_sha256 != TOOL_CONTRACT_SHA256:
            raise ContinualToySweTaskV8Error("task-specific action contract is forbidden")
        expected = _digest(b"continual-repository-swe-observation-v8", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskV8Error("learner observation changed")


@dataclass(frozen=True, slots=True)
class FrozenWitnessV8:
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
            raise ContinualToySweTaskV8Error("witness request is not canonical JSON")
        if _frozen_json(self.expected_json.decode("ascii")) != self.expected_json:
            raise ContinualToySweTaskV8Error("witness expectation is not canonical JSON")
        expected = _digest(b"continual-repository-swe-witness-v8", self.record())
        if self.witness_sha256 != expected:
            raise ContinualToySweTaskV8Error("acceptance witness changed")


@dataclass(frozen=True, slots=True)
class FrozenEvaluatorV8:
    entry_module: str
    entry_point: str
    initial_cases: tuple[FrozenWitnessV8, ...]
    revised_cases: tuple[FrozenWitnessV8, ...] | None
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
            raise ContinualToySweTaskV8Error("revised evaluator cases are unavailable")
        return _digest(
            b"continual-repository-swe-evaluator-phase-v8",
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
            raise ContinualToySweTaskV8Error("private evaluator entry point changed")
        if type(self.initial_cases) is not tuple or len(self.initial_cases) < 2:
            raise ContinualToySweTaskV8Error("private evaluator lacks frozen cases")
        for cases in (self.initial_cases, self.revised_cases):
            if cases is not None:
                if type(cases) is not tuple or len(cases) < 2:
                    raise ContinualToySweTaskV8Error("private evaluator cases changed")
                for case in cases:
                    if type(case) is not FrozenWitnessV8:
                        raise TypeError("private evaluator case type changed")
                    case.authenticate()
        expected = _digest(b"continual-repository-swe-evaluator-v8", self.record())
        if self.evaluator_sha256 != expected:
            raise ContinualToySweTaskV8Error("private evaluator changed")


@dataclass(frozen=True, slots=True)
class ReviewPlanV8:
    review_text: str
    revised_witnesses: tuple[FrozenWitnessV8, ...]
    plan_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "review_text": self.review_text,
            "revised_witness_sha256s": tuple(
                witness.witness_sha256 for witness in self.revised_witnesses
            ),
        }

    def authenticate(self, initial_witnesses: tuple[FrozenWitnessV8, ...]) -> None:
        if not self.review_text or len(self.revised_witnesses) < 2:
            raise ContinualToySweTaskV8Error("review plan lacks public evidence")
        for witness in self.revised_witnesses:
            witness.authenticate()
        initial = {witness.request_json: witness.expected_json for witness in initial_witnesses}
        revised = {
            witness.request_json: witness.expected_json for witness in self.revised_witnesses
        }
        shared = initial.keys() & revised.keys()
        if not shared or not any(initial[key] != revised[key] for key in shared):
            raise ContinualToySweTaskV8Error("review does not require a behavioral revision")
        expected = _digest(b"continual-repository-swe-review-plan-v8", self.record())
        if self.plan_sha256 != expected:
            raise ContinualToySweTaskV8Error("review plan changed")


@dataclass(frozen=True, slots=True)
class AmbiguityCertificateV8:
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
            raise ContinualToySweTaskV8Error("ambiguity evidence is empty")
        if _frozen_json(self.witness_request_json.decode("ascii")) != self.witness_request_json:
            raise ContinualToySweTaskV8Error("ambiguity request is not canonical")
        if len(set(self.compatible_outcome_jsons)) != 2:
            raise ContinualToySweTaskV8Error("ambiguity outcomes do not disagree")
        for outcome in self.compatible_outcome_jsons:
            if _frozen_json(outcome.decode("ascii")) != outcome:
                raise ContinualToySweTaskV8Error("ambiguity outcome is not canonical")
        expected = _digest(b"continual-repository-swe-ambiguity-v8", self.record())
        if self.certificate_sha256 != expected:
            raise ContinualToySweTaskV8Error("ambiguity certificate changed")


@dataclass(frozen=True, slots=True)
class PrivateAcceptanceV8:
    family_id: str
    stage: str
    expected_action: str
    retention_anchor_ordinal: int | None
    transfer_relation_sha256: str | None
    witnesses: tuple[FrozenWitnessV8, ...]
    evaluator: FrozenEvaluatorV8 | None
    review_plan: ReviewPlanV8 | None
    ambiguity_certificate: AmbiguityCertificateV8 | None
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
            raise ContinualToySweTaskV8Error("private family or stage changed")
        if self.expected_action not in (SUBMIT_PATCH, STOP):
            raise ContinualToySweTaskV8Error("private terminal action changed")
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
                raise ContinualToySweTaskV8Error("STOP task is not evidence-limited")
            self.ambiguity_certificate.authenticate()
        else:
            if (
                not self.witnesses
                or self.evaluator is None
                or self.ambiguity_certificate is not None
            ):
                raise ContinualToySweTaskV8Error("executable task acceptance changed")
            self.evaluator.authenticate()
            if self.evaluator.initial_cases != self.witnesses:
                raise ContinualToySweTaskV8Error("initial evaluator cases changed")
            if self.stage == _REVISION:
                if self.review_plan is None or self.evaluator.revised_cases is None:
                    raise ContinualToySweTaskV8Error("revision task lost its review plan")
                self.review_plan.authenticate(self.witnesses)
                if self.evaluator.revised_cases != self.review_plan.revised_witnesses:
                    raise ContinualToySweTaskV8Error("revised evaluator cases changed")
            elif self.review_plan is not None or self.evaluator.revised_cases is not None:
                raise ContinualToySweTaskV8Error("non-revision task gained future review text")
        if self.stage == _RETENTION:
            if (
                type(self.retention_anchor_ordinal) is not int
                or not 0 <= self.retention_anchor_ordinal < ordinal
                or not _is_sha256(self.transfer_relation_sha256)
            ):
                raise ContinualToySweTaskV8Error("retention lineage changed")
        elif self.retention_anchor_ordinal is not None:
            raise ContinualToySweTaskV8Error("non-retention task gained an anchor")
        elif self.transfer_relation_sha256 is not None and self.stage != _ACQUISITION:
            raise ContinualToySweTaskV8Error("non-anchor task gained a transfer relation")
        expected = _digest(b"continual-repository-swe-private-v8", self.record())
        if self.bundle_sha256 != expected:
            raise ContinualToySweTaskV8Error("private acceptance bundle changed")


@dataclass(frozen=True, slots=True)
class _EvaluatorTaskV8:
    ordinal: int
    public: LearnerVisibleToySweObservationV8
    repository: RepositorySnapshotV8
    private: PrivateAcceptanceV8

    def authenticate(self) -> None:
        self.public.authenticate()
        self.repository.authenticate()
        self.private.authenticate(self.ordinal)
        expected_commitment = _digest(
            b"continual-repository-swe-task-v8",
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
            raise ContinualToySweTaskV8Error("public task is not bound to private acceptance")
        if self.public.repository_snapshot_sha256 != self.repository.snapshot_sha256:
            raise ContinualToySweTaskV8Error("public task is not bound to its repository")


@dataclass(frozen=True, slots=True)
class ActionReceiptV8:
    """Action-bound public feedback produced after a learner action."""

    task_commitment_sha256: str
    sequence: int
    prior_receipt_sha256: str
    action: str
    action_sha256: str
    evaluator_sha256: str
    conclusion: str
    feedback: str
    receipt_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": "continual-repository-swe-action-receipt-v8",
            "task_commitment_sha256": self.task_commitment_sha256,
            "sequence": self.sequence,
            "prior_receipt_sha256": self.prior_receipt_sha256,
            "action": self.action,
            "action_sha256": self.action_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "conclusion": self.conclusion,
            "feedback": self.feedback,
        }

    def authenticate(self) -> None:
        if (
            not _is_sha256(self.task_commitment_sha256)
            or not _is_sha256(self.prior_receipt_sha256)
            or not _is_sha256(self.action_sha256)
            or not _is_sha256(self.evaluator_sha256)
            or self.action != SUBMIT_PATCH
            or self.conclusion not in _RECEIPT_CONCLUSIONS
            or type(self.sequence) is not int
            or not 1 <= self.sequence <= MAX_PATCH_SUBMISSIONS
            or not self.feedback
        ):
            raise ContinualToySweTaskV8Error("action receipt fields changed")
        expected = _digest(b"continual-repository-swe-action-receipt-v8", self.record())
        if self.receipt_sha256 != expected:
            raise ContinualToySweTaskV8Error("action receipt changed")


@dataclass(frozen=True, slots=True)
class SealedActionSummaryV8:
    """Immutable action-kind summary for posthoc scoring after all arms finish."""

    task_commitment_sha256: str
    attempt: int
    action: str
    conclusion: str
    receipt: ActionReceiptV8 | None
    summary_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": "continual-repository-swe-sealed-action-v8",
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
            raise ContinualToySweTaskV8Error("sealed action fields changed")
        if self.receipt is None:
            expected_conclusion = _NEUTRAL if self.action == STOP else _FAILURE
            if self.conclusion != expected_conclusion:
                raise ContinualToySweTaskV8Error("unreceipted action conclusion changed")
        else:
            if type(self.receipt) is not ActionReceiptV8 or self.action != SUBMIT_PATCH:
                raise ContinualToySweTaskV8Error("sealed receipt action changed")
            self.receipt.authenticate()
            if (
                self.receipt.task_commitment_sha256 != self.task_commitment_sha256
                or self.conclusion != self.receipt.conclusion
            ):
                raise ContinualToySweTaskV8Error("sealed receipt conclusion changed")
        expected = _digest(b"continual-repository-swe-sealed-action-v8", self.record())
        if self.summary_sha256 != expected:
            raise ContinualToySweTaskV8Error("sealed action summary changed")


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


def _repository(
    package: str,
    readme: str,
    modules: tuple[tuple[str, str], ...],
    public_test: str,
) -> tuple[tuple[str, str], ...]:
    files = [
        ("README.md", _source(readme)),
        ("pyproject.toml", _PYPROJECT),
        (f"{package}/__init__.py", ""),
        *((f"{package}/{name}.py", _source(source)) for name, source in modules),
        ("tests/test_public.py", _source(public_test)),
    ]
    return tuple(sorted(files))


def _catalog_acquisition_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "catalog",
        """# Catalog index — Archive Key R17

        Archive Key R17 compares catalog labels after removing surrounding
        whitespace and ignoring letter case. It never changes the stored display
        label. Catalog records arrive as `sku|display label`.
        """,
        (
            (
                "codec",
                """
                def decode(line: str) -> tuple[str, str]:
                    sku, label = line.split("|", 1)
                    return sku, label
                """,
            ),
            (
                "keys",
                """
                def comparison_key(value: str) -> str:
                    return value.strip()
                """,
            ),
            (
                "index",
                """
                from catalog.codec import decode
                from catalog.keys import comparison_key


                def catalog_snapshot(lines: list[str], query: str) -> dict:
                    by_key = {}
                    for line in lines:
                        sku, label = decode(line)
                        by_key[comparison_key(label)] = (sku, label)
                    record = by_key.get(comparison_key(query))
                    return {
                        "display_labels": sorted(label for _, label in by_key.values()),
                        "matched_sku": None if record is None else record[0],
                    }
                """,
            ),
        ),
        """
        from catalog.index import catalog_snapshot


        def test_padding_is_ignored() -> None:
            result = catalog_snapshot(["sku-1|  Alpha  "], "Alpha")
            assert result == {
                "display_labels": ["  Alpha  "],
                "matched_sku": "sku-1",
            }
        """,
    )


def _catalog_composition_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "catalog",
        """# Partner alias index

        Partner feeds wrap aliases in one pair of boundary markers. The boundary
        markers are transport syntax and are not part of the searchable alias.
        """,
        (
            (
                "codec",
                """
                def split_record(line: str) -> tuple[str, str]:
                    record_id, alias = line.split("=", 1)
                    return record_id, alias
                """,
            ),
            (
                "keys",
                """
                def searchable_alias(value: str) -> str:
                    return value.strip().lower()
                """,
            ),
            (
                "index",
                """
                from catalog.codec import split_record
                from catalog.keys import searchable_alias


                def build_alias_index(lines: list[str]) -> dict[str, str]:
                    result = {}
                    for line in lines:
                        record_id, alias = split_record(line)
                        result[searchable_alias(alias)] = record_id
                    return result


                def resolve_alias(lines: list[str], query: str) -> str | None:
                    return build_alias_index(lines).get(searchable_alias(query))
                """,
            ),
        ),
        """
        from catalog.index import resolve_alias


        def test_bare_query_finds_wrapped_alias() -> None:
            assert resolve_alias(["r-1=[  Main Door  ]"], "main door") == "r-1"
        """,
    )


def _catalog_revision_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "catalog",
        """# Alias registry

        Display labels are retained verbatim. Alias matching currently remains
        case-sensitive.
        """,
        (
            (
                "codec",
                """
                def unpack(line: str) -> tuple[str, str]:
                    alias_id, label = line.split(":", 1)
                    return alias_id, label
                """,
            ),
            (
                "identity",
                """
                def alias_identity(label: str) -> str:
                    return label
                """,
            ),
            (
                "registry",
                """
                from catalog.codec import unpack
                from catalog.identity import alias_identity


                def registry(lines: list[str]) -> dict[str, str]:
                    return {
                        alias_identity(label): alias_id
                        for alias_id, label in map(unpack, lines)
                    }


                def lookup(entries: dict[str, str], label: str) -> str | None:
                    return entries.get(alias_identity(label))


                def find_alias(lines: list[str], query: str) -> str | None:
                    return lookup(registry(lines), query)
                """,
            ),
        ),
        """
        from catalog.registry import find_alias


        def test_outer_padding_does_not_change_alias_identity() -> None:
            assert find_alias(["a-1: Alpha "], "Alpha") == "a-1"
        """,
    )


def _catalog_retention_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "directory",
        """# Contact directory

        Imported names keep their original spelling for display. Directory search
        must use the previously established Archive Key R17 convention.
        """,
        (
            (
                "records",
                """
                def read_record(raw: str) -> tuple[str, str]:
                    identifier, name = raw.split(";", 1)
                    return identifier, name
                """,
            ),
            (
                "identity",
                """
                def comparison_identity(name: str) -> str:
                    return name.strip()
                """,
            ),
            (
                "service",
                """
                from directory.identity import comparison_identity
                from directory.records import read_record


                def identifier_for(lines: list[str], query: str) -> str | None:
                    pairs = (read_record(raw) for raw in lines)
                    identifiers = {
                        comparison_identity(display): identifier
                        for identifier, display in pairs
                    }
                    return identifiers.get(comparison_identity(query))
                """,
            ),
        ),
        """
        from directory.service import identifier_for


        def test_search_uses_comparison_identity() -> None:
            assert identifier_for(["u-7;  Ada Lovelace "], " Ada Lovelace ") == "u-7"
        """,
    )


def _routing_acquisition_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "routing",
        """# Route registry

        Route rows contain a route id and integer destination. The configured
        fallback is used only when the requested route is absent.
        """,
        (
            (
                "rows",
                """
                def decode(row: str) -> tuple[str, int]:
                    route_id, destination = row.split("=", 1)
                    return route_id, int(destination)
                """,
            ),
            (
                "registry",
                """
                from routing.rows import decode


                def find_destination(rows: list[str], route_id: str, fallback: int) -> int:
                    decoded = [decode(row) for row in rows]
                    if decoded:
                        return decoded[0][1]
                    return fallback
                """,
            ),
            (
                "service",
                """
                from routing.registry import find_destination


                def destination_for(rows: list[str], requested: str, fallback: int) -> int:
                    return find_destination(rows, requested, fallback)
                """,
            ),
        ),
        """
        from routing.service import destination_for


        def test_requested_route_is_selected() -> None:
            rows = ["north=4", "west=9"]
            assert destination_for(rows, "west", -1) == 9
            assert destination_for(rows, "east", -1) == -1
        """,
    )


def _routing_composition_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "routing",
        """# Eligible route report

        Deployment configuration selects the ranked-interior policy. Eligible
        routes must be enabled and healthy.
        """,
        (
            ("config", 'ACTIVE_POLICY = "ranked-interior"\n'),
            (
                "sets",
                """
                def shared(left: list[int], right: list[int]) -> list[int]:
                    return sorted(set(left) & set(right))


                def without_boundaries(values: list[int]) -> list[int]:
                    return values[1:-1]
                """,
            ),
            (
                "service",
                """
                from routing.config import ACTIVE_POLICY
                from routing.sets import shared, without_boundaries


                def eligible_routes(enabled: list[int], healthy: list[int]) -> list[int]:
                    if ACTIVE_POLICY != "ranked-interior":
                        return []
                    return shared(without_boundaries(enabled), without_boundaries(healthy))
                """,
            ),
        ),
        """
        from routing.service import eligible_routes


        def test_policy_omits_boundaries_of_shared_set() -> None:
            assert eligible_routes([9, 1, 2, 3, 8], [7, 1, 2, 3, 6]) == [2]
        """,
    )


def _routing_revision_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "routing",
        """# Deployment route choice

        A request may carry an explicit route. Regions and the deployment also
        provide fallback routes.
        """,
        (
            (
                "config",
                'REGION_ROUTE = {"eu": "eu-central", "us": "us-east"}\nDEFAULT_ROUTE = "global"\n',
            ),
            (
                "request",
                """
                def clean_route(route: str | None) -> str | None:
                    if route is None:
                        return None
                    cleaned = route.strip()
                    return cleaned or None
                """,
            ),
            (
                "service",
                """
                from routing.config import DEFAULT_ROUTE, REGION_ROUTE
                from routing.request import clean_route


                def choose_route(explicit: str | None, region: str) -> str:
                    return REGION_ROUTE.get(region, DEFAULT_ROUTE)
                """,
            ),
        ),
        """
        from routing.service import choose_route


        def test_explicit_route_precedes_fallbacks() -> None:
            assert choose_route(" private ", "apac") == "private"
            assert choose_route(None, "eu") == "eu-central"
        """,
    )


def _routing_stop_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "routing",
        """# Deployment tie policies

        Deployments select one of the registered tie policies. This checkout is a
        reusable library and contains the policy implementations and report API.
        """,
        (
            (
                "policies",
                """
                def stable(routes: list[str]) -> list[str]:
                    return list(routes)


                def ranked(routes: list[str]) -> list[str]:
                    return sorted(routes)


                POLICIES = {"stable": stable, "ranked": ranked}
                """,
            ),
            (
                "registry",
                """
                from routing.policies import POLICIES


                def policy(name: str):
                    return POLICIES[name]
                """,
            ),
            (
                "service",
                """
                from routing.registry import policy


                def deployment_report(routes: list[str], policy_name: str) -> list[str]:
                    return policy(policy_name)(routes)
                """,
            ),
        ),
        """
        from routing.policies import ranked, stable
        from routing.service import deployment_report


        def test_registered_policies() -> None:
            routes = ["west", "east"]
            assert stable(routes) == ["west", "east"]
            assert ranked(routes) == ["east", "west"]
            assert deployment_report(routes, "stable") == ["west", "east"]
            assert deployment_report(routes, "ranked") == ["east", "west"]
        """,
    )


def _telemetry_acquisition_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "telemetry",
        """# Sample deviation — Beacon Transform Q4

        Beacon Transform Q4 profiles the captured sample first, then measures its
        distance from the unchanged reference value.
        """,
        (
            (
                "calibration",
                """
                def calibrate(sample: int, gain: int, bias: int) -> int:
                    return sample * gain + bias
                """,
            ),
            (
                "metrics",
                """
                def distance(left: int, right: int) -> int:
                    return abs(left - right)
                """,
            ),
            (
                "service",
                """
                from telemetry.calibration import calibrate
                from telemetry.metrics import distance


                def sample_deviation(sample: int, reference: int, gain: int, bias: int) -> int:
                    return calibrate(distance(sample, reference), gain, bias)
                """,
            ),
        ),
        """
        from telemetry.service import sample_deviation


        def test_calibration_precedes_comparison() -> None:
            assert sample_deviation(2, 5, 3, -1) == 0
        """,
    )


def _telemetry_composition_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "telemetry",
        """# Deployment telemetry summary

        The active profile calibrates every reading before the summary selects
        calibrated even values and multiplies their nonzero values.
        """,
        (
            ("config", "ACTIVE_PROFILE = (2, 1)\n"),
            (
                "pipeline",
                """
                def calibrate(values: list[int], gain: int, bias: int) -> list[int]:
                    return [value * gain + bias for value in values]


                def even(values: list[int]) -> list[int]:
                    return [value for value in values if value % 2 == 0]
                """,
            ),
            (
                "summary",
                """
                from telemetry.config import ACTIVE_PROFILE
                from telemetry.pipeline import calibrate, even


                def deployment_product(
                    readings: list[int], profile: list[int] | tuple[int, int] | None = None
                ) -> int:
                    gain, bias = ACTIVE_PROFILE if profile is None else profile
                    result = 1
                    for value in calibrate(even(readings), gain, bias):
                        if value != 0:
                            result *= value
                    return result
                """,
            ),
        ),
        """
        from telemetry.summary import deployment_product


        def test_even_selection_uses_calibrated_values() -> None:
            assert deployment_product([1, 2, 3]) == 1
        """,
    )


def _telemetry_retention_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "signals",
        """# Signal offset report

        Signal offset reports must use the previously established Beacon Transform
        Q4 convention.
        """,
        (
            (
                "profile",
                """
                def apply_profile(captured: int, scale: int, offset: int) -> int:
                    return captured * scale + offset
                """,
            ),
            (
                "measure",
                """
                def separation(value: int, baseline: int) -> int:
                    return abs(value - baseline)
                """,
            ),
            (
                "report",
                """
                from signals.measure import separation
                from signals.profile import apply_profile


                def offset_report(captured: int, baseline: int, scale: int, offset: int) -> int:
                    return apply_profile(separation(captured, baseline), scale, offset)
                """,
            ),
        ),
        """
        from signals.report import offset_report


        def test_profiled_value_is_compared_with_baseline() -> None:
            assert offset_report(2, 5, 1, 0) == 3
        """,
    )


def _telemetry_stop_repository() -> tuple[tuple[str, str], ...]:
    return _repository(
        "telemetry",
        """# Calibration profiles

        The reusable package contains the supported calibration profiles and the
        export service used by deployments.
        """,
        (
            (
                "profiles",
                """
                PROFILES = {"lab": (2, 1), "field": (-1, 4)}


                def apply(sample: int, profile: str) -> int:
                    gain, bias = PROFILES[profile]
                    return sample * gain + bias
                """,
            ),
            (
                "registry",
                """
                from telemetry.profiles import PROFILES


                def available_profiles() -> tuple[str, ...]:
                    return tuple(sorted(PROFILES))
                """,
            ),
            (
                "service",
                """
                from telemetry.profiles import apply


                def export_sample(sample: int, profile: str) -> int:
                    return apply(sample, profile)
                """,
            ),
        ),
        """
        from telemetry.profiles import apply
        from telemetry.service import export_sample


        def test_profile_implementations() -> None:
            assert apply(3, "lab") == 7
            assert apply(3, "field") == 1
            assert export_sample(3, "lab") == 7
            assert export_sample(3, "field") == 1
        """,
    )


_SPECS = (
    _TaskSpec(
        instance_id="train-93e4c2a18f",
        issue=(
            "Adopt Archive Key R17: compare labels without surrounding whitespace or "
            "letter case while preserving stored display labels verbatim."
        ),
        family_id=_CATALOG,
        stage=_ACQUISITION,
        files=_catalog_acquisition_repository(),
        expected_action=SUBMIT_PATCH,
        transfer_relation_sha256=_RELATION_CATALOG,
        witnesses=(
            (
                "catalog-acquisition-mixed-case",
                '{"lines":["sku-1|  Alpha  "],"query":" alpha "}',
                '{"display_labels":["  Alpha  "],"matched_sku":"sku-1"}',
            ),
            (
                "catalog-acquisition-uppercase-query",
                '{"lines":["sku-2|Beta"],"query":"BETA"}',
                '{"display_labels":["Beta"],"matched_sku":"sku-2"}',
            ),
        ),
        entry_module="catalog.index",
        entry_point="catalog_snapshot",
    ),
    _TaskSpec(
        instance_id="train-b17d05f49a",
        issue=(
            "Destination lookup must return the requested route and use the configured "
            "fallback only when that route is absent."
        ),
        family_id=_ROUTING,
        stage=_ACQUISITION,
        files=_routing_acquisition_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "routing-acquisition-present",
                '{"fallback":-1,"requested":"west","rows":["north=4","west=9"]}',
                "9",
            ),
            (
                "routing-acquisition-absent",
                '{"fallback":12,"requested":"east","rows":["north=4","west=9"]}',
                "12",
            ),
        ),
        entry_module="routing.service",
        entry_point="destination_for",
    ),
    _TaskSpec(
        instance_id="train-64a8d921ce",
        issue=(
            "Adopt Beacon Transform Q4: profile the captured sample before measuring "
            "distance from the unchanged reference value."
        ),
        family_id=_TELEMETRY,
        stage=_ACQUISITION,
        files=_telemetry_acquisition_repository(),
        expected_action=SUBMIT_PATCH,
        transfer_relation_sha256=_RELATION_TELEMETRY,
        witnesses=(
            (
                "telemetry-acquisition-negative-gain",
                '{"bias":4,"gain":-2,"reference":1,"sample":3}',
                "3",
            ),
            (
                "telemetry-acquisition-wide-gap",
                '{"bias":1,"gain":3,"reference":7,"sample":-2}',
                "12",
            ),
        ),
        entry_module="telemetry.service",
        entry_point="sample_deviation",
    ),
    _TaskSpec(
        instance_id="train-f2397ac541",
        issue=(
            "Partner aliases wrapped in one pair of transport boundary markers must "
            "remain searchable by the bare alias, ignoring padding and letter case."
        ),
        family_id=_CATALOG,
        stage=_COMPOSITION,
        files=_catalog_composition_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "catalog-composition-square-boundary",
                '{"lines":["r-1=[  Main Door  ]"],"query":"main door"}',
                '"r-1"',
            ),
            (
                "catalog-composition-angle-boundary",
                '{"lines":["r-2=<North Wing>"],"query":" NORTH WING "}',
                '"r-2"',
            ),
        ),
        entry_module="catalog.index",
        entry_point="resolve_alias",
    ),
    _TaskSpec(
        instance_id="train-0d681ae734",
        issue=(
            "The ranked-interior deployment report must remove the outer routes from "
            "the shared eligible set, not from each input before eligibility is known."
        ),
        family_id=_ROUTING,
        stage=_COMPOSITION,
        files=_routing_composition_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "routing-composition-shared-boundaries",
                '{"enabled":[9,1,2,3,8],"healthy":[7,1,2,3,6]}',
                "[2]",
            ),
            (
                "routing-composition-no-interior",
                '{"enabled":[8,2,5],"healthy":[2,5,7]}',
                "[]",
            ),
        ),
        entry_module="routing.service",
        entry_point="eligible_routes",
    ),
    _TaskSpec(
        instance_id="train-c87ef41352",
        issue=(
            "The deployment summary must calibrate every reading before selecting "
            "even values and multiplying its nonzero selection."
        ),
        family_id=_TELEMETRY,
        stage=_COMPOSITION,
        files=_telemetry_composition_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "telemetry-composition-all-odd",
                '{"readings":[1,2,3]}',
                "1",
            ),
            (
                "telemetry-composition-custom-profile",
                '{"profile":[1,1],"readings":[1,2,3]}',
                "8",
            ),
        ),
        entry_module="telemetry.summary",
        entry_point="deployment_product",
    ),
    _TaskSpec(
        instance_id="train-5ba93618dc",
        issue=(
            "Alias lookup should ignore surrounding whitespace while retaining the "
            "display label exactly as received."
        ),
        family_id=_CATALOG,
        stage=_REVISION,
        files=_catalog_revision_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "catalog-revision-shared",
                '{"lines":["a-1: Alpha "],"query":"alpha"}',
                "null",
            ),
            (
                "catalog-revision-padding",
                '{"lines":["a-2: Beta "],"query":" Beta "}',
                '"a-2"',
            ),
        ),
        entry_module="catalog.registry",
        entry_point="find_alias",
        review_text=(
            "Review correction: alias matching must be case-insensitive as well as "
            "whitespace-tolerant. Continue to retain display labels verbatim."
        ),
        revised_witnesses=(
            (
                "catalog-revision-shared",
                '{"lines":["a-1: Alpha "],"query":"alpha"}',
                '"a-1"',
            ),
            (
                "catalog-revision-uppercase",
                '{"lines":["a-3: Gamma "],"query":"GAMMA"}',
                '"a-3"',
            ),
        ),
    ),
    _TaskSpec(
        instance_id="train-2e40c17bad",
        issue=(
            "A cleaned explicit route should take precedence over regional and global "
            "fallback routes."
        ),
        family_id=_ROUTING,
        stage=_REVISION,
        files=_routing_revision_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "routing-revision-shared",
                '{"explicit":" private ","region":"eu"}',
                '"private"',
            ),
            (
                "routing-revision-fallback",
                '{"explicit":null,"region":"eu"}',
                '"eu-central"',
            ),
        ),
        entry_module="routing.service",
        entry_point="choose_route",
        review_text=(
            "Review correction: the regional deployment route must override an explicit "
            "request. Use the explicit route only when the region has no configured route."
        ),
        revised_witnesses=(
            (
                "routing-revision-shared",
                '{"explicit":" private ","region":"eu"}',
                '"eu-central"',
            ),
            (
                "routing-revision-unconfigured",
                '{"explicit":" private ","region":"apac"}',
                '"private"',
            ),
        ),
    ),
    _TaskSpec(
        instance_id="train-a8137e0c64",
        issue="Make the signal offset report conform to Beacon Transform Q4.",
        family_id=_TELEMETRY,
        stage=_RETENTION,
        files=_telemetry_retention_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "telemetry-retention-zero-gap",
                '{"baseline":5,"captured":2,"offset":-1,"scale":3}',
                "0",
            ),
            (
                "telemetry-retention-negative-scale",
                '{"baseline":2,"captured":4,"offset":1,"scale":-2}',
                "9",
            ),
        ),
        entry_module="signals.report",
        entry_point="offset_report",
        retention_anchor_ordinal=2,
        transfer_relation_sha256=_RELATION_TELEMETRY,
    ),
    _TaskSpec(
        instance_id="train-71c04db9ef",
        issue="Make directory lookup conform to Archive Key R17.",
        family_id=_CATALOG,
        stage=_RETENTION,
        files=_catalog_retention_repository(),
        expected_action=SUBMIT_PATCH,
        witnesses=(
            (
                "catalog-retention-mixed-case",
                '{"lines":["d-1;  South Hall  "],"query":"south hall"}',
                '"d-1"',
            ),
            (
                "catalog-retention-uppercase",
                '{"lines":["d-2;Atrium"],"query":" ATRIUM "}',
                '"d-2"',
            ),
        ),
        entry_module="directory.service",
        entry_point="identifier_for",
        retention_anchor_ordinal=0,
        transfer_relation_sha256=_RELATION_CATALOG,
    ),
    _TaskSpec(
        instance_id="train-d98f260b13",
        issue="Make the deployment route report apply the active tie policy to its routes.",
        family_id=_ROUTING,
        stage=_INSUFFICIENT,
        files=_routing_stop_repository(),
        expected_action=STOP,
        witnesses=(),
        entry_module=None,
        entry_point=None,
        ambiguity=(
            "no deployment policy selection appears in the repository snapshot",
            '{"routes":["west","east"]}',
            '["west","east"]',
            '["east","west"]',
        ),
    ),
    _TaskSpec(
        instance_id="train-418fc7e25d",
        issue="Make exported samples use the current deployment calibration profile.",
        family_id=_TELEMETRY,
        stage=_INSUFFICIENT,
        files=_telemetry_stop_repository(),
        expected_action=STOP,
        witnesses=(),
        entry_module=None,
        entry_point=None,
        ambiguity=(
            "no current deployment profile selection appears in the repository snapshot",
            '{"sample":3}',
            "7",
            "1",
        ),
    ),
)


def _build_repository(files: tuple[tuple[str, str], ...]) -> RepositorySnapshotV8:
    repository_files = tuple(
        RepositoryFileV8(
            relative_path=relative_path,
            content_utf8=content.encode("utf-8"),
            file_sha256=sha256(content.encode("utf-8")).hexdigest(),
        )
        for relative_path, content in files
    )
    provisional = RepositorySnapshotV8(files=repository_files, snapshot_sha256="")
    return RepositorySnapshotV8(
        files=repository_files,
        snapshot_sha256=_digest(b"continual-repository-snapshot-v8", provisional.record()),
    )


def _build_witnesses(
    records: tuple[tuple[str, str, str], ...],
) -> tuple[FrozenWitnessV8, ...]:
    witnesses = []
    for case_id, request_json, expected_json in records:
        request = _frozen_json(request_json)
        expected = _frozen_json(expected_json)
        provisional = FrozenWitnessV8(case_id, request, expected, "")
        witnesses.append(
            FrozenWitnessV8(
                case_id,
                request,
                expected,
                _digest(b"continual-repository-swe-witness-v8", provisional.record()),
            )
        )
    return tuple(witnesses)


def _build_review_plan(spec: _TaskSpec) -> ReviewPlanV8 | None:
    if spec.review_text is None:
        return None
    revised = _build_witnesses(spec.revised_witnesses)
    provisional = ReviewPlanV8(spec.review_text, revised, "")
    return ReviewPlanV8(
        spec.review_text,
        revised,
        _digest(b"continual-repository-swe-review-plan-v8", provisional.record()),
    )


def _build_ambiguity(spec: _TaskSpec) -> AmbiguityCertificateV8 | None:
    if spec.ambiguity is None:
        return None
    fingerprint, request_json, first_json, second_json = spec.ambiguity
    request = _frozen_json(request_json)
    outcomes = (_frozen_json(first_json), _frozen_json(second_json))
    provisional = AmbiguityCertificateV8(fingerprint, request, outcomes, "")
    return AmbiguityCertificateV8(
        fingerprint,
        request,
        outcomes,
        _digest(b"continual-repository-swe-ambiguity-v8", provisional.record()),
    )


def _build_evaluator(
    spec: _TaskSpec,
    witnesses: tuple[FrozenWitnessV8, ...],
    review_plan: ReviewPlanV8 | None,
) -> FrozenEvaluatorV8 | None:
    if spec.entry_module is None or spec.entry_point is None:
        if spec.entry_module is not None or spec.entry_point is not None or witnesses:
            raise ContinualToySweTaskV8Error("STOP task gained an evaluator")
        return None
    revised = None if review_plan is None else review_plan.revised_witnesses
    provisional = FrozenEvaluatorV8(
        spec.entry_module,
        spec.entry_point,
        witnesses,
        revised,
        "",
    )
    return FrozenEvaluatorV8(
        spec.entry_module,
        spec.entry_point,
        witnesses,
        revised,
        _digest(b"continual-repository-swe-evaluator-v8", provisional.record()),
    )


def _build_task(ordinal: int, spec: _TaskSpec) -> _EvaluatorTaskV8:
    repository = _build_repository(spec.files)
    witnesses = _build_witnesses(spec.witnesses)
    review_plan = _build_review_plan(spec)
    ambiguity = _build_ambiguity(spec)
    evaluator = _build_evaluator(spec, witnesses, review_plan)
    provisional_private = PrivateAcceptanceV8(
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
    private = PrivateAcceptanceV8(
        family_id=spec.family_id,
        stage=spec.stage,
        expected_action=spec.expected_action,
        retention_anchor_ordinal=spec.retention_anchor_ordinal,
        transfer_relation_sha256=spec.transfer_relation_sha256,
        witnesses=witnesses,
        evaluator=evaluator,
        review_plan=review_plan,
        ambiguity_certificate=ambiguity,
        bundle_sha256=_digest(b"continual-repository-swe-private-v8", provisional_private.record()),
    )
    commitment = _digest(
        b"continual-repository-swe-task-v8",
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
    provisional_public = LearnerVisibleToySweObservationV8(
        task_commitment_sha256=commitment,
        instance_id=spec.instance_id,
        issue=spec.issue,
        repository_snapshot_sha256=repository.snapshot_sha256,
        tool_contract_sha256=TOOL_CONTRACT_SHA256,
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV8(
        task_commitment_sha256=commitment,
        instance_id=spec.instance_id,
        issue=spec.issue,
        repository_snapshot_sha256=repository.snapshot_sha256,
        tool_contract_sha256=TOOL_CONTRACT_SHA256,
        observation_sha256=_digest(
            b"continual-repository-swe-observation-v8", provisional_public.record()
        ),
    )
    return _EvaluatorTaskV8(ordinal, public, repository, private)


_EVALUATOR_STREAM = tuple(_build_task(ordinal, spec) for ordinal, spec in enumerate(_SPECS))
_PUBLIC_STREAM = tuple(task.public for task in _EVALUATOR_STREAM)

_EXPECTED_STAGES = (
    _ACQUISITION,
    _ACQUISITION,
    _ACQUISITION,
    _COMPOSITION,
    _COMPOSITION,
    _COMPOSITION,
    _REVISION,
    _REVISION,
    _RETENTION,
    _RETENTION,
    _INSUFFICIENT,
    _INSUFFICIENT,
)
_EXPECTED_FAMILIES = (
    _CATALOG,
    _ROUTING,
    _TELEMETRY,
    _CATALOG,
    _ROUTING,
    _TELEMETRY,
    _CATALOG,
    _ROUTING,
    _TELEMETRY,
    _CATALOG,
    _ROUTING,
    _TELEMETRY,
)


def _authenticate_evaluator_stream(stream: tuple[_EvaluatorTaskV8, ...]) -> None:
    if type(stream) is not tuple or len(stream) != TASK_COUNT:
        raise ContinualToySweTaskV8Error("evaluator stream size changed")
    if tuple(task.ordinal for task in stream) != tuple(range(TASK_COUNT)):
        raise ContinualToySweTaskV8Error("task order changed")
    if tuple(task.private.stage for task in stream) != _EXPECTED_STAGES:
        raise ContinualToySweTaskV8Error("stage progression changed")
    if tuple(task.private.family_id for task in stream) != _EXPECTED_FAMILIES:
        raise ContinualToySweTaskV8Error("repository family lineage changed")
    if Counter(task.private.family_id for task in stream) != Counter(
        {_CATALOG: 4, _ROUTING: 4, _TELEMETRY: 4}
    ):
        raise ContinualToySweTaskV8Error("repository family balance changed")
    if tuple(
        (task.ordinal, task.private.retention_anchor_ordinal)
        for task in stream
        if task.private.stage == _RETENTION
    ) != ((8, 2), (9, 0)):
        raise ContinualToySweTaskV8Error("delayed-transfer lineage changed")
    for task in stream:
        anchor = task.private.retention_anchor_ordinal
        if (
            anchor is not None
            and task.private.transfer_relation_sha256
            != stream[anchor].private.transfer_relation_sha256
        ):
            raise ContinualToySweTaskV8Error("delayed-transfer relation changed")
    if tuple(task.ordinal for task in stream if task.private.review_plan is not None) != (6, 7):
        raise ContinualToySweTaskV8Error("review-revision placement changed")
    if tuple(task.ordinal for task in stream if task.private.expected_action == STOP) != (10, 11):
        raise ContinualToySweTaskV8Error("evidence-limited pair changed")
    instance_ids = tuple(task.public.instance_id for task in stream)
    commitments = tuple(task.public.task_commitment_sha256 for task in stream)
    snapshots = tuple(task.repository.snapshot_sha256 for task in stream)
    if (
        len(set(instance_ids)) != TASK_COUNT
        or len(set(commitments)) != TASK_COUNT
        or len(set(snapshots)) != TASK_COUNT
    ):
        raise ContinualToySweTaskV8Error("tasks are not unique")
    for task in stream:
        task.authenticate()
        paths = tuple(file.relative_path for file in task.repository.files)
        if (
            len(paths) != 7
            or "README.md" not in paths
            or "pyproject.toml" not in paths
            or "tests/test_public.py" not in paths
        ):
            raise ContinualToySweTaskV8Error("repository shape changed")


def _authenticate_public_stream(
    stream: tuple[LearnerVisibleToySweObservationV8, ...],
) -> None:
    if type(stream) is not tuple or len(stream) != TASK_COUNT:
        raise ContinualToySweTaskV8Error("public stream size changed")
    for public, evaluator_task in zip(stream, _EVALUATOR_STREAM, strict=True):
        if type(public) is not LearnerVisibleToySweObservationV8:
            raise TypeError("public stream contains a private task type")
        public.authenticate()
        if public != evaluator_task.public:
            raise ContinualToySweTaskV8Error("public stream order changed")


def _stream_root(stream: tuple[LearnerVisibleToySweObservationV8, ...]) -> str:
    _authenticate_public_stream(stream)
    return _digest(
        b"continual-repository-swe-stream-v8",
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


TRAIN_STREAM_CUSTODY_SHA256 = _digest(b"continual-repository-swe-custody-v8", _custody_record())


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
        raise ContinualToySweTaskV8Error("external custody manifest is unavailable") from error
    if manifest != _computed_manifest_record():
        raise ContinualToySweTaskV8Error("external custody manifest does not pin this source")


def preflight_execution_substrate_v8() -> dict[str, str]:
    """Authenticate frozen worker, launcher, and interpreter before reserving a cohort."""

    _authenticate_external_manifest()
    _authenticate_execution_substrate()
    return {
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }


def public_train_tasks_v8() -> tuple[LearnerVisibleToySweObservationV8, ...]:
    """Return the authenticated public TRAIN stream without evaluator material."""

    _authenticate_external_manifest()
    _authenticate_public_stream(_PUBLIC_STREAM)
    return tuple(_PUBLIC_STREAM)


def global_tool_contract_v8() -> dict[str, object]:
    """Return a detached copy of the one uniform action contract."""

    return json.loads(_canonical_json(_tool_contract_record()).decode("ascii"))


def _evaluator_task(task_commitment_sha256: str) -> _EvaluatorTaskV8:
    matches = tuple(
        task
        for task in _EVALUATOR_STREAM
        if task.public.task_commitment_sha256 == task_commitment_sha256
    )
    if len(matches) != 1:
        raise ContinualToySweTaskV8Error("unknown task commitment")
    task = matches[0]
    task.authenticate()
    return task


def runner_mount_allowlist_v8(task_commitment_sha256: str) -> tuple[str, ...]:
    """Return every learner-readable public repository path."""

    task = _evaluator_task(task_commitment_sha256)
    return tuple(file.relative_path for file in task.repository.files)


def runner_writable_source_allowlist_v8(task_commitment_sha256: str) -> tuple[str, ...]:
    """Return existing package sources writable through ``submit_patch``."""

    readable = runner_mount_allowlist_v8(task_commitment_sha256)
    writable = tuple(
        path
        for path in readable
        if path.endswith(".py") and "/" in path and not path.startswith("tests/")
    )
    if not writable:
        raise ContinualToySweTaskV8Error("repository has no writable package source")
    return writable


def learner_workspace_files_v8(task_commitment_sha256: str) -> dict[str, bytes]:
    """Materialize detached public workspace bytes and nothing evaluator-private."""

    task = _evaluator_task(task_commitment_sha256)
    allowlist = runner_mount_allowlist_v8(task_commitment_sha256)
    workspace = {file.relative_path: bytes(file.content_utf8) for file in task.repository.files}
    if tuple(workspace) != allowlist:
        raise ContinualToySweTaskV8Error("runner mount exceeded its allowlist")
    return workspace


def _validated_patch(task_commitment_sha256: str, payload: dict[str, str]) -> tuple[str, str]:
    if type(payload) is not dict or set(payload) != {"patch"}:
        raise ContinualToySweTaskV8Error(
            'submit_patch payload must be exactly {"patch": <unified diff>}'
        )
    patch = payload["patch"]
    if type(patch) is not str:
        raise ContinualToySweTaskV8Error("submit_patch.patch must be a string")
    if not patch or "\r" in patch or not patch.endswith("\n"):
        raise CandidatePatchRejectedV8("patch must be a canonical LF-terminated string")
    try:
        encoded = patch.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CandidatePatchRejectedV8("patch is not valid UTF-8") from error
    if len(encoded) > _EXECUTION_LIMITS.max_completion_bytes:
        raise CandidatePatchRejectedV8("patch exceeds the global byte budget")

    writable = set(runner_writable_source_allowlist_v8(task_commitment_sha256))
    lines = patch.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts or starts[0] != 0:
        raise CandidatePatchRejectedV8("patch is not a unified Git diff")
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
            raise CandidatePatchRejectedV8("patch path header is not canonical")
        old_name, new_name = header[2], header[3]
        if not old_name.startswith("a/") or not new_name.startswith("b/"):
            raise CandidatePatchRejectedV8("patch paths must use a/ and b/ prefixes")
        path = old_name[2:]
        if path != new_name[2:] or path not in writable or path in touched:
            raise CandidatePatchRejectedV8("patch path is outside the writable allowlist")
        canonical = PurePosixPath(path)
        if canonical.is_absolute() or ".." in canonical.parts or canonical.as_posix() != path:
            raise CandidatePatchRejectedV8("patch path is not canonical")
        if any(line.startswith(forbidden_headers) for line in block):
            raise CandidatePatchRejectedV8("patch changes repository shape or file mode")
        if block.count(f"--- a/{path}") != 1 or block.count(f"+++ b/{path}") != 1:
            raise CandidatePatchRejectedV8("patch file headers changed")
        if not any(line.startswith("@@ ") for line in block):
            raise CandidatePatchRejectedV8("patch has no textual hunk")
        touched.append(path)
    return patch, sha256(encoded).hexdigest()


def _authenticate_execution_substrate() -> tuple[bytes, Path]:
    try:
        worker_source = _REPOSITORY_WORKER_PATH.read_bytes()
        sandbox_source = _SANDBOX_SOURCE_PATH.read_bytes()
    except OSError as error:
        raise ContinualToySweTaskV8Error("frozen evaluator substrate is unavailable") from error
    if sha256(worker_source).hexdigest() != REPOSITORY_WORKER_SHA256:
        raise ContinualToySweTaskV8Error("repository worker differs from its freeze")
    if sha256(sandbox_source).hexdigest() != SIMPLE_SWE_SANDBOX_SHA256:
        raise ContinualToySweTaskV8Error("sandbox launcher differs from its freeze")
    executable = resolve_python_executable()
    if sha256(executable.read_bytes()).hexdigest() != PYTHON_EXECUTABLE_SHA256:
        raise ContinualToySweTaskV8Error("sandbox Python executable differs from its freeze")
    return worker_source, executable


def _module_name(relative_path: str) -> str:
    parts = PurePosixPath(relative_path).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _run_private_evaluator(
    task: _EvaluatorTaskV8,
    patch: str,
    patch_sha256: str,
    *,
    revised: bool,
) -> bool:
    workspace = learner_workspace_files_v8(task.public.task_commitment_sha256)
    with tempfile.TemporaryDirectory(prefix="continual-swe-v8-") as root_name:
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
            raise CandidatePatchRejectedV8(f"submitted patch does not apply cleanly: {detail}")
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
            raise CandidatePatchRejectedV8("submitted patch could not be applied")

        evaluator = task.private.evaluator
        if evaluator is None:
            raise ContinualToySweTaskV8Error("task has no executable evaluator")
        cases = evaluator.revised_cases if revised else evaluator.initial_cases
        if cases is None:
            raise ContinualToySweTaskV8Error("revised evaluator cases are unavailable")
        writable = runner_writable_source_allowlist_v8(task.public.task_commitment_sha256)
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
            raise ContinualToySweTaskV8Error(
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
    conclusion: str,
    feedback: str,
) -> ActionReceiptV8:
    provisional = ActionReceiptV8(
        task_commitment_sha256,
        sequence,
        prior_receipt_sha256,
        action,
        action_sha256,
        evaluator_sha256,
        conclusion,
        feedback,
        "",
    )
    receipt = ActionReceiptV8(
        task_commitment_sha256,
        sequence,
        prior_receipt_sha256,
        action,
        action_sha256,
        evaluator_sha256,
        conclusion,
        feedback,
        _digest(b"continual-repository-swe-action-receipt-v8", provisional.record()),
    )
    receipt.authenticate()
    return receipt


def seal_action_summary_v8(
    task_commitment_sha256: str,
    attempt: int,
    action: str,
    receipt: ActionReceiptV8 | None = None,
) -> SealedActionSummaryV8:
    """Seal one receipt, neutral STOP, or unreceipted invalid/local rejection."""

    if receipt is not None:
        conclusion = receipt.conclusion
    elif action == STOP:
        conclusion = _NEUTRAL
    else:
        conclusion = _FAILURE
    provisional = SealedActionSummaryV8(
        task_commitment_sha256,
        attempt,
        action,
        conclusion,
        receipt,
        "",
    )
    summary = SealedActionSummaryV8(
        task_commitment_sha256,
        attempt,
        action,
        conclusion,
        receipt,
        _digest(b"continual-repository-swe-sealed-action-v8", provisional.record()),
    )
    summary.authenticate()
    return summary


def submit_patch_v8(
    task_commitment_sha256: str,
    payload: dict[str, str],
    prior_receipt: ActionReceiptV8 | None = None,
) -> ActionReceiptV8:
    """Apply one complete diff and run hidden cases in the frozen repository worker."""

    task = _evaluator_task(task_commitment_sha256)
    patch, patch_sha256 = _validated_patch(task_commitment_sha256, payload)
    evaluator = task.private.evaluator
    if task.private.expected_action == STOP:
        certificate = task.private.ambiguity_certificate
        if certificate is None:
            raise ContinualToySweTaskV8Error("STOP acceptance certificate is unavailable")
        stop_evaluator_sha256 = _digest(
            b"continual-repository-swe-stop-acceptance-v8",
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
                raise ContinualToySweTaskV8Error("prior receipt cannot authorize this submission")
            sequence = prior_receipt.sequence + 1
            prior_sha256 = prior_receipt.receipt_sha256
        return _make_receipt(
            task_commitment_sha256=task_commitment_sha256,
            sequence=sequence,
            prior_receipt_sha256=prior_sha256,
            action=SUBMIT_PATCH,
            action_sha256=patch_sha256,
            evaluator_sha256=stop_evaluator_sha256,
            conclusion=_FAILURE,
            feedback="Patch rejected by the committed acceptance protocol.",
        )
    if evaluator is None:
        raise ContinualToySweTaskV8Error("task has no executable evaluator")

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
            raise ContinualToySweTaskV8Error("prior receipt cannot authorize this submission")
        initial_sha256 = evaluator.phase_sha256(revised=False)
        revised_sha256 = (
            None if evaluator.revised_cases is None else evaluator.phase_sha256(revised=True)
        )
        if prior_receipt.evaluator_sha256 not in {initial_sha256, revised_sha256}:
            raise ContinualToySweTaskV8Error("prior receipt evaluator changed")
        if prior_receipt.conclusion == _ACTION_REQUIRED:
            plan = task.private.review_plan
            if (
                plan is None
                or revised_sha256 is None
                or prior_receipt.evaluator_sha256 != initial_sha256
                or prior_receipt.feedback != plan.review_text
                or prior_receipt.action_sha256 == patch_sha256
            ):
                raise ContinualToySweTaskV8Error("receipt does not authorize a distinct revision")
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
        conclusion=conclusion,
        feedback=feedback,
    )


def train_stream_sha256_v8() -> str:
    """Return the authenticated logical TRAIN stream commitment."""

    _authenticate_external_manifest()
    return _stream_root(_PUBLIC_STREAM)


def train_stream_custody_v8() -> dict[str, object]:
    """Return detached source, prior, contract, and stream custody commitments."""

    _authenticate_external_manifest()
    record = _custody_record()
    record["train_stream_custody_sha256"] = TRAIN_STREAM_CUSTODY_SHA256
    return json.loads(_canonical_json(record).decode("ascii"))


def posthoc_score_terminal_actions_v8(
    task_commitment_sha256: str,
    completed_actions: tuple[SealedActionSummaryV8, ...],
    *,
    all_arm_streams_finished: bool,
) -> dict[str, object]:
    """Score sealed actions only after every experimental arm stream has finished."""

    if all_arm_streams_finished is not True:
        raise ContinualToySweTaskV8Error("posthoc scoring is forbidden before arm completion")
    if type(completed_actions) is not tuple:
        raise ContinualToySweTaskV8Error("posthoc scoring requires a sealed action tuple")
    if len(completed_actions) > MAX_PATCH_SUBMISSIONS:
        raise ContinualToySweTaskV8Error("sealed action sequence exceeds the global budget")
    task = _evaluator_task(task_commitment_sha256)
    action_names: list[str] = []
    prior_receipt_sha256 = _RECEIPT_GENESIS_SHA256
    expected_receipt_sequence = 1
    terminal_seen = False
    for attempt, summary in enumerate(completed_actions, start=1):
        if type(summary) is not SealedActionSummaryV8 or terminal_seen:
            raise ContinualToySweTaskV8Error("sealed action sequence changed")
        summary.authenticate()
        if summary.task_commitment_sha256 != task_commitment_sha256 or summary.attempt != attempt:
            raise ContinualToySweTaskV8Error("sealed action order changed")
        action_names.append(summary.action)
        if summary.receipt is not None:
            receipt = summary.receipt
            if (
                receipt.sequence != expected_receipt_sequence
                or receipt.prior_receipt_sha256 != prior_receipt_sha256
            ):
                raise ContinualToySweTaskV8Error("sealed receipt chain changed")
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


def _materialized_witnesses_v8(
    task_commitment_sha256: str, *, revised: bool = False
) -> tuple[dict[str, object], ...]:
    """Evaluator-only detached witness material used by fixture tests."""

    task = _evaluator_task(task_commitment_sha256)
    if revised:
        if task.private.review_plan is None:
            raise ContinualToySweTaskV8Error("task has no revised witness set")
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
    "ActionReceiptV8",
    "CandidatePatchRejectedV8",
    "ContinualToySweTaskV8Error",
    "LearnerVisibleToySweObservationV8",
    "SealedActionSummaryV8",
    "global_tool_contract_v8",
    "learner_workspace_files_v8",
    "posthoc_score_terminal_actions_v8",
    "preflight_execution_substrate_v8",
    "public_train_tasks_v8",
    "runner_mount_allowlist_v8",
    "runner_writable_source_allowlist_v8",
    "seal_action_summary_v8",
    "submit_patch_v8",
    "train_stream_custody_v8",
    "train_stream_sha256_v8",
]
