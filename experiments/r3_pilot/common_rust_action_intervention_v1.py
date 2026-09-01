"""Neutral TRAIN-only custody and execution of prebound Rust actions.

The authority owns a small opaque action catalogue and accepts only explicit
candidate and intervention roots.  It never accepts or retains task text,
expected outputs, or a memory checkpoint.  Its public catalogue and selection
API exposes only roots; the authority necessarily retains Rust source in the
Python process for execution, so this is API opacity rather than process secrecy.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from secrets import token_bytes
from threading import Lock

from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "humanbrain.rust-action-intervention.v1"
SPLIT = "TRAIN"
CLAIM_BOUNDARY = (
    "train-only-neutral-custody-and-one-shot-execution-of-candidate-provided-"
    "bounded-ascii-rust-actions-not-task-understanding-source-synthesis-"
    "expected-output-access-memory-state-transfer-or-intelligence"
)
MIN_ACTIONS = 2
MAX_ACTIONS = 4
MAX_ISSUED_SELECTIONS = 4_096

_BINDING_NAMESPACE = b"rust-action-intervention-binding-v1"
_CATALOGUE_NAMESPACE = b"rust-action-intervention-catalogue-v1"
_AUTHORITY_NAMESPACE = b"rust-action-intervention-authority-v1"
_SELECTION_NAMESPACE = b"rust-action-intervention-selection-v1"
_RELEASE_NAMESPACE = b"rust-action-intervention-release-v1"
_CONSTRUCTION_TOKEN = object()
_SHA256_HEX = frozenset("0123456789abcdef")


class RustActionInterventionError(ValueError):
    """An intervention catalogue, selection, or release failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


@dataclass(frozen=True, slots=True)
class _RustActionBindingV1:
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1
    rust_source_sha256: str
    rust_source: str
    sha256: str

    def authenticate(self) -> None:
        if type(self.candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
            raise TypeError("intervention bindings require exact candidate projections")
        try:
            self.candidate.authenticate()
            source_bytes = self.rust_source.encode("ascii", errors="strict")
        except (knowledge_action.KnowledgeActionMemoryError, UnicodeError) as exc:
            raise RustActionInterventionError(
                "Rust intervention binding authentication failed"
            ) from exc
        payload = {
            "candidate_sha256": self.candidate.sha256,
            "rust_source_sha256": self.rust_source_sha256,
        }
        if (
            not source_bytes
            or len(source_bytes) > rust_executor.MAX_SOURCE_BYTES
            or self.rust_source_sha256 != sha256(source_bytes).hexdigest()
            or self.sha256 != _digest(_BINDING_NAMESPACE, payload)
        ):
            raise RustActionInterventionError("Rust intervention binding authentication failed")


def _make_binding(
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
    rust_source: str,
) -> _RustActionBindingV1:
    if type(candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
        raise TypeError("candidate_sources require exact candidate projections")
    if type(rust_source) is not str:
        raise TypeError("candidate_sources require exact Rust source strings")
    candidate.authenticate()
    try:
        source_bytes = rust_source.encode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RustActionInterventionError("Rust action source must be ASCII") from exc
    if not source_bytes or len(source_bytes) > rust_executor.MAX_SOURCE_BYTES:
        raise RustActionInterventionError("Rust action source size is invalid")
    rust_source_sha256 = sha256(source_bytes).hexdigest()
    payload = {
        "candidate_sha256": candidate.sha256,
        "rust_source_sha256": rust_source_sha256,
    }
    binding = _RustActionBindingV1(
        candidate=candidate,
        rust_source_sha256=rust_source_sha256,
        rust_source=rust_source,
        sha256=_digest(_BINDING_NAMESPACE, payload),
    )
    binding.authenticate()
    return binding


def _canonical_bindings(
    candidate_sources: tuple[
        tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, str], ...
    ],
) -> tuple[_RustActionBindingV1, ...]:
    if type(candidate_sources) is not tuple:
        raise TypeError("candidate_sources must be one exact tuple")
    if not MIN_ACTIONS <= len(candidate_sources) <= MAX_ACTIONS:
        raise RustActionInterventionError("candidate catalogue must contain 2 to 4 actions")
    bindings: list[_RustActionBindingV1] = []
    for pair in candidate_sources:
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError("each candidate source binding must be one exact pair")
        bindings.append(_make_binding(pair[0], pair[1]))
    if len({binding.candidate.sha256 for binding in bindings}) != len(bindings):
        raise RustActionInterventionError("candidate projection is duplicated")
    if len({binding.candidate.candidate_id for binding in bindings}) != len(bindings):
        raise RustActionInterventionError("opaque candidate identity is duplicated")
    if len({binding.rust_source_sha256 for binding in bindings}) != len(bindings):
        raise RustActionInterventionError("Rust action source is duplicated")
    return tuple(sorted(bindings, key=lambda binding: binding.candidate.sha256))


@dataclass(frozen=True, slots=True)
class RustActionInterventionCatalogueV1:
    schema: str
    split: str
    claim_boundary: str
    candidate_sha256s: tuple[str, ...]
    binding_sha256s: tuple[str, ...]
    rust_source_sha256s: tuple[str, ...]
    action_count: int
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            *self.candidate_sha256s,
            *self.binding_sha256s,
            *self.rust_source_sha256s,
        )
        if (
            self.schema != SCHEMA
            or self.split != SPLIT
            or self.claim_boundary != CLAIM_BOUNDARY
            or type(self.candidate_sha256s) is not tuple
            or type(self.binding_sha256s) is not tuple
            or type(self.rust_source_sha256s) is not tuple
            or type(self.action_count) is not int
            or not MIN_ACTIONS <= self.action_count <= MAX_ACTIONS
            or len(self.candidate_sha256s) != self.action_count
            or len(self.binding_sha256s) != self.action_count
            or len(self.rust_source_sha256s) != self.action_count
            or tuple(sorted(self.candidate_sha256s)) != self.candidate_sha256s
            or len(set(self.candidate_sha256s)) != self.action_count
            or len(set(self.binding_sha256s)) != self.action_count
            or len(set(self.rust_source_sha256s)) != self.action_count
            or any(not _valid_sha256(root) for root in roots)
            or self.sha256 != _digest(_CATALOGUE_NAMESPACE, payload)
        ):
            raise RustActionInterventionError("Rust intervention catalogue authentication failed")


def _make_catalogue(
    bindings: tuple[_RustActionBindingV1, ...],
) -> RustActionInterventionCatalogueV1:
    for binding in bindings:
        binding.authenticate()
    payload = {
        "schema": SCHEMA,
        "split": SPLIT,
        "claim_boundary": CLAIM_BOUNDARY,
        "candidate_sha256s": tuple(binding.candidate.sha256 for binding in bindings),
        "binding_sha256s": tuple(binding.sha256 for binding in bindings),
        "rust_source_sha256s": tuple(binding.rust_source_sha256 for binding in bindings),
        "action_count": len(bindings),
    }
    catalogue = RustActionInterventionCatalogueV1(
        **payload,
        sha256=_digest(_CATALOGUE_NAMESPACE, payload),
    )
    catalogue.authenticate_structure()
    return catalogue


@dataclass(frozen=True, slots=True)
class RustActionInterventionAuthorityV1:
    schema: str
    split: str
    catalogue_sha256: str
    instance_nonce_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        if (
            self.schema != SCHEMA
            or self.split != SPLIT
            or not _valid_sha256(self.catalogue_sha256)
            or not _valid_sha256(self.instance_nonce_sha256)
            or self.sha256 != _digest(_AUTHORITY_NAMESPACE, payload)
        ):
            raise RustActionInterventionError("Rust intervention authority authentication failed")


def _make_authority(
    catalogue: RustActionInterventionCatalogueV1,
) -> RustActionInterventionAuthorityV1:
    catalogue.authenticate_structure()
    payload = {
        "schema": SCHEMA,
        "split": SPLIT,
        "catalogue_sha256": catalogue.sha256,
        "instance_nonce_sha256": sha256(token_bytes(32)).hexdigest(),
    }
    authority = RustActionInterventionAuthorityV1(
        **payload,
        sha256=_digest(_AUTHORITY_NAMESPACE, payload),
    )
    authority.authenticate_structure()
    return authority


@dataclass(frozen=True, slots=True)
class RustActionInterventionSelectionV1:
    schema: str
    split: str
    authority_sha256: str
    catalogue_sha256: str
    candidate_sha256: str
    intervention_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            self.authority_sha256,
            self.catalogue_sha256,
            self.candidate_sha256,
            self.intervention_sha256,
        )
        if (
            self.schema != SCHEMA
            or self.split != SPLIT
            or any(not _valid_sha256(root) for root in roots)
            or self.sha256 != _digest(_SELECTION_NAMESPACE, payload)
        ):
            raise RustActionInterventionError("Rust intervention selection authentication failed")


def _make_selection(
    *,
    authority_sha256: str,
    catalogue_sha256: str,
    candidate_sha256: str,
    intervention_sha256: str,
) -> RustActionInterventionSelectionV1:
    payload = {
        "schema": SCHEMA,
        "split": SPLIT,
        "authority_sha256": authority_sha256,
        "catalogue_sha256": catalogue_sha256,
        "candidate_sha256": candidate_sha256,
        "intervention_sha256": intervention_sha256,
    }
    selection = RustActionInterventionSelectionV1(
        **payload,
        sha256=_digest(_SELECTION_NAMESPACE, payload),
    )
    selection.authenticate_structure()
    return selection


@dataclass(frozen=True, slots=True)
class _RustActionInterventionReleaseV1:
    authority_sha256: str
    catalogue_sha256: str
    selection_sha256: str
    candidate_sha256: str
    intervention_sha256: str
    binding_sha256: str
    rust_source_sha256: str
    sha256: str

    def authenticate(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        if any(not _valid_sha256(root) for root in payload.values()) or self.sha256 != _digest(
            _RELEASE_NAMESPACE, payload
        ):
            raise RustActionInterventionError("Rust intervention release authentication failed")


def _make_release(
    selection: RustActionInterventionSelectionV1,
    binding: _RustActionBindingV1,
) -> _RustActionInterventionReleaseV1:
    selection.authenticate_structure()
    binding.authenticate()
    payload = {
        "authority_sha256": selection.authority_sha256,
        "catalogue_sha256": selection.catalogue_sha256,
        "selection_sha256": selection.sha256,
        "candidate_sha256": selection.candidate_sha256,
        "intervention_sha256": selection.intervention_sha256,
        "binding_sha256": binding.sha256,
        "rust_source_sha256": binding.rust_source_sha256,
    }
    release = _RustActionInterventionReleaseV1(
        **payload,
        sha256=_digest(_RELEASE_NAMESPACE, payload),
    )
    release.authenticate()
    return release


@dataclass(frozen=True, slots=True)
class RustActionInterventionWorkCountersV1:
    issued_selection_count: int
    spent_selection_count: int
    source_release_count: int
    execution_attempt_count: int
    successful_execution_count: int

    def authenticate_structure(self) -> None:
        values = (
            self.issued_selection_count,
            self.spent_selection_count,
            self.source_release_count,
            self.execution_attempt_count,
            self.successful_execution_count,
        )
        if (
            any(type(value) is not int or value < 0 for value in values)
            or self.spent_selection_count > self.issued_selection_count
            or self.source_release_count > self.spent_selection_count
            or self.execution_attempt_count > self.source_release_count
            or self.successful_execution_count > self.execution_attempt_count
        ):
            raise RustActionInterventionError("Rust intervention work counters are invalid")


@dataclass(frozen=True, slots=True)
class _PreparedRustExecutionV1:
    rustc: rust_executor.RustcIdentityV1
    preparation: rust_executor.RustSelectedSourcePreparationV2
    cases: tuple[tuple[int, ...], ...]
    stdin: bytes


def _prepare_execution(
    authority_sha256: str,
    batch: tuple[tuple[int, ...], ...],
) -> _PreparedRustExecutionV1:
    """Build the existing V2 preparation without registering a new authority type."""

    if os.name != "nt":
        raise rust_executor.RustSelectedActionExecutionError(
            "Rust selected-source execution v2 requires Windows Job Objects"
        )
    cases = rust_executor._validate_batch(batch)
    stdin = rust_executor._encode_batch(cases)
    identity = rust_executor._discover_rustc()
    compile_environment, run_environment = rust_executor._frozen_environments(identity)
    payload = {
        "schema": rust_executor.V2_SCHEMA,
        "claim_boundary": rust_executor.V2_CLAIM_BOUNDARY,
        "compiler_policy_sha256": rust_executor.V2_COMPILER_POLICY_SHA256,
        "rustc_identity_sha256": identity.sha256,
        "expected_authority_sha256": authority_sha256,
        "harness_sha256": rust_executor.HARNESS_SHA256,
        "batch_protocol_sha256": rust_executor.V2_BATCH_PROTOCOL_SHA256,
        "batch_sha256": rust_executor._digest(
            rust_executor._V2_BATCH_NAMESPACE,
            cases,
        ),
        "stdin_sha256": sha256(stdin).hexdigest(),
        "compile_environment": compile_environment,
        "compile_environment_sha256": rust_executor._environment_sha256(
            "compile",
            compile_environment,
        ),
        "run_environment": run_environment,
        "run_environment_sha256": rust_executor._environment_sha256(
            "run",
            run_environment,
        ),
        "case_count": len(cases),
        "compile_limits_sha256": rust_executor.COMPILE_LIMITS.sha256,
        "run_limits_sha256": rust_executor.RUN_LIMITS.sha256,
    }
    preparation = rust_executor.RustSelectedSourcePreparationV2(
        **payload,
        sha256=rust_executor._digest(
            rust_executor._V2_PREPARATION_NAMESPACE,
            payload,
        ),
    )
    preparation.authenticate_structure()
    return _PreparedRustExecutionV1(
        rustc=identity,
        preparation=preparation,
        cases=cases,
        stdin=stdin,
    )


def _authenticate_execution_preparation(
    prepared: _PreparedRustExecutionV1,
    *,
    authority_sha256: str,
) -> None:
    if type(prepared) is not _PreparedRustExecutionV1:
        raise TypeError("Rust intervention execution preparation must be exact")
    if type(prepared.rustc) is not rust_executor.RustcIdentityV1:
        raise TypeError("Rust intervention compiler identity must be exact")
    if type(prepared.preparation) is not rust_executor.RustSelectedSourcePreparationV2:
        raise TypeError("Rust intervention V2 preparation must be exact")
    prepared.rustc.authenticate_structure()
    prepared.preparation.authenticate_structure()
    cases = rust_executor._validate_batch(prepared.cases)
    if (
        cases != prepared.cases
        or type(prepared.stdin) is not bytes
        or prepared.stdin != rust_executor._encode_batch(cases)
        or prepared.preparation.expected_authority_sha256 != authority_sha256
        or prepared.preparation.rustc_identity_sha256 != prepared.rustc.sha256
        or prepared.preparation.batch_sha256
        != rust_executor._digest(rust_executor._V2_BATCH_NAMESPACE, cases)
        or prepared.preparation.stdin_sha256 != sha256(prepared.stdin).hexdigest()
        or prepared.preparation.case_count != len(cases)
    ):
        raise rust_executor.RustSelectedActionExecutionError(
            "Rust intervention execution preparation changed"
        )
    rust_executor._authenticate_rust_toolchain_current(prepared.rustc)


def _execute_prepared(
    selected_source: rust_executor.RustSelectedSourceV2,
    prepared: _PreparedRustExecutionV1,
    *,
    authenticate_current: object,
) -> rust_executor.RustSelectedActionExecutionV2:
    if type(selected_source) is not rust_executor.RustSelectedSourceV2:
        raise TypeError("Rust intervention selected-source grant must be exact")
    if not callable(authenticate_current):
        raise TypeError("authenticate_current must be callable")
    selected_source.authenticate_structure()
    observation = rust_executor._execute_authenticated_source_observation_once(
        selected_source,
        rustc=prepared.rustc,
        preparation=prepared.preparation,
        cases=prepared.cases,
        stdin=prepared.stdin,
        authenticate_current=authenticate_current,
    )
    dependency_receipt, compile_receipt, run_receipt = rust_executor._make_v2_execution_receipts(
        selected_source,
        rustc=prepared.rustc,
        preparation=prepared.preparation,
        cases=prepared.cases,
        stdin=prepared.stdin,
        observation=observation,
    )
    outputs = observation.outputs
    payload = {
        "selected_source_grant_sha256": selected_source.sha256,
        "rustc_identity_sha256": prepared.rustc.sha256,
        "preparation_sha256": prepared.preparation.sha256,
        "dependency_receipt_sha256": dependency_receipt.sha256,
        "compile_receipt_sha256": compile_receipt.sha256,
        "run_receipt_sha256": run_receipt.sha256,
        "outputs": outputs,
    }
    execution = rust_executor.RustSelectedActionExecutionV2(
        selected_source=selected_source,
        rustc_identity=prepared.rustc,
        preparation=prepared.preparation,
        dependency_receipt=dependency_receipt,
        compile_receipt=compile_receipt,
        run_receipt=run_receipt,
        outputs=outputs,
        sha256=rust_executor._digest(
            rust_executor._V2_EXECUTION_NAMESPACE,
            payload,
        ),
    )
    execution.authenticate_structure()
    return execution


class PreparedRustActionInterventionV1:
    """Factory-owned, API-opaque authority for neutral TRAIN interventions."""

    __slots__ = (
        "_authority",
        "_bindings",
        "_catalogue",
        "_execution_attempt_count",
        "_issued_intervention_sha256s",
        "_issued_selections",
        "_lock",
        "_source_release_count",
        "_spent_selection_sha256s",
        "_successful_execution_count",
    )

    def __init__(
        self,
        *,
        bindings: tuple[_RustActionBindingV1, ...],
        catalogue: RustActionInterventionCatalogueV1,
        authority: RustActionInterventionAuthorityV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("Rust intervention authorities are factory-owned")
        self._bindings = bindings
        self._catalogue = catalogue
        self._authority = authority
        self._lock = Lock()
        self._issued_selections: dict[str, RustActionInterventionSelectionV1] = {}
        self._issued_intervention_sha256s: set[str] = set()
        self._spent_selection_sha256s: set[str] = set()
        self._source_release_count = 0
        self._execution_attempt_count = 0
        self._successful_execution_count = 0
        self._authenticate_current()

    def _authenticate_current(self) -> None:
        if (
            type(self._bindings) is not tuple
            or not MIN_ACTIONS <= len(self._bindings) <= MAX_ACTIONS
            or any(type(binding) is not _RustActionBindingV1 for binding in self._bindings)
        ):
            raise RustActionInterventionError("Rust intervention bindings changed")
        for binding in self._bindings:
            binding.authenticate()
        if (
            tuple(sorted(self._bindings, key=lambda binding: binding.candidate.sha256))
            != self._bindings
        ):
            raise RustActionInterventionError("Rust intervention catalogue order changed")
        expected_catalogue = _make_catalogue(self._bindings)
        if type(self._catalogue) is not RustActionInterventionCatalogueV1:
            raise TypeError("Rust intervention catalogue must be exact")
        self._catalogue.authenticate_structure()
        if self._catalogue != expected_catalogue:
            raise RustActionInterventionError("Rust intervention catalogue changed")
        if type(self._authority) is not RustActionInterventionAuthorityV1:
            raise TypeError("Rust intervention authority receipt must be exact")
        self._authority.authenticate_structure()
        if self._authority.catalogue_sha256 != self._catalogue.sha256:
            raise RustActionInterventionError("Rust intervention authority catalogue changed")

    def _binding_for_candidate(self, candidate_sha256: str) -> _RustActionBindingV1:
        for binding in self._bindings:
            if binding.candidate.sha256 == candidate_sha256:
                return binding
        raise RustActionInterventionError("candidate is not in the intervention catalogue")

    def _consume_issued_selection(
        self,
        selection: RustActionInterventionSelectionV1,
    ) -> _RustActionBindingV1:
        self._authenticate_current()
        if type(selection) is not RustActionInterventionSelectionV1:
            raise TypeError("selection must be one exact Rust intervention selection")
        selection.authenticate_structure()
        if (
            selection.authority_sha256 != self._authority.sha256
            or selection.catalogue_sha256 != self._catalogue.sha256
        ):
            raise RustActionInterventionError(
                "selection belongs to a different intervention authority"
            )
        binding = self._binding_for_candidate(selection.candidate_sha256)
        with self._lock:
            issued = self._issued_selections.get(selection.sha256)
            if issued != selection:
                raise RustActionInterventionError(
                    "selection was not issued by this intervention authority"
                )
            if selection.sha256 in self._spent_selection_sha256s:
                raise RustActionInterventionError("selection was already consumed")
            self._spent_selection_sha256s.add(selection.sha256)
        return binding

    @property
    def catalogue(self) -> RustActionInterventionCatalogueV1:
        self._authenticate_current()
        return self._catalogue

    @property
    def authority(self) -> RustActionInterventionAuthorityV1:
        self._authenticate_current()
        return self._authority

    @property
    def authority_sha256(self) -> str:
        return self.authority.sha256

    @property
    def catalogue_sha256(self) -> str:
        return self.catalogue.sha256

    @property
    def candidate_sha256s(self) -> tuple[str, ...]:
        return self.catalogue.candidate_sha256s

    @property
    def work_counters(self) -> RustActionInterventionWorkCountersV1:
        self._authenticate_current()
        with self._lock:
            counters = RustActionInterventionWorkCountersV1(
                issued_selection_count=len(self._issued_selections),
                spent_selection_count=len(self._spent_selection_sha256s),
                source_release_count=self._source_release_count,
                execution_attempt_count=self._execution_attempt_count,
                successful_execution_count=self._successful_execution_count,
            )
        counters.authenticate_structure()
        return counters

    def issue(
        self,
        candidate_sha256: str,
        intervention_sha256: str,
    ) -> RustActionInterventionSelectionV1:
        """Issue one API-opaque selection for one unique TRAIN intervention."""

        self._authenticate_current()
        if not _valid_sha256(candidate_sha256):
            raise RustActionInterventionError("candidate root is invalid")
        if not _valid_sha256(intervention_sha256):
            raise RustActionInterventionError("intervention root is invalid")
        self._binding_for_candidate(candidate_sha256)
        selection = _make_selection(
            authority_sha256=self._authority.sha256,
            catalogue_sha256=self._catalogue.sha256,
            candidate_sha256=candidate_sha256,
            intervention_sha256=intervention_sha256,
        )
        with self._lock:
            if intervention_sha256 in self._issued_intervention_sha256s:
                raise RustActionInterventionError("intervention root was already issued")
            if len(self._issued_selections) >= MAX_ISSUED_SELECTIONS:
                raise RustActionInterventionError("issued selection capacity exceeded")
            if selection.sha256 in self._issued_selections:
                raise RustActionInterventionError("selection root was already issued")
            self._issued_intervention_sha256s.add(intervention_sha256)
            self._issued_selections[selection.sha256] = selection
        return selection

    def execute_selected(
        self,
        selection: RustActionInterventionSelectionV1,
        batch: tuple[tuple[int, ...], ...],
    ) -> rust_executor.RustSelectedActionExecutionV2:
        """Spend one issued selection, then release, compile, and run its source once."""

        binding = self._consume_issued_selection(selection)
        prepared = _prepare_execution(self._authority.sha256, batch)
        release = _make_release(selection, binding)
        with self._lock:
            self._source_release_count += 1

        selected_source = rust_executor.make_rust_selected_source_v2(
            authority_sha256=self._authority.sha256,
            selection_sha256=selection.sha256,
            release_sha256=release.sha256,
            action_handle=binding.candidate.candidate_id,
            source=binding.rust_source,
        )
        if type(selected_source) is not rust_executor.RustSelectedSourceV2:
            raise TypeError("selected-source factory returned an inexact grant")
        selected_source.authenticate_structure()
        if (
            selected_source.authority_sha256 != self._authority.sha256
            or selected_source.selection_sha256 != selection.sha256
            or selected_source.release_sha256 != release.sha256
            or selected_source.action_handle != binding.candidate.candidate_id
            or selected_source.source_sha256 != binding.rust_source_sha256
            or selected_source.source != binding.rust_source
        ):
            raise RustActionInterventionError(
                "selected-source grant differs from intervention release"
            )
        with self._lock:
            self._execution_attempt_count += 1

        def authenticate_current() -> None:
            self._authenticate_current()
            _authenticate_execution_preparation(
                prepared,
                authority_sha256=self._authority.sha256,
            )

        result = _execute_prepared(
            selected_source,
            prepared,
            authenticate_current=authenticate_current,
        )
        if type(result) is not rust_executor.RustSelectedActionExecutionV2:
            raise TypeError("Rust intervention execution result must be exact V2")
        result.authenticate_structure()
        with self._lock:
            self._successful_execution_count += 1
        return result


def prepare_rust_action_intervention_v1(
    candidate_sources: tuple[
        tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, str], ...
    ],
) -> PreparedRustActionInterventionV1:
    """Bind an API-opaque, 2-to-4 action catalogue for neutral TRAIN use."""

    bindings = _canonical_bindings(candidate_sources)
    catalogue = _make_catalogue(bindings)
    authority = _make_authority(catalogue)
    return PreparedRustActionInterventionV1(
        bindings=bindings,
        catalogue=catalogue,
        authority=authority,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "MAX_ACTIONS",
    "MIN_ACTIONS",
    "SCHEMA",
    "SPLIT",
    "PreparedRustActionInterventionV1",
    "RustActionInterventionAuthorityV1",
    "RustActionInterventionCatalogueV1",
    "RustActionInterventionError",
    "RustActionInterventionSelectionV1",
    "RustActionInterventionWorkCountersV1",
    "prepare_rust_action_intervention_v1",
]
