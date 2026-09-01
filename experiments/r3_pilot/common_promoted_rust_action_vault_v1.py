"""Authenticated custody for Rust sources selected by continual lexical memory.

The module binds the complete opaque candidate catalogue of one immutable
continual-documentation checkpoint to candidate-provided Rust sources before a
selection is released. It does not infer source semantics, synthesize source,
score candidates, compile Rust, or inspect execution cases and outcomes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from secrets import token_bytes
from threading import Lock
from typing import TypeVar

from experiments.r3_pilot import common_continual_documentation_action_memory_v1 as continual
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from humanbrain import rust_selected_action_executor

SCHEMA = "humanbrain.promoted-rust-action-vault.v1"
CLAIM = (
    "authenticated-custody-of-candidate-provided-bounded-ascii-rust-sources-"
    "selected-by-one-exact-continual-lexical-memory-checkpoint-not-source-"
    "synthesis-semantics-debugging-rust-correctness-transfer-or-intelligence"
)

_BINDING_NAMESPACE = b"promoted-rust-action-vault-binding-v1"
_RECEIPT_NAMESPACE = b"promoted-rust-action-vault-receipt-v1"
_AUTHORITY_NAMESPACE = b"promoted-rust-action-vault-authority-v1"
_RELEASE_NAMESPACE = b"promoted-rust-action-vault-release-v1"
_CONSTRUCTION_TOKEN = object()
_SHA256_HEX = frozenset("0123456789abcdef")
_T = TypeVar("_T")
MAX_ISSUED_RECALLS = 4_096


class PromotedRustActionVaultError(ValueError):
    """A source binding, checkpoint, selection, or release failed authentication."""


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


def _candidate_record(
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
) -> dict[str, str]:
    return {
        "candidate_id": candidate.candidate_id,
        "symbol": candidate.symbol,
        "source_sha256": candidate.source_sha256,
        "ast_sha256": candidate.ast_sha256,
        "sha256": candidate.sha256,
    }


@dataclass(frozen=True, slots=True)
class _PromotedRustActionBindingV1:
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1
    rust_source_sha256: str
    rust_source: str
    sha256: str

    def record(self, *, include_source: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "candidate": _candidate_record(self.candidate),
            "rust_source_sha256": self.rust_source_sha256,
            "sha256": self.sha256,
        }
        if include_source:
            result["rust_source"] = self.rust_source
        return result

    def authenticate(self) -> None:
        if type(self.candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
            raise TypeError("vault bindings require exact candidate projections")
        try:
            self.candidate.authenticate()
            source_bytes = self.rust_source.encode("ascii", errors="strict")
        except (knowledge_action.KnowledgeActionMemoryError, UnicodeError) as exc:
            raise PromotedRustActionVaultError("Rust action binding authentication failed") from exc
        payload = {
            "candidate": _candidate_record(self.candidate),
            "rust_source_sha256": self.rust_source_sha256,
        }
        if (
            not source_bytes
            or len(source_bytes) > rust_selected_action_executor.MAX_SOURCE_BYTES
            or self.rust_source_sha256 != sha256(source_bytes).hexdigest()
            or self.sha256 != _digest(_BINDING_NAMESPACE, payload)
        ):
            raise PromotedRustActionVaultError("Rust action binding authentication failed")


def _make_binding(
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
    rust_source: str,
) -> _PromotedRustActionBindingV1:
    if type(candidate) is not knowledge_action.KnowledgeActionCandidateProjectionV1:
        raise TypeError("candidate_sources require exact candidate projections")
    if type(rust_source) is not str:
        raise TypeError("candidate_sources require exact Rust source strings")
    candidate.authenticate()
    try:
        source_bytes = rust_source.encode("ascii", errors="strict")
    except UnicodeError as exc:
        raise PromotedRustActionVaultError("Rust action source must be ASCII") from exc
    if not source_bytes or len(source_bytes) > rust_selected_action_executor.MAX_SOURCE_BYTES:
        raise PromotedRustActionVaultError("Rust action source size is invalid")
    rust_source_sha256 = sha256(source_bytes).hexdigest()
    payload = {
        "candidate": _candidate_record(candidate),
        "rust_source_sha256": rust_source_sha256,
    }
    binding = _PromotedRustActionBindingV1(
        candidate=candidate,
        rust_source_sha256=rust_source_sha256,
        rust_source=rust_source,
        sha256=_digest(_BINDING_NAMESPACE, payload),
    )
    binding.authenticate()
    return binding


@dataclass(frozen=True, slots=True)
class PromotedRustActionVaultReceiptV1:
    schema: str
    claim: str
    memory_checkpoint_sha256: str
    candidate_sha256s: tuple[str, ...]
    promoted_candidate_sha256s: tuple[str, ...]
    provenance_sha256s: tuple[str, ...]
    binding_sha256s: tuple[str, ...]
    rust_source_sha256s: tuple[str, ...]
    action_count: int
    promoted_action_count: int
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            self.memory_checkpoint_sha256,
            *self.candidate_sha256s,
            *self.promoted_candidate_sha256s,
            *self.provenance_sha256s,
            *self.binding_sha256s,
            *self.rust_source_sha256s,
        )
        if (
            self.schema != SCHEMA
            or self.claim != CLAIM
            or not self.candidate_sha256s
            or not self.promoted_candidate_sha256s
            or len(self.candidate_sha256s) != self.action_count
            or len(self.binding_sha256s) != self.action_count
            or len(self.rust_source_sha256s) != self.action_count
            or len(self.promoted_candidate_sha256s) != self.promoted_action_count
            or type(self.action_count) is not int
            or not 1 <= self.action_count <= continual.MAX_ACTIONS
            or type(self.promoted_action_count) is not int
            or not 1 <= self.promoted_action_count <= self.action_count
            or len(set(self.candidate_sha256s)) != self.action_count
            or len(set(self.promoted_candidate_sha256s)) != self.promoted_action_count
            or not set(self.promoted_candidate_sha256s).issubset(self.candidate_sha256s)
            or not self.provenance_sha256s
            or len(set(self.binding_sha256s)) != self.action_count
            or len(set(self.rust_source_sha256s)) != self.action_count
            or len(set(self.provenance_sha256s)) != len(self.provenance_sha256s)
            or any(not _valid_sha256(value) for value in roots)
            or self.sha256 != _digest(_RECEIPT_NAMESPACE, payload)
        ):
            raise PromotedRustActionVaultError("vault receipt authentication failed")


@dataclass(frozen=True, slots=True)
class PromotedRustActionVaultAuthorityV1:
    schema: str
    vault_receipt_sha256: str
    instance_nonce_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        if (
            self.schema != SCHEMA
            or not _valid_sha256(self.vault_receipt_sha256)
            or not _valid_sha256(self.instance_nonce_sha256)
            or self.sha256 != _digest(_AUTHORITY_NAMESPACE, payload)
        ):
            raise PromotedRustActionVaultError("vault authority authentication failed")


def _make_authority(
    receipt: PromotedRustActionVaultReceiptV1,
) -> PromotedRustActionVaultAuthorityV1:
    receipt.authenticate_structure()
    payload = {
        "schema": SCHEMA,
        "vault_receipt_sha256": receipt.sha256,
        "instance_nonce_sha256": sha256(token_bytes(32)).hexdigest(),
    }
    authority = PromotedRustActionVaultAuthorityV1(
        **payload,
        sha256=_digest(_AUTHORITY_NAMESPACE, payload),
    )
    authority.authenticate_structure()
    return authority


def _make_receipt(
    memory: continual.FrozenContinualDocumentationActionMemoryV1,
    bindings: tuple[_PromotedRustActionBindingV1, ...],
) -> PromotedRustActionVaultReceiptV1:
    memory.authenticate()
    for binding in bindings:
        binding.authenticate()
    payload = {
        "schema": SCHEMA,
        "claim": CLAIM,
        "memory_checkpoint_sha256": memory.checkpoint_sha256,
        "candidate_sha256s": tuple(binding.candidate.sha256 for binding in bindings),
        "promoted_candidate_sha256s": memory.promoted_candidate_sha256s,
        "provenance_sha256s": tuple(row.sha256 for row in memory.provenance),
        "binding_sha256s": tuple(binding.sha256 for binding in bindings),
        "rust_source_sha256s": tuple(binding.rust_source_sha256 for binding in bindings),
        "action_count": len(bindings),
        "promoted_action_count": len(memory.promoted_candidate_sha256s),
    }
    receipt = PromotedRustActionVaultReceiptV1(
        **payload,
        sha256=_digest(_RECEIPT_NAMESPACE, payload),
    )
    receipt.authenticate_structure()
    return receipt


@dataclass(frozen=True, slots=True)
class _PromotedRustActionReleaseV1:
    authority_sha256: str
    memory_checkpoint_sha256: str
    selection_sha256: str
    candidate_set_sha256: str
    selected_action_index: int
    candidate_id: str
    candidate_sha256: str
    candidate_projection_source_sha256: str
    binding_sha256: str
    rust_source_sha256: str
    promotion_provenance_sha256s: tuple[str, ...]
    sha256: str

    def authenticate(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            self.authority_sha256,
            self.memory_checkpoint_sha256,
            self.selection_sha256,
            self.candidate_set_sha256,
            self.candidate_sha256,
            self.candidate_projection_source_sha256,
            self.binding_sha256,
            self.rust_source_sha256,
            *self.promotion_provenance_sha256s,
        )
        if (
            type(self.selected_action_index) is not int
            or self.selected_action_index < 0
            or type(self.candidate_id) is not str
            or not self.candidate_id
            or not self.promotion_provenance_sha256s
            or len(set(self.promotion_provenance_sha256s)) != len(self.promotion_provenance_sha256s)
            or any(not _valid_sha256(value) for value in roots)
            or self.sha256 != _digest(_RELEASE_NAMESPACE, payload)
        ):
            raise PromotedRustActionVaultError("Rust action release authentication failed")


def _make_release(
    *,
    authority_sha256: str,
    selection: continual.ContinualDocumentationActionRecallV1,
    binding: _PromotedRustActionBindingV1,
    promotion_provenance_sha256s: tuple[str, ...],
) -> _PromotedRustActionReleaseV1:
    payload = {
        "authority_sha256": authority_sha256,
        "memory_checkpoint_sha256": selection.checkpoint_sha256,
        "selection_sha256": selection.sha256,
        "candidate_set_sha256": selection.candidate_set_sha256,
        "selected_action_index": selection.selected_action_index,
        "candidate_id": binding.candidate.candidate_id,
        "candidate_sha256": binding.candidate.sha256,
        "candidate_projection_source_sha256": binding.candidate.source_sha256,
        "binding_sha256": binding.sha256,
        "rust_source_sha256": binding.rust_source_sha256,
        "promotion_provenance_sha256s": promotion_provenance_sha256s,
    }
    release = _PromotedRustActionReleaseV1(
        **payload,
        sha256=_digest(_RELEASE_NAMESPACE, payload),
    )
    release.authenticate()
    return release


def _canonical_bindings(
    memory: continual.FrozenContinualDocumentationActionMemoryV1,
    candidate_sources: tuple[
        tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, str], ...
    ],
) -> tuple[_PromotedRustActionBindingV1, ...]:
    if type(candidate_sources) is not tuple:
        raise TypeError("candidate_sources must be one exact tuple")
    candidates = memory.candidate_projections
    if len(candidate_sources) != len(candidates):
        raise PromotedRustActionVaultError("candidate source coverage differs from memory")
    bindings: list[_PromotedRustActionBindingV1] = []
    for pair in candidate_sources:
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError("each candidate source binding must be one exact pair")
        bindings.append(_make_binding(pair[0], pair[1]))
    by_candidate_sha256: dict[str, _PromotedRustActionBindingV1] = {}
    for binding in bindings:
        if binding.candidate.sha256 in by_candidate_sha256:
            raise PromotedRustActionVaultError("candidate source binding is duplicated")
        by_candidate_sha256[binding.candidate.sha256] = binding
    if len({binding.rust_source_sha256 for binding in bindings}) != len(bindings):
        raise PromotedRustActionVaultError("Rust action source is duplicated")
    expected_roots = tuple(candidate.sha256 for candidate in candidates)
    if set(by_candidate_sha256) != set(expected_roots):
        raise PromotedRustActionVaultError("candidate source bindings differ from memory")
    ordered = tuple(by_candidate_sha256[root] for root in expected_roots)
    if tuple(binding.candidate for binding in ordered) != candidates:
        raise PromotedRustActionVaultError("candidate source projections differ from memory")
    return ordered


class PreparedPromotedRustActionVaultV1:
    """Factory-owned source custody rooted in one immutable continual checkpoint."""

    __slots__ = (
        "_authority",
        "_bindings",
        "_consumed_release_sha256s",
        "_issued_recalls",
        "_lock",
        "_memory",
        "_receipt",
    )

    def __init__(
        self,
        *,
        memory: continual.FrozenContinualDocumentationActionMemoryV1,
        bindings: tuple[_PromotedRustActionBindingV1, ...],
        receipt: PromotedRustActionVaultReceiptV1,
        authority: PromotedRustActionVaultAuthorityV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("promoted Rust action vaults are factory-owned")
        self._memory = memory
        self._bindings = bindings
        self._receipt = receipt
        self._authority = authority
        self._lock = Lock()
        self._issued_recalls: dict[str, continual.ContinualDocumentationActionRecallV1] = {}
        self._consumed_release_sha256s: set[str] = set()
        self._authenticate_current()

    def _authenticate_current(self) -> None:
        if type(self._memory) is not continual.FrozenContinualDocumentationActionMemoryV1:
            raise TypeError("vault memory must be one exact frozen checkpoint")
        self._memory.authenticate()
        expected_candidates = self._memory.candidate_projections
        if (
            type(self._bindings) is not tuple
            or len(self._bindings) != len(expected_candidates)
            or any(type(binding) is not _PromotedRustActionBindingV1 for binding in self._bindings)
        ):
            raise PromotedRustActionVaultError("vault bindings changed")
        for binding in self._bindings:
            binding.authenticate()
        if tuple(binding.candidate for binding in self._bindings) != expected_candidates:
            raise PromotedRustActionVaultError("vault candidate catalogue changed")
        if type(self._receipt) is not PromotedRustActionVaultReceiptV1:
            raise TypeError("vault receipt must be exact")
        self._receipt.authenticate_structure()
        expected_receipt = _make_receipt(self._memory, self._bindings)
        if self._receipt != expected_receipt:
            raise PromotedRustActionVaultError("vault receipt changed")
        if type(self._authority) is not PromotedRustActionVaultAuthorityV1:
            raise TypeError("vault authority must be exact")
        self._authority.authenticate_structure()
        if self._authority.vault_receipt_sha256 != self._receipt.sha256:
            raise PromotedRustActionVaultError("vault authority receipt changed")

    @property
    def receipt(self) -> PromotedRustActionVaultReceiptV1:
        self._authenticate_current()
        return self._receipt

    @property
    def authority(self) -> PromotedRustActionVaultAuthorityV1:
        self._authenticate_current()
        return self._authority

    @property
    def authority_sha256(self) -> str:
        return self.authority.sha256

    def recall(
        self,
        task: knowledge_action.KnowledgeActionTaskProjectionV1,
        *,
        cell_gate: bool = True,
        documentation_gate: bool = True,
    ) -> continual.ContinualDocumentationActionRecallV1:
        """Issue one exact memory recall for later source release."""

        self._authenticate_current()
        result = self._memory.recall(
            task,
            cell_gate=cell_gate,
            documentation_gate=documentation_gate,
        )
        result.authenticate()
        self._authenticate_current()
        with self._lock:
            if (
                result.sha256 not in self._issued_recalls
                and len(self._issued_recalls) >= MAX_ISSUED_RECALLS
            ):
                raise PromotedRustActionVaultError("issued recall capacity exceeded")
            self._issued_recalls[result.sha256] = result
        return result

    def execute_selected(
        self,
        selection: continual.ContinualDocumentationActionRecallV1,
        callback: Callable[[rust_selected_action_executor.RustSelectedSourceV2], _T],
    ) -> _T:
        """Consume one authenticated recall and release its prebound Rust source."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        self._authenticate_current()
        if type(selection) is not continual.ContinualDocumentationActionRecallV1:
            raise TypeError("selection must be one exact continual recall")
        try:
            selection.authenticate()
        except (
            continual.ContinualDocumentationActionMemoryError,
            knowledge_action.KnowledgeActionMemoryError,
            TypeError,
        ) as exc:
            raise PromotedRustActionVaultError("selection authentication failed") from exc
        if selection.status != continual.RECALLED:
            raise PromotedRustActionVaultError("stopped recall cannot release a Rust action")
        if not selection.cell_gate or not selection.documentation_gate:
            raise PromotedRustActionVaultError("recalled selection gates changed")
        if selection.checkpoint_sha256 != self._receipt.memory_checkpoint_sha256:
            raise PromotedRustActionVaultError("selection checkpoint differs from vault")

        candidates = self._memory.candidate_projections
        candidate_sha256s = tuple(candidate.sha256 for candidate in candidates)
        if (
            selection.candidates != candidates
            or candidate_sha256s != self._receipt.candidate_sha256s
        ):
            raise PromotedRustActionVaultError("selection candidates differ from vault")
        selected_action_index = selection.selected_action_index
        if type(selected_action_index) is not int or not 0 <= selected_action_index < len(
            candidates
        ):
            raise PromotedRustActionVaultError("selection action index is invalid")

        candidate = candidates[selected_action_index]
        binding = self._bindings[selected_action_index]
        if (
            selection.selected_candidate_id != candidate.candidate_id
            or selection.selected_candidate_sha256 != candidate.sha256
            or selection.selected_source_sha256 != candidate.source_sha256
            or binding.candidate != candidate
            or binding.sha256 != self._receipt.binding_sha256s[selected_action_index]
            or binding.rust_source_sha256
            != self._receipt.rust_source_sha256s[selected_action_index]
        ):
            raise PromotedRustActionVaultError("selected candidate differs from vault binding")
        if (
            candidate.sha256 not in self._memory.promoted_candidate_sha256s
            or candidate.sha256 not in self._receipt.promoted_candidate_sha256s
        ):
            raise PromotedRustActionVaultError("selected candidate is not promoted")

        promotion_rows = tuple(
            row
            for row in self._memory.provenance
            if row.candidate_sha256 == candidate.sha256 and row.passed_cell == 1
        )
        if (
            len({row.task_projection_sha256 for row in promotion_rows})
            < continual.MIN_PROMOTION_TASK_ROOTS
            or len({row.task_source_sha256 for row in promotion_rows})
            < continual.MIN_PROMOTION_SOURCE_ROOTS
            or any(row.sha256 not in self._receipt.provenance_sha256s for row in promotion_rows)
        ):
            raise PromotedRustActionVaultError("selected candidate promotion quorum changed")
        promotion_provenance_sha256s = tuple(row.sha256 for row in promotion_rows)
        release = _make_release(
            authority_sha256=self._authority.sha256,
            selection=selection,
            binding=binding,
            promotion_provenance_sha256s=promotion_provenance_sha256s,
        )
        selected_source = rust_selected_action_executor.make_rust_selected_source_v2(
            authority_sha256=self._authority.sha256,
            selection_sha256=selection.sha256,
            release_sha256=release.sha256,
            action_handle=candidate.candidate_id,
            source=binding.rust_source,
        )
        if type(selected_source) is not rust_selected_action_executor.RustSelectedSourceV2:
            raise TypeError("selected-source factory returned an inexact grant")
        selected_source.authenticate_structure()
        if (
            selected_source.authority_sha256 != self._authority.sha256
            or selected_source.selection_sha256 != selection.sha256
            or selected_source.release_sha256 != release.sha256
            or selected_source.action_handle != candidate.candidate_id
            or selected_source.source_sha256 != binding.rust_source_sha256
            or selected_source.source != binding.rust_source
        ):
            raise PromotedRustActionVaultError("selected-source grant differs from release")

        with self._lock:
            issued = self._issued_recalls.get(selection.sha256)
            if issued != selection:
                raise PromotedRustActionVaultError(
                    "selection was not issued by this vault authority"
                )
            if release.sha256 in self._consumed_release_sha256s:
                raise PromotedRustActionVaultError("selection was already released")
            self._consumed_release_sha256s.add(release.sha256)
        return callback(selected_source)


def prepare_promoted_rust_action_vault_v1(
    memory: continual.FrozenContinualDocumentationActionMemoryV1,
    *,
    candidate_sources: tuple[
        tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, str], ...
    ],
) -> PreparedPromotedRustActionVaultV1:
    """Bind the full candidate catalogue to immutable candidate-provided Rust sources."""

    if type(memory) is not continual.FrozenContinualDocumentationActionMemoryV1:
        raise TypeError("one exact frozen continual documentation memory is required")
    memory.authenticate()
    if not memory.promoted_candidate_sha256s:
        raise PromotedRustActionVaultError("vault requires at least one promoted action")
    bindings = _canonical_bindings(memory, candidate_sources)
    receipt = _make_receipt(memory, bindings)
    authority = _make_authority(receipt)
    return PreparedPromotedRustActionVaultV1(
        memory=memory,
        bindings=bindings,
        receipt=receipt,
        authority=authority,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "PreparedPromotedRustActionVaultV1",
    "PromotedRustActionVaultAuthorityV1",
    "PromotedRustActionVaultError",
    "PromotedRustActionVaultReceiptV1",
    "prepare_promoted_rust_action_vault_v1",
]
