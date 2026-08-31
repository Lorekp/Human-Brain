"""Type-neutral authenticated two-action repository frontier seam.

The leaf module owns stable policy-qualified action identity and frontier-bound
candidate identity.  String and List frontiers are the two concrete adapters;
the repository controller needs no task-type, operator, path, or relation label.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import NamedTuple, Protocol

from experiments.r3_pilot import public_typed_output_cells as output_cells

CANDIDATE_COUNT = 2
OUTCOME_COMMIT = "COMMIT"
OUTCOME_REVISE = "REVISE"

_ACTION_NAMESPACE = b"common-repository-action-abstraction-v1"
_RECORD_NAMESPACE = b"common-repository-action-record-v1"
_BANK_NAMESPACE = b"common-repository-action-bank-v1"
_WITNESS_NAMESPACE = b"common-repository-patch-witness-v1"
_DECISION_NAMESPACE = b"common-repository-consequence-decision-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def valid_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_repository_action_receipts_v1(
    values: Sequence[str],
) -> tuple[str, ...]:
    """Preserve exact learned actuation order; empty actions are forbidden."""

    receipts = tuple(values)
    if (
        not receipts
        or len(set(receipts)) != len(receipts)
        or any(not valid_sha256(value) for value in receipts)
    ):
        raise ValueError("repository action receipts must be distinct SHA-256 roots")
    return receipts


def repository_action_sha256_v1(
    policy_checkpoint_sha256: str,
    values: Sequence[str],
) -> str:
    if not valid_sha256(policy_checkpoint_sha256):
        raise ValueError("repository action requires one policy checkpoint root")
    return _digest(
        _ACTION_NAMESPACE,
        {
            "policy": policy_checkpoint_sha256,
            "receipts": canonical_repository_action_receipts_v1(values),
        },
    )


def _output_roots(
    values: Sequence[output_cells.TypedOutputEncoding],
) -> tuple[str, ...]:
    outputs = tuple(values)
    if (
        not outputs
        or any(type(value) is not output_cells.TypedOutputEncoding for value in outputs)
        or any(not value.supported or value.unknown_active for value in outputs)
        or len({value.output_type for value in outputs}) != 1
    ):
        raise ValueError("repository action outputs must be one supported physical type")
    return tuple(value.sha256 for value in outputs)


class RepositoryActionCandidateRecordV1(NamedTuple):
    frontier_sha256: str
    program_sha256: str
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    applied_relation_receipt_sha256s: tuple[str, ...]
    receipt_cell_support: int
    sha256: str

    def authenticate(self) -> None:
        receipts = canonical_repository_action_receipts_v1(
            self.applied_relation_receipt_sha256s
        )
        outputs = _output_roots(self.physical_outputs)
        if (
            not valid_sha256(self.frontier_sha256)
            or not valid_sha256(self.program_sha256)
            or self.receipt_cell_support != len(receipts)
            or self.sha256
            != _digest(
                _RECORD_NAMESPACE,
                {
                    "frontier": self.frontier_sha256,
                    "program": self.program_sha256,
                    "outputs": outputs,
                    "receipts": receipts,
                    "support": self.receipt_cell_support,
                },
            )
        ):
            raise ValueError("repository action candidate authentication failed")


def make_repository_action_candidate_v1(
    *,
    frontier_sha256: str,
    program_sha256: str,
    physical_outputs: Sequence[output_cells.TypedOutputEncoding],
    applied_relation_receipt_sha256s: Sequence[str],
) -> RepositoryActionCandidateRecordV1:
    receipts = canonical_repository_action_receipts_v1(
        applied_relation_receipt_sha256s
    )
    outputs = tuple(physical_outputs)
    payload = {
        "frontier": frontier_sha256,
        "program": program_sha256,
        "outputs": _output_roots(outputs),
        "receipts": receipts,
        "support": len(receipts),
    }
    result = RepositoryActionCandidateRecordV1(
        frontier_sha256=frontier_sha256,
        program_sha256=program_sha256,
        physical_outputs=outputs,
        applied_relation_receipt_sha256s=receipts,
        receipt_cell_support=len(receipts),
        sha256=_digest(_RECORD_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class FrozenRepositoryActionBankV1:
    """Exactly two candidates behind one immutable policy/frontier identity."""

    __slots__ = (
        "_action_sha256s",
        "_case_count",
        "_frontier_sha256",
        "_policy_checkpoint_sha256",
        "_records",
        "_root_program_sha256",
        "_sha256",
    )

    def __init__(
        self,
        *,
        policy_checkpoint_sha256: str,
        frontier_sha256: str,
        root_program_sha256: str,
        case_count: int,
        records: Sequence[RepositoryActionCandidateRecordV1],
    ) -> None:
        frozen = tuple(sorted(tuple(records), key=lambda item: item.sha256))
        for record in frozen:
            if type(record) is not RepositoryActionCandidateRecordV1:
                raise TypeError("repository action bank requires exact neutral records")
            record.authenticate()
        if (
            not valid_sha256(policy_checkpoint_sha256)
            or not valid_sha256(frontier_sha256)
            or not valid_sha256(root_program_sha256)
            or type(case_count) is not int
            or case_count < 1
            or len(frozen) != CANDIDATE_COUNT
            or len({item.sha256 for item in frozen}) != CANDIDATE_COUNT
            or any(item.frontier_sha256 != frontier_sha256 for item in frozen)
            or len({len(item.physical_outputs) for item in frozen}) != 1
            or next(iter({len(item.physical_outputs) for item in frozen})) != case_count
        ):
            raise ValueError("repository action bank shape changed")
        actions = tuple(
            repository_action_sha256_v1(
                policy_checkpoint_sha256,
                item.applied_relation_receipt_sha256s,
            )
            for item in frozen
        )
        if len(set(actions)) != CANDIDATE_COUNT:
            raise ValueError("repository action bank actions must be distinct")
        self._policy_checkpoint_sha256 = policy_checkpoint_sha256
        self._frontier_sha256 = frontier_sha256
        self._root_program_sha256 = root_program_sha256
        self._case_count = case_count
        self._records = frozen
        self._action_sha256s = actions
        self._sha256 = _digest(_BANK_NAMESPACE, self._payload())
        self.authenticate()

    def _payload(self) -> object:
        return {
            "policy": self._policy_checkpoint_sha256,
            "frontier": self._frontier_sha256,
            "root_program": self._root_program_sha256,
            "case_count": self._case_count,
            "records": tuple(item.sha256 for item in self._records),
            "actions": self._action_sha256s,
        }

    @property
    def policy_checkpoint_sha256(self) -> str:
        return self._policy_checkpoint_sha256

    @property
    def frontier_sha256(self) -> str:
        return self._frontier_sha256

    @property
    def root_program_sha256(self) -> str:
        return self._root_program_sha256

    @property
    def case_count(self) -> int:
        return self._case_count

    @property
    def records(self) -> tuple[RepositoryActionCandidateRecordV1, ...]:
        self.authenticate()
        return self._records

    @property
    def action_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return self._action_sha256s

    @property
    def sha256(self) -> str:
        self.authenticate()
        return self._sha256

    def action_sha256_for(self, record: RepositoryActionCandidateRecordV1) -> str:
        self.authenticate()
        matches = tuple(
            action
            for candidate, action in zip(
                self._records, self._action_sha256s, strict=True
            )
            if candidate == record
        )
        if len(matches) != 1:
            raise ValueError("candidate is outside the authenticated action bank")
        return matches[0]

    def authenticate(self) -> None:
        for record in self._records:
            record.authenticate()
        if (
            len(self._records) != CANDIDATE_COUNT
            or len(set(self._records)) != CANDIDATE_COUNT
            or tuple(sorted(self._records, key=lambda item: item.sha256))
            != self._records
            or tuple(
                repository_action_sha256_v1(
                    self._policy_checkpoint_sha256,
                    item.applied_relation_receipt_sha256s,
                )
                for item in self._records
            )
            != self._action_sha256s
            or len(set(self._action_sha256s)) != CANDIDATE_COUNT
            or self._sha256 != _digest(_BANK_NAMESPACE, self._payload())
        ):
            raise RuntimeError("repository action bank authentication failed")


class RepositoryPatchWitnessV1(NamedTuple):
    frontier_sha256: str
    candidate_sha256: str
    program_sha256: str
    source: str
    applied_relation_receipt_sha256s: tuple[str, ...]
    sha256: str

    def authenticate(self) -> None:
        receipts = canonical_repository_action_receipts_v1(
            self.applied_relation_receipt_sha256s
        )
        if (
            any(
                not valid_sha256(value)
                for value in (
                    self.frontier_sha256,
                    self.candidate_sha256,
                    self.program_sha256,
                )
            )
            or type(self.source) is not str
            or self.sha256
            != _digest(
                _WITNESS_NAMESPACE,
                {
                    "frontier": self.frontier_sha256,
                    "candidate": self.candidate_sha256,
                    "program": self.program_sha256,
                    "source": sha256(self.source.encode("utf-8")).hexdigest(),
                    "receipts": receipts,
                },
            )
        ):
            raise ValueError("repository patch witness authentication failed")


def make_repository_patch_witness_v1(
    *,
    candidate: RepositoryActionCandidateRecordV1,
    source: str,
) -> RepositoryPatchWitnessV1:
    candidate.authenticate()
    payload = {
        "frontier": candidate.frontier_sha256,
        "candidate": candidate.sha256,
        "program": candidate.program_sha256,
        "source": sha256(source.encode("utf-8")).hexdigest(),
        "receipts": candidate.applied_relation_receipt_sha256s,
    }
    result = RepositoryPatchWitnessV1(
        frontier_sha256=candidate.frontier_sha256,
        candidate_sha256=candidate.sha256,
        program_sha256=candidate.program_sha256,
        source=source,
        applied_relation_receipt_sha256s=(
            candidate.applied_relation_receipt_sha256s
        ),
        sha256=_digest(_WITNESS_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class RepositoryConsequenceDecisionV1(NamedTuple):
    policy_checkpoint_sha256: str
    passed_cell: int
    outcome_key: str
    physical_race_sha256: str
    sha256: str

    def authenticate(self) -> None:
        expected = OUTCOME_COMMIT if self.passed_cell == 1 else OUTCOME_REVISE
        payload = {
            "policy": self.policy_checkpoint_sha256,
            "passed_cell": self.passed_cell,
            "outcome": self.outcome_key,
            "physical_race": self.physical_race_sha256,
        }
        if (
            not valid_sha256(self.policy_checkpoint_sha256)
            or self.passed_cell not in (0, 1)
            or self.outcome_key != expected
            or not valid_sha256(self.physical_race_sha256)
            or self.sha256 != _digest(_DECISION_NAMESPACE, payload)
        ):
            raise ValueError("repository consequence decision authentication failed")


def make_repository_consequence_decision_v1(
    *,
    policy_checkpoint_sha256: str,
    passed_cell: int,
    outcome_key: str,
    physical_race_sha256: str,
) -> RepositoryConsequenceDecisionV1:
    payload = {
        "policy": policy_checkpoint_sha256,
        "passed_cell": passed_cell,
        "outcome": outcome_key,
        "physical_race": physical_race_sha256,
    }
    result = RepositoryConsequenceDecisionV1(
        policy_checkpoint_sha256=policy_checkpoint_sha256,
        passed_cell=passed_cell,
        outcome_key=outcome_key,
        physical_race_sha256=physical_race_sha256,
        sha256=_digest(_DECISION_NAMESPACE, payload),
    )
    result.authenticate()
    return result


class RepositoryActionFrontierV1(Protocol):
    @property
    def repository_action_bank(self) -> FrozenRepositoryActionBankV1: ...

    def encode_visible_expected(
        self, raw: object
    ) -> tuple[output_cells.TypedOutputEncoding, ...]: ...

    def materialize_selected(
        self, candidate: RepositoryActionCandidateRecordV1
    ) -> RepositoryPatchWitnessV1: ...

    def race_repository_consequence(
        self, passed_cell: int
    ) -> RepositoryConsequenceDecisionV1: ...


__all__ = [
    "CANDIDATE_COUNT",
    "OUTCOME_COMMIT",
    "OUTCOME_REVISE",
    "FrozenRepositoryActionBankV1",
    "RepositoryActionCandidateRecordV1",
    "RepositoryActionFrontierV1",
    "RepositoryConsequenceDecisionV1",
    "RepositoryPatchWitnessV1",
    "canonical_repository_action_receipts_v1",
    "make_repository_action_candidate_v1",
    "make_repository_consequence_decision_v1",
    "make_repository_patch_witness_v1",
    "repository_action_sha256_v1",
    "valid_sha256",
]
