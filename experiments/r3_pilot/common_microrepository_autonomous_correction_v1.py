"""Bounded cell-state correction after one real repository-test failure.

The adapter consumes the type-neutral source-free candidate-record seam.  Two
relation-supported candidates are frozen before
one desired read.  A two-cell E/H race first prefers the candidate with two
physical receipt cells.  A failed repository test produces only a PASS/FAIL
consequence receipt; the recurrent action cells select REVISE, exact rollback
is verified, and the candidate inhibitory cell is updated.  The same bank then
reraces and may select the other candidate for one final test.

Candidate order, terminal names, operator tokens, repository paths, and test
contents never enter the candidate race.  Manifest decoding, source
attestation, selected compilation, filesystem/process I/O, hashes, and exact
rollback remain host scaffolding.  The attempt budget is exactly two.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import common_continual_repository_action_memory_v1 as continual_memory
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from experiments.r3_pilot import common_repository_action_memory_v2 as memory_v2
from experiments.r3_pilot import common_repository_source_views_v1 as source_views
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from humanbrain.resting_glif_tier_beam import (
    RETAINED,
    CompiledRestingGlifTierBeam,
    RestingGlifTierBeamResult,
)

SCHEMA = "COMMON_MICROREPOSITORY_AUTONOMOUS_CORRECTION_V1"
PREPARED = f"{SCHEMA}_PREPARED"
COMMITTED = f"{SCHEMA}_COMMITTED"
STOP_ATTESTATION = f"STOP_{SCHEMA}_ATTESTATION"
STOP_RELATION_SUPPORT = f"STOP_{SCHEMA}_RELATION_SUPPORT"
STOP_LEARNED_RECALL = f"STOP_{SCHEMA}_LEARNED_RECALL"
STOP_ADAPTER_INPUT = f"STOP_{SCHEMA}_ADAPTER_INPUT"
STOP_SOURCE_DRIFT = f"STOP_{SCHEMA}_SOURCE_DRIFT"
STOP_ELIGIBILITY = f"STOP_{SCHEMA}_ELIGIBILITY"
STOP_CANDIDATE_RACE = f"STOP_{SCHEMA}_CANDIDATE_RACE"
STOP_REPEATED_WINNER = f"STOP_{SCHEMA}_REPEATED_WINNER"
STOP_MATERIALIZATION = f"STOP_{SCHEMA}_MATERIALIZATION"
STOP_PATCH_WRITE = f"STOP_{SCHEMA}_PATCH_WRITE"
STOP_FORGED_CONSEQUENCE = f"STOP_{SCHEMA}_FORGED_CONSEQUENCE"
STOP_CONSEQUENCE_POLICY = f"STOP_{SCHEMA}_CONSEQUENCE_POLICY"
STOP_REVISION_GATE = f"STOP_{SCHEMA}_REVISION_GATE"
STOP_REPLAN_GATE = f"STOP_{SCHEMA}_REPLAN_GATE"
STOP_ROLLBACK = f"STOP_{SCHEMA}_ROLLBACK"
STOP_ATTEMPT_BUDGET = f"STOP_{SCHEMA}_ATTEMPT_BUDGET"

CANDIDATE_COUNT = action_bank.CANDIDATE_COUNT
ATTEMPT_LIMIT = 2
TONIC_CURRENT = 1
RACE_TICKS = 600
STOP_CELL = 2
CANDIDATE_CURRENT_BASE_PA = 600.0
CANDIDATE_CURRENT_GAIN_PA = 800.0
CANDIDATE_CURRENT_MAX_PA = 3_400.0
CANDIDATE_INHIBITION_PA = 3_000.0
PROVENANCE_ROW_BYTES = 37
TRAINABLE_STATE_BYTES = 4
ADDRESS_STATE_BYTES = 8
PROVENANCE_STATE_BYTES = 74
CHECKPOINT_ARRAY_BYTES = 86
UNSELECTED_SOURCE_MATERIALIZATIONS = 0
UNSELECTED_REPOSITORY_TESTS = 0

CLAIM_BOUNDARY = (
    "one fresh self-posed repository fixture with a two-candidate learned-prefix "
    "frontier; cell state revises once after a real failed test and commits a "
    "different second patch; manifests, attestation, selected compilation, "
    "filesystem/process I/O, PASS/FAIL hashing, and rollback remain host scaffolding"
)

_METADATA_NAMESPACE = b"common-autonomous-correction-metadata-v1"
_STATE_NAMESPACE = b"common-autonomous-correction-state-v1"
_PREPARED_NAMESPACE = b"common-autonomous-correction-prepared-v1"
_CONSEQUENCE_NAMESPACE = b"common-autonomous-correction-consequence-v1"
_ATTEMPT_NAMESPACE = b"common-autonomous-correction-attempt-v1"
_RESULT_TRAJECTORY_NAMESPACE = b"common-autonomous-correction-result-trajectory-v1"
_RESULT_NAMESPACE = b"common-autonomous-correction-result-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(value.tobytes(order="C")).hexdigest()


def _encoding_vector_sha256(
    values: Sequence[output_cells.TypedOutputEncoding],
) -> str:
    return sha256(
        b"".join(item.factor_winners_le + item.physical_digest for item in values)
    ).hexdigest()


def _physical_tier_sha256(value: RestingGlifTierBeamResult) -> str:
    return _digest(
        b"common-autonomous-correction-physical-tier-v1",
        {
            "status": value.status,
            "retained_cells": value.retained_cells,
            "retained_candidates": value.retained_candidate_cells,
            "stop_cell": value.stop_cell,
            "rounds": tuple(
                {
                    "tier": item.tier_index,
                    "winner_cell": item.race.winner_cell,
                    "spike_step": item.race.spike_step,
                    "simultaneous": item.race.simultaneous_spike_candidates,
                    "tier_cells": item.tier_cells,
                    "leading_stop": item.leading_stop,
                    "currents_before": item.currents_before_pa,
                    "currents_after": item.currents_after_pa,
                }
                for item in value.rounds
            ),
            "projection_before": value.inhibition_projection_before_sha256,
            "projection_after": value.inhibition_projection_after_sha256,
        },
    )


class AutonomousCorrectionSpecV1(NamedTuple):
    repo_root: Path
    source_relative_path: str
    expected_manifest_relative_path: str
    test_command: tuple[str, ...]
    root_program_sha256: str
    root_source_sha256: str
    test_timeout_seconds: float = 30.0


class CorrectionCellStateV1(NamedTuple):
    status: str
    frontier_sha256: str
    candidate_sha256s: tuple[str, ...]
    receipt_cell_supports: tuple[int, ...]
    tonic_current: int
    excitatory_sha256: str
    inhibitory_sha256: str
    address_sha256: str
    provenance_sha256: str
    physical_tier_sha256s: tuple[str, ...]
    selector_projection_sha256: str
    filled_provenance_rows: int
    candidate_races: int
    replan_updates: int
    candidate_work_cell_steps: int
    trainable_state_bytes: int
    address_state_bytes: int
    provenance_state_bytes: int
    checkpoint_array_bytes: int
    canonical_metadata_bytes: int
    canonical_metadata_sha256: str
    sha256: str


class RepositoryConsequenceV1(NamedTuple):
    attempt: int
    candidate_sha256: str
    selected_source_sha256: str
    passed_cell: int
    stdout_sha256: str
    stderr_sha256: str
    sha256: str


class AutonomousCorrectionAttemptV1(NamedTuple):
    attempt: int
    candidate_sha256: str
    candidate_address: int
    receipt_cell_support: int
    program_sha256: str
    witness_sha256: str
    selected_source_sha256: str
    repository_test_exit_code: int
    repository_test_stdout_sha256: str
    repository_test_stderr_sha256: str
    consequence_sha256: str
    consequence_action_key: str
    rolled_back: bool
    committed: bool
    sha256: str


class AutonomousCorrectionWorkV1(NamedTuple):
    source_reads: int
    desired_manifest_reads: int
    eligibility_races: int
    candidate_races: int
    source_materializations: int
    unselected_source_materializations: int
    selected_patch_sources: int
    patch_writes: int
    repository_consequence_roots: int
    consequence_policy_races: int
    repository_tests: int
    unselected_repository_tests: int
    revisions: int
    replan_updates: int
    reverts: int
    rollback_verifications: int
    commits: int
    candidate_work_cell_steps: int


class PreparedAutonomousCorrectionStateV1(NamedTuple):
    status: str
    frontier_sha256: str
    policy_checkpoint_sha256: str
    root_program_sha256: str
    root_source_sha256: str
    source_context_sha256: str
    source_context_frame_sha256: str
    candidate_checkpoint_sha256: str | None
    learned_recall_sha256: str | None
    candidate_count: int
    source_reads_during_prepare: int
    desired_reads_during_prepare: int
    source_materializations_during_prepare: int
    patch_writes_during_prepare: int
    repository_tests_during_prepare: int
    sha256: str


class AutonomousCorrectionResultV1(NamedTuple):
    status: str
    prepared_sha256: str
    initial_checkpoint_sha256: str | None
    final_checkpoint: CorrectionCellStateV1 | None
    desired_vector_sha256: str | None
    eligibility_race_sha256: str | None
    attempts: tuple[AutonomousCorrectionAttemptV1, ...]
    repository_actions: tuple[continual_memory.PhysicalRepositoryActionLessonV1, ...]
    source_before_sha256: str
    source_after_sha256: str
    source_context_sha256: str
    work: AutonomousCorrectionWorkV1
    claim_boundary: str
    trajectory_sha256: str
    sha256: str

    def authenticate(self) -> None:
        for attempt in self.attempts:
            if not _authenticate_attempt(attempt):
                raise ValueError("autonomous correction attempt authentication failed")
        final_checkpoint_sha256 = (
            None if self.final_checkpoint is None else self.final_checkpoint.sha256
        )
        trajectory = _result_trajectory_sha256(
            status=self.status,
            prepared_sha256=self.prepared_sha256,
            initial_checkpoint_sha256=self.initial_checkpoint_sha256,
            final_checkpoint_sha256=final_checkpoint_sha256,
            desired_sha256=self.desired_vector_sha256,
            eligibility_sha256=self.eligibility_race_sha256,
            attempts=self.attempts,
            source_before_sha256=self.source_before_sha256,
            source_after_sha256=self.source_after_sha256,
            source_context_sha256=self.source_context_sha256,
            work=self.work,
            claim_boundary=self.claim_boundary,
        )
        if trajectory != self.trajectory_sha256:
            raise ValueError("autonomous correction trajectory authentication failed")
        if self.status == COMMITTED:
            if not self.repository_actions or len(self.repository_actions) != len(
                self.attempts
            ):
                raise ValueError("committed correction lesson cohort changed")
            for ordinal, (attempt, lesson) in enumerate(
                zip(self.attempts, self.repository_actions, strict=True),
                start=1,
            ):
                lesson.authenticate()
                expected_outcome = attempt.consequence_action_key
                if (
                    lesson.controller_trajectory_sha256 != trajectory
                    or lesson.controller_result_sha256 != self.sha256
                    or lesson.controller_status != self.status
                    or lesson.attempt_ordinal != ordinal
                    or lesson.attempt_count != len(self.attempts)
                    or lesson.attempt_sha256 != attempt.sha256
                    or lesson.consequence_sha256 != attempt.consequence_sha256
                    or lesson.candidate_sha256 != attempt.candidate_sha256
                    or lesson.program_sha256 != attempt.program_sha256
                    or lesson.passed_cell
                    != int(attempt.repository_test_exit_code == 0)
                    or lesson.rolled_back != attempt.rolled_back
                    or lesson.committed != attempt.committed
                    or lesson.outcome_key != expected_outcome
                    or lesson.source_sha256 != self.source_before_sha256
                    or lesson.source_context_sha256
                    != self.source_context_sha256
                ):
                    raise ValueError("committed correction lesson binding changed")
            bodies = tuple(
                lesson.lesson_body_sha256 for lesson in self.repository_actions
            )
            expected_result = continual_memory.controller_result_sha256_v1(
                trajectory,
                bodies,
            )
        else:
            if self.repository_actions:
                raise ValueError("non-committed correction exposed learnable lessons")
            expected_result = _digest(
                _RESULT_NAMESPACE,
                {"trajectory": trajectory, "lesson_bodies": ()},
            )
        if self.sha256 != expected_result:
            raise ValueError("autonomous correction result authentication failed")


class _CorrectionCellsV1:
    """Two bounded E/H candidate cells plus fixed two-row provenance cells."""

    __slots__ = (
        "_addresses",
        "_candidate_races",
        "_candidates",
        "_excitatory",
        "_filled_rows",
        "_frontier_sha256",
        "_inhibitory",
        "_metadata",
        "_previous_winner",
        "_provenance",
        "_replan_updates",
        "_selector",
        "_steps",
        "_tier_sha256s",
    )

    def __init__(
        self,
        frontier_sha256: str,
        bank: action_bank.FrozenRepositoryActionBankV1,
        candidates: Sequence[action_bank.RepositoryActionCandidateRecordV1],
        *,
        learned_evidence: dict[str, tuple[int, int]] | None = None,
        learned_recall_sha256: str | None = None,
    ) -> None:
        bank.authenticate()
        frozen = tuple(sorted(candidates, key=lambda item: item.sha256))
        if (
            len(frozen) != CANDIDATE_COUNT
            or len({item.sha256 for item in frozen}) != CANDIDATE_COUNT
            or any(
                item.frontier_sha256 != frontier_sha256
                or item.receipt_cell_support <= 0
                or item not in bank.records
                for item in frozen
            )
        ):
            raise ValueError("correction cells require two distinct supported candidates")
        addresses = tuple(
            int.from_bytes(
                sha256(b"candidate-address-v1\0" + bytes.fromhex(item.sha256)).digest()[:4],
                "little",
            )
            for item in frozen
        )
        if len(set(addresses)) != CANDIDATE_COUNT:
            raise ValueError("candidate cell addresses collided")
        evidence = {} if learned_evidence is None else dict(learned_evidence)
        action_sha256s = tuple(
            bank.action_sha256_for(item)
            for item in frozen
        )
        if evidence and set(evidence) != set(action_sha256s):
            raise ValueError("learned evidence must address the exact stable action bank")
        if evidence and (
            sum(value == (0, 0) for value in evidence.values()) != 1
            or sum(value == (0, 2) for value in evidence.values()) != 1
        ):
            raise ValueError("learned recall must inject one bounded physical preference")
        self._frontier_sha256 = frontier_sha256
        self._candidates = frozen
        self._excitatory = np.asarray(
            tuple(
                item.receipt_cell_support + evidence.get(action_sha256, (0, 0))[0]
                for item, action_sha256 in zip(frozen, action_sha256s, strict=True)
            ),
            dtype=np.uint8,
        )
        self._inhibitory = np.asarray(
            tuple(
                evidence.get(action_sha256, (0, 0))[1]
                for action_sha256 in action_sha256s
            ),
            dtype=np.uint8,
        )
        self._addresses = np.asarray(addresses, dtype="<u4")
        self._provenance = np.zeros(
            (ATTEMPT_LIMIT, PROVENANCE_ROW_BYTES),
            dtype=np.uint8,
        )
        self._candidate_races = 0
        self._replan_updates = 0
        self._filled_rows = 0
        self._steps = 0
        self._tier_sha256s: list[str] = []
        self._previous_winner: str | None = None
        self._selector = CompiledRestingGlifTierBeam.compile(
            candidate_count=CANDIDATE_COUNT + 1,
            step_count=RACE_TICKS,
            min_current_pa=CANDIDATE_CURRENT_BASE_PA,
            max_current_pa=CANDIDATE_CURRENT_MAX_PA,
            inhibition_current_pa=CANDIDATE_INHIBITION_PA,
            beam_width=1,
        )
        self._metadata = _canonical_json(
            {
                "schema": SCHEMA,
                "frontier": frontier_sha256,
                "candidates": tuple(
                    {
                        "candidate": item.sha256,
                        "action": action_sha256s[index],
                        "program": item.program_sha256,
                        "receipts": item.applied_relation_receipt_sha256s,
                        "support": item.receipt_cell_support,
                        "outputs": _encoding_vector_sha256(item.physical_outputs),
                    }
                    for index, item in enumerate(frozen)
                ),
                "learned_recall": learned_recall_sha256,
                "tonic_current": TONIC_CURRENT,
                "attempt_limit": ATTEMPT_LIMIT,
                "physical_selector": {
                    "cells": CANDIDATE_COUNT + 1,
                    "stop_cell": STOP_CELL,
                    "steps": RACE_TICKS,
                    "current_base_pa": CANDIDATE_CURRENT_BASE_PA,
                    "current_gain_pa": CANDIDATE_CURRENT_GAIN_PA,
                    "projection": self._selector.inhibition_projection_sha256,
                },
            }
        )
        if self.checkpoint_array_bytes != CHECKPOINT_ARRAY_BYTES:
            raise RuntimeError("candidate checkpoint array budget changed")

    @property
    def candidates(
        self,
    ) -> tuple[action_bank.RepositoryActionCandidateRecordV1, ...]:
        return self._candidates

    @property
    def checkpoint_array_bytes(self) -> int:
        return int(
            self._excitatory.nbytes
            + self._inhibitory.nbytes
            + self._addresses.nbytes
            + self._provenance.nbytes
        )

    @property
    def state(self) -> CorrectionCellStateV1:
        partial = CorrectionCellStateV1(
            status=PREPARED,
            frontier_sha256=self._frontier_sha256,
            candidate_sha256s=tuple(item.sha256 for item in self._candidates),
            receipt_cell_supports=tuple(item.receipt_cell_support for item in self._candidates),
            tonic_current=TONIC_CURRENT,
            excitatory_sha256=_array_sha256(self._excitatory),
            inhibitory_sha256=_array_sha256(self._inhibitory),
            address_sha256=_array_sha256(self._addresses),
            provenance_sha256=_array_sha256(self._provenance),
            physical_tier_sha256s=tuple(self._tier_sha256s),
            selector_projection_sha256=(self._selector.inhibition_projection_sha256),
            filled_provenance_rows=self._filled_rows,
            candidate_races=self._candidate_races,
            replan_updates=self._replan_updates,
            candidate_work_cell_steps=self._steps,
            trainable_state_bytes=int(self._excitatory.nbytes + self._inhibitory.nbytes),
            address_state_bytes=int(self._addresses.nbytes),
            provenance_state_bytes=int(self._provenance.nbytes),
            checkpoint_array_bytes=self.checkpoint_array_bytes,
            canonical_metadata_bytes=len(self._metadata),
            canonical_metadata_sha256=_bytes_sha256(self._metadata),
            sha256="",
        )
        payload = partial._asdict()
        payload.pop("sha256")
        return partial._replace(sha256=_digest(_STATE_NAMESPACE, payload))

    def _candidate_index(
        self,
        candidate: action_bank.RepositoryActionCandidateRecordV1,
    ) -> int:
        matches = tuple(index for index, item in enumerate(self._candidates) if item == candidate)
        if len(matches) != 1:
            raise ValueError("candidate is not in the authenticated cell bank")
        return matches[0]

    def address(
        self,
        candidate: action_bank.RepositoryActionCandidateRecordV1,
    ) -> int:
        return int(self._addresses[self._candidate_index(candidate)])

    def race(
        self,
        eligible_sha256s: Sequence[str],
    ) -> tuple[
        str,
        action_bank.RepositoryActionCandidateRecordV1 | None,
        str | None,
    ]:
        if set(eligible_sha256s) != {item.sha256 for item in self._candidates}:
            return STOP_ELIGIBILITY, None, None
        candidate_drive = (
            self._excitatory.astype(np.int16) + TONIC_CURRENT - self._inhibitory.astype(np.int16)
        )
        currents = np.empty(CANDIDATE_COUNT + 1, dtype=np.float64)
        currents[:CANDIDATE_COUNT] = (
            CANDIDATE_CURRENT_BASE_PA + CANDIDATE_CURRENT_GAIN_PA * candidate_drive
        )
        currents[STOP_CELL] = CANDIDATE_CURRENT_BASE_PA + CANDIDATE_CURRENT_GAIN_PA * TONIC_CURRENT
        if np.any(currents < CANDIDATE_CURRENT_BASE_PA) or np.any(
            currents > CANDIDATE_CURRENT_MAX_PA
        ):
            raise RuntimeError("candidate recall escaped the compiled current envelope")
        physical = self._selector.select(currents, stop_cell=STOP_CELL)
        tier_sha256 = _physical_tier_sha256(physical)
        self._tier_sha256s.append(tier_sha256)
        self._steps += (CANDIDATE_COUNT + 1) * RACE_TICKS
        self._candidate_races += 1
        if physical.status != RETAINED:
            return STOP_CANDIDATE_RACE, None, tier_sha256
        if len(physical.retained_candidate_cells) != 1:
            return STOP_CANDIDATE_RACE, None, tier_sha256
        winner_index = physical.retained_candidate_cells[0]
        if not 0 <= winner_index < CANDIDATE_COUNT:
            return STOP_CANDIDATE_RACE, None, tier_sha256
        winner = self._candidates[winner_index]
        if self._previous_winner == winner.sha256:
            return STOP_REPEATED_WINNER, None, tier_sha256
        self._previous_winner = winner.sha256
        return PREPARED, winner, tier_sha256

    def record_action(
        self,
        attempt: int,
        candidate: action_bank.RepositoryActionCandidateRecordV1,
        outcome_key: str,
        *,
        update_inhibitory: bool,
    ) -> None:
        if attempt != self._filled_rows + 1 or not 1 <= attempt <= ATTEMPT_LIMIT:
            raise ValueError("candidate provenance attempt order changed")
        index = self._candidate_index(candidate)
        action_code = (
            1
            if outcome_key == action_bank.OUTCOME_COMMIT
            else 2
            if outcome_key == action_bank.OUTCOME_REVISE
            else 3
        )
        row = (
            int(self._addresses[index]).to_bytes(4, "little")
            + bytes.fromhex(candidate.sha256)
            + bytes((action_code,))
        )
        if len(row) != PROVENANCE_ROW_BYTES:
            raise RuntimeError("candidate provenance row budget changed")
        self._provenance[attempt - 1, :] = np.frombuffer(row, dtype=np.uint8)
        self._filled_rows += 1
        if outcome_key == action_bank.OUTCOME_REVISE and update_inhibitory:
            self._inhibitory.fill(0)
            self._inhibitory[index] = np.uint8(2)
            self._replan_updates += 1


def _consequence(
    *,
    attempt: int,
    candidate_sha256: str,
    selected_source_sha256: str,
    passed_cell: int,
    stdout_sha256: str,
    stderr_sha256: str,
) -> RepositoryConsequenceV1:
    payload = {
        "attempt": attempt,
        "candidate": candidate_sha256,
        "source": selected_source_sha256,
        "passed_cell": passed_cell,
        "stdout": stdout_sha256,
        "stderr": stderr_sha256,
    }
    return RepositoryConsequenceV1(
        attempt=attempt,
        candidate_sha256=candidate_sha256,
        selected_source_sha256=selected_source_sha256,
        passed_cell=passed_cell,
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
        sha256=_digest(_CONSEQUENCE_NAMESPACE, payload),
    )


def _authenticate_consequence(value: RepositoryConsequenceV1) -> bool:
    expected = _consequence(
        attempt=value.attempt,
        candidate_sha256=value.candidate_sha256,
        selected_source_sha256=value.selected_source_sha256,
        passed_cell=value.passed_cell,
        stdout_sha256=value.stdout_sha256,
        stderr_sha256=value.stderr_sha256,
    )
    return value == expected and value.passed_cell in (0, 1)


def _attempt(
    *,
    attempt: int,
    candidate: action_bank.RepositoryActionCandidateRecordV1,
    address: int,
    witness: action_bank.RepositoryPatchWitnessV1,
    selected_source_sha256: str,
    exit_code: int,
    stdout_sha256: str,
    stderr_sha256: str,
    consequence_sha256: str,
    action_key: str,
    rolled_back: bool,
    committed: bool,
) -> AutonomousCorrectionAttemptV1:
    payload = {
        "attempt": attempt,
        "candidate": candidate.sha256,
        "address": address,
        "support": candidate.receipt_cell_support,
        "program": witness.program_sha256,
        "witness": witness.sha256,
        "source": selected_source_sha256,
        "exit": exit_code,
        "stdout": stdout_sha256,
        "stderr": stderr_sha256,
        "consequence": consequence_sha256,
        "action": action_key,
        "rolled_back": rolled_back,
        "committed": committed,
    }
    return AutonomousCorrectionAttemptV1(
        attempt=attempt,
        candidate_sha256=candidate.sha256,
        candidate_address=address,
        receipt_cell_support=candidate.receipt_cell_support,
        program_sha256=witness.program_sha256,
        witness_sha256=witness.sha256,
        selected_source_sha256=selected_source_sha256,
        repository_test_exit_code=exit_code,
        repository_test_stdout_sha256=stdout_sha256,
        repository_test_stderr_sha256=stderr_sha256,
        consequence_sha256=consequence_sha256,
        consequence_action_key=action_key,
        rolled_back=rolled_back,
        committed=committed,
        sha256=_digest(_ATTEMPT_NAMESPACE, payload),
    )


def _authenticate_attempt(value: AutonomousCorrectionAttemptV1) -> bool:
    if type(value) is not AutonomousCorrectionAttemptV1:
        return False
    payload = {
        "attempt": value.attempt,
        "candidate": value.candidate_sha256,
        "address": value.candidate_address,
        "support": value.receipt_cell_support,
        "program": value.program_sha256,
        "witness": value.witness_sha256,
        "source": value.selected_source_sha256,
        "exit": value.repository_test_exit_code,
        "stdout": value.repository_test_stdout_sha256,
        "stderr": value.repository_test_stderr_sha256,
        "consequence": value.consequence_sha256,
        "action": value.consequence_action_key,
        "rolled_back": value.rolled_back,
        "committed": value.committed,
    }
    return (
        value.sha256 == _digest(_ATTEMPT_NAMESPACE, payload)
        and value.attempt in (1, 2)
        and value.receipt_cell_support > 0
    )


def _result_trajectory_sha256(
    *,
    status: str,
    prepared_sha256: str,
    initial_checkpoint_sha256: str | None,
    final_checkpoint_sha256: str | None,
    desired_sha256: str | None,
    eligibility_sha256: str | None,
    attempts: Sequence[AutonomousCorrectionAttemptV1],
    source_before_sha256: str,
    source_after_sha256: str,
    source_context_sha256: str,
    work: AutonomousCorrectionWorkV1,
    claim_boundary: str,
) -> str:
    return _digest(
        _RESULT_TRAJECTORY_NAMESPACE,
        {
            "status": status,
            "prepared": prepared_sha256,
            "initial_checkpoint": initial_checkpoint_sha256,
            "final_checkpoint": final_checkpoint_sha256,
            "desired": desired_sha256,
            "eligibility": eligibility_sha256,
            "attempts": tuple(item.sha256 for item in attempts),
            "source_before": source_before_sha256,
            "source_after": source_after_sha256,
            "source_context": source_context_sha256,
            "work": work._asdict(),
            "claim": claim_boundary,
        },
    )


def _empty_work() -> dict[str, int]:
    return {
        "source_reads": 0,
        "desired_manifest_reads": 0,
        "eligibility_races": 0,
        "candidate_races": 0,
        "source_materializations": 0,
        "unselected_source_materializations": 0,
        "selected_patch_sources": 0,
        "patch_writes": 0,
        "repository_consequence_roots": 0,
        "consequence_policy_races": 0,
        "repository_tests": 0,
        "unselected_repository_tests": 0,
        "revisions": 0,
        "replan_updates": 0,
        "reverts": 0,
        "rollback_verifications": 0,
        "commits": 0,
        "candidate_work_cell_steps": 0,
    }


def _read_visible_expected(path: Path) -> object:
    record = json.loads(path.read_text(encoding="utf-8"))
    if type(record) is not dict or set(record) != {"expected"}:
        raise ValueError("expected manifest schema changed")
    return record["expected"]


def _child_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("repository paths must be non-empty relative paths")
    child = (root / relative).resolve()
    if child == root or root not in child.parents:
        raise ValueError("repository path escaped its authenticated root")
    return child


class PreparedAutonomousCorrectionV1:
    __slots__ = (
        "_bank",
        "_cells",
        "_expected_path",
        "_frontier",
        "_initial_checkpoint_sha256",
        "_original_source",
        "_repo_root",
        "_source_context",
        "_source_path",
        "_spec",
        "state",
    )

    def __init__(
        self,
        *,
        bank: action_bank.FrozenRepositoryActionBankV1,
        frontier: action_bank.RepositoryActionFrontierV1,
        spec: AutonomousCorrectionSpecV1,
        repo_root: Path,
        source_path: Path,
        expected_path: Path,
        original_source: bytes,
        source_context: source_views.RepositorySourceContextV1 | None,
        cells: _CorrectionCellsV1 | None,
        state: PreparedAutonomousCorrectionStateV1,
    ) -> None:
        self._bank = bank
        self._frontier = frontier
        self._spec = spec
        self._repo_root = repo_root
        self._source_path = source_path
        self._expected_path = expected_path
        self._original_source = original_source
        self._source_context = source_context
        self._cells = cells
        self._initial_checkpoint_sha256 = None if cells is None else cells.state.sha256
        self.state = state

    def _work(self, counters: dict[str, int]) -> AutonomousCorrectionWorkV1:
        if self._cells is not None:
            counters["candidate_work_cell_steps"] = self._cells.state.candidate_work_cell_steps
            counters["replan_updates"] = self._cells.state.replan_updates
        return AutonomousCorrectionWorkV1(**counters)

    def _result(
        self,
        *,
        status: str,
        counters: dict[str, int],
        attempts: Sequence[AutonomousCorrectionAttemptV1] = (),
        desired_sha256: str | None = None,
        eligibility_sha256: str | None = None,
        source_after: bytes | None = None,
    ) -> AutonomousCorrectionResultV1:
        final_source = self._original_source if source_after is None else source_after
        work = self._work(counters)
        final_checkpoint = None if self._cells is None else self._cells.state
        attempt_tuple = tuple(attempts)
        source_before_sha256 = _bytes_sha256(self._original_source)
        source_after_sha256 = _bytes_sha256(final_source)
        source_context_sha256 = (
            "0" * 64 if self._source_context is None else self._source_context.sha256
        )
        trajectory_sha256 = _result_trajectory_sha256(
            status=status,
            prepared_sha256=self.state.sha256,
            initial_checkpoint_sha256=self._initial_checkpoint_sha256,
            final_checkpoint_sha256=(
                None if final_checkpoint is None else final_checkpoint.sha256
            ),
            desired_sha256=desired_sha256,
            eligibility_sha256=eligibility_sha256,
            attempts=attempt_tuple,
            source_before_sha256=source_before_sha256,
            source_after_sha256=source_after_sha256,
            source_context_sha256=source_context_sha256,
            work=work,
            claim_boundary=CLAIM_BOUNDARY,
        )
        repository_actions: tuple[
            continual_memory.PhysicalRepositoryActionLessonV1, ...
        ] = ()
        if status == COMMITTED:
            if self._cells is None or self._source_context is None:
                raise RuntimeError("committed correction lost its authenticated source cells")
            self._source_context.authenticate()
            candidates = {item.sha256: item for item in self._cells.candidates}
            consequences: list[
                continual_memory._RepositoryActionConsequenceV1
            ] = []
            for attempt in attempt_tuple:
                candidate = candidates.get(attempt.candidate_sha256)
                if candidate is None or candidate.program_sha256 != attempt.program_sha256:
                    raise RuntimeError("repository action candidate provenance changed")
                if attempt.consequence_action_key == action_bank.OUTCOME_COMMIT:
                    outcome_key = continual_memory.OUTCOME_COMMIT
                elif attempt.consequence_action_key == action_bank.OUTCOME_REVISE:
                    outcome_key = continual_memory.OUTCOME_REVISE
                else:
                    raise RuntimeError("repository consequence action is not learnable")
                consequences.append(
                    continual_memory._RepositoryActionConsequenceV1(
                        candidate_sha256=candidate.sha256,
                        applied_relation_receipt_sha256s=(
                            candidate.applied_relation_receipt_sha256s
                        ),
                        program_sha256=candidate.program_sha256,
                        attempt_sha256=attempt.sha256,
                        consequence_sha256=attempt.consequence_sha256,
                        passed_cell=int(attempt.repository_test_exit_code == 0),
                        outcome_key=outcome_key,
                        rolled_back=attempt.rolled_back,
                        committed=attempt.committed,
                    )
                )
            result_sha256, repository_actions = (
                continual_memory._seal_physical_repository_action_lessons_v1(
                    controller_trajectory_sha256=trajectory_sha256,
                    controller_status=status,
                    bank_checkpoint=self._bank,
                    source_context=self._source_context,
                    consequences=tuple(consequences),
                )
            )
        else:
            result_sha256 = _digest(
                _RESULT_NAMESPACE,
                {"trajectory": trajectory_sha256, "lesson_bodies": ()},
            )
        result = AutonomousCorrectionResultV1(
            status=status,
            prepared_sha256=self.state.sha256,
            initial_checkpoint_sha256=self._initial_checkpoint_sha256,
            final_checkpoint=final_checkpoint,
            desired_vector_sha256=desired_sha256,
            eligibility_race_sha256=eligibility_sha256,
            attempts=attempt_tuple,
            repository_actions=repository_actions,
            source_before_sha256=source_before_sha256,
            source_after_sha256=source_after_sha256,
            source_context_sha256=source_context_sha256,
            work=work,
            claim_boundary=CLAIM_BOUNDARY,
            trajectory_sha256=trajectory_sha256,
            sha256=result_sha256,
        )
        result.authenticate()
        return result

    def _rollback(self, counters: dict[str, int]) -> tuple[bool, bytes]:
        try:
            self._source_path.write_bytes(self._original_source)
            counters["reverts"] += 1
            restored = self._source_path.read_bytes()
            counters["source_reads"] += 1
            counters["rollback_verifications"] += 1
        except OSError:
            return False, b""
        return restored == self._original_source, restored

    def repair(
        self,
        *,
        revision_cell_gate: bool = True,
        replan_cell_gate: bool = True,
        inhibitory_update_gate: bool = True,
        forge_consequence_attempt: int | None = None,
    ) -> AutonomousCorrectionResultV1:
        counters = _empty_work()
        counters["source_reads"] = self.state.source_reads_during_prepare
        attempts: list[AutonomousCorrectionAttemptV1] = []
        if self.state.status != PREPARED or self._cells is None:
            return self._result(status=self.state.status, counters=counters)
        try:
            current_source = self._source_path.read_bytes()
        except OSError:
            current_source = b""
        counters["source_reads"] += 1
        if current_source != self._original_source:
            return self._result(
                status=STOP_SOURCE_DRIFT,
                counters=counters,
                source_after=current_source,
            )
        try:
            raw_expected = _read_visible_expected(self._expected_path)
            desired = self._frontier.encode_visible_expected(raw_expected)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            counters["desired_manifest_reads"] = 1
            return self._result(status=STOP_ADAPTER_INPUT, counters=counters)
        counters["desired_manifest_reads"] = 1
        desired_sha256 = _encoding_vector_sha256(desired)
        candidates = self._cells.candidates
        eligibility = semantic_race.select_exact_semantic_vector(
            tuple(item.sha256 for item in candidates),
            tuple(item.physical_outputs for item in candidates),
            desired,
        )
        counters["eligibility_races"] = 1
        eligible = tuple(eligibility.exact_winner_keys)
        if (
            eligibility.status != semantic_race.STOP_MULTIPLE_EXACT
            or len(eligible) != CANDIDATE_COUNT
            or set(eligible) != {item.sha256 for item in candidates}
        ):
            return self._result(
                status=STOP_ELIGIBILITY,
                counters=counters,
                desired_sha256=desired_sha256,
                eligibility_sha256=eligibility.sha256,
            )
        for attempt_number in range(1, ATTEMPT_LIMIT + 1):
            race_status, candidate, _ = self._cells.race(eligible)
            counters["candidate_races"] += 1
            if race_status != PREPARED or candidate is None:
                return self._result(
                    status=race_status,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=self._original_source,
                )
            try:
                witness = self._frontier.materialize_selected(candidate)
            except (RuntimeError, TypeError, ValueError):
                return self._result(
                    status=STOP_MATERIALIZATION,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                )
            try:
                if type(witness) is not action_bank.RepositoryPatchWitnessV1:
                    raise TypeError("frontier returned a non-neutral patch witness")
                witness.authenticate()
                if (
                    witness.frontier_sha256 != self._bank.frontier_sha256
                    or witness.candidate_sha256 != candidate.sha256
                    or witness.program_sha256 != candidate.program_sha256
                    or witness.applied_relation_receipt_sha256s
                    != candidate.applied_relation_receipt_sha256s
                ):
                    raise ValueError("patch witness is outside the selected action")
            except (RuntimeError, TypeError, ValueError):
                return self._result(
                    status=STOP_MATERIALIZATION,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                )
            counters["source_materializations"] += 1
            counters["selected_patch_sources"] += 1
            selected_source = witness.source.encode("utf-8")
            selected_source_sha256 = _bytes_sha256(selected_source)
            try:
                self._source_path.write_bytes(selected_source)
            except OSError:
                return self._result(
                    status=STOP_PATCH_WRITE,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                )
            counters["patch_writes"] += 1
            try:
                completed = subprocess.run(
                    self._spec.test_command,
                    cwd=self._repo_root,
                    capture_output=True,
                    check=False,
                    timeout=self._spec.test_timeout_seconds,
                )
                exit_code = int(completed.returncode)
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                exit_code = -1
                stdout = exc.stdout or b""
                stderr = exc.stderr or b""
            except OSError as exc:
                exit_code = -1
                stdout = b""
                stderr = str(exc).encode("utf-8", errors="replace")
            counters["repository_tests"] += 1
            stdout_sha256 = _bytes_sha256(stdout)
            stderr_sha256 = _bytes_sha256(stderr)
            try:
                observed_after_test = self._source_path.read_bytes()
            except OSError:
                observed_after_test = b""
            counters["source_reads"] += 1
            if observed_after_test != selected_source:
                exact, restored = self._rollback(counters)
                status = STOP_SOURCE_DRIFT if exact else STOP_ROLLBACK
                return self._result(
                    status=status,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
            consequence = _consequence(
                attempt=attempt_number,
                candidate_sha256=candidate.sha256,
                selected_source_sha256=selected_source_sha256,
                passed_cell=int(exit_code == 0),
                stdout_sha256=stdout_sha256,
                stderr_sha256=stderr_sha256,
            )
            counters["repository_consequence_roots"] += 1
            if forge_consequence_attempt == attempt_number:
                consequence = consequence._replace(sha256="0" * 64)
            if not _authenticate_consequence(consequence):
                exact, restored = self._rollback(counters)
                status = STOP_FORGED_CONSEQUENCE if exact else STOP_ROLLBACK
                return self._result(
                    status=status,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
            try:
                decision = self._frontier.race_repository_consequence(
                    consequence.passed_cell
                )
                if type(decision) is not action_bank.RepositoryConsequenceDecisionV1:
                    raise TypeError("frontier returned a non-neutral consequence")
                decision.authenticate()
                if (
                    decision.policy_checkpoint_sha256
                    != self._bank.policy_checkpoint_sha256
                    or decision.passed_cell != consequence.passed_cell
                ):
                    raise ValueError("consequence decision is outside the action bank")
            except (RuntimeError, TypeError, ValueError):
                exact, restored = self._rollback(counters)
                return self._result(
                    status=STOP_CONSEQUENCE_POLICY if exact else STOP_ROLLBACK,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
            counters["consequence_policy_races"] += 1
            action_key = decision.outcome_key
            if exit_code == 0 and action_key == action_bank.OUTCOME_COMMIT:
                self._cells.record_action(
                    attempt_number,
                    candidate,
                    action_key,
                    update_inhibitory=False,
                )
                counters["commits"] += 1
                attempts.append(
                    _attempt(
                        attempt=attempt_number,
                        candidate=candidate,
                        address=self._cells.address(candidate),
                        witness=witness,
                        selected_source_sha256=selected_source_sha256,
                        exit_code=exit_code,
                        stdout_sha256=stdout_sha256,
                        stderr_sha256=stderr_sha256,
                        consequence_sha256=consequence.sha256,
                        action_key=action_key,
                        rolled_back=False,
                        committed=True,
                    )
                )
                return self._result(
                    status=COMMITTED,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=selected_source,
                )
            exact, restored = self._rollback(counters)
            if not exact:
                return self._result(
                    status=STOP_ROLLBACK,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
            attempts.append(
                _attempt(
                    attempt=attempt_number,
                    candidate=candidate,
                    address=self._cells.address(candidate),
                    witness=witness,
                    selected_source_sha256=selected_source_sha256,
                    exit_code=exit_code,
                    stdout_sha256=stdout_sha256,
                    stderr_sha256=stderr_sha256,
                    consequence_sha256=consequence.sha256,
                    action_key=action_key,
                    rolled_back=True,
                    committed=False,
                )
            )
            if exit_code == 0 or action_key != action_bank.OUTCOME_REVISE:
                return self._result(
                    status=STOP_CONSEQUENCE_POLICY,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
            if not revision_cell_gate:
                return self._result(
                    status=STOP_REVISION_GATE,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
            counters["revisions"] += 1
            self._cells.record_action(
                attempt_number,
                candidate,
                action_key,
                update_inhibitory=replan_cell_gate and inhibitory_update_gate,
            )
            if not replan_cell_gate:
                return self._result(
                    status=STOP_REPLAN_GATE,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
            if attempt_number == ATTEMPT_LIMIT:
                return self._result(
                    status=STOP_ATTEMPT_BUDGET,
                    counters=counters,
                    attempts=attempts,
                    desired_sha256=desired_sha256,
                    eligibility_sha256=eligibility.sha256,
                    source_after=restored,
                )
        raise RuntimeError("bounded correction loop escaped its attempt budget")


def _prepared_state(
    *,
    status: str,
    bank: action_bank.FrozenRepositoryActionBankV1,
    spec: AutonomousCorrectionSpecV1,
    observed_source_sha256: str,
    source_context_sha256: str,
    source_context_frame_sha256: str,
    cells: _CorrectionCellsV1 | None,
    source_reads: int,
    learned_recall_sha256: str | None,
) -> PreparedAutonomousCorrectionStateV1:
    candidate_checkpoint = None if cells is None else cells.state.sha256
    partial = PreparedAutonomousCorrectionStateV1(
        status=status,
        frontier_sha256=bank.frontier_sha256,
        policy_checkpoint_sha256=bank.policy_checkpoint_sha256,
        root_program_sha256=spec.root_program_sha256,
        root_source_sha256=observed_source_sha256,
        source_context_sha256=source_context_sha256,
        source_context_frame_sha256=source_context_frame_sha256,
        candidate_checkpoint_sha256=candidate_checkpoint,
        learned_recall_sha256=learned_recall_sha256,
        candidate_count=0 if cells is None else len(cells.candidates),
        source_reads_during_prepare=source_reads,
        desired_reads_during_prepare=0,
        source_materializations_during_prepare=0,
        patch_writes_during_prepare=0,
        repository_tests_during_prepare=0,
        sha256="",
    )
    payload = partial._asdict()
    payload.pop("sha256")
    return partial._replace(sha256=_digest(_PREPARED_NAMESPACE, payload))


def prepare_autonomous_correction_v1(
    frontier: action_bank.RepositoryActionFrontierV1,
    spec: AutonomousCorrectionSpecV1,
    *,
    candidate_records: Sequence[action_bank.RepositoryActionCandidateRecordV1]
    | None = None,
    relation_support_gate: bool = True,
    action_memory: memory_v2.FrozenRepositoryActionMemoryV2 | None = None,
    action_observation: memory_v2.RepositoryActionObservationV2 | None = None,
    memory_recall_gate: bool = True,
    memory_context_gate: bool = True,
    memory_candidate_gate: bool = True,
    memory_structural_role_gate: bool = True,
    memory_lexical_factor_gate: bool = True,
) -> PreparedAutonomousCorrectionV1:
    """Attest one frozen frontier and allocate the 86-byte candidate checkpoint."""
    bank = frontier.repository_action_bank
    bank.authenticate()
    repo_root = spec.repo_root.resolve(strict=True)
    if not repo_root.is_dir() or not spec.test_command:
        raise ValueError("repository root and test command are required")
    source_path = _child_path(repo_root, spec.source_relative_path)
    expected_path = _child_path(repo_root, spec.expected_manifest_relative_path)
    try:
        original_source = source_path.read_bytes()
    except OSError:
        original_source = b""
    observed_source_sha256 = _bytes_sha256(original_source)
    authentic_records = bank.records
    supplied_records = authentic_records if candidate_records is None else tuple(candidate_records)
    canonical = {item.sha256: item for item in authentic_records}
    records_authentic = (
        len(supplied_records) == CANDIDATE_COUNT
        and len(canonical) == CANDIDATE_COUNT
        and all(
            type(item) is action_bank.RepositoryActionCandidateRecordV1
            for item in supplied_records
        )
        and all(canonical.get(item.sha256) == item for item in supplied_records)
        and len({item.sha256 for item in supplied_records}) == CANDIDATE_COUNT
    )
    attested = (
        bank.root_program_sha256 == spec.root_program_sha256
        and observed_source_sha256 == spec.root_source_sha256
        and records_authentic
    )
    if any(
        type(gate) is not bool
        for gate in (
            relation_support_gate,
            memory_recall_gate,
            memory_context_gate,
            memory_candidate_gate,
            memory_structural_role_gate,
            memory_lexical_factor_gate,
        )
    ):
        raise TypeError("correction preparation gates must be exact booleans")
    if (
        action_memory is not None
        and type(action_memory) is not memory_v2.FrozenRepositoryActionMemoryV2
    ):
        raise TypeError(
            "action_memory must be one exact frozen V2 repository memory"
        )
    if (
        action_observation is not None
        and type(action_observation) is not memory_v2.RepositoryActionObservationV2
    ):
        raise TypeError(
            "action_observation must be one exact V2 repository observation"
        )
    if (action_memory is None) != (action_observation is None):
        raise TypeError("action_memory and action_observation must be supplied together")
    source_context_frame_sha256 = "0" * 64
    source_context_sha256 = "0" * 64
    source_context: source_views.RepositorySourceContextV1 | None = None
    learned_recall = None
    learned_evidence: dict[str, tuple[int, int]] | None = None
    if attested and action_observation is None:
        source_context = source_views.project_attested_repository_source_v1(
            original_source
        )
        source_context_frame_sha256 = source_context.frame_sha256
        source_context_sha256 = source_context.sha256
    elif attested and action_observation is not None:
        action_observation.authenticate()
        if (
            action_observation.source_sha256 == observed_source_sha256
            and action_observation.source_views.exact_context.source_sha256
            == observed_source_sha256
        ):
            source_context = action_observation.source_views.exact_context
            source_context_frame_sha256 = source_context.frame_sha256
            source_context_sha256 = source_context.sha256
            learned_recall = action_memory.recall_attested_observation(
                action_observation,
                bank,
                candidate_records=supplied_records,
                recall_gate=memory_recall_gate,
                exact_context_gate=memory_context_gate,
                exact_candidate_gate=memory_candidate_gate,
                structural_role_gate=memory_structural_role_gate,
                lexical_factor_gate=memory_lexical_factor_gate,
            )
            learned_recall.authenticate()
            if (
                learned_recall.status == memory_v2.RECALLED
                and learned_recall.memory_checkpoint_sha256
                == action_memory.checkpoint_sha256
                and learned_recall.policy_checkpoint_sha256
                == bank.policy_checkpoint_sha256
                and learned_recall.observation_sha256 == action_observation.sha256
                and learned_recall.source_sha256 == observed_source_sha256
                and learned_recall.source_context_frame_sha256
                == source_context_frame_sha256
                and learned_recall.source_context_trace_cells
                == source_context.trace_cells
                and learned_recall.frontier_sha256 == bank.frontier_sha256
                and learned_recall.action_bank_sha256s == bank.action_sha256s
                and learned_recall.candidates == supplied_records
                and learned_recall.selected_candidate_action_sha256 is not None
                and learned_recall.selected_candidate_action_sha256
                in bank.action_sha256s
            ):
                learned_evidence = {
                    bank.action_sha256_for(item): (
                        0,
                        0
                        if bank.action_sha256_for(item)
                        == learned_recall.selected_candidate_action_sha256
                        else 2,
                    )
                    for item in supplied_records
                }
    cells: _CorrectionCellsV1 | None = None
    if not attested:
        status = STOP_ATTESTATION
    elif action_memory is not None and learned_evidence is None:
        status = STOP_LEARNED_RECALL
    elif not relation_support_gate:
        status = STOP_RELATION_SUPPORT
    else:
        try:
            cells = _CorrectionCellsV1(
                bank.frontier_sha256,
                bank,
                supplied_records,
                learned_evidence=learned_evidence,
                learned_recall_sha256=(
                    None if learned_recall is None else learned_recall.sha256
                ),
            )
        except ValueError:
            status = STOP_RELATION_SUPPORT
        else:
            status = PREPARED
    state = _prepared_state(
        status=status,
        bank=bank,
        spec=spec,
        observed_source_sha256=observed_source_sha256,
        source_context_sha256=source_context_sha256,
        source_context_frame_sha256=source_context_frame_sha256,
        cells=cells,
        source_reads=1,
        learned_recall_sha256=(
            None if learned_recall is None else learned_recall.sha256
        ),
    )
    return PreparedAutonomousCorrectionV1(
        bank=bank,
        frontier=frontier,
        spec=spec,
        repo_root=repo_root,
        source_path=source_path,
        expected_path=expected_path,
        original_source=original_source,
        source_context=source_context,
        cells=cells,
        state=state,
    )
