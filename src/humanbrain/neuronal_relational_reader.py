"""Learned two-hop reading over complete opaque document observations.

The public learner sees only successful raw queries, raw documents, observed
answer bytes, and provenance.  It discovers one recurring query frame and the
document sentence frames that repeatedly carried each hop.  Endpoint values
are deliberately discarded: only literal surface synapses, argument
orientation, and support remain in the checkpoint.

Inference enumerates every ordered pair of live lexical occurrences in every
sentence.  Fitted simulated-cell races detect learned frames, join the query
filler to a first edge by equality, recurrently join that edge to a second one,
and finally race every live lexical occurrence against a tonic STOP cell.  The
host prepares the complete candidate sheet but does not select an answer.

This is a narrow relational-reading mechanism, not broad language
understanding.  In particular, it currently learns one fixed query surface and
requires a unique two-document proof in each successful training observation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.opaque_document_event_tape import (
    OpaqueDocument,
    OpaqueDocumentEventTape,
    OpaqueSentenceOccurrence,
    OpaqueTokenOccurrence,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace

__all__ = [
    "NeuronalRelationalReader",
    "RelationalRaceTrace",
    "RelationalReaderAblation",
    "RelationalReaderTrainingSummary",
    "RelationalReadingDecision",
    "RelationalReadingSuccess",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_DRIVE_SCORE = 2
_TONIC_SCORE = 1
_MIN_SUCCESSES = 4
_MAX_SUCCESSES = 64
_MIN_DOCUMENTS = 2
_MAX_DOCUMENTS = 4
_MAX_TOKENS = 512
_MAX_VALUE_BYTES = 4096

_CHECKPOINT_NAMESPACE = b"humanbrain.neuronal-relational-reader.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.neuronal-relational-reader.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.neuronal-relational-reader.race-v1\0"
_DECISION_NAMESPACE = b"humanbrain.neuronal-relational-reader.decision-v1\0"
_PROVENANCE_NAMESPACE = b"humanbrain.neuronal-relational-reader.provenance-v1\0"
_CONSTRUCTION_TOKEN = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _require_bytes(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be non-empty bytes")
    if len(value) > _MAX_VALUE_BYTES:
        raise ValueError(f"{name} exceeds the byte bound")
    return value


def _require_documents(value: object) -> tuple[OpaqueDocument, ...]:
    if type(value) is not tuple:
        raise TypeError("documents must be a tuple")
    if not _MIN_DOCUMENTS <= len(value) <= _MAX_DOCUMENTS:
        raise ValueError("documents must contain two to four values")
    if any(not isinstance(document, OpaqueDocument) for document in value):
        raise TypeError("documents must contain OpaqueDocument values")
    return value


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(_RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (voltage - threshold)
    currents: list[float] = []
    for score in range(_DRIVE_SCORE + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("relational-reader fitted-GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, result in enumerate(controls)
    ):
        raise RuntimeError("relational-reader fitted-GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class RelationalReaderAblation(StrEnum):
    NONE = "none"
    EQUALITY_OFF = "equality_off"
    FRAME_BINDING_OFF = "frame_binding_off"
    RECURRENCE_OFF = "recurrence_off"
    MOTOR_OFF = "motor_off"


@dataclass(frozen=True, slots=True)
class RelationalReadingSuccess:
    """One successful raw reading observation.

    No relation labels, spans, candidates, parsed facts, or answer locations
    are accepted.  ``observed_answer`` must be one complete live lexical value.
    """

    query: bytes
    documents: tuple[OpaqueDocument, ...]
    observed_answer: bytes
    provenance_id: bytes

    def __post_init__(self) -> None:
        _require_bytes(self.query, "query")
        _require_documents(self.documents)
        _require_bytes(self.observed_answer, "observed_answer")
        _require_bytes(self.provenance_id, "provenance_id")


@dataclass(frozen=True, slots=True)
class RelationalRaceTrace:
    population: str
    cell_count: int
    cell_scores_sha256: str
    spike_step: int
    simultaneous_spike_cells: int
    stopped: bool
    stop_reason: str | None
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RelationalReaderTrainingSummary:
    success_count: int
    hop_count: int
    query_literal_synapse_count: int
    surface_frame_count: int
    forward_surface_count: int
    reverse_surface_count: int
    minimum_surface_support: int
    retained_concrete_endpoint_count: int
    provenance_root_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class RelationalReadingDecision:
    decision: str
    answer: bytes | None
    answer_occurrence_event: bytes | None
    completed_hops: int
    stop_stage: str | None
    stop_reason: str | None
    races: tuple[RelationalRaceTrace, ...]
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _QueryFrame:
    sentence_lengths: tuple[int, ...]
    subject_flat_index: int
    literal_events: tuple[bytes | None, ...]
    support_ticks: int

    def matches(self, tokens: tuple[OpaqueTokenOccurrence, ...]) -> bool:
        if len(tokens) != len(self.literal_events):
            return False
        return all(
            expected is None or occurrence.identity_event == expected
            for occurrence, expected in zip(tokens, self.literal_events, strict=True)
        )


@dataclass(frozen=True, slots=True)
class _SurfaceFrame:
    hop: int
    token_count: int
    source_token_index: int
    destination_token_index: int
    literal_events: tuple[bytes | None, ...]
    support_ticks: int

    def matches(self, edge: _EdgeCandidate) -> bool:
        if (
            len(edge.sentence_tokens) != self.token_count
            or edge.source_token_index != self.source_token_index
            or edge.destination_token_index != self.destination_token_index
        ):
            return False
        return all(
            expected is None or occurrence.identity_event == expected
            for occurrence, expected in zip(
                edge.sentence_tokens,
                self.literal_events,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class _Schema:
    query_frame: _QueryFrame
    hop_frames: tuple[tuple[_SurfaceFrame, ...], tuple[_SurfaceFrame, ...]]


@dataclass(frozen=True, slots=True)
class _EdgeCandidate:
    document_index: int
    sentence_index: int
    source_token_index: int
    destination_token_index: int
    source: OpaqueTokenOccurrence
    destination: OpaqueTokenOccurrence
    sentence_tokens: tuple[OpaqueTokenOccurrence, ...]


@dataclass(frozen=True, slots=True)
class _TrainingPath:
    first: _EdgeCandidate
    second: _EdgeCandidate


@dataclass(frozen=True, slots=True)
class _RaceDecision:
    column: int | None
    trace: RelationalRaceTrace


def _trace(
    *,
    population: str,
    scores: tuple[int, ...],
    spike_step: int,
    simultaneous_spike_cells: int,
    stopped: bool,
    stop_reason: str | None,
    physical_cell_steps: int,
) -> RelationalRaceTrace:
    score_root = _digest(_RACE_NAMESPACE, {"cell_scores": scores})
    record = {
        "cell_count": len(scores),
        "cell_scores_sha256": score_root,
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "simultaneous_spike_cells": simultaneous_spike_cells,
        "spike_step": spike_step,
        "stop_reason": stop_reason,
        "stopped": stopped,
    }
    return RelationalRaceTrace(**record, sha256=_digest(_RACE_NAMESPACE, record))


def _physical_choice(*, population: str, scores: tuple[int, ...]) -> _RaceDecision:
    if not scores or any(score not in (0, _DRIVE_SCORE) for score in scores):
        raise ValueError("choice scores must be a non-empty binary drive vector")
    all_scores = (*scores, _TONIC_SCORE)
    currents = _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(all_scores) * _RACE_STEPS
    stopped = False
    stop_reason: str | None = None
    selected: int | None = None
    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_TONIC_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("relational-reader collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        stopped = True
        stop_reason = "collision"
    elif result.winner_cell == len(scores):
        stopped = True
        stop_reason = "tonic_stop"
    else:
        selected = result.winner_cell
    return _RaceDecision(
        column=selected,
        trace=_trace(
            population=population,
            scores=all_scores,
            spike_step=result.spike_step,
            simultaneous_spike_cells=result.simultaneous_spike_candidates,
            stopped=stopped,
            stop_reason=stop_reason,
            physical_cell_steps=physical_cell_steps,
        ),
    )


def _physical_detector(*, population: str, matched: bool) -> tuple[bool, RelationalRaceTrace]:
    scores = (_DRIVE_SCORE if matched else 0, _TONIC_SCORE)
    currents = _SCORE_CURRENTS[np.asarray(scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    if result.simultaneous_spike_candidates != 1:
        raise RuntimeError("relational-reader detector race ceased to be unique")
    return result.winner_cell == 0, _trace(
        population=population,
        scores=scores,
        spike_step=result.spike_step,
        simultaneous_spike_cells=result.simultaneous_spike_candidates,
        stopped=False,
        stop_reason=None,
        physical_cell_steps=len(scores) * _RACE_STEPS,
    )


def _query_tokens(tape: OpaqueDocumentEventTape) -> tuple[OpaqueTokenOccurrence, ...]:
    return tuple(
        occurrence for sentence in tape.query_sentences for occurrence in sentence.token_occurrences
    )


def _query_sentence_lengths(tape: OpaqueDocumentEventTape) -> tuple[int, ...]:
    return tuple(len(sentence.token_occurrences) for sentence in tape.query_sentences)


def _document_sentences(
    tape: OpaqueDocumentEventTape,
) -> tuple[OpaqueSentenceOccurrence, ...]:
    return tuple(sentence for document in tape.document_sentences for sentence in document)


def _document_tokens(tape: OpaqueDocumentEventTape) -> tuple[OpaqueTokenOccurrence, ...]:
    return tuple(
        occurrence for document in tape.document_token_occurrences for occurrence in document
    )


def _all_edges(tape: OpaqueDocumentEventTape) -> tuple[_EdgeCandidate, ...]:
    candidates: list[_EdgeCandidate] = []
    for sentence in _document_sentences(tape):
        tokens = tuple(sentence.token_occurrences)
        for source_index, source in enumerate(tokens):
            for destination_index, destination in enumerate(tokens):
                if source_index == destination_index:
                    continue
                if sentence.document_index is None:
                    raise RuntimeError("document sentence lost its document index")
                candidates.append(
                    _EdgeCandidate(
                        document_index=sentence.document_index,
                        sentence_index=sentence.sentence_index,
                        source_token_index=source_index,
                        destination_token_index=destination_index,
                        source=source,
                        destination=destination,
                        sentence_tokens=tokens,
                    )
                )
    return tuple(candidates)


def _surface_key(edge: _EdgeCandidate) -> tuple[int, int, tuple[bytes | None, ...]]:
    literals: list[bytes | None] = []
    for index, occurrence in enumerate(edge.sentence_tokens):
        if index in (edge.source_token_index, edge.destination_token_index):
            literals.append(None)
        else:
            literals.append(occurrence.identity_event)
    return edge.source_token_index, edge.destination_token_index, tuple(literals)


def _unique_training_path(
    tape: OpaqueDocumentEventTape,
    *,
    subject: bytes,
    answer: bytes,
) -> _TrainingPath:
    edges = _all_edges(tape)
    paths = {
        (
            first.source.occurrence_event,
            first.destination.occurrence_event,
            second.source.occurrence_event,
            second.destination.occurrence_event,
        ): _TrainingPath(first, second)
        for first in edges
        if first.source.identity_event == subject
        for second in edges
        if first.document_index != second.document_index
        and first.destination.identity_event == second.source.identity_event
        and second.destination.identity_event == answer
    }
    if len(paths) != 1:
        raise ValueError("each success must contain one unique cross-document two-hop proof")
    return next(iter(paths.values()))


def _frame_record(frame: _SurfaceFrame) -> dict[str, object]:
    return {
        "destination_token_index": frame.destination_token_index,
        "hop": frame.hop,
        "literal_events_hex": tuple(
            None if event is None else event.hex() for event in frame.literal_events
        ),
        "source_token_index": frame.source_token_index,
        "support_ticks": frame.support_ticks,
        "token_count": frame.token_count,
    }


class NeuronalRelationalReader:
    """A frozen learned query/frame circuit for a recurrent two-hop read."""

    __slots__ = ("_checkpoint_sha256", "_schema", "_summary")

    def __init__(
        self,
        token: object,
        *,
        schema: _Schema,
        summary: RelationalReaderTrainingSummary,
        checkpoint_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use NeuronalRelationalReader.grow")
        self._schema = schema
        self._summary = summary
        self._checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def grow(cls, successes: Iterable[RelationalReadingSuccess]) -> NeuronalRelationalReader:
        """Consolidate recurring query and sentence frames without raw replay."""

        rows = tuple(successes)
        if not _MIN_SUCCESSES <= len(rows) <= _MAX_SUCCESSES:
            raise ValueError("successes must contain four to sixty-four observations")
        if any(not isinstance(row, RelationalReadingSuccess) for row in rows):
            raise TypeError("successes must contain RelationalReadingSuccess values")
        provenance_ids = tuple(row.provenance_id for row in rows)
        if len(set(provenance_ids)) != len(provenance_ids):
            raise ValueError("success provenance IDs must be independent")

        tapes = tuple(OpaqueDocumentEventTape.observe(row.query, row.documents) for row in rows)
        query_tokens = tuple(_query_tokens(tape) for tape in tapes)
        sentence_shapes = tuple(_query_sentence_lengths(tape) for tape in tapes)
        if len(set(sentence_shapes)) != 1 or len({len(tokens) for tokens in query_tokens}) != 1:
            raise ValueError("successes do not share one recurring query sequence frame")
        if not query_tokens[0]:
            raise ValueError("queries must contain lexical events")
        if len(query_tokens[0]) > _MAX_TOKENS:
            raise ValueError("query exceeds the lexical event bound")

        variable_positions = tuple(
            index
            for index in range(len(query_tokens[0]))
            if len({tokens[index].identity_event for tokens in query_tokens}) == len(rows)
        )
        if len(variable_positions) != 1:
            raise ValueError("successes must reveal one independently varying query filler")
        subject_index = variable_positions[0]
        if any(
            len({tokens[index].identity_event for tokens in query_tokens}) != 1
            for index in range(len(query_tokens[0]))
            if index != subject_index
        ):
            raise ValueError("query literals must recur exactly across successes")

        subjects = tuple(tokens[subject_index].identity_event for tokens in query_tokens)
        for subject, tape in zip(subjects, tapes, strict=True):
            if subject not in {token.identity_event for token in _document_tokens(tape)}:
                raise ValueError("each varying query filler must recur in its live documents")

        paths = tuple(
            _unique_training_path(
                tape,
                subject=subject,
                answer=row.observed_answer,
            )
            for row, tape, subject in zip(rows, tapes, subjects, strict=True)
        )
        intermediates = tuple(path.first.destination.identity_event for path in paths)
        answers = tuple(path.second.destination.identity_event for path in paths)
        if len(set(intermediates)) != len(rows) or len(set(answers)) != len(rows):
            raise ValueError("successes must use independent intermediate and answer fillers")
        if any(
            observed != live
            for observed, live in zip(
                (row.observed_answer for row in rows),
                answers,
                strict=True,
            )
        ):
            raise RuntimeError("training path lost the observed live answer")

        frame_groups: list[Counter[tuple[int, int, tuple[bytes | None, ...]]]] = [
            Counter(),
            Counter(),
        ]
        for path in paths:
            frame_groups[0][_surface_key(path.first)] += 1
            frame_groups[1][_surface_key(path.second)] += 1
        if any(support < 2 for group in frame_groups for support in group.values()):
            raise ValueError("each learned document surface needs two independent successes")

        hop_frames: list[tuple[_SurfaceFrame, ...]] = []
        for hop, group in enumerate(frame_groups, start=1):
            frames = tuple(
                _SurfaceFrame(
                    hop=hop,
                    token_count=len(literals),
                    source_token_index=source_index,
                    destination_token_index=destination_index,
                    literal_events=literals,
                    support_ticks=support,
                )
                for (source_index, destination_index, literals), support in sorted(
                    group.items(),
                    key=lambda item: (
                        item[0][0],
                        item[0][1],
                        tuple(b"" if event is None else event for event in item[0][2]),
                    ),
                )
            )
            hop_frames.append(frames)
        all_frames = (*hop_frames[0], *hop_frames[1])
        forward_count = sum(
            frame.source_token_index < frame.destination_token_index for frame in all_frames
        )
        reverse_count = len(all_frames) - forward_count
        if not forward_count or not reverse_count:
            raise ValueError("successes must teach both forward and reverse argument surfaces")

        query_literals = tuple(
            None if index == subject_index else token.identity_event
            for index, token in enumerate(query_tokens[0])
        )
        query_frame = _QueryFrame(
            sentence_lengths=sentence_shapes[0],
            subject_flat_index=subject_index,
            literal_events=query_literals,
            support_ticks=len(rows),
        )
        schema = _Schema(
            query_frame=query_frame,
            hop_frames=(hop_frames[0], hop_frames[1]),
        )
        provenance_root = _digest(
            _PROVENANCE_NAMESPACE,
            {"provenance_sha256s": tuple(sha256(value).hexdigest() for value in provenance_ids)},
        )
        checkpoint_record = {
            "hop_frames": tuple(
                tuple(_frame_record(frame) for frame in frames) for frames in schema.hop_frames
            ),
            "provenance_root_sha256": provenance_root,
            "query_frame": {
                "literal_events_hex": tuple(
                    None if event is None else event.hex() for event in query_frame.literal_events
                ),
                "sentence_lengths": query_frame.sentence_lengths,
                "subject_flat_index": query_frame.subject_flat_index,
                "support_ticks": query_frame.support_ticks,
            },
        }
        checkpoint_sha256 = _digest(_CHECKPOINT_NAMESPACE, checkpoint_record)
        summary = RelationalReaderTrainingSummary(
            success_count=len(rows),
            hop_count=2,
            query_literal_synapse_count=sum(event is not None for event in query_literals),
            surface_frame_count=len(all_frames),
            forward_surface_count=forward_count,
            reverse_surface_count=reverse_count,
            minimum_surface_support=min(frame.support_ticks for frame in all_frames),
            retained_concrete_endpoint_count=0,
            provenance_root_sha256=provenance_root,
            checkpoint_sha256=checkpoint_sha256,
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            schema=schema,
            summary=summary,
            checkpoint_sha256=checkpoint_sha256,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> RelationalReaderTrainingSummary:
        return self._summary

    @property
    def learned_surface_events(self) -> tuple[bytes, ...]:
        """Return the literal identity synapses retained by the frozen circuit."""

        events = [event for event in self._schema.query_frame.literal_events if event is not None]
        events.extend(
            event
            for frames in self._schema.hop_frames
            for frame in frames
            for event in frame.literal_events
            if event is not None
        )
        return tuple(sorted(set(events)))

    def respond(
        self,
        query: bytes,
        documents: tuple[OpaqueDocument, ...],
        ablation: RelationalReaderAblation = RelationalReaderAblation.NONE,
    ) -> RelationalReadingDecision:
        """Run the learned frame circuit over one complete live observation."""

        _require_bytes(query, "query")
        docs = _require_documents(documents)
        if not isinstance(ablation, RelationalReaderAblation):
            raise TypeError("ablation must be RelationalReaderAblation")
        tape = OpaqueDocumentEventTape.observe(query, docs)
        query_tokens = _query_tokens(tape)
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": self._checkpoint_sha256,
                "observation_sha256": tape.stable_hash.hex(),
            },
        )
        races: list[RelationalRaceTrace] = []

        query_matched = (
            _query_sentence_lengths(tape) == self._schema.query_frame.sentence_lengths
            and self._schema.query_frame.matches(query_tokens)
            and ablation is not RelationalReaderAblation.FRAME_BINDING_OFF
        )
        query_choice = _physical_choice(
            population="query_frame_binding",
            scores=(_DRIVE_SCORE if query_matched else 0,),
        )
        races.append(query_choice.trace)
        if query_choice.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                stage="query_frame",
                reason=query_choice.trace.stop_reason,
                completed_hops=0,
            )
        subject = query_tokens[self._schema.query_frame.subject_flat_index]
        edges = _all_edges(tape)
        if not edges:
            return self._tonic_stop(
                query_sha256=query_sha256,
                races=races,
                stage="first_join",
                completed_hops=0,
            )

        first_matches = self._detect_frames(
            edges,
            frames=self._schema.hop_frames[0],
            hop=1,
            ablation=ablation,
            races=races,
        )
        first_join = _physical_choice(
            population="query_to_document_equality_join",
            scores=tuple(
                _DRIVE_SCORE
                if (
                    matched
                    and ablation is not RelationalReaderAblation.EQUALITY_OFF
                    and edge.source.identity_event == subject.identity_event
                )
                else 0
                for edge, matched in zip(edges, first_matches, strict=True)
            ),
        )
        races.append(first_join.trace)
        if first_join.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                stage="first_join",
                reason=first_join.trace.stop_reason,
                completed_hops=0,
            )
        first_edge = edges[first_join.column]

        second_matches = self._detect_frames(
            edges,
            frames=self._schema.hop_frames[1],
            hop=2,
            ablation=ablation,
            races=races,
        )
        second_join = _physical_choice(
            population="recurrent_document_equality_join",
            scores=tuple(
                _DRIVE_SCORE
                if (
                    matched
                    and ablation is not RelationalReaderAblation.EQUALITY_OFF
                    and ablation is not RelationalReaderAblation.RECURRENCE_OFF
                    and edge.source.identity_event == first_edge.destination.identity_event
                    and edge.document_index != first_edge.document_index
                )
                else 0
                for edge, matched in zip(edges, second_matches, strict=True)
            ),
        )
        races.append(second_join.trace)
        if second_join.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                stage="recurrent_join",
                reason=second_join.trace.stop_reason,
                completed_hops=1,
            )
        second_edge = edges[second_join.column]

        live_tokens = tuple(tape.token_occurrences)
        motor = _physical_choice(
            population="live_answer_motor",
            scores=tuple(
                _DRIVE_SCORE
                if (
                    ablation is not RelationalReaderAblation.MOTOR_OFF
                    and occurrence.identity_event == second_edge.destination.identity_event
                )
                else 0
                for occurrence in live_tokens
            ),
        )
        races.append(motor.trace)
        if motor.column is None:
            return self._stop(
                query_sha256=query_sha256,
                races=races,
                stage="motor",
                reason=motor.trace.stop_reason,
                completed_hops=2,
            )
        answer_occurrence = live_tokens[motor.column]
        return self._decision(
            decision="EMIT",
            answer=answer_occurrence.identity_event,
            answer_occurrence_event=answer_occurrence.occurrence_event,
            completed_hops=2,
            stop_stage=None,
            stop_reason=None,
            races=races,
            query_sha256=query_sha256,
        )

    def _detect_frames(
        self,
        edges: Sequence[_EdgeCandidate],
        *,
        frames: tuple[_SurfaceFrame, ...],
        hop: int,
        ablation: RelationalReaderAblation,
        races: list[RelationalRaceTrace],
    ) -> tuple[bool, ...]:
        matched_edges: list[bool] = []
        enabled = ablation is not RelationalReaderAblation.FRAME_BINDING_OFF
        for index, edge in enumerate(edges):
            matched, trace = _physical_detector(
                population=f"hop_{hop}_shared_frame_{index}",
                matched=enabled and any(frame.matches(edge) for frame in frames),
            )
            matched_edges.append(matched)
            races.append(trace)
        return tuple(matched_edges)

    def _tonic_stop(
        self,
        *,
        query_sha256: str,
        races: list[RelationalRaceTrace],
        stage: str,
        completed_hops: int,
    ) -> RelationalReadingDecision:
        stop = _physical_choice(population=f"{stage}_empty", scores=(0,))
        races.append(stop.trace)
        return self._stop(
            query_sha256=query_sha256,
            races=races,
            stage=stage,
            reason=stop.trace.stop_reason,
            completed_hops=completed_hops,
        )

    def _stop(
        self,
        *,
        query_sha256: str,
        races: list[RelationalRaceTrace],
        stage: str,
        reason: str | None,
        completed_hops: int,
    ) -> RelationalReadingDecision:
        if reason is None:
            raise RuntimeError("physical STOP lost its race reason")
        return self._decision(
            decision="STOP",
            answer=None,
            answer_occurrence_event=None,
            completed_hops=completed_hops,
            stop_stage=stage,
            stop_reason=f"{stage}:{reason}",
            races=races,
            query_sha256=query_sha256,
        )

    def _decision(
        self,
        *,
        decision: str,
        answer: bytes | None,
        answer_occurrence_event: bytes | None,
        completed_hops: int,
        stop_stage: str | None,
        stop_reason: str | None,
        races: list[RelationalRaceTrace],
        query_sha256: str,
    ) -> RelationalReadingDecision:
        race_tuple = tuple(races)
        physical_cell_steps = sum(race.physical_cell_steps for race in race_tuple)
        record = {
            "answer_hex": None if answer is None else answer.hex(),
            "answer_occurrence_event_hex": (
                None if answer_occurrence_event is None else answer_occurrence_event.hex()
            ),
            "checkpoint_sha256": self._checkpoint_sha256,
            "completed_hops": completed_hops,
            "decision": decision,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256s": tuple(race.sha256 for race in race_tuple),
            "stop_reason": stop_reason,
            "stop_stage": stop_stage,
        }
        return RelationalReadingDecision(
            decision=decision,
            answer=answer,
            answer_occurrence_event=answer_occurrence_event,
            completed_hops=completed_hops,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            races=race_tuple,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
