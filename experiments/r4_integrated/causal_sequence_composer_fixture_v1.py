"""Frozen opaque fixture for factorized causal-sequence composition.

Three public demonstrations occupy three cells of a two-by-two cue grid.  The
fresh query occupies the fourth cell: both emitted chunks have been observed,
but their adjacency has not.  Every record uses a disjoint role inventory, so
passing the query requires rebinding rather than replaying literal pointers.

The two remaining queries are deliberately unresolved: one contains two known
motifs in the same cue slot and one contains an unseen motif.  Scoring maps both
to abstention.  No task names, source language, syntax tree, candidate answer,
or target annotation is present in the public payload.

This fixture is data and scoring custody only.  It does not import, inspect, or
configure a learner.  Cardinality stays fixed at two so this gate isolates
causal composition; variable-cardinality role cursors are tested separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import product

SCHEMA = "causal-sequence-composer-fixture-v1"

_OPEN = b"\xe1"
_MIDDLE = b"\xe2"
_CLOSE = b"\xe3"

_CUE_MOTIFS = (
    ((b"\x91", b"\x13"), (b"\x91", b"\x2b")),
    ((b"\xb7", b"\x35"), (b"\xb7", b"\x4d")),
)
_EMISSION_CHUNKS = (
    (b"\x51\xa1", b"\x52\xb2"),
    (b"\x61\xc3", b"\x62\xd4"),
)
_UNKNOWN_MOTIF = (b"\x91", b"\x7f")


Emission = tuple[tuple[bytes, bytes], tuple[bytes, bytes]]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _cue(first: int, second: int) -> tuple[bytes, ...]:
    return (
        _OPEN,
        *_CUE_MOTIFS[0][first],
        _MIDDLE,
        *_CUE_MOTIFS[1][second],
        _CLOSE,
    )


def _compose(first: int, second: int, roles: tuple[bytes, bytes]) -> Emission:
    return (
        (_EMISSION_CHUNKS[0][first], roles[0]),
        (_EMISSION_CHUNKS[1][second], roles[1]),
    )


def _serialized_emissions(emissions: Emission) -> tuple[dict[str, str], ...]:
    return tuple(
        {"chunk": chunk.hex(), "role_byte": role.hex()}
        for chunk, role in emissions
    )


def _validate_cue(cue_events: tuple[bytes, ...]) -> None:
    if (
        not isinstance(cue_events, tuple)
        or not cue_events
        or not all(isinstance(event, bytes) and len(event) == 1 for event in cue_events)
    ):
        raise ValueError("cue events must be a non-empty tuple of opaque bytes")


def _validate_roles(role_inventory: tuple[bytes, bytes]) -> None:
    if (
        not isinstance(role_inventory, tuple)
        or len(role_inventory) != 2
        or not all(isinstance(role, bytes) and len(role) == 1 for role in role_inventory)
        or len(set(role_inventory)) != 2
    ):
        raise ValueError("role inventory must contain two distinct opaque bytes")


def _contains(stream: tuple[bytes, ...], motif: tuple[bytes, ...]) -> bool:
    width = len(motif)
    return any(stream[index : index + width] == motif for index in range(len(stream) - width + 1))


def _cue_combinations(cue_events: tuple[bytes, ...]) -> tuple[tuple[int, int], ...]:
    present = tuple(
        tuple(index for index, motif in enumerate(axis) if _contains(cue_events, motif))
        for axis in _CUE_MOTIFS
    )
    return tuple(product(*present)) if all(present) else ()


@dataclass(frozen=True, slots=True)
class CausalSequenceLessonV1:
    """One public demonstration with an independently renamed role inventory."""

    cue_events: tuple[bytes, ...]
    role_inventory: tuple[bytes, bytes]
    demonstrated_emissions: Emission
    record_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "cue_events": tuple(event.hex() for event in self.cue_events),
            "role_inventory": tuple(role.hex() for role in self.role_inventory),
            "demonstrated_emissions": _serialized_emissions(self.demonstrated_emissions),
        }

    def record(self) -> dict[str, object]:
        return {**self._body(), "record_sha256": self.record_sha256}

    def authenticate(self) -> None:
        _validate_cue(self.cue_events)
        _validate_roles(self.role_inventory)
        combinations = _cue_combinations(self.cue_events)
        if len(combinations) != 1:
            raise ValueError("training cue must resolve to one combination")
        first, second = combinations[0]
        if self.demonstrated_emissions != _compose(
            first,
            second,
            self.role_inventory,
        ):
            raise ValueError("demonstrated emissions changed")
        if self.record_sha256 != _digest(
            b"causal-sequence-lesson-v1",
            self._body(),
        ):
            raise ValueError("training record changed")


@dataclass(frozen=True, slots=True)
class CausalSequenceObservationV1:
    """One opaque query; scoring information is absent from its public record."""

    cue_events: tuple[bytes, ...]
    role_inventory: tuple[bytes, bytes]
    observation_sha256: str

    def _body(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "cue_events": tuple(event.hex() for event in self.cue_events),
            "role_inventory": tuple(role.hex() for role in self.role_inventory),
        }

    def public_observation(self) -> dict[str, object]:
        return {**self._body(), "observation_sha256": self.observation_sha256}

    def authenticate(self) -> None:
        _validate_cue(self.cue_events)
        _validate_roles(self.role_inventory)
        if self.observation_sha256 != _digest(
            b"causal-sequence-observation-v1",
            self._body(),
        ):
            raise ValueError("query record changed")


def _lesson(
    first: int,
    second: int,
    roles: tuple[bytes, bytes],
) -> CausalSequenceLessonV1:
    provisional = CausalSequenceLessonV1(
        cue_events=_cue(first, second),
        role_inventory=roles,
        demonstrated_emissions=_compose(first, second, roles),
        record_sha256="",
    )
    return CausalSequenceLessonV1(
        cue_events=provisional.cue_events,
        role_inventory=provisional.role_inventory,
        demonstrated_emissions=provisional.demonstrated_emissions,
        record_sha256=_digest(b"causal-sequence-lesson-v1", provisional._body()),
    )


def _observation(
    cue_events: tuple[bytes, ...],
    roles: tuple[bytes, bytes],
) -> CausalSequenceObservationV1:
    provisional = CausalSequenceObservationV1(cue_events, roles, "")
    return CausalSequenceObservationV1(
        cue_events=cue_events,
        role_inventory=roles,
        observation_sha256=_digest(
            b"causal-sequence-observation-v1",
            provisional._body(),
        ),
    )


TRAINING_LESSONS = (
    _lesson(0, 0, (b"\x01", b"\x02")),
    _lesson(0, 1, (b"\x03", b"\x04")),
    _lesson(1, 0, (b"\x05", b"\x06")),
)

HELD_OUT_CASE = _observation(_cue(1, 1), (b"\x07", b"\x08"))

PROBES = (
    _observation(
        (
            _OPEN,
            *_CUE_MOTIFS[0][0],
            *_CUE_MOTIFS[0][1],
            _MIDDLE,
            *_CUE_MOTIFS[1][0],
            _CLOSE,
        ),
        (b"\x09", b"\x0a"),
    ),
    _observation(
        (_OPEN, *_UNKNOWN_MOTIF, _MIDDLE, *_CUE_MOTIFS[1][1], _CLOSE),
        (b"\x0b", b"\x0c"),
    ),
)

QUERIES = (HELD_OUT_CASE, *PROBES)


def scoring_emissions(observation: CausalSequenceObservationV1) -> Emission | None:
    """Return scoring-only emissions, or ``None`` for an unresolved cue."""

    observation.authenticate()
    combinations = _cue_combinations(observation.cue_events)
    if len(combinations) != 1:
        return None
    first, second = combinations[0]
    return _compose(first, second, observation.role_inventory)


def public_payload() -> dict[str, object]:
    """Return a fresh public data tree with no scoring annotations."""

    return {
        "schema": SCHEMA,
        "lessons": tuple(lesson.record() for lesson in TRAINING_LESSONS),
        "queries": tuple(query.public_observation() for query in QUERIES),
    }


def _payload_sha256() -> str:
    return _digest(b"causal-sequence-public-payload-v1", public_payload())


def _custody_sha256() -> str:
    scoring_roots = tuple(
        _digest(
            b"causal-sequence-query-score-v1",
            None if emissions is None else _serialized_emissions(emissions),
        )
        for emissions in (scoring_emissions(query) for query in QUERIES)
    )
    return _digest(
        b"causal-sequence-custody-v1",
        {"payload_sha256": _payload_sha256(), "scoring_roots": scoring_roots},
    )


def _assert_data_only(value: object) -> None:
    forbidden_key_fragments = (
        "answer",
        "ast",
        "candidate",
        "correct",
        "label",
        "python",
        "solution",
        "target",
        "task",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or any(
                fragment in key.lower() for fragment in forbidden_key_fragments
            ):
                raise ValueError("public payload contains answer-bearing metadata")
            _assert_data_only(child)
        return
    if isinstance(value, (tuple, list)):
        for child in value:
            _assert_data_only(child)
        return
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("public payload must contain data primitives only")


def _authenticate_fixture() -> None:
    if len(TRAINING_LESSONS) != 3 or len(QUERIES) != 3:
        raise ValueError("fixture cardinality changed")
    for lesson in TRAINING_LESSONS:
        lesson.authenticate()
    for query in QUERIES:
        query.authenticate()

    training_combinations = tuple(
        _cue_combinations(lesson.cue_events)[0] for lesson in TRAINING_LESSONS
    )
    full_grid = set(product(range(2), repeat=2))
    if len(set(training_combinations)) != 3:
        raise ValueError("training must occupy exactly three distinct grid cells")
    missing = full_grid - set(training_combinations)
    held_out_combinations = _cue_combinations(HELD_OUT_CASE.cue_events)
    if len(missing) != 1 or held_out_combinations != tuple(missing):
        raise ValueError("fresh query must occupy the fourth grid cell")

    all_roles = tuple(
        role
        for item in (*TRAINING_LESSONS, *QUERIES)
        for role in item.role_inventory
    )
    if len(set(all_roles)) != len(all_roles):
        raise ValueError("every role byte must be globally fresh")
    cue_atoms = {
        event
        for item in (*TRAINING_LESSONS, *QUERIES)
        for event in item.cue_events
    }
    chunks = {chunk for axis in _EMISSION_CHUNKS for chunk in axis}
    if set(all_roles) & cue_atoms or set(all_roles) & chunks:
        raise ValueError("role bytes must be disjoint from cues and chunks")

    training_pairs = {
        tuple(chunk for chunk, _role in lesson.demonstrated_emissions)
        for lesson in TRAINING_LESSONS
    }
    held_out = scoring_emissions(HELD_OUT_CASE)
    if held_out is None:
        raise ValueError("fresh composition unexpectedly unresolved")
    held_out_pair = tuple(chunk for chunk, _role in held_out)
    observed_chunks = {chunk for pair in training_pairs for chunk in pair}
    if held_out_pair in training_pairs or not set(held_out_pair) <= observed_chunks:
        raise ValueError("fresh query must reuse chunks in an unseen adjacency")

    first, second = held_out_combinations[0]
    seen_after_held_out_first = {
        right
        for left, right in training_pairs
        if left == _EMISSION_CHUNKS[0][first]
    }
    donor_first = _EMISSION_CHUNKS[0][1 - first]
    selected_right = _EMISSION_CHUNKS[1][1 - second]
    alternate_right = _EMISSION_CHUNKS[1][second]
    if (
        seen_after_held_out_first != {selected_right}
        or (donor_first, selected_right) not in training_pairs
        or (donor_first, alternate_right) not in training_pairs
    ):
        raise ValueError("branch-local revision evidence changed")

    resolution_counts = tuple(
        len(_cue_combinations(probe.cue_events)) for probe in PROBES
    )
    if resolution_counts != (2, 0):
        raise ValueError("fixture must contain one ambiguous and one unknown cue")
    if any(scoring_emissions(probe) is not None for probe in PROBES):
        raise ValueError("unresolved probes must score as abstentions")

    payload = public_payload()
    _assert_data_only(payload)
    serialized = _canonical_json(payload).lower()
    if any(
        marker in serialized
        for marker in (b"python", b"ast", b"answer", b"candidate", b"target", b"task")
    ):
        raise ValueError("public payload contains forbidden semantic metadata")


_authenticate_fixture()

CAUSAL_SEQUENCE_PAYLOAD_SHA256 = (
    "28a0801a7e9b50e6235c6a793c9481c186e056ba0e8b48a3c332e31539c18e56"
)
CAUSAL_SEQUENCE_CUSTODY_SHA256 = (
    "c98abb519bbd6a4cc81b291cb1c45d39f21b991752353cdd2f980f4a5af40439"
)

if _payload_sha256() != CAUSAL_SEQUENCE_PAYLOAD_SHA256:
    raise ValueError("causal-sequence fixture differs from its frozen payload root")
if _custody_sha256() != CAUSAL_SEQUENCE_CUSTODY_SHA256:
    raise ValueError("causal-sequence fixture differs from its frozen custody root")


def authenticate_fixture() -> None:
    """Re-authenticate records, invariants, and both frozen roots."""

    _authenticate_fixture()
    if _payload_sha256() != CAUSAL_SEQUENCE_PAYLOAD_SHA256:
        raise ValueError("causal-sequence public payload changed")
    if _custody_sha256() != CAUSAL_SEQUENCE_CUSTODY_SHA256:
        raise ValueError("causal-sequence scoring custody changed")


__all__ = [
    "CAUSAL_SEQUENCE_CUSTODY_SHA256",
    "CAUSAL_SEQUENCE_PAYLOAD_SHA256",
    "HELD_OUT_CASE",
    "PROBES",
    "QUERIES",
    "SCHEMA",
    "TRAINING_LESSONS",
    "CausalSequenceLessonV1",
    "CausalSequenceObservationV1",
    "authenticate_fixture",
    "public_payload",
    "scoring_emissions",
]
