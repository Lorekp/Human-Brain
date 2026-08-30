"""Fail-closed source for the G0 signed-regulatory generator.

The adjudicative seeds are frozen elsewhere and are never executed here.  This
module pins the generator algorithm and permits only two quarantined fixture
seeds for source-level validation.  Frozen smoke/learning/adjudicative cohort
generation remains unavailable until the separate nine-artifact common-
architecture binding exists.

This is generator/oracle infrastructure.  Candidate code must not import it.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import NoReturn

from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as g0

GENERATOR_SCOPE = "FROZEN_G0_REGULATORY_GENERATOR_V1"
GENERATOR_STATUS = "SOURCE_FROZEN_GENERATION_FAIL_CLOSED"

EXPECTED_G0_COMMIT = "42278a75647951a7da7a51c442ac14bb54db7dab"
EXPECTED_G0_SOURCE_SHA256 = "e027e598aa926b108e90208e1586a93c0da35716e165a746bb2732d980774395"
EXPECTED_G0_PROTOCOL_SHA256 = "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"

ENTITY_COUNT = 8
EDGE_COUNT_RANGE = (10, 16)
IN_DEGREE_RANGE = (1, 3)
EDGE_WEIGHTS = (-1, 1)
BIASES = (-1, 0, 1)
MAX_REJECTION_ATTEMPTS = 100_000

FROZEN_COHORT_SEEDS = (
    g0.REGULATORY_SPLIT.smoke_seed,
    g0.REGULATORY_SPLIT.learning_seed,
    g0.REGULATORY_SPLIT.adjudicative_seed,
    g0.REGULATORY_SPLIT.rename_seed,
)
FIXTURE_SEEDS = (9_170_001, 9_170_019)

ARCHITECTURE_BINDING_COUNT = 0
FROZEN_COHORT_EXECUTION_COUNT = 0
FROZEN_SYSTEM_GENERATION_COUNT = 0
ADJUDICATIVE_SYSTEM_GENERATION_COUNT = 0
GENERATION_AUTHORIZED = False

_ROOT = Path(__file__).resolve().parents[2]
_PREREG_PATH = _ROOT / "experiments/g0/frozen_cross_domain_g0_preregistration_v1.py"
_SYSTEM_NAMESPACE = b"frozen-g0-regulatory-system-v1\0"
_SOURCE_SPEC_NAMESPACE = b"frozen-g0-regulatory-generator-source-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class SignedEdge:
    parent_index: int
    child_index: int
    weight: int

    def record(self) -> dict[str, int]:
        return {
            "parent_index": self.parent_index,
            "child_index": self.child_index,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class RegulatorySystem:
    seed: int
    system_index: int
    profile: str
    accepted_attempt: int
    entity_tokens: tuple[int, ...]
    edges: tuple[SignedEdge, ...]
    biases: tuple[int, ...]
    composition_motifs: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "seed": self.seed,
            "system_index": self.system_index,
            "profile": self.profile,
            "accepted_attempt": self.accepted_attempt,
            "entity_tokens": self.entity_tokens,
            "edges": tuple(edge.record() for edge in self.edges),
            "biases": self.biases,
            "composition_motifs": self.composition_motifs,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record

    def transition(
        self,
        current: tuple[int, ...],
        *,
        clamp_entity_token: int | None = None,
        clamp_value: int | None = None,
    ) -> tuple[int, ...]:
        clamp_index: int | None = None
        if clamp_entity_token is not None or clamp_value is not None:
            if clamp_entity_token is None or clamp_value not in (0, 1):
                raise ValueError("a clamp requires one raw entity token and binary value")
            try:
                clamp_index = self.entity_tokens.index(clamp_entity_token)
            except ValueError as error:
                raise ValueError("clamp entity token is absent from this system") from error
        return _transition(
            self.edges,
            self.biases,
            current,
            clamp_index=clamp_index,
            clamp_value=clamp_value,
        )


def _authenticate_preregistration() -> None:
    if (
        sha256(_PREREG_PATH.read_bytes()).hexdigest() != EXPECTED_G0_SOURCE_SHA256
        or g0.PROTOCOL.protocol_sha256 != EXPECTED_G0_PROTOCOL_SHA256
        or g0.PROTOCOL.regulatory.generator_specification != g0.REGULATORY_SPECIFICATION
        or g0.PROTOCOL.regulatory.split != g0.REGULATORY_SPLIT
        or tuple(g0.REGULATORY_SPECIFICATION["edge_count_range"]) != EDGE_COUNT_RANGE
        or tuple(g0.REGULATORY_SPECIFICATION["in_degree_range"]) != IN_DEGREE_RANGE
        or g0.REGULATORY_SPECIFICATION["entity_count"] != ENTITY_COUNT
        or g0.SURFACE_BOUNDARY_RECORD["generation_authorized"] is not False
    ):
        raise RuntimeError("frozen G0 regulatory preregistration changed")


def _transition(
    edges: tuple[SignedEdge, ...],
    biases: tuple[int, ...],
    current: tuple[int, ...],
    *,
    clamp_index: int | None = None,
    clamp_value: int | None = None,
) -> tuple[int, ...]:
    if len(current) != ENTITY_COUNT or any(value not in (0, 1) for value in current):
        raise ValueError("regulatory state must contain exactly eight binary values")
    effective = list(current)
    if clamp_index is not None:
        if clamp_value not in (0, 1):
            raise ValueError("clamp value must be binary")
        effective[clamp_index] = clamp_value
    scores = list(biases)
    for edge in edges:
        bipolar_parent = 1 if effective[edge.parent_index] else -1
        scores[edge.child_index] += edge.weight * bipolar_parent
    following = [int(score >= 0) for score in scores]
    if clamp_index is not None:
        following[clamp_index] = int(clamp_value)
    return tuple(following)


def _underlying_connected(edges: tuple[SignedEdge, ...]) -> bool:
    neighbors: list[set[int]] = [set() for _ in range(ENTITY_COUNT)]
    for edge in edges:
        neighbors[edge.parent_index].add(edge.child_index)
        neighbors[edge.child_index].add(edge.parent_index)
    reached = {0}
    queue = deque((0,))
    while queue:
        node = queue.popleft()
        for neighbor in neighbors[node] - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return len(reached) == ENTITY_COUNT


def _transition_table(
    edges: tuple[SignedEdge, ...],
    biases: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _transition(
            edges,
            biases,
            tuple((mask >> index) & 1 for index in range(ENTITY_COUNT)),
        )
        for mask in range(1 << ENTITY_COUNT)
    )


def _every_edge_is_effective(
    edges: tuple[SignedEdge, ...],
    table: tuple[tuple[int, ...], ...],
) -> bool:
    return all(
        any(
            table[mask][edge.child_index]
            != table[mask ^ (1 << edge.parent_index)][edge.child_index]
            for mask in range(1 << ENTITY_COUNT)
        )
        for edge in edges
    )


def _every_node_takes_both_values(table: tuple[tuple[int, ...], ...]) -> bool:
    return all({row[node] for row in table} == {0, 1} for node in range(ENTITY_COUNT))


def _composition_motifs(edges: tuple[SignedEdge, ...]) -> tuple[str, ...]:
    by_pair = {(edge.parent_index, edge.child_index): edge.weight for edge in edges}
    outgoing: dict[int, tuple[int, ...]] = {
        node: tuple(child for parent, child in by_pair if parent == node)
        for node in range(ENTITY_COUNT)
    }
    motifs: set[str] = set()

    for source in range(ENTITY_COUNT):
        children = outgoing[source]
        for left_index, left in enumerate(children):
            for right in children[left_index + 1 :]:
                common = set(outgoing[left]) & set(outgoing[right])
                if common:
                    motifs.add("converging-diamond")

    def walk(path: tuple[int, ...], length: int) -> bool:
        if len(path) - 1 == length:
            terminal = path[-1]
            children = tuple(child for child in outgoing[terminal] if child not in path)
            if len(children) < 2:
                return False
            signs = {by_pair[(terminal, child)] for child in children}
            return signs == {-1, 1}
        return any(
            walk((*path, child), length) for child in outgoing[path[-1]] if child not in path
        )

    for start in range(ENTITY_COUNT):
        if walk((start,), 3) or walk((start,), 4):
            motifs.add("signed-path-joined-to-fork")
            break
    return tuple(sorted(motifs))


def _candidate(
    seed: int,
    system_index: int,
    profile: str,
    attempt: int,
) -> RegulatorySystem | None:
    material = _digest(
        b"frozen-g0-regulatory-candidate-seed-v1\0",
        (seed, system_index, profile, attempt),
    )
    rng = Random(int(material, 16))
    tokens: set[int] = set()
    while len(tokens) < ENTITY_COUNT:
        tokens.add(rng.getrandbits(32))
    entity_tokens = tuple(sorted(tokens))

    edge_count = rng.randint(*EDGE_COUNT_RANGE)
    pairs: set[tuple[int, int]] = set()
    in_degree = [0] * ENTITY_COUNT
    for child in range(ENTITY_COUNT):
        parent = rng.randrange(ENTITY_COUNT - 1)
        if parent >= child:
            parent += 1
        pairs.add((parent, child))
        in_degree[child] += 1
    while len(pairs) < edge_count:
        parent = rng.randrange(ENTITY_COUNT)
        child = rng.randrange(ENTITY_COUNT)
        if parent == child or in_degree[child] >= IN_DEGREE_RANGE[1] or (parent, child) in pairs:
            continue
        pairs.add((parent, child))
        in_degree[child] += 1
    weighted = tuple(
        sorted(SignedEdge(parent, child, EDGE_WEIGHTS[rng.randrange(2)]) for parent, child in pairs)
    )
    biases = tuple(BIASES[rng.randrange(len(BIASES))] for _ in range(ENTITY_COUNT))
    table = _transition_table(weighted, biases)
    motifs = _composition_motifs(weighted)
    if (
        not _underlying_connected(weighted)
        or not _every_edge_is_effective(weighted, table)
        or not _every_node_takes_both_values(table)
        or (profile == "learning" and motifs)
        or (profile == "heldout-composition" and not motifs)
        or profile not in {"smoke", "learning", "fresh-structure", "heldout-composition"}
    ):
        return None
    record = {
        "seed": seed,
        "system_index": system_index,
        "profile": profile,
        "accepted_attempt": attempt,
        "entity_tokens": entity_tokens,
        "edges": tuple(edge.record() for edge in weighted),
        "biases": biases,
        "composition_motifs": motifs,
    }
    return RegulatorySystem(
        seed=seed,
        system_index=system_index,
        profile=profile,
        accepted_attempt=attempt,
        entity_tokens=entity_tokens,
        edges=weighted,
        biases=biases,
        composition_motifs=motifs,
        sha256=_digest(_SYSTEM_NAMESPACE, record),
    )


def _generate_system(seed: int, system_index: int, profile: str) -> RegulatorySystem:
    for attempt in range(MAX_REJECTION_ATTEMPTS):
        system = _candidate(seed, system_index, profile, attempt)
        if system is not None:
            return system
    raise RuntimeError("regulatory rejection sampler exhausted its frozen attempt cap")


def generate_fixture_system(seed: int, *, system_index: int = 0) -> RegulatorySystem:
    """Generate one quarantined smoke fixture; frozen cohort seeds are rejected."""

    if type(seed) is not int or seed not in FIXTURE_SEEDS:
        raise ValueError("only the two frozen non-cohort fixture seeds are executable")
    if type(system_index) is not int or system_index < 0 or system_index >= 4:
        raise ValueError("fixture system_index must be an integer in [0, 4)")
    return _generate_system(seed, system_index, "smoke")


def generate_frozen_cohort(*_args: object, **_kwargs: object) -> NoReturn:
    """Fail closed until a separately frozen common-architecture binding exists."""

    raise RuntimeError(
        "G0 regulatory cohort generation requires the frozen nine-artifact "
        "common-architecture binding"
    )


def validate_system(system: RegulatorySystem) -> dict[str, object]:
    """Return generator-only validity facts; never expose this to the candidate."""

    table = _transition_table(system.edges, system.biases)
    indegrees = tuple(
        sum(edge.child_index == node for edge in system.edges) for node in range(ENTITY_COUNT)
    )
    return {
        "entity_count": len(system.entity_tokens),
        "distinct_entity_token_count": len(set(system.entity_tokens)),
        "edge_count": len(system.edges),
        "self_edge_count": sum(edge.parent_index == edge.child_index for edge in system.edges),
        "minimum_in_degree": min(indegrees),
        "maximum_in_degree": max(indegrees),
        "underlying_connected": _underlying_connected(system.edges),
        "every_edge_effective": _every_edge_is_effective(system.edges, table),
        "every_node_takes_both_values": _every_node_takes_both_values(table),
        "composition_motifs": system.composition_motifs,
        "sha256": system.sha256,
    }


_authenticate_preregistration()

SOURCE_SPECIFICATION = {
    "scope": GENERATOR_SCOPE,
    "status": GENERATOR_STATUS,
    "g0_commit": EXPECTED_G0_COMMIT,
    "g0_protocol_sha256": EXPECTED_G0_PROTOCOL_SHA256,
    "transition": "lagged-bipolar-signed-threshold-score-greater-than-or-equal-zero",
    "entity_count": ENTITY_COUNT,
    "edge_count_range": EDGE_COUNT_RANGE,
    "in_degree_range": IN_DEGREE_RANGE,
    "edge_weights": EDGE_WEIGHTS,
    "biases": BIASES,
    "generator_rejections": tuple(g0.REGULATORY_SPECIFICATION["generator_rejections"]),
    "composition_motifs": tuple(g0.REGULATORY_SPECIFICATION["composition_motifs"]),
    "frozen_cohort_seeds": FROZEN_COHORT_SEEDS,
    "fixture_seeds": FIXTURE_SEEDS,
    "generation_authorized": GENERATION_AUTHORIZED,
    "architecture_binding_count": ARCHITECTURE_BINDING_COUNT,
    "frozen_cohort_execution_count": FROZEN_COHORT_EXECUTION_COUNT,
    "frozen_system_generation_count": FROZEN_SYSTEM_GENERATION_COUNT,
    "adjudicative_system_generation_count": ADJUDICATIVE_SYSTEM_GENERATION_COUNT,
}
SOURCE_SPECIFICATION_SHA256 = _digest(_SOURCE_SPEC_NAMESPACE, SOURCE_SPECIFICATION)


__all__ = [
    "ADJUDICATIVE_SYSTEM_GENERATION_COUNT",
    "ARCHITECTURE_BINDING_COUNT",
    "EXPECTED_G0_COMMIT",
    "EXPECTED_G0_PROTOCOL_SHA256",
    "EXPECTED_G0_SOURCE_SHA256",
    "FIXTURE_SEEDS",
    "FROZEN_COHORT_EXECUTION_COUNT",
    "FROZEN_COHORT_SEEDS",
    "FROZEN_SYSTEM_GENERATION_COUNT",
    "GENERATION_AUTHORIZED",
    "GENERATOR_SCOPE",
    "GENERATOR_STATUS",
    "SOURCE_SPECIFICATION",
    "SOURCE_SPECIFICATION_SHA256",
    "RegulatorySystem",
    "SignedEdge",
    "generate_fixture_system",
    "generate_frozen_cohort",
    "validate_system",
]
