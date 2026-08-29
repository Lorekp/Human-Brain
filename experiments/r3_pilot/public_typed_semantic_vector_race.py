"""Physical exact selection over one deduplicated typed semantic cost tier.

Candidate keys are opaque deterministic identifiers supplied by a separate
enumerator.  This module sees only per-case physical output encodings.  Fixed
case detectors drive a fixed all-case coincidence projection, then a resting
GLIF tier race with a real STOP cell selects one exact semantic vector.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.resting_glif_tier_beam import (
    RETAINED,
    STOP_LEADING_TIER,
    CompiledRestingGlifTierBeam,
    RestingGlifTierBeamResult,
)


SELECTED = "SELECTED"
STOP_NO_EXACT = "STOP_NO_EXACT_SEMANTIC_VECTOR"
STOP_MULTIPLE_EXACT = "STOP_MULTIPLE_EXACT_SEMANTIC_VECTORS"

_MIN_CURRENT_PA = 600.0
_STOP_CURRENT_PA = 2_000.0
_MAX_CURRENT_PA = 3_400.0
_CASE_WEIGHT_PA = 2_800.0
_INHIBITION_CURRENT_PA = 3_000.0
_RACE_STEPS = 600


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _vector_sha256(
    row: Sequence[output_cells.TypedOutputEncoding],
) -> str:
    digest = sha256()
    digest.update(b"public-typed-semantic-vector-v1\0")
    for encoded in row:
        digest.update(encoded.sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _projection(
    case_count: int,
) -> tuple[np.ndarray, float, str]:
    weights = np.ascontiguousarray(
        np.full(case_count, _CASE_WEIGHT_PA, dtype=np.float64)
    )
    bias = _STOP_CURRENT_PA - (case_count - 0.5) * _CASE_WEIGHT_PA
    digest = sha256()
    digest.update(b"public-typed-semantic-vector-projection-v1\0")
    case_count_bytes = np.asarray((case_count,), dtype="<i8")
    digest.update(memoryview(case_count_bytes).cast("B"))
    digest.update(memoryview(weights).cast("B"))
    bias_bytes = np.asarray((bias,), dtype="<f8")
    digest.update(memoryview(bias_bytes).cast("B"))
    limits_bytes = np.asarray(
        (
            _MIN_CURRENT_PA,
            _STOP_CURRENT_PA,
            _MAX_CURRENT_PA,
            _INHIBITION_CURRENT_PA,
        ),
        dtype="<f8",
    )
    digest.update(memoryview(limits_bytes).cast("B"))
    weights.setflags(write=False)
    return weights, bias, digest.hexdigest()


def _tier_sha256(tier: RestingGlifTierBeamResult) -> str:
    return sha256(_canonical_json(asdict(tier))).hexdigest()


@dataclass(frozen=True, slots=True)
class TypedSemanticVectorRaceAudit:
    """Immutable physical-selection ledger for one AST-cost tier."""

    status: str
    candidate_keys: tuple[str, ...]
    candidate_vector_sha256s: tuple[str, ...]
    desired_vector_sha256: str
    detector_sha256s: tuple[str, ...]
    exact_match_spikes: tuple[tuple[int, ...], ...]
    projection_sha256: str
    currents_pa: tuple[float, ...]
    currents_sha256: str
    stop_cell: int
    tier: RestingGlifTierBeamResult
    tier_sha256: str
    exact_winner_keys: tuple[str, ...]
    winner_key: str | None
    sha256: str


def select_exact_semantic_vector(
    candidate_keys: Sequence[str],
    candidate_outputs: Sequence[Sequence[output_cells.TypedOutputEncoding]],
    desired_outputs: Sequence[output_cells.TypedOutputEncoding],
) -> TypedSemanticVectorRaceAudit:
    """Select one all-case exact candidate from a caller-owned AST-cost tier."""

    if isinstance(candidate_keys, (str, bytes)):
        raise TypeError("candidate_keys must be a sequence of deterministic strings")
    keys = tuple(candidate_keys)
    if not keys or not all(type(key) is str and key for key in keys):
        raise ValueError("candidate_keys must contain nonempty deterministic strings")
    if len(set(keys)) != len(keys):
        raise ValueError("candidate_keys must be unique")

    desired = tuple(desired_outputs)
    if not desired or not all(
        isinstance(item, output_cells.TypedOutputEncoding) for item in desired
    ):
        raise TypeError("desired_outputs must contain physical typed encodings")
    detectors = tuple(
        output_cells.FixedTypedCoincidenceDetector.for_target(target)
        for target in desired
    )
    desired = tuple(detector.target for detector in detectors)
    if not all(item.supported for item in desired):
        raise ValueError("desired_outputs must be physically supported")

    rows = tuple(tuple(row) for row in candidate_outputs)
    if len(rows) != len(keys) or any(len(row) != len(desired) for row in rows):
        raise ValueError("candidate output rows must align with keys and cases")
    if not all(
        isinstance(item, output_cells.TypedOutputEncoding)
        for row in rows
        for item in row
    ):
        raise TypeError("candidate_outputs must contain physical typed encodings")
    rows = tuple(
        tuple(output_cells.TypedOutputCellBank.validate(item) for item in row)
        for row in rows
    )
    predictions_by_case = tuple(zip(*rows, strict=True))
    results_by_case = tuple(
        detector._detect_many_trusted(predictions)
        for detector, predictions in zip(
            detectors, predictions_by_case, strict=True
        )
    )
    spikes = np.ascontiguousarray(
        np.asarray(
            [
                [int(results_by_case[case][candidate].spiked) for case in range(len(desired))]
                for candidate in range(len(rows))
            ],
            dtype=np.float64,
        )
    )
    weights, bias, projection_sha256 = _projection(len(desired))
    candidate_currents = np.clip(
        spikes @ weights + bias,
        _MIN_CURRENT_PA,
        _MAX_CURRENT_PA,
    )
    currents = np.ascontiguousarray(
        np.concatenate((candidate_currents, np.asarray((_STOP_CURRENT_PA,))))
    )
    stop_cell = len(keys)
    selector = CompiledRestingGlifTierBeam.compile(
        candidate_count=len(currents),
        beam_width=1,
        step_count=_RACE_STEPS,
        min_current_pa=_MIN_CURRENT_PA,
        max_current_pa=_MAX_CURRENT_PA,
        inhibition_current_pa=_INHIBITION_CURRENT_PA,
    )
    tier = selector.select(currents, stop_cell=stop_cell)
    if len(tier.rounds) != 1:
        raise RuntimeError("semantic selector must resolve in one physical tier")

    exact_cells = tier.retained_candidate_cells
    exact_keys = tuple(keys[cell] for cell in exact_cells)
    if tier.status == STOP_LEADING_TIER:
        status = STOP_NO_EXACT
        winner = None
    elif tier.status == RETAINED and len(exact_keys) == 1:
        if not all(spikes[exact_cells[0]] == 1.0):
            raise RuntimeError("non-exact semantic vector entered the winning tier")
        status = SELECTED
        winner = exact_keys[0]
    elif tier.status == RETAINED and len(exact_keys) > 1:
        if not all(np.all(spikes[cell] == 1.0) for cell in exact_cells):
            raise RuntimeError("non-exact semantic vector entered the tied tier")
        status = STOP_MULTIPLE_EXACT
        winner = None
    else:
        raise RuntimeError("semantic tier returned an unsupported state")

    currents_tuple = tuple(float(value) for value in currents)
    currents_le = np.ascontiguousarray(currents, dtype="<f8")
    currents_sha256 = sha256(memoryview(currents_le).cast("B")).hexdigest()
    tier_sha256 = _tier_sha256(tier)
    candidate_hashes = tuple(_vector_sha256(row) for row in rows)
    desired_hash = _vector_sha256(desired)
    detector_hashes = tuple(detector.sha256 for detector in detectors)
    spike_rows = tuple(tuple(int(value) for value in row) for row in spikes)
    audit_payload = {
        "status": status,
        "candidate_keys": keys,
        "candidate_vector_sha256s": candidate_hashes,
        "desired_vector_sha256": desired_hash,
        "detector_sha256s": detector_hashes,
        "exact_match_spikes": spike_rows,
        "projection_sha256": projection_sha256,
        "currents_sha256": currents_sha256,
        "stop_cell": stop_cell,
        "tier_sha256": tier_sha256,
        "exact_winner_keys": exact_keys,
        "winner_key": winner,
    }
    return TypedSemanticVectorRaceAudit(
        status=status,
        candidate_keys=keys,
        candidate_vector_sha256s=candidate_hashes,
        desired_vector_sha256=desired_hash,
        detector_sha256s=detector_hashes,
        exact_match_spikes=spike_rows,
        projection_sha256=projection_sha256,
        currents_pa=currents_tuple,
        currents_sha256=currents_sha256,
        stop_cell=stop_cell,
        tier=tier,
        tier_sha256=tier_sha256,
        exact_winner_keys=exact_keys,
        winner_key=winner,
        sha256=sha256(_canonical_json(audit_payload)).hexdigest(),
    )


__all__ = [
    "SELECTED",
    "STOP_MULTIPLE_EXACT",
    "STOP_NO_EXACT",
    "TypedSemanticVectorRaceAudit",
    "select_exact_semantic_vector",
]
