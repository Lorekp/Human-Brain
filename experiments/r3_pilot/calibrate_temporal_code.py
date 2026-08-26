"""Synthetic pre-outcome collision calibration for SparseTemporalCortex.

The fast ``--check`` path validates source hashes and the committed artifact
without constructing cells.  ``--replay-check`` explicitly repeats the small
synthetic physical experiment.  No corpus, held-out split, or C1 outcome is
opened by either path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(__file__).with_name("temporal_code_calibration.json")
SPARSE_SOURCE = REPO_ROOT / "src" / "humanbrain" / "sparse_temporal_cortex.py"
CORTEX_SOURCE = REPO_ROOT / "src" / "humanbrain" / "cortex.py"
SDR_SOURCE = REPO_ROOT / "src" / "humanbrain" / "sdr_union.py"

ARTIFACT_ID = "r3-temporal-code-collision-v2"
SCHEMA_VERSION = 2
SOURCE_COMMIT = "df26cc2"
SPARSE_SOURCE_SHA256 = "0aca721d0c10b88ad91764f499b0b8e6cdde4b08b35e862f7dcfecf42f46fac3"
CORTEX_SOURCE_SHA256 = "16545b390c516b6f5817366946f0f516cec5b049bb9e1f6021184d9e156e0a8c"
SDR_SOURCE_SHA256 = "1de7f6edd914584c4c72d7e42426254590049cfb69cbbfa730afa3179afa63f9"
SYNTHETIC_CODEBOOK_SHA256 = (
    "4ab66dab1657c4e4353267e7fe58371cfc7e833fe9c50f9bb7e9ce3bc34077a9"
)

COLUMN_COUNT = 4_096
ACTIVE_COLUMNS = 11
BASE_DEPTH = 8
LEARNED_CONTEXTS = 128
NOVEL_CONTEXTS = 128
TARGET_TOKEN = 256
PROJECTION_SEED = 17
CONSTRUCTION_SEEDS = (3, 7, 17, 29)
THRESHOLD_ARMS = ((10, 13), (16, 18), (18, 20))
CONTACTS_PER_SEGMENT = 20
REJECT_TOKEN_SDR_OVERLAP = 5


class CalibrationError(RuntimeError):
    """The artifact or replay violates the frozen synthetic calibration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _claim_boundary() -> dict[str, bool]:
    return {
        "capability_evidence": False,
        "held_out_data_accessed": False,
        "independent_confirmatory_seed_set": False,
        "synthetic_collision_calibration": True,
        "threshold_selection_is_evaluation": False,
        "transfer_evidence": False,
    }


def _protocol() -> dict[str, Any]:
    return {
        "active_columns": ACTIVE_COLUMNS,
        "base_cells_per_column": BASE_DEPTH,
        "column_count": COLUMN_COUNT,
        "contacts_per_segment": CONTACTS_PER_SEGMENT,
        "construction_seeds": list(CONSTRUCTION_SEEDS),
        "context_physical_width": 2 * ACTIVE_COLUMNS,
        "context_source_groups": ACTIVE_COLUMNS,
        "context_source_structure": "injective (pyramidal winner, relay twin) pairs",
        "direct_recurrent_source_probe": (
            "diagnose each emitted target context as the next step's W_prev"
        ),
        "learned_contexts": LEARNED_CONTEXTS,
        "learned_target_pair_overlap_gate": (
            "reject max winner-pair overlap >= ceil(theta_match/2)"
        ),
        "novel_contexts": NOVEL_CONTEXTS,
        "projection_seed": PROJECTION_SEED,
        "reject_token_sdr_overlap_at_or_above": REJECT_TOKEN_SDR_OVERLAP,
        "same_token_collision_model": (
            "measured grouped-pair intersections; physical overlap is exactly twice "
            "winner-pair overlap"
        ),
        "target_token": TARGET_TOKEN,
        "threshold_arms": [list(pair) for pair in THRESHOLD_ARMS],
        "token_codebook": "token_sdr_indices(seed=17,n_bits=4096,w_ones=11)",
    }


def _arm_name(theta_match: int, theta_active: int) -> str:
    return f"theta_m{theta_match}_theta_a{theta_active}"


def _source_hashes() -> dict[str, str]:
    return {
        "cortex.py": _sha256(CORTEX_SOURCE.read_bytes()),
        "sdr_union.py": _sha256(SDR_SOURCE.read_bytes()),
        "sparse_temporal_cortex.py": _sha256(SPARSE_SOURCE.read_bytes()),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("artifact_id") == ARTIFACT_ID, "artifact id mismatch")
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
    _require(artifact.get("source_commit") == SOURCE_COMMIT, "source commit drift")
    _require(artifact.get("claim_boundary") == _claim_boundary(), "claim drift")
    _require(artifact.get("protocol") == _protocol(), "protocol drift")
    _require(
        artifact.get("source_sha256")
        == {
            "cortex.py": CORTEX_SOURCE_SHA256,
            "sdr_union.py": SDR_SOURCE_SHA256,
            "sparse_temporal_cortex.py": SPARSE_SOURCE_SHA256,
        },
        "source hash evidence drift",
    )
    codebook_overlap = artifact.get("synthetic_codebook_overlap")
    _require(
        artifact.get("synthetic_codebook_sha256") == SYNTHETIC_CODEBOOK_SHA256,
        "synthetic codebook hash drift",
    )
    _require(isinstance(codebook_overlap, dict), "codebook overlap is missing")
    _require(codebook_overlap.get("unique_rows") == 257, "token SDR collision")
    _require(
        codebook_overlap.get("total_intersection", 0) > 0,
        "synthetic token SDRs do not exercise the shared column sheet",
    )
    _require(
        codebook_overlap.get("max_intersection", REJECT_TOKEN_SDR_OVERLAP)
        < REJECT_TOKEN_SDR_OVERLAP,
        "token SDR proximity gate failed",
    )

    arms = artifact.get("arms")
    _require(isinstance(arms, dict), "threshold arms are missing")
    expected_names = {_arm_name(*pair) for pair in THRESHOLD_ARMS}
    _require(set(arms) == expected_names, "threshold grid drift")
    for theta_match, theta_active in THRESHOLD_ARMS:
        arm = arms[_arm_name(theta_match, theta_active)]
        _require(
            arm.get("thresholds")
            == {"theta_active": theta_active, "theta_match": theta_match},
            "arm threshold drift",
        )
        seeds = arm.get("seeds")
        _require(isinstance(seeds, list) and len(seeds) == 4, "seed rows drift")
        _require(
            [row.get("construction_seed") for row in seeds]
            == list(CONSTRUCTION_SEEDS),
            "construction seed order drift",
        )
        overlap_bound = (theta_match + 1) // 2
        aggregate = arm.get("aggregate")
        _require(isinstance(aggregate, dict), "arm aggregate is missing")
        _require(
            aggregate.get("learned_code_pair_overlap_rejection_bound")
            == overlap_bound,
            "paired-source rejection bound drift",
        )
        for row in seeds:
            parity = row.get("parity")
            _require(isinstance(parity, dict), "parity evidence is missing")
            _require(all(parity.values()), "replay/read-only parity failed")
            _require(
                row["learned_codes"]["contexts"] == LEARNED_CONTEXTS,
                "learned context count drift",
            )
            _require(
                row["novel_probe"]["contexts"] == NOVEL_CONTEXTS,
                "novel context count drift",
            )
            _require(
                row["learned_codes"].get(
                    "physical_overlap_is_exactly_twice_winner_overlap"
                )
                is True,
                "target grouped-pair identity failed",
            )
            _require(
                row["learned_source_context_intersections"].get(
                    "physical_overlap_is_exactly_twice_pair_overlap"
                )
                is True,
                "learned grouped-pair identity failed",
            )
            _require(
                row["novel_probe"]["grouped_pair_overlap"].get(
                    "physical_overlap_is_exactly_twice_pair_overlap"
                )
                is True,
                "novel grouped-pair identity failed",
            )
            max_pair_overlap = row["learned_codes"].get(
                "max_pairwise_winner_intersection"
            )
            _require(isinstance(max_pair_overlap, int), "target overlap evidence missing")
            _require(
                row["learned_codes"].get("pair_overlap_rejection_bound")
                == overlap_bound,
                "seed paired-source rejection bound drift",
            )
            _require(
                row["learned_codes"].get("below_pair_overlap_rejection_bound")
                is bool(max_pair_overlap < overlap_bound),
                "seed paired-source gate drift",
            )
            recurrent = row.get("recurrent_source_diagnostics")
            _require(isinstance(recurrent, dict), "recurrent-source probe is missing")
            _require(
                recurrent.get("contexts") == LEARNED_CONTEXTS,
                "recurrent-source context count drift",
            )
            pre_exposure = row.get("target_pre_exposure_diagnostics")
            _require(
                isinstance(pre_exposure, dict)
                and pre_exposure.get("contexts") == LEARNED_CONTEXTS,
                "target pre-exposure diagnostic count drift",
            )
        _require(
            aggregate == _aggregate(seeds, theta_match=theta_match),
            "arm aggregate does not match seed evidence",
        )

    selection = artifact.get("selection")
    _require(isinstance(selection, dict), "selection record is missing")
    _require(
        selection.get("rule_frozen_before_stable_source_replay") is True,
        "stable-source selection freeze is missing",
    )
    _require(
        selection.get("rule_revision_basis")
        == (
            "paired-source structural correction after a preliminary synthetic "
            "artifact exposed unsafe learned-code overlap"
        ),
        "selection revision provenance drift",
    )
    selected = selection.get("selected_thresholds")
    _require(selected is None or selected in [list(pair) for pair in THRESHOLD_ARMS], "bad selection")
    expected_selected = next(
        (
            list(pair)
            for pair in THRESHOLD_ARMS
            if arms[_arm_name(*pair)]["aggregate"]["screening_verdict"]
            == "PASS_SYNTHETIC_COLLISION_SCREEN"
        ),
        None,
    )
    _require(selected == expected_selected, "selection does not follow frozen rule")


def load_and_validate(path: Path) -> dict[str, Any]:
    current_hashes = _source_hashes()
    _require(
        current_hashes["sparse_temporal_cortex.py"] == SPARSE_SOURCE_SHA256,
        "sparse temporal source changed",
    )
    _require(current_hashes["cortex.py"] == CORTEX_SOURCE_SHA256, "cortex source changed")
    _require(current_hashes["sdr_union.py"] == SDR_SOURCE_SHA256, "SDR source changed")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(artifact, dict), "artifact root must be an object")
    validate_artifact(artifact)
    return artifact


def _tuple_rows_hash(rows: Iterable[Sequence[int]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        values = tuple(int(value) for value in row)
        hasher.update(struct.pack("<I", len(values)))
        hasher.update(struct.pack("<" + "I" * len(values), *values))
    return hasher.hexdigest()


def _snapshot_hash(snapshot: Any) -> str:
    hasher = hashlib.sha256()
    circuit_identity = snapshot.circuit_identity.encode("ascii")
    hasher.update(struct.pack("<I", len(circuit_identity)))
    hasher.update(circuit_identity)
    learned = snapshot.learned
    for column, cells in enumerate(learned.column_cells):
        hasher.update(struct.pack("<II", column, len(cells)))
        hasher.update(struct.pack("<" + "I" * len(cells), *cells))
    for owner, contacts, permanences in learned.segments:
        hasher.update(struct.pack("<II", owner, len(contacts)))
        hasher.update(struct.pack("<" + "I" * len(contacts), *contacts))
        for permanence in permanences:
            hasher.update(struct.pack("<d", permanence))
    hasher.update(struct.pack("<q", learned.learning_step))
    hasher.update(struct.pack("<q", learned.next_reserve_cell))
    for context in (snapshot.previous_context, snapshot.previous_winner_context):
        hasher.update(struct.pack("<I", len(context)))
        if context:
            hasher.update(struct.pack("<" + "I" * len(context), *context))
    hasher.update(struct.pack("<q", snapshot.event_step))
    return hasher.hexdigest()


def _intersection_summary(rows: Sequence[Sequence[int]]) -> dict[str, Any]:
    histogram: Counter[int] = Counter()
    for left_index, left in enumerate(rows):
        left_set = set(int(value) for value in left)
        for right in rows[left_index + 1 :]:
            histogram[len(left_set.intersection(right))] += 1
    pair_count = len(rows) * (len(rows) - 1) // 2
    _require(sum(histogram.values()) == pair_count, "pairwise intersection drift")
    return {
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "max_intersection": max(histogram, default=0),
        "pairs": pair_count,
        "total_intersection": sum(key * count for key, count in histogram.items()),
        "unique_rows": len({tuple(int(value) for value in row) for row in rows}),
    }


def _cross_intersection_summary(
    learned: Sequence[Sequence[int]],
    novel: Sequence[Sequence[int]],
) -> dict[str, Any]:
    histogram: Counter[int] = Counter()
    for left in learned:
        left_set = set(int(value) for value in left)
        for right in novel:
            histogram[len(left_set.intersection(right))] += 1
    pair_count = len(learned) * len(novel)
    _require(sum(histogram.values()) == pair_count, "cross-intersection drift")
    return {
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "max_intersection": max(histogram, default=0),
        "pairs": pair_count,
        "total_intersection": sum(key * count for key, count in histogram.items()),
    }


def _paired_overlap_identity(
    pair_summary: dict[str, Any],
    physical_summary: dict[str, Any],
) -> bool:
    expected_histogram = {
        str(2 * int(overlap)): count
        for overlap, count in pair_summary["histogram"].items()
    }
    return (
        physical_summary["histogram"] == expected_histogram
        and physical_summary["pairs"] == pair_summary["pairs"]
        and physical_summary["max_intersection"]
        == 2 * pair_summary["max_intersection"]
        and physical_summary["total_intersection"]
        == 2 * pair_summary["total_intersection"]
    )


def _diagnostic_summary(rows: Sequence[Any]) -> dict[str, Any]:
    return {
        "active_segments": sum(row.active_segments for row in rows),
        "candidate_segments": sum(row.candidate_segments for row in rows),
        "contexts": len(rows),
        "contexts_with_active_segments": sum(row.active_segments > 0 for row in rows),
        "contexts_with_matching_segments": sum(
            row.matching_segments > 0 for row in rows
        ),
        "matching_segments": sum(row.matching_segments for row in rows),
        "max_connected_contacts": max(
            (row.max_connected_contacts for row in rows), default=0
        ),
        "max_potential_contacts": max(
            (row.max_potential_contacts for row in rows), default=0
        ),
    }


def _step_signature(step: Any) -> dict[str, Any]:
    """Integer-only output signature; exact dataclass equality is checked too."""
    return {
        "allocated_segments": step.allocated_segments,
        "basket_spikes": list(step.basket_spikes),
        "context_spikes": list(step.context_spikes),
        "predicted": bool(step.predicted),
        "predicted_columns": step.predicted_columns,
        "proximal_context_spikes": list(step.proximal_context_spikes),
        "proximal_relay_spikes": list(step.proximal_relay_spikes),
        "proximal_spikes": list(step.proximal_spikes),
        "punished_segments": step.punished_segments,
        "pyramidal_spikes_by_column": [
            list(column) for column in step.pyramidal_spikes_by_column
        ],
        "relay_spikes": list(step.relay_spikes),
        "token_id": step.token_id,
        "winner_cells": list(step.winner_cells),
        "winner_context_spikes": list(step.winner_context_spikes),
    }


def _build_codebook() -> tuple[dict[int, tuple[int, ...]], str, dict[str, Any]]:
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from humanbrain.sdr_union import token_sdr_indices  # noqa: PLC0415

    codebook: dict[int, tuple[int, ...]] = {}
    hasher = hashlib.sha256()
    for token in range(TARGET_TOKEN + 1):
        columns = tuple(
            int(value)
            for value in token_sdr_indices(
                token,
                n_bits=COLUMN_COUNT,
                w_ones=ACTIVE_COLUMNS,
                seed=PROJECTION_SEED,
            )
        )
        _require(len(columns) == ACTIVE_COLUMNS, "synthetic SDR width drift")
        codebook[token] = columns
        hasher.update(struct.pack("<I", token))
        hasher.update(np.asarray(columns, dtype="<u2").tobytes())
    overlap = _intersection_summary(list(codebook.values()))
    _require(overlap["unique_rows"] == len(codebook), "synthetic token SDR collision")
    _require(
        overlap["max_intersection"] < REJECT_TOKEN_SDR_OVERLAP,
        "synthetic token SDRs are too close for theta_m=10 paired contexts",
    )
    return codebook, hasher.hexdigest(), overlap


def _new_circuit(
    codebook: dict[int, tuple[int, ...]],
    *,
    theta_match: int,
    theta_active: int,
    construction_seed: int,
) -> Any:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from humanbrain.sparse_temporal_cortex import (  # noqa: PLC0415
        SparseTemporalConfig,
        SparseTemporalCortex,
    )

    return SparseTemporalCortex(
        codebook,
        config=SparseTemporalConfig(
            column_count=COLUMN_COUNT,
            active_columns=ACTIVE_COLUMNS,
            base_cells_per_column=BASE_DEPTH,
            max_pyramidal_cells=COLUMN_COUNT * BASE_DEPTH,
            contacts_per_segment=CONTACTS_PER_SEGMENT,
            theta_match=theta_match,
            theta_active=theta_active,
            construction_seed=construction_seed,
        ),
    )


def _learn_pair(circuit: Any, context_token: int) -> tuple[Any, Any, Any]:
    circuit.reset_transient()
    context_step = circuit.learn(context_token)
    pre_exposure = circuit.diagnose_context(context_step.context_spikes)
    target_step = circuit.learn(TARGET_TOKEN)
    return context_step, pre_exposure, target_step


def _recognize_pair(circuit: Any, context_token: int) -> tuple[Any, Any, Any]:
    circuit.reset_transient()
    context_step = circuit.recognize(context_token)
    diagnostics = circuit.diagnose_context(context_step.context_spikes)
    target_step = circuit.recognize(TARGET_TOKEN)
    return context_step, diagnostics, target_step


def _seed_run(
    codebook: dict[int, tuple[int, ...]],
    *,
    theta_match: int,
    theta_active: int,
    construction_seed: int,
) -> dict[str, Any]:
    from humanbrain.sparse_temporal_cortex import SparseTemporalCortex

    circuit = _new_circuit(
        codebook,
        theta_match=theta_match,
        theta_active=theta_active,
        construction_seed=construction_seed,
    )
    learned_codes: list[tuple[int, ...]] = []
    learned_contexts: list[tuple[int, ...]] = []
    learned_target_contexts: list[tuple[int, ...]] = []
    target_pre_exposure_diagnostics: list[Any] = []
    allocated_segments = 0
    predicted_before_exposure = 0
    predicted_columns_before_exposure = 0
    for context_token in range(LEARNED_CONTEXTS):
        context_step, pre_exposure, target_step = _learn_pair(circuit, context_token)
        target_pre_exposure_diagnostics.append(pre_exposure)
        _require(
            context_step.context_spikes == context_step.winner_context_spikes,
            "paired-source calibration requires bound-winner recurrent context",
        )
        _require(
            target_step.context_spikes == target_step.winner_context_spikes,
            "target output left the bound-winner recurrent policy",
        )
        learned_contexts.append(context_step.winner_context_spikes)
        learned_codes.append(target_step.winner_cells)
        learned_target_contexts.append(target_step.winner_context_spikes)
        allocated_segments += target_step.allocated_segments
        predicted_before_exposure += int(target_step.predicted)
        predicted_columns_before_exposure += target_step.predicted_columns

    learned_snapshot = circuit.learned_snapshot()
    learned_diagnostics = [
        circuit.diagnose_context(context) for context in learned_contexts
    ]
    recurrent_source_diagnostics = [
        circuit.diagnose_context(context) for context in learned_target_contexts
    ]
    _require(
        circuit.learned_snapshot() == learned_snapshot,
        "diagnose_context mutated learned state",
    )

    circuit.reset_transient()
    replay_origin = circuit.snapshot()
    restored = SparseTemporalCortex.from_snapshot(
        codebook,
        config=circuit.config,
        snapshot=replay_origin,
    )
    original_steps: list[Any] = []
    restored_steps: list[Any] = []
    novel_diagnostics: list[Any] = []
    novel_contexts: list[tuple[int, ...]] = []
    false_predictions = 0
    predicted_columns = 0
    max_predicted_columns = 0
    for context_token in range(LEARNED_CONTEXTS, LEARNED_CONTEXTS + NOVEL_CONTEXTS):
        original_context, original_diag, original_target = _recognize_pair(
            circuit, context_token
        )
        restored_context, restored_diag, restored_target = _recognize_pair(
            restored, context_token
        )
        _require(original_context == restored_context, "context replay drift")
        _require(original_diag == restored_diag, "diagnostic replay drift")
        _require(original_target == restored_target, "target replay drift")
        original_steps.extend((original_context, original_target))
        restored_steps.extend((restored_context, restored_target))
        novel_diagnostics.append(original_diag)
        _require(
            original_context.context_spikes
            == original_context.winner_context_spikes,
            "novel probe left the bound-winner recurrent policy",
        )
        novel_contexts.append(original_context.winner_context_spikes)
        false_predictions += int(original_target.predicted)
        predicted_columns += original_target.predicted_columns
        max_predicted_columns = max(
            max_predicted_columns, original_target.predicted_columns
        )

    learned_after = circuit.learned_snapshot()
    restored_learned_after = restored.learned_snapshot()
    parity = {
        "deep_state_equal": circuit.snapshot() == restored.snapshot(),
        "diagnose_context_read_only": circuit.segment_count == len(learned_snapshot.segments),
        "learned_state_unchanged": (
            learned_after == learned_snapshot == restored_learned_after
        ),
        "recognition_steps_equal": original_steps == restored_steps,
    }
    _require(all(parity.values()), "replay/read-only parity failed")

    unique_codes = len(set(learned_codes))
    winner_code_intersections = _intersection_summary(learned_codes)
    physical_code_intersections = _intersection_summary(learned_target_contexts)
    _require(
        _paired_overlap_identity(winner_code_intersections, physical_code_intersections),
        "target winner/relay overlap lost pair identity",
    )
    max_pair_overlap = winner_code_intersections["max_intersection"]
    pair_overlap_rejection_bound = (theta_match + 1) // 2
    learned_source_pairs = [context[:ACTIVE_COLUMNS] for context in learned_contexts]
    learned_source_pair_intersections = _intersection_summary(learned_source_pairs)
    learned_source_physical_intersections = _intersection_summary(learned_contexts)
    _require(
        _paired_overlap_identity(
            learned_source_pair_intersections,
            learned_source_physical_intersections,
        ),
        "learned source winner/relay overlap lost pair identity",
    )
    novel_source_pairs = [context[:ACTIVE_COLUMNS] for context in novel_contexts]
    cross_pair_intersections = _cross_intersection_summary(
        learned_source_pairs,
        novel_source_pairs,
    )
    cross_physical_intersections = _cross_intersection_summary(
        learned_contexts,
        novel_contexts,
    )
    _require(
        _paired_overlap_identity(cross_pair_intersections, cross_physical_intersections),
        "novel source winner/relay overlap lost pair identity",
    )
    novel_summary = _diagnostic_summary(novel_diagnostics)
    novel_summary.update(
        {
            "false_predictions": false_predictions,
            "grouped_pair_overlap": {
                "learned_vs_novel_physical_sources": cross_physical_intersections,
                "learned_vs_novel_winner_pairs": cross_pair_intersections,
                "physical_overlap_is_exactly_twice_pair_overlap": True,
            },
            "max_predicted_columns": max_predicted_columns,
            "predicted_columns": predicted_columns,
        }
    )
    return {
        "construction_seed": construction_seed,
        "hashes": {
            "final_snapshot_sha256": _snapshot_hash(circuit.snapshot()),
            "learned_contexts_sha256": _tuple_rows_hash(learned_contexts),
            "learned_snapshot_sha256": _snapshot_hash(replay_origin),
            "novel_step_signatures_sha256": _sha256(
                _canonical_json_bytes([_step_signature(step) for step in original_steps])
            ),
            "target_winner_codes_sha256": _tuple_rows_hash(learned_codes),
        },
        "learned_codes": {
            "allocated_segments": allocated_segments,
            "collision_count": LEARNED_CONTEXTS - unique_codes,
            "contexts": LEARNED_CONTEXTS,
            "below_pair_overlap_rejection_bound": (
                max_pair_overlap < pair_overlap_rejection_bound
            ),
            "max_pairwise_winner_intersection": max_pair_overlap,
            "pair_overlap_rejection_bound": pair_overlap_rejection_bound,
            "pairwise_physical_pair_intersections": physical_code_intersections,
            "pairwise_winner_intersections": winner_code_intersections,
            "physical_overlap_is_exactly_twice_winner_overlap": True,
            "predicted_columns_before_exposure": predicted_columns_before_exposure,
            "predicted_contexts_before_exposure": predicted_before_exposure,
            "unique_codes": unique_codes,
        },
        "learned_context_diagnostics": _diagnostic_summary(learned_diagnostics),
        "learned_source_context_intersections": {
            "pairwise_physical_sources": learned_source_physical_intersections,
            "pairwise_winner_pairs": learned_source_pair_intersections,
            "physical_overlap_is_exactly_twice_pair_overlap": True,
        },
        "recurrent_source_diagnostics": _diagnostic_summary(
            recurrent_source_diagnostics
        ),
        "target_pre_exposure_diagnostics": _diagnostic_summary(
            target_pre_exposure_diagnostics
        ),
        "novel_probe": novel_summary,
        "parity": parity,
        "segment_count": circuit.segment_count,
    }


def _aggregate(
    seed_rows: Sequence[dict[str, Any]],
    *,
    theta_match: int,
) -> dict[str, Any]:
    collisions = sum(row["learned_codes"]["collision_count"] for row in seed_rows)
    false_predictions = sum(
        row["novel_probe"]["false_predictions"] for row in seed_rows
    )
    false_matching = sum(
        row["novel_probe"]["matching_segments"] for row in seed_rows
    )
    false_active = sum(row["novel_probe"]["active_segments"] for row in seed_rows)
    learned_active = sum(
        row["learned_context_diagnostics"]["contexts_with_active_segments"]
        for row in seed_rows
    )
    learned_expected = len(seed_rows) * LEARNED_CONTEXTS
    pair_overlap_rejection_bound = (theta_match + 1) // 2
    max_pair_overlap = max(
        (
            row["learned_codes"]["max_pairwise_winner_intersection"]
            for row in seed_rows
        ),
        default=0,
    )
    safe_overlap_seeds = sum(
        row["learned_codes"]["max_pairwise_winner_intersection"]
        < pair_overlap_rejection_bound
        for row in seed_rows
    )
    recurrent_matching = sum(
        row["recurrent_source_diagnostics"]["matching_segments"]
        for row in seed_rows
    )
    recurrent_active = sum(
        row["recurrent_source_diagnostics"]["active_segments"]
        for row in seed_rows
    )
    pre_exposure_matching = sum(
        row["target_pre_exposure_diagnostics"]["matching_segments"]
        for row in seed_rows
    )
    pre_exposure_active = sum(
        row["target_pre_exposure_diagnostics"]["active_segments"]
        for row in seed_rows
    )
    pre_exposure_predicted_columns = sum(
        row["learned_codes"]["predicted_columns_before_exposure"]
        for row in seed_rows
    )
    pre_exposure_predicted_contexts = sum(
        row["learned_codes"]["predicted_contexts_before_exposure"]
        for row in seed_rows
    )
    passes = (
        collisions == 0
        and learned_active == learned_expected
        and safe_overlap_seeds == len(seed_rows)
        and false_predictions == 0
        and false_matching == 0
        and false_active == 0
        and recurrent_matching == 0
        and recurrent_active == 0
        and pre_exposure_matching == 0
        and pre_exposure_active == 0
        and pre_exposure_predicted_columns == 0
        and pre_exposure_predicted_contexts == 0
    )
    return {
        "learned_code_collisions": collisions,
        "learned_code_max_pair_overlap": max_pair_overlap,
        "learned_code_pair_overlap_rejection_bound": pair_overlap_rejection_bound,
        "learned_code_seeds_below_rejection_bound": safe_overlap_seeds,
        "learned_contexts_active": learned_active,
        "learned_contexts_expected": learned_expected,
        "novel_active_segments": false_active,
        "novel_false_predictions": false_predictions,
        "novel_matching_segments": false_matching,
        "pre_exposure_active_segments": pre_exposure_active,
        "pre_exposure_matching_segments": pre_exposure_matching,
        "pre_exposure_predicted_columns": pre_exposure_predicted_columns,
        "pre_exposure_predicted_contexts": pre_exposure_predicted_contexts,
        "recurrent_source_active_segments": recurrent_active,
        "recurrent_source_matching_segments": recurrent_matching,
        "screening_verdict": (
            "PASS_SYNTHETIC_COLLISION_SCREEN"
            if passes
            else "REJECT_SYNTHETIC_COLLISION_SCREEN"
        ),
        "seed_runs": len(seed_rows),
    }


def run_replay() -> dict[str, Any]:
    source_hashes = _source_hashes()
    _require(source_hashes["sparse_temporal_cortex.py"] == SPARSE_SOURCE_SHA256, "source drift")
    _require(source_hashes["cortex.py"] == CORTEX_SOURCE_SHA256, "cortex drift")
    _require(source_hashes["sdr_union.py"] == SDR_SOURCE_SHA256, "SDR source drift")
    codebook, codebook_sha, codebook_overlap = _build_codebook()
    _require(codebook_sha == SYNTHETIC_CODEBOOK_SHA256, "codebook construction drift")
    arms: dict[str, Any] = {}
    for theta_match, theta_active in THRESHOLD_ARMS:
        seed_rows = [
            _seed_run(
                codebook,
                theta_match=theta_match,
                theta_active=theta_active,
                construction_seed=construction_seed,
            )
            for construction_seed in CONSTRUCTION_SEEDS
        ]
        arms[_arm_name(theta_match, theta_active)] = {
            "aggregate": _aggregate(seed_rows, theta_match=theta_match),
            "seeds": seed_rows,
            "thresholds": {
                "theta_active": theta_active,
                "theta_match": theta_match,
            },
        }

    selected: list[int] | None = None
    for thresholds in THRESHOLD_ARMS:
        aggregate = arms[_arm_name(*thresholds)]["aggregate"]
        if aggregate["screening_verdict"] == "PASS_SYNTHETIC_COLLISION_SCREEN":
            selected = list(thresholds)
            break
    return {
        "arms": arms,
        "artifact_id": ARTIFACT_ID,
        "claim_boundary": _claim_boundary(),
        "protocol": _protocol(),
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "selection": {
            "rule": (
                "first listed arm with every learned context active, zero learned-code "
                "collisions, max learned target winner-pair overlap below "
                "ceil(theta_match/2) in every seed, zero direct recurrent-source "
                "matching/active segments, and zero novel false predictions, "
                "matching segments, or active segments; every new learned target "
                "exposure must also have zero pre-exposure matching/active segments "
                "and zero predicted columns across every construction seed"
            ),
            "rule_frozen_before_stable_source_replay": True,
            "rule_revision_basis": (
                "paired-source structural correction after a preliminary synthetic "
                "artifact exposed unsafe learned-code overlap"
            ),
            "selected_thresholds": selected,
        },
        "source_sha256": {
            "cortex.py": CORTEX_SOURCE_SHA256,
            "sdr_union.py": SDR_SOURCE_SHA256,
            "sparse_temporal_cortex.py": SPARSE_SOURCE_SHA256,
        },
        "synthetic_codebook_sha256": codebook_sha,
        "synthetic_codebook_overlap": codebook_overlap,
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true", help="fast artifact check")
    actions.add_argument(
        "--replay-check",
        action="store_true",
        help="repeat the synthetic physical calibration and compare the artifact",
    )
    actions.add_argument(
        "--write",
        action="store_true",
        help="repeat the synthetic physical calibration and write the artifact",
    )
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.check:
            load_and_validate(args.artifact)
            print("temporal-code calibration artifact check: PASS")
            return 0
        started = time.perf_counter()
        observed = run_replay()
        validate_artifact(observed)
        elapsed = time.perf_counter() - started
        if args.write:
            _write_artifact(args.artifact, observed)
            print(f"temporal-code calibration artifact written: {args.artifact} ({elapsed:.1f}s)")
            return 0
        expected = load_and_validate(args.artifact)
        _require(
            _canonical_json_bytes(observed) == _canonical_json_bytes(expected),
            "synthetic replay differs from committed artifact",
        )
        print(f"temporal-code calibration replay check: PASS ({elapsed:.1f}s)")
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError) as error:
        print(f"temporal-code calibration: FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
