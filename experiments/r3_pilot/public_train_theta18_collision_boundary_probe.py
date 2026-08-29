"""One-shot public TRAIN check of the known theta-16 collision at theta 18.

The committed arm-(16,18) localization is the immutable reference.  This probe
replays only Genesis through its zero-based token 7,933 at the already
preregistered arm (18,20), then checks the same eleven foreign segment rows.
It does not select an arm, rerun calibration-v3, or inspect any evaluation
surface.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot import public_train_first_key_failure_probe as reference_probe
from experiments.r3_pilot.calibration_v3_definitions import build_r3_f0_definition
from experiments.r3_pilot.r3_f0_train_input import load_verified_train_input
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    TemporalCircuitStep,
    TemporalStepMode,
)


ACTIVE_COLUMNS = 11
ARM = (18, 20)
CONSTRUCTION_SEED = 17
TARGET_TOKEN_INDEX = 7_933
EXECUTED_STEPS = TARGET_TOKEN_INDEX + 1
REFERENCE_COMMIT = "0a6488ac52d17deb384aec4e20c67883ebef5f2e"
REFERENCE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_first_key_failure_probe.json"
)
REFERENCE_ARTIFACT_REPO_PATH = (
    "experiments/r3_pilot/public_train_first_key_failure_probe.json"
)
REFERENCE_ARTIFACT_SHA256_CANONICAL_LF = (
    "1197fcd76b9257d653f91288f6313eaf75649239269d2511dd50aba2b21e9dbe"
)
OUTPUT_PATH = Path(__file__).with_name(
    "public_train_theta18_collision_boundary_probe.json"
)
CLAIM_BOUNDARY = (
    "one actual public TRAIN-only Genesis replay through zero-based token 7,933 "
    "at the preregistered arm (18,20), seed 17; known arm-(16,18) collision "
    "boundary only; no calibration-v3 reinterpretation/rerun, threshold tuning, "
    "later TRAIN tokens, threshold-selection/candidate-repair, confirmation, held-out, "
    "C1, capability, or transfer evidence"
)
INTERPRETATION = {
    "result_scope": "localized_counterfactual_only",
    "what_pass_means": (
        "the authenticated 8/11 foreign row remains candidate-reachable but is below "
        "theta_match=18 at this one boundary"
    ),
    "threshold_selection_evidence": False,
    "candidate_repair_evidence": False,
    "preexisting_q255_constraint": (
        "independent public q255 production-overlap evidence already STOPs arm (18,20) "
        "at recurrence 82/123 with matching=11, active=0, predicted_columns=0, and "
        "stored-owner drift; this probe neither reruns nor reinterprets it"
    ),
}


def _canonical_lf_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_authenticated_reference(
    path: Path = REFERENCE_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Load the exact committed localization used as this probe's authority."""

    raw = path.read_bytes()
    if _canonical_lf_sha256(raw) != REFERENCE_ARTIFACT_SHA256_CANONICAL_LF:
        raise RuntimeError("reference artifact SHA-256 mismatch")
    payload = json.loads(raw)
    result = payload.get("result", {})
    lifecycle = result.get("lifecycle", {})
    groups = lifecycle.get("segment_groups", [])
    _require(
        payload.get("artifact_id")
        == "public-train-first-recurrent-key-failure-v1",
        "reference artifact identity mismatch",
    )
    _require(
        payload.get("status") == "BANKED_FIRST_RECURRENT_KEY_MISMATCH",
        "reference artifact is not the banked localization",
    )
    _require(
        payload.get("definition", {}).get("arm") == [16, 18],
        "reference artifact arm mismatch",
    )
    _require(
        result.get("global_zero_based_step_index") == TARGET_TOKEN_INDEX
        and result.get("token_id") == 32,
        "reference artifact boundary mismatch",
    )
    _require(
        result.get("expected_source_fanout") == 0
        and result.get("observed", {}).get("matching_segments") == ACTIVE_COLUMNS
        and result.get("observed", {}).get("active_segments") == 0,
        "reference artifact recognition ledger mismatch",
    )
    _require(
        lifecycle.get("classification") == "WINNER_KEY_DIVERGENCE"
        and lifecycle.get("foreign_matching_segments") == ACTIVE_COLUMNS
        and lifecycle.get("exact_source_deactivated_segments") == 0,
        "reference artifact lifecycle classification mismatch",
    )
    _require(len(groups) == 1, "reference artifact segment grouping mismatch")
    group = groups[0]
    _require(
        group.get("source_relation") == "DIVERGED_SOURCE"
        and group.get("segment_ids") == list(range(2_728, 2_739))
        and group.get("origin_global_step_index") == 249
        and group.get("origin_current_source_overlap_cells") == 8
        and group.get("potential_contacts") == 16
        and group.get("connected_contacts") == 16,
        "reference artifact foreign-row provenance mismatch",
    )
    return payload


def _winner_cells(step: TemporalCircuitStep) -> tuple[int, ...]:
    winners = tuple(int(spike.cell_id) for spike in step.winner_spikes)
    if len(winners) != ACTIVE_COLUMNS or len(set(winners)) != ACTIVE_COLUMNS:
        raise RuntimeError("native step did not expose 11 unique winners")
    return winners


def _sha256_u32(values: Sequence[int]) -> str:
    return reference_probe._sha256_u32(values)


def _row_observation(
    *,
    checkpoint: bytes,
    recognition_segment_limit: int,
    current_source_context: tuple[int, ...],
    current_token_columns: tuple[int, ...],
    definition: NativeTemporalCircuitDefinition,
    reference_group: dict[str, Any],
    origin: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    parsed = reference_probe.parse_checkpoint_rows(checkpoint)
    segment_ids = tuple(int(value) for value in reference_group["segment_ids"])
    if any(not 0 <= segment_id < recognition_segment_limit for segment_id in segment_ids):
        raise RuntimeError("reference row is outside the pre-step recognition store")

    context_set = frozenset(current_source_context)
    current_column_set = frozenset(current_token_columns)
    rows: list[tuple[int, ...]] = []
    tick_rows: list[tuple[int, ...]] = []
    owner_cells: list[int] = []
    owner_columns: list[int] = []
    overlap_cells: list[int] = []
    potential_contacts: list[int] = []
    connected_contacts: list[int] = []
    for segment_id in segment_ids:
        row = tuple(
            int(value)
            for value in parsed.sources[segment_id]
            if int(value) != reference_probe.NONE
        )
        ticks = tuple(
            int(value) for value in parsed.permanence_ticks[segment_id][: len(row)]
        )
        owner = int(parsed.owners[segment_id])
        owner_column = reference_probe._owner_column(
            owner,
            cohort_columns=parsed.cohort_columns,
            column_count=definition.config.column_count,
            base_cells_per_column=definition.config.base_cells_per_column,
            novel_race_cap=definition.config.novel_race_cap,
        )
        overlap = sum(source in context_set for source in row)
        connected = sum(
            source in context_set
            and tick >= definition.config.connected_permanence_ticks
            for source, tick in zip(row, ticks, strict=True)
        )
        rows.append(row)
        tick_rows.append(ticks)
        owner_cells.append(owner)
        owner_columns.append(owner_column)
        overlap_cells.append(overlap)
        potential_contacts.append(2 * overlap)
        connected_contacts.append(2 * connected)

    uniform_rows = len(set(rows)) == 1
    uniform_ticks = len(set(tick_rows)) == 1
    all_candidate_reachable = all(value > 0 for value in overlap_cells)
    all_below_theta_match = all(
        value < definition.config.theta_match for value in potential_contacts
    )
    any_matching = any(
        value >= definition.config.theta_match for value in potential_contacts
    )
    any_active = any(
        value >= definition.config.theta_active for value in connected_contacts
    )
    owners_in_current_columns = any(
        column in current_column_set for column in owner_columns
    )
    origin_context = tuple(int(value) for value in origin["source_context"])
    origin_overlap = len(frozenset(origin_context) & context_set)

    observation = {
        "segment_ids": list(segment_ids),
        "recognition_segment_limit_before_step": recognition_segment_limit,
        "checkpoint_segment_count_after_step": parsed.segment_count,
        "owner_cells": owner_cells,
        "owner_columns": owner_columns,
        "origin_global_step_index": int(origin["global_step_index"]),
        "origin_target_token_id": int(origin["target_token_id"]),
        "origin_source_context": list(origin_context),
        "origin_source_context_sha256": _sha256_u32(origin_context),
        "current_source_context_sha256": _sha256_u32(current_source_context),
        "origin_current_source_overlap_cells": origin_overlap,
        "row_sources": list(rows[0]) if uniform_rows else [list(row) for row in rows],
        "permanence_ticks": (
            list(tick_rows[0]) if uniform_ticks else [list(row) for row in tick_rows]
        ),
        "potential_contacts": (
            potential_contacts[0]
            if len(set(potential_contacts)) == 1
            else potential_contacts
        ),
        "connected_contacts": (
            connected_contacts[0]
            if len(set(connected_contacts)) == 1
            else connected_contacts
        ),
        "candidate_reachable": all_candidate_reachable,
        "below_theta_match": all_below_theta_match,
        "matching": any_matching,
        "active": any_active,
        "owners_in_current_token_columns": owners_in_current_columns,
    }
    checks = {
        "known_origin_reproduced": (
            int(origin["global_step_index"])
            == int(reference_group["origin_global_step_index"])
            and list(segment_ids) == list(origin["segment_ids"])
            and owner_cells == list(origin["owner_cells"])
            and _sha256_u32(origin_context)
            == reference_group["origin_source_context_sha256"]
            and origin_overlap
            == int(reference_group["origin_current_source_overlap_cells"])
        ),
        "known_rows_reproduced": (
            uniform_rows
            and uniform_ticks
            and list(rows[0]) == list(reference_group["row_sources"])
            and list(tick_rows[0]) == list(reference_group["permanence_ticks"])
            and owner_cells == list(reference_group["owner_cells"])
            and owner_columns == list(reference_group["owner_columns"])
            and potential_contacts == [16] * ACTIVE_COLUMNS
            and connected_contacts == [16] * ACTIVE_COLUMNS
        ),
        "known_rows_candidate_reachable": all_candidate_reachable,
        "known_rows_below_theta_match": (
            all_below_theta_match and not any_matching and not any_active
        ),
    }
    return observation, checks


def _run_boundary(
    definition: NativeTemporalCircuitDefinition,
    tokens: np.ndarray,
    *,
    book: object,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], object]:
    if len(tokens) != EXECUTED_STEPS:
        raise RuntimeError("boundary runner requires exactly 7,934 Genesis tokens")
    reference_result = reference["result"]
    reference_lifecycle = reference_result["lifecycle"]
    reference_group = reference_lifecycle["segment_groups"][0]
    token_columns = dict(definition.token_codes)
    previous_winners: tuple[int, ...] | None = None
    previous_context: tuple[int, ...] | None = None
    live_segments = 0
    origin: dict[str, Any] | None = None

    with reference_probe._synthetic_preoutcome_authority():
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for token_index, raw_token in enumerate(tokens):
                token_id = int(raw_token)
                recognition_segment_limit = live_segments
                step = circuit.advance_block(
                    [token_id],
                    mode=TemporalStepMode.LEARN,
                    reset_before=token_index == 0,
                )[0]
                winners = _winner_cells(step)

                if token_index == int(reference_group["origin_global_step_index"]):
                    if previous_context is None:
                        raise RuntimeError("reference origin has no prior context")
                    origin = {
                        "global_step_index": token_index,
                        "target_token_id": token_id,
                        "segment_ids": list(
                            range(
                                recognition_segment_limit,
                                recognition_segment_limit
                                + int(step.allocated_segments),
                            )
                        ),
                        "source_context": list(previous_context),
                        "owner_cells": list(winners),
                    }

                if token_index == TARGET_TOKEN_INDEX:
                    if previous_winners is None or previous_context is None or origin is None:
                        raise RuntimeError("boundary replay lost required prior provenance")
                    checkpoint = circuit.checkpoint()
                    identity = circuit.checkpoint_identity()
                    current_columns = tuple(
                        int(value) for value in token_columns[token_id]
                    )
                    row, row_checks = _row_observation(
                        checkpoint=checkpoint,
                        recognition_segment_limit=recognition_segment_limit,
                        current_source_context=previous_context,
                        current_token_columns=current_columns,
                        definition=definition,
                        reference_group=reference_group,
                        origin=origin,
                    )
                    observed = {
                        "candidate_segments": int(step.candidate_segments),
                        "matching_segments": int(step.matching_segments),
                        "active_segments": int(step.active_segments),
                        "allocated_segments": int(step.allocated_segments),
                        "predicted_columns": int(step.predicted_columns),
                        "predicted": bool(step.predicted),
                        "punished_segments": int(step.punished_segments),
                        "postings_visited": int(step.postings_visited),
                    }
                    checks = {
                        "reference_authenticated": True,
                        "stopped_at_exact_public_boundary": (
                            token_index == TARGET_TOKEN_INDEX
                            and int(book.ordinal) == 0
                            and str(book.short_name) == "Genesis"
                            and token_id == int(reference_result["token_id"])
                            and int(identity.event_count) == EXECUTED_STEPS
                        ),
                        "current_source_reproduced": (
                            list(previous_winners)
                            == list(reference_result["previous_winners"])
                            and list(previous_context)
                            == list(reference_lifecycle["current_source_context"])
                            and list(current_columns)
                            == list(reference_lifecycle["current_token_columns"])
                        ),
                        "known_origin_reproduced": row_checks[
                            "known_origin_reproduced"
                        ],
                        "known_rows_reproduced": row_checks[
                            "known_rows_reproduced"
                        ],
                        "known_rows_candidate_reachable": (
                            row_checks["known_rows_candidate_reachable"]
                            and observed["candidate_segments"]
                            == int(reference_result["observed"]["candidate_segments"])
                            and observed["postings_visited"]
                            == int(reference_result["observed"]["postings_visited"])
                        ),
                        "known_rows_below_theta_match": row_checks[
                            "known_rows_below_theta_match"
                        ],
                        "global_recognition_below_boundary": (
                            observed["matching_segments"] == 0
                            and observed["active_segments"] == 0
                        ),
                        "current_key_outcome_preserved": (
                            observed["allocated_segments"] == ACTIVE_COLUMNS
                            and observed["predicted_columns"] == 0
                            and observed["predicted"] is False
                            and observed["punished_segments"] == 0
                        ),
                    }
                    failures = [name for name, passed in checks.items() if not passed]
                    return (
                        {
                            "book_ordinal": int(book.ordinal),
                            "book_short_name": str(book.short_name),
                            "book_zero_based_token_index": token_index,
                            "global_zero_based_step_index": token_index,
                            "token_id": token_id,
                            "previous_winners": list(previous_winners),
                            "previous_winner_key_sha256": _sha256_u32(
                                previous_winners
                            ),
                            "current_source_context": list(previous_context),
                            "current_source_context_sha256": _sha256_u32(
                                previous_context
                            ),
                            "reference_current_source_reproduced": checks[
                                "current_source_reproduced"
                            ],
                            "observed": observed,
                            "row_provenance": row,
                            "checks": checks,
                            "failures": failures,
                        },
                        identity,
                    )

                live_segments += int(step.allocated_segments)
                previous_winners = winners
                previous_context = tuple(
                    int(value) for value in step.context[:ACTIVE_COLUMNS]
                )

    raise RuntimeError("boundary replay terminated before the frozen token")


def validate_artifact(payload: dict[str, Any]) -> None:
    """Validate the compact PASS/STOP artifact without executing the probe."""

    _require(
        payload.get("artifact_id")
        == "public-train-theta18-collision-boundary-v1",
        "artifact identity mismatch",
    )
    _require(payload.get("schema_version") == 1, "artifact schema mismatch")
    _require(payload.get("claim_boundary") == CLAIM_BOUNDARY, "claim boundary drift")
    _require(
        payload.get("interpretation") == INTERPRETATION,
        "artifact localized-counterfactual interpretation drift",
    )
    _require(
        payload.get("reference")
        == {
            "artifact_id": "public-train-first-recurrent-key-failure-v1",
            "artifact_path": REFERENCE_ARTIFACT_REPO_PATH,
            "artifact_sha256_canonical_lf": (
                REFERENCE_ARTIFACT_SHA256_CANONICAL_LF
            ),
            "commit": REFERENCE_COMMIT,
        },
        "artifact reference authority mismatch",
    )
    _require(
        payload.get("definition", {}).get("arm") == list(ARM)
        and payload.get("definition", {}).get("construction_seed")
        == CONSTRUCTION_SEED,
        "artifact definition mismatch",
    )
    _require(
        payload.get("probe_bound", {}).get("executed_steps") == EXECUTED_STEPS
        and payload.get("probe_bound", {}).get("last_zero_based_token_index")
        == TARGET_TOKEN_INDEX,
        "artifact probe bound mismatch",
    )
    _require(
        payload.get("train_input", {}).get("held_out_books_materialized") == 0,
        "artifact crossed into held-out material",
    )
    result = payload.get("result", {})
    checks = result.get("checks", {})
    failures = result.get("failures", [])
    expected_failures = sorted(name for name, passed in checks.items() if not passed)
    _require(
        sorted(failures) == expected_failures,
        "artifact failure ledger does not match its checks",
    )
    status = payload.get("status")
    if status == "PASS_LOCALIZED_THETA18_COLLISION_BOUNDARY":
        _require(bool(checks) and all(checks.values()), "PASS artifact has failed checks")
        _require(failures == [], "PASS artifact has failure reasons")
    elif status == "STOP_THETA18_KNOWN_COLLISION_BOUNDARY":
        _require(bool(failures), "STOP artifact has no failure reason")
    else:
        raise RuntimeError("artifact status mismatch")


def read_banked_artifact(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    """Read and validate an already-spent result; never construct a circuit."""

    payload = json.loads(path.read_bytes())
    validate_artifact(payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; this one-shot boundary probe refuses overwrite")
    return args


def main() -> int:
    args = _parse_args()
    reference = load_authenticated_reference()
    verified = load_verified_train_input()
    books = verified.iter_books()
    book = next(books)
    if int(book.ordinal) != 0 or str(book.short_name) != "Genesis":
        raise RuntimeError("first public TRAIN book identity drifted")
    all_tokens = np.frombuffer(book.token_bytes_le_i32, dtype="<i4")
    if len(all_tokens) <= TARGET_TOKEN_INDEX:
        raise RuntimeError("Genesis is shorter than the frozen collision boundary")
    tokens = all_tokens[:EXECUTED_STEPS]

    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=False)
    production = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    definition = NativeTemporalCircuitDefinition(
        replace(
            production.config,
            segment_capacity=(EXECUTED_STEPS - 1) * ACTIVE_COLUMNS,
        ),
        production.token_codes,
        production.threshold_offsets_mv,
    )
    result, identity = _run_boundary(
        definition,
        tokens,
        book=book,
        reference=reference,
    )
    native_identity = loaded_library_identity()
    status = (
        "PASS_LOCALIZED_THETA18_COLLISION_BOUNDARY"
        if not result["failures"]
        else "STOP_THETA18_KNOWN_COLLISION_BOUNDARY"
    )
    payload = {
        "artifact_id": "public-train-theta18-collision-boundary-v1",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "interpretation": INTERPRETATION,
        "reference": {
            "artifact_id": reference["artifact_id"],
            "artifact_path": REFERENCE_ARTIFACT_REPO_PATH,
            "artifact_sha256_canonical_lf": (
                REFERENCE_ARTIFACT_SHA256_CANONICAL_LF
            ),
            "commit": REFERENCE_COMMIT,
        },
        "probe_bound": {
            "book_ordinal": int(book.ordinal),
            "book_short_name": str(book.short_name),
            "executed_steps": EXECUTED_STEPS,
            "last_zero_based_token_index": TARGET_TOKEN_INDEX,
        },
        "train_input": {
            "book_ordinal": int(book.ordinal),
            "book_short_name": str(book.short_name),
            "book_token_sha256_le_i32": str(book.token_sha256_le_i32),
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "held_out_books_materialized": 0,
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "bounded_segment_capacity": definition.config.segment_capacity,
            "production_circuit_identity_sha256": (
                production.circuit_identity_sha256()
            ),
            "bounded_circuit_identity_sha256": definition.circuit_identity_sha256(),
            "protocol_sha256": protocol_authority.protocol_sha256(),
        },
        "result": result,
        "checkpoint_identity_after_boundary": {
            "segment_count": identity.segment_count,
            "cohort_count": identity.cohort_count,
            "next_reserve_cell": identity.next_reserve_cell,
            "event_count": identity.event_count,
            "learning_step": identity.learning_step,
            "logical_state_sha256": identity.logical_state_sha256,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
        },
    }
    validate_artifact(payload)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    raw_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output}")
    print(f"artifact_sha256={raw_sha256}")
    print(f"status={status}")
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
