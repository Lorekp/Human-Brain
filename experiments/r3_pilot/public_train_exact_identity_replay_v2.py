"""One-shot public TRAIN replay for persistent exact transition identity.

The semantic key is ``(previous winner tuple, observed token)``.  A novel key
must allocate eleven segments and elect a globally new ordered winner tuple; a
repeated key must allocate nothing and re-elect its stored tuple exactly.
Active-path prediction is same-token surface telemetry and never gates this
identity verdict.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot import public_train_first_owner_outcome_failure_probe as v1
from experiments.r3_pilot.calibration_v3_definitions import build_r3_f0_definition
from experiments.r3_pilot.r3_f0_train_input import load_verified_train_input
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    NativeTemporalRuntimeError,
    NativeTemporalRuntimeStatus,
    TemporalStepMode,
)


ACTIVE_COLUMNS = v1.ACTIVE_COLUMNS
ARM = v1.ARM
CONSTRUCTION_SEED = v1.CONSTRUCTION_SEED
EXPECTED = v1.EXPECTED
SPEND_MARKER_PATH = Path(__file__).with_suffix(".spend.json")
OUTPUT_PATH = Path(__file__).with_suffix(".json")
V1_INTERPRETATION_PATH = Path(__file__).with_name(
    "public_train_first_owner_outcome_failure_probe.interpretation.json"
)
DIAGNOSTIC_COUNTER_CONTRACT = (
    "0 <= active <= matching <= candidate <= postings; prediction and "
    "punishment counters are non-gating telemetry"
)
CRITERION = {
    "key": "K=(previous ordered winner tuple, observed token)",
    "novel": "allocated=11 and globally-new ordered winner tuple",
    "repeated": "allocated=0 and winners equal stored O(K)",
    "segments": "terminal native segment_count=11*accepted novel keys",
    "telemetry": (
        "predicted, predicted_columns, active, matching, candidate, and "
        "punishment never determine the identity verdict"
    ),
}
CLAIM_BOUNDARY = (
    "one spent public TRAIN-only exact-identity replay at arm (16,18), seed 17, "
    "over the canonical 49-book stream in frozen order, stopping on the first "
    "identity/native/diagnostic-shape failure or otherwise finishing TRAIN; no "
    "calibration-v3 reinterpretation/retry, threshold change, confirmation, "
    "held-out, C1, capability, or transfer evidence"
)


class ExactIdentityLedger(v1.OwnerOutcomeLedger):
    """Compact exact ``K -> O(K)`` ledger with surface telemetry non-gating."""

    def observe(
        self,
        *,
        reset_before: bool,
        step: object,
        book_ordinal: int,
        book_token_index: int,
        global_step_index: int,
    ) -> dict[str, Any]:
        try:
            token_id = self._counter(step, "token_id")
            event_count = self._counter(step, "event_count")
            candidate = self._counter(step, "candidate_segments")
            matching = self._counter(step, "matching_segments")
            active = self._counter(step, "active_segments")
            postings = self._counter(step, "postings_visited")
            allocated = self._counter(step, "allocated_segments")
            predicted_columns = self._counter(step, "predicted_columns")
            punished = self._counter(step, "punished_segments")
            predicted = step.predicted
            if type(predicted) is not bool:
                raise ValueError("predicted")
            winners = self._winners(step)
        except (AttributeError, ValueError) as error:
            return self._outcome(
                self._failure(
                    "STEP_SHAPE_CONTRACT",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=None,
                    expected="non-negative counters, boolean prediction, 11 unique winners",
                    observed=str(error),
                )
            )

        diagnostic = {
            "candidate_segments": candidate,
            "matching_segments": matching,
            "active_segments": active,
            "postings_visited": postings,
            "allocated_segments": allocated,
            "predicted_columns": predicted_columns,
            "predicted": int(predicted),
            "punished_segments": punished,
        }
        if (
            active > matching
            or matching > candidate
            or candidate > postings
            or predicted_columns > ACTIVE_COLUMNS
            or allocated > ACTIVE_COLUMNS
        ):
            return self._outcome(
                self._failure(
                    "DIAGNOSTIC_SHAPE_OR_ORDER",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=token_id,
                    expected=DIAGNOSTIC_COUNTER_CONTRACT,
                    observed=[active, matching, candidate, postings],
                    diagnostic=diagnostic,
                ),
                diagnostic=diagnostic,
                winners=winners,
            )

        if reset_before:
            if book_token_index != 0 or event_count != 1 or allocated != 0:
                return self._outcome(
                    self._failure(
                        "BOOK_RESET_IDENTITY",
                        book_ordinal=book_ordinal,
                        book_token_index=book_token_index,
                        global_step_index=global_step_index,
                        token_id=token_id,
                        expected=[0, 1, 0],
                        observed=[book_token_index, event_count, allocated],
                        diagnostic=diagnostic,
                    ),
                    diagnostic=diagnostic,
                    winners=winners,
                )
            current_tag = self._intern_winners(winners)
            self.previous_source_tag = current_tag
            self.previous_winners = winners
            self.local_event = 1
            self.resets += 1
            self.calls += 1
            return self._outcome(None, diagnostic=diagnostic, winners=winners)

        if self.previous_source_tag is None or self.previous_winners is None:
            return self._outcome(
                self._failure(
                    "MISSING_BOOK_RESET",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=token_id,
                    expected="reset context seed",
                    observed="transition without prior winners",
                    diagnostic=diagnostic,
                ),
                diagnostic=diagnostic,
                winners=winners,
            )
        if event_count != self.local_event + 1 or book_token_index != self.local_event:
            return self._outcome(
                self._failure(
                    "LOCAL_EVENT_COUNT",
                    book_ordinal=book_ordinal,
                    book_token_index=book_token_index,
                    global_step_index=global_step_index,
                    token_id=token_id,
                    expected=[self.local_event + 1, self.local_event],
                    observed=[event_count, book_token_index],
                    diagnostic=diagnostic,
                ),
                diagnostic=diagnostic,
                winners=winners,
                source=self.previous_winners,
            )

        source_tag = self.previous_source_tag
        source = self.previous_winners
        key_record = self.key_index.lookup(source_tag, token_id)
        observed_tag = self.lookup_winner(winners)
        if key_record is None:
            if allocated != ACTIVE_COLUMNS:
                return self._outcome(
                    self._failure(
                        "NOVEL_KEY_DID_NOT_ALLOCATE_IDENTITY",
                        book_ordinal=book_ordinal,
                        book_token_index=book_token_index,
                        global_step_index=global_step_index,
                        token_id=token_id,
                        expected=ACTIVE_COLUMNS,
                        observed=allocated,
                        diagnostic=diagnostic,
                        source_winners=list(source),
                        observed_winners=list(winners),
                    ),
                    diagnostic=diagnostic,
                    winners=winners,
                    source=source,
                    key_was_known=False,
                )
            if observed_tag is not None:
                return self._outcome(
                    self._failure(
                        "NOVEL_KEY_REUSED_OWNER_TUPLE",
                        book_ordinal=book_ordinal,
                        book_token_index=book_token_index,
                        global_step_index=global_step_index,
                        token_id=token_id,
                        expected="globally new ordered winner tuple",
                        observed=list(winners),
                        diagnostic=diagnostic,
                        source_winners=list(source),
                        prior_winner_tag=observed_tag,
                    ),
                    diagnostic=diagnostic,
                    winners=winners,
                    source=source,
                    key_was_known=False,
                )
            if self.key_count >= len(self.key_tokens):
                raise RuntimeError("exact identity key ledger capacity exhausted")
            owner_tag = self._intern_winners(winners)
            record = self.key_count
            self.key_source_tags[record] = source_tag
            self.key_owner_tags[record] = owner_tag
            self.key_tokens[record] = token_id
            self.key_global_steps[record] = global_step_index
            self.key_book_ordinals[record] = book_ordinal
            self.key_token_indices[record] = book_token_index
            self.key_index.insert(source_tag, token_id, record)
            self.key_count += 1
            current_tag = owner_tag
            key_was_known = False
        else:
            owner_tag = int(self.key_owner_tags[key_record])
            expected_owners = self.winner_tuple(owner_tag)
            if allocated != 0 or winners != expected_owners:
                return self._outcome(
                    self._failure(
                        "REPEATED_KEY_IDENTITY_DRIFT",
                        book_ordinal=book_ordinal,
                        book_token_index=book_token_index,
                        global_step_index=global_step_index,
                        token_id=token_id,
                        expected=[0, list(expected_owners)],
                        observed=[allocated, list(winners)],
                        diagnostic=diagnostic,
                        source_winners=list(source),
                        expected_key_record_index=key_record,
                    ),
                    diagnostic=diagnostic,
                    winners=winners,
                    source=source,
                    key_was_known=True,
                    key_record_index=key_record,
                    expected_owners=expected_owners,
                )
            current_tag = owner_tag
            key_was_known = True
            self.repeats += 1

        self.previous_source_tag = current_tag
        self.previous_winners = winners
        self.local_event += 1
        self.calls += 1
        return self._outcome(
            None,
            diagnostic=diagnostic,
            winners=winners,
            source=source,
            key_was_known=key_was_known,
            key_record_index=key_record,
        )

    @staticmethod
    def _outcome(
        violation: dict[str, Any] | None,
        *,
        diagnostic: dict[str, int] | None = None,
        winners: tuple[int, ...] | None = None,
        source: tuple[int, ...] | None = None,
        key_was_known: bool | None = None,
        key_record_index: int | None = None,
        expected_owners: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        return {
            "violation": violation,
            "key_was_known": key_was_known,
            "key_record_index": key_record_index,
            "source_winners": None if source is None else list(source),
            "observed_winners": None if winners is None else list(winners),
            "expected_owner_cells": (
                None if expected_owners is None else list(expected_owners)
            ),
            "diagnostic_counters": diagnostic,
        }


def create_spend_marker(
    *,
    marker_path: Path = SPEND_MARKER_PATH,
    output_path: Path = OUTPUT_PATH,
    authority: dict[str, Any],
) -> dict[str, Any]:
    if marker_path.exists() or output_path.exists():
        raise RuntimeError("refusing spent v2 replay; marker or artifact exists")
    payload = {
        "schema_version": 1,
        "experiment": "public-train-exact-identity-replay-v2",
        "state": "EXACT_IDENTITY_V2_REPLAY_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arm": list(ARM),
        "construction_seed": CONSTRUCTION_SEED,
        "criterion": CRITERION,
        "claim_boundary": CLAIM_BOUNDARY,
        "canonical_artifact_path": v1._display_path(output_path),
        "probe_source_sha256": v1._sha256_file(Path(__file__)),
        "authority": authority,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with marker_path.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing spent v2 replay; marker exists") from error
    return {
        "path": v1._display_path(marker_path),
        "sha256": v1._sha256_file(marker_path),
        "sha256_canonical_lf": v1._canonical_lf_sha256(marker_path.read_bytes()),
        "state": payload["state"],
        "exclusive_create": True,
        "flushed_and_fsynced_before_run": True,
        "probe_source_sha256": payload["probe_source_sha256"],
    }


def _new_metrics() -> dict[str, int]:
    return {
        "executed_token_steps": 0,
        "executed_transition_steps": 0,
        "allocated_segments": 0,
        "punished_segments": 0,
        "predicted_steps": 0,
        "native_step_call_ns_total": 0,
        "native_step_call_ns_max": 0,
        "maximum_candidate_segments": 0,
        "maximum_matching_segments": 0,
        "maximum_active_segments": 0,
        "maximum_postings_visited": 0,
    }


def _record_metrics(metrics: dict[str, int], step: object, call_ns: int) -> None:
    metrics["executed_token_steps"] += 1
    metrics["allocated_segments"] += int(step.allocated_segments)
    metrics["punished_segments"] += int(step.punished_segments)
    metrics["predicted_steps"] += int(step.predicted)
    metrics["native_step_call_ns_total"] += call_ns
    metrics["native_step_call_ns_max"] = max(
        metrics["native_step_call_ns_max"], call_ns
    )
    for metric, field in (
        ("maximum_candidate_segments", "candidate_segments"),
        ("maximum_matching_segments", "matching_segments"),
        ("maximum_active_segments", "active_segments"),
        ("maximum_postings_visited", "postings_visited"),
    ):
        metrics[metric] = max(metrics[metric], int(getattr(step, field)))


def _timing(
    metrics: dict[str, int],
    *,
    wall_ns: int,
    ledger: ExactIdentityLedger,
    tracked_segments: int,
) -> dict[str, Any]:
    executed = metrics["executed_token_steps"]
    return {
        **metrics,
        "replay_wall_ns": wall_ns,
        "steps_per_second": 0.0 if wall_ns == 0 else executed * 1e9 / wall_ns,
        "native_step_call_ns_mean": (
            0.0 if executed == 0 else metrics["native_step_call_ns_total"] / executed
        ),
        "ledger_key_count": ledger.key_count,
        "ledger_repeat_count": ledger.repeats,
        "ledger_winner_tuple_count": ledger.winner_count,
        "ledger_book_resets": ledger.resets,
        "ledger_preallocated_storage_bytes": ledger.storage_bytes,
        "tracked_segment_count": tracked_segments,
        "gc_during_replay": "disabled",
        "single_token_native_calls": True,
    }


def _identity_record(identity: object) -> dict[str, Any]:
    return {
        "segment_count": identity.segment_count,
        "cohort_count": identity.cohort_count,
        "next_reserve_cell": identity.next_reserve_cell,
        "event_count": identity.event_count,
        "learning_step": identity.learning_step,
        "checkpoint_bytes": identity.checkpoint_bytes,
        "body_sha256": identity.body_sha256,
        "logical_state_sha256": identity.logical_state_sha256,
    }


def run_stream(
    circuit: NativeTemporalCircuit,
    books: tuple[tuple[object, np.ndarray], ...],
    *,
    ledger: ExactIdentityLedger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = _new_metrics()
    tracked_segments = 0
    global_step = 0
    started_ns = time.perf_counter_ns()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for book, tokens in books:
            for token_index, raw_token in enumerate(tokens):
                token_id = int(raw_token)
                segment_count_before = tracked_segments
                call_started_ns = time.perf_counter_ns()
                try:
                    emitted = circuit.advance_block(
                        [token_id],
                        mode=TemporalStepMode.LEARN,
                        reset_before=token_index == 0,
                    )
                except NativeTemporalRuntimeError as error:
                    wall_ns = time.perf_counter_ns() - started_ns
                    result = {
                        "kind": "FIRST_NATIVE_FAILURE",
                        "book_ordinal": int(book.ordinal),
                        "book_short_name": str(book.short_name),
                        "book_zero_based_token_index": token_index,
                        "global_zero_based_step_index": global_step,
                        "token_id": token_id,
                        "native_status": error.status.name,
                        "native_status_code": int(error.status),
                        "native_scientific_stop": (
                            error.status is NativeTemporalRuntimeStatus.SCIENTIFIC_STOP
                        ),
                        "error_message": str(error),
                        "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
                    }
                    return result, _timing(
                        metrics,
                        wall_ns=wall_ns,
                        ledger=ledger,
                        tracked_segments=tracked_segments,
                    )
                call_ns = time.perf_counter_ns() - call_started_ns
                if len(emitted) != 1 or int(emitted[0].token_id) != token_id:
                    raise RuntimeError("single-token native call returned the wrong step")
                step = emitted[0]
                _record_metrics(metrics, step, call_ns)
                if token_index != 0:
                    metrics["executed_transition_steps"] += 1
                tracked_segments += int(step.allocated_segments)
                outcome = ledger.observe(
                    reset_before=token_index == 0,
                    step=step,
                    book_ordinal=int(book.ordinal),
                    book_token_index=token_index,
                    global_step_index=global_step,
                )
                if outcome["violation"] is not None:
                    identity = circuit.checkpoint_identity()
                    wall_ns = time.perf_counter_ns() - started_ns
                    result = {
                        "kind": "FIRST_EXACT_IDENTITY_FAILURE",
                        "book_ordinal": int(book.ordinal),
                        "book_short_name": str(book.short_name),
                        "book_zero_based_token_index": token_index,
                        "global_zero_based_step_index": global_step,
                        "token_id": token_id,
                        "key_was_known": outcome["key_was_known"],
                        "violation": outcome["violation"],
                        "source_winners": outcome["source_winners"],
                        "observed_winners": outcome["observed_winners"],
                        "expected_owner_cells": outcome["expected_owner_cells"],
                        "observed_diagnostics": outcome["diagnostic_counters"],
                        "segment_accounting": {
                            "before_step": segment_count_before,
                            "after_step_from_native_counters": tracked_segments,
                            "after_step_checkpoint_identity": identity.segment_count,
                            "accepted_novel_keys_before_failure": ledger.key_count,
                            "accepted_novel_segment_count": (
                                ledger.key_count * ACTIVE_COLUMNS
                            ),
                        },
                        "checkpoint_identity_after_failure": _identity_record(identity),
                        "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
                    }
                    return result, _timing(
                        metrics,
                        wall_ns=wall_ns,
                        ledger=ledger,
                        tracked_segments=tracked_segments,
                    )
                expected_segments = ledger.key_count * ACTIVE_COLUMNS
                if tracked_segments != expected_segments:
                    raise RuntimeError(
                        "native allocation counters diverged from accepted novel identities"
                    )
                global_step += 1
                if (token_index + 1) % 32_768 == 0:
                    print(
                        f"{book.short_name}: {token_index + 1:,}/{book.token_count:,} "
                        f"tokens, {ledger.key_count:,} keys, {ledger.repeats:,} repeats",
                        flush=True,
                    )
            print(
                f"{book.short_name}: complete, {ledger.key_count:,} keys, "
                f"{ledger.repeats:,} repeats",
                flush=True,
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    identity = circuit.checkpoint_identity()
    wall_ns = time.perf_counter_ns() - started_ns
    expected_segments = ledger.key_count * ACTIVE_COLUMNS
    if identity.segment_count != expected_segments:
        result = {
            "kind": "FIRST_EXACT_IDENTITY_FAILURE",
            "violation": {
                "kind": "TERMINAL_SEGMENT_COUNT_DIVERGENCE",
                "expected": expected_segments,
                "observed": identity.segment_count,
            },
            "checkpoint_identity_after_failure": _identity_record(identity),
            "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
        }
    else:
        result = {
            "kind": "TRAIN_EXACT_IDENTITIES_COMPLETE",
            "completed_books": len(books),
            "completed_token_steps": metrics["executed_token_steps"],
            "completed_transition_steps": metrics["executed_transition_steps"],
            "novel_keys": ledger.key_count,
            "repeated_keys": ledger.repeats,
            "terminal_segment_count": identity.segment_count,
            "terminal_checkpoint_identity": _identity_record(identity),
            "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
        }
    return result, _timing(
        metrics,
        wall_ns=wall_ns,
        ledger=ledger,
        tracked_segments=tracked_segments,
    )


def validate_artifact(payload: dict[str, Any]) -> None:
    v1._require(
        payload.get("artifact_id") == "public-train-exact-identity-replay-v2",
        "v2 artifact identity mismatch",
    )
    v1._require(payload.get("schema_version") == 1, "v2 schema mismatch")
    v1._require(payload.get("criterion") == CRITERION, "v2 criterion drift")
    v1._require(payload.get("claim_boundary") == CLAIM_BOUNDARY, "v2 claim drift")
    v1._require(
        payload.get("train_input", {}).get("held_out_books_materialized") == 0,
        "v2 artifact crossed into held-out material",
    )
    v1._require(
        payload.get("v1_interpretation", {}).get("classification")
        == "STOP_INVALID_PROBE_CRITERION"
        and payload.get("v1_interpretation", {}).get("native_contract_verdict")
        == "PASS_IDENTITY_SURFACE_SPLIT",
        "v1 caveat is absent",
    )
    v1._require(
        payload.get("spend_marker", {}).get("state")
        == "EXACT_IDENTITY_V2_REPLAY_SPENT"
        and payload.get("spend_marker", {}).get("exclusive_create") is True
        and payload.get("spend_marker", {}).get("flushed_and_fsynced_before_run")
        is True,
        "v2 spend marker contract mismatch",
    )
    result = payload.get("result", {})
    v1._require(
        result.get("diagnostic_counter_contract") == DIAGNOSTIC_COUNTER_CONTRACT,
        "v2 diagnostic contract drift",
    )
    status = payload.get("status")
    if status == "BANKED_FIRST_EXACT_IDENTITY_FAILURE":
        v1._require(
            result.get("kind") == "FIRST_EXACT_IDENTITY_FAILURE"
            and bool(result.get("violation", {}).get("kind")),
            "v2 identity failure is incomplete",
        )
    elif status == "PASS_ALL_CANONICAL_TRAIN_EXACT_IDENTITIES":
        v1._require(
            result.get("kind") == "TRAIN_EXACT_IDENTITIES_COMPLETE",
            "v2 TRAIN-complete result mismatch",
        )
    elif status == "BANKED_REPLAY_INFRASTRUCTURE_FAILURE":
        v1._require(bool(result.get("kind")), "v2 infrastructure failure is empty")
    else:
        raise RuntimeError("v2 artifact status mismatch")


def read_banked_artifact(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    validate_artifact(payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    if SPEND_MARKER_PATH.exists() or OUTPUT_PATH.exists():
        parser.error("refusing spent v2 replay; marker or artifact exists")
    return args


def _write_result(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing to overwrite v2 exact-identity artifact") from error


def main() -> int:
    _parse_args()
    preparation_started_ns = time.perf_counter_ns()
    interpretation = json.loads(V1_INTERPRETATION_PATH.read_bytes())
    if (
        interpretation.get("classification") != "STOP_INVALID_PROBE_CRITERION"
        or interpretation.get("native_contract_verdict")
        != "PASS_IDENTITY_SURFACE_SPLIT"
    ):
        raise RuntimeError("v1 interpretation caveat is missing or invalid")

    verified = load_verified_train_input()
    encoded_books = tuple(verified.iter_books())
    if len(encoded_books) != EXPECTED.expected_book_resets:
        raise RuntimeError("canonical TRAIN book count drifted")
    books: list[tuple[object, np.ndarray]] = []
    for ordinal, book in enumerate(encoded_books):
        if int(book.ordinal) != ordinal or int(book.token_count) < 2:
            raise RuntimeError("canonical TRAIN order drifted")
        tokens = np.frombuffer(book.token_bytes_le_i32, dtype="<i4")
        if len(tokens) != int(book.token_count):
            raise RuntimeError("canonical TRAIN token materialization drifted")
        books.append((book, tokens))
    total_tokens = sum(len(tokens) for _book, tokens in books)
    transitions = sum(len(tokens) - 1 for _book, tokens in books)
    if (
        total_tokens != EXPECTED.expected_tokens
        or transitions != EXPECTED.expected_transitions
        or verified.train_stream_sha256 != EXPECTED.train_stream_sha256
    ):
        raise RuntimeError("canonical TRAIN authority/count contract drifted")

    protocol = protocol_authority.build_protocol()
    protocol_authority.validate_protocol(protocol, require_runnable=True)
    production = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    bounded = NativeTemporalCircuitDefinition(
        replace(production.config, segment_capacity=transitions * ACTIVE_COLUMNS),
        production.token_codes,
        production.threshold_offsets_mv,
    )
    codebook_tokens = {token for token, _columns in bounded.token_codes}
    for book, tokens in books:
        if any(int(token) not in codebook_tokens for token in np.unique(tokens)):
            raise RuntimeError(f"{book.short_name} contains an out-of-codebook token")

    ledger = ExactIdentityLedger(
        expected_keys=transitions,
        expected_winner_rows=transitions + len(books),
    )
    native_before = loaded_library_identity()
    preparation_ns = time.perf_counter_ns() - preparation_started_ns
    authority = {
        "train_input_authority_sha256": verified.train_input_authority_sha256,
        "train_stream_sha256": verified.train_stream_sha256,
        "tokenizer_sha256": verified.tokenizer_sha256,
        "protocol_sha256": protocol_authority.protocol_sha256(),
        "production_definition_identity_sha256": production.circuit_identity_sha256(),
        "bounded_definition_identity_sha256": bounded.circuit_identity_sha256(),
        "native_source_digest": native_before.source_digest,
        "native_binary_sha256": native_before.binary_sha256,
        "v1_probe_source_sha256": v1._sha256_file(Path(v1.__file__)),
        "v1_artifact_sha256": v1._sha256_file(v1.OUTPUT_PATH),
        "v1_marker_sha256": v1._sha256_file(v1.SPEND_MARKER_PATH),
        "v1_interpretation_sha256": v1._sha256_file(V1_INTERPRETATION_PATH),
        "expected_train_books": len(books),
        "expected_train_tokens": total_tokens,
        "expected_train_transitions": transitions,
    }
    spend_marker = create_spend_marker(authority=authority)

    construction_ns = 0
    try:
        with v1.lifecycle_probe._synthetic_preoutcome_authority():
            construction_started_ns = time.perf_counter_ns()
            with NativeTemporalCircuit.fresh(bounded) as circuit:
                construction_ns = time.perf_counter_ns() - construction_started_ns
                result, timing = run_stream(
                    circuit,
                    tuple(books),
                    ledger=ledger,
                )
        if result["kind"] == "FIRST_EXACT_IDENTITY_FAILURE":
            status = "BANKED_FIRST_EXACT_IDENTITY_FAILURE"
        elif result["kind"] == "TRAIN_EXACT_IDENTITIES_COMPLETE":
            status = "PASS_ALL_CANONICAL_TRAIN_EXACT_IDENTITIES"
        else:
            status = "BANKED_REPLAY_INFRASTRUCTURE_FAILURE"
    except Exception as error:
        status = "BANKED_REPLAY_INFRASTRUCTURE_FAILURE"
        result = {
            "kind": "UNEXPECTED_REPLAY_EXCEPTION",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "diagnostic_counter_contract": DIAGNOSTIC_COUNTER_CONTRACT,
        }
        timing = _timing(
            _new_metrics(),
            wall_ns=0,
            ledger=ledger,
            tracked_segments=ledger.key_count * ACTIVE_COLUMNS,
        )

    native_after = loaded_library_identity()
    payload = {
        "artifact_id": "public-train-exact-identity-replay-v2",
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "criterion": CRITERION,
        "claim_boundary": CLAIM_BOUNDARY,
        "spend_marker": spend_marker,
        "v1_interpretation": {
            "path": v1._display_path(V1_INTERPRETATION_PATH),
            "sha256": v1._sha256_file(V1_INTERPRETATION_PATH),
            "classification": interpretation["classification"],
            "native_contract_verdict": interpretation["native_contract_verdict"],
            "raw_v1_artifact_sha256": interpretation["raw_evidence"][
                "artifact_sha256"
            ],
        },
        "train_input": {
            "candidate_id": verified.candidate_id,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "train_stream_sha256": verified.train_stream_sha256,
            "tokenizer_sha256": verified.tokenizer_sha256,
            "full_train_book_count": len(books),
            "full_train_token_count": total_tokens,
            "full_train_transition_count": transitions,
            "held_out_books_materialized": 0,
            "books": [
                {
                    "ordinal": int(book.ordinal),
                    "corpus_index": int(book.corpus_index),
                    "short_name": str(book.short_name),
                    "token_count": int(book.token_count),
                    "token_sha256_le_i32": str(book.token_sha256_le_i32),
                }
                for book, _tokens in books
            ],
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "protocol_sha256": protocol_authority.protocol_sha256(),
            "protocol_runnable": protocol["source_binding"]["runnable"],
            "production_definition_identity_sha256": production.circuit_identity_sha256(),
            "bounded_definition_identity_sha256": bounded.circuit_identity_sha256(),
            "production_segment_capacity": production.config.segment_capacity,
            "bounded_segment_capacity": bounded.config.segment_capacity,
        },
        "result": result,
        "timing_load": {
            "pre_run_preparation_ns": preparation_ns,
            "native_construction_ns": construction_ns,
            **timing,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest_before": native_before.source_digest,
            "native_source_digest_after": native_after.source_digest,
            "native_binary_sha256_before": native_before.binary_sha256,
            "native_binary_sha256_after": native_after.binary_sha256,
        },
    }
    validate_artifact(payload)
    _write_result(payload)
    print(f"wrote {OUTPUT_PATH}", flush=True)
    print(f"status={status}", flush=True)
    print(f"artifact_sha256={v1._sha256_file(OUTPUT_PATH)}", flush=True)
    return 0 if status != "BANKED_REPLAY_INFRASTRUCTURE_FAILURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
