from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from experiments.r3_pilot import public_train_exact_identity_replay_v2 as probe


EXPECTED_ARTIFACT_SHA256_CANONICAL_LF = (
    "e5a619b747e7626f21c06a5d5c211fd9c128cf0a70d37692ee407d358a4fb2ec"
)
EXPECTED_MARKER_SHA256_CANONICAL_LF = (
    "0c7e25462130b413879742a3c918f75ddcece5a4b28cd871ffcaf08ad3df4947"
)


@dataclass(frozen=True)
class _Spike:
    cell_id: int


@dataclass(frozen=True)
class _Step:
    token_id: int
    event_count: int
    candidate_segments: int
    matching_segments: int
    active_segments: int
    postings_visited: int
    allocated_segments: int
    predicted_columns: int
    predicted: bool
    punished_segments: int
    winner_spikes: tuple[_Spike, ...]


def _step(
    token_id: int,
    event_count: int,
    winners: tuple[int, ...],
    *,
    allocated: int,
    predicted_columns: int,
    candidate: int = 20,
    matching: int = 11,
    active: int = 11,
    postings: int = 30,
    punished: int = 0,
) -> _Step:
    return _Step(
        token_id=token_id,
        event_count=event_count,
        candidate_segments=candidate,
        matching_segments=matching,
        active_segments=active,
        postings_visited=postings,
        allocated_segments=allocated,
        predicted_columns=predicted_columns,
        predicted=predicted_columns == 11,
        punished_segments=punished,
        winner_spikes=tuple(_Spike(cell) for cell in winners),
    )


def _observe(
    ledger: probe.ExactIdentityLedger,
    step: _Step,
    *,
    reset: bool,
    book: int,
    token_index: int,
    global_step: int,
) -> dict[str, object]:
    return ledger.observe(
        reset_before=reset,
        step=step,
        book_ordinal=book,
        book_token_index=token_index,
        global_step_index=global_step,
    )


def test_same_token_surface_prediction_can_allocate_then_exact_identity_reuses() -> None:
    source = tuple(range(11))
    owners = tuple(range(100, 111))
    ledger = probe.ExactIdentityLedger(expected_keys=8, expected_winner_rows=12)

    assert _observe(
        ledger,
        _step(10, 1, source, allocated=0, predicted_columns=0, candidate=0, matching=0, active=0, postings=0),
        reset=True,
        book=0,
        token_index=0,
        global_step=0,
    )["violation"] is None
    novel = _observe(
        ledger,
        _step(20, 2, owners, allocated=11, predicted_columns=11),
        reset=False,
        book=0,
        token_index=1,
        global_step=1,
    )
    _observe(
        ledger,
        _step(10, 1, source, allocated=0, predicted_columns=0, candidate=0, matching=0, active=0, postings=0),
        reset=True,
        book=1,
        token_index=0,
        global_step=2,
    )
    repeated = _observe(
        ledger,
        _step(20, 2, owners, allocated=0, predicted_columns=0, active=0),
        reset=False,
        book=1,
        token_index=1,
        global_step=3,
    )

    assert novel["violation"] is None and novel["key_was_known"] is False
    assert repeated["violation"] is None and repeated["key_was_known"] is True
    assert ledger.key_count == 1
    assert ledger.repeats == 1


def test_novel_tuple_reuse_and_repeated_owner_drift_stop_exact_identity() -> None:
    source_a = tuple(range(11))
    source_b = tuple(range(20, 31))
    owners = tuple(range(100, 111))
    drifted = tuple(range(200, 211))

    reused = probe.ExactIdentityLedger(expected_keys=8, expected_winner_rows=12)
    _observe(reused, _step(10, 1, source_a, allocated=0, predicted_columns=0, candidate=0, matching=0, active=0, postings=0), reset=True, book=0, token_index=0, global_step=0)
    _observe(reused, _step(20, 2, owners, allocated=11, predicted_columns=0), reset=False, book=0, token_index=1, global_step=1)
    _observe(reused, _step(11, 1, source_b, allocated=0, predicted_columns=0, candidate=0, matching=0, active=0, postings=0), reset=True, book=1, token_index=0, global_step=2)
    collision = _observe(reused, _step(20, 2, owners, allocated=11, predicted_columns=11), reset=False, book=1, token_index=1, global_step=3)
    assert collision["violation"]["kind"] == "NOVEL_KEY_REUSED_OWNER_TUPLE"

    drift = probe.ExactIdentityLedger(expected_keys=8, expected_winner_rows=12)
    _observe(drift, _step(10, 1, source_a, allocated=0, predicted_columns=0, candidate=0, matching=0, active=0, postings=0), reset=True, book=0, token_index=0, global_step=0)
    _observe(drift, _step(20, 2, owners, allocated=11, predicted_columns=11), reset=False, book=0, token_index=1, global_step=1)
    _observe(drift, _step(10, 1, source_a, allocated=0, predicted_columns=0, candidate=0, matching=0, active=0, postings=0), reset=True, book=1, token_index=0, global_step=2)
    divergence = _observe(drift, _step(20, 2, drifted, allocated=0, predicted_columns=11), reset=False, book=1, token_index=1, global_step=3)
    assert divergence["violation"]["kind"] == "REPEATED_KEY_IDENTITY_DRIFT"


def test_v1_caveat_authenticates_immutable_spent_evidence() -> None:
    interpretation = json.loads(probe.V1_INTERPRETATION_PATH.read_bytes())
    assert interpretation["classification"] == "STOP_INVALID_PROBE_CRITERION"
    assert interpretation["native_contract_verdict"] == "PASS_IDENTITY_SURFACE_SPLIT"
    assert hashlib.sha256(Path(probe.v1.__file__).read_bytes()).hexdigest() == (
        interpretation["raw_evidence"]["probe_source_sha256"]
    )
    assert hashlib.sha256(probe.v1.SPEND_MARKER_PATH.read_bytes()).hexdigest() == (
        interpretation["raw_evidence"]["marker_sha256"]
    )
    assert hashlib.sha256(probe.v1.OUTPUT_PATH.read_bytes()).hexdigest() == (
        interpretation["raw_evidence"]["artifact_sha256"]
    )


def test_v2_spend_marker_is_exclusive(tmp_path: Path) -> None:
    marker = tmp_path / "v2.spend.json"
    output = tmp_path / "v2.json"
    record = probe.create_spend_marker(
        marker_path=marker,
        output_path=output,
        authority={"train_stream_sha256": "a" * 64},
    )
    assert record["state"] == "EXACT_IDENTITY_V2_REPLAY_SPENT"
    assert record["exclusive_create"] is True
    assert record["flushed_and_fsynced_before_run"] is True
    with pytest.raises(RuntimeError, match="refusing spent v2 replay"):
        probe.create_spend_marker(
            marker_path=marker,
            output_path=output,
            authority={"train_stream_sha256": "a" * 64},
        )


def test_banked_v2_artifact_is_inspected_without_replay(monkeypatch) -> None:
    monkeypatch.setattr(
        probe,
        "run_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ordinary guard reran the spent TRAIN replay")
        ),
    )
    marker_raw = probe.SPEND_MARKER_PATH.read_bytes()
    artifact_raw = probe.OUTPUT_PATH.read_bytes()
    payload = probe.read_banked_artifact()

    assert hashlib.sha256(marker_raw.replace(b"\r\n", b"\n")).hexdigest() == EXPECTED_MARKER_SHA256_CANONICAL_LF
    assert hashlib.sha256(artifact_raw.replace(b"\r\n", b"\n")).hexdigest() == EXPECTED_ARTIFACT_SHA256_CANONICAL_LF
    assert payload["criterion"]["novel"] == "allocated=11 and globally-new ordered winner tuple"
    assert payload["criterion"]["repeated"] == "allocated=0 and winners equal stored O(K)"
    assert payload["train_input"]["held_out_books_materialized"] == 0
    assert payload["status"] == "BANKED_REPLAY_INFRASTRUCTURE_FAILURE"
    assert payload["result"] == {
        "diagnostic_counter_contract": probe.DIAGNOSTIC_COUNTER_CONTRACT,
        "error_message": "'int' object is not iterable",
        "error_type": "TypeError",
        "kind": "UNEXPECTED_REPLAY_EXCEPTION",
    }
    interpretation_path = probe.OUTPUT_PATH.with_name(
        "public_train_exact_identity_replay_v2.interpretation.json"
    )
    interpretation = json.loads(interpretation_path.read_bytes())
    assert interpretation["classification"] == (
        "STOP_INVALID_MONOLITHIC_ALLOCATION_CRITERION"
    )
    assert interpretation["verified_prefix"]["identity_verdict"] == (
        "PASS_STABLE_COMPLETE_TUPLE_PREFIX"
    )
    assert interpretation["verified_prefix"]["accepted_novel_keys"] == 975_952
    assert interpretation["verified_prefix"]["accepted_repeated_keys"] == 69
    assert interpretation["verified_prefix"][
        "historical_v2_expected_segments_from_accepted_novel_keys"
    ] == 10_735_472
    assert "expected_segments_from_accepted_novel_keys" not in interpretation[
        "verified_prefix"
    ]
    assert interpretation["first_failure_boundary"] == {
        "book": "2 Chronicles",
        "book_ordinal": 13,
        "criterion_branch": "NOVEL_KEY_DID_NOT_ALLOCATE_IDENTITY",
        "expected_allocated_segments": 11,
        "global_zero_based_step_index": 976_035,
        "observed_allocated_segments": "not preserved; known only to differ from 11",
        "token_id": 1073,
        "zero_based_token_index": 2,
    }
    assert interpretation["criterion_correction"] == {
        "invalid_assumption": (
            "every novel full (source winner tuple, token) key must allocate 11 new segments"
        ),
        "native_contract": (
            "persistent identity is factorized by output column; an exact-source "
            "segment may reserve its owner in a column shared by a previously learned "
            "successor token, while only the other columns allocate"
        ),
        "replacement_rule": (
            "a novel full key may allocate 0..11 segments but must produce a previously "
            "unseen complete ordered winner tuple; an exact repeated key must allocate 0 "
            "and reproduce its stored complete tuple"
        ),
        "terminal_segment_rule": (
            "terminal segment count equals the sum of native allocated_segments, not 11 "
            "times the number of novel full keys"
        ),
    }
    assert interpretation["boundary_localization"]["production_code_overlap_with_1073"] == {
        "32": [],
        "1260": [4078],
        "1366": [],
    }
    assert interpretation["boundary_localization"]["most_likely_allocated_segments"] == 10
    assert interpretation["target_independent_native_fixture"]["test"] == (
        "novel_token_reuses_exact_source_owner_only_in_shared_column"
    )
    assert interpretation["target_independent_native_fixture"]["status"] == (
        "PASS_FACTORIZED_IDENTITY_ONE_COLUMN_OVERLAP"
    )
    assert hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest() == (
        interpretation["raw_evidence"]["probe_source_sha256"]
    )
    assert hashlib.sha256(probe.SPEND_MARKER_PATH.read_bytes()).hexdigest() == (
        interpretation["raw_evidence"]["marker_sha256"]
    )
    assert hashlib.sha256(probe.OUTPUT_PATH.read_bytes()).hexdigest() == (
        interpretation["raw_evidence"]["artifact_sha256"]
    )
