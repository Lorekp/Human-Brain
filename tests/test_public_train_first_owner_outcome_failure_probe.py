from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pytest

from experiments.r3_pilot import public_train_first_owner_outcome_failure_probe as probe


EXPECTED_ARTIFACT_SHA256_CANONICAL_LF = (
    "7b936ce1913949733f7c46078579dee411d34954f5c1d4ef0fbe246cdcbb6db1"
)
EXPECTED_MARKER_SHA256_CANONICAL_LF = (
    "99e603c99a854c0699930a438df54437649c2ed0a4ef04eddedaeb40fe2c508a"
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
    candidate: int,
    matching: int,
    active: int,
    postings: int,
    allocated: int,
    predicted_columns: int,
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


def test_owner_ledger_ignores_ordered_global_counts_and_checks_owner_outcomes() -> None:
    source = tuple(range(11))
    owners = tuple(range(100, 111))
    divergent = tuple(range(200, 211))
    ledger = probe.OwnerOutcomeLedger(expected_keys=8, expected_winner_rows=12)

    seed = ledger.observe(
        reset_before=True,
        step=_step(
            10,
            1,
            source,
            candidate=0,
            matching=0,
            active=0,
            postings=0,
            allocated=0,
            predicted_columns=0,
        ),
        book_ordinal=0,
        book_token_index=0,
        global_step_index=0,
    )
    novel = ledger.observe(
        reset_before=False,
        step=_step(
            20,
            2,
            owners,
            candidate=99,
            matching=11,
            active=0,
            postings=200,
            allocated=11,
            predicted_columns=0,
        ),
        book_ordinal=0,
        book_token_index=1,
        global_step_index=1,
    )
    ledger.observe(
        reset_before=True,
        step=_step(
            10,
            1,
            source,
            candidate=0,
            matching=0,
            active=0,
            postings=0,
            allocated=0,
            predicted_columns=0,
        ),
        book_ordinal=1,
        book_token_index=0,
        global_step_index=2,
    )
    repeated = ledger.observe(
        reset_before=False,
        step=_step(
            20,
            2,
            owners,
            candidate=123,
            matching=22,
            active=11,
            postings=240,
            allocated=0,
            predicted_columns=11,
        ),
        book_ordinal=1,
        book_token_index=1,
        global_step_index=3,
    )
    ledger.observe(
        reset_before=True,
        step=_step(
            10,
            1,
            source,
            candidate=0,
            matching=0,
            active=0,
            postings=0,
            allocated=0,
            predicted_columns=0,
        ),
        book_ordinal=2,
        book_token_index=0,
        global_step_index=4,
    )
    failure = ledger.observe(
        reset_before=False,
        step=_step(
            20,
            2,
            divergent,
            candidate=321,
            matching=44,
            active=11,
            postings=600,
            allocated=0,
            predicted_columns=11,
        ),
        book_ordinal=2,
        book_token_index=1,
        global_step_index=5,
    )

    assert seed["violation"] is None
    assert novel["violation"] is None and novel["key_was_known"] is False
    assert repeated["violation"] is None and repeated["key_was_known"] is True
    assert failure["violation"]["kind"] == "REPEATED_KEY_OWNER_OUTCOME"
    assert failure["violation"]["expected_owner_cells"] == list(owners)
    assert failure["violation"]["winner_identity_diverged"] is True
    assert ledger.key_count == 1


def test_owner_ledger_rejects_only_invalid_diagnostic_counter_ordering() -> None:
    ledger = probe.OwnerOutcomeLedger(expected_keys=2, expected_winner_rows=4)
    outcome = ledger.observe(
        reset_before=True,
        step=_step(
            10,
            1,
            tuple(range(11)),
            candidate=3,
            matching=4,
            active=0,
            postings=5,
            allocated=0,
            predicted_columns=0,
        ),
        book_ordinal=0,
        book_token_index=0,
        global_step_index=0,
    )
    assert outcome["violation"]["kind"] == "DIAGNOSTIC_COUNTER_CONTRACT"
    assert outcome["violation"]["kind"] not in probe.OWNER_FAILURE_KINDS
    assert ledger.key_count == 0


def test_spend_marker_is_exclusive_and_durable(tmp_path: Path) -> None:
    marker = tmp_path / "probe.spend.json"
    output = tmp_path / "probe.json"
    record = probe.create_spend_marker(
        marker_path=marker,
        output_path=output,
        authority={"train_stream_sha256": "a" * 64},
    )
    assert record["exclusive_create"] is True
    assert record["flushed_and_fsynced_before_run"] is True
    assert record["state"] == "OWNER_OUTCOME_REPLAY_SPENT"
    with pytest.raises(RuntimeError, match="refusing spent replay"):
        probe.create_spend_marker(
            marker_path=marker,
            output_path=output,
            authority={"train_stream_sha256": "a" * 64},
        )


def test_banked_owner_outcome_artifact_is_inspected_without_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ordinary artifact guard reran the TRAIN replay")

    monkeypatch.setattr(probe, "run_stream", forbidden_run)
    marker_raw = probe.SPEND_MARKER_PATH.read_bytes()
    artifact_raw = probe.OUTPUT_PATH.read_bytes()
    payload = probe.read_banked_artifact()

    assert hashlib.sha256(marker_raw.replace(b"\r\n", b"\n")).hexdigest() == (
        EXPECTED_MARKER_SHA256_CANONICAL_LF
    )
    assert hashlib.sha256(artifact_raw.replace(b"\r\n", b"\n")).hexdigest() == (
        EXPECTED_ARTIFACT_SHA256_CANONICAL_LF
    )
    assert payload["status"] == "BANKED_FIRST_OWNER_OUTCOME_FAILURE"
    assert payload["definition"]["arm"] == [16, 18]
    assert payload["train_input"]["held_out_books_materialized"] == 0
    assert payload["spend_marker"]["state"] == "OWNER_OUTCOME_REPLAY_SPENT"
    assert payload["result"]["kind"] == "FIRST_OWNER_OUTCOME_FAILURE"
    assert payload["result"]["violation"]["kind"] in {
        "NOVEL_KEY_OWNER_OUTCOME",
        "REPEATED_KEY_OWNER_OUTCOME",
    }
    assert payload["result"]["book_short_name"] == "Genesis"
    assert payload["result"]["book_zero_based_token_index"] == 15_482
    assert payload["result"]["global_zero_based_step_index"] == 15_482
    assert payload["result"]["token_id"] == 110
    assert payload["result"]["violation"]["expected"] == [11, 0, 0]
    assert payload["result"]["violation"]["observed"] == [11, 11, 1]
    assert payload["result"]["diagnostic_counter_contract"] == (
        "0 <= active <= matching <= candidate <= postings; no exact global-count test"
    )
    lifecycle = payload["result"]["lifecycle"]
    assert lifecycle["counter_checkpoint_temporal_relation"].startswith(
        "native counters are pre-learning-commit"
    )
    assert lifecycle["foreign_segments"]["matching"] == 11
    assert lifecycle["foreign_segments"]["active_post_checkpoint"] == 11
    assert lifecycle["segment_groups"][0]["origin_current_source_overlap_cells"] == 9
    assert lifecycle["segment_groups"][0]["owner_tuple_matches_observed_winners"] is False
    assert lifecycle["failing_step_appended_segments"]["owner_cells"] == (
        payload["result"]["violation"]["observed_winners"]
    )
    assert payload["timing_load"]["executed_token_steps"] >= 2
    assert payload["timing_load"]["replay_wall_ns"] > 0
