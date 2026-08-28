from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "r3_owner_reuse_first_divergence.py"
)
SPEC = importlib.util.spec_from_file_location("r3_owner_reuse_localizer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _step(
    winners: tuple[int, ...],
    *,
    event_count: int,
    candidate: int = 0,
    matching: int = 0,
    active: int = 0,
    predicted_columns: int = 0,
    predicted: bool = False,
    allocated: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        winner_spikes=tuple(SimpleNamespace(cell_id=cell) for cell in winners),
        event_count=event_count,
        candidate_segments=candidate,
        matching_segments=matching,
        active_segments=active,
        predicted_columns=predicted_columns,
        predicted=predicted,
        allocated_segments=allocated,
        punished_segments=0,
        postings_visited=candidate,
    )


def _observe(
    ledger: object,
    step: object,
    *,
    call_index: int,
    token_index: int,
    token_id: int,
    reset: bool,
) -> object:
    return ledger.observe(
        step,
        call_index=call_index,
        book_ordinal=1,
        book_name="Genesis",
        token_index=token_index,
        token_id=token_id,
        reset_before=reset,
    )


def test_foreign_matching_is_diagnostic_and_below_active_repeat_reuses_owner() -> None:
    ledger = MODULE.OwnerReuseLedger()
    source = tuple(range(100, 111))
    owner = tuple(range(200, 211))

    assert _observe(
        ledger,
        _step(source, event_count=1),
        call_index=0,
        token_index=0,
        token_id=7,
        reset=True,
    ) is None
    assert _observe(
        ledger,
        _step(owner, event_count=2, candidate=11, matching=11, allocated=11),
        call_index=1,
        token_index=1,
        token_id=9,
        reset=False,
    ) is None
    assert ledger.first_matching_diagnostic is not None
    assert (
        ledger.first_matching_diagnostic["kind"]
        == "FOREIGN_OR_MISSING_SOURCE_MATCH_DIAGNOSTIC"
    )

    assert _observe(
        ledger,
        _step(source, event_count=1),
        call_index=2,
        token_index=0,
        token_id=7,
        reset=True,
    ) is None
    assert _observe(
        ledger,
        _step(owner, event_count=2, candidate=11, matching=11),
        call_index=3,
        token_index=1,
        token_id=9,
        reset=False,
    ) is None
    assert ledger.repeats == 1
    assert ledger.maximum_candidate_segments == 11
    assert ledger.maximum_active_segments == 0


def test_changed_owner_is_the_exact_repeat_divergence() -> None:
    ledger = MODULE.OwnerReuseLedger()
    source = tuple(range(100, 111))
    owner = tuple(range(200, 211))
    changed_owner = tuple(range(300, 311))

    assert _observe(
        ledger,
        _step(source, event_count=1),
        call_index=0,
        token_index=0,
        token_id=7,
        reset=True,
    ) is None
    assert _observe(
        ledger,
        _step(owner, event_count=2, allocated=11),
        call_index=1,
        token_index=1,
        token_id=9,
        reset=False,
    ) is None
    assert _observe(
        ledger,
        _step(source, event_count=1),
        call_index=2,
        token_index=0,
        token_id=7,
        reset=True,
    ) is None

    divergence = _observe(
        ledger,
        _step(changed_owner, event_count=2, candidate=11, matching=11),
        call_index=3,
        token_index=1,
        token_id=9,
        reset=False,
    )
    assert divergence is not None
    assert divergence["kind"] == "REPEAT_DID_NOT_REUSE_STORED_OWNER"
    assert divergence["call_index"] == 3
    assert divergence["observed"]["owners_equal"] is False
