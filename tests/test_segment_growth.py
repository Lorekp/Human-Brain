"""Red-first mechanics tests for the R3 segment-growth pilot (path-2 mainline).

Every constant and rule asserted here is frozen in
``docs/research/r3-segment-growth-preregistration.md`` (§2): k=32 candidate
cells per token pool, theta_a=13 connected-active / theta_m=10 potential-active,
connectedPermanence=0.50, initialPermanence=0.21, s=20-contact cap,
maxSegmentsPerCell=255, +/-0.10 confirm increments, delta_neg = 0.04 x
increment = 0.004 negative-evidence decay ON, least-used/best-match winner
election with deterministic tie-breaks.

G-S0 fixture wiring (replay corpus) is a follow-up dispatch; these tests drive
the tracker directly with caller-supplied winner sets W_{t-1}.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from humanbrain.segment_growth import SegmentGrowthTracker


# Two token pools of k = 32 disjoint candidate cells each (global cell ids).
POOL_A = tuple(range(900, 932))
POOL_B = tuple(range(932, 964))
POOLS = {"A": POOL_A, "B": POOL_B}

# A rich upstream context W: more active cells than the 20-contact budget,
# so a fresh segment cannot contact all of them at once.
CONTEXT_1 = list(range(400, 421))
CONTEXT_2 = list(range(500, 525))  # disjoint from CONTEXT_1


def make_tracker(**overrides) -> SegmentGrowthTracker:
    kwargs = {
        "pools": POOLS,
        "growth_on": True,
        "construction_seed": 17,
    }
    kwargs.update(overrides)
    return SegmentGrowthTracker(**kwargs)


def test_frozen_constants_match_prereg():
    """Prereg §2 literals are the tracker defaults; delta_neg = 0.04 x increment."""
    assert SegmentGrowthTracker.CELLS_PER_POOL == 32
    assert SegmentGrowthTracker.THETA_ACTIVE == 13  # theta_a connected-active
    assert SegmentGrowthTracker.THETA_MATCH == 10  # theta_m potential-active
    assert SegmentGrowthTracker.CONNECTED_PERMANENCE == 0.50
    assert SegmentGrowthTracker.INITIAL_PERMANENCE == 0.21
    assert SegmentGrowthTracker.MAX_SYNAPSES_PER_SEGMENT == 20  # s cap
    assert SegmentGrowthTracker.MAX_SEGMENTS_PER_CELL == 255
    assert SegmentGrowthTracker.PERMANENCE_INCREMENT == 0.10
    assert SegmentGrowthTracker.DELTA_NEGATIVE == 0.004
    # delta_neg is defined as 0.04 x increment.
    assert SegmentGrowthTracker.DELTA_NEGATIVE == pytest.approx(
        0.04 * SegmentGrowthTracker.PERMANENCE_INCREMENT
    )


def test_growth_off_parity():
    """A2 control identity: election machinery runs for diagnostics, but no
    segment is ever created, no permanence ever changes, and the predicted
    path is structurally zero on every step."""
    off = make_tracker(growth_on=False)
    on = make_tracker(growth_on=True)

    script = [
        ("A", CONTEXT_1),
        ("B", CONTEXT_1),
        ("A", CONTEXT_1),
        ("A", CONTEXT_2),
        ("B", CONTEXT_2),
        ("A", CONTEXT_1),
    ]
    off_winners = []
    for pool_id, winners in script:
        result = off.step(pool_id, winners)
        off_winners.append(result.elected_winner)
        # Structurally zero predicted path in the growth-off arm.
        assert result.burst is True
        assert result.fired_cells == ()
        assert result.allocated_segment is False

    # No segments, no contacts, no permanence anywhere.
    assert off.total_segments() == 0
    for cell in POOL_A + POOL_B:
        assert off.segment_count(cell) == 0
    assert off.total_contacts() == 0

    # Election machinery still ran: a deterministic least-used winner per step.
    assert all(w is not None for w in off_winners)
    # Before the growth-on arm creates any matching segment both arms elect
    # identically (parity of the shared election machinery).
    first_on = on.step("A", CONTEXT_1)
    assert first_on.elected_winner == off_winners[0]
    assert on.total_segments() >= 1


def test_election_least_used_then_lowest_index():
    """No-candidate bursts elect the least-used cell (fewest owned segments,
    then fewest prior wins, then lowest cell index); with candidates present
    the best-matching cell wins and repeats never allocate duplicates."""
    tracker = make_tracker()

    # Four pairwise-disjoint contexts -> four no-candidate bursts. Every cell
    # starts at 0 owned segments / 0 wins, so ties fall to the lowest index:
    # strict round-robin over POOL_A.
    winners = []
    contexts = [
        list(range(600, 625)),
        list(range(700, 725)),
        list(range(800, 825)),
        list(range(850, 875)),
    ]
    for context in contexts:
        result = tracker.step("A", context)
        assert result.burst and result.allocated_segment
        winners.append(result.elected_winner)
    assert winners == [POOL_A[0], POOL_A[1], POOL_A[2], POOL_A[3]]
    assert [tracker.prior_wins(c) for c in winners] == [1, 1, 1, 1]

    # Re-presenting context 1: cell POOL_A[0]'s segment matches (>= theta_m)
    # and it is the only candidate, so it is re-elected and NO duplicate
    # segment is allocated (one-pattern-per-new-context rule).
    again = tracker.step("A", contexts[0])
    assert again.elected_winner == POOL_A[0]
    assert again.burst is True  # not yet connected-active enough to fire
    assert again.allocated_segment is False
    assert tracker.segment_count(POOL_A[0]) == 1
    assert tracker.prior_wins(POOL_A[0]) == 2

    # Tie-break comparator: highest best-count, then fewest prior wins, then
    # lowest cell index.
    key = SegmentGrowthTracker.election_sort_key
    assert key(best_count=20, wins=3, index=7) < key(best_count=19, wins=0, index=99)
    assert key(best_count=20, wins=1, index=7) < key(best_count=20, wins=2, index=3)
    assert key(best_count=20, wins=1, index=7) < key(best_count=20, wins=1, index=9)


def test_bit_identical_reruns():
    """Determinism tripwire (prereg §1): same (seed, script, constants) =>
    bit-identical segment arrays."""
    script = [
        ("A", CONTEXT_1),
        ("A", CONTEXT_1),
        ("B", CONTEXT_1),
        ("A", CONTEXT_1),
        ("A", CONTEXT_1),
    ]
    runs = []
    for _ in range(2):
        tracker = make_tracker()
        trace = []
        for pool_id, context in script:
            result = tracker.step(pool_id, context)
            trace.append(
                (
                    result.elected_winner,
                    result.fired_cells,
                    result.punished_segments,
                    result.removed_contacts,
                    tracker.total_segments(),
                )
            )
        runs.append((trace, tracker.all_permanences().tobytes()))
    assert runs[0][0] == runs[1][0]
    assert runs[0][1] == runs[1][1]


def test_contact_cap_enforcement():
    """s = 20 contact budget: growth fills to exactly 20 and never beyond;
    contacts reaching p <= 0 are removed and free budget; the
    maxSegmentsPerCell ceiling raises STOP when exceeded."""
    tracker = make_tracker()
    cell = POOL_A[0]

    # First burst: |CONTEXT_1| > s, so the fresh segment saturates at 20.
    tracker.step("A", CONTEXT_1)
    presyn, perm = tracker.segment_contacts(cell, 0)
    assert len(presyn) == SegmentGrowthTracker.MAX_SYNAPSES_PER_SEGMENT == 20
    assert np.all(perm == SegmentGrowthTracker.INITIAL_PERMANENCE)

    # Repeated reinforcement against the same context cannot exceed the cap.
    for _ in range(4):
        tracker.step("A", CONTEXT_1)
        presyn, perm = tracker.segment_contacts(cell, 0)
        assert len(presyn) <= 20

    # Sustained exposure to an overlapping context (shares 16 of CONTEXT_1's
    # 21 cells, so the segment stays >= theta_m matching throughout) drives
    # the non-shared contacts inactive (-0.10 per exposure) until they hit
    # p <= 0 and are removed, freeing budget that refill must respect.
    overlap = list(range(405, 426))
    removed_events = 0
    for _ in range(12):
        result = tracker.step("A", overlap)
        removed_events += result.removed_contacts
        presyn, _perm = tracker.segment_contacts(cell, 0)
        assert len(presyn) <= 20
    assert removed_events >= 1, "expected p <= 0 contacts to be removed"

    # Hard ceiling on segments per cell: STOP discipline (prereg §1/§2).
    # Least-used election round-robins, so saturating the pool takes
    # 2 segments x 32 cells = 64 pairwise-disjoint contexts; every cell ends
    # owning exactly 2 segments and 2 wins, and the next no-candidate burst
    # must allocate onto an already-full cell => STOP.
    capped = make_tracker(max_segments_per_cell=2)
    for i in range(64):
        capped.step("A", list(range(5000 + i * 25, 5000 + i * 25 + 21)))
    assert {capped.segment_count(c) for c in POOL_A} == {2}
    with pytest.raises(RuntimeError, match="STOP"):
        capped.step("A", list(range(9000, 9021)))


def test_negative_evidence_decay():
    """delta_neg = 0.004 x increment punishes unconfirmed depolarizations:
    a converged segment whose cell is NOT in the actual next input loses
    exactly 0.004 from its then-active-connected permanences per miss."""
    tracker = make_tracker()
    cell = POOL_A[0]

    # Converge the segment by repeated exposure to the same context:
    # permanences walk 0.21 -> 0.31 -> 0.41 -> 0.51 (>= connectedPermanence).
    exposures = []
    for i in range(4):
        result = tracker.step("A", CONTEXT_1)
        exposures.append(result)
        if i < 3:
            assert result.burst and result.fired_cells == ()
    _, perm_before = tracker.segment_contacts(cell, 0)
    assert np.allclose(perm_before, 0.51)

    # Now the segment is recall-competent: 20 connected-active >= theta_a = 13
    # so the next presentation takes the predicted path (no burst).
    predicted = tracker.step("A", CONTEXT_1)
    assert predicted.burst is False
    assert predicted.fired_cells == (cell,)
    _, perm = tracker.segment_contacts(cell, 0)
    assert np.allclose(perm, 0.61)
    snapshot = perm.copy()

    # Feed the OTHER pool as the actual next input: the depolarized cell is
    # not in it => one negative-evidence step of exactly delta_neg.
    miss = tracker.step("B", CONTEXT_1)
    _, perm_after = tracker.segment_contacts(cell, 0)
    assert miss.punished_segments == ((cell, 0),)
    expected = snapshot - SegmentGrowthTracker.DELTA_NEGATIVE
    assert np.allclose(perm_after, expected, atol=1e-12)
    assert math.isclose(float(perm_after[0]), float(snapshot[0]) - 0.004, abs_tol=1e-12)

    # Sustained misses keep punishing while the wrong predictor stays
    # competent (prereg G-S0 (iii)): every B-step subtracts another 0.004
    # until the segment drops below theta_a competence.
    for _ in range(30):
        tracker.step("B", CONTEXT_1)
    _, perm_late = tracker.segment_contacts(cell, 0)
    assert float(perm_late.min()) < SegmentGrowthTracker.CONNECTED_PERMANENCE
    late = tracker.step("B", CONTEXT_1)
    assert cell not in late.fired_cells  # no longer recall-competent

    # Control: pool B's own segment (grown and confirmed on B-steps) keeps
    # its exact permanences — negative evidence only touches unconfirmed
    # depolarizations, never bystanders.
    b_cell = POOL_B[0]
    _, b_perm_before = tracker.segment_contacts(b_cell, 0)
    assert b_perm_before.size > 0
    tracker.step("A", CONTEXT_2)  # unrelated context; no new punishment
    _, b_perm_after = tracker.segment_contacts(b_cell, 0)
    assert np.array_equal(b_perm_after, b_perm_before)
