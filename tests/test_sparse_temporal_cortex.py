"""R3-F0 fixtures for the repaired physical temporal-memory seam.

These tests intentionally exercise the public deep-module boundary.  They do
not call a Python threshold tracker or inject elected winner ids directly.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import pytest

from humanbrain.sparse_temporal_cortex import (
    GrowthPolicy,
    SparseTemporalConfig,
    SparseTemporalCortex,
)


K = 11


def _disjoint_codebook(token_count: int) -> dict[int, tuple[int, ...]]:
    return {
        token: tuple(range(token * K, (token + 1) * K))
        for token in range(token_count)
    }


def _config(
    token_count: int,
    *,
    base_cells_per_column: int = 4,
    initial_permanence: float = 0.51,
    growth_policy: GrowthPolicy = GrowthPolicy.BOUND_WINNERS,
    construction_seed: int = 17,
) -> SparseTemporalConfig:
    columns = token_count * K
    return SparseTemporalConfig(
        column_count=columns,
        active_columns=K,
        base_cells_per_column=base_cells_per_column,
        max_pyramidal_cells=columns * base_cells_per_column,
        contacts_per_segment=20,
        theta_match=16,
        theta_active=18,
        initial_permanence=initial_permanence,
        growth_policy=growth_policy,
        construction_seed=construction_seed,
    )


def _learn_pair(circuit: SparseTemporalCortex, left: int, right: int):
    circuit.reset_transient()
    first = circuit.learn(left)
    second = circuit.learn(right)
    return first, second


def _recognize_pair(circuit: SparseTemporalCortex, left: int, right: int):
    circuit.reset_transient()
    first = circuit.recognize(left)
    second = circuit.recognize(right)
    return first, second


def test_one_exposure_replays_through_glif3_basket_and_relay_spikes() -> None:
    codebook = _disjoint_codebook(3)
    circuit = SparseTemporalCortex(codebook, config=_config(3))

    _a, learned_b = _learn_pair(circuit, 0, 1)
    learned_state = circuit.learned_snapshot()
    _a_replay, replayed_b = _recognize_pair(circuit, 0, 1)

    assert learned_b.predicted is False
    assert learned_b.allocated_segments == K
    assert replayed_b.predicted is True
    assert replayed_b.predicted_columns == K
    assert replayed_b.winner_cells == learned_b.winner_cells
    assert len(replayed_b.winner_cells) == K
    assert replayed_b.pyramidal_spikes_by_column == tuple(
        (winner,) for winner in replayed_b.winner_cells
    )
    assert replayed_b.pyramidal_spike_times_by_column_ms == tuple(
        (winner_time,) for winner_time in replayed_b.winner_spike_times_ms
    )
    assert len(replayed_b.basket_spikes) == K
    assert len(replayed_b.inhibition_onset_times_ms) == K
    assert len(replayed_b.relay_spikes) == K
    assert len(replayed_b.context_spikes) == 2 * K
    assert circuit.context_size == 2 * K
    assert circuit.physical_cell_count == (
        2 * circuit.config.max_pyramidal_cells + 3 * circuit.config.column_count
    )
    assert len(replayed_b.proximal_spikes) == K
    assert len(replayed_b.proximal_relay_spikes) == K
    assert replayed_b.context_spikes == replayed_b.winner_context_spikes
    physical_populations = (
        set(replayed_b.winner_cells),
        set(replayed_b.relay_spikes),
        set(replayed_b.basket_spikes),
        set(replayed_b.proximal_spikes),
        set(replayed_b.proximal_relay_spikes),
    )
    assert all(
        left.isdisjoint(right)
        for left, right in combinations(physical_populations, 2)
    )
    assert set(replayed_b.winner_cells).isdisjoint(replayed_b.relay_spikes)
    for _owner, contacts, _permanences in circuit.learned_snapshot().segments:
        pyramidal = {cell for cell in contacts if cell < circuit.config.max_pyramidal_cells}
        relays = {cell for cell in contacts if cell >= circuit.config.max_pyramidal_cells}
        assert len(pyramidal) == len(relays) == K - 1
        assert relays == {
            circuit.config.max_pyramidal_cells + cell for cell in pyramidal
        }
    assert all(time_ms > 0.0 for time_ms in replayed_b.winner_spike_times_ms)
    assert all(time_ms > 0.0 for time_ms in replayed_b.basket_spike_times_ms)
    assert all(time_ms > 0.0 for time_ms in replayed_b.relay_spike_times_ms)
    assert all(
        replay < novel
        for replay, novel in zip(
            replayed_b.winner_spike_times_ms,
            learned_b.winner_spike_times_ms,
            strict=True,
        )
    )
    assert all(
        basket > winner
        for basket, winner in zip(
            replayed_b.basket_spike_times_ms,
            replayed_b.winner_spike_times_ms,
            strict=True,
        )
    )
    assert all(
        onset > basket > winner
        for onset, basket, winner in zip(
            replayed_b.inhibition_onset_times_ms,
            replayed_b.basket_spike_times_ms,
            replayed_b.winner_spike_times_ms,
            strict=True,
        )
    )
    assert all(
        relay > source
        for relay, source in zip(
            replayed_b.proximal_relay_spike_times_ms,
            replayed_b.proximal_spike_times_ms,
            strict=True,
        )
    )
    assert circuit.learned_snapshot() == learned_state


def test_same_token_in_two_contexts_gets_distinct_replayable_cell_codes() -> None:
    codebook = _disjoint_codebook(3)
    circuit = SparseTemporalCortex(codebook, config=_config(3))

    _a, b_after_a = _learn_pair(circuit, 0, 1)
    _c, b_after_c = _learn_pair(circuit, 2, 1)

    assert b_after_a.winner_cells != b_after_c.winner_cells
    assert b_after_a.context_spikes != b_after_c.context_spikes
    assert b_after_a.proximal_spikes == b_after_c.proximal_spikes
    assert b_after_a.proximal_relay_spikes == b_after_c.proximal_relay_spikes
    assert _recognize_pair(circuit, 0, 1)[1].winner_cells == b_after_a.winner_cells
    assert _recognize_pair(circuit, 2, 1)[1].winner_cells == b_after_c.winner_cells


@pytest.mark.parametrize("construction_seed", [3, 7, 17, 29, 43])
def test_sixteen_novel_contexts_do_not_cycle_at_base_depth_eight(
    construction_seed: int,
) -> None:
    context_count = 16
    target = context_count
    codebook = _disjoint_codebook(context_count + 1)
    circuit = SparseTemporalCortex(
        codebook,
        config=_config(
            context_count + 1,
            base_cells_per_column=8,
            construction_seed=construction_seed,
        ),
    )

    learned_codes: dict[int, tuple[int, ...]] = {}
    learned_contexts: dict[int, tuple[int, ...]] = {}
    for context_token in range(context_count):
        _context, target_step = _learn_pair(circuit, context_token, target)
        assert target_step.predicted is False
        assert all(
            len(spikes) == 1
            for spikes in target_step.pyramidal_spikes_by_column
        )
        learned_codes[context_token] = target_step.winner_cells
        learned_contexts[context_token] = target_step.context_spikes

    assert len(set(learned_codes.values())) == context_count
    assert len(set(learned_contexts.values())) == context_count
    for left, right in combinations(learned_contexts.values(), 2):
        assert len(set(left).intersection(right)) < circuit.config.theta_match

    frozen = circuit.learned_snapshot()
    for context_token, expected in learned_codes.items():
        replay = _recognize_pair(circuit, context_token, target)[1]
        assert replay.predicted is True
        assert all(len(spikes) == 1 for spikes in replay.pyramidal_spikes_by_column)
        assert replay.winner_cells == expected
    assert circuit.learned_snapshot() == frozen


def test_potential_only_contacts_reconnect_before_becoming_predictive() -> None:
    codebook = _disjoint_codebook(2)
    circuit = SparseTemporalCortex(
        codebook,
        config=_config(2, initial_permanence=0.41),
    )

    _learn_pair(circuit, 0, 1)
    assert _recognize_pair(circuit, 0, 1)[1].predicted is False

    _learn_pair(circuit, 0, 1)
    assert _recognize_pair(circuit, 0, 1)[1].predicted is True


def test_reinforcement_regrows_freed_capacity_as_complete_relay_pairs() -> None:
    codebook = {
        0: (0, 1),
        1: (0, 2),
        2: (3, 4),
    }
    config = SparseTemporalConfig(
        column_count=5,
        active_columns=2,
        base_cells_per_column=2,
        max_pyramidal_cells=10,
        contacts_per_segment=4,
        theta_match=2,
        theta_active=4,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    context_one, _target = _learn_pair(circuit, 0, 2)
    assert circuit.segment_count == 2

    for _ in range(6):
        context_two, _target = _learn_pair(circuit, 1, 2)

    context_two_pairs = set(context_two.context_spikes)
    for _owner, contacts, _permanences in circuit.learned_snapshot().segments:
        assert len(contacts) == 4
        assert set(contacts) == context_two_pairs
        pyramidal = {
            cell for cell in contacts if cell < circuit.config.max_pyramidal_cells
        }
        assert {
            circuit.config.max_pyramidal_cells + cell for cell in pyramidal
        } == set(contacts) - pyramidal
    assert set(context_one.context_spikes) != context_two_pairs


def test_grouped_contact_counts_cross_each_threshold_at_the_boundary() -> None:
    codebook = _disjoint_codebook(2)
    circuit = SparseTemporalCortex(codebook, config=_config(2))
    _learn_pair(circuit, 0, 1)
    contacts = circuit.learned_snapshot().segments[0][1]
    pyramidal = [
        cell for cell in contacts if cell < circuit.config.max_pyramidal_cells
    ]

    def paired_context(pair_count: int) -> tuple[int, ...]:
        selected = pyramidal[:pair_count]
        return tuple(
            selected
            + [circuit.config.max_pyramidal_cells + cell for cell in selected]
        )

    below_match = circuit.diagnose_context(paired_context(7))
    at_match = circuit.diagnose_context(paired_context(8))
    below_active = circuit.diagnose_context(paired_context(8))
    at_active = circuit.diagnose_context(paired_context(9))

    assert below_match.max_potential_contacts == 14
    assert below_match.matching_segments == 0
    assert at_match.max_potential_contacts == 16
    assert at_match.matching_segments >= 1
    assert below_active.active_segments == 0
    assert at_active.max_connected_contacts == 18
    assert at_active.active_segments >= 1


def test_three_wrong_predictions_disconnect_the_confirmed_segments() -> None:
    codebook = _disjoint_codebook(3)
    circuit = SparseTemporalCortex(codebook, config=_config(3))
    _learn_pair(circuit, 0, 1)

    assert _recognize_pair(circuit, 0, 1)[1].predicted is True
    for _ in range(3):
        wrong = _learn_pair(circuit, 0, 2)[1]
        assert wrong.punished_segments == K

    _a, b = _recognize_pair(circuit, 0, 1)
    assert b.predicted is False
    assert sum(value == pytest.approx(0.498) for value in circuit.permanence_values()) == K * 20


def test_book_boundary_proximal_code_is_stable_after_segment_load_changes() -> None:
    codebook = _disjoint_codebook(3)
    circuit = SparseTemporalCortex(codebook, config=_config(3))

    circuit.reset_transient()
    before = circuit.recognize(0).winner_cells
    _learn_pair(circuit, 2, 0)
    circuit.reset_transient()
    after = circuit.recognize(0).winner_cells

    assert after == before


def test_growth_off_remains_structurally_zero() -> None:
    codebook = _disjoint_codebook(2)
    circuit = SparseTemporalCortex(
        codebook,
        config=_config(2, growth_policy=GrowthPolicy.GROWTH_OFF),
    )

    frozen = circuit.learned_snapshot()
    for _ in range(4):
        _learn_pair(circuit, 0, 1)
    assert circuit.segment_count == 0
    assert circuit.learned_snapshot() == frozen
    assert _recognize_pair(circuit, 0, 1)[1].predicted is False


def test_generic_afferent_control_uses_mass_matched_physical_spikes() -> None:
    codebook = _disjoint_codebook(2)
    bound = SparseTemporalCortex(
        codebook,
        config=_config(2, growth_policy=GrowthPolicy.BOUND_WINNERS),
    )
    generic = SparseTemporalCortex(
        codebook,
        config=_config(2, growth_policy=GrowthPolicy.GENERIC_AFFERENTS),
    )

    bound_source, bound_target = _learn_pair(bound, 0, 1)
    generic_source, generic_target = _learn_pair(generic, 0, 1)

    assert bound_source.winner_cells == generic_source.winner_cells
    assert bound_target.winner_cells != generic_target.winner_cells
    assert bound_source.context_spikes == bound_source.winner_context_spikes
    assert generic_source.context_spikes == generic_source.proximal_context_spikes
    assert len(bound_source.context_spikes) == len(generic_source.context_spikes) == 2 * K
    assert set(bound_source.context_spikes).isdisjoint(
        generic_source.context_spikes
    )
    assert len(bound.learned_snapshot().segments) == K
    assert len(generic.learned_snapshot().segments) == K
    generic_replay = _recognize_pair(generic, 0, 1)[1]
    assert generic_replay.predicted is True
    assert generic_replay.winner_cells == generic_target.winner_cells
    for bound_segment, generic_segment in zip(
        bound.learned_snapshot().segments,
        generic.learned_snapshot().segments,
        strict=True,
    ):
        segment = bound_segment
        assert len(segment[1]) == 20
        assert set(segment[1]).issubset(bound_source.winner_context_spikes)
        assert len(generic_segment[1]) == 20
        assert set(generic_segment[1]).issubset(
            generic_source.proximal_context_spikes
        )
        bound_positions = {
            bound_source.winner_cells.index(cell)
            for cell in segment[1]
            if cell in bound_source.winner_cells
        }
        generic_positions = {
            generic_source.proximal_spikes.index(cell)
            for cell in generic_segment[1]
            if cell in generic_source.proximal_spikes
        }
        assert bound_positions == generic_positions


def test_replay_is_bit_exact_for_results_and_deep_state() -> None:
    codebook = _disjoint_codebook(4)
    config = _config(4)
    left = SparseTemporalCortex(codebook, config=config)
    right = SparseTemporalCortex(codebook, config=config)
    script = ((0, 1), (2, 1), (3, 1), (0, 1))

    left_results = [_learn_pair(left, a, b) for a, b in script]
    right_results = [_learn_pair(right, a, b) for a, b in script]

    assert left_results == right_results
    assert left.snapshot() == right.snapshot()


def test_checkpoint_resume_matches_uninterrupted_next_spikes_and_state() -> None:
    codebook = _disjoint_codebook(4)
    config = _config(4)
    uninterrupted = SparseTemporalCortex(codebook, config=config)
    _learn_pair(uninterrupted, 0, 1)
    _learn_pair(uninterrupted, 2, 1)

    restored = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=uninterrupted.snapshot(),
    )
    expected = uninterrupted.learn(3)
    actual = restored.learn(3)

    assert actual == expected
    assert restored.snapshot() == uninterrupted.snapshot()


def test_reset_removes_event_order_from_the_next_novel_race() -> None:
    codebook = _disjoint_codebook(10)
    config = _config(10, base_cells_per_column=8)
    clean = SparseTemporalCortex(codebook, config=config)
    advanced = SparseTemporalCortex(codebook, config=config)

    for context_token in range(5):
        _recognize_pair(advanced, context_token, 8)

    clean.reset_transient()
    advanced.reset_transient()
    clean_result = (clean.recognize(6), clean.recognize(9))
    advanced_result = (advanced.recognize(6), advanced.recognize(9))

    assert advanced_result == clean_result


def test_checkpoint_rejects_a_different_config_or_codebook() -> None:
    codebook = _disjoint_codebook(4)
    config = _config(4)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()

    with pytest.raises(ValueError, match="identity"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=replace(config, construction_seed=99),
            snapshot=snapshot,
        )

    changed_codebook = dict(codebook)
    changed_codebook[0] = tuple(range(1, K + 1))
    with pytest.raises(ValueError, match="identity"):
        SparseTemporalCortex.from_snapshot(
            changed_codebook,
            config=config,
            snapshot=snapshot,
        )


def test_checkpoint_rejects_a_segment_with_a_non_source_contact() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _source, target = _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    owner, contacts, permanences = snapshot.learned.segments[0]
    corrupted_contacts = (target.basket_spikes[0],) + contacts[1:]
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=((owner, corrupted_contacts, permanences),)
            + snapshot.learned.segments[1:],
        ),
    )

    with pytest.raises(ValueError, match="source/relay pairing"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_token_column_codes_must_be_unique_as_unordered_sdrs() -> None:
    config = SparseTemporalConfig(
        column_count=4,
        active_columns=2,
        base_cells_per_column=1,
        max_pyramidal_cells=4,
        contacts_per_segment=4,
        theta_match=2,
        theta_active=4,
    )
    with pytest.raises(ValueError, match="must not share"):
        SparseTemporalCortex(
            {0: (0, 1), 1: (1, 0)},
            config=config,
        )

def test_another_active_segment_on_the_correct_winner_is_not_punished() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    duplicate = snapshot.learned.segments[0]
    duplicated = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=snapshot.learned.segments + (duplicate,),
        ),
    )
    circuit = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=duplicated,
    )

    _learn_pair(circuit, 0, 1)

    assert circuit.learned_snapshot().segments[-1][2] == duplicate[2]


def test_multiple_active_owners_in_one_column_stop_before_state_changes() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    owner, contacts, permanences = snapshot.learned.segments[0]
    column = owner // config.base_cells_per_column
    other_owner = column * config.base_cells_per_column
    if other_owner == owner:
        other_owner += 1
    ambiguous = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=snapshot.learned.segments
            + ((other_owner, contacts, permanences),),
        ),
    )
    circuit = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=ambiguous,
    )
    circuit.reset_transient()
    circuit.recognize(0)
    before = circuit.snapshot()

    with pytest.raises(RuntimeError, match="multiple active segment owners"):
        circuit.recognize(1)

    assert circuit.snapshot() == before


def test_underpowered_gaba_stops_without_advancing_transient_state() -> None:
    codebook = _disjoint_codebook(2)
    config = replace(
        _config(2),
        novel_current_pa=650.0,
        novel_rank_spacing_pa=50.0,
        matching_current_pa=705.0,
        active_current_pa=760.0,
        homeostatic_current_pa=1.0,
        basket_inhibition_pa=1.0,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    before = circuit.snapshot()

    with pytest.raises(RuntimeError, match="multiple pyramidal spikes"):
        circuit.learn(0)

    assert circuit.snapshot() == before


def test_proximal_source_failure_stops_before_any_state_change() -> None:
    codebook = _disjoint_codebook(2)
    circuit = SparseTemporalCortex(
        codebook,
        config=replace(_config(2), proximal_current_pa=1.0),
    )
    before = circuit.snapshot()

    with pytest.raises(RuntimeError, match="did not spike"):
        circuit.learn(0)

    assert circuit.snapshot() == before


def test_column_capacity_stop_is_global_and_transactional() -> None:
    codebook = {
        0: (3, 4),
        1: (5, 6),
        2: (7, 8),
        3: (0, 1),
        4: (2, 1),
    }
    config = SparseTemporalConfig(
        column_count=9,
        active_columns=2,
        base_cells_per_column=2,
        max_pyramidal_cells=18,
        contacts_per_segment=4,
        theta_match=4,
        theta_active=4,
        max_segments_per_cell=1,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 3)
    _learn_pair(circuit, 1, 3)
    circuit.reset_transient()
    circuit.learn(2)
    before = circuit.snapshot()

    with pytest.raises(RuntimeError, match="column capacity exhausted"):
        circuit.learn(4)

    assert circuit.snapshot() == before


def test_exhausted_column_recruits_a_physical_reserve_cell() -> None:
    codebook = {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3,),
    }
    config = SparseTemporalConfig(
        column_count=4,
        active_columns=1,
        base_cells_per_column=1,
        max_pyramidal_cells=5,
        contacts_per_segment=2,
        theta_match=2,
        theta_active=2,
        max_segments_per_cell=1,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    assert circuit.committed_pyramidal_cell_count == 4
    assert circuit.reserve_pyramidal_cells == 1

    _learn_pair(circuit, 0, 3)
    _learn_pair(circuit, 1, 3)

    assert circuit.committed_pyramidal_cell_count == 5
    assert circuit.reserve_pyramidal_cells == 0
    assert _recognize_pair(circuit, 0, 3)[1].predicted is True
    assert _recognize_pair(circuit, 1, 3)[1].predicted is True

    circuit.reset_transient()
    circuit.learn(2)
    before = circuit.snapshot()
    with pytest.raises(RuntimeError, match="physical reserve exhausted"):
        circuit.learn(3)
    assert circuit.snapshot() == before
