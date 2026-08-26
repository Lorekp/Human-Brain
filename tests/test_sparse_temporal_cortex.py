"""R3-F0 fixtures for the repaired physical temporal-memory seam.

These tests intentionally exercise the public deep-module boundary.  They do
not call a Python threshold tracker or inject elected winner ids directly.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import pytest

import humanbrain.sparse_temporal_cortex as sparse_temporal
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
    initial_permanence_ticks: int = 255,
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
        theta_match=18,
        theta_active=20,
        initial_permanence_ticks=initial_permanence_ticks,
        growth_policy=growth_policy,
        construction_seed=construction_seed,
    )


def test_permanence_config_is_tick_native_and_requires_explicit_thresholds() -> None:
    with pytest.raises(TypeError, match="theta_match"):
        SparseTemporalConfig(column_count=11)

    config = SparseTemporalConfig(
        column_count=11,
        active_columns=11,
        base_cells_per_column=1,
        max_pyramidal_cells=11,
        contacts_per_segment=20,
        theta_match=18,
        theta_active=20,
    )

    assert config.theta_match == 18
    assert config.theta_active == 20
    assert config.connected_permanence_ticks == 250
    assert config.initial_permanence_ticks == 255
    assert config.permanence_increment_ticks == 50
    assert config.negative_decrement_ticks == 2
    assert not hasattr(config, "connected_permanence")
    assert not hasattr(config, "initial_permanence")
    assert not hasattr(config, "permanence_increment")
    assert not hasattr(config, "negative_decrement")
    with pytest.raises(TypeError, match="connected_permanence"):
        SparseTemporalConfig(
            column_count=11,
            active_columns=11,
            base_cells_per_column=1,
            max_pyramidal_cells=11,
            contacts_per_segment=20,
            connected_permanence=0.5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connected_permanence_ticks", 0),
        ("connected_permanence_ticks", 250.0),
        ("initial_permanence_ticks", 0),
        ("initial_permanence_ticks", 501),
        ("permanence_increment_ticks", 0),
        ("negative_decrement_ticks", True),
    ],
)
def test_permanence_config_rejects_non_live_or_non_integer_ticks(
    field: str,
    value: object,
) -> None:
    config = SparseTemporalConfig(
        column_count=11,
        theta_match=18,
        theta_active=20,
    )

    with pytest.raises(ValueError, match=field):
        replace(config, **{field: value})


def test_max_segments_per_cell_matches_native_owner_load_domain() -> None:
    config = _config(1)

    assert replace(config, max_segments_per_cell=1).max_segments_per_cell == 1
    assert replace(config, max_segments_per_cell=255).max_segments_per_cell == 255
    with pytest.raises(ValueError, match=r"max_segments_per_cell.*\[1, 255\]"):
        replace(config, max_segments_per_cell=256)


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        (
            {
                "column_count": 256,
                "active_columns": 256,
                "max_pyramidal_cells": 4 * 256,
                "prediction_quorum": 1,
            },
            "active_columns",
        ),
        (
            {
                "base_cells_per_column": 256,
                "max_pyramidal_cells": 11 * 256,
                "novel_race_cap": 256,
                "novel_current_pa": 20_000.0,
                "matching_current_pa": 20_100.0,
                "active_current_pa": 20_200.0,
            },
            "base_cells_per_column",
        ),
        (
            {
                "novel_race_cap": 256,
                "novel_current_pa": 20_000.0,
                "matching_current_pa": 20_100.0,
                "active_current_pa": 20_200.0,
            },
            "novel_race_cap",
        ),
        ({"theta_match": 256, "theta_active": 256}, "theta_match"),
        ({"theta_active": 256}, "theta_active"),
        (
            {
                "column_count": 255,
                "active_columns": 255,
                "max_pyramidal_cells": 4 * 255,
                "prediction_quorum": 256,
            },
            "prediction_quorum",
        ),
    ],
)
def test_rust_uint8_config_fields_reject_out_of_domain_values(
    updates: dict[str, int | float],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{field}.*\[1, 255\]"):
        replace(_config(1), **updates)


@pytest.mark.parametrize(
    "field",
    [
        "active_columns",
        "base_cells_per_column",
        "theta_match",
        "theta_active",
        "novel_race_cap",
        "prediction_quorum",
    ],
)
@pytest.mark.parametrize("value", [True, 1.0])
def test_rust_uint8_config_fields_reject_bool_and_float(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field}.*\[1, 255\]"):
        replace(_config(1), **{field: value})


def test_novel_race_cap_rejects_values_above_the_native_structural_cap() -> None:
    with pytest.raises(ValueError, match=r"novel_race_cap.*\[1, 8\]"):
        replace(_config(1), novel_race_cap=9)


@pytest.mark.parametrize(
    ("field", "updates"),
    [
        (
            "column_count",
            {
                "column_count": 1 << 32,
                "base_cells_per_column": 1,
                "max_pyramidal_cells": (1 << 32) - 1,
            },
        ),
        ("max_pyramidal_cells", {"max_pyramidal_cells": 1 << 32}),
        ("construction_seed", {"construction_seed": 1 << 32}),
    ],
)
def test_native_uint32_config_fields_reject_out_of_domain_values(
    field: str,
    updates: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match=f"{field}.*uint32"):
        replace(_config(1), **updates)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("column_count", True),
        ("column_count", 11.0),
        ("max_pyramidal_cells", True),
        ("max_pyramidal_cells", 44.0),
        ("construction_seed", True),
        ("construction_seed", 17.0),
    ],
)
def test_native_uint32_config_fields_reject_bool_and_float(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field}.*uint32"):
        replace(_config(1), **{field: value})


def test_config_rejects_a_derived_physical_cell_id_overflow() -> None:
    with pytest.raises(ValueError, match=r"physical cell.*uint32"):
        replace(
            _config(1),
            base_cells_per_column=1,
            max_pyramidal_cells=1 << 31,
        )


@pytest.mark.parametrize(
    ("codebook", "match"),
    [
        ({0.0: tuple(range(K))}, "token ID.*uint32"),
        ({(1 << 32) - 1: tuple(range(K))}, "token ID.*live uint32"),
        ({1 << 32: tuple(range(K))}, "token ID.*uint32"),
        (
            {0: (0.0, *range(1, K))},
            "column ID.*uint32",
        ),
        (
            {0: ((1 << 32) - 1, *range(1, K))},
            "column ID.*live uint32",
        ),
    ],
)
def test_codebook_ids_must_be_exact_native_uint32_values(
    codebook: dict[object, tuple[object, ...]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SparseTemporalCortex(codebook, config=_config(1))


@pytest.mark.parametrize("method", ["learn", "recognize"])
@pytest.mark.parametrize("token_id", [0.0, True, -1, (1 << 32) - 1, 1 << 32])
def test_arriving_token_ids_must_be_exact_native_uint32_values(
    method: str,
    token_id: object,
) -> None:
    circuit = SparseTemporalCortex(_disjoint_codebook(1), config=_config(1))

    with pytest.raises(ValueError, match=r"token_id.*uint32"):
        getattr(circuit, method)(token_id)


def test_learned_snapshot_stores_one_integer_tick_per_physical_contact_pair() -> None:
    codebook = _disjoint_codebook(2)
    circuit = SparseTemporalCortex(codebook, config=_config(2))

    source, _target = _learn_pair(circuit, 0, 1)
    learned = circuit.learned_snapshot()

    assert len(learned.segments) == K
    for _owner, sources, permanence_ticks in learned.segments:
        assert len(sources) == len(permanence_ticks) == 10
        assert set(sources).issubset(source.winner_cells)
        assert permanence_ticks == (255,) * 10
        assert all(type(value) is int for value in permanence_ticks)
    assert circuit.permanence_tick_values() == (255,) * (K * 10)
    assert not hasattr(circuit, "permanence_values")


def test_contact_position_sampler_freezes_philox_domains_and_literal_vector() -> None:
    assert sparse_temporal.contact_position_sampling_contract() == {
        "protocol": "random123-philox2x64-10-contact-positions-v1",
        "oracle": "humanbrain.procedural_topology._philox2x64_10",
        "initial_domain_hex": "52334630494e4954",
        "regrowth_domain_hex": "5233463052475257",
        "bounded_mapping": "lemire-multiply-high-u64-rejection",
        "without_replacement": "partial-fisher-yates",
        "initial_address": (
            "key=seed^initial_domain,"
            "counter=(learning_step,(attempt<<32)|draw)"
        ),
        "regrowth_address": (
            "key=philox(owner,segment_id,seed^regrowth_domain)[0],"
            "counter=(learning_step,(attempt<<32)|draw)"
        ),
    }

    active_columns = 8
    codebook = {
        token: tuple(range(token * active_columns, (token + 1) * active_columns))
        for token in range(2)
    }
    config = SparseTemporalConfig(
        column_count=16,
        active_columns=active_columns,
        base_cells_per_column=2,
        max_pyramidal_cells=32,
        contacts_per_segment=8,
        theta_match=6,
        theta_active=8,
        construction_seed=17,
    )
    circuit = SparseTemporalCortex(codebook, config=config)

    source, _target = _learn_pair(circuit, 0, 1)
    expected_sources = tuple(source.winner_cells[index] for index in (2, 3, 5, 7))

    assert {segment[1] for segment in circuit.learned_snapshot().segments} == {
        expected_sources
    }

    regrowth_key, _ = sparse_temporal._philox2x64_10(
        44,
        0,
        17 ^ sparse_temporal._REGROWTH_CONTACT_DOMAIN,
    )
    assert regrowth_key == 0xC5AC1D69C2B8FBC5
    assert sparse_temporal._sample_contact_positions(
        11,
        3,
        key=regrowth_key,
        learning_step=3,
    ) == (2, 6, 7)


def test_contact_position_sampler_rejects_biased_low_words(monkeypatch) -> None:
    calls: list[tuple[int, int, int]] = []
    words = iter((0, (1 << 64) - 1))

    def fake_philox(counter_0: int, counter_1: int, key: int):
        calls.append((counter_0, counter_1, key))
        return next(words), 0

    monkeypatch.setattr(sparse_temporal, "_philox2x64_10", fake_philox)

    assert sparse_temporal._bounded_philox_index(
        3,
        key=19,
        learning_step=23,
        draw=7,
    ) == 2
    assert calls == [
        (23, 7, 19),
        (23, (1 << 32) | 7, 19),
    ]


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

    source_a, learned_b = _learn_pair(circuit, 0, 1)
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
    for _owner, sources, permanence_ticks in circuit.learned_snapshot().segments:
        assert len(sources) == len(permanence_ticks) == K - 1
        assert set(sources).issubset(source_a.winner_cells)
        assert {
            circuit.config.max_pyramidal_cells + cell for cell in sources
        }.issubset(source_a.relay_spikes)
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
        config=_config(2, initial_permanence_ticks=205),
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

    context_two_sources = set(context_two.winner_cells)
    for _owner, sources, permanence_ticks in circuit.learned_snapshot().segments:
        assert len(sources) == len(permanence_ticks) == 2
        assert set(sources) == context_two_sources
        assert all(1 <= ticks <= 500 for ticks in permanence_ticks)
    assert set(context_one.winner_cells) != context_two_sources


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

    below_match = circuit.diagnose_context(paired_context(8))
    at_match = circuit.diagnose_context(paired_context(9))
    below_active = circuit.diagnose_context(paired_context(9))
    at_active = circuit.diagnose_context(paired_context(10))

    assert below_match.max_potential_contacts == 16
    assert below_match.matching_segments == 0
    assert at_match.max_potential_contacts == 18
    assert at_match.matching_segments >= 1
    assert below_active.active_segments == 0
    assert at_active.max_connected_contacts == 20
    assert at_active.active_segments >= 1


@pytest.mark.parametrize("cell_id", [0.0, True, -1, (1 << 32) - 1, 1 << 32])
def test_context_diagnostics_requires_exact_native_uint32_cell_ids(
    cell_id: object,
) -> None:
    circuit = SparseTemporalCortex(_disjoint_codebook(1), config=_config(1))

    with pytest.raises(ValueError, match=r"context_spikes cell ID.*uint32"):
        circuit.diagnose_context((cell_id,))


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
    assert sum(value == 249 for value in circuit.permanence_tick_values()) == K * 10


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
        assert len(segment[1]) == 10
        assert set(segment[1]).issubset(bound_source.winner_cells)
        assert len(generic_segment[1]) == 10
        assert set(generic_segment[1]).issubset(
            generic_source.proximal_spikes
        )
        bound_positions = {
            bound_source.winner_cells.index(cell)
            for cell in segment[1]
        }
        generic_positions = {
            generic_source.proximal_spikes.index(cell)
            for cell in generic_segment[1]
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


def test_generic_checkpoint_resume_accepts_its_exact_reachable_context() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2, growth_policy=GrowthPolicy.GENERIC_AFFERENTS)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()

    restored = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=snapshot,
    )

    assert restored.snapshot() == snapshot


def test_checkpoint_round_trips_segments_after_punishment_drops_all_pairs() -> None:
    codebook = _disjoint_codebook(3)
    config = replace(
        _config(3),
        connected_permanence_ticks=1,
        initial_permanence_ticks=2,
        negative_decrement_ticks=2,
    )
    circuit = SparseTemporalCortex(codebook, config=config)

    _learn_pair(circuit, 0, 1)
    _learn_pair(circuit, 0, 2)
    snapshot = circuit.snapshot()
    empty_segments = tuple(
        segment for segment in snapshot.learned.segments if not segment[1]
    )

    assert len(empty_segments) == K
    assert all(permanence_ticks == () for _, _, permanence_ticks in empty_segments)

    restored = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=snapshot,
    )

    assert restored.segment_count == circuit.segment_count
    assert restored.snapshot() == snapshot


@pytest.mark.parametrize("empty_field", ["previous_context", "previous_winner_context"])
def test_checkpoint_rejects_only_one_empty_transient_context(empty_field: str) -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    circuit.recognize(0)
    snapshot = circuit.snapshot()
    corrupted = replace(snapshot, **{empty_field: ()})

    with pytest.raises(ValueError, match="both be empty or nonempty"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_bound_checkpoint_rejects_different_valid_paired_contexts() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    left = SparseTemporalCortex(codebook, config=config)
    right = SparseTemporalCortex(codebook, config=config)
    left.recognize(0)
    right.recognize(1)
    snapshot = left.snapshot()
    corrupted = replace(
        snapshot,
        previous_winner_context=right.snapshot().previous_winner_context,
    )

    with pytest.raises(ValueError, match="bound-winner contexts must be identical"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


@pytest.mark.parametrize(
    ("growth_policy", "corrupt_field", "match"),
    [
        (
            GrowthPolicy.BOUND_WINNERS,
            "both",
            "bound-winner context must contain committed pyramidal winners",
        ),
        (
            GrowthPolicy.GENERIC_AFFERENTS,
            "previous_context",
            "generic-afferent context must use canonical proximal cells",
        ),
    ],
)
def test_checkpoint_rejects_the_other_policy_context_layout(
    growth_policy: GrowthPolicy,
    corrupt_field: str,
    match: str,
) -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2, growth_policy=growth_policy)
    circuit = SparseTemporalCortex(codebook, config=config)
    circuit.recognize(0)
    snapshot = circuit.snapshot()
    if growth_policy is GrowthPolicy.BOUND_WINNERS:
        wrong = tuple(
            2 * config.max_pyramidal_cells + config.column_count + column
            for column in codebook[0]
        )
        wrong += tuple(
            2 * config.max_pyramidal_cells + 2 * config.column_count + column
            for column in codebook[0]
        )
        corrupted = replace(
            snapshot,
            previous_context=wrong,
            previous_winner_context=wrong,
        )
    else:
        corrupted = replace(
            snapshot,
            previous_context=snapshot.previous_winner_context,
        )

    with pytest.raises(ValueError, match=match):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_generic_checkpoint_contexts_must_describe_the_same_token_code() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2, growth_policy=GrowthPolicy.GENERIC_AFFERENTS)
    left = SparseTemporalCortex(codebook, config=config)
    right = SparseTemporalCortex(codebook, config=config)
    left.recognize(0)
    right.recognize(1)
    snapshot = left.snapshot()
    corrupted = replace(
        snapshot,
        previous_context=right.snapshot().previous_context,
    )

    with pytest.raises(ValueError, match="same canonical token code"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_checkpoint_rejects_allocated_segment_count_above_owner_cap() -> None:
    codebook = _disjoint_codebook(2)
    config = replace(_config(2), max_segments_per_cell=1)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    owner, _sources, _ticks = snapshot.learned.segments[0]
    empty_segment = (owner, (), ())
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=(
                empty_segment,
                empty_segment,
                *snapshot.learned.segments[1:],
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"snapshot segment owner.*max_segments_per_cell",
    ):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


@pytest.mark.parametrize(
    ("location", "match"),
    [
        ("column_cell", "snapshot column cell ID.*uint32"),
        ("reserve_cursor", "snapshot reserve cursor.*uint32"),
        ("segment_owner", "snapshot segment owner.*uint32"),
        ("segment_source", "snapshot segment source.*uint32"),
        ("previous_context", "snapshot previous_context cell ID.*uint32"),
        (
            "previous_winner_context",
            "snapshot previous_winner_context cell ID.*uint32",
        ),
    ],
)
def test_checkpoint_ids_must_be_exact_native_uint32_values(
    location: str,
    match: str,
) -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()

    if location == "column_cell":
        column_cells = [list(cells) for cells in snapshot.learned.column_cells]
        column_cells[0][0] = float(column_cells[0][0])
        corrupted = replace(
            snapshot,
            learned=replace(
                snapshot.learned,
                column_cells=tuple(tuple(cells) for cells in column_cells),
            ),
        )
    elif location == "reserve_cursor":
        corrupted = replace(
            snapshot,
            learned=replace(
                snapshot.learned,
                next_reserve_cell=float(snapshot.learned.next_reserve_cell),
            ),
        )
    elif location in {"segment_owner", "segment_source"}:
        owner, sources, ticks = snapshot.learned.segments[0]
        if location == "segment_owner":
            row = (float(owner), sources, ticks)
        else:
            row = (owner, (float(sources[0]), *sources[1:]), ticks)
        corrupted = replace(
            snapshot,
            learned=replace(
                snapshot.learned,
                segments=(row, *snapshot.learned.segments[1:]),
            ),
        )
    else:
        context = getattr(snapshot, location)
        corrupted = replace(
            snapshot,
            **{location: (float(context[0]), *context[1:])},
        )

    with pytest.raises(ValueError, match=match):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


@pytest.mark.parametrize(
    ("location", "label"),
    [
        ("column_cell", "snapshot column cell ID"),
        ("segment_owner", "snapshot segment owner"),
        ("segment_source", "snapshot segment source"),
        ("previous_context", "snapshot previous_context cell ID"),
        (
            "previous_winner_context",
            "snapshot previous_winner_context cell ID",
        ),
    ],
)
def test_checkpoint_rejects_uint32_max_as_a_live_identity(
    location: str,
    label: str,
) -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    sentinel = (1 << 32) - 1

    if location == "column_cell":
        column_cells = [list(cells) for cells in snapshot.learned.column_cells]
        column_cells[0][0] = sentinel
        corrupted = replace(
            snapshot,
            learned=replace(
                snapshot.learned,
                column_cells=tuple(tuple(cells) for cells in column_cells),
            ),
        )
    elif location in {"segment_owner", "segment_source"}:
        owner, sources, ticks = snapshot.learned.segments[0]
        if location == "segment_owner":
            row = (sentinel, sources, ticks)
        else:
            row = (owner, (sentinel, *sources[1:]), ticks)
        corrupted = replace(
            snapshot,
            learned=replace(
                snapshot.learned,
                segments=(row, *snapshot.learned.segments[1:]),
            ),
        )
    else:
        context = getattr(snapshot, location)
        corrupted = replace(
            snapshot,
            **{location: (sentinel, *context[1:])},
        )

    with pytest.raises(ValueError, match=rf"{label}.*live uint32"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


@pytest.mark.parametrize(
    ("counter", "value"),
    [
        ("learning_step", -1),
        ("learning_step", 1 << 64),
        ("learning_step", True),
        ("learning_step", 1.0),
        ("event_step", -1),
        ("event_step", 1 << 64),
        ("event_step", True),
        ("event_step", 1.0),
    ],
)
def test_checkpoint_rejects_counters_outside_the_native_uint64_domain(
    counter: str,
    value: object,
) -> None:
    codebook = _disjoint_codebook(1)
    config = _config(1)
    circuit = SparseTemporalCortex(codebook, config=config)
    snapshot = circuit.snapshot()
    if counter == "learning_step":
        corrupted = replace(
            snapshot,
            learned=replace(snapshot.learned, learning_step=value),
        )
    else:
        corrupted = replace(snapshot, event_step=value)

    with pytest.raises(ValueError, match=f"snapshot {counter}.*uint64"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


@pytest.mark.parametrize(
    "growth_policy",
    [GrowthPolicy.BOUND_WINNERS, GrowthPolicy.GENERIC_AFFERENTS],
)
@pytest.mark.parametrize("contexts_present", [False, True])
def test_checkpoint_event_step_must_match_transient_context_presence(
    growth_policy: GrowthPolicy,
    contexts_present: bool,
) -> None:
    codebook = _disjoint_codebook(1)
    config = _config(1, growth_policy=growth_policy)
    circuit = SparseTemporalCortex(codebook, config=config)
    if contexts_present:
        circuit.recognize(0)
    snapshot = circuit.snapshot()
    corrupted = replace(snapshot, event_step=0 if contexts_present else 1)

    with pytest.raises(ValueError, match="event_step must be zero exactly"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_checkpoint_accepts_the_inclusive_uint64_counter_boundary() -> None:
    codebook = _disjoint_codebook(1)
    config = _config(1)
    circuit = SparseTemporalCortex(codebook, config=config)
    circuit.recognize(0)
    snapshot = circuit.snapshot()
    boundary = replace(
        snapshot,
        learned=replace(snapshot.learned, learning_step=(1 << 64) - 1),
        event_step=(1 << 64) - 1,
    )

    restored = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=boundary,
    )

    assert restored.snapshot() == boundary


@pytest.mark.parametrize(
    ("counter", "method"),
    [("learning_step", "learn"), ("event_step", "recognize")],
)
def test_uint64_counter_overflow_stops_before_state_changes(
    counter: str,
    method: str,
) -> None:
    codebook = _disjoint_codebook(1)
    config = _config(1)
    circuit = SparseTemporalCortex(codebook, config=config)
    circuit.recognize(0)
    snapshot = circuit.snapshot()
    if counter == "learning_step":
        boundary = replace(
            snapshot,
            learned=replace(snapshot.learned, learning_step=(1 << 64) - 1),
        )
    else:
        boundary = replace(snapshot, event_step=(1 << 64) - 1)
    circuit = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=boundary,
    )
    before = circuit.snapshot()

    with pytest.raises(RuntimeError, match=f"STOP: {counter}.*uint64 overflow"):
        getattr(circuit, method)(0)

    assert circuit.snapshot() == before


@pytest.mark.parametrize(
    ("counter", "method"),
    [("learning_step", "learn"), ("event_step", "recognize")],
)
def test_unknown_token_preflight_precedes_uint64_counter_stop(
    counter: str,
    method: str,
) -> None:
    codebook = _disjoint_codebook(1)
    config = _config(1)
    circuit = SparseTemporalCortex(codebook, config=config)
    circuit.recognize(0)
    snapshot = circuit.snapshot()
    if counter == "learning_step":
        boundary = replace(
            snapshot,
            learned=replace(snapshot.learned, learning_step=(1 << 64) - 1),
        )
    else:
        boundary = replace(snapshot, event_step=(1 << 64) - 1)
    circuit = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=boundary,
    )
    before = circuit.snapshot()

    with pytest.raises(ValueError, match="unknown token 1"):
        getattr(circuit, method)(1)

    assert circuit.snapshot() == before


@pytest.mark.parametrize(
    ("counter", "method"),
    [("learning_step", "learn"), ("event_step", "recognize")],
)
def test_physics_stop_precedes_counter_overflow_without_state_changes(
    counter: str,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codebook = _disjoint_codebook(1)
    config = _config(1)
    circuit = SparseTemporalCortex(codebook, config=config)
    circuit.recognize(0)
    snapshot = circuit.snapshot()
    if counter == "learning_step":
        boundary = replace(
            snapshot,
            learned=replace(snapshot.learned, learning_step=(1 << 64) - 1),
        )
    else:
        boundary = replace(snapshot, event_step=(1 << 64) - 1)
    circuit = SparseTemporalCortex.from_snapshot(
        codebook,
        config=config,
        snapshot=boundary,
    )
    before = circuit.snapshot()

    def fail_physics(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("STOP: active human cells did not spike")

    monkeypatch.setattr(circuit._physics, "race_currents", fail_physics)

    with pytest.raises(RuntimeError, match="active human cells did not spike"):
        getattr(circuit, method)(0)

    assert circuit.snapshot() == before


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

    with pytest.raises(ValueError, match="non-normalized source"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


@pytest.mark.parametrize(
    "growth_policy",
    [GrowthPolicy.BOUND_WINNERS, GrowthPolicy.GENERIC_AFFERENTS],
)
def test_checkpoint_rejects_segment_sources_from_the_other_policy_domain(
    growth_policy: GrowthPolicy,
) -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2, growth_policy=growth_policy)
    circuit = SparseTemporalCortex(codebook, config=config)
    source_step, _target_step = _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    owner, sources, permanence_ticks = snapshot.learned.segments[0]
    if growth_policy is GrowthPolicy.BOUND_WINNERS:
        wrong_sources = tuple(
            2 * config.max_pyramidal_cells + config.column_count + column
            for column in codebook[0][: len(sources)]
        )
    else:
        wrong_sources = tuple(sorted(source_step.winner_cells[: len(sources)]))
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=(
                (owner, wrong_sources, permanence_ticks),
                *snapshot.learned.segments[1:],
            ),
        ),
    )

    with pytest.raises(ValueError, match="configured source policy"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_bound_checkpoint_rejects_an_uncommitted_pyramidal_segment_source() -> None:
    codebook = _disjoint_codebook(2)
    base_config = _config(2)
    config = replace(
        base_config,
        max_pyramidal_cells=(
            base_config.max_pyramidal_cells + base_config.novel_race_cap
        ),
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    owner, sources, permanence_ticks = snapshot.learned.segments[0]
    uncommitted_source = snapshot.learned.next_reserve_cell
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=(
                (owner, (*sources[:-1], uncommitted_source), permanence_ticks),
                *snapshot.learned.segments[1:],
            ),
        ),
    )

    with pytest.raises(ValueError, match="segment source is not a committed"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_checkpoint_rejects_a_noncanonical_segment_source_order() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    owner, sources, permanence_ticks = snapshot.learned.segments[0]
    unsorted_sources = (sources[1], sources[0], *sources[2:])
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=(
                (owner, unsorted_sources, permanence_ticks),
                *snapshot.learned.segments[1:],
            ),
        ),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_checkpoint_rejects_unequal_or_zero_live_permanence_ticks() -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    owner, sources, permanence_ticks = snapshot.learned.segments[0]

    unequal = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=((owner, sources, permanence_ticks[:-1]),)
            + snapshot.learned.segments[1:],
        ),
    )
    with pytest.raises(ValueError, match="source/permanence-tick lengths"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=unequal,
        )

    empty_sources_with_tick = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=((owner, (), (1,)), *snapshot.learned.segments[1:]),
        ),
    )
    with pytest.raises(ValueError, match="source/permanence-tick lengths"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=empty_sources_with_tick,
        )

    zero_live = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            segments=((owner, sources, (0,) + permanence_ticks[1:]),)
            + snapshot.learned.segments[1:],
        ),
    )
    with pytest.raises(ValueError, match="live permanence tick"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=zero_live,
        )


@pytest.mark.parametrize("context_field", ["previous_context", "previous_winner_context"])
def test_checkpoint_rejects_width_correct_nonpaired_transient_context(
    context_field: str,
) -> None:
    codebook = _disjoint_codebook(2)
    config = _config(2)
    circuit = SparseTemporalCortex(codebook, config=config)
    _source, target = _learn_pair(circuit, 0, 1)
    snapshot = circuit.snapshot()
    context = getattr(snapshot, context_field)
    malformed = context[:K] + (target.basket_spikes[0],) + context[K + 1 :]
    corrupted = replace(snapshot, **{context_field: malformed})

    with pytest.raises(ValueError, match=f"snapshot {context_field}.*source/relay"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


@pytest.mark.parametrize("corruption", ["partial", "noncontiguous"])
def test_checkpoint_rejects_malformed_reserve_cohorts(corruption: str) -> None:
    active_columns = 2
    context_count = 3
    target = context_count
    codebook = {
        token: tuple(
            range(token * active_columns, (token + 1) * active_columns)
        )
        for token in range(context_count + 1)
    }
    base_cells = len(codebook) * active_columns * 2
    config = SparseTemporalConfig(
        column_count=len(codebook) * active_columns,
        active_columns=active_columns,
        base_cells_per_column=2,
        max_pyramidal_cells=base_cells + 2 * active_columns,
        contacts_per_segment=4,
        theta_match=4,
        theta_active=4,
        max_segments_per_cell=1,
        novel_race_cap=2,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, target)
    _learn_pair(circuit, 1, target)
    _learn_pair(circuit, 2, target)
    snapshot = circuit.snapshot()
    column_cells = [list(cells) for cells in snapshot.learned.column_cells]
    left_column, right_column = codebook[target]
    if corruption == "partial":
        column_cells[right_column].append(column_cells[left_column].pop())
    else:
        left_recruited = -1
        right_recruited = -2
        column_cells[left_column][left_recruited], column_cells[right_column][right_recruited] = (
            column_cells[right_column][right_recruited],
            column_cells[left_column][left_recruited],
        )
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            column_cells=tuple(tuple(cells) for cells in column_cells),
        ),
    )

    with pytest.raises(ValueError, match="reserve cohort"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_checkpoint_rejects_complete_reserve_cohorts_in_reverse_gid_order() -> None:
    codebook = _disjoint_codebook(1)
    base_cells = K * 2
    config = SparseTemporalConfig(
        column_count=K,
        active_columns=K,
        base_cells_per_column=2,
        max_pyramidal_cells=base_cells + 16,
        contacts_per_segment=20,
        theta_match=18,
        theta_active=20,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    snapshot = circuit.snapshot()
    column_cells = [list(cells) for cells in snapshot.learned.column_cells]
    column_cells[0].extend(range(base_cells + 8, base_cells + 16))
    column_cells[0].extend(range(base_cells, base_cells + 8))
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            column_cells=tuple(tuple(cells) for cells in column_cells),
            next_reserve_cell=base_cells + 16,
        ),
    )

    with pytest.raises(ValueError, match="column cells must be strictly increasing"):
        SparseTemporalCortex.from_snapshot(
            codebook,
            config=config,
            snapshot=corrupted,
        )


def test_checkpoint_rejects_a_committed_reserve_cohort_without_an_owned_segment() -> None:
    codebook = _disjoint_codebook(1)
    base_cells = K * 2
    config = SparseTemporalConfig(
        column_count=K,
        active_columns=K,
        base_cells_per_column=2,
        max_pyramidal_cells=base_cells + 8,
        contacts_per_segment=20,
        theta_match=18,
        theta_active=20,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    snapshot = circuit.snapshot()
    column_cells = [list(cells) for cells in snapshot.learned.column_cells]
    column_cells[0].extend(range(base_cells, base_cells + 8))
    corrupted = replace(
        snapshot,
        learned=replace(
            snapshot.learned,
            column_cells=tuple(tuple(cells) for cells in column_cells),
            next_reserve_cell=base_cells + 8,
        ),
    )

    with pytest.raises(ValueError, match="reserve cohort has no owned segment"):
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


def test_exhausted_columns_recruit_full_cohorts_for_distinct_replayable_codes(
    monkeypatch,
) -> None:
    active_columns = 2
    context_count = 4
    target = context_count
    codebook = {
        token: tuple(
            range(token * active_columns, (token + 1) * active_columns)
        )
        for token in range(context_count + 1)
    }
    base_cells = len(codebook) * active_columns * 2
    config = SparseTemporalConfig(
        column_count=len(codebook) * active_columns,
        active_columns=active_columns,
        base_cells_per_column=2,
        max_pyramidal_cells=base_cells + 2 * active_columns,
        contacts_per_segment=4,
        theta_match=4,
        theta_active=4,
        max_segments_per_cell=1,
        novel_race_cap=2,
    )
    circuit = SparseTemporalCortex(codebook, config=config)

    _learn_pair(circuit, 0, target)
    _learn_pair(circuit, 1, target)
    _source_two, after_recruit = _learn_pair(circuit, 2, target)

    assert circuit.committed_pyramidal_cell_count == base_cells + 4
    _source_three, next_cohort_code = _learn_pair(circuit, 3, target)
    assert next_cohort_code.winner_cells != after_recruit.winner_cells
    driven_widths: list[int] = []
    race_currents = circuit._physics.race_currents_with_inhibition

    def record_race_width(cell_ids, currents, **kwargs):
        driven_widths.append(sum(float(current) > 0.0 for current in currents))
        return race_currents(cell_ids, currents, **kwargs)

    monkeypatch.setattr(
        circuit._physics,
        "race_currents_with_inhibition",
        record_race_width,
    )
    assert _recognize_pair(circuit, 2, target)[1].winner_cells == (
        after_recruit.winner_cells
    )
    assert _recognize_pair(circuit, 3, target)[1].winner_cells == (
        next_cohort_code.winner_cells
    )
    assert driven_widths
    assert max(driven_widths) <= config.novel_race_cap


def test_full_multicolumn_cohort_reservation_stops_atomically_when_short() -> None:
    active_columns = 2
    context_count = 3
    target = context_count
    codebook = {
        token: tuple(
            range(token * active_columns, (token + 1) * active_columns)
        )
        for token in range(context_count + 1)
    }
    base_cells = len(codebook) * active_columns * 2
    config = SparseTemporalConfig(
        column_count=len(codebook) * active_columns,
        active_columns=active_columns,
        base_cells_per_column=2,
        max_pyramidal_cells=base_cells + 3,
        contacts_per_segment=4,
        theta_match=4,
        theta_active=4,
        max_segments_per_cell=1,
        novel_race_cap=2,
    )
    circuit = SparseTemporalCortex(codebook, config=config)
    _learn_pair(circuit, 0, target)
    _learn_pair(circuit, 1, target)
    circuit.reset_transient()
    circuit.learn(2)
    before = circuit.snapshot()

    with pytest.raises(RuntimeError, match="full reserve cohort"):
        circuit.learn(target)

    assert circuit.snapshot() == before
