from __future__ import annotations

import numpy as np
from humanbrain.development import (
    DevelopmentalGrowthConfig,
    DevelopmentalGrowthController,
    developmental_cell_order,
    grow_developmentally,
)
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentModelConfig


def _circuit(active_cells: int = 8) -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=16,
        config=RecurrentCircuitConfig(
            reservoir_cells=24,
            active_reservoir_cells=active_cells,
            input_fanout=4,
            recurrent_fanout=4,
            token_interval_ms=2.0,
            input_pulse_ms=1.2,
            input_current_pa=8_000.0,
            recurrent_current_pa=325.0,
            eligibility_tau_ms=40.0,
            output_base_current_pa=700.0,
            output_race_ms=20.0,
            readout_learning_rule="three_factor",
            recurrent_topology="uniform_no_self",
            random_seed=17,
        ),
    )


def _development(**changes: object) -> DevelopmentalGrowthConfig:
    values: dict[str, object] = {
        "stage_active_cells": (8, 16, 24),
        "initial_recurrent_fanout": 2,
        "final_recurrent_fanout": 2,
        "structural_seed": 91,
    }
    values.update(changes)
    return DevelopmentalGrowthConfig(**values)


def test_cell_birth_orders_are_deterministic_and_explicitly_different() -> None:
    layered = developmental_cell_order(96, "layered_prefix", 12)
    random = developmental_cell_order(96, "random", 12)

    np.testing.assert_array_equal(
        layered,
        developmental_cell_order(96, "layered_prefix", 999),
    )
    np.testing.assert_array_equal(random, developmental_cell_order(96, "random", 12))
    assert sorted(layered.tolist()) == list(range(96))
    assert sorted(random.tolist()) == list(range(96))
    assert not np.array_equal(layered, random)
    assert len(set((layered[:6] // 6).tolist())) == 1
    assert set((layered[:6] % 6).tolist()) == set(range(6))


def test_recruitment_keeps_unborn_cells_dormant_then_grows_their_synapses() -> None:
    circuit = _circuit()
    controller = DevelopmentalGrowthController(circuit, _development())
    unborn = ~controller.active_mask

    for token in range(8):
        circuit.observe(token)
        circuit.strengthen((token + 1) % circuit.token_cells)
        controller.record_plasticity()

    assert not np.any(circuit.eligibility[unborn])
    assert not np.any(circuit._reservoir.state.voltage_relative_mv[unborn])
    assert not np.any(circuit._recurrent_weights_pa[unborn])
    assert not np.any(circuit._readout_synapses_pa[:, unborn])

    old_mask = controller.active_mask.copy()
    recruited = controller.recruit_to(16)
    new_cells = controller.active_mask & ~old_mask

    assert recruited == 8
    assert circuit.active_reservoir_cells == 16
    assert np.all(np.count_nonzero(circuit._recurrent_weights_pa[new_cells], axis=1) == 4)
    assert np.all(controller.active_mask[circuit._afferent_targets])


def test_every_runtime_structure_mutation_synchronizes_bound_backend(monkeypatch) -> None:
    circuit = _circuit()
    synchronize_calls = 0
    synchronize = circuit.synchronize_structure_from_host

    def record_synchronization() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1
        synchronize()

    monkeypatch.setattr(circuit, "synchronize_structure_from_host", record_synchronization)
    controller = DevelopmentalGrowthController(circuit, _development())
    assert synchronize_calls == 1

    controller.recruit_to(16)
    assert synchronize_calls == 2

    controller.cell_activity[controller.active_mask] = np.linspace(0.25, 2.0, 16)
    controller._stage_examples = 1
    controller.apply_activity_homeostasis()
    assert synchronize_calls == 3

    controller.prune_to_final_fanout()
    assert synchronize_calls == 4


def test_activity_pruning_selects_trained_structural_synapses() -> None:
    circuit = _circuit(active_cells=24)
    controller = DevelopmentalGrowthController(
        circuit,
        _development(
            stage_active_cells=(24, 24, 24),
            homeostatic_scaling=False,
            pruning_rule="activity",
        ),
    )
    controller.structural_exposure[circuit._recurrent_weights_pa != 0.0] = 1.0
    controller.structural_score[:] = np.arange(4, dtype=np.float64)[None, :]

    pruned = controller.prune_to_final_fanout()

    assert pruned == 24 * 2
    assert controller.recurrent_synapse_count == 24 * 2
    np.testing.assert_array_equal(
        np.flatnonzero(circuit._recurrent_weights_pa[0]),
        np.asarray((2, 3)),
    )


def test_pruning_homeostasis_preserves_surviving_cells_incoming_strength() -> None:
    circuit = _circuit(active_cells=24)
    controller = DevelopmentalGrowthController(
        circuit,
        _development(
            stage_active_cells=(24, 24, 24),
            homeostatic_scaling=True,
            pruning_rule="activity",
        ),
    )
    controller.structural_exposure[circuit._recurrent_weights_pa != 0.0] = 1.0
    controller.structural_score[:] = np.arange(4, dtype=np.float64)[None, :]
    before = np.zeros(24, dtype=np.float64)
    mask = circuit._recurrent_weights_pa != 0.0
    np.add.at(
        before,
        circuit._recurrent_targets[mask],
        np.abs(circuit._recurrent_weights_pa[mask]),
    )

    controller.prune_to_final_fanout()
    after = np.zeros(24, dtype=np.float64)
    mask = circuit._recurrent_weights_pa != 0.0
    np.add.at(
        after,
        circuit._recurrent_targets[mask],
        np.abs(circuit._recurrent_weights_pa[mask]),
    )
    surviving = after > 0.0

    np.testing.assert_allclose(after[surviving], before[surviving], rtol=1e-14, atol=1e-14)


def test_staged_and_full_start_conditions_receive_matched_examples() -> None:
    model_config = RecurrentModelConfig(
        growth_epochs=2,
        circuit=_circuit().config,
    )
    staged = grow_developmentally(
        ["cell growth cell growth"],
        model_config,
        _development(),
    )
    full = grow_developmentally(
        ["cell growth cell growth"],
        model_config,
        _development(stage_active_cells=(24, 24, 24)),
    )

    assert staged.attempted_updates == full.attempted_updates
    assert staged.model._circuit._growth_examples == full.model._circuit._growth_examples
    assert staged.model.active_reservoir_cells == full.model.active_reservoir_cells == 24
    assert staged.final_recurrent_synapses == full.final_recurrent_synapses == 24 * 2
    assert tuple(stage.active_cells for stage in staged.stages) == (8, 16, 24)
    np.testing.assert_array_equal(
        staged.model._circuit._output_heterogeneity_pa,
        full.model._circuit._output_heterogeneity_pa,
    )


def test_developmental_checkpoint_preserves_cell_and_structural_synapse_state(
    tmp_path,
) -> None:
    training = grow_developmentally(
        ["cell growth cell growth"],
        RecurrentModelConfig(growth_epochs=2, circuit=_circuit().config),
        _development(),
    )
    path = tmp_path / "developmental-cell.npz"
    training.model.save(path)

    loaded = training.model.load(path)

    assert loaded.active_reservoir_cells == 24
    np.testing.assert_array_equal(
        loaded._circuit._recurrent_weights_pa,
        training.model._circuit._recurrent_weights_pa,
    )
    np.testing.assert_array_equal(
        loaded._circuit._structural_scores,
        training.model._circuit._structural_scores,
    )
    np.testing.assert_array_equal(
        loaded._circuit._development_active_mask,
        training.model._circuit._development_active_mask,
    )
    assert loaded.predict_next("cell") == training.model.predict_next("cell")
