from __future__ import annotations

import numpy as np
import pytest

from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig


def _grow_sparse_model() -> RecurrentCellLanguageModel:
    return RecurrentCellLanguageModel.grow(
        ["abcabcabc"],
        RecurrentModelConfig(
            growth_epochs=2,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=16,
                input_fanout=4,
                recurrent_fanout=2,
                readout_learning_rule="sparse_three_factor",
                readout_fan_in=2,
                readout_structural_interval=2,
                readout_structural_grace=2,
                readout_turnover_probability=1.0,
            ),
        ),
    )


def test_sparse_checkpoint_owns_no_dense_classifier_and_continues_exactly(
    tmp_path,
) -> None:
    model = _grow_sparse_model()
    before = model._circuit._sparse_readout
    assert before is not None
    path = tmp_path / "sparse-readout.npz"

    model.save(path)

    with np.load(path, allow_pickle=False) as state:
        assert "readout_synapses_pa" not in state.files
        assert set(before.checkpoint_arrays()).issubset(state.files)
    loaded = RecurrentCellLanguageModel.load(path)
    after = loaded._circuit._sparse_readout
    assert after is not None
    assert loaded._circuit.local_update_count == model._circuit.local_update_count
    assert loaded._circuit.structural_recruitment_count == (
        model._circuit.structural_recruitment_count
    )
    assert loaded._circuit.structural_deletion_count == (
        model._circuit.structural_deletion_count
    )
    for key, value in before.checkpoint_arrays().items():
        np.testing.assert_array_equal(after.checkpoint_arrays()[key], value)
    assert loaded.predict_next("ab") == model.predict_next("ab")

    # Seed every row of the (K, R) trace bank identically in both circuits:
    # the sparse source space spans all K·R feature sources, so the exact-
    # continuation check below must cover the whole multi-trace state.
    eligibility = np.arange(1.0, 17.0)
    model._circuit._eligibility_traces[:] = eligibility
    loaded._circuit._eligibility_traces[:] = eligibility
    for readout in (before, after):
        readout.teach(
            target=99,
            winner=98,
            learning_rate_pa=8.0,
            maximum_weight_pa=1_600.0,
            # Production per-source decay table over the full K*R feature
            # space (exp(-T / tau_k), k = s // R); identical for both circuits.
            trace_decays=model._circuit.source_trace_decays(),
            target_post_gate=1.0,
            winner_post_gate=1.0,
            activity_decay=0.99,
        )
        presentation = readout.advance_target(99)
        readout.turnover(
            output=99,
            structural_event=presentation // 2,
            random_seed=17,
            minimum_age=2,
            turnover_probability=1.0,
            learning_rate_pa=8.0,
            maximum_weight_pa=1_600.0,
            homeostatic_rate=0.05,
        )
    for key, value in before.checkpoint_arrays().items():
        np.testing.assert_array_equal(after.checkpoint_arrays()[key], value)
    assert after.applied_updates == before.applied_updates
    assert after.structural_recruitments == before.structural_recruitments
    assert after.structural_deletions == before.structural_deletions


@pytest.mark.parametrize("corruption", ["candidate_order", "contact_subset"])
def test_sparse_checkpoint_rejects_malformed_anatomical_subsets(
    tmp_path,
    corruption: str,
) -> None:
    model = _grow_sparse_model()
    source = tmp_path / "source.npz"
    corrupt = tmp_path / f"corrupt-{corruption}.npz"
    model.save(source)
    with np.load(source, allow_pickle=False) as state:
        arrays = {key: np.array(state[key], copy=True) for key in state.files}
    if corruption == "candidate_order":
        arrays["readout_potential_sources"][0, 1] = arrays[
            "readout_potential_sources"
        ][0, 0]
    else:
        potential = set(arrays["readout_potential_sources"][0])
        replacement = next(source for source in range(16) if source not in potential)
        arrays["readout_contact_sources"][0, 0] = replacement
        arrays["readout_contact_sources"][0].sort()
    with corrupt.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    with pytest.raises(ValueError, match="potential|contact"):
        RecurrentCellLanguageModel.load(corrupt)
