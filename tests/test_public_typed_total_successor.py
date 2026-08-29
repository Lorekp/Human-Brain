import ast
import inspect
from functools import cache
from pathlib import Path

import numpy as np

from experiments.r3_pilot import public_typed_total_successor as total


@cache
def _trained() -> tuple[
    total.FrozenTotalSuccessor,
    total.TotalSuccessorTrainingSummary,
]:
    return total.train_public_total_successor()


def test_total_successor_preserves_local_known_transition_bank() -> None:
    successor, summary = _trained()

    assert successor.weights_pa.shape == (10, 66, 66)
    assert successor.weights_pa.flags.writeable is False
    assert successor.bias_pa.flags.writeable is False
    assert summary.observations == summary.learned_known_synapses == 410
    assert summary.fixed_unknown_synapses == 10
    assert (summary.known_pre_min, summary.known_pre_max) == (-20, 20)
    assert summary.unknown_cell == 65
    assert successor.live_checkpoint_sha256 == successor.checkpoint_sha256

    predicted, race, _ = successor.predict(-7, "WRAP_ADD[2]")
    assert predicted == -5
    assert race.winner == "-5"
    assert race.simultaneous_spike_candidates == 1


def test_unseen_pre_action_has_one_physical_unknown_winner() -> None:
    successor, _ = _trained()

    predicted, race, _ = successor.predict(21, "DONE")

    assert predicted is total.UNKNOWN
    assert race.winner_cell == total.UNKNOWN_CELL
    assert race.winner == "UNKNOWN"
    assert race.simultaneous_spike_candidates == 1


def test_unknown_propagates_for_every_fixed_action() -> None:
    successor, _ = _trained()

    for action in total.banked.banked.ACTIONS:
        predicted, race, _ = successor.predict(total.UNKNOWN, action.label)
        assert predicted is total.UNKNOWN
        assert race.winner_cell == total.UNKNOWN_CELL
        assert race.simultaneous_spike_candidates == 1


def test_unknown_has_no_downstream_target_coincidence() -> None:
    features, digest = total.coincidence_features(
        (total.UNKNOWN, 3, 4),
        (3, 3, 5),
    )

    assert features.shape == (3, 66)
    assert np.count_nonzero(features[0]) == 0
    assert np.flatnonzero(features[1]).tolist() == [3 - total.banked.VALUE_MIN]
    assert np.count_nonzero(features[2]) == 0
    assert len(digest) == 64


def test_module_contains_no_composite_or_host_fallback_path() -> None:
    source = Path(total.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    predict = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "predict"
    )
    predict_source = ast.get_source_segment(source, predict)
    assert predict_source is not None

    assert "_apply_action" not in source
    assert "_execute" not in source
    assert "target_program" not in source
    assert "run_fresh" not in source
    assert "argmin" not in source
    assert "argmax" not in source
    assert "UNKNOWN" in predict_source
    assert "_physical_race(" in predict_source
    assert "coincidence_features" in source

    inference_source = "\n".join(
        inspect.getsource(function)
        for function in (
            total.FrozenTotalSuccessor.predict,
            total.FrozenTotalSuccessor.predict_many,
            total.coincidence_features,
        )
    )
    for forbidden in ("_apply_action", "_execute", "argmin", "argmax"):
        assert forbidden not in inference_source
