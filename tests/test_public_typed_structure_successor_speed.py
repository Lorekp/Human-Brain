from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from time import perf_counter

import numpy as np

from experiments.r3_pilot import public_typed_structure_successor as structure


_CHECKPOINT_SHA256 = "f65d43dfb24d500a96cee4ff0b4906546ada6474e3f3f621c1270655ed686519"
_SUPPORTED_LEDGER_SHA256 = "61d0361b0bb782dce09999e7ed3dc387f645460107105c0f42b393d2751575f1"


def _supported_fixture():
    signature = structure.TypedSignature(
        "int_identity",
        (structure.ProgramParameter("value", structure.ProgramType.Int()),),
        structure.ProgramType.Int(),
    )
    program = structure.tp.Program(("variable-0",), signature)
    observation = structure.observe_program_structure(program)
    action = structure.tp.FactorAction(
        (),
        "APPLY1",
        "absolute",
        "SELECTED",
        "NONE",
        None,
    )
    model, _ = structure.train_public_structure_successor()
    return model, observation, action


def test_compiled_supported_prediction_matches_every_stepwise_physical_race() -> None:
    model, observation, action = _supported_fixture()
    started = perf_counter()
    prediction = model.predict(observation, action)
    compiled_seconds = perf_counter() - started

    payload = json.dumps(
        asdict(prediction),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == _SUPPORTED_LEDGER_SHA256
    assert len(prediction.races) == 83

    started = perf_counter()
    references = tuple(
        structure.physical._physical_race(
            np.asarray(race.currents_pa, dtype=np.float64),
            race.candidates,
            bank=race.bank,
        )
        for race in prediction.races
    )
    stepwise_seconds = perf_counter() - started
    assert prediction.races == references

    assert model.checkpoint_sha256 == _CHECKPOINT_SHA256
    assert model.live_checkpoint_sha256 == _CHECKPOINT_SHA256
    assert prediction.checkpoint_before_sha256 == _CHECKPOINT_SHA256
    assert prediction.checkpoint_after_sha256 == _CHECKPOINT_SHA256
    assert not structure._RACE._unit_voltage.flags.writeable
    assert not structure._RACE._unit_voltage_threshold.flags.writeable
    print(
        {
            "compiled_prediction_seconds": compiled_seconds,
            "stepwise_83_races_seconds": stepwise_seconds,
        }
    )
