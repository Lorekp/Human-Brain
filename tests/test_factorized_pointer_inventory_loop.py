from __future__ import annotations

from experiments.r4_integrated.variable_cardinality_inventory_fixture_v1 import (
    BINDING_OFF as FIXTURE_BINDING_OFF,
)
from experiments.r4_integrated.variable_cardinality_inventory_fixture_v1 import (
    FIXED_ARITY_LOOP_OFF,
    FRESH_CASES,
    PUBLIC_LESSONS,
    STOP_PROBES,
)
from humanbrain.factorized_pointer import (
    BINDING_OFF,
    INVENTORY_LOOP_OFF,
    FactorizedPointerCellModel,
    FactorizedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import SymbolInventory


def test_learned_inventory_loop_changes_cardinality_through_physical_races() -> None:
    assert FIXED_ARITY_LOOP_OFF == INVENTORY_LOOP_OFF
    assert FIXTURE_BINDING_OFF == BINDING_OFF
    model = FactorizedPointerCellModel.grow(
        tuple(
            FactorizedPointerTrainingExample(
                prompt=lesson.lesson_id,
                output=lesson.demonstrated_output,
                inventory=SymbolInventory(lesson.inventory),
                cue_text=lesson.cue,
                provenance_id=lesson.lesson_id,
            )
            for lesson in PUBLIC_LESSONS
        ),
        ModelConfig(max_order=8, max_phrases=64, min_phrase_frequency=2),
    )
    initial_digest = model.training_digest
    decisions: list[dict[str, object]] = []

    for case in FRESH_CASES:
        observation = case.public_observation()
        inventory = SymbolInventory(tuple(observation["inventory"]))
        treatment = model.complete(
            str(observation["case_id"]),
            inventory,
            cue_text=str(observation["cue"]),
        )
        treatment_trace = model.last_trace
        loop_off = model.complete(
            str(observation["case_id"]),
            inventory,
            cue_text=str(observation["cue"]),
            ablation=INVENTORY_LOOP_OFF,
        )
        binding_off = model.complete(
            str(observation["case_id"]),
            inventory,
            cue_text=str(observation["cue"]),
            ablation=BINDING_OFF,
        )
        decisions.append(
            {
                "case": case,
                "treatment": treatment,
                "treatment_trace": treatment_trace,
                "loop_off": loop_off,
                "binding_off": binding_off,
            }
        )

    stop_decisions: list[dict[str, object]] = []
    for probe in STOP_PROBES:
        observation = probe.public_observation()
        output = model.complete(
            str(observation["probe_id"]),
            SymbolInventory(tuple(observation["inventory"])),
            cue_text=str(observation["cue"]),
        )
        stop_decisions.append(
            {"probe": probe, "output": output, "trace": model.last_trace}
        )

    for decision in decisions:
        case = decision["case"]
        trace = decision["treatment_trace"]
        assert case.accepts_treatment_output(decision["treatment"])
        assert not case.accepts_treatment_output(decision["loop_off"])
        assert case.accepts_binding_off_output(decision["binding_off"])
        assert trace is not None
        assert tuple(race.decision for race in trace.inventory_loop_races) == (
            *("continue" for _ in case.inventory[:-1]),
            "exit",
        )
        assert all(
            race.spike_time_ms > 0.0 and race.simulated_steps > 0
            for race in trace.inventory_loop_races
        )

    for decision in stop_decisions:
        probe = decision["probe"]
        trace = decision["trace"]
        assert probe.accepts_output(decision["output"])
        assert trace is not None and trace.selected_stop
        assert trace.fragment_steps[-1].winner_kind == "stop"

    assert model.training_digest == initial_digest
