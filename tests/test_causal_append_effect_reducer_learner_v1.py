from __future__ import annotations

import inspect

import pytest
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as learner
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as ordered
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_append_effect_reducer_frontier_v1 as frontier_v1,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

UPSTREAM_CASES = (
    ([],),
    ([-20, -10, 0, 10],),
    ([-5, 1, 9],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
TRAIN_PROBE = (([-4, 2],), ([-4, 2, 5],), ([-4, 2, -8],))
COUNT_PROBE = (([3, 0],), ([3, 0, 6],), ([3, 0, -12],))
SUM_PROBE = (([2, 0],), ([2, 0, -10],), ([2, 0, 8],))
FINAL_CASES = UPSTREAM_CASES
SIGNATURE_ROLE = {
    "parameters": [
        {
            "role": "ARG0",
            "type": {
                "constructor": "List",
                "arguments": [{"constructor": "Int", "arguments": []}],
            },
        }
    ],
    "result": {"constructor": "Int", "arguments": []},
    "role_binding": "reverse-declaration-depth",
    "entrypoint_spelling_present": False,
    "parameter_spelling_present": False,
}


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "aggregate",
        (ProgramParameter("numbers", ProgramType.List(integer)),),
        integer,
    )


def _spans(seed: int, shift: int) -> tuple[dict[str, object], ...]:
    return (
        {
            "start_token": shift,
            "stop_token": shift + 8,
            "first_spike_step": 25,
            "assembly_cells": tuple(range(seed, seed + 12)),
            "spike_cells": tuple(range(seed, seed + 12)),
        },
        {
            "start_token": shift + 2,
            "stop_token": shift + 12,
            "first_spike_step": 26,
            "assembly_cells": tuple(range(seed + 16, seed + 28)),
            "spike_cells": tuple(range(seed + 16, seed + 28)),
        },
    )


def _cells(value: int) -> tuple[int, ...]:
    return output_cells.TypedOutputCellBank.encode(
        value, declared_type=output_cells.OutputType.INT
    ).active_global_cell_indices()


def _outputs(*values: int) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value, declared_type=output_cells.OutputType.INT
        )
        for value in values
    )


def _literals(
    repeated: int,
    residual_value: int,
    positions: tuple[int, int, int],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": position,
            "typed_output_active_cells": _cells(value),
        }
        for position, value in zip(
            positions, (repeated, residual_value, repeated), strict=True
        )
    )


def _desired(
    hole: int,
    offset: int,
    operand_action: int,
    cases: tuple[tuple[list[int]], ...] = UPSTREAM_CASES,
) -> tuple[int, ...]:
    if operand_action == ordered.REPEATED_TO_OPERAND_ZERO:
        return tuple(sum(hole < item + offset for item in values) for (values,) in cases)
    return tuple(sum(item + offset < hole for item in values) for (values,) in cases)


def _banked_residual_checkpoint(
    signature: TypedSignature,
) -> residual.FrozenCausalResidualLiteralRouteV1:
    literal_checkpoint = literal_role.fit_causal_relative_literal_role_v1(
        (
            literal_role.CausalLiteralEpisode(
                _spans(10, 0),
                _literals(-14, -8, (2, 4, 7)),
                SIGNATURE_ROLE,
                _desired(-14, -15, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
            literal_role.CausalLiteralEpisode(
                _spans(70, 10),
                _literals(-12, -6, (13, 15, 19)),
                SIGNATURE_ROLE,
                _desired(-12, -15, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
        ),
        signature=signature,
        cases=UPSTREAM_CASES,
        literals=(-15, -14, -12, -8, -6, 0),
        max_cost=8,
    )
    ordered_checkpoint = ordered.fit_causal_relative_ordered_operand_v1(
        literal_checkpoint,
        (
            ordered.CausalOrderedOperandEpisode(
                _spans(150, 30),
                _literals(-9, 5, (32, 39, 34)),
                SIGNATURE_ROLE,
                _desired(-9, -15, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
            ordered.CausalOrderedOperandEpisode(
                _spans(220, 50),
                _literals(-7, 13, (55, 52, 58)),
                SIGNATURE_ROLE,
                _desired(-7, -15, ordered.REPEATED_TO_OPERAND_ONE),
            ),
        ),
        signature=signature,
        cases=UPSTREAM_CASES,
        literals=(-15, -9, -7, 0),
        max_cost=8,
    )
    return residual.fit_causal_residual_literal_route_v1(
        ordered_checkpoint,
        (
            residual.CausalResidualLiteralEpisode(
                _spans(290, 80),
                _literals(-5, 2, (82, 90, 85)),
                SIGNATURE_ROLE,
                _desired(-5, 2, ordered.REPEATED_TO_OPERAND_ZERO),
            ),
            residual.CausalResidualLiteralEpisode(
                _spans(350, 100),
                _literals(-1, -3, (105, 102, 108)),
                SIGNATURE_ROLE,
                _desired(-1, -3, ordered.REPEATED_TO_OPERAND_ONE),
            ),
        ),
        signature=signature,
        cases=UPSTREAM_CASES,
        literals=(-15, -5, -3, -1, 0, 2),
        max_cost=8,
    )


def test_append_effect_learns_count_vs_sum_and_transfers() -> None:
    signature = _signature()
    residual_checkpoint = _banked_residual_checkpoint(signature)
    dependency_sha = residual_checkpoint.checkpoint_sha256
    train_count = learner.CausalAppendEffectEpisode(
        _spans(30, 130),
        _literals(-5, 2, (132, 140, 135)),
        SIGNATURE_ROLE,
        _outputs(2, 3, 2),
    )
    train_sum = learner.CausalAppendEffectEpisode(
        _spans(90, 150),
        _literals(-1, -3, (155, 152, 158)),
        SIGNATURE_ROLE,
        _outputs(-2, 3, -10),
    )
    checkpoint = learner.fit_causal_append_effect_reducer_v1(
        residual_checkpoint,
        (train_count, train_sum),
        signature=signature,
        probe_cases=TRAIN_PROBE,
        literals=(-15, -5, -3, -1, 0, 2),
        max_cost=8,
    )
    summary = checkpoint.training_summary
    assert residual_checkpoint.checkpoint_sha256 == dependency_sha
    assert checkpoint.residual_checkpoint_sha256 == dependency_sha
    assert summary.episode_count == 2
    assert summary.packed_build_count == 1
    assert summary.packed_authorization_count == 4
    assert summary.effect_race_count == summary.packed_outcome_race_count == 2
    assert summary.upstream_binding_cell_steps == 2 * 1831 == 3662
    assert summary.probe_comparison_count == 8
    assert summary.probe_comparison_cell_steps == 8 * 271 == 2168
    assert summary.completeness_cell_steps == 2
    assert summary.excitatory_update_ticks == summary.inhibitory_update_ticks == 2
    assert summary.maximum_excitatory_tick == summary.maximum_inhibitory_tick == 1
    assert summary.trainable_state_bytes == learner.TRAINABLE_STATE_BYTES == 8
    assert checkpoint.learned_state_shapes == ((2, 2), (2, 2))
    assert {audit.effect_relation for audit in summary.intervention_audits} == {0, 1}
    assert {audit.pass_action for audit in summary.intervention_audits} == {0, 1}

    count_spans = tuple(reversed(_spans(410, 180)))
    count_literals = tuple(reversed(_literals(-7, 3, (182, 190, 185))))
    sum_spans = _spans(470, 210)
    sum_literals = _literals(-11, -6, (215, 212, 218))
    count_certificate = checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
    )
    sum_certificate = checkpoint.bind(
        sum_spans,
        sum_literals,
        SIGNATURE_ROLE,
        SUM_PROBE,
        _outputs(2, -8, 10),
    )
    count_binding = learner.require_append_effect_reducer_binding(count_certificate)
    sum_binding = learner.require_append_effect_reducer_binding(sum_certificate)
    assert count_binding.effect_relation == learner.PREDICATE_EFFECT_RELATION
    assert sum_binding.effect_relation == learner.PAYLOAD_EFFECT_RELATION
    assert count_binding.selected_action != sum_binding.selected_action
    assert count_binding.inserted_values == (6, -12)
    assert sum_binding.inserted_values == (-10, 8)
    assert count_binding.predicate_spikes == sum_binding.predicate_spikes == (1, 0)
    assert count_certificate.work.upstream_binding_cell_steps == 1831
    assert count_certificate.work.probe_comparison_count == 4
    assert count_certificate.work.probe_comparison_cell_steps == 1084
    assert count_certificate.work.completeness_cell_steps == 1
    assert count_certificate.work.effect_race_cell_steps == 1800
    assert count_certificate.work.reducer_action_race_cell_steps == 60
    assert count_certificate.work.stacked_binding_cell_steps == 4776

    flipped = learner.require_append_effect_reducer_binding(
        checkpoint.bind(
            count_spans,
            count_literals,
            SIGNATURE_ROLE,
            COUNT_PROBE,
            _outputs(3, 9, -9),
        )
    )
    assert flipped.effect_relation == learner.PAYLOAD_EFFECT_RELATION
    assert flipped.selected_action == sum_binding.selected_action
    assert flipped.probe_sha256 != count_binding.probe_sha256

    assert checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        (([3, 0],), ([3, 0, 1],), ([3, 0, -12],)),
        _outputs(2, 3, 2),
    ).status == learner.STOP_NONDIAGNOSTIC
    assert checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        (([3, 0],), ([3, 0, 6],), ([3, 0, 5],)),
        _outputs(2, 3, 3),
    ).status == learner.STOP_PREDICATE_STRADDLE
    assert checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 4, 2),
    ).status == learner.STOP_EFFECT_ZERO
    assert checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        equality_gate=False,
    ).status == learner.STOP_DEPENDENCY
    assert checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        predicate_gate=False,
    ).status == learner.STOP_PREDICATE_GATE
    assert checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        completeness_gate=False,
    ).status == learner.STOP_COMPLETENESS
    assert checkpoint.bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        effect_gate=False,
    ).status == learner.STOP_EFFECT_GATE
    assert learner.freeze_outcome_off_control(checkpoint).bind(
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
    ).status == learner.STOP_ZERO

    first = summary.intervention_audits[0]
    with pytest.raises(learner.AppendEffectFitStop) as conflict:
        learner._learn_policy(
            (first, first._replace(pass_action=1 - first.pass_action)),
            outcome_gate=True,
        )
    assert conflict.value.status == learner.STOP_CONFLICT
    with pytest.raises(learner.AppendEffectFitStop) as alias:
        learner._require_distinct_reducer_outputs(((), ()))
    assert alias.value.status == learner.STOP_ACTION_ALIAS

    overlap = frontier_v1.PackedAppendEffectReducerFrontierV1.build(
        checkpoint,
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=(([3, 0],),),
        literals=(-7, 0, 3),
        max_cost=8,
    )
    assert overlap.status == learner.STOP_PROBE_FINAL_OVERLAP
    assert overlap.state.packed_build_count == 0

    assert "desired" not in inspect.signature(
        frontier_v1.PackedAppendEffectReducerFrontierV1.build
    ).parameters
    count_frontier = frontier_v1.PackedAppendEffectReducerFrontierV1.build(
        checkpoint,
        count_spans,
        count_literals,
        SIGNATURE_ROLE,
        COUNT_PROBE,
        _outputs(2, 3, 2),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-7, 0, 3),
        max_cost=8,
    )
    assert count_frontier.status == frontier_v1.FRONTIER_READY
    assert count_frontier.state.packed_build_count == 1
    assert count_frontier.state.desired_reads_during_build == 0
    count_selection = count_frontier.select(
        _desired(-7, 3, ordered.REPEATED_TO_OPERAND_ZERO)
    )
    assert count_selection.status == frontier_v1.SELECTED
    assert count_selection.delegated is not None
    assert count_selection.delegated.exact_proof_count == 1
    assert count_selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if -7 < item_0 + 3))\n"
    )

    sum_frontier = frontier_v1.PackedAppendEffectReducerFrontierV1.build(
        checkpoint,
        sum_spans,
        sum_literals,
        SIGNATURE_ROLE,
        SUM_PROBE,
        _outputs(2, -8, 10),
        signature=signature,
        cases=FINAL_CASES,
        literals=(-11, -6, 0),
        max_cost=8,
    )
    assert sum_frontier.status == frontier_v1.FRONTIER_READY
    sum_selection = sum_frontier.select(tuple(sum(values) for (values,) in FINAL_CASES))
    assert sum_selection.status == frontier_v1.SELECTED
    assert sum_selection.delegated is not None
    assert sum_selection.delegated.exact_proof_count == 1
    assert sum_selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum(numbers)\n"
    )

    source = inspect.getsource(learner) + inspect.getsource(frontier_v1)
    assert all(
        forbidden not in source
        for forbidden in (
            "public_train_fit_prompt_record_source_event_producer_v1",
            "FIT_SOURCE_BUNDLES",
            "protected_panel",
        )
    )
    assert learner.TOTAL_CELL_COUNT == residual.TOTAL_CELL_COUNT + 5 == 3449
    assert learner.TRAINABLE_STATE_BYTES == 8
    assert frontier_v1.PUBLIC_TRAIN_INPUT_COUNT == 0
    assert frontier_v1.CANONICAL_PANEL_SPEND_COUNT == 0
    assert frontier_v1.DEV_INPUT_COUNT == frontier_v1.M0_INPUT_COUNT == 0
    assert frontier_v1.C1_INPUT_COUNT == frontier_v1.G0_INPUT_COUNT == 0
    assert frontier_v1.PROTECTED_INPUT_COUNT == 0
