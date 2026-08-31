from __future__ import annotations

import inspect

import pytest
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as learner
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_relative_ordered_operand_frontier_v1 as frontier_v1,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([],),
    ([0, 1, 2, 3],),
    ([1, 2, 10],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
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
        value,
        declared_type=output_cells.OutputType.INT,
    ).active_global_cell_indices()


def _literals(
    repeated: int,
    singleton: int,
    positions: tuple[int, int, int],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": position,
            "typed_output_active_cells": _cells(value),
        }
        for position, value in zip(
            positions,
            (repeated, singleton, repeated),
            strict=True,
        )
    )


def _double_repeated(
    left: int,
    right: int,
    positions: tuple[int, int, int, int],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": position,
            "typed_output_active_cells": _cells(value),
        }
        for position, value in zip(positions, (left, right, left, right), strict=True)
    )


def _desired(hole: int, action: int) -> tuple[int, ...]:
    if action == learner.REPEATED_TO_OPERAND_ZERO:
        return tuple(
            sum(hole < item + -15 for item in values)
            for (values,) in CASES
        )
    return tuple(
        sum(item + -15 < hole for item in values)
        for (values,) in CASES
    )


def _banked_literal_role_checkpoint(
    signature: TypedSignature,
) -> literal_role.FrozenCausalRelativeLiteralRoleV1:
    episodes = (
        literal_role.CausalLiteralEpisode(
            _spans(10, 0),
            _literals(-14, -8, (2, 4, 7)),
            SIGNATURE_ROLE,
            _desired(-14, learner.REPEATED_TO_OPERAND_ZERO),
        ),
        literal_role.CausalLiteralEpisode(
            _spans(70, 10),
            _literals(-12, -6, (13, 15, 19)),
            SIGNATURE_ROLE,
            _desired(-12, learner.REPEATED_TO_OPERAND_ZERO),
        ),
    )
    return literal_role.fit_causal_relative_literal_role_v1(
        episodes,
        signature=signature,
        cases=CASES,
        literals=(-15, -14, -12, -8, -6, 0),
        max_cost=8,
    )


class _Explodes:
    def __iter__(self):
        raise AssertionError("a stopped binding iterated packed or desired evidence")


def test_causal_relative_ordered_operand_transfers_unseen_values_end_to_end() -> None:
    signature = _signature()
    literal_checkpoint = _banked_literal_role_checkpoint(signature)
    dependency_sha = literal_checkpoint.checkpoint_sha256
    literal_checkpoint.authenticate()

    train_rrs = learner.CausalOrderedOperandEpisode(
        _spans(150, 30),
        _literals(-9, 5, (32, 39, 34)),
        SIGNATURE_ROLE,
        _desired(-9, learner.REPEATED_TO_OPERAND_ZERO),
    )
    train_srr = learner.CausalOrderedOperandEpisode(
        _spans(220, 50),
        _literals(-7, 13, (55, 52, 58)),
        SIGNATURE_ROLE,
        _desired(-7, learner.REPEATED_TO_OPERAND_ONE),
    )
    checkpoint = learner.fit_causal_relative_ordered_operand_v1(
        literal_checkpoint,
        (train_rrs, train_srr),
        signature=signature,
        cases=CASES,
        literals=(-15, -9, -7, 0),
        max_cost=8,
    )
    summary = checkpoint.training_summary
    assert literal_checkpoint.checkpoint_sha256 == dependency_sha
    assert checkpoint.literal_role_checkpoint_sha256 == dependency_sha
    assert summary.episode_count == 2
    assert summary.packed_build_count == 1
    assert summary.packed_authorization_count == 4
    assert summary.packed_outcome_race_count == 2
    assert summary.order_event_count == 6
    assert summary.order_event_cell_steps == 2 * learner.ORDER_EVENT_CELL_STEPS == 48
    assert summary.action_race_cell_steps == 0
    assert summary.excitatory_update_ticks == 2
    assert summary.inhibitory_update_ticks == 2
    assert summary.maximum_excitatory_tick == 1
    assert summary.maximum_inhibitory_tick == 1
    assert summary.trainable_state_bytes == learner.TRAINABLE_STATE_BYTES == 8
    assert checkpoint.learned_state_shapes == ((2, 2), (2, 2))
    assert {audit.relation for audit in summary.intervention_audits} == {0, 1}
    assert {audit.pass_action for audit in summary.intervention_audits} == {0, 1}
    assert all(audit.pass_outcome_count == 1 for audit in summary.intervention_audits)

    recall_rrs_spans = _spans(300, 100)
    recall_rrs_literals = _literals(-13, 21, (102, 110, 105))
    recall_srr_spans = _spans(370, 120)
    recall_srr_literals = _literals(-11, 23, (125, 120, 128))
    rrs_certificate = checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
    )
    srr_certificate = checkpoint.bind(
        recall_srr_spans,
        recall_srr_literals,
        SIGNATURE_ROLE,
    )
    rrs = learner.require_relative_ordered_operand_binding(rrs_certificate)
    srr = learner.require_relative_ordered_operand_binding(srr_certificate)
    assert literal_role.decode_int_payload(rrs.repeated_payload_cells) == -13
    assert literal_role.decode_int_payload(srr.repeated_payload_cells) == -11
    assert rrs.relation == learner.REPEATED_BEFORE_SINGLETON
    assert rrs.selected_action == learner.REPEATED_TO_OPERAND_ZERO
    assert srr.relation == learner.SINGLETON_BEFORE_REPEATED
    assert srr.selected_action == learner.REPEATED_TO_OPERAND_ONE
    assert rrs_certificate.work.action_race_cell_steps == learner.ACTION_RACE_CELL_STEPS
    assert tuple(reversed(recall_rrs_literals)) != recall_rrs_literals
    permuted = checkpoint.bind(
        tuple(reversed(recall_rrs_spans)),
        tuple(reversed(recall_rrs_literals)),
        SIGNATURE_ROLE,
    )
    assert permuted.sha256 == rrs_certificate.sha256

    same_payload_flip = learner.require_relative_ordered_operand_binding(
        checkpoint.bind(
            _spans(440, 140),
            _literals(-13, 25, (145, 140, 148)),
            SIGNATURE_ROLE,
        )
    )
    assert same_payload_flip.selected_action == learner.REPEATED_TO_OPERAND_ONE
    assert (
        learner.freeze_ordered_operand_scaffold(-13, rrs.selected_action).sha256
        != learner.freeze_ordered_operand_scaffold(
            -13,
            same_payload_flip.selected_action,
        ).sha256
    )
    opaque = learner.require_relative_ordered_operand_binding(
        checkpoint.bind(
            _spans(100, 160),
            _literals(17, 31, (162, 170, 165)),
            SIGNATURE_ROLE,
        )
    )
    assert literal_role.decode_int_payload(opaque.repeated_payload_cells) == 17
    assert opaque.selected_action == learner.REPEATED_TO_OPERAND_ZERO
    assert literal_checkpoint.checkpoint_sha256 == dependency_sha

    assert checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        order_gate=False,
    ).status == learner.STOP_ORDER_GATE
    assert checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        equality_gate=False,
    ).status == learner.STOP_DEPENDENCY
    assert checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        literal_role_gate=False,
    ).status == learner.STOP_DEPENDENCY
    assert checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        cell_gate=False,
    ).status == learner.STOP_ZERO
    assert learner.freeze_outcome_off_control(checkpoint).bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
    ).status == learner.STOP_ZERO
    assert checkpoint.bind(
        _spans(110, 180),
        _literals(-13, 33, (182, 182, 185)),
        SIGNATURE_ROLE,
    ).status == learner.STOP_TIE
    straddled_spans = _spans(120, 200)
    straddled_literals = _literals(-13, 35, (202, 205, 208))
    assert checkpoint.bind(
        straddled_spans,
        straddled_literals,
        SIGNATURE_ROLE,
    ).status == learner.STOP_STRADDLED
    assert checkpoint.bind(
        _spans(130, 220),
        _double_repeated(-13, -11, (222, 225, 228, 231)),
        SIGNATURE_ROLE,
    ).status == learner.STOP_DEPENDENCY

    with pytest.raises(learner.OrderedOperandFitStop) as incomplete:
        learner.fit_causal_relative_ordered_operand_v1(
            literal_checkpoint,
            (
                train_rrs,
                learner.CausalOrderedOperandEpisode(
                    train_srr.span_records,
                    _literals(-7, 13, (52, 58, 55)),
                    SIGNATURE_ROLE,
                    _desired(-7, learner.REPEATED_TO_OPERAND_ZERO),
                ),
            ),
            signature=signature,
            cases=CASES,
            literals=(-15, -9, -7, 0),
            max_cost=8,
        )
    assert incomplete.value.status == learner.STOP_INCOMPLETE
    first_audit = summary.intervention_audits[0]
    with pytest.raises(learner.OrderedOperandFitStop) as conflict:
        learner._learn_policy(
            (
                first_audit,
                first_audit._replace(pass_action=1 - first_audit.pass_action),
                summary.intervention_audits[1],
            ),
            outcome_gate=True,
        )
    assert conflict.value.status == learner.STOP_CONFLICT
    with pytest.raises(learner.OrderedOperandFitStop) as alias:
        learner._require_distinct_action_outputs(((), ()))
    assert alias.value.status == learner.STOP_ALIAS

    stopped = frontier_v1.PackedRelativeOrderedOperandFrontierV1.build(
        checkpoint,
        straddled_spans,
        straddled_literals,
        SIGNATURE_ROLE,
        signature=signature,
        cases=_Explodes(),
        literals=_Explodes(),
        max_cost=8,
    )
    assert stopped.status == learner.STOP_STRADDLED
    assert stopped.state.packed_build_count == 0
    assert stopped.select(_Explodes()).status == learner.STOP_STRADDLED

    assert "desired" not in inspect.signature(
        frontier_v1.PackedRelativeOrderedOperandFrontierV1.build
    ).parameters
    frontier = frontier_v1.PackedRelativeOrderedOperandFrontierV1.build(
        checkpoint,
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        signature=signature,
        cases=CASES,
        literals=(-15, -13, 0),
        max_cost=8,
    )
    assert frontier.status == frontier_v1.FRONTIER_READY
    assert frontier.literal_hole_value == -13
    assert frontier.selected_action == learner.REPEATED_TO_OPERAND_ZERO
    assert frontier.state.repeated_token_indices == (102, 105)
    assert frontier.state.singleton_token_index == 110
    assert frontier.state.packed_build_count == 1
    assert frontier.state.desired_reads_during_build == 0
    selection = frontier.select(_desired(-13, learner.REPEATED_TO_OPERAND_ZERO))
    assert selection.status == frontier_v1.SELECTED
    assert selection.delegated is not None
    assert selection.delegated.exact_proof_count == 1
    assert len(selection.delegated.witnesses) == 1
    assert selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if -13 < item_0 + -15))\n"
    )

    source = inspect.getsource(learner) + inspect.getsource(frontier_v1)
    assert "fit_causal_relative_literal_role_v1(" not in inspect.getsource(learner)
    assert all(
        forbidden not in source
        for forbidden in (
            "public_train_fit_prompt_record_source_event_producer_v1",
            "public_train_fit_occurrence_factor_static_checkpoint_capsule_v1",
            "typed_semantic_packed_static_occurrence_factor_authorized_plan_slice_v1",
            "StructuralOccurrenceCertificate",
            "FIT_SOURCE_BUNDLES",
        )
    )
    assert learner.TOTAL_CELL_COUNT == literal_role.TOTAL_CELL_COUNT + 14 == 3441
    assert frontier_v1.PUBLIC_TRAIN_INPUT_COUNT == 0
    assert frontier_v1.CANONICAL_PANEL_SPEND_COUNT == 0
    assert frontier_v1.DEV_INPUT_COUNT == frontier_v1.M0_INPUT_COUNT == 0
    assert frontier_v1.C1_INPUT_COUNT == frontier_v1.G0_INPUT_COUNT == 0
    assert frontier_v1.PROTECTED_INPUT_COUNT == 0
    assert frontier_v1.OLD_OCCURRENCE_CERTIFICATE_IMPORT_COUNT == 0
