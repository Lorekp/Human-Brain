from __future__ import annotations

import inspect

import pytest
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as ordered
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as learner
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_residual_literal_route_frontier_v1 as frontier_v1,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([],),
    ([-20, -10, 0, 10],),
    ([-5, 1, 9],),
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
    residual: int,
    positions: tuple[int, int, int],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": position,
            "typed_output_active_cells": _cells(value),
        }
        for position, value in zip(
            positions,
            (repeated, residual, repeated),
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


def _desired(hole: int, offset: int, operand_action: int) -> tuple[int, ...]:
    if operand_action == ordered.REPEATED_TO_OPERAND_ZERO:
        return tuple(
            sum(hole < item + offset for item in values)
            for (values,) in CASES
        )
    return tuple(
        sum(item + offset < hole for item in values)
        for (values,) in CASES
    )


def _banked_ordered_checkpoint(
    signature: TypedSignature,
) -> ordered.FrozenCausalRelativeOrderedOperandV1:
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
        cases=CASES,
        literals=(-15, -14, -12, -8, -6, 0),
        max_cost=8,
    )
    return ordered.fit_causal_relative_ordered_operand_v1(
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
        cases=CASES,
        literals=(-15, -9, -7, 0),
        max_cost=8,
    )


class _Explodes:
    def __iter__(self):
        raise AssertionError("a stopped residual route iterated packed evidence")


def test_residual_literal_route_transfers_unseen_values_end_to_end() -> None:
    signature = _signature()
    ordered_checkpoint = _banked_ordered_checkpoint(signature)
    dependency_sha = ordered_checkpoint.checkpoint_sha256
    train_rrs = learner.CausalResidualLiteralEpisode(
        _spans(290, 80),
        _literals(-5, 2, (82, 90, 85)),
        SIGNATURE_ROLE,
        _desired(-5, 2, ordered.REPEATED_TO_OPERAND_ZERO),
    )
    train_srr = learner.CausalResidualLiteralEpisode(
        _spans(350, 100),
        _literals(-1, -3, (105, 102, 108)),
        SIGNATURE_ROLE,
        _desired(-1, -3, ordered.REPEATED_TO_OPERAND_ONE),
    )
    checkpoint = learner.fit_causal_residual_literal_route_v1(
        ordered_checkpoint,
        (train_rrs, train_srr),
        signature=signature,
        cases=CASES,
        literals=(-15, -5, -3, -1, 0, 2),
        max_cost=8,
    )
    summary = checkpoint.training_summary
    assert ordered_checkpoint.checkpoint_sha256 == dependency_sha
    assert checkpoint.ordered_checkpoint_sha256 == dependency_sha
    assert summary.episode_count == 2
    assert summary.packed_build_count == 1
    assert summary.packed_authorization_count == 4
    assert summary.packed_outcome_race_count == 2
    assert summary.upstream_binding_cell_steps == 2 * 1770 == 3540
    assert summary.route_phase_cell_steps == 2
    assert summary.route_action_race_cell_steps == 0
    assert summary.excitatory_update_ticks == 2
    assert summary.inhibitory_update_ticks == 2
    assert summary.maximum_excitatory_tick == 2
    assert summary.maximum_inhibitory_tick == 2
    assert summary.trainable_state_bytes == learner.TRAINABLE_STATE_BYTES == 4
    assert checkpoint.learned_state_shapes == ((2,), (2,))
    assert {audit.relation for audit in summary.intervention_audits} == {0, 1}
    assert {
        audit.pass_route for audit in summary.intervention_audits
    } == {learner.RESIDUAL_TO_LITERAL_HOLE}

    recall_rrs_spans = _spans(410, 120)
    recall_rrs_literals = _literals(-13, 4, (122, 130, 125))
    recall_srr_spans = _spans(100, 140)
    recall_srr_literals = _literals(-11, -6, (145, 142, 148))
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
    rrs = learner.require_residual_literal_route_binding(rrs_certificate)
    srr = learner.require_residual_literal_route_binding(srr_certificate)
    assert literal_role.decode_int_payload(rrs.repeated_payload_cells) == -13
    assert literal_role.decode_int_payload(rrs.residual_payload_cells) == 4
    assert literal_role.decode_int_payload(srr.repeated_payload_cells) == -11
    assert literal_role.decode_int_payload(srr.residual_payload_cells) == -6
    assert rrs.operand_action == ordered.REPEATED_TO_OPERAND_ZERO
    assert srr.operand_action == ordered.REPEATED_TO_OPERAND_ONE
    assert rrs.selected_route == srr.selected_route == learner.RESIDUAL_TO_LITERAL_HOLE
    assert rrs_certificate.work.upstream_binding_cell_steps == 1770
    assert rrs_certificate.work.route_phase_cell_steps == 1
    assert rrs_certificate.work.route_action_race_cell_steps == 60
    assert rrs_certificate.work.stacked_binding_cell_steps == 1831

    permuted = checkpoint.bind(
        tuple(reversed(recall_rrs_spans)),
        tuple(reversed(recall_rrs_literals)),
        SIGNATURE_ROLE,
    )
    assert permuted.sha256 == rrs_certificate.sha256
    same_h_different_d = learner.require_residual_literal_route_binding(
        checkpoint.bind(
            _spans(120, 160),
            _literals(-13, -6, (162, 170, 165)),
            SIGNATURE_ROLE,
        )
    )
    assert literal_role.decode_int_payload(
        same_h_different_d.residual_payload_cells
    ) == -6
    assert _desired(-13, 4, ordered.REPEATED_TO_OPERAND_ZERO) != _desired(
        -13,
        -6,
        ordered.REPEATED_TO_OPERAND_ZERO,
    )
    assert (
        learner.freeze_residual_literal_route_scaffold(
            -13,
            4,
            ordered.REPEATED_TO_OPERAND_ZERO,
            learner.RESIDUAL_TO_LITERAL_HOLE,
        ).sha256
        != learner.freeze_residual_literal_route_scaffold(
            -13,
            -6,
            ordered.REPEATED_TO_OPERAND_ZERO,
            learner.RESIDUAL_TO_LITERAL_HOLE,
        ).sha256
    )
    opaque = learner.require_residual_literal_route_binding(
        checkpoint.bind(
            _spans(130, 180),
            _literals(17, 31, (182, 190, 185)),
            SIGNATURE_ROLE,
        )
    )
    assert literal_role.decode_int_payload(opaque.repeated_payload_cells) == 17
    assert literal_role.decode_int_payload(opaque.residual_payload_cells) == 31
    assert opaque.selected_route == learner.RESIDUAL_TO_LITERAL_HOLE
    assert ordered_checkpoint.checkpoint_sha256 == dependency_sha

    assert checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        residual_gate=False,
    ).status == learner.STOP_RESIDUAL_GATE
    assert checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        cell_gate=False,
    ).status == learner.STOP_ZERO
    assert checkpoint.bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        order_gate=False,
    ).status == learner.STOP_DEPENDENCY
    assert learner.freeze_outcome_off_control(checkpoint).bind(
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
    ).status == learner.STOP_ZERO
    assert checkpoint.bind(
        _spans(140, 200),
        _double_repeated(-13, -11, (202, 205, 208, 211)),
        SIGNATURE_ROLE,
    ).status == learner.STOP_DEPENDENCY

    with pytest.raises(learner.ResidualLiteralFitStop) as incomplete:
        learner.fit_causal_residual_literal_route_v1(
            ordered_checkpoint,
            (
                train_rrs,
                learner.CausalResidualLiteralEpisode(
                    train_srr.span_records,
                    _literals(-1, -3, (102, 108, 105)),
                    SIGNATURE_ROLE,
                    _desired(-1, -3, ordered.REPEATED_TO_OPERAND_ZERO),
                ),
            ),
            signature=signature,
            cases=CASES,
            literals=(-15, -5, -3, -1, 0, 2),
            max_cost=8,
        )
    assert incomplete.value.status == learner.STOP_INCOMPLETE
    first_audit = summary.intervention_audits[0]
    with pytest.raises(learner.ResidualLiteralFitStop) as conflict:
        learner._learn_policy(
            (
                first_audit,
                first_audit._replace(pass_route=learner.KEEP_DEFAULT_LITERAL),
            ),
            outcome_gate=True,
        )
    assert conflict.value.status == learner.STOP_CONFLICT
    with pytest.raises(learner.ResidualLiteralFitStop) as alias:
        learner._require_distinct_route_outputs(((), ()))
    assert alias.value.status == learner.STOP_ALIAS
    with pytest.raises(TypeError):
        learner.require_residual_literal_route_binding(object())  # type: ignore[arg-type]

    stopped = frontier_v1.PackedResidualLiteralRouteFrontierV1.build(
        checkpoint,
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        signature=signature,
        cases=_Explodes(),
        literals=_Explodes(),
        max_cost=8,
        residual_gate=False,
    )
    assert stopped.status == learner.STOP_RESIDUAL_GATE
    assert stopped.state.packed_build_count == 0
    assert stopped.select(_Explodes()).status == learner.STOP_RESIDUAL_GATE

    assert "desired" not in inspect.signature(
        frontier_v1.PackedResidualLiteralRouteFrontierV1.build
    ).parameters
    frontier = frontier_v1.PackedResidualLiteralRouteFrontierV1.build(
        checkpoint,
        recall_rrs_spans,
        recall_rrs_literals,
        SIGNATURE_ROLE,
        signature=signature,
        cases=CASES,
        literals=(-13, 0, 4),
        max_cost=8,
    )
    assert frontier.status == frontier_v1.FRONTIER_READY
    assert frontier.state.repeated_value == -13
    assert frontier.state.residual_value == 4
    assert frontier.state.effective_additive_literal == 4
    assert frontier.state.additive_literal_occurrence == 6
    assert frontier.state.additive_literal_slot == 0
    assert frontier.state.packed_build_count == 1
    assert frontier.state.desired_reads_during_build == 0
    selection = frontier.select(
        _desired(-13, 4, ordered.REPEATED_TO_OPERAND_ZERO)
    )
    assert selection.status == frontier_v1.SELECTED
    assert selection.delegated is not None
    assert selection.delegated.exact_proof_count == 1
    assert len(selection.delegated.witnesses) == 1
    assert selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if -13 < item_0 + 4))\n"
    )

    source = inspect.getsource(learner) + inspect.getsource(frontier_v1)
    assert "fit_causal_relative_ordered_operand_v1(" not in inspect.getsource(learner)
    assert all(
        forbidden not in source
        for forbidden in (
            "public_train_fit_prompt_record_source_event_producer_v1",
            "public_train_fit_occurrence_factor_static_checkpoint_capsule_v1",
            "StructuralOccurrenceCertificate",
            "FIT_SOURCE_BUNDLES",
        )
    )
    assert learner.TOTAL_CELL_COUNT == ordered.TOTAL_CELL_COUNT + 3 == 3444
    assert learner.ADDITIVE_LITERAL_OCCURRENCE == 6
    assert learner.ADDITIVE_LITERAL_SLOT == 0
    assert frontier_v1.PUBLIC_TRAIN_INPUT_COUNT == 0
    assert frontier_v1.CANONICAL_PANEL_SPEND_COUNT == 0
    assert frontier_v1.DEV_INPUT_COUNT == frontier_v1.M0_INPUT_COUNT == 0
    assert frontier_v1.C1_INPUT_COUNT == frontier_v1.G0_INPUT_COUNT == 0
    assert frontier_v1.PROTECTED_INPUT_COUNT == 0
    assert frontier_v1.OLD_OCCURRENCE_CERTIFICATE_IMPORT_COUNT == 0
