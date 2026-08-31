from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as learner
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_relative_literal_hole_frontier_v1 as frontier_v1,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([],),
    ([0, 1, 2, 3],),
    ([1, 2, 10],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
TRAIN_LITERALS = (
    -15,
    -14,
    -12,
    -8,
    -6,
    0,
)
RECALL_LITERALS = (-15, -13, -11, 0)
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
    unique: int,
    *,
    positions: tuple[int, int, int],
) -> tuple[dict[str, object], ...]:
    return (
        {
            "canonical_token_index": positions[0],
            "typed_output_active_cells": _cells(repeated),
        },
        {
            "canonical_token_index": positions[1],
            "typed_output_active_cells": _cells(unique),
        },
        {
            "canonical_token_index": positions[2],
            "typed_output_active_cells": _cells(repeated),
        },
    )


def _double_repeated(
    left: int,
    right: int,
    *,
    positions: tuple[int, int, int, int],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": position,
            "typed_output_active_cells": _cells(value),
        }
        for position, value in zip(
            positions,
            (left, right, left, right),
            strict=True,
        )
    )


def _desired(hole_value: int) -> tuple[int, ...]:
    return tuple(
        sum(1 for item in values if hole_value < item + -15)
        for (values,) in CASES
    )


class _Explodes:
    def __iter__(self):
        raise AssertionError("a stopped relative binding iterated packed evidence")


def test_causal_relative_literal_role_transfers_unseen_payload_end_to_end() -> None:
    signature = _signature()
    train_a_spans = _spans(10, 0)
    train_b_spans = _spans(70, 10)
    episodes = (
        learner.CausalLiteralEpisode(
            train_a_spans,
            _literals(-14, -8, positions=(2, 4, 7)),
            SIGNATURE_ROLE,
            _desired(-14),
        ),
        learner.CausalLiteralEpisode(
            train_b_spans,
            _literals(-12, -6, positions=(13, 15, 19)),
            SIGNATURE_ROLE,
            _desired(-12),
        ),
    )
    checkpoint = learner.fit_causal_relative_literal_role_v1(
        episodes,
        signature=signature,
        cases=CASES,
        literals=TRAIN_LITERALS,
        max_cost=8,
    )
    summary = checkpoint.training_summary
    assert summary.episode_count == 2
    assert summary.packed_build_count == 1
    assert summary.packed_authorization_count == 4
    assert summary.packed_outcome_race_count == 2
    assert summary.literal_occurrence_count == 6
    assert summary.equality_comparison_count == 12
    assert summary.equality_physical_cell_steps == 12 * (
        2 * learner.COMPARATOR_CELL_COUNT
        + learner.PHASE_GATE_CELL_COUNT
        + learner.FULL_MATCH_CELL_COUNT
    )
    assert summary.distinct_positive_payload_count == 2
    assert summary.distinct_positive_position_count == 2
    assert summary.distinct_prompt_extent_count == 2
    assert summary.outcome_gate_on
    assert summary.excitatory_update_ticks == summary.inhibitory_update_ticks == 2
    assert summary.maximum_excitatory_tick == summary.maximum_inhibitory_tick == 2
    assert summary.trainable_state_bytes == learner.TRAINABLE_STATE_BYTES == 4
    assert checkpoint.learned_state_shapes == ((2,), (2,))
    assert checkpoint.__slots__ == (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_summary",
    )
    assert all(audit.pass_outcome_count == 1 for audit in summary.intervention_audits)
    assert all(
        audit.pass_relation_cell
        == learner.RELATION_CELL_OFFSET + learner.REPEATED_RELATION
        for audit in summary.intervention_audits
    )

    heldout_spans = _spans(150, 30)
    heldout_literals = _literals(-13, -11, positions=(32, 35, 39))
    all_train_assemblies = {
        tuple(row["assembly_cells"])
        for spans in (train_a_spans, train_b_spans)
        for row in spans
    }
    assert all(
        tuple(row["assembly_cells"]) not in all_train_assemblies
        for row in heldout_spans
    )
    binding_certificate = checkpoint.bind(
        heldout_spans,
        heldout_literals,
        SIGNATURE_ROLE,
    )
    binding = learner.require_relative_literal_binding(binding_certificate)
    assert binding_certificate.status == learner.READY
    assert learner.decode_int_payload(binding.payload_cells) == -13
    assert binding.member_token_indices == (32, 39)

    permuted = checkpoint.bind(
        tuple(reversed(heldout_spans)),
        tuple(reversed(heldout_literals)),
        SIGNATURE_ROLE,
    )
    assert permuted.status == learner.READY
    assert permuted.sha256 == binding_certificate.sha256

    swapped = checkpoint.bind(
        _spans(220, 45),
        _literals(-11, -13, positions=(47, 50, 54)),
        SIGNATURE_ROLE,
    )
    swapped_binding = learner.require_relative_literal_binding(swapped)
    assert learner.decode_int_payload(swapped_binding.payload_cells) == -11
    assert _desired(-11) != _desired(-13)
    assert (
        learner.freeze_literal_hole_scaffold(-11).sha256
        != learner.freeze_literal_hole_scaffold(-13).sha256
    )

    relabeled = checkpoint.bind(
        _spans(280, 60),
        _literals(5, 13, positions=(62, 65, 69)),
        SIGNATURE_ROLE,
    )
    assert learner.decode_int_payload(
        learner.require_relative_literal_binding(relabeled).payload_cells
    ) == 5

    assert (
        checkpoint.bind(
            heldout_spans,
            heldout_literals,
            SIGNATURE_ROLE,
            equality_gate=False,
        ).status
        == learner.STOP_EQUALITY_OFF
    )
    assert (
        checkpoint.bind(
            heldout_spans,
            heldout_literals,
            SIGNATURE_ROLE,
            cell_gate=False,
        ).status
        == learner.STOP_ZERO
    )
    outcome_off = learner.freeze_outcome_off_control(checkpoint)
    assert not outcome_off.training_summary.outcome_gate_on
    assert outcome_off.training_summary.excitatory_update_ticks == 0
    assert outcome_off.training_summary.inhibitory_update_ticks == 0
    assert (
        outcome_off.bind(
            heldout_spans,
            heldout_literals,
            SIGNATURE_ROLE,
        ).status
        == learner.STOP_ZERO
    )

    ambiguous_literals = _double_repeated(
        -13,
        -11,
        positions=(72, 75, 79, 82),
    )
    ambiguous = checkpoint.bind(
        _spans(340, 70),
        ambiguous_literals,
        SIGNATURE_ROLE,
    )
    assert ambiguous.status == learner.STOP_AMBIGUOUS
    with pytest.raises(TypeError):
        learner.require_relative_literal_binding(object())  # type: ignore[arg-type]

    stopped_frontier = frontier_v1.PackedRelativeLiteralHoleFrontierV1.build(
        checkpoint,
        _spans(340, 70),
        ambiguous_literals,
        SIGNATURE_ROLE,
        signature=signature,
        cases=_Explodes(),
        literals=_Explodes(),
        max_cost=8,
    )
    assert stopped_frontier.status == learner.STOP_AMBIGUOUS
    assert stopped_frontier.state.packed_build_count == 0
    assert stopped_frontier.select(_Explodes()).status == learner.STOP_AMBIGUOUS

    assert "desired" not in inspect.signature(
        frontier_v1.PackedRelativeLiteralHoleFrontierV1.build
    ).parameters
    frontier = frontier_v1.PackedRelativeLiteralHoleFrontierV1.build(
        checkpoint,
        heldout_spans,
        heldout_literals,
        SIGNATURE_ROLE,
        signature=signature,
        cases=CASES,
        literals=RECALL_LITERALS,
        max_cost=8,
    )
    assert frontier.status == frontier_v1.FRONTIER_READY
    assert frontier.literal_hole_value == -13
    assert frontier.state.selected_member_token_indices == (32, 39)
    assert frontier.state.packed_build_count == 1
    assert frontier.state.desired_reads_during_build == 0
    selection = frontier.select(_desired(-13))
    assert selection.status == frontier_v1.SELECTED
    assert selection.delegated is not None
    assert selection.delegated.exact_proof_count == 1
    assert len(selection.delegated.witnesses) == 1
    assert selection.delegated.witnesses[0].source == (
        "def aggregate(numbers):\n"
        "    return sum((1 for item_0 in numbers if -13 < item_0 + -15))\n"
    )

    forbidden_runtime_names = (
        "public_train_fit_prompt_record_source_event_producer_v1",
        "public_train_fit_occurrence_factor_static_checkpoint_capsule_v1",
        "typed_semantic_packed_static_occurrence_factor_authorized_plan_slice_v1",
        "StructuralOccurrenceCertificate",
        "FIT_SOURCE_BUNDLES",
    )
    for module in (learner, frontier_v1):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert all(name not in source for name in forbidden_runtime_names)
    assert learner.PUBLIC_TRAIN_INPUT_COUNT == 0
    assert learner.CANONICAL_PANEL_SPEND_COUNT == 0
    assert learner.DEV_INPUT_COUNT == learner.M0_INPUT_COUNT == learner.C1_INPUT_COUNT == 0
    assert learner.PROTECTED_INPUT_COUNT == 0
    assert frontier_v1.OLD_OCCURRENCE_CERTIFICATE_IMPORT_COUNT == 0
