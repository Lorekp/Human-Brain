from __future__ import annotations

import pytest
from humanbrain.recurrent_assembly_stack import (
    AssemblyAblation,
    RecurrentAssemblyStack,
    Valence,
)

NESTED_SUCCESS_ROWS = (
    (
        (10, 20, 30, 1_001, 1_002),
        (401, 501, 1_001, 601, 1_002, 502, 402),
    ),
    (
        (10, 21, 31, 1_003, 1_004),
        (401, 511, 1_004, 611, 1_003, 512, 402),
    ),
    (
        (11, 20, 31, 1_005, 1_006),
        (411, 501, 1_006, 611, 1_005, 502, 412),
    ),
    (
        (11, 21, 30, 1_007, 1_008),
        (411, 511, 1_007, 601, 1_008, 512, 412),
    ),
)
ATOM_TRAIN_CUE = (90, 2_001)
ATOM_TRAIN_SEQUENCE = (2_001,)
ATOM_FRESH_CUE = (90, 9_003)
ATOM_FRESH_SEQUENCE = (9_003,)
FRESH_CUE = (10, 20, 31, 9_001, 9_002)
FRESH_SEQUENCE = (401, 501, 9_002, 611, 9_001, 502, 402)


def _grown_stack() -> RecurrentAssemblyStack:
    stack = RecurrentAssemblyStack()
    for cue, sequence in NESTED_SUCCESS_ROWS:
        stack.observe_success(cue, sequence)
    stack.observe_success(ATOM_TRAIN_CUE, ATOM_TRAIN_SEQUENCE)
    return stack


def test_opaque_nested_recombination_copies_fresh_afferents_through_races() -> None:
    stack = _grown_stack()
    digest = stack.model_digest

    pending = stack.respond(FRESH_CUE)

    assert pending.tokens == FRESH_SEQUENCE
    assert pending.trace.selected_stop is False
    assert pending.trace.max_depth == 2
    control_operations = tuple(
        step.operation for step in pending.trace.steps if step.phase == "control"
    )
    assert control_operations.count("push") == 2
    assert control_operations.count("pop") == 2
    assert all(step.simulated_steps > 0 for step in pending.trace.steps)
    update = pending.resolve(Valence.NEUTRAL)
    assert update.changed_edges == ()
    assert stack.model_digest == digest


def test_stack_off_loses_nested_route_but_retains_direct_atom() -> None:
    stack = _grown_stack()

    nested = stack.respond(FRESH_CUE, ablation=AssemblyAblation.STACK_OFF)
    assert nested.tokens == ()
    assert nested.trace.selected_stop is True
    assert any(
        step.phase == "frame" and step.operation == "stop"
        for step in nested.trace.steps
    )
    nested.resolve(Valence.NEUTRAL)

    atom = stack.respond(ATOM_FRESH_CUE, ablation=AssemblyAblation.STACK_OFF)
    assert atom.tokens == ATOM_FRESH_SEQUENCE
    assert atom.trace.max_depth == 0
    atom.resolve(Valence.NEUTRAL)


def test_unknown_cue_physically_selects_stop() -> None:
    stack = _grown_stack()

    pending = stack.respond((7_777, 7_778))

    assert pending.tokens == ()
    assert pending.trace.selected_stop is True
    assert pending.trace.steps[0].phase == "root"
    assert pending.trace.steps[0].operation == "stop"
    pending.resolve(Valence.NEUTRAL)


def test_pending_decision_is_exactly_once_and_blocks_state_observation() -> None:
    stack = _grown_stack()
    pending = stack.respond(FRESH_CUE)

    with pytest.raises(RuntimeError, match="resolve"):
        stack.respond(FRESH_CUE)
    with pytest.raises(RuntimeError, match="unresolved"):
        _ = stack.model_digest
    with pytest.raises(RuntimeError, match="resolve"):
        stack.observe_success((91,), (991,))

    pending.resolve(Valence.NEUTRAL)
    with pytest.raises(RuntimeError, match="already resolved"):
        pending.resolve(Valence.NEUTRAL)


def test_negative_consequence_changes_only_newest_structural_eligibility() -> None:
    stack = _grown_stack()
    prior = stack.model_digest
    pending = stack.respond(FRESH_CUE)
    selected = pending.trace.selected_structural_edges

    assert len(selected) == 3
    update = pending.resolve(Valence.NEGATIVE)

    assert update.prior_digest == prior
    assert update.changed_edges == (selected[-1],)
    assert update.model_digest != prior
    assert stack.model_digest == update.model_digest


def test_incremental_counter_learning_is_order_invariant_and_matches_batch_result() -> None:
    forward = RecurrentAssemblyStack()
    reverse = RecurrentAssemblyStack()
    for cue, sequence in NESTED_SUCCESS_ROWS:
        forward.observe_success(cue, sequence)
    for cue, sequence in reversed(NESTED_SUCCESS_ROWS):
        reverse.observe_success(cue, sequence)

    assert forward.model_digest == reverse.model_digest
    outcomes = []
    for stack in (forward, reverse):
        pending = stack.respond(FRESH_CUE)
        outcomes.append(
            (
                pending.tokens,
                pending.trace.max_depth,
                tuple(
                    step.operation
                    for step in pending.trace.steps
                    if step.phase == "control"
                ),
            )
        )
        pending.resolve(Valence.NEUTRAL)

    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == FRESH_SEQUENCE
    assert outcomes[0][1] == 2


def test_exact_cue_revision_retracts_old_support_and_can_restore_composition() -> None:
    stack = RecurrentAssemblyStack()
    for cue, sequence in NESTED_SUCCESS_ROWS:
        stack.observe_success(cue, sequence)
    original_digest = stack.model_digest
    cue, original = NESTED_SUCCESS_ROWS[0]
    contradictory = (999, *original[1:])

    stack.observe_success(cue, contradictory)

    assert stack.last_coactivity_write_count == 94
    blocked = stack.respond(FRESH_CUE)
    assert blocked.tokens == ()
    assert blocked.trace.selected_stop is True
    blocked.resolve(Valence.NEUTRAL)

    stack.observe_success(cue, original)
    recovered = stack.respond(FRESH_CUE)
    assert recovered.tokens == FRESH_SEQUENCE
    recovered.resolve(Valence.NEUTRAL)
    assert stack.model_digest == original_digest


def test_duplicate_observations_do_not_promote_unique_payloads_to_axes() -> None:
    stack = RecurrentAssemblyStack()
    for cue, sequence in NESTED_SUCCESS_ROWS:
        stack.observe_success(cue, sequence)
        stack.observe_success(cue, sequence)

    state = stack._shape_states[(5, 7)]
    assert stack.incremental_update_count == len(NESTED_SUCCESS_ROWS)
    assert all(
        support == 1
        for (cue_index, _), support in state.cue_value_support.items()
        if cue_index in (3, 4)
    )
    controllers = {
        program.controller_index
        for program in state.active_programs.values()
        if program.controller_index is not None
    }
    assert controllers == {0, 1, 2}


def test_incremental_update_never_replays_prior_contributions() -> None:
    stack = RecurrentAssemblyStack()
    for cue, sequence in NESTED_SUCCESS_ROWS:
        stack.observe_success(cue, sequence)

    stack.observe_success(
        (10, 20, 30, 1_009, 1_010),
        (401, 501, 1_010, 601, 1_009, 502, 402),
    )

    cue_arity = 5
    output_length = 7
    assert stack.last_coactivity_write_count == (
        cue_arity + output_length + cue_arity * output_length
    )
    assert stack.prior_contribution_scan_count == 0
    assert stack.incremental_update_count == 5
