from __future__ import annotations

import ast
import dataclasses
import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from experiments.r3_pilot import (
    public_train_prompt_factor_core_composition_probe_v1 as probe,
)


@pytest.fixture(scope="module")
def canonical_execution() -> probe._ProbeExecution:
    return probe._execute_fixture(probe._canonical_fixture())


def test_exact_self_posed_108_lesson_table_and_source_boundary() -> None:
    specs = probe._FROZEN_SPECS
    assert len(specs) == probe.LESSON_COUNT == 108
    assert len({spec.canonical_tokens for spec in specs}) == 108
    assert Counter(spec.split for spec in specs) == {
        probe.FIT: 72,
        probe.HELDOUT: 36,
    }
    assert len({spec.group_index for spec in specs}) == probe.GROUP_COUNT == 42
    assert all(len(spec.canonical_tokens) == 4 for spec in specs)
    assert all(len(spec.present_factors) == 1 for spec in specs[:54])
    assert all(len(spec.present_factors) == 2 for spec in specs[54:])

    for spec in specs[54:]:
        family = next(
            family
            for family in probe._COMPOSITION_TRIADS
            if set(family) <= set(spec.canonical_tokens)
        )
        factor = spec.canonical_tokens[0]
        companion = family[(family.index(factor) + 1) % 3]
        assert spec.present_factors == probe._registry_ordered_pair(
            factor,
            companion,
        )
    for group_index in range(probe.GROUP_COUNT):
        grouped = tuple(
            spec for spec in specs if spec.group_index == group_index
        )
        assert len(
            {tuple(sorted(spec.canonical_tokens)) for spec in grouped}
        ) == 1

    source_path = Path(inspect.getsourcefile(probe) or "")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
    assert "build_frozen_bridge" not in source
    assert not any(
        blocked in imported
        for imported in imported_modules
        for blocked in (
            "simple_swe",
            "capsule",
            "manifest",
            "protected",
        )
    )


def test_canonical_terminal_stop_and_hashes_are_exact(
    canonical_execution: probe._ProbeExecution,
) -> None:
    result = canonical_execution.result

    assert result.status == probe.STOP_STATUS == (
        "STOP_SELF_POSED_CORE_PROMPT_COMPOSITION_V1"
    )
    assert (
        result.group_count,
        result.lesson_count,
        result.fit_count,
        result.heldout_count,
        result.taught_factor_count,
        result.silent_head_count,
    ) == (42, 108, 72, 36, 9, 15)
    assert (
        result.excitatory_head_update_events,
        result.inhibitory_head_update_events,
        result.committed_cell_ticks,
        result.excitatory_max_ticks,
        result.inhibitory_max_ticks,
    ) == (77, 43, 15_282, 14, 10)
    assert result.mismatches == probe.EXPECTED_MISMATCHES
    assert len(result.mismatches) == 11
    assert (
        result.input_sha256,
        result.curriculum_sha256,
        result.checkpoint_sha256,
        result.result_sha256,
    ) == (
        probe.EXPECTED_INPUT_SHA256,
        probe.EXPECTED_CURRICULUM_SHA256,
        probe.EXPECTED_CHECKPOINT_SHA256,
        probe.EXPECTED_RESULT_SHA256,
    )
    assert result.checkpoint_immutable
    assert (
        result.self_posed_input_count,
        result.public_input_count,
        result.canonical_input_count,
        result.protected_input_count,
    ) == (108, 0, 0, 0)
    assert probe.run_probe() == result


def test_deterministic_rebuild_and_authentication_group_noncause(
    canonical_execution: probe._ProbeExecution,
) -> None:
    repeated = probe._execute_fixture(probe._canonical_fixture())
    reordered = probe._execute_fixture(
        probe._controlled_fixture(relabel_authentication=False)
    )
    relabeled = probe._execute_fixture(
        probe._controlled_fixture(relabel_authentication=True)
    )

    assert repeated.result == canonical_execution.result
    assert repeated.checkpoint.checkpoint_bytes() == (
        canonical_execution.checkpoint.checkpoint_bytes()
    )
    assert reordered.result == canonical_execution.result
    assert reordered.semantic_races == canonical_execution.semantic_races
    assert relabeled.result.input_sha256 != canonical_execution.result.input_sha256
    assert relabeled.result.result_sha256 != canonical_execution.result.result_sha256
    assert relabeled.result.curriculum_sha256 == (
        canonical_execution.result.curriculum_sha256
    )
    assert relabeled.result.checkpoint_sha256 == (
        canonical_execution.result.checkpoint_sha256
    )
    assert relabeled.result.mismatches == canonical_execution.result.mismatches
    assert relabeled.semantic_races == canonical_execution.semantic_races
    np.testing.assert_array_equal(
        relabeled.checkpoint.excitatory_ticks,
        canonical_execution.checkpoint.excitatory_ticks,
    )
    np.testing.assert_array_equal(
        relabeled.checkpoint.inhibitory_ticks,
        canonical_execution.checkpoint.inhibitory_ticks,
    )


def test_checkpoint_is_frozen_and_result_has_no_tuning_or_retry_fields(
    canonical_execution: probe._ProbeExecution,
) -> None:
    checkpoint = canonical_execution.checkpoint
    checkpoint_before = checkpoint.checkpoint_bytes()
    checkpoint_sha_before = checkpoint.checkpoint_sha256

    for prompt_input in canonical_execution.prompt_inputs:
        checkpoint.infer(prompt_input)
    excitatory = checkpoint.excitatory_ticks
    inhibitory = checkpoint.inhibitory_ticks
    excitatory.setflags(write=True)
    inhibitory.setflags(write=True)
    excitatory[0, 0] = (int(excitatory[0, 0]) + 1) % (probe.bridge.TICK_CAP + 1)
    inhibitory[1, 1] = (int(inhibitory[1, 1]) + 1) % (probe.bridge.TICK_CAP + 1)

    assert checkpoint.checkpoint_bytes() == checkpoint_before
    assert checkpoint.checkpoint_sha256 == checkpoint_sha_before
    assert not np.shares_memory(excitatory, checkpoint.excitatory_ticks)
    assert not np.shares_memory(inhibitory, checkpoint.inhibitory_ticks)

    result_fields = {
        field.name
        for field in dataclasses.fields(probe.CompositionProbeResult)
    }
    result_keys = set(canonical_execution.result.to_record())
    forbidden_fragments = ("tuning", "retry")
    assert not any(
        fragment in name
        for name in result_fields | result_keys | set(probe.__all__)
        for fragment in forbidden_fragments
    )
    assert tuple(inspect.signature(probe.run_probe).parameters) == ()
