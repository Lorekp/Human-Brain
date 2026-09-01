from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from threading import Barrier, Lock

import pytest
from experiments.r3_pilot import common_continual_documentation_action_memory_v1 as continual
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from experiments.r3_pilot import common_promoted_rust_action_vault_v1 as vaults
from humanbrain import rust_selected_action_executor

MAP_SOURCE = """pub fn solve(values: &[i64]) -> Vec<i64> {
    values.iter().map(|value| *value * 2).collect()
}
"""

FILTER_SOURCE = """pub fn solve(values: &[i64]) -> Vec<i64> {
    values
        .iter()
        .copied()
        .filter(|value| *value >= 0)
        .collect()
}
"""


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _candidate(candidate_id: str, symbol: str):
    return knowledge_action.project_knowledge_action_candidate_v1(
        candidate_id,
        f"def solve(values):\n    return tools.{symbol}(values)\n",
    )


def _lesson(*, candidate, task_text: str, factor: str, label: str):
    task = knowledge_action.project_knowledge_action_task_v1(task_text)
    evidence = continual.make_documentation_action_evidence_v1(
        factors=(factor,),
        source_sha256=_root(f"documentation-source:{label}"),
        provenance_roots=(_root(f"documentation-provenance:{label}"),),
    )
    outputs = ((2, 4), (-6,))
    return continual.seal_continual_documentation_action_lesson_v1(
        task=task,
        task_instance_sha256=_root(f"task-instance:{label}"),
        task_source_sha256=_root(f"task-source:{label}"),
        evidence=evidence,
        candidate=candidate,
        execution_sha256=_root(f"execution:{label}"),
        public_feedback_sha256=_root(f"feedback:{label}"),
        observed_outputs=outputs,
        expected_outputs=outputs,
    )


def _curriculum_memories():
    map_action = _candidate("map_action", "map_values")
    filter_action = _candidate("filter_action", "filter_values")
    lessons = (
        _lesson(
            candidate=map_action,
            task_text="Using iterator map, double every input value.",
            factor="tok:map",
            label="a1",
        ),
        _lesson(
            candidate=filter_action,
            task_text="Using iterator filter, keep nonnegative values.",
            factor="tok:filter",
            label="b1",
        ),
        _lesson(
            candidate=map_action,
            task_text="Apply map to double another input sequence.",
            factor="tok:map",
            label="a2",
        ),
        _lesson(
            candidate=filter_action,
            task_text="Apply filter to another input sequence.",
            factor="tok:filter",
            label="b2",
        ),
    )
    memories = [continual.start_continual_documentation_action_memory_v1(lessons[0])]
    for lesson in lessons[1:]:
        memories.append(
            continual.advance_continual_documentation_action_memory_v1(memories[-1], lesson)
        )
    return tuple(memories), map_action, filter_action


def _promoted_memory():
    memories, map_action, filter_action = _curriculum_memories()
    return memories[-1], map_action, filter_action


def test_prepare_vault_commits_full_candidate_catalogue_in_memory_order() -> None:
    memory, map_action, filter_action = _promoted_memory()

    prepared = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=(
            (filter_action, FILTER_SOURCE),
            (map_action, MAP_SOURCE),
        ),
    )
    ordered = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=(
            (map_action, MAP_SOURCE),
            (filter_action, FILTER_SOURCE),
        ),
    )

    assert type(prepared) is vaults.PreparedPromotedRustActionVaultV1
    assert prepared.authority.vault_receipt_sha256 == prepared.receipt.sha256
    assert prepared.authority_sha256 == prepared.authority.sha256
    assert prepared.authority_sha256 != ordered.authority_sha256
    assert prepared.receipt == ordered.receipt
    assert prepared.receipt.memory_checkpoint_sha256 == memory.checkpoint_sha256
    assert prepared.receipt.candidate_sha256s == ordered.receipt.candidate_sha256s
    assert prepared.receipt.binding_sha256s == ordered.receipt.binding_sha256s
    assert prepared.receipt.rust_source_sha256s == ordered.receipt.rust_source_sha256s
    assert prepared.receipt.candidate_sha256s == tuple(
        candidate.sha256 for candidate in memory.candidate_projections
    )
    assert prepared.receipt.promoted_candidate_sha256s == memory.promoted_candidate_sha256s
    assert prepared.receipt.provenance_sha256s == tuple(row.sha256 for row in memory.provenance)
    assert prepared.receipt.rust_source_sha256s == (
        sha256(MAP_SOURCE.encode("ascii")).hexdigest(),
        sha256(FILTER_SOURCE.encode("ascii")).hexdigest(),
    )


def test_execute_selected_releases_exact_rust_source_once_before_callback() -> None:
    memory, map_action, filter_action = _promoted_memory()
    prepared = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=(
            (map_action, MAP_SOURCE),
            (filter_action, FILTER_SOURCE),
        ),
    )
    recall = prepared.recall(
        knowledge_action.project_knowledge_action_task_v1(
            "Build the output with map for this fresh sequence."
        )
    )
    observed: list[rust_selected_action_executor.RustSelectedSourceV2] = []

    def callback(selected: rust_selected_action_executor.RustSelectedSourceV2):
        with pytest.raises(vaults.PromotedRustActionVaultError, match="already released"):
            prepared.execute_selected(recall, lambda _value: None)
        observed.append(selected)
        return selected

    selected = prepared.execute_selected(recall, callback)

    assert type(selected) is rust_selected_action_executor.RustSelectedSourceV2
    selected.authenticate_structure()
    assert observed == [selected]
    assert selected.authority_sha256 == prepared.authority_sha256
    assert selected.selection_sha256 == recall.sha256
    assert selected.action_handle == map_action.candidate_id
    assert selected.source_sha256 == sha256(MAP_SOURCE.encode("ascii")).hexdigest()
    assert selected.source == MAP_SOURCE


def test_re_signed_but_unissued_recall_cannot_release_source() -> None:
    memory, map_action, filter_action = _promoted_memory()
    prepared = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=(
            (map_action, MAP_SOURCE),
            (filter_action, FILTER_SOURCE),
        ),
    )
    task = knowledge_action.project_knowledge_action_task_v1(
        "Use map for one issued withheld sequence."
    )
    unissued = memory.recall(task)
    with pytest.raises(vaults.PromotedRustActionVaultError, match="not issued"):
        prepared.execute_selected(unissued, lambda value: value)
    recall = prepared.recall(task)
    assert recall == unissued
    partial = replace(
        recall,
        unknown_factor_count=recall.unknown_factor_count + 1,
        sha256="",
    )
    forged = replace(
        partial,
        sha256=continual._digest(
            continual._RECALL_NAMESPACE,
            partial.record(include_sha256=False),
        ),
    )
    forged.authenticate()
    callbacks: list[rust_selected_action_executor.RustSelectedSourceV2] = []

    with pytest.raises(vaults.PromotedRustActionVaultError, match="not issued"):
        prepared.execute_selected(forged, callbacks.append)
    selected = prepared.execute_selected(recall, lambda value: value)

    assert callbacks == []
    assert selected.selection_sha256 == recall.sha256


def test_equivalent_preparations_have_distinct_explicit_authority_custody() -> None:
    memory, map_action, filter_action = _promoted_memory()
    bindings = (
        (map_action, MAP_SOURCE),
        (filter_action, FILTER_SOURCE),
    )
    first = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=bindings,
    )
    second = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=bindings,
    )
    task = knowledge_action.project_knowledge_action_task_v1(
        "Use map for one explicitly authorized withheld sequence."
    )
    first_recall = first.recall(task)
    second_recall = second.recall(task)

    first_source = first.execute_selected(first_recall, lambda value: value)
    second_source = second.execute_selected(second_recall, lambda value: value)

    assert first_recall == second_recall
    assert first.authority_sha256 != second.authority_sha256
    assert first_source.authority_sha256 == first.authority_sha256
    assert second_source.authority_sha256 == second.authority_sha256
    assert first_source.release_sha256 != second_source.release_sha256
    assert first_source.sha256 != second_source.sha256


def test_stop_checkpoint_splice_and_selected_field_change_never_reach_callback() -> None:
    memories, map_action, filter_action = _curriculum_memories()
    memory = memories[-1]
    prepared = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=(
            (map_action, MAP_SOURCE),
            (filter_action, FILTER_SOURCE),
        ),
    )
    task = knowledge_action.project_knowledge_action_task_v1(
        "Build the output with map for a withheld sequence."
    )
    stopped = prepared.recall(task, cell_gate=False)
    parent_recall = memories[-2].recall(task)
    live_recall = prepared.recall(task)
    changed_selection = replace(live_recall, selected_candidate_id="filter_action")
    callback_count = 0

    def callback(_selected):
        nonlocal callback_count
        callback_count += 1

    with pytest.raises(vaults.PromotedRustActionVaultError, match="stopped recall"):
        prepared.execute_selected(stopped, callback)
    with pytest.raises(vaults.PromotedRustActionVaultError, match="checkpoint differs"):
        prepared.execute_selected(parent_recall, callback)
    with pytest.raises(vaults.PromotedRustActionVaultError, match="authentication failed"):
        prepared.execute_selected(changed_selection, callback)

    assert callback_count == 0


def test_prepare_rejects_incomplete_duplicate_and_foreign_candidate_bindings() -> None:
    memories, map_action, filter_action = _curriculum_memories()
    memory = memories[-1]
    foreign_action = _candidate("foreign_action", "foreign_values")

    with pytest.raises(vaults.PromotedRustActionVaultError, match="promoted action"):
        vaults.prepare_promoted_rust_action_vault_v1(
            memories[0],
            candidate_sources=((map_action, MAP_SOURCE),),
        )

    invalid_bindings = (
        ((map_action, MAP_SOURCE),),
        ((map_action, MAP_SOURCE), (map_action, FILTER_SOURCE)),
        ((map_action, MAP_SOURCE), (foreign_action, FILTER_SOURCE)),
    )
    for candidate_sources in invalid_bindings:
        with pytest.raises(vaults.PromotedRustActionVaultError):
            vaults.prepare_promoted_rust_action_vault_v1(
                memory,
                candidate_sources=candidate_sources,
            )

    with pytest.raises(vaults.PromotedRustActionVaultError, match="ASCII"):
        vaults.prepare_promoted_rust_action_vault_v1(
            memory,
            candidate_sources=(
                (map_action, MAP_SOURCE),
                (filter_action, "pub fn solve(_: &[i64]) -> Vec<i64> { vec![1] } // ©"),
            ),
        )


def test_prepare_rejects_one_actual_rust_source_bound_to_two_candidates() -> None:
    memory, map_action, filter_action = _promoted_memory()

    with pytest.raises(vaults.PromotedRustActionVaultError, match="source is duplicated"):
        vaults.prepare_promoted_rust_action_vault_v1(
            memory,
            candidate_sources=(
                (map_action, MAP_SOURCE),
                (filter_action, MAP_SOURCE),
            ),
        )


def test_concurrent_reuse_releases_to_exactly_one_callback() -> None:
    memory, map_action, filter_action = _promoted_memory()
    prepared = vaults.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=(
            (map_action, MAP_SOURCE),
            (filter_action, FILTER_SOURCE),
        ),
    )
    recall = prepared.recall(
        knowledge_action.project_knowledge_action_task_v1(
            "Use map for one concurrent withheld sequence."
        )
    )
    start = Barrier(2)
    callback_lock = Lock()
    callback_count = 0

    def attempt() -> str:
        nonlocal callback_count
        start.wait()

        def callback(_selected):
            nonlocal callback_count
            with callback_lock:
                callback_count += 1
            return "released"

        try:
            return prepared.execute_selected(recall, callback)
        except vaults.PromotedRustActionVaultError as exc:
            assert "already released" in str(exc)
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(lambda _index: attempt(), range(2)))

    assert sorted(outcomes) == ["rejected", "released"]
    assert callback_count == 1
