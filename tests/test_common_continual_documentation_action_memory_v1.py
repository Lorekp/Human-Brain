from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_continual_documentation_action_memory_v1 as continual
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action

MAP_A1 = "Using iterator map, double every input value and collect the results in order."
MAP_A2 = "Apply map to produce two times each item, preserving order in the output vector."
MAP_PROBE = (
    "Build the output with map: each input item becomes twice its original value, "
    "in the same order."
)
FILTER_B1 = (
    "Using iterator filter, keep values greater than or equal to zero and preserve their order."
)
FILTER_B2 = "Apply filter to remove every negative item while leaving the surviving items ordered."
FILTER_PROBE = (
    "Build the output with filter: retain only zero-or-positive items in their original order."
)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _candidate(candidate_id: str, symbol: str):
    return knowledge_action.project_knowledge_action_candidate_v1(
        candidate_id,
        f"def solve(values):\n    return tools.{symbol}(values)\n",
    )


def _lesson(
    *,
    candidate,
    task_text: str,
    label: str,
    factors: tuple[str, ...],
    passed: bool = True,
    task_instance_label: str | None = None,
    task_source_label: str | None = None,
):
    task = knowledge_action.project_knowledge_action_task_v1(task_text)
    documentation_source = _root(f"documentation-source:{label}")
    evidence = continual.make_documentation_action_evidence_v1(
        factors=factors,
        source_sha256=documentation_source,
        provenance_roots=(_root(f"documentation-provenance:{label}"),),
    )
    expected = ((2, 4), (-6,))
    observed = expected if passed else ((2, 4), (-5,))
    return continual.seal_continual_documentation_action_lesson_v1(
        task=task,
        task_instance_sha256=_root(task_instance_label or f"task-instance:{label}"),
        task_source_sha256=_root(task_source_label or f"task-source:{label}"),
        evidence=evidence,
        candidate=candidate,
        execution_sha256=_root(f"execution:{label}"),
        public_feedback_sha256=_root(f"feedback:{label}"),
        observed_outputs=observed,
        expected_outputs=expected,
    )


def _assert_physical_stop(recall) -> None:
    assert recall.status != continual.RECALLED
    assert recall.selected_action_index is None
    assert recall.selected_candidate_id is None
    assert recall.selected_candidate_sha256 is None
    assert recall.selected_source_sha256 is None
    assert recall.physical_output_record["decision"] == "STOP"
    recall.authenticate()


def test_interleaved_actions_promote_recall_and_do_not_forget() -> None:
    map_action = _candidate("map_action", "map_values")
    filter_action = _candidate("filter_action", "filter_values")
    a1 = _lesson(
        candidate=map_action,
        task_text=MAP_A1,
        label="a1",
        factors=("tok:map",),
    )
    b1 = _lesson(
        candidate=filter_action,
        task_text=FILTER_B1,
        label="b1",
        factors=("tok:filter",),
    )
    a2 = _lesson(
        candidate=map_action,
        task_text=MAP_A2,
        label="a2",
        factors=("tok:map",),
    )
    b2 = _lesson(
        candidate=filter_action,
        task_text=FILTER_B2,
        label="b2",
        factors=("tok:filter",),
    )

    memory_a1 = continual.start_continual_documentation_action_memory_v1(a1)
    unpromoted = memory_a1.recall(knowledge_action.project_knowledge_action_task_v1(MAP_PROBE))
    assert unpromoted.status == continual.STOP_ACTION_UNPROMOTED
    _assert_physical_stop(unpromoted)

    memory_b1 = continual.advance_continual_documentation_action_memory_v1(memory_a1, b1)
    parent_root = memory_b1.checkpoint_sha256
    parent_arrays = memory_b1.checkpoint_array_sha256s
    memory_a2 = continual.advance_continual_documentation_action_memory_v1(memory_b1, a2)
    assert memory_a2.promoted_candidate_sha256s == (map_action.sha256,)
    map_recall = memory_a2.recall(knowledge_action.project_knowledge_action_task_v1(MAP_PROBE))
    assert map_recall.status == continual.RECALLED
    assert map_recall.selected_candidate_sha256 == map_action.sha256
    filter_unpromoted = memory_a2.recall(
        knowledge_action.project_knowledge_action_task_v1(FILTER_PROBE)
    )
    assert filter_unpromoted.status == continual.STOP_ACTION_UNPROMOTED
    _assert_physical_stop(filter_unpromoted)

    memory_b2 = continual.advance_continual_documentation_action_memory_v1(memory_a2, b2)
    assert memory_b2.candidate_projections == (map_action, filter_action)
    assert memory_b2.promoted_candidate_sha256s == (
        map_action.sha256,
        filter_action.sha256,
    )
    assert (
        memory_b2.recall(
            knowledge_action.project_knowledge_action_task_v1(MAP_PROBE)
        ).selected_candidate_sha256
        == map_action.sha256
    )
    assert (
        memory_b2.recall(
            knowledge_action.project_knowledge_action_task_v1(FILTER_PROBE)
        ).selected_candidate_sha256
        == filter_action.sha256
    )

    memory_b1.authenticate()
    assert memory_b1.checkpoint_sha256 == parent_root
    assert memory_b1.checkpoint_array_sha256s == parent_arrays
    summary = memory_b2.training_summary
    assert summary.raw_task_text_count == 0
    assert summary.raw_output_count == 0
    assert summary.parent_lesson_replay_count == 0
    assert summary.parent_mutation_count == 0


def test_promotion_requires_fresh_task_projection_and_task_source() -> None:
    map_action = _candidate("map_action", "map_values")
    a1 = _lesson(
        candidate=map_action,
        task_text=MAP_A1,
        label="a1",
        factors=("tok:map",),
        task_source_label="shared-source",
    )
    base = continual.start_continual_documentation_action_memory_v1(a1)

    same_source = continual.advance_continual_documentation_action_memory_v1(
        base,
        _lesson(
            candidate=map_action,
            task_text=MAP_A2,
            label="same-source",
            factors=("tok:map",),
            task_source_label="shared-source",
        ),
    )
    assert same_source.promoted_candidate_sha256s == ()

    same_projection = continual.advance_continual_documentation_action_memory_v1(
        base,
        _lesson(
            candidate=map_action,
            task_text=MAP_A1,
            label="same-projection",
            factors=("tok:map",),
            task_source_label="fresh-source",
        ),
    )
    assert same_projection.promoted_candidate_sha256s == ()

    failed_fresh = continual.advance_continual_documentation_action_memory_v1(
        base,
        _lesson(
            candidate=map_action,
            task_text=MAP_A2,
            label="failed-fresh",
            factors=("tok:map",),
            passed=False,
        ),
    )
    assert failed_fresh.promoted_candidate_sha256s == ()

    promoted = continual.advance_continual_documentation_action_memory_v1(
        base,
        _lesson(
            candidate=map_action,
            task_text=MAP_A2,
            label="fresh-pair",
            factors=("tok:map",),
        ),
    )
    assert promoted.promoted_candidate_sha256s == (map_action.sha256,)

    filter_action = _candidate("filter_action", "filter_values")
    failed_new_action = _lesson(
        candidate=filter_action,
        task_text=FILTER_B1,
        label="failed-new-action",
        factors=("tok:filter",),
        passed=False,
    )
    with pytest.raises(
        continual.ContinualDocumentationActionMemoryError,
        match="failed lesson cannot introduce",
    ):
        continual.advance_continual_documentation_action_memory_v1(promoted, failed_new_action)


def test_unknown_unpromoted_tie_and_gates_stop_physically() -> None:
    map_action = _candidate("map_action", "map_values")
    filter_action = _candidate("filter_action", "filter_values")
    lessons = (
        _lesson(
            candidate=map_action,
            task_text="Use shared evidence to map the first task.",
            label="shared-a1",
            factors=("tok:shared",),
        ),
        _lesson(
            candidate=filter_action,
            task_text="Use shared evidence to filter the first task.",
            label="shared-b1",
            factors=("tok:shared",),
        ),
        _lesson(
            candidate=map_action,
            task_text="Apply the shared documentation while mapping another task.",
            label="shared-a2",
            factors=("tok:shared",),
        ),
        _lesson(
            candidate=filter_action,
            task_text="Apply shared documentation while filtering another task.",
            label="shared-b2",
            factors=("tok:shared",),
        ),
    )
    memory = continual.start_continual_documentation_action_memory_v1(lessons[0])
    for lesson in lessons[1:]:
        memory = continual.advance_continual_documentation_action_memory_v1(memory, lesson)

    probe = knowledge_action.project_knowledge_action_task_v1(
        "Use shared documentation for this probe."
    )
    tied = memory.recall(probe)
    assert tied.status == continual.STOP_PHYSICAL_RACE
    assert tied.candidate_scores[0][-1] == tied.candidate_scores[1][-1] == 2
    _assert_physical_stop(tied)

    unknown = memory.recall(
        knowledge_action.project_knowledge_action_task_v1(
            "A xenolith quasar has no learned lexical factor."
        )
    )
    assert unknown.status == continual.STOP_NO_MATCHED_FACTOR
    _assert_physical_stop(unknown)

    cell_off = memory.recall(probe, cell_gate=False)
    assert cell_off.status == continual.STOP_CELL_OFF
    _assert_physical_stop(cell_off)
    documentation_off = memory.recall(probe, documentation_gate=False)
    assert documentation_off.status == continual.STOP_DOCUMENTATION_OFF
    _assert_physical_stop(documentation_off)


def test_authentication_replay_free_advance_and_output_discard() -> None:
    map_action = _candidate("map_action", "map_values")
    lesson = _lesson(
        candidate=map_action,
        task_text=MAP_A1,
        label="authentication",
        factors=("tok:map",),
    )
    memory = continual.start_continual_documentation_action_memory_v1(lesson)
    assert (
        continual.advance_continual_documentation_action_memory_v1(
            memory, lesson, plasticity_gate=False
        )
        is memory
    )
    with pytest.raises(
        continual.ContinualDocumentationActionMemoryError,
        match="lesson replay",
    ):
        continual.advance_continual_documentation_action_memory_v1(memory, lesson)
    with pytest.raises(TypeError, match="exact boolean"):
        continual.advance_continual_documentation_action_memory_v1(
            memory, lesson, plasticity_gate=1
        )

    evidence = continual.make_documentation_action_evidence_v1(
        factors=("tok:map",),
        source_sha256=_root("tamper-source"),
        provenance_roots=(_root("tamper-provenance"),),
    )
    with pytest.raises(continual.ContinualDocumentationActionMemoryError):
        replace(evidence, source_sha256=_root("changed-source")).authenticate()
    with pytest.raises(continual.ContinualDocumentationActionMemoryError):
        replace(lesson, passed_cell=0).authenticate()
    with pytest.raises(continual.ContinualDocumentationActionMemoryError):
        replace(lesson, observed_outputs=((999,),)).authenticate()
    with pytest.raises(ValueError):
        memory._excitatory[0, 0] = 7

    colliding_action = knowledge_action.project_knowledge_action_candidate_v1(
        "map_action",
        "def solve(items):\n    return tools.map_values(items)\n",
    )
    colliding_lesson = _lesson(
        candidate=colliding_action,
        task_text=MAP_A2,
        label="candidate-collision",
        factors=("tok:map",),
    )
    with pytest.raises(
        continual.ContinualDocumentationActionMemoryError,
        match="candidate id or symbol collision",
    ):
        continual.advance_continual_documentation_action_memory_v1(memory, colliding_lesson)

    promoted = continual.advance_continual_documentation_action_memory_v1(
        memory,
        _lesson(
            candidate=map_action,
            task_text=MAP_A2,
            label="receipt-promotion",
            factors=("tok:map",),
        ),
    )
    recall = promoted.recall(knowledge_action.project_knowledge_action_task_v1(MAP_PROBE))
    forged = replace(recall, selected_candidate_id="forged", sha256="")
    forged = replace(
        forged,
        sha256=continual._digest(
            continual._RECALL_NAMESPACE,
            forged.record(include_sha256=False),
        ),
    )
    with pytest.raises(
        continual.ContinualDocumentationActionMemoryError,
        match="recalled selection changed",
    ):
        forged.authenticate()

    summary = memory.training_summary
    assert summary.raw_task_text_count == 0
    assert summary.raw_output_count == 0
    assert all(not hasattr(row, "observed_outputs") for row in memory.provenance)
    assert all(not hasattr(row, "expected_outputs") for row in memory.provenance)
