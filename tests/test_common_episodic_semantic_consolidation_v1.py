from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_cells_v1 as projection
from experiments.r3_pilot import common_episodic_semantic_consolidation_v1 as memory


def test_self_posed_episodes_consolidate_abstain_revise_and_keep_provenance() -> None:
    subject = b"\xa1\x04"
    relation = b"\x19\xfe"
    old_object = b"\x31"
    new_object = b"\x72"
    source_1, source_2, source_3, source_4 = (
        b"\xc1\x01",
        b"\xc1\x02",
        b"\xc1\x03",
        b"\xc1\x04",
    )
    other_subject = b"\x88\x03"
    other_relation = b"\x0d\xaa"
    other_object = b"\x99"
    other_source_1, other_source_2 = b"\xd0\x01", b"\xd0\x02"
    silent_subject, silent_relation, silent_source = (
        b"\xee\x01",
        b"\xee\x02",
        b"\xee\x03",
    )

    learned = memory.fit_common_episodic_semantic_consolidation(
        (
            memory.EpisodicSemanticLesson(
                subject, relation, old_object, source_1
            ),
            memory.EpisodicSemanticLesson(
                other_subject,
                other_relation,
                other_object,
                other_source_1,
            ),
            memory.EpisodicSemanticLesson(
                subject, relation, old_object, source_2
            ),
            # Same physical key+source is locally non-novel and cannot vote twice.
            memory.EpisodicSemanticLesson(
                subject, relation, new_object, source_2
            ),
            memory.EpisodicSemanticLesson(
                other_subject,
                other_relation,
                other_object,
                other_source_2,
            ),
            memory.EpisodicSemanticLesson(
                subject, relation, new_object, source_3
            ),
            memory.EpisodicSemanticLesson(
                subject, relation, new_object, source_4
            ),
            memory.EpisodicSemanticLesson(
                silent_subject,
                silent_relation,
                b"\x44",
                silent_source,
                teacher_gate=False,
            ),
        )
    )

    summary = learned.training_summary
    assert summary.novel_episode_count == 6
    assert summary.duplicate_source_count == 1
    assert summary.semantic_neuron_count == 3
    assert summary.teacher_off_lesson_count == 1
    assert summary.negative_evidence_update_events == 2
    assert tuple(step.status for step in summary.consolidation_steps) == (
        "SUPPORT_PENDING",
        "SUPPORT_PENDING",
        "CONSOLIDATED",
        "DUPLICATE_SOURCE_IGNORED",
        "CONSOLIDATED",
        "CONTRADICTION_ABSTAIN",
        "REVISED",
        "TEACHER_OFF",
    )

    revised = learned.recall_semantic(subject, relation)
    retained = learned.recall_semantic(other_subject, other_relation)
    assert revised.status == retained.status == "RECALLED"
    assert revised.output.token == new_object[0]
    assert retained.output.token == other_object[0]

    old_episode_1 = learned.recall_episode(subject, relation, source_1)
    old_episode_2 = learned.recall_episode(subject, relation, source_2)
    new_episode = learned.recall_episode(subject, relation, source_4)
    assert old_episode_1.output.token == old_episode_2.output.token == old_object[0]
    assert new_episode.output.token == new_object[0]
    expected_source_cells = tuple(
        memory.ROLE_CELL_OFFSETS[3] + cell
        for cell in projection.CommonEventCells().project(source_1).trace_cells
    )
    assert old_episode_1.conditioned_source_cells == expected_source_cells

    assert learned.recall_semantic(
        subject, relation, cell_gate=False
    ).status == "ABSTAIN_UNKNOWN"
    assert learned.recall_semantic(
        b"\xff\x00", relation
    ).status == "ABSTAIN_LOW_SUPPORT"
    assert learned.recall_semantic(
        silent_subject, silent_relation
    ).status == "ABSTAIN_UNKNOWN"
    assert learned.recall_episode(
        silent_subject, silent_relation, silent_source
    ).status == "STOP_UNKNOWN"

    assert memory.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA == 0
    assert len(memory.CONFIG_SHA256) == len(learned.checkpoint_sha256) == 64
    assert sha256(Path(memory.__file__).read_bytes()).hexdigest() == (
        memory.SOURCE_SHA256
    )
