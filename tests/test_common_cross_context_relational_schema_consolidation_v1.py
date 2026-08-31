from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import (
    common_cross_context_relational_schema_consolidation_v1 as schemas,
)


def test_relation_schema_transfers_abstains_revises_and_keeps_v1_episodes() -> None:
    relation = b"\x19\xfe"
    old_object = b"\x31"
    new_object = b"\x72"
    subject_1, subject_2, subject_3, subject_4 = (
        b"\xa1\x01",
        b"\xa1\x02",
        b"\xa1\x03",
        b"\xa1\x04",
    )
    source_1, source_2, source_3, source_4, source_5 = (
        b"\xc1\x01",
        b"\xc1\x02",
        b"\xc1\x03",
        b"\xc1\x04",
        b"\xc1\x05",
    )

    other_relation = b"\x0d\xaa"
    other_object = b"\x99"
    other_subject_1, other_subject_2 = b"\x88\x01", b"\x88\x02"
    other_source_1, other_source_2 = b"\xd0\x01", b"\xd0\x02"

    sparse_relation = b"\x44\xbb"
    sparse_object = b"\x27"
    sparse_subject, sparse_source = b"\x55\x01", b"\xe0\x01"
    silent_subject, silent_relation, silent_source = (
        b"\xee\x01",
        b"\xee\x02",
        b"\xee\x03",
    )

    learned = schemas.fit_common_cross_context_relational_schemas(
        (
            schemas.RelationalSchemaLesson(subject_1, relation, old_object, source_1),
            # A second source for the same subject remains an episode but is
            # not independent relation-wide schema evidence.
            schemas.RelationalSchemaLesson(subject_1, relation, old_object, source_2),
            schemas.RelationalSchemaLesson(subject_2, relation, old_object, source_3),
            schemas.RelationalSchemaLesson(
                other_subject_1,
                other_relation,
                other_object,
                other_source_1,
            ),
            schemas.RelationalSchemaLesson(
                other_subject_2,
                other_relation,
                other_object,
                other_source_2,
            ),
            schemas.RelationalSchemaLesson(
                sparse_subject,
                sparse_relation,
                sparse_object,
                sparse_source,
            ),
            schemas.RelationalSchemaLesson(subject_3, relation, new_object, source_4),
            schemas.RelationalSchemaLesson(subject_4, relation, new_object, source_5),
            schemas.RelationalSchemaLesson(
                silent_subject,
                silent_relation,
                b"\x66",
                silent_source,
                teacher_gate=False,
            ),
        )
    )

    summary = learned.training_summary
    assert summary.teacher_on_lesson_count == 8
    assert summary.teacher_off_lesson_count == 1
    assert summary.episodic_novel_episode_count == 8
    assert summary.independent_schema_context_count == 7
    assert summary.dependent_context_count == 1
    assert summary.schema_neuron_count == 4
    assert summary.excitatory_update_events == 7
    assert summary.negative_evidence_update_events == 2
    assert tuple(step.status for step in summary.consolidation_steps) == (
        "SUPPORT_PENDING",
        "DEPENDENT_CONTEXT_IGNORED",
        "CONSOLIDATED",
        "SUPPORT_PENDING",
        "CONSOLIDATED",
        "SUPPORT_PENDING",
        "CONTRADICTION_ABSTAIN",
        "REVISED",
        "TEACHER_OFF",
    )

    novel_subject = b"\xf9\x11"
    revised = learned.recall(novel_subject, relation)
    unrelated = learned.recall(b"\xf9\x12", other_relation)
    sparse = learned.recall(b"\xf9\x13", sparse_relation)
    assert revised.status == unrelated.status == "RECALLED"
    assert revised.output.token == new_object[0]
    assert unrelated.output.token == other_object[0]
    assert sparse.status == "ABSTAIN_LOW_SUPPORT"
    assert (
        revised.subject_context_cells != learned.recall(subject_1, relation).subject_context_cells
    )

    old_episode_1 = learned.recall(subject_1, relation, source_event=source_1)
    old_episode_2 = learned.recall(subject_1, relation, source_event=source_2)
    old_episode_3 = learned.recall(subject_2, relation, source_event=source_3)
    new_episode = learned.recall(subject_4, relation, source_event=source_5)
    assert old_episode_1.output.token == old_episode_2.output.token == old_object[0]
    assert old_episode_3.output.token == old_object[0]
    assert new_episode.output.token == new_object[0]
    assert old_episode_1.conditioned_source_cells != (old_episode_2.conditioned_source_cells)

    assert learned.recall(novel_subject, relation, cell_gate=False).status == "ABSTAIN_UNKNOWN"
    assert learned.recall(silent_subject, silent_relation).status == "ABSTAIN_UNKNOWN"
    assert (
        learned.recall(
            silent_subject,
            silent_relation,
            source_event=silent_source,
        ).status
        == "STOP_UNKNOWN"
    )

    assert schemas.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA == 0
    assert schemas.NEW_SCHEMA_CELL_COUNT == 33
    assert schemas.TOTAL_DECLARED_CELL_COUNT > schemas.REUSED_EPISODIC_CELL_COUNT
    assert sha256(Path(schemas.__file__).read_bytes()).hexdigest() == (schemas.SOURCE_SHA256)
    print(
        json.dumps(
            {
                "source_sha256": schemas.SOURCE_SHA256,
                "config_sha256": schemas.CONFIG_SHA256,
                "checkpoint_sha256": learned.checkpoint_sha256,
                "v1_checkpoint_sha256": summary.v1_checkpoint_sha256,
                "v1_episodic_checkpoint_sha256": (summary.v1_episodic_checkpoint_sha256),
                "reused_common_event_cells": schemas.REUSED_COMMON_EVENT_CELL_COUNT,
                "reused_episodic_cells": schemas.REUSED_EPISODIC_CELL_COUNT,
                "new_schema_cells": schemas.NEW_SCHEMA_CELL_COUNT,
                "total_declared_cells": schemas.TOTAL_DECLARED_CELL_COUNT,
            },
            sort_keys=True,
        )
    )
