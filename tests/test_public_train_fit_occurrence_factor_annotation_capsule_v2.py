from __future__ import annotations

from dataclasses import replace

import pytest
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_annotation_capsule_v1 as v1,
)
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_annotation_capsule_v2 as v2,
)


def _walk(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def test_v2_atomically_publishes_complete_producer_owned_fit_targets() -> None:
    assert v1.CAPSULE.status == v2.PREDECESSOR_STATUS
    assert v1.CAPSULE.capsule_sha256 == v2.PREDECESSOR_CAPSULE_SHA256
    assert v1.STOP_LEDGER.sha256 == v2.PREDECESSOR_STOP_SHA256
    assert v1.EMITTED_TARGET_RECORD_COUNT == 0

    capsule = v2.CAPSULE
    assert capsule.status == v2.READY
    assert capsule.stop_reason is None
    assert capsule.counts == v2.CapsuleCounts(
        fit_identity_count=9,
        fit_appearance_count=12,
        target_shape_count=2,
        source_count=77,
        raw_source_count=59,
        literal_source_count=18,
        edge_count=68,
        factor_slot_count=164,
        relation_count=9,
        domain_count=18,
        literal_copy_group_count=18,
        signature_count=18,
        emitted_target_record_count=372,
        carrier_conflict_count=0,
        missing_raw_observation_count=0,
        producer_geometry_conflict_count=0,
    )
    targets = v2.require_ready_annotation_capsule_v2(capsule)
    assert targets is capsule.targets
    with pytest.raises(RuntimeError, match=v2.STOP_INCOMPLETE):
        v2.require_ready_annotation_capsule_v2(replace(capsule))

    all_carriers: set[tuple[int, ...]] = set()
    raw_observation_ids: set[str] = set()
    for target in targets:
        source_by_carrier = {
            source.node_occurrence_cells: source for source in target.sources
        }
        assert len(source_by_carrier) == len(target.sources)
        assert target.root_occurrence_cells in source_by_carrier
        all_carriers.update(source_by_carrier)

        raw_sources = tuple(
            source
            for source in target.sources
            if isinstance(source, v2.RawSourceTarget)
        )
        literal_sources = tuple(
            source
            for source in target.sources
            if isinstance(source, v2.LiteralSourceTarget)
        )
        assert len(
            {
                (
                    source.typed_value_event_cells,
                    source.typed_value_type_cells,
                )
                for source in literal_sources
            }
        ) == len(literal_sources)
        assert len({
            (
                source.observation.direction,
                source.observation.cue_slot,
                source.observation.selected_slot,
            )
            for source in raw_sources
        }) == len(raw_sources)
        for source in raw_sources:
            observation = source.observation
            assert len(observation.assembly_cells) == 12
            assert tuple(sorted(set(observation.assembly_cells))) == (
                observation.assembly_cells
            )
            assert observation.direction in ("NEXT", "PREV")
            expected_selected = (
                (observation.cue_slot + 1) % 4
                if observation.direction == "NEXT"
                else (observation.cue_slot - 1) % 4
            )
            assert observation.selected_slot == expected_selected
            assert observation.physical_span_event_sha256
            assert observation.producer_provenance_sha256
            raw_observation_ids.add(observation.sha256)

        factor_carriers = {row.node_occurrence_cells for row in target.factor_slots}
        assert factor_carriers == set(source_by_carrier)
        assert all(
            row.parent_occurrence_cells in source_by_carrier
            and row.child_occurrence_cells in source_by_carrier
            for row in target.edges
        )
        assert all(
            row.node_occurrence_cells in source_by_carrier
            for row in (*target.relations, *target.domains)
        )
        for domain in target.domains:
            literal = source_by_carrier[domain.node_occurrence_cells]
            assert isinstance(literal, v2.LiteralSourceTarget)
            assert domain.complete_allowed_value_event_cells == (
                literal.typed_value_event_cells,
            )
        for group in target.literal_copy_groups:
            assert len(group.copy_group_cells) == 8
            assert len(group.terminal_occurrence_slot_cells) == 1
            carrier, slot = group.terminal_occurrence_slot_cells[0]
            assert isinstance(source_by_carrier[carrier], v2.LiteralSourceTarget)
            assert slot == 0
        assert all(
            signature.root_occurrence_cells == target.root_occurrence_cells
            for signature in target.signatures
        )

    assert len(all_carriers) == 77
    assert len(raw_observation_ids) == 59

    neural_records = capsule.neural_records()
    flattened = tuple(_walk(neural_records))
    assert not any(
        isinstance(value, str) and "sha256" in value for value in flattened
    )
    neural_string_values = {
        value for value in flattened if isinstance(value, str) and not value.isidentifier()
    }
    assert neural_string_values <= {"NEXT", "PREV"}
    assert v2.HELDOUT_INPUT_LOOKUP_COUNT == 0
    assert v2.HELDOUT_TARGET_READ_COUNT == 0
    assert v2.LEARNER_ARRAY_COUNT == 0
    assert v2.LEARNING_RUN_COUNT == 0
    assert v2.WEIGHT_READ_COUNT == 0
    assert v2.CHECKPOINT_READ_COUNT == 0
    assert v2.DESIRED_READ_COUNT == 0
    assert v2.SOURCE_ORDER_GEOMETRY_CHOICE_COUNT == 0
    assert v2.FIXED_CANDIDATE_GEOMETRY_CHOICE_COUNT == 0
    assert v2.HASH_GEOMETRY_CHOICE_COUNT == 0
