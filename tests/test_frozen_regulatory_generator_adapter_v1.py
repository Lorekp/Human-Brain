from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as g0
from experiments.g0 import frozen_regulatory_adapter_v1 as adapter
from experiments.g0 import frozen_regulatory_generator_v1 as generator


def test_generator_pins_g0_and_frozen_cohorts_remain_fail_closed() -> None:
    assert generator.EXPECTED_G0_COMMIT == "42278a75647951a7da7a51c442ac14bb54db7dab"
    assert g0.PROTOCOL.protocol_sha256 == generator.EXPECTED_G0_PROTOCOL_SHA256
    assert sha256(Path(g0.__file__).read_bytes()).hexdigest() == (
        generator.EXPECTED_G0_SOURCE_SHA256
    )
    assert generator.FROZEN_COHORT_SEEDS == (470000, 470001, 470002, 470003)
    assert set(generator.FIXTURE_SEEDS).isdisjoint(generator.FROZEN_COHORT_SEEDS)
    assert generator.GENERATION_AUTHORIZED is False
    assert generator.ARCHITECTURE_BINDING_COUNT == 0
    assert generator.FROZEN_COHORT_EXECUTION_COUNT == 0
    assert generator.FROZEN_SYSTEM_GENERATION_COUNT == 0
    assert generator.ADJUDICATIVE_SYSTEM_GENERATION_COUNT == 0
    assert generator.SOURCE_SPECIFICATION_SHA256 == (
        "e65bfcb7eccc32817d284df3c90243a8780fa755a20611d338590025792b4d05"
    )
    with pytest.raises(RuntimeError, match="nine-artifact common-architecture binding"):
        generator.generate_frozen_cohort(split="adjudicative")
    with pytest.raises(ValueError, match="non-cohort fixture seeds"):
        generator.generate_fixture_system(g0.REGULATORY_SPLIT.adjudicative_seed)


def test_quarantined_fixture_is_deterministic_and_meets_generator_rejections() -> None:
    first = generator.generate_fixture_system(generator.FIXTURE_SEEDS[0])
    second = generator.generate_fixture_system(generator.FIXTURE_SEEDS[0])
    audit = generator.validate_system(first)

    assert first == second
    assert first.sha256 == "7b69e72d3a16ff1a3a6edc2ee4c054c18ba0e3a29140d08822d2fe774d9e696f"
    assert audit == {
        "entity_count": 8,
        "distinct_entity_token_count": 8,
        "edge_count": 10,
        "self_edge_count": 0,
        "minimum_in_degree": 1,
        "maximum_in_degree": 2,
        "underlying_connected": True,
        "every_edge_effective": True,
        "every_node_takes_both_values": True,
        "composition_motifs": (),
        "sha256": first.sha256,
    }
    assert all(0 <= token <= 0xFFFF_FFFF for token in first.entity_tokens)
    assert len(set(first.entity_tokens)) == 8


def test_parameter_free_adapter_round_trips_only_permitted_raw_fields() -> None:
    system = generator.generate_fixture_system(generator.FIXTURE_SEEDS[0])
    current = (0, 1, 0, 1, 0, 1, 0, 1)
    following = system.transition(
        current,
        clamp_entity_token=system.entity_tokens[0],
        clamp_value=1,
    )
    events: tuple[adapter.RegulatoryEvent, ...] = (
        adapter.ObservationEvent(tuple(zip(system.entity_tokens, current, strict=True)), 0),
        adapter.ActionEvent(adapter.ACTION_CLAMP, system.entity_tokens[0], 1),
        adapter.OutcomeEvent(tuple(zip(system.entity_tokens, following, strict=True)), 1),
        adapter.BoundaryEvent(False, False),
        adapter.QueryEvent(system.entity_tokens[1], adapter.ANSWER_SLOT_BINARY),
    )

    assert tuple(adapter.deserialize_event(adapter.serialize_event(event)) for event in events) == (
        events
    )
    assert adapter.transcript_sha256(events) == (
        "8de5707495291a45aeebb03a89fce3581f097213e97e3fb2cc0a09d8e90976a4"
    )
    assert adapter.ADAPTER_SHA256 == (
        "e95420beb0fbe108a6a07c8c9d2543334d539118e77122d9c1506e1ec951b9fc"
    )
    assert g0.PROTOCOL.adapter_sha256 == adapter.EXPECTED_G0_ADAPTER_CONTRACT_SHA256
    assert adapter.negative_field_audit()["clean"] is True
    assert adapter.ADAPTER_RECORD["fitted_parameter_count"] == 0
    assert adapter.ADAPTER_RECORD["generator_dependency_count"] == 0
    assert adapter.ADAPTER_RECORD["evaluator_dependency_count"] == 0
    assert adapter.ADAPTER_RECORD["learner_dependency_count"] == 0

    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "from experiments.g0 import frozen_regulatory_generator" not in source
    assert "import frozen_regulatory_generator" not in source
