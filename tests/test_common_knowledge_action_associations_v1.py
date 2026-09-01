from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot import common_knowledge_action_memory_v1 as memory_v1

MESSAGE_SYMBOL = f"rust_action.ra_{'1' * 64}"
SHARED_STATE_SYMBOL = f"rust_action.ra_{'2' * 64}"
FACTORS = ("tok:alpha", "tok:beta", "tok:shared")
ASSOCIATIONS = (
    ("tok:alpha", MESSAGE_SYMBOL),
    ("tok:beta", SHARED_STATE_SYMBOL),
    ("tok:shared", MESSAGE_SYMBOL),
    ("tok:shared", SHARED_STATE_SYMBOL),
)


def _root(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _projection(
    *,
    associations: tuple[tuple[str, str], ...] = ASSOCIATIONS,
    source_projection_cell_steps: int = 10_000,
) -> memory_v1.KnowledgeActionAssociationProjectionV1:
    source_sha256 = _root("source")
    graph_sha256 = _root("graph")
    producer_receipt_sha256 = _root("producer")
    cognition_sha256 = _root("cognition")
    authorization_roots = tuple(sorted((_root("registry"), _root("selector"))))
    provenance_roots = tuple(
        sorted(
            {
                source_sha256,
                graph_sha256,
                producer_receipt_sha256,
                cognition_sha256,
                *authorization_roots,
                _root("card-receipt"),
            }
        )
    )
    return memory_v1.make_knowledge_action_association_projection_v1(
        source_sha256=source_sha256,
        graph_sha256=graph_sha256,
        producer_receipt_sha256=producer_receipt_sha256,
        cognition_sha256=cognition_sha256,
        authorization_roots=authorization_roots,
        factor_vocabulary=FACTORS,
        symbol_vocabulary=tuple(sorted((MESSAGE_SYMBOL, SHARED_STATE_SYMBOL))),
        associations=tuple(sorted(associations)),
        source_projection_cell_steps=source_projection_cell_steps,
        provenance_roots=provenance_roots,
    )


def _task(text: str) -> memory_v1.KnowledgeActionTaskProjectionV1:
    return memory_v1.project_knowledge_action_task_v1(text)


def _candidates() -> tuple[memory_v1.KnowledgeActionCandidateProjectionV1, ...]:
    return (
        memory_v1.project_knowledge_action_candidate_v1(
            "message_action",
            f"def choose():\n    return {MESSAGE_SYMBOL}()\n",
        ),
        memory_v1.project_knowledge_action_candidate_v1(
            "shared_state_action",
            f"def choose():\n    return {SHARED_STATE_SYMBOL}()\n",
        ),
    )


def test_shared_associations_abstain_and_unique_evidence_selects() -> None:
    projection = _projection()
    memory = memory_v1.fit_common_knowledge_action_associations_v1(projection)
    candidates = _candidates()

    shared_only = memory.recall(_task("shared"), candidates)
    selected = memory.recall(_task("shared alpha"), candidates)
    reversed_candidates = memory.recall(
        _task("shared alpha"), tuple(reversed(candidates))
    )
    equal_unique = memory.recall(_task("alpha beta"), candidates)

    assert shared_only.status == memory_v1.STOP_PHYSICAL_RACE
    assert shared_only.candidate_scores == (
        (MESSAGE_SYMBOL, 1),
        (SHARED_STATE_SYMBOL, 1),
    )
    assert shared_only.selected_symbol is None
    assert selected.status == memory_v1.RECALLED
    assert selected.selected_symbol == MESSAGE_SYMBOL
    assert selected.candidate_scores == (
        (MESSAGE_SYMBOL, 2),
        (SHARED_STATE_SYMBOL, 1),
    )
    assert selected.record() == reversed_candidates.record()
    assert equal_unique.status == memory_v1.STOP_PHYSICAL_RACE
    assert equal_unique.selected_symbol is None

    assert memory.training_summary.config_sha256 == memory_v1.ASSOCIATION_CONFIG_SHA256
    assert memory.training_summary.association_update_count == len(ASSOCIATIONS)
    assert memory.training_summary.training_cell_steps <= 1_800_000
    assert set(projection.provenance_roots).issubset(memory.provenance_roots)
    assert projection.sha256 in memory.provenance_roots


def test_plasticity_off_tamper_splice_and_training_cap_stop() -> None:
    projection = _projection()
    candidates = _candidates()
    no_plasticity = memory_v1.fit_common_knowledge_action_associations_v1(
        projection, plasticity_gate=False
    )
    absent = no_plasticity.recall(_task("shared alpha"), candidates)

    assert absent.status == memory_v1.STOP_PHYSICAL_RACE
    assert absent.selected_symbol is None
    assert absent.candidate_scores == (
        (MESSAGE_SYMBOL, 0),
        (SHARED_STATE_SYMBOL, 0),
    )

    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="projection"):
        replace(
            projection,
            associations=projection.associations[:-1],
        ).authenticate()
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="projection"):
        memory_v1.fit_common_knowledge_action_associations_v1(
            replace(projection, producer_receipt_sha256=_root("spliced-producer"))
        )

    extended = memory_v1.fit_common_knowledge_action_associations_v1(
        _projection(source_projection_cell_steps=1_000_001)
    )
    assert 1_000_000 < extended.training_summary.training_cell_steps <= 1_800_000

    over_cap = _projection(
        source_projection_cell_steps=(
            memory_v1.ASSOCIATION_MAX_TRAINING_CELL_STEPS
        )
    )
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="capacity"):
        memory_v1.fit_common_knowledge_action_associations_v1(over_cap)


def test_projection_rejects_noncanonical_or_unrooted_associations() -> None:
    projection = _projection()
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="projection"):
        memory_v1.make_knowledge_action_association_projection_v1(
            source_sha256=projection.source_sha256,
            graph_sha256=projection.graph_sha256,
            producer_receipt_sha256=projection.producer_receipt_sha256,
            cognition_sha256=projection.cognition_sha256,
            authorization_roots=projection.authorization_roots,
            factor_vocabulary=projection.factor_vocabulary,
            symbol_vocabulary=projection.symbol_vocabulary,
            associations=tuple(reversed(projection.associations)),
            source_projection_cell_steps=projection.source_projection_cell_steps,
            provenance_roots=projection.provenance_roots,
        )
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="projection"):
        memory_v1.make_knowledge_action_association_projection_v1(
            source_sha256=projection.source_sha256,
            graph_sha256=projection.graph_sha256,
            producer_receipt_sha256=projection.producer_receipt_sha256,
            cognition_sha256=projection.cognition_sha256,
            authorization_roots=projection.authorization_roots,
            factor_vocabulary=projection.factor_vocabulary,
            symbol_vocabulary=projection.symbol_vocabulary,
            associations=projection.associations,
            source_projection_cell_steps=projection.source_projection_cell_steps,
            provenance_roots=tuple(
                root
                for root in projection.provenance_roots
                if root != projection.authorization_roots[0]
            ),
        )
