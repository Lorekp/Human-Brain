from __future__ import annotations

from humanbrain.neutral_neuronal_document_learner import (
    NeutralFrameAblation,
    start,
)
from humanbrain.opaque_document_event_tape import OpaqueDocument


def _document(name: str, content: bytes) -> OpaqueDocument:
    encoded = name.encode("ascii")
    return OpaqueDocument(
        document_id=b"document/" + encoded,
        provenance_id=b"provenance/" + encoded,
        content=content,
    )


def _training_batches() -> tuple[
    tuple[OpaqueDocument, ...],
    tuple[OpaqueDocument, ...],
]:
    return (
        (
            _document("pavo-1", b"moru pavo orli"),
            _document("kenu-1", b"orli kenu moru"),
        ),
        (
            _document("pavo-2", b"sena pavo ulon"),
            _document("kenu-2", b"ulon kenu sena"),
        ),
    )


def _fresh(*contents: bytes) -> tuple[OpaqueDocument, ...]:
    return tuple(
        _document(f"fresh-{index}", content)
        for index, content in enumerate(contents)
    )


def test_neutral_frames_learn_swapped_transfer_and_preserve_ambiguity() -> None:
    first_batch, second_batch = _training_batches()
    empty = start()
    pending = empty.consolidate(first_batch)

    assert empty.training_summary.retained_evidence_count == 0
    assert pending.training_summary.promoted_frame_count == 0
    assert pending.respond(b"neri kenu", _fresh(b"tava pavo neri")).stopped

    brain = pending.consolidate(second_batch)
    summary = brain.training_summary
    assert summary.promoted_frame_count == 2
    assert summary.directional_route_count == 2
    assert summary.retained_raw_document_count == 0
    assert summary.retained_full_sentence_count == 0
    assert summary.retained_lexical_byte_count == 0

    forward = brain.respond(b"neri kenu", _fresh(b"tava pavo neri"))
    assert forward.status == "EMIT"
    assert forward.emitted == b"tava"
    assert not forward.stopped
    assert forward.support_path_count >= 1
    assert forward.cell_races[-1].population == "response/motor"
    assert forward.cell_races[-1].winner_cell_id == 0

    reverse_slot = brain.respond(b"kenu tava", _fresh(b"tava pavo neri"))
    assert reverse_slot.emitted == b"neri"

    reverse_route = brain.respond(b"tava pavo", _fresh(b"neri kenu tava"))
    assert reverse_route.emitted == b"neri"

    fact_documents = (
        _document("fact-a", b"tava pavo neri"),
        _document("fact-b", b"tava pavo neri"),
    )
    parent_checkpoint = brain.checkpoint_sha256
    remembered = brain.consolidate(fact_documents)
    assert brain.checkpoint_sha256 == parent_checkpoint
    assert brain.training_summary.consolidated_fact_count == 0
    assert brain.respond(b"neri kenu").stopped
    assert remembered.training_summary.consolidated_fact_count == 1
    assert remembered.training_summary.fact_source_count == 2
    assert remembered.training_summary.retained_lexical_byte_count > 0

    delayed = remembered.respond(b"neri kenu")
    assert delayed.status == "EMIT"
    assert delayed.emitted == b"tava"
    assert len(delayed.support_provenance_root_sha256s) == 2
    assert remembered.respond(b"kenu tava").emitted == b"neri"

    write_off = brain.consolidate(
        fact_documents,
        ablation=NeutralFrameAblation.FACT_CONSOLIDATION_OFF,
    )
    assert write_off.training_summary.consolidated_fact_count == 0
    assert write_off.respond(b"neri kenu").stopped
    retrieval_off = remembered.respond(
        b"neri kenu",
        ablation=NeutralFrameAblation.FACT_RETRIEVAL_OFF,
    )
    assert retrieval_off.stopped
    assert retrieval_off.emitted is None

    conflicted = remembered.consolidate(
        (_document("fact-conflict", b"zora pavo neri"),)
    )
    assert conflicted.training_summary.consolidated_fact_count == 2
    delayed_conflict = conflicted.respond(b"neri kenu")
    assert delayed_conflict.stopped
    assert delayed_conflict.emitted is None
    assert delayed_conflict.cell_races[-1].stop_reason == "collision"
    assert remembered.respond(b"neri kenu").emitted == b"tava"

    relabeled = start().consolidate(
        (
            _document("relabeled-zupi-1", b"fira zupi lome"),
            _document("relabeled-qari-1", b"lome qari fira"),
            _document("relabeled-zupi-2", b"dexa zupi weno"),
            _document("relabeled-qari-2", b"weno qari dexa"),
        )
    )
    assert relabeled.respond(
        b"ravo qari",
        _fresh(b"noku zupi ravo"),
    ).emitted == b"noku"

    reversed_order = (
        start()
        .consolidate(tuple(reversed(first_batch)))
        .consolidate(tuple(reversed(second_batch)))
    )
    assert reversed_order.checkpoint_sha256 == brain.checkpoint_sha256
    assert reversed_order.respond(
        b"neri kenu",
        tuple(reversed(_fresh(b"tava pavo neri", b"junk tokens remain"))),
    ).emitted == b"tava"

    conflict = brain.respond(
        b"neri kenu",
        _fresh(b"tava pavo neri", b"zora pavo neri"),
    )
    assert conflict.stopped
    assert conflict.emitted is None
    assert conflict.cell_races[-1].stop_reason == "collision"
    assert conflict.cell_races[-1].winner_cell_id is None

    shuffled = start().consolidate(
        (
            _document("shuffled-pavo-1", b"moru pavo orli"),
            _document("shuffled-pavo-2", b"sena pavo ulon"),
            _document("shuffled-kenu-1", b"orli kenu sena"),
            _document("shuffled-kenu-2", b"ulon kenu moru"),
        )
    )
    assert shuffled.training_summary.promoted_frame_count == 2
    assert shuffled.training_summary.directional_route_count == 0
    assert shuffled.respond(b"neri kenu", _fresh(b"tava pavo neri")).stopped

    promotion_off = start().consolidate(
        (*first_batch, *second_batch),
        ablation=NeutralFrameAblation.FRAME_PROMOTION_OFF,
    )
    association_off = start().consolidate(
        (*first_batch, *second_batch),
        ablation=NeutralFrameAblation.ASSOCIATION_LEARNING_OFF,
    )
    assert promotion_off.respond(b"neri kenu", _fresh(b"tava pavo neri")).stopped
    assert association_off.respond(b"neri kenu", _fresh(b"tava pavo neri")).stopped

    for ablation in (
        NeutralFrameAblation.FRAME_MATCH_OFF,
        NeutralFrameAblation.READOUT_OFF,
        NeutralFrameAblation.ALIGNMENT_OFF,
        NeutralFrameAblation.MOTOR_OFF,
    ):
        decision = brain.respond(
            b"neri kenu",
            _fresh(b"tava pavo neri"),
            ablation=ablation,
        )
        assert decision.stopped, ablation
        assert decision.emitted is None
        assert decision.cell_races[-1].population == "response/motor"
