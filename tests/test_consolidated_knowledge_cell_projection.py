from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import Parameter, signature
from pathlib import Path

import pytest
from humanbrain.consolidated_knowledge_cell_projection import (
    ConsolidatedKnowledgeCellProjectionError,
    project_consolidated_knowledge_to_cells_v1,
)
from humanbrain.knowledge_consolidation import consolidate_knowledge_corpus_v1
from humanbrain.knowledge_corpus import (
    BenchmarkDenyManifest,
    KnowledgeCorpus,
    KnowledgeCorpusCaps,
    KnowledgeCorpusManifest,
    KnowledgeFileSpec,
    KnowledgeSourceSpec,
    build_local_knowledge_corpus,
)
from tools import build_rust_book_knowledge_corpus as rust_book


def _build_corpus(
    root: Path,
    files: tuple[tuple[str, str, str, bytes], ...],
) -> KnowledgeCorpus:
    specs: list[KnowledgeFileSpec] = []
    for path, chapter, symbol, raw in files:
        local_path = root / path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(raw)
        specs.append(
            KnowledgeFileSpec(
                path=path,
                expected_raw_sha256=sha256(raw).hexdigest(),
                split="acquire",
                content_type="plain-text",
                chapter=chapter,
                symbol=symbol,
            )
        )
    manifest = KnowledgeCorpusManifest(
        sources=(
            KnowledgeSourceSpec(
                source_id="projection-test-docs",
                canonical_url="https://docs.example.org/projection/",
                upstream_revision="projection-test-v1",
                license_spdx="MIT",
                license_evidence_url="https://docs.example.org/license",
                attribution="Projection Test Authors",
                provenance_confidence=1.0,
                files=tuple(specs),
            ),
        )
    )
    caps = KnowledgeCorpusCaps(
        max_bytes_per_source=1_000_000,
        max_total_bytes=1_000_000,
        max_tokenish_per_source=250_000,
        max_total_tokenish=250_000,
    )
    return build_local_knowledge_corpus(
        manifest,
        {"projection-test-docs": root},
        BenchmarkDenyManifest(),
        caps,
    )


def _two_concept_corpus(
    root: Path,
    *,
    chapters: tuple[str, str],
) -> KnowledgeCorpus:
    return _build_corpus(
        root,
        (
            (
                "alpha.txt",
                chapters[0],
                "rust.audit.alpha",
                b"alpha common first\n\nalpha common second\n",
            ),
            (
                "beta.txt",
                chapters[1],
                "rust.audit.beta",
                b"beta common first\n\nbeta common second\n",
            ),
        ),
    )


def test_real_rust_consolidation_projects_every_card_without_audit_labels() -> None:
    archive = rust_book.DEFAULT_CACHE_DIR / rust_book.ARCHIVE_PATH
    if not archive.is_file():
        pytest.skip("ignored authenticated Rust Book cache is unavailable")

    corpus, _ = rust_book.build_from_cache()
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    projection = project_consolidated_knowledge_to_cells_v1(
        consolidation,
        corpus=corpus,
    )

    parameters = signature(project_consolidated_knowledge_to_cells_v1).parameters
    assert tuple(parameters) == ("consolidation", "corpus")
    assert parameters["corpus"].kind is Parameter.KEYWORD_ONLY
    assert projection.concept_count == len(consolidation.cards) == 10
    assert projection.factor_association_count == 1_280
    assert projection.distinct_factor_count == 791
    assert sum(len(concept.content_factors) for concept in projection.concepts) == 1_280
    assert projection.graph.event_count == projection.concept_count == 10
    assert len({event.sha256 for event in projection.graph.event_roots}) == 10
    assert projection.graph.carrier_count == 2
    assert projection.graph.incidence_count == 11
    assert projection.producer_receipt.source_sha256 == projection.provenance_sha256

    declared = {symbol for card in consolidation.cards for symbol in card.declared_symbols}
    assert declared
    assert declared.isdisjoint(concept.concept_symbol for concept in projection.concepts)
    assert all(
        concept.concept_symbol.startswith("knowledge.concept.c") for concept in projection.concepts
    )
    projection.authenticate(consolidation, corpus=corpus)


def test_chapter_rename_changes_only_audit_projection_roots(tmp_path: Path) -> None:
    original_corpus = _two_concept_corpus(
        tmp_path / "original",
        chapters=("chapter-alpha", "chapter-beta"),
    )
    renamed_corpus = _two_concept_corpus(
        tmp_path / "renamed",
        chapters=("renamed-zeta", "renamed-eta"),
    )
    original_consolidation = consolidate_knowledge_corpus_v1(
        original_corpus,
        split="acquire",
    )
    renamed_consolidation = consolidate_knowledge_corpus_v1(
        renamed_corpus,
        split="acquire",
    )

    original = project_consolidated_knowledge_to_cells_v1(
        original_consolidation,
        corpus=original_corpus,
    )
    renamed = project_consolidated_knowledge_to_cells_v1(
        renamed_consolidation,
        corpus=renamed_corpus,
    )

    assert original.concepts == renamed.concepts
    assert original.cognition_corpus_sha256 == renamed.cognition_corpus_sha256
    assert original.graph == renamed.graph
    assert original.concept_receipts != renamed.concept_receipts
    assert original.provenance_sha256 != renamed.provenance_sha256
    assert original.sha256 != renamed.sha256
    renamed.authenticate_structure()
    with pytest.raises(
        ConsolidatedKnowledgeCellProjectionError,
        match="not exactly derived",
    ):
        renamed.authenticate(
            original_consolidation,
            corpus=original_corpus,
        )


def test_projection_rejects_tamper_and_cross_consolidation_splice(
    tmp_path: Path,
) -> None:
    first_corpus = _two_concept_corpus(
        tmp_path / "first",
        chapters=("first-alpha", "first-beta"),
    )
    second_corpus = _two_concept_corpus(
        tmp_path / "second",
        chapters=("second-alpha", "second-beta"),
    )
    first = consolidate_knowledge_corpus_v1(first_corpus, split="acquire")
    second = consolidate_knowledge_corpus_v1(second_corpus, split="acquire")
    projection = project_consolidated_knowledge_to_cells_v1(
        first,
        corpus=first_corpus,
    )

    with pytest.raises(
        ConsolidatedKnowledgeCellProjectionError,
        match="provenance root",
    ):
        replace(
            projection,
            provenance_sha256="0" * 64,
        ).authenticate_structure()

    spliced = replace(first, cards=(second.cards[0], first.cards[1]))
    with pytest.raises(
        ConsolidatedKnowledgeCellProjectionError,
        match="corpus-bound knowledge consolidation",
    ):
        project_consolidated_knowledge_to_cells_v1(
            spliced,
            corpus=first_corpus,
        )


def test_distinct_cards_with_same_factor_tuple_fail_closed(tmp_path: Path) -> None:
    corpus = _build_corpus(
        tmp_path,
        (
            (
                "punctuation-a.txt",
                "punctuation-a",
                "rust.audit.a",
                b"same token!\n\nsame token?\n",
            ),
            (
                "punctuation-b.txt",
                "punctuation-b",
                "rust.audit.b",
                b"same token.\n\nsame token,\n",
            ),
        ),
    )
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    assert len(consolidation.cards) == 2
    assert consolidation.cards[0].selected_factors == consolidation.cards[1].selected_factors

    with pytest.raises(
        ConsolidatedKnowledgeCellProjectionError,
        match="duplicate cognition factor tuples",
    ):
        project_consolidated_knowledge_to_cells_v1(
            consolidation,
            corpus=corpus,
        )
