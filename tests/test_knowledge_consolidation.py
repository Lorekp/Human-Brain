from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import Parameter, signature
from pathlib import Path

import humanbrain.knowledge_consolidation as consolidation
import pytest
from humanbrain.knowledge_consolidation import (
    MAX_SELECTED_FACTORS_PER_CARD,
    KnowledgeConsolidationError,
    consolidate_knowledge_corpus_v1,
)
from humanbrain.knowledge_corpus import (
    BenchmarkDenyManifest,
    KnowledgeCorpus,
    KnowledgeCorpusCaps,
    KnowledgeCorpusManifest,
    KnowledgeFileSpec,
    KnowledgeSourceSpec,
    build_local_knowledge_corpus,
)


def _build_corpus(
    tmp_path: Path,
    files: tuple[tuple[str, str | None, str, bytes], ...],
) -> KnowledgeCorpus:
    specs: list[KnowledgeFileSpec] = []
    for path, chapter, symbol, raw in files:
        local_path = tmp_path / path
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
                source_id="test-docs",
                canonical_url="https://docs.example.org/test/",
                upstream_revision="test-revision-v1",
                license_spdx="MIT",
                license_evidence_url="https://docs.example.org/license",
                attribution="Test Documentation Authors",
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
        {"test-docs": tmp_path},
        BenchmarkDenyManifest(),
        caps,
    )


def test_consolidation_is_deterministic_scope_discriminated_and_provenance_bound(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(
        tmp_path,
        (
            (
                "chapter-one.txt",
                "shared-scope",
                "rust.alpha",
                b"common alpha first\n\nduplicate common exact\n",
            ),
            (
                "chapter-two.txt",
                "shared-scope",
                "rust.beta",
                b"common beta second\n\nduplicate common exact\n",
            ),
            (
                "shared-scope",
                None,
                "rust.path",
                b"pathfactor common one\n\npathfactor common two\n",
            ),
        ),
    )

    first = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    second = consolidate_knowledge_corpus_v1(corpus, split="acquire")

    assert first == second
    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert first.input_records_sha256 == corpus.receipt.records_sha256
    assert first.group_count == first.scope_count == first.card_count == 2
    assert tuple((card.scope_kind, card.scope_value) for card in first.cards) == (
        ("chapter", "shared-scope"),
        ("path", "shared-scope"),
    )

    chapter_card, path_card = first.cards
    assert chapter_card.declared_symbols == ("rust.alpha", "rust.beta")
    assert path_card.declared_symbols == ("rust.path",)
    assert len(chapter_card.file_sha256s) == 2
    assert len(path_card.file_sha256s) == 1
    assert "tok:duplicate" not in chapter_card.selected_factors

    common = next(factor for factor in chapter_card.factors if factor.factor == "tok:common")
    assert common.support == 3
    assert len(common.supporting_exact_dedup_cluster_sha256s) == 3
    assert len(common.supporting_member_chunk_ids) == 4
    assert len(common.supporting_member_sha256s) == 4
    first.authenticate(corpus)


def test_consolidation_caps_and_ranks_without_task_inputs(
    tmp_path: Path,
) -> None:
    recurring = " ".join(f"factor{index:03d}" for index in range(129))
    corpus = _build_corpus(
        tmp_path,
        (
            (
                "factors.txt",
                None,
                "rust.factors",
                (
                    f"priority {recurring} left\n\n"
                    f"priority {recurring} right\n\n"
                    "priority third\n"
                ).encode(),
            ),
        ),
    )

    result = consolidate_knowledge_corpus_v1(corpus, split="acquire")

    parameters = signature(consolidate_knowledge_corpus_v1).parameters
    assert tuple(parameters) == ("corpus", "split")
    assert parameters["split"].kind is Parameter.KEYWORD_ONLY
    assert len(result.cards) == 1
    assert len(result.cards[0].selected_factors) == MAX_SELECTED_FACTORS_PER_CARD
    assert result.cards[0].selected_factors == tuple(
        sorted(
            (
                "tok:priority",
                *(f"tok:factor{index:03d}" for index in range(127)),
            )
        )
    )
    priority = next(
        factor
        for factor in result.cards[0].factors
        if factor.factor == "tok:priority"
    )
    assert priority.support == 3
    assert "tok:factor127" not in result.cards[0].selected_factors
    assert "tok:factor128" not in result.cards[0].selected_factors


def test_consolidation_rejects_a_cross_corpus_concept_card_splice(
    tmp_path: Path,
) -> None:
    first_corpus = _build_corpus(
        tmp_path / "first",
        (
            (
                "facts.txt",
                None,
                "rust.facts",
                b"shared factor one\n\nshared factor two\n",
            ),
        ),
    )
    second_corpus = _build_corpus(
        tmp_path / "second",
        (
            (
                "facts.txt",
                None,
                "rust.facts",
                b"different factor one\n\ndifferent factor two\n",
            ),
        ),
    )
    first = consolidate_knowledge_corpus_v1(first_corpus, split="acquire")
    second = consolidate_knowledge_corpus_v1(second_corpus, split="acquire")
    spliced = replace(first, cards=second.cards)

    with pytest.raises(
        KnowledgeConsolidationError,
        match=r"factor receipt|concept card",
    ):
        spliced.authenticate(first_corpus)


def test_corpus_bound_authentication_rejects_redigested_source_metadata(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(
        tmp_path,
        (
            (
                "facts.txt",
                None,
                "rust.facts",
                b"shared factor one\n\nshared factor two\n",
            ),
        ),
    )
    result = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    source_draft = replace(
        result.sources[0],
        canonical_url="https://attacker.invalid/forged",
        sha256="",
    )
    source = replace(
        source_draft,
        sha256=consolidation._digest(
            consolidation._SOURCE_NAMESPACE,
            consolidation._source_payload(source_draft),
        ),
    )
    sources = (source,)
    split_sha256 = consolidation._digest(
        consolidation._SPLIT_NAMESPACE,
        consolidation._split_payload(
            result.split,
            result.members,
            result.files,
            result.groups,
            sources,
        ),
    )
    cards = []
    for card in result.cards:
        card_draft = replace(
            card,
            source_sha256=source.sha256,
            split_sha256=split_sha256,
            sha256="",
        )
        cards.append(
            replace(
                card_draft,
                sha256=consolidation._digest(
                    consolidation._CARD_NAMESPACE,
                    consolidation._card_payload(card_draft),
                ),
            )
        )
    forged_draft = replace(
        result,
        sources=sources,
        split_sha256=split_sha256,
        cards=tuple(cards),
        corpus_sha256="",
        producer_sha256="",
    )
    forged_with_corpus = replace(
        forged_draft,
        corpus_sha256=consolidation._digest(
            consolidation._CORPUS_NAMESPACE,
            consolidation._corpus_payload(forged_draft),
        ),
    )
    forged = replace(
        forged_with_corpus,
        producer_sha256=consolidation._digest(
            consolidation._PRODUCER_NAMESPACE,
            consolidation._producer_payload(forged_with_corpus),
        ),
    )

    forged.authenticate_structure()
    with pytest.raises(KnowledgeConsolidationError, match="source provenance"):
        forged.authenticate(corpus)


def test_consolidation_rejects_a_tampered_input_receipt(tmp_path: Path) -> None:
    corpus = _build_corpus(
        tmp_path,
        (
            (
                "facts.txt",
                None,
                "rust.facts",
                b"shared factor one\n\nshared factor two\n",
            ),
        ),
    )
    tampered = replace(
        corpus,
        receipt=replace(
            corpus.receipt,
            input_manifest_sha256="0" * 64,
            raw_bytes=-1,
            sources=(),
        ),
    )

    with pytest.raises(KnowledgeConsolidationError, match="corpus authentication"):
        consolidate_knowledge_corpus_v1(tampered, split="acquire")
