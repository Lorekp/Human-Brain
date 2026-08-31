from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from humanbrain.knowledge_corpus import (
    BenchmarkDenyManifest,
    KnowledgeCorpusCapacityError,
    KnowledgeCorpusCaps,
    KnowledgeCorpusIntegrityError,
    KnowledgeCorpusManifest,
    KnowledgeCorpusPolicyError,
    KnowledgeFileSpec,
    KnowledgeSourceSpec,
    build_local_knowledge_corpus,
)


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _fixture(tmp_path: Path) -> tuple[KnowledgeCorpusManifest, dict[str, Path], KnowledgeCorpusCaps]:
    alpha_raw = b"Algorithms are reusable.\r\n\r\n```python\r\nx = 1\r\n```\r\n"
    beta_raw = b"Data structures organize state.\n\n```python\nx = 1\n```\n"
    alpha_root = tmp_path / "alpha"
    alpha_file = alpha_root / "guide.md"
    alpha_root.mkdir()
    alpha_file.write_bytes(alpha_raw)
    beta_file = tmp_path / "renamed-local-copy.md"
    beta_file.write_bytes(beta_raw)

    manifest = KnowledgeCorpusManifest(
        sources=(
            KnowledgeSourceSpec(
                source_id="alpha-docs",
                canonical_url="https://docs.example.org/algorithms/",
                upstream_revision="v1.2.3",
                license_spdx="MIT",
                license_evidence_url="https://docs.example.org/license",
                attribution="Alpha Authors, Algorithms Guide v1.2.3",
                provenance_confidence=1.0,
                files=(
                    KnowledgeFileSpec(
                        path="guide.md",
                        expected_raw_sha256=_sha(alpha_raw),
                        split="acquire",
                        content_type="markdown",
                        chapter="Algorithms",
                        symbol="reverse",
                    ),
                ),
            ),
            KnowledgeSourceSpec(
                source_id="beta-docs",
                canonical_url="https://reference.example.net/data/",
                upstream_revision="8f5d918b6d7fdd08a7db6cba8a607c99d4fe8a31",
                license_spdx="Apache-2.0",
                license_evidence_url="https://reference.example.net/license",
                attribution="Beta Foundation, Data Reference",
                provenance_confidence=0.95,
                files=(
                    KnowledgeFileSpec(
                        path="upstream/data.md",
                        expected_raw_sha256=_sha(beta_raw),
                        split="acquire",
                        content_type="markdown",
                        chapter="Data structures",
                        symbol="array",
                    ),
                ),
            ),
        )
    )
    caps = KnowledgeCorpusCaps(
        max_bytes_per_source=1024,
        max_total_bytes=2048,
        max_tokenish_per_source=256,
        max_total_tokenish=512,
    )
    return manifest, {"alpha-docs": alpha_root, "beta-docs": beta_file}, caps


def test_local_corpus_is_deterministic_provenance_bound_and_exact_deduplicated(
    tmp_path: Path,
) -> None:
    manifest, sources, caps = _fixture(tmp_path)
    deny = BenchmarkDenyManifest(paths=("benchmarks/heldout.py",))

    first = build_local_knowledge_corpus(manifest, sources, deny, caps)
    second = build_local_knowledge_corpus(
        manifest, dict(reversed(tuple(sources.items()))), deny, caps
    )

    assert first.canonical_manifest_bytes() == second.canonical_manifest_bytes()
    assert first.canonical_jsonl_bytes() == second.canonical_jsonl_bytes()
    assert first.receipt.chunk_count == 4
    assert first.receipt.exact_dedup_cluster_count == 3
    code = [chunk for chunk in first.chunks if chunk.kind == "code-block"]
    assert len(code) == 2
    assert code[0].exact_dedup_cluster_sha256 == code[1].exact_dedup_cluster_sha256
    assert code[0].exact_dedup_cluster_size == code[1].exact_dedup_cluster_size == 2
    assert code[0].exact_dedup_representative_chunk_id == code[0].chunk_id
    assert code[1].exact_dedup_representative_chunk_id == code[0].chunk_id
    assert code[0].source_file_raw_sha256 != code[0].source_file_normalized_sha256
    assert all(chunk.benchmark_deny_manifest_sha256 == first.receipt.benchmark_deny_manifest_sha256 for chunk in first.chunks)

    changed_source = replace(
        manifest.sources[0], attribution="Alpha Authors, revised attribution"
    )
    rebound = build_local_knowledge_corpus(
        replace(manifest, sources=(changed_source, manifest.sources[1])),
        sources,
        deny,
        caps,
    )
    assert rebound.receipt.input_manifest_sha256 != first.receipt.input_manifest_sha256
    assert tuple(chunk.chunk_id for chunk in rebound.chunks) != tuple(
        chunk.chunk_id for chunk in first.chunks
    )

    alpha_path = sources["alpha-docs"] / "guide.md"
    alpha_path.write_bytes(alpha_path.read_bytes() + b"tamper\n")
    with pytest.raises(KnowledgeCorpusIntegrityError, match="differ from allowlist"):
        build_local_knowledge_corpus(manifest, sources, deny, caps)


def test_hard_source_and_total_caps_reject_before_output(tmp_path: Path) -> None:
    manifest, sources, caps = _fixture(tmp_path)
    source_bytes = len((sources["alpha-docs"] / "guide.md").read_bytes())
    with pytest.raises(KnowledgeCorpusCapacityError, match="source byte cap"):
        build_local_knowledge_corpus(
            manifest,
            sources,
            BenchmarkDenyManifest(),
            replace(caps, max_bytes_per_source=source_bytes - 1),
        )
    beta_bytes = len(sources["beta-docs"].read_bytes())
    one_source_cap = max(source_bytes, beta_bytes)
    with pytest.raises(KnowledgeCorpusCapacityError, match="total byte cap"):
        build_local_knowledge_corpus(
            manifest,
            sources,
            BenchmarkDenyManifest(),
            KnowledgeCorpusCaps(
                max_bytes_per_source=one_source_cap,
                max_total_bytes=one_source_cap,
                max_tokenish_per_source=256,
                max_total_tokenish=512,
            ),
        )


@pytest.mark.parametrize("deny_kind", ["path", "domain", "hash"])
def test_benchmark_deny_manifest_rejects_named_content(
    tmp_path: Path, deny_kind: str
) -> None:
    manifest, sources, caps = _fixture(tmp_path)
    alpha_hash = manifest.sources[0].files[0].expected_raw_sha256
    deny = {
        "path": BenchmarkDenyManifest(paths=("guide.md",)),
        "domain": BenchmarkDenyManifest(domains=("example.org",)),
        "hash": BenchmarkDenyManifest(sha256=(alpha_hash,)),
    }[deny_kind]
    with pytest.raises(KnowledgeCorpusPolicyError, match="benchmark-denied"):
        build_local_knowledge_corpus(manifest, sources, deny, caps)


@pytest.mark.parametrize(
    "license_spdx",
    ["", "NOASSERTION", "LicenseRef-custom", "CC-BY-NC-ND-4.0", "CC-BY-SA-4.0", "GPL-3.0-only"],
)
def test_permissive_lane_rejects_unknown_restricted_and_copyleft_licenses(
    tmp_path: Path, license_spdx: str
) -> None:
    manifest, sources, caps = _fixture(tmp_path)
    rejected = replace(manifest.sources[0], license_spdx=license_spdx)
    with pytest.raises(KnowledgeCorpusPolicyError, match="permissive lane"):
        build_local_knowledge_corpus(
            replace(manifest, sources=(rejected, manifest.sources[1])),
            sources,
            BenchmarkDenyManifest(),
            caps,
        )
