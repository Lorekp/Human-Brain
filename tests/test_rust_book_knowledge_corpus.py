from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools import build_rust_book_knowledge_corpus as builder


LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "knowledge"
    / "rust-book-nostarch-third-printing.lock.json"
)


def test_rust_book_lock_authenticates_exact_raw_markdown_allowlist(
    tmp_path: Path,
) -> None:
    payload, raw = builder.load_lock(LOCK_PATH)
    assert raw == builder.canonical_lock_bytes(payload)
    assert payload["source"]["release_tag"] == "nostarch-third-printing"
    assert (
        payload["source"]["upstream_revision"]
        == "844ddcd4d6d7726143ed8165950e395ccfd1f155"
    )
    assert payload["source"]["license_expression_spdx"] == "MIT OR Apache-2.0"
    assert payload["source"]["license_spdx"] == "MIT"
    assert [license_["spdx"] for license_ in payload["source"]["licenses"]] == [
        "Apache-2.0",
        "MIT",
    ]
    assert payload["raw_markdown_policy"] == builder.RAW_MARKDOWN_POLICY

    rows = payload["files"]
    assert [row["path"] for row in rows] == list(builder.FILE_METADATA)
    assert sum(row["bytes"] for row in rows) == 185_367
    assert {row["content_type"] for row in rows} == {"markdown"}
    assert {row["split"] for row in rows} == {"acquire"}
    assert all("text" not in row and "content" not in row for row in rows)
    symbols = {row["symbol"] for row in rows}
    assert {
        "rust.concurrent.message_passing",
        "rust.concurrent.shared_state",
    }.issubset(symbols)

    manifest, deny_manifest, caps = builder.corpus_inputs_from_lock(payload)
    assert len(manifest.sources) == 1
    assert len(manifest.sources[0].files) == 10
    assert manifest.sources[0].license_spdx == "MIT"
    assert deny_manifest.domains == ("huggingface.co", "swebench.com")
    assert caps.max_total_bytes == 190_000

    tampered = copy.deepcopy(payload)
    tampered["files"][-1]["symbol"] = "rust.concurrent.hidden_answer"
    tampered_path = tmp_path / "tampered.lock.json"
    tampered_path.write_bytes(builder.canonical_lock_bytes(tampered))
    with pytest.raises(builder.RustBookLockError, match="allowlist differs"):
        builder.load_lock(tampered_path)

    with pytest.raises(builder.RustBookLockError, match="local archive is absent"):
        builder.build_from_cache(lock_path=LOCK_PATH, cache_dir=tmp_path)
