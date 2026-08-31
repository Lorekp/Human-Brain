from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import pytest

from tools import build_python_docs_knowledge_corpus as builder


LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "knowledge"
    / "python-3.14.7-docs.lock.json"
)


def test_official_python_docs_lock_is_explicit_canonical_and_hash_frozen(
    tmp_path: Path,
) -> None:
    payload, raw = builder.load_lock(LOCK_PATH)
    assert raw == builder.canonical_lock_bytes(payload)
    assert payload["source"] == {
        "archive_bytes": 4_617_392,
        "archive_path": "python-3.14-docs-text.zip",
        "archive_sha256": "de728eb5b1de8d44ab04f234caeef0bc237bebcb8a6af8a219f35f9d9e35c0c0",
        "attribution": "Python Software Foundation, Python 3.14.7 documentation",
        "canonical_url": "https://docs.python.org/3/archives/python-3.14-docs-text.zip",
        "extracted_root": "python-3.14-docs-text/python-3.14-docs-text",
        "license_evidence_url": "https://docs.python.org/3.14/license.html",
        "license_spdx": "PSF-2.0",
        "provenance_confidence": 1.0,
        "source_id": "python-3.14.7-docs",
        "upstream_revision": "3.14.7",
    }
    assert payload["family_splits"] == {
        "acquire": [
            "deprecations",
            "distributing",
            "faq",
            "glossary",
            "installing",
            "library",
        ],
        "validation": ["c-api", "extending", "tutorial"],
        "holdout": ["howto", "reference", "using"],
        "excluded": ["meta", "whatsnew"],
    }

    rows = payload["files"]
    assert len(rows) == 536
    assert sum(row["bytes"] for row in rows) == 14_662_987
    assert Counter(row["split"] for row in rows) == {
        "acquire": 354,
        "validation": 101,
        "holdout": 49,
        "excluded": 32,
    }
    included = [row for row in rows if row["split"] != "excluded"]
    assert len(included) == 504
    assert sum(row["bytes"] for row in included) == 10_302_122
    assert all("text" not in row and "content" not in row for row in rows)
    assert all(
        token not in row["path"].casefold()
        for row in rows
        for token in payload["benchmark_policy"]["forbidden_path_tokens"]
    )

    manifest, deny_manifest, caps = builder.corpus_inputs_from_lock(payload)
    assert len(manifest.sources) == 1
    assert len(manifest.sources[0].files) == 504
    assert Counter(file.split for file in manifest.sources[0].files) == {
        "acquire": 354,
        "validation": 101,
        "holdout": 49,
    }
    assert deny_manifest.domains == ("huggingface.co", "swebench.com")
    assert caps.max_total_bytes == 10_500_000

    hash_tamper = copy.deepcopy(payload)
    hash_tamper["files"][0]["sha256"] = "0" * 64
    tampered_path = tmp_path / "hash-tampered.lock.json"
    tampered_path.write_bytes(builder.canonical_lock_bytes(hash_tamper))
    with pytest.raises(builder.PythonDocsLockError, match="inventory digest"):
        builder.load_lock(tampered_path)

    split_tamper = copy.deepcopy(payload)
    split_tamper["family_splits"]["acquire"].remove("faq")
    split_tamper["family_splits"]["validation"].append("faq")
    split_tamper["family_split_sha256"] = builder.digest(
        split_tamper["family_splits"]
    )
    split_path = tmp_path / "split-tampered.lock.json"
    split_path.write_bytes(builder.canonical_lock_bytes(split_tamper))
    with pytest.raises(builder.PythonDocsLockError, match="split assignment"):
        builder.load_lock(split_path)
