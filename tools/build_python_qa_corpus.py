#!/usr/bin/env python3
"""Python-source-text QA corpus builder → experiments/python_qa_corpus/.

Mines this repo's own ``src/humanbrain`` modules for raw-Python-text QA
triples (function_summary / module_containment / signature_recall) via
``humanbrain.python_qa_corpus``, enforces in-process determinism (two
independent builds must agree byte-for-byte), and writes the frozen corpus
plus a hashed manifest:

    experiments/python_qa_corpus/python_qa_corpus_v1.json
    experiments/python_qa_corpus/manifest.json

Deterministic: no wall-clock enters either file; rerunning on an unchanged
tree reproduces byte-identical artifacts (asserted in-process before writing,
the red-test-15 analogue from tools/build_bible_qa_corpus.py).

stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanbrain.python_qa_corpus import (  # noqa: E402
    SCHEMA_VERSION,
    FAMILIES,
    canonical_json,
    corpus_digest,
    discover_source_files,
    build_corpus,
)

MANIFEST_SCHEMA_VERSION = "humanbrain/python-qa-corpus-manifest/v1"
BUILDER = "tools/build_python_qa_corpus.py"
DEFAULT_OUT_DIR = ROOT / "experiments" / "python_qa_corpus"
CORPUS_ARTIFACT_NAME = "python_qa_corpus_v1.json"


def git_short_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def build_artifact_bytes(git_head: str) -> bytes:
    """Canonical artifact bytes; deterministic given sources + head."""
    corpus = build_corpus()
    payload = dict(corpus)
    payload["git_head"] = git_head
    payload["corpus_sha256"] = corpus_digest()
    return json.dumps(payload, indent=1, sort_keys=True).encode("utf-8") + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)

    relpaths = discover_source_files()
    print(f"mining {len(relpaths)} humanbrain modules ...")
    first_bytes = build_artifact_bytes(git_head := git_short_head())
    second_bytes = build_artifact_bytes(git_head)
    if first_bytes != second_bytes:
        print("FATAL: two independent builds disagree — generation is not deterministic")
        return 2
    digest = sha256(first_bytes).hexdigest()

    corpus_payload = json.loads(first_bytes.decode("utf-8"))
    counts = corpus_payload["counts"]
    total = counts["TRAIN"]["total"] + counts["VAL"]["total"]
    print(
        f"selected {total} triples "
        f"(train {counts['TRAIN']['total']} / val {counts['VAL']['total']}), "
        f"split BY FILE across {len(relpaths)} modules"
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder": BUILDER,
        "library_schema_version": SCHEMA_VERSION,
        "artifact": CORPUS_ARTIFACT_NAME,
        "artifact_sha256": digest,
        "corpus_sha256": corpus_digest(),
        "git_head": git_head,
        "families": list(FAMILIES),
        "counts": counts,
        "source_files": len(relpaths),
        "determinism_check_in_process": True,
    }
    manifest_bytes = json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8") + b"\n"

    out_dir: Path = args.out_dir
    write_atomic(out_dir / CORPUS_ARTIFACT_NAME, first_bytes)
    write_atomic(out_dir / "manifest.json", manifest_bytes)
    print(f"corpus   -> {out_dir / CORPUS_ARTIFACT_NAME} (sha256 {digest[:16]}…)")
    print(f"manifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
