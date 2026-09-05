"""Fixed cumulative public-QA data custody; no encoding, learning or execution.

Labels remain in the external teaching/evaluator view. Semantic admission is a
separate source-referenced review, never inferred from these mechanical checks.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.r4_integrated.public_qa_data import file_hash, load_selection, text_hash

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "experiments/r4_integrated/fresh_source_cases_v1.json"
SELECTION = "experiments/r4_integrated/public_qa_selection_v1.json"
OLD_FIXTURE = "experiments/r4_integrated/real_input_qualification_cases_v1.json"
PARENT_REVISION = "2eb0fb19bacbf1e6b08b265840bba2b9e70f7d32"
PINS = {
    SELECTION: "72fa7ac873643be36701c486a66c24ae67631fcd95c6fe18c2021539dd3cfe76",
    OLD_FIXTURE: "aa988986d8f04d7e796e485ccea33f24c96cadad09f03779de844905ee5ad41b",
    "experiments/r4_integrated/public_qa_data.py": (
        "36403b86f462201d529f4ec6a24fc297e75ba3641c10f1168586f157005e0bd6"
    ),
    ".knowledge-cache/public_qa/squad2/train-v2.0.json": (
        "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002"
    ),
    "benchmarks/fixed_rehearsal_v1.json": (
        "a219ece11778ae7b9ded22e1103fb33556baf50f8aebbf66e4deef4c3312779d"
    ),
    "benchmarks/fixed_rehearsal_v1.completion.json": (
        "f6ec3c827009b7a1c33b99323bfd8abed1d4ec170e299ec2b04d01ecc4b135c3"
    ),
    "benchmarks/fixed_rehearsal_v1.launch.json": (
        "6d2216fa6390d947268740b02ad23ad1bce161b8b18d2c297f2b3afb766fb711"
    ),
}
SLOTS = tuple((f"E{i}", i in (3, 7)) for i in range(8)) + tuple(
    (f"L{i}", i in (1, 3)) for i in range(4)
)


def _read(path: str) -> dict:
    return json.loads((ROOT / path).read_bytes())


def select_fresh(selection: dict, old: dict) -> tuple[list[dict], list[dict]]:
    """Choose the prescribed first available row, without semantic filtering."""
    documents = selection["documents"]
    used_paragraphs = {r["paragraph_id"] for r in old["cases"]}
    used_articles = {documents[p]["article_sha256"] for p in used_paragraphs}
    used_contexts = {documents[p]["context_sha256"] for p in used_paragraphs}
    rows, trace = [], []
    for slot, label in SLOTS:
        skipped = []
        candidates = sorted(
            (r for r in selection["probe"] if r["is_impossible"] == label),
            key=lambda r: (text_hash(r["id"]), r["id"]),
        )
        for row in candidates:
            pid = row["paragraph_id"]
            doc = documents[pid]
            reasons = []
            for reason, value, used in (
                ("used_article", doc["article_sha256"], used_articles),
                ("used_paragraph", pid, used_paragraphs),
                ("used_context", doc["context_sha256"], used_contexts),
            ):
                if value in used:
                    reasons.append(reason)
            if reasons:
                skipped.append({"id": row["id"], "reasons": reasons})
                continue
            rows.append(row)
            trace.append({"slot": slot, "selected_id": row["id"], "skipped": skipped})
            used_articles.add(doc["article_sha256"])
            used_paragraphs.add(pid)
            used_contexts.add(doc["context_sha256"])
            break
        else:
            raise ValueError(f"Fixed selection exhausted at {slot}; no replacement policy")
    return rows, trace


def prepare_inputs() -> dict:
    """Reconstruct the fixed manifest from authenticated existing local bytes."""
    for path, expected in PINS.items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"Source identity differs: {path}")
    selection = load_selection(ROOT / SELECTION, PINS[SELECTION])
    old = _read(OLD_FIXTURE)
    if old["cases"] != selection["train"][:16]:
        raise ValueError("Parent questions differ from the original first sixteen")
    parent = _read("benchmarks/fixed_rehearsal_v1.json")
    completion = _read("benchmarks/fixed_rehearsal_v1.completion.json")
    launch = _read("benchmarks/fixed_rehearsal_v1.launch.json")
    if (
        parent["source"]["revision"] != PARENT_REVISION
        or parent["status"] != "completed"
        or not parent["decision"]["qualified"]
        or completion["status"] != "passed"
        or completion["report_sha256"] != PINS["benchmarks/fixed_rehearsal_v1.json"]
        or launch["status"] != "passed"
        or launch["child_completion"] != completion
    ):
        raise ValueError("Parent completion chain differs")
    for path in (SELECTION, OLD_FIXTURE, "experiments/r4_integrated/public_qa_data.py"):
        if parent["source"]["files"][path] != PINS[path]:
            raise ValueError("Selected source differs from the frozen parent")
    for record in (old["encoded_inputs"], parent["checkpoint_archive"]):
        if file_hash(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Saved input/checkpoint archive differs")
    fresh, trace = select_fresh(selection, old)
    cases = old["cases"] + fresh
    order = [r["paragraph_id"] for r in cases]
    documents = {p: selection["documents"][p] for p in order}
    if len(documents) != 28:
        raise ValueError("Cumulative paragraph inventory is not twenty-eight")
    blocks = {
        "O": [r["id"] for r in cases[:16]],
        "E": [r["id"] for r in fresh[:8]],
        "L": [r["id"] for r in fresh[8:]],
    }
    return {
        "schema": "fresh_source_action_cases_v1",
        "status": "aborted_semantic_review",
        "admission": {
            "allowed": False,
            "slot": "E3",
            "reason": (
                "America is ambiguous between the US and the continent; the original source "
                "explicitly describes Canadian representatives. Original STOP cannot resolve"
                " that wording without rewriting. No replacement selection is authorized."
            ),
        },
        "exposure": (
            "Original public TRAIN; project-exposed probe questions absent from these "
            "parents' sixteen-task experience; not independent evaluation."
        ),
        "attribution": selection["attribution"],
        "source_custody": {
            "files": PINS,
            "original_download": selection["source"],
            "parent_revision": PARENT_REVISION,
            "parent_checkpoint_archive": parent["checkpoint_archive"],
        },
        "cases": cases,
        "blocks": blocks,
        "documents": documents,
        "document_order": order,
        "task_slots": {
            qa_id: f"{block}{i}" for block, ids in blocks.items() for i, qa_id in enumerate(ids)
        },
        "selection_trace": trace,
        "old_saved_inputs": old["saved_inputs"],
        "old_input_links": old["input_links"],
        "old_encoded_inputs": old["encoded_inputs"],
    }


def load_inputs(path: Path = FIXTURE, expected_sha256: str | None = None) -> dict:
    """Reject this aborted cohort; retained solely to prevent accidental admission."""
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError("Fresh-source fixture identity differs")
    actual = json.loads(path.read_bytes())
    if actual != prepare_inputs():
        raise ValueError("Fresh-source fixture differs from its fixed original data")
    raise ValueError("Cohort aborted by independent semantic review; no session admission")
