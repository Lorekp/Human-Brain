"""Frozen-artifact Python-QA corpus bridge for ``benchmarks/qa_experiments``.

The QA harness grows/evaluates whatever a corpus handle serves through the
``qa_corpus`` surface (``build_curriculum`` / ``QATask`` / ``prompt_for_task``
/ ``growth_text`` / ``CORPUS_SOURCE`` / ``corpus_digest``). The Bible arm is
served straight from ``humanbrain.qa_corpus``; THIS module is the equivalent
handle for the raw-Python-text coding corpus (M0/M1 channel), with three
deliberate differences from ``humanbrain.python_qa_corpus``'s own
``build_curriculum``:

1. **Frozen, not live.** ``python_qa_corpus.build_curriculum`` mines the
   CURRENT ``src/humanbrain`` tree, so its digest drifts whenever a module
   docstring changes after a corpus pin (it already has: T-ROUTE added
   ``complete_broadcast``, which the pinned artifact cannot contain). The M1
   n-gram comparators are sha256-keyed to the PINNED artifact
   (``experiments/python_qa_corpus/python_qa_corpus_v1.json``, corpus digest
   ``789e3ac592fd8aa1d4f68af8704a4a9b19d64db7277efea59ebc50ff8dd446ec``), so
   this handle loads and verifies that exact file — artifact bytes checked
   against ``manifest.json`` and every entry re-validated — instead of
   re-mining. Cell scores stay comparable to the locked n-gram records.

2. **Full-cohort evaluation.** The harness evaluates a corpus handle's
   ``calibration_tasks``; the n-gram baselines scored ALL 25 VAL items. This
   handle therefore returns the whole TRAIN cohort (75) as ``train_tasks``
   and the whole VAL cohort (25) as ``calibration_tasks`` — the fixed BY-FILE
   split of the artifact, not a seed-carved CALIBRATION slice of TRAIN (VAL
   files can never leak into TRAIN by the split rule, so no seed keying of
   the split is needed or permitted).

3. **Identical framing.** Prompts reuse ``humanbrain.qa_corpus.qa_prompt``
   verbatim (prompt-last, ends ``answer:``) — byte-identical to both the
   Bible arm and ``tools/python_qa_baselines.py``'s replica, so the
   teacher-forced answer-span seam behaves identically everywhere.

Cohort labels are preserved as the artifact spells them (``TRAIN``/``VAL``);
records therefore show which cohort an evaluated task came from without a
rename indirection.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from humanbrain.python_qa_corpus import (
    FAMILIES,
    SCHEMA_VERSION,
    _validate_entry_record,
    canonical_json,
)
from humanbrain.qa_corpus import QATask, growth_text, prompt_for_task

CORPUS_SOURCE = "repo-src:humanbrain:python-qa-corpus-v1"

__all__ = [
    "CORPUS_SOURCE",
    "PINNED_CORPUS_SHA256",
    "QATask",
    "SCHEMA_VERSION",
    "artifact_paths",
    "build_curriculum",
    "corpus_digest",
    "corpus_entries",
    "growth_text",
    "load_frozen_corpus",
    "prompt_for_task",
]

# Pinned in experiments/python_qa_baselines_v1/FINDINGS.md BEFORE any cell
# run; every record this arm produces must carry this corpus digest or it is
# not comparable to the M1 comparators.
PINNED_CORPUS_SHA256 = "789e3ac592fd8aa1d4f68af8704a4a9b19d64db7277efea59ebc50ff8dd446ec"


def _repo_root() -> Path:
    """Repo root: this package lives at ``<root>/src/humanbrain``."""
    return Path(__file__).resolve().parents[2]


def artifact_paths() -> tuple[Path, Path]:
    """(corpus JSON, manifest JSON) paths inside the repo checkout."""
    directory = _repo_root() / "experiments" / "python_qa_corpus"
    return directory / "python_qa_corpus_v1.json", directory / "manifest.json"


_FROZEN_CACHE: dict[str, object] | None = None


def load_frozen_corpus() -> dict[str, object]:
    """Load + verify the pinned artifact (cached; deterministic content).

    Verification discipline mirrors ``tools/python_qa_baselines.load_corpus``:
    artifact bytes must match ``manifest.json``'s ``artifact_sha256``, the
    schema version must be the pinned library schema, cohort counts must be
    internally consistent, the embedded ``corpus_sha256`` must equal the
    digest recomputed over the canonical entries record AND the pinned M1
    value, and every entry must pass the shared answer-derivability validator.
    """
    global _FROZEN_CACHE
    if _FROZEN_CACHE is not None:
        return _FROZEN_CACHE
    artifact_path, manifest_path = artifact_paths()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = artifact_path.read_bytes()
    if sha256(raw).hexdigest() != str(manifest["artifact_sha256"]):
        raise ValueError(
            f"corpus artifact digest mismatch: {artifact_path} does not match "
            f"manifest.json artifact_sha256 {manifest['artifact_sha256']}"
        )
    artifact = json.loads(raw.decode("utf-8"))
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected corpus schema {artifact['schema_version']!r}")
    entries: dict[str, list[dict[str, object]]] = artifact["entries"]
    for cohort in ("TRAIN", "VAL"):
        for record in entries[cohort]:
            _validate_entry_record(record)

    recomputed = sha256(canonical_json({"entries": entries})).hexdigest()
    embedded = str(artifact.get("corpus_sha256", ""))
    if embedded and embedded != recomputed:
        raise ValueError(
            f"corpus digest drift: embedded {embedded!r} != recomputed {recomputed!r}"
        )
    if recomputed != PINNED_CORPUS_SHA256:
        raise ValueError(
            "pinned python-QA corpus changed: "
            f"{recomputed!r} != M1 comparator pin {PINNED_CORPUS_SHA256!r} "
            "(re-pin the n-gram baselines before running cell arms)"
        )
    counts = artifact["counts"]
    for cohort in ("TRAIN", "VAL"):
        if int(counts[cohort]["total"]) != len(entries[cohort]):
            raise ValueError(f"manifest count mismatch for cohort {cohort}")

    _FROZEN_CACHE = artifact
    return artifact


def _frozen_task(record: dict[str, object]) -> QATask:
    """One artifact entry as a harness-native ``QATask`` (labels preserved)."""
    references = record["references"]
    context = record["context"]
    return QATask(
        cohort=str(record["cohort"]),
        qa_id=str(record["qa_id"]),
        ordinal=int(record["ordinal"]),  # type: ignore[arg-type]
        references=tuple(str(item) for item in references),  # type: ignore[arg-type]
        context=tuple(str(item) for item in context),  # type: ignore[arg-type]
        question=str(record["question"]),
        answer=str(record["answer"]),
    )


def corpus_entries() -> dict[str, tuple[QATask, ...]]:
    """Both frozen cohorts materialized as ``QATask`` tuples."""
    entries: dict[str, list[dict[str, object]]] = load_frozen_corpus()["entries"]  # type: ignore[assignment]
    return {
        "TRAIN": tuple(_frozen_task(record) for record in entries["TRAIN"]),
        "VAL": tuple(_frozen_task(record) for record in entries["VAL"]),
    }


def corpus_digest() -> str:
    """SHA-256 of the frozen corpus record — always the pinned M1 value."""
    return PINNED_CORPUS_SHA256


def build_curriculum(
    seed: int,
    *,
    calibration_task_count: int | None = None,
) -> dict[str, object]:
    """Harness-compatible curriculum over the FROZEN artifact.

    ``train_tasks`` = all 75 TRAIN items; ``calibration_tasks`` = all 25 VAL
    items (full-cohort evaluation, matching the locked n-gram baselines).
    The seed enters the identifier/order domains and the curriculum digest —
    runs at different seeds are distinct records over the same fixed split;
    it does NOT carve cohorts (see module docstring, point 2).
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    cohorts = corpus_entries()
    train_tasks = cohorts["TRAIN"]
    val_tasks = cohorts["VAL"]
    if calibration_task_count is not None and calibration_task_count != len(val_tasks):
        raise ValueError(
            "the python-QA arm evaluates the FULL frozen VAL cohort "
            f"({len(val_tasks)} items); refusing a partial slice"
        )

    identifier_domain = f"{SCHEMA_VERSION}/curriculum-identifier/frozen/seed/{seed}"
    order_domain = f"{SCHEMA_VERSION}/curriculum-order/frozen/seed/{seed}"

    def _record(task: QATask) -> dict[str, object]:
        return {
            "qa_id": task.qa_id,
            "cohort": task.cohort,
            "ordinal": task.ordinal,
            "references": list(task.references),
            "context": list(task.context),
            "question": task.question,
            "answer": task.answer,
        }

    curriculum_digest = sha256(
        canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "corpus_source": CORPUS_SOURCE,
                "corpus_sha256": corpus_digest(),
                "seed": seed,
                "identifier_domain": identifier_domain,
                "order_domain": order_domain,
                "train": [_record(task) for task in train_tasks],
                "calibration": [_record(task) for task in val_tasks],
            }
        )
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_source": CORPUS_SOURCE,
        "corpus_size": len(train_tasks) + len(val_tasks),
        "corpus_sha256": corpus_digest(),
        "seed": seed,
        "identifier_domain": identifier_domain,
        "order_domain": order_domain,
        "families": list(FAMILIES),
        "train_tasks": train_tasks,
        "calibration_tasks": val_tasks,
        "sha256": curriculum_digest,
    }
