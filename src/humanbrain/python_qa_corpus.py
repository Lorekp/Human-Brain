"""Python-source-text QA corpus for the coding-transfer experiment (R3-prep).

The M0 audit found the typed-grammar SWE channel has a 0.0 ceiling, so the
next coding-adjacent experiment runs on RAW-PYTHON-TEXT QA: the same QA
machinery as ``humanbrain.qa_corpus`` (identical ``qa_prompt`` framing,
canonical-JSON digests, seed-keyed curriculum materialization), but mined
from THIS REPOSITORY'S OWN ``src/humanbrain`` modules instead of KJV verses.

Three deterministic families (docstring/function-level granularity):

- ``function_summary``   : docstring first line → "what does X do?" answer;
- ``module_containment`` : which humanbrain module defines a public name;
- ``signature_recall``   : named parameters of a function/method.

Discipline inherited from ``qa_corpus.py`` / ``tools/build_bible_qa_corpus.py``:

- generation is fully deterministic (AST over pinned files, sha256
  presentation-order sort keys, no wall-clock anywhere);
- TRAIN/VAL split is BY FILE (every triple mined from a file lives wholly in
  one cohort — no function from a VAL file can leak into TRAIN);
- every answer occurs verbatim (case-folded) inside its context lines, so
  held-out evaluation stays mechanically gradeable;
- budgets are pinned: ~100 triples (75 TRAIN / 25 VAL) with per-family
  quotas and deterministic backfill;
- artifacts land in ``experiments/python_qa_corpus/`` via
  ``tools/build_python_qa_corpus.py`` with manifest + sha256 hashes.

stdlib-only; ``ast`` does all parsing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from humanbrain.qa_corpus import QATask, qa_prompt
from humanbrain.typed_composition import derive_opaque_identifier

SCHEMA_VERSION = "humanbrain/python-qa-corpus/v1"
CORPUS_SOURCE = "repo-src:humanbrain"
FAMILIES = ("function_summary", "module_containment", "signature_recall")

TRAIN_TARGET = 75
VAL_TARGET = 25
# Per-family TRAIN quotas (families in FAMILIES order) and VAL quotas
# (alphabetical, ceil on the last family) — sums match the cohort targets.
TRAIN_FAMILY_QUOTA = {family: TRAIN_TARGET // len(FAMILIES) for family in FAMILIES}
VAL_FAMILY_QUOTA = {
    family: VAL_TARGET // len(FAMILIES) + (1 if index == len(FAMILIES) - 1 else 0)
    for index, family in enumerate(sorted(FAMILIES))
}
VAL_FILE_FRACTION = 0.25
MIN_VAL_FILES = 3
MIN_TRAIN_FILES = 8

_SPLIT_DOMAIN = f"{SCHEMA_VERSION}/file-split"
_ORDER_DOMAIN = f"{SCHEMA_VERSION}/task-order"

MAX_SUMMARY_ANSWER_CHARS = 140


# ---------------------------------------------------------------------------
# File discovery and split
# ---------------------------------------------------------------------------


def discover_source_files(src_dir: str | Path | None = None) -> tuple[str, ...]:
    """Pinned module list: top-level ``humanbrain/*.py``, no ``__init__.py``."""
    directory = Path(src_dir) if src_dir is not None else Path(__file__).resolve().parent
    names = sorted(
        path.name
        for path in directory.glob("*.py")
        if path.name != "__init__.py" and path.is_file()
    )
    if len(names) < MIN_TRAIN_FILES + MIN_VAL_FILES:
        raise ValueError(
            f"expected ~30 humanbrain modules, found only {len(names)} under {directory}"
        )
    return tuple(names)


def _file_sort_key(relpath: str) -> bytes:
    return sha256(_SPLIT_DOMAIN.encode("utf-8") + b"\x00" + relpath.encode("utf-8")).digest()


def split_file_roles(relpaths: tuple[str, ...]) -> dict[str, str]:
    """Deterministic BY-FILE TRAIN/VAL split (~25% VAL, floor-guarded)."""
    ranked = sorted(relpaths, key=_file_sort_key)
    val_count = max(MIN_VAL_FILES, round(len(ranked) * VAL_FILE_FRACTION))
    val_count = min(val_count, len(ranked) - MIN_TRAIN_FILES)
    if val_count < MIN_VAL_FILES:
        raise ValueError("file pool too small to form both cohorts")
    roles = {relpath: ("VAL" if index < val_count else "TRAIN") for index, relpath in enumerate(ranked)}
    return roles


def read_source_files(relpaths: tuple[str, ...], src_dir: str | Path | None = None) -> dict[str, str]:
    directory = Path(src_dir) if src_dir is not None else Path(__file__).resolve().parent
    return {
        relpath: (directory / relpath).read_text(encoding="utf-8")
        for relpath in relpaths
    }


# ---------------------------------------------------------------------------
# Candidate mining
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One unselected QA triple mined from a source file."""

    family: str
    source_file: str  # relpath, e.g. "recurrent.py"
    module: str  # dotted, e.g. "humanbrain.recurrent"
    target: str  # qualname (functions) or bare name (containment)
    anchor: str  # globally unique reference
    context: tuple[str, ...]
    question: str
    answer: str

    @property
    def sort_key(self) -> bytes:
        return sha256(
            _ORDER_DOMAIN.encode("utf-8") + b"\x00" + self.anchor.encode("utf-8")
        ).digest()


def _display_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Named-parameter display string (leading ``self`` stripped)."""
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    while names and names[0] == "self":
        names.pop(0)
    parts = list(names)
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _doc_first_line(node) -> str | None:
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return None
    first = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    if not first or len(first) > MAX_SUMMARY_ANSWER_CHARS or first.endswith(":"):
        return None
    return first


def mine_candidates(sources: dict[str, str]) -> list[Candidate]:
    """Deterministically mine validated QA candidates from source texts."""
    # Pass 1: collect top-level public definitions per file for uniqueness.
    parsed: dict[str, ast.Module] = {}
    containment_index: dict[str, set[str]] = {}
    for relpath, text in sources.items():
        tree = ast.parse(text)
        parsed[relpath] = tree
        stem = Path(relpath).stem
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    containment_index.setdefault(node.name, set()).add(stem)

    candidates: list[Candidate] = []
    seen_anchors: set[str] = set()

    def emit(candidate: Candidate) -> None:
        if candidate.anchor in seen_anchors:
            raise ValueError(f"duplicate candidate anchor {candidate.anchor}")
        validate_candidate(candidate)
        seen_anchors.add(candidate.anchor)
        candidates.append(candidate)

    for relpath in sorted(sources):
        tree = parsed[relpath]
        module = f"humanbrain.{Path(relpath).stem}"
        module_line = f"# module {module} ({relpath})"

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") or _has_decorator(node, "overload"):
                    continue
                qualname = node.name
                kind = "function"
                signature = _display_signature(node)
                def_line = f"def {node.name}({signature}):"
                first_line = _doc_first_line(node)
                if first_line:
                    emit(
                        Candidate(
                            family="function_summary",
                            source_file=relpath,
                            module=module,
                            target=qualname,
                            anchor=f"{relpath}::{qualname}::function_summary",
                            context=(
                                module_line,
                                def_line,
                                f'"""{first_line}"""',
                            ),
                            question=(
                                f"In the module {module}, what does the {kind} "
                                f"{node.name} do?"
                            ),
                            answer=first_line,
                        )
                    )
                if signature:
                    emit(
                        Candidate(
                            family="signature_recall",
                            source_file=relpath,
                            module=module,
                            target=qualname,
                            anchor=f"{relpath}::{qualname}::signature_recall",
                            context=(module_line, def_line),
                            question=(
                                f"In the module {module}, what parameters does "
                                f"the {kind} {node.name} take?"
                            ),
                            answer=signature,
                        )
                    )
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                if not class_name.startswith("_"):
                    if len(containment_index[class_name]) == 1:
                        emit(
                            _containment_candidate(
                                relpath,
                                module,
                                module_line,
                                class_name,
                                kind="class",
                                def_line=f"class {class_name}:",
                            )
                        )
                class_def_line = f"class {class_name}:"
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.name.startswith("_") or _has_decorator(child, "overload"):
                            continue
                        qualname = f"{class_name}.{child.name}"
                        kind = "method"
                        signature = _display_signature(child)
                        method_line = f"def {child.name}({signature}):"
                        first_line = _doc_first_line(child)
                        if first_line:
                            emit(
                                Candidate(
                                    family="function_summary",
                                    source_file=relpath,
                                    module=module,
                                    target=qualname,
                                    anchor=f"{relpath}::{qualname}::function_summary",
                                    context=(
                                        module_line,
                                        class_def_line,
                                        method_line,
                                        f'"""{first_line}"""',
                                    ),
                                    question=(
                                        f"In the module {module}, what does the "
                                        f"{kind} {qualname} do?"
                                    ),
                                    answer=first_line,
                                )
                            )
                        if signature:
                            emit(
                                Candidate(
                                    family="signature_recall",
                                    source_file=relpath,
                                    module=module,
                                    target=qualname,
                                    anchor=f"{relpath}::{qualname}::signature_recall",
                                    context=(module_line, class_def_line, method_line),
                                    question=(
                                        f"In the module {module}, what parameters "
                                        f"does the {kind} {qualname} take?"
                                    ),
                                    answer=signature,
                                )
                            )
                # Containment for public METHODS is intentionally out of scope:
                # bare-name ambiguity makes "which module defines run" ill-posed.

            # Top-level containment for functions:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and len(containment_index[node.name]) == 1:
                    emit(
                        _containment_candidate(
                            relpath,
                            module,
                            module_line,
                            node.name,
                            kind="function",
                            def_line=f"def {node.name}():",
                        )
                    )

    return candidates


def _has_decorator(node, name_suffix: str) -> bool:
    for decorator in node.decorator_list:
        text = decorator.id if isinstance(decorator, ast.Name) else (
            decorator.attr if isinstance(decorator, ast.Attribute) else ""
        )
        if text == name_suffix or text.endswith(name_suffix):
            return True
    return False


def _containment_candidate(
    relpath: str,
    module: str,
    module_line: str,
    name: str,
    *,
    kind: str,
    def_line: str,
) -> Candidate:
    return Candidate(
        family="module_containment",
        source_file=relpath,
        module=module,
        target=name,
        anchor=f"{relpath}::{name}::module_containment",
        context=(module_line, def_line),
        question=f"Which humanbrain module defines the {kind} {name}?",
        answer=module,
    )


def validate_candidate(candidate: Candidate) -> None:
    """Mirrors qa_corpus._validate_entry: answers must be context-derivable."""
    if candidate.family not in FAMILIES:
        raise ValueError(f"unknown family {candidate.family!r}")
    if not candidate.question.strip():
        raise ValueError(f"{candidate.anchor}: empty question")
    if not candidate.answer.strip():
        raise ValueError(f"{candidate.anchor}: empty answer")
    if not candidate.context or not all(line.strip() for line in candidate.context):
        raise ValueError(f"{candidate.anchor}: context lines must be non-empty")
    if candidate.answer.casefold() not in " ".join(candidate.context).casefold():
        raise ValueError(
            f"{candidate.anchor}: answer {candidate.answer!r} is not derivable from context"
        )


# ---------------------------------------------------------------------------
# Budgeted selection and materialization
# ---------------------------------------------------------------------------


def select_entries(
    candidates: list[Candidate], roles: dict[str, str]
) -> dict[str, list[Candidate]]:
    """Per-cohort, per-family quota selection with deterministic backfill."""
    pooled: dict[str, dict[str, list[Candidate]]] = {"TRAIN": {}, "VAL": {}}
    for family in FAMILIES:
        for cohort in ("TRAIN", "VAL"):
            pooled[cohort][family] = []
    for candidate in sorted(candidates, key=lambda item: item.sort_key):
        cohort = roles[candidate.source_file]
        pooled[cohort][candidate.family].append(candidate)

    quotas: dict[str, dict[str, int]] = {
        "TRAIN": {family: TRAIN_FAMILY_QUOTA[family] for family in FAMILIES},
        "VAL": {family: VAL_FAMILY_QUOTA[family] for family in sorted(FAMILIES)},
    }
    targets = {"TRAIN": TRAIN_TARGET, "VAL": VAL_TARGET}

    selected: dict[str, list[Candidate]] = {}
    for cohort in ("TRAIN", "VAL"):
        picked: list[Candidate] = []
        used: set[str] = set()
        for family in quotas[cohort]:
            for candidate in pooled[cohort][family][: quotas[cohort][family]]:
                used.add(candidate.anchor)
                picked.append(candidate)
        # Deterministic round-robin backfill across families until target hit.
        family_cycle = sorted(FAMILIES)
        progress = True
        while len(picked) < targets[cohort] and progress:
            progress = False
            for family in family_cycle:
                if len(picked) >= targets[cohort]:
                    break
                for candidate in pooled[cohort][family]:
                    if candidate.anchor not in used:
                        used.add(candidate.anchor)
                        picked.append(candidate)
                        progress = True
                        break
        selected[cohort] = picked
    return selected


def _entry_record(cohort: str, ordinal: int, candidate: Candidate) -> dict[str, object]:
    identifier_domain = f"{SCHEMA_VERSION}/identifier/{cohort.lower()}"
    return {
        "qa_id": derive_opaque_identifier(identifier_domain, ordinal),
        "cohort": cohort,
        "ordinal": ordinal,
        "family": candidate.family,
        "module": candidate.module,
        "source_file": candidate.source_file,
        "target": candidate.target,
        "references": [candidate.anchor],
        "context": list(candidate.context),
        "question": candidate.question,
        "answer": candidate.answer,
    }


def build_corpus(
    sources: dict[str, str] | None = None,
) -> dict[str, object]:
    """Mine, split, select, and materialize the full corpus record."""
    relpaths = discover_source_files()
    if sources is None:
        sources = read_source_files(relpaths)
    missing = set(relpaths) - set(sources)
    if missing:
        raise ValueError(f"sources missing pinned files: {sorted(missing)}")
    extra = set(sources) - set(relpaths)
    if extra:
        raise ValueError(f"unexpected files passed as sources: {sorted(extra)}")

    candidates = mine_candidates(sources)
    roles = split_file_roles(relpaths)
    selected = select_entries(candidates, roles)

    entries: dict[str, list[dict[str, object]]] = {}
    offset = 0
    for cohort in ("TRAIN", "VAL"):
        records = [
            _entry_record(cohort, offset + index, candidate)
            for index, candidate in enumerate(selected[cohort])
        ]
        entries[cohort] = records
        offset += len(records)

    file_hashes = {
        relpath: sha256(sources[relpath].encode("utf-8")).hexdigest()
        for relpath in relpaths
    }
    counts = {
        cohort: {
            "total": len(entries[cohort]),
            **{
                family: sum(1 for e in entries[cohort] if e["family"] == family)
                for family in FAMILIES
            },
        }
        for cohort in ("TRAIN", "VAL")
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_source": CORPUS_SOURCE,
        "split_rule": (
            "BY FILE: sha256(file-split-domain, relpath)-ranked modules; lowest "
            f"{VAL_FILE_FRACTION:.0%}-ish share (floor {MIN_VAL_FILES}) are VAL; "
            "every triple's source_file lies wholly in one cohort"
        ),
        "files": [{"path": relpath, "sha256": file_hashes[relpath]} for relpath in relpaths],
        "roles": roles,
        "entries": entries,
        "counts": counts,
    }


def corpus_entries() -> dict[str, list[dict[str, object]]]:
    """Validated corpus entries (freshly built; cheap at this scale)."""
    corpus = build_corpus()
    for cohort in ("TRAIN", "VAL"):
        for record in corpus["entries"][cohort]:  # type: ignore[index]
            _validate_entry_record(record)  # type: ignore[arg-type]
    return corpus["entries"]  # type: ignore[return-value]


def _validate_entry_record(record: dict[str, object]) -> None:
    label = str(record.get("references", ["<no-anchor>"])[0])
    for field in ("question", "answer"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}: empty {field}")
    context = record.get("context")
    if (
        not isinstance(context, list)
        or not context
        or not all(isinstance(line, str) and line for line in context)
    ):
        raise ValueError(f"{label}: malformed context")
    if str(record["answer"]).casefold() not in " ".join(context).casefold():  # type: ignore[attr-defined]
        raise ValueError(
            f"{label}: answer not derivable from its context lines"
        )
    if record.get("family") not in FAMILIES:
        raise ValueError(f"{label}: unknown family {record.get('family')!r}")


def canonical_json(value: object) -> bytes:
    """Canonical JSON encoding (sort_keys, tight separators, UTF-8)."""
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def corpus_record() -> dict[str, object]:
    """Canonical JSON-ready record covering BOTH cohorts' entries."""
    return {"entries": corpus_entries()}


def corpus_digest() -> str:
    """SHA-256 over the canonical JSON of the validated two-cohort corpus."""
    return sha256(canonical_json(corpus_record())).hexdigest()


def prompt_for_entry(record: dict[str, object]) -> str:
    """Same prompt-last framing as the Bible corpus (ends in ``answer:``)."""
    return qa_prompt(tuple(str(line) for line in record["context"]), str(record["question"]))  # type: ignore[arg-type]


def growth_text_for_entry(record: dict[str, object]) -> str:
    """Growth text pairs the prompt with its answer: ``<prompt> <answer>``."""
    return prompt_for_entry(record) + " " + str(record["answer"])


# ---------------------------------------------------------------------------
# Seed-keyed curriculum over the TRAIN cohort (mirrors qa_corpus.build_curriculum)
# ---------------------------------------------------------------------------


def _task_sort_key(domain: str, task_id: str) -> bytes:
    return sha256(domain.encode("utf-8") + b"\x00" + task_id.encode("utf-8")).digest()


def build_curriculum(
    seed: int,
    *,
    calibration_task_count: int = 6,
) -> dict[str, object]:
    """Seed-keyed TRAIN/CALIBRATION cohort pair drawn from the TRAIN file-split.

    Mirrors ``qa_corpus.build_curriculum``: one global sha256 order picks the
    calibration subset, per-cohort order domains, dense ordinals (calibration
    offset past train), opaque identifiers per ordinal, and one canonical-JSON
    curriculum SHA-256. VAL-file entries NEVER enter either cohort.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    train_entries = corpus_entries()["TRAIN"]
    if not 1 <= calibration_task_count < len(train_entries):
        raise ValueError("calibration_task_count must leave both cohorts non-empty")

    specs = [
        (str(entry["references"][0]), dict(entry)) for entry in train_entries
    ]
    identifier_domain = f"{SCHEMA_VERSION}/curriculum-identifier/seed/{seed}"
    order_domain = f"{SCHEMA_VERSION}/curriculum-order/seed/{seed}"
    global_order = sorted(
        specs, key=lambda item: _task_sort_key(f"{order_domain}/global", item[0])
    )
    calibration_ids = {task_id for task_id, _ in global_order[:calibration_task_count]}
    calibration_specs = [item for item in specs if item[0] in calibration_ids]
    train_specs = [item for item in specs if item[0] not in calibration_ids]

    def _materialize(
        items: list[tuple[str, dict]], *, cohort: str, domain: str, ordinal_offset: int
    ) -> list[QATask]:
        ordered = sorted(items, key=lambda item: _task_sort_key(domain, item[0]))
        tasks: list[QATask] = []
        for index, (task_id, spec) in enumerate(ordered):
            tasks.append(
                QATask(
                    cohort=cohort,
                    qa_id=derive_opaque_identifier(domain + "/qa-id", ordinal_offset + index),
                    ordinal=ordinal_offset + index,
                    references=(task_id,),
                    context=tuple(str(item) for item in spec["context"]),
                    question=str(spec["question"]),
                    answer=str(spec["answer"]),
                )
            )
        return tasks

    train_tasks = _materialize(
        train_specs, cohort="TRAIN", domain=f"{order_domain}/train", ordinal_offset=0
    )
    calibration_tasks = _materialize(
        calibration_specs,
        cohort="CALIBRATION",
        domain=f"{order_domain}/calibration",
        ordinal_offset=len(train_tasks),
    )

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
                "calibration": [_record(task) for task in calibration_tasks],
            }
        )
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_source": CORPUS_SOURCE,
        "corpus_size": len(train_entries),
        "corpus_sha256": corpus_digest(),
        "seed": seed,
        "identifier_domain": identifier_domain,
        "order_domain": order_domain,
        "train_tasks": tuple(train_tasks),
        "calibration_tasks": tuple(calibration_tasks),
        "sha256": curriculum_digest,
    }
