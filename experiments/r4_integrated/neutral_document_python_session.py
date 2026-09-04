"""Repeatable public development: neutral reading, delayed memory, executed edits.

See the adjacent .md for the declared workload, controls, and budget. Expected
answers and teacher repairs stay in this evaluator; inference receives only raw
requests and workspace bytes. This runner never accesses protected evaluations.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from humanbrain.knowledge_conditioned_occurrence_substitution import (
    KnowledgeSubstitutionExecutionAuthority,
    KnowledgeSubstitutionSuccess,
)
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
from humanbrain.neutral_document_action import DocumentActionAblation, start
from humanbrain.opaque_document_event_tape import OpaqueDocument
from humanbrain.unified_opaque_observation_tape import UnifiedOpaqueObservationTape


@dataclass(frozen=True)
class Case:
    name: str
    key: str
    value: str
    module: str
    stale: str

    @property
    def request(self) -> bytes:
        return f"serves {self.key}".encode()

    def files(self, *, corrected: bool = False, annotated: bool = False):
        value = self.value if corrected else self.stale
        annotation = ": str" if annotated else ""
        return tuple(
            sorted(
                (
                    OpaqueWorkspaceFile(
                        f"{self.module}.py", f"{self.key}{annotation} = '{value}'\n".encode()
                    ),
                    OpaqueWorkspaceFile("decoy.py", b"foreign = 'spare'\n"),
                    OpaqueWorkspaceFile(
                        "main.py",
                        (
                            f"import {self.module}\n"
                            f"print(next(v for k, v in vars({self.module}).items() "
                            "if not k.startswith('_')))\n"
                        ).encode(),
                    ),
                ),
                key=lambda item: item.path,
            )
        )


TRAIN = (
    Case("train-a", "kelo", "vesa", "a_cfg", "dull"),
    Case("train-b", "rilo", "luma", "b_settings", "pale"),
)
TRANSFER = (
    Case("transfer-a", "moro", "zani", "fresh_options", "gray"),
    Case("transfer-b", "qelo", "raku", "renamed_values", "white"),
)


def documents(label: str, *texts: str) -> tuple[OpaqueDocument, ...]:
    return tuple(
        OpaqueDocument(f"{label}/{i}".encode(), f"provenance/{label}/{i}".encode(), text.encode())
        for i, text in enumerate(texts)
    )


def execute(case: Case, files, before, root: Path) -> dict:
    """Materialize and run these trusted, generated public fixtures only."""
    began = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    for item in files:
        path = (root / item.path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("workspace path escapes execution directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(item.content)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import sys, runpy; sys.path.insert(0, '.'); "
                "runpy.run_path('main.py', run_name='__main__')",
            ],
            cwd=root,
            capture_output=True,
            timeout=10,
            check=False,
        )
        returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        returncode, stdout, stderr = None, error.stdout or b"", error.stderr or b""
    same_paths = tuple(item.path for item in files) == tuple(item.path for item in before)
    untouched = same_paths and all(
        item.content == original.content
        for item, original in zip(files, before, strict=True)
        if original.path != f"{case.module}.py"
    )
    materialized = all((root / item.path).read_bytes() == item.content for item in files)
    return {
        "returncode": returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "decoys_unchanged": untouched,
        "materialized_matches_action": materialized,
        "passed": returncode == 0
        and stdout.strip() == case.value.encode()
        and untouched
        and materialized,
        "workspace": str(root),
        "elapsed_seconds": time.perf_counter() - began,
    }


def snapshot(learner, event: str) -> dict:
    summary = asdict(learner.document_learner.training_summary)
    return {
        "event": event,
        "checkpoint": learner.checkpoint_sha256,
        "parent": learner.parent_checkpoint_sha256,
        "document_summary": summary,
        "demonstrations": learner.demonstration_count,
        "pending_demonstration_event_bytes": learner.retained_demonstration_event_bytes,
        "serialized_checkpoint_bytes": len(pickle.dumps(learner, protocol=5)),
        "binder_summary": None
        if learner.binder is None
        else asdict(learner.binder.training_summary),
    }


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tracemalloc.start()
    seed = start()
    arms = {"UPDATE": seed, "NO_UPDATE": seed}
    history = {name: [snapshot(brain, "seed")] for name, brain in arms.items()}
    attempts, language, teachers, controls = [], [], [], []
    reference_facts: dict[str, set[str]] = {}

    def acquire(label: str, *texts: str):
        batch = documents(label, *texts)
        began = time.perf_counter()
        before = arms["UPDATE"]
        arms["UPDATE"] = before.consolidate(batch)
        elapsed = time.perf_counter() - began
        assert arms["UPDATE"].parent_checkpoint_sha256 == before.checkpoint_sha256
        for name, brain in arms.items():
            record = snapshot(brain, label)
            record["offered_documents"] = [
                {"id": doc.document_id.decode(), "sha256": sha256(doc.content).hexdigest()}
                for doc in batch
            ]
            record["update_applied"] = name == "UPDATE"
            record["acquisition_seconds"] = elapsed if name == "UPDATE" else 0.0
            history[name].append(record)
        for text in texts:
            words = text.split()
            if len(words) == 3 and words[1] == "uses":
                reference_facts.setdefault(words[0], set()).add(words[2])

    def recall(stage: str):
        for name, brain in arms.items():
            for case in TRAIN:
                for request, expected in (
                    (case.request, case.value),
                    (f"{case.value} serves".encode(), case.key),
                ):
                    began = time.perf_counter()
                    decision = brain.respond(request)
                    language.append(
                        {
                            "arm": name,
                            "stage": stage,
                            "request": request.decode(),
                            "status": decision.status,
                            "answer": None
                            if decision.emitted is None
                            else decision.emitted.decode(),
                            "passed": decision.status == "EMIT"
                            and decision.emitted == expected.encode(),
                            "checkpoint": decision.checkpoint_sha256,
                            "physical_cell_steps": decision.physical_cell_steps,
                            "elapsed_seconds": time.perf_counter() - began,
                        }
                    )

    def attempt(
        name,
        brain,
        case,
        stage,
        *,
        ablation=DocumentActionAblation.NONE,
        files=None,
        no_action=False,
    ):
        files = case.files() if files is None else files
        baseline = execute(case, files, files, output / f"run-{len(attempts):03d}-before")
        began = time.perf_counter()
        checkpoint = brain.checkpoint_sha256
        decision = brain.respond(case.request, files=files, ablation=ablation)
        if brain.checkpoint_sha256 != checkpoint:
            raise AssertionError("inference changed checkpoint")
        row = {
            "arm": name,
            "stage": stage,
            "case": case.name,
            "ablation": ablation.value,
            "status": decision.status,
            "stop_reason": decision.stop_reason,
            "checkpoint": checkpoint,
            "decision_sha256": decision.sha256,
            "physical_cell_steps": decision.physical_cell_steps,
            "source_documents_present": False,
            "hidden_retries": 0,
            "inference_seconds": time.perf_counter() - began,
            "before_execution": baseline,
        }
        after = decision.after_workspace or files
        row["execution"] = (
            execute(case, after, files, output / f"run-{len(attempts):03d}")
            if after != files
            else baseline
        )
        row["passed"] = (
            decision.status == "STOP" and after == files
            if no_action
            else decision.status == "EDIT" and row["execution"]["passed"]
        )
        attempts.append(row)
        return row

    acquire("structure-a", "moru uses orli", "orli serves moru")
    acquire("structure-b", "sena uses ulon", "ulon serves sena")
    acquire("training-facts", *(f"{case.key} uses {case.value}" for case in TRAIN))
    acquire("interference", "unrelated birds migrate")
    recall("delayed")

    for case in TRAIN:
        before_files, after_files = case.files(), case.files(corrected=True)
        for name, brain in tuple(arms.items()):
            attempt(name, brain, case, "first-training-attempt")
            observation = UnifiedOpaqueObservationTape.observe(case.request, (), before_files)

            def evaluate(*, observation, after_workspace):
                execution = execute(
                    case,
                    after_workspace,
                    observation.reconstruct_workspace(),
                    output / f"teacher-{len(teachers):03d}",
                )
                teachers.append({"arm": name, "case": case.name, "execution": execution})
                return execution["passed"]

            authority = KnowledgeSubstitutionExecutionAuthority(
                evaluate, sha256(Path(__file__).read_bytes()).hexdigest()
            )
            receipt = authority.execute_and_issue(
                producer_checkpoint_sha256=brain.checkpoint_sha256,
                observation=observation,
                after_workspace=after_files,
            )
            success = KnowledgeSubstitutionSuccess.seal(
                observation=observation,
                after_workspace=after_files,
                execution_receipt=receipt,
                provenance_id=f"teacher/{case.name}".encode(),
            )
            if name == "UPDATE":
                arms[name] = brain.learn_action(success)
            teachers[-1]["receipt_sha256"] = receipt.sha256
            teachers[-1]["applied"] = name == "UPDATE"
            history[name].append(snapshot(arms[name], case.name))

    acquire("fresh-facts", *(f"{case.key} uses {case.value}" for case in TRANSFER))
    for case in TRANSFER:
        for name, brain in arms.items():
            attempt(name, brain, case, "transfer")
    recall("retention")
    for case in TRAIN:
        for name, brain in arms.items():
            attempt(name, brain, case, "action-retention")

    trained = arms["UPDATE"]
    for ablation in DocumentActionAblation:
        if ablation is not DocumentActionAblation.NONE:
            controls.append(attempt("UPDATE", trained, TRANSFER[0], "causal", ablation=ablation))
    missing = Case("unsupported", "absent", "unavailable", "unknown", "old")
    attempt("UPDATE", trained, missing, "no-action", no_action=True)
    duplicate = tuple(
        sorted(
            (*TRANSFER[0].files(), OpaqueWorkspaceFile("duplicate.py", b"moro = 'other'\n")),
            key=lambda item: item.path,
        )
    )
    attempt("UPDATE", trained, TRANSFER[0], "duplicate-anchor", files=duplicate, no_action=True)
    conflicted = trained.consolidate(documents("conflict", "moro uses different"))
    history["CONFLICT_FORK"] = [snapshot(conflicted, "conflict")]
    attempt("UPDATE", conflicted, TRANSFER[0], "conflict", no_action=True)
    attempt(
        "UPDATE", trained, TRANSFER[0], "layout-variation", files=TRANSFER[0].files(annotated=True)
    )

    references = []
    for case in TRANSFER:
        before = case.files()
        values = reference_facts.get(case.key, set())
        after = list(before)
        matches = []
        if len(values) == 1:
            pattern = re.compile(rb"(?m)^" + re.escape(case.key.encode()) + rb" = '([^']*)'")
            matches = [
                (i, match)
                for i, item in enumerate(before)
                for match in pattern.finditer(item.content)
            ]
            if len(matches) == 1:
                i, match = matches[0]
                item = before[i]
                after[i] = OpaqueWorkspaceFile(
                    item.path,
                    item.content[: match.start(1)]
                    + next(iter(values)).encode()
                    + item.content[match.end(1) :],
                )
        references.append(
            {
                "case": case.name,
                "execution": execute(case, tuple(after), before, output / f"reference-{case.name}"),
            }
        )

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    required_stages = {"transfer", "action-retention", "no-action", "duplicate-anchor", "conflict"}
    required = [
        row for row in attempts if row["arm"] == "UPDATE" and row["stage"] in required_stages
    ]
    treatment_language = [row for row in language if row["arm"] == "UPDATE"]
    frozen_language = [row for row in language if row["arm"] == "NO_UPDATE"]
    frozen_actions = [row for row in attempts if row["arm"] == "NO_UPDATE"]
    passed = (
        len(required) == 7
        and all(row["passed"] for row in required)
        and len(treatment_language) == 8
        and all(row["passed"] for row in treatment_language)
        and len(frozen_language) == 8
        and all(not row["passed"] for row in frozen_language)
        and len(controls) == 4
        and all(row["status"] == "STOP" for row in controls)
        and len(frozen_actions) == 6
        and all(not row["passed"] for row in frozen_actions)
        and len(teachers) == 4
        and sum(row["applied"] for row in teachers) == 2
        and all(row["checkpoint"] == seed.checkpoint_sha256 for row in history["NO_UPDATE"])
        and all(not row["before_execution"]["passed"] for row in attempts)
    )
    source_root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve()] + [
        source_root / "src/humanbrain" / name
        for name in (
            "neutral_document_action.py",
            "neutral_neuronal_document_learner.py",
            "knowledge_conditioned_occurrence_substitution.py",
            "variability_aware_segment_binder.py",
            "opaque_document_event_tape.py",
            "unified_opaque_observation_tape.py",
            "sparse_neuronal_role_filler.py",
            "resting_glif_race.py",
            "cells.py",
        )
    ]
    result = {
        "kind": "repeatable public development; narrow literal-edit integration",
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(source_root)).replace("\\", "/"): sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_paths
        },
        "narrow_integration_passed": passed,
        "history": history,
        "language": language,
        "attempts": attempts,
        "teachers": teachers,
        "reference": references,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_traced_python_bytes": peak,
        "memory_limit": "tracemalloc excludes native allocations; total process peak not measured",
        "limitations": [
            "three-token positional reading",
            "two-demonstration lexical-offset motor",
            "host literal substitution and execution",
            "no autonomous recovery",
            "public development reuse; no protected or external-model comparison",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(
        json.dumps({key: report[key] for key in ("narrow_integration_passed", "elapsed_seconds")})
    )
