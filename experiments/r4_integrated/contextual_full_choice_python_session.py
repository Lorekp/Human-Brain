"""Supervised full-choice acquisition in the persistent executed Python loop.

Run only after the adjacent development protocol, source and input package are
fixed. No protected or fresh adjudication material is accessed by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
import tracemalloc
from collections import Counter
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = HERE / "contextual_full_choice_train.json"
PROTOCOL = HERE / "contextual_full_choice_protocol.md"
MAX_SECONDS = 7200


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_fixture(path: Path = FIXTURE) -> dict:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    tasks, documents = fixture["tasks"], fixture["documents"]
    names = [task["id"] for task in tasks]
    if len(tasks) != 64 or len(set(names)) != 64:
        raise ValueError("requires 64 unique development tasks")
    if sum(task["answer"] is not None for task in tasks) != 48:
        raise ValueError("requires 48 supported and 16 explicitly unsupported tasks")
    if Counter(len(task["answer"].split()) for task in tasks if task["answer"]) != {
        1: 16,
        2: 16,
        3: 16,
    }:
        raise ValueError("supported answer lengths must be balanced")
    doc_ids = [doc["id"] for doc in documents]
    if len(doc_ids) < 16 or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("requires at least sixteen unique TRAIN documents")
    if any(task["expected_source_id"] not in doc_ids for task in tasks):
        raise ValueError("task refers to a missing source")
    if set(fixture["alternate_queries"]) != set(names):
        raise ValueError("alternate queries must cover exactly the same tasks")
    if any(fixture["alternate_queries"][task["id"]] == task["query"] for task in tasks):
        raise ValueError("every alternate form must differ from its actual teaching query")
    schedule = fixture["schedule"]
    if len(schedule) != 256 or any(
        Counter(schedule[start : start + 64]) != Counter(names) for start in range(0, 256, 64)
    ):
        raise ValueError("requires four complete schedule passes")
    if any(left == right for left, right in pairwise(schedule)):
        raise ValueError("preceding-encounter replay must name a distinct task")
    growth = fixture["growth_documents"]
    growth_ids = [doc["id"] for doc in growth]
    if len(growth) != 4 or len(set(growth_ids)) != 4 or set(growth_ids) & set(doc_ids):
        raise ValueError("requires four distinct growth-only documents")
    if any(task["answer"] == task["workspace"]["stale"] for task in tasks):
        raise ValueError("supported teacher answer already equals the initial workspace")
    return fixture


def _item(task: dict, query: str) -> dict:
    return {
        "name": task["id"],
        "key": task["workspace"]["key"],
        "module": task["workspace"]["module"],
        "stale": task["workspace"]["stale"],
        "value": task["answer"],
        "request": query,
    }


def _summary(rows: list[dict], tasks: list[dict]) -> dict:
    counts = {"supported": 0, "executed_success": 0, "unsupported": 0, "correct_stop": 0}
    groups: dict[str, dict[str, int]] = {}
    for row, task in zip(rows, tasks, strict=True):
        supported = task["answer"] is not None
        good = bool(row["execution"]["passed"]) if supported else row["correct_no_action"]
        counts["supported" if supported else "unsupported"] += 1
        counts["executed_success" if supported else "correct_stop"] += int(good)
        labels = [f"role:{task['property']}", f"form:{task['query_form_group']}"]
        if supported:
            labels.append(f"length:{len(task['answer'].split())}")
        else:
            labels.append("answerability:unsupported")
        for label in labels:
            group = groups.setdefault(label, {"count": 0, "correct": 0})
            group["count"] += 1
            group["correct"] += int(good)
    return {
        **counts,
        "groups": groups,
        "correct_total": counts["executed_success"] + counts["correct_stop"],
    }


def acquisition_gate(phases: dict) -> dict:
    before = phases["actual_before_growth"]["update"]["summary"]
    after = phases["actual_after_growth"]["update"]["summary"]
    thresholds = {
        phase: summary["executed_success"] >= 44 and summary["correct_stop"] >= 15
        for phase, summary in (("before_growth", before), ("after_growth", after))
    }
    groups = {
        phase: all(
            5 * g["correct"] >= 4 * g["count"]
            for g in result["update"]["summary"]["groups"].values()
        )
        for phase, result in phases.items()
    }
    # All phases contain the same 64 tasks. Three losses are below five points;
    # four losses are above it. Preserve signed gain rather than clipping it.
    growth_nonregression = 100 * (before["correct_total"] - after["correct_total"]) <= 5 * 64
    return {
        "passed": all(thresholds.values()) and all(groups.values()) and growth_nonregression,
        "actual_thresholds": thresholds,
        "all_declared_groups_at_least_80_percent": groups,
        "growth_loss_at_most_five_percentage_points": growth_nonregression,
        "scope": "public development acquisition only; no fresh or integrated gate claim",
    }


def run(output: Path, model_dir: Path, fixture_path: Path = FIXTURE) -> dict:
    output = output.resolve()
    if any(
        p.exists()
        for p in (output, output.with_suffix(".npz"), output.with_suffix(".failure.json"))
    ):
        raise FileExistsError("choose a new output path; prior evidence is immutable")
    if (
        os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
        or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve()
        != (ROOT / ".native-cache").resolve()
    ):
        raise RuntimeError("set CPU and checkout native-cache before starting Python")
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".runtime").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextual-full-choice-", dir=ROOT / ".runtime"))
    started = time.perf_counter()
    tracemalloc.start()
    audit_basis = update = frozen = execution_error = None
    report = {
        "terminal": "running",
        "workspace_root": str(work),
        "training": [],
        "phases": {},
        "events": [],
    }

    def record(event: str, **fields):
        row = {"event": event, **fields}
        report["events"].append(row)
        with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(row) + "\n")
        print(_json(row), flush=True)

    def check_budget():
        if time.perf_counter() - started > MAX_SECONDS:
            raise TimeoutError("declared two-hour acquisition execution budget exhausted")

    try:
        import numpy as np
        from experiments.r4_integrated import contextual_evidence_python_session as shared
        from experiments.r4_integrated import contextual_pairwise_python_session as prior
        from humanbrain.frozen_context_encoder import FrozenContextEncoder
        from humanbrain.full_choice_evidence_memory import FullChoiceEvidenceMemory
        from humanbrain.native import loaded_build_info
        from humanbrain.opaque_document_event_tape import OpaqueDocument

        def save_checkpoint(arrays):
            checkpoint = output.with_suffix(".npz")
            with checkpoint.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            report["checkpoint"] = {
                "path": str(checkpoint),
                "sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "array_bytes": sum(array.nbytes for array in arrays.values()),
                "compressed_bytes": checkpoint.stat().st_size,
            }

        def diagnose(memory, query, task, manifest):
            parent = memory.checkpoint_sha256
            inventory = memory.score_inventory(query.encode())
            logits = np.asarray(inventory.pop("logits"), dtype=np.float64)
            stop = inventory["stop_index"]
            gold = (
                [stop]
                if task["answer"] is None
                else [
                    i
                    for i, candidate in enumerate(manifest)
                    if candidate["emitted_hex"] == task["answer"].encode().hex()
                ]
            )
            if not gold:
                raise RuntimeError("whole teacher span is missing from the diagnostic inventory")
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            wrong = np.ones(len(logits), dtype=bool)
            wrong[gold] = False
            wrong_indices = np.flatnonzero(wrong)
            worst = int(wrong_indices[np.argmax(logits[wrong])])
            leader = int(np.argmax(logits))
            if parent != memory.checkpoint_sha256:
                raise RuntimeError("score diagnostic changed the learner")
            return {
                **inventory,
                "scope": "evaluator-only; never schedules learning",
                "gold_indices": gold,
                "gold_probability": float(probabilities[gold].sum()),
                "best_gold_logit": float(logits[gold].max()),
                "best_gold_rank": int(1 + np.count_nonzero(logits > logits[gold].max())),
                "leader_index": leader,
                "leader_is_unique": int(np.count_nonzero(logits == logits[leader])) == 1,
                "worst_wrong_index": worst,
                "worst_wrong_logit": float(logits[worst]),
                "worst_wrong_candidate": None if worst == stop else manifest[worst],
                "gold_to_worst_wrong_gap": float(logits[gold].max() - logits[worst]),
                "stop_probability": float(probabilities[stop]),
            }

        paths = tuple(
            dict.fromkeys(
                (
                    *shared.SOURCE_PATHS,
                    "experiments/r4_integrated/contextual_pairwise_python_session.py",
                    "src/humanbrain/full_choice_evidence_memory.py",
                    "experiments/r4_integrated/contextual_full_choice_python_session.py",
                    PROTOCOL.relative_to(ROOT).as_posix(),
                    fixture_path.resolve().relative_to(ROOT).as_posix(),
                )
            )
        )
        report["source_sha256"] = {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
        report["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        fixture = load_fixture(fixture_path)
        tasks = fixture["tasks"]
        by_id = {task["id"]: task for task in tasks}
        report["kind"] = (
            "public development full-choice supervised acquisition; no fresh adjudication"
        )
        report["source_policy"] = (
            "raw sources and frozen candidate vectors retained internally; "
            "live action input contains no sources"
        )
        report["control_scope"] = (
            "evidence learning disabled or external selector; all share previously learned "
            "neuronal literal-edit motor"
        )
        report["external_policy"] = (
            "unique maximum cosine of normalized query and span vectors, regardless of sign; "
            "STOP only for empty inventory or exact tied maxima; ignores learned weights and STOP"
        )
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to((ROOT / ".native-cache").resolve()):
            raise RuntimeError("native library is outside the declared cache")
        report["native"] = {
            "source_digest": native.source_digest,
            "library": str(native.library_path),
            "binary_sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        began = time.perf_counter()
        motor = prior._motor(work, report)
        report["motor_setup_seconds"] = time.perf_counter() - began
        began = time.perf_counter()
        encoder = FrozenContextEncoder(model_dir)
        report["encoder"], report["encoder_load_seconds"] = (
            encoder.metadata,
            time.perf_counter() - began,
        )
        update = FullChoiceEvidenceMemory(encoder)
        frozen = FullChoiceEvidenceMemory(encoder, learning_enabled=False)
        if update.checkpoint_sha256 != frozen.checkpoint_sha256:
            raise RuntimeError("initial paired checkpoints differ")
        report["parameters"], report["seed"] = update.parameters, update.snapshot()
        began = time.perf_counter()

        def ingest_document(document, provenance):
            source = OpaqueDocument(
                document["id"].encode(), provenance.encode(), document["text"].encode()
            )
            update.ingest((source,))
            frozen.ingest((source,))
            return {
                "document_id": document["id"],
                "provenance_id": provenance,
                "raw_sha256": sha256(document["text"].encode()).hexdigest(),
                "update": update.snapshot(),
                "frozen": frozen.snapshot(),
            }

        report["ingestion"] = [
            ingest_document(document, "full-choice-public-train-v1")
            for document in fixture["documents"]
        ]
        report["training_pool"] = update.seal_training_pool()
        if frozen.seal_training_pool() != report["training_pool"]:
            raise RuntimeError("paired TRAIN source pools differ")
        report["ingestion_seconds"] = time.perf_counter() - began
        report["after_ingestion"] = update.snapshot()
        began = time.perf_counter()
        audit_basis = update.export_audit_basis(
            tuple(
                query.encode()
                for task in tasks
                for query in (task["query"], fixture["alternate_queries"][task["id"]])
            )
        )
        report["audit_basis"] = audit_basis["metadata"]
        report["audit_basis_seconds"] = time.perf_counter() - began
        report["group_scoring"] = (
            "roles pool supported and STOP decisions; length groups use supported targets; "
            "all eight form groups apply separately to actual/alternate phases"
        )

        def attempt(memory, method, task, query, folder):
            check_budget()
            row, decision = shared.attempt(memory, method, motor, _item(task, query), folder)
            if method == "respond_affine_interpolated":
                parent = memory.checkpoint_sha256
                raw = memory.raw_score_reference(query.encode())
                row["raw_score_reference"] = shared.decision_record(raw)
                row["raw_affine_agree"] = (
                    raw.candidate_index == decision.candidate_index
                    and raw.emitted == decision.emitted
                )
                if memory.checkpoint_sha256 != parent:
                    raise RuntimeError("raw score reference changed the learner checkpoint")
            return row, decision

        began = time.perf_counter()
        previous = None
        for step, name in enumerate(fixture["schedule"], 1):
            task = by_id[name]
            folder = work / "teaching" / f"{step:03d}-{name}"
            row = {"step": step, "pass": (step - 1) // 64 + 1, "task": name}
            report["training"].append(row)
            row["first"], original = attempt(
                update, "respond_affine_interpolated", task, task["query"], folder / "update"
            )
            row["frozen_first"], _ = attempt(
                frozen, "respond_affine_interpolated", task, task["query"], folder / "frozen"
            )
            row["external_first"], _ = attempt(
                update, "external_only", task, task["query"], folder / "external"
            )
            case = shared.Case(
                name=task["id"],
                key=task["workspace"]["key"],
                module=task["workspace"]["module"],
                stale=task["workspace"]["stale"],
                value=task["answer"] or task["workspace"]["stale"],
            )
            before = case.files()
            after = case.files(corrected=True) if task["answer"] is not None else before
            teacher = shared.execute(case, after, before, folder / "teacher")
            teacher.update(
                teacher_kind="authorized_exact_answer"
                if task["answer"] is not None
                else "authorized_absent_evidence_stop",
                answer_hex=None if task["answer"] is None else task["answer"].encode().hex(),
                workspace_unchanged=after == before,
            )
            row["teacher"] = teacher
            if not teacher["passed"]:
                raise RuntimeError("declared teacher action/no-action check failed")
            lesson = update.teacher_lesson(
                original,
                None if task["answer"] is None else task["answer"].encode(),
                teacher_id=f"{step:03d}:{name}",
                attempt_receipt={**row["first"]["execution"], "action": row["first"]},
                teacher_receipt=teacher,
            )
            row["lesson"] = lesson.record()
            row["new"] = {
                "update": update.observe_teacher(lesson),
                "frozen": frozen.observe_teacher(lesson),
            }
            if previous is not None:
                row["replay"] = {
                    "update": update.observe_teacher(previous),
                    "frozen": frozen.observe_teacher(previous),
                }
            previous = lesson
            record(
                "teaching",
                step=step,
                task=name,
                first_pass=row["first"]["execution"]["passed"],
                first_correct_stop=row["first"]["correct_no_action"],
                offers=2 * step - 1,
            )
        report["teaching_seconds"] = time.perf_counter() - began
        report["pass_summaries"] = []
        for number in range(1, 5):
            selected = [row for row in report["training"] if row["pass"] == number]
            corresponding_tasks = [by_id[row["task"]] for row in selected]
            report["pass_summaries"].append(
                {
                    "pass": number,
                    **{
                        policy: _summary([row[key] for row in selected], corresponding_tasks)
                        for policy, key in (
                            ("update", "first"),
                            ("frozen", "frozen_first"),
                            ("external", "external_first"),
                        )
                    },
                }
            )
        if update.feedback_count != 511 or frozen.feedback_count != 511 or frozen.update_count:
            raise RuntimeError("paired supervised offers or frozen mutations differ from protocol")

        report["trained"], report["frozen_trained"] = update.snapshot(), frozen.snapshot()
        arrays = {}
        for prefix, memory in (("update", update), ("frozen", frozen)):
            arrays.update(
                {f"{prefix}_{key}": value for key, value in memory.checkpoint_arrays().items()}
            )
        learned = {key: value.copy() for key, value in arrays.items()}
        arrays.update(audit_basis["arrays"])

        for phase in (
            "actual_before_growth",
            "alternate_before_growth",
            "actual_after_growth",
            "alternate_after_growth",
        ):
            check_budget()
            if phase == "actual_after_growth":
                report["growth_ingestion"] = [
                    ingest_document(document, "full-choice-growth-only-v1")
                    for document in fixture["growth_documents"]
                ]
                report["after_growth"] = update.snapshot()
            began = time.perf_counter()
            report["phases"][phase] = {}
            manifest = update.candidate_manifest()
            for policy, memory, method in (
                ("update", update, "respond_affine_interpolated"),
                ("frozen", frozen, "respond_affine_interpolated"),
                ("external", update, "external_only"),
            ):
                rows = []
                report["phases"][phase][policy] = {"rows": rows}
                for task in tasks:
                    query = (
                        fixture["alternate_queries"][task["id"]]
                        if phase.startswith("alternate")
                        else task["query"]
                    )
                    row, _ = attempt(
                        memory, method, task, query, work / phase / policy / task["id"]
                    )
                    rows.append(row)
                    if policy == "update":
                        row["score_diagnostics"] = diagnose(update, query, task, manifest)
                summary = _summary(rows, tasks)
                report["phases"][phase][policy]["summary"] = summary
                record("evaluation", phase=phase, policy=policy, summary=summary)
            report["phases"][phase]["seconds"] = time.perf_counter() - began
            for prefix, memory in (("update", update), ("frozen", frozen)):
                for key, value in memory.checkpoint_arrays().items():
                    if not np.array_equal(value, learned[f"{prefix}_{key}"]):
                        raise RuntimeError("evaluation/source growth changed learned state")
        report["acquisition_gate"] = acquisition_gate(report["phases"])
        report["end"], report["frozen_end"] = update.snapshot(), frozen.snapshot()
        save_checkpoint(arrays)
        if {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in paths} != report[
            "source_sha256"
        ]:
            raise RuntimeError("source changed during acquisition")
        report["terminal"] = "completed"
        report["process_memory"] = shared._process_memory()
    except Exception as error:
        execution_error = error
        report["terminal"], report["error"] = "error", traceback.format_exc()
        if audit_basis is not None and not output.with_suffix(".npz").exists():
            try:
                partial_arrays = dict(audit_basis["arrays"])
                for prefix, memory in (("update", update), ("frozen", frozen)):
                    partial_arrays.update(
                        {
                            f"{prefix}_{key}": value
                            for key, value in memory.checkpoint_arrays().items()
                        }
                    )
                save_checkpoint(partial_arrays)
                report["partial_state"] = {"update": update.snapshot(), "frozen": frozen.snapshot()}
            except Exception as recording_error:
                error.add_note(f"Could not preserve partial checkpoint: {recording_error}")
                report["partial_checkpoint_error"] = str(recording_error)
        try:
            with output.with_suffix(".failure.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(
                    _json(
                        {"terminal": "error", "workspace_root": str(work), "error": report["error"]}
                    )
                    + "\n"
                )
        except OSError as recording_error:
            error.add_note(f"Could not preserve failure sidecar: {recording_error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["python_traced_peak_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        report["resource_scope"] = (
            "parent memory includes evaluator/ledger and excludes children; "
            "wall time includes child waits"
        )
        try:
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(_json(report) + "\n")
        except Exception as recording_error:
            if execution_error is None:
                raise
            execution_error.add_note(f"Could not preserve partial JSON report: {recording_error}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/contextual_full_choice_acquisition_v1.json",
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(_json({"terminal": result["terminal"], "output": str(args.output)}))
