"""Bounded public development of hybrid evidence selection and executed edits.

The adjacent protocol and cases own the teaching/transfer contract. Expected
values enter only execution scoring and explicitly executed teacher corrections.
Inference receives a request and raw workspace; all document-derived candidates
remain in the disclosed external store. No protected evaluation is accessed.
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
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from benchmarks.native_scaling import _process_memory
from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
from experiments.r4_integrated.neutral_document_python_session import Case, documents, execute
from experiments.reading_exam_development.run import score, validate_inputs
from humanbrain.contextual_evidence_memory import ContextualEvidenceMemory
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import loaded_build_info
from humanbrain.opaque_document_event_tape import OpaqueDocument

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXAM = ROOT / "experiments/reading_exam_development"
SOURCE_PATHS = (
    "experiments/r4_integrated/contextual_evidence_python_session.py",
    "experiments/r4_integrated/contextual_occurrence_motor.py",
    "experiments/r4_integrated/contextual_evidence_protocol.md",
    "experiments/r4_integrated/contextual_evidence_cases.json",
    "experiments/r4_integrated/neutral_document_python_session.py",
    "experiments/reading_exam_development/run.py",
    "experiments/reading_exam_development/passage.txt",
    "experiments/reading_exam_development/questions.json",
    "experiments/reading_exam_development/rubric.json",
    "experiments/reading_exam_development/manifest.json",
    "src/humanbrain/contextual_evidence_memory.py",
    "src/humanbrain/frozen_context_encoder.py",
    "src/humanbrain/knowledge_conditioned_occurrence_substitution.py",
    "src/humanbrain/variability_aware_segment_binder.py",
    "src/humanbrain/resting_glif_race.py",
    "src/humanbrain/native.py",
    "benchmarks/native_scaling.py",
)


def snapshot(memory: ContextualEvidenceMemory) -> dict:
    arrays = memory.checkpoint_arrays()
    difference = arrays["readout_synapses_pa"][0] - arrays["readout_synapses_pa"][1]
    return {
        "checkpoint_sha256": memory.checkpoint_sha256,
        "synapse_sha256": memory.synapse_sha256,
        "storage": memory.storage_summary,
        "weight_difference_l2": float(np.linalg.norm(difference)),
        "bias_difference": float(np.diff(arrays["pacemaker_synapses_pa"])[0] * -1),
        "feature_block_l2": {
            name: float(np.linalg.norm(difference[left:right]))
            for name, left, right in (
                ("query", 0, 384),
                ("span", 384, 768),
                ("query_times_span", 768, 1152),
                ("query_times_window", 1152, 1536),
                ("lexical_overlap", 1536, 1537),
            )
        },
    }


def decision_record(decision) -> dict:
    row = asdict(decision)
    eligibility = np.asarray(row.pop("eligibility"), dtype=np.float64)
    row["eligibility_sha256"] = sha256(eligibility.tobytes()).hexdigest()
    row["query"] = decision.query.decode("utf-8")
    row["emitted"] = None if decision.emitted is None else decision.emitted.decode("utf-8")
    row["stopped"] = decision.stopped
    return row


def attempt(memory, policy, motor, item, root) -> tuple[dict, object]:
    case = Case(
        **{key: item[key] for key in ("name", "key", "module", "stale")},
        value=item["value"] or "__unsupported__",
    )
    before = case.files()
    parent, motor_parent = memory.checkpoint_sha256, motor.checkpoint_sha256
    began = time.perf_counter()
    decision = getattr(memory, policy)(item["request"].encode())
    selection_seconds = time.perf_counter() - began
    action = None
    if not decision.stopped:
        action = motor.apply(item["request"].encode(), before, decision.emitted)
    files = before if action is None else action.files
    execution = execute(case, files, before, root)
    if memory.checkpoint_sha256 != parent or motor.checkpoint_sha256 != motor_parent:
        raise RuntimeError("inference changed the memory or shared motor")
    return {
        "case": case.name,
        "policy": policy,
        "selection": decision_record(decision),
        "selection_seconds": selection_seconds,
        "live_document_count": 0,
        "source_storage": "all raw documents and derived candidate vectors retained",
        "expected_value": item["value"],
        "evidence_correct": decision.emitted
        == (None if item["value"] is None else item["value"].encode()),
        "edited": action is not None and action.edited,
        "action_stop_reason": "evidence_STOP" if action is None else action.stop_reason,
        "motor": None
        if action is None
        else {
            key: getattr(action, key)
            for key in (
                "request_seed_count",
                "supported_role_count",
                "physical_cell_steps",
                "evaluated_cells",
                "checkpoint_sha256",
            )
        },
        "execution": execution,
        "correct_no_action": item["value"] is None and decision.stopped and files == before,
    }, decision


def aggregate(rows: list[dict]) -> dict:
    supported = [row for row in rows if row["expected_value"] is not None]
    unsupported = [row for row in rows if row["expected_value"] is None]
    return {
        "supported_tasks": len(supported),
        "evidence_correct": sum(row["evidence_correct"] for row in supported),
        "executed_success": sum(row["execution"]["passed"] for row in supported),
        "unsupported_tasks": len(unsupported),
        "correct_no_action": sum(row["correct_no_action"] for row in unsupported),
        "evidence_stops": sum(row["selection"]["stopped"] for row in rows),
        "action_stops_after_selection": sum(
            not row["selection"]["stopped"] and not row["edited"] for row in rows
        ),
        "decoy_failures": sum(not row["execution"]["decoys_unchanged"] for row in rows),
        "selection_seconds": sum(row["selection_seconds"] for row in rows),
    }


def run(output: Path, model_dir: Path) -> dict:
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("choose a fresh output path; earlier evidence is retained")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextual-python-", dir=runtime))
    began = time.perf_counter()
    tracemalloc.start()
    report = {
        "kind": "public repeatable development; hybrid candidate; no protected evaluation",
        "terminal": "running",
        "workspace_root": str(work),
        "source_sha256": {
            path: sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS
        },
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "case_exposure": "fixed before first candidate outputs; self-authored development",
        "cell_step_accounting": "equivalent GLIF simulation work, not CPU instructions; "
        "failed evidence races report the conservative full 600-step bound",
        "controls": "paired frozen synapses replay the updater's exact executed feedback; "
        "external and conventional references ablate only evidence selection; "
        "all arms share the same learned occurrence motor",
        "motor_demonstrations": [],
        "arms": {},
        "events": [],
    }

    def record(event, **fields):
        row = {"event": event, **fields}
        report["events"].append(row)
        with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "event": event,
                    **{k: v for k, v in fields.items() if k in ("arm", "case", "phase", "summary")},
                }
            ),
            flush=True,
        )

    try:
        fixtures = json.loads((HERE / "contextual_evidence_cases.json").read_text("utf-8"))
        motor = ContextualOccurrenceMotor()
        for case in (
            Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
            Case("motor-b", "bora", "ready", "motor_beta", "warm"),
        ):
            before, after = case.files(), case.files(corrected=True)
            execution = execute(case, after, before, work / case.name)
            if not execution["passed"]:
                raise RuntimeError("motor teacher correction did not execute successfully")
            motor.learn(f"Configure {case.key}".encode(), before, after)
            report["motor_demonstrations"].append({"case": case.name, "execution": execution})
        report["motor"] = {
            "checkpoint_sha256": motor.checkpoint_sha256,
            "demonstration_count": motor.demonstration_count,
            "retained_demonstration_event_bytes": motor.retained_demonstration_event_bytes,
        }
        load_began = time.perf_counter()
        encoder = FrozenContextEncoder(model_dir)
        report["encoder"] = encoder.metadata
        report["encoder_load_seconds"] = time.perf_counter() - load_began
        arrays = {}
        for rate in (1.0, 3.0):
            label = f"lr{int(rate)}"
            update = ContextualEvidenceMemory(encoder, learning_rate=rate)
            frozen = ContextualEvidenceMemory(encoder, learning_rate=rate, learning_enabled=False)
            seed = snapshot(update)
            assert update.checkpoint_sha256 == frozen.checkpoint_sha256
            arm = {
                "learning_rate": rate,
                "seed": seed,
                "training": [],
                "feedback": [],
                "transfer": {},
                "retention": {},
                "exam": {},
            }
            report["arms"][label] = arm

            def feedback(decision, reward, execution, *, update=update, frozen=frozen, arm=arm):
                # Bind the same executed action/outcome to the frozen control's
                # current parent before either learner's ledger advances.
                replayed = frozen.replay_decision(decision)
                first = update.observe_feedback(decision, reward)
                second = frozen.observe_feedback(replayed, reward)
                arm["feedback"].append(
                    {
                        "update": first,
                        "frozen": second,
                        "execution_workspace": execution["workspace"],
                    }
                )

            for item in fixtures["training"]:
                docs = documents(item["name"], item["document"])
                ingest_start = time.perf_counter()
                for memory in (update, frozen):
                    memory.ingest(docs)
                row = {
                    "case": item["name"],
                    "paired_ingestion_seconds": time.perf_counter() - ingest_start,
                }
                for name, memory in (("frozen_first", frozen), ("first", update)):
                    result, decision = attempt(
                        memory, "respond", motor, item, work / label / item["name"] / name
                    )
                    row[name] = result
                    if name == "first":
                        if result["edited"]:
                            feedback(
                                decision,
                                1 if result["execution"]["passed"] else -1,
                                result["execution"],
                            )
                        elif decision.stopped:
                            exploration, proposal = attempt(
                                update,
                                "external_only",
                                motor,
                                item,
                                work / label / item["name"] / "exploration",
                            )
                            row["exploration"] = exploration
                            if exploration["edited"]:
                                feedback(
                                    proposal,
                                    1 if exploration["execution"]["passed"] else -1,
                                    exploration["execution"],
                                )
                case = Case(
                    **{key: item[key] for key in ("name", "key", "value", "module", "stale")}
                )
                correction = execute(
                    case,
                    case.files(corrected=True),
                    case.files(),
                    work / label / item["name"] / "teacher",
                )
                row["teacher_execution"] = correction
                if not correction["passed"]:
                    raise RuntimeError("evidence teacher correction failed execution")
                try:
                    teacher = update.decision_for_span(
                        item["request"].encode(), item["value"].encode()
                    )
                except ValueError as error:
                    row["teacher_rejection"] = str(error)
                else:
                    feedback(teacher, 1, correction)
                row["after"] = snapshot(update)
                arm["training"].append(row)
                record(
                    "teaching",
                    arm=label,
                    case=item["name"],
                    summary={
                        "first_pass": row["first"]["execution"]["passed"],
                        "updates": update.update_count,
                    },
                )

            arm["trained"] = snapshot(update)
            arm["frozen_trained"] = snapshot(frozen)
            for item in fixtures["transfer"]:
                docs = documents(item["name"], item["document"])
                for memory in (update, frozen):
                    memory.ingest(docs)
            arm["transfer_start"] = snapshot(update)
            policies = (
                ("update", update, "respond"),
                ("frozen", frozen, "respond"),
                ("external", frozen, "external_only"),
                ("conventional", update, "conventional"),
            )
            for phase, cases in (
                ("transfer", fixtures["transfer"]),
                ("retention", fixtures["training"]),
            ):
                for name, memory, policy in policies:
                    rows = [
                        attempt(
                            memory, policy, motor, item, work / label / phase / name / item["name"]
                        )[0]
                        for item in cases
                    ]
                    arm[phase][name] = {"summary": aggregate(rows), "rows": rows}
                    record("evaluation", arm=label, phase=phase, summary={name: aggregate(rows)})
            assert frozen.synapse_sha256 == seed["synapse_sha256"]
            assert update.synapse_sha256 == arm["trained"]["synapse_sha256"]
            assert update.feedback_count == frozen.feedback_count
            for name, memory in (("update", update), ("frozen", frozen)):
                for key, value in memory.checkpoint_arrays().items():
                    arrays[f"{label}_{name}_{key}"] = value

            # Untuned, original exam questions. Earlier questions remain in
            # context as required by their published shared-stem interface.
            validate_inputs(evaluator=True)
            passage = (EXAM / "passage.txt").read_bytes()
            questions = json.loads((EXAM / "questions.json").read_text("utf-8"))
            rubric = json.loads((EXAM / "rubric.json").read_text("utf-8"))
            document = OpaqueDocument(b"ks2-2024-passage-1", b"sta-24-8811", passage)
            for memory in (update, frozen):
                memory.ingest((document,))
            for name, memory, policy in policies:
                context, events = questions["shared_stem"], []
                parent = memory.checkpoint_sha256
                for part in questions["parts"]:
                    context += "\n\n" + part["text"]
                    decision = getattr(memory, policy)(context.encode())
                    events.append(
                        {
                            "event": "answer",
                            "item": part["id"],
                            "stopped": decision.stopped,
                            "emitted_hex": None
                            if decision.emitted is None
                            else decision.emitted.hex(),
                            "selection": decision_record(decision),
                        }
                    )
                assert memory.checkpoint_sha256 == parent
                arm["exam"][name] = {
                    "events": events,
                    "scoring": score(events, rubric, "completed"),
                }
            arm["end"] = snapshot(update)
            arm["frozen_end"] = snapshot(frozen)
            record(
                "exam",
                arm=label,
                summary={name: result["scoring"] for name, result in arm["exam"].items()},
            )
        checkpoint = output.with_suffix(".npz")
        np.savez_compressed(checkpoint, **arrays)
        report["synapse_checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            "scope": "trained synapses; reconstruct storage from ordered fixtures and encoder",
        }
        native = loaded_build_info()
        report["native"] = {
            "source_digest": native.source_digest,
            "library": str(native.library_path),
            "binary_sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        report["sources_unchanged_during_run"] = all(
            sha256((ROOT / path).read_bytes()).hexdigest() == digest
            for path, digest in report["source_sha256"].items()
        )
        if not report["sources_unchanged_during_run"]:
            raise RuntimeError("source files changed during the comparison")
        report["terminal"] = "completed"
    except Exception:
        report["terminal"] = "error"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - began
        report["python_traced_peak_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        report["process_memory"] = _process_memory() if os.name == "nt" else None
        report["resource_scope"] = (
            "whole sequential comparison process with shared encoder, paired memories, "
            "evaluator and result records; excludes child executors"
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output.resolve(), args.model_dir)
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "output": str(args.output),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )
