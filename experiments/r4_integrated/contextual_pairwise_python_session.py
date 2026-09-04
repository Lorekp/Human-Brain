"""Fresh public comparison of absolute and pairwise executed corrective feedback.

The neighboring protocol and fixed cases own this run. Inference receives raw
requests/workspaces only. Both objectives need an actually executed wrong
proposal and a separately executed successful correction; neither invents a
negative from an answer key. Original experiment artifacts remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r4_integrated import contextual_evidence_python_session as shared
from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
from humanbrain.contextual_evidence_memory import ContextualEvidenceMemory
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import loaded_build_info
from humanbrain.opaque_document_event_tape import OpaqueDocument

HERE, ROOT, EXAM = shared.HERE, shared.ROOT, shared.EXAM
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *shared.SOURCE_PATHS,
            "experiments/r4_integrated/contextual_pairwise_python_session.py",
            "experiments/r4_integrated/contextual_pairwise_cases.json",
            "experiments/r4_integrated/contextual_pairwise_protocol.md",
        )
    )
)


def _source_hashes() -> dict[str, str]:
    return {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def _case(item):
    return shared.Case(**{key: item[key] for key in ("name", "key", "value", "module", "stale")})


def _summary(rows, cases):
    result = shared.aggregate(rows)
    by_entity = {}
    for row, case in zip(rows, cases, strict=True):
        if case["value"] is not None:
            by_entity.setdefault(case["key"], []).append(row["execution"]["passed"])
    pairs = [values for values in by_entity.values() if len(values) == 2]
    answers = Counter(
        row["selection"]["emitted"] for row in rows if row["selection"]["emitted"] is not None
    )
    result.update(
        {
            "entities_with_two_queries": len(pairs),
            "entities_with_both_queries_correct": sum(all(values) for values in pairs),
            "distinct_emitted_answers": len(answers),
            "most_frequent_answer_count": max(answers.values(), default=0),
        }
    )
    return result


def _motor(work, report):
    motor = ContextualOccurrenceMotor()
    report["motor_demonstrations"] = []
    for case in (
        shared.Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
        shared.Case("motor-b", "bora", "ready", "motor_beta", "warm"),
    ):
        before, after = case.files(), case.files(corrected=True)
        execution = shared.execute(case, after, before, work / case.name)
        report["motor_demonstrations"].append({"case": case.name, "execution": execution})
        if not execution["passed"]:
            raise RuntimeError("shared motor teacher correction did not execute successfully")
        motor.learn(f"Configure {case.key}".encode(), before, after)
    report["motor"] = {
        "checkpoint_sha256": motor.checkpoint_sha256,
        "demonstration_count": motor.demonstration_count,
        "retained_demonstration_event_bytes": motor.retained_demonstration_event_bytes,
    }
    return motor


def _ingest_once(item, memories, seen):
    identity, content = item["document_id"], item["document"]
    if identity in seen:
        if seen[identity] != content:
            raise ValueError("one document_id has conflicting raw content")
        return False
    documents = shared.documents(identity, content)
    for memory in memories:
        memory.ingest(documents)
    seen[identity] = content
    return True


def _correct(objective, update, frozen, rejected, accepted, row):
    """Apply identical available contrasts; only their learning objective differs."""
    if objective == "PAIRWISE":
        frozen_rejected = frozen.replay_decision(rejected)
        frozen_accepted = frozen.replay_decision(accepted)
        row["feedback"].append(
            {
                "update": update.observe_comparison(rejected, accepted),
                "frozen": frozen.observe_comparison(frozen_rejected, frozen_accepted),
            }
        )
    else:
        replayed = frozen.replay_decision(rejected)
        row["feedback"].append(
            {
                "update": update.observe_feedback(rejected, -1),
                "frozen": frozen.observe_feedback(replayed, -1),
            }
        )
        # The negative update changed the parent: reissue the already executed
        # teacher target against that new checkpoint, without executing it twice.
        accepted = update.decision_for_span(accepted.query, accepted.emitted)
        replayed = frozen.replay_decision(accepted)
        row["feedback"].append(
            {
                "update": update.observe_feedback(accepted, 1),
                "frozen": frozen.observe_feedback(replayed, 1),
            }
        )


def _exam(policies):
    shared.validate_inputs(evaluator=True)
    questions = json.loads((EXAM / "questions.json").read_text("utf-8"))
    rubric = json.loads((EXAM / "rubric.json").read_text("utf-8"))
    document = OpaqueDocument(
        b"ks2-2024-passage-1", b"sta-24-8811", (EXAM / "passage.txt").read_bytes()
    )
    for memory in {id(memory): memory for _, memory, _ in policies}.values():
        memory.ingest((document,))
    results = {}
    for name, memory, policy in policies:
        context, events, parent = questions["shared_stem"], [], memory.checkpoint_sha256
        for part in questions["parts"]:
            context += "\n\n" + part["text"]
            decision = getattr(memory, policy)(context.encode())
            events.append(
                {
                    "event": "answer",
                    "item": part["id"],
                    "stopped": decision.stopped,
                    "emitted_hex": None if decision.emitted is None else decision.emitted.hex(),
                    "selection": shared.decision_record(decision),
                }
            )
        if memory.checkpoint_sha256 != parent:
            raise RuntimeError("exam inference changed checkpoint")
        results[name] = {"events": events, "scoring": shared.score(events, rubric, "completed")}
    return results


def run(output: Path, model_dir: Path) -> dict:
    output = output.resolve()
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("choose a fresh output path; existing evidence is immutable")
    os.environ["HUMANBRAIN_DEVICE"] = "cpu"
    os.environ["HUMANBRAIN_NATIVE_CACHE"] = str(ROOT / ".native-cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextual-pairwise-", dir=runtime))
    started = time.perf_counter()
    shared.tracemalloc.start()
    report = {
        "kind": "public repeatable development; absolute versus pairwise corrective learning",
        "terminal": "running",
        "workspace_root": str(work),
        "arms": {},
        "events": [],
        "native_environment": {
            key: os.environ[key] for key in ("HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "feedback_contract": "Only executed wrong proposal versus executed successful teacher; "
        "correct proposals and absent executable negatives skip correction in both objectives",
        "retained_source_policy": "external storage retains raw source bytes and candidates",
    }

    def record(event, **fields):
        row = {"event": event, **fields}
        report["events"].append(row)
        with (work / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        report["source_sha256"] = _source_hashes()
        report["base_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        fixtures = json.loads((HERE / "contextual_pairwise_cases.json").read_text("utf-8"))
        if len(fixtures["training"]) != 16 or len(fixtures["transfer"]) != 10:
            raise ValueError("comparison requires exactly 16 training and 10 transfer requests")
        if sum(item["value"] is not None for item in fixtures["transfer"]) != 8:
            raise ValueError("transfer roster requires eight supported and two unsupported queries")
        motor = _motor(work, report)
        load_start = time.perf_counter()
        encoder = FrozenContextEncoder(model_dir)
        report["encoder"], report["encoder_load_seconds"] = (
            encoder.metadata,
            time.perf_counter() - load_start,
        )
        arrays = {}
        for objective in ("ABSOLUTE", "PAIRWISE"):
            update = ContextualEvidenceMemory(encoder, learning_rate=1.0)
            frozen = ContextualEvidenceMemory(encoder, learning_rate=1.0, learning_enabled=False)
            if update.checkpoint_sha256 != frozen.checkpoint_sha256:
                raise RuntimeError("paired arms did not start from identical state")
            seed = shared.snapshot(update)
            arm = {
                "objective": objective,
                "learning_rate": 1.0,
                "seed": seed,
                "training": [],
                "transfer": {},
                "retention": {},
            }
            report["arms"][objective] = arm
            seen = {}
            for item in fixtures["training"]:
                began = time.perf_counter()
                ingested = _ingest_once(item, (update, frozen), seen)
                row = {
                    "case": item["name"],
                    "document_id": item["document_id"],
                    "ingested": ingested,
                    "paired_ingestion_seconds": time.perf_counter() - began,
                    "feedback": [],
                }
                arm["training"].append(row)
                folder = work / objective / "training" / item["name"]
                row["frozen_first"], _ = shared.attempt(
                    frozen, "respond", motor, item, folder / "frozen"
                )
                row["first"], decision = shared.attempt(
                    update, "respond", motor, item, folder / "first"
                )
                candidate_result, candidate = row["first"], decision
                if decision.stopped:
                    row["exploration"], candidate = shared.attempt(
                        update, "external_only", motor, item, folder / "exploration"
                    )
                    candidate_result = row["exploration"]
                case = _case(item)
                correction = shared.execute(
                    case, case.files(corrected=True), case.files(), folder / "teacher"
                )
                row["teacher_execution"] = correction
                if not correction["passed"]:
                    raise RuntimeError("teacher correction did not execute successfully")
                if candidate_result["edited"] and candidate_result["execution"]["passed"]:
                    row["correction_status"] = "no_needed_correction"
                elif not candidate_result["edited"]:
                    row["correction_status"] = "no_executed_negative"
                else:
                    try:
                        teacher = update.decision_for_span(
                            item["request"].encode(), item["value"].encode()
                        )
                    except ValueError as error:
                        row["correction_status"], row["teacher_rejection"] = (
                            "ambiguous_teacher_span",
                            str(error),
                        )
                    else:
                        row["accepted_decision"] = shared.decision_record(teacher)
                        row["rejected_decision"] = shared.decision_record(candidate)
                        row["rejected_execution"] = candidate_result["execution"]["workspace"]
                        _correct(objective, update, frozen, candidate, teacher, row)
                        row["correction_status"] = "applied"
                row["after"], row["frozen_after"] = shared.snapshot(update), shared.snapshot(frozen)
                record(
                    "teaching",
                    objective=objective,
                    case=item["name"],
                    status=row["correction_status"],
                    first_success=row["first"]["execution"]["passed"],
                    updates=update.update_count,
                )
            if len(seen) != 8:
                raise ValueError("teaching roster did not contain eight independent documents")
            arm["trained"], arm["frozen_trained"] = shared.snapshot(update), shared.snapshot(frozen)
            for item in fixtures["transfer"]:
                _ingest_once(item, (update, frozen), seen)
            if len(seen) != 14:
                raise ValueError("transfer roster did not add six independent documents")
            arm["transfer_start"] = shared.snapshot(update)
            policies = (
                ("update", update, "respond"),
                ("frozen", frozen, "respond"),
                ("conventional", update, "conventional"),
                ("external", frozen, "external_only"),
            )
            for phase in ("transfer", "retention"):
                cases = fixtures["transfer" if phase == "transfer" else "training"]
                for name, memory, policy in policies:
                    rows = [
                        shared.attempt(
                            memory,
                            policy,
                            motor,
                            item,
                            work / objective / phase / name / item["name"],
                        )[0]
                        for item in cases
                    ]
                    arm[phase][name] = {"summary": _summary(rows, cases), "rows": rows}
                    record(
                        "evaluation",
                        objective=objective,
                        phase=phase,
                        policy=name,
                        summary=_summary(rows, cases),
                    )
                arm[phase + "_score_spike_disagreements"] = sum(
                    neuronal["selection"]["emitted"] != conventional["selection"]["emitted"]
                    for neuronal, conventional in zip(
                        arm[phase]["update"]["rows"],
                        arm[phase]["conventional"]["rows"],
                        strict=True,
                    )
                )
            if (
                frozen.synapse_sha256 != seed["synapse_sha256"]
                or update.synapse_sha256 != arm["trained"]["synapse_sha256"]
            ):
                raise RuntimeError("frozen or evaluation synapses changed")
            if update.feedback_count != frozen.feedback_count:
                raise RuntimeError("paired feedback exposure differs")
            for name, memory in (("update", update), ("frozen", frozen)):
                for key, value in memory.checkpoint_arrays().items():
                    arrays[f"{objective}_{name}_{key}"] = value
            arm["exam"] = _exam(policies)
            arm["end"], arm["frozen_end"] = shared.snapshot(update), shared.snapshot(frozen)
            record(
                "exam",
                objective=objective,
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
        report["sources_unchanged_during_run"] = _source_hashes() == report["source_sha256"]
        if not report["sources_unchanged_during_run"]:
            raise RuntimeError("source files changed during the comparison")
        report["terminal"] = "completed"
    except Exception:
        report["terminal"], report["error"] = "error", traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["python_traced_peak_bytes"] = shared.tracemalloc.get_traced_memory()[1]
        shared.tracemalloc.stop()
        report["process_memory"] = shared._process_memory() if os.name == "nt" else None
        report["resource_scope"] = (
            "whole sequential parent process including evaluator; child executors excluded"
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(json.dumps({"terminal": result["terminal"], "output": str(args.output)}))
