"""Fixed-architecture repeated teaching of complete, existing language/action tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT = False
MANIFEST_SHA = "528f6599d0ff5af487a418a7a0fa06b340ba0d026328df4bab534af671bd64cf"
BANK_SHA = "1bd3073fac22fdd54cc27e16ad7c8e75d106b15e9263e44bb1161b1a85c54919"
BASE_SHA = "7b07b6984d6511a5f854d530775a4e7d9772d45ab9ce4330c2e3f4a74259c481"
ROUTES = ("online", "learning_disabled", "simple")
CHECKPOINTS = (0, 8, 16, 32, 64)
ARTIFACT_CAP = 768 * 1024**2
PENDING_BYTES = 256 * 1024**2
FREE_RESERVE = 8 * 1024**3
SECONDS = 3600


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value, *, artifact_root=None):
    raw = encode(value)
    if len(raw) > 64 * 1024**2 or shutil.disk_usage(Path(path).parent).free - len(raw) < FREE_RESERVE:
        raise RuntimeError("receipt publication exceeds size/free-space bound")
    if artifact_root is not None:
        used = sum(p.stat().st_size for p in Path(artifact_root).rglob("*") if p.is_file())
        if used + len(raw) > ARTIFACT_CAP:
            raise RuntimeError("receipt publication exceeds complete artifact cap")
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def make_manifest(base):
    if sha(base) != BASE_SHA:
        raise ValueError("repetition requires exact previously exposed task bytes")
    prior = json.loads(Path(base).read_bytes())
    by_id = {e["episode_id"]: e for e in prior["episodes"]}
    names = ["A00-R"] * 8 + ["A00-E"] * 8 + ["A00-R", "A00-E"] * 24
    return {
        "schema": "repeated-teaching-acquisition-v1", "base_manifest_sha256": BASE_SHA,
        "parent_bank_sha256": BANK_SHA,
        "tasks": {k: by_id[k] for k in ("A00-R", "A00-E", "R02-R", "R12-R")},
        "prior_regression_bindings": prior["prior_regression_bindings"],
        "schedule": [{"index": i, "episode_id": k} for i, k in enumerate(names)],
        "routes": list(ROUTES), "checkpoints": list(CHECKPOINTS),
        "checkpoint_cases": ["A00-R", "A00-E", "R02-R", "R12-R"],
        "live_teacher_per_neural": {"occasions": 64, "raw_bytes": 1056,
                                    "eos": 64, "symbols": 1120, "motor_labels": 96},
        "diagnostic_teacher_per_neural": {"occasions": 10, "symbols": 175,
                                          "motor_labels": 15, "slow_publications": 0},
        "attempts": 252, "executions": 111, "new_source_offers": 0,
        "architecture": "unchanged causal-credit-v2; rate0.01; same persistent learner/actor",
        "durability": "authenticated parent and one final complete block; scalar checkpoints are nondurable",
        "budget": {"seconds": SECONDS, "artifact_bytes": ARTIFACT_CAP,
                   "pending_bytes": PENDING_BYTES, "free_reserve_bytes": FREE_RESERVE},
    }


def run(args):
    if DRAFT:
        raise RuntimeError("DRAFT: source/protocol review required before execution")
    started = time.perf_counter()
    deadline = started + SECONDS - 120
    output = Path(args.output).resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError("output must be a new local evidence directory")
    if shutil.disk_usage(ROOT).free < FREE_RESERVE + 1024**3:
        raise RuntimeError("run requires9GiB free before initialization")
    output.mkdir(parents=True, exist_ok=False)
    def publish(name, value):
        save(output / name, value, artifact_root=output)
    manifest_path = Path(args.manifest).resolve()
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("run requires the root-frozen repeated-teaching manifest")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest != make_manifest(args.base):
        raise ValueError("manifest differs from the bounded fixed repetition schedule")
    publish("manifest.json", manifest)
    sources = {p.relative_to(ROOT).as_posix(): sha(p)
               for folder in (ROOT / "src/humanbrain", ROOT / "experiments/r4_integrated")
               for p in sorted(folder.glob("*.py"))}
    publish("sources.json", sources)
    live = {"completed_attempts": 0, "completed_teachers": 0, "executions": 0}
    pending = {}
    journals = []
    store = None

    def admit(reserve=True):
        used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        free = shutil.disk_usage(output).free
        live["disk"] = {"artifact_bytes": used, "free_bytes": free}
        if (used + (PENDING_BYTES if reserve else 0) > ARTIFACT_CAP
                or free < FREE_RESERVE + (PENDING_BYTES if reserve else 0)):
            raise RuntimeError("bounded acquisition artifact/free-space limit")
        if time.perf_counter() >= (deadline if reserve else started + SECONDS - 5):
            raise TimeoutError("acquisition complete-run reserve reached")

    try:
        import torch

        from experiments.r4_integrated.persistent_language_action_execution_v2 import (
            grade_execution,
            proposal_from_record,
            workspace_from_records,
        )
        from experiments.r4_integrated.persistent_language_action_v8 import _plain, reader_summary
        from experiments.r4_integrated.persistent_language_action_v8_execution import (
            execute_proposal,
            qualified_launch,
        )
        from experiments.r4_integrated.repeated_teaching_checkpoint_v1 import (
            RepeatedTeachingCheckpoint,
            audit_saved,
            clone_for_probe,
            initialize_retained,
        )
        from experiments.r4_integrated.repeated_teaching_diagnostics_v1 import (
            analyze_teacher,
            capture_literal,
            free_summary,
        )
        from humanbrain.neuronal_splice_learner import _digest
        from humanbrain.recurrent_content_learner import RecurrentContentSpliceLearner
        from humanbrain.sequence_assembly_observation import RawObservation, ToolFeedback

        torch.set_num_threads(1)
        admit()
        initialized = initialize_retained(
            Path(args.prior), Path(args.bank), bank_sha256=BANK_SHA,
            actor_class=RecurrentContentSpliceLearner,
            frozen_auditor_path=Path(args.frozen_auditor))
        models, states, baseline = (initialized[k] for k in ("models", "states", "baseline"))
        publish("initialization.json", initialized["receipt"])
        store = RepeatedTeachingCheckpoint.create(
            output / "checkpoints", evidence_root=output,
            manifest_path=output / "manifest.json", manifest_sha256=sha(output / "manifest.json"),
            source_sha256=sources, source_revision=args.source_revision,
            parent_receipt=initialized["receipt"])
        owned = store.create_owned_journals()
        journals = list(owned.values())
        launch = qualified_launch(ROOT)
        publish("launch.json", launch.to_record())
        work = {}

        def hashes(actor, state):
            return {"model": _digest(actor.state_dict(), work), "memory": _digest(vars(state), work)}

        def baseline_hashes(reference):
            return {"model": _digest(_plain(reference.demonstrations), work),
                    "memory": _digest(_plain(reference.documents), work)}

        initial = {r: hashes(models[r], states[r]) for r in models}
        initial["simple"] = baseline_hashes(baseline)
        diagnostic_rng = models["online"].reader.controller.rng_state.detach().clone()
        disabled_slow = _digest({k: v for k, v in models["learning_disabled"].state_dict().items()
                                 if k != "reader.controller.rng_state"}, work)
        results, probes = [], []

        def attempt(route, episode_id, actor, state, reference, *, scope, index, teaching):
            admit()
            tick = time.perf_counter()
            episode = manifest["tasks"][episode_id]
            files, feedback = episode["files"], None
            if episode_id.startswith("R"):
                retained = manifest["prior_regression_bindings"]["routes"][
                    "simple" if route == "simple" else "online"][episode["prior_episode_id"]]["observation"]
                files = retained["files"] or []
                if retained["feedback"] is not None:
                    f = retained["feedback"]
                    feedback = ToolFeedback(**{k: v for k, v in f.items()
                                               if k not in ("stdout_hex", "stderr_hex")},
                                            stdout=bytes.fromhex(f["stdout_hex"]),
                                            stderr=bytes.fromhex(f["stderr_hex"]))
            workspace = workspace_from_records(files) if episode["kind"] == "edit" else None
            observation = RawObservation(bytes.fromhex(episode["request_hex"]), workspace, feedback)
            before = baseline_hashes(reference) if actor is None else hashes(actor, state)
            row = {"scope": scope, "step_index": index, "route": route, "episode_id": episode_id,
                   "kind": episode["kind"], "observation": observation.to_record(),
                   "model_before_sha256": before["model"], "memory_before_sha256": before["memory"],
                   "execution": None, "proposal": None, "answer_hex": None}
            pending["attempt"] = row
            prediction = proposal = envelope = None
            if actor is None:
                row["prediction"] = reference.predict(observation, kind=episode["kind"])
                if workspace is None:
                    row["answer_hex"] = row["prediction"]["answer_hex"]
                else:
                    proposal = proposal_from_record(row["prediction"]["proposal"], workspace)
            elif workspace is None:
                prediction = actor.reader.predict_with_receipt(observation, state, 128)
                row["prediction"] = reader_summary(prediction)
                row["answer_hex"] = prediction.generation.data.hex()
            else:
                prediction = actor.propose(observation, state)
                proposal = prediction.proposal
                row["prediction"] = {"payload": reader_summary(prediction.payload),
                                     "heads": {k: {"logits": v.logits.tolist(), "selected_index": v.selected_index}
                                               for k, v in prediction.choices.items()}}
            if workspace is not None:
                row["proposal"] = proposal.to_record()
                after_files, execution = execute_proposal(
                    workspace, proposal, output / "executions" / scope / str(index) / route / episode_id,
                    launch=launch, deadline=deadline)
                live["executions"] += 1
                row.update(execution=execution, after_files=after_files)
                row["grade"] = grade_execution(execution, episode["grade"], episode["files"])
                row["correct"] = row["grade"]["task_success"]
                if prediction is not None:
                    envelope = {"attempt_id": prediction.attempt_id,
                                "proposal_sha256": prediction.proposal_sha256,
                                "workspace_sha256": workspace.sha256, "execution": execution}
            else:
                row["correct"] = row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                if prediction is not None:
                    row["correct"] = row["correct"] and prediction.generation.stop_reason == "eos"
            enabled = scope == "main" and route == "online"
            teacher_row = None
            if teaching:
                teacher = episode["teacher"]
                teacher_before = baseline_hashes(reference) if actor is None else hashes(actor, state)
                target = bytes.fromhex(teacher["target_hex"] if workspace is None else teacher["replacement_hex"])
                teacher_row = {"scope": scope, "step_index": index, "route": route,
                               "episode_id": episode_id, "teacher": teacher,
                               "learning_enabled": enabled, "reader_offers": 0, "action_labels": 0,
                               "model_before_sha256": teacher_before["model"]}
                pending["teacher"] = teacher_row
                if actor is None:
                    teacher_row["publication"] = (
                        reference.observe_teacher(observation, kind=episode["kind"], teacher=teacher)
                        if scope == "main" else {"published": False, "scope": "probe reference history held"})
                else:
                    literal = capture_literal(actor.reader)
                    if workspace is None:
                        taught = actor.reader.teach(observation, target, state, 0.01,
                                                  prediction=prediction, learning_enabled=enabled)
                        rf = taught
                    else:
                        taught = actor.teach(prediction, observation, state,
                                             proposal_from_record(teacher, workspace), envelope,
                                             learning_enabled=enabled)
                        rf = taught.reader_feedback
                        teacher_row["action_labels"] = len(taught.action_losses)
                        teacher_row["actions"] = _plain(taught.action_updates)
                    teacher_row.update(reader_offers=rf.teacher_count,
                                       teacher_steps=_plain(rf.teacher_steps),
                                       controller=_plain(rf.controller), work=_plain(taught.work))
                    teacher_row["diagnostic"] = analyze_teacher(
                        rf, target, literal_snapshot=literal, learning_enabled=enabled)
                    row["free_response"] = free_summary(prediction if workspace is None else prediction.payload, target)
            elif actor is not None:
                row["ack_work"] = (actor.reader.acknowledge(prediction, state, learning_enabled=False)
                                   if workspace is None else actor.finish_without_teaching(
                                       prediction, observation, state, envelope, learning_enabled=False))
            after = baseline_hashes(reference) if actor is None else hashes(actor, state)
            if before["memory"] != after["memory"]:
                raise ValueError("repeated teaching changed factual/source memory")
            row.update(model_after_sha256=after["model"], memory_after_sha256=after["memory"],
                       elapsed_seconds=time.perf_counter() - tick)
            if teacher_row is not None:
                teacher_row["model_after_sha256"] = after["model"]
                teacher_row["memory_before_sha256"] = before["memory"]
                teacher_row["memory_after_sha256"] = after["memory"]
            live["completed_attempts"] += 1
            live["completed_teachers"] += int(teaching)
            return row, teacher_row

        def checkpoint(index):
            before = {r: hashes(models[r], states[r]) for r in models}
            counters_before = {r: (m._experience, m.reader._active_experience) for r, m in models.items()}
            before["simple"] = baseline_hashes(baseline)
            rows, clone_receipts = [], {}
            pending["probe_rows"] = rows
            for route in ROUTES:
                actor, state, reference = (None, None, copy.deepcopy(baseline))
                if route != "simple":
                    cloned = clone_for_probe(models[route], states[route])
                    actor, state = cloned["model"], cloned["state"]
                    actor.reader.controller.rng_state.copy_(diagnostic_rng)
                    clone_receipts[route] = {"work": cloned["work"],
                        "incoming_rng_sha256": _digest({"rng_state": diagnostic_rng}, work),
                        "policy": "fixed initial parent RNG on disposable panel only"}
                for name in manifest["checkpoint_cases"]:
                    row, teacher = attempt(route, name, actor, state, reference, scope="probe", index=index,
                                           teaching=name.startswith("A00"))
                    rows.append({"attempt": row, "teacher": teacher})
                del actor, state, reference
            after = {r: hashes(models[r], states[r]) for r in models}
            after["simple"] = baseline_hashes(baseline)
            if before != after or counters_before != {
                    r: (m._experience, m.reader._active_experience) for r, m in models.items()}:
                raise ValueError("disposable diagnostics changed persistent learner")
            record = {"checkpoint": index, "live_before": before, "live_after": after,
                      "clone_receipts": clone_receipts, "rows": rows}
            owned["probes"].append(record)
            probes.append(record)
            pending.clear()
            print(json.dumps({"checkpoint": index, "free_correct": {
                r: sum(x["attempt"]["correct"] for x in rows if x["attempt"]["route"] == r)
                for r in ROUTES}, "elapsed": time.perf_counter() - started}), flush=True)
            admit()

        checkpoint(0)
        for item in manifest["schedule"]:
            for route in ROUTES:
                row, teacher = attempt(route, item["episode_id"], models.get(route), states.get(route),
                                       baseline, scope="main", index=item["index"], teaching=True)
                owned["attempts"].append(row)
                owned["teachers"].append(teacher)
                pending.clear()
                results.append({k: row[k] for k in ("step_index", "route", "episode_id", "correct")})
                if _digest({k: v for k, v in models["learning_disabled"].state_dict().items()
                            if k != "reader.controller.rng_state"}, work) != disabled_slow:
                    raise ValueError("learning-disabled slow state changed")
            if item["index"] + 1 in CHECKPOINTS:
                checkpoint(item["index"] + 1)
        host = {"initial_hashes": initial, "counts": live, "schedule_completed": 64,
                "results": results, "work": work, "elapsed_seconds": time.perf_counter() - started}
        admit()
        final = store.commit_final(models, states, baseline, host)
        for journal in journals:
            journal.close()
        store.close()
        restore = audit_saved(output / "checkpoints", config_sha256=final["config_sha256"],
                              actor_class=RecurrentContentSpliceLearner)
        publish("restore.json", restore)
        from experiments.r4_integrated.repeated_teaching_review_v1 import review
        learning_review = review(output)
        publish("learning-review.json", learning_review)
        report = {"status": "completed", "manifest_sha256": sha(output / "manifest.json"),
                  "initial_hashes": initial, "final": final, "counts": live,
                  "restore": restore, "learning_review_sha256": sha(output / "learning-review.json"),
                  "results": results, "checkpoints": [{"checkpoint": p["checkpoint"],
                      "outcomes": [{k: x["attempt"][k] for k in ("route", "episode_id", "correct")}
                                   for x in p["rows"]]} for p in probes],
                  "elapsed_seconds": time.perf_counter() - started}
        report["acquisition"] = {r: {"both_skills_at_final_two_checkpoints": all(
            x["attempt"]["correct"] for p in probes if p["checkpoint"] in (32, 64)
            for x in p["rows"] if x["attempt"]["route"] == r
            and x["attempt"]["episode_id"].startswith("A00")),
            "complete_response_denominator": 4, "expansion_released": False} for r in ROUTES}
        if (live["completed_attempts"], live["executions"]) != (252, 111):
            raise ValueError("fixed exposure denominators differ")
        if any(sha(ROOT / p) != value for p, value in sources.items()):
            raise ValueError("source changed during run")
        publish("report.json", report)
        admit(False)
        print(json.dumps({"status": "completed", "counts": live, "seconds": time.perf_counter() - started}), flush=True)
    except BaseException as error:
        try:
            publish("failure.json", {"type": type(error).__name__, "error": str(error),
                                           "counts": live, "pending_evidence": pending,
                                           "elapsed_seconds": time.perf_counter() - started,
                                           "automatic_resume": False})
        except BaseException as reporting_error:
            print(f"Failure receipt error: {reporting_error!r}; original: {error!r}", file=sys.stderr)
        raise
    finally:
        primary_failure = sys.exc_info()[0] is not None
        cleanup_errors = []
        for journal in journals:
            try:
                if not journal.closed:
                    journal.close()
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "error": str(error)})
        if store is not None:
            try:
                store.close()
            except BaseException as error:
                cleanup_errors.append({"type": type(error).__name__, "error": str(error)})
        if cleanup_errors:
            try:
                publish("cleanup-errors.json", cleanup_errors)
            except BaseException as reporting_error:
                print(f"Cleanup receipt error: {reporting_error!r}", file=sys.stderr)
            if not primary_failure:
                raise RuntimeError("acquisition cleanup failed; saved cleanup errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--frozen-auditor", type=Path)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if args.command == "prepare":
        save(args.manifest, make_manifest(args.base))
        print(json.dumps({"manifest_sha256": sha(args.manifest), "model_calls": 0}), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
