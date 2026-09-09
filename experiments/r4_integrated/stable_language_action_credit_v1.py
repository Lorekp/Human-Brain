"""Source-only stable comparison worker; explicit runtime selection precedes models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT = False
MANIFEST_SHA = "7b07b6984d6511a5f854d530775a4e7d9772d45ab9ce4330c2e3f4a74259c481"
COMPLETE_SECONDS = 10800.0
AUDIT_SECONDS = 600.0
RUNNER_SECONDS = COMPLETE_SECONDS - AUDIT_SECONDS
CHECKPOINT_RESERVE_SECONDS = 300.0
PEAK_MIB = 4096
ARTIFACT_BYTES = 46 * 1024**3
INITIAL_FREE_BYTES = 54 * 1024**3
FREE_RESERVE_BYTES = 8 * 1024**3
# Admission headroom within ARTIFACT_BYTES for bounded parent/retained writes;
# this is not a filesystem quota against unrelated writers.
PENDING_WRITE_RESERVE_BYTES = 2 * 1024**3


def _run(args, output, started, live, initializer, checkpoint_callback, runtime_receipt):
    from experiments.r4_integrated.persistent_language_action_credit_runtime import check_origins
    from experiments.r4_integrated.persistent_language_action_v8 import (
        ROUTES,
        SessionJournal,
        _plain,
        encode,
        reader_summary,
        save,
        sha,
        summarize,
    )

    deadline = args.deadline
    disk_seconds, disk_checks = 0.0, 0

    def admit_work():
        nonlocal disk_seconds, disk_checks
        if time.perf_counter() + CHECKPOINT_RESERVE_SECONDS >= deadline:
            raise TimeoutError("whole-session reserve reached before admitting new work")
        tick = time.perf_counter()
        used = sum(path.stat().st_size for path in args.comparison_root.rglob("*") if path.is_file())
        free = shutil.disk_usage(output).free
        disk_seconds += time.perf_counter() - tick
        disk_checks += 1
        live["disk"] = {"artifact_bytes": used, "free_bytes": free,
                        "pending_write_reserve_bytes": PENDING_WRITE_RESERVE_BYTES,
                        "pending_write_scope": "bounded parent and retained writes within artifact cap",
                        "checks": disk_checks, "seconds": disk_seconds}
        if (used + PENDING_WRITE_RESERVE_BYTES > ARTIFACT_BYTES
                or free < FREE_RESERVE_BYTES + PENDING_WRITE_RESERVE_BYTES):
            raise RuntimeError("insufficient artifact/free-disk headroom for the next bounded writes")

    admit_work()
    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        grade_execution,
        proposal_from_record,
        workspace_from_records,
    )
    from experiments.r4_integrated.persistent_language_action_v8_checkpoint import controller_hashes
    from experiments.r4_integrated.persistent_language_action_v8_execution import (
        execute_proposal,
        feedback_from_receipt,
        qualified_launch,
    )
    from experiments.r4_integrated.persistent_session_checkpoint import COUNTERS
    from experiments.r4_integrated.stable_language_action_credit_checkpoint_v1 import (
        StableLanguageActionCreditCheckpoint,
    )
    from experiments.r4_integrated.stable_language_action_credit_workload_v1 import (
        CHECKPOINTS,
        check_saved_manifest,
    )

    if MANIFEST_SHA is None or args.manifest_sha != MANIFEST_SHA or sha(args.manifest) != MANIFEST_SHA:
        raise ValueError("stable comparison requires its root-frozen manifest")
    manifest_tick = time.perf_counter()
    check_saved_manifest(args.manifest, MANIFEST_SHA, base_path=args.base_manifest)
    manifest_validation_seconds = time.perf_counter() - manifest_tick
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["schema"] != "stable-language-action-credit-stream-v1"
            or manifest["continuation"]["initial_memory_slots"] != 1982):
        raise ValueError("stable continuation manifest differs")
    documents = {row["document_id"]: row for row in manifest["documents"]}
    if (any(len(e["append_document_ids"]) > 1 for e in manifest["episodes"])
            or [key for e in manifest["episodes"] for key in e["append_document_ids"]]
            != list(documents)):
        raise ValueError("each whole source must occur once in fixed public order")
    final_memory = 1982 + sum(len(bytes.fromhex(d["text_hex"])) + 1
                             for d in documents.values() if d["neural_tape_admission"])
    if final_memory > 2048:
        raise ValueError("declared factual admissions exceed retained capacity")
    expected_attempts = sum(e["attempt_limit"] for e in manifest["episodes"]) * 3
    expected_children = sum(e["attempt_limit"] for e in manifest["episodes"] if e["kind"] == "edit") * 3
    expected_teachers = sum(e.get("teacher") is not None for e in manifest["episodes"]) * 3
    with (output / "manifest.json").open("xb") as handle:
        handle.write(manifest_bytes)
    if sha(output / "manifest.json") != MANIFEST_SHA:
        raise ValueError("manifest publication changed bytes")
    capture_tick = time.perf_counter()
    sources = {p.relative_to(ROOT).as_posix(): sha(p)
               for directory in (ROOT / "src/humanbrain", ROOT / "experiments/r4_integrated")
               for p in sorted(directory.glob("*.py"))}
    sources.update(runtime_receipt["source_sha256"])
    save(output / "sources.json", sources)
    source_capture_seconds = time.perf_counter() - capture_tick
    tick = time.perf_counter()
    store = StableLanguageActionCreditCheckpoint.create(
        output / "checkpoints", source_root=ROOT, evidence_root=output,
        manifest_path=output / "manifest.json", manifest_sha256=MANIFEST_SHA,
        source_revision=args.source_revision, source_sha256=sources, session_id=args.session_id)
    live["store"] = store
    create_seconds = time.perf_counter() - tick
    admit_work()  # Initialization intent is durable before any model/decoder call.
    preinitialization_free = shutil.disk_usage(output).free
    if preinitialization_free < FREE_RESERVE_BYTES:
        raise RuntimeError("comparison free-disk reserve exhausted before initialization")
    import torch

    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.sequence_assembly_observation import RawObservation, ToolFeedback

    parent_work = {}
    live["parent_custody_work"] = parent_work
    torch.set_num_threads(1)
    initialized = initializer(args.prior, args.bank)
    runtime_receipt["initialization_origins"] = check_origins(runtime_receipt)
    models, states, baseline = (initialized[name] for name in ("models", "states", "baseline"))
    initialization = initialized["receipt"]
    if set(models) != {"online", "learning_disabled"}:
        raise ValueError("initializer must produce two independent matched online descendants")
    save(output / "initialization.json", {
        **initialization, "runtime": runtime_receipt, "prior_host": initialized["prior_host"],
        "prior_frontiers": initialized["prior_frontiers"],
        "baseline_record": initialized["baseline_record"],
    })
    launch = qualified_launch(ROOT)
    save(output / "launch-record.json", launch.to_record())
    owned = store.create_owned_journals(
        {name: [name + ".jsonl"] for name in ("attempts", "teachers", "ingests")})
    journals = {name: owned[name + ".jsonl"] for name in ("attempts", "teachers", "ingests")}
    checkpoint_journal = SessionJournal(output / "checkpoints.jsonl")
    live["journals"] = [*journals.values(), checkpoint_journal]
    projects = {route: {} for route in ROUTES}
    host = {
        "cursor": {"episode_index": 0, "route_index": 0, "attempt_index": 1, "append_done": False},
        "projects": projects, "current": None,
        "counters": {r: dict.fromkeys(COUNTERS, 0) for r in ROUTES},
        "elapsed_seconds": time.perf_counter() - started,
        "budget_seconds": COMPLETE_SECONDS, "budget_mib": PEAK_MIB,
        "initial_bindings": {name: sha(output / name) for name in
                             ("initialization.json", "manifest.json", "sources.json", "launch-record.json")},
    }
    checkpoint_timing = dict.fromkeys(("create", "begin", "capture", "commit", "journal"), 0.0)
    checkpoint_timing["create"] = create_seconds

    def checkpoint_event(phase, tick, before, result=None):
        elapsed = time.perf_counter() - tick
        checkpoint_timing[phase] += elapsed
        work = dict(store.work)
        row = {"schema": "persistent-language-action-checkpoint-work-v8", "phase": phase,
               "cursor": dict(host["cursor"]), "elapsed_seconds": elapsed,
               "session_elapsed_seconds": time.perf_counter() - started,
               "work_delta": {k: v - before.get(k, 0) for k, v in work.items()},
               "cumulative_work": work, "result": result}
        write = time.perf_counter()
        checkpoint_journal.append(row)
        checkpoint_timing["journal"] += time.perf_counter() - write
        live["checkpoint_active"] = None

    journal_tick = time.perf_counter()
    checkpoint_journal.append({"schema": "persistent-language-action-checkpoint-work-v8",
                               "phase": "create", "elapsed_seconds": create_seconds,
                               "cumulative_work": dict(store.work)})
    checkpoint_timing["journal"] += time.perf_counter() - journal_tick

    def begin(kind, observation_sha256=None):
        admit_work()
        tick, before = time.perf_counter(), dict(store.work)
        live["checkpoint_active"] = {"phase": "begin", "started": tick, "work_before": before}
        value = store.begin(kind, host["cursor"], observation_sha256=observation_sha256)
        checkpoint_event("begin", tick, before, {"intent_sha256": value})
        # If begin itself spends the reserve, preserve its unresolved intent.
        admit_work()

    def commit_owned():
        tick, before = time.perf_counter(), dict(store.work)
        live["checkpoint_active"] = {"phase": "capture", "started": tick, "work_before": before}
        frontier = store.capture_frontiers({name: [name + ".jsonl"] for name in journals},
                                          execution_root="executions")
        checkpoint_event("capture", tick, before)
        host["elapsed_seconds"] = time.perf_counter() - started
        tick, before = time.perf_counter(), dict(store.work)
        live["checkpoint_active"] = {"phase": "commit", "started": tick, "work_before": before}
        value = store.commit(models, states, baseline, host, frontier)
        live["last_safe"] = value
        checkpoint_event("commit", tick, before, value)
        if time.perf_counter() >= deadline:
            raise TimeoutError("whole-run deadline reached after safe publication")

    def commit():
        with store.checkpoint_scope():
            commit_owned()

    def hashes(route):
        if route == "simple":
            return (_digest(_plain(baseline.demonstrations), parent_work),
                    _digest(_plain(baseline.documents), parent_work))
        return (_digest(models[route].state_dict(), parent_work),
                _digest(vars(states[route]), parent_work))

    def held_reader(model):
        return _digest({k: v for k, v in model.reader.state_dict().items()
                        if not k.startswith("controller.")}, parent_work)

    def slow_actor(model):
        return _digest({k: v for k, v in model.state_dict().items()
                        if k != "reader.controller.rng_state"}, parent_work)

    initial_hashes = {r: dict(zip(("model", "memory"), hashes(r), strict=True)) for r in ROUTES}
    expected_memory = {r: row["memory"] for r, row in initial_hashes.items()}
    prior_hashes = {r: held_reader(m) for r, m in models.items()}
    disabled_slow = slow_actor(models["learning_disabled"])
    attempts, actual_children = [], 0
    commit()
    probe_receipts = []

    def probe(checkpoint_index):
        admit_work()
        before = {route: hashes(route) for route in ROUTES}
        counters_before = {route: (model._experience, model.reader._active_experience)
                           for route, model in models.items()}
        result = checkpoint_callback(
            checkpoint_index=checkpoint_index, checkpoint_directory=store.directory,
            boundary_sha256=live["last_safe"]["boundary_sha256"],
            config_sha256=store.config_sha256, manifest=manifest, output=output,
            deadline=deadline - CHECKPOINT_RESERVE_SECONDS, credit_rule=args.credit_rule,
            old_cp0_reference=args.old_cp0_reference if checkpoint_index == 0 else None,
            admit_work=admit_work)
        if before != {route: hashes(route) for route in ROUTES} or counters_before != {
                route: (model._experience, model.reader._active_experience)
                for route, model in models.items()}:
            raise ValueError("checkpoint probe changed a live actor, reference or counter")
        probe_receipts.append({"checkpoint_index": checkpoint_index, "receipt": result})
        save(output / f"probe-checkpoint-{checkpoint_index}.json", probe_receipts[-1])
        admit_work()

    probe(0)

    def controls(route):
        if route in models:
            if held_reader(models[route]) != prior_hashes[route]:
                raise ValueError("learning changed a held inherited reader tensor or metadata")
            if route == "learning_disabled" and slow_actor(models[route]) != disabled_slow:
                raise ValueError("disabled slow/controller/action state changed")

    def public_feedback(record):
        if record is None:
            return None
        return ToolFeedback(**{k: v for k, v in record.items()
                               if k not in ("stdout_hex", "stderr_hex")},
                            stdout=bytes.fromhex(record["stdout_hex"]),
                            stderr=bytes.fromhex(record["stderr_hex"]))

    for episode_index, episode in enumerate(manifest["episodes"]):
        for route_index, route in enumerate(ROUTES):
            admit_work()
            for document_id in episode["append_document_ids"]:
                begin("append")
                source_tick = time.perf_counter()
                document = documents[document_id]
                raw = bytes.fromhex(document["text_hex"])
                before_model, before_memory = hashes(route)
                if before_memory != expected_memory[route]:
                    raise ValueError("memory changed outside its declared source transaction")
                work, feedback = {}, None
                rng_before = rng_after = slow_before = slow_after = None
                used_before = used_after = None
                admitted = route != "simple" and document["neural_tape_admission"]
                if route == "simple":
                    baseline.append_documents([raw])
                else:
                    model, old = models[route], states[route]
                    before_controller = controller_hashes(model.reader, work)
                    rng_before, slow_before = (before_controller[k] for k in ("rng_sha256", "slow_sha256"))
                    used_before = old.used
                    prefix = _digest({k: getattr(old, k)[:used_before]
                                      for k in ("keys", "sensory", "successor")}, work)
                    grown, feedback = model.observe_source(raw, old, learning_enabled=route == "online")
                    for key, value in feedback.work.items():
                        work[key] = work.get(key, 0) + value
                    used_after = grown.used
                    if prefix != _digest({k: getattr(grown, k)[:used_before]
                                          for k in ("keys", "sensory", "successor")}, work):
                        raise ValueError("source transaction changed an old factual association")
                    if (used_after != used_before + (len(raw) + 1 if admitted else 0)
                            or (not admitted and grown is not old)):
                        raise ValueError("source mode changed its factual allocation semantics")
                    states[route] = grown
                    after_controller = controller_hashes(model.reader, work)
                    rng_after, slow_after = (after_controller[k] for k in ("rng_sha256", "slow_sha256"))
                    if rng_after == rng_before or (route == "learning_disabled" and slow_after != slow_before):
                        raise ValueError("source RNG publication or disabled controller changed")
                after_model, after_memory = hashes(route)
                if not admitted and route != "simple" and before_memory != after_memory:
                    raise ValueError("transient source changed the retained tape")
                controls(route)
                expected_memory[route] = after_memory
                journals["ingests"].append({
                    "schema": "persistent-language-action-ingest-v8", "route": route,
                    "episode_id": episode["episode_id"], "document_ids": [document_id],
                    "documents": [document], "raw_bytes": len(raw), "eos": 1,
                    "source_kind": document["source_kind"],
                    "neural_tape_admission": document["neural_tape_admission"],
                    "fact_admitted": admitted, "reference_retained": route == "simple",
                    "model_before_sha256": None if route == "simple" else before_model,
                    "model_after_sha256": None if route == "simple" else after_model,
                    "memory_before_sha256": None if route == "simple" else before_memory,
                    "memory_after_sha256": None if route == "simple" else after_memory,
                    "memory_used_before": used_before, "memory_used_after": used_after,
                    "controller_rng_before_sha256": rng_before, "controller_rng_after_sha256": rng_after,
                    "controller_slow_before_sha256": slow_before, "controller_slow_after_sha256": slow_after,
                    "learning_enabled": route != "learning_disabled", "writes_enabled": admitted,
                    "source_feedback": None if feedback is None else _plain(feedback),
                    "work": work, "elapsed_seconds": time.perf_counter() - source_tick,
                    "model_instance_id": id(baseline if route == "simple" else models[route]),
                })
                counts = host["counters"][route]
                counts["ingest_documents"] += 1
                counts["ingest_raw_bytes"] += len(raw)
                counts["ingest_eos"] += 1
                host["cursor"]["append_done"] = True
                commit()
                del feedback

            project_id, predecessor = episode.get("project_id"), episode.get("predecessor_episode_id")
            lifecycle = episode["workspace_lifecycle"]
            feedback_input = None
            if lifecycle == "carry_actual_project":
                previous = projects[route].get(project_id)
                if previous is None or previous["episode_id"] != predecessor:
                    raise ValueError("project must follow the actual same-route predecessor")
                files, feedback_input = previous["files"], public_feedback(previous["feedback"])
            elif episode["phase"] == "retention":
                parent_route = "simple" if route == "simple" else "online"
                saved = manifest["prior_regression_bindings"]["routes"][parent_route][episode["prior_episode_id"]]
                observed = saved["observation"]
                files, feedback_input = observed["files"] or [], public_feedback(observed["feedback"])
                if observed["request_hex"] != episode["request_hex"]:
                    raise ValueError("historical regression changed its actual request")
            else:
                files = episode["files"]
            teacher = episode.get("teacher")
            for attempt_index in range(1, episode["attempt_limit"] + 1):
                admit_work()
                tick = time.perf_counter()
                request = bytes.fromhex(episode["request_hex"])
                before_model, before_memory = hashes(route)
                if before_memory != expected_memory[route]:
                    raise ValueError("attempt changed its admitted memory before observation")
                controller_before = None if route == "simple" else controller_hashes(models[route].reader, parent_work)
                motor_before = None if route == "simple" else _digest(models[route].actions.state_dict(), parent_work)
                row = {
                    "schema": "persistent-language-action-attempt-v8", "route": route,
                    "episode_id": episode["episode_id"], "phase": episode["phase"],
                    "kind": episode["kind"], "attempt_index": attempt_index,
                    "encounter": episode.get("encounter"), "fresh_stratum": episode.get("fresh_stratum"),
                    "requirement_changed": bool(episode.get("requirement_changed")),
                    "project_id": project_id, "predecessor_episode_id": predecessor,
                    "workspace_lifecycle": lifecycle, "request_hex": request.hex(),
                    "observation": {"request_hex": request.hex(), "files": files,
                                    "feedback": None if feedback_input is None else feedback_input.to_record()},
                    "before_files": files, "after_files": files, "proposal": None,
                    "answer_hex": None, "prediction": None, "execution": None,
                    "status": "completed", "correct": False, "public_checks_passed": False,
                    "model_before_sha256": before_model, "memory_before_sha256": before_memory,
                    "memory_used": sum(len(d) + 1 for d in baseline.documents)
                    if route == "simple" else states[route].used,
                    "model_instance_id": id(baseline if route == "simple" else models[route]),
                    "controller_before": controller_before, "work": {}, "feedback_consumed": False,
                }
                prediction = workspace = proposal = envelope = observation = None
                try:
                    workspace = workspace_from_records(files) if episode["kind"] == "edit" else None
                    observation = RawObservation(request, workspace, feedback_input)
                    row["observation"] = observation.to_record()
                except (ValueError, TypeError) as error:
                    row["status"], row["error"] = "unsupported", str(error)
                begin_tick = time.perf_counter()
                observation_sha = hashlib.sha256(encode(row["observation"])).hexdigest()
                begin("attempt", observation_sha)
                row["included_checkpoint_begin_seconds"] = time.perf_counter() - begin_tick
                predict_tick = time.perf_counter()
                if observation is not None:
                    if route == "simple":
                        result = baseline.predict(observation, kind=episode["kind"])
                        row["prediction"] = result
                        if episode["kind"] == "reading":
                            row["answer_hex"] = result["answer_hex"]
                        else:
                            proposal = proposal_from_record(result["proposal"], workspace)
                    elif episode["kind"] == "reading":
                        prediction = models[route].reader.predict_with_receipt(observation, states[route], 128)
                        row["prediction"] = reader_summary(prediction)
                        row["answer_hex"] = prediction.generation.data.hex()
                        row["work"] = prediction.work
                    else:
                        prediction = models[route].propose(observation, states[route])
                        proposal = prediction.proposal
                        row["prediction"] = {
                            "attempt_id": prediction.attempt_id, "proposal_sha256": prediction.proposal_sha256,
                            "observation_sha256": prediction.observation_sha256,
                            "model_sha256": prediction.model_sha256, "memory_sha256": prediction.memory_sha256,
                            "payload": reader_summary(prediction.payload),
                            "heads": {k: {"logits": v.logits.tolist(), "selected_index": v.selected_index}
                                      for k, v in prediction.choices.items()},
                        }
                        row["work"] = prediction.work
                row["prediction_and_trace_seconds"] = time.perf_counter() - predict_tick
                row["prediction_timing_scope"] = "combined public scan/free/old prior plus trace conversion; no isolated scan clock"
                if route in models and hashes(route)[0] != before_model:
                    raise ValueError("unresolved free prediction published model/RNG state")
                if episode["kind"] == "edit":
                    row["proposal"] = None if proposal is None else proposal.to_record()
                    after, execution = execute_proposal(
                        workspace if workspace is not None else files, proposal,
                        output / "executions" / route / episode["episode_id"] / str(attempt_index),
                        launch=launch, deadline=deadline)
                    actual_children += 1
                    row["after_files"], row["execution"] = after, execution
                    row["public_checks_passed"] = bool(
                        execution["public_driver_policy"]["driver_integrity"]
                        and execution["returncode"] == 0 and not execution["timeout"]
                        and execution["capture_complete"] and execution["files_complete"]
                        and not execution["capture_errors"]
                        and not execution["output_limit_exceeded"])
                    row["grade"] = grade_execution(execution, episode["grade"], episode["files"])
                    row["correct"] = row["status"] != "unsupported" and row["grade"]["task_success"]
                    if prediction is not None:
                        envelope = {"attempt_id": prediction.attempt_id,
                                    "proposal_sha256": prediction.proposal_sha256,
                                    "workspace_sha256": workspace.sha256, "execution": execution}
                else:
                    row["correct"] = row["status"] != "unsupported" and row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                execution_outcome = None
                if row["execution"] is not None:
                    if proposal is None:
                        execution_outcome = "no_proposal"
                    elif not row["execution"]["accepted"]:
                        execution_outcome = "rejected_proposal"
                    elif proposal.operation == "STOP":
                        execution_outcome = "correct_stop" if row["correct"] else "incorrect_stop"
                    elif not row["correct"]:
                        execution_outcome = "incorrect_accepted_splice"
                    else:
                        changed = row["execution"]["proposed_files"] != row["execution"]["before_files"]
                        execution_outcome = ("accepted_correct_splice_with_change" if changed
                                             else "accepted_correct_splice_without_change")
                row["execution_outcome"] = execution_outcome
                if prediction is not None and teacher is None:
                    ack_tick = time.perf_counter()
                    if episode["kind"] == "edit":
                        row["ack_work"] = models[route].finish_without_teaching(
                            prediction, observation, states[route], envelope, learning_enabled=route == "online")
                    else:
                        row["ack_work"] = models[route].reader.acknowledge(
                            prediction, states[route], learning_enabled=route == "online")
                    row["controller_resolution"] = _plain(models[route].reader.last_controller_resolution)
                    row["acknowledgement_seconds"] = time.perf_counter() - ack_tick
                row["feedback_consumed"] = observation is not None and feedback_input is not None and row["prediction"] is not None
                row["model_after_sha256"], row["memory_after_sha256"] = hashes(route)
                if row["memory_after_sha256"] != before_memory:
                    raise ValueError("attempt changed factual memory")
                if route in models:
                    row["controller_after"] = controller_hashes(models[route].reader, parent_work)
                    if _digest(models[route].actions.state_dict(), parent_work) != motor_before:
                        raise ValueError("free or consequence phase changed motor weights")
                controls(route)
                row["observation_record_sha256"] = observation_sha
                row["proposal_signature"] = None if row["proposal"] is None else hashlib.sha256(encode(
                    {k: v for k, v in row["proposal"].items() if k != "workspace_sha256"})).hexdigest()
                row["elapsed_seconds"] = time.perf_counter() - tick
                attempt_sha = journals["attempts"].append(row)
                attempts.append({k: row[k] for k in (
                    "route", "episode_id", "phase", "kind", "attempt_index", "correct", "status",
                    "project_id", "predecessor_episode_id", "workspace_lifecycle",
                    "memory_used", "feedback_consumed", "proposal_signature", "public_checks_passed",
                    "execution_outcome")}
                    | {"observation_sha256": observation_sha})
                if row["execution"] is not None and not row["execution"]["files_complete"]:
                    raise ValueError("incomplete actual files cannot form a safe continuation")
                if teacher is not None:
                    teach_tick = time.perf_counter()
                    taught = {
                        "schema": "persistent-language-action-teacher-v8", "route": route,
                        "episode_id": episode["episode_id"], "attempt_sha256": attempt_sha,
                        "teacher": teacher, "learning_enabled": route != "learning_disabled",
                        "model_before_sha256": row["model_after_sha256"],
                        "memory_before_sha256": row["memory_after_sha256"],
                        "controller_before": row.get("controller_after"),
                        "reader_offers": 0, "action_labels": 0, "work": {},
                    }
                    if observation is None:
                        raise ValueError("scheduled correction requires a supported actual observation")
                    if route == "simple":
                        taught["publication"] = baseline.observe_teacher(observation, kind=episode["kind"], teacher=teacher)
                    else:
                        if episode["kind"] == "reading":
                            feedback = models[route].reader.teach(
                                observation, bytes.fromhex(teacher["target_hex"]), states[route], 0.01,
                                prediction=prediction, learning_enabled=route == "online")
                            reader_feedback, publication = feedback, {}
                        else:
                            feedback = models[route].teach(
                                prediction, observation, states[route], proposal_from_record(teacher, workspace),
                                envelope, learning_enabled=route == "online")
                            reader_feedback = feedback.reader_feedback
                            publication = {"actions": feedback.action_updates}
                            taught["action_labels"] = len(feedback.action_losses)
                        taught["work"] = feedback.work
                        if reader_feedback is not None:
                            taught["reader_offers"] = reader_feedback.teacher_count
                            taught["reader_feedback"] = {
                                "total_loss": reader_feedback.total_loss, "mean_loss": reader_feedback.mean_loss,
                                    "steps": _plain(reader_feedback.teacher_steps),
                                    "actual_response_id": prediction.attempt_id if episode["kind"] == "reading"
                                    else prediction.payload.attempt_id,
                                "controller": _plain(reader_feedback.controller),
                                "scope": "fresh causal teacher draws from common prescan state, monotonic RNG; old prior losses diagnostic",
                            }
                            publication.update(address=reader_feedback.address_updated,
                                               output=reader_feedback.output_updated,
                                               legacy_codec=reader_feedback.legacy_codec_updated)
                        taught["publication"] = publication
                        taught["controller_resolution"] = _plain(models[route].reader.last_controller_resolution)
                        taught["controller_after"] = controller_hashes(models[route].reader, parent_work)
                    taught["model_after_sha256"], taught["memory_after_sha256"] = hashes(route)
                    if taught["memory_after_sha256"] != before_memory:
                        raise ValueError("correction changed factual memory")
                    controls(route)
                    taught["elapsed_seconds"] = time.perf_counter() - teach_tick
                    journals["teachers"].append(taught)
                    counts = host["counters"][route]
                    counts["teacher_occasions"] += 1
                    counts["reader_offers"] += taught["reader_offers"]
                    counts["action_labels"] += taught["action_labels"]
                    if route != "simple":
                        del feedback, reader_feedback
                    del taught
                files = row["after_files"]
                if row["execution"] is not None:
                    feedback_input = feedback_from_receipt(row["execution"])
                actual = {"episode_id": episode["episode_id"], "files": files,
                          "feedback": None if feedback_input is None else feedback_input.to_record()}
                host["current"] = {"route": route, **actual}
                if project_id is not None:
                    projects[route][project_id] = actual
                host["counters"][route]["attempts"] += 1
                host["counters"][route]["executions"] += int(row["execution"] is not None)
                if attempt_index < episode["attempt_limit"]:
                    host["cursor"] = {"episode_index": episode_index, "route_index": route_index,
                                      "attempt_index": attempt_index + 1, "append_done": True}
                else:
                    host["cursor"] = {"episode_index": episode_index + int(route_index == 2),
                                      "route_index": (route_index + 1) % 3,
                                      "attempt_index": 1, "append_done": False}
                commit()
                del prediction, row

        if episode_index + 1 in CHECKPOINTS:
            probe(episode_index + 1)

    cleanup = _cleanup(live, close_store=False)
    if cleanup:
        raise cleanup[0]
    for route in models:
        controls(route)
        if states[route].used != final_memory:
            raise ValueError("final admitted factual count differs")
    if ({k: j.rows for k, j in journals.items()}
            != {"attempts": expected_attempts, "teachers": expected_teachers, "ingests": 3 * len(documents)}
            or actual_children != expected_children
            or store.work["checkpoint_head_publications"] != 1 + expected_attempts + 3 * len(documents)):
        raise ValueError("complete stream counts disagree with its fixed manifest")
    if sum(len(d) + 1 for d in baseline.documents) != 11498 + sum(
            len(bytes.fromhex(d["text_hex"])) + 1 for d in documents.values()):
        raise ValueError("reference did not retain every actual offered source")
    close_tick = time.perf_counter()
    cleanup = _cleanup(live)
    if cleanup:
        raise cleanup[0]
    close_seconds = time.perf_counter() - close_tick
    metrics = summarize(attempts, manifest)
    for metric in metrics.values():
        metric["exposed_changed_context_strata"] = metric.pop("fresh_strata")
    runtime_receipt["final_origins"] = check_origins(runtime_receipt)
    report = {
        "schema": "stable-language-action-credit-report-v1", "status": "completed",
        "pid": os.getpid(), "parent_pid": os.getppid(), "routes": list(ROUTES),
        "credit_rule": args.credit_rule, "metrics": metrics, "initial_hashes": initial_hashes,
        "probe_checkpoints": probe_receipts, "runtime": runtime_receipt,
        "final_hashes": {r: dict(zip(("model", "memory"), hashes(r), strict=True)) for r in ROUTES},
        "counts": {k: j.rows for k, j in journals.items()}, "actual_executions": actual_children,
        "journals": {j.path.name: j.metrics() for j in live["journals"]},
        "initial_memory_slots": 1982, "final_memory_slots": {r: s.used for r, s in states.items()},
        "final_reader_experience": {r: m.reader._active_experience for r, m in models.items()},
        "final_model_experience": {r: m._experience for r, m in models.items()},
        "held_reader_sha256": prior_hashes, "disabled_slow_sha256": disabled_slow,
        "final_controller": {r: controller_hashes(m.reader, parent_work) for r, m in models.items()},
        "checkpoint_store": {"directory": "checkpoints", "config_sha256": store.config_sha256,
                             "last_safe": live["last_safe"], "head_sha256": sha(store.directory / "HEAD.json"),
                             "phase_seconds": checkpoint_timing, "work": dict(store.work),
                             "close_seconds": close_seconds,
                             "final_checkpoint_scope": "final safe HEAD is authoritative; no duplicate tensor export"},
        "initialization_seconds": initialization["elapsed_seconds"],
        "manifest_validation_seconds": manifest_validation_seconds,
        "manifest_validation_scope": "exact saved V8 JSON transformation; no model or old journal replay",
        "parent_custody_work": dict(parent_work),
        "source_capture_seconds": source_capture_seconds, "disk_accounting": live["disk"],
        "budget": {"complete_seconds": COMPLETE_SECONDS,
                   "runner_seconds": RUNNER_SECONDS, "audit_seconds": AUDIT_SECONDS,
                   "reserve_seconds": CHECKPOINT_RESERVE_SECONDS, "peak_mib": PEAK_MIB,
                   "new_artifact_bytes": ARTIFACT_BYTES, "initial_free_bytes": INITIAL_FREE_BYTES,
                   "pending_write_reserve_bytes": PENDING_WRITE_RESERVE_BYTES,
                   "pending_write_scope": "inside artifact cap; bounded parent/retained writes, not a filesystem quota"},
        "preinitialization_free_bytes": preinitialization_free,
        "clock": "time.perf_counter", "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "complete initialization, sources, prediction/trace, action, resolution, journals and persistence; attempt includes its recorded begin interval",
        "learning_scope": "selected old or causal credit with retained V8 actor; disabled slow state fixed; actual RNG advances",
        "functional_prior_parity": "reported by mandatory CP0 paired callback", "declared_exposed_source_passes": 1, "automatic_replays": 0,
    }
    save(output / "report.json", report)
    if sources != {name: sha(ROOT / name) for name in sources}:
        raise ValueError("qualified source bytes changed during the new session")
    inventory_tick = time.perf_counter()
    inventory = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    inventory_seconds = time.perf_counter() - inventory_tick
    artifact_size = sum(p.stat().st_size for p in args.comparison_root.rglob("*") if p.is_file())
    if (time.perf_counter() >= deadline or artifact_size > ARTIFACT_BYTES
            or shutil.disk_usage(output).free < FREE_RESERVE_BYTES):
        raise TimeoutError("complete final inventory exceeded its elapsed/artifact bound")
    save(output / "completion.json", {"schema": "stable-language-action-credit-completion-v1",
                                      "status": "completed", "files": inventory,
                                      "final_inventory_seconds": inventory_seconds,
                                      "artifact_bytes_before_completion": artifact_size,
                                      "elapsed_seconds": time.perf_counter() - started})


def _closed(journal):
    return journal.closed if hasattr(journal, "closed") else journal.handle.closed


def _cleanup(live, *, close_store=True):
    """Attempt every owned release; callers keep the primary failure authoritative."""
    failures = []
    for index, journal in enumerate(live.get("journals", [])):
        try:
            if not _closed(journal):
                journal.close()
        except BaseException as error:
            failures.append(error)
            live.setdefault("cleanup_errors", []).append({
                "owner": "journal", "index": index,
                "type": type(error).__name__, "error": str(error)})
    if close_store and live.get("store") is not None:
        try:
            live["store"].close()
        except BaseException as error:
            failures.append(error)
            live.setdefault("cleanup_errors", []).append({
                "owner": "checkpoint_store", "type": type(error).__name__, "error": str(error)})
    return failures


def run(args, *, initializer=None, checkpoint_callback=None):
    """One explicitly released triad; preserve failure and never resume/replay."""
    if DRAFT or MANIFEST_SHA is None:
        raise RuntimeError("DRAFT: reviewed source and manifest freeze required")
    started, live = time.perf_counter(), {}
    args.comparison_root = Path(args.comparison_root).resolve()
    output = Path(args.output).resolve()
    if (not output.is_relative_to(args.comparison_root) or output == args.comparison_root
            or args.comparison_root == ROOT or not args.comparison_root.is_relative_to(ROOT)):
        raise ValueError("worker output must be inside its distinct comparison evidence root")
    if (not math.isfinite(args.deadline) or not math.isfinite(args.comparison_started)
            or args.comparison_started > started
            or not math.isclose(args.deadline - args.comparison_started, RUNNER_SECONDS,
                                rel_tol=0, abs_tol=1e-6)
            or started + CHECKPOINT_RESERVE_SECONDS >= args.deadline):
        raise ValueError("worker needs the shared complete comparison clock and remaining reserve")
    if args.credit_rule not in {"old", "causal"}:
        raise ValueError("credit rule must be old or causal")
    if (args.credit_rule == "causal") != (args.old_cp0_reference is not None):
        raise ValueError("only causal worker requires the authenticated saved old CP0 reference")
    if args.credit_rule == "old" and shutil.disk_usage(args.comparison_root).free < INITIAL_FREE_BYTES:
        raise RuntimeError("comparison requires54 GiB free before its old worker starts")
    if shutil.disk_usage(args.comparison_root).free < FREE_RESERVE_BYTES:
        raise RuntimeError("comparison free-disk reserve is exhausted")
    output.mkdir(parents=True, exist_ok=False)
    try:
        # This is the first import capable of selecting neuronal source. All
        # experiment helpers and initializers import only after its origin seal.
        from experiments.r4_integrated.persistent_language_action_credit_runtime import select

        runtime_receipt = select(args.credit_rule, args.old_snapshot)
        if initializer is None:
            from experiments.r4_integrated.persistent_language_action_credit_initialize import (
                initialize_credit_v2,
                initialize_v8,
            )
            from humanbrain.recurrent_content_learner import RecurrentContentSpliceLearner

            initialize = initialize_v8 if args.credit_rule == "old" else initialize_credit_v2

            def initializer(prior, bank):
                return initialize(prior, bank, actor_class=RecurrentContentSpliceLearner)

        if checkpoint_callback is None:
            from experiments.r4_integrated.persistent_language_action_credit_probes import (
                checkpoint_callback,
            )
        _run(args, output, started, live, initializer, checkpoint_callback, runtime_receipt)
        cleanup = _cleanup(live)
        if cleanup:
            raise cleanup[0]
    except BaseException as error:
        _cleanup(live)
        failure = {
            "schema": "stable-language-action-credit-failure-v1", "type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.perf_counter() - started,
            "credit_rule": args.credit_rule, "last_safe_returned": live.get("last_safe"),
            "parent_custody_work": live.get("parent_custody_work"),
            "disk_accounting": live.get("disk"), "resumed": False,
            "dirty_intents_preserved": True, "cleanup_errors": live.get("cleanup_errors", []),
        }
        try:
            store, active = live.get("store"), live.get("checkpoint_active")
            if active is not None:
                active = {"phase": active["phase"],
                          "elapsed_seconds": time.perf_counter() - active["started"],
                          "work_delta": None if store is None else
                          {k: v - active["work_before"].get(k, 0) for k, v in store.work.items()}}
            failure.update(interrupted_checkpoint_phase=active,
                           checkpoint_work_at_failure=None if store is None else dict(store.work),
                           journals={j.path.name: j.metrics() for j in live.get("journals", [])})
            head_path = output / "checkpoints/HEAD.json"
            raw = head_path.read_bytes() if head_path.exists() else None
            failure["head_pointer"] = None if raw is None else {
                "record": json.loads(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        except BaseException as collection_error:
            failure["failure_collection_error"] = {
                "type": type(collection_error).__name__, "error": str(collection_error)}
        try:
            with (output / "failure.json").open("xb") as handle:
                handle.write((json.dumps(failure, sort_keys=True, indent=2) + "\n").encode())
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException as receipt_error:
            error.add_note("Failure receipt publication also failed: " + repr(receipt_error))
        if live.get("cleanup_errors"):
            error.add_note("Owned cleanup failures: " + repr(live["cleanup_errors"]))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--comparison-started", required=True, type=float)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--credit-rule", required=True, choices=("old", "causal"))
    parser.add_argument("--old-snapshot", required=True, type=Path)
    parser.add_argument("--old-cp0-path", type=Path)
    parser.add_argument("--old-cp0-sha")
    parser.add_argument("--base-manifest", type=Path,
                        default=ROOT / ".pytest-tmp/persistent-language-action-v8-manifest.json")
    parser.add_argument("--prior", type=Path,
                        default=ROOT / ".pytest-tmp/persistent-language-action-v8-corrected-run/checkpoints")
    parser.add_argument("--bank", type=Path,
                        default=ROOT / "benchmarks/persistent_language_action_v8.result.json")
    args = parser.parse_args()
    if bool(args.old_cp0_path) != bool(args.old_cp0_sha):
        parser.error("old CP0 path and SHA must be supplied together")
    args.old_cp0_reference = None if args.old_cp0_path is None else {
        "path": str(args.old_cp0_path.resolve()), "sha256": args.old_cp0_sha}
    run(args)


if __name__ == "__main__":
    main()
