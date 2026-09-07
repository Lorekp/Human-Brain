"""One explicitly released recurrent continuation with durable safe boundaries.

Import performs no model construction, corpus build or checkpoint decoding.
The frozen manifest supplies every source and encounter; no retry is selected
by private correctness and no historical missing attempt is replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.r4_integrated.persistent_language_action_v7 import (  # noqa: E402
    ROUTES,
    SessionJournal,
    _plain,
    encode,
    save,
    sha,
)

DRAFT = False
MANIFEST_SHA = "66b0bd96be3b9e97c4eb5f9812c2b5977f294fc5900d55109445c0a303e522d0"
RUNNER_SECONDS = 3600.0
AUDIT_SECONDS = 120.0
CHECKPOINT_RESERVE_SECONDS = 120.0
PEAK_MIB = 4096
ARTIFACT_BYTES = 26 * 1024**3
INITIAL_FREE_BYTES = 34 * 1024**3
SAVED_REVIEW_SOURCES = {
    "experiments/r4_integrated/persistent_language_action_v8_review.py",
    "experiments/r4_integrated/bank_persistent_language_action_v8.py",
}


def reader_summary(prediction):
    """Actual scan/free draws, visibility and RNG; never a second generation."""
    return {
        "schema": "recurrent-content-output-prediction-v1",
        "attempt_id": prediction.attempt_id,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "trace_sha256": prediction.trace_sha256,
        "observation_sha256": prediction.observation.sha256,
        "max_bytes": prediction.max_bytes,
        "answer_hex": prediction.generation.data.hex(),
        "stop_reason": prediction.generation.stop_reason,
        "decisions": len(prediction.steps),
        "scan_steps": _plain(prediction.scan_steps),
        "controller": _plain(prediction.controller),
        "steps": [{
            "output_step": step.output_step, "prefix_hex": step.prefix.hex(),
            "token": step.observed_token, "selected_index": step.selected_index,
            "address_scores": step.address_scores.tolist(),
            "assembly_index": step.assembly_index,
            "assembly_scores": step.assembly_scores.tolist(),
            "routing": _plain(step.routing), "binding": _plain(step.binding),
            "controller": _plain(step.controller), "prior_logits": step.logits.tolist(),
            "prior_nll": step.nll,
        } for step in prediction.steps],
        "work": dict(prediction.work),
    }


def summarize(attempts, manifest):
    """Complete-stream metrics only; a partial saved audit must report missing outcomes."""
    result = {}
    for route in ROUTES:
        rows = [r for r in attempts if r["route"] == route]
        first = {r["episode_id"]: r for r in rows if r["attempt_index"] == 1}
        phases = {}
        for phase in ("acquisition", "transfer", "retention", "recovery"):
            selected = [r for r in rows if r["phase"] == phase]
            phases[phase] = {kind: {
                "episodes": sum(r["attempt_index"] == 1 for r in selected if r["kind"] == kind),
                "attempts": sum(r["kind"] == kind for r in selected),
                "first_correct": sum(r["correct"] for r in selected
                                     if r["kind"] == kind and r["attempt_index"] == 1),
                "unsupported": sum(r["status"] == "unsupported" for r in selected
                                   if r["kind"] == kind),
            } for kind in ("reading", "edit")}
        encounters, fresh, retained, repeats = {}, {}, [], []
        for name in ("first", "changed_form", "exact_delayed"):
            members = [e for e in manifest["episodes"]
                       if e["phase"] == "acquisition" and e.get("encounter") == name]
            encounters[name] = {"episodes": len(members),
                                "first_correct": sum(first[e["episode_id"]]["correct"]
                                                     for e in members),
                                "requirement_changed": sum(bool(e.get("requirement_changed"))
                                                           for e in members)}
        for name in ("instruction_construction", "structural_combination"):
            members = [e for e in manifest["episodes"]
                       if e["phase"] == "transfer" and e.get("fresh_stratum") == name]
            fresh[name] = {"episodes": len(members),
                           "first_correct": sum(first[e["episode_id"]]["correct"] for e in members),
                           "unsupported": sum(first[e["episode_id"]]["status"] == "unsupported"
                                              for e in members)}
            fresh[name]["dependency"] = {
                label: {"episodes": sum(bool(e.get("predecessor_episode_id")) == dependent
                                        for e in members),
                        "first_correct": sum(first[e["episode_id"]]["correct"] for e in members
                                             if bool(e.get("predecessor_episode_id")) == dependent),
                        "unsupported": sum(first[e["episode_id"]]["status"] == "unsupported"
                                           for e in members
                                           if bool(e.get("predecessor_episode_id")) == dependent)}
                for label, dependent in (("independent", False), ("actual_project_dependent", True))}
        for episode in manifest["episodes"]:
            after = first[episode["episode_id"]]
            if episode.get("encounter") in ("changed_form", "exact_delayed"):
                before = first[episode["same_session_origin_id"]]
                age = rows.index(after) - rows.index(before) - 1
                same = before["observation_sha256"] == after["observation_sha256"]
                if age != episode["intervening_attempts"] or (
                        episode["encounter"] == "exact_delayed" and not same):
                    raise ValueError("declared exact encounter input or age changed")
                repeats.append({"episode_id": episode["episode_id"],
                                "origin": before["episode_id"], "encounter": episode["encounter"],
                                "initial_correct": before["correct"], "correct": after["correct"],
                                "initial_memory_used": before["memory_used"],
                                "memory_used": after["memory_used"],
                                "intervening_attempts": age, "same_observation": same})
            if episode["phase"] == "retention":
                old_id = episode["prior_episode_id"]
                parent_route = "simple" if route == "simple" else "online"
                old = manifest["prior_regression_bindings"]["routes"][parent_route][old_id]
                retained.append({
                    "episode_id": episode["episode_id"], "prior_episode_id": old_id,
                    "prior_route": parent_route, "prior_attempt_sha256": old["attempt_sha256"],
                    "initial_correct": old["correct"], "correct": after["correct"],
                    "memory_used": after["memory_used"],
                    "memory_scope": "neuronal admitted facts; simple counts all retained source symbols",
                    "same_observation": hashlib.sha256(encode(old["observation"])).hexdigest()
                    == after["observation_sha256"],
                    "intervening_current_attempts": rows.index(after),
                })
        recovery = [e for e in manifest["episodes"] if e["phase"] == "recovery"]
        chains = [[r for r in rows if r["episode_id"] == e["episode_id"]] for e in recovery]
        result[route] = {
            "phases": phases, "acquisition_encounters": encounters, "fresh_strata": fresh,
            "same_session_encounters": repeats, "retention": retained,
            "same_session_exact_retained": sum(r["initial_correct"] and r["correct"]
                                               for r in repeats if r["encounter"] == "exact_delayed"),
            "project_chains": [{k: row[k] for k in (
                "episode_id", "project_id", "predecessor_episode_id", "workspace_lifecycle",
                "correct", "status")}
                for row in rows if row["project_id"] is not None and row["attempt_index"] == 1],
            "retained": sum(r["initial_correct"] and r["correct"] for r in retained),
            "forgotten": sum(r["initial_correct"] and not r["correct"] for r in retained),
            "improved": sum(not r["initial_correct"] and r["correct"] for r in retained),
            "recovery": {
                "episodes": len(chains), "first_correct": sum(c[0]["correct"] for c in chains),
                "later_correct": sum(not c[0]["correct"] and any(r["correct"] for r in c[1:])
                                     for c in chains),
                "final_correct": sum(c[-1]["correct"] for c in chains),
                "final_recovered": sum(not c[0]["correct"] and c[-1]["correct"] for c in chains),
                "feedback_consuming_attempts": sum(r["feedback_consumed"] for c in chains for r in c),
                "identical_consecutive_proposals": sum(
                    a["proposal_signature"] is not None
                    and a["proposal_signature"] == b["proposal_signature"]
                    for c in chains for a, b in zip(c, c[1:])),
                "policy": "fixed manifest attempts; private grade never controls retry",
            },
            "public_checks_passed": sum(r.get("public_checks_passed", False) for r in rows),
            "execution_outcomes": {name: sum(r["execution_outcome"] == name for r in rows)
                                   for name in (
                                       "correct_stop", "incorrect_stop",
                                       "accepted_correct_splice_with_change",
                                       "accepted_correct_splice_without_change",
                                       "incorrect_accepted_splice", "rejected_proposal", "no_proposal")},
            "scope": "private correctness measures remain separate from public controller reward",
        }
    return result


def _run(args, output, started, live):
    deadline = started + RUNNER_SECONDS
    disk_seconds, disk_checks = 0.0, 0

    def admit_work():
        nonlocal disk_seconds, disk_checks
        if time.perf_counter() + CHECKPOINT_RESERVE_SECONDS >= deadline:
            raise TimeoutError("whole-session reserve reached before admitting new work")
        tick = time.perf_counter()
        used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        free = shutil.disk_usage(output).free
        disk_seconds += time.perf_counter() - tick
        disk_checks += 1
        live["disk"] = {"artifact_bytes": used, "free_bytes": free,
                        "checks": disk_checks, "seconds": disk_seconds}
        if used > ARTIFACT_BYTES or free < INITIAL_FREE_BYTES - ARTIFACT_BYTES:
            raise RuntimeError("complete artifact/free-disk budget exhausted")

    admit_work()
    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        grade_execution,
        proposal_from_record,
        workspace_from_records,
    )
    from experiments.r4_integrated.persistent_language_action_v8_checkpoint import (
        PersistentLanguageActionV8Checkpoint,
        controller_hashes,
    )
    from experiments.r4_integrated.persistent_language_action_v8_execution import (
        execute_proposal,
        feedback_from_receipt,
        qualified_launch,
    )
    from experiments.r4_integrated.persistent_language_action_v8_initialize import initialize
    from experiments.r4_integrated.persistent_language_action_v8_workload import (
        check_saved_manifest,
    )
    from experiments.r4_integrated.persistent_session_checkpoint import COUNTERS

    if MANIFEST_SHA is None or args.manifest_sha != MANIFEST_SHA or sha(args.manifest) != MANIFEST_SHA:
        raise ValueError("V8 requires the one root-frozen authored manifest")
    manifest_tick = time.perf_counter()
    check_saved_manifest(args.manifest, MANIFEST_SHA)
    manifest_validation_seconds = time.perf_counter() - manifest_tick
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["schema"] != "persistent-language-action-stream-v8"
            or manifest["continuation"]["initial_memory_slots"] != 1614):
        raise ValueError("V8 continuation manifest differs")
    documents = {row["document_id"]: row for row in manifest["documents"]}
    if (any(len(e["append_document_ids"]) > 1 for e in manifest["episodes"])
            or [key for e in manifest["episodes"] for key in e["append_document_ids"]]
            != list(documents)):
        raise ValueError("each whole source must occur once in fixed public order")
    final_memory = 1614 + sum(len(bytes.fromhex(d["text_hex"])) + 1
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
               for p in sorted(directory.glob("*.py"))
               if p.relative_to(ROOT).as_posix() not in SAVED_REVIEW_SOURCES}
    save(output / "sources.json", sources)
    source_capture_seconds = time.perf_counter() - capture_tick
    tick = time.perf_counter()
    store = PersistentLanguageActionV8Checkpoint.create(
        output / "checkpoints", source_root=ROOT, evidence_root=output,
        manifest_path=output / "manifest.json", manifest_sha256=MANIFEST_SHA,
        source_revision=args.source_revision, source_sha256=sources, session_id=args.session_id)
    live["store"] = store
    create_seconds = time.perf_counter() - tick
    admit_work()  # Initialization intent is durable before any model/decoder call.
    preinitialization_free = shutil.disk_usage(output).free
    if preinitialization_free < INITIAL_FREE_BYTES:
        raise RuntimeError("at least34 GiB must remain after manifest/source/intent publication")
    import torch

    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.sequence_assembly_observation import RawObservation, ToolFeedback

    parent_work = {}
    live["parent_custody_work"] = parent_work
    torch.set_num_threads(1)
    initialized = initialize(args.prior, args.bank)
    models, states, baseline = (initialized[name] for name in ("models", "states", "baseline"))
    initialization = initialized["receipt"]
    if set(models) != {"online", "learning_disabled"}:
        raise ValueError("initializer must produce two independent matched online descendants")
    save(output / "initialization.json", {
        **initialization, "prior_host": initialized["prior_host"],
        "prior_frontiers": initialized["prior_frontiers"],
        "baseline_record": initialized["baseline_record"],
    })
    launch = qualified_launch(ROOT)
    save(output / "launch-record.json", launch.to_record())
    journals = {name: SessionJournal(output / (name + ".jsonl"))
                for name in ("attempts", "teachers", "ingests")}
    checkpoint_journal = SessionJournal(output / "checkpoints.jsonl")
    live["journals"] = [*journals.values(), checkpoint_journal]
    projects = {route: {} for route in ROUTES}
    host = {
        "cursor": {"episode_index": 0, "route_index": 0, "attempt_index": 1, "append_done": False},
        "projects": projects, "current": None,
        "counters": {r: dict.fromkeys(COUNTERS, 0) for r in ROUTES},
        "elapsed_seconds": time.perf_counter() - started,
        "budget_seconds": RUNNER_SECONDS + AUDIT_SECONDS, "budget_mib": PEAK_MIB,
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

    def commit():
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
                    if admitted:
                        grown, feedback = model.ingest_with_receipt(
                            [raw], old, learning_enabled=route == "online", writes_enabled=True)
                    else:
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

    for journal in live["journals"]:
        journal.close()
    for route in models:
        controls(route)
        if states[route].used != final_memory:
            raise ValueError("final admitted factual count differs")
    if ({k: j.rows for k, j in journals.items()}
            != {"attempts": expected_attempts, "teachers": expected_teachers, "ingests": 3 * len(documents)}
            or actual_children != expected_children
            or store.work["checkpoint_head_publications"] != 1 + expected_attempts + 3 * len(documents)):
        raise ValueError("complete stream counts disagree with its fixed manifest")
    if sum(len(d) + 1 for d in baseline.documents) != 1614 + sum(
            len(bytes.fromhex(d["text_hex"])) + 1 for d in documents.values()):
        raise ValueError("reference did not retain every actual offered source")
    close_tick = time.perf_counter()
    store.close()
    close_seconds = time.perf_counter() - close_tick
    report = {
        "schema": "persistent-language-action-report-v8", "status": "completed",
        "pid": os.getpid(), "parent_pid": os.getppid(), "routes": list(ROUTES),
        "metrics": summarize(attempts, manifest), "initial_hashes": initial_hashes,
        "final_hashes": {r: dict(zip(("model", "memory"), hashes(r), strict=True)) for r in ROUTES},
        "counts": {k: j.rows for k, j in journals.items()}, "actual_executions": actual_children,
        "journals": {j.path.name: j.metrics() for j in live["journals"]},
        "initial_memory_slots": 1614, "final_memory_slots": {r: s.used for r, s in states.items()},
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
        "manifest_validation_scope": "one saved validation including authenticated historical JSON and one old attempt-journal stream; zero builder/model calls",
        "parent_custody_work": dict(parent_work),
        "source_capture_seconds": source_capture_seconds, "disk_accounting": live["disk"],
        "budget": {"runner_seconds": RUNNER_SECONDS, "audit_seconds": AUDIT_SECONDS,
                   "reserve_seconds": CHECKPOINT_RESERVE_SECONDS, "peak_mib": PEAK_MIB,
                   "new_artifact_bytes": ARTIFACT_BYTES, "initial_free_bytes": INITIAL_FREE_BYTES},
        "preinitialization_free_bytes": preinitialization_free,
        "clock": "time.perf_counter", "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "complete initialization, sources, prediction/trace, action, resolution, journals and persistence; attempt includes its recorded begin interval",
        "learning_scope": "new controller with held old reader; local motor corrections; disabled slow state fixed but actual RNG advances",
        "functional_prior_parity": False, "automatic_replays": 0,
    }
    save(output / "report.json", report)
    if sources != {name: sha(ROOT / name) for name in sources}:
        raise ValueError("qualified source bytes changed during the new session")
    inventory_tick = time.perf_counter()
    inventory = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    inventory_seconds = time.perf_counter() - inventory_tick
    artifact_size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if time.perf_counter() >= deadline or artifact_size > ARTIFACT_BYTES:
        raise TimeoutError("complete final inventory exceeded its elapsed/artifact bound")
    save(output / "completion.json", {"schema": "persistent-language-action-completion-v8",
                                      "status": "completed", "files": inventory,
                                      "final_inventory_seconds": inventory_seconds,
                                      "artifact_bytes_before_completion": artifact_size,
                                      "elapsed_seconds": time.perf_counter() - started})


def run(args):
    """One fresh launch; on failure retain evidence, close leases, never replay."""
    if DRAFT or MANIFEST_SHA is None:
        raise RuntimeError("DRAFT: root source/manifest release is required")
    started, live = time.perf_counter(), {}
    output = args.output.resolve()
    if shutil.disk_usage(output.parent).free < INITIAL_FREE_BYTES:
        raise RuntimeError("new complete session requires at least34 GiB free before initialization")
    output.mkdir(parents=True, exist_ok=False)
    try:
        _run(args, output, started, live)
    except BaseException as error:
        for journal in live.get("journals", []):
            if not journal.handle.closed:
                journal.close()
        store, active = live.get("store"), live.get("checkpoint_active")
        if active is not None:
            active = {"phase": active["phase"], "elapsed_seconds": time.perf_counter() - active["started"],
                      "work_delta": None if store is None else
                      {k: v - active["work_before"].get(k, 0) for k, v in store.work.items()}}
        head_path = output / "checkpoints/HEAD.json"
        raw = head_path.read_bytes() if head_path.exists() else None
        save(output / "failure.json", {
            "schema": "persistent-language-action-failure-v8", "type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.perf_counter() - started,
            "last_safe_returned": live.get("last_safe"), "interrupted_checkpoint_phase": active,
            "head_pointer": None if raw is None else {"record": json.loads(raw),
                                                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes_read": len(raw)},
            "checkpoint_work_at_failure": None if store is None else dict(store.work),
            "parent_custody_work": live.get("parent_custody_work"),
            "journals": {j.path.name: j.metrics() for j in live.get("journals", [])},
            "disk_accounting": live.get("disk"), "resumed": False,
            "failure_inspect_calls": 0, "dirty_intents_preserved": True,
            "scope": "partial evidence and last returned safe pointer; no complete audit claim",
        })
        raise
    finally:
        for journal in live.get("journals", []):
            if not journal.handle.closed:
                journal.close()
        if live.get("store") is not None:
            live["store"].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--prior", type=Path, default=ROOT / ".pytest-tmp/persistent-language-action-v7-run/checkpoints")
    parser.add_argument("--bank", type=Path, default=ROOT / "benchmarks/persistent_language_action_v7.result.json")
    args = parser.parse_args()
    if DRAFT or MANIFEST_SHA is None:
        raise SystemExit("DRAFT: source review, authored manifest and root release required")
    run(args)


if __name__ == "__main__":
    main()
