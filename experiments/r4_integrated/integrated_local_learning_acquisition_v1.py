"""Draft compact512-correction comparison worker; fixed existing calls and custody."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.r4_integrated import integrated_local_learning_runtime_v1 as runtime

ROOT = Path(__file__).resolve().parents[2]
DRAFT = True
BANK_SHA = runtime.BANK_SHA
BASE_SHA = "528f6599d0ff5af487a418a7a0fa06b340ba0d026328df4bab534af671bd64cf"
ROUTES = ("online", "learning_disabled", "simple")
CHECKPOINTS = (0, 64, 128, 256, 512, 768, 1024)
ARTIFACT_CAP = 768 * 1024**2
PENDING_BYTES = 256 * 1024**2
FREE_RESERVE = 6 * 1024**3
SECONDS = 43200


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def qualified_revision(expected):
    """Bind the launch label to actual HEAD; source bytes are separately sealed."""
    if (
        not isinstance(expected, str)
        or len(expected) != 40
        or any(c not in "0123456789abcdef" for c in expected)
    ):
        raise ValueError("source revision must be an exact lowercase Git commit SHA")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if actual != expected:
        raise ValueError("requested source revision differs from actual launch HEAD")
    return actual


def save(path, value, *, artifact_root=None):
    raw = encode(value)
    if (
        len(raw) > 64 * 1024**2
        or shutil.disk_usage(Path(path).parent).free - len(raw) < FREE_RESERVE
    ):
        raise RuntimeError("receipt publication exceeds size/free-space bound")
    if artifact_root is not None:
        used = sum(p.stat().st_size for p in Path(artifact_root).rglob("*") if p.is_file())
        if used + len(raw) > ARTIFACT_CAP:
            raise RuntimeError("receipt publication exceeds complete artifact cap")
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def compact_prediction(prediction):
    """Complete actual output and counted acting trace; no omitted-vector claims."""
    scans = prediction.scan_steps
    free = [step.controller for step in prediction.steps]
    return {
        "schema": "integrated-local-learning-compact-prediction-v1",
        "attempt_id": prediction.attempt_id,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "trace_sha256": prediction.trace_sha256,
        "observation_sha256": prediction.observation.sha256,
        "answer_hex": prediction.generation.data.hex(),
        "stop_reason": prediction.generation.stop_reason,
        "decisions": len(free),
        "scan_offers": len(scans),
        "controller_draws": sum(len(row["draws"]) for row in (*scans, *free)),
        "rng_before_sha256": scans[0]["draws"][0]["rng_before_sha256"],
        "rng_after_sha256": free[-1]["draws"][-1]["rng_after_sha256"],
        "work": dict(prediction.work),
        "scope": "complete free bytes/EOS; trace hash binds omitted vectors without retaining them",
    }


def make_manifest(base):
    if sha(base) != BASE_SHA:
        raise ValueError("repetition requires exact previously exposed task bytes")
    prior = json.loads(Path(base).read_bytes())
    by_id = prior["tasks"]
    names = ["A00-R"] * 64 + ["A00-E"] * 64 + ["A00-R", "A00-E"] * 448
    return {
        "schema": "integrated-local-learning-acquisition-v1",
        "base_manifest_sha256": BASE_SHA,
        "parent_bank_sha256": BANK_SHA,
        "tasks": {k: by_id[k] for k in ("A00-R", "A00-E", "R02-R", "R12-R")},
        "prior_regression_bindings": prior["prior_regression_bindings"],
        "schedule": [{"index": i, "episode_id": k} for i, k in enumerate(names)],
        "routes": list(ROUTES),
        "checkpoints": list(CHECKPOINTS),
        "checkpoint_cases": ["A00-R", "A00-E", "R02-R", "R12-R"],
        "live_teacher_per_neural": {
            "occasions": 1024,
            "raw_bytes": 16896,
            "eos": 1024,
            "symbols": 17920,
            "motor_labels": 1536,
        },
        "diagnostic_teacher_per_neural": {
            "occasions": 14,
            "symbols": 245,
            "motor_labels": 21,
            "slow_publications": 0,
        },
        "attempts": 3156,
        "executions": 1557,
        "new_source_offers": 0,
        "architecture": "existing recurrent actor; predecessor/current source-isolated; eta0.01",
        "durability": "authenticated parent and final block; scalar panels are nondurable",
        "budget": {
            "seconds": SECONDS,
            "artifact_bytes": ARTIFACT_CAP,
            "pending_bytes": PENDING_BYTES,
            "free_reserve_bytes": FREE_RESERVE,
        },
    }


def run(args):
    if DRAFT:
        raise RuntimeError("DRAFT: source/protocol review required before execution")
    started = time.perf_counter()
    deadline = started + SECONDS - 120
    output = Path(args.output).resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError("output must be a new local evidence directory")
    if shutil.disk_usage(ROOT).free < FREE_RESERVE + ARTIFACT_CAP:
        raise RuntimeError("practice needs its768MiB allowance above6GiB free")
    output.mkdir(parents=True, exist_ok=False)

    def publish(name, value):
        save(output / name, value, artifact_root=output)

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_bytes())
    if manifest != make_manifest(args.base):
        raise ValueError("manifest differs from the bounded fixed repetition schedule")
    publish("manifest.json", manifest)
    live = {"completed_attempts": 0, "completed_teachers": 0, "executions": 0}
    phases = {}
    pending = {}
    journals = []
    store = None

    def admit(reserve=True):
        admission_start = time.perf_counter()
        used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        free = shutil.disk_usage(output).free
        phases["admission_inventory_seconds"] = (
            phases.get("admission_inventory_seconds", 0) + time.perf_counter() - admission_start
        )
        live["disk"] = {"artifact_bytes": used, "free_bytes": free}
        live["process_memory"] = process_memory()
        if live["process_memory"]["working_set_mib"] > 4096:
            raise RuntimeError("practice working set exceeds4096MiB")
        if used + (PENDING_BYTES if reserve else 0) > ARTIFACT_CAP or free < FREE_RESERVE + (
            PENDING_BYTES if reserve else 0
        ):
            raise RuntimeError("bounded acquisition artifact/free-space limit")
        if time.perf_counter() >= (deadline if reserve else started + SECONDS - 5):
            raise TimeoutError("acquisition complete-run reserve reached")

    try:
        source_revision = qualified_revision(args.source_revision)
        selection = runtime.select(args.rule, args.predecessor_snapshot)
        import torch

        from experiments.r4_integrated.integrated_local_learning_checkpoint_v1 import (
            IntegratedLearningCheckpoint,
        )
        from experiments.r4_integrated.integrated_local_learning_diagnostics_v1 import (
            free_teacher_alignment,
            panel_teacher,
            practice_teacher,
            score_anchors,
        )
        from experiments.r4_integrated.integrated_local_learning_initialize_v1 import (
            clone_for_probe as clone_current,
        )
        from experiments.r4_integrated.integrated_local_learning_initialize_v1 import (
            initialize_current,
            initialize_predecessor,
        )
        from experiments.r4_integrated.integrated_local_learning_preservation_v1 import (
            _check_sources,
            _sources,
        )
        from experiments.r4_integrated.neuronal_splice_io_v1 import process_memory
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
            clone_for_probe as clone_predecessor,
        )
        from experiments.r4_integrated.repeated_teaching_diagnostics_v1 import (
            capture_literal,
            free_summary,
        )
        from humanbrain.neuronal_splice_learner import _digest
        from humanbrain.recurrent_content_learner import RecurrentContentSpliceLearner
        from humanbrain.sequence_assembly_observation import RawObservation, ToolFeedback

        torch.set_num_threads(1)
        admit()
        initializer = initialize_predecessor if args.rule == "predecessor" else initialize_current
        clone_for_probe = clone_predecessor if args.rule == "predecessor" else clone_current
        phase_start = time.perf_counter()
        initialized = initializer(
            runtime.PARENT / "checkpoints",
            runtime.BANK,
            actor_class=RecurrentContentSpliceLearner,
            source_receipt=selection,
            predecessor_snapshot=args.predecessor_snapshot,
        )
        phases["initialization_seconds"] = time.perf_counter() - phase_start
        sources = _sources()
        runtime.check_origins(selection)
        publish(
            "sources.json",
            {"source_sha256": sources, "selection": selection, "source_revision": source_revision},
        )
        models, states, baseline = (initialized[k] for k in ("models", "states", "baseline"))
        publish("initialization.json", initialized["receipt"])
        store = IntegratedLearningCheckpoint.create(
            output / "checkpoints",
            evidence_root=output,
            manifest_path=output / "manifest.json",
            manifest_sha256=sha(output / "manifest.json"),
            source_sha256=sources,
            source_revision=source_revision,
            parent_receipt=initialized["receipt"],
        )
        owned = store.create_owned_journals()
        journals = list(owned.values())
        launch = qualified_launch(ROOT)
        publish("launch.json", launch.to_record())
        work = {}

        def hashes(actor, state):
            tick = time.perf_counter()
            result = {
                "model": _digest(actor.state_dict(), work),
                "memory": _digest(vars(state), work),
            }
            phases["explicit_neural_hash_seconds"] = (
                phases.get("explicit_neural_hash_seconds", 0) + time.perf_counter() - tick
            )
            return result

        def baseline_hashes(reference):
            return {
                "model": _digest(_plain(reference.demonstrations), work),
                "memory": _digest(_plain(reference.documents), work),
            }

        initial = {r: hashes(models[r], states[r]) for r in models}
        initial["simple"] = baseline_hashes(baseline)
        diagnostic_rng = models["online"].reader.controller.rng_state.detach().clone()
        disabled_slow = _digest(
            {
                k: v
                for k, v in models["learning_disabled"].state_dict().items()
                if k != "reader.controller.rng_state"
            },
            work,
        )
        results, probes, anchors = [], [], {}

        def attempt(route, episode_id, actor, state, reference, *, scope, index, teaching):
            admit()
            tick = time.perf_counter()
            episode = manifest["tasks"][episode_id]
            files, feedback = episode["files"], None
            if episode_id.startswith("R"):
                retained = manifest["prior_regression_bindings"]["routes"][
                    "simple" if route == "simple" else "online"
                ][episode["prior_episode_id"]]["observation"]
                files = retained["files"] or []
                if retained["feedback"] is not None:
                    f = retained["feedback"]
                    feedback = ToolFeedback(
                        **{k: v for k, v in f.items() if k not in ("stdout_hex", "stderr_hex")},
                        stdout=bytes.fromhex(f["stdout_hex"]),
                        stderr=bytes.fromhex(f["stderr_hex"]),
                    )
            workspace = workspace_from_records(files) if episode["kind"] == "edit" else None
            observation = RawObservation(bytes.fromhex(episode["request_hex"]), workspace, feedback)
            before = baseline_hashes(reference) if actor is None else hashes(actor, state)
            row = {
                "scope": scope,
                "step_index": index,
                "route": route,
                "episode_id": episode_id,
                "kind": episode["kind"],
                "observation": observation.to_record(),
                "model_before_sha256": before["model"],
                "memory_before_sha256": before["memory"],
                "execution": None,
                "proposal": None,
                "answer_hex": None,
            }
            pending["attempt"] = row
            prediction = proposal = envelope = None
            phase_start = time.perf_counter()
            if actor is None:
                row["prediction"] = reference.predict(observation, kind=episode["kind"])
                if workspace is None:
                    row["answer_hex"] = row["prediction"]["answer_hex"]
                else:
                    proposal = proposal_from_record(row["prediction"]["proposal"], workspace)
            elif workspace is None:
                prediction = actor.reader.predict_with_receipt(observation, state, 128)
                row["prediction"] = (
                    reader_summary(prediction)
                    if scope == "probe"
                    else compact_prediction(prediction)
                )
                row["answer_hex"] = prediction.generation.data.hex()
            else:
                prediction = actor.propose(observation, state)
                proposal = prediction.proposal
                row["prediction"] = {
                    "payload": (
                        reader_summary(prediction.payload)
                        if scope == "probe"
                        else compact_prediction(prediction.payload)
                    ),
                    "heads": {
                        k: {"logits": v.logits.tolist(), "selected_index": v.selected_index}
                        for k, v in prediction.choices.items()
                    },
                }
            row["phase_seconds"] = {
                "prediction_and_trace_projection": time.perf_counter() - phase_start
            }
            if workspace is not None:
                row["proposal"] = proposal.to_record()
                phase_start = time.perf_counter()
                after_files, execution = execute_proposal(
                    workspace,
                    proposal,
                    output / "executions" / scope / str(index) / route / episode_id,
                    launch=launch,
                    deadline=deadline,
                )
                row["phase_seconds"]["actual_execution"] = time.perf_counter() - phase_start
                live["executions"] += 1
                row.update(execution=execution, after_files=after_files)
                row["grade"] = grade_execution(execution, episode["grade"], episode["files"])
                row["correct"] = row["grade"]["task_success"]
                if prediction is not None:
                    envelope = {
                        "attempt_id": prediction.attempt_id,
                        "proposal_sha256": prediction.proposal_sha256,
                        "workspace_sha256": workspace.sha256,
                        "execution": execution,
                    }
            else:
                row["correct"] = row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                if prediction is not None:
                    row["correct"] = row["correct"] and prediction.generation.stop_reason == "eos"
            enabled = scope == "main" and route == "online"
            teacher_row = None
            if teaching:
                teacher = episode["teacher"]
                teacher_before = (
                    baseline_hashes(reference) if actor is None else hashes(actor, state)
                )
                target = bytes.fromhex(
                    teacher["target_hex"] if workspace is None else teacher["replacement_hex"]
                )
                teacher_row = {
                    "scope": scope,
                    "step_index": index,
                    "route": route,
                    "episode_id": episode_id,
                    "teacher": teacher,
                    "learning_enabled": enabled,
                    "reader_offers": 0,
                    "action_labels": 0,
                    "model_before_sha256": teacher_before["model"],
                }
                pending["teacher"] = teacher_row
                if actor is None:
                    phase_start = time.perf_counter()
                    teacher_row["publication"] = (
                        reference.observe_teacher(
                            observation, kind=episode["kind"], teacher=teacher
                        )
                        if scope == "main"
                        else {"published": False, "scope": "probe reference history held"}
                    )
                    row["phase_seconds"]["reference_correction"] = time.perf_counter() - phase_start
                else:
                    phase_start = time.perf_counter()
                    literal = capture_literal(actor.reader) if scope == "probe" else None
                    row["phase_seconds"]["diagnostic_snapshot"] = time.perf_counter() - phase_start
                    phase_start = time.perf_counter()
                    if workspace is None:
                        taught = actor.reader.teach(
                            observation,
                            target,
                            state,
                            0.01,
                            prediction=prediction,
                            learning_enabled=enabled,
                        )
                        rf = taught
                    else:
                        taught = actor.teach(
                            prediction,
                            observation,
                            state,
                            proposal_from_record(teacher, workspace),
                            envelope,
                            learning_enabled=enabled,
                        )
                        rf = taught.reader_feedback
                        teacher_row["action_labels"] = len(taught.action_losses)
                        teacher_row["actions"] = _plain(taught.action_updates)
                    row["phase_seconds"]["teacher_resolution"] = time.perf_counter() - phase_start
                    phase_start = time.perf_counter()
                    teacher_row.update(
                        reader_offers=rf.teacher_count,
                        controller=_plain(rf.controller),
                        work=_plain(taught.work),
                    )
                    teacher_row["free_teacher_alignment"] = free_teacher_alignment(
                        prediction, rf, target
                    )
                    if scope == "probe":
                        teacher_row["teacher_steps"] = _plain(rf.teacher_steps)
                        teacher_row["diagnostic"] = panel_teacher(
                            rf, target, literal_snapshot=literal, learning_enabled=False
                        )
                        key = route, episode_id
                        if index == 0:
                            anchors[key] = copy.deepcopy(teacher_row["diagnostic"]["anchors"])
                        teacher_row["fixed_anchor_scores"] = score_anchors(literal, anchors[key])
                    else:
                        teacher_row["diagnostic"] = practice_teacher(
                            rf, target, learning_enabled=enabled
                        )
                    row["phase_seconds"]["teacher_observer_and_projection"] = (
                        time.perf_counter() - phase_start
                    )
                    row["free_response"] = free_summary(
                        prediction if workspace is None else prediction.payload, target
                    )
            elif actor is not None:
                phase_start = time.perf_counter()
                row["ack_work"] = (
                    actor.reader.acknowledge(prediction, state, learning_enabled=False)
                    if workspace is None
                    else actor.finish_without_teaching(
                        prediction, observation, state, envelope, learning_enabled=False
                    )
                )
                row["phase_seconds"]["disabled_acknowledgement"] = time.perf_counter() - phase_start
            after = baseline_hashes(reference) if actor is None else hashes(actor, state)
            if before["memory"] != after["memory"]:
                raise ValueError("repeated teaching changed factual/source memory")
            row.update(
                model_after_sha256=after["model"],
                memory_after_sha256=after["memory"],
                elapsed_seconds=time.perf_counter() - tick,
            )
            if teacher_row is not None:
                teacher_row["model_after_sha256"] = after["model"]
                teacher_row["memory_before_sha256"] = before["memory"]
                teacher_row["memory_after_sha256"] = after["memory"]
            live["completed_attempts"] += 1
            live["completed_teachers"] += int(teaching)
            for phase, seconds in row["phase_seconds"].items():
                key = scope + "_" + phase + "_seconds"
                phases[key] = phases.get(key, 0) + seconds
            return row, teacher_row

        def checkpoint(index):
            panel_start = time.perf_counter()
            before = {r: hashes(models[r], states[r]) for r in models}
            counters_before = {
                r: (m._experience, m.reader._active_experience) for r, m in models.items()
            }
            before["simple"] = baseline_hashes(baseline)
            rows, clone_receipts, copy_seconds = [], {}, {}
            pending["probe_rows"] = rows
            for route in ROUTES:
                copy_start = time.perf_counter()
                actor, state, reference = (None, None, copy.deepcopy(baseline))
                if route != "simple":
                    cloned = clone_for_probe(models[route], states[route])
                    actor, state = cloned["model"], cloned["state"]
                    actor.reader.controller.rng_state.copy_(diagnostic_rng)
                    clone_receipts[route] = {
                        "work": cloned["work"],
                        "incoming_rng_sha256": _digest({"rng_state": diagnostic_rng}, work),
                        "policy": "fixed initial parent RNG on disposable panel only",
                    }
                copy_seconds[route] = time.perf_counter() - copy_start
                phases["panel_clone_copy_seconds"] = (
                    phases.get("panel_clone_copy_seconds", 0) + copy_seconds[route]
                )
                for name in manifest["checkpoint_cases"]:
                    row, teacher = attempt(
                        route,
                        name,
                        actor,
                        state,
                        reference,
                        scope="probe",
                        index=index,
                        teaching=name.startswith("A00"),
                    )
                    rows.append({"attempt": row, "teacher": teacher})
                del actor, state, reference
            after = {r: hashes(models[r], states[r]) for r in models}
            after["simple"] = baseline_hashes(baseline)
            if before != after or counters_before != {
                r: (m._experience, m.reader._active_experience) for r, m in models.items()
            }:
                raise ValueError("disposable diagnostics changed persistent learner")
            record = {
                "checkpoint": index,
                "live_before": before,
                "live_after": after,
                "clone_receipts": clone_receipts,
                "copy_seconds": copy_seconds,
                "rows": rows,
            }
            append_start = time.perf_counter()
            owned["probes"].append(record)
            phases["panel_journal_append_seconds"] = (
                phases.get("panel_journal_append_seconds", 0) + time.perf_counter() - append_start
            )
            live["process_memory"] = process_memory()
            if live["process_memory"]["working_set_mib"] > 4096:
                raise RuntimeError(
                    "practice process exceeds4096MiB; outer watchdog owns tree limit"
                )
            probes.append(
                {
                    "checkpoint": index,
                    "rows": [
                        {
                            "attempt": {
                                k: value["attempt"][k] for k in ("route", "episode_id", "correct")
                            }
                        }
                        for value in rows
                    ],
                }
            )
            pending.clear()
            print(
                json.dumps(
                    {
                        "checkpoint": index,
                        "free_correct": {
                            r: sum(
                                x["attempt"]["correct"] for x in rows if x["attempt"]["route"] == r
                            )
                            for r in ROUTES
                        },
                        "elapsed": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
            admit()
            phases["panel_total_seconds"] = (
                phases.get("panel_total_seconds", 0) + time.perf_counter() - panel_start
            )

        checkpoint(0)
        for item in manifest["schedule"]:
            for route in ROUTES:
                row, teacher = attempt(
                    route,
                    item["episode_id"],
                    models.get(route),
                    states.get(route),
                    baseline,
                    scope="main",
                    index=item["index"],
                    teaching=True,
                )
                phase_start = time.perf_counter()
                owned["attempts"].append(row)
                owned["teachers"].append(teacher)
                phases["practice_journal_append_seconds"] = (
                    phases.get("practice_journal_append_seconds", 0)
                    + time.perf_counter()
                    - phase_start
                )
                pending.clear()
                results.append(
                    {k: row[k] for k in ("step_index", "route", "episode_id", "correct")}
                )
                if (
                    _digest(
                        {
                            k: v
                            for k, v in models["learning_disabled"].state_dict().items()
                            if k != "reader.controller.rng_state"
                        },
                        work,
                    )
                    != disabled_slow
                ):
                    raise ValueError("learning-disabled slow state changed")
            if item["index"] + 1 in CHECKPOINTS:
                checkpoint(item["index"] + 1)
        host = {
            "initial_hashes": initial,
            "counts": live,
            "schedule_completed": 1024,
            "results": results,
            "work": work,
            "phase_seconds": phases,
            "elapsed_seconds": time.perf_counter() - started,
        }
        admit()
        final = store.commit_final(models, states, baseline, host)
        for journal in journals:
            journal.close()
        store.close()
        report = {
            "status": "completed_pending_saved_audit",
            "manifest_sha256": sha(output / "manifest.json"),
            "initial_hashes": initial,
            "final": final,
            "counts": live,
            "restore": None,
            "rule": args.rule,
            "source_revision": source_revision,
            "phase_seconds": phases,
            "timing_scope": (
                "Case phases exclude journal append. Teacher includes internal custody/hashes. "
                "Explicit hashes exclude nested model hashes. Admission inventory measures "
                "artifact traversal and free-space lookup, excluding RSS checks. Panel clone/copy "
                "includes reference copies, independent neural snapshots/restores and diagnostic "
                "RNG assignment/hash. Panel total includes its attempts, hashes, copies, append, "
                "and admission; these component times must not be added to panel total. "
                "Phases overlap case/run elapsed."
            ),
            "results": results,
            "checkpoints": [
                {
                    "checkpoint": p["checkpoint"],
                    "outcomes": [
                        {k: x["attempt"][k] for k in ("route", "episode_id", "correct")}
                        for x in p["rows"]
                    ],
                }
                for p in probes
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
        report["acquisition"] = {
            r: {
                "both_skills_at_final_two_checkpoints": all(
                    x["attempt"]["correct"]
                    for p in probes
                    if p["checkpoint"] in (768, 1024)
                    for x in p["rows"]
                    if x["attempt"]["route"] == r and x["attempt"]["episode_id"].startswith("A00")
                ),
                "complete_response_denominator": 4,
                "expansion_released": False,
            }
            for r in ROUTES
        }
        if (live["completed_attempts"], live["completed_teachers"], live["executions"]) != (
            3156,
            3114,
            1557,
        ):
            raise ValueError("fixed exposure denominators differ")
        _check_sources(sources)
        runtime.check_origins(selection)
        publish("report.json", report)
        admit(False)
        publish(
            "completion.json",
            {
                "schema": "integrated-local-learning-worker-completion-v1",
                "status": "completed_pending_saved_audit",
                "rule": args.rule,
                "metadata_sha256": {
                    name: sha(output / name)
                    for name in (
                        "report.json",
                        "manifest.json",
                        "sources.json",
                        "initialization.json",
                        "launch.json",
                    )
                },
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed_pending_saved_audit",
                    "counts": live,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        try:
            publish(
                "failure.json",
                {
                    "type": type(error).__name__,
                    "error": str(error),
                    "counts": live,
                    "pending_evidence": pending,
                    "elapsed_seconds": time.perf_counter() - started,
                    "automatic_resume": False,
                },
            )
        except BaseException as reporting_error:
            print(
                f"Failure receipt error: {reporting_error!r}; original: {error!r}", file=sys.stderr
            )
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


def audit(args):
    """Fresh selected worker; exact saved bytes and strict restore, no observations."""
    if DRAFT:
        raise RuntimeError("DRAFT practice requires source/protocol release")
    started = time.perf_counter()
    output = Path(args.output).resolve()
    audit_output = Path(args.audit_output).resolve()
    if not audit_output.is_relative_to(ROOT) or audit_output.is_relative_to(output):
        raise ValueError("fresh audit receipt must be separate checkout-local evidence")
    completion_raw = (output / "completion.json").read_bytes()
    if len(completion_raw) > 1024**2:
        raise ValueError("practice completion exceeds metadata bound")
    completion = json.loads(completion_raw)
    if set(completion["metadata_sha256"]) != {
        "report.json",
        "manifest.json",
        "sources.json",
        "initialization.json",
        "launch.json",
    }:
        raise ValueError("practice completion needs all five metadata bindings")
    if (
        completion["schema"] != "integrated-local-learning-worker-completion-v1"
        or completion["status"] != "completed_pending_saved_audit"
        or completion["rule"] != args.rule
        or (output / "failure.json").exists()
        or (output / "cleanup-errors.json").exists()
    ):
        raise ValueError("practice worker did not complete a clean full schedule")
    for name, identity in completion["metadata_sha256"].items():
        if name not in {
            "report.json",
            "manifest.json",
            "sources.json",
            "initialization.json",
            "launch.json",
        }:
            raise ValueError("unknown completion metadata path")
        if sha(output / name) != identity:
            raise ValueError("completed practice metadata changed")
    selection = runtime.select(args.rule, args.predecessor_snapshot)
    from experiments.r4_integrated.integrated_local_learning_checkpoint_v1 import audit_saved
    from experiments.r4_integrated.integrated_local_learning_preservation_v1 import _check_sources
    from experiments.r4_integrated.neuronal_splice_io_v1 import process_memory
    from humanbrain.recurrent_content_learner import RecurrentContentSpliceLearner

    sources = json.loads((output / "sources.json").read_bytes())
    if sources["selection"]["module_origins"] != selection["module_origins"]:
        raise ValueError("fresh audit selected different neuronal implementations")
    _check_sources(sources["source_sha256"])
    report = json.loads((output / "report.json").read_bytes())
    checked = audit_saved(
        output / "checkpoints",
        config_sha256=report["final"]["config_sha256"],
        actor_class=RecurrentContentSpliceLearner,
    )
    runtime.check_origins(selection)
    _check_sources(sources["source_sha256"])
    if time.perf_counter() - started >= 600:
        raise TimeoutError("fresh saved audit exceeded its600second outer budget")
    memory = process_memory()
    if memory["working_set_mib"] > 2048:
        raise RuntimeError("fresh saved audit exceeds2048MiB")
    checked.update(
        rule=args.rule,
        worker_completion_sha256=hashlib.sha256(completion_raw).hexdigest(),
        elapsed_seconds=time.perf_counter() - started,
        process_memory=memory,
    )
    if (
        sum(p.stat().st_size for p in audit_output.parent.rglob("*") if p.is_file())
        + len(encode(checked))
        > 64 * 1024**2
    ):
        raise RuntimeError("shared audit receipt directory exceeds64MiB")
    save(audit_output, checked)
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "audit"))
    parser.add_argument("--base", type=Path, default=runtime.PARENT / "checkpoints/manifest.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rule", choices=("predecessor", "current"))
    parser.add_argument("--predecessor-snapshot", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        save(args.manifest, make_manifest(args.base))
        print(json.dumps({"manifest_sha256": sha(args.manifest), "model_calls": 0}), flush=True)
    elif args.command == "run":
        run(args)
    else:
        result = audit(args)
        print(
            json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]})
        )


if __name__ == "__main__":
    main()
