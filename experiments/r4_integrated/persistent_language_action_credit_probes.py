"""Disposable exact-checkpoint measurements for the stable credit comparison.

Import performs no model construction, checkpoint load, prediction or execution.
The caller selects a source-bound old/current runtime before invoking the callback.
Probe actors never escape this module or publish state to the persistent actors.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
import weakref
from pathlib import Path

from experiments.r4_integrated.stable_language_action_credit_workload_v1 import (
    CHECKPOINTS,
    probe_episodes,
)

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("online", "learning_disabled", "simple")
ROW_LIMIT = 64 * 1024 * 1024
FREE_RESERVE = 8 * 1024**3


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _seal(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_identity(path):
    digest, length = hashlib.sha256(), 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            length += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": length}


def _save(path, value):
    raw = _json_bytes(value)
    if len(raw) > ROW_LIMIT:
        raise ValueError("probe summary exceeds its bounded record size")
    with Path(path).open("xb") as handle:
        if handle.write(raw) != len(raw):
            raise OSError("probe summary write shortened")
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _feedback(record):
    from humanbrain.sequence_assembly_observation import ToolFeedback

    if record is None:
        return None
    return ToolFeedback(**{key: value for key, value in record.items()
                           if key not in ("stdout_hex", "stderr_hex")},
                        stdout=bytes.fromhex(record["stdout_hex"]),
                        stderr=bytes.fromhex(record["stderr_hex"]))


def _initial_observation(episode, manifest, route):
    """Use actual saved historical inputs; independent cases have no host repair."""
    if episode.get("predecessor_episode_id") is not None:
        raise ValueError("an independent probe cannot fabricate a project predecessor")
    if episode["phase"] == "retention":
        parent = "simple" if route == "simple" else "online"
        record = manifest["prior_regression_bindings"]["routes"][parent][
            episode["prior_episode_id"]]["observation"]
        if record["request_hex"] != episode["request_hex"]:
            raise ValueError("probe changed the saved historical request")
        return record["files"] or [], _feedback(record["feedback"])
    return episode["files"], None


def _numeric(actor, state, work):
    import torch

    from humanbrain.neuronal_splice_learner import _digest

    values = {key: value for key, value in actor.state_dict().items()
              if isinstance(value, torch.Tensor)}
    rng = values.pop("reader.controller.rng_state")
    return {"slow_numeric_sha256": _digest(values, work),
            "rng_sha256": _digest({"rng_state": rng}, work),
            "factual_sha256": _digest(vars(state), work), "memory_used": state.used,
            "actor_experience": actor._experience,
            "reader_experience": actor.reader._active_experience}


def _tick_signature(tick):
    # Whitelist the original forward contract. New local-credit diagnostics,
    # stage seals and metadata hashes are intentionally outside this comparison.
    result = {key: tick[key] for key in (
        "category", "frame_sha256", "position_sha256", "field", "offset", "arrived",
        "h_before", "h_after", "cursors_before", "cursors_after", "register_tokens",
        "prior_token", "retrieved_index", "emitted_token")}
    result["draws"] = [{key: draw[key] for key in (
        "population", "category", "index", "logits", "probabilities", "uniform",
        "activity_sha256", "rng_before_sha256", "rng_after_sha256", "ordinal")}
        for draw in tick["draws"]]
    return result


def _reader_signature(prediction):
    result = {key: prediction[key] for key in ("answer_hex", "stop_reason", "decisions")}
    result["scan"] = [_tick_signature(tick) for tick in prediction["scan_steps"]]
    result["free"] = [{
        **{key: step[key] for key in (
            "output_step", "prefix_hex", "token", "selected_index", "address_scores",
            "assembly_index", "assembly_scores", "prior_logits", "prior_nll")},
        "controller": _tick_signature(step["controller"]),
    } for step in prediction["steps"]]
    return result


def _semantic_signature(row):
    result = {key: row[key] for key in (
        "episode_id", "attempt_index", "kind", "status", "observation",
        "answer_hex", "proposal", "before_files", "after_files", "correct", "grade")}
    prediction = row["prediction"]
    if row["route"] != "simple" and prediction is not None:
        if row["kind"] == "reading":
            result["forward"] = _reader_signature(prediction)
        else:
            result["forward"] = {"payload": _reader_signature(prediction["payload"]),
                                 "heads": prediction["heads"]}
    if row["execution"] is not None:
        execution = row["execution"]
        result["execution"] = {key: execution[key] for key in (
            "accepted", "rejection", "returncode", "timeout", "output_limit_exceeded",
            "capture_complete", "files_complete",
            "before_files", "proposed_files", "after_files")}
        result["driver"] = {key: execution["public_driver_policy"][key] for key in (
            "proposal_rejected", "before_driver", "proposed_driver", "after_driver",
            "driver_integrity")}
    return _seal(result)


def _compare(left, right):
    """Independent cases must pair; recovery retains actual context divergence."""
    if len(left) != 28 or len(right) != 28:
        raise ValueError("migration comparison is missing a required probe attempt")
    rows = []
    for index, (before, after) in enumerate(zip(left, right, strict=True)):
        coordinate = (before["episode_id"], before["attempt_index"])
        if coordinate != (after["episode_id"], after["attempt_index"]):
            raise ValueError("migration comparison changed the fixed battery order")
        paired = (before["observation_sha256"] == after["observation_sha256"]
                  and before["rng_before_sha256"] == after["rng_before_sha256"])
        equal = before["semantic_sha256"] == after["semantic_sha256"]
        # The first 22 independent encounters are mandatory, as is the first
        # recovery encounter after their equal trajectory. Later real feedback
        # can differ and is never normalized or replaced to manufacture pairing.
        required = index <= 22
        rows.append({"episode_id": coordinate[0], "attempt_index": coordinate[1],
                     "required_independent_or_first_recovery": required,
                     "status": "paired_equal" if paired and equal else (
                         "paired_mismatch" if paired else "unpaired_context_or_rng"),
                     "raw_diagnostics_equal": before["diagnostics"] == after["diagnostics"],
                     "left": before, "right": after})
    return {"passed": all(row["status"] == "paired_equal" if
                           row["required_independent_or_first_recovery"] else
                           row["status"] != "paired_mismatch" for row in rows),
            "rows": rows,
            "scope": ("required neuronal forward and task-outcome parity for paired actual "
                      "context/RNG; raw diagnostic parity reported separately, unnormalized")}


def _run_route(route, model, state, baseline, episodes, manifest, directory, journal,
               launch, deadline, admit, work, progress):
    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        grade_execution,
        proposal_from_record,
        workspace_from_records,
    )
    from experiments.r4_integrated.persistent_language_action_v8 import _plain, reader_summary
    from experiments.r4_integrated.persistent_language_action_v8_execution import (
        execute_proposal,
        feedback_from_receipt,
    )
    from experiments.r4_integrated.persistent_session_checkpoint import _baseline_record
    from humanbrain.sequence_assembly_observation import RawObservation

    initial = _numeric(model, state, work) if model is not None else {
        "reference_sha256": _seal(_baseline_record(baseline))}
    summaries, children, totals = [], 0, {}
    for episode in episodes:
        files, feedback = _initial_observation(episode, manifest, route)
        if episode["teacher"] is not None or episode["append_document_ids"]:
            raise ValueError("probe received a forbidden source or teacher offer")
        attempts = 3 if episode["probe_group"] == "recovery" else 1
        if episode["attempt_limit"] != attempts:
            raise ValueError("probe attempt count differs from its fixed battery")
        for attempt_index in range(1, attempts + 1):
            admit()
            started = time.perf_counter()
            before = _numeric(model, state, work) if model is not None else initial
            row = {"schema": "stable-credit-checkpoint-probe-attempt-v1", "route": route,
                   "episode_id": episode["episode_id"], "probe_group": episode["probe_group"],
                   "attempt_index": attempt_index, "kind": episode["kind"],
                   "observation": {"request_hex": episode["request_hex"], "files": files,
                                   "feedback": None if feedback is None else feedback.to_record()},
                   "before_files": files, "after_files": files, "status": "completed",
                   "answer_hex": None, "proposal": None, "prediction": None,
                   "execution": None, "grade": None, "correct": False,
                   "numeric_before": before, "work": {}, "ack_work": {},
                   "learning_enabled": False, "writes_enabled": False,
                   "source_offers": 0, "teacher_offers": 0}
            row["model_instance_id"] = id(baseline if model is None else model)
            progress["pending_attempt"] = row
            observation = workspace = prediction = proposal = None
            try:
                workspace = workspace_from_records(files) if episode["kind"] == "edit" else None
                observation = RawObservation(bytes.fromhex(episode["request_hex"]),
                                             workspace, feedback)
                row["observation"] = observation.to_record()
            except (TypeError, ValueError) as error:
                row["status"], row["error"] = "unsupported", str(error)
            if observation is not None:
                if model is None:
                    row["prediction"] = baseline.predict(observation, kind=episode["kind"])
                    if episode["kind"] == "reading":
                        row["answer_hex"] = row["prediction"]["answer_hex"]
                    else:
                        proposal = proposal_from_record(row["prediction"]["proposal"], workspace)
                elif episode["kind"] == "reading":
                    prediction = model.reader.predict_with_receipt(observation, state, 128)
                    row["prediction"] = reader_summary(prediction)
                    row["answer_hex"] = prediction.generation.data.hex()
                    row["work"] = dict(prediction.work)
                else:
                    prediction = model.propose(observation, state)
                    proposal = prediction.proposal
                    row["prediction"] = {
                        "attempt_id": prediction.attempt_id,
                        "proposal_sha256": prediction.proposal_sha256,
                        "observation_sha256": prediction.observation_sha256,
                        "model_sha256": prediction.model_sha256,
                        "memory_sha256": prediction.memory_sha256,
                        "payload": reader_summary(prediction.payload),
                        "heads": {key: {"logits": value.logits.tolist(),
                                        "selected_index": value.selected_index}
                                  for key, value in prediction.choices.items()}}
                    row["work"] = dict(prediction.work)
            if model is not None:
                pending = _numeric(model, state, work)
                if any(pending[key] != before[key] for key in (
                        "slow_numeric_sha256", "rng_sha256", "factual_sha256", "memory_used")):
                    raise ValueError("unresolved disposable prediction changed published state")
                if (pending["reader_experience"] != before["reader_experience"]
                        + int(prediction is not None)
                        or pending["actor_experience"] != before["actor_experience"]
                        + int(prediction is not None and episode["kind"] == "edit")):
                    raise ValueError("probe prediction custody counters changed unexpectedly")
            if episode["kind"] == "edit":
                row["proposal"] = None if proposal is None else proposal.to_record()
                admit()
                after_files, execution = execute_proposal(
                    workspace if workspace is not None else files, proposal,
                    directory / "executions" / route / episode["episode_id"] / str(attempt_index),
                    launch=launch, deadline=deadline)
                children += 1
                row["execution"], row["after_files"] = execution, after_files
                row["grade"] = grade_execution(execution, episode["grade"], episode["files"])
                row["correct"] = row["status"] != "unsupported" and row["grade"]["task_success"]
                if prediction is not None:
                    envelope = {"attempt_id": prediction.attempt_id,
                                "proposal_sha256": prediction.proposal_sha256,
                                "workspace_sha256": workspace.sha256, "execution": execution}
                    row["ack_work"] = model.finish_without_teaching(
                        prediction, observation, state, envelope, learning_enabled=False)
                if not execution["files_complete"]:
                    raise ValueError("probe cannot continue from incomplete actual execution files")
                feedback = feedback_from_receipt(execution)
            else:
                row["correct"] = (row["status"] != "unsupported" and
                                  row["answer_hex"] in episode["grade"]["accepted_answer_hex"])
                if prediction is not None:
                    row["ack_work"] = model.reader.acknowledge(
                        prediction, state, learning_enabled=False)
            if model is not None:
                row["controller_resolution"] = _plain(model.reader.last_controller_resolution)
                after = _numeric(model, state, work)
                if any(after[key] != initial[key] for key in (
                        "slow_numeric_sha256", "factual_sha256", "memory_used")):
                    raise ValueError("probe published slow state or factual writes")
                if any(after[key] != pending[key] for key in (
                        "actor_experience", "reader_experience")):
                    raise ValueError("probe acknowledgement changed prediction custody counters")
                if model._pending is not None or model.reader.pending_prediction is not None:
                    raise ValueError("probe left an unresolved neuronal response")
            else:
                after = {"reference_sha256": _seal(_baseline_record(baseline))}
                if after != initial:
                    raise ValueError("probe changed reference history")
            row["numeric_after"] = after
            row["elapsed_seconds"] = time.perf_counter() - started
            semantic = _semantic_signature(row)
            row["semantic_sha256"] = semantic
            admit()
            identity = journal.append(row)
            progress.pop("pending_attempt")
            summaries.append({
                "episode_id": episode["episode_id"], "attempt_index": attempt_index,
                "probe_group": episode["probe_group"], "kind": episode["kind"],
                "correct": row["correct"], "status": row["status"],
                "observation_sha256": _seal(row["observation"]),
                "rng_before_sha256": before.get("rng_sha256"),
                "rng_after_sha256": after.get("rng_sha256"),
                "semantic_sha256": semantic, "row_sha256": identity,
                "feedback": row["observation"]["feedback"],
                "diagnostics": None if row["execution"] is None else {
                    key: row["execution"][key]
                    for key in ("stdout_hex", "stderr_hex", "capture_errors")},
            })
            for scope in ("work", "ack_work"):
                for key, value in row[scope].items():
                    if type(value) in (int, float):
                        name = scope + "/" + key
                        totals[name] = totals.get(name, 0) + value
            files = row["after_files"]
            del prediction, row
    if len(summaries) != 28 or children != 17:
        raise ValueError("probe battery did not perform exactly28 attempts and17 verifications")
    return {"initial": initial, "final": after, "attempts": summaries,
            "actual_executions": children, "canonical_work": totals}


def checkpoint_callback(*, checkpoint_index, checkpoint_directory, boundary_sha256,
                        config_sha256, manifest, output, deadline, credit_rule,
                        old_cp0_reference=None, admit_work=None):
    """Restore disposable arms, measure once, discard, and return only a receipt."""
    if checkpoint_index not in CHECKPOINTS or credit_rule not in {"old", "causal"}:
        raise ValueError("unknown fixed checkpoint or credit rule")
    if checkpoint_index == 0 and credit_rule == "causal" and old_cp0_reference is None:
        raise ValueError("causal CP0 requires the authenticated saved old CP0 reference")
    directory = Path(output).resolve() / "probes" / f"cp-{checkpoint_index}"
    directory.mkdir(parents=True, exist_ok=False)
    started, work = time.perf_counter(), {}

    def admit():
        if time.perf_counter() >= deadline:
            raise TimeoutError("probe reached the shared stop-reserve deadline")
        if shutil.disk_usage(directory).free < FREE_RESERVE:
            raise RuntimeError("probe reached the declared free-space reserve")
        if admit_work is not None:
            admit_work()

    models, states, restored, references, results = {}, {}, {}, [], {}
    progress = {}
    store = journal = baseline = None
    baseline_reference, restore_receipt = None, None
    completed, comparison = False, {}
    primary_error, cleanup_errors = None, []
    try:
        admit()
        from experiments.r4_integrated.persistent_checkpoint_journal import OwnedJournal
        from experiments.r4_integrated.persistent_language_action_v8_execution import (
            qualified_launch,
        )
        from experiments.r4_integrated.stable_language_action_credit_checkpoint_v1 import (
            StableLanguageActionCreditCheckpoint,
        )
        from humanbrain.recurrent_content_learner import RecurrentContentSpliceLearner

        episodes = probe_episodes(manifest)
        store = StableLanguageActionCreditCheckpoint.open(
            Path(checkpoint_directory).resolve(), config_sha256=config_sha256)
        if store.manifest != manifest:
            raise ValueError("probe manifest differs from the source-bound checkpoint")
        models = {route: RecurrentContentSpliceLearner(seed=733) for route in ROUTES[:2]}
        references = [weakref.ref(model) for model in models.values()]
        restored = store.restore_into(models)
        states, baseline = restored["states"], restored["baseline"]
        baseline_reference = weakref.ref(baseline)
        restore_receipt = {key: restored[key] for key in (
            "boundary_sha256", "elapsed_seconds", "work")}
        if (restored["boundary_sha256"] != boundary_sha256 or restored["host"]["cursor"] != {
                "episode_index": checkpoint_index, "route_index": 0,
                "attempt_index": 1, "append_done": False}):
            raise ValueError("probe did not restore the exact scheduled safe boundary")
        store.close()
        launch = qualified_launch(ROOT)
        _save(directory / "launch.json", launch.to_record())
        journal = OwnedJournal(directory / "attempts.jsonl", work)
        for route in ROUTES:
            admit()
            results[route] = _run_route(
                route, models.get(route), states.get(route), baseline, episodes, manifest,
                directory, journal, launch, deadline, admit, work, progress)
        if checkpoint_index == 0:
            comparison["within_worker"] = _compare(
                results["online"]["attempts"], results["learning_disabled"]["attempts"])
            if results["online"]["initial"] != results["learning_disabled"]["initial"]:
                raise ValueError("CP0 neural routes did not start from identical numeric state")
            if old_cp0_reference is not None:
                reference_path = Path(old_cp0_reference["path"]).resolve()
                with reference_path.open("rb") as handle:
                    raw = handle.read(ROW_LIMIT + 1)
                if len(raw) > ROW_LIMIT or hashlib.sha256(raw).hexdigest() != old_cp0_reference[
                        "sha256"]:
                    raise ValueError("saved old CP0 reference changed")
                old = json.loads(raw)
                if (old["status"] != "completed" or old["checkpoint_index"] != 0
                        or old["credit_rule"] != "old"
                        or old["manifest_sha256"] != store.config["manifest_sha256"]
                        or old["manifest_content_sha256"] != _seal(manifest)
                        or not old["destruction"]["all_actors_released"]):
                    raise ValueError("saved old CP0 is not a complete matched battery")
                for route in ROUTES:
                    if old["routes"][route]["initial"] != results[route]["initial"]:
                        raise ValueError("CP0 migration changed retained numeric/reference state")
                    comparison[route] = _compare(old["routes"][route]["attempts"],
                                                 results[route]["attempts"])
            _save(directory / "migration.json", comparison)
            if any(not value["passed"] for value in comparison.values()):
                raise ValueError("required paired CP0 forward or task behavior differs")
        completed = True
    except BaseException as error:
        primary_error = error
        partial, partial_error = None, None
        if progress.get("pending_attempt") is not None:
            try:
                partial = _save(directory / "interrupted-attempt.json", {
                    "status": "incomplete", "attempt": progress["pending_attempt"],
                    "journal_append_completion": "not acknowledged; inspect retained bytes",
                    "scope": "actual partial receipt; no inferred missing draws or outcomes"})
            except BaseException as receipt_error:
                partial_error = f"{type(receipt_error).__name__}: {receipt_error}"
        try:
            _save(directory / "failure.json", {
                "status": "failed", "type": type(error).__name__, "error": str(error),
                "checkpoint_index": checkpoint_index, "boundary_sha256": boundary_sha256,
                "completed_routes": list(results), "work": work,
                "interrupted_attempt": partial, "interrupted_receipt_error": partial_error,
                "journal": None if journal is None else journal.metrics(),
                "elapsed_seconds": time.perf_counter() - started,
                "live_state_publication": False, "automatic_retry": False})
        except BaseException as receipt_error:
            cleanup_errors.append(
                f"failure receipt: {type(receipt_error).__name__}: {receipt_error}")
        raise
    finally:
        for name, resource in (("journal", journal), ("store", store)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as close_error:
                    cleanup_errors.append(f"{name}: {type(close_error).__name__}: {close_error}")
        models.clear()
        states.clear()
        restored.clear()
        baseline = None
        try:
            gc.collect()
        except BaseException as collect_error:
            cleanup_errors.append(f"gc: {type(collect_error).__name__}: {collect_error}")
        destruction = {"constructed_actors": len(references),
                       "all_actors_released": all(reference() is None for reference in references),
                       "reference_released": (baseline_reference is None
                                              or baseline_reference() is None),
                       "published_state_to_live": False, "cleanup_errors": cleanup_errors}
        try:
            _save(directory / "destruction.json", destruction)
        except BaseException as receipt_error:
            cleanup_errors.append(
                f"destruction receipt: {type(receipt_error).__name__}: {receipt_error}")
        if primary_error is not None and cleanup_errors:
            primary_error.add_note("Probe cleanup errors: " + "; ".join(cleanup_errors))
    if cleanup_errors:
        error = RuntimeError("Probe cleanup failed: " + "; ".join(cleanup_errors))
        try:
            _save(directory / "failure.json", {
                "status": "failed", "type": type(error).__name__, "error": str(error),
                "checkpoint_index": checkpoint_index, "boundary_sha256": boundary_sha256,
                "cleanup_errors": cleanup_errors, "live_state_publication": False})
        except BaseException as receipt_error:
            error.add_note(f"Failure receipt: {type(receipt_error).__name__}: {receipt_error}")
        raise error
    if (not completed or not destruction["all_actors_released"]
            or not destruction["reference_released"]):
        raise ValueError("disposable probe actor destruction was not established")
    admit()
    journal_identity = _file_identity(directory / "attempts.jsonl")
    admit()
    summary = {
        "schema": "stable-credit-checkpoint-probe-summary-v1", "status": "completed",
        "checkpoint_index": checkpoint_index, "boundary_sha256": boundary_sha256,
        "config_sha256": config_sha256, "manifest_sha256": store.config["manifest_sha256"],
        "manifest_content_sha256": _seal(manifest),
        "credit_rule": credit_rule, "routes": results, "comparison": comparison,
        "destruction": destruction, "attempts": 84, "actual_executions": 51,
        "source_offers": 0, "teacher_offers": 0, "live_state_publications": 0,
        "journal": journal_identity, "work": work,
        "restore": restore_receipt, "new_actors": 2, "restored_references": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "final_rng_policy": "report actual frontiers; no equality after unpaired recovery contexts",
        "comparison_semantics": ("required paired forward and task outcomes; raw diagnostics "
                                 "reported separately and remain unchanged in recovery input"),
    }
    return _save(directory / "summary.json", summary)
