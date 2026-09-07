"""Dendritic continuation with source learning and durable idle boundaries.

Only stdlib and frozen IO helpers import here. Neural initialization and actual
execution belong to run(), held behind the reviewed CLI release guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Direct script execution needs the repository import root; these helpers are stdlib only.
from experiments.r4_integrated.persistent_language_action_v5 import (  # noqa: E402
    PHASES,
    ROUTES,
    Journal,
    _plain,
    encode,
    save,
    sha,
)

DRAFT = False
PRIOR_ATTEMPTS_SHA = "9139824c7897f15e3a3aed71a7692f7a15c28ad18fa4d82d48072dee5981e638"
MANIFEST_SHA = "c2e1cb6237fdff468c6d413ce62c9dfbf514a607fb1a8af45bdde285c3f4b898"
RUNNER_SECONDS = 1800.0  # one root-selected complete guarded attempt
AUDIT_SECONDS = 120.0
CHECKPOINT_RESERVE_SECONDS = 30.0
SAVED_REVIEW_SOURCES = {
    "experiments/r4_integrated/persistent_language_action_v7_review.py",
    "experiments/r4_integrated/bank_persistent_language_action_v7.py",
}


class SessionJournal(Journal):
    """The existing durable encoding with actual byte and complete append timing."""

    def __init__(self, path):
        super().__init__(path)
        self.bytes_written = 0
        self.max_row_bytes = 0
        self.append_seconds = 0.0

    def append(self, row):
        started = time.perf_counter()
        try:
            raw = encode(row)
            self.handle.write(raw)
            self.handle.flush()
            os.fsync(self.handle.fileno())
            digest = hashlib.sha256(raw).hexdigest()
            self.rows += 1
            self.bytes_written += len(raw)
            self.max_row_bytes = max(self.max_row_bytes, len(raw))
            return digest
        finally:
            self.append_seconds += time.perf_counter() - started

    def metrics(self):
        return {
            "rows": self.rows,
            "bytes_written": self.bytes_written,
            "max_row_bytes": self.max_row_bytes,
            "append_seconds": self.append_seconds,
            "scope": "encoding, write, flush, fsync and row hash; failed append time included",
        }


def reader_summary(prediction):
    """Actual scalar routing and output trace; no prototype or replay claim."""
    return {
        "schema": "dendritic-sequence-output-prediction-v1",
        "attempt_id": prediction.attempt_id,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "trace_sha256": prediction.trace_sha256,
        "max_bytes": prediction.max_bytes,
        "observation_sha256": prediction.observation.sha256,
        "answer_hex": prediction.generation.data.hex(),
        "stop_reason": prediction.generation.stop_reason,
        "decisions": len(prediction.steps),
        "tokens": [step.observed_token for step in prediction.steps],
        "steps": [
            {
                "output_step": step.output_step,
                "prefix_hex": step.prefix.hex(),
                "token": step.observed_token,
                "selected_index": step.selected_index,
                "address_scores": step.address_scores.tolist(),
                "assembly_index": step.assembly_index,
                "assembly_scores": step.assembly_scores.tolist(),
                "exact_context": step.exact_context,
                "routing": _plain(step.routing),
                "binding": _plain(step.binding),
                "prior_logits": step.logits.tolist(),
                "prior_nll": step.nll,
            }
            for step in prediction.steps
        ],
        "work": prediction.work,
    }


def load_prior_outcomes(path, expected_sha):
    """One authenticated scalar/raw-observation stream; no neural reconstruction."""
    if expected_sha != PRIOR_ATTEMPTS_SHA:
        raise ValueError("prior outcomes require the fixed completed v5 journal")
    tick = time.perf_counter()
    digest, total_bytes, count = hashlib.sha256(), 0, 0
    routes = {route: {} for route in ROUTES}
    ordinals = dict.fromkeys(ROUTES, 0)
    with Path(path).open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            total_bytes += len(raw)
            count += 1
            row = json.loads(raw)
            if (
                row["schema"] != "persistent-language-action-attempt-v5"
                or row["route"] not in routes
            ):
                raise ValueError("prior outcome row identity changed")
            route = row["route"]
            routes[route][row["episode_id"]] = {
                "episode_id": row["episode_id"],
                "attempt_index": row["attempt_index"],
                "correct": row["correct"],
                "request_hex": row["request_hex"],
                "before_files": row["before_files"],
                "observation": row["observation"],
                "observation_sha256": hashlib.sha256(encode(row["observation"])).hexdigest(),
                "route_ordinal": ordinals[route],
            }
            ordinals[route] += 1
    if digest.hexdigest() != expected_sha or ordinals != dict.fromkeys(ROUTES, 80):
        raise ValueError("prior journal changed or lacks the complete three-route stream")
    receipt = {
        "schema": "persistent-language-action-prior-outcomes-v7",
        "path": str(Path(path).resolve()),
        "sha256": expected_sha,
        "read_passes": 1,
        "bytes_read": total_bytes,
        "rows_read": count,
        "route_attempts": ordinals,
        "elapsed_seconds": time.perf_counter() - tick,
        "scope": "last actual v5 attempt and raw observation; no prediction/model replay",
    }
    return routes, receipt


def summarize(attempts, manifest, prior_outcomes):
    """Four primary outcomes, explicit encounter classes and actual retention age."""
    result = {}
    for route in ROUTES:
        rows = [row for row in attempts if row["route"] == route]
        first = {row["episode_id"]: row for row in rows if row["attempt_index"] == 1}
        phases = {}
        for phase in PHASES:
            selected = [row for row in rows if row["phase"] == phase]
            phases[phase] = {
                kind: {
                    "episodes": sum(
                        row["kind"] == kind and row["attempt_index"] == 1 for row in selected
                    ),
                    "attempts": sum(row["kind"] == kind for row in selected),
                    "first_correct": sum(
                        row["correct"]
                        for row in selected
                        if row["kind"] == kind and row["attempt_index"] == 1
                    ),
                    "unsupported": sum(
                        row["status"] == "unsupported" for row in selected if row["kind"] == kind
                    ),
                }
                for kind in ("reading", "edit")
            }
        recovery = [row for row in rows if row["phase"] == "recovery"]
        failed = {
            row["episode_id"]
            for row in recovery
            if row["attempt_index"] == 1 and not row["correct"]
        }
        recovered = {
            row["episode_id"] for row in recovery if row["episode_id"] in failed and row["correct"]
        }
        final_recovery = [row for row in recovery if row["attempt_index"] == 3]
        retained, same_session, changed_forms = [], [], []
        for episode in manifest["episodes"]:
            after = first[episode["episode_id"]]
            if episode.get("encounter") == "changed_form":
                before = first[episode["same_session_origin_id"]]
                gap = rows.index(after) - rows.index(before) - 1
                if gap != episode["intervening_attempts"]:
                    raise ValueError("changed-form encounter age changed")
                changed_forms.append(
                    {
                        "episode_id": episode["episode_id"],
                        "origin_episode_id": before["episode_id"],
                        "kind": after["kind"],
                        "initial_correct": before["correct"],
                        "changed_form_correct": after["correct"],
                        "intervening_attempts": gap,
                        "memory_slots": after["memory_used"],
                    }
                )
            if episode.get("encounter") == "exact_delayed":
                before = first[episode["same_session_origin_id"]]
                gap = rows.index(after) - rows.index(before) - 1
                if (
                    gap != episode["intervening_attempts"]
                    or before["observation_sha256"] != after["observation_sha256"]
                ):
                    raise ValueError("same-session exact repeat input or age changed")
                same_session.append(
                    {
                        "episode_id": episode["episode_id"],
                        "origin_episode_id": before["episode_id"],
                        "kind": after["kind"],
                        "initial_correct": before["correct"],
                        "delayed_correct": after["correct"],
                        "intervening_attempts": gap,
                        "initial_memory_slots": before["memory_used"],
                        "memory_slots": after["memory_used"],
                        "scope": "first response before its one correction vs delayed reset "
                        "observation",
                    }
                )
            if episode["phase"] != "retention":
                continue
            prior_route = "simple" if route == "simple" else "online"
            old_id = episode["previous_session_episode_id"]
            before = prior_outcomes[prior_route][old_id]
            earlier_gap = 80 - before["route_ordinal"] - 1
            current_gap = rows.index(after)
            retained.append(
                {
                    "episode_id": episode["episode_id"],
                    "prior_episode_id": old_id,
                    "prior_route": prior_route,
                    "prior_attempt_index": before["attempt_index"],
                    "initial_correct": before["correct"],
                    "delayed_correct": after["correct"],
                    "intervening_prior_attempts": earlier_gap,
                    "intervening_current_attempts": current_gap,
                    "intervening_attempts": earlier_gap + current_gap,
                    "memory_slots": after["memory_used"],
                    "same_raw_observation_as_prior": (
                        before["observation_sha256"] == after["observation_sha256"]
                    ),
                    "observation_scope": "current authored reset vs previous actual "
                    "last-attempt input",
                    "encounter": episode["encounter"],
                }
            )
        encounters = {}
        for name in ("first", "changed_form", "exact_delayed"):
            selected = [
                first[e["episode_id"]]
                for e in manifest["episodes"]
                if e["phase"] == "acquisition" and e["encounter"] == name
            ]
            encounters[name] = {
                kind: {
                    "episodes": sum(row["kind"] == kind for row in selected),
                    "first_correct": sum(row["correct"] for row in selected if row["kind"] == kind),
                }
                for kind in ("reading", "edit")
            }
            encounters[name]["changed_requirements"] = sum(
                bool(e.get("requirement_changed"))
                for e in manifest["episodes"]
                if e["phase"] == "acquisition" and e["encounter"] == name
            )
        fresh_strata = {}
        for stratum in ("instruction_construction", "structural_combination"):
            selected = [
                e
                for e in manifest["episodes"]
                if e["phase"] == "transfer" and e["fresh_stratum"] == stratum
            ]
            fresh_strata[stratum] = {}
            for dependency in ("independent", "dependent_project"):
                members = [
                    e
                    for e in selected
                    if (e.get("predecessor_episode_id") is not None)
                    == (dependency == "dependent_project")
                ]
                fresh_strata[stratum][dependency] = {
                    kind: {
                        "episodes": sum(e["kind"] == kind for e in members),
                        "first_correct": sum(
                            first[e["episode_id"]]["correct"] for e in members if e["kind"] == kind
                        ),
                        "unsupported": sum(
                            first[e["episode_id"]]["status"] == "unsupported"
                            for e in members
                            if e["kind"] == kind
                        ),
                    }
                    for kind in ("reading", "edit")
                }
        result[route] = {
            "phases": phases,
            "acquisition_encounters": encounters,
            "acquisition_scope": (
                "first before correction; pure changed forms retain identical requirements; "
                "exact delayed cases retain the original dated requirement"
            ),
            "fresh_strata": fresh_strata,
            "changed_form_encounters": changed_forms,
            "recovery": {
                "first_failures": len(failed),
                "first_correct": len(final_recovery) - len(failed),
                "later_correct": len(recovered),
                "final_attempt_correct": sum(row["correct"] for row in final_recovery),
                "final_recovered_after_first_failure": sum(
                    row["correct"] and row["episode_id"] in failed for row in final_recovery
                ),
                "policy": "three fixed attempts; private grade never controls retries",
                "tool_feedback_consumed": any(row["feedback_consumed"] for row in recovery),
                "feedback_consuming_attempts": sum(row["feedback_consumed"] for row in recovery),
                "identical_consecutive_proposals": sum(
                    row["proposal_signature"] == earlier["proposal_signature"]
                    for earlier, row in pairwise(recovery)
                    if row["episode_id"] == earlier["episode_id"]
                    and row["proposal_signature"] is not None
                ),
            },
            "retention": retained,
            "retained": sum(x["initial_correct"] and x["delayed_correct"] for x in retained),
            "forgotten": sum(x["initial_correct"] and not x["delayed_correct"] for x in retained),
            "improved": sum(not x["initial_correct"] and x["delayed_correct"] for x in retained),
            "same_session_retention": same_session,
            "same_session_retained": sum(
                x["initial_correct"] and x["delayed_correct"] for x in same_session
            ),
            "same_session_forgotten": sum(
                x["initial_correct"] and not x["delayed_correct"] for x in same_session
            ),
            "same_session_improved": sum(
                not x["initial_correct"] and x["delayed_correct"] for x in same_session
            ),
            "project_chains": [
                {
                    "episode_id": row["episode_id"],
                    "project_id": row["project_id"],
                    "predecessor_episode_id": row["predecessor_episode_id"],
                    "correct": row["correct"],
                    "status": row["status"],
                    "workspace_from_actual_predecessor": row["workspace_lifecycle"]
                    == "carry_actual_project",
                }
                for row in rows
                if row["project_id"] is not None and row["attempt_index"] == 1
            ],
        }
    return result


def _run(args, output, started, live):
    deadline = started + RUNNER_SECONDS

    def admit_work():
        if time.perf_counter() + CHECKPOINT_RESERVE_SECONDS >= deadline:
            raise TimeoutError("whole-session deadline reserve reached before new work")

    admit_work()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch

    from experiments.r4_integrated.persistent_language_action_baseline_restore_v6 import (
        restore_baseline,
    )
    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        feedback_from_receipt,
        grade_execution,
        proposal_from_record,
        qualified_launch,
        workspace_from_records,
    )
    from experiments.r4_integrated.persistent_language_action_initial_v7 import initialize
    from experiments.r4_integrated.persistent_session_checkpoint import (
        COUNTERS,
        PersistentSessionCheckpoint,
    )
    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.sequence_assembly_observation import RawObservation, ToolFeedback

    torch.set_num_threads(1)
    if args.manifest_sha != MANIFEST_SHA or sha(args.manifest) != args.manifest_sha:
        raise ValueError("frozen public manifest hash mismatch")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest["schema"] != "persistent-language-action-stream-v7"
        or len(manifest["documents"]) != 12
        or sum(len(bytes.fromhex(d["text_hex"])) for d in manifest["documents"]) > 960
        or [key for e in manifest["episodes"] for key in e["append_document_ids"]]
        != [d["document_id"] for d in manifest["documents"]]
    ):
        raise ValueError("v7 requires twelve whole public documents appended once in order")
    if (
        len(manifest["episodes"]) != 48
        or sum(e["attempt_limit"] for e in manifest["episodes"]) != 64
        or sum(e["attempt_limit"] for e in manifest["episodes"] if e["kind"] == "edit") != 44
        or sum(e.get("teacher") is not None for e in manifest["episodes"]) != 4
        or any(
            e.get("teacher") is not None and e["phase"] != "acquisition"
            for e in manifest["episodes"]
        )
        or manifest["counts"]["reader_teacher_symbols"] != 71
    ):
        raise ValueError("v7 requires the frozen48/64/44 stream and four fixed corrections")
    final_memory_slots = 864 + sum(
        len(bytes.fromhex(d["text_hex"])) + 1 for d in manifest["documents"]
    )
    with (output / "manifest.json").open("xb") as handle:
        handle.write(manifest_bytes)
    if sha(output / "manifest.json") != args.manifest_sha:
        raise ValueError("manifest changed during publication")
    source_tick = time.perf_counter()
    sources = {
        path.relative_to(ROOT).as_posix(): sha(path)
        for parent in (ROOT / "src/humanbrain", ROOT / "experiments/r4_integrated")
        for path in sorted(parent.glob("*.py"))
        if path.relative_to(ROOT).as_posix() not in SAVED_REVIEW_SOURCES
    }
    save(output / "sources.json", sources)
    source_capture_seconds = time.perf_counter() - source_tick
    checkpoint_tick = time.perf_counter()
    live["checkpoint_active"] = {"phase": "create", "started": checkpoint_tick, "work_before": {}}
    store = PersistentSessionCheckpoint.create(
        output / "checkpoints",
        source_root=ROOT,
        evidence_root=output,
        manifest_path=output / "manifest.json",
        manifest_sha256=args.manifest_sha,
        source_revision=args.source_revision,
        source_sha256=sources,
        session_id=args.session_id,
    )
    live["store"] = store
    checkpoint_create_seconds = time.perf_counter() - checkpoint_tick
    live["checkpoint_active"] = None
    admit_work()  # initialization intent already durable; no prior/model work preceded it
    initialization_tick = time.perf_counter()
    models, states, initialization = initialize(args.prior.resolve(), args.bank.resolve())
    if set(models) != {"online", "learning_disabled"}:
        raise ValueError("initializer must return the two matched persistent routes")
    initialization["elapsed_seconds"] = time.perf_counter() - initialization_tick
    save(output / "initialization.json", initialization)
    launch = qualified_launch(ROOT)
    save(output / "launch-record.json", launch.to_record())
    baseline_tick = time.perf_counter()
    baseline, restoration = restore_baseline(ROOT)
    restoration["elapsed_seconds"] = time.perf_counter() - baseline_tick
    save(output / "baseline-restoration.json", restoration)
    prior_outcomes, prior_outcome_receipt = load_prior_outcomes(
        args.prior.resolve().parent / "attempts.jsonl", initialization["prior_attempts_sha256"]
    )
    save(output / "prior-outcomes.json", prior_outcome_receipt)
    documents = {
        row["document_id"]: bytes.fromhex(row["text_hex"]) for row in manifest["documents"]
    }
    journals = {
        name: SessionJournal(output / f"{name}.jsonl")
        for name in ("attempts", "teachers", "ingests")
    }
    checkpoint_journal = SessionJournal(output / "checkpoints.jsonl")
    live["journals"] = [*journals.values(), checkpoint_journal]
    attempts = []
    projects = {route: {} for route in ROUTES}
    host = {
        "cursor": {"episode_index": 0, "route_index": 0, "attempt_index": 1, "append_done": False},
        "projects": projects,
        "current": None,
        "counters": {route: dict.fromkeys(COUNTERS, 0) for route in ROUTES},
        "elapsed_seconds": time.perf_counter() - started,
        "budget_seconds": RUNNER_SECONDS + AUDIT_SECONDS,
        "budget_mib": 2048,
        "initial_bindings": {
            name: sha(output / name)
            for name in (
                "initialization.json",
                "baseline-restoration.json",
                "manifest.json",
                "sources.json",
                "prior-outcomes.json",
                "launch-record.json",
            )
        },
    }
    checkpoint_timing = {name: 0.0 for name in ("create", "begin", "capture", "commit", "journal")}
    checkpoint_timing["create"] = checkpoint_create_seconds

    def checkpoint_event(phase, tick, before, result=None):
        elapsed = time.perf_counter() - tick
        checkpoint_timing[phase] += elapsed
        work = dict(store.work)
        record = {
            "schema": "persistent-language-action-checkpoint-work-v7",
            "phase": phase,
            "cursor": dict(host["cursor"]),
            "elapsed_seconds": elapsed,
            "session_elapsed_seconds": time.perf_counter() - started,
            "work_delta": {key: value - before.get(key, 0) for key, value in work.items()},
            "cumulative_work": work,
            "result": result,
        }
        write_tick = time.perf_counter()
        checkpoint_journal.append(record)
        checkpoint_timing["journal"] += time.perf_counter() - write_tick
        live["checkpoint_active"] = None

    # Creation precedes initializer; its already-measured work is reported here.
    write_tick = time.perf_counter()
    checkpoint_journal.append(
        {
            "schema": "persistent-language-action-checkpoint-work-v7",
            "phase": "create",
            "elapsed_seconds": checkpoint_create_seconds,
            "cumulative_work": dict(store.work),
            "config_sha256": store.config_sha256,
        }
    )
    checkpoint_timing["journal"] += time.perf_counter() - write_tick

    def begin_transaction(kind, *, observation_sha256=None):
        admit_work()
        tick, before = time.perf_counter(), dict(store.work)
        live["checkpoint_active"] = {"phase": "begin", "started": tick, "work_before": before}
        result = store.begin(kind, host["cursor"], observation_sha256=observation_sha256)
        checkpoint_event("begin", tick, before, {"intent_sha256": result})
        # A slow integrity pass never grants a renewed task budget.
        admit_work()

    def commit_boundary():
        tick, before = time.perf_counter(), dict(store.work)
        live["checkpoint_active"] = {"phase": "capture", "started": tick, "work_before": before}
        frontiers = store.capture_frontiers(
            {name: [f"{name}.jsonl"] for name in journals}, execution_root="executions"
        )
        checkpoint_event("capture", tick, before)
        host["elapsed_seconds"] = time.perf_counter() - started
        tick, before = time.perf_counter(), dict(store.work)
        live["checkpoint_active"] = {"phase": "commit", "started": tick, "work_before": before}
        result = store.commit(models, states, baseline, host, frontiers)
        live["last_safe"] = result
        checkpoint_event("commit", tick, before, result)
        if time.perf_counter() >= deadline:
            raise TimeoutError("session deadline reached after safe boundary publication")

    expected_memory_hashes = {}
    actual_executions = 0

    def hashes(route):
        if route == "simple":
            return (_digest(_plain(baseline.demonstrations)), _digest(_plain(baseline.documents)))
        return (_digest(models[route].state_dict()), _digest(vars(states[route])))

    initial_hashes = {
        route: dict(zip(("model", "memory"), hashes(route), strict=True)) for route in ROUTES
    }
    expected_memory_hashes.update(
        {route: values["memory"] for route, values in initial_hashes.items()}
    )

    def retained_reader_hash(model):
        return _digest(
            {
                key: value
                for key, value in model.reader.state_dict().items()
                if not key.startswith("binding.")
            }
        )

    retained_reader_initial = {
        route: retained_reader_hash(model) for route, model in models.items()
    }
    commit_boundary()
    try:
        for episode_index, episode in enumerate(manifest["episodes"]):
            for route_index, route in enumerate(ROUTES):
                admit_work()
                document_ids = episode["append_document_ids"]
                if document_ids:
                    begin_transaction("append")
                    ingest_tick = time.perf_counter()
                    raw_documents = [documents[key] for key in document_ids]
                    before_model, before_memory = hashes(route)
                    ingest_work = {}
                    source_feedback = None
                    if before_memory != expected_memory_hashes[route]:
                        raise ValueError("memory changed outside a declared append")
                    if route == "simple":
                        used_before = sum(len(doc) + 1 for doc in baseline.documents)
                        baseline.append_documents(raw_documents)
                        used_after = sum(len(doc) + 1 for doc in baseline.documents)
                        prefix_before = prefix_after = None
                    else:
                        model = models[route]
                        model._require_idle()
                        model.reader._require_idle()
                        model._check(ingest_work)
                        old = states[route]
                        model.reader.memory.check_state(old)
                        used_before = old.used
                        prefix_before = _digest(
                            {
                                key: getattr(old, key)[:used_before]
                                for key in ("keys", "sensory", "successor")
                            },
                            ingest_work,
                        )
                        grown, source_feedback = model.ingest_with_receipt(
                            raw_documents,
                            old,
                            writes_enabled=True,
                            learning_enabled=route == "online",
                        )
                        for key, value in source_feedback.work.items():
                            ingest_work[key] = ingest_work.get(key, 0) + value
                        model.reader.memory.check_state(grown)
                        used_after = grown.used
                        prefix_after = _digest(
                            {
                                key: getattr(grown, key)[:used_before]
                                for key in ("keys", "sensory", "successor")
                            },
                            ingest_work,
                        )
                        if prefix_before != prefix_after:
                            raise ValueError("append changed an existing factual slot")
                        states[route] = grown
                    after_model, after_memory = hashes(route)
                    expected_used = used_before + sum(len(doc) + 1 for doc in raw_documents)
                    if (
                        route != "online" and after_model != before_model
                    ) or used_after != expected_used:
                        raise ValueError(
                            "source admission changed disabled parameters or slot arithmetic"
                        )
                    expected_memory_hashes[route] = after_memory
                    journals["ingests"].append(
                        {
                            "schema": "persistent-language-action-ingest-v7",
                            "route": route,
                            "episode_id": episode["episode_id"],
                            "document_ids": document_ids,
                            "documents": [
                                {
                                    "document_id": key,
                                    "text_hex": documents[key].hex(),
                                    "sha256": hashlib.sha256(documents[key]).hexdigest(),
                                }
                                for key in document_ids
                            ],
                            "raw_bytes": sum(map(len, raw_documents)),
                            "eos": len(raw_documents),
                            "used_before": used_before,
                            "used_after": used_after,
                            "model_before_sha256": before_model,
                            "model_after_sha256": after_model,
                            "memory_before_sha256": before_memory,
                            "memory_after_sha256": after_memory,
                            "old_prefix_before_sha256": prefix_before,
                            "old_prefix_after_sha256": prefix_after,
                            "association_writes_enabled": True,
                            "source_next_byte_teacher_symbols": (
                                0 if source_feedback is None else source_feedback.symbol_count
                            ),
                            "source_learning_enabled": route == "online",
                            "source_feedback": None
                            if source_feedback is None
                            else {
                                "binding_updated": source_feedback.binding_updated,
                                "steps": _plain(source_feedback.source_steps),
                                "reader_model_before_sha256": source_feedback.model_before_sha256,
                                "reader_model_after_sha256": source_feedback.model_after_sha256,
                                "scope": (
                                    "causal whole-source bytes and EOS; no request/answer labels"
                                ),
                            },
                            "model_instance_id": id(
                                baseline if route == "simple" else models[route]
                            ),
                            "work": ingest_work,
                            "elapsed_seconds": time.perf_counter() - ingest_tick,
                        }
                    )
                    counts = host["counters"][route]
                    counts["ingest_documents"] += len(raw_documents)
                    counts["ingest_raw_bytes"] += sum(map(len, raw_documents))
                    counts["ingest_eos"] += len(raw_documents)
                    host["cursor"]["append_done"] = True
                    commit_boundary()
                project_id = episode.get("project_id")
                predecessor = episode.get("predecessor_episode_id")
                lifecycle = episode["workspace_lifecycle"]
                public_feedback = None
                if lifecycle == "carry_actual_project":
                    previous = projects[route].get(project_id)
                    if previous is None or previous["episode_id"] != predecessor:
                        raise ValueError("continued project must bind its actual route predecessor")
                    files = previous["files"]
                    saved_feedback = previous["feedback"]
                    public_feedback = (
                        None
                        if saved_feedback is None
                        else ToolFeedback(
                            **{
                                k: v
                                for k, v in saved_feedback.items()
                                if k not in ("stdout_hex", "stderr_hex")
                            },
                            stdout=bytes.fromhex(saved_feedback["stdout_hex"]),
                            stderr=bytes.fromhex(saved_feedback["stderr_hex"]),
                        )
                    )
                else:
                    files = episode["files"]
                teacher = episode.get("teacher")
                teaching = teacher is not None
                for index in range(1, episode["attempt_limit"] + 1):
                    admit_work()
                    tick = time.perf_counter()
                    request = bytes.fromhex(episode["request_hex"])
                    before_model, before_memory = hashes(route)
                    public = {
                        "request_hex": request.hex(),
                        "files": files,
                        "feedback": None
                        if public_feedback is None
                        else public_feedback.to_record(),
                    }
                    row = {
                        "schema": "persistent-language-action-attempt-v7",
                        "route": route,
                        "episode_id": episode["episode_id"],
                        "phase": episode["phase"],
                        "kind": episode["kind"],
                        "encounter": episode.get("encounter"),
                        "fresh_stratum": episode.get("fresh_stratum"),
                        "requirement_changed": bool(episode.get("requirement_changed")),
                        "attempt_index": index,
                        "project_id": project_id,
                        "predecessor_episode_id": predecessor,
                        "workspace_lifecycle": lifecycle,
                        "request_hex": request.hex(),
                        "observation": public,
                        "before_files": files,
                        "after_files": files,
                        "proposal": None,
                        "answer_hex": None,
                        "prediction": None,
                        "status": "completed",
                        "execution": None,
                        "correct": False,
                        "model_before_sha256": before_model,
                        "memory_before_sha256": before_memory,
                        "memory_used": (
                            sum(len(doc) + 1 for doc in baseline.documents)
                            if route == "simple"
                            else states[route].used
                        ),
                        "model_instance_id": id(baseline if route == "simple" else models[route]),
                        "feedback_consumed": False,
                        "work": {},
                    }
                    prediction, workspace, proposal, envelope = None, None, None, None
                    raw_observation = None
                    try:
                        workspace = (
                            workspace_from_records(files) if episode["kind"] == "edit" else None
                        )
                        raw_observation = RawObservation(request, workspace, public_feedback)
                        row["observation"] = raw_observation.to_record()
                    except (TypeError, ValueError) as error:
                        row["status"], row["error"] = "unsupported", str(error)
                    begin_tick = time.perf_counter()
                    begin_transaction(
                        "attempt",
                        observation_sha256=hashlib.sha256(encode(row["observation"])).hexdigest(),
                    )
                    row["included_checkpoint_begin_seconds"] = time.perf_counter() - begin_tick
                    if raw_observation is not None and episode["kind"] == "reading":
                        if route == "simple":
                            reading = baseline.predict(raw_observation, kind="reading")
                            row["prediction"] = reading
                            row["answer_hex"] = reading["answer_hex"]
                        else:
                            prediction = models[route].reader.predict_with_receipt(
                                raw_observation, states[route], 128
                            )
                            row["prediction"] = reader_summary(prediction)
                            row["answer_hex"] = prediction.generation.data.hex()
                            row["work"] = prediction.work
                    if episode["kind"] == "edit":
                        if raw_observation is not None:
                            if route == "simple":
                                result = baseline.predict(raw_observation, kind="edit")
                                row["prediction"] = result
                                proposal = proposal_from_record(result["proposal"], workspace)
                            else:
                                prediction = models[route].propose(raw_observation, states[route])
                                proposal = prediction.proposal
                                row["prediction"] = {
                                    "attempt_id": prediction.attempt_id,
                                    "proposal_sha256": prediction.proposal_sha256,
                                    "observation_sha256": prediction.observation_sha256,
                                    "model_sha256": prediction.model_sha256,
                                    "memory_sha256": prediction.memory_sha256,
                                    "payload": reader_summary(prediction.payload),
                                    "heads": {
                                        key: {
                                            "logits": value.logits.tolist(),
                                            "selected_index": value.selected_index,
                                        }
                                        for key, value in prediction.choices.items()
                                    },
                                }
                                row["work"] = prediction.work
                            row["proposal"] = proposal.to_record()
                        after, execution = execute_proposal(
                            workspace if workspace is not None else files,
                            proposal,
                            output / "executions" / route / episode["episode_id"] / str(index),
                            launch=launch,
                            deadline=deadline,
                        )
                        row["after_files"], row["execution"] = after, execution
                        actual_executions += 1
                        row["grade"] = grade_execution(
                            execution, episode["grade"], episode["files"]
                        )
                        row["correct"] = (
                            row["status"] != "unsupported" and row["grade"]["task_success"]
                        )
                        if prediction is not None:
                            envelope = {
                                "attempt_id": prediction.attempt_id,
                                "proposal_sha256": prediction.proposal_sha256,
                                "workspace_sha256": workspace.sha256,
                                "execution": execution,
                            }
                            if not teaching:
                                row["ack_work"] = models[route].finish_without_teaching(
                                    prediction, raw_observation, states[route], envelope
                                )
                    else:
                        row["correct"] = (
                            row["status"] != "unsupported"
                            and row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                        )
                        if prediction is not None and not teaching:
                            row["ack_work"] = models[route].reader.acknowledge(
                                prediction, states[route]
                            )
                    row["feedback_consumed"] = (
                        raw_observation is not None
                        and public_feedback is not None
                        and row["prediction"] is not None
                    )
                    row["model_after_sha256"], row["memory_after_sha256"] = hashes(route)
                    if (
                        row["model_after_sha256"] != before_model
                        or before_memory != expected_memory_hashes[route]
                        or row["memory_after_sha256"] != before_memory
                    ):
                        raise ValueError("free response/action changed slow or factual state")
                    row["observation_record_sha256"] = hashlib.sha256(
                        encode(row["observation"])
                    ).hexdigest()
                    row["proposal_signature"] = (
                        None
                        if row["proposal"] is None
                        else hashlib.sha256(
                            encode(
                                {
                                    key: value
                                    for key, value in row["proposal"].items()
                                    if key != "workspace_sha256"
                                }
                            )
                        ).hexdigest()
                    )
                    row["elapsed_seconds"] = time.perf_counter() - tick
                    attempt_sha = journals["attempts"].append(row)
                    # Full evidence is durable above. Summary memory keeps only
                    # scalar metric fields rather than every converted neural trace.
                    attempts.append(
                        {
                            key: row[key]
                            for key in (
                                "route",
                                "episode_id",
                                "phase",
                                "kind",
                                "attempt_index",
                                "correct",
                                "status",
                                "project_id",
                                "predecessor_episode_id",
                                "workspace_lifecycle",
                                "memory_used",
                                "feedback_consumed",
                            )
                        }
                        | {
                            "observation_sha256": row["observation_record_sha256"],
                            "proposal_signature": row["proposal_signature"],
                        }
                    )
                    if row["execution"] is not None and not row["execution"]["files_complete"]:
                        raise ValueError(
                            "incomplete actual workspace retained; cannot resume from partial files"
                        )
                    # Durable actual action/response precedes any target-bearing call.
                    if teaching:
                        teacher_tick = time.perf_counter()
                        teacher_row = {
                            "schema": "persistent-language-action-teacher-v7",
                            "route": route,
                            "episode_id": episode["episode_id"],
                            "attempt_sha256": attempt_sha,
                            "teacher": teacher,
                            "learning_enabled": route != "learning_disabled",
                            "model_before_sha256": row["model_after_sha256"],
                            "memory_before_sha256": row["memory_after_sha256"],
                            "reader_offers": 0,
                            "action_labels": 0,
                            "work": {},
                        }
                        if row["status"] == "unsupported":
                            raise ValueError("authored acquisition must fit actual interface")
                        if route == "simple":
                            teacher_row["publication"] = baseline.observe_teacher(
                                raw_observation, kind=episode["kind"], teacher=teacher
                            )
                        else:
                            if episode["kind"] == "reading":
                                feedback = models[route].reader.teach(
                                    raw_observation,
                                    bytes.fromhex(teacher["target_hex"]),
                                    states[route],
                                    0.01,
                                    prediction=prediction,
                                    learning_enabled=route == "online",
                                )
                                reader_feedback = feedback
                                teacher_row["publication"] = {}
                            else:
                                feedback = models[route].teach(
                                    prediction,
                                    raw_observation,
                                    states[route],
                                    proposal_from_record(teacher, workspace),
                                    envelope,
                                    learning_enabled=route == "online",
                                )
                                reader_feedback = feedback.reader_feedback
                                teacher_row["action_labels"] = len(feedback.action_losses)
                                teacher_row["publication"] = {"actions": feedback.action_updates}
                            teacher_row["work"] = feedback.work
                            if reader_feedback is not None:
                                teacher_row["reader_offers"] = reader_feedback.teacher_count
                                teacher_row["reader_feedback"] = {
                                    "total_loss": reader_feedback.total_loss,
                                    "mean_loss": reader_feedback.mean_loss,
                                    "actual_response_id": (
                                        prediction.attempt_id
                                        if episode["kind"] == "reading"
                                        else prediction.payload.attempt_id
                                    ),
                                    "steps": _plain(reader_feedback.teacher_steps),
                                    "scope": "causal teacher-prefix states separate from actual "
                                    "response; current target after selection; losses describe "
                                    "held prior potentials, not tuple-arbitrated output likelihood",
                                }
                                teacher_row["publication"].update(
                                    address=reader_feedback.address_updated,
                                    output=reader_feedback.output_updated,
                                    legacy_codec=reader_feedback.legacy_codec_updated,
                                    routing=reader_feedback.routing_updated,
                                    recognition=reader_feedback.recognition_updated,
                                    binding=reader_feedback.binding_updated,
                                )
                                teacher_row["publication_scope"] = (
                                    "output aliases binding contact change; inherited assemblies, "
                                    "recognition, router, B/C/U and codec held"
                                )
                        teacher_row["model_after_sha256"], teacher_row["memory_after_sha256"] = (
                            hashes(route)
                        )
                        teacher_row["elapsed_seconds"] = time.perf_counter() - teacher_tick
                        if teacher_row["memory_after_sha256"] != row["memory_after_sha256"] or (
                            route == "learning_disabled"
                            and teacher_row["model_after_sha256"]
                            != teacher_row["model_before_sha256"]
                        ):
                            raise ValueError(
                                "correction changed factual memory or disabled slow state"
                            )
                        journals["teachers"].append(teacher_row)
                        host["counters"][route]["teacher_occasions"] += 1
                        host["counters"][route]["reader_offers"] += teacher_row["reader_offers"]
                        host["counters"][route]["action_labels"] += teacher_row["action_labels"]
                    files = row["after_files"]
                    if row["execution"] is not None:
                        public_feedback = feedback_from_receipt(row["execution"])
                    actual = {
                        "episode_id": episode["episode_id"],
                        "files": files,
                        "feedback": None
                        if public_feedback is None
                        else public_feedback.to_record(),
                    }
                    host["current"] = {"route": route, **actual}
                    if project_id is not None:
                        projects[route][project_id] = actual
                    host["counters"][route]["attempts"] += 1
                    host["counters"][route]["executions"] += int(row["execution"] is not None)
                    if index < episode["attempt_limit"]:
                        host["cursor"] = {
                            "episode_index": episode_index,
                            "route_index": route_index,
                            "attempt_index": index + 1,
                            "append_done": True,
                        }
                    else:
                        host["cursor"] = {
                            "episode_index": episode_index + int(route_index == len(ROUTES) - 1),
                            "route_index": (route_index + 1) % len(ROUTES),
                            "attempt_index": 1,
                            "append_done": False,
                        }
                    commit_boundary()
        for journal in journals.values():
            journal.close()
        checkpoint_journal.close()
        retained_reader_final = {
            route: retained_reader_hash(model) for route, model in models.items()
        }
        if retained_reader_final != retained_reader_initial:
            raise ValueError("held inherited reader state changed during dendritic learning")
        if (
            hashes("learning_disabled")[0] != initial_hashes["learning_disabled"]["model"]
            or any(hashes(route)[1] != expected_memory_hashes[route] for route in ROUTES)
            or any(state.used != final_memory_slots for state in states.values())
            or sum(len(doc) + 1 for doc in baseline.documents) != final_memory_slots
        ):
            raise ValueError("disabled slow state or declared factual growth changed")
        if (
            {name: journal.rows for name, journal in journals.items()}
            != {
                "attempts": 192,
                "teachers": 12,
                "ingests": 36,
            }
            or actual_executions != 132
            or store.work.get("checkpoint_head_publications") != 229
        ):
            raise ValueError("completed stream does not match its fixed exposure")
        checkpoint_tick = time.perf_counter()
        checkpoint = {
            "schema": "persistent-language-action-state-v7",
            "routes": {route: model.snapshot(states[route]) for route, model in models.items()},
        }
        for route, model in models.items():
            checkpoint["routes"][route]["reader_experience"] = model.reader._active_experience
        final_model_experience = {
            route: snapshot["experience"] for route, snapshot in checkpoint["routes"].items()
        }

        def tensor_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            return 0

        checkpoint_tensor_bytes = tensor_bytes(checkpoint)
        with (output / "final-state.pt").open("xb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        del checkpoint
        checkpoint_seconds = time.perf_counter() - checkpoint_tick
        final_hash_tick = time.perf_counter()
        checkpoint_sha = sha(output / "final-state.pt")
        checkpoint_hash_seconds = time.perf_counter() - final_hash_tick
        report = {
            "schema": "persistent-language-action-report-v7",
            "status": "completed",
            "routes": list(ROUTES),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "initial_hashes": initial_hashes,
            "final_hashes": {
                route: dict(zip(("model", "memory"), hashes(route), strict=True))
                for route in ROUTES
            },
            "metrics": summarize(attempts, manifest, prior_outcomes),
            "counts": {k: v.rows for k, v in journals.items()},
            "journals": {
                **{k: v.metrics() for k, v in journals.items()},
                "checkpoints": checkpoint_journal.metrics(),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "clock": "time.perf_counter",
            "clock_metadata": {
                "implementation": time.get_clock_info("perf_counter").implementation,
                "monotonic": time.get_clock_info("perf_counter").monotonic,
                "adjustable": time.get_clock_info("perf_counter").adjustable,
                "resolution_seconds": time.get_clock_info("perf_counter").resolution,
            },
            "budget": {
                "complete_seconds": RUNNER_SECONDS + AUDIT_SECONDS,
                "runner_seconds": RUNNER_SECONDS,
                "reserved_audit_seconds": AUDIT_SECONDS,
                "before_work_checkpoint_reserve_seconds": CHECKPOINT_RESERVE_SECONDS,
                "peak_mib": 2048,
            },
            "scope": (
                "whole raw observation, paired dendritic source/answer learning, local actions "
                "and whole factual append; no old lesson replay"
            ),
            "prior_loads": 1,
            "models_constructed": 2,
            "teacher_free_recovery_evaluated": True,
            "final_checkpoint": {
                "path": "final-state.pt",
                "sha256": checkpoint_sha,
                "tensor_bytes": checkpoint_tensor_bytes,
                "snapshots": 2,
                "scope": "both model/tape/experience; continued prior online memory",
            },
            "baseline_limits": "same bounded raw observation; trigram retrieval and literal "
            "edit reuse",
            "retention_scope": (
                "actual v5 last-attempt outcome vs authored reset regression; observation "
                "equality explicit; same-session exact repeats compare pre-correction first "
                "response after59intervening attempts"
            ),
            "continuation": manifest["continuation"],
            "initial_memory_slots": 864,
            "final_memory_slots": {route: states[route].used for route in models},
            "final_assembly_occupied": {
                route: int(model.reader.assemblies.used.item()) for route, model in models.items()
            },
            "retained_reader_initial_sha256": retained_reader_initial,
            "retained_reader_final_sha256": retained_reader_final,
            "functional_prior_parity": False,
            "final_reader_experience": {
                route: model.reader._active_experience for route, model in models.items()
            },
            "reader_experience_scope": "persisted response custody counter, not neuronal weights",
            "final_model_experience": final_model_experience,
            "actual_executions": actual_executions,
            "checkpoint_store": {
                "directory": "checkpoints",
                "config_sha256": store.config_sha256,
                "last_safe": live["last_safe"],
                "head_sha256": sha(store.directory / "HEAD.json"),
                "phase_seconds": checkpoint_timing,
                "work": dict(store.work),
                "head_publications": store.work["checkpoint_head_publications"],
                "work_scope": "complete create/begin/capture/commit including returned tail; "
                "legacy snapshot validation passes named; internal operation counts are partial",
                "restores": 0,
                "automatic_replays": 0,
            },
            "ordinary_source_observation": {
                "documents_per_route": 12,
                "raw_bytes_per_route": final_memory_slots - 864 - 12,
                "eos_per_route": 12,
                "source_next_byte_teacher_symbols_per_neural_route": final_memory_slots - 864,
                "control": "association writes enabled in both neural routes; corrective "
                "slow learning disabled in control",
            },
            "initialization_seconds": initialization["elapsed_seconds"],
            "baseline_restoration_seconds": restoration["elapsed_seconds"],
            "prior_outcome_read_seconds": prior_outcome_receipt["elapsed_seconds"],
            "source_capture_seconds": source_capture_seconds,
            "checkpoint_serialization_seconds": checkpoint_seconds,
            "final_checkpoint_hash_seconds": checkpoint_hash_seconds,
            "timing_scope": "complete run includes source/receipt hashes, durable journal "
            "writes and initialization; per-attempt timing excludes its following journal write "
            "but includes its explicitly recorded checkpoint begin seconds; do not add that "
            "overlap again to checkpoint phase totals",
        }
        save(output / "report.json", report)
        if sources != {key: sha(ROOT / key) for key in sources}:
            raise ValueError("source bytes changed during session")
        if time.perf_counter() >= deadline:
            raise TimeoutError("session exceeded its fixed deadline including final serialization")
        final_inventory_tick = time.perf_counter()
        final_inventory = {
            path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        final_inventory_seconds = time.perf_counter() - final_inventory_tick
        if time.perf_counter() >= deadline:
            raise TimeoutError("session deadline reached during final evidence inventory")
        save(
            output / "completion.json",
            {
                "schema": "persistent-language-action-completion-v7",
                "status": "completed",
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "files": final_inventory,
                "final_inventory_seconds": final_inventory_seconds,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    except BaseException as error:
        # The outer owner records failure even when initialization did not finish.
        live["failed_phase_error"] = type(error).__name__
        raise


def run(args):
    """One launch only; failure preserves intents and never loads a safe prior."""
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    live = {}
    try:
        return _run(args, output, started, live)
    except BaseException as error:
        for journal in live.get("journals", []):
            if not journal.handle.closed:
                journal.close()
        tick = time.perf_counter()
        store = live.get("store")
        active = live.get("checkpoint_active")
        if active is not None:
            active = {
                "phase": active["phase"],
                "elapsed_seconds": time.perf_counter() - active["started"],
                "work_delta": None
                if store is None
                else {
                    key: value - active["work_before"].get(key, 0)
                    for key, value in store.work.items()
                },
            }
        head_path = output / "checkpoints" / "HEAD.json"
        head = None
        if head_path.exists():
            raw = head_path.read_bytes()
            head = {
                "record": json.loads(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes_read": len(raw),
            }
        # No expensive inspector/deserializer is started inside an exhausted budget.
        # This is an identity pointer, not a clean-boundary validation claim.
        save(
            output / "failure.json",
            {
                "schema": "persistent-language-action-failure-v7",
                "type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
                "last_safe_returned": live.get("last_safe"),
                "head_pointer": head,
                "checkpoint_work_at_failure": None if store is None else dict(store.work),
                "interrupted_checkpoint_phase": active,
                "failure_head_read_seconds": time.perf_counter() - tick,
                "failure_inspect_calls": 0,
                "journals": {
                    journal.path.name: journal.metrics() for journal in live.get("journals", [])
                },
                "dirty_intents_preserved": True,
                "resumed": False,
                "scope": "partial custody only; no completed audit claim",
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--prior",
        type=Path,
        default=ROOT / ".pytest-tmp/persistent-language-action-v5-run/final-state.pt",
    )
    parser.add_argument(
        "--bank", type=Path, default=ROOT / "benchmarks/persistent_language_action_v5.result.json"
    )
    args = parser.parse_args()
    if DRAFT or MANIFEST_SHA is None:
        raise SystemExit("DRAFT: source review and root release required")
    run(args)


if __name__ == "__main__":
    main()
