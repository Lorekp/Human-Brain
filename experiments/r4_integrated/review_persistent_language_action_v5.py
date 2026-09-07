"""Saved-only scalar/custody audit of the factorized persistent session.

No model, checkpoint decoder, task builder or policy is imported. Full routing
weight updates and lexical gates are outside the scalar receipt's audit scope.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_review_v4 as prior
from experiments.r4_integrated.persistent_language_action_review_v4 import (
    INPUT_WIDTH,
    TARGET_CURRENT,
    append_demonstration,
    assembly_winner,
    check_execution,
    check_heads,
    close,
    digest,
    file_sha,
    hash_value,
    hex_bytes,
    log_softmax,
    memory_winner,
    observation_hash,
    plain_digest,
    read_json,
    records,
    require,
    same_vector,
    similarity_query,
    vector,
)

DRAFT = True
ROUTES = ("online", "learning_disabled", "simple")
PHASES = ("acquisition", "transfer", "recovery", "retention")
PRIOR_BANK_SHA = "3122aed0c040ae086597cbdec6c0e2535a268cc62f7c72da05453ab88dd35466"
PRIOR_CHECKPOINT_SHA = "26cabfa5aa12ce362c51fcac2f57d5fdf05be6a5041a33f00acd47c05f618817"
PRIOR_REPORT_SHA = "5c63ac8bbbb2992a8dfe94d078801c06748208aa96d5a0ab0cd4711a661e32aa"
PRIOR_SOURCES_SHA = "4c8d21ad74b4c6a44d0dec34e4c903114f13b34974d3c900629e4ea17c4f4395"
PRIOR_REVISION = "07ab1047b19a3f39c82ca2e70aaa255f55a3923e"
PRIOR_HISTORY = ".pytest-tmp/persistent-language-action-v4-run"
PRIOR_HISTORY_SHA = {
    "manifest.json": "9d3399a1360ca9ddb4ba9361f06b70deb6063ddcf6461ed1aa414b0d54f4f191",
    "attempts.jsonl": "803bacd6949a9c695c30d30e8d3ff1f86c4203b0c019bf1df86919e91f6ee400",
    "teachers.jsonl": "60ebc2d149a9452211872b9507c078b71413cbd399b6c89449032aafe7b8aad7",
    "ingests.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
FEATURE_WIDTH = 1033304
MAX_SPARSE = 10992
MODEL_BYTES = 48764528
TAPE_BYTES = 1650696


def competition_work(count):
    if not count:
        return {}
    pairs = count * (count - 1) // 2
    return {
        "competition_candidates": count,
        "competition_pairs": pairs,
        "competition_threshold_layers": 2,
        "competition_comparator_neurons": pairs,
        "competition_winner_neurons": count,
        "competition_comparator_synapse_applications": 2 * pairs,
        "competition_winner_synapse_applications": 2 * pairs,
        "competition_comparator_subtractions": pairs,
        "competition_winner_accumulations": 2 * pairs,
        "competition_winner_bias_values": count,
        "competition_winner_threshold_subtractions": count,
        "competition_threshold_evaluations": pairs + count,
        "competition_identity_synapse_applications": count,
        "competition_identity_reductions": count - 1,
        "competition_winner_count_reductions": count - 1,
        "competition_index_values_constructed": 2 * pairs + count,
        "competition_receipt_tensor_bytes": 17 * count + 9 * pairs,
        "competition_named_allocated_tensor_bytes": 41 * count + 49 * pairs + 16,
        "competition_pair_index_bytes": 16 * pairs,
        "competition_score_gather_bytes": 16 * pairs,
    }


def check_routing(scores, selected, receipt, count):
    """Check all-owner f64 ranking, without rounding through f32."""
    vector(scores, count)
    require(0 <= count <= 512, "routing owner capacity")
    require(receipt["schema"] == "factorized-routing-decision-v1", "routing schema")
    require(receipt["selected_index"] == selected, "routing selected owner binding")
    hash_value(receipt["context_sha256"])
    energy = receipt["energy"]
    require(type(energy) in (int, float) and math.isfinite(energy), "routing energy")
    require(
        any(math.isclose(energy, x, abs_tol=1e-10) for x in (0, 0.5, 1)),
        "independent half-energy blocks",
    )
    baseline = receipt["selected_baseline"]
    require(
        type(baseline) in (int, float) and math.isfinite(baseline) and 0 <= baseline <= 1.000001,
        "fixed baseline range",
    )
    if count:
        require(
            type(selected) is int and selected == scores.index(max(scores)),
            "routing winner differs from lower-index f64 maximum",
        )
        require(math.isfinite(max(scores) - min(scores)), "routing pair-current overflow")
        close(receipt["selected_score"], scores[selected], "selected routing current")
    else:
        require(
            selected is None and baseline == energy == receipt["selected_score"] == 0,
            "empty routing bank fabricated an owner",
        )
    require(receipt["competition_work"] == competition_work(count), "f64 competition accounting")


def check_reader_work(
    work,
    used,
    decisions,
    request_length,
    *,
    teacher=False,
    staged=0,
    recruits=0,
    exhausted=0,
    candidates=0,
    owner_counts=(),
    routing_staged=0,
    routing_skipped=0,
    local_values=0,
):
    require(
        all(type(value) is int and value >= 0 for value in work.values()),
        "negative or noninteger reader work",
    )
    chunks = max(1, (request_length + 30) // 31)
    pairs = used * (1024 + (decisions - 1) * 32)
    expected = {
        "query_raw_bytes": request_length,
        "query_eos_symbols": chunks,
        "sequence_query_chunks": chunks,
        "output_address_steps": decisions,
        "output_address_synapse_products": used * decisions * 2048,
        "sequence_input_values": INPUT_WIDTH * decisions,
        "sequence_sensor_symbols": 3471,
        "sequence_sensor_flags": 12,
        "output_context_bank_bytes": used * 1152,
        "output_context_cursor_advances": used * 31,
        "output_legacy_current_potentials": used * 257,
        "prefix_cache_constructions": 1,
        "prefix_cache_advances": decisions - 1,
        "prefix_cache_feature_calls": decisions,
        "prefix_cache_rolling_drops": max(0, decisions - 32),
        "prefix_cache_lag_pairs": pairs,
        "prefix_cache_threshold_evaluations": 19 * pairs,
        "prefix_cache_input_applications": 54 * pairs,
        "prefix_cache_shift_copy_values": 2 * (decisions - 1) * used * 31 * 32,
        "prefix_cache_reused_activity_values": (decisions - 1) * used * 31 * 32,
        "prefix_cache_feature_cast_values": decisions * used * 1024,
        "prefix_cache_feature_scalings": decisions * used * 1024,
        "output_decisions": 0 if teacher else decisions,
        "assembly_evaluations": decisions if teacher else 0,
        "assembly_candidates": candidates if teacher else 0,
        "assembly_agreement_cells": candidates * INPUT_WIDTH if teacher else 0,
        "assembly_agreement_thresholds": 3 * candidates * INPUT_WIDTH if teacher else 0,
        "assembly_agreement_input_applications": 4 * candidates * INPUT_WIDTH if teacher else 0,
        "assembly_output_offers": decisions if teacher else 0,
        "assembly_staged_output_updates": staged,
        "assembly_staged_recruitments": recruits,
        "assembly_exhausted_offers": exhausted,
        "assembly_publications": int(staged > 0),
        "assembly_output_local_products": (decisions - exhausted) * 257 if teacher else 0,
        "output_query_address_enabled_groups": 0,
        "output_prefix_address_enabled_groups": 0,
        "output_legacy_codec_enabled_groups": 0,
        "output_enabled_groups": 0,
        "output_address_local_products": 0,
        "active_output_clamp_offers": 0,
        "router_evaluations": decisions,
        "router_candidates": sum(owner_counts),
        "router_outcome_offers": decisions if teacher else 0,
        "router_outcome_comparisons": decisions if teacher else 0,
        "router_skipped_outcomes": routing_skipped,
        "router_staged_updates": routing_staged,
        "router_publications": int(routing_staged > 0),
        "router_publication_copy_bytes": int(routing_staged > 0) * FEATURE_WIDTH * 8,
        "router_stage_copy_bytes": int(teacher) * FEATURE_WIDTH * 8,
        "router_local_products": local_values,
        "router_local_divisions": local_values,
        "router_local_additions": local_values,
        "router_local_error_values": decisions - routing_skipped if teacher else 0,
        "router_baseline_additions": sum(owner_counts)
        + (decisions - routing_skipped if teacher else 0),
        "factorized_selected_output_reads": 257 * sum(bool(n) for n in owner_counts),
        "factorized_fallback_output_reads": 257 * sum(not n for n in owner_counts),
        "factorized_output_argmaxes": decisions,
        "factorized_output_log_softmaxes": decisions,
        "factorized_context_builds": decisions,
        "factorized_activity_calls": sum(n + int(n > 0) for n in owner_counts),
    }
    require(len(owner_counts) == decisions, "per-event owner accounting")
    for count in owner_counts:
        for key, value in competition_work(count).items():
            name = "router_" + key
            expected[name] = expected.get(name, 0) + value
    products = work.get("router_weight_products", 0)
    require(
        14 * sum(owner_counts) <= products - local_values <= (MAX_SPARSE + 14) * sum(owner_counts),
        "sparse all-owner product bounds",
    )
    require(
        work.get("router_score_reductions", 0) == products - expected["router_baseline_additions"],
        "sparse score reductions",
    )
    fallback_count = min(decisions, 17) if request_length <= 31 else 0
    expected.update(
        sequence_conditioned_fallbacks=fallback_count,
        sequence_codec_fallbacks=decisions - fallback_count,
    )
    for key, value in expected.items():
        require(work.get(key, 0) == value, "assembly reader work mismatch: " + key)


def check_prediction(prediction, request_hex, used, bank):
    require(prediction["schema"] == "factorized-sequence-output-prediction-v1", "reader schema")
    require(prediction["max_bytes"] == 128, "free response capacity changed")
    for key in ("model_sha256", "memory_sha256", "trace_sha256"):
        hash_value(prediction[key])
    steps = prediction["steps"]
    require(1 <= len(steps) <= 129 and prediction["decisions"] == len(steps), "free step count")
    output = bytearray()
    for position, step in enumerate(steps):
        require(
            step["output_step"] == position and step["prefix_hex"] == bytes(output).hex(),
            "free generation used future or teacher bytes",
        )
        memory_winner(step["address_scores"], step["selected_index"], used)
        require(step["exact_context"] is None, "free path ran recruitment/exact-context gate")
        check_routing(step["assembly_scores"], step["assembly_index"], step["routing"], len(bank))
        logits = vector(step["logits"], 257)
        if bank:
            same_vector(logits, bank[step["assembly_index"]], "free output changed stored assembly")
        token = step["token"]
        require(type(token) is int and token == logits.index(max(logits)), "free byte selection")
        close(step["nll"], -log_softmax(logits)[token], "free byte NLL")
        if position < len(steps) - 1:
            require(token != 256 and position < 128, "generation continued after EOS/cap")
            output.append(token)
    last = steps[-1]["token"]
    if last == 256:
        require(prediction["stop_reason"] == "eos", "EOS status mismatch")
    else:
        require(len(steps) == 129 and prediction["stop_reason"] == "limit", "missing limit event")
    require(prediction["answer_hex"] == bytes(output).hex(), "generated raw bytes changed")
    if "tokens" in prediction:
        require(prediction["tokens"] == [step["token"] for step in steps], "token list drift")
    check_reader_work(
        prediction["work"],
        used,
        len(steps),
        len(hex_bytes(request_hex)),
        candidates=len(bank) * len(steps),
        owner_counts=[len(bank)] * len(steps),
    )
    return prediction["attempt_id"]


def check_teacher_feedback(
    feedback, target, used, actual_id, bank, input_ids, enabled, work, request_length
):
    require(feedback["actual_response_id"] == actual_id, "teacher lost actual-response custody")
    steps = feedback["steps"]
    require(len(steps) == len(target) + 1, "teacher sequence/EOS exposure")
    stage = [list(values) for values in bank]
    stage_inputs = list(input_ids)
    losses = []
    staged = recruits = exhausted = candidates = routing_staged = routing_skipped = local_values = 0
    owner_counts = []
    for position, (step, token) in enumerate(zip(steps, (*target, 256), strict=True)):
        require(
            step["position"] == position
            and step["prefix_hex"] == target[:position].hex()
            and step["target"] == token,
            "noncausal or altered teacher prefix/target",
        )
        memory_winner(step["address_scores"], step["memory_index"], used)
        count = len(stage)
        owner_counts.append(count)
        candidates += count
        check_routing(step["assembly_scores"], step["free_assembly_index"], step["routing"], count)
        assembly_winner(
            step["recruitment_scores"], step["recruitment_winner"], step["exact"], count
        )
        free = vector(step["free_currents"], 257)
        if count:
            same_vector(
                free, stage[step["free_assembly_index"]], "teacher changed pre-clamp readout"
            )
        emitted = step["emitted_byte"]
        require(
            type(emitted) is int and emitted == free.index(max(free)), "actual own-emission outcome"
        )
        outcome, routing = step["routing_outcome"], step["routing"]
        desired = 1.0 if emitted == token else -1.0
        close(outcome["desired"], desired, "direct outcome target")
        close(outcome["score_before"], routing["selected_score"], "pre-target routing score")
        close(outcome["error"], desired - routing["selected_score"], "own-potential error")
        close(outcome["energy"], routing["energy"], "own input energy")
        skipped = not count or routing["energy"] == 0
        require(
            outcome["skipped"] is skipped and outcome["staged"] is (enabled and not skipped),
            "routing enable/zero-energy policy",
        )
        close(
            outcome["score_after"],
            routing["selected_score"] if skipped else desired,
            "local potential clamp scalar consequence",
        )
        values = outcome["local_values"]
        require(
            type(values) is int and (14 <= values <= MAX_SPARSE + 14 if count else values == 0),
            "selected sparse row width",
        )
        routing_skipped += int(skipped)
        routing_staged += int(enabled and not skipped)
        local_values += 0 if skipped else values
        before = vector(step["currents_before"], 257)
        after = vector(step["currents_after"], 257)
        hash_value(step["input_sha256"])
        exact = step["exact"]
        recruit = not exact and count < 512
        full = not exact and count == 512
        selected = count if recruit else None if full else step["recruitment_winner"]
        require(
            step["selected_index"] == selected
            and step["recruited"] is recruit
            and step["exhausted"] is full
            and step["staged"] is (enabled and not full),
            "assembly recruitment/publication policy changed",
        )
        if exact:
            require(
                step["input_sha256"] == stage_inputs[selected], "exact context changed input hash"
            )
            same_vector(before, stage[selected], "exact assembly initial current mismatch")
        elif recruit:
            require(step["input_sha256"] not in stage_inputs, "recruited an already exact input")
        if full:
            same_vector(after, before, "exhausted novel offer modified output")
            exhausted += 1
        else:
            expected = [
                value + ((TARGET_CURRENT if index == token else 0.0) - value) * 1.0 / 1.0
                for index, value in enumerate(before)
            ]
            same_vector(after, expected, "local own-assembly current update differs")
        nll = -log_softmax(free)[token]
        close(step["nll"], nll, "pre-clamp teacher NLL")
        losses.append(nll)
        if enabled and not full:
            staged += 1
            if recruit:
                stage.append(list(after))
                stage_inputs.append(step["input_sha256"])
                recruits += 1
            else:
                stage[selected] = list(after)
    close(feedback["total_loss"], sum(losses), "teacher total NLL")
    close(feedback["mean_loss"], sum(losses) / len(losses), "teacher mean NLL")
    check_reader_work(
        work,
        used,
        len(steps),
        request_length,
        teacher=True,
        staged=staged,
        recruits=recruits,
        exhausted=exhausted,
        candidates=candidates,
        owner_counts=owner_counts,
        routing_staged=routing_staged,
        routing_skipped=routing_skipped,
        local_values=local_values,
    )
    if enabled:
        bank[:] = stage
        input_ids[:] = stage_inputs
    return {
        "offers": len(steps),
        "staged": staged,
        "recruits": recruits,
        "exhausted": exhausted,
        "routing_staged": routing_staged,
        "routing_skipped": routing_skipped,
    }


def check_initialization(initialization, report):
    expected = {
        "schema": "persistent-language-action-initial-custody-v5",
        "status": "initialized",
        "bank_sha256": PRIOR_BANK_SHA,
        "prior_sha256": PRIOR_CHECKPOINT_SHA,
        "prior_report_sha256": PRIOR_REPORT_SHA,
        "prior_sources_sha256": PRIOR_SOURCES_SHA,
        "prior_attempts_sha256": PRIOR_HISTORY_SHA["attempts.jsonl"],
        "prior_source_revision": PRIOR_REVISION,
        "prior_selected_route": "online",
        "loaded_payload_tensor_bytes": 82646896,
        "loaded_tensor_count": 38,
        "torch_load_calls": 1,
        "checkpoint_hash_passes": 3,
        "loaded_model_state_containers": 2,
        "loaded_tape_state_containers": 2,
        "discarded_prior_routes": ["learning_disabled"],
        "new_combined_model_constructions": 2,
        "new_reader_constructions": 2,
        "new_action_circuit_constructions": 2,
        "new_assembly_bank_constructions": 2,
        "new_router_constructions": 2,
        "old_model_constructions": 0,
        "promoted_tape_states": 2,
        "ingest_calls": 0,
        "checkpoint_free_replay_calls": 0,
        "prior_attempt_prediction_replays": 0,
        "restored_occupied_slots_per_route": 505,
        "preserved_assembly_owners_per_route": 235,
        "initial_experience_per_route": 186,
        "initial_reader_experience_per_route": 72,
        "prior_model_tensor_bytes_per_route": 40498096,
        "prior_tape_tensor_bytes_per_route": 825352,
        "initial_model_tensor_bytes": dict.fromkeys(ROUTES[:2], MODEL_BYTES),
        "initial_model_tensor_counts": dict.fromkeys(ROUTES[:2], 16),
        "initial_tape_tensor_bytes": dict.fromkeys(ROUTES[:2], TAPE_BYTES),
        "new_numeric_tensor_names": ["reader.router.theta"],
        "new_router_tensor_bytes_per_route": FEATURE_WIDTH * 8,
        "metadata_adapters": ["_extra_state", "reader._extra_state"],
        "added_metadata": ["reader.router._extra_state"],
        "model_modes": dict.fromkeys(ROUTES[:2], "learned"),
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES[:2], True),
    }
    for key, value in expected.items():
        require(initialization[key] == value, "initialization scope: " + key)
    for stem in ("bank", "prior", "prior_report", "prior_sources", "prior_attempts"):
        require(
            file_sha(initialization[stem + "_path"]) == initialization[stem + "_sha256"],
            "initialization input custody: " + stem,
        )
    bank = read_json(initialization["bank_path"])
    original = read_json(initialization["prior_report_path"])
    require(
        bank["status"] == "completed_and_independently_reviewed"
        and bank["source_revision"] == PRIOR_REVISION,
        "completed original continuation",
    )
    for name, identity in {
        "final-state.pt": PRIOR_CHECKPOINT_SHA,
        "report.json": PRIOR_REPORT_SHA,
        "sources.json": PRIOR_SOURCES_SHA,
        "attempts.jsonl": PRIOR_HISTORY_SHA["attempts.jsonl"],
    }.items():
        require(
            bank["evidence"][PRIOR_HISTORY + "/" + name] == identity,
            "original bank evidence binding",
        )
    require(
        initialization["checkpoint_file_bytes"]
        == Path(initialization["prior_path"]).stat().st_size,
        "loaded file size",
    )
    require(
        initialization["prior_memory_sha256"] == original["final_hashes"]["online"]["memory"],
        "original tape hash",
    )
    # The initializer seals all fifteen old numeric tensors in one aggregate.
    # Its frozen source/strict loader owns the name set; no per-tensor journal exists.
    hash_value(initialization["old_numeric_tensor_sha256"])
    for route in ROUTES[:2]:
        require(
            report["initial_hashes"][route]
            == {
                "model": initialization["initial_model_sha256"][route],
                "memory": initialization["initial_memory_sha256"][route],
            },
            "initial state chain",
        )
        views = initialization["derived_views"][route]
        require(
            views["owners"] == 235
            and 0 <= views["entries"] <= 235 * MAX_SPARSE
            and views["tensor_bytes"] == 8 * views["entries"] + 235 * 12,
            "derived owner-view storage",
        )
        hash_value(views["source_sha256"])
        hash_value(views["views_sha256"])
        require(
            initialization["initialization_work"][route]["factorized_candidate_builds"] == 235,
            "one canonical view per original owner",
        )
    require(
        report["initial_hashes"]["online"] == report["initial_hashes"]["learning_disabled"]
        and initialization["derived_views"]["online"]
        == initialization["derived_views"]["learning_disabled"],
        "matched initial controls",
    )
    require(
        report["final_checkpoint"]["tensor_bytes"] == 2 * (MODEL_BYTES + TAPE_BYTES),
        "complete final model/tape tensor payload",
    )
    return original


def check_history(root, restoration, outcome_receipt):
    """Read authenticated old bytes; reconstruct saved currents, never neuronal inputs."""
    demonstrations, documents, _, _ = prior.check_history(root, restoration["previous_restoration"])
    directory = root / PRIOR_HISTORY
    for name, identity in PRIOR_HISTORY_SHA.items():
        require(file_sha(directory / name) == identity, "fourth-session history identity")
    manifest = read_json(directory / "manifest.json")
    teachers = {}
    for row, _ in records(directory / "teachers.jsonl"):
        teachers[row["attempt_sha256"]] = row
    require(len(teachers) == 36, "complete historical correction stream")
    bank, input_ids, bindings = [], [], []
    outcomes = {route: {} for route in ROUTES}
    ordinals = dict.fromkeys(ROUTES, 0)
    for row, identity in records(directory / "attempts.jsonl"):
        route = row["route"]
        require(route in ROUTES, "historical route")
        outcomes[route][row["episode_id"]] = {
            "episode_id": row["episode_id"],
            "attempt_index": row["attempt_index"],
            "correct": row["correct"],
            "request_hex": row["request_hex"],
            "before_files": row["before_files"],
            "observation": row["observation"],
            "observation_sha256": digest(prior.json_bytes(row["observation"]) + b"\n"),
            "route_ordinal": ordinals[route],
        }
        ordinals[route] += 1
        teacher = teachers.get(identity)
        if teacher is None:
            continue
        if route == "online":
            for step in teacher["reader_feedback"]["steps"]:
                index = step["selected_index"]
                require(step["staged"] and not step["exhausted"], "banked original publication")
                after = vector(step["currents_after"], 257)
                hash_value(step["input_sha256"])
                if step["recruited"]:
                    require(index == len(bank), "historical owner order")
                    bank.append(list(after))
                    input_ids.append(step["input_sha256"])
                else:
                    require(input_ids[index] == step["input_sha256"], "historical exact owner")
                    bank[index] = list(after)
        elif route == "simple" and teacher["teacher"]["kind"] == "edit":
            append_demonstration(
                demonstrations, teacher["teacher"], row, similarity_query(row["observation"])
            )
            bindings.append({"episode_id": row["episode_id"], "attempt_sha256": identity})
    require(
        ordinals == dict.fromkeys(ROUTES, 72) and len(bank) == len(input_ids) == 235,
        "complete original owners and responses",
    )
    require(len(documents) == 18 and len(demonstrations) == 42, "restored reference scope")
    expected = {
        "schema": "persistent-language-action-baseline-restoration-v5",
        "fourth_session_bindings": bindings,
        "input_sha256": PRIOR_HISTORY_SHA,
        "retained_fourth_session_attempts": 6,
        "documents": 18,
        "edit_demonstrations": 42,
        "document_raw_bytes": 487,
        "document_eos": 18,
        "new_teacher_offers": 0,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "reconstructed_sessions": 4,
        "journal_rows_read": {"teachers.jsonl": 36, "attempts.jsonl": 216},
        "journal_bytes_read": {
            name: (directory / name).stat().st_size for name in ("teachers.jsonl", "attempts.jsonl")
        },
    }
    for key, value in expected.items():
        require(restoration[key] == value, "historical reference restoration: " + key)
    require(
        outcome_receipt["schema"] == "persistent-language-action-prior-outcomes-v5"
        and Path(outcome_receipt["path"]).resolve() == (directory / "attempts.jsonl").resolve()
        and outcome_receipt["sha256"] == PRIOR_HISTORY_SHA["attempts.jsonl"]
        and outcome_receipt["read_passes"] == 1
        and outcome_receipt["rows_read"] == 216
        and outcome_receipt["bytes_read"] == (directory / "attempts.jsonl").stat().st_size
        and outcome_receipt["route_attempts"] == ordinals,
        "prior outcome stream receipt",
    )
    return demonstrations, documents, manifest, outcomes, bank, input_ids


def check_metrics(report, compact, manifest, prior_outcomes):
    """Independently recount the four pressures and explicitly qualified repeats."""
    summaries = {}
    for route in ROUTES:
        rows = [row for row in compact if row["route"] == route]
        first = {row["episode_id"]: row for row in rows if row["attempt_index"] == 1}
        phases = {}
        for phase in PHASES:
            phases[phase] = {}
            for kind in ("reading", "edit"):
                selected = [r for r in rows if r["phase"] == phase and r["kind"] == kind]
                phases[phase][kind] = {
                    "episodes": sum(r["attempt_index"] == 1 for r in selected),
                    "attempts": len(selected),
                    "first_correct": sum(r["correct"] for r in selected if r["attempt_index"] == 1),
                    "unsupported": sum(r["status"] == "unsupported" for r in selected),
                }
        recovery = [r for r in rows if r["phase"] == "recovery"]
        failed = {r["episode_id"] for r in recovery if r["attempt_index"] == 1 and not r["correct"]}
        recovered = {
            r["episode_id"] for r in recovery if r["episode_id"] in failed and r["correct"]
        }
        final = [r for r in recovery if r["attempt_index"] == 3]
        retention, same_session = [], []
        for episode in manifest["episodes"]:
            after = first[episode["episode_id"]]
            if episode.get("encounter") == "exact_delayed":
                before = first[episode["same_session_origin_id"]]
                gap = rows.index(after) - rows.index(before) - 1
                require(
                    gap == 15 and before["observation_sha256"] == after["observation_sha256"],
                    "same-session repeat input/age",
                )
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
                        "scope": (
                            "first response before its one correction vs delayed reset observation"
                        ),
                    }
                )
            if episode["phase"] == "retention":
                previous_route = "simple" if route == "simple" else "online"
                old_id = episode["previous_session_episode_id"]
                before = prior_outcomes[previous_route][old_id]
                old_gap, current_gap = 71 - before["route_ordinal"], rows.index(after)
                retention.append(
                    {
                        "episode_id": episode["episode_id"],
                        "prior_episode_id": old_id,
                        "prior_route": previous_route,
                        "prior_attempt_index": before["attempt_index"],
                        "initial_correct": before["correct"],
                        "delayed_correct": after["correct"],
                        "intervening_prior_attempts": old_gap,
                        "intervening_current_attempts": current_gap,
                        "intervening_attempts": old_gap + current_gap,
                        "memory_slots": after["memory_used"],
                        "same_raw_observation_as_prior": before["observation_sha256"]
                        == after["observation_sha256"],
                        "observation_scope": (
                            "current authored reset vs previous actual last-attempt input"
                        ),
                        "encounter": episode["encounter"],
                    }
                )
        encounters = {}
        for encounter in ("corrected_first", "changed_form", "exact_delayed"):
            selected = [
                first[e["episode_id"]]
                for e in manifest["episodes"]
                if e["phase"] == "acquisition" and e["encounter"] == encounter
            ]
            encounters[encounter] = {
                kind: {
                    "episodes": sum(r["kind"] == kind for r in selected),
                    "first_correct": sum(r["correct"] for r in selected if r["kind"] == kind),
                }
                for kind in ("reading", "edit")
            }
        consecutive = sum(
            left["episode_id"] == right["episode_id"]
            and left["proposal_signature"] is not None
            and left["proposal_signature"] == right["proposal_signature"]
            for left, right in pairwise(recovery)
        )
        expected = {
            "phases": phases,
            "acquisition_encounters": encounters,
            "recovery": {
                "first_correct": sum(r["correct"] for r in recovery if r["attempt_index"] == 1),
                "first_failures": len(failed),
                "later_correct": len(recovered),
                "final_attempt_correct": sum(r["correct"] for r in final),
                "final_recovered_after_first_failure": sum(
                    r["correct"] and r["episode_id"] in failed for r in final
                ),
                "identical_consecutive_proposals": consecutive,
                "policy": "three fixed attempts; private grade never controls retries",
                "tool_feedback_consumed": any(r["feedback_consumed"] for r in recovery),
                "feedback_consuming_attempts": sum(r["feedback_consumed"] for r in recovery),
            },
            "retention": retention,
            "same_session_retention": same_session,
            "project_chains": [
                {
                    key: row[key]
                    for key in (
                        "episode_id",
                        "project_id",
                        "predecessor_episode_id",
                        "correct",
                        "status",
                    )
                }
                | {
                    "workspace_from_actual_predecessor": row["workspace_lifecycle"]
                    == "carry_actual_project"
                }
                for row in rows
                if row["project_id"] is not None and row["attempt_index"] == 1
            ],
        }
        for prefix, collection in (("", retention), ("same_session_", same_session)):
            expected[prefix + "retained"] = sum(
                x["initial_correct"] and x["delayed_correct"] for x in collection
            )
            expected[prefix + "forgotten"] = sum(
                x["initial_correct"] and not x["delayed_correct"] for x in collection
            )
            expected[prefix + "improved"] = sum(
                not x["initial_correct"] and x["delayed_correct"] for x in collection
            )
        require(
            report["metrics"][route] == expected, "independent outcomes, encounters and retention"
        )
        summaries[route] = expected
    return summaries


def check_ingest(row, episode, route, current, used, documents, public_documents, instances):
    raw = [public_documents[key] for key in episode["append_document_ids"]]
    expected_documents = [
        {"document_id": key, "text_hex": value.hex(), "sha256": digest(value)}
        for key, value in zip(episode["append_document_ids"], raw, strict=True)
    ]
    require(
        row["schema"] == "persistent-language-action-ingest-v5"
        and row["route"] == route
        and row["episode_id"] == episode["episode_id"]
        and row["document_ids"] == episode["append_document_ids"]
        and row["documents"] == expected_documents,
        "declared whole-document append",
    )
    require(
        row["used_before"] == used[route]
        and row["raw_bytes"] == sum(map(len, raw))
        and row["eos"] == len(raw)
        and row["used_after"] == used[route] + sum(len(value) + 1 for value in raw),
        "append symbol/capacity arithmetic",
    )
    require(
        row["model_before_sha256"] == row["model_after_sha256"] == current[route]["model"]
        and row["memory_before_sha256"] == current[route]["memory"]
        and row["association_writes_enabled"] is True
        and row["source_next_byte_teacher_symbols"] == 0,
        "append state/teacher boundary",
    )
    if route == "simple":
        require(
            row["old_prefix_before_sha256"] is None
            and row["old_prefix_after_sha256"] is None
            and row["work"] == {},
            "reference append invented neuronal work",
        )
        documents.extend({"bytes_hex": value.hex()} for value in raw)
        require(row["memory_after_sha256"] == plain_digest(documents), "reference document hash")
    else:
        hash_value(row["old_prefix_before_sha256"])
        require(
            row["old_prefix_before_sha256"] == row["old_prefix_after_sha256"],
            "append changed an earlier factual slot",
        )
        require(all(type(v) is int and v >= 0 for v in row["work"].values()), "append work")
    hash_value(row["memory_after_sha256"])
    if route in instances:
        require(instances[route] == row["model_instance_id"], "append replaced learner")
    else:
        require(row["model_instance_id"] not in instances.values(), "routes share append learner")
        instances[route] = row["model_instance_id"]
    current[route]["memory"], used[route] = row["memory_after_sha256"], row["used_after"]


def review(run, launch_path, manifest_sha):
    started = time.perf_counter()
    run = Path(run).resolve()
    root = Path(__file__).resolve().parents[2]
    completion = read_json(run / "completion.json")
    require(
        completion["schema"] == "persistent-language-action-completion-v5"
        and completion["status"] == "completed",
        "session incomplete",
    )
    required = {
        "manifest.json",
        "report.json",
        "attempts.jsonl",
        "teachers.jsonl",
        "ingests.jsonl",
        "initialization.json",
        "sources.json",
        "launch-record.json",
        "baseline-restoration.json",
        "prior-outcomes.json",
        "final-state.pt",
    }
    bindings = completion["files"]
    require(required <= set(bindings), "missing complete artifact binding")
    for relative, expected in bindings.items():
        path = (run / relative).resolve()
        require(
            path.is_relative_to(run) and file_sha(path) == expected,
            "saved artifact changed: " + relative,
        )
    require(bindings["manifest.json"] == manifest_sha, "manifest differs from prospective freeze")
    source_bindings = read_json(run / "sources.json")
    source_names = {name.replace("\\", "/") for name in source_bindings}
    expected_sources = {
        path.relative_to(root).as_posix()
        for directory in (root / "src/humanbrain", root / "experiments/r4_integrated")
        for path in directory.glob("*.py")
    }
    require(
        source_names == expected_sources and len(source_bindings) == len(source_names),
        "complete current Python source inventory",
    )
    require(
        "experiments\\r4_integrated\\persistent_language_action_v5.py" in source_bindings
        or "experiments/r4_integrated/persistent_language_action_v5.py" in source_bindings,
        "runner absent from source closure",
    )
    for relative, expected in source_bindings.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and file_sha(path) == expected, "run source changed")
    manifest, report = read_json(run / "manifest.json"), read_json(run / "report.json")
    require(
        manifest["schema"] == "persistent-language-action-development-data-v5"
        and report["schema"] == "persistent-language-action-report-v5"
        and report["status"] == "completed"
        and report["routes"] == list(ROUTES),
        "session schema",
    )
    episodes = manifest["episodes"]
    require(
        manifest_sha == "de182acf644b18fed25948c5c7886006253bee8618a4e6ce647508f624b86fca",
        "unselected public workload",
    )
    require(len(episodes) == len({e["episode_id"] for e in episodes}) == 64, "episode cardinality")
    require(
        [e["phase"] for e in episodes]
        == [
            phase
            for phase, count in zip(PHASES, (24, 16, 8, 16), strict=True)
            for _ in range(count)
        ],
        "phase ordering",
    )
    require(
        all(e["attempt_limit"] == (3 if e["phase"] == "recovery" else 1) for e in episodes),
        "retry schedule must be fixed independently of private grades",
    )
    require(
        [e["episode_id"] for e in episodes if e["teacher"] is not None]
        == [f"A{index:02d}-{kind}" for index in range(4) for kind in ("R", "E")],
        "only eight fixed correction occasions may be exposed",
    )
    require(
        len(manifest["documents"]) == 6
        and sum(len(hex_bytes(d["text_hex"])) for d in manifest["documents"]) == 353
        and [key for e in episodes for key in e["append_document_ids"]]
        == [d["document_id"] for d in manifest["documents"]]
        and manifest["capacity"]["memory_slots"] == 1024,
        "ordinary factual append schedule",
    )
    teachers = {}
    for row, _ in records(run / "teachers.jsonl", bindings["teachers.jsonl"]):
        require(row["attempt_sha256"] not in teachers, "duplicate teacher for actual attempt")
        teachers[row["attempt_sha256"]] = row
    require(len(teachers) == 24, "correction occasion count")
    initialization = read_json(run / "initialization.json")
    check_initialization(initialization, report)
    demonstrations, documents, old_manifest, prior_outcomes, old_bank, old_inputs = check_history(
        root, read_json(run / "baseline-restoration.json"), read_json(run / "prior-outcomes.json")
    )
    require(
        report["initial_hashes"]["simple"]
        == {"model": plain_digest(demonstrations), "memory": plain_digest(documents)},
        "reference initial state",
    )
    require(
        manifest["continuation"]["prior_documents"]
        == old_manifest["continuation"]["prior_documents"],
        "whole prior memory documents changed",
    )
    require(
        manifest["continuation"]["prior_final_state_sha256"] == PRIOR_CHECKPOINT_SHA
        and manifest["continuation"]["prior_bank_sha256"] == PRIOR_BANK_SHA
        and report["continuation"] == manifest["continuation"],
        "continuation identity",
    )
    old_episodes = {e["episode_id"]: e for e in old_manifest["episodes"]}
    for episode in episodes:
        if episode["phase"] == "retention":
            old = old_episodes[episode["previous_session_episode_id"]]
            require(
                all(episode[key] == old[key] for key in ("request_hex", "files", "grade", "kind")),
                "frozen retention observation or grading changed",
            )
    launch = read_json(run / "launch-record.json")
    for path, expected in launch["artifacts"].items():
        require(file_sha(path) == expected, "qualified runtime artifact changed")
    current = {route: dict(report["initial_hashes"][route]) for route in ROUTES}
    banks, input_ids, projects = (
        {route: [list(values) for values in old_bank] for route in ROUTES[:2]},
        {route: list(old_inputs) for route in ROUTES[:2]},
        {route: {} for route in ROUTES},
    )
    instances, responses, compact = {}, set(), []
    counts = {route: defaultdict(int) for route in ROUTES}
    used = dict.fromkeys(ROUTES, 505)
    public_documents = {d["document_id"]: hex_bytes(d["text_hex"]) for d in manifest["documents"]}
    ingest_stream = records(run / "ingests.jsonl", bindings["ingests.jsonl"])
    stream = records(run / "attempts.jsonl", bindings["attempts.jsonl"])
    for episode in episodes:
        for route in ROUTES:
            if episode["append_document_ids"]:
                ingest, _ = next(ingest_stream)
                check_ingest(
                    ingest, episode, route, current, used, documents, public_documents, instances
                )
                counts[route]["ingests"] += 1
                counts[route]["ingested_raw_bytes"] += ingest["raw_bytes"]
                counts[route]["ingested_eos"] += ingest["eos"]
            project, predecessor = episode.get("project_id"), episode.get("predecessor_episode_id")
            lifecycle = episode["workspace_lifecycle"]
            feedback = None
            if lifecycle == "carry_actual_project":
                require(project in projects[route], "project lacks prior actual state")
                previous = projects[route][project]
                require(previous["episode_id"] == predecessor, "project predecessor order")
                before, feedback = previous["files"], previous["feedback"]
            else:
                before = episode["files"]
            for index in range(1, episode["attempt_limit"] + 1):
                row, identity = next(stream)
                require(
                    row["schema"] == "persistent-language-action-attempt-v5"
                    and (row["route"], row["episode_id"], row["attempt_index"])
                    == (route, episode["episode_id"], index)
                    and all(
                        row[key] == episode[key]
                        for key in ("phase", "kind", "request_hex", "workspace_lifecycle")
                    )
                    and row["project_id"] == project
                    and row["predecessor_episode_id"] == predecessor,
                    "actual stream order or public episode changed",
                )
                require(row["before_files"] == before, "workspace reset or substitute predecessor")
                public = {
                    "schema": "sequence-assembly-raw-observation-v1",
                    "request_hex": episode["request_hex"],
                    "files": before if episode["kind"] == "edit" else None,
                    "feedback": feedback,
                }
                require(
                    row["status"] in ("completed", "unsupported"), "unrecorded execution failure"
                )
                if row["status"] == "completed":
                    require(
                        row["observation"] == public, "target-bearing or altered public observation"
                    )
                else:
                    require(
                        row["prediction"] is None
                        and row["answer_hex"] is None
                        and row["proposal"] is None,
                        "unsupported input silently produced a response",
                    )
                    require(
                        row["observation"]
                        == {key: public[key] for key in ("request_hex", "files", "feedback")},
                        "unsupported raw observation was dropped",
                    )
                require(
                    row["model_before_sha256"] == current[route]["model"]
                    and row["memory_before_sha256"] == current[route]["memory"]
                    and row["model_after_sha256"] == row["model_before_sha256"]
                    and row["memory_after_sha256"] == row["memory_before_sha256"]
                    and row["memory_used"] == used[route],
                    "free state mutation or broken persistence",
                )
                if route in instances:
                    require(
                        row["model_instance_id"] == instances[route], "learner instance replaced"
                    )
                else:
                    require(
                        row["model_instance_id"] not in instances.values(),
                        "routes share a live model",
                    )
                    instances[route] = row["model_instance_id"]
                actual_id = None
                if row["prediction"] is not None:
                    if route == "simple":
                        attribution = row["prediction"]["attribution"]
                        query = similarity_query(public)
                        require(
                            attribution["similarity_query_sha256"] == digest(query)
                            and attribution["similarity_query_bytes"] == len(query)
                            and attribution["observation_sha256"] == observation_hash(public)
                            and attribution["feedback_in_similarity"] is (feedback is not None),
                            "reference raw-input attribution",
                        )
                    else:
                        reader = (
                            row["prediction"]
                            if episode["kind"] == "reading"
                            else row["prediction"]["payload"]
                        )
                        actual_id = check_prediction(
                            reader, episode["request_hex"], used[route], banks[route]
                        )
                        require(actual_id not in responses, "duplicate actual response")
                        responses.add(actual_id)
                        require(
                            reader["memory_sha256"] == current[route]["memory"]
                            and reader["observation_sha256"] == observation_hash(public),
                            "reader observation/tape identity",
                        )
                        counts[route]["free_queries"] += 1
                        counts[route]["free_decisions"] += len(reader["steps"])
                        if episode["kind"] == "reading":
                            require(
                                row["answer_hex"] == reader["answer_hex"],
                                "reading byte substitution",
                            )
                        else:
                            counts[route]["action_proposals"] += 1
                            check_heads(row["prediction"], row["proposal"], before)
                            require(
                                row["prediction"]["model_sha256"] == current[route]["model"]
                                and row["prediction"]["memory_sha256"] == current[route]["memory"]
                                and row["prediction"]["observation_sha256"]
                                == observation_hash(public),
                                "action pending identity",
                            )
                require(
                    row["feedback_consumed"]
                    is (feedback is not None and row["prediction"] is not None),
                    "feedback consumption disclosure",
                )
                observed_hash = digest(prior.json_bytes(row["observation"]) + b"\n")
                require(row["observation_record_sha256"] == observed_hash, "raw observation hash")
                signature = (
                    None
                    if row["proposal"] is None
                    else digest(
                        prior.json_bytes(
                            {
                                key: value
                                for key, value in row["proposal"].items()
                                if key != "workspace_sha256"
                            }
                        )
                        + b"\n"
                    )
                )
                require(row["proposal_signature"] == signature, "semantic proposal signature")
                if episode["kind"] == "edit":
                    feedback = check_execution(row, episode, run, launch)
                    expected_correct = (
                        row["status"] != "unsupported" and row["grade"]["task_success"]
                    )
                    counts[route]["executions"] += 1
                else:
                    require(
                        row["execution"] is None
                        and row["after_files"] == before
                        and row["proposal"] is None,
                        "reading ran a hidden action",
                    )
                    expected_correct = (
                        row["status"] != "unsupported"
                        and row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                    )
                require(
                    type(row["correct"]) is bool and row["correct"] == expected_correct,
                    "actual behavioral grade",
                )
                teacher_row = teachers.pop(identity, None)
                if episode["teacher"] is None:
                    require(teacher_row is None, "unselected correction outside fixed occasions")
                    if actual_id is not None:
                        ack = row["ack_work"]
                        reader_ack = ack.get("reader", ack)
                        require(
                            reader_ack.get("sequence_acknowledgements") == 1
                            and not reader_ack.get("output_decisions", 0)
                            and not reader_ack.get("assembly_output_offers", 0)
                            and not reader_ack.get("router_outcome_offers", 0),
                            "teacher-free response not acknowledged",
                        )
                else:
                    require(teacher_row is not None, "missing predetermined correction")
                    check_teacher_row(
                        teacher_row,
                        row,
                        episode,
                        current,
                        banks,
                        input_ids,
                        demonstrations,
                        public,
                        actual_id,
                        counts,
                    )
                compact.append(
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
                        )
                    }
                    | {
                        "observation_sha256": observed_hash,
                        "proposal_signature": signature,
                        "feedback_consumed": row["feedback_consumed"],
                    }
                )
                before = row["after_files"]
                counts[route]["attempts"] += 1
            if project is not None:
                projects[route][project] = {
                    "episode_id": episode["episode_id"],
                    "files": before,
                    "feedback": feedback,
                }
    require(
        next(stream, None) is None and next(ingest_stream, None) is None and not teachers,
        "extra attempt, append or teacher rows",
    )
    require(
        report["counts"] == {"attempts": 240, "teachers": 24, "ingests": 18},
        "total exposure counts",
    )
    symbols = sum(
        len(hex_bytes(e["teacher"].get("target_hex", e["teacher"].get("replacement_hex")))) + 1
        for e in episodes
        if e["teacher"] is not None
    )
    require(
        symbols == manifest["counts"]["reader_teacher_symbols"] == 163, "offered bytes/EOS manifest"
    )
    for route in ROUTES:
        require(
            counts[route]["attempts"] == 80
            and counts[route]["executions"] == 52
            and counts[route]["teacher_rows"] == 8
            and counts[route]["ingests"] == 6
            and counts[route]["ingested_raw_bytes"] == 353
            and counts[route]["ingested_eos"] == 6,
            "per-route bounded exposure",
        )
        require(current[route] == report["final_hashes"][route], "final persistent state chain")
        if route != "simple":
            require(
                counts[route]["reader_offers"] == symbols and counts[route]["action_labels"] == 12,
                "neural target exposure changed",
            )
            require(
                report["final_reader_experience"][route] == 72 + counts[route]["free_queries"]
                and report["final_model_experience"][route]
                == 186 + counts[route]["action_proposals"],
                "persisted experience counts actual supported responses",
            )
    require(
        current["learning_disabled"]["model"]
        == report["initial_hashes"]["learning_disabled"]["model"],
        "disabled route learned",
    )
    require(
        report["initial_memory_slots"] == 505
        and report["final_memory_slots"] == dict.fromkeys(ROUTES[:2], 864)
        and used == dict.fromkeys(ROUTES, 864),
        "memory capacity/timeline changed",
    )
    require(len(documents) == 24 and len(demonstrations) == 46, "final reference state counts")
    require(
        report["actual_executions"] == 156 and report["functional_prior_parity"] is False,
        "complete action scope and prior function disclosure",
    )
    require(
        report["budget"]
        == {
            "complete_seconds": 900,
            "runner_seconds": 840,
            "reserved_audit_seconds": 60,
            "peak_mib": 2048,
        },
        "complete attempt resource envelope",
    )
    summaries = check_metrics(report, compact, manifest, prior_outcomes)
    require(
        report["final_assembly_occupied"] == {route: len(bank) for route, bank in banks.items()},
        "final assembly occupancy disagrees with actual staged recruitment",
    )
    require(
        report["retained_reader_initial_sha256"] == report["retained_reader_final_sha256"]
        and set(report["retained_reader_initial_sha256"]) == set(ROUTES[:2]),
        "inherited reader tensors changed during assembly learning",
    )
    for value in report["retained_reader_initial_sha256"].values():
        hash_value(value)
    monitor = read_json(launch_path)
    require(
        monitor["status"] == "passed"
        and monitor["elapsed_seconds"] <= 840
        and monitor["observed_tree_peak_sum_mib"] <= 2048,
        "workload monitor/resource bound",
    )
    command = monitor["command"]
    require(
        command[1:5]
        == ["-X", "utf8", "-m", "experiments.r4_integrated.persistent_language_action_v5"]
        and Path(command[command.index("--output") + 1]).resolve() == run
        and command[command.index("--manifest-sha") + 1] == manifest_sha,
        "workload command custody",
    )
    require(
        any(
            key.startswith(str(completion["pid"]) + "/")
            for key in monitor["observed_process_peaks_mib"]
        )
        and report["pid"] == completion["pid"]
        and report["parent_pid"] == completion["parent_pid"],
        "actual parent/owned-process custody",
    )
    require(
        report["final_checkpoint"]["sha256"] == bindings["final-state.pt"]
        and report["final_checkpoint"]["snapshots"] == 2,
        "final checkpoint custody",
    )
    require(
        report["prior_loads"] == 1
        and report["models_constructed"] == 2
        and report["clock"] == "time.perf_counter"
        and report["clock_metadata"]["monotonic"],
        "run accounting",
    )
    return {
        "schema": "persistent-language-action-saved-review-v5",
        "status": "passed",
        "summaries": summaries,
        "exposures": {key: dict(value) for key, value in counts.items()},
        "attempt_rows": len(compact),
        "teacher_rows": 24,
        "ingest_rows": 18,
        "assembly_occupied": {key: len(value) for key, value in banks.items()},
        "manifest_sha256": manifest_sha,
        "completion_sha256": file_sha(run / "completion.json"),
        "launch_sha256": file_sha(launch_path),
        "initial_memory_slots": 505,
        "final_memory_slots": 864,
        "elapsed_seconds": time.perf_counter() - started,
        "model_constructions": 0,
        "checkpoint_loads": 0,
        "actual_executions": 0,
        "checkpoint_scope": "byte hashes and source-bound tensor geometry; zero deserializations",
        "work_scope": (
            "exact address/prefix/competition/offer/publication counts; sparse work bounds "
            "and scalar consistency, without reconstructing all validation/hash traffic or theta"
        ),
        "scope": (
            "saved f64 routing selection and direct outcome scalars, separate recruitment/local "
            "outgoing algebra, causal prefixes, actual raw actions, factual append and custody; "
            "full theta updates and omitted lexical/receptive gates are not reconstructed"
        ),
    }


def check_teacher_row(
    teacher_row,
    row,
    episode,
    current,
    banks,
    input_ids,
    demonstrations,
    observation,
    actual_id,
    counts,
):
    route = row["route"]
    require(
        teacher_row["schema"] == "persistent-language-action-teacher-v5"
        and teacher_row["episode_id"] == episode["episode_id"]
        and teacher_row["route"] == route
        and teacher_row["teacher"] == episode["teacher"]
        and teacher_row["learning_enabled"] is (route != "learning_disabled"),
        "teacher identity/gate",
    )
    require(
        teacher_row["model_before_sha256"] == current[route]["model"]
        and teacher_row["memory_before_sha256"] == current[route]["memory"]
        and teacher_row["memory_after_sha256"] == current[route]["memory"],
        "teacher state chronology",
    )
    teacher = episode["teacher"]
    publication = teacher_row["publication"]
    if route == "simple":
        require(
            teacher_row["reader_offers"] == teacher_row["action_labels"] == 0
            and not teacher_row["work"],
            "reference invented neural work",
        )
        if episode["kind"] == "reading":
            require(
                publication
                == {"offered": True, "stored": False, "reason": "reading_targets_not_stored"},
                "reference reading answer retained",
            )
        else:
            query = similarity_query(observation)
            append_demonstration(demonstrations, teacher, row, query)
            require(
                publication
                == {
                    "offered": True,
                    "stored": True,
                    "reason": "supplied_literal_edit",
                    "demonstration_index": len(demonstrations) - 1,
                    "similarity_query_sha256": digest(query),
                },
                "reference demonstration custody",
            )
        require(
            teacher_row["model_after_sha256"] == plain_digest(demonstrations),
            "reference publication hash",
        )
    else:
        target = hex_bytes(
            teacher["target_hex"] if episode["kind"] == "reading" else teacher["replacement_hex"]
        )
        enabled = route == "online"
        work = teacher_row["work"]
        reader_work = work if episode["kind"] == "reading" else work["reader"]
        evidence = check_teacher_feedback(
            teacher_row["reader_feedback"],
            target,
            row["memory_used"],
            actual_id,
            banks[route],
            input_ids[route],
            enabled,
            reader_work,
            len(hex_bytes(episode["request_hex"])),
        )
        require(
            publication["address"] is False
            and publication["legacy_codec"] is False
            and publication["output"] is (evidence["staged"] > 0)
            and publication["routing"] is (evidence["routing_staged"] > 0),
            "forbidden old-group publication",
        )
        labels = 0
        if episode["kind"] == "edit":
            require(teacher["operation"] == "SPLICE", "correction schedule changed operation")
            labels = 3
            changed = publication["actions"]
            require(
                set(changed) == {"start", "end", "operation"}
                and all(
                    type(value) is bool and (enabled or not value) for value in changed.values()
                ),
                "action publication gate",
            )
            for head in changed:
                require(
                    work["action"].get(head + "_enabled_update_groups", 0) == int(enabled),
                    "action local teaching gate",
                )
        require(
            teacher_row["reader_offers"] == len(target) + 1
            and teacher_row["action_labels"] == labels,
            "teacher byte/action label accounting",
        )
        if not enabled:
            require(
                teacher_row["model_after_sha256"] == current[route]["model"], "disabled publication"
            )
        for key, value in evidence.items():
            counts[route]["assembly_" + key] += value
        counts[route]["reader_offers"] += len(target) + 1
        counts[route]["action_labels"] += labels
    hash_value(teacher_row["model_after_sha256"])
    current[route]["model"] = teacher_row["model_after_sha256"]
    counts[route]["teacher_rows"] += 1


def main():
    if DRAFT:
        raise RuntimeError("source review draft: saved audit not released")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    try:
        result = review(args.run, args.launch, args.manifest_sha)
    except BaseException as error:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": "persistent-language-action-saved-review-v5",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "manifest_sha256": args.manifest_sha,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        raise
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "attempt_rows": result["attempt_rows"]}))


if __name__ == "__main__":
    main()
