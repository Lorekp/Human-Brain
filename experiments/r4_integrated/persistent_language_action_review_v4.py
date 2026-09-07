"""Independent streaming review of saved assembly-session evidence.

Pure stdlib: no model, checkpoint deserializer, task builder, reference predictor
or execution helper. Saved potentials/scores are checked algebraically; omitted
receptive input gates are not claimed to have been independently replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from collections import defaultdict
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_review_v3 import (
    FIRST_HISTORY,
    FIRST_HISTORY_SHA,
    HISTORY,
    HISTORY_SHA,
    close,
    digest,
    files,
    log_softmax,
    plain_digest,
    require,
)

ROUTES = ("online", "learning_disabled", "simple")
PHASES = ("acquisition", "transfer", "recovery", "retention")
PRIOR_BANK_SHA = "39e899d8ae5fa5bb741edab2681ccf214d6f1883fd258f96d043a02b3609022e"
PRIOR_CHECKPOINT_SHA = "692634352ea323ff7737425a4d5c4e69cd3b39f1bcb9a6c85443a6612e59d03d"
PRIOR_HISTORY = ".pytest-tmp/persistent-language-action-v3-run"
PRIOR_HISTORY_SHA = {
    "manifest.json": "fc83513a6d6accee744c5d03188bc9cec75022db3cd00ff5c49f561745aa3056",
    "attempts.jsonl": "14bab3ac2908b3977736cadfa5886db2786a73dc48c504630213fa60eabe72d9",
    "teachers.jsonl": "b785cb87167e33233a81bb7125e3d2f8cdaa0e60b1009fe9ea100b385099f08f",
    "ingests.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
INPUT_WIDTH = 66093
TARGET_CURRENT = math.log(256)
FEEDBACK_KEYS = {
    "returncode",
    "stdout_hex",
    "stderr_hex",
    "accepted",
    "timeout",
    "capture_complete",
    "files_complete",
    "output_limit_exceeded",
}


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def records(path, expected_sha=None):
    """Bound each decoded row; retain no earlier row inside the stream."""
    overall = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for raw in handle:
            require(
                raw.endswith(b"\n") and len(raw) <= 8 * 1024 * 1024,
                "unterminated or oversized journal row",
            )
            overall.update(raw)
            row = json.loads(raw)
            require(isinstance(row, dict), "journal row must be a mapping")
            yield row, digest(raw)
    if expected_sha is not None:
        require(overall.hexdigest() == expected_sha, "journal bytes differ from completion")


def hex_bytes(value):
    require(isinstance(value, str), "raw bytes require hexadecimal text")
    result = bytes.fromhex(value)
    require(result.hex() == value, "noncanonical hexadecimal bytes")
    return result


def hash_value(value):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        "invalid identity digest",
    )


def vector(value, count, *, float32=False, integers=False):
    require(isinstance(value, list) and len(value) == count, "saved population shape mismatch")
    require(
        all(
            type(item) in ((int,) if integers else (int, float)) and math.isfinite(item)
            for item in value
        ),
        "nonfinite or mistyped population",
    )
    if float32:
        for item in value:
            require(
                struct.unpack("<f", struct.pack("<f", item))[0] == item,
                "saved address score is not an actual float32 value",
            )
    return value


def same_vector(actual, expected, message):
    require(len(actual) == len(expected), message)
    for left, right in zip(actual, expected, strict=True):
        close(left, right, message)


def memory_winner(scores, selected, used):
    vector(scores, used, float32=True)
    require(
        type(selected) is int and selected == scores.index(max(scores)),
        "memory winner differs from lower-index maximum",
    )


def assembly_winner(scores, selected, exact, count):
    vector(scores, count, integers=True)
    require(all(0 <= value <= INPUT_WIDTH for value in scores), "agreement score outside input")
    require(type(exact) is bool, "exact-context activity is not Boolean")
    if not count:
        require(selected is None and exact is False, "empty bank cannot choose an assembly")
    else:
        require(
            type(selected) is int and selected == scores.index(max(scores)),
            "assembly winner differs from lower-index maximum",
        )
        require(exact is (scores[selected] == INPUT_WIDTH), "exact-context threshold mismatch")


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
        "assembly_evaluations": decisions,
        "assembly_candidates": candidates,
        "assembly_agreement_cells": candidates * INPUT_WIDTH,
        "assembly_agreement_thresholds": 3 * candidates * INPUT_WIDTH,
        "assembly_agreement_input_applications": 4 * candidates * INPUT_WIDTH,
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
    }
    fallback_count = min(decisions, 17) if request_length <= 31 else 0
    expected.update(
        sequence_conditioned_fallbacks=fallback_count,
        sequence_codec_fallbacks=decisions - fallback_count,
    )
    for key, value in expected.items():
        require(work.get(key, 0) == value, "assembly reader work mismatch: " + key)


def check_prediction(prediction, request_hex, used, bank):
    require(prediction["schema"] == "sequence-assembly-output-prediction-v1", "reader schema")
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
        assembly_winner(
            step["assembly_scores"], step["assembly_index"], step["exact_context"], len(bank)
        )
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
    staged = recruits = exhausted = candidates = 0
    for position, (step, token) in enumerate(zip(steps, (*target, 256), strict=True)):
        require(
            step["position"] == position
            and step["prefix_hex"] == target[:position].hex()
            and step["target"] == token,
            "noncausal or altered teacher prefix/target",
        )
        memory_winner(step["address_scores"], step["memory_index"], used)
        count = len(stage)
        candidates += count
        assembly_winner(step["assembly_scores"], step["free_assembly_index"], step["exact"], count)
        free = vector(step["free_currents"], 257)
        if count:
            same_vector(
                free, stage[step["free_assembly_index"]], "teacher changed pre-clamp readout"
            )
        before = vector(step["currents_before"], 257)
        after = vector(step["currents_after"], 257)
        hash_value(step["input_sha256"])
        exact = step["exact"]
        recruit = not exact and count < 512
        full = not exact and count == 512
        selected = count if recruit else None if full else step["free_assembly_index"]
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
    )
    if enabled:
        bank[:] = stage
        input_ids[:] = stage_inputs
    return {"offers": len(steps), "staged": staged, "recruits": recruits, "exhausted": exhausted}


def append_demonstration(demonstrations, teacher, attempt, query):
    source = files(attempt["before_files"])[teacher["path"]]
    start, end = teacher["start"], teacher["end"]
    require(0 <= start <= end <= len(source), "demonstration outside actual source")
    demonstrations.append(
        {
            "query": {"bytes_hex": query.hex()},
            "old": {"bytes_hex": source[start:end].hex()},
            "replacement": {"bytes_hex": teacher["replacement_hex"]},
            "source_sha256": digest(source),
            "query_sha256": digest(query),
        }
    )


def check_history(root, restoration):
    """Restore only scalar/raw reference custody, never invoke its policy."""
    documents, demonstrations, all_bindings, manifests = [], [], [], []
    prior_retention = {}
    histories = (
        (FIRST_HISTORY, FIRST_HISTORY_SHA),
        (HISTORY, HISTORY_SHA),
        (PRIOR_HISTORY, PRIOR_HISTORY_SHA),
    )
    for directory, hashes in histories:
        base = root / directory
        require(file_sha(base / "manifest.json") == hashes["manifest.json"], "history manifest")
        manifest = read_json(base / "manifest.json")
        manifests.append(manifest)
        episodes = {row["episode_id"]: row for row in manifest["episodes"]}
        docs = {row["document_id"]: row["text_hex"] for row in manifest["documents"]}
        seen = []
        for row, _ in records(base / "ingests.jsonl", hashes["ingests.jsonl"]):
            if row["route"] == "simple":
                seen.extend(row["document_ids"])
        require(seen == list(docs), "historical reference ingestion order")
        documents.extend({"bytes_hex": docs[key]} for key in seen)
        actual = {}
        for row, identity in records(base / "attempts.jsonl", hashes["attempts.jsonl"]):
            if row["route"] == "simple" and row["kind"] == "edit" and row["attempt_index"] == 1:
                actual[identity] = {
                    key: row[key] for key in ("episode_id", "request_hex", "before_files", "phase")
                }
                require(row["execution"] is not None, "historical edit lacks execution")
            if directory == PRIOR_HISTORY and row["phase"] == "retention":
                prior_retention[(row["route"], row["episode_id"])] = bool(row["correct"])
        bindings = []
        for row, _ in records(base / "teachers.jsonl", hashes["teachers.jsonl"]):
            teacher = row["teacher"]
            if row["route"] != "simple" or teacher["kind"] != "edit":
                continue
            require(teacher["operation"] == "SPLICE", "historical nonliteral demonstration")
            attempt = actual[row["attempt_sha256"]]
            require(
                attempt["episode_id"] == row["episode_id"]
                and attempt["phase"] == "acquisition"
                and teacher == episodes[row["episode_id"]]["teacher"],
                "historical teacher custody",
            )
            append_demonstration(
                demonstrations, teacher, attempt, hex_bytes(attempt["request_hex"])
            )
            bindings.append(
                {"episode_id": row["episode_id"], "attempt_sha256": row["attempt_sha256"]}
            )
        require(len(bindings) == 12, "historical demonstration count")
        all_bindings.append(bindings)
    first = {
        "schema": "persistent-language-action-baseline-restoration-v2",
        "input_sha256": FIRST_HISTORY_SHA,
        "documents": 16,
        "edit_demonstrations": 12,
        "document_raw_bytes": 437,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "new_teacher_offers": 0,
        "bindings": all_bindings[0],
    }
    second = {
        "schema": "persistent-language-action-baseline-restoration-v3",
        "input_sha256": {"v1": FIRST_HISTORY_SHA, "v2": HISTORY_SHA},
        "previous_restoration": first,
        "documents": 18,
        "edit_demonstrations": 24,
        "document_raw_bytes": 487,
        "document_eos": 18,
        "new_teacher_offers": 0,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "reconstructed_sessions": 2,
        "second_session_bindings": all_bindings[1],
    }
    expected = {
        "schema": "persistent-language-action-baseline-restoration-v4",
        "input_sha256": {"v1": FIRST_HISTORY_SHA, "v2": HISTORY_SHA, "v3": PRIOR_HISTORY_SHA},
        "previous_restoration": second,
        "documents": 18,
        "edit_demonstrations": 36,
        "document_raw_bytes": 487,
        "document_eos": 18,
        "new_teacher_offers": 0,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "reconstructed_sessions": 3,
        "third_session_bindings": all_bindings[2],
    }
    require(restoration == expected, "restored reference custody changed")
    require(
        len(documents) == 18
        and len(demonstrations) == 36
        and sum(len(hex_bytes(d["bytes_hex"])) + 1 for d in documents) == 505,
        "restored reference scope",
    )
    return demonstrations, documents, manifests[-1], prior_retention


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def observation_hash(observation):
    return digest(b"humanbrain.sequence-assembly-observation.v1\0" + json_bytes(observation))


def workspace_hash(rows):
    entries = files(rows)
    raw = bytearray(b"humanbrain.sequence-assembly-workspace.v1\0")
    raw.extend(len(entries).to_bytes(4, "big"))
    for path, content in sorted(entries.items(), key=lambda item: item[0].encode()):
        for value in (path.encode(), content):
            raw.extend(len(value).to_bytes(8, "big"))
            raw.extend(value)
    return digest(raw)


def similarity_query(observation):
    query = hex_bytes(observation["request_hex"])
    feedback = observation["feedback"]
    if feedback is None:
        return query
    stdout, stderr = (hex_bytes(feedback[key]) for key in ("stdout_hex", "stderr_hex"))
    code = feedback["returncode"]
    return (
        query
        + b"\0"
        + len(stdout).to_bytes(4, "little")
        + stdout
        + len(stderr).to_bytes(4, "little")
        + stderr
        + bytes([code is not None])
        + (0 if code is None else code).to_bytes(8, "little", signed=True)
        + bytes(
            feedback[key]
            for key in (
                "accepted",
                "timeout",
                "capture_complete",
                "files_complete",
                "output_limit_exceeded",
            )
        )
    )


def check_heads(prediction, proposal, before):
    heads = prediction["heads"]
    require(set(heads) == {"start", "end", "operation"}, "action head set")
    boundaries = [
        (path, index)
        for path, raw in sorted(files(before).items(), key=lambda item: item[0].encode())
        for index in range(len(raw) + 1)
    ]
    start = heads["start"]
    vector(start["logits"], len(boundaries), float32=True)
    require(start["selected_index"] == start["logits"].index(max(start["logits"])), "start maximum")
    path, offset = boundaries[start["selected_index"]]
    end = heads["end"]
    vector(end["logits"], len(files(before)[path]) - offset + 1, float32=True)
    require(end["selected_index"] == end["logits"].index(max(end["logits"])), "end maximum")
    operation = heads["operation"]
    vector(operation["logits"], 2, float32=True)
    require(
        operation["selected_index"] == operation["logits"].index(max(operation["logits"])),
        "operation maximum",
    )
    require(
        proposal["operation"] == ("STOP", "SPLICE")[operation["selected_index"]]
        and proposal["path"] == path
        and proposal["start"] == offset
        and proposal["end"] == offset + end["selected_index"],
        "proposal differs from action winners",
    )
    require(
        proposal["replacement_hex"] == prediction["payload"]["answer_hex"]
        and proposal["stop_reason"] == prediction["payload"]["stop_reason"],
        "action payload substitution",
    )


def check_execution(row, episode, run, launch):
    execution, proposal = row["execution"], row["proposal"]
    require(execution["schema"] == "persistent-language-action-execution-v2", "execution schema")
    before = files(row["before_files"])
    proposed = dict(before)
    if proposal is not None:
        require(
            proposal["workspace_sha256"] == workspace_hash(row["before_files"]), "proposal prestate"
        )
        proposal_sha = digest(b"humanbrain.sequence-assembly-proposal.v1\0" + json_bytes(proposal))
        require(execution["proposal_sha256"] == proposal_sha, "executed proposal identity")
    else:
        require(
            not execution["accepted"] and execution["proposal_sha256"] is None,
            "missing accepted output",
        )
    if execution["accepted"]:
        require(proposal["operation"] in ("STOP", "SPLICE"), "unknown action")
        if proposal["operation"] == "SPLICE":
            path, start, end = (proposal[key] for key in ("path", "start", "end"))
            require(
                path in before
                and type(start) is int
                and type(end) is int
                and 0 <= start <= end <= len(before[path]),
                "accepted span invalid",
            )
            replacement = hex_bytes(proposal["replacement_hex"])
            require(
                len(replacement) <= 128 and proposal["stop_reason"] == "eos",
                "accepted payload invalid",
            )
            proposed[path] = before[path][:start] + replacement + before[path][end:]
    require(files(execution["proposed_files"]) == proposed, "host substituted or repaired edit")
    require(
        execution["before_files"] == row["before_files"]
        and execution["after_files"] == row["after_files"],
        "execution workspace binding",
    )
    require(execution["files_complete"], "partial workspace cannot continue")
    directory = Path(execution["workspace"]).resolve()
    require(directory.is_relative_to(run.resolve()), "execution workspace outside run")
    for path, raw in files(row["after_files"]).items():
        saved = (directory / path).resolve()
        require(
            saved.is_relative_to(directory) and saved.read_bytes() == raw,
            "actual after-file changed",
        )
    for name in ("stdout", "stderr"):
        raw = hex_bytes(execution[name + "_hex"])
        require(
            len(raw) <= 512 and digest(raw) == execution[name + "_sha256"], "captured byte binding"
        )
        count = execution[name + "_observed_bytes"]
        require(type(count) is int and count >= len(raw), "capture observed count")
        require(
            count == len(raw) or execution["output_limit_exceeded"], "silently omitted diagnostic"
        )
    require(
        not execution["output_limit_exceeded"] or not execution["capture_complete"],
        "capped complete capture",
    )
    require(
        execution["launch_record_sha256"] == digest(json_bytes(launch))
        and execution["bootstrap_sha256"]
        == "7213881ba88871e024789913aa742bba520cc79520b8a7b87292c0d3b2b745b9",
        "child bootstrap custody",
    )
    require(
        execution["child_timeout_seconds"] == 3 and execution["output_limit_per_stream"] == 512,
        "child execution limits",
    )
    command = execution["child_command"]
    require(
        len(command) == 8
        and command[1:7] == ["-I", "-S", "-X", "utf8", "-B", "-c"]
        and command[0] == launch["base_executable"]
        and digest(command[-1].encode()) == execution["bootstrap_sha256"],
        "child isolation flags",
    )
    grade = episode["grade"]
    original = files(episode["files"])
    protected = all(
        proposed.get(path) == original[path]
        and files(row["after_files"]).get(path) == original[path]
        for path in grade["protected_paths"]
    )
    behavior = (
        execution["returncode"] == grade["expected_returncode"]
        and execution["stdout_hex"] == grade["expected_stdout_hex"]
    )
    complete = (
        execution["capture_complete"]
        and not execution["timeout"]
        and not execution["output_limit_exceeded"]
        and not execution["capture_errors"]
    )
    expected_grade = {
        "task_success": bool(behavior and protected and complete),
        "behavior_correct": behavior,
        "protected_files_unchanged": protected,
        "complete_execution": complete,
    }
    require(row["grade"] == expected_grade, "private behavior/integrity grade changed")
    return {key: execution[key] for key in FEEDBACK_KEYS}


def check_initialization(initialization, report):
    require(
        initialization["schema"] == "persistent-language-action-initial-custody-v4"
        and initialization["status"] == "initialized",
        "initialization schema/status",
    )
    expected = {
        "bank_sha256": PRIOR_BANK_SHA,
        "prior_sha256": PRIOR_CHECKPOINT_SHA,
        "prior_selected_route": "online",
        "loaded_payload_tensor_bytes": 4275920,
        "loaded_tensor_count": 32,
        "torch_load_calls": 1,
        "checkpoint_hash_passes": 3,
        "bank_authenticated_read_passes": 2,
        "report_authenticated_read_passes": 2,
        "loaded_model_state_containers": 2,
        "loaded_tape_state_containers": 2,
        "discarded_prior_routes": ["learning_disabled"],
        "new_combined_model_constructions": 2,
        "new_reader_constructions": 2,
        "new_action_circuit_constructions": 2,
        "new_assembly_bank_constructions": 2,
        "old_model_constructions": 0,
        "restored_tape_states": 2,
        "ingest_calls": 0,
        "restored_occupied_slots_per_route": 505,
        "initial_experience_per_route": 138,
        "prior_model_tensor_bytes_per_route": 1312608,
        "prior_tape_tensor_bytes_per_route": 825352,
        "checkpoint_free_replay_calls": 0,
        "metadata_adapters": ["_extra_state", "reader._extra_state"],
        "added_metadata": ["reader.assemblies._extra_state", "actions._extra_state"],
        "model_modes": dict.fromkeys(ROUTES[:2], "learned"),
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES[:2], True),
    }
    for key, value in expected.items():
        require(initialization[key] == value, "initialization contract: " + key)
    for kind, value in (("bank", PRIOR_BANK_SHA), ("prior", PRIOR_CHECKPOINT_SHA)):
        require(file_sha(initialization[kind + "_path"]) == value, "prior custody bytes changed")
    bank = read_json(initialization["bank_path"])
    prior_report_path = Path(initialization["prior_report_path"])
    report_sha = file_sha(prior_report_path)
    require(
        report_sha
        == initialization["prior_report_sha256"]
        == bank["evidence"][PRIOR_HISTORY + "/report.json"],
        "prior report custody",
    )
    require(
        Path(initialization["prior_path"]).stat().st_size
        == initialization["checkpoint_file_bytes"],
        "prior container byte count",
    )
    require(
        bank["source_revision"] == initialization["prior_source_revision"], "prior source revision"
    )
    assembly_bytes = 512 * (INPUT_WIDTH + 257 * 8) + 8
    widths = {
        "start": 1298 * 577 + 1,
        "end": 1298 * 32 + 1024 + 1,
        "operation": 1298 * 64 + 513 * 129 + 3,
    }
    action_bytes = 4 * (widths["start"] + widths["end"] + 2 * widths["operation"] + 2)
    model_bytes = 1312608 - 66864 + action_bytes + assembly_bytes
    require(
        initialization["new_assembly_tensor_bytes_per_route"] == assembly_bytes
        and initialization["initial_model_tensor_bytes"] == dict.fromkeys(ROUTES[:2], model_bytes)
        and initialization["initial_model_tensor_counts"] == dict.fromkeys(ROUTES[:2], 15),
        "continued tensor geometry",
    )
    require(
        initialization["new_assembly_tensor_names"]
        == ["reader.assemblies.prototypes", "reader.assemblies.outgoing", "reader.assemblies.used"],
        "new numerical state contract",
    )
    indices = {
        "start": (
            [i * 32 + j for i in range(32) for j in range(32)]
            + [1298 * 32 + i * 513 + j for i in range(32) for j in range(129)]
            + [1298 * 545 + i * 32 + j for i in range(32) for j in range(32)]
            + [widths["start"] - 1]
        ),
        "end": (list(range(1024)) + list(range(1298 * 32, 1298 * 32 + 1024)) + [widths["end"] - 1]),
        "operation": (
            [
                block * 1298 * 32 + i * 32 + j
                for block in range(2)
                for i in range(32)
                for j in range(32)
            ]
            + [1298 * 64 + i * 129 + j for i in range(129) for j in range(17)]
            + list(range(widths["operation"] - 3, widths["operation"]))
        ),
    }
    for head, mapping in indices.items():
        receipt = initialization["action_coordinate_mappings"][head]
        shape = [2, widths[head]] if head == "operation" else [widths[head]]
        old_shape = [2, 4244] if head == "operation" else [6177 if head == "start" else 2049]
        multiplier = 2 if head == "operation" else 1
        require(
            receipt["old_shape"] == old_shape
            and receipt["new_shape"] == shape
            and receipt["mapped_values"] == len(mapping) * multiplier
            and receipt["zero_new_values"] == (widths[head] - len(mapping)) * multiplier
            and receipt["indices_sha256"]
            == digest(json.dumps(mapping, separators=(",", ":")).encode())
            and receipt["inverse_values_sha256"] == receipt["prior_values_sha256"],
            "independent legacy coordinate map receipt",
        )
    for route in ROUTES[:2]:
        require(
            report["initial_hashes"][route]
            == {
                "model": initialization["initial_model_sha256"][route],
                "memory": initialization["prior_memory_sha256"],
            },
            "initial state lineage",
        )
        require(
            initialization["initial_memory_sha256"][route] == initialization["prior_memory_sha256"],
            "whole continued memory mismatch",
        )
    require(
        report["initial_hashes"]["online"] == report["initial_hashes"]["learning_disabled"],
        "unmatched initial controls",
    )
    require(
        report["final_checkpoint"]["tensor_bytes"] == 2 * (model_bytes + 825352),
        "final continuation tensor payload",
    )
    return read_json(prior_report_path)


def check_metrics(report, compact, manifest, prior_retention):
    summaries = {}
    for route in ROUTES:
        rows = [row for row in compact if row["route"] == route]
        first = {row["episode_id"]: row for row in rows if row["attempt_index"] == 1}
        phases = {}
        for phase in PHASES:
            phases[phase] = {}
            for kind in ("reading", "edit"):
                selected = [row for row in rows if row["phase"] == phase and row["kind"] == kind]
                phases[phase][kind] = {
                    "episodes": sum(row["attempt_index"] == 1 for row in selected),
                    "attempts": len(selected),
                    "first_correct": sum(
                        row["correct"] for row in selected if row["attempt_index"] == 1
                    ),
                    "unsupported": sum(row["status"] == "unsupported" for row in selected),
                }
        recoveries = [row for row in rows if row["phase"] == "recovery"]
        failed = {
            row["episode_id"]
            for row in recoveries
            if row["attempt_index"] == 1 and not row["correct"]
        }
        recovered = {
            row["episode_id"]
            for row in recoveries
            if row["episode_id"] in failed and row["correct"]
        }
        final = [row for row in recoveries if row["attempt_index"] == 3]
        retention = []
        prior_ids = [key[1] for key in prior_retention if key[0] == "online"]
        for episode in manifest["episodes"]:
            if episode["phase"] == "retention":
                prior_route = "simple" if route == "simple" else "online"
                retention.append(
                    {
                        "episode_id": episode["episode_id"],
                        "prior_acquisition_id": episode["prior_episode_id"],
                        "prior_retention_id": episode["prior_retention_id"],
                        "prior_route": prior_route,
                        "initial_correct": prior_retention[
                            (prior_route, episode["prior_retention_id"])
                        ],
                        "delayed_correct": first[episode["episode_id"]]["correct"],
                        "intervening_prior_retention_attempts": (
                            len(prior_ids) - prior_ids.index(episode["prior_retention_id"]) - 1
                        ),
                        "intervening_current_attempts": rows.index(first[episode["episode_id"]]),
                        "intervening_attempts": (
                            len(prior_ids)
                            - prior_ids.index(episode["prior_retention_id"])
                            - 1
                            + rows.index(first[episode["episode_id"]])
                        ),
                        "memory_slots": first[episode["episode_id"]]["memory_used"],
                    }
                )
        expected = {
            "phases": phases,
            "recovery": {
                "first_failures": len(failed),
                "later_correct": len(recovered),
                "final_attempt_correct": sum(row["correct"] for row in final),
                "final_recovered_after_first_failure": sum(
                    row["correct"] and row["episode_id"] in failed for row in final
                ),
                "policy": "three fixed attempts; private grade never controls retries",
                "tool_feedback_consumed": True,
            },
            "retention": retention,
            "retained": sum(row["initial_correct"] and row["delayed_correct"] for row in retention),
            "forgotten": sum(
                row["initial_correct"] and not row["delayed_correct"] for row in retention
            ),
            "improved": sum(
                not row["initial_correct"] and row["delayed_correct"] for row in retention
            ),
            "acquisition_blocks": [
                sum(
                    first[e["episode_id"]]["correct"]
                    for e in manifest["episodes"][start : start + 12]
                )
                for start in (0, 12)
            ],
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
        require(report["metrics"][route] == expected, "independent phase/chain/retention totals")
        summaries[route] = expected
    return summaries


def review(run, launch_path, manifest_sha):
    started = time.perf_counter()
    run = Path(run).resolve()
    root = Path(__file__).resolve().parents[2]
    completion = read_json(run / "completion.json")
    require(
        completion["schema"] == "persistent-language-action-completion-v4"
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
    require(
        "experiments\\r4_integrated\\persistent_language_action_v4.py" in source_bindings
        or "experiments/r4_integrated/persistent_language_action_v4.py" in source_bindings,
        "runner absent from source closure",
    )
    for relative, expected in source_bindings.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and file_sha(path) == expected, "run source changed")
    manifest, report = read_json(run / "manifest.json"), read_json(run / "report.json")
    require(
        manifest["schema"] == "persistent-language-action-development-data-v4"
        and report["schema"] == "persistent-language-action-report-v4"
        and report["status"] == "completed"
        and report["routes"] == list(ROUTES),
        "session schema",
    )
    episodes = manifest["episodes"]
    require(len(episodes) == len({e["episode_id"] for e in episodes}) == 56, "episode cardinality")
    require(
        [e["phase"] for e in episodes]
        == [
            phase for phase, count in zip(PHASES, (24, 16, 8, 8), strict=True) for _ in range(count)
        ],
        "phase ordering",
    )
    require(
        all(e["attempt_limit"] == (3 if e["phase"] == "recovery" else 1) for e in episodes),
        "retry schedule must be fixed independently of private grades",
    )
    require(
        all((e["teacher"] is not None) == (index < 12) for index, e in enumerate(episodes)),
        "only first twelve predetermined corrections may be exposed",
    )
    require(
        not manifest["documents"]
        and all(not e["append_document_ids"] for e in episodes)
        and not list(records(run / "ingests.jsonl", bindings["ingests.jsonl"])),
        "unexpected memory write",
    )
    teachers = {}
    for row, _ in records(run / "teachers.jsonl", bindings["teachers.jsonl"]):
        require(row["attempt_sha256"] not in teachers, "duplicate teacher for actual attempt")
        teachers[row["attempt_sha256"]] = row
    require(len(teachers) == 36, "correction occasion count")
    initialization = read_json(run / "initialization.json")
    prior_report = check_initialization(initialization, report)
    demonstrations, documents, old_manifest, prior_retention = check_history(
        root, read_json(run / "baseline-restoration.json")
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
            old = old_episodes[episode["prior_retention_id"]]
            require(
                all(episode[key] == old[key] for key in ("request_hex", "files", "grade", "kind")),
                "frozen retention observation or grading changed",
            )
            for route in ROUTES:
                previous = next(
                    r
                    for r in prior_report["metrics"][route]["retention"]
                    if r["episode_id"] == episode["prior_retention_id"]
                )
                require(
                    previous["delayed_correct"] == prior_retention[(route, previous["episode_id"])],
                    "prior reported retention disagrees with actual row",
                )
    launch = read_json(run / "launch-record.json")
    for path, expected in launch["artifacts"].items():
        require(file_sha(path) == expected, "qualified runtime artifact changed")
    current = {route: dict(report["initial_hashes"][route]) for route in ROUTES}
    banks, input_ids, projects = (
        {route: [] for route in ROUTES[:2]},
        {route: [] for route in ROUTES[:2]},
        {route: {} for route in ROUTES},
    )
    instances, responses, compact = {}, set(), []
    counts = {route: defaultdict(int) for route in ROUTES}
    stream = records(run / "attempts.jsonl", bindings["attempts.jsonl"])
    for episode in episodes:
        for route in ROUTES:
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
                    row["schema"] == "persistent-language-action-attempt-v4"
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
                    and row["memory_used"] == 505,
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
                            reader, episode["request_hex"], 505, banks[route]
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
                    require(teacher_row is None, "unselected correction outside first twelve")
                    if actual_id is not None:
                        ack = row["ack_work"]
                        reader_ack = ack.get("reader", ack)
                        require(
                            reader_ack.get("sequence_acknowledgements") == 1
                            and not reader_ack.get("output_decisions", 0)
                            and not reader_ack.get("assembly_output_offers", 0),
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
                )
                before = row["after_files"]
                counts[route]["attempts"] += 1
            if project is not None:
                projects[route][project] = {
                    "episode_id": episode["episode_id"],
                    "files": before,
                    "feedback": feedback,
                }
    require(next(stream, None) is None and not teachers, "extra attempt or teacher rows")
    require(
        report["counts"] == {"attempts": 216, "teachers": 36, "ingests": 0}, "total exposure counts"
    )
    symbols = sum(
        len(hex_bytes(e["teacher"].get("target_hex", e["teacher"].get("replacement_hex")))) + 1
        for e in episodes[:12]
    )
    require(
        symbols == manifest["counts"]["reader_teacher_symbols"] == 235, "offered bytes/EOS manifest"
    )
    for route in ROUTES:
        require(
            counts[route]["attempts"] == 72
            and counts[route]["executions"] == 48
            and counts[route]["teacher_rows"] == 12,
            "per-route bounded exposure",
        )
        require(current[route] == report["final_hashes"][route], "final persistent state chain")
        if route != "simple":
            require(
                counts[route]["reader_offers"] == symbols and counts[route]["action_labels"] == 18,
                "neural target exposure changed",
            )
            require(
                current[route]["memory"] == initialization["prior_memory_sha256"],
                "prior tape modified",
            )
    require(
        current["learning_disabled"] == report["initial_hashes"]["learning_disabled"],
        "disabled route learned",
    )
    require(
        report["initial_memory_slots"] == 505
        and report["final_memory_slots"] == dict.fromkeys(ROUTES[:2], 505),
        "memory capacity/timeline changed",
    )
    summaries = check_metrics(report, compact, manifest, prior_retention)
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
        == ["-X", "utf8", "-m", "experiments.r4_integrated.persistent_language_action_v4"]
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
        "schema": "persistent-language-action-saved-review-v4",
        "status": "passed",
        "summaries": summaries,
        "exposures": {key: dict(value) for key, value in counts.items()},
        "attempt_rows": len(compact),
        "teacher_rows": 36,
        "ingest_rows": 0,
        "assembly_occupied": {key: len(value) for key, value in banks.items()},
        "manifest_sha256": manifest_sha,
        "completion_sha256": file_sha(run / "completion.json"),
        "launch_sha256": file_sha(launch_path),
        "initial_memory_slots": 505,
        "final_memory_slots": 505,
        "elapsed_seconds": time.perf_counter() - started,
        "model_constructions": 0,
        "checkpoint_loads": 0,
        "actual_executions": 0,
        "scope": (
            "saved scores/local outgoing algebra, causal prefixes, actual raw actions, "
            "project feedback chains and custody; omitted neuronal inputs/gates are not replayed"
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
        teacher_row["schema"] == "persistent-language-action-teacher-v4"
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
            505,
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
            and publication["output"] is (evidence["staged"] > 0),
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
                    "schema": "persistent-language-action-saved-review-v4",
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
