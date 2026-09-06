"""Independent saved-record review of the public persistent development session.

No learner, project scorer, checkpoint loader, or execution helper is imported.
This verifies recorded behavior and custody, not hidden neuronal arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

ROUTES = ("online", "learning_disabled", "simple")
PHASES = ("acquisition", "transfer", "recovery", "retention")
PRIOR_BANK_SHA = "39eca6edf662f0e580bfbacd931d512ab1eb009f6daadd459d046e6e78037799"
PRIOR_CHECKPOINT_SHA = "e0c581ab907310e846a256fbd868c667e93c8d6cf96626c60b1177d3cdf3da7b"
HISTORY = ".pytest-tmp/persistent-language-action-v1-run"
HISTORY_SHA = {
    "manifest.json": "8ff7e338fb7649d8b230fc54507a1265a406cadf3f183c9c6585cbfa47b62ef8",
    "attempts.jsonl": "24b4096773ed6fa761715a53688800094ce793445068101789b22b4143ebb71a",
    "teachers.jsonl": "98585a318401b5aa77c56e0eb6a2258c121355e634496ec8fc7d294ed5763a6f",
    "ingests.jsonl": "3f6e6bcaa2bbd0054d57a23a982cf1406e21fa507fc36ecb52918aa973abb147",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def records(path):
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            require(len(raw) <= 4 * 1024 * 1024, "journal row exceeds review bound")
            rows.append((json.loads(raw), digest(raw)))
    return rows


def files(rows):
    result = {row["path"]: bytes.fromhex(row["content_hex"]) for row in rows}
    require(len(result) == len(rows), "duplicate file in saved observation")
    return result


def close(actual, expected, message):
    require(type(actual) in (int, float) and math.isfinite(actual), message)
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), message)


def log_softmax(values):
    require(
        values and all(type(v) in (int, float) and math.isfinite(v) for v in values),
        "nonfinite or empty saved currents",
    )
    maximum = max(values)
    offset = math.log(sum(math.exp(v - maximum) for v in values))
    return [(v - maximum) - offset for v in values]


def check_reader_work(work, used, decisions, query_bytes, *, teacher=False, enabled=False):
    require(all(type(v) is int and v >= 0 for v in work.values()), "invalid reader work counter")
    populations = used * decisions if teacher else decisions
    expected = {
        "query_raw_bytes": query_bytes,
        "query_eos_symbols": 1,
        "output_query_encodings": 1,
        "output_prefix_encodings": decisions,
        "output_prefix_symbols": decisions * (decisions + 1) // 2,
        "output_query_feature_slots": used,
        "output_prefix_feature_slots": used * decisions,
        "output_context_bank_values": used * 1152,
        "output_context_bank_bytes": used * 1152,
        "output_context_cursor_advances": used * 31,
        "output_legacy_current_slots": used,
        "output_legacy_current_potentials": used * 257,
        "output_address_steps": decisions,
        "output_address_synapse_products": used * decisions * 2048,
        "output_conditional_populations": populations,
        "output_synapse_products": populations * 257 * 292,
        "output_softmaxes": populations,
        "output_decisions": 0 if teacher else decisions,
        "output_target_offers": decisions if teacher else 0,
        "teacher_offers": decisions if teacher else 0,
        "output_legacy_codec_enabled_groups": 0,
        "output_query_address_enabled_groups": int(teacher and enabled),
        "output_prefix_address_enabled_groups": int(teacher and enabled),
        "output_enabled_groups": int(teacher and enabled),
        "output_staged_update_steps": decisions if teacher and enabled else 0,
        "output_staging_tensor_bytes": 608544 if teacher else 0,
        "output_local_synapse_products": populations * 257 * 292 if teacher else 0,
        "output_local_slot_reductions": (used - 1) * decisions * 257 * 292 if teacher else 0,
        "output_local_denominator_values": used * decisions * 292 if teacher else 0,
        "output_address_local_products": used * decisions * 2048 if teacher else 0,
        "output_held_context_gathers": 0 if teacher else decisions * 1152,
        "output_held_current_gathers": 0 if teacher else decisions * 257,
    }
    for key, value in expected.items():
        require(work.get(key) == value, "conditioned reader work: " + key)


def check_prediction(row, request_hex, used):
    require(
        set(row) == {"schema", "answer_hex", "stop_reason", "decisions", "tokens", "steps", "work"},
        "unexpected prediction schema",
    )
    require(row["schema"] == "request-conditioned-output-prediction-v1", "old reader policy")
    steps, tokens = row["steps"], row["tokens"]
    require(
        1 <= len(steps) <= 17 and row["decisions"] == len(steps) == len(tokens),
        "unbounded output decisions",
    )
    prefix = bytearray()
    for index, step in enumerate(steps):
        require(
            set(step)
            == {
                "output_step",
                "prefix_hex",
                "token",
                "selected_index",
                "address_scores",
                "competition_work",
                "prior_log_belief",
                "logits",
                "probabilities",
                "nll",
            },
            "unexpected output step fields",
        )
        require(
            step["output_step"] == index and step["prefix_hex"] == bytes(prefix).hex(),
            "output used a teacher or future prefix",
        )
        scores, prior = step["address_scores"], step["prior_log_belief"]
        require(len(scores) == len(prior) == used, "address candidate count changed")
        expected_prior = log_softmax(scores)
        for actual, expected in zip(prior, expected_prior, strict=True):
            close(actual, expected, "saved address prior mismatch")
        selected = step["selected_index"]
        require(
            type(selected) is int and selected == scores.index(max(scores)),
            "saved competition did not select the first maximum",
        )
        competition = step["competition_work"]
        pairs = used * (used - 1) // 2
        require(
            all(type(v) is int and v >= 0 for v in competition.values()), "competition work type"
        )
        for key, value in {
            "competition_candidates": used,
            "competition_pairs": pairs,
            "competition_comparator_neurons": pairs,
            "competition_winner_neurons": used,
            "competition_threshold_evaluations": pairs + used,
        }.items():
            require(competition.get(key) == value, "per-step competition work mismatch")
        require(
            len(step["logits"]) == 1 and len(step["logits"][0]) == 257,
            "free conditional output must be one population",
        )
        logits = step["logits"][0]
        conditional = log_softmax(logits)
        require(len(step["probabilities"]) == 257, "output alphabet changed")
        for actual, expected in zip(step["probabilities"], conditional, strict=True):
            close(actual, math.exp(expected), "conditional probability mismatch")
        token = step["token"]
        require(
            type(token) is int and token == logits.index(max(logits)) == tokens[index],
            "generated byte is not actual first-maximum output",
        )
        close(step["nll"], -conditional[token], "free output NLL mismatch")
        if index < len(steps) - 1:
            require(token < 256, "output continued after EOS")
            prefix.append(token)
    require(
        row["answer_hex"] == bytes(prefix).hex(), "generation differs from actual causal prefix"
    )
    require(
        (row["stop_reason"] == "eos" and tokens[-1] == 256)
        or (row["stop_reason"] == "limit" and len(steps) == 17 and tokens[-1] < 256),
        "unexplained generation ending",
    )
    check_reader_work(row["work"], used, len(steps), len(bytes.fromhex(request_hex)))


def check_teacher_feedback(feedback, target, used):
    require(
        set(feedback) == {"total_loss", "mean_loss", "steps", "scope"},
        "unexpected corrective summary",
    )
    require(
        feedback["scope"] == "actual mixtures and current teacher responsibility; logits omitted",
        "undeclared corrective arithmetic evidence",
    )
    targets = (*target, 256)
    require(len(feedback["steps"]) == len(targets), "corrective trace offer count")
    losses = []
    for position, (step, token) in enumerate(zip(feedback["steps"], targets, strict=True)):
        require(
            set(step)
            == {
                "output_step",
                "prefix_hex",
                "token",
                "nll",
                "probabilities",
                "prior_log_belief",
                "posterior_log_belief",
            },
            "unexpected corrective step",
        )
        require(
            step["output_step"] == position
            and step["token"] == token
            and step["prefix_hex"] == target[:position].hex(),
            "teacher prefix includes current/future label or differs from offered history",
        )
        mixture = step["probabilities"]
        require(
            len(mixture) == 257
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in mixture),
            "invalid teacher mixture",
        )
        close(sum(mixture), 1.0, "teacher mixture not normalized")
        loss = step["nll"]
        require(
            type(loss) in (int, float) and math.isfinite(loss) and loss >= -1e-10,
            "invalid corrective loss",
        )
        close(mixture[token], math.exp(-loss), "teacher target loss disagrees with saved mixture")
        for name in ("prior_log_belief", "posterior_log_belief"):
            values = step[name]
            require(
                len(values) == used
                and all(
                    type(v) in (int, float) and math.isfinite(v) and v <= 1e-10 for v in values
                ),
                "invalid all-address corrective belief",
            )
            close(sum(math.exp(v) for v in values), 1.0, "teacher address belief not normalized")
        # These posterior/current-mixture records do not recover other candidate output
        # columns or the fixed context feature bank. Do not claim a full teacher replay.
        losses.append(loss)
    close(feedback["total_loss"], sum(losses), "teacher total loss mismatch")
    close(feedback["mean_loss"], sum(losses) / len(losses), "teacher mean loss mismatch")


def plain_digest(value):
    # The reference stores only plain lists/dicts, encoded exactly like the runner's _plain.
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def check_history(root, restoration, initial_hashes):
    raw = {name: (root / HISTORY / name).read_bytes() for name in HISTORY_SHA}
    require(
        {name: digest(value) for name, value in raw.items()} == HISTORY_SHA,
        "prior public history changed",
    )
    old = json.loads(raw["manifest.json"])
    attempts = {
        digest(line): json.loads(line) for line in raw["attempts.jsonl"].splitlines(keepends=True)
    }
    old_attempts = {(r["route"], r["episode_id"], r["attempt_index"]): r for r in attempts.values()}
    docs = {d["document_id"]: d["text_hex"] for d in old["documents"]}
    ingests = [json.loads(line) for line in raw["ingests.jsonl"].splitlines()]
    observed = [d for r in ingests if r["route"] == "simple" for d in r["document_ids"]]
    require(observed == list(docs), "reference did not observe all prior documents in order")
    demonstrations, bindings = [], []
    for line in raw["teachers.jsonl"].splitlines():
        row = json.loads(line)
        teacher = row["teacher"]
        if row["route"] != "simple" or teacher["kind"] != "edit":
            continue
        attempt = attempts[row["attempt_sha256"]]
        require(
            attempt["route"] == "simple"
            and attempt["phase"] == "acquisition"
            and attempt["attempt_index"] == 1
            and attempt["execution"] is not None
            and attempt["episode_id"] == row["episode_id"],
            "unobserved reference demonstration",
        )
        source = files(attempt["before_files"])[teacher["path"]]
        demonstrations.append(
            {
                "query": {"bytes_hex": attempt["request_hex"]},
                "old": {"bytes_hex": source[teacher["start"] : teacher["end"]].hex()},
                "replacement": {"bytes_hex": teacher["replacement_hex"]},
                "source_sha256": digest(source),
                "query_sha256": digest(bytes.fromhex(attempt["request_hex"])),
            }
        )
        bindings.append({"episode_id": row["episode_id"], "attempt_sha256": row["attempt_sha256"]})
    expected = {
        "schema": "persistent-language-action-baseline-restoration-v2",
        "input_sha256": HISTORY_SHA,
        "documents": 16,
        "edit_demonstrations": 12,
        "document_raw_bytes": 437,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "new_teacher_offers": 0,
        "bindings": bindings,
    }
    require(restoration == expected and len(demonstrations) == 12, "reference restoration changed")
    require(
        initial_hashes["simple"]
        == {
            "model": plain_digest(demonstrations),
            "memory": plain_digest([{"bytes_hex": docs[d]} for d in observed]),
        },
        "restored reference input state mismatch",
    )
    return old, old_attempts


def expected_score(episode, row):
    if row["status"] == "unsupported":
        return False
    if episode["kind"] == "reading":
        return row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
    execution = row["execution"]
    if not execution:
        return False
    grade = episode["grade"]
    before, after = files(episode["files"]), files(row["after_files"])
    proposed = files(execution["proposed_files"])
    protected = all(
        after.get(path) == before[path] and proposed.get(path) == before[path]
        for path in grade["protected_paths"]
    )
    return bool(
        execution["returncode"] == grade["expected_returncode"]
        and execution["stdout_hex"] == grade["expected_stdout_hex"]
        and protected
        and execution["files_complete"]
        and execution["capture_complete"]
        and not execution["timeout"]
        and not execution["output_limit_exceeded"]
        and not execution["capture_errors"]
    )


def verify_application(row):
    execution = row["execution"]
    if not execution:
        return
    before = files(row["before_files"])
    proposed = dict(before)
    proposal = row["proposal"]
    if execution["accepted"]:
        require(proposal is not None, "accepted missing proposal")
        require(proposal["operation"] in {"STOP", "SPLICE"}, "unknown operation")
        if proposal["operation"] == "SPLICE":
            path, start, end = proposal["path"], proposal["start"], proposal["end"]
            require(
                path in before and 0 <= start <= end <= len(before[path]), "invalid accepted span"
            )
            replacement = bytes.fromhex(proposal["replacement_hex"])
            require(
                len(replacement) <= 16 and proposal["stop_reason"] == "eos",
                "invalid accepted payload",
            )
            proposed[path] = before[path][:start] + replacement + before[path][end:]
    require(proposed == files(execution["proposed_files"]), "host substituted or repaired an edit")
    require(execution["before_files"] == row["before_files"], "executor input mismatch")
    require(execution["after_files"] == row["after_files"], "executor result mismatch")
    for stream in ("stdout", "stderr"):
        require(
            digest(bytes.fromhex(execution[stream + "_hex"])) == execution[stream + "_sha256"],
            "captured output hash mismatch",
        )
    # Bind saved workspace bytes when a complete snapshot was possible.
    if execution["files_complete"]:
        directory = Path(execution["workspace"])
        for path, raw in files(row["after_files"]).items():
            target = (directory / path).resolve()
            require(target.is_relative_to(directory.resolve()), "saved workspace path escapes")
            require(target.read_bytes() == raw, "saved after-file mismatch")


def review(run, launch_path, manifest_sha):
    started = time.perf_counter()
    completion = read_json(run / "completion.json")
    require(completion["status"] == "completed", "incomplete session")
    require(
        completion["schema"] == "persistent-language-action-completion-v2", "old completion schema"
    )
    required_files = {
        "manifest.json",
        "report.json",
        "attempts.jsonl",
        "teachers.jsonl",
        "ingests.jsonl",
        "initialization.json",
        "sources.json",
        "launch-record.json",
        "baseline-restoration.json",
    }
    require(required_files <= set(completion["files"]), "incomplete artifact binding")
    for relative, expected in completion["files"].items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run.resolve()), "completion path escapes run")
        require(digest(path.read_bytes()) == expected, f"changed artifact: {relative}")
    manifest = read_json(run / "manifest.json")
    require(
        digest((run / "manifest.json").read_bytes()) == manifest_sha,
        "manifest differs from pre-run freeze",
    )
    source_root = Path(__file__).resolve().parents[2]
    for relative, expected in read_json(run / "sources.json").items():
        require(
            digest((source_root / relative).read_bytes()) == expected, "reviewed run source changed"
        )
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    require(len(episodes) == 56, "wrong episode roster")
    require(
        manifest["schema"] == "persistent-language-action-development-data-v2", "old data schema"
    )
    require(
        manifest["counts"]["occupied_memory_slots"] == 505
        and manifest["counts"]["initial_occupied_memory_slots"] == 453
        and manifest["counts"]["new_memory_slots"] == 52,
        "wrong continued memory stream",
    )
    attempt_rows = records(run / "attempts.jsonl")
    teacher_rows = records(run / "teachers.jsonl")
    ingest_rows = records(run / "ingests.jsonl")
    report = read_json(run / "report.json")
    require(report["schema"] == "persistent-language-action-report-v2", "old report schema")
    initialization = read_json(run / "initialization.json")
    require(
        initialization["schema"] == "persistent-language-action-initial-custody-v2"
        and initialization["bank_sha256"] == PRIOR_BANK_SHA
        and initialization["prior_sha256"] == PRIOR_CHECKPOINT_SHA,
        "wrong continuation prior",
    )
    require(
        digest(Path(initialization["bank_path"]).read_bytes()) == PRIOR_BANK_SHA,
        "prior bank changed",
    )
    require(
        digest(Path(initialization["prior_path"]).read_bytes()) == PRIOR_CHECKPOINT_SHA,
        "prior checkpoint bytes changed",
    )
    prior_bank = read_json(Path(initialization["bank_path"]))
    prior_report_path = source_root / HISTORY / "report.json"
    require(
        digest(prior_report_path.read_bytes()) == prior_bank["evidence"][HISTORY + "/report.json"],
        "prior report is not authenticated",
    )
    prior_report = read_json(prior_report_path)
    require(
        initialization["prior_selected_route"] == "online"
        and initialization["restored_occupied_slots_per_route"] == 453
        and initialization["initial_experience_per_route"] == 46
        and initialization["functional_prior_parity"] is False,
        "continuation state or retrieval-policy declaration changed",
    )
    old, old_attempts = check_history(
        source_root, read_json(run / "baseline-restoration.json"), report["initial_hashes"]
    )
    require(
        manifest["continuation"]["prior_documents"] == old["documents"],
        "prior memory was replaced or task-selected",
    )
    require(
        initialization["torch_load_calls"] == 1
        and initialization["loaded_payload_tensor_bytes"] == 2919568
        and initialization["new_combined_model_constructions"] == 2
        and initialization["new_reader_constructions"] == 2
        and initialization["new_action_circuit_constructions"] == 2
        and initialization["restored_tape_states"] == 2
        and initialization["loaded_model_state_containers"] == 2
        and initialization["loaded_tape_state_containers"] == 2,
        "prior load accounting",
    )
    for route in ROUTES[:2]:
        require(
            report["initial_hashes"][route]["model"]
            == initialization["initial_model_sha256"][route],
            "initialization state changed before stream",
        )
        require(
            report["initial_hashes"][route]["memory"]
            == initialization["initial_memory_sha256"][route]
            == prior_report["final_hashes"]["online"]["memory"],
            "accumulated prior memory changed",
        )
    child_launch = read_json(run / "launch-record.json")
    child_launch_sha = digest(
        json.dumps(child_launch, sort_keys=True, separators=(",", ":")).encode()
    )
    require(
        report["initial_hashes"]["online"]["model"]
        == report["initial_hashes"]["learning_disabled"]["model"],
        "unmatched initial neuronal models",
    )
    by_episode = defaultdict(list)
    attempt_by_sha = {}
    for row, sha in attempt_rows:
        require(row["schema"] == "persistent-language-action-attempt-v2", "old attempt schema")
        route, key = row["route"], row["episode_id"]
        require(route in ROUTES and key in episodes, "unknown route or episode")
        episode = episodes[key]
        require(row["phase"] == episode["phase"] and row["kind"] == episode["kind"], "phase drift")
        require(row["request_hex"] == episode["request_hex"], "request changed or truncated")
        require(row["feedback_consumed"] is False, "undeclared feedback encoder")
        obs = row["observation"]
        require(set(obs) == {"kind", "request_hex", "files"}, "private observation field")
        require(
            obs["request_hex"] == row["request_hex"] and obs["files"] == row["before_files"],
            "observation drift",
        )
        previous = by_episode[route, key]
        require(row["attempt_index"] == len(previous) + 1, "attempt numbering")
        require(len(previous) < episode["attempt_limit"], "attempt budget exceeded")
        require(
            row["before_files"] == (previous[-1]["after_files"] if previous else episode["files"]),
            "workspace reset or repair within retry",
        )
        require(row["correct"] is expected_score(episode, row), "recorded correctness mismatch")
        verify_application(row)
        execution = row["execution"]
        if execution:
            require(
                execution["launch_record_sha256"] == child_launch_sha,
                "child launch identity changed",
            )
            command = execution["child_command"]
            require(
                command[:-1] == [child_launch["base_executable"], *child_launch["flags"], "-c"],
                "child runtime or flags changed",
            )
            require(
                digest(command[-1].encode())
                == execution["bootstrap_sha256"]
                == "7213881ba88871e024789913aa742bba520cc79520b8a7b87292c0d3b2b745b9",
                "child bootstrap changed",
            )
        if route != "simple":
            require(
                row["model_before_sha256"] == row["model_after_sha256"],
                "free step changed slow weights",
            )
            require(
                row["memory_before_sha256"] == row["memory_after_sha256"], "free step changed tape"
            )
            if row["prediction"] is not None:
                prediction = (
                    row["prediction"] if row["kind"] == "reading" else row["prediction"]["payload"]
                )
                check_prediction(prediction, row["request_hex"], row["memory_used"])
                if row["kind"] == "reading":
                    require(prediction["answer_hex"] == row["answer_hex"], "reading bytes changed")
                else:
                    require(
                        prediction["answer_hex"] == row["proposal"]["replacement_hex"]
                        and prediction["stop_reason"] == row["proposal"]["stop_reason"],
                        "host changed generated action payload",
                    )
                    require(
                        row["prediction"]["model_sha256"] == row["model_before_sha256"]
                        and row["prediction"]["memory_sha256"] == row["memory_before_sha256"],
                        "pending proposal is not the acting model/tape",
                    )
        previous.append(row)
        require(sha not in attempt_by_sha, "duplicate attempt bytes")
        attempt_by_sha[sha] = row

    teaching = defaultdict(list)
    for row, _ in teacher_rows:
        require(row["schema"] == "persistent-language-action-teacher-v2", "old correction schema")
        require(row["attempt_sha256"] in attempt_by_sha, "teacher lacks preceding attempt")
        attempt = attempt_by_sha[row["attempt_sha256"]]
        episode = episodes[attempt["episode_id"]]
        require(
            attempt["phase"] == "acquisition" and attempt["attempt_index"] == 1,
            "teacher outside acquisition",
        )
        require(row["teacher"] == episode["teacher"], "teacher differs from frozen offer")
        require(row["route"] == attempt["route"], "teacher route mismatch")
        if row["route"] != "simple":
            require(
                row["learning_enabled"] is (row["route"] == "online"),
                "wrong local publication gate",
            )
            require(
                row["model_before_sha256"] == attempt["model_after_sha256"], "teacher state custody"
            )
            if row["route"] == "learning_disabled":
                require(
                    row["model_before_sha256"] == row["model_after_sha256"],
                    "disabled model changed",
                )
            publication = row["publication"]
            require(
                publication["legacy_codec"] is False
                and publication["codec"] is publication["output"]
                and publication["address"]
                == (publication["query_address"] or publication["prefix_address"])
                and row["publication_alias"] == "codec means new output U",
                "old and new output publication conflated",
            )
            require(
                all(
                    type(publication[key]) is bool
                    for key in (
                        "legacy_codec",
                        "codec",
                        "output",
                        "address",
                        "query_address",
                        "prefix_address",
                    )
                ),
                "publication flag types",
            )
            if row["route"] == "learning_disabled":
                require(
                    not any(
                        publication[key]
                        for key in ("codec", "output", "address", "query_address", "prefix_address")
                    ),
                    "disabled reader publication",
                )
                require(
                    not any(publication.get("actions", {}).values()), "disabled action publication"
                )
        teaching[row["route"]].append(row)

    documents = {d["document_id"]: bytes.fromhex(d["text_hex"]) for d in manifest["documents"]}
    ingests = defaultdict(list)
    for row, _ in ingest_rows:
        require(row["schema"] == "persistent-language-action-ingest-v2", "old append schema")
        ingests[row["route"]].append(row)
    summaries = {}
    for route in ROUTES:
        require(len(teaching[route]) == 24, "missing or repeated correction offer")
        require(len({t["attempt_sha256"] for t in teaching[route]}) == 24, "replayed correction")
        selected = [r for r, _ in attempt_rows if r["route"] == route]
        require(
            len({r["model_instance_id"] for r in selected}) == 1, "model replaced during session"
        )
        require(
            [r["episode_id"] for r in selected if r["attempt_index"] == 1] == list(episodes),
            "episode order or missing exposure",
        )
        phase_scores = {}
        for phase in PHASES:
            subset = [e for e in episodes.values() if e["phase"] == phase]
            first = [by_episode[route, e["episode_id"]][0] for e in subset]
            phase_scores[phase] = {
                "first_correct": sum(r["correct"] for r in first),
                "episodes": len(first),
                "reading_correct": sum(r["correct"] for r in first if r["kind"] == "reading"),
                "reading_total": sum(r["kind"] == "reading" for r in first),
                "edit_correct": sum(r["correct"] for r in first if r["kind"] == "edit"),
                "edit_total": sum(r["kind"] == "edit" for r in first),
                "supported_correct": sum(
                    r["correct"] for r in first if episodes[r["episode_id"]]["supported"]
                ),
                "supported_total": sum(episodes[r["episode_id"]]["supported"] for r in first),
            }
        recovery = [
            by_episode[route, e["episode_id"]]
            for e in episodes.values()
            if e["phase"] == "recovery"
        ]
        require(all(len(rows) == 3 for rows in recovery), "recovery must use three fixed attempts")
        phase_scores["recovery"].update(
            first_failures=sum(not rows[0]["correct"] for rows in recovery),
            recovered=sum(
                not rows[0]["correct"] and any(r["correct"] for r in rows[1:]) for rows in recovery
            ),
            any_attempt_correct=sum(any(r["correct"] for r in rows) for rows in recovery),
            final_correct=sum(rows[-1]["correct"] for rows in recovery),
            final_recovered=sum(
                not rows[0]["correct"] and rows[-1]["correct"] for rows in recovery
            ),
            attempts=sum(len(rows) for rows in recovery),
        )
        retention_pairs = []
        for e in episodes.values():
            if e["phase"] != "retention":
                continue
            earlier = e["prior_episode_id"]
            prior_route = "simple" if route == "simple" else "online"
            prior_id = "T" + earlier[1:]
            old_episode = next(r for r in old["episodes"] if r["episode_id"] == prior_id)
            before = old_attempts[prior_route, prior_id, 1]
            require(
                e["request_hex"] == old_episode["request_hex"]
                and e["files"] == old_episode["files"]
                and e["grade"] == old_episode["grade"]
                and e["teacher"] is None,
                "old regression observation or target changed",
            )
            require(
                before["correct"] is expected_score(old_episode, before),
                "prior delayed grade mismatch",
            )
            retention_pairs.append(
                {
                    "episode_id": e["episode_id"],
                    "prior_acquisition_id": earlier,
                    "prior_retention_id": prior_id,
                    "prior_route": prior_route,
                    "prior_acquisition_first_correct": old_attempts[prior_route, earlier, 1][
                        "correct"
                    ],
                    "prior_delayed_correct": before["correct"],
                    "retention_correct": by_episode[route, e["episode_id"]][0]["correct"],
                }
            )
        phase_scores["retention"]["pairs"] = retention_pairs
        phase_scores["acquisition"]["blocks_of_eight"] = [
            sum(
                by_episode[route, e["episode_id"]][0]["correct"]
                for e in list(episodes.values())[i : i + 8]
            )
            for i in (0, 8, 16)
        ]
        appended = [key for r in ingests[route] for key in r["document_ids"]]
        require(appended == list(documents), "memory append order, duplication, or target routing")
        require(
            sum(len(documents[key]) + 1 for key in appended) == 52, "new memory byte/EOS exposure"
        )
        current = dict(report["initial_hashes"][route])
        used = 453
        ingest_by_episode = {r["episode_id"]: r for r in ingests[route]}
        require(len(ingest_by_episode) == len(ingests[route]), "duplicate ingest event")
        teacher_by_attempt = {r["attempt_sha256"]: r for r in teaching[route]}
        for row, row_sha in attempt_rows:
            if row["route"] != route:
                continue
            episode = episodes[row["episode_id"]]
            if row["attempt_index"] == 1 and episode["append_document_ids"]:
                ingest = ingest_by_episode[row["episode_id"]]
                require(
                    ingest["document_ids"] == episode["append_document_ids"],
                    "append scheduled at wrong episode",
                )
                require(ingest["used_before"] == used, "ingest cursor reset")
                raw_bytes = sum(len(documents[key]) for key in ingest["document_ids"])
                require(
                    ingest["raw_bytes"] == raw_bytes
                    and ingest["eos"] == len(ingest["document_ids"]),
                    "ingest exposure mismatch",
                )
                used += raw_bytes + ingest["eos"]
                require(ingest["used_after"] == used, "ingest occupancy mismatch")
                for kind in ("model", "memory"):
                    require(ingest[kind + "_before_sha256"] == current[kind], "ingest state chain")
                    current[kind] = ingest[kind + "_after_sha256"]
                require(
                    ingest["model_before_sha256"] == ingest["model_after_sha256"],
                    "ingest changed slow weights",
                )
            require(row["memory_used"] == used, "task-selected or truncated memory")
            for kind in ("model", "memory"):
                require(row[kind + "_before_sha256"] == current[kind], "attempt state chain")
                current[kind] = row[kind + "_after_sha256"]
            teacher = teacher_by_attempt.get(row_sha)
            if teacher:
                for kind in ("model", "memory"):
                    require(
                        teacher[kind + "_before_sha256"] == current[kind], "correction state chain"
                    )
                    current[kind] = teacher[kind + "_after_sha256"]
                require(
                    teacher["memory_before_sha256"] == teacher["memory_after_sha256"],
                    "teacher secretly wrote memory",
                )
                if route != "simple":
                    target = teacher["teacher"].get(
                        "target_hex", teacher["teacher"].get("replacement_hex")
                    )
                    require(
                        teacher["reader_offers"] == len(bytes.fromhex(target)) + 1,
                        "teacher-symbol exposure mismatch",
                    )
                    require(
                        teacher["action_labels"] == (3 if row["kind"] == "edit" else 0),
                        "action teacher exposure mismatch",
                    )
                    work = teacher["work"]["reader"] if row["kind"] == "edit" else teacher["work"]
                    check_reader_work(
                        work,
                        used,
                        teacher["reader_offers"],
                        len(bytes.fromhex(row["request_hex"])),
                        teacher=True,
                        enabled=route == "online",
                    )
                    check_teacher_feedback(teacher["reader_feedback"], bytes.fromhex(target), used)
        require(current == report["final_hashes"][route], "final persistent state mismatch")
        require(
            used == 505 and len(selected) == 72 and len(ingests[route]) == 2,
            "continued session exposure mismatch",
        )
        if route != "simple":
            require(
                sum(t["reader_offers"] for t in teaching[route]) == 98
                and sum(t["action_labels"] for t in teaching[route]) == 36,
                "reader/action correction totals",
            )
        recorded = report["metrics"][route]
        for phase in PHASES:
            for kind in ("reading", "edit"):
                summary = phase_scores[phase]
                require(
                    recorded["phases"][phase][kind]["first_correct"] == summary[kind + "_correct"],
                    "report first-score aggregation mismatch",
                )
                require(
                    recorded["phases"][phase][kind]["episodes"] == summary[kind + "_total"],
                    "report phase denominator mismatch",
                )
        require(
            recorded["recovery"]["later_correct"] == phase_scores["recovery"]["recovered"],
            "report recovery mismatch",
        )
        require(
            recorded["recovery"]["final_attempt_correct"]
            == phase_scores["recovery"]["final_correct"]
            and recorded["recovery"]["final_recovered_after_first_failure"]
            == phase_scores["recovery"]["final_recovered"],
            "terminal recovery aggregation mismatch",
        )
        require(
            recorded["acquisition_blocks"] == phase_scores["acquisition"]["blocks_of_eight"],
            "acquisition blocks mismatch",
        )
        expected_retention = [
            {
                "episode_id": p["episode_id"],
                "prior_acquisition_id": p["prior_acquisition_id"],
                "prior_retention_id": p["prior_retention_id"],
                "prior_route": p["prior_route"],
                "initial_correct": p["prior_delayed_correct"],
                "delayed_correct": p["retention_correct"],
            }
            for p in retention_pairs
        ]
        require(
            recorded["retention"] == expected_retention,
            "cross-session retention paired to wrong session",
        )
        for key, expected in {
            "retained": sum(
                p["prior_delayed_correct"] and p["retention_correct"] for p in retention_pairs
            ),
            "forgotten": sum(
                p["prior_delayed_correct"] and not p["retention_correct"] for p in retention_pairs
            ),
            "improved": sum(
                not p["prior_delayed_correct"] and p["retention_correct"] for p in retention_pairs
            ),
        }.items():
            require(recorded[key] == expected, "retention aggregation mismatch")
        summaries[route] = phase_scores

    launch = read_json(launch_path)
    require(launch["status"] == "passed", "workload watchdog failed")
    require(
        launch["elapsed_seconds"] <= 840 and launch["observed_tree_peak_sum_mib"] <= 2048,
        "workload resource envelope",
    )
    command = launch["command"]
    require(
        command[1:5]
        == ["-X", "utf8", "-m", "experiments.r4_integrated.persistent_language_action_v2"],
        "unrelated workload command",
    )
    require(
        Path(command[command.index("--output") + 1]).resolve() == run.resolve(),
        "launch output mismatch",
    )
    require(
        command[command.index("--manifest-sha") + 1] == manifest_sha,
        "launch manifest freeze mismatch",
    )
    require(
        any(
            key.startswith(str(completion["pid"]) + "/")
            for key in launch["observed_process_peaks_mib"]
        ),
        "runner PID absent from owned process tree",
    )
    require(report["pid"] == completion["pid"], "report/completion PID mismatch")
    require(
        report["final_checkpoint"]["sha256"] == completion["files"]["final-state.pt"],
        "final continuation checkpoint unbound",
    )
    require(
        len(attempt_rows) == 216 and len(teacher_rows) == 72 and len(ingest_rows) == 6,
        "complete session count mismatch",
    )
    require(
        sum(r["execution"] is not None for r, _ in attempt_rows) == 144,
        "actual execution count mismatch",
    )
    require(
        report["counts"] == {"attempts": 216, "teachers": 72, "ingests": 6},
        "reported exposure count mismatch",
    )
    require(
        initialization["retained_model_tensor_bytes"] == 2625216
        and report["final_checkpoint"]["tensor_bytes"] == 4275920,
        "new model plus persistent tape payload mismatch",
    )
    return {
        "schema": "persistent-language-action-saved-review-v2",
        "status": "passed",
        "summaries": summaries,
        "attempt_rows": len(attempt_rows),
        "teacher_rows": len(teacher_rows),
        "ingest_rows": len(ingest_rows),
        "manifest_sha256": digest((run / "manifest.json").read_bytes()),
        "completion_sha256": digest((run / "completion.json").read_bytes()),
        "launch_sha256": digest(launch_path.read_bytes()),
        "baseline_restoration_sha256": digest((run / "baseline-restoration.json").read_bytes()),
        "prior_evidence_sha256": HISTORY_SHA,
        "initial_memory_slots": 453,
        "added_memory_slots": 52,
        "final_memory_slots": 505,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": (
            "behavior, causal prefixes, output normalization/selection; exposure and custody; "
            "no feature/neuron replay, teacher conditional arithmetic replay "
            "or checkpoint deserialization"
        ),
        "model_constructions": 0,
        "checkpoint_loads": 0,
        "actual_executions": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    result = review(args.run, args.launch, args.manifest_sha)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "attempt_rows": result["attempt_rows"]}))


if __name__ == "__main__":
    main()
