"""Saved V8 evidence: scalar stochastic receipts and durable actual frontiers.

No neuron, controller, feature, predictor, executor, task builder or tensor decoder
is invoked. Eligibility tensors and learned logits are source/custody claims;
the saved probability, relay, reward and normalization arithmetic is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_v7_review as prior

scalar = prior.scalar
require = scalar.require
encoded = prior.encoded
digest = prior.digest
read_json = prior.read_json
safe_path = prior.safe_path
journal = prior.journal
read_row = prior.read_row
work_valid = prior.work_valid
ROOT = Path(__file__).resolve().parents[2]
ROUTES = prior.ROUTES
COUNTERS = prior.COUNTERS
DRAFT = False
MANIFEST_SHA = "66b0bd96be3b9e97c4eb5f9812c2b5977f294fc5900d55109445c0a303e522d0"
SAVED_ONLY_SOURCES = {
    "experiments/r4_integrated/persistent_language_action_v8_review.py",
    "experiments/r4_integrated/bank_persistent_language_action_v8.py",
}
POPULATIONS = {"context": 32, "gate": 8, "mode": 4, "literal": 257}


def close(a, b, label):
    require(
        type(a) in (int, float)
        and math.isfinite(a)
        and math.isclose(a, b, rel_tol=2e-9, abs_tol=2e-10),
        label,
    )


def scan_schedule(observation=None, raw=None):
    """Mechanical field framing, never a feature or neural activity rebuild."""
    if raw is not None:
        return [(13, i, i, token, False) for i, token in enumerate([*raw, 256])]
    feedback, files = observation["feedback"], observation["files"]
    fields = [bytes.fromhex(observation["request_hex"])]
    fields += (
        [None, None, None]
        if feedback is None
        else [
            bytes.fromhex(feedback["stdout_hex"]),
            bytes.fromhex(feedback["stderr_hex"]),
            None
            if feedback["returncode"] is None
            else feedback["returncode"].to_bytes(8, "little", signed=True),
        ]
    )
    items = files or []
    for index in range(4):
        fields += (
            [items[index]["path"].encode(), bytes.fromhex(items[index]["content_hex"])]
            if index < len(items)
            else [None, None]
        )
    flags = [files is not None, *(i < len(items) for i in range(4))]
    flags += (
        [False] * 7
        if feedback is None
        else [
            True,
            feedback["returncode"] is not None,
            *(
                feedback[k]
                for k in (
                    "accepted",
                    "timeout",
                    "capture_complete",
                    "files_complete",
                    "output_limit_exceeded",
                )
            ),
        ]
    )
    fields.append(bytes(map(int, flags)))
    schedule, arrived = [], 0
    for field, value in enumerate(fields):
        if value is None:
            arrived += 1  # An absent marker is framing, not an offered target.
            continue
        for offset, token in enumerate([*value, 256]):
            schedule.append((field, offset, arrived, token, field == 12))
            arrived += 1
    return schedule


def new_symbol_exposures():
    """Counts from checked ticks; free output has no observed target."""
    return {
        category: {
            "ticks": 0,
            "draws": 0,
            "sampled_modes": Counter(dict.fromkeys(range(4), 0)),
            "sampled_gates": Counter(dict.fromkeys(range(8), 0)),
            "observed_symbols": None if category == "free" else {
                kind: Counter(offers=0, matches=0)
                for kind in ("raw_content", "eos", "status")
            },
        }
        for category in ("source_scan", "public_scan", "free", "teacher")
    }


def new_trajectory(*, scan_kind="public_scan"):
    return {
        "ordinal": 0,
        "rng": None,
        "h": 0,
        "cursors": [None, None],
        "first_rng": None,
        "frame": None,
        "ticks": Counter(),
        "rewards": Counter(),
        "public_symbols": {},
        "public_successors": {},
        "public_reveal": {},
        "scan_kind": scan_kind,
        "symbol_exposures": new_symbol_exposures(),
    }


def check_tick(tick, category, trajectory, *, target=None, prior_token=None, address=None):
    require(
        tick["schema"] == "recurrent-content-tick-v1" and tick["category"] == category,
        "controller tick schema/category",
    )
    for key in ("frame_sha256", "position_sha256"):
        scalar.hash_value(tick[key])
    if trajectory["frame"] is None:
        trajectory["frame"] = tick["frame_sha256"]
    require(
        tick["frame_sha256"] == trajectory["frame"]
        and tick["h_before"] == trajectory["h"]
        and tick["cursors_before"] == trajectory["cursors"],
        "causal context/cursor chain",
    )
    if category != "scan":
        require(
            (tick["field"], tick["offset"], tick["arrived"]) == trajectory["output_frontier"],
            "output public visibility frontier",
        )
    draws = tick["draws"]
    require(len(draws) in (3, 4), "stochastic tick draw count")
    for number, draw in enumerate(draws):
        name = tuple(POPULATIONS)[number]
        require(
            draw["schema"] == "recurrent-content-draw-v1"
            and draw["population"] == name
            and draw["category"] == category,
            "context/gate/mode/literal ordering",
        )
        logits = scalar.vector(draw["logits"], POPULATIONS[name])
        probabilities = scalar.vector(draw["probabilities"], len(logits))
        maximum = max(logits)
        exps = [math.exp(value - maximum) for value in logits]
        total = sum(exps)
        require(all(p > 0 for p in probabilities), "finite strictly positive choice probabilities")
        for actual, expected in zip(probabilities, exps, strict=True):
            close(actual, expected / total, "own saved softmax probability")
        u = draw["uniform"]
        require(
            type(u) is float and math.isfinite(u) and 0 <= u < 1,
            "saved uniform outside unit interval",
        )
        cumulative, winner = 0.0, len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if u < (1.0 if index == len(probabilities) - 1 else cumulative):
                winner = index
                break
        require(
            draw["index"] == winner and draw["ordinal"] == trajectory["ordinal"] + 1,
            "saved categorical draw/ordinal",
        )
        for key in ("activity_sha256", "rng_before_sha256", "rng_after_sha256"):
            scalar.hash_value(draw[key])
        require(
            draw["rng_before_sha256"] != draw["rng_after_sha256"]
            and (trajectory["rng"] is None or trajectory["rng"] == draw["rng_before_sha256"]),
            "monotonic actual RNG chain",
        )
        trajectory["rng"] = draw["rng_after_sha256"]
        if trajectory["first_rng"] is None:
            trajectory["first_rng"] = draw["rng_before_sha256"]
        trajectory["ordinal"] += 1
    mode = draws[2]["index"]
    require(len(draws) == 3 + int(mode == 3), "unselected literal draw")
    registers = tick["register_tokens"]
    require(
        len(registers) == 2
        and all(v is None or (type(v) is int and 0 <= v <= 256) for v in registers),
        "emittable register symbols",
    )
    require(
        tick["prior_token"] == prior_token and tick["retrieved_index"] == address,
        "causal prior/address substitution",
    )
    token = (
        (258 if prior_token is None else prior_token)
        if mode == 0
        else (256 if registers[mode - 1] is None else registers[mode - 1])
        if mode < 3
        else draws[3]["index"]
    )
    require(
        tick["emitted_token"] == token and tick["h_after"] == draws[0]["index"],
        "selected content emission",
    )
    require(
        len(tick["cursors_after"]) == 2
        and all(v is None or (type(v) is int and 0 <= v < 5533) for v in tick["cursors_after"]),
        "cursor population",
    )
    gate = draws[1]["index"]
    expected_cursors = list(tick["cursors_before"])
    if gate == 7:
        expected_cursors.reverse()
    elif gate in (1, 2):
        current = None
        if category == "scan" and tick["offset"]:
            current = trajectory["field_addresses"][(tick["field"], tick["offset"] - 1)]
        expected_cursors[gate - 1] = current
    elif gate in (3, 4):
        expected_cursors[gate - 3] = None if address is None else 3485 + address
    elif gate in (5, 6):
        cursor = expected_cursors[gate - 5]
        if cursor is not None and cursor < 3485:
            expected_cursors[gate - 5] = trajectory["public_successors"].get(cursor)
        elif cursor is not None:
            # Factual successors are hash-bound tensors, not decoded in this audit.
            expected_cursors[gate - 5] = tick["cursors_after"][gate - 5]
    require(expected_cursors == tick["cursors_after"], "sampled fixed cursor relay")
    for cursor, value in zip(expected_cursors, registers, strict=True):
        if cursor is None or cursor < 3485:
            visible = (
                cursor is not None
                and trajectory["public_reveal"].get(cursor, 10**9) < tick["arrived"]
            )
            expected = trajectory["public_symbols"].get(cursor) if visible else None
            require(value == expected, "public register read visibility/raw value")
    trajectory["h"], trajectory["cursors"] = tick["h_after"], tick["cursors_after"]
    trajectory["ticks"][category] += 1
    if target is not None:
        reward = 1 if token == target else -1
        trajectory["rewards"][category] += reward
    exposure = trajectory["symbol_exposures"][
        trajectory["scan_kind"] if category == "scan" else category
    ]
    exposure["ticks"] += 1
    exposure["draws"] += len(draws)
    exposure["sampled_modes"][mode] += 1
    exposure["sampled_gates"][gate] += 1
    if target is not None:
        kind = (
            "status" if category == "scan" and tick["status_offer"]
            else "eos" if target == 256 else "raw_content"
        )
        exposure["observed_symbols"][kind]["offers"] += 1
        exposure["observed_symbols"][kind]["matches"] += int(token == target)
    return token


def check_scan(steps, schedule, trajectory):
    require(len(steps) == len(schedule), "whole public/source offer count")
    starts = (0, 257, 770, 1283, 1291, 1323, 1836, 1868, 2381, 2413, 2926, 2958, 3472, 0)
    addresses = {}
    for field, offset, arrived, token, _ in schedule:
        address = 3471 if field == 3 and token == 256 else starts[field] + offset
        addresses[field, offset] = address
        trajectory["public_symbols"][address] = token
        trajectory["public_reveal"][address] = arrived
    trajectory["field_addresses"] = addresses
    for (field, offset), address in addresses.items():
        trajectory["public_successors"][address] = addresses.get((field, offset + 1))
    for tick, (field, offset, arrived, target, status) in zip(steps, schedule, strict=True):
        require(
            (
                tick["field"],
                tick["offset"],
                tick["arrived"],
                tick["observed_token"],
                tick["status_offer"],
            )
            == (field, offset, arrived, target, status),
            "public field order or offered raw symbol changed",
        )
        token = check_tick(tick, "scan", trajectory, target=target)
        require(tick["reward"] == (1 if token == target else -1), "pre-arrival scan reward")


def check_prior(step, prefix, used, binding_hash=None):
    prior.check_binding(step["binding"], prefix, request_present=True)
    if binding_hash is not None:
        require(step["binding"]["bank_sha256"] == binding_hash, "held binding changed")
    scalar.memory_winner(step["address_scores"], step["selected_index"], used)
    scores = scalar.vector(step["assembly_scores"], 398) if "assembly_scores" in step else None
    if scores is not None:
        require(
            step["assembly_index"] == max(range(398), key=lambda i: (scores[i], -i)),
            "all inherited event owners eligible",
        )
    logits = scalar.vector(step["prior_logits"], 257)
    require(logits == step["binding"]["prior"], "actual old prior delivered to binding")
    return step["binding"]["winner"]


def check_prediction(prediction, observation, used, responses):
    require(
        prediction["schema"] == "recurrent-content-output-prediction-v1"
        and prediction["observation_sha256"] == scalar.observation_hash(observation),
        "actual full observation prediction binding",
    )
    identity = prediction["attempt_id"]
    require(identity not in responses, "response identity reused")
    responses.add(identity)
    for key in ("model_sha256", "memory_sha256", "trace_sha256"):
        scalar.hash_value(prediction[key])
    trajectory = new_trajectory()
    schedule = scan_schedule(observation)
    check_scan(prediction["scan_steps"], schedule, trajectory)
    trajectory["output_frontier"] = (12, 13, schedule[-1][2] + 1)
    receipt = prediction["controller"]
    require(
        receipt["schema"] == "recurrent-content-response-v1"
        and receipt["kind"] == ("reading" if observation["files"] is None else "edit")
        and receipt["scan_offers"] == len(schedule)
        and receipt["scan_status_offers"] == 13
        and receipt["frame_sha256"] == trajectory["frame"]
        and receipt["preoutput_h"] == trajectory["h"]
        and receipt["preoutput_cursors"] == trajectory["cursors"],
        "single common scan before actual output",
    )
    for key in (
        "preoutput_position_sha256",
        "pending_stage_sha256",
        "frame_content_sha256",
        "prior_request_sha256",
    ):
        scalar.hash_value(receipt[key])
    steps, output = prediction["steps"], bytearray()
    require(
        steps[0]["controller"]["position_sha256"] == receipt["preoutput_position_sha256"],
        "actual free starts at the saved preoutput frontier",
    )
    limit = prediction["max_bytes"]
    require(
        type(limit) is int
        and 0 <= limit <= 128
        and 1 <= len(steps) <= limit + 1
        and prediction["decisions"] == len(steps),
        "actual response decision bound",
    )
    bank = steps[0]["binding"]["bank_sha256"]
    request_sha = steps[0]["binding"]["request_sha256"]
    for index, step in enumerate(steps):
        require(
            step["output_step"] == index and step["prefix_hex"] == output.hex(),
            "free output uses generated prefix",
        )
        require(step["binding"]["request_sha256"] == request_sha, "held request input")
        winner = check_prior(step, bytes(output), used, bank)
        token = check_tick(
            step["controller"],
            "free",
            trajectory,
            prior_token=winner,
            address=step["selected_index"],
        )
        require(
            step["token"] == token and (index == len(steps) - 1 or token != 256),
            "free output token/early EOS",
        )
        logits = step["prior_logits"]
        maximum = max(logits)
        normalizer = maximum + math.log(sum(math.exp(x - maximum) for x in logits))
        require(
            math.isclose(step["prior_nll"], normalizer - logits[token], rel_tol=2e-6, abs_tol=2e-6),
            "float32 old prior diagnostic NLL",
        )
        if token != 256 and index < limit:
            output.append(token)
    require(token == 256 or len(steps) == limit + 1, "premature LIMIT")
    require(
        prediction["answer_hex"] == output.hex()
        and prediction["stop_reason"] == ("eos" if token == 256 else "limit"),
        "actual output bytes/EOS status",
    )
    work_valid(prediction["work"])
    return trajectory


def check_resolution(record, trajectory, kind, enabled, baseline, public_reward=None):
    """Validate exposed category arithmetic; no eligibility/weight reconstruction."""
    require(
        record["schema"] == "recurrent-content-update-v1"
        and record["learning_enabled"] is enabled
        and record["rng_published"] is True,
        "controller publication contract",
    )
    for key in ("source_sha256", "state_sha256"):
        scalar.hash_value(record[key])
    ticks = {k: trajectory["ticks"][k] for k in ("scan", "free", "teacher")}
    require(
        record["ticks"] == ticks and record["draws"] == trajectory["ordinal"],
        "actual stochastic exposure ledger",
    )
    scan = trajectory["rewards"]["scan"] / ticks["scan"]
    teacher = trajectory["rewards"]["teacher"] / ticks["teacher"] if ticks["teacher"] else None
    si = 0 if kind == "source" else 1
    saved = record["rewards"]
    close(saved["scan"], scan, "mean observed-symbol reward")
    require((saved["teacher"] is None) == (teacher is None), "teacher category presence")
    if teacher is not None:
        close(saved["teacher"], teacher, "mean correction reward")
    require(saved["public"] == public_reward, "actual public consequence reward")
    close(saved["scan_baseline_before"], baseline[si][0], "online scan baseline chronology")
    close(saved["teacher_baseline_before"], baseline[2][0], "online teacher baseline chronology")
    categories = 1 + int(teacher is not None) + int(kind == "edit")
    length = ticks["scan"] + (0 if kind == "source" else ticks["teacher"] + 129)
    require(
        record["category_count"] == categories and record["normalization_ticks"] == length,
        "fixed per-interaction normalization",
    )
    teacher_error = 0 if teacher is None else teacher - baseline[2][0]
    public = 0 if public_reward is None else public_reward
    coefficients = dict(saved["coefficients"])
    require(set(coefficients) == {"scan", "free", "teacher"}, "three eligibility categories")
    for key, value in {
        "scan": (scan - baseline[si][0] + teacher_error + public) / categories,
        "free": public / categories,
        "teacher": teacher_error / categories,
    }.items():
        close(coefficients[key], value, "shared scan/fork category coefficient")
    norm = record["delta_norm"]
    require(
        type(norm) in (float, int)
        and math.isfinite(norm)
        and 0 <= norm <= 0.02 * math.sqrt(8) + 1e-10,
        "local update norm bound",
    )
    if enabled:
        for index, value in ((si, scan), (2, teacher)):
            if value is not None:
                mean, count = baseline[index]
                baseline[index] = [mean + (value - mean) / (count + 1), count + 1]


def check_source(feedback, raw, enabled, admitted, baseline):
    require(
        feedback["document_count"] == 1
        and feedback["raw_bytes"] == len(raw)
        and feedback["symbol_count"] == len(raw) + 1
        and feedback["learning_enabled"] is enabled
        and feedback["fact_admitted"] is admitted
        and feedback["writes_enabled"] is admitted,
        "one complete admitted/transient source contract",
    )
    trajectory = new_trajectory(scan_kind="source_scan")
    check_scan(feedback["source_steps"], scan_schedule(raw=raw), trajectory)
    check_resolution(feedback["controller"], trajectory, "source", enabled, baseline)
    work_valid(feedback["work"])
    return trajectory


def check_teacher_feedback(
    feedback, prediction, target, used, trajectory, enabled, baseline, kind, public_reward
):
    steps = feedback["steps"]
    require(
        feedback["actual_response_id"] == prediction["attempt_id"]
        and len(steps) == len(target) + 1,
        "actual response/full correction binding",
    )
    common = prediction["controller"]
    trajectory["h"], trajectory["cursors"] = common["preoutput_h"], common["preoutput_cursors"]
    # Only neural context forks. ordinal/RNG and common scan eligibility do not rewind.
    bank = prediction["steps"][0]["binding"]["bank_sha256"]
    request = prediction["steps"][0]["binding"]["request_sha256"]
    loss = 0.0
    for index, (step, token) in enumerate(zip(steps, [*target, 256], strict=True)):
        if index == 0:
            require(
                step["controller"]["position_sha256"] == common["preoutput_position_sha256"],
                "teacher context fork starts before actual free output",
            )
        require(
            step["output_step"] == index
            and step["prefix_hex"] == target[:index].hex()
            and step["observed_token"] == token
            and step["binding"]["request_sha256"] == request,
            "teacher uses earlier offered prefix, no current target input",
        )
        winner = check_prior(step, target[:index], used, bank)
        emitted = check_tick(
            step["controller"],
            "teacher",
            trajectory,
            target=token,
            prior_token=winner,
            address=step["selected_index"],
        )
        require(step["reward"] == (1 if emitted == token else -1), "pre-correction reward")
        logits = step["prior_logits"]
        maximum = max(logits)
        expected = maximum + math.log(sum(math.exp(v - maximum) for v in logits)) - logits[token]
        require(
            math.isclose(step["prior_nll"], expected, rel_tol=2e-6, abs_tol=2e-6),
            "float32 teacher old prior diagnostic",
        )
        loss += step["prior_nll"]
    close(feedback["total_loss"], loss, "teacher diagnostic total")
    close(feedback["mean_loss"], loss / len(steps), "teacher diagnostic mean")
    check_resolution(feedback["controller"], trajectory, kind, enabled, baseline, public_reward)
    return len(steps)


def _check_saved_execution(row, episode, run, launch):
    execution, proposal = row["execution"], row["proposal"]
    require(execution["schema"] == "persistent-language-action-execution-v2", "execution schema")
    before = scalar.files(row["before_files"])
    proposed = dict(before)
    if proposal is not None:
        require(
            proposal["workspace_sha256"] == scalar.workspace_hash(row["before_files"]),
            "proposal prestate",
        )
        proposal_sha = digest(
            b"humanbrain.sequence-assembly-proposal.v1\0" + scalar.json_bytes(proposal)
        )
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
            replacement = scalar.hex_bytes(proposal["replacement_hex"])
            require(
                len(replacement) <= 128 and proposal["stop_reason"] == "eos",
                "accepted payload invalid",
            )
            proposed[path] = before[path][:start] + replacement + before[path][end:]
    require(
        scalar.files(execution["proposed_files"]) == proposed, "host substituted or repaired edit"
    )
    require(
        execution["before_files"] == row["before_files"]
        and execution["after_files"] == row["after_files"],
        "execution workspace binding",
    )
    require(type(execution["files_complete"]) is bool, "actual file completeness flag")
    directory = Path(execution["workspace"]).resolve()
    require(directory.is_relative_to(run.resolve()), "execution workspace outside run")
    for path, raw in scalar.files(row["after_files"]).items():
        saved = (directory / path).resolve()
        require(
            saved.is_relative_to(directory) and saved.read_bytes() == raw,
            "actual after-file changed",
        )
    for name in ("stdout", "stderr"):
        raw = scalar.hex_bytes(execution[name + "_hex"])
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
        execution["launch_record_sha256"] == digest(scalar.json_bytes(launch))
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
    original = scalar.files(episode["files"])
    protected = all(
        proposed.get(path) == original[path]
        and scalar.files(row["after_files"]).get(path) == original[path]
        for path in grade["protected_paths"]
    )
    behavior = (
        execution["returncode"] == grade["expected_returncode"]
        and execution["stdout_hex"] == grade["expected_stdout_hex"]
    )
    complete = (
        execution["files_complete"]
        and execution["capture_complete"]
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
    return {key: execution[key] for key in scalar.FEEDBACK_KEYS}


def check_execution(row, episode, run, launch):
    feedback = _check_saved_execution(row, episode, run, launch)
    execution, proposal = row["execution"], row["proposal"]
    policy = execution["public_driver_policy"]
    require(
        policy["schema"] == "persistent-language-action-public-driver-v1"
        and policy["protected_path"] == "main.py"
        and policy["path_comparison"] == "casefold exact relative path",
        "public driver policy schema",
    )
    require(
        execution["public_driver_policy_sha256"]
        == digest(b"humanbrain.public-driver-execution.v1\0" + scalar.json_bytes(policy)),
        "public policy receipt seal",
    )

    def driver(rows):
        matches = [r for r in rows if r["path"].casefold() == "main.py"]
        if len(matches) != 1:
            return None
        item = matches[0]
        raw = bytes.fromhex(item["content_hex"])
        return {"path": item["path"], "bytes": len(raw), "sha256": digest(raw)}

    blocked = (
        proposal is not None
        and proposal["operation"] == "SPLICE"
        and proposal["path"].casefold() == "main.py"
    )
    before, proposed, after = [
        driver(execution[k]) for k in ("before_files", "proposed_files", "after_files")
    ]
    require(
        policy["proposal_rejected"] is blocked
        and policy["before_driver"] == before
        and policy["proposed_driver"] == proposed
        and policy["after_driver"] == after
        and policy["driver_integrity"]
        is bool(execution["files_complete"] and before is not None and before == proposed == after)
        and policy["submitted_proposal_sha256"] == execution["proposal_sha256"],
        "actual public driver bytes",
    )
    if blocked:
        require(
            not execution["accepted"]
            and policy["delegated_proposal_sha256"] is None
            and execution["rejection"]
            == "ValueError: public driver main.py cannot be a SPLICE target",
            "protected proposed edit was applied",
        )
    else:
        require(
            policy["delegated_proposal_sha256"] == execution["proposal_sha256"]
            and policy["delegated_rejection"] == execution["rejection"],
            "delegated actual execution",
        )
    public = bool(
        policy["driver_integrity"]
        and execution["returncode"] == 0
        and not execution["timeout"]
        and execution["capture_complete"]
        and execution["files_complete"]
        and not execution["output_limit_exceeded"]
        and not execution["capture_errors"]
    )
    require(row["public_checks_passed"] is public, "public and private outcome separation")
    return feedback


def check_initialization(row, manifest):
    """Authenticate historical JSON/bytes; trust source-bound strict tensor validation."""
    require(
        row["schema"] == "persistent-language-action-initial-custody-v8"
        and row["status"] == "initialized",
        "V8 initializer receipt",
    )
    expected = {
        "bank_sha256": "b11b0e0517ef7939cf624e68dfdce05cdd75bc75aa8a81e400ddfd7974fcc67e",
        "prior_sha256": "1aaeb3b211872c3d6beb014ced2d7459f8b00520fb0bc462536fb5ec30cd38fc",
        "prior_head_sha256": "3bffe29937c840121fdd0e920bbfc3e54067389e0649d43530ce3df392d94656",
        "prior_boundary_sha256": "b5f529a842d06e9a5d7236787a8d8c57b6eb6aa8abc90622bb7d13448276c92a",
        "source_model_sha256": "cc49bf4fd4b50a6115fc182548f00a5edb459b40e0750a51810304c92d1bb5c6",
        "source_memory_sha256": "f4201afc7b2322994f7a9caafeffd02c570d7760ec6cb0f6f6613b3e737c825e",
        "prior_boundary_sequence": 221,
        "loaded_model_tensor_count": 22,
        "loaded_tape_tensor_count": 4,
        "loaded_tensor_bytes": 91_061_944,
        "serialized_payload_bytes": 91_080_533,
        "torch_load_calls": 1,
        "old_model_constructions": 0,
        "new_combined_model_constructions": 2,
        "initial_experience_per_route": 281,
        "initial_reader_experience_per_route": 214,
        "initial_factual_slots_per_route": 1614,
        "initial_event_owners_per_route": 398,
        "baseline_documents": 36,
        "baseline_demonstrations": 48,
    }
    require(all(row[k] == v for k, v in expected.items()), "actual V7 continuation contract")
    for key in (
        "reference_prediction_calls",
        "reference_teacher_calls",
        "source_ingestions",
        "source_observations",
        "answer_teacher_calls",
        "neural_prediction_calls",
        "execution_children",
        "unrun_v7_attempts_replayed",
    ):
        require(row[key] == 0, "initializer replayed historical experience")
    directory = Path(row["prior_path"]).resolve().parent
    require(
        directory == (ROOT / ".pytest-tmp/persistent-language-action-v7-run/checkpoints").resolve(),
        "selected historical checkpoint directory",
    )
    for path, identity in (
        (Path(row["bank_path"]), row["bank_sha256"]),
        (directory / "HEAD.json", row["prior_head_sha256"]),
        (directory / "boundary-000000221.json", row["prior_boundary_sha256"]),
        (Path(row["prior_path"]), row["prior_sha256"]),
    ):
        require(scalar.file_sha(path) == identity, "authenticated prior bytes changed")
    old = read_json(directory / "boundary-000000221.json")
    historical_config = read_json(directory / "config.json")
    historical_head = read_json(directory / "HEAD.json")
    require(
        historical_head["sha256"] == row["prior_boundary_sha256"]
        and scalar.file_sha(directory / "config.json") == old["config_sha256"]
        and historical_config["source_sha256"] == row["historical_source_sha256"],
        "original source closure under authenticated safe HEAD",
    )
    require(
        row["prior_host"] == old["host"]
        and row["prior_frontiers"] == old["frontiers"]
        and row["baseline_record"] == old["baseline"],
        "actual prior host/reference lineage",
    )
    baseline = row["baseline_record"]
    documents = [{"bytes_hex": value} for value in baseline["documents_hex"]]
    demonstrations = [
        {
            k: {"bytes_hex": v} if k in ("query", "old", "replacement") else v
            for k, v in demo.items()
        }
        for demo in baseline["demonstrations"]
    ]
    require(
        len(documents) == 36
        and len(demonstrations) == 48
        and sum(len(bytes.fromhex(d["bytes_hex"])) + 1 for d in documents) == 1614,
        "whole prior simple history",
    )
    require(
        [d["text_hex"] for d in manifest["continuation"]["prior_documents"]]
        == baseline["documents_hex"],
        "unselected prior context",
    )
    routes = row["initial_routes"]
    require(
        set(routes) == set(ROUTES[:2]) and routes["online"] == routes["learning_disabled"],
        "matched same-online initial numerical lineage",
    )
    for route in ROUTES[:2]:
        require(
            routes[route]["memory_sha256"] == row["source_memory_sha256"], "retained tape bytes"
        )
        require(
            routes[route]["model_tensor_count"] == 29
            and routes[route]["tape_tensor_bytes"] == 3_301_384
            and routes[route]["model_tensor_bytes"]
            == 87_760_560 + row["route_work"][route]["pristine_controller_strict_copy_bytes"],
            "declared retained/new numerical tensor accounting",
        )
        for key in ("model_sha256", "controller_sha256", "views_sha256", "self_wiring_sha256"):
            scalar.hash_value(routes[route][key])
        work_valid(row["route_work"][route])
        require(
            row["route_work"][route]["factorized_candidate_builds"] == 398
            and row["route_work"][route]["recognition_self_builds"] == 398,
            "one canonical derived-state rebuild per route",
        )
    work_valid(row["work"])
    bindings = manifest["prior_regression_bindings"]
    selected = bindings["routes"]
    h, byte_count, row_count, ordinals, found = hashlib.sha256(), 0, 0, Counter(), set()
    with safe_path(ROOT, bindings["path"]).open("rb") as handle:
        while raw := handle.readline(prior.ROW_LIMIT + 1):
            require(
                len(raw) <= prior.ROW_LIMIT and raw.endswith(b"\n"),
                "historical actual journal complete row",
            )
            h.update(raw)
            byte_count += len(raw)
            row_count += 1
            actual = json.loads(raw)
            route = actual["route"]
            episode_id = actual["episode_id"]
            if episode_id in selected[route] and actual["attempt_index"] == 1:
                expected_record = selected[route][episode_id]
                actual_record = {
                    "episode_id": episode_id,
                    "attempt_index": 1,
                    "route_ordinal": ordinals[route],
                    "attempt_sha256": digest(raw),
                    "correct": actual["correct"],
                    "observation": actual["observation"],
                }
                require(
                    expected_record == actual_record and (route, episode_id) not in found,
                    "actual observed V7 regression binding",
                )
                found.add((route, episode_id))
            ordinals[route] += 1
    require(
        h.hexdigest() == bindings["sha256"] == row["prior_attempts_sha256"]
        and byte_count == bindings["bytes_read"]
        and row_count == bindings["rows_read"] == 185
        and dict(ordinals) == bindings["route_attempts"]
        and len(found) == bindings["retained_observations"] == 42,
        "whole historical journal authentication without missing-case reconstruction",
    )
    return documents, demonstrations, dict(baseline["new_teacher_offers"]), routes


def metrics(compact, manifest, prior_outcomes=None):
    """Only observed denominators; exposed old examples and source asymmetry explicit."""
    result = {}
    for route in ROUTES:
        rows = [r for r in compact if r["route"] == route]
        first = {r["episode_id"]: r for r in rows if r["attempt_index"] == 1}
        groups = {}
        for name in (
            "acquisition",
            "transfer",
            "retention",
            "recovery",
            "first",
            "changed_form",
            "exact_delayed",
            "instruction_construction",
            "structural_combination",
        ):
            planned = [
                e
                for e in manifest["episodes"]
                if name in (e["phase"], e.get("encounter"), e.get("fresh_stratum"))
            ]
            groups[name] = {}
            for kind in ("reading", "edit"):
                members = [e for e in planned if e["kind"] == kind]
                actual = [first[e["episode_id"]] for e in members if e["episode_id"] in first]
                groups[name][kind] = {
                    "planned": len(members),
                    "observed": len(actual),
                    "missing": len(members) - len(actual),
                    "correct": sum(r["correct"] for r in actual),
                    "supported": sum(r["status"] != "unsupported" for r in actual),
                    "dependent": sum(bool(r["predecessor_episode_id"]) for r in actual),
                    "dependent_correct": sum(
                        r["correct"] and bool(r["predecessor_episode_id"]) for r in actual
                    ),
                }
        retained, encounters = [], []
        for episode in manifest["episodes"]:
            if episode["episode_id"] not in first:
                continue
            after = first[episode["episode_id"]]
            if episode["phase"] == "retention":
                route_parent = "simple" if route == "simple" else "online"
                before = manifest["prior_regression_bindings"]["routes"][route_parent][
                    episode["prior_episode_id"]
                ]
                retained.append(
                    {
                        "episode_id": episode["episode_id"],
                        "prior_route": route_parent,
                        "prior_episode_id": before["episode_id"],
                        "prior_attempt_sha256": before["attempt_sha256"],
                        "initial_correct": before["correct"],
                        "correct": after["correct"],
                        "same_observation": digest(encoded(before["observation"]))
                        == after["observation_sha256"],
                        "memory_used": after["memory_used"],
                        "intervening_current_attempts": rows.index(after),
                    }
                )
            if episode.get("encounter") in ("changed_form", "exact_delayed"):
                before = first[episode["same_session_origin_id"]]
                gap = rows.index(after) - rows.index(before) - 1
                same = before["observation_sha256"] == after["observation_sha256"]
                require(
                    gap == episode["intervening_attempts"]
                    and (episode["encounter"] != "exact_delayed" or same),
                    "actual revisit input/age",
                )
                encounters.append(
                    {
                        "episode_id": episode["episode_id"],
                        "encounter": episode["encounter"],
                        "initial_correct": before["correct"],
                        "correct": after["correct"],
                        "intervening_attempts": gap,
                        "same_observation": same,
                        "requirement_changed": bool(episode.get("requirement_changed")),
                        "initial_memory_used": before["memory_used"],
                        "memory_used": after["memory_used"],
                    }
                )
        recovery = [e for e in manifest["episodes"] if e["phase"] == "recovery"]
        chains = [[r for r in rows if r["episode_id"] == e["episode_id"]] for e in recovery]
        seen = [c for c in chains if c]
        complete = [
            c for c, e in zip(chains, recovery, strict=True) if len(c) == e["attempt_limit"]
        ]
        pairs = [(a, b) for c in seen for a, b in pairwise(c)]
        result[route] = {
            "groups": groups,
            "attempts": len(rows),
            "retention": retained,
            "same_session_encounters": encounters,
            "retained": sum(r["initial_correct"] and r["correct"] for r in retained),
            "forgotten": sum(r["initial_correct"] and not r["correct"] for r in retained),
            "improved": sum(not r["initial_correct"] and r["correct"] for r in retained),
            "project_chains": [r for r in rows if r["project_id"] is not None],
            "execution_outcomes": dict(
                Counter(
                    r.get("execution_outcome", r.get("behavior"))
                    for r in rows
                    if r["kind"] == "edit"
                )
            ),
            "public_checks_passed": sum(r.get("public_checks_passed", False) for r in rows),
            "recovery": {
                "planned_episodes": len(chains),
                "observed_episodes": len(seen),
                "missing_episodes": len(chains) - len(seen),
                "completed_episodes": len(complete),
                "observed_attempts": sum(map(len, seen)),
                "first_correct": sum(c[0]["correct"] for c in seen),
                "observed_first_failures": sum(not c[0]["correct"] for c in seen),
                "completed_first_failures": sum(not c[0]["correct"] for c in complete),
                "ever_recovered": sum(
                    not c[0]["correct"] and any(r["correct"] for r in c[1:]) for c in seen
                ),
                "later_correct_attempts": sum(r["correct"] for c in seen for r in c[1:]),
                "terminal_correct": sum(c[-1]["correct"] for c in complete),
                "terminal_recovered": sum(
                    not c[0]["correct"] and c[-1]["correct"] for c in complete
                ),
                "feedback_consuming_attempts": sum(r["feedback_consumed"] for c in seen for r in c),
                "identical_consecutive_proposals": sum(
                    a["proposal_signature"] is not None
                    and a["proposal_signature"] == b["proposal_signature"]
                    for a, b in pairs
                ),
            },
            "memory_scope": "neuronal admitted facts; simple retains every whole source",
        }
    return result


def check_checkpoints(run, sources, manifest_sha, rows_by_kind, partial, resolved):
    """Independent immutable-byte and scalar frontier audit; no tensor decoder."""
    directory = run / "checkpoints"
    if not directory.exists():
        require(partial, "checkpoint store absent")
        return {
            "head": None,
            "boundaries": 0,
            "dirty": True,
            "reason": "initialization store absent",
        }
    config = read_json(directory / "config.json")
    require(
        config["schema"] == "persistent-session-checkpoint-config-v1"
        and config["manifest_sha256"] == manifest_sha
        and config["source_sha256"] == sources,
        "checkpoint source/manifest binding",
    )
    require(
        Path(config["source_root"]).resolve() == ROOT
        and Path(config["evidence_root"]).resolve() == run,
        "checkpoint configured roots",
    )
    require(
        scalar.file_sha(directory / "manifest.json") == manifest_sha, "checkpoint manifest bytes"
    )
    config_sha = scalar.file_sha(directory / "config.json")
    intents = {}
    for path in sorted(directory.glob("intent-*.json")):
        value = read_json(path)
        require(
            value["sequence"] not in intents and value["config_sha256"] == config_sha,
            "intent identity",
        )
        intents[value["sequence"]] = (value, scalar.file_sha(path))
    require(
        0 in intents and intents[0][0]["kind"] == "initialize", "pre-initialization durable intent"
    )
    if not (directory / "HEAD.json").exists():
        require(partial, "complete run has no safe checkpoint")
        return {"head": None, "boundaries": 0, "dirty": True, "intents": len(intents)}
    head = read_json(directory / "HEAD.json")
    require(head["schema"] == "persistent-session-checkpoint-head-v1", "HEAD schema")
    previous, previous_hash, hashes, last = None, None, {}, None
    boundary_events = []
    for sequence in range(head["sequence"] + 1):
        path = directory / f"boundary-{sequence:09d}.json"
        boundary = read_json(path)
        boundary_sha256 = scalar.file_sha(path)
        require(
            boundary["schema"] == "persistent-session-checkpoint-boundary-v1"
            and boundary["sequence"] == sequence
            and boundary["previous_sha256"] == previous_hash
            and boundary["config_sha256"] == config_sha
            and boundary["intent_sha256"] == intents[sequence][1],
            "immutable boundary lineage",
        )
        intent = intents[sequence][0]
        require(sequence < len(resolved), "HEAD exceeds resolved journal transactions")
        expected = resolved[sequence]
        require(
            intent["kind"] == expected["kind"] and intent["cursor"] == expected["before_cursor"],
            "checkpoint intended transaction differs",
        )
        host = boundary["host"]
        require(
            host["cursor"] == expected["cursor"]
            and host["counters"] == expected["counters"]
            and host["projects"] == expected["projects"]
            and host["current"] == expected["current"],
            "checkpoint cursor/counter/actual workspace mismatch",
        )
        if previous is None:
            for name, initial_sha256 in host["initial_bindings"].items():
                require(
                    scalar.file_sha(safe_path(run, name)) == initial_sha256,
                    "checkpoint initial evidence",
                )
        else:
            require(
                host["initial_bindings"] == previous["host"]["initial_bindings"]
                and host["budget_seconds"] == previous["host"]["budget_seconds"]
                and host["budget_mib"] == previous["host"]["budget_mib"],
                "checkpoint budget/bindings reset",
            )
        require(boundary["baseline"] == expected["baseline"], "checkpoint actual reference state")
        require(set(boundary["routes"]) == set(ROUTES[:2]), "checkpoint route set")
        require(intent["previous_sha256"] == previous_hash, "intent predecessor")
        if previous is not None:
            require(
                intent["cursor"] == previous["host"]["cursor"], "intent resumes another frontier"
            )
        for route, value in boundary["routes"].items():
            require(route in ROUTES[:2], "checkpoint neuronal route")
            payload = safe_path(directory, value["payload"])
            if payload not in hashes:
                hashes[payload] = scalar.file_sha(payload)
            require(
                hashes[payload] == value["sha256"] and payload.stat().st_size == value["bytes"],
                "immutable tensor payload bytes",
            )
            for key in ("model_sha256", "memory_sha256"):
                scalar.hash_value(value[key])
            require(
                value["model_sha256"] == expected["hashes"][route]["model"]
                and value["memory_sha256"] == expected["hashes"][route]["memory"]
                and value["reader_experience"] == 214 + expected["free"][route]
                and value["experience"] == 281 + expected["actions"][route],
                "checkpoint actual learner lineage/counters",
            )
        frontier = boundary["frontiers"]
        for item in frontier["journals"]:
            kind = item["kind"]
            require(item["path"] == kind + ".jsonl", "unexpected journal segment")
            require(0 <= item["rows"] <= len(rows_by_kind[kind]), "boundary beyond durable journal")
            entry = rows_by_kind[kind][item["rows"] - 1] if item["rows"] else None
            require(
                (0 if entry is None else entry["end"]) == item["bytes"]
                and (digest(b"") if entry is None else entry["prefix_sha256"]) == item["sha256"],
                "journal prefix hash",
            )
        for item in frontier["execution_files"]:
            path = safe_path(run, item["path"])
            if path not in hashes:
                hashes[path] = scalar.file_sha(path)
            require(
                hashes[path] == item["sha256"] and path.stat().st_size == item["bytes"],
                "execution frontier bytes",
            )
        for name in frontier["execution_directories"]:
            require(safe_path(run, name).is_dir(), "lost actual execution directory")
        work_valid(boundary["work"])
        require(
            type(host["elapsed_seconds"]) in (int, float)
            and math.isfinite(host["elapsed_seconds"])
            and host["elapsed_seconds"]
            >= (0 if previous is None else previous["host"]["elapsed_seconds"]),
            "checkpoint cumulative clock reset",
        )
        boundary_events.append(
            {
                "sequence": sequence,
                "kind": intent["kind"],
                "cursor": intent["cursor"],
                "host_cursor": boundary["host"]["cursor"],
                "sha256": boundary_sha256,
            }
        )
        previous, previous_hash, last = boundary, boundary_sha256, boundary
    require(
        head["manifest"] == f"boundary-{head['sequence']:09d}.json"
        and head["sha256"] == previous_hash,
        "atomic HEAD does not select final published boundary",
    )
    extra_intents = sorted(i for i in intents if i > head["sequence"])
    final_rows = {x["kind"]: x["rows"] for x in last["frontiers"]["journals"]}
    beyond = {k: len(v) - final_rows[k] for k, v in rows_by_kind.items()}
    beyond_bytes = {
        item["kind"]: (run / item["path"]).stat().st_size - item["bytes"]
        for item in last["frontiers"]["journals"]
    }
    expected_dirs = set(last["frontiers"]["execution_directories"])
    actual_dirs = {
        p.relative_to(run).as_posix() for p in (run / "executions").rglob("*") if p.is_dir()
    }
    expected_files = {r["path"] for r in last["frontiers"]["execution_files"]}
    actual_files = {
        p.relative_to(run).as_posix() for p in (run / "executions").rglob("*") if p.is_file()
    }
    dirty = bool(
        extra_intents
        or any(beyond.values())
        or any(beyond_bytes.values())
        or actual_dirs != expected_dirs
        or actual_files != expected_files
    )
    require(partial or not dirty, "complete session has work beyond safe HEAD")
    # Invoke only the existing byte/JSON inspector, never restore_into or a
    # learner factory. This retains the exact current V8 subclass source guards.
    from experiments.r4_integrated.persistent_language_action_v8_checkpoint import (
        PersistentLanguageActionV8Checkpoint,
    )

    inspect_started = time.perf_counter()
    with PersistentLanguageActionV8Checkpoint.open(directory, config_sha256=config_sha) as store:
        checked = store.inspect(require_clean=not partial, fresh_payloads=True)
        require(
            checked["boundary_sha256"] == previous_hash and (checked["status"] == "dirty") is dirty,
            "independent frontier disagrees with full V8 byte inspector",
        )
        inspector_work = dict(store.work)
    inspector_seconds = time.perf_counter() - inspect_started
    return {
        "head": head,
        "head_sha256": scalar.file_sha(directory / "HEAD.json"),
        "boundaries": len(boundary_events),
        "boundary_events": boundary_events,
        "dirty": dirty,
        "outstanding_intents": extra_intents,
        "journal_rows_beyond_head": beyond,
        "journal_bytes_beyond_head": beyond_bytes,
        "execution_files_beyond_head": sorted(actual_files - expected_files),
        "execution_directories_beyond_head": sorted(actual_dirs - expected_dirs),
        "last_host": last["host"],
        "last_routes": last["routes"],
        "last_baseline": last["baseline"],
        "unique_hashed_payload_and_execution_files": len(hashes),
        "checkpoint_decodes": 0,
        "v8_fresh_inspector_work": inspector_work,
        "v8_fresh_inspector_seconds": inspector_seconds,
    }

# Dispatcher and checkpoint/frontier audit follow below. No module import runs a review.


def review(run, launch_path, manifest_sha, *, partial=False):
    """Audit one durable chronological prefix without reconstructing missing exposures."""
    started = time.perf_counter()
    run = Path(run).resolve()
    require(manifest_sha == MANIFEST_SHA and scalar.file_sha(run / "manifest.json") == manifest_sha,
            "unselected or changed frozen V8 manifest")
    manifest, monitor = read_json(run / "manifest.json"), read_json(launch_path)
    require(manifest["schema"] == "persistent-language-action-stream-v8"
            and len(manifest["episodes"]) == 48
            and sum(e["attempt_limit"] for e in manifest["episodes"]) == 64,
            "fixed complete V8 schedule")
    require(monitor["status"] == ("failed" if partial else "passed"), "run monitor status")
    command = monitor["command"]
    require(command[1:5] == ["-X", "utf8", "-m",
                            "experiments.r4_integrated.persistent_language_action_v8"]
            and Path(command[command.index("--output") + 1]).resolve() == run
            and command[command.index("--manifest-sha") + 1] == manifest_sha,
            "actual monitored V8 invocation")
    sources = read_json(run / "sources.json")
    expected = {p.relative_to(ROOT).as_posix()
                for directory in (ROOT / "src/humanbrain", ROOT / "experiments/r4_integrated")
                for p in directory.glob("*.py")} - SAVED_ONLY_SOURCES
    require(set(sources) == expected, "complete selected source closure")
    for name, identity in sources.items():
        require(scalar.file_sha(safe_path(ROOT, name)) == identity, "changed source: " + name)
    saved_tool_sources = {name: scalar.file_sha(ROOT / name) for name in SAVED_ONLY_SOURCES
                          if (ROOT / name).exists()}
    if partial:
        require(not (run / "completion.json").exists(), "partial audit of completed evidence")
    else:
        completion = read_json(run / "completion.json")
        require(completion["schema"] == "persistent-language-action-completion-v8"
                and completion["status"] == "completed", "completion marker")
        require(set(completion["files"]) == {
            p.relative_to(run).as_posix() for p in run.rglob("*")
            if p.is_file() and p.name != "completion.json"}, "complete inventory coverage")
        for name, identity in completion["files"].items():
            require(scalar.file_sha(safe_path(run, name)) == identity, "changed evidence: " + name)
    indexed, frontiers = {}, {}
    for kind in ("attempts", "teachers", "ingests"):
        indexed[kind], frontiers[kind] = journal(run / (kind + ".jsonl"), partial=partial)
    telemetry, telemetry_frontier = journal(run / "checkpoints.jsonl", partial=partial)
    counts = {r: dict.fromkeys(COUNTERS, 0) for r in ROUTES}
    consumed = dict.fromkeys(indexed, 0)
    compact, resolved = [], []
    costs = {r: {"work": {k: {} for k in ("source", "free", "teacher", "ack")},
                 "elapsed_seconds": dict.fromkeys(("source", "attempt", "teacher"), 0.0),
                 "included_checkpoint_begin_seconds": 0.0} for r in ROUTES}
    stochastic = {r: {"ticks": Counter(), "draws": 0,
                      "by_category": new_symbol_exposures()} for r in ROUTES[:2]}
    last_draw_rng = dict.fromkeys(ROUTES[:2])
    source_offers = dict.fromkeys(ROUTES, 0)
    free, actions = dict.fromkeys(ROUTES[:2], 0), dict.fromkeys(ROUTES[:2], 0)
    used = dict.fromkeys(ROUTES, 1614)
    baselines = {r: [[0.0, 0], [0.0, 0], [0.0, 0]] for r in ROUTES[:2]}
    projects, instances, responses = {r: {} for r in ROUTES}, {}, set()
    cursor = {"episode_index": 0, "route_index": 0, "attempt_index": 1, "append_done": False}
    actual_current, pending = None, None

    def add_work(destination, values, prefix=""):
        work_valid(values)
        for key, value in values.items():
            if isinstance(value, dict):
                add_work(destination, value, prefix + key + ".")
            else:
                name = prefix + key
                destination[name] = destination.get(name, 0) + value

    def add_cost(row, category, clock):
        ledger = costs[row["route"]]
        add_work(ledger["work"][category], row["work"])
        ledger["elapsed_seconds"][clock] += prior.elapsed_value(row["elapsed_seconds"])
        if category == "free":
            add_work(ledger["work"]["ack"], row.get("ack_work", {}))
            ledger["included_checkpoint_begin_seconds"] += prior.elapsed_value(
                row["included_checkpoint_begin_seconds"])

    def add_trajectory(route, trajectory):
        require(last_draw_rng[route] is None or trajectory["first_rng"] == last_draw_rng[route],
                "actual cross-interaction draw RNG continuity")
        last_draw_rng[route] = trajectory["rng"]
        stochastic[route]["ticks"].update(trajectory["ticks"])
        stochastic[route]["draws"] += trajectory["ordinal"]
        for category, source in trajectory["symbol_exposures"].items():
            destination = stochastic[route]["by_category"][category]
            for key in ("ticks", "draws"):
                destination[key] += source[key]
            for key in ("sampled_modes", "sampled_gates"):
                destination[key].update(source[key])
            if source["observed_symbols"] is not None:
                for kind, values in source["observed_symbols"].items():
                    destination["observed_symbols"][kind].update(values)

    def controller_transition(route, before, after, *, advanced):
        require(controllers[route] is None or before == controllers[route], "controller published-state hash chain")
        for value in (*before.values(), *after.values()):
            scalar.hash_value(value)
        require(set(before) == set(after) == {"rng_sha256", "slow_sha256"}, "controller hash keys")
        require((after["rng_sha256"] != before["rng_sha256"]) is advanced,
                "actual resolved RNG advancement")
        require(route == "online" or after["slow_sha256"] == before["slow_sha256"],
                "disabled slow controller publication")
        controllers[route] = dict(after)

    initialized = (run / "initialization.json").exists()
    if not initialized:
        require(partial and not any(indexed.values()), "task rows without initialization")
        pending = {"kind": "initialization", "cursor": dict(cursor)}
        current, controllers = {}, {}
    else:
        initialization = read_json(run / "initialization.json")
        documents, demos, offers, initial_routes = check_initialization(initialization, manifest)
        current = {r: {k: initial_routes[r][k + "_sha256"] for k in ("model", "memory")}
                   for r in ROUTES[:2]}
        current["simple"] = {"model": scalar.plain_digest(demos),
                             "memory": scalar.plain_digest(documents)}
        # Initialization seals the complete controller. Split RNG/slow hashes
        # start at the first actual row, in a distinct digest domain.
        controllers = dict.fromkeys(ROUTES[:2])
        initial_hashes = json.loads(encoded(current))
        launch = read_json(run / "launch-record.json")
        for name, identity in launch["artifacts"].items():
            require(scalar.file_sha(name) == identity, "qualified execution artifact")

        def record_boundary(kind, before):
            baseline = {"documents_hex": [d["bytes_hex"] for d in documents],
                        "demonstrations": [{k: v["bytes_hex"] if isinstance(v, dict) else v
                                            for k, v in d.items()} for d in demos],
                        "new_teacher_offers": offers}
            resolved.append(json.loads(encoded({
                "kind": kind, "before_cursor": before, "cursor": cursor, "counters": counts,
                "projects": projects, "current": actual_current, "hashes": current,
                "free": free, "actions": actions, "baseline": baseline,
            })))

        record_boundary("initialize", dict(cursor))
        docs = {d["document_id"]: d for d in manifest["documents"]}
        for ei, episode in enumerate(manifest["episodes"]):
            if pending:
                break
            for ri, route in enumerate(ROUTES):
                if pending:
                    break
                enabled, neural = route == "online", route != "simple"
                if episode["append_document_ids"]:
                    if consumed["ingests"] == len(indexed["ingests"]):
                        pending = {"kind": "append", "cursor": dict(cursor)}
                        break
                    row = read_row(indexed["ingests"][consumed["ingests"]])
                    doc = docs[episode["append_document_ids"][0]]
                    raw = bytes.fromhex(doc["text_hex"])
                    admitted = neural and doc["neural_tape_admission"]
                    require(row["schema"] == "persistent-language-action-ingest-v8"
                            and (row["route"], row["episode_id"], row["document_ids"], row["documents"])
                            == (route, episode["episode_id"], episode["append_document_ids"], [doc])
                            and row["raw_bytes"] == len(raw) and row["eos"] == 1
                            and row["source_kind"] == doc["source_kind"]
                            and row["neural_tape_admission"] is doc["neural_tape_admission"]
                            and row["fact_admitted"] is admitted and row["writes_enabled"] is admitted
                            and row["reference_retained"] is (not neural)
                            and row["learning_enabled"] is (route != "learning_disabled"),
                            "fixed raw source admission/exposure")
                    if neural:
                        require(row["model_before_sha256"] == current[route]["model"]
                                and row["memory_before_sha256"] == current[route]["memory"]
                                and row["memory_used_before"] == used[route]
                                and row["memory_used_after"] == used[route] + (len(raw) + 1 if admitted else 0),
                                "source neuronal state chronology")
                        trajectory = check_source(row["source_feedback"], raw, enabled, admitted,
                                                  baselines[route])
                        require(row["source_feedback"]["memory_before_sha256"] == current[route]["memory"]
                                and row["source_feedback"]["memory_after_sha256"] == row["memory_after_sha256"],
                                "source feedback tape custody")
                        controller_transition(route,
                            {k: row["controller_" + n + "_before_sha256"]
                             for k, n in (("rng_sha256", "rng"), ("slow_sha256", "slow"))},
                            {k: row["controller_" + n + "_after_sha256"]
                             for k, n in (("rng_sha256", "rng"), ("slow_sha256", "slow"))}, advanced=True)
                        require(admitted or row["memory_after_sha256"] == current[route]["memory"],
                                "transient source allocated factual storage")
                        current[route] = {"model": row["model_after_sha256"], "memory": row["memory_after_sha256"]}
                        for value in current[route].values():
                            scalar.hash_value(value)
                        used[route] = row["memory_used_after"]
                        source_offers[route] += len(raw) + 1
                        add_trajectory(route, trajectory)
                    else:
                        require(row["source_feedback"] is None and all(row[k] is None for k in (
                            "model_before_sha256", "model_after_sha256", "memory_before_sha256",
                            "memory_after_sha256", "memory_used_before", "memory_used_after",
                            "controller_rng_before_sha256", "controller_rng_after_sha256",
                            "controller_slow_before_sha256", "controller_slow_after_sha256")),
                            "reference claimed neural source state")
                        documents.append({"bytes_hex": raw.hex()})
                        current[route]["memory"] = scalar.plain_digest(documents)
                        used[route] += len(raw) + 1
                    add_cost(row, "source", "source")
                    counts[route]["ingest_documents"] += 1
                    counts[route]["ingest_raw_bytes"] += len(raw)
                    counts[route]["ingest_eos"] += 1
                    consumed["ingests"] += 1
                    before = dict(cursor)
                    cursor["append_done"] = True
                    record_boundary("append", before)
                    del row
                project, feedback_input = episode["project_id"], None
                if episode["workspace_lifecycle"] == "carry_actual_project":
                    preceding = projects[route][project]
                    require(preceding["episode_id"] == episode["predecessor_episode_id"], "actual project predecessor")
                    files, feedback_input = preceding["files"], preceding["feedback"]
                elif episode["phase"] == "retention":
                    parent_route = "simple" if route == "simple" else "online"
                    old = manifest["prior_regression_bindings"]["routes"][parent_route][episode["prior_episode_id"]]
                    files, feedback_input = old["observation"]["files"] or [], old["observation"]["feedback"]
                else:
                    files = episode["files"]
                for ai in range(1, episode["attempt_limit"] + 1):
                    if consumed["attempts"] == len(indexed["attempts"]):
                        pending = {"kind": "attempt", "cursor": dict(cursor)}
                        break
                    entry = indexed["attempts"][consumed["attempts"]]
                    row = read_row(entry)
                    require(row["schema"] == "persistent-language-action-attempt-v8"
                            and (row["route"], row["episode_id"], row["attempt_index"])
                            == (route, episode["episode_id"], ai)
                            and all(row[k] == episode[k] for k in (
                                "phase", "kind", "request_hex", "project_id", "predecessor_episode_id", "workspace_lifecycle"))
                            and row["before_files"] == files, "actual fixed attempt schedule")
                    public = {"schema": "sequence-assembly-raw-observation-v1", "request_hex": episode["request_hex"],
                              "files": files if episode["kind"] == "edit" else None, "feedback": feedback_input}
                    require(row["status"] in ("completed", "unsupported"), "attempt status")
                    if row["status"] == "completed":
                        require(row["observation"] == public, "complete public observation, no grade input")
                    else:
                        require(row["prediction"] is None and row["proposal"] is None and row["answer_hex"] is None,
                                "unsupported input predicted")
                        require(row["observation"] == {"request_hex": episode["request_hex"],
                                                       "files": files, "feedback": feedback_input},
                                "unsupported input dropped actual raw state")
                    require(row["observation_record_sha256"] == digest(encoded(row["observation"]))
                            and row["model_before_sha256"] == current[route]["model"]
                            and row["memory_before_sha256"] == row["memory_after_sha256"] == current[route]["memory"]
                            and row["memory_used"] == used[route], "attempt state and observation custody")
                    if route in instances:
                        require(row["model_instance_id"] == instances[route], "actor replaced")
                    else:
                        require(row["model_instance_id"] not in instances.values(), "aliased acting routes")
                        instances[route] = row["model_instance_id"]
                    trajectory = prediction = None
                    if row["prediction"] is not None:
                        if not neural:
                            prior.check_reference(row, documents, demos)
                        else:
                            prediction = row["prediction"] if episode["kind"] == "reading" else row["prediction"]["payload"]
                            trajectory = check_prediction(prediction, public, used[route], responses)
                            require(prediction["memory_sha256"] == current[route]["memory"], "prediction tape")
                            free[route] += 1
                            if episode["kind"] == "reading":
                                require(row["answer_hex"] == prediction["answer_hex"], "reading substitution")
                            else:
                                scalar.check_heads(row["prediction"], row["proposal"], files)
                                require(row["prediction"]["model_sha256"] == current[route]["model"], "action actor hash")
                                actions[route] += 1
                    require(row["feedback_consumed"] is (feedback_input is not None and row["prediction"] is not None),
                            "actual feedback consumption")
                    reward, outcome = None, None
                    if episode["kind"] == "edit":
                        feedback_input = check_execution(row, episode, run, launch)
                        correct = row["status"] != "unsupported" and row["grade"]["task_success"]
                        ex, proposal = row["execution"], row["proposal"]
                        failed = (not ex["accepted"] or ex["returncode"] != 0 or ex["timeout"]
                                  or not ex["capture_complete"] or not ex["files_complete"] or ex["output_limit_exceeded"])
                        reward = -int(failed and proposal is not None and proposal["operation"] != "STOP")
                        if proposal is None:
                            outcome = "no_proposal"
                        elif not ex["accepted"]:
                            outcome = "rejected_proposal"
                        elif proposal["operation"] == "STOP":
                            outcome = "correct_stop" if correct else "incorrect_stop"
                        elif not correct:
                            outcome = "incorrect_accepted_splice"
                        else:
                            outcome = ("accepted_correct_splice_with_change" if ex["proposed_files"] != ex["before_files"]
                                       else "accepted_correct_splice_without_change")
                        counts[route]["executions"] += 1
                    else:
                        require(row["execution"] is None and row["after_files"] == files, "reading performed action")
                        correct = row["status"] != "unsupported" and row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                    require(row["correct"] is bool(correct) and row["execution_outcome"] == outcome, "actual private outcome")
                    signature = None if row["proposal"] is None else digest(encoded(
                        {k: v for k, v in row["proposal"].items() if k != "workspace_sha256"}))
                    require(row["proposal_signature"] == signature, "actual proposal signature")
                    require(row["model_after_sha256"] == current[route]["model"]
                            if not neural or episode["teacher"] is not None or prediction is None else True,
                            "unresolved or unsupported response changed model")
                    if neural:
                        controller_transition(route, row["controller_before"], row["controller_after"],
                                              advanced=prediction is not None and episode["teacher"] is None)
                        if trajectory is not None and episode["teacher"] is None:
                            check_resolution(row["controller_resolution"], trajectory, episode["kind"], enabled,
                                             baselines[route], reward)
                            add_trajectory(route, trajectory)
                    current[route]["model"] = row["model_after_sha256"]
                    counts[route]["attempts"] += 1
                    compact.append({k: row[k] for k in (
                        "route", "episode_id", "phase", "kind", "attempt_index", "correct", "status", "project_id",
                        "predecessor_episode_id", "workspace_lifecycle", "memory_used", "feedback_consumed",
                        "proposal_signature", "public_checks_passed", "execution_outcome")}
                        | {"observation_sha256": row["observation_record_sha256"]})
                    consumed["attempts"] += 1
                    add_cost(row, "free", "attempt")
                    if row["execution"] is not None and not row["execution"]["files_complete"]:
                        require(partial, "complete session has incomplete actual file capture")
                        if trajectory is not None and episode["teacher"] is not None:
                            add_trajectory(route, trajectory)
                        pending = {"kind": "incomplete_execution_after_durable_attempt", "cursor": dict(cursor),
                                   "attempt_sha256": entry["sha256"],
                                   "scope": "captured subset retained; no safe continuation or invented missing files"}
                        break
                    if episode["teacher"] is not None:
                        if consumed["teachers"] == len(indexed["teachers"]):
                            if trajectory is not None:
                                add_trajectory(route, trajectory)
                            pending = {"kind": "correction_after_durable_attempt", "cursor": dict(cursor),
                                       "attempt_sha256": entry["sha256"]}
                            break
                        taught = read_row(indexed["teachers"][consumed["teachers"]])
                        target_record = episode["teacher"]
                        require(taught["schema"] == "persistent-language-action-teacher-v8"
                                and (taught["route"], taught["episode_id"], taught["teacher"], taught["attempt_sha256"])
                                == (route, episode["episode_id"], target_record, entry["sha256"])
                                and taught["learning_enabled"] is (route != "learning_disabled")
                                and taught["model_before_sha256"] == current[route]["model"]
                                and taught["memory_before_sha256"] == taught["memory_after_sha256"] == current[route]["memory"],
                                "causal actual correction identity/state")
                        if not neural:
                            label = "reading" if episode["kind"] == "reading" else target_record["operation"]
                            offers[label] += 1
                            require(taught["reader_offers"] == taught["action_labels"] == 0 and not taught["work"],
                                    "reference neural correction")
                            if label == "SPLICE":
                                scalar.append_demonstration(demos, target_record, row, scalar.similarity_query(public))
                                require(taught["publication"]["stored"] is True
                                        and taught["publication"]["demonstration_index"] == len(demos) - 1,
                                        "reference supplied demonstration")
                            else:
                                require(taught["publication"]["stored"] is False, "reference stored reading/STOP target")
                            require(taught["model_after_sha256"] == scalar.plain_digest(demos), "reference correction state")
                        else:
                            require(prediction is not None and trajectory is not None, "correction without actual response")
                            target = bytes.fromhex(target_record.get("target_hex", target_record.get("replacement_hex")))
                            n = check_teacher_feedback(taught["reader_feedback"], prediction, target, used[route],
                                trajectory, enabled, baselines[route], episode["kind"], reward)
                            require(taught["reader_offers"] == n and taught["action_labels"] == (3 if episode["kind"] == "edit" else 0),
                                    "actual correction offer/label counts")
                            require(taught["controller_resolution"] == taught["reader_feedback"]["controller"],
                                    "single controller correction publication")
                            publication = taught["publication"]
                            require(publication["address"] is False and publication["legacy_codec"] is False
                                    and type(publication["output"]) is bool and (enabled or not publication["output"]),
                                    "held old reader/control gate")
                            if episode["kind"] == "edit":
                                require(set(publication["actions"]) == {"start", "end", "operation"}
                                        and all(type(v) is bool and (enabled or not v) for v in publication["actions"].values()),
                                        "motor correction gate")
                            controller_transition(route, taught["controller_before"], taught["controller_after"], advanced=True)
                            add_trajectory(route, trajectory)
                        scalar.hash_value(taught["model_after_sha256"])
                        current[route]["model"] = taught["model_after_sha256"]
                        counts[route]["teacher_occasions"] += 1
                        counts[route]["reader_offers"] += taught["reader_offers"]
                        counts[route]["action_labels"] += taught["action_labels"]
                        consumed["teachers"] += 1
                        add_cost(taught, "teacher", "teacher")
                        del taught
                    files = row["after_files"]
                    actual_current = {"route": route, "episode_id": episode["episode_id"], "files": files, "feedback": feedback_input}
                    if project:
                        projects[route][project] = {k: v for k, v in actual_current.items() if k != "route"}
                    before = dict(cursor)
                    cursor = ({"episode_index": ei, "route_index": ri, "attempt_index": ai + 1, "append_done": True}
                              if ai < episode["attempt_limit"] else
                              {"episode_index": ei + int(ri == 2), "route_index": (ri + 1) % 3,
                               "attempt_index": 1, "append_done": False})
                    record_boundary("attempt", before)
                    del row
    require(all(consumed[k] == len(indexed[k]) for k in indexed), "unplanned durable rows beyond chronological prefix")
    require(partial or pending is None, "complete run omitted scheduled outcomes")
    checkpoint = check_checkpoints(run, sources, manifest_sha, indexed, partial, resolved)
    if not partial:
        require(consumed == {"attempts": 192, "teachers": 6, "ingests": 72}
                and checkpoint["boundaries"] == 265
                and all(used[r] == manifest["counts"]["occupied_memory_slots"] for r in ROUTES[:2])
                and all(source_offers[r] == manifest["counts"]["source_next_byte_teacher_symbols"] for r in ROUTES[:2]),
                "complete V8 exposure scope")
        report = read_json(run / "report.json")
        require(report["schema"] == "persistent-language-action-report-v8" and report["status"] == "completed"
                and report["counts"] == consumed and report["actual_executions"] == 132
                and report["initial_hashes"] == initial_hashes
                and report["final_hashes"] == current
                and report["final_reader_experience"] == {r: 214 + free[r] for r in free}
                and report["final_model_experience"] == {r: 281 + actions[r] for r in actions}
                and report["final_controller"] == controllers, "complete report state/counters")
        require(monitor["elapsed_seconds"] <= 3600 and monitor["observed_tree_peak_sum_mib"] <= 4096,
                "complete monitored budget")
    telemetry_counts, telemetry_seconds, previous_work = Counter(), Counter(), {}
    for entry in telemetry:
        item = read_row(entry)
        require(item["schema"] == "persistent-language-action-checkpoint-work-v8", "checkpoint telemetry schema")
        telemetry_counts[item["phase"]] += 1
        telemetry_seconds[item["phase"]] += prior.elapsed_value(item["elapsed_seconds"])
        work_valid(item["cumulative_work"])
        require(all(item["cumulative_work"].get(k, 0) >= v for k, v in previous_work.items()), "checkpoint work reset")
        if "work_delta" in item:
            require(item["work_delta"] == {k: v - previous_work.get(k, 0) for k, v in item["cumulative_work"].items()},
                    "complete checkpoint phase accounting")
        previous_work = item["cumulative_work"]
    if not partial:
        require(telemetry_counts == {"create": 1, "begin": 264, "capture": 265, "commit": 265}, "checkpoint complete phase counts")
    result_metrics = metrics(compact, manifest)
    if not partial:
        for route in ROUTES:
            actual, saved = result_metrics[route], report["metrics"][route]
            for key in ("retained", "forgotten", "improved", "public_checks_passed"):
                require(saved[key] == actual[key], "independent report metric: " + key)
            require(Counter(saved["execution_outcomes"]) == Counter(actual["execution_outcomes"]),
                    "independent action outcome counts")
            for key in ("retention", "same_session_encounters", "project_chains"):
                require(len(saved[key]) == len(actual[key]), "report detail count")
                for left, right in zip(saved[key], actual[key], strict=True):
                    require(all(left[name] == right[name] for name in set(left) & set(right)),
                            "report observed detail")
            require(saved["same_session_exact_retained"] == sum(
                r["encounter"] == "exact_delayed" and r["initial_correct"] and r["correct"]
                for r in actual["same_session_encounters"]), "report exact repeat retention")
            for detail in saved["same_session_encounters"]:
                episode = next(e for e in manifest["episodes"] if e["episode_id"] == detail["episode_id"])
                require(detail["origin"] == episode["same_session_origin_id"], "report exact encounter origin")
            for phase in ("acquisition", "transfer", "retention", "recovery"):
                for kind in ("reading", "edit"):
                    group = actual["groups"][phase][kind]
                    row = saved["phases"][phase][kind]
                    require(row["episodes"] == group["observed"] and row["first_correct"] == group["correct"]
                            and row["attempts"] == sum(r["phase"] == phase and r["kind"] == kind for r in compact if r["route"] == route)
                            and row["unsupported"] == sum(r["phase"] == phase and r["kind"] == kind and r["status"] == "unsupported"
                                                          for r in compact if r["route"] == route), "report phase counts")
            for name in ("first", "changed_form", "exact_delayed"):
                group = actual["groups"][name]
                require(saved["acquisition_encounters"][name]["episodes"] == sum(g["observed"] for g in group.values())
                        and saved["acquisition_encounters"][name]["first_correct"] == sum(g["correct"] for g in group.values())
                        and saved["acquisition_encounters"][name]["requirement_changed"] == sum(
                            bool(e.get("requirement_changed")) for e in manifest["episodes"]
                            if e["phase"] == "acquisition" and e.get("encounter") == name),
                        "report acquisition encounters")
            for name in ("instruction_construction", "structural_combination"):
                group = actual["groups"][name]
                row = saved["fresh_strata"][name]
                require(row["episodes"] == sum(g["observed"] for g in group.values())
                        and row["first_correct"] == sum(g["correct"] for g in group.values())
                        and row["unsupported"] == sum(g["observed"] - g["supported"] for g in group.values()),
                        "report fresh strata")
                for label, dependent in (("independent", False), ("actual_project_dependent", True)):
                    members = [r for r in compact if r["route"] == route and r["attempt_index"] == 1
                               and bool(r["predecessor_episode_id"]) == dependent
                               and next(e for e in manifest["episodes"] if e["episode_id"] == r["episode_id"]).get("fresh_stratum") == name]
                    require(row["dependency"][label] == {"episodes": len(members), "first_correct": sum(r["correct"] for r in members),
                                                         "unsupported": sum(r["status"] == "unsupported" for r in members)},
                            "report actual project dependency split")
            old, row = actual["recovery"], saved["recovery"]
            for name, source in (("episodes", "observed_episodes"), ("first_correct", "first_correct"),
                                 ("later_correct", "ever_recovered"), ("final_correct", "terminal_correct"),
                                 ("final_recovered", "terminal_recovered"),
                                 ("feedback_consuming_attempts", "feedback_consuming_attempts"),
                                 ("identical_consecutive_proposals", "identical_consecutive_proposals")):
                require(row[name] == old[source], "report recovery metric")
    evidence = {p.relative_to(run).as_posix(): scalar.file_sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
    return {"schema": "persistent-language-action-saved-review-v8", "status": "passed_partial" if partial else "passed",
            "complete_run": not partial, "run": str(run), "manifest_sha256": manifest_sha,
            "source_revision": command[command.index("--source-revision") + 1],
            "run_launch_sha256": scalar.file_sha(launch_path), "saved_tool_sources": saved_tool_sources,
            "attempt_rows": len(compact), "teacher_rows": consumed["teachers"], "ingest_rows": consumed["ingests"],
            "exposures": counts, "source_offers": source_offers, "controller_exposures": stochastic,
            "controller_exposure_scope": "checked saved ticks only; status includes its framing EOS and is disjoint from raw_content and other eos offers; free has no target or match count; aggregate matches are not a temporal improvement or task-success claim",
            "saved_costs": costs, "metrics": result_metrics, "compact_attempts": compact,
            "pending_transaction": pending, "checkpoint": checkpoint, "journal_frontiers": frontiers,
            "checkpoint_telemetry_counts": dict(telemetry_counts), "checkpoint_telemetry_seconds": dict(telemetry_seconds),
            "checkpoint_telemetry_frontier": telemetry_frontier, "evidence": evidence,
            "payload_read_scope": "complete inventory (complete only), independent checkpoint authentication, fresh store inspect, and final evidence inventory are distinct real byte-read passes",
            "elapsed_seconds": time.perf_counter() - started, "model_constructions": 0, "checkpoint_decodes": 0,
            "predictor_calls": 0, "actual_executions": 0, "teacher_replays": 0,
            "cost_scope": "saved category counters; attempt timer includes its recorded begin interval and execution; no isolated scan/free latency",
            "scope": "durable actual prefix, local scalar receipts and custody; no reconstruction of omitted eligibility, synapses, gates or interrupted unsaved exposure"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if DRAFT or MANIFEST_SHA is None:
        raise SystemExit("DRAFT: saved audit requires root source/manifest release")
    try:
        result = review(args.run, args.launch, args.manifest_sha, partial=args.partial)
    except BaseException as error:
        with args.output.open("xb") as handle:
            handle.write(encoded({"schema": "persistent-language-action-saved-review-v8", "status": "failed",
                                  "error_type": type(error).__name__, "error": str(error)}))
        raise
    with args.output.open("xb") as handle:
        handle.write(encoded(result))
    print(json.dumps({"status": result["status"], "attempt_rows": result["attempt_rows"]}))


if __name__ == "__main__":
    main()
