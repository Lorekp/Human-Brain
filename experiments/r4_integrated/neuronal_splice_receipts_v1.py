"""Bounded, teacher-free JSON receipts for the fixed neuronal splice interface.

No model, tensor library, experiment, roster or executor is imported here.
Tensor-like values are converted only when supplied by the actual caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping

LINE_LIMIT = TREE_LIMIT = 1 << 20
MAX_OUTPUT_BYTES = 16384
HEAD_LIMITS = {"start": 516, "end": 129, "operation": 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if type(value) in (str, int, float, bool, type(None)):
        if type(value) is float:
            require(math.isfinite(value), "nonfinite receipt number")
        return value
    if isinstance(value, Mapping):
        require(all(type(key) is str for key in value), "receipt keys must be strings")
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    raise TypeError("receipt values must be JSON primitives or supplied tensor activities")


def tree_bytes(value):
    size = sys.getsizeof(value)
    if type(value) is dict:
        return size + sum(tree_bytes(key) + tree_bytes(item) for key, item in value.items())
    if type(value) is list:
        return size + sum(tree_bytes(item) for item in value)
    require(type(value) in (str, int, float, bool, type(None)), "non-JSON row value")
    if type(value) is float:
        require(math.isfinite(value), "nonfinite row number")
    return size


def bounded_record(row):
    require(type(row) is dict, "record must be an object")
    require(tree_bytes(row) <= TREE_LIMIT, "receipt exceeds tree limit")
    encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("ascii")
    require(len(encoded) <= LINE_LIMIT, "receipt exceeds line limit")
    return encoded


def exact_keys(row, names):
    require(type(row) is dict and set(row) == set(names), "receipt field mismatch")


def hex_bytes(value, maximum=None):
    require(type(value) is str, "hex bytes must be a string")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("invalid hex bytes") from error
    require(raw.hex() == value, "hex bytes must be canonical")
    require(maximum is None or len(raw) <= maximum, "raw bytes exceed receipt bound")
    return raw


def hash_value(value):
    require(type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value), "invalid SHA256")
    return value


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "expected finite number")
    return float(value)


def vector(values, count):
    require(type(values) is list and len(values) == count, "vector geometry mismatch")
    return [number(value) for value in values]


def close(actual, expected, *, atol=2e-10, rtol=2e-10):
    require(math.isclose(number(actual), number(expected), abs_tol=atol, rel_tol=rtol),
            "receipt arithmetic mismatch")


def logsumexp(values):
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


def softmax(logits):
    denominator = logsumexp(logits)
    return [math.exp(value - denominator) for value in logits]


def validate_head(row, head, *, count=None):
    exact_keys(row, ("logits", "selected_index"))
    require(head in HEAD_LIMITS, "unknown action head")
    require(type(row["logits"]) is list, "head logits must be a list")
    n = len(row["logits"])
    require(1 <= n <= HEAD_LIMITS[head] and (head != "operation" or n == 2),
            "head population exceeds topology")
    require(count is None or n == count, "head candidate count mismatch")
    logits = vector(row["logits"], n)
    selected = row["selected_index"]
    require(type(selected) is int and selected == max(range(n), key=logits.__getitem__),
            "head decision is not the first maximum")
    return softmax(logits)


def compact_head(decision, head):
    row = {"logits": plain(decision.logits), "selected_index": decision.selected_index}
    expected = validate_head(row, head)
    actual = vector(plain(decision.probabilities), len(expected))
    for left, right in zip(actual, expected, strict=True):
        close(left, right)
    return row


def tape_adjacency(state):
    if isinstance(state, Mapping):
        used = plain(state["write_index"])
        successor = plain(state["successor"])
    else:
        used, successor = state.used, plain(state.successor)
    require(type(used) is int and 1 <= used <= 128, "reader needs occupied slots")
    require(type(successor) is list and len(successor) >= used, "missing tape adjacency")
    successor = successor[:used]
    for index, target in enumerate(successor):
        require(type(target) is int and target in (index, index + 1) and target < used,
                "invalid forward tape adjacency")
    return successor


def validate_work(work):
    require(type(work) is dict and len(work) <= 160, "invalid work counters")
    for key, value in work.items():
        require(type(key) is str and len(key) <= 96, "invalid work counter name")
        require(type(value) is int and 0 <= value <= 10**12, "invalid work counter")


def validate_reader(row, successor, *, max_bytes=16, targets=None):
    exact_keys(row, ("kind", "initial_log_belief", "steps", "generated_hex",
                     "stop_reason", "work"))
    require(row["kind"] in ("free", "teacher"), "invalid reader receipt kind")
    require(type(max_bytes) is int and 0 <= max_bytes <= 16, "invalid reader cap")
    used = len(successor)
    require(1 <= used <= 128, "invalid reader population")
    belief = vector(row["initial_log_belief"], used)
    close(logsumexp(belief), 0)
    steps = row["steps"]
    require(type(steps) is list and 1 <= len(steps) <= max_bytes + 1, "invalid step count")
    data = hex_bytes(row["generated_hex"], max_bytes)
    require(row["stop_reason"] in ("eos", "limit"), "invalid stopping reason")
    cursors, tokens, loss = list(range(used)), [], 0.0
    for index, step in enumerate(steps):
        exact_keys(step, ("token", "nll", "mixture", "posterior"))
        token = step["token"]
        require(type(token) is int and 0 <= token <= 256, "invalid observed token")
        probabilities = vector(step["mixture"], 257)
        require(all(0 <= value <= 1 for value in probabilities), "invalid probability")
        close(math.fsum(probabilities), 1)
        nll = number(step["nll"])
        require(nll >= -2e-10, "negative NLL")
        # A finite log probability may underflow when exponentiated for the saved mixture.
        if probabilities[token] > 1e-250:
            close(-math.log(probabilities[token]), nll)
        else:
            close(probabilities[token], math.exp(-nll), atol=5e-324, rtol=2e-10)
        if row["kind"] == "free":
            require(token == max(range(257), key=probabilities.__getitem__),
                    "free token differs from actual mixture argmax")
        require(index == len(steps) - 1 or token != 256, "steps continued after EOS")
        belief = vector(step["posterior"], used)
        close(logsumexp(belief), 0)
        tokens.append(token)
        loss += nll
        if index + 1 < len(steps):
            cursors = [successor[cursor] for cursor in cursors]
    eos = tokens[-1] == 256
    require(row["stop_reason"] == ("eos" if eos else "limit"), "stopping receipt mismatch")
    if row["kind"] == "teacher":
        require(eos, "teacher must offer EOS")
        offered = bytes(tokens[:-1])
        require(offered == data, "teacher bytes mismatch")
        require(targets is None or offered == targets, "teacher differs from correction")
    else:
        require(eos or len(steps) == max_bytes + 1, "early non-EOS stopping")
        require(bytes(tokens[:-1]) == data, "free emitted bytes mismatch")
    validate_work(row["work"])
    return {"decisions": len(steps), "loss": loss, "final_log_belief": belief,
            "final_cursors": cursors}


def compact_reader(receipt, state):
    successor = tape_adjacency(state)
    belief = plain(receipt.initial_log_belief)
    initial, cursors, steps = belief, list(range(len(successor))), []
    for index, step in enumerate(receipt.steps):
        require(step.output_step == index, "reader step order changed")
        require(plain(step.prior_log_belief) == belief, "omitted prior is not redundant")
        require(plain(step.branch_cursors) == cursors, "omitted cursor is not deterministic")
        belief = plain(step.posterior_log_belief)
        steps.append({"token": step.observed_token, "nll": step.nll,
                      "mixture": plain(step.mixture_probabilities), "posterior": belief})
        cursors = [successor[cursor] for cursor in cursors]
    require(plain(receipt.final_log_belief) == belief, "omitted final belief changed")
    generation = getattr(receipt, "generation", None)
    kind = "free" if generation is not None else "teacher"
    data = generation.data if generation is not None else bytes(step["token"] for step in steps[:-1])
    reason = generation.stop_reason if generation is not None else "eos"
    row = {"kind": kind, "initial_log_belief": initial, "steps": steps,
           "generated_hex": data.hex(), "stop_reason": reason, "work": plain(receipt.work)}
    cap = len(steps) - 1 if kind == "free" and reason == "limit" else 16
    summary = validate_reader(row, successor, max_bytes=cap)
    if generation is None:
        require(receipt.teacher_count == len(steps), "teacher count mismatch")
        close(receipt.total_loss, summary["loss"])
        close(receipt.mean_loss, summary["loss"] / len(steps))
        expected = [math.exp(a) - math.exp(b) for a, b in zip(initial, belief, strict=True)]
        for actual, value in zip(plain(receipt.address_error), expected, strict=True):
            close(actual, value)
    return row


def validate_proposal(row):
    exact_keys(row, ("operation", "path", "start", "end", "replacement_hex",
                     "stop_reason", "workspace_sha256"))
    require(row["operation"] in ("STOP", "SPLICE"), "invalid operation")
    require(type(row["path"]) is str and len(row["path"].encode("utf-8")) <= 31,
            "invalid proposal path")
    require(type(row["start"]) is int and type(row["end"]) is int, "invalid offsets")
    hex_bytes(row["replacement_hex"], 16)
    require(row["stop_reason"] in ("eos", "limit"), "invalid payload status")
    hash_value(row["workspace_sha256"])


def proposal_digest(row):
    validate_proposal(row)
    raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"humanbrain.raw-splice-proposal.v1\0" + raw).hexdigest()


def compact_prediction(prediction, state):
    row = {"proposal": prediction.proposal.to_record(), "attempt_id": prediction.attempt_id,
           "proposal_sha256": prediction.proposal_sha256, "model_sha256": prediction.model_sha256,
           "memory_sha256": prediction.memory_sha256,
           "choices": {head: compact_head(value, head) for head, value in prediction.choices.items()},
           "payload": compact_reader(prediction.payload, state), "work": plain(prediction.work)}
    validate_prediction(row, tape_adjacency(state))
    return row


def validate_prediction(row, successor, *, boundaries=None):
    exact_keys(row, ("proposal", "attempt_id", "proposal_sha256", "model_sha256",
                     "memory_sha256", "choices", "payload", "work"))
    require(type(row["attempt_id"]) is str and 0 < len(row["attempt_id"]) <= 128,
            "invalid attempt identity")
    for name in ("proposal_sha256", "model_sha256", "memory_sha256"):
        hash_value(row[name])
    require(proposal_digest(row["proposal"]) == row["proposal_sha256"], "proposal hash mismatch")
    exact_keys(row["choices"], HEAD_LIMITS)
    for head in HEAD_LIMITS:
        validate_head(row["choices"][head], head)
    proposal = row["proposal"]
    require(proposal["operation"] == ("STOP" if row["choices"]["operation"]["selected_index"] == 0
                                     else "SPLICE"), "operation receipt mismatch")
    validate_reader(row["payload"], successor)
    require(row["payload"]["kind"] == "free", "prediction contains a teacher")
    require(proposal["replacement_hex"] == row["payload"]["generated_hex"]
            and proposal["stop_reason"] == row["payload"]["stop_reason"], "payload mismatch")
    if boundaries is not None:
        require(len(boundaries) == len(row["choices"]["start"]["logits"]), "start population mismatch")
        path, start = boundaries[row["choices"]["start"]["selected_index"]]
        require((proposal["path"], proposal["start"]) == (path, start), "start selection mismatch")
        ends = [offset for name, offset in boundaries if name == path and offset >= start]
        require(len(ends) == len(row["choices"]["end"]["logits"]), "end population mismatch")
        require(proposal["end"] == ends[row["choices"]["end"]["selected_index"]], "end selection mismatch")
    exact_keys(row["work"], ("action", "reader"))
    for counters in row["work"].values():
        validate_work(counters)
    require(row["work"]["reader"] == row["payload"]["work"], "reader work mismatch")


def compact_feedback(feedback, state):
    return {"attempt_id": feedback.attempt_id, "action_updates": plain(feedback.action_updates),
            "action_losses": plain(feedback.action_losses),
            "teacher_choices": {head: compact_head(value, head)
                                for head, value in feedback.teacher_choices.items()},
            "reader": None if feedback.reader_feedback is None else compact_reader(feedback.reader_feedback, state),
            "reader_updates": None if feedback.reader_feedback is None else {
                "codec": feedback.reader_feedback.codec_updated,
                "address": feedback.reader_feedback.address_updated},
            "work": plain(feedback.work)}
