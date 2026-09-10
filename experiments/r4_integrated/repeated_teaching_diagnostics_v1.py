"""Read-only output diagnostics; no teacher, predictor, RNG or publication calls.

The experiment owns the actual response and any separately counted diagnostic
teacher. This observer consumes their receipts and a prepublication weight copy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import torch

from humanbrain.neuronal_splice_learner import _digest
from humanbrain.recurrent_content_controller import _digest as _activity_digest

WIDTH, SYMBOLS, BOS, ABSENT = 1072, 257, 257, 258


def _value(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@torch.no_grad()
def capture_literal(reader):
    """Call before teach publishes; the observer never borrows a live weight view."""
    started, work = time.perf_counter(), {}
    weight = reader.controller.literal_weight.detach().clone()
    _require(
        weight.shape == (SYMBOLS, WIDTH)
        and weight.dtype == torch.float64
        and weight.device.type == "cpu"
        and not weight.requires_grad
        and bool(torch.isfinite(weight).all()),
        "diagnostic snapshot requires the fixed CPU literal population",
    )
    work["literal_snapshot_copy_bytes"] = weight.numel() * weight.element_size()
    identity = _digest({"literal_weight": weight}, work)
    return {
        "weight": weight,
        "sha256": identity,
        "work": work,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _activity(tick, previous, work):
    """Mechanical fixed emission layout, independently checked against acting hash."""
    h, registers, prior = tick["h_after"], tick["register_tokens"], tick["prior_token"]
    _require(
        type(h) is int
        and 0 <= h < 32
        and len(registers) == 2
        and type(prior) is int
        and 0 <= prior < SYMBOLS,
        "invalid recorded teacher state",
    )
    _require(
        all(token is None or (type(token) is int and 0 <= token < SYMBOLS) for token in registers),
        "invalid recorded register token",
    )
    blocks = [(32, (h,)), (259, (previous,))]
    blocks += [(259, (ABSENT if token is None else token,)) for token in registers]
    blocks += [
        (2, tuple(i for i, token in enumerate(registers) if token is not None)),
        (3, (2,)),
        (257, (prior,)),
        (1, (0,)),
    ]
    indices, offset = [], 0
    for width, active in blocks:
        indices.extend(offset + index for index in active)
        offset += width
    _require(offset == WIDTH, "diagnostic emission layout changed")
    indices = torch.tensor(indices, dtype=torch.int64)
    raw = torch.ones(len(indices), dtype=torch.float64)
    values = raw / raw.square().sum().sqrt()
    work["activity_builds"] = work.get("activity_builds", 0) + 1
    work["activity_nonzeros"] = work.get("activity_nonzeros", 0) + len(indices)
    work["activity_allocated_bytes"] = (
        work.get("activity_allocated_bytes", 0) + 32 * len(indices) + 16
    )
    # Acting activity receipts use the controller's distinct hash domain.
    identity = _activity_digest({"indices": indices, "values": values, "width": WIDTH}, work)
    _require(
        identity == tick["draws"][2]["activity_sha256"],
        "diagnostic activity differs from actual held mode activity",
    )
    return indices, values, identity


@torch.no_grad()
def analyze_teacher(feedback, target_bytes, *, literal_snapshot, learning_enabled):
    """Score the intended prefix at sampled h/gates; not a marginal sequence likelihood."""
    started, work = time.perf_counter(), {}
    _require(
        type(target_bytes) is bytes and len(target_bytes) <= 128 and type(learning_enabled) is bool,
        "bounded raw target and explicit actual learning gate required",
    )
    weight = literal_snapshot["weight"]
    _require(
        weight.shape == (SYMBOLS, WIDTH)
        and weight.dtype == torch.float64
        and weight.device.type == "cpu"
        and not weight.requires_grad,
        "invalid held diagnostic weight snapshot",
    )
    identity = _digest({"literal_weight": weight}, work)
    _require(identity == literal_snapshot["sha256"], "held literal weight copy changed")
    # The live feedback dataclass retains teacher_steps; reader_summary serializes
    # those dictionary receipts under steps. Neither path invokes the learner.
    steps = feedback["steps"] if isinstance(feedback, Mapping) else feedback.teacher_steps
    _require(len(steps) == len(target_bytes) + 1, "complete teacher bytes and EOS required")
    _require(
        _value(feedback, "controller")["learning_enabled"] is learning_enabled,
        "diagnostic gate differs from actual teacher resolution",
    )
    rows, selected_logscore, mixture_logscore = [], 0.0, 0.0
    selected_zero = False
    for position, (step, target) in enumerate(zip(steps, (*target_bytes, 256), strict=True)):
        tick = step["controller"]
        prefix = target_bytes[:position]
        _require(
            step["output_step"] == position
            and step["prefix_hex"] == prefix.hex()
            and step["observed_token"] == target
            and tick["category"] == "teacher",
            "teacher prefix or genuine offered target changed",
        )
        draws = tick["draws"]
        mode, probabilities = draws[2]["index"], draws[2]["probabilities"]
        if isinstance(probabilities, torch.Tensor):
            _require(
                probabilities.device.type == "cpu" and not probabilities.requires_grad,
                "actual mode receipt must remain detached on CPU",
            )
            probabilities = probabilities.tolist()
            work["mode_receipt_scalar_reads"] = work.get("mode_receipt_scalar_reads", 0) + 4
        _require(
            len(probabilities) == 4
            and type(mode) is int
            and 0 <= mode < 4
            and len(draws) == 3 + int(mode == 3),
            "actual mode/literal draw ordering",
        )
        _require(
            all(type(p) in (float, int) and math.isfinite(p) and 0 < p <= 1 for p in probabilities)
            and math.isclose(sum(probabilities), 1, rel_tol=1e-12, abs_tol=1e-12),
            "invalid saved mode probabilities",
        )
        indices, values, activity_sha = _activity(tick, BOS if not prefix else prefix[-1], work)
        gathered = weight.index_select(1, indices)
        held_logits = (gathered * values).sum(1)
        count = SYMBOLS * len(indices)
        work["held_literal_score_evaluations"] = work.get("held_literal_score_evaluations", 0) + 1
        work["held_literal_weight_products"] = work.get("held_literal_weight_products", 0) + count
        work["held_literal_score_allocated_bytes"] = (
            work.get("held_literal_score_allocated_bytes", 0) + 16 * count + 8 * SYMBOLS
        )
        if mode == 3:
            literal = draws[3]
            _require(
                literal["population"] == "literal" and literal["activity_sha256"] == activity_sha,
                "selected literal activity differs",
            )
            literal_logits = torch.as_tensor(literal["logits"], dtype=torch.float64)
            literal_probabilities = torch.as_tensor(literal["probabilities"], dtype=torch.float64)
            _require(
                torch.equal(literal_logits, held_logits),
                "prepublication snapshot differs from actual literal currents",
            )
            work["reused_actual_literal_distributions"] = (
                work.get("reused_actual_literal_distributions", 0) + 1
            )
        else:
            literal_logits = held_logits
            exponentials = (literal_logits - literal_logits.max()).exp()
            literal_probabilities = exponentials / exponentials.sum()
            work["extra_literal_evaluations"] = work.get("extra_literal_evaluations", 0) + 1
            work["extra_literal_allocated_bytes"] = (
                work.get("extra_literal_allocated_bytes", 0) + 24 * SYMBOLS
            )
        _require(
            literal_logits.shape == literal_probabilities.shape == (SYMBOLS,)
            and bool(torch.isfinite(literal_logits).all())
            and bool(torch.isfinite(literal_probabilities).all())
            and bool((literal_probabilities > 0).all()),
            "nonfinite diagnostic literal distribution",
        )
        literal_p = float(literal_probabilities[target])
        shifted = literal_logits - literal_logits.max()
        literal_logp = float(shifted[target]) - math.log(float(shifted.exp().sum()))
        tokens = [
            tick["prior_token"],
            *(256 if token is None else token for token in tick["register_tokens"]),
        ]
        selected_p = literal_p if mode == 3 else float(tokens[mode] == target)
        components = [probabilities[i] * int(token == target) for i, token in enumerate(tokens)]
        components.append(probabilities[3] * literal_p)
        mixture_p = sum(components)
        _require(
            math.isfinite(mixture_p) and 0 < mixture_p <= 1 + 1e-12,
            "invalid full mode-mixture target probability",
        )
        mixture_logp = math.log(mixture_p)
        mixture_logscore += mixture_logp
        selected_logp = literal_logp if mode == 3 else 0.0 if selected_p else None
        if selected_logp is None:
            selected_zero = True
        else:
            selected_logscore += selected_logp
        credit = tick["credit"]
        _require(
            credit["literal_taught"] is (mode == 3)
            and credit["observed_token"] == target
            and credit["sampled_reward"] == (1 if tick["emitted_token"] == target else -1),
            "actual direct-teacher inclusion changed",
        )
        rows.append(
            {
                "position": position,
                "target": target,
                "eos": target == 256,
                "prefix_hex": prefix.hex(),
                "selected_mode": mode,
                "sampled_match": tick["emitted_token"] == target,
                "selected_literal": mode == 3,
                "direct_teacher_proposed": mode == 3,
                "direct_teacher_published": mode == 3 and learning_enabled,
                "diagnostic_extra_literal_evaluation": mode != 3,
                "activity_sha256": activity_sha,
                "selected_mode_target_probability": selected_p,
                "selected_mode_target_log_probability": selected_logp,
                "literal_target_probability": literal_p,
                "literal_target_log_probability": literal_logp,
                "mode_mixture_components": components,
                "mode_mixture_target_probability": mixture_p,
                "mode_mixture_target_log_probability": mixture_logp,
            }
        )
    _require(
        _digest({"literal_weight": weight}, work) == identity,
        "diagnostic changed held literal copy",
    )
    path_score = math.exp(mixture_logscore)
    return {
        "schema": "repeated-teaching-conditional-output-diagnostics-v1",
        "target_hex": target_bytes.hex(),
        "offers": len(rows),
        "raw_offers": len(target_bytes),
        "eos_offers": 1,
        "literal_weight_sha256": identity,
        "snapshot_work": dict(literal_snapshot["work"]),
        "snapshot_seconds": literal_snapshot["elapsed_seconds"],
        "positions": rows,
        "literal_offers": sum(row["selected_literal"] for row in rows),
        "direct_teacher_published_offers": sum(row["direct_teacher_published"] for row in rows),
        "selected_mode_path_has_zero": selected_zero,
        "selected_mode_path_logscore": None if selected_zero else selected_logscore,
        "mode_mixture_path_logscore": mixture_logscore,
        "mode_mixture_path_score": path_score,
        "mode_mixture_path_score_underflow": path_score == 0,
        "scope": (
            "target-prefix emission scores conditioned on actual sampled context/gate path; "
            "not globally marginalized complete-output probability "
            "or a free-response success estimate"
        ),
        "extra_literal_scope": (
            "observer only; missing branch evaluated from prepublication "
            "private weight copy; never feeds acting or credit"
        ),
        "draws": 0,
        "teacher_calls": 0,
        "prior_calls": 0,
        "publications": 0,
        "work": work,
        "elapsed_seconds": time.perf_counter() - started,
    }


def free_summary(prediction, target_bytes):
    """Compare the actual free bytes including EOS; never score a forced suffix."""
    _require(
        type(target_bytes) is bytes and len(target_bytes) <= 128, "bounded intended bytes required"
    )
    if isinstance(prediction, Mapping):
        if "payload" in prediction:
            prediction = prediction["payload"]
        answer, stop = bytes.fromhex(prediction["answer_hex"]), prediction["stop_reason"]
    else:
        if hasattr(prediction, "payload"):
            prediction = prediction.payload
        answer, stop = prediction.generation.data, prediction.generation.stop_reason
    steps = _value(prediction, "steps")
    rows, aligned = [], True
    for position, step in enumerate(steps):
        token = step["token"] if isinstance(step, Mapping) else step.observed_token
        expected = (
            target_bytes[position]
            if position < len(target_bytes)
            else 256
            if position == len(target_bytes)
            else None
        )
        rows.append(
            {
                "position": position,
                "actual_token": token,
                "intended_token": expected,
                "intended_prefix_aligned": aligned,
                "matches_intended_here": aligned and expected is not None and token == expected,
            }
        )
        aligned = aligned and expected is not None and token == expected
    prefix_length = 0
    for actual, intended in zip(answer, target_bytes, strict=False):
        if actual != intended:
            break
        prefix_length += 1
    return {
        "answer_hex": answer.hex(),
        "bytes": len(answer),
        "stop_reason": stop,
        "exact_bytes": answer == target_bytes,
        "complete_bytes_and_eos": answer == target_bytes and stop == "eos",
        "matching_raw_prefix_bytes": prefix_length,
        "decisions": len(steps),
        "positions": rows,
        "scope": (
            "actual free response; positions after divergence do not estimate "
            "intended-prefix probabilities"
        ),
    }
