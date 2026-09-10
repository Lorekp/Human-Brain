"""Observers of actual integrated practice and fixed checkpoint output activities.

No predictor, teacher, prior, RNG or publication calls. Counterfactual literal
scoring uses only the checkpoint's private prepublication weight snapshot.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import torch

from experiments.r4_integrated.repeated_teaching_diagnostics_v1 import (
    BOS,
    SYMBOLS,
    WIDTH,
    _activity,
    _require,
    _value,
    analyze_teacher,
)
from humanbrain.neuronal_splice_learner import _digest
from humanbrain.recurrent_content_controller import _digest as activity_digest


def _steps(feedback):
    return feedback["steps"] if isinstance(feedback, Mapping) else feedback.teacher_steps


def _list(value):
    return value.tolist() if isinstance(value, torch.Tensor) else list(value)


def practice_teacher(feedback, target_bytes, *, learning_enabled):
    """Keep actual teacher coverage; never evaluate the unselected literal branch."""
    started = time.perf_counter()
    _require(type(target_bytes) is bytes and len(target_bytes) <= 128, "bounded target required")
    _require(type(learning_enabled) is bool, "explicit learning gate required")
    _require(_value(feedback, "controller")["learning_enabled"] is learning_enabled,
             "observer learning gate differs from actual resolution")
    steps = _steps(feedback)
    _require(len(steps) == len(target_bytes) + 1, "complete teacher including EOS required")
    rows, actual_draws = [], 0
    for position, (step, target) in enumerate(zip(steps, (*target_bytes, 256), strict=True)):
        tick = step["controller"]
        _require(step["output_step"] == position
                 and step["observed_token"] == target
                 and step["prefix_hex"] == target_bytes[:position].hex()
                 and tick["category"] == "teacher", "actual teacher sequence differs")
        draws, credit = tick["draws"], tick["credit"]
        mode = draws[2]["index"]
        _require(type(mode) is int and 0 <= mode < 4
                 and len(draws) == 3 + int(mode == 3), "actual draw ordering differs")
        actual_draws += len(draws)
        probabilities = _list(draws[2]["probabilities"])
        _require(len(probabilities) == 4
                 and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                 and math.isclose(sum(probabilities), 1, rel_tol=1e-12, abs_tol=1e-12),
                 "invalid actual mode probabilities")
        _require(credit["literal_taught"] is (mode == 3)
                 and credit["observed_token"] == target, "teacher contact receipt differs")
        literal_p, literal_logp = None, None
        if mode == 3:
            literal = draws[3]
            _require(literal["population"] == "literal"
                     and literal["activity_sha256"] == draws[2]["activity_sha256"],
                     "literal contact is not the actual selected activity")
            literal_p = float(literal["probabilities"][target])
            logits = _list(literal["logits"])
            _require(len(logits) == SYMBOLS and all(math.isfinite(x) for x in logits)
                     and math.isfinite(literal_p) and 0 <= literal_p <= 1,
                     "invalid actual selected literal distribution")
            maximum = max(logits)
            literal_logp = logits[target] - maximum - math.log(
                sum(math.exp(x - maximum) for x in logits))
        tokens = [tick["prior_token"],
                  *(256 if token is None else token for token in tick["register_tokens"])]
        rows.append({
            "position": position, "target": target, "eos": target == 256,
            "offered": True, "selected_literal": mode == 3,
            "direct_teacher_published": mode == 3 and learning_enabled,
            "selected_mode": mode, "mode_probabilities": probabilities,
            "deterministic_tokens": tokens,
            "selected_literal_target_probability": literal_p,
            "selected_literal_target_log_probability": literal_logp,
            "mode_activity_sha256": draws[2]["activity_sha256"],
            "h_after": tick["h_after"], "register_tokens": list(tick["register_tokens"]),
            "prior_token": tick["prior_token"],
            "credit": {key: credit[key] for key in (
                "signal", "signal_objective", "baseline_before", "sampled_reward"
            ) if key in credit},
        })
    return {
        "schema": "integrated-local-learning-practice-teacher-v1",
        "target_hex": target_bytes.hex(), "offers": len(rows), "raw_offers": len(target_bytes),
        "eos_offers": 1, "literal_offers": sum(r["selected_literal"] for r in rows),
        "direct_teacher_published_offers": sum(r["direct_teacher_published"] for r in rows),
        "positions": rows, "extra_literal_evaluations": 0,
        "actual_teacher_controller_draws": actual_draws,
        "scope": "actual selected-contact scores; no unselected-branch or free-success estimate",
        "draws": 0, "teacher_calls": 0, "prior_calls": 0, "publications": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def score_anchors(literal_snapshot, anchors):
    """Score fixed CP0 activities and competitors using only held literal weights."""
    started, work = time.perf_counter(), {}
    weight = literal_snapshot["weight"]
    _require(weight.shape == (SYMBOLS, WIDTH) and weight.dtype == torch.float64
             and weight.device.type == "cpu" and not weight.requires_grad,
             "invalid private literal snapshot")
    identity = _digest({"literal_weight": weight}, work)
    _require(identity == literal_snapshot["sha256"], "private literal snapshot changed")
    rows = []
    for anchor in anchors:
        target, competitor = anchor["target"], anchor["competitor"]
        _require(type(target) is int and 0 <= target < SYMBOLS
                 and type(competitor) is int and 0 <= competitor < SYMBOLS
                 and competitor != target, "invalid fixed target/competitor")
        indices = torch.tensor(anchor["indices"], dtype=torch.int64)
        values = torch.tensor(anchor["values"], dtype=torch.float64)
        _require(indices.ndim == values.ndim == 1 and 6 <= len(indices) <= 9
                 and indices.shape == values.shape and bool((indices >= 0).all())
                 and bool((indices < WIDTH).all()) and len(set(indices.tolist())) == len(indices)
                 and bool(torch.isfinite(values).all())
                 and math.isclose(float(values.square().sum()), 1, abs_tol=1e-12),
                 "invalid fixed emission activity")
        _require(activity_digest({"indices": indices, "values": values, "width": WIDTH}, work)
                 == anchor["activity_sha256"], "fixed activity identity differs")
        logits = (weight.index_select(1, indices) * values).sum(1)
        shifted = logits - logits.max()
        exponentials = shifted.exp()
        probabilities = exponentials / exponentials.sum()
        candidates = logits.clone()
        candidates[target] = -math.inf
        winner = int(candidates.argmax())
        rows.append({
            "position": anchor["position"], "target": target,
            "activity_sha256": anchor["activity_sha256"],
            "target_current": float(logits[target]),
            "target_probability": float(probabilities[target]),
            "target_log_probability": float(shifted[target]) - math.log(float(exponentials.sum())),
            "fixed_competitor": competitor,
            "fixed_competitor_current": float(logits[competitor]),
            "fixed_competitor_margin": float(logits[target] - logits[competitor]),
            "strongest_competitor": winner,
            "strongest_competitor_margin": float(logits[target] - logits[winner]),
        })
        work["fixed_activity_score_evaluations"] = (
            work.get("fixed_activity_score_evaluations", 0) + 1)
    _require(_digest({"literal_weight": weight}, work) == identity, "observer changed snapshot")
    return {"schema": "integrated-local-learning-fixed-activity-scores-v1",
            "literal_weight_sha256": identity, "positions": rows, "work": work,
            "scope": (
                "fixed literal-site weight changes, not full-network or free-response probability"),
            "model_calls": 0, "draws": 0, "publications": 0,
            "elapsed_seconds": time.perf_counter() - started}


@torch.no_grad()
def panel_teacher(feedback, target_bytes, *, literal_snapshot, learning_enabled=False):
    """Extend full conditional checkpoint diagnostics with reusable fixed activities."""
    result = analyze_teacher(feedback, target_bytes, literal_snapshot=literal_snapshot,
                             learning_enabled=learning_enabled)
    work, anchors = {}, []
    weight = literal_snapshot["weight"]
    for row, step in zip(result["positions"], _steps(feedback), strict=True):
        position, target = row["position"], row["target"]
        indices, values, identity = _activity(
            step["controller"], BOS if position == 0 else target_bytes[position - 1], work)
        logits = (weight.index_select(1, indices) * values).sum(1)
        candidates = logits.clone()
        candidates[target] = -math.inf
        anchors.append({"position": position, "target": target,
                        "competitor": int(candidates.argmax()), "activity_sha256": identity,
                        "indices": indices.tolist(), "values": values.tolist()})
    result["anchors"] = anchors
    result["current_activity_scores"] = score_anchors(literal_snapshot, anchors)
    result["anchor_work"] = work
    return result


def free_teacher_alignment(prediction, feedback, target_bytes):
    """Keep sampled state differences separate from wrong-prefix consequences."""
    if hasattr(prediction, "payload"):
        prediction = prediction.payload
    if isinstance(prediction, Mapping) and "payload" in prediction:
        prediction = prediction["payload"]
    free_steps, teacher_steps = _value(prediction, "steps"), _steps(feedback)
    _require(len(teacher_steps) == len(target_bytes) + 1, "complete teacher required")
    rows, aligned = [], True

    def state(tick):
        return {"h_before": tick["h_before"], "h_after": tick["h_after"],
                "cursors_before": list(tick["cursors_before"]),
                "cursors_after": list(tick["cursors_after"]),
                "register_tokens": list(tick["register_tokens"]),
                "prior_token": tick["prior_token"],
                "mode_activity_sha256": tick["draws"][2]["activity_sha256"],
                "incoming_rng_sha256": tick["draws"][0]["rng_before_sha256"]}

    for position, step in enumerate(free_steps):
        tick = _value(step, "controller")
        token = step["token"] if isinstance(step, Mapping) else step.observed_token
        target = (target_bytes[position] if position < len(target_bytes)
                  else 256 if position == len(target_bytes) else None)
        free_state = state(tick)
        taught_state = (state(teacher_steps[position]["controller"])
                        if position < len(teacher_steps) else None)
        rows.append({"position": position, "intended_prefix_aligned": aligned,
                     "actual_token": token, "target": target,
                     "free": free_state, "teacher": taught_state,
                     "same_emission_activity": taught_state is not None and (
                         free_state["mode_activity_sha256"]
                         == taught_state["mode_activity_sha256"]),
                     "same_incoming_rng": taught_state is not None and (
                         free_state["incoming_rng_sha256"] == taught_state["incoming_rng_sha256"])})
        aligned = aligned and target is not None and token == target
        if not aligned:
            break
    return {"schema": "integrated-local-learning-free-teacher-alignment-v1", "positions": rows,
            "free_decisions": len(free_steps),
            "omitted_after_divergence": len(free_steps) - len(rows),
            "scope": (
                "actual sampled states through first divergent output; "
                "no marginal success probability"),
            "model_calls": 0, "draws": 0, "publications": 0}
