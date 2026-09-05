"""Frozen actual-input numerical admission; disposable copies, never an acquisition fit.

Call only from the admitted runner. Directions use one RNG(seed=17), in cell,
conventional order; supported then warmed STOP; parameter insertion order. The
64 perturbed evaluations call the existing forward/objective without BPTT.
Finite differences check BPTT against the scalar objective, not an independent
implementation of the forward model. Four validation offers are counted apart
from the main teaching schedule. No old source or model configuration changes.
"""

from __future__ import annotations

import math
import time
from hashlib import sha256
from pathlib import Path

import numpy as np

STEP = 1e-5
FD_ATOL, FD_RTOL = 1e-7, 1e-4
ADAM_ATOL = ADAM_RTOL = 1e-10


def _hash(array):
    return sha256(array.tobytes()).hexdigest()


def _comparison(actual, expected, *, atol=ADAM_ATOL, rtol=ADAM_RTOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    difference = np.abs(actual - expected)
    allowance = atol + rtol * np.abs(expected)
    return {
        "passed": finite and bool(np.all(difference <= allowance)),
        "finite": finite,
        "max_absolute_error": float(difference.max(initial=0)),
        "max_tolerance_ratio": float((difference / allowance).max(initial=0)),
    }


def run_numerical_check(supported, unsupported, *, guard=None):
    """Accept two (TokenInput, SpanInventory, gold tuple/None) triples.

    A failed check returns its partial receipt; the caller must reject admission.
    guard(label), when supplied, shares the parent runner's time/resource budget.
    Inputs remain read-only. Only disposable private parameter arrays are perturbed.
    """
    from humanbrain.conventional_attention_reader import ConventionalAttentionReader
    from humanbrain.query_aligned_rate_cell_reader import QueryAlignedRateCellReader

    began = time.perf_counter()
    report = {
        "status": "running",
        "configuration": {
            "reader_seed": 0,
            "direction_seed": 17,
            "finite_difference_step": STEP,
            "finite_difference_tolerance": [FD_ATOL, FD_RTOL],
            "Adam_tolerance": [ADAM_ATOL, ADAM_RTOL],
            "learning_rate": 0.001,
            "gradient_clip_norm": 1.0,
            "direction_order": "cell,conventional; supported,warmed_STOP; parameter_names",
        },
        "counts": {
            "perturbed_forward": 0,
            "baseline_gradients": 0,
            "predictions": 0,
            "validation_offers": 0,
        },
        "arms": {},
        "scope": (
            "Actual-input engineering check; no qualification or useful-learning claim. "
            "observe internally computes one gradient and one post-update forward/objective "
            "per validation offer."
        ),
    }

    def checkpoint(label):
        if guard is not None:
            guard("numerical:" + label)

    try:
        assert supported[2] is not None and len(supported[2]) > 0
        assert unsupported[2] is None
        root = Path(__file__).resolve().parents[2]
        names = (
            "experiments/r4_integrated/real_input_numerical_check.py",
            "src/humanbrain/rate_cell_reader.py",
            "src/humanbrain/query_aligned_rate_cell_reader.py",
            "src/humanbrain/conventional_attention_reader.py",
        )
        report["source_sha256"] = {
            name: sha256((root / name).read_bytes()).hexdigest() for name in names
        }
        rng = np.random.default_rng(17)
        for arm, constructor in (
            ("cell", QueryAlignedRateCellReader),
            ("conventional", ConventionalAttentionReader),
        ):
            checkpoint(arm + ":construct")
            core = constructor(
                mode="trainable", seed=0, learning_rate=0.001, gradient_clip_norm=1.0
            )
            arm_report = report["arms"][arm] = {"initial": core.snapshot(), "checks": []}
            parameter_names = tuple(core.parameters["parameter_names"])
            assert len(parameter_names) == 8
            for label, (query, inventory, gold) in (
                ("supported", supported),
                ("warmed_STOP", unsupported),
            ):
                checkpoint(arm + ":" + label)
                before = core.checkpoint_sha256
                previous_updates = core.update_count
                item = {
                    "label": label,
                    "before": before,
                    "query_sha256": query.sha256,
                    "inventory_sha256": inventory.sha256,
                    "gold_indices": None if gold is None else list(gold),
                    "directions": [],
                    "Adam_arrays": {},
                }
                arm_report["checks"].append(item)
                report["counts"]["baseline_gradients"] += 1
                loss, gradients, audit = core.loss_and_gradients(query, inventory, gold)
                assert math.isfinite(loss) and all(np.isfinite(g).all() for g in gradients.values())
                item.update(
                    loss=loss,
                    objective=audit,
                    raw_gradient_sha256={k: _hash(v) for k, v in gradients.items()},
                    raw_gradient_norms={k: float(np.linalg.norm(v)) for k, v in gradients.items()},
                )
                assert core.checkpoint_sha256 == before
                report["counts"]["predictions"] += 1
                prediction = core.predict(query, inventory)
                item["prediction"] = {
                    "candidate_index": prediction.candidate_index,
                    "stopped": prediction.stopped,
                    "p_stop": prediction.p_stop,
                    "span_scores_sha256": _hash(prediction.span_scores),
                }
                assert core.checkpoint_sha256 == before
                for name in parameter_names:
                    checkpoint(arm + ":" + label + ":FD:" + name)
                    original = core._weights[name]
                    direction = rng.choice((-1.0, 1.0), size=original.shape) / math.sqrt(
                        original.size
                    )
                    values = []
                    try:
                        for sign in (1, -1):
                            core._weights[name] = original + sign * STEP * direction
                            report["counts"]["perturbed_forward"] += 1
                            values.append(
                                float(core._objective(core._forward(query, inventory), gold)[0])
                            )
                    finally:
                        core._weights[name] = original
                    numerical = (values[0] - values[1]) / (2 * STEP)
                    analytic = math.fsum(float(v) for v in (gradients[name] * direction).flat)
                    result = _comparison(numerical, analytic, atol=FD_ATOL, rtol=FD_RTOL)
                    item["directions"].append(
                        {
                            "parameter": name,
                            "direction_sha256": _hash(direction),
                            "numerical": numerical,
                            "analytic": analytic,
                            **result,
                        }
                    )
                    assert result["passed"], (
                        "finite difference mismatch: " + arm + "/" + label + "/" + name
                    )
                    assert core.checkpoint_sha256 == before
                checkpoint(arm + ":" + label + ":Adam")
                active = [k for k in parameter_names if not (gold is None and k == "span_heads")]
                norm = math.sqrt(
                    math.fsum(float(v) * float(v) for k in active for v in gradients[k].flat)
                )
                scale = min(1.0, 1.0 / norm) if norm else 1.0
                expected, expected_steps = {}, dict(core._steps)
                for name in parameter_names:
                    weight, first, second = (
                        core._weights[name],
                        core._first[name],
                        core._second[name],
                    )
                    if name in active:
                        g = gradients[name] * scale
                        step = expected_steps[name] = expected_steps[name] + 1
                        first = 0.9 * first + 0.1 * g
                        second = 0.999 * second + 0.001 * np.square(g)
                        weight = weight - 0.001 * (first / (1 - 0.9**step)) / (
                            np.sqrt(second / (1 - 0.999**step)) + 1e-8
                        )
                    expected[name] = (weight.copy(), first.copy(), second.copy())
                warm_span = tuple(
                    x.copy()
                    for x in (
                        core._weights["span_heads"],
                        core._first["span_heads"],
                        core._second["span_heads"],
                    )
                )
                span_step = core._steps["span_heads"]
                if gold is None:
                    assert span_step == 1 and np.any(warm_span[1] != 0), (
                        "supported offer did not warm span Adam state"
                    )
                    assert np.count_nonzero(gradients["span_heads"]) == 0
                changed = any(
                    not np.array_equal(core._weights[k], expected[k][0]) for k in parameter_names
                )
                report["counts"]["validation_offers"] += 1
                receipt = core.observe(query, inventory, gold)
                item["validation_offer"] = receipt
                item["norm_check"], item["clip_check"] = (
                    _comparison(receipt["gradient_norm"], norm),
                    _comparison(receipt["clip_scale"], scale),
                )
                assert item["norm_check"]["passed"] and item["clip_check"]["passed"]
                assert receipt["before"] == before and receipt["after"] == core.checkpoint_sha256
                assert receipt["gradient_sha256"] == item["raw_gradient_sha256"]
                assert (
                    receipt["active_parameters"] == active
                    and receipt["adam_steps_after"] == expected_steps
                )
                assert (
                    receipt["parameter_update_applied"] == changed
                    and receipt["native_update_applied"] is False
                )
                assert receipt["offer_count"] == (1 if gold is not None else 2)
                assert core._steps == expected_steps
                assert (
                    receipt["update_count"] == core.update_count == previous_updates + int(changed)
                )
                assert receipt["query_sha256"] == query.sha256
                assert receipt["inventory_sha256"] == inventory.sha256
                assert receipt["gold_indices"] == (None if gold is None else list(gold))
                for name in parameter_names:
                    results = [
                        _comparison(actual, ref)
                        for actual, ref in zip(
                            (core._weights[name], core._first[name], core._second[name]),
                            expected[name],
                            strict=True,
                        )
                    ]
                    item["Adam_arrays"][name] = dict(
                        zip(("weight", "first_moment", "second_moment"), results, strict=True)
                    )
                    assert all(r["passed"] for r in results), "Adam array mismatch: " + name
                if gold is None:
                    assert core._steps["span_heads"] == span_step
                    assert all(
                        np.array_equal(a, b)
                        for a, b in zip(
                            warm_span,
                            (
                                core._weights["span_heads"],
                                core._first["span_heads"],
                                core._second["span_heads"],
                            ),
                            strict=True,
                        )
                    )
                item["after"] = core.snapshot()
                checkpoint(arm + ":" + label + ":finished")
            arm_report["final"] = core.snapshot()
        assert report["counts"] == {
            "perturbed_forward": 64,
            "baseline_gradients": 4,
            "predictions": 4,
            "validation_offers": 4,
        }
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["elapsed_seconds"] = time.perf_counter() - began
    return report
