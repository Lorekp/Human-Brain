"""Trainable conventional attention control with the existing span/STOP objective.

There is no recurrent state or simulated-cell claim. Shared token projection
gives q=tanh(W*x_query+b), d=tanh(W*x_document+b); attention is softmax(d*A*q.T/8),
c=attention*q, and h=tanh(F*[d;c]+b_f). Inputs retain their existing modality tags.
Only the unchanged read-only objective, prediction and identity helpers are
inherited. All producer and optimizer arrays below belong to this control.
"""

from __future__ import annotations

import numpy as np

from humanbrain.rate_cell_reader import RateCellReader, SpanInventory, TokenInput, _array_hash

PARAMETER_NAMES = (
    "input_weights",
    "input_bias",
    "attention_weights",
    "fusion_weights",
    "fusion_bias",
    "span_heads",
    "stop_head",
    "stop_bias",
)
POLICY = "conventional_feedforward_attention_word_span_hierarchical_stop_normalized_input_v2"


class ConventionalAttentionReader(RateCellReader):
    """Parameter-matched feedforward comparator; no invocation of the cell producer."""

    def __init__(
        self,
        *,
        mode: str = "trainable",
        seed: int = 0,
        learning_rate: float = 0.001,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        if mode not in ("trainable", "frozen"):
            raise ValueError("conventional reader mode must be trainable or frozen")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("positive finite learning rate required")
        if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
            raise ValueError("positive finite gradient clip required")
        self.mode = mode
        self.learning_rate, self.gradient_clip_norm = learning_rate, gradient_clip_norm
        rng = np.random.default_rng(seed)
        self._weights = {
            "input_weights": rng.standard_normal((64, 386)) / np.sqrt(2),
            "input_bias": np.zeros(64),
            "attention_weights": np.eye(64),
            "fusion_weights": rng.standard_normal((64, 128)) / np.sqrt(128),
            "fusion_bias": np.zeros(64),
            "span_heads": rng.standard_normal((3, 64)) * (0.01 / np.sqrt(64)),
            "stop_head": rng.standard_normal(64) * (0.01 / np.sqrt(64)),
            "stop_bias": np.zeros(1),
        }
        self._first = {k: np.zeros_like(v) for k, v in self._weights.items()}
        self._second = {k: np.zeros_like(v) for k, v in self._weights.items()}
        self._steps = {k: 0 for k in PARAMETER_NAMES}
        self.offer_count = self.update_count = 0
        self.parameters = {
            "policy": POLICY,
            "mode": mode,
            "seed": seed,
            "hidden_dimension": 64,
            "parameter_count": sum(v.size for v in self._weights.values()),
            "parameter_names": PARAMETER_NAMES,
            "producer": "q=tanh(W*xq+b);d=tanh(W*xd+b);a=softmax(d*A*q.T/8);h=tanh(F*[d;a*q]+bf)",
            "initialization": (
                "normalized_input_v2;nonzero_unit_token_plus_modality_tag_squared_norm=2;"
                "default_rng(seed);draw W normal/sqrt2 then F normal/sqrt128 then "
                "span_heads normal*.01/sqrt64 then stop_head normal*.01/sqrt64;"
                "A=I64;input_bias=fusion_bias=stop_bias=0;no_other_draws"
            ),
            "empty_query": "zero context;empty attention;attention_weights gradient zero",
            "learning_rate": learning_rate,
            "gradient_clip_norm": gradient_clip_norm,
            "optimizer": "Adam;beta1=.9;beta2=.999;epsilon=1e-8;no_decay;per_parameter_steps",
            "mask": "STOP_skips_span_parameters_moments_steps;clip_active_coordinates_only",
            "gate_pool": "mean_all_document_word_states_token_weighted;exclude_query_padding",
            "inference": "p_stop>=.5_STOP;otherwise_unique_span_max;tie_STOP",
        }

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        arrays = {"weight_" + k: v.copy() for k, v in self._weights.items()}
        arrays.update({"adam_first_" + k: v.copy() for k, v in self._first.items()})
        arrays.update({"adam_second_" + k: v.copy() for k, v in self._second.items()})
        arrays["adam_steps"] = np.asarray([self._steps[k] for k in PARAMETER_NAMES], dtype=np.int64)
        return arrays

    def snapshot(self) -> dict:
        result = super().snapshot()
        result["allocated_optimizer_bytes"] = sum(
            v.nbytes for v in self._first.values()
        ) * 2 + 8 * len(PARAMETER_NAMES)
        return result

    def _forward(self, query: TokenInput, inventory: SpanInventory) -> dict:
        w = self._weights
        qx = np.zeros((len(query.vectors), 386))
        qx[:, :384], qx[:, 384] = query.vectors, 1
        count = sum(len(window.vectors) for window in inventory.windows)
        dx = np.zeros((count, 386))
        offset = 0
        for window in inventory.windows:
            end = offset + len(window.vectors)
            dx[offset:end, :384], dx[offset:end, 385] = window.vectors, 1
            offset = end
        q = np.tanh(qx @ w["input_weights"].T + w["input_bias"])
        d = np.tanh(dx @ w["input_weights"].T + w["input_bias"])
        if len(q):
            logits = (d @ w["attention_weights"] @ q.T) / 8.0
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            attention = exponentials / exponentials.sum(axis=1, keepdims=True)
        else:
            attention = np.zeros((count, 0))
        context = attention @ q
        fused = np.concatenate((d, context), axis=1)
        h = np.tanh(fused @ w["fusion_weights"].T + w["fusion_bias"])
        scores = np.empty(len(inventory.candidates))
        offset = 0
        for i, window in enumerate(inventory.windows):
            end_word = offset + len(window.vectors)
            heads = h[offset:end_word] @ w["span_heads"].T
            start, end = inventory.local_spans[i].T
            prefix = np.r_[0.0, np.cumsum(heads[:, 2])]
            begin, finish = inventory.window_candidate_ranges[i]
            scores[begin:finish] = (
                heads[start, 0] + heads[end - 1, 1] + (prefix[end] - prefix[start]) / (end - start)
            )
            offset = end_word
        pooled = h.sum(axis=0) / max(1, count)
        stop_logit = float(pooled @ w["stop_head"] + w["stop_bias"][0])
        p_stop = float(np.exp(-np.logaddexp(0.0, -stop_logit)))
        if not np.isfinite(scores).all() or not np.isfinite(stop_logit):
            raise ValueError("nonfinite conventional forward state")
        return {
            "query_input": qx,
            "document_input": dx,
            "query_projection": q,
            "document_projection": d,
            "attention": attention,
            "context": context,
            "fused": fused,
            "hidden": h,
            "scores": scores,
            "pooled": pooled,
            "stop_logit": stop_logit,
            "p_stop": p_stop,
            "document_count": count,
        }

    def loss_and_gradients(
        self, query: TokenInput, inventory: SpanInventory, gold_indices: tuple[int, ...] | None
    ) -> tuple[float, dict, dict]:
        f = self._forward(query, inventory)
        loss, gate_loss, span_loss, coefficients, gate_derivative = self._objective(f, gold_indices)
        w = self._weights
        gradient = {k: np.zeros_like(v) for k, v in w.items()}
        gradient["stop_head"] = gate_derivative * f["pooled"]
        gradient["stop_bias"][0] = gate_derivative
        dh = np.zeros_like(f["hidden"])
        offset = 0
        for i, window in enumerate(inventory.windows):
            count = len(window.vectors)
            h = f["hidden"][offset : offset + count]
            start, end = inventory.local_spans[i].T
            begin, finish = inventory.window_candidate_ranges[i]
            coef = coefficients[begin:finish]
            start_weight, end_weight = np.zeros(count), np.zeros(count)
            np.add.at(start_weight, start, coef)
            np.add.at(end_weight, end - 1, coef)
            difference = np.zeros(count + 1)
            middle = coef / (end - start)
            np.add.at(difference, start, middle)
            np.add.at(difference, end, -middle)
            weights = np.stack((start_weight, end_weight, np.cumsum(difference)[:-1]))
            gradient["span_heads"] += weights @ h
            dh[offset : offset + count] = weights.T @ w["span_heads"] + gate_derivative * w[
                "stop_head"
            ] / max(1, f["document_count"])
            offset += count
        delta = dh * (1 - f["hidden"] ** 2)
        gradient["fusion_weights"] = delta.T @ f["fused"]
        gradient["fusion_bias"] = delta.sum(axis=0)
        dfused = delta @ w["fusion_weights"]
        dd, dc = dfused[:, :64], dfused[:, 64:]
        q, d = f["query_projection"], f["document_projection"]
        da = dc @ q.T
        ds = f["attention"] * (da - (da * f["attention"]).sum(axis=1, keepdims=True))
        gradient["attention_weights"] = d.T @ ds @ q / 8.0
        dd += ds @ q @ w["attention_weights"].T / 8.0
        dq = f["attention"].T @ dc + ds.T @ d @ w["attention_weights"] / 8.0
        dquery = dq * (1 - q**2)
        ddocument = dd * (1 - d**2)
        gradient["input_weights"] = dquery.T @ f["query_input"] + ddocument.T @ f["document_input"]
        gradient["input_bias"] = dquery.sum(axis=0) + ddocument.sum(axis=0)
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradient.values()):
            raise ValueError("nonfinite conventional objective or gradient")
        audit = {
            "loss": loss,
            "gate_loss": gate_loss,
            "span_loss": span_loss,
            "stop_logit": f["stop_logit"],
            "p_stop": f["p_stop"],
            "scores_sha256": _array_hash(f["scores"]),
            "document_word_count": f["document_count"],
            "recurrent_cell_steps": 0,
            "feedforward_hidden_activations": 64 * (len(q) + 2 * len(d)),
            "trajectory_array_bytes": 0,
            "input_array_bytes": f["query_input"].nbytes + f["document_input"].nbytes,
            "producer_array_bytes": sum(
                f[k].nbytes
                for k in (
                    "query_projection",
                    "document_projection",
                    "attention",
                    "context",
                    "fused",
                    "hidden",
                )
            ),
            "span_score_array_bytes": f["scores"].nbytes,
            "attention_entries": f["attention"].size,
            "attention_sha256": _array_hash(f["attention"]),
        }
        return float(loss), gradient, audit

    def observe(
        self, query: TokenInput, inventory: SpanInventory, gold_indices: tuple[int, ...] | None
    ) -> dict:
        # Same frozen Adam arithmetic, using this comparator's actual parameters.
        before = self.checkpoint_sha256
        _, gradients, audit = self.loss_and_gradients(query, inventory, gold_indices)
        active = (
            []
            if self.mode == "frozen"
            else [k for k in PARAMETER_NAMES if not (gold_indices is None and k == "span_heads")]
        )
        norm = float(np.sqrt(sum(float(np.sum(gradients[k] ** 2)) for k in active)))
        if not np.isfinite(norm):
            raise ValueError("nonfinite active gradient norm; no state published")
        scale = min(1.0, self.gradient_clip_norm / norm) if norm else 1.0
        steps_before, proposed = dict(self._steps), {}
        for k in active:
            g = gradients[k] * scale
            step = self._steps[k] + 1
            first = 0.9 * self._first[k] + 0.1 * g
            second = 0.999 * self._second[k] + 0.001 * (g * g)
            corrected_first = first / (1 - 0.9**step)
            corrected_second = second / (1 - 0.999**step)
            weight = self._weights[k] - self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
            if not all(np.isfinite(a).all() for a in (first, second, weight)):
                raise ValueError("nonfinite Adam proposal; no state published")
            proposed[k] = (weight, first, second, step)
        changed = any(
            not np.array_equal(self._weights[k], value[0]) for k, value in proposed.items()
        )
        for k, (weight, first, second, step) in proposed.items():
            self._weights[k], self._first[k], self._second[k], self._steps[k] = (
                weight,
                first,
                second,
                step,
            )
        self.offer_count += 1
        self.update_count += int(changed)
        after_loss, after_gate, after_span, _, _ = self._objective(
            self._forward(query, inventory), gold_indices
        )
        return {
            "mode": self.mode,
            "before": before,
            "after": self.checkpoint_sha256,
            "query_sha256": query.sha256,
            "inventory_sha256": inventory.sha256,
            "gold_indices": None if gold_indices is None else list(gold_indices),
            "before_objective": audit,
            "loss_after": after_loss,
            "gate_loss_after": after_gate,
            "span_loss_after": after_span,
            "gradient_sha256": {k: _array_hash(v) for k, v in gradients.items()},
            "active_parameters": active,
            "gradient_norm": norm,
            "clip_scale": scale,
            "adam_steps_before": steps_before,
            "adam_steps_after": dict(self._steps),
            "parameter_update_applied": changed,
            "native_update_applied": False,
            "offer_count": self.offer_count,
            "update_count": self.update_count,
        }
