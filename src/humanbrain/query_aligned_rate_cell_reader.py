"""Direct query attention for the existing supervised 64-cell span reader.

The old prefix reader is unchanged. Attention and manual BPTT are disclosed
conventional gradient computation over trainable simulated rate-cell state.
"""

from __future__ import annotations

import numpy as np

from humanbrain.rate_cell_reader import (
    PARAMETER_NAMES as PREFIX_PARAMETER_NAMES,
)
from humanbrain.rate_cell_reader import (
    PRODUCER_NAMES as PREFIX_PRODUCER_NAMES,
)
from humanbrain.rate_cell_reader import (
    RateCellReader,
    SpanInventory,
    TokenInput,
    _array_hash,
)

PARAMETER_NAMES = (*PREFIX_PARAMETER_NAMES, "attention_weights", "query_context_weights")
PRODUCER_NAMES = (*PREFIX_PRODUCER_NAMES, *PARAMETER_NAMES[-2:])
POLICY = "query_aligned_rate_cell_word_span_hierarchical_stop_v1"


class QueryAlignedRateCellReader(RateCellReader):
    """Reuse input custody, objective, decision policy and initial prefix state."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        for name in PARAMETER_NAMES[-2:]:
            self._weights[name] = np.eye(64)
            self._first[name] = np.zeros((64, 64))
            self._second[name] = np.zeros((64, 64))
            self._steps[name] = 0
        self.parameters.update(
            policy=POLICY,
            alignment="r=Wx_doc+b;s=r^T*A*u/8;a=softmax_query(s);c=sum(a*u)",
            alignment_initialization="A=I64;V=I64;no_additional_random_draws",
            query_states="recurrent_prefix_rate_states;reset_per_window",
            document_recurrence="h=.5*hprev+.5*tanh(Wx_doc+U*hprev+V*c+b)",
            empty_query="zero_context;empty_attention;A_V_gradient_zero",
            bypass="u=tanh(W0*x_query+b0);h_doc=tanh(W0*x_doc+b0+V0*c);no_U",
            frozen_producer="W_U_b_A_V_parameters_moments_steps_fixed",
            parameter_names=PARAMETER_NAMES,
        )

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().checkpoint_arrays()
        arrays["adam_steps"] = np.asarray([self._steps[k] for k in PARAMETER_NAMES], dtype=np.int64)
        return arrays

    def snapshot(self) -> dict:
        result = super().snapshot()
        result["allocated_optimizer_bytes"] = sum(v.nbytes for v in self._first.values()) * 2 + 64
        return result

    def _forward(self, query: TokenInput, inventory: SpanInventory) -> dict:
        windows = inventory.windows
        width = max((len(window.vectors) for window in windows), default=0)
        q, batch = len(query.vectors), len(windows)
        x = np.zeros((batch, q + width, 386))
        mask = np.zeros((batch, q + width, 1))
        for i, window in enumerate(windows):
            count = len(window.vectors)
            x[i, :q, :384], x[i, :q, 384] = query.vectors, 1
            x[i, q : q + count, :384], x[i, q : q + count, 385] = window.vectors, 1
            mask[i, : q + count] = 1
        w = self._weights
        states = np.zeros((batch, q + width + 1, 64))
        activation = np.zeros((batch, q + width, 64))
        drive = x @ w["input_weights"].T + w["cell_bias"]
        for t in range(q):
            if self.mode == "bypass":
                activation[:, t] = np.tanh(drive[:, t])
                states[:, t + 1] = activation[:, t]
            else:
                activation[:, t] = np.tanh(drive[:, t] + states[:, t] @ w["recurrent_weights"].T)
                states[:, t + 1] = states[:, t] + 0.5 * (activation[:, t] - states[:, t])
        u = states[:, 1 : q + 1]
        transformed_query = u @ w["attention_weights"].T
        if q:
            logits = (drive[:, q:] @ transformed_query.transpose(0, 2, 1)) / 8.0
            exponentials = np.exp(logits - logits.max(axis=2, keepdims=True))
            attention = exponentials / exponentials.sum(axis=2, keepdims=True)
        else:
            attention = np.zeros((batch, width, 0))
        context = attention @ u
        context_drive = context @ w["query_context_weights"].T
        for t in range(q, q + width):
            local = drive[:, t] + context_drive[:, t - q]
            if self.mode == "bypass":
                activation[:, t] = np.tanh(local)
                states[:, t + 1] = activation[:, t] * mask[:, t]
            else:
                activation[:, t] = np.tanh(local + states[:, t] @ w["recurrent_weights"].T)
                states[:, t + 1] = states[:, t] + mask[:, t] * 0.5 * (
                    activation[:, t] - states[:, t]
                )
        scores = np.empty(len(inventory.candidates))
        pooled = np.zeros(64)
        document_count = sum(len(window.vectors) for window in windows)
        for i, window in enumerate(windows):
            h = states[i, q + 1 : q + 1 + len(window.vectors)]
            pooled += h.sum(axis=0)
            start, end = inventory.local_spans[i].T
            heads = h @ w["span_heads"].T
            prefix = np.r_[0.0, np.cumsum(heads[:, 2])]
            begin, finish = inventory.window_candidate_ranges[i]
            scores[begin:finish] = (
                heads[start, 0] + heads[end - 1, 1] + (prefix[end] - prefix[start]) / (end - start)
            )
        pooled /= max(1, document_count)
        stop_logit = float(pooled @ w["stop_head"] + w["stop_bias"][0])
        p_stop = float(np.exp(-np.logaddexp(0.0, -stop_logit)))
        if not np.isfinite(scores).all() or not np.isfinite(stop_logit):
            raise ValueError("nonfinite aligned forward state")
        return {
            "x": x,
            "mask": mask,
            "states": states,
            "activation": activation,
            "scores": scores,
            "pooled": pooled,
            "stop_logit": stop_logit,
            "p_stop": p_stop,
            "query_count": q,
            "document_count": document_count,
            "drive": drive,
            "attention": attention,
            "context": context,
            "transformed_query": transformed_query,
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
        dh = np.zeros_like(f["states"])
        q = f["query_count"]
        for i, window in enumerate(inventory.windows):
            count = len(window.vectors)
            h = f["states"][i, q + 1 : q + 1 + count]
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
            dh[i, q + 1 : q + 1 + count] = weights.T @ w["span_heads"] + gate_derivative * w[
                "stop_head"
            ] / max(1, f["document_count"])
        if self.mode != "bypass":
            delta = np.zeros_like(f["activation"])

            def back_step(t):
                delta[:, t] = (
                    0.5 * f["mask"][:, t] * dh[:, t + 1] * (1 - f["activation"][:, t] ** 2)
                )
                dh[:, t] += (1 - 0.5 * f["mask"][:, t]) * dh[:, t + 1] + delta[:, t] @ w[
                    "recurrent_weights"
                ]

            for t in range(f["activation"].shape[1] - 1, q - 1, -1):
                back_step(t)
            document_delta = delta[:, q:]
            gradient["query_context_weights"] = document_delta.reshape(-1, 64).T @ f[
                "context"
            ].reshape(-1, 64)
            dc = document_delta @ w["query_context_weights"]
            u = f["states"][:, 1 : q + 1]
            da = dc @ u.transpose(0, 2, 1)
            ds = f["attention"] * (da - (da * f["attention"]).sum(axis=2, keepdims=True))
            dr = (ds @ f["transformed_query"]) / 8.0
            dtransformed = (ds.transpose(0, 2, 1) @ f["drive"][:, q:]) / 8.0
            gradient["attention_weights"] = dtransformed.reshape(-1, 64).T @ u.reshape(-1, 64)
            dh[:, 1 : q + 1] += (
                f["attention"].transpose(0, 2, 1) @ dc + dtransformed @ w["attention_weights"]
            )
            for t in range(q - 1, -1, -1):
                back_step(t)
            drive_gradient = delta.copy()
            drive_gradient[:, q:] += dr
            flat = drive_gradient.reshape(-1, 64)
            gradient["input_weights"] = flat.T @ f["x"].reshape(-1, 386)
            gradient["cell_bias"] = flat.sum(axis=0)
            gradient["recurrent_weights"] = delta.reshape(-1, 64).T @ f["states"][:, :-1].reshape(
                -1, 64
            )
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradient.values()):
            raise ValueError("nonfinite aligned objective or gradient")
        audit = {
            "loss": loss,
            "gate_loss": gate_loss,
            "span_loss": span_loss,
            "stop_logit": f["stop_logit"],
            "p_stop": f["p_stop"],
            "scores_sha256": _array_hash(f["scores"]),
            "document_word_count": f["document_count"],
            "recurrent_cell_steps": 0 if self.mode == "bypass" else int(f["mask"].sum()) * 64,
            "trajectory_array_bytes": f["states"].nbytes + f["activation"].nbytes,
            "input_array_bytes": f["x"].nbytes,
            "span_score_array_bytes": f["scores"].nbytes,
            "alignment_array_bytes": sum(
                f[k].nbytes for k in ("drive", "attention", "context", "transformed_query")
            ),
            "attention_entries": f["attention"].size,
            "valid_attention_entries": f["document_count"] * q,
            "attention_sha256": _array_hash(f["attention"]),
        }
        return float(loss), gradient, audit

    def observe(
        self, query: TokenInput, inventory: SpanInventory, gold_indices: tuple[int, ...] | None
    ) -> dict:
        # The frozen predecessor uses module-global parameter tuples. Keep its
        # Adam arithmetic here unchanged while including both new producer arrays.
        before = self.checkpoint_sha256
        _, gradients, audit = self.loss_and_gradients(query, inventory, gold_indices)
        active = (
            []
            if self.mode == "frozen"
            else [
                k
                for k in PARAMETER_NAMES
                if not (
                    (self.mode in ("frozen_producer", "bypass") and k in PRODUCER_NAMES)
                    or (gold_indices is None and k == "span_heads")
                )
            ]
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
        changed = any(not np.array_equal(self._weights[k], v[0]) for k, v in proposed.items())
        for k, (weight, first, second, step) in proposed.items():
            self._weights[k], self._first[k], self._second[k], self._steps[k] = (
                weight,
                first,
                second,
                step,
            )
        self.offer_count += 1
        self.update_count += int(changed)
        after_forward = self._forward(query, inventory)
        after_loss, after_gate, after_span, _, _ = self._objective(after_forward, gold_indices)
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
