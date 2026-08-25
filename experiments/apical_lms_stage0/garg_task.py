"""Garg-2022-style synthetic in-context regression generator.

Episode shape (DendriCL benchmark, DD-DCL SS1.1/SS1.3; Garg et al. 2022):

    w   ~ N(0, w_std^2 I_d)                    # one draw per episode
    x_t ~ N(0, I_d)                            # every position, iid
    y_t = w.x_t + eps,  eps ~ N(0, sigma^2)    # supervised positions only

An episode is a single stream of T = context_mult*d + query_len positions:
the first context_mult*d are supervised (flag=0: after presenting x_t the
label y_t is revealed and drives the local update), the trailing query_len
are decision points (flag=1: error gated OFF, state frozen).

The SOMATIC channel sees only one byte per token: the PREVIOUS label,
tanh-squashed to 0..255 (sentinels at the stream head and on query
positions). The apical channel carries the full analog regressor; nothing
about the current label reaches any compartment before its reveal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    d_task: int = 10
    context_mult: int = 2
    query_len: int = 8
    noise_sigma: float = 1.0
    w_std: float = 1.0

    @property
    def context_len(self) -> int:
        return self.context_mult * self.d_task

    @property
    def total_len(self) -> int:
        return self.context_len + self.query_len


@dataclass(frozen=True)
class Episode:
    """One in-context regression episode (all arrays contiguous)."""

    w: np.ndarray            # (d_task,) latent target of this episode
    xs: np.ndarray           # (T, d_task) regressors, every position
    ys: np.ndarray           # (T,) labels; NaN on decision points
    flags: np.ndarray        # (T,) bool; True = decision point (error gated OFF)
    token_bytes: np.ndarray  # (T,) uint8 somatic channel (previous-label code)

    @property
    def n_positions(self) -> int:
        return int(self.flags.size)


def quantize_label(value: float, scale: float = 16.0) -> int:
    """Deterministic tanh squash of a scalar label onto the byte range."""
    unit = math.tanh(float(value) / float(scale))
    return int(min(255, max(0, int(round(127.5 * unit + 127.5)))))


def build_wa(d_task: int, d_apical: int, seed: int) -> np.ndarray:
    """Frozen input projection W_A with ORTHONORMAL columns, (d_apical, d_task).

    Orthonormality makes the hosted recursion u_A an exact lift of the
    classical W_A = I online-LMS recursion: with z_t = W_A x_t,
    u_A(t+1) = alpha*u_A(t) + gamma*e_t*z_t satisfies u_A(t) = W_A w_ref(t).
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d_apical, d_task)))
    return np.ascontiguousarray(q[:, :d_task])


def generate_episode(rng: np.random.Generator, spec: TaskSpec, *, label_byte_scale: float = 16.0) -> Episode:
    """Draw one episode deterministically from the given generator state."""
    t_total = spec.total_len
    w = rng.normal(0.0, spec.w_std, size=spec.d_task)
    xs = rng.standard_normal((t_total, spec.d_task))
    noise = rng.normal(0.0, spec.noise_sigma, size=t_total)

    ys = np.full(t_total, np.nan, dtype=np.float64)
    supervised = slice(0, spec.context_len)
    ys[supervised] = xs[supervised] @ w + noise[supervised]

    flags = np.zeros(t_total, dtype=np.bool_)
    flags[spec.context_len :] = True

    # Somatic byte stream: strictly past labels; sentinels elsewhere.
    token_bytes = np.empty(t_total, dtype=np.uint8)
    token_bytes[0] = 128  # neutral head
    for t in range(1, t_total):
        if flags[t]:
            token_bytes[t] = 129  # decision-point sentinel
        elif np.isfinite(ys[t - 1]):
            token_bytes[t] = quantize_label(ys[t - 1], label_byte_scale)
        else:
            token_bytes[t] = 128

    return Episode(
        w=np.ascontiguousarray(w),
        xs=np.ascontiguousarray(xs),
        ys=ys,
        flags=flags,
        token_bytes=token_bytes,
    )
