"""Hebbian Q-to-P projection (T-ROUTE) — week-1 implementation.

Implements the representation frozen in
docs/research/trained-inter-space-preregistration.md BEFORE any E1 execution:

- sparse ``n x n`` projection W: fan-in ``k_W`` pre partners per post bit drawn
  without replacement via ``numpy.random.default_rng((w_seed, n, k, post))``
  (same projector-keyed convention as ``sdr_union.py``); seeded-random init
  values from the same generator continuation;
- ONE-SHOT outer-product accumulation at token granularity:
  ``dW[pre_bits, post_bits] += M`` per pairing event — synapse-local, no
  epochs, no schedule; VAL episodes never touch W;
- homeostatic IDF-like row scaling ``ln((N_pass+1)/(df+1))`` followed by row
  L2 normalization of nonzero rows (zero rows stay zero);
- kWinner route selection: top-k drive indices, ties broken by ascending bit
  index, exact-k and unique by construction;
- shuffled-W control: nonzero VALUES permuted among the existing support
  (topology fixed), destroying pre->post specificity while preserving the
  value multiset.

Pure numpy, CPU-only, fully deterministic given the frozen config. This module
never reads VAL answers for training and never touches probes.py comparators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_N_BITS = 2048
DEFAULT_FAN_IN = 32
DEFAULT_K_WINNERS = 128
DEFAULT_W_SEED = 7


@dataclass(frozen=True)
class HebbianConfig:
    """Frozen constants of the preregistration §1/§3/§5."""

    n_bits: int = DEFAULT_N_BITS
    w_ones: int = 32
    fan_in: int = DEFAULT_FAN_IN
    w_seed: int = DEFAULT_W_SEED
    k_winners: int = DEFAULT_K_WINNERS

    def __post_init__(self) -> None:
        if not 1 <= self.w_ones <= self.n_bits:
            raise ValueError(f"w_ones must be in [1, n_bits], got {self.w_ones!r}")
        if not 1 <= self.fan_in <= self.n_bits:
            raise ValueError(f"fan_in must be in [1, n_bits], got {self.fan_in!r}")
        if not 1 <= self.k_winners <= self.n_bits:
            raise ValueError(f"k_winners must be in [1, n_bits], got {self.k_winners!r}")

    def idf_scale(self, df_bit: int, n_passages: int) -> float:
        """Homeostatic scale for one post bit: ln((N_pass + 1) / (df + 1))."""
        return math.log((float(n_passages) + 1.0) / (float(df_bit) + 1.0))


def kwinner_indices(drive: np.ndarray, k: int) -> np.ndarray:
    """Top-k drive indices, ties broken by ascending bit index.

    Deterministic under identical inputs; returns exactly ``k`` unique
    indices (k clamped to the drive length).
    """
    drive = np.asarray(drive, dtype=np.float64)
    k = max(1, min(int(k), int(drive.size)))
    # Lexicographic sort on (-value, index): negate for descending value,
    # ascending index breaks ties because argsort is stable on the second key.
    order = np.lexsort((np.arange(drive.size), -drive))
    return order[:k].astype(np.int64)


class SparseProjection:
    """Sparse W with seeded-random topology/init plus one-shot Hebbian dW."""

    def __init__(self, config: HebbianConfig):
        if not isinstance(config, HebbianConfig):
            raise TypeError("config must be a HebbianConfig")
        self.config = config
        n, k, seed = config.n_bits, config.fan_in, config.w_seed
        support = np.empty((n, k), dtype=np.int64)
        values = np.empty((n, k), dtype=np.float64)
        # Same projector-keyed convention as sdr_union.token_sdr_indices:
        # one generator per row keyed by (seed, n, k, post_bit).
        for post in range(n):
            rng = np.random.default_rng((int(seed), int(n), int(k), int(post)))
            support[post] = rng.choice(n, size=k, replace=False)
            values[post] = rng.uniform(0.5, 1.5, size=k)
        self._support = support
        self._values = values
        self._delta = np.zeros((n, k), dtype=np.float64)
        self._position: list[dict[int, int]] = [
            {int(p): j for j, p in enumerate(self._support[post])} for post in range(n)
        ]

    # ------------------------------------------------------------ structure

    @property
    def n_bits(self) -> int:
        return self.config.n_bits

    def support(self) -> np.ndarray:
        """(n_bits, fan_in) int64 matrix of pre-synaptic partner indices."""
        return self._support.copy()

    def values(self) -> np.ndarray:
        """Current weight values on the support (init + accumulated delta)."""
        return (self._values + self._delta).copy()

    def to_dense(self) -> np.ndarray:
        dense = np.zeros((self.n_bits, self.n_bits), dtype=np.float64)
        rows = np.repeat(np.arange(self.n_bits, dtype=np.int64), self.config.fan_in)
        dense[rows, self._support.ravel()] = (self._values + self._delta).ravel()
        return dense

    def route(self, q_pop: np.ndarray, k: int | None = None) -> np.ndarray:
        """drive = W @ q_pop; winners = kWinner selection (prereg §4)."""
        q = np.asarray(q_pop, dtype=np.float64)
        if q.shape != (self.n_bits,):
            raise ValueError(f"q_pop shape {q.shape} != ({self.n_bits},)")
        vals = (self._values + self._delta).astype(np.float64, copy=False)
        drive = np.zeros(self.n_bits, dtype=np.float64)
        gathered = q[self._support]  # (n, fan_in)
        np.einsum("ij,ij->i", gathered, vals, out=drive)
        return kwinner_indices(drive, self.config.k_winners if k is None else k)

    def drive(self, q_pop: np.ndarray) -> np.ndarray:
        """Full drive vector (used by diagnostics; routing selects from it)."""
        return self.to_dense() @ np.asarray(q_pop, dtype=np.float64)

    # ------------------------------------------------------------ learning

    def accumulate_events(
        self, events: list[tuple[list[int], list[int]]]
    ) -> None:
        """One-shot accumulation of dW = pre x post x M over pairing events.

        Each event carries the ACTIVE BIT INDICES (sorted lists) of the pre
        (question-token) and post (passage-token) SDRs; M is binary and folded
        into event multiplicity. Additive; an empty event list is a no-op.
        """
        for pre_bits, post_bits in events:
            for post in post_bits:
                row = self._position[int(post)]
                delta_row = self._delta[int(post)]
                for pre in pre_bits:
                    col = row.get(int(pre))
                    if col is not None:
                        delta_row[col] += 1.0

    def apply_homeostasis(
        self, df: np.ndarray, n_passages: int
    ) -> "HomeostaticProjection":
        """Freeze §3 arm A: IDF-like row scaling + row L2 normalization.

        Returns a NEW read-only view; the trained accumulator is untouched.
        Rows whose scale is zero (df >= n_passages... impossible here since
        ln(1)=0 only when df == n_passages) or whose norm collapses stay zero.
        """
        df = np.asarray(df, dtype=np.int64)
        if df.shape != (self.n_bits,):
            raise ValueError(f"df shape {df.shape} != ({self.n_bits},)")
        scales = np.array(
            [self.config.idf_scale(int(d), int(n_passages)) for d in df],
            dtype=np.float64,
        )
        vals = (self._values + self._delta).astype(np.float64, copy=True)
        norms = np.linalg.norm(vals, axis=1)
        safe = norms > 0.0
        vals[safe] /= norms[safe, None]      # homeostatic target: unit incoming mass
        vals *= scales[:, None]              # IDF gain survives normalization
        return HomeostaticProjection(self.config, self._support.copy(), vals)

    def shuffled_values_copy(self, seed: int) -> "SparseProjection":
        """G-T3b control: permute this projection's values over its support."""
        return shuffled_values_copy(self, seed)


class HomeostaticProjection:
    """Read-only scaled view used for all routing (arm A final weights)."""

    def __init__(
        self,
        config: HebbianConfig,
        support: np.ndarray,
        values: np.ndarray,
    ):
        self.config = config
        self._support = support
        self._values = values

    @property
    def n_bits(self) -> int:
        return self.config.n_bits

    def support(self) -> np.ndarray:
        return self._support.copy()

    def values(self) -> np.ndarray:
        return self._values.copy()

    def to_dense(self) -> np.ndarray:
        dense = np.zeros((self.n_bits, self.n_bits), dtype=np.float64)
        rows = np.repeat(np.arange(self.n_bits, dtype=np.int64), self.config.fan_in)
        dense[rows, self._support.ravel()] = self._values.ravel()
        return dense

    def route(self, q_pop: np.ndarray, k: int | None = None) -> np.ndarray:
        q = np.asarray(q_pop, dtype=np.float64)
        if q.shape != (self.n_bits,):
            raise ValueError(f"q_pop shape {q.shape} != ({self.n_bits},)")
        drive = np.zeros(self.n_bits, dtype=np.float64)
        gathered = q[self._support]
        np.einsum("ij,ij->i", gathered, self._values.astype(np.float64, copy=False), out=drive)
        return kwinner_indices(drive, self.config.k_winners if k is None else k)

    def drive(self, q_pop: np.ndarray) -> np.ndarray:
        return self.to_dense() @ np.asarray(q_pop, dtype=np.float64)


def shuffled_values_copy(proj: SparseProjection | HomeostaticProjection, seed: int):
    """G-T3b shuffled-W control: permute VALUES among the existing support."""
    cls = type(proj)
    shuf_vals = proj.values().copy()
    rng = np.random.default_rng(int(seed))
    flat = shuf_vals.ravel()
    rng.shuffle(flat)  # element-wise permutation, not row-wise
    shuf_vals = flat.reshape(shuf_vals.shape)
    if cls is SparseProjection:
        out = object.__new__(SparseProjection)
        out.config = proj.config
        out._support = proj._support.copy()
        out._values = shuf_vals - proj._delta  # keep delta zeroed in the copy
        out._delta = np.zeros_like(out._values)
        out._position = proj._position
        return out
    if cls is HomeostaticProjection:
        return HomeostaticProjection(proj.config, proj.support(), shuf_vals)
    raise TypeError(f"unsupported projection type {cls!r}")


def population_union(projector, token_ids) -> np.ndarray:
    """Undecayed union (lam = 1) of token SDRs: Q-pop builder (freeze §1)."""
    union = np.zeros(projector.n_bits, dtype=np.float64)
    for token in token_ids:
        union[projector.indices(int(token))] = 1.0
    return union


def passage_union_with_decay(projector, token_ids, lam: float):
    """P-pop with recency decay lam (capacity-study winner lam = 0.90)."""
    if not 0.0 < float(lam) <= 1.0:
        raise ValueError(f"lam must lie in (0, 1], got {lam!r}")
    union = np.zeros(projector.n_bits, dtype=np.float64)
    for token in token_ids:
        union *= float(lam)
        np.maximum(union, projector.vector(int(token)), out=union)
    return union


def pairing_events(
    question_token_ids: list[int],
    passage_token_ids: list[int],
    projector,
    *,
    window: int,
    max_anchor_occurrences: int,
) -> list[tuple[list[int], list[int]]]:
    """Preregistration §2 pairing rule.

    For each question token id tq: anchor positions j with matching passage id
    (first ``max_anchor_occurrences`` by ascending position); each anchor pairs
    tq's SDR bits with every passage token SDR within +/- ``window`` positions.
    Returns events as (pre_bit_list, post_bit_list) with M folded into
    multiplicity (each qualifying (tq, t_p[j']) pair contributes once).
    """
    events: list[tuple[list[int], list[int]]] = []
    passage = [int(t) for t in passage_token_ids]
    n_p = len(passage)
    occurrences: dict[int, list[int]] = {}
    for j, t in enumerate(passage):
        occurrences.setdefault(t, []).append(j)
    pre_cache: dict[int, list[int]] = {}
    post_cache: dict[int, list[int]] = {}
    for tq in dict.fromkeys(int(t) for t in question_token_ids):
        anchors = occurrences.get(tq)
        if not anchors:
            continue
        pre_bits = pre_cache.get(tq)
        if pre_bits is None:
            pre_bits = projector.indices(tq).tolist()
            pre_cache[tq] = pre_bits
        for j in anchors[: int(max_anchor_occurrences)]:
            lo, hi = max(0, j - window), min(n_p, j + window + 1)
            for jp in range(lo, hi):
                tp = passage[jp]
                post_bits = post_cache.get(tp)
                if post_bits is None:
                    post_bits = projector.indices(tp).tolist()
                    post_cache[tp] = post_bits
                events.append((pre_bits, post_bits))
    return events
