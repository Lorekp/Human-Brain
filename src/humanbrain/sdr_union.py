"""SDR union-state memory (U-STATE, Direction A) — week-1 implementation.

Implements the representation frozen in docs/research/ustate-preregistration.md
§1 BEFORE any probe ran:

- seeded token->SDR projection: token id -> exactly ``w_ones`` active bits of
  ``n_bits`` via ``numpy.random.default_rng((seed, n, w, token_id)).choice``;
  ids are deterministic and order-independent by construction;
- union state ``U_t = max(lam * U_{t-1}, s_t)`` — the network state proper
  (custom-architecture-directions.md: "union state IS the network state");
- count-union variant ``C_t = lam * C_{t-1} + s_t`` (recency-weighted
  multiplicities; the analog of an eligibility-trace bank; diagnostic only);
- encode/decode helpers plus pooled cosine-separation diagnostics consumed by
  the D3 logging harness and the capacity precondition reproduction check.

Structural properties (tested in tests/test_sdr_union.py): pairwise token-SDR
overlap concentrates at w^2/n (0.5 bit at defaults), fresh max-unions have
support ~ n(1-(1-w/n)^L), decay steps scale both states by lam^k exactly,
re-encoding a stream is bit-identical, and unions over disjoint alphabets are
near-disjoint BY CONSTRUCTION — cosine collapse is structurally impossible
until decay saturation, and lam is a preregistered knob ({0.90, 0.97, 0.99}).

No reservoir, no learned parameters: pure numpy, fully deterministic given
(seed, n, w, lam, token stream). This module never touches VAL labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DEFAULT_N_BITS = 2048
DEFAULT_W_ONES = 32
DEFAULT_SEED = 7
DEFAULT_LAMBDA = 0.97
# Frozen lambda sweep of the week-1 preregistration (§3).
LAMBDA_SWEEP = (0.90, 0.97, 0.99)
# Week-1 corpora use BytePhraseTokenizer with max_phrases=0: ids 0..255.
BYTE_VOCAB_SIZE = 256


def _validate_dims(n_bits: int, w_ones: int) -> None:
    if not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError(f"n_bits must be a positive int, got {n_bits!r}")
    if not isinstance(w_ones, int) or not 1 <= w_ones <= n_bits:
        raise ValueError(f"w_ones must be an int in [1, n_bits], got {w_ones!r}")


def token_sdr_indices(
    token_id: int,
    *,
    n_bits: int = DEFAULT_N_BITS,
    w_ones: int = DEFAULT_W_ONES,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Pinned projection: sorted active-bit indices for one token id.

    The RNG is keyed by ``(seed, n_bits, w_ones, token_id)`` so each token's
    SDR is independent of iteration order, cache state, or platform — the
    determinism contract the D3 logging harness relies on.
    """
    _validate_dims(n_bits, w_ones)
    token = int(token_id)
    if not 0 <= token < 2**32:
        raise ValueError(f"token_id must be an int in [0, 2**32), got {token_id!r}")
    rng = np.random.default_rng((int(seed), n_bits, w_ones, token))
    return np.sort(rng.choice(n_bits, size=w_ones, replace=False)).astype(np.int64)


class SDRProjector:
    """Cached seeded token->SDR projection (defaults pinned by the freeze)."""

    def __init__(
        self,
        n_bits: int = DEFAULT_N_BITS,
        w_ones: int = DEFAULT_W_ONES,
        seed: int = DEFAULT_SEED,
    ):
        _validate_dims(n_bits, w_ones)
        self.n_bits = int(n_bits)
        self.w_ones = int(w_ones)
        self.seed = int(seed)
        self._indices: dict[int, np.ndarray] = {}
        self._vectors: dict[int, np.ndarray] = {}

    def indices(self, token_id: int) -> np.ndarray:
        cached = self._indices.get(int(token_id))
        if cached is None:
            cached = token_sdr_indices(
                token_id, n_bits=self.n_bits, w_ones=self.w_ones, seed=self.seed
            )
            self._indices[int(token_id)] = cached
        return cached

    def vector(self, token_id: int) -> np.ndarray:
        """Float64 w-hot indicator vector over n_bits."""
        cached = self._vectors.get(int(token_id))
        if cached is None:
            vec = np.zeros(self.n_bits, dtype=np.float64)
            vec[self.indices(token_id)] = 1.0
            cached = vec
            self._vectors[int(token_id)] = cached
        return cached

    # ---------------------------------------------------------------- state

    class _State:
        """Shared mutable engine behind UnionState (max-union + count-union)."""

        def __init__(self, projector: "SDRProjector", lam: float):
            if not isinstance(lam, (int, float)) or not 0.0 < float(lam) <= 1.0:
                raise ValueError(f"lam must lie in (0, 1], got {lam!r}")
            self.projector = projector
            self.lam = float(lam)
            self._union = np.zeros(projector.n_bits, dtype=np.float64)
            self._count = np.zeros(projector.n_bits, dtype=np.float64)
            self._last_sdr = np.zeros(projector.n_bits, dtype=np.float64)
            self.steps = 0

        def step(self, token_id: int) -> None:
            sdr = self.projector.vector(token_id)
            np.multiply(self._union, self.lam, out=self._union)
            np.maximum(self._union, sdr, out=self._union)
            np.multiply(self._count, self.lam, out=self._count)
            np.add(self._count, sdr, out=self._count)
            self._last_sdr = sdr
            self.steps += 1

        def decay(self) -> None:
            """One no-input step: both states scale by lam exactly."""
            np.multiply(self._union, self.lam, out=self._union)
            np.multiply(self._count, self.lam, out=self._count)
            self.steps += 1

        def reset(self) -> None:
            self._union.fill(0.0)
            self._count.fill(0.0)
            self._last_sdr = np.zeros(self.projector.n_bits, dtype=np.float64)
            self.steps = 0


class UnionState:
    """Incremental union state: ``step`` tokens, ``decay`` without input."""

    def __init__(
        self,
        projector: SDRProjector,
        lam: float = DEFAULT_LAMBDA,
    ):
        if not isinstance(projector, SDRProjector):
            raise TypeError("projector must be an SDRProjector")
        self.projector = projector
        self.lam = float(lam)
        self._engine = SDRProjector._State(projector, lam)

    @property
    def value(self) -> np.ndarray:
        return self._engine._union

    @property
    def count_value(self) -> np.ndarray:
        return self._engine._count

    @property
    def last_sdr(self) -> np.ndarray:
        return self._engine._last_sdr

    @property
    def steps(self) -> int:
        return self._engine.steps

    def step(self, token_id: int) -> None:
        self._engine.step(token_id)

    def decay(self) -> None:
        self._engine.decay()

    def reset(self) -> None:
        self._engine.reset()

    def snapshot(self, dtype: type = np.float64) -> np.ndarray:
        return self._engine._union.astype(dtype, copy=True)

    def count_snapshot(self, dtype: type = np.float64) -> np.ndarray:
        return self._engine._count.astype(dtype, copy=True)


@dataclass(frozen=True)
class EncodedSequence:
    """Boundary artifacts of one teacher-forced encode pass."""

    union: np.ndarray  # U_T float64 (the decision-point state)
    count_union: np.ndarray  # C_T float64 (diagnostic feature set)
    last_sdr: np.ndarray  # s_T of the final token
    length: int


@dataclass
class UnionEncoder:
    """Encode token-id streams into union states under one lambda arm."""

    n_bits: int = DEFAULT_N_BITS
    w_ones: int = DEFAULT_W_ONES
    seed: int = DEFAULT_SEED
    lam: float = DEFAULT_LAMBDA
    projector: SDRProjector = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.lam) <= 1.0:
            raise ValueError(f"lam must lie in (0, 1], got {self.lam!r}")
        self.projector = SDRProjector(
            n_bits=self.n_bits, w_ones=self.w_ones, seed=self.seed
        )

    def encode(self, token_ids: Sequence[int]) -> EncodedSequence:
        """Reset-and-replay: bit-identical for identical streams, always."""
        state = UnionState(self.projector, lam=self.lam)
        for token in token_ids:
            state.step(int(token))
        return EncodedSequence(
            union=state.snapshot(),
            count_union=state.count_snapshot(),
            last_sdr=state.last_sdr.copy(),
            length=len(token_ids),
        )


def decode_tokens(
    union: np.ndarray,
    projector: SDRProjector,
    top_k: int = 8,
    candidates: Iterable[int] | None = None,
) -> list[int]:
    """Rank candidate token ids by mean union value over their SDR bits.

    Pinned decode rule of the preregistration §1: score(t) = mean_b U[b] over
    bits b of sdr(t); ties broken by ascending token id. ``candidates``
    defaults to the byte vocabulary (ids 0..255).
    """
    union = np.asarray(union, dtype=np.float64)
    if union.shape != (projector.n_bits,):
        raise ValueError(
            f"union shape {union.shape} does not match projector n_bits "
            f"{projector.n_bits}"
        )
    pool = range(BYTE_VOCAB_SIZE) if candidates is None else candidates
    scored = [
        (-(float(union[projector.indices(t)].mean())), int(t)) for t in pool
    ]
    scored.sort()
    return [t for _, t in scored[: max(1, int(top_k))]]


def cosine_separation(matrix: np.ndarray) -> dict[str, float | int]:
    """Pooled cross-item cosine statistics for boundary states (G-U1 metric).

    Returns the pooled median 1-cosine / cosine across ALL distinct row pairs,
    the max cross-item cosine, and the share of pairs with cosine > 0.90 —
    the exact statistics recorded by the r0b G1 precedent this gate mirrors.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 2:
        raise ValueError("need at least two state rows")
    norms = np.linalg.norm(mat, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("zero-norm state row: separation undefined")
    gram = mat @ mat.T
    cos = gram / np.outer(norms, norms)
    iu = np.triu_indices(mat.shape[0], k=1)
    pair_cos = np.clip(cos[iu], -1.0, 1.0)
    return {
        "n_items": int(mat.shape[0]),
        "n_pairs": int(pair_cos.size),
        "pooled_median_one_minus_cosine": float(np.median(1.0 - pair_cos)),
        "pooled_median_cosine": float(np.median(pair_cos)),
        "max_cross_item_cosine": float(pair_cos.max()),
        "frac_pairs_cosine_above_090": float((pair_cos > 0.90).mean()),
    }
