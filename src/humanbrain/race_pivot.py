"""Race/output-pivot production layer (protocol docs/research/race-pivot-protocol.md §3).

Host-side implementation of the preregistered winner-feedback broadcast loop
(§3.1) and the HOT-2 margin consumers (§3.2). ZERO changes to race semantics,
cell dynamics, or native kernels: everything here drives the existing public
primitives ``RecurrentCellCircuit.race()`` (non-mutating argmax-current probe)
and ``RecurrentCellCircuit.observe()`` (afferent event) exactly as pinned in
protocol §3.1.

Arms
----
- P-S ("context shaping", PRIMARY): the emitted token is ALWAYS the initial
  race winner ``w0``; broadcast/settle shapes only the state carried into
  subsequent positions. Chasing disabled.
- P-R ("settled answer", SECONDARY): chasing enabled; the emitted token is the
  SETTLED winner after the loop terminates (STABLE, UNSTABLE_AT_POSITION, or
  MAX_SETTLE_ROUNDS exhausted -> UNCONVERGED, recorded honestly).
- P-M1: margin abstention consumer (:func:`freeze_tau`, :func:`apply_abstention`,
  :func:`selective_accuracy`) — calibrated on TRAIN margins, τ hash-sealed
  before any held-out scoring.

Logging contract (observation-sink discipline)
----------------------------------------------
The circuit-layer broadcast driver produces EXACTLY ONE sink-visible afferent
event per output decision — the commit injection — because the sink is the
per-decision log seam (protocol §7 T-P8): the driver publishes
``circuit.last_margin_pa`` (decision-time competitive margin) and
``circuit.last_settle_record`` (provisional before the commit observe, replaced
by the final record once the settle loop terminates) immediately before the
commit, so every sink-visible row carries that position's decision data.
Prompt ingestion and broadcast/chase re-injections are settling mechanics, not
decisions: they are injected through the identical state-update path with sink
delivery suspended (the user's callback object is restored verbatim; results
are unaffected — observers are read-only by contract). The MODEL-layer driver
(``RecurrentCellLanguageModel.complete_broadcast``) mirrors ``complete()``
bit-for-bit INCLUDING logging, so it consumes the prompt through plain visible
observes and delegates only the per-position loop to this module.

Determinism: the loop introduces no stochasticity anywhere (argmax race +
deterministic injections); identical seeds/configs give bit-identical streams,
settle profiles, and feature vectors (T-P1).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

MAX_SETTLE_ROUNDS_DEFAULT = 4  # frozen, protocol §3.1
_VARIANTS = ("P-S", "P-R")


@dataclass(frozen=True)
class SettleRecord:
    """Per-output-position broadcast/settle record (protocol §3.1)."""

    position: int
    initial_winner: int
    settled_winner: int
    emitted_token: int
    outcome: str  # STABLE | UNSTABLE_AT_POSITION | UNCONVERGED | PENDING | SETTLE_PROFILE_TERMINAL
    settle_rounds: int  # probe rounds executed (== max_settle_rounds iff UNCONVERGED)
    flipped: bool  # settled != initial
    margin_pa: float  # decision-time competitive margin of the INITIAL winner
    broadcast_b: int
    variant: str
    chase_to_settle: bool
    injections: int  # afferent events driven at this position (commit + broadcasts + chases)


@dataclass(frozen=True)
class BroadcastRunResult:
    """Result payload of one broadcast free-run (duck-typed by the red harness)."""

    tokens: tuple[int, ...]
    settle_records: tuple[SettleRecord, ...]
    position_records: tuple[SettleRecord, ...]
    sequence_flip_rate: float
    stable_fraction: float
    broadcast_b: int
    variant: str
    max_settle_rounds: int
    injection_count: int


def reference_margin(
    currents: np.ndarray,
    winner: int,
    allowed: list[int] | tuple[int, ...] | None = None,
) -> float:
    """Frozen margin formula (protocol §3.2, generalizing ladder_experiments:797).

    ``m = currents[winner] − max(currents[c] for c in allowed_set − {winner})``.
    A single-candidate allowed set yields m = 0.0 explicitly — never a max over
    an empty set.
    """
    current = np.asarray(currents, dtype=np.float64)
    candidates = range(len(current)) if allowed is None else list(allowed)
    competitors = [float(current[c]) for c in candidates if c != int(winner)]
    if not competitors:
        return 0.0
    return float(current[int(winner)]) - max(competitors)


def _inject(circuit, token_id: int, *, visible: bool) -> None:
    """One afferent event through the public observe() path.

    ``visible=False`` suspends sink DELIVERY around the identical state update
    (see module docstring: settling mechanics are not decisions). The user's
    callback attribute is restored verbatim; the state trajectory is untouched
    either way.
    """
    if visible:
        circuit.observe(token_id)
        return
    saved = circuit.observation_sink
    if saved is not None:
        circuit.observation_sink = None
    try:
        circuit.observe(token_id)
    finally:
        if saved is not None:
            circuit.observation_sink = saved


def _validate(
    broadcast_b: int, variant: str, max_settle_rounds: int, chase_to_settle: bool
) -> bool:
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
    if int(broadcast_b) < 1:
        raise ValueError("broadcast_b must be >= 1")
    if int(max_settle_rounds) < 1:
        raise ValueError("max_settle_rounds must be >= 1")
    # P-S never chases (frozen arm definition, §3.1); P-R defaults to chasing.
    return bool(chase_to_settle) and variant == "P-R"


def broadcast_free_run(
    circuit,
    prompt_ids,
    max_new_tokens: int,
    *,
    broadcast_b: int = 1,
    variant: str = "P-S",
    chase_to_settle: bool | None = None,
    max_settle_rounds: int = MAX_SETTLE_ROUNDS_DEFAULT,
    consume_prompt: bool = True,
    record_settle_profile: bool = False,
    margin_allowed_token_ids: list[int] | tuple[int, ...] | None = None,
) -> BroadcastRunResult:
    """Winner-feedback broadcast free-run (protocol §3.1 pseudocode, verbatim).

    ``prompt_ids``: token-id sequence consumed before generation (visible sink
    events only when ``consume_prompt=False`` — the model-layer parity path;
    see module docstring for the logging contract). ``record_settle_profile``
    switches the returned ``settle_records`` from per-position records to the
    POST-RUN TERMINAL settle profile: one record describing how the finished
    sequence settled (committed winner = argmax of the terminal currents under
    the race's own tie-to-lowest-id rule, with its terminal competitive
    margin); the full per-position records remain available on
    ``result.position_records``.
    """
    chase = _validate(broadcast_b, variant, max_settle_rounds, chase_to_settle)
    b = int(broadcast_b)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    prompt = [int(token) for token in prompt_ids]

    if consume_prompt:
        # Circuit-layer entry point owns the sequence lifecycle (mirrors
        # complete()): fresh sequence, then prompt ingestion as internal
        # (non-decision) injections per the module logging contract.
        circuit.reset_sequence()
        for token in prompt:
            _inject(circuit, token, visible=False)

    circuit.last_settle_record = None
    circuit.last_margin_pa = None

    stream: list[int] = []
    records: list[SettleRecord] = []
    total_injections = 0
    for position in range(int(max_new_tokens)):
        # ---- decision point: pure reads only -------------------------------
        currents = np.ascontiguousarray(circuit.output_current_pa())
        w0 = int(circuit.race(margin_allowed_token_ids).winner)
        margin0 = reference_margin(currents, w0, margin_allowed_token_ids)
        w = w0
        injections = 0
        emitted = w0 if variant == "P-S" else w0  # P-R provisional until settled
        provisional = SettleRecord(
            position=position,
            initial_winner=w0,
            settled_winner=w,
            emitted_token=emitted,
            outcome="PENDING",
            settle_rounds=0,
            flipped=False,
            margin_pa=float(margin0),
            broadcast_b=b,
            variant=variant,
            chase_to_settle=chase,
            injections=0,
        )
        circuit.last_margin_pa = float(margin0)
        circuit.last_settle_record = provisional
        # ---- injection #1: the ordinary commit observe (sink-visible) ------
        _inject(circuit, w0, visible=True)
        injections += 1
        # ---- settle loop: protocol §3.1, verbatim control flow -------------
        outcome = "UNCONVERGED"
        rounds = 0
        for _round in range(int(max_settle_rounds)):
            for _ in range(b - 1):
                _inject(circuit, w, visible=False)  # broadcast re-injection
                injections += 1
            probe = int(circuit.race(margin_allowed_token_ids).winner)
            rounds += 1
            if probe == w:
                outcome = "STABLE"
                break
            if not chase:
                outcome = "UNSTABLE_AT_POSITION"
                break
            w = probe  # P-R: attractor moved — chase it
            _inject(circuit, w, visible=False)
            injections += 1
        settled = w
        record = SettleRecord(
            position=position,
            initial_winner=w0,
            settled_winner=settled,
            emitted_token=settled if variant == "P-R" else w0,
            outcome=outcome,
            settle_rounds=rounds,
            flipped=bool(settled != w0),
            margin_pa=float(margin0),
            broadcast_b=b,
            variant=variant,
            chase_to_settle=chase,
            injections=injections,
        )
        circuit.last_settle_record = record
        stream.append(record.emitted_token)
        records.append(record)
        total_injections += injections

    if record_settle_profile:
        # Terminal settle profile (post-run): describe the finished sequence's
        # settled commitment truthfully from the terminal state. Pure reads.
        final_currents = np.ascontiguousarray(circuit.output_current_pa())
        terminal_winner = int(np.argmax(final_currents))  # first occurrence == race tie rule
        terminal_margin = reference_margin(final_currents, terminal_winner)
        last_position = records[-1].position if records else -1
        summary = SettleRecord(
            position=last_position,
            initial_winner=terminal_winner,
            settled_winner=terminal_winner,
            emitted_token=terminal_winner,
            outcome="SETTLE_PROFILE_TERMINAL",
            settle_rounds=sum(rec.settle_rounds for rec in records),
            flipped=bool(records and records[-1].settled_winner != records[-1].initial_winner),
            margin_pa=float(terminal_margin),
            broadcast_b=b,
            variant=variant,
            chase_to_settle=chase,
            injections=total_injections,
        )
        reported_records: tuple[SettleRecord, ...] = (summary,)
    else:
        reported_records = tuple(records)

    flips = sum(1 for rec in records if rec.flipped)
    stables = sum(1 for rec in records if rec.outcome == "STABLE")
    return BroadcastRunResult(
        tokens=tuple(stream),
        settle_records=reported_records,
        position_records=tuple(records),
        sequence_flip_rate=(flips / len(records)) if records else 0.0,
        stable_fraction=(stables / len(records)) if records else 0.0,
        broadcast_b=b,
        variant=variant,
        max_settle_rounds=int(max_settle_rounds),
        injection_count=total_injections,
    )


# ---------------------------------------------------------------------------
# P-M1 margin consumers (protocol §3.2)
# ---------------------------------------------------------------------------
def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def freeze_tau(
    pool,
    tau_low_percentile: float = 5.0,
    tau_high_percentile: float = 95.0,
    sealed_block: dict | None = None,
) -> dict:
    """Calibration freeze (protocol §3.2): τ_low/τ_high percentiles of a TRAIN
    margin pool, returned as a hash-sealed pre-registration block.

    Deterministic: identical inputs give bit-identical blocks (T-P6). When
    ``sealed_block`` is provided, the block recomputed from ``pool`` must match
    its seal EXACTLY — any post-freeze mutation of the calibration inputs fails
    the anchor check (ValueError mentioning anchor/hash ⇒ ABORT before any
    cohort spending, r0b_gate.py pattern).
    """
    values = np.asarray(pool, dtype=np.float64)
    if values.size < 1:
        raise ValueError("tau calibration pool must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("tau calibration pool must be finite")
    if not (0.0 <= tau_low_percentile <= 100.0 and 0.0 <= tau_high_percentile <= 100.0):
        raise ValueError("tau percentiles must lie in [0, 100]")
    tau_low = float(np.percentile(values, tau_low_percentile))
    tau_high = float(np.percentile(values, tau_high_percentile))
    block = {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "tau_low_percentile": float(tau_low_percentile),
        "tau_high_percentile": float(tau_high_percentile),
        "pool_n": int(values.size),
        "pool_sha256": sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
        "method": "numpy.percentile linear interpolation over TRAIN decision margins",
    }
    block["block_sha256"] = sha256(_canonical_json(block)).hexdigest()
    if sealed_block is not None:
        sealed_seal = sealed_block.get("block_sha256")
        if sealed_seal != block["block_sha256"]:
            raise ValueError(
                "anchor check FAILED: recomputed calibration block hash "
                f"{block['block_sha256']} does not match the sealed block hash "
                f"{sealed_seal!r} — post-freeze mutation of calibration inputs "
                "detected; ABORT before cohort spending"
            )
    return block


def apply_abstention(decisions, tau_low: float, tau_high: float) -> "AbstentionLabeling":
    """P-M1 labeling (protocol §3.2): ``m < tau_low`` -> ABSTAIN (excluded from
    the selective-accuracy denominator, counted in coverage; abstained positions
    can never score as correct); ``m > tau_high`` -> high-confidence flag
    (diagnostic); boundary values answer (strict inequalities, frozen).

    Returns the labeled rows (a list of dicts) carrying the P-M1 aggregate
    (``coverage``, ``selective_accuracy``/``selective_span_accuracy``, counts)
    as attributes, so one canonical callable serves both the labeling and the
    scoring seams."""
    labeled: list[dict] = []
    for decision in decisions:
        row = dict(decision)
        margin = float(row["margin_pa"])
        abstain = margin < tau_low
        row["abstained"] = abstain
        row["label"] = "ABSTAIN" if abstain else "ANSWER"
        row["high_confidence"] = bool(margin > tau_high)
        labeled.append(row)
    return AbstentionLabeling(labeled, tau_low=float(tau_low), tau_high=float(tau_high))


class AbstentionLabeling(list):
    """Labeled decision rows plus the frozen selective-accuracy aggregate."""

    def __init__(self, rows, *, tau_low: float, tau_high: float) -> None:
        super().__init__(rows)
        self._tau_low = float(tau_low)
        self._tau_high = float(tau_high)

    @property
    def n_total(self) -> int:
        return len(self)

    @property
    def n_answered(self) -> int:
        return sum(1 for row in self if not row["abstained"])

    @property
    def n_abstained(self) -> int:
        return self.n_total - self.n_answered

    @property
    def n_correct_answered(self) -> int:
        return sum(
            1 for row in self if not row["abstained"] and bool(row.get("would_be_correct"))
        )

    @property
    def coverage(self) -> float:
        return (self.n_answered / self.n_total) if self.n_total else 0.0

    @property
    def selective_accuracy(self) -> float:
        answered = self.n_answered
        return (self.n_correct_answered / answered) if answered else 0.0

    @property
    def selective_span_accuracy(self) -> float:
        return self.selective_accuracy

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "selective_accuracy": self.selective_accuracy,
            "n_total": self.n_total,
            "n_answered": self.n_answered,
            "n_abstained": self.n_abstained,
            "n_correct_answered": self.n_correct_answered,
            "tau_low": self._tau_low,
            "tau_high": self._tau_high,
        }


# Scoring alias: identical canonical object (labeling + aggregate attributes),
# so both P-M1 consumer seams bind the same behavior.
selective_accuracy = apply_abstention
