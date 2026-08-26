"""Probe P-B: HTM predictive-state depolarizing bias into the first-spike race.

Lane-native instantiation of the preregistered mechanism
(docs/research/path-decision-synthesis.md, Probe P-B @ e16d22c; frozen constants
and gates in experiments/race_bias_probes/preregistration.json):

    small sustained somatic bias, proportional to matched-segment count, on
    cells whose distal branch matched the recent winner history, consumed by
    the existing basket-cell first-spike race.

Mapping onto the GLIF3 dendritic lane (``humanbrain.cortex.DendriticCortex``)
adopts the HTM principles and constants without porting the algorithmic
simulator (synthesis 2.3 avoid row):

- a *segment* is one grown dendritic branch: a phase-ordered presynaptic
  recent-winner-history key terminating on one postsynaptic token cell;
- a segment *matches* when its key equals an equally long tail of the current
  recent-winner history AND its connected contact count reaches theta
  (13 connected of ~20 grown; each co-occurrence potentiates one contact,
  contact budget cap 20);
- the bias is a somatic CURRENT addition of ``bias_gain_pa`` pA per matched
  segment on the matching cell. Pre-first-spike the fitted membrane response
  is linear in current, so current addition and threshold offset are
  interchangeable up to scaling; addition consumes the incumbent race code
  path unchanged.

Flag-off parity contract: with ``PredictiveBiasConfig(mode="off")`` (or no
config at all) every code path here is skipped entirely — race records and
grown branch state are byte-identical to the incumbent cortex (tested FIRST in
tests/test_race_bias.py).

Determinism: no global RNG state. The shuffled-bias control draws from a
private ``Generator(PCG64(control_seed))`` created once per cortex instance
and advanced exactly once per biased race call, so identical seeds and call
sequences reproduce bit-identical assignments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Frozen constants (experiments/race_bias_probes/preregistration.json).
THETA_CONNECTED_DEFAULT = 13  # connected contacts needed for a segment to match
CONTACT_CAP = 20  # contact budget cap per segment (~20 grown to start)
GROWTH_INCREMENT = 1.0  # strength added per co-occurrence by grow_sequence
BIAS_GAIN_PA_DEFAULT = 30.0  # somatic bias per matched segment [pA, drive units]
DECREMENT_FRACTION_DEFAULT = 0.04  # predictedSegmentDecrement analog, x increment
# HTM-native alias for the same frozen value.
PREDICTED_SEGMENT_DECREMENT_FRACTION = DECREMENT_FRACTION_DEFAULT

_MODES = ("off", "proportional", "shuffled")


@dataclass(frozen=True)
class PredictiveBiasConfig:
    """Frozen arm configuration for the P-B probe."""

    mode: str = "off"  # off | proportional | shuffled
    bias_gain_pa: float = BIAS_GAIN_PA_DEFAULT
    theta_connected: int = THETA_CONNECTED_DEFAULT
    contact_cap: int = CONTACT_CAP
    control_seed: int = 11

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if not float(self.bias_gain_pa) > 0.0:
            raise ValueError("bias_gain_pa must be positive")
        if int(self.theta_connected) < 1:
            raise ValueError("theta_connected must be >= 1")
        if int(self.contact_cap) < 1:
            raise ValueError("contact_cap must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


def connected_contact_count(
    strength: float, contact_cap: int = CONTACT_CAP
) -> int:
    """Connected synapse analog: repetitions potentiate one contact each,
    capped by the ~20-contact growth budget; decayed segments floor at zero."""
    value = float(strength)
    if value <= 0.0:
        return 0
    return min(int(contact_cap), int(value))


def matched_segment_counts(
    cortex,
    context,
    *,
    theta_connected: int = THETA_CONNECTED_DEFAULT,
    contact_cap: int = CONTACT_CAP,
) -> dict[int, int]:
    """Matched-segment count per cell for one recent-winner history."""
    counts: dict[int, int] = {}
    for _order, target, strength in cortex.matching_segments(context):
        if connected_contact_count(strength, contact_cap) >= theta_connected:
            counts[target] = counts.get(target, 0) + 1
    return counts


def apply_predictive_bias(cortex, active, association_drive, context, config):
    """Return a NEW drive array carrying the somatic predictive bias.

    ``active``/``association_drive`` are the incumbent race inputs (aligned).
    Bias lands only on cells already in the raced population: any
    theta-connected matching branch already feeds the Witten-Bell drive, so
    the bias sharpens the existing basket-cell competition instead of
    recruiting cells the incumbent race would never run.
    """
    arr = np.asarray(association_drive, dtype=np.float64).copy()
    if config.mode == "off":
        return arr
    counts = matched_segment_counts(
        cortex,
        context,
        theta_connected=config.theta_connected,
        contact_cap=config.contact_cap,
    )
    lookup = {int(cell): index for index, cell in enumerate(np.asarray(active))}
    biased_cells = sorted(cell for cell in counts if cell in lookup)
    if not biased_cells:
        return arr
    magnitudes = [float(config.bias_gain_pa) * counts[cell] for cell in biased_cells]

    if config.mode == "proportional":
        for cell, magnitude in zip(biased_cells, magnitudes):
            arr[lookup[cell]] += magnitude
        return arr

    # Shuffled control: identical magnitude multiset, random distinct active
    # assignment. One deterministic Generator per cortex, advanced exactly
    # once per biased race call.
    rng = getattr(cortex, "_predictive_bias_rng", None)
    if rng is None:
        rng = np.random.default_rng(config.control_seed)
        cortex._predictive_bias_rng = rng
    positions = rng.choice(len(arr), size=len(magnitudes), replace=False)
    for position, magnitude in zip(positions, magnitudes):
        arr[int(position)] += magnitude
    return arr


def _grow_one_with_history(cortex, history, token) -> None:
    """Grow one token given its full predecessor history.

    Mirrors the per-token body of ``DendriticCortex.grow_sequence`` exactly
    (unigram potentiation + one contact per suffix-order branch, capped by
    ``max_order``), because ``grow_sequence`` itself is sequence-relative and
    cannot be handed a mid-stream continuation without re-counting the whole
    prefix. Coupling note: if ``grow_sequence`` ever changes its per-token
    update, this mirror must change identically (parity tests pin both).
    """
    strength = GROWTH_INCREMENT
    cortex._unigram[int(token)] += strength
    available = min(cortex.max_order, len(history))
    for order in range(1, available + 1):
        ctx = tuple(int(t) for t in history[len(history) - order :])
        targets = cortex._branches.setdefault(ctx, {})
        targets[int(token)] = targets.get(int(token), 0.0) + strength
        cortex._grown_transitions += 1


def grow_with_segment_decrement(
    cortex,
    token_ids,
    *,
    decrement_fraction: float = DECREMENT_FRACTION_DEFAULT,
    sign: int = 1,
    theta_connected: int = THETA_CONNECTED_DEFAULT,
    contact_cap: int = CONTACT_CAP,
) -> dict[str, int]:
    """Replay a ground-truth stream applying the predictedSegmentDecrement
    analog. ``sign=+1`` (the preregistered direction) DECAYS every
    theta-connected converged-but-lost segment by
    ``decrement_fraction x GROWTH_INCREMENT`` (floor 0); ``sign=-1`` is the
    diagnostic '-' side that strengthens losers instead.

    At each replayed position the incumbent race resolves a winner from the
    recent winner history; the ground-truth continuation is then grown
    exactly as ``grow_sequence`` would (see ``_grow_one_with_history``). With
    ``decrement_fraction=0`` no evidence events fire and the replay mutates
    nothing beyond plain growth (bit-parity tested).

    Returns event telemetry (positions, converged events, adjusted losers).
    """
    magnitude = float(decrement_fraction) * GROWTH_INCREMENT
    if magnitude < 0.0:
        raise ValueError("decrement_fraction must be non-negative")
    step = -magnitude if sign >= 0 else magnitude
    history: list[int] = []
    events = {"positions": 0, "converged": 0, "losers_adjusted": 0}
    for raw_token in token_ids:
        token = int(raw_token)
        if history:
            try:
                winner = int(cortex.race(list(history)).winner)
            except RuntimeError:
                winner = None  # nothing grown yet: no race, no evidence
            if winner is not None:
                for order, target, strength in cortex.matching_segments(history):
                    if connected_contact_count(strength, contact_cap) < theta_connected:
                        continue
                    events["converged"] += 1
                    if target == winner or step == 0.0:
                        continue
                    if cortex.adjust_branch_strength(history[-order:], target, step):
                        events["losers_adjusted"] += 1
        _grow_one_with_history(cortex, history, token)
        history.append(token)
        events["positions"] += 1
    return events


def adjust_branch_strength(cortex, context, target, delta) -> bool:
    """Module-level passthrough to the cortex helper so probe code and tests
    have a single import site for segment-strength adjustment."""
    return cortex.adjust_branch_strength(
        tuple(int(token) for token in context), int(target), float(delta)
    )
