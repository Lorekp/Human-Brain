"""R3 pilot glue: frozen tokenizer->pool mapping + observation-sink replay driver.

Wires the D2 segment-growth mechanics (``humanbrain.segment_growth``) to the
GLIF3 dendritic lane for the G-S0 fixture health check
(``docs/research/r3-segment-growth-preregistration.md`` §6 G-S0) under the
frozen concept-token encoding of ``docs/research/r3-tokenizer-freeze.md``.

Frozen pool-mapping convention (tokenizer-freeze §4, used identically in every
arm, pure arithmetic, no RNG at mapping time):

- ``initial_pool(id) = id % k`` with ``k = 32`` (prereg §2 ``cellsPerColumn``):
  the token's home-candidate offset inside its own 32-cell block.
- ``slot(id) = id // k``: the slot band holding the token's block.
- Each token identity owns a DISJOINT block of ``k = 32`` candidate cells
  (prereg §2); blocks are laid out as ``base = slot*k*k + initial_pool*k``
  above the lane's gid space so presynaptic winner ids (lane cells) and pool
  cells never share addresses.  ``(pool, slot) <-> id`` is a pure bijection;
  zero stored state.
- Feasibility assertion at build time (freeze §4 band-aware form):
  ``(max_token_id + 1) * k <= configured_cells`` -- violation => STOP before
  the run.  Shrink TRAIN whole, never constants.

Replay driver contract (this dispatch): winner sets ``W_{t-1}`` are supplied
per token from the lane's observation-sink spike stream -- a
``RecurrentCellCircuit.observation_sink`` callback reads the per-cell
per-token spike-count bank after each :meth:`observe`, and the driver feeds
the previous token's spiking set to ``SegmentGrowthTracker.step``.  Arms share
corpus, slice, seed, schedule, and instrumentation; A2 growth-off keeps its
structural-zero predicted-path tripwire (any nonzero value => STOP).

No forbidden trainable components: no gradient objects, no ANN modules --
pure local TM-style rules over recorded spikes (prereg §1 tripwire).
Determinism: identical (seed, slice, constants) => bit-identical telemetry,
segment arrays, and logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.segment_growth import SegmentGrowthTracker

# Frozen concept-band layout (humanbrain.concept_tokens; freeze §1).
try:  # single source of truth if importable
    from humanbrain.concept_tokens import CONCEPT_ID_BASE, MAX_CONCEPTS

    CONCEPT_ID_CEILING = CONCEPT_ID_BASE + MAX_CONCEPTS  # exclusive: [1024, 1536)
except Exception:  # pragma: no cover - frozen literals fallback
    CONCEPT_ID_BASE = 1024
    MAX_CONCEPTS = 512
    CONCEPT_ID_CEILING = 1536

# k = 32 candidate cells per token pool (prereg §2; freeze §4).
K_POOL = SegmentGrowthTracker.CELLS_PER_POOL

# Construction seed 17 for ALL arms (prereg §3).
CONSTRUCTION_SEED = 17


def initial_pool(token_id: int, k: int = K_POOL) -> int:
    """Frozen §4 mapping: home-candidate offset = ``id mod k``."""
    return int(token_id) % k


def pool_slot(token_id: int, k: int = K_POOL) -> int:
    """Frozen §4 mapping: within-family position = ``id // k``."""
    return int(token_id) // k


def pool_cells_for_id(token_id: int, *, cell_offset: int, k: int = K_POOL) -> tuple[int, ...]:
    """The token's disjoint k-cell candidate block (see module docstring).

    ``cell_offset`` lifts pool cells above the lane's gid space so contacts
    grown toward lane-spike winner ids can never alias a pool address.
    """
    p = initial_pool(token_id, k)
    s = pool_slot(token_id, k)
    base = cell_offset + s * k * k + p * k
    return tuple(base + j for j in range(k))


def required_pool_cells(max_token_id: int, k: int = K_POOL) -> int:
    """Cell count needed by the block layout up to ``max_token_id``."""
    slots = max_token_id // k + 1
    return int(slots) * k * k


def assert_feasibility(max_token_id: int, configured_cells: int, *, k: int = K_POOL) -> None:
    """Freeze §4 build-time assertion; violation => STOP before the run."""
    literal = (int(max_token_id) + 1) * k
    layout = required_pool_cells(int(max_token_id), k)
    if literal > int(configured_cells) or layout > int(configured_cells):
        raise RuntimeError(
            "STOP: feasibility assertion failed - "
            f"(max_token_id+1)*k = {literal}, layout needs {layout}, "
            f"configured_cells = {configured_cells}. "
            "Shrink TRAIN books whole; never the constants."
        )


class SinkSpikeRecorder:
    """Observation-sink callback recording the per-token spike stream.

    Attached as ``circuit.observation_sink``; invoked once per
    :meth:`observe` AFTER the boundary update, where reading the raw banks is
    sanctioned (recurrent.py observation_sink docstring).  Records, per token:
    the spiking-cell set (the winner-set source for ``W_{t-1}``), total spike
    count, simulated sub-steps, and per-step activity density for the prereg
    §6 spike-rate health pre-flight.
    """

    def __init__(self) -> None:
        self.token_ids: list[int] = []
        self.winner_sets: list[tuple[int, ...]] = []
        self.spike_totals: list[int] = []
        self.simulated_steps: list[int] = []
        self.activity_density: list[float] = []

    def __call__(self, circuit: RecurrentCellCircuit, step) -> None:
        counts = np.asarray(circuit._token_spike_counts, dtype=np.uint32)
        winners = tuple(int(g) for g in np.flatnonzero(counts))
        density = len(winners) / float(counts.size * max(step.simulated_steps, 1))
        self.token_ids.append(int(step.token_id))
        self.winner_sets.append(winners)
        self.spike_totals.append(int(step.reservoir_spikes))
        self.simulated_steps.append(int(step.simulated_steps))
        self.activity_density.append(float(density))

    def mean_activity_density(self) -> float:
        if not self.activity_density:
            raise ValueError("no tokens observed")
        return float(np.mean(self.activity_density))

    def spike_rate_health(self, floor_fraction: float = 0.001) -> dict[str, Any]:
        """Prereg §6 pre-flight: population activity >= 0.1% of the analytic
        all-cell/all-substep steady-state ceiling on average."""
        mean_density = self.mean_activity_density()
        return {
            "mean_activity_density": mean_density,
            "floor_fraction": floor_fraction,
            "passed": bool(mean_density >= floor_fraction),
        }


@dataclass(frozen=True)
class ReplayRow:
    """One replay step's telemetry (burst telemetry contract)."""

    step_index: int
    token_id: int
    burst: bool
    elected_winner: int | None
    fired_cells: tuple[int, ...]
    punished_segments: tuple[tuple[int, int], ...]
    allocated_segment: bool
    removed_contacts: int
    winner_set_size: int
    pass_index: int


@dataclass
class ArmRun:
    """Result of one arm's replay over the slice."""

    arm: str
    rows: list[ReplayRow] = field(default_factory=list)
    passes: list[dict[str, Any]] = field(default_factory=list)
    permanences_sha256: str = ""
    telemetry_sha256: str = ""
    total_segments: int = 0
    total_contacts: int = 0
    elections: int = 0

    @property
    def predicted_path_fraction(self) -> float:
        """f_pred over the whole sweep (prereg §5): fraction of steps whose
        arriving pool contained >= 1 segment-depolarized cell."""
        if not self.rows:
            raise ValueError("no replay steps recorded")
        predicted = sum(1 for r in self.rows if r.fired_cells)
        return predicted / len(self.rows)

    def burst_fraction_by_pass(self) -> list[float]:
        out = []
        for entry in self.passes:
            out.append(float(entry["burst_fraction"]))
        return out


def run_arm(
    token_stream: Sequence[int],
    *,
    arm: str,
    lane_cells: int = 96,
    passes: int = 2,
    construction_seed: int = CONSTRUCTION_SEED,
    token_cells: int = CONCEPT_ID_CEILING,
    spike_rate_floor_fraction: float = 0.001,
) -> tuple[ArmRun, SinkSpikeRecorder, dict[str, Any]]:
    """Replay ``token_stream`` for ``passes`` ordered passes through one arm.

    A1 = growth-on (full mechanism), A2 = growth-off control.  Winner sets
    W_{t-1} come from the observation-sink spike stream of a fresh N=lane_cells
    GLIF3 circuit consuming the same stream (seeded config).  Step t uses the
    spiking set captured at step t-1; step 0 runs with an empty history.
    """
    if arm not in ("A1", "A2"):
        raise ValueError(f"arm must be 'A1' or 'A2', got {arm!r}")
    stream = [int(t) for t in token_stream]
    if not stream:
        raise ValueError("token_stream must be non-empty")
    max_id = max(stream)
    if max_id >= token_cells:
        raise ValueError("token id outside circuit output population")

    # Freeze §4 build-time feasibility assertion, evaluated against the full
    # concept-band ceiling so the wiring is proven for C1's worst case too.
    ceiling_id = int(token_cells) - 1
    configured_cells = int(lane_cells) + required_pool_cells(ceiling_id)
    assert_feasibility(ceiling_id, configured_cells)

    distinct = sorted(set(stream))
    offset = lane_cells  # pools live above lane gids; namespaces disjoint
    pools = {tid: pool_cells_for_id(tid, cell_offset=offset) for tid in distinct}
    tracker = SegmentGrowthTracker(
        pools,
        growth_on=(arm == "A1"),
        construction_seed=construction_seed,
    )

    recorder = SinkSpikeRecorder()
    config = RecurrentCircuitConfig(reservoir_cells=lane_cells)
    circuit = RecurrentCellCircuit(token_cells, config=config)
    # Attach through the documented R0-c Thread-1 seam: RecurrentCellCircuit
    #.__init__ resets _observation_sink to None after storing the kwarg, so
    # the property setter is the only reliable attachment point.
    circuit.observation_sink = recorder

    run = ArmRun(arm=arm)
    full_stream: list[int] = []
    for _ in range(passes):
        full_stream.extend(stream)

    w_prev: tuple[int, ...] = ()
    pass_index = 0
    steps_in_pass = 0
    for i, tok in enumerate(full_stream):
        circuit.observe(tok)
        result = tracker.step(int(tok), w_prev)
        row = ReplayRow(
            step_index=i,
            token_id=tok,
            burst=result.burst,
            elected_winner=result.elected_winner,
            fired_cells=tuple(result.fired_cells),
            punished_segments=tuple(result.punished_segments),
            allocated_segment=result.allocated_segment,
            removed_contacts=result.removed_contacts,
            winner_set_size=len(w_prev),
            pass_index=pass_index,
        )
        run.rows.append(row)
        run.elections += 1 if result.elected_winner is not None else 0
        steps_in_pass += 1
        w_prev = recorder.winner_sets[-1]
        if steps_in_pass == len(stream):
            window = run.rows[-steps_in_pass:]
            predicted = sum(1 for r in window if r.fired_cells)
            burst = sum(1 for r in window if r.burst)
            run.passes.append(
                {
                    "pass_index": pass_index,
                    "steps": steps_in_pass,
                    "predicted_steps": predicted,
                    "burst_steps": burst,
                    "predicted_fraction": predicted / steps_in_pass,
                    "burst_fraction": burst / steps_in_pass,
                }
            )
            pass_index += 1
            steps_in_pass = 0

    # Structural-zero tripwire for the growth-off control (prereg §4).
    if arm == "A2":
        nonzero = [r for r in run.rows if r.fired_cells]
        if nonzero or tracker.total_segments() or tracker.total_contacts():
            raise RuntimeError(
                "STOP: A2 growth-off produced a nonzero predicted path or "
                f"grew state ({len(nonzero)} predicted steps, "
                f"{tracker.total_segments()} segments) => leakage/bug"
            )
        if any(r.burst is False for r in run.rows):
            raise RuntimeError("STOP: A2 must burst on every step")

    run.total_segments = tracker.total_segments()
    run.total_contacts = tracker.total_contacts()
    run.permanences_sha256 = hashlib.sha256(
        np.ascontiguousarray(tracker.all_permanences()).tobytes()
    ).hexdigest()
    run.telemetry_sha256 = hashlib.sha256(
        json.dumps(
            [
                [
                    r.step_index,
                    r.token_id,
                    r.burst,
                    r.elected_winner,
                    list(r.fired_cells),
                    [list(p) for p in r.punished_segments],
                    r.allocated_segment,
                    r.removed_contacts,
                    r.winner_set_size,
                ]
                for r in run.rows
            ],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    diagnostics = {
        "lane_cells": lane_cells,
        "token_cells": token_cells,
        "distinct_ids": len(distinct),
        "max_token_id": max_id,
        "construction_seed": construction_seed,
        "spike_rate_health": recorder.spike_rate_health(spike_rate_floor_fraction),
        "first_step_burst": bool(run.rows[0].burst),
        "mean_winner_set_size": float(np.mean([len(w) for w in recorder.winner_sets])),
        "tracker_growth_on": tracker.growth_on,
    }
    return run, recorder, diagnostics


def gs0_health_checks(a1: ArmRun, a2: ArmRun, a1_rerun: ArmRun, recorder: SinkSpikeRecorder) -> dict[str, Any]:
    """G-S0 wiring-health battery (health only; gates need C1).

    (i) clean run: no STOP raised anywhere above; every step logged.
    (ii) predicted-path fraction computable in both arms.
    (iii) burst telemetry present for every step.
    (iv) bit-identical reruns (prereg §1 determinism tripwire).
    (v) A2 structural-zero predicted-path tripwire held.
    (vi) spike-rate pre-flight (prereg §6) passed.
    """
    checks: dict[str, Any] = {}
    checks["clean_run"] = bool(a1.rows) and bool(a2.rows) and bool(a1_rerun.rows)
    try:
        checks["f_pred_A1"] = a1.predicted_path_fraction
        checks["f_pred_A2"] = a2.predicted_path_fraction
        checks["predicted_path_fraction_computable"] = True
    except Exception as exc:  # pragma: no cover - defensive
        checks["predicted_path_fraction_computable"] = False
        checks["error"] = str(exc)
    checks["burst_telemetry_present"] = all(isinstance(r.burst, bool) for r in (*a1.rows, *a2.rows))
    checks["bit_identical_reruns"] = (
        a1.telemetry_sha256 == a1_rerun.telemetry_sha256
        and a1.permanences_sha256 == a1_rerun.permanences_sha256
    )
    checks["a2_structural_zero"] = (
        all(r.burst for r in a2.rows)
        and all(r.fired_cells == () for r in a2.rows)
        and a2.total_segments == 0
        and a2.total_contacts == 0
    )
    checks["spike_rate_preflight"] = bool(recorder.spike_rate_health()["passed"])
    checks["all_passed"] = bool(
        checks["clean_run"]
        and checks["predicted_path_fraction_computable"]
        and checks["burst_telemetry_present"]
        and checks["bit_identical_reruns"]
        and checks["a2_structural_zero"]
        and checks["spike_rate_preflight"]
    )
    return checks
