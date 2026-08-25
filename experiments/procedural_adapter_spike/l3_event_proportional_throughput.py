"""L3 spike benchmark: event-proportional stepping throughput at 1e6 scale.

Measures procedural-runtime token throughput when residency has ratcheted to
a large dormant population while only a sparse set of cells integrates per
token -- the steady-state training regime that motivates event-proportional
work semantics.

Everything is driven through the raw C-ABI with a documented stress GLIF
configuration: unit leak makes voltage mirror synaptic current exactly, so a
spiked cell's fast state reaches canonical zero in tens of steps and its
eligibility tail can retire to the dormant store inside this script's runtime
budget. The mechanism, fail-stop contracts, and bit-parity obligations are
identical under production constants; only the tail-flush horizon differs
(see FINDINGS.md for the measured production flush horizon and why it
dominates near-term training runs).

Output: l3_event_proportional_throughput.json next to this script.
"""

from __future__ import annotations

import ctypes
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from humanbrain.cells import AllenGlif5Parameters  # noqa: E402
from humanbrain.native import _F64_POINTER, _GlifConfig, _load_library  # noqa: E402

MASK64 = 0xFFFFFFFFFFFFFFFF


class NativeEvent(ctypes.Structure):
    _fields_ = [
        ("due_step", ctypes.c_uint64),
        ("target_gid", ctypes.c_uint64),
        ("source_gid", ctypes.c_uint64),
        ("edge_slot", ctypes.c_uint64),
        ("current_pa", ctypes.c_double),
    ]


class NativeConfig(ctypes.Structure):
    _fields_ = [
        ("configured_cells", ctypes.c_uint64),
        ("fanout", ctypes.c_uint64),
        ("seed", ctypes.c_uint64),
        ("resident_capacity", ctypes.c_size_t),
        ("event_capacity", ctypes.c_size_t),
        ("spike_capacity", ctypes.c_size_t),
        ("interval_distinct_capacity", ctypes.c_size_t),
        ("recurrent_weight_pa", ctypes.c_double),
        ("recurrent_delay_steps", ctypes.c_uint64),
        ("recurrent_decay", ctypes.c_double),
        ("eligibility_decay", ctypes.c_double),
        ("stepping_mode", ctypes.c_uint32),
    ]


class NativeCounters(ctypes.Structure):
    _fields_ = [
        ("current_step", ctypes.c_uint64),
        ("resident_peak", ctypes.c_size_t),
        ("resident_distinct", ctypes.c_size_t),
        ("updated_distinct", ctypes.c_size_t),
        ("spiking_distinct", ctypes.c_size_t),
        ("cell_step_evaluations", ctypes.c_uint64),
        ("spike_events", ctypes.c_uint64),
        ("generated_edges", ctypes.c_uint64),
        ("delivered_events", ctypes.c_uint64),
        ("configured_ids_scanned", ctypes.c_uint64),
        ("queue_peak", ctypes.c_size_t),
        ("native_payload_bytes", ctypes.c_size_t),
        ("retained_nonzero_cells", ctypes.c_size_t),
        ("overflow_code", ctypes.c_uint32),
    ]


LIB = _load_library()
LIB.humanbrain_procedural_runtime_create.argtypes = [
    ctypes.POINTER(_GlifConfig),
    ctypes.POINTER(NativeConfig),
    _F64_POINTER,
    _F64_POINTER,
    _F64_POINTER,
]
LIB.humanbrain_procedural_runtime_create.restype = ctypes.c_void_p
LIB.humanbrain_procedural_runtime_destroy.argtypes = [ctypes.c_void_p]
LIB.humanbrain_procedural_runtime_advance.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.POINTER(NativeEvent),
    ctypes.c_size_t,
]
LIB.humanbrain_procedural_runtime_advance.restype = ctypes.c_uint32
LIB.humanbrain_procedural_runtime_counts.argtypes = [ctypes.c_void_p] + [
    ctypes.POINTER(ctypes.c_size_t)
] * 3
LIB.humanbrain_procedural_runtime_counts.restype = ctypes.c_uint32
LIB.humanbrain_procedural_runtime_dormant_snapshot.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
LIB.humanbrain_procedural_runtime_dormant_snapshot.restype = ctypes.c_uint32
LIB.humanbrain_procedural_runtime_snapshot.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(NativeCounters),
]
LIB.humanbrain_procedural_runtime_snapshot.restype = ctypes.c_uint32


STRESS_GLIF = _GlifConfig(
    1,      # cell_count (unused by the sparse runtime)
    2,      # after_channel_count
    3,      # spike_cut_steps -- short cut keeps integration lifetimes tight
    1.0,    # leak_ns -> v' = I exactly; fast state flushes with syn current
    1.0,    # voltage_step_scale
    0.0,    # threshold_beta_scale
    0.9,    # threshold_exp_decay
    0.0,    # threshold_voltage_factor
    0.0,    # threshold_beta_factor
    0.9,    # spike_threshold_decay
    1.0,    # threshold_inf_relative_mv -- spike floor
    0.0,    # voltage_reset_scale
    0.0,    # voltage_reset_offset_mv
    0.5,    # spike_cut_threshold_decay
    0.0,    # threshold_spike_increment_mv
)
STRESS_DECAYS = np.array([0.5, 0.5], dtype=np.float64)
STRESS_AMPLITUDES = np.array([0.0, 0.0], dtype=np.float64)


def production_flush_horizon() -> dict:
    """Measured decay arithmetic for the shipped constants: how many steps
    until a post-spike cell's fast state reaches canonical rest."""
    parameters = AllenGlif5Parameters()
    per_step = abs(
        1.0 - parameters.dt_ms / parameters.capacitance_pf * parameters.leak_ns
    )
    eligibility_factor = math.exp(-parameters.dt_ms / 80.0)

    def horizon(factor: float) -> int:
        if not (0.0 < factor < 1.0):
            return -1  # sentinel: never reaches canonical rest
        return int(math.log(1e-300 / 30.0) / math.log(factor))

    return {
        "voltage_decay_per_step": per_step,
        "voltage_flush_horizon_steps": horizon(per_step),
        "eligibility_decay_per_step": eligibility_factor,
        "eligibility_flush_horizon_steps": horizon(eligibility_factor),
        "note": (
            "Under shipped constants a spiked cell keeps sub-normal "
            "voltage/threshold tails for ~10^6 steps, so natural dormancy "
            "horizons dominate near-term training; the stress configuration "
            "below isolates the mechanism's throughput instead."
        ),
    }


class Runtime:
    """Raw C-ABI runtime with internal step bookkeeping."""

    def __init__(
        self,
        *,
        stepping_mode: str,
        configured_cells: int,
        resident_capacity: int,
        event_capacity: int,
        weight_pa: float,
    ) -> None:
        mode_id = {"full_touch": 0, "event_proportional": 1}[stepping_mode]
        self.stepping_mode = stepping_mode
        config = NativeConfig(
            configured_cells,
            4,
            71,
            resident_capacity,
            event_capacity,
            event_capacity,
            max(resident_capacity, event_capacity),
            weight_pa,
            1,
            # Aggressive synaptic decay: the 150 k pA drive must flush below
            # threshold within a few steps or driven cells re-spike every
            # refractory cycle and blow the spike budget (the shipped
            # production tau keeps them supra-threshold for thousands of
            # steps -- that ratchet is exactly what the flush-horizon metric
            # above documents).
            1e-8,
            math.exp(-0.05 / 80.0),
            mode_id,
        )
        self.handle = LIB.humanbrain_procedural_runtime_create(
            ctypes.byref(STRESS_GLIF),
            ctypes.byref(config),
            STRESS_DECAYS.ctypes.data_as(_F64_POINTER),
            STRESS_DECAYS.ctypes.data_as(_F64_POINTER),
            STRESS_AMPLITUDES.ctypes.data_as(_F64_POINTER),
        )
        if not self.handle:
            raise RuntimeError(f"native create failed ({stepping_mode})")
        self.buffer = (NativeEvent * (event_capacity // 4))()
        self.counters = NativeCounters()
        self.step = 0

    def close(self) -> None:
        if self.handle:
            LIB.humanbrain_procedural_runtime_destroy(self.handle)
            self.handle = None

    def advance(self, steps: int, targets) -> float:
        count = len(targets)
        capacity = len(self.buffer)
        if count > capacity:
            raise RuntimeError("event batch exceeds buffer")
        buffer = self.buffer
        due = self.step + steps - 1  # deliver on the window's final step
        for index, gid in enumerate(targets):
            entry = buffer[index]
            entry.due_step = due
            entry.target_gid = gid
            entry.source_gid = gid
            entry.edge_slot = 0
            entry.current_pa = 150_000.0
        start = time.perf_counter()
        status = LIB.humanbrain_procedural_runtime_advance(
            self.handle, steps, buffer, count
        )
        elapsed = time.perf_counter() - start
        if status != 0:
            raise RuntimeError(
                f"advance failed with native status {status} at step {self.step}"
            )
        self.step += steps
        return elapsed

    def settle(self, steps: int) -> None:
        """Advance without events until every transient tail has flushed."""
        self.advance(steps, [])

    def live_state(self) -> dict:
        active = ctypes.c_size_t()
        pending = ctypes.c_size_t()
        spikes = ctypes.c_size_t()
        dormant = ctypes.c_size_t()
        assert LIB.humanbrain_procedural_runtime_counts(
            self.handle,
            ctypes.byref(active),
            ctypes.byref(pending),
            ctypes.byref(spikes),
        ) == 0
        assert LIB.humanbrain_procedural_runtime_dormant_snapshot(
            self.handle, None, 0, ctypes.byref(dormant)
        ) == 0
        return {
            "active_residents": active.value - dormant.value,
            "dormant": dormant.value,
            "pending": pending.value,
        }

    def total_work(self) -> int:
        """Cumulative cell-step evaluations (counters-only snapshot)."""
        assert (
            LIB.humanbrain_procedural_runtime_snapshot(
                self.handle, None, 0, None, 0, None, 0, ctypes.byref(self.counters)
            )
            == 0
        )
        return self.counters.cell_step_evaluations


def fresh_targets(count: int, token_index: int, low: int, high: int) -> list[int]:
    """Deterministic xorshift pseudo-random distinct targets inside
    ``[low, high)`` -- callers keep this range disjoint from any fixed
    reactivation targets so canonical event keys stay unique."""
    span = high - low
    state = (0x9E3779B97F4A7C15 * (token_index + 1)) & MASK64
    out: list[int] = []
    seen: set[int] = set()
    guard = 0
    while len(out) < count and guard < count * 20:
        guard += 1
        state ^= (state << 13) & MASK64
        state ^= state >> 7
        state ^= (state << 17) & MASK64
        gid = low + state % span
        if gid not in seen:
            seen.add(gid)
            out.append(gid)
    if len(out) < count:
        raise RuntimeError("target generation exhausted")
    return out


def build_residency(runtime: Runtime, total: int, per_window: int) -> float:
    wall = 0.0
    built = 0
    while built < total:
        chunk = min(per_window, total - built)
        wall += runtime.advance(100, range(built, built + chunk))
        built += chunk
    return wall


def token_stream(runtime: Runtime, *, tokens: int, fresh: int, reactivations: int,
                 fresh_low: int, fresh_high: int) -> dict:
    """`fresh_low:fresh_high` supplies never-built targets; `reactivations`
    re-drives gids [0, reactivations) which are dormant by construction time.
    Keep the ranges disjoint so canonical event keys stay unique."""
    walls = []
    work_sum = 0
    for token in range(tokens):
        targets = fresh_targets(fresh, token, fresh_low, fresh_high)
        if reactivations:
            # Re-drive the oldest built gids: they are dormant by now and the
            # ranges are disjoint, so canonical keys stay unique.
            targets = targets[: max(fresh - reactivations, 0)] + list(
                range(reactivations)
            )
        walls.append(runtime.advance(100, targets) * 1000.0)
        # cell_step_evaluations is per advance() call -- accumulate.
        work_sum += runtime.total_work()
    median = statistics.median(walls)
    return {
        "fresh_events_per_token": fresh,
        "reactivations_per_token": reactivations,
        "median_token_ms": round(median, 3),
        "p90_token_ms": round(sorted(walls)[int(0.9 * len(walls))], 3),
        "tokens_per_second_end_to_end": round(1000.0 / median, 1),
        "cell_steps_per_token_median_call": int(work_sum / tokens),
        "implied_cell_steps_per_second": int((work_sum / tokens) / (median / 1000.0)),
    }


def locality_stream(runtime: Runtime, *, tokens: int, pool: int,
                    fresh_per_token: int) -> dict:
    """Text-like traffic: a bounded active-vocabulary pool that keeps being
    re-hit (short replay gaps) plus a trickle of brand-new cells."""
    walls = []
    work_sum = 0
    for token in range(tokens):
        # Rotating window over the built dormant region: recently used words
        # keep getting re-hit, so replay gaps stay short.
        start = (token * 7) % 4096
        targets = [(start + i) % 4096 for i in range(pool)]
        targets.extend(fresh_targets(fresh_per_token, token, 1 << 18, 1 << 20))
        walls.append(runtime.advance(100, targets) * 1000.0)
        work_sum += runtime.total_work()
    median = statistics.median(walls)
    return {
        "pool_rehits_per_token": pool,
        "fresh_events_per_token": fresh_per_token,
        "median_token_ms": round(median, 3),
        "tokens_per_second_end_to_end": round(1000.0 / median, 1),
        "cell_steps_per_token_median_call": int(work_sum / tokens),
    }


def main() -> None:
    results: dict = {"production_flush": production_flush_horizon()}
    configured_cells = 1 << 20

    # Scenario A: ~256k dormant residents at 1M configured, sparse tokens.
    ep = Runtime(
        stepping_mode="event_proportional",
        configured_cells=configured_cells,
        resident_capacity=configured_cells,
        event_capacity=1 << 16,
        weight_pa=0.25,
    )
    try:
        build_wall = build_residency(ep, total=1 << 18, per_window=2048)
        ep.settle(400)
        results["scenario_a"] = {
            "configured_cells": configured_cells,
            "dormant_residency": 1 << 18,
            "build_wall_s": round(build_wall, 3),
            "state_after_settle": ep.live_state(),
        }
        scenario_a = []
        for fresh in (256, 512, 1024, 2048):
            entry = token_stream(ep, tokens=30, fresh=fresh, reactivations=0,
                                 fresh_low=1 << 18, fresh_high=1 << 20)
            scenario_a.append(entry)
        # Reactivation traffic over the dormant pool.
        scenario_a.append(token_stream(ep, tokens=30, fresh=512, reactivations=256,
                                       fresh_low=1 << 18, fresh_high=1 << 20))
        # Text-like locality: a rotating active-vocabulary pool plus a
        # trickle of new cells.
        scenario_a.append(locality_stream(ep, tokens=60, pool=512,
                                          fresh_per_token=128))
        results["scenario_a"]["tokens"] = scenario_a
        results["scenario_a"]["final_state"] = ep.live_state()
    finally:
        ep.close()

    # Scenario B: head-to-head full-touch vs event-proportional at a
    # residency the full-touch path can still build (~16k), identical
    # schedules, then the extrapolation to 1e6 residents against the
    # committed 9.18 M cell-steps/s anchor.
    head = {}
    for mode in ("full_touch", "event_proportional"):
        rt = Runtime(
            stepping_mode=mode,
            configured_cells=configured_cells,
            resident_capacity=65_536,
            event_capacity=1 << 16,
            weight_pa=0.25,
        )
        try:
            build_wall = build_residency(rt, total=16_384, per_window=2048)
            rt.settle(400)
            stream = token_stream(rt, tokens=15, fresh=512, reactivations=128,
                                  fresh_low=16_384, fresh_high=configured_cells)
            stream["build_wall_s"] = round(build_wall, 3)
            stream["final_state"] = rt.live_state()
            head[mode] = stream
        finally:
            rt.close()
    ft_rate = head["full_touch"]["tokens_per_second_end_to_end"]
    ep_rate = head["event_proportional"]["tokens_per_second_end_to_end"]
    head["speedup_ratio"] = round(ep_rate / ft_rate, 1) if ft_rate else None
    # Full-touch cost scales linearly with integrating residents; extrapolate
    # from this run's own final resident count to one million.
    head["full_touch_extrapolated_at_1m_residents_tok_s"] = round(
        ft_rate
        * head["full_touch"]["final_state"]["active_residents"]
        / configured_cells,
        3,
    )
    head["committed_anchor_1m_full_touch_tok_s"] = 0.092
    results["scenario_b_head_to_head"] = head

    out_path = Path(__file__).with_name("l3_event_proportional_throughput.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
