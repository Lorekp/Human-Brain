"""L3 spike benchmark: event-proportional stepping throughput at 1e6 scale.

Measures procedural-runtime token throughput when residency has ratcheted to
a large dormant population while only a sparse set of cells integrates per
token -- the steady-state training regime that motivates event-proportional
work semantics.

Measurement matrix (replaces the earlier extrapolated estimates): both
stepping modes drive IDENTICAL schedules at 1,000,000 configured cells with
per-token activity of 1%, 2%, and 5% of configured cells (10k / 20k / 50k
events per token). Each mode first ratchets the same residency fixture
(2^18 cells built in 2,048-event windows, then settled), then re-drives a
bounded vocabulary pool (2x the per-token event count, alternating halves,
so replay gaps stay at one token window -- text-like short-gap locality).

Everything is driven through the raw C-ABI with a documented stress GLIF
configuration: unit leak makes voltage mirror synaptic current exactly, so a
spiked cell's fast state reaches canonical zero in tens of steps and its
eligibility tail can retire to the dormant store inside this script's runtime
budget. The mechanism, fail-stop contracts, and bit-parity obligations are
identical under production constants; only the tail-flush horizon differs
(see FINDINGS.md for the measured production flush horizon and why it
dominates near-term training runs).

Capacity planning against the fail-stop contracts: overflow poisons the
runtime mid-measurement (nonzero advance status), so every capacity is sized
for the LARGEST single advance() call, not steady state:

- ``event_capacity`` = QUEUE_HEADROOM x max batch. The injected batch sits in
  the queue until its due step, then each spike generates ``fanout`` edges
  that coexist in the queue: peak <= batch + fanout*batch = 5x batch.
- ``spike_capacity`` = SPIKE_HEADROOM x max batch. Spikes per call are
  bounded by the injected batch size (subthreshold edge weights cannot
  cascade); headroom covers transient multi-spike behaviour (status-9 guard).
- ``interval_distinct_capacity`` >= resident_capacity (native contract
  minimum) and above the worst-case union of entry residents + staged
  insertions + spiked gids marked in one call (status-10 guard).
- The residency build advances in small fixed windows (``build_per_window``)
  so no single call ever exceeds the per-call sizing above.

Two auxiliary sections price components the matrix conflates:

- ``injection_path_diagnostic_5pct`` re-runs the 5% activity cell at shipped
  fanout=4 vs minimal fanout=1. Generated-edge volume drops 4x; wall-time
  movement between rows prices the shared injection path independently of
  stepping mode.
- ``low_activity_reference`` repeats the original spike head-to-head fixture
  (16,384 residents, 640 events/token) where injection volume is small
  enough that the mode difference IS the stepping cost being priced.

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

# --- capacity-planning headroom constants (documented, not tuned) ----------
SPIKE_HEADROOM = 4    # status-9 guard: max spikes/call <= injected batch
QUEUE_HEADROOM = 8    # status-3 guard: queue peak <= batch + fanout*batch = 5x
BUILD_PER_WINDOW = 2048  # build phase chunk; keeps every call within sizing


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

    def horizon(factor: float) -> float:
        # Log-domain guard: a decay factor that computes to 0.0 (or rounds
        # to >= 1.0) never reaches canonical rest. Return an infinite
        # sentinel instead of feeding it to math.log (ValueError).
        if factor <= 0.0 or factor >= 1.0:
            return math.inf
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
    """Raw C-ABI runtime with internal step bookkeeping.

    Capacities scale off ``max_batch`` -- the largest single advance() call
    this runtime will ever see -- with documented headroom, because the
    fail-stop contract poisons the runtime on overflow instead of degrading.
    """

    def __init__(
        self,
        *,
        stepping_mode: str,
        configured_cells: int,
        resident_capacity: int,
        max_batch: int,
        weight_pa: float,
        fanout: int = 4,
    ) -> None:
        mode_id = {"full_touch": 0, "event_proportional": 1}[stepping_mode]
        self.stepping_mode = stepping_mode
        self.max_batch = max_batch
        self.fanout = fanout
        event_capacity = QUEUE_HEADROOM * max_batch
        spike_capacity = SPIKE_HEADROOM * max_batch
        interval_distinct_capacity = max(resident_capacity, event_capacity)
        config = NativeConfig(
            configured_cells,
            fanout,
            71,
            resident_capacity,
            event_capacity,
            spike_capacity,
            interval_distinct_capacity,
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
        # Buffer holds one injected batch; event_capacity/4 slots = 2x batch.
        self.buffer = (NativeEvent * (event_capacity // 4))()
        self.counters = NativeCounters()
        self.step = 0
        self.capacity_plan = {
            "resident_capacity": resident_capacity,
            "event_capacity": event_capacity,
            "spike_capacity": spike_capacity,
            "interval_distinct_capacity": interval_distinct_capacity,
            "max_batch": max_batch,
            "spike_headroom_multiplier": SPIKE_HEADROOM,
            "queue_headroom_multiplier": QUEUE_HEADROOM,
        }

    def close(self) -> None:
        if self.handle:
            LIB.humanbrain_procedural_runtime_destroy(self.handle)
            self.handle = None

    def advance(self, steps: int, targets) -> float:
        count = len(targets)
        if count > len(self.buffer):
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


def build_residency(runtime: Runtime, total: int, per_window: int) -> float:
    """Ratchet residency by driving fresh gids in small windows.

    Windows stay at ``per_window`` events so every advance() call fits the
    per-call capacity sizing regardless of total residency built.
    """
    wall = 0.0
    built = 0
    while built < total:
        chunk = min(per_window, total - built)
        wall += runtime.advance(100, range(built, built + chunk))
        built += chunk
    return wall


def activity_stream(runtime: Runtime, *, tokens: int, events_per_token: int,
                    pool: int) -> dict:
    """Steady-state activity at a fixed fraction of configured cells.

    Each token drives ``events_per_token`` distinct cells drawn from a
    bounded vocabulary pool of ``pool`` gids, alternating halves so a cell is
    re-hit every second token (replay gap <= two token windows -- text-like
    short-gap locality). Contiguous mod-pool slices keep targets
    distinct inside a batch; across tokens the later due step keeps
    canonical event keys unique.
    """
    walls = []
    work_sum = 0
    for token in range(tokens):
        offset = ((token % 2) * events_per_token) % pool
        targets = [(offset + i) % pool for i in range(events_per_token)]
        walls.append(runtime.advance(100, targets) * 1000.0)
        # cell_step_evaluations is per advance() call -- accumulate.
        work_sum += runtime.total_work()
    median = statistics.median(walls)
    return {
        "activity_fraction_of_configured": round(events_per_token / 1_000_000, 4),
        "events_per_token": events_per_token,
        "vocabulary_pool": pool,
        "tokens_measured": tokens,
        "median_token_ms": round(median, 3),
        "p90_token_ms": round(sorted(walls)[int(0.9 * len(walls))], 3),
        "tokens_per_second_end_to_end": round(1000.0 / median, 1),
        "cell_steps_per_token_median_call": int(work_sum / tokens),
        "implied_cell_steps_per_second": int((work_sum / tokens) / (median / 1000.0)),
    }


ACTIVITY_CELLS = ((1, 10_000), (2, 20_000), (5, 50_000))
MAX_EVENTS_PER_TOKEN = max(count for _, count in ACTIVITY_CELLS)
TOKENS_PER_CELL = 12


def fresh_targets(count: int, token_index: int, low: int, high: int) -> list[int]:
    """Deterministic xorshift pseudo-random distinct targets inside
    ``[low, high)`` (used by the low-activity reference fixture)."""
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


def run_mode(stepping_mode: str, configured_cells: int, *, fanout: int = 4,
             activities: tuple = ACTIVITY_CELLS,
             tokens_per_cell: int = TOKENS_PER_CELL) -> dict:
    max_batch = max(count for _, count in activities)
    rt = Runtime(
        stepping_mode=stepping_mode,
        configured_cells=configured_cells,
        resident_capacity=configured_cells,
        max_batch=max_batch,
        weight_pa=0.25,
        fanout=fanout,
    )
    try:
        result: dict = {"capacity_plan": rt.capacity_plan}
        result["fixture"] = {"fanout": fanout}
        build_wall = build_residency(rt, total=1 << 18, per_window=BUILD_PER_WINDOW)
        rt.settle(400)
        result["build"] = {
            "total_residents_built": 1 << 18,
            "per_window": BUILD_PER_WINDOW,
            "settle_steps": 400,
            "build_wall_s": round(build_wall, 3),
            "state_after_settle": rt.live_state(),
        }
        streams = {}
        for pct, count in activities:
            streams[f"{pct}pct"] = activity_stream(
                rt,
                tokens=tokens_per_cell,
                events_per_token=count,
                pool=2 * count,
            )
        streams["final_state"] = rt.live_state()
        result["activity_streams"] = streams
        return result
    finally:
        rt.close()


def low_activity_reference(configured_cells: int) -> dict:
    """Head-to-head at the original spike fixture (build 16,384 residents;
    512 fresh + 128 reactivation events per token): event volume stays small
    enough that the shared injection path is subdominant, so the difference
    IS the stepping-mode cost being priced."""
    head = {}
    for mode in ("full_touch", "event_proportional"):
        rt = Runtime(
            stepping_mode=mode,
            configured_cells=configured_cells,
            resident_capacity=65_536,
            # Largest single call = the 2,048-wide residency-build window,
            # not the 640-event token batch.
            max_batch=max(640, BUILD_PER_WINDOW),
            weight_pa=0.25,
        )
        try:
            build_wall = build_residency(rt, total=16_384, per_window=BUILD_PER_WINDOW)
            rt.settle(400)
            walls = []
            work_sum = 0
            for token in range(15):
                targets = fresh_targets(512, token, 16_384, configured_cells)
                targets += list(range(128))  # re-drive dormant built gids
                walls.append(rt.advance(100, targets) * 1000.0)
                work_sum += rt.total_work()
            median = statistics.median(walls)
            head[mode] = {
                "events_per_token": 640,
                "median_token_ms": round(median, 3),
                "tokens_per_second_end_to_end": round(1000.0 / median, 1),
                "cell_steps_per_token_median_call": int(work_sum / 15),
                "build_wall_s": round(build_wall, 3),
                "final_state": rt.live_state(),
            }
        finally:
            rt.close()
    ft_rate = head["full_touch"]["tokens_per_second_end_to_end"]
    ep_rate = head["event_proportional"]["tokens_per_second_end_to_end"]
    head["speedup_event_proportional_vs_full_touch"] = (
        round(ep_rate / ft_rate, 1) if ft_rate else None
    )
    return head


def main() -> None:
    results: dict = {
        "production_flush": production_flush_horizon(),
        "fixture": {
            "configured_cells": 1_000_000,
            "residency_built": 1 << 18,
            "build_per_window": BUILD_PER_WINDOW,
            "settle_steps": 400,
            "tokens_per_cell": TOKENS_PER_CELL,
            "token_window_steps": 100,
            "pool_rule": (
                "2x events_per_token, alternating halves; each cell re-hit "
                "every second token (replay gap <= two token windows)"
            ),
        },
    }
    configured_cells = results["fixture"]["configured_cells"]

    ep = run_mode("event_proportional", configured_cells)
    ft = run_mode("full_touch", configured_cells)

    speedups = {}
    for pct, _ in ACTIVITY_CELLS:
        key = f"{pct}pct"
        ft_rate = ft["activity_streams"][key]["tokens_per_second_end_to_end"]
        ep_rate = ep["activity_streams"][key]["tokens_per_second_end_to_end"]
        speedups[key] = round(ep_rate / ft_rate, 1) if ft_rate else None
    results["matrix"] = {
        "event_proportional": ep,
        "full_touch": ft,
        "speedup_event_proportional_vs_full_touch": speedups,
    }

    # Diagnostic ablation at 5% activity: shipped fanout=4 vs minimal
    # fanout=1. Generated-edge volume drops 4x, shrinking the staged batch
    # that dominates the shared injection path -- wall-time movement between
    # the two rows prices that path independently of stepping mode.
    diag = {}
    for fanout in (4, 1):
        key = f"fanout{fanout}"
        diag[key] = {
            "event_proportional": run_mode(
                "event_proportional", configured_cells, fanout=fanout,
                activities=((5, 50_000),), tokens_per_cell=4,
            )["activity_streams"]["5pct"],
            "full_touch": run_mode(
                "full_touch", configured_cells, fanout=fanout,
                activities=((5, 50_000),), tokens_per_cell=4,
            )["activity_streams"]["5pct"],
        }
    results["injection_path_diagnostic_5pct"] = diag

    results["low_activity_reference"] = low_activity_reference(configured_cells)

    out_path = Path(__file__).with_name("l3_event_proportional_throughput.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

