"""Throughput benchmark for the native sparse procedural runtime.

This drives the public ``NativeProceduralRuntime.advance`` seam only. It
measures wall-clock steps/second, resident cell-steps/second, and delivered +
generated events/second at bounded resident working-set sizes, plus a
density-response sweep that mirrors the v2 frontier activity-density protocol
(updated fraction per step). Every scenario is fully deterministic: the event
schedule comes from a seeded NumPy generator, so the run's canonical state
digest is a bit-exactness oracle across code changes.

Workload shape (one biological step = one fitted 0.05 ms GLIF5 step):

- configured address space is fixed at 1e9 cells with fanout 8, while the
  resident working set ramps to the scenario target; nothing in the runtime
  ever scales with the 1e9 configured count;
- each step injects a mix of above-threshold (150,000 pA, guaranteed spike)
  and subthreshold (325 pA) external events at uniformly random targets over
  the full address space;
- spiking sources regenerate their procedural fanout rows natively, so the
  queue carries both injected and spike-generated traffic;
- recurrent tau is shortened (0.5 ms) so subthreshold dust turns over inside a
  scenario window; spiked cells keep their fitted after-spike tails and remain
  resident, which the schedule budget accounts for explicitly.

The scenario ends before the touched-cell population can exceed the resident
capacity, so no fail-stop path is exercised here (the pytest suite owns those).

Usage from the repository or worktree root:

    python benchmarks/scale_opt_cpu_throughput.py --quick
    python benchmarks/scale_opt_cpu_throughput.py --json-out results.json
    python benchmarks/scale_opt_cpu_throughput.py --check-golden \
        benchmarks/scale_opt_cpu_golden_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humanbrain.procedural_runtime import (  # noqa: E402
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
)
from humanbrain.procedural_topology import TopologyManifest  # noqa: E402

CONFIGURED_CELLS = 1_000_000_000
FANOUT = 8
SEED = 71
SPIKE_CURRENT_PA = 150_000.0
SUBTHERSHOLD_CURRENT_PA = 325.0
SPIKING_SHARE = 0.5
STEPS_PER_CALL = 20
SOURCE_POOL = 64
RESERVED_SOURCE_BASE = 900_000_000

RESIDENT_SIZES = (1_000, 10_000, 100_000, 1_000_000)
DENSITY_FRACTIONS = (0.01, 0.05, 0.20, 1.00)
DEFAULT_WINDOW_STEPS = {1_000: 20_000, 10_000: 10_000, 100_000: 4_000, 1_000_000: 2_000}


def scenario_window_steps(resident_size: int) -> int:
    return DEFAULT_WINDOW_STEPS.get(resident_size, max(1_000, 2_000_000 // resident_size))


def _next_pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def make_config(resident_size: int, density: float, window_steps: int) -> ProceduralRuntimeConfig:
    """Size capacities for one scenario so no bound trips before the ramp ends."""
    touch_rate_per_step = (
        density * resident_size / (window_steps * (1.0 + FANOUT * SPIKING_SHARE))
    )
    spiking_per_step = touch_rate_per_step * SPIKING_SHARE
    generated_per_step = spiking_per_step * FANOUT
    queue_peak_bound = int(math.ceil((touch_rate_per_step + generated_per_step) * 8)) + 1024
    spikes_per_call = int(math.ceil(spiking_per_step * STEPS_PER_CALL)) + 16
    # Contract floors first (interval capacity >= resident capacity), then
    # scenario headroom on top; documented in scale_opt_cpu_report.md.
    distinct_bound = max(
        resident_size,
        int(math.ceil(density * resident_size * 1.25)) + 4096,
    )
    return ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=CONFIGURED_CELLS,
            fanout=FANOUT,
            topology="uniform_no_self",
            seed=SEED,
        ),
        resident_capacity=resident_size,
        event_capacity=max(4_096, _next_pow2(queue_peak_bound)),
        spike_capacity=max(1_024, _next_pow2(spikes_per_call)),
        interval_distinct_capacity=distinct_bound,
        recurrent_weight_pa=SUBTHERSHOLD_CURRENT_PA,
        recurrent_delay_steps=1,
        recurrent_tau_ms=0.5,
        eligibility_tau_ms=0.05,
    )


def iter_scenario_events(
    resident_size: int,
    density: float,
    window_steps: int,
):
    """Yield one deterministic event batch per advance call.

    Targets are uniform over the whole 1e9 address space; provenance sources
    come from a tiny fixed pool so injected keys cannot collide with
    spike-generated keys in practice (and any collision would fail loudly).
    """
    total_touch_rate = density * resident_size / (window_steps * (1.0 + FANOUT * SPIKING_SHARE))
    spiking_per_step_f = total_touch_rate * SPIKING_SHARE
    sub_rate_f = total_touch_rate - spiking_per_step_f

    rng = np.random.Generator(np.random.PCG64(SEED * 1_000_003 + resident_size))
    current_step = 0
    used_keys: set[tuple[int, int, int, int]] = set()
    for step_index in range(window_steps):
        spiking_count = int(spiking_per_step_f * (step_index + 1)) - int(
            spiking_per_step_f * step_index
        )
        subthreshold_count = int(sub_rate_f * (step_index + 1)) - int(
            sub_rate_f * step_index
        )
        events: list[ProceduralInputEvent] = []
        for _ in range(spiking_count + subthreshold_count):
            target = int(rng.integers(0, CONFIGURED_CELLS))
            source = RESERVED_SOURCE_BASE + int(rng.integers(0, SOURCE_POOL))
            slot = int(rng.integers(0, FANOUT))
            key = (current_step, target, source, slot)
            while key in used_keys:
                target = (target + 1) % CONFIGURED_CELLS
                key = (current_step, target, source, slot)
            used_keys.add(key)
            events.append(
                ProceduralInputEvent(
                    due_step=current_step,
                    target_gid=target,
                    source_gid=source,
                    edge_slot=slot,
                    current_pa=(
                        SPIKE_CURRENT_PA
                        if len(events) < spiking_count
                        else SUBTHERSHOLD_CURRENT_PA
                    ),
                )
            )
        yield current_step, events
        current_step += STEPS_PER_CALL


def canonical_state_digest(runtime: NativeProceduralRuntime) -> str:
    """SHA-256 over exact simulation-state bits: residents, pending, spikes.

    Deliberately excludes ``native_payload_bytes``: that accounting value may
    legitimately change when the runtime's internal structure is reorganized,
    whereas bit-exactness is about the simulated state. Payload bytes are
    reported separately per scenario and must stay O(resident capacity).
    """
    snapshot = runtime.snapshot()
    hasher = hashlib.sha256()
    for record in snapshot.residents:
        hasher.update(
            struct.pack(
                "<Q5di2d",
                record.gid,
                record.voltage_relative_mv,
                record.after_spike_currents_pa[0],
                record.after_spike_currents_pa[1],
                record.threshold_spike_mv,
                record.threshold_voltage_mv,
                record.refractory_steps,
                record.synaptic_current_pa,
                record.eligibility,
            )
        )
    for record in snapshot.pending_events:
        hasher.update(
            struct.pack(
                "<QQQQd",
                record.due_step,
                record.target_gid,
                record.source_gid,
                record.edge_slot,
                record.current_pa,
            )
        )
    for record in snapshot.spikes:
        hasher.update(struct.pack("<QQ", record.step, record.gid))
    hasher.update(str(snapshot.current_step).encode())
    return hasher.hexdigest()


def run_scenario(
    resident_size: int,
    density: float,
    window_steps: int | None = None,
) -> dict:
    window_steps = window_steps or scenario_window_steps(resident_size)
    config = make_config(resident_size, density, window_steps)

    wall_seconds = 0.0
    totals = {
        "cell_step_evaluations": 0,
        "spike_events": 0,
        "generated_edges": 0,
        "delivered_events": 0,
    }
    peaks = {
        "resident_peak": 0,
        "queue_peak": 0,
        "updated_distinct": 0,
        "spiking_distinct": 0,
        "resident_distinct": 0,
    }
    calls = 0
    digest = ""
    with NativeProceduralRuntime(config) as runtime:
        started = time.perf_counter()
        for current_step, events in iter_scenario_events(
            resident_size, density, window_steps
        ):
            result = runtime.advance(STEPS_PER_CALL, tuple(events))
            assert result.spikes is not None
            accounting = result.accounting
            totals["cell_step_evaluations"] += int(accounting.work.cell_step_evaluations)
            totals["spike_events"] += int(accounting.work.spike_events)
            totals["generated_edges"] += int(accounting.edges.generated)
            totals["delivered_events"] += int(accounting.events.delivered)
            cells = accounting.cells
            peaks["resident_peak"] = max(peaks["resident_peak"], int(cells.resident_peak))
            peaks["queue_peak"] = max(peaks["queue_peak"], int(accounting.events.queue_peak))
            peaks["updated_distinct"] = max(peaks["updated_distinct"], int(cells.updated_distinct))
            peaks["spiking_distinct"] = max(peaks["spiking_distinct"], int(cells.spiking_distinct))
            peaks["resident_distinct"] = max(peaks["resident_distinct"], int(cells.resident_distinct))
            calls += 1
        wall_seconds = time.perf_counter() - started
        digest = canonical_state_digest(runtime)
        payload_bytes = int(runtime.snapshot().native_payload_bytes)
        final_residents = len(runtime.snapshot().residents)

    steps = calls * STEPS_PER_CALL
    return {
        "scenario": {
            "resident_capacity": resident_size,
            "density_target": density,
            "window_steps": window_steps,
            "steps_per_call": STEPS_PER_CALL,
            "configured_cells": CONFIGURED_CELLS,
            "fanout": FANOUT,
        },
        "wall_seconds": round(wall_seconds, 6),
        "steps": steps,
        "steps_per_second": round(steps / wall_seconds, 3),
        "v2_equivalent_tokens_per_second": round(steps / wall_seconds / 100.0, 4),
        "cell_steps": totals["cell_step_evaluations"],
        "cell_steps_per_second": round(totals["cell_step_evaluations"] / wall_seconds, 1),
        "delivered_events": totals["delivered_events"],
        "generated_edges": totals["generated_edges"],
        "events_per_second": round(
            (totals["delivered_events"] + totals["generated_edges"]) / wall_seconds, 1
        ),
        "spike_events": totals["spike_events"],
        "peaks": peaks,
        "final_residents": final_residents,
        "native_payload_bytes": payload_bytes,
        "payload_bytes_per_resident_cell": round(
            payload_bytes / max(1, peaks["resident_peak"]), 3
        ),
        "achieved_updated_fraction_of_capacity": round(
            totals["cell_step_evaluations"]
            / max(1, resident_size * steps),
            6,
        ),
        "state_digest_sha256": digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=list(RESIDENT_SIZES),
        help="Resident capacity sizes to benchmark",
    )
    parser.add_argument(
        "--densities",
        type=float,
        nargs="*",
        default=list(DENSITY_FRACTIONS),
        help="Updated-fraction-of-capacity targets per scenario",
    )
    parser.add_argument("--quick", action="store_true", help="Smoke run: smallest size only")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--golden-out",
        type=Path,
        default=None,
        help="Write scenario->state-digest mapping for bit-exactness gating",
    )
    parser.add_argument(
        "--check-golden",
        type=Path,
        default=None,
        help="Compare state digests against a golden JSON file (bit-exactness gate)",
    )
    args = parser.parse_args(argv)

    sizes = args.sizes
    densities = args.densities
    if args.quick:
        sizes = [1_000]
        densities = [0.05]

    results = []
    for resident_size in sizes:
        for density in densities:
            print(
                f"[scale-opt-cpu] resident={resident_size:>9,d} "
                f"density={density:>5.0%} ...",
                flush=True,
            )
            outcome = run_scenario(resident_size, density)
            results.append(outcome)
            print(
                f"    steps/s={outcome['steps_per_second']:>12,.1f}  "
                f"cell-steps/s={outcome['cell_steps_per_second']:>16,.1f}  "
                f"events/s={outcome['events_per_second']:>14,.1f}  "
                f"peak_resident={outcome['peaks']['resident_peak']:>9,d}  "
                f"payload/cell={outcome['payload_bytes_per_resident_cell']:>9,.1f}B  "
                f"digest={outcome['state_digest_sha256'][:12]}",
                flush=True,
            )

    payload = {
        "benchmark": "scale-opt-cpu-throughput-v1",
        "machine_note": "single process, public advance() seam only",
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    if args.golden_out is not None:
        golden = {
            "benchmark": "scale-opt-cpu-golden-v1",
            "note": (
                "Bit-exactness oracle: canonical snapshot digests captured on the "
                "baseline implementation. Optimizations must reproduce every digest "
                "exactly (struct-packed resident/event/spike bits + counters)."
            ),
            "scenarios": [
                {
                    "resident_capacity": entry["scenario"]["resident_capacity"],
                    "density_target": entry["scenario"]["density_target"],
                    "window_steps": entry["scenario"]["window_steps"],
                    "state_digest_sha256": entry["state_digest_sha256"],
                    "final_residents": entry["final_residents"],
                    "cell_steps": entry["cell_steps"],
                    "delivered_events": entry["delivered_events"],
                    "generated_edges": entry["generated_edges"],
                    "spike_events": entry["spike_events"],
                }
                for entry in results
            ],
        }
        args.golden_out.write_text(json.dumps(golden, indent=2), encoding="utf-8")
        print(f"wrote {args.golden_out}")

    if args.check_golden is not None:
        golden = json.loads(Path(args.check_golden).read_text(encoding="utf-8"))
        entries = golden.get("scenarios") or golden["results"]
        expected = {
            (entry["resident_capacity"], entry["density_target"]): entry[
                "state_digest_sha256"
            ]
            for entry in entries
        }
        failures = []
        checked = 0
        for entry in results:
            key = (
                entry["scenario"]["resident_capacity"],
                entry["scenario"]["density_target"],
            )
            if key in expected:
                checked += 1
                if expected[key] != entry["state_digest_sha256"]:
                    failures.append(key)
        if failures:
            print(f"GOLDEN MISMATCH for {failures}", file=sys.stderr)
            return 1
        print(f"golden check OK: {checked} scenario digests match exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
