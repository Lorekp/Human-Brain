"""L4 seam acceptance: targeted read_cells FFI vs the snapshot() wall.

Seam A acceptance microbenchmark (docs/research/cpu-seams-spec.md A.7).
The seam under test is ``NativeProceduralRuntime.read_cells``: an O(K log R)
targeted resident read that must make a K=16 eligibility probe affordable at
R = 1e6 residents, where the only prior read seam (``snapshot()``) costs
O(R) native copy + O(R) Python dataclass conversion -- measured 161 ms/token
at R=100k (L2), i.e. ~26x the entire 160 tok/s floor budget.

Fixture (prescribed, do not improvise): N_cfg = 2^20,
resident_capacity = interval_distinct_capacity = 2^20 (the address space
doubles as the headroom above R; host validation requires
resident_capacity <= cell_count), and R =
1,000,000 residents built with the L2 bulk recipe (one subthreshold event
per gid, one ``advance(TOKEN_STEPS, ...)`` per chunk) in k = 8 chunks of
125k across separate advance calls. Known cost warning: the public-API
fixture build pays quadratic-class terms (linear `.position()` staged scans
during delivery; sorted-insert ledger shifts) and takes **minutes**; the
timed section starts only after the build. Nothing is cached across
processes.

CPU-only. Gates recorded in l4_seams_acceptance.json:
  A-G1  median read_cells K=16 @ R=1e6            < 50 us
  A-G2  median read_cells K=16 @ R~100k           < 20 us
  A-G3  speedup vs snapshot()-extract @ R=100k    >= 1000x
  A-G4  two identically-seeded runtimes produce byte-identical raw buffers
  A-G5  procedural test suites green (run in-process via subprocess pytest)
"""

from __future__ import annotations

import ctypes
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Self-contained bootstrap: run from any cwd, the worktree's src/ is put on
# sys.path before the humanbrain imports below.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np

from humanbrain.procedural_runtime import (
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
    _NativeResident,
)
from humanbrain.procedural_topology import TopologyManifest

# Spec A.7 pins cfg = 2^20 with R = 1e6 residents filled into it.
# ProceduralRuntimeConfig requires resident_capacity <= cell_count and
# interval_distinct_capacity >= resident_capacity; the 2^20 address space
# itself provides the headroom above R = 1e6. Event/spike capacities carry
# the L3-style headroom over the largest fill batch (125k): 4x queue,
# generous spike slack (the fill drive is subthreshold and asserts no
# spikes).
CONFIGURED_CELLS = 1_048_576             # 2^20
RESIDENT_CAPACITY = 1_048_576            # = cell_count (validation-consistent)
INTERVAL_DISTINCT_CAPACITY = 1_048_576
EVENT_CAPACITY = 524_288                 # >= 4x the 125k fill chunk
SPIKE_CAPACITY = 65_536
TARGET_RESIDENTS = 1_000_000
SMALL_RESIDENTS = 100_000
CHUNKS = 8                            # spec-pinned mitigation factor
FILL_CURRENT_PA = 100.0               # subthreshold drive: perturbs, no spikes
TOKEN_STEPS = 100
REQUEST_K = 16
WARMUP_CALLS = 3
TIMED_CALLS = 151
POOL_SEED = 20260824


def make_runtime(fill_count: int) -> NativeProceduralRuntime:
    """L2/L3-style bounded runtime sized for the prescribed fixture."""
    return NativeProceduralRuntime(
        ProceduralRuntimeConfig(
            manifest=TopologyManifest(
                cell_count=CONFIGURED_CELLS,
                fanout=8,
                topology="uniform_no_self",
                seed=71,
            ),
            resident_capacity=RESIDENT_CAPACITY,
            event_capacity=EVENT_CAPACITY,
            spike_capacity=SPIKE_CAPACITY,
            interval_distinct_capacity=INTERVAL_DISTINCT_CAPACITY,
            recurrent_weight_pa=325.0,
            recurrent_delay_steps=1,
        )
    )


def fill_residents(
    runtime: NativeProceduralRuntime, count: int, chunk_size: int, label: str
) -> float:
    """Make exactly `count` cells resident, `chunk_size` per advance call."""
    started = time.perf_counter()
    done = 0
    while done < count:
        batch = min(chunk_size, count - done)
        step = runtime._current_step
        events = [
            ProceduralInputEvent(step, gid, gid, 0, FILL_CURRENT_PA)
            for gid in range(done, done + batch)
        ]
        result = runtime.advance(TOKEN_STEPS, events)
        assert result.spikes == (), (
            f"{label}: fill current produced {len(result.spikes)} spikes; "
            "lower FILL_CURRENT_PA"
        )
        done += batch
        elapsed = time.perf_counter() - started
        print(
            f"  [{label}] {done:,}/{count:,} residents "
            f"(elapsed {elapsed:6.1f}s)",
            flush=True,
        )
    return time.perf_counter() - started


def draw_requests(residents: int, count: int, pool: np.ndarray, cursor: int):
    """K distinct gids per repetition, consumed from one shuffled pool so no
    working-set page is reused across repetitions."""
    needed = count * REQUEST_K
    block = pool[cursor : cursor + needed].reshape(count, REQUEST_K)
    return [row.tolist() for row in block], cursor + needed


def measure_read_cells(runtime: NativeProceduralRuntime, requests):
    # Let the post-build thermal/scheduler state settle before the timed
    # section; the protocol pins only that timing starts after the build.
    time.sleep(3.0)
    for warmup in requests[:WARMUP_CALLS]:
        runtime.read_cells(warmup)
    samples_ns = []
    for request in requests[WARMUP_CALLS:]:
        start = time.perf_counter_ns()
        runtime.read_cells(request)
        samples_ns.append(time.perf_counter_ns() - start)
    ordered = sorted(samples_ns)
    median_ns = statistics.median(samples_ns)
    return {
        "calls": len(samples_ns),
        "median_us": round(median_ns / 1e3, 3),
        "p10_us": round(ordered[int(0.10 * len(ordered))] / 1e3, 3),
        "p90_us": round(ordered[int(0.90 * len(ordered))] / 1e3, 3),
        "ns_per_request": round(median_ns / REQUEST_K, 1),
        "min_us": round(ordered[0] / 1e3, 3),
        "max_us": round(ordered[-1] / 1e3, 3),
    }


def measure_snapshot_extract_once(
    runtime: NativeProceduralRuntime, request
) -> dict:
    """The incumbent seam: one full snapshot() + dict extraction of the same
    K cells. Measured once (it is the thing being replaced)."""
    start = time.perf_counter_ns()
    snapshot = runtime.snapshot()
    states = {state.gid: state for state in snapshot.residents}
    values = [states[gid].eligibility for gid in request]
    elapsed_ns = time.perf_counter_ns() - start
    assert len(values) == REQUEST_K
    return {
        "snapshot_extract_once_ms": round(elapsed_ns / 1e6, 2),
        "residents_in_snapshot": len(snapshot.residents),
    }


def raw_read_bytes(runtime: NativeProceduralRuntime, gids: list[int]):
    """Drive the C-ABI export directly; return raw out/found buffer bytes."""
    count = len(gids)
    gid_buffer = (ctypes.c_uint64 * count)(*gids)
    out = (_NativeResident * count)()
    found = (ctypes.c_uint8 * count)()
    status = int(
        runtime._library.humanbrain_procedural_runtime_read_cells(
            runtime._handle,
            gid_buffer,
            count,
            out,
            found,
        )
    )
    assert status == 0
    return (
        bytes(memoryview(out)).hex(),
        bytes(memoryview(found)),
    )


def _raise_process_priority() -> None:
    """Best-effort shielding from ambient desktop load (Windows only)."""
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetCurrentProcess()
        # HIGH_PRIORITY_CLASS
        kernel32.SetPriorityClass(handle, 0x00000080)
    except (AttributeError, OSError):
        pass


def main() -> int:
    _raise_process_priority()
    repo_root = Path(__file__).resolve().parents[2]
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    print(f"L4 seams acceptance @ commit {commit}")
    print(
        f"WARNING: fixture builds pay quadratic-class public-API terms and "
        f"take minutes; the timed section starts after the build."
    )

    results: dict[str, object] = {
        "spec": "docs/research/cpu-seams-spec.md Seam A (A.7)",
        "commit": commit,
        "configured_cells": CONFIGURED_CELLS,
        "request_k": REQUEST_K,
        "warmup_calls": WARMUP_CALLS,
        "timed_calls": TIMED_CALLS,
        "fill_current_pa": FILL_CURRENT_PA,
        "token_steps_per_fill_call": TOKEN_STEPS,
        "chunks": CHUNKS,
        "python": sys.version.split()[0],
    }

    # ---- fixture: R = 1e6 -------------------------------------------------
    print(f"\nbuilding R={TARGET_RESIDENTS:,} fixture ...", flush=True)
    big = make_runtime(TARGET_RESIDENTS)
    try:
        big_build_s = fill_residents(
            big, TARGET_RESIDENTS, TARGET_RESIDENTS // CHUNKS, "R1e6"
        )
        results["build_r1e6_seconds"] = round(big_build_s, 1)

        # Shuffled pool spanning the full [0, R) range; disjoint slices per
        # repetition keep steady-state numbers honest about cache misses.
        rng = np.random.default_rng(POOL_SEED)
        pool_big = rng.permutation(TARGET_RESIDENTS)
        requests_big, _ = draw_requests(
            TARGET_RESIDENTS, WARMUP_CALLS + TIMED_CALLS, pool_big, 0
        )

        big_stats = measure_read_cells(big, requests_big)
        results["read_cells_r1e6"] = big_stats
        print(
            f"\nread_cells K={REQUEST_K} @ R={TARGET_RESIDENTS:,}: "
            f"median {big_stats['median_us']}us  "
            f"p10 {big_stats['p10_us']}us  p90 {big_stats['p90_us']}us  "
            f"({big_stats['ns_per_request']} ns/request)"
        )

        # Context only: the incumbent snapshot()-based extraction, once.
        context_request = requests_big[WARMUP_CALLS]
        results["snapshot_extract_r1e6"] = measure_snapshot_extract_once(
            big, context_request
        )
        print(
            "snapshot()-based extraction (once): "
            f"{results['snapshot_extract_r1e6']['snapshot_extract_once_ms']}ms"
        )

        # Determinism gate at the large scale: identical seeds + schedules.
        twin = make_runtime(TARGET_RESIDENTS)
        try:
            fill_residents(twin, TARGET_RESIDENTS, TARGET_RESIDENTS // CHUNKS, "twin")
            pool_twin = np.random.default_rng(POOL_SEED).permutation(
                TARGET_RESIDENTS
            )
            twin_requests, _ = draw_requests(
                TARGET_RESIDENTS, 4, pool_twin, 0
            )
            deterministic = True
            for request in twin_requests:
                if raw_read_bytes(big, request) != raw_read_bytes(twin, request):
                    deterministic = False
                    break
            results["determinism_r1e6_byte_identical"] = deterministic
            print(f"A-G4 determinism (raw buffers): {'PASS' if deterministic else 'FAIL'}")
        finally:
            twin.close()
    finally:
        big.close()

    # ---- fixture: R = 100k -------------------------------------------------
    print(f"\nbuilding R={SMALL_RESIDENTS:,} fixtures ...", flush=True)
    small = make_runtime(SMALL_RESIDENTS)
    try:
        small_build_s = fill_residents(
            small, SMALL_RESIDENTS, SMALL_RESIDENTS // CHUNKS, "R100k"
        )
        results["build_r100k_seconds"] = round(small_build_s, 1)

        pool_small = np.random.default_rng(POOL_SEED + 1).permutation(
            SMALL_RESIDENTS
        )
        requests_small, _ = draw_requests(
            SMALL_RESIDENTS, WARMUP_CALLS + TIMED_CALLS, pool_small, 0
        )
        small_stats = measure_read_cells(small, requests_small)
        results["read_cells_r100k"] = small_stats
        print(
            f"read_cells K={REQUEST_K} @ R={SMALL_RESIDENTS:,}: "
            f"median {small_stats['median_us']}us  "
            f"p10 {small_stats['p10_us']}us  p90 {small_stats['p90_us']}us  "
            f"({small_stats['ns_per_request']} ns/request)"
        )

        snapshot_context = measure_snapshot_extract_once(
            small, requests_small[WARMUP_CALLS]
        )
        results["snapshot_extract_r100k"] = snapshot_context
        snapshot_ms = snapshot_context["snapshot_extract_once_ms"]
        speedup = snapshot_ms * 1e3 / small_stats["median_us"]
        results["speedup_vs_snapshot_extract_r100k_x"] = round(speedup, 1)
        print(
            f"snapshot()-based extraction (once): {snapshot_ms}ms -> "
            f"speedup {speedup:,.0f}x"
        )
    finally:
        small.close()

    # ---- gates -------------------------------------------------------------
    gate_a_g1 = big_stats["median_us"] < 50.0
    gate_a_g2 = small_stats["median_us"] < 20.0
    gate_a_g3 = speedup >= 1_000.0
    gate_a_g4 = bool(results.get("determinism_r1e6_byte_identical"))
    print("\ngates:")
    print(
        f"  A-G1 median K=16 @ R=1e6 < 50us:        "
        f"{'PASS' if gate_a_g1 else 'FAIL'} ({big_stats['median_us']}us)"
    )
    print(
        f"  A-G2 median K=16 @ R=100k < 20us:       "
        f"{'PASS' if gate_a_g2 else 'FAIL'} ({small_stats['median_us']}us)"
    )
    print(
        f"  A-G3 speedup vs snapshot-extract >= 1000x @ R=100k: "
        f"{'PASS' if gate_a_g3 else 'FAIL'} ({speedup:,.0f}x)"
    )
    print(
        f"  A-G4 byte-identical raw buffers:         "
        f"{'PASS' if gate_a_g4 else 'FAIL'}"
    )

    print("\nrunning procedural suites for A-G5 ...", flush=True)
    suite = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_procedural_runtime.py",
            "tests/test_procedural_runtime_ep.py",
            "tests/test_procedural_read_cells.py",
        ],
        cwd=repo_root,
        check=False,
    )
    gate_a_g5 = suite.returncode == 0
    print(f"  A-G5 procedural suites green:            {'PASS' if gate_a_g5 else 'FAIL'}")

    results["gates"] = {
        "A-G1": {"pass": gate_a_g1, "threshold_us": 50.0, **big_stats},
        "A-G2": {"pass": gate_a_g2, "threshold_us": 20.0, **small_stats},
        "A-G3": {"pass": gate_a_g3, "threshold_x": 1000.0, "speedup_x": round(speedup, 1)},
        "A-G4": {"pass": gate_a_g4},
        "A-G5": {"pass": gate_a_g5},
    }

    out_path = Path(__file__).with_name("l4_seams_acceptance.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    all_pass = all(
        gate["pass"] for gate in results["gates"].values()  # type: ignore[union-attr]
    )
    print("ALL GATES:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
