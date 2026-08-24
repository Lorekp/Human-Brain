#!/usr/bin/env python3
"""APICAL-AND week-1 experiment driver (preregistration §6-7, commit 05fb3ef).

Modes
-----
--fixture : N=96, 3 VAL items, 2 noise reps. Infrastructure leg only; its
            numbers gate nothing except readiness (preregistration §6.1).
--cohort  : the single decisive pass. N=1024, 18-arm batched grid:
            (1) flag-off parity gate vs native RecurrentCellCircuit on real
            corpus text (>=300 tokens, bit-exact; aborts everything on miss);
            (2) G-A1 conjunction battery: VAL items x {CT,C,T} x noise reps;
            (3) full TRAIN+VAL teacher-forced replay logging decision-point
            somatic state per arm (consumed by probes.py for G-A2/G-A3).

Outputs land under experiments/apical_probes/: parity_check_<tag>.json,
conjunction_results_<tag>.json, logged_states_apical_<tag>.npz +
logged_states_apical_<tag>_manifest.json (the npz stays local).

Deterministic: fixed seeds everywhere; items are independent (each starts from
reset), so they distribute over worker processes without changing any bit of
arithmetic — wiring is rebuilt deterministically per process from the frozen
seed and results are reassembled in submission order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.cells import build_arm_grid  # noqa: E402

HARNESS_PATH = Path(__file__).resolve().parent / "two_comp_harness.py"
SPEC = importlib.util.spec_from_file_location("two_comp_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules["two_comp_harness"] = HARNESS
SPEC.loader.exec_module(HARNESS)

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig  # noqa: E402

LADDER_PATH = REPO / "benchmarks" / "ladder_experiments.py"
LSPEC = importlib.util.spec_from_file_location("hb_ladder_experiments", LADDER_PATH)
assert LSPEC is not None and LSPEC.loader is not None
LADDER = importlib.util.module_from_spec(LSPEC)
sys.modules["hb_ladder_experiments"] = LADDER
LSPEC.loader.exec_module(LADDER)

SCHEMA_VERSION = "humanbrain/apical-probes/v1"
DEFAULT_FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)
OUT_DIR = Path(__file__).resolve().parent

# ---- frozen constants (preregistration sections 4 and 7) -------------------
SEED = 7
W_CTX_SEED = HARNESS.ContextSummarizer.W_CTX_SEED  # 20260824
NOISE_SEEDS = tuple(range(9001, 9009))
NOISE_SIGMA_PA = 50.0
T_CTX_MS = 300.0
RESPONSE_MS = 150.0
MAX_BURST_TOKENS = 30   # 30 x 5 ms ladder interval = 150 ms of delivery
DT_MS = 0.05
TOKEN_INTERVAL_MS = 5.0
# ----------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def split_prompt(serialized: str) -> tuple[str, str]:
    marker = "\nA:"
    cut = serialized.rindex(marker) + len(marker)
    return serialized[:cut], serialized[cut:]


def training_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus(freeze_dir: Path):
    evidence = freeze_dir / "evidence"
    train_man = json.loads((evidence / "qa_manifest_train.json").read_text(encoding="utf-8"))
    val_man = json.loads((evidence / "qa_manifest_val.json").read_text(encoding="utf-8"))
    return train_man["items"], val_man["items"]


def make_wiring(n_cells: int):
    """Ladder-config circuit whose wiring arrays are copied bit-faithfully."""
    config = RecurrentCircuitConfig(
        reservoir_cells=n_cells,
        input_fanout=LADDER._INPUT_FANOUT,
        recurrent_fanout=LADDER._RECURRENT_FANOUT,
        random_seed=SEED,
        readout_learning_rule="ridge",
        inhibitory_fraction=LADDER.DEFAULT_INHIBITORY_FRACTION,
        feature_family="traces",
    )
    return RecurrentCellCircuit(token_cells=256, config=config)


def parity_gate(n_cells: int, text: str, min_tokens: int = 300) -> dict:
    """Flag-off numpy token loop must equal native observe() bit-for-bit."""
    circuit = make_wiring(n_cells)
    runner = HARNESS.ApicalTokenRunner.from_circuit(circuit, arm_count=1)
    ids = list(bytes(text, "utf-8"))[:min_tokens]
    diverged = []
    native_spikes = 0
    for index, token in enumerate(ids):
        step = circuit.observe(int(token))
        native_spikes += int(step.reservoir_spikes)
        runner.observe(int(token))
        checks = {
            "voltage": np.array_equal(
                circuit._reservoir.state.voltage_relative_mv,
                runner.population.state.voltage_relative_mv[0],
            ),
            "threshold_spike": np.array_equal(
                circuit._reservoir.state.threshold_spike_mv,
                runner.population.state.threshold_spike_mv[0],
            ),
            "threshold_voltage": np.array_equal(
                circuit._reservoir.state.threshold_voltage_mv,
                runner.population.state.threshold_voltage_mv[0],
            ),
            "after_currents": np.array_equal(
                circuit._reservoir.state.after_spike_currents_pa,
                runner.population.state.after_spike_currents_pa[0],
            ),
            "refractory": np.array_equal(
                circuit._reservoir.state.refractory_steps,
                runner.population.state.refractory_steps[0],
            ),
            "synaptic_current": np.array_equal(
                circuit._synaptic_current_pa, runner.synaptic_current_pa[0]
            ),
            "eligibility": np.array_equal(
                circuit._feature_traces.reshape(-1, n_cells), runner.eligibility[0]
            ),
            "token_spikes": np.array_equal(
                circuit._token_spike_counts, runner.token_spike_counts[0]
            ),
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            diverged.append({"token_index": index, "arrays": bad})
            break
    return {
        "n_cells": n_cells,
        "tokens_checked": len(ids),
        "bit_exact": not diverged,
        "divergence": diverged,
        "native_spikes_seen": native_spikes,
    }


class ArmBatch:
    """One TwoCompartmentGLIF5 grid instance plus its synaptic surroundings."""

    def __init__(self, n_cells: int):
        self.n_cells = n_cells
        grid = build_arm_grid()
        self.arm_names = list(grid.keys())
        values = list(grid.values())
        wiring_circuit = make_wiring(n_cells)
        self.runner = HARNESS.ApicalTokenRunner.from_circuit(
            wiring_circuit,
            arm_count=len(values),
            tau_a_ms=[v["tau_a_ms"] for v in values],
            theta_p=[v["theta_p"] for v in values],
            g_gate=[v["g_gate"] for v in values],
            gate_duration_ms=100.0,
        )
        del wiring_circuit
        self.population = self.runner.population

    def reset(self) -> None:
        self.runner.reset()

    def snapshot(self) -> dict[str, np.ndarray]:
        pop = self.population
        return {
            "voltage": pop.state.voltage_relative_mv.copy(),
            "after": pop.state.after_spike_currents_pa.copy(),
            "th_spike": pop.state.threshold_spike_mv.copy(),
            "th_voltage": pop.state.threshold_voltage_mv.copy(),
            "refractory": pop.state.refractory_steps.copy(),
            "apical": pop.apical_voltage_mv.copy(),
            "gate": pop.gate_remaining.copy(),
            "synaptic": self.runner.synaptic_current_pa.copy(),
            "eligibility": self.runner.eligibility.copy(),
        }

    def restore(self, snap: dict[str, np.ndarray]) -> None:
        pop = self.population
        pop.state.voltage_relative_mv[...] = snap["voltage"]
        pop.state.after_spike_currents_pa[...] = snap["after"]
        pop.state.threshold_spike_mv[...] = snap["th_spike"]
        pop.state.threshold_voltage_mv[...] = snap["th_voltage"]
        pop.state.refractory_steps[...] = snap["refractory"]
        pop.apical_voltage_mv[...] = snap["apical"]
        pop.gate_remaining[...] = snap["gate"]
        self.runner.synaptic_current_pa[...] = snap["synaptic"]
        self.runner.eligibility[...] = snap["eligibility"]


# --------------------------- multiprocessing plumbing ------------------------
# Every QA item and every conjunction item is independent (each starts from
# batch.reset()), so items distribute over workers without changing any bit of
# arithmetic: wiring is rebuilt deterministically per process from the frozen
# seed, and results are reassembled in submission order.

_WORKER_BATCH: ArmBatch | None = None


def _init_worker(n_cells: int) -> None:
    global _WORKER_BATCH
    _WORKER_BATCH = ArmBatch(n_cells)


def _worker_batch() -> ArmBatch:
    assert _WORKER_BATCH is not None
    return _WORKER_BATCH


def _conjunction_item(job: tuple[int, dict], reps: int) -> dict:
    """All reps x {CT,C,T} for ONE VAL item; returns its record.

    The driven context phase (T_CTX_MS, no burst) runs once per rep; CT and C
    branch from its snapshot with identical continuing noise streams (generator
    state captured/restored). T runs its own undriven phase. Responses are
    total somatic spikes over the RESPONSE_MS window starting at burst onset.
    """
    item_index, item = job
    batch = _worker_batch()
    arms = batch.population.arm_count
    ctx_steps = round(T_CTX_MS / DT_MS)
    window_steps = round(RESPONSE_MS / DT_MS)
    prompt_text, answer_text = split_prompt(item["serialized_prompt_answer"])
    answer_ids = list(bytes(answer_text, "utf-8"))
    schedule = burst_schedule(answer_ids, batch.runner.pulse_steps)
    drive = full_prompt_drive(prompt_text, batch.n_cells)
    ct_reps: list[np.ndarray] = []
    c_reps: list[np.ndarray] = []
    t_reps: list[np.ndarray] = []
    context_spikes_reps: list[float] = []
    for noise_seed in NOISE_SEEDS[:reps]:
        # shared DRIVEN context phase
        batch.reset()
        batch.population.set_apical_drive(np.repeat(drive, arms, axis=0))
        rng = np.random.default_rng(noise_seed)
        batch.runner.run_free_window(ctx_steps, noise_rng=rng, noise_sigma=NOISE_SIGMA_PA)
        context_driven_spikes = float(batch.runner.last_total_spikes.mean())
        context_spikes_reps.append(context_driven_spikes)
        shared_state = batch.snapshot()
        shared_rng_state = rng.bit_generator.state
        # CT branch: burst arrives, drive stays on
        ct_reps.append(batch.runner.run_free_window(
            window_steps, schedule=schedule, noise_rng=rng, noise_sigma=NOISE_SIGMA_PA
        ))
        # C branch: same trajectory, no burst
        batch.restore(shared_state)
        rng_c = np.random.default_rng(noise_seed)
        rng_c.bit_generator.state = shared_rng_state
        c_reps.append(batch.runner.run_free_window(
            window_steps, schedule=None, noise_rng=rng_c, noise_sigma=NOISE_SIGMA_PA
        ))
        # T branch: undriven context, then burst
        batch.reset()
        batch.population.set_apical_drive(None)
        rng_t = np.random.default_rng(noise_seed)
        batch.runner.run_free_window(ctx_steps, noise_rng=rng_t, noise_sigma=NOISE_SIGMA_PA)
        t_reps.append(batch.runner.run_free_window(
            window_steps, schedule=schedule, noise_rng=rng_t, noise_sigma=NOISE_SIGMA_PA
        ))
    ct_mean = np.mean([r.sum(axis=1) for r in ct_reps], axis=0)
    c_mean = np.mean([r.sum(axis=1) for r in c_reps], axis=0)
    t_mean = np.mean([r.sum(axis=1) for r in t_reps], axis=0)
    denom = np.maximum(c_mean, t_mean)
    arms_count = arms
    ci_pair = np.full(arms_count, np.nan)
    positive_denom = denom > 0
    ci_pair[positive_denom] = ct_mean[positive_denom] / denom[positive_denom]
    silent_and_positive = (~positive_denom) & (ct_mean > 0)
    ci_pair[silent_and_positive] = np.inf
    ct_sums = [r.sum(axis=1) for r in ct_reps]
    t_sums = [r.sum(axis=1) for r in t_reps]
    return {
        "val_index": item_index,
        "qa_id": item.get("qa_id"),
        "burst_len": len(answer_ids),
        "context_driven_spikes_mean_per_rep": context_spikes_reps,
        "ct_mean_per_arm": ct_mean.tolist(),
        "c_mean_per_arm": c_mean.tolist(),
        "t_mean_per_arm": t_mean.tolist(),
        "ci_pair_per_arm": [None if np.isnan(v) else float(v) for v in ci_pair],
        "ct_sd_across_reps": (
            np.std(ct_sums, axis=0, ddof=1).tolist() if reps > 1 else [0.0] * arms_count
        ),
        "t_sd_across_reps": (
            np.std(t_sums, axis=0, ddof=1).tolist() if reps > 1 else [0.0] * arms_count
        ),
    }


def _replay_item(job: tuple[int, list[int], int]) -> dict:
    """Teacher-forced replay of ONE item; returns its decision-point store."""
    qa_index, full_ids, boundary_index = job
    batch = _worker_batch()
    arms = batch.population.arm_count
    n_cells = batch.n_cells
    batch.reset()
    summarizer = HARNESS.ContextSummarizer(n_cells)
    totals = np.zeros(arms, dtype=np.int64)
    frozen_drive: np.ndarray | None = None
    boundary: dict | None = None
    for index, token in enumerate(full_ids):
        if frozen_drive is None:
            # causal context summary: includes the byte observed now
            summarizer.update_byte(int(token))
            drive_row = summarizer.drive_vector()[None, :]
        else:
            drive_row = frozen_drive
        batch.population.set_apical_drive(np.repeat(drive_row, arms, axis=0))
        batch.runner.observe(int(token))
        totals += batch.runner.last_total_spikes
        if index == boundary_index:
            frozen_drive = drive_row  # u_a freezes at its decision-point value
            pop = batch.population
            boundary = {
                "voltage": pop.state.voltage_relative_mv.astype(np.float32).copy(),
                "spikes": batch.runner.token_spike_counts.astype(np.uint32).copy(),
                "traces": (
                    batch.runner.eligibility.astype(np.float32)
                    .reshape(arms, -1)
                    .copy()
                ),
                "apical": pop.apical_voltage_mv.astype(np.float32).copy(),
                "gated_frac": float((pop.gate_remaining > 0).mean()),
            }
    if boundary is None:
        raise RuntimeError(f"item {qa_index}: decision point never reached")
    return {
        "qa_index": qa_index,
        "boundary": boundary,
        "total_somatic_spikes": int(totals.sum()),
    }


def full_prompt_drive(prompt_text: str, n_cells: int) -> np.ndarray:
    summarizer = HARNESS.ContextSummarizer(n_cells)
    for byte in bytes(prompt_text, "utf-8"):
        summarizer.update_byte(int(byte))
    return summarizer.drive_vector()[None, :]  # (1, cells)


def burst_schedule(answer_ids: list[int], pulse_steps: int) -> dict[int, int]:
    """Relative-substep -> token deliveries on the ladder cadence.

    Each token drives its afferents continuously for ``pulse_steps`` (the
    fitted 1.5 ms input pulse = 30 substeps), exactly like the native
    observe(); successive tokens start one 5 ms token-interval apart.
    """
    interval_steps = round(TOKEN_INTERVAL_MS / DT_MS)
    schedule: dict[int, int] = {}
    for slot, token in enumerate(answer_ids[:MAX_BURST_TOKENS]):
        base = slot * interval_steps
        for offset in range(min(pulse_steps, interval_steps)):
            schedule[base + offset] = int(token)
    return schedule


def _run_pool(jobs_with_args, n_cells: int, workers: int, mapper):
    import multiprocessing as mp

    with mp.Pool(
        processes=workers, initializer=_init_worker, initargs=(n_cells,)
    ) as pool:
        return mapper(pool)


def summarize_conjunction(records: list[dict], arm_names: list[str]) -> dict:
    arms = len(arm_names)
    per_arm_ci: list[list[float]] = [[] for _ in range(arms)]
    frac_gt1 = [0] * arms
    excluded = [0] * arms
    for record in records:
        for arm in range(arms):
            value = record["ci_pair_per_arm"][arm]
            if value is None:
                excluded[arm] += 1
                continue
            per_arm_ci[arm].append(value)
            if value > 1.0:
                frac_gt1[arm] += 1
    summary = {}
    n_pairs = len(records)
    for arm in range(arms):
        values = per_arm_ci[arm]
        entry = {
            "n_pairs_scored": len(values),
            "n_pairs_excluded_both_silent": excluded[arm],
            "median_ci": float(np.median(values)) if values else None,
            "mean_ci": float(np.mean(values)) if values else None,
            "fraction_pairs_ci_above_1": frac_gt1[arm] / max(len(values), 1),
        }
        ci = entry["median_ci"]
        entry["GA1_pass"] = bool(ci is not None and np.isfinite(ci) and ci >= 1.5)
        summary[arm_names[arm]] = entry
    return summary


def build_tokenizer(train_items: list[dict]):
    from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

    texts = [it["serialized_prompt_answer"] for it in train_items]
    return BytePhraseTokenizer.fit(
        texts, TokenizerConfig(max_phrases=0, min_frequency=3)
    )


def verify_byte_identity(tokenizer, sample_text: str) -> bool:
    ids = tokenizer.encode(sample_text)
    return tuple(ids) == tuple(bytes(sample_text, "utf-8"))


def item_token_pairs(tokenizer, items: list[dict]) -> list[tuple[list[int], int]]:
    pairs = []
    for item in items:
        prompt_text, _rest = split_prompt(item["serialized_prompt_answer"])
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(item["serialized_prompt_answer"])
        if tuple(full_ids[: len(prompt_ids)]) != tuple(prompt_ids):
            raise ValueError("prompt is not a byte-token prefix")
        pairs.append((list(full_ids), len(prompt_ids) - 1))
    return pairs


def replay_qa(n_cells: int, tokenizer_items, label: str, workers: int) -> dict:
    jobs = [(i, ids, b) for i, (ids, b) in enumerate(tokenizer_items)]
    t0 = time.perf_counter()
    import multiprocessing as mp

    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(n_cells,)) as pool:
        results = pool.map(_replay_item, jobs, chunksize=1)
    results.sort(key=lambda r: r["qa_index"])
    store = {
        "qa_index": np.asarray([r["qa_index"] for r in results], dtype=np.int32),
        "boundary_voltage": np.stack([r["boundary"]["voltage"] for r in results]),
        "boundary_spikes": np.stack([r["boundary"]["spikes"] for r in results]),
        "boundary_traces": np.stack([r["boundary"]["traces"] for r in results]),
        "boundary_apical": np.stack([r["boundary"]["apical"] for r in results]),
        "boundary_gated_frac": np.asarray(
            [r["boundary"]["gated_frac"] for r in results], dtype=np.float32
        ),
        "total_somatic_spikes": np.asarray(
            [r["total_somatic_spikes"] for r in results], dtype=np.int64
        ),
    }
    log(f"replay[{label}]: {len(results)} items in {time.perf_counter() - t0:.0f}s")
    return store


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--cohort", action="store_true")
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--skip-parity", action="store_true",
                        help="cohort only: reuse an existing bit-exact parity_check_cohort.json")
    args = parser.parse_args()
    if args.fixture == args.cohort:
        raise SystemExit("choose exactly one of --fixture / --cohort")

    freeze_dir = Path(args.freeze_dir)
    train_items, val_items = load_corpus(freeze_dir)
    fixture = args.fixture
    tag = "fixture" if fixture else "cohort"
    n_cells = 96 if fixture else 1024
    reps = 2 if fixture else len(NOISE_SEEDS)
    val_slice = val_items[:3] if fixture else val_items
    workers = 1 if fixture else max(1, args.workers)

    out = OUT_DIR

    # ---------------- (1) flag-off parity gate on real corpus text ----------
    parity_path = out / f"parity_check_{tag}.json"
    if not (args.skip_parity and parity_path.exists()):
        log(f"parity gate: N={n_cells}, first TRAIN item prompt")
        parity = parity_gate(
            n_cells, split_prompt(train_items[0]["serialized_prompt_answer"])[0]
        )
        parity_path.write_text(json.dumps(parity, indent=2), encoding="utf-8")
        if not parity["bit_exact"]:
            log("PARITY FAILED - aborting before any flag-on run")
            raise SystemExit(2)
        log(
            f"parity gate PASS ({parity['tokens_checked']} tokens, "
            f"{parity['native_spikes_seen']} native spikes)"
        )

    # ---------------- wiring sanity + tokenizer ------------------------------
    tokenizer = build_tokenizer(train_items)
    check_text = train_items[0]["serialized_prompt_answer"]
    if not verify_byte_identity(tokenizer, check_text):
        raise RuntimeError("tokenizer is not byte-identity at vocab 256")
    log("byte-token identity verified")

    # ---------------- (2) G-A1 conjunction battery ---------------------------
    log(f"conjunction battery: {len(val_slice)} items x 3 conditions x {reps} reps "
        f"on {workers} worker(s)")
    import multiprocessing as mp

    jobs = list(enumerate(val_slice))
    t0 = time.perf_counter()
    if workers <= 1:
        _init_worker(n_cells)
        records = [_conjunction_item(job, reps) for job in jobs]
    else:
        with mp.Pool(processes=workers, initializer=_init_worker,
                     initargs=(n_cells,)) as pool:
            records = pool.starmap(
                _conjunction_item, [(job, reps) for job in jobs], chunksize=1
            )
    records.sort(key=lambda r: r["val_index"])
    log(f"battery finished in {time.perf_counter() - t0:.0f}s")

    grid = build_arm_grid()
    arm_names = list(grid.keys())
    summary = summarize_conjunction(records, arm_names)
    conj_out = {
        "schema_version": SCHEMA_VERSION,
        "mode": tag,
        "n_cells": n_cells,
        "noise_seeds": list(NOISE_SEEDS[:reps]),
        "noise_sigma_pa": NOISE_SIGMA_PA,
        "t_ctx_ms": T_CTX_MS,
        "response_ms": RESPONSE_MS,
        "max_burst_tokens": MAX_BURST_TOKENS,
        "burst_delivery_interval_ms": TOKEN_INTERVAL_MS,
        "arm_names": arm_names,
        "summary_per_arm": summary,
        "records": records,
    }
    (out / f"conjunction_results_{tag}.json").write_text(
        canonical_json(conj_out).decode("utf-8"), encoding="utf-8"
    )
    passing = [(name, s) for name, s in summary.items() if s["GA1_pass"]]
    log(f"G-A1: {len(passing)}/{len(arm_names)} arms pass (median CI >= 1.5)")
    for name, s in sorted(passing, key=lambda kv: -(kv[1]["median_ci"] or 0))[:6]:
        log(f"  {name}: median CI={s['median_ci']:.3f}")

    # ---------------- (3) QA decision-point replay ---------------------------
    train_slice = train_items if not fixture else train_items[:8]
    train_pairs = item_token_pairs(tokenizer, train_slice)
    val_pairs = item_token_pairs(tokenizer, val_items if not fixture else val_items[:3])
    train_store = replay_qa(n_cells, train_pairs, "train", workers)
    val_store = replay_qa(n_cells, val_pairs, "val", workers)

    arrays = {f"train_{k}": v for k, v in train_store.items()}
    arrays.update({f"val_{k}": v for k, v in val_store.items()})
    npz_path = out / f"logged_states_apical_{tag}.npz"
    np.savez_compressed(npz_path, **arrays)

    meta_rows = []
    for split, items in (("train", train_slice), ("val", val_items if not fixture else val_items[:3])):
        for position, item in enumerate(items):
            prompt_text, _rest = split_prompt(item["serialized_prompt_answer"])
            meta_rows.append({
                "split": split,
                "position": position,
                "qa_id": item.get("qa_id"),
                "family": item.get("family"),
                "answer_text": item.get("answer"),
                "chapter_distance": (item.get("meta") or {}).get("chapter_distance"),
                "prompt_bytes": len(bytes(prompt_text, "utf-8")),
            })
    array_hashes = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_bytes(np.ascontiguousarray(value).tobytes()),
        }
        for key, value in arrays.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": tag,
        "n_cells": n_cells,
        "seed": SEED,
        "w_ctx_seed": W_CTX_SEED,
        "arm_names": arm_names,
        "grid": grid,
        "gate_duration_ms": 100.0,
        "training_digest": training_digest(
            [it["serialized_prompt_answer"] for it in train_items]
        ),
        "curriculum_digest": sha256_bytes(canonical_json([
            [
                {k: row[k] for k in ("qa_id", "family", "prompt_bytes")}
                for row in meta_rows if row["split"] == split
            ]
            for split in ("train", "val")
        ])),
        "rows": meta_rows,
        "arrays": array_hashes,
        "ustate_findings_present": (
            REPO / "experiments" / "ustate_probes" / "FINDINGS.md"
        ).exists(),
        "context_drive_source": "fallback_context_summary",
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }
    (out / f"logged_states_apical_{tag}_manifest.json").write_text(
        canonical_json(manifest).decode("utf-8"), encoding="utf-8"
    )
    log(f"wrote {npz_path.name} + manifest ({tag})")


if __name__ == "__main__":
    main()
