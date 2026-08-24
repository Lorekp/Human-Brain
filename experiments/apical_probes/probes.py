#!/usr/bin/env python3
"""APICAL-AND offline probes (preregistration §4-5, commit 05fb3ef).

Consumes the cohort artifacts and evaluates the frozen gates per arm:

- G-A1 conjunction index (from conjunction_results_cohort.json, already run);
- G-A2 separation: R0-G1 definition on VAL decision-point trace-bank vectors
  (f32, bias excluded): pooled MEDIAN 1 - cosine over unordered pairs within
  the same QA family stratum; PASS iff > 0.10;
- G-A3 decode z >= 2 vs unigram: span accuracy via the r0-c machinery VERBATIM
  (logistic + 2-layer MLP, TRAIN-answer-vocab head, ranked-consumption,
  639-position denominator); PASS iff best decoder >= 0.061851478566709396
  (= unigram 29/639 + 2 sigma). Controls recomputed identically.

Emits probes_results.json with every number verbatim plus the mechanical
per-arm and program verdicts (including the pre-declared falsifier clause).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

R0C_PATH = REPO / "experiments" / "r0c_probes" / "probes.py"
RSPEC = importlib.util.spec_from_file_location("r0c_probes", R0C_PATH)
assert RSPEC is not None and RSPEC.loader is not None
R0C = importlib.util.module_from_spec(RSPEC)
sys.modules["r0c_probes"] = R0C
RSPEC.loader.exec_module(R0C)

SCHEMA_VERSION = "humanbrain/apical-verdict/v1"
GA2_THRESHOLD = 0.10           # median 1-cosine must exceed this
GA3_THRESHOLD = 0.061851478566709396  # unigram 29/639 + 2 sigma (frozen)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def separation_1minus_cos(matrix: np.ndarray, families: list[str]) -> dict:
    """Pooled median 1-cosine over unordered pairs within a family stratum."""
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = matrix / safe[:, None]
    cos = unit @ unit.T
    one_minus = 1.0 - cos
    values: list[float] = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if families[i] == families[j]:
                values.append(float(one_minus[i, j]))
    return {
        "n_pairs": len(values),
        "median_1minus_cos": float(np.median(values)) if values else None,
    }


def decode_arm(Xtr: np.ndarray, Xva: np.ndarray) -> dict:
    """The r0-c P1 decoder pipeline verbatim for ONE standardized feature set."""
    train_items = _TRAIN_ITEMS
    val_items = _VAL_ITEMS
    classes = R0C.answer_classes(train_items)
    class_names = sorted(classes, key=classes.get)
    Y_train = np.stack(
        [R0C._soft_targets(R0C.words(it["answer"]), classes) for it in train_items]
    )
    scorer = R0C.SpanScorer(val_items, R0C.SPAN_PREDICTIONS)
    prior_ranking = R0C.majority_ranking(train_items, classes)
    controls = {
        "majority_prior_within_answer_vocab": scorer.score(
            [prior_ranking] * len(val_items)
        ),
    }
    rng = np.random.default_rng(R0C.CONTROL_SEED)
    random_rankings = []
    for _ in val_items:
        order = class_names[:]
        rng.shuffle(order)
        random_rankings.append(order)
    controls["random_ranking_seed11"] = scorer.score(random_rankings)

    d = Xtr.shape[1]
    max_d = 4096
    if d > max_d:
        rng_cols = np.random.default_rng(R0C.PROBE_SEED)
        cols = np.sort(rng_cols.choice(d, size=max_d, replace=False))
        Xtr, Xva = Xtr[:, cols], Xva[:, cols]

    W, b = R0C.train_softmax_logistic(Xtr, Y_train)
    logits = Xva @ W + b
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    logistic_score = scorer.score(
        [[class_names[j] for j in np.argsort(-row)[:64]] for row in probs]
    )

    model = R0C.train_mlp(Xtr, Y_train)
    probs_mlp = R0C.mlp_probs(model, Xva)
    mlp_score = scorer.score(
        [[class_names[j] for j in np.argsort(-row)[:64]] for row in probs_mlp]
    )

    rng_lab = np.random.default_rng(R0C.CONTROL_SEED)
    perm = rng_lab.permutation(len(train_items))
    model_shuf = R0C.train_mlp(Xtr, Y_train[perm])
    probs_shuf = R0C.mlp_probs(model_shuf, Xva)
    shuf_score = scorer.score(
        [[class_names[j] for j in np.argsort(-row)[:64]] for row in probs_shuf]
    )

    return {
        "feature_dim_used": int(min(d, max_d)),
        "logistic": logistic_score,
        "mlp": mlp_score,
        "mlp_shuffled_label_control": shuf_score,
        "controls": controls,
    }


_TRAIN_ITEMS: list[dict] | None = None
_VAL_ITEMS: list[dict] | None = None


def _init_decode_worker(train_items: list[dict], val_items: list[dict]) -> None:
    global _TRAIN_ITEMS, _VAL_ITEMS
    _TRAIN_ITEMS = train_items
    _VAL_ITEMS = val_items


def _decode_sets_worker(arm: str, raw_pair: tuple, traces_pair: tuple):
    t0 = time.perf_counter()
    sets = {
        "raw_voltage_spikes": decode_arm(*raw_pair),
        "traces_bank": decode_arm(*traces_pair),
    }
    log(f"G-A3 {arm} done in {time.perf_counter() - t0:.0f}s")
    return arm, sets


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(
        __import__("os").environ.get(
            "LOCALAPPDATA",
            str(Path.home() / "AppData" / "Local"),
        )
    ) + "\\Temp\\hb_bible_freeze")
    parser.add_argument("--workers", type=int, default=min(9, max(1, (__import__("os").cpu_count() or 4) - 3)))
    args = parser.parse_args()

    conj_path = OUT_DIR / "conjunction_results_cohort.json"
    npz_path = OUT_DIR / "logged_states_apical_cohort.npz"
    manifest_path = OUT_DIR / "logged_states_apical_cohort_manifest.json"

    conjunction = json.loads(conj_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    z = np.load(npz_path)

    freeze_dir = Path(args.freeze_dir)
    ev = freeze_dir / "evidence"
    train_items = json.loads((ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
    val_items = json.loads((ev / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]

    arm_names = manifest["arm_names"]
    rows = manifest["rows"]
    train_meta = [r for r in rows if r["split"] == "train"]
    val_meta = [r for r in rows if r["split"] == "val"]
    assert len(val_meta) == z["val_qa_index"].shape[0]
    train_families = [r["family"] or "unknown" for r in train_meta]
    val_families = [r["family"] or "unknown" for r in val_meta]

    log(f"evaluating {len(arm_names)} arms x {{G-A2 separation, G-A3 decode}}")

    # ---------------- G-A2 separation (cheap, inline) ------------------------
    ga2 = {}
    for arm_index, arm in enumerate(arm_names):
        traces_val = z["val_boundary_traces"][:, arm_index, :].astype(np.float64)
        ga2[arm] = separation_1minus_cos(traces_val, val_families)
        log(f"G-A2 {arm}: median 1-cos={ga2[arm]['median_1minus_cos']:.6f} "
            f"({ga2[arm]['n_pairs']} pairs)")

    # ---------------- G-A3 decode (parallel over arms) -----------------------
    def job_sets(arm_index: int):
        voltage_tr = z["train_boundary_voltage"][:, arm_index, :].astype(np.float64)
        spikes_tr = z["train_boundary_spikes"][:, arm_index, :].astype(np.float64)
        voltage_va = z["val_boundary_voltage"][:, arm_index, :].astype(np.float64)
        spikes_va = z["val_boundary_spikes"][:, arm_index, :].astype(np.float64)
        raw_tr = np.concatenate([voltage_tr, spikes_tr], axis=1)
        raw_va = np.concatenate([voltage_va, spikes_va], axis=1)
        traces_tr = z["train_boundary_traces"][:, arm_index, :].astype(np.float64)
        traces_va = z["val_boundary_traces"][:, arm_index, :].astype(np.float64)
        return (
            R0C.standardize(raw_tr, raw_va),
            R0C.standardize(traces_tr, traces_va),
        )

    ga3: dict[str, dict] = {}
    if args.workers <= 1:
        _init_decode_worker(train_items, val_items)
        for arm_index, arm in enumerate(arm_names):
            raw_pair = job_sets(arm_index)[0]
            traces_pair = job_sets(arm_index)[1]
            ga3[arm] = {
                "raw_voltage_spikes": decode_arm(*raw_pair),
                "traces_bank": decode_arm(*traces_pair),
            }
            log(f"G-A3 {arm} done")
    else:
        import multiprocessing as mp

        jobs = []
        for arm_index, arm in enumerate(arm_names):
            raw_pair, traces_pair = job_sets(arm_index)
            jobs.append((arm, raw_pair, traces_pair))
        with mp.Pool(
            processes=args.workers,
            initializer=_init_decode_worker,
            initargs=(train_items, val_items),
        ) as pool:
            results = pool.starmap(_decode_sets_worker, jobs, chunksize=1)
        ga3 = {arm: sets for arm, sets in results}

    # ---------------- verdict assembly ---------------------------------------
    majority_prior = (
        ga3[arm_names[0]]["raw_voltage_spikes"]["controls"]
        ["majority_prior_within_answer_vocab"]["accuracy"]
    )
    verdicts = {}
    falsified_arms = []
    passing_arms = []
    for arm in arm_names:
        g1 = conjunction["summary_per_arm"][arm]["GA1_pass"]
        med = ga2[arm]["median_1minus_cos"]
        g2 = bool(med is not None and med > GA2_THRESHOLD)
        best_acc = max(
            max(ga3[arm][fs]["logistic"]["accuracy"],
                ga3[arm][fs]["mlp"]["accuracy"])
            for fs in ga3[arm]
        )
        g3 = bool(best_acc >= GA3_THRESHOLD)
        arm_pass = bool(g1 and g2 and g3)
        verdicts[arm] = {
            "GA1_conjunction_pass": g1,
            "GA1_median_ci": conjunction["summary_per_arm"][arm]["median_ci"],
            "GA2_separation_median_1minus_cos": med,
            "GA2_pass": g2,
            "GA3_best_span_accuracy": best_acc,
            "GA3_pass": g3,
            "ARM_PASS": arm_pass,
        }
        if g1 and best_acc < majority_prior:
            falsified_arms.append({
                "arm": arm,
                "GA1_median_ci": conjunction["summary_per_arm"][arm]["median_ci"],
                "best_decode": best_acc,
                "majority_prior": majority_prior,
            })
        if arm_pass:
            passing_arms.append(arm)

    program_verdict = "PASS" if passing_arms else "NEGATIVE-FALSIFIED"
    results = {
        "schema_version": SCHEMA_VERSION,
        "gates": {
            "GA2_threshold_median_1minus_cos": GA2_THRESHOLD,
            "GA3_threshold_span_accuracy": GA3_THRESHOLD,
        },
        "majority_prior_within_answer_vocab": majority_prior,
        "comparators": {
            "unigram_smoothed": R0C.UNIGRAM_SPAN_ACC,
            "trigram_smoothed": R0C.TRIGRAM_SPAN_ACC,
        },
        "GA2_per_arm": ga2,
        "GA3_per_arm": ga3,
        "verdicts_per_arm": verdicts,
        "passing_arms": passing_arms,
        "falsifier_triggered_arms": falsified_arms,
        "PROGRAM_VERDICT": program_verdict,
    }
    (OUT_DIR / "probes_results.json").write_text(
        canonical_json_safe(results), encoding="utf-8"
    )
    log(f"PROGRAM VERDICT: {program_verdict}")
    for arm, v in verdicts.items():
        log(f"  {arm}: G1={v['GA1_conjunction_pass']} "
            f"(CI={v['GA1_median_ci']}) G2={v['GA2_pass']} "
            f"({v['GA2_separation_median_1minus_cos']:.4f}) "
            f"G3={v['GA3_pass']} ({v['GA3_best_span_accuracy']:.4f})")


def canonical_json_safe(value) -> str:
    import json as _json

    return _json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=True, indent=1)


if __name__ == "__main__":
    main()
