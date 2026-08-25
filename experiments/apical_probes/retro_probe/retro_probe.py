#!/usr/bin/env python3
"""P-A step -1 retro-probe (path-decision-synthesis.md @ e16d22c, §Probe P-A).

Question: do the SAVED continuous decision-point voltages in
logged_states_apical_cohort.npz linearly decode decision-point variables
ABOVE the shuffled-label band?

The APICAL-AND G-A3 decode used {raw voltage + spike counts} and {trace
banks}; the continuous APICAL compartment voltage (`boundary_apical`, u_A)
was never probed. This probe covers, per simulation arm:

  F-voltage : somatic membrane voltage at the decision point (1024 cells)
  F-apical  : apical compartment voltage u_A at the decision point (1024)
  F-concat  : both concatenated (2048)

Targets (frozen before any probe ran):
  T1 family3        strict split: fit TRAIN (n=200), eval VAL (n=60),
                    3-way {who_said, address_lookup, verse_continuation}.
  T2 speaker_pooled POOLED train+val who_said items (n=80), speakers with
                    >=2 pooled occurrences + OTHER bucket; stratified 5-fold
                    CV (seed 7). DEVIATION from strict split, forced by the
                    corpus: train/val speaker vocabularies share only
                    {And, Moses}, which makes strict-split speaker naming
                    vacuous. Disclosed in FINDINGS.
  T2b moses_binary  pooled who_said, binary {Moses, rest} - high-power
                    version of T2 (same CV machinery).
  T3 chapter_dist   ridge regression on chapter_distance (who_said items,
                    strict split 60 train / 20 val), R^2 + permutation null.
  T4 span_fidelity  r0-c G-A3 protocol VERBATIM (softmax logistic,
                    TRAIN-answer-vocab head, SpanScorer 639 positions,
                    CONTROL_SEED=11 shuffled-label control) with feature sets
                    {apical_only} and {voltage_plus_apical}; directly
                    comparable to the committed G-A3 numbers. Also reproduces
                    one committed cell (reference arm x raw_voltage_spikes
                    logistic) as a harness-fidelity check.

Nulls: label-permutation replicates through the IDENTICAL pipeline (same
fixed C / alpha, same splits). Band reported verbatim (mean/sd/min/max/q95);
empirical one-sided p = (1 + #{null >= obs}) / (1 + n_perms).
Effect size: obs - shuffle_mean and Cohen's d. Wilson 95% CI on accuracies.

Decision rule (from synthesis, applied at program level): B' Stage-0 is
licensed iff a PRIMARY probe (T1 or T4) exceeds its shuffled band with
Bonferroni-adjusted p < 0.05 across arms AND a meaningful margin
(T1: >= +2 points absolute over band mean; T4: above band max). Otherwise
the falsifier fires: upstream state construction is dead, corpus-level B'
dies before any build. T2/T2b/T3 are secondary (power-limited) evidence.

CPU-only; deterministic seeds; npz integrity verified against the manifest
sha256 table before anything runs.
"""

from __future__ import annotations

import os
# Cap BLAS threads BEFORE numpy/sklearn imports: with N worker processes each
# spawning 12 BLAS threads, oversubscription made T2 ~10x slower (observed).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_NPZ = (r"C:/Users/laure/Desktop/DeepSim/.worktrees/"
               r"human-brain-a013-cognition-design/experiments/apical_probes/"
               r"logged_states_apical_cohort.npz")
MANIFEST = HERE.parent / "logged_states_apical_cohort_manifest.json"
RESULTS_PATH = HERE / "retro_probe_results.json"

MASTER_SEED = 7
N_PERMS_FAMILY = 200
N_PERMS_SPEAKER = 200
N_PERMS_RIDGE = 500
FIXED_C = 0.01          # frozen a priori; sensitivity grid run separately
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
FAMILIES = ["who_said", "address_lookup", "verse_continuation"]

SCHEMA_VERSION = "humanbrain/retro-probe/v1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_npz(npz_path: Path) -> dict:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    z = np.load(npz_path)
    report = {}
    for key, spec in man["arrays"].items():
        digest = sha256_bytes(np.ascontiguousarray(z[key]).tobytes())
        ok = digest == spec["sha256"]
        if not ok:
            raise RuntimeError(f"npz array {key} sha256 MISMATCH vs manifest")
        report[key] = {"shape": list(z[key].shape), "verified": True}
    log(f"npz integrity: all {len(report)} arrays match manifest sha256")
    return report


# ---------------------------------------------------------------- features

def feature_sets(z, arm_index: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tr_v = z["train_boundary_voltage"][:, arm_index, :].astype(np.float64)
    va_v = z["val_boundary_voltage"][:, arm_index, :].astype(np.float64)
    tr_a = z["train_boundary_apical"][:, arm_index, :].astype(np.float64)
    va_a = z["val_boundary_apical"][:, arm_index, :].astype(np.float64)
    return {
        "voltage": (tr_v, va_v),
        "apical": (tr_a, va_a),
        "concat": (np.concatenate([tr_v, tr_a], axis=1),
                   np.concatenate([va_v, va_a], axis=1)),
    }


def standardize(tr: np.ndarray, other: np.ndarray):
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (tr - mu) / sd, (other - mu) / sd


def wilson(k: int, n: int, zc: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + zc * zc / n
    center = (p + zc * zc / (2 * n)) / den
    half = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / den
    return (center - half, center + half)


def band_stats(obs: float, nulls: np.ndarray) -> dict:
    return {
        "observed": obs,
        "shuffle_mean": float(nulls.mean()),
        "shuffle_sd": float(nulls.std(ddof=1)),
        "shuffle_min": float(nulls.min()),
        "shuffle_max": float(nulls.max()),
        "shuffle_q95": float(np.quantile(nulls, 0.95)),
        "effect_abs": float(obs - nulls.mean()),
        "effect_cohens_d": (float((obs - nulls.mean()) / nulls.std(ddof=1))
                            if nulls.std(ddof=1) > 0 else None),
        "empirical_p_one_sided": float(
            (1 + int((nulls >= obs).sum())) / (1 + len(nulls))),
        "n_perms": int(len(nulls)),
    }


# ---------------------------------------------------------------- targets

def load_targets() -> dict:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"]
    # positional alignment verified against writer order (see FINDINGS):
    # rows[:200] <-> z['train_*'] axis 0, rows[200:] <-> z['val_*'] axis 0.
    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    assert len(tr_rows) == 200 and len(va_rows) == 60
    y_tr_family = np.array([FAMILIES.index(r["family"]) for r in tr_rows])
    y_va_family = np.array([FAMILIES.index(r["family"]) for r in va_rows])
    ws_tr = [r for r in tr_rows if r["family"] == "who_said"]
    ws_va = [r for r in va_rows if r["family"] == "who_said"]
    pooled = ws_tr + ws_va
    speakers = [r["answer_text"] for r in pooled]
    cnt = Counter(speakers)
    named = sorted(a for a, c in cnt.items() if c >= 2)
    name_to_cls = {a: i for i, a in enumerate(named)}
    OTHER = len(named)
    y_pool = np.array([name_to_cls.get(s, OTHER) for s in speakers])
    y_moses = np.array([1 if s == "Moses" else 0 for s in speakers])
    cd_tr = np.array([r["chapter_distance"] for r in ws_tr], dtype=np.float64)
    cd_va = np.array([r["chapter_distance"] for r in ws_va], dtype=np.float64)
    assert not np.isnan(cd_tr).any() and not np.isnan(cd_va).any()
    return {
        "arm_names": man["arm_names"],
        "y_tr_family": y_tr_family,
        "y_va_family": y_va_family,
        "ws_n_train": len(ws_tr),
        "ws_n_val": len(ws_va),
        "y_pool_speaker": y_pool,
        "speaker_class_names": named + ["OTHER"],
        "speaker_pooled_counts": dict(cnt),
        "y_moses": y_moses,
        "cd_tr": cd_tr,
        "cd_va": cd_va,
    }


# ------------------------------------------------------- T1 family probes

def run_family_config(seed_seq, Xtr, Xva, ytr, yva, n_perms) -> dict:
    rng = np.random.default_rng(seed_seq)
    Xtr_s, Xva_s = standardize(Xtr, Xva)
    model = LogisticRegression(C=FIXED_C, max_iter=1000)
    model.fit(Xtr_s, ytr)
    pred = model.predict(Xva_s)
    obs_acc = float((pred == yva).mean())
    obs_bacc = float(balanced_accuracy_score(yva, pred))
    k = int((pred == yva).sum())
    lo, hi = wilson(k, len(yva))
    maj = float(max((yva == c).mean() for c in np.unique(yva)))
    nulls = np.empty(n_perms)
    nulls_bacc = np.empty(n_perms)
    for i in range(n_perms):
        yp = rng.permutation(ytr)
        m = LogisticRegression(C=FIXED_C, max_iter=1000)
        m.fit(Xtr_s, yp)
        p = m.predict(Xva_s)
        nulls[i] = (p == yva).mean()
        nulls_bacc[i] = balanced_accuracy_score(yva, p)
    return {
        "val_accuracy": obs_acc,
        "wilson95": [lo, hi],
        "val_balanced_accuracy": obs_bacc,
        "majority_class_freq_val": maj,
        "shuffle_accuracy": band_stats(obs_acc, nulls),
        "shuffle_balanced_accuracy": band_stats(obs_bacc, nulls_bacc),
    }


def sensitivity_grid(Xtr, Xva, ytr, yva) -> list:
    Xtr_s, Xva_s = standardize(Xtr, Xva)
    out = []
    for C in (0.001, 0.01, 0.05, 0.1, 0.3, 1.0):
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr_s, ytr)
        pred = m.predict(Xva_s)
        acc = float((pred == yva).mean())
        # inner-CV pick on TRAIN only, then its VAL accuracy
        inner = StratifiedKFold(5, shuffle=True, random_state=MASTER_SEED)
        cvs = []
        for tr_i, va_i in inner.split(Xtr_s, ytr):
            mi = LogisticRegression(C=C, max_iter=1000).fit(
                Xtr_s[tr_i], ytr[tr_i])
            cvs.append(float((mi.predict(Xtr_s[va_i]) == ytr[va_i]).mean()))
        out.append({"C": C, "val_accuracy": acc,
                    "train_cv_accuracy_mean": float(np.mean(cvs))})
    return out


# ---------------------------------------------- T2/T2b pooled-CV probes

def run_pooled_cv_config(seed_seq, X, y, n_perms, n_classes) -> dict:
    rng = np.random.default_rng(seed_seq)
    skf = StratifiedKFold(5, shuffle=True, random_state=int(
        seed_seq.generate_state(1)[0]))
    Xs, _ = standardize(X, X)

    def cv_acc(labels):
        preds = np.empty_like(labels)
        for tr_i, te_i in skf.split(Xs, labels):
            m = LogisticRegression(C=FIXED_C, max_iter=1000)
            m.fit(Xs[tr_i], labels[tr_i])
            preds[te_i] = m.predict(Xs[te_i])
        return labels, preds

    labels, preds = cv_acc(y)
    obs_acc = float((preds == labels).mean())
    obs_bacc = float(balanced_accuracy_score(labels, preds))
    k = int((preds == labels).sum())
    lo, hi = wilson(k, len(labels))
    maj = float(max((labels == c).mean() for c in np.unique(labels)))
    nulls = np.empty(n_perms)
    nulls_bacc = np.empty(n_perms)
    for i in range(n_perms):
        yp = rng.permutation(y)
        lab_i, pr_i = cv_acc(yp)
        nulls[i] = (pr_i == lab_i).mean()
        nulls_bacc[i] = balanced_accuracy_score(lab_i, pr_i)
    return {
        "cv_accuracy": obs_acc,
        "wilson95": [lo, hi],
        "cv_balanced_accuracy": obs_bacc,
        "majority_class_freq": maj,
        "n_classes": int(n_classes),
        "shuffle_accuracy": band_stats(obs_acc, nulls),
        "shuffle_balanced_accuracy": band_stats(obs_bacc, nulls_bacc),
    }


# ------------------------------------------------------------ T3 ridge

def run_ridge_config(seed_seq, Xtr, Xte, ytr, yte, n_perms) -> dict:
    rng = np.random.default_rng(seed_seq)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(Xtr_s, ytr)
    obs_r2 = float(model.score(Xte_s, yte))
    obs_r = float(np.corrcoef(model.predict(Xte_s), yte)[0, 1])
    nulls = np.empty(n_perms)
    for i in range(n_perms):
        m = RidgeCV(alphas=RIDGE_ALPHAS)
        m.fit(Xtr_s, rng.permutation(ytr))
        nulls[i] = m.score(Xte_s, yte)
    return {
        "val_R2": obs_r2,
        "val_pearson_r": obs_r,
        "alpha_selected": float(model.alpha_),
        "shuffle_R2": band_stats(obs_r2, nulls),
    }


# ------------------------------------------- T4 r0-c protocol fidelity

def load_r0c():
    path = REPO / "experiments" / "r0c_probes" / "probes.py"
    spec = importlib.util.spec_from_file_location("r0c_probes_retro", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r0c_probes_retro"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_t4_arm(r0c, z, arm_index, freeze_dir, feats) -> dict:
    ev = Path(freeze_dir) / "evidence"
    train_items = json.loads(
        (ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
    val_items = json.loads(
        (ev / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]

    def build(fs_name):
        if fs_name == "apical_only":
            tr = z["train_boundary_apical"][:, arm_index, :]
            va = z["val_boundary_apical"][:, arm_index, :]
        elif fs_name == "voltage_plus_apical":
            tr = np.concatenate(
                [z["train_boundary_voltage"][:, arm_index, :],
                 z["train_boundary_apical"][:, arm_index, :]], axis=1)
            va = np.concatenate(
                [z["val_boundary_voltage"][:, arm_index, :],
                 z["val_boundary_apical"][:, arm_index, :]], axis=1)
        elif fs_name == "raw_voltage_spikes":
            tr = np.concatenate(
                [z["train_boundary_voltage"][:, arm_index, :],
                 z["train_boundary_spikes"][:, arm_index, :]], axis=1)
            va = np.concatenate(
                [z["val_boundary_voltage"][:, arm_index, :],
                 z["val_boundary_spikes"][:, arm_index, :]], axis=1)
        else:
            raise ValueError(fs_name)
        return r0c.standardize(tr.astype(np.float64), va.astype(np.float64))

    classes = r0c.answer_classes(train_items)
    class_names = sorted(classes, key=classes.get)
    Y_train = np.stack(
        [r0c._soft_targets(r0c.words(it["answer"]), classes)
         for it in train_items])
    scorer = r0c.SpanScorer(val_items, r0c.SPAN_PREDICTIONS)
    prior_ranking = r0c.majority_ranking(train_items, classes)
    controls = {"majority_prior_within_answer_vocab": scorer.score(
        [prior_ranking] * len(val_items))}

    results = {}
    for fs_name in feats:
        Xtr_s, Xva_s = build(fs_name)
        d = Xtr_s.shape[1]
        if d > 4096:  # mirror worker cap (never triggers here)
            rng_cols = np.random.default_rng(r0c.PROBE_SEED)
            cols = np.sort(rng_cols.choice(d, size=4096, replace=False))
            Xtr_s, Xva_s = Xtr_s[:, cols], Xva_s[:, cols]
        W, b = r0c.train_softmax_logistic(Xtr_s, Y_train)
        logits = Xva_s @ W + b
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        logistic = scorer.score(
            [[class_names[j] for j in np.argsort(-row)[:64]] for row in probs])

        rng_lab = np.random.default_rng(r0c.CONTROL_SEED)
        perm = rng_lab.permutation(len(train_items))
        Ws, bs = r0c.train_softmax_logistic(Xtr_s, Y_train[perm])
        logits_s = Xva_s @ Ws + bs
        probs_s = np.exp(logits_s - logits_s.max(axis=1, keepdims=True))
        probs_s /= probs_s.sum(axis=1, keepdims=True)
        shuf = scorer.score(
            [[class_names[j] for j in np.argsort(-row)[:64]]
             for row in probs_s])

        results[fs_name] = {
            "feature_dim_used": int(min(d, 4096)),
            "logistic_span_accuracy": logistic["accuracy"],
            "logistic_correct": logistic["correct"],
            "shuffled_label_control_span_accuracy": shuf["accuracy"],
            "shuffled_label_control_correct": shuf["correct"],
        }
    results["controls"] = controls
    return results


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--freeze-dir",
                    default=str(Path(os.environ.get("LOCALAPPDATA",
                                                    str(Path.home() / "AppData" / "Local")))
                                  / "Temp" / "hb_bible_freeze"))
    ap.add_argument("--sections", default="integrity,t1,sens,t2,t3,t4")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes; 1 (default) = serial.")
    args = ap.parse_args()
    sections = set(args.sections.split(","))
    npz_path = Path(args.npz)
    t_wall0 = time.perf_counter()

    # resume semantics: keep sections already computed by earlier invocations
    if RESULTS_PATH.exists():
        try:
            results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            log(f"resuming into existing {RESULTS_PATH.name} "
                f"(keys: {sorted(results.keys())})")
        except Exception:
            results = {}
    else:
        results = {}

    meta = {
        "schema_version": SCHEMA_VERSION,
        "probe": "P-A step -1 retro-probe (continuous decision-point voltages)",
        "npz_path": str(npz_path),
        "manifest": str(MANIFEST),
        "fixed_C": FIXED_C,
        "ridge_alphas": [float(a) for a in RIDGE_ALPHAS],
        "n_perms": {"family": N_PERMS_FAMILY, "speaker": N_PERMS_SPEAKER,
                    "ridge": N_PERMS_RIDGE},
        "master_seed": MASTER_SEED,
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
    }
    results.update(meta)

    z = np.load(npz_path)
    if "integrity" in sections:
        results["integrity"] = verify_npz(npz_path)
        tgt_probe = load_targets()
        results["targets_summary"] = {
            "train_n": int(len(tgt_probe["y_tr_family"])),
            "val_n": int(len(tgt_probe["y_va_family"])),
            "family_counts_train": dict(
                zip(FAMILIES, np.bincount(tgt_probe["y_tr_family"],
                                          minlength=3).tolist())),
            "family_counts_val": dict(
                zip(FAMILIES, np.bincount(tgt_probe["y_va_family"],
                                          minlength=3).tolist())),
            "who_said_train": tgt_probe["ws_n_train"],
            "who_said_val": tgt_probe["ws_n_val"],
            "speaker_pooled_counts": tgt_probe["speaker_pooled_counts"],
            "speaker_named_classes": tgt_probe["speaker_class_names"],
        }
        json.dump(results, open(RESULTS_PATH, "w"), indent=1)

    tg = load_targets()
    arms = tg["arm_names"]
    ss = np.random.SeedSequence(MASTER_SEED)

    if "t1" in sections or "sens" in sections:
        children = ss.spawn(64)  # deterministic budget of child streams
    ref_idx = arms.index("A100_0.25_0.6")

    if "t1" in sections:
        log("T1 family3: launching 18 arms x 3 feature sets")
        jobs = []
        meta = []
        ci = 0
        for ai in range(18):
            fsets = feature_sets(z, ai)
            for fs in ("apical", "concat", "voltage"):
                Xtr, Xva = fsets[fs]
                jobs.append(delayed(run_family_config)(
                    children[ci], Xtr, Xva, tg["y_tr_family"],
                    tg["y_va_family"], N_PERMS_FAMILY))
                meta.append((ai, fs))
                ci += 1
        t1_out = Parallel(n_jobs=args.jobs, verbose=0)(jobs)
        t1 = {}
        for (ai, fs), cfg in zip(meta, t1_out):
            t1.setdefault(fs, {})[arms[ai]] = cfg
        results["T1_family3_strict_split"] = t1
        json.dump(results, open(RESULTS_PATH, "w"), indent=1)
        log("T1 done")

    if "sens" in sections:
        fsets = feature_sets(z, ref_idx)
        sens = {}
        for fs in ("apical", "concat", "voltage"):
            Xtr, Xva = fsets[fs]
            sens[fs] = sensitivity_grid(
                Xtr, Xva, tg["y_tr_family"], tg["y_va_family"])
        results["sensitivity_reference_arm_A100_0.25_0.6"] = {
            "note": "true labels only; C grid + train-inner-CV curve",
            "grid": sens,
        }
        json.dump(results, open(RESULTS_PATH, "w"), indent=1)
        log("sensitivity done")

    if "t2" in sections:
        children2 = np.random.SeedSequence(MASTER_SEED + 1000).spawn(72)

        def pooled_feats(ai, fs):
            fsets = feature_sets(z, ai)
            tr, va = fsets[fs]
            return np.concatenate([tr[:tg["ws_n_train"]],
                                   va[:tg["ws_n_val"]]], axis=0)

        jobs = []
        meta2 = []
        ci = 0
        for ai in range(18):
            for fs in ("apical", "concat"):
                X_pool = pooled_feats(ai, fs)
                jobs.append(delayed(run_pooled_cv_config)(
                    children2[ci], X_pool, tg["y_pool_speaker"],
                    N_PERMS_SPEAKER, int(len(np.unique(tg["y_pool_speaker"])))))
                meta2.append((ai, fs, "speaker_multiclass"))
                ci += 1
                jobs.append(delayed(run_pooled_cv_config)(
                    children2[ci], X_pool, tg["y_moses"],
                    N_PERMS_SPEAKER, int(len(np.unique(tg["y_moses"])))))
                meta2.append((ai, fs, "moses_binary"))
                ci += 1
        t2_out = Parallel(n_jobs=args.jobs, verbose=0)(jobs)
        t2 = {}
        for (ai, fs, tn), cfg in zip(meta2, t2_out):
            t2.setdefault(tn, {}).setdefault(fs, {})[arms[ai]] = cfg
        results["T2_speaker_pooledCV"] = t2
        json.dump(results, open(RESULTS_PATH, "w"), indent=1)
        log("T2 done")

    if "t3" in sections:
        children3 = np.random.SeedSequence(MASTER_SEED + 2000).spawn(36)
        jobs = []
        meta3 = []
        ci = 0
        for ai in range(18):
            fsets = feature_sets(z, ai)
            for fs in ("apical", "concat"):
                Xtr, Xva = fsets[fs]
                jobs.append(delayed(run_ridge_config)(
                    children3[ci], Xtr[:tg["ws_n_train"]],
                    Xva[:tg["ws_n_val"]], tg["cd_tr"], tg["cd_va"],
                    N_PERMS_RIDGE))
                meta3.append((ai, fs))
                ci += 1
        t3_out = Parallel(n_jobs=args.jobs, verbose=0)(jobs)
        t3 = {}
        for (ai, fs), cfg in zip(meta3, t3_out):
            t3.setdefault(fs, {})[arms[ai]] = cfg
        results["T3_chapter_distance_ridge_strict_split"] = t3
        json.dump(results, open(RESULTS_PATH, "w"), indent=1)
        log("T3 done")

    if "t4" in sections:
        r0c = load_r0c()
        log(f"T4: r0c module loaded, SPAN_PREDICTIONS={r0c.SPAN_PREDICTIONS}")
        t4 = {}
        for ai in range(18):
            t4[arms[ai]] = run_t4_arm(
                r0c, z, ai, args.freeze_dir,
                ("apical_only", "voltage_plus_apical"))
            log(f"T4 arm {arms[ai]} done")
        # harness-fidelity check: reproduce ONE committed G-A3 cell
        fid = run_t4_arm(r0c, z, ref_idx, args.freeze_dir,
                         ("raw_voltage_spikes",))
        committed = json.loads(
            (HERE.parent / "probes_results.json").read_text(encoding="utf-8")
        )["GA3_per_arm"]["A100_0.25_0.6"]["raw_voltage_spikes"]["logistic"]
        results["T4_harness_fidelity_check"] = {
            "note": ("re-run of a committed G-A3 cell through this script's "
                     "r0c invocation; must equal the committed number"),
            "reproduced_span_accuracy":
                fid["raw_voltage_spikes"]["logistic_span_accuracy"],
            "committed_span_accuracy": committed["accuracy"],
            "exact_match": bool(
                fid["raw_voltage_spikes"]["logistic_span_accuracy"]
                == committed["accuracy"]),
        }
        results["T4_span_protocol_with_apical_features"] = t4
        json.dump(results, open(RESULTS_PATH, "w"), indent=1)
        log("T4 done")

    results["wall_seconds_total"] = round(time.perf_counter() - t_wall0, 1)
    json.dump(results, open(RESULTS_PATH, "w"), indent=1)
    log(f"ALL DONE in {results['wall_seconds_total']}s -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
