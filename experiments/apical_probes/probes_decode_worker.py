#!/usr/bin/env python3
"""One-arm G-A3 decode worker with checkpointing (APICAL-AND diagnostics).

Runs the r0-c decode pipeline verbatim for ONE arm of the cohort npz and
writes the result to decode_cache/arm_<name>.json. Invoked as a subprocess so
a killed worker loses at most one arm; the driver retries missing arms.

Usage: python probes_decode_worker.py --arm-index I --npz PATH --freeze-dir DIR --out PATH
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

R0C_PATH = REPO / "experiments" / "r0c_probes" / "probes.py"
RSPEC = importlib.util.spec_from_file_location("r0c_probes", R0C_PATH)
assert RSPEC is not None and RSPEC.loader is not None
R0C = importlib.util.module_from_spec(RSPEC)
sys.modules["r0c_probes"] = R0C
RSPEC.loader.exec_module(R0C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-index", type=int, required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--freeze-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    z = np.load(args.npz)
    ev = Path(args.freeze_dir) / "evidence"
    train_items = json.loads((ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
    val_items = json.loads((ev / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]

    a = args.arm_index

    def std(train: np.ndarray, other: np.ndarray):
        return R0C.standardize(
            train.astype(np.float64), other.astype(np.float64)
        )

    raw_tr = np.concatenate(
        [z["train_boundary_voltage"][:, a, :], z["train_boundary_spikes"][:, a, :]],
        axis=1,
    )
    raw_va = np.concatenate(
        [z["val_boundary_voltage"][:, a, :], z["val_boundary_spikes"][:, a, :]],
        axis=1,
    )
    sets = {
        "raw_voltage_spikes": std(raw_tr, raw_va),
        "traces_bank": std(z["train_boundary_traces"][:, a, :], z["val_boundary_traces"][:, a, :]),
    }

    classes = R0C.answer_classes(train_items)
    class_names = sorted(classes, key=classes.get)
    Y_train = np.stack(
        [R0C._soft_targets(R0C.words(it["answer"]), classes) for it in train_items]
    )
    scorer = R0C.SpanScorer(val_items, R0C.SPAN_PREDICTIONS)
    prior_ranking = R0C.majority_ranking(train_items, classes)
    controls = {
        "majority_prior_within_answer_vocab": scorer.score([prior_ranking] * len(val_items)),
    }
    rng = np.random.default_rng(R0C.CONTROL_SEED)
    random_rankings = []
    for _ in val_items:
        order = class_names[:]
        rng.shuffle(order)
        random_rankings.append(order)
    controls["random_ranking_seed11"] = scorer.score(random_rankings)

    results: dict[str, dict] = {}
    for fs_name, (Xtr, Xva) in sets.items():
        t0 = time.perf_counter()
        d = Xtr.shape[1]
        max_d = 4096
        if d > max_d:
            rng_cols = np.random.default_rng(R0C.PROBE_SEED)
            cols = np.sort(rng_cols.choice(d, size=max_d, replace=False))
            Xtr_s, Xva_s = Xtr[:, cols], Xva[:, cols]
        else:
            Xtr_s, Xva_s = Xtr, Xva

        W, b = R0C.train_softmax_logistic(Xtr_s, Y_train)
        logits = Xva_s @ W + b
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        logistic_score = scorer.score(
            [[class_names[j] for j in np.argsort(-row)[:64]] for row in probs]
        )

        model = R0C.train_mlp(Xtr_s, Y_train)
        probs_mlp = R0C.mlp_probs(model, Xva_s)
        mlp_score = scorer.score(
            [[class_names[j] for j in np.argsort(-row)[:64]] for row in probs_mlp]
        )

        rng_lab = np.random.default_rng(R0C.CONTROL_SEED)
        perm = rng_lab.permutation(len(train_items))
        model_shuf = R0C.train_mlp(Xtr_s, Y_train[perm])
        probs_shuf = R0C.mlp_probs(model_shuf, Xva_s)
        shuf_score = scorer.score(
            [[class_names[j] for j in np.argsort(-row)[:64]] for row in probs_shuf]
        )

        results[fs_name] = {
            "feature_dim_used": int(min(d, max_d)),
            "logistic": logistic_score,
            "mlp": mlp_score,
            "mlp_shuffled_label_control": shuf_score,
            "wall_seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"[worker arm {a}] {fs_name} done in {results[fs_name]['wall_seconds']}s",
              flush=True)

    results["controls"] = controls
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(results, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(out)
    print(f"[worker arm {a}] wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
