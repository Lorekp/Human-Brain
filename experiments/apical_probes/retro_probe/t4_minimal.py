#!/usr/bin/env python3
"""T4-minimal: reference-arm apical_only span decode (budget-capped).

The full T4 protocol (~7-15 min PER softmax fit on this contended box) was
cut to the single most informative cell: reference arm A100_0.25_0.6,
feature set apical_only, TRUE labels only. Comparators come from the
committed G-A3 artifacts (probes_results.json): per-arm logistic decodes,
shuffled-label MLP controls, and the frozen majority-prior control.
Disclosed in FINDINGS as a budget cut.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

from retro_probe import DEFAULT_NPZ, load_r0c

OUT = HERE / "t4_minimal_results.json"
FREEZE_DIR = (Path(os.environ.get("LOCALAPPDATA",
                                  str(Path.home() / "AppData" / "Local")))
              / "Temp" / "hb_bible_freeze")


def log(m):
    print(f"[{time.strftime('%H:%M:%Z', time.gmtime())}] {m}", flush=True)


def main():
    man = json.loads((HERE.parent / "logged_states_apical_cohort_manifest.json")
                     .read_text(encoding="utf-8"))
    ref = man["arm_names"].index("A100_0.25_0.6")
    z = np.load(DEFAULT_NPZ)
    r0c = load_r0c()

    ev = Path(FREEZE_DIR) / "evidence"
    train_items = json.loads(
        (ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
    val_items = json.loads(
        (ev / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]
    classes = r0c.answer_classes(train_items)
    class_names = sorted(classes, key=classes.get)
    Y_train = np.stack([r0c._soft_targets(r0c.words(it["answer"]), classes)
                        for it in train_items])
    scorer = r0c.SpanScorer(val_items, r0c.SPAN_PREDICTIONS)

    tr = z["train_boundary_apical"][:, ref, :].astype(np.float64)
    va = z["val_boundary_apical"][:, ref, :].astype(np.float64)
    Xtr_s, Xva_s = r0c.standardize(tr, va)
    log(f"training softmax logistic on {Xtr_s.shape} -> {Y_train.shape[1]} classes")
    t0 = time.perf_counter()
    W, b = r0c.train_softmax_logistic(Xtr_s, Y_train)
    log(f"fit done in {time.perf_counter()-t0:.0f}s")
    logits = Xva_s @ W + b
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    score = scorer.score([[class_names[j] for j in np.argsort(-row)[:64]]
                          for row in probs])

    committed = json.loads(
        (HERE.parent / "probes_results.json").read_text(encoding="utf-8"))
    ga3 = committed["GA3_per_arm"]["A100_0.25_0.6"]
    out = {
        "schema_version": "humanbrain/retro-probe-t4min/v1",
        "note": ("single budget-capped cell: A100_0.25_0.6 x apical_only, "
                 "true labels; comparators from committed G-A3 artifact"),
        "apical_only_span_accuracy": score["accuracy"],
        "apical_only_correct": score["correct"],
        "committed_reference_arm": {
            fs: {"logistic_accuracy": ga3[fs]["logistic"]["accuracy"],
                 "mlp_accuracy": ga3[fs]["mlp"]["accuracy"],
                 "mlp_shuffled_label_control":
                     ga3[fs]["mlp_shuffled_label_control"]}
            for fs in ("raw_voltage_spikes", "traces_bank")},
        "committed_majority_prior": committed[
            "majority_prior_within_answer_vocab"],
        "frozen_GA3_threshold": committed["gates"]["GA3_threshold_span_accuracy"],
    }
    json.dump(out, open(OUT, "w"), indent=1)
    log(f"apical_only span accuracy = {score['accuracy']!r} "
        f"({score['correct']}/639)")
    log("done -> " + str(OUT))


if __name__ == "__main__":
    main()
