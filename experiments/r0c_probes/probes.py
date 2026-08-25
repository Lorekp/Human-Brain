#!/usr/bin/env python3
"""R0-c Thread-1 D2-4 offline probes P1/P2/P3 + mechanical D5 checkpoint.

Consumes experiments/r0c_probes/logged_states.npz (+ manifest, + freeze QA
manifests) and runs the three preregistered probes from
docs/research/r0c-strategy-decision.md (Thread 1):

P1 ceiling   - multinomial logistic regression + 2-layer MLP (numpy/scipy,
               deterministic seeds) trained on logged RAW decision-point
               states (voltage || spike-count snapshots; traces-space variant)
               to map boundary state -> answer-word distribution; evaluated by
               VAL teacher-forced span accuracy over the same word stream the
               qa_baselines comparators used (639 predictions).
               Decision rule (verbatim from the strategy doc):
                 MLP < unigram 4.54%   => reservoir-state route dead for QA;
                                          keys fall back to input-assembly space
                 MLP >= trigram 16.28% => escalate (d) as constraint decision

P2 memory    - SDM-style store/query: binarize the answer:-boundary state as
               a key, store TRAIN key -> answer-token-set, query VAL by
               thresholded overlap; sweep radius x top-k.
               PASS gate (all four):
                 a) span-acc > unigram 4.54% at >= 2 sigma binomial
                    (H0 p = 29/639, n = 639 -> threshold ~= 6.19%);
                 b) distinct predicted answers >= 3;
                 c) shuffled-key control <= unigram + 2 sigma (anti-leakage);
                 d) (reported) full sweep table, best config selected ON VAL
                    per the strategy's explicit sweep instruction.

P3 attribution - who-said-X accuracy by chapter-distance bucket (VAL + TRAIN),
               using the best P2 configuration; localizes where context
               identity dies if the route is negative.

Decoder note (pinned here because both probe families emit RANKED candidate
sets from ONE boundary state): ranked-consumption decoding -- prediction at
answer-word position j is the highest-ranked candidate not yet consumed in
that item; a predicted word is consumed when emitted. Scoring bookkeeping
(regex wordization, casefold, per-position match counting, 639-position VAL
denominator) mirrors tools/qa_baselines.py answer_span_eval exactly.

Outputs: probes_results.json + FINDINGS.md in experiments/r0c_probes/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)

# Frozen comparator anchors (experiments/qa_baselines_bible_v1/*.json, verbatim)
UNIGRAM_SPAN_ACC = 0.04538341158059468  # 29/639
TRIGRAM_SPAN_ACC = 0.162754303599374  # 104/639
SPAN_PREDICTIONS = 639
GUARD_DISTINCT_MIN = 3
CONTROL_SEED = 11
PROBE_SEED = 7

WORD_RE = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


# ------------------------------------------------------------------ scoring


class SpanScorer:
    """qa_baselines-shaped span scoring with ranked-consumption decoding."""

    def __init__(
        self,
        val_items: list[dict],
        expected_predictions: int = SPAN_PREDICTIONS,
    ):
        self.items = val_items
        self.targets: list[list[str]] = []
        for item in val_items:
            prompt_w = words(item["serialized_prompt_answer"].split("\nA:")[0])
            del prompt_w  # prompt conditions the logged state, not the scorer
            self.targets.append(words(item["answer"]))
        self.predictions = sum(len(t) for t in self.targets)
        if expected_predictions and self.predictions != expected_predictions:
            raise ValueError(
                f"VAL stream has {self.predictions} answer-word positions, "
                f"expected {expected_predictions}"
            )

    def score(self, rankings: list[list[str]]) -> dict:
        correct = 0
        distinct_answers = set()
        per_task: list[dict] = []
        for ranking, targets in zip(rankings, self.targets):
            used: set[str] = set()
            task_correct = 0
            for target in targets:
                prediction = None
                for candidate in ranking:
                    if candidate not in used:
                        prediction = candidate
                        break
                if prediction is None:
                    prediction = ""
                used.add(prediction)
                distinct_answers.add(prediction)
                correct += int(prediction == target)
                task_correct += int(prediction == target)
            per_task.append({"targets": targets, "correct": task_correct})
        return {
            "predictions": self.predictions,
            "correct": correct,
            "accuracy": correct / self.predictions,
            "distinct_predicted_answers": len(distinct_answers),
            "per_task_correct": [row["correct"] for row in per_task],
        }


def binomial_two_sigma_threshold(p0: float, n: int) -> float:
    return p0 + 2.0 * float(np.sqrt(p0 * (1.0 - p0) / n))


# ------------------------------------------------------------------ P1 models


def _soft_targets(answer_words: list[str], classes: dict[str, int]) -> np.ndarray:
    row = np.zeros(len(classes), dtype=np.float64)
    for word in answer_words:
        if word in classes:
            row[classes[word]] += 1.0
    total = row.sum()
    return row / total if total else row


def train_softmax_logistic(
    X: np.ndarray, Y: np.ndarray, l2: float = 1e-3, iters: int = 1500, seed: int = PROBE_SEED
) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    c = Y.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.01, size=(d, c))
    b = np.zeros(c)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.05
    for t in range(1, iters + 1):
        logits = X @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        gW = X.T @ G + l2 * W
        gb = G.sum(axis=0)
        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * gW * gW
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb * gb
        c1 = 1 - beta1 ** t
        c2 = 1 - beta2 ** t
        W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
        b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        if t % 500 == 0:
            lr *= 0.5
    return W, b


def train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    hidden: int = 256,
    l2: float = 1e-4,
    epochs: int = 400,
    seed: int = PROBE_SEED,
) -> dict:
    n, d = X.shape
    c = Y.shape[1]
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, c))
    b2 = np.zeros(c)
    params = [W1, b1, W2, b2]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 2e-3
    t = 0
    for epoch in range(epochs):
        h_pre = X @ W1 + b1
        h = np.maximum(h_pre, 0.0)
        logits = h @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        G2 = (P - Y) / n
        gW2 = h.T @ G2 + l2 * W2
        gb2 = G2.sum(axis=0)
        Gh = (G2 @ W2.T) * (h_pre > 0.0)
        gW1 = X.T @ Gh + l2 * W1
        gb1 = Gh.sum(axis=0)
        grads = [gW1, gb1, gW2, gb2]
        t += 1
        for param, grad, m, v in zip(params, grads, ms, vs):
            m *= beta1
            m += (1 - beta1) * grad
            v *= beta2
            v += (1 - beta2) * grad * grad
            param -= lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)
        if (epoch + 1) % 100 == 0:
            lr *= 0.7
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def mlp_probs(model: dict, X: np.ndarray) -> np.ndarray:
    h = np.maximum(X @ model["W1"] + model["b1"], 0.0)
    logits = h @ model["W2"] + model["b2"]
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return P


# ------------------------------------------------------------------ data


class ProbeData:
    def __init__(self, npz_path: Path, freeze_dir: Path):
        self.z = np.load(npz_path)
        ev = freeze_dir / "evidence"
        self.train_items = json.loads(
            (ev / "qa_manifest_train.json").read_text(encoding="utf-8")
        )["items"]
        self.val_items = json.loads(
            (ev / "qa_manifest_val.json").read_text(encoding="utf-8")
        )["items"]
        # Slice manifests down if the npz was produced with a debug item cap.
        # The harness manifest records the exact per-split item counts.
        manifest_path = npz_path.parent / "logged_states_manifest.json"
        n_train_logged = len(self.train_items)
        n_val_logged = len(self.val_items)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first_arm = manifest["arms"][0]
            n_train_logged = int(first_arm.get("items_train", n_train_logged))
            n_val_logged = int(first_arm.get("items_val", n_val_logged))
        else:
            any_voltage = next(
                name
                for name in sorted(self.z.files)
                if name.endswith("_boundary_voltage")
            )
            total_logged = int(self.z[any_voltage].shape[0])
            n_val_logged = max(0, total_logged - n_train_logged)
        self.train_items = self.train_items[:n_train_logged]
        self.val_items = self.val_items[:n_val_logged]
        self.n_train = len(self.train_items)
        self.n_val = len(self.val_items)

    def _slice(self, split: str) -> tuple[int, int]:
        if split == "val":
            return self.n_train, self.n_val
        return 0, self.n_train

    def arms(self) -> list[str]:
        prefixes = {name.rsplit("_n", 1)[0] for name in self.z.files}
        return sorted({p.split("arm_", 1)[1] for p in prefixes})

    def ns(self, arm: str) -> list[int]:
        return sorted(
            {
                int(name.rsplit("_n", 1)[1].split("_", 1)[0])
                for name in self.z.files
                if name.startswith(f"arm_{arm}_n")
            }
        )

    def prefix(self, arm: str, n: int) -> str:
        return f"arm_{arm}_n{n}"

    # ---- feature matrices -------------------------------------------------
    def raw_state_matrix(self, arm: str, n: int, split: str) -> np.ndarray:
        """voltage || spike-counts raw decision-point snapshots."""
        p = self.prefix(arm, n)
        offset, count = self._slice(split)
        voltage = self.z[f"{p}_boundary_voltage"][offset : offset + count]
        spikes = self.z[f"{p}_boundary_spikes"][offset : offset + count]
        return np.concatenate([voltage, spikes.astype(np.float64)], axis=1)

    def traces_matrix(self, arm: str, n: int, split: str) -> np.ndarray:
        p = self.prefix(arm, n)
        offset, count = self._slice(split)
        return self.z[f"{p}_boundary_traces"][offset : offset + count]

    def asmrand_key_matrix(self, n: int, split: str) -> np.ndarray:
        """assembly_overlap-rand boundary feature vector (arm asm_rand only)."""
        p = self.prefix("asm_rand", n)
        offset, count = self._slice(split)
        return self.z[f"{p}_boundary_feature"][offset : offset + count]


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (train - mu) / sd, (other - mu) / sd


def answer_classes(train_items: list[dict]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for item in train_items:
        for word in words(item["answer"]):
            counter.setdefault(word, len(counter))
    return counter


def majority_ranking(train_items: list[dict], classes: dict[str, int]) -> list[str]:
    counts: dict[str, int] = dict.fromkeys(classes, 0)
    for item in train_items:
        for word in words(item["answer"]):
            counts[word] += 1
    return sorted(counts, key=lambda w: (-counts[w], w))


# ------------------------------------------------------------------ probes


def run_p1(data: ProbeData, span_expected: int = SPAN_PREDICTIONS) -> dict:
    results: dict = {}
    classes = answer_classes(data.train_items)
    class_names = sorted(classes, key=classes.get)
    Y_train = np.stack(
        [_soft_targets(words(it["answer"]), classes) for it in data.train_items]
    )
    scorer = SpanScorer(data.val_items, span_expected)
    prior_ranking = majority_ranking(data.train_items, classes)

    controls = {
        "majority_prior_within_answer_vocab": scorer.score([prior_ranking] * len(data.val_items)),
        "random_ranking_seed11": None,
    }
    rng = np.random.default_rng(CONTROL_SEED)
    random_pool = class_names[:]
    random_rankings = []
    for _ in data.val_items:
        order = random_pool[:]
        rng.shuffle(order)
        random_rankings.append(order)
    controls["random_ranking_seed11"] = scorer.score(random_rankings)

    for arm in data.arms():
        for n in data.ns(arm):
            feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            feature_sets["raw_voltage_spikes"] = (
                standardize(
                    data.raw_state_matrix(arm, n, "train"),
                    data.raw_state_matrix(arm, n, "val"),
                )
            )
            feature_sets["traces_bank"] = standardize(
                data.traces_matrix(arm, n, "train"),
                data.traces_matrix(arm, n, "val"),
            )
            if arm == "asm_rand":
                feature_sets["asm_overlap_rand_features"] = standardize(
                    data.asmrand_key_matrix(n, "train"),
                    data.asmrand_key_matrix(n, "val"),
                )
            arm_results = {}
            for fs_name, (Xtr, Xva) in feature_sets.items():
                # subsample features deterministically when huge (logistic is O(d*c))
                d = Xtr.shape[1]
                max_d = 4096
                cols = slice(None)
                if d > max_d:
                    rng_cols = np.random.default_rng(PROBE_SEED)
                    cols = np.sort(rng_cols.choice(d, size=max_d, replace=False))
                    Xtr, Xva = Xtr[:, cols], Xva[:, cols]
                W, b = train_softmax_logistic(Xtr, Y_train)
                logits = Xva @ W + b
                probs = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs /= probs.sum(axis=1, keepdims=True)
                rankings = [
                    [class_names[j] for j in np.argsort(-row)[:64]]
                    for row in probs
                ]
                logistic_score = scorer.score(rankings)

                model = train_mlp(Xtr, Y_train)
                probs_mlp = mlp_probs(model, Xva)
                rankings_mlp = [
                    [class_names[j] for j in np.argsort(-row)[:64]]
                    for row in probs_mlp
                ]
                mlp_score = scorer.score(rankings_mlp)

                # shuffled-label control (labels permuted across TRAIN items)
                rng_lab = np.random.default_rng(CONTROL_SEED)
                perm = rng_lab.permutation(len(data.train_items))
                model_shuf = train_mlp(Xtr, Y_train[perm])
                probs_shuf = mlp_probs(model_shuf, Xva)
                rankings_shuf = [
                    [class_names[j] for j in np.argsort(-row)[:64]]
                    for row in probs_shuf
                ]
                shuf_score = scorer.score(rankings_shuf)

                arm_results[fs_name] = {
                    "feature_dim_used": int(min(d, max_d)),
                    "logistic": logistic_score,
                    "mlp": mlp_score,
                    "mlp_shuffled_label_control": shuf_score,
                }
                log(
                    f"P1 arm={arm} N={n} {fs_name}: "
                    f"logistic={logistic_score['accuracy']:.4f} "
                    f"mlp={mlp_score['accuracy']:.4f}"
                )
            results.setdefault("arms", {})[f"{arm}_n{n}"] = arm_results
    results["controls"] = controls
    results["answer_vocab_size"] = len(class_names)
    return results


# ---------------------------------------------------------------------- P2


def binarize(matrix: np.ndarray, train_ref: np.ndarray) -> np.ndarray:
    medians = np.median(train_ref, axis=0, keepdims=True)
    return matrix > medians


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    """(N, D) bool -> (N, ceil(D/64) words) uint64, little-endian order."""
    pad = (-int(bits.shape[1])) % 64
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    packed = np.packbits(bits, axis=1)
    return np.ascontiguousarray(packed).view(np.uint64)


def overlap_all(query_bits: np.ndarray, store_bits: np.ndarray) -> np.ndarray:
    """Pairwise bit-agreement fraction between every query and every store key."""
    qp = _pack_bits(query_bits)
    sp = _pack_bits(store_bits)
    differ = qp[:, None, :] != sp[None, :, :]
    counts = np.bitwise_count(differ).sum(axis=2)
    dim = query_bits.shape[1]
    return 1.0 - counts.astype(np.float64) / float(dim)


def rank_from_overlaps(
    overlaps_row: np.ndarray,
    values: list[set[str]],
    radius: float | None,
    top_k: int | None,
) -> list[str]:
    candidates = np.argsort(-overlaps_row, kind="stable")  # ties -> lower index
    if radius is not None:
        candidates = candidates[overlaps_row[candidates] >= radius]
    if top_k is not None:
        candidates = candidates[:top_k]
    scores: dict[str, float] = {}
    for idx in candidates:
        weight = float(overlaps_row[idx])
        for word in values[int(idx)]:
            scores[word] = scores.get(word, 0.0) + weight
    ranking = sorted(scores, key=lambda w: (-scores[w], w))
    return ranking[:128]


def run_p2(data: ProbeData, span_expected: int = SPAN_PREDICTIONS) -> dict:
    results: dict = {"gate_threshold": {}, "arms": {}}
    p0 = UNIGRAM_SPAN_ACC
    n_pred = span_expected or SPAN_PREDICTIONS
    thr = binomial_two_sigma_threshold(p0, n_pred)
    results["gate_threshold"] = {
        "unigram_span_acc": p0,
        "binomial_sigma": float(np.sqrt(p0 * (1 - p0) / n_pred)),
        "two_sigma_accuracy_threshold": thr,
        "trigram_span_acc": TRIGRAM_SPAN_ACC,
        "note": (
            "task brief approximated the 2-sigma threshold as '~7%'; the exact "
            "preregistered z>=2 test on p0=29/639 gives "
            f"{thr:.4f}"
        ),
    }
    scorer = SpanScorer(data.val_items, span_expected)
    train_values = [set(words(it["answer"])) for it in data.train_items]
    shuffled_values = []
    rng = np.random.default_rng(CONTROL_SEED)
    perm = rng.permutation(len(train_values))
    shuffled_values = [train_values[i] for i in perm]

    radii = [None, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    topks = [1, 2, 4, 8, 16, None]

    for arm in data.arms():
        for n in data.ns(arm):
            if arm == "asm_rand":
                key_train = data.asmrand_key_matrix(n, "train")
                key_val = data.asmrand_key_matrix(n, "val")
                key_source = "asm_overlap_rand_boundary_feature"
            else:
                key_train = data.raw_state_matrix(arm, n, "train")
                key_val = data.raw_state_matrix(arm, n, "val")
                key_source = "raw_voltage_spikes"
            bits_train = binarize(key_train, key_train)
            bits_val = binarize(key_val, key_train)
            ov_val = overlap_all(bits_val, bits_train)  # (n_val, n_store)
            sweep = []
            best = None
            for radius in radii:
                for top_k in topks:
                    rankings = [
                        rank_from_overlaps(ov_val[i], train_values, radius, top_k)
                        for i in range(ov_val.shape[0])
                    ]
                    score = scorer.score(rankings)
                    # Shuffled-key control: identical keys/overlaps, VALUES
                    # permuted across stored items (seeded), same config.
                    control_rankings = [
                        rank_from_overlaps(ov_val[i], shuffled_values, radius, top_k)
                        for i in range(ov_val.shape[0])
                    ]
                    control = scorer.score(control_rankings)
                    passed = (
                        score["accuracy"] >= thr
                        and score["distinct_predicted_answers"] >= GUARD_DISTINCT_MIN
                        and control["accuracy"] <= thr
                    )
                    row = {
                        "radius": radius,
                        "top_k": top_k,
                        "span_acc": score["accuracy"],
                        "correct": score["correct"],
                        "distinct_predicted_answers": score["distinct_predicted_answers"],
                        "shuffled_key_control_acc": control["accuracy"],
                        "z_vs_unigram": float(
                            (score["accuracy"] - p0)
                            / np.sqrt(p0 * (1 - p0) / n_pred)
                        ),
                        "gate_pass": bool(passed),
                    }
                    sweep.append(row)
                    if best is None or score["accuracy"] > best["span_acc"]:
                        best = row
            log(
                f"P2 arm={arm} N={n}: best span_acc={best['span_acc']:.4f} "
                f"(r={best['radius']}, k={best['top_k']}, "
                f"control={best['shuffled_key_control_acc']:.4f}, "
                f"gate={'PASS' if best['gate_pass'] else 'FAIL'})"
            )
            results["arms"][f"{arm}_n{n}"] = {
                "key_source": key_source,
                "key_bits": int(bits_train.shape[1]),
                "store_items": len(train_values),
                "sweep": sweep,
                "best": best,
                "gate_pass": bool(best["gate_pass"]),
            }
    return results


# ---------------------------------------------------------------------- P3


def run_p3(data: ProbeData, p2: dict) -> dict:
    results: dict = {}
    train_values = [set(words(it["answer"])) for it in data.train_items]

    def buckets_for(items: list[dict]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            if item["family"] != "who_said":
                continue
            dist = (item.get("meta") or {}).get("chapter_distance")
            if dist is None:
                continue
            dist = int(dist)
            bucket = "<=1" if dist <= 1 else ("2-4" if dist <= 4 else ("5-8" if dist <= 8 else ">8"))
            out.setdefault(bucket, []).append(i)
        return out

    for arm in data.arms():
        for n in data.ns(arm):
            cfg = p2["arms"][f"{arm}_n{n}"]["best"]
            radius, top_k = cfg["radius"], cfg["top_k"]
            if arm == "asm_rand":
                key_train = data.asmrand_key_matrix(n, "train")
                key_full = data.asmrand_key_matrix(n, "val")
            else:
                key_train = data.raw_state_matrix(arm, n, "train")
                key_full = data.raw_state_matrix(arm, n, "val")
            bits_train = binarize(key_train, key_train)
            report = {}
            for split, items, eval_mat in (
                ("val", data.val_items, key_full),
                ("train", data.train_items, key_train),
            ):
                bits_eval = binarize(eval_mat, key_train)
                overlaps = overlap_all(bits_eval, bits_train)
                rankings = [
                    rank_from_overlaps(overlaps[i], train_values, radius, top_k)
                    for i in range(overlaps.shape[0])
                ]
                buckets = buckets_for(items)
                rows = {}
                for bucket, indices in sorted(buckets.items()):
                    hits = totals = 0
                    distances = []
                    for i in indices:
                        meta = items[i].get("meta") or {}
                        speaker = words(meta.get("speaker_token", ""))
                        distances.append(int(meta.get("chapter_distance", -1)))
                        decoded = []
                        used = set()
                        for target in words(items[i]["answer"]):
                            pred = next((c for c in rankings[i] if c not in used), "")
                            used.add(pred)
                            decoded.append(pred)
                        hit = int(bool(set(decoded) & set(speaker)))
                        hits += hit
                        totals += 1
                    rows[bucket] = {
                        "n": totals,
                        "speaker_hits": hits,
                        "accuracy": hits / totals if totals else None,
                        "distances": distances,
                    }
                pooled_n = sum(r["n"] for r in rows.values())
                pooled_hits = sum(r["speaker_hits"] for r in rows.values())
                report[split] = {
                    "buckets": rows,
                    "pooled": {
                        "n": pooled_n,
                        "speaker_hits": pooled_hits,
                        "accuracy": pooled_hits / pooled_n if pooled_n else None,
                    },
                }
            results[f"{arm}_n{n}"] = {
                "config": {"radius": radius, "top_k": top_k},
                **report,
            }
            log(f"P3 arm={arm} N={n}: VAL pooled speaker-hit acc="
                f"{report['val']['pooled']['accuracy']}")
    return results


# ------------------------------------------------------------------- D5


def d5_verdict(p1: dict, p2: dict) -> dict:
    mlp_best = -1.0
    mlp_best_where = None
    logistic_best = -1.0
    for key, arm_res in p1["arms"].items():
        for fs_name, res in arm_res.items():
            if res["mlp"]["accuracy"] >= mlp_best:
                mlp_best = res["mlp"]["accuracy"]
                mlp_best_where = f"{key}/{fs_name}"
            if res["logistic"]["accuracy"] >= logistic_best:
                logistic_best = res["logistic"]["accuracy"]
    p2_passes = [
        key for key, res in p2["arms"].items() if res["gate_pass"]
    ]
    if mlp_best >= TRIGRAM_SPAN_ACC:
        verdict = "ESCALATE_(d)"
        reason = (
            f"best MLP span-acc {mlp_best:.4f} >= trigram {TRIGRAM_SPAN_ACC:.4f}; "
            "modern learned readout becomes a constraint decision for the user"
        )
    elif p2_passes:
        verdict = "INTEGRATE"
        reason = (
            f"P2 PASS at {', '.join(p2_passes)} (span-acc > unigram at >=2sigma, "
            "distinct >= 3, shuffled-key control clean); proceed to D6-9 "
            "integration behind config flag"
        )
    else:
        verdict = "NEGATIVE-WITH-LOCALIZATION"
        relation = (
            "below-unigram (reservoir-state route dead for QA; keys fall back "
            "to input-assembly space)"
            if mlp_best < UNIGRAM_SPAN_ACC
            else "above-unigram but below-trigram and below-gate (sub-threshold signal)"
        )
        reason = f"best MLP span-acc {mlp_best:.4f} is {relation}; P2 no-pass"
    return {
        "verdict": verdict,
        "reason": reason,
        "best_mlp": {"accuracy": mlp_best, "where": mlp_best_where},
        "best_logistic": logistic_best,
        "unigram": UNIGRAM_SPAN_ACC,
        "trigram": TRIGRAM_SPAN_ACC,
        "two_sigma_threshold": binomial_two_sigma_threshold(
            UNIGRAM_SPAN_ACC, SPAN_PREDICTIONS
        ),
        "p2_passing_configs": p2_passes,
    }


# ------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--probes-dir", default=str(REPO / "experiments" / "r0c_probes"))
    parser.add_argument(
        "--span-predictions",
        type=int,
        default=SPAN_PREDICTIONS,
        help="expected VAL answer-word positions (639 for the full manifest; "
        "0 disables the check for smoke runs)",
    )
    args = parser.parse_args()

    probes_dir = Path(args.probes_dir)
    data = ProbeData(probes_dir / "logged_states.npz", Path(args.freeze_dir))

    log(f"loaded npz with {len(data.z.files)} arrays; arms={data.arms()}")
    t0 = time.perf_counter()
    p1 = run_p1(data, span_expected=args.span_predictions)
    log(f"P1 done in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    p2 = run_p2(data, span_expected=args.span_predictions)
    log(f"P2 done in {time.perf_counter()-t0:.1f}s")
    p3 = run_p3(data, p2)
    verdict = d5_verdict(p1, p2)

    results = {
        "schema_version": "humanbrain/r0c-probes/v1",
        "comparators": {
            "unigram_smoothed_span_acc": UNIGRAM_SPAN_ACC,
            "trigram_smoothed_span_acc": TRIGRAM_SPAN_ACC,
            "source": "experiments/qa_baselines_bible_v1/*.json",
        },
        "decoder_rule": "ranked-consumption (see module docstring)",
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "D5_decision_checkpoint": verdict,
    }
    out_json = probes_dir / "probes_results.json"
    out_json.write_bytes(canonical_json(results) + b"\n")
    log(f"wrote {out_json}")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
