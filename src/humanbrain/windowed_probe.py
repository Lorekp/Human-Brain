"""Windowed-decoder diagnostic tier (srlm deep-dive adopt row A4).

Week-1 U-STATE probing scored ONE state per item: the answer:-boundary
snapshot. That instrument cannot separate two failure hypotheses:

  H1 "state-empty":      the union state stream never carries answer-relevant
                         information anywhere in the recent history;
  H2 "decoder-limited":  information IS present in recent states but the
                         single-timestep readout lacks the capacity/context to
                         express it.

This module sharpens the instrument. Given logged state sequences from an
observation-sink replay (ustate canonical format; generic fallback for other
observation-sink npz layouts), it builds BOUNDED WINDOWS of recent states per
decision point -- the last-k logged states leading up to the boundary,
concatenated -- for a depth sweep (default k in {1, 4, 16, 64}), and scores
them against decision variables with two decoders:

  logistic  - the UNCHANGED r0c softmax trainer (same seeds/schedule), so the
              k=1 anchor is directly comparable to the archived P1 numbers;
              feature dimension capped at ``max_logistic_d`` columns by
              deterministic subsample when a window exceeds it (r0c precedent).
  ridge     - EXACT linear-kernel kernel-ridge regression (closed form through
              the Gram matrix), which consumes the FULL concatenated window at
              any depth with no dimension cap -- the capacity-unlimited arm of
              the diagnostic.

Decision variables:
  span_decode      - the preregistered G-U2 task: ranked-consumption
                     answer-word decoding over the 639 VAL positions, compared
                     verbatim against frozen comparators (unigram 29/639 =
                     4.538%, majority prior 26/639, trigram 104/639, G-U2
                     two-sigma threshold 6.185%).
  speaker_identity - secondary who-said-X readout on the manifest subset that
                     carries ``meta.speaker_token`` (TRAIN 60 / VAL 20),
                     top-1 classification vs the majority-speaker baseline.

Controls: shuffled-label controls (TRAIN labels permuted, seed 11 -- the r0c
control seed) for every decoder x depth x task; majority-prior and
random-ranking span controls recomputed from the manifests. Every accuracy is
reported with a Wilson 95% confidence interval.

Ridge lambda is selected ON VAL per configuration (the r0c/P2 precedent of
VAL-side config selection); this is conservative for the question asked: if
even a VAL-selected lambda cannot beat the bars, no honest tuning would.

Window rule (pinned): for item i occupying global teacher-forced tokens
[start_i, end_i) with boundary b_i = start_i + prompt_tokens - 1, the eligible
history is the logged stride snapshots with start_i <= t < b_i (states observed
while reading THIS item's prompt; answer-region snapshots of earlier items are
never admitted, and windows never cross the per-item encoder reset). A depth-k
window is the LAST min(k-1, available) eligible snapshots in chronological
order followed by the boundary snapshot itself; shortfalls are padded at the
FRONT by repeating the earliest available snapshot (zero-padding would inject
fake certainty; hold-first preserves scale). k=1 degenerates to the boundary
snapshot alone -- the archived single-timestep instrument.

Archive integrity: callers pass the COMMITTED manifest json; every npz array's
sha256 (canonical bytes, matching log_states_union.py) is verified before any
scoring, and the reconstructed per-item segmentation is verified twice --
against the committed curriculum digest and against the archived
``*_boundary_idx`` arrays -- so a retro-probe here is cryptographically pinned
to the exact states week 1 scored, even though the binary npz files were
local-only and had to be regenerated deterministically.

CPU-only: numpy dense linear algebra throughout; no torch, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- comparators
# Frozen anchors (experiments/qa_baselines_bible_v1 + ustate FINDINGS, verbatim)
UNIGRAM_SPAN_ACC = 0.04538341158059468  # 29/639
TRIGRAM_SPAN_ACC = 0.162754303599374  # 104/639
SPAN_PREDICTIONS = 639
GU2_TWO_SIGMA_THRESHOLD = 0.061851478566709396
# Week-1 frozen scorer anchors (ustate_probes FINDINGS, verbatim): the ranked-
# consumption SpanScorer must reproduce these EXACTLY for this instrument to
# be the same measuring device as the archived P1/P2 numbers.
FROZEN_MAJORITY_PRIOR_CORRECT = 26  # 0.040689 of 639
FROZEN_RANDOM_RANKING_CORRECT = 1  # seed 11, 0.001565 of 639
PROBE_SEED = 7
CONTROL_SEED = 11
WILSON_Z = 1.959963984540054  # two-sided 95%
DEFAULT_DEPTHS = (1, 4, 16, 64)
DEFAULT_FAMILIES = ("union", "count_union")
DEFAULT_RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0)
MAX_LOGISTIC_D = 4096
LOGISTIC_ITERS = 1500

SCHEMA_VERSION = "humanbrain/windowed-probe/v1"

WORD_RE = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wilson_interval(correct: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = correct / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def packed_result(correct: int, n: int, **extra) -> dict:
    lo, hi = wilson_interval(correct, n)
    out = {
        "correct": int(correct),
        "predictions": int(n),
        "accuracy": correct / n if n else 0.0,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
    }
    out.update(extra)
    return out


# ------------------------------------------------------- r0c-mirror machinery


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


def soft_targets(answer_words: list[str], classes: dict[str, int]) -> np.ndarray:
    row = np.zeros(len(classes), dtype=np.float64)
    for word in answer_words:
        if word in classes:
            row[classes[word]] += 1.0
    total = row.sum()
    return row / total if total else row


class SpanScorer:
    """qa_baselines-shaped span scoring with ranked-consumption decoding.

    Verbatim mirror of experiments/r0c_probes/probes.py SpanScorer semantics:
    prediction at answer-word position j is the highest-ranked candidate not
    yet consumed in that item; a predicted word is consumed when emitted.
    """

    def __init__(self, val_items: list[dict], expected_predictions: int = SPAN_PREDICTIONS):
        self.items = val_items
        self.targets = [words(item["answer"]) for item in val_items]
        self.predictions = sum(len(t) for t in self.targets)
        if expected_predictions and self.predictions != expected_predictions:
            raise ValueError(
                f"VAL stream has {self.predictions} answer-word positions, "
                f"expected {expected_predictions}"
            )

    def score(self, rankings: list[list[str]]) -> dict:
        correct = 0
        distinct_answers: set[str] = set()
        per_task_correct: list[int] = []
        for ranking, targets in zip(rankings, self.targets):
            used: set[str] = set()
            task_correct = 0
            for target in targets:
                prediction = ""
                for candidate in ranking:
                    if candidate not in used:
                        prediction = candidate
                        break
                used.add(prediction)
                distinct_answers.add(prediction)
                hit = int(prediction == target)
                correct += hit
                task_correct += hit
            per_task_correct.append(task_correct)
        return {
            "predictions": self.predictions,
            "correct": correct,
            "accuracy": correct / self.predictions,
            "distinct_predicted_answers": len(distinct_answers),
            "per_task_correct": per_task_correct,
        }


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (train - mu) / sd, (other - mu) / sd


def subsample_columns(
    Xtr: np.ndarray, Xva: np.ndarray, max_d: int, seed: int = PROBE_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic column subsample when d > max_d (r0c precedent)."""
    d = Xtr.shape[1]
    if d <= max_d:
        return Xtr, Xva
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.choice(d, size=max_d, replace=False))
    return Xtr[:, cols], Xva[:, cols]


def train_softmax_logistic(
    X: np.ndarray,
    Y: np.ndarray,
    l2: float = 1e-3,
    iters: int = LOGISTIC_ITERS,
    seed: int = PROBE_SEED,
    decay_every: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """UNCHANGED r0c P1 softmax trainer (full-batch Adam).

    ``decay_every`` defaults to the r0c schedule (lr halves every 500 iters);
    the throttled-CPU sweep runs fewer iterations with a proportionally
    compressed decay cadence -- a disclosed budget deviation, applied
    identically to real fits and their shuffled-label controls.
    """
    n, d = X.shape
    c = Y.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.01, size=(d, c))
    b = np.zeros(c)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
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
        c1 = 1 - beta1**t
        c2 = 1 - beta2**t
        W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
        b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        if t % decay_every == 0:
            lr *= 0.5
    return W, b


def softmax_probs(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    logits = X @ W + b
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return P


def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float64)


def kernel_ridge_fit(K: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form dual weights: alpha = (K + lam*I)^-1 Y (exactly solves
    min ||Y - X w||^2 + lam ||w||^2 in function space; dimension-free)."""
    n = K.shape[0]
    return np.linalg.solve(K + lam * np.eye(n), Y)


def topk_rankings(scores: np.ndarray, class_names: list[str], k: int = 64) -> list[list[str]]:
    return [[class_names[j] for j in np.argsort(-row)[:k]] for row in scores]


# ------------------------------------------------------------ archive loading


@dataclass
class ArmStates:
    """One arm's logged observation-sink arrays."""

    arm: str
    n_bits: int
    token_ids: np.ndarray
    families: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def boundary_idx(self, family: str) -> np.ndarray:
        return self.families[family]["boundary_idx"]

    def stride_idx(self, family: str) -> np.ndarray:
        return self.families[family]["stride_idx"]


def _arm_prefixes(files: list[str]) -> dict[str, int]:
    """Map 'arm_<name>_n<bits>' prefix -> n_bits for ustate-canonical layouts."""
    prefixes: dict[str, int] = {}
    for name in files:
        m = re.match(r"^(arm_.+_n(\d+))_", name)
        if m:
            prefixes[m.group(1)] = int(m.group(2))
    return prefixes


def load_logged_states(npz_path: str | Path) -> dict[str, ArmStates]:
    """Load an observation-sink logged-states npz into per-arm state bundles.

    Canonical ustate layout: arrays keyed ``arm_<arm>_n<N>_token_ids``,
    ``arm_<arm>_n<N>_tokfeat__<family>_{stride_idx,stride_rows,boundary_idx,
    boundary_rows}`` plus ``_boundary_{voltage,spikes,feature,traces}``.
    Generic fallback: any other layout keeps every array in ``extras`` so
    downstream tooling can still discover sequence/boundary pairs by name.
    """
    z = np.load(npz_path)
    arms: dict[str, ArmStates] = {}
    for prefix, n_bits in sorted(_arm_prefixes(list(z.files)).items()):
        arm = prefix.split("_n", 1)[0][len("arm_") :]
        fams: dict[str, dict[str, np.ndarray]] = {}
        extras: dict[str, np.ndarray] = {}
        for name in z.files:
            if not name.startswith(prefix + "_"):
                continue
            tail = name[len(prefix) + 1 :]
            fm = re.match(r"^tokfeat__(.+?)_(stride_idx|stride_rows|boundary_idx|boundary_rows)$", tail)
            if fm:
                fams.setdefault(fm.group(1), {})[fm.group(2)] = z[name]
            else:
                extras[tail] = z[name]
        holder = ArmStates(
            arm=arm,
            n_bits=n_bits,
            token_ids=extras.pop("token_ids"),
            families=fams,
            extras=extras,
        )
        arms[arm] = holder
    if not arms:
        # generic observation-sink fallback: expose everything under "" and let
        # the caller interpret names; keeps this loader usable for apical /
        # race-pivot style sinks whose keying differs.
        arms[""] = ArmStates(arm="", n_bits=0, token_ids=np.array([]), extras={k: z[k] for k in z.files})
    return arms


def verify_archive(npz_path: str | Path, manifest_path: str | Path) -> dict:
    """Verify every npz array's sha256/shape/dtype against the committed manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    z = np.load(npz_path)
    checked, mismatches = 0, []
    for name, meta in sorted(manifest.get("npz_arrays", {}).items()):
        arr = z[name]
        h = sha256_bytes(np.ascontiguousarray(arr).tobytes())
        checked += 1
        if h != meta["sha256"] or list(arr.shape) != meta["shape"] or str(arr.dtype) != meta["dtype"]:
            mismatches.append(name)
    return {"checked": checked, "mismatches": mismatches, "pass": not mismatches}


# ----------------------------------------------------- replay reconstruction


def split_prompt(serialized: str) -> tuple[str, str]:
    marker = "\nA:"
    cut = serialized.rindex(marker) + len(marker)
    return serialized[:cut], serialized[cut:]


def reconstruct_replay_rows(freeze_dir: str | Path, encoding: str = "byte") -> list[dict]:
    """Deterministically rebuild the teacher-forced replay bookkeeping.

    Mirrors log_states_union.replay_items' counting WITHOUT running the encoder
    (pure tokenization), producing per-item start/boundary global token offsets
    in replay order (TRAIN items then VAL items). Verified downstream against
    the committed curriculum digest and archived boundary indices.
    """
    from humanbrain.concept_tokens import ConceptConfig, ConceptTokenizer
    from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

    ev = Path(freeze_dir) / "evidence"
    train_items = json.loads((ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
    val_items = json.loads((ev / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]
    train_texts = [it["serialized_prompt_answer"] for it in train_items]
    tokenizer = BytePhraseTokenizer.fit(train_texts, TokenizerConfig(max_phrases=0, min_frequency=3))
    if encoding == "concept":
        stack = ConceptTokenizer.fit(train_texts, base=tokenizer, config=ConceptConfig())
        encode_text = stack.encode
        token_bytes_of = stack.token_bytes
    else:
        encode_text = tokenizer.encode
        token_bytes_of = None

    rows: list[dict] = []
    offset = 0
    for split_name, items in (("train", train_items), ("val", val_items)):
        for position, item in enumerate(items):
            prompt_text, _rest = split_prompt(item["serialized_prompt_answer"])
            full_ids = [int(t) for t in encode_text(item["serialized_prompt_answer"])]
            if encoding == "byte":
                prompt_ids = [int(t) for t in encode_text(prompt_text)]
                if tuple(full_ids[: len(prompt_ids)]) != tuple(prompt_ids):
                    raise ValueError(f"{split_name}[{position}]: prompt is not a byte-token prefix")
                prompt_count = len(prompt_ids)
            else:
                prompt_byte_len = len(prompt_text.encode("utf-8"))
                acc = 0
                prompt_count = 0
                for tid in full_ids:
                    blen = len(token_bytes_of(tid))
                    if acc + blen <= prompt_byte_len:
                        acc += blen
                        prompt_count += 1
                    else:
                        break
                if not 0 < prompt_count < len(full_ids):
                    raise ValueError(f"{split_name}[{position}]: degenerate boundary cut")
            rows.append(
                {
                    "qa_id": item["qa_id"],
                    "family": item["family"],
                    "split": split_name,
                    "cohort_position": position,
                    "start_global": offset,
                    "boundary_global": offset + prompt_count - 1,
                    "prompt_tokens": prompt_count,
                    "total_tokens": len(full_ids),
                    "answer_text": item["answer"],
                    "speaker_token": (item.get("meta") or {}).get("speaker_token"),
                }
            )
            offset += len(full_ids)
    return rows


def curriculum_digest_of(rows: list[dict]) -> str:
    """Mirror of log_states_union.main's curriculum digest over replay rows."""
    payload = [
        [
            {key: row[key] for key in ("qa_id", "family", "prompt_tokens", "total_tokens")}
            for row in rows
            if row["split"] == split
        ]
        for split in ("train", "val")
    ]
    return sha256_bytes(canonical_json(payload))


# ----------------------------------------------------------- window features


def extract_windows(
    stride_idx: np.ndarray,
    stride_rows: np.ndarray,
    boundary_rows: np.ndarray,
    item_starts: np.ndarray,
    item_boundaries: np.ndarray,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-item depth-k window features.

    Returns (X [n_items, depth * dim], filled [n_items]) where filled counts
    REAL history snapshots admitted (0 .. depth-1; the rest are hold-first
    padding). Row layout: [oldest ... newest-history, boundary].
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    dim = boundary_rows.shape[1]
    order = np.argsort(stride_idx, kind="stable")
    s_idx = stride_idx[order]
    s_rows = stride_rows[order].astype(np.float32)
    n = len(item_boundaries)
    X = np.empty((n, depth * dim), dtype=np.float32)
    filled = np.zeros(n, dtype=np.int64)
    lo = np.searchsorted(s_idx, item_starts, side="left")
    hi = np.searchsorted(s_idx, item_boundaries, side="left")  # t < boundary
    for i in range(n):
        seg = s_rows[lo[i] : hi[i]]
        take = min(depth - 1, len(seg))
        filled[i] = take
        block = np.empty((depth, dim), dtype=np.float32)
        if take == 0:
            block[:, :] = boundary_rows[i].astype(np.float32)
        else:
            history = seg[-take:]
            pad = np.repeat(history[:1], depth - 1 - take, axis=0)
            block[: depth - 1 - take] = pad
            block[depth - 1 - take : depth - 1] = history
        block[depth - 1] = boundary_rows[i].astype(np.float32)
        X[i] = block.reshape(-1)
    return X, filled


# --------------------------------------------------------------- evaluation


@dataclass
class EvalConfig:
    depths: tuple[int, ...] = DEFAULT_DEPTHS
    families: tuple[str, ...] = DEFAULT_FAMILIES
    ridge_lambdas: tuple[float, ...] = DEFAULT_RIDGE_LAMBDAS
    logistic_iters: int = LOGISTIC_ITERS
    logistic_decay_every: int = 500
    max_logistic_d: int = MAX_LOGISTIC_D
    logistic_families: tuple[str, ...] = ("union",)
    probe_seed: int = PROBE_SEED
    control_seed: int = CONTROL_SEED
    ranking_depth: int = 64


def _score_rankings(scorer: SpanScorer, probs: np.ndarray, class_names: list[str], cfg: EvalConfig) -> dict:
    raw = scorer.score(topk_rankings(probs, class_names, cfg.ranking_depth))
    res = packed_result(raw["correct"], raw["predictions"])
    res["distinct_predicted_answers"] = raw["distinct_predicted_answers"]
    return res


def eval_span_features(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Ytr: np.ndarray,
    class_names: list[str],
    scorer: SpanScorer,
    cfg: EvalConfig,
    cap_logistic_d: bool = True,
    with_logistic: bool = True,
) -> dict:
    """Fit decoders (+ shuffled-label controls) on one feature matrix pair.

    ``with_logistic=False`` restricts the node to exact kernel ridge (used on
    secondary state families to fit the throttled-CPU wall-clock budget).
    """
    out: dict[str, dict] = {}
    rng_lab = np.random.default_rng(cfg.control_seed)
    perm = rng_lab.permutation(Xtr.shape[0])

    # ---- logistic (dimension-capped, r0c-identical trainer)
    if with_logistic:
        Ltr, Lva = (
            subsample_columns(Xtr, Xva, cfg.max_logistic_d, cfg.probe_seed)
            if cap_logistic_d
            else (Xtr, Xva)
        )
        W, b = train_softmax_logistic(
            Ltr,
            Ytr,
            iters=cfg.logistic_iters,
            seed=cfg.probe_seed,
            decay_every=cfg.logistic_decay_every,
        )
        out["logistic"] = _score_rankings(scorer, softmax_probs(W, b, Lva), class_names, cfg)
        Ws, bs = train_softmax_logistic(
            Ltr,
            Ytr[perm],
            iters=cfg.logistic_iters,
            seed=cfg.probe_seed,
            decay_every=cfg.logistic_decay_every,
        )
        shuf = _score_rankings(scorer, softmax_probs(Ws, bs, Lva), class_names, cfg)
        shuf.pop("distinct_predicted_answers", None)
        out["logistic_shuffled_label_control"] = shuf

    # ---- exact linear-kernel ridge over the FULL uncapped window
    K_tr = linear_kernel(Xtr, Xtr)
    K_va = linear_kernel(Xva, Xtr)
    best = None
    for lam in cfg.ridge_lambdas:
        alpha = kernel_ridge_fit(K_tr, Ytr, lam)
        res = _score_rankings(scorer, K_va @ alpha, class_names, cfg)
        if best is None or res["accuracy"] > best[1]["accuracy"]:
            best = (lam, res)
    lam_best, ridge_res = best
    ridge_res["lambda_selected_on_val"] = lam_best
    out["ridge"] = ridge_res
    alpha_s = kernel_ridge_fit(K_tr, Ytr[perm], lam_best)
    shuf_r = _score_rankings(scorer, K_va @ alpha_s, class_names, cfg)
    shuf_r.pop("distinct_predicted_answers", None)
    shuf_r["lambda"] = lam_best
    out["ridge_shuffled_label_control"] = shuf_r
    return out


def eval_speaker_features(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    ytr: np.ndarray,
    yva: np.ndarray,
    class_names: list[str],
    cfg: EvalConfig,
    with_logistic: bool = True,
) -> dict:
    """Top-1 who-said-X classification on the speaker-token subset."""
    n_c = len(class_names)
    if n_c < 2 or len(ytr) < 4 or len(yva) < 2:
        return {"skipped": "insufficient labeled items"}
    Ytr = np.zeros((len(ytr), n_c))
    Ytr[np.arange(len(ytr)), ytr] = 1.0
    out: dict[str, dict] = {}
    rng_lab = np.random.default_rng(cfg.control_seed)
    perm = rng_lab.permutation(len(ytr))

    def acc_from(scores: np.ndarray) -> dict:
        pred = scores.argmax(axis=1)
        correct = int((pred == yva).sum())
        return packed_result(correct, len(yva), distinct_predicted_answers=int(len(set(pred.tolist()))))

    if with_logistic:
        Ltr, Lva = subsample_columns(Xtr, Xva, cfg.max_logistic_d, cfg.probe_seed)
        W, b = train_softmax_logistic(
            Ltr, Ytr, iters=cfg.logistic_iters, seed=cfg.probe_seed,
            decay_every=cfg.logistic_decay_every,
        )
        out["logistic"] = acc_from(softmax_probs(W, b, Lva))
        Ws, bs = train_softmax_logistic(
            Ltr, Ytr[perm], iters=cfg.logistic_iters, seed=cfg.probe_seed,
            decay_every=cfg.logistic_decay_every,
        )
        out["logistic_shuffled_label_control"] = acc_from(softmax_probs(Ws, bs, Lva))

    K_tr = linear_kernel(Xtr, Xtr)
    K_va = linear_kernel(Xva, Xtr)
    best = None
    for lam in cfg.ridge_lambdas:
        alpha = kernel_ridge_fit(K_tr, Ytr, lam)
        res = acc_from(K_va @ alpha)
        if best is None or res["accuracy"] > best[1]["accuracy"]:
            best = (lam, res)
    lam_best, ridge_res = best
    ridge_res["lambda_selected_on_val"] = lam_best
    out["ridge"] = ridge_res
    alpha_s = kernel_ridge_fit(K_tr, Ytr[perm], lam_best)
    out["ridge_shuffled_label_control"] = acc_from(K_va @ alpha_s)
    return out


# -------------------------------------------------------------- orchestration


def run_windowed_probe(
    npz_path: str | Path,
    committed_manifest: str | Path,
    freeze_dir: str | Path,
    encoding: str = "byte",
    cfg: EvalConfig | None = None,
    arms_filter: tuple[str, ...] | None = None,
    log=print,
) -> dict:
    """Full retro-probe: verify archive -> segment replay -> sweep depths."""
    cfg = cfg or EvalConfig()
    started = time.perf_counter()

    integ = verify_archive(npz_path, committed_manifest)
    if not integ["pass"]:
        raise RuntimeError(f"archive hash mismatch vs committed manifest: {integ['mismatches']}")
    log(f"archive integrity: {integ['checked']} arrays sha256-verified")

    manifest = json.loads(Path(committed_manifest).read_text(encoding="utf-8"))
    rows = reconstruct_replay_rows(freeze_dir, encoding=encoding)
    dig = curriculum_digest_of(rows)
    expected = manifest["curriculum_digest"]
    if dig != expected:
        raise RuntimeError(f"curriculum digest mismatch: rebuilt {dig} != committed {expected}")
    log(f"segmentation verified: curriculum digest matches committed {dig[:16]}...")

    starts = np.array([r["start_global"] for r in rows], dtype=np.int64)
    bounds = np.array([r["boundary_global"] for r in rows], dtype=np.int64)
    splits = np.array([r["split"] for r in rows])

    arms = load_logged_states(npz_path)
    if arms_filter:
        arms = {k: v for k, v in arms.items() if k in arms_filter}

    # span-task scaffolding (r0c-mirrored)
    ev = Path(freeze_dir) / "evidence"
    train_items = json.loads((ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
    val_items = json.loads((ev / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]
    classes = answer_classes(train_items)
    class_names = sorted(classes, key=classes.get)
    Y_all = np.stack([soft_targets(words(r["answer_text"]), classes) for r in rows])
    scorer = SpanScorer(val_items, SPAN_PREDICTIONS)
    # Majority-prior control: EXACTLY r0c's path -- the full 673-class
    # frequency ranking scored once per VAL item (no top-64 truncation;
    # consumption must be able to walk the whole ranking).
    prior_ranking = majority_ranking(train_items, classes)
    mp_raw = scorer.score([prior_ranking] * len(val_items))
    controls = {
        "majority_prior_within_answer_vocab": packed_result(
            mp_raw["correct"], mp_raw["predictions"],
            distinct_predicted_answers=mp_raw["distinct_predicted_answers"],
        ),
    }
    rng = np.random.default_rng(CONTROL_SEED)
    pool = class_names[:]
    random_rankings = []
    for _ in val_items:
        order = pool[:]
        rng.shuffle(order)
        random_rankings.append(order)
    rr = scorer.score(random_rankings)
    controls["random_ranking_seed11"] = packed_result(rr["correct"], rr["predictions"])
    anchors = {
        "scorer_majority_prior_matches_frozen_26_of_639": bool(
            controls["majority_prior_within_answer_vocab"]["correct"] == FROZEN_MAJORITY_PRIOR_CORRECT
        ),
        "scorer_random_ranking_seed11_matches_frozen_1_of_639": bool(
            controls["random_ranking_seed11"]["correct"] == FROZEN_RANDOM_RANKING_CORRECT
        ),
        "answer_vocab_size": len(classes),
        "note": (
            "k=1 windows equal the archived boundary arrays by construction "
            "(unit-tested) and the archive itself is sha256-pinned to the "
            "committed manifest, so this instrument measures the same states "
            "and scores them with the same bookkeeping as week 1"
        ),
    }

    # speaker-task scaffolding (subset with speaker_token labels)
    spk_mask = np.array([r["speaker_token"] is not None for r in rows])
    spk_classes: dict[str, int] = {}
    for r, m in zip(rows, spk_mask):
        if m and r["split"] == "train":
            spk_classes.setdefault(r["speaker_token"], len(spk_classes))
    spk_index = np.full(len(rows), -1, dtype=np.int64)
    for i, (r, m) in enumerate(zip(rows, spk_mask)):
        if m and r["speaker_token"] in spk_classes:
            spk_index[i] = spk_classes[r["speaker_token"]]
    tr_spk = np.where(spk_mask & (splits == "train"))[0]
    va_spk = np.where(spk_mask & (splits == "val"))[0]
    va_covered = int((spk_index[va_spk] >= 0).sum())
    spk_counts = {}
    for i in tr_spk:
        spk_counts[rows[i]["speaker_token"]] = spk_counts.get(rows[i]["speaker_token"], 0) + 1
    maj_spk = max(spk_counts, key=lambda s: (spk_counts[s], s)) if spk_counts else None
    maj_hits = sum(1 for i in va_spk if rows[i]["speaker_token"] == maj_spk)
    speaker_baseline = packed_result(maj_hits, len(va_spk), majority_speaker=maj_spk) if va_spk.size else {}

    results: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "encoding": encoding,
        "archive": {
            "npz": str(npz_path),
            "committed_manifest": str(committed_manifest),
            "integrity": integ,
            "curriculum_digest": dig,
        },
        "protocol": {
            "window_rule": (
                "last min(k-1, available) in-item pre-boundary stride snapshots "
                "(chronological) + boundary snapshot; hold-first padding; windows "
                "never cross the per-item encoder reset"
            ),
            "depths": list(cfg.depths),
            "families": list(cfg.families),
            "decoders": ["logistic(dim-capped, r0c trainer)", "linear-kernel ridge(full-dim, closed form)"],
            "ridge_lambdas": list(cfg.ridge_lambdas),
            "lambda_selected_on_val": True,
            "logistic_iters": cfg.logistic_iters,
            "logistic_decay_every": cfg.logistic_decay_every,
            "logistic_families": list(cfg.logistic_families),
            "logistic_budget_note": (
                "throttled CPU (~0.7 GFLOPS measured): logistic runs at a reduced "
                "iteration budget on the primary state family only; exact kernel "
                "ridge carries the full depth curve on every family at full "
                "window dimensionality"
            ),
            "max_logistic_d": cfg.max_logistic_d,
            "probe_seed": cfg.probe_seed,
            "control_seed": cfg.control_seed,
            "controls": ["shuffled-label(TRAIN perm)", "majority-prior", "random-ranking"],
        },
        "comparators": {
            "unigram_span_acc": UNIGRAM_SPAN_ACC,
            "trigram_span_acc": TRIGRAM_SPAN_ACC,
            "gu2_two_sigma_threshold": GU2_TWO_SIGMA_THRESHOLD,
            "span_predictions": SPAN_PREDICTIONS,
        },
        "span_controls": controls,
        "anchors": anchors,
        "speaker_subset": {
            "items_train": int(len(tr_spk)),
            "items_val": int(len(va_spk)),
            "train_speakers": len(spk_classes),
            "val_within_train_vocab": va_covered,
            "baseline_majority_speaker": speaker_baseline,
        },
        "arms": {},
    }

    for arm_name, arm in sorted(arms.items()):
        arm_out: dict = {}
        for family in cfg.families:
            if family not in arm.families:
                continue
            f = arm.families[family]
            # NOTE: log_states_union.UnionReplayLogger.mark_boundary appends
            # ``token_offset - 1`` BEFORE incrementing, so the archived
            # boundary_idx convention is one BEHIND the true global token index
            # t at which the state was captured. The digest-verified
            # reconstruction here is the true t.
            if not np.array_equal(f["boundary_idx"] + 1, bounds):
                raise RuntimeError(
                    f"{arm}/{family}: archived boundary_idx (+logger offset) != "
                    "reconstructed segmentation"
                )
            fam_out: dict = {}
            with_logistic = family in cfg.logistic_families
            for depth in cfg.depths:
                X, filled = extract_windows(
                    f["stride_idx"], f["stride_rows"], f["boundary_rows"], starts, bounds, depth
                )
                tr_idx = np.where(splits == "train")[0]
                va_idx = np.where(splits == "val")[0]
                Xtr_s, Xva_s = standardize(X[tr_idx].astype(np.float64), X[va_idx].astype(np.float64))
                entry: dict = {
                    "feature_dim": int(X.shape[1]),
                    "with_logistic": with_logistic,
                    "window_fill_fraction_mean": float(filled.mean()) / max(1, depth - 1) if depth > 1 else 1.0,
                }
                entry["span_decode"] = eval_span_features(
                    Xtr_s, Xva_s, Y_all[tr_idx], class_names, scorer, cfg,
                    with_logistic=with_logistic,
                )
                if va_spk.size and va_covered:
                    tr_mask_blk = spk_mask[tr_idx]
                    va_mask_blk = spk_mask[va_idx]
                    entry["speaker_identity"] = eval_speaker_features(
                        Xtr_s[tr_mask_blk],
                        Xva_s[va_mask_blk],
                        spk_index[tr_idx][tr_mask_blk],
                        spk_index[va_idx][va_mask_blk],
                        sorted(spk_classes, key=spk_classes.get),
                        cfg,
                        with_logistic=with_logistic,
                    )
                fam_out[str(depth)] = entry
                sd = entry["span_decode"]
                log_acc = sd.get("logistic", {}).get("accuracy")
                best_acc = max(v["accuracy"] for k_, v in sd.items() if not k_.endswith("control"))
                log(
                    f"arm={arm_name} family={family} k={depth}: "
                    + (
                        f"logistic={log_acc:.4f} "
                        if log_acc is not None
                        else "logistic=SKIP "
                    )
                    + f"ridge={sd['ridge']['accuracy']:.4f} best={best_acc:.4f}"
                )
            arm_out[family] = fam_out
        results["arms"][f"{arm_name}_n{arm.n_bits}"] = arm_out

    results["wall_seconds"] = round(time.perf_counter() - started, 1)
    results["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
    return results


def summarize(results: dict) -> dict:
    """Depth-scaling digest: does ANY depth beat the bars anywhere?"""
    bars = results["comparators"]
    best_overall = None
    curve = []
    for arm, fams in results["arms"].items():
        for family, depths in fams.items():
            for depth, entry in depths.items():
                sd = entry["span_decode"]
                row = {
                    "arm": arm,
                    "family": family,
                    "k": int(depth),
                    "logistic": sd.get("logistic", {}).get("accuracy"),
                    "ridge": sd["ridge"]["accuracy"],
                    "logistic_shuf": sd.get("logistic_shuffled_label_control", {}).get("accuracy"),
                    "ridge_shuf": sd["ridge_shuffled_label_control"]["accuracy"],
                    "ridge_lambda": sd["ridge"]["lambda_selected_on_val"],
                }
                curve.append(row)
                candidates = [row["ridge"]] + ([row["logistic"]] if row["logistic"] is not None else [])
                cand = max(candidates)
                if best_overall is None or cand > best_overall["acc"]:
                    best_overall = {
                        "acc": cand,
                        "arm": arm,
                        "family": family,
                        "k": int(depth),
                        "decoder": "logistic" if (row["logistic"] or 0) >= row["ridge"] else "ridge",
                    }
    def _best(r):
        cands = [r["ridge"]] + ([r["logistic"]] if r["logistic"] is not None else [])
        return max(cands)
    above_unigram = [r for r in curve if _best(r) > bars["unigram_span_acc"]]
    above_gu2 = [r for r in curve if _best(r) >= bars["gu2_two_sigma_threshold"]]
    return {
        "curve": curve,
        "best_overall": best_overall,
        "configs_above_unigram": len(above_unigram),
        "configs_above_gu2_threshold": len(above_gu2),
        "any_recovery_above_unigram": bool(above_unigram),
        "any_recovery_above_gu2": bool(above_gu2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--npz", required=True)
    p.add_argument("--manifest", required=True, help="COMMITTED logged_states_manifest.json (hash anchor)")
    p.add_argument("--freeze-dir", required=True)
    p.add_argument("--encoding", choices=("byte", "concept"), default="byte")
    p.add_argument("--arms", default=None, help="comma filter, e.g. lam090,lam099")
    p.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    p.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    p.add_argument("--ridge-lambdas", default=",".join(str(x) for x in DEFAULT_RIDGE_LAMBDAS))
    p.add_argument("--logistic-iters", type=int, default=LOGISTIC_ITERS)
    p.add_argument("--logistic-decay-every", type=int, default=500)
    p.add_argument("--logistic-families", default="union")
    p.add_argument("--max-logistic-d", type=int, default=MAX_LOGISTIC_D)
    p.add_argument("--out", required=True, help="output results json path")
    args = p.parse_args()

    sys_path_setup()
    cfg = EvalConfig(
        depths=tuple(int(x) for x in args.depths.split(",")),
        families=tuple(args.families.split(",")),
        ridge_lambdas=tuple(float(x) for x in args.ridge_lambdas.split(",")),
        logistic_iters=args.logistic_iters,
        logistic_decay_every=args.logistic_decay_every,
        logistic_families=tuple(args.logistic_families.split(",")),
        max_logistic_d=args.max_logistic_d,
    )
    results = run_windowed_probe(
        args.npz,
        args.manifest,
        args.freeze_dir,
        encoding=args.encoding,
        cfg=cfg,
        arms_filter=tuple(args.arms.split(",")) if args.arms else None,
    )
    results["summary"] = summarize(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_json(results) + b"\n")
    print(json.dumps(results["summary"]["best_overall"], indent=1))
    print(f"wrote {out}")


def sys_path_setup() -> None:
    """Make humanbrain importable when invoked as a script from experiments/."""
    import os
    import sys

    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _ = os  # keep import local-scope minimal


if __name__ == "__main__":
    main()
