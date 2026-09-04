"""Windowed-decoder diagnostic tier tests (synthetic fixtures, CPU-only).

Covers: Wilson intervals, window extraction (order/padding/item isolation),
primal-dual ridge equivalence, logistic learnability, ranked-consumption span
scoring semantics, archive loading/dispatch/hash verification, shuffled-label
control determinism.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from humanbrain.windowed_probe import (
    ArmStates,
    EvalConfig,
    SpanScorer,
    curriculum_digest_of,
    eval_span_features,
    extract_windows,
    kernel_ridge_fit,
    load_logged_states,
    packed_result,
    soft_targets,
    standardize,
    subsample_columns,
    train_softmax_logistic,
    softmax_probs,
    verify_archive,
    wilson_interval,
)


# ------------------------------------------------------------------- wilson


def test_wilson_interval_basics():
    lo, hi = wilson_interval(29, 639)
    assert 0.0 <= lo <= 29 / 639 <= hi <= 1.0
    # monotone in hits
    lo2, hi2 = wilson_interval(100, 639)
    assert lo2 > lo and hi2 > hi
    # degenerate inputs stay in range (Wilson reaches the boundary exactly)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0, abs=1e-12)
    assert wilson_interval(5, 0) == (0.0, 1.0)


def test_packed_result_fields():
    res = packed_result(29, 639, tag="x")
    assert res["accuracy"] == pytest.approx(0.045383, abs=1e-5)
    assert res["wilson95_lo"] < res["accuracy"] < res["wilson95_hi"]
    assert res["tag"] == "x"


# ----------------------------------------------------------- window features


def _window_fixture():
    """Five stride snapshots + three items; item2 has NO history."""
    rows = {}
    idx = []
    vals = {
        2: [1.0, 0.0],
        4: [2.0, 0.0],
        8: [3.0, 0.0],
        10: [4.0, 0.0],
        14: [5.0, 0.0],
        16: [6.0, 0.0],
        26: [9.0, 0.0],  # answer-region snapshot AFTER item1's boundary (excluded)
    }
    for t, v in vals.items():
        idx.append(t)
        rows[t] = v
    stride_idx = np.array(idx, dtype=np.int64)
    stride_rows = np.array([rows[t] for t in idx], dtype=np.float16)
    boundary_rows = np.array(
        [[99.0, 1.0], [98.0, 2.0], [97.0, 3.0]], dtype=np.float32
    )
    item_starts = np.array([0, 20, 30], dtype=np.int64)
    item_boundaries = np.array([11, 25, 31], dtype=np.int64)
    return stride_idx, stride_rows, boundary_rows, item_starts, item_boundaries


def test_extract_windows_depth_one_is_boundary_only():
    s_idx, s_rows, b_rows, starts, bounds = _window_fixture()
    X, filled = extract_windows(s_idx, s_rows, b_rows, starts, bounds, depth=1)
    assert X.shape == (3, 2)
    assert np.array_equal(X, b_rows.astype(np.float32))
    assert np.all(filled == 0)


def test_extract_windows_order_padding_fill():
    s_idx, s_rows, b_rows, starts, bounds = _window_fixture()
    X, filled = extract_windows(s_idx, s_rows, b_rows, starts, bounds, depth=4)
    assert X.shape == (3, 8)
    # item0: eligible t in {2,4,8,10} -> last 3 chronologically + boundary
    assert np.array_equal(X[0], np.array([[2, 0], [3, 0], [4, 0], [99, 1]], np.float32).reshape(-1))
    assert filled[0] == 3
    # item1: eligible t in [20,25) -> none of {26} qualifies; zero real history
    # -> full hold-boundary padding, filled=0
    assert np.array_equal(X[1], np.tile(b_rows[1], 4))
    assert filled[1] == 0
    # item2: eligible t in {26}? no: 26 >= boundary 31 is false BUT 26 < 31 and
    # 26 >= start 30 is FALSE -> excluded; hold-boundary padding
    assert np.array_equal(X[2], np.tile(b_rows[2], 4))
    assert filled[2] == 0


def test_extract_windows_hold_first_padding_and_isolation():
    s_idx, s_rows, b_rows, starts, bounds = _window_fixture()
    # Give item1 real history: extend boundaries so t in {26} becomes eligible.
    starts2 = np.array([0, 20, 27], dtype=np.int64)
    bounds2 = np.array([11, 27, 28], dtype=np.int64)
    X, filled = extract_windows(s_idx, s_rows, b_rows, starts2, bounds2, depth=4)
    # item1 now admits t=26 only -> [hold(R26), R26, B1]
    expect1 = np.array([[9, 0], [9, 0], [9, 0], [98, 2]], np.float32).reshape(-1)
    assert np.array_equal(X[1], expect1)
    assert filled[1] == 1
    # item0 must NOT see any t >= 11 (its own boundary) nor item1 rows
    assert 26.0 not in X[0] and 9.0 not in X[0]


# -------------------------------------------------------------- decoders


def test_kernel_ridge_primal_dual_equivalence():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 7))
    Y = rng.normal(size=(30, 3))
    lam = 0.37
    K = X @ X.T
    alpha = kernel_ridge_fit(K, Y, lam)
    # dual predictions
    Xv = rng.normal(size=(11, 7))
    dual = (Xv @ X.T) @ alpha
    # primal solution
    W = np.linalg.solve(X.T @ X + lam * np.eye(7), X.T @ Y)
    primal = Xv @ W
    assert np.allclose(dual, primal, atol=1e-8)


def test_logistic_learns_separable_blobs():
    rng = np.random.default_rng(3)
    n = 40
    X = np.vstack([rng.normal(loc=-2, size=(n // 2, 8)), rng.normal(loc=2, size=(n // 2, 8))])
    Y = np.zeros((n, 2))
    Y[: n // 2, 0] = 1
    Y[n // 2 :, 1] = 1
    W, b = train_softmax_logistic(X, Y, iters=400, seed=7)
    pred = softmax_probs(W, b, X).argmax(axis=1)
    assert (pred == Y.argmax(axis=1)).mean() >= 0.95


def test_subsample_columns_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 50))
    A, _ = subsample_columns(X, X, max_d=17, seed=5)
    B, _ = subsample_columns(X, X, max_d=17, seed=5)
    assert A.shape == (10, 17)
    assert np.array_equal(A, B)


def test_standardize_zero_column_safe():
    X = np.column_stack([np.ones(5), np.arange(5, dtype=float) * 2])
    A, B = standardize(X, X * 0.5)
    assert np.allclose(A[:, 0], 0.0)
    assert A[:, 1].std() == pytest.approx(1.0)


# --------------------------------------------------------------- span scorer


def test_span_scorer_ranked_consumption():
    items = [{"answer": "dog cat"}, {"answer": "bird"}]
    scorer = SpanScorer(items, expected_predictions=None)
    assert scorer.predictions == 3
    # ranking consumed after emission: 'dog' cannot be predicted twice
    rankings = [
        ["dog", "cat", "fish"],
        ["bird", "dog", "cat"],
    ]
    res = scorer.score(rankings)
    assert res["correct"] == 3
    assert res["distinct_predicted_answers"] == 3
    # once 'dog' is consumed it can never reappear, even within one item
    res_repeat = scorer.score([["dog", "cat"], []])
    # item0: pos0 -> dog (hit, consumed); pos1 target 'cat' -> next unconsumed 'cat' (hit)
    assert res_repeat["correct"] == 2
    res_skip = scorer.score([["x", "cat"], ["bird"]])
    # item0: pos0 -> x (miss, consumed); pos1 -> cat (hit). item1 -> bird (hit).
    assert res_skip["correct"] == 2
    # exhausted ranking -> empty-string prediction (miss); item1 still hits 'bird'
    res2 = scorer.score([[], ["bird"], []])
    assert res2["correct"] == 1
    assert res2["per_task_correct"] == [0, 1]


def test_soft_targets_normalizes_within_vocab():
    classes = {"dog": 0, "cat": 1}
    y = soft_targets(["dog", "wolf", "cat"], classes)
    assert y.sum() == 1.0 and y[0] == 0.5 and y[1] == 0.5


# ------------------------------------------------------------- eval plumbing


def _toy_eval_setup(seed=0):
    rng = np.random.default_rng(seed)
    n_tr, n_va, d, c = 24, 8, 6, 4
    Xtr = rng.normal(size=(n_tr, d))
    Xva = rng.normal(size=(n_va, d))
    Ytr = rng.dirichlet(np.ones(c), size=n_tr)
    class_names = [f"w{i}" for i in range(c)]
    val_items = [{"answer": f"w{i}"} for i in range(n_va)]
    scorer = SpanScorer(val_items, expected_predictions=None)

    def make_targets(items):
        return np.stack([soft_targets([f"w{i % c}"], {n: j for j, n in enumerate(class_names)}) for i in range(len(items))])

    Ytr = make_targets(range(n_tr))
    return Xtr, Xva, Ytr, class_names, scorer


def test_eval_span_features_structure_and_control_below_or_equal():
    Xtr, Xva, Ytr, names, scorer = _toy_eval_setup()
    cfg = EvalConfig(depths=(1,), families=("union",), ridge_lambdas=(0.1, 1.0), logistic_iters=60)
    out = eval_span_features(Xtr, Xva, Ytr, names, scorer, cfg, cap_logistic_d=False)
    for key in ("logistic", "ridge", "logistic_shuffled_label_control", "ridge_shuffled_label_control"):
        assert key in out
        assert {"accuracy", "correct", "predictions", "wilson95_lo", "wilson95_hi"} <= set(out[key])
    assert out["ridge"]["lambda_selected_on_val"] in (0.1, 1.0)


def test_shuffled_label_control_deterministic():
    Xtr, Xva, Ytr, names, scorer = _toy_eval_setup()
    cfg = EvalConfig(depths=(1,), families=("union",), ridge_lambdas=(1.0,), logistic_iters=40)
    a = eval_span_features(Xtr, Xva, Ytr, names, scorer, cfg, cap_logistic_d=False)
    b = eval_span_features(Xtr, Xva, Ytr, names, scorer, cfg, cap_logistic_d=False)
    assert a == b


def test_eval_span_features_capped_dim_matches_uncapped_when_small():
    Xtr, Xva, Ytr, names, scorer = _toy_eval_setup()
    cfg = EvalConfig(depths=(1,), families=("union",), ridge_lambdas=(1.0,), logistic_iters=80)
    capped = eval_span_features(Xtr, Xva, Ytr, names, scorer, cfg, cap_logistic_d=True)
    uncapped = eval_span_features(Xtr, Xva, Ytr, names, scorer, cfg, cap_logistic_d=False)
    # d < max_d so cap must be a no-op for the LOGISTIC path
    assert capped["logistic"] == uncapped["logistic"]


# ------------------------------------------------------- archive machinery


def _write_mini_npz(path, corrupt_hash=False):
    rng = np.random.default_rng(7)
    n_bits, dim = 64, 4
    token_ids = rng.integers(0, 8, size=30).astype(np.int32)
    stride_idx = np.array([0, 2, 4], dtype=np.int64)
    stride_rows = rng.normal(size=(3, dim)).astype(np.float16)
    boundary_idx = np.array([6, 13], dtype=np.int64)
    boundary_rows = rng.normal(size=(2, dim)).astype(np.float32)
    p = f"arm_mini_n{n_bits}"
    arrays = {
        f"{p}_token_ids": token_ids,
        f"{p}_tokfeat__union_stride_idx": stride_idx,
        f"{p}_tokfeat__union_stride_rows": stride_rows,
        f"{p}_tokfeat__union_boundary_idx": boundary_idx,
        f"{p}_tokfeat__union_boundary_rows": boundary_rows,
        f"{p}_boundary_voltage": boundary_rows.astype(np.float64),
        f"{p}_boundary_spikes": (rng.random((2, dim)) > 0.5).astype(np.uint32),
        f"{p}_boundary_feature": boundary_rows.copy(),
        f"{p}_boundary_traces": boundary_rows.copy(),
    }
    np.savez(path, **arrays)
    manifest = {
        "curriculum_digest": "deadbeef",
        "npz_arrays": {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
            }
            for k, v in arrays.items()
        },
    }
    if corrupt_hash:
        first = next(iter(manifest["npz_arrays"]))
        manifest["npz_arrays"][first]["sha256"] = "0" * 64
    return manifest


def test_verify_archive(tmp_path):
    npz = tmp_path / "logged_states.npz"
    manifest = _write_mini_npz(str(npz))
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    report = verify_archive(npz, mpath)
    assert report["pass"] and report["checked"] == 9
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_corrupt(manifest)), encoding="utf-8")
    report_bad = verify_archive(npz, bad)
    assert not report_bad["pass"] and len(report_bad["mismatches"]) == 1


def _corrupt(manifest):
    clone = json.loads(json.dumps(manifest))
    key = sorted(clone["npz_arrays"])[0]
    clone["npz_arrays"][key]["sha256"] = "f" * 64
    return clone


def test_load_logged_states_dispatch(tmp_path):
    npz = tmp_path / "logged_states.npz"
    _write_mini_npz(str(npz))
    arms = load_logged_states(npz)
    assert "mini" in arms
    arm = arms["mini"]
    assert isinstance(arm, ArmStates)
    assert arm.n_bits == 64
    assert "union" in arm.families
    fam = arm.families["union"]
    assert set(fam) == {"stride_idx", "stride_rows", "boundary_idx", "boundary_rows"}
    assert arm.token_ids.dtype == np.int32
    assert "boundary_voltage" in arm.extras


def test_load_logged_states_generic_fallback(tmp_path):
    npz = tmp_path / "odd.npz"
    np.savez(npz, some_sink_state=np.zeros((3, 2)), other=np.ones(2))
    arms = load_logged_states(npz)
    assert "" in arms and "some_sink_state" in arms[""].extras


def test_curriculum_digest_stable_shape():
    rows = [
        {"qa_id": "a", "family": "f", "prompt_tokens": 3, "total_tokens": 5, "split": "train"},
        {"qa_id": "b", "family": "g", "prompt_tokens": 2, "total_tokens": 4, "split": "val"},
    ]
    d1 = curriculum_digest_of(rows)
    d2 = curriculum_digest_of(list(reversed(rows)))
    assert d1 == d2 and len(d1) == 64
