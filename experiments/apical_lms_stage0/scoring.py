"""Stage-0 scoring: linear-probe R^2 vs the reference online-LMS trajectory.

Gate (FROZEN before implementation; synthesis doc @ e16d22c):
    linear-probe R^2 of the apical trajectory vs a reference online-LMS
    trajectory >= 0.85 (benchmark band 0.85-0.93, DD-DCL SS1.3).
Below threshold -> a fitted GLIF5 apical membrane cannot host the recursion;
B' dies at Stage 0 (stop-on-failure).

Method (adopted from DendriCL's verification methodology, DD-DCL SS4.1):
ridge-probe internal state against a reference algorithm trajectory.
Features = logged apical voltages v_A (context positions only); targets =
the classical leaky-Widrow-Hoff reference states w_ref(t). Train/test split
is BY EPISODE (last ``n_test_episodes`` episodes held out); per-component
R^2 on the test episodes; gate metric = mean R^2 across components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.apical_lms_stage0 import garg_task


@dataclass(frozen=True)
class ProbeResult:
    mean_r2: float
    median_r2: float
    min_r2: float
    per_dim_r2: np.ndarray


@dataclass(frozen=True)
class Verdict:
    passed: bool
    mean_r2: float
    median_r2: float
    gate: float
    n_train_episodes: int
    n_test_episodes: int
    task_dim: int
    d_apical: int
    alpha: float
    gamma: float
    per_dim_r2: list = field(default_factory=list)


def ridge_fit(
    features: np.ndarray, targets: np.ndarray, ridge_lambda: float
) -> dict:
    """Closed-form ridge on standardized features plus unpenalized intercept."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0
    design = (features - mean) / std
    design = np.concatenate([np.ones((design.shape[0], 1)), design], axis=1)
    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0  # intercept unpenalized
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {"weights": weights, "mean": mean, "std": std}


def ridge_predict(model: dict, features: np.ndarray) -> np.ndarray:
    design = (features - model["mean"]) / model["std"]
    design = np.concatenate([np.ones((design.shape[0], 1)), design], axis=1)
    return design @ model["weights"]


def r2_per_component(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    out = np.where(ss_tot > 0.0, 1.0 - ss_res / np.where(ss_tot > 0.0, ss_tot, 1.0), 0.0)
    return out


def probe_and_score(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_test_rows: int,
    ridge_lambda: float,
) -> ProbeResult:
    """Fit on all but the last ``n_test_rows`` rows; score R^2 on them."""
    if n_test_rows <= 0 or n_test_rows >= features.shape[0]:
        raise ValueError("n_test_rows must leave a non-empty train split")
    split = features.shape[0] - n_test_rows
    model = ridge_fit(features[:split], targets[:split], ridge_lambda)
    predicted = ridge_predict(model, features[split:])
    per_dim = r2_per_component(targets[split:], predicted)
    return ProbeResult(
        mean_r2=float(per_dim.mean()),
        median_r2=float(np.median(per_dim)),
        min_r2=float(per_dim.min()),
        per_dim_r2=per_dim,
    )


def collect_trajectories(runner, episodes: list) -> tuple[np.ndarray, np.ndarray]:
    """Replay episodes through ONE runner; keep CONTEXT-position rows only."""
    feats, targets = [], []
    for episode in episodes:
        out = runner.run_episode(episode)
        keep = ~episode.flags  # decision-point rows carry a frozen duplicate state
        feats.append(out.v_traj[keep])
        targets.append(out.ref_traj[keep])
    return np.concatenate(feats), np.concatenate(targets)


def evaluate(episodes: list, runner, *, n_test_episodes: int) -> Verdict:
    """Full Stage-0 leg: replay, probe, gate. Episodes split by index."""
    if not 0 < n_test_episodes < len(episodes):
        raise ValueError("n_test_episodes must split episodes into two non-empty parts")
    features, targets = collect_trajectories(runner, episodes)
    rows_per_episode = episodes[0].n_positions - int(episodes[0].flags.sum())
    result = probe_and_score(
        features,
        targets,
        n_test_rows=n_test_episodes * rows_per_episode,
        ridge_lambda=runner.constants.ridge_lambda,
    )
    c = runner.constants
    return Verdict(
        passed=bool(result.mean_r2 >= c.r2_gate),
        mean_r2=result.mean_r2,
        median_r2=result.median_r2,
        gate=c.r2_gate,
        n_train_episodes=len(episodes) - n_test_episodes,
        n_test_episodes=n_test_episodes,
        task_dim=runner.task_dim,
        d_apical=c.d_apical,
        alpha=c.alpha,
        gamma=c.gamma,
        per_dim_r2=[float(v) for v in result.per_dim_r2],
    )


def verdict_to_dict(verdict: Verdict) -> dict:
    return {
        "passed": verdict.passed,
        "gate": verdict.gate,
        "mean_r2": verdict.mean_r2,
        "median_r2": verdict.median_r2,
        "per_dim_r2": verdict.per_dim_r2,
        "n_train_episodes": verdict.n_train_episodes,
        "n_test_episodes": verdict.n_test_episodes,
        "task_dim": verdict.task_dim,
        "d_apical": verdict.d_apical,
        "alpha": verdict.alpha,
        "gamma": verdict.gamma,
    }
