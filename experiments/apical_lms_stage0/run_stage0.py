#!/usr/bin/env python3
"""Stage-0 entrypoint: Garg regression through TwoCompartmentGLIF5 + R^2 gate.

Modes
-----
(default) fixture-scale leg: small episode counts, infrastructure smoke +
readiness numbers ONLY -- gates nothing except readiness.
--cohort : the decisive corpus leg. BLOCKED behind the retro-probe license
(execution-order step 0 / step -1): pass --license-token or set
APICAL_LMS_STAGE0_LICENSE after the sibling retro-probe verdict commits.
Stop-on-failure: a below-gate verdict exits nonzero and B' dies at Stage 0.

Deterministic end to end: fixed seeds; episodes are independent (each starts
from full reset), so results assemble in submission order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _entry in (str(REPO), str(REPO / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.apical_lms_stage0 import freeze, garg_task, runner, scoring  # noqa: E402

COHORT_LICENSE_ENV = "APICAL_LMS_STAGE0_LICENSE"


def enforce_cohort_guard(*, cohort: bool, license_token: str | None) -> None:
    """The corpus leg may not fire before the step -1 retro-probe licenses it."""
    if cohort and not license_token:
        raise SystemExit(
            "corpus cohort leg BLOCKED: awaiting the P-A step -1 retro-probe "
            f"verdict (set --license-token or {COHORT_LICENSE_ENV}). "
            "Fixture-scale legs need no license."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--task-dim", type=int, default=freeze.FROZEN.task_dim_default)
    parser.add_argument("--query-len", type=int, default=freeze.FROZEN.query_len)
    parser.add_argument("--seed", type=int, default=freeze.FROZEN.base_seed)
    parser.add_argument(
        "--input-current-pa", type=float, default=None,
        help="override somatic pulse current (default: frozen stack value)",
    )
    parser.add_argument("--cohort", action="store_true",
                        help="decisive corpus leg (license-gated)")
    parser.add_argument("--license-token", type=str,
                        default=None, help="retro-probe verdict token")
    parser.add_argument("--out", type=str, default=None, help="verdict JSON path")
    args = parser.parse_args(argv)

    license_token = args.license_token
    if license_token is None:
        import os

        license_token = os.environ.get(COHORT_LICENSE_ENV)
    enforce_cohort_guard(cohort=args.cohort, license_token=license_token)

    c = freeze.FROZEN
    spec = garg_task.TaskSpec(
        d_task=args.task_dim,
        context_mult=c.context_mult,
        query_len=args.query_len,
        noise_sigma=c.noise_sigma,
        w_std=c.w_std,
    )
    rng = np.random.default_rng(args.seed)
    n_episodes = args.train_episodes + args.test_episodes
    episodes = [garg_task.generate_episode(rng, spec, label_byte_scale=c.label_byte_scale)
                for _ in range(n_episodes)]

    sim = runner.Stage0Runner.build(
        c,
        task_dim=args.task_dim,
        lms_enabled=True,
        seed=args.seed,
        input_current_pa=args.input_current_pa,
    )
    verdict = scoring.evaluate(episodes, sim, n_test_episodes=args.test_episodes)

    payload = scoring.verdict_to_dict(verdict)
    payload.update(
        {
            "leg": "cohort" if args.cohort else "fixture",
            "license": license_token if args.cohort else None,
            "seed": args.seed,
            "n_episodes": n_episodes,
            "context_len": spec.context_len,
            "query_len": spec.query_len,
            "tau_a_ms": c.tau_a_ms,
            "stability_bound_gamma": freeze.stability_bound(c.d_apical),
        }
    )
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")

    # Stop-on-failure: below-gate verdicts exit nonzero (B' dies at Stage 0).
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
