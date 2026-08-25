"""Race-pivot mechanical gate runner / terminal summarizer (protocol §6).

Structure of tools/r0b_gate.py: anchor preflight -> per-arm runs -> aggregate.
Arms execute through ``benchmarks/ladder_experiments.py --experiment
race_pivot_gate`` (one metrics JSON per arm, refuse-overwrite = one-shot
discipline). This runner:

1. verifies the anchor preflight (typed + QA curriculum/corpus digests, git
   lineage present) BEFORE spending anything;
2. executes any arms whose records do not exist yet in preregistered order,
   never re-executing a completed arm (the ladder additionally refuses to
   overwrite an existing metrics file);
3. recomputes eligibility at the terminal summary FROM RAW METRICS — stored
   booleans are never trusted;
4. prints the verbatim per-arm numbers and the leg verdict table.

Exit codes: 0 = all executed gates PASS; 2 = valid terminal INELIGIBILITY (a
scientific result; preserved, NEVER retried silently); 3 = preterminal failure
(anchor mismatch / crash before records) with partials preserved.

Usage:
    python tools/race_pivot_gate.py --leg 96 [--arms P-S1 P-S2 P-S4 P-R4 P-M1]
                                     [--skip-missing]   # summarize only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))

import ladder_experiments as le  # noqa: E402 - benchmarks sibling module

EXPERIMENT_ID = "race_pivot_gate"
ARMS = ("P-S1", "P-S2", "P-S4", "P-R4", "P-M1")
SEED = 7


def _anchor_preflight() -> None:
    """Anchor checks BEFORE any cohort consumption (protocol §6)."""
    from humanbrain import qa_corpus

    qa_curriculum = qa_corpus.build_curriculum(SEED)
    if str(qa_curriculum["sha256"]) != le.RACE_PIVOT_QA_CURRICULUM_ANCHOR:
        raise RuntimeError("QA curriculum anchor mismatch — cohort not spent")
    if str(qa_corpus.corpus_digest()) != le.RACE_PIVOT_QA_CORPUS_SHA256:
        raise RuntimeError("QA corpus digest mismatch — cohort not spent")
    typed = le.build_curriculum(SEED, 42)
    if str(typed["sha256"]) != le.RACE_PIVOT_TYPED_CURRICULUM_ANCHOR:
        raise RuntimeError("typed curriculum anchor mismatch — cohort not spent")


def _recompute(record: dict) -> dict:
    """Eligibility recomputed from raw metrics; stored booleans ignored."""
    return {
        "g_p1_distinct_pass": int(record["distinct_answers"]) >= le.RACE_PIVOT_G1_DISTINCT_MIN,
        "g_p2_exact_pass": int(record["free_run_exact_matches"]) >= le.RACE_PIVOT_G2_EXACT_MIN,
        "g_p3_replay_pass": int(record["train_replay_tasks"]) >= le.RACE_PIVOT_G3_REPLAY_MIN,
        "distinct_answers": int(record["distinct_answers"]),
        "free_run_exact_matches": int(record["free_run_exact_matches"]),
        "train_replay_tasks": int(record["train_replay_tasks"]),
        "train_cases_total": int(record["train_cases_total"]),
        "train_cases_passed": int(record["train_cases_passed"]),
        "modal_share": float(record["modal_share"]),
        "margin_median_pa": record.get("margin_median_pa"),
        "stable_fraction_mean": (record.get("settle") or {}).get("stable_fraction_mean"),
        "flip_rate_mean": (record.get("settle") or {}).get("flip_rate_mean"),
        "wall_s": float(record["wall_s"]),
        "tau_low": ((record.get("preregistration") or {}).get("tau_block") or {}).get(
            "tau_low"
        ),
        "tau_high": ((record.get("preregistration") or {}).get("tau_block") or {}).get(
            "tau_high"
        ),
        "abstention": record.get("abstention"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leg", type=int, required=True, choices=(96, 1024))
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="summarize existing records only; do not execute missing arms",
    )
    arguments = parser.parse_args()

    try:
        _anchor_preflight()
    except RuntimeError as exc:
        print(f"PREFLIGHT ABORT (cohort NOT spent): {exc}", file=sys.stderr)
        return 3

    out_dir = ROOT / "experiments" / EXPERIMENT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for arm in arguments.arms:
        record_path = out_dir / f"{arm}.json"
        if record_path.exists():
            print(f"[reuse existing record] {record_path}")
            continue
        if arguments.skip_missing:
            print(f"[missing, skipped] {arm}")
            continue
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "ladder_experiments.py"),
                "--experiment",
                EXPERIMENT_ID,
                "--arm",
                arm,
                "--N",
                str(arguments.leg),
                "--seed",
                str(SEED),
                "--check-determinism",
            ]
        )
        if result.returncode != 0:
            failures.append(arm)

    rows = []
    overall = True
    seen = False
    for arm in ARMS:
        record_path = out_dir / f"{arm}.json"
        if not record_path.exists():
            continue
        seen = True
        verdict = _recompute(json.loads(record_path.read_text(encoding="utf-8")))
        gate_pass = (
            verdict["g_p1_distinct_pass"]
            and verdict["g_p2_exact_pass"]
            and verdict["g_p3_replay_pass"]
        )
        overall = overall and gate_pass
        rows.append((arm, gate_pass, verdict))

    print("\n=== race_pivot_gate leg N=%d — terminal summary ===" % arguments.leg)
    for arm, gate_pass, v in rows:
        pm1 = ""
        if isinstance(v["abstention"], dict):
            pm1 = (
                f" selective={v['abstention']['selective_accuracy']:.6f}"
                f" coverage={v['abstention']['coverage']:.6f}"
            )
        print(
            f"{arm}: gates={'PASS' if gate_pass else 'FAIL'} "
            f"distinct={v['distinct_answers']} exact={v['free_run_exact_matches']} "
            f"replay={v['train_replay_tasks']}/42 "
            f"cases={v['train_cases_passed']}/{v['train_cases_total']} "
            f"modal_share={v['modal_share']:.4f} "
            f"margin_median_pa={v['margin_median_pa']:.4f} "
            f"stable_frac={v['stable_fraction_mean']:.4f} "
            f"flip_rate={v['flip_rate_mean']:.4f}"
            f"{pm1} wall_s={v['wall_s']:.1f}"
        )
    if failures:
        print(f"ARM EXECUTION FAILURES (preterminal): {failures}", file=sys.stderr)
        return 3
    if not seen:
        print("no records present", file=sys.stderr)
        return 3
    print(f"LEG VERDICT: {'ALL GATES PASS' if overall else 'INELIGIBLE (gates fail)'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
