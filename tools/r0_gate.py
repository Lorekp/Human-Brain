"""R0 mechanical gate runner (frozen protocol docs/research/r0-multitrace-protocol.md §3).

Executes the single confirmatory arm and derives the preregistered verdicts:

- Arm: N=1024, seed=7, massed/42/all_transition, ridge readout, explicit
  codebook, CPU-pinned, --check-determinism enforced byte-identical modulo
  wall_s, curriculum anchor checked BEFORE execution.
- G1 (separation, primary): pooled MEDIAN cosine across calibration-task
  framing-state pairs with different first teacher token within stratum
  < 0.90, i.e. collect_state_separation's median_1minuscos_pooled > 0.10 on
  the flat [e_0;..;e_K-1] buffer.
- G2 (answers, primary): qa_experiments.run(ExperimentConfig(seed=7)) at
  committed defaults yields distinct_answers >= 3 (GUARD_DISTINCT_MIN).
- G3 (capacity non-regression, secondary): train_replay_tasks >= 39 of 42.

Verdict semantics (§3.3): PASS iff G1 & G2 & G3; primary failure (not G1 or
not G2) falsifies the eligibility-feature family; secondary-only failure is
"capacity-regressive". One shot: nothing may be re-seeded or re-run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))

CURRICULUM_ANCHOR = "48219eda110f09f735525a7de00e56596ffac729746654e809c3c5eb525ccae6"
G1_MIN_MEDIAN_ONE_MINUS_COSINE = 0.10  # <=> pooled median cosine < 0.90
G2_MIN_DISTINCT_ANSWERS = 3  # GUARD_DISTINCT_MIN (benchmarks/qa_experiments.py)
G3_MIN_TRAIN_REPLAY = 39  # of 42


def main() -> int:
    import ladder_experiments as le
    import qa_experiments

    # --- Preflight: freeze-anchor check before any token is observed. ---
    curriculum = le.build_curriculum(seed=7, data_scale=42)
    if curriculum["sha256"] != CURRICULUM_ANCHOR:
        print(
            json.dumps(
                {
                    "verdict": "ABORTED",
                    "reason": (
                        "curriculum anchor mismatch: generators drifted; cohort "
                        "not spent; fix provenance first"
                    ),
                    "curriculum_sha256": curriculum["sha256"],
                },
                indent=2,
            )
        )
        return 2

    experiment_id = "r0_multitrace"
    arm = "ridge_N1024_s7"

    # --- The single confirmatory run (byte-determinism enforced). ---
    command = [
        sys.executable,
        str(ROOT / "benchmarks" / "ladder_experiments.py"),
        "--experiment",
        experiment_id,
        "--arm",
        arm,
        "--N",
        "1024",
        "--seed",
        "7",
        "--schedule",
        "massed",
        "--data-scale",
        "42",
        "--supervision",
        "all_transition",
        "--readout-rule",
        "ridge",
        "--codebook",
        "explicit",
        "--device",
        "cpu",
        "--check-determinism",
    ]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    print(completed.stdout.strip())
    if completed.returncode != 0:
        print(completed.stderr.strip(), file=sys.stderr)
        print(json.dumps({"verdict": "ABORTED", "reason": "ladder arm failed"}))
        return 2

    record = json.loads(
        (ROOT / "experiments" / experiment_id / f"{arm}.json").read_text(
            encoding="utf-8"
        )
    )

    # --- G1: separation over CALIBRATION framing states (flat K*R buffer). ---
    circuit = le._grow_circuit(
        curriculum,
        n_cells=1024,
        seed=7,
        readout_rule="ridge",
        schedule="massed",
        supervision="all_transition",
        ridge_backend="scipy",
    )
    separation = le.collect_state_separation(circuit, curriculum["calibration_tasks"])
    median_one_minus_cos = separation["eligibility"]["median_1minuscos_pooled"]
    g1_pass = bool(
        median_one_minus_cos is not None
        and median_one_minus_cos > G1_MIN_MEDIAN_ONE_MINUS_COSINE
    )

    # --- G2: QA fixture answers at committed defaults, seed 7. ---
    qa_record = qa_experiments.run(qa_experiments.ExperimentConfig(seed=7))
    degeneracy = qa_record["metrics"]["degeneracy"]
    distinct_answers = int(degeneracy["distinct_answers"])
    g2_pass = bool(distinct_answers >= G2_MIN_DISTINCT_ANSWERS)

    # --- G3: capacity non-regression from the arm record. ---
    train_replay = int(record["train_replay_tasks"])
    train_total = int(record["train_task_count"]) if "train_task_count" in record else 42
    g3_pass = bool(train_replay >= G3_MIN_TRAIN_REPLAY)

    if g1_pass and g2_pass and g3_pass:
        verdict = "PASS"
    elif not (g1_pass and g2_pass):
        verdict = "PRIMARY FAILURE: eligibility-feature family FALSIFIED"
    else:
        verdict = "SECONDARY-ONLY FAILURE: capacity-regressive"

    gate = {
        "protocol": "docs/research/r0-multitrace-protocol.md @ 771bd93",
        "git_head": record["git_head"],
        "curriculum_sha256": record["curriculum_sha256"],
        "curriculum_anchor_match": True,
        "arm": {
            "experiment_id": experiment_id,
            "arm": arm,
            "n_cells": 1024,
            "seed": 7,
            "schedule": "massed",
            "data_scale": 42,
            "supervision": "all_transition",
            "readout_rule": "ridge",
            "codebook": "explicit",
            "device": "cpu",
            "check_determinism": True,
            "eligibility_taus_ms": list(circuit.config.eligibility_taus_ms),
            "trace_count": circuit.trace_count,
            "feature_cell_count": circuit.trace_count * 1024,
        },
        "gates": {
            "G1": {
                "definition": "pooled median post-framing cosine < 0.90 across 18 CALIBRATION tasks",
                "median_1minuscos_pooled": separation["eligibility"][
                    "median_1minuscos_pooled"
                ],
                "median_1minuscos_Int": separation["eligibility"]["median_1minuscos_Int"],
                "median_1minuscos_List_Int": separation["eligibility"][
                    "median_1minuscos_List_Int"
                ],
                "pairs_Int": separation["eligibility"]["pairs_Int"],
                "pairs_List_Int": separation["eligibility"]["pairs_List_Int"],
                "pass": g1_pass,
            },
            "G2": {
                "definition": "QA fixture distinct_answers >= 3 (held-out CALIBRATION free-run)",
                "distinct_answers": distinct_answers,
                "modal_share": degeneracy["modal_share"],
                "evaluated": degeneracy["evaluated"],
                "answer_span_accuracy": qa_record["metrics"]["answer_span"]["accuracy"],
                "free_run_exact_matches": qa_record["metrics"]["free_run"][
                    "exact_matches"
                ],
                "pass": g2_pass,
            },
            "G3": {
                "definition": "train_replay_tasks >= 39 of 42",
                "train_replay_tasks": train_replay,
                "train_task_total": train_total,
                "train_cases_passed": record["train_cases_passed"],
                "train_cases_total": record["train_cases_total"],
                "calib_cases_passed": record["calib_cases_passed"],
                "calib_cases_total": record["calib_cases_total"],
                "pass": g3_pass,
            },
        },
        "diagnostics_never_gate": {
            "state_separation_train_channel": record.get("diagnostics"),
            "wall_s": record["wall_s"],
            "margin_min_pa": record["margin_min_pa"],
            "margin_median_pa": record["margin_median_pa"],
            "simulation_backend": record["config_dump"]["simulation_backend"],
            "ridge_backend": record["config_dump"]["ridge_backend"],
        },
        "verdict": verdict,
    }
    output = ROOT / "experiments" / experiment_id / "gate_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: gate[k] for k in ("verdict",)}, indent=2))
    print(f"gate results written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
