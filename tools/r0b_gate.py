"""R0-b mechanical gate runner (frozen protocol docs/research/r0b-state-feature-protocol.md §4).

Executes the five preregistered arms in pinned order and derives the
per-family verdicts:

- Arms (§4.2): r0b_A_spikewin (spike_counts/explicit),
  r0b_B_voltq (voltage_quantiles/explicit), r0b_C_asmbl_sim
  (assembly_overlap/similarity k=16), r0b_C_asmbl_rand (assembly_overlap/
  random k=16, control), then r0b_combo_<X>_<Y> chosen by the frozen
  §4.4 selection rule over {A, B, C_sim} - the control is never a candidate.
  Every arm: N=1024, seed=7, massed/42/all_transition, ridge readout,
  scipy ridge backend, CPU-pinned, --check-determinism enforced,
  curriculum anchor checked ONCE before the first arm.
- RESUME / ONE-SHOT DISCIPLINE: completion is tracked per arm on disk -
  ``experiments/r0b_state_features/<arm>.json`` written by the ladder
  subprocess. A rerun of this runner REUSES every existing arm record and
  never re-executes a completed arm (the ladder additionally refuses to
  overwrite an existing metrics file), so an interrupted execution can be
  continued without spending any arm twice.
- G1 (separation, primary): collect_state_separation over the 18
  CALIBRATION framing replays on feature_state_vector(); pooled median
  1-cosine > 0.10.
- G2 (answers, primary): qa_experiments.run(ExperimentConfig(seed=7,
  circuit=RecurrentCircuitConfig(reservoir_cells=96, feature_family=<family>)))
  yields distinct_answers >= 3 (GUARD_DISTINCT_MIN).
- G3 (capacity non-regression, secondary): train_replay_tasks >= 39 of 42.

Per-arm verdict semantics (§4.3): PASS iff G1 & G2 & G3; ¬G1 or ¬G2 is
FAMILY-FALSIFIED; G1 & G2 & ¬G3 is CAPACITY-REGRESSIVE. Program outcomes
(§4.5): any PASS -> de-collapse; ALL ¬G1 -> state-feature class falsified,
program pivots to the race/output layer; ∃G1 passes but ALL ¬G2 ->
readout/race blocker; only capacity failures -> banked dispositions.

Exit codes: 0 = gate executed to completion with verdicts recorded;
2 = ABORTED before the cohort was spent (anchor mismatch or an arm crashed
before producing its record; completed records are preserved and marked
spent-with-partials). One shot: no re-seeding, no rerunning "to check".
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

PROTOCOL = "docs/research/r0b-state-feature-protocol.md @ d2db8be"
EXPERIMENT_ID = "r0b_state_features"
N_CELLS = 1024
SEED = 7

# Pinned execution order (§4.2): (arm, family, codebook, assembly_k).
SINGLE_ARMS = (
    ("r0b_A_spikewin", "spike_counts", "explicit", None),
    ("r0b_B_voltq", "voltage_quantiles", "explicit", None),
    ("r0b_C_asmbl_sim", "assembly_overlap", "similarity", 16),
    ("r0b_C_asmbl_rand", "assembly_overlap", "random", 16),
)

# Canonical family order for every ordering decision (protocol §2): A < B < C.
_CANONICAL_ORDER = ("spike_counts", "voltage_quantiles", "assembly_overlap")


def select_combo_members(records: list[dict]) -> tuple[str, str]:
    """Frozen §4.4 combination-arm selection over synthetic/live records.

    Candidates are CONFIRMATORY single-family arms only - the control
    (role != 'confirmatory') is never considered. Ranking (pinned by
    R0B-T12 against synthetic ties and missing diagnostics):

    1. ``median_1minuscos_pooled`` ascending (missing diagnostics sort
       last, deterministically);
    2. higher ``train_replay_tasks``;
    3. lower ``wall_s``;
    4. canonical family order (spike_counts < voltage_quantiles <
       assembly_overlap).

    Returns ``(best, second)`` in rank order. Raises ValueError unless at
    least two candidates exist.
    """
    candidates = [
        record for record in records if record.get("role") == "confirmatory"
    ]
    if len(candidates) < 2:
        raise ValueError(
            "combination arm needs at least two confirmatory candidate arms; "
            f"got {len(candidates)}"
        )

    def sort_key(record: dict) -> tuple[int, float, int, float, int]:
        median = record.get("median_1minuscos_pooled")
        missing = 0 if isinstance(median, (int, float)) else 1
        replay = int(record.get("train_replay_tasks") or 0)
        wall = float(record.get("wall_s") or 0.0)
        family = str(record.get("family"))
        canonical = (
            _CANONICAL_ORDER.index(family) if family in _CANONICAL_ORDER else len(
                _CANONICAL_ORDER
            )
        )
        return (
            missing,
            float(median) if missing == 0 else 0.0,
            -replay,
            wall,
            canonical,
        )

    ranked = sorted(candidates, key=sort_key)
    best, second = ranked[0], ranked[1]
    return str(best["family"]), str(second["family"])


def _canonical_combo(members: tuple[str, str]) -> str:
    ordered = sorted(
        members, key=lambda name: _CANONICAL_ORDER.index(name)
    )
    return f"combo:{ordered[0]}+{ordered[1]}"


def _ladder_command(
    arm: str,
    family: str,
    codebook: str,
    assembly_k: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "benchmarks" / "ladder_experiments.py"),
        "--experiment",
        EXPERIMENT_ID,
        "--arm",
        arm,
        "--N",
        str(N_CELLS),
        "--seed",
        str(SEED),
        "--schedule",
        "massed",
        "--data-scale",
        "42",
        "--supervision",
        "all_transition",
        "--readout-rule",
        "ridge",
        "--ridge-backend",
        "scipy",
        "--feature-family",
        family,
        "--codebook",
        codebook,
        "--device",
        "cpu",
        "--check-determinism",
    ]
    if assembly_k is not None:
        command += ["--assembly-k", str(assembly_k)]
    return command


def _load_or_run_arm(
    arm: str,
    family: str,
    codebook: str,
    assembly_k: int | None,
    experiments_root: Path,
) -> dict:
    """Run one ladder arm (or resume its existing record after a crash)."""
    record_path = experiments_root / EXPERIMENT_ID / f"{arm}.json"
    if record_path.is_file():
        print(f"[{arm}] existing record found; reusing (cohort already spent)")
        return json.loads(record_path.read_text(encoding="utf-8"))
    completed = subprocess.run(
        _ladder_command(arm, family, codebook, assembly_k),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print(completed.stdout.strip())
    if completed.returncode != 0 or not record_path.is_file():
        print(completed.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"ladder arm {arm} failed before producing its record")
    return json.loads(record_path.read_text(encoding="utf-8"))


def _evaluate_gates(record: dict, family: str, curriculum: dict) -> dict:
    """Derive G1/G2/G3 plus diagnostics for one completed arm record."""
    from humanbrain.recurrent import RecurrentCircuitConfig

    import ladder_experiments as le
    import qa_experiments

    codebook_mode = str(record["config_dump"]["codebook"])
    assembly_k = record["config_dump"].get("assembly_k")
    codebook_arm = le.build_codebook(
        codebook_mode,
        token_count=le._TOKEN_CELLS,
        n_e=N_CELLS,
        k=int(assembly_k) if assembly_k else le._DEFAULT_ASSEMBLY_K,
        seed=SEED,
    ) if codebook_mode != "explicit" else None
    circuit = le._grow_circuit(
        curriculum,
        n_cells=N_CELLS,
        seed=SEED,
        readout_rule="ridge",
        schedule="massed",
        supervision="all_transition",
        codebook=codebook_arm,
        ridge_backend="scipy",
        feature_family=family,
    )
    separation = le.collect_state_separation(
        circuit, curriculum["calibration_tasks"]
    )
    median_one_minus_cos = separation["eligibility"]["median_1minuscos_pooled"]
    g1_pass = bool(
        median_one_minus_cos is not None
        and median_one_minus_cos > G1_MIN_MEDIAN_ONE_MINUS_COSINE
    )

    qa_record = qa_experiments.run(
        qa_experiments.ExperimentConfig(
            seed=SEED,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=96,
                feature_family=family,
            ),
        )
    )
    degeneracy = qa_record["metrics"]["degeneracy"]
    distinct_answers = int(degeneracy["distinct_answers"])
    g2_pass = bool(distinct_answers >= G2_MIN_DISTINCT_ANSWERS)

    train_replay = int(record["train_replay_tasks"])
    train_total = (
        int(record["train_task_count"]) if "train_task_count" in record else 42
    )
    g3_pass = bool(train_replay >= G3_MIN_TRAIN_REPLAY)

    if g1_pass and g2_pass and g3_pass:
        verdict = "PASS"
    elif not (g1_pass and g2_pass):
        verdict = "FAMILY-FALSIFIED"
    else:
        verdict = "CAPACITY-REGRESSIVE"

    return {
        "experiment_id": EXPERIMENT_ID,
        "arm": record["arm"],
        "family": family,
        "role": "confirmatory" if family != "assembly_overlap" or codebook_mode != "random" else "control",
        "codebook": codebook_mode,
        "git_head": record["git_head"],
        "curriculum_sha256": record["curriculum_sha256"],
        "curriculum_anchor_match": True,
        "median_1minuscos_pooled": median_one_minus_cos,
        "train_replay_tasks": train_replay,
        "wall_s": record["wall_s"],
        "gates": {
            "G1": {
                "definition": (
                    "pooled median post-framing cosine < 0.90 across 18 "
                    "CALIBRATION tasks on feature_state_vector()"
                ),
                "median_1minuscos_pooled": separation["eligibility"]["median_1minuscos_pooled"],
                "median_1minuscos_Int": separation["eligibility"]["median_1minuscos_Int"],
                "median_1minuscos_List_Int": separation["eligibility"]["median_1minuscos_List_Int"],
                "pairs_Int": separation["eligibility"]["pairs_Int"],
                "pairs_List_Int": separation["eligibility"]["pairs_List_Int"],
                "pass": g1_pass,
            },
            "G2": {
                "definition": (
                    "QA fixture distinct_answers >= 3 (held-out CALIBRATION "
                    "free-run, seed 7, 96-cell circuit)"
                ),
                "distinct_answers": distinct_answers,
                "modal_share": degeneracy["modal_share"],
                "evaluated": degeneracy["evaluated"],
                "answer_span_accuracy": qa_record["metrics"]["answer_span"]["accuracy"],
                "free_run_exact_matches": qa_record["metrics"]["free_run"]["exact_matches"],
                "pass": g2_pass,
            },
            "G3": {
                "definition": f"train_replay_tasks >= {G3_MIN_TRAIN_REPLAY} of {train_total}",
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
            "margin_min_pa": record["margin_min_pa"],
            "margin_median_pa": record["margin_median_pa"],
            "simulation_backend": record["config_dump"]["simulation_backend"],
            "ridge_backend": record["config_dump"]["ridge_backend"],
        },
        "verdict": verdict,
    }


def main() -> int:
    import ladder_experiments as le

    experiments_root = ROOT / "experiments"

    # --- Preflight: freeze-anchor check ONCE before any token is observed. ---
    curriculum = le.build_curriculum(seed=SEED, data_scale=42)
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

    summaries: list[dict] = []

    # --- Arms 1-4 in pinned order. ---
    try:
        for arm, family, codebook, assembly_k in SINGLE_ARMS:
            print(f"=== arm {arm}: family={family} codebook={codebook} ===")
            record = _load_or_run_arm(
                arm, family, codebook, assembly_k, experiments_root
            )
            gate = _evaluate_gates(record, family, curriculum)
            gate_path = experiments_root / EXPERIMENT_ID / f"{arm}.gate.json"
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate_path.write_text(
                json.dumps(gate, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            summaries.append(gate)
            print(f"[{arm}] verdict: {gate['verdict']}")
    except (RuntimeError, OSError) as error:
        results = {
            "protocol": PROTOCOL,
            "verdict": "ABORTED (cohort spent-with-partials)",
            "reason": str(error),
            "completed_records": [gate["arm"] for gate in summaries],
        }
        output = experiments_root / EXPERIMENT_ID / "gate_results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 2

    # --- Arm 5: frozen §4.4 combination selection. ---
    first, second = select_combo_members(summaries)
    combo_family = _canonical_combo((first, second))
    # Combo codebooks (§4.2): both members A/B -> explicit; any member C ->
    # that C arm's codebook (similarity; the control is never a candidate).
    needs_similarity = "assembly_overlap" in (first, second)
    combo_codebook = "similarity" if needs_similarity else "explicit"
    combo_assembly_k = 16 if needs_similarity else None
    canonical_pair = sorted(
        (first, second), key=_CANONICAL_ORDER.index
    )
    combo_arm = f"r0b_combo_{canonical_pair[0]}_{canonical_pair[1]}"
    print(f"=== arm {combo_arm}: family={combo_family} codebook={combo_codebook} ===")
    try:
        record = _load_or_run_arm(
            combo_arm, combo_family, combo_codebook, combo_assembly_k, experiments_root
        )
        gate = _evaluate_gates(record, combo_family, curriculum)
        gate["selection_rule"] = {
            "members_rank_order": [first, second],
            "canonical_family": combo_family,
            "note": "frozen §4.4; control excluded from candidates",
        }
        gate_path = experiments_root / EXPERIMENT_ID / f"{combo_arm}.gate.json"
        gate_path.write_text(
            json.dumps(gate, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(gate)
        print(f"[{combo_arm}] verdict: {gate['verdict']}")
    except (RuntimeError, OSError) as error:
        results = {
            "protocol": PROTOCOL,
            "verdict": "ABORTED (cohort spent-with-partials)",
            "reason": str(error),
            "completed_records": [item["arm"] for item in summaries],
            "arms": summaries,
        }
        output = experiments_root / EXPERIMENT_ID / "gate_results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 2

    # --- Program-level outcome (§4.5). ---
    g1_passes = sum(1 for item in summaries if item["gates"]["G1"]["pass"])
    g2_passes = sum(1 for item in summaries if item["gates"]["G2"]["pass"])
    pass_arms = [item for item in summaries if item["verdict"] == "PASS"]
    if pass_arms:
        program_outcome = (
            "FEATURE DE-COLLAPSE ACHIEVED: "
            + ", ".join(sorted(item["arm"] for item in pass_arms))
            + "; R2 sizing proceeds with the passing basis (lowest "
            "median_1minuscos_pooled among passing arms, tie-break lower wall_s)"
        )
    elif g1_passes == 0:
        program_outcome = (
            "STATE-FEATURE CLASS FALSIFIED at this operating point: features "
            "are NOT the bottleneck. Preregistered pivot: race/output layer - "
            "winner-feedback broadcast loop seeded by HOT-2 (roadmap :119-120)"
        )
    elif g2_passes == 0:
        program_outcome = (
            "READOUT/RACE BLOCKER: features separate but answers stay "
            "degenerate; downstream localization to the race/output layer "
            "(NOT a class falsification)"
        )
    else:
        program_outcome = (
            "BANKED DISPOSITIONS: only capacity failures; follow-ups require "
            "fresh v2 protocols per family"
        )

    results = {
        "protocol": PROTOCOL,
        "git_heads": sorted({item["git_head"] for item in summaries}),
        "curriculum_anchor_match": True,
        "arms": summaries,
        "per_family_verdicts": {
            item["family"]: item["verdict"] for item in summaries
        },
        "program_outcome": program_outcome,
        "verdict": program_outcome.split(":")[0],
    }
    output = experiments_root / EXPERIMENT_ID / "gate_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"program_outcome": program_outcome}, indent=2))
    print(f"gate results written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
