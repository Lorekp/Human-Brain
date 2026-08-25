"""Post-hoc mechanism probe for ei_sweep (beyond-schema metrics).

Regrows every arm's circuit from its RECORDED arguments (the harness
determinism contract makes this exact, not approximate) and records the
metrics that live outside fixed-schema v1:

- recurrent sign-pattern hash + per-sign-group counts and mean |w|
  (via ``ladder_experiments.recurrent_sign_metrics``);
- margin distribution skew (Fisher-Pearson g1) over TRAIN decision margins;
- state separation, sealed-diagnostic analog: snapshot the eligibility trace
  AND synaptic-current vector after replaying each TRAIN task's framing
  observations (the state the first decision race sees), then pairwise
  1-cosine over task pairs with DIFFERENT first teacher token; medians per
  stratum (Int / List[Int]) and pooled. Sealed null ~0.001.

Cross-checks before recording: rebuilt curriculum sha256 must equal the
record's; recomputed margin min/median/count must equal the record's.
Writes one JSON line per arm to _probe_results.jsonl and a human table to
_probe_output.txt. Non-gating diagnostics only (PREREGISTRATION.md).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks" / "ladder_experiments.py"
SPEC = importlib.util.spec_from_file_location("ladder_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("ladder_experiments", HARNESS)
SPEC.loader.exec_module(HARNESS)

from humanbrain import typed_program  # noqa: E402

HERE = Path(__file__).resolve().parent


def skew_g1(values: list[float]) -> float:
    """Fisher-Pearson standardized moment g1 (biased sample skewness)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size < 3:
        return 0.0
    centered = array - array.mean()
    m2 = float(np.mean(centered**2))
    m3 = float(np.mean(centered**3))
    if m2 <= 0.0:
        return 0.0
    return float(m3 / m2**1.5)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - float(np.dot(left, right)) / (left_norm * right_norm)


def state_separation(
    circuit: HARNESS.RecurrentCellCircuit, tasks: tuple
) -> dict[str, object]:
    """Sealed-diagnostic analog on post-framing first-decision state."""
    protocol = typed_program.Layer1Protocol.for_lane("A2")
    channels: dict[str, list[np.ndarray]] = {
        "eligibility": [],
        "synaptic_current": [],
    }
    strata: list[str] = []
    first_teacher: list[str] = []
    for task in tasks:
        input_ids = typed_program._model_input(
            task.prompt,
            task.signature,
            framing_order=protocol.framing_order,
        )
        circuit.reset_sequence()
        for token_id in input_ids:
            circuit.observe(token_id)
        channels["eligibility"].append(circuit._eligibility.copy())
        channels["synaptic_current"].append(circuit._synaptic_current_pa.copy())
        strata.append("List[Int]" if task.family_id.startswith("list_") else "Int")
        first_teacher.append(task.teacher_tokens[0])

    result: dict[str, object] = {}
    for channel, vectors in channels.items():
        by_stratum: dict[str, list[float]] = {"Int": [], "List[Int]": []}
        for index in range(len(vectors)):
            for other in range(index + 1, len(vectors)):
                if first_teacher[index] == first_teacher[other]:
                    continue
                if strata[index] != strata[other]:
                    continue  # sealed diagnostic pairs within a stratum only
                distance = _cosine_distance(vectors[index], vectors[other])
                by_stratum[strata[index]].append(distance)
        pooled = by_stratum["Int"] + by_stratum["List[Int]"]

        def _median(values: list[float]) -> float | None:
            return round(float(np.median(values)), 9) if values else None

        result[channel] = {
            "pairs_Int": len(by_stratum["Int"]),
            "pairs_List_Int": len(by_stratum["List[Int]"]),
            "median_1minuscos_Int": _median(by_stratum["Int"]),
            "median_1minuscos_List_Int": _median(by_stratum["List[Int]"]),
            "median_1minuscos_pooled": _median(pooled),
        }
    return result


def probe_arm(record_path: Path) -> dict[str, object]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    dump = record["config_dump"]
    seed = int(record["seed"])
    n_cells = int(record["N"])
    fraction = float(dump["circuit_config"]["inhibitory_fraction"])
    assert fraction == float(dump["inhibitory_fraction"])

    curriculum = HARNESS.build_curriculum(seed, int(dump["data_scale"]))
    assert curriculum["sha256"] == record["curriculum_sha256"], "curriculum drift"

    circuit = HARNESS._grow_circuit(
        curriculum,
        n_cells=n_cells,
        seed=seed,
        readout_rule=str(dump["readout_rule"]),
        schedule=str(dump["schedule"]),
        supervision=str(dump["supervision"]),
        codebook=None,  # explicit arm: built-in wiring kept
        inhibitory_fraction=fraction,
    )

    sign_metrics = HARNESS.recurrent_sign_metrics(circuit)

    train_eval = HARNESS._evaluate_cohort(
        circuit, curriculum["train_tasks"], collect_margins=True
    )
    margins: list[float] = train_eval["margins"]  # type: ignore[assignment]
    assert len(margins) == int(record["margin_count"]), "margin count drift"
    assert abs(min(margins) - float(record["margin_min_pa"])) < 1e-6, "margin min drift"
    assert (
        abs(float(np.median(margins)) - float(record["margin_median_pa"])) < 1e-6
    ), "margin median drift"

    separation = state_separation(circuit, curriculum["train_tasks"])  # type: ignore[arg-type]

    return {
        "arm": record["arm"],
        "fraction": fraction,
        "N": n_cells,
        "seed": seed,
        "git_head_record": record["git_head"],
        "sign_metrics": sign_metrics,
        "margin_skew_g1": round(skew_g1(margins), 9),
        "state_separation": separation,
    }


def main() -> int:
    records = sorted(HERE.glob("f*_N*_s*.json"))
    if not records:
        raise SystemExit("no ei_sweep metrics records found next to the probe")
    lines: list[str] = []
    out_path = HERE / "_probe_results.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for path in records:
            result = probe_arm(path)
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            sign = result["sign_metrics"]
            sep_elig = result["state_separation"]["eligibility"]
            sep_curr = result["state_separation"]["synaptic_current"]
            lines.append(
                f"{result['arm']}: inh={int(sign['inhibitory_cells'])} "
                f"exc={int(sign['excitatory_cells'])} "
                f"|w|_inh={sign['mean_abs_weight_inhibitory_pa']} "
                f"|w|_exc={sign['mean_abs_weight_excitatory_pa']} "
                f"sign_sha={str(sign['sign_sha256'])[:12]} "
                f"skew={result['margin_skew_g1']:.4f} "
                f"sep_elig_pooled={sep_elig['median_1minuscos_pooled']} "
                f"sep_curr_pooled={sep_curr['median_1minuscos_pooled']}"
            )
            print(lines[-1])
    (HERE / "_probe_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
