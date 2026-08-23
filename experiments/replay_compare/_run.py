"""Replay-consolidation dev-gate driver (T6-lite) — preregistered in PLAN.md.

Two-pass design per (N, seed) cell, all through the UNMODIFIED harness internals
of ``benchmarks/ladder_experiments.py`` (imported, never edited):

- pass 1 (arm ``all42_N{n}_s{s}``): grow on all 42 TRAIN tasks, freeze, evaluate;
  the record embeds the per-task error table. One deterministic grow serves as
  BOTH the error-ranking source and the full-data control arm.
- pass 2 (arms ``errtop21_*`` / ``rand21_*``): fresh circuit grown ONLY on the
  K=21 highest-pass-1-error tasks (tie-break lower train index) or a seeded
  uniform-random 21-subset, freeze, then evaluation on the FULL 42-task train
  cohort (forgetting visible) + the fixed 18-task calibration cohort.

Records keep every schema-v1 field byte-compatible and add exactly one
namespaced top-level key ``replay_extension_v1`` (declared deviation, PLAN.md).

One process per (N, seed) cell (3 arms) to preserve the harness's per-process
determinism contract; the shell loop serializes cells. Each invocation acquires
``benchmarks/locks/cpu-bench.lock`` (O_EXCL advisory lock) for its duration.

Run:
    python experiments/replay_compare/_run.py --N 512 --seed 7 [--faithfulness-check]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks"
for candidate in (str(BENCH), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np  # noqa: E402

import ladder_experiments as lx  # noqa: E402

OUT_DIR = ROOT / "experiments" / "replay_compare"
LOCK_PATH = ROOT / "benchmarks" / "locks" / "cpu-bench.lock"

EXPERIMENT_ID = "replay_compare"
READOUT_RULE = "ridge"
SCHEDULE = "massed"
SUPERVISION = "all_transition"
DATA_SCALE = 42
K_REPLAY = 21
RAND_SALT = 0x5EEDC0DE  # domain-separated from every harness-documented stream
SELECTION_RULE_TOP = (
    "top-21 by (-pass1_error_count, train_index) ascending; error := train "
    "cases_total - cases_passed from the pass-1 record's per_task_errors table"
)
SELECTION_RULE_RAND = (
    "uniform 21-of-42 without replacement via "
    "np.random.default_rng([seed, 0x5EEDC0DE]).choice(42, size=21, "
    "replace=False), indices sorted ascending"
)


def acquire_lock(path: Path, timeout_s: float = 600.0) -> None:
    """O_EXCL advisory lock; waits for any concurrent CPU-BENCH holder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                raise SystemExit(
                    f"_run: cpu-bench lock {path} held elsewhere for over "
                    f"{timeout_s:.0f}s; refusing to contend indefinitely"
                )
            time.sleep(0.5)
            continue
        os.write(handle, f"pid={os.getpid()} acquired={time.time():.3f}\n".encode())
        os.close(handle)
        return


def release_lock(path: Path) -> None:
    path.unlink(missing_ok=True)


def evaluate_per_task(
    circuit,
    tasks: tuple,
    *,
    collect_margins: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Per-task mirror of ``_evaluate_cohort`` (identical aggregate semantics).

    Completion failures contribute cases_total but no replay/margins/case
    passes, exactly like the harness; the finite-margin guard is preserved.
    """
    tp = lx.typed_program
    framing = tp.Layer1Protocol.for_lane("A2").framing_order
    per_task: list[dict[str, object]] = []
    replayed = cases_total = cases_passed = 0
    margins: list[float] = []
    for train_index, task in enumerate(tasks):
        input_ids = tp._model_input(task.prompt, task.signature, framing_order=framing)
        tokens: tuple[str, ...] = ()
        replay_ok = False
        task_margins: list[float] = []
        try:
            tokens, replay_ok, task_margins = lx._complete_program(
                circuit,
                input_ids,
                task.signature,
                task.teacher_tokens if collect_margins else None,
            )
        except (tp.ProgramFormatError, RuntimeError):
            cases_total += len(task.cases)
            continue
        replayed += int(replay_ok)
        if collect_margins:
            margins.extend(task_margins)
        passed, total = lx._case_results(task, tokens)
        cases_passed += passed
        cases_total += total
        per_task.append(
            {
                "train_index": train_index,
                "task_id": task.task_id,
                "family_id": task.family_id,
                "template_id": task.template_id,
                "label_replay": bool(replay_ok),
                "emitted_program": bool(tokens),
                "cases_total": int(total),
                "cases_passed": int(passed),
                "error_count": int(total - passed),
            }
        )
    if any(not np.isfinite(value) for value in margins):
        raise RuntimeError("non-finite competitive margin observed")
    aggregates = {
        "replayed": replayed,
        "cases_total": cases_total,
        "cases_passed": cases_passed,
        "margins": margins,
    }
    return aggregates, per_task


def build_record(
    *,
    arm: str,
    n_cells: int,
    seed: int,
    curriculum: dict[str, object],
    codebook_arm,
    grow_task_count: int,
    train_aggregates: dict[str, object],
    calibration: dict[str, object],
    started: float,
    role: str,
    selection_rule: str,
    selected_indices: list[int],
    per_task_table: list[dict[str, object]] | None,
) -> dict[str, object]:
    """Schema-v1-compatible record; mirrors ``run_experiment`` field-for-field."""
    margins = sorted(train_aggregates["margins"])
    config_dump = {
        "corpus_passes": lx._CORPUS_PASSES,
        "data_scale": DATA_SCALE,  # materialized pool scale (see PLAN.md)
        "framing_order": lx.typed_program.Layer1Protocol.for_lane(
            "A2"
        ).framing_order,
        "input_fanout": lx._INPUT_FANOUT,
        "n_cells": n_cells,
        "pool_literal_count": lx._POOL_LITERAL_COUNT,
        "readout_rule": READOUT_RULE,
        "recurrent_fanout": lx._RECURRENT_FANOUT,
        "schedule": SCHEDULE,
        "seed": seed,
        "supervision": SUPERVISION,
        "supervision_scope": lx._SUPERVISION_SCOPE[SUPERVISION],
        "token_cells": lx._TOKEN_CELLS,
        "codebook": codebook_arm.mode,
        "assembly_k": codebook_arm.k if codebook_arm.mode != "explicit" else None,
        "codebook_sha256": codebook_arm.sha256,
        "train_task_count": grow_task_count,  # actual growth-set size
        "calibration_task_count": len(curriculum["calibration_tasks"]),
        "circuit_config": lx.asdict(
            lx.RecurrentCircuitConfig(
                reservoir_cells=n_cells,
                input_fanout=lx._INPUT_FANOUT,
                recurrent_fanout=lx._RECURRENT_FANOUT,
                random_seed=seed,
                readout_learning_rule=READOUT_RULE,
                local_learning_rate_pa=64.0,
                local_competitive_margin_pa=64.0,
            )
        ),
        "frozen_protocol_sha256": curriculum["frozen_protocol_sha256"],
    }
    extension = {
        "schema_note": (
            "schema v1 fields unchanged; single namespaced extension declared "
            "in experiments/replay_compare/PLAN.md before execution"
        ),
        "role": role,
        "k_replayed": grow_task_count,
        "selection_rule": selection_rule,
        "selected_train_indices": selected_indices,
        "eval_train_scope": "full_42_tasks",
    }
    if per_task_table is not None:
        extension["per_task_errors"] = per_task_table
    return {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "N": n_cells,
        "seed": seed,
        "readout_rule": READOUT_RULE,
        "schedule": SCHEDULE,
        "data_scale": DATA_SCALE,
        "supervision": SUPERVISION,
        "curriculum_sha256": curriculum["sha256"],
        "train_replay_tasks": train_aggregates["replayed"],
        "train_cases_total": train_aggregates["cases_total"],
        "train_cases_passed": train_aggregates["cases_passed"],
        "calib_tasks": len(curriculum["calibration_tasks"]),
        "calib_cases_total": calibration["cases_total"],
        "calib_cases_passed": calibration["cases_passed"],
        "margin_count": len(margins),
        "margin_min_pa": round(min(margins), 9) if margins else 0.0,
        "margin_median_pa": round(float(np.median(margins)), 9) if margins else 0.0,
        "wall_s": round(time.perf_counter() - started, 3),
        "git_head": lx._git_head(),
        "config_dump": config_dump,
        "replay_extension_v1": extension,
    }


def run_arm(
    *,
    arm: str,
    role: str,
    n_cells: int,
    seed: int,
    curriculum: dict[str, object],
    codebook_arm,
    grow_indices: list[int],
    selection_rule: str,
    per_task_table_source: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    train_tasks = curriculum["train_tasks"]
    grow_tasks = tuple(train_tasks[index] for index in grow_indices)
    circuit = lx._grow_circuit(
        {"train_tasks": grow_tasks},
        n_cells=n_cells,
        seed=seed,
        readout_rule=READOUT_RULE,
        schedule=SCHEDULE,
        supervision=SUPERVISION,
        codebook=codebook_arm,
    )
    # _grow_circuit freezes internally (recurrent.py freeze_learning is
    # idempotent); no second freeze is issued, mirroring run_experiment exactly.
    # Pass-2 arms are evaluated on the FULL 42-task cohort so forgetting of
    # unconsolidated tasks is measurable; pass 1 grows on all 42 anyway.
    train, per_task = evaluate_per_task(circuit, train_tasks, collect_margins=True)
    calibration, _ = evaluate_per_task(
        circuit, curriculum["calibration_tasks"], collect_margins=False
    )
    return build_record(
        arm=arm,
        n_cells=n_cells,
        seed=seed,
        curriculum=curriculum,
        codebook_arm=codebook_arm,
        grow_task_count=len(grow_indices),
        train_aggregates=train,
        calibration=calibration,
        started=started,
        role=role,
        selection_rule=selection_rule,
        selected_indices=list(grow_indices),
        per_task_table=per_task if per_task_table_source else None,
    )


def write_record(record: dict[str, object]) -> Path:
    out_path = OUT_DIR / f"{record['arm']}.json"
    if out_path.exists():
        raise SystemExit(f"_run: refusing to overwrite existing metrics {out_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def faithfulness_check(curriculum: dict[str, object], codebook_arm) -> None:
    """Pinned pre-execution gate: mirror == direct run_experiment, minus wall_s."""
    reference = lx.run_experiment(
        experiment_id=EXPERIMENT_ID,
        arm="all42_N512_s7",
        n_cells=512,
        seed=7,
        readout_rule=READOUT_RULE,
        schedule=SCHEDULE,
        data_scale=DATA_SCALE,
        supervision=SUPERVISION,
        codebook="explicit",
        assembly_k=16,
    )
    mine = run_arm(
        arm="all42_N512_s7",
        role="_faithfulness_probe_unused",
        n_cells=512,
        seed=7,
        curriculum=curriculum,
        codebook_arm=codebook_arm,
        grow_indices=list(range(DATA_SCALE)),
        selection_rule=SELECTION_RULE_TOP,
        per_task_table_source=False,
    )

    def strip(rec: dict[str, object]) -> bytes:
        return lx._canonical_json(
            {k: v for k, v in rec.items() if k not in ("wall_s", "replay_extension_v1")}
        )

    if strip(mine) != strip(reference):
        raise SystemExit(
            "_run: FAITHFULNESS CHECK FAILED - mirrored record differs from "
            "ladder_experiments.run_experiment beyond wall_s; aborting batch"
        )
    print("[faithfulness] mirror == run_experiment (minus wall_s): OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--N", dest="n_cells", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--faithfulness-check",
        action="store_true",
        help="before the cell, assert the mirrored pass-1 record equals a "
        "direct ladder_experiments.run_experiment call minus wall_s",
    )
    arguments = parser.parse_args(argv)
    n_cells = arguments.n_cells
    seed = arguments.seed

    acquire_lock(LOCK_PATH)
    try:
        curriculum = lx.build_curriculum(seed, DATA_SCALE)
        codebook_arm = lx.build_codebook(
            "explicit",
            token_count=lx._TOKEN_CELLS,
            n_e=n_cells,
            k=16,
            seed=seed,
        )
        if arguments.faithfulness_check:
            faithfulness_check(curriculum, codebook_arm)

        # ---- pass 1: all-42 grow; ranks errors; doubles as full-data control
        control = run_arm(
            arm=f"all42_N{n_cells}_s{seed}",
            role="pass1_all42_rank_and_control",
            n_cells=n_cells,
            seed=seed,
            curriculum=curriculum,
            codebook_arm=codebook_arm,
            grow_indices=list(range(DATA_SCALE)),
            selection_rule=SELECTION_RULE_TOP,
            per_task_table_source=True,
        )
        table = control["replay_extension_v1"]["per_task_errors"]
        ranking = sorted(table, key=lambda row: (-row["error_count"], row["train_index"]))
        top_indices = sorted(int(row["train_index"]) for row in ranking[:K_REPLAY])
        rng = np.random.default_rng([seed, RAND_SALT])
        rand_indices = sorted(
            int(index)
            for index in rng.choice(DATA_SCALE, size=K_REPLAY, replace=False)
        )
        overlap = len(set(top_indices) & set(rand_indices))

        # ---- pass 2a: error-prioritized replay subset
        errtop = run_arm(
            arm=f"errtop21_N{n_cells}_s{seed}",
            role="pass2_error_top21_replay",
            n_cells=n_cells,
            seed=seed,
            curriculum=curriculum,
            codebook_arm=codebook_arm,
            grow_indices=top_indices,
            selection_rule=SELECTION_RULE_TOP,
            per_task_table_source=False,
        )

        # ---- pass 2b: random-subset replay control
        randarm = run_arm(
            arm=f"rand21_N{n_cells}_s{seed}",
            role="pass2_random21_replay",
            n_cells=n_cells,
            seed=seed,
            curriculum=curriculum,
            codebook_arm=codebook_arm,
            grow_indices=rand_indices,
            selection_rule=SELECTION_RULE_RAND,
            per_task_table_source=False,
        )

        written = [write_record(control), write_record(errtop), write_record(randarm)]
        for record, path in zip((control, errtop, randarm), written):
            print(
                json.dumps(
                    {
                        "written": str(path.relative_to(ROOT)),
                        "role": record["replay_extension_v1"]["role"],
                        "k_replayed": record["config_dump"]["train_task_count"],
                        "train_replay_tasks": record["train_replay_tasks"],
                        "train_cases_passed": record["train_cases_passed"],
                        "train_cases_total": record["train_cases_total"],
                        "calib_cases_passed": record["calib_cases_passed"],
                        "calib_cases_total": record["calib_cases_total"],
                        "wall_s": record["wall_s"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "cell_done": f"N{n_cells}_s{seed}",
                    "error_hist": {
                        str(e): sum(1 for r in table if r["error_count"] == e)
                        for e in (3, 2, 1, 0)
                    },
                    "top_vs_rand_overlap": overlap,
                    "curriculum_sha256": control["curriculum_sha256"][:12],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        release_lock(LOCK_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
