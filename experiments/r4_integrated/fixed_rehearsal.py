"""One equal-offer rehearsal comparison from authenticated fitted-A parents."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "cell_b_only": "trainable",
    "cell_rehearsal": "trainable",
    "conventional_rehearsal": "conventional",
    "frozen_producer_rehearsal": "frozen_producer",
    "cell_no_learning": "trainable",
    "conventional_no_learning": "conventional",
}
ARMS = (*SOURCES, "external")
UPDATING = ARMS[:4]
NEW_SOURCES = (
    "experiments/r4_integrated/fixed_rehearsal.py",
    "experiments/r4_integrated/rehearsal_state.py",
    "experiments/r4_integrated/utf8_action_bridge.py",
    "docs/research/fixed-rehearsal-plan.md",
)
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def streams(blocks):
    """Fixed 2,048-offer streams; replay allocation never depends on predictions."""
    paired = [item for b, a in zip(blocks["B"], blocks["A"], strict=True) for item in (b, a)]
    return {a: blocks["B"] * 256 if a == "cell_b_only" else paired * 128 for a in UPDATING}


def source_identity(revision, parent_files):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != revision:
        raise ValueError("exact frozen source revision required")
    files = {}
    for name in (*parent_files, *NEW_SOURCES):
        raw = (ROOT / name).read_bytes()
        if raw != subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT):
            raise ValueError("working source differs from frozen Git bytes: " + name)
        files[name] = sha256(raw).hexdigest()
    return {"revision": actual, "files": files}


def decision(report):
    result = {"complete": False, "qualified": False}
    if report["status"] != "completed" or report["replication"]["status"] != "passed":
        return result
    ids = report["blocks"]["A"] + report["blocks"]["B"]
    schedule = streams(report["blocks"])
    expected = [
        (arm, step + 1, schedule[arm][step])
        for offset in range(0, 2048, 128)
        for arm in UPDATING
        for step in range(offset, offset + 128)
    ]
    if [(r["arm"], r["new_offer"], r["id"]) for r in report["training"]] != expected:
        return result
    phases = ["initial", *(f"offers_{n:04d}" for n in range(128, 2049, 128))]
    if list(report["probes"]) != phases or set(report["motor_return"]) != set(ARMS):
        return result
    for phase in phases:
        if set(report["probes"][phase]) != set(ARMS) or any(
            [r["id"] for r in report["probes"][phase][a]] != ids for a in ARMS
        ):
            return result
    if any([r["id"] for r in report["motor_return"][a]] != ids for a in ARMS):
        return result
    checks, gains = {}, {}
    final = report["probes"]["offers_2048"]
    for arm, frozen in (
        ("cell_rehearsal", "cell_no_learning"),
        ("conventional_rehearsal", "conventional_no_learning"),
    ):
        rows, control = final[arm], report["probes"]["initial"][frozen]
        gain = sum(r["correct"] and r["supported"] for r in rows[8:]) - sum(
            r["correct"] and r["supported"] for r in control[8:]
        )
        gains[arm] = gain
        checks[arm] = {
            "A_retained": all(r["correct"] for r in rows[:8]),
            "B_acquired": all(r["correct"] for r in rows[8:]),
            "new_supported_B_gain": gain >= 1,
            "all_actions_and_STOPs": all(r["success"] for r in report["motor_return"][arm]),
            "no_collateral_edits": all(
                r["execution"]["decoys_unchanged"] for r in report["motor_return"][arm]
            ),
        }
    lost = [i for i, r in enumerate(final["cell_b_only"][:8]) if not r["correct"]]
    checks["cell_rehearsal"]["six_lost_A_cases_recovered"] = len(lost) == 6 and all(
        final["cell_rehearsal"][i]["correct"] for i in lost
    )
    return {
        "complete": True,
        "qualified": all(all(v.values()) for v in checks.values()),
        "checks": checks,
        "supported_B_gains": gains,
        "scope": "Exposed TRAIN supervised rehearsal; no adoption, fresh transfer or gate pass",
    }


def run(output, revision):
    began = time.perf_counter()
    report = {
        "status": "failed",
        "started_unix": time.time(),
        "training": [],
        "probes": {},
        "motor_return": {},
        "events": [],
        "first_complete_fit": {},
        "replication": {"status": "pending", "matched_offers": 0, "matched_probes": 0},
        "limits": {"seconds": 1200, "peak_parent_mib": 2048},
        "configuration": {
            "arms": ARMS,
            "updating": UPDATING,
            "new_offers_per_arm": 2048,
            "probe_every_new_offers": 128,
            "rate": 0.001,
            "clip": 1.0,
            "inherited_A_offers_per_parent": 2048,
            "replay_is_external_labeled_store": True,
        },
        "environment": {
            k: os.environ.get(k) for k in (*THREADS, "HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "python": {"executable": os.sys.executable, "version": os.sys.version},
        "finalization": "Execution status requires matching passed completion and launch receipts",
    }
    cores, arrays, final_predictions = {}, {}, {}
    work, memory_reader, error = None, None, None

    def guard(label):
        if memory_reader is not None:
            report["last_memory"] = {"label": label, **memory_reader()}
            if report["last_memory"]["peak_working_set_mib"] > 2048:
                raise MemoryError("fixed 2 GiB parent peak limit exceeded")
        if time.perf_counter() - began > 1200:
            raise TimeoutError("fixed 20 minute comparison budget exhausted")

    def event(name, **values):
        item = {"event": name, "elapsed_seconds": time.perf_counter() - began, **values}
        report["events"].append(item)
        if work is not None:
            with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(item, allow_nan=False), flush=True)

    def checkpoint(name):
        report.setdefault("checkpoints", {})[name] = {a: c.snapshot() for a, c in cores.items()}
        for arm in SOURCES:
            arrays.update(
                {f"{name}_{arm}_{k}": v for k, v in cores[arm].checkpoint_arrays().items()}
            )

    try:
        if output.exists() or output.with_suffix(".completion.json").exists():
            raise FileExistsError("one-shot output already exists")
        if any(os.environ.get(k) != "1" for k in THREADS):
            raise RuntimeError("set BLAS thread counts to 1 before Python")
        if (
            os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
            or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve()
            != ROOT / ".native-cache"
        ):
            raise RuntimeError("set CPU and checkout native cache before Python")
        import numpy as np

        from benchmarks.native_scaling import _process_memory
        from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
        from experiments.r4_integrated.neutral_document_python_session import Case
        from experiments.r4_integrated.rate_cell_session_support import retained_size
        from experiments.r4_integrated.real_input_action_bridge import ActionTask
        from experiments.r4_integrated.real_input_qualification import score_prediction
        from experiments.r4_integrated.rehearsal_state import load_state, restore_core
        from experiments.r4_integrated.utf8_action_bridge import execute_prediction, execute_utf8
        from humanbrain.native import loaded_build_info

        memory_reader = _process_memory
        guard("imports_complete")
        fixture, inputs, parent, archive, custody = load_state()
        report["custody"] = custody
        identity = report["source"] = source_identity(revision, parent["source"]["files"])
        report["blocks"] = fixture["blocks"]
        lessons = fixture["cases"]
        schedule = streams(fixture["blocks"])
        report["streams"] = {
            a: {
                "ids": v,
                "A_offers": sum(i in fixture["blocks"]["A"] for i in v),
                "B_offers": sum(i in fixture["blocks"]["B"] for i in v),
            }
            for a, v in schedule.items()
        }
        (ROOT / ".runtime").mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="fixed-rehearsal-", dir=ROOT / ".runtime"))
        report["workspace_root"] = str(work)
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to(ROOT / ".native-cache"):
            raise ValueError("native library lies outside checkout cache")
        report["native_motor"] = {
            "library": str(native.library_path),
            "source_digest": native.source_digest,
            "sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        cores = {a: restore_core(parent, archive, source_arm) for a, source_arm in SOURCES.items()}
        cores["external"] = cores["cell_no_learning"]
        checkpoint("initial")
        frozen_hashes = {a: cores[a].checkpoint_sha256 for a in ARMS[4:]}
        inherited = {a: c.offer_count for a, c in cores.items()}
        reference = [r for r in parent["training"] if r["block"] == "B"]
        if len(reference) != 2048:
            raise ValueError("original B-only trajectory is incomplete")
        event("parents_restored", identities={a: c.checkpoint_sha256 for a, c in cores.items()})

        def probe(name):
            before = {a: c.checkpoint_sha256 for a, c in cores.items()}
            rows = report["probes"][name] = {a: [] for a in ARMS}
            for lesson in lessons:
                query, inventory, gold = inputs[lesson["id"]]
                for arm, core in cores.items():
                    guard("probe:" + arm)
                    prediction = (core.external_only if arm == "external" else core.predict)(
                        query, inventory
                    )
                    row = score_prediction(prediction, lesson, inventory, gold)
                    rows[arm].append(row)
                    if name == "offers_2048":
                        final_predictions[(arm, lesson["id"])] = prediction
            if before != {a: c.checkpoint_sha256 for a, c in cores.items()}:
                raise RuntimeError("read-only probe changed a core")
            if name != "initial":
                offers = int(name.removeprefix("offers_"))
                if rows["cell_b_only"] != parent["probes"][f"B_{offers // 8:03d}"]["trainable"]:
                    raise RuntimeError("B-only probe differs from original trajectory")
                report["replication"]["matched_probes"] += 16
            for arm in UPDATING:
                if arm not in report["first_complete_fit"] and all(r["correct"] for r in rows[arm]):
                    report["first_complete_fit"][arm] = {
                        "probe": name,
                        "checkpoint": cores[arm].snapshot(),
                    }
            event(
                name,
                scores={
                    a: {
                        block: {
                            "supported": sum(r["correct"] and r["supported"] for r in v[i : i + 8]),
                            "STOP": sum(r["correct"] and not r["supported"] for r in v[i : i + 8]),
                        }
                        for block, i in (("A", 0), ("B", 8))
                    }
                    for a, v in rows.items()
                },
            )

        probe("initial")
        for offset in range(0, 2048, 128):
            for arm in UPDATING:
                for step in range(offset, offset + 128):
                    item = schedule[arm][step]
                    row = {"arm": arm, "new_offer": step + 1, "id": item}
                    report["training"].append(row)
                    guard("observe:" + arm)
                    query, inventory, gold = inputs[item]
                    row["receipt"] = cores[arm].observe(query, inventory, gold)
                    if arm == "cell_b_only":
                        if (
                            item != reference[step]["id"]
                            or row["receipt"] != reference[step]["offers"]["trainable"]
                        ):
                            raise RuntimeError("B-only update differs from saved continuation")
                        report["replication"]["matched_offers"] += 1
            probe(f"offers_{offset + 128:04d}")
        checkpoint("final")
        # Normalize tuple-valued configuration to the saved JSON representation.
        if json.dumps(cores["cell_b_only"].snapshot(), sort_keys=True) != json.dumps(
            parent["checkpoints"]["after_B"]["trainable"], sort_keys=True
        ):
            raise RuntimeError("B-only final complete checkpoint differs")
        if any(cores[a].offer_count - inherited[a] != 2048 for a in UPDATING):
            raise RuntimeError("fixed new-offer count differs")
        if frozen_hashes != {a: cores[a].checkpoint_sha256 for a in frozen_hashes}:
            raise RuntimeError("no-further-learning control changed")
        report["replication"].update(
            status="passed", final_checkpoint=cores["cell_b_only"].checkpoint_sha256
        )
        final_hashes = {a: c.checkpoint_sha256 for a, c in cores.items()}
        motor = ContextualOccurrenceMotor()
        report["motor_demonstrations"] = []
        for case in (
            Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
            Case("motor-b", "bora", "ready", "motor_beta", "warm"),
        ):
            guard("motor_demonstration")
            before, after = case.files(), case.files(corrected=True)
            execution = execute_utf8(case, after, before, work / case.name)
            report["motor_demonstrations"].append({"case": case.name, "execution": execution})
            if not execution["passed"]:
                raise RuntimeError("original motor demonstration failed UTF8 execution")
            motor.learn(f"Configure {case.key}".encode(), before, after)
        report["motor"] = {
            "checkpoint_sha256": motor.checkpoint_sha256,
            "demonstration_count": motor.demonstration_count,
            "retained_demonstration_event_bytes": motor.retained_demonstration_event_bytes,
        }
        if motor.checkpoint_sha256 != parent["motor"]["checkpoint_sha256"]:
            raise RuntimeError("original two-demo motor state differs")
        for arm in ARMS:
            rows = report["motor_return"][arm] = []
            for ordinal, lesson in enumerate(lessons):
                guard("motor:" + arm)
                query, inventory, gold = inputs[lesson["id"]]
                task = ActionTask(
                    f"{arm}-{ordinal:03d}",
                    f"Configure qa{ordinal:03d}".encode(),
                    f"qa{ordinal:03d}",
                    f"qa_config{ordinal:03d}",
                    "pending",
                    None if gold is None else bytes.fromhex(lesson["accepted_text_hex"][0]),
                )
                row = execute_prediction(
                    final_predictions[(arm, lesson["id"])],
                    query,
                    inventory,
                    motor,
                    task,
                    work / task.name,
                    reader=cores[arm],
                )
                rows.append({"id": lesson["id"], **row})
            event(
                "motor_" + arm,
                supported=sum(r["success"] and r["expected_answer_hex"] is not None for r in rows),
                STOP=sum(r["success"] and r["expected_answer_hex"] is None for r in rows),
            )
        if final_hashes != {a: c.checkpoint_sha256 for a, c in cores.items()}:
            raise RuntimeError("action changed learned reader state")
        if source_identity(revision, parent["source"]["files"]) != identity:
            raise RuntimeError("source changed during comparison")
        if sha256(native.library_path.read_bytes()).hexdigest() != report["native_motor"]["sha256"]:
            raise RuntimeError("native binary changed during comparison")
        report["retained_learner_input_motor_bytes"] = retained_size(cores, inputs, motor)
        report["retained_source_label_store_bytes"] = retained_size(fixture)
        report["retained_experiment_bytes"] = retained_size(
            cores, inputs, motor, fixture, arrays, final_predictions, parent, archive, report
        )
        guard("scientific_work_complete")
        report["status"] = "completed"
    except Exception as exc:
        error = exc
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["elapsed_before_artifacts_seconds"] = time.perf_counter() - began
        finalization_errors = report["finalization_errors"] = []
        try:
            if report["status"] != "completed" and cores:
                checkpoint("partial")
        except Exception as exc:
            finalization_errors.append({"stage": "partial_checkpoint", "error": str(exc)})
            error = error or exc
        try:
            if work is not None and arrays:
                archive_path = work / "checkpoints.npz"
                with archive_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                report["checkpoint_archive"] = {
                    "path": str(archive_path),
                    "sha256": sha256(archive_path.read_bytes()).hexdigest(),
                    "bytes": archive_path.stat().st_size,
                }
        except Exception as exc:
            finalization_errors.append({"stage": "checkpoint_archive", "error": str(exc)})
            error = error or exc
        if finalization_errors:
            report["status"] = "failed"
        report["decision"] = decision(report)
        report_hash = None
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, allow_nan=False, separators=(",", ":"))
                stream.write("\n")
            report_hash = sha256(output.read_bytes()).hexdigest()
            guard("artifacts_written")
        except Exception as exc:
            finalization_errors.append({"stage": "report_or_final_budget", "error": str(exc)})
            error = error or exc
        try:
            completion = {
                "status": "passed"
                if report["status"] == "completed" and error is None
                else "failed",
                "report_sha256": report_hash,
                "elapsed_seconds_including_artifacts": time.perf_counter() - began,
                "last_memory": report.get("last_memory"),
                "finalization_errors": finalization_errors,
                "excludes": "Python bootstrap and completion receipt's own write",
            }
            with output.with_suffix(".completion.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(completion, stream, allow_nan=False)
                stream.write("\n")
            print(
                json.dumps({"completion": completion, "decision": report["decision"]}), flush=True
            )
        except Exception as exc:
            error = error or exc
            print(json.dumps({"finalization_error": str(exc)}), flush=True)
    if error is not None:
        raise RuntimeError(
            "comparison failed; retain partial artifacts and launch output"
        ) from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.revision)
