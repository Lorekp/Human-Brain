"""One persistent public-source session with untaught later Python actions."""

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
PARENTS = {
    "cell_update": "cell_rehearsal",
    "cell_frozen": "cell_rehearsal",
    "conventional_update": "conventional_rehearsal",
    "conventional_frozen": "conventional_rehearsal",
    "frozen_producer_update": "frozen_producer_rehearsal",
}
ARMS = (*PARENTS, "external")
UPDATING = ("cell_update", "conventional_update", "frozen_producer_update")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
NEW_SOURCES = (
    "experiments/r4_integrated/fresh_source_session.py",
    "experiments/r4_integrated/fresh_source_inputs.py",
    "experiments/r4_integrated/fresh_source_state.py",
    "experiments/r4_integrated/fresh_source_actions.py",
    "experiments/r4_integrated/fresh_source_cases_v1.json",
    "benchmarks/fresh_source_preparation_v1.json",
    "benchmarks/fresh_source_aborted_preparation_v1.json",
    "docs/research/fresh-source-action-plan.md",
)
FIXTURE_SHA256 = "6b78781c78c1d7b6d2474cbd8185123f9349e682e2ecde900ed2a852fbf17faf"


def diagnostic_ids(blocks):
    return {blocks["E"][3], blocks["L"][3]}


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


def schedule(blocks):
    """Actions and teaching have fixed ordering, independent of predicted outcomes."""
    return {
        "initial": blocks["O"],
        **{f"episode_{i}": [item] for i, item in enumerate(blocks["E"])},
        "late": blocks["L"],
        "final": blocks["O"] + blocks["E"],
    }


def decision(report):
    result = {"complete": False, "qualified": False}
    if report["status"] != "completed":
        return result
    blocks = report["blocks"]
    roster = schedule(blocks)
    if list(report["actions"]) != list(roster) or list(report["oracle"]) != list(roster):
        return result
    for phase, ids in roster.items():
        for lane in ("actions", "oracle"):
            rows = report[lane][phase]
            if list(rows) != list(ARMS) or any([r["id"] for r in rows[a]] != ids for a in ARMS):
                return result
            for arm_rows in rows.values():
                for row in arm_rows:
                    scored = row["id"] not in diagnostic_ids(blocks)
                    verdict = row["success"] if lane == "actions" else row["correct"]
                    if row["semantic_scored"] != scored or (verdict is None) == scored:
                        return result
    expected = [
        (i, arm, kind, item)
        for i, fresh in enumerate(blocks["E"])
        for arm in UPDATING
        for kind, item in (
            ("feedback", fresh),
            ("replay", blocks["O"][2 * i]),
            ("replay", blocks["O"][2 * i + 1]),
        )
        if kind != "feedback" or fresh not in diagnostic_ids(blocks)
    ]
    if [(r["episode"], r["arm"], r["kind"], r["id"]) for r in report["training"]] != expected:
        return result
    required_teachers = [
        (i, arm, item)
        for i, item in enumerate(blocks["E"])
        for arm in UPDATING
        if report["actions"][f"episode_{i}"][arm][0]["success"] is False
    ]
    if [(r["episode"], r["arm"], r["id"]) for r in report["teachers"]] != required_teachers:
        return result
    if any(
        not r["passed"]
        or r["counts_as_learner_success"] is not False
        or r["teacher_executions"] != 1
        for r in report["teachers"]
    ):
        return result
    summaries = {}
    for arm, frozen in (
        ("cell_update", "cell_frozen"),
        ("conventional_update", "conventional_frozen"),
    ):
        fresh = [report["actions"][f"episode_{i}"][arm][0] for i in range(8)]
        late = report["actions"]["late"][arm]
        fresh += late
        old = report["actions"]["final"][arm][:16]
        supported_late = sum(r["success"] is True and r["supported"] for r in late)
        frozen_late = sum(
            r["success"] is True and r["supported"] for r in report["actions"]["late"][frozen]
        )
        summaries[arm] = {
            "fresh_supported": sum(r["success"] is True and r["supported"] for r in fresh),
            "fresh_STOP": sum(r["success"] is True and not r["supported"] for r in fresh),
            "late_supported": supported_late,
            "late_STOP": sum(r["success"] is True and not r["supported"] for r in late),
            "late_supported_gain": supported_late - frozen_late,
            "frozen_late_supported_ceiling": frozen_late == 2,
            "old_retained": sum(r["success"] for r in old),
            "initial_old_correct": sum(r["success"] for r in report["actions"]["initial"][arm]),
        }
    c = summaries["cell_update"]
    checks = {
        "fresh_supported_floor": c["fresh_supported"] >= 6,
        "fresh_STOP_floor": c["fresh_STOP"] == 2,
        "all_late_supported": c["late_supported"] == 2,
        "all_late_STOP": c["late_STOP"] == 1,
        "later_supported_gain": c["late_supported_gain"] >= 1,
        "old_retained": c["old_retained"] == 16,
        "decoys_unchanged": all(
            r["execution"]["decoys_unchanged"]
            for rows in report["actions"].values()
            for arm_rows in rows.values()
            for r in arm_rows
        ),
    }
    return {
        "complete": True,
        "qualified": all(checks.values()),
        "checks": checks,
        "summaries": summaries,
        "gain_interpretation": "inconclusive_at_frozen_ceiling"
        if c["frozen_late_supported_ceiling"]
        else "matched_late_first_attempt_difference",
        "scope": "Public development with retained sources; no source-absent memory or gate pass",
    }


def run(output, revision):
    began = time.perf_counter()
    report = {
        "status": "failed",
        "started_unix": time.time(),
        "training": [],
        "actions": {},
        "oracle": {},
        "teachers": [],
        "events": [],
        "inventories": [],
        "encoded_inputs": [],
        "limits": {"seconds": 1200, "peak_parent_mib": 2048},
        "configuration": {
            "arms": ARMS,
            "updating": UPDATING,
            "new_offers_per_arm": 23,
            "primary_predictions": 312,
            "oracle_predictions": 312,
            "maximum_teacher_executions": 21,
            "diagnostic_only_slots": ["E3", "L3"],
            "scorable_fresh_STOPs": 2,
            "rate": 0.001,
            "clip": 1.0,
            "replay_is_external_labeled_store": True,
        },
        "environment": {
            k: os.environ.get(k) for k in (*THREADS, "HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "python": {"executable": os.sys.executable, "version": os.sys.version},
        "finalization": "Execution status requires passed matching completion and launch receipts",
        "record_hash_scope": (
            "Adapter record hashes exclude runner envelope fields id/supported/reading for actions "
            "and episode/arm/id for teachers; preserved teacher first_attempt is the original "
            "adapter record"
        ),
    }
    cores, arrays = {}, {}
    work, memory_reader, error = None, None, None

    def guard(label):
        if memory_reader is not None:
            report["last_memory"] = {"label": label, **memory_reader()}
            if report["last_memory"]["peak_working_set_mib"] > 2048:
                raise MemoryError("fixed 2 GiB parent peak limit exceeded")
        if time.perf_counter() - began > 1200:
            raise TimeoutError("fixed 20 minute session budget exhausted")

    def event(name, **values):
        item = {"event": name, "elapsed_seconds": time.perf_counter() - began, **values}
        report["events"].append(item)
        if work is not None:
            with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(item, allow_nan=False), flush=True)

    def checkpoint(name):
        report.setdefault("checkpoints", {})[name] = {a: c.snapshot() for a, c in cores.items()}
        for arm in PARENTS:
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
        from experiments.r4_integrated.fresh_source_actions import (
            execute_prediction,
            execute_teacher,
        )
        from experiments.r4_integrated.fresh_source_inputs import prepare_inputs
        from experiments.r4_integrated.fresh_source_state import (
            SourcePool,
            load_state,
            restore_core,
        )
        from experiments.r4_integrated.neutral_document_python_session import Case
        from experiments.r4_integrated.rate_cell_session_support import retained_size
        from experiments.r4_integrated.real_input_action_bridge import ActionTask
        from experiments.r4_integrated.real_input_qualification import score_prediction
        from experiments.r4_integrated.utf8_action_bridge import execute_utf8
        from humanbrain.frozen_context_encoder import FrozenContextEncoder
        from humanbrain.native import loaded_build_info
        from humanbrain.rate_cell_reader import TokenInput

        memory_reader = _process_memory
        guard("imports_complete")
        old_fixture, old_inputs, parent, archive, custody = load_state()
        fixture_raw = (ROOT / "experiments/r4_integrated/fresh_source_cases_v1.json").read_bytes()
        fixture = json.loads(fixture_raw)
        if sha256(fixture_raw).hexdigest() != FIXTURE_SHA256 or fixture != prepare_inputs():
            raise ValueError("unchanged fixed cohort custody differs")
        if fixture["status"] != "aborted_semantic_review" or fixture["admission"]["allowed"]:
            raise ValueError("original failed admission must remain unchanged")
        report["admission"] = {
            "original": fixture["admission"],
            "revision": "same_inputs_E3_L3_diagnostic_only",
            "diagnostic_ids": sorted(diagnostic_ids(fixture["blocks"])),
            "scope": "Separate contract; original cohort did not gain admission",
        }
        report["custody"] = custody
        report["input_custody"] = fixture["source_custody"]
        identity = report["source"] = source_identity(revision, parent["source"]["files"])
        report["blocks"] = fixture["blocks"]
        lessons = {x["id"]: x for x in fixture["cases"]}
        ids = fixture["blocks"]["O"] + fixture["blocks"]["E"] + fixture["blocks"]["L"]
        if len(ids) != 28 or len(set(ids)) != 28 or list(lessons) != ids:
            raise ValueError("exact original/fresh/late roster required")
        (ROOT / ".runtime").mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="fresh-source-", dir=ROOT / ".runtime"))
        report["workspace_root"] = str(work)
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to(ROOT / ".native-cache"):
            raise ValueError("native library lies outside checkout cache")
        report["native_motor"] = {
            "library": str(native.library_path),
            "source_digest": native.source_digest,
            "sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        cores = {a: restore_core(parent, archive, p) for a, p in PARENTS.items()}
        cores["external"] = cores["cell_frozen"]
        checkpoint("initial")
        frozen = {a: cores[a].checkpoint_sha256 for a in ("cell_frozen", "conventional_frozen")}
        inherited = {a: c.offer_count for a, c in cores.items()}
        pool = SourcePool(old_fixture, old_inputs, expected_order=fixture["document_order"])
        report["inventories"].append(pool.manifest())
        encoder = FrozenContextEncoder(ROOT / ".models/all-MiniLM-L6-v2", max_tokens=256, threads=1)
        report["encoder"] = {"sha256": encoder.sha256, "metadata": encoder.metadata}
        if {v[0].encoder_sha256 for v in old_inputs.values()} != {encoder.sha256}:
            raise ValueError("encoder differs from authenticated old representations")
        fresh_tokens = {}
        for item in ids[16:]:
            lesson = lessons[item]
            doc = fixture["documents"][lesson["paragraph_id"]]
            for kind, text in (("document", doc["context"]), ("query", lesson["question"])):
                guard("encode:" + kind)
                raw = text.encode("utf-8")
                encoding = encoder.encode(raw)
                tokens = TokenInput.from_encoding(raw, encoding)
                key = f"{item}_{kind}"
                fresh_tokens[key] = tokens
                arrays["input_" + key] = tokens.vectors
                report["encoded_inputs"].append(
                    {
                        "id": item,
                        "kind": kind,
                        "array": "input_" + key,
                        "token_count": encoding.token_count,
                        **tokens.record(),
                    }
                )
        del encoding, tokens, encoder
        event("inputs_ready", fresh_encoder_calls=len(report["encoded_inputs"]))

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
                raise RuntimeError("original motor demonstration failed")
            motor.learn(f"Configure {case.key}".encode(), before, after)
        report["motor"] = {
            "checkpoint_sha256": motor.checkpoint_sha256,
            "demonstration_count": motor.demonstration_count,
            "retained_demonstration_event_bytes": motor.retained_demonstration_event_bytes,
        }
        if motor.checkpoint_sha256 != parent["motor"]["checkpoint_sha256"]:
            raise RuntimeError("original two-demo motor differs")

        def task_for(phase, arm, item):
            ordinal = ids.index(item)
            lesson = lessons[item]
            return ActionTask(
                f"{phase}-{arm}-{ordinal:03d}",
                f"Configure qa{ordinal:03d}".encode(),
                f"qa{ordinal:03d}",
                f"qa_config{ordinal:03d}",
                "pending",
                None if lesson["is_impossible"] else bytes.fromhex(lesson["accepted_text_hex"][0]),
            )

        def add(item):
            lesson = lessons[item]
            pool.add_document(
                lesson["paragraph_id"],
                fixture["documents"][lesson["paragraph_id"]],
                fresh_tokens[item + "_document"],
            )
            pool.add_query(lesson, fresh_tokens[item + "_query"])

        def probe(phase, items):
            before = {a: c.checkpoint_sha256 for a, c in cores.items()}
            actual = report["actions"][phase] = {a: [] for a in ARMS}
            oracle_rows = report["oracle"][phase] = {a: [] for a in ARMS}
            for item in items:
                lesson = lessons[item]
                query, inventory, gold = pool.inputs(lesson)
                _, oracle_inventory, oracle_gold = pool.oracle(lesson)
                for arm, core in cores.items():
                    guard("primary:" + phase + ":" + arm)
                    predict = core.external_only if arm == "external" else core.predict
                    prediction = predict(query, inventory)
                    reading = score_prediction(prediction, lesson, inventory, gold)
                    reading["correct"] = (
                        prediction.stopped
                        if gold is None
                        else (prediction.candidate_index in gold and not prediction.stopped)
                    )
                    scored = item not in diagnostic_ids(fixture["blocks"])
                    reading["official_label_agreement"] = reading["correct"]
                    reading["semantic_scored"] = scored
                    if not scored:
                        reading["correct"] = None
                    task = task_for(phase, arm, item)
                    row = execute_prediction(
                        prediction,
                        query,
                        inventory,
                        motor,
                        task,
                        work / task.name,
                        reader=core,
                        accepted_gold_indices=gold,
                        semantic_scored=scored,
                    )
                    actual[arm].append(
                        {"id": item, "supported": gold is not None, "reading": reading, **row}
                    )
                    guard("oracle:" + phase + ":" + arm)
                    reference = predict(query, oracle_inventory)
                    oracle_row = score_prediction(reference, lesson, oracle_inventory, oracle_gold)
                    oracle_row["correct"] = (
                        reference.stopped
                        if oracle_gold is None
                        else (reference.candidate_index in oracle_gold and not reference.stopped)
                    )
                    oracle_row["official_label_agreement"] = oracle_row["correct"]
                    oracle_row["semantic_scored"] = scored
                    if not scored:
                        oracle_row["correct"] = None
                    oracle_rows[arm].append(oracle_row)
            if before != {a: c.checkpoint_sha256 for a, c in cores.items()}:
                raise RuntimeError("probe/action/oracle changed reader state")
            event(
                phase,
                documents=pool.document_count,
                scores={
                    a: {
                        "supported": sum(r["success"] is True and r["supported"] for r in rows),
                        "STOP": sum(r["success"] is True and not r["supported"] for r in rows),
                        "unscored": sum(r["success"] is None for r in rows),
                    }
                    for a, rows in actual.items()
                },
            )

        blocks = fixture["blocks"]
        probe("initial", blocks["O"])
        for i, item in enumerate(blocks["E"]):
            add(item)
            report["inventories"].append(pool.manifest())
            phase = f"episode_{i}"
            probe(phase, [item])
            for arm in UPDATING:
                first = report["actions"][phase][arm][0]
                if first["success"] is False:
                    query, inventory, gold = pool.inputs(lessons[item])
                    guard("teacher:" + arm)
                    row = execute_teacher(
                        {k: v for k, v in first.items() if k not in ("id", "supported", "reading")},
                        task_for(phase, arm, item),
                        query,
                        inventory,
                        reader=cores[arm],
                        motor=motor,
                        accepted_gold_indices=gold,
                    )
                    report["teachers"].append({"episode": i, "arm": arm, "id": item, **row})
                    if not row["passed"]:
                        raise RuntimeError(
                            "teacher correction failed; no feedback offer authorized"
                        )
                for kind, target in (
                    ("feedback", item),
                    ("replay", blocks["O"][2 * i]),
                    ("replay", blocks["O"][2 * i + 1]),
                ):
                    if kind == "feedback" and item in diagnostic_ids(blocks):
                        continue
                    guard("observe:" + arm)
                    query, inventory, gold = pool.inputs(lessons[target])
                    row = {
                        "episode": i,
                        "arm": arm,
                        "kind": kind,
                        "id": target,
                        "inventory_sha256": inventory.sha256,
                    }
                    report["training"].append(row)
                    row["receipt"] = cores[arm].observe(query, inventory, gold)
            if frozen != {a: cores[a].checkpoint_sha256 for a in frozen}:
                raise RuntimeError("learning-disabled state changed")
            event("feedback_complete", episode=i)
        for item in blocks["L"]:
            add(item)
        report["inventories"].append(pool.manifest())
        probe("late", blocks["L"])
        probe("final", blocks["O"] + blocks["E"])
        checkpoint("final")
        if any(cores[a].offer_count - inherited[a] != 23 for a in UPDATING):
            raise RuntimeError("fixed new-offer budget differs")
        if frozen != {a: cores[a].checkpoint_sha256 for a in frozen}:
            raise RuntimeError("learning-disabled state changed")
        if motor.checkpoint_sha256 != report["motor"]["checkpoint_sha256"]:
            raise RuntimeError("motor changed after demonstrations")
        if source_identity(revision, parent["source"]["files"]) != identity:
            raise RuntimeError("source changed during session")
        if sha256(native.library_path.read_bytes()).hexdigest() != report["native_motor"]["sha256"]:
            raise RuntimeError("native binary changed during session")
        report["retained_learner_input_motor_bytes"] = retained_size(
            cores, old_inputs, fresh_tokens, pool, motor
        )
        report["retained_source_label_store_bytes"] = retained_size(fixture, old_fixture)
        report["source_storage"] = pool.storage_summary()
        report["retained_experiment_bytes"] = retained_size(
            cores,
            old_inputs,
            fresh_tokens,
            pool,
            motor,
            fixture,
            old_fixture,
            arrays,
            parent,
            archive,
            report,
        )
        guard("scientific_work_complete")
        report["status"] = "completed"
        if not decision(report)["complete"]:
            report["status"] = "failed"
            raise RuntimeError("completed session violates its fixed evidence roster")
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
            "session failed; preserve partial artifacts and launch output"
        ) from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.revision)
