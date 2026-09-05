"""One fixed actual-language fit, retention, and executed-action qualification."""

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
FIXTURE = "experiments/r4_integrated/real_input_qualification_cases_v1.json"
FIXTURE_HASH = "aa988986d8f04d7e796e485ccea33f24c96cadad09f03779de844905ee5ad41b"
ADDED_SOURCES = (
    FIXTURE,
    "experiments/r4_integrated/real_input_qualification.py",
    "experiments/r4_integrated/real_input_action_bridge.py",
    "experiments/r4_integrated/real_input_numerical_check.py",
    "docs/research/real-input-qualification-plan.md",
)
ARMS = ("trainable", "frozen_producer", "frozen", "conventional", "conventional_frozen", "external")
UPDATING = ("trainable", "frozen_producer", "conventional")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def score_prediction(prediction, lesson, inventory, gold):
    """Score a previously selected answer without another inference call."""
    import numpy as np

    candidate = (
        None
        if prediction.candidate_index is None
        else inventory.candidates[prediction.candidate_index]
    )
    emitted = None if candidate is None else candidate.emitted.hex()
    scores = prediction.span_scores
    best_gold = None if gold is None else float(scores[list(gold)].max())
    return {
        "id": lesson["id"],
        "supported": not lesson["is_impossible"],
        "correct": prediction.stopped if gold is None else emitted in lesson["accepted_text_hex"],
        "emitted_hex": emitted,
        "stopped": prediction.stopped,
        "candidate_index": prediction.candidate_index,
        "candidate_count": len(scores),
        "query_sha256": prediction.query_sha256,
        "inventory_sha256": prediction.inventory_sha256,
        "checkpoint_sha256": prediction.checkpoint_sha256,
        "p_stop": prediction.p_stop,
        "stop_logit": prediction.stop_logit,
        "stop_reason": prediction.stop_reason,
        "span_scores_sha256": sha256(scores.tobytes()).hexdigest(),
        "best_gold_rank": None if gold is None else int(1 + np.count_nonzero(scores > best_gold)),
        "conditional_best_is_gold": None
        if gold is None
        else bool(best_gold == scores.max() and np.count_nonzero(scores == best_gold) == 1),
    }


def decision(report):
    """Require complete fixed rosters before interpreting fit or retention."""
    result = {"complete": False, "qualified": False}
    if report["status"] != "completed":
        return result
    ids = report["blocks"]["A"] + report["blocks"]["B"]
    expected = {"initial": ids}
    for block in ("A", "B"):
        expected.update(
            {f"{block}_{p:03d}": ids[:8] if block == "A" else ids for p in range(16, 257, 16)}
        )
    if set(report["probes"]) != set(expected):
        return result
    for name, roster in expected.items():
        if set(report["probes"][name]) != set(ARMS) or any(
            [r["id"] for r in report["probes"][name][a]] != roster for a in ARMS
        ):
            return result
    teaching_roster = [
        (block, p, item)
        for block in ("A", "B")
        for p in range(1, 257)
        for item in report["blocks"][block]
    ]
    if [(r["block"], r["pass"], r["id"]) for r in report["training"]] != teaching_roster or any(
        set(r["offers"]) != set(UPDATING) for r in report["training"]
    ):
        return result
    if set(report["motor_return"]) != set(ARMS) or any(
        [r["id"] for r in report["motor_return"][a]] != ids for a in ARMS
    ):
        return result
    checks, interpretation = {}, {}
    for arm, frozen in (("trainable", "frozen"), ("conventional", "conventional_frozen")):
        early, final = report["probes"]["A_256"][arm], report["probes"]["B_256"][arm]
        initial = report["probes"]["initial"][frozen]
        early_fit, final_fit = all(r["correct"] for r in early), all(r["correct"] for r in final)
        gained = sum(r["supported"] and r["correct"] for r in final) - sum(
            r["supported"] and r["correct"] for r in initial
        )
        motor = report["motor_return"][arm]
        checks[arm] = {
            "A_fit": early_fit,
            "AB_fit_retained": final_fit,
            "supported_gain_at_least_four": gained >= 4,
            "all_executed_actions_and_STOPs": all(r["success"] for r in motor),
            "all_decoys_unchanged": all(r["execution"]["decoys_unchanged"] for r in motor),
        }
        interpretation[arm] = {
            "supported_gain": gained,
            "A_errors_after_B": sum(not r["correct"] for r in final[:8]),
            "forgetting_of_fitted_A": early_fit and any(not r["correct"] for r in final[:8]),
            "A_acquisition_failed": not early_fit,
        }
    return {
        "complete": True,
        "qualified": all(all(c.values()) for c in checks.values()),
        "checks": checks,
        "interpretation": interpretation,
        "scope": "Exposed TRAIN engineering qualification; no architecture adoption or gate pass",
    }


def run(output, revision):
    started, started_unix = time.perf_counter(), time.time()
    report = {
        "status": "failed",
        "started_unix": started_unix,
        "training": [],
        "probes": {},
        "motor_return": {},
        "events": [],
        "first_complete_fit": {},
        "limits": {"seconds": 1200, "peak_parent_mib": 2048},
        "configuration": {
            "seed": 0,
            "passes_per_block": 256,
            "probe_every": 16,
            "learning_rate": 0.001,
            "gradient_clip_norm": 1.0,
            "updating_arms": UPDATING,
            "arms": ARMS,
            "additional_replay": False,
        },
        "environment": {
            k: os.environ.get(k) for k in (*THREADS, "HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "python": {"executable": os.sys.executable, "version": os.sys.version},
        "scope": "Supervised complete-paragraph public TRAIN; external encoder vectors retained",
        "finalization": "Requires passed completion receipt; status is execution, not capability",
    }
    arrays, cores, inputs, predictions = {}, {}, {}, {}
    work = None
    memory_reader = None
    error = None

    def guard(label):
        if memory_reader is not None:
            report["last_memory"] = {"label": label, **memory_reader()}
            if report["last_memory"]["peak_working_set_mib"] > 2048:
                raise MemoryError("fixed 2 GiB parent peak limit exceeded")
        if time.perf_counter() - started > 1200:
            raise TimeoutError("fixed 20 minute diagnostic budget exhausted")

    def event(name, **values):
        item = {"event": name, "elapsed_seconds": time.perf_counter() - started, **values}
        report["events"].append(item)
        if work is not None:
            with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(item, allow_nan=False), flush=True)

    def checkpoint(name):
        report.setdefault("checkpoints", {})[name] = {a: c.snapshot() for a, c in cores.items()}
        for arm in ARMS[:-1]:
            arrays.update(
                {f"{name}_{arm}_{k}": v for k, v in cores[arm].checkpoint_arrays().items()}
            )

    try:
        if output.exists() or output.with_suffix(".completion.json").exists():
            raise FileExistsError("one-shot output already exists")
        if any(os.environ.get(k) != "1" for k in THREADS):
            raise RuntimeError("set all three BLAS thread counts to 1 before Python")
        if (
            os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
            or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve()
            != ROOT / ".native-cache"
        ):
            raise RuntimeError("set CPU and checkout native cache before Python")
        # Imports that may build/load native code are inside the failure receipt boundary.
        import numpy as np

        from benchmarks.native_scaling import _process_memory
        from experiments.r4_integrated.contextual_pairwise_python_session import _motor
        from experiments.r4_integrated.public_qa_acquisition import (
            gold_indices,
            new_cores,
            qa_summary,
            source_identity,
        )
        from experiments.r4_integrated.rate_cell_session_support import retained_size
        from experiments.r4_integrated.real_input_action_bridge import (
            ActionTask,
            execute_prediction,
        )
        from experiments.r4_integrated.real_input_numerical_check import run_numerical_check
        from humanbrain.conventional_attention_reader import ConventionalAttentionReader
        from humanbrain.native import loaded_build_info
        from humanbrain.rate_cell_reader import SpanInventory, TokenDocument, TokenInput

        memory_reader = _process_memory
        guard("imports_complete")
        identity = source_identity(revision)
        for name in ADDED_SOURCES:
            raw = (ROOT / name).read_bytes()
            if raw != subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT):
                raise ValueError("new source differs from frozen Git bytes: " + name)
            identity["files"][name] = sha256(raw).hexdigest()
        report["source"] = identity
        if sha256((ROOT / FIXTURE).read_bytes()).hexdigest() != FIXTURE_HASH:
            raise ValueError("selected actual-input fixture differs")
        fixture = json.loads((ROOT / FIXTURE).read_bytes())
        report["fixture"] = {
            "path": FIXTURE,
            "sha256": FIXTURE_HASH,
            "custody": fixture["source_custody"],
        }
        report["blocks"] = fixture["blocks"]
        lessons = fixture["cases"]
        ids = fixture["blocks"]["A"] + fixture["blocks"]["B"]
        if (
            [r["id"] for r in lessons] != ids
            or len(set(ids)) != 16
            or any(sum(r["is_impossible"] for r in lessons[i : i + 8]) != 2 for i in (0, 8))
        ):
            raise ValueError("fixed 6 supported/2 STOP block roster differs")
        (ROOT / ".runtime").mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="real-input-", dir=ROOT / ".runtime"))
        report["workspace_root"] = str(work)
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to(ROOT / ".native-cache"):
            raise ValueError("native library lies outside checkout cache")
        report["native_motor"] = {
            "library": str(native.library_path),
            "source_digest": native.source_digest,
            "sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        encoded = fixture["encoded_inputs"]
        archive = ROOT / encoded["path"]
        if sha256(archive.read_bytes()).hexdigest() != encoded["sha256"]:
            raise ValueError("saved vector archive differs")
        cache = {}
        with np.load(archive, allow_pickle=False) as saved:
            if set(saved.files) != {r["array"] for r in fixture["saved_inputs"]}:
                raise ValueError("saved vector roster differs")
            for row in fixture["saved_inputs"]:
                tokens = TokenInput(
                    bytes.fromhex(row["raw_hex"]),
                    tuple(map(tuple, row["byte_spans"])),
                    saved[row["array"]],
                    row["encoder_sha256"],
                )
                if (
                    tokens.sha256 != row["sha256"]
                    or sha256(tokens.vectors.tobytes()).hexdigest() != row["vectors_sha256"]
                ):
                    raise ValueError("restored immutable TokenInput differs")
                cache[row["array"]] = tokens
        report["saved_inputs"] = encoded
        report["vector_bytes"] = sum(v.vectors.nbytes for v in cache.values())
        if report["vector_bytes"] != 6021120:
            raise ValueError("saved vector payload differs")
        for lesson in lessons:
            link, doc = (
                fixture["input_links"][lesson["id"]],
                fixture["documents"][lesson["paragraph_id"]],
            )
            query, tokens = cache[link["query_array"]], cache[link["document_array"]]
            if query.raw != lesson["question"].encode() or tokens.raw != doc["context"].encode():
                raise ValueError("original complete question/paragraph differs")
            inventory = SpanInventory(
                (
                    TokenDocument(
                        lesson["paragraph_id"].encode(), doc["article_sha256"].encode(), tokens
                    ),
                )
            )
            if inventory.sha256 != link["original_paragraph_inventory_sha256"]:
                raise ValueError("original paragraph competition differs")
            inputs[lesson["id"]] = (query, inventory, gold_indices(lesson, inventory))
            if not lesson["is_impossible"] and len(lesson["accepted_text_hex"]) != 1:
                raise ValueError("fixed motor fixture expects one official accepted value")
        event("inputs_authenticated", inputs=len(cache), lessons=len(lessons))
        supported = next(inputs[r["id"]] for r in lessons if not r["is_impossible"])
        unsupported = next(inputs[r["id"]] for r in lessons if r["is_impossible"])
        report["numerical"] = run_numerical_check(supported, unsupported, guard=guard)
        if report["numerical"]["status"] != "passed":
            raise RuntimeError("actual-input numerical admission failed")
        event("numerical_admitted", counts=report["numerical"]["counts"])
        cores = new_cores()
        cores["conventional_frozen"] = ConventionalAttentionReader(mode="frozen")
        cores = {a: cores[a] for a in ARMS}
        for k, value in cores["conventional"].checkpoint_arrays().items():
            if not np.array_equal(value, cores["conventional_frozen"].checkpoint_arrays()[k]):
                raise ValueError("conventional frozen initialization differs")
        checkpoint("initial")
        frozen_hashes = {
            a: cores[a].checkpoint_sha256 for a in ("frozen", "conventional_frozen", "external")
        }

        def probe(name, cohort):
            before = {a: c.checkpoint_sha256 for a, c in cores.items()}
            rows = report["probes"][name] = {a: [] for a in ARMS}
            for lesson in cohort:
                query, inventory, gold = inputs[lesson["id"]]
                for arm, core in cores.items():
                    guard("probe:" + name + ":" + arm)
                    prediction = (core.external_only if arm == "external" else core.predict)(
                        query, inventory
                    )
                    rows[arm].append(score_prediction(prediction, lesson, inventory, gold))
                    if name == "B_256":
                        predictions[(arm, lesson["id"])] = prediction
            if before != {a: c.checkpoint_sha256 for a, c in cores.items()}:
                raise RuntimeError("read-only probe changed a checkpoint")
            summaries = {
                a: qa_summary(
                    v,
                    sum(not r["is_impossible"] for r in cohort),
                    sum(r["is_impossible"] for r in cohort),
                )
                for a, v in rows.items()
            }
            if name != "initial":
                for arm in UPDATING:
                    key = name[0] + ":" + arm
                    if key not in report["first_complete_fit"] and all(
                        r["correct"] for r in rows[arm]
                    ):
                        report["first_complete_fit"][key] = {
                            "probe": name,
                            "cohort": "A" if name.startswith("A") else "A+B",
                            "checkpoint": cores[arm].snapshot(),
                        }
            event(name, scores=summaries)

        probe("initial", lessons)
        for block, cohort in (("A", lessons[:8]), ("B", lessons[8:])):
            for pass_number in range(1, 257):
                for lesson in cohort:
                    row = {"block": block, "pass": pass_number, "id": lesson["id"], "offers": {}}
                    report["training"].append(row)
                    query, inventory, gold = inputs[lesson["id"]]
                    for arm in UPDATING:
                        guard("observe:" + arm)
                        row["offers"][arm] = cores[arm].observe(query, inventory, gold)
                if pass_number % 16 == 0:
                    probe(f"{block}_{pass_number:03d}", lessons[:8] if block == "A" else lessons)
            checkpoint("after_" + block)
        if any(cores[a].offer_count != 4096 for a in UPDATING) or frozen_hashes != {
            a: cores[a].checkpoint_sha256 for a in frozen_hashes
        }:
            raise RuntimeError("teaching count or frozen state changed")
        final_hashes = {a: c.checkpoint_sha256 for a, c in cores.items()}
        guard("motor_demonstrations")
        motor = _motor(work, report)
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
                    predictions[(arm, lesson["id"])],
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
            raise RuntimeError("motor phase changed reader state")
        if source_identity(revision)["files"] != {
            k: v for k, v in identity["files"].items() if k not in ADDED_SOURCES
        }:
            raise RuntimeError("old source changed during qualification")
        if any(
            sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in identity["files"].items()
        ):
            raise RuntimeError("frozen source changed during qualification")
        if sha256(native.library_path.read_bytes()).hexdigest() != report["native_motor"]["sha256"]:
            raise RuntimeError("native binary changed during qualification")
        report["retained_python_numpy_bytes"] = retained_size(cores, inputs, motor)
        report["retained_python_numpy_scope"] = (
            "Unique learner/optimizer state, retained questions/paragraphs/vectors/inventories, "
            "and motor; excludes experiment checkpoint copies, predictions and report objects"
        )
        report["retained_source_label_bytes"] = retained_size(fixture)
        report["retained_experiment_python_numpy_bytes"] = retained_size(
            cores, inputs, motor, fixture, arrays, predictions, report
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
        report["elapsed_before_artifacts_seconds"] = time.perf_counter() - started
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
                "elapsed_seconds_including_artifacts": time.perf_counter() - started,
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
            if error is None:
                error = exc
            print(json.dumps({"finalization_error": str(exc)}), flush=True)
    if error is not None:
        raise RuntimeError(
            "qualification failed; retain partial report and launch output"
        ) from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.revision)
