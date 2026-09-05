"""One admitted token-pointer acquisition run and exposed persistent-action regression.

Importing this module never builds the cache, constructs a reader or evaluates a
target. The admission and all executable sources must match the frozen revision.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from benchmarks.native_scaling import _process_memory
from experiments.r4_integrated.token_pointer_data import (
    ARCHIVE,
    ENCODER_SHA256,
    INTEGRATION,
    PINS,
    RAW,
    REPORT,
    SELECTION,
    WordCache,
    build_cache,
    load_selection,
)
from experiments.r4_integrated.token_pointer_preflight import (
    ROOT,
    THREADS,
    environment,
    file_hash,
    source_identity,
)
from experiments.r4_integrated.token_pointer_runtime import (
    SEED,
    PointerPolicy,
    local_examples,
    score_counts,
    score_local,
    state_hash,
)
from experiments.r4_integrated.token_pointer_session import UPDATING, run_session

OUTPUT = ROOT / "benchmarks/token_pointer_acquisition_v1.json"
ADMISSION = ROOT / "benchmarks/token_pointer_acquisition_v1.admission.json"
LIMITS = {
    "cache_byte_budget": 1_079_773_324,
    "seconds": 14_400,
    "peak_parent_mib": 4096,
    "reserve_bytes": 2**30,
    "artifact_allowance_bytes": 96 * 2**20,
}


def epoch_batches(train, epoch):
    """One identical batch order for every arm; no score-dependent curriculum."""
    if not 1 <= epoch <= 8:
        raise ValueError("exactly eight epochs required")
    supported = [i for i, row in enumerate(train) if not row["is_impossible"]]
    stopped = [i for i, row in enumerate(train) if row["is_impossible"]]
    if len(train) != 8192 or len(supported) != 6144 or len(stopped) != 2048:
        raise ValueError("fixed 8192-question 3:1 teaching roster required")
    rng = np.random.default_rng(SEED + epoch)
    supported = rng.permutation(supported)
    stopped = rng.permutation(stopped)
    for batch in range(256):
        indices = np.concatenate(
            (supported[24 * batch : 24 * (batch + 1)], stopped[8 * batch : 8 * (batch + 1)])
        )
        rng.shuffle(indices)
        yield [train[int(index)] for index in indices]


def acquisition_decision(arms):
    """Prospective engineering margins; no significance or architecture-adoption claim."""
    failure = {"complete": False, "qualified": False}
    if tuple(arms) != UPDATING:
        return failure
    for row in arms.values():
        if row.get("steps") != 2048 or row.get("presentations") != 65536:
            return failure
        if len(row.get("initial", [])) != 256 or len(row.get("final", [])) != 256:
            return failure
        if set(row.get("train_probes", {})) != {"1", "4", "8"}:
            return failure
        if any(len(rows) != 128 for rows in row["train_probes"].values()):
            return failure
    initial = {a: score_counts(row["initial"]) for a, row in arms.items()}
    final = {a: score_counts(row["final"]) for a, row in arms.items()}
    trained = {a: score_counts(row["train_probes"]["8"]) for a, row in arms.items()}
    if any(
        c["supported_total"] != 192 or c["STOP_total"] != 64
        for c in (*initial.values(), *final.values())
    ):
        return failure
    if any(c["supported_total"] != 96 or c["STOP_total"] != 32 for c in trained.values()):
        return failure

    def comparison(candidate, reference):
        supported = candidate["supported"] - reference["supported"]
        stops = candidate["STOP"] - reference["STOP"]
        # BA=(supported+3*STOP)/384. Compare the 5pp boundary as exact integers.
        delta = supported + 3 * stops
        return {
            "supported_gain": supported,
            "STOP_change": stops,
            "balanced_accuracy_gain": delta / 384,
            "supported_gain_passed": supported >= 20,
            "balanced_gain_passed": 100 * delta >= 5 * 384,
        }

    cell = {
        "initial_frozen": comparison(final["cell"], initial["cell"]),
        "frozen_producer": comparison(final["cell"], final["frozen_producer"]),
        "bypass": comparison(final["cell"], final["bypass"]),
    }
    conventional = comparison(final["lstm"], initial["lstm"])
    cell_pass = final["cell"]["STOP"] >= 48 and all(
        c["supported_gain_passed"] and c["balanced_gain_passed"] for c in cell.values()
    )
    reference_pass = (
        conventional["supported_gain_passed"]
        and conventional["balanced_gain_passed"]
        and final["lstm"]["STOP"] >= 48
        and trained["lstm"]["supported"] >= 48
    )
    return {
        "complete": True,
        "qualified": cell_pass and reference_pass,
        "cell_qualified": cell_pass,
        "positive_control_qualified": reference_pass,
        "initial_counts": initial,
        "final_counts": final,
        "final_train_counts": trained,
        "cell_comparisons": cell,
        "conventional_vs_initial": conventional,
        "cell_STOP_floor_passed": final["cell"]["STOP"] >= 48,
        "conventional_STOP_floor_passed": final["lstm"]["STOP"] >= 48,
        "scope": "One-seed prospective recipe evidence; not independent adjudication or adoption",
    }


def _write_json(path, record):
    # Serialize before opening the destination: malformed fields cannot destroy
    # an existing failure report. The outer finalizer preserves the original error.
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _retained_input_dependencies(encoder_metadata, source):
    """Unique retained file costs, separately from incremental run artifacts.

    Call only after build_cache succeeds: that path authenticates the old archive,
    report and original integration fixture. Reuse those verified pins and the
    encoder constructor's checks instead of rereading large payloads again.
    """
    model = ROOT / ".models/all-MiniLM-L6-v2"
    selected = SELECTION.relative_to(ROOT).as_posix()
    declared = [
        (ARCHIVE, "inherited_word_vector_archive", PINS[ARCHIVE], "cache_build_pin_check"),
        (REPORT, "inherited_input_manifest_report", PINS[REPORT], "cache_build_pin_check"),
        (RAW, "official_original_QA_dataset", PINS[RAW], "streamed_now"),
        (
            SELECTION,
            "selected_question_manifest",
            source["files"][selected],
            "frozen_source_admission",
        ),
        (INTEGRATION, "original_integration_fixture", PINS[INTEGRATION], "cache_build_pin_check"),
        (model / "manifest.json", "encoder_manifest", None, "streamed_now"),
    ]
    declared.extend(
        (model / name, "encoder_file", record["sha256"], "encoder_constructor_check")
        for name, record in encoder_metadata["files"].items()
    )
    unique = {}
    for path, role, expected, verification in declared:
        path = path.resolve()
        if verification == "streamed_now":
            actual = file_hash(path)
            if expected is not None and actual != expected:
                raise ValueError("retained original input differs: " + str(path))
            expected = actual
        key = os.path.normcase(str(path))
        if key in unique:
            if unique[key]["sha256"] != expected:
                raise ValueError("conflicting hashes for one retained input file")
            unique[key]["roles"].append(role)
            unique[key]["verification"].append(verification)
            continue
        unique[key] = {
            "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "absolute_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": expected,
            "roles": [role],
            "verification": [verification],
        }
    records = sorted(unique.values(), key=lambda row: row["absolute_path"])
    return {
        "files": records,
        "unique_file_count": len(records),
        "unique_file_bytes": sum(row["bytes"] for row in records),
        "scope": (
            "Entire retained input dependency files, including unused portions of the old archive; "
            "not resident arrays or newly learned weights. Paths counted once across roles."
        ),
        "separate_scopes": (
            "New cache and checkpoint/action/report artifacts are accounted separately. "
            "Interpreter/package installation, native runtime and source checkout are environment/"
            "source dependencies, not included in this input-file total."
        ),
    }


def _verify_admission(revision, digest):
    if file_hash(ADMISSION) != digest:
        raise ValueError("exact frozen broad-run admission required")
    admission = json.loads(ADMISSION.read_bytes())
    if admission["status"] != "admitted" or any(admission[k] != v for k, v in LIMITS.items()):
        raise ValueError("fixed acquisition limits/admission differ")
    required = {
        "src/humanbrain/token_pointer_reader.py",
        "experiments/r4_integrated/token_pointer_runtime.py",
        "experiments/r4_integrated/token_pointer_data.py",
        "experiments/r4_integrated/token_pointer_actions.py",
        "experiments/r4_integrated/token_pointer_session.py",
        "experiments/r4_integrated/token_pointer_preflight.py",
        "experiments/r4_integrated/token_pointer_acquisition.py",
        "docs/research/token-pointer-acquisition.md",
    }
    if not required <= set(admission["source_files"]):
        raise ValueError("admission omits a required reader/driver source")
    names = (*admission["source_files"], ADMISSION.relative_to(ROOT).as_posix())
    source = source_identity(revision, names)
    if any(source["files"][p] != h for p, h in admission["source_files"].items()):
        raise ValueError("working source differs from admitted sources")
    if SELECTION.relative_to(ROOT).as_posix() not in admission["source_files"]:
        raise ValueError("selection must be included in frozen source inventory")
    if file_hash(SELECTION) != admission["selection_sha256"]:
        raise ValueError("exact selected data required")
    small = {}
    for key in ("report", "completion"):
        item = admission["small_fit"][key]
        if file_hash(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("small-fit evidence differs")
        small[key] = json.loads((ROOT / item["path"]).read_bytes())
    if (
        small["report"]["status"] != "completed"
        or not small["report"].get("qualified")
        or small["completion"]["status"] != "passed"
        or not small["completion"].get("qualified")
        or small["completion"]["report_sha256"] != admission["small_fit"]["report"]["sha256"]
    ):
        raise ValueError("both disposable small-fit branches must have qualified")
    env = environment()
    if (
        env != admission["environment"]
        or env["torch_threads"] != 1
        or torch.__version__ != "2.14.0+cpu"
        or any(os.environ.get(n) != "1" for n in THREADS)
        or os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
        or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve() != ROOT / ".native-cache"
    ):
        raise ValueError("qualified single-thread CPU environment required")
    needed = (
        admission["cache_byte_budget"]
        + admission["artifact_allowance_bytes"]
        + admission["reserve_bytes"]
    )
    if shutil.disk_usage(ROOT).free < needed:
        raise OSError("cache/artifacts would breach disk reserve")
    return admission, source, env


def run(output, revision, admission_sha256):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay in the selected checkout")
    completion = output.with_suffix(".completion.json")
    work = ROOT / ".runtime" / output.stem.replace("_", "-")
    if output.exists() or completion.exists() or work.exists():
        raise FileExistsError("preserve the original run and partial artifacts")
    admission, source, env = _verify_admission(revision, admission_sha256)
    began, began_unix = time.perf_counter(), time.time()
    work.mkdir(parents=True)
    result = {
        "status": "failed",
        "started_unix": began_unix,
        "source": source,
        "admission_sha256": admission_sha256,
        "environment": env,
        "limits": LIMITS,
        "selection_sha256": admission["selection_sha256"],
        "small_fit": admission["small_fit"],
        "resource_projection": admission["resource_projection"],
        "arms": {},
        "events": [],
        "counters": {"optimizer_steps": 0, "presentations": 0, "local_predictions": 0},
        "exposure": (
            "Broad local original-paragraph QA; separate exposed 28-question pooled regression"
        ),
        "frozen_initial_reference": (
            "Reuse initial cell scores and retained initial checkpoint; no extra forward"
        ),
    }
    policies, checkpoints, cache = {}, {}, None
    cache_dir = work / "word-cache"

    def guard(label):
        memory = _process_memory()
        result["last_memory"] = {"label": label, **memory}
        if time.perf_counter() - began > admission["seconds"]:
            raise TimeoutError("fixed four-hour comparison budget exhausted")
        if memory["peak_working_set_mib"] > admission["peak_parent_mib"]:
            raise MemoryError("fixed 4 GiB parent peak exceeded")
        if shutil.disk_usage(ROOT).free < admission["reserve_bytes"]:
            raise OSError("fixed disk reserve breached")

    def event(label, **values):
        record = {"event": label, "elapsed_seconds": time.perf_counter() - began, **values}
        result["events"].append(record)
        print(json.dumps(record, allow_nan=False), flush=True)

    def probe(policy, cases, destination):
        before = policy.checkpoint_sha256
        for offset in range(0, len(cases), 32):
            guard("local_probe")
            rows = score_local(policy, cases[offset : offset + 32], data["documents"], cache.get)
            destination.extend(rows)
            result["counters"]["local_predictions"] += len(rows)
        if [r["id"] for r in destination] != [
            r["id"] for r in cases
        ] or policy.checkpoint_sha256 != before:
            raise RuntimeError("fixed local probe roster or read-only state differs")

    try:
        guard("before_cache")
        data = load_selection(expected_sha256=admission["selection_sha256"])
        result["selection"] = {
            "path": SELECTION.relative_to(ROOT).as_posix(),
            "train_ids": [r["id"] for r in data["train"]],
            "probe_ids": [r["id"] for r in data["probe"]],
            "train_probe_ids": data["train_probe_ids"],
            "cache_plan": data["cache_plan"],
            "overlap_checks": data["overlap_checks"],
        }
        if data["cache_plan"]["new_disk_bytes"] != admission["cache_byte_budget"]:
            raise ValueError("fixed cache byte plan differs")
        from humanbrain.frozen_context_encoder import FrozenContextEncoder

        encoder = FrozenContextEncoder(ROOT / ".models/all-MiniLM-L6-v2", max_tokens=256, threads=1)
        if encoder.sha256 != ENCODER_SHA256:
            raise ValueError("qualified generic encoder identity differs")
        result["encoding"] = {
            "encoder_sha256": encoder.sha256,
            "metadata": encoder.metadata,
            "new_calls": 0,
            "seconds": 0.0,
        }

        def encode(raw):
            guard("encode_input")
            tick = time.perf_counter()
            encoded = encoder.encode(raw)
            result["encoding"]["new_calls"] += 1
            result["encoding"]["seconds"] += time.perf_counter() - tick
            guard("encoded_input")
            return encoded

        tick = time.perf_counter()
        result["cache"] = build_cache(
            data,
            cache_dir,
            encode=encode,
            byte_budget=admission["cache_byte_budget"],
            deadline_unix=began_unix + admission["seconds"],
            minimum_free_bytes=admission["reserve_bytes"] + admission["artifact_allowance_bytes"],
        )
        result["cache"]["seconds"] = time.perf_counter() - tick
        result["cache"]["path"] = cache_dir.relative_to(ROOT).as_posix()
        if result["encoding"]["new_calls"] != result["cache"]["new_encoder_calls"]:
            raise RuntimeError("encoder call accounting differs")
        result["retained_input_dependencies"] = _retained_input_dependencies(
            result["encoding"]["metadata"], source
        )
        encoder = None
        gc.collect()
        cache = WordCache(cache_dir, result["cache"]["index_sha256"])
        guard("cache_loaded")
        event("cache_complete", new_encoder_calls=result["encoding"]["new_calls"])
        for arm in UPDATING:
            policies[arm] = PointerPolicy(
                "cell" if arm == "frozen_producer" else arm,
                frozen_producer=arm == "frozen_producer",
            )
            policy = policies[arm]
            result["arms"][arm] = {
                "initial": [],
                "final": [],
                "train_probes": {},
                "steps": 0,
                "presentations": 0,
                "initial_state_sha256": policy.checkpoint_sha256,
                "parameters": sum(p.numel() for p in policy.reader.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in policy.reader.parameters() if p.requires_grad
                ),
            }
            checkpoints[arm + "_initial"] = policy.checkpoint()
            probe(policy, data["probe"], result["arms"][arm]["initial"])
        if state_hash(checkpoints["cell_initial"]["model"]) != state_hash(
            checkpoints["frozen_producer_initial"]["model"]
        ):
            raise RuntimeError("frozen producer must start at the exact cell weights")
        by_id = {row["id"]: row for row in data["train"]}
        train_probe = [by_id[i] for i in data["train_probe_ids"]]
        with (work / "updates.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
            for epoch in range(1, 9):
                for batch_number, batch in enumerate(epoch_batches(data["train"], epoch), 1):
                    queries, contexts, gold = local_examples(batch, data["documents"], cache.get)
                    for arm, policy in policies.items():
                        guard("update:" + arm)
                        row = {
                            "epoch": epoch,
                            "batch": batch_number,
                            "arm": arm,
                            "ids": [r["id"] for r in batch],
                            "before": policy.checkpoint_sha256,
                            "steps_before": policy.steps,
                            "presentations_before": policy.presentations,
                        }
                        result["pending_update"] = row
                        tick = time.perf_counter()
                        row["receipt"] = policy.update_batch(queries, contexts, gold)
                        row.update(
                            seconds=time.perf_counter() - tick,
                            after=policy.checkpoint_sha256,
                            steps_after=policy.steps,
                            presentations_after=policy.presentations,
                        )
                        if (
                            policy.steps != row["steps_before"] + 1
                            or policy.presentations != row["presentations_before"] + 32
                        ):
                            raise RuntimeError(
                                "one global batch update must advance one step and 32 presentations"
                            )
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        journal.flush()
                        result["counters"]["optimizer_steps"] += 1
                        result["counters"]["presentations"] += 32
                        result["arms"][arm].update(
                            steps=policy.steps, presentations=policy.presentations
                        )
                        result.pop("pending_update")
                    if any(not torch.equal(p.rng, policies["cell"].rng) for p in policies.values()):
                        raise RuntimeError("matched policies consumed different dropout streams")
                if epoch in (1, 4, 8):
                    for arm, policy in policies.items():
                        rows = result["arms"][arm]["train_probes"][str(epoch)] = []
                        probe(policy, train_probe, rows)
                event(
                    "epoch_complete", epoch=epoch, steps={a: p.steps for a, p in policies.items()}
                )
        for arm, policy in policies.items():
            probe(policy, data["probe"], result["arms"][arm]["final"])
            checkpoints[arm + "_post_acquisition"] = policy.checkpoint()
            result["arms"][arm]["post_acquisition_state_sha256"] = policy.checkpoint_sha256
        if result["counters"] != {
            "optimizer_steps": 8192,
            "presentations": 262144,
            "local_predictions": 3584,
        }:
            raise RuntimeError("fixed acquisition work counts differ")
        result["acquisition_decision"] = acquisition_decision(result["arms"])
        if not result["acquisition_decision"]["complete"]:
            raise RuntimeError("acquisition evidence incomplete")
        event("acquisition_complete", qualified=result["acquisition_decision"]["qualified"])
        result["integration"] = run_session(policies, cache.get, work / "integration", guard, event)
        if (
            result["integration"]["status"] != "completed"
            or not result["integration"]["decision"]["complete"]
        ):
            raise RuntimeError(
                "persistent-action session incomplete; preserve its partial evidence"
            )
        result["qualified"] = (
            result["acquisition_decision"]["qualified"]
            and result["integration"]["decision"]["qualified"]
        )
        result["architecture_adopted"] = False
        result["status"] = "completed"
        guard("comparison_complete")
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
    finally:
        result["finalization_errors"] = []
        for arm, policy in policies.items():
            try:
                checkpoints[arm + ("_final" if result["status"] == "completed" else "_partial")] = (
                    policy.checkpoint()
                )
            except BaseException:
                result["finalization_errors"].append(traceback.format_exc())
        try:
            if cache is not None:
                result["cache"]["reused_float32_ram_bytes"] = cache._reused_bytes
                cache.close()
            source_after = source_identity(revision, tuple(source["files"]))
            if source_after != source:
                raise RuntimeError("frozen source changed during run")
        except BaseException:
            result["finalization_errors"].append(traceback.format_exc())
        try:
            archive = work / "checkpoints.pt"
            torch.save(checkpoints, archive)
            restored = torch.load(archive, map_location="cpu", weights_only=True)
            if state_hash(restored) != state_hash(checkpoints):
                raise RuntimeError("serialized complete checkpoint states differ")
            result["checkpoint_archive"] = {
                "path": archive.relative_to(ROOT).as_posix(),
                "sha256": file_hash(archive),
                "bytes": archive.stat().st_size,
                "state_sha256": {k: state_hash(v) for k, v in checkpoints.items()},
                "frozen_continuation_reconstruction": (
                    "Corresponding post_acquisition snapshot with updating=False"
                ),
            }
            del restored
            result["journals"] = {
                p.relative_to(ROOT).as_posix(): file_hash(p) for p in work.glob("*.jsonl")
            }
            cache_bytes = (
                sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
                if cache_dir.exists()
                else 0
            )
            artifact_bytes = (
                sum(p.stat().st_size for p in work.rglob("*") if p.is_file()) - cache_bytes
            )
            result["artifact_bytes_before_report"] = artifact_bytes
            if (
                cache_bytes > admission["cache_byte_budget"]
                or artifact_bytes > admission["artifact_allowance_bytes"]
            ):
                raise OSError("cache or artifact allowance exceeded")
            guard("archive_written_and_restored")
        except BaseException:
            result["finalization_errors"].append(traceback.format_exc())
        result["elapsed_seconds"] = time.perf_counter() - began
        try:
            result["final_memory"] = _process_memory()
        except BaseException:
            result["finalization_errors"].append(traceback.format_exc())
        if result["finalization_errors"]:
            result["status"] = "failed"
        result["finalization"] = "Authoritative only with matching passed completion receipt"
        try:
            _write_json(output, result)
        except BaseException:
            result = {
                "status": "failed",
                "error": result.get("error"),
                "serialization_error": traceback.format_exc(),
                "partial_report_repr": repr(result),
            }
            _write_json(output, result)
        try:
            guard("report_written")
            if (
                result.get("artifact_bytes_before_report", 0) + output.stat().st_size
                > admission["artifact_allowance_bytes"]
            ):
                raise OSError("report exceeded the total artifact allowance")
        except BaseException:
            result["status"] = "failed"
            result["report_finalization_error"] = traceback.format_exc()
            _write_json(output, result)
        finish = {
            "status": "passed" if result["status"] == "completed" else "failed",
            "qualified": result["status"] == "completed" and result.get("qualified", False),
            "report_sha256": file_hash(output),
            "source_revision": revision,
            "elapsed_seconds": time.perf_counter() - began,
            "final_memory": result.get("last_memory", result.get("final_memory")),
        }
        try:
            finish["free_disk_bytes"] = shutil.disk_usage(ROOT).free
        except BaseException:
            finish.update(status="failed", qualified=False, metadata_error=traceback.format_exc())
        _write_json(completion, finish)
        try:
            guard("completion_written")
            total = (
                result.get("artifact_bytes_before_report", 0)
                + output.stat().st_size
                + completion.stat().st_size
            )
            if total > admission["artifact_allowance_bytes"]:
                raise OSError("completion exceeded total artifact allowance")
        except BaseException:
            finish.update(status="failed", qualified=False, budget_error=traceback.format_exc())
            _write_json(completion, finish)
        print(json.dumps(finish), flush=True)
    return 0 if result["status"] == "completed" and finish["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.revision, args.admission_sha256))
