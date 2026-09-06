"""One fixed matched-prefix diagnostic of authenticated document-memory weights."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = time.perf_counter()
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated import local_memory_localization_v1 as operations  # noqa: E402
from experiments.r4_integrated import local_memory_value_credit_v1 as previous  # noqa: E402

file_hash, write_json, digest = previous.file_hash, previous.write_json, previous.digest
clone, tensor_bytes = operations.clone, operations.tensor_bytes
canonical, progress, score = operations.canonical, operations.progress, operations.score
CONFIG = {
    "schema": "local-memory-binding-v1", "seconds": 60, "parent_mib": 2048,
    "checkpoint_modes": ["tagged", "writer_maps_frozen"],
    "traced_ingestions": 16, "ordinary_parity_ingestions": 16,
    "recorded_trace_steps": 800, "ingestion_byte_steps": 1600,
    "free_queries": 32, "output_phase_evaluations": 96,
    "teacher_offers": 0, "actions": 0, "motor_demonstrations": 0,
    "prefix_milestones": [9, 20, 21, 22, 24, 49],
    "expected_derived_prefix_projections": 192,
    "endpoint_revision": "86cc7232320cf4cef9337cf4f3b79a017bb55171",
}
ENDPOINT = {
    ".pytest-tmp/local-memory-value-credit-v1-run/source-free/source_free.pt":
        "27750d54f96c48acd3c51d6b9669c336ebde16b69b202eb04bd629b4d46855a8",
    ".pytest-tmp/local-memory-value-credit-v1-run/report.json":
        "0b66fba5b9ea8981cac8883eea07abc4d3e283e4809f4166ea1b0b6ba2afc146",
    ".pytest-tmp/local-memory-value-credit-v1-run/completion.json":
        "2f9d63268ea195e0a82d83d51da6dd19a2551fee3aae62a3e092454acdbc8c45",
    ".pytest-tmp/local-memory-value-credit-v1-run-independent-audit.json":
        "f29992900df13adeab81bdae91a3f5c9e6d6a35cdab4900b16b8ab531020f268",
}


def sources():
    paths = (Path(__file__), Path(__file__).with_suffix(".md"),
             ROOT / "docs/research/local-memory-binding-review.md")
    return {**operations.sources(), **previous.sources(),
            **{p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}}


def roster():
    """Fixed offline factors; never invoked in the source-free worker."""
    assignments = (
        ("A", (("k00", "ab"), ("k01", "bc"))),
        ("B", (("k00", "bc"), ("k01", "ab"))),
        ("C", (("k02", "cd"), ("k03", "da"))),
        ("D", (("k02", "da"), ("k03", "cd"))),
    )
    memories, cases = [], []
    for mode in CONFIG["checkpoint_modes"]:
        for reverse in (False, True):
            for name, pairs in assignments:
                mid = f"{mode}-{name}-{'reversed' if reverse else 'original'}"
                documents = [f"Record {key} has value {value}.".encode().hex()
                             for key, value in pairs]
                memories.append({"memory_id": mid, "model_id": mode,
                                 "assignment": name, "reversed": reverse,
                                 "document_hex": documents[::-1] if reverse else documents})
                for key, value in pairs:
                    cases.append({"id": len(cases), "memory_id": mid, "mode": mode,
                                  "query_hex": f"What is the value of {key}?".encode().hex(),
                                  "target_hex": value.encode().hex()})
    assert len(memories) == 16 and len(cases) == 32
    return {"memories": memories, "cases": cases}


def new_model(mode, weights):
    if mode not in CONFIG["checkpoint_modes"]:
        raise ValueError("unknown binding checkpoint mode")
    model = previous.new_model(previous.SPEC, mode)
    model.load_state_dict(weights, strict=True)
    return model


def worker(args):
    """Load only authenticated weights, retained states and query-only requests."""
    import torch

    from benchmarks.native_scaling import _process_memory
    from experiments.r4_integrated.local_memory_localization_core import observed_query

    assert all(p.is_absolute() for p in (args.output, args.checkpoint, args.requests))
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    requests = json.loads(args.requests.read_bytes())
    assert set(requests) == {"sources", "cases"} and requests["sources"] == sources()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == {"models", "memories"}
    observations, outputs = {}, []
    for mid, weights in saved["models"].items():
        mode = weights["_extra_state"]["mode"]
        assert mode in CONFIG["checkpoint_modes"]
        model = new_model(mode, weights)
        model_before = digest(model.state_dict())
        for request in requests["cases"]:
            assert set(request) == {"id", "memory_id", "query_hex"}
            entry = saved["memories"][request["memory_id"]]
            assert set(entry) == {"model_id", "state"}
            if entry["model_id"] != mid:
                continue
            if time.time() > args.deadline_unix:
                raise TimeoutError("source-free binding deadline")
            if _process_memory()["peak_working_set_mib"] > CONFIG["parent_mib"]:
                raise MemoryError("source-free binding memory bound")
            state = model.memory.restore_local_state(entry["state"])
            before = digest(state.snapshot())
            progress(
                args.output.with_suffix(".jsonl"),
                "query_started",
                id=request["id"],
                maximum_phase_advances=3,
            )
            observation = observed_query(model, bytes.fromhex(request["query_hex"]), state)
            assert digest(state.snapshot()) == before
            assert digest(model.state_dict()) == model_before
            observations[request["id"]] = observation
            progress(
                args.output.with_suffix(".jsonl"),
                "query_completed",
                id=request["id"],
                decisions=observation["decisions"],
                supplements=observation["extra_phase_evaluations"],
            )
            outputs.append(
                {
                    **request,
                    **{
                        key: observation[key]
                        for key in (
                            "generated_hex",
                            "stop_reason",
                            "decisions",
                            "extra_phase_evaluations",
                        )
                    },
                    "memory_sha256": before,
                    "weights_sha256": model_before,
                    "model_id": mid,
                }
            )
    outputs.sort(key=lambda row: row["id"])
    assert [x["id"] for x in outputs] == [x["id"] for x in requests["cases"]]
    observation_path = args.output.with_suffix(".pt")
    torch.save(observations, observation_path)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha and sources() == requests["sources"]
    write_json(
        args.output,
        {
            "status": "completed",
            "outputs": outputs,
            "observations_sha256": file_hash(observation_path),
            "checkpoint_sha256": args.checkpoint_sha,
            "requests_sha256": args.requests_sha,
            "worker_pid": os.getpid(),
            "memory": _process_memory(),
            "elapsed_seconds": time.perf_counter() - START,
            "duration_clock": previous.duration_clock(),
            "query_traversals": len(outputs),
            "query_byte_steps": sum(len(bytes.fromhex(x["query_hex"])) + 1 for x in outputs),
            "phase_evaluations": 3 * len(outputs),
            "roster_or_documents_or_targets_supplied_to_model": False,
        },
    )


def query_stage(directory, saved, cases, manifest, deadline, guard):
    import torch

    directory.mkdir()
    checkpoint, requests = directory / "checkpoint.pt", directory / "query_only.json"
    torch.save(saved, checkpoint)
    write_json(
        requests,
        {
            "sources": manifest,
            "cases": [{key: row[key] for key in ("id", "memory_id", "query_hex")} for row in cases],
        },
    )
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "worker",
        "--output",
        str(directory / "observations.json"),
        "--checkpoint",
        str(checkpoint),
        "--requests",
        str(requests),
        "--checkpoint-sha",
        file_hash(checkpoint),
        "--requests-sha",
        file_hash(requests),
        "--deadline-unix",
        str(deadline),
    ]
    guard()
    start = time.perf_counter()
    process = subprocess.run(
        command, cwd=directory, capture_output=True, timeout=max(0.001, deadline - time.time())
    )
    (directory / "worker.stdout").write_bytes(process.stdout)
    (directory / "worker.stderr").write_bytes(process.stderr)
    if process.returncode:
        raise RuntimeError("source-free binding worker failed")
    receipt = json.loads((directory / "observations.json").read_bytes())
    receipt.update(command=command, package_seconds=time.perf_counter() - start)
    assert receipt["checkpoint_sha256"] == file_hash(checkpoint)
    assert receipt["requests_sha256"] == file_hash(requests)
    assert receipt["observations_sha256"] == file_hash(directory / "observations.pt")
    assert receipt["duration_clock"] == previous.duration_clock()
    return receipt, torch.load(directory / "observations.pt", map_location="cpu", weights_only=True)


def run(args):
    import torch

    from benchmarks.native_scaling import _process_memory
    from experiments.r4_integrated.local_memory_localization_core import traced_ingest
    from humanbrain.native import loaded_library_identity

    assert args.output.is_absolute() and args.admission.is_absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed", "config": CONFIG, "source_revision": args.revision,
        "admission_sha256": args.admission_sha, "pid": os.getpid(),
        "launch_started_unix": args.launch_started_unix, "deadline_unix": args.deadline_unix,
        "duration_clock": previous.duration_clock(), "workers": [], "parity": [],
    }

    def guard():
        if time.time() > args.deadline_unix:
            raise TimeoutError("binding aggregate deadline")
        if _process_memory()["peak_working_set_mib"] > CONFIG["parent_mib"]:
            raise MemoryError("binding process memory bound")

    try:
        assert file_hash(args.admission) == args.admission_sha
        admission = json.loads(args.admission.read_bytes())
        assert admission["status"] == "admitted" and admission["config"] == CONFIG
        assert admission["source_revision"] == args.revision == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        assert 0 < args.deadline_unix - args.launch_started_unix <= CONFIG["seconds"]
        assert args.launch_started_unix <= time.time() < args.deadline_unix
        assert all(os.environ.get(k) == "1" for k in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        assert os.environ.get("HUMANBRAIN_DEVICE") == "cpu"
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        manifest = sources()
        assert admission["source_sha256"] == manifest
        assert admission["endpoint_sha256"] == ENDPOINT
        for path, expected in ENDPOINT.items():
            assert file_hash(ROOT / path) == expected
        assert admission["duration_clock"] == report["duration_clock"]
        report["source_sha256"], report["endpoint_sha256"] = manifest, ENDPOINT
        report["packages"] = {name: version(name) for name in ("torch", "numpy")}
        report["native"] = loaded_library_identity().to_record()
        assert report["packages"] == admission["packages"]
        assert report["native"] == admission["native"]
        plan = roster()
        assert canonical(plan) == admission["roster_sha256"]
        report["roster_sha256"] = canonical(plan)
        write_json(args.output / "roster.json", plan)
        original = torch.load(ROOT / next(iter(ENDPOINT)), map_location="cpu", weights_only=True)
        endpoints = {mode: clone(original[mode]["weights"]) for mode in CONFIG["checkpoint_modes"]}
        del original
        torch.save(endpoints, args.output / "endpoints.pt")
        endpoint_hashes = {mode: digest(weights) for mode, weights in endpoints.items()}
        report["endpoint_model_sha256"] = endpoint_hashes
        models = {mode: new_model(mode, weights) for mode, weights in endpoints.items()}
        baseline = {"models": endpoints, "memories": {}}
        traces = {}
        for entry in plan["memories"]:
            guard()
            model = models[entry["model_id"]]
            documents = [bytes.fromhex(d) for d in entry["document_hex"]]
            assert sum(len(d) + 1 for d in documents) == 50
            mid = entry["memory_id"]
            progress(args.output / "progress.jsonl", "ingestion_started",
                     kind="traced", id=mid, maximum_byte_steps=50)
            state, trace = traced_ingest(model, documents)
            assert len(trace) == 50
            progress(args.output / "progress.jsonl", "ingestion_completed",
                     kind="traced", id=mid, byte_steps=50)
            guard()
            progress(args.output / "progress.jsonl", "ingestion_started",
                     kind="ordinary_parity", id=mid, maximum_byte_steps=50)
            mate = model.ingest(documents)
            traced_hash, ordinary_hash = digest(state.snapshot()), digest(mate.snapshot())
            assert traced_hash == ordinary_hash
            del mate
            report["parity"].append({"memory_id": mid, "traced_state_sha256": traced_hash,
                                    "ordinary_state_sha256": ordinary_hash,
                                    "exact_full_state": True})
            progress(args.output / "progress.jsonl", "ingestion_completed",
                     kind="ordinary_parity", id=mid, byte_steps=50)
            assert digest(model.state_dict()) == endpoint_hashes[entry["model_id"]]
            baseline["memories"][mid] = {"model_id": entry["model_id"],
                                         "state": clone(state.snapshot())}
            traces[mid] = trace
        torch.save(traces, args.output / "baseline_traces.pt")
        report["baseline_trace_steps"] = sum(map(len, traces.values()))
        report["baseline_trace_tensor_bytes"] = sum(
            tensor_bytes(row) for rows in traces.values() for row in rows
        )
        del traces
        report["baseline_retained_tensor_bytes"] = tensor_bytes(baseline)
        receipt, observations = query_stage(args.output / "baseline", baseline,
                                           plan["cases"], manifest, args.deadline_unix, guard)
        report["workers"].append(receipt)
        del baseline
        report["observation_tensor_bytes"] = sum(tensor_bytes(x) for x in observations.values())
        report["scores"] = [{"id": case["id"],
                             **score(observations[case["id"]], case["target_hex"])}
                            for case in plan["cases"]]
        report["free_exact"] = [
            {"id": case["id"], "correct":
             observations[case["id"]]["stop_reason"] == "eos" and
             observations[case["id"]]["generated_hex"] == case["target_hex"]}
            for case in plan["cases"]
        ]
        outputs = receipt["outputs"]
        report["counts"] = {
            "ingestions": 32, "traced_ingestions": 16, "ordinary_parity_ingestions": 16,
            "ingestion_byte_steps": 1600, "recorded_trace_steps": report["baseline_trace_steps"],
            "free_queries": len(outputs), "free_query_byte_steps": receipt["query_byte_steps"],
            "free_generation_decisions": sum(x["decisions"] for x in outputs),
            "supplemental_phase_evaluations": sum(x["extra_phase_evaluations"] for x in outputs),
            "free_and_supplemental_phase_advances": 96, "actual_output_score_evaluations": 96,
            "external_target_scoring_phase_vectors": 96,
            "external_target_scoring_scalar_residuals": 24672,
            "query_key_observation_recomputations": 32,
            "trace_writer_map_observation_steps": 800,
            "trace_writer_map_observation_linear_evaluations": 2400,
            "prefix_queries": 0, "teacher_offers": 0, "teacher_query_traversals": 0,
            "teacher_phase_advances": 0, "parameter_updates": 0,
            "action_attempts": 0, "motor_demonstrations": 0,
        }
        report["saved_audit_accounting"] = {"prefix_matrix_query_projections": 192,
                                            "model_calls": 0}
        report["count_scope"] = (
            "32 ordinary query traversals; 96 total LMS phases include generated and supplemental "
            "phases. Saved auditor separately derives 192 prefix M64*q64 projections; no prefix "
            "query or output call. Observation maps and external target scoring are counted."
        )
        assert len(report["parity"]) == 16 and len(outputs) == 32
        assert report["counts"]["recorded_trace_steps"] == 800
        assert report["counts"]["free_query_byte_steps"] == 832
        assert (report["counts"]["free_generation_decisions"] +
                report["counts"]["supplemental_phase_evaluations"]) == 96
        after_hashes = {mode: digest(model.state_dict()) for mode, model in models.items()}
        assert after_hashes == endpoint_hashes
        report["endpoint_model_sha256_after"] = after_hashes
        assert loaded_library_identity().to_record() == report["native"]
        guard()
        report["status"] = "completed"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        try:
            report["source_sha256_after"] = sources()
            assert report["source_sha256_after"] == report.get("source_sha256")
            for path, expected in ENDPOINT.items():
                assert file_hash(ROOT / path) == expected
            report["memory"] = _process_memory()
            report["elapsed_seconds"] = time.perf_counter() - START
            guard()
        except BaseException:
            report["status"] = "failed"
            report["finalization_error"] = traceback.format_exc()
        write_json(args.output / "report.json", report)
        artifacts = {p.relative_to(args.output).as_posix(): file_hash(p)
                     for p in args.output.rglob("*") if p.is_file() and p.name != "report.json"}
        write_json(args.output / "completion.json", {
            "status": "passed" if report["status"] == "completed" else "failed",
            "report_sha256": file_hash(args.output / "report.json"), "artifacts": artifacts,
            "artifact_bytes": {name: (args.output / name).stat().st_size for name in artifacts},
            "source_revision": args.revision, "admission_sha256": args.admission_sha,
            "elapsed_seconds": time.perf_counter() - START,
        })
    return report["status"] == "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker"))
    for name in ("output", "admission", "checkpoint", "requests"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    for name in ("admission-sha", "revision", "checkpoint-sha", "requests-sha"):
        parser.add_argument("--" + name)
    parser.add_argument(
        "--deadline-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_DEADLINE_UNIX")
    )
    parser.add_argument(
        "--launch-started-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_START_UNIX")
    )
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
    else:
        raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
