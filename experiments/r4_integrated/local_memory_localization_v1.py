"""One admitted frozen-checkpoint localization with discarded local proposals."""

from __future__ import annotations

import argparse
import hashlib
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

from experiments.r4_integrated import local_memory_two_byte_v1 as previous  # noqa: E402

file_hash, write_json, digest = previous.file_hash, previous.write_json, previous.digest
clone, tensor_bytes = previous.detached_state, previous.tensor_bytes
CONFIG = {
    "schema": "local-memory-localization-v1",
    "seconds": 60,
    "parent_mib": 2048,
    "checkpoint_modes": ["tagged", "writer_maps_frozen"],
    "learning_rate": 0.01,
    "baseline_ingestions": 16,
    "proposal_ingestions": 8,
    "perturbed_ingestions": 32,
    "free_queries": 64,
    "teacher_offers": 24,
    "motor_demonstrations": 2,
    "endpoint_revision": "a94b466f1b2c34e83b53d19e43f9430e52191648",
    "blocks": ["writer", "address"],
    "signs": [1, -1],
}
ENDPOINT = {
    ".pytest-tmp/local-memory-two-byte-v1-run/source-free/source_free.pt": (
        "d05a6a2d78e46619d49f3e7d25999649830d2ed4a545a6a853c62927e7dacca1"
    ),
    ".pytest-tmp/local-memory-two-byte-v1-run/report.json": (
        "57f6ff8330e90c536be40809579ff6c31917ea23076e711a5d61a6d1727c40c5"
    ),
    ".pytest-tmp/local-memory-two-byte-v1-run/completion.json": (
        "a9d34214a8e5ecd227d4496f09e328f80f650411910c64b964e0657b05f28181"
    ),
    ".pytest-tmp/local-memory-two-byte-v1-run-independent-audit.json": (
        "1ef342ddac2cabae3195d23007354ec8dd70b29171a1cc7c56f2bb49216c80a0"
    ),
}


def sources():
    paths = (
        Path(__file__),
        Path(__file__).with_suffix(".md"),
        ROOT / "experiments/r4_integrated/local_memory_localization_core.py",
        ROOT / "docs/research/local-document-memory-return-review.md",
    )
    return {**previous.sources(), **{p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}}


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def progress(path, event, **fields):
    """Small durable operation bounds, including when the watchdog interrupts."""
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")


def roster():
    """Offline fixed authoring; never invoked by the source-free worker."""
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
                documents = [
                    f"Record {key} has value {value}.".encode().hex() for key, value in pairs
                ]
                memories.append(
                    {
                        "memory_id": mid,
                        "model_id": mode,
                        "assignment": name,
                        "reversed": reverse,
                        "document_hex": documents[::-1] if reverse else documents,
                    }
                )
                for key, value in pairs:
                    cases.append(
                        {
                            "id": len(cases),
                            "memory_id": mid,
                            "mode": mode,
                            "query_hex": f"What is the value of {key}?".encode().hex(),
                            "target_hex": value.encode().hex(),
                            "kind": "baseline",
                        }
                    )
    baseline = list(cases)
    for case in baseline[:8]:
        for block in CONFIG["blocks"]:
            for sign in CONFIG["signs"]:
                pid = f"proposal-{case['id']}"
                mid = f"{pid}-{block}-{'plus' if sign == 1 else 'minus'}"
                cases.append(
                    {
                        **case,
                        "id": len(cases),
                        "memory_id": mid,
                        "kind": "perturbed",
                        "baseline_id": case["id"],
                        "proposal_id": pid,
                        "block": block,
                        "sign": sign,
                    }
                )
    assert len(memories) == 16 and len(cases) == 64
    return {"memories": memories, "cases": cases}


def new_model(mode, weights):
    model = previous.new_model(previous.SPEC, mode)
    model.load_state_dict(weights, strict=True)
    return model


def worker(args):
    """Load only authenticated weights, retained states and query-only requests."""
    import torch

    from benchmarks.native_scaling import _process_memory
    from experiments.r4_integrated.local_memory_localization_core import observed_query

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
                raise TimeoutError("source-free localization deadline")
            if _process_memory()["peak_working_set_mib"] > CONFIG["parent_mib"]:
                raise MemoryError("source-free localization memory bound")
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
        raise RuntimeError("source-free localization worker failed")
    receipt = json.loads((directory / "observations.json").read_bytes())
    receipt.update(command=command, package_seconds=time.perf_counter() - start)
    assert receipt["checkpoint_sha256"] == file_hash(checkpoint)
    assert receipt["requests_sha256"] == file_hash(requests)
    assert receipt["observations_sha256"] == file_hash(directory / "observations.pt")
    assert receipt["duration_clock"] == previous.duration_clock()
    return receipt, torch.load(directory / "observations.pt", map_location="cpu", weights_only=True)


def score(observation, target_hex):
    """External fixed-target scoring over already observed potentials only."""
    import torch

    targets = [*bytes.fromhex(target_hex), 256]
    losses, margins = [], []
    for phase, target in enumerate(targets):
        scores = observation["phase_scores"][phase]
        residual = -scores.clone()
        residual[target] += 1
        other = torch.cat((scores[:target], scores[target + 1 :]))
        losses.append(float(0.5 * residual.square().sum()))
        margins.append(float(scores[target] - other.max()))
    return {
        "target_hex": target_hex,
        "phase_losses": losses,
        "phase_margins": margins,
        "mean_loss": sum(losses) / 3,
        "loss_kind": "half_sum_squared_potential_error",
    }


def actions(output, cases, outputs, guard):
    from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
    from experiments.r4_integrated.utf8_action_bridge import Case, execute_utf8

    motor, demos, rows = ContextualOccurrenceMotor(), [], []
    for case in (
        Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
        Case("motor-b", "bora", "ready", "motor_beta", "warm"),
    ):
        guard()
        progress(output / "progress.jsonl", "motor_demo_started", id=case.name)
        before, after = case.files(), case.files(corrected=True)
        receipt = execute_utf8(case, after, before, output / case.name)
        assert receipt["passed"]
        motor.learn(f"Configure {case.key}".encode(), before, after)
        demos.append(receipt)
        progress(output / "progress.jsonl", "motor_demo_completed", id=case.name)
    for generated, case in zip(outputs, cases, strict=True):
        guard()
        assert generated["id"] == case["id"]
        progress(output / "progress.jsonl", "action_started", id=case["id"])
        result = previous.reference.actuate(
            generated, case["id"], motor, output / f"action-{case['id']:02d}"
        )
        rows.append(
            {
                "id": case["id"],
                "correct": bool(
                    result["emitted"]
                    and result["generated_hex"] == case["target_hex"]
                    and result["lossless_generated_execution"]
                ),
                **result,
            }
        )
        progress(output / "progress.jsonl", "action_completed", id=case["id"])
    return demos, rows


def run(args):
    import torch

    from benchmarks.native_scaling import _process_memory
    from experiments.r4_integrated.local_memory_localization_core import (
        perturbed_model,
        propose_blocks,
        traced_ingest,
    )
    from humanbrain.native import loaded_library_identity

    assert args.output.is_absolute() and args.admission.is_absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "config": CONFIG,
        "source_revision": args.revision,
        "admission_sha256": args.admission_sha,
        "pid": os.getpid(),
        "launch_started_unix": args.launch_started_unix,
        "deadline_unix": args.deadline_unix,
        "duration_clock": previous.duration_clock(),
        "workers": [],
        "actions": [],
    }

    def guard():
        if time.time() > args.deadline_unix:
            raise TimeoutError("localization aggregate deadline")
        if _process_memory()["peak_working_set_mib"] > CONFIG["parent_mib"]:
            raise MemoryError("localization process memory bound")

    try:
        assert file_hash(args.admission) == args.admission_sha
        admission = json.loads(args.admission.read_bytes())
        assert admission["status"] == "admitted" and admission["config"] == CONFIG
        assert (
            admission["source_revision"]
            == args.revision
            == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
        assert 0 < args.deadline_unix - args.launch_started_unix <= 60
        assert args.launch_started_unix <= time.time() < args.deadline_unix
        assert all(
            os.environ.get(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        )
        assert os.environ.get("HUMANBRAIN_DEVICE") == "cpu"
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        manifest = sources()
        assert admission["source_sha256"] == manifest
        assert admission["endpoint_sha256"] == ENDPOINT
        for path, expected in ENDPOINT.items():
            assert file_hash(ROOT / path) == expected
        assert admission["duration_clock"] == report["duration_clock"]
        report["source_sha256"] = manifest
        report["endpoint_sha256"] = ENDPOINT
        report["packages"] = {name: version(name) for name in ("torch", "numpy")}
        report["native"] = loaded_library_identity().to_record()
        assert (
            report["packages"] == admission["packages"] and report["native"] == admission["native"]
        )
        plan = roster()
        assert canonical(plan) == admission["roster_sha256"]
        report["roster_sha256"] = canonical(plan)
        write_json(args.output / "roster.json", plan)
        original = torch.load(ROOT / next(iter(ENDPOINT)), map_location="cpu", weights_only=True)
        endpoints = {mode: clone(original[mode]["weights"]) for mode in CONFIG["checkpoint_modes"]}
        del original
        endpoint_hashes = {mode: digest(weights) for mode, weights in endpoints.items()}
        torch.save(endpoints, args.output / "endpoints.pt")
        report["endpoint_model_sha256"] = endpoint_hashes
        models = {mode: new_model(mode, weights) for mode, weights in endpoints.items()}
        baseline = {"models": endpoints, "memories": {}}
        traces = {}
        for entry in plan["memories"]:
            guard()
            model = models[entry["model_id"]]
            progress(
                args.output / "progress.jsonl",
                "ingestion_started",
                kind="baseline",
                id=entry["memory_id"],
                maximum_byte_steps=50,
            )
            state, trace = traced_ingest(model, [bytes.fromhex(d) for d in entry["document_hex"]])
            assert digest(model.state_dict()) == endpoint_hashes[entry["model_id"]]
            baseline["memories"][entry["memory_id"]] = {
                "model_id": entry["model_id"],
                "state": clone(state.snapshot()),
            }
            traces[entry["memory_id"]] = trace
            progress(
                args.output / "progress.jsonl",
                "ingestion_completed",
                kind="baseline",
                id=entry["memory_id"],
                byte_steps=len(trace),
            )
        torch.save(traces, args.output / "baseline_traces.pt")
        report["baseline_trace_steps"] = sum(map(len, traces.values()))
        report["baseline_trace_tensor_bytes"] = sum(
            tensor_bytes(row) for rows in traces.values() for row in rows
        )
        del traces
        first_receipt, observed = query_stage(
            args.output / "baseline",
            baseline,
            plan["cases"][:32],
            manifest,
            args.deadline_unix,
            guard,
        )
        report["workers"].append(first_receipt)
        report["baseline_retained_tensor_bytes"] = tensor_bytes(baseline)
        del baseline
        perturbations = {"models": {}, "memories": {}}
        proposals = {}
        memory_roster = {x["memory_id"]: x for x in plan["memories"]}
        for case in plan["cases"][:8]:
            guard()
            documents = [bytes.fromhex(d) for d in memory_roster[case["memory_id"]]["document_hex"]]
            model = models["tagged"]
            progress(
                args.output / "progress.jsonl",
                "ingestion_started",
                kind="proposal",
                id=case["id"],
                maximum_byte_steps=50,
            )
            state = model.ingest(documents)
            progress(
                args.output / "progress.jsonl",
                "ingestion_completed",
                kind="proposal",
                id=case["id"],
                byte_steps=50,
            )
            assert digest(state.snapshot()) == first_receipt["outputs"][case["id"]]["memory_sha256"]
            progress(
                args.output / "progress.jsonl",
                "teacher_started",
                id=case["id"],
                maximum_teacher_offers=3,
            )
            proposal = propose_blocks(
                model,
                bytes.fromhex(case["query_hex"]),
                bytes.fromhex(case["target_hex"]),
                state,
                observed[case["id"]],
            )
            progress(
                args.output / "progress.jsonl",
                "teacher_completed",
                id=case["id"],
                teacher_offers=proposal["teacher_count"],
            )
            pid = f"proposal-{case['id']}"
            proposals[pid] = {
                "baseline_id": case["id"],
                "state_sha256": digest(state.snapshot()),
                "endpoint_sha256": endpoint_hashes["tagged"],
                "receipt": proposal,
            }
            assert digest(model.state_dict()) == endpoint_hashes["tagged"]
            for condition in [x for x in plan["cases"][32:] if x["proposal_id"] == pid]:
                guard()
                candidate = perturbed_model(model, proposal, condition["block"], condition["sign"])
                assert torch.equal(candidate.decoder.output_weights, model.decoder.output_weights)
                mid = condition["memory_id"]
                progress(
                    args.output / "progress.jsonl",
                    "ingestion_started",
                    kind="perturbed",
                    id=mid,
                    maximum_byte_steps=50,
                )
                fresh = candidate.ingest(documents)
                progress(
                    args.output / "progress.jsonl",
                    "ingestion_completed",
                    kind="perturbed",
                    id=mid,
                    byte_steps=50,
                )
                perturbations["models"][mid] = clone(candidate.state_dict())
                perturbations["memories"][mid] = {"model_id": mid, "state": clone(fresh.snapshot())}
        torch.save(proposals, args.output / "proposals.pt")
        report["proposal_teacher_offers"] = sum(
            p["receipt"]["teacher_count"] for p in proposals.values()
        )
        report["proposal_tensor_bytes"] = sum(tensor_bytes(p) for p in proposals.values())
        report["perturbed_retained_tensor_bytes"] = tensor_bytes(perturbations)
        del proposals
        second_receipt, changed = query_stage(
            args.output / "perturbed",
            perturbations,
            plan["cases"][32:],
            manifest,
            args.deadline_unix,
            guard,
        )
        report["workers"].append(second_receipt)
        del perturbations
        observed.update(changed)
        report["observation_tensor_bytes"] = sum(tensor_bytes(value) for value in observed.values())
        report["scores"] = [
            {"id": case["id"], **score(observed[case["id"]], case["target_hex"])}
            for case in plan["cases"]
        ]
        for row in report["scores"][32:]:
            base = report["scores"][plan["cases"][row["id"]]["baseline_id"]]
            row["mean_loss_difference"] = row["mean_loss"] - base["mean_loss"]
            row["phase_margin_differences"] = [
                x - y for x, y in zip(row["phase_margins"], base["phase_margins"], strict=True)
            ]
        report["finite_update_effects"] = [
            {
                "block": block,
                "sign": sign,
                "conditions": 8,
                "mean_loss_difference": sum(
                    report["scores"][case["id"]]["mean_loss_difference"]
                    for case in plan["cases"][32:]
                    if case["block"] == block and case["sign"] == sign
                )
                / 8,
            }
            for block in CONFIG["blocks"]
            for sign in CONFIG["signs"]
        ]
        outputs = first_receipt["outputs"] + second_receipt["outputs"]
        report["motor_demonstrations"], report["actions"] = actions(
            args.output, plan["cases"], outputs, guard
        )
        report["counts"] = {
            "ingestions": 56,
            "baseline_ingestions": 16,
            "proposal_ingestions": 8,
            "perturbed_ingestions": 32,
            "ingestion_byte_steps": 2800,
            "free_queries": len(outputs),
            "free_query_byte_steps": sum(w["query_byte_steps"] for w in report["workers"]),
            "teacher_query_traversals": 8,
            "teacher_query_byte_steps": 208,
            "free_generation_decisions": sum(x["decisions"] for x in outputs),
            "supplemental_phase_evaluations": sum(x["extra_phase_evaluations"] for x in outputs),
            "free_and_supplemental_phase_advances": 192,
            "teacher_phase_advances": 24,
            "teacher_score_validation_reevaluations": 24,
            "actual_output_score_evaluations": 240,
            "teacher_offers": report["proposal_teacher_offers"],
            "teacher_residual_offers": 6168,
            "actual_teacher_context_signal_projections": 24,
            "external_proposal_residual_vectors": 24,
            "external_proposal_signal_projections": 24,
            "external_target_scoring_phase_vectors": 192,
            "query_key_observation_recomputations": 64,
            "trace_writer_map_observation_recomputations": report["baseline_trace_steps"],
            "action_attempts": len(report["actions"]),
            "motor_demonstrations": 2,
        }
        assert (
            report["proposal_teacher_offers"] == 24 and len(outputs) == len(report["actions"]) == 64
        )
        assert (
            report["counts"]["free_generation_decisions"]
            + report["counts"]["supplemental_phase_evaluations"]
            == 192
        )
        assert {mode: digest(m.state_dict()) for mode, m in models.items()} == endpoint_hashes
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
        artifacts = {
            p.relative_to(args.output).as_posix(): file_hash(p)
            for p in args.output.rglob("*")
            if p.is_file() and p.name != "report.json"
        }
        write_json(
            args.output / "completion.json",
            {
                "status": "passed" if report["status"] == "completed" else "failed",
                "report_sha256": file_hash(args.output / "report.json"),
                "artifacts": artifacts,
                "artifact_bytes": {name: (args.output / name).stat().st_size for name in artifacts},
                "source_revision": args.revision,
                "admission_sha256": args.admission_sha,
                "elapsed_seconds": time.perf_counter() - START,
            },
        )
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
