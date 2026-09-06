"""Cost-gated successor using unchanged local learning and action contracts.

Frozen v1 helpers remain inert building blocks. Orchestration is copied because
v1 hardcodes its worker identity and an invalid Windows redirector PID check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = time.monotonic()
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated import local_memory_byte_v1 as previous  # noqa: E402

reference = previous.reference
file_hash, write_json = previous.file_hash, previous.write_json
detached_state, same_state, tensor_bytes = (
    previous.detached_state, previous.same_state, previous.tensor_bytes
)
digest, new_model, generation_record = (
    previous.digest, previous.new_model, previous.generation_record
)
lesson, export_memories = previous.lesson, previous.export_memories
ARMS = previous.ARMS
CONFIG = {**previous.CONFIG, "schema": "local-memory-byte-v2"}


def sources():
    names = (
        "experiments/r4_integrated/local_memory_byte_v2.py",
        "experiments/r4_integrated/local_memory_byte_v2.md",
    )
    return {**previous.sources(), **{name: file_hash(ROOT / name) for name in names}}


def validate_cost_launch(launch, report):
    """Bind actual Python to the launcher's observed, creation-checked tree.

    The existing launcher adds only the root and descendants created since the
    root started. Its identity keys include PID and creation ticks. Windows'
    virtualenv redirector can therefore differ from the report's actual PID.
    """
    assert launch["status"] == "passed" and launch["exit_code"] == 0
    command = launch["command"]
    assert Path(command[0]).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve()
    assert command[1:6] == [
        "-X", "utf8", "-m", "experiments.r4_integrated.local_memory_byte_v2", "cost"
    ]
    assert launch["started_unix"] == report["launch_started_unix"]
    assert launch["deadline_unix"] == report["deadline_unix"]
    assert 0 < launch["seconds_limit"] <= CONFIG["seconds"]
    assert launch["tree_peak_sum_mib_limit"] == CONFIG["parent_mib"]
    assert launch["launcher_sha256"] == report["source_sha256"][
        "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1"
    ]
    observed = launch["observed_process_peaks_mib"]
    for pid in (launch["child_pid"], report["pid"], report["worker"]["worker_pid"]):
        identities = [key for key in observed if key.split("/")[0] == str(pid)]
        assert len(identities) == 1, "process absent from owned tree or PID reused"
        assert int(identities[0].split("/")[1]) > 0
    arguments = dict(zip(command[6::2], command[7::2], strict=True))
    assert set(arguments) == {"--output", "--admission", "--admission-sha", "--revision"}
    assert arguments["--revision"] == report["source_revision"]
    assert arguments["--admission-sha"] == report["admission_sha256"]
    assert file_hash(Path(arguments["--admission"])) == report["admission_sha256"]
    assert json.loads((Path(arguments["--output"]) / "report.json").read_bytes()) == report


def project_cost(report, launch):
    """Recompute the original conservative formula; never accept a supplied estimate."""
    assert report["status"] == "completed" and report["mode"] == "cost"
    assert report["config"] == CONFIG
    assert len(report["lessons"]) == len(ARMS) * CONFIG["cost_lessons_per_arm"]
    assert len(report["actions"]) == 100 and len(report["motor_demonstrations"]) == 2
    assert report["counts"]["first_attempts"] == 20
    assert report["counts"]["teacher_byte_offers"] == 200
    assert report["counts"]["enabled_decoder_step_opportunities"] == 160

    def positive(value):
        assert math.isfinite(value) and value > 0
        return value

    per_arm = {}
    for arm in ARMS:
        rows = [row for row in report["lessons"] if row["arm"] == arm]
        assert len(rows) == CONFIG["cost_lessons_per_arm"]
        assert [row["index"] for row in rows] == report["selected_lesson_indices"]
        capped = report["cap_generation"][arm]
        assert capped["stop_reason"] == "limit" and capped["decisions"] == 33
        assert len(bytes.fromhex(capped["generated_hex"])) == CONFIG["max_bytes"]
        per_arm[arm] = {
            "max_complete_lesson_seconds": max(positive(row["seconds"]) for row in rows),
            "additional_capped_prediction_seconds": positive(capped["seconds"]),
        }
    learning = 2 * sum(CONFIG["lessons"] * sum(row.values()) for row in per_arm.values())
    sampled = sum(row["seconds"] for row in report["lessons"])
    other = 2 * positive(positive(launch["elapsed_seconds"]) - sampled)
    parent_worker = positive(report["memory"]["peak_working_set_mib"]) + positive(
        report["worker"]["memory"]["peak_working_set_mib"]
    )
    peak = max(parent_worker, positive(launch["observed_tree_peak_sum_mib"]))
    projected = learning + other + 30
    return {
        "per_arm": per_arm,
        "projected_learning_seconds": learning,
        "projected_other_seconds": other,
        "finalization_allowance_seconds": 30,
        "projected_seconds": projected,
        "parent_plus_worker_peak_mib": parent_worker,
        "peak_upper_mib": peak,
        "admissible": projected <= CONFIG["seconds"] and peak <= CONFIG["parent_mib"],
    }


def worker(args):
    """Query-only fresh process; source authoring and teaching are not invoked."""
    import torch

    from benchmarks.native_scaling import _process_memory

    torch.set_num_threads(1)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    requests = json.loads(args.requests.read_bytes())
    assert set(requests) == {"source_sha256", "cases"} and requests["source_sha256"] == sources()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(ARMS)
    rows, query_steps = [], 0
    for arm in ARMS:
        entry = saved[arm]
        assert set(entry) == {"weights", "memories"}
        model = new_model(arm)
        model.load_state_dict(entry["weights"], strict=True)
        states = {
            key: model.memory.restore_local_state(value) for key, value in entry["memories"].items()
        }
        for request in requests["cases"]:
            assert set(request) == {"id", "memory_id", "query_hex", "delay_reads"}
            if time.time() > args.deadline_unix:
                raise TimeoutError("worker deadline")
            if _process_memory()["peak_working_set_mib"] > CONFIG["parent_mib"]:
                raise MemoryError("worker peak budget")
            state = states[request["memory_id"]]
            before = digest(state.snapshot())
            with torch.no_grad():
                for _ in range(request["delay_reads"]):
                    model._query(b"What is the value of k99?", state)
                    query_steps += 26
                query = bytes.fromhex(request["query_hex"])
                begin = time.monotonic()
                generated = model.predict(query, state, CONFIG["max_bytes"])
                query_steps += len(query) + 1
            assert digest(state.snapshot()) == before
            rows.append(
                {
                    "arm": arm,
                    **request,
                    **generation_record(generated),
                    "seconds": time.monotonic() - begin,
                    "memory_sha256": before,
                }
            )
        assert same_state(entry["weights"], model.state_dict())
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha and sources() == requests["source_sha256"]
    write_json(
        args.output,
        {
            "status": "completed",
            "outputs": rows,
            "query_byte_steps": query_steps,
            "checkpoint_sha256": args.checkpoint_sha,
            "requests_sha256": args.requests_sha,
            "worker_pid": os.getpid(),
            "memory": _process_memory(),
            "elapsed_seconds": time.monotonic() - START,
            "roster_invoked_or_documents_or_targets_supplied_to_model": False,
            "routing_boundary_not_filesystem_sandbox": True,
        },
    )


def final_actions(args, checks, report, guard):
    isolated = args.output / "source-free"
    checkpoint, requests = isolated / "source_free.pt", isolated / "query_only.json"
    generated_path = isolated / "generated.json"
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "worker",
        "--output",
        str(generated_path),
        "--checkpoint",
        str(checkpoint),
        "--requests",
        str(requests),
        "--checkpoint-sha",
        file_hash(checkpoint),
        "--requests-sha",
        file_hash(requests),
        "--deadline-unix",
        str(args.deadline_unix),
    ]
    begin = time.monotonic()
    child = subprocess.run(
        command,
        cwd=isolated,
        capture_output=True,
        timeout=max(0.001, args.deadline_unix - time.time()),
    )
    (args.output / "worker.stdout").write_bytes(child.stdout)
    (args.output / "worker.stderr").write_bytes(child.stderr)
    report["worker"] = {
        "command": command,
        "returncode": child.returncode,
        "package_seconds": time.monotonic() - begin,
    }
    if child.returncode:
        raise RuntimeError("source-free worker failed")
    generated = json.loads(generated_path.read_bytes())
    report["worker"].update({k: generated[k] for k in ("memory", "worker_pid", "elapsed_seconds")})
    assert generated["checkpoint_sha256"] == file_hash(checkpoint)
    assert generated["requests_sha256"] == file_hash(requests)
    assert [(r["arm"], r["id"], r["memory_id"]) for r in generated["outputs"]] == [
        (a, c["id"], c["memory_id"]) for a in ARMS for c in checks
    ]
    assert generated["query_byte_steps"] == 19240
    guard()
    from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
    from experiments.r4_integrated.utf8_action_bridge import Case, execute_utf8

    begin = time.monotonic()
    motor = ContextualOccurrenceMotor()
    report["motor_demonstrations"] = []
    for case in (
        Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
        Case("motor-b", "bora", "ready", "motor_beta", "warm"),
    ):
        guard()
        before, after = case.files(), case.files(corrected=True)
        execution = execute_utf8(case, after, before, args.output / case.name)
        assert execution["passed"]
        motor.learn(f"Configure {case.key}".encode(), before, after)
        report["motor_demonstrations"].append(execution)
    for row in generated["outputs"]:
        guard()
        case = checks[row["id"]]
        action = reference.actuate(
            row, row["id"], motor, args.output / f"action-{row['arm']}-{row['id']:02d}"
        )
        report["actions"].append(
            {
                "arm": row["arm"],
                "id": row["id"],
                "memory_id": row["memory_id"],
                "group": case["group"],
                "correct": bool(
                    action["emitted"]
                    and row["generated_hex"] == case["expected_hex"]
                    and action["lossless_generated_execution"]
                ),
                **action,
            }
        )
    report["action_package_seconds"] = time.monotonic() - begin
    report["scores"] = {
        a: {
            g: sum(r["correct"] for r in report["actions"] if r["arm"] == a and r["group"] == g)
            for g in ("fit", "unseen_value", "delayed", "interference")
        }
        for a in ARMS
    }
    score = report["scores"]["tagged"]
    report["qualification"] = (
        args.mode == "run"
        and score == {"fit": 8, "unseen_value": 4, "delayed": 4, "interference": 4}
        and all(
            score["unseen_value"] > report["scores"][a]["unseen_value"]
            for a in ("writer_maps_frozen", "frozen", "writes_disabled")
        )
    )
    report["tag_specific_advantage"] = args.mode == "run" and (
        score["unseen_value"] > report["scores"]["aggregate"]["unseen_value"]
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "mode": args.mode,
        "config": CONFIG,
        "source_revision": args.revision,
        "admission_sha256": args.admission_sha,
        "pid": os.getpid(),
        "launch_started_unix": args.launch_started_unix,
        "deadline_unix": args.deadline_unix,
        "lessons": [],
        "actions": [],
        "action_scope": (
            "Generated answer bytes come from the local neuronal learner; the occurrence motor "
            "uses host-authored boundaries/candidate drives as a narrow action-schema scaffold. "
            "Execution does not establish entirely neuron-only action selection."
        ),
    }
    memory = None
    passed = False

    def guard():
        if time.time() > args.deadline_unix:
            raise TimeoutError("local development deadline")
        if memory is not None:
            peak = memory()["peak_working_set_mib"]
            peak += report.get("worker", {}).get("memory", {}).get("peak_working_set_mib", 0)
            if peak > CONFIG["parent_mib"]:
                raise MemoryError("conservative parent plus worker peak budget")

    try:
        assert file_hash(args.admission) == args.admission_sha
        admission = json.loads(args.admission.read_bytes())
        assert admission["status"] == ("engineering_qa" if args.mode == "cost" else "admitted")
        assert admission["config"] == CONFIG
        assert 0 < args.deadline_unix - args.launch_started_unix <= CONFIG["seconds"]
        assert args.launch_started_unix <= time.time() < args.deadline_unix
        assert all(
            os.environ.get(k) == "1"
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        )
        assert os.environ.get("HUMANBRAIN_DEVICE") == "cpu"
        assert (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            == args.revision
            == admission["source_revision"]
        )
        if args.mode == "run":
            cost = admission["cost"]
            assert file_hash(ROOT / cost["completion_path"]) == cost["completion_sha256"]
            completed = json.loads((ROOT / cost["completion_path"]).read_bytes())
            assert completed["status"] == "passed" and completed["mode"] == "cost"
            cost_report_path = (ROOT / cost["completion_path"]).with_name("report.json")
            assert file_hash(cost_report_path) == completed["report_sha256"]
            cost_report = json.loads(cost_report_path.read_bytes())
            assert cost_report["status"] == "completed" and cost_report["mode"] == "cost"
            assert cost_report["config"] == CONFIG
            assert cost_report["source_revision"] == args.revision
            for name in ("source_sha256", "packages", "native", "teaching_schedule_sha256"):
                assert cost_report[name] == admission[name]
            assert file_hash(ROOT / cost["launcher_path"]) == cost["launcher_sha256"]
            launch = json.loads((ROOT / cost["launcher_path"]).read_bytes())
            assert launch["status"] == "passed" and launch["exit_code"] == 0
            validate_cost_launch(launch, cost_report)
            projection = project_cost(cost_report, launch)
            assert cost["projected_seconds"] == projection["projected_seconds"]
            assert cost["peak_upper_mib"] == projection["peak_upper_mib"]
            assert launch["started_unix"] == cost_report["launch_started_unix"]
            assert launch["deadline_unix"] == cost_report["deadline_unix"]
            assert (
                launch["launcher_sha256"]
                == admission["source_sha256"][
                    "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1"
                ]
            )
            assert 0 < cost["projected_seconds"] <= CONFIG["seconds"]
            assert 0 < cost["peak_upper_mib"] <= CONFIG["parent_mib"]
        report["source_sha256"] = sources()
        assert report["source_sha256"] == admission["source_sha256"]
        report["packages"] = {name: version(name) for name in ("torch", "numpy")}
        assert report["packages"] == admission["packages"]
        import torch

        from benchmarks.native_scaling import _process_memory
        from humanbrain.native import loaded_library_identity

        memory = _process_memory
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        report["native"] = loaded_library_identity().to_record()
        assert report["native"] == admission["native"]
        train, checks = reference.roster()
        grouped = reference.teaching_schedule(train)
        schedule_sha = hashlib.sha256(
            json.dumps(grouped, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert schedule_sha == admission["teaching_schedule_sha256"]
        schedule = [episode for batch in grouped for episode in batch]
        report["teaching_schedule_sha256"] = schedule_sha
        indices = list(range(1024))
        if args.mode == "cost":
            indices = sorted(
                indices,
                key=lambda i: (
                    -sum(len(bytes.fromhex(x)) for x in schedule[i]["document_hex"]),
                    -len(bytes.fromhex(schedule[i]["target_hex"])),
                    -len(bytes.fromhex(schedule[i]["query_hex"])),
                    i,
                ),
            )[:4]
        report["selected_lesson_indices"] = indices
        write_json(
            args.output / "offline_teacher_roster.json",
            {"train": train, "checks": checks, "schedule": grouped, "selected_indices": indices},
        )
        models = {a: new_model(a) for a in ARMS}
        initial = {a: detached_state(m.state_dict()) for a, m in models.items()}

        def tensor_only(state):
            return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}

        assert all(
            same_state(tensor_only(initial["tagged"]), tensor_only(initial[a])) for a in ARMS
        )
        torch.save(initial, args.output / "initial_weights.pt")
        report["model_tensor_bytes"] = {a: tensor_bytes(s) for a, s in initial.items()}
        for index in indices:
            for arm in ARMS:
                record = lesson(models[arm], schedule[index], guard)
                report["lessons"].append({"arm": arm, "index": index, **record})
        assert same_state(initial["frozen"], models["frozen"].state_dict())
        fixed = [
            k
            for k in initial["writer_maps_frozen"]
            if k.startswith("memory.") and not k.startswith("memory.query_")
        ]
        assert all(
            same_state(
                initial["writer_maps_frozen"][k], models["writer_maps_frozen"].state_dict()[k]
            )
            for k in fixed
        )
        report["frozen_invariants"] = {"all_frozen": True, "writer_maps_frozen_keys": fixed}
        if args.mode == "cost":
            report["cap_generation"] = {}
            for arm, model in models.items():
                guard()
                weights = detached_state(model.state_dict())
                state = model.memory.initial_local_state(
                    1, credit_mode=("aggregate" if arm == "aggregate" else "tagged")
                )
                model.decoder.output.weight[256].zero_()
                model.decoder.output.bias[256] = torch.finfo(model.decoder.output.bias.dtype).min
                begin = time.monotonic()
                generated = model.predict(b"What is the value of k00?", state, CONFIG["max_bytes"])
                report["cap_generation"][arm] = {
                    "seconds": time.monotonic() - begin,
                    **generation_record(generated),
                }
                model.load_state_dict(weights)
                assert same_state(weights, model.state_dict()) and generated.stop_reason == "limit"
                assert len(generated.data) == 32
        isolated = args.output / "source-free"
        isolated.mkdir()
        export_memories(models, checks, isolated, report, guard)
        del models, initial
        final_actions(args, checks, report, guard)
        report["counts"] = {
            "first_attempts": len(report["lessons"]),
            "teacher_byte_offers": sum(x["teacher_count"] for x in report["lessons"]),
            "document_byte_steps": 5
            * sum(
                sum(len(bytes.fromhex(d)) + 1 for d in schedule[i]["document_hex"]) for i in indices
            ),
            "acquisition_query_byte_steps": 10
            * sum(len(bytes.fromhex(schedule[i]["query_hex"])) + 1 for i in indices),
            "final_query_byte_steps": 19240,
            "actions": len(report["actions"]),
            "motor_demos": 2,
        }
        report["counts"]["enabled_decoder_step_opportunities"] = sum(
            row["teacher_count"] for row in report["lessons"] if row["arm"] != "frozen"
        )
        report["counts"]["first_generation_decisions"] = sum(
            row["decisions"] for row in report["lessons"]
        )
        report["count_scope"] = (
            "Decoder changed flags are lesson-level ORs, not changed-step counts; "
            "teacher bytes and enabled steps are exposures/opportunities. "
            "Acquisition is byte-supervised, without action feedback."
        )
        assert len(report["lessons"]) == len(indices) * 5 and len(report["actions"]) == 100
        assert loaded_library_identity().to_record() == report["native"]
        guard()
        report["status"] = "completed"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        try:
            if "source_sha256" in report:
                report["source_sha256_after"] = sources()
                assert report["source_sha256_after"] == report["source_sha256"]
            if memory:
                report["memory"] = memory()
            report["elapsed_seconds"] = time.monotonic() - START
            write_json(args.output / "report.json", report)
            completion = {
                "status": "passed" if report["status"] == "completed" else "failed",
                "mode": args.mode,
                "source_revision": args.revision,
                "admission_sha256": args.admission_sha,
                "report_sha256": file_hash(args.output / "report.json"),
                "artifacts": {
                    p.relative_to(args.output).as_posix(): file_hash(p)
                    for p in (
                        *args.output.glob("source-free/*"),
                        args.output / "initial_weights.pt",
                        args.output / "offline_teacher_roster.json",
                    )
                    if p.is_file()
                },
            }
            try:
                guard()
            except BaseException:
                completion["status"] = "failed"
                completion["resource_error"] = traceback.format_exc()
            completion["elapsed_seconds"] = time.monotonic() - START
            completion["outer_elapsed_seconds"] = (
                time.time() - args.launch_started_unix if args.launch_started_unix else None
            )
            completion["memory"] = memory() if memory else None
            write_json(args.output / "completion.json", completion)
            passed = completion["status"] == "passed"
        except BaseException:
            write_json(
                args.output / "finalization-error.json",
                {
                    "original_error": report.get("error"),
                    "finalization_error": traceback.format_exc(),
                },
            )
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cost", "run", "worker"))
    for name in ("output", "admission", "checkpoint", "requests"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    for name in ("admission-sha", "revision", "checkpoint-sha", "requests-sha"):
        parser.add_argument("--" + name)
    parser.add_argument(
        "--launch-started-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_START_UNIX")
    )
    parser.add_argument(
        "--deadline-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_DEADLINE_UNIX")
    )
    arguments = parser.parse_args()
    if arguments.mode == "worker":
        worker(arguments)
    else:
        raise SystemExit(0 if run(arguments) else 1)
