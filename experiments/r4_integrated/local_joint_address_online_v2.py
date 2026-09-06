"""Same local online learner with qualified direct Python action execution."""

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
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = time.perf_counter()
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated import direct_python_action_bridge_v1 as action_bridge  # noqa: E402
from experiments.r4_integrated import local_joint_address_online_v1 as original  # noqa: E402

ARMS = original.ARMS
file_hash, write_json = original.file_hash, original.write_json
digest, generation_record = original.digest, original.generation_record
detached_state, same_state, tensor_bytes = (
    original.detached_state,
    original.same_state,
    original.tensor_bytes,
)
duration_clock, validate_cost_launch = original.duration_clock, original.validate_cost_launch
roster, teaching_schedule = original.roster, original.teaching_schedule
plain, add_work = original.plain, original.add_work
ingestion_work, restore_memory = original.ingestion_work, original.restore_memory
new_model, capped_prediction = original.new_model, original.capped_prediction
matched_control_advantage = original.matched_control_advantage
preserves_workload, qualifies = original.preserves_workload, original.qualifies
JsonlJournal, LessonJournal = original.JsonlJournal, original.LessonJournal
external_grade, read_row, host_tree_bytes = (
    original.external_grade,
    original.read_row,
    original.host_tree_bytes,
)


@dataclass(frozen=True)
class JointOnlineSpec(original.JointOnlineSpec):
    @property
    def config(self):
        return {
            **super().config,
            "action_execution_policy": action_bridge.POLICY,
            "accepted_action_probe": {
                "generated_hex": "6162",
                "stop_reason": "eos",
                "ordinal": 1023,
            },
            "cost_execution_receipts": 99,
            "full_execution_receipts": 4178,
        }


SPEC = JointOnlineSpec(
    schema="local-joint-address-online-v2",
    module="experiments.r4_integrated.local_joint_address_online_v2",
    entrypoint=Path(__file__).resolve(),
    recipe=Path(__file__).with_suffix(".md").resolve(),
)


def sources(spec=SPEC):
    paths = (
        "experiments/r4_integrated/local_joint_address_online_v1.py",
        "experiments/r4_integrated/direct_python_action_bridge_v1.py",
        "docs/research/local-action-execution-speed-review.md",
    )
    return {**original.sources(spec), **{path: file_hash(ROOT / path) for path in paths}}


def export_memories(spec, models, checks, isolated, report, guard, launch):
    original.export_memories(spec, models, checks, isolated, report, guard)
    begin = time.perf_counter()
    path = isolated / "query_only.json"
    request = json.loads(path.read_bytes())
    request["action_launch_record_sha256"] = launch.sha256
    write_json(path, request)
    report["action_identity_binding_seconds"] = time.perf_counter() - begin


def accepted_action_probe(spec, output, motor, guard, launch):
    """One supplied cost input covers the accepted motor branch, not a model success."""
    begin = time.perf_counter()
    guard()
    supplied = dict(spec.config["accepted_action_probe"])
    ordinal = supplied.pop("ordinal")
    action = action_bridge.actuate(
        supplied, ordinal, motor, output / "accepted-action-probe", launch=launch
    )
    grade = external_grade(supplied, action, b"ab")
    assert action["emitted"] and action["lossless_generated_execution"] and grade["task_success"]
    journal = JsonlJournal(output / "accepted_action_probe.jsonl", spec)
    try:
        journal.append(
            {"ordinal": ordinal, "stop_reason": supplied["stop_reason"], "grade": grade, **action}
        )
    finally:
        log = journal.close()
    result = dict(
        action_log=log,
        generated_hex="6162",
        stop_reason="eos",
        ordinal=1023,
        authored_cost_only=True,
        neuronal_predictions=0,
        teacher_offers=0,
    )
    result["action_seconds"] = time.perf_counter() - begin
    return result


def prepare_motor(output, guard, launch):
    from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
    from experiments.r4_integrated.direct_python_action_bridge_v1 import Case, execute_utf8

    motor, demos = ContextualOccurrenceMotor(), []
    for case in (
        Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
        Case("motor-b", "bora", "ready", "motor_beta", "warm"),
    ):
        guard()
        before, after = case.files(), case.files(corrected=True)
        receipt = execute_utf8(case, after, before, output / case.name, launch=launch)
        assert receipt["passed"]
        motor.learn(f"Configure {case.key}".encode(), before, after)
        demos.append(receipt)
    return motor, demos


def lesson(spec, arm, model, episode, guard, *, index, motor, action_root, action_journal, launch):
    """Same object/state predicts, executes, is graded, then receives correction."""
    begin = time.perf_counter()
    guard()
    documents = [bytes.fromhex(value) for value in episode["document_hex"]]
    query = bytes.fromhex(episode["query_hex"])
    before = digest(model.state_dict())
    state = model.ingest(documents)
    memory_before = digest(state.snapshot())
    prediction = model.predict_with_receipt(query, state, spec.config["max_bytes"])
    assert digest(model.state_dict()) == before and digest(state.snapshot()) == memory_before
    instance = f"{os.getpid()}:{id(model)}"
    identity = dict(
        arm=arm,
        index=index,
        attempt_id=f"{arm}:{index}",
        experience_ordinal=index,
        model_instance_token=instance,
    )
    model_seconds = time.perf_counter() - begin
    action_begin = time.perf_counter()
    first = generation_record(prediction.generation)
    action = action_bridge.actuate(
        first, index, motor, action_root / f"{arm}-{index:04d}", launch=launch
    )
    # Target decoding is after the actual output/action, never a model/motor input.
    target = bytes.fromhex(episode["target_hex"])
    grade = external_grade(first, action, target)
    action_row = {
        **identity,
        "before_sha256": before,
        "memory_sha256": memory_before,
        **first,
        "grade": grade,
        **action,
    }
    action_sha = action_journal.append(action_row)
    del action_row, action
    action_done = time.perf_counter()
    action_seconds = action_done - action_begin
    teacher_begin = time.perf_counter()
    guard()
    teacher_before, teacher_memory = digest(model.state_dict()), digest(state.snapshot())
    assert teacher_before == before and teacher_memory == memory_before
    assert instance == f"{os.getpid()}:{id(model)}"
    feedback = model.teach(query, target, state, spec.config["learning_rate"])
    assert feedback.teacher_count == 3
    assert digest(state.snapshot()) == digest(feedback.state.snapshot()) == memory_before
    after = digest(model.state_dict())
    if arm == "slow_frozen":
        assert before == after and not feedback.codec_updated and not feedback.address_updated
    teacher_done = time.perf_counter()
    row = {
        **identity,
        **first,
        "first_exact": grade["model_correct"],
        "task_success": grade["task_success"],
        "grade": grade,
        "teacher_count": feedback.teacher_count,
        "mean_loss": feedback.mean_loss,
        "total_loss": feedback.total_loss,
        "loss_kind": spec.config["loss_kind"],
        "codec_updated": feedback.codec_updated,
        "address_updated": feedback.address_updated,
        "codec_group_enabled": arm != "slow_frozen",
        "address_group_enabled": arm in {"learned", "writes_disabled"},
        "before_sha256": before,
        "after_sha256": after,
        "memory_sha256": memory_before,
        "teacher_before_sha256": teacher_before,
        "teacher_memory_sha256": teacher_memory,
        "action_row_sha256": action_sha,
        "action_completed_perf": action_done,
        "teacher_started_perf": teacher_begin,
        "teacher_completed_perf": teacher_done,
        "first_steps": prediction.steps,
        "teacher_steps": feedback.steps,
        "teacher_address_error": feedback.address_error,
        "retained_state_tensor_bytes": tensor_bytes(state.snapshot()),
        "work": add_work(ingestion_work(arm, documents), prediction.work, feedback.work),
        "action_seconds": action_seconds,
    }
    row["model_seconds"] = model_seconds + time.perf_counter() - teacher_begin
    return row


def project_cost(spec, report, launch, *, audit_launch=None, audit=None):
    """Separate model/action/journal maxima; final audit row work is measured too."""
    assert report["status"] == "completed" and report["mode"] == "cost"
    assert report["config"] == spec.config and report["duration_clock"] == duration_clock()
    assert report["counts"]["first_attempts"] == 16
    assert report["counts"]["teacher_byte_offers"] == 48
    assert report["counts"]["acquisition_actions"] == 16
    assert report["final_action_log"]["rows"] == 80 and len(report["motor_demonstrations"]) == 2

    def positive(value):
        value = float(value)
        assert math.isfinite(value) and value > 0
        return value

    probe = report["accepted_action_probe"]
    assert probe["authored_cost_only"] and probe["ordinal"] == 1023
    assert probe["generated_hex"] == "6162" and probe["stop_reason"] == "eos"
    assert probe["neuronal_predictions"] == probe["teacher_offers"] == 0
    assert probe["action_log"]["rows"] == 1
    assert report["counts"]["total_execution_receipts"] == 99
    probe_seconds = positive(probe["action_seconds"])
    per_arm = {}
    for arm in ARMS:
        summary = report["lesson_summary"][arm]
        assert summary["rows"] == 4
        cap = report["cap_generation"][arm]
        assert cap["stop_reason"] == "limit" and cap["decisions"] == 3
        assert len(bytes.fromhex(cap["generated_hex"])) == 2
        per_arm[arm] = {
            "max_model_seconds": positive(summary["max_model_seconds"]),
            "max_action_seconds": max(positive(summary["max_action_seconds"]), probe_seconds),
            "sampled_max_action_seconds": positive(summary["max_action_seconds"]),
            "max_journal_seconds": positive(summary["max_journal_seconds"]),
            "additional_capped_prediction_seconds": positive(cap["seconds"]),
        }
    learning = (
        2
        * 1024
        * sum(
            row["max_model_seconds"]
            + row["max_action_seconds"]
            + row["max_journal_seconds"]
            + row["additional_capped_prediction_seconds"]
            for row in per_arm.values()
        )
    )
    measured = sum(
        row["model_seconds"] + row["action_seconds"] + row["journal_seconds"]
        for row in report["lesson_summary"].values()
    )
    other = 2 * positive(positive(launch["elapsed_seconds"]) - measured)
    parent_worker = positive(report["memory"]["peak_working_set_mib"]) + positive(
        report["worker"]["memory"]["peak_working_set_mib"]
    )
    limits = spec.config["receipt_limits"]
    allowance = (
        limits["allocation_factor"] * limits["max_live_rows"] * limits["max_row_tree_bytes"]
        + limits["max_line_copies"] * limits["max_line_bytes"]
    )
    assert allowance == 3 * 2**20 and spec.config["full_only_checkpoint_bytes"] == 0
    run_peak = (
        max(parent_worker, positive(launch["observed_tree_peak_sum_mib"]))
        + (allowance + spec.config["run_memory_reserve_bytes"]) / 2**20
    )
    audit_peak, audit_seconds = None, None
    assert (audit is None) == (audit_launch is None)
    if audit is not None:
        assert (
            audit["status"] == audit_launch["status"] == "passed" and audit_launch["exit_code"] == 0
        )
        pairs = audit["learning_pair_timing"]
        assert set(pairs) == set(ARMS) and all(row["rows"] == 4 for row in pairs.values())
        for row in pairs.values():
            maximum, total = positive(row["max_seconds"]), positive(row["seconds"])
            assert maximum <= total <= 4 * maximum + 1e-9
        residual = positive(audit_launch["elapsed_seconds"]) - sum(
            row["seconds"] for row in pairs.values()
        )
        audit_seconds = 2 * (
            1024 * sum(row["max_seconds"] for row in pairs.values()) + positive(residual)
        )
        audit_peak = (
            positive(audit_launch["observed_tree_peak_sum_mib"])
            + (allowance + spec.config["audit_memory_reserve_bytes"]) / 2**20
        )
    projected = learning + other + 30 + (audit_seconds or 0)
    peak = max(run_peak, audit_peak or 0)
    return {
        "per_arm": per_arm,
        "accepted_probe_action_seconds": probe_seconds,
        "projected_learning_seconds": learning,
        "projected_other_seconds": other,
        "finalization_allowance_seconds": 30,
        "projected_run_seconds": learning + other + 30,
        "projected_audit_seconds": audit_seconds,
        "projected_seconds": projected,
        "parent_plus_worker_peak_mib": parent_worker,
        "receipt_memory_projection": {
            "streaming_allowance_bytes": allowance,
            "run_engineering_reserve_bytes": spec.config["run_memory_reserve_bytes"],
            "audit_engineering_reserve_bytes": spec.config["audit_memory_reserve_bytes"],
            "full_only_checkpoint_bytes": 0,
            "run_projected_peak_mib": run_peak,
            "audit_projected_peak_mib": audit_peak,
        },
        "audit_peak_pending": audit is None,
        "peak_upper_mib": peak,
        "admissible": audit is not None and projected <= 900 and peak <= 2048,
    }


def worker(spec, args):
    import torch

    from benchmarks.native_scaling import _process_memory

    torch.set_num_threads(1)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    requests = json.loads(args.requests.read_bytes())
    assert set(requests) == {"source_sha256", "cases", "action_launch_record_sha256"} and requests[
        "source_sha256"
    ] == sources(spec)
    assert isinstance(requests["action_launch_record_sha256"], str)
    assert len(requests["action_launch_record_sha256"]) == 64
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(ARMS)
    journal = JsonlJournal(args.output.with_suffix(".jsonl"), spec)
    work_by_arm = {}
    try:
        for arm in ARMS:
            entry = saved[arm]
            assert set(entry) == {"weights", "memories"} and len(entry["memories"]) == 12
            model = new_model(spec, arm)
            model.load_state_dict(entry["weights"], strict=True)
            states = {
                key: restore_memory(model, arm, value) for key, value in entry["memories"].items()
            }
            total = {}
            for request in requests["cases"]:
                assert set(request) == {"id", "memory_id", "query_hex", "delay_reads"}
                begin = time.perf_counter()
                if time.time() > args.deadline_unix:
                    raise TimeoutError("worker deadline")
                if _process_memory()["peak_working_set_mib"] > spec.config["parent_mib"]:
                    raise MemoryError("worker peak budget")
                state = states[request["memory_id"]]
                before = digest(state.snapshot())
                work = {}
                for _ in range(request["delay_reads"]):
                    work = add_work(
                        work, model.query_receipt(b"What is the value of k99?", state).work
                    )
                receipt = model.predict_with_receipt(
                    bytes.fromhex(request["query_hex"]), state, spec.config["max_bytes"]
                )
                assert digest(state.snapshot()) == before
                work = add_work(work, receipt.work)
                model_seconds = time.perf_counter() - begin
                journal_begin = time.perf_counter()
                row = {
                    "arm": arm,
                    **request,
                    **generation_record(receipt.generation),
                    "memory_sha256": before,
                    "steps": plain(receipt.steps),
                    "work": work,
                    "model_seconds": model_seconds,
                }
                journal.append(row)
                del row, receipt
                total = add_work(total, work)
                # Final workload is identically sized in cost/full; package timer covers this I/O.
                assert time.perf_counter() >= journal_begin
            assert same_state(entry["weights"], model.state_dict())
            assert all(
                same_state(entry["memories"][key], state.snapshot())
                for key, state in states.items()
            )
            work_by_arm[arm] = total
            del model, states, state
    finally:
        log = journal.close()
    assert (
        file_hash(args.checkpoint) == args.checkpoint_sha
        and file_hash(args.requests) == args.requests_sha
    )
    assert sources(spec) == requests["source_sha256"]
    write_json(
        args.output,
        {
            "status": "completed",
            "output_log": log,
            "work": work_by_arm,
            "query_byte_steps": sum(
                row["query_raw_bytes"] + row["query_eos_symbols"] for row in work_by_arm.values()
            ),
            "action_launch_record_sha256": requests["action_launch_record_sha256"],
            "checkpoint_sha256": args.checkpoint_sha,
            "requests_sha256": args.requests_sha,
            "worker_pid": os.getpid(),
            "memory": _process_memory(),
            "elapsed_seconds": time.perf_counter() - START,
            "duration_clock": duration_clock(),
            "roster_invoked_or_documents_or_targets_supplied_to_model": False,
            "routing_boundary_not_filesystem_sandbox": True,
            "timing_scope": "model includes hashes/delay; package includes conversion/journal",
        },
    )


def final_action(spec, stream, journal, ordinal, checks, motor, output, launch):
    """Bounded scope releases this prediction/action pair before the next read."""
    begin = time.perf_counter()
    row = read_row(stream, spec)
    if row is None:
        return None
    arm, case = ARMS[ordinal // len(checks)], checks[ordinal % len(checks)]
    assert (row["arm"], row["id"], row["memory_id"]) == (arm, case["id"], case["memory_id"])
    action = action_bridge.actuate(
        row, row["id"], motor, output / f"action-{arm}-{row['id']:02d}", launch=launch
    )
    grade = external_grade(row, action, bytes.fromhex(case["expected_hex"]))
    journal.append(
        {
            "arm": arm,
            "id": row["id"],
            "memory_id": row["memory_id"],
            "group": case["group"],
            "memory_sha256": row["memory_sha256"],
            "grade": grade,
            **action,
        }
    )
    return arm, case["group"], grade["task_success"], time.perf_counter() - begin


def final_actions(spec, args, checks, report, guard, motor, launch):
    isolated = args.output / "source-free"
    checkpoint, requests = isolated / "source_free.pt", isolated / "query_only.json"
    generated_path = isolated / "generated.json"
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(spec.entrypoint),
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
    begin = time.perf_counter()
    child = subprocess.run(
        command,
        cwd=isolated,
        capture_output=True,
        timeout=max(0.001, args.deadline_unix - time.time()),
    )
    (args.output / "worker.stdout").write_bytes(child.stdout)
    (args.output / "worker.stderr").write_bytes(child.stderr)
    report["worker"] = dict(
        command=command, returncode=child.returncode, package_seconds=time.perf_counter() - begin
    )
    if child.returncode:
        raise RuntimeError("source-free worker failed")
    generated = json.loads(generated_path.read_bytes())
    assert generated["duration_clock"] == report["duration_clock"]
    assert generated["action_launch_record_sha256"] == launch.sha256
    report["worker"].update(
        {key: generated[key] for key in ("memory", "worker_pid", "elapsed_seconds")}
    )
    assert generated["checkpoint_sha256"] == file_hash(checkpoint)
    assert generated["requests_sha256"] == file_hash(requests)
    assert generated["output_log"]["path"] == "generated.jsonl"
    log_path = isolated / "generated.jsonl"
    assert file_hash(log_path) == generated["output_log"]["sha256"]
    assert generated["output_log"]["rows"] == 80 and generated["query_byte_steps"] == 15392
    report["scores"] = {
        arm: dict.fromkeys(("fit", "unseen_value", "delayed", "interference"), 0) for arm in ARMS
    }
    journal = JsonlJournal(args.output / "actions.jsonl", spec)
    begin = time.perf_counter()
    try:
        with log_path.open("rb") as stream:
            for ordinal in range(80):
                guard()
                result = final_action(
                    spec, stream, journal, ordinal, checks, motor, args.output, launch
                )
                assert result is not None
                arm, group, correct, _ = result
                report["scores"][arm][group] += int(correct)
            assert read_row(stream, spec) is None
    finally:
        report["final_action_log"] = journal.close()
    report["action_package_seconds"] = time.perf_counter() - begin
    report["qualification"] = args.mode == "run" and qualifies(report["scores"])
    report["compliant_executed_learning_preserved"] = args.mode == "run" and preserves_workload(
        report["scores"]
    )
    report["learned_retrieval_advantage"] = args.mode == "run" and matched_control_advantage(
        report["scores"]
    )


def run(spec, args):
    assert spec.config["arms"] == list(ARMS)
    if not args.output.is_absolute() or not args.admission.is_absolute():
        raise ValueError("output and admission paths must be absolute")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "mode": args.mode,
        "config": spec.config,
        "source_revision": args.revision,
        "admission_sha256": args.admission_sha,
        "pid": os.getpid(),
        "launch_started_unix": args.launch_started_unix,
        "deadline_unix": args.deadline_unix,
        "duration_clock": duration_clock(),
        "action_scope": (
            "Generated answer bytes come from the local neuronal learner; the occurrence motor "
            "uses host-authored boundaries/candidate drives as a narrow action-schema scaffold. "
            "Execution does not establish entirely neuron-only action selection."
        ),
    }
    memory = None
    journal = None
    action_journal = None
    passed = False
    action_launch = None

    def guard():
        if time.time() > args.deadline_unix:
            raise TimeoutError("local development deadline")
        if memory is not None:
            peak = memory()["peak_working_set_mib"]
            peak += report.get("worker", {}).get("memory", {}).get("peak_working_set_mib", 0)
            if peak > spec.config["parent_mib"]:
                raise MemoryError("conservative parent plus worker peak budget")

    try:
        assert file_hash(args.admission) == args.admission_sha
        admission = json.loads(args.admission.read_bytes())
        assert admission["status"] == ("engineering_qa" if args.mode == "cost" else "admitted")
        assert admission["config"] == spec.config
        execution = admission["action_execution"]
        action_launch = action_bridge.load_qualified_launch(
            Path(execution["qualification_path"]),
            execution["qualification_sha256"],
            ROOT / ".venv/Scripts/python.exe",
        )
        assert action_launch.to_record() == execution["launch_record"]
        report["action_execution"] = execution
        assert admission["duration_clock"] == report["duration_clock"]
        assert 0 < args.deadline_unix - args.launch_started_unix <= spec.config["seconds"]
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
            assert cost_report["config"] == spec.config
            assert cost_report["source_revision"] == args.revision
            for name in (
                "action_execution",
                "source_sha256",
                "packages",
                "native",
                "teaching_schedule_sha256",
                "duration_clock",
            ):
                assert cost_report[name] == admission[name]
            assert file_hash(ROOT / cost["launcher_path"]) == cost["launcher_sha256"]
            launch = json.loads((ROOT / cost["launcher_path"]).read_bytes())
            assert launch["status"] == "passed" and launch["exit_code"] == 0
            validate_cost_launch(spec, launch, cost_report)
            audit_binding = admission["independent_audit"]
            audit_path = ROOT / audit_binding["path"]
            audit_launch_path = ROOT / audit_binding["launcher_path"]
            assert file_hash(audit_path) == audit_binding["sha256"]
            assert file_hash(audit_launch_path) == audit_binding["launcher_sha256"]
            audit = json.loads(audit_path.read_bytes())
            audit_launch = json.loads(audit_launch_path.read_bytes())
            checker = admission["independent_auditors"]["cost"]
            assert file_hash(ROOT / checker["path"]) == checker["sha256"]
            assert audit["status"] == "passed" and audit["checker_sha256"] == checker["sha256"]
            bound_inputs = {
                str((ROOT / path).resolve()): sha for path, sha in audit["inputs"].items()
            }
            for path in (
                cost_report_path,
                ROOT / cost["completion_path"],
                ROOT / cost["launcher_path"],
            ):
                assert bound_inputs[str(path.resolve())] == file_hash(path)
            assert audit["projection"] == project_cost(spec, cost_report, launch)
            command = audit_launch["command"]
            assert len(command) == 14 and command[1:3] == ["-X", "utf8"]
            assert Path(command[0]).resolve() == ROOT / ".venv/Scripts/python.exe"
            assert Path(command[3]).resolve() == (ROOT / checker["path"]).resolve()
            flags = dict(zip(command[4::2], command[5::2], strict=True))
            assert set(flags) == {
                "--output",
                "--cost-dir",
                "--launch",
                "--admission",
                "--admission-sha",
            }
            for flag, path in (
                ("--output", audit_path),
                ("--cost-dir", cost_report_path.parent),
                ("--launch", ROOT / cost["launcher_path"]),
            ):
                assert (
                    Path(flags[flag]).is_absolute()
                    and Path(flags[flag]).resolve() == path.resolve()
                )
            base_admission_path = Path(flags["--admission"])
            assert base_admission_path.is_absolute()
            assert file_hash(base_admission_path) == flags["--admission-sha"]
            assert flags["--admission-sha"] == cost_report["admission_sha256"]
            assert bound_inputs[str(base_admission_path.resolve())] == flags["--admission-sha"]
            assert audit_launch["launcher_sha256"] == launch["launcher_sha256"]
            assert audit_launch["seconds_limit"] == 900
            assert audit_launch["tree_peak_sum_mib_limit"] == 2048
            assert audit_launch["deadline_unix"] - audit_launch["started_unix"] == 900
            assert launch["elapsed_seconds"] + audit_launch["elapsed_seconds"] <= 900
            for pid in (audit_launch["child_pid"], audit["pid"]):
                identities = [
                    key
                    for key in audit_launch["observed_process_peaks_mib"]
                    if key.split("/")[0] == str(pid)
                ]
                assert len(identities) == 1
                created = (int(identities[0].split("/")[1]) - 621355968000000000) / 10_000_000
                assert (
                    audit_launch["started_unix"]
                    <= created
                    <= (audit_launch["started_unix"] + audit_launch["elapsed_seconds"])
                )
            projection = project_cost(
                spec, cost_report, launch, audit_launch=audit_launch, audit=audit
            )
            assert projection == admission["projection"] and projection["admissible"]
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
            assert 0 < cost["projected_seconds"] <= spec.config["seconds"]
            assert 0 < cost["peak_upper_mib"] <= spec.config["parent_mib"]
        report["source_sha256"] = sources(spec)
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
        train, checks = roster()
        grouped = teaching_schedule(train)
        schedule_sha = hashlib.sha256(
            json.dumps(grouped, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert schedule_sha == admission["teaching_schedule_sha256"]
        schedule = [episode for batch in grouped for episode in batch]
        report["teaching_schedule_sha256"] = schedule_sha
        indices = list(range(1024))
        if args.mode == "cost":
            indices = spec.config["cost_indices"]
        report["selected_lesson_indices"] = indices
        write_json(
            args.output / "offline_teacher_roster.json",
            {"train": train, "checks": checks, "schedule": grouped, "selected_indices": indices},
        )
        models = {arm: new_model(spec, arm) for arm in ARMS}
        initial = {arm: detached_state(model.state_dict()) for arm, model in models.items()}
        for arm in ARMS[:4]:
            tensor_keys = {key for key, value in initial[arm].items() if torch.is_tensor(value)}
            assert tensor_keys == {
                key for key, value in initial["learned"].items() if torch.is_tensor(value)
            }
            for key in tensor_keys:
                assert same_state(initial[arm][key], initial["learned"][key])
        report["common_initial_tensor_keys"] = sorted(tensor_keys)
        torch.save(initial, args.output / "initial_weights.pt")
        report["model_tensor_bytes"] = {arm: tensor_bytes(state) for arm, state in initial.items()}
        report["tape_accounting"] = models["learned"].memory.accounting(
            models["learned"].memory.initial_state()
        )
        report["joint_model_accounting"] = models["learned"].accounting()
        motor, report["motor_demonstrations"] = prepare_motor(args.output, guard, action_launch)
        action_journal = JsonlJournal(args.output / "acquisition_actions.jsonl", spec)
        journal = LessonJournal(args.output / "lessons.jsonl", spec)
        report["lesson_summary"] = journal.summary
        report["counts"] = journal.counts
        for completed_indices, index in enumerate(indices, 1):
            for arm in ARMS:
                row = lesson(
                    spec,
                    arm,
                    models[arm],
                    schedule[index],
                    guard,
                    index=index,
                    motor=motor,
                    action_root=args.output / "acquisition",
                    action_journal=action_journal,
                    launch=action_launch,
                )
                journal.append(row)
                del row
            if completed_indices % 128 == 0:
                print(
                    f"completed_lessons={journal.rows} "
                    f"elapsed_seconds={time.perf_counter() - START:.3f}",
                    flush=True,
                )
        report["lesson_log"] = journal.close()
        journal = None
        report["acquisition_action_log"] = action_journal.close()
        action_journal = None
        assert same_state(initial["slow_frozen"], models["slow_frozen"].state_dict())
        address_fixed = ["address_weight"]
        assert all(
            same_state(initial["address_frozen"][key], models["address_frozen"].state_dict()[key])
            for key in address_fixed
        )
        assert all(
            same_state(initial[arm]["input_codes"], models[arm].state_dict()["input_codes"])
            for arm in ARMS
        )
        report["frozen_invariants"] = {
            "slow_frozen": True,
            "address_frozen_keys": address_fixed,
            "input_codes_fixed": True,
        }
        report["accepted_action_probe"] = None
        if args.mode == "cost":
            report["accepted_action_probe"] = accepted_action_probe(
                spec, args.output, motor, guard, action_launch
            )
            report["cap_generation"] = {
                arm: capped_prediction(spec, arm, model) for arm, model in models.items()
            }
        isolated = args.output / "source-free"
        isolated.mkdir()
        export_memories(spec, models, checks, isolated, report, guard, action_launch)
        del models, initial
        final_actions(spec, args, checks, report, guard, motor, action_launch)
        generated = json.loads((isolated / "generated.json").read_bytes())
        report["work_by_arm"] = {
            arm: {
                "acquisition": report["lesson_summary"][arm]["work"],
                "final_ingestion": report["final_ingestion_work"][arm],
                "final_query": generated["work"][arm],
                **(
                    {
                        "cap_ingestion": report["cap_generation"][arm]["ingestion_work"],
                        "cap_query": report["cap_generation"][arm]["work"],
                    }
                    if args.mode == "cost"
                    else {}
                ),
            }
            for arm in ARMS
        }
        total_work = add_work(
            *(work for stages in report["work_by_arm"].values() for work in stages.values())
        )
        report["counts"].update(
            actions=report["final_action_log"]["rows"],
            motor_demos=2,
            accepted_action_probes=int(args.mode == "cost"),
            total_execution_receipts=report["acquisition_action_log"]["rows"]
            + 82
            + int(args.mode == "cost"),
        )
        assert report["counts"]["acquisition_actions"] == report["lesson_log"]["rows"]
        assert report["acquisition_action_log"]["rows"] == report["lesson_log"]["rows"]
        report["work_total"] = total_work
        report["count_scope"] = (
            "Candidate teacher offers and two held group opportunities are separate from actual "
            "changes; all acquisition outputs execute before same-instance supervised teaching. "
            "query-only distractions, "
            "branches, codec passes and cost-only cap ingestions are counted. Literal sensory "
            "content is retained in neuronal synapses; no source-file or target-routing claim "
            "implies compression."
        )
        assert (
            report["counts"]["total_execution_receipts"]
            == spec.config[
                "cost_execution_receipts" if args.mode == "cost" else "full_execution_receipts"
            ]
        )
        report["status"] = "completed"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        try:
            if journal is not None:
                report["lesson_log"] = journal.close()
            if action_journal is not None:
                report["acquisition_action_log"] = action_journal.close()
            if action_launch is not None:
                action_launch.verify()
                assert (
                    file_hash(Path(report["action_execution"]["qualification_path"]))
                    == report["action_execution"]["qualification_sha256"]
                )
            if "source_sha256" in report:
                report["source_sha256_after"] = sources(spec)
                assert report["source_sha256_after"] == report["source_sha256"]
            if memory:
                report["memory"] = memory()
            report["elapsed_seconds"] = time.perf_counter() - START
            write_json(args.output / "report.json", report)
            completion = {
                "status": "passed" if report["status"] == "completed" else "failed",
                "mode": args.mode,
                "source_revision": args.revision,
                "admission_sha256": args.admission_sha,
                "report_sha256": file_hash(args.output / "report.json"),
                "lesson_log": report.get("lesson_log"),
                "acquisition_action_log": report.get("acquisition_action_log"),
                "final_action_log": report.get("final_action_log"),
                "artifacts": {
                    p.relative_to(args.output).as_posix(): file_hash(p)
                    for p in (
                        *args.output.glob("source-free/*"),
                        args.output / "initial_weights.pt",
                        args.output / "offline_teacher_roster.json",
                        args.output / "lessons.jsonl",
                        args.output / "acquisition_actions.jsonl",
                        args.output / "actions.jsonl",
                        args.output / "accepted_action_probe.jsonl",
                    )
                    if p.is_file()
                },
            }
            try:
                guard()
            except BaseException:
                completion["status"] = "failed"
                completion["resource_error"] = traceback.format_exc()
            completion["elapsed_seconds"] = time.perf_counter() - START
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


def main(spec=SPEC):
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
    for name in ("output", "admission", "checkpoint", "requests"):
        value = getattr(arguments, name)
        if value is not None and not value.is_absolute():
            parser.error(f"--{name} must be an absolute path")
    if arguments.mode == "worker":
        worker(spec, arguments)
    else:
        raise SystemExit(0 if run(spec, arguments) else 1)


if __name__ == "__main__":
    main()
