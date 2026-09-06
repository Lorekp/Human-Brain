"""Fixed five-arm shared temporal-query retrieval and source-free action return."""

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

from experiments.r4_integrated import local_lag_query_v1 as preceding  # noqa: E402
from experiments.r4_integrated import local_memory_two_byte_v1 as previous  # noqa: E402

ARMS = (
    "learned",
    "address_frozen",
    "slow_frozen",
    "writes_disabled",
    "dense_reference",
)
file_hash, write_json = previous.file_hash, previous.write_json
digest, generation_record = previous.digest, previous.generation_record
detached_state, same_state, tensor_bytes = (
    previous.detached_state,
    previous.same_state,
    previous.tensor_bytes,
)
duration_clock, validate_cost_launch = previous.duration_clock, previous.validate_cost_launch
roster, teaching_schedule = previous.roster, previous.teaching_schedule


@dataclass(frozen=True)
class SharedTemporalSpec(previous.operations.ExperimentSpec):
    @property
    def config(self):
        return {
            "schema": self.schema,
            "seed": 733,
            "lessons": 1024,
            "learning_rate": 0.01,
            "capacity": 128,
            "lags": 32,
            "input_code_size": 32,
            "query_size": 1024,
            "query_energy": 32,
            "query_lags": 32,
            "output_neurons": 257,
            "codec_identity_probability": 0.5,
            "max_bytes": 2,
            "seconds": 900,
            "parent_mib": 2048,
            "cost_lessons_per_arm": 4,
            "retained_memories_per_arm": 12,
            "cost_sample": (
                "descending document bytes, teacher bytes, query bytes; original index tie"
            ),
            "query_traversals_per_lesson": 2,
            "receipt_format": "one complete scientific lesson per UTF8 JSONL record",
            "receipt_limits": {
                "max_line_bytes": 262144,
                "max_row_tree_bytes": 262144,
                "max_live_rows": 4,
                "max_line_copies": 4,
                "allocation_factor": 2,
            },
            "run_memory_reserve_bytes": 134217728,
            "audit_memory_reserve_bytes": 33554432,
            "historical_full_checkpoint_bytes": {
                "dense_reference": 105236476,
            },
            "arms": list(ARMS),
            "loss_kind": {arm: "sequence_mixture_negative_log_likelihood" for arm in ARMS},
            "dense_reference_revision": "8ef70d6f512caffc0cbc9eac2f19b772298447b2",
            "temporal_map_shape": [32, 32],
            "code_channels": 32,
            "cap_document_hex": (b"." * 127).hex(),
        }


SPEC = SharedTemporalSpec(
    schema="local-shared-temporal-query-v1",
    module="experiments.r4_integrated.local_shared_temporal_query_v1",
    entrypoint=Path(__file__).resolve(),
    recipe=Path(__file__).with_suffix(".md").resolve(),
)


def sources(spec=SPEC):
    paths = (
        "experiments/r4_integrated/local_lag_query_v1.py",
        "experiments/r4_integrated/local_lag_query_reference.py",
        "docs/research/local-shared-temporal-query-review.md",
    )
    return {**preceding.sources(spec), **{path: file_hash(ROOT / path) for path in paths}}


def new_model(spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown shared temporal experiment arm")
    if arm == "dense_reference":
        from experiments.r4_integrated.local_lag_query_reference import LocalMemoryLearner

        return LocalMemoryLearner(mode="learned", seed=spec.config["seed"]).eval()
    from humanbrain.local_memory_learner import LocalMemoryLearner

    return LocalMemoryLearner(mode=arm, seed=spec.config["seed"]).eval()


plain, add_work = preceding.plain, preceding.add_work
ingestion_work, restore_memory = preceding.ingestion_work, preceding.restore_memory
free_attempt, capped_prediction = preceding.free_attempt, preceding.capped_prediction
export_memories = preceding.export_memories


def lesson(spec, arm, model, episode, guard):
    # Exact dense scientific receipt, with actual supplied model, never old globals.
    return preceding.lesson(
        spec, "learned" if arm == "dense_reference" else arm, model, episode, guard
    )


def matched_control_advantage(scores):
    return all(
        scores["learned"]["unseen_value"] > scores[arm]["unseen_value"] for arm in ARMS[1:4]
    )


def qualifies(scores):
    learned, reference = scores["learned"], scores["dense_reference"]
    return (
        learned == {"fit": 8, "unseen_value": 4, "delayed": 4, "interference": 4}
        and matched_control_advantage(scores)
        and all(learned[group] >= reference[group] for group in learned)
        and sum(learned.values()) > sum(reference.values())
    )


def host_tree_bytes(value):
    """Conservative no-alias-dedup CPython receipt payload, never tensor/model data."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(host_tree_bytes(k) + host_tree_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(host_tree_bytes(item) for item in value)
    if type(value) not in (str, int, float, bool, type(None)):
        raise TypeError("journal rows must contain JSON primitives only")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("journal scalar must be finite")
    return size


class LessonJournal:
    """One live row plus encoding, full receipts on disk and fixed scalar summaries."""

    def __init__(self, path, spec):
        self.path, self.limits = path, spec.config["receipt_limits"]
        self.stream = path.open("xb")
        self.hasher = hashlib.sha256()
        self.rows = self.bytes = self.max_line = self.max_tree = 0
        self.summary = {
            arm: {
                "rows": 0,
                "seconds": 0.0,
                "max_seconds": 0.0,
                "journal_seconds": 0.0,
                "max_journal_seconds": 0.0,
                "logged_seconds": 0.0,
                "max_logged_seconds": 0.0,
                "first_exact": 0,
                "work": {},
            }
            for arm in ARMS
        }
        self.counts = {
            "first_attempts": 0,
            "first_generation_decisions": 0,
            "teacher_byte_offers": 0,
            "enabled_teacher_offers": 0,
        }

    def append(self, row):
        begin = time.perf_counter()
        tree_bytes = host_tree_bytes(row)
        if tree_bytes > self.limits["max_row_tree_bytes"]:
            raise MemoryError("lesson receipt exceeds fixed host-tree limit")
        encoded = (
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(encoded) > self.limits["max_line_bytes"]:
            raise MemoryError("lesson receipt exceeds fixed line limit")
        self.stream.write(encoded)
        self.stream.flush()
        self.hasher.update(encoded)
        self.rows += 1
        self.bytes += len(encoded)
        self.max_line, self.max_tree = (
            max(self.max_line, len(encoded)),
            max(self.max_tree, tree_bytes),
        )
        arm, row_seconds = row["arm"], row["seconds"]
        summary = self.summary[arm]
        summary["rows"] += 1
        summary["seconds"] += row_seconds
        summary["max_seconds"] = max(summary["max_seconds"], row_seconds)
        summary["first_exact"] += row["first_exact"]
        summary["work"] = add_work(summary["work"], row["work"])
        self.counts["first_attempts"] += 1
        self.counts["first_generation_decisions"] += row["decisions"]
        self.counts["teacher_byte_offers"] += row["teacher_count"]
        if arm != "slow_frozen":
            self.counts["enabled_teacher_offers"] += row["teacher_count"]
        prefix = "dense_reference_" if arm == "dense_reference" else ""
        for group in ("codec", "address"):
            for suffix, amount in (
                ("group_offers", 1),
                ("enabled_groups", row[group + "_group_enabled"]),
                ("changed_groups", row[group + "_updated"]),
            ):
                key = prefix + group + "_" + suffix
                self.counts[key] = self.counts.get(key, 0) + int(amount)
        journal_seconds = time.perf_counter() - begin
        summary["journal_seconds"] += journal_seconds
        summary["max_journal_seconds"] = max(summary["max_journal_seconds"], journal_seconds)
        summary["logged_seconds"] += row_seconds + journal_seconds
        summary["max_logged_seconds"] = max(
            summary["max_logged_seconds"], row_seconds + journal_seconds
        )

    def close(self):
        self.stream.close()
        return {
            "path": self.path.name,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.hasher.hexdigest(),
            "max_line_bytes": self.max_line,
            "max_row_tree_bytes": self.max_tree,
            "timing_scope": (
                "original row.seconds plus serialization/write/flush/hash and scalar summary"
            ),
        }


def project_cost(spec, report, launch, *, audit_launch=None):
    """Measured model/journal maxima; final memory admission needs the finished audit."""
    assert report["status"] == "completed" and report["mode"] == "cost"
    assert report["config"] == spec.config and report["duration_clock"] == duration_clock()
    assert report["counts"]["first_attempts"] == 20
    assert report["counts"]["teacher_byte_offers"] == 60
    assert len(report["actions"]) == 100 and len(report["motor_demonstrations"]) == 2

    def positive(value):
        value = float(value)
        assert math.isfinite(value) and value > 0
        return value

    per_arm = {}
    for arm in ARMS:
        summary = report["lesson_summary"][arm]
        assert summary["rows"] == 4
        cap = report["cap_generation"][arm]
        assert cap["stop_reason"] == "limit" and cap["decisions"] == 3
        assert len(bytes.fromhex(cap["generated_hex"])) == 2
        per_arm[arm] = {
            "max_complete_lesson_seconds": positive(summary["max_seconds"]),
            "max_journal_seconds": positive(summary["max_journal_seconds"]),
            "additional_capped_prediction_seconds": positive(cap["seconds"]),
        }
    assert report["lesson_log"]["rows"] == 20
    learning = 2 * sum(1024 * sum(row.values()) for row in per_arm.values())
    measured = sum(
        row["seconds"] + row["journal_seconds"] for row in report["lesson_summary"].values()
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
    run_reserve = spec.config["run_memory_reserve_bytes"]
    audit_reserve = spec.config["audit_memory_reserve_bytes"]
    historical = sum(spec.config["historical_full_checkpoint_bytes"].values())
    run_peak = (
        max(parent_worker, positive(launch["observed_tree_peak_sum_mib"]))
        + (allowance + run_reserve) / 2**20
    )
    audit_peak = None
    if audit_launch is not None:
        assert audit_launch["status"] == "passed" and audit_launch["exit_code"] == 0
        positive(audit_launch["elapsed_seconds"])
        audit_peak = (
            positive(audit_launch["observed_tree_peak_sum_mib"])
            + (allowance + 2 * historical + audit_reserve) / 2**20
        )
    peak = max(run_peak, audit_peak or 0)
    return {
        "per_arm": per_arm,
        "projected_learning_seconds": learning,
        "projected_other_seconds": other,
        "finalization_allowance_seconds": 30,
        "projected_seconds": learning + other + 30,
        "parent_plus_worker_peak_mib": parent_worker,
        "receipt_memory_projection": {
            "streaming_allowance_bytes": allowance,
            "run_engineering_reserve_bytes": run_reserve,
            "audit_engineering_reserve_bytes": audit_reserve,
            "historical_full_checkpoint_bytes": historical,
            "run_projected_peak_mib": run_peak,
            "audit_projected_peak_mib": audit_peak,
        },
        "audit_peak_pending": audit_launch is None,
        "peak_upper_mib": peak,
        "admissible": audit_launch is not None and learning + other + 30 <= 900 and peak <= 2048,
    }


def worker(spec, args):
    import torch

    from benchmarks.native_scaling import _process_memory

    torch.set_num_threads(1)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    requests = json.loads(args.requests.read_bytes())
    assert set(requests) == {"source_sha256", "cases"}
    assert requests["source_sha256"] == sources(spec)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(ARMS)
    rows, all_work = [], {}
    for arm in ARMS:
        entry = saved[arm]
        assert set(entry) == {"weights", "memories"} and len(entry["memories"]) == 12
        model = new_model(spec, arm)
        model.load_state_dict(entry["weights"], strict=True)
        states = {
            key: restore_memory(model, arm, state) for key, state in entry["memories"].items()
        }
        arm_work = []
        for request in requests["cases"]:
            assert set(request) == {"id", "memory_id", "query_hex", "delay_reads"}
            if time.time() > args.deadline_unix:
                raise TimeoutError("worker deadline")
            if _process_memory()["peak_working_set_mib"] > spec.config["parent_mib"]:
                raise MemoryError("worker peak budget")
            state = states[request["memory_id"]]
            before = digest(state.snapshot())
            begin = time.perf_counter()
            distraction = []
            for _ in range(request["delay_reads"]):
                query = b"What is the value of k99?"
                distraction.append(model.query_receipt(query, state).work)
            generated, steps, work = free_attempt(
                spec, arm, model, bytes.fromhex(request["query_hex"]), state
            )
            assert digest(state.snapshot()) == before
            work = add_work(work, *distraction)
            rows.append(
                {
                    "arm": arm,
                    **request,
                    **generated,
                    "seconds": time.perf_counter() - begin,
                    "memory_sha256": before,
                    "steps": steps,
                    "work": work,
                }
            )
            arm_work.append(work)
        assert same_state(entry["weights"], model.state_dict())
        all_work[arm] = add_work(*arm_work)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    assert sources(spec) == requests["source_sha256"]
    write_json(
        args.output,
        {
            "status": "completed",
            "outputs": rows,
            "work": all_work,
            "query_byte_steps": sum(
                row["query_raw_bytes"] + row["query_eos_symbols"] for row in all_work.values()
            ),
            "checkpoint_sha256": args.checkpoint_sha,
            "requests_sha256": args.requests_sha,
            "worker_pid": os.getpid(),
            "memory": _process_memory(),
            "elapsed_seconds": time.perf_counter() - START,
            "duration_clock": duration_clock(),
            "roster_invoked_or_documents_or_targets_supplied_to_model": False,
            "routing_boundary_not_filesystem_sandbox": True,
            "timing_scope": "per-output seconds includes its preceding distractor queries",
        },
    )


def final_actions(spec, args, checks, report, guard):
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
    report["worker"] = {
        "command": command,
        "returncode": child.returncode,
        "package_seconds": time.perf_counter() - begin,
    }
    if child.returncode:
        raise RuntimeError("source-free worker failed")
    generated = json.loads(generated_path.read_bytes())
    assert generated["duration_clock"] == report["duration_clock"]
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

    begin = time.perf_counter()
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
        action = previous.reference.actuate(
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
    report["action_package_seconds"] = time.perf_counter() - begin
    report["scores"] = {
        a: {
            g: sum(r["correct"] for r in report["actions"] if r["arm"] == a and r["group"] == g)
            for g in ("fit", "unseen_value", "delayed", "interference")
        }
        for a in ARMS
    }
    report["qualification"] = args.mode == "run" and qualifies(report["scores"])
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
        "actions": [],
        "action_scope": (
            "Generated answer bytes come from the local neuronal learner; the occurrence motor "
            "uses host-authored boundaries/candidate drives as a narrow action-schema scaffold. "
            "Execution does not establish entirely neuron-only action selection."
        ),
    }
    memory = None
    journal = None
    passed = False

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
            projection = project_cost(spec, cost_report, launch, audit_launch=audit_launch)
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
        models = {arm: new_model(spec, arm) for arm in ARMS}
        initial = {arm: detached_state(model.state_dict()) for arm, model in models.items()}
        for arm in ARMS[:4]:
            tensor_keys = {key for key, value in initial[arm].items() if torch.is_tensor(value)}
            assert tensor_keys == {
                key for key, value in initial["learned"].items() if torch.is_tensor(value)
            }
            for key in tensor_keys:
                assert same_state(initial[arm][key], initial["learned"][key])
        shared = ["input_codes", "codec.weight", "codec.bias"]
        assert all(
            same_state(initial["learned"][key], initial["dense_reference"][key]) for key in shared
        )
        report["common_initial_tensor_keys"] = {
            "dense_reference": shared,
        }
        torch.save(initial, args.output / "initial_weights.pt")
        report["model_tensor_bytes"] = {arm: tensor_bytes(state) for arm, state in initial.items()}
        report["tape_accounting"] = models["learned"].memory.accounting(
            models["learned"].memory.initial_state()
        )
        report["shared_model_accounting"] = models["learned"].accounting()
        journal = LessonJournal(args.output / "lessons.jsonl", spec)
        report["lesson_summary"] = journal.summary
        report["counts"] = journal.counts
        for completed_indices, index in enumerate(indices, 1):
            for arm in ARMS:
                row = lesson(spec, arm, models[arm], schedule[index], guard)
                journal.append({"arm": arm, "index": index, **row})
                del row
                guard()
            if completed_indices % 128 == 0:
                print(
                    f"completed_lessons={journal.rows} "
                    f"elapsed_seconds={time.perf_counter() - START:.3f}",
                    flush=True,
                )
        report["lesson_log"] = journal.close()
        journal = None
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
        if args.mode == "cost":
            report["cap_generation"] = {
                arm: capped_prediction(spec, arm, model) for arm, model in models.items()
            }
        isolated = args.output / "source-free"
        isolated.mkdir()
        export_memories(spec, models, checks, isolated, report, guard)
        del models, initial
        final_actions(spec, args, checks, report, guard)
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
        report["counts"].update(actions=len(report["actions"]), motor_demos=2)
        report["work_total"] = total_work
        report["count_scope"] = (
            "Candidate teacher offers and two held group opportunities are separate from actual "
            "changes; dense reference operations are separate. All fast association writes, "
            "query-only distractions, "
            "branches, codec passes and cost-only cap ingestions are counted. Literal sensory "
            "content is retained in neuronal synapses; no source-file or target-routing claim "
            "implies compression."
        )
        report["status"] = "completed"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        try:
            if journal is not None:
                report["lesson_log"] = journal.close()
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
                "artifacts": {
                    p.relative_to(args.output).as_posix(): file_hash(p)
                    for p in (
                        *args.output.glob("source-free/*"),
                        args.output / "initial_weights.pt",
                        args.output / "offline_teacher_roster.json",
                        args.output / "lessons.jsonl",
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
