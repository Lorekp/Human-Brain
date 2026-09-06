"""Fixed five-arm raw-document sequence retrieval and source-free action return."""

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
from dataclasses import dataclass, is_dataclass
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = time.perf_counter()
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated import local_memory_two_byte_v1 as previous  # noqa: E402

ARMS = ("learned", "address_frozen", "slow_frozen", "writes_disabled", "byte_reference")
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
class SequenceSpec(previous.operations.ExperimentSpec):
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
            "query_size": 32,
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
            "arms": list(ARMS),
            "loss_kind": {
                arm: "half_sum_squared_potential_error"
                if arm == "byte_reference"
                else "sequence_mixture_negative_log_likelihood"
                for arm in ARMS
            },
            "byte_reference_revision": "a1ab5ee0d82c5cc047392ec8e6b22be3a4386438",
            "cap_document_hex": (b"." * 127).hex(),
        }


SPEC = SequenceSpec(
    schema="local-sequence-memory-v1",
    module="experiments.r4_integrated.local_sequence_memory_v1",
    entrypoint=Path(__file__).resolve(),
    recipe=Path(__file__).with_suffix(".md").resolve(),
)


def sources(spec=SPEC):
    paths = (
        "experiments/r4_integrated/local_memory_two_byte_v1.py",
        "experiments/r4_integrated/local_memory_two_byte_v1.md",
        "experiments/r4_integrated/local_memory_event_reference.py",
        "src/humanbrain/neuronal_sequence_memory.py",
        "docs/research/local-sequence-memory-review.md",
    )
    return {**previous.sources(spec), **{path: file_hash(ROOT / path) for path in paths}}


def new_model(spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown sequence experiment arm")
    if arm == "byte_reference":
        from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner

        return LocalMemoryLearner(mode="tagged", seed=spec.config["seed"]).eval()
    from humanbrain.local_memory_learner import LocalMemoryLearner

    return LocalMemoryLearner(mode=arm, seed=spec.config["seed"]).eval()


def plain(value):
    """Serialize already observed receipts; never invoke a model to fill fields."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return {key: plain(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def add_work(*rows):
    keys = set().union(*(row.keys() for row in rows))
    return {key: sum(row.get(key, 0) for row in rows) for key in sorted(keys)}


def ingestion_work(arm, documents):
    raw = sum(len(document) for document in documents)
    ends = len(documents)
    steps = raw + ends
    candidate = arm != "byte_reference"
    enabled = candidate and arm != "writes_disabled"
    return {
        "document_raw_bytes": raw,
        "document_eos_symbols": ends,
        "tape_allocation_steps": steps if candidate else 0,
        "tape_key_normalizations": steps if enabled else 0,
        "tape_lag_shift_steps": raw if candidate else 0,
        "tape_adjacency_assignments": steps if candidate else 0,
        "sensory_association_event_opportunities": steps if enabled else 0,
        "key_association_event_opportunities": steps if enabled else 0,
        "byte_reference_writer_steps": 0 if candidate else steps,
    }


def restore_memory(model, arm, snapshot):
    if arm == "byte_reference":
        return model.memory.restore_local_state(snapshot)
    return model.memory.restore_state(snapshot)


def baseline_query_work(query, *, learning=False):
    return {
        "query_raw_bytes": len(query),
        "query_eos_symbols": 1,
        "query_controller_steps": len(query) + 1,
        "query_eligibility_steps": len(query) + 1 if learning else 0,
        "byte_reference_memory_reads": 1,
    }


def free_attempt(spec, arm, model, query, state):
    if arm == "byte_reference":
        generated = model.predict(query, state, spec.config["max_bytes"])
        work = baseline_query_work(query)
        work["byte_reference_output_steps"] = len(generated.data) + 1
        return generation_record(generated), [], work
    receipt = model.predict_with_receipt(query, state, spec.config["max_bytes"])
    return generation_record(receipt.generation), plain(receipt.steps), receipt.work


def qualifies(scores):
    learned = scores["learned"]
    return learned == {"fit": 8, "unseen_value": 4, "delayed": 4, "interference": 4} and all(
        learned["unseen_value"] > scores[arm]["unseen_value"] for arm in ARMS[1:]
    )


def lesson(spec, arm, model, episode, guard):
    guard()
    begin = time.perf_counter()
    documents = [bytes.fromhex(raw) for raw in episode["document_hex"]]
    query, target = bytes.fromhex(episode["query_hex"]), bytes.fromhex(episode["target_hex"])
    if arm == "byte_reference":
        row = previous.lesson(spec, model, episode, guard)
        query_work = add_work(baseline_query_work(query), baseline_query_work(query, learning=True))
        query_work.update(
            byte_reference_output_steps=row["decisions"] + row["teacher_count"],
            teacher_offers=row["teacher_count"],
        )
        row["work"] = add_work(ingestion_work(arm, documents), query_work)
    else:
        before = digest(model.state_dict())
        state = model.ingest(documents)
        memory_before = digest(state.snapshot())
        first, first_steps, first_work = free_attempt(spec, arm, model, query, state)
        assert digest(model.state_dict()) == before and digest(state.snapshot()) == memory_before
        feedback = model.teach(query, target, state, spec.config["learning_rate"])
        assert feedback.teacher_count == 3
        assert digest(state.snapshot()) == digest(feedback.state.snapshot()) == memory_before
        after = digest(model.state_dict())
        row = {
            **first,
            "first_exact": first["stop_reason"] == "eos" and first["generated_hex"] == target.hex(),
            "teacher_count": feedback.teacher_count,
            "mean_loss": feedback.mean_loss,
            "total_loss": feedback.total_loss,
            "loss_kind": "sequence_mixture_negative_log_likelihood",
            "codec_updated": feedback.codec_updated,
            "address_updated": feedback.address_updated,
            "query_updated": feedback.query_updated,
            "codec_group_enabled": arm != "slow_frozen",
            "address_group_enabled": arm in {"learned", "writes_disabled"},
            "query_group_enabled": arm in {"learned", "writes_disabled"},
            "before_sha256": before,
            "after_sha256": after,
            "memory_sha256": memory_before,
            "retired_memory_sha256": memory_before,
            "retained_state_tensor_bytes": tensor_bytes(state.snapshot()),
            "first_steps": first_steps,
            "teacher_steps": plain(feedback.steps),
            "teacher_address_error": plain(feedback.address_error),
            "teacher_query_signal": plain(feedback.query_signal),
            "work": add_work(ingestion_work(arm, documents), first_work, feedback.work),
        }
        if arm == "slow_frozen":
            assert after == before and not any(
                (feedback.codec_updated, feedback.address_updated, feedback.query_updated)
            )
    row["seconds"] = time.perf_counter() - begin
    return row


def capped_prediction(spec, arm, model):
    """One fully occupied tape cap; ordinary output forcing is exactly restored."""
    import torch

    before = detached_state(model.state_dict())
    document = bytes.fromhex(spec.config["cap_document_hex"])
    start = time.perf_counter()
    state = model.ingest([document])
    ingestion_seconds = time.perf_counter() - start
    if arm != "byte_reference":
        assert state.used == 128
    with torch.no_grad():
        if arm == "byte_reference":
            model.decoder.output_weights.zero_()
            model.decoder.output_weights[0, :3] = 1
        else:
            model.codec.weight.zero_()
            model.codec.bias.zero_()
            model.codec.bias[0] = 1
        begin = time.perf_counter()
        try:
            generated, steps, work = free_attempt(
                spec, arm, model, b"What is the value of k00?", state
            )
            seconds = time.perf_counter() - begin
        finally:
            model.load_state_dict(before, strict=True)
    assert same_state(before, model.state_dict())
    assert generated["stop_reason"] == "limit" and generated["decisions"] == 3
    return {
        **generated,
        "seconds": seconds,
        "steps": steps,
        "work": work,
        "ingestion_seconds": ingestion_seconds,
        "ingestion_work": ingestion_work(arm, [document]),
        "occupied_slots": 128 if arm != "byte_reference" else None,
        "original_weights_restored": True,
    }


def project_cost(spec, report, launch):
    """Original complete-cost margins, freshly bound to these five arms."""
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
        rows = [row for row in report["lessons"] if row["arm"] == arm]
        assert [row["index"] for row in rows] == report["selected_lesson_indices"]
        assert len(rows) == 4
        cap = report["cap_generation"][arm]
        assert cap["stop_reason"] == "limit" and cap["decisions"] == 3
        assert len(bytes.fromhex(cap["generated_hex"])) == 2
        per_arm[arm] = {
            "max_complete_lesson_seconds": max(positive(row["seconds"]) for row in rows),
            "additional_capped_prediction_seconds": positive(cap["seconds"]),
        }
    learning = 2 * sum(1024 * sum(row.values()) for row in per_arm.values())
    other = 2 * positive(
        positive(launch["elapsed_seconds"]) - sum(row["seconds"] for row in report["lessons"])
    )
    parent_worker = positive(report["memory"]["peak_working_set_mib"]) + positive(
        report["worker"]["memory"]["peak_working_set_mib"]
    )
    peak = max(parent_worker, positive(launch["observed_tree_peak_sum_mib"]))
    return {
        "per_arm": per_arm,
        "projected_learning_seconds": learning,
        "projected_other_seconds": other,
        "finalization_allowance_seconds": 30,
        "projected_seconds": learning + other + 30,
        "parent_plus_worker_peak_mib": parent_worker,
        "peak_upper_mib": peak,
        "admissible": learning + other + 30 <= 900 and peak <= 2048,
    }


def export_memories(spec, models, checks, isolated, report, guard):
    import torch

    saved, work = {}, {}
    begin = time.perf_counter()
    for arm, model in models.items():
        before = digest(model.state_dict())
        memories, arm_work = {}, []
        for case in checks:
            if case["memory_id"] in memories:
                continue
            guard()
            base = case["base_memory_id"]
            documents = [
                bytes.fromhex(raw)
                for raw in (case["additional_document_hex"] if base else case["document_hex"])
            ]
            state = model.ingest(
                documents, restore_memory(model, arm, memories[base]) if base else None
            )
            memories[case["memory_id"]] = detached_state(state.snapshot())
            arm_work.append(ingestion_work(arm, documents))
        assert len(memories) == 12 and digest(model.state_dict()) == before
        saved[arm] = {"weights": detached_state(model.state_dict()), "memories": memories}
        work[arm] = add_work(*arm_work)
    report["final_ingestion_seconds"] = time.perf_counter() - begin
    report["final_ingestion_work"] = work
    report["source_free_tensor_bytes"] = tensor_bytes(saved)
    begin = time.perf_counter()
    torch.save(saved, isolated / "source_free.pt")
    report["source_free_serialization_seconds"] = time.perf_counter() - begin
    write_json(
        isolated / "query_only.json",
        {
            "source_sha256": report["source_sha256"],
            "cases": [
                {key: case[key] for key in ("id", "memory_id", "query_hex", "delay_reads")}
                for case in checks
            ],
        },
    )


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
                if arm == "byte_reference":
                    model._query(query, state)
                    distraction.append(baseline_query_work(query, learning=True))
                else:
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
            "query_byte_steps": sum(row["query_controller_steps"] for row in all_work.values()),
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
    report["learned_retrieval_advantage"] = args.mode == "run" and all(
        report["scores"]["learned"]["unseen_value"] > report["scores"][a]["unseen_value"]
        for a in ARMS[1:]
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
            projection = project_cost(spec, cost_report, launch)
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
        shared = ["input_codes", "query_cell.weight", "query_cell.bias", "query_cell.recurrent"]
        assert all(
            same_state(initial["learned"][key], initial["byte_reference"][key]) for key in shared
        )
        report["common_initial_tensor_keys"] = shared
        torch.save(initial, args.output / "initial_weights.pt")
        report["model_tensor_bytes"] = {arm: tensor_bytes(state) for arm, state in initial.items()}
        report["tape_accounting"] = models["learned"].memory.accounting(
            models["learned"].memory.initial_state()
        )
        report["sequence_model_accounting"] = models["learned"].accounting()
        for completed_indices, index in enumerate(indices, 1):
            for arm in ARMS:
                row = lesson(spec, arm, models[arm], schedule[index], guard)
                report["lessons"].append({"arm": arm, "index": index, **row})
            if completed_indices % 128 == 0:
                print(
                    f"completed_lessons={len(report['lessons'])} "
                    f"elapsed_seconds={time.perf_counter() - START:.3f}",
                    flush=True,
                )
        assert same_state(initial["slow_frozen"], models["slow_frozen"].state_dict())
        address_fixed = [
            "address_weight",
            "query_cell.weight",
            "query_cell.bias",
            "query_cell.recurrent",
        ]
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
                "acquisition": add_work(
                    *(row["work"] for row in report["lessons"] if row["arm"] == arm)
                ),
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
        report["counts"] = {
            "first_attempts": len(report["lessons"]),
            "first_generation_decisions": sum(row["decisions"] for row in report["lessons"]),
            "teacher_byte_offers": sum(row["teacher_count"] for row in report["lessons"]),
            "enabled_teacher_offers": sum(
                row["teacher_count"] for row in report["lessons"] if row["arm"] != "slow_frozen"
            ),
            "actions": len(report["actions"]),
            "motor_demos": 2,
        }
        for group in ("codec", "address", "query"):
            rows = [row for row in report["lessons"] if row["arm"] != "byte_reference"]
            report["counts"][group + "_group_offers"] = len(rows)
            report["counts"][group + "_enabled_groups"] = sum(
                row[group + "_group_enabled"] for row in rows
            )
            report["counts"][group + "_changed_groups"] = sum(
                row[group + "_updated"] for row in rows
            )
        baseline = [row for row in report["lessons"] if row["arm"] == "byte_reference"]
        report["counts"]["byte_reference_changed_decoder_steps"] = sum(
            row["decoder_changed_steps"] for row in baseline
        )
        report["counts"]["byte_reference_query_updated_lessons"] = sum(
            row["query_updated"] for row in baseline
        )
        report["counts"]["byte_reference_memory_updated_lessons"] = sum(
            row["memory_updated"] for row in baseline
        )
        report["work_total"] = total_work
        report["count_scope"] = (
            "Candidate teacher offers and three held group opportunities are separate from actual "
            "changes; baseline LMS is bytewise. All fast association writes, "
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
