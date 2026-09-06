"""Fixed two-byte return: local document acquisition and source-free Python actions."""

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

from experiments.r4_integrated import local_memory_byte_harness as operations  # noqa: E402

reference = operations.reference
ARMS = operations.ARMS
file_hash, write_json = operations.file_hash, operations.write_json
digest, generation_record = operations.digest, operations.generation_record
detached_state, same_state, tensor_bytes = (
    operations.detached_state,
    operations.same_state,
    operations.tensor_bytes,
)
duration_clock = operations.duration_clock
validate_cost_launch = operations.validate_cost_launch
final_actions = operations.final_actions


@dataclass(frozen=True)
class TwoByteSpec(operations.ExperimentSpec):
    """Only this successor's immutable identity and fixed workload."""

    @property
    def config(self):
        return {
            "schema": self.schema,
            "seed": 733,
            "lessons": 1024,
            "learning_rate": 0.01,
            "input_code_size": 32,
            "query_size": 32,
            "hidden_size": 64,
            "key_size": 64,
            "value_size": 64,
            "output": "phase_context_lms",
            "loss_kind": "half_sum_squared_potential_error",
            "max_bytes": 2,
            "seconds": 900,
            "parent_mib": 2048,
            "cost_lessons_per_arm": 4,
            "cost_sample": (
                "descending document bytes, teacher bytes, query bytes; original index tie"
            ),
            "query_traversals_per_lesson": 2,
            "retained_memories_per_arm": 12,
            "arms": list(ARMS),
        }


SPEC = TwoByteSpec(
    schema="local-memory-two-byte-v1",
    module="experiments.r4_integrated.local_memory_two_byte_v1",
    entrypoint=Path(__file__).resolve(),
    recipe=Path(__file__).with_suffix(".md").resolve(),
)


def sources(spec=SPEC):
    paths = (
        ROOT / "src/humanbrain/phase_context_lms.py",
        ROOT / "docs/research/local-phase-lms-development.md",
        ROOT / "docs/research/local-phase-memory-integration.md",
    )
    return {
        **operations.sources(spec),
        **{p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths},
    }


def new_model(spec, arm):
    from humanbrain.local_memory_learner import LocalMemoryLearner

    return LocalMemoryLearner(
        mode=arm,
        seed=spec.config["seed"],
        input_code_size=spec.config["input_code_size"],
        hidden_size=spec.config["hidden_size"],
        key_size=spec.config["key_size"],
        value_size=spec.config["value_size"],
    ).eval()


def roster():
    """Offline authoring only; the worker never calls this function."""
    values = ("aa", "ab", "bb", "bc", "cc", "cd", "dd", "da")
    train = [{"key": f"k{i:02d}", "value": value} for i, value in enumerate(values)]
    novel = ((0, "ac", "ca"), (0, "ca", "ac"), (2, "bd", "db"), (2, "db", "bd"))
    checks = []
    for group, count in (("fit", 8), ("unseen_value", 4), ("delayed", 4), ("interference", 4)):
        for i in range(count):
            if group == "fit":
                pair = i // 2
                keys = [2 * pair, 2 * pair + 1]
                if pair % 2:
                    keys.reverse()
                assignments = [(key, values[key]) for key in keys]
                query_key, value, base_id = i, values[i], f"fit-{pair}"
            else:
                key, value, other = novel[i]
                assignments = [(key, value), (key + 1, other)]
                query_key, base_id = key, f"novel-{i}"
            documents = [
                f"Record k{key:02d} has value {value}.".encode() for key, value in assignments
            ]
            additional = (
                [f"Record k{key:02d} has value {values[key]}.".encode() for key in (4, 5, 6)]
                if group == "interference"
                else []
            )
            checks.append(
                {
                    "id": len(checks),
                    "group": group,
                    "memory_id": f"interference-{i}" if additional else base_id,
                    "base_memory_id": base_id if additional else None,
                    "additional_document_hex": [d.hex() for d in additional],
                    "document_hex": [d.hex() for d in documents + additional],
                    "query_hex": f"What is the value of k{query_key:02d}?".encode().hex(),
                    "expected_hex": value.encode().hex(),
                    "delay_reads": 32 if group == "delayed" else 0,
                }
            )
    return train, checks


def teaching_schedule(train):
    # The frozen authoring helper supplies the same private RNG and 256x4 episode
    # algorithm; only our explicit TRAIN roster changes. It never enters inference.
    assert reference.CONFIG["seed"] == 733
    assert reference.CONFIG["steps"] == 256 and reference.CONFIG["batch_size"] == 4
    return reference.teaching_schedule(train)


def lesson(spec, model, episode, guard):
    """Same ingestion/first/teacher boundary with actual LMS exposure receipts."""
    guard()
    start = time.perf_counter()
    documents = [bytes.fromhex(x) for x in episode["document_hex"]]
    query, target = bytes.fromhex(episode["query_hex"]), bytes.fromhex(episode["target_hex"])
    before = digest(model.state_dict())
    state = model.ingest(documents)
    del documents
    state_before = digest(state.snapshot())
    retained_bytes = tensor_bytes(state.snapshot())
    prediction = model.predict(query, state, spec.config["max_bytes"])
    assert digest(model.state_dict()) == before and digest(state.snapshot()) == state_before
    feedback = model.teach(query, target, state, spec.config["learning_rate"])
    assert feedback.teacher_count == len(target) + 1 == 3
    assert same_state(state.synapses, feedback.state.synapses)
    after = digest(model.state_dict())
    if model.mode == "frozen":
        assert after == before and not any(
            (feedback.decoder_updated, feedback.query_updated, feedback.memory_updated)
        )
    if model.mode == "writes_disabled":
        assert not state.synapses.count_nonzero()
    return {
        **generation_record(prediction),
        "first_exact": prediction.stop_reason == "eos" and prediction.data == target,
        "teacher_count": feedback.teacher_count,
        "mean_loss": feedback.mean_loss,
        "loss_kind": feedback.loss_kind,
        "decoder_changed_in_lesson": feedback.decoder_updated,
        "decoder_changed_steps": feedback.decoder_changed_steps,
        "nonzero_residual_offers": feedback.nonzero_residual_offers,
        "enabled_nonzero_residual_offers": feedback.enabled_nonzero_residual_offers,
        "query_updated": feedback.query_updated,
        "memory_updated": feedback.memory_updated,
        "before_sha256": before,
        "after_sha256": after,
        "memory_sha256": state_before,
        "retired_memory_sha256": digest(feedback.state.snapshot()),
        "retained_state_tensor_bytes": retained_bytes,
        "context_signal_sha256": digest(feedback.mean_context_signal),
        "seconds": time.perf_counter() - start,
    }


def capped_prediction(spec, arm, model):
    import torch

    weights = detached_state(model.state_dict())
    state = model.memory.initial_local_state(
        1, credit_mode="aggregate" if arm == "aggregate" else "tagged"
    )
    with torch.no_grad():
        model.decoder.output_weights.zero_()
    begin = time.perf_counter()
    generated = model.predict(b"What is the value of k00?", state, spec.config["max_bytes"])
    record = {"seconds": time.perf_counter() - begin, **generation_record(generated)}
    model.load_state_dict(weights)
    assert same_state(weights, model.state_dict())
    assert generated.stop_reason == "limit" and len(generated.data) == 2
    return record


def export_memories(models, checks, isolated, report, guard):
    import torch

    saved = {}
    begin = time.perf_counter()
    for arm, model in models.items():
        memories, steps = {}, 0
        before = digest(model.state_dict())
        for case in checks:
            if case["memory_id"] in memories:
                continue
            guard()
            base = case["base_memory_id"]
            documents = [
                bytes.fromhex(x)
                for x in (case["additional_document_hex"] if base else case["document_hex"])
            ]
            state = model.ingest(
                documents, model.memory.restore_local_state(memories[base]) if base else None
            )
            steps += sum(len(x) + 1 for x in documents)
            memories[case["memory_id"]] = detached_state(state.snapshot())
        assert len(memories) == 12 and digest(model.state_dict()) == before
        saved[arm] = {"weights": detached_state(model.state_dict()), "memories": memories}
        report.setdefault("final_ingestion_byte_steps", {})[arm] = steps
    report["final_ingestion_seconds"] = time.perf_counter() - begin
    report["source_free_tensor_bytes"] = tensor_bytes(saved)
    begin = time.perf_counter()
    torch.save(saved, isolated / "source_free.pt")
    report["source_free_serialization_seconds"] = time.perf_counter() - begin
    write_json(
        isolated / "query_only.json",
        {
            "source_sha256": report["source_sha256"],
            "cases": [
                {key: c[key] for key in ("id", "memory_id", "query_hex", "delay_reads")}
                for c in checks
            ],
        },
    )


def project_cost(spec, report, launch):
    """Recompute the original conservative formula; never accept a supplied estimate."""
    assert report["status"] == "completed" and report["mode"] == "cost"
    assert report["config"] == spec.config
    assert report["duration_clock"] == duration_clock()
    assert len(report["lessons"]) == len(ARMS) * spec.config["cost_lessons_per_arm"]
    assert len(report["actions"]) == 100 and len(report["motor_demonstrations"]) == 2
    assert report["counts"]["first_attempts"] == 20
    assert report["counts"]["teacher_byte_offers"] == 60
    assert report["counts"]["enabled_decoder_step_opportunities"] == 48

    def positive(value):
        assert math.isfinite(value) and value > 0
        return value

    per_arm = {}
    for arm in ARMS:
        rows = [row for row in report["lessons"] if row["arm"] == arm]
        assert len(rows) == spec.config["cost_lessons_per_arm"]
        assert [row["index"] for row in rows] == report["selected_lesson_indices"]
        capped = report["cap_generation"][arm]
        assert capped["stop_reason"] == "limit" and capped["decisions"] == 3
        assert len(bytes.fromhex(capped["generated_hex"])) == spec.config["max_bytes"]
        per_arm[arm] = {
            "max_complete_lesson_seconds": max(positive(row["seconds"]) for row in rows),
            "additional_capped_prediction_seconds": positive(capped["seconds"]),
        }
    learning = 2 * sum(spec.config["lessons"] * sum(row.values()) for row in per_arm.values())
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
        "admissible": projected <= spec.config["seconds"] and peak <= spec.config["parent_mib"],
    }


def worker(spec, args):
    """Query-only fresh process; source authoring and teaching are not invoked."""
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
    rows, query_steps = [], 0
    for arm in ARMS:
        entry = saved[arm]
        assert set(entry) == {"weights", "memories"}
        assert len(entry["memories"]) == spec.config["retained_memories_per_arm"]
        model = new_model(spec, arm)
        model.load_state_dict(entry["weights"], strict=True)
        states = {
            key: model.memory.restore_local_state(value) for key, value in entry["memories"].items()
        }
        for request in requests["cases"]:
            assert set(request) == {"id", "memory_id", "query_hex", "delay_reads"}
            if time.time() > args.deadline_unix:
                raise TimeoutError("worker deadline")
            if _process_memory()["peak_working_set_mib"] > spec.config["parent_mib"]:
                raise MemoryError("worker peak budget")
            state = states[request["memory_id"]]
            before = digest(state.snapshot())
            with torch.no_grad():
                for _ in range(request["delay_reads"]):
                    model._query(b"What is the value of k99?", state)
                    query_steps += 26
                query = bytes.fromhex(request["query_hex"])
                begin = time.perf_counter()
                generated = model.predict(query, state, spec.config["max_bytes"])
                query_steps += len(query) + 1
            assert digest(state.snapshot()) == before
            rows.append(
                {
                    "arm": arm,
                    **request,
                    **generation_record(generated),
                    "seconds": time.perf_counter() - begin,
                    "memory_sha256": before,
                }
            )
        assert same_state(entry["weights"], model.state_dict())
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    assert sources(spec) == requests["source_sha256"]
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
            "elapsed_seconds": time.perf_counter() - START,
            "duration_clock": duration_clock(),
            "roster_invoked_or_documents_or_targets_supplied_to_model": False,
            "routing_boundary_not_filesystem_sandbox": True,
        },
    )


def run(spec, args):
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
        models = {a: new_model(spec, a) for a in ARMS}
        initial = {a: detached_state(m.state_dict()) for a, m in models.items()}

        def tensor_only(state):
            return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}

        assert all(
            same_state(tensor_only(initial["tagged"]), tensor_only(initial[a])) for a in ARMS
        )
        torch.save(initial, args.output / "initial_weights.pt")
        report["model_tensor_bytes"] = {a: tensor_bytes(s) for a, s in initial.items()}
        report["output_accounting"] = models["tagged"].decoder.accounting()
        report["input_code_tensor_bytes"] = tensor_bytes(models["tagged"].input_codes)
        for completed_indices, index in enumerate(indices, 1):
            for arm in ARMS:
                record = lesson(spec, models[arm], schedule[index], guard)
                report["lessons"].append({"arm": arm, "index": index, **record})
            if completed_indices % 128 == 0:
                print(
                    f"completed_lessons={len(report['lessons'])} "
                    f"elapsed_seconds={time.perf_counter() - START:.3f}",
                    flush=True,
                )
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
                report["cap_generation"][arm] = capped_prediction(spec, arm, model)
        isolated = args.output / "source-free"
        isolated.mkdir()
        export_memories(models, checks, isolated, report, guard)
        del models, initial
        final_actions(spec, args, checks, report, guard)
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
        report["counts"]["residual_offers"] = 257 * report["counts"]["teacher_byte_offers"]
        report["counts"]["enabled_residual_offers"] = (
            257 * report["counts"]["enabled_decoder_step_opportunities"]
        )
        for field in (
            "decoder_changed_steps",
            "nonzero_residual_offers",
            "enabled_nonzero_residual_offers",
        ):
            report["counts"][field] = sum(row[field] for row in report["lessons"])
        report["count_scope"] = (
            "Decoder changed flags are lesson-level ORs; "
            "decoder_changed_steps counts actual byte updates. "
            "Teacher bytes and all-257 residual opportunities include frozen offers. "
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
    if arguments.mode == "worker":
        worker(spec, arguments)
    else:
        raise SystemExit(0 if run(spec, arguments) else 1)


if __name__ == "__main__":
    main()
