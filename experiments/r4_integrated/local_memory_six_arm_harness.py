"""Explicit six-arm orchestration for fixed local-credit comparisons.

Closed predecessors remain unchanged. The caller binds its only spec, factory
and complete source manifest; no user-defined hooks are loaded by the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from experiments.r4_integrated import local_memory_query_credit_v1 as previous

ROOT = Path(__file__).resolve().parents[2]
START = time.perf_counter()
ARMS = previous.ARMS
file_hash, write_json = previous.file_hash, previous.write_json
digest, generation_record = previous.digest, previous.generation_record
detached_state, same_state, tensor_bytes = (
    previous.detached_state, previous.same_state, previous.tensor_bytes,
)
duration_clock, validate_cost_launch = previous.duration_clock, previous.validate_cost_launch
roster, teaching_schedule = previous.roster, previous.teaching_schedule
lesson, capped_prediction, export_memories = (
    previous.lesson, previous.capped_prediction, previous.export_memories,
)
project_cost, final_actions = previous.project_cost, previous.final_actions


@dataclass(frozen=True)
class Experiment:
    """One fixed source-bound comparison, including its worker model factory."""

    spec: previous.QueryCreditSpec
    new_model: Callable
    sources: Callable


def validate_initial_models(spec, initial):
    """Match all common arrays and only the three actual fixed encoder tensors."""
    import torch

    assert set(initial) == set(ARMS)
    tensors = {
        arm: {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
        for arm, state in initial.items()
    }
    extra = set(spec.config["candidate_only_initial_tensor_keys"])
    assert extra == {"event_encoder.weight", "event_encoder.bias", "event_encoder.recurrent"}
    reference = tensors["current_tagged"]
    assert not extra.intersection(reference)
    rate_key = "memory.controller.update_fraction"
    rates = torch.tensor([0.5] * 32 + [0.125] * 16 + [0.03125] * 16)
    for arm in ARMS:
        current = tensors[arm]
        assert set(current) == set(reference) | (extra if arm != "current_tagged" else set())
        for key, value in reference.items():
            assert current[key].dtype == value.dtype and same_state(current[key], value)
        assert spec.config["writer_update_fraction"][arm] == rates.tolist()
        assert current[rate_key].dtype == torch.float32 and same_state(current[rate_key], rates)
        if arm != "current_tagged":
            for key in extra:
                assert current[key].dtype == torch.float32
                assert same_state(current[key], tensors["tagged"][key])
            assert sum(tensor_bytes(current[key]) for key in extra) == 8320


def input_work(spec, arm, raw, repeats=1):
    """Count a declared input without rerunning its neuronal encoder.

    The extra host segmentation occurs once here; actual model repetitions are
    separately multiplied. No raw/event cache is retained by the learner.
    """
    from humanbrain.byte_events import segment_bytes

    assert repeats >= 1
    eventized = spec.config["input_encoding"][arm] == "fixed_rate_cell_byte_events"
    raw_events = len(segment_bytes(raw)) if eventized else len(raw)
    return {
        "raw_bytes": repeats * len(raw), "string_eos": repeats,
        "encoder_steps": repeats * (len(raw) + 1) if eventized else 0,
        "controller_events": repeats * (raw_events + 1),
        "encoding_host_segmentations": repeats if eventized else 0,
        "encoding_host_bytes": repeats * len(raw) if eventized else 0,
        "accounting_host_segmentations": int(eventized),
        "accounting_host_bytes": len(raw) if eventized else 0,
    }


def sum_input_work(rows):
    rows = list(rows)
    assert rows
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def event_lesson(spec, arm, model, episode, guard):
    begin = time.perf_counter()
    record = lesson(spec, model, episode, guard)
    record["input_work"] = {
        "document": sum_input_work(input_work(spec, arm, bytes.fromhex(raw))
                                   for raw in episode["document_hex"]),
        "query": input_work(spec, arm, bytes.fromhex(episode["query_hex"]), repeats=2),
    }
    record["seconds"] = time.perf_counter() - begin
    return record


def event_export(spec, models, checks, isolated, report, guard):
    export_memories(models, checks, isolated, report, guard)
    begin = time.perf_counter()
    report["final_ingestion_input_work"] = {}
    for arm in models:
        seen, work = set(), []
        for case in checks:
            if case["memory_id"] in seen:
                continue
            seen.add(case["memory_id"])
            field = "additional_document_hex" if case["base_memory_id"] else "document_hex"
            work.extend(input_work(spec, arm, bytes.fromhex(raw)) for raw in case[field])
        report["final_ingestion_input_work"][arm] = sum_input_work(work)
    report["final_ingestion_seconds"] += time.perf_counter() - begin


def worker(experiment: Experiment, args):
    """Query-only fresh process; source authoring and teaching are not invoked."""
    spec = experiment.spec
    assert spec.config["arms"] == list(ARMS)
    import torch

    from benchmarks.native_scaling import _process_memory

    torch.set_num_threads(1)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    requests = json.loads(args.requests.read_bytes())
    assert set(requests) == {"source_sha256", "cases"}
    assert requests["source_sha256"] == experiment.sources(spec)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(ARMS)
    rows, query_steps, work_by_arm = [], 0, {}
    for arm in ARMS:
        entry = saved[arm]
        assert set(entry) == {"weights", "memories"}
        assert len(entry["memories"]) == spec.config["retained_memories_per_arm"]
        model = experiment.new_model(spec, arm)
        model.load_state_dict(entry["weights"], strict=True)
        states = {
            key: model.memory.restore_local_state(value) for key, value in entry["memories"].items()
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
            with torch.no_grad():
                for _ in range(request["delay_reads"]):
                    model._query(b"What is the value of k99?", state)
                    query_steps += 26
                query = bytes.fromhex(request["query_hex"])
                begin = time.perf_counter()
                generated = model.predict(query, state, spec.config["max_bytes"])
                query_steps += len(query) + 1
            work = [input_work(spec, arm, query)]
            if request["delay_reads"]:
                work.append(input_work(spec, arm, b"What is the value of k99?",
                                       repeats=request["delay_reads"]))
            request_work = sum_input_work(work)
            arm_work.append(request_work)
            assert digest(state.snapshot()) == before
            rows.append(
                {
                    "arm": arm,
                    **request,
                    **generation_record(generated),
                    "seconds": time.perf_counter() - begin,
                    "memory_sha256": before,
                    "input_work": request_work,
                }
            )
        assert same_state(entry["weights"], model.state_dict())
        work_by_arm[arm] = sum_input_work(arm_work)
    assert file_hash(args.checkpoint) == args.checkpoint_sha
    assert file_hash(args.requests) == args.requests_sha
    assert experiment.sources(spec) == requests["source_sha256"]
    write_json(
        args.output,
        {
            "status": "completed",
            "outputs": rows,
            "query_byte_steps": query_steps,
            "input_work": work_by_arm,
            "query_byte_steps_scope": "raw query bytes plus EOS, not controller events",
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


def run(experiment: Experiment, args):
    spec = experiment.spec
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
        report["source_sha256"] = experiment.sources(spec)
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
        models = {a: experiment.new_model(spec, a) for a in ARMS}
        initial = {a: detached_state(m.state_dict()) for a, m in models.items()}

        def tensor_only(state):
            return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}

        validate_initial_models(spec, initial)
        torch.save(initial, args.output / "initial_weights.pt")
        report["model_tensor_bytes"] = {a: tensor_bytes(s) for a, s in initial.items()}
        report["output_accounting"] = models["tagged"].decoder.accounting()
        report["input_code_tensor_bytes"] = tensor_bytes(models["tagged"].input_codes)
        for completed_indices, index in enumerate(indices, 1):
            for arm in ARMS:
                record = event_lesson(spec, arm, models[arm], schedule[index], guard)
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
        encoder_keys = spec.config["candidate_only_initial_tensor_keys"]
        assert all(same_state(initial[arm][key], models[arm].state_dict()[key])
                   for arm in ARMS[:-1] for key in encoder_keys)
        report["frozen_invariants"]["event_encoder_fixed_keys"] = encoder_keys
        if args.mode == "cost":
            report["cap_generation"] = {}
            for arm, model in models.items():
                guard()
                report["cap_generation"][arm] = capped_prediction(spec, arm, model)
                begin = time.perf_counter()
                report["cap_generation"][arm]["input_work"] = input_work(
                    spec, arm, b"What is the value of k00?")
                report["cap_generation"][arm]["seconds"] += time.perf_counter() - begin
        isolated = args.output / "source-free"
        isolated.mkdir()
        event_export(spec, models, checks, isolated, report, guard)
        initial_reference = set(tensor_only(initial["current_tagged"]))
        initial_repaired = set(tensor_only(initial["tagged"]))
        del models, initial
        final_actions(spec, args, checks, report, guard)
        generated = json.loads((isolated / "generated.json").read_bytes())
        report["worker"]["input_work"] = generated["input_work"]
        report["input_work"] = {
            arm: {
                "acquisition_document": sum_input_work(
                    row["input_work"]["document"] for row in report["lessons"]
                    if row["arm"] == arm),
                "acquisition_query": sum_input_work(
                    row["input_work"]["query"] for row in report["lessons"] if row["arm"] == arm),
                "final_document": report["final_ingestion_input_work"][arm],
                "final_query": generated["input_work"][arm],
                **({"cost_cap_query": report["cap_generation"][arm]["input_work"]}
                   if args.mode == "cost" else {}),
            } for arm in ARMS}
        report["counts"] = {
            "first_attempts": len(report["lessons"]),
            "teacher_byte_offers": sum(x["teacher_count"] for x in report["lessons"]),
            "document_byte_steps": len(ARMS)
            * sum(
                sum(len(bytes.fromhex(d)) + 1 for d in schedule[i]["document_hex"]) for i in indices
            ),
            "acquisition_query_byte_steps": 2 * len(ARMS)
            * sum(len(bytes.fromhex(schedule[i]["query_hex"])) + 1 for i in indices),
            "final_query_byte_steps": 23088,
            "actions": len(report["actions"]),
            "motor_demos": 2,
        }
        all_work = [work for stages in report["input_work"].values() for work in stages.values()]
        total_work = sum_input_work(all_work)
        for key in ("raw_bytes", "string_eos", "encoder_steps", "encoding_host_segmentations",
                    "encoding_host_bytes", "accounting_host_segmentations",
                    "accounting_host_bytes"):
            report["counts"]["input_" + key] = total_work[key]
        for kind in ("document", "query"):
            report["counts"]["writer_controller_events" if kind == "document"
                             else "query_controller_events"] = sum(
                work["controller_events"] for stages in report["input_work"].values()
                for stage, work in stages.items() if kind in stage)
        for counter, excluded in (
            ("memory_write_event_opportunities", {"writes_disabled"}),
            ("writer_controller_eligibility_events", {"writer_maps_frozen", "frozen"}),
            ("writer_credit_transport_event_opportunities",
             {"writer_maps_frozen", "frozen", "writes_disabled"}),
        ):
            report["counts"][counter] = sum(
                work["controller_events"] for arm, stages in report["input_work"].items()
                for stage, work in stages.items() if "document" in stage and arm not in excluded)
        for arm, stages in report["input_work"].items():
            final_doc = stages["final_document"]
            assert final_doc["raw_bytes"] + final_doc["string_eos"] == (
                report["final_ingestion_byte_steps"][arm])
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
        for field, rule_map, rule in (
            ("actual_read_credit_lessons", "query_credit", "actual_instantaneous_read_path"),
            ("current_fixed_projection_lessons", "query_credit", "current_fixed_projection"),
            ("direct_value_credit_lessons", "value_credit", "direct_read_neuron_error"),
            ("random_value_credit_lessons", "value_credit", "fixed_random_context_projection"),
            ("multiscale_writer_lessons", "writer_timescale", "fixed_multiscale_32_16_16"),
            ("scalar_writer_lessons", "writer_timescale", "scalar_half"),
        ):
            report["counts"][field] = len(indices) * sum(
                value == rule for value in spec.config[rule_map].values()
            )
        report["query_credit_scope"] = {
            "rules": spec.config["query_credit"],
            "includes_frozen_signal_computation": True,
            "reference_only_initial_tensor_keys": sorted(
                set(initial_reference) - set(initial_repaired)
            ),
        }
        report["value_credit_scope"] = {
            "rules": spec.config["value_credit"],
            "includes_frozen_signal_computation": True,
            "reference_only_initial_tensor_keys": sorted(initial_reference - initial_repaired),
        }
        report["writer_timescale_scope"] = {
            "rules": spec.config["writer_timescale"],
            "update_fractions": spec.config["writer_update_fraction"],
            "includes_frozen_forward_computation": True,
            "candidate_only_initial_tensor_keys": [],
            "reference_only_initial_tensor_keys": sorted(initial_reference - initial_repaired),
        }
        report["input_encoding_scope"] = {
            "rules": spec.config["input_encoding"], "encoder": spec.config["event_encoder"],
            "candidate_only_initial_tensor_keys": sorted(initial_repaired - initial_reference),
            "reference_only_initial_tensor_keys": [],
            "byte_steps_mean": "raw bytes plus EOS, not writer/query controller events",
            "extra_host_accounting": "segment once per distinct call; no encoder/query replay",
        }
        report["count_scope"] = (
            "Decoder changed flags are lesson-level ORs; "
            "decoder_changed_steps counts actual byte updates. "
            "Teacher bytes and all-257 residual opportunities include frozen offers. "
            "Acquisition is byte-supervised, without action feedback. "
            "Memory-write and writer-credit event opportunities include final ingestion; "
            "they are not changed-matrix or slow-parameter update counts. "
            "Writes-disabled retains controller eligibility steps "
            "but skips writes/credit transport. "
            "Writer-frozen and frozen perform writes without writer eligibility/credit transport. "
            "Worker per-output seconds excludes preceding distractor reads; "
            "worker.package_seconds includes those reads and all input accounting."
        )
        assert len(report["lessons"]) == len(indices) * len(ARMS) and len(report["actions"]) == 120
        assert loaded_library_identity().to_record() == report["native"]
        guard()
        report["status"] = "completed"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        try:
            if "source_sha256" in report:
                report["source_sha256_after"] = experiment.sources(spec)
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


def main(experiment: Experiment):
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
        worker(experiment, arguments)
    else:
        raise SystemExit(0 if run(experiment, arguments) else 1)
