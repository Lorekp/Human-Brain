"""Five persistent neuronal learners with explicit competitive/marginal free recall.

Parent-only teachers and external grading. The unchanged action executor uses
its stable action-row-v1 format; enclosing learning and endpoint schemas are v2.
Importing this draft performs no model or artifact work.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from experiments.r4_integrated.neuronal_splice_models_v2 import ARMS, reader_policy
from experiments.r4_integrated.neuronal_splice_online_v1 import (
    cap_fixture,
    category_probes,
    execute_action,
    proposal_from_record,
    qualified_launch,
    workspace_from_records,
    write_json,
)
from experiments.r4_integrated.neuronal_splice_online_v1 import (
    source_manifest as inherited_source_manifest,
)
from experiments.r4_integrated.neuronal_splice_prior_v1 import file_sha256

SCHEMA = "neuronal-splice-online-v2"
DRAFT = False
RECEIPT_LIMITS = {"max_line_bytes": 2**21, "max_row_tree_bytes": 2**21,
                  "max_live_rows": 2, "max_line_copies": 2, "allocation_factor": 2}
CONFIG = {
    "schema": SCHEMA, "arms": list(ARMS), "seed": 733, "learning_rate": .01,
    "reader_policy": {arm: reader_policy(arm) for arm in ARMS},
    "lessons_per_arm": 256, "cost_task_ids": ["T00", "T08", "T16", "T24"],
    "fresh_per_arm": 32, "retention_per_arm": 16, "panel_per_pass_per_arm": 20,
    "retained_memories_per_arm": 72, "retained_tensor_bytes": 239657600,
    "receipt_limits": RECEIPT_LIMITS, "seconds": 900, "tree_mib": 2048,
    "run_reserve_mib": 128, "audit_reserve_mib": 32, "buffer_mib": 24,
    "duration_clock": "perf_counter", "full_only_tensor_bytes": 0,
}
SPEC = SimpleNamespace(config=CONFIG)


def new_timing_summary():
    return {arm: {phase: {"rows": 0, **{f"{stage}_{kind}": 0.0
                                       for stage in ("model", "action", "journal")
                                       for kind in ("seconds", "max_seconds")}}
                  for phase in ("regular", "cap")} for arm in ARMS}


def learning_experience(*, model, state, task, arm, index, phase, output,
                        launch, action_journal, lesson_journal, timing_journal, summary):
    from experiments.r4_integrated.neuronal_splice_receipts_v2 import (
        compact_feedback,
        compact_prediction,
    )
    from humanbrain.neuronal_splice_learner import _digest

    model_started = time.perf_counter()
    workspace = workspace_from_records(task["files"])
    if workspace.sha256 != task["workspace_sha256"]:
        raise ValueError("task workspace identity changed")
    prediction = model.propose(bytes.fromhex(task["request_hex"]), workspace, state)
    free_seconds = time.perf_counter() - model_started
    envelope, action_hash, action_seconds, grade = execute_action(
        task=task if phase == "regular" else None, workspace=workspace,
        proposal=prediction.proposal, attempt_id=prediction.attempt_id, phase=phase,
        arm=arm, index=index, output=output, launch=launch, journal=action_journal,
    )
    teacher_started = time.perf_counter()
    teacher = proposal_from_record(task["teacher"])
    feedback = model.teach(prediction, workspace, state, teacher, envelope, learning_rate=.01)
    model_after, memory_after = _digest(model.state_dict()), _digest(state.snapshot())
    if memory_after != prediction.memory_sha256:
        raise ValueError("teaching changed a retained literal tape")
    teacher_completed = time.perf_counter()
    model_seconds = free_seconds + teacher_completed - teacher_started
    journal_started = time.perf_counter()
    row = {
        "schema": "neuronal-splice-learning-row-v2", "phase": phase, "arm": arm,
        "index": index, "task_id": task["task_id"], "memory_id": task["memory_id"],
        "request_hex": task["request_hex"], "workspace_sha256": workspace.sha256,
        "action_line_sha256": action_hash, "prediction": compact_prediction(
            prediction, state, expected_policy=reader_policy(arm)),
        "teacher": teacher.to_record(), "feedback": compact_feedback(feedback, state),
        "model_after_sha256": model_after, "memory_after_sha256": memory_after,
        "model_seconds": model_seconds, "model_instance_token": str(id(model)),
        "teacher_started_perf": teacher_started, "teacher_completed_perf": teacher_completed,
    }
    lesson_journal.append(row)
    del row
    totals = summary[arm][phase]
    totals["rows"] += 1
    for stage, seconds in (("model", model_seconds), ("action", action_seconds)):
        totals[f"{stage}_seconds"] += seconds
        totals[f"{stage}_max_seconds"] = max(totals[f"{stage}_max_seconds"], seconds)
    partial_journal = time.perf_counter() - journal_started
    timing_journal.append({"phase": phase, "arm": arm, "index": index,
                           "model_seconds": model_seconds, "action_seconds": action_seconds,
                           "journal_before_timing_append_seconds": partial_journal})
    complete_journal = time.perf_counter() - journal_started
    totals["journal_seconds"] += complete_journal
    totals["journal_max_seconds"] = max(totals["journal_max_seconds"], complete_journal)
    if phase == "cap":
        expected = b"\0" * 16 if arm == "memory_disabled" else b"A" * 16
        reason = "limit" if arm == "memory_disabled" else "eos"
        if (prediction.proposal.operation != "STOP" or prediction.proposal.replacement != expected
                or prediction.proposal.stop_reason != reason or len(prediction.payload.steps) != 17
                or prediction.choices["start"].logits.numel() != 516
                or prediction.choices["end"].logits.numel() != 129
                or feedback.reader_feedback.teacher_count != 17
                or set(feedback.teacher_choices) != {"start", "end", "operation"}):
            raise ValueError("authored cap did not exercise its declared complete geometry")
    return grade


def ingest_stage(models, states, roster, stage):
    """One write per declared episode, identical source scope in COST and FULL."""
    ledger = {}
    ids = roster[f"{stage}_memory_ids"]
    for arm, model in models.items():
        raw_bytes = eos = 0
        for memory_id in ids:
            if memory_id in states[arm]:
                raise ValueError("an episode would be ingested twice")
            documents = [bytes.fromhex(item) for item in
                         roster["memory_packages"][memory_id]["document_hex"]]
            state = model.ingest(documents)
            states[arm][memory_id] = state
            raw_bytes += sum(map(len, documents))
            eos += len(documents)
            if arm == "memory_disabled" and (
                    state.keys.count_nonzero() or state.sensory.count_nonzero()):
                raise ValueError("disabled memory contains retained data")
        ledger[arm] = {"states": len(ids), "raw_bytes": raw_bytes, "eos": eos}
    return ledger


def reading_panel(models, states, roster, journal):
    from experiments.r4_integrated.neuronal_splice_receipts_v2 import compact_reader
    from humanbrain.neuronal_splice_learner import _digest

    for arm, model in models.items():
        for index, case in enumerate(roster["old_panel"]):
            started = time.perf_counter()
            state = states[arm][case["memory_id"]]
            before_model, before_memory = _digest(model.state_dict()), _digest(state.snapshot())
            prediction = model.reader.predict_with_receipt(
                bytes.fromhex(case["query_hex"]), state, 2)
            if (_digest(model.state_dict()) != before_model
                    or _digest(state.snapshot()) != before_memory):
                raise ValueError("reader-only panel changed its model or tape")
            seconds = time.perf_counter() - started
            journal.append({"schema": "neuronal-splice-panel-row-v2", "phase": "pre",
                            "arm": arm, "index": index, "memory_id": case["memory_id"],
                            "request_hex": case["query_hex"], "model_sha256": before_model,
                            "memory_sha256": before_memory,
                            "prediction": compact_reader(
                                prediction, state, expected_policy=reader_policy(arm)),
                            "model_seconds": seconds})


def source_manifest(root: Path) -> dict:
    """Bind the complete inherited parent closure plus every successor source."""
    paths = set(inherited_source_manifest(root))
    paths.update(f"experiments/r4_integrated/{name}.py" for name in (
        "neuronal_splice_online_v2", "neuronal_splice_models_v2", "neuronal_splice_prior_v2",
        "neuronal_splice_roster_v2", "neuronal_splice_worker_v2", "neuronal_splice_receipts_v2",
        "neuronal_splice_endpoint_v2", "check_neuronal_splice_online_v2",
        "check_neuronal_address_competition_v1",
    ))
    paths.update((
        "src/humanbrain/competitive_local_memory_learner.py",
        "src/humanbrain/neuronal_address_competition.py",
        "docs/research/neuronal-address-competition-v1.md",
        "docs/research/neuronal-address-competition-integration-v1.md",
        "docs/research/neuronal-address-competition-comparison-v1.md",
    ))
    return {path: file_sha256(root / path) for path in sorted(paths)}


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("integrated run deadline")


def run_experiment(root, output, mode, admission_path, admission_sha, deadline):
    """Cost/full share one loop; execution requires a separately frozen admission."""
    if DRAFT:
        raise RuntimeError("draft harness requires completed source and ordinary QA review")
    import os
    import traceback

    import torch

    from benchmarks.native_scaling import _process_memory
    from experiments.r4_integrated.neuronal_splice_endpoint_v2 import export_and_evaluate
    from experiments.r4_integrated.neuronal_splice_io_v1 import JsonlJournal
    from experiments.r4_integrated.neuronal_splice_prior_v1 import authenticated_json, tensor_bytes
    from experiments.r4_integrated.neuronal_splice_prior_v2 import initialize_models
    from humanbrain.neuronal_splice_learner import _digest

    started = time.perf_counter()
    if mode not in {"cost", "run"}:
        raise ValueError("unknown experiment phase")
    admission = authenticated_json(admission_path, admission_sha)
    if (admission["schema"] != "neuronal-splice-admission-v2"
            or admission["config"] != CONFIG or admission["status"] != "passed"
            or admission["authorized_phase"] != mode
            or admission["qa_passed"] is not True):
        raise ValueError("execution lacks exact source/QA admission")
    current_sources = source_manifest(root)
    if admission["source_sha256"] != current_sources:
        raise ValueError("admitted source changed")
    if mode == "run":
        from experiments.r4_integrated.check_neuronal_splice_online_v2 import validate_cost_decision

        validate_cost_decision(Path(admission["cost_decision_path"]),
                               admission["cost_decision_sha256"], CONFIG,
                               current_sources, admission["roster_sha256"])
    roster = authenticated_json(Path(admission["roster_path"]), admission["roster_sha256"])
    if roster["schema"] != "neuronal-splice-roster-v2":
        raise ValueError("admitted roster schema changed")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    journals = {name: JsonlJournal(output / f"{name}.jsonl", SPEC)
                for name in ("actions", "lessons", "timings", "pre_panel")}
    report = {"schema": SCHEMA, "mode": mode, "status": "running", "config": CONFIG,
              "source_sha256": current_sources, "admission_path": str(admission_path),
              "admission_sha256": admission_sha, "roster_path": str(admission["roster_path"]),
              "roster_sha256": admission["roster_sha256"], "pid": os.getpid(),
              "started_unix": time.time(), "deadline_unix": deadline,
              "duration_clock": {key: getattr(time.get_clock_info("perf_counter"), key)
                                 for key in ("implementation", "resolution", "monotonic",
                                             "adjustable")}}
    logs = {}
    try:
        launch = qualified_launch(root)
        report["action_launch_record"] = launch.to_record()
        models, report["prior"] = initialize_models(root)
        initial_hashes = {arm: _digest(model.state_dict()) for arm, model in models.items()}
        initial_action_hashes = {
            arm: _digest(model.actions.state_dict()) for arm, model in models.items()}
        report["initial_action_sha256"] = initial_action_hashes
        states = {arm: {} for arm in ARMS}
        summary = new_timing_summary()
        report["timing_summary"] = summary
        report["category_probes"] = []
        if mode == "cost":
            for arm, model in models.items():
                check_deadline(deadline)
                clone, state, task = cap_fixture(model)
                learning_experience(model=clone, state=state, task=task, arm=arm, index=0,
                                    phase="cap", output=output, launch=launch,
                                    action_journal=journals["actions"],
                                    lesson_journal=journals["lessons"],
                                    timing_journal=journals["timings"], summary=summary)
                if _digest(model.state_dict()) != initial_hashes[arm]:
                    raise ValueError("disposable cap changed its production learner")
                del clone, state, task
            report["category_probes"] = category_probes(
                output=output, launch=launch, journal=journals["actions"])
        report["ingestion_before"] = ingest_stage(models, states, roster, "before_learning")
        reading_panel(models, states, roster, journals["pre_panel"])
        selected = roster["cost_task_ids"] if mode == "cost" else roster["schedule"]
        report["selected_task_ids"] = selected
        curves = {arm: [] for arm in ARMS}
        report["acquisition_first_attempts"] = curves
        for index, task_id in enumerate(selected):
            for arm, model in models.items():
                check_deadline(deadline)
                task = roster["tasks"][task_id]
                grade = learning_experience(
                    model=model, state=states[arm][task["memory_id"]], task=task,
                    arm=arm, index=index, phase="regular", output=output, launch=launch,
                    action_journal=journals["actions"], lesson_journal=journals["lessons"],
                    timing_journal=journals["timings"], summary=summary,
                )
                curves[arm].append(grade["task_success"])
        report["ingestion_after"] = ingest_stage(models, states, roster, "after_learning")
        for arm in ARMS:
            before, after = report["ingestion_before"][arm], report["ingestion_after"][arm]
            if (len(states[arm]) != 72 or before["raw_bytes"] + after["raw_bytes"] != 4022
                    or before["eos"] + after["eos"] != 168
                    or max(state.used for state in states[arm].values()) != 125):
                raise ValueError("actual retained ingestion scope changed")
        retained = sum(tensor_bytes(model.state_dict()) for model in models.values())
        retained += sum(tensor_bytes(vars(state)) for memories in states.values()
                        for state in memories.values())
        if retained != CONFIG["retained_tensor_bytes"]:
            raise ValueError("actual retained tensor scope changed")
        report["retained_tensor_bytes"] = retained
        report["worker"] = export_and_evaluate(root, output, models, states, roster, launch,
                                                journals["actions"], deadline)
        report["final_model_sha256"] = {
            arm: _digest(model.state_dict()) for arm, model in models.items()}
        report["final_action_sha256"] = {
            arm: _digest(model.actions.state_dict()) for arm, model in models.items()}
        if report["final_model_sha256"]["slow_frozen"] != initial_hashes["slow_frozen"]:
            raise ValueError("all-slow-frozen model changed")
        for arm in ("action_frozen", "slow_frozen"):
            if report["final_action_sha256"][arm] != initial_action_hashes[arm]:
                raise ValueError("frozen action parameters changed")
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        for name, journal in journals.items():
            logs[name] = journal.close()
        report["logs"] = logs
        report["source_sha256_after"] = source_manifest(root)
        if report["source_sha256_after"] != current_sources:
            report["status"] = "failed"
            report["source_changed"] = True
        report["elapsed_seconds"] = time.perf_counter() - started
        report["memory"] = _process_memory()
        report_sha = write_json(output / "report.json", report)
        write_json(output / "completion.json", {
            "schema": "neuronal-splice-completion-v2", "status": report["status"],
            "mode": mode, "report_sha256": report_sha, "admission_sha256": admission_sha,
            "elapsed_seconds": time.perf_counter() - started, "logs": logs,
        })


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cost", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha", required=True)
    parser.add_argument("--deadline-unix", type=float)
    args = parser.parse_args()
    for path in (args.output, args.admission):
        if not path.is_absolute():
            raise ValueError("all execution paths must be absolute")
    root = Path(__file__).resolve().parents[2]
    deadline = args.deadline_unix
    if deadline is None:
        deadline = float(os.environ["HUMANBRAIN_RUN_DEADLINE_UNIX"])
    run_experiment(root, args.output, args.mode, args.admission, args.admission_sha, deadline)


if __name__ == "__main__":
    main()
