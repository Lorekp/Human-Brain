"""One public persistent stream; no fit selection, replay or hidden repair.

The CLI is held until source review. Imports of neuronal code and all runtime
work occur only inside run(). Root's independent saved review is external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

DRAFT = False
ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("online", "learning_disabled", "simple")
PHASES = ("acquisition", "transfer", "recovery", "retention")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def save(path, row):
    with Path(path).open("xb") as handle:
        handle.write(encode(row))
        handle.flush()
        os.fsync(handle.fileno())


class Journal:
    def __init__(self, path):
        self.path = path
        self.handle = path.open("xb")
        self.rows = 0

    def append(self, row):
        raw = encode(row)
        self.handle.write(raw)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.rows += 1
        return hashlib.sha256(raw).hexdigest()

    def close(self):
        self.handle.close()


def _plain(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def reader_summary(prediction):
    """Bounded actual trace summary, not a complete independent neural replay."""
    return {
        "schema": "active-path-output-prediction-v1",
        "attempt_id": prediction.attempt_id,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "trace_sha256": prediction.trace_sha256,
        "max_bytes": prediction.max_bytes,
        "selected_features": [value.tolist() for value in prediction.selected_features],
        "answer_hex": prediction.generation.data.hex(),
        "stop_reason": prediction.generation.stop_reason,
        "decisions": len(prediction.steps),
        "tokens": [step.observed_token for step in prediction.steps],
        "steps": [
            {
                "output_step": step.output_step,
                "prefix_hex": step.prefix.hex(),
                "token": step.observed_token,
                "selected_index": step.selected_index,
                "address_scores": step.competition.scores.tolist(),
                "competition_work": step.competition.work,
                "prior_log_belief": step.prior_log_belief.tolist(),
                "logits": step.logits.tolist(),
                "probabilities": step.probabilities.tolist(),
                "nll": step.nll,
            }
            for step in prediction.steps
        ],
        "work": prediction.work,
    }


def summarize(attempts, manifest, prior_report):
    result = {}
    for route in ROUTES:
        rows = [row for row in attempts if row["route"] == route]
        first = {row["episode_id"]: row for row in rows if row["attempt_index"] == 1}
        phases = {}
        for phase in PHASES:
            selected = [row for row in rows if row["phase"] == phase]
            phases[phase] = {
                kind: {
                    "episodes": sum(
                        row["attempt_index"] == 1 for row in selected if row["kind"] == kind
                    ),
                    "attempts": sum(row["kind"] == kind for row in selected),
                    "first_correct": sum(
                        row["correct"]
                        for row in selected
                        if row["kind"] == kind and row["attempt_index"] == 1
                    ),
                    "unsupported": sum(
                        row["status"] == "unsupported" for row in selected if row["kind"] == kind
                    ),
                }
                for kind in ("reading", "edit")
            }
        recovery = [row for row in rows if row["phase"] == "recovery"]
        failed = {
            row["episode_id"]
            for row in recovery
            if row["attempt_index"] == 1 and not row["correct"]
        }
        recovered = {
            row["episode_id"] for row in recovery if row["episode_id"] in failed and row["correct"]
        }
        final_recovery = [row for row in recovery if row["attempt_index"] == 3]
        retained = []
        for episode in manifest["episodes"]:
            if episode["phase"] != "retention":
                continue
            prior_route = "simple" if route == "simple" else "online"
            matches = [old for old in prior_report["metrics"][prior_route]["retention"]
                       if old["episode_id"] == episode["prior_retention_id"]]
            if len(matches) != 1:
                raise ValueError("retention requires one authenticated prior-session outcome")
            before, after = matches[0], first[episode["episode_id"]]
            retained.append(
                {
                    "episode_id": episode["episode_id"],
                    "prior_acquisition_id": episode["prior_episode_id"],
                    "prior_retention_id": before["episode_id"],
                    "prior_route": prior_route,
                    "initial_correct": before["delayed_correct"],
                    "delayed_correct": after["correct"],
                }
            )
        result[route] = {
            "phases": phases,
            "recovery": {
                "first_failures": len(failed),
                "later_correct": len(recovered),
                "final_attempt_correct": sum(row["correct"] for row in final_recovery),
                "final_recovered_after_first_failure": sum(
                    row["correct"] and row["episode_id"] in failed for row in final_recovery
                ),
                "policy": "three fixed attempts; private grade never controls retries",
                "tool_feedback_consumed": False,
            },
            "retention": retained,
            "retained": sum(x["initial_correct"] and x["delayed_correct"] for x in retained),
            "forgotten": sum(x["initial_correct"] and not x["delayed_correct"] for x in retained),
            "improved": sum(not x["initial_correct"] and x["delayed_correct"] for x in retained),
            "acquisition_blocks": [
                sum(
                    first[e["episode_id"]]["correct"]
                    for e in manifest["episodes"][start : start + 12]
                )
                for start in (0, 12)
            ],
        }
    return result


def run(args):
    started = time.perf_counter()
    deadline = started + 840.0  # separate saved audit retains60s of complete900s envelope
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch

    from experiments.r4_integrated.persistent_language_action_execution_v1 import (
        execute_proposal,
        grade_execution,
        proposal_from_record,
        qualified_launch,
        workspace_from_records,
    )
    from experiments.r4_integrated.persistent_language_action_initial_v3 import initialize
    from experiments.r4_integrated.persistent_language_action_tasks_v3 import (
        observation,
        restore_baseline,
    )
    from humanbrain.neuronal_splice_learner import _digest

    torch.set_num_threads(1)
    if sha(args.manifest) != args.manifest_sha:
        raise ValueError("frozen public manifest hash mismatch")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["schema"] != "persistent-language-action-development-data-v3"
            or manifest["documents"]
            or any(e["append_document_ids"] for e in manifest["episodes"])):
        raise ValueError("third session must retain its whole tape without append or eviction")
    with (output / "manifest.json").open("xb") as handle:
        handle.write(manifest_bytes)
    if sha(output / "manifest.json") != args.manifest_sha:
        raise ValueError("manifest changed during publication")
    sources = {
        str(path.relative_to(ROOT)): sha(path)
        for parent in (ROOT / "src/humanbrain", ROOT / "experiments/r4_integrated")
        for path in sorted(parent.glob("*.py"))
    }
    save(output / "sources.json", sources)
    models, states, initialization = initialize(args.prior.resolve(), args.bank.resolve())
    if set(models) != {"online", "learning_disabled"}:
        raise ValueError("initializer must return the two matched persistent routes")
    save(output / "initialization.json", initialization)
    launch = qualified_launch(ROOT)
    save(output / "launch-record.json", launch.to_record())
    baseline, restoration = restore_baseline(ROOT)
    save(output / "baseline-restoration.json", restoration)
    # Initialization authenticates this report against the saved result bank.
    prior_report_bytes = (args.prior.resolve().parent / "report.json").read_bytes()
    if hashlib.sha256(prior_report_bytes).hexdigest() != initialization["prior_report_sha256"]:
        raise ValueError("authenticated prior report changed before retention comparison")
    prior_report = json.loads(prior_report_bytes)
    journals = {
        name: Journal(output / f"{name}.jsonl") for name in ("attempts", "teachers", "ingests")
    }
    attempts = []

    def hashes(route):
        if route == "simple":
            return (_digest(_plain(baseline.demonstrations)), _digest(_plain(baseline.documents)))
        return (_digest(models[route].state_dict()), _digest(vars(states[route])))

    initial_hashes = {
        route: dict(zip(("model", "memory"), hashes(route), strict=True)) for route in ROUTES
    }
    try:
        for episode in manifest["episodes"]:
            for route in ROUTES:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("session840-second deadline;60seconds reserved for review")
                files = episode["files"]
                for index in range(1, episode["attempt_limit"] + 1):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("session deadline exceeded before attempt")
                    tick = time.perf_counter()
                    request = bytes.fromhex(episode["request_hex"])
                    before_model, before_memory = hashes(route)
                    public = observation(episode)
                    public["files"] = files
                    row = {
                        "schema": "persistent-language-action-attempt-v3",
                        "route": route,
                        "episode_id": episode["episode_id"],
                        "phase": episode["phase"],
                        "kind": episode["kind"],
                        "attempt_index": index,
                        "request_hex": request.hex(),
                        "observation": public,
                        "before_files": files,
                        "after_files": files,
                        "proposal": None,
                        "answer_hex": None,
                        "prediction": None,
                        "status": "completed",
                        "execution": None,
                        "correct": False,
                        "model_before_sha256": before_model,
                        "memory_before_sha256": before_memory,
                        "memory_used": (
                            sum(len(doc) + 1 for doc in baseline.documents)
                            if route == "simple"
                            else states[route].used
                        ),
                        "model_instance_id": id(baseline if route == "simple" else models[route]),
                        "feedback_consumed": False,
                        "work": {},
                    }
                    prediction, workspace, proposal, envelope = None, None, None, None
                    if route != "simple" and len(request) > 31:
                        row["status"] = "unsupported"
                        row["error"] = "raw request exceeds declared31-byte interface; unchanged"
                    elif episode["kind"] == "reading":
                        if route == "simple":
                            reading = baseline.predict(request, files, kind="reading")
                            row["prediction"] = reading
                            row["answer_hex"] = reading["answer_hex"]
                        else:
                            prediction = models[route].reader.predict_with_receipt(
                                request, states[route], 16
                            )
                            row["prediction"] = reader_summary(prediction)
                            row["answer_hex"] = prediction.generation.data.hex()
                            row["work"] = prediction.work
                    if episode["kind"] == "edit":
                        try:
                            workspace = workspace_from_records(files)
                        except ValueError as error:
                            row["status"], row["error"] = "unsupported", str(error)
                        if workspace is not None and row["status"] != "unsupported":
                            if route == "simple":
                                result = baseline.predict(request, files, kind="edit")
                                row["prediction"] = result
                                proposal = proposal_from_record(result["proposal"], workspace)
                            else:
                                prediction = models[route].propose(
                                    request, workspace, states[route]
                                )
                                proposal = prediction.proposal
                                row["prediction"] = {
                                    "attempt_id": prediction.attempt_id,
                                    "proposal_sha256": prediction.proposal_sha256,
                                    "model_sha256": prediction.model_sha256,
                                    "memory_sha256": prediction.memory_sha256,
                                    "payload": reader_summary(prediction.payload),
                                    "heads": {
                                        key: {
                                            "logits": value.logits.tolist(),
                                            "selected_index": value.selected_index,
                                        }
                                        for key, value in prediction.choices.items()
                                    },
                                }
                                row["work"] = prediction.work
                            row["proposal"] = proposal.to_record()
                        after, execution = execute_proposal(
                            workspace if workspace is not None else files,
                            proposal,
                            output / "executions" / route / episode["episode_id"] / str(index),
                            launch=launch,
                            deadline=deadline,
                        )
                        row["after_files"], row["execution"] = after, execution
                        row["grade"] = grade_execution(
                            execution, episode["grade"], episode["files"]
                        )
                        row["correct"] = (
                            row["status"] != "unsupported" and row["grade"]["task_success"]
                        )
                        if prediction is not None:
                            envelope = {
                                "attempt_id": prediction.attempt_id,
                                "proposal_sha256": prediction.proposal_sha256,
                                "workspace_sha256": workspace.sha256,
                                "execution": execution,
                            }
                            if episode["phase"] != "acquisition":
                                row["ack_work"] = models[route].finish_without_teaching(
                                    prediction, workspace, states[route], envelope
                                )
                    else:
                        row["correct"] = (
                            row["status"] != "unsupported"
                            and row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                        )
                        if prediction is not None and episode["phase"] != "acquisition":
                            row["ack_work"] = models[route].reader.acknowledge(
                                prediction, states[route]
                            )
                    row["model_after_sha256"], row["memory_after_sha256"] = hashes(route)
                    row["elapsed_seconds"] = time.perf_counter() - tick
                    attempt_sha = journals["attempts"].append(row)
                    attempts.append(row)
                    if row["execution"] is not None and not row["execution"]["files_complete"]:
                        raise ValueError(
                            "incomplete actual workspace retained; cannot resume from partial files"
                        )
                    # Durable actual action/response precedes any target-bearing call.
                    if episode["phase"] == "acquisition":
                        teacher = episode["teacher"]
                        teacher_tick = time.perf_counter()
                        teacher_row = {
                            "schema": "persistent-language-action-teacher-v3",
                            "route": route,
                            "episode_id": episode["episode_id"],
                            "attempt_sha256": attempt_sha,
                            "teacher": teacher,
                            "learning_enabled": route != "learning_disabled",
                            "model_before_sha256": row["model_after_sha256"],
                            "memory_before_sha256": row["memory_after_sha256"],
                            "reader_offers": 0,
                            "action_labels": 0,
                            "work": {},
                        }
                        if row["status"] == "unsupported":
                            raise ValueError("authored acquisition must fit actual interface")
                        if route == "simple":
                            baseline.observe_teacher(
                                request, files, kind=episode["kind"], teacher=teacher
                            )
                            teacher_row["publication"] = {
                                "demonstration_stored": episode["kind"] == "edit"
                            }
                        else:
                            if episode["kind"] == "reading":
                                feedback = models[route].reader.teach(
                                    request,
                                    bytes.fromhex(teacher["target_hex"]),
                                    states[route],
                                    0.01,
                                    prediction=prediction,
                                    learning_enabled=route == "online",
                                )
                                reader_feedback = feedback
                                teacher_row["publication"] = {}
                            else:
                                feedback = models[route].teach(
                                    prediction,
                                    workspace,
                                    states[route],
                                    proposal_from_record(teacher, workspace),
                                    envelope,
                                    learning_enabled=route == "online",
                                )
                                reader_feedback = feedback.reader_feedback
                                teacher_row["action_labels"] = len(feedback.action_losses)
                                teacher_row["publication"] = {"actions": feedback.action_updates}
                            teacher_row["work"] = feedback.work
                            if reader_feedback is not None:
                                teacher_row["reader_offers"] = reader_feedback.teacher_count
                                teacher_row["reader_feedback"] = {
                                    "total_loss": reader_feedback.total_loss,
                                    "mean_loss": reader_feedback.mean_loss,
                                    "correction_position": reader_feedback.correction_position,
                                    "correction_target": reader_feedback.correction_target,
                                    "active_output_offers": reader_feedback.active_output_offers,
                                    "actual_response_id": (
                                        prediction.attempt_id if episode["kind"] == "reading"
                                        else prediction.payload.attempt_id
                                    ),
                                    "steps": [
                                        {
                                            "output_step": step.output_step,
                                            "prefix_hex": step.prefix.hex(),
                                            "token": step.observed_token,
                                            "nll": step.nll,
                                            "probabilities": step.probabilities.tolist(),
                                            "prior_log_belief": step.prior_log_belief.tolist(),
                                            "posterior_log_belief": step.posterior_log_belief.tolist(),
                                        }
                                        for step in reader_feedback.steps
                                    ],
                                    "scope": "actual mixtures and current teacher responsibility; logits omitted",
                                }
                                teacher_row["publication"].update(
                                    codec=reader_feedback.codec_updated,
                                    address=reader_feedback.address_updated,
                                    output=reader_feedback.output_updated,
                                    query_address=reader_feedback.query_address_updated,
                                    prefix_address=reader_feedback.prefix_address_updated,
                                    legacy_codec=reader_feedback.legacy_codec_updated,
                                )
                                teacher_row["publication_alias"] = "codec means active output U"
                        teacher_row["model_after_sha256"], teacher_row["memory_after_sha256"] = (
                            hashes(route)
                        )
                        teacher_row["elapsed_seconds"] = time.perf_counter() - teacher_tick
                        journals["teachers"].append(teacher_row)
                    files = row["after_files"]
        for journal in journals.values():
            journal.close()
        checkpoint = {
            "schema": "persistent-language-action-state-v3",
            "routes": {route: model.snapshot(states[route]) for route, model in models.items()},
        }

        def tensor_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            return 0

        checkpoint_tensor_bytes = tensor_bytes(checkpoint)
        with (output / "final-state.pt").open("xb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        del checkpoint
        report = {
            "schema": "persistent-language-action-report-v3",
            "status": "completed",
            "routes": list(ROUTES),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "initial_hashes": initial_hashes,
            "final_hashes": {
                route: dict(zip(("model", "memory"), hashes(route), strict=True))
                for route in ROUTES
            },
            "metrics": summarize(attempts, manifest, prior_report),
            "counts": {k: v.rows for k, v in journals.items()},
            "elapsed_seconds": time.perf_counter() - started,
            "clock": "time.perf_counter",
            "clock_metadata": {
                "implementation": time.get_clock_info("perf_counter").implementation,
                "monotonic": time.get_clock_info("perf_counter").monotonic,
                "adjustable": time.get_clock_info("perf_counter").adjustable,
                "resolution_seconds": time.get_clock_info("perf_counter").resolution,
            },
            "budget": {
                "complete_seconds": 900,
                "runner_seconds": 840,
                "reserved_audit_seconds": 60,
                "peak_mib": 2048,
            },
            "scope": (
                "source-bound actual summaries; no complete neural replay; no feedback encoder"
            ),
            "prior_loads": 1,
            "models_constructed": 2,
            "teacher_free_recovery_evaluated": True,
            "final_checkpoint": {
                "path": "final-state.pt",
                "sha256": sha(output / "final-state.pt"),
                "tensor_bytes": checkpoint_tensor_bytes,
                "snapshots": 2,
                "scope": "both model/tape/experience; continued prior online memory",
            },
            "baseline_limits": "raw long queries allowed; trigram retrieval has no31-byte cap",
            "retention_scope": (
                "previous-session delayed outcome vs current delayed outcome; "
                "no immediate post-teacher probe"
            ),
            "continuation": manifest["continuation"],
            "initial_memory_slots": 505,
            "final_memory_slots": {route: states[route].used for route in models},
            "functional_prior_parity": False,
        }
        save(output / "report.json", report)
        if sources != {key: sha(ROOT / key) for key in sources}:
            raise ValueError("source bytes changed during session")
        if time.perf_counter() >= deadline:
            raise TimeoutError("session exceeded840seconds including final serialization")
        save(
            output / "completion.json",
            {
                "schema": "persistent-language-action-completion-v3",
                "status": "completed",
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "files": {
                    str(path.relative_to(output)).replace("\\", "/"): sha(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    except BaseException as error:
        for journal in journals.values():
            if not journal.handle.closed:
                journal.close()
        save(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument(
        "--prior",
        type=Path,
        default=ROOT / ".pytest-tmp/persistent-language-action-v2-run/final-state.pt",
    )
    parser.add_argument(
        "--bank", type=Path, default=ROOT / "benchmarks/persistent_language_action_v2.result.json"
    )
    args = parser.parse_args()
    if DRAFT:
        raise SystemExit("DRAFT: source review and root release required")
    run(args)


if __name__ == "__main__":
    main()
