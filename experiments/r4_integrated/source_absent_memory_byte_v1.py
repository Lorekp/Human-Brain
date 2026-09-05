"""Bounded engineering seam. Running fit requires an exact frozen admission.

The fresh worker receives tensor checkpoints and query bytes, never the offline
document/answer roster. No parser, span selector, retrieval cache or answer list
is available to network inference. This is data routing, not an OS sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED = time.monotonic()
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
ARMS = ("full", "writes_disabled", "frozen", "controller_bypass")
CONFIG = {
    "steps": 256,
    "batch_size": 4,
    "learning_rate": 0.001,
    "clip": 1.0,
    "embedding_size": 32,
    "hidden_size": 64,
    "query_hidden_size": 32,
    "max_bytes": 32,
    "seed": 733,
    "seconds": 900,
    "parent_mib": 2048,
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, record):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sources():
    from humanbrain.native import procedural_runtime_source_manifest

    names = [
        "experiments/r4_integrated/source_absent_memory_byte_v1.md",
        "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1",
        "experiments/r4_integrated/__init__.py",
        "src/humanbrain/__init__.py",
        "src/humanbrain/plastic_memory.py",
        "src/humanbrain/byte_sequence_decoder.py",
        "experiments/r4_integrated/contextual_occurrence_motor.py",
        "experiments/r4_integrated/utf8_action_bridge.py",
        "experiments/r4_integrated/real_input_action_bridge.py",
        "experiments/r4_integrated/neutral_document_python_session.py",
        "src/humanbrain/opaque_document_event_tape.py",
        "src/humanbrain/multiscale_opaque_event_tape.py",
        "src/humanbrain/unified_opaque_observation_tape.py",
        "src/humanbrain/knowledge_conditioned_occurrence_substitution.py",
        "src/humanbrain/variability_aware_segment_binder.py",
        "src/humanbrain/resting_glif_race.py",
        "src/humanbrain/neuronal_relational_reader.py",
        "src/humanbrain/sparse_neuronal_role_filler.py",
        "src/humanbrain/persistent_neuronal_document_learner.py",
        "src/humanbrain/neutral_neuronal_document_learner.py",
        "src/humanbrain/neutral_document_action.py",
        "src/humanbrain/readout.py",
        "src/humanbrain/recurrent.py",
        "src/humanbrain/recurrent_state.py",
        "src/humanbrain/topology.py",
        "benchmarks/native_scaling.py",
        "pyproject.toml",
    ]
    return {
        **{r["path"]: r["sha256"] for r in procedural_runtime_source_manifest(ROOT)},
        Path(__file__).relative_to(ROOT).as_posix(): file_hash(__file__),
        **{name: file_hash(ROOT / name) for name in names},
    }


def roster():
    """Offline synthetic authoring, never called by the inference worker."""
    # Values are raw original bytes, including spaces, accents and apostrophes.
    values = ["brin", "sora", "lune", "mavi", "brin sora", "mavi lune", "café", "o'lin"]
    novel = ["sora brin", "lune mavi", "café sora", "o'lin brin"]
    train = [{"key": f"k{i:02d}", "value": v} for i, v in enumerate(values)]
    checks = []
    for group, count in (("fit", 8), ("unseen_value", 4), ("delayed", 4), ("interference", 4)):
        for i in range(count):
            value = values[i] if group == "fit" else novel[i]
            pair = i // 2
            keys = [2 * pair, 2 * pair + 1]
            if pair % 2:
                keys.reverse()
            pool_values = values if group == "fit" else novel
            docs = [f"Record k{j:02d} has value {pool_values[j]}.".encode() for j in keys]
            base_id = f"{'fit' if group == 'fit' else 'novel'}-{pair}"
            additional = (
                [f"Record k{j:02d} has value {values[j]}.".encode() for j in (4, 5, 6)]
                if group == "interference"
                else []
            )
            checks.append(
                {
                    "id": len(checks),
                    "group": group,
                    "memory_id": f"interference-{pair}" if additional else base_id,
                    "base_memory_id": base_id if additional else None,
                    "additional_document_hex": [d.hex() for d in additional],
                    "query_hex": f"What is the value of k{i:02d}?".encode().hex(),
                    "document_hex": [d.hex() for d in docs + additional],
                    "expected_hex": value.encode().hex(),
                    "delay_reads": 32 if group == "delayed" else 0,
                }
            )
    assert not set(novel) & set(values)
    return train, checks


def teaching_schedule(train):
    """Offline labels only: independently vary each key's value before fitting."""
    rng = random.Random(CONFIG["seed"])
    schedule = []
    for _ in range(CONFIG["steps"]):
        batch = []
        for _ in range(CONFIG["batch_size"]):
            keys = rng.sample([r["key"] for r in train], 2)
            values = rng.sample([r["value"] for r in train], 2)
            queried = rng.randrange(2)
            batch.append(
                {
                    "document_hex": [
                        f"Record {key} has value {value}.".encode().hex()
                        for key, value in zip(keys, values, strict=True)
                    ],
                    "query_hex": f"What is the value of {keys[queried]}?".encode().hex(),
                    "target_hex": values[queried].encode().hex(),
                }
            )
        schedule.append(batch)
    return schedule


def network(kind="cell"):
    import torch

    from humanbrain.byte_sequence_decoder import ByteSequenceDecoder
    from humanbrain.plastic_memory import PlasticMemory, StreamingRateCell

    class Joined(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(CONFIG["seed"])
                self.embedding = torch.nn.Embedding(257, 32)
                torch.nn.init.normal_(self.embedding.weight, std=0.02)
            self.memory = PlasticMemory(
                32, hidden_size=64, key_size=64, value_size=64, kind=kind, seed=CONFIG["seed"]
            )
            # Bypass ablates only the document writer, preserving query information.
            self.query_cell = StreamingRateCell(32, 32, kind="cell", seed=CONFIG["seed"] + 1)
            self.decoder = ByteSequenceDecoder(96, 64, 32, kind="cell", seed=CONFIG["seed"] + 2)

        def encoded(self, raw):
            # 256 is an explicit document/query terminator, not an extracted answer boundary.
            indices = torch.tensor([*raw, 256], dtype=torch.long)[None]
            return self.embedding(indices)

        def ingest(self, documents, *, state=None, writes_enabled=True):
            if state is None:
                state = self.memory.initial_state(1)
            for document in documents:
                words = self.encoded(document)
                state = self.memory.write(
                    words,
                    torch.ones(words.shape[:2], dtype=torch.bool),
                    state,
                    writes_enabled=writes_enabled,
                )
            return state

        def context(self, query, state):
            hidden = self.query_cell.initial_state(1)
            for value in self.encoded(query).unbind(1):
                hidden = self.query_cell(value, hidden)
            value = self.memory.read(hidden, state)
            return torch.cat((hidden, value), dim=-1)

    return Joined()


def tensor_bytes(values):
    import torch

    if isinstance(values, torch.Tensor):
        return values.numel() * values.element_size()
    if isinstance(values, dict):
        return sum(tensor_bytes(v) for v in values.values())
    return 0


def detached_state(value):
    """Permit only detached CPU tensors and primitive architecture metadata."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: detached_state(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detached_state(child) for child in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def same_state(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(same_state(x, y) for x, y in zip(a, b, strict=True))
        )
    return a == b


def worker(checkpoint, requests_path, output, checkpoint_sha, requests_sha):
    """No training/roster/evaluator/motor calls in this fresh process."""
    import torch

    from benchmarks.native_scaling import _process_memory

    assert file_hash(checkpoint) == checkpoint_sha and file_hash(requests_path) == requests_sha
    requests = json.loads(requests_path.read_bytes())
    assert set(requests) == {"source_sha256", "cases"}
    assert requests["source_sha256"] == sources()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(saved) == set(ARMS)
    outputs = []
    for arm in ARMS:
        entry = saved[arm]
        assert set(entry) == {"weights", "memories"}
        model = network("bypass" if arm == "controller_bypass" else "cell").eval()
        model.load_state_dict(entry["weights"], strict=True)
        with torch.no_grad():
            states = {}
            for memory_id, snapshot in entry["memories"].items():
                assert set(snapshot) == {"synapses", "writer"}
                assert not snapshot["writer"].count_nonzero()
                states[memory_id] = model.memory.restore_state(snapshot)
            for request in requests["cases"]:
                assert set(request) == {"id", "memory_id", "query_hex", "delay_reads"}
                snapshot = entry["memories"][request["memory_id"]]
                # Both keys and their delayed checks share this exact restored state object.
                state = states[request["memory_id"]]
                for _ in range(request["delay_reads"]):
                    model.context(b"What is the value of k99?", state)
                generated = model.decoder.generate(
                    model.context(bytes.fromhex(request["query_hex"]), state),
                    max_bytes=CONFIG["max_bytes"],
                )[0]
                assert all(torch.equal(v, state.snapshot()[k]) for k, v in snapshot.items())
                outputs.append(
                    {
                        "arm": arm,
                        "id": request["id"],
                        "memory_id": request["memory_id"],
                        "generated_hex": generated.data.hex(),
                        "stop_reason": generated.stop_reason,
                    }
                )
        assert same_state(entry["weights"], model.state_dict())
    assert file_hash(checkpoint) == checkpoint_sha and file_hash(requests_path) == requests_sha
    write_json(
        output,
        {
            "status": "completed",
            "checkpoint_sha256": checkpoint_sha,
            "query_requests_sha256": requests_sha,
            "outputs": outputs,
            "roster_invoked_or_documents_or_targets_supplied_to_model": False,
            "routing_boundary_not_filesystem_sandbox": True,
            "worker_pid": os.getpid(),
            "elapsed_seconds": time.monotonic() - PROCESS_STARTED,
            "memory": _process_memory(),
        },
    )


def actuate(generated, ordinal, motor, root):
    """Only generated bytes enter the existing motor; no target argument exists."""
    from experiments.r4_integrated.real_input_action_bridge import (
        _files_record,
        escape_literal_content,
    )
    from experiments.r4_integrated.utf8_action_bridge import Case, execute_utf8

    raw = bytes.fromhex(generated["generated_hex"])
    reason = generated["stop_reason"]
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        reason = "invalid_utf8"
    accepted = reason == "eos" and bool(raw)
    value = raw if accepted else b"pending"
    case = Case(
        f"generated-{ordinal}",
        f"qa{ordinal:03d}",
        value.decode(),
        f"qa_config{ordinal:03d}",
        "pending",
    )
    before = case.files()
    motor_before = motor.checkpoint_sha256
    action = (
        motor.apply(f"Configure {case.key}".encode(), before, escape_literal_content(raw))
        if accepted
        else None
    )
    after = before if action is None else action.files
    receipt = execute_utf8(case, after, before, root)
    assert motor.checkpoint_sha256 == motor_before
    lossless = (
        receipt["returncode"] == 0
        and receipt["stdout_hex"] == (value + b"\n").hex()
        and receipt["stderr_hex"] == ""
        and receipt["decoys_unchanged"]
        and receipt["materialized_matches_action"]
    )
    return {
        "generated_hex": raw.hex(),
        "generation_status": reason,
        "emitted": accepted,
        "motor_checkpoint_sha256": motor_before,
        "before": _files_record(before),
        "after": _files_record(after),
        "lossless_generated_execution": lossless,
        "execution": receipt,
    }


def run(output, admission_path, admission_sha, revision, launch_started_unix, deadline_unix):
    output.mkdir(parents=True, exist_ok=False)
    start = PROCESS_STARTED
    report = {
        "status": "failed",
        "config": CONFIG,
        "source_revision": revision,
        "admission_sha256": admission_sha,
        "launch_started_unix": launch_started_unix,
        "deadline_unix": deadline_unix,
        "pid": os.getpid(),
        "updates": [],
        "actions": [],
    }

    def guard():
        if time.monotonic() - start > CONFIG["seconds"] or time.time() > deadline_unix:
            raise TimeoutError("fixed tiny development budget exhausted")
        if _process_memory()["peak_working_set_mib"] > CONFIG["parent_mib"]:
            raise MemoryError("fixed parent memory budget exhausted")

    try:
        assert file_hash(admission_path) == admission_sha
        admission = json.loads(admission_path.read_bytes())
        assert all(
            os.environ.get(k) == "1"
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        )
        assert os.environ.get("HUMANBRAIN_DEVICE") == "cpu"
        assert 0 < deadline_unix - launch_started_unix <= CONFIG["seconds"]
        assert launch_started_unix <= time.time() < deadline_unix
        actual_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        assert actual_revision == revision == admission["source_revision"]
        assert admission["status"] == "admitted" and admission["config"] == CONFIG
        report["source_sha256"] = sources()
        assert admission["source_sha256"] == report["source_sha256"]
        report["packages"] = {name: version(name) for name in ("torch", "numpy")}
        assert admission["packages"] == report["packages"]
        import torch

        from benchmarks.native_scaling import _process_memory
        from humanbrain.native import loaded_library_identity

        report["native"] = loaded_library_identity().to_record()
        assert report["native"] == admission["native"]
        guard()
        train, checks = roster()
        schedule = teaching_schedule(train)
        report["teaching_schedule_sha256"] = hashlib.sha256(
            json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert report["teaching_schedule_sha256"] == admission["teaching_schedule_sha256"]
        write_json(
            output / "offline_teacher_roster.json",
            {
                "training_vocabulary": train,
                "checks": checks,
                "schedule": schedule,
                "teaching_schedule_sha256": report["teaching_schedule_sha256"],
            },
        )
        models = {a: network("bypass" if a == "controller_bypass" else "cell") for a in ARMS}
        initial_weights = {a: detached_state(m.state_dict()) for a, m in models.items()}
        assert all(
            same_state(initial_weights["full"], initial_weights[a])
            for a in ("frozen", "writes_disabled")
        )
        for component in ("embedding", "query_cell", "decoder"):
            assert same_state(
                getattr(models["full"], component).state_dict(),
                getattr(models["controller_bypass"], component).state_dict(),
            )
        torch.save(initial_weights, output / "initial_weights.pt")
        report["initial_weights_sha256"] = file_hash(output / "initial_weights.pt")
        report["slow_parameter_counts"] = {
            a: sum(p.numel() for p in m.parameters()) for a, m in models.items()
        }
        report["slow_parameter_bytes"] = {
            a: sum(p.numel() * p.element_size() for p in m.parameters()) for a, m in models.items()
        }
        optimizers = {
            a: torch.optim.Adam(m.parameters(), lr=CONFIG["learning_rate"])
            for a, m in models.items()
            if a != "frozen"
        }
        # Four fixed supervised episodes per step; each contains two complete documents.
        for step, episodes in enumerate(schedule):
            batch = [
                (
                    [bytes.fromhex(x) for x in e["document_hex"]],
                    bytes.fromhex(e["query_hex"]),
                    bytes.fromhex(e["target_hex"]),
                )
                for e in episodes
            ]
            for arm, optimizer in optimizers.items():
                guard()
                model = models[arm]
                optimizer.zero_grad(set_to_none=True)
                losses = [
                    model.decoder.loss(
                        model.context(
                            query, model.ingest(docs, writes_enabled=arm != "writes_disabled")
                        ),
                        [target],
                    )
                    for docs, query, target in batch
                ]
                loss = torch.stack(losses).mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["clip"], error_if_nonfinite=True
                )
                optimizer.step()
                report["updates"].append(
                    {
                        "arm": arm,
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "norm": float(norm),
                    }
                )
        saved = {}
        assert same_state(initial_weights["frozen"], models["frozen"].state_dict())
        report["optimizer_tensor_bytes"] = {
            a: tensor_bytes(o.state_dict()) for a, o in optimizers.items()
        }
        with torch.no_grad():
            for arm, model in models.items():
                model.eval()
                memories = {}
                for case in checks:
                    memory_id = case["memory_id"]
                    if memory_id in memories:
                        continue
                    base = case["base_memory_id"]
                    state = model.ingest(
                        [
                            bytes.fromhex(x)
                            for x in (
                                case["additional_document_hex"] if base else case["document_hex"]
                            )
                        ],
                        state=model.memory.restore_state(memories[base]) if base else None,
                        writes_enabled=arm != "writes_disabled",
                    )
                    memories[memory_id] = model.memory.purge_writer(state).snapshot()
                    assert not memories[memory_id]["writer"].count_nonzero()
                assert len(memories) == 8
                saved[arm] = {
                    "weights": detached_state(model.state_dict()),
                    "memories": detached_state(memories),
                }
        isolated = output / "source-free"
        isolated.mkdir()
        checkpoint = isolated / "source_free.pt"
        torch.save(saved, checkpoint)
        report["source_free_tensor_bytes"] = tensor_bytes(saved)
        requests = isolated / "query_only.json"
        write_json(
            requests,
            {
                "source_sha256": sources(),
                "cases": [
                    {k: c[k] for k in ("id", "memory_id", "query_hex", "delay_reads")}
                    for c in checks
                ],
            },
        )
        del (
            saved,
            initial_weights,
            models,
            optimizers,
            train,
            schedule,
            batch,
            losses,
            model,
            optimizer,
        )
        guard()
        generated_path = isolated / "generated.json"
        command = [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "worker",
            "--checkpoint",
            str(checkpoint),
            "--requests",
            str(requests),
            "--output",
            str(generated_path),
            "--checkpoint-sha",
            file_hash(checkpoint),
            "--requests-sha",
            file_hash(requests),
        ]
        child = subprocess.run(
            command,
            cwd=isolated,
            capture_output=True,
            timeout=max(
                0.001,
                min(CONFIG["seconds"] - (time.monotonic() - start), deadline_unix - time.time()),
            ),
        )
        (output / "worker.stdout").write_bytes(child.stdout)
        (output / "worker.stderr").write_bytes(child.stderr)
        report["worker"] = {"command": command, "returncode": child.returncode}
        if child.returncode:
            raise RuntimeError("source-free reload/generation process failed")
        generated = json.loads(generated_path.read_bytes())
        report["worker"]["memory"] = generated["memory"]
        report["worker"]["elapsed_seconds"] = generated["elapsed_seconds"]
        report["worker"]["pid"] = generated["worker_pid"]
        report["parent_plus_worker_peak_mib_upper_sum"] = (
            _process_memory()["peak_working_set_mib"] + generated["memory"]["peak_working_set_mib"]
        )
        if report["parent_plus_worker_peak_mib_upper_sum"] > CONFIG["parent_mib"]:
            raise MemoryError("conservative parent plus worker peak exceeds budget")
        assert [(r["arm"], r["id"], r["memory_id"]) for r in generated["outputs"]] == [
            (a, c["id"], c["memory_id"]) for a in ARMS for c in checks
        ]
        from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
        from experiments.r4_integrated.utf8_action_bridge import Case, execute_utf8

        motor = ContextualOccurrenceMotor()
        demonstrations = []
        for case in (
            Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
            Case("motor-b", "bora", "ready", "motor_beta", "warm"),
        ):
            guard()
            before, after = case.files(), case.files(corrected=True)
            execution = execute_utf8(case, after, before, output / case.name)
            assert execution["passed"]
            motor.learn(f"Configure {case.key}".encode(), before, after)
            demonstrations.append(execution)
        report["motor_demonstrations"] = demonstrations
        for row in generated["outputs"]:
            guard()
            case = checks[row["id"]]
            action = actuate(row, row["id"], motor, output / f"action-{row['arm']}-{row['id']:02d}")
            correct = (
                action["emitted"]
                and row["generated_hex"] == case["expected_hex"]
                and action["lossless_generated_execution"]
            )
            report["actions"].append(
                {
                    "arm": row["arm"],
                    "id": row["id"],
                    "memory_id": row["memory_id"],
                    "group": case["group"],
                    "correct": correct,
                    **action,
                }
            )
        report["scores"] = {
            a: {
                g: sum(r["correct"] for r in report["actions"] if r["arm"] == a and r["group"] == g)
                for g in ("fit", "unseen_value", "delayed", "interference")
            }
            for a in ARMS
        }
        report["tiny_fit_passed"] = report["scores"]["full"]["fit"] == 8
        report["native_after"] = loaded_library_identity().to_record()
        assert report["native_after"] == report["native"]
        assert len(report["updates"]) == 768 and len(report["actions"]) == 80
        report["status"] = "completed"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        try:
            if "_process_memory" in locals():
                report["memory"] = _process_memory()
            if "source_sha256" in report:
                report["source_sha256_after"] = sources()
                if report["source_sha256_after"] != report["source_sha256"]:
                    report["status"] = "failed"
            if time.time() > deadline_unix:
                report["status"] = "failed"
            write_json(output / "report.json", report)
            completion = {
                "status": "passed" if report["status"] == "completed" else "failed",
                "report_sha256": file_hash(output / "report.json"),
                "source_revision": revision,
                "admission_sha256": admission_sha,
                "elapsed_seconds": time.monotonic() - start,
                "outer_elapsed_seconds": time.time() - launch_started_unix,
                "artifacts": {
                    p.relative_to(output).as_posix(): file_hash(p)
                    for p in (
                        *output.glob("source-free/*"),
                        output / "initial_weights.pt",
                        output / "offline_teacher_roster.json",
                    )
                    if p.is_file()
                },
            }
            if "_process_memory" in locals():
                completion["memory"] = _process_memory()
                completion["peak_sum_upper_mib"] = completion["memory"][
                    "peak_working_set_mib"
                ] + report.get("worker", {}).get("memory", {}).get("peak_working_set_mib", 0)
                if completion["peak_sum_upper_mib"] > CONFIG["parent_mib"]:
                    completion["status"] = "failed"
            completion["elapsed_seconds"] = time.monotonic() - start
            completion["outer_elapsed_seconds"] = time.time() - launch_started_unix
            if time.time() > deadline_unix:
                completion["status"] = "failed"
            write_json(output / "completion.json", completion)
            if completion["status"] != "passed":
                report["status"] = "failed"
        except BaseException:
            write_json(
                output / "finalization-error.json",
                {
                    "original_error": report.get("error"),
                    "finalization_error": traceback.format_exc(),
                },
            )
            report["status"] = "failed"
    return report["status"] == "completed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha")
    parser.add_argument("--revision")
    parser.add_argument(
        "--launch-started-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_START_UNIX")
    )
    parser.add_argument(
        "--deadline-unix", type=float, default=os.environ.get("HUMANBRAIN_RUN_DEADLINE_UNIX")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--checkpoint-sha")
    parser.add_argument("--requests-sha")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.checkpoint, args.requests, args.output, args.checkpoint_sha, args.requests_sha)
    else:
        raise SystemExit(
            0
            if run(
                args.output,
                args.admission,
                args.admission_sha,
                args.revision,
                args.launch_started_unix,
                args.deadline_unix,
            )
            else 1
        )
