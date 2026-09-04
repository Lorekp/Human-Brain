"""Public KS2 Q6 development diagnostic; no reserved evaluation or answer training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_json(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    names = (
        "neutral_neuronal_document_learner.py", "opaque_document_event_tape.py",
        "sparse_neuronal_role_filler.py", "cells.py", "resting_glif_race.py",
    )
    return {name: digest(ROOT / "src/humanbrain" / name) for name in names}


def validate_inputs(*, evaluator: bool) -> dict:
    manifest = read_json("manifest.json")
    names = ["passage.txt", "questions.json"]
    if evaluator:
        names.append("rubric.json")
    for name in names:
        if digest(HERE / name) != manifest["fixture_sha256"][name]:
            raise ValueError(f"Fixture changed without a manifest update: {name}")
    return manifest


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=True), flush=True)


def worker(arm: str) -> None:
    # Rubric bytes and expected answers are never read in this process.
    validate_inputs(evaluator=False)
    tracemalloc.start()
    from humanbrain.neutral_neuronal_document_learner import start
    from humanbrain.opaque_document_event_tape import OpaqueDocument

    content = (HERE / "passage.txt").read_bytes()
    questions = read_json("questions.json")
    document = OpaqueDocument(b"ks2-2024-passage-1", b"sta-24-8811", content)
    network = start()
    emit({
        "event": "seed", "arm": arm,
        "checkpoint_sha256": network.checkpoint_sha256,
        "training_summary": asdict(network.training_summary),
        "passage_bytes": len(content),
        "acquisition_exposures": int(arm == "UPDATE"),
    })
    if arm == "UPDATE":
        emit({"event": "acquisition_started"})
        begun = time.perf_counter()
        network = network.consolidate((document,))
        emit({
            "event": "acquisition_completed",
            "elapsed_seconds": time.perf_counter() - begun,
            "checkpoint_sha256": network.checkpoint_sha256,
            "training_summary": asdict(network.training_summary),
        })
    context = questions["shared_stem"]
    checkpoint = network.checkpoint_sha256
    for part in questions["parts"]:
        context += "\n\n" + part["text"]
        query = context.encode("utf-8")
        emit({"event": "query_started", "item": part["id"]})
        begun = time.perf_counter()
        decision = network.respond(query, documents=(document,))
        elapsed = time.perf_counter() - begun
        network.authenticate()
        if network.checkpoint_sha256 != checkpoint:
            raise RuntimeError("Inference modified the checkpoint")
        emit({
            "event": "answer", "item": part["id"],
            "query_sha256": sha256(query).hexdigest(),
            "query_bytes": len(query),
            "emitted_hex": None if decision.emitted is None else decision.emitted.hex(),
            "status": decision.status, "stopped": decision.stopped,
            "stop_stage": decision.stop_stage, "stop_reason": decision.stop_reason,
            "checkpoint_sha256": checkpoint,
            "decision_sha256": decision.sha256,
            "support_path_count": decision.support_path_count,
            "physical_cell_steps": decision.inference_physical_cell_steps,
            "active_synapses": decision.inference_active_synapses,
            "elapsed_seconds": elapsed,
            "python_traced_peak_bytes": tracemalloc.get_traced_memory()[1],
        })
    emit({"event": "completed", "python_traced_peak_bytes": tracemalloc.get_traced_memory()[1]})


def normalized(value: str) -> str:
    return " ".join(value.strip().casefold().removesuffix(".").split())


def score(events: list[dict], rubric: dict, terminal: str) -> dict:
    answers = {event["item"]: event for event in events if event["event"] == "answer"}
    scored = {}
    for item, rule in rubric["items"].items():
        answer = answers.get(item)
        if answer is None:
            scored[item] = {"marks": 0, "outcome": f"unanswered_{terminal}"}
        elif answer["stopped"] or answer["emitted_hex"] is None:
            scored[item] = {"marks": 0, "outcome": "STOP" if answer["stopped"] else "missing"}
        else:
            raw = bytes.fromhex(answer["emitted_hex"])
            try:
                response = raw.decode("utf-8")
            except UnicodeDecodeError:
                scored[item] = {"marks": 0, "outcome": "invalid_utf8"}
                continue
            if normalized(response) in rule["automatic_accept"]:
                scored[item] = {"marks": 1, "outcome": "accepted"}
            else:
                scored[item] = {"marks": None, "outcome": "pending_official_rubric_human_scoring"}
    marks = [row["marks"] for row in scored.values()]
    return {
        "items": scored,
        "earned_marks": sum(marks) if all(mark is not None for mark in marks) else None,
        "confirmed_marks": sum(mark for mark in marks if mark is not None),
        "pending_items": sum(mark is None for mark in marks),
        "available_marks": sum(row["available_marks"] for row in rubric["items"].values()),
    }


def run_arm(arm: str, seconds: float, rubric: dict) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", arm]
    begun = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            timeout=seconds, check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        terminal = "completed" if completed.returncode == 0 else "worker_error"
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        # subprocess.run kills and waits for the child before raising.
        stdout, stderr = error.stdout or b"", error.stderr or b""
        terminal, returncode = "timeout", None
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            stderr += "\nNon-JSON worker output: " + line
    if terminal == "completed" and (not events or events[-1].get("event") != "completed"):
        terminal = "incomplete_worker_output"
    return {
        "arm": arm, "terminal": terminal, "returncode": returncode,
        "elapsed_seconds": time.perf_counter() - begun,
        "worker_timeout_seconds": seconds, "events": events, "stderr": stderr,
        "scoring": score(events, rubric, terminal),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", choices=["NO_UPDATE", "UPDATE"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    manifest = validate_inputs(evaluator=True)
    questions, rubric = read_json("questions.json"), read_json("rubric.json")
    assert [part["id"] for part in questions["parts"]] == list(rubric["items"])
    assert len(questions["parts"]) == 2
    if args.check:
        print("PASS: fixture hashes, item identities, and rubric mapping")
        return 0
    if args.output is None or args.output.exists() or args.timeout_seconds <= 0:
        parser.error("Supply a fresh --output path and positive --timeout-seconds")
    result = {
        "protocol": "ks2-2024-streaky-squeaky-q6-public-development-v1",
        "condition": "text-only passage-visible; Q6a/Q6b diagnostic subset",
        "manifest": manifest,
        "runner_sha256": digest(Path(__file__)),
        "source_sha256": source_hashes(),
        "python": sys.version,
        "memory_metric": (
            "tracemalloc Python allocations; excludes native allocations and total "
            "process RSS; tracing overhead included in runtime"
        ),
        "arms": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a fresh artifact before running. Never overwrite prior evidence.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    for arm in ("NO_UPDATE", "UPDATE"):
        result["arms"].append(run_arm(arm, args.timeout_seconds, rubric))
        result["source_sha256_after"] = source_hashes()
        result["source_stable"] = (
            result.get("source_stable", True)
            and result["source_sha256"] == result["source_sha256_after"]
        )
        result["evidence_status"] = (
            "public_development_observation"
            if result["source_stable"] else "invalid_source_changed_during_run"
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        current = result["arms"][-1]
        print(f"{arm}: {current['terminal']}; {current['scoring']}", flush=True)
    print(f"Result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
