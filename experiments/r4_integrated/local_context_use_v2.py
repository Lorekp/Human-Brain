"""Fixed eight-arm direct-context local hidden-feedback comparison."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

START = time.perf_counter()

import torch  # noqa: E402

from experiments.r4_integrated import local_context_use_v1 as previous  # noqa: E402
from humanbrain.byte_sequence_decoder import ByteSequenceDecoder  # noqa: E402

ROOT = previous.ROOT
# Only identity/mode inventory changes; inherited helper numeric contracts are fixed.
CONFIG = {**previous.CONFIG, "schema": "local-context-use-v2"}
ARM_SPECS = {f"{mode}_{condition}": (mode, condition)
             for mode in ("random", "symmetric") for condition in previous.ARMS}
ARMS = tuple(ARM_SPECS)
CONFIG["arm_specs"] = ARM_SPECS
TRAIN, HELDOUT = previous.TRAIN, previous.HELDOUT
context, schedule, final_context = previous.context, previous.schedule, previous.final_context
file_hash, digest, write_json = previous.file_hash, previous.digest, previous.write_json
lesson, generation = previous.lesson, previous.generation
same_state, state_hash, summarize = previous.same_state, previous.state_hash, previous.summarize


def sources():
    return {**previous.sources(), **{path.relative_to(ROOT).as_posix(): file_hash(path)
                                    for path in (Path(__file__), Path(__file__).with_suffix(".md"))}}


def new_decoder(mode):
    return ByteSequenceDecoder(
        CONFIG["context_size"], CONFIG["hidden_size"], CONFIG["embedding_size"],
        seed=CONFIG["decoder_seed"], local_learning=True, hidden_feedback_mode=mode,
    )


def matched_initial_tensors(states):
    baseline = states["random_correct"]
    for arm, state in states.items():
        if state.keys() != baseline.keys():
            return False
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.equal(value, baseline[key]):
                    return False
            elif key == "_extra_state":
                expected = {**baseline[key], "hidden_feedback_mode": ARM_SPECS[arm][0]}
                if value != expected:
                    return False
            elif value != baseline[key]:
                return False
    return True


def qualifies(scores):
    candidate = scores["symmetric_correct"]
    controls = ("symmetric_zero", "symmetric_shuffled", "symmetric_frozen", "random_correct")
    return candidate == {"train": 12, "heldout": 4} and all(
        candidate["heldout"] > scores[arm]["heldout"] for arm in controls
    )


def run(output, revision, sources_sha):
    manifest = sources()
    current_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if revision != current_revision or sources_sha != digest(manifest):
        raise ValueError("revision or source manifest mismatch")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    clock = time.get_clock_info("perf_counter")
    deadline = float(os.environ["HUMANBRAIN_RUN_DEADLINE_UNIX"])

    def guard():
        if time.perf_counter() - START > CONFIG["seconds"] or time.time() > deadline:
            raise TimeoutError("diagnostic time ceiling exhausted")

    roster = schedule()
    write_json(output / "schedule.json", {"train": TRAIN, "heldout": HELDOUT, "rows": roster,
               "context_vectors": {pair: context(pair).tolist() for pair in TRAIN + HELDOUT}})
    admission = {"config": CONFIG, "sources": manifest, "sources_sha256": sources_sha,
                 "revision": revision, "pid": os.getpid(),
                 "versions": {name: version(name) for name in ("torch", "numpy")},
                 "memory_monitor": "external launcher's owned process-tree peak working set",
                 "duration_clock": {"name": "perf_counter", "implementation": clock.implementation,
                                    "resolution_seconds": clock.resolution,
                                    "monotonic": clock.monotonic, "adjustable": clock.adjustable},
                 "deadline_unix": deadline, "schedule_sha256": file_hash(output / "schedule.json")}
    write_json(output / "admission.json", admission)
    guard()
    models = {arm: new_decoder(ARM_SPECS[arm][0]) for arm in ARMS}
    initial = {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()}
    if not matched_initial_tensors(initial):
        raise AssertionError("initial arm tensors or declared metadata differ")
    torch.save(initial, output / "initial.pt")
    records = []
    # No adaptive ordering, scores, early success stop, or extra generation.
    with (output / "lessons.jsonl").open("x", encoding="utf-8") as stream:
        for row in roster:
            for arm, model in models.items():
                mode, condition = ARM_SPECS[arm]
                guard()
                receipt = lesson(model, context(row["contexts"][condition]), row["target"],
                                 enabled=condition != "frozen")
                record = {"index": row["index"], "arm": arm, "target": row["target"],
                          "context": row["contexts"][condition], "feedback_mode": mode,
                          "condition": condition, **receipt}
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            if (row["index"] + 1) % 128 == 0:
                stream.flush()
                print(json.dumps({"completed_lessons": len(records),
                                  "elapsed_seconds": time.perf_counter() - START}), flush=True)
    final = {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()}
    torch.save(final, output / "final.pt")
    checks = []
    for arm, model in models.items():
        mode, condition = ARM_SPECS[arm]
        for pair in TRAIN + HELDOUT:
            guard()
            before = state_hash(model)
            generated = generation(model, final_context(condition, pair))
            checks.append({"arm": arm, "target": pair, "group": "train" if pair in TRAIN else "heldout",
                           "context": None if condition == "zero" else pair, **generated,
                           "feedback_mode": mode, "condition": condition,
                           "before_sha256": before, "after_sha256": state_hash(model),
                           "exact": generated["hex"] == pair.encode().hex() and generated["stop_reason"] == "eos"})
        if not same_state(final[arm], model.state_dict()):
            raise AssertionError("final generation mutated decoder")
    for arm in ("random_frozen", "symmetric_frozen"):
        if not same_state(initial[arm], final[arm]):
            raise AssertionError("frozen decoder changed")
    sources_after = sources()
    revision_after = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if manifest != sources_after or revision != revision_after:
        raise ValueError("sources changed during diagnostic")
    scores = {arm: {group: sum(row["exact"] for row in checks if row["arm"] == arm and row["group"] == group)
                    for group in ("train", "heldout")} for arm in ARMS}
    report = {
        **admission, "sources_after": sources_after, "revision_after": revision_after,
        "status": "completed", "checks": checks, "scores": scores,
        "context_use_positive": qualifies(scores),
        "acquisition": {}, "counts": {
            "first_attempts": len(records), "final_generations": len(checks),
            "byte_offers": sum(row["byte_offers"] for row in records),
            "enabled_steps": sum(row["enabled_steps"] for row in records),
            "changed_steps": sum(row["changed_steps"] for row in records),
            "generation_decisions": sum(row["first"]["decisions"] for row in records)
            + sum(row["decisions"] for row in checks),
        },
        "artifact_sha256": {name: file_hash(output / name) for name in
                            ("schedule.json", "admission.json", "initial.pt", "final.pt", "lessons.jsonl")},
        "model_tensor_bytes_per_arm": {
            arm: sum(value.numel() * value.element_size() for value in state.values()
                     if isinstance(value, torch.Tensor)) for arm, state in final.items()
        },
        "artifact_bytes": {name: (output / name).stat().st_size for name in
                           ("schedule.json", "admission.json", "initial.pt", "final.pt", "lessons.jsonl")},
    }
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        report["acquisition"][arm] = {"all": summarize(rows), "first128": summarize(rows[:128]),
                                      "last128": summarize(rows[-128:])}
    guard()
    report["elapsed_seconds"] = time.perf_counter() - START
    write_json(output / "report.json", report)
    guard()
    write_json(output / "completion.json", {"status": "completed", "report_sha256": file_hash(output / "report.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sources-sha", required=True)
    args = parser.parse_args()
    run(args.output, args.revision, args.sources_sha)
