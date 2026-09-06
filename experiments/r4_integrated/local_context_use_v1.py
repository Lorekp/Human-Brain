"""One fixed supervised context-use diagnostic; no document or action claims."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

START = time.perf_counter()

import torch  # noqa: E402

from humanbrain.byte_sequence_decoder import (  # noqa: E402
    BYTE_BOS,
    BYTE_EOS,
    ByteSequenceDecoder,
)

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("correct", "zero", "shuffled", "frozen")
HELDOUT = ("ab", "bc", "cd", "da")
TRAIN = tuple(a + b for a in "abcd" for b in "abcd" if a + b not in HELDOUT)
CONFIG = {
    "schema": "local-context-use-v1", "seed": 733, "decoder_seed": 735, "shuffle_seed": 734,
    "lessons_per_arm": 384, "cycles": 32, "learning_rate": 0.01,
    "context_size": 96, "embedding_size": 32, "hidden_size": 64,
    "max_bytes": 2, "seconds": 120, "max_mib": 2048,
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sources():
    paths = (
        "experiments/r4_integrated/local_context_use_v1.py",
        "experiments/r4_integrated/local_context_use_v1.md",
        "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1",
        "src/humanbrain/__init__.py", "src/humanbrain/byte_sequence_decoder.py",
        "src/humanbrain/plastic_memory.py",
        "src/humanbrain/local_memory_learner.py",
        "experiments/r4_integrated/__init__.py",
    )
    return {path: file_hash(ROOT / path) for path in paths}


def context(pair):
    """Authored rates, two active units in two factor populations; no learned encoder."""
    value = torch.zeros(1, CONFIG["context_size"])
    if pair is not None:
        if pair not in TRAIN + HELDOUT:
            raise ValueError("context must be a declared pair")
        value[0, "abcd".index(pair[0])] = 1
        value[0, 4 + "abcd".index(pair[1])] = 1
    return value


def schedule():
    """Commit every context assignment before any predictions or teacher corrections."""
    rng = random.Random(CONFIG["shuffle_seed"])
    rows = []
    for cycle in range(CONFIG["cycles"]):
        shuffled = list(TRAIN)
        rng.shuffle(shuffled)
        for index, pair in enumerate(TRAIN):
            rows.append({
                "index": len(rows), "cycle": cycle, "target": pair,
                "contexts": {"correct": pair, "zero": None,
                             "shuffled": shuffled[index], "frozen": pair},
            })
    return rows


def new_decoder():
    return ByteSequenceDecoder(
        CONFIG["context_size"], CONFIG["hidden_size"], CONFIG["embedding_size"],
        seed=CONFIG["decoder_seed"], local_learning=True,
    )


def same_state(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(value, right[name]) if isinstance(value, torch.Tensor)
        else value == right[name] for name, value in left.items()
    )


def state_hash(model):
    hashed = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        hashed.update(name.encode())
        if isinstance(value, torch.Tensor):
            hashed.update(str((str(value.dtype), tuple(value.shape))).encode())
            hashed.update(value.detach().contiguous().numpy().tobytes())
        else:
            hashed.update(json.dumps(value, sort_keys=True).encode())
    return hashed.hexdigest()


def generation(model, value):
    output = model.generate(value, CONFIG["max_bytes"])[0]
    return {"hex": output.data.hex(), "stop_reason": output.stop_reason,
            "decisions": len(output.data) + 1}


@torch.no_grad()
def lesson(model, value, target, *, enabled):
    # Context is already fixed. First generation precedes all teacher byte offers.
    before = state_hash(model)
    first = generation(model, value)
    after_first = state_hash(model)
    if before != after_first:
        raise AssertionError("first generation mutated decoder")
    target_bytes = target.encode("ascii")
    first["exact"] = first["hex"] == target_bytes.hex() and first["stop_reason"] == "eos"
    state, previous = None, BYTE_BOS
    losses, offers, updates = [], 0, 0
    for token in (*target_bytes, BYTE_EOS):
        step = model.local_step(value, torch.tensor([previous]), state)
        receipt = model.teach_local(
            step, torch.tensor([token]), CONFIG["learning_rate"], learning_enabled=enabled,
        )
        losses.append(receipt.loss)
        offers += receipt.teacher_count
        updates += int(receipt.updated)
        state, previous = step.state, token
    return {"first": first, "teacher_losses": losses, "mean_loss": sum(losses) / 3,
            "byte_offers": offers, "enabled_steps": 3 if enabled else 0,
            "changed_steps": updates, "before_sha256": before,
            "after_first_sha256": after_first, "after_teacher_sha256": state_hash(model)}


def final_context(arm, pair):
    return context(None if arm == "zero" else pair)


def summarize(rows):
    return {"lessons": len(rows), "first_exact": sum(row["first"]["exact"] for row in rows),
            "mean_teacher_loss": sum(row["mean_loss"] for row in rows) / len(rows),
            "first_stop_reasons": dict(Counter(row["first"]["stop_reason"] for row in rows))}


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
    models = {arm: new_decoder() for arm in ARMS}
    initial = {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()}
    if not all(same_state(initial["correct"], state) for state in initial.values()):
        raise AssertionError("initial arm states differ")
    torch.save(initial, output / "initial.pt")
    records = []
    # No adaptive ordering, scores, early success stop, or extra generation.
    with (output / "lessons.jsonl").open("x", encoding="utf-8") as stream:
        for row in roster:
            for arm, model in models.items():
                guard()
                receipt = lesson(model, context(row["contexts"][arm]), row["target"],
                                 enabled=arm != "frozen")
                record = {"index": row["index"], "arm": arm, "target": row["target"],
                          "context": row["contexts"][arm], **receipt}
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
        for pair in TRAIN + HELDOUT:
            guard()
            before = state_hash(model)
            generated = generation(model, final_context(arm, pair))
            checks.append({"arm": arm, "target": pair, "group": "train" if pair in TRAIN else "heldout",
                           "context": None if arm == "zero" else pair, **generated,
                           "before_sha256": before, "after_sha256": state_hash(model),
                           "exact": generated["hex"] == pair.encode().hex() and generated["stop_reason"] == "eos"})
        if not same_state(final[arm], model.state_dict()):
            raise AssertionError("final generation mutated decoder")
    if not same_state(initial["frozen"], final["frozen"]):
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
        "context_use_positive": scores["correct"] == {"train": 12, "heldout": 4}
        and all(scores["correct"]["heldout"] > scores[arm]["heldout"] for arm in ARMS[1:]),
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
