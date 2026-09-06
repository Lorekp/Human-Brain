"""One fixed six-arm exposure/order bracket with a local learnability reference."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

START = time.perf_counter()

import torch  # noqa: E402

from experiments.r4_integrated import local_context_use_v1 as previous  # noqa: E402
from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS  # noqa: E402
from humanbrain.phase_context_perceptron import PhaseContextPerceptron  # noqa: E402

ROOT = previous.ROOT
CONFIG = {**previous.CONFIG, "schema": "local-context-learnability-v1",
          "cycles": 64, "lessons_per_arm": 768, "checkpoints": [384, 768]}
ARMS = ("decoder_lexical", "decoder_permuted", "reference_lexical", "reference_permuted",
        "reference_zero", "reference_frozen")
TRAIN, HELDOUT = previous.TRAIN, previous.HELDOUT
context, lesson, generation = previous.context, previous.lesson, previous.generation
file_hash, digest, write_json = previous.file_hash, previous.digest, previous.write_json
same_state, state_hash = previous.same_state, previous.state_hash


def sources():
    names = ("experiments/r4_integrated/local_context_learnability_v1.py",
             "experiments/r4_integrated/local_context_learnability_v1.md",
             "src/humanbrain/phase_context_perceptron.py",
             "docs/research/local-decoder-learnability-review.md")
    return {**previous.sources(), **{name: file_hash(ROOT / name) for name in names}}


def schedule():
    rng = random.Random(CONFIG["shuffle_seed"])
    rows = []
    for cycle in range(CONFIG["cycles"]):
        permuted = list(TRAIN)
        rng.shuffle(permuted)
        for position, target in enumerate(TRAIN):
            rows.append({"index": len(rows), "cycle": cycle, "lexical_target": target,
                         "permuted_target": permuted[position]})
    return rows


def assignment(arm, row):
    target = row["permuted_target"] if arm.endswith("_permuted") else row["lexical_target"]
    return target, None if arm == "reference_zero" else target


def new_model(arm):
    if arm not in ARMS:
        raise ValueError("unknown bracket arm")
    return previous.new_decoder() if arm.startswith("decoder_") else PhaseContextPerceptron()


def summarize(rows):
    result = previous.summarize(rows)
    result["loss_kind"] = rows[0]["loss_kind"]
    # The generic v1 receipt field is retained, but comparisons must use this label.
    result["mean_loss_by_byte1_byte2_eos"] = [
        sum(row["teacher_losses"][i] for row in rows) / len(rows) for i in range(3)
    ]
    return result


@torch.no_grad()
def reference_lesson(model, value, target, *, enabled):
    before = state_hash(model)
    first = generation(model, value)
    after_first = state_hash(model)
    if before != after_first:
        raise AssertionError("reference first generation mutated state")
    target_bytes = target.encode("ascii")
    first["exact"] = first["hex"] == target_bytes.hex() and first["stop_reason"] == "eos"
    state, previous_token = None, BYTE_BOS
    teachers = []
    for token in (*target_bytes, BYTE_EOS):
        step = model.local_step(value, torch.tensor([previous_token]), state)
        receipt = model.teach_local(
            step, torch.tensor([token]), CONFIG["learning_rate"], learning_enabled=enabled,
        )
        teachers.append({"winner": receipt.winner, "target": receipt.target,
                         "phase": receipt.phase, "mistake": int(receipt.loss),
                         "updated": receipt.updated,
                         "active_features": step.features[0].nonzero().flatten().tolist(),
                         "winner_score": int(step.scores[0, receipt.winner]),
                         "target_score": int(step.scores[0, receipt.target])})
        state, previous_token = step.state, token
    losses = [row["mistake"] for row in teachers]
    return {"first": first, "teacher_losses": losses, "mean_loss": sum(losses) / 3,
            "byte_offers": len(teachers), "enabled_steps": 3 if enabled else 0,
            "changed_steps": sum(row["updated"] for row in teachers), "teachers": teachers,
            "before_sha256": before, "after_first_sha256": after_first,
            "after_teacher_sha256": state_hash(model)}


def certificate(rows, train_score):
    mistakes = [sum(row["teacher_losses"]) for row in rows]
    per_cycle = [sum(mistakes[index:index + 12]) for index in range(0, len(rows), 12)]
    zero_cycles = [index + 1 for index, count in enumerate(per_cycle) if count == 0]
    total = sum(mistakes)
    return {"teacher_mistakes": total, "mistakes_per_cycle": per_cycle,
            "first_zero_mistake_cycle": zero_cycles[0] if zero_cycles else None,
            "passed": total <= 54 and bool(zero_cycles) and zero_cycles[0] <= 55
            and train_score == 12}


def useful(scores):
    return {arm: scores[arm] == {"train": 12, "heldout": 4}
            and all(scores[arm]["heldout"] > scores[control]["heldout"]
                    for control in ("reference_zero", "reference_frozen"))
            for arm in ARMS[:4]}


def checkpoint(models, output, count, guard):
    states = {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()}
    torch.save(states, output / f"checkpoint-{count}.pt")
    checks = []
    for arm, model in models.items():
        for pair in TRAIN + HELDOUT:
            guard()
            supplied = None if arm == "reference_zero" else pair
            before = state_hash(model)
            emitted = generation(model, context(supplied))
            checks.append({"arm": arm, "target": pair, "context": supplied,
                           "group": "train" if pair in TRAIN else "heldout", **emitted,
                           "before_sha256": before, "after_sha256": state_hash(model),
                           "exact": emitted["hex"] == pair.encode().hex()
                           and emitted["stop_reason"] == "eos"})
        if not same_state(states[arm], model.state_dict()):
            raise AssertionError("checkpoint probes changed model state")
    scores = {arm: {group: sum(row["exact"] for row in checks
                              if row["arm"] == arm and row["group"] == group)
                    for group in ("train", "heldout")} for arm in ARMS}
    return {"lessons_per_arm": count, "checks": checks, "scores": scores}


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
            raise TimeoutError("bracket time ceiling exhausted")

    roster = schedule()
    write_json(output / "schedule.json", {"train": TRAIN, "heldout": HELDOUT, "rows": roster,
               "context_vectors": {pair: context(pair).tolist() for pair in TRAIN + HELDOUT}})
    admission = {"config": CONFIG, "arms": ARMS, "sources": manifest,
                 "sources_sha256": sources_sha, "revision": revision, "pid": os.getpid(),
                 "versions": {name: version(name) for name in ("torch", "numpy")},
                 "memory_monitor": "external launcher's owned process-tree peak working set",
                 "duration_clock": {"name": "perf_counter", "implementation": clock.implementation,
                                    "resolution_seconds": clock.resolution,
                                    "monotonic": clock.monotonic, "adjustable": clock.adjustable},
                 "deadline_unix": deadline, "schedule_sha256": file_hash(output / "schedule.json")}
    write_json(output / "admission.json", admission)
    guard()
    models = {arm: new_model(arm) for arm in ARMS}
    initial = {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()}
    if not same_state(initial["decoder_lexical"], initial["decoder_permuted"]) or not all(
        same_state(initial["reference_lexical"], initial[arm]) for arm in ARMS[2:]
    ):
        raise AssertionError("within-architecture initial states differ")
    torch.save(initial, output / "initial.pt")
    records, checkpoints = [], {}
    with (output / "lessons.jsonl").open("x", encoding="utf-8") as stream:
        for row in roster:
            for arm, model in models.items():
                guard()
                target, supplied = assignment(arm, row)
                teach_lesson = lesson if arm.startswith("decoder_") else reference_lesson
                receipt = teach_lesson(model, context(supplied), target, enabled=arm != "reference_frozen")
                record = {"index": row["index"], "arm": arm, "target": target, "context": supplied,
                          "loss_kind": "decoder_cross_entropy" if arm.startswith("decoder_")
                          else "reference_teacher_mistake", **receipt}
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            count = row["index"] + 1
            if count in CONFIG["checkpoints"]:
                stream.flush()
                checkpoints[str(count)] = checkpoint(models, output, count, guard)
                write_json(output / f"checkpoint-{count}.json", checkpoints[str(count)])
            if count % 128 == 0:
                stream.flush()
                print(json.dumps({"completed_lessons": len(records),
                                  "elapsed_seconds": time.perf_counter() - START}), flush=True)
    if not same_state(initial["reference_frozen"], models["reference_frozen"].state_dict()):
        raise AssertionError("frozen reference changed")
    sources_after = sources()
    revision_after = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if manifest != sources_after or revision != revision_after:
        raise ValueError("sources changed during bracket")
    artifacts = ("schedule.json", "admission.json", "initial.pt", "lessons.jsonl",
                 "checkpoint-384.pt", "checkpoint-384.json", "checkpoint-768.pt", "checkpoint-768.json")
    final_scores = checkpoints["768"]["scores"]
    report = {**admission, "sources_after": sources_after, "revision_after": revision_after,
              "status": "completed", "checkpoints": checkpoints,
              "useful_endpoint": useful(final_scores), "certificates": {}, "acquisition": {},
              "counts": {"first_attempts": len(records),
                         "checkpoint_probes": sum(len(entry["checks"]) for entry in checkpoints.values()),
                         "byte_offers": sum(row["byte_offers"] for row in records),
                         "enabled_steps": sum(row["enabled_steps"] for row in records),
                         "changed_steps": sum(row["changed_steps"] for row in records),
                         "generation_decisions": sum(row["first"]["decisions"] for row in records)
                         + sum(row["decisions"] for entry in checkpoints.values() for row in entry["checks"])},
              "artifact_sha256": {name: file_hash(output / name) for name in artifacts},
              "artifact_bytes": {name: (output / name).stat().st_size for name in artifacts},
              "model_tensor_bytes_per_arm": {arm: sum(value.numel() * value.element_size()
                  for value in model.state_dict().values() if isinstance(value, torch.Tensor))
                  for arm, model in models.items()},
              "reference_accounting": models["reference_lexical"].accounting()}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        report["acquisition"][arm] = {"all": summarize(rows), "first128": summarize(rows[:128]),
                                      "last128": summarize(rows[-128:]),
                                      "byte_offers": sum(row["byte_offers"] for row in rows),
                                      "enabled_steps": sum(row["enabled_steps"] for row in rows),
                                      "changed_steps": sum(row["changed_steps"] for row in rows)}
        if arm in ("reference_lexical", "reference_permuted"):
            report["certificates"][arm] = certificate(rows, final_scores[arm]["train"])
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
