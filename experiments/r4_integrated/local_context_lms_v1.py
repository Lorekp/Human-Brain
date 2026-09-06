"""One bounded all-output potential-learning comparison, with saved free-call margins."""
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
from humanbrain.phase_context_lms import PhaseContextLMS  # noqa: E402

ROOT = previous.ROOT
ARMS = ("lexical", "permuted", "zero", "frozen")
TRAIN, HELDOUT = previous.TRAIN, previous.HELDOUT
CONFIG = {"schema": "local-context-lms-v1", "experiment_seed": 733, "shuffle_seed": 734,
          "cycles": 160, "lessons_per_arm": 1920, "context_size": 96, "phases": 3,
          "features": 291, "outputs": 257, "dtype": "float64", "learning_rate": 0.01,
          "max_bytes": 2, "minimum_final_margin_exclusive": 0.10,
          "seconds": 120, "max_mib": 2048}
file_hash, digest, write_json = previous.file_hash, previous.digest, previous.write_json
same_state, state_hash = previous.same_state, previous.state_hash


def sources():
    names = ("experiments/r4_integrated/local_context_lms_v1.py",
             "experiments/r4_integrated/local_context_lms_v1.md",
             "src/humanbrain/phase_context_lms.py",
             "docs/research/local-composition-return-review.md")
    return {**previous.sources(), **{name: file_hash(ROOT / name) for name in names}}


def context(pair):
    return previous.context(pair).double()


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
    target = row["permuted_target"] if arm == "permuted" else row["lexical_target"]
    return target, None if arm == "zero" else target


def sparse(value):
    """All nonzero entries of one vector; every unlisted entry is exactly zero."""
    return [[index, float(value[index])] for index in value.nonzero().flatten().tolist()]


@torch.no_grad()
def generation(model, value):
    """One free clock traversal, recording the same potentials used to choose bytes."""
    output, state, phases, token = bytearray(), None, [], BYTE_BOS
    for position in range(CONFIG["max_bytes"] + 1):
        step = model.local_step(value, torch.tensor([token]), state)
        token = int(step.scores.argmax(-1)[0])
        phases.append({"phase": int(step.features[0, :3].argmax()), "winner": token,
                       "nonzero_scores": sparse(step.scores[0]),
                       "nonzero_features": sparse(step.features[0])})
        state = step.state
        if token == BYTE_EOS or position == CONFIG["max_bytes"]:
            return {"hex": bytes(output).hex(), "stop_reason": "eos" if token == BYTE_EOS else "limit",
                    "decisions": len(phases), "phases": phases}
        output.append(token)
    raise AssertionError("bounded generation must stop")


@torch.no_grad()
def lesson(model, value, target, *, enabled):
    before = state_hash(model)
    first = generation(model, value)
    after_first = state_hash(model)
    if before != after_first:
        raise AssertionError("first generation changed model state")
    target_bytes = target.encode("ascii")
    first["exact"] = first["hex"] == target_bytes.hex() and first["stop_reason"] == "eos"
    teachers, state, previous_token = [], None, BYTE_BOS
    for token in (*target_bytes, BYTE_EOS):
        step = model.local_step(value, torch.tensor([previous_token]), state)
        receipt = model.teach_local(
            step, torch.tensor([token]), CONFIG["learning_rate"], learning_enabled=enabled,
        )
        teachers.append({"phase": receipt.phase, "winner": receipt.winner, "target": receipt.target,
                         "loss": receipt.loss, "updated": receipt.updated,
                         "teacher_count": receipt.teacher_count, "residual_offers": CONFIG["outputs"],
                         "nonzero_scores": sparse(step.scores[0]),
                         "nonzero_residuals": sparse(receipt.residual[0]),
                         "nonzero_features": sparse(step.features[0])})
        state, previous_token = step.state, token
    losses = [row["loss"] for row in teachers]
    return {"first": first, "teachers": teachers, "teacher_losses": losses,
            "loss_kind": "half_sum_squared_potential_error", "mean_loss": sum(losses) / len(losses),
            "byte_offers": sum(row["teacher_count"] for row in teachers),
            "residual_offers": sum(row["residual_offers"] for row in teachers),
            "enabled_residual_offers": CONFIG["outputs"] * len(teachers) if enabled else 0,
            "nonzero_residual_offers": sum(len(row["nonzero_residuals"]) for row in teachers),
            "enabled_nonzero_residual_offers": sum(len(row["nonzero_residuals"]) for row in teachers) if enabled else 0,
            "enabled_steps": len(teachers) if enabled else 0,
            "changed_steps": sum(row["updated"] for row in teachers),
            "before_sha256": before, "after_first_sha256": after_first,
            "after_teacher_sha256": state_hash(model)}


def target_margins(generated, target):
    """Offline scoring of the saved free call; a missing phase cannot be filled in."""
    margins = []
    for phase, token in enumerate((*target.encode("ascii"), BYTE_EOS)):
        if phase >= len(generated["phases"]):
            margins.append(None)
            continue
        row = generated["phases"][phase]
        if row["phase"] != phase:
            raise ValueError("saved generation phase order differs")
        scores = dict(row["nonzero_scores"])
        margins.append(scores.get(token, 0.0) - max(
            scores.get(other, 0.0) for other in range(CONFIG["outputs"]) if other != token
        ))
    return margins


def qualifies(scores, checks):
    return {arm: scores[arm] == {"train": 12, "heldout": 4}
            and all(scores[arm]["heldout"] > scores[control]["heldout"] for control in ("zero", "frozen"))
            and all(row["margin_passed"] for row in checks if row["arm"] == arm)
            for arm in ("lexical", "permuted")}


def summarize(rows):
    return {**previous.summarize(rows), "loss_kind": "half_sum_squared_potential_error",
            "mean_loss_by_byte1_byte2_eos": [sum(row["teacher_losses"][i] for row in rows) / len(rows)
                                             for i in range(3)]}


def run(output, revision, sources_sha):
    manifest = sources()
    current_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != current_revision or sources_sha != digest(manifest):
        raise ValueError("revision or source manifest mismatch")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    clock = time.get_clock_info("perf_counter")
    deadline = float(os.environ["HUMANBRAIN_RUN_DEADLINE_UNIX"])

    def guard():
        if time.perf_counter() - START > CONFIG["seconds"] or time.time() > deadline:
            raise TimeoutError("LMS comparison time ceiling exhausted")

    roster = schedule()
    write_json(output / "schedule.json", {"train": TRAIN, "heldout": HELDOUT, "rows": roster,
               "context_vectors": {pair: context(pair).tolist() for pair in TRAIN + HELDOUT}})
    admission = {"config": CONFIG, "arms": ARMS, "sources": manifest, "sources_sha256": sources_sha,
                 "revision": revision, "pid": os.getpid(),
                 "versions": {name: version(name) for name in ("torch", "numpy")},
                 "memory_monitor": "external launcher's owned process-tree peak working set",
                 "sparse_semantics": "all unlisted vector entries are exactly zero; every nonzero is recorded",
                 "duration_clock": {"name": "perf_counter", "implementation": clock.implementation,
                                    "resolution_seconds": clock.resolution,
                                    "monotonic": clock.monotonic, "adjustable": clock.adjustable},
                 "deadline_unix": deadline, "schedule_sha256": file_hash(output / "schedule.json")}
    write_json(output / "admission.json", admission)
    guard()
    models = {arm: PhaseContextLMS() for arm in ARMS}
    initial = {arm: copy.deepcopy(model.state_dict()) for arm, model in models.items()}
    if not all(same_state(initial["lexical"], state) for state in initial.values()):
        raise AssertionError("initial arm states differ")
    torch.save(initial, output / "initial.pt")
    records = []
    with (output / "lessons.jsonl").open("x", encoding="utf-8") as stream:
        for row in roster:
            for arm, model in models.items():
                guard()
                target, supplied = assignment(arm, row)
                receipt = lesson(model, context(supplied), target, enabled=arm != "frozen")
                record = {"index": row["index"], "arm": arm, "target": target, "context": supplied, **receipt}
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
            supplied = None if arm == "zero" else pair
            before = state_hash(model)
            emitted = generation(model, context(supplied))
            margins = target_margins(emitted, pair)
            checks.append({"arm": arm, "target": pair, "context": supplied,
                           "group": "train" if pair in TRAIN else "heldout", **emitted,
                           "target_margins": margins,
                           "margin_passed": all(value is not None and value > CONFIG["minimum_final_margin_exclusive"]
                                                for value in margins),
                           "before_sha256": before, "after_sha256": state_hash(model),
                           "exact": emitted["hex"] == pair.encode().hex() and emitted["stop_reason"] == "eos"})
        if not same_state(final[arm], model.state_dict()):
            raise AssertionError("final generation changed model state")
    if not same_state(initial["frozen"], final["frozen"]):
        raise AssertionError("frozen model changed")
    sources_after = sources()
    revision_after = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if manifest != sources_after or revision != revision_after:
        raise ValueError("sources changed during comparison")
    scores = {arm: {group: sum(row["exact"] for row in checks if row["arm"] == arm and row["group"] == group)
                    for group in ("train", "heldout")} for arm in ARMS}
    artifacts = ("schedule.json", "admission.json", "initial.pt", "final.pt", "lessons.jsonl")
    report = {**admission, "sources_after": sources_after, "revision_after": revision_after,
              "status": "completed", "checks": checks, "scores": scores,
              "useful_endpoint": qualifies(scores, checks), "acquisition": {},
              "counts": {"first_attempts": len(records), "final_generations": len(checks),
                         "byte_offers": sum(row["byte_offers"] for row in records),
                         "residual_offers": sum(row["residual_offers"] for row in records),
                         "enabled_residual_offers": sum(row["enabled_residual_offers"] for row in records),
                         "nonzero_residual_offers": sum(row["nonzero_residual_offers"] for row in records),
                         "enabled_nonzero_residual_offers": sum(row["enabled_nonzero_residual_offers"] for row in records),
                         "enabled_steps": sum(row["enabled_steps"] for row in records),
                         "changed_steps": sum(row["changed_steps"] for row in records),
                         "generation_decisions": sum(row["first"]["decisions"] for row in records)
                         + sum(row["decisions"] for row in checks)},
              "artifact_sha256": {name: file_hash(output / name) for name in artifacts},
              "artifact_bytes": {name: (output / name).stat().st_size for name in artifacts},
              "model_accounting": models["lexical"].accounting()}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        report["acquisition"][arm] = {"all": summarize(rows), "first128": summarize(rows[:128]),
                                      "last128": summarize(rows[-128:]),
                                      "byte_offers": sum(row["byte_offers"] for row in rows),
                                      "residual_offers": sum(row["residual_offers"] for row in rows),
                                      "enabled_residual_offers": sum(row["enabled_residual_offers"] for row in rows),
                                      "nonzero_residual_offers": sum(row["nonzero_residual_offers"] for row in rows),
                                      "enabled_nonzero_residual_offers": sum(row["enabled_nonzero_residual_offers"] for row in rows),
                                      "enabled_steps": sum(row["enabled_steps"] for row in rows),
                                      "changed_steps": sum(row["changed_steps"] for row in rows)}
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
