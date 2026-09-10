"""Saved-only acquisition aggregates; no model, teacher or execution calls."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

ROUTES = ("online", "learning_disabled", "simple")


def rows(path, expected):
    digest, size, count = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n") or len(raw) > 64 * 1024**2:
                raise ValueError("incomplete or oversized acquisition journal row")
            digest.update(raw)
            size += len(raw)
            count += 1
            yield json.loads(raw)
    if (size, count, digest.hexdigest()) != (expected["bytes"], expected["rows"], expected["sha256"]):
        raise ValueError("journal differs from final authenticated frontier")


def review(output):
    started = time.perf_counter()
    output = Path(output)
    head = json.loads((output / "checkpoints/HEAD.json").read_bytes())
    boundary_raw = (output / "checkpoints/boundary-000000000.json").read_bytes()
    if hashlib.sha256(boundary_raw).hexdigest() != head["sha256"]:
        raise ValueError("saved review boundary hash differs")
    boundary = json.loads(boundary_raw)
    config_raw = (output / "checkpoints/config.json").read_bytes()
    if hashlib.sha256(config_raw).hexdigest() != boundary["config_sha256"]:
        raise ValueError("saved review config hash differs")
    config = json.loads(config_raw)
    manifest_raw = (output / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("saved review manifest hash differs")
    manifest = json.loads(manifest_raw)
    frontiers = {item["kind"]: item for item in boundary["frontiers"]["journals"]}

    def check_correct(row):
        from experiments.r4_integrated.persistent_language_action_execution_v2 import (
            grade_execution,
        )
        task = manifest["tasks"][row["episode_id"]]
        if row["kind"] == "reading":
            correct = row["answer_hex"] in task["grade"]["accepted_answer_hex"]
            if row["route"] != "simple":
                correct = correct and row["prediction"]["stop_reason"] == "eos"
                if row["answer_hex"] != row["prediction"]["answer_hex"]:
                    raise ValueError("reading outcome differs from free prediction")
        else:
            grade = grade_execution(row["execution"], task["grade"], task["files"])
            if grade != row["grade"]:
                raise ValueError("saved edit grade differs")
            correct = grade["task_success"]
        if correct != row["correct"]:
            raise ValueError("saved free task correctness differs")
        return correct
    attempts = {r: [] for r in ROUTES}
    for i, row in enumerate(rows(output / "attempts.jsonl", frontiers["attempts"])):
        step, route = i // 3, ROUTES[i % 3]
        if (row["step_index"] != step or row["route"] != route
                or row["episode_id"] != manifest["schedule"][step]["episode_id"]):
            raise ValueError("attempt journal order differs from fixed schedule")
        check_correct(row)
        attempts[route].append({k: row[k] for k in ("step_index", "episode_id", "correct")})
    if any(len(items) != 64 for items in attempts.values()):
        raise ValueError("incomplete192 main responses")
    coverage = {r: {} for r in ROUTES[:2]}
    teachers = {r: [] for r in ROUTES}
    for i, row in enumerate(rows(output / "teachers.jsonl", frontiers["teachers"])):
        step, route = i // 3, ROUTES[i % 3]
        name = manifest["schedule"][step]["episode_id"]
        if (row["step_index"], row["route"], row["episode_id"]) != (step, route, name):
            raise ValueError("teacher journal order differs")
        if route == "simple":
            teachers[route].append({"step_index": step, "episode_id": name})
            continue
        diagnostic = row["diagnostic"]
        skill = coverage[route].setdefault(name, [{"position": p["position"], "target": p["target"],
            "eos": p["eos"], "offers": 0, "literal_proposals": 0, "literal_publications": 0}
            for p in diagnostic["positions"]])
        if row["reader_offers"] != len(skill) or row["action_labels"] != (3 if name.endswith("E") else 0):
            raise ValueError("actual teacher budget differs")
        for total, p in zip(skill, diagnostic["positions"], strict=True):
            if (total["position"], total["target"], total["eos"]) != (p["position"], p["target"], p["eos"]):
                raise ValueError("target position changed during repeated teaching")
            total["offers"] += 1
            total["literal_proposals"] += p["direct_teacher_proposed"]
            total["literal_publications"] += p["direct_teacher_published"]
        teachers[route].append({"step_index": step, "episode_id": name,
            "offers": diagnostic["offers"], "literal_offers": diagnostic["literal_offers"],
            "mode_mixture_path_logscore": diagnostic["mode_mixture_path_logscore"],
            "literal_mean_target_probability": sum(p["literal_target_probability"] for p in diagnostic["positions"]) / len(skill),
            "controller": row["controller"], "actions": row.get("actions")})
    if any(len(items) != 64 for items in teachers.values()):
        raise ValueError("incomplete192 corrections")
    checkpoints = []
    for expected, row in zip(manifest["checkpoints"], rows(output / "probes.jsonl", frontiers["probes"]), strict=True):
        if row["checkpoint"] != expected or row["live_before"] != row["live_after"] or len(row["rows"]) != 12:
            raise ValueError("incomplete probe or live state mutation")
        outcomes = []
        roster = [(r, case) for r in ROUTES for case in manifest["checkpoint_cases"]]
        for item, coordinate in zip(row["rows"], roster, strict=True):
            attempt, teacher = item["attempt"], item["teacher"]
            if (attempt["route"], attempt["episode_id"]) != coordinate:
                raise ValueError("checkpoint roster differs")
            check_correct(attempt)
            outcome = {k: attempt[k] for k in ("route", "episode_id", "correct", "answer_hex", "proposal")}
            outcome["free_response"] = attempt.get("free_response")
            outcome["diagnostic"] = None if teacher is None else teacher.get("diagnostic")
            if outcome["diagnostic"] is not None and outcome["diagnostic"]["direct_teacher_published_offers"]:
                raise ValueError("probe published teacher learning")
            outcomes.append(outcome)
        checkpoints.append({"checkpoint": expected, "outcomes": outcomes,
                            "clone_receipts": row["clone_receipts"]})
    interference = []
    for route in ROUTES[:2]:
        for name in ("A00-R", "A00-E"):
            series = [next(r for r in cp["outcomes"] if r["route"] == route and r["episode_id"] == name)
                      for cp in checkpoints]
            for prev, current, left, right in zip(series, series[1:], manifest["checkpoints"], manifest["checkpoints"][1:]):
                before, after = prev["diagnostic"], current["diagnostic"]
                interference.append({"route": route, "episode_id": name, "from": left, "to": right,
                    "correct_before": prev["correct"], "correct_after": current["correct"],
                    "conditional_logscore_delta": after["mode_mixture_path_logscore"] - before["mode_mixture_path_logscore"],
                    "literal_mean_logp_delta": sum(p["literal_target_log_probability"] for p in after["positions"]) / after["offers"]
                    - sum(p["literal_target_log_probability"] for p in before["positions"]) / before["offers"]})
    acquisitions = {route: all(r["correct"] for cp in checkpoints if cp["checkpoint"] in (32, 64)
                              for r in cp["outcomes"] if r["route"] == route and r["episode_id"].startswith("A00"))
                    for route in ROUTES}
    for route in coverage.values():
        if any(p["offers"] != 32 for skill in route.values() for p in skill):
            raise ValueError("target position did not receive32 offers")
    if any(not math.isfinite(x["conditional_logscore_delta"]) for x in interference):
        raise ValueError("nonfinite score comparison")
    return {"schema": "repeated-teaching-saved-learning-review-v1", "status": "passed",
            "main_attempts": attempts, "teacher_progress": teachers, "effective_coverage": coverage,
            "checkpoints": checkpoints, "interference": interference, "complete_acquisition": acquisitions,
            "scope": "fixed common initial RNG panel; sampled upstream paths can diverge; saved observer score aggregates (no independent tensor recomputation), not latent-marginal likelihood; task grades recomputed",
            "expanded_transfer_or_retention": False, "model_calls": 0, "teacher_calls": 0,
            "elapsed_seconds": time.perf_counter() - started}
