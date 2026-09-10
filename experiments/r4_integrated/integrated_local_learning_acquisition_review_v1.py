"""Saved-only science review of the fixed 512-per-skill integrated comparison.

No model, tensor decoder, teacher, task builder or executor imports/calls. The
old independent scalar execution reader checks actual saved edits and grades.
Compact practice receipts cannot reconstruct omitted probability vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_v8_review as saved

DRAFT = True
ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("online", "learning_disabled", "simple")
NEURAL = ROUTES[:2]
SKILLS = ("A00-R", "A00-E")
CASES = (*SKILLS, "R02-R", "R12-R")
CHECKPOINTS = (0, 64, 128, 256, 512, 768, 1024)
SCHEDULE = ["A00-R"] * 64 + ["A00-E"] * 64 + ["A00-R", "A00-E"] * 448
JSON_LIMIT = 16 * 1024**2
ROW_LIMIT = 64 * 1024**2
OUTPUT_LIMIT = 64 * 1024**2
FREE_RESERVE = 6 * 1024**3
require = saved.require


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def close(left, right, label):
    require(
        type(left) in (int, float)
        and math.isfinite(left)
        and type(right) in (int, float)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12),
        label,
    )


def probability(value):
    require(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
        "invalid probability",
    )
    return value


def path_within(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "relative path")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path != root.resolve(), "path escapes root")
    return path


class Evidence:
    """One bounded saved review's explicit fresh read counters."""

    def __init__(self, deadline):
        self.deadline = deadline
        self.bytes = 0
        self.passes = 0
        self.rows = 0
        self.pending_read = False

    def admit(self):
        require(time.perf_counter() < self.deadline, "saved review deadline exceeded")

    def read(self, path, limit=JSON_LIMIT):
        self.admit()
        require(path.stat().st_size <= limit, "saved metadata exceeds bound")
        self.pending_read = True
        raw = path.read_bytes()
        self.bytes += len(raw)
        self.passes += 1
        self.pending_read = False
        return raw

    def json(self, path, expected=None):
        raw = self.read(path)
        if expected is not None:
            require(hashlib.sha256(raw).hexdigest() == expected, "metadata identity differs")
        return json.loads(raw)

    def file(self, path, expected):
        self.admit()
        h, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            while True:
                self.admit()
                self.pending_read = True
                raw = stream.read(1024**2)
                self.bytes += len(raw)
                self.pending_read = False
                if not raw:
                    break
                size += len(raw)
                h.update(raw)
        self.passes += 1
        require(h.hexdigest() == expected["sha256"], "saved file identity differs")
        require("bytes" not in expected or size == expected["bytes"], "saved file size differs")

    def journal(self, path, expected):
        self.admit()
        h, size, count = hashlib.sha256(), 0, 0
        with path.open("rb") as stream:
            while True:
                self.admit()
                self.pending_read = True
                raw = stream.readline(ROW_LIMIT + 1)
                self.bytes += len(raw)
                self.pending_read = False
                if not raw:
                    break
                require(raw.endswith(b"\n") and len(raw) <= ROW_LIMIT, "incomplete journal row")
                h.update(raw)
                size += len(raw)
                count += 1
                self.rows += 1
                yield json.loads(raw)
        self.passes += 1
        require(
            (size, count, h.hexdigest())
            == (expected["bytes"], expected["rows"], expected["sha256"]),
            "journal differs from authenticated final frontier",
        )


def expected_observation(manifest, name, route):
    task = manifest["tasks"][name]
    if name in SKILLS:
        return {
            "schema": "sequence-assembly-raw-observation-v1",
            "request_hex": task["request_hex"],
            "files": task["files"] if task["kind"] == "edit" else None,
            "feedback": None,
        }
    retained = manifest["prior_regression_bindings"]["routes"][
        "simple" if route == "simple" else "online"
    ][task["prior_episode_id"]]["observation"]
    return {
        "schema": "sequence-assembly-raw-observation-v1",
        "request_hex": task["request_hex"],
        "files": retained["files"] if task["kind"] == "edit" else None,
        "feedback": retained["feedback"],
    }


def target_for(task):
    teacher = task["teacher"]
    return bytes.fromhex(
        teacher["replacement_hex"] if task["kind"] == "edit" else teacher["target_hex"]
    )


def _free(row, prediction, target):
    answer, stop = bytes.fromhex(prediction["answer_hex"]), prediction["stop_reason"]
    require(stop in ("eos", "limit") and len(answer) <= 128, "free stop/byte bound")
    summary = row.get("free_response")
    if summary is None:
        return None
    tokens = [item["actual_token"] for item in summary["positions"]]
    require(
        len(tokens) == len(answer) + 1 and tokens[: len(answer)] == list(answer),
        "free sampled tokens differ from returned bytes",
    )
    require(
        tokens[-1] == 256
        if stop == "eos"
        else len(answer) == 128 and type(tokens[-1]) is int and 0 <= tokens[-1] < 256,
        "invalid EOS or unreturned limit draw",
    )
    if "steps" in prediction:
        require(
            tokens == [item["token"] for item in prediction["steps"]],
            "free summary differs from retained full panel steps",
        )
    if "decisions" in prediction:
        require(prediction["decisions"] == len(tokens), "compact actual free decision count")
    require(
        summary["answer_hex"] == answer.hex()
        and summary["stop_reason"] == stop
        and summary["bytes"] == len(answer)
        and summary["decisions"] == len(tokens),
        "free scalar summary differs from complete output",
    )
    require(
        summary["exact_bytes"] is (answer == target)
        and summary["complete_bytes_and_eos"] is (answer == target and stop == "eos"),
        "free exactness differs",
    )
    expected_prefix = 0
    for left, right in zip(answer, target, strict=False):
        if left != right:
            break
        expected_prefix += 1
    require(summary["matching_raw_prefix_bytes"] == expected_prefix, "raw prefix length differs")
    aligned = True
    for position, (actual, item) in enumerate(zip(tokens, summary["positions"], strict=True)):
        intended = (
            target[position] if position < len(target) else 256 if position == len(target) else None
        )
        require(
            item["position"] == position
            and item["actual_token"] == actual
            and item["intended_token"] == intended
            and item["intended_prefix_aligned"] is aligned
            and item["matches_intended_here"] is (aligned and intended == actual),
            "free prefix scalar differs",
        )
        aligned = aligned and intended == actual
    result = {k: v for k, v in summary.items() if k != "positions"}
    result["unreturned_limit_token"] = tokens[-1] if stop == "limit" else None
    return result


def check_attempt(row, manifest, output, launch, evidence):
    route, name = row["route"], row["episode_id"]
    task = manifest["tasks"][name]
    require(
        row["kind"] == task["kind"]
        and row["observation"] == expected_observation(manifest, name, route),
        "actual task input differs",
    )
    require(row["memory_before_sha256"] == row["memory_after_sha256"], "factual memory changed")
    neural = route in NEURAL
    prediction = row["prediction"]
    if task["kind"] == "edit" and neural:
        prediction = prediction["payload"]
    free = None
    if task["kind"] == "reading":
        correct = row["answer_hex"] in task["grade"]["accepted_answer_hex"]
        require(row["answer_hex"] == prediction["answer_hex"], "actual reading bytes differ")
        if neural:
            correct = correct and prediction["stop_reason"] == "eos"
        require(row["proposal"] is None and row["execution"] is None, "reading executed action")
    else:
        # Reuse the pure saved reader, not the executor module (which imports neurons).
        ex = row["execution"]
        public = bool(
            ex["public_driver_policy"]["driver_integrity"]
            and ex["returncode"] == 0
            and not ex["timeout"]
            and ex["capture_complete"]
            and ex["files_complete"]
            and not ex["output_limit_exceeded"]
            and not ex["capture_errors"]
        )
        execution_row = dict(
            row, before_files=row["observation"]["files"], public_checks_passed=public
        )
        evidence.admit()
        evidence.pending_read = True
        saved.check_execution(execution_row, task, output, launch)
        evidence.bytes += sum(len(bytes.fromhex(r["content_hex"])) for r in row["after_files"])
        evidence.passes += len(row["after_files"])
        evidence.pending_read = False
        correct = row["grade"]["task_success"]
        if neural:
            require(
                row["proposal"]["replacement_hex"] == prediction["answer_hex"]
                and row["proposal"]["stop_reason"] == prediction["stop_reason"],
                "actual payload differs from free generation",
            )
    require(type(row["correct"]) is bool and row["correct"] == correct, "saved grade differs")
    rng = None
    scan_offers = free_decisions = controller_draws = 0
    if neural:
        if row["scope"] == "main":
            require(
                prediction["schema"] == "integrated-local-learning-compact-prediction-v1",
                "compact prediction schema",
            )
            rng = prediction["rng_before_sha256"]
            scan_offers = prediction["scan_offers"]
            free_decisions = prediction["decisions"]
            controller_draws = prediction["controller_draws"]
            require(
                prediction["observation_sha256"]
                == saved.scalar.observation_hash(row["observation"]),
                "compact observation identity differs",
            )
        else:
            rng = prediction["scan_steps"][0]["draws"][0]["rng_before_sha256"]
            scan_offers = len(prediction["scan_steps"])
            free_decisions = len(prediction["steps"])
            controller_draws = sum(len(r["draws"]) for r in prediction["scan_steps"])
            controller_draws += sum(len(r["controller"]["draws"]) for r in prediction["steps"])
        if name in SKILLS:
            free = _free(row, prediction, target_for(task))
            require(free is not None, "taught free response summary omitted")
    proposal = row["proposal"]
    span = None
    if name == "A00-E":
        wanted = task["teacher"]
        span = all(proposal.get(k) == wanted[k] for k in ("operation", "path", "start", "end"))
    behavior = {
        "answer_hex": row["answer_hex"],
        "proposal": proposal,
        "stop_reason": prediction.get("stop_reason") if neural else None,
        "grade": row.get("grade"),
        "correct": correct,
    }
    return {
        "step_index": row["step_index"],
        "route": route,
        "episode_id": name,
        "scope": row["scope"],
        **behavior,
        "span_correct": span,
        "actual_input_sha256": digest(row["observation"]),
        "rng_before_sha256": rng,
        "behavior_sha256": digest(behavior),
        "free_response": free,
        "model_before_sha256": row["model_before_sha256"],
        "model_after_sha256": row["model_after_sha256"],
        "memory_before_sha256": row["memory_before_sha256"],
        "memory_after_sha256": row["memory_after_sha256"],
        "diagnostics": None
        if row["execution"] is None
        else {k: row["execution"][k] for k in saved.scalar.FEEDBACK_KEYS},
        "scan_offers": scan_offers,
        "free_decisions": free_decisions,
        "controller_draws": controller_draws,
        "phase_seconds": row.get("phase_seconds", {}),
        "elapsed_seconds": row["elapsed_seconds"],
    }


def check_alignment(value, free, target_bytes):
    require(
        value["schema"] == "integrated-local-learning-free-teacher-alignment-v1", "alignment schema"
    )
    tokens = [
        *bytes.fromhex(free["answer_hex"]),
        256 if free["stop_reason"] == "eos" else free["unreturned_limit_token"],
    ]
    expected = []
    for position, token in enumerate(tokens):
        target = (
            target_bytes[position]
            if position < len(target_bytes)
            else 256
            if position == len(target_bytes)
            else None
        )
        actual = {
            "position": position,
            "actual_token": token,
            "intended_token": target,
            "intended_prefix_aligned": True,
        }
        expected.append(actual)
        if actual["actual_token"] != actual["intended_token"]:
            break
    require(
        value["free_decisions"] == free["decisions"]
        and value["omitted_after_divergence"] == free["decisions"] - len(expected),
        "alignment omitted/free count differs",
    )
    for row, actual in zip(value["positions"], expected, strict=True):
        require(
            row["position"] == actual["position"]
            and row["actual_token"] == actual["actual_token"]
            and row["target"] == actual["intended_token"]
            and row["intended_prefix_aligned"] is actual["intended_prefix_aligned"],
            "alignment did not use actual free prefix",
        )
        teacher = row["teacher"]
        require(
            row["same_emission_activity"]
            is (
                teacher is not None
                and row["free"]["mode_activity_sha256"] == teacher["mode_activity_sha256"]
            ),
            "activity alignment differs",
        )
        require(
            row["same_incoming_rng"]
            is (
                teacher is not None
                and row["free"]["incoming_rng_sha256"] == teacher["incoming_rng_sha256"]
            ),
            "RNG alignment differs",
        )
    return value


def check_teacher(row, attempt, manifest, *, panel):
    route, name = attempt["route"], attempt["episode_id"]
    require(
        (row["scope"], row["step_index"], row["route"], row["episode_id"])
        == (attempt["scope"], attempt["step_index"], route, name),
        "teacher identity differs",
    )
    enabled = not panel and route == "online"
    require(row["learning_enabled"] is enabled, "teacher publication gate differs")
    task, expected = manifest["tasks"][name], target_for(manifest["tasks"][name])
    require(row["teacher"] == task["teacher"], "teacher target differs")
    require(
        row["model_after_sha256"] == attempt["model_after_sha256"]
        and row["model_before_sha256"] == attempt["model_before_sha256"]
        and row["memory_before_sha256"] == attempt["memory_before_sha256"]
        and row["memory_after_sha256"] == attempt["memory_after_sha256"],
        "teacher resolution/attempt binding differs",
    )
    if route == "simple":
        require(row["reader_offers"] == row["action_labels"] == 0, "reference neural teaching")
        return row
    diagnostic = row["diagnostic"]
    require(
        row["reader_offers"] == len(expected) + 1
        and row["action_labels"] == (3 if name == "A00-E" else 0)
        and diagnostic["target_hex"] == expected.hex()
        and diagnostic["offers"] == len(expected) + 1
        and diagnostic["raw_offers"] == len(expected)
        and diagnostic["eos_offers"] == 1,
        "complete target offer budget differs",
    )
    literal_count, published_count = 0, 0
    if panel:
        require(len(row["teacher_steps"]) == len(expected) + 1, "complete retained panel teacher")
    for index, (target, item) in enumerate(
        zip((*expected, 256), diagnostic["positions"], strict=True)
    ):
        require(
            item["position"] == index
            and item["target"] == target
            and item["eos"] is (target == 256),
            "target position differs",
        )
        mode = item["selected_mode"]
        require(
            type(mode) is int
            and 0 <= mode < 4
            and item["selected_literal"] is (mode == 3)
            and item["direct_teacher_published"] is (mode == 3 and enabled),
            "actual selected literal contact differs",
        )
        literal_count += mode == 3
        published_count += mode == 3 and enabled
        if panel:
            p, lp = item["literal_target_probability"], item["literal_target_log_probability"]
            check_panel_position(item, row["teacher_steps"][index], target, expected[:index])
            components = item["mode_mixture_components"]
            require(len(components) == 4, "mode mixture component count")
            for component in components:
                probability(component)
            close(sum(components), item["mode_mixture_target_probability"], "mixture sum")
            close(
                math.exp(item["mode_mixture_target_log_probability"]),
                sum(components),
                "mixture log probability",
            )
        else:
            require(item["offered"] is True, "missing actual teacher offer")
            probabilities = item["mode_probabilities"]
            require(len(probabilities) == 4, "mode population width")
            for p_mode in probabilities:
                probability(p_mode)
            close(sum(probabilities), 1, "mode probability normalization")
            p, lp = (
                item["selected_literal_target_probability"],
                item["selected_literal_target_log_probability"],
            )
            require(
                (p is None) is (mode != 3) and (lp is None) is (mode != 3),
                "unselected literal mislabeled evaluated",
            )
        if p is not None:
            probability(p)
            require(
                type(lp) in (int, float) and math.isfinite(lp) and lp <= 1e-12,
                "invalid literal log probability",
            )
            close(math.exp(lp), p, "literal probability/log probability")
    require(
        diagnostic["literal_offers"] == literal_count
        and diagnostic["direct_teacher_published_offers"] == published_count,
        "literal inclusion/publication denominator differs",
    )
    resolution = row["controller"]
    require(
        resolution["learning_enabled"] is enabled and resolution["rng_published"] is True,
        "resolution enabled/RNG gate differs",
    )
    require(resolution["ticks"]["teacher"] == len(expected) + 1, "resolution teacher ticks differ")
    if not panel:
        require(
            diagnostic["actual_teacher_controller_draws"]
            == 3 * len(diagnostic["positions"]) + literal_count,
            "actual teacher draw count differs",
        )
    check_alignment(row["free_teacher_alignment"], attempt["free_response"], expected)
    if panel:
        close(
            sum(p["mode_mixture_target_log_probability"] for p in diagnostic["positions"]),
            diagnostic["mode_mixture_path_logscore"],
            "conditional path log score",
        )
    return {k: row[k] for k in row if k != "teacher_steps"}


def check_panel_position(item, step, target, prefix):
    """Independently check each mixture component against the retained actual teacher tick."""
    tick = step["controller"]
    draws, mode = tick["draws"], item["selected_mode"]
    require(
        step["output_step"] == item["position"]
        and step["observed_token"] == target
        and step["prefix_hex"] == prefix.hex()
        and tick["category"] == "teacher"
        and draws[2]["index"] == mode
        and len(draws) == 3 + int(mode == 3),
        "panel teacher actual prefix/draw sequence",
    )
    probabilities = draws[2]["probabilities"]
    require(len(probabilities) == 4, "panel mode distribution width")
    for p_mode in probabilities:
        probability(p_mode)
    close(sum(probabilities), 1, "panel mode normalization")
    tokens = [tick["prior_token"], *(256 if x is None else x for x in tick["register_tokens"])]
    literal_p = item["literal_target_probability"]
    components = [probabilities[i] * int(token == target) for i, token in enumerate(tokens)]
    components.append(probabilities[3] * literal_p)
    for observed, expected in zip(item["mode_mixture_components"], components, strict=True):
        close(observed, expected, "panel mode mixture component")
    require(item["activity_sha256"] == draws[2]["activity_sha256"], "panel emission activity")
    selected_p = literal_p if mode == 3 else float(tokens[mode] == target)
    close(item["selected_mode_target_probability"], selected_p, "selected emitter probability")
    if mode == 3:
        close(draws[3]["probabilities"][target], literal_p, "actual selected literal probability")
    require(item["sampled_match"] is (tick["emitted_token"] == target), "panel sampled match")


def score_identity(row):
    return row["position"], row["target"], row["activity_sha256"], row["fixed_competitor"]


def check_anchor_scores(teacher, reference):
    current = teacher["fixed_anchor_scores"]["positions"]
    for item, anchor in zip(current, reference, strict=True):
        require(
            score_identity(item)
            == (
                anchor["position"],
                anchor["target"],
                anchor["activity_sha256"],
                anchor["competitor"],
            ),
            "fixed CP0 site changed",
        )
        probability(item["target_probability"])
        close(math.exp(item["target_log_probability"]), item["target_probability"], "anchor p/logp")
        close(
            item["target_current"] - item["fixed_competitor_current"],
            item["fixed_competitor_margin"],
            "fixed competitor margin",
        )
        require(
            item["strongest_competitor"] != item["target"]
            and item["strongest_competitor_margin"] <= item["fixed_competitor_margin"] + 1e-10,
            "strongest competitor margin differs",
        )


def coverage_add(coverage, name, diagnostic):
    positions = coverage.setdefault(
        name,
        [
            {
                "position": p["position"],
                "target": p["target"],
                "eos": p["eos"],
                "offers": 0,
                "literal_contacts": 0,
                "literal_publications": 0,
                "selected_literal_probability_sum": 0.0,
                "selected_literal_ce_sum": 0.0,
                "selected_literal_probability_first": None,
                "selected_literal_probability_last": None,
                "selected_literal_probability_min": None,
                "selected_literal_probability_max": None,
                "selected_activity_first": None,
                "selected_activity_last": None,
                "selected_activity_differs_from_first": 0,
            }
            for p in diagnostic["positions"]
        ],
    )
    for total, item in zip(positions, diagnostic["positions"], strict=True):
        require(
            (total["position"], total["target"]) == (item["position"], item["target"]),
            "coverage target drift",
        )
        total["offers"] += 1
        total["literal_contacts"] += item["selected_literal"]
        total["literal_publications"] += item["direct_teacher_published"]
        if item["selected_literal"]:
            p = item.get(
                "selected_literal_target_probability", item.get("literal_target_probability")
            )
            lp = item.get(
                "selected_literal_target_log_probability",
                item.get("literal_target_log_probability"),
            )
            total["selected_literal_probability_sum"] += p
            total["selected_literal_ce_sum"] -= lp
            identity = item.get("mode_activity_sha256", item.get("activity_sha256"))
            if total["selected_literal_probability_first"] is None:
                total["selected_literal_probability_first"] = p
                total["selected_literal_probability_min"] = p
                total["selected_literal_probability_max"] = p
                total["selected_activity_first"] = identity
            total["selected_literal_probability_last"] = p
            total["selected_literal_probability_min"] = min(
                total["selected_literal_probability_min"], p
            )
            total["selected_literal_probability_max"] = max(
                total["selected_literal_probability_max"], p
            )
            total["selected_activity_last"] = identity
            total["selected_activity_differs_from_first"] += (
                identity != total["selected_activity_first"]
            )


def finish_coverage(coverage):
    for positions in coverage.values():
        for p in positions:
            count = p["literal_contacts"]
            p["selected_literal_probability_mean"] = (
                p["selected_literal_probability_sum"] / count if count else None
            )
            p["selected_literal_ce_mean"] = p["selected_literal_ce_sum"] / count if count else None
            p["scope"] = "actual selected-contact subset; activity-varied, not fixed-site change"


def progress_add(progress, row, attempt):
    """Aggregate every practice occasion into the fixed declared interval, without sampling."""
    step, name = row["step_index"], row["episode_id"]
    left, right = next((a, b) for a, b in pairwise(CHECKPOINTS) if a <= step < b)
    key = f"{left}:{right}:{name}"
    item = progress.setdefault(
        key,
        {
            "from": left,
            "to": right,
            "episode_id": name,
            "teacher_occasions": 0,
            "reader_symbols": 0,
            "motor_labels": 0,
            "actual_teacher_draws": 0,
            "positions": {},
            "alignment": {
                "responses": 0,
                "first_same_activity": 0,
                "first_same_rng": 0,
                "aligned_prefix_rows": 0,
                "same_activity_aligned_rows": 0,
                "omitted_after_divergence": 0,
                "complete_bytes_and_eos": 0,
            },
        },
    )
    item["teacher_occasions"] += 1
    item["reader_symbols"] += row["reader_offers"]
    item["motor_labels"] += row["action_labels"]
    if row["route"] not in NEURAL:
        return
    diagnostic = row["diagnostic"]
    item["actual_teacher_draws"] += diagnostic["actual_teacher_controller_draws"]
    coverage_add(item["positions"], name, diagnostic)
    alignment = row["free_teacher_alignment"]
    rows = alignment["positions"]
    counts = item["alignment"]
    counts["responses"] += 1
    counts["first_same_activity"] += rows[0]["same_emission_activity"]
    counts["first_same_rng"] += rows[0]["same_incoming_rng"]
    counts["aligned_prefix_rows"] += len(rows)
    counts["same_activity_aligned_rows"] += sum(p["same_emission_activity"] for p in rows)
    counts["omitted_after_divergence"] += alignment["omitted_after_divergence"]
    counts["complete_bytes_and_eos"] += attempt["free_response"]["complete_bytes_and_eos"]


def metric(rows):
    return {
        name: {
            "correct": sum(r["correct"] for r in rows if r["episode_id"] == name),
            "denominator": sum(r["episode_id"] == name for r in rows),
        }
        for name in CASES
    }


def paired(left, right):
    same_input = left["actual_input_sha256"] == right["actual_input_sha256"]
    neural = left["route"] in NEURAL or right["route"] in NEURAL
    same_rng = (
        (
            left["rng_before_sha256"] is not None
            and left["rng_before_sha256"] == right["rng_before_sha256"]
        )
        if neural
        else None
    )
    paired_input = same_input and (same_rng if neural else True)
    result = {
        "step_index": left["step_index"],
        "episode_id": left["episode_id"],
        "left_route": left["route"],
        "right_route": right["route"],
        "same_actual_input": same_input,
        "same_incoming_rng": same_rng,
        "paired_actual_input_rng": paired_input,
        "behavior_equal_when_paired": (
            left["behavior_sha256"] == right["behavior_sha256"] if paired_input else None
        ),
        "raw_diagnostics_equal": left["diagnostics"] == right["diagnostics"],
        "left_correct": left["correct"],
        "right_correct": right["correct"],
        "scope": "saved complete outputs/proposals/grades; not omitted full forward vectors",
    }
    if (left.get("teacher") or {}).get("fixed_anchor_scores") and (right.get("teacher") or {}).get(
        "fixed_anchor_scores"
    ):
        result["fixed_site_scores"] = []
        for a, b in zip(
            left["teacher"]["fixed_anchor_scores"]["positions"],
            right["teacher"]["fixed_anchor_scores"]["positions"],
            strict=True,
        ):
            same = score_identity(a) == score_identity(b)
            result["fixed_site_scores"].append(
                {
                    "position": a["position"],
                    "target": a["target"],
                    "same_fixed_site_and_competitor": same,
                    "target_probability_delta": b["target_probability"] - a["target_probability"]
                    if same
                    else None,
                    "target_log_probability_delta": b["target_log_probability"]
                    - a["target_log_probability"]
                    if same
                    else None,
                    "fixed_competitor_margin_delta": b["fixed_competitor_margin"]
                    - a["fixed_competitor_margin"]
                    if same
                    else None,
                }
            )
    return result


def _review(output, manifest_sha256, evidence):
    output = Path(output).resolve()
    directory = output / "checkpoints"
    head_raw = evidence.read(directory / "HEAD.json")
    head = json.loads(head_raw)
    require(
        head["schema"] == "repeated-teaching-head-v1"
        and head["sequence"] == 0
        and head["manifest"] == "boundary-000000000.json",
        "final safe HEAD differs",
    )
    boundary = evidence.json(directory / head["manifest"], head["sha256"])
    config = evidence.json(directory / "config.json", boundary["config_sha256"])
    require(
        config["manifest_sha256"] == manifest_sha256
        and Path(config["source_root"]).resolve() == ROOT
        and Path(config["evidence_root"]).resolve() == output,
        "source/evidence root binding",
    )
    manifest = evidence.json(output / "manifest.json", manifest_sha256)
    require(
        evidence.json(directory / "manifest.json", manifest_sha256) == manifest,
        "checkpoint manifest differs",
    )
    require(
        manifest["schema"] == "integrated-local-learning-acquisition-v1"
        and manifest["schedule"] == [{"index": i, "episode_id": n} for i, n in enumerate(SCHEDULE)]
        and manifest["checkpoints"] == list(CHECKPOINTS)
        and manifest["routes"] == list(ROUTES)
        and manifest["checkpoint_cases"] == list(CASES),
        "fixed manifest schedule differs",
    )
    completion_raw = evidence.read(output / "completion.json")
    completion = json.loads(completion_raw)
    require(
        completion["schema"] == "integrated-local-learning-worker-completion-v1"
        and completion["status"] == "completed_pending_saved_audit"
        and set(completion["metadata_sha256"])
        == {"report.json", "manifest.json", "sources.json", "initialization.json", "launch.json"},
        "clean worker completion missing",
    )
    metadata = {
        name: evidence.json(path_within(output, name), identity)
        for name, identity in completion["metadata_sha256"].items()
    }
    report = metadata["report.json"]
    require(
        report["status"] in ("completed", "completed_pending_saved_audit")
        and report["manifest_sha256"] == manifest_sha256
        and report["final"]["boundary_sha256"] == head["sha256"],
        "worker incomplete/binding",
    )
    require(completion["rule"] == report["rule"], "completion rule differs")
    sources = metadata["sources.json"]
    require(sources["source_sha256"] == config["source_sha256"], "source inventory differs")
    source_map = dict(sources["source_sha256"])
    for name, identity in sources["selection"]["source_sha256"].items():
        require(name not in source_map or source_map[name] == identity, "source closure conflict")
        source_map[name] = identity
    for name, identity in source_map.items():
        evidence.file(path_within(ROOT, name), {"sha256": identity})
    initialization = metadata["initialization.json"]
    require(
        initialization == config["parent"]
        and initialization["rule"] == report["rule"]
        and initialization["parent_boundary_sha256"] == boundary["parent"]
        and initialization["bank_sha256"] == manifest["parent_bank_sha256"],
        "initializer binding",
    )
    require(
        initialization["initial_routes"]["online"]
        == initialization["initial_routes"]["learning_disabled"],
        "initial neural copies differ",
    )
    for item in initialization["initial_routes"].values():
        require(
            (item["experience"], item["reader_experience"]) == (401, 406), "retained counter reset"
        )
    launch = metadata["launch.json"]
    frontiers = {x["kind"]: x for x in boundary["frontiers"]["journals"]}
    require(list(frontiers) == ["attempts", "teachers", "probes"], "journal inventory differs")
    for kind, count in (("attempts", 3072), ("teachers", 3072), ("probes", 7)):
        require(
            frontiers[kind]["path"] == kind + ".jsonl" and frontiers[kind]["rows"] == count,
            "complete journal denominator differs",
        )
    inventory = boundary["frontiers"]["execution_files"]
    actual_files = {
        p.relative_to(output).as_posix() for p in (output / "executions").rglob("*") if p.is_file()
    }
    require(actual_files == {r["path"] for r in inventory}, "actual execution inventory differs")
    for item in inventory:
        evidence.file(path_within(output, item["path"]), item)
    require(set(boundary["routes"]) == set(NEURAL), "missing final neural route")
    for route, item in boundary["routes"].items():
        require(
            route in NEURAL and (item["experience"], item["reader_experience"]) == (913, 1430),
            "final route experiences differ",
        )
        # Byte authentication is not a tensor decode or a strict restoration claim.
        evidence.file(path_within(directory, item["payload"]), item)
    intent = evidence.json(directory / "intent-000000000.json", boundary["intent_sha256"])
    require(intent["config_sha256"] == boundary["config_sha256"], "block intent binding")
    attempts = {route: [] for route in ROUTES}
    last_state = dict(report["initial_hashes"])
    for i, row in enumerate(evidence.journal(output / "attempts.jsonl", frontiers["attempts"])):
        step, route = i // 3, ROUTES[i % 3]
        require(
            (row["step_index"], row["route"], row["episode_id"], row["scope"])
            == (step, route, SCHEDULE[step], "main"),
            "main attempt schedule differs",
        )
        require(
            last_state[route]
            == {"model": row["model_before_sha256"], "memory": row["memory_before_sha256"]},
            "actual main state chain differs",
        )
        last_state[route] = {
            "model": row["model_after_sha256"],
            "memory": row["memory_after_sha256"],
        }
        attempts[route].append(check_attempt(row, manifest, output, launch, evidence))
    for route in NEURAL:
        require(
            last_state[route]
            == {
                "model": boundary["routes"][route]["model_sha256"],
                "memory": boundary["routes"][route]["memory_sha256"],
            },
            "last actual state differs from safe checkpoint",
        )
    coverage = {r: {} for r in NEURAL}
    teachers = {r: {} for r in ROUTES}
    for i, row in enumerate(evidence.journal(output / "teachers.jsonl", frontiers["teachers"])):
        step, route = i // 3, ROUTES[i % 3]
        check_teacher(row, attempts[route][step], manifest, panel=False)
        progress_add(teachers[route], row, attempts[route][step])
        if route in NEURAL:
            coverage_add(coverage[route], row["episode_id"], row["diagnostic"])
    for route in NEURAL:
        require(set(coverage[route]) == set(SKILLS), "missing taught skill")
        for positions in coverage[route].values():
            require(
                all(p["offers"] == 512 for p in positions), "position did not receive512 offers"
            )
        finish_coverage(coverage[route])
    for progress in teachers.values():
        for interval in progress.values():
            finish_coverage(interval["positions"])
    checkpoints, anchors, probe_coverage = [], {}, {r: {} for r in NEURAL}
    for expected, row in zip(
        CHECKPOINTS, evidence.journal(output / "probes.jsonl", frontiers["probes"]), strict=True
    ):
        require(
            row["checkpoint"] == expected and row["live_before"] == row["live_after"],
            "panel index or live state preservation differs",
        )
        outcomes = []
        for item, (route, name) in zip(
            row["rows"], [(r, n) for r in ROUTES for n in CASES], strict=True
        ):
            raw = item["attempt"]
            require(
                (raw["scope"], raw["step_index"], raw["route"], raw["episode_id"])
                == ("probe", expected, route, name),
                "panel roster differs",
            )
            outcome = check_attempt(raw, manifest, output, launch, evidence)
            teacher = item["teacher"]
            require((teacher is not None) is (name in SKILLS), "panel teacher schedule differs")
            outcome["teacher"] = (
                check_teacher(teacher, outcome, manifest, panel=True)
                if teacher is not None
                else None
            )
            if teacher is not None and route in NEURAL:
                if expected == 0:
                    anchors[route, name] = teacher["diagnostic"]["anchors"]
                check_anchor_scores(teacher, anchors[route, name])
                coverage_add(probe_coverage[route], name, teacher["diagnostic"])
            outcomes.append(outcome)
        checkpoints.append(
            {"checkpoint": expected, "outcomes": outcomes, "clone_receipts": row["clone_receipts"]}
        )
    required_counts = {"completed_attempts": 3156, "completed_teachers": 3114, "executions": 1557}
    require(
        all(
            report["counts"][k] == n and boundary["host"]["counts"][k] == n
            for k, n in required_counts.items()
        ),
        "complete counters differ",
    )
    compact_results = [
        {k: attempts[r][i][k] for k in ("step_index", "route", "episode_id", "correct")}
        for i in range(1024)
        for r in ROUTES
    ]
    require(report["results"] == compact_results == boundary["host"]["results"], "results binding")
    require(evidence.read(directory / "HEAD.json") == head_raw, "HEAD changed during saved review")
    require(
        evidence.read(output / "completion.json") == completion_raw,
        "completion changed during saved review",
    )
    acquisition, metrics, span = {}, {}, {}
    for route in ROUTES:
        final = [
            r
            for cp in checkpoints
            if cp["checkpoint"] in (768, 1024)
            for r in cp["outcomes"]
            if r["route"] == route and r["episode_id"] in SKILLS
        ]
        require(len(final) == 4, "final acquisition denominator")
        acquisition[route] = {
            "both_skills_at_final_two_checkpoints": all(r["correct"] for r in final),
            "complete_response_denominator": 4,
            "expansion_released": False,
        }
        require(report["acquisition"][route] == acquisition[route], "reported acquisition differs")
        metrics[route] = {
            "main": metric(attempts[route]),
            "checkpoints": {
                str(cp["checkpoint"]): metric([r for r in cp["outcomes"] if r["route"] == route])
                for cp in checkpoints
            },
        }
        span[route] = {
            "main": [
                {
                    "step_index": r["step_index"],
                    "span_correct": r["span_correct"],
                    "complete_correct": r["correct"],
                }
                for r in attempts[route]
                if r["episode_id"] == "A00-E"
            ],
            "checkpoints": [
                {
                    "checkpoint": cp["checkpoint"],
                    "span_correct": next(
                        r["span_correct"]
                        for r in cp["outcomes"]
                        if r["route"] == route and r["episode_id"] == "A00-E"
                    ),
                }
                for cp in checkpoints
            ],
        }
    interference, collisions = _panel_analysis(checkpoints)
    scoped_counts = {}
    for route in ROUTES:
        scoped_counts[route] = {}
        for scope in ("main", "probe"):
            scope_attempts = (
                attempts[route]
                if scope == "main"
                else [r for cp in checkpoints for r in cp["outcomes"] if r["route"] == route]
            )
            if scope == "main":
                summaries = list(teachers[route].values())
                teacher_count = sum(r["teacher_occasions"] for r in summaries)
                reader_symbols = sum(r["reader_symbols"] for r in summaries)
                motor_labels = sum(r["motor_labels"] for r in summaries)
                teacher_draws = sum(r["actual_teacher_draws"] for r in summaries)
            else:
                scope_teachers = [r["teacher"] for r in scope_attempts if r["teacher"] is not None]
                teacher_count = len(scope_teachers)
                reader_symbols = sum(r["reader_offers"] for r in scope_teachers)
                motor_labels = sum(r["action_labels"] for r in scope_teachers)
                teacher_draws = sum(
                    3 * r["diagnostic"]["offers"] + r["diagnostic"]["literal_offers"]
                    for r in scope_teachers
                    if route in NEURAL
                )
            counts = {
                "responses": len(scope_attempts),
                "executions": sum(r["episode_id"] == "A00-E" for r in scope_attempts),
                "teacher_occasions": teacher_count,
                "reader_symbols": reader_symbols,
                "motor_labels": motor_labels,
                "public_input_scan_offers": sum(r["scan_offers"] for r in scope_attempts),
                "free_ticks": sum(r["free_decisions"] for r in scope_attempts),
                "acting_draws": sum(r["controller_draws"] for r in scope_attempts),
                "actual_teacher_draws": teacher_draws,
                "new_source_offers": 0,
            }
            if route in NEURAL:
                counts["teacher_raw_symbols"] = counts["reader_symbols"] - teacher_count
                counts["teacher_eos_symbols"] = teacher_count
                expected_offers, expected_labels = (17920, 1536) if scope == "main" else (245, 21)
                require(
                    (counts["reader_symbols"], counts["motor_labels"])
                    == (expected_offers, expected_labels),
                    "scoped neural exposure differs",
                )
            scoped_counts[route][scope] = counts
    panel_controls = [
        {
            "checkpoint": cp["checkpoint"],
            "rows": [
                paired(
                    next(
                        r
                        for r in cp["outcomes"]
                        if r["route"] == "online" and r["episode_id"] == name
                    ),
                    next(
                        r
                        for r in cp["outcomes"]
                        if r["route"] == "learning_disabled" and r["episode_id"] == name
                    ),
                )
                for name in CASES
            ],
        }
        for cp in checkpoints
    ]
    return {
        "schema": "integrated-local-learning-saved-worker-review-v1",
        "passed": True,
        "status": "passed",
        "rule": report["rule"],
        "worker": str(output),
        "manifest_sha256": manifest_sha256,
        "boundary_sha256": head["sha256"],
        "completion_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "source_sha256": source_map,
        "initialization": initialization,
        "journal_inventory": list(frontiers.values()),
        "counts": required_counts,
        "scoped_counts": scoped_counts,
        "main_attempts": attempts,
        "teacher_progress": teachers,
        "practice_detail": {
            "path": "teachers.jsonl",
            "sha256": frontiers["teachers"]["sha256"],
            "scope": "every raw scalar row remains in authenticated journal",
        },
        "effective_coverage": coverage,
        "diagnostic_coverage": probe_coverage,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "complete_acquisition": acquisition,
        "span_retention": span,
        "fixed_site_interference": interference,
        "activity_target_collisions": collisions,
        "enabled_disabled_checkpoints": panel_controls,
        "phase_seconds": report["phase_seconds"],
        "timing_scope": report["timing_scope"],
        "enabled_disabled_main": [
            paired(a, b)
            for a, b in zip(attempts["online"], attempts["learning_disabled"], strict=True)
        ],
        "scope": (
            "Authenticated scalar evidence and actual execution grades; compact omitted "
            "vectors are not reconstructed. Fixed-site changes are not unique scan/teacher "
            "attribution; conditional scores are not free-response probabilities."
        ),
        "model_calls": 0,
        "teacher_calls": 0,
        "tensor_decodes": 0,
        "fresh_restore_claim": False,
    }


def _panel_analysis(checkpoints):
    interference, collisions = [], []
    for cp in checkpoints:
        for route in NEURAL:
            groups = defaultdict(list)
            for outcome in cp["outcomes"]:
                if outcome["route"] != route or outcome["episode_id"] not in SKILLS:
                    continue
                for row in outcome["teacher"]["diagnostic"]["anchors"]:
                    groups[row["activity_sha256"]].append(
                        {
                            "episode_id": outcome["episode_id"],
                            "position": row["position"],
                            "target": row["target"],
                        }
                    )
            for identity, members in groups.items():
                if len({m["target"] for m in members}) > 1:
                    collisions.append(
                        {
                            "checkpoint": cp["checkpoint"],
                            "route": route,
                            "activity_sha256": identity,
                            "members": members,
                            "scope": "same diagnostic emission activity; different targets",
                        }
                    )
    for left, right in pairwise(checkpoints):
        for route in NEURAL:
            for name in SKILLS:
                before, after = [
                    next(
                        r for r in cp["outcomes"] if r["route"] == route and r["episode_id"] == name
                    )
                    for cp in (left, right)
                ]
                positions = []
                for a, b in zip(
                    before["teacher"]["fixed_anchor_scores"]["positions"],
                    after["teacher"]["fixed_anchor_scores"]["positions"],
                    strict=True,
                ):
                    require(score_identity(a) == score_identity(b), "anchor changed across block")
                    positions.append(
                        {
                            "position": a["position"],
                            "target": a["target"],
                            "activity_sha256": a["activity_sha256"],
                            "fixed_competitor": a["fixed_competitor"],
                            "target_probability_delta": b["target_probability"]
                            - a["target_probability"],
                            "target_log_probability_delta": b["target_log_probability"]
                            - a["target_log_probability"],
                            "target_current_delta": b["target_current"] - a["target_current"],
                            "fixed_competitor_margin_delta": b["fixed_competitor_margin"]
                            - a["fixed_competitor_margin"],
                        }
                    )
                exposed = Counter(SCHEDULE[left["checkpoint"] : right["checkpoint"]])
                interference.append(
                    {
                        "route": route,
                        "episode_id": name,
                        "from": left["checkpoint"],
                        "to": right["checkpoint"],
                        "corrections": dict(exposed),
                        "positions": positions,
                        "correct_before": before["correct"],
                        "correct_after": after["correct"],
                        "scope": (
                            "total weight change at fixed CP0 sites; no unique cause attribution"
                        ),
                    }
                )
    return interference, collisions


def review(output, *, manifest_sha256, deadline=None):
    """Read one complete worker once per journal; never invoke its runtime."""
    started = time.perf_counter()
    evidence = Evidence(deadline if deadline is not None else started + 300)
    try:
        result = _review(output, manifest_sha256, evidence)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        result = {
            "schema": "integrated-local-learning-saved-worker-review-v1",
            "passed": False,
            "status": "failed",
            "worker": str(output),
            "error": {"type": type(error).__name__, "message": str(error)},
            "scope": "No missing or unauthenticated completion inferred",
            "model_calls": 0,
            "teacher_calls": 0,
            "tensor_decodes": 0,
        }
    result["read_work"] = {
        "bytes": evidence.bytes,
        "passes": evidence.passes,
        "journal_rows": evidence.rows,
        "exact": not evidence.pending_read,
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def compare(predecessor, current):
    """Contrasts require authenticated worker results and explicit actual inputs."""
    require(predecessor["passed"] and current["passed"], "both complete saved workers required")
    require(
        (predecessor["rule"], current["rule"]) == ("predecessor", "current")
        and predecessor["manifest_sha256"] == current["manifest_sha256"],
        "comparison pairing",
    )
    for route in NEURAL:
        left, right = [
            worker["initialization"]["initial_routes"][route] for worker in (predecessor, current)
        ]
        require(
            all(
                left[key] == right[key]
                for key in (
                    "numeric_model_sha256",
                    "memory_sha256",
                    "controller_rng_sha256",
                    "experience",
                    "reader_experience",
                    "baseline_mean",
                    "baseline_count",
                )
            ),
            "matched retained start differs",
        )
    require(
        predecessor["initialization"]["baseline_record"]
        == current["initialization"]["baseline_record"],
        "continued reference differs",
    )
    contrasts = {
        route: [
            paired(a, b)
            for a, b in zip(
                predecessor["main_attempts"][route], current["main_attempts"][route], strict=True
            )
        ]
        for route in ROUTES
    }
    panels = []
    for old, new in zip(predecessor["checkpoints"], current["checkpoints"], strict=True):
        require(old["checkpoint"] == new["checkpoint"], "comparison checkpoint mismatch")
        panels.append(
            {
                "checkpoint": old["checkpoint"],
                "rows": [
                    paired(a, b) for a, b in zip(old["outcomes"], new["outcomes"], strict=True)
                ],
            }
        )
    return {
        "schema": "integrated-local-learning-saved-comparison-v1",
        "passed": True,
        "manifest_sha256": current["manifest_sha256"],
        "old_current_main": contrasts,
        "old_current_checkpoints": panels,
        "workers": {"predecessor": predecessor, "current": current},
        "scope": "Behavior equality only for identical actual input and neural RNG; no reseeding",
        "model_calls": 0,
        "teacher_calls": 0,
        "tensor_decodes": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    if DRAFT:
        raise RuntimeError("DRAFT saved reviewer requires source review")
    started = time.perf_counter()
    deadline = started + 300
    old = review(args.predecessor, manifest_sha256=args.manifest_sha256, deadline=deadline)
    new = review(args.current, manifest_sha256=args.manifest_sha256, deadline=deadline)
    try:
        result = (
            compare(old, new)
            if old["passed"] and new["passed"]
            else {
                "schema": "integrated-local-learning-saved-comparison-v1",
                "passed": False,
                "workers": {"predecessor": old, "current": new},
            }
        )
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        result = {
            "schema": "integrated-local-learning-saved-comparison-v1",
            "passed": False,
            "workers": {"predecessor": old, "current": new},
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    result["elapsed_seconds"] = time.perf_counter() - started
    source = Path(__file__).read_bytes()
    result["reviewer_sha256"] = hashlib.sha256(source).hexdigest()
    result["publication_source_read_bytes"] = len(source)
    raw = encode(result)
    artifact_root = (args.artifact_root or args.output.parent).resolve()
    require(
        args.output.resolve().is_relative_to(artifact_root), "output outside shared artifact root"
    )
    used = sum(p.stat().st_size for p in artifact_root.rglob("*") if p.is_file())
    free = shutil.disk_usage(artifact_root).free
    require(
        used + len(raw) <= OUTPUT_LIMIT and free - len(raw) >= FREE_RESERVE,
        "shared64MiB/free6GiB publication admission",
    )
    require(len(raw) <= OUTPUT_LIMIT and time.perf_counter() < deadline, "saved result budget")
    with args.output.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    used = sum(p.stat().st_size for p in artifact_root.rglob("*") if p.is_file())
    free = shutil.disk_usage(artifact_root).free
    require(
        used <= OUTPUT_LIMIT and free >= FREE_RESERVE and time.perf_counter() < deadline,
        "postpublication resource bound",
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "shared_artifact_bytes": used,
                "free_bytes": free,
                "elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
