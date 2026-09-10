"""Attribute saved literal increments; never construct, run or update a learner.

This is historical arithmetic at recorded activities and probabilities, not a
counterfactual training run. Only selected literal distributions are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_SHA = "3783e751a40c0150da8adc091a976d91ab70c879dc1b2a5c6ffc020c27e8e262"
BOUNDARY_SHA = "3408eecadf057ce322f5063a879584e82ea993c433e74d1de290e5114e1ae677"
CHECKPOINTS = (0, 8, 16, 32, 64)
ROUTES = ("online", "learning_disabled", "simple")
SHAPE = (257, 1072)
COMPONENTS = ("teacher_A00-R", "teacher_A00-E", "scan", "public")
ATOL = 2e-11


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = dict(items)
    require(len(result) == len(items), "duplicate saved coefficient/count key")
    return result


def softmax(logits):
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def activity(tick, previous):
    """Rebuild the source's fixed sparse emission wiring, including hash bytes."""
    h, regs, prior = tick["h_after"], tick["register_tokens"], tick["prior_token"]
    require(type(h) is int and 0 <= h < 32 and len(regs) == 2, "invalid emission state")
    require(type(previous) is int and 0 <= previous <= 257, "invalid previous symbol")
    require(all(v is None or type(v) is int and 0 <= v <= 256 for v in regs), "invalid registers")
    category = tick["category"]
    require(category in ("scan", "free", "teacher"), "invalid tick category")
    require((prior is None) == (category == "scan"), "scan/output prior mismatch")
    require(prior is None or type(prior) is int and 0 <= prior <= 256, "invalid prior")
    indices = [h, 32 + previous, 291 + (258 if regs[0] is None else regs[0]),
               550 + (258 if regs[1] is None else regs[1])]
    indices += [809 + i for i, token in enumerate(regs) if token is not None]
    indices += [811 + (1 if category == "scan" else 2)]
    if prior is not None:
        indices.append(814 + prior)
    indices.append(1071)
    indices = np.asarray(indices, dtype="<i8")
    values = np.ones(len(indices), dtype="<f8") / math.sqrt(len(indices))
    digest = hashlib.sha256(b"humanbrain.recurrent-controller.v1\0")
    for name, raw, metadata in (
        ("indices", indices.tobytes(), f"(({len(indices)},), torch.int64)"),
        ("values", values.tobytes(), f"(({len(indices)},), torch.float64)"),
        ("width", b"1072", ""),
    ):
        digest.update(name.encode())
        digest.update(metadata.encode())
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    identity = digest.hexdigest()
    draws = tick["draws"]
    require([d["population"] for d in draws[:3]] == ["context", "gate", "mode"], "draw order")
    require(draws[2]["activity_sha256"] == identity, "mode emission activity differs")
    selected = draws[2]["index"] == 3
    require(len(draws) == 3 + selected, "literal branch inclusion differs")
    if selected:
        require(draws[3]["population"] == "literal" and draws[3]["activity_sha256"] == identity,
                "literal emission activity differs")
    return indices, values


def distribution(tick):
    draw = tick["draws"][3]
    logits, probabilities = (np.asarray(draw[k], dtype=np.float64) for k in ("logits", "probabilities"))
    require(logits.shape == probabilities.shape == (257,), "literal population size")
    require(np.isfinite(logits).all() and np.isfinite(probabilities).all()
            and (probabilities > 0).all(), "nonfinite literal distribution")
    require(np.allclose(softmax(logits), probabilities, rtol=1e-12, atol=1e-14),
            "saved literal probabilities differ from currents")
    require(type(draw["index"]) is int and 0 <= draw["index"] <= 256, "selected literal cell")
    require(tick["emitted_token"] == draw["index"], "literal emission mismatch")
    return logits, probabilities


def accumulate(bank, indices, values, probabilities, target):
    error = -probabilities.copy()
    error[target] += 1.0
    bank[:, indices] += error[:, None] * values[None, :]


def trace_events(prediction, teacher):
    previous, field, offset = 257, None, -1
    for tick in prediction["scan_steps"]:
        require(tick["category"] == "scan", "scan trace category")
        if tick["field"] != field:
            require(tick["offset"] == 0, "missing start of scan field")
            previous, offset = 257, -1
        require(tick["offset"] == offset + 1, "missing scan position")
        yield tick, previous, tick["observed_token"]
        previous, field, offset = tick["observed_token"], tick["field"], tick["offset"]
    for category, steps in (("free", prediction["steps"]), ("teacher", teacher["teacher_steps"])):
        previous = 257
        expected_prefix = b""
        for position, step in enumerate(steps):
            tick = step["controller"]
            require(tick["category"] == category, "output trace category")
            prefix = bytes.fromhex(step["prefix_hex"])
            require(prefix == expected_prefix and step["output_step"] == position,
                    "output prefix continuity")
            target = step["observed_token"] if category == "teacher" else tick["emitted_token"]
            if category == "free":
                require(step["token"] == target, "saved free token differs")
            require(type(target) is int and 0 <= target <= 256, "invalid trace symbol")
            yield tick, previous, target
            previous = target
            require(target != 256 or position == len(steps) - 1, "trace continues after EOS")
            if target != 256:
                expected_prefix += bytes([target])


def resolution(prediction, teacher):
    """Reconstruct proposed increments with the actual held per-tick probabilities."""
    update = teacher["controller"]
    require(update["learning_enabled"] is teacher["learning_enabled"], "publication gate differs")
    reward = update["rewards"]
    scan_count, teacher_count = len(prediction["scan_steps"]), len(teacher["teacher_steps"])
    require(scan_count > 0 and teacher_count > 0, "empty live observation category")
    edit = teacher["episode_id"] == "A00-E"
    require(teacher["episode_id"] in ("A00-R", "A00-E"), "unexpected taught task")
    public = reward["public"]
    require(type(public) is int and public in (-1, 0) if edit else public is None,
            "invalid actual public consequence")
    target = bytes.fromhex(teacher["teacher"]["replacement_hex" if edit else "target_hex"])
    require([s["observed_token"] for s in teacher["teacher_steps"]] == [*target, 256],
            "teacher steps differ from complete offered answer")
    k = 3 if edit else 2
    coefficients = pairs(reward["coefficients"])
    expected = {"literal_scan": .01 / (k * scan_count),
                "literal_teacher": .01 / (k * teacher_count),
                "public": .01 * (public or 0) / (k * (scan_count + teacher_count + 129))}
    require(all(coefficients[n] == value for n, value in expected.items()), "literal coefficients differ")
    require(pairs(reward["external_offers"]) == {"scan": scan_count, "teacher": teacher_count},
            "external offer counts differ")
    direct_scan, direct_teacher, eligible, positive = (np.zeros(SHAPE) for _ in range(4))
    counts = {"scan": 0, "free": 0, "teacher": 0}
    for tick, previous, target in trace_events(prediction, teacher):
        indices, values = activity(tick, previous)
        category = tick["category"]
        selected = len(tick["draws"]) == 4
        if category != "free":
            credit = tick["credit"]
            require(credit["observed_token"] == target and credit["literal_taught"] is selected,
                    "observed target or inclusion mismatch")
        if not selected:
            continue
        counts[category] += 1
        _, probabilities = distribution(tick)
        if category != "free":
            require(math.isclose(credit["target_probability"], probabilities[target], rel_tol=1e-12),
                    "direct teacher held probability mismatch")
            accumulate(direct_scan if category == "scan" else direct_teacher,
                       indices, values, probabilities, target)
            if category == "teacher":
                positive[target, indices] += values
        if category != "teacher":
            accumulate(eligible, indices, values, probabilities, tick["emitted_token"])
    require(pairs(reward["literal_offers"]) == {"scan": counts["scan"], "teacher": counts["teacher"]},
            "literal offer counts differ")
    parts = {"scan": direct_scan * expected["literal_scan"],
             "teacher_" + teacher["episode_id"]: direct_teacher * expected["literal_teacher"],
             "public": eligible * expected["public"]}
    require(all(np.max(np.abs(part.sum(axis=0))) < ATOL for part in parts.values()),
            "local error no longer sums to zero across output cells")
    return parts, positive * expected["literal_teacher"], counts


def project(bank, site):
    return (bank[:, site["indices"]] * site["values"]).sum(axis=1)


def checkpoint_report(banks, positive, sites, checkpoint):
    result = []
    for name, site in sites.items():
        current = {key: project(bank, site) for key, bank in banks.items()}
        total = sum(current.values())
        target = site["target"]
        row = {"site": name, "target": target, "checkpoint": checkpoint,
               "target_current_by_component": {k: float(v[target]) for k, v in current.items()},
               "target_current_total": float(total[target]),
               "teacher_positive_target_current": float(project(positive, site)[target]),
               "full_logits_independently_anchored": site["logits"][0] is not None}
        teacher_current = sum(v[target] for k, v in current.items() if k.startswith("teacher_"))
        row["teacher_negative_target_current"] = float(teacher_current - row["teacher_positive_target_current"])
        if row["full_logits_independently_anchored"]:
            initial, final = site["logits"][0], site["logits"][checkpoint]
            residual = total - (final - initial)
            require(np.max(np.abs(residual)) <= ATOL, "reconstructed literal logit changes do not close")
            predicted = softmax(initial + total)
            require(np.max(np.abs(predicted - softmax(final))) <= ATOL, "reconstructed softmax does not close")
            rivals = initial.copy()
            rivals[target] = -np.inf
            rival = int(np.argmax(rivals))
            row.update(max_all_cell_logit_residual=float(np.max(np.abs(residual))),
                       initial_top_competing_cell=rival,
                       target_margin_by_component={k: float(v[target] - v[rival]) for k, v in current.items()},
                       observed_target_probability=float(softmax(final)[target]),
                       reconstructed_target_probability=float(predicted[target]),
                       observed_target_log_probability_delta=float(np.log(softmax(final)[target])
                                                                  - np.log(softmax(initial)[target])))
        result.append(row)
    return result


def run(evidence, destination, *, work=None):
    started = time.perf_counter()
    if work is None:
        work = {}
    work.update(metadata_bytes=0, source_bytes=0, journal_bytes=0, journal_rows=0)
    evidence, destination = Path(evidence).resolve(), Path(destination).resolve()
    require(not destination.exists(), "attribution output must be new")

    def read(path, expected=None, scope="metadata_bytes"):
        with Path(path).open("rb") as stream:
            raw = stream.read(16 * 1024**2 + 1)
        work[scope] += len(raw)
        require(len(raw) <= 16 * 1024**2, "oversized metadata/source")
        if expected:
            require(hashlib.sha256(raw).hexdigest() == expected, "authenticated metadata/source differs")
        return raw

    report = json.loads(read(ROOT / "benchmarks/repeated_teaching_acquisition_v1.run.json", RUN_SHA))
    require(report["status"] == "completed" and report["restore"]["status"] == "passed", "unfinished source run")
    head = json.loads(read(evidence / "checkpoints/HEAD.json"))
    require(head["sha256"] == BOUNDARY_SHA == report["final"]["boundary_sha256"], "wrong final boundary")
    boundary = json.loads(read(evidence / "checkpoints/boundary-000000000.json", BOUNDARY_SHA))
    config = json.loads(read(evidence / "checkpoints/config.json", boundary["config_sha256"]))
    manifest = json.loads(read(evidence / "manifest.json", config["manifest_sha256"]))
    sources = config["source_sha256"]
    for path, sha in sources.items():
        read(ROOT / path, sha, "source_bytes")
    frontiers = {item["kind"]: item for item in boundary["frontiers"]["journals"]}

    def rows(kind):
        digest, count, size = hashlib.sha256(), 0, 0
        with (evidence / (kind + ".jsonl")).open("rb") as stream:
            while raw := stream.readline(64 * 1024**2 + 1):
                require(time.perf_counter() - started < 570, "saved attribution deadline")
                require(raw.endswith(b"\n") and len(raw) <= 64 * 1024**2, "incomplete/oversized journal row")
                digest.update(raw)
                size += len(raw)
                count += 1
                work["journal_bytes"] += len(raw)
                work["journal_rows"] += 1
                yield json.loads(raw)
        expected = frontiers[kind]
        require((size, count, digest.hexdigest()) == (expected["bytes"], expected["rows"], expected["sha256"]),
                "journal differs from authenticated final frontier")

    sites, probe_count = {}, 0
    for cp, row in zip(CHECKPOINTS, rows("probes"), strict=True):
        require(row["checkpoint"] == cp and row["live_before"] == row["live_after"], "probe boundary changed")
        require(len(row["rows"]) == 12, "incomplete probe roster")
        for item, coordinate in zip(row["rows"], ((r, e) for r in ROUTES for e in manifest["checkpoint_cases"]), strict=True):
            attempt, teacher = item["attempt"], item["teacher"]
            require((attempt["route"], attempt["episode_id"]) == coordinate, "probe order")
            if attempt["route"] == "simple" or teacher is None:
                continue
            require(teacher["learning_enabled"] is False, "probe published learning")
            for step, diagnostic in zip(teacher["teacher_steps"], teacher["diagnostic"]["positions"], strict=True):
                prefix, tick = bytes.fromhex(step["prefix_hex"]), step["controller"]
                indices, values = activity(tick, prefix[-1] if prefix else 257)
                name = attempt["episode_id"] + ":" + str(step["output_step"])
                logits = distribution(tick)[0] if len(tick["draws"]) == 4 else None
                if name not in sites:
                    require(cp == 0 and attempt["route"] == "online", "missing initial diagnostic site")
                    sites[name] = {"indices": indices, "values": values, "target": step["observed_token"], "logits": {}}
                site = sites[name]
                require(np.array_equal(indices, site["indices"]) and np.array_equal(values, site["values"])
                        and step["observed_token"] == site["target"], "diagnostic activity/target changed")
                if attempt["route"] == "online":
                    if cp != 0:
                        require((logits is None) == (site["logits"][0] is None), "online diagnostic branch differs")
                    site["logits"][cp] = logits
                else:
                    require((logits is None) == (site["logits"][0] is None), "disabled branch differs")
                    if logits is not None:
                        require(np.array_equal(logits, site["logits"][0]), "disabled literal currents changed")
                require(diagnostic["activity_sha256"] == tick["draws"][2]["activity_sha256"], "observer activity differs")
        probe_count += 1
    require(len(sites) == 35 and probe_count == 5, "incomplete intended-prefix diagnostic coverage")
    require(sum(s["logits"][0] is not None for s in sites.values()) == 10, "unexpected full-current anchor count")
    banks = {k: np.zeros(SHAPE) for k in COMPONENTS}
    positive = np.zeros(SHAPE)
    checkpoints = [checkpoint_report(banks, positive, sites, 0)]
    totals = {r: {k: 0 for k in ("scan", "free", "teacher")} for r in ROUTES[:2]}
    count = 0
    previous_models = {}
    for index, (attempt, teacher) in enumerate(zip(rows("attempts"), rows("teachers"), strict=True)):
        step, route = index // 3, ROUTES[index % 3]
        name = manifest["schedule"][step]["episode_id"]
        for record in (attempt, teacher):
            require((record["step_index"], record["route"], record["episode_id"], record["scope"])
                    == (step, route, name, "main"), "main attempt/teacher pairing differs")
        require(attempt["model_before_sha256"] == teacher["model_before_sha256"]
                and attempt["model_after_sha256"] == teacher["model_after_sha256"], "publication pairing differs")
        if route in previous_models:
            require(attempt["model_before_sha256"] == previous_models[route], "broken main model chain")
        previous_models[route] = attempt["model_after_sha256"]
        if route != "simple":
            require(teacher["learning_enabled"] is (route == "online"), "live learning gate differs")
            if name.endswith("E"):
                execution = attempt["execution"]
                failed = (not execution["accepted"] or execution["returncode"] != 0
                          or execution["timeout"] or not execution["capture_complete"]
                          or not execution["files_complete"] or execution["output_limit_exceeded"])
                public = -int(failed and attempt["proposal"]["operation"] != "STOP")
                require(teacher["controller"]["rewards"]["public"] == public, "public reward differs from execution")
            prediction = attempt["prediction"] if name.endswith("R") else attempt["prediction"]["payload"]
            parts, match, counts = resolution(prediction, teacher)
            for key, value in counts.items():
                totals[route][key] += value
            if route == "online":
                for key, value in parts.items():
                    banks[key] += value
                positive += match
        if route == "simple" and step + 1 in CHECKPOINTS:
            checkpoints.append(checkpoint_report(banks, positive, sites, step + 1))
        count += 1
    require(count == 192 and len(checkpoints) == 5, "incomplete main schedule")
    require(all(v["teacher"] == 273 for v in totals.values()), "teacher literal coverage changed")
    require(work["journal_bytes"] == 510318993 and work["journal_rows"] == 389, "one-pass journal scope changed")
    result = {"schema": "repeated-teaching-literal-attribution-v1", "status": "passed",
              "scope": "additive historical literal-current accounting at fixed recorded activities; not a counterfactual training run",
              "run_report_sha256": RUN_SHA, "boundary_sha256": BOUNDARY_SHA,
              "source_sha256": hashlib.sha256(read(__file__, scope="source_bytes")).hexdigest(),
              "absolute_closure_tolerance": ATOL, "independent_full_logit_sites": 10,
              "projected_only_sites": 25, "selected_literal_counts": totals,
              "checkpoints": checkpoints, "work": work, "model_calls": 0, "teacher_calls": 0,
              "payload_decodes": 0, "learning_publications": 0, "rng_draws": 0,
              "elapsed_seconds": time.perf_counter() - started}
    raw = (json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    require(len(raw) <= 16 * 1024**2, "attribution report exceeds declared cap")
    require(shutil.disk_usage(destination.parent).free - len(raw) >= 8 * 1024**3, "publication free-space floor")
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(shutil.disk_usage(destination.parent).free >= 8 * 1024**3, "postpublication free-space floor")
    require(time.perf_counter() - started < 600, "attribution completion deadline")
    print(json.dumps({"status": "passed", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "work": work, "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work, started = {}, time.perf_counter()
    try:
        run(args.evidence, args.output, work=work)
    except BaseException as error:
        print(json.dumps({"status": "failed", "type": type(error).__name__, "error": str(error),
                          "work": work, "seconds": time.perf_counter() - started,
                          "model_calls": 0, "teacher_calls": 0, "automatic_retry": False}), flush=True)
        raise
