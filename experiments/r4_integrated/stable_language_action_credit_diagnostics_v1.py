"""Bounded supplemental extraction from authenticated saved credit journals only.

No model, tensor decoder, task builder, predictor or executor is invoked. This
analysis is separate from the frozen comparison and requires a passed saved audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_v8_review as saved

ROOT = Path(__file__).resolve().parents[2]
EXECUTION_REVISION = "5622f600921e6811d66254e25c8e956a31cd19ab"
SECONDS, PEAK_MIB = 600, 2048
OUTPUT_LIMIT, METADATA_LIMIT = 128 * 1024**2, 16 * 1024**2
ARTIFACT_LIMIT, FREE_RESERVE = 46 * 1024**3, 8 * 1024**3
CHECKPOINTS, NEURAL = (0, 16, 32, 48), ("online", "learning_disabled")
KINDS = ("raw_content", "eos", "status")
require = saved.require


def _empty():
    return {"offers": 0, "sampled_matches": 0, "conditional_match_sum": 0.0,
            "literal_offers": 0, "literal_sampled_matches": 0,
            "literal_target_probability_sum": 0.0, "literal_ce_sum": 0.0,
            "direct_literal_proposed_offers": 0, "direct_literal_published_offers": 0,
            "sampled_modes": [0, 0, 0, 0]}


def _means(values):
    result = dict(values)
    n, literal = values["offers"], values["literal_offers"]
    result.update(
        sampled_match_rate=values["sampled_matches"] / n if n else None,
        conditional_match_mean=values["conditional_match_sum"] / n if n else None,
        sampled_reward_mean=2 * values["sampled_matches"] / n - 1 if n else None,
        conditional_reward_mean=2 * values["conditional_match_sum"] / n - 1 if n else None,
        literal_inclusion_rate=literal / n if n else None,
        literal_target_probability_mean=values["literal_target_probability_sum"] / literal if literal else None,
        literal_ce_mean=values["literal_ce_sum"] / literal if literal else None,
        literal_ce_per_external_offer=values["literal_ce_sum"] / n if n else None)
    return result


def _add(destination, source):
    for key, value in source.items():
        if key == "sampled_modes":
            destination[key] = [a + b for a, b in zip(destination[key], value, strict=True)]
        else:
            destination[key] += value


def _offer(tick, target, status, *, causal, published):
    require(type(target) is int and 0 <= target <= 256, "invalid saved offered symbol")
    mode = tick["draws"][2]["index"]
    require(type(mode) is int and 0 <= mode <= 3
            and len(tick["draws"]) == 3 + int(mode == 3), "actual literal branch inclusion")
    match = int(tick["emitted_token"] == target)
    value, probability, loss = _empty(), None, None
    if mode == 3:
        draw = tick["draws"][3]
        require(draw["population"] == "literal" and len(draw["probabilities"]) == 257
                and len(draw["logits"]) == 257, "saved literal population geometry")
        probability = draw["probabilities"][target]
        require(type(probability) in (int, float) and math.isfinite(probability)
                and 0 <= probability <= 1, "finite literal target probability")
        logits = draw["logits"]
        require(all(type(v) in (float, int) and math.isfinite(v) for v in logits), "finite literal logits")
        maximum = max(logits)
        loss = maximum - logits[target] + math.log(sum(math.exp(v - maximum) for v in logits))
        require(math.isfinite(loss) and loss >= 0, "finite stable literal cross entropy")
        conditional = probability
        value.update(literal_offers=1, literal_sampled_matches=match,
                     literal_target_probability_sum=probability, literal_ce_sum=loss,
                     direct_literal_proposed_offers=int(causal),
                     direct_literal_published_offers=int(causal and published))
    else:
        conditional = float(match)
    if causal:
        credit = tick["credit"]
        require(credit["observed_token"] == target and credit["mode"] == mode
                and credit["sampled_reward"] == 2 * match - 1, "actual conditional credit target")
        saved.close(credit["signal"], 2 * conditional - 1, "identical old/causal conditional metric")
        require(credit["literal_taught"] is (mode == 3), "literal proposal inclusion receipt")
        if mode == 3:
            saved.close(credit["target_probability"], probability, "held pretarget literal probability")
            saved.close(credit["literal_loss"], loss, "stable literal CE receipt")
        else:
            require(credit["target_probability"] is None and credit["literal_loss"] is None,
                    "unselected literal distribution invented")
    value.update(offers=1, sampled_matches=match, conditional_match_sum=conditional)
    value["sampled_modes"][mode] = 1
    return "status" if status else "eos" if target == 256 else "raw_content", value


def _case(row, entry, *, rule, scope, checkpoint, journal_kind, admit):
    route = row["route"]
    if route not in NEURAL:
        require(route == "simple", "unknown saved route")
        return None
    causal, output = rule == "causal", None
    if journal_kind == "ingests":
        category = "source_scan"
        feedback = row["source_feedback"]
        offers = ((tick, tick["observed_token"], tick["status_offer"])
                  for tick in feedback["source_steps"])
    elif journal_kind == "teachers":
        category = "teacher"
        feedback = row["reader_feedback"]
        offers = ((step["controller"], step["observed_token"], False) for step in feedback["steps"])
    else:
        category = "public_scan"
        prediction = row["prediction"]
        if prediction is not None:
            prediction = prediction if row["kind"] == "reading" else prediction["payload"]
            offers = ((tick, tick["observed_token"], tick["status_offer"])
                      for tick in prediction["scan_steps"])
            output = {"bytes": len(bytes.fromhex(prediction["answer_hex"])),
                      "stop_reason": prediction["stop_reason"], "decisions": prediction["decisions"],
                      "free_draws": sum(len(step["controller"]["draws"]) for step in prediction["steps"])}
        else:
            offers = iter(())
    published = scope == "main" and route == "online"
    values = {kind: _empty() for kind in KINDS}
    for index, (tick, target, status) in enumerate(offers):
        if index % 128 == 0:
            admit()
        kind, increment = _offer(tick, target, status, causal=causal, published=published)
        _add(values[kind], increment)
    all_offers = _empty()
    for value in values.values():
        _add(all_offers, value)
    result = {"rule": rule, "scope": scope, "checkpoint": checkpoint, "route": route,
              "episode_id": row["episode_id"], "attempt_index": row.get("attempt_index"),
              "category": category, "journal": journal_kind, "row_sha256": entry["sha256"],
              "row_offset": entry["offset"], "row_bytes": entry["bytes"],
              "values": values, "all_offers": _means(all_offers), "output": output}
    if journal_kind == "teachers":
        result.update(teacher=row["teacher"], reader_offers=row["reader_offers"],
                      action_labels=row["action_labels"], attempt_sha256=row["attempt_sha256"],
                      learning_enabled=row["learning_enabled"],
                      teacher_prior_nll={"total": feedback["total_loss"], "mean": feedback["mean_loss"],
                                         "scope": "held inherited prior diagnostic, not controller literal CE"})
        require(sum(v["offers"] for v in values.values()) == row["reader_offers"],
                "complete correction denominator")
    if journal_kind == "attempts":
        result.update(kind=row["kind"], status=row["status"], correct=row["correct"],
                      observation_sha256=saved.digest(saved.encoded(row["observation"])),
                      proposal_signature=row.get("proposal_signature"),
                      probe_group=row.get("probe_group"))
    return result


def extract(args):
    """Authenticate two real journal passes and return compact diagnostic records."""
    started = time.perf_counter()
    deadline = started + SECONDS
    root, output = Path(args.comparison_root).resolve(), Path(args.output).resolve()
    require(root.is_relative_to(ROOT) and output.is_relative_to(root), "supplemental paths must remain inside comparison")
    require(not output.exists(), "supplemental output already exists")
    report = {"schema": "stable-language-action-credit-diagnostics-v1", "status": "incomplete",
              "execution_revision": EXECUTION_REVISION, "comparison_root": str(root),
              "manifest_sha256": args.manifest_sha, "cases": [], "aggregates": [],
              "journals": [], "source_sha256": {}, "review_bindings": {},
              "model_calls": 0, "tensor_decodes": 0, "teacher_replays": 0, "execution_children": 0,
              "budget": {"seconds": SECONDS, "peak_mib": PEAK_MIB, "output_bytes": OUTPUT_LIMIT,
                         "rss_enforcement": "external owned-process monitor required"},
              "io": {"metadata_read_bytes": 0, "source_read_bytes": 0, "journal_index_read_bytes": 0,
                     "journal_parse_read_bytes": 0, "parsed_rows": 0, "parsed_neural_cases": 0,
                     "read_bytes_exact": True, "pending_read": None},
              "timing": {}, "scope": "supplemental saved-only analysis outside frozen comparison ceiling"}
    io = report["io"]
    metadata_budget = 12 * METADATA_LIMIT

    def admit():
        require(time.perf_counter() < deadline, "supplemental diagnostic deadline reached")

    def metadata(path, expected=None):
        admit()
        path = Path(path).resolve()
        size = path.stat().st_size
        require(not path.is_symlink() and size <= METADATA_LIMIT, "bounded metadata file")
        require(io["metadata_read_bytes"] + size <= metadata_budget, "metadata read budget exceeded")
        io["pending_read"] = {"kind": "metadata", "path": str(path),
                              "maximum_bytes": METADATA_LIMIT + 1}
        with path.open("rb") as handle:
            raw = handle.read(METADATA_LIMIT + 1)
        io["metadata_read_bytes"] += len(raw)
        io["pending_read"] = None
        identity = saved.digest(raw)
        require(len(raw) == size and (expected is None or identity == expected), "metadata byte binding changed")
        return json.loads(raw), {"path": str(path), "sha256": identity, "bytes": len(raw)}

    groups = {}
    try:
        review, review_identity = metadata(args.review, args.review_sha)
        failed, failed_identity = metadata(args.initial_failed_review, args.initial_failed_review_sha)
        report["review_bindings"] = {"corrected": review_identity, "initial_failed": failed_identity}
        require(review["passed"] is True and review["status"] == "passed"
                and failed["passed"] is False and failed["status"] == "failed",
                "requires completed corrected audit and retained initial failed audit")
        require(all(Path(item["comparison_root"]).resolve() == root
                    and item["manifest_sha256"] == args.manifest_sha for item in (review, failed)),
                "review comparison/manifest identity")
        require(review["workers"]["causal"]["source_revision"] == EXECUTION_REVISION,
                "different frozen execution revision")
        report["initial_failure"] = {key: failed.get(key) for key in ("error_type", "error", "elapsed_seconds")}
        report["corrected_review_tool_sources"] = review["workers"]["old"]["saved_tool_sources"]
        modules = {Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
                   if name.startswith("experiments.r4_integrated.") and getattr(module, "__file__", None)}
        modules.add(Path(__file__).resolve())
        for path in sorted(modules):
            admit()
            require(path.is_relative_to(ROOT) and path.suffix == ".py", "source closure escaped checkout")
            size = path.stat().st_size
            io["pending_read"] = {"kind": "source", "path": str(path), "maximum_bytes": size + 1}
            with path.open("rb") as handle:
                raw = handle.read(size + 1)
            io["source_read_bytes"] += len(raw)
            io["pending_read"] = None
            require(len(raw) == size, "source size changed during extraction")
            report["source_sha256"][path.relative_to(ROOT).as_posix()] = saved.digest(raw)
        plan, completions = [], {}
        for rule in ("old", "causal"):
            completion, identity = metadata(root / rule / "completion.json")
            require(completion["status"] == "completed" and review["workers"][rule]["complete_run"]
                    and review["workers"][rule]["status"] == "passed", "worker incomplete")
            completions[rule] = identity
            require(review["workers"][rule]["evidence"] == completion["files"], "passed audit inventory changed")
            _, manifest_identity = metadata(root / rule / "manifest.json", args.manifest_sha)
            require(completion["files"]["manifest.json"] == manifest_identity["sha256"], "worker manifest inventory")
            sources, _ = metadata(root / rule / "sources.json", completion["files"]["sources.json"])
            for name, source_identity in report["source_sha256"].items():
                if ROOT / name != Path(__file__).resolve():
                    require(sources.get(name) == source_identity, "parser helper differs from frozen execution inventory")
            names = [(kind + ".jsonl", "main", None, kind, count) for kind, count in
                     (("ingests", 72), ("attempts", 192), ("teachers", 6))]
            names += [(f"probes/cp-{cp}/attempts.jsonl", "checkpoint", cp, "attempts", 84)
                      for cp in CHECKPOINTS]
            for name, scope, cp, kind, count in names:
                path = saved.safe_path(root / rule, name)
                require(path.is_file() and not path.is_symlink(), "journal missing or symlinked")
                plan.append({"path": str(path), "name": name, "rule": rule, "scope": scope,
                             "checkpoint": cp, "kind": kind, "rows": count,
                             "bytes": path.stat().st_size, "sha256": completion["files"][name]})
        report["completion_bindings"] = completions
        expected_bytes = sum(item["bytes"] for item in plan)
        report["budget"].update(journal_bytes_per_pass=expected_bytes,
                                journal_read_bytes=2 * expected_bytes, metadata_read_bytes=metadata_budget,
                                planned_total_read_bytes=2 * expected_bytes + io["metadata_read_bytes"]
                                                         + io["source_read_bytes"])
        report["timing"]["authentication_setup_seconds"] = time.perf_counter() - started
        index_seconds, parse_seconds = 0.0, 0.0
        for item in plan:
            admit()
            path = Path(item["path"])
            tick = time.perf_counter()
            io["pending_read"] = {"kind": "journal_index", "path": str(path),
                                  "planned_bytes": item["bytes"], "maximum_bytes": None}
            entries, frontier = saved.journal(path, partial=False)
            io["journal_index_read_bytes"] += frontier["bytes"]
            io["pending_read"] = None
            require(frontier["bytes"] == item["bytes"] and frontier["sha256"] == item["sha256"]
                    and len(entries) == item["rows"], "complete authenticated journal inventory")
            index_elapsed = time.perf_counter() - tick
            index_seconds += index_elapsed
            tick = time.perf_counter()
            for entry in entries:
                admit()
                io["pending_read"] = {"kind": "journal_parse", "path": str(path),
                                      "offset": entry["offset"], "maximum_bytes": entry["bytes"]}
                row = saved.read_row(entry)
                io["journal_parse_read_bytes"] += entry["bytes"]
                io["parsed_rows"] += 1
                io["pending_read"] = None
                case = _case(row, entry, rule=item["rule"], scope=item["scope"],
                             checkpoint=item["checkpoint"], journal_kind=item["kind"], admit=admit)
                if case is not None:
                    io["parsed_neural_cases"] += 1
                    case["journal_path"] = str(path)
                    for kind, values in case["values"].items():
                        key = (item["rule"], case["route"], item["scope"], item["checkpoint"], case["category"], kind)
                        _add(groups.setdefault(key, _empty()), values)
                    case["values"] = {kind: _means(value) for kind, value in case["values"].items()}
                    report["cases"].append(case)
                del row, case
            require(path.stat().st_size == item["bytes"], "journal size changed during extraction")
            parse_elapsed = time.perf_counter() - tick
            parse_seconds += parse_elapsed
            report["journals"].append({**item, "index_seconds": index_elapsed,
                                       "parse_and_aggregate_seconds": parse_elapsed})
        require(io["journal_index_read_bytes"] == io["journal_parse_read_bytes"] == expected_bytes
                and io["parsed_rows"] == 1212 and io["parsed_neural_cases"] == 808,
                "two complete real passes and fixed case denominators")
        populations = {}
        for key, value in sorted(groups.items(), key=lambda pair: str(pair[0])):
            rule, route, scope, cp, category, kind = key
            if scope == "main":
                audited = review["workers"][rule]["controller_exposures"][route]["by_category"][
                    category]["observed_symbols"][kind]
                require(value["offers"] == audited["offers"] and value["sampled_matches"] == audited["matches"],
                        "extracted main matches differ from passed audit")
            report["aggregates"].append({"rule": rule, "route": route, "scope": scope,
                "checkpoint": cp, "category": category, "symbol_kind": kind, **_means(value)})
            _add(populations.setdefault(key[:-1], _empty()), value)
        report["population_aggregates"] = [
            {"rule": rule, "route": route, "scope": scope, "checkpoint": cp, "category": category,
             "denominator": "all actual external offers in this category/scope, including nonliteral offers",
             **_means(value)} for (rule, route, scope, cp, category), value in populations.items()]
        corrections = [case for case in report["cases"] if case["category"] == "teacher"]
        require(len(corrections) == 8 and sum(case["reader_offers"] for case in corrections) == 140
                and sum(case["action_labels"] for case in corrections) == 12, "actual correction ledger")
        report["corrections"] = corrections
        report["timing"].update(journal_index_seconds=index_seconds,
                                 parse_and_aggregate_seconds=parse_seconds)
        report["interpretation"] = {
            "conditional_match": "p(target) on the actual literal branch, otherwise actual match; identical formula for both rules",
            "literal_ce": "selected literal subset only; nonliteral distribution is never invented; per-external-offer CE includes zero for nonliteral offers",
            "direct_literal_proposed": "causal-rule evaluated literal offer, including disabled proposals; old-rule literal CE is an analysis metric",
            "direct_literal_published": "causal main online only; probes and disabled arms publish no slow state",
            "status": "status field including its framing EOS; disjoint from raw_content and other eos",
            "denominators": "aggregates split disjoint raw/EOS/status offers; population_aggregates sum them once and must not be added again",
            "scan_custody": "main common scan counted once from actual prediction; teacher record contributes only teacher ticks",
            "scope": "main and disposable checkpoint exposures remain separate; single-run diagnostics do not establish task learning or isolate a cause of failure",
            "task_outcomes": "use the authenticated corrected audit for task success, migration, retention and repair; no grading replay"}
        admit()
        report["status"] = "completed"
    except Exception as error:
        if io["pending_read"] is not None:
            io["read_bytes_exact"] = False
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      failure_scope="confirmed completed reads only; pending failed-read bytes are unavailable, with an upper bound only where explicitly present")
    io["confirmed_read_bytes"] = sum(io[key] for key in ("metadata_read_bytes", "source_read_bytes",
                                   "journal_index_read_bytes", "journal_parse_read_bytes"))
    report["timing"]["extraction_seconds"] = time.perf_counter() - started
    return report, started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison-root", "review", "review-sha", "initial-failed-review",
                 "initial-failed-review-sha", "manifest-sha", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report, started = extract(args)
    output, root = Path(args.output).resolve(), Path(args.comparison_root).resolve()
    serialization_started = time.perf_counter()
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    report["timing"]["initial_serialization_seconds"] = time.perf_counter() - serialization_started
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    require(len(raw) <= OUTPUT_LIMIT, "supplemental output cap exceeded")
    used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    free_before = shutil.disk_usage(root).free
    require(used + len(raw) <= ARTIFACT_LIMIT and free_before - len(raw) >= FREE_RESERVE,
            "supplemental artifact/free-space reserve exhausted")
    require(report["status"] != "completed" or time.perf_counter() - started < SECONDS,
            "supplemental publication deadline reached")
    with output.open("xb") as handle:
        require(handle.write(raw) == len(raw), "short supplemental report write")
        handle.flush()
        os.fsync(handle.fileno())
    used_after = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    free_after = shutil.disk_usage(root).free
    output_bytes = output.stat().st_size
    disk_met = used_after <= ARTIFACT_LIMIT and free_after >= FREE_RESERVE
    publication_met = output_bytes == len(raw) and output_bytes <= OUTPUT_LIMIT
    elapsed = time.perf_counter() - started
    status = report["status"] if elapsed < SECONDS and disk_met and publication_met else "failed"
    receipt = {"status": status, "report_status": report["status"], "deadline_met": elapsed < SECONDS,
               "disk_limits_met": disk_met, "publication_size_met": publication_met,
               "disk_before_write": {"artifact_bytes": used, "free_bytes": free_before},
               "disk_after_write": {"artifact_bytes": used_after, "free_bytes": free_after,
                                    "output_bytes": output_bytes},
               "path": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
               "bytes": len(raw), "complete_seconds": elapsed,
               "confirmed_read_bytes": report["io"]["confirmed_read_bytes"],
               "scope": "separately monitored supplemental saved-only extraction"}
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
