"""Saved-only stable credit comparison audit; no models, tensors, tasks or children."""

from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_v8_review as v8
from experiments.r4_integrated.stable_language_action_credit_workload_v1 import (
    BASE_MANIFEST_SHA,
    CHECKPOINTS,
    PROBE_GROUPS,
    ROSTER,
    SCHEMA,
)

prior = v8.prior
scalar = v8.scalar
require, close = v8.require, v8.close
encoded, digest, read_json, safe_path = v8.encoded, v8.digest, v8.read_json, v8.safe_path
journal, read_row, work_valid = v8.journal, v8.read_row, v8.work_valid
ROOT = Path(__file__).resolve().parents[2]
ROUTES, COUNTERS = v8.ROUTES, v8.COUNTERS
new_symbol_exposures, new_trajectory = v8.new_symbol_exposures, v8.new_trajectory
check_scan, scan_schedule = v8.check_scan, v8.scan_schedule
check_tick, check_prior, check_execution = v8.check_tick, v8.check_prior, v8.check_execution
metrics = v8.metrics
AUDIT_SECONDS = 600
FROZEN_AUDITOR = "experiments/r4_integrated/stable_language_action_credit_review_v1.py"
FROZEN_AUDITOR_SHA256 = "2f53d1edea559fee230205d9e107fe4df54e9366d793a9781009d2d0db864c0c"


def _admit(deadline):
    require(time.perf_counter() < deadline, "saved audit exceeded declared 600 seconds")


def _behavior_signature(row):
    return digest(encoded({key: row[key] for key in (
        "answer_hex", "proposal", "correct", "after_files", "status")}
        | {"grade": row["grade"] if row["kind"] == "edit" else None}))


def _phase_work_gap(previous, item, row_bytes=()):
    current = item["cumulative_work"]
    delta = item["work_delta"]
    work_valid(delta)
    require(set(delta) == set(current), "checkpoint phase counter set")
    gap = {k: v - previous.get(k, 0) - delta[k] for k, v in current.items()}
    require(all(v >= 0 for v in gap.values()), "checkpoint phase exceeds total work")
    expected = {}
    if item["phase"] == "capture":
        n, size = len(row_bytes), sum(row_bytes)
        require(n <= 2 and all(type(v) is int and v > 0 for v in row_bytes),
                "checkpoint append row sizes")
        checks = 2 * n + 3
        expected["freeze_scopes"] = 3
        if n:
            expected.update(authenticated_rows=n, bytes_written=size, tail_bytes_read=size,
                            read_buffer_bytes=sum(min(1048576, v) for v in row_bytes),
                            tail_read_calls=sum((v + 1048575) // 1048576 for v in row_bytes),
                            write_buffer_bytes=size + n, write_calls=n,
                            write_through_completions=n)
    elif item["phase"] == "begin":
        require(not row_bytes, "begin has no appended row")
        checks = 6
    else:
        require(item["phase"] == "commit" and not row_bytes, "unexpected phase gap")
        checks = 0
    if checks:
        expected.update({key: checks for key in (
            "identity_checks", "identity_open_attempts", "identity_opens", "identity_closes")})
        expected["metadata_queries"] = 8 * checks
    expected = {"checkpoint_journal_" + key: value for key, value in expected.items()}
    require({key: value for key, value in gap.items() if value} == expected,
            "checkpoint intervening native journal work differs from actual rows")
    return expected


def _paired(left, right, *, migration=False):
    require(len(left) == len(right), "paired comparison missing declared outcomes")
    results = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        require((a["episode_id"], a["attempt_index"]) ==
                (b["episode_id"], b["attempt_index"]), "paired comparison changed roster")
        same_input = a["observation_sha256"] == b["observation_sha256"]
        same_rng = a.get("rng_before_sha256") == b.get("rng_before_sha256")
        paired = same_input and same_rng
        equal = a.get("semantic_sha256", a.get("behavior_sha256")) == b.get(
            "semantic_sha256", b.get("behavior_sha256"))
        results.append({"episode_id": a["episode_id"], "attempt_index": a["attempt_index"],
                        "paired_actual_input_rng": paired,
                        "required_migration_pair": migration and index <= 22,
                        "status": "paired_equal" if paired and equal else (
                            "paired_mismatch" if paired else "unpaired_context_or_rng"),
                        "left_correct": a["correct"], "right_correct": b["correct"],
                        "raw_diagnostics_equal": a.get("diagnostics") == b.get("diagnostics")})
    return {"passed": all(r["status"] == "paired_equal" if r["required_migration_pair"] else
                           r["status"] != "paired_mismatch" for r in results), "rows": results,
            "paired_attempts": sum(r["paired_actual_input_rng"] for r in results),
            "unpaired_attempts": sum(not r["paired_actual_input_rng"] for r in results),
            "complete_parity_established": all(r["status"] == "paired_equal" for r in results),
            "comparison_content": "full forward and task behavior" if all(
                "semantic_sha256" in row for row in [*left, *right]) else
                "task outputs/proposals/grades/files only; main forward parity is not claimed",
            "scope": "parity applies only to matching actual input/RNG; unpaired context remains disclosed"}


def review_probes(run, worker, manifest_sha, deadline):
    # Only this module's pure forward whitelist is reused; no callback/restore/action is called.
    from experiments.r4_integrated.persistent_language_action_credit_probes import (
        _semantic_signature,
    )

    manifest = read_json(run / "manifest.json")
    episodes = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    rule, batteries = worker["credit_rule"], {}
    config_sha = scalar.file_sha(run / "checkpoints/config.json")
    require(set(worker["probe_states"]) == set(CHECKPOINTS), "missing scheduled safe checkpoint")
    report = read_json(run / "report.json")
    require(len(report["probe_checkpoints"]) == 4, "missing checkpoint callback receipt")
    for cp in CHECKPOINTS:
        _admit(deadline)
        directory = run / "probes" / f"cp-{cp}"
        path = directory / "summary.json"
        summary = read_json(path)
        receipt = next((r["receipt"] for r in report["probe_checkpoints"]
                        if r["checkpoint_index"] == cp), None)
        require(receipt is not None and receipt["sha256"] == scalar.file_sha(path)
                and Path(receipt["path"]).resolve() == path
                and receipt["bytes"] == path.stat().st_size, "scheduled probe receipt binding")
        state = worker["probe_states"][cp]
        boundary_path = run / "checkpoints" / f"boundary-{state['boundary_sequence']:09d}.json"
        require(summary["schema"] == "stable-credit-checkpoint-probe-summary-v1"
                and summary["status"] == "completed" and summary["credit_rule"] == rule
                and summary["checkpoint_index"] == cp and summary["manifest_sha256"] == manifest_sha
                and summary["manifest_content_sha256"] == digest(encoded(manifest))
                and summary["boundary_sha256"] == scalar.file_sha(boundary_path)
                and summary["config_sha256"] == config_sha, "exact scheduled probe origin")
        require(summary["attempts"] == 84 and summary["actual_executions"] == 51
                and all(summary[key] == 0 for key in ("source_offers", "teacher_offers", "live_state_publications")),
                "full fixed probe exposure/no teaching")
        destruction = summary["destruction"]
        require(destruction == read_json(directory / "destruction.json")
                and destruction["constructed_actors"] == 2 and destruction["all_actors_released"]
                and destruction["reference_released"] and not destruction["published_state_to_live"]
                and not destruction["cleanup_errors"], "disposable actor custody")
        entries, frontier = journal(directory / "attempts.jsonl")
        require(len(entries) == 84 and summary["journal"] == {
            "sha256": frontier["sha256"], "bytes": frontier["bytes"]}, "complete probe journal binding")
        launch = read_json(directory / "launch.json")
        for name, identity in launch["artifacts"].items():
            require(scalar.file_sha(name) == identity, "probe actual execution qualification")
        base = state["baseline"]
        documents = [{"bytes_hex": value} for value in base["documents_hex"]]
        demos = [{key: {"bytes_hex": value} if key in {"query", "old", "replacement"} else value
                  for key, value in demo.items()} for demo in base["demonstrations"]]
        offset, routes = 0, {}
        for route in ROUTES:
            saved = summary["routes"][route]
            before = saved["initial"]
            neural, responses, rng, attempts, totals = route != "simple", set(), None, [], {}
            if neural:
                require(before["factual_sha256"] == state["hashes"][route]["memory"]
                        and before["memory_used"] == 1982
                        and before["actor_experience"] == 325 + state["actions"][route]
                        and before["reader_experience"] == 278 + state["free"][route],
                        "checkpoint retained numeric and prediction counters")
                if state["controllers"][route] is not None:
                    require(before["rng_sha256"] == state["controllers"][route]["rng_sha256"],
                            "checkpoint initial RNG")
            else:
                require(before == {"reference_sha256": digest(encoded(base))},
                        "checkpoint complete simple history")
            groups = {}
            for group, identifiers in PROBE_GROUPS:
                details = []
                for identifier in identifiers:
                    episode = episodes[identifier]
                    if episode["phase"] == "retention":
                        parent = "online" if neural else "simple"
                        observation = manifest["prior_regression_bindings"]["routes"][parent][
                            episode["prior_episode_id"]]["observation"]
                        files, feedback = observation["files"] or [], observation["feedback"]
                    else:
                        files, feedback = episode["files"], None
                    outcomes, proposed, repairs = [], [], []
                    for ai in range(1, 4 if group == "recovery" else 2):
                        _admit(deadline)
                        entry, row = entries[offset], read_row(entries[offset])
                        offset += 1
                        require(row["schema"] == "stable-credit-checkpoint-probe-attempt-v1"
                                and (row["route"], row["episode_id"], row["attempt_index"], row["probe_group"])
                                == (route, identifier, ai, group) and row["kind"] == episode["kind"]
                                and row["numeric_before"] == before and row["before_files"] == files
                                and row["learning_enabled"] is False and row["writes_enabled"] is False
                                and row["source_offers"] == row["teacher_offers"] == 0,
                                "fixed probe sequence/state/learning gate")
                        observation = {"request_hex": episode["request_hex"], "files": files,
                                       "feedback": feedback}
                        require(row["status"] in {"completed", "unsupported"}, "probe outcome status")
                        if row["status"] == "completed":
                            observation.update(schema="sequence-assembly-raw-observation-v1",
                                               files=files if episode["kind"] == "edit" else None)
                        else:
                            require(row["prediction"] is None and row["answer_hex"] is None
                                    and row["proposal"] is None, "unsupported probe produced prediction")
                        require(row["observation"] == observation, "probe actual context/feedback")
                        trajectory = None
                        if row["prediction"] is not None:
                            if neural:
                                prediction = row["prediction"] if episode["kind"] == "reading" else row["prediction"]["payload"]
                                trajectory = check_prediction(prediction, observation, 1982, responses, rule)
                                require(rng is None or trajectory["first_rng"] == rng,
                                        "probe actual draw RNG continuity")
                                rng = trajectory["rng"]
                                if episode["kind"] == "reading":
                                    require(row["answer_hex"] == prediction["answer_hex"], "probe answer substitution")
                                else:
                                    scalar.check_heads(row["prediction"], row["proposal"], files)
                            else:
                                prior.check_reference(row, documents, demos)
                        reward = None
                        if episode["kind"] == "edit":
                            execution = row["execution"]
                            public = (execution["public_driver_policy"]["driver_integrity"]
                                      and execution["returncode"] == 0 and not execution["timeout"]
                                      and execution["capture_complete"] and execution["files_complete"]
                                      and not execution["output_limit_exceeded"] and not execution["capture_errors"])
                            feedback = check_execution(dict(row, public_checks_passed=bool(public)),
                                                       episode, directory, launch)
                            correct = row["status"] != "unsupported" and row["grade"]["task_success"]
                            failed = (not execution["accepted"] or execution["returncode"] != 0
                                      or execution["timeout"] or not execution["capture_complete"]
                                      or not execution["files_complete"] or execution["output_limit_exceeded"])
                            reward = -int(failed and row["proposal"] is not None
                                          and row["proposal"]["operation"] != "STOP")
                        else:
                            require(row["execution"] is None and row["after_files"] == files,
                                    "reading probe performed action")
                            correct = row["status"] != "unsupported" and row["answer_hex"] in episode[
                                "grade"]["accepted_answer_hex"]
                        require(row["correct"] is bool(correct), "independent probe actual grade")
                        if trajectory is not None:
                            check_resolution(row["controller_resolution"], trajectory, episode["kind"],
                                             False, state["baselines"][route], reward)
                        after = row["numeric_after"]
                        if neural:
                            require(all(after[key] == saved["initial"][key] for key in (
                                "slow_numeric_sha256", "factual_sha256", "memory_used")), "probe held slow state")
                            require(after["reader_experience"] == before["reader_experience"] + int(trajectory is not None)
                                    and after["actor_experience"] == before["actor_experience"]
                                    + int(trajectory is not None and episode["kind"] == "edit"), "probe custody counter")
                        else:
                            require(after == saved["initial"], "probe changed simple history")
                        semantic = _semantic_signature(row)
                        require(row["semantic_sha256"] == semantic, "complete paired forward signature")
                        compact = {key: row[key] for key in ("episode_id", "attempt_index", "probe_group",
                                                           "kind", "correct", "status")}
                        compact.update(observation_sha256=digest(encoded(observation)),
                                       rng_before_sha256=before.get("rng_sha256"),
                                       rng_after_sha256=after.get("rng_sha256"), semantic_sha256=semantic,
                                       row_sha256=entry["sha256"], feedback=observation["feedback"],
                                       diagnostics=None if row["execution"] is None else {
                                           key: row["execution"][key] for key in (
                                               "stdout_hex", "stderr_hex", "capture_errors")})
                        attempts.append(compact)
                        for scope in ("work", "ack_work"):
                            work_valid(row[scope])
                            for key, value in row[scope].items():
                                if type(value) in (int, float):
                                    name = scope + "/" + key
                                    totals[name] = totals.get(name, 0) + value
                        outcomes.append(bool(correct))
                        proposed.append(row["proposal"])
                        repairs.append(bool(correct and row["execution"] is not None
                                            and row["execution"]["accepted"]
                                            and row["proposal"] is not None
                                            and row["proposal"]["operation"] == "SPLICE"
                                            and row["execution"]["proposed_files"] != row["before_files"]))
                        before, files = after, row["after_files"]
                    details.append({"episode_id": identifier, "attempt_correct": outcomes,
                                    "first_correct": outcomes[0], "final_correct": outcomes[-1],
                                    "recovered": not outcomes[0] and any(outcomes[1:]),
                                    "successful_later_repair": not outcomes[0] and any(repairs[1:]),
                                    "identical_consecutive_proposals": sum(a == b for a, b in
                                        zip(proposed, proposed[1:]))})
                groups[group] = {"episodes": len(details), "first_correct": sum(d["first_correct"] for d in details),
                                 "final_correct": sum(d["final_correct"] for d in details),
                                 "recovered": sum(d["recovered"] for d in details), "details": details}
                groups[group]["successful_later_repairs"] = sum(d["successful_later_repair"] for d in details)
            require(saved["attempts"] == attempts and saved["final"] == before
                    and saved["actual_executions"] == 17 and saved["canonical_work"] == totals,
                    "saved probe summary differs from actual journal")
            routes[route] = {"initial": saved["initial"], "final": before, "groups": groups,
                             "teaching_age": state["teaching_age"][route],
                             "attempts": attempts, "canonical_work": totals}
        require(offset == 84, "probe journal omitted/excess outcomes")
        if cp == 0:
            require(routes["online"]["initial"] == routes["learning_disabled"]["initial"],
                    "CP0 enabled/disabled retained state mismatch")
            require(_paired(routes["online"]["attempts"], routes["learning_disabled"]["attempts"],
                            migration=True)["passed"], "CP0 enabled/disabled behavior mismatch")
            require(summary["comparison"] == read_json(directory / "migration.json")
                    and all(value["passed"] for value in summary["comparison"].values()),
                    "CP0 migration declaration failed")
        batteries[cp] = {"routes": routes, "summary_sha256": scalar.file_sha(path)}
    return batteries

def check_initialization(row, manifest, rule):
    expected = {
        "schema": "persistent-language-action-credit-initialization-v1", "status": "initialized",
        "algorithm": "v8-original-credit" if rule == "old" else "causal-observation-credit-v1",
        "prior_sha256": "f49a6496f91bca6dc431ae221034dfda9e897d0f2476ae18d2bb4b65f8617592",
        "source_model_sha256": "376c5a6223b727020a56bf09d62b7aed5ead069431fd989de5faf7f94adce853",
        "source_memory_sha256": "cdd1f887f80ec2e8ae6ee38ae8a15786d9155160cf6beb18def635a2a66da73e",
        "prior_boundary_sequence": 264, "loaded_model_tensor_count": 29,
        "loaded_tape_tensor_count": 4, "loaded_tensor_bytes": 115690664,
        "serialized_payload_bytes": 115712128, "initial_experience_per_route": 325,
        "initial_reader_experience_per_route": 278, "initial_factual_slots_per_route": 1982,
        "baseline_documents": 60, "baseline_demonstrations": 49,
        "bank_sha256": "87f0c8e39cbbaaed0558d1affb1ca87f52177f4cb6aee0650b74cfcbf06f392a",
        "prior_head_sha256": "98cb75894fdb2b5f9b4886b0e44b95e03be5ccd1d644ee6e409d4a0e360e2993",
        "prior_boundary_sha256": "e2acf10914cab2da891352e97eed4dc1e665e02e54e62508c58f858b3c94fdc4",
    }
    require(all(row.get(key) == value for key, value in expected.items()), "retained V8 initialization")
    require(all(row[key] == 0 for key in ("source_ingestions", "source_observations",
            "answer_teacher_calls", "neural_prediction_calls", "execution_children")),
            "initializer introduced learning or action")
    directory = ROOT / ".pytest-tmp/persistent-language-action-v8-corrected-run/checkpoints"
    require(Path(row["prior_path"]).resolve() == directory / "route-online-000000262.pt",
            "different retained ONLINE payload")
    for path, identity in ((Path(row["prior_path"]), row["prior_sha256"]),
                           (Path(row["bank_path"]), row["bank_sha256"]),
                           (directory / "HEAD.json", row["prior_head_sha256"]),
                           (directory / "boundary-000000264.json", row["prior_boundary_sha256"])):
        require(scalar.file_sha(path) == identity, "historical initializer byte binding")
    historical = read_json(directory / "boundary-000000264.json")
    require(row["prior_host"] == historical["host"] and row["prior_frontiers"] == historical["frontiers"]
            and row["baseline_record"] == historical["baseline"], "actual retained host/reference")
    require(historical["routes"]["online"]["sha256"] == row["prior_sha256"]
            and historical["routes"]["online"]["model_sha256"] == row["source_model_sha256"],
            "historical selected actor")
    base = row["baseline_record"]
    documents = [{"bytes_hex": value} for value in base["documents_hex"]]
    demos = [{key: {"bytes_hex": value} if key in {"query", "old", "replacement"} else value
              for key, value in demo.items()} for demo in base["demonstrations"]]
    require(len(documents) == 60 and len(demos) == 49
            and sum(len(bytes.fromhex(d["bytes_hex"])) + 1 for d in documents) == 11498,
            "retained complete simple history")
    routes = row["initial_routes"]
    require(set(routes) == set(ROUTES[:2]) and routes["online"] == routes["learning_disabled"],
            "matched independent descendants")
    for route in ROUTES[:2]:
        state = routes[route]
        require(state["numeric_model_sha256"] == row["source_numeric_model_sha256"]
                and state["memory_sha256"] == row["source_memory_sha256"], "numerical lineage")
        for key in ("model_sha256", "numeric_model_sha256", "controller_rng_sha256",
                    "views_sha256", "self_wiring_sha256"):
            scalar.hash_value(state[key])
        require(len(state["baseline_mean"]) == len(state["baseline_count"]) == 3
                and all(type(v) is int and v >= 0 for v in state["baseline_count"])
                and all(type(v) in (float, int) and math.isfinite(v) and abs(v) <= 1
                        for v in state["baseline_mean"]), "held baseline scalars")
    require(manifest["continuation"]["payload_sha256"] == row["prior_sha256"],
            "manifest initializer selection")
    return documents, demos, dict(base["new_teacher_offers"]), routes


def check_prediction(prediction, observation, used, responses, rule):
    trajectory = v8.check_prediction(prediction, observation, used, responses)
    trajectory["rule"] = rule
    trajectory["credit_ticks"] = {"scan": prediction["scan_steps"], "teacher": []}
    return trajectory


def check_resolution(record, trajectory, kind, enabled, baseline, public_reward=None):
    if trajectory["rule"] == "old":
        return v8.check_resolution(record, trajectory, kind, enabled, baseline, public_reward)
    require(record["schema"] == "recurrent-content-update-v2"
            and record["learning_enabled"] is enabled and record["rng_published"] is True,
            "causal publication contract")
    for key in ("source_sha256", "state_sha256", "stage_sha256"):
        scalar.hash_value(record[key])
    ticks = {key: trajectory["ticks"][key] for key in ("scan", "free", "teacher")}
    require(record["ticks"] == ticks and record["draws"] == trajectory["ordinal"],
            "causal actual draw/tick ledger")
    totals, signals, inclusions = {}, {}, {}
    for category, index in (("scan", 0 if kind == "source" else 1), ("teacher", 2)):
        running, signal_sum, literal = 0.0, 0.0, 0
        offers = trajectory["credit_ticks"][category]
        require(len(offers) == ticks[category], "causal offered-symbol denominator")
        for tick in offers:
            credit = tick["credit"]
            target, mode = credit["observed_token"], tick["draws"][2]["index"]
            require(type(target) is int and 0 <= target <= 256
                    and credit["schema"] == "causal-observation-credit-offer-v1"
                    and credit["category"] == category and credit["mode"] == mode,
                    "causal observed symbol/category")
            if category == "scan":
                require(target == tick["observed_token"], "scan actual target")
            sampled = 1 if tick["emitted_token"] == target else -1
            require(credit["sampled_reward"] == sampled, "sampled diagnostic reward")
            if mode == 3:
                draw = tick["draws"][3]
                probability = draw["probabilities"][target]
                close(credit["target_probability"], probability, "held pretarget target probability")
                logits = draw["logits"]
                maximum = max(logits)
                loss = maximum - logits[target] + math.log(sum(math.exp(x - maximum) for x in logits))
                close(credit["literal_loss"], loss, "stable held literal cross entropy")
                require(credit["literal_taught"] is True
                        and credit["literal_activity_sha256"] == draw["activity_sha256"],
                        "direct own evaluated literal activity")
                signal = 2 * probability - 1
                literal += 1
            else:
                require(credit["literal_taught"] is False and all(credit[key] is None for key in
                        ("target_probability", "literal_loss", "literal_activity_sha256")),
                        "unevaluated literal branch was taught")
                signal = sampled
            close(credit["signal"], signal, "conditional upstream match signal")
            close(credit["baseline_before"], baseline[index][0], "held category baseline")
            close(credit["causal_total_before"], running, "causal A before offered symbol")
            running += signal - baseline[index][0]
            close(credit["causal_total_after"], running, "causal A after offered symbol")
            signal_sum += signal
        totals[category], signals[category], inclusions[category] = running, signal_sum, literal
    saved = record["rewards"]
    means = {key: signals[key] / ticks[key] if ticks[key] else None for key in signals}
    for category in means:
        if means[category] is None:
            require(saved[category] is None and saved["sampled_" + category] is None,
                    "absent teacher category")
        else:
            close(saved[category], means[category], "category mean conditional reward")
            close(saved["sampled_" + category], trajectory["rewards"][category] / ticks[category],
                  "category mean sampled reward")
        close(dict(saved["causal_totals"])[category], totals[category], "final category A")
    require(dict(saved["literal_offers"]) == inclusions
            and dict(saved["external_offers"]) == {key: ticks[key] for key in signals}
            and saved["public"] == public_reward, "actual inclusions/external/public offers")
    si = 0 if kind == "source" else 1
    close(saved["scan_baseline_before"], baseline[si][0], "scan baseline receipt")
    close(saved["teacher_baseline_before"], baseline[2][0], "teacher baseline receipt")
    count = 1 + int(bool(ticks["teacher"])) + int(kind == "edit")
    length = ticks["scan"] + (0 if kind == "source" else ticks["teacher"] + 129)
    require(record["category_count"] == count and record["normalization_ticks"] == length,
            "fixed causal outer normalization")
    scale, teacher = .01 / (count * length), ticks["teacher"]
    expected = {
        "causal_scan": scale / ticks["scan"], "causal_teacher": scale / teacher if teacher else 0,
        "common_scan_teacher": scale * totals["teacher"] / teacher if teacher else 0,
        "public": scale * (public_reward or 0), "literal_scan": .01 / (count * ticks["scan"]),
        "literal_teacher": .01 / (count * teacher) if teacher else 0,
    }
    require(set(dict(saved["coefficients"])) == set(expected), "separate causal/literal objectives")
    for key, value in expected.items():
        close(dict(saved["coefficients"])[key], value, "causal coefficient " + key)
    norm = record["delta_norm"]
    require(type(norm) in (int, float) and math.isfinite(norm)
            and 0 <= norm <= math.sqrt(.0026) + 1e-10, "bounded causal proposal")
    if enabled:
        for index, value in ((si, means["scan"]), (2, means["teacher"])):
            if value is not None:
                mean, count = baseline[index]
                baseline[index] = [mean + (value - mean) / (count + 1), count + 1]


def check_source(feedback, raw, enabled, admitted, baseline, rule):
    require(feedback["document_count"] == 1 and feedback["raw_bytes"] == len(raw)
            and feedback["symbol_count"] == len(raw) + 1 and feedback["learning_enabled"] is enabled
            and feedback["fact_admitted"] is admitted and feedback["writes_enabled"] is admitted,
            "single transient source receipt")
    trajectory = new_trajectory(scan_kind="source_scan")
    check_scan(feedback["source_steps"], scan_schedule(raw=raw), trajectory)
    trajectory.update(rule=rule, credit_ticks={"scan": feedback["source_steps"], "teacher": []})
    check_resolution(feedback["controller"], trajectory, "source", enabled, baseline)
    work_valid(feedback["work"])
    return trajectory


def check_teacher_feedback(feedback, prediction, target, used, trajectory,
                           enabled, baseline, kind, public_reward):
    steps, common = feedback["steps"], prediction["controller"]
    require(feedback["actual_response_id"] == prediction["attempt_id"] and len(steps) == len(target) + 1,
            "same actual response before teacher")
    trajectory["h"], trajectory["cursors"] = common["preoutput_h"], common["preoutput_cursors"]
    bank = prediction["steps"][0]["binding"]["bank_sha256"]
    request = prediction["steps"][0]["binding"]["request_sha256"]
    loss = 0.0
    for index, (step, token) in enumerate(zip(steps, [*target, 256], strict=True)):
        require(step["output_step"] == index and step["prefix_hex"] == target[:index].hex()
                and step["observed_token"] == token and step["binding"]["request_sha256"] == request,
                "causal teacher prefix")
        if index == 0:
            require(step["controller"]["position_sha256"] == common["preoutput_position_sha256"],
                    "teacher preoutput context fork")
        winner = check_prior(step, target[:index], used, bank)
        emitted = check_tick(step["controller"], "teacher", trajectory, target=token,
                             prior_token=winner, address=step["selected_index"])
        require(step["reward"] == (1 if emitted == token else -1), "sampled teacher diagnostic")
        if trajectory["rule"] == "causal":
            require(step["controller"]["credit"]["observed_token"] == token, "teacher target receipt")
        logits = step["prior_logits"]
        maximum = max(logits)
        expected = maximum + math.log(sum(math.exp(x - maximum) for x in logits)) - logits[token]
        require(math.isclose(step["prior_nll"], expected, rel_tol=2e-6, abs_tol=2e-6),
                "held teacher prior NLL")
        loss += step["prior_nll"]
    close(feedback["total_loss"], loss, "teacher total diagnostic loss")
    close(feedback["mean_loss"], loss / len(steps), "teacher mean diagnostic loss")
    trajectory["credit_ticks"]["teacher"] = [step["controller"] for step in steps]
    check_resolution(feedback["controller"], trajectory, kind, enabled, baseline, public_reward)
    return len(steps)


def check_checkpoints(run, sources, manifest_sha, rows_by_kind, partial, resolved,
                      *, frozen_auditor_path=None):
    """Independent immutable-byte and scalar frontier audit; no tensor decoder."""
    directory = run / "checkpoints"
    if not directory.exists():
        require(partial, "checkpoint store absent")
        return {
            "head": None,
            "boundaries": 0,
            "dirty": True,
            "reason": "initialization store absent",
        }
    config = read_json(directory / "config.json")
    require(
        config["schema"] == "persistent-session-checkpoint-config-v1"
        and config["manifest_sha256"] == manifest_sha
        and config["source_sha256"] == sources,
        "checkpoint source/manifest binding",
    )
    require(
        Path(config["source_root"]).resolve() == ROOT
        and Path(config["evidence_root"]).resolve() == run,
        "checkpoint configured roots",
    )
    require(
        scalar.file_sha(directory / "manifest.json") == manifest_sha, "checkpoint manifest bytes"
    )
    config_sha = scalar.file_sha(directory / "config.json")
    intents = {}
    for path in sorted(directory.glob("intent-*.json")):
        value = read_json(path)
        require(
            value["sequence"] not in intents and value["config_sha256"] == config_sha,
            "intent identity",
        )
        intents[value["sequence"]] = (value, scalar.file_sha(path))
    require(
        0 in intents and intents[0][0]["kind"] == "initialize", "pre-initialization durable intent"
    )
    if not (directory / "HEAD.json").exists():
        require(partial, "complete run has no safe checkpoint")
        return {"head": None, "boundaries": 0, "dirty": True, "intents": len(intents)}
    head = read_json(directory / "HEAD.json")
    require(head["schema"] == "persistent-session-checkpoint-head-v1", "HEAD schema")
    previous, previous_hash, hashes, last = None, None, {}, None
    boundary_events = []
    for sequence in range(head["sequence"] + 1):
        path = directory / f"boundary-{sequence:09d}.json"
        boundary = read_json(path)
        boundary_sha256 = scalar.file_sha(path)
        require(
            boundary["schema"] == "persistent-session-checkpoint-boundary-v1"
            and boundary["sequence"] == sequence
            and boundary["previous_sha256"] == previous_hash
            and boundary["config_sha256"] == config_sha
            and boundary["intent_sha256"] == intents[sequence][1],
            "immutable boundary lineage",
        )
        intent = intents[sequence][0]
        require(sequence < len(resolved), "HEAD exceeds resolved journal transactions")
        expected = resolved[sequence]
        require(
            intent["kind"] == expected["kind"] and intent["cursor"] == expected["before_cursor"],
            "checkpoint intended transaction differs",
        )
        host = boundary["host"]
        require(
            host["cursor"] == expected["cursor"]
            and host["counters"] == expected["counters"]
            and host["projects"] == expected["projects"]
            and host["current"] == expected["current"],
            "checkpoint cursor/counter/actual workspace mismatch",
        )
        if previous is None:
            for name, initial_sha256 in host["initial_bindings"].items():
                require(
                    scalar.file_sha(safe_path(run, name)) == initial_sha256,
                    "checkpoint initial evidence",
                )
        else:
            require(
                host["initial_bindings"] == previous["host"]["initial_bindings"]
                and host["budget_seconds"] == previous["host"]["budget_seconds"]
                and host["budget_mib"] == previous["host"]["budget_mib"],
                "checkpoint budget/bindings reset",
            )
        require(boundary["baseline"] == expected["baseline"], "checkpoint actual reference state")
        require(set(boundary["routes"]) == set(ROUTES[:2]), "checkpoint route set")
        require(intent["previous_sha256"] == previous_hash, "intent predecessor")
        if previous is not None:
            require(
                intent["cursor"] == previous["host"]["cursor"], "intent resumes another frontier"
            )
        for route, value in boundary["routes"].items():
            require(route in ROUTES[:2], "checkpoint neuronal route")
            payload = safe_path(directory, value["payload"])
            if payload not in hashes:
                hashes[payload] = scalar.file_sha(payload)
            require(
                hashes[payload] == value["sha256"] and payload.stat().st_size == value["bytes"],
                "immutable tensor payload bytes",
            )
            for key in ("model_sha256", "memory_sha256"):
                scalar.hash_value(value[key])
            require(
                value["model_sha256"] == expected["hashes"][route]["model"]
                and value["memory_sha256"] == expected["hashes"][route]["memory"]
                and value["reader_experience"] == 278 + expected["free"][route]
                and value["experience"] == 325 + expected["actions"][route],
                "checkpoint actual learner lineage/counters",
            )
        frontier = boundary["frontiers"]
        for item in frontier["journals"]:
            kind = item["kind"]
            require(item["path"] == kind + ".jsonl", "unexpected journal segment")
            require(0 <= item["rows"] <= len(rows_by_kind[kind]), "boundary beyond durable journal")
            entry = rows_by_kind[kind][item["rows"] - 1] if item["rows"] else None
            require(
                (0 if entry is None else entry["end"]) == item["bytes"]
                and (digest(b"") if entry is None else entry["prefix_sha256"]) == item["sha256"],
                "journal prefix hash",
            )
        for item in frontier["execution_files"]:
            path = safe_path(run, item["path"])
            if path not in hashes:
                hashes[path] = scalar.file_sha(path)
            require(
                hashes[path] == item["sha256"] and path.stat().st_size == item["bytes"],
                "execution frontier bytes",
            )
        for name in frontier["execution_directories"]:
            require(safe_path(run, name).is_dir(), "lost actual execution directory")
        work_valid(boundary["work"])
        require(
            type(host["elapsed_seconds"]) in (int, float)
            and math.isfinite(host["elapsed_seconds"])
            and host["elapsed_seconds"]
            >= (0 if previous is None else previous["host"]["elapsed_seconds"]),
            "checkpoint cumulative clock reset",
        )
        boundary_events.append(
            {
                "sequence": sequence,
                "kind": intent["kind"],
                "cursor": intent["cursor"],
                "host_cursor": boundary["host"]["cursor"],
                "sha256": boundary_sha256,
                "journal_appended_row_bytes": [entry["bytes"]
                    for item in frontier["journals"]
                    for entry in rows_by_kind[item["kind"]][
                        0 if previous is None else next(j["rows"] for j in
                            previous["frontiers"]["journals"] if j["kind"] == item["kind"]):item["rows"]]],
            }
        )
        previous, previous_hash, last = boundary, boundary_sha256, boundary
    require(
        head["manifest"] == f"boundary-{head['sequence']:09d}.json"
        and head["sha256"] == previous_hash,
        "atomic HEAD does not select final published boundary",
    )
    extra_intents = sorted(i for i in intents if i > head["sequence"])
    final_rows = {x["kind"]: x["rows"] for x in last["frontiers"]["journals"]}
    beyond = {k: len(v) - final_rows[k] for k, v in rows_by_kind.items()}
    beyond_bytes = {
        item["kind"]: (run / item["path"]).stat().st_size - item["bytes"]
        for item in last["frontiers"]["journals"]
    }
    expected_dirs = set(last["frontiers"]["execution_directories"])
    actual_dirs = {
        p.relative_to(run).as_posix() for p in (run / "executions").rglob("*") if p.is_dir()
    }
    expected_files = {r["path"] for r in last["frontiers"]["execution_files"]}
    actual_files = {
        p.relative_to(run).as_posix() for p in (run / "executions").rglob("*") if p.is_file()
    }
    dirty = bool(
        extra_intents
        or any(beyond.values())
        or any(beyond_bytes.values())
        or actual_dirs != expected_dirs
        or actual_files != expected_files
    )
    require(partial or not dirty, "complete session has work beyond safe HEAD")
    # Invoke only the existing byte/JSON inspector, never restore_into or a
    # learner factory. This retains the exact current V8 subclass source guards.
    from experiments.r4_integrated.stable_language_action_credit_checkpoint_v1 import (
        StableLanguageActionCreditCheckpoint,
    )

    class SavedAuditCheckpoint(StableLanguageActionCreditCheckpoint):
        def _file(self, path, *, prefix=None):
            if frozen_auditor_path is not None and path.resolve() == (ROOT / FROZEN_AUDITOR).resolve():
                require(sources[FROZEN_AUDITOR] == FROZEN_AUDITOR_SHA256,
                        "unknown original inspector auditor")
                path = Path(frozen_auditor_path).resolve()
            return super()._file(path, prefix=prefix)

    inspect_started = time.perf_counter()
    with SavedAuditCheckpoint.open(directory, config_sha256=config_sha) as store:
        checked = store.inspect(require_clean=not partial, fresh_payloads=True)
        require(
            checked["boundary_sha256"] == previous_hash and (checked["status"] == "dirty") is dirty,
            "independent frontier disagrees with full V8 byte inspector",
        )
        inspector_work = dict(store.work)
    inspector_seconds = time.perf_counter() - inspect_started
    return {
        "head": head,
        "head_sha256": scalar.file_sha(directory / "HEAD.json"),
        "boundaries": len(boundary_events),
        "boundary_events": boundary_events,
        "dirty": dirty,
        "outstanding_intents": extra_intents,
        "journal_rows_beyond_head": beyond,
        "journal_bytes_beyond_head": beyond_bytes,
        "execution_files_beyond_head": sorted(actual_files - expected_files),
        "execution_directories_beyond_head": sorted(actual_dirs - expected_dirs),
        "last_host": last["host"],
        "last_routes": last["routes"],
        "last_baseline": last["baseline"],
        "unique_hashed_payload_and_execution_files": len(hashes),
        "checkpoint_decodes": 0,
        "v8_fresh_inspector_work": inspector_work,
        "v8_fresh_inspector_seconds": inspector_seconds,
    }

# Dispatcher and checkpoint/frontier audit follow below. No module import runs a review.


def review_worker(run, manifest_sha, deadline, *, frozen_auditor_path=None):
    started, partial = time.perf_counter(), False
    run = Path(run).resolve()
    require(scalar.file_sha(run / "manifest.json") == manifest_sha, "changed stable manifest")
    manifest = read_json(run / "manifest.json")
    require(manifest["schema"] == SCHEMA and [e["episode_id"] for e in manifest["episodes"]] == ROSTER,
            "stable complete roster")
    require(manifest["credit_comparison"]["base_manifest_sha256"] == BASE_MANIFEST_SHA,
            "unbound exposed source lineage")
    base_path = ROOT / ".pytest-tmp/persistent-language-action-v8-manifest.json"
    require(scalar.file_sha(base_path) == BASE_MANIFEST_SHA, "changed historical roster bytes")
    base_manifest = read_json(base_path)
    expected_documents = copy.deepcopy(base_manifest["documents"])
    for document in expected_documents:
        document["neural_tape_admission"] = False
    require(manifest["episodes"] == base_manifest["episodes"]
            and manifest["documents"] == expected_documents
            and manifest["prior_regression_bindings"] == base_manifest["prior_regression_bindings"],
            "stable tasks/source ordering/teachers/historical actual contexts changed")
    require(manifest["credit_comparison"]["checkpoint_episodes"] == list(CHECKPOINTS)
            and manifest["credit_comparison"]["probe_groups"] == [
                {"name": name, "episode_ids": list(ids)} for name, ids in PROBE_GROUPS],
            "fixed complete checkpoint battery")
    require(all(manifest["credit_comparison"][key] == value for key, value in {
        "pending_write_reserve_bytes": 2 * 1024**3, "new_artifact_bytes": 46 * 1024**3,
        "free_reserve_bytes": 8 * 1024**3, "complete_seconds": 10800,
        "runner_seconds": 10200, "audit_seconds": 600, "stop_reserve_seconds": 300,
        "probe_learning_enabled": False, "probe_writes_enabled": False,
        "probe_teacher_offers": 0, "probe_source_offers": 0,
        "main_teacher_budget_reduction": False, "main_difficulty_escalation": False}.items()),
        "frozen complete comparison limits")
    report = read_json(run / "report.json")
    rule = report["credit_rule"]
    require(rule in {"old", "causal"}, "unknown credit worker")
    completion = read_json(run / "completion.json")
    require(completion["schema"] == "stable-language-action-credit-completion-v1"
            and completion["status"] == "completed", "worker incomplete")
    require(set(completion["files"]) == {p.relative_to(run).as_posix() for p in run.rglob("*")
                                        if p.is_file() and p.name != "completion.json"},
            "complete saved worker inventory")
    for name, identity in completion["files"].items():
        _admit(deadline)
        require(scalar.file_sha(safe_path(run, name)) == identity, "changed saved evidence: " + name)
    sources = read_json(run / "sources.json")
    auditor_provenance = {"current_sha256": scalar.file_sha(ROOT / FROZEN_AUDITOR),
                          "frozen_sha256": sources[FROZEN_AUDITOR],
                          "frozen_source_copy": None}
    for name, identity in sources.items():
        source_path = safe_path(ROOT, name)
        if name == FROZEN_AUDITOR and frozen_auditor_path is not None:
            # A post-run reader correction never changes the execution-source map.
            require(identity == FROZEN_AUDITOR_SHA256, "unknown original auditor")
            source_path = Path(frozen_auditor_path).resolve()
            auditor_provenance["frozen_source_copy"] = str(source_path)
        require(scalar.file_sha(source_path) == identity, "changed source: " + name)
    require(all(sources.get(name) == identity for name, identity in
                report["runtime"]["source_sha256"].items()), "runtime source selection not bound")
    runtime = report["runtime"]
    require(runtime["rule"] == rule and runtime["old_source_revision"] ==
            "8e8df91d7e7c6889dff15f97d9fec379ebff71bb", "selected source revision")
    historical_sources_path = ROOT / ".pytest-tmp/persistent-language-action-v8-corrected-run/sources.json"
    require(runtime["historical_inventory_sha256"] == scalar.file_sha(historical_sources_path)
            == "ae23542c22d89a1e1bbbd4babb3739928a550a60da89b6bc50fbef06f92148ac",
            "banked old source inventory")
    historical_sources = read_json(historical_sources_path)
    changed_modules = {"src/humanbrain/recurrent_content_controller.py",
                       "src/humanbrain/recurrent_content_learner.py"}
    require(all(runtime["source_sha256"].get(name) == identity for name, identity in
                historical_sources.items() if name.startswith("src/humanbrain/") and name not in changed_modules),
            "unreviewed change outside selected credit source closure")
    for phase in ("imported_neuronal_origins", "initialization_origins", "final_origins"):
        origins = runtime[phase]
        for module, name in origins.items():
            path = Path(name).resolve()
            require(path.is_relative_to(ROOT) and runtime["source_sha256"].get(
                path.relative_to(ROOT).as_posix()) == scalar.file_sha(path), "imported module origin binding")
            if module in runtime["module_origins"]:
                require(name == runtime["module_origins"][module], "mixed old/current canonical source")
        require(set(runtime["module_origins"]) <= set(origins), "selected canonical source absent")
    old_sources = {
        "humanbrain.recurrent_content_controller": "e2fb738b86d820db85f3e8b6190a93a9c85fbc704bbfe897d8ac7c5a335f56b5",
        "humanbrain.recurrent_content_learner": "e89eb5ccfa44eda3b9a8306bc8603b2aeab11d78630e47de4cc2d8f1e22c9ece"}
    require(set(runtime["module_origins"]) == set(old_sources), "source isolation module closure")
    for module, name in runtime["module_origins"].items():
        require(scalar.file_sha(name) == old_sources[module] if rule == "old" else
                Path(name).resolve() == ROOT / ("src/" + module.replace(".", "/") + ".py"),
                "source-isolated exact credit modules")
    saved_tool_sources = {__file__: scalar.file_sha(__file__)}
    indexed, frontiers = {}, {}
    for kind in ("attempts", "teachers", "ingests"):
        indexed[kind], frontiers[kind] = journal(run / (kind + ".jsonl"), partial=partial)
    telemetry, telemetry_frontier = journal(run / "checkpoints.jsonl", partial=partial)
    counts = {r: dict.fromkeys(COUNTERS, 0) for r in ROUTES}
    consumed = dict.fromkeys(indexed, 0)
    compact, resolved, probe_states = [], [], {}
    corrections = {route: {} for route in ROUTES}
    costs = {r: {"work": {k: {} for k in ("source", "free", "teacher", "ack")},
                 "elapsed_seconds": dict.fromkeys(("source", "attempt", "teacher"), 0.0),
                 "included_checkpoint_begin_seconds": 0.0} for r in ROUTES}
    stochastic = {r: {"ticks": Counter(), "draws": 0,
                      "by_category": new_symbol_exposures()} for r in ROUTES[:2]}
    last_draw_rng = dict.fromkeys(ROUTES[:2])
    source_offers = dict.fromkeys(ROUTES, 0)
    free, actions = dict.fromkeys(ROUTES[:2], 0), dict.fromkeys(ROUTES[:2], 0)
    used = {"online": 1982, "learning_disabled": 1982, "simple": 11498}
    baselines = {r: [[0.0, 0], [0.0, 0], [0.0, 0]] for r in ROUTES[:2]}
    projects, instances, responses = {r: {} for r in ROUTES}, {}, set()
    cursor = {"episode_index": 0, "route_index": 0, "attempt_index": 1, "append_done": False}
    actual_current, pending = None, None

    def add_work(destination, values, prefix=""):
        work_valid(values)
        for key, value in values.items():
            if isinstance(value, dict):
                add_work(destination, value, prefix + key + ".")
            else:
                name = prefix + key
                destination[name] = destination.get(name, 0) + value

    def add_cost(row, category, clock):
        ledger = costs[row["route"]]
        add_work(ledger["work"][category], row["work"])
        ledger["elapsed_seconds"][clock] += prior.elapsed_value(row["elapsed_seconds"])
        if category == "free":
            add_work(ledger["work"]["ack"], row.get("ack_work", {}))
            ledger["included_checkpoint_begin_seconds"] += prior.elapsed_value(
                row["included_checkpoint_begin_seconds"])

    def add_trajectory(route, trajectory):
        require(last_draw_rng[route] is None or trajectory["first_rng"] == last_draw_rng[route],
                "actual cross-interaction draw RNG continuity")
        last_draw_rng[route] = trajectory["rng"]
        stochastic[route]["ticks"].update(trajectory["ticks"])
        stochastic[route]["draws"] += trajectory["ordinal"]
        for category, source in trajectory["symbol_exposures"].items():
            destination = stochastic[route]["by_category"][category]
            for key in ("ticks", "draws"):
                destination[key] += source[key]
            for key in ("sampled_modes", "sampled_gates"):
                destination[key].update(source[key])
            if source["observed_symbols"] is not None:
                for kind, values in source["observed_symbols"].items():
                    destination["observed_symbols"][kind].update(values)

    def controller_transition(route, before, after, *, advanced):
        require(controllers[route] is None or before == controllers[route], "controller published-state hash chain")
        for value in (*before.values(), *after.values()):
            scalar.hash_value(value)
        require(set(before) == set(after) == {"rng_sha256", "slow_sha256"}, "controller hash keys")
        require((after["rng_sha256"] != before["rng_sha256"]) is advanced,
                "actual resolved RNG advancement")
        require(route == "online" or after["slow_sha256"] == before["slow_sha256"],
                "disabled slow controller publication")
        controllers[route] = dict(after)

    initialized = (run / "initialization.json").exists()
    if not initialized:
        require(partial and not any(indexed.values()), "task rows without initialization")
        pending = {"kind": "initialization", "cursor": dict(cursor)}
        current, controllers = {}, {}
    else:
        initialization = read_json(run / "initialization.json")
        documents, demos, offers, initial_routes = check_initialization(initialization, manifest, rule)
        baselines = {route: [list(pair) for pair in zip(state["baseline_mean"],
                     state["baseline_count"], strict=True)] for route, state in initial_routes.items()}
        current = {r: {k: initial_routes[r][k + "_sha256"] for k in ("model", "memory")}
                   for r in ROUTES[:2]}
        current["simple"] = {"model": scalar.plain_digest(demos),
                             "memory": scalar.plain_digest(documents)}
        # Initialization seals the complete controller. Split RNG/slow hashes
        # start at the first actual row, in a distinct digest domain.
        controllers = dict.fromkeys(ROUTES[:2])
        initial_hashes = json.loads(encoded(current))
        launch = read_json(run / "launch-record.json")
        for name, identity in launch["artifacts"].items():
            require(scalar.file_sha(name) == identity, "qualified execution artifact")

        def record_boundary(kind, before):
            baseline = {"documents_hex": [d["bytes_hex"] for d in documents],
                        "demonstrations": [{k: v["bytes_hex"] if isinstance(v, dict) else v
                                            for k, v in d.items()} for d in demos],
                        "new_teacher_offers": offers}
            resolved.append(json.loads(encoded({
                "kind": kind, "before_cursor": before, "cursor": cursor, "counters": counts,
                "projects": projects, "current": actual_current, "hashes": current,
                "free": free, "actions": actions, "baseline": baseline,
            })))
            if (cursor["episode_index"] in CHECKPOINTS and cursor["route_index"] == 0
                    and cursor["attempt_index"] == 1 and not cursor["append_done"]):
                teaching_age = {}
                for route in ROUTES:
                    public_ticks = (stochastic[route]["by_category"]["public_scan"]["ticks"]
                                    if route in stochastic else None)
                    teaching_age[route] = {skill: None if skill not in corrections[route] else {
                        "main_attempts_since_correction": counts[route]["attempts"] - corrections[route][skill]["attempts"],
                        "source_symbols_since_correction": counts[route]["ingest_raw_bytes"] + counts[route]["ingest_eos"] - corrections[route][skill]["source"],
                        "public_scan_symbols_since_correction": None if public_ticks is None else
                            public_ticks - corrections[route][skill]["public"],
                        "scope": "actual persistent-actor exposures after correction; disposable probes excluded"}
                        for skill in ("A00-R", "A00-E")}
                probe_states[cursor["episode_index"]] = copy.deepcopy({
                    "boundary_sequence": len(resolved) - 1, "baselines": baselines,
                    "baseline": baseline, "free": free, "actions": actions,
                    "hashes": current, "controllers": controllers, "teaching_age": teaching_age})

        record_boundary("initialize", dict(cursor))
        docs = {d["document_id"]: d for d in manifest["documents"]}
        for ei, episode in enumerate(manifest["episodes"]):
            _admit(deadline)
            if pending:
                break
            for ri, route in enumerate(ROUTES):
                if pending:
                    break
                enabled, neural = route == "online", route != "simple"
                if episode["append_document_ids"]:
                    if consumed["ingests"] == len(indexed["ingests"]):
                        pending = {"kind": "append", "cursor": dict(cursor)}
                        break
                    row = read_row(indexed["ingests"][consumed["ingests"]])
                    doc = docs[episode["append_document_ids"][0]]
                    raw = bytes.fromhex(doc["text_hex"])
                    admitted = neural and doc["neural_tape_admission"]
                    require(row["schema"] == "persistent-language-action-ingest-v8"
                            and (row["route"], row["episode_id"], row["document_ids"], row["documents"])
                            == (route, episode["episode_id"], episode["append_document_ids"], [doc])
                            and row["raw_bytes"] == len(raw) and row["eos"] == 1
                            and row["source_kind"] == doc["source_kind"]
                            and row["neural_tape_admission"] is doc["neural_tape_admission"]
                            and row["fact_admitted"] is admitted and row["writes_enabled"] is admitted
                            and row["reference_retained"] is (not neural)
                            and row["learning_enabled"] is (route != "learning_disabled"),
                            "fixed raw source admission/exposure")
                    if neural:
                        require(row["model_before_sha256"] == current[route]["model"]
                                and row["memory_before_sha256"] == current[route]["memory"]
                                and row["memory_used_before"] == used[route]
                                and row["memory_used_after"] == used[route] + (len(raw) + 1 if admitted else 0),
                                "source neuronal state chronology")
                        trajectory = check_source(row["source_feedback"], raw, enabled, admitted,
                                                  baselines[route], rule)
                        require(row["source_feedback"]["memory_before_sha256"] == current[route]["memory"]
                                and row["source_feedback"]["memory_after_sha256"] == row["memory_after_sha256"],
                                "source feedback tape custody")
                        controller_transition(route,
                            {k: row["controller_" + n + "_before_sha256"]
                             for k, n in (("rng_sha256", "rng"), ("slow_sha256", "slow"))},
                            {k: row["controller_" + n + "_after_sha256"]
                             for k, n in (("rng_sha256", "rng"), ("slow_sha256", "slow"))}, advanced=True)
                        require(admitted or row["memory_after_sha256"] == current[route]["memory"],
                                "transient source allocated factual storage")
                        current[route] = {"model": row["model_after_sha256"], "memory": row["memory_after_sha256"]}
                        for value in current[route].values():
                            scalar.hash_value(value)
                        used[route] = row["memory_used_after"]
                        source_offers[route] += len(raw) + 1
                        add_trajectory(route, trajectory)
                    else:
                        require(row["source_feedback"] is None and all(row[k] is None for k in (
                            "model_before_sha256", "model_after_sha256", "memory_before_sha256",
                            "memory_after_sha256", "memory_used_before", "memory_used_after",
                            "controller_rng_before_sha256", "controller_rng_after_sha256",
                            "controller_slow_before_sha256", "controller_slow_after_sha256")),
                            "reference claimed neural source state")
                        documents.append({"bytes_hex": raw.hex()})
                        current[route]["memory"] = scalar.plain_digest(documents)
                        used[route] += len(raw) + 1
                    add_cost(row, "source", "source")
                    counts[route]["ingest_documents"] += 1
                    counts[route]["ingest_raw_bytes"] += len(raw)
                    counts[route]["ingest_eos"] += 1
                    consumed["ingests"] += 1
                    before = dict(cursor)
                    cursor["append_done"] = True
                    record_boundary("append", before)
                    del row
                project, feedback_input = episode["project_id"], None
                if episode["workspace_lifecycle"] == "carry_actual_project":
                    preceding = projects[route][project]
                    require(preceding["episode_id"] == episode["predecessor_episode_id"], "actual project predecessor")
                    files, feedback_input = preceding["files"], preceding["feedback"]
                elif episode["phase"] == "retention":
                    parent_route = "simple" if route == "simple" else "online"
                    old = manifest["prior_regression_bindings"]["routes"][parent_route][episode["prior_episode_id"]]
                    files, feedback_input = old["observation"]["files"] or [], old["observation"]["feedback"]
                else:
                    files = episode["files"]
                for ai in range(1, episode["attempt_limit"] + 1):
                    if consumed["attempts"] == len(indexed["attempts"]):
                        pending = {"kind": "attempt", "cursor": dict(cursor)}
                        break
                    entry = indexed["attempts"][consumed["attempts"]]
                    row = read_row(entry)
                    require(row["schema"] == "persistent-language-action-attempt-v8"
                            and (row["route"], row["episode_id"], row["attempt_index"])
                            == (route, episode["episode_id"], ai)
                            and all(row[k] == episode[k] for k in (
                                "phase", "kind", "request_hex", "project_id", "predecessor_episode_id", "workspace_lifecycle"))
                            and row["before_files"] == files, "actual fixed attempt schedule")
                    public = {"schema": "sequence-assembly-raw-observation-v1", "request_hex": episode["request_hex"],
                              "files": files if episode["kind"] == "edit" else None, "feedback": feedback_input}
                    require(row["status"] in ("completed", "unsupported"), "attempt status")
                    if row["status"] == "completed":
                        require(row["observation"] == public, "complete public observation, no grade input")
                    else:
                        require(row["prediction"] is None and row["proposal"] is None and row["answer_hex"] is None,
                                "unsupported input predicted")
                        require(row["observation"] == {"request_hex": episode["request_hex"],
                                                       "files": files, "feedback": feedback_input},
                                "unsupported input dropped actual raw state")
                    require(row["observation_record_sha256"] == digest(encoded(row["observation"]))
                            and row["model_before_sha256"] == current[route]["model"]
                            and row["memory_before_sha256"] == row["memory_after_sha256"] == current[route]["memory"]
                            and row["memory_used"] == used[route], "attempt state and observation custody")
                    if route in instances:
                        require(row["model_instance_id"] == instances[route], "actor replaced")
                    else:
                        require(row["model_instance_id"] not in instances.values(), "aliased acting routes")
                        instances[route] = row["model_instance_id"]
                    trajectory = prediction = None
                    if row["prediction"] is not None:
                        if not neural:
                            prior.check_reference(row, documents, demos)
                        else:
                            prediction = row["prediction"] if episode["kind"] == "reading" else row["prediction"]["payload"]
                            trajectory = check_prediction(prediction, public, used[route], responses, rule)
                            require(prediction["memory_sha256"] == current[route]["memory"], "prediction tape")
                            free[route] += 1
                            if episode["kind"] == "reading":
                                require(row["answer_hex"] == prediction["answer_hex"], "reading substitution")
                            else:
                                scalar.check_heads(row["prediction"], row["proposal"], files)
                                require(row["prediction"]["model_sha256"] == current[route]["model"], "action actor hash")
                                actions[route] += 1
                    require(row["feedback_consumed"] is (feedback_input is not None and row["prediction"] is not None),
                            "actual feedback consumption")
                    reward, outcome = None, None
                    if episode["kind"] == "edit":
                        feedback_input = check_execution(row, episode, run, launch)
                        correct = row["status"] != "unsupported" and row["grade"]["task_success"]
                        ex, proposal = row["execution"], row["proposal"]
                        failed = (not ex["accepted"] or ex["returncode"] != 0 or ex["timeout"]
                                  or not ex["capture_complete"] or not ex["files_complete"] or ex["output_limit_exceeded"])
                        reward = -int(failed and proposal is not None and proposal["operation"] != "STOP")
                        if proposal is None:
                            outcome = "no_proposal"
                        elif not ex["accepted"]:
                            outcome = "rejected_proposal"
                        elif proposal["operation"] == "STOP":
                            outcome = "correct_stop" if correct else "incorrect_stop"
                        elif not correct:
                            outcome = "incorrect_accepted_splice"
                        else:
                            outcome = ("accepted_correct_splice_with_change" if ex["proposed_files"] != ex["before_files"]
                                       else "accepted_correct_splice_without_change")
                        counts[route]["executions"] += 1
                    else:
                        require(row["execution"] is None and row["after_files"] == files, "reading performed action")
                        correct = row["status"] != "unsupported" and row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                    require(row["correct"] is bool(correct) and row["execution_outcome"] == outcome, "actual private outcome")
                    signature = None if row["proposal"] is None else digest(encoded(
                        {k: v for k, v in row["proposal"].items() if k != "workspace_sha256"}))
                    require(row["proposal_signature"] == signature, "actual proposal signature")
                    require(row["model_after_sha256"] == current[route]["model"]
                            if not neural or episode["teacher"] is not None or prediction is None else True,
                            "unresolved or unsupported response changed model")
                    if neural:
                        controller_transition(route, row["controller_before"], row["controller_after"],
                                              advanced=prediction is not None and episode["teacher"] is None)
                        if trajectory is not None and episode["teacher"] is None:
                            check_resolution(row["controller_resolution"], trajectory, episode["kind"], enabled,
                                             baselines[route], reward)
                            add_trajectory(route, trajectory)
                    current[route]["model"] = row["model_after_sha256"]
                    counts[route]["attempts"] += 1
                    compact.append({k: row[k] for k in (
                        "route", "episode_id", "phase", "kind", "attempt_index", "correct", "status", "project_id",
                        "predecessor_episode_id", "workspace_lifecycle", "memory_used", "feedback_consumed",
                        "proposal_signature", "public_checks_passed", "execution_outcome")}
                        | {"observation_sha256": row["observation_record_sha256"],
                           "rng_before_sha256": row["controller_before"]["rng_sha256"] if neural else None,
                           "diagnostics": None if row["execution"] is None else {
                               key: row["execution"][key] for key in ("stdout_hex", "stderr_hex", "capture_errors")},
                           "behavior_sha256": _behavior_signature(row)})
                    consumed["attempts"] += 1
                    add_cost(row, "free", "attempt")
                    if row["execution"] is not None and not row["execution"]["files_complete"]:
                        require(partial, "complete session has incomplete actual file capture")
                        if trajectory is not None and episode["teacher"] is not None:
                            add_trajectory(route, trajectory)
                        pending = {"kind": "incomplete_execution_after_durable_attempt", "cursor": dict(cursor),
                                   "attempt_sha256": entry["sha256"],
                                   "scope": "captured subset retained; no safe continuation or invented missing files"}
                        break
                    if episode["teacher"] is not None:
                        if consumed["teachers"] == len(indexed["teachers"]):
                            if trajectory is not None:
                                add_trajectory(route, trajectory)
                            pending = {"kind": "correction_after_durable_attempt", "cursor": dict(cursor),
                                       "attempt_sha256": entry["sha256"]}
                            break
                        taught = read_row(indexed["teachers"][consumed["teachers"]])
                        target_record = episode["teacher"]
                        require(taught["schema"] == "persistent-language-action-teacher-v8"
                                and (taught["route"], taught["episode_id"], taught["teacher"], taught["attempt_sha256"])
                                == (route, episode["episode_id"], target_record, entry["sha256"])
                                and taught["learning_enabled"] is (route != "learning_disabled")
                                and taught["model_before_sha256"] == current[route]["model"]
                                and taught["memory_before_sha256"] == taught["memory_after_sha256"] == current[route]["memory"],
                                "causal actual correction identity/state")
                        if not neural:
                            label = "reading" if episode["kind"] == "reading" else target_record["operation"]
                            offers[label] += 1
                            require(taught["reader_offers"] == taught["action_labels"] == 0 and not taught["work"],
                                    "reference neural correction")
                            if label == "SPLICE":
                                scalar.append_demonstration(demos, target_record, row, scalar.similarity_query(public))
                                require(taught["publication"]["stored"] is True
                                        and taught["publication"]["demonstration_index"] == len(demos) - 1,
                                        "reference supplied demonstration")
                            else:
                                require(taught["publication"]["stored"] is False, "reference stored reading/STOP target")
                            require(taught["model_after_sha256"] == scalar.plain_digest(demos), "reference correction state")
                        else:
                            require(prediction is not None and trajectory is not None, "correction without actual response")
                            target = bytes.fromhex(target_record.get("target_hex", target_record.get("replacement_hex")))
                            n = check_teacher_feedback(taught["reader_feedback"], prediction, target, used[route],
                                trajectory, enabled, baselines[route], episode["kind"], reward)
                            require(taught["reader_offers"] == n and taught["action_labels"] == (3 if episode["kind"] == "edit" else 0),
                                    "actual correction offer/label counts")
                            require(taught["controller_resolution"] == taught["reader_feedback"]["controller"],
                                    "single controller correction publication")
                            publication = taught["publication"]
                            require(publication["address"] is False and publication["legacy_codec"] is False
                                    and type(publication["output"]) is bool and (enabled or not publication["output"]),
                                    "held old reader/control gate")
                            if episode["kind"] == "edit":
                                require(set(publication["actions"]) == {"start", "end", "operation"}
                                        and all(type(v) is bool and (enabled or not v) for v in publication["actions"].values()),
                                        "motor correction gate")
                            controller_transition(route, taught["controller_before"], taught["controller_after"], advanced=True)
                            add_trajectory(route, trajectory)
                        scalar.hash_value(taught["model_after_sha256"])
                        current[route]["model"] = taught["model_after_sha256"]
                        counts[route]["teacher_occasions"] += 1
                        counts[route]["reader_offers"] += taught["reader_offers"]
                        counts[route]["action_labels"] += taught["action_labels"]
                        corrections[route][episode["episode_id"]] = {
                            "attempts": counts[route]["attempts"],
                            "source": counts[route]["ingest_raw_bytes"] + counts[route]["ingest_eos"],
                            "public": stochastic[route]["by_category"]["public_scan"]["ticks"] if neural else None}
                        consumed["teachers"] += 1
                        add_cost(taught, "teacher", "teacher")
                        del taught
                    files = row["after_files"]
                    actual_current = {"route": route, "episode_id": episode["episode_id"], "files": files, "feedback": feedback_input}
                    if project:
                        projects[route][project] = {k: v for k, v in actual_current.items() if k != "route"}
                    before = dict(cursor)
                    cursor = ({"episode_index": ei, "route_index": ri, "attempt_index": ai + 1, "append_done": True}
                              if ai < episode["attempt_limit"] else
                              {"episode_index": ei + int(ri == 2), "route_index": (ri + 1) % 3,
                               "attempt_index": 1, "append_done": False})
                    record_boundary("attempt", before)
                    del row
    require(all(consumed[k] == len(indexed[k]) for k in indexed), "unplanned durable rows beyond chronological prefix")
    require(partial or pending is None, "complete run omitted scheduled outcomes")
    checkpoint = check_checkpoints(run, sources, manifest_sha, indexed, partial, resolved,
                                   frozen_auditor_path=frozen_auditor_path)
    if not partial:
        require(consumed == {"attempts": 192, "teachers": 6, "ingests": 72}
                and checkpoint["boundaries"] == 265
                and all(used[r] == 1982 for r in ROUTES[:2])
                and all(source_offers[r] == 9884 for r in ROUTES[:2]),
                "complete V8 exposure scope")
        report = read_json(run / "report.json")
        require(report["schema"] == "stable-language-action-credit-report-v1" and report["status"] == "completed"
                and report["counts"] == consumed and report["actual_executions"] == 132
                and report["initial_hashes"] == initial_hashes
                and report["final_hashes"] == current
                and report["final_reader_experience"] == {r: 278 + free[r] for r in free}
                and report["final_model_experience"] == {r: 325 + actions[r] for r in actions}
                and report["final_controller"] == controllers, "complete report state/counters")
        require(report["budget"]["complete_seconds"] == 10800 and report["budget"]["peak_mib"] == 4096,
                "declared complete worker budget")
    telemetry_counts, telemetry_seconds, previous_work = Counter(), Counter(), {}
    between_phase_work = Counter()
    for entry in telemetry:
        item = read_row(entry)
        require(item["schema"] == "persistent-language-action-checkpoint-work-v8", "checkpoint telemetry schema")
        telemetry_counts[item["phase"]] += 1
        telemetry_seconds[item["phase"]] += prior.elapsed_value(item["elapsed_seconds"])
        work_valid(item["cumulative_work"])
        require(all(item["cumulative_work"].get(k, 0) >= v for k, v in previous_work.items()), "checkpoint work reset")
        if "work_delta" in item:
            row_bytes = ()
            if item["phase"] == "capture":
                index = telemetry_counts["capture"] - 1
                require(index < len(checkpoint["boundary_events"]), "capture lacks saved boundary")
                row_bytes = checkpoint["boundary_events"][index]["journal_appended_row_bytes"]
            between_phase_work.update(_phase_work_gap(previous_work, item, row_bytes))
        previous_work = item["cumulative_work"]
    if not partial:
        require(telemetry_counts == {"create": 1, "begin": 264, "capture": 265, "commit": 265}, "checkpoint complete phase counts")
    result_metrics = metrics(compact, manifest)
    if not partial:
        for route in ROUTES:
            actual, saved = result_metrics[route], report["metrics"][route]
            for key in ("retained", "forgotten", "improved", "public_checks_passed"):
                require(saved[key] == actual[key], "independent report metric: " + key)
            require(Counter(saved["execution_outcomes"]) == Counter(actual["execution_outcomes"]),
                    "independent action outcome counts")
            for key in ("retention", "same_session_encounters", "project_chains"):
                require(len(saved[key]) == len(actual[key]), "report detail count")
                for left, right in zip(saved[key], actual[key], strict=True):
                    require(all(left[name] == right[name] for name in set(left) & set(right)),
                            "report observed detail")
            require(saved["same_session_exact_retained"] == sum(
                r["encounter"] == "exact_delayed" and r["initial_correct"] and r["correct"]
                for r in actual["same_session_encounters"]), "report exact repeat retention")
            for detail in saved["same_session_encounters"]:
                episode = next(e for e in manifest["episodes"] if e["episode_id"] == detail["episode_id"])
                require(detail["origin"] == episode["same_session_origin_id"], "report exact encounter origin")
            for phase in ("acquisition", "transfer", "retention", "recovery"):
                for kind in ("reading", "edit"):
                    group = actual["groups"][phase][kind]
                    row = saved["phases"][phase][kind]
                    require(row["episodes"] == group["observed"] and row["first_correct"] == group["correct"]
                            and row["attempts"] == sum(r["phase"] == phase and r["kind"] == kind for r in compact if r["route"] == route)
                            and row["unsupported"] == sum(r["phase"] == phase and r["kind"] == kind and r["status"] == "unsupported"
                                                          for r in compact if r["route"] == route), "report phase counts")
            for name in ("first", "changed_form", "exact_delayed"):
                group = actual["groups"][name]
                require(saved["acquisition_encounters"][name]["episodes"] == sum(g["observed"] for g in group.values())
                        and saved["acquisition_encounters"][name]["first_correct"] == sum(g["correct"] for g in group.values())
                        and saved["acquisition_encounters"][name]["requirement_changed"] == sum(
                            bool(e.get("requirement_changed")) for e in manifest["episodes"]
                            if e["phase"] == "acquisition" and e.get("encounter") == name),
                        "report acquisition encounters")
            for name in ("instruction_construction", "structural_combination"):
                group = actual["groups"][name]
                row = saved["exposed_changed_context_strata"][name]
                require(row["episodes"] == sum(g["observed"] for g in group.values())
                        and row["first_correct"] == sum(g["correct"] for g in group.values())
                        and row["unsupported"] == sum(g["observed"] - g["supported"] for g in group.values()),
                        "report fresh strata")
                for label, dependent in (("independent", False), ("actual_project_dependent", True)):
                    members = [r for r in compact if r["route"] == route and r["attempt_index"] == 1
                               and bool(r["predecessor_episode_id"]) == dependent
                               and next(e for e in manifest["episodes"] if e["episode_id"] == r["episode_id"]).get("fresh_stratum") == name]
                    require(row["dependency"][label] == {"episodes": len(members), "first_correct": sum(r["correct"] for r in members),
                                                         "unsupported": sum(r["status"] == "unsupported" for r in members)},
                            "report actual project dependency split")
            old, row = actual["recovery"], saved["recovery"]
            for name, source in (("episodes", "observed_episodes"), ("first_correct", "first_correct"),
                                 ("later_correct", "ever_recovered"), ("final_correct", "terminal_correct"),
                                 ("final_recovered", "terminal_recovered"),
                                 ("feedback_consuming_attempts", "feedback_consuming_attempts"),
                                 ("identical_consecutive_proposals", "identical_consecutive_proposals")):
                require(row[name] == old[source], "report recovery metric")
    evidence = dict(completion["files"])
    return {"schema": "stable-language-action-credit-worker-review-v1", "status": "passed",
            "complete_run": not partial, "run": str(run), "manifest_sha256": manifest_sha,
            "source_revision": report["runtime"]["old_source_revision"] if rule == "old" else read_json(run / "checkpoints/config.json")["source_revision"],
            "saved_tool_sources": saved_tool_sources,
            "auditor_provenance": auditor_provenance,
            "attempt_rows": len(compact), "teacher_rows": consumed["teachers"], "ingest_rows": consumed["ingests"],
            "exposures": counts, "source_offers": source_offers, "controller_exposures": stochastic,
            "controller_exposure_scope": "checked saved ticks only; status includes its framing EOS and is disjoint from raw_content and other eos offers; free has no target or match count; aggregate matches are not a temporal improvement or task-success claim",
            "saved_costs": costs, "metrics": result_metrics, "compact_attempts": compact,
            "probe_states": probe_states, "credit_rule": rule,
            "pending_transaction": pending, "checkpoint": checkpoint, "journal_frontiers": frontiers,
            "checkpoint_telemetry_counts": dict(telemetry_counts), "checkpoint_telemetry_seconds": dict(telemetry_seconds),
            "checkpoint_between_phase_work": dict(between_phase_work),
            "checkpoint_phase_work_scope": "phase-start snapshots were not saved; phase deltas plus source/row-derived native journal gaps reconcile cumulative counters; final cleanup remains in final store work",
            "checkpoint_telemetry_frontier": telemetry_frontier, "evidence": evidence,
            "payload_read_scope": "complete inventory (complete only), independent checkpoint authentication, fresh store inspect, and final evidence inventory are distinct real byte-read passes",
            "elapsed_seconds": time.perf_counter() - started, "model_constructions": 0, "checkpoint_decodes": 0,
            "predictor_calls": 0, "actual_executions": 0, "teacher_replays": 0,
            "cost_scope": "saved category counters; attempt timer includes its recorded begin interval and execution; no isolated scan/free latency",
            "scope": "durable actual prefix, local scalar receipts and custody; no reconstruction of omitted eligibility, synapses, gates or interrupted unsaved exposure"}


def review(comparison_root, manifest_sha, *, frozen_auditor_path=None):
    """Read existing bytes only and return an audit, including an honest failed prefix."""
    started = time.perf_counter()
    deadline = started + AUDIT_SECONDS
    root = Path(comparison_root).resolve()
    result = {"schema": "stable-language-action-credit-comparison-review-v1",
              "passed": False, "status": "incomplete", "comparison_root": str(root),
              "manifest_sha256": manifest_sha, "workers": {}, "checkpoints": {},
              "audit_budget_seconds": AUDIT_SECONDS, "model_constructions": 0,
              "tensor_decodes": 0, "predictor_calls": 0, "execution_children": 0,
              "teacher_replays": 0,
              "scope": "saved bytes, scalar receipts and actual outcomes; no new trial or omitted-state reconstruction"}
    try:
        scalar.hash_value(manifest_sha)
        for rule in ("old", "causal"):
            _admit(deadline)
            run = root / rule
            worker = review_worker(run, manifest_sha, deadline,
                                   frozen_auditor_path=frozen_auditor_path)
            require(worker["credit_rule"] == rule, "comparison worker source assignment")
            result["workers"][rule] = worker
            result["checkpoints"][rule] = review_probes(run, worker, manifest_sha, deadline)
        old, causal = result["workers"]["old"], result["workers"]["causal"]
        before = read_json(root / "old/initialization.json")
        after = read_json(root / "causal/initialization.json")
        require(all(before[key] == after[key] for key in (
            "source_numeric_model_sha256", "source_model_sha256", "source_memory_sha256",
            "baseline_record", "prior_host", "prior_frontiers")), "old/new lineage or history changed")
        for route in ROUTES[:2]:
            require(all(before["initial_routes"][route][key] == after["initial_routes"][route][key]
                        for key in ("numeric_model_sha256", "memory_sha256", "controller_rng_sha256",
                                    "baseline_mean", "baseline_count", "views_sha256", "self_wiring_sha256")),
                    "migration numerical/derived/RNG preservation")
            old_metadata = before["initial_routes"][route]["metadata_adaptation"]
            new_metadata = after["initial_routes"][route]["metadata_adaptation"]
            schemas = {"_extra_state": "recurrent-content-splice-learner-",
                       "reader._extra_state": "recurrent-content-memory-learner-",
                       "reader.controller._extra_state": "recurrent-content-controller-"}
            allowed = {"_extra_state": {"schema", "reader"},
                       "reader._extra_state": {"schema", "publication", "initialization", "local_credit"},
                       "reader.controller._extra_state": {"schema", "local_rule", "upstream_objective",
                           "literal_objective", "public_consequence", "update_norm_bound", "update_norm_tolerance"}}
            require(set(old_metadata) == set(new_metadata) == set(schemas), "explicit three metadata adapters")
            for name, schema in schemas.items():
                previous, current = old_metadata[name], new_metadata[name]
                require(previous["before"] == previous["after"] == current["before"]
                        and current["before"]["schema"] == schema + "v1"
                        and current["after"]["schema"] == schema + "v2", "strict old/exact new metadata lineage")
                require({key: value for key, value in current["before"].items() if key not in allowed[name]}
                        == {key: value for key, value in current["after"].items() if key not in allowed[name]},
                        "metadata changed beyond selected rule declaration")
        cp_old, cp_new = result["checkpoints"]["old"], result["checkpoints"]["causal"]
        migration = {}
        for route in ROUTES:
            a, b = cp_old[0]["routes"][route], cp_new[0]["routes"][route]
            require(a["initial"] == b["initial"], "actual CP0 starting numeric state changed")
            migration[route] = _paired(a["attempts"], b["attempts"], migration=True)
            require(migration[route]["passed"], "actual CP0 forward/task migration mismatch")
        result["migration"] = migration
        comparisons, transitions, simple_parity = {}, {}, {}
        enabled_disabled = {}
        main_contrasts = {}
        for rule, worker in (("old", old), ("causal", causal)):
            enabled_disabled[rule] = {}
            enabled = worker["metrics"]["online"]
            disabled = worker["metrics"]["learning_disabled"]
            main_contrasts[rule] = {"enabled_minus_disabled": {
                name: {kind: {"episodes": enabled["groups"][name][kind]["observed"],
                              "enabled_correct": enabled["groups"][name][kind]["correct"],
                              "disabled_correct": disabled["groups"][name][kind]["correct"],
                              "difference": enabled["groups"][name][kind]["correct"] -
                                            disabled["groups"][name][kind]["correct"]}
                       for kind in ("reading", "edit")} for name in enabled["groups"]},
                "recovery": {key: {"enabled": enabled["recovery"][key],
                                   "disabled": disabled["recovery"][key],
                                   "difference": enabled["recovery"][key] - disabled["recovery"][key]}
                             for key in ("first_correct", "ever_recovered", "terminal_correct", "terminal_recovered")},
                "actual_input_pairs": _paired(
                    [r for r in worker["compact_attempts"] if r["route"] == "online"],
                    [r for r in worker["compact_attempts"] if r["route"] == "learning_disabled"]),
                "scope": "descriptive fixed-roster scores; actual contexts may diverge"}
            for cp in CHECKPOINTS:
                routes = result["checkpoints"][rule][cp]["routes"]
                a, b = routes["online"], routes["learning_disabled"]
                held = result["checkpoints"][rule][0]["routes"]["learning_disabled"]["initial"]
                require(all(b["initial"][key] == held[key] for key in (
                    "slow_numeric_sha256", "factual_sha256", "memory_used")),
                    "persistent disabled route changed slow state across learning checkpoints")
                enabled_disabled[rule][cp] = {
                    "groups": {group: {"episodes": len(ids), **{metric: {
                        "enabled": a["groups"][group][metric], "disabled": b["groups"][group][metric],
                        "difference": a["groups"][group][metric] - b["groups"][group][metric]}
                        for metric in ("first_correct", "final_correct", "recovered")}}
                        for group, ids in PROBE_GROUPS},
                    "actual_pairs": _paired(a["attempts"], b["attempts"]),
                    "scope": "descriptive fixed-roster scores; paired contrasts require actual inputs/RNG"}
        for cp in CHECKPOINTS:
            comparisons[cp], simple_parity[cp] = {}, _paired(
                cp_old[cp]["routes"]["simple"]["attempts"], cp_new[cp]["routes"]["simple"]["attempts"])
            require(simple_parity[cp]["passed"], "duplicate simple changed behavior on matched actual input")
            for route in ROUTES[:2]:
                a, b = cp_old[cp]["routes"][route], cp_new[cp]["routes"][route]
                comparisons[cp][route] = {"actual_pairs": _paired(a["attempts"], b["attempts"]),
                    "group_scores": {group: {"episodes": len(ids),
                        "old_first_correct": a["groups"][group]["first_correct"],
                        "causal_first_correct": b["groups"][group]["first_correct"],
                        "difference": b["groups"][group]["first_correct"] - a["groups"][group]["first_correct"]}
                        for group, ids in PROBE_GROUPS},
                    "scope": "descriptive same-roster scores; paired causal contrast requires actual input/RNG pairing"}
        for rule, batteries in (("old", cp_old), ("causal", cp_new)):
            transitions[rule] = {}
            for route in ROUTES:
                route_transitions = []
                for earlier, later in zip(CHECKPOINTS, CHECKPOINTS[1:]):
                    for group, _ in PROBE_GROUPS:
                        a = batteries[earlier]["routes"][route]["groups"][group]["details"]
                        b = batteries[later]["routes"][route]["groups"][group]["details"]
                        route_transitions.append({"from_checkpoint": earlier, "to_checkpoint": later,
                            "group": group, "episodes": len(a),
                            "retained": sum(x["first_correct"] and y["first_correct"] for x, y in zip(a, b, strict=True)),
                            "lost": sum(x["first_correct"] and not y["first_correct"] for x, y in zip(a, b, strict=True)),
                            "gained": sum(not x["first_correct"] and y["first_correct"] for x, y in zip(a, b, strict=True)),
                            "scope": "scheduled exposed checkpoint outcomes, with full per-skill outcomes above"})
                transitions[rule][route] = route_transitions
        symbols = sum(worker["exposures"][route]["reader_offers"]
                      for worker in (old, causal) for route in ROUTES[:2])
        labels = sum(worker["exposures"][route]["action_labels"]
                     for worker in (old, causal) for route in ROUTES[:2])
        require(symbols == 140 and labels == 12, "fixed complete neural teaching budget changed")
        main_simple = _paired([r for r in old["compact_attempts"] if r["route"] == "simple"],
                              [r for r in causal["compact_attempts"] if r["route"] == "simple"])
        require(main_simple["passed"], "main duplicate simple changed on matched actual input")
        result.update(passed=True, status="passed", old_new=comparisons,
                      enabled_minus_disabled=enabled_disabled, main_strata_contrasts=main_contrasts,
                      checkpoint_transitions=transitions, simple_duplicate_parity=simple_parity,
                      main_simple_duplicate_parity=main_simple,
                      neural_teacher_symbols=symbols, neural_motor_labels=labels,
                      attempts=1056, actual_saved_executions=672,
                      measurement_scope="all fixed exposed taught skills and declared probes; no fresh transfer or broader completeness claim")
        for worker in result["workers"].values():
            worker.pop("probe_states")
        _admit(deadline)
    except Exception as error:
        result.update(passed=False, status="failed", error_type=type(error).__name__, error=str(error),
                      failure_scope="retained validated prefix only; missing/interrupted evidence is not inferred")
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
