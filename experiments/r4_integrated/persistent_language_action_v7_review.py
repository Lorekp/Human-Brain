"""V7 saved evidence review, including an explicitly incomplete durable prefix.

Only stdlib and old independent scalar reviewers import here. No model, task
builder, reference predictor, checkpoint decoder or execution helper is called.
Contact activation is not reconstructed from compact receipts: tuple arbitration,
causal offers, stage/publication gates, raw actions and byte custody are checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_review_v4 as scalar
from experiments.r4_integrated import persistent_language_action_v6_review as history

DRAFT = False
MANIFEST_SHA = "c2e1cb6237fdff468c6d413ce62c9dfbf514a607fb1a8af45bdde285c3f4b898"
ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("online", "learning_disabled", "simple")
ROW_LIMIT = 64 * 1024 * 1024
JSON_LIMIT = 16 * 1024 * 1024
COUNTERS = (
    "attempts",
    "executions",
    "teacher_occasions",
    "reader_offers",
    "action_labels",
    "ingest_documents",
    "ingest_raw_bytes",
    "ingest_eos",
)
SAVED_ONLY_SOURCES = {
    "experiments/r4_integrated/persistent_language_action_v7_review.py",
    "experiments/r4_integrated/bank_persistent_language_action_v7.py",
}
require = scalar.require


def encoded(value):
    return scalar.json_bytes(value) + b"\n"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def elapsed_value(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid elapsed")
    return value


def read_json(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(JSON_LIMIT + 1)
    require(len(raw) <= JSON_LIMIT, "bounded JSON exceeded")
    return json.loads(raw)


def safe_path(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path != root.resolve(), "artifact escaped root")
    return path


def journal(path, *, partial):
    """Read each saved line once; an unfinished last line is evidence, not a row."""
    rows, prefix, size, tail = [], hashlib.sha256(), 0, None
    if not path.exists():
        require(partial, "required journal absent")
        return rows, {"bytes": 0, "rows": 0, "sha256": prefix.hexdigest(), "tail": None}
    with path.open("rb") as handle:
        while raw := handle.readline(ROW_LIMIT + 1):
            require(len(raw) <= ROW_LIMIT, "journal row exceeds explicit bound")
            if not raw.endswith(b"\n"):
                require(partial and not handle.read(1), "malformed nonterminal journal line")
                tail = {"bytes": len(raw), "sha256": digest(raw), "offset": size}
                break
            prefix.update(raw)
            rows.append(
                {
                    "path": str(path),
                    "offset": size,
                    "bytes": len(raw),
                    "sha256": digest(raw),
                    "end": size + len(raw),
                    "prefix_sha256": prefix.hexdigest(),
                }
            )
            size += len(raw)
    return rows, {"bytes": size, "rows": len(rows), "sha256": prefix.hexdigest(), "tail": tail}


def read_row(index):
    with Path(index["path"]).open("rb") as handle:
        handle.seek(index["offset"])
        raw = handle.read(index["bytes"])
    require(
        len(raw) == index["bytes"] and digest(raw) == index["sha256"], "indexed journal changed"
    )
    return json.loads(raw)


def work_valid(work):
    require(isinstance(work, dict), "missing work ledger")
    for value in work.values():
        if isinstance(value, dict):
            work_valid(value)
        else:
            require(type(value) is int and value >= 0, "invalid operation count")


def check_binding(row, prefix, *, request_present):
    require(row["schema"] == "dendritic-binding-decision-v1", "binding decision schema")
    require(row["prefix_hex"] == prefix.hex(), "binding prefix is not causal input")
    scalar.hash_value(row["bank_sha256"])
    if request_present:
        scalar.hash_value(row["request_sha256"])
    else:
        require(row["request_sha256"] is None, "source received a task request")
    classes = scalar.vector(row["classes"], 257, integers=True)
    prior = scalar.vector(row["prior"], 257)
    require(all(value in (0, 1, 2) for value in classes), "invalid output class")
    if not request_present:
        require(
            2 not in classes and all(value == 0 for value in prior), "source apical/prior input"
        )
    winner = max(range(257), key=lambda i: (classes[i], prior[i], -i))
    require(row["winner"] == winner, "actual tuple arbitration differs")
    active, levels = list(range(257)), {}
    while len(active) > 1:
        following = []
        for i in range(0, len(active) - 1, 2):
            a, b = active[i : i + 2]
            require(math.isfinite(prior[a] - prior[b]), "tuple comparator overflow")
            following.append(a if (classes[a], prior[a]) >= (classes[b], prior[b]) else b)
        if len(active) % 2:
            following.append(active[-1])
        levels[str(len(levels))] = following
        active = following
    require(
        row["competition"] == {"winner": winner, "pairs": 256, "levels": levels},
        "saved tournament identity relay differs",
    )
    selected = row["selected_branch"]
    require((selected is None) == (classes[winner] == 0), "branch/class disagreement")
    if selected is not None:
        require(
            type(selected) is int and winner * 64 <= selected < (winner + 1) * 64,
            "branch belongs to another output column",
        )
        matches = scalar.vector(row["selected_matches"], 4, integers=True)
        require(
            all(0 <= x <= cap for x, cap in zip(matches, (16, 8, 16, 8), strict=True)),
            "selected match capacity",
        )
    else:
        require(row["selected_matches"] is None, "absent branch supplied matches")
    require(type(row["stage_revision"]) is int and row["stage_revision"] >= 0, "stage revision")
    return winner


def check_update(decision, update, target, enabled):
    require(update["schema"] == "dendritic-binding-update-v1", "binding update schema")
    require(
        update["symbol"] == target and update["winner_before"] == decision["winner"],
        "offered symbol/winner custody",
    )
    require(
        update["before_bank_sha256"] == decision["bank_sha256"]
        and update["stage_revision_before"] == decision["stage_revision"],
        "stage input seal",
    )
    scalar.hash_value(update["proposal_sha256"])
    for key in ("staged", "recruited", "attached_apical", "exhausted"):
        require(type(update[key]) is bool, "invalid update flag")
    require(
        type(update["changed_contacts"]) is int and update["changed_contacts"] >= 0,
        "contact change count",
    )
    require(not update["staged"] or enabled, "disabled private stage advanced")
    require(
        update["staged"]
        is bool(
            enabled
            and (update["recruited"] or update["attached_apical"] or update["changed_contacts"])
        ),
        "stage gate",
    )
    require(
        update["stage_revision_after"] == update["stage_revision_before"] + int(update["staged"]),
        "stage advancement differs",
    )
    if update["proposed_bank_sha256"] is not None:
        scalar.hash_value(update["proposed_bank_sha256"])
    if update["staged"]:
        require(update["proposed_bank_sha256"] is not None, "staged bank has no identity")
    elif enabled:
        require(
            update["proposed_bank_sha256"] == decision["bank_sha256"], "unchanged stage identity"
        )
    else:
        require(update["proposed_bank_sha256"] is None, "disabled proposed stage identity")
    selected = update["selected_branch"]
    require((selected is None) == update["exhausted"], "exhaustion identity")
    if selected is not None:
        require(
            type(selected) is int and target * 64 <= selected < (target + 1) * 64,
            "target branch column",
        )
    require(
        all(
            type(i) is int and 0 <= i < 16448 and i // 64 != target
            for i in update["false_prediction_branches"]
        ),
        "negative branch identity",
    )
    contacts = update["selected_contacts"]
    require(set(contacts) == {str(i) for i in range(len(contacts))}, "contact record ordering")
    for contact in contacts.values():
        compartment, n = contact["compartment"], contact["count"]
        require(
            compartment in range(4) and type(n) is int and 0 <= n <= (16, 8, 16, 8)[compartment],
            "contact compartment capacity",
        )
        indices, active, before, after = (
            scalar.vector(contact[key], n, integers=True)
            for key in ("indices", "activity", "before", "after")
        )
        require(
            len(set(indices)) == n
            and all(0 <= i < (8256, 256, 66564, 2048)[compartment] for i in indices)
            and all(a in (0, 1) for a in active)
            and all(0 <= p <= 3 for p in before + after)
            and contact["threshold"] == (3 * n + 3) // 4,
            "bounded local contact receipt",
        )
        expected = (
            [1] * n
            if contact["created"]
            else [min(3, max(0, p + 2 * a - 1)) for p, a in zip(before, active, strict=True)]
        )
        require(after == expected, "selected contact local permanence arithmetic")
    return bool(
        update["staged"]
        and (update["changed_contacts"] or update["recruited"] or update["attached_apical"])
    )


def check_prediction(row, observation, used, old_bank, responses):
    require(row["schema"] == "dendritic-sequence-output-prediction-v1", "free reader schema")
    require(row["attempt_id"] not in responses, "duplicate actual response")
    responses.add(row["attempt_id"])
    require(row["observation_sha256"] == scalar.observation_hash(observation), "free observation")
    for key in ("model_sha256", "memory_sha256", "trace_sha256"):
        scalar.hash_value(row[key])
    steps = row["steps"]
    require(
        row["max_bytes"] == 128 and 1 <= len(steps) <= 129 and row["decisions"] == len(steps),
        "free output limit",
    )
    prefix, bank_hash, request_hash = b"", None, None
    for i, step in enumerate(steps):
        require(step["output_step"] == i and step["prefix_hex"] == prefix.hex(), "own free prefix")
        scalar.memory_winner(step["address_scores"], step["selected_index"], used)
        potentials = scalar.vector(step["assembly_scores"], 398)
        selected = max(range(398), key=lambda j: (potentials[j], -j))
        require(step["assembly_index"] == selected, "prior all-owner competition")
        prior = scalar.vector(step["prior_logits"], 257)
        scalar.same_vector(prior, old_bank[selected], "held owner outgoing prior changed")
        token = check_binding(step["binding"], prefix, request_present=True)
        require(step["token"] == token and step["exact_context"] is None, "free emitted token")
        scalar.same_vector(step["binding"]["prior"], prior, "tuple prior substituted")
        scalar.close(step["prior_nll"], -scalar.log_softmax(prior)[token], "prior diagnostic NLL")
        if bank_hash is not None:
            require(
                step["binding"]["bank_sha256"] == bank_hash
                and step["binding"]["request_sha256"] == request_hash,
                "free bank/request changed",
            )
        bank_hash, request_hash = step["binding"]["bank_sha256"], step["binding"]["request_sha256"]
        if token == 256 or i == 128:
            require(i == len(steps) - 1, "generation continued after termination")
        else:
            prefix += bytes([token])
    require(steps[-1]["token"] == 256 or len(steps) == 129, "premature non-EOS limit")
    require(
        row["answer_hex"] == prefix.hex()
        and row["tokens"] == [s["token"] for s in steps]
        and row["stop_reason"] == ("eos" if steps[-1]["token"] == 256 else "limit"),
        "generation payload/stop mismatch",
    )
    work_valid(row["work"])
    require(row["work"].get("output_decisions") == len(steps), "decision work count")
    return len(steps)


def check_source_steps(feedback, raw_documents, enabled):
    steps = feedback["steps"]
    expected = [
        (d, i, raw[:i], token)
        for d, raw in enumerate(raw_documents)
        for i, token in enumerate([*raw, 256])
    ]
    require(len(steps) == len(expected), "source must offer every byte/EOS once")
    changed, following, revision = False, None, 0
    for step, (document, position, prefix, token) in zip(steps, expected, strict=True):
        require(
            (step["document_index"], step["output_step"], step["observed_token"])
            == (document, position, token),
            "source causal ordering",
        )
        check_binding(step["binding"], prefix, request_present=False)
        require(step["binding"]["stage_revision"] == revision, "source stage revision continuity")
        if following is not None:
            require(step["binding"]["bank_sha256"] == following, "source staged continuity")
        changed |= check_update(step["binding"], step["update"], token, enabled)
        require(
            not step["update"]["attached_apical"]
            and not step["update"]["false_prediction_branches"]
            and all(c["compartment"] == 0 for c in step["update"]["selected_contacts"].values()),
            "source performed answer/request learning",
        )
        revision = step["update"]["stage_revision_after"]
        following = (
            step["update"]["proposed_bank_sha256"]
            if step["update"]["staged"]
            else step["binding"]["bank_sha256"]
        )
    require(feedback["binding_updated"] is changed, "source publication flag")
    return len(steps)


def similarity(a, b):
    def counts(raw):
        raw = b" " + raw.lower() + b" "
        return Counter(raw[i : i + 3] for i in range(max(0, len(raw) - 2)))

    x, y = counts(a), counts(b)
    denominator = math.sqrt(sum(v * v for v in x.values()) * sum(v * v for v in y.values()))
    return sum(v * y[k] for k, v in x.items()) / denominator if denominator else 0.0


def check_reference(row, documents, demonstrations):
    public, result = row["observation"], row["prediction"]
    query = scalar.similarity_query(public)
    attribution = result["attribution"]
    require(
        attribution["similarity_query_sha256"] == digest(query)
        and attribution["similarity_query_bytes"] == len(query)
        and attribution["observation_sha256"] == scalar.observation_hash(public)
        and attribution["feedback_in_similarity"] is (public["feedback"] is not None),
        "reference full-input attribution",
    )
    if row["kind"] == "reading":
        scores = [similarity(query, bytes.fromhex(d["bytes_hex"])) for d in documents]
        index = max(range(len(scores)), key=lambda i: (scores[i], -i))
        raw = bytes.fromhex(documents[index]["bytes_hex"])
        match = re.search(rb"(?<![A-Za-z0-9])'([^'\r\n]*)'", raw)
        require(
            attribution["document_index"] == index
            and attribution["document_sha256"] == digest(raw)
            and attribution["literal_span"] == (list(match.span(1)) if match else None)
            and row["answer_hex"]
            == result["answer_hex"]
            == (match.group(1) if match else b"").hex(),
            "reference literal attribution",
        )
        scalar.close(attribution["score"], scores[index], "reference lexical score")
    else:
        scores = [similarity(query, bytes.fromhex(d["query"]["bytes_hex"])) for d in demonstrations]
        index = max(range(len(scores)), key=lambda i: (scores[i], -i))
        demo, locations = demonstrations[index], []
        anchor = bytes.fromhex(demo["old"]["bytes_hex"])
        if anchor:
            for file in row["before_files"]:
                raw, offset = bytes.fromhex(file["content_hex"]), 0
                while (found := raw.find(anchor, offset)) >= 0:
                    locations.append((file["path"], found))
                    offset = found + 1
        require(
            attribution["demonstration_index"] == index
            and attribution["demonstration_query_sha256"] == demo["query_sha256"]
            and attribution["demonstration_source_sha256"] == demo["source_sha256"]
            and attribution["raw_anchor_matches"] == len(locations),
            "reference taught anchor",
        )
        scalar.close(attribution["score"], scores[index], "reference edit lexical score")
        expected = {
            "operation": "STOP",
            "path": row["before_files"][0]["path"],
            "start": 0,
            "end": 0,
            "replacement_hex": "",
            "stop_reason": "eos",
        }
        if len(locations) == 1:
            path, start = locations[0]
            expected.update(
                operation="SPLICE",
                path=path,
                start=start,
                end=start + len(anchor),
                replacement_hex=demo["replacement"]["bytes_hex"],
            )
        require(
            all(row["proposal"][key] == value for key, value in expected.items())
            and row["proposal"] == result["proposal"],
            "reference actual proposed action",
        )


def metrics(compact, manifest, prior_outcomes):
    """Observed denominators only; four phases with explicit acquisition subgroups."""
    result = {}
    for route in ROUTES:
        rows = [row for row in compact if row["route"] == route]
        first = {r["episode_id"]: r for r in rows if r["attempt_index"] == 1}
        phases = {}
        for phase in ("acquisition", "transfer", "retention", "recovery"):
            phases[phase] = {}
            for kind in ("reading", "edit"):
                planned = [
                    e for e in manifest["episodes"] if e["phase"] == phase and e["kind"] == kind
                ]
                selected = [r for r in rows if r["phase"] == phase and r["kind"] == kind]
                observed = [r for r in selected if r["attempt_index"] == 1]
                phases[phase][kind] = {
                    "planned_episodes": len(planned),
                    "observed_episodes": len(observed),
                    "missing_episodes": len(planned) - len(observed),
                    "attempts": len(selected),
                    "supported_first": sum(r["status"] != "unsupported" for r in observed),
                    "first_correct": sum(r["correct"] for r in observed),
                }
        groups = {}
        for name in (
            "first",
            "changed_form",
            "exact_delayed",
            "instruction_construction",
            "structural_combination",
        ):
            planned = [
                e
                for e in manifest["episodes"]
                if e.get("encounter") == name or e.get("fresh_stratum") == name
            ]
            groups[name] = {
                kind: {
                    "planned": sum(e["kind"] == kind for e in planned),
                    "observed": sum(
                        e["kind"] == kind and e["episode_id"] in first for e in planned
                    ),
                    "correct": sum(
                        first[e["episode_id"]]["correct"]
                        for e in planned
                        if e["kind"] == kind and e["episode_id"] in first
                    ),
                }
                for kind in ("reading", "edit")
            }
        retention, repeated, changed = [], [], []
        for e in manifest["episodes"]:
            if e["episode_id"] not in first:
                continue
            after = first[e["episode_id"]]
            if e["phase"] == "retention":
                prior_route = "simple" if route == "simple" else "online"
                before = prior_outcomes[prior_route][e["previous_session_episode_id"]]
                retention.append(
                    {
                        "episode_id": e["episode_id"],
                        "prior_episode_id": before["episode_id"],
                        "prior_route": prior_route,
                        "initial_correct": before["correct"],
                        "delayed_correct": after["correct"],
                        "intervening_prior_attempts": 79 - before["route_ordinal"],
                        "intervening_current_attempts": rows.index(after),
                        "same_raw_observation_as_prior": before["observation_sha256"]
                        == after["observation_sha256"],
                    }
                )
            if e.get("encounter") in ("changed_form", "exact_delayed"):
                before = first[e["same_session_origin_id"]]
                gap = rows.index(after) - rows.index(before) - 1
                require(gap == e["intervening_attempts"], "actual encounter age differs")
                if e["encounter"] == "exact_delayed":
                    require(
                        before["observation_sha256"] == after["observation_sha256"],
                        "exact repeated observation",
                    )
                (repeated if e["encounter"] == "exact_delayed" else changed).append(
                    {
                        "episode_id": e["episode_id"],
                        "origin_episode_id": before["episode_id"],
                        "initial_correct": before["correct"],
                        "later_correct": after["correct"],
                        "intervening_attempts": gap,
                        "requirement_changed": bool(e.get("requirement_changed")),
                    }
                )
        recovery = [r for r in rows if r["phase"] == "recovery"]
        complete_recovery = [r for r in recovery if r["attempt_index"] == 3]
        recovery_first = [r for r in recovery if r["attempt_index"] == 1]
        recovery_later = [r for r in recovery if r["attempt_index"] > 1]
        recovered_ids = {
            r["episode_id"]
            for r in recovery_later
            if r["correct"] and not first[r["episode_id"]]["correct"]
        }
        recovery_pairs = [
            (before, after)
            for before, after in pairwise(recovery)
            if before["episode_id"] == after["episode_id"]
        ]
        result[route] = {
            "phases": phases,
            "groups": groups,
            "attempts": len(rows),
            "retention": retention,
            "same_session_exact": repeated,
            "pure_changed_forms": changed,
            "retained": sum(r["initial_correct"] and r["delayed_correct"] for r in retention),
            "forgotten": sum(r["initial_correct"] and not r["delayed_correct"] for r in retention),
            "improved": sum(not r["initial_correct"] and r["delayed_correct"] for r in retention),
            "persistent_failure": sum(
                not r["initial_correct"] and not r["delayed_correct"] for r in retention
            ),
            "action_behavior": dict(Counter(r["behavior"] for r in rows if r["kind"] == "edit")),
            "chains": [r for r in rows if r["project_id"] is not None],
            "recovery": {
                "observed_attempts": len(recovery),
                "observed_episodes": len(recovery_first),
                "first_correct": sum(r["correct"] for r in recovery_first),
                "observed_first_failures": sum(not r["correct"] for r in recovery_first),
                "completed_episodes": len(complete_recovery),
                "completed_first_failed_episodes": sum(
                    not first[r["episode_id"]]["correct"] for r in complete_recovery
                ),
                "incomplete_episodes": len(recovery_first) - len(complete_recovery),
                "later_correct_attempts": sum(r["correct"] for r in recovery_later),
                "ever_recovered_after_first_failure": len(recovered_ids),
                "terminal_correct": sum(r["correct"] for r in complete_recovery),
                "completed_terminal_failures": sum(not r["correct"] for r in complete_recovery),
                "final_recovered_after_first_failure": sum(
                    r["correct"] and not first[r["episode_id"]]["correct"]
                    for r in complete_recovery
                ),
                "feedback_consuming_attempts": sum(r["feedback_consumed"] for r in recovery),
                "observed_consecutive_pairs": len(recovery_pairs),
                "identical_consecutive_proposals": sum(
                    a["proposal_signature"] == b["proposal_signature"] for a, b in recovery_pairs
                ),
                "feedback_consuming_changed_proposals": sum(
                    b["feedback_consumed"] and a["proposal_signature"] != b["proposal_signature"]
                    for a, b in recovery_pairs
                ),
            },
        }
    return result


def check_initialization(row):
    expected = {
        "schema": "persistent-language-action-initial-custody-v7",
        "status": "initialized",
        "bank_sha256": history.PRIOR_BANK_SHA,
        "prior_sha256": history.PRIOR_CHECKPOINT_SHA,
        "prior_report_sha256": history.PRIOR_REPORT_SHA,
        "prior_sources_sha256": history.PRIOR_SOURCES_SHA,
        "prior_attempts_sha256": history.PRIOR_HISTORY_SHA["attempts.jsonl"],
        "prior_source_revision": history.PRIOR_REVISION,
        "prior_selected_route": "online",
        "loaded_payload_tensor_bytes": 100830448,
        "loaded_tensor_count": 40,
        "torch_load_calls": 1,
        "old_model_constructions": 0,
        "new_combined_model_constructions": 2,
        "promoted_tape_states": 2,
        "restored_occupied_slots_per_route": 864,
        "preserved_assembly_owners_per_route": 398,
        "initial_experience_per_route": 238,
        "initial_reader_experience_per_route": 152,
        "source_next_symbol_offers": 0,
        "dendritic_observation_offers": 0,
        "ingest_calls": 0,
        "checkpoint_free_replay_calls": 0,
        "teacher_replay_calls": 0,
        "initial_model_tensor_bytes": dict.fromkeys(ROUTES[:2], 87760560),
        "initial_tape_tensor_bytes": dict.fromkeys(ROUTES[:2], 3301384),
        "initial_model_tensor_counts": dict.fromkeys(ROUTES[:2], 22),
    }
    for key, value in expected.items():
        require(row[key] == value, "initial custody: " + key)
    for label in ("bank", "prior"):
        require(
            scalar.file_sha(row[label + "_path"]) == row[label + "_sha256"], "initial input bytes"
        )
    require(
        len(set(row["initial_model_sha256"].values())) == 1
        and len(set(row["initial_memory_sha256"].values())) == 1,
        "matched initial state",
    )
    for route in ROUTES[:2]:
        for key in ("initial_model_sha256", "initial_memory_sha256"):
            scalar.hash_value(row[key][route])
        require(
            row["derived_views"][route]["owners"] == 398
            and row["initialization_work"][route]["factorized_candidate_builds"] == 398
            and row["initialization_work"][route]["recognition_self_builds"] == 398,
            "derived wiring rebuild scope",
        )
        work_valid(row["initialization_work"][route])


def check_teacher(row, attempt, episode, current, demos, old_bank, counts):
    route, offered = attempt["route"], episode["teacher"]
    require(
        row["schema"] == "persistent-language-action-teacher-v7"
        and row["route"] == route
        and row["episode_id"] == episode["episode_id"]
        and row["teacher"] == offered
        and row["learning_enabled"] is (route != "learning_disabled"),
        "actual correction identity/gate",
    )
    require(
        row["model_before_sha256"] == current[route]["model"]
        and row["memory_before_sha256"] == row["memory_after_sha256"] == current[route]["memory"],
        "correction chronology",
    )
    if route == "simple":
        require(
            row["reader_offers"] == row["action_labels"] == 0 and not row["work"],
            "reference neural work",
        )
        if offered["kind"] == "edit":
            scalar.append_demonstration(
                demos, offered, attempt, scalar.similarity_query(attempt["observation"])
            )
            require(
                row["publication"]["stored"] is True
                and row["publication"]["demonstration_index"] == len(demos) - 1,
                "actual reference correction",
            )
        else:
            require(
                row["publication"]
                == {"offered": True, "stored": False, "reason": "reading_targets_not_stored"},
                "reference reading target stored",
            )
        require(
            row["model_after_sha256"] == scalar.plain_digest(demos), "reference teacher state hash"
        )
    else:
        enabled = route == "online"
        target = bytes.fromhex(offered.get("target_hex", offered.get("replacement_hex")))
        feedback, changed, loss, next_bank, revision = row["reader_feedback"], False, 0.0, None, 0
        require(
            len(feedback["steps"]) == len(target) + 1 and row["reader_offers"] == len(target) + 1,
            "whole answer offers",
        )
        prediction = (
            attempt["prediction"]
            if episode["kind"] == "reading"
            else attempt["prediction"]["payload"]
        )
        require(
            feedback["actual_response_id"] == prediction["attempt_id"], "teacher pending response"
        )
        next_bank = prediction["steps"][0]["binding"]["bank_sha256"]
        request_hash = prediction["steps"][0]["binding"]["request_sha256"]
        for position, step in enumerate(feedback["steps"]):
            token = target[position] if position < len(target) else 256
            require(
                step["output_step"] == position
                and step["prefix_hex"] == target[:position].hex()
                and step["observed_token"] == token,
                "teacher causal prefix/target",
            )
            scalar.memory_winner(
                step["address_scores"], step["selected_index"], attempt["memory_used"]
            )
            binding = step["binding"]
            check_binding(binding, target[:position], request_present=True)
            require(
                binding["stage_revision"] == revision and binding["request_sha256"] == request_hash,
                "teacher stage/request continuity",
            )
            if next_bank is not None:
                require(binding["bank_sha256"] == next_bank, "teacher private stage continuity")
            changed |= check_update(binding, step["update"], token, enabled)
            next_bank = (
                step["update"]["proposed_bank_sha256"]
                if step["update"]["staged"]
                else binding["bank_sha256"]
            )
            revision = step["update"]["stage_revision_after"]
            logits = scalar.vector(step["prior_logits"], 257)
            owner = step["routing"]["selected_index"]
            scalar.same_vector(logits, old_bank[owner], "teacher changed old outgoing prior")
            scalar.same_vector(binding["prior"], logits, "teacher tuple prior")
            nll = -scalar.log_softmax(logits)[token]
            scalar.close(step["prior_nll"], nll, "teacher prior diagnostic")
            loss += nll
        scalar.close(feedback["total_loss"], loss, "teacher prior loss sum")
        scalar.close(feedback["mean_loss"], loss / (len(target) + 1), "teacher prior mean")
        publication = row["publication"]
        require(
            publication["binding"] is changed
            and publication["output"] is changed
            and all(
                publication[k] is False
                for k in ("address", "legacy_codec", "routing", "recognition")
            ),
            "held prior or binding publication",
        )
        labels = 3 if episode["kind"] == "edit" else 0
        require(row["action_labels"] == labels, "actual action correction labels")
        if labels:
            require(
                set(publication["actions"]) == {"start", "end", "operation"}
                and all(
                    type(v) is bool and (enabled or not v) for v in publication["actions"].values()
                ),
                "action local publication control",
            )
        work_valid(row["work"])
        require(enabled or row["model_after_sha256"] == current[route]["model"], "disabled learned")
    scalar.hash_value(row["model_after_sha256"])
    current[route]["model"] = row["model_after_sha256"]
    counts[route]["teacher_occasions"] += 1
    counts[route]["reader_offers"] += row["reader_offers"]
    counts[route]["action_labels"] += row["action_labels"]


def check_checkpoints(run, sources, manifest_sha, rows_by_kind, partial, resolved):
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
        identity = scalar.file_sha(path)
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
            for name, identity in host["initial_bindings"].items():
                require(
                    scalar.file_sha(safe_path(run, name)) == identity, "checkpoint initial evidence"
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
                and value["reader_experience"] == 152 + expected["free"][route]
                and value["experience"] == 238 + expected["actions"][route],
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
                "sha256": identity,
            }
        )
        previous, previous_hash, last = boundary, identity, boundary
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
    }


def review(run, launch_path, manifest_sha, *, partial=False):
    started = time.perf_counter()
    saved_tool_sources = {name: scalar.file_sha(ROOT / name) for name in SAVED_ONLY_SOURCES}
    run = Path(run).resolve()
    require(manifest_sha == MANIFEST_SHA, "unselected frozen manifest")
    require(scalar.file_sha(run / "manifest.json") == manifest_sha, "frozen manifest bytes changed")
    manifest, monitor = read_json(run / "manifest.json"), read_json(launch_path)
    require(
        manifest["schema"] == "persistent-language-action-stream-v7"
        and len(manifest["episodes"]) == 48
        and sum(e["attempt_limit"] for e in manifest["episodes"]) == 64,
        "V7 fixed workload scope",
    )
    require(monitor["status"] == ("failed" if partial else "passed"), "run monitor status")
    command = monitor["command"]
    require(
        command[1:5]
        == ["-X", "utf8", "-m", "experiments.r4_integrated.persistent_language_action_v7"]
        and Path(command[command.index("--output") + 1]).resolve() == run
        and command[command.index("--manifest-sha") + 1] == manifest_sha,
        "monitored run command",
    )
    sources = read_json(run / "sources.json")
    expected_sources = {
        p.relative_to(ROOT).as_posix()
        for folder in (ROOT / "src/humanbrain", ROOT / "experiments/r4_integrated")
        for p in folder.glob("*.py")
    }
    require(set(sources) == expected_sources - SAVED_ONLY_SOURCES, "full runtime source closure")
    for name, identity in sources.items():
        require(
            scalar.file_sha(safe_path(ROOT, name)) == identity, "frozen source changed: " + name
        )
    completion = None
    if not partial:
        completion = read_json(run / "completion.json")
        require(
            completion["schema"] == "persistent-language-action-completion-v7"
            and completion["status"] == "completed",
            "complete session marker",
        )
        for name, identity in completion["files"].items():
            require(
                scalar.file_sha(safe_path(run, name)) == identity,
                "complete evidence changed: " + name,
            )
    else:
        require(
            not (run / "completion.json").exists(),
            "partial review cannot replace a completed audit",
        )
    indexed, journals = {}, {}
    for kind in ("attempts", "teachers", "ingests"):
        indexed[kind], journals[kind] = journal(run / (kind + ".jsonl"), partial=partial)
    telemetry, telemetry_frontier = journal(run / "checkpoints.jsonl", partial=partial)
    if not all(
        (run / name).exists()
        for name in ("initialization.json", "prior-outcomes.json", "baseline-restoration.json")
    ):
        require(
            partial and not any(indexed.values()),
            "attempted work without completed initialization evidence",
        )
        checkpoint = check_checkpoints(run, sources, manifest_sha, indexed, True, [])
        require(checkpoint["head"] is None, "safe state without completed initial records")
        return {
            "schema": "persistent-language-action-saved-review-v7",
            "status": "passed_partial",
            "complete_run": False,
            "run": str(run),
            "manifest_sha256": manifest_sha,
            "source_revision": command[command.index("--source-revision") + 1],
            "run_launch_sha256": scalar.file_sha(launch_path),
            "saved_tool_sources": saved_tool_sources,
            "attempt_rows": 0,
            "teacher_rows": 0,
            "ingest_rows": 0,
            "metrics": metrics([], manifest, {r: {} for r in ROUTES}),
            "checkpoint": checkpoint,
            "pending_transaction": {"kind": "initialization"},
            "journal_frontiers": journals,
            "exposures": {r: dict.fromkeys(COUNTERS, 0) for r in ROUTES},
            "source_offers": dict.fromkeys(ROUTES, 0),
            "evidence": {
                p.relative_to(run).as_posix(): scalar.file_sha(p)
                for p in sorted(run.rglob("*"))
                if p.is_file()
            },
            "elapsed_seconds": time.perf_counter() - started,
            "model_constructions": 0,
            "checkpoint_decodes": 0,
            "predictor_calls": 0,
            "actual_executions": 0,
            "teacher_replays": 0,
            "scope": "incomplete initialization, no observed task outcomes or decoded checkpoint",
        }
    initialization = read_json(run / "initialization.json")
    check_initialization(initialization)
    outcome = read_json(run / "prior-outcomes.json")
    require(
        outcome["schema"] == "persistent-language-action-prior-outcomes-v7", "prior scalar schema"
    )
    # Explicit schema-only adapter to the unchanged V5 historical-data validator.
    demos, documents, old_manifest, prior_outcomes, old_bank, _ = history.check_history(
        ROOT,
        read_json(run / "baseline-restoration.json"),
        {**outcome, "schema": "persistent-language-action-prior-outcomes-v6"},
    )
    require(
        manifest["continuation"]["prior_documents"]
        == old_manifest["continuation"]["prior_documents"] + old_manifest["documents"],
        "whole retained factual context changed",
    )
    old_by_id = {e["episode_id"]: e for e in old_manifest["episodes"]}
    for e in manifest["episodes"]:
        if e["phase"] == "retention":
            require(
                all(
                    e[k] == old_by_id[e["previous_session_episode_id"]][k]
                    for k in ("kind", "request_hex", "files", "grade")
                ),
                "historical regression changed",
            )
    launch = read_json(run / "launch-record.json")
    for path, identity in launch["artifacts"].items():
        require(scalar.file_sha(path) == identity, "qualified executor input changed")
    current = {
        r: {
            "model": initialization["initial_model_sha256"][r],
            "memory": initialization["initial_memory_sha256"][r],
        }
        for r in ROUTES[:2]
    }
    current["simple"] = {
        "model": scalar.plain_digest(demos),
        "memory": scalar.plain_digest(documents),
    }
    counts = {r: dict.fromkeys(COUNTERS, 0) for r in ROUTES}
    used, free, actions = (
        dict.fromkeys(ROUTES, 864),
        dict.fromkeys(ROUTES[:2], 0),
        dict.fromkeys(ROUTES[:2], 0),
    )
    projects, instances, responses = {r: {} for r in ROUTES}, {}, set()
    offers = {"reading": 0, "STOP": 0, "SPLICE": 0}
    source_offers = dict.fromkeys(ROUTES, 0)
    saved_costs = {
        r: {
            "work": {kind: {} for kind in ("source", "free", "teacher", "ack")},
            "elapsed_seconds": dict.fromkeys(("source", "attempt", "teacher"), 0.0),
            "included_checkpoint_begin_seconds": 0.0,
        }
        for r in ROUTES
    }

    def add_work(destination, values, prefix=""):
        work_valid(values)
        for key, value in values.items():
            name = prefix + key
            if isinstance(value, dict):
                add_work(destination, value, name + ".")
            else:
                destination[name] = destination.get(name, 0) + value

    def add_cost(row, category, elapsed_category=None):
        costs = saved_costs[row["route"]]
        add_work(costs["work"][category], row["work"])
        if elapsed_category is not None:
            elapsed = elapsed_value(row["elapsed_seconds"])
            costs["elapsed_seconds"][elapsed_category] += elapsed
        if category == "free":
            add_work(costs["work"]["ack"], row.get("ack_work", {}))
            included = elapsed_value(row["included_checkpoint_begin_seconds"])
            costs["included_checkpoint_begin_seconds"] += included

    cursor = {"episode_index": 0, "route_index": 0, "attempt_index": 1, "append_done": False}
    resolved, compact, actual_current = [], [], None
    consumed = dict.fromkeys(indexed, 0)
    transaction_order = []

    def record_boundary(kind, before):
        baseline = {
            "documents_hex": [d["bytes_hex"] for d in documents],
            "demonstrations": [
                {k: v["bytes_hex"] if isinstance(v, dict) else v for k, v in d.items()}
                for d in demos
            ],
            "new_teacher_offers": offers,
        }
        resolved.append(
            json.loads(
                encoded(
                    {
                        "kind": kind,
                        "before_cursor": before,
                        "cursor": cursor,
                        "counters": counts,
                        "projects": projects,
                        "current": actual_current,
                        "hashes": current,
                        "free": free,
                        "actions": actions,
                        "baseline": baseline,
                    }
                )
            )
        )

    record_boundary("initialize", dict(cursor))
    docs = {d["document_id"]: bytes.fromhex(d["text_hex"]) for d in manifest["documents"]}
    pending = None
    for ei, episode in enumerate(manifest["episodes"]):
        if pending:
            break
        for ri, route in enumerate(ROUTES):
            if pending:
                break
            if episode["append_document_ids"]:
                if consumed["ingests"] == len(indexed["ingests"]):
                    pending = {"kind": "append", "cursor": dict(cursor)}
                    break
                entry = indexed["ingests"][consumed["ingests"]]
                row = read_row(entry)
                require(
                    row["schema"] == "persistent-language-action-ingest-v7"
                    and (row["route"], row["episode_id"], row["document_ids"])
                    == (route, episode["episode_id"], episode["append_document_ids"]),
                    "exact append schedule",
                )
                raw = [docs[key] for key in episode["append_document_ids"]]
                require(
                    row["raw_bytes"] == sum(map(len, raw))
                    and row["eos"] == len(raw)
                    and row["used_before"] == used[route]
                    and row["used_after"] == used[route] + sum(len(d) + 1 for d in raw)
                    and row["model_before_sha256"] == current[route]["model"]
                    and row["memory_before_sha256"] == current[route]["memory"],
                    "source append chronology",
                )
                require(
                    row["documents"]
                    == [
                        {
                            "document_id": key,
                            "text_hex": docs[key].hex(),
                            "sha256": digest(docs[key]),
                        }
                        for key in episode["append_document_ids"]
                    ]
                    and row["association_writes_enabled"] is True,
                    "whole unselected source admission",
                )
                if route == "simple":
                    require(
                        row["source_feedback"] is None
                        and row["source_next_byte_teacher_symbols"] == 0,
                        "reference neural source teaching",
                    )
                    documents.extend({"bytes_hex": d.hex()} for d in raw)
                    require(
                        row["memory_after_sha256"] == scalar.plain_digest(documents),
                        "reference appended documents",
                    )
                else:
                    n = check_source_steps(row["source_feedback"], raw, route == "online")
                    require(
                        row["source_next_byte_teacher_symbols"] == n
                        and row["source_learning_enabled"] is (route == "online")
                        and row["old_prefix_before_sha256"] == row["old_prefix_after_sha256"],
                        "source learning gates/prefix",
                    )
                    source_offers[route] += n
                require(
                    route == "online" or row["model_after_sha256"] == current[route]["model"],
                    "disabled/source prior changed",
                )
                work_valid(row["work"])
                add_cost(row, "source", "source")
                counts[route]["ingest_documents"] += len(raw)
                counts[route]["ingest_raw_bytes"] += row["raw_bytes"]
                counts[route]["ingest_eos"] += len(raw)
                used[route] = row["used_after"]
                current[route] = {
                    "model": row["model_after_sha256"],
                    "memory": row["memory_after_sha256"],
                }
                before_cursor = dict(cursor)
                cursor["append_done"] = True
                consumed["ingests"] += 1
                record_boundary("append", before_cursor)
                transaction_order.append(("ingest", route, episode["episode_id"]))
                del row
            project = episode["project_id"]
            feedback = None
            if episode["workspace_lifecycle"] == "carry_actual_project":
                previous = projects[route][project]
                require(
                    previous["episode_id"] == episode["predecessor_episode_id"],
                    "actual project predecessor",
                )
                files, feedback = previous["files"], previous["feedback"]
            else:
                files = episode["files"]
            for attempt_index in range(1, episode["attempt_limit"] + 1):
                if consumed["attempts"] == len(indexed["attempts"]):
                    pending = {"kind": "attempt", "cursor": dict(cursor)}
                    break
                entry = indexed["attempts"][consumed["attempts"]]
                row = read_row(entry)
                require(
                    row["schema"] == "persistent-language-action-attempt-v7"
                    and (row["route"], row["episode_id"], row["attempt_index"])
                    == (route, episode["episode_id"], attempt_index),
                    "exact actual attempt prefix",
                )
                require(
                    all(
                        row[k] == episode[k]
                        for k in (
                            "phase",
                            "kind",
                            "request_hex",
                            "project_id",
                            "predecessor_episode_id",
                            "workspace_lifecycle",
                        )
                    )
                    and row["before_files"] == files,
                    "private routing or nominal project repair",
                )
                public = {
                    "schema": "sequence-assembly-raw-observation-v1",
                    "request_hex": episode["request_hex"],
                    "files": files if episode["kind"] == "edit" else None,
                    "feedback": feedback,
                }
                require(row["status"] in ("completed", "unsupported"), "attempt status")
                if row["status"] == "completed":
                    require(row["observation"] == public, "target-free complete raw observation")
                else:
                    require(
                        row["prediction"] is None
                        and row["proposal"] is None
                        and row["answer_hex"] is None,
                        "unsupported input still predicted",
                    )
                observed_hash = digest(encoded(row["observation"]))
                require(
                    row["observation_record_sha256"] == observed_hash
                    and row["model_before_sha256"]
                    == row["model_after_sha256"]
                    == current[route]["model"]
                    and row["memory_before_sha256"]
                    == row["memory_after_sha256"]
                    == current[route]["memory"]
                    and row["memory_used"] == used[route],
                    "free response/state custody",
                )
                if route in instances:
                    require(row["model_instance_id"] == instances[route], "actor replaced")
                else:
                    require(
                        row["model_instance_id"] not in instances.values(), "routes alias actor"
                    )
                    instances[route] = row["model_instance_id"]
                if row["prediction"] is not None:
                    if route == "simple":
                        check_reference(row, documents, demos)
                    else:
                        prediction = (
                            row["prediction"]
                            if episode["kind"] == "reading"
                            else row["prediction"]["payload"]
                        )
                        check_prediction(prediction, public, used[route], old_bank, responses)
                        require(
                            prediction["memory_sha256"] == current[route]["memory"],
                            "prediction tape",
                        )
                        free[route] += 1
                        if episode["kind"] == "reading":
                            require(
                                row["answer_hex"] == prediction["answer_hex"],
                                "reading substitution",
                            )
                        else:
                            scalar.check_heads(row["prediction"], row["proposal"], files)
                            require(
                                row["prediction"]["model_sha256"] == current[route]["model"],
                                "action model seal",
                            )
                            actions[route] += 1
                require(
                    row["feedback_consumed"]
                    is (feedback is not None and row["prediction"] is not None),
                    "actual feedback consumed",
                )
                signature = (
                    None
                    if row["proposal"] is None
                    else digest(
                        encoded(
                            {k: v for k, v in row["proposal"].items() if k != "workspace_sha256"}
                        )
                    )
                )
                require(row["proposal_signature"] == signature, "actual proposal signature")
                behavior = None
                if episode["kind"] == "edit":
                    feedback = scalar.check_execution(row, episode, run, launch)
                    correct = row["status"] != "unsupported" and row["grade"]["task_success"]
                    behavior = (
                        "unsupported"
                        if row["status"] == "unsupported"
                        else "rejected"
                        if not row["execution"]["accepted"]
                        else "STOP"
                        if row["proposal"]["operation"] == "STOP"
                        else "accepted_correct"
                        if correct
                        else "accepted_incorrect"
                    )
                    counts[route]["executions"] += 1
                else:
                    require(
                        row["execution"] is None and row["after_files"] == files,
                        "reading hidden action",
                    )
                    correct = (
                        row["status"] != "unsupported"
                        and row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                    )
                require(row["correct"] is bool(correct), "private actual grade")
                counts[route]["attempts"] += 1
                compact.append(
                    {
                        k: row[k]
                        for k in (
                            "route",
                            "episode_id",
                            "phase",
                            "kind",
                            "attempt_index",
                            "correct",
                            "status",
                            "project_id",
                            "predecessor_episode_id",
                            "workspace_lifecycle",
                            "memory_used",
                            "feedback_consumed",
                        )
                    }
                    | {
                        "observation_sha256": observed_hash,
                        "proposal_signature": signature,
                        "behavior": behavior,
                    }
                )
                consumed["attempts"] += 1
                add_cost(row, "free", "attempt")
                if episode["teacher"] is not None:
                    if consumed["teachers"] == len(indexed["teachers"]):
                        pending = {
                            "kind": "correction_after_durable_attempt",
                            "cursor": dict(cursor),
                            "attempt_sha256": entry["sha256"],
                        }
                        break
                    teacher = read_row(indexed["teachers"][consumed["teachers"]])
                    require(
                        teacher["attempt_sha256"] == entry["sha256"],
                        "actual response before correction",
                    )
                    check_teacher(teacher, row, episode, current, demos, old_bank, counts)
                    add_cost(teacher, "teacher", "teacher")
                    consumed["teachers"] += 1
                    if route == "simple":
                        offers["reading" if episode["kind"] == "reading" else "SPLICE"] += 1
                    del teacher
                elif route != "simple" and row["prediction"] is not None:
                    ack = row["ack_work"]
                    work_valid(ack)
                    require(
                        ack.get("reader", ack).get("teacher_offers", 0) == 0,
                        "teacher-free acknowledgment offered targets",
                    )
                files = row["after_files"]
                actual_current = {
                    "route": route,
                    "episode_id": episode["episode_id"],
                    "files": files,
                    "feedback": feedback,
                }
                if project:
                    projects[route][project] = {
                        k: v for k, v in actual_current.items() if k != "route"
                    }
                before_cursor = dict(cursor)
                cursor = (
                    {
                        "episode_index": ei,
                        "route_index": ri,
                        "attempt_index": attempt_index + 1,
                        "append_done": True,
                    }
                    if attempt_index < episode["attempt_limit"]
                    else {
                        "episode_index": ei + int(ri == 2),
                        "route_index": (ri + 1) % 3,
                        "attempt_index": 1,
                        "append_done": False,
                    }
                )
                record_boundary("attempt", before_cursor)
                transaction_order.append(("attempt", route, episode["episode_id"], attempt_index))
                del row
    require(
        all(consumed[k] == len(indexed[k]) for k in indexed),
        "orphan/unplanned durable rows beyond observed prefix",
    )
    require(partial or pending is None, "complete session misses planned outcomes")
    checkpoint = check_checkpoints(run, sources, manifest_sha, indexed, partial, resolved)
    if not partial:
        require(
            len(compact) == 192
            and consumed == {"attempts": 192, "teachers": 12, "ingests": 36}
            and checkpoint["boundaries"] == 229
            and used == dict.fromkeys(ROUTES, 1614)
            and source_offers == {"online": 750, "learning_disabled": 750, "simple": 0},
            "complete exposure scope",
        )
        report = read_json(run / "report.json")
        require(
            report["schema"] == "persistent-language-action-report-v7"
            and report["status"] == "completed"
            and report["counts"] == consumed
            and report["actual_executions"] == 132,
            "report scope",
        )
        require(
            report["retained_reader_initial_sha256"] == report["retained_reader_final_sha256"]
            and report["final_assembly_occupied"] == dict.fromkeys(ROUTES[:2], 398),
            "inherited state publication",
        )
        require(
            report["final_hashes"] == current
            and report["final_reader_experience"] == {r: 152 + free[r] for r in free}
            and report["final_model_experience"] == {r: 238 + actions[r] for r in actions},
            "final state/counters",
        )
        for route in ROUTES:
            for phase in ("acquisition", "transfer", "retention", "recovery"):
                for kind in ("reading", "edit"):
                    selected = [
                        r
                        for r in compact
                        if r["route"] == route and r["phase"] == phase and r["kind"] == kind
                    ]
                    saved = report["metrics"][route]["phases"][phase][kind]
                    require(
                        saved["episodes"] == sum(r["attempt_index"] == 1 for r in selected)
                        and saved["attempts"] == len(selected)
                        and saved["first_correct"]
                        == sum(r["correct"] for r in selected if r["attempt_index"] == 1),
                        "report metrics",
                    )
        require(
            monitor["elapsed_seconds"] <= report["budget"]["runner_seconds"]
            and monitor["observed_tree_peak_sum_mib"] <= 2048,
            "complete monitored budget",
        )
    telemetry_counts, telemetry_seconds, previous_work = Counter(), Counter(), {}
    for entry in telemetry:
        item = read_row(entry)
        require(
            item["schema"] == "persistent-language-action-checkpoint-work-v7",
            "checkpoint telemetry schema",
        )
        telemetry_counts[item["phase"]] += 1
        elapsed = elapsed_value(item["elapsed_seconds"])
        telemetry_seconds[item["phase"]] += elapsed
        work_valid(item["cumulative_work"])
        require(
            all(item["cumulative_work"].get(k, 0) >= v for k, v in previous_work.items()),
            "checkpoint work reset",
        )
        if "work_delta" in item:
            require(
                item["work_delta"]
                == {k: v - previous_work.get(k, 0) for k, v in item["cumulative_work"].items()},
                "checkpoint full-phase work delta",
            )
        previous_work = item["cumulative_work"]
    if not partial:
        require(
            telemetry_counts == {"create": 1, "begin": 228, "capture": 229, "commit": 229},
            "complete checkpoint timings",
        )
    evidence = {
        p.relative_to(run).as_posix(): scalar.file_sha(p)
        for p in sorted(run.rglob("*"))
        if p.is_file()
    }
    return {
        "schema": "persistent-language-action-saved-review-v7",
        "status": "passed_partial" if partial else "passed",
        "complete_run": not partial,
        "manifest_sha256": manifest_sha,
        "run": str(run),
        "source_revision": command[command.index("--source-revision") + 1],
        "run_launch_sha256": scalar.file_sha(launch_path),
        "saved_tool_sources": saved_tool_sources,
        "attempt_rows": len(compact),
        "teacher_rows": consumed["teachers"],
        "ingest_rows": consumed["ingests"],
        "exposures": counts,
        "source_offers": source_offers,
        "saved_costs": saved_costs,
        "saved_cost_scope": "Observed journal work grouped by source/free/teacher/ack. "
        "Attempt elapsed includes the separately reported checkpoint begin interval and "
        "actual execution; categories are not an additive complete runtime decomposition.",
        "metrics": metrics(compact, manifest, prior_outcomes),
        "compact_attempts": compact,
        "pending_transaction": pending,
        "journal_frontiers": journals,
        "checkpoint": checkpoint,
        "checkpoint_telemetry_counts": dict(telemetry_counts),
        "checkpoint_telemetry_seconds": dict(telemetry_seconds),
        "checkpoint_telemetry_frontier": telemetry_frontier,
        "evidence": evidence,
        "elapsed_seconds": time.perf_counter() - started,
        "model_constructions": 0,
        "checkpoint_decodes": 0,
        "predictor_calls": 0,
        "actual_executions": 0,
        "teacher_replays": 0,
        "scope": "saved tuple arbitration, selected-contact arithmetic, causal source/answer "
        "offers, raw actions, actual reference attribution and durable prefixes; omitted contact "
        "activations/negative vectors and old neural forward are not recomputed; "
        "tensor payloads authenticated only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if DRAFT or MANIFEST_SHA is None:
        raise SystemExit("DRAFT: saved review requires root source release")
    try:
        result = review(args.run, args.launch, args.manifest_sha, partial=args.partial)
    except BaseException as error:
        with args.output.open("xb") as handle:
            handle.write(
                encoded(
                    {
                        "schema": "persistent-language-action-saved-review-v7",
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            )
        raise
    with args.output.open("xb") as handle:
        handle.write(encoded(result))
    print(json.dumps({"status": result["status"], "attempt_rows": result["attempt_rows"]}))


if __name__ == "__main__":
    main()
