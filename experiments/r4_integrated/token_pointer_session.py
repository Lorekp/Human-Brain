"""Fixed nine-policy exposed integration regression for token-pointer readers.

The outer runner owns source freeze, admission, cache/learner construction and
artifact publication. This session owns the cumulative-source action schedule;
its failed report preserves all completed work and must stop the outer run.
"""

from __future__ import annotations

import json
import time
import traceback
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.r4_integrated.token_pointer_actions import (
    ActionTask,
    PointerCustody,
    _occurrence,
    execute_prediction,
    execute_teacher,
    source_pool_sha256,
)
from experiments.r4_integrated.token_pointer_runtime import PointerDocument

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "experiments/r4_integrated/fresh_source_cases_v1.json"
FIXTURE_SHA256 = "6b78781c78c1d7b6d2474cbd8185123f9349e682e2ecde900ed2a852fbf17faf"
MOTOR_SHA256 = "575aa2c6ebadce0fccf21b556ad1a5bf702985cbb77eb8e41b7c84dfb49301e1"
UPDATING = ("cell", "frozen_producer", "bypass", "lstm")
ARMS = (*(name for arm in UPDATING for name in (arm, arm + "_frozen")), "external")


class SensoryCosine:
    """Fixed float64 cosine over every complete-paragraph word interval."""

    checkpoint_sha256 = sha256(
        b"token_pointer_sensory_v1;cast_float64;mean_query;mean_span;L2_floor1e-12;unique_max"
    ).hexdigest()

    def predict(self, query, documents):
        q = (
            np.asarray(query.vectors, dtype=np.float64).mean(axis=0)
            if len(query.vectors)
            else np.zeros(384)
        )
        q /= max(float(np.linalg.norm(q)), 1e-12)
        best, selected, ties = -np.inf, None, 0
        for d, document in enumerate(documents):
            vectors = np.asarray(document.tokens.vectors, dtype=np.float64)
            prefix = np.vstack((np.zeros((1, 384)), np.cumsum(vectors, axis=0)))
            for start in range(len(vectors)):
                lengths = np.arange(1, len(vectors) - start + 1)
                means = (prefix[start + 1 :] - prefix[start]) / lengths[:, None]
                means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
                scores = means @ q
                if not np.isfinite(scores).all():
                    raise FloatingPointError("nonfinite sensory cosine")
                maximum = float(scores.max())
                winners = np.flatnonzero(scores == maximum)
                if maximum > best:
                    best, selected, ties = (
                        maximum,
                        (d, start, start + int(winners[0])),
                        len(winners),
                    )
                elif maximum == best:
                    ties += len(winners)
        return selected if ties == 1 else None


def _gold(lesson, documents):
    """Original annotations only; return evaluator bytes and teacher pointer targets."""
    matches = [
        (i, d) for i, d in enumerate(documents) if d.document_id == lesson["paragraph_id"].encode()
    ]
    if len(matches) != 1:
        raise ValueError("original annotated source must occur exactly once")
    if lesson["is_impossible"]:
        if lesson["answers"] or lesson["accepted_text_hex"]:
            raise ValueError("official STOP carries accepted answers")
        return None, None
    d, document = matches[0]
    spans = document.tokens.byte_spans
    starts = {a: i for i, (a, _) in enumerate(spans)}
    ends = {b: i for i, (_, b) in enumerate(spans)}
    accepted, pointers = [], []
    for answer in lesson["answers"]:
        a, b = answer["byte_start"], answer["byte_end"]
        if (
            a not in starts
            or b not in ends
            or starts[a] > ends[b]
            or document.tokens.raw[a:b] != answer["text"].encode("utf-8")
        ):
            raise ValueError("official answer does not preserve original source word endpoints")
        accepted.append((document.document_id, a, b))
        pointers.append((d, starts[a], ends[b]))
    if not accepted or lesson["accepted_text_hex"] != list(
        dict.fromkeys(a["text"].encode().hex() for a in lesson["answers"])
    ):
        raise ValueError("all official accepted answers must remain intact")
    return tuple(dict.fromkeys(accepted)), tuple(dict.fromkeys(pointers))


def _counts(rows, verdict="success"):
    return {
        "supported": sum(r[verdict] is True and r["supported"] for r in rows),
        "STOP": sum(r[verdict] is True and not r["supported"] for r in rows),
        "supported_total": sum(r["supported"] and r["semantic_scored"] for r in rows),
        "STOP_total": sum(not r["supported"] and r["semantic_scored"] for r in rows),
        "unscored": sum(not r["semantic_scored"] for r in rows),
    }


def decision(report):
    """Validate exact evidence rosters before applying the integration criterion."""
    failure = {"complete": False, "qualified": False}
    if report["status"] != "completed":
        return failure
    blocks = report["blocks"]
    diagnostic = {blocks["E"][3], blocks["L"][3]}
    roster = {
        "initial": blocks["O"],
        **{f"episode_{i}": [v] for i, v in enumerate(blocks["E"])},
        "late": blocks["L"],
        "final": blocks["O"] + blocks["E"],
    }
    for lane, verdict in (("actions", "success"), ("oracle", "correct")):
        if list(report[lane]) != list(roster):
            return failure
        for phase, ids in roster.items():
            if list(report[lane][phase]) != list(ARMS):
                return failure
            for rows in report[lane][phase].values():
                if [r["id"] for r in rows] != ids:
                    return failure
                for row in rows:
                    scored = row["id"] not in diagnostic
                    if row["semantic_scored"] != scored or (row[verdict] is None) == scored:
                        return failure
    expected = [
        (i, arm, kind, item)
        for i, fresh in enumerate(blocks["E"])
        for arm in UPDATING
        for kind, item in (
            ("feedback", fresh),
            ("replay", blocks["O"][2 * i]),
            ("replay", blocks["O"][2 * i + 1]),
        )
        if kind != "feedback" or fresh not in diagnostic
    ]
    if [(r["episode"], r["arm"], r["kind"], r["id"]) for r in report["training"]] != expected:
        return failure
    required = [
        (i, arm, item)
        for i, item in enumerate(blocks["E"])
        for arm in UPDATING
        if report["actions"][f"episode_{i}"][arm][0]["success"] is False
    ]
    if [(r["episode"], r["arm"], r["id"]) for r in report["teachers"]] != required or any(
        not r["passed"]
        or r["teacher_executions"] != 1
        or r["counts_as_learner_success"] is not False
        for r in report["teachers"]
    ):
        return failure
    if report["counters"] != {
        "primary_predictions": 468,
        "oracle_predictions": 468,
        "primary_actions": 468,
        "teacher_executions": len(required),
        "motor_demonstrations": 2,
        "child_executions": 470 + len(required),
        "offers": 92,
        "offers_per_updating_arm": {a: 23 for a in UPDATING},
    }:
        return failure
    fresh = {
        a: [report["actions"][f"episode_{i}"][a][0] for i in range(8)]
        + report["actions"]["late"][a]
        for a in ARMS
    }
    summaries = {
        a: {
            "fresh": _counts(fresh[a]),
            "initial_old": _counts(report["actions"]["initial"][a]),
            "final_old": _counts(report["actions"]["final"][a][:16]),
            "late": _counts(report["actions"]["late"][a]),
        }
        for a in ARMS
    }
    losses = {}
    for lane, verdict in (("actions", "success"), ("oracle", "correct")):
        final = {r["id"]: r for r in report[lane]["final"]["cell"]}
        losses[lane] = [
            r["id"]
            for r in report[lane]["initial"]["cell"]
            if r[verdict] is True and final[r["id"]][verdict] is not True
        ]
    cell, frozen = summaries["cell"]["fresh"], summaries["cell_frozen"]["fresh"]
    checks = {
        "supported_first_attempt_gain": cell["supported"] - frozen["supported"] >= 2,
        "no_pooled_old_loss": not losses["actions"],
        "no_oracle_old_loss": not losses["oracle"],
        "no_worse_fresh_STOP": cell["STOP"] >= frozen["STOP"],
        "no_collateral_edits": all(
            r["execution"]["decoys_unchanged"]
            for phase in report["actions"].values()
            for rows in phase.values()
            for r in rows
        ),
    }
    return {
        "complete": True,
        "qualified": all(checks.values()),
        "checks": checks,
        "summaries": summaries,
        "lost_initial_old_ids": losses,
        "headroom_inconclusive": cell["supported_total"] - frozen["supported"] < 2,
        "scope": "Exposed retained-source regression; no independent or source-absent claim",
    }


def run_session(policies, get_word, work: Path, guard, event) -> dict:
    """Run once on four acquired policies; return failed partial evidence on error."""
    began = time.perf_counter()
    report = {
        "status": "failed",
        "actions": {},
        "oracle": {},
        "training": [],
        "teachers": [],
        "inventories": [],
        "motor_demonstrations": [],
        "events": [],
        "record_hash_scope": "Envelopes add id/supported to actions; episode/arm/id to teachers",
        "counters": {
            "primary_predictions": 0,
            "oracle_predictions": 0,
            "primary_actions": 0,
            "teacher_executions": 0,
            "motor_demonstrations": 0,
            "child_executions": 0,
            "offers": 0,
            "offers_per_updating_arm": {a: 0 for a in UPDATING},
        },
    }
    cores = {}

    def note(label, **values):
        report["events"].append(
            {"event": label, "elapsed_seconds": time.perf_counter() - began, **values}
        )
        event(label, **values)

    def snapshots():
        return {
            a: {
                "checkpoint_sha256": p.checkpoint_sha256,
                "steps": p.steps,
                "presentations": p.presentations,
            }
            for a, p in cores.items()
            if a != "external"
        }

    try:
        from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor
        from experiments.r4_integrated.utf8_action_bridge import Case, execute_utf8
        from humanbrain.native import loaded_build_info

        guard("integration_start")
        if tuple(policies) != UPDATING or any(not p.updating for p in policies.values()):
            raise ValueError("exact four enabled acquired policies required in declared order")
        raw = FIXTURE.read_bytes()
        if sha256(raw).hexdigest() != FIXTURE_SHA256:
            raise ValueError("original exposed integration fixture changed")
        fixture = json.loads(raw)
        if fixture["status"] != "aborted_semantic_review" or fixture["admission"]["allowed"]:
            raise ValueError("original aborted admission must remain preserved")
        blocks = report["blocks"] = fixture["blocks"]
        ids = blocks["O"] + blocks["E"] + blocks["L"]
        lessons = {r["id"]: r for r in fixture["cases"]}
        if (
            list(lessons) != ids
            or len(set(ids)) != 28
            or [len(blocks[k]) for k in ("O", "E", "L")] != [16, 8, 4]
        ):
            raise ValueError("exact original 28-task roster required")
        diagnostic = {blocks["E"][3], blocks["L"][3]}
        if any(not lessons[i]["is_impossible"] for i in diagnostic):
            raise ValueError("diagnostic slots must retain their official STOP labels")
        report["fixture"] = {
            "path": str(FIXTURE),
            "sha256": FIXTURE_SHA256,
            "original_admission": fixture["admission"],
            "diagnostic_only_ids": sorted(diagnostic),
        }
        work = Path(work)
        work.mkdir(parents=True, exist_ok=False)
        report["workspace_root"] = str(work.resolve())
        native = loaded_build_info()
        if not native.library_path.resolve().is_relative_to(ROOT / ".native-cache"):
            raise ValueError("native motor is outside checkout cache")
        report["native_motor"] = {
            "path": str(native.library_path),
            "source_digest": native.source_digest,
            "sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        for arm in UPDATING:
            cores[arm] = policies[arm]
            parent = cores[arm].checkpoint_sha256
            cores[arm + "_frozen"] = cores[arm].clone(frozen=True)
            if cores[arm].checkpoint_sha256 != parent:
                raise RuntimeError("cloning changed the acquired parent")
        cores["external"] = SensoryCosine()
        report["initial_states"] = snapshots()
        frozen = {
            a: p.checkpoint_sha256
            for a, p in cores.items()
            if a.endswith("_frozen") or a == "external"
        }
        inherited = {a: (p.steps, p.presentations) for a, p in policies.items()}
        motor = ContextualOccurrenceMotor()
        for case in (
            Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
            Case("motor-b", "bora", "ready", "motor_beta", "warm"),
        ):
            guard("motor_demonstration")
            before, after = case.files(), case.files(corrected=True)
            execution = execute_utf8(case, after, before, work / case.name)
            report["motor_demonstrations"].append({"case": case.name, "execution": execution})
            report["counters"]["motor_demonstrations"] += 1
            report["counters"]["child_executions"] += 1
            if not execution["passed"]:
                raise RuntimeError("original motor demonstration failed")
            motor.learn(f"Configure {case.key}".encode(), before, after)
        if motor.checkpoint_sha256 != MOTOR_SHA256 or motor.demonstration_count != 2:
            raise RuntimeError("original learned two-demo motor differs")
        report["motor"] = {
            "checkpoint_sha256": motor.checkpoint_sha256,
            "demonstration_count": motor.demonstration_count,
            "retained_demonstration_event_bytes": motor.retained_demonstration_event_bytes,
        }
        documents, queries = [], {}

        def add(item):
            lesson = lessons[item]
            pid = lesson["paragraph_id"]
            record = fixture["documents"][pid]
            if pid != fixture["document_order"][len(documents)]:
                raise ValueError("cumulative source order differs")
            source = record["context"].encode("utf-8")
            question = lesson["question"].encode("utf-8")
            tokens = get_word(sha256(source).hexdigest())
            query = get_word(sha256(question).hexdigest())
            if (
                tokens.raw != source
                or query.raw != question
                or tokens.byte_spans != tuple(map(tuple, record["word_byte_spans"]))
                or query.byte_spans != tuple(map(tuple, lesson["question_word_byte_spans"]))
            ):
                raise ValueError("cached source/question does not preserve original inputs")
            documents.append(
                PointerDocument(pid.encode(), record["article_sha256"].encode(), tokens)
            )
            queries[item] = query
            _gold(lesson, documents)

        def inventory():
            report["inventories"].append(
                {
                    "source_pool_sha256": source_pool_sha256(documents),
                    "document_ids": [d.document_id.decode() for d in documents],
                    "source_count": len(documents),
                    "source_bytes": sum(len(d.tokens.raw) for d in documents),
                    "word_count": sum(len(d.tokens.byte_spans) for d in documents),
                    "legal_span_count": sum(
                        len(d.tokens.byte_spans) * (len(d.tokens.byte_spans) + 1) // 2
                        for d in documents
                    ),
                }
            )

        def task_for(phase, arm, item):
            ordinal = ids.index(item)
            lesson = lessons[item]
            return ActionTask(
                f"{phase}-{arm}-{ordinal:03d}",
                f"Configure qa{ordinal:03d}".encode(),
                f"qa{ordinal:03d}",
                f"qa_config{ordinal:03d}",
                "pending",
                None if lesson["is_impossible"] else bytes.fromhex(lesson["accepted_text_hex"][0]),
            )

        def probe(phase, items):
            before = {a: p.checkpoint_sha256 for a, p in cores.items()}
            actual = report["actions"][phase] = {a: [] for a in ARMS}
            oracle = report["oracle"][phase] = {a: [] for a in ARMS}
            pool = source_pool_sha256(documents)
            for item in items:
                lesson = lessons[item]
                query = queries[item]
                gold, _ = _gold(lesson, documents)
                original = tuple(
                    d for d in documents if d.document_id == lesson["paragraph_id"].encode()
                )
                _, oracle_gold = _gold(lesson, original)
                for arm, core in cores.items():
                    guard("primary:" + phase + ":" + arm)
                    selected = core.predict(query, documents)
                    report["counters"]["primary_predictions"] += 1
                    if core.checkpoint_sha256 != before[arm]:
                        raise RuntimeError("primary prediction changed state")
                    task = task_for(phase, arm, item)
                    row = execute_prediction(
                        selected,
                        query,
                        documents,
                        motor,
                        task,
                        work / task.name,
                        reader=core,
                        custody=PointerCustody(selected, before[arm], query.sha256, pool),
                        accepted_occurrences=gold,
                        semantic_scored=item not in diagnostic,
                    )
                    actual[arm].append({"id": item, "supported": gold is not None, **row})
                    report["counters"]["primary_actions"] += 1
                    report["counters"]["child_executions"] += 1
                    guard("oracle:" + phase + ":" + arm)
                    selected = core.predict(query, original)
                    report["counters"]["oracle_predictions"] += 1
                    occurrence = None if selected is None else _occurrence(original, selected)
                    agreement = selected is None if oracle_gold is None else selected in oracle_gold
                    oracle[arm].append(
                        {
                            "id": item,
                            "supported": gold is not None,
                            "semantic_scored": item not in diagnostic,
                            "correct": agreement if item not in diagnostic else None,
                            "official_label_agreement": agreement,
                            "selected_pointer": selected,
                            "selected_occurrence": occurrence,
                            "reader_checkpoint_sha256": before[arm],
                            "source_pool_sha256": source_pool_sha256(original),
                            "query_sha256": query.sha256,
                        }
                    )
                    if core.checkpoint_sha256 != before[arm]:
                        raise RuntimeError("oracle prediction changed state")
            if before != {a: p.checkpoint_sha256 for a, p in cores.items()}:
                raise RuntimeError("probe/action changed another policy state")
            note(
                phase,
                documents=len(documents),
                scores={a: _counts(rows) for a, rows in actual.items()},
            )

        for item in blocks["O"]:
            add(item)
        inventory()
        probe("initial", blocks["O"])
        for i, item in enumerate(blocks["E"]):
            guard("source_growth")
            add(item)
            inventory()
            phase = f"episode_{i}"
            probe(phase, [item])
            note("all_first_attempts_complete", episode=i, policies=list(ARMS))
            for arm in UPDATING:
                core = cores[arm]
                first = report["actions"][phase][arm][0]
                gold, _ = _gold(lessons[item], documents)
                if first["success"] is False:
                    guard("teacher:" + arm)
                    teacher = execute_teacher(
                        {k: v for k, v in first.items() if k not in ("id", "supported")},
                        task_for(phase, arm, item),
                        queries[item],
                        documents,
                        reader=core,
                        motor=motor,
                        accepted_occurrences=gold,
                    )
                    report["teachers"].append({"episode": i, "arm": arm, "id": item, **teacher})
                    report["counters"]["teacher_executions"] += 1
                    report["counters"]["child_executions"] += 1
                    if not teacher["passed"]:
                        raise RuntimeError("teacher correction failed; feedback not authorized")
                for kind, target in (
                    ("feedback", item),
                    ("replay", blocks["O"][2 * i]),
                    ("replay", blocks["O"][2 * i + 1]),
                ):
                    if kind == "feedback" and item in diagnostic:
                        continue
                    guard("observe:" + arm)
                    accepted, pointers = _gold(lessons[target], documents)
                    before = core.checkpoint_sha256
                    row = {
                        "episode": i,
                        "arm": arm,
                        "kind": kind,
                        "id": target,
                        "query_sha256": queries[target].sha256,
                        "source_pool_sha256": source_pool_sha256(documents),
                        "gold_pointers": pointers,
                        "accepted_occurrences": None
                        if accepted is None
                        else [(d.hex(), a, b) for d, a, b in accepted],
                        "before": before,
                        "steps_before": core.steps,
                        "presentations_before": core.presentations,
                    }
                    report["training"].append(row)
                    row["receipt"] = core.update_pool(queries[target], documents, pointers)
                    row.update(
                        after=core.checkpoint_sha256,
                        steps_after=core.steps,
                        presentations_after=core.presentations,
                    )
                    if (
                        core.steps != row["steps_before"] + 1
                        or core.presentations != row["presentations_before"] + 1
                    ):
                        raise RuntimeError("continuation offer counters differ")
                    report["counters"]["offers"] += 1
                    report["counters"]["offers_per_updating_arm"][arm] += 1
            if frozen != {a: cores[a].checkpoint_sha256 for a in frozen}:
                raise RuntimeError("frozen continuation state changed")
            note("feedback_complete", episode=i)
        for item in blocks["L"]:
            add(item)
        inventory()
        probe("late", blocks["L"])
        probe("final", blocks["O"] + blocks["E"])
        if any(
            (p.steps, p.presentations) != (inherited[a][0] + 23, inherited[a][1] + 23)
            for a, p in policies.items()
        ):
            raise RuntimeError("fixed continuation offer budget differs")
        if (
            frozen != {a: cores[a].checkpoint_sha256 for a in frozen}
            or motor.checkpoint_sha256 != MOTOR_SHA256
        ):
            raise RuntimeError("frozen policy or motor changed")
        if (
            sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA256
            or sha256(native.library_path.read_bytes()).hexdigest()
            != report["native_motor"]["sha256"]
        ):
            raise RuntimeError("fixture/native source changed during session")
        report["final_states"] = snapshots()
        report["status"] = "completed"
        if not decision(report)["complete"]:
            report["status"] = "failed"
            raise RuntimeError("completed integration evidence roster differs")
        guard("integration_complete")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            report["partial_states"] = snapshots()
        except Exception as state_error:
            report["partial_state_error"] = repr(state_error)
    report["elapsed_seconds"] = time.perf_counter() - began
    report["decision"] = decision(report)
    return report
