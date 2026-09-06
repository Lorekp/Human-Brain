"""Independent saved-record review of the public persistent development session.

No learner, project scorer, checkpoint loader, or execution helper is imported.
This verifies recorded behavior and custody, not hidden neuronal arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

ROUTES = ("online", "learning_disabled", "simple")
PHASES = ("acquisition", "transfer", "recovery", "retention")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def records(path):
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            require(len(raw) <= 4 * 1024 * 1024, "journal row exceeds review bound")
            rows.append((json.loads(raw), digest(raw)))
    return rows


def files(rows):
    result = {row["path"]: bytes.fromhex(row["content_hex"]) for row in rows}
    require(len(result) == len(rows), "duplicate file in saved observation")
    return result


def expected_score(episode, row):
    if row["status"] == "unsupported":
        return False
    if episode["kind"] == "reading":
        return row["answer_hex"] in episode["grade"]["accepted_answer_hex"]
    execution = row["execution"]
    if not execution:
        return False
    grade = episode["grade"]
    before, after = files(episode["files"]), files(row["after_files"])
    proposed = files(execution["proposed_files"])
    protected = all(
        after.get(path) == before[path] and proposed.get(path) == before[path]
        for path in grade["protected_paths"]
    )
    return bool(
        execution["returncode"] == grade["expected_returncode"]
        and execution["stdout_hex"] == grade["expected_stdout_hex"]
        and protected
        and execution["files_complete"]
        and execution["capture_complete"]
        and not execution["timeout"]
        and not execution["output_limit_exceeded"]
        and not execution["capture_errors"]
    )


def verify_application(row):
    execution = row["execution"]
    if not execution:
        return
    before = files(row["before_files"])
    proposed = dict(before)
    proposal = row["proposal"]
    if execution["accepted"]:
        require(proposal is not None, "accepted missing proposal")
        require(proposal["operation"] in {"STOP", "SPLICE"}, "unknown operation")
        if proposal["operation"] == "SPLICE":
            path, start, end = proposal["path"], proposal["start"], proposal["end"]
            require(
                path in before and 0 <= start <= end <= len(before[path]), "invalid accepted span"
            )
            replacement = bytes.fromhex(proposal["replacement_hex"])
            require(
                len(replacement) <= 16 and proposal["stop_reason"] == "eos",
                "invalid accepted payload",
            )
            proposed[path] = before[path][:start] + replacement + before[path][end:]
    require(proposed == files(execution["proposed_files"]), "host substituted or repaired an edit")
    require(execution["before_files"] == row["before_files"], "executor input mismatch")
    require(execution["after_files"] == row["after_files"], "executor result mismatch")
    for stream in ("stdout", "stderr"):
        require(
            digest(bytes.fromhex(execution[stream + "_hex"])) == execution[stream + "_sha256"],
            "captured output hash mismatch",
        )
    # Bind saved workspace bytes when a complete snapshot was possible.
    if execution["files_complete"]:
        directory = Path(execution["workspace"])
        for path, raw in files(row["after_files"]).items():
            target = (directory / path).resolve()
            require(target.is_relative_to(directory.resolve()), "saved workspace path escapes")
            require(target.read_bytes() == raw, "saved after-file mismatch")


def review(run, launch_path, manifest_sha):
    started = time.perf_counter()
    completion = read_json(run / "completion.json")
    require(completion["status"] == "completed", "incomplete session")
    required_files = {
        "manifest.json",
        "report.json",
        "attempts.jsonl",
        "teachers.jsonl",
        "ingests.jsonl",
        "initialization.json",
        "sources.json",
        "launch-record.json",
    }
    require(required_files <= set(completion["files"]), "incomplete artifact binding")
    for relative, expected in completion["files"].items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run.resolve()), "completion path escapes run")
        require(digest(path.read_bytes()) == expected, f"changed artifact: {relative}")
    manifest = read_json(run / "manifest.json")
    require(
        digest((run / "manifest.json").read_bytes()) == manifest_sha,
        "manifest differs from pre-run freeze",
    )
    source_root = Path(__file__).resolve().parents[2]
    for relative, expected in read_json(run / "sources.json").items():
        require(
            digest((source_root / relative).read_bytes()) == expected, "reviewed run source changed"
        )
    episodes = {e["episode_id"]: e for e in manifest["episodes"]}
    require(len(episodes) == 56, "wrong episode roster")
    require(manifest["counts"]["occupied_memory_slots"] == 453, "wrong memory stream")
    attempt_rows = records(run / "attempts.jsonl")
    teacher_rows = records(run / "teachers.jsonl")
    ingest_rows = records(run / "ingests.jsonl")
    report = read_json(run / "report.json")
    initialization = read_json(run / "initialization.json")
    require(
        initialization["bank_sha256"]
        == "807e256dbb750a41e24a00997bee89dbfb7ad4d6e2ee7ab3dc556686f93770a8",
        "wrong prior bank",
    )
    require(
        initialization["prior_sha256"]
        == "deeb951704e59abf1e819172ffa7345558bc94151af4783970d2864016ea8de7",
        "wrong numerical prior",
    )
    require(
        initialization["torch_load_calls"] == 1
        and initialization["loaded_payload_tensor_bytes"] == 109896320,
        "prior load accounting",
    )
    for route in ROUTES[:2]:
        require(
            report["initial_hashes"][route]["model"]
            == initialization["initial_model_sha256"][route],
            "initialization state changed before stream",
        )
        require(
            initialization["initial_model_tensor_sha256"][route]
            == initialization["prior_numeric_tensor_sha256"],
            "numeric prior mismatch",
        )
    child_launch = read_json(run / "launch-record.json")
    child_launch_sha = digest(
        json.dumps(child_launch, sort_keys=True, separators=(",", ":")).encode()
    )
    require(
        report["initial_hashes"]["online"]["model"]
        == report["initial_hashes"]["learning_disabled"]["model"],
        "unmatched initial neuronal models",
    )
    by_episode = defaultdict(list)
    attempt_by_sha = {}
    for row, sha in attempt_rows:
        route, key = row["route"], row["episode_id"]
        require(route in ROUTES and key in episodes, "unknown route or episode")
        episode = episodes[key]
        require(row["phase"] == episode["phase"] and row["kind"] == episode["kind"], "phase drift")
        require(row["request_hex"] == episode["request_hex"], "request changed or truncated")
        require(row["feedback_consumed"] is False, "undeclared feedback encoder")
        obs = row["observation"]
        require(set(obs) == {"kind", "request_hex", "files"}, "private observation field")
        require(
            obs["request_hex"] == row["request_hex"] and obs["files"] == row["before_files"],
            "observation drift",
        )
        previous = by_episode[route, key]
        require(row["attempt_index"] == len(previous) + 1, "attempt numbering")
        require(len(previous) < episode["attempt_limit"], "attempt budget exceeded")
        require(
            row["before_files"] == (previous[-1]["after_files"] if previous else episode["files"]),
            "workspace reset or repair within retry",
        )
        require(row["correct"] is expected_score(episode, row), "recorded correctness mismatch")
        verify_application(row)
        execution = row["execution"]
        if execution:
            require(
                execution["launch_record_sha256"] == child_launch_sha,
                "child launch identity changed",
            )
            command = execution["child_command"]
            require(
                command[:-1] == [child_launch["base_executable"], *child_launch["flags"], "-c"],
                "child runtime or flags changed",
            )
            require(
                digest(command[-1].encode())
                == execution["bootstrap_sha256"]
                == "7213881ba88871e024789913aa742bba520cc79520b8a7b87292c0d3b2b745b9",
                "child bootstrap changed",
            )
        if route != "simple":
            require(
                row["model_before_sha256"] == row["model_after_sha256"],
                "free step changed slow weights",
            )
            require(
                row["memory_before_sha256"] == row["memory_after_sha256"], "free step changed tape"
            )
        previous.append(row)
        require(sha not in attempt_by_sha, "duplicate attempt bytes")
        attempt_by_sha[sha] = row

    teaching = defaultdict(list)
    for row, _ in teacher_rows:
        require(row["attempt_sha256"] in attempt_by_sha, "teacher lacks preceding attempt")
        attempt = attempt_by_sha[row["attempt_sha256"]]
        episode = episodes[attempt["episode_id"]]
        require(
            attempt["phase"] == "acquisition" and attempt["attempt_index"] == 1,
            "teacher outside acquisition",
        )
        require(row["teacher"] == episode["teacher"], "teacher differs from frozen offer")
        require(row["route"] == attempt["route"], "teacher route mismatch")
        if row["route"] != "simple":
            require(
                row["learning_enabled"] is (row["route"] == "online"),
                "wrong local publication gate",
            )
            require(
                row["model_before_sha256"] == attempt["model_after_sha256"], "teacher state custody"
            )
            if row["route"] == "learning_disabled":
                require(
                    row["model_before_sha256"] == row["model_after_sha256"],
                    "disabled model changed",
                )
        teaching[row["route"]].append(row)

    documents = {d["document_id"]: bytes.fromhex(d["text_hex"]) for d in manifest["documents"]}
    ingests = defaultdict(list)
    for row, _ in ingest_rows:
        ingests[row["route"]].append(row)
    summaries = {}
    for route in ROUTES:
        require(len(teaching[route]) == 24, "missing or repeated correction offer")
        require(len({t["attempt_sha256"] for t in teaching[route]}) == 24, "replayed correction")
        selected = [r for r, _ in attempt_rows if r["route"] == route]
        require(
            len({r["model_instance_id"] for r in selected}) == 1, "model replaced during session"
        )
        require(
            [r["episode_id"] for r in selected if r["attempt_index"] == 1] == list(episodes),
            "episode order or missing exposure",
        )
        phase_scores = {}
        for phase in PHASES:
            subset = [e for e in episodes.values() if e["phase"] == phase]
            first = [by_episode[route, e["episode_id"]][0] for e in subset]
            phase_scores[phase] = {
                "first_correct": sum(r["correct"] for r in first),
                "episodes": len(first),
                "reading_correct": sum(r["correct"] for r in first if r["kind"] == "reading"),
                "reading_total": sum(r["kind"] == "reading" for r in first),
                "edit_correct": sum(r["correct"] for r in first if r["kind"] == "edit"),
                "edit_total": sum(r["kind"] == "edit" for r in first),
                "supported_correct": sum(
                    r["correct"] for r in first if episodes[r["episode_id"]]["supported"]
                ),
                "supported_total": sum(episodes[r["episode_id"]]["supported"] for r in first),
            }
        recovery = [
            by_episode[route, e["episode_id"]]
            for e in episodes.values()
            if e["phase"] == "recovery"
        ]
        require(all(len(rows) == 3 for rows in recovery), "recovery must use three fixed attempts")
        phase_scores["recovery"].update(
            first_failures=sum(not rows[0]["correct"] for rows in recovery),
            recovered=sum(
                not rows[0]["correct"] and any(r["correct"] for r in rows[1:]) for rows in recovery
            ),
            any_attempt_correct=sum(any(r["correct"] for r in rows) for rows in recovery),
            final_correct=sum(rows[-1]["correct"] for rows in recovery),
            attempts=sum(len(rows) for rows in recovery),
        )
        retention_pairs = []
        for e in episodes.values():
            if e["phase"] != "retention":
                continue
            earlier = "A" + e["episode_id"][1:]
            retention_pairs.append(
                {
                    "episode_id": e["episode_id"],
                    "acquisition_first_correct": by_episode[route, earlier][0]["correct"],
                    "retention_correct": by_episode[route, e["episode_id"]][0]["correct"],
                }
            )
        phase_scores["retention"]["pairs"] = retention_pairs
        phase_scores["acquisition"]["blocks_of_eight"] = [
            sum(
                by_episode[route, e["episode_id"]][0]["correct"]
                for e in list(episodes.values())[i : i + 8]
            )
            for i in (0, 8, 16)
        ]
        appended = [key for r in ingests[route] for key in r["document_ids"]]
        require(appended == list(documents), "memory append order, duplication, or target routing")
        require(sum(len(documents[key]) + 1 for key in appended) == 453, "memory byte/EOS exposure")
        current = dict(report["initial_hashes"][route])
        used = 0
        ingest_by_episode = {r["episode_id"]: r for r in ingests[route]}
        require(len(ingest_by_episode) == len(ingests[route]), "duplicate ingest event")
        teacher_by_attempt = {r["attempt_sha256"]: r for r in teaching[route]}
        for row, row_sha in attempt_rows:
            if row["route"] != route:
                continue
            episode = episodes[row["episode_id"]]
            if row["attempt_index"] == 1 and episode["append_document_ids"]:
                ingest = ingest_by_episode[row["episode_id"]]
                require(
                    ingest["document_ids"] == episode["append_document_ids"],
                    "append scheduled at wrong episode",
                )
                require(ingest["used_before"] == used, "ingest cursor reset")
                raw_bytes = sum(len(documents[key]) for key in ingest["document_ids"])
                require(
                    ingest["raw_bytes"] == raw_bytes
                    and ingest["eos"] == len(ingest["document_ids"]),
                    "ingest exposure mismatch",
                )
                used += raw_bytes + ingest["eos"]
                require(ingest["used_after"] == used, "ingest occupancy mismatch")
                for kind in ("model", "memory"):
                    require(ingest[kind + "_before_sha256"] == current[kind], "ingest state chain")
                    current[kind] = ingest[kind + "_after_sha256"]
                require(
                    ingest["model_before_sha256"] == ingest["model_after_sha256"],
                    "ingest changed slow weights",
                )
            require(row["memory_used"] == used, "task-selected or truncated memory")
            for kind in ("model", "memory"):
                require(row[kind + "_before_sha256"] == current[kind], "attempt state chain")
                current[kind] = row[kind + "_after_sha256"]
            teacher = teacher_by_attempt.get(row_sha)
            if teacher:
                for kind in ("model", "memory"):
                    require(
                        teacher[kind + "_before_sha256"] == current[kind], "correction state chain"
                    )
                    current[kind] = teacher[kind + "_after_sha256"]
                require(
                    teacher["memory_before_sha256"] == teacher["memory_after_sha256"],
                    "teacher secretly wrote memory",
                )
                if route != "simple":
                    target = teacher["teacher"].get(
                        "target_hex", teacher["teacher"].get("replacement_hex")
                    )
                    require(
                        teacher["reader_offers"] == len(bytes.fromhex(target)) + 1,
                        "teacher-symbol exposure mismatch",
                    )
                    require(
                        teacher["action_labels"] == (3 if row["kind"] == "edit" else 0),
                        "action teacher exposure mismatch",
                    )
        require(current == report["final_hashes"][route], "final persistent state mismatch")
        recorded = report["metrics"][route]
        for phase in PHASES:
            for kind in ("reading", "edit"):
                summary = phase_scores[phase]
                require(
                    recorded["phases"][phase][kind]["first_correct"] == summary[kind + "_correct"],
                    "report first-score aggregation mismatch",
                )
                require(
                    recorded["phases"][phase][kind]["episodes"] == summary[kind + "_total"],
                    "report phase denominator mismatch",
                )
        require(
            recorded["recovery"]["later_correct"] == phase_scores["recovery"]["recovered"],
            "report recovery mismatch",
        )
        summaries[route] = phase_scores

    launch = read_json(launch_path)
    require(launch["status"] == "passed", "workload watchdog failed")
    require(
        launch["elapsed_seconds"] <= 840 and launch["observed_tree_peak_sum_mib"] <= 2048,
        "workload resource envelope",
    )
    command = launch["command"]
    require(
        command[1:5]
        == ["-X", "utf8", "-m", "experiments.r4_integrated.persistent_language_action_v1"],
        "unrelated workload command",
    )
    require(
        Path(command[command.index("--output") + 1]).resolve() == run.resolve(),
        "launch output mismatch",
    )
    require(
        command[command.index("--manifest-sha") + 1] == manifest_sha,
        "launch manifest freeze mismatch",
    )
    require(
        any(
            key.startswith(str(completion["pid"]) + "/")
            for key in launch["observed_process_peaks_mib"]
        ),
        "runner PID absent from owned process tree",
    )
    require(report["pid"] == completion["pid"], "report/completion PID mismatch")
    require(
        report["final_checkpoint"]["sha256"] == completion["files"]["final-state.pt"],
        "final continuation checkpoint unbound",
    )
    return {
        "schema": "persistent-language-action-saved-review-v1",
        "status": "passed",
        "summaries": summaries,
        "attempt_rows": len(attempt_rows),
        "teacher_rows": len(teacher_rows),
        "ingest_rows": len(ingest_rows),
        "manifest_sha256": digest((run / "manifest.json").read_bytes()),
        "completion_sha256": digest((run / "completion.json").read_bytes()),
        "launch_sha256": digest(launch_path.read_bytes()),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "saved behavior, exposure and state-custody review; no neuronal arithmetic replay",
        "model_constructions": 0,
        "checkpoint_loads": 0,
        "actual_executions": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    result = review(args.run, args.launch, args.manifest_sha)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "attempt_rows": result["attempt_rows"]}))


if __name__ == "__main__":
    main()
