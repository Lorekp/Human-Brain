"""Restore the reference's actual V1–V5 experience without prediction replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_baseline_restore_v5 import (
    restore_baseline as restore_previous_baseline,
)
from experiments.r4_integrated.persistent_language_action_baseline_v2 import SimpleReference

HISTORY_DIRECTORY = ".pytest-tmp/persistent-language-action-v5-run"
MANIFEST_SHA256 = "de182acf644b18fed25948c5c7886006253bee8618a4e6ce647508f624b86fca"
JOURNAL_SHA256 = {
    "teachers.jsonl": "51713e14cbc82e575106adef182ef5e13d8af97be8a5f0c11eba453786712d74",
    "attempts.jsonl": "9139824c7897f15e3a3aed71a7692f7a15c28ad18fa4d82d48072dee5981e638",
    "ingests.jsonl": "a4ea76e0aca10a801b4815927933b41dfd3ac5672fde659f9f0e9ae00aa17f3c",
}


def restore_baseline(root: Path):
    """Authenticate complete journals, decode only four bound attempt rows.

    Earlier restoration retains its own full-history costs. This layer adds
    only documents actually observed and corrections following actual actions;
    it invokes neither predict, observe_teacher nor the workspace executor.
    """
    root = Path(root)
    baseline, previous = restore_previous_baseline(root)
    directory = root / HISTORY_DIRECTORY
    raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("fifth-session reference manifest changed")
    manifest = json.loads(raw)
    journal_bytes, journal_rows = {}, {}

    def read_journal(name, retain):
        digest, rows = hashlib.sha256(), []
        total = count = 0
        with (directory / name).open("rb") as handle:
            for line in handle:
                digest.update(line)
                total += len(line)
                count += 1
                identity = hashlib.sha256(line).hexdigest()
                if retain(identity):
                    rows.append((identity, json.loads(line)))
        if digest.hexdigest() != JOURNAL_SHA256[name]:
            raise ValueError("fifth-session reference journal changed: " + name)
        journal_bytes[name], journal_rows[name] = total, count
        return rows

    teachers = [row for _, row in read_journal("teachers.jsonl", lambda _: True)
                if row["route"] == "simple" and row["teacher"]["kind"] == "edit"]
    wanted = {row["attempt_sha256"] for row in teachers}
    if len(teachers) != 4 or len(wanted) != 4:
        raise ValueError("reference requires four distinct fifth-session corrections")
    attempt_rows = read_journal("attempts.jsonl", wanted.__contains__)
    attempts = dict(attempt_rows)
    if len(attempt_rows) != 4 or set(attempts) != wanted:
        raise ValueError("reference corrections lack their four actual attempts")
    ingests = [row for _, row in read_journal("ingests.jsonl", lambda _: True)
               if row["route"] == "simple"]
    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    documents = {row["document_id"]: row for row in manifest["documents"]}
    expected_ingests = [(row["episode_id"], row["append_document_ids"])
                        for row in manifest["episodes"] if row["append_document_ids"]]
    if (len(ingests) != 6 or len(documents) != 6
            or [(row["episode_id"], row["document_ids"]) for row in ingests] != expected_ingests):
        raise ValueError("reference factual observation order changed")
    used, document_bindings = 505, []
    for row in ingests:
        observed = []
        for document in row["documents"]:
            expected = documents[document["document_id"]]
            content = bytes.fromhex(document["text_hex"])
            if (document["text_hex"] != expected["text_hex"]
                    or hashlib.sha256(content).hexdigest() != document["sha256"]):
                raise ValueError("reference document differs from its actual observation")
            observed.append(content)
        symbols = sum(map(len, observed)) + len(observed)
        if ([document["document_id"] for document in row["documents"]] != row["document_ids"]
                or row["used_before"] != used or row["used_after"] != used + symbols
                or row["raw_bytes"] != symbols - len(observed) or row["eos"] != len(observed)
                or row["source_next_byte_teacher_symbols"] != 0
                or row["association_writes_enabled"] is not True):
            raise ValueError("reference factual append custody changed")
        baseline.documents.extend(observed)
        used += symbols
        document_bindings.append({"episode_id": row["episode_id"],
                                  "document_ids": row["document_ids"]})
    bindings = []
    for row in teachers:
        teacher, attempt = row["teacher"], attempts[row["attempt_sha256"]]
        episode = episodes[row["episode_id"]]
        if (attempt["route"] != "simple" or attempt["phase"] != "acquisition"
                or attempt["attempt_index"] != 1 or attempt["episode_id"] != row["episode_id"]
                or attempt["execution"] is None or teacher["operation"] != "SPLICE"
                or teacher != episode["teacher"] or attempt["request_hex"] != episode["request_hex"]
                or attempt["observation"]["feedback"] is not None):
            raise ValueError("reference correction lacks its actual preceding action")
        source = next(bytes.fromhex(file["content_hex"]) for file in attempt["before_files"]
                      if file["path"] == teacher["path"])
        start, end = teacher["start"], teacher["end"]
        if not 0 <= start <= end <= len(source):
            raise ValueError("reference correction span outside actual workspace")
        query = bytes.fromhex(attempt["request_hex"])
        baseline.demonstrations.append({
            "query": query, "old": source[start:end],
            "replacement": bytes.fromhex(teacher["replacement_hex"]),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "query_sha256": hashlib.sha256(query).hexdigest(),
        })
        bindings.append({"episode_id": row["episode_id"], "attempt_sha256": row["attempt_sha256"]})
    if (len(baseline.documents) != 24 or len(baseline.demonstrations) != 46
            or sum(len(document) + 1 for document in baseline.documents) != 864 or used != 864):
        raise ValueError("reference must restore24documents/46demonstrations/864symbols")
    return SimpleReference.from_restored(baseline), {
        "schema": "persistent-language-action-baseline-restoration-v6",
        "previous_restoration": previous, "fifth_session_bindings": bindings,
        "fifth_session_document_bindings": document_bindings,
        "input_sha256": {"manifest.json": MANIFEST_SHA256, **JOURNAL_SHA256},
        "journal_bytes_read": journal_bytes, "journal_rows_read": journal_rows,
        "retained_fifth_session_attempts": len(attempts), "documents": 24,
        "edit_demonstrations": 46, "document_raw_bytes": 840, "document_eos": 24,
        "new_teacher_offers": 0, "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0, "reconstructed_sessions": 5,
    }
