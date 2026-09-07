"""Restore actual simple-reference observations; no predictions or teaching replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_baseline_v2 import SimpleReference
from experiments.r4_integrated.persistent_language_action_tasks_v4 import (
    restore_baseline as restore_previous_baseline,
)
from experiments.r4_integrated.persistent_language_action_tasks_v5 import (
    HISTORY_DIRECTORY,
    HISTORY_SHA256,
)

JOURNAL_SHA256 = {
    "teachers.jsonl": "60ebc2d149a9452211872b9507c078b71413cbd399b6c89449032aafe7b8aad7",
    "attempts.jsonl": "803bacd6949a9c695c30d30e8d3ff1f86c4203b0c019bf1df86919e91f6ee400",
    "ingests.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def restore_baseline(root: Path):
    """Authenticate complete journals while retaining only six relevant rows.

    Full neuronal attempt lines are hashed without JSON decoding. Restoring an
    observed demonstration does not invoke the reference's teacher method or
    execute its old program. The historical receipt retains those exposures.
    """
    root = Path(root)
    baseline, previous = restore_previous_baseline(root)
    directory = root / HISTORY_DIRECTORY
    manifest_raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != HISTORY_SHA256["manifest.json"]:
        raise ValueError("fourth-session reference manifest changed")
    manifest = json.loads(manifest_raw)
    if manifest["documents"] or (directory / "ingests.jsonl").read_bytes() != b"":
        raise ValueError("fourth-session reference has unexpected new documents")
    teachers, journal_bytes, journal_rows = [], {}, {}
    digest = hashlib.sha256()
    total = count = 0
    with (directory / "teachers.jsonl").open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            total += len(raw)
            count += 1
            row = json.loads(raw)
            if row["route"] == "simple" and row["teacher"]["kind"] == "edit":
                teachers.append(row)
    if digest.hexdigest() != JOURNAL_SHA256["teachers.jsonl"] or len(teachers) != 6:
        raise ValueError("fourth-session simple corrections changed")
    journal_bytes["teachers.jsonl"], journal_rows["teachers.jsonl"] = total, count
    wanted = {row["attempt_sha256"] for row in teachers}
    if len(wanted) != 6:
        raise ValueError("simple corrections must bind six distinct real attempts")
    attempts, digest = {}, hashlib.sha256()
    total = count = 0
    with (directory / "attempts.jsonl").open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            total += len(raw)
            count += 1
            identity = hashlib.sha256(raw).hexdigest()
            if identity in wanted:
                if identity in attempts:
                    raise ValueError("duplicated historical corrected attempt")
                attempts[identity] = json.loads(raw)
    if digest.hexdigest() != JOURNAL_SHA256["attempts.jsonl"] or set(attempts) != wanted:
        raise ValueError("fourth-session corrected attempts changed")
    journal_bytes["attempts.jsonl"], journal_rows["attempts.jsonl"] = total, count
    episode_by_id = {row["episode_id"]: row for row in manifest["episodes"]}
    bindings = []
    for row in teachers:
        teacher, attempt = row["teacher"], attempts[row["attempt_sha256"]]
        episode = episode_by_id[row["episode_id"]]
        if (attempt["route"] != "simple" or attempt["phase"] != "acquisition"
                or attempt["attempt_index"] != 1 or attempt["episode_id"] != row["episode_id"]
                or attempt["execution"] is None or teacher["operation"] != "SPLICE"
                or teacher != episode["teacher"] or attempt["request_hex"] != episode["request_hex"]):
            raise ValueError("historical reference correction lacks its actual preceding action")
        source = next(bytes.fromhex(file["content_hex"]) for file in attempt["before_files"]
                      if file["path"] == teacher["path"])
        start, end = teacher["start"], teacher["end"]
        if not 0 <= start <= end <= len(source):
            raise ValueError("historical correction span outside actual workspace")
        # These six first teaching episodes carry no tool feedback. Their actual
        # reference similarity input is therefore the unmodified raw request.
        if attempt["observation"]["feedback"] is not None:
            raise ValueError("historical first demonstration unexpectedly contains feedback")
        query = bytes.fromhex(attempt["request_hex"])
        baseline.demonstrations.append({
            "query": query, "old": source[start:end],
            "replacement": bytes.fromhex(teacher["replacement_hex"]),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "query_sha256": hashlib.sha256(query).hexdigest(),
        })
        bindings.append({"episode_id": row["episode_id"], "attempt_sha256": row["attempt_sha256"]})
    if (len(baseline.documents) != 18 or len(baseline.demonstrations) != 42
            or sum(len(document) + 1 for document in baseline.documents) != 505):
        raise ValueError("reference must restore18documents/42demonstrations/505symbols")
    return SimpleReference.from_restored(baseline), {
        "schema": "persistent-language-action-baseline-restoration-v5",
        "previous_restoration": previous, "fourth_session_bindings": bindings,
        "input_sha256": {"manifest.json": HISTORY_SHA256["manifest.json"], **JOURNAL_SHA256},
        "journal_bytes_read": journal_bytes, "journal_rows_read": journal_rows,
        "retained_fourth_session_attempts": len(attempts), "documents": 18,
        "edit_demonstrations": 42, "document_raw_bytes": 487, "document_eos": 18,
        "new_teacher_offers": 0, "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0, "reconstructed_sessions": 4,
    }
