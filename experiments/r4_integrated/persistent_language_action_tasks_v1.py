"""Public development data and explicitly conventional, target-free references.

No model, executor or historical experiment is imported. Teacher/grade records
are parent-only. Baseline prediction accepts only raw observations and history
that was actually presented, never an episode ID or grading object.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter

SCHEMA = "persistent-language-action-development-data-v1"
PHASES = ("acquisition", "transfer", "recovery", "retention")
FACTS = (
    ("The badge is 'blue'.\n", "What colour is the badge?", b"blue", b"'blue'"),
    ("Harbor's code is 'Q7'.\n", "What is Harbor's code?", b"Q7", b"'Q7'"),
    ("Retries default to '3'.\n", "How many retries by default?", b"3", b"3"),
    ("On delay, wait '2' seconds.\n", "How many seconds on delay?", b"2", b"2"),
    ("Archive files use '.bak'.\n", "What suffix do archives use?", b".bak", b"'.bak'"),
    ("Logs are named 'audit'.\n", "What are the logs named?", b"audit", b"'audit'"),
    ("Dry runs make '0' writes.\n", "How many writes in a dry run?", b"0", b"0"),
    ("Batch size is '4'.\n", "What is the batch size?", b"4", b"4"),
    ("The backup region is 'west'.\n", "Which region holds backups?", b"west", b"'west'"),
    ("Tickets expire in '6' days.\n", "How many days until expiry?", b"6", b"6"),
    ("Empty input returns '0'.\n", "What does empty input return?", b"0", b"0"),
    ("Labels use 'lowercase'.\n", "Which letter case for labels?", b"lowercase", b"'lowercase'"),
)
EXTRA_DOCUMENTS = (
    "During testing, the badge is 'green'.\n",
    "In production, the badge is 'blue'.\n",
    "Large batches contain '8' jobs.\n",
    "Temporary logs are named 'trace'.\n",
)
ACTION_REQUESTS = (
    "Set cfg.py to the badge colour",
    "Set app.py to Harbor's code",
    "Set cfg.py to default retries",
    "Set app.py to the delay time",
    "Set cfg.py to archive suffix",
    "Set app.py to the log name",
    "Set cfg.py to dry run writes",
    "Set app.py to batch size",
    "Set cfg.py to backup region",
    "Set app.py to ticket days",
    "Set cfg.py to empty output",
    "Set app.py to label case",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _files(module: str, source: bytes, main: bytes) -> list[dict]:
    return sorted(
        [
            {"path": module, "content_hex": source.hex()},
            {"path": "main.py", "content_hex": main.hex()},
        ],
        key=lambda row: row["path"].encode(),
    )


def _base(episode_id: str, phase: str, kind: str, query: str, append: tuple[str, ...] = ()) -> dict:
    return {
        "episode_id": episode_id,
        "phase": phase,
        "kind": kind,
        "request_hex": query.encode("utf-8").hex(),
        "files": [],
        "append_document_ids": list(append),
        "attempt_limit": 3 if phase == "recovery" else 1,
        "supported": len(query.encode("utf-8")) <= 31,
        "unsupported_dimension": None if len(query.encode("utf-8")) <= 31 else "request_bytes",
        "exposure_class": "reused_acquisition" if phase == "retention" else phase,
        "teacher": None,
        "grade": None,
    }


def _reading(
    episode_id: str, phase: str, query: str, answer: bytes, append: tuple[str, ...] = ()
) -> dict:
    row = _base(episode_id, phase, "reading", query, append)
    row["grade"] = {"kind": "exact_answer", "accepted_answer_hex": [answer.hex()]}
    if phase == "acquisition":
        row["teacher"] = {"kind": "reading", "target_hex": answer.hex()}
    return row


def _editing(
    episode_id: str,
    phase: str,
    query: str,
    source: bytes,
    marker: bytes,
    replacement: bytes,
    stdout: bytes,
    *,
    module: str = "app.py",
    main: bytes = b"import app\nprint(app.value())\n",
) -> dict:
    # Author-only construction of targets. Never called by reference prediction.
    if not marker or source.count(marker) != 1:
        raise ValueError("authored correction must identify one explicit raw span")
    start = source.index(marker)
    after = source[:start] + replacement + source[start + len(marker) :]
    row = _base(episode_id, phase, "edit", query)
    row["files"] = _files(module, source, main)
    row["grade"] = {
        "kind": "executed_behavior",
        "expected_stdout_hex": stdout.hex(),
        "expected_returncode": 0,
        "expected_after_files": _files(module, after, main),
        "protected_paths": ["main.py"],
        "canonical_proposal": {
            "operation": "SPLICE",
            "path": module,
            "start": start,
            "end": start + len(marker),
            "replacement_hex": replacement.hex(),
            "stop_reason": "eos",
        },
    }
    if phase == "acquisition":
        row["teacher"] = {"kind": "edit", **row["grade"]["canonical_proposal"]}
    return row


def observation(episode: dict) -> dict:
    """Only public raw inputs; phase/IDs/teacher/grade stay outside the model."""
    return {
        "kind": episode["kind"],
        "request_hex": episode["request_hex"],
        "files": copy.deepcopy(episode["files"]),
    }


def build_stream() -> dict:
    """Construct the fixed public data once; never runs any authored program."""
    documents = [
        {"document_id": f"N{i:02d}", "text_hex": raw.encode().hex(), "sha256": _sha(raw.encode())}
        for i, raw in enumerate([fact[0] for fact in FACTS] + list(EXTRA_DOCUMENTS))
    ]
    episodes = []
    for i, (_, query, answer, replacement) in enumerate(FACTS):
        episodes.append(_reading(f"A{i:02d}-R", "acquisition", query, answer, (f"N{i:02d}",)))
        if i % 2 == 0:
            row = _editing(
                f"A{i:02d}-E",
                "acquisition",
                ACTION_REQUESTS[i],
                b"value = 'old'\n",
                b"'old'",
                replacement,
                answer + b"\n",
                module="cfg.py",
                main=b"import cfg\nprint(cfg.value)\n",
            )
        else:
            row = _editing(
                f"A{i:02d}-E",
                "acquisition",
                ACTION_REQUESTS[i],
                b"def value():\n    return 'old'\n",
                b"'old'",
                replacement,
                answer + b"\n",
            )
        episodes.append(row)

    fresh_readings = (
        ("In tests, colour; otherwise?", b"green; blue"),
        ("How long should a delay last?", b"2"),
        ("Name the backup destination.", b"west"),
        ("What follows an empty input?", b"0"),
        ("Give the temporary log name.", b"trace"),
        ("Which suffix marks archives?", b".bak"),
        ("Please tell me the default number of retries documented for this project.", b"3"),
        (
            "Without making edits, explain the letter case required for labels in this project.",
            b"lowercase",
        ),
    )
    fresh_edits = (
        (
            "Use testing colour in app.py",
            b"def value():\n    return ['old'][0]\n",
            b"'old'",
            b"'green'",
            b"green",
        ),
        (
            "Let app.py report large batches",
            b"def value():\n    return {'n': 1}['n']\n",
            b"1",
            b"8",
            b"8",
        ),
        (
            "On delay app.py should wait 2",
            b"def value():\n    return 1 if True else 9\n",
            b"1",
            b"2",
            b"2",
        ),
        (
            "Use temporary logs in app.py",
            b"def value():\n    return str('old')\n",
            b"'old'",
            b"'trace'",
            b"trace",
        ),
        (
            "Make app.py name the backup",
            b"def value():\n    x = ['old']\n    return x[0]\n",
            b"'old'",
            b"'west'",
            b"west",
        ),
        (
            "Report archive suffix in app.py",
            b"def value():\n    x = {'ext': 'old'}\n    return x['ext']\n",
            b"'old'",
            b"'.bak'",
            b".bak",
        ),
        (
            "Please change the function in app.py so that it returns "
            "the documented badge colour in production.",
            b"def value():\n    return ('old')\n",
            b"'old'",
            b"'blue'",
            b"blue",
        ),
        (
            "When no input is supplied, arrange for app.py to return the documented "
            "empty-input result without changing main.py.",
            b"def value():\n    return [1][0]\n",
            b"1",
            b"0",
            b"0",
        ),
    )
    for i, ((query, answer), edit) in enumerate(zip(fresh_readings, fresh_edits, strict=True)):
        episodes.append(
            _reading(
                f"F{i:02d}-R",
                "transfer",
                query,
                answer,
                tuple(f"N{j:02d}" for j in range(12, 16)) if i == 0 else (),
            )
        )
        q, source, marker, replacement, output = edit
        episodes.append(
            _editing(f"F{i:02d}-E", "transfer", q, source, marker, replacement, output + b"\n")
        )

    recovery = (
        (
            "Repair the badge output.",
            b"def value():\n    return missing\n",
            b"missing",
            b"'blue'",
            b"blue",
        ),
        ("Return the first list item.", b"def value():\n    return ['Q7'][3]\n", b"3", b"0", b"Q7"),
        (
            "Repair default retries.",
            b"def value():\n    return int('bad')\n",
            b"'bad'",
            b"'3'",
            b"3",
        ),
        ("Repair the delay result.", b"def value():\n    return 2 / 0\n", b"2 / 0", b"2", b"2"),
        (
            "Make the archive result valid.",
            b"def value():\n    return '.bak' + None\n",
            b" + None",
            b"",
            b".bak",
        ),
        (
            "Repair the log-name lookup.",
            b"def value():\n    return {'log':'audit'}['bad']\n",
            b"'bad'",
            b"'log'",
            b"audit",
        ),
        (
            "Stop the dry-run crash.",
            b"def value():\n    return int(None)\n",
            b"int(None)",
            b"0",
            b"0",
        ),
        (
            "Repair the batch-size output.",
            b"def value():\n    return len(4)\n",
            b"len(4)",
            b"4",
            b"4",
        ),
    )
    for i, (query, source, marker, replacement, output) in enumerate(recovery):
        episodes.append(
            _editing(f"X{i:02d}", "recovery", query, source, marker, replacement, output + b"\n")
        )
    for i in (0, 3, 8, 11):
        for kind in ("R", "E"):
            previous = next(row for row in episodes if row["episode_id"] == f"A{i:02d}-{kind}")
            row = copy.deepcopy(previous)
            row.update(
                episode_id=f"T{i:02d}-{kind}",
                phase="retention",
                teacher=None,
                append_document_ids=[],
                exposure_class="reused_acquisition",
            )
            episodes.append(row)

    # Public author annotations for split auditing, never baseline/model inputs.
    reading_forms = (
        "what_attribute",
        "what_attribute",
        "how_many",
        "how_many",
        "what_attribute",
        "what_attribute",
        "how_many",
        "what_attribute",
        "which_attribute",
        "how_many",
        "what_result",
        "which_attribute",
    )
    fresh_reading_forms = (
        "conditional_otherwise_contrast",
        "how_long",
        "imperative_name",
        "what_follows",
        "imperative_give",
        "which_attribute",
        "polite_full_request",
        "negative_edit_qualification",
    )
    fresh_edit_forms = (
        "imperative_use",
        "imperative_let",
        "conditional_on",
        "imperative_use",
        "imperative_make",
        "imperative_report",
        "polite_full_request",
        "conditional_when",
    )
    fresh_structures = (
        "list_index",
        "dictionary_lookup",
        "conditional_expression",
        "call_argument",
        "assigned_list_index",
        "assigned_dictionary_lookup",
        "parenthesized_return",
        "list_index",
    )
    for row in episodes:
        number = int(row["episode_id"][1:3])
        if row["phase"] in {"acquisition", "retention"}:
            form = reading_forms[number] if row["kind"] == "reading" else "imperative_set"
            structure = (
                "none"
                if row["kind"] == "reading"
                else ("scalar_assignment" if number % 2 == 0 else "direct_function_return")
            )
        elif row["phase"] == "transfer":
            form = (
                fresh_reading_forms[number]
                if row["kind"] == "reading"
                else fresh_edit_forms[number]
            )
            structure = "none" if row["kind"] == "reading" else fresh_structures[number]
        else:
            form, structure = "imperative_repair", "runtime_error_function"
        family = (
            "wh_question"
            if form.startswith(("what_", "which_", "how_"))
            else "imperative"
            if form.startswith(("imperative_", "polite_"))
            else "conditional"
            if form.startswith("conditional_")
            else "qualified_request"
        )
        row["split_annotation"] = {
            "instruction_construction": form,
            "instruction_family": family,
            "workspace_structure": structure,
            "requires_note_relation": row["episode_id"] == "F00-R",
            "relation_note_ids": ["N12", "N13"] if row["episode_id"] == "F00-R" else [],
        }
        row["workspace_lifecycle"] = "reset_at_episode_start_continue_within_retries"

    acquisition = [row for row in episodes if row["phase"] == "acquisition"]
    train_forms = {row["split_annotation"]["instruction_construction"] for row in acquisition}
    train_families = {row["split_annotation"]["instruction_family"] for row in acquisition}
    train_queries = [
        set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower()))
        for row in acquisition
    ]
    lexical_overlap = []
    for row in episodes:
        if row["phase"] != "transfer":
            continue
        words = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower()))
        similarities = [len(words & old) / len(words | old) for old in train_queries]
        annotation = row["split_annotation"]
        lexical_overlap.append(
            {
                "episode_id": row["episode_id"],
                "maximum_acquisition_token_jaccard": max(similarities),
                "instruction_construction_seen": annotation["instruction_construction"]
                in train_forms,
                "broad_instruction_family_seen": annotation["instruction_family"] in train_families,
                "identical_acquisition_request": any(
                    row["request_hex"] == old["request_hex"] for old in acquisition
                ),
            }
        )

    raw_bytes = sum(len(bytes.fromhex(d["text_hex"])) for d in documents)
    counts = {
        "episodes": len(episodes),
        "phase_episodes": {phase: sum(e["phase"] == phase for e in episodes) for phase in PHASES},
        "reading_episodes": sum(e["kind"] == "reading" for e in episodes),
        "edit_episodes": sum(e["kind"] == "edit" for e in episodes),
        "teacher_offers": sum(e["teacher"] is not None for e in episodes),
        "unsupported_episodes": sum(not e["supported"] for e in episodes),
        "maximum_episode_attempts": sum(e["attempt_limit"] for e in episodes),
        "maximum_action_attempts": sum(e["attempt_limit"] for e in episodes if e["kind"] == "edit"),
        "document_raw_bytes": raw_bytes,
        "document_eos": len(documents),
        "occupied_memory_slots": raw_bytes + len(documents),
    }
    if (
        counts["phase_episodes"] != dict(zip(PHASES, (24, 16, 8, 8), strict=True))
        or counts["teacher_offers"] != 24
        or counts["unsupported_episodes"] != 4
        or counts["occupied_memory_slots"] > 512
        or counts["maximum_episode_attempts"] != 72
        or counts["maximum_action_attempts"] != 48
    ):
        raise ValueError(f"public stream scope mismatch: {counts}")
    appended = [doc for e in episodes for doc in e["append_document_ids"]]
    if appended != [d["document_id"] for d in documents]:
        raise ValueError("documents must append once in fixed public order")
    for row in episodes:
        if any(len(bytes.fromhex(f["content_hex"])) > 128 for f in row["files"]):
            raise ValueError("supported workspace geometry exceeded")
        if row["teacher"] is not None and row["phase"] != "acquisition":
            raise ValueError("teacher outside supervised acquisition")
        if row["kind"] == "edit":
            canonical = row["grade"]["canonical_proposal"]
            if len(bytes.fromhex(canonical["replacement_hex"])) > 16:
                raise ValueError("authored replacement exceeds current capacity")
    return {
        "schema": SCHEMA,
        "provenance": {
            "kind": "agent_authored_open_development",
            "official_exam": False,
            "protected_evaluation": False,
            "source_sha256": _sha(__file_bytes()),
        },
        "documents": documents,
        "episodes": episodes,
        "counts": counts,
        "capacity": {
            "memory_slots": 512,
            "query_raw_bytes": 31,
            "file_bytes": 128,
            "files": 4,
            "replacement_bytes": 16,
        },
        "fresh_split": {
            "claim": (
                "new combinations/paraphrases; common question and imperative families overlap"
            ),
            "lexical_overlap": lexical_overlap,
            "fully_withheld_wording_families_claimed": False,
            "new_structures": ["list_index", "dict_lookup", "conditional", "call_argument"],
            "over_capacity_ids": [e["episode_id"] for e in episodes if not e["supported"]],
            "retention_is_exposed": True,
        },
        "persistence": {
            "weights_and_shared_tape": "continue_across_all_episodes",
            "workspace": "reset_at_each_episode_start; continue_only_within_recovery_retries",
            "shared_project_across_episodes": False,
        },
    }


def __file_bytes() -> bytes:
    from pathlib import Path

    return Path(__file__).read_bytes()


def _ngrams(raw: bytes) -> Counter:
    value = b" " + raw.lower() + b" "
    return Counter(value[i : i + 3] for i in range(max(0, len(value) - 2)))


def _similarity(left: bytes, right: bytes) -> float:
    a, b = _ngrams(left), _ngrams(right)
    denom = math.sqrt(sum(x * x for x in a.values()) * sum(x * x for x in b.values()))
    return sum(value * b[key] for key, value in a.items()) / denom if denom else 0.0


class SimpleBaseline:
    """Lexical document retrieval and demonstrated literal edit reuse.

    This conventional baseline does not claim neuronal or semantic inference.
    Reading copies the first single-quoted literal whose opening quote is not
    preceded by an ASCII letter/digit; possessive apostrophes are not openers.
    Editing reuses the closest taught query's raw replacement only where its
    demonstrated nonempty old byte span occurs uniquely in current files.
    """

    def __init__(self) -> None:
        self.documents: list[bytes] = []
        self.demonstrations: list[dict] = []

    def append_documents(self, documents: list[bytes]) -> None:
        if any(not isinstance(raw, bytes) for raw in documents):
            raise TypeError("baseline observes actual raw documents")
        self.documents.extend(documents)

    def observe_teacher(
        self, request: bytes, files: list[dict], *, kind: str, teacher: dict
    ) -> None:
        if kind != "edit":
            return  # Reading is document retrieval, not a target answer table.
        if teacher["kind"] != "edit" or teacher["operation"] != "SPLICE":
            raise ValueError("reference requires an actual supplied edit demonstration")
        source = next(
            bytes.fromhex(f["content_hex"]) for f in files if f["path"] == teacher["path"]
        )
        start, end = teacher["start"], teacher["end"]
        if not 0 <= start <= end <= len(source):
            raise ValueError("demonstration span is outside observed bytes")
        self.demonstrations.append(
            {
                "query": request,
                "old": source[start:end],
                "replacement": bytes.fromhex(teacher["replacement_hex"]),
                "source_sha256": _sha(source),
                "query_sha256": _sha(request),
            }
        )

    def predict(self, request: bytes, files: list[dict], *, kind: str) -> dict:
        if not isinstance(request, bytes) or kind not in {"reading", "edit"}:
            raise ValueError("reference only accepts the raw public observation")
        if kind == "reading":
            if not self.documents:
                return {"answer_hex": "", "attribution": {"method": "empty_document_store"}}
            index = max(
                range(len(self.documents)), key=lambda i: _similarity(request, self.documents[i])
            )
            document = self.documents[index]
            match = re.search(rb"(?<![A-Za-z0-9])'([^'\r\n]*)'", document)
            return {
                "answer_hex": (match.group(1) if match else b"").hex(),
                "attribution": {
                    "method": "trigram_cosine_first_boundary_quoted_literal",
                    "document_index": index,
                    "document_sha256": _sha(document),
                    "score": _similarity(request, document),
                    "literal_span": list(match.span(1)) if match else None,
                },
            }
        stop = {
            "operation": "STOP",
            "path": files[0]["path"] if files else "",
            "start": 0,
            "end": 0,
            "replacement_hex": "",
            "stop_reason": "eos",
        }
        if not self.demonstrations:
            return {"proposal": stop, "attribution": {"method": "no_edit_demonstration"}}
        index = max(
            range(len(self.demonstrations)),
            key=lambda i: _similarity(request, self.demonstrations[i]["query"]),
        )
        demo = self.demonstrations[index]
        locations = []
        if demo["old"]:
            for file in files:
                raw, offset = bytes.fromhex(file["content_hex"]), 0
                while (found := raw.find(demo["old"], offset)) >= 0:
                    locations.append((file["path"], found))
                    offset = found + 1
        attribution = {
            "method": "nearest_taught_query_unique_literal_reuse",
            "demonstration_index": index,
            "demonstration_query_sha256": demo["query_sha256"],
            "demonstration_source_sha256": demo["source_sha256"],
            "score": _similarity(request, demo["query"]),
            "raw_anchor_matches": len(locations),
        }
        if len(locations) != 1:
            return {"proposal": stop, "attribution": attribution}
        path, start = locations[0]
        return {
            "proposal": {
                "operation": "SPLICE",
                "path": path,
                "start": start,
                "end": start + len(demo["old"]),
                "replacement_hex": demo["replacement"].hex(),
                "stop_reason": "eos",
            },
            "attribution": attribution,
        }


def write_stream(path) -> dict:
    """Explicit pure-data publication; never called during import."""
    stream = build_stream()
    encoded = (json.dumps(stream, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)
    return {"path": str(path), "sha256": _sha(encoded), "counts": stream["counts"]}
