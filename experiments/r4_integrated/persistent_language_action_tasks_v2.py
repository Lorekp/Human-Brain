"""Second public session data; no model, executor, or task-program evaluation.

Continue the actual first session's memory and observed reference history.
Private author targets remain outside observation() and baseline.predict().
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_tasks_v1 import (
    PHASES,
    SimpleBaseline,
    _editing,
    _reading,
)
from experiments.r4_integrated.persistent_language_action_tasks_v1 import (
    observation as observation,
)

SCHEMA = "persistent-language-action-development-data-v2"
ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIRECTORY = ".pytest-tmp/persistent-language-action-v1-run"
HISTORY_SHA256 = {
    "manifest.json": "8ff7e338fb7649d8b230fc54507a1265a406cadf3f183c9c6585cbfa47b62ef8",
    "attempts.jsonl": "24b4096773ed6fa761715a53688800094ce793445068101789b22b4143ebb71a",
    "teachers.jsonl": "98585a318401b5aa77c56e0eb6a2258c121355e634496ec8fc7d294ed5763a6f",
    "ingests.jsonl": "3f6e6bcaa2bbd0054d57a23a982cf1406e21fa507fc36ecb52918aa973abb147",
}
FINAL_STATE_SHA256 = "e0c581ab907310e846a256fbd868c667e93c8d6cf96626c60b1177d3cdf3da7b"
INITIAL_OCCUPIED_SLOTS = 453
NEW_DOCUMENTS = ("Cold backups use 'north'.\n", "Debug logs use 'debug'.\n")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _history_bytes(root: Path, name: str) -> bytes:
    raw = (root / HISTORY_DIRECTORY / name).read_bytes()
    if _sha(raw) != HISTORY_SHA256[name]:
        raise ValueError(f"first-session evidence changed: {name}")
    return raw


def _annotate(row: dict, construction: str, structure: str, *, relation: bool = False) -> dict:
    row["split_annotation"] = {
        "instruction_construction": construction,
        "workspace_structure": structure,
        "requires_note_relation": relation,
    }
    row["workspace_lifecycle"] = "reset_at_episode_start_continue_within_retries"
    return row


def build_stream() -> dict:
    """One pure-data construction using authenticated old data, never the old builder."""
    old = json.loads(_history_bytes(ROOT, "manifest.json"))
    old_documents = old["documents"]
    if sum(len(bytes.fromhex(d["text_hex"])) + 1 for d in old_documents) != 453:
        raise ValueError("the continued stream must begin with all453 prior slots")
    documents = [
        {
            "document_id": f"N{16 + i:02d}",
            "text_hex": raw.encode().hex(),
            "sha256": _sha(raw.encode()),
        }
        for i, raw in enumerate(NEW_DOCUMENTS)
    ]
    # These are author targets, never runtime document/answer selection rules.
    acquisition = (
        ("Badge colour, please.", b"blue", "Change app.py: badge colour", b"'blue'"),
        ("Harbor code, please.", b"Q7", "Change app.py: Harbor code", b"'Q7'"),
        ("Default retry count, please.", b"3", "Change app.py: default retries", b"3"),
        ("Delay duration, please.", b"2", "Change app.py: delay duration", b"2"),
        ("Archive ending, please.", b".bak", "Change app.py: archive ending", b"'.bak'"),
        ("Ordinary log name, please.", b"audit", "Change app.py: ordinary logs", b"'audit'"),
        ("Debug log name, please.", b"debug", "Change app.py: debug logs", b"'debug'"),
        ("Dry-run write count, please.", b"0", "Change app.py: dry-run writes", b"0"),
        ("Usual batch size, please.", b"4", "Change app.py: usual batch size", b"4"),
        ("Cold backup region, please.", b"north", "Change app.py: cold backups", b"'north'"),
        ("Ticket lifetime, please.", b"6", "Change app.py: ticket lifetime", b"6"),
        ("Empty-input result, please.", b"0", "Change app.py: empty input", b"0"),
    )
    episodes = []
    for i, (query, answer, action_query, replacement) in enumerate(acquisition):
        append = ("N16",) if i == 0 else ("N17",) if i == 6 else ()
        episodes.append(
            _annotate(
                _reading(f"A{i:02d}-R", "acquisition", query, answer, append),
                "noun_phrase_please",
                "none",
            )
        )
        source = (
            b"def value():\n    result = 'old'\n    return result\n"
            if i % 2
            else b"RESULT = 'old'\ndef value():\n    return RESULT\n"
        )
        episodes.append(
            _annotate(
                _editing(
                    f"A{i:02d}-E",
                    "acquisition",
                    action_query,
                    source,
                    b"'old'",
                    replacement,
                    answer + b"\n",
                ),
                "change_path_colon_requirement",
                "local_binding_return" if i % 2 else "global_binding_return",
            )
        )

    fresh_readings = (
        ("Not testing: which badge?", b"blue", "negated_condition_question", True),
        ("Cold backup, not the usual?", b"north", "contrastive_ellipsis", True),
        ("Retries then delay; use /.", b"3/2", "ordered_join_instruction", True),
        ("Debug/normal logs; use /.", b"debug/audit", "paired_join_instruction", True),
        ("Only the archive ending.", b".bak", "exclusive_fragment_request", False),
        ("Before expiry, how many days?", b"6", "temporal_fronted_question", False),
        (
            "For a production deployment rather than a testing session, "
            "report only the badge colour.",
            b"blue",
            "long_contrastive_scope",
            True,
        ),
        (
            "Considering the note about unusually large batches, "
            "give the number of jobs per batch.",
            b"8",
            "long_evidence_qualification",
            False,
        ),
    )
    fresh_edits = (
        (
            "app.py: badge outside testing",
            b"def value():\n    a,b = ('old','spare')\n    return a\n",
            b"'old'",
            b"'blue'",
            b"blue",
            "path_condition_fragment",
            "tuple_unpacking",
            True,
        ),
        (
            "app.py: cold, not usual backup",
            b"def value():\n    return next(iter(['old']))\n",
            b"'old'",
            b"'north'",
            b"north",
            "path_contrast_fragment",
            "iterator_composition",
            True,
        ),
        (
            "app.py: retries/delay, use /",
            b"def value():\n    return '/'.join(('old','2'))\n",
            b"'old'",
            b"'3'",
            b"3/2",
            "path_ordered_join",
            "join_tuple",
            True,
        ),
        (
            "app.py: debug/normal, use /",
            b"def value():\n    return 'old' + '/audit'\n",
            b"'old'",
            b"'debug'",
            b"debug/audit",
            "path_paired_join",
            "string_composition",
            True,
        ),
        (
            "app.py: keep the archive lookup",
            b"def value():\n    return {'ext':'.bak'}.get('bad','old')\n",
            b"'bad'",
            b"'ext'",
            b".bak",
            "path_preserve_lookup",
            "dictionary_default",
            False,
        ),
        (
            "app.py: days before expiry",
            b"def value():\n    return (lambda x:x)('old')\n",
            b"'old'",
            b"6",
            b"6",
            "path_temporal_fragment",
            "lambda_application",
            False,
        ),
        (
            "Adjust app.py so a production deployment gets the documented badge colour, "
            "keeping main.py intact.",
            b"def value(flag=False):\n    return 'old' if not flag else 'green'\n",
            b"'old'",
            b"'blue'",
            b"blue",
            "long_conditional_edit",
            "default_argument_conditional",
            True,
        ),
        (
            "Have app.py report the documented number of jobs in a large batch "
            "without changing its caller.",
            b"def value(size=1):\n    return size\n",
            b"1",
            b"8",
            b"8",
            "long_caller_preserving_edit",
            "default_argument_return",
            False,
        ),
    )
    for i, ((q, answer, form, relation), edit) in enumerate(
        zip(fresh_readings, fresh_edits, strict=True)
    ):
        episodes.append(
            _annotate(
                _reading(f"F{i:02d}-R", "transfer", q, answer), form, "none", relation=relation
            )
        )
        q, source, marker, replacement, answer, form, structure, relation = edit
        episodes.append(
            _annotate(
                _editing(f"F{i:02d}-E", "transfer", q, source, marker, replacement, answer + b"\n"),
                form,
                structure,
                relation=relation,
            )
        )

    recovery = (
        (
            "Fix app.py's first-item access",
            b"def value():\n    return ['Q7'].pop(2)\n",
            b"2",
            b"0",
            b"Q7",
        ),
        (
            "Fix app.py's cold backup lookup",
            b"def value():\n    return {'cold':'north'}['absent']\n",
            b"'absent'",
            b"'cold'",
            b"north",
        ),
        (
            "Fix app.py's debug-log call",
            b"def name():\n    return 'debug'\ndef value():\n    return name.upper()\n",
            b"name.upper()",
            b"name()",
            b"debug",
        ),
        (
            "Fix app.py's retry conversion",
            b"def value():\n    return int('three')\n",
            b"'three'",
            b"'3'",
            b"3",
        ),
        (
            "Fix app.py's archive joining",
            b"def value():\n    return ''.join(['.bak',None])\n",
            b",None",
            b"",
            b".bak",
        ),
        (
            "Fix app.py's empty-input crash",
            b"def value():\n    return [0][1]\n",
            b"[1]",
            b"[0]",
            b"0",
        ),
        (
            "Fix app.py's expiry arithmetic",
            b"def value():\n    return '6' - 0\n",
            b"'6'",
            b"6",
            b"6",
        ),
        (
            "Fix app.py's delay formatting",
            b"def value():\n    return '{missing}'.format(delay=2)\n",
            b"{missing}",
            b"{delay}",
            b"2",
        ),
    )
    for i, (q, source, marker, replacement, answer) in enumerate(recovery):
        episodes.append(
            _annotate(
                _editing(f"X{i:02d}", "recovery", q, source, marker, replacement, answer + b"\n"),
                "fix_possessive_path_issue",
                "runtime_error_composition",
            )
        )
    prior_episodes = {row["episode_id"]: row for row in old["episodes"]}
    for i in (0, 3, 8, 11):
        for kind in ("R", "E"):
            row = copy.deepcopy(prior_episodes[f"A{i:02d}-{kind}"])
            row.update(
                episode_id=f"T{i:02d}-{kind}",
                phase="retention",
                teacher=None,
                append_document_ids=[],
                exposure_class="exposed_previous_session",
                prior_episode_id=f"A{i:02d}-{kind}",
            )
            episodes.append(row)

    acquisition_queries = [r["request_hex"] for r in episodes if r["phase"] == "acquisition"]
    old_queries = [r["request_hex"] for r in old["episodes"]]
    acquisition_forms = {
        r["split_annotation"]["instruction_construction"]
        for r in episodes
        if r["phase"] == "acquisition"
    }
    old_forms = {r["split_annotation"]["instruction_construction"] for r in old["episodes"]}
    old_structures = {r["split_annotation"]["workspace_structure"] for r in old["episodes"]}
    comparisons = []
    for row in episodes:
        if row["phase"] != "transfer":
            continue
        words = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower()))
        similarities = []
        for raw in acquisition_queries + old_queries:
            earlier = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(raw).lower()))
            similarities.append(len(words & earlier) / len(words | earlier))
        annotation = row["split_annotation"]
        comparisons.append(
            {
                "episode_id": row["episode_id"],
                "maximum_prior_or_acquisition_token_jaccard": max(similarities),
                "identical_previous_request": row["request_hex"]
                in acquisition_queries + old_queries,
                "construction_label_seen": annotation["instruction_construction"]
                in acquisition_forms | old_forms,
                "structure_label_seen": annotation["workspace_structure"] in old_structures,
            }
        )
    raw_bytes = sum(len(bytes.fromhex(d["text_hex"])) for d in documents)
    counts = {
        "episodes": len(episodes),
        "phase_episodes": {p: sum(e["phase"] == p for e in episodes) for p in PHASES},
        "reading_episodes": sum(e["kind"] == "reading" for e in episodes),
        "edit_episodes": sum(e["kind"] == "edit" for e in episodes),
        "teacher_offers": sum(e["teacher"] is not None for e in episodes),
        "unsupported_episodes": sum(not e["supported"] for e in episodes),
        "maximum_episode_attempts": sum(e["attempt_limit"] for e in episodes),
        "maximum_action_attempts": sum(e["attempt_limit"] for e in episodes if e["kind"] == "edit"),
        "document_raw_bytes": raw_bytes,
        "document_eos": len(documents),
        "initial_occupied_memory_slots": 453,
        "new_memory_slots": raw_bytes + len(documents),
        "occupied_memory_slots": 453 + raw_bytes + len(documents),
        "reader_teacher_symbols": sum(
            len(bytes.fromhex(e["teacher"].get("target_hex", e["teacher"].get("replacement_hex"))))
            + 1
            for e in episodes
            if e["teacher"] is not None
        ),
        "action_teacher_labels": 36,
    }
    if (
        counts["phase_episodes"] != dict(zip(PHASES, (24, 16, 8, 8), strict=True))
        or counts["teacher_offers"] != 24
        or counts["unsupported_episodes"] != 4
        or counts["occupied_memory_slots"] > 512
        or counts["maximum_episode_attempts"] != 72
        or counts["maximum_action_attempts"] != 48
    ):
        raise ValueError(f"continuation scope mismatch: {counts}")
    if [d for e in episodes for d in e["append_document_ids"]] != [
        d["document_id"] for d in documents
    ]:
        raise ValueError("each new document must append once in public order")
    for row in episodes:
        if len(row["files"]) > 4 or any(
            len(bytes.fromhex(f["content_hex"])) > 128 for f in row["files"]
        ):
            raise ValueError("workspace exceeds current interface")
        if row["teacher"] is not None and row["phase"] != "acquisition":
            raise ValueError("teacher outside acquisition")
        if row["kind"] == "edit":
            if len(bytes.fromhex(row["grade"]["canonical_proposal"]["replacement_hex"])) > 16:
                raise ValueError("replacement exceeds current interface")
        elif any(len(bytes.fromhex(a)) > 16 for a in row["grade"]["accepted_answer_hex"]):
            raise ValueError("reading target exceeds current interface")
    if any(c["identical_previous_request"] for c in comparisons):
        raise ValueError("fresh instruction exactly repeats an exposed request")
    source = Path(__file__).read_bytes()
    return {
        "schema": SCHEMA,
        "documents": documents,
        "episodes": episodes,
        "counts": counts,
        "capacity": copy.deepcopy(old["capacity"]),
        "provenance": {
            "kind": "agent_authored_open_development",
            "official_exam": False,
            "protected_evaluation": False,
            "source_sha256": _sha(source),
            "prior_manifest_sha256": HISTORY_SHA256["manifest.json"],
        },
        "continuation": {
            "prior_final_state_path": HISTORY_DIRECTORY + "/final-state.pt",
            "prior_final_state_sha256": FINAL_STATE_SHA256,
            "selected_neural_route": "online",
            "initial_memory_slots": 453,
            "prior_documents": copy.deepcopy(old_documents),
            "baseline_history_sha256": dict(HISTORY_SHA256),
            "baseline_prior_documents": 16,
            "baseline_prior_edit_demonstrations": 12,
            "baseline_predict_replays": 0,
            "initial_control_policy": "same_online_final_state; disable_current_session_learning",
            "new_parameter_policy": (
                "explicit_selected_successor_initialization; no_old_generation_parity"
            ),
        },
        "fresh_split": {
            "claim": "new constructions and combinations; shared facts/vocabulary remain exposed",
            "lexical_overlap": comparisons,
            "fully_withheld_wording_families_claimed": False,
            "annotation_labels_are_author_judgment": True,
            "over_capacity_ids": [e["episode_id"] for e in episodes if not e["supported"]],
            "retention_is_exposed": True,
            "acquisition_reuses_prior_fact_targets": True,
        },
        "persistence": {
            "weights_and_shared_tape": "continue_actual_online_final_state",
            "workspace": "reset_at_each_episode_start; continue_only_within_recovery_retries",
            "shared_project_across_episodes": False,
        },
    }


def restore_baseline(root: Path) -> tuple[SimpleBaseline, dict]:
    """Reconstruct only actual old reference observations/corrections, without prediction."""
    raw = {name: _history_bytes(root, name) for name in HISTORY_SHA256}
    manifest = json.loads(raw["manifest.json"])
    documents = {d["document_id"]: bytes.fromhex(d["text_hex"]) for d in manifest["documents"]}
    attempts = {
        _sha(line): json.loads(line) for line in raw["attempts.jsonl"].splitlines(keepends=True)
    }
    ingests = [json.loads(line) for line in raw["ingests.jsonl"].splitlines()]
    teachers = [json.loads(line) for line in raw["teachers.jsonl"].splitlines()]
    baseline = SimpleBaseline()
    observed = [d for r in ingests if r["route"] == "simple" for d in r["document_ids"]]
    if observed != list(documents):
        raise ValueError("reference history must contain the actual full ordered document stream")
    baseline.documents = [documents[d] for d in observed]
    bindings = []
    for row in teachers:
        if row["route"] != "simple" or row["teacher"]["kind"] != "edit":
            continue
        attempt = attempts[row["attempt_sha256"]]
        if (
            attempt["route"] != "simple"
            or attempt["phase"] != "acquisition"
            or attempt["attempt_index"] != 1
            or attempt["episode_id"] != row["episode_id"]
            or attempt["execution"] is None
        ):
            raise ValueError("demonstration lacks an actual prior supervised action")
        teacher = row["teacher"]
        if teacher["operation"] != "SPLICE":
            raise ValueError("unexpected historical demonstration operation")
        source = next(
            bytes.fromhex(f["content_hex"])
            for f in attempt["before_files"]
            if f["path"] == teacher["path"]
        )
        start, end = teacher["start"], teacher["end"]
        if not 0 <= start <= end <= len(source):
            raise ValueError("historical demonstration lies outside actual input")
        request = bytes.fromhex(attempt["request_hex"])
        baseline.demonstrations.append(
            {
                "query": request,
                "old": source[start:end],
                "replacement": bytes.fromhex(teacher["replacement_hex"]),
                "source_sha256": _sha(source),
                "query_sha256": _sha(request),
            }
        )
        bindings.append({"episode_id": row["episode_id"], "attempt_sha256": row["attempt_sha256"]})
    if len(baseline.documents) != 16 or len(baseline.demonstrations) != 12:
        raise ValueError("historical reference scope must be16 documents/12 editing demonstrations")
    return baseline, {
        "schema": "persistent-language-action-baseline-restoration-v2",
        "input_sha256": dict(HISTORY_SHA256),
        "documents": 16,
        "edit_demonstrations": 12,
        "document_raw_bytes": 437,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "new_teacher_offers": 0,
        "bindings": bindings,
    }


def write_stream(path: Path) -> dict:
    stream = build_stream()
    raw = (json.dumps(stream, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return {"path": str(path), "sha256": _sha(raw), "counts": stream["counts"]}
