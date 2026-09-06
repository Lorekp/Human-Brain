"""Third public continuation: interleaved corrections on sustained505-slot memory.

No new documents, model calls, prediction replay, or authored program execution.
Targets and concept/schedule annotations are parent-only development data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_tasks_v1 import PHASES, _editing, _reading
from experiments.r4_integrated.persistent_language_action_tasks_v1 import observation as observation
from experiments.r4_integrated.persistent_language_action_tasks_v2 import (
    HISTORY_SHA256 as FIRST_HISTORY_SHA,
)
from experiments.r4_integrated.persistent_language_action_tasks_v2 import (
    restore_baseline as restore_previous_baseline,
)

SCHEMA = "persistent-language-action-development-data-v3"
ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIRECTORY = ".pytest-tmp/persistent-language-action-v2-run"
HISTORY_SHA256 = {
    "manifest.json": "efac496ad0f5e8e6de982c5a6b9418621359996ba9064465e117a1e396d5e721",
    "attempts.jsonl": "7b21e2cacc6d39ff0a1b4ab7398e872195c5bed82c190e14a851dd2c85f93ccf",
    "teachers.jsonl": "10df7e75af48eb6ffc8b82b69640bce5534a601f25be5e16895a7bf4abdf7571",
    "ingests.jsonl": "040426d1390847bcb51b9035bc39212f00de14b96de835c8109e23f1e2f4eebc",
}
PRIOR_BANK_SHA256 = "de1eac1cc27a961ba1525c71ee90ff599eb5dff31b25ff8e4f1260bcd09304c8"
FINAL_STATE_SHA256 = "bdc2db89692fee5d49b305935020199c06e8cf0282324e5c1a15bfe6fd9db860"

# Six requirements span conditional factual use and compound output. Selection
# is independent of saved incorrect prefixes; every item is offered once.
CONCEPTS = (
    ("production_colour", b"blue", b"'blue'"),
    ("cold_backup_region", b"north", b"'north'"),
    ("debug_log_name", b"debug", b"'debug'"),
    ("archive_suffix", b".bak", b"'.bak'"),
    ("retry_delay_join", b"3/2", b"'3/2'"),
    ("testing_production_join", b"green/blue", b"'green/blue'"),
)
READING_REQUESTS = (
    (
        "Reply: production colour.",
        "Reply: cold backup region.",
        "Reply: debug log name.",
        "Reply: archive suffix.",
        "Reply: retries/delay, use /.",
        "Reply: tests/production, use /.",
    ),
    (
        "I need the production colour.",
        "I need the cold backup region.",
        "I need the debug log name.",
        "I need the archive suffix.",
        "I need retries/delay, use /.",
        "I need tests/production, use /.",
    ),
)
ACTION_REQUESTS = (
    (
        "app.py must show prod colour.",
        "app.py must show cold backup.",
        "app.py must show debug logs.",
        "app.py must show archive end.",
        "app.py: retries/delay using /",
        "app.py: tests/prod using /",
    ),
    (
        "Need prod colour from app.py",
        "Need cold backup from app.py",
        "Need debug logs from app.py",
        "Need archive suffix in app.py",
        "app.py: join retries/delay by /",
        "app.py: join tests/prod by /",
    ),
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _history_bytes(root: Path, name: str) -> bytes:
    raw = (root / HISTORY_DIRECTORY / name).read_bytes()
    if _sha(raw) != HISTORY_SHA256[name]:
        raise ValueError(f"second-session evidence changed: {name}")
    return raw


def _annotate(row, construction, structure, *, relation=False):
    row["split_annotation"] = {
        "instruction_construction": construction,
        "workspace_structure": structure,
        "requires_note_relation": relation,
    }
    row["workspace_lifecycle"] = "reset_at_episode_start_continue_within_retries"
    return row


def build_stream() -> dict:
    """Construct once after source review; read old JSON without running older builders."""
    previous = json.loads(_history_bytes(ROOT, "manifest.json"))
    first_raw = (ROOT / ".pytest-tmp/persistent-language-action-v1-run/manifest.json").read_bytes()
    if _sha(first_raw) != FIRST_HISTORY_SHA["manifest.json"]:
        raise ValueError("first-session public split changed")
    first = json.loads(first_raw)
    documents = previous["continuation"]["prior_documents"] + previous["documents"]
    if len(documents) != 18 or sum(len(bytes.fromhex(d["text_hex"])) + 1 for d in documents) != 505:
        raise ValueError("continue all18 documents and505 slots without re-ingestion")
    episodes = []
    for block in range(2):
        for concept, (name, answer, replacement) in enumerate(CONCEPTS):
            number = block * 6 + concept
            read = _annotate(
                _reading(
                    f"A{number:02d}-R", "acquisition", READING_REQUESTS[block][concept], answer
                ),
                "reply_colon_requirement" if block == 0 else "first_person_need",
                "none",
                relation=concept in (0, 1, 4, 5),
            )
            source = (
                b"def value():\n    parts = ['old']\n    return parts.pop()\n"
                if block == 0
                else b"DATA = {'answer':'old'}\ndef value():\n    return DATA['answer']\n"
            )
            edit = _annotate(
                _editing(
                    f"A{number:02d}-E",
                    "acquisition",
                    ACTION_REQUESTS[block][concept],
                    source,
                    b"'old'",
                    replacement,
                    answer + b"\n",
                ),
                "path_obligation" if block == 0 else "need_from_path",
                "assigned_list_pop" if block == 0 else "global_dictionary_return",
                relation=concept in (0, 1, 4, 5),
            )
            for row, kind in ((read, "R"), (edit, "E")):
                row["concept_id"] = name
                row["acquisition_block"] = block
                row["related_previous_episode_id"] = f"A{concept:02d}-{kind}" if block else None
                row["exposure_class"] = "new_instruction_on_previously_present_facts"
                episodes.append(row)

    fresh_readings = (
        ("Exclude tests: badge colour?", b"blue", "exclude_condition_question", True),
        ("Use /: debug then normal logs.", b"debug/audit", "delimiter_then_ordered_items", True),
        ("Reverse retries/delay; use /.", b"2/3", "reverse_order_instruction", True),
        ("Twice the default retry count?", b"6", "arithmetic_fact_question", False),
        ("Which region isn't cold backup?", b"west", "negative_exception_question", True),
        ("Append archive ending to logs.", b"audit.bak", "compose_name_and_suffix", True),
        (
            "Identify the required letter case for labels, giving the complete word "
            "and no surrounding explanation.",
            b"lowercase",
            "long_exact_scope_instruction",
            False,
        ),
        (
            "For an unusually large batch rather than an ordinary batch, "
            "state the number of jobs required.",
            b"8",
            "long_contrastive_quantity",
            True,
        ),
    )
    fresh_edits = (
        (
            "app.py: badge excluding tests",
            b"def value():\n    return f\"{'old'}\"\n",
            b"'old'",
            b"'blue'",
            b"blue",
            "path_excluded_condition",
            "fstring_expression",
            True,
        ),
        (
            "app.py: debug then normal, /",
            b"def value():\n    parts=['old','/audit']\n    return ''.join(parts)\n",
            b"'old'",
            b"'debug'",
            b"debug/audit",
            "path_ordered_items_delimiter",
            "assigned_list_join",
            True,
        ),
        (
            "app.py: delay/retries using /",
            b"def value():\n    left='2'\n    right='old'\n    return f'{left}/{right}'\n",
            b"'old'",
            b"'3'",
            b"2/3",
            "path_reversed_items",
            "fstring_local_bindings",
            True,
        ),
        (
            "app.py: twice default retries",
            b"def value():\n    return (lambda n:n*2)('old')\n",
            b"'old'",
            b"3",
            b"6",
            "path_arithmetic_requirement",
            "lambda_application",
            False,
        ),
        (
            "app.py: backup unless cold",
            b"def value():\n    return {'cold':'north','normal':'old'}['normal']\n",
            b"'old'",
            b"'west'",
            b"west",
            "path_unless_condition",
            "dictionary_lookup",
            True,
        ),
        (
            "app.py: logs plus archive end",
            b"def value():\n    return 'audit' + 'old'\n",
            b"'old'",
            b"'.bak'",
            b"audit.bak",
            "path_suffix_composition",
            "string_composition",
            True,
        ),
        (
            "Change app.py so its class-backed result follows the documented letter case "
            "for labels while retaining the caller.",
            b"class Config:\n    style='old'\ndef value():\n    return Config.style\n",
            b"'old'",
            b"'lowercase'",
            b"lowercase",
            "long_class_requirement",
            "class_attribute_return",
            False,
        ),
        (
            "Make the computation in app.py produce the documented large-batch total "
            "without changing main.py or its output formatting.",
            b"def value():\n    total=1\n    for n in (2,3):\n        total+=n\n    return total\n",
            b"1",
            b"3",
            b"8",
            "long_computed_requirement",
            "accumulating_loop",
            False,
        ),
    )
    for number, (reading, edit) in enumerate(zip(fresh_readings, fresh_edits, strict=True)):
        query, answer, form, relation = reading
        episodes.append(
            _annotate(
                _reading(f"F{number:02d}-R", "transfer", query, answer),
                form,
                "none",
                relation=relation,
            )
        )
        query, source, marker, replacement, answer, form, structure, relation = edit
        episodes.append(
            _annotate(
                _editing(
                    f"F{number:02d}-E",
                    "transfer",
                    query,
                    source,
                    marker,
                    replacement,
                    answer + b"\n",
                ),
                form,
                structure,
                relation=relation,
            )
        )

    recovery = (
        (
            "Restore app.py's joined counts",
            b"def value():\n    return '/'.join([3,2])\n",
            b"[3,2]",
            b"['3','2']",
            b"3/2",
        ),
        (
            "Restore app.py's Harbor label",
            b"def value():\n    return '{name}'.format(code='Q7')\n",
            b"{name}",
            b"{code}",
            b"Q7",
        ),
        (
            "Restore app.py's cold lookup",
            b"def value():\n    return dict([('cold','north')])['missing']\n",
            b"'missing'",
            b"'cold'",
            b"north",
        ),
        (
            "Restore app.py's archive name",
            b"def value():\n    return 'audit'.append('.bak')\n",
            b"'audit'.append",
            b"'audit'.__add__",
            b"audit.bak",
        ),
        (
            "Restore app.py's dry-run result",
            b"def value():\n    return {'writes':0}.writes\n",
            b".writes",
            b"['writes']",
            b"0",
        ),
        (
            "Restore app.py's normal log",
            b"def value():\n    return ('audit',)[1]\n",
            b"[1]",
            b"[0]",
            b"audit",
        ),
        (
            "Restore app.py's ticket result",
            b"def value():\n    return int('6',1)\n",
            b",1",
            b",10",
            b"6",
        ),
        (
            "Restore app.py's delay result",
            b"def value():\n    return 2 + None\n",
            b" + None",
            b"",
            b"2",
        ),
    )
    for number, (query, source, marker, replacement, answer) in enumerate(recovery):
        episodes.append(
            _annotate(
                _editing(
                    f"X{number:02d}", "recovery", query, source, marker, replacement, answer + b"\n"
                ),
                "restore_path_behavior",
                "runtime_error_composition",
            )
        )

    for old in previous["episodes"]:
        if old["phase"] != "retention":
            continue
        row = copy.deepcopy(old)
        row.update(
            teacher=None,
            append_document_ids=[],
            exposure_class="exposed_previous_session",
            prior_retention_id=old["episode_id"],
            prior_session=2,
        )
        episodes.append(row)
    previous_queries = first["episodes"] + previous["episodes"]
    acquisition = [e for e in episodes if e["phase"] == "acquisition"]
    previous_forms = {
        e["split_annotation"]["instruction_construction"] for e in previous_queries + acquisition
    }
    previous_structures = {
        e["split_annotation"]["workspace_structure"] for e in previous_queries + acquisition
    }
    overlap = []
    for row in episodes:
        if row["phase"] != "transfer":
            continue
        words = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower()))
        scores = []
        for old in previous_queries + acquisition:
            tokens = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(old["request_hex"]).lower()))
            scores.append(len(words & tokens) / len(words | tokens))
        overlap.append(
            {
                "episode_id": row["episode_id"],
                "maximum_prior_or_acquisition_token_jaccard": max(scores),
                "identical_previous_request": any(
                    row["request_hex"] == e["request_hex"] for e in previous_queries + acquisition
                ),
                "construction_label_seen": row["split_annotation"]["instruction_construction"]
                in previous_forms,
                "structure_label_seen": row["split_annotation"]["workspace_structure"]
                in previous_structures,
            }
        )
    teacher_raw_bytes = sum(
        len(bytes.fromhex(e["teacher"].get("target_hex", e["teacher"].get("replacement_hex"))))
        for e in acquisition
    )
    counts = {
        "episodes": len(episodes),
        "phase_episodes": {p: sum(e["phase"] == p for e in episodes) for p in PHASES},
        "reading_episodes": sum(e["kind"] == "reading" for e in episodes),
        "edit_episodes": sum(e["kind"] == "edit" for e in episodes),
        "teacher_offers": sum(e["teacher"] is not None for e in episodes),
        "unsupported_episodes": sum(not e["supported"] for e in episodes),
        "maximum_episode_attempts": sum(e["attempt_limit"] for e in episodes),
        "maximum_action_attempts": sum(e["attempt_limit"] for e in episodes if e["kind"] == "edit"),
        "document_raw_bytes": 0,
        "document_eos": 0,
        "initial_occupied_memory_slots": 505,
        "new_memory_slots": 0,
        "occupied_memory_slots": 505,
        "reader_teacher_raw_bytes": teacher_raw_bytes,
        "reader_teacher_eos": 24,
        "reader_teacher_symbols": teacher_raw_bytes + 24,
        "action_teacher_labels": 36,
        "maximum_active_output_clamps": 24,
    }
    if (
        counts["phase_episodes"] != dict(zip(PHASES, (24, 16, 8, 8), strict=True))
        or counts["teacher_offers"] != 24
        or counts["unsupported_episodes"] != 4
        or counts["maximum_episode_attempts"] != 72
        or counts["maximum_action_attempts"] != 48
        or counts["reader_teacher_raw_bytes"] != 148
    ):
        raise ValueError(f"third-session scope mismatch: {counts}")
    if len({e["episode_id"] for e in episodes}) != 56:
        raise ValueError("duplicate episode identity")
    if len({e["request_hex"] for e in acquisition}) != 24:
        raise ValueError("acquisition must use24 distinct public requests")
    for row in episodes:
        if row["append_document_ids"]:
            raise ValueError("third session never re-ingests prior facts or new documents")
        if len(row["files"]) > 4 or any(
            len(bytes.fromhex(f["content_hex"])) > 128 for f in row["files"]
        ):
            raise ValueError("workspace exceeds declared bounds")
        if row["teacher"] is not None and (row["phase"] != "acquisition" or not row["supported"]):
            raise ValueError("unsupported or out-of-phase correction")
        if row["kind"] == "edit":
            if len(bytes.fromhex(row["grade"]["canonical_proposal"]["replacement_hex"])) > 16:
                raise ValueError("replacement exceeds output capacity")
        elif any(len(bytes.fromhex(answer)) > 16 for answer in row["grade"]["accepted_answer_hex"]):
            raise ValueError("reading answer exceeds output capacity")
    if any(row["identical_previous_request"] for row in overlap):
        raise ValueError("fresh request repeats exposed wording")
    return {
        "schema": SCHEMA,
        "documents": [],
        "episodes": episodes,
        "counts": counts,
        "capacity": copy.deepcopy(previous["capacity"]),
        "provenance": {
            "kind": "agent_authored_open_development",
            "official_exam": False,
            "protected_evaluation": False,
            "source_sha256": _sha(Path(__file__).read_bytes()),
            "prior_manifest_sha256": HISTORY_SHA256["manifest.json"],
        },
        "continuation": {
            "prior_final_state_path": HISTORY_DIRECTORY + "/final-state.pt",
            "prior_final_state_sha256": FINAL_STATE_SHA256,
            "prior_bank_sha256": PRIOR_BANK_SHA256,
            "selected_neural_route": "online",
            "initial_memory_slots": 505,
            "prior_documents": copy.deepcopy(documents),
            "baseline_history_sha256": {"v1": dict(FIRST_HISTORY_SHA), "v2": dict(HISTORY_SHA256)},
            "baseline_prior_documents": 18,
            "baseline_prior_edit_demonstrations": 24,
            "baseline_predict_replays": 0,
            "initial_control_policy": "same_online_final_state; disable_current_session_learning",
            "memory_pressure": "sustained505slots; longer age with no capacity growth or newfacts",
        },
        "fresh_split": {
            "claim": "new requests/combinations over previously observed facts; no protected split",
            "lexical_overlap": overlap,
            "fully_withheld_wording_families_claimed": False,
            "annotation_labels_are_author_judgment": True,
            "over_capacity_ids": [e["episode_id"] for e in episodes if not e["supported"]],
            "retention_is_exposed": True,
            "acquisition_reuses_prior_fact_targets": True,
            "acquisition_concepts": [c[0] for c in CONCEPTS],
            "acquisition_schedule": (
                "six concepts reading/action, then six related forms reading/action"
            ),
            "related_first_response_gap_episodes": 12,
            "no_immediate_post_teacher_probe": True,
            "no_output_conditioned_task_selection": True,
        },
        "persistence": {
            "weights_and_shared_tape": "continue_actual_online_final_state",
            "workspace": "reset_at_each_episode_start; continue_only_within_recovery_retries",
            "shared_project_across_episodes": False,
        },
    }


def restore_baseline(root: Path):
    """Restore actual two-session reference history; never call predict or replay an action."""
    baseline, previous_restoration = restore_previous_baseline(root)
    raw = {name: _history_bytes(root, name) for name in HISTORY_SHA256}
    manifest = json.loads(raw["manifest.json"])
    documents = {d["document_id"]: bytes.fromhex(d["text_hex"]) for d in manifest["documents"]}
    attempts = {
        _sha(line): json.loads(line) for line in raw["attempts.jsonl"].splitlines(keepends=True)
    }
    ingests = [json.loads(line) for line in raw["ingests.jsonl"].splitlines()]
    observed = [d for row in ingests if row["route"] == "simple" for d in row["document_ids"]]
    if observed != list(documents):
        raise ValueError("second-session public append order changed")
    baseline.documents.extend(documents[d] for d in observed)
    bindings = []
    for line in raw["teachers.jsonl"].splitlines():
        row = json.loads(line)
        teacher = row["teacher"]
        if row["route"] != "simple" or teacher["kind"] != "edit":
            continue
        attempt = attempts[row["attempt_sha256"]]
        if (
            attempt["route"] != "simple"
            or attempt["phase"] != "acquisition"
            or attempt["attempt_index"] != 1
            or attempt["episode_id"] != row["episode_id"]
            or attempt["execution"] is None
            or teacher["operation"] != "SPLICE"
        ):
            raise ValueError("historical correction lacks its actual preceding action")
        source = next(
            bytes.fromhex(f["content_hex"])
            for f in attempt["before_files"]
            if f["path"] == teacher["path"]
        )
        start, end = teacher["start"], teacher["end"]
        if not 0 <= start <= end <= len(source):
            raise ValueError("historical correction span exceeds actual input")
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
    if (
        len(baseline.documents) != 18
        or len(baseline.demonstrations) != 24
        or sum(len(d) + 1 for d in baseline.documents) != 505
        or len(bindings) != 12
    ):
        raise ValueError(
            "reference continuation must restore18documents/24demonstrations/505symbols"
        )
    return baseline, {
        "schema": "persistent-language-action-baseline-restoration-v3",
        "input_sha256": {"v1": dict(FIRST_HISTORY_SHA), "v2": dict(HISTORY_SHA256)},
        "previous_restoration": previous_restoration,
        "documents": 18,
        "edit_demonstrations": 24,
        "document_raw_bytes": 487,
        "document_eos": 18,
        "new_teacher_offers": 0,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "reconstructed_sessions": 2,
        "second_session_bindings": bindings,
    }


def write_stream(path: Path) -> dict:
    stream = build_stream()
    raw = (json.dumps(stream, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return {"path": str(path), "sha256": _sha(raw), "counts": stream["counts"]}
