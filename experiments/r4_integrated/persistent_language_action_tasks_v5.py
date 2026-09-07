"""Interleaved public requirements, delayed recall and broader combinations.

This is author-only data construction. No builder, model, prediction or authored
program runs on import. Nominal solutions never enter a learner observation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_tasks_v4 import (
    HISTORY_SHA256 as V3_SHA,
)
from experiments.r4_integrated.persistent_language_action_tasks_v4 import (
    ROOT,
    V1_SHA,
    V2_SHA,
    _editing,
    _project,
    _reading,
)

SCHEMA = "persistent-language-action-development-data-v5"
HISTORY_DIRECTORY = ".pytest-tmp/persistent-language-action-v4-run"
HISTORY_SHA256 = {
    "manifest.json": "9d3399a1360ca9ddb4ba9361f06b70deb6063ddcf6461ed1aa414b0d54f4f191",
    "report.json": "5c63ac8bbbb2992a8dfe94d078801c06748208aa96d5a0ab0cd4711a661e32aa",
    "final-state.pt": "26cabfa5aa12ce362c51fcac2f57d5fdf05be6a5041a33f00acd47c05f618817",
}
PRIOR_BANK_SHA256 = "3122aed0c040ae086597cbdec6c0e2535a268cc62f7c72da05453ab88dd35466"
DOCUMENTS = (
    b"Quiet alerts use 'mute'; urgent alerts use 'ping'.",
    b"East cache lifetime is now '5' minutes; west remains '9'.",
    b"A report tag starts with 'log:' and uses a lowercase name.",
    b"Empty queues produce 'idle'; otherwise use the first job.",
    b"def initials(names):\n    return ''.join(name[0] for name in names if name)\n",
    b"def bounded(n, limit):\n    return max(0, min(n, limit))\n",
)
READ_REQUESTS = (
    "Read the retained alert note. State the quiet alert followed by the urgent alert, "
    "separated by /.",
    "Using the latest retained cache note, give the east lifetime and then the west lifetime, "
    "separated by /.",
    "Apply the retained report-tag rule to the name SKY and return the resulting tag alone.",
    "What result does the retained queue rule require when no jobs are waiting? Return only "
    "that result.",
)
READ_TARGETS = (b"mute/ping", b"5/9", b"log:sky", b"idle")
EDIT_TARGETS = (
    b"return 'ping' if urgent else 'mute'",
    b"return 5 if region == 'east' else 9",
    b"return 'log:' + name.lower()",
    b"return jobs[0] if jobs else 'idle'",
)
SOURCES = (
    b"def alert(urgent):\n    pass\n",
    b"def lifetime(region):\n    pass\n",
    b"def tag(name):\n    pass\n",
    b"def queue(jobs):\n    pass\n",
)
MAINS = (
    b"import app\nfor x in (True,False,'urgent','quiet',''):\n    print(app.alert(x))\n",
    b"import app\nfor x in ('east','west'):\n    print(app.lifetime(x))\n",
    b"import app\nfor x in ('SKY','',' Red '):\n    print(app.tag(x))\n",
    b"import app\nfor x in ([],['build'],['test','ship']):\n    print(app.queue(x))\n",
)
STDOUTS = (
    b"ping\nmute\nping\nping\nmute\n",
    b"5\n9\n",
    b"log:sky\nlog:\nlog: red \n",
    b"idle\nbuild\ntest\n",
)
EDIT_REQUESTS = (
    "Implement alert(urgent) in app.py using the retained quiet and urgent names; use ordinary "
    "truth testing and preserve main.py.",
    "Complete lifetime(region) in app.py using the latest retained east and west cache "
    "lifetimes; the driver supplies those two regions.",
    "Complete tag(name) in app.py using the retained tag prefix and lowercase rule, including "
    "an empty name; preserve the driver.",
    "Implement queue(jobs) in app.py from the retained queue rule, returning the first waiting "
    "job or the documented empty result.",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _prior_manifest(root):
    raw = (root / HISTORY_DIRECTORY / "manifest.json").read_bytes()
    if _sha(raw) != HISTORY_SHA256["manifest.json"]:
        raise ValueError("fourth-session manifest changed")
    return json.loads(raw)


def _earlier_request_history(root):
    """Authenticate JSON only; no baseline, model, builder or prediction replay."""
    episodes = []
    for version, hashes in ((1, V1_SHA), (2, V2_SHA), (3, V3_SHA)):
        raw = (
            root / f".pytest-tmp/persistent-language-action-v{version}-run/manifest.json"
        ).read_bytes()
        if _sha(raw) != hashes["manifest.json"]:
            raise ValueError("earlier exposed request history changed")
        episodes.extend(json.loads(raw)["episodes"])
    return episodes


def _encounter(row, family, kind, original=None):
    row["encounter"] = kind
    row["requirement_family"] = family
    row["same_session_origin_id"] = original
    if kind == "exact_delayed":
        row["exposure_class"] = "same_session_exposed_retention"
    elif kind == "changed_form":
        row["exposure_class"] = "changed_form_before_correction"
    return row


def _new_edit(identity, phase, query, signature, expression, driver, stdout, **kwargs):
    return _editing(
        identity,
        phase,
        query,
        (signature + "\n    pass\n").encode(),
        b"pass",
        ("return " + expression).encode(),
        stdout.encode(),
        driver.encode(),
        **kwargs,
    )


def _opportunities():
    rows = []
    for i in range(4):
        read = _reading(f"A{i:02d}-R", "acquisition", READ_REQUESTS[i], READ_TARGETS[i], teach=True)
        read["append_document_ids"] = [f"N{i + 18}"]
        edit = _editing(
            f"A{i:02d}-E",
            "acquisition",
            EDIT_REQUESTS[i],
            SOURCES[i],
            b"pass",
            EDIT_TARGETS[i],
            STDOUTS[i],
            MAINS[i],
            teach=True,
        )
        if i in (0, 2):
            _project(edit, "alert" if i == 0 else "tag")
        rows.extend(
            (_encounter(read, i, "corrected_first"), _encounter(edit, i, "corrected_first"))
        )
    changed_reads = (
        "For an alert comparison, list the urgent name before the quiet name from memory, with "
        "/ between them.",
        "A cache report lists west before east. Give the latest lifetimes in that order with / "
        "between the numbers.",
        "The next report is named RAIN. Following the remembered naming rule, what complete tag"
        " should it have?",
        "Jobs are waiting in the order ship, then test. Give the result required by the "
        "retained queue rule.",
    )
    changed_answers = (b"ping/mute", b"9/5", b"log:rain", b"ship")
    changed_edits = (
        (
            "Continue alert: the strings urgent and quiet now mean their named state, while "
            "True and False keep their earlier behavior; preserve main.py.",
            b"return 'ping' if urgent is True or urgent == 'urgent' else 'mute'",
            b"ping\nmute\nping\nmute\nmute\n",
        ),
        (
            "Fill in lifetimes(regions) in app.py: return the documented cache lifetime for "
            "every region in order, and an empty list for no regions.",
            "[5 if r == 'east' else 9 for r in regions]",
            "[9, 5, 9]\n[]\n",
        ),
        (
            "Continue the tag project: remove surrounding whitespace from each name before "
            "applying the same lowercase and prefix rule; preserve main.py.",
            b"return 'log:' + name.strip().lower()",
            b"log:sky\nlog:\nlog:red\n",
        ),
        (
            "Complete queues(groups) so each group follows the retained queue rule "
            "independently, keeping the group order; an empty collection stays empty.",
            "[jobs[0] if jobs else 'idle' for jobs in groups]",
            "['idle', 'ship', 'test']\n[]\n",
        ),
    )
    for i in range(4):
        read = _reading(
            f"A{i + 4:02d}-R",
            "acquisition",
            changed_reads[i],
            changed_answers[i],
            construction="reordered_or_new_input_question",
        )
        if i in (0, 2):
            read["append_document_ids"] = ["N22" if i == 0 else "N23"]
        query, replacement, stdout = changed_edits[i]
        if i in (0, 2):
            source = SOURCES[i].replace(b"pass", EDIT_TARGETS[i])
            edit = _editing(
                f"A{i + 4:02d}-E",
                "acquisition",
                query,
                source,
                EDIT_TARGETS[i],
                replacement,
                stdout,
                MAINS[i],
            )
            _project(edit, "alert" if i == 0 else "tag", f"A{i:02d}-E")
        else:
            signature = "def lifetimes(regions):" if i == 1 else "def queues(groups):"
            driver = (
                "import app\nprint(app.lifetimes(['west','east','west']))\n"
                "print(app.lifetimes([]))\n"
                if i == 1
                else "import app\nprint(app.queues([[],['ship'],['test','build']]))\n"
                "print(app.queues([]))\n"
            )
            edit = _new_edit(
                f"A{i + 4:02d}-E",
                "acquisition",
                query,
                signature,
                replacement,
                driver,
                stdout,
                structure="collection_transform",
            )
        rows.extend(
            (
                _encounter(read, i, "changed_form", f"A{i:02d}-R"),
                _encounter(edit, i, "changed_form", f"A{i:02d}-E"),
            )
        )
    for i in range(8):
        row = copy.deepcopy(rows[i])
        row.update(
            episode_id=f"A{8 + i // 2:02d}-{'R' if i % 2 == 0 else 'E'}",
            teacher=None,
            append_document_ids=[],
            project_id=None,
            predecessor_episode_id=None,
        )
        rows.append(_encounter(row, i // 2, "exact_delayed", rows[i]["episode_id"]))
    return rows


def _fresh():
    reading = (
        (
            "Without labels, pair a quiet alert with the east lifetime as alert:minutes, using "
            "the latest notes.",
            "mute:5",
        ),
        (
            "Suppose a report named WIND belongs to the west cache. Write its documented tag "
            "followed by / and that cache's lifetime.",
            "log:wind/9",
        ),
        (
            "The queue is empty and an urgent alert is due. Put the queue result before the "
            "alert name with a comma separating them.",
            "idle,ping",
        ),
        (
            "For the queued jobs BUILD then TEST, turn only the selected job into a report tag "
            "using the remembered rules.",
            "log:build",
        ),
        (
            "Imagine quiet east and urgent west alerts. Encode each as alert:minutes and join "
            "the two in that order with /.",
            "mute:5/ping:9",
        ),
        (
            "A report named SKY is the only queued job. Give its tag and the queue-rule result "
            "after that job is removed, separated by /.",
            "log:sky/idle",
        ),
        (
            "Which lifetime is shorter under the latest cache note? Reply with its region, then"
            " : and the numerical difference from the other lifetime.",
            "east:4",
        ),
        (
            "The jobs are empty, so use the documented empty-queue result as a report name. "
            "Return the resulting report tag.",
            "log:idle",
        ),
    )
    edits = (
        (
            "Complete status(urgent, region): join the documented alert and cache lifetime as "
            "alert:minutes; preserve the driver.",
            "def status(urgent, region):",
            "('ping' if urgent else 'mute') + ':' + str(5 if region == 'east' else 9)",
            "print(app.status(False,'east'))\nprint(app.status(True,'west'))\n",
            "mute:5\nping:9\n",
        ),
        (
            "For every supplied name, tags(names) should produce its retained report tag; "
            "preserve order, including an empty name and collection.",
            "def tags(names):",
            "['log:' + name.lower() for name in names]",
            "print(app.tags(['SUN','']))\nprint(app.tags([]))\n",
            "['log:sun', 'log:']\n[]\n",
        ),
        (
            "Implement head_tag(jobs): select the queue result from memory, then apply the "
            "report-tag rule to that result.",
            "def head_tag(jobs):",
            "'log:' + (jobs[0] if jobs else 'idle').lower()",
            "print(app.head_tag([]))\nprint(app.head_tag(['BUILD','TEST']))\n",
            "log:idle\nlog:build\n",
        ),
        (
            "Complete ttl_total(regions) by adding the latest documented lifetimes of all "
            "regions; no entries must total zero.",
            "def ttl_total(regions):",
            "sum(5 if r == 'east' else 9 for r in regions)",
            "print(app.ttl_total(['east','west','east']))\nprint(app.ttl_total([]))\n",
            "19\n0\n",
        ),
        (
            "Implement urgent_tags(items), where each item is a name and urgency flag: retain "
            "only urgent names and return their documented tags in order.",
            "def urgent_tags(items):",
            "['log:' + name.lower() for name, urgent in items if urgent]",
            "print(app.urgent_tags([('A',False),('B',True),('',True)]))\nprint(app.urgent_tags([]))\n",
            "['log:b', 'log:']\n[]\n",
        ),
        (
            "The function minutes(region,n) must clamp n to zero through that region's retained"
            " cache lifetime. Complete it without changing main.py.",
            "def minutes(region, n):",
            "max(0, min(n, 5 if region == 'east' else 9))",
            "print(app.minutes('east',-2))\nprint(app.minutes('east',8))\nprint(app.minutes('west',8))\n",
            "0\n5\n8\n",
        ),
        (
            "Implement summaries(groups): for each queue, return its documented selected result"
            " and urgent alert joined by :, preserving the group order.",
            "def summaries(groups):",
            "[(jobs[0] if jobs else 'idle') + ':ping' for jobs in groups]",
            "print(app.summaries([[],['ship','test']]))\nprint(app.summaries([]))\n",
            "['idle:ping', 'ship:ping']\n[]\n",
        ),
        (
            "Complete cache_tags(names,region): attach / and the latest regional lifetime to "
            "each report tag, including empty names and no names.",
            "def cache_tags(names, region):",
            "['log:' + name.lower() + '/' + str(5 if region == 'east' else 9) for name in names]",
            "print(app.cache_tags(['UP',''],'west'))\nprint(app.cache_tags([],'east'))\n",
            "['log:up/9', 'log:/9']\n[]\n",
        ),
    )
    rows = []
    for i, ((query, answer), (request, signature, expr, driver, stdout)) in enumerate(
        zip(reading, edits, strict=True)
    ):
        rows.append(
            _reading(
                f"F{i:02d}-R",
                "transfer",
                query,
                answer.encode(),
                construction=f"held_composition_question_{i}",
            )
        )
        rows.append(
            _new_edit(
                f"F{i:02d}-E",
                "transfer",
                request,
                signature,
                expr,
                "import app\n" + driver,
                stdout,
                construction=f"held_joint_program_requirement_{i}",
                structure="new_requirement_combination",
            )
        )
    return rows


def _recoveries():
    examples = (
        (
            "Repair alert_names(flags) so it returns the documented alert for every flag. Use "
            "actual execution feedback and preserve main.py.",
            "def alert_names(flags):",
            "['mute' for f in flags]",
            "['ping' if f else 'mute' for f in flags]",
            "print(app.alert_names([False,True]))\nprint(app.alert_names([]))\n",
            "['mute', 'ping']\n[]\n",
        ),
        (
            "Fix expired(age,region): a cache expires when age reaches its latest documented "
            "lifetime, including equality; preserve main.py.",
            "def expired(age, region):",
            "age > (5 if region == 'east' else 9)",
            "age >= (5 if region == 'east' else 9)",
            "print(app.expired(5,'east'))\nprint(app.expired(8,'west'))\nprint(app.expired(9,'west'))\n",
            "True\nFalse\nTrue\n",
        ),
        (
            "Repair tag(name) to apply the retained prefix and lowercase rule to every supplied"
            " name; use the actual diagnostic output.",
            "def tag(name):",
            "'log:' + name.lower",
            "'log:' + name.lower()",
            "print(app.tag('SKY'))\nprint(app.tag(''))\n",
            "log:sky\nlog:\n",
        ),
        (
            "The queue reader crashes on an empty list. Restore the complete retained queue "
            "rule while preserving ordinary nonempty behavior.",
            "def queue(jobs):",
            "jobs[0]",
            "jobs[0] if jobs else 'idle'",
            "print(app.queue([]))\nprint(app.queue(['ship','test']))\n",
            "idle\nship\n",
        ),
        (
            "Repair total(regions) to sum the latest cache lifetimes rather than their string "
            "forms; no regions should total zero.",
            "def total(regions):",
            "sum('5' if r == 'east' else '9' for r in regions)",
            "sum(5 if r == 'east' else 9 for r in regions)",
            "print(app.total(['east','west']))\nprint(app.total([]))\n",
            "14\n0\n",
        ),
        (
            "Fix first_tag(jobs) so it uses the queue rule and then the report-tag rule, even "
            "when there are no jobs. Preserve the driver.",
            "def first_tag(jobs):",
            "'log:' + jobs[0].lower()",
            "'log:' + (jobs[0] if jobs else 'idle').lower()",
            "print(app.first_tag([]))\nprint(app.first_tag(['SHIP','TEST']))\n",
            "log:idle\nlog:ship\n",
        ),
        (
            "Repair names(flags): output names must use each flag independently and stay in the"
            " original order; the retained alert note supplies the names.",
            "def names(flags):",
            "['ping' if flags else 'mute' for flag in flags]",
            "['ping' if flag else 'mute' for flag in flags]",
            "print(app.names([True,False,True]))\nprint(app.names([]))\n",
            "['ping', 'mute', 'ping']\n[]\n",
        ),
        (
            "Fix budget(n,region) so negative requests get zero and positive requests are "
            "capped by the latest regional lifetime, preserving smaller requests.",
            "def budget(n, region):",
            "min(n, 5 if region == 'east' else 9)",
            "max(0, min(n, 5 if region == 'east' else 9))",
            "print(app.budget(-1,'west'))\nprint(app.budget(7,'east'))\nprint(app.budget(3,'west'))\n",
            "0\n5\n3\n",
        ),
    )
    rows = []
    for i, (query, signature, before, after, driver, stdout) in enumerate(examples):
        rows.append(
            _editing(
                f"C{i:02d}-E",
                "recovery",
                query,
                (signature + "\n    return " + before + "\n").encode(),
                before.encode(),
                after.encode(),
                stdout.encode(),
                ("import app\n" + driver).encode(),
                construction="actual_feedback_repair",
                structure="faulty_function",
            )
        )
    return rows


def _validate_schedule(documents, episodes, previous):
    """Check author chronology and nominal references, never choose learner input."""
    prior_documents = previous["continuation"]["prior_documents"]
    if (
        len(prior_documents) != 18
        or sum(len(bytes.fromhex(row["text_hex"])) + 1 for row in prior_documents) != 505
    ):
        raise ValueError("prior factual memory must remain18documents/505slots")
    if len(documents) != 6:
        raise ValueError("six whole public documents are required")
    prior_ids = {row["document_id"] for row in prior_documents}
    for index, document in enumerate(documents):
        raw = bytes.fromhex(document["text_hex"])
        if (
            document["document_id"] != f"N{index + 18}"
            or document["document_id"] in prior_ids
            or not 0 < len(raw) <= (64 if index < 4 else 128)
        ):
            raise ValueError("new document identity or individual raw bound changed")
    appends = {f"A{i:02d}-R": [f"N{i + 18}"] for i in range(4)}
    appends.update({"A04-R": ["N22"], "A06-R": ["N23"]})
    corrections = [f"A{i:02d}-{kind}" for i in range(4) for kind in ("R", "E")]
    actual_corrections, actual_appends, seen = [], [], {}
    expected_projects = {
        "A00-E": ("alert", None),
        "A02-E": ("tag", None),
        "A04-E": ("alert", "A00-E"),
        "A06-E": ("tag", "A02-E"),
    }
    reset = "reset_at_episode_start_continue_within_retries"
    for ordinal, row in enumerate(episodes):
        identity = row["episode_id"]
        if row["append_document_ids"] != appends.get(identity, []):
            raise ValueError("public document append chronology changed")
        actual_appends.extend(row["append_document_ids"])
        if row["attempt_limit"] != (3 if row["phase"] == "recovery" else 1):
            raise ValueError("fixed attempt allocation changed")
        if row["teacher"] is not None:
            actual_corrections.append(identity)
            if row["phase"] != "acquisition" or row["teacher"]["kind"] != row["kind"]:
                raise ValueError("teacher escaped the fixed acquisition boundary")
            if row["kind"] == "reading":
                if row["teacher"]["target_hex"] not in row["grade"]["accepted_answer_hex"]:
                    raise ValueError("reading correction differs from authored answer")
            elif {key: value for key, value in row["teacher"].items() if key != "kind"} != row[
                "grade"
            ]["canonical_proposal"]:
                raise ValueError("edit correction differs from authored proposal")
        project, predecessor = expected_projects.get(identity, (None, None))
        if (row["project_id"], row["predecessor_episode_id"]) != (project, predecessor):
            raise ValueError("project identity or predecessor changed")
        if row["workspace_lifecycle"] != ("carry_actual_project" if predecessor else reset):
            raise ValueError("actual project carry/reset policy changed")
        if predecessor is not None:
            origin = seen[predecessor][1]
            if (
                origin["project_id"] != project
                or row["files"] != origin["grade"]["expected_after_files"]
                or row.get("nominal_files_scope")
                != "parent_only_intended_reference_never_materialize"
            ):
                raise ValueError(
                    "nominal project lineage differs from the prior authored after-state"
                )
        if row.get("encounter") == "exact_delayed":
            origin_index, origin = seen[row["same_session_origin_id"]]
            if (
                ordinal - origin_index - 1 != 15
                or row["teacher"] is not None
                or any(row[key] != origin[key] for key in ("kind", "request_hex", "files", "grade"))
            ):
                raise ValueError("exact delayed repeat or its fixed age changed")
        if identity in seen:
            raise ValueError("duplicate episode identity")
        seen[identity] = (ordinal, row)
    if actual_appends != [row["document_id"] for row in documents]:
        raise ValueError("documents must be appended exactly once in their public order")
    if actual_corrections != corrections:
        raise ValueError("the eight fixed first-encounter corrections changed")
    for phase, readings, edits in (
        ("acquisition", 12, 12),
        ("transfer", 8, 8),
        ("recovery", 0, 8),
        ("retention", 8, 8),
    ):
        if (
            sum(row["phase"] == phase and row["kind"] == "reading" for row in episodes) != readings
            or sum(row["phase"] == phase and row["kind"] == "edit" for row in episodes) != edits
        ):
            raise ValueError("phase reading/editing balance changed")
    old = [row for row in previous["episodes"] if row["phase"] == "retention"]
    prior_by_id = {row["episode_id"]: row for row in previous["episodes"]}
    expected_old = old + [
        prior_by_id[f"A{i:02d}-{kind}"] for i in (0, 1, 4, 5) for kind in ("R", "E")
    ]
    repeats = [row for row in episodes if row["phase"] == "retention"]
    for row, origin in zip(repeats, expected_old, strict=True):
        if row["previous_session_episode_id"] != origin["episode_id"] or any(
            row[key] != origin[key] for key in ("kind", "request_hex", "files", "grade")
        ):
            raise ValueError("exposed old regression differs from its authenticated episode")


def build_stream():
    """Prospective source-owned build; invokes no models or authored programs."""
    previous = _prior_manifest(ROOT)
    documents = [
        {"document_id": f"N{i + 18}", "text_hex": raw.hex()} for i, raw in enumerate(DOCUMENTS)
    ]
    episodes = _opportunities() + _fresh() + _recoveries()
    old = [e for e in previous["episodes"] if e["phase"] == "retention"]
    by_id = {e["episode_id"]: e for e in previous["episodes"]}
    # Coverage chosen before successor execution: ordered facts, exceptions,
    # bounds and structural composition. Keep failures, not a selected success set.
    selected = old + [by_id[f"A{i:02d}-{kind}"] for i in (0, 1, 4, 5) for kind in ("R", "E")]
    for i, original in enumerate(selected):
        row = copy.deepcopy(original)
        row.update(
            episode_id=f"R{i:02d}-{'R' if row['kind'] == 'reading' else 'E'}",
            phase="retention",
            teacher=None,
            append_document_ids=[],
            attempt_limit=1,
            project_id=None,
            predecessor_episode_id=None,
            workspace_lifecycle="reset_at_episode_start_continue_within_retries",
            exposure_class="previous_session_exposed_regression",
            previous_session_episode_id=original["episode_id"],
            prior_outcome_scope="v4_last_attempt",
            encounter="old_exact" if i < 8 else "v4_failed_coverage",
        )
        episodes.append(row)
    _validate_schedule(documents, episodes, previous)
    teachers = [e for e in episodes if e["teacher"] is not None]
    symbols = sum(
        len(bytes.fromhex(e["teacher"].get("target_hex", e["teacher"].get("replacement_hex")))) + 1
        for e in teachers
    )
    source_bytes = sum(map(len, DOCUMENTS))
    counts = {
        "episodes": len(episodes),
        "teacher_offers": len(teachers),
        "phase_episodes": {
            p: sum(e["phase"] == p for e in episodes)
            for p in ("acquisition", "transfer", "recovery", "retention")
        },
        "maximum_episode_attempts": sum(e["attempt_limit"] for e in episodes),
        "maximum_action_attempts": sum(e["attempt_limit"] for e in episodes if e["kind"] == "edit"),
        "reader_teacher_symbols": symbols,
        "reader_teacher_eos": 8,
        "reader_teacher_raw_bytes": symbols - 8,
        "action_teacher_labels": 12,
        "document_raw_bytes": source_bytes,
        "document_eos": 6,
        "initial_occupied_memory_slots": 505,
        "new_memory_slots": source_bytes + 6,
        "occupied_memory_slots": 511 + source_bytes,
        "source_next_byte_teacher_symbols": 0,
    }
    if counts["phase_episodes"] != {
        "acquisition": 24,
        "transfer": 16,
        "recovery": 8,
        "retention": 16,
    }:
        raise ValueError("v5 phase allocation changed")
    if (
        len(teachers) != 8
        or symbols > 200
        or source_bytes > 512
        or counts["maximum_episode_attempts"] != 80
        or counts["maximum_action_attempts"] != 52
        or len({e["episode_id"] for e in episodes}) != 64
    ):
        raise ValueError("v5 exposure or attempt budget changed")
    for row in episodes:
        if len(bytes.fromhex(row["request_hex"])) > 256 or len(row["files"]) > 4:
            raise ValueError("observation outside selected bounds")
        for file in row["files"]:
            if len(bytes.fromhex(file["content_hex"])) > 512 or len(file["path"].encode()) > 31:
                raise ValueError("file outside selected bounds")
        grade = row["grade"]
        if row["kind"] == "edit":
            proposal = grade["canonical_proposal"]
            before = {file["path"]: bytes.fromhex(file["content_hex"]) for file in row["files"]}
            after = {
                file["path"]: bytes.fromhex(file["content_hex"])
                for file in grade["expected_after_files"]
            }
            path, start, end = proposal["path"], proposal["start"], proposal["end"]
            if (
                proposal["operation"] != "SPLICE"
                or proposal["stop_reason"] != "eos"
                or path not in before
                or not 0 <= start <= end <= len(before[path])
                or set(before) != set(after)
            ):
                raise ValueError("invalid authored nominal splice")
            expected = dict(before)
            expected[path] = (
                before[path][:start]
                + bytes.fromhex(proposal["replacement_hex"])
                + before[path][end:]
            )
            if expected != after or any(
                after[path] != before[path] for path in grade["protected_paths"]
            ):
                raise ValueError("nominal splice or protected driver mismatch")
        targets = (
            grade["accepted_answer_hex"]
            if row["kind"] == "reading"
            else [grade["canonical_proposal"]["replacement_hex"]]
        )
        if any(len(bytes.fromhex(target)) > 128 for target in targets):
            raise ValueError("authored target outside selected output bound")
    overlap = []
    history = previous["episodes"] + [e for e in episodes if e["phase"] == "acquisition"]
    all_history = _earlier_request_history(ROOT) + history
    for row in episodes:
        if row["phase"] != "transfer":
            continue
        words = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower()))
        others = [
            set(re.findall(rb"[a-z0-9]+", bytes.fromhex(e["request_hex"]).lower())) for e in history
        ]
        all_others = [
            set(re.findall(rb"[a-z0-9]+", bytes.fromhex(e["request_hex"]).lower()))
            for e in all_history
        ]
        if any(row["request_hex"] == e["request_hex"] for e in all_history):
            raise ValueError("fresh request exactly repeats an exposed v1-v4/current request")
        overlap.append(
            {
                "episode_id": row["episode_id"],
                "maximum_v4_or_current_token_jaccard": max(
                    len(words & x) / len(words | x) for x in others
                ),
                "maximum_all_exposed_token_jaccard": max(
                    len(words & x) / len(words | x) for x in all_others
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "documents": documents,
        "episodes": episodes,
        "counts": counts,
        "capacity": {**previous["capacity"], "memory_slots": 1024},
        "provenance": {
            "kind": "agent_authored_open_development",
            "protected_evaluation": False,
            "official_exam": False,
            "source_sha256": _sha(Path(__file__).read_bytes()),
            "prior_manifest_sha256": HISTORY_SHA256["manifest.json"],
            "exposed_history_manifest_sha256": {
                "v1": V1_SHA["manifest.json"],
                "v2": V2_SHA["manifest.json"],
                "v3": V3_SHA["manifest.json"],
                "v4": HISTORY_SHA256["manifest.json"],
            },
        },
        "continuation": {
            "prior_final_state_path": HISTORY_DIRECTORY + "/final-state.pt",
            "prior_final_state_sha256": HISTORY_SHA256["final-state.pt"],
            "prior_bank_sha256": PRIOR_BANK_SHA256,
            "selected_neural_route": "online",
            "prior_documents": copy.deepcopy(previous["continuation"]["prior_documents"]),
            "initial_memory_slots": 505,
            "prior_assemblies": 235,
            "baseline_prior_documents": 18,
            "baseline_prior_edit_demonstrations": 42,
            "baseline_predict_replays": 0,
            "control_policy": "same prior; correction/router disabled; factual association "
            "writes enabled",
        },
        "fresh_split": {
            "lexical_overlap": overlap,
            "fully_withheld_wording_families_claimed": False,
            "annotation_labels_are_author_judgment": True,
            "correction_episode_ids": [e["episode_id"] for e in teachers],
            "claim": "unseen complete requests and structural combinations; exposed facts and "
            "code primitives",
            "overlap_history_scope": "authenticated v1-v4 manifests and current acquisition",
            "same_session_exact_intervening_attempts": 15,
            "same_session_changed_form_intervening_attempts": 7,
        },
        "persistence": {
            "weights_and_shared_tape": "continue_actual_v4_online_final_state",
            "workspace": "two actual project continuations; exact repeats reset; recovery "
            "carries actual files",
            "nominal_chain_files": "author_only; never substitute for actual predecessor files",
        },
    }


def write_stream(path: Path):
    stream = build_stream()
    raw = (json.dumps(stream, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return {"path": str(path), "sha256": _sha(raw), "counts": stream["counts"]}
