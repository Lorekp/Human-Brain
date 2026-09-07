"""Fourth public session: missing logic, long instructions and actual project carry.

Author-only targets and nominal project states never enter an observation.
Construction is explicit and side-effect free except authenticated JSON reads.
No builder, reference prediction or authored Python program runs on import.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_baseline_v2 import SimpleReference
from experiments.r4_integrated.persistent_language_action_tasks_v1 import PHASES
from experiments.r4_integrated.persistent_language_action_tasks_v2 import HISTORY_SHA256 as V1_SHA
from experiments.r4_integrated.persistent_language_action_tasks_v3 import HISTORY_SHA256 as V2_SHA
from experiments.r4_integrated.persistent_language_action_tasks_v3 import (
    restore_baseline as restore_previous_baseline,
)

SCHEMA = "persistent-language-action-development-data-v4"
ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIRECTORY = ".pytest-tmp/persistent-language-action-v3-run"
HISTORY_SHA256 = {
    "manifest.json": "fc83513a6d6accee744c5d03188bc9cec75022db3cd00ff5c49f561745aa3056",
    "attempts.jsonl": "14bab3ac2908b3977736cadfa5886db2786a73dc48c504630213fa60eabe72d9",
    "teachers.jsonl": "b785cb87167e33233a81bb7125e3d2f8cdaa0e60b1009fe9ea100b385099f08f",
    "ingests.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
PRIOR_BANK_SHA256 = "39e899d8ae5fa5bb741edab2681ccf214d6f1883fd258f96d043a02b3609022e"
FINAL_STATE_SHA256 = "692634352ea323ff7737425a4d5c4e69cd3b39f1bcb9a6c85443a6612e59d03d"

# Six declared teaching families. These bytes are parent-only corrections.
READ_TARGETS = (b"green/blue", b"north/west", b"debug/audit", b"report.bak", b"0/3", b"blue:3/2")
EDIT_TARGETS = (
    b"'green' if mode == 'test' else 'blue'",
    b"'north' if cold else 'west'",
    b"'debug' if debug else 'audit'",
    b"name + '.bak'",
    b"max(0, min(n, 3))",
    b"('green' if mode == 'test' else 'blue') + ':3/2'",
)
READ_REQUESTS = (
    (
        "Consult the retained mode notes: give testing colour first and "
        "production colour second, separated by /, with no explanation."
    ),
    (
        "Using the retained regional notes, give the cold-backup region "
        "and then the ordinary region; separate the two names with /."
    ),
    (
        "From the existing logging notes, name the debug log followed by "
        "the normal log; return only the names separated by /."
    ),
    (
        "Take the filename report and apply the archive suffix stated in "
        "the retained notes; reply with the complete resulting filename "
        "only."
    ),
    (
        "Use the retained retry limit: a negative request gets zero "
        "attempts and a request for nine is capped; give both counts "
        "using /."
    ),
    (
        "Combine the retained production colour, retry count and delay as "
        "colour:retries/delay; return that single line without extra "
        "words."
    ),
    (
        "I am preparing a mode comparison. Put production before testing, "
        "use a comma between their colours, and omit labels and "
        "commentary."
    ),
    (
        "For a deployment summary, put the ordinary region before the "
        "cold-backup region and separate them with a comma, not a slash."
    ),
    (
        "A reviewer wants normal logging listed before debugging; use the "
        "names from the notes and join them with a comma without labels."
    ),
    (
        "The next archive is named events rather than report. Apply the "
        "same documented suffix and return the filename, without "
        "quotation marks."
    ),
    (
        "A retry request of two stays below the documented cap, while a "
        "negative request cannot run; write their counts in that order "
        "using /."
    ),
    (
        "Compose a testing-mode summary using the retained colour, "
        "retries and delay; format the answer as colour:retries/delay "
        "only."
    ),
)
EDIT_REQUESTS = (
    (
        "Implement colour(mode) in app.py: testing mode uses the retained "
        "testing colour and every other mode uses the production colour; "
        "preserve main.py."
    ),
    (
        "Complete region(cold) in app.py so a true cold-backup flag "
        "chooses its documented region and a false flag chooses the "
        "ordinary region; keep main.py."
    ),
    (
        "Fill in log_name(debug) in app.py: use the debug log when the "
        "flag is true and the normal log otherwise, using the retained "
        "notes; leave main.py alone."
    ),
    (
        "Implement archive(name) in app.py by appending the documented "
        "archive suffix to the supplied name, including an empty name; do "
        "not alter main.py."
    ),
    (
        "Implement budget(n) in app.py: negative requests receive zero "
        "retries, small nonnegative requests keep their count, and larger "
        "requests use the noted cap."
    ),
    (
        "Complete summary(mode) in app.py to combine its mode colour with "
        "the documented retry count and delay as colour:retries/delay; "
        "preserve the driver."
    ),
    (
        "Continue the colour project: accept uppercase TEST as testing "
        "too, treat None as an ordinary mode, and preserve all previous "
        "lowercase behaviour and main.py."
    ),
    (
        "Continue the region project: the string cold should select the "
        "cold-backup region, but all other strings must remain ordinary; "
        "retain both Boolean cases."
    ),
    (
        "Implement names(flags) in app.py to return a list of documented "
        "log names, one for each Boolean flag in order; an empty input "
        "must give an empty list."
    ),
    (
        "Continue archive(name): do not append a second suffix when the "
        "name already ends with the documented archive suffix; retain "
        "empty and ordinary names."
    ),
    (
        "Complete total_delay(n) in app.py: first clamp retry requests to "
        "zero through the documented cap, then multiply that count by the "
        "documented delay."
    ),
    (
        "Implement settings(mode) in app.py to return a dictionary "
        "containing its mode colour and the retained retry count, "
        "preserving the keys in the supplied driver."
    ),
)
COLOUR_MAIN = (
    b"import app\nfor x in ('test','TEST','production','dry',None):\n    print(app.colour(x))\n"
)
REGION_MAIN = (
    b"import app\nfor x in (True,False,'cold','','warm',None):\n    print(app.region(x))\n"
)
ARCHIVE_MAIN = (
    b"import app\nfor x in ('report','','events.bak'"
    b",'.bak','report.BAK'):\n    print(app.archive("
    b"x))\n"
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _history_bytes(root, name):
    raw = (root / HISTORY_DIRECTORY / name).read_bytes()
    if _sha(raw) != HISTORY_SHA256[name]:
        raise ValueError(f"third-session evidence changed: {name}")
    return raw


def _files(source, main):
    return [
        {"path": "app.py", "content_hex": source.hex()},
        {"path": "main.py", "content_hex": main.hex()},
    ]


def _base(identity, phase, kind, query, construction, structure):
    return {
        "episode_id": identity,
        "phase": phase,
        "kind": kind,
        "request_hex": query.encode().hex(),
        "files": [],
        "append_document_ids": [],
        "attempt_limit": 3 if phase == "recovery" else 1,
        "supported": len(query.encode()) <= 256,
        "unsupported_dimension": None,
        "exposure_class": phase,
        "teacher": None,
        "grade": None,
        "workspace_lifecycle": "reset_at_episode_start_continue_within_retries",
        "project_id": None,
        "predecessor_episode_id": None,
        "split_annotation": {
            "instruction_construction": construction,
            "workspace_structure": structure,
        },
    }


def _reading(
    identity, phase, query, answer, *, teach=False, construction="ordered_fact_composition"
):
    row = _base(identity, phase, "reading", query, construction, "none")
    row["grade"] = {"kind": "exact_answer", "accepted_answer_hex": [answer.hex()]}
    if teach:
        row["teacher"] = {"kind": "reading", "target_hex": answer.hex()}
    return row


def _editing(
    identity,
    phase,
    query,
    source,
    marker,
    replacement,
    stdout,
    main,
    *,
    teach=False,
    construction="implement_conditional_requirement",
    structure="function_expression",
):
    if not marker or source.count(marker) != 1:
        raise ValueError("author correction requires one declared raw marker")
    start = source.index(marker)
    after = source[:start] + replacement + source[start + len(marker) :]
    row = _base(identity, phase, "edit", query, construction, structure)
    row["files"] = _files(source, main)
    proposal = {
        "operation": "SPLICE",
        "path": "app.py",
        "start": start,
        "end": start + len(marker),
        "replacement_hex": replacement.hex(),
        "stop_reason": "eos",
    }
    row["grade"] = {
        "kind": "executed_behavior",
        "expected_stdout_hex": stdout.hex(),
        "expected_returncode": 0,
        "expected_after_files": _files(after, main),
        "protected_paths": ["main.py"],
        "canonical_proposal": proposal,
    }
    if teach:
        row["teacher"] = {"kind": "edit", **proposal}
    return row


def observation(episode, *, actual_files=None):
    """Followers require actual route files; the nominal teacher solution is private."""
    if episode["workspace_lifecycle"] == "carry_actual_project" and actual_files is None:
        raise ValueError("project continuation requires actual predecessor files")
    return {
        "kind": episode["kind"],
        "request_hex": episode["request_hex"],
        "files": copy.deepcopy(episode["files"] if actual_files is None else actual_files),
    }


def _project(row, project, predecessor=None):
    row.update(project_id=project, predecessor_episode_id=predecessor)
    if predecessor is not None:
        row["workspace_lifecycle"] = "carry_actual_project"
        row["nominal_files_scope"] = "parent_only_intended_reference_never_materialize"
    return row


def build_stream():
    """One prospective pure build, only after root reviews this complete source."""
    previous = json.loads(_history_bytes(ROOT, "manifest.json"))
    documents = copy.deepcopy(previous["continuation"]["prior_documents"])
    if len(documents) != 18 or sum(len(bytes.fromhex(d["text_hex"])) + 1 for d in documents) != 505:
        raise ValueError("sustained whole memory must remain18documents/505slots")
    prior_episodes = []
    for number, hashes in ((1, V1_SHA), (2, V2_SHA), (3, HISTORY_SHA256)):
        raw = (
            ROOT / f".pytest-tmp/persistent-language-action-v{number}-run/manifest.json"
        ).read_bytes()
        if _sha(raw) != hashes["manifest.json"]:
            raise ValueError("exposed request history changed")
        prior_episodes.extend(json.loads(raw)["episodes"])
    first_edits = (
        (b"def colour(mode):\n    return None\n", COLOUR_MAIN, b"green\nblue\nblue\nblue\nblue\n"),
        (
            b"def region(cold):\n    return None\n",
            REGION_MAIN,
            b"north\nwest\nnorth\nwest\nnorth\nwest\n",
        ),
        (
            b"def log_name(debug):\n    return None\n",
            b"import app\nfor x in (True,False):\n    print(app.log_name(x))\n",
            b"debug\naudit\n",
        ),
        (
            b"def archive(name):\n    return None\n",
            ARCHIVE_MAIN,
            b"report.bak\n.bak\nevents.bak.bak\n.bak.bak\nreport.BAK.bak\n",
        ),
        (
            b"def budget(n):\n    return None\n",
            b"import app\nfor x in (-2,0,2,9):\n    print(app.budget(x))\n",
            b"0\n0\n2\n3\n",
        ),
        (
            b"def summary(mode):\n    return None\n",
            b"import app\nfor x in ('test','production'):\n    print(app.summary(x))\n",
            b"green:3/2\nblue:3/2\n",
        ),
    )
    episodes = []
    for i, (source, main, stdout) in enumerate(first_edits):
        episodes.append(
            _reading(f"A{i:02d}-R", "acquisition", READ_REQUESTS[i], READ_TARGETS[i], teach=True)
        )
        row = _editing(
            f"A{i:02d}-E",
            "acquisition",
            EDIT_REQUESTS[i],
            source,
            b"None",
            EDIT_TARGETS[i],
            stdout,
            main,
            teach=True,
        )
        if i in (0, 1, 3):
            _project(row, ("colour" if i == 0 else "region" if i == 1 else "archive"))
        episodes.append(row)
    nominal = {
        e["episode_id"]: bytes.fromhex(e["grade"]["expected_after_files"][0]["content_hex"])
        for e in episodes
        if e["kind"] == "edit"
    }
    second_edits = (
        (
            nominal["A00-E"],
            EDIT_TARGETS[0],
            b"'green' if str(mode).lower() == 'test' else 'blue'",
            COLOUR_MAIN,
            b"green\ngreen\nblue\nblue\nblue\n",
            "colour",
            "A00-E",
        ),
        (
            nominal["A01-E"],
            EDIT_TARGETS[1],
            b"'north' if cold is True or cold == 'cold' else 'west'",
            REGION_MAIN,
            b"north\nwest\nnorth\nwest\nwest\nwest\n",
            "region",
            "A01-E",
        ),
        (
            b"def names(flags):\n    return None\n",
            b"None",
            b"['debug' if flag else 'audit' for flag in flags]",
            b"import app\nfor x in ([True,False],[],[False]):\n    print(app.names(x))\n",
            b"['debug', 'audit']\n[]\n['audit']\n",
            None,
            None,
        ),
        (
            nominal["A03-E"],
            EDIT_TARGETS[3],
            b"name if name.endswith('.bak') else name + '.bak'",
            ARCHIVE_MAIN,
            b"report.bak\n.bak\nevents.bak\n.bak\nreport.BAK.bak\n",
            "archive",
            "A03-E",
        ),
        (
            b"def total_delay(n):\n    return None\n",
            b"None",
            b"2 * max(0, min(n, 3))",
            b"import app\nfor x in (-1,0,2,7):\n    print(app.total_delay(x))\n",
            b"0\n0\n4\n6\n",
            None,
            None,
        ),
        (
            b"def settings(mode):\n    return None\n",
            b"None",
            b"{'colour': 'green' if mode == 'test' else 'blue', 'retries': 3}",
            (
                b"import app\nfor x in ('test','production'):\n  "
                b"  y=app.settings(x)\n    print(y['colour'],y['"
                b"retries'])\n"
            ),
            b"green 3\nblue 3\n",
            None,
            None,
        ),
    )
    second_answers = (
        b"blue,green",
        b"west,north",
        b"audit,debug",
        b"events.bak",
        b"2/0",
        b"green:3/2",
    )
    for i, (source, marker, replacement, main, stdout, project, predecessor) in enumerate(
        second_edits, 6
    ):
        episodes.append(
            _reading(
                f"A{i:02d}-R",
                "acquisition",
                READ_REQUESTS[i],
                second_answers[i - 6],
                construction="related_requirement_without_correction",
            )
        )
        row = _editing(
            f"A{i:02d}-E",
            "acquisition",
            EDIT_REQUESTS[i],
            source,
            marker,
            replacement,
            stdout,
            main,
            construction="extend_requirement_without_correction",
            structure="continued_or_collection_function",
        )
        if project:
            _project(row, project, predecessor)
        episodes.append(row)
    nominal.update(
        {
            e["episode_id"]: bytes.fromhex(e["grade"]["expected_after_files"][0]["content_hex"])
            for e in episodes
            if e["kind"] == "edit"
        }
    )
    fresh_reading = (
        (
            (
                "Before sending the deployment note, replace its labels with "
                "values: production then cold backup, joined by @; give only the "
                "assembled text."
            ),
            b"blue@north",
        ),
        (
            (
                "Although debugging is disabled, testing is enabled. Report the "
                "resulting log name and colour in log:colour order, with neither "
                "quotes nor explanation."
            ),
            b"audit:green",
        ),
        (
            (
                "Instead of describing the policy, show the concrete filename for "
                "an archive named audit, followed by the ordinary region in "
                "parentheses."
            ),
            b"audit.bak(west)",
        ),
        (
            (
                "Unless a request is negative, its retry budget is capped by the "
                "notes. For negative two and positive eight, state the counts as "
                "low=number,high=number."
            ),
            b"low=0,high=3",
        ),
        (
            (
                "Having read both environment notes, write a compact mapping with "
                "testing first: test=colour;prod=colour. Supply the actual "
                "colours, not their labels."
            ),
            b"test=green;prod=blue",
        ),
        (
            (
                "Neither the debug name nor the cold-backup region is wanted "
                "here: join the normal log and ordinary region using a hyphen, "
                "and return only that pair."
            ),
            b"audit-west",
        ),
        (
            (
                "When the retained retry count is multiplied by the retained "
                "delay, what total delay results? Include the count and delay as "
                "count*delay=total."
            ),
            b"3*2=6",
        ),
        (
            (
                "Rather than naming just the suffix, construct archives for a and "
                "b, join the complete filenames with a comma, and put the testing "
                "colour after a colon."
            ),
            b"a.bak,b.bak:green",
        ),
    )
    fresh_edits = (
        (
            (
                "On the existing colour project, dry mode must now return an "
                "empty string; uppercase testing and all previous ordinary-mode "
                "cases must keep working."
            ),
            nominal["A06-E"],
            second_edits[0][2],
            b"'' if mode == 'dry' else ('green' if str(mode).lower() == 'test' else 'blue')",
            COLOUR_MAIN,
            b"green\ngreen\nblue\n\nblue\n",
            "colour",
            "A06-E",
        ),
        (
            (
                "The region project must now distinguish missing information: "
                "None returns unknown, while both Boolean flags and the explicit "
                "cold string retain their behavior."
            ),
            nominal["A07-E"],
            second_edits[1][2],
            (
                b"'unknown' if cold is None else ('north' if co"
                b"ld is True or cold == 'cold' else 'west')"
            ),
            REGION_MAIN,
            b"north\nwest\nnorth\nwest\nwest\nunknown\n",
            "region",
            "A07-E",
        ),
        (
            (
                "Implement enabled_logs(flags) so it returns debug names only for "
                "true flags, retaining their order; false flags and an empty "
                "input must contribute nothing."
            ),
            b"def enabled_logs(flags):\n    return None\n",
            b"None",
            b"['debug' for flag in flags if flag]",
            (
                b"import app\nfor x in ([False,True,True],[],[Fa"
                b"lse]):\n    print(app.enabled_logs(x))\n"
            ),
            b"['debug', 'debug']\n[]\n[]\n",
            None,
            None,
        ),
        (
            (
                "Without changing the archive project's driver, recognize an "
                "existing suffix regardless of letter case and preserve its "
                "original spelling; append only if absent."
            ),
            nominal["A09-E"],
            second_edits[3][2],
            b"name if name.lower().endswith('.bak') else name + '.bak'",
            ARCHIVE_MAIN,
            b"report.bak\n.bak\nevents.bak\n.bak\nreport.BAK\n",
            "archive",
            "A09-E",
        ),
        (
            (
                "Complete delays(counts) to return the total delay for each "
                "requested retry count, clamping negatives and excessive counts "
                "first; preserve the order of the list."
            ),
            b"def delays(counts):\n    return None\n",
            b"None",
            b"[2 * max(0, min(n, 3)) for n in counts]",
            b"import app\nfor x in ([-1,2,8],[],[0]):\n    print(app.delays(x))\n",
            b"[0, 4, 6]\n[]\n[0]\n",
            None,
            None,
        ),
        (
            (
                "Implement describe(mode,debug) to combine the mode colour with "
                "the log selected by the debug flag, separated by a colon; "
                "neither choice may ignore its input."
            ),
            b"def describe(mode,debug):\n    return None\n",
            b"None",
            b"('green' if mode == 'test' else 'blue') + ':' + ('debug' if debug else 'audit')",
            (
                b"import app\nfor m,d in (('test',False),('produ"
                b"ction',True),('test',True)):\n    print(app.de"
                b"scribe(m,d))\n"
            ),
            b"green:audit\nblue:debug\ngreen:debug\n",
            None,
            None,
        ),
        (
            (
                "Finish locations(flags) as a dictionary mapping each supplied "
                "name to its cold or ordinary region according to the Boolean "
                "flag, including the empty mapping."
            ),
            b"def locations(flags):\n    return None\n",
            b"None",
            b"{name: 'north' if cold else 'west' for name,cold in flags.items()}",
            (
                b"import app\nfor x in ({'a':True,'b':False},{})"
                b":\n    print(sorted(app.locations(x).items()))"
                b"\n"
            ),
            b"[('a', 'north'), ('b', 'west')]\n[]\n",
            None,
            None,
        ),
        (
            (
                "Complete archive_all(names) using the documented suffix: return "
                "complete archive names in order and preserve duplicates; an "
                "empty input must still be a list."
            ),
            b"def archive_all(names):\n    return None\n",
            b"None",
            b"[name + '.bak' for name in names]",
            b"import app\nfor x in (['a','a',''],[]):\n    print(app.archive_all(x))\n",
            b"['a.bak', 'a.bak', '.bak']\n[]\n",
            None,
            None,
        ),
    )
    for i, ((query, answer), edit) in enumerate(zip(fresh_reading, fresh_edits, strict=True)):
        episodes.append(
            _reading(f"F{i:02d}-R", "transfer", query, answer, construction=f"held_multiclause_{i}")
        )
        query, source, marker, replacement, main, stdout, project, predecessor = edit
        row = _editing(
            f"F{i:02d}-E",
            "transfer",
            query,
            source,
            marker,
            replacement,
            stdout,
            main,
            construction=f"held_composed_requirement_{i}",
            structure="continued_exception" if project else "new_filter_mapping_or_joint_condition",
        )
        if project:
            _project(row, project, predecessor)
        episodes.append(row)
    recovery = (
        (
            (
                "Repair app.py so numeric retry counts are displayed as "
                "count/delay using the retained delay; use the actual error "
                "output to diagnose the failure and preserve main.py."
            ),
            b"def value():\n    return '/'.join([3,2])\n",
            b"[3,2]",
            b"['3','2']",
            b"3/2\n",
        ),
        (
            (
                "The retained Harbor code belongs in the formatted result. Repair "
                "the mismatched field in app.py while keeping the provided public "
                "driver unchanged."
            ),
            b"def value():\n    return '{name}'.format(code='Q7')\n",
            b"{name}",
            b"{code}",
            b"Q7\n",
        ),
        (
            (
                "Resolve the failing lookup in app.py so the cold-backup region "
                "is returned from the mapping already present; inspect the actual "
                "diagnostic rather than changing main.py."
            ),
            b"def value():\n    return {'cold':'north'}['missing']\n",
            b"'missing'",
            b"'cold'",
            b"north\n",
        ),
        (
            (
                "The archive operation crashes instead of returning the audit "
                "filename with its documented suffix. Repair the operation in "
                "app.py without modifying the driver."
            ),
            b"def value():\n    return 'audit'.append('.bak')\n",
            b".append",
            b".__add__",
            b"audit.bak\n",
        ),
        (
            (
                "Repair the raw program so its dry-run write count can be read "
                "from the existing dictionary; preserve the zero value and all "
                "bytes of main.py."
            ),
            b"def value():\n    return {'writes':0}.writes\n",
            b".writes",
            b"['writes']",
            b"0\n",
        ),
        (
            (
                "The normal log name is already held in the one-element tuple, "
                "but app.py cannot return it. Correct the indexing fault and "
                "preserve the public driver."
            ),
            b"def value():\n    return ('audit',)[1]\n",
            b"[1]",
            b"[0]",
            b"audit\n",
        ),
        (
            (
                "The program should interpret the retained ticket count as an "
                "ordinary decimal number. Repair the invalid conversion in app.py "
                "and keep main.py unchanged."
            ),
            b"def value():\n    return int('6',1)\n",
            b",1",
            b",10",
            b"6\n",
        ),
        (
            (
                "Repair app.py to return the documented delay as a number: the "
                "accidental addition of a missing value must not change the delay "
                "or require a driver edit."
            ),
            b"def value():\n    return 2 + None\n",
            b" + None",
            b"",
            b"2\n",
        ),
    )
    for i, (query, source, marker, replacement, stdout) in enumerate(recovery):
        episodes.append(
            _editing(
                f"X{i:02d}",
                "recovery",
                query,
                source,
                marker,
                replacement,
                stdout,
                b"import app\nprint(app.value())\n",
                construction="diagnose_public_tool_feedback",
                structure="runtime_failure",
            )
        )
    for old in previous["episodes"]:
        if old["phase"] == "retention":
            row = copy.deepcopy(old)
            row.update(
                teacher=None,
                append_document_ids=[],
                exposure_class="exposed_previous_session",
                prior_retention_id=old["episode_id"],
                prior_session=3,
                project_id=None,
                predecessor_episode_id=None,
            )
            episodes.append(row)
    return _finish(episodes, documents, prior_episodes)


def _finish(episodes, documents, prior_episodes):
    acquisition = [e for e in episodes if e["phase"] == "acquisition"]
    history = prior_episodes + acquisition
    overlap = []
    for row in episodes:
        query = bytes.fromhex(row["request_hex"])
        if row["phase"] != "retention" and not 60 <= len(query) <= 200:
            raise ValueError("new instructions must be complete60..200byte requests")
        if not row["supported"] or len(query) > 256 or row["append_document_ids"]:
            raise ValueError("no request truncation, unsupported subset or document append")
        for file in row["files"]:
            if len(bytes.fromhex(file["content_hex"])) > 512 or len(file["path"].encode()) > 31:
                raise ValueError("workspace outside selected geometry")
        if len(row["files"]) > 4:
            raise ValueError("file count exceeds four")
        if row["kind"] == "edit":
            targets = [row["grade"]["canonical_proposal"]["replacement_hex"]]
            if len(bytes.fromhex(row["grade"]["expected_stdout_hex"])) > 512:
                raise ValueError("expected output exceeds actual capture bound")
        else:
            targets = row["grade"]["accepted_answer_hex"]
        if any(len(bytes.fromhex(target)) > 128 for target in targets):
            raise ValueError("target exceeds128rawbytes")
        if row["phase"] == "transfer":
            words = set(re.findall(rb"[a-z0-9]+", query.lower()))
            other = [
                set(re.findall(rb"[a-z0-9]+", bytes.fromhex(e["request_hex"]).lower()))
                for e in history
            ]
            identical = any(row["request_hex"] == e["request_hex"] for e in history)
            if identical:
                raise ValueError("fresh wording exactly repeats exposed request")
            overlap.append(
                {
                    "episode_id": row["episode_id"],
                    "identical_previous_request": identical,
                    "maximum_prior_or_acquisition_token_jaccard": max(
                        len(words & x) / len(words | x) for x in other
                    ),
                }
            )
    teachers = [e for e in episodes if e["teacher"] is not None]
    raw_teacher = sum(
        len(bytes.fromhex(e["teacher"].get("target_hex", e["teacher"].get("replacement_hex"))))
        for e in teachers
    )
    counts = {
        "episodes": len(episodes),
        "phase_episodes": {p: sum(e["phase"] == p for e in episodes) for p in PHASES},
        "reading_episodes": sum(e["kind"] == "reading" for e in episodes),
        "edit_episodes": sum(e["kind"] == "edit" for e in episodes),
        "teacher_offers": len(teachers),
        "unsupported_episodes": 0,
        "maximum_episode_attempts": sum(e["attempt_limit"] for e in episodes),
        "maximum_action_attempts": sum(e["attempt_limit"] for e in episodes if e["kind"] == "edit"),
        "document_raw_bytes": 0,
        "document_eos": 0,
        "initial_occupied_memory_slots": 505,
        "new_memory_slots": 0,
        "occupied_memory_slots": 505,
        "reader_teacher_raw_bytes": raw_teacher,
        "reader_teacher_eos": 12,
        "reader_teacher_symbols": raw_teacher + 12,
        "action_teacher_labels": 18,
        "project_chains": 3,
        "project_continuation_episodes": 6,
    }
    if (
        counts["phase_episodes"] != dict(zip(PHASES, (24, 16, 8, 8), strict=True))
        or len({e["episode_id"] for e in episodes}) != 56
    ):
        raise ValueError("fourth-session56episode scope changed")
    if [e["episode_id"] for e in teachers] != [
        f"A{i:02d}-{kind}" for i in range(6) for kind in ("R", "E")
    ]:
        raise ValueError("only the first six reading/edit pairs receive correction")
    if counts["maximum_episode_attempts"] != 72 or counts["maximum_action_attempts"] != 48:
        raise ValueError("actual attempt scope changed")
    by_id = {e["episode_id"]: e for e in episodes}
    for index, row in enumerate(episodes):
        predecessor = row["predecessor_episode_id"]
        if predecessor:
            previous = by_id[predecessor]
            if (
                predecessor not in {e["episode_id"] for e in episodes[:index]}
                or previous["project_id"] != row["project_id"]
                or previous["grade"]["expected_after_files"] != row["files"]
            ):
                raise ValueError("nominal chain identity/order mismatch")
            old_main = next(f for f in previous["files"] if f["path"] == "main.py")
            if old_main != next(f for f in row["files"] if f["path"] == "main.py"):
                raise ValueError("public project driver may not change between steps")
    return {
        "schema": SCHEMA,
        "documents": [],
        "episodes": episodes,
        "counts": counts,
        "capacity": {
            "request_bytes": 256,
            "workspace_files": 4,
            "file_bytes": 512,
            "path_bytes": 31,
            "output_bytes": 128,
            "stdout_bytes": 512,
            "stderr_bytes": 512,
            "memory_slots": 512,
        },
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
            "prior_documents": documents,
            "baseline_history_sha256": {
                "v1": dict(V1_SHA),
                "v2": dict(V2_SHA),
                "v3": dict(HISTORY_SHA256),
            },
            "baseline_prior_documents": 18,
            "baseline_prior_edit_demonstrations": 36,
            "baseline_predict_replays": 0,
            "initial_control_policy": "same_online_final_state; disable_current_session_learning",
            "memory_pressure": "sustained505slots; no newfacts or capacity growth",
        },
        "fresh_split": {
            "claim": (
                "held complete multiclause requests and structural combinations "
                "over exposed facts; public development"
            ),
            "lexical_overlap": overlap,
            "fully_withheld_wording_families_claimed": False,
            "annotation_labels_are_author_judgment": True,
            "retention_is_exposed": True,
            "over_capacity_ids": [],
            "acquisition_schedule": (
                "six interleaved reading/edit pairs corrected once; six related "
                "pairs with no correction"
            ),
            "correction_episode_ids": [e["episode_id"] for e in teachers],
            "chain_dependence": (
                "report fresh dependent cases separately; predecessor failures "
                "persist without repair"
            ),
        },
        "persistence": {
            "weights_and_shared_tape": "continue_actual_online_final_state",
            "workspace": (
                "three explicit actual-file project chains; other episodes reset; "
                "retries continue actual files"
            ),
            "shared_project_across_episodes": True,
            "nominal_chain_files": (
                "author-only intended state; never fallback or neural observation"
            ),
        },
    }


def restore_baseline(root: Path):
    """Restore actual three-session history, with no prediction or teacher replay."""
    baseline, previous_restoration = restore_previous_baseline(root)
    raw = {name: _history_bytes(root, name) for name in HISTORY_SHA256}
    if raw["ingests.jsonl"] or json.loads(raw["manifest.json"])["documents"]:
        raise ValueError("third session cannot supply additional documents")
    attempts = {
        _sha(line): json.loads(line) for line in raw["attempts.jsonl"].splitlines(keepends=True)
    }
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
            raise ValueError("historical demonstration lacks its preceding actual action")
        source = next(
            bytes.fromhex(f["content_hex"])
            for f in attempt["before_files"]
            if f["path"] == teacher["path"]
        )
        start, end = teacher["start"], teacher["end"]
        if not 0 <= start <= end <= len(source):
            raise ValueError("historical teacher span outside actual workspace")
        query = bytes.fromhex(attempt["request_hex"])
        baseline.demonstrations.append(
            {
                "query": query,
                "old": source[start:end],
                "replacement": bytes.fromhex(teacher["replacement_hex"]),
                "source_sha256": _sha(source),
                "query_sha256": _sha(query),
            }
        )
        bindings.append({"episode_id": row["episode_id"], "attempt_sha256": row["attempt_sha256"]})
    if (
        len(baseline.documents) != 18
        or len(baseline.demonstrations) != 36
        or len(bindings) != 12
        or sum(len(d) + 1 for d in baseline.documents) != 505
    ):
        raise ValueError("reference must restore18documents/36demonstrations/505symbols")
    return SimpleReference.from_restored(baseline), {
        "schema": "persistent-language-action-baseline-restoration-v4",
        "input_sha256": {"v1": dict(V1_SHA), "v2": dict(V2_SHA), "v3": dict(HISTORY_SHA256)},
        "previous_restoration": previous_restoration,
        "documents": 18,
        "edit_demonstrations": 36,
        "document_raw_bytes": 487,
        "document_eos": 18,
        "new_teacher_offers": 0,
        "prior_teacher_reading_answers_stored": 0,
        "prediction_replays": 0,
        "reconstructed_sessions": 3,
        "third_session_bindings": bindings,
    }


def write_stream(path: Path):
    stream = build_stream()
    raw = (json.dumps(stream, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return {"path": str(path), "sha256": _sha(raw), "counts": stream["counts"]}
