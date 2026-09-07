"""Source-owned broader V6 tasks; no builder or program runs on import.

All answer witnesses, coverage labels and nominal project states are parent-only.
Only complete public requests/files/feedback and scheduled documents reach actors.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_tasks_v5 as previous_tasks
from experiments.r4_integrated.persistent_language_action_tasks_v4 import (
    ROOT,
    _editing,
    _project,
    _reading,
)

SCHEMA = "persistent-language-action-development-data-v6"
HISTORY_DIRECTORY = ".pytest-tmp/persistent-language-action-v5-run"
HISTORY_SHA256 = {
    "manifest.json": "de182acf644b18fed25948c5c7886006253bee8618a4e6ce647508f624b86fca",
    "report.json": "05307278505cbb3337c44e6bbc8d5168f898446c9f5f69c1486d233dfca8ad7c",
    "final-state.pt": "a141230e836d6da83e9914ae52f665d39d0d3e59089a60203b7a963ecd59ea51",
}
PRIOR_BANK_SHA256 = "c60ca50a1800f5431e1a20c8f1aa9c6bcd6e3c82e45ebfa72977735e52728d42"
DOCUMENTS = (
    b"2026-09-01 parcel dispatch: ordinary is 'hold'; priority is 'express'.",
    b"2026-09-01 parcel quota: every region has cap '4'; negative requests get zero.",
    b"Parcel labels use 'parcel:' plus the trimmed lowercase name.",
    b"Parcels in order: oak/east/3, ash/west/0, elm/east/7, oak/east/3.",
    b"Parcel reports keep input order. Separate fields with ':' and records with ';'.",
    b"A parcel is eligible exactly when its requested amount is positive.",
    b"2026-09-02 parcel dispatch: paused parcels use 'hold' even when priority is true.",
    b"2026-09-02 parcel quota replaces September 1: east cap '2', west cap '6'.",
    b"Museum tickets are unrelated: ordinary 'walk', priority 'wait', cap '9'.",
    b"def stable_unique(xs):\n    return list(dict.fromkeys(xs))\n",
    b"def optional_label(name,code):\n"
    b"    return 'parcel:'+name.strip().lower()+('@'+code.lower() if code else '')\n",
    b"def bounded_total(ns,cap):\n    return sum(max(0,min(n,cap)) for n in ns)\n",
)
PROJECTS = ("parcel_dispatch", "regional_quota", "parcel_naming")
SIGNATURES = (
    "def dispatch(priority, paused=False):",
    "def quota(n, region='east'):",
    "def label(name, code=''):",
)
EDIT_TARGETS = (
    b"return 'express' if priority else 'hold'",
    b"return 0 if n<0 else min(n,4)",
    b"return 'parcel:'+name.strip().lower()",
)
READ_TARGETS = (b"hold/express", b"4", b"parcel:oak")
READ_REQUESTS = (
    "For the parcel dispatch policy dated 2026-09-01, give the ordinary service followed "
    "by the priority service, with / between the two names and no extra words.",
    "Read the parcel quota dated 2026-09-01. State its cap for the east region as one number.",
    "Use the retained parcel naming rule for the name ' OAK '. Return the complete label alone.",
)
EDIT_REQUESTS = (
    "Implement dispatch(priority,paused=False) in app.py using the parcel policy dated "
    "2026-09-01. Use priority truth testing and ignore paused for this dated version. "
    "Preserve main.py.",
    "Implement quota(n,region='east') in app.py using the parcel quota dated 2026-09-01. "
    "Return zero for negative n and cap nonnegative n; this dated version ignores region. "
    "Preserve main.py.",
    "Implement label(name,code='') in app.py using the retained parcel naming rule. "
    "For this initial version ignore code, including on a blank name. Preserve main.py.",
)
MAINS = (
    b"import app\nfor p,s in ((True,False),(False,False),(True,True)):\n"
    b"    print(app.dispatch(p,s))\n",
    b"import app\nfor n,r in ((-1,'east'),(3,'west'),(7,'east'),(7,'west')):\n"
    b"    print(app.quota(n,r))\n",
    b"import app\nfor n,c in ((' OAK ','X'),('',''),('Elm','')):\n    print(app.label(n,c))\n",
)
STDOUTS = (b"express\nhold\nexpress\n", b"0\n3\n4\n4\n", b"parcel:oak\nparcel:\nparcel:elm\n")
CHANGED_TARGETS = (
    b"return 'express' if priority and not paused else 'hold'",
    b"return max(0,min(n,2 if region=='east' else 6))",
    b"return 'parcel:'+name.strip().lower()+('@'+code.lower() if code else '')",
)
CHANGED_STDOUTS = (
    b"express\nhold\nhold\n",
    b"0\n3\n2\n6\n",
    b"parcel:oak@x\nparcel:\nparcel:elm\n",
)
RESET = "reset_at_episode_start_continue_within_retries"
RETENTION_IDS = (
    "R00-R",
    "R01-E",
    "R02-R",
    "R03-E",
    "R04-R",
    "R05-E",
    "R06-R",
    "R07-E",
    "R08-R",
    "R09-E",
    "R10-R",
    "R11-E",
    "R12-R",
    "R13-E",
    "R14-R",
    "R15-E",
    "A11-R",
    "A11-E",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _history():
    raw = (ROOT / HISTORY_DIRECTORY / "manifest.json").read_bytes()
    if _sha(raw) != HISTORY_SHA256["manifest.json"]:
        raise ValueError("banked V5 manifest changed")
    return json.loads(raw)


def _encounter(row, family, kind, origin=None):
    row.update(
        encounter=kind,
        requirement_family=family,
        same_session_origin_id=origin,
        requirement_changed=kind == "changed_form",
    )
    if kind == "changed_form":
        row["exposure_class"] = "changed_language_and_requirement_uncorrected"
    elif kind == "exact_delayed":
        row["exposure_class"] = "same_session_exposed_retention"
    return row


def _fresh_edit(
    identity, query, signature, expression, driver, stdout, *, construction, helpers=None
):
    source = (signature + "\n    pass\n").encode()
    if helpers:
        source = b"import rules\n" + source
    row = _editing(
        identity,
        "transfer",
        query,
        source,
        b"pass",
        ("return " + expression).encode(),
        stdout.encode(),
        ("import app\n" + driver).encode(),
        construction=construction,
        structure="raw_multifile_function" if helpers else "function_expression",
    )
    if helpers:
        file = {"path": "rules.py", "content_hex": helpers.hex()}
        row["files"].append(copy.deepcopy(file))
        row["grade"]["expected_after_files"].append(file)
        row["grade"]["protected_paths"].append("rules.py")
    return row


def _first():
    rows = []
    for index in range(3):
        read = _reading(
            f"N{index:02d}-R",
            "acquisition",
            READ_REQUESTS[index],
            READ_TARGETS[index],
            teach=True,
            construction="dated_direct_request",
        )
        edit = _editing(
            f"N{index:02d}-E",
            "acquisition",
            EDIT_REQUESTS[index],
            (SIGNATURES[index] + "\n    pass\n").encode(),
            b"pass",
            EDIT_TARGETS[index],
            STDOUTS[index],
            MAINS[index],
            teach=True,
            construction="dated_direct_implementation",
        )
        _project(edit, PROJECTS[index])
        rows.extend(
            (
                _encounter(read, PROJECTS[index], "corrected_first"),
                _encounter(edit, PROJECTS[index], "corrected_first"),
            )
        )
    for index, row in enumerate(rows):
        row["append_document_ids"] = [f"N{24 + index}"]
    return rows


def _changed(first):
    reads = (
        (
            "A priority parcel is paused. Under the parcel dispatch revision dated 2026-09-02, "
            "which service does it get? Return that service alone.",
            b"hold",
        ),
        (
            "For the parcel quota revision dated 2026-09-02, list the west cap before the east "
            "cap with / between them. Ignore the unrelated museum policy.",
            b"6/2",
        ),
        (
            "Now include a nonempty tracking code after @ in the parcel label, lowercasing "
            "the code. What label does name ' OAK ' with code X require?",
            b"parcel:oak@x",
        ),
    )
    requests = (
        "Continue the actual dispatch project under the 2026-09-02 parcel revision. "
        "Paused parcels get the ordinary service even if priority is true; retain "
        "the unpaused behavior and main.py.",
        "Continue the actual quota project under the 2026-09-02 parcel revision. "
        "Use the current cap for each supplied east or west region, preserving "
        "negative and smaller requests and main.py.",
        "Continue the actual naming project: append @ and the lowercase tracking code "
        "only when code is nonempty. Keep the existing trimmed lowercase parcel label "
        "and preserve main.py.",
    )
    rows = []
    for index in range(3):
        read = _reading(
            f"N{index + 3:02d}-R",
            "acquisition",
            *reads[index],
            construction="changed_wording_and_requirement",
        )
        origin = first[2 * index + 1]
        source = bytes.fromhex(origin["grade"]["expected_after_files"][0]["content_hex"])
        edit = _editing(
            f"N{index + 3:02d}-E",
            "acquisition",
            requests[index],
            source,
            EDIT_TARGETS[index],
            CHANGED_TARGETS[index],
            CHANGED_STDOUTS[index],
            MAINS[index],
            construction="changed_wording_and_requirement",
        )
        _project(edit, PROJECTS[index], origin["episode_id"])
        rows.extend(
            (
                _encounter(read, PROJECTS[index], "changed_form", f"N{index:02d}-R"),
                _encounter(edit, PROJECTS[index], "changed_form", origin["episode_id"]),
            )
        )
    for index, row in enumerate(rows):
        row["append_document_ids"] = [f"N{30 + index}"]
    return rows


def _construction_cases():
    # Whole raw requests are checked against all exposed histories at build.
    # These names are parent-only author judgments, not a semantic routing input.
    cases = (
        (
            "exception_scope",
            "Use the September 1 parcel dispatch rule: a parcel stays ordinary unless its "
            "priority flag is true. List the services for false,true,false in that order with /.",
            b"hold/express/hold",
            "A parcel stays on the ordinary service unless its priority flag is true. "
            "Complete service(p) using the September 1 names; keep main.py.",
            "def service(p):",
            "'express' if p else 'hold'",
            "for p in (False,True,False):\n    print(app.service(p))\n",
            "hold\nexpress\nhold\n",
        ),
        (
            "conditional_inclusion",
            "Only parcel records whose requested amount is positive are eligible. From the "
            "retained collection, give eligible names, including repeated records, joined by ;.",
            b"oak;elm;oak",
            "Include a requested amount only when it is positive. Complete eligible(ns) "
            "without sorting or deduplicating the supplied sequence; preserve main.py.",
            "def eligible(ns):",
            "[n for n in ns if n>0]",
            "print(app.eligible([3,0,-1,7,3]))\nprint(app.eligible([]))\n",
            "[3, 7, 3]\n[]\n",
        ),
        (
            "inclusive_boundary",
            "For the September 1 parcel quota, amounts on either boundary stay inside "
            "the allowed interval. Give the bounded amounts for -1,0,4,5 with / separators.",
            b"0/0/4/4",
            "Complete inside(n): an amount is inside the September 1 parcel interval "
            "when it lies between zero and the cap, both endpoints included. Keep main.py.",
            "def inside(n):",
            "0<=n<=4",
            "for n in (-1,0,4,5):\n    print(app.inside(n))\n",
            "False\nTrue\nTrue\nFalse\n",
        ),
        (
            "stable_order",
            "Among the retained parcel records, discard zero-amount records but preserve "
            "the occurrence order and duplicates. Write their region names with ; separators.",
            b"east;east;east",
            "Discard empty names while preserving every remaining occurrence in its original "
            "position. Complete names(xs); do not sort the result or collapse repeats.",
            "def names(xs):",
            "[x for x in xs if x]",
            "print(app.names(['elm','','oak','elm']))\nprint(app.names([]))\n",
            "['elm', 'oak', 'elm']\n[]\n",
        ),
        (
            "requested_order",
            "The September 1 note names ordinary dispatch before priority. Reverse that "
            "presentation in your answer: priority first, ordinary second, separated by :.",
            b"express:hold",
            "Present the September 1 priority service before the ordinary service, reversing "
            "the note's order. Complete pair() with a colon between the names; keep main.py.",
            "def pair():",
            "'express:hold'",
            "print(app.pair())\n",
            "express:hold\n",
        ),
        (
            "nested_contrast",
            "A parcel uses the September 1 cap rather than the unrelated service's cap. "
            "If the request is negative use zero; otherwise retain it except above that cap. "
            "Give results for -2 and 9 with /.",
            b"0/4",
            "Complete allowance(n): use zero rather than a negative request; otherwise keep "
            "the request except where it exceeds the September 1 parcel cap. Preserve main.py.",
            "def allowance(n):",
            "0 if n<0 else (n if n<=4 else 4)",
            "for n in (-2,1,9):\n    print(app.allowance(n))\n",
            "0\n1\n4\n",
        ),
    )
    rows = []
    for index, (
        family,
        question,
        answer,
        request,
        signature,
        expression,
        driver,
        stdout,
    ) in enumerate(cases):
        read = _reading(f"F{index:02d}-R", "transfer", question, answer, construction=family)
        edit = _fresh_edit(
            f"F{index:02d}-E", request, signature, expression, driver, stdout, construction=family
        )
        for row in (read, edit):
            row["fresh_stratum"] = "instruction_construction"
            row["requirement_family"] = family
        rows.extend((read, edit))
    return rows


def _combination_cases(changed):
    reads = (
        (
            "Use the current paused-dispatch rule and parcel naming together: a priority "
            "unpaused parcel gets its service name labelled, while a paused parcel keeps "
            "the ordinary service alone. Give those two results separated by /.",
            b"parcel:express/hold",
        ),
        (
            "Under the September 2 parcel quota, bound request 7 for east and request 3 for "
            "west, then format region:amount records in that order with ; between them.",
            b"east:2;west:3",
        ),
        (
            "Use parcel naming with optional lowercase tracking after @, but substitute the "
            "ordinary dispatch name for a blank trimmed name. Format names ' OAK ' with X "
            "and '' with no code, separated by /.",
            b"parcel:oak@x/hold",
        ),
        (
            "Compare parcel quota history rather than museum policy. For west, report the "
            "September 1 cap followed by the September 2 cap, with / between them.",
            b"4/6",
        ),
        (
            "Bound the retained parcel amounts by the current east cap, preserving all "
            "four occurrences, then add those bounded amounts. Return the sum alone.",
            b"6",
        ),
        (
            "From the retained parcel collection, keep positive requests, form ordinary "
            "parcel labels, collapse repeated labels at their first occurrence, and join "
            "the result with ;.",
            b"parcel:oak;parcel:elm",
        ),
    )
    replacements = (
        b"return 'parcel:express' if priority and not paused else 'hold'",
        b"return region+':'+str(max(0,min(n,2 if region=='east' else 6)))",
        b"return ('parcel:'+name.strip().lower()+('@'+code.lower() if code else '')) "
        b"if name.strip() else 'hold'",
    )
    requests = (
        "Continue actual dispatch: apply the current paused rule, then turn an express "
        "service into its parcel label. Leave the ordinary service alone. Preserve main.py.",
        "Continue actual quota: first apply the current regional bounds, then return "
        "region:amount using the supplied region and the bounded integer. Preserve main.py.",
        "Continue actual naming: keep optional lowercase tracking for nonblank trimmed "
        "names, but replace a blank name with the ordinary parcel service. Preserve main.py.",
    )
    stdout = (
        b"parcel:express\nhold\nhold\n",
        b"east:0\nwest:3\neast:2\nwest:6\n",
        b"parcel:oak@x\nhold\nparcel:elm\n",
    )
    rows = []
    for index in range(6):
        identity = index + 6
        read = _reading(
            f"F{identity:02d}-R",
            "transfer",
            *reads[index],
            construction="multiple_public_requirements",
        )
        if index < 3:
            origin = changed[2 * index + 1]
            source = bytes.fromhex(origin["grade"]["expected_after_files"][0]["content_hex"])
            edit = _editing(
                f"F{identity:02d}-E",
                "transfer",
                requests[index],
                source,
                CHANGED_TARGETS[index],
                replacements[index],
                stdout[index],
                MAINS[index],
                construction="actual_project_composition",
            )
            _project(edit, PROJECTS[index], origin["episode_id"])
        elif index == 3:
            edit = _fresh_edit(
                f"F{identity:02d}-E",
                "Compare both dated parcel quota policies for each region. "
                "Complete history(region) "
                "as old/current using /; do not replace the old date with the new one.",
                "def history(region):",
                "'4/'+str(2 if region=='east' else 6)",
                "print(app.history('east'))\nprint(app.history('west'))\n",
                "4/2\n4/6\n",
                construction="dated_revision_comparison",
            )
        elif index == 4:
            edit = _fresh_edit(
                f"F{identity:02d}-E",
                "Complete total(ns): bound every amount independently using the current east "
                "parcel cap, then sum the results. Empty input totals zero; preserve main.py.",
                "def total(ns):",
                "sum(max(0,min(n,2)) for n in ns)",
                "print(app.total([-1,1,7]))\nprint(app.total([]))\nprint(app.total([2,2]))\n",
                "3\n0\n4\n",
                construction="bounds_and_aggregation",
            )
        else:
            edit = _fresh_edit(
                f"F{identity:02d}-E",
                "Complete report(rows): omit blank trimmed names, use rules.optional_label "
                "for each remaining (name,code), retain each label's first occurrence via "
                "rules.stable_unique, and join labels with ;. Keep both other files unchanged.",
                "def report(rows):",
                "';'.join(rules.stable_unique(rules.optional_label(n,c) for n,c in rows "
                "if n.strip()))",
                "print(app.report([(' OAK ','X'),('',''),(' OAK ','X'),('Elm','')]))\n"
                "print(app.report([]))\n",
                "parcel:oak@x;parcel:elm\n\n",
                construction="optional_fields_stable_deduplication_and_format",
                helpers=DOCUMENTS[9] + DOCUMENTS[10],
            )
        for row in (read, edit):
            row["fresh_stratum"] = "structural_combination"
            row["dependent_project"] = row["predecessor_episode_id"] is not None
        rows.extend((read, edit))
    return rows


def _recoveries():
    examples = (
        (
            "undefined_identifier",
            "Fix quota(n) for the current east parcel cap, including "
            "negative requests; use actual feedback and preserve main.py.",
            "def quota(n):",
            "min(amount,2)",
            "max(0,min(n,2))",
            "for n in (-1,4,1):\n    print(app.quota(n))\n",
            "0\n2\n1\n",
            None,
        ),
        (
            "malformed_expression",
            "Repair the expression in quota(n) so the current east "
            "parcel interval applies to every input; preserve the supplied driver.",
            "def quota(n):",
            "max(0,min(n,2)",
            "max(0,min(n,2))",
            "for n in (-1,2,5):\n    print(app.quota(n))\n",
            "0\n2\n2\n",
            None,
        ),
        (
            "wrong_attribute",
            "Repair clean(name): trim outer whitespace and lowercase the "
            "name, including a blank string. Use the actual diagnostic and keep main.py.",
            "def clean(name):",
            "name.trim().lower()",
            "name.strip().lower()",
            "print(app.clean(' OAK '))\nprint(app.clean(''))\n",
            "oak\n\n",
            None,
        ),
        (
            "empty_collection",
            "Repair first(items): keep the first supplied item, but return "
            "the ordinary parcel service when there are none. Preserve main.py.",
            "def first(items):",
            "items[0]",
            "items[0] if items else 'hold'",
            "print(app.first([]))\nprint(app.first(['oak','elm']))\n",
            "hold\noak\n",
            None,
        ),
        (
            "exception_precedence",
            "Restore the September 2 parcel dispatch rule: paused "
            "overrides priority, and an unpaused ordinary parcel still stays ordinary.",
            "def dispatch(priority,paused):",
            "'express' if priority or not paused else 'hold'",
            "'express' if priority and not paused else 'hold'",
            "for p,s in ((True,True),(False,False),(True,False)):\n    print(app.dispatch(p,s))\n",
            "hold\nhold\nexpress\n",
            None,
        ),
        (
            "cap_equality",
            "Repair cap(n): the current east parcel cap includes equality. "
            "Negative requests get zero and values above the cap get the cap.",
            "def cap(n):",
            "0 if n==2 else min(n,2)",
            "max(0,min(n,2))",
            "for n in (2,3,-1):\n    print(app.cap(n))\n",
            "2\n2\n0\n",
            None,
        ),
        (
            "stable_order",
            "Repair unique(xs): retain each item's first occurrence, in that "
            "order, instead of sorting the input; preserve empty-input behavior and main.py.",
            "def unique(xs):",
            "sorted(set(xs))",
            "list(dict.fromkeys(xs))",
            "print(app.unique(['b','a','b']))\nprint(app.unique([]))\n",
            "['b', 'a']\n[]\n",
            None,
        ),
        (
            "case_and_whitespace",
            "Repair label(name) to follow the retained parcel prefix, "
            "trimming and lowercase rules, including a blank name; preserve main.py.",
            "def label(name):",
            "'parcel:'+name.lower()",
            "'parcel:'+name.strip().lower()",
            "print(app.label(' OAK '))\nprint(app.label(''))\n",
            "parcel:oak\nparcel:\n",
            None,
        ),
        (
            "helper_interface",
            "Repair quota(n) using the actual rules.py bounded(n,cap) "
            "interface and the current east cap; preserve both other files.",
            "def quota(n):",
            "rules.limit(n)",
            "rules.bounded(n,2)",
            "print(app.quota(-1))\nprint(app.quota(8))\n",
            "0\n2\n",
            b"def bounded(n,cap):\n    return max(0,min(n,cap))\n",
        ),
        (
            "cross_file_format",
            "Repair report(xs): use the names supplied by rules.tags and "
            "separate records with ; rather than the field separator. "
            "Preserve rules.py and main.py.",
            "def report(xs):",
            "':'.join(rules.tags(xs))",
            "';'.join(rules.tags(xs))",
            "print(app.report(['OAK','elm']))\nprint(app.report([]))\n",
            "parcel:oak;parcel:elm\n\n",
            b"def tags(xs):\n    return ['parcel:'+x.lower() for x in xs]\n",
        ),
        (
            "cross_file_aggregation",
            "Repair total(ns): sum the numeric bounds from rules.py "
            "before any formatting, using the current east cap. Preserve both other files.",
            "def total(ns):",
            "sum(str(rules.bounded(n,2)) for n in ns)",
            "sum(rules.bounded(n,2) for n in ns)",
            "print(app.total([-1,1,7]))\nprint(app.total([]))\n",
            "3\n0\n",
            b"def bounded(n,cap):\n    return max(0,min(n,cap))\n",
        ),
        (
            "cross_file_argument_order",
            "Repair label(name,code): call the actual rules.optional "
            "interface in its declared order and retain blank-code behavior. Preserve rules.py "
            "and main.py.",
            "def label(name,code):",
            "rules.optional(code,name)",
            "rules.optional(name,code)",
            "print(app.label(' OAK ','X'))\nprint(app.label('',''))\n",
            "parcel:oak@x\nparcel:\n",
            b"def optional(name,code):\n"
            b"    return 'parcel:'+name.strip().lower()+('@'+code.lower() if code else '')\n",
        ),
    )
    rows = []
    for index, (fault, request, signature, before, after, driver, stdout, helpers) in enumerate(
        examples
    ):
        source = (signature + "\n    return " + before + "\n").encode()
        if helpers:
            source = b"import rules\n" + source
        row = _editing(
            f"C{index:02d}-E",
            "recovery",
            request,
            source,
            before.encode(),
            after.encode(),
            stdout.encode(),
            ("import app\n" + driver).encode(),
            construction="actual_feedback_repair",
            structure=fault,
        )
        if helpers:
            file = {"path": "rules.py", "content_hex": helpers.hex()}
            row["files"].append(copy.deepcopy(file))
            row["grade"]["expected_after_files"].append(file)
            row["grade"]["protected_paths"].append("rules.py")
        rows.append(row)
    return rows


def _retention(previous):
    by_id = {row["episode_id"]: row for row in previous["episodes"]}
    rows = []
    for index, identity in enumerate(RETENTION_IDS):
        original = by_id[identity]
        row = copy.deepcopy(original)
        row.update(
            episode_id=f"R{index:02d}-{'R' if row['kind'] == 'reading' else 'E'}",
            phase="retention",
            teacher=None,
            append_document_ids=[],
            attempt_limit=1,
            project_id=None,
            predecessor_episode_id=None,
            workspace_lifecycle=RESET,
            exposure_class="previous_session_exposed_regression",
            previous_session_episode_id=identity,
            prior_outcome_scope="v5_last_attempt",
            encounter="old_exact",
            requirement_changed=False,
            same_session_origin_id=None,
        )
        rows.append(row)
    return rows


def _repeats(first):
    rows = []
    for index, original in enumerate(first):
        row = copy.deepcopy(original)
        row.update(
            episode_id=f"N{index // 2 + 6:02d}-{'R' if index % 2 == 0 else 'E'}",
            teacher=None,
            append_document_ids=[],
            project_id=None,
            predecessor_episode_id=None,
            workspace_lifecycle=RESET,
        )
        rows.append(
            _encounter(row, original["requirement_family"], "exact_delayed", original["episode_id"])
        )
    return rows


def _unsupported_dimensions(row):
    dimensions = []
    if len(bytes.fromhex(row["request_hex"])) > 256:
        dimensions.append("request_bytes")
    if len(row["files"]) > 4:
        dimensions.append("files")
    if any(
        len(file["path"].encode()) > 31 or len(bytes.fromhex(file["content_hex"])) > 512
        for file in row["files"]
    ):
        dimensions.append("file_bytes_or_path")
    return dimensions


def _validate_rows(episodes):
    """Parse answer witnesses, but never run or materialize them."""
    for row in episodes:
        dimensions = _unsupported_dimensions(row)
        if len({file["path"] for file in row["files"]}) != len(row["files"]):
            raise ValueError("duplicate authored file path")
        for file in row["files"]:
            path = file["path"]
            if not path or path in (".", "..") or any(c in path for c in "/\\:"):
                raise ValueError("author files must use plain relative names")
        if row["supported"] != (not dimensions) or row["unsupported_dimension"] != (
            None if not dimensions else ",".join(dimensions)
        ):
            raise ValueError("unsupported forms must be declared and remain in the schedule")
        grade = row["grade"]
        if row["kind"] == "reading":
            targets = grade["accepted_answer_hex"]
            if row["files"] or grade["kind"] != "exact_answer":
                raise ValueError("reading input/grade mismatch")
        else:
            p = grade["canonical_proposal"]
            targets = [p["replacement_hex"]]
            before = {f["path"]: bytes.fromhex(f["content_hex"]) for f in row["files"]}
            after = {
                f["path"]: bytes.fromhex(f["content_hex"]) for f in grade["expected_after_files"]
            }
            if (
                p["operation"] != "SPLICE"
                or p["stop_reason"] != "eos"
                or p["path"] not in before
                or set(before) != set(after)
                or not 0 <= p["start"] <= p["end"] <= len(before[p["path"]])
            ):
                raise ValueError("invalid authored answer witness")
            expected = dict(before)
            raw = before[p["path"]]
            expected[p["path"]] = raw[: p["start"]] + bytes.fromhex(targets[0]) + raw[p["end"] :]
            if (
                expected != after
                or any(before[p] != after[p] for p in grade["protected_paths"])
                or any(len(raw) > 512 for raw in after.values())
            ):
                raise ValueError("witness changes protected files or violates raw geometry")
            for name, raw in after.items():
                if name.endswith(".py"):
                    ast.parse(raw, filename=name)
        if not targets or any(len(bytes.fromhex(target)) > 128 for target in targets):
            raise ValueError("whole authored target exceeds selected payload capacity")
        teacher = row["teacher"]
        if teacher is not None:
            if teacher["kind"] != row["kind"]:
                raise ValueError("correction kind mismatch")
            if row["kind"] == "reading":
                if teacher["target_hex"] not in targets:
                    raise ValueError("reading correction differs from exact answer")
            elif {k: v for k, v in teacher.items() if k != "kind"} != grade["canonical_proposal"]:
                raise ValueError("edit correction differs from complete witness")


def _validate_schedule(documents, episodes, previous):
    if len(episodes) != 72 or len(documents) != 12:
        raise ValueError("selected V6 episode/document allocation changed")
    expected_documents = [
        {"document_id": f"N{24 + i}", "text_hex": raw.hex()} for i, raw in enumerate(DOCUMENTS)
    ]
    if documents != expected_documents:
        raise ValueError("complete source-defined public documents changed")

    def pairs(prefix, start, end):
        return [f"{prefix}{i:02d}-{kind}" for i in range(start, end) for kind in ("R", "E")]

    retained_ids = [f"R{i:02d}-{'R' if i % 2 == 0 else 'E'}" for i in range(18)]
    expected_ids = (
        pairs("N", 0, 3)
        + pairs("F", 0, 6)
        + pairs("N", 3, 6)
        + pairs("F", 6, 12)
        + retained_ids[:12]
        + pairs("N", 6, 9)
        + retained_ids[12:]
        + [f"C{i:02d}-E" for i in range(12)]
    )
    if [row["episode_id"] for row in episodes] != expected_ids:
        raise ValueError("fixed72-episode order changed")
    first_ids = [f"N{i:02d}-{kind}" for i in range(3) for kind in ("R", "E")]
    if [row["episode_id"] for row in episodes if row["teacher"] is not None] != first_ids:
        raise ValueError("exact six first-encounter corrections changed")
    seen, appended = {}, []
    append_positions = [*range(6), *range(18, 24)]
    project_origins = {}
    for ordinal, row in enumerate(episodes):
        identity = row["episode_id"]
        phase = (
            "acquisition"
            if ordinal < 6 or 18 <= ordinal < 24 or 48 <= ordinal < 54
            else "transfer"
            if 6 <= ordinal < 18 or 24 <= ordinal < 36
            else "recovery"
            if ordinal >= 60
            else "retention"
        )
        if row["phase"] != phase or row["kind"] != (
            "reading" if identity.endswith("-R") else "edit"
        ):
            raise ValueError("phase or reading/editing allocation changed")
        if identity in seen or row["attempt_limit"] != (3 if ordinal >= 60 else 1):
            raise ValueError("duplicate identity or fixed attempts changed")
        expected_append = (
            [f"N{24 + append_positions.index(ordinal)}"] if ordinal in append_positions else []
        )
        if row["append_document_ids"] != expected_append:
            raise ValueError("whole document append order changed")
        appended.extend(row["append_document_ids"])
        predecessor = row["predecessor_episode_id"]
        if predecessor is not None:
            origin = seen[predecessor][1]
            if (
                row["workspace_lifecycle"] != "carry_actual_project"
                or row["project_id"] != origin["project_id"]
                or row["files"] != origin["grade"]["expected_after_files"]
                or row.get("nominal_files_scope")
                != "parent_only_intended_reference_never_materialize"
            ):
                raise ValueError("actual project lineage or nominal-only scope changed")
        elif row["workspace_lifecycle"] != RESET:
            raise ValueError("independent starts must retain explicit reset policy")
        if row["project_id"] is not None:
            project_origins.setdefault(row["project_id"], []).append(identity)
        if row.get("encounter") == "exact_delayed":
            before, origin = seen[row["same_session_origin_id"]]
            if ordinal - before - 1 != 47 or any(
                row[key] != origin[key] for key in ("kind", "request_hex", "files", "grade")
            ):
                raise ValueError("same-target exact repeat or47-attempt age changed")
        if row.get("encounter") == "changed_form" and not row["requirement_changed"]:
            raise ValueError("requirement change cannot be presented as pure paraphrase")
        seen[identity] = (ordinal, row)
    if appended != [row["document_id"] for row in documents]:
        raise ValueError("documents must be appended once, whole, in their declared order")
    expected_projects = {
        PROJECTS[i]: [f"N{i:02d}-E", f"N{i + 3:02d}-E", f"F{i + 6:02d}-E"] for i in range(3)
    }
    if project_origins != expected_projects:
        raise ValueError("three genuine three-stage projects are required")
    old_by_id = {row["episode_id"]: row for row in previous["episodes"]}
    retained = [row for row in episodes if row["phase"] == "retention"]
    for row, prior_id in zip(retained, RETENTION_IDS, strict=True):
        old = old_by_id[prior_id]
        if row["previous_session_episode_id"] != prior_id or any(
            row[key] != old[key] for key in ("kind", "request_hex", "files", "grade")
        ):
            raise ValueError("retained V5 observation or authored grade changed")
    _validate_rows(episodes)


def _fresh_split(episodes, previous):
    history = (
        previous_tasks._earlier_request_history(ROOT)
        + previous_tasks._prior_manifest(ROOT)["episodes"]
        + previous["episodes"]
        + [row for row in episodes if row["phase"] == "acquisition"]
    )
    old_words = [
        set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower())) for row in history
    ]
    old_constructions = {
        row.get("split_annotation", {}).get("instruction_construction") for row in history
    }
    overlaps = []
    for row in episodes:
        if row["phase"] != "transfer":
            continue
        if any(row["request_hex"] == old["request_hex"] for old in history):
            raise ValueError("fresh complete request repeats exposed v1-v5/current material")
        words = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower()))
        overlaps.append(
            {
                "episode_id": row["episode_id"],
                "maximum_all_exposed_token_jaccard": max(
                    len(words & old) / len(words | old) for old in old_words
                ),
                "annotation_name_seen_in_history": row["split_annotation"][
                    "instruction_construction"
                ]
                in old_constructions,
                "unless_word_seen_in_history": any(b"unless" in old for old in old_words),
            }
        )
    return {
        "lexical_overlap": overlaps,
        "overlap_history_scope": "authenticated v1-v5 and acquisition",
        "fully_withheld_wording_families_claimed": False,
        "annotation_labels_are_author_judgment": True,
        "claim": "new complete requests and combinations; no linguistic-family independence proof",
        "correction_episode_ids": [row["episode_id"] for row in episodes if row["teacher"]],
        "same_session_exact_intervening_attempts": 47,
        "same_session_changed_form_intervening_attempts": 17,
        "changed_forms_also_change_requirements": True,
    }


def _counts(documents, episodes):
    teachers = [row for row in episodes if row["teacher"] is not None]
    symbols = sum(
        len(bytes.fromhex(row["teacher"].get("target_hex", row["teacher"].get("replacement_hex"))))
        + 1
        for row in teachers
    )
    raw_bytes = sum(len(bytes.fromhex(row["text_hex"])) for row in documents)
    counts = {
        "episodes": len(episodes),
        "teacher_offers": len(teachers),
        "phase_episodes": {
            phase: sum(row["phase"] == phase for row in episodes)
            for phase in ("acquisition", "transfer", "recovery", "retention")
        },
        "maximum_episode_attempts": sum(row["attempt_limit"] for row in episodes),
        "maximum_action_attempts": sum(
            row["attempt_limit"] for row in episodes if row["kind"] == "edit"
        ),
        "reader_teacher_symbols": symbols,
        "reader_teacher_eos": 6,
        "reader_teacher_raw_bytes": symbols - 6,
        "action_teacher_labels": 9,
        "document_raw_bytes": raw_bytes,
        "document_eos": 12,
        "initial_occupied_memory_slots": 864,
        "new_memory_slots": raw_bytes + 12,
        "occupied_memory_slots": 864 + raw_bytes + 12,
        "source_next_byte_teacher_symbols": 0,
        "initial_assemblies": 398,
        "maximum_final_assemblies": 398 + symbols,
        "canonical_edit_witnesses": sum(row["kind"] == "edit" for row in episodes),
        "dependent_project_followers": sum(
            row["predecessor_episode_id"] is not None for row in episodes
        ),
    }
    if (
        counts["phase_episodes"]
        != {"acquisition": 18, "transfer": 24, "recovery": 12, "retention": 18}
        or len(teachers) != 6
        or symbols != 135
        or raw_bytes != 884
        or counts["maximum_episode_attempts"] != 96
        or counts["maximum_action_attempts"] != 66
        or counts["canonical_edit_witnesses"] != 42
        or counts["dependent_project_followers"] != 6
    ):
        raise ValueError("selected V6 exposure/allocation bound changed")
    return counts


def build_stream():
    """Author-only JSON construction and witness parsing, never program execution."""
    previous = _history()
    prior_documents = copy.deepcopy(previous["continuation"]["prior_documents"]) + copy.deepcopy(
        previous["documents"]
    )
    if (
        len(prior_documents) != 24
        or sum(len(bytes.fromhex(d["text_hex"])) + 1 for d in prior_documents) != 864
    ):
        raise ValueError("whole prior factual memory must remain24documents/864slots")
    documents = [
        {"document_id": f"N{24 + i}", "text_hex": raw.hex()} for i, raw in enumerate(DOCUMENTS)
    ]
    first = _first()
    changed = _changed(first)
    old = _retention(previous)
    episodes = (
        first
        + _construction_cases()
        + changed
        + _combination_cases(changed)
        + old[:12]
        + _repeats(first)
        + old[12:]
        + _recoveries()
    )
    for row in episodes:
        dimensions = _unsupported_dimensions(row)
        row["supported"] = not dimensions
        row["unsupported_dimension"] = ",".join(dimensions) if dimensions else None
    _validate_schedule(documents, episodes, previous)
    counts = _counts(documents, episodes)
    return {
        "schema": SCHEMA,
        "documents": documents,
        "episodes": episodes,
        "counts": counts,
        "capacity": {**previous["capacity"], "memory_slots": 2048, "event_owners": 1024},
        "provenance": {
            "kind": "agent_authored_open_development",
            "protected_evaluation": False,
            "official_exam": False,
            "source_sha256": _sha(Path(__file__).read_bytes()),
            "prior_manifest_sha256": HISTORY_SHA256["manifest.json"],
            "exposed_history_manifest_sha256": {
                **previous["provenance"]["exposed_history_manifest_sha256"],
                "v5": HISTORY_SHA256["manifest.json"],
            },
            "witness_policy": "parse only in builder; qualified author-only execution separate",
        },
        "continuation": {
            "prior_final_state_path": HISTORY_DIRECTORY + "/final-state.pt",
            "prior_final_state_sha256": HISTORY_SHA256["final-state.pt"],
            "prior_bank_sha256": PRIOR_BANK_SHA256,
            "selected_neural_route": "online",
            "prior_documents": prior_documents,
            "initial_memory_slots": 864,
            "prior_assemblies": 398,
            "baseline_prior_documents": 24,
            "baseline_prior_edit_demonstrations": 46,
            "baseline_predict_replays": 0,
            "control_policy": (
                "same V5 online prior; correction disabled; ordinary factual writes on"
            ),
            "recognition_initialization": (
                "zero gains; self-wiring from authoritative patterns only"
            ),
            "shared_theta_learning": "frozen prior; no byte-outcome update or historical replay",
        },
        "fresh_split": _fresh_split(episodes, previous),
        "persistence": {
            "weights_and_shared_tape": "continue_actual_v5_online_final_state_losslessly",
            "workspace": (
                "three actual three-stage projects; repeats reset; recovery carries actual"
            ),
            "nominal_chain_files": "author_only; never substitute for actual predecessor files",
        },
    }


def write_stream(path: Path):
    stream = build_stream()
    raw = (json.dumps(stream, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return {"path": str(path), "sha256": _sha(raw), "counts": stream["counts"]}


def verify_witnesses(manifest_path: Path, output: Path, expected_sha256: str, *, deadline: float):
    """Execute the 42 frozen nominal edit witnesses once, only after root release.

    This author-only check builds no task or model. The executor receives the
    nominal raw before-workspace/proposal; private grading follows its actual
    qualified child. It never becomes an inference or teacher exposure.
    """
    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        grade_execution,
        proposal_from_record,
        qualified_launch,
        workspace_from_records,
    )

    began = time.perf_counter()
    if not math.isfinite(deadline) or deadline <= began:
        raise ValueError("author verification requires an unexpired perf_counter deadline")
    raw = Path(manifest_path).read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("frozen author manifest hash mismatch")
    manifest = json.loads(raw)
    source_sha = _sha(Path(__file__).read_bytes())
    if manifest["schema"] != SCHEMA or manifest["provenance"]["source_sha256"] != source_sha:
        raise ValueError("frozen author manifest source/schema changed")
    _validate_schedule(manifest["documents"], manifest["episodes"], _history())
    if manifest["counts"] != _counts(manifest["documents"], manifest["episodes"]):
        raise ValueError("frozen author counts differ from actual records")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "persistent-language-action-v6-author-verification-v1",
        "status": "running",
        "manifest_sha256": expected_sha256,
        "source_sha256": source_sha,
        "model_constructions": 0,
        "checkpoint_loads": 0,
        "learner_calls": 0,
        "teacher_offers": 0,
        "task_builds": 0,
        "planned_children": 42,
        "attempted_witnesses": 0,
        "completed_children": 0,
        "passed_witnesses": 0,
        "failed_episode_ids": [],
    }
    try:
        launch = qualified_launch(ROOT)
        result["qualified_launch_sha256"] = launch.sha256
        with (output / "actions.jsonl").open("xb") as journal:
            for row in manifest["episodes"]:
                if row["kind"] != "edit":
                    continue
                result["attempted_witnesses"] += 1
                workspace = workspace_from_records(row["files"])
                proposal = proposal_from_record(row["grade"]["canonical_proposal"], workspace)
                after, execution = execute_proposal(
                    workspace,
                    proposal,
                    output / row["episode_id"],
                    launch=launch,
                    deadline=deadline,
                )
                result["completed_children"] += 1
                grade = grade_execution(execution, row["grade"], row["files"])
                actual_files = {file["path"]: file["content_hex"] for file in after}
                expected_files = {
                    file["path"]: file["content_hex"]
                    for file in row["grade"]["expected_after_files"]
                }
                exact_files = actual_files == expected_files
                passed = grade["task_success"] and execution["accepted"] and exact_files
                record = {
                    "episode_id": row["episode_id"],
                    "execution": execution,
                    "grade": grade,
                    "nominal_after_files_equal": exact_files,
                    "passed": passed,
                }
                journal.write((json.dumps(record, sort_keys=True) + "\n").encode())
                journal.flush()
                if passed:
                    result["passed_witnesses"] += 1
                else:
                    result["failed_episode_ids"].append(row["episode_id"])
        result["status"] = (
            "passed"
            if result["passed_witnesses"] == result["completed_children"] == 42
            else "failed"
        )
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - began
        journal_path = output / "actions.jsonl"
        if journal_path.exists():
            result["actions_sha256"] = _sha(journal_path.read_bytes())
        with (output / "report.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, sort_keys=True, indent=2)
            handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "verify"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.command == "build":
        result = write_stream(args.path)
    elif args.command == "verify":
        if (
            args.output is None
            or args.sha256 is None
            or args.seconds is None
            or not math.isfinite(args.seconds)
            or args.seconds <= 0
        ):
            parser.error("verify requires --output, --sha256 and positive finite --seconds")
        result = verify_witnesses(
            args.path, args.output, args.sha256, deadline=time.perf_counter() + args.seconds
        )
    else:
        raw = args.path.read_bytes()
        manifest = json.loads(raw)
        if manifest["schema"] != SCHEMA or manifest["provenance"]["source_sha256"] != _sha(
            Path(__file__).read_bytes()
        ):
            raise ValueError("manifest source/schema binding changed")
        _validate_schedule(manifest["documents"], manifest["episodes"], _history())
        if manifest["counts"] != _counts(manifest["documents"], manifest["episodes"]):
            raise ValueError("saved manifest exposure counts differ from its actual rows")
        result = {
            "path": str(args.path),
            "sha256": _sha(raw),
            "status": "passed",
            "scope": "saved structural/witness AST check; zero model or child calls",
        }
    print(json.dumps(result, sort_keys=True))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
