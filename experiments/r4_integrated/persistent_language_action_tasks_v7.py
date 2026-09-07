"""Held V7 public authoring source; constructing data never executes its programs.

No model, baseline prediction, source lesson or checkpoint decoder is called here.
Historical JSON is authenticated as data. The held author CLI can build once,
check frozen JSON or verify nominal witnesses only after a reviewed release.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_tasks_v5 as history_tasks
from experiments.r4_integrated import persistent_language_action_tasks_v6 as author
from experiments.r4_integrated.persistent_language_action_tasks_v4 import (
    ROOT,
    _editing,
    _project,
    _reading,
    observation,
)

SCHEMA = "persistent-language-action-stream-v7"
DRAFT = False  # root reviewed: one pure build, then once-only nominal author verification
HISTORY_DIRECTORY = author.HISTORY_DIRECTORY
HISTORY_SHA256 = dict(author.HISTORY_SHA256)
PRIOR_BANK_SHA256 = author.PRIOR_BANK_SHA256
PRIOR_ATTEMPTS_SHA256 = "9139824c7897f15e3a3aed71a7692f7a15c28ad18fa4d82d48072dee5981e638"
V6_MANIFEST_PATH = ".pytest-tmp/persistent-language-action-v6-manifest.json"
V6_MANIFEST_SHA256 = "9ac45b7754f9ac033cc85aba49d66bee1b471e9a6da7ee26ecd883c25d65051f"
RESET = "reset_at_episode_start_continue_within_retries"

DOCUMENTS = (
    b"2026-09-10 depot storage: frozen goods use 'cold'; other goods use 'warm'.",
    b"2026-09-10 depot quota: at most '2' units per order; negative units count as zero.",
    b"2026-09-11 depot storage: fragile goods use 'padded' even when frozen.",
    b"2026-09-11 depot quota: north allows '3' units, south '5'; replace the earlier cap.",
    b"Depot orders, in order: elm/north/4, ash/south/-1, oak/south/7, elm/north/4.",
    b"Unrelated gallery storage is 'dry'; its quota is '8'. Depot names keep input order.",
    b"def positive(xs):\n    return [x for x in xs if x>0]\n",
    b"def normal(s):\n    return s.strip().lower()\n",
    b"def unique(xs):\n    return list(dict.fromkeys(xs))\n",
    b"def bounded(n,cap):\n    return max(0,min(n,cap))\n",
    b"def joined(xs):\n    return '|'.join(xs)\n",
    b"def total(xs):\n    return sum(xs)\n",
)
RETENTION_IDS = (
    "R00-R",
    "R01-E",
    "R02-R",
    "R03-E",
    "R06-R",
    "R07-E",
    "R10-R",
    "R11-E",
    "R14-R",
    "R15-E",
    "A11-R",
    "A11-E",
)
ORDER = (
    "A00-R",
    "A00-E",
    "A01-R",
    "A01-E",
    "C00-E",
    "A02-R",
    "A02-E",
    "F00-R",
    "F00-E",
    "C01-E",
    "A03-R",
    "A03-E",
    "F01-R",
    "F01-E",
    "C02-E",
    "F02-R",
    "F02-E",
    "C03-E",
    "F03-R",
    "F03-E",
    "F04-R",
    "F04-E",
    "F05-R",
    "F05-E",
    "C04-E",
    "F06-R",
    "F06-E",
    "C05-E",
    "F07-R",
    "F07-E",
    "C06-E",
    "C07-E",
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
    "A04-R",
    "A04-E",
    "A05-R",
    "A05-E",
)
READ_TARGETS = (b"cold/warm", b"2")
EDIT_TARGETS = (
    b"return 'cold' if frozen else 'warm'",
    b"return max(0,min(n,2))",
)
SIGNATURES = ("def storage(frozen, fragile=False):", "def quota(n, region='north'):")
MAINS = (
    b"import app\nfor f,g in ((True,False),(False,True),(True,True)):\n"
    b"    print(app.storage(f,g))\n",
    b"import app\nfor n,r in ((-1,'north'),(4,'north'),(9,'south'),(1,'south')):\n"
    b"    print(app.quota(n,r))\n",
)
STDOUTS = (b"cold\nwarm\ncold\n", b"0\n2\n2\n1\n")
FIRST_REQUESTS = (
    (
        "Read the depot storage note dated 2026-09-10. Give the frozen choice then the "
        "ordinary choice, separated by /, without any explanation.",
        "Implement storage(frozen, fragile=False) in app.py from the depot storage note "
        "dated 2026-09-10. This dated version ignores fragile. Preserve main.py.",
    ),
    (
        "Read the depot quota note dated 2026-09-10. Give its upper limit as one number, "
        "without explaining the later regional revision.",
        "Implement quota(n, region='north') in app.py from the depot quota dated "
        "2026-09-10: negatives become zero, larger values stop at its cap, and region "
        "is ignored. Preserve main.py.",
    ),
)
PARAPHRASES = (
    "Using only the depot's 2026-09-10 storage rule, write frozen/ordinary choices "
    "as a slash-separated pair. Leave out explanations and later exceptions.",
    "Complete app.py so storage chooses the depot's frozen or ordinary location "
    "under its 2026-09-10 rule. Do not let fragile affect this version; keep main.py.",
    "Under the depot's earlier quota dated 2026-09-10, what is the greatest allowed "
    "unit count? Reply with the number alone, not the later regional limits.",
    "Complete app.py's quota using the depot's 2026-09-10 limit: clamp n between zero "
    "and that limit for either region. Do not change main.py.",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _history():
    raw = (ROOT / HISTORY_DIRECTORY / "manifest.json").read_bytes()
    if _sha(raw) != HISTORY_SHA256["manifest.json"]:
        raise ValueError("complete V5 manifest changed")
    return json.loads(raw)


def _prior_regression_bindings(previous):
    """Authenticate all saved rows, retaining only actual selected observations.

    Raw JSON and Boolean outcomes are data; this invokes no historical actor,
    grade program, reference prediction, teacher or checkpoint deserializer.
    """
    routes = ("online", "learning_disabled", "simple")
    selected = {
        row["episode_id"]: row for row in previous["episodes"] if row["episode_id"] in RETENTION_IDS
    }
    if set(selected) != set(RETENTION_IDS):
        raise ValueError("selected V5 regression IDs are missing")
    retained = {route: {} for route in routes}
    ordinals = dict.fromkeys(routes, 0)
    digest, total_bytes, row_count = hashlib.sha256(), 0, 0
    path = ROOT / HISTORY_DIRECTORY / "attempts.jsonl"
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            total_bytes += len(raw)
            row_count += 1
            row = json.loads(raw)
            route = row["route"]
            if row["schema"] != "persistent-language-action-attempt-v5" or route not in retained:
                raise ValueError("prior actual row schema/route mismatch")
            identity = row["episode_id"]
            if identity in selected:
                original = selected[identity]
                expected_observation = {
                    "schema": "sequence-assembly-raw-observation-v1",
                    "request_hex": original["request_hex"],
                    "files": sorted(original["files"], key=lambda item: item["path"].encode())
                    if original["files"]
                    else None,
                    "feedback": None,
                }
                if (
                    identity in retained[route]
                    or row["attempt_index"] != 1
                    or type(row["correct"]) is not bool
                    or row["kind"] != original["kind"]
                    or row["request_hex"] != original["request_hex"]
                    or row["before_files"] != original["files"]
                    or row["observation"] != expected_observation
                ):
                    raise ValueError("selected historical actual observation differs from reset")
                retained[route][identity] = {
                    "episode_id": identity,
                    "attempt_index": row["attempt_index"],
                    "route_ordinal": ordinals[route],
                    "correct": row["correct"],
                    "observation": row["observation"],
                    "attempt_sha256": _sha(raw),
                }
            ordinals[route] += 1
    if (
        digest.hexdigest() != PRIOR_ATTEMPTS_SHA256
        or row_count != 240
        or ordinals != dict.fromkeys(routes, 80)
        or any(set(rows) != set(RETENTION_IDS) for rows in retained.values())
    ):
        raise ValueError("complete V5 journal/selected observations changed")
    return {
        "schema": "persistent-language-action-v7-prior-regression-bindings-v1",
        "path": HISTORY_DIRECTORY + "/attempts.jsonl",
        "sha256": PRIOR_ATTEMPTS_SHA256,
        "read_passes": 1,
        "bytes_read": total_bytes,
        "rows_read": row_count,
        "route_attempts": ordinals,
        "retained_observations": 36,
        "routes": retained,
        "scope": "actual raw V5 inputs/outcomes only; zero prediction or grade replay",
    }


def _first():
    rows = []
    for i in range(2):
        reading = _reading(
            f"A{i:02d}-R",
            "acquisition",
            FIRST_REQUESTS[i][0],
            READ_TARGETS[i],
            teach=True,
            construction="direct_dated_requirement",
        )
        editing = _editing(
            f"A{i:02d}-E",
            "acquisition",
            FIRST_REQUESTS[i][1],
            (SIGNATURES[i] + "\n    pass\n").encode(),
            b"pass",
            EDIT_TARGETS[i],
            STDOUTS[i],
            MAINS[i],
            teach=True,
            construction="direct_dated_requirement",
            structure="conditional_expression" if i == 0 else "bounded_expression",
        )
        _project(editing, ("depot_storage", "depot_quota")[i])
        for row in (reading, editing):
            row.update(
                encounter="first",
                requirement_family=("storage", "quota")[i],
                same_session_origin_id=None,
                requirement_changed=False,
            )
        rows.extend((reading, editing))
    return rows


def _revisits(first):
    rows = []
    for repeat in (False, True):
        for i, origin in enumerate(first):
            row = copy.deepcopy(origin)
            row.update(
                episode_id=f"A{(4 if repeat else 2) + i // 2:02d}-{'R' if i % 2 == 0 else 'E'}",
                teacher=None,
                project_id=None,
                predecessor_episode_id=None,
                workspace_lifecycle=RESET,
                append_document_ids=[],
                encounter="exact_delayed" if repeat else "changed_form",
                same_session_origin_id=origin["episode_id"],
                requirement_changed=False,
                exposure_class="same_session_exposed_retention" if repeat else "pure_changed_form",
            )
            if not repeat:
                row["request_hex"] = PARAPHRASES[i].encode().hex()
                row["split_annotation"]["instruction_construction"] = "dated_paraphrase"
            rows.append(row)
    return rows


def _fresh(first):
    # First four pairs reserve whole constructions against current answer lessons.
    reads = (
        (
            "Suppose normal receives ' MIXED '. Trace the observed source on that input "
            "and return only its resulting string, without quotes.",
            b"mixed",
            "hypothetical_trace",
        ),
        (
            "Acceptance examples use bounded from the observed source: the input is n=6 "
            "and cap=3. Give the returned number alone.",
            b"3",
            "acceptance_examples",
        ),
        (
            "Instead of sorting names, remove later duplicates from ['b','a','b'] and "
            "join the remaining names with |. Return just that string.",
            b"b|a",
            "contrastive_order",
        ),
        (
            "A caller relies on the historical depot quota contract dated 2026-09-10. "
            "For n=9, what number must it return? State only that result.",
            b"2",
            "behavioral_contract",
        ),
        (
            "Normalize each name in [' ELM ','Ash','elm'], remove repeated normalized "
            "names while retaining first order, and join with |. Return the result alone.",
            b"elm|ash",
            "normalize_unique_join",
        ),
        (
            "For the four documented depot orders, apply each region's 2026-09-11 quota "
            "to its units, including the repeated order. Return their total as one number.",
            b"11",
            "regional_clamp_aggregate",
        ),
        (
            "Combine the depot's 2026-09-11 storage exception with its earlier rule. "
            "Give frozen-fragile/nonfrozen-ordinary locations separated by /.",
            b"padded/warm",
            "dated_exception_precedence",
        ),
        (
            "Using the documented depot orders and revised regional quotas, keep names "
            "whose allowed count is positive, deduplicate in first order and join with |.",
            b"elm|oak",
            "clamp_filter_unique_join",
        ),
    )
    edits = (
        (
            "Suppose app.py's clean receives padded uppercase text or an empty string. "
            "Trace the observed normal rule into its implementation; preserve main.py.",
            "def clean(s):",
            "s.strip().lower()",
            "for s in (' MIXED ','',' Ash '):\n    print(app.clean(s))\n",
            "mixed\n\nash\n",
            "hypothetical_trace",
            None,
        ),
        (
            "Acceptance examples for count_positive are [-1,0,2,3] -> 2 and [] -> 0. "
            "Complete app.py to count strictly positive values for any list; keep main.py.",
            "def count_positive(xs):",
            "sum(x>0 for x in xs)",
            "for xs in ([-1,0,2,3],[],[0,-2]):\n    print(app.count_positive(xs))\n",
            "2\n0\n0\n",
            "acceptance_examples",
            None,
        ),
        (
            "Instead of alphabetizing the list, unique_names must keep the first copy "
            "of each name in original order. Complete app.py and leave main.py intact.",
            "def unique_names(xs):",
            "list(dict.fromkeys(xs))",
            "for xs in (['b','a','b'],[],['a','a']):\n    print(app.unique_names(xs))\n",
            "['b', 'a']\n[]\n['a']\n",
            "contrastive_order",
            None,
        ),
        (
            "A caller relies on a nonnegative subtotal: positive values contribute "
            "their value, zero and negatives contribute nothing. Complete app.py; keep main.py.",
            "def subtotal(xs):",
            "sum(x for x in xs if x>0)",
            "for xs in ([-1,0,2,3],[],[-4]):\n    print(app.subtotal(xs))\n",
            "5\n0\n0\n",
            "behavioral_contract",
            None,
        ),
        (
            "Complete app.py's labels by normalizing every name, removing duplicates "
            "in first order, then joining with |. Use the available helpers; "
            "preserve both other files.",
            "def labels(xs):",
            "rules.joined(rules.unique([rules.normal(x) for x in xs]))",
            "for xs in ([' ELM ','Ash','elm'],[],[' A ','a']):\n    print(app.labels(xs))\n",
            "elm|ash\n\na\n",
            "normalize_unique_join",
            "def normal(s):\n    return s.strip().lower()\n"
            "def unique(xs):\n    return list(dict.fromkeys(xs))\n"
            "def joined(xs):\n    return '|'.join(xs)\n",
        ),
        (
            "Complete app.py's allowed_total for (region,units) rows: clamp each count "
            "using the revised 2026-09-11 depot quota and add them. Preserve main.py.",
            "def allowed_total(rows):",
            "sum(max(0,min(n,3 if r=='north' else 5)) for r,n in rows)",
            "for rows in ([('north',4),('south',-1),('south',7),('north',4)],[]):\n"
            "    print(app.allowed_total(rows))\n",
            "11\n0\n",
            "regional_clamp_aggregate",
            None,
        ),
    )
    rows = []
    for i, (query, target, construction) in enumerate(reads):
        rows.append(_reading(f"F{i:02d}-R", "transfer", query, target, construction=construction))
    for i, (query, signature, expression, driver, stdout, construction, helpers) in enumerate(
        edits
    ):
        rows.append(
            author._fresh_edit(
                f"F{i:02d}-E",
                query,
                signature,
                expression,
                driver,
                stdout,
                construction=construction,
                helpers=helpers.encode() if helpers is not None else None,
            )
        )
    revisions = (
        (
            "Update the actual app.py from the earlier depot storage version: the dated "
            "2026-09-11 fragile exception overrides frozen; ordinary cases keep the old rule. "
            "Preserve main.py.",
            b"return 'padded' if fragile else 'cold' if frozen else 'warm'",
            b"cold\npadded\npadded\n",
            "dated_exception_precedence",
        ),
        (
            "Update the actual app.py quota to the depot's 2026-09-11 regional revision. "
            "Keep the negative guard and return uncapped smaller counts. Preserve main.py.",
            b"return max(0,min(n,3 if region=='north' else 5))",
            b"0\n3\n5\n1\n",
            "dated_regional_revision",
        ),
    )
    for i, (query, target, stdout, construction) in enumerate(revisions):
        origin = first[2 * i + 1]
        nominal = bytes.fromhex(origin["grade"]["expected_after_files"][0]["content_hex"])
        row = _editing(
            f"F{6 + i:02d}-E",
            "transfer",
            query,
            nominal,
            EDIT_TARGETS[i],
            target,
            stdout,
            MAINS[i],
            construction=construction,
            structure="actual_project_revision",
        )
        _project(row, origin["project_id"], origin["episode_id"])
        rows.append(row)
    for row in rows:
        row["fresh_stratum"] = (
            "instruction_construction"
            if int(row["episode_id"][1:3]) < 4
            else "structural_combination"
        )
        row["requirement_changed"] = row["predecessor_episode_id"] is not None
    return rows


def _recoveries():
    cases = (
        (
            "Fix positive in app.py so zero is excluded along with negative values. "
            "The driver checks the boundary and an empty list. Preserve main.py.",
            "def positive(xs):\n    return [x for x in xs if x>=0]\n",
            b"x>=0",
            b"x>0",
            "for xs in ([-1,0,2],[]):\n    print(app.positive(xs))\n",
            "[2]\n[]\n",
        ),
        (
            "Fix normal in app.py to trim outer whitespace before lowercasing, including "
            "whitespace-only input. Keep internal characters and preserve main.py.",
            "def normal(s):\n    return s.lower()\n",
            b"s.lower()",
            b"s.strip().lower()",
            "for s in (' A ','   ',' B C '):\n    print(app.normal(s))\n",
            "a\n\nb c\n",
        ),
        (
            "Repair first_or_blank: return the first list element when one exists and "
            "an empty string for an empty list. Do not change main.py or discard false values.",
            "def first_or_blank(xs):\n    return xs[0]\n",
            b"xs[0]",
            b"xs[0] if xs else ''",
            "for xs in ([],['a'],[0]):\n    print(app.first_or_blank(xs))\n",
            "\na\n0\n",
        ),
        (
            "Repair storage for the revised depot policy: fragile overrides frozen, "
            "and otherwise the old frozen/ordinary choice applies. Keep main.py unchanged.",
            "def storage(frozen,fragile):\n    return 'cold' if frozen else 'padded'\n",
            b"'cold' if frozen else 'padded'",
            b"'padded' if fragile else 'cold' if frozen else 'warm'",
            "for f,g in ((True,True),(False,False),(True,False)):\n    print(app.storage(f,g))\n",
            "padded\nwarm\ncold\n",
        ),
        (
            "Repair regional quota in app.py using the depot's 2026-09-11 note. Keep "
            "negative counts at zero and smaller counts unchanged; preserve main.py.",
            "def quota(n,r):\n    return min(n,2)\n",
            b"min(n,2)",
            b"max(0,min(n,3 if r=='north' else 5))",
            "for n,r in ((-2,'north'),(7,'south'),(2,'north')):\n    print(app.quota(n,r))\n",
            "0\n5\n2\n",
        ),
        (
            "Repair names in app.py: remove duplicates without sorting, preserving "
            "first appearance and returning a list even on empty input. Preserve main.py.",
            "def names(xs):\n    return sorted(set(xs))\n",
            b"sorted(set(xs))",
            b"list(dict.fromkeys(xs))",
            "for xs in (['z','a','z'],[]):\n    print(app.names(xs))\n",
            "['z', 'a']\n[]\n",
        ),
        (
            "Repair capped in app.py so each input is clamped between zero and its "
            "cap rather than taking the larger number. Keep the function interface and main.py.",
            "def capped(n,cap):\n    return max(n,cap)\n",
            b"max(n,cap)",
            b"max(0,min(n,cap))",
            "for n,c in ((-1,3),(9,3),(2,3)):\n    print(app.capped(n,c))\n",
            "0\n3\n2\n",
        ),
        (
            "Repair subtotal in app.py: sum strictly positive elements, do not merely "
            "count them. Empty lists must total zero. Preserve the public main.py checks.",
            "def subtotal(xs):\n    return sum(x>0 for x in xs)\n",
            b"sum(x>0 for x in xs)",
            b"sum(x for x in xs if x>0)",
            "for xs in ([-1,2,3],[]):\n    print(app.subtotal(xs))\n",
            "5\n0\n",
        ),
    )
    return [
        _editing(
            f"C{i:02d}-E",
            "recovery",
            query,
            source.encode(),
            marker,
            replacement,
            stdout.encode(),
            ("import app\n" + driver).encode(),
            construction="public_behavioral_bug_report",
            structure="faulty_function",
        )
        for i, (query, source, marker, replacement, driver, stdout) in enumerate(cases)
    ]


def _retention(previous):
    old = {row["episode_id"]: row for row in previous["episodes"]}
    rows = []
    for i, identity in enumerate(RETENTION_IDS):
        row = copy.deepcopy(old[identity])
        row.update(
            episode_id=f"R{i:02d}-{'R' if i % 2 == 0 else 'E'}",
            phase="retention",
            teacher=None,
            append_document_ids=[],
            attempt_limit=1,
            exposure_class="exact_exposed_v5_regression",
            previous_session_episode_id=identity,
            previous_session="v5",
            encounter="prior_regression",
            same_session_origin_id=None,
            requirement_changed=False,
            project_id=None,
            predecessor_episode_id=None,
            workspace_lifecycle=RESET,
        )
        rows.append(row)
    return rows


def _validate(documents, episodes, previous):
    author._validate_rows(episodes)
    if tuple(row["episode_id"] for row in episodes) != ORDER or len(set(ORDER)) != 48:
        raise ValueError("fixed V7 order changed")
    seen, attempts, projects = {}, 0, {}
    for ordinal, row in enumerate(episodes):
        identity = row["episode_id"]
        phase = {"A": "acquisition", "F": "transfer", "R": "retention", "C": "recovery"}[
            identity[0]
        ]
        if (
            row["phase"] != phase
            or row["attempt_limit"] != (3 if phase == "recovery" else 1)
            or row["kind"] != ("reading" if identity.endswith("-R") else "edit")
            or row["append_document_ids"] != ([f"N{24 + ordinal}"] if ordinal < 12 else [])
        ):
            raise ValueError("phase, whole append, or attempt allocation changed")
        if bool(row["teacher"] is not None) != (ordinal < 4):
            raise ValueError("only first four responses receive corrections")
        if row.get("encounter") in ("changed_form", "exact_delayed"):
            before, origin = seen[row["same_session_origin_id"]]
            if any(row[k] != origin[k] for k in ("kind", "files", "grade")):
                raise ValueError("revisit must preserve the entire original requirement")
            if row["requirement_changed"]:
                raise ValueError("pure paraphrase cannot change requirements")
            exact = row["encounter"] == "exact_delayed"
            if (row["request_hex"] == origin["request_hex"]) != exact:
                raise ValueError("exact/paraphrase input identity mismatch")
            row["intervening_attempts"] = attempts - before - 1
            if exact and row["intervening_attempts"] != 59:
                raise ValueError("exact repeat age changed")
        if row["project_id"]:
            projects.setdefault(row["project_id"], []).append(identity)
        if row["predecessor_episode_id"]:
            origin = seen[row["predecessor_episode_id"]][1]
            if (
                row["files"] != origin["grade"]["expected_after_files"]
                or row["workspace_lifecycle"] != "carry_actual_project"
                or row["nominal_files_scope"] != "parent_only_intended_reference_never_materialize"
            ):
                raise ValueError("actual project dependency contract changed")
        elif row["workspace_lifecycle"] != RESET:
            raise ValueError("independent episode must start with its declared workspace")
        seen[identity] = (attempts, row)
        attempts += row["attempt_limit"]
    if projects != {"depot_storage": ["A00-E", "F06-E"], "depot_quota": ["A01-E", "F07-E"]}:
        raise ValueError("two actual two-stage projects required")
    if [i for i, row in enumerate(episodes) if row["phase"] == "recovery"] != [
        4,
        9,
        14,
        17,
        24,
        27,
        30,
        31,
    ]:
        raise ValueError("early interleaved recovery schedule changed")
    old = {row["episode_id"]: row for row in previous["episodes"]}
    for row, identity in zip(
        (r for r in episodes if r["phase"] == "retention"), RETENTION_IDS, strict=True
    ):
        if any(row[k] != old[identity][k] for k in ("kind", "request_hex", "files", "grade")):
            raise ValueError("historical raw observation or private grade changed")
    if documents != [
        {"document_id": f"N{24 + i}", "text_hex": raw.hex()} for i, raw in enumerate(DOCUMENTS)
    ]:
        raise ValueError("whole authored documents changed")


def _fresh_split(episodes, previous):
    raw = (ROOT / V6_MANIFEST_PATH).read_bytes()
    if _sha(raw) != V6_MANIFEST_SHA256:
        raise ValueError("exposed partial V6 manifest changed")
    older = (
        history_tasks._earlier_request_history(ROOT)
        + history_tasks._prior_manifest(ROOT)["episodes"]
        + previous["episodes"]
        + json.loads(raw)["episodes"]
    )
    current = [r for r in episodes if r["phase"] == "acquisition"]
    history = older + current
    words = [
        set(re.findall(rb"[a-z0-9]+", bytes.fromhex(r["request_hex"]).lower())) for r in history
    ]
    current_families = {r["split_annotation"]["instruction_construction"] for r in current}
    held = {
        r["split_annotation"]["instruction_construction"]
        for r in episodes
        if r.get("fresh_stratum") == "instruction_construction"
    }
    if len(held) != 4 or held & current_families:
        raise ValueError("four construction families must be absent from current answer lessons")
    overlaps = []
    for row in (r for r in episodes if r["phase"] == "transfer"):
        if any(row["request_hex"] == r["request_hex"] for r in history):
            raise ValueError("fresh complete wording repeats an exposed request")
        tokens = set(re.findall(rb"[a-z0-9]+", bytes.fromhex(row["request_hex"]).lower()))
        overlaps.append(
            {
                "episode_id": row["episode_id"],
                "maximum_all_exposed_token_jaccard": max(
                    len(tokens & w) / len(tokens | w) for w in words
                ),
                "annotation_name_seen_in_older_history": any(
                    row["split_annotation"]["instruction_construction"]
                    == r.get("split_annotation", {}).get("instruction_construction")
                    for r in older
                ),
            }
        )
    return {
        "lexical_overlap": overlaps,
        "overlap_history_scope": "authenticated v1-v6 and current acquisition",
        "current_answer_construction_holdouts": sorted(held),
        "fully_withheld_wording_families_claimed": False,
        "annotation_labels_are_author_judgment": True,
        "changed_forms_also_change_requirements": False,
        "same_session_exact_intervening_attempts": 59,
        "same_session_changed_form_intervening_attempts": [6, 6, 11, 11],
        "claim": (
            "current construction holdouts; historical quota/exception/normalization "
            "overlap remains"
        ),
    }


def _counts(documents, episodes):
    teachers = [r for r in episodes if r["teacher"] is not None]
    answer_raw = sum(
        len(bytes.fromhex(r["teacher"].get("target_hex", r["teacher"].get("replacement_hex"))))
        for r in teachers
    )
    source_raw = sum(len(bytes.fromhex(d["text_hex"])) for d in documents)
    counts = {
        "episodes": len(episodes),
        "teacher_offers": len(teachers),
        "phase_episodes": {
            p: sum(r["phase"] == p for r in episodes)
            for p in ("acquisition", "transfer", "retention", "recovery")
        },
        "maximum_episode_attempts": sum(r["attempt_limit"] for r in episodes),
        "maximum_action_attempts": sum(r["attempt_limit"] for r in episodes if r["kind"] == "edit"),
        "reader_teacher_raw_bytes": answer_raw,
        "reader_teacher_eos": 4,
        "reader_teacher_symbols": answer_raw + 4,
        "action_teacher_labels": 6,
        "document_raw_bytes": source_raw,
        "document_eos": 12,
        "new_memory_slots": source_raw + 12,
        "initial_occupied_memory_slots": 864,
        "occupied_memory_slots": 864 + source_raw + 12,
        "source_next_byte_teacher_symbols": source_raw + 12,
        "combined_observed_symbol_offers": source_raw + 12 + answer_raw + 4,
        "initial_assemblies": 398,
        "canonical_edit_witnesses": 28,
        "dependent_project_followers": 2,
    }
    if (
        counts["phase_episodes"]
        != {"acquisition": 12, "transfer": 16, "retention": 12, "recovery": 8}
        or len(teachers) != 4
        or answer_raw != 67
        or source_raw != 738
        or source_raw > 960
        or counts["maximum_episode_attempts"] != 64
        or counts["maximum_action_attempts"] != 44
        or sum(r["kind"] == "edit" for r in episodes) != 28
    ):
        raise ValueError("fixed V7 exposure or capacity bound changed")
    return counts


def build_stream():
    """One pure author build after source review; AST checks, never program execution."""
    previous = _history()
    prior_bindings = _prior_regression_bindings(previous)
    prior_documents = copy.deepcopy(
        previous["continuation"]["prior_documents"] + previous["documents"]
    )
    if (
        len(prior_documents) != 24
        or sum(len(bytes.fromhex(d["text_hex"])) + 1 for d in prior_documents) != 864
    ):
        raise ValueError("complete V5 memory lineage changed")
    documents = [
        {"document_id": f"N{24 + i}", "text_hex": raw.hex()} for i, raw in enumerate(DOCUMENTS)
    ]
    first = _first()
    rows = first + _revisits(first) + _fresh(first) + _retention(previous) + _recoveries()
    by_id = {row["episode_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate authored episode")
    episodes = [by_id[identity] for identity in ORDER]
    for ordinal, row in enumerate(episodes):
        row["append_document_ids"] = [f"N{24 + ordinal}"] if ordinal < 12 else []
        dimensions = author._unsupported_dimensions(row)
        row["supported"] = not dimensions
        row["unsupported_dimension"] = ",".join(dimensions) if dimensions else None
    _validate(documents, episodes, previous)
    return {
        "schema": SCHEMA,
        "documents": documents,
        "episodes": episodes,
        "counts": _counts(documents, episodes),
        "prior_regression_bindings": prior_bindings,
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
                "v6": V6_MANIFEST_SHA256,
            },
            "witness_policy": (
                "builder AST parse only; qualified author execution separately reviewed"
            ),
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
            "partial_v6_state_restored": False,
            "control_policy": (
                "same V5 online prior; contacts/action updates disabled; whole factual writes on"
            ),
            "new_learning_state": "empty paired dendritic contacts; no old source/answer replay",
            "source_learning": "one causal rawbyte-plus-EOS pass per new whole source",
            "public_feedback_learning": False,
        },
        "fresh_split": _fresh_split(episodes, previous),
        "persistence": {
            "weights_and_shared_tape": "continue_actual_v5_online_final_state_losslessly",
            "workspace": (
                "two actual two-stage projects; all other starts reset; retries carry actual"
            ),
            "nominal_chain_files": "author_only; never substitute for actual predecessor files",
            "safe_boundaries": "initialization, every append, every resolved attempt",
            "interrupted_intent": "retain last safe boundary; refuse automatic same-stream replay",
        },
    }


def write_stream(path: Path):
    """Exactly one pure author build; exclusive creation prevents replacement."""
    path = Path(path)
    if path.exists():
        raise ValueError("author manifest destination already exists")
    stream = build_stream()
    raw = (json.dumps(stream, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return {
        "path": str(path),
        "sha256": _sha(raw),
        "counts": stream["counts"],
        "task_builds": 1,
        "model_constructions": 0,
        "children": 0,
        "prior_observation_reads": {
            key: value
            for key, value in stream["prior_regression_bindings"].items()
            if key != "routes"
        },
    }


def check_saved_manifest(path: Path, expected_sha256: str):
    """Authenticate frozen author JSON and selected history without rebuilding."""
    raw = Path(path).read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("frozen author manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest["schema"] != SCHEMA or manifest["provenance"]["source_sha256"] != _sha(
        Path(__file__).read_bytes()
    ):
        raise ValueError("frozen author source/schema mismatch")
    previous = _history()
    checked = copy.deepcopy(manifest["episodes"])
    _validate(manifest["documents"], checked, previous)
    if checked != manifest["episodes"]:
        raise ValueError("saved episode annotations differ from validated schedule")
    if manifest["counts"] != _counts(manifest["documents"], manifest["episodes"]):
        raise ValueError("saved author exposure counts changed")
    if manifest["prior_regression_bindings"] != _prior_regression_bindings(previous):
        raise ValueError("saved prior observation/outcome bindings changed")
    return manifest


def verify_witnesses(manifest_path: Path, output: Path, expected_sha256: str, *, deadline: float):
    """Verify28 frozen nominal edit witnesses once; never build tasks or actors."""
    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        grade_execution,
        proposal_from_record,
        qualified_launch,
        workspace_from_records,
    )

    began = time.perf_counter()
    if not math.isfinite(deadline) or deadline <= began:
        raise ValueError("author verification needs an unexpired perf_counter deadline")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "persistent-language-action-v7-author-verification-v1",
        "status": "running",
        "manifest_sha256": expected_sha256,
        "source_sha256": _sha(Path(__file__).read_bytes()),
        "model_constructions": 0,
        "checkpoint_loads": 0,
        "learner_calls": 0,
        "teacher_offers": 0,
        "task_builds": 0,
        "planned_children": 28,
        "attempted_witnesses": 0,
        "completed_children": 0,
        "passed_witnesses": 0,
        "failed_episode_ids": [],
    }
    try:
        manifest = check_saved_manifest(manifest_path, expected_sha256)
        result["prior_observation_reads"] = {
            key: value
            for key, value in manifest["prior_regression_bindings"].items()
            if key != "routes"
        }
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
                exact_files = {item["path"]: item["content_hex"] for item in after} == {
                    item["path"]: item["content_hex"]
                    for item in row["grade"]["expected_after_files"]
                }
                passed = grade["task_success"] and execution["accepted"] and exact_files
                record = {
                    "episode_id": row["episode_id"],
                    "execution": execution,
                    "grade": grade,
                    "nominal_after_files_equal": exact_files,
                    "passed": passed,
                    "scope": "author nominal witness; not learner output or teaching",
                }
                journal.write((json.dumps(record, sort_keys=True) + "\n").encode())
                journal.flush()
                if passed:
                    result["passed_witnesses"] += 1
                else:
                    result["failed_episode_ids"].append(row["episode_id"])
        result["status"] = (
            "passed"
            if result["passed_witnesses"] == result["completed_children"] == 28
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
    if DRAFT:
        raise SystemExit("V7 author CLI is held pending root source review and release")
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
            parser.error("verify needs --output, --sha256 and positive finite --seconds")
        result = verify_witnesses(
            args.path, args.output, args.sha256, deadline=time.perf_counter() + args.seconds
        )
    else:
        if args.sha256 is None:
            parser.error("check requires --sha256")
        manifest = check_saved_manifest(args.path, args.sha256)
        result = {
            "status": "passed",
            "sha256": args.sha256,
            "counts": manifest["counts"],
            "task_builds": 0,
            "children": 0,
            "model_constructions": 0,
        }
    print(json.dumps(result, sort_keys=True))
    if result.get("status") == "failed":
        raise SystemExit(1)


__all__ = [
    "build_stream",
    "check_saved_manifest",
    "observation",
    "verify_witnesses",
    "write_stream",
]


if __name__ == "__main__":
    main()
