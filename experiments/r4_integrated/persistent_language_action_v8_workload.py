"""V8 source-preparation draft: pure authored data and authenticated historical JSON.

No module import builds a stream, predicts, teaches, decodes a checkpoint or runs
an authored program. Root releases the pure build/check CLI; runtime remains held.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_tasks_v4 import (
    ROOT,
    _editing,
    _project,
    _reading,
    observation,
)
from experiments.r4_integrated.persistent_language_action_v8_sources import (
    ADMITTED,
    PROGRAMS,
    PROSE,
)

SCHEMA = "persistent-language-action-stream-v8"
DRAFT = False
ROUTES = ("online", "learning_disabled", "simple")
HISTORY_DIRECTORY = ".pytest-tmp/persistent-language-action-v7-run"
PRIOR_BANK = "benchmarks/persistent_language_action_v7.result.json"
PRIOR_BANK_SHA256 = "b11b0e0517ef7939cf624e68dfdce05cdd75bc75aa8a81e400ddfd7974fcc67e"
PRIOR_MANIFEST_SHA256 = "c2e1cb6237fdff468c6d413ce62c9dfbf514a607fb1a8af45bdde285c3f4b898"
PRIOR_ATTEMPTS_SHA256 = "ad90077b640f9e652053737fc5ec3fd2d37ff218e97cc7e5da6d4465506f0031"
RESET = "reset_at_episode_start_continue_within_retries"
RETENTION_IDS = (
    *(f"R{i:02d}-{'R' if i % 2 == 0 else 'E'}" for i in range(12)),
    "A04-R",
    "A00-E",
)
ORDER = (
    "A00-R",
    "A00-E",
    "C00-E",
    "F00-R",
    "F00-E",
    "R00-R",
    "R01-E",
    "F01-R",
    "A01-R",
    "A01-E",
    "C01-E",
    "F01-E",
    "F02-R",
    "R02-R",
    "R03-E",
    "C02-E",
    "F02-E",
    "F03-R",
    "C03-E",
    "F03-E",
    "R04-R",
    "R05-E",
    "F04-R",
    "F04-E",
    "F05-R",
    "F05-E",
    "C04-E",
    "F06-R",
    "F06-E",
    "R06-R",
    "R07-E",
    "C05-E",
    "F07-R",
    "F07-E",
    "C06-E",
    "F08-R",
    "F08-E",
    "R08-R",
    "R09-E",
    "C07-E",
    "F09-R",
    "F09-E",
    "R10-R",
    "R11-E",
    "R12-R",
    "R13-E",
    "A02-R",
    "A02-E",
)
CONSTRUCTIONS = (
    "reviewer_counterexample",
    "contract_completion",
    "reverse_question",
    "boundary_partition",
    "negative_condition",
)
COMBINATIONS = (
    "positive_then_first_unique",
    "trim_then_remove_suffix",
    "clamp_then_sum",
    "conditional_choice_then_join",
    "length_order_then_join",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_json(path, expected, reads):
    raw = (ROOT / path).read_bytes()
    if _sha(raw) != expected:
        raise ValueError(f"historical author data changed: {path}")
    reads.append({"path": path, "sha256": expected, "bytes_read": len(raw), "read_passes": 1})
    return json.loads(raw)


def _history():
    reads = []
    bank = _read_json(PRIOR_BANK, PRIOR_BANK_SHA256, reads)
    manifest_path = HISTORY_DIRECTORY + "/manifest.json"
    attempt_path = HISTORY_DIRECTORY + "/attempts.jsonl"
    if (
        bank["evidence"][manifest_path] != PRIOR_MANIFEST_SHA256
        or bank["evidence"][attempt_path] != PRIOR_ATTEMPTS_SHA256
        or bank["observed_attempts"] != 185
        or any(bank["checkpoint"]["journal_bytes_beyond_head"].values())
    ):
        raise ValueError("V7 bank does not bind the completed durable journal prefix")
    previous = _read_json(manifest_path, PRIOR_MANIFEST_SHA256, reads)
    selected = {
        row["episode_id"]: row for row in previous["episodes"] if row["episode_id"] in RETENTION_IDS
    }
    if set(selected) != set(RETENTION_IDS):
        raise ValueError("selected V7 regression roster missing")
    retained = {route: {} for route in ROUTES}
    ordinals = dict.fromkeys(ROUTES, 0)
    digest, byte_count, row_count = hashlib.sha256(), 0, 0
    with (ROOT / attempt_path).open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                raise ValueError("V7 durable journal has a torn row")
            digest.update(raw)
            byte_count += len(raw)
            row_count += 1
            row = json.loads(raw)
            route, identity = row["route"], row["episode_id"]
            if route not in retained or row["schema"] != "persistent-language-action-attempt-v7":
                raise ValueError("V7 actual attempt schema/route changed")
            if identity in selected:
                original = selected[identity]
                expected = {
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
                    or row["observation"] != expected
                ):
                    raise ValueError("selected V7 actual observation differs from intended reset")
                retained[route][identity] = {
                    "episode_id": identity,
                    "attempt_index": 1,
                    "route_ordinal": ordinals[route],
                    "correct": row["correct"],
                    "observation": row["observation"],
                    "attempt_sha256": _sha(raw),
                }
            ordinals[route] += 1
    if (
        digest.hexdigest() != PRIOR_ATTEMPTS_SHA256
        or row_count != 185
        or byte_count != 174106376
        or ordinals != {"online": 62, "learning_disabled": 62, "simple": 61}
        or any(set(rows) != set(RETENTION_IDS) for rows in retained.values())
    ):
        raise ValueError("V7 durable journal/selected completed observations changed")
    bindings = {
        "schema": "persistent-language-action-v8-prior-regression-bindings-v1",
        "path": attempt_path,
        "sha256": PRIOR_ATTEMPTS_SHA256,
        "bytes_read": byte_count,
        "rows_read": row_count,
        "read_passes": 1,
        "route_attempts": ordinals,
        "retained_observations": 42,
        "routes": retained,
        "scope": "actual completed V7 raw observations/outcomes; seven missing attempts excluded",
    }
    reads.append({key: bindings[key] for key in ("path", "sha256", "bytes_read", "read_passes")})
    return previous, bindings, reads


def _documents():
    """One literal sequence, not a corpus generator or target-conditioned source pack."""
    topic_order = (0, 1, 2, 3, 4, 5, 6, 8, 7, 9)
    transient = [("prose", PROSE[i]) for i in topic_order] + [
        ("python", PROGRAMS[i]) for i in topic_order
    ]
    # Interleave kinds and retain one complete note at the start of blocks0..3.
    transient = [transient[i] for pair in zip(range(10), range(10, 20), strict=True) for i in pair]
    rows, index = [], 0
    for block in range(6):
        for slot in range(4):
            admitted = block < 4 and slot == 0
            if admitted:
                kind, text = "factual_note", ADMITTED[block]
            else:
                kind, text = transient[index]
                index += 1
            raw = text.encode("utf-8")
            rows.append(
                {
                    "document_id": f"N{36 + len(rows)}",
                    "text_hex": raw.hex(),
                    "sha256": _sha(raw),
                    "neural_tape_admission": admitted,
                    "source_kind": kind,
                }
            )
    return rows


def _edit(
    identity,
    phase,
    query,
    signature,
    expression,
    driver,
    *,
    teach=False,
    construction="direct_requirement",
    marker=b"pass",
    source=None,
    stdout=b"ok\n",
):
    if source is None:
        source = (signature + "\n    pass\n").encode()
    main = ("# Public read-only driver: main.py\nimport app\n" + driver).encode()
    return _editing(
        identity,
        phase,
        query,
        source,
        marker,
        ("return " + expression).encode(),
        stdout,
        main,
        teach=teach,
        construction=construction,
        structure="missing_function_logic",
    )


def _first_and_revisits():
    reading = _reading(
        "A00-R",
        "acquisition",
        "Use Bay17's receiving note dated 2026-09-18. Give the cold location then the ordinary "
        "location, joined by /, with no other text.",
        b"cedar/maple",
        teach=True,
        construction="direct_dated_requirement",
    )
    editing = _edit(
        "A00-E",
        "acquisition",
        "Implement tickets(n) in app.py: a negative count becomes zero, counts above6 become6, "
        "and other counts stay unchanged. main.py is a protected public driver.",
        "def tickets(n):",
        "max(0,min(n,6))",
        "for n,want in ((-3,0),(0,0),(2,2),(6,6),(11,6)):\n"
        "    got=app.tickets(n)\n"
        "    assert got==want, ('tickets',n,got,want)\nprint('ok')\n",
        teach=True,
    )
    first = [reading, editing]
    for row in first:
        row.update(encounter="first", same_session_origin_id=None, requirement_changed=False)
    rows = list(first)
    paraphrases = (
        "For the earlier Bay17 receiving rule, dated 2026-09-18, write cold/ordinary destinations "
        "as a slash-separated pair. Omit labels and later exceptions.",
        "Complete app.py's tickets so its result stays between0 and6 inclusive and equals n "
        "inside that interval. Leave the protected main.py unchanged.",
    )
    for repeat in (False, True):
        for i, origin in enumerate(first):
            row = copy.deepcopy(origin)
            row.update(
                episode_id=f"A{2 if repeat else 1:02d}-{'R' if i == 0 else 'E'}",
                teacher=None,
                encounter="exact_delayed" if repeat else "changed_form",
                same_session_origin_id=origin["episode_id"],
                exposure_class="same_session_exact" if repeat else "pure_paraphrase",
            )
            if not repeat:
                row["request_hex"] = paraphrases[i].encode().hex()
                row["split_annotation"]["instruction_construction"] = (
                    "unchanged_requirement_paraphrase"
                )
            rows.append(row)
    return rows


def _fresh():
    readings = (
        (
            "A reviewer rejects a report that kept zero. From [-2,3,0,6], list only the "
            "strictly positive values in arrival order, separated by /.",
            b"3/6",
        ),
        (
            "Complete this contract's example: remove one terminal .tmp from report.tmp. "
            "Write the resulting name alone; do not remove letters inside the name.",
            b"report",
        ),
        (
            "Which result belongs in the blank? First appearances from ['z','y','z'] are __. "
            "Give the values in that order joined by /, without list notation.",
            b"z/y",
        ),
        (
            "Partition the inputs -3,2,9 by the ticket desk's allowed interval zero through6. "
            "Return their clamped counts in input order using /.",
            b"0/2/6",
        ),
        (
            "Unless the parcel is fragile, use Bay17's original cold rule. For a fragile cold "
            "parcel followed by a nonfragile cold parcel, give the revised locations using /.",
            b"wicker/cedar",
        ),
        (
            "From [3,-1,3,0,1], keep strictly positive values, then keep each first appearance. "
            "Give the remaining sequence joined with /.",
            b"3/1",
        ),
        (
            "Take '  memo.tmp  ': trim surrounding whitespace and then remove one terminal .tmp. "
            "Return only the resulting name.",
            b"memo",
        ),
        (
            "For the three regional desk orders in the retained note, clamp each count using its "
            "region's cap and zero lower bound, then add the allowed counts. Return one number.",
            b"11",
        ),
        (
            "Choose Bay17's revised locations for a fragile cold parcel and an ordinary noncold "
            "parcel, in that order. Join the two chosen names with a hyphen and nothing else.",
            b"wicker-maple",
        ),
        (
            "Order ['ccc','a','bb'] by increasing string length, then join that ordered sequence "
            "with |. Return only the combined text.",
            b"a|bb|ccc",
        ),
    )
    rows = [
        _reading(
            f"F{i:02d}-R",
            "transfer",
            query,
            target,
            construction=(CONSTRUCTIONS if i < 5 else COMBINATIONS)[i % 5],
        )
        for i, (query, target) in enumerate(readings)
    ]
    edits = [
        _edit(
            "F00-E",
            "transfer",
            "A reviewer rejects positive(xs) if it keeps zero or a negative item. Implement "
            "app.py to retain all strictly positive items in their original order, including "
            "repeats. Keep the protected main.py.",
            "def positive(xs):",
            "[x for x in xs if x>0]",
            "for xs in ([3,-1,3,0,1],[],[2,2]):\n"
            "    got=app.positive(xs)\n"
            "    assert isinstance(got,list) and all(x>0 for x in got), ('positive',got)\n"
            "    print(got)\n",
            stdout=b"[3, 3, 1]\n[]\n[2, 2]\n",
            construction=CONSTRUCTIONS[0],
        ),
        _edit(
            "F01-E",
            "transfer",
            "Complete this contract: trim(s) returns s without surrounding whitespace, "
            "preserving its other characters. Implement app.py; main.py is protected.",
            "def trim(s):",
            "s.strip()",
            "for s in (' Memo.tmp ','','x.tmp.tmp'):\n"
            "    got=app.trim(s)\n"
            "    assert isinstance(got,str) and got==got.strip(), ('trim',got)\n"
            "    print(got)\n",
            stdout=b"Memo.tmp\n\nx.tmp.tmp\n",
            construction=CONSTRUCTIONS[1],
        ),
        _edit(
            "F02-E",
            "transfer",
            "Which implementation belongs in app.py's cap(n)? Its result is n within0..4, "
            "zero below that interval, and4 above it. Supply the missing logic; "
            "main.py is protected.",
            "def cap(n):",
            "max(0,min(n,4))",
            "for n,want in ((-2,0),(0,0),(3,3),(4,4),(9,4)):\n"
            "    got=app.cap(n)\n    assert got==want, ('cap',n,got,want)\nprint('ok')\n",
            construction=CONSTRUCTIONS[2],
        ),
        _edit(
            "F03-E",
            "transfer",
            "Partition join_names(xs)'s behavior by empty, singleton, and longer sequences: "
            "return names joined with :, preserving order and adding no end separators. "
            "Implement app.py; leave main.py intact.",
            "def join_names(xs):",
            "':'.join(xs)",
            "for xs,want in (([],''),(['r'],'r'),(['b','a','b'],'b:a:b')):\n"
            "    got=app.join_names(xs)\n"
            "    assert got==want, ('join',xs,got,want)\nprint('ok')\n",
            construction=CONSTRUCTIONS[3],
        ),
        _edit(
            "F04-E",
            "transfer",
            "Unless fragile is true, location(cold,fragile) follows Bay17's original receiving "
            "rule. Fragile uses the revised exception regardless of cold. Complete app.py "
            "and preserve the protected driver.",
            "def location(cold,fragile):",
            "'wicker' if fragile else 'cedar' if cold else 'maple'",
            "for c,f,want in ((True,False,'cedar'),(False,False,'maple'),"
            "(True,True,'wicker'),(False,True,'wicker')):\n"
            "    got=app.location(c,f)\n"
            "    assert got==want, ('location',c,f,got,want)\nprint('ok')\n",
            construction=CONSTRUCTIONS[4],
        ),
    ]
    # Followers preserve the actual original driver. Its invariant checks remain
    # public; exact printed behavior additionally distinguishes the new contract.
    _project(edits[0], "positive_sequence")
    _project(edits[1], "trimmed_name")
    for index, origin, expression, query, stdout in (
        (
            5,
            edits[0],
            "list(dict.fromkeys(x for x in xs if x>0))",
            "Continue the actual positive project in app.py: retain strictly positive items "
            "and now remove later repeats while keeping first order. Preserve main.py.",
            b"[3, 1]\n[]\n[2]\n",
        ),
        (
            6,
            edits[1],
            "s.strip().removesuffix('.tmp')",
            "Continue the actual trim project in app.py: trim surrounding whitespace, then "
            "remove exactly one terminal .tmp. Preserve spelling and the protected main.py.",
            b"Memo\n\nx.tmp\n",
        ),
    ):
        source = bytes.fromhex(origin["grade"]["expected_after_files"][0]["content_hex"])
        main = bytes.fromhex(origin["files"][1]["content_hex"])
        marker = bytes.fromhex(origin["grade"]["canonical_proposal"]["replacement_hex"])
        row = _editing(
            f"F{index:02d}-E",
            "transfer",
            query,
            source,
            marker,
            ("return " + expression).encode(),
            stdout,
            main,
            construction=COMBINATIONS[index - 5],
            structure="actual_project_revision",
        )
        _project(row, origin["project_id"], origin["episode_id"])
        edits.append(row)
    edits.extend(
        (
            _edit(
                "F07-E",
                "transfer",
                "Implement total(xs) in app.py by clamping each individual count to0..6, then "
                "adding those clamped counts. Do not clamp only the final sum. "
                "Keep main.py protected.",
                "def total(xs):",
                "sum(max(0,min(x,6)) for x in xs)",
                "for xs,want in (([9,-4,2],8),([],0),([6,6],12),([-1],0)):\n"
                "    got=app.total(xs)\n    assert got==want, ('total',xs,got,want)\nprint('ok')\n",
                construction=COMBINATIONS[2],
            ),
            _edit(
                "F08-E",
                "transfer",
                "Implement locations(rows) for (cold,fragile) pairs: choose each Bay17 revised "
                "location, then join the names in row order with /. Empty rows return empty text. "
                "main.py is protected.",
                "def locations(rows):",
                "'/'.join('wicker' if f else 'cedar' if c else 'maple' for c,f in rows)",
                "for rows,want in (([],''),([(True,True),(False,False)],'wicker/maple'),"
                "([(True,False)],'cedar')):\n"
                "    got=app.locations(rows)\n"
                "    assert got==want, ('locations',rows,got,want)\nprint('ok')\n",
                construction=COMBINATIONS[3],
            ),
            _edit(
                "F09-E",
                "transfer",
                "Implement short_line(xs) in app.py: order strings shortest first, retaining "
                "relative order at equal lengths, then join with |. Do not alphabetize ties. "
                "The public main.py is protected.",
                "def short_line(xs):",
                "'|'.join(sorted(xs,key=len))",
                "for xs,want in ((['bbb','a','cc'],'a|cc|bbb'),([],''),(['b','a'],'b|a')):\n"
                "    got=app.short_line(xs)\n"
                "    assert got==want, ('short_line',xs,got,want)\nprint('ok')\n",
                construction=COMBINATIONS[4],
            ),
        )
    )
    rows.extend(edits)
    for row in rows:
        index = int(row["episode_id"][1:3])
        row.update(
            fresh_stratum="instruction_construction" if index < 5 else "structural_combination",
            requirement_changed=bool(row["predecessor_episode_id"]),
            public_check_scope=(
                "unchanged predecessor invariants plus visible output"
                if row["predecessor_episode_id"]
                else "public finite examples; not exhaustive unseen-input proof"
            ),
        )
    return rows


def _recoveries():
    specs = (
        (
            "Repair quota(n): return n limited to the inclusive interval0..6; negatives must "
            "become zero. Use the actual public check diagnostics. main.py is protected.",
            "def quota(n):",
            "min(n,6)",
            "max(0,min(n,6))",
            "for n,w in ((-5,0),(2,2),(12,6)):\n"
            "    g=app.quota(n)\n    assert g==w, ('quota',n,g,w)\n",
        ),
        (
            "Repair clean(s) to trim surrounding whitespace and lowercase the remaining text, "
            "including empty strings. main.py is protected; inspect its actual failures.",
            "def clean(s):",
            "s.lower()",
            "s.strip().lower()",
            "for s,w in ((' OAK ','oak'),('',''),('A B','a b')):\n"
            "    g=app.clean(s)\n    assert g==w, ('clean',s,g,w)\n",
        ),
        (
            "Repair positive(xs): keep strictly positive values, preserving order and duplicates. "
            "Zero is not positive. Do not edit the protected main.py.",
            "def positive(xs):",
            "[x for x in xs if x>=0]",
            "[x for x in xs if x>0]",
            "for xs,w in (([-1,0,2],[2]),([],[]),([3,3],[3,3])):\n"
            "    g=app.positive(xs)\n    assert g==w, ('positive',xs,g,w)\n",
        ),
        (
            "Repair archive(name) to append .bak only when it is not already the terminal "
            "suffix. Preserve other characters and the protected main.py.",
            "def archive(name):",
            "name+'.bak'",
            "name if name.endswith('.bak') else name+'.bak'",
            "for name,w in (('run','run.bak'),('run.bak','run.bak'),('','.bak')):\n"
            "    g=app.archive(name)\n    assert g==w, ('archive',name,g,w)\n",
        ),
        (
            "Repair firsts(xs) so duplicates disappear but first-appearance order remains. "
            "Alphabetic sorting changes the contract. Keep main.py protected.",
            "def firsts(xs):",
            "sorted(set(xs))",
            "list(dict.fromkeys(xs))",
            "for xs,w in ((['z','a','z'],['z','a']),([],[]),(['x','x'],['x'])):\n"
            "    g=app.firsts(xs)\n    assert g==w, ('firsts',xs,g,w)\n",
        ),
        (
            "Repair odd_doubles(xs): retain odd integers and double each retained value, in "
            "input order. Negative odd integers qualify too. main.py is protected.",
            "def odd_doubles(xs):",
            "[2*x for x in xs if x%2==0]",
            "[2*x for x in xs if x%2]",
            "for xs,w in (([1,2,-3],[2,-6]),([],[]),([0,4],[])):\n"
            "    g=app.odd_doubles(xs)\n    assert g==w, ('odd_doubles',xs,g,w)\n",
        ),
        (
            "Repair line(xs): trim and lowercase every string, then join in the given order "
            "with commas. Empty input produces empty text. Leave main.py unchanged.",
            "def line(xs):",
            "','.join(xs).lower()",
            "','.join(s.strip().lower() for s in xs)",
            "for xs,w in (([' A ','B '],'a,b'),([],''),([' '],'')):\n"
            "    g=app.line(xs)\n    assert g==w, ('line',xs,g,w)\n",
        ),
        (
            "Repair total(xs) by limiting each individual value to0..6 before adding. "
            "Capping the final sum loses allowed values. Keep main.py protected.",
            "def total(xs):",
            "max(0,min(sum(xs),6))",
            "sum(max(0,min(x,6)) for x in xs)",
            "for xs,w in (([6,6],12),([-3,4],4),([],0)):\n"
            "    g=app.total(xs)\n    assert g==w, ('total',xs,g,w)\n",
        ),
    )
    rows = []
    for i, (query, signature, broken, fixed, driver) in enumerate(specs):
        marker = ("return " + broken).encode()
        row = _edit(
            f"C{i:02d}-E",
            "recovery",
            query,
            signature,
            fixed,
            driver + "print('ok')\n",
            marker=marker,
            source=(signature + "\n    return " + broken + "\n").encode(),
            construction="public_assertion_repair",
        )
        row.update(
            feedback_policy="actual previous workspace and raw execution diagnostics",
            correction_policy="none; fixed three attempts independent of grade",
            public_check_scope="read-only executable finite assertion examples",
        )
        rows.append(row)
    return rows


def _retention(previous):
    old = {row["episode_id"]: row for row in previous["episodes"]}
    rows = []
    for index, identity in enumerate(RETENTION_IDS):
        row = copy.deepcopy(old[identity])
        row.update(
            episode_id=f"R{index:02d}-{'R' if index % 2 == 0 else 'E'}",
            phase="retention",
            teacher=None,
            append_document_ids=[],
            attempt_limit=1,
            project_id=None,
            predecessor_episode_id=None,
            workspace_lifecycle=RESET,
            exposure_class="prior_completed_v7_regression",
            encounter="prior_regression",
            same_session_origin_id=None,
            prior_episode_id=identity,
            prior_session="v7",
            requirement_changed=False,
        )
        row.pop("intervening_attempts", None)
        rows.append(row)
    return rows


def _validate(documents, episodes, previous):
    if (
        len(documents) != 24
        or documents != _documents()
        or tuple(row["episode_id"] for row in episodes) != ORDER
        or len(set(ORDER)) != 48
    ):
        raise ValueError("fixed authored sources or schedule changed")
    all_source = []
    for document in documents:
        raw = bytes.fromhex(document["text_hex"])
        if (
            raw.hex() != document["text_hex"]
            or not 0 < len(raw) <= 3484
            or _sha(raw) != document["sha256"]
            or type(document["neural_tape_admission"]) is not bool
        ):
            raise ValueError("source bytes, bounds or admission declaration changed")
        if document["source_kind"] == "python":
            ast.parse(raw.decode("utf-8"), filename=document["document_id"] + ".py")
        all_source.append(raw)
    if (
        sum(len(raw) + 1 for raw in all_source) > 10500
        or sum(
            len(raw)
            for raw, doc in zip(all_source, documents, strict=True)
            if doc["neural_tape_admission"]
        )
        > 400
        or sum(doc["neural_tape_admission"] for doc in documents) != 4
    ):
        raise ValueError("new source/admitted factual ceiling exceeded")
    seen, attempt_ordinal, source_ids, projects = {}, 0, [], {}
    previous_by_id = {row["episode_id"]: row for row in previous["episodes"]}
    for ordinal, row in enumerate(episodes):
        identity = row["episode_id"]
        query = bytes.fromhex(row["request_hex"])
        if query.hex() != row["request_hex"] or not 0 < len(query) <= 256:
            raise ValueError("authored request exceeds raw interface or canonical hex")
        expected_sources = [f"N{36 + ordinal // 8 * 4 + ordinal % 8}"] if ordinal % 8 < 4 else []
        if row["append_document_ids"] != expected_sources:
            raise ValueError("one whole source before the first four rows of each block required")
        source_ids.extend(row["append_document_ids"])
        if row["attempt_limit"] != (3 if row["phase"] == "recovery" else 1):
            raise ValueError("attempt policy must be fixed independently of grade")
        if row["teacher"] is not None and identity not in ("A00-R", "A00-E"):
            raise ValueError("only the two first responses may receive corrections")
        if row["kind"] == "reading":
            if row["files"] or row["grade"]["kind"] != "exact_answer":
                raise ValueError("reading observation/private grade contract changed")
            for value in row["grade"]["accepted_answer_hex"]:
                raw = bytes.fromhex(value)
                if raw.hex() != value or len(raw) > 128:
                    raise ValueError("reading answer must retain the raw output bound")
        else:
            files = row["files"]
            names = [item["path"] for item in files]
            if (
                not 1 <= len(files) <= 4
                or len(set(names)) != len(names)
                or "main.py" not in names
                or "main.py" not in row["grade"]["protected_paths"]
            ):
                raise ValueError("edit needs a public protected driver and distinct bounded files")
            for item in files + row["grade"]["expected_after_files"]:
                path, raw = item["path"], bytes.fromhex(item["content_hex"])
                if len(path.encode()) > 31 or len(raw) > 512 or raw.hex() != item["content_hex"]:
                    raise ValueError("author workspace exceeds the unchanged raw interface")
                if path.endswith(".py"):
                    ast.parse(raw.decode(), filename=identity + "/" + path)
            before = {item["path"]: bytes.fromhex(item["content_hex"]) for item in files}
            after = {
                item["path"]: bytes.fromhex(item["content_hex"])
                for item in row["grade"]["expected_after_files"]
            }
            proposal = row["grade"]["canonical_proposal"]
            replacement = bytes.fromhex(proposal["replacement_hex"])
            path, start, end = proposal["path"], proposal["start"], proposal["end"]
            if (
                proposal["operation"] != "SPLICE"
                or path == "main.py"
                or path not in before
                or not 0 <= start <= end <= len(before[path])
                or len(replacement) > 128
                or proposal["stop_reason"] != "eos"
            ):
                raise ValueError("nominal author witness must be one bounded implementation splice")
            predicted = dict(before)
            predicted[path] = before[path][:start] + replacement + before[path][end:]
            if predicted != after or any(
                after[name] != before[name] for name in row["grade"]["protected_paths"]
            ):
                raise ValueError("nominal raw splice/protected driver math mismatch")
            if identity[0] != "R":
                driver = ast.parse(before["main.py"].decode())
                if not any(isinstance(node, ast.Assert) for node in ast.walk(driver)):
                    raise ValueError("new tasks need actual executable public assertions")
            if row["teacher"] is not None and row["teacher"] != {"kind": "edit", **proposal}:
                raise ValueError("single edit correction differs from its declared witness")
            if row.get("fresh_stratum") == "structural_combination" and any(
                replacement in raw for raw in all_source
            ):
                raise ValueError("fresh composite solution is already present in a source")
        origin_id = row.get("same_session_origin_id")
        if origin_id is not None:
            before_ordinal, origin = seen[origin_id]
            exact = row["encounter"] == "exact_delayed"
            if (
                row["kind"] != origin["kind"]
                or row["files"] != origin["files"]
                or row["grade"] != origin["grade"]
                or row["requirement_changed"]
                or (row["request_hex"] == origin["request_hex"]) != exact
            ):
                raise ValueError("changed-form/exact revisit changes requirements or observation")
            row["intervening_attempts"] = attempt_ordinal - before_ordinal - 1
        if row["project_id"]:
            projects.setdefault(row["project_id"], []).append(identity)
        predecessor = row["predecessor_episode_id"]
        if predecessor:
            origin = seen[predecessor][1]
            if (
                row["workspace_lifecycle"] != "carry_actual_project"
                or row["nominal_files_scope"] != "parent_only_intended_reference_never_materialize"
                or row["files"] != origin["grade"]["expected_after_files"]
            ):
                raise ValueError("project follower requires actual carry, not teacher substitution")
        elif row["workspace_lifecycle"] != RESET:
            raise ValueError("independent observation must use its declared reset")
        if row["phase"] == "retention":
            old = previous_by_id[row["prior_episode_id"]]
            if any(row[key] != old[key] for key in ("kind", "request_hex", "files", "grade")):
                raise ValueError("exposed regression raw input/grade changed")
        seen[identity] = (attempt_ordinal, row)
        attempt_ordinal += row["attempt_limit"]
    if source_ids != [doc["document_id"] for doc in documents]:
        raise ValueError("all sources must be observed exactly once in literal order")
    if projects != {"positive_sequence": ["F00-E", "F05-E"], "trimmed_name": ["F01-E", "F06-E"]}:
        raise ValueError("two fixed actual two-stage project chains required")
    if [
        row["intervening_attempts"] for row in episodes if row.get("encounter") == "exact_delayed"
    ] != [61, 61]:
        raise ValueError("frozen exact-repeat attempt age changed")


def _counts(documents, episodes):
    teachers = [row for row in episodes if row["teacher"] is not None]
    answer_raw = sum(
        len(bytes.fromhex(row["teacher"].get("target_hex", row["teacher"].get("replacement_hex"))))
        for row in teachers
    )
    admitted = [doc for doc in documents if doc["neural_tape_admission"]]
    transient = [doc for doc in documents if not doc["neural_tape_admission"]]

    def raw(rows):
        return sum(len(bytes.fromhex(doc["text_hex"])) for doc in rows)

    # Parent-only nominal observation accounting, not a fabricated actual scan.
    nominal_scans = {
        row["episode_id"]: len(bytes.fromhex(row["request_hex"]))
        + 1
        + 13
        + sum(
            len(item["path"].encode()) + 1 + len(bytes.fromhex(item["content_hex"])) + 1
            for item in row["files"]
        )
        for row in episodes
    }
    counts = {
        "episodes": len(episodes),
        "teacher_offers": len(teachers),
        "phase_episodes": {
            phase: sum(row["phase"] == phase for row in episodes)
            for phase in ("acquisition", "transfer", "retention", "recovery")
        },
        "maximum_episode_attempts": sum(row["attempt_limit"] for row in episodes),
        "maximum_action_attempts": sum(
            row["attempt_limit"] for row in episodes if row["kind"] == "edit"
        ),
        "canonical_edit_records": sum(row["kind"] == "edit" for row in episodes),
        "distinct_new_canonical_workspaces": len(
            {
                tuple(
                    (item["path"], item["content_hex"])
                    for item in row["grade"]["expected_after_files"]
                )
                for row in episodes
                if row["kind"] == "edit" and row["phase"] != "retention"
            }
        ),
        "broken_recovery_workspaces": sum(row["phase"] == "recovery" for row in episodes),
        "historical_edit_records_not_replayed": sum(
            row["kind"] == "edit" and row["phase"] == "retention" for row in episodes
        ),
        "author_children_authorized": 0,
        "reader_teacher_raw_bytes": answer_raw,
        "reader_teacher_eos": len(teachers),
        "reader_teacher_symbols": answer_raw + len(teachers),
        "action_teacher_labels": 3,
        "document_count": len(documents),
        "document_raw_bytes": raw(documents),
        "document_eos": len(documents),
        "source_next_byte_teacher_symbols": raw(documents) + len(documents),
        "admitted_documents": len(admitted),
        "admitted_raw_bytes": raw(admitted),
        "admitted_eos": len(admitted),
        "new_memory_slots": raw(admitted) + len(admitted),
        "transient_documents": len(transient),
        "transient_raw_bytes": raw(transient),
        "transient_eos": len(transient),
        "initial_occupied_memory_slots": 1614,
        "occupied_memory_slots": 1614 + raw(admitted) + len(admitted),
        "source_plus_answer_symbol_offers": raw(documents)
        + len(documents)
        + answer_raw
        + len(teachers),
        "public_scan_nominal_offers_by_episode": nominal_scans,
        "public_scan_nominal_total_offers": sum(
            nominal_scans[row["episode_id"]] * row["attempt_limit"] for row in episodes
        ),
        "public_scan_flags_per_attempt": 13,
        "public_scan_interface_upper_bound": 64 * 3485,
        "public_scan_count_scope": (
            "nominal reset/no-feedback bytes; actual carry/retry diagnostics separately counted"
        ),
        "dependent_project_followers": 2,
        "maximum_safe_boundaries": 265,
        "maximum_neural_payload_versions": 178,
    }
    if (
        counts["phase_episodes"]
        != {"acquisition": 6, "transfer": 20, "retention": 14, "recovery": 8}
        or len(teachers) != 2
        or answer_raw > 64
        or counts["maximum_episode_attempts"] != 64
        or counts["maximum_action_attempts"] != 44
        or counts["canonical_edit_records"] != 28
        or len(admitted) != 4
        or len(transient) != 20
        or raw(admitted) > 400
        or not 9500 <= counts["source_next_byte_teacher_symbols"] <= 10500
    ):
        raise ValueError("fixed V8 roster/exposure bound changed")
    return counts


def _fresh_split(episodes, documents, previous, reads):
    older = list(previous["episodes"])
    for version, digest in previous["provenance"]["exposed_history_manifest_sha256"].items():
        if not re.fullmatch(r"v[1-6]", version):
            raise ValueError("unexpected exposed history manifest version")
        path = f".pytest-tmp/persistent-language-action-{version}-run/manifest.json"
        older.extend(_read_json(path, digest, reads)["episodes"])
    acquisition = [row for row in episodes if row["phase"] == "acquisition"]
    history = older + acquisition

    def tokenize(value):
        return set(re.findall(rb"[a-z0-9]+", value.lower()))

    history_tokens = [tokenize(bytes.fromhex(row["request_hex"])) for row in history]
    source_raw = [bytes.fromhex(doc["text_hex"]) for doc in documents]
    source_tokens = [tokenize(raw) for raw in source_raw]
    overlap = []
    for row in (row for row in episodes if row["phase"] == "transfer"):
        query = bytes.fromhex(row["request_hex"])
        if any(row["request_hex"] == old["request_hex"] for old in history) or any(
            query in raw for raw in source_raw
        ):
            raise ValueError("fresh complete request repeats exposed wording")
        words = tokenize(query)
        overlap.append(
            {
                "episode_id": row["episode_id"],
                "maximum_exposed_request_jaccard": max(
                    len(words & prior) / len(words | prior) for prior in history_tokens
                ),
                "maximum_source_jaccard": max(
                    len(words & prior) / len(words | prior) for prior in source_tokens
                ),
                "annotation_name_seen_in_older_history": any(
                    row["split_annotation"]["instruction_construction"]
                    == old.get("split_annotation", {}).get("instruction_construction")
                    for old in older
                ),
            }
        )
    return {
        "lexical_overlap": overlap,
        "overlap_history_scope": (
            "authenticated v1-v7 manifests, current acquisition, all new literal sources"
        ),
        "current_answer_construction_holdouts": list(CONSTRUCTIONS),
        "structural_combinations": list(COMBINATIONS),
        "fully_withheld_wording_families_claimed": False,
        "annotation_labels_are_author_judgment": True,
        "changed_forms_also_change_requirements": False,
        "same_session_exact_intervening_attempts": 61,
        "claim": (
            "whole new requests and unshown composite programs; "
            "historical semantic overlap retained"
        ),
        "public_check_scope": (
            "finite readable examples, not independent hidden program-generalization tests"
        ),
    }


def build_stream():
    """One pure build after source review: static AST/byte checks and saved JSON only."""
    previous, bindings, reads = _history()
    prior_documents = copy.deepcopy(
        previous["continuation"]["prior_documents"] + previous["documents"]
    )
    if (
        len(prior_documents) != 36
        or sum(len(bytes.fromhex(doc["text_hex"])) + 1 for doc in prior_documents) != 1614
    ):
        raise ValueError("V7 factual/source lineage changed")
    documents = _documents()
    rows = _first_and_revisits() + _fresh() + _recoveries() + _retention(previous)
    by_id = {row["episode_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate authored episode identity")
    episodes = [by_id[identity] for identity in ORDER]
    for ordinal, row in enumerate(episodes):
        row["append_document_ids"] = (
            [f"N{36 + ordinal // 8 * 4 + ordinal % 8}"] if ordinal % 8 < 4 else []
        )
        row["supported"], row["unsupported_dimension"] = True, None
    _validate(documents, episodes, previous)
    split = _fresh_split(episodes, documents, previous, reads)
    return {
        "schema": SCHEMA,
        "documents": documents,
        "episodes": episodes,
        "counts": _counts(documents, episodes),
        "prior_regression_bindings": bindings,
        "capacity": {
            **previous["capacity"],
            "memory_slots": 2048,
            "event_owners": 1024,
            "transient_source_raw_bytes": 3484,
            "controller_public_addresses": 3485,
        },
        "provenance": {
            "kind": "agent_authored_open_development",
            "protected_evaluation": False,
            "official_exam": False,
            "source_sha256": _sha(Path(__file__).read_bytes()),
            "literal_source_path": (
                "experiments/r4_integrated/persistent_language_action_v8_sources.py"
            ),
            "literal_source_sha256": _sha(
                Path(__file__).with_name("persistent_language_action_v8_sources.py").read_bytes()
            ),
            "prior_manifest_sha256": PRIOR_MANIFEST_SHA256,
            "exposed_history_manifest_sha256": {
                **previous["provenance"]["exposed_history_manifest_sha256"],
                "v7": PRIOR_MANIFEST_SHA256,
            },
            "authenticated_author_reads": reads,
            "witness_policy": (
                "builder AST/raw splice checks only; no author child execution released; "
                "historical programs and duplicate revisit witnesses need no replay"
            ),
        },
        "continuation": {
            "prior_run_directory": HISTORY_DIRECTORY,
            "prior_bank_path": PRIOR_BANK,
            "prior_bank_sha256": PRIOR_BANK_SHA256,
            "selected_neural_route": "online",
            "prior_documents": prior_documents,
            "initial_memory_slots": 1614,
            "baseline_prior_documents": 36,
            "baseline_prior_edit_demonstrations": 48,
            "baseline_predict_replays": 0,
            "missing_v7_attempts_reconstructed": 0,
            "new_learning_state": (
                "zero recurrent controller/baselines; seed733; all old numerical prior held"
            ),
            "control_policy": (
                "same V7 online; disabled controller/action weights and baselines; "
                "actual RNG advances; factual writes matched"
            ),
            "source_learning": (
                "one causal rawbyte/EOS pass per document; "
                "neural_tape_admission selects retained versus transient"
            ),
            "baseline_source_retention": (
                "all24 new observed sources retained for transparent lexical reference"
            ),
            "public_feedback_learning": True,
        },
        "fresh_split": split,
        "public_execution_policy": {
            "protected_paths": ["main.py"],
            "rejection": "fixed public policy; proposed driver modification is not applied",
            "controller_reward": "actual failure=-1; otherwise0; unused STOP payload0",
        },
        "persistence": {
            "weights_and_shared_tape": (
                "continue authenticated V7 online safe state; no historical source replay"
            ),
            "workspace": "two actual two-stage projects; other starts reset; retries carry actual",
            "nominal_chain_files": (
                "parent-only witness; never substitute for actual predecessor files"
            ),
            "safe_boundaries": (
                "initialization and every resolved route source/attempt transaction; "
                "optional teacher co-publishes with its attempt"
            ),
            "interrupted_intent": (
                "last safe boundary; no automatic replay or missing-outcome reconstruction"
            ),
            "payload_storage": (
                "existing full actor serialization, not a split prior/controller store"
            ),
        },
    }


def write_stream(path: Path):
    path = Path(path)
    if path.exists():
        raise ValueError("manifest destination already exists")
    stream = build_stream()
    raw = (json.dumps(stream, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return {
        "path": str(path),
        "sha256": _sha(raw),
        "bytes": len(raw),
        "counts": stream["counts"],
        "task_builds": 1,
        "model_constructions": 0,
        "checkpoint_loads": 0,
        "children": 0,
        "authenticated_author_reads": stream["provenance"]["authenticated_author_reads"],
    }


def check_saved_manifest(path: Path, expected_sha256: str):
    """Check saved literal contract and authenticated actual history; no new build."""
    raw = Path(path).read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("frozen V8 manifest hash mismatch")
    manifest = json.loads(raw)
    if (
        manifest["schema"] != SCHEMA
        or manifest["provenance"]["source_sha256"] != _sha(Path(__file__).read_bytes())
        or manifest["provenance"]["literal_source_sha256"]
        != _sha(Path(__file__).with_name("persistent_language_action_v8_sources.py").read_bytes())
    ):
        raise ValueError("frozen V8 manifest source/schema mismatch")
    previous, bindings, _ = _history()
    rows = copy.deepcopy(manifest["episodes"])
    _validate(manifest["documents"], rows, previous)
    if (
        rows != manifest["episodes"]
        or bindings != manifest["prior_regression_bindings"]
        or manifest["counts"] != _counts(manifest["documents"], rows)
    ):
        raise ValueError("saved schedule/history/exposure contract changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if DRAFT:
        raise SystemExit("V8 author CLI held for source review; runtime not released")
    if args.command == "build":
        result = write_stream(args.path)
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


__all__ = ["build_stream", "check_saved_manifest", "observation", "write_stream"]


if __name__ == "__main__":
    main()
