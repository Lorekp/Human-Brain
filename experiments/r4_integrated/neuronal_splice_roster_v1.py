"""Frozen byte-splice task data; no neuronal, runner or executable-code imports.

Teachers and expected results are parent/evaluator data. Only task_observation's
explicit raw fields may cross into the endpoint worker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "neuronal-splice-roster-v1"
OLD_ROSTER_SHA256 = "009b42e22976733bba207e46337ec23d5f6de5ab15ab061bc6fd08539e44a2b1"
R = tuple(bytes.fromhex(value) for value in (
    "6162", "6261", "6263", "6362", "615c2762", "635c2764", "6364", "6463"))
INSERTION_PAYLOADS = (b"aa", b"bb", b"cc", b"dd", b"ab", b"ba", b"cd", b"dc")
P = R + INSERTION_PAYLOADS + (b"",) * 8 + INSERTION_PAYLOADS
LAYOUTS = {
    "A": (b"", ("x", "y")),
    "B": (b"# pad\n", ("y", "x")),
    "C": (b"# fresh 0123456789\n\n", ("x", "y")),
    "D": (b"# fresh 0123456789\n\n", ("y", "x")),
}
QUOTED_DISPLAY = {
    bytes.fromhex(raw): bytes.fromhex(display) for raw, display in (
        ("615c2762", "612762"), ("635c2764", "632764"),
        ("625c2761", "622761"), ("645c2763", "642763"),
        ("615c2764", "612764"), ("635c2761", "632761"),
        ("625c2763", "622763"), ("645c2761", "642761"),
    )
}


def _key(index: int) -> bytes:
    return f"k{index:02d}".encode("ascii")


def _document(key: bytes, payload: bytes) -> bytes:
    return b"Record " + key + b" has value " + payload + b"."


def _display(payload: bytes) -> bytes:
    """Only the frozen authored literal/display pairs, outside free encoding."""
    if payload in QUOTED_DISPLAY:
        return QUOTED_DISPLAY[payload]
    if any(byte in payload for byte in (b"\\", b"'", b"\n", b"\r")):
        raise ValueError("unlisted authored Python fragment/display pair")
    return payload


def _files_record(files: list[tuple[str, bytes]]) -> list[dict]:
    return [{"path": path, "content_hex": raw.hex()} for path, raw in files]


def workspace_digest(files: list[dict]) -> str:
    """Exact generic raw ABI digest, without importing the neuronal package."""
    ordered = sorted(files, key=lambda item: item["path"].encode("utf-8"))
    digest = hashlib.sha256(b"humanbrain.raw-splice-workspace.v1\0")
    digest.update(len(ordered).to_bytes(4, "big"))
    for item in ordered:
        for raw in (item["path"].encode("utf-8"), bytes.fromhex(item["content_hex"])):
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _task(task_id: str, group: str, memory_id: str, paths: tuple[str, str],
          target: str, variable: str, key: bytes, layout: str, interior: bytes,
          payload: bytes, *, keep: bool = False, pair_id: str | None = None,
          member: int | None = None, reverse_files: bool = False) -> dict:
    prefix, variables = LAYOUTS[layout]
    lines = {name: name.encode() + b" = '" + interior + b"'\n" for name in ("x", "y")}
    data = prefix + b"".join(lines[name] for name in variables)
    # Offsets follow construction, not a search over candidate occurrences.
    start = len(prefix) + 5 + sum(
        len(lines[name]) for name in variables[:variables.index(variable)])
    end = start + len(interior)
    a, b = (path[:-3] for path in paths)
    main = (f"import {a}, {b}\n"
            f"print({a}.x, {a}.y, {b}.x, {b}.y, sep='|')\n").encode("ascii")
    before = sorted([(paths[0], data), (paths[1], data), ("main.py", main)])
    request = (b"KEEP" if keep else b"SET") + b" " + target.encode() + b" " + variable.encode()
    request += b" " + key
    if len(request) > 31 or len(payload) > 16 or any(
        len(path.encode("utf-8")) > 31 or len(raw) > 128 for path, raw in before
    ):
        raise ValueError("authored task exceeds the selected input geometry")
    recorded = _files_record(list(reversed(before)) if reverse_files else before)
    workspace_hash = workspace_digest(recorded)
    operation = "STOP" if keep else "SPLICE"
    teacher = {
        "operation": operation, "path": "" if keep else target,
        "start": 0 if keep else start, "end": 0 if keep else end,
        "replacement_hex": "" if keep else payload.hex(), "stop_reason": "eos",
        "workspace_sha256": workspace_hash,
    }
    after = [(path, raw if keep or path != target else raw[:start] + payload + raw[end:])
             for path, raw in before]
    displayed = [_display(interior)] * 4
    if not keep:
        displayed[2 * paths.index(target) + (variable == "y")] = _display(payload)
    return {
        "task_id": task_id, "group": group, "pair_id": pair_id, "member": member,
        "memory_id": memory_id, "request_hex": request.hex(), "files": recorded,
        "workspace_sha256": workspace_hash, "teacher": teacher,
        "expected_after_files": _files_record(after),
        "expected_stdout_hex": (b"|".join(displayed) + b"\n").hex(),
        "required_operation": operation,
    }


def _package(packages: dict, memory_id: str, documents: list[bytes], stage: str) -> None:
    if memory_id in packages or stage not in {"before_learning", "after_learning"}:
        raise ValueError("duplicate memory package or unknown observation stage")
    if not documents or sum(len(raw) + 1 for raw in documents) > 128:
        raise ValueError("authored document package exceeds the fixed tape")
    packages[memory_id] = {"memory_id": memory_id, "stage": stage,
                           "document_hex": [raw.hex() for raw in documents]}


def _pair_documents(key: int, payload: bytes, decoy_key: int,
                    decoy_payload: bytes, reverse: bool) -> list[bytes]:
    documents = [_document(_key(key), payload), _document(_key(decoy_key), decoy_payload)]
    return list(reversed(documents)) if reverse else documents


def _new_tasks() -> tuple[dict, dict]:
    tasks, packages = {}, {}
    for t in range(32):
        r, task_id = t % 8, f"T{t:02d}"
        memory_id = f"M-{task_id}"
        _package(packages, memory_id, _pair_documents(
            t, P[t], (t + 9) % 32, P[(t + 9) % 32], bool(t % 2)), "before_learning")
        tasks[task_id] = _task(
            task_id, ("replace", "insert", "delete", "keep")[t // 8], memory_id,
            ("a0.py", "b0.py"), "a0.py" if r % 2 == 0 else "b0.py",
            "x" if (r // 2) % 2 == 0 else "y", _key(t), "A" if r < 4 else "B",
            b"" if 8 <= t < 16 else b"old", P[t], keep=t >= 24)

    for category in ("P", "L", "V", "O"):
        for r in range(4):
            pair_id = f"{category}{r}"
            parity_layout = "C" if r % 2 == 0 else "D"
            parity_variable = "x" if r % 2 == 0 else "y"
            if category == "P":
                paths, key = (f"a{r + 1}.py", f"b{r + 1}.py"), 32 + r
                payload = (b"ac", b"ca", b"bd", b"db")[r]
                _package(packages, f"M-{pair_id}", _pair_documents(
                    key, payload, 52 + r, payload[::-1], bool(r % 2)), "after_learning")
            elif category == "L":
                paths, key = (f"m{r}.py", f"n{r}.py"), 36 + r
                payload = (b"ad", b"da", b"bc", b"cb")[r]
                _package(packages, f"M-{pair_id}", _pair_documents(
                    key, payload, 56 + r, payload[::-1], bool(r % 2)), "after_learning")
            elif category == "V":
                paths, key = (f"p{r}.py", f"q{r}.py"), 40 + r
                values = ((b"ac", b"ca"), (b"bd", b"db"),
                          (bytes.fromhex("625c2761"), bytes.fromhex("645c2763")),
                          (bytes.fromhex("615c2764"), bytes.fromhex("635c2761")))[r]
            else:
                paths, key = (f"u{r}.py", f"v{r}.py"), 44 + r
                payload = (b"ad", b"da", b"", bytes.fromhex("625c2763"))[r]
                decoy = (b"da", b"ad", b"ab", bytes.fromhex("645c2761"))[r]
                _package(packages, f"M-{pair_id}", _pair_documents(
                    key, payload, 64 + r, decoy, bool(r % 2)), "after_learning")
            for member in range(2):
                task_id = f"{pair_id}-{member}"
                memory_id = f"M-{pair_id}"
                layout, variable, interior, keep = parity_layout, parity_variable, b"old", False
                target = paths[0]
                if category == "P":
                    target = paths[member]
                elif category == "L":
                    target = paths[r % 2]
                    layout = ("C" if member == 0 else "D") if r < 2 else "C"
                    variable = "x" if r < 2 or member == 0 else "y"
                elif category == "V":
                    memory_id += f"-{member}"
                    payload = values[member]
                    documents = _pair_documents(key, payload, 60 + r, values[1 - member], False)
                    if r >= 2:
                        documents.extend(_document(_key(68 + 3 * (r - 2) + j), value)
                                         for j, value in enumerate((b"cc", b"cd", b"dd")))
                    _package(packages, memory_id, documents,
                             "before_learning" if r < 2 else "after_learning")
                else:
                    target, keep = paths[r % 2], bool(member)
                    interior = b"" if r == 1 else b"old"
                tasks[task_id] = _task(
                    task_id, category, memory_id, paths, target, variable, _key(key), layout,
                    interior, payload, keep=keep, pair_id=pair_id, member=member,
                    reverse_files=category == "P" and member == 1)
    return tasks, packages


def _load_old_panel(path: Path, packages: dict) -> tuple[list[dict], dict]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != OLD_ROSTER_SHA256:
        raise ValueError("old reader panel must use the exact authenticated v6 roster")
    checks = json.loads(raw)["checks"]
    if len(checks) != 20 or [row["id"] for row in checks] != list(range(20)):
        raise ValueError("old reader panel case ordering changed")
    panel = []
    for row in checks:
        memory_id = "M-old-" + row["memory_id"]
        documents = [bytes.fromhex(value) for value in row["document_hex"]]
        if memory_id not in packages:
            _package(packages, memory_id, documents, "before_learning")
        elif packages[memory_id]["document_hex"] != row["document_hex"]:
            raise ValueError("old cases disagree on a shared memory's literal bytes")
        panel.append({"case_id": row["id"], "memory_id": memory_id,
                      "query_hex": row["query_hex"], "expected_hex": row["expected_hex"],
                      "max_bytes": 2, "originally_correct": row["id"] not in (16, 18)})
    old_ids = {"M-old-" + group + f"-{index}" for group in ("fit", "novel", "interference")
               for index in range(4)}
    if {item["memory_id"] for item in panel} != old_ids:
        raise ValueError("old reader panel memory scope changed")
    return panel, {"old_roster_path": str(path), "old_roster_sha256": actual,
                   "old_roster_bytes": len(raw)}


def task_observation(task: dict) -> dict:
    """The only task projection intended for teacher-free action inference."""
    return {"task_id": task["task_id"], "memory_id": task["memory_id"],
            "request_hex": task["request_hex"],
            "files": [{"path": item["path"], "content_hex": item["content_hex"]}
                      for item in task["files"]],
            "workspace_sha256": task["workspace_sha256"]}


def build_roster(old_roster_path: Path) -> dict:
    """Build the selected parent-side roster, authenticating old data without imports."""
    tasks, packages = _new_tasks()
    panel, source = _load_old_panel(old_roster_path, packages)
    train = [f"T{index:02d}" for index in range(32)]
    fresh = [f"{group}{r}-{member}" for group in ("P", "L", "V", "O")
             for r in range(4) for member in range(2)]
    retention = [f"T{index:02d}" for base in (0, 8, 16, 24) for index in range(base, base + 4)]
    schedule = [f"T{(index + 4 * repeat) % 32:02d}" for repeat in range(8) for index in range(32)]
    before = [f"M-{task_id}" for task_id in train]
    before += [f"M-V{r}-{member}" for r in range(2) for member in range(2)]
    before += list(dict.fromkeys(item["memory_id"] for item in panel))
    after = [memory_id for memory_id in packages
             if packages[memory_id]["stage"] == "after_learning"]
    document_hex = [raw for package in packages.values() for raw in package["document_hex"]]
    raw_bytes = sum(len(bytes.fromhex(raw)) for raw in document_hex)
    payload_offers = sum(
        len(bytes.fromhex(tasks[task_id]["teacher"]["replacement_hex"])) + 1
        for task_id in schedule if tasks[task_id]["required_operation"] == "SPLICE")
    action_labels = sum(3 if tasks[task_id]["required_operation"] == "SPLICE" else 1
                        for task_id in schedule)
    counts = {
        "tasks": len(tasks), "train_tasks": len(train), "fresh_tasks": len(fresh),
        "retention_tasks": len(retention), "lessons_per_arm": len(schedule),
        "memory_packages_per_arm": len(packages), "before_learning_packages": len(before),
        "after_learning_packages": len(after), "documents_per_arm": len(document_hex),
        "document_raw_bytes_per_arm": raw_bytes, "document_eos_per_arm": len(document_hex),
        "ingestion_symbols_per_arm": raw_bytes + len(document_hex),
        "maximum_memory_slots": max(sum(len(bytes.fromhex(raw)) + 1
                                          for raw in value["document_hex"])
                                    for value in packages.values()),
        "reader_teacher_offers_per_arm": payload_offers,
        "action_teacher_labels_per_arm": action_labels,
        "full_action_verifications": 4 * (len(schedule) + len(fresh) + len(retention)),
        "old_panel_reader_calls": 4 * 2 * len(panel),
        "full_reader_teacher_offers": 4 * payload_offers,
        "full_action_teacher_labels": 4 * action_labels,
    }
    expected = (64, 64, 48, 16, 152, 3490, 124, 480, 640)
    actual = (len(tasks), len(packages), len(before), len(after), len(document_hex), raw_bytes,
              counts["maximum_memory_slots"], payload_offers, action_labels)
    if actual != expected or set(before + after) != set(packages):
        raise ValueError(f"frozen roster arithmetic mismatch: actual={actual}, expected={expected}")
    return {"schema": SCHEMA, "source_inputs": source, "tasks": tasks,
            "memory_packages": packages, "train_task_ids": train, "schedule": schedule,
            "cost_task_ids": ["T00", "T08", "T16", "T24"], "fresh_task_ids": fresh,
            "retention_task_ids": retention, "old_panel": panel,
            "before_learning_memory_ids": before, "after_learning_memory_ids": after,
            "counts": counts}
