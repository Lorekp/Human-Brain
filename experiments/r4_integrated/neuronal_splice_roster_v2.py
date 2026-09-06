"""Parent-only structural transfer data for the fixed five-arm comparison.

No model, executor or worker imports. TRAIN and the authenticated old reading
panel reuse v1's immutable constructors; its exposed fresh builder is never run.
Only task_observation's five raw fields belong in a worker request.
"""

from __future__ import annotations

from pathlib import Path

from experiments.r4_integrated import neuronal_splice_roster_v1 as original

SCHEMA = "neuronal-splice-roster-v2"
GROUPS = ("D", "C", "V", "O")
CONTINUATIONS = ((b"ab", b"ababab"), (b"ba", b"bababa"),
                 (b"cd", b"cdcdcd"), (b"dc", b"dcdcdc"))
ROLE_VALUES = ((b"ab", b"ba"), (b"cd", b"dc"),
               (b"ac", b"ca"), (b"bd", b"db"))


def _training() -> tuple[dict, dict]:
    """Exact v1 TRAIN loop, without constructing its exposed fresh cohort."""
    tasks, packages = {}, {}
    for t in range(32):
        r, task_id = t % 8, f"T{t:02d}"
        memory_id = f"M-{task_id}"
        original._package(packages, memory_id, original._pair_documents(
            t, original.P[t], (t + 9) % 32, original.P[(t + 9) % 32], bool(t % 2)),
            "before_learning")
        tasks[task_id] = original._task(
            task_id, ("replace", "insert", "delete", "keep")[t // 8], memory_id,
            ("a0.py", "b0.py"), "a0.py" if r % 2 == 0 else "b0.py",
            "x" if (r // 2) % 2 == 0 else "y", original._key(t), "A" if r < 4 else "B",
            b"" if 8 <= t < 16 else b"old", original.P[t], keep=t >= 24)
    return tasks, packages


def _layout_task(r: int, member: int, memory_id: str, payload: bytes) -> dict:
    """Offsets and expected bytes follow construction, never code matching."""
    paths = (f"op{r}.py", f"peer{r}.py")
    target, variable = paths[r % 2], "x" if r % 2 == 0 else "y"
    # Both replacement layouts are absent from TRAIN. Insertion/deletion retain
    # ordinary assignments but move their raw boundaries with blank/comment bytes.
    prefixes = (b"", b"", b"\n# insertion\n\n", b"# deletion\n\n")
    assignments = ((b"  =\t'", b"\t=  '"), (b"\t = '", b" =\t '"),
                   (b" = '", b" = '"), (b" = '", b" = '"))
    prefix, separators = prefixes[r], assignments[r]
    interior = b"" if r == 2 else b"old"
    lines = [name.encode() + separators[i] + interior + b"'\n"
             for i, name in enumerate(("x", "y"))]
    selected = int(variable == "y")
    start = len(prefix) + sum(len(line) for line in lines[:selected])
    start += 1 + len(separators[selected])
    end = start + len(interior)
    data = prefix + b"".join(lines)
    a, b = (path[:-3] for path in paths)
    main = (f"import {a}, {b}\n"
            f"print({a}.x, {a}.y, {b}.x, {b}.y, sep='|')\n").encode("ascii")
    before = sorted([(paths[0], data), (paths[1], data), ("main.py", main)])
    files = original._files_record(before)
    digest = original.workspace_digest(files)
    keep = bool(member)
    request = (b"KEEP" if keep else b"SET") + b" " + target.encode() + b" "
    request += variable.encode() + b" " + original._key(92 + r)
    teacher = {"operation": "STOP" if keep else "SPLICE", "path": "" if keep else target,
               "start": 0 if keep else start, "end": 0 if keep else end,
               "replacement_hex": "" if keep else payload.hex(), "stop_reason": "eos",
               "workspace_sha256": digest}
    after = [(path, raw if keep or path != target else raw[:start] + payload + raw[end:])
             for path, raw in before]
    displayed = [interior] * 4
    if not keep:
        displayed[2 * paths.index(target) + selected] = payload
    return {"task_id": f"O{r}-{member}", "group": "O", "pair_id": f"O{r}",
            "member": member, "memory_id": memory_id, "request_hex": request.hex(),
            "files": files, "workspace_sha256": digest, "teacher": teacher,
            "expected_after_files": original._files_record(after),
            "expected_stdout_hex": (b"|".join(displayed) + b"\n").hex(),
            "required_operation": teacher["operation"]}


def _fresh(tasks: dict, packages: dict) -> None:
    for group in GROUPS:
        for r in range(4):
            pair = f"{group}{r}"
            key = 80 + 4 * GROUPS.index(group) + r
            paths = ((f"ord{r}.py", f"alt{r}.py") if group == "D" else
                     (f"seq{r}.py", f"aux{r}.py") if group == "C" else
                     (f"val{r}.py", f"dec{r}.py"))
            if group == "O":
                payload = (b"ad", b"da", b"ab", b"")[r]
                original._package(packages, f"M-{pair}", original._pair_documents(
                    key, payload, 96 + r, b"bc", False), "after_learning")
            for member in range(2):
                memory_id = f"M-{pair}" + ("" if group == "O" else f"-{member}")
                if group == "O":
                    task = _layout_task(r, member, memory_id, payload)
                else:
                    if group == "D":
                        payload = (b"ac", b"ca", b"bd", b"db")[r]
                        documents = original._pair_documents(
                            key, payload, 96 + r, payload[::-1], bool(member))
                    elif group == "C":
                        payload = CONTINUATIONS[r][member]
                        documents = original._pair_documents(key, payload, 96 + r, b"cc", False)
                    else:
                        payload = ROLE_VALUES[r][member]
                        documents = original._pair_documents(
                            key, payload, 96, ROLE_VALUES[r][1 - member], False)
                        if r >= 2:
                            documents.extend(original._document(original._key(97 + j), value)
                                             for j, value in enumerate((b"aa", b"bb", b"dd")))
                    original._package(packages, memory_id, documents,
                                      "before_learning" if group == "V" and r < 2
                                      else "after_learning")
                    task = original._task(
                        f"{pair}-{member}", group, memory_id, paths, paths[r % 2],
                        "x" if r % 2 == 0 else "y", original._key(key),
                        "A" if r < 2 else "B", b"old", payload,
                        pair_id=pair, member=member)
                if task["task_id"] in tasks:
                    raise ValueError("duplicate task identity")
                tasks[task["task_id"]] = task


def task_observation(task: dict) -> dict:
    """Export only raw request/workspace identity; never labels or pair metadata."""
    return original.task_observation(task)


def _validate(tasks: dict, packages: dict, fresh: list[str]) -> None:
    fresh_paths, fresh_keys = set(), set()
    for task_id, task in tasks.items():
        files, teacher = task["files"], task["teacher"]
        request = bytes.fromhex(task["request_hex"])
        if task["task_id"] != task_id or task["memory_id"] not in packages:
            raise ValueError("task/package identity mismatch")
        paths = [item["path"] for item in files]
        if (not 1 <= len(files) <= 4 or len(set(paths)) != len(paths)
                or not 1 <= len(request) <= 31 or "main.py" not in paths
                or any(len(item["path"].encode()) > 31
                       or len(bytes.fromhex(item["content_hex"])) > 128 for item in files)):
            raise ValueError("task input geometry changed")
        digest = original.workspace_digest(files)
        if digest != task["workspace_sha256"] or digest != teacher["workspace_sha256"]:
            raise ValueError("workspace binding changed")
        before = {item["path"]: bytes.fromhex(item["content_hex"]) for item in files}
        after = dict(before)
        payload = bytes.fromhex(teacher["replacement_hex"])
        if teacher["operation"] != task["required_operation"] or teacher["stop_reason"] != "eos":
            raise ValueError("canonical operation binding changed")
        if teacher["operation"] == "SPLICE":
            raw = before[teacher["path"]]
            start, end = teacher["start"], teacher["end"]
            if not 0 <= start <= end <= len(raw) or len(payload) > 16:
                raise ValueError("canonical raw span exceeds bounds")
            after[teacher["path"]] = raw[:start] + payload + raw[end:]
        elif teacher["operation"] != "STOP" or payload or teacher["path"]:
            raise ValueError("invalid STOP correction")
        expected = {item["path"]: bytes.fromhex(item["content_hex"])
                    for item in task["expected_after_files"]}
        if after != expected or any(len(raw) > 128 for raw in after.values()):
            raise ValueError("canonical splice does not produce the declared raw files")
        if task_id in fresh:
            fresh_paths.update(path for path in paths if path != "main.py")
            fresh_keys.add(request.split(b" ")[-1])
    # All v1 new-task key indices occupy 00..73; its old panel uses a different
    # raw question format. Three-byte new keys preserve 125-slot interference.
    exposed_paths = {"a0.py", "b0.py"}
    for r in range(4):
        exposed_paths.update((f"a{r + 1}.py", f"b{r + 1}.py"))
        exposed_paths.update(f"{stem}{r}.py" for stem in "mnpquv")
    if fresh_paths & exposed_paths or fresh_keys != {original._key(i) for i in range(80, 96)}:
        raise ValueError("fresh path/key identity overlaps or scope changed")
    for group in GROUPS:
        for r in range(4):
            left, right = (tasks[f"{group}{r}-{m}"] for m in range(2))
            if left["files"] != right["files"]:
                raise ValueError("paired workspace must remain identical")
            if group != "O" and left["request_hex"] != right["request_hex"]:
                raise ValueError("paired raw request changed")
            docs = [packages[t["memory_id"]]["document_hex"] for t in (left, right)]
            if group == "D" and (docs[0] != docs[1][::-1] or left["teacher"] != right["teacher"]):
                raise ValueError("document-order contrast changed")
            if group == "C" and tuple(bytes.fromhex(t["teacher"]["replacement_hex"])
                                      for t in (left, right)) != CONTINUATIONS[r]:
                raise ValueError("continuation lengths changed")
            if group == "V" and any(len(d) != (2 if r < 2 else 5) for d in docs):
                raise ValueError("delay/interference package geometry changed")
            if group == "V":
                for member, document_hex in enumerate(docs):
                    expected_docs = original._pair_documents(
                        88 + r, ROLE_VALUES[r][member], 96, ROLE_VALUES[r][1 - member], False)
                    if r >= 2:
                        expected_docs.extend(original._document(original._key(97 + j), value)
                                             for j, value in enumerate((b"aa", b"bb", b"dd")))
                    if document_hex != [raw.hex() for raw in expected_docs]:
                        raise ValueError("value-role exchange changed")
            if group == "O" and (left["memory_id"] != right["memory_id"]
                                 or (left["required_operation"], right["required_operation"])
                                 != ("SPLICE", "STOP")):
                raise ValueError("operation contrast changed")
    for memory_id in {tasks[key]["memory_id"] for key in fresh}:
        for raw_hex in packages[memory_id]["document_hex"]:
            raw = bytes.fromhex(raw_hex)
            if (raw[:7] != b"Record " or raw[7:10] not in
                    {original._key(index) for index in range(80, 100)}):
                raise ValueError("fresh literal document key overlaps exposed keys")


def build_roster(old_roster_path: Path) -> dict:
    """Construct once after source approval; authenticate the unchanged old panel."""
    tasks, packages = _training()
    _fresh(tasks, packages)
    panel, source = original._load_old_panel(old_roster_path, packages)
    train = [f"T{index:02d}" for index in range(32)]
    fresh = [f"{group}{r}-{member}" for group in GROUPS
             for r in range(4) for member in range(2)]
    retention = [f"T{index:02d}" for base in (0, 8, 16, 24) for index in range(base, base + 4)]
    schedule = [f"T{(index + 4 * repeat) % 32:02d}" for repeat in range(8) for index in range(32)]
    before = [f"M-{task_id}" for task_id in train]
    before += [f"M-V{r}-{member}" for r in range(2) for member in range(2)]
    before += list(dict.fromkeys(item["memory_id"] for item in panel))
    after = [key for key, value in packages.items() if value["stage"] == "after_learning"]
    _validate(tasks, packages, fresh)
    documents = [bytes.fromhex(raw) for package in packages.values()
                 for raw in package["document_hex"]]
    raw_bytes = sum(map(len, documents))
    payload_offers = sum(len(bytes.fromhex(tasks[key]["teacher"]["replacement_hex"])) + 1
                         for key in schedule if tasks[key]["required_operation"] == "SPLICE")
    action_labels = sum(3 if tasks[key]["required_operation"] == "SPLICE" else 1
                        for key in schedule)
    counts = {
        "tasks": len(tasks), "train_tasks": len(train), "fresh_tasks": len(fresh),
        "retention_tasks": len(retention), "lessons_per_arm": len(schedule),
        "memory_packages_per_arm": len(packages), "before_learning_packages": len(before),
        "after_learning_packages": len(after), "documents_per_arm": len(documents),
        "document_raw_bytes_per_arm": raw_bytes, "document_eos_per_arm": len(documents),
        "ingestion_symbols_per_arm": raw_bytes + len(documents),
        "maximum_memory_slots": max(sum(len(bytes.fromhex(raw)) + 1
                                        for raw in value["document_hex"])
                                    for value in packages.values()),
        "reader_teacher_offers_per_arm": payload_offers,
        "action_teacher_labels_per_arm": action_labels,
        "full_action_verifications": 5 * (len(schedule) + len(fresh) + len(retention)),
        "old_panel_reader_calls": 5 * 2 * len(panel),
        "full_reader_teacher_offers": 5 * payload_offers,
        "full_action_teacher_labels": 5 * action_labels,
    }
    expected = (64, 72, 48, 24, 168, 4022, 128, 480, 640)
    actual = (len(tasks), len(packages), len(before), len(after), len(documents), raw_bytes,
              counts["maximum_memory_slots"], payload_offers, action_labels)
    if (actual != expected or len(set(before + after)) != len(packages)
            or set(before + after) != set(packages)
            or any(packages[key]["stage"] != "before_learning" for key in before)
            or sum(tasks[key]["required_operation"] == "SPLICE" for key in fresh) != 28):
        raise ValueError(f"roster arithmetic mismatch: actual={actual}, expected={expected}")
    return {"schema": SCHEMA, "source_inputs": source, "tasks": tasks,
            "memory_packages": packages, "train_task_ids": train, "schedule": schedule,
            "cost_task_ids": ["T00", "T08", "T16", "T24"], "fresh_task_ids": fresh,
            "retention_task_ids": retention, "old_panel": panel,
            "before_learning_memory_ids": before, "after_learning_memory_ids": after,
            "counts": counts}
