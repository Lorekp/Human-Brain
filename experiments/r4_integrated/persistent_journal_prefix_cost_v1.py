"""Source-only protocol for one bounded, model-free V8 journal-prefix comparison.

One serial ordinary-then-owned pass; no sweep, model import/load, action child,
teacher exposure or whole-checkpoint latency claim. Replay the exact 270 saved
rows at their 265 authenticated boundaries. Each noninitial boundary verifies
previous before append, current capture, current entry, previous with tail,
then current prepublication: 3 current + 2 previous prefixes. Boundary zero has
only its three empty current checks. Include encoding/writes/durability and
actual owned-tail reads. Both copies receive full fresh final verification.

Proposed complete budget: 900 seconds, 4096 MiB process peak, 1 GiB new files.
Import is inert. Calling run requires a reviewed exact source hash inventory;
this source does not itself release execution. Results describe this ordered
journal-path comparison, including cache/order effects, never total checkpoint
or learner speed. Committed-frontier advancement is an in-memory schedule
marker: no boundary/HEAD serialization or publication is measured. Existing
historical files are read-only and remain intact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from experiments.r4_integrated.neuronal_splice_io_v1 import process_identity, process_memory
from experiments.r4_integrated.persistent_checkpoint_journal import BLOCK, ROW_LIMIT, OwnedJournal
from experiments.r4_integrated.persistent_language_action_credit_initialize import (
    BANK_SHA,
    PREFIX,
    _authenticate,
)
from experiments.r4_integrated.persistent_language_action_v1 import encode
from experiments.r4_integrated.persistent_language_action_v8_initialize import _path, _read

KINDS = ("attempts", "teachers", "ingests")
SECONDS = 900.0
PEAK_MIB = 4096
ARTIFACT_BYTES = 1024**3
FINAL_BYTES = 420_178_380
REFERENCE_HASH_BYTES = 276_856_555_770
SOURCE_PATHS = (
    "experiments/r4_integrated/persistent_journal_prefix_cost_v1.py",
    "experiments/r4_integrated/persistent_checkpoint_journal.py",
    "experiments/r4_integrated/persistent_checkpoint_payload_lease.py",
    "experiments/r4_integrated/persistent_language_action_credit_initialize.py",
    "experiments/r4_integrated/persistent_language_action_v8_initialize.py",
    "experiments/r4_integrated/persistent_language_action_v1.py",
    "experiments/r4_integrated/neuronal_splice_io_v1.py",
    "experiments/r4_integrated/__init__.py",
)


def _add(work, key, amount=1):
    work[key] = work.get(key, 0) + amount


def _sources(root, expected):
    if set(expected) != set(SOURCE_PATHS):
        raise ValueError("journal comparison requires its exact reviewed source inventory")
    if Path(__file__).resolve() != root / SOURCE_PATHS[0]:
        raise ValueError("journal comparison source root differs")
    for name in SOURCE_PATHS:
        if hashlib.sha256(_path(root, name).read_bytes()).hexdigest() != expected[name]:
            raise ValueError("reviewed journal comparison source changed")
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name.startswith("experiments.r4_integrated."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in expected:
                raise ValueError("undeclared experiment module imported into comparison")
        if name == "torch" or name.startswith("humanbrain."):
            raise ValueError("journal comparison must run in a model-free process")


class _Budget:
    def __init__(self, output, started):
        self.output, self.started = output, started
        self.peak = 0.0

    def check(self):
        memory = process_memory()
        self.peak = max(self.peak, memory["peak_working_set_mib"])
        if time.perf_counter() - self.started > SECONDS or self.peak > PEAK_MIB:
            raise RuntimeError("journal comparison exhausted its complete time or process budget")
        size = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())
        if size > ARTIFACT_BYTES:
            raise RuntimeError("journal comparison exceeded its new artifact budget")


def _schedule(directory, bank_path, work, budget):
    final, _ = _authenticate(directory, bank_path, work)
    budget.check()
    bank = json.loads(_read(bank_path, BANK_SHA, work))
    schedule, previous_sha, previous_rows = [], None, None
    for sequence in range(265):
        name = f"checkpoints/boundary-{sequence:09d}.json"
        expected = bank["evidence"][PREFIX + name]
        boundary = json.loads(_read(_path(directory.parent, name), expected, work))
        if boundary["sequence"] != sequence or boundary["previous_sha256"] != previous_sha:
            raise ValueError("saved V8 boundary lineage changed")
        rows = boundary["frontiers"]["journals"]
        if [row["kind"] for row in rows] != list(KINDS):
            raise ValueError("saved V8 journal order changed")
        for index, row in enumerate(rows):
            if row["path"] != KINDS[index] + ".jsonl":
                raise ValueError("saved V8 segment identity changed")
            old = previous_rows[index] if previous_rows else {"bytes": 0, "rows": 0}
            delta = row["rows"] - old["rows"]
            if (delta not in (0, 1) or row["bytes"] < old["bytes"]
                    or (row["bytes"] == old["bytes"]) != (delta == 0)
                    or row["bytes"] - old["bytes"] > ROW_LIMIT):
                raise ValueError("saved V8 event delta violates the bounded append schedule")
        if sequence == 0 and any(row["bytes"] or row["rows"] for row in rows):
            raise ValueError("V8 comparison must start with empty journals")
        schedule.append(rows)
        previous_sha, previous_rows = expected, rows
    if schedule[-1] != final["frontiers"]["journals"]:
        raise ValueError("scheduled final frontier differs from authenticated HEAD")
    predicted = sum(3 * sum(row["bytes"] for row in rows)
                    + (2 * sum(row["bytes"] for row in schedule[index - 1]) if index else 0)
                    for index, rows in enumerate(schedule))
    if predicted != REFERENCE_HASH_BYTES or sum(row["bytes"] for row in schedule[-1]) != FINAL_BYTES:
        raise ValueError("saved V8 prefix attribution differs from reviewed byte totals")
    return schedule


def _full_prefix(path, row, current_bytes, work, *, fresh=False):
    digest, size, count, last = hashlib.sha256(), 0, 0, b""
    with path.open("rb") as source:
        while size < row["bytes"]:
            chunk = source.read(min(BLOCK, row["bytes"] - size))
            _add(work, "full_prefix_read_calls")
            if not chunk:
                raise ValueError("journal prefix was shortened")
            digest.update(chunk)
            size += len(chunk)
            count += chunk.count(b"\n")
            last = chunk[-1:]
    _add(work, "fresh_final_bytes_read" if fresh else "schedule_hash_bytes_read", size)
    _add(work, "fresh_final_passes" if fresh else "schedule_hash_passes")
    item = {"bytes": size, "rows": count, "sha256": digest.hexdigest(),
            "complete": not size or last == b"\n"}
    if path.stat().st_size != current_bytes:
        raise ValueError("journal has unexpected EOF beyond its scheduled frontier")
    _match(item, row)
    return item


def _match(item, row):
    if not item["complete"] or any(item[key] != row[key] for key in ("bytes", "rows", "sha256")):
        raise ValueError("actual journal receipt differs from the saved V8 frontier")


def _lane(name, directory, output, schedule, budget, result):
    tick, work = time.perf_counter(), {}
    output.mkdir(exist_ok=False)
    boundary_receipts, row_receipts = [], []
    result.update(lane=name, work=work, boundaries=boundary_receipts, rows=row_receipts)
    owned = name == "owned"
    with ExitStack() as stack:
        sources = [stack.enter_context((directory.parent / (kind + ".jsonl")).open("rb"))
                   for kind in KINDS]
        writers = []
        for kind in KINDS:
            path = output / (kind + ".jsonl")
            if owned:
                writer = OwnedJournal(path, work)
                stack.callback(writer.close)
                if not writer.cached_prefixes:
                    raise ValueError("comparison requires proven local NTFS ownership")
            else:
                writer = stack.enter_context(path.open("xb"))
            writers.append(writer)

        def checks(rows, current):
            results = []
            for index, row in enumerate(rows):
                if owned:
                    item = writers[index].frontier(prefix=row["bytes"])
                    _match(item, row)
                    _add(work, "avoided_prefix_hash_bytes", row["bytes"])
                    _add(work, "cached_prefix_checks")
                else:
                    item = _full_prefix(output / row["path"], row, current[index]["bytes"], work)
                results.append(item)
            budget.check()
            return results

        for sequence, current in enumerate(schedule):
            previous = schedule[sequence - 1] if sequence else None
            if previous is not None:
                with ExitStack() as frozen:
                    if owned:
                        for writer in writers:
                            frozen.enter_context(writer.freeze())
                    checks(previous, previous)  # begin -> inspect, before the next append
            for index, row in enumerate(current):
                before = previous[index]["bytes"] if previous else 0
                added = row["bytes"] - before
                if not added:
                    continue
                raw = sources[index].read(added)
                _add(work, "saved_row_bytes_read", len(raw))
                if len(raw) != added or raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
                    raise ValueError("saved row is not the exact bounded complete delta")
                event = json.loads(raw)
                if encode(event) != raw:
                    raise ValueError("saved row is not the unchanged journal encoding")
                wanted = hashlib.sha256(raw).hexdigest()
                append_tick = time.perf_counter()
                if owned:
                    actual_sha = writers[index].append(event)
                else:
                    encoded = encode(event)
                    count = writers[index].write(encoded)
                    _add(work, "journal_write_calls")
                    _add(work, "journal_bytes_written", count or 0)
                    if count != len(encoded):
                        raise ValueError("ordinary journal write shortened")
                    writers[index].flush()
                    os.fsync(writers[index].fileno())
                    _add(work, "journal_fsync_calls")
                    actual_sha = hashlib.sha256(encoded).hexdigest()
                _add(work, "append_seconds", time.perf_counter() - append_tick)
                if actual_sha != wanted:
                    raise ValueError("written row hash differs from the exact saved row")
                row_receipts.append({"sequence": sequence, "kind": KINDS[index],
                                     "bytes": added, "sha256": actual_sha})
            with ExitStack() as frozen:
                if owned:
                    for writer in writers:
                        frozen.enter_context(writer.freeze())
                captured = checks(current, current)
                checks(current, current)  # commit entry
                if previous is not None:
                    checks(previous, current)  # previous allow_tail
                checks(current, current)  # immediately before publication
                if owned:
                    prepared = [writer.prepare_commit(row)
                                for writer, row in zip(writers, current, strict=True)]
                    for writer, receipt in zip(writers, prepared, strict=True):
                        writer.commit_validated(receipt)
                boundary_receipts.append({"sequence": sequence, "frontiers": captured})
            budget.check()
        if any(source.read(1) for source in sources):
            raise ValueError("saved outcome journal extends beyond the final frontier")
    schedule_seconds = time.perf_counter() - tick
    fresh_tick = time.perf_counter()
    final = [_full_prefix(output / row["path"], row, row["bytes"], work, fresh=True)
             for row in schedule[-1]]
    budget.check()
    result.update(schedule_seconds=schedule_seconds,
                  fresh_final_seconds=time.perf_counter() - fresh_tick,
                  fresh_final_frontiers=final)


def run(output, *, checkpoint_directory, bank_path, source_root, source_sha256):
    """Execute only after explicit source/protocol review; preserve failures and copies."""
    started = time.perf_counter()
    root, output = Path(source_root).resolve(), Path(output).resolve()
    directory, bank_path = Path(checkpoint_directory).resolve(), Path(bank_path).resolve()
    _sources(root, source_sha256)
    if output.is_relative_to(directory.parent) or directory.parent.is_relative_to(output):
        raise ValueError("new comparison output must be separate from historical evidence")
    if shutil.disk_usage(output.parent).free < ARTIFACT_BYTES:
        raise RuntimeError("comparison requires at least its full new artifact budget free")
    output.mkdir(parents=True, exist_ok=False)
    budget = _Budget(output, started)
    receipt = {"schema": "persistent-journal-prefix-cost-v1", "status": "failed",
               "scope": "selected journal-prefix path, not complete checkpoint or learner latency",
               "order": ["ordinary", "owned"], "source_sha256": source_sha256,
               "bank_sha256": BANK_SHA, "reference_schedule_hash_bytes": REFERENCE_HASH_BYTES,
               "budget_seconds": SECONDS, "budget_peak_mib": PEAK_MIB,
               "budget_artifact_bytes": ARTIFACT_BYTES, "process": process_identity(),
               "clock": vars(time.get_clock_info("perf_counter")), "authentication_work": {},
               "elapsed_scope": "run entry through final checks; excludes this receipt write/fsync",
               "models": 0, "tensor_decodes": 0, "children": 0}
    try:
        schedule = _schedule(directory, bank_path, receipt["authentication_work"], budget)
        receipt["authentication_seconds"] = time.perf_counter() - budget.started
        lanes = []
        receipt["lanes"] = lanes
        for name in ("ordinary", "owned"):
            lane = {}
            lanes.append(lane)
            _lane(name, directory, output / name, schedule, budget, lane)
        old, new = lanes
        if any(old[key] != new[key] for key in ("rows", "boundaries", "fresh_final_frontiers")):
            raise ValueError("ordinary and owned journals disagree on actual saved evidence")
        if (old["work"]["schedule_hash_bytes_read"] != REFERENCE_HASH_BYTES
                or new["work"]["avoided_prefix_hash_bytes"] != REFERENCE_HASH_BYTES
                or new["work"]["checkpoint_journal_tail_bytes_read"] != FINAL_BYTES
                or len(old["rows"]) != 270):
            raise ValueError("measured journal work differs from the exact V8 schedule")
        _sources(root, source_sha256)
        budget.check()
        receipt["status"] = "completed"
    except BaseException as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - budget.started
        receipt["peak_working_set_mib"] = max(budget.peak, process_memory()["peak_working_set_mib"])
        receipt["retained_artifact_bytes_before_receipt"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file())
        with (output / "receipt.json").open("xb") as handle:
            handle.write(encode(receipt))
            handle.flush()
            os.fsync(handle.fileno())
    return receipt
