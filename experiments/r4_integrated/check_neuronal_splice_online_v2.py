"""Independent saved-evidence checks for the selected neuronal splice protocol.

This evaluator may read the parent-only roster. It never belongs in the worker
package and never imports or executes a neuronal model or action executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time
from pathlib import Path

from experiments.r4_integrated.check_neuronal_address_competition_v1 import check_reader
from experiments.r4_integrated.neuronal_splice_receipts_v1 import (
    HEAD_LIMITS,
    close,
    exact_keys,
    hash_value,
    hex_bytes,
    logsumexp,
    number,
    proposal_digest,
    require,
    tree_bytes,
    validate_head,
    validate_proposal,
    validate_work,
)

DRAFT = False
LINE_LIMIT = TREE_LIMIT = 2 << 20
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled", "marginal_reference")
MODEL_BYTES = 634432
TAPE_BYTES = 656904
WORKER_SOURCE = "experiments/r4_integrated/neuronal_splice_worker_v2.py"
WORKER_MODULES = {
    "humanbrain": "src/humanbrain/__init__.py",
    **{
        "humanbrain." + name: "src/humanbrain/" + name + ".py"
        for name in (
            "neuronal_splice_learner",
            "neuronal_splice_action",
            "local_memory_learner",
            "competitive_local_memory_learner",
            "neuronal_address_competition",
            "neuronal_sequence_memory",
            "byte_sequence_decoder",
            "plastic_memory",
            "multiscale_opaque_event_tape",
        )
    },
    "experiments.r4_integrated": "experiments/r4_integrated/__init__.py",
    **{
        "experiments.r4_integrated." + name: "experiments/r4_integrated/" + name + ".py"
        for name in (
            "neuronal_splice_io_v1",
            "neuronal_splice_receipts_v1",
            "neuronal_splice_receipts_v2",
            "neuronal_splice_models_v2",
        )
    },
}


def reader_policy(arm):
    require(arm in ARMS, "unknown comparison arm")
    return "marginal_free" if arm == "marginal_reference" else "competitive_free"


def expected_reader_metadata(original, arm):
    """Literal permitted phase declarations, independent of the model/factory."""
    mode = {
        "action_frozen": "learned",
        "memory_disabled": "writes_disabled",
        "marginal_reference": "learned",
    }.get(arm, arm)
    expected = {**original, "mode": mode}
    if arm != "marginal_reference":
        expected.update(
            {
                "schema": "local-neuronal-competitive-address-learner-v1",
                "recall": (
                    "two fixed threshold layers select first maximum float32 address score; "
                    "one actual cursor follows forward EOS-absorbing adjacency"
                ),
                "competition": (
                    "float64 pair currents for i<j; nonnegative comparator thresholds; "
                    "fixed signed winner currents and nonnegative thresholds; "
                    "lower-index ties; sum(i*w_i) identity readout"
                ),
                "free_belief": "one normalized branch; full soft address prior is diagnostic only",
                "codec_activity": (
                    "evaluate every occupied sensory slot once with held weights; "
                    "free recall gathers one actual cursor at every decision including first; "
                    "teacher retains inherited all-address bank and prefix gathers"
                ),
                "teacher_phase": (
                    "after actual execution, reopen all address candidates on the same synapses; "
                    "inherited raw-byte/EOS prefix evidence and local p-minus-final-posterior "
                    "surrogate; no competition pass, hard-gate derivative or backward transport"
                ),
            }
        )
    return expected


def check_prediction(row, successor, *, arm, boundary_list):
    exact_keys(
        row,
        (
            "proposal",
            "attempt_id",
            "proposal_sha256",
            "model_sha256",
            "memory_sha256",
            "choices",
            "payload",
            "work",
        ),
    )
    require(
        type(row["attempt_id"]) is str and 0 < len(row["attempt_id"]) <= 128,
        "invalid pending attempt identity",
    )
    for key in ("proposal_sha256", "model_sha256", "memory_sha256"):
        hash_value(row[key])
    require(row["proposal_sha256"] == proposal_digest(row["proposal"]), "proposal digest mismatch")
    exact_keys(row["choices"], HEAD_LIMITS)
    start = row["choices"]["start"]
    validate_head(start, "start", count=len(boundary_list))
    path, offset = boundary_list[start["selected_index"]]
    ends = [end for name, end in boundary_list if name == path and end >= offset]
    validate_head(row["choices"]["end"], "end", count=len(ends))
    validate_head(row["choices"]["operation"], "operation", count=2)
    proposal = row["proposal"]
    require(
        (proposal["path"], proposal["start"], proposal["end"])
        == (path, offset, ends[row["choices"]["end"]["selected_index"]]),
        "proposal differs from selected raw boundaries",
    )
    require(
        proposal["operation"]
        == ("STOP" if row["choices"]["operation"]["selected_index"] == 0 else "SPLICE"),
        "operation head mismatch",
    )
    result = check_reader(row["payload"], successor, reader_policy(arm))
    require(
        proposal["replacement_hex"] == row["payload"]["generated_hex"]
        and proposal["stop_reason"] == row["payload"]["stop_reason"],
        "free payload replaced",
    )
    exact_keys(row["work"], ("action", "reader"))
    for work in row["work"].values():
        validate_work(work)
    require(row["work"]["reader"] == row["payload"]["work"], "payload work mismatch")
    return result


def expected_config():
    return {
        "schema": "neuronal-splice-online-v2",
        "arms": list(ARMS),
        "seed": 733,
        "reader_policy": {arm: reader_policy(arm) for arm in ARMS},
        "learning_rate": 0.01,
        "lessons_per_arm": 256,
        "cost_task_ids": ["T00", "T08", "T16", "T24"],
        "fresh_per_arm": 32,
        "retention_per_arm": 16,
        "panel_per_pass_per_arm": 20,
        "retained_memories_per_arm": 72,
        "retained_tensor_bytes": 239657600,
        "receipt_limits": {
            "max_line_bytes": 2097152,
            "max_row_tree_bytes": 2097152,
            "max_live_rows": 2,
            "max_line_copies": 2,
            "allocation_factor": 2,
        },
        "seconds": 900,
        "tree_mib": 2048,
        "run_reserve_mib": 128,
        "audit_reserve_mib": 32,
        "buffer_mib": 24,
        "duration_clock": "perf_counter",
        "full_only_tensor_bytes": 0,
    }


def check_roster(roster):
    """Recount authenticated raw data; never call the task constructor."""
    train = [f"T{i:02d}" for i in range(32)]
    fresh = [
        f"{group}{pair}-{member}" for group in "DCVO" for pair in range(4) for member in range(2)
    ]
    retention = [f"T{i:02d}" for base in (0, 8, 16, 24) for i in range(base, base + 4)]
    schedule = [f"T{(i + 4 * repeat) % 32:02d}" for repeat in range(8) for i in range(32)]
    require(
        roster["schema"] == "neuronal-splice-roster-v2"
        and roster["train_task_ids"] == train
        and roster["fresh_task_ids"] == fresh
        and roster["retention_task_ids"] == retention
        and roster["schedule"] == schedule
        and roster["cost_task_ids"] == ["T00", "T08", "T16", "T24"],
        "selected task order/allocation changed",
    )
    tasks, packages = roster["tasks"], roster["memory_packages"]
    require(
        set(tasks) == set(train + fresh) and len(roster["old_panel"]) == 20,
        "task/panel scope changed",
    )
    before, after = (
        roster[stage + "_memory_ids"] for stage in ("before_learning", "after_learning")
    )
    require(
        len(before) == 48
        and len(after) == 24
        and len(set(before + after)) == 72
        and set(before + after) == set(packages),
        "memory stage allocation changed",
    )
    for stage, ids in (("before_learning", before), ("after_learning", after)):
        require(
            all(packages[key]["stage"] == stage for key in ids), "memory ingestion stage changed"
        )
    documents = [hex_bytes(raw) for package in packages.values() for raw in package["document_hex"]]
    slots = [len(documents_adjacency(package["document_hex"])) for package in packages.values()]
    offers = sum(
        len(hex_bytes(tasks[key]["teacher"]["replacement_hex"], 16)) + 1
        for key in schedule
        if tasks[key]["required_operation"] == "SPLICE"
    )
    labels = sum(3 if tasks[key]["required_operation"] == "SPLICE" else 1 for key in schedule)
    require(
        sum(tasks[key]["required_operation"] == "SPLICE" for key in fresh) == 28
        and all(tasks[key]["required_operation"] in ("SPLICE", "STOP") for key in tasks),
        "fresh edit/STOP allocation changed",
    )
    counts = {
        "tasks": 64,
        "train_tasks": 32,
        "fresh_tasks": 32,
        "retention_tasks": 16,
        "lessons_per_arm": 256,
        "memory_packages_per_arm": 72,
        "before_learning_packages": 48,
        "after_learning_packages": 24,
        "documents_per_arm": len(documents),
        "document_raw_bytes_per_arm": sum(map(len, documents)),
        "document_eos_per_arm": len(documents),
        "ingestion_symbols_per_arm": sum(map(len, documents)) + len(documents),
        "maximum_memory_slots": max(slots),
        "reader_teacher_offers_per_arm": offers,
        "action_teacher_labels_per_arm": labels,
        "full_action_verifications": 1520,
        "old_panel_reader_calls": 200,
        "full_reader_teacher_offers": 5 * offers,
        "full_action_teacher_labels": 5 * labels,
    }
    require(
        (len(documents), sum(map(len, documents)), max(slots), offers, labels)
        == (168, 4022, 125, 480, 640)
        and roster["counts"] == counts,
        "independent raw/EOS/teacher arithmetic changed",
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def documents_adjacency(documents):
    successors = []
    for encoded in documents:
        raw = hex_bytes(encoded)
        start = len(successors)
        successors.extend(range(start + 1, start + len(raw) + 1))
        successors.append(start + len(raw))
    require(1 <= len(successors) <= 128, "document package exceeds tape geometry")
    return successors


def ordered_files(files, *, maximum=128):
    require(type(files) is list and 1 <= len(files) <= 4, "invalid workspace file count")
    result = []
    for item in files:
        exact_keys(item, ("path", "content_hex"))
        path = item["path"]
        require(
            type(path) is str and 0 < len(path.encode("utf-8")) <= 31,
            "invalid workspace path bound",
        )
        raw = hex_bytes(item["content_hex"], maximum)
        result.append((path, raw))
    require(len({path.casefold() for path, _ in result}) == len(result), "duplicate file path")
    return sorted(result, key=lambda item: item[0].encode("utf-8"))


def workspace_digest(files):
    ordered = ordered_files(files)
    digest = hashlib.sha256(b"humanbrain.raw-splice-workspace.v1\0")
    digest.update(len(ordered).to_bytes(4, "big"))
    for path, raw in ordered:
        for value in (path.encode("utf-8"), raw):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def boundaries(files):
    return [(path, offset) for path, raw in ordered_files(files) for offset in range(len(raw) + 1)]


def apply_saved(files, proposal):
    """Independent literal byte splice; no teacher or semantic target is accepted."""
    validate_proposal(proposal)
    before = ordered_files(files)
    rejection = None
    if proposal["workspace_sha256"] != workspace_digest(files):
        rejection = "stale_workspace"
    elif proposal["operation"] == "SPLICE":
        found = [(path, raw) for path, raw in before if path == proposal["path"]]
        if proposal["stop_reason"] != "eos":
            rejection = "missing_eos"
        elif not found or not 0 <= proposal["start"] <= proposal["end"] <= len(found[0][1]):
            rejection = "invalid_span"
        else:
            start, end = proposal["start"], proposal["end"]
            replacement = hex_bytes(proposal["replacement_hex"], 16)
            before = [
                (path, raw[:start] + replacement + raw[end:] if path == proposal["path"] else raw)
                for path, raw in before
            ]
    return [{"path": path, "content_hex": raw.hex()} for path, raw in before], rejection


def task_grade(task, proposal, after_files, rejection, execution):
    expected_after, expected_rejection = apply_saved(task["files"], proposal)
    require(
        ordered_files(after_files, maximum=144) == ordered_files(expected_after, maximum=144),
        "saved after-files differ from actual selected action",
    )
    require((rejection is None) == (expected_rejection is None), "rejection status mismatch")
    if rejection is not None:
        require(
            type(rejection) is str and 0 < len(rejection.encode("utf-8")) <= 1024,
            "invalid rejection diagnostic",
        )
    stdout = hex_bytes(execution["stdout_hex"], 16384)
    stderr = hex_bytes(execution["stderr_hex"], 16384)
    require(
        type(execution["returncode"]) is int or execution["returncode"] is None,
        "invalid process outcome",
    )
    for name, raw in (("stdout", stdout), ("stderr", stderr)):
        require(
            execution[name + "_sha256"] == hashlib.sha256(raw).hexdigest(), "output digest mismatch"
        )
    require(type(execution["materialized_matches_action"]) is bool, "invalid materialization flag")
    success = (
        rejection is None
        and execution["returncode"] == 0
        and not stderr
        and execution["materialized_matches_action"]
        and not execution["decode_errors"]
        and stdout == hex_bytes(task["expected_stdout_hex"])
        and ordered_files(after_files, maximum=144)
        == ordered_files(task["expected_after_files"], maximum=144)
        and proposal["operation"] == task["required_operation"]
    )
    canonical_correct = (
        proposal["operation"] == "STOP"
        if task["required_operation"] == "STOP"
        else proposal == task["teacher"]
    )
    return {"task_success": success, "canonical_tuple_correct": canonical_correct}


def reader_work(query, slots, decisions, *, teaching, arm):
    require(
        arm in ARMS and 1 <= slots <= 128 and 1 <= decisions <= 17, "invalid reader work geometry"
    )
    work = dict(
        query_raw_bytes=len(query),
        query_eos_symbols=1,
        query_lag_shift_steps=len(query) + 1,
        query_population_normalizations=1,
        address_feature_slots=slots,
        address_feature_products=slots * 32768,
        address_feature_channel_reductions=slots * 31744,
        address_score_synapse_evaluations=slots * 1024,
        address_score_reductions=slots * 1023,
        address_score_evaluations=slots,
        address_normalizations=1,
        codec_slot_evaluations=slots,
        codec_output_potentials=slots * 257,
        codec_normalizations=slots,
        codec_response_gathers=slots * (decisions - 1),
        codec_response_values_gathered=slots * (decisions - 1) * 1028,
        codec_response_bytes_gathered=slots * (decisions - 1) * 7196,
        mixture_output_reductions=257 * decisions,
        prefix_belief_updates=slots * decisions,
        branch_cursor_initializations=slots,
        branch_cursor_advances=slots * (decisions - 1),
        teacher_offers=decisions if teaching else 0,
        codec_update_group_offers=int(teaching),
        address_update_group_offers=int(teaching),
        enabled_codec_update_groups=int(teaching and arm != "slow_frozen"),
        enabled_address_update_groups=int(teaching and arm != "slow_frozen"),
        address_local_synapse_contributions=slots * 1024 if teaching else 0,
        address_local_slot_reductions=(slots - 1) * 1024 if teaching else 0,
    )
    if not teaching and reader_policy(arm) == "competitive_free":
        work.update(
            {
                "codec_response_gathers": decisions,
                "codec_response_values_gathered": decisions * 1028,
                "codec_response_bytes_gathered": decisions * 7196,
                "prefix_belief_updates": decisions,
                "branch_cursor_initializations": 1,
                "branch_cursor_advances": decisions - 1,
                "codec_bank_slot_indices": slots,
            }
        )
        pairs = slots * (slots - 1) // 2
        work.update(
            {
                "competition_candidates": slots,
                "competition_pairs": pairs,
                "competition_threshold_layers": 2,
                "competition_comparator_neurons": pairs,
                "competition_winner_neurons": slots,
                "competition_comparator_synapse_applications": 2 * pairs,
                "competition_winner_synapse_applications": 2 * pairs,
                "competition_comparator_subtractions": pairs,
                "competition_winner_accumulations": 2 * pairs,
                "competition_winner_bias_values": slots,
                "competition_winner_threshold_subtractions": slots,
                "competition_threshold_evaluations": pairs + slots,
                "competition_identity_synapse_applications": slots,
                "competition_identity_reductions": slots - 1,
                "competition_winner_count_reductions": slots - 1,
                "competition_index_values_constructed": 2 * pairs + slots,
                "competition_receipt_tensor_bytes": 13 * slots + 9 * pairs,
                "competition_named_allocated_tensor_bytes": 45 * slots + 49 * pairs + 16,
                "competition_pair_index_bytes": 16 * pairs,
                "competition_score_gather_bytes": 16 * pairs,
            }
        )
    return work


def action_work(request, files, source, *, teaching, arm, stop_teacher=False):
    """Shape arithmetic independent of the neuronal feature implementation."""
    ordered = ordered_files(files)
    lengths = dict((path, len(raw)) for path, raw in ordered)
    f, b = len(ordered), sum(length + 1 for length in lengths.values())
    chosen = lengths[source["path"]]
    ends = chosen - source["start"] + 1
    span, payload = source["end"] - source["start"], len(hex_bytes(source["replacement_hex"], 16))
    raw = len(request) + sum(len(path.encode("utf-8")) for path in lengths) + sum(lengths.values())
    raw += chosen + span + payload
    eos = 3 + 2 * f + int(source["stop_reason"] == "eos")
    context_count = 1
    if not stop_teacher:
        raw += sum(lengths.values()) + chosen
        eos += f + 1
        context_count += b + ends
    work = dict(
        raw_bytes_encoded=raw,
        eos_codes_encoded=eos,
        code_rows_loaded=raw + eos,
        boundary_contexts=context_count,
        context_code_gathers=32 * context_count,
        custody_tensor_bytes_hashed=(MODEL_BYTES + TAPE_BYTES) * (1 if teaching else 2),
    )
    shapes = {"operation": (1, 4244, 4241 * 32)}
    if not stop_teacher:
        shapes = {
            "start": (b, 6177, (f * 5152 + b * 1024) * 32),
            "end": (ends, 2049, ends * 2048 * 32),
            **shapes,
        }
    for head, (rows, width, products) in shapes.items():
        work[head + "_feature_products"] = products
        work[head + "_feature_channel_reductions"] = products // 32 * 31
        work[head + "_feature_normalizations"] = rows
        work[head + "_feature_components"] = rows * width
        outputs = 2 if head == "operation" else rows
        work[head + "_score_products"] = outputs * width
        work[head + "_score_reductions"] = outputs * (width - 1)
        work[head + "_score_evaluations"] = outputs
        work[head + "_softmaxes"] = 1
        if teaching:
            work[head + "_local_products"] = outputs * width
            work[head + "_local_reductions"] = 0 if head == "operation" else (rows - 1) * width
            work[head + "_teacher_offers"] = 1
            work[head + "_log_softmaxes"] = 1
            work[head + "_enabled_update_groups"] = int(
                arm in ("learned", "memory_disabled", "marginal_reference")
            )
    work["operation_bias_additions"] = 2
    return work


def check_feedback(feedback, prediction, teacher, files, successor, request, arm):
    exact_keys(
        feedback,
        (
            "attempt_id",
            "action_updates",
            "action_losses",
            "teacher_choices",
            "reader",
            "reader_updates",
            "work",
        ),
    )
    require(feedback["attempt_id"] == prediction["attempt_id"], "feedback attempt mismatch")
    validate_proposal(teacher)
    stop = teacher["operation"] == "STOP"
    heads = {"operation"} if stop else set(HEAD_LIMITS)
    for name in ("action_updates", "action_losses", "teacher_choices"):
        exact_keys(feedback[name], heads)
    boundary_list = boundaries(files)
    targets = {"operation": int(not stop)}
    if not stop:
        targets["start"] = boundary_list.index((teacher["path"], teacher["start"]))
        targets["end"] = teacher["end"] - teacher["start"]
    source = prediction["proposal"] if stop else teacher
    counts = {
        "start": len(boundary_list),
        "end": sum(
            path == source["path"] and offset >= source["start"] for path, offset in boundary_list
        ),
        "operation": 2,
    }
    for head in heads:
        decision = feedback["teacher_choices"][head]
        validate_head(decision, head, count=counts[head])
        logits = decision["logits"]
        close(feedback["action_losses"][head], logsumexp(logits) - logits[targets[head]])
        require(type(feedback["action_updates"][head]) is bool, "invalid changed flag")
        if arm in ("action_frozen", "slow_frozen"):
            require(not feedback["action_updates"][head], "frozen action group changed")
    exact_keys(feedback["work"], ("action", "reader"))
    require(
        feedback["work"]["action"]
        == action_work(request, files, source, teaching=True, arm=arm, stop_teacher=stop),
        "action teacher work mismatch",
    )
    if stop:
        require(
            feedback["reader"] is None
            and feedback["reader_updates"] is None
            and feedback["work"]["reader"] == {},
            "STOP offered reader teaching",
        )
        return 0, 1
    reader = feedback["reader"]
    exact_keys(feedback["reader_updates"], ("codec", "address"))
    require(
        all(type(value) is bool for value in feedback["reader_updates"].values()),
        "invalid reader changed flags",
    )
    require(
        arm != "slow_frozen" or not any(feedback["reader_updates"].values()),
        "frozen reader changed",
    )
    require(
        arm != "memory_disabled" or feedback["reader_updates"]["address"] is False,
        "zero episodic features changed address synapses",
    )
    require(reader["kind"] == "teacher", "missing reader teacher")
    result = check_reader(
        reader, successor, "all_address_teacher", targets=hex_bytes(teacher["replacement_hex"], 16)
    )
    expected = reader_work(request, len(successor), result["decisions"], teaching=True, arm=arm)
    require(
        reader["work"] == feedback["work"]["reader"] == expected, "reader teacher work mismatch"
    )
    return result["decisions"], 3


class Stream:
    """Keeps scalar/hash state only; the bounded pair function owns current rows."""

    def __init__(self, path, metadata):
        exact_keys(
            metadata, ("path", "rows", "bytes", "sha256", "max_line_bytes", "max_row_tree_bytes")
        )
        require(Path(path).name == metadata["path"], "journal path mismatch")
        hash_value(metadata["sha256"])
        self.handle = Path(path).open("rb")  # noqa: SIM115 - closed after mandatory EOF verification
        self.metadata, self.hasher = metadata, hashlib.sha256()
        self.rows = self.bytes = self.max_line = self.max_tree = 0

    def next(self):
        line = self.handle.readline(LINE_LIMIT + 1)
        require(
            line and len(line) <= LINE_LIMIT and line.endswith(b"\n") and line.isascii(),
            "invalid bounded line",
        )
        row = json.loads(line)
        require(type(row) is dict, "journal row must be an object")
        size = tree_bytes(row)
        require(size <= TREE_LIMIT, "parsed receipt exceeds tree limit")
        self.hasher.update(line)
        self.rows += 1
        self.bytes += len(line)
        self.max_line, self.max_tree = max(self.max_line, len(line)), max(self.max_tree, size)
        return row, hashlib.sha256(line).hexdigest()

    def finish(self):
        require(self.handle.read(1) == b"", "journal contains trailing records")
        self.handle.close()
        for key, value in (
            ("rows", self.rows),
            ("bytes", self.bytes),
            ("max_line_bytes", self.max_line),
            ("sha256", self.hasher.hexdigest()),
        ):
            require(self.metadata[key] == value, "journal completion mismatch")
        require(self.metadata["max_row_tree_bytes"] <= TREE_LIMIT, "writer tree exceeded bound")


def check_action(row, *, task, files, phase, arm, index, launch, output):
    """Reconstruct the actual edit and inspect the saved executed source bytes."""
    exact_keys(
        row,
        (
            "schema",
            "phase",
            "arm",
            "index",
            "task_id",
            "attempt_id",
            "proposal_sha256",
            "workspace_sha256",
            "proposal",
            "before_files",
            "after_files",
            "rejection",
            "execution",
            "grade",
            "envelope",
            "apply_execute_grade_seconds",
            "action_completed_perf",
        ),
    )
    require(
        row["schema"] == "neuronal-splice-action-row-v1"
        and (row["phase"], row["arm"], row["index"]) == (phase, arm, index),
        "action identity/order mismatch",
    )
    require(
        row["task_id"] == (task["task_id"] if task else f"{phase}-{index}"), "action task mismatch"
    )
    require(
        ordered_files(row["before_files"]) == ordered_files(files), "action observation changed"
    )
    require(row["workspace_sha256"] == workspace_digest(files), "action workspace hash mismatch")
    require(
        row["proposal_sha256"] == proposal_digest(row["proposal"]), "action proposal hash mismatch"
    )
    require(
        row["proposal"]["workspace_sha256"] == row["workspace_sha256"], "stale action workspace"
    )
    after, rejection = apply_saved(files, row["proposal"])
    require(
        ordered_files(after, maximum=144) == ordered_files(row["after_files"], maximum=144),
        "action does not apply its selected bytes",
    )
    require((row["rejection"] is None) == (rejection is None), "action rejection mismatch")
    execution = row["execution"]
    validate_action_launch(execution, launch)
    require(
        "stdout" not in execution and "stderr" not in execution, "duplicate decoded process output"
    )
    for name in ("stdout", "stderr"):
        raw = hex_bytes(execution[name + "_hex"], 16384)
        require(
            hashlib.sha256(raw).hexdigest() == execution[name + "_sha256"],
            "raw output hash mismatch",
        )
    directory = output / "executions" / f"{phase}-{arm}-{index:06d}"
    require(
        Path(execution["workspace"]).resolve() == directory.resolve(),
        "execution directory mismatch",
    )
    require(len(str(directory).encode("utf-8")) <= 1024, "execution path exceeds declared bound")
    for path, content in ordered_files(after, maximum=144):
        require((directory / path).read_bytes() == content, "saved executed file changed")
    require(
        execution["materialized_matches_action"] is True, "executed source did not match proposal"
    )
    envelope = {
        "attempt_id": row["attempt_id"],
        "proposal_sha256": row["proposal_sha256"],
        "workspace_sha256": row["workspace_sha256"],
        "execution": {key: execution[key] for key in ("returncode", "stdout_hex", "stderr_hex")},
    }
    require(row["envelope"] == envelope, "action feedback differs from actual execution")
    positive(row["apply_execute_grade_seconds"])
    positive(row["action_completed_perf"])
    if task is None:
        require(row["grade"] is None, "engineering probe was graded as skill")
    else:
        require(
            row["grade"] == task_grade(task, row["proposal"], after, row["rejection"], execution),
            "external task grade disagrees with actual executed edit",
        )
    return row["grade"]


def cap_task():
    files = [
        {"path": "main.py", "content_hex": (b"print('cap')\n" + b"#" + b"A" * 113 + b"\n").hex()}
    ]
    files += [
        {"path": f"z{i}.py", "content_hex": (b"#" + b"A" * 126 + b"\n").hex()} for i in range(3)
    ]
    return {
        "task_id": "cap-0",
        "memory_id": "cap",
        "files": files,
        "workspace_sha256": workspace_digest(files),
        "request_hex": (b"Q" * 31).hex(),
        "teacher": {
            "operation": "SPLICE",
            "path": "main.py",
            "start": 0,
            "end": 0,
            "replacement_hex": (b"A" * 16).hex(),
            "stop_reason": "eos",
            "workspace_sha256": workspace_digest(files),
        },
    }


def check_learning_pair(
    actions,
    lessons,
    *,
    task,
    documents,
    phase,
    arm,
    index,
    launch,
    output,
    expected_before=None,
    expected_memory=None,
):
    """Own two full rows only; return scalar bindings for the later timing row."""
    action, action_hash = actions.next()
    row, _ = lessons.next()
    grade = check_action(
        action,
        task=task if phase == "regular" else None,
        files=task["files"],
        phase=phase,
        arm=arm,
        index=index,
        launch=launch,
        output=output,
    )
    exact_keys(
        row,
        (
            "schema",
            "phase",
            "arm",
            "index",
            "task_id",
            "memory_id",
            "request_hex",
            "workspace_sha256",
            "action_line_sha256",
            "prediction",
            "teacher",
            "feedback",
            "model_after_sha256",
            "memory_after_sha256",
            "model_seconds",
            "model_instance_token",
            "teacher_started_perf",
            "teacher_completed_perf",
        ),
    )
    require(
        row["schema"] == "neuronal-splice-learning-row-v2"
        and (row["phase"], row["arm"], row["index"]) == (phase, arm, index),
        "learning order changed",
    )
    for key in ("task_id", "memory_id", "request_hex", "workspace_sha256", "teacher"):
        require(row[key] == task[key], "learning task fields changed")
    prediction = row["prediction"]
    require(
        row["action_line_sha256"] == action_hash, "correction does not bind published actual action"
    )
    for key in ("attempt_id", "proposal_sha256", "proposal"):
        require(action[key] == prediction[key], "execution replaced the free proposal")
    require(
        prediction["attempt_id"].split(":")[0] == row["model_instance_token"],
        "correction did not retain the proposing instance",
    )
    require(
        positive(action["action_completed_perf"])
        <= positive(row["teacher_started_perf"])
        <= positive(row["teacher_completed_perf"]),
        "teacher preceded execution",
    )
    successor = documents_adjacency(documents)
    check_prediction(prediction, successor, arm=arm, boundary_list=boundaries(task["files"]))
    query = hex_bytes(task["request_hex"], 31)
    decisions = len(prediction["payload"]["steps"])
    require(
        prediction["work"]["reader"]
        == reader_work(query, len(successor), decisions, teaching=False, arm=arm),
        "free reader work mismatch",
    )
    require(
        prediction["work"]["action"]
        == action_work(query, task["files"], prediction["proposal"], teaching=False, arm=arm),
        "free action work mismatch",
    )
    offers, labels = check_feedback(
        row["feedback"], prediction, row["teacher"], task["files"], successor, query, arm
    )
    hash_value(row["model_after_sha256"])
    require(
        row["memory_after_sha256"] == prediction["memory_sha256"],
        "local correction rewrote episode memory",
    )
    require(
        expected_before is None or prediction["model_sha256"] == expected_before,
        "persistent learner weight chain broke",
    )
    require(
        expected_memory is None or prediction["memory_sha256"] == expected_memory,
        "learning episode identity changed",
    )
    if arm == "slow_frozen":
        require(row["model_after_sha256"] == prediction["model_sha256"], "frozen learner changed")
    if phase == "cap":
        proposal = prediction["proposal"]
        expected_payload = b"\0" * 16 if arm == "memory_disabled" else b"A" * 16
        require(
            proposal["operation"] == "STOP"
            and proposal["path"] == "main.py"
            and proposal["start"] == proposal["end"] == 0
            and hex_bytes(proposal["replacement_hex"]) == expected_payload
            and proposal["stop_reason"] == ("limit" if arm == "memory_disabled" else "eos")
            and decisions == offers == 17
            and labels == 3,
            "cap geometry/output incomplete",
        )
    return {
        "model_after": row["model_after_sha256"],
        "memory": row["memory_after_sha256"],
        "instance": row["model_instance_token"],
        "attempt_id": prediction["attempt_id"],
        "decisions": decisions,
        "offers": offers,
        "labels": labels,
        "publications": {
            **{
                "action_" + head: int(row["feedback"]["action_updates"].get(head, False))
                for head in HEAD_LIMITS
            },
            **{
                "reader_" + group: int((row["feedback"]["reader_updates"] or {}).get(group, False))
                for group in ("codec", "address")
            },
        },
        "model_seconds": positive(row["model_seconds"]),
        "action_partial_seconds": positive(action["apply_execute_grade_seconds"]),
        "grade": grade,
    }


def check_panel(row, case, *, phase, arm, index, documents, model_hash, memory_hash):
    exact_keys(
        row,
        (
            "schema",
            "phase",
            "arm",
            "index",
            "memory_id",
            "request_hex",
            "model_sha256",
            "memory_sha256",
            "prediction",
            "model_seconds",
        ),
    )
    require(
        row["schema"] == "neuronal-splice-panel-row-v2"
        and (row["phase"], row["arm"], row["index"]) == (phase, arm, index),
        "panel order mismatch",
    )
    require(
        row["memory_id"] == case["memory_id"] and row["request_hex"] == case["query_hex"],
        "panel request changed",
    )
    require(
        row["model_sha256"] == model_hash and row["memory_sha256"] == memory_hash,
        "reading panel state changed",
    )
    successor = documents_adjacency(documents)
    prediction = row["prediction"]
    require(prediction["kind"] == "free", "panel contains teacher activity")
    result = check_reader(prediction, successor, reader_policy(arm), max_bytes=2)
    require(
        prediction["work"]
        == reader_work(
            hex_bytes(case["query_hex"]),
            len(successor),
            result["decisions"],
            teaching=False,
            arm=arm,
        ),
        "panel work mismatch",
    )
    positive(row["model_seconds"])
    return (
        prediction["stop_reason"] == "eos" and prediction["generated_hex"] == case["expected_hex"]
    )


def tensor_digest(value, torch):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            raw = item.detach().contiguous().numpy().tobytes()
            digest.update(str((str(item.dtype), tuple(item.shape), len(raw))).encode())
            digest.update(raw)
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(json.dumps(key).encode())
                visit(item[key])
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())

    visit(value)
    return digest.hexdigest()


def validate_tensor(value, shape, dtype, torch):
    require(
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == shape
        and value.dtype == dtype
        and value.device.type == "cpu"
        and not value.requires_grad
        and value.grad_fn is None
        and value.grad is None
        and bool(torch.isfinite(value).all()),
        "invalid saved tensor contract",
    )
    return value.numel() * value.element_size()


def reconstruct_tape(snapshot, documents, codes, enabled, torch):
    """Independent literal storage reconstruction, never a neuronal model call."""
    exact_keys(snapshot, ("keys", "sensory", "successor", "write_index"))
    total = 0
    for name, shape, dtype in (
        ("keys", (128, 1024), torch.float32),
        ("sensory", (128, 257), torch.float32),
        ("successor", (128,), torch.int64),
        ("write_index", (), torch.int64),
    ):
        total += validate_tensor(snapshot[name], shape, dtype, torch)
    expected_keys = torch.zeros(128, 1024, dtype=torch.float64)
    expected_sensory = torch.zeros(128, 257, dtype=torch.float32)
    expected_successor = torch.full((128,), -1, dtype=torch.int64)
    position = 0
    for encoded in documents:
        raw = hex_bytes(encoded)
        preceding = []
        for token in (*raw, 256):
            require(position < 128, "saved tape exceeds capacity")
            if enabled:
                expected_sensory[position, token] = 1
                for lag, previous in enumerate(reversed(preceding[-32:])):
                    expected_keys[position, lag * 32 : (lag + 1) * 32] = codes[previous].double()
                norm = max(float(expected_keys[position].square().sum().sqrt()), 1e-8)
                expected_keys[position] /= norm
            expected_successor[position] = position if token == 256 else position + 1
            preceding.append(token)
            position += 1
    require(int(snapshot["write_index"].item()) == position, "saved tape allocation mismatch")
    require(
        torch.equal(snapshot["sensory"], expected_sensory)
        and torch.equal(snapshot["successor"], expected_successor),
        "saved literal/adjacency mismatch",
    )
    require(not bool(snapshot["keys"][position:].count_nonzero()), "unused keys are not exact zero")
    if not enabled:
        require(not bool(snapshot["keys"].count_nonzero()), "disabled keys are not exact zero")
    else:
        torch.testing.assert_close(
            snapshot["keys"][:position].double(), expected_keys[:position], rtol=2e-6, atol=2e-7
        )
    require(total == TAPE_BYTES, "saved tape storage changed")
    return position


def initial_action_tensors(torch):
    """Independent private Xavier initialization; no neuronal model construction."""
    result = {}
    for head, width, offset in (("start", 6177, 5), ("end", 2049, 6), ("operation", 4244, 7)):
        outputs = 2 if head == "operation" else 1
        generator = torch.Generator(device="cpu").manual_seed(733 + offset)
        value = torch.empty(outputs, width, dtype=torch.float32)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (outputs + width))
        value.uniform_(-bound, bound, generator=generator)
        result[f"actions.{head}_weight"] = value if outputs == 2 else value[0]
    result["actions.operation_bias"] = torch.zeros(2, dtype=torch.float32)
    return result


def tensor_payload_bytes(value, torch):
    """Count the complete loaded historical payload, including unreused tapes."""
    if isinstance(value, torch.Tensor):
        return validate_tensor(value, tuple(value.shape), value.dtype, torch)
    if isinstance(value, dict):
        return sum(tensor_payload_bytes(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_payload_bytes(item, torch) for item in value)
    require(type(value) in (str, int, float, bool, type(None)), "unexpected saved payload type")
    if type(value) is float:
        require(math.isfinite(value), "nonfinite saved metadata")
    return 0


def check_saved_state(output, report, roster):
    """Authenticate slow-state lineage and every retained episode without recall."""
    import torch

    torch.set_num_threads(1)
    prior_path = ROOT / "benchmarks/local_joint_address_online_v6.decision.json"
    require(
        sha(prior_path) == "f9b3b02b3bacf9977bb990f115072d41a043adeb074283d390bf45b5a0bf7a97",
        "prior completed decision changed",
    )
    prior_decision = read(prior_path)
    require(
        all(
            prior_decision.get(key) is True
            for key in (
                "full_run_completed",
                "full_audit_passed",
                "compliant_executed_learning_preserved",
            )
        ),
        "prior has no completed compliant learning evidence",
    )
    prior_completion_path = ROOT / ".pytest-tmp/local-joint-address-online-v6-run/completion.json"
    require(
        sha(prior_completion_path)
        == prior_decision["input_sha256"][
            str(prior_completion_path.relative_to(ROOT)).replace("\\", "/")
        ],
        "prior completion changed",
    )
    prior_completion = read(prior_completion_path)
    evidence_paths = {
        "report": prior_completion_path.parent / "report.json",
        "completion": prior_completion_path,
        "audit": ROOT / ".pytest-tmp/local-joint-address-online-v6-run-independent-audit.json",
    }
    evidence_hashes = {
        path.relative_to(ROOT).as_posix(): sha(path) for path in evidence_paths.values()
    }
    require(
        all(
            prior_decision["input_sha256"].get(path) == digest
            for path, digest in evidence_hashes.items()
        )
        and report["prior"]["evidence_sha256"] == evidence_hashes,
        "prior report/audit evidence changed",
    )
    prior_report, prior_audit = read(evidence_paths["report"]), read(evidence_paths["audit"])
    require(
        prior_report["status"] == "completed"
        and prior_report["mode"] == "run"
        and prior_completion["status"] == "passed"
        and prior_completion["mode"] == "run"
        and prior_audit["status"] == "passed"
        and prior_report["source_revision"]
        == prior_completion["source_revision"]
        == prior_decision["source_revision"]
        and prior_completion["report_sha256"] == sha(evidence_paths["report"])
        and prior_report["source_sha256"] == prior_report["source_sha256_after"],
        "prior completed source/audit chain mismatch",
    )
    reader_source = "src/humanbrain/local_memory_learner.py"
    require(
        prior_report["source_sha256"][reader_source]
        == sha(ROOT / reader_source)
        == report["prior"]["reader_source_sha256"]
        == "2a6bfe480757b248c585a00bdc9c79155af6abd3a162b3e93a098d406b73a3b8",
        "continued reader source changed",
    )
    expected_sources = set(prior_report["source_sha256"]) | {
        f"experiments/r4_integrated/{name}.py"
        for name in (
            "neuronal_splice_online_v1",
            "neuronal_splice_prior_v1",
            "neuronal_splice_roster_v1",
            "neuronal_splice_worker_v1",
            "neuronal_splice_io_v1",
            "neuronal_splice_receipts_v1",
            "neuronal_splice_endpoint_v1",
            "check_neuronal_splice_online_v1",
        )
    }
    expected_sources.update(
        ("src/humanbrain/neuronal_splice_learner.py", "src/humanbrain/neuronal_splice_action.py")
    )
    expected_sources.update(
        f"experiments/r4_integrated/{name}.py"
        for name in (
            "neuronal_splice_online_v2",
            "neuronal_splice_prior_v2",
            "neuronal_splice_roster_v2",
            "neuronal_splice_worker_v2",
            "neuronal_splice_endpoint_v2",
            "neuronal_splice_receipts_v2",
            "neuronal_splice_models_v2",
            "check_neuronal_address_competition_v1",
            "check_neuronal_splice_online_v2",
        )
    )
    expected_sources.update(
        (
            "src/humanbrain/competitive_local_memory_learner.py",
            "src/humanbrain/neuronal_address_competition.py",
        )
    )
    expected_sources.update(
        f"docs/research/neuronal-address-competition{suffix}.md"
        for suffix in ("-v1", "-integration-v1", "-comparison-v1")
    )
    require(set(report["source_sha256"]) == expected_sources, "parent source inventory changed")
    generated_path = prior_completion_path.parent / "source-free/generated.json"
    require(
        sha(generated_path) == prior_completion["artifacts"]["source-free/generated.json"],
        "prior endpoint metadata changed",
    )
    prior_generated = read(generated_path)
    prior_path = prior_completion_path.parent / "source-free/source_free.pt"
    require(
        sha(prior_path)
        == prior_completion["artifacts"]["source-free/source_free.pt"]
        == report["prior"]["checkpoint_sha256"]
        == "03654f61af4cccb93688a75e75a84c089800672530cdaffdc7a30f7cc5181180",
        "prior checkpoint changed",
    )
    require(
        prior_generated["checkpoint_sha256"] == report["prior"]["checkpoint_sha256"]
        and prior_generated["status"] == "completed"
        and prior_audit["source_free_tensor_bytes"]
        == report["prior"]["historical_payload_tensor_bytes"]
        == 33801664,
        "prior audited payload scope mismatch",
    )
    previous = torch.load(prior_path, map_location="cpu", weights_only=True)
    exact_keys(previous, ("learned", "address_frozen", "slow_frozen", "writes_disabled"))
    for historical_entry in previous.values():
        exact_keys(historical_entry, ("weights", "memories"))
    require(
        tensor_payload_bytes(previous, torch) == 33801664,
        "actual historical payload storage changed",
    )
    prior = previous["learned"]["weights"]
    exact_keys(
        prior, ("input_codes", "address_weight", "codec.weight", "codec.bias", "_extra_state")
    )
    require(
        report["prior"]["schema"] == "neuronal-splice-prior-custody-v2"
        and report["prior"]["selected_entry"] == "learned.weights"
        and report["prior"]["decision_path"]
        == "benchmarks/local_joint_address_online_v6.decision.json"
        and report["prior"]["decision_sha256"] == sha(ROOT / report["prior"]["decision_path"])
        and Path(report["prior"]["checkpoint_path"]).resolve() == prior_path.resolve()
        and report["prior"]["actual_loaded_payload_tensor_bytes"] == 33801664
        and report["prior"]["new_model_tensor_bytes"] == 3172160
        and report["prior"]["reader_policy"] == {arm: reader_policy(arm) for arm in ARMS},
        "prior continuation declaration changed",
    )
    numerical_prior = {key: value for key, value in prior.items() if key != "_extra_state"}
    require(
        tensor_digest(numerical_prior, torch) == report["prior"]["initial_reader_tensor_sha256"],
        "original reader numerical identity changed",
    )
    exact_keys(report["prior"]["metadata_fields_changed"], ARMS)
    expected_prior_sources = {
        name: report["source_sha256"][name]
        for name in (
            "experiments/r4_integrated/neuronal_splice_models_v2.py",
            "src/humanbrain/competitive_local_memory_learner.py",
            "src/humanbrain/neuronal_address_competition.py",
        )
    }
    require(
        report["prior"]["source_sha256"] == expected_prior_sources,
        "prior adapter source identity mismatch",
    )
    checkpoint = output / "endpoint/source_free.pt"
    require(sha(checkpoint) == report["worker"]["checkpoint_sha256"], "final checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    exact_keys(saved, ("schema", "arms"))
    require(saved["schema"] == "neuronal-splice-endpoint-state-v2", "saved state schema changed")
    exact_keys(saved["arms"], ARMS)
    geometries = {
        "reader.input_codes": ((258, 32), torch.float32),
        "reader.address_weight": ((32, 32), torch.float32),
        "reader.codec.weight": ((257, 257), torch.float64),
        "reader.codec.bias": ((257,), torch.float64),
        "actions.start_weight": ((6177,), torch.float32),
        "actions.end_weight": ((2049,), torch.float32),
        "actions.operation_weight": ((2, 4244), torch.float32),
        "actions.operation_bias": ((2,), torch.float32),
    }
    slots = retained = 0
    initial_hashes, final_hashes, memory_hashes = {}, {}, {}
    fresh_actions = initial_action_tensors(torch)
    action_hash = tensor_digest(
        {key.removeprefix("actions."): value for key, value in fresh_actions.items()}, torch
    )
    require(
        report["initial_action_sha256"] == {arm: action_hash for arm in ARMS},
        "initial action synapses differ from independent private initializer",
    )
    for arm in ARMS:
        entry = saved["arms"][arm]
        exact_keys(entry, ("weights", "memories", "experience"))
        require(isinstance(entry["weights"], dict), "saved model state must be a mapping")
        weights = dict(entry["weights"])
        exact_keys(weights, (*geometries, "_extra_state", "reader._extra_state"))
        model_bytes = sum(
            validate_tensor(weights[key], shape, dtype, torch)
            for key, (shape, dtype) in geometries.items()
        )
        require(model_bytes == MODEL_BYTES, "combined model storage changed")
        combined_mode = "learned" if arm == "marginal_reference" else arm
        metadata = {
            "schema": "neuronal-byte-splice-learner-v1",
            "mode": combined_mode,
            "seed": 733,
            "widths": {"start": 6177, "end": 2049, "operation": 4244},
            "feature_energy": 32,
            "operation_augmented_energy": 33,
            "learning_rate": 0.01,
            "payload_cap": 16,
            "publication": "validated action then atomic reader then copies",
        }
        require(weights["_extra_state"] == metadata, "action model metadata changed")
        expected_reader = expected_reader_metadata(prior["_extra_state"], arm)
        require(weights["reader._extra_state"] == expected_reader, "reader policy metadata changed")
        changed = sorted(
            key
            for key in set(prior["_extra_state"]) | set(expected_reader)
            if prior["_extra_state"].get(key) != expected_reader.get(key)
        )
        require(
            report["prior"]["metadata_fields_changed"][arm] == changed,
            "reported prior adaptation differs from exact declared fields",
        )
        require(
            torch.equal(weights["reader.input_codes"], prior["input_codes"]),
            "fixed input codes changed",
        )
        initial = {
            **fresh_actions,
            **{"reader." + key: value for key, value in prior.items()},
            "_extra_state": metadata,
            "reader._extra_state": expected_reader,
        }
        initial_hashes[arm] = tensor_digest(initial, torch)
        final_hashes[arm] = tensor_digest(weights, torch)
        require(
            initial_hashes[arm] == report["prior"]["initial_model_sha256"][arm],
            "initial model lineage mismatch",
        )
        require(
            final_hashes[arm]
            == report["final_model_sha256"][arm]
            == report["worker"]["initial_hashes"][arm]["model_sha256"],
            "final model identity mismatch",
        )
        if arm == "slow_frozen":
            require(initial_hashes[arm] == final_hashes[arm], "all-frozen slow state changed")
        if arm in ("action_frozen", "slow_frozen"):
            require(
                all(torch.equal(weights[key], value) for key, value in fresh_actions.items()),
                "frozen action synapses changed",
            )
        if arm == "memory_disabled":
            require(
                torch.equal(weights["reader.address_weight"], prior["address_weight"])
                and torch.equal(weights["reader.codec.weight"], prior["codec.weight"]),
                "zero sensory/key features changed their outgoing synapses",
            )
        require(
            entry["experience"] == (4 if report["mode"] == "cost" else 256),
            "saved action count changed",
        )
        require(
            len(entry["memories"]) == 72
            and set(entry["memories"]) == set(roster["memory_packages"]),
            "retained memory package scope changed",
        )
        memory_hashes[arm] = {}
        for memory_id, snapshot in entry["memories"].items():
            slots += reconstruct_tape(
                snapshot,
                roster["memory_packages"][memory_id]["document_hex"],
                weights["reader.input_codes"],
                arm != "memory_disabled",
                torch,
            )
            memory_hashes[arm][memory_id] = tensor_digest(snapshot, torch)
            retained += TAPE_BYTES
        retained += model_bytes
        require(
            memory_hashes[arm] == report["worker"]["initial_hashes"][arm]["memories_sha256"],
            "exported episode hashes mismatch",
        )
    require(
        retained == 239657600 and slots == 20950, "retained-state storage/ingestion ledger mismatch"
    )
    require(
        report["retained_tensor_bytes"] == report["worker"]["checkpoint_tensor_bytes"] == retained,
        "reported tensor storage differs from saved arrays",
    )
    for stage in ("before_learning", "after_learning"):
        ids = roster[stage + "_memory_ids"]
        documents = [raw for key in ids for raw in roster["memory_packages"][key]["document_hex"]]
        expected = {
            "states": len(ids),
            "raw_bytes": sum(len(hex_bytes(raw)) for raw in documents),
            "eos": len(documents),
        }
        for arm in ARMS:
            require(
                report["ingestion_" + stage.split("_")[0]][arm] == expected,
                "actual ingestion stage ledger differs from frozen packages",
            )
    del prior, numerical_prior, previous, historical_entry, saved, entry, weights, initial
    return {
        "initial_model_sha256": initial_hashes,
        "final_model_sha256": final_hashes,
        "memory_sha256": memory_hashes,
        "retained_tensor_bytes": retained,
        "reconstructed_slots": slots,
        "model_forwards": 0,
    }


def check_monitor(outer, *, pid, command=None):
    require(outer["status"] == "passed" and outer["exit_code"] == 0, "monitor did not complete")
    require(
        outer["seconds_limit"] == 900 and outer["tree_peak_sum_mib_limit"] == 2048,
        "monitor ceilings changed",
    )
    require(
        0 < number(outer["elapsed_seconds"]) <= 900
        and 0 < number(outer["observed_tree_peak_sum_mib"]) <= 2048,
        "observed budget exceeded",
    )
    require(outer["deadline_unix"] - outer["started_unix"] == 900, "deadline mismatch")
    require(command is None or outer["command"] == command, "monitor command mismatch")
    for actual in {pid, outer["child_pid"]}:
        identities = [
            key for key in outer["observed_process_peaks_mib"] if key.split("/")[0] == str(actual)
        ]
        require(len(identities) == 1, "actual process absent or creation identity ambiguous")
        created = (int(identities[0].split("/")[1]) - 621355968000000000) / 10_000_000
        require(
            outer["started_unix"] <= created <= outer["started_unix"] + outer["elapsed_seconds"],
            "process creation outside selected launch",
        )


def owned_identity(identity, outer, *, lower=None):
    exact_keys(identity, ("pid", "creation_ticks"))
    require(
        type(identity["pid"]) is int
        and identity["pid"] > 0
        and type(identity["creation_ticks"]) is int,
        "invalid process identity",
    )
    key = f"{identity['pid']}/{identity['creation_ticks']}"
    require(
        key in outer["observed_process_peaks_mib"], "actual process lacks owned creation identity"
    )
    created = (identity["creation_ticks"] - 621355968000000000) / 10_000_000
    require(
        max(outer["started_unix"], lower or outer["started_unix"])
        <= created
        <= outer["started_unix"] + outer["elapsed_seconds"],
        "worker creation outside selected launch",
    )


def check_worker_custody(output, report, worker, requests, lineage, outer):
    """Bind the actual isolated child, its imports, saved input and final state."""
    isolated = output / "endpoint"
    package = isolated / "package"
    launch_path = isolated / "worker-launch.json"
    require(
        sha(launch_path) == report["worker"]["worker_launch_sha256"],
        "worker launcher record changed",
    )
    launch = read(launch_path)
    require(
        launch["parent_pid"] == report["pid"] and Path(launch["cwd"]).resolve() == isolated,
        "worker launcher belongs to another parent/directory",
    )
    require(
        outer["started_unix"]
        <= number(launch["started_unix"])
        <= outer["started_unix"] + outer["elapsed_seconds"],
        "worker launched outside monitored turn",
    )
    expected = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-I",
        "-X",
        "utf8",
        str(package / WORKER_SOURCE),
        "--package-root",
        str(package),
        "--checkpoint",
        str(isolated / "source_free.pt"),
        "--checkpoint-sha",
        report["worker"]["checkpoint_sha256"],
        "--requests",
        str(isolated / "query_only.json"),
        "--requests-sha",
        report["worker"]["requests_sha256"],
        "--output",
        str(isolated / "worker"),
        "--deadline-unix",
        repr(report["deadline_unix"]),
    ]
    require(
        launch["command"] == expected, "actual worker command differs from selected source/input"
    )
    require(
        worker["schema"] == "neuronal-splice-endpoint-completion-v2"
        and worker["scope"] == "integrated",
        "QA worker scope cannot admit integrated work",
    )
    require(
        worker["process_identity"]["pid"] == worker["pid"], "worker identity names another process"
    )
    require(
        worker["pid"] == launch["pid"] or worker["parent_pid"] == launch["pid"],
        "actual interpreter is not the selected launcher or its direct child",
    )
    owned_identity(worker["process_identity"], outer, lower=launch["started_unix"])
    if launch["process_identity"] is not None:
        require(
            launch["process_identity"]["pid"] == launch["pid"], "launcher identity PID mismatch"
        )
        owned_identity(launch["process_identity"], outer, lower=launch["started_unix"])
    else:
        require(
            type(launch.get("identity_read_error")) is str, "unexplained missing launcher identity"
        )
        keys = [
            key
            for key in outer["observed_process_peaks_mib"]
            if key.split("/")[0] == str(launch["pid"])
        ]
        require(len(keys) == 1, "launcher lacks independent creation evidence")
        owned_identity(
            {"pid": launch["pid"], "creation_ticks": int(keys[0].split("/")[1])},
            outer,
            lower=launch["started_unix"],
        )
    for key in ("checkpoint_sha256", "requests_sha256"):
        require(
            worker[key] == launch[key] == report["worker"][key], "worker input custody mismatch"
        )
    require(
        requests["schema"] == "neuronal-splice-endpoint-request-v2"
        and requests["checkpoint_sha256"] == worker["checkpoint_sha256"],
        "request checkpoint mismatch",
    )
    require(
        set(requests["source_sha256"]) == {*WORKER_MODULES.values(), WORKER_SOURCE},
        "isolated source inventory is not the fixed teacher-free closure",
    )
    require(
        worker["source_origins"]
        == {name: str(package / relative) for name, relative in WORKER_MODULES.items()},
        "actual imported module origins differ from isolated package",
    )
    exact_keys(worker["final"], ARMS)
    expected_experience = 4 if report["mode"] == "cost" else 256
    for arm in ARMS:
        initial = report["worker"]["initial_hashes"][arm]
        require(initial["experience"] == expected_experience, "exported experience mismatch")
        require(
            worker["final"][arm]
            == {
                "model_sha256": lineage["final_model_sha256"][arm],
                "memories_sha256": lineage["memory_sha256"][arm],
                "experience": expected_experience + 48,
            },
            "worker final state/experience changed",
        )
    stderr_path = isolated / "worker-stderr.bin"
    require(
        stderr_path.stat().st_size <= 1024
        and sha(stderr_path) == report["worker"]["worker_stderr_sha256"],
        "worker diagnostics changed/exceeded bound",
    )
    positive(worker["memory"]["peak_working_set_mib"])


def expected_runtime_bootstrap():
    return (
        "import sys, runpy; assert sys.flags.utf8_mode == 1; "
        "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
        "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
        "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
        "sys.path.insert(0, '.'); "
        "import io, os\n"
        "with io.open_code(os.path.abspath('main.py')) as source:\n"
        "    code = compile(source.read(), 'main.py', 'exec')\n"
        "runpy._run_module_code(code, mod_name='__main__', pkg_name='', script_name='main.py')"
    )


def expected_runtime_bootstrap_sha256():
    return hashlib.sha256(expected_runtime_bootstrap().encode()).hexdigest()


def expected_action_bootstrap():
    return (
        "import sys; "
        "assert sys.flags.utf8_mode == 1; "
        "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
        "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
        "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
        "sys.path.insert(0, '.'); "
        "import io, os\n"
        "with io.open_code(os.path.abspath('main.py')) as source:\n"
        "    code = compile(source.read(), 'main.py', 'exec')\n"
        "def _run_source(code):\n"
        "    temporary_main = type(sys)('__main__')\n"
        "    original_main = sys.modules['__main__']\n"
        "    sys.modules['__main__'] = temporary_main\n"
        "    try:\n"
        "        original_argv0 = sys.argv[0]\n"
        "        sys.argv[0] = 'main.py'\n"
        "        try:\n"
        "            namespace = temporary_main.__dict__\n"
        "            namespace.update(__name__='__main__', __file__='main.py',\n"
        "                             __cached__=None, __doc__=None, __loader__=None,\n"
        "                             __package__='', __spec__=None)\n"
        "            exec(code, namespace)\n"
        "        finally:\n"
        "            sys.argv[0] = original_argv0\n"
        "    finally:\n"
        "        sys.modules['__main__'] = original_main\n"
        "    return namespace.copy()\n"
        "_run_source(code)"
    )


def expected_action_bootstrap_sha256():
    return hashlib.sha256(expected_action_bootstrap().encode()).hexdigest()


def validate_bootstrap_qualification(execution):
    """Authenticate eight reviewed temporary-main pairs, never reexecute them."""
    policy = "regular_source_public_temporary_module_v1"
    assert (
        execution["qualification_sha256"]
        == "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c"
    )
    assert execution["bootstrap_policy"] == policy
    assert execution["bootstrap_sha256"] == expected_action_bootstrap_sha256()
    path = pathlib.Path(execution["bootstrap_qualification_path"])
    assert path.is_absolute() and sha(path) == execution["bootstrap_qualification_sha256"]
    manifest = read(path)
    assert set(manifest) == {
        "schema",
        "status",
        "runtime_qualification_sha256",
        "bridge_sha256",
        "bootstrap_policy",
        "bootstrap_sha256",
        "paired_cases",
        "child_executions",
        "evidence",
    }
    assert (
        manifest["schema"] == "temporary-module-bootstrap-qualification-v1"
        and manifest["status"] == "passed"
    )
    assert manifest["runtime_qualification_sha256"] == execution["qualification_sha256"]
    assert manifest["bridge_sha256"] == sha(
        ROOT / "experiments/r4_integrated/temporary_module_action_bridge_v1.py"
    )
    assert manifest["bootstrap_policy"] == policy
    assert manifest["bootstrap_sha256"] == expected_action_bootstrap_sha256()
    assert type(manifest["paired_cases"]) is int and manifest["paired_cases"] == 8
    assert type(manifest["child_executions"]) is int and manifest["child_executions"] == 16
    assert isinstance(manifest["evidence"], dict) and manifest["evidence"]
    for evidence, expected in manifest["evidence"].items():
        assert pathlib.Path(evidence).is_absolute() and sha(evidence) == expected
    return {
        "bootstrap_qualification_sha256": execution["bootstrap_qualification_sha256"],
        "paired_cases": 8,
        "child_executions": 16,
        "authenticated_evidence_files": len(manifest["evidence"]),
    }


def validate_action_launch(execution, launch):
    """Scalar identity only per action; fixed file hashing occurs at boundaries."""
    assert execution["bootstrap_policy"] == "regular_source_public_temporary_module_v1"
    assert execution["bootstrap_sha256"] == expected_action_bootstrap_sha256()
    assert execution["launch_record_sha256"] == hashlib.sha256(canonical(launch)).hexdigest()
    assert execution["child_command"] == [
        launch["base_executable"],
        "-I",
        "-S",
        "-X",
        "utf8",
        "-B",
        "-c",
        expected_action_bootstrap(),
    ]
    assert execution["child_environment_override"] == {}
    assert execution["child_environment_removed"] == ["__PYVENV_LAUNCHER__"]
    assert execution["child_python_flags"] == ["-I", "-S", "-X", "utf8", "-B"]


def validate_launch_record(launch):
    assert set(launch) == {
        "policy",
        "parent_runtime",
        "base_executable",
        "flags",
        "bootstrap_policy",
        "bootstrap_sha256",
        "environment_removed",
        "artifacts",
    }
    assert launch["policy"] == "base_stdlib_python_no_site_v1"
    validate_parent_runtime(launch["parent_runtime"])
    assert launch["base_executable"] == launch["parent_runtime"]["base_executable"]
    assert launch["flags"] == ["-I", "-S", "-X", "utf8", "-B"]
    assert launch["environment_removed"] == ["__PYVENV_LAUNCHER__"]
    assert launch["bootstrap_policy"] == "regular_source_runpy_module_code_v1"
    assert launch["bootstrap_sha256"] == expected_runtime_bootstrap_sha256()
    expected = dict(launch["parent_runtime"]["artifacts"])
    for name in ("stdlib_python_action_bridge_v1.py", "source_python_action_bridge_v1.py"):
        path = (ROOT / "experiments/r4_integrated" / name).resolve()
        expected[str(path)] = sha(path)
    assert launch["artifacts"] == expected
    return hashlib.sha256(canonical(launch)).hexdigest()


def validate_parent_runtime(launch):
    assert set(launch) == {
        "policy",
        "venv_executable",
        "base_executable",
        "venv_config",
        "python_version",
        "implementation",
        "base_prefix",
        "base_exec_prefix",
        "artifacts",
    }
    assert launch["policy"] == "direct_base_python_with_selected_venv_launcher_v1"
    assert launch["implementation"] == "cpython" and launch["python_version"].startswith("3.11.")
    for key in (
        "venv_executable",
        "base_executable",
        "venv_config",
        "base_prefix",
        "base_exec_prefix",
    ):
        assert pathlib.Path(launch[key]).is_absolute()
    venv, base, config = (
        pathlib.Path(launch[key]).resolve()
        for key in ("venv_executable", "base_executable", "venv_config")
    )
    assert venv == (ROOT / ".venv/Scripts/python.exe").resolve() and base != venv
    assert config == (ROOT / ".venv/pyvenv.cfg").resolve()
    entries = dict(
        line.split(" = ", 1) for line in config.read_text().splitlines() if " = " in line
    )
    assert pathlib.Path(entries["home"]).resolve() == base.parent
    assert pathlib.Path(launch["base_prefix"]).resolve() == base.parent
    assert pathlib.Path(launch["base_exec_prefix"]).resolve() == base.parent
    expected = {
        venv,
        base,
        config,
        (ROOT / "experiments/r4_integrated/direct_python_action_bridge_v1.py").resolve(),
        base.parent / "python311.dll",
    }
    expected.update(
        base.parent / "Lib" / name
        for name in (
            "runpy.py",
            "site.py",
            "subprocess.py",
            "encodings/__init__.py",
            "multiprocessing/popen_spawn_win32.py",
        )
    )
    artifacts = launch["artifacts"]
    assert {pathlib.Path(path).resolve() for path in artifacts} == expected
    assert len(artifacts) == len(expected)
    for path, expected_hash in artifacts.items():
        assert pathlib.Path(path).is_absolute() and sha(path) == expected_hash
    return hashlib.sha256(canonical(launch)).hexdigest()


def validate_qualification(path, expected_hash, launch):
    """Authenticate reviewed identity/semantic/isolation evidence, never execute it."""
    path = pathlib.Path(path)
    assert path.is_absolute() and sha(path) == expected_hash
    qualification = read(path)
    assert set(qualification) == {
        "schema",
        "status",
        "launch_record",
        "executions",
        "identity",
        "semantic_pairs",
        "contamination",
        "evidence",
    }
    assert qualification["schema"] == "stdlib-python-action-qualification-v1"
    assert qualification["status"] == "passed" and qualification["launch_record"] == launch
    assert qualification["executions"] == {"identity": 1, "semantic_pairs": 14, "contamination": 2}
    assert all(type(value) is int for value in qualification["executions"].values())
    evidence = qualification["evidence"]
    assert isinstance(evidence, dict) and evidence
    for filename, expected in evidence.items():
        assert pathlib.Path(filename).is_absolute() and sha(filename) == expected
    names = (
        "namespace_cleanup",
        "utf8_bom_crlf",
        "latin1_cookie",
        "syntax_error",
        "runtime_error",
        "exit_invalid_utf8",
        "pending",
    )
    assert len(qualification["semantic_pairs"]) == 7
    for name in ("identity", "contamination"):
        item = qualification[name]
        assert set(item) == {"passed", "receipt_path", "receipt_sha256"}
        assert item["passed"] is True
        assert pathlib.Path(item["receipt_path"]).is_absolute()
        assert evidence[item["receipt_path"]] == item["receipt_sha256"]
    for index, (name, item) in enumerate(zip(names, qualification["semantic_pairs"], strict=True)):
        assert set(item) == {"passed", "receipt_path", "receipt_sha256", "case", "case_index"}
        assert item["passed"] is True and item["case"] == name
        assert type(item["case_index"]) is int and item["case_index"] == index
        assert pathlib.Path(item["receipt_path"]).is_absolute()
        assert evidence[item["receipt_path"]] == item["receipt_sha256"]
    return {
        "qualification_sha256": expected_hash,
        "identity_executions": 1,
        "semantic_executions": 14,
        "contamination_executions": 2,
        "authenticated_evidence_files": len(evidence),
    }


def positive(value):
    require(number(value) > 0, "expected positive resource measurement")
    return float(value)


def project_cost(report, outer, *, audit=None, audit_outer=None):
    """Independent separate maxima, including the replicated timing append."""
    require(report["mode"] == "cost" and report["status"] == "completed", "incomplete cost")
    require((audit is None) == (audit_outer is None), "unfinished cost-audit monitor")
    require(len(report["category_probes"]) == 4, "missing action category coverage")
    category = max(positive(item["action_seconds"]) for item in report["category_probes"])
    maxima, regular_total = {}, 0.0
    for arm in ARMS:
        phases = report["timing_summary"][arm]
        require(
            phases["regular"]["rows"] == 4 and phases["cap"]["rows"] == 1,
            "incomplete cost learning/cap scope",
        )
        maxima[arm] = {}
        for stage in ("model", "action", "journal"):
            samples = []
            for phase, count in (("regular", 4), ("cap", 1)):
                total = positive(phases[phase][stage + "_seconds"])
                maximum = positive(phases[phase][stage + "_max_seconds"])
                require(maximum <= total <= count * maximum + 1e-8, "invalid stage summary")
                samples.append(maximum)
                if phase == "regular":
                    regular_total += total
            if stage == "action":
                samples.append(category)
            maxima[arm][stage] = max(samples)
    residual = positive(outer["elapsed_seconds"]) - regular_total
    require(residual >= 0, "cost stage sums exceed outer duration")
    run_seconds = 512 * sum(sum(row.values()) for row in maxima.values()) + 2 * residual + 30
    parent_worker = positive(report["memory"]["peak_working_set_mib"]) + positive(
        report["worker"]["worker"]["memory"]["peak_working_set_mib"]
    )
    run_peak = max(parent_worker, positive(outer["observed_tree_peak_sum_mib"])) + 24 + 128
    audit_seconds = audit_peak = None
    if audit is not None:
        require(audit["status"] == "passed", "independent cost audit failed")
        check_monitor(audit_outer, pid=audit["pid"])
        sampled, maximum_sum = 0.0, 0.0
        for arm in ARMS:
            phases = audit["learning_pair_timing"][arm]
            require(
                phases["regular"]["rows"] == 4 and phases["cap"]["rows"] == 1,
                "audit omitted regular/cap pairs",
            )
            for phase, count in (("regular", 4), ("cap", 1)):
                row = phases[phase]
                require(
                    0
                    < number(row["max_seconds"])
                    <= number(row["seconds"])
                    <= count * number(row["max_seconds"]) + 1e-8,
                    "invalid audit pair timing",
                )
            sampled += phases["regular"]["seconds"]
            maximum_sum += max(phases[phase]["max_seconds"] for phase in ("regular", "cap"))
        remainder = positive(audit_outer["elapsed_seconds"]) - sampled
        require(remainder >= 0, "audit pair sums exceed monitor duration")
        audit_seconds = 2 * (256 * maximum_sum + remainder)
        audit_peak = positive(audit_outer["observed_tree_peak_sum_mib"]) + 24 + 32
    total = run_seconds + (audit_seconds or 0)
    peak = max(run_peak, audit_peak or 0)
    return dict(
        per_arm=maxima,
        category_action_seconds=category,
        projected_run_seconds=run_seconds,
        projected_audit_seconds=audit_seconds,
        projected_seconds=total,
        run_projected_peak_mib=run_peak,
        audit_projected_peak_mib=audit_peak,
        projected_tree_mib=peak,
        audit_peak_pending=audit is None,
        admissible=audit is not None and total <= 900 and peak <= 2048,
    )


def validate_cost_decision(path, expected_sha, expectedconfig, expectedsources, rostersha):
    """Reauthenticate finished admission using saved JSON/hashes, never models.

    Called before acquisition by the parent and again by the final auditor.
    The completed audit's own watchdog supplies its otherwise unavailable peak.
    """
    path = Path(path)
    require(path.is_absolute() and sha(path) == expected_sha, "cost decision identity changed")
    decision = read(path)
    exact_keys(
        decision,
        (
            "schema",
            "status",
            "config",
            "source_sha256",
            "roster_sha256",
            "completed_cost_audit",
            "inputs",
            "projection",
            "projected_seconds",
            "projected_tree_mib",
        ),
    )
    require(
        decision["schema"] == "neuronal-splice-cost-decision-v2"
        and decision["status"] == "passed"
        and decision["config"] == expectedconfig == expected_config()
        and decision["source_sha256"] == expectedsources
        and decision["roster_sha256"] == rostersha
        and decision["completed_cost_audit"] is True,
        "cost decision is not this complete protocol",
    )
    names = (
        "cost_report",
        "cost_completion",
        "cost_launch",
        "cost_admission",
        "cost_audit",
        "cost_audit_launch",
    )
    exact_keys(decision["inputs"], names)
    paths, values, hashes = {}, {}, {}
    for name in names:
        entry = decision["inputs"][name]
        exact_keys(entry, ("path", "sha256"))
        paths[name] = Path(entry["path"])
        require(
            paths[name].is_absolute() and sha(paths[name]) == entry["sha256"],
            "cost evidence changed",
        )
        values[name], hashes[name] = read(paths[name]), entry["sha256"]
    report, complete, outer, admission, audited, audit_outer = (values[name] for name in names)
    base = paths["cost_report"].parent
    require(
        paths["cost_report"].name == "report.json"
        and paths["cost_completion"] == base / "completion.json",
        "cost report/completion pairing mismatch",
    )
    require(
        admission["schema"] == "neuronal-splice-admission-v2"
        and admission["status"] == "passed"
        and admission["authorized_phase"] == "cost"
        and admission["qa_passed"] is True
        and admission["source_sha256"] == expectedsources
        and admission["config"] == expectedconfig
        and admission["roster_sha256"] == rostersha,
        "base cost admission mismatch",
    )
    require(
        report["schema"] == "neuronal-splice-online-v2"
        and report["status"] == "completed"
        and report["mode"] == "cost"
        and report["config"] == expectedconfig
        and report["source_sha256"] == report["source_sha256_after"] == expectedsources
        and report["admission_sha256"] == hashes["cost_admission"]
        and report["roster_sha256"] == rostersha,
        "cost report bindings mismatch",
    )
    require(
        complete["schema"] == "neuronal-splice-completion-v2"
        and complete["status"] == "completed"
        and complete["mode"] == "cost"
        and complete["report_sha256"] == hashes["cost_report"]
        and complete["admission_sha256"] == hashes["cost_admission"]
        and complete["logs"] == report["logs"],
        "cost completion mismatch",
    )
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-X",
        "utf8",
        "-m",
        "experiments.r4_integrated.neuronal_splice_online_v2",
        "cost",
        "--output",
        str(base),
        "--admission",
        str(paths["cost_admission"]),
        "--admission-sha",
        hashes["cost_admission"],
    ]
    check_monitor(outer, pid=report["pid"], command=command)
    require(report["deadline_unix"] == outer["deadline_unix"], "cost monitor deadline mismatch")
    require(
        audited["schema"] == "neuronal-splice-independent-audit-v2"
        and audited["status"] == "passed"
        and audited["mode"] == "cost"
        and audited["source_sha256"] == expectedsources
        and audited["roster_sha256"] == rostersha
        and audited["report_sha256"] == hashes["cost_report"]
        and audited["completion_sha256"] == hashes["cost_completion"]
        and audited["launch_sha256"] == hashes["cost_launch"]
        and audited["admission_sha256"] == hashes["cost_admission"],
        "independent cost audit pairing mismatch",
    )
    checker = Path(__file__).resolve()
    require(
        expectedsources.get("experiments/r4_integrated/check_neuronal_splice_online_v2.py")
        == sha(checker),
        "completed audit checker is not the held source",
    )
    for name in ("cost_report", "cost_completion", "cost_launch", "cost_admission"):
        require(
            audited["input_sha256"].get(str(paths[name])) == hashes[name],
            "audit did not authenticate selected input",
        )
    require(
        audited["input_sha256"].get(str(checker)) == sha(checker), "audit checker identity absent"
    )
    roster_path = Path(admission["roster_path"])
    require(
        roster_path.is_absolute()
        and sha(roster_path) == rostersha
        and audited["input_sha256"].get(str(roster_path)) == rostersha,
        "audit roster custody mismatch",
    )
    audit_command = audit_outer["command"]
    prefix = [str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8"]
    require(audit_command[:3] == prefix, "audit ran outside the selected interpreter")
    if audit_command[3:5] == ["-m", "experiments.r4_integrated.check_neuronal_splice_online_v2"]:
        flags = audit_command[5:]
    else:
        require(audit_command[3] == str(checker), "audit ran a different checker")
        flags = audit_command[4:]
    require(len(flags) == 10, "audit argument count mismatch")
    expected_flags = {
        "--output": str(paths["cost_audit"]),
        "--cost-dir": str(base),
        "--launch": str(paths["cost_launch"]),
        "--admission": str(paths["cost_admission"]),
        "--admission-sha": hashes["cost_admission"],
    }
    require(
        dict(zip(flags[::2], flags[1::2], strict=True)) == expected_flags,
        "audit arguments describe other artifacts",
    )
    check_monitor(audit_outer, pid=audited["pid"])
    require(
        audit_outer["launcher_sha256"] == outer["launcher_sha256"],
        "cost/audit monitor source differs",
    )
    preliminary = project_cost(report, outer)
    require(
        audited["projection"] == preliminary
        and preliminary["audit_peak_pending"]
        and not preliminary["admissible"],
        "preliminary audit incorrectly admitted acquisition",
    )
    final = project_cost(report, outer, audit=audited, audit_outer=audit_outer)
    require(
        final == decision["projection"]
        and final["admissible"]
        and not final["audit_peak_pending"]
        and decision["projected_seconds"] == final["projected_seconds"]
        and decision["projected_tree_mib"] == final["projected_tree_mib"],
        "final admission projection mismatch",
    )
    return final


def audit(args, result):
    """Stream completed records; no model, correction or child execution occurs."""
    require(not DRAFT, "draft auditor cannot read genuine artifacts")
    started = time.perf_counter()
    require((args.cost_dir is None) != (args.run_dir is None), "select exactly one saved phase")
    output = (args.cost_dir or args.run_dir).resolve()
    mode = "cost" if args.cost_dir else "run"
    require(sha(args.admission) == args.admission_sha, "admission identity changed")
    admission, report, completion, outer = (
        read(path)
        for path in (
            args.admission,
            output / "report.json",
            output / "completion.json",
            args.launch,
        )
    )
    require(
        admission["schema"] == "neuronal-splice-admission-v2"
        and admission["status"] == "passed"
        and admission["authorized_phase"] == mode
        and admission["qa_passed"] is True,
        "execution was not admitted",
    )
    require(
        report["schema"] == "neuronal-splice-online-v2"
        and report["status"] == "completed"
        and report["mode"] == mode
        and report["admission_sha256"] == args.admission_sha
        and report["config"] == admission["config"] == expected_config()
        and report["source_sha256"] == report["source_sha256_after"] == admission["source_sha256"],
        "report/admission/source chain mismatch",
    )
    require(
        completion["schema"] == "neuronal-splice-completion-v2"
        and completion["status"] == "completed"
        and completion["mode"] == mode
        and completion["report_sha256"] == sha(output / "report.json")
        and completion["admission_sha256"] == args.admission_sha
        and completion["logs"] == report["logs"],
        "completion chain mismatch",
    )
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-X",
        "utf8",
        "-m",
        "experiments.r4_integrated.neuronal_splice_online_v2",
        mode,
        "--output",
        str(output),
        "--admission",
        str(args.admission),
        "--admission-sha",
        args.admission_sha,
    ]
    check_monitor(outer, pid=report["pid"], command=command)
    require(report["deadline_unix"] == outer["deadline_unix"], "run deadline differs from watchdog")
    for relative, expected in report["source_sha256"].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == expected, "held source changed")
    roster_path = Path(admission["roster_path"])
    require(
        sha(roster_path) == admission["roster_sha256"] == report["roster_sha256"],
        "frozen roster changed",
    )
    roster = read(roster_path)
    check_roster(roster)
    if mode == "run":
        validate_cost_decision(
            Path(admission["cost_decision_path"]),
            admission["cost_decision_sha256"],
            expected_config(),
            report["source_sha256"],
            report["roster_sha256"],
        )
    runtime = report["action_launch_record"]
    validate_launch_record(runtime)
    validate_qualification(
        (ROOT / ".pytest-tmp/stdlib-python-action-v1-qualification.json").resolve(),
        "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c",
        runtime,
    )
    validate_bootstrap_qualification(
        {
            "qualification_sha256": (
                "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c"
            ),
            "bootstrap_policy": "regular_source_public_temporary_module_v1",
            "bootstrap_sha256": expected_action_bootstrap_sha256(),
            "bootstrap_qualification_path": str(
                (ROOT / ".pytest-tmp/temporary-module-bootstrap-v1-qualification.json").resolve()
            ),
            "bootstrap_qualification_sha256": (
                "0bb809fd33aee145f169776a03a884fd62daf381ff4e67ed333a2a378c46d92b"
            ),
        }
    )
    lineage = check_saved_state(output, report, roster)
    worker_path = output / "endpoint/worker/completion.json"
    worker = read(worker_path)
    require(
        sha(worker_path) == report["worker"]["worker_completion_sha256"]
        and worker == report["worker"]["worker"]
        and worker["status"] == "passed",
        "worker completion changed",
    )
    request_path = output / "endpoint/query_only.json"
    require(
        sha(request_path) == worker["requests_sha256"] == report["worker"]["requests_sha256"],
        "worker observations changed",
    )
    requests = read(request_path)
    task_ids = roster["fresh_task_ids"] + roster["retention_task_ids"]
    expected_cases = [
        {
            "id": task_id,
            "memory_id": roster["tasks"][task_id]["memory_id"],
            "request_hex": roster["tasks"][task_id]["request_hex"],
            "files": roster["tasks"][task_id]["files"],
        }
        for task_id in task_ids
    ]
    expected_reading = [
        {
            "id": str(case["case_id"]),
            "memory_id": case["memory_id"],
            "request_hex": case["query_hex"],
        }
        for case in roster["old_panel"]
    ]
    exact_keys(
        requests,
        (
            "schema",
            "checkpoint_sha256",
            "source_sha256",
            "receipt_limits",
            "arms",
            "cases",
            "reading_cases",
        ),
    )
    require(
        requests["cases"] == expected_cases
        and requests["reading_cases"] == expected_reading
        and requests["arms"] == list(ARMS)
        and requests["receipt_limits"] == report["config"]["receipt_limits"],
        "teacher-free worker request contains changed observations",
    )
    check_worker_custody(output, report, worker, requests, lineage, outer)
    package = output / "endpoint/package"
    require(
        requests["source_sha256"] == worker["source_sha256"] == report["worker"]["source_sha256"],
        "worker source chain mismatch",
    )
    require(
        {path.relative_to(package).as_posix() for path in package.rglob("*.py")}
        == set(requests["source_sha256"]),
        "isolated package contains undeclared source",
    )
    for relative, expected in requests["source_sha256"].items():
        require(
            sha(package / relative) == sha(ROOT / relative) == expected, "isolated source changed"
        )
    actions = Stream(output / "actions.jsonl", report["logs"]["actions"])
    lessons = Stream(output / "lessons.jsonl", report["logs"]["lessons"])
    timing = Stream(output / "timings.jsonl", report["logs"]["timings"])
    pairs = {
        arm: {
            phase: {"rows": 0, "seconds": 0.0, "max_seconds": 0.0} for phase in ("regular", "cap")
        }
        for arm in ARMS
    }
    measured = {
        arm: {
            phase: {
                "rows": 0,
                "model": 0.0,
                "action": 0.0,
                "journal_partial": 0.0,
                "model_max": 0.0,
                "action_max": 0.0,
                "journal_partial_max": 0.0,
            }
            for phase in ("regular", "cap")
        }
        for arm in ARMS
    }
    chain = dict(lineage["initial_model_sha256"])
    instances = {}
    counts = {"learning_pairs": 0, "free_decisions": 0, "reader_offers": 0, "action_labels": 0}
    curves = {arm: [] for arm in ARMS}
    publications = {
        arm: {
            phase: {
                key: 0
                for key in (
                    "action_start",
                    "action_end",
                    "action_operation",
                    "reader_codec",
                    "reader_address",
                )
            }
            for phase in ("regular", "cap")
        }
        for arm in ARMS
    }

    def pair(task, documents, phase, arm, index):
        began = time.perf_counter()
        values = check_learning_pair(
            actions,
            lessons,
            task=task,
            documents=documents,
            phase=phase,
            arm=arm,
            index=index,
            launch=runtime,
            output=output,
            expected_before=chain[arm] if phase == "regular" else None,
            expected_memory=lineage["memory_sha256"][arm][task["memory_id"]]
            if phase == "regular"
            else None,
        )
        # The two full rows have been released before reading this small timing row.
        row, _ = timing.next()
        exact_keys(
            row,
            (
                "phase",
                "arm",
                "index",
                "model_seconds",
                "action_seconds",
                "journal_before_timing_append_seconds",
            ),
        )
        require(
            (row["phase"], row["arm"], row["index"]) == (phase, arm, index), "timing order mismatch"
        )
        close(row["model_seconds"], values["model_seconds"])
        require(
            positive(row["action_seconds"]) >= values["action_partial_seconds"],
            "action timing omits execution",
        )
        measured[arm][phase]["rows"] += 1
        for stage, field in (
            ("model", "model_seconds"),
            ("action", "action_seconds"),
            ("journal_partial", "journal_before_timing_append_seconds"),
        ):
            value = positive(row[field])
            measured[arm][phase][stage] += value
            measured[arm][phase][stage + "_max"] = max(measured[arm][phase][stage + "_max"], value)
        for key, value in (
            ("learning_pairs", 1),
            ("free_decisions", values["decisions"]),
            ("reader_offers", values["offers"]),
            ("action_labels", values["labels"]),
        ):
            counts[key] += value
        for key, value in values["publications"].items():
            publications[arm][phase][key] += value
        if phase == "regular":
            if arm in instances:
                require(instances[arm] == values["instance"], "learning switched model instance")
            instances[arm] = values["instance"]
            require(
                values["attempt_id"] == f"{values['instance']}:{index + 1}",
                "experience sequence changed",
            )
            chain[arm] = values["model_after"]
            curves[arm].append(values["grade"]["task_success"])
        seconds = time.perf_counter() - began
        totals = pairs[arm][phase]
        totals["rows"] += 1
        totals["seconds"] += seconds
        totals["max_seconds"] = max(totals["max_seconds"], seconds)

    if mode == "cost":
        task = cap_task()
        documents = [(b"A" * 16).hex()] * 7 + [(b"A" * 8).hex()]
        for arm in ARMS:
            pair(task, documents, "cap", arm, 0)
        comment = (b"#" + b"A" * 14 + b"\n").hex()
        definitions = (
            ("STOP", 0, "", "eos"),
            ("SPLICE", 0, comment, "eos"),
            ("SPLICE", 0, comment, "limit"),
            ("SPLICE", 1, "21", "eos"),
        )
        for index, (operation, end, payload, reason) in enumerate(definitions):
            row, digest = actions.next()
            check_action(
                row,
                task=None,
                files=task["files"],
                phase="category",
                arm="shared",
                index=index,
                launch=runtime,
                output=output,
            )
            require(
                row["proposal"]
                == {
                    **task["teacher"],
                    "operation": operation,
                    "end": end,
                    "replacement_hex": payload,
                    "stop_reason": reason,
                },
                "category proposal changed",
            )
            code = row["execution"]["returncode"]
            require(
                code == 0
                if index < 3
                else type(code) is int
                and code != 0
                and b"SyntaxError" in hex_bytes(row["execution"]["stderr_hex"]),
                "category outcome mismatch",
            )
            require(
                report["category_probes"][index]["action_line_sha256"] == digest,
                "category timing binding changed",
            )
            exact_keys(
                report["category_probes"][index], ("index", "action_seconds", "action_line_sha256")
            )
            require(
                report["category_probes"][index]["index"] == index
                and positive(report["category_probes"][index]["action_seconds"])
                >= positive(row["apply_execute_grade_seconds"]),
                "category guarding duration omits actual action",
            )
            if index < 3:
                require(
                    row["execution"]["stdout_hex"] == b"cap\n".hex()
                    and row["execution"]["stderr_hex"] == "",
                    "category expected output changed",
                )
            del row
    else:
        require(report["category_probes"] == [], "full run unexpectedly includes cost probes")
    selected = roster["cost_task_ids"] if mode == "cost" else roster["schedule"]
    require(report["selected_task_ids"] == selected, "acquisition schedule changed")
    for index, task_id in enumerate(selected):
        task = roster["tasks"][task_id]
        for arm in ARMS:
            pair(
                task,
                roster["memory_packages"][task["memory_id"]]["document_hex"],
                "regular",
                arm,
                index,
            )
    lessons.finish()
    timing.finish()
    require(
        chain == lineage["final_model_sha256"] and curves == report["acquisition_first_attempts"],
        "learning trajectory/final export mismatch",
    )
    require(
        (counts["learning_pairs"], counts["reader_offers"], counts["action_labels"])
        == ((25, 120, 65) if mode == "cost" else (1280, 2400, 3200)),
        "teaching exposure changed",
    )
    for arm in ARMS:
        for phase in ("regular", "cap"):
            summary, rows = report["timing_summary"][arm][phase], measured[arm][phase]
            require(summary["rows"] == rows["rows"], "timing summary count mismatch")
            for stage in ("model", "action"):
                close(summary[stage + "_seconds"], rows[stage])
                close(summary[stage + "_max_seconds"], rows[stage + "_max"])
            require(
                summary["journal_seconds"] >= rows["journal_partial"]
                and summary["journal_max_seconds"] >= rows["journal_partial_max"],
                "complete journal summary omits timing-row append",
            )
    endpoint_stream = Stream(output / "endpoint/worker/endpoints.jsonl", worker["endpoint_log"])
    scores = {arm: [] for arm in ARMS}
    endpoint_canonical = {arm: [] for arm in ARMS}
    endpoint_diagnostics = {arm: [] for arm in ARMS}
    worker_instances = {}
    endpoint_decisions = 0
    for arm in ARMS:
        for index, task_id in enumerate(task_ids):
            task = roster["tasks"][task_id]
            proposal, _ = endpoint_stream.next()
            exact_keys(
                proposal,
                (
                    "schema",
                    "type",
                    "arm",
                    "index",
                    "id",
                    "attempt_id",
                    "prediction",
                    "model_seconds",
                ),
            )
            require(
                (
                    proposal["schema"],
                    proposal["type"],
                    proposal["arm"],
                    proposal["index"],
                    proposal["id"],
                )
                == ("neuronal-splice-endpoint-ipc-v2", "proposal", arm, index, task_id),
                "worker endpoint order mismatch",
            )
            successor = documents_adjacency(
                roster["memory_packages"][task["memory_id"]]["document_hex"]
            )
            prediction = proposal["prediction"]
            require(
                proposal["attempt_id"] == prediction["attempt_id"],
                "outer endpoint token replaced pending identity",
            )
            parts = prediction["attempt_id"].split(":")
            require(
                len(parts) == 2 and all(part.isdigit() for part in parts),
                "invalid worker attempt format",
            )
            instance, ordinal = map(int, parts)
            require(
                instance > 0 and ordinal == (4 if mode == "cost" else 256) + index + 1,
                "worker experience order mismatch",
            )
            require(
                arm not in worker_instances or worker_instances[arm] == instance,
                "worker replaced per-arm learner",
            )
            worker_instances[arm] = instance
            check_prediction(
                prediction, successor, arm=arm, boundary_list=boundaries(task["files"])
            )
            query = hex_bytes(task["request_hex"], 31)
            decisions = len(prediction["payload"]["steps"])
            require(
                prediction["work"]["reader"]
                == reader_work(query, len(successor), decisions, teaching=False, arm=arm)
                and prediction["work"]["action"]
                == action_work(
                    query, task["files"], prediction["proposal"], teaching=False, arm=arm
                ),
                "endpoint work mismatch",
            )
            positive(proposal["model_seconds"])
            endpoint_decisions += decisions
            require(
                prediction["model_sha256"] == lineage["final_model_sha256"][arm]
                and prediction["memory_sha256"] == lineage["memory_sha256"][arm][task["memory_id"]],
                "endpoint state differs from learned export",
            )
            action, action_hash = actions.next()
            grade = check_action(
                action,
                task=task,
                files=task["files"],
                phase="endpoint",
                arm=arm,
                index=index,
                launch=runtime,
                output=output,
            )
            for key in ("attempt_id", "proposal", "proposal_sha256"):
                require(
                    action[key] == prediction[key], "endpoint execution replaced a neuronal choice"
                )
            scores[arm].append(grade["task_success"])
            endpoint_canonical[arm].append(grade["canonical_tuple_correct"])
            actual, teacher = prediction["proposal"], task["teacher"]
            editing = task["required_operation"] == "SPLICE"
            endpoint_diagnostics[arm].append(
                {
                    "task_id": task_id,
                    "required_operation": task["required_operation"],
                    "proposal": dict(actual),
                    "task_success": grade["task_success"],
                    "canonical_tuple_correct": grade["canonical_tuple_correct"],
                    "canonical_location_correct": all(
                        actual[key] == teacher[key] for key in ("path", "start", "end")
                    )
                    if editing
                    else None,
                    "canonical_payload_correct": actual["replacement_hex"]
                    == teacher["replacement_hex"]
                    if editing
                    else None,
                    "missing_payload_eos": actual["stop_reason"] != "eos",
                    "rejected": action["rejection"] is not None,
                    "returncode": action["execution"]["returncode"],
                    "process_error": action["execution"]["returncode"] != 0,
                    "stdout_sha256": action["execution"]["stdout_sha256"],
                    "stderr_sha256": action["execution"]["stderr_sha256"],
                    "action_line_sha256": action_hash,
                }
            )
            identity = {
                key: proposal[key] for key in ("schema", "arm", "index", "id", "attempt_id")
            }
            model_hash, memory_hash = prediction["model_sha256"], prediction["memory_sha256"]
            del proposal, prediction, action, actual, teacher
            acknowledgement, _ = endpoint_stream.next()
            exact_keys(
                acknowledgement,
                (
                    *identity,
                    "type",
                    "action_sha256",
                    "model_sha256",
                    "memory_sha256",
                    "work",
                    "model_seconds",
                ),
            )
            require(
                all(acknowledgement[key] == value for key, value in identity.items())
                and acknowledgement["type"] == "acknowledged"
                and acknowledgement["action_sha256"] == action_hash
                and acknowledgement["model_sha256"] == model_hash
                and acknowledgement["memory_sha256"] == memory_hash
                and acknowledgement["work"]
                == {
                    "action": {"custody_tensor_bytes_hashed": MODEL_BYTES + TAPE_BYTES},
                    "reader": {},
                },
                "worker acknowledgement changed state/action",
            )
            positive(acknowledgement["model_seconds"])
            del acknowledgement
    endpoint_stream.finish()
    actions.finish()
    require(scores == report["worker"]["task_success"], "endpoint scores disagree")
    require(
        endpoint_canonical == report["worker"]["canonical_tuple_correct"],
        "endpoint tuple diagnostics disagree",
    )
    require(
        actions.rows == (269 if mode == "cost" else 1520) and endpoint_stream.rows == 480,
        "actual action/endpoint exposure mismatch",
    )
    panels = {}
    for phase, path, metadata, hashes in (
        (
            "pre",
            output / "pre_panel.jsonl",
            report["logs"]["pre_panel"],
            lineage["initial_model_sha256"],
        ),
        (
            "post",
            output / "endpoint/worker/reading.jsonl",
            worker["reading_log"],
            lineage["final_model_sha256"],
        ),
    ):
        panel = Stream(path, metadata)
        panels[phase] = {}
        for arm in ARMS:
            panels[phase][arm] = []
            for index, case in enumerate(roster["old_panel"]):
                row, _ = panel.next()
                panels[phase][arm].append(
                    check_panel(
                        row,
                        case,
                        phase=phase,
                        arm=arm,
                        index=index,
                        documents=roster["memory_packages"][case["memory_id"]]["document_hex"],
                        model_hash=hashes[arm],
                        memory_hash=lineage["memory_sha256"][arm][case["memory_id"]],
                    )
                )
        panel.finish()
    fresh_edit_indices = [
        i
        for i, task_id in enumerate(task_ids[:32])
        if roster["tasks"][task_id]["required_operation"] == "SPLICE"
    ]
    learned_edits = sum(scores["learned"][i] for i in fresh_edit_indices)
    old_required = [i for i in range(20) if i not in (16, 18)]
    criterion_parts = {
        "all_32_fresh_tasks": all(scores["learned"][:32]),
        "all_16_retention_tasks": all(scores["learned"][32:]),
        "preserves_original_18_reader_successes": all(
            panels["post"]["learned"][i] for i in old_required
        ),
        "strict_fresh_edit_superiority": {
            arm: learned_edits > sum(scores[arm][i] for i in fresh_edit_indices) for arm in ARMS[1:]
        },
    }
    criterion = (
        criterion_parts["all_32_fresh_tasks"]
        and criterion_parts["all_16_retention_tasks"]
        and criterion_parts["preserves_original_18_reader_successes"]
        and all(criterion_parts["strict_fresh_edit_superiority"].values())
    )
    comparison = {}
    for arm in ARMS:
        comparison[arm] = {
            "fresh_correct": sum(scores[arm][:32]),
            "fresh_edits_correct": sum(scores[arm][i] for i in fresh_edit_indices),
            "fresh_stop_correct": sum(
                scores[arm][i] for i in range(32) if i not in fresh_edit_indices
            ),
            "retention_correct": sum(scores[arm][32:]),
            "factor_pairs": {
                group: [
                    {
                        "ids": task_ids[base : base + 2],
                        "successes": scores[arm][base : base + 2],
                        "both_correct": all(scores[arm][base : base + 2]),
                    }
                    for base in range(offset, offset + 8, 2)
                ]
                for group, offset in zip("DCVO", range(0, 32, 8), strict=True)
            },
            "acquisition_round_successes": [
                sum(curves[arm][start : start + 32]) for start in range(0, 256, 32)
            ]
            if mode == "run"
            else None,
        }
    result.update(
        status="passed",
        mode=mode,
        report_sha256=sha(output / "report.json"),
        completion_sha256=sha(output / "completion.json"),
        launch_sha256=sha(args.launch),
        admission_sha256=args.admission_sha,
        source_sha256=report["source_sha256"],
        learning_pair_timing=pairs,
        counts=counts,
        endpoint_scores=scores,
        reading_panels=panels,
        endpoint_diagnostics=endpoint_diagnostics,
        comparison=comparison,
        actual_changed_group_counts=publications,
        criterion_components=criterion_parts,
        scoped_criterion_passed=criterion if mode == "run" else False,
        lineage=lineage,
        seconds=time.perf_counter() - started,
        limits=[
            "No neuronal forward or action replay in saved audit.",
            "Saved normalized beliefs, losses and local-work receipts do not independently "
            "reconstruct unsaved per-lesson synapse updates; source and ordinary QA "
            "establish that arithmetic.",
            "Receipt/source custody is not a filesystem sandbox or formal timing bound.",
            "Narrow fragment-copy edit tasks do not establish general software ability.",
        ],
    )
    result["counts"].update(
        action_verifications=actions.rows,
        endpoint_proposals=240,
        endpoint_payload_decisions=endpoint_decisions,
        old_panel_calls=200,
    )
    inputs = [
        output / "report.json",
        output / "completion.json",
        args.launch,
        args.admission,
        roster_path,
        Path(__file__).resolve(),
        output / "endpoint/worker-launch.json",
        worker_path,
        request_path,
        output / "endpoint/source_free.pt",
    ]
    result["input_sha256"] = {str(Path(path).resolve()): sha(path) for path in inputs}
    result["roster_sha256"] = report["roster_sha256"]
    result["projection"] = project_cost(report, outer) if mode == "cost" else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha", required=True)
    args = parser.parse_args()
    if DRAFT:
        raise RuntimeError("draft checker cannot audit genuine artifacts")
    result = {"schema": "neuronal-splice-independent-audit-v2", "pid": os.getpid()}
    audit(args, result)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
