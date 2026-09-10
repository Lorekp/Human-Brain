"""One explicit repeated-teaching block over the retained causal ONLINE state.

The parent checkpoint remains the only safe state until final publication. Every
attempt/correction is journaled; scalar checkpoints are not resumable snapshots.
No model work occurs on import. Ordinary per-attempt checkpoint APIs are disabled.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

from experiments.r4_integrated.persistent_checkpoint_journal import OwnedJournal
from experiments.r4_integrated.persistent_language_action_v8_initialize import _add, _hash, _read
from experiments.r4_integrated.persistent_session_checkpoint import (
    JSON_LIMIT,
    NEURAL_ROUTES,
    ROUTES,
    DirtyCheckpoint,
    PersistentSessionCheckpoint,
    _actors,
    _baseline_record,
    _baseline_values,
    _keys,
    _relative,
    _restore_baseline,
    _sha,
    _tensors,
    encode,
)

ROOT = Path(__file__).resolve().parents[2]
BANK_SHA = "1bd3073fac22fdd54cc27e16ad7c8e75d106b15e9263e44bb1161b1a85c54919"
AUDITOR = "experiments/r4_integrated/stable_language_action_credit_review_v1.py"
AUDITOR_SHA = "2f53d1edea559fee230205d9e107fe4df54e9366d793a9781009d2d0db864c0c"
ACTOR_MODULE = "humanbrain.recurrent_content_learner"
ACTOR_NAME = "RecurrentContentSpliceLearner"
JOURNALS = ("attempts", "teachers", "probes")
# Two route payloads plus at most 16 MiB of boundary metadata and the small
# HEAD receipt stay within the runner's 256 MiB pending-write reserve.
ROUTE_LIMIT = 118 * 1024**2
PARENT_PAYLOAD_SHA = "6e9be59b28807a0790f06b8861596d0d0db1639e84714aef84939a1c27cdc5e1"
PARENT_MODEL_SHA = "f4f39f528da8e0d0968ec74998d6a74cb33adecc06ec43b5cfbd0f1466a8adbc"
PARENT_MEMORY_SHA = "cdd1f887f80ec2e8ae6ee38ae8a15786d9155160cf6beb18def635a2a66da73e"


def _canonical(actor_class):
    if (
        actor_class.__module__ != ACTOR_MODULE
        or actor_class.__qualname__ != ACTOR_NAME
        or getattr(sys.modules.get(ACTOR_MODULE), ACTOR_NAME, None) is not actor_class
    ):
        raise ValueError("requires the current canonical retained recurrent actor")


def _parent(directory, bank_path, bank_sha256, frozen_auditor_path, work):
    """Authenticate bank, complete metadata inventory, sources and fresh frontier."""
    from experiments.r4_integrated.stable_language_action_credit_checkpoint_v1 import (
        StableLanguageActionCreditCheckpoint,
    )

    if bank_sha256 != BANK_SHA:
        raise ValueError("unknown retained causal result bank")
    bank = json.loads(_read(Path(bank_path), BANK_SHA, work))
    worker = bank["workers"]["causal"]
    expected = worker["checkpoint"]
    directory = Path(directory).resolve()
    root = directory.parent
    if (
        not bank["passed"]
        or bank["status"] != "passed"
        or not worker["complete_run"]
        or worker["credit_rule"] != "causal"
        or expected["dirty"]
        or expected["outstanding_intents"]
        or expected["boundaries"] != 265
        or Path(worker["run"]).resolve() != root
    ):
        raise ValueError("retained causal bank is not its completed clean parent")
    evidence = worker["evidence"]
    metadata = {}
    for name in (
        "sources.json",
        "checkpoints/config.json",
        "checkpoints/HEAD.json",
        "checkpoints/boundary-000000264.json",
        "manifest.json",
    ):
        metadata[name] = json.loads(_read(_relative(root, name), evidence[name], work))
    boundary = metadata["checkpoints/boundary-000000264.json"]
    config = metadata["checkpoints/config.json"]
    if (
        metadata["checkpoints/HEAD.json"] != expected["head"]
        or boundary["routes"] != expected["last_routes"]
        or boundary["baseline"] != expected["last_baseline"]
        or boundary["host"] != expected["last_host"]
        or config["source_sha256"] != metadata["sources.json"]
        or config["source_revision"] != worker["source_revision"]
        or config["manifest_sha256"] != bank["manifest_sha256"]
    ):
        raise ValueError("parent bank, source map, HEAD or boundary disagree")
    wanted_files = {
        name.removeprefix("checkpoints/") for name in evidence if name.startswith("checkpoints/")
    }
    if {p.name for p in directory.iterdir()} != wanted_files:
        raise DirtyCheckpoint("parent checkpoint inventory changed or attempted work remains")
    for name in sorted(wanted_files):
        path = _relative(directory, name)
        if not path.is_file():
            raise ValueError("parent checkpoint inventory contains a non-file")
        if name.endswith(".json"):
            _read(path, evidence["checkpoints/" + name], work)
    frozen = Path(frozen_auditor_path).resolve()
    _read(frozen, AUDITOR_SHA, work)

    class ParentInspector(StableLanguageActionCreditCheckpoint):
        def _file(self, path, *, prefix=None):
            if path.resolve() == (ROOT / AUDITOR).resolve():
                if self.config["source_sha256"].get(AUDITOR) != AUDITOR_SHA:
                    raise ValueError("unknown historical auditor exception")
                path = frozen
            return super()._file(path, prefix=prefix)

    with ParentInspector.open(
        directory, config_sha256=evidence["checkpoints/config.json"]
    ) as store:
        checked = store.inspect(require_clean=True, fresh_payloads=True)
        if checked["boundary_sha256"] != expected["head"]["sha256"]:
            raise ValueError("fresh parent inspector selected a different boundary")
        for name, count in store.work.items():
            _add(work, "parent_" + name, count)
    selected = boundary["routes"]["online"]
    if (
        selected["sha256"] != PARENT_PAYLOAD_SHA
        or selected["bytes"] != 115712640
        or selected["model_sha256"] != PARENT_MODEL_SHA
        or selected["memory_sha256"] != PARENT_MEMORY_SHA
        or selected["experience"] != 369
        or selected["reader_experience"] != 342
        or selected["actor_class"] != ACTOR_MODULE + "." + ACTOR_NAME
    ):
        raise ValueError("different retained causal ONLINE actor")
    return boundary, config


def _copy_actor(payload, row, actor_class, work):
    """Strict same-schema restore into fresh independent storage, without decode."""
    import torch

    from humanbrain.neuronal_splice_learner import _digest

    _canonical(actor_class)
    _keys(payload, ("schema", "model", "memory"))
    if payload["schema"] != "persistent-session-route-tensors-v1":
        raise ValueError("unknown retained tensor schema")
    model = actor_class(mode=row["mode"], seed=row["seed"])
    _add(work, "model_constructions")
    expected = model.state_dict()
    _keys(payload["model"], expected)
    if expected["_extra_state"]["schema"] != "recurrent-content-splice-learner-v2":
        raise ValueError("repeated teaching requires unchanged causal v2 architecture")
    for name, prototype in expected.items():
        incoming = payload["model"][name]
        if isinstance(prototype, torch.Tensor):
            if (
                not isinstance(incoming, torch.Tensor)
                or incoming.shape != prototype.shape
                or incoming.dtype != prototype.dtype
                or incoming.device != prototype.device
                or incoming.requires_grad
                or incoming.grad_fn is not None
                or incoming.grad is not None
                or not bool(torch.isfinite(incoming).all())
            ):
                raise ValueError("retained tensor geometry or finite-state contract differs")
        elif incoming != prototype:
            raise ValueError("same-v2 initializer cannot adapt metadata")
    if (
        _digest(payload["model"], work) != row["model_sha256"]
        or _digest(payload["memory"], work) != row["memory_sha256"]
    ):
        raise ValueError("retained numerical state disagrees with authenticated receipt")
    model.load_state_dict(payload["model"], strict=True, assign=False, work=work)
    state = model.reader.memory.restore_state(payload["memory"])
    model._experience, model.reader._active_experience = row["experience"], row["reader_experience"]
    model._require_idle()
    model._check(work)
    if (
        _digest(model.state_dict(), work) != row["model_sha256"]
        or _digest(vars(state), work) != row["memory_sha256"]
    ):
        raise ValueError("strict restoration changed retained state")
    incoming_storage = {t.untyped_storage().data_ptr() for t in _tensors(payload) if t.numel()}
    values = list(_tensors(model.state_dict())) + list(_tensors(vars(state)))
    for view in model.reader._candidate_views:
        values.extend((view.indices, view.values, view.flags))
    for wiring in model.reader._self_wirings:
        values.extend((wiring.indices, wiring.values))
    if incoming_storage & {t.untyped_storage().data_ptr() for t in values if t.numel()}:
        raise ValueError("copy retained decoded or parent storage")
    return model, state


def initialize_retained(
    prior_directory, bank_path, *, bank_sha256, actor_class, frozen_auditor_path
):
    """One authenticated causal ONLINE decode, copied independently into both arms."""
    import torch

    started, work = time.perf_counter(), {}
    boundary, config = _parent(prior_directory, bank_path, bank_sha256, frozen_auditor_path, work)
    row = boundary["routes"]["online"]
    raw = _read(
        _relative(Path(prior_directory).resolve(), row["payload"]),
        row["sha256"],
        work,
        limit=ROUTE_LIMIT,
    )
    decoded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _add(work, "tensor_decodes")
    del raw
    models, states = {}, {}
    for route in NEURAL_ROUTES:
        models[route], states[route] = _copy_actor(decoded, row, actor_class, work)
    _actors(models, work)
    tape_storages = [
        {t.untyped_storage().data_ptr() for t in _tensors(vars(state)) if t.numel()}
        for state in states.values()
    ]
    if tape_storages[0] & tape_storages[1]:
        raise ValueError("matched initialization shares factual storage")
    baseline_record = copy.deepcopy(boundary["baseline"])
    baseline = _restore_baseline(baseline_record, document_limit=3484, symbol_limit=21382)
    if _baseline_record(baseline) != baseline_record:
        raise ValueError("retained simple history changed")
    final, final_config = _parent(
        prior_directory, bank_path, bank_sha256, frozen_auditor_path, work
    )
    if final != boundary or final_config != config:
        raise ValueError("parent changed during initialization")
    _hash(
        _relative(Path(prior_directory).resolve(), row["payload"]),
        row["sha256"],
        row["bytes"],
        work,
    )
    receipt = {
        "schema": "repeated-teaching-initialization-v1",
        "bank_sha256": bank_sha256,
        "prior_directory": str(Path(prior_directory).resolve()),
        "parent_boundary_sha256": hashlib.sha256(encode(boundary)).hexdigest(),
        "parent_routes": copy.deepcopy(boundary["routes"]),
        "initial_routes": {route: dict(row) for route in NEURAL_ROUTES},
        "baseline_record": baseline_record,
        "work": work,
        "elapsed_seconds": time.perf_counter() - started,
        "metadata_adaptation": False,
        "source_offers": 0,
        "teacher_offers": 0,
    }
    return {
        "models": models,
        "states": states,
        "baseline": baseline,
        "baseline_record": baseline_record,
        "prior_host": copy.deepcopy(boundary["host"]),
        "prior_frontiers": copy.deepcopy(boundary["frontiers"]),
        "receipt": receipt,
    }


def clone_for_probe(model, state):
    """Copy a live idle actor into a disposable probe actor; no file or decoder."""
    from humanbrain.neuronal_splice_learner import _digest

    work = {}
    model._require_idle()
    before = (
        _digest(model.state_dict(), work),
        _digest(vars(state), work),
        model._experience,
        model.reader._active_experience,
    )
    snapshot = model.snapshot(state)
    payload = {
        "schema": "persistent-session-route-tensors-v1",
        "model": snapshot["model"],
        "memory": snapshot["memory"],
    }
    row = dict(
        model_sha256=before[0],
        memory_sha256=before[1],
        experience=before[2],
        reader_experience=before[3],
        mode=model.mode,
        seed=model.seed,
    )
    copied, copied_state = _copy_actor(payload, row, type(model), work)
    if before != (
        _digest(model.state_dict(), work),
        _digest(vars(state), work),
        model._experience,
        model.reader._active_experience,
    ):
        raise ValueError("probe cloning changed its live parent")
    return {"model": copied, "state": copied_state, "work": work, "tensor_decodes": 0}


class RepeatedTeachingCheckpoint(PersistentSessionCheckpoint):
    """Experiment-only full-block intent and final snapshot; no in-block resume."""

    @classmethod
    def create(
        cls,
        directory,
        *,
        evidence_root,
        manifest_path,
        manifest_sha256,
        source_sha256,
        source_revision,
        parent_receipt,
    ):
        directory = Path(directory).resolve()
        directory.mkdir(parents=False, exist_ok=False)
        obj = cls.__new__(cls)
        obj.directory, obj.work = directory, {}
        obj._intent, obj._payload_backend = None, None
        obj._payload_leases, obj._journals = {}, {}
        obj._closed, obj._journal_poisoned, obj._journal_busy = False, False, False
        obj._journal_thread = None
        raw = Path(manifest_path).read_bytes()
        if len(raw) > JSON_LIMIT or hashlib.sha256(raw).hexdigest() != _sha(manifest_sha256):
            raise ValueError("block manifest identity or bound differs")
        obj.config = {
            "schema": "repeated-teaching-checkpoint-config-v1",
            "source_root": str(ROOT),
            "evidence_root": str(Path(evidence_root).resolve()),
            "source_revision": source_revision,
            "source_sha256": dict(source_sha256),
            "manifest_sha256": manifest_sha256,
            "parent": copy.deepcopy(parent_receipt),
        }
        config_raw = encode(obj.config)
        if len(config_raw) > JSON_LIMIT:
            raise ValueError("block configuration exceeds metadata bound")
        obj.config_sha256 = hashlib.sha256(config_raw).hexdigest()
        obj._write(directory / "config.json", config_raw)
        obj._write(directory / "manifest.json", raw)
        obj._binding()
        intent = {
            "schema": "repeated-teaching-block-intent-v1",
            "sequence": 0,
            "kind": "whole_teaching_block",
            "config_sha256": obj.config_sha256,
            "manifest_sha256": manifest_sha256,
            "parent": parent_receipt["parent_boundary_sha256"],
            "resume_policy": "never replay; incomplete block retains parent and observed prefix",
        }
        intent_raw = encode(intent)
        obj._write(directory / "intent-000000000.json", intent_raw)
        obj._intent = intent, hashlib.sha256(intent_raw).hexdigest()
        return obj

    def _binding(self):
        raw = self._read(self.directory / "config.json", JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != self.config_sha256:
            raise ValueError("block configuration changed")
        if self.config["schema"] != "repeated-teaching-checkpoint-config-v1":
            raise ValueError("not a repeated teaching block")
        if Path(self.config["source_root"]).resolve() != ROOT or not self.config["source_sha256"]:
            raise ValueError("block source binding missing")
        for name, identity in self.config["source_sha256"].items():
            if self._file(_relative(ROOT, name))["sha256"] != _sha(identity):
                raise ValueError("current block source changed")
        raw = self._read(self.directory / "manifest.json", JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != self.config["manifest_sha256"]:
            raise ValueError("block manifest changed")
        self.manifest = json.loads(raw)
        schedule = self.manifest["schedule"]
        names = ["A00-R"] * 8 + ["A00-E"] * 8 + ["A00-R", "A00-E"] * 24
        if len(schedule) != 64 or any(
            row != {"index": i, "episode_id": names[i]} for i, row in enumerate(schedule)
        ):
            raise ValueError("block must declare exactly32 unchanged reading/edit rounds")
        if self.config["parent"]["bank_sha256"] != BANK_SHA:
            raise ValueError("block parent bank changed")

    def create_journals(self):
        return self.create_owned_journals()

    def _baseline_values(self, record):
        return _baseline_values(record, document_limit=3484, symbol_limit=21382)

    def _restore_baseline(self, record):
        return _restore_baseline(record, document_limit=3484, symbol_limit=21382)

    def create_owned_journals(self, journals=None):
        self._require_open()
        if journals is not None or self._journals or self._intent is None:
            raise ValueError("block owns exactly its three new journals")
        root = Path(self.config["evidence_root"])
        try:
            for kind in JOURNALS:
                name = kind + ".jsonl"
                self._journals[name] = OwnedJournal(_relative(root, name), self.work)
            return {kind: self._journals[kind + ".jsonl"] for kind in JOURNALS}
        except BaseException:
            self._journal_poisoned = True
            self.close()
            raise

    def begin(self, *args, **kwargs):
        raise ValueError("block intent already spends all64 steps; no per-attempt begin")

    def commit(self, *args, **kwargs):
        raise ValueError("use explicit commit_final after the entire declared block")

    def capture_frontiers(self):
        self._require_journal_scope()
        rows = []
        for kind in JOURNALS:
            item = self._journal_file(kind + ".jsonl")
            if not item.pop("complete"):
                raise ValueError("incomplete block journal")
            rows.append({"kind": kind, "path": kind + ".jsonl", **item})
        root = Path(self.config["evidence_root"])
        execution = _relative(root, "executions")
        files, directories = [], []
        for path in sorted(execution.rglob("*")):
            relative = path.relative_to(root).as_posix()
            path = _relative(root, relative)
            if path.is_file():
                item = self._file(path)
                files.append({"path": relative, "bytes": item["bytes"], "sha256": item["sha256"]})
            elif path.is_dir():
                directories.append(relative)
            else:
                raise ValueError("unsupported execution inventory entry")
        return {
            "journals": rows,
            "execution_root": "executions",
            "execution_files": files,
            "execution_directories": directories,
        }

    def _frontiers(self, value, *, allow_tail=False, fresh=False):
        # Retain the base execution/file checks; only this experiment's journal
        # names differ. Do not map probes into fictitious ingest observations.
        _keys(value, ("journals", "execution_root", "execution_files", "execution_directories"))
        root = Path(self.config["evidence_root"])
        if [r["kind"] for r in value["journals"]] != list(JOURNALS):
            raise ValueError("block needs all three ordered journal frontiers")
        dirty = []
        for row in value["journals"]:
            if row["path"] != row["kind"] + ".jsonl":
                raise ValueError("unexpected block journal path")
            item = self._journal_file(row["path"], prefix=row["bytes"], fresh=fresh)
            if not item["complete"] or any(item[k] != row[k] for k in ("bytes", "rows", "sha256")):
                raise ValueError("block journal prefix changed")
            if _relative(root, row["path"]).stat().st_size != row["bytes"]:
                dirty.append("uncommitted journal tail: " + row["path"])
        execution = _relative(root, value["execution_root"])
        expected = set()
        for row in value["execution_files"]:
            path = _relative(root, row["path"])
            if not path.is_relative_to(execution) or row["path"] in expected:
                raise ValueError("invalid block execution path")
            expected.add(row["path"])
            item = self._file(path)
            if any(item[k] != row[k] for k in ("bytes", "sha256")):
                raise ValueError("block execution bytes changed")
        actual, directories = set(), []
        for path in execution.rglob("*"):
            name = path.relative_to(root).as_posix()
            checked = _relative(root, name)
            if checked.is_file():
                actual.add(name)
            elif checked.is_dir():
                directories.append(name)
        if actual != expected or sorted(directories) != value["execution_directories"]:
            dirty.append("block execution inventory differs")
        if dirty and not allow_tail:
            raise DirtyCheckpoint("; ".join(dirty))
        return dirty

    def _boundary(self, *, fresh_payloads=False):
        self._binding()
        head = json.loads(self._read(self.directory / "HEAD.json", JSON_LIMIT))
        if head["schema"] != "repeated-teaching-head-v1" or head["sequence"] != 0:
            raise ValueError("unknown block HEAD")
        if head["manifest"] != "boundary-000000000.json":
            raise ValueError("unexpected final block boundary name")
        raw = self._read(self.directory / head["manifest"], JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != _sha(head["sha256"]):
            raise ValueError("block HEAD/boundary differs")
        boundary = json.loads(raw)
        if (
            boundary["schema"] != "repeated-teaching-boundary-v1"
            or boundary["config_sha256"] != self.config_sha256
            or boundary["parent"] != self.config["parent"]["parent_boundary_sha256"]
        ):
            raise ValueError("block boundary parent/config differs")
        raw = self._read(self.directory / "intent-000000000.json", JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != boundary["intent_sha256"]:
            raise ValueError("block intent changed")
        _keys(boundary["routes"], NEURAL_ROUTES)
        for row in boundary["routes"].values():
            self._payload_file(row, fresh=fresh_payloads)
        self._baseline_values(boundary["baseline"])
        return boundary, head["sha256"]

    def inspect(self, *, require_clean=True, fresh_payloads=False):
        self._require_open()
        if not (self.directory / "HEAD.json").exists():
            raise DirtyCheckpoint(
                "block has no final snapshot; parent remains latest safe state, never replay"
            )
        boundary, digest = self._boundary(fresh_payloads=fresh_payloads)
        dirty = self._frontiers(boundary["frontiers"], allow_tail=True, fresh=fresh_payloads)
        expected = {
            "config.json",
            "manifest.json",
            "intent-000000000.json",
            "HEAD.json",
            "boundary-000000000.json",
            *(r["payload"] for r in boundary["routes"].values()),
        }
        if {p.name for p in self.directory.iterdir()} != expected:
            dirty.append("unexplained checkpoint directory entries")
        if require_clean and dirty:
            raise DirtyCheckpoint("; ".join(dirty))
        return {
            "status": "dirty" if dirty else "clean",
            "reasons": dirty,
            "boundary": boundary,
            "boundary_sha256": digest,
        }

    def _validate_block(self, models, states, baseline, host, frontiers):
        """Authenticate every complete main chain and the declared probe panels."""
        from experiments.r4_integrated.persistent_language_action_v7_review import journal, read_row
        from experiments.r4_integrated.persistent_language_action_v8 import _plain
        from humanbrain.neuronal_splice_learner import _digest

        root = Path(self.config["evidence_root"])
        indexed = {}
        for item in frontiers["journals"]:
            entries, actual = journal(_relative(root, item["path"]), partial=False)
            self._add("block_ledger_index_bytes", actual["bytes"])
            if any(actual[k] != item[k] for k in ("bytes", "rows", "sha256")):
                raise ValueError("block ledger changed after frontier capture")
            indexed[item["kind"]] = entries

        def read(entry):
            value = read_row(entry)
            self._add("block_ledger_parse_bytes", entry["bytes"])
            return value

        initial = host["initial_hashes"]
        _keys(initial, ROUTES)
        for route in NEURAL_ROUTES:
            if initial[route] != {"model": PARENT_MODEL_SHA, "memory": PARENT_MEMORY_SHA}:
                raise ValueError("block did not start from paired causal ONLINE state")
        parent = self._restore_baseline(self.config["parent"]["baseline_record"])
        if initial["simple"] != {
            "model": _digest(_plain(parent.demonstrations), self.work),
            "memory": _digest(_plain(parent.documents), self.work),
        }:
            raise ValueError("block reset the reference's retained history")
        current = copy.deepcopy(initial)
        panels = {0: copy.deepcopy(current)}
        results, executions = [], 0
        for offset, item in enumerate(self.manifest["schedule"]):
            for ri, route in enumerate(ROUTES):
                index = offset * 3 + ri
                row, taught = read(indexed["attempts"][index]), read(indexed["teachers"][index])
                coordinate = (offset, route, item["episode_id"])
                if any(
                    (r["step_index"], r["route"], r["episode_id"]) != coordinate
                    or r["scope"] != "main"
                    for r in (row, taught)
                ):
                    raise ValueError("main row differs from block order")
                episode = self.manifest["tasks"][item["episode_id"]]
                expected_observation = {
                    "schema": "sequence-assembly-raw-observation-v1",
                    "request_hex": episode["request_hex"],
                    "files": episode["files"] if episode["kind"] == "edit" else None,
                    "feedback": None,
                }
                if row["observation"] != expected_observation:
                    raise ValueError("main response changed the fixed complete public input")
                before = {
                    "model": row["model_before_sha256"],
                    "memory": row["memory_before_sha256"],
                }
                after = {"model": row["model_after_sha256"], "memory": row["memory_after_sha256"]}
                if (
                    before != current[route]
                    or before["memory"] != after["memory"]
                    or taught["model_before_sha256"] != before["model"]
                    or taught["model_after_sha256"] != after["model"]
                    or taught["memory_before_sha256"] != before["memory"]
                    or taught["memory_after_sha256"] != after["memory"]
                    or taught["teacher"] != episode["teacher"]
                    or taught["learning_enabled"] is not (route == "online")
                    or row["kind"] != episode["kind"]
                ):
                    raise ValueError("attempt/correction state chain or target differs")
                neural, edit = route in NEURAL_ROUTES, episode["kind"] == "edit"
                if taught["reader_offers"] != ((23 if edit else 12) if neural else 0) or taught[
                    "action_labels"
                ] != (3 if neural and edit else 0):
                    raise ValueError("declared complete answer/motor teacher count differs")
                if neural:
                    target = episode["teacher"]["replacement_hex" if edit else "target_hex"]
                    if [step["observed_token"] for step in taught["teacher_steps"]] != [
                        *bytes.fromhex(target),
                        256,
                    ]:
                        raise ValueError(
                            "teacher journal omits or changes a complete target symbol"
                        )
                if edit:
                    execution = row["execution"]
                    if not execution["files_complete"] or not execution["capture_complete"]:
                        raise ValueError("incomplete execution cannot form final block state")
                    executions += 1
                elif row["execution"] is not None:
                    raise ValueError("reading unexpectedly performed an edit execution")
                if route == "simple":
                    # Validate raw reference history without invoking a teacher
                    # or policy a second time during checkpoint/audit work.
                    teacher = episode["teacher"]
                    if edit:
                        source = bytes.fromhex(
                            next(
                                value["content_hex"]
                                for value in episode["files"]
                                if value["path"] == teacher["path"]
                            )
                        )
                        query = bytes.fromhex(episode["request_hex"])
                        parent.demonstrations.append(
                            {
                                "query": query,
                                "old": source[teacher["start"] : teacher["end"]],
                                "replacement": bytes.fromhex(teacher["replacement_hex"]),
                                "source_sha256": hashlib.sha256(source).hexdigest(),
                                "query_sha256": hashlib.sha256(query).hexdigest(),
                            }
                        )
                        parent.new_teacher_offers["SPLICE"] += 1
                    else:
                        parent.new_teacher_offers["reading"] += 1
                current[route] = after
                results.append(
                    {k: row[k] for k in ("step_index", "route", "episode_id", "correct")}
                )
            if offset + 1 in self.manifest["checkpoints"]:
                panels[offset + 1] = copy.deepcopy(current)
        if host["schedule_completed"] != 64 or host["results"] != results:
            raise ValueError("final host omits/reorders actual main attempts")
        if _baseline_record(parent) != _baseline_record(baseline):
            raise ValueError("reference final history differs from its actual supplied corrections")
        if current["simple"] != {
            "model": _digest(_plain(baseline.demonstrations), self.work),
            "memory": _digest(_plain(baseline.documents), self.work),
        }:
            raise ValueError("reference final state differs from durable chain")
        for route, model in models.items():
            if current[route] != {
                "model": _digest(model.state_dict(), self.work),
                "memory": _digest(vars(states[route]), self.work),
            }:
                raise ValueError("final actor differs from complete durable state chain")
        if len(indexed["probes"]) != len(self.manifest["checkpoints"]):
            raise ValueError("missing diagnostic panel")
        for entry, cp in zip(indexed["probes"], self.manifest["checkpoints"], strict=True):
            panel = read(entry)
            if (
                panel["checkpoint"] != cp
                or panel["live_before"] != panels[cp]
                or panel["live_after"] != panels[cp]
                or len(panel["rows"]) != 12
            ):
                raise ValueError("diagnostic panel changed live state or roster")
            for record, coordinate in zip(
                panel["rows"],
                ((r, case) for r in ROUTES for case in self.manifest["checkpoint_cases"]),
                strict=True,
            ):
                row, taught = record["attempt"], record["teacher"]
                if (row["route"], row["episode_id"]) != coordinate or row["scope"] != "probe":
                    raise ValueError("probe panel case identity differs")
                if row["execution"] is not None:
                    executions += 1
                if coordinate[1].startswith("A00"):
                    if taught is None or taught["learning_enabled"] is not False:
                        raise ValueError("probe teacher must remain disabled")
                elif taught is not None:
                    raise ValueError("probe added an undeclared correction")
        counts = host["counts"]
        if (counts["completed_attempts"], counts["completed_teachers"], counts["executions"]) != (
            252,
            222,
            executions,
        ):
            raise ValueError("actual block+probe exposure counts disagree")
        if executions != 111:
            raise ValueError("declared main and probe execution count differs")

    def commit_final(self, models, states, baseline, host, *, validate_block=None):
        """Publish only after the caller's complete task ledger validation passes.

        An optional extra validator receives (manifest, parent_receipt, models,
        states, baseline, host, frontiers), after the required built-in validator.
        No checkpoint means no automatic block restart, even after partial failure.
        """
        import torch

        from humanbrain.neuronal_splice_learner import _digest

        started = time.perf_counter()
        if (
            (validate_block is not None and not callable(validate_block))
            or self._intent is None
            or (self.directory / "HEAD.json").exists()
        ):
            raise ValueError("only the owning complete block can publish once")
        with self.checkpoint_scope(), self._journal_operation():
            self._binding()
            _actors(models, self.work)
            _keys(states, NEURAL_ROUTES)
            frontiers = self.capture_frontiers()
            if [r["rows"] for r in frontiers["journals"][:2]] != [192, 192]:
                raise ValueError("final block needs192 actual attempts and192 bound corrections")
            self._validate_block(models, states, baseline, host, frontiers)
            if validate_block is not None:
                validate_block(
                    self.manifest, self.config["parent"], models, states, baseline, host, frontiers
                )
            seal = {
                route: (
                    _digest(model.state_dict(), self.work),
                    _digest(vars(states[route]), self.work),
                    model._experience,
                    model.reader._active_experience,
                )
                for route, model in models.items()
            }
            routes = {}
            for route, model in models.items():
                values = seal[route]
                if values[1] != PARENT_MEMORY_SHA or values[2:] != (401, 406):
                    raise ValueError(
                        "block must preserve tape and count32 edits/64 responses per neural route"
                    )
                snapshot = model.snapshot(states[route])
                payload = {
                    "schema": "persistent-session-route-tensors-v1",
                    "model": snapshot["model"],
                    "memory": snapshot["memory"],
                }
                with io.BytesIO() as buffer:
                    torch.save(payload, buffer)
                    raw = buffer.getvalue()
                if len(raw) > ROUTE_LIMIT:
                    raise ValueError("final route exceeds118MiB pending-write envelope")
                name = f"route-{route}-000000000.pt"
                temporary = self.directory / (name + ".tmp")
                self._write(temporary, raw)
                item = self._file(temporary)
                if item["bytes"] != len(raw) or item["sha256"] != hashlib.sha256(raw).hexdigest():
                    raise ValueError("final payload readback mismatch")
                temporary.rename(self.directory / name)
                self._own_payload(name, item)
                self._add("checkpoint_payload_writes", 1)
                routes[route] = {
                    "payload": name,
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "model_sha256": values[0],
                    "memory_sha256": values[1],
                    "experience": values[2],
                    "reader_experience": values[3],
                    "mode": model.mode,
                    "seed": model.seed,
                    "actor_class": f"{type(model).__module__}.{type(model).__qualname__}",
                }
                del snapshot, payload, raw
            baseline_record = _baseline_record(baseline)
            self._baseline_values(baseline_record)
            boundary = {
                "schema": "repeated-teaching-boundary-v1",
                "sequence": 0,
                "config_sha256": self.config_sha256,
                "parent": self.config["parent"]["parent_boundary_sha256"],
                "intent_sha256": self._intent[1],
                "routes": routes,
                "baseline": baseline_record,
                "host": copy.deepcopy(host),
                "frontiers": frontiers,
                "work": dict(self.work),
            }
            self._binding()
            self._frontiers(frontiers)
            _actors(models, self.work)
            if (
                seal
                != {
                    route: (
                        _digest(model.state_dict(), self.work),
                        _digest(vars(states[route]), self.work),
                        model._experience,
                        model.reader._active_experience,
                    )
                    for route, model in models.items()
                }
                or _baseline_record(baseline) != baseline_record
            ):
                raise ValueError("live state changed during final snapshot")
            for row in routes.values():
                self._payload_file(row)
            prepared = [
                (self._journals[row["path"]], self._journals[row["path"]].prepare_commit(row))
                for row in frontiers["journals"]
            ]
            raw = encode(boundary)
            if len(raw) > JSON_LIMIT:
                raise ValueError("final boundary exceeds metadata bound")
            self._write(self.directory / "boundary-000000000.json", raw)
            head = {
                "schema": "repeated-teaching-head-v1",
                "sequence": 0,
                "manifest": "boundary-000000000.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            temporary = self.directory / f"HEAD-{uuid.uuid4().hex}.tmp"
            self._write(temporary, encode(head))
            os.replace(temporary, self.directory / "HEAD.json")
            for journal, frontier in prepared:
                journal.commit_validated(frontier)
            self._intent = None
            self._add("checkpoint_head_publications", 1)
            return {
                "boundary_sha256": head["sha256"],
                "sequence": 0,
                "config_sha256": self.config_sha256,
                "elapsed_seconds": time.perf_counter() - started,
                "work": dict(self.work),
            }


def audit_saved(directory, *, config_sha256, actor_class):
    """Fresh final payload/journal reads plus strict restore; never perform tasks."""
    from humanbrain.neuronal_splice_learner import _digest

    _canonical(actor_class)
    started, models = time.perf_counter(), {}
    with RepeatedTeachingCheckpoint.open(directory, config_sha256=config_sha256) as store:
        checked = store.inspect(require_clean=True, fresh_payloads=True)
        for route, row in checked["boundary"]["routes"].items():
            models[route] = actor_class(mode=row["mode"], seed=row["seed"])
        restored = store.restore_into(models)
        if restored["boundary_sha256"] != checked["boundary_sha256"]:
            raise ValueError("final snapshot changed during fresh restore")
        store._validate_block(
            models,
            restored["states"],
            restored["baseline"],
            restored["host"],
            restored["frontiers"],
        )
        parity = {
            route: {
                "model": _digest(model.state_dict(), store.work),
                "memory": _digest(vars(restored["states"][route]), store.work),
                "experience": model._experience,
                "reader_experience": model.reader._active_experience,
            }
            for route, model in models.items()
        }
        return {
            "status": "passed",
            "boundary_sha256": restored["boundary_sha256"],
            "restore_parity": parity,
            "work": dict(store.work),
            "model_constructions": 2,
            "elapsed_seconds": time.perf_counter() - started,
            "task_executions": 0,
            "teacher_replays": 0,
            "scope": "fresh complete files and strict independent final restoration",
        }
