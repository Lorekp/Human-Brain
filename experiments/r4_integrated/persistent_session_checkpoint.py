"""Idle parent-session checkpoints, with durable intent and one atomic HEAD.

No model is constructed here. Restore accepts two fresh caller-owned actors and
loads only authenticated local tensor bytes. Dirty frontiers never replay work.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import uuid
from pathlib import Path, PurePosixPath

from experiments.r4_integrated.persistent_language_action_v1 import encode

ROUTES = ("online", "learning_disabled", "simple")
NEURAL_ROUTES = ROUTES[:2]
KINDS = ("attempts", "teachers", "ingests")
JSON_LIMIT = 16 * 1024 * 1024
PAYLOAD_LIMIT = 512 * 1024 * 1024
COUNTERS = (
    "attempts", "executions", "teacher_occasions", "reader_offers", "action_labels",
    "ingest_documents", "ingest_raw_bytes", "ingest_eos",
)


class DirtyCheckpoint(ValueError):
    """A safe prior exists, but further attempted work forbids automatic resume."""


def _keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("checkpoint record fields mismatch")


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("checkpoint counter must be a nonnegative integer")
    return value


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("checkpoint requires a lowercase SHA256")
    return value


def _raw(value, limit):
    if not isinstance(value, str) or len(value) > 2 * limit:
        raise ValueError("checkpoint byte field exceeds its bound")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("invalid checkpoint hexadecimal bytes") from error
    if raw.hex() != value:
        raise ValueError("checkpoint bytes must use canonical hexadecimal")
    return raw


def _relative(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("checkpoint path must be canonical relative POSIX")
    path = root.joinpath(*name.split("/"))
    if not path.resolve().is_relative_to(root):
        raise ValueError("checkpoint path escaped its bound root")
    if any(item.is_symlink() for item in (path, *path.parents) if item != root.parent):
        raise ValueError("checkpoint paths cannot traverse symlinks")
    return path


def _files(value):
    # Retain complete executor snapshots, including unsupported observation sizes.
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("checkpoint needs a complete bounded executor file snapshot")
    names = []
    for item in value:
        _keys(item, ("path", "content_hex"))
        name = item["path"]
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("invalid actual file snapshot path")
        _raw(item["content_hex"], 4096)
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("actual file snapshot must be ordered and unique")


def _feedback(value):
    if value is None:
        return
    _keys(value, ("returncode", "stdout_hex", "stderr_hex", "accepted", "timeout",
                  "capture_complete", "files_complete", "output_limit_exceeded"))
    code = value["returncode"]
    if code is not None and (type(code) is not int or not -(1 << 63) <= code < (1 << 63)):
        raise ValueError("invalid actual return code")
    _raw(value["stdout_hex"], 512)
    _raw(value["stderr_hex"], 512)
    for name in ("accepted", "timeout", "capture_complete", "files_complete",
                 "output_limit_exceeded"):
        if type(value[name]) is not bool:
            raise ValueError("actual feedback flags must be booleans")
    if not value["files_complete"]:
        raise ValueError("cannot checkpoint a resumable incomplete actual workspace")


def _workspace(value, *, current=False):
    _keys(value, ("route", "episode_id", "files", "feedback") if current
          else ("episode_id", "files", "feedback"))
    if current and value["route"] not in ROUTES:
        raise ValueError("invalid current workspace route")
    if not isinstance(value["episode_id"], str) or not value["episode_id"]:
        raise ValueError("workspace requires its actual predecessor episode")
    _files(value["files"])
    _feedback(value["feedback"])


def _baseline_record(baseline):
    from experiments.r4_integrated.persistent_language_action_baseline_v2 import SimpleReference

    if not isinstance(baseline, SimpleReference):
        raise ValueError("checkpoint requires the actual simple reference")
    return {
        "documents_hex": [doc.hex() for doc in baseline.documents],
        "demonstrations": [
            {name: value.hex() if isinstance(value, bytes) else value
             for name, value in demo.items()} for demo in baseline.demonstrations
        ],
        "new_teacher_offers": dict(baseline.new_teacher_offers),
    }


def _baseline_values(record):
    _keys(record, ("documents_hex", "demonstrations", "new_teacher_offers"))
    if (not isinstance(record["documents_hex"], list)
            or not isinstance(record["demonstrations"], list)):
        raise ValueError("reference history must be chronological lists")
    documents = [_raw(item, 2048) for item in record["documents_hex"]]
    if sum(len(doc) + 1 for doc in documents) > 2048 or len(record["demonstrations"]) > 4096:
        raise ValueError("reference history exceeds the declared session capacity")
    demos = []
    for demo in record["demonstrations"]:
        _keys(demo, ("query", "old", "replacement", "source_sha256", "query_sha256"))
        row = {name: _raw(demo[name], limit)
               for name, limit in (("query", 2048), ("old", 4096), ("replacement", 128))}
        for name in ("source_sha256", "query_sha256"):
            row[name] = _sha(demo[name])
        if hashlib.sha256(row["query"]).hexdigest() != row["query_sha256"]:
            raise ValueError("reference query hash changed")
        demos.append(row)
    _keys(record["new_teacher_offers"], ("reading", "STOP", "SPLICE"))
    for value in record["new_teacher_offers"].values():
        _integer(value)
    return documents, demos, dict(record["new_teacher_offers"])


def _restore_baseline(record):
    from experiments.r4_integrated.persistent_language_action_baseline_v2 import SimpleReference

    documents, demos, offers = _baseline_values(record)
    result = SimpleReference()
    result.documents, result.demonstrations = documents, demos
    result.new_teacher_offers = offers
    return result


def _tensors(value):
    # Called only in explicit snapshot/restore, never during import or inspect.
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _actors(models, work):
    from humanbrain.sequence_assembly_splice_learner import SequenceAssemblySpliceLearner

    _keys(models, NEURAL_ROUTES)
    if models["online"] is models["learning_disabled"]:
        raise ValueError("routes must own independent actor instances")
    storages = []
    for model in models.values():
        if not isinstance(model, SequenceAssemblySpliceLearner):
            raise ValueError("checkpoint requires strict sequence-assembly combined actors")
        model._require_idle()
        model._check(work)
        values = list(_tensors(model.state_dict()))
        for view in model.reader._candidate_views:
            values.extend((view.indices, view.values, view.flags))
        for wiring in model.reader._self_wirings:
            values.extend((wiring.indices, wiring.values))
        storages.append({v.untyped_storage().data_ptr() for v in values if v.numel()})
    if storages[0] & storages[1]:
        raise ValueError("checkpoint routes share numerical or derived storage")


class PersistentSessionCheckpoint:
    """One source-bound session store; only this instance may finish its intent.

    Host state is parent-only. No private grade, cursor or checkpoint identifier
    is passed to a learner by this module. Failed restore actors must be discarded.
    """

    @classmethod
    def create(cls, directory, *, source_root, evidence_root, manifest_path,
               manifest_sha256, source_revision, source_sha256, session_id):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        obj = cls.__new__(cls)
        obj.directory = directory
        obj.work = {}
        obj._intent = None
        raw = obj._read(Path(manifest_path), JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != _sha(manifest_sha256):
            raise ValueError("frozen session manifest hash mismatch")
        if (not isinstance(session_id, str) or not session_id
                or not isinstance(source_revision, str) or len(source_revision) != 40
                or any(c not in "0123456789abcdef" for c in source_revision)):
            raise ValueError("session requires an identity and committed source revision")
        if not isinstance(source_sha256, dict) or not source_sha256:
            raise ValueError("session source inventory is required")
        obj.config = {
            "schema": "persistent-session-checkpoint-config-v1", "session_id": session_id,
            "source_revision": source_revision, "source_root": str(Path(source_root).resolve()),
            "evidence_root": str(Path(evidence_root).resolve()),
            "source_sha256": dict(source_sha256), "manifest_sha256": manifest_sha256,
        }
        obj._write(directory / "manifest.json", raw)
        obj._write(directory / "config.json", encode(obj.config))
        obj.config_sha256 = hashlib.sha256(encode(obj.config)).hexdigest()
        obj._binding()
        # Call create before initializer/ingests: this intent protects that work.
        obj._initial_intent()
        return obj

    @classmethod
    def open(cls, directory, *, config_sha256):
        obj = cls.__new__(cls)
        obj.directory = Path(directory).resolve()
        obj.work = {}
        obj._intent = None
        raw = obj._read(obj.directory / "config.json", JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != _sha(config_sha256):
            raise ValueError("checkpoint configuration hash mismatch")
        obj.config = json.loads(raw)
        obj.config_sha256 = config_sha256
        obj._binding()
        return obj

    def _add(self, name, value):
        self.work[name] = self.work.get(name, 0) + value

    def _initial_intent(self):
        intent = {"schema": "persistent-session-checkpoint-intent-v1", "sequence": 0,
                  "kind": "initialize", "cursor": {"episode_index": 0, "route_index": 0,
                                                     "attempt_index": 1, "append_done": False},
                  "observation_sha256": None, "previous_sha256": None,
                  "config_sha256": self.config_sha256}
        raw = encode(intent)
        self._write(self.directory / "intent-000000000.json", raw)
        self._intent = intent, hashlib.sha256(raw).hexdigest()

    def _read(self, path, limit):
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        self._add("checkpoint_bytes_read", len(raw))
        if len(raw) > limit:
            raise ValueError("checkpoint input exceeds its declared bound")
        return raw

    def _write(self, path, raw):
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self._add("checkpoint_bytes_written", len(raw))
        self._add("checkpoint_fsync_calls", 1)

    def _file(self, path, *, prefix=None):
        digest = hashlib.sha256()
        size, rows, last = 0, 0, b""
        with path.open("rb") as handle:
            while prefix is None or size < prefix:
                chunk = handle.read(1024 * 1024 if prefix is None
                                    else min(1024 * 1024, prefix - size))
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                rows += chunk.count(b"\n")
                last = chunk[-1:]
        self._add("checkpoint_hash_bytes_read", size)
        self._add("checkpoint_file_hash_passes", 1)
        if prefix is not None and size != prefix:
            raise ValueError("checkpoint evidence prefix was shortened")
        return {"bytes": size, "sha256": digest.hexdigest(), "rows": rows,
                "complete": not size or last == b"\n"}

    def _binding(self):
        _keys(self.config, ("schema", "session_id", "source_revision", "source_root",
                            "evidence_root", "source_sha256", "manifest_sha256"))
        raw = self._read(self.directory / "config.json", JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != self.config_sha256:
            raise ValueError("checkpoint binding changed")
        if self.config["schema"] != "persistent-session-checkpoint-config-v1":
            raise ValueError("unknown checkpoint configuration")
        root = Path(self.config["source_root"]).resolve()
        for name, expected in self.config["source_sha256"].items():
            if self._file(_relative(root, name))["sha256"] != _sha(expected):
                raise ValueError("session source changed")
        manifest_raw = self._read(self.directory / "manifest.json", JSON_LIMIT)
        if hashlib.sha256(manifest_raw).hexdigest() != _sha(self.config["manifest_sha256"]):
            raise ValueError("session manifest changed")
        self.manifest = json.loads(manifest_raw)
        episodes = self.manifest.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("session manifest needs its fixed episode order")

    def _cursor(self, cursor):
        _keys(cursor, ("episode_index", "route_index", "attempt_index", "append_done"))
        for name in ("episode_index", "route_index", "attempt_index"):
            _integer(cursor[name])
        if type(cursor["append_done"]) is not bool:
            raise ValueError("append frontier must be explicit")
        i, route, attempt = (cursor[name] for name in
                             ("episode_index", "route_index", "attempt_index"))
        if i == len(self.manifest["episodes"]):
            if (route, attempt, cursor["append_done"]) != (0, 1, False):
                raise ValueError("invalid terminal cursor")
        elif (i > len(self.manifest["episodes"]) or route >= len(ROUTES)
              or not 1 <= attempt <= self.manifest["episodes"][i]["attempt_limit"]):
            raise ValueError("cursor lies outside the fixed stream")

    def _next(self, intent):
        cursor = dict(intent["cursor"])
        if intent["kind"] == "append":
            cursor["append_done"] = True
        elif cursor["attempt_index"] < self.manifest["episodes"][cursor["episode_index"]][
                "attempt_limit"]:
            cursor["attempt_index"] += 1
            cursor["append_done"] = True
        else:
            cursor["route_index"] += 1
            if cursor["route_index"] == len(ROUTES):
                cursor["route_index"] = 0
                cursor["episode_index"] += 1
            cursor["attempt_index"], cursor["append_done"] = 1, False
        return cursor

    def _host(self, host):
        _keys(host, ("cursor", "projects", "current", "counters", "elapsed_seconds",
                     "budget_seconds", "budget_mib", "initial_bindings"))
        self._cursor(host["cursor"])
        _keys(host["projects"], ROUTES)
        for projects in host["projects"].values():
            if not isinstance(projects, dict) or len(projects) > 72:
                raise ValueError("project map exceeds the bounded session")
            for key, value in projects.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("project map needs explicit public identities")
                _workspace(value)
        if host["current"] is not None:
            _workspace(host["current"], current=True)
        _keys(host["counters"], ROUTES)
        for counts in host["counters"].values():
            _keys(counts, COUNTERS)
            for value in counts.values():
                _integer(value)
        for name in ("elapsed_seconds", "budget_seconds", "budget_mib"):
            value = host[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid accumulated timing/budget")
        if host["budget_seconds"] <= 0 or host["budget_mib"] <= 0:
            raise ValueError("session budget must be positive")
        if not isinstance(host["initial_bindings"], dict) or not host["initial_bindings"]:
            raise ValueError("initialization and prior evidence bindings are required")
        for value in host["initial_bindings"].values():
            _sha(value)
        if len(encode(host)) > JSON_LIMIT:
            raise ValueError("host checkpoint exceeds JSON bound")

    def capture_frontiers(self, journals, *, execution_root):
        """Hash complete closed/flushed evidence; paths are relative to evidence_root.

        journals maps each kind to its chronological segment file names. This
        method neither appends/replays journals nor launches or parses programs.
        """
        _keys(journals, KINDS)
        root = Path(self.config["evidence_root"]).resolve()
        records, seen = [], set()
        for kind in KINDS:
            if not isinstance(journals[kind], list) or not journals[kind]:
                raise ValueError("each journal requires its ordered segment paths")
            for name in journals[kind]:
                if name in seen:
                    raise ValueError("duplicate journal segment")
                seen.add(name)
                item = self._file(_relative(root, name))
                if not item.pop("complete"):
                    raise DirtyCheckpoint("journal ends inside a record")
                records.append({"kind": kind, "path": name, **item})
        execution = _relative(root, execution_root)
        files, directories = [], []
        if execution.exists():
            for path in sorted(execution.rglob("*")):
                if path.is_file():
                    name = path.relative_to(root).as_posix()
                    item = self._file(_relative(root, name))
                    files.append({"path": name, "bytes": item["bytes"], "sha256": item["sha256"]})
                elif path.is_dir():
                    name = path.relative_to(root).as_posix()
                    _relative(root, name)
                    directories.append(name)
        return {"journals": records, "execution_root": execution_root,
                "execution_files": files, "execution_directories": directories}

    def _frontiers(self, value, *, allow_tail=False):
        _keys(value, ("journals", "execution_root", "execution_files", "execution_directories"))
        root = Path(self.config["evidence_root"]).resolve()
        dirty = []
        paths = {kind: [] for kind in KINDS}
        for row in value["journals"]:
            _keys(row, ("kind", "path", "bytes", "rows", "sha256"))
            if row["kind"] not in KINDS:
                raise ValueError("unknown checkpoint journal")
            path = _relative(root, row["path"])
            if any(row["path"] in names for names in paths.values()):
                raise ValueError("duplicate evidence journal path")
            paths[row["kind"]].append(row["path"])
            item = self._file(path, prefix=_integer(row["bytes"]))
            if (not item["complete"] or item["sha256"] != _sha(row["sha256"])
                    or item["rows"] != _integer(row["rows"])):
                raise ValueError("committed journal prefix changed")
            if path.stat().st_size != row["bytes"]:
                dirty.append("journal beyond boundary: " + row["path"])
        if any(not values for values in paths.values()):
            raise ValueError("checkpoint lacks a journal kind")
        execution = _relative(root, value["execution_root"])
        actual = ({path.relative_to(root).as_posix()
                   for path in execution.rglob("*") if path.is_file()}
                  if execution.exists() else set())
        expected = set()
        for row in value["execution_files"]:
            _keys(row, ("path", "bytes", "sha256"))
            path = _relative(root, row["path"])
            if not path.is_relative_to(execution) or row["path"] in expected:
                raise ValueError("invalid execution inventory")
            expected.add(row["path"])
            item = self._file(path)
            if item["bytes"] != _integer(row["bytes"]) or item["sha256"] != _sha(row["sha256"]):
                raise ValueError("committed actual execution file changed")
        if actual != expected:
            dirty.append("execution files beyond boundary")
        actual_dirs = sorted(path.relative_to(root).as_posix()
                             for path in execution.rglob("*") if path.is_dir())
        if actual_dirs != value["execution_directories"]:
            dirty.append("execution directories beyond boundary")
        if dirty and not allow_tail:
            raise DirtyCheckpoint("; ".join(dirty))
        return dirty

    def _boundary(self):
        self._binding()
        raw = self._read(self.directory / "HEAD.json", JSON_LIMIT)
        head = json.loads(raw)
        _keys(head, ("schema", "sequence", "manifest", "sha256"))
        if head["schema"] != "persistent-session-checkpoint-head-v1":
            raise ValueError("invalid checkpoint HEAD schema")
        sequence = _integer(head["sequence"])
        if head["manifest"] != f"boundary-{sequence:09d}.json":
            raise ValueError("HEAD boundary coordinate mismatch")
        raw = self._read(_relative(self.directory, head["manifest"]), JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != _sha(head["sha256"]):
            raise ValueError("checkpoint boundary hash mismatch")
        boundary = json.loads(raw)
        _keys(boundary, ("schema", "sequence", "config_sha256", "previous_sha256",
                         "intent_sha256", "routes", "baseline", "host", "frontiers", "work"))
        if (boundary["schema"] != "persistent-session-checkpoint-boundary-v1"
                or boundary["sequence"] != sequence
                or boundary["config_sha256"] != self.config_sha256):
            raise ValueError("checkpoint boundary identity mismatch")
        intent_raw = self._read(self.directory / f"intent-{sequence:09d}.json", JSON_LIMIT)
        if hashlib.sha256(intent_raw).hexdigest() != _sha(boundary["intent_sha256"]):
            raise ValueError("committed boundary lost its actual intent")
        intent = json.loads(intent_raw)
        if (intent["sequence"] != sequence or intent["config_sha256"] != self.config_sha256
                or intent["previous_sha256"] != boundary["previous_sha256"]):
            raise ValueError("committed intent and boundary lineage differ")
        self._host(boundary["host"])
        _baseline_values(boundary["baseline"])
        _keys(boundary["routes"], NEURAL_ROUTES)
        for row in boundary["routes"].values():
            _keys(row, ("payload", "sha256", "bytes", "model_sha256", "memory_sha256",
                        "experience", "reader_experience", "mode", "seed", "actor_class"))
            for name in ("experience", "reader_experience"):
                _integer(row[name])
            _sha(row["model_sha256"])
            _sha(row["memory_sha256"])
            item = self._file(_relative(self.directory, row["payload"]))
            if (item["bytes"] != _integer(row["bytes"]) or item["bytes"] > PAYLOAD_LIMIT
                    or item["sha256"] != _sha(row["sha256"])):
                raise ValueError("checkpoint tensor payload changed")
        return boundary, head["sha256"]

    def _events(self, previous, frontiers, intent):
        """Require exactly the scheduled journal deltas, without redoing any task."""
        root = Path(self.config["evidence_root"]).resolve()
        old = previous["frontiers"]["journals"] if previous else []
        events = {}
        for kind in KINDS:
            before = [row for row in old if row["kind"] == kind]
            after = [row for row in frontiers["journals"] if row["kind"] == kind]
            if [row["path"] for row in after[:len(before)]] != [row["path"] for row in before]:
                raise ValueError("checkpoint removed or reordered evidence segments")
            by_path = {row["path"]: row for row in before}
            added = []
            for index, row in enumerate(after):
                prior = by_path.get(row["path"], {"bytes": 0, "rows": 0})
                if row["rows"] < prior["rows"] or row["bytes"] < prior["bytes"]:
                    raise ValueError("checkpoint shortened an evidence segment")
                if row["bytes"] != prior["bytes"]:
                    if index < len(before) - 1:
                        raise ValueError("only the latest journal segment can advance")
                    added_bytes = row["bytes"] - prior["bytes"]
                    if added_bytes > 64 * 1024 * 1024:
                        raise ValueError("one journal event exceeds its read bound")
                    with _relative(root, row["path"]).open("rb") as handle:
                        handle.seek(prior["bytes"])
                        raw = handle.read(added_bytes + 1)
                    self._add("checkpoint_event_bytes_read", len(raw))
                    if (len(raw) != added_bytes or raw.count(b"\n") != 1
                            or not raw.endswith(b"\n")):
                        raise ValueError("a boundary can add only one bounded record per journal")
                    added.append((json.loads(raw), hashlib.sha256(raw).hexdigest()))
            wanted = 0
            if intent["kind"] == "append":
                wanted = int(kind == "ingests")
            elif intent["kind"] == "attempt":
                episode = self.manifest["episodes"][intent["cursor"]["episode_index"]]
                wanted = int(kind == "attempts" or (
                    kind == "teachers" and episode.get("teacher") is not None))
            if len(added) != wanted:
                raise ValueError("boundary journal counts do not match the scheduled transaction")
            if added:
                events[kind] = added[0]
        if intent["kind"] != "initialize":
            episode = self.manifest["episodes"][intent["cursor"]["episode_index"]]
            route = ROUTES[intent["cursor"]["route_index"]]
            for row, _digest in events.values():
                if row.get("route") != route or row.get("episode_id") != episode["episode_id"]:
                    raise ValueError("journal row does not belong to the owned intent")
            if intent["kind"] == "append":
                if events["ingests"][0]["document_ids"] != episode["append_document_ids"]:
                    raise ValueError("checkpoint binds a different document append")
            else:
                row, attempt_sha = events["attempts"]
                if (row["attempt_index"] != intent["cursor"]["attempt_index"]
                        or row["observation_record_sha256"] != intent["observation_sha256"]):
                    raise ValueError("attempt identity or actual observation changed")
                if (events.get("teachers")
                        and events["teachers"][0]["attempt_sha256"] != attempt_sha):
                    raise ValueError("correction is not bound to the durable actual response")
                if "teachers" in events:
                    teacher = events["teachers"][0]
                    if (teacher["teacher"] != episode["teacher"]
                            or teacher["learning_enabled"] is not (route != "learning_disabled")):
                        raise ValueError("correction changed the offered target or route control")
        return events

    def _progress(self, previous, host, events, intent):
        """Bind host counts and actual post-attempt workspace to its durable row."""
        if previous is None:
            if any(value for counts in host["counters"].values() for value in counts.values()):
                raise ValueError("initial session counters must be zero")
            return
        expected = json.loads(encode(previous["host"]["counters"]))
        route = ROUTES[intent["cursor"]["route_index"]]
        counts = expected[route]
        if "ingests" in events:
            row = events["ingests"][0]
            counts["ingest_documents"] += len(row["document_ids"])
            counts["ingest_raw_bytes"] += _integer(row["raw_bytes"])
            counts["ingest_eos"] += _integer(row["eos"])
        if "attempts" in events:
            row = events["attempts"][0]
            counts["attempts"] += 1
            counts["executions"] += int(row["execution"] is not None)
            feedback = row["observation"].get("feedback")
            if row["execution"] is not None:
                feedback = {name: row["execution"][name] for name in (
                    "returncode", "stdout_hex", "stderr_hex", "accepted", "timeout",
                    "capture_complete", "files_complete", "output_limit_exceeded")}
            actual = {"route": route, "episode_id": row["episode_id"],
                      "files": row["after_files"], "feedback": feedback}
            if host["current"] != actual:
                raise ValueError("checkpoint current workspace is not the actual outcome")
            projects = json.loads(encode(previous["host"]["projects"]))
            if row.get("project_id") is not None:
                projects[route][row["project_id"]] = {
                    k: v for k, v in actual.items() if k != "route"}
            if host["projects"] != projects:
                raise ValueError("project checkpoint must preserve all actual route predecessors")
        elif (host["projects"] != previous["host"]["projects"]
              or host["current"] != previous["host"]["current"]):
            raise ValueError("document append changed actual workspace state")
        if "teachers" in events:
            row = events["teachers"][0]
            counts["teacher_occasions"] += 1
            counts["reader_offers"] += _integer(row["reader_offers"])
            counts["action_labels"] += _integer(row["action_labels"])
        if host["counters"] != expected:
            raise ValueError("session exposure counters differ from actual journal deltas")

    def _reference_progress(self, previous, record, events, intent):
        if previous is None:
            return
        expected = json.loads(encode(previous["baseline"]))
        route = ROUTES[intent["cursor"]["route_index"]]
        if route == "simple":
            if "ingests" in events:
                expected["documents_hex"].extend(
                    row["text_hex"] for row in events["ingests"][0]["documents"])
            if "teachers" in events:
                teacher = events["teachers"][0]["teacher"]
                kind = "reading" if teacher["kind"] == "reading" else teacher["operation"]
                expected["new_teacher_offers"][kind] += 1
                if kind == "SPLICE":
                    if len(record["demonstrations"]) != len(expected["demonstrations"]) + 1:
                        raise ValueError("reference needs exactly one new SPLICE demonstration")
                    new = record["demonstrations"][-1]
                    if new["replacement"] != teacher["replacement_hex"]:
                        raise ValueError("reference checkpoint changed the actual supplied payload")
                    expected["demonstrations"].append(new)
        if record != expected:
            raise ValueError("reference history changed outside its actual append/correction")

    def inspect(self, *, require_clean=True):
        """Authenticate the last safe prior without importing/deserializing tensors."""
        boundary, digest = self._boundary()
        dirty = self._frontiers(boundary["frontiers"], allow_tail=True)
        for path in self.directory.glob("intent-*.json"):
            intent = json.loads(self._read(path, JSON_LIMIT))
            if _integer(intent["sequence"]) > boundary["sequence"]:
                dirty.append("outstanding intent: " + path.name)
        if dirty and require_clean:
            raise DirtyCheckpoint("; ".join(dirty))
        return {"status": "dirty" if dirty else "clean", "reasons": dirty,
                "boundary_sha256": digest, "boundary": boundary}

    def begin(self, kind, cursor, *, observation_sha256=None):
        """Durably spend the next append/attempt before it can perform any work."""
        if self._intent is not None:
            raise DirtyCheckpoint("this store already has an unresolved intent")
        current = self.inspect()
        previous = current["boundary"]
        self._cursor(cursor)
        if cursor != previous["host"]["cursor"] or cursor["episode_index"] == len(
                self.manifest["episodes"]):
            raise ValueError("intent must be the exact next planned transaction")
        episode = self.manifest["episodes"][cursor["episode_index"]]
        if kind == "append":
            if cursor["append_done"] or not episode["append_document_ids"] or observation_sha256:
                raise ValueError("invalid document append intent")
        elif kind == "attempt":
            _sha(observation_sha256)
            if episode["append_document_ids"] and not cursor["append_done"]:
                raise ValueError("the planned append must commit before prediction")
        else:
            raise ValueError("intent kind must be append or attempt")
        intent = {"schema": "persistent-session-checkpoint-intent-v1",
                  "sequence": previous["sequence"] + 1, "kind": kind,
                  "cursor": dict(cursor), "observation_sha256": observation_sha256,
                  "previous_sha256": current["boundary_sha256"],
                  "config_sha256": self.config_sha256}
        raw = encode(intent)
        self._write(self.directory / f"intent-{intent['sequence']:09d}.json", raw)
        self._intent = intent, hashlib.sha256(raw).hexdigest()
        return self._intent[1]

    def commit(self, models, states, baseline, host, frontiers):
        """Commit an initial idle state or this instance's fully resolved intent.

        Caller must have reaped every action child and flushed all journals.
        A successful commit never clears pending state on the caller's behalf.
        """
        import torch

        from humanbrain.neuronal_splice_learner import _digest

        tick = time.perf_counter()
        self._binding()
        self._host(host)
        _actors(models, self.work)
        _keys(states, NEURAL_ROUTES)
        self._frontiers(frontiers)
        previous, previous_sha = None, None
        if (self.directory / "HEAD.json").exists():
            previous, previous_sha = self._boundary()
            if self._intent is None or self._intent[0]["previous_sha256"] != previous_sha:
                raise DirtyCheckpoint("only the owning live intent can publish the next boundary")
            intent, intent_sha = self._intent
            if host["cursor"] != self._next(intent):
                raise ValueError("checkpoint does not advance exactly one transaction")
            self._frontiers(previous["frontiers"], allow_tail=True)
            if (host["initial_bindings"] != previous["host"]["initial_bindings"]
                    or host["budget_seconds"] != previous["host"]["budget_seconds"]
                    or host["budget_mib"] != previous["host"]["budget_mib"]
                    or host["elapsed_seconds"] < previous["host"]["elapsed_seconds"]):
                raise ValueError("continuation resets a prior binding or resource budget")
        else:
            if (self._intent is None or self._intent[0]["kind"] != "initialize"
                    or host["cursor"] != {
                    "episode_index": 0, "route_index": 0,
                    "attempt_index": 1, "append_done": False}):
                raise ValueError("initial checkpoint must precede the first transaction")
            intent, intent_sha = self._intent
        sequence = intent["sequence"]
        raw_intent = self._read(self.directory / f"intent-{sequence:09d}.json", JSON_LIMIT)
        if hashlib.sha256(raw_intent).hexdigest() != intent_sha:
            raise ValueError("owned intent changed")
        events = self._events(previous, frontiers, intent)
        self._progress(previous, host, events, intent)
        routes, seal = {}, {}
        for route in NEURAL_ROUTES:
            model, state = models[route], states[route]
            model.reader.memory.check_state(state)
            model_hash = _digest(model.state_dict(), self.work)
            memory_hash = _digest(vars(state), self.work)
            counters = (_integer(model._experience), _integer(model.reader._active_experience))
            seal[route] = model_hash, memory_hash, counters
            old = None if previous is None else previous["routes"][route]
            if old:
                current_route = ROUTES[intent["cursor"]["route_index"]]
                predicted = (route == current_route and "attempts" in events
                             and events["attempts"][0]["prediction"] is not None)
                edited = predicted and events["attempts"][0]["kind"] == "edit"
                if counters != (old["experience"] + int(edited),
                                old["reader_experience"] + int(predicted)):
                    raise ValueError("reader/action custody counters differ from actual responses")
                if route != current_route:
                    wanted = old["model_sha256"], old["memory_sha256"]
                else:
                    latest = events.get("teachers", events.get("attempts", events.get("ingests")))
                    wanted = latest[0]["model_after_sha256"], latest[0]["memory_after_sha256"]
                if (model_hash, memory_hash) != wanted:
                    raise ValueError("checkpoint state differs from its durable outcome frontier")
            if old and (old["model_sha256"], old["memory_sha256"]) == (model_hash, memory_hash):
                record = dict(old)
                self._add("checkpoint_payload_reuses", 1)
            else:
                snapshot = model.snapshot(state)
                # Numeric payloads are reusable across free custody-counter changes.
                payload = {"schema": "persistent-session-route-tensors-v1",
                           "model": snapshot["model"], "memory": snapshot["memory"]}
                self._add("checkpoint_snapshot_calls", 1)
                self._add("checkpoint_snapshot_uninstrumented_validation_passes", 1)
                self._add("checkpoint_snapshot_tensor_bytes", sum(
                    value.numel() * value.element_size() for value in _tensors(payload)))
                name = f"route-{route}-{sequence:09d}.pt"
                temporary = self.directory / (name + ".tmp")
                with temporary.open("xb") as handle:
                    torch.save(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._add("checkpoint_fsync_calls", 1)
                item = self._file(temporary)
                if item["bytes"] > PAYLOAD_LIMIT:
                    raise ValueError("route checkpoint exceeds payload bound")
                temporary.rename(self.directory / name)
                self._add("checkpoint_bytes_written", item["bytes"])
                self._add("checkpoint_payload_writes", 1)
                record = {"payload": name, "sha256": item["sha256"], "bytes": item["bytes"],
                          "model_sha256": model_hash, "memory_sha256": memory_hash}
                del snapshot, payload
            record.update(experience=counters[0], reader_experience=counters[1],
                          mode=model.mode, seed=model.seed,
                          actor_class=f"{type(model).__module__}.{type(model).__qualname__}")
            routes[route] = record
        baseline_record = _baseline_record(baseline)
        _baseline_values(baseline_record)
        self._reference_progress(previous, baseline_record, events, intent)
        self._add("checkpoint_reference_records", 1)
        boundary = {"schema": "persistent-session-checkpoint-boundary-v1", "sequence": sequence,
                    "config_sha256": self.config_sha256, "previous_sha256": previous_sha,
                    "intent_sha256": intent_sha, "routes": routes, "baseline": baseline_record,
                    "host": host, "frontiers": frontiers, "work": dict(self.work)}
        _actors(models, self.work)
        for route in NEURAL_ROUTES:
            model = models[route]
            if seal[route] != (_digest(model.state_dict(), self.work),
                               _digest(vars(states[route]), self.work),
                               (model._experience, model.reader._active_experience)):
                raise ValueError("actor changed during checkpoint construction")
        self._frontiers(frontiers)
        self._binding()
        raw = encode(boundary)
        if len(raw) > JSON_LIMIT:
            raise ValueError("boundary exceeds JSON bound")
        name = f"boundary-{sequence:09d}.json"
        self._write(self.directory / name, raw)
        head = {"schema": "persistent-session-checkpoint-head-v1", "sequence": sequence,
                "manifest": name, "sha256": hashlib.sha256(raw).hexdigest()}
        temporary = self.directory / f"HEAD-{uuid.uuid4().hex}.tmp"
        self._write(temporary, encode(head))
        os.replace(temporary, self.directory / "HEAD.json")
        self._intent = None
        self._add("checkpoint_head_publications", 1)
        return {"boundary_sha256": head["sha256"], "sequence": sequence,
                "elapsed_seconds": time.perf_counter() - tick, "work": dict(self.work)}

    def restore_into(self, models):
        """Restore clean state into fresh caller-owned actors; never replay a lesson.

        Both payloads and derived caches are prevalidated before the first copy.
        Caller must discard both actors if any exception occurs, and launch no
        task until this method returns. Actual constructors are caller-accounted.
        """
        import torch

        from humanbrain.neuronal_splice_learner import _digest

        tick = time.perf_counter()
        checked = self.inspect()
        boundary = checked["boundary"]
        _actors(models, self.work)
        if any(model._experience != 0 or model.reader._active_experience != 0
               for model in models.values()):
            raise ValueError("restore requires fresh zero-experience caller actors")
        baseline = _restore_baseline(boundary["baseline"])
        payloads, states = {}, {}
        for route in NEURAL_ROUTES:
            model, row = models[route], boundary["routes"][route]
            actor_class = f"{type(model).__module__}.{type(model).__qualname__}"
            if ((model.mode, model.seed) != (row["mode"], row["seed"])
                    or actor_class != row["actor_class"]):
                raise ValueError("restore actor mode/seed differs from the saved actor")
            # Bind exactly the bytes passed to the restricted decoder, avoiding
            # a path reopen between authentication and torch.load.
            raw = self._read(_relative(self.directory, row["payload"]), PAYLOAD_LIMIT)
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("tensor payload changed before restricted decoding")
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            self._add("checkpoint_torch_load_calls", 1)
            del raw
            _keys(payload, ("schema", "model", "memory"))
            if payload["schema"] != "persistent-session-route-tensors-v1":
                raise ValueError("unknown tensor payload schema")
            expected = model.state_dict()
            _keys(payload["model"], expected)
            for name, prototype in expected.items():
                value = payload["model"][name]
                if isinstance(prototype, torch.Tensor):
                    if (not isinstance(value, torch.Tensor) or value.shape != prototype.shape
                            or value.dtype != prototype.dtype or value.device != prototype.device
                            or value.requires_grad or value.grad_fn is not None
                            or value.grad is not None
                            or not bool(torch.isfinite(value).all())):
                        raise ValueError("incoming route tensor contract mismatch")
                    self._add("checkpoint_prevalidation_tensor_values", value.numel())
                elif value != prototype:
                    raise ValueError("incoming route metadata mismatch")
            for name in model.reader.FIXED_STATE_NAMES:
                if not torch.equal(payload["model"]["reader." + name], expected["reader." + name]):
                    raise ValueError("incoming fixed reader wiring changed")
            model.reader.context.validate_fixed_state(payload["model"], "reader.context.")
            prepared = model.reader.prepare_state_views(payload["model"], "reader.", self.work)
            del prepared  # The strict public loader rebuilds these again; both calls are charged.
            states[route] = model.reader.memory.restore_state(payload["memory"])
            if (_digest(payload["model"], self.work) != row["model_sha256"]
                    or _digest(vars(states[route]), self.work) != row["memory_sha256"]):
                raise ValueError("incoming numerical state hash mismatch")
            payloads[route] = payload
        # No caller tensor has been overwritten before every route is checked.
        for route in NEURAL_ROUTES:
            model, row = models[route], boundary["routes"][route]
            model.load_state_dict(
                payloads[route]["model"], strict=True, assign=False, work=self.work)
            model._experience = row["experience"]
            model.reader._active_experience = row["reader_experience"]
        _actors(models, self.work)
        storage = [{v.untyped_storage().data_ptr() for v in _tensors(vars(state)) if v.numel()}
                   for state in states.values()]
        if storage[0] & storage[1]:
            raise ValueError("restored tapes share storage")
        for route in NEURAL_ROUTES:
            if (_digest(models[route].state_dict(), self.work)
                    != boundary["routes"][route]["model_sha256"]):
                raise ValueError("strict restore changed numerical state")
        final = self.inspect()
        if final["boundary_sha256"] != checked["boundary_sha256"]:
            raise ValueError("checkpoint advanced during restore")
        self._add("checkpoint_restore_calls", 1)
        return {"states": states, "baseline": baseline, "host": boundary["host"],
                "boundary_sha256": checked["boundary_sha256"], "frontiers": boundary["frontiers"],
                "elapsed_seconds": time.perf_counter() - tick, "work": dict(self.work)}
