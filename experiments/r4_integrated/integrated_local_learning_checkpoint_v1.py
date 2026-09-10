"""Source-bound compact512-per-skill block using unchanged strict payload custody."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import time
import uuid
from pathlib import Path

from experiments.r4_integrated import integrated_local_learning_runtime_v1 as runtime
from experiments.r4_integrated.persistent_session_checkpoint import (
    JSON_LIMIT,
    NEURAL_ROUTES,
    ROUTES,
    _actors,
    _baseline_record,
    _keys,
    _relative,
    _sha,
    encode,
)
from experiments.r4_integrated.repeated_teaching_checkpoint_v1 import (
    PARENT_MEMORY_SHA,
    ROOT,
    ROUTE_LIMIT,
    RepeatedTeachingCheckpoint,
    _canonical,
)

SCHEDULE = ["A00-R"] * 64 + ["A00-E"] * 64 + ["A00-R", "A00-E"] * 448
CHECKPOINTS = (0, 64, 128, 256, 512, 768, 1024)


def validate_compact_teacher(value, target, *, learning_enabled):
    """Validate the declared compact evidence, without inventing omitted vectors."""
    if type(target) is not bytes or type(learning_enabled) is not bool:
        raise ValueError("compact validation needs exact target bytes and Boolean gate")
    tokens = [*target, 256]
    rows = value["positions"]
    if (
        value["schema"] != "integrated-local-learning-practice-teacher-v1"
        or value["target_hex"] != target.hex()
        or len(rows) != len(tokens)
        or value["offers"] != len(tokens)
        or value["raw_offers"] != len(target)
        or value["eos_offers"] != 1
        or value["extra_literal_evaluations"] != 0
        or any(
            type(value[key]) is not int or value[key] != 0
            for key in ("draws", "teacher_calls", "prior_calls", "publications")
        )
    ):
        raise ValueError("compact teacher denominator or target changed")
    selected = 0
    for index, (row, token) in enumerate(zip(rows, tokens, strict=True)):
        mode = row["selected_mode"]
        if (
            type(mode) is not int
            or not 0 <= mode < 4
            or row["position"] != index
            or row["target"] != token
            or row["eos"] is not (token == 256)
            or row["offered"] is not True
            or row["selected_literal"] is not (mode == 3)
            or row["direct_teacher_published"] is not (mode == 3 and learning_enabled)
        ):
            raise ValueError("compact teacher contact or publication differs")
        probabilities = row["mode_probabilities"]
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != 4
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1
                for p in probabilities
            )
            or not math.isclose(sum(probabilities), 1, abs_tol=1e-12, rel_tol=1e-12)
        ):
            raise ValueError("compact actual mode probabilities differ")
        p, logp = (
            row["selected_literal_target_probability"],
            row["selected_literal_target_log_probability"],
        )
        if mode == 3:
            if (
                type(p) not in (int, float)
                or type(logp) not in (int, float)
                or not math.isfinite(p)
                or not 0 < p <= 1
                or not math.isfinite(logp)
                or logp > 1e-12
                or not math.isclose(p, math.exp(logp), rel_tol=1e-10, abs_tol=math.ulp(p))
            ):
                raise ValueError("invalid selected literal contact probability")
            selected += 1
        elif p is not None or logp is not None:
            raise ValueError("compact teacher evaluated an unselected literal")
    if value["literal_offers"] != selected or value["direct_teacher_published_offers"] != (
        selected if learning_enabled else 0
    ):
        raise ValueError("compact effective contact totals differ")
    return selected


class IntegratedLearningCheckpoint(RepeatedTeachingCheckpoint):
    """One whole declared practice block; no intermediate resumable checkpoint."""

    def _binding(self):
        raw = self._read(self.directory / "config.json", JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != self.config_sha256:
            raise ValueError("practice config changed")
        if (
            self.config["schema"] != "repeated-teaching-checkpoint-config-v1"
            or Path(self.config["source_root"]).resolve() != ROOT
            or not self.config["source_sha256"]
        ):
            raise ValueError("practice source binding missing")
        for name, identity in self.config["source_sha256"].items():
            if self._file(_relative(ROOT, name))["sha256"] != _sha(identity):
                raise ValueError("practice source changed")
        raw = self._read(self.directory / "manifest.json", JSON_LIMIT)
        if hashlib.sha256(raw).hexdigest() != self.config["manifest_sha256"]:
            raise ValueError("practice manifest changed")
        self.manifest = json.loads(raw)
        if (
            self.manifest["schema"] != "integrated-local-learning-acquisition-v1"
            or self.manifest["schedule"]
            != [{"index": i, "episode_id": name} for i, name in enumerate(SCHEDULE)]
            or self.manifest["checkpoints"] != list(CHECKPOINTS)
            or self.manifest["routes"] != list(ROUTES)
            or self.manifest["checkpoint_cases"] != ["A00-R", "A00-E", "R02-R", "R12-R"]
            or self.config["parent"]["bank_sha256"] != runtime.BANK_SHA
            or self.config["parent"]["parent_boundary_sha256"] != runtime.BOUNDARY_SHA
        ):
            raise ValueError("practice declaration or retained parent differs")

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

        if any(len(indexed[kind]) != 3072 for kind in ("attempts", "teachers")):
            raise ValueError("complete block requires exactly3072 attempts and teachers")

        def read(entry):
            value = read_row(entry)
            self._add("block_ledger_parse_bytes", entry["bytes"])
            return value

        initial = host["initial_hashes"]
        _keys(initial, ROUTES)
        for route in NEURAL_ROUTES:
            if initial[route] != {
                "model": self.config["parent"]["initial_routes"][route]["model_sha256"],
                "memory": PARENT_MEMORY_SHA,
            }:
                raise ValueError("block did not start from paired selected retained ONLINE state")
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
                    validate_compact_teacher(
                        taught["diagnostic"],
                        bytes.fromhex(target),
                        learning_enabled=route == "online",
                    )
                    if [step["target"] for step in taught["diagnostic"]["positions"]] != [
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
        if host["schedule_completed"] != 1024 or host["results"] != results:
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
            3156,
            3114,
            executions,
        ):
            raise ValueError("actual block+probe exposure counts disagree")
        if executions != 1557:
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
            if [r["rows"] for r in frontiers["journals"][:2]] != [3072, 3072]:
                raise ValueError("final block needs3072 actual attempts and3072 bound corrections")
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
                if values[1] != PARENT_MEMORY_SHA or values[2:] != (913, 1430):
                    raise ValueError(
                        "block must preserve tape and count512 edits/1024 responses per route"
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
    with IntegratedLearningCheckpoint.open(directory, config_sha256=config_sha256) as store:
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
