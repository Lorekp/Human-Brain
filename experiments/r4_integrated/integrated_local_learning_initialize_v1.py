"""Explicit predecessor/v3 continuation from the completed repeated-teaching actor.

One restricted parent payload decode supplies two independent matched actors.
Only the current entry adapts the three declared algorithm metadata dictionaries;
ordinary model loaders and all historical checkpoint inspectors stay unchanged.
"""

from __future__ import annotations

import copy
import io
import json
import math
import time
from pathlib import Path

from experiments.r4_integrated import integrated_local_learning_runtime_v1 as runtime
from experiments.r4_integrated.persistent_language_action_v8_initialize import _add, _hash, _read
from experiments.r4_integrated.persistent_session_checkpoint import (
    NEURAL_ROUTES,
    _actors,
    _baseline_record,
    _restore_baseline,
    _tensors,
)
from experiments.r4_integrated.repeated_teaching_checkpoint_v1 import (
    RepeatedTeachingCheckpoint,
    _canonical,
)

PAYLOAD_NAME = "route-online-000000000.pt"
PAYLOAD_SHA = "f77695cd9603e00e031cc867ce53573e2148eb14bb39d2e70b576b317ef66f0e"
PAYLOAD_BYTES = 115712640
MODEL_SHA = "07855c61ca02e1dea8b1e8044df2576643689b8e55a28a1d50109a3580b54d7a"
MEMORY_SHA = "cdd1f887f80ec2e8ae6ee38ae8a15786d9155160cf6beb18def635a2a66da73e"
MODEL_BYTES, TAPE_BYTES = 112389280, 3301384
METADATA_NAMES = ("_extra_state", "reader._extra_state", "reader.controller._extra_state")


def _authenticate(directory, bank_path, predecessor_snapshot, work):
    """Use the historical inspector with exactly two authenticated source aliases."""
    directory, bank_path = Path(directory).resolve(), Path(bank_path).resolve()
    if directory != runtime.PARENT / "checkpoints":
        raise ValueError("initializer requires the exact completed repeated-teaching parent")
    bank = json.loads(_read(bank_path, runtime.BANK_SHA, work))
    if (
        bank["status"] != "completed"
        or bank["restore"]["status"] != "passed"
        or bank["final"]["boundary_sha256"] != runtime.BOUNDARY_SHA
        or bank["restore"]["boundary_sha256"] != runtime.BOUNDARY_SHA
        or bank["final"]["config_sha256"] != runtime.CONFIG_SHA
    ):
        raise ValueError("parent bank does not name the completed restored boundary")
    snapshot = Path(predecessor_snapshot).resolve()
    if snapshot == runtime.ROOT or not snapshot.is_relative_to(runtime.ROOT):
        raise ValueError("predecessor source aliases must stay inside the checkout")
    aliases = {}
    for name, identity in runtime.MODULES.items():
        path = snapshot / (name.rsplit(".", 1)[1] + ".py")
        if path.is_symlink():
            raise ValueError("predecessor source alias cannot be a symbolic link")
        _read(path, identity, work)
        aliases[(runtime.ROOT / runtime._relative(name)).resolve()] = path, identity

    class ParentInspector(RepeatedTeachingCheckpoint):
        def _file(self, path, *, prefix=None):
            replacement = aliases.get(path.resolve())
            if replacement is not None:
                original = path.resolve().relative_to(runtime.ROOT).as_posix()
                path, expected = replacement
                if self.config["source_sha256"].get(original) != expected:
                    raise ValueError("unknown predecessor source exception")
            return super()._file(path, prefix=prefix)

    with ParentInspector.open(directory, config_sha256=runtime.CONFIG_SHA) as store:
        checked = store.inspect(require_clean=True, fresh_payloads=True)
        if checked["boundary_sha256"] != runtime.BOUNDARY_SHA:
            raise ValueError("fresh parent inspector selected a different boundary")
        boundary, config = checked["boundary"], store.config
        if (
            config["source_revision"] != runtime.PARENT_REVISION
            or Path(config["evidence_root"]).resolve() != runtime.PARENT
            or config["manifest_sha256"] != bank["manifest_sha256"]
        ):
            raise ValueError("parent source, evidence root or manifest differs")
        for name, count in store.work.items():
            _add(work, "parent_" + name, count)
    row = boundary["routes"]["online"]
    if any(
        row[key] != value
        for key, value in {
            "payload": PAYLOAD_NAME,
            "sha256": PAYLOAD_SHA,
            "bytes": PAYLOAD_BYTES,
            "model_sha256": MODEL_SHA,
            "memory_sha256": MEMORY_SHA,
            "experience": 401,
            "reader_experience": 406,
            "mode": "learned",
            "seed": 733,
            "actor_class": "humanbrain.recurrent_content_learner.RecurrentContentSpliceLearner",
        }.items()
    ):
        raise ValueError("parent is not the selected retained ONLINE actor")
    return boundary, config


def _mapping(state, model, *, adapt_v3):
    """Validate the complete old state before replacing only algorithm metadata."""
    import torch

    target = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(target):
        raise ValueError("retained model state keys differ")
    expected = {name: copy.deepcopy(target[name]) for name in METADATA_NAMES}
    schemas = (
        "recurrent-content-splice-learner-",
        "recurrent-content-memory-learner-",
        "recurrent-content-controller-",
    )
    suffix = "v3" if adapt_v3 else "v2"
    if any(
        expected[name]["schema"] != schema + suffix
        for name, schema in zip(METADATA_NAMES, schemas, strict=True)
    ):
        raise ValueError("initializer entry differs from the selected algorithm metadata")
    if adapt_v3:
        _predecessor_metadata(expected)
    for name, prototype in target.items():
        value = state[name]
        if isinstance(prototype, torch.Tensor):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != prototype.shape
                or value.dtype != prototype.dtype
                or value.device != prototype.device
                or value.requires_grad
                or value.grad_fn is not None
                or value.grad is not None
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("retained tensor geometry or finite-state contract differs")
        elif value != expected.get(name, prototype):
            raise ValueError("predecessor metadata differs beyond the three declared dictionaries")
    mapped = dict(state)
    if adapt_v3:
        mapped.update({name: copy.deepcopy(target[name]) for name in METADATA_NAMES})
    return mapped, {
        name: {"before": copy.deepcopy(state[name]), "after": copy.deepcopy(mapped[name])}
        for name in METADATA_NAMES
    }


def _predecessor_metadata(metadata):
    """Reverse the exact source-reviewed v3 metadata changes; reject other revisions."""
    changes = {
        "_extra_state": {
            "schema": (
                "recurrent-content-splice-learner-v3",
                "recurrent-content-splice-learner-v2",
            ),
            "reader": (
                "recurrent-content-memory-learner-v3",
                "recurrent-content-memory-learner-v2",
            ),
        },
        "reader._extra_state": {
            "schema": (
                "recurrent-content-memory-learner-v3",
                "recurrent-content-memory-learner-v2",
            ),
            "publication": (
                "held eligibility, prefix credit and phase moments; joint validated copies",
                "held scan/free/teacher eligibility and causal credit; joint validated copies",
            ),
            "local_credit": (
                "integrated-local-learning-revision-v1; Brier; contact CE; scan covariance",
                "causal-observation-credit-v1; direct literal CE; upstream own eligibility",
            ),
        },
        "reader.controller._extra_state": {
            "schema": ("recurrent-content-controller-v3", "recurrent-content-controller-v2"),
            "local_rule": ("integrated-local-learning-revision-v1", "causal-observation-credit-v1"),
            "upstream_objective": (
                "selected-emitter-Brier; context/gate-causal-prefix-means",
                "conditional-match-causal-category-means-over-fixed-L",
            ),
            "literal_objective": (
                "actual-contact-CE; scan-phase-covariance; uncentered-teacher",
                "evaluated-branch-observed-byte-category-mean-cross-entropy",
            ),
            "update_norm_bound": (
                "eta*(sqrt71*S+sqrt74*T+abs(r)*sqrt8*(S+F)/(K*L))",
                math.sqrt(0.0026),
            ),
        },
    }
    additions = {
        "mode_objective": "immediate-own-score-Brier; scan-phase-covariance",
        "scan_association": "alpha-weighted-own-contact-covariance; phase-only-mean",
        "baseline_history": "retained numerical priors; subsequent means use Brier utility",
    }
    if set(metadata) != set(METADATA_NAMES):
        raise ValueError("metadata adapter requires exactly the three algorithm dictionaries")
    for name, fields in changes.items():
        if any(metadata[name].get(key) != current for key, (current, _) in fields.items()):
            raise ValueError("current metadata differs from the declared v3 revision")
    controller = metadata["reader.controller._extra_state"]
    if any(controller.get(key) != value for key, value in additions.items()):
        raise ValueError("current controller metadata additions differ")
    for name, fields in changes.items():
        metadata[name].update({key: previous for key, (_, previous) in fields.items()})
    for key in additions:
        del controller[key]


def _storage(model, state):
    values = list(_tensors(model.state_dict())) + list(_tensors(vars(state)))
    for view in model.reader._candidate_views:
        values.extend((view.indices, view.values, view.flags))
    for wiring in model.reader._self_wirings:
        values.extend((wiring.indices, wiring.values))
    return {value.untyped_storage().data_ptr() for value in values if value.numel()}, len(values)


def _copy_actor(payload, row, actor_class, work, *, adapt_v3):
    """Strict independent copy; used directly by bounded ordinary fixture QA."""
    import torch

    from humanbrain.neuronal_splice_learner import _digest

    _canonical(actor_class)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "model", "memory"}
        or payload["schema"] != "persistent-session-route-tensors-v1"
        or _digest(payload["model"], work) != row["model_sha256"]
        or _digest(payload["memory"], work) != row["memory_sha256"]
    ):
        raise ValueError("decoded state differs from its authenticated numerical hashes")
    model = actor_class(mode=row["mode"], seed=row["seed"])
    _add(work, "model_constructions")
    mapped, metadata = _mapping(payload["model"], model, adapt_v3=adapt_v3)
    model.load_state_dict(mapped, strict=True, assign=False, work=work)
    _add(work, "strict_model_loads")
    state = model.reader.memory.restore_state(payload["memory"])
    model._experience, model.reader._active_experience = row["experience"], row["reader_experience"]
    actual = model.state_dict()
    source_tensors = {
        name: value for name, value in payload["model"].items() if isinstance(value, torch.Tensor)
    }
    if (
        any(not torch.equal(actual[name], value) for name, value in source_tensors.items())
        or _digest(actual, work) != _digest(mapped, work)
        or _digest(vars(state), work) != row["memory_sha256"]
    ):
        raise ValueError("strict copy changed retained tensor, metadata or factual values")
    model._require_idle()
    model._check(work)
    source_storage = {
        value.untyped_storage().data_ptr() for value in _tensors(payload) if value.numel()
    }
    storage, storage_checks = _storage(model, state)
    if source_storage & storage:
        raise ValueError("restored actor shares decoded model, tape or derived storage")
    _add(
        work,
        "retained_model_copy_bytes",
        sum(v.numel() * v.element_size() for v in source_tensors.values()),
    )
    _add(
        work,
        "tape_restore_clone_bytes",
        sum(v.numel() * v.element_size() for v in _tensors(payload["memory"])),
    )
    _add(work, "tensor_equality_values", sum(v.numel() for v in source_tensors.values()))
    _add(work, "storage_identity_checks", storage_checks)
    return model, state, metadata, storage


def clone_for_probe(model, state):
    """Strict same-v3 idle RAM copy, with no metadata adaptation or payload decode."""
    from humanbrain.neuronal_splice_learner import _digest

    started, work = time.perf_counter(), {}
    _canonical(type(model))
    model._require_idle()
    if model.get_extra_state()["schema"] != "recurrent-content-splice-learner-v3":
        raise ValueError("current probe clone requires the selected v3 actor")
    before = (
        _digest(model.state_dict(), work),
        _digest(vars(state), work),
        model._experience,
        model.reader._active_experience,
        model.reader._view_sha256,
        model.reader._self_wiring_sha256,
    )
    parent_storage, _ = _storage(model, state)
    snapshot = model.snapshot(state)
    _add(work, "idle_ram_snapshots")
    _add(
        work,
        "snapshot_tensor_clone_bytes",
        sum(value.numel() * value.element_size() for value in _tensors(snapshot)),
    )
    copied = type(model)(mode=model.mode, seed=model.seed)
    _add(work, "model_constructions")
    copied.load_state_dict(snapshot["model"], strict=True, assign=False, work=work)
    _add(work, "strict_model_loads")
    copied_state = copied.reader.memory.restore_state(snapshot["memory"])
    copied._experience, copied.reader._active_experience = before[2:4]
    copied._require_idle()
    copied._check(work)
    after = (
        _digest(copied.state_dict(), work),
        _digest(vars(copied_state), work),
        copied._experience,
        copied.reader._active_experience,
        copied.reader._view_sha256,
        copied.reader._self_wiring_sha256,
    )
    if after != before or before != (
        _digest(model.state_dict(), work),
        _digest(vars(state), work),
        model._experience,
        model.reader._active_experience,
        model.reader._view_sha256,
        model.reader._self_wiring_sha256,
    ):
        raise ValueError("strict v3 probe copy changed its parent or retained values")
    snapshot_storage = {v.untyped_storage().data_ptr() for v in _tensors(snapshot) if v.numel()}
    copied_storage, storage_checks = _storage(copied, copied_state)
    if copied_storage & (parent_storage | snapshot_storage):
        raise ValueError("probe actor shares live or snapshot storage")
    _add(work, "storage_identity_checks", storage_checks)
    _add(
        work,
        "retained_model_copy_bytes",
        sum(v.numel() * v.element_size() for v in _tensors(snapshot["model"])),
    )
    _add(
        work,
        "tape_restore_clone_bytes",
        sum(v.numel() * v.element_size() for v in _tensors(snapshot["memory"])),
    )
    return {
        "model": copied,
        "state": copied_state,
        "work": work,
        "tensor_decodes": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def initialize_predecessor(
    prior_directory, bank_path, *, actor_class, source_receipt, predecessor_snapshot
):
    """Strict retained-v2 continuation in the selected predecessor worker."""
    return _initialize(
        prior_directory,
        bank_path,
        actor_class=actor_class,
        source_receipt=source_receipt,
        predecessor_snapshot=predecessor_snapshot,
        adapt_v3=False,
    )


def initialize_current(
    prior_directory, bank_path, *, actor_class, source_receipt, predecessor_snapshot
):
    """Explicit retained-v2 to current-v3 metadata-only initialization."""
    return _initialize(
        prior_directory,
        bank_path,
        actor_class=actor_class,
        source_receipt=source_receipt,
        predecessor_snapshot=predecessor_snapshot,
        adapt_v3=True,
    )


def _initialize(
    prior_directory, bank_path, *, actor_class, source_receipt, predecessor_snapshot, adapt_v3
):
    import torch

    from humanbrain.neuronal_splice_learner import _digest

    started, work = time.perf_counter(), {}
    _canonical(actor_class)
    expected_rule = "current" if adapt_v3 else "predecessor"
    if (
        source_receipt["schema"] != "integrated-local-learning-source-selection-v1"
        or source_receipt["rule"] != expected_rule
        or source_receipt["parent_bank_sha256"] != runtime.BANK_SHA
        or source_receipt["parent_config_sha256"] != runtime.CONFIG_SHA
    ):
        raise ValueError("initializer differs from the selected source-bound worker")
    runtime.check_origins(source_receipt)
    boundary, config = _authenticate(prior_directory, bank_path, predecessor_snapshot, work)
    raw = _read(Path(prior_directory) / PAYLOAD_NAME, PAYLOAD_SHA, work, limit=PAYLOAD_BYTES)
    if len(raw) != PAYLOAD_BYTES:
        raise ValueError("selected parent payload length changed")
    decoded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _add(work, "tensor_decodes")
    del raw
    if (
        not isinstance(decoded, dict)
        or set(decoded) != {"schema", "model", "memory"}
        or decoded["schema"] != "persistent-session-route-tensors-v1"
        or not isinstance(decoded["model"], dict)
        or not isinstance(decoded["memory"], dict)
    ):
        raise ValueError("unknown retained parent payload schema")
    source_tensors = {
        key: value for key, value in decoded["model"].items() if isinstance(value, torch.Tensor)
    }
    tape = list(_tensors(decoded["memory"]))
    if (
        len(source_tensors) != 29
        or len(tape) != 4
        or sum(v.numel() * v.element_size() for v in source_tensors.values()) != MODEL_BYTES
        or sum(v.numel() * v.element_size() for v in tape) != TAPE_BYTES
    ):
        raise ValueError("retained parent tensor geometry changed")
    numeric_sha = _digest(source_tensors, work)
    models, states, records, storages = {}, {}, {}, []
    for route in NEURAL_ROUTES:
        model, state, metadata, storage = _copy_actor(
            decoded, boundary["routes"]["online"], actor_class, work, adapt_v3=adapt_v3
        )
        if state.used != 1982 or any(storage & previous for previous in storages):
            raise ValueError("matched actors lost facts or share model, tape or derived storage")
        storages.append(storage)
        actual = model.state_dict()
        records[route] = {
            "model_sha256": _digest(actual, work),
            "memory_sha256": _digest(vars(state), work),
            "numeric_model_sha256": _digest(
                {key: value for key, value in actual.items() if isinstance(value, torch.Tensor)},
                work,
            ),
            "controller_rng_sha256": _digest(
                {"rng_state": model.reader.controller.rng_state}, work
            ),
            "baseline_mean": model.reader.controller.baseline_mean.tolist(),
            "baseline_count": model.reader.controller.baseline_count.tolist(),
            "experience": model._experience,
            "reader_experience": model.reader._active_experience,
            "views_sha256": model.reader._view_sha256,
            "self_wiring_sha256": model.reader._self_wiring_sha256,
            "metadata_adaptation": metadata,
        }
        _add(work, "baseline_scalar_reads", 6)
        if records[route]["numeric_model_sha256"] != numeric_sha:
            raise ValueError("metadata adaptation changed numerical model identity")
        models[route], states[route] = model, state
    _actors(models, work)
    if records["online"] != records["learning_disabled"]:
        raise ValueError("matched independent initial routes differ")
    baseline_record = copy.deepcopy(boundary["baseline"])
    baseline = _restore_baseline(baseline_record, document_limit=3484, symbol_limit=21382)
    if (
        len(baseline.documents) != 84
        or len(baseline.demonstrations) != 82
        or sum(len(document) + 1 for document in baseline.documents) != 21382
        or baseline.new_teacher_offers != {"SPLICE": 36, "STOP": 0, "reading": 36}
        or _baseline_record(baseline) != baseline_record
    ):
        raise ValueError("initializer changed the actual retained reference history")
    final, final_config = _authenticate(prior_directory, bank_path, predecessor_snapshot, work)
    if final != boundary or final_config != config:
        raise ValueError("parent changed during explicit initialization")
    _hash(Path(prior_directory) / PAYLOAD_NAME, PAYLOAD_SHA, PAYLOAD_BYTES, work)
    runtime.check_origins(source_receipt)
    receipt = {
        "schema": "integrated-local-learning-initialization-v1",
        "status": "initialized",
        "rule": expected_rule,
        "bank_sha256": runtime.BANK_SHA,
        "parent_boundary_sha256": runtime.BOUNDARY_SHA,
        "parent_config_sha256": runtime.CONFIG_SHA,
        "parent_payload_sha256": PAYLOAD_SHA,
        "parent_model_sha256": MODEL_SHA,
        "parent_numeric_model_sha256": numeric_sha,
        "parent_memory_sha256": MEMORY_SHA,
        "parent_selected_route": "online",
        "initial_routes": records,
        "baseline_record": baseline_record,
        "metadata_adaptation": adapt_v3,
        "work": work,
        "elapsed_seconds": time.perf_counter() - started,
        "source_offers": 0,
        "teacher_offers": 0,
        "task_executions": 0,
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
