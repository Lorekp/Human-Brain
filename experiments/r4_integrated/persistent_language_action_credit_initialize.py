"""Explicit retained V8 ONLINE initialization, without import-time model work.

The selected launch supplies its source-bound canonical actor class. The old
worker strictly loads v1; the current worker adapts exactly three metadata
records. Neither entry resumes V8 under changed sources or replays a lesson.
"""

from __future__ import annotations

import copy
import io
import json
import sys
import time
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_v8_initialize import (
    _add,
    _hash,
    _path,
    _read,
)

PREFIX = ".pytest-tmp/persistent-language-action-v8-corrected-run/"
BANK_SHA = "87f0c8e39cbbaaed0558d1affb1ca87f52177f4cb6aee0650b74cfcbf06f392a"
HEAD_SHA = "98cb75894fdb2b5f9b4886b0e44b95e03be5ccd1d644ee6e409d4a0e360e2993"
BOUNDARY_SHA = "e2acf10914cab2da891352e97eed4dc1e665e02e54e62508c58f858b3c94fdc4"
CONFIG_SHA = "8b8b992c5135b86f1351fcb62611b841dec6b09f8054b424088e654aee7caf14"
SOURCE_REVISION = "8e8df91d7e7c6889dff15f97d9fec379ebff71bb"
SOURCES_SHA = "ae23542c22d89a1e1bbbd4babb3739928a550a60da89b6bc50fbef06f92148ac"
MANIFEST_SHA = "66b0bd96be3b9e97c4eb5f9812c2b5977f294fc5900d55109445c0a303e522d0"
PAYLOAD_NAME = "route-online-000000262.pt"
PAYLOAD_SHA = "f49a6496f91bca6dc431ae221034dfda9e897d0f2476ae18d2bb4b65f8617592"
MODEL_SHA = "376c5a6223b727020a56bf09d62b7aed5ead069431fd989de5faf7f94adce853"
MEMORY_SHA = "cdd1f887f80ec2e8ae6ee38ae8a15786d9155160cf6beb18def635a2a66da73e"
PAYLOAD_BYTES = 115_712_128
MODEL_BYTES = 112_389_280
TAPE_BYTES = 3_301_384
EXPERIENCE, READER_EXPERIENCE = 325, 278
ROUTES = ("online", "learning_disabled")
ACTOR_MODULE = "humanbrain.recurrent_content_learner"
ACTOR_NAME = "RecurrentContentSpliceLearner"
CONTROLLER_MODULE = "humanbrain.recurrent_content_controller"
OLD_SOURCES = {
    "src/humanbrain/recurrent_content_controller.py":
        "e2fb738b86d820db85f3e8b6190a93a9c85fbc704bbfe897d8ac7c5a335f56b5",
    "src/humanbrain/recurrent_content_learner.py":
        "e89eb5ccfa44eda3b9a8306bc8603b2aeab11d78630e47de4cc2d8f1e22c9ece",
}


def _authenticate(directory, bank_path, work):
    bank = json.loads(_read(bank_path, BANK_SHA, work))
    checked, evidence = bank["checkpoint"], bank["evidence"]
    if (bank["status"] != "completed_and_independently_reviewed"
            or not bank["complete_run"] or bank["source_revision"] != SOURCE_REVISION
            or checked["dirty"] or checked["outstanding_intents"]
            or checked["boundaries"] != 265 or checked["head_sha256"] != HEAD_SHA):
        raise ValueError("selected V8 bank does not establish its completed clean boundary")
    root = directory.parent

    def metadata(name, expected=None):
        identity = evidence.get(PREFIX + name)
        if identity is None or (expected is not None and identity != expected):
            raise ValueError("historical metadata is not bound by the selected V8 bank")
        return json.loads(_read(_path(root, name), identity, work))

    head = metadata("checkpoints/HEAD.json", HEAD_SHA)
    boundary = metadata("checkpoints/boundary-000000264.json", BOUNDARY_SHA)
    config = metadata("checkpoints/config.json", CONFIG_SHA)
    intent = metadata("checkpoints/intent-000000264.json", boundary["intent_sha256"])
    if (head != checked["head"] or head["sequence"] != 264
            or head["sha256"] != BOUNDARY_SHA
            or boundary["schema"] != "persistent-session-checkpoint-boundary-v1"
            or boundary["sequence"] != 264 or boundary["config_sha256"] != CONFIG_SHA
            or intent["sequence"] != 264 or intent["config_sha256"] != CONFIG_SHA
            or intent["previous_sha256"] != boundary["previous_sha256"]
            or boundary["host"] != checked["last_host"]
            or boundary["baseline"] != checked["last_baseline"]
            or boundary["routes"] != checked["last_routes"]):
        raise ValueError("V8 HEAD, boundary, intent and bank disagree")
    sources = metadata("sources.json", SOURCES_SHA)
    if (config["schema"] != "persistent-session-checkpoint-config-v1"
            or config["source_revision"] != SOURCE_REVISION
            or config["source_sha256"] != sources
            or config["manifest_sha256"] != MANIFEST_SHA
            or bank["manifest_sha256"] != MANIFEST_SHA
            or any(sources.get(name) != sha for name, sha in OLD_SOURCES.items())):
        raise ValueError("V8 source or manifest identity changed")
    metadata("manifest.json", MANIFEST_SHA)
    metadata("checkpoints/manifest.json", MANIFEST_SHA)
    for name, identity in boundary["host"]["initial_bindings"].items():
        metadata(name, identity)
    expected = {name.removeprefix(PREFIX + "checkpoints/")
                for name in evidence if name.startswith(PREFIX + "checkpoints/")}
    actual = set()
    for entry in directory.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("unexplained V8 checkpoint directory or symlink")
        actual.add(entry.name)
    if actual != expected:
        raise ValueError("V8 checkpoint inventory has missing or unbanked attempted work")
    for name in sorted(expected):
        if name.endswith(".json") and name not in {
                "HEAD.json", "config.json", "manifest.json",
                "boundary-000000264.json", "intent-000000264.json"}:
            _read(_path(directory, name), evidence[PREFIX + "checkpoints/" + name], work)
    _add(work, "checkpoint_inventory_entries", len(actual))
    frontiers = boundary["frontiers"]
    if [row["kind"] for row in frontiers["journals"]] != ["attempts", "teachers", "ingests"]:
        raise ValueError("V8 journals are not its three complete streams")
    for row, count in zip(frontiers["journals"], (192, 6, 72), strict=True):
        if (row["path"] != row["kind"] + ".jsonl" or row["rows"] != count
                or evidence.get(PREFIX + row["path"]) != row["sha256"]):
            raise ValueError("V8 journal frontier differs from its bank")
        _hash(_path(root, row["path"]), row["sha256"], row["bytes"], work, journal_rows=count)
    execution = _path(root, frontiers["execution_root"])
    files, directories = set(), set()
    for entry in execution.rglob("*"):
        if entry.is_symlink():
            raise ValueError("historical execution inventory contains a symlink")
        name = entry.relative_to(root).as_posix()
        (files if entry.is_file() else directories).add(name)
    if (files != {row["path"] for row in frontiers["execution_files"]}
            or directories != set(frontiers["execution_directories"])):
        raise ValueError("V8 execution inventory extends beyond its safe boundary")
    for row in frontiers["execution_files"]:
        if evidence.get(PREFIX + row["path"]) != row["sha256"]:
            raise ValueError("V8 execution file is not banked")
        _hash(_path(root, row["path"]), row["sha256"], row["bytes"], work)
    selected = boundary["routes"]["online"]
    if selected != {
            "payload": PAYLOAD_NAME, "sha256": PAYLOAD_SHA, "bytes": PAYLOAD_BYTES,
            "model_sha256": MODEL_SHA, "memory_sha256": MEMORY_SHA,
            "experience": EXPERIENCE, "reader_experience": READER_EXPERIENCE,
            "mode": "learned", "seed": 733, "actor_class": ACTOR_MODULE + "." + ACTOR_NAME}:
        raise ValueError("selected V8 ONLINE continuation changed")
    if evidence.get(PREFIX + "checkpoints/" + PAYLOAD_NAME) != PAYLOAD_SHA:
        raise ValueError("selected V8 tensor payload is not banked")
    return boundary, config


def _mapping(state, model, *, adapt_v2):
    """Exact metadata-only migration specified by the reviewed V8/v2 source diff."""
    target = model.state_dict()
    if set(state) != set(target):
        raise ValueError("retained V8 state keys differ from the selected actor")
    names = ("_extra_state", "reader._extra_state", "reader.controller._extra_state")
    expected = {name: copy.deepcopy(target[name]) for name in names}
    suffix = "v2" if adapt_v2 else "v1"
    schemas = ("recurrent-content-splice-learner-", "recurrent-content-memory-learner-",
               "recurrent-content-controller-")
    if any(expected[name]["schema"] != schema + suffix
           for name, schema in zip(names, schemas, strict=True)):
        raise ValueError("initializer entry does not match the selected actor algorithm")
    if adapt_v2:
        expected[names[0]].update(schema=schemas[0] + "v1", reader=schemas[1] + "v1")
        reader = expected[names[1]]
        reader.update(schema=schemas[1] + "v1",
                      publication="scan/free/teacher eligibility held; validated joint controller copies",
                      initialization="all V7 numeric values held; only new controller starts afresh")
        if reader.pop("local_credit") != (
                "causal-observation-credit-v1; direct literal CE; upstream own eligibility"):
            raise ValueError("unknown v2 reader metadata adaptation")
        controller = expected[names[2]]
        controller.update(schema=schemas[2] + "v1",
                          local_rule="own-softmax-eligibility-held-trajectory-v1")
        for key in ("upstream_objective", "literal_objective", "public_consequence",
                    "update_norm_bound", "update_norm_tolerance"):
            controller.pop(key)
    if any(state[name] != expected[name] for name in names):
        raise ValueError("historical metadata differs beyond the declared V8 adaptation")
    mapped = dict(state)
    if adapt_v2:
        mapped.update({name: copy.deepcopy(target[name]) for name in names})
    return mapped, {name: {"before": copy.deepcopy(state[name]),
                           "after": copy.deepcopy(mapped[name])} for name in names}


def initialize_v8(checkpoint_directory, bank_path, *, actor_class):
    """Strict original-credit continuation in the source-bound old worker."""
    return _initialize(checkpoint_directory, bank_path, actor_class=actor_class, adapt_v2=False)


def initialize_credit_v2(checkpoint_directory, bank_path, *, actor_class):
    """Explicit metadata-only v2 adaptation; all historical numerical state survives."""
    return _initialize(checkpoint_directory, bank_path, actor_class=actor_class, adapt_v2=True)


def _initialize(checkpoint_directory, bank_path, *, actor_class, adapt_v2):
    started, work = time.perf_counter(), {}
    directory, bank_path = Path(checkpoint_directory).resolve(), Path(bank_path).resolve()
    if (actor_class.__module__ != ACTOR_MODULE or actor_class.__qualname__ != ACTOR_NAME
            or getattr(sys.modules.get(ACTOR_MODULE), ACTOR_NAME, None) is not actor_class):
        raise ValueError("initializer requires the source-bound canonical recurrent actor")
    boundary, config = _authenticate(directory, bank_path, work)
    raw = _read(_path(directory, PAYLOAD_NAME), PAYLOAD_SHA, work, limit=PAYLOAD_BYTES)
    if len(raw) != PAYLOAD_BYTES:
        raise ValueError("selected V8 serialized payload length changed")
    import torch

    from experiments.r4_integrated.persistent_session_checkpoint import (
        _baseline_record,
        _restore_baseline,
    )
    from humanbrain.neuronal_splice_learner import _digest

    decoded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _add(work, "torch_load_calls")
    del raw
    if (not isinstance(decoded, dict) or set(decoded) != {"schema", "model", "memory"}
            or decoded["schema"] != "persistent-session-route-tensors-v1"
            or not isinstance(decoded["model"], dict) or not isinstance(decoded["memory"], dict)
            or set(decoded["memory"]) != {"keys", "sensory", "successor", "write_index"}
            or _digest(decoded["model"], work) != MODEL_SHA
            or _digest(decoded["memory"], work) != MEMORY_SHA):
        raise ValueError("decoded V8 state differs from its authenticated numerical hashes")

    def tensors(mapping):
        return [value for value in mapping.values() if isinstance(value, torch.Tensor)]

    def size(values):
        return sum(value.numel() * value.element_size() for value in values)

    originals, tape = tensors(decoded["model"]), tensors(decoded["memory"])
    if (len(originals) != 29 or size(originals) != MODEL_BYTES
            or len(tape) != 4 or size(tape) != TAPE_BYTES):
        raise ValueError("V8 model/tape tensor geometry changed")
    source_storage = {v.untyped_storage().data_ptr() for v in originals + tape if v.numel()}
    numeric_sha = _digest({key: value for key, value in decoded["model"].items()
                           if isinstance(value, torch.Tensor)}, work)
    models, states, records, storages = {}, {}, {}, []
    for route in ROUTES:
        model = actor_class(mode="learned", seed=733)
        _add(work, "combined_model_constructions")
        mapped, metadata = _mapping(decoded["model"], model, adapt_v2=adapt_v2)
        model.load_state_dict(mapped, strict=True, assign=False, work=work)
        state = model.reader.memory.restore_state(decoded["memory"])
        model._experience, model.reader._active_experience = EXPERIENCE, READER_EXPERIENCE
        actual = model.state_dict()
        if (state.used != 1982 or _digest(vars(state), work) != MEMORY_SHA
                or any(not torch.equal(actual[name], value)
                       for name, value in decoded["model"].items()
                       if isinstance(value, torch.Tensor))
                or _digest(actual, work) != _digest(mapped, work)):
            raise ValueError("initialization changed retained weights, RNG, baselines or factual bytes")
        model._require_idle()
        model._check(work)
        views, wirings = model.reader._candidate_views, model.reader._self_wirings
        owners = int(model.reader.assemblies.used)
        if len(views) != owners or len(wirings) != owners:
            raise ValueError("strict initialization omitted existing derived owners")
        values = tensors(actual) + tensors(vars(state))
        values += [t for view in views for t in (view.indices, view.values, view.flags)]
        values += [t for wiring in wirings for t in (wiring.indices, wiring.values)]
        storage = {v.untyped_storage().data_ptr() for v in values if v.numel()}
        if storage & source_storage or any(storage & prior for prior in storages):
            raise ValueError("initialized routes share decoded, live or derived tensor storage")
        storages.append(storage)
        records[route] = {
            "model_sha256": _digest(actual, work), "memory_sha256": MEMORY_SHA,
            "numeric_model_sha256": _digest({key: value for key, value in actual.items()
                                              if isinstance(value, torch.Tensor)}, work),
            "controller_rng_sha256": _digest({"rng_state": model.reader.controller.rng_state}, work),
            "baseline_mean": model.reader.controller.baseline_mean.tolist(),
            "baseline_count": model.reader.controller.baseline_count.tolist(),
            "views_sha256": model.reader._view_sha256,
            "self_wiring_sha256": model.reader._self_wiring_sha256,
            "event_owners": owners, "metadata_adaptation": metadata,
        }
        if records[route]["numeric_model_sha256"] != numeric_sha:
            raise ValueError("initialized numerical model identity changed")
        models[route], states[route] = model, state
        _add(work, "retained_model_copy_bytes", MODEL_BYTES)
        _add(work, "tape_restore_clone_bytes", TAPE_BYTES)
        _add(work, "tensor_equality_values", sum(value.numel() for value in originals))
        _add(work, "storage_identity_checks", len(values))
        _add(work, "baseline_scalar_reads", 6)
    if records["online"] != records["learning_disabled"]:
        raise ValueError("independent initialized neural routes differ")
    baseline_record = copy.deepcopy(boundary["baseline"])
    baseline = _restore_baseline(baseline_record, document_limit=3484, symbol_limit=11498)
    if (len(baseline.documents) != 60 or len(baseline.demonstrations) != 49
            or sum(len(doc) + 1 for doc in baseline.documents) != 11498
            or _baseline_record(baseline) != baseline_record):
        raise ValueError("simple reference restoration changed its actual V8 history")
    final_boundary, final_config = _authenticate(directory, bank_path, work)
    if final_boundary != boundary or final_config != config:
        raise ValueError("V8 safe frontier changed during initialization")
    _hash(_path(directory, PAYLOAD_NAME), PAYLOAD_SHA, PAYLOAD_BYTES, work)
    receipt = {
        "schema": "persistent-language-action-credit-initialization-v1", "status": "initialized",
        "algorithm": "causal-observation-credit-v1" if adapt_v2 else "v8-original-credit",
        "bank_path": str(bank_path), "bank_sha256": BANK_SHA,
        "prior_path": str(directory / PAYLOAD_NAME), "prior_sha256": PAYLOAD_SHA,
        "prior_head_sha256": HEAD_SHA, "prior_boundary_sha256": BOUNDARY_SHA,
        "prior_boundary_sequence": 264, "prior_source_revision": SOURCE_REVISION,
        "historical_source_sha256": config["source_sha256"],
        "prior_selected_route": "online", "neural_route_parent": dict.fromkeys(ROUTES, "online"),
        "source_model_sha256": MODEL_SHA, "source_numeric_model_sha256": numeric_sha,
        "source_memory_sha256": MEMORY_SHA, "initial_routes": records,
        "initial_experience_per_route": EXPERIENCE,
        "initial_reader_experience_per_route": READER_EXPERIENCE,
        "initial_factual_slots_per_route": 1982,
        "loaded_model_tensor_count": 29, "loaded_tape_tensor_count": 4,
        "loaded_tensor_bytes": MODEL_BYTES + TAPE_BYTES, "serialized_payload_bytes": PAYLOAD_BYTES,
        "baseline_documents": 60, "baseline_demonstrations": 49,
        "baseline_new_teacher_offers": dict(baseline.new_teacher_offers),
        "source_ingestions": 0, "source_observations": 0, "answer_teacher_calls": 0,
        "neural_prediction_calls": 0, "execution_children": 0,
        "prior_cursor": copy.deepcopy(boundary["host"]["cursor"]),
        "historical_frontier_authentication_passes": 2,
        "selected_payload_authentication_passes": 2,
        "work": work, "elapsed_seconds": time.perf_counter() - started,
        "scope": "explicit new session; complete numeric/history preservation; no historical replay",
    }
    return {"models": models, "states": states, "baseline": baseline,
            "baseline_record": baseline_record, "prior_host": copy.deepcopy(boundary["host"]),
            "prior_frontiers": copy.deepcopy(boundary["frontiers"]), "receipt": receipt}
