"""Explicit new continuation from V7's banked clean online boundary.

No I/O, model construction or decoding occurs on import. This is not a changed
source V7 resume: the old store is never opened with relaxed source guards. One
authenticated local route payload is decoded only inside initialize().
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ".pytest-tmp/persistent-language-action-v7-run/"
BANK_SHA = "b11b0e0517ef7939cf624e68dfdce05cdd75bc75aa8a81e400ddfd7974fcc67e"
HEAD_SHA = "3bffe29937c840121fdd0e920bbfc3e54067389e0649d43530ce3df392d94656"
BOUNDARY_SHA = "b5f529a842d06e9a5d7236787a8d8c57b6eb6aa8abc90622bb7d13448276c92a"
CONFIG_SHA = "b582e7f3ab7d452a58c54021aa301c4ddc1de3faee4530d34f7a0c4349962687"
PAYLOAD_SHA = "1aaeb3b211872c3d6beb014ced2d7459f8b00520fb0bc462536fb5ec30cd38fc"
MODEL_SHA = "cc49bf4fd4b50a6115fc182548f00a5edb459b40e0750a51810304c92d1bb5c6"
MEMORY_SHA = "f4201afc7b2322994f7a9caafeffd02c570d7760ec6cb0f6f6613b3e737c825e"
SOURCE_REVISION = "a1d900a688d353fcc00e43b86ccb14b1547f3065"
SOURCES_SHA = "3cf9116c24cae30d959bc2830dd969c3c58101abe7a2e1e4e69a42f2f3cbb98c"
MANIFEST_SHA = "c2e1cb6237fdff468c6d413ce62c9dfbf514a607fb1a8af45bdde285c3f4b898"
PAYLOAD_NAME = "route-online-000000079.pt"
PAYLOAD_FILE_BYTES = 91_080_533
PRIOR_MODEL_BYTES = 87_760_560
TAPE_BYTES = 3_301_384
PRIOR_MODEL_TENSORS = 22
OCCUPIED_FACTS = 1614
EVENT_OWNERS = 398
EXPERIENCE = 281
READER_EXPERIENCE = 214
ROUTES = ("online", "learning_disabled")
JSON_LIMIT = 16 * 1024 * 1024


def _add(work, key, count=1):
    work[key] = work.get(key, 0) + count


def _path(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("continuation paths must be canonical relative POSIX")
    result = root.joinpath(*name.split("/"))
    if (not result.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (result, *result.parents) if p != root.parent)):
        raise ValueError("continuation evidence escaped its trusted local directory")
    return result


def _read(path, expected, work, *, limit=JSON_LIMIT):
    if path.stat().st_size > limit:
        raise ValueError("continuation metadata exceeds its explicit bound")
    raw = path.read_bytes()
    _add(work, "authenticated_read_bytes", len(raw))
    _add(work, "authenticated_read_passes")
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("banked continuation evidence bytes changed")
    return raw


def _hash(path, expected, size, work, *, journal_rows=None):
    digest, total, rows, last = hashlib.sha256(), 0, 0, b""
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            total += len(block)
            if journal_rows is not None:
                rows += block.count(b"\n")
                last = block[-1:]
    _add(work, "authenticated_read_bytes", total)
    _add(work, "authenticated_read_passes")
    if total != size or digest.hexdigest() != expected:
        raise ValueError("saved frontier file changed after its clean boundary")
    if journal_rows is not None and (rows != journal_rows or (total and last != b"\n")):
        raise ValueError("saved journal no longer ends at the exact complete-row frontier")


def _authenticate(directory, bank_path, work):
    bank_raw = _read(bank_path, BANK_SHA, work)
    bank = json.loads(bank_raw)
    checked = bank["checkpoint"]
    if (bank["status"] != "incomplete_resource_or_runtime_failure"
            or bank["source_revision"] != SOURCE_REVISION or bank["complete_run"]
            or not bank["saved_prefix_review"]["passed"] or checked["dirty"]
            or checked["outstanding_intents"] or checked["boundaries"] != 222
            or checked["head_sha256"] != HEAD_SHA):
        raise ValueError("selected V7 bank does not establish its clean partial boundary")
    evidence_root = directory.parent
    evidence = bank["evidence"]

    def metadata(name, expected=None):
        identity = evidence.get(PREFIX + name)
        if identity is None or (expected is not None and identity != expected):
            raise ValueError("required historical evidence is not bound by the result bank")
        return json.loads(_read(_path(evidence_root, name), identity, work))

    head = metadata("checkpoints/HEAD.json", HEAD_SHA)
    boundary = metadata("checkpoints/boundary-000000221.json", BOUNDARY_SHA)
    config = metadata("checkpoints/config.json", CONFIG_SHA)
    intent = metadata("checkpoints/intent-000000221.json", boundary["intent_sha256"])
    if (head != checked["head"] or head["sequence"] != 221
            or head["sha256"] != BOUNDARY_SHA
            or boundary["schema"] != "persistent-session-checkpoint-boundary-v1"
            or boundary["sequence"] != 221 or boundary["config_sha256"] != CONFIG_SHA
            or intent["sequence"] != 221 or intent["config_sha256"] != CONFIG_SHA
            or intent["previous_sha256"] != boundary["previous_sha256"]
            or boundary["host"] != checked["last_host"]
            or boundary["baseline"] != checked["last_baseline"]
            or boundary["routes"] != checked["last_routes"]):
        raise ValueError("safe HEAD, boundary, intent or parent-only records disagree")
    sources = metadata("sources.json", SOURCES_SHA)
    if (config["schema"] != "persistent-session-checkpoint-config-v1"
            or config["source_revision"] != SOURCE_REVISION
            or config["source_sha256"] != sources or len(sources) != 332
            or config["manifest_sha256"] != MANIFEST_SHA
            or bank["manifest_sha256"] != MANIFEST_SHA):
        raise ValueError("historical source/manifest bindings differ from banked V7")
    required = (
        "src/humanbrain/dendritic_sequence_learner.py",
        "src/humanbrain/dendritic_request_binding.py",
        "experiments/r4_integrated/persistent_session_checkpoint.py",
        "experiments/r4_integrated/persistent_language_action_v7.py",
    )
    if any(name not in sources for name in required):
        raise ValueError("historical source closure omits a selected actor or custody component")
    for name, identity in boundary["host"]["initial_bindings"].items():
        metadata(name, identity)

    # The reviewed bank authenticates historical lineage. Check exact present
    # checkpoint inventory and all small metadata, but do not decode or rehash
    # every superseded numerical payload to initialize one selected online state.
    expected_files = {name.removeprefix(PREFIX + "checkpoints/")
                      for name in evidence if name.startswith(PREFIX + "checkpoints/")}
    actual_files = set()
    for entry in directory.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("unexplained checkpoint directory or symlink")
        actual_files.add(entry.name)
    if actual_files != expected_files:
        raise ValueError("checkpoint inventory has missing or unbanked attempted work")
    for name in sorted(expected_files):
        if name.endswith(".json") and name not in {
                "HEAD.json", "config.json", "boundary-000000221.json", "intent-000000221.json"}:
            _read(_path(directory, name), evidence[PREFIX + "checkpoints/" + name], work)
    _add(work, "checkpoint_inventory_entries", len(actual_files))

    frontiers = boundary["frontiers"]
    if [row["kind"] for row in frontiers["journals"]] != ["attempts", "teachers", "ingests"]:
        raise ValueError("historical journals are not the exact three committed streams")
    for row, count in zip(frontiers["journals"], (185, 12, 36), strict=True):
        if (row["path"] != row["kind"] + ".jsonl" or row["rows"] != count
                or evidence.get(PREFIX + row["path"]) != row["sha256"]):
            raise ValueError("banked journal frontier differs")
        _hash(_path(evidence_root, row["path"]), row["sha256"], row["bytes"], work,
              journal_rows=count)
    execution_root = _path(evidence_root, frontiers["execution_root"])
    files, directories = set(), set()
    for entry in execution_root.rglob("*"):
        if entry.is_symlink():
            raise ValueError("actual execution inventory contains a symlink")
        name = entry.relative_to(evidence_root).as_posix()
        (files if entry.is_file() else directories).add(name)
    if (files != {row["path"] for row in frontiers["execution_files"]}
            or directories != set(frontiers["execution_directories"])):
        raise ValueError("actual execution evidence extends beyond the clean safe frontier")
    for row in frontiers["execution_files"]:
        if evidence.get(PREFIX + row["path"]) != row["sha256"]:
            raise ValueError("actual execution file not bound by the saved result bank")
        _hash(_path(evidence_root, row["path"]), row["sha256"], row["bytes"], work)
    _add(work, "execution_inventory_files", len(files))
    _add(work, "execution_inventory_directories", len(directories))
    selected = boundary["routes"]["online"]
    if (selected["payload"] != PAYLOAD_NAME or selected["sha256"] != PAYLOAD_SHA
            or selected["bytes"] != PAYLOAD_FILE_BYTES or selected["model_sha256"] != MODEL_SHA
            or selected["memory_sha256"] != MEMORY_SHA or selected["experience"] != EXPERIENCE
            or selected["reader_experience"] != READER_EXPERIENCE
            or selected["mode"] != "learned" or selected["seed"] != 733
            or selected["actor_class"] != (
                "humanbrain.dendritic_sequence_learner.DendriticSequenceSpliceLearner")
            or evidence.get(PREFIX + "checkpoints/" + PAYLOAD_NAME) != PAYLOAD_SHA):
        raise ValueError("selected online numerical continuation is not the banked route")
    return bank, boundary, config


def initialize(checkpoint_directory, bank_path):
    """Return two independent new actors, exact tape copies and raw historical host.

    The future caller owns a new session cursor, budget, source inventory and
    route dictionary. It must not run an actor unless this entire call returns.
    Any initialization exception abandons both newly constructed actors.
    """
    started, work = time.perf_counter(), {}
    directory, bank_path = Path(checkpoint_directory).resolve(), Path(bank_path).resolve()
    if directory != (ROOT / PREFIX / "checkpoints").resolve():
        raise ValueError("this explicit initializer selects the retained local V7 checkpoint")
    _, boundary, config = _authenticate(directory, bank_path, work)
    authentication_seconds = time.perf_counter() - started
    # Preserve the exact verified bytes in memory across restricted decoding;
    # there is no mutable path reopen between hashing and torch.load.
    raw = _read(_path(directory, PAYLOAD_NAME), PAYLOAD_SHA, work, limit=PAYLOAD_FILE_BYTES)
    if len(raw) != PAYLOAD_FILE_BYTES:
        raise ValueError("selected serialized route size changed")

    import torch

    from experiments.r4_integrated.persistent_session_checkpoint import (
        _baseline_record,
        _restore_baseline,
    )
    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.recurrent_content_learner import RecurrentContentSpliceLearner

    decode_started = time.perf_counter()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _add(work, "torch_load_calls")
    del raw
    decode_seconds = time.perf_counter() - decode_started
    if (not isinstance(payload, dict) or set(payload) != {"schema", "model", "memory"}
            or payload["schema"] != "persistent-session-route-tensors-v1"
            or not isinstance(payload["model"], dict) or not isinstance(payload["memory"], dict)
            or set(payload["memory"]) != {"keys", "sensory", "successor", "write_index"}
            or _digest(payload["model"], work) != MODEL_SHA
            or _digest(payload["memory"], work) != MEMORY_SHA):
        raise ValueError("decoded selected route differs from its authenticated numerical hashes")

    def tensors(mapping):
        return [value for value in mapping.values() if isinstance(value, torch.Tensor)]

    def tensor_bytes(values):
        return sum(value.numel() * value.element_size() for value in values)

    old_tensors = tensors(payload["model"])
    tape_tensors = tensors(payload["memory"])
    if (len(old_tensors) != PRIOR_MODEL_TENSORS or tensor_bytes(old_tensors) != PRIOR_MODEL_BYTES
            or len(tape_tensors) != 4 or tensor_bytes(tape_tensors) != TAPE_BYTES):
        raise ValueError("selected historical model/tape tensor geometry changed")
    source_storage = {v.untyped_storage().data_ptr()
                      for v in old_tensors + tape_tensors if v.numel()}
    models, states, route_work, route_storage, route_records = {}, {}, {}, {}, {}
    construction_started = time.perf_counter()
    for route in ROUTES:
        model = RecurrentContentSpliceLearner(mode="learned", seed=733)
        local = {"combined_model_constructions": 1,
                 "inherited_empty_512_event_bank_bytes": 34_892_296,
                 "retained_1024_event_bank_construction_bytes": 69_784_584,
                 "retained_dendritic_bank_construction_bytes": 4_095_552,
                 "controller_weight_zero_bytes": 24_623_616}
        controller_before = _digest(model.reader.controller.state_dict(), local)
        model.load_v7_state_dict(payload["model"], work=local)
        state = model.reader.memory.restore_state(payload["memory"])
        local["tape_restore_calls"] = 1
        local["tape_restore_clone_bytes"] = TAPE_BYTES
        local["tape_restore_uninstrumented_validation_passes"] = 1
        model._experience = EXPERIENCE
        model.reader._active_experience = READER_EXPERIENCE
        actual = model.state_dict()
        if (state.used != OCCUPIED_FACTS or int(model.reader.assemblies.used) != EVENT_OWNERS
                or _digest(vars(state), local) != MEMORY_SHA
                or _digest(model.reader.controller.state_dict(), local) != controller_before
                or any(not torch.equal(actual[name], value)
                       for name, value in payload["model"].items()
                       if isinstance(value, torch.Tensor))):
            raise ValueError("new actor changed the old prior, tape or pristine controller")
        views, wirings = model.reader._candidate_views, model.reader._self_wirings
        if (len(views) != EVENT_OWNERS or len(wirings) != EVENT_OWNERS
                or local.get("factorized_candidate_builds") != EVENT_OWNERS
                or local.get("recognition_self_builds") != EVENT_OWNERS):
            raise ValueError("strict continuation must derive every existing owner exactly once")
        values = tensors(actual) + tensors(vars(state))
        values += [t for view in views for t in (view.indices, view.values, view.flags)]
        values += [t for wiring in wirings for t in (wiring.indices, wiring.values)]
        storage = {v.untyped_storage().data_ptr() for v in values if v.numel()}
        if storage & source_storage or any(storage & prior for prior in route_storage.values()):
            raise ValueError("new routes share prior, live numerical, tape or derived storage")
        local["retained_model_copy_bytes"] = PRIOR_MODEL_BYTES
        local["pristine_controller_strict_copy_bytes"] = model.reader.controller.tensor_bytes
        local["model_construction_named_retained_bytes"] = tensor_bytes(tensors(actual))
        local["tensor_equality_values"] = sum(t.numel() for t in old_tensors)
        local["storage_identity_checks"] = len(values)
        route_records[route] = {
            "model_sha256": _digest(actual, local), "memory_sha256": MEMORY_SHA,
            "controller_sha256": controller_before,
            "model_tensor_count": len(tensors(actual)),
            "model_tensor_bytes": tensor_bytes(tensors(actual)), "tape_tensor_bytes": TAPE_BYTES,
            "views_sha256": model.reader._view_sha256,
            "self_wiring_sha256": model.reader._self_wiring_sha256,
            "derived_view_bytes": sum(v.indices.numel() * 4 + v.values.numel() * 4
                                      + v.flags.numel() for v in views),
            "derived_self_wiring_bytes": sum(v.indices.numel() * 4 + v.values.numel() * 8
                                             for v in wirings),
        }
        models[route], states[route], route_work[route] = model, state, local
        route_storage[route] = storage
        del actual, values
    if route_records[ROUTES[0]] != route_records[ROUTES[1]]:
        raise ValueError("independent online/control initial numerical lineages differ")
    construction_seconds = time.perf_counter() - construction_started
    baseline_record = copy.deepcopy(boundary["baseline"])
    baseline = _restore_baseline(baseline_record)
    if (len(baseline.documents) != 36 or len(baseline.demonstrations) != 48
            or sum(len(doc) + 1 for doc in baseline.documents) != OCCUPIED_FACTS
            or _baseline_record(baseline) != baseline_record):
        raise ValueError("simple reference restoration changed its own actual history")
    host = copy.deepcopy(boundary["host"])
    _add(work, "baseline_record_restorations")
    _add(work, "baseline_object_constructions")
    _add(work, "baseline_document_records", len(baseline.documents))
    _add(work, "baseline_demonstration_records", len(baseline.demonstrations))
    # A final complete frontier authentication detects intervening new attempted
    # work. The one selected payload has already been copied from verified bytes.
    final_started = time.perf_counter()
    _, final_boundary, _ = _authenticate(directory, bank_path, work)
    if final_boundary != boundary:
        raise ValueError("historical safe checkpoint changed during new initialization")
    _hash(_path(directory, PAYLOAD_NAME), PAYLOAD_SHA, PAYLOAD_FILE_BYTES, work)
    final_authentication_seconds = time.perf_counter() - final_started
    receipt = {
        "schema": "persistent-language-action-initial-custody-v8", "status": "initialized",
        "bank_path": str(bank_path), "bank_sha256": BANK_SHA,
        "prior_path": str(directory / PAYLOAD_NAME), "prior_sha256": PAYLOAD_SHA,
        "prior_head_sha256": HEAD_SHA, "prior_boundary_sha256": BOUNDARY_SHA,
        "prior_boundary_sequence": 221, "prior_source_revision": SOURCE_REVISION,
        "prior_sources_sha256": SOURCES_SHA, "prior_manifest_sha256": MANIFEST_SHA,
        "historical_source_sha256": config["source_sha256"],
        "historical_source_scope": (
            "bank-authenticated original source inventory; changed checkout is not V7 resume"),
        "prior_attempts_sha256": boundary["frontiers"]["journals"][0]["sha256"],
        "prior_selected_route": "online", "neural_route_parent": dict.fromkeys(ROUTES, "online"),
        "source_model_sha256": MODEL_SHA, "source_memory_sha256": MEMORY_SHA,
        "loaded_model_tensor_count": PRIOR_MODEL_TENSORS, "loaded_tape_tensor_count": 4,
        "loaded_tensor_bytes": PRIOR_MODEL_BYTES + TAPE_BYTES,
        "serialized_payload_bytes": PAYLOAD_FILE_BYTES, "torch_load_calls": 1,
        "old_model_constructions": 0, "new_combined_model_constructions": 2,
        "initial_experience_per_route": EXPERIENCE,
        "initial_reader_experience_per_route": READER_EXPERIENCE,
        "initial_factual_slots_per_route": OCCUPIED_FACTS,
        "initial_event_owners_per_route": EVENT_OWNERS,
        "initial_routes": route_records, "work": work, "route_work": route_work,
        "baseline_documents": 36, "baseline_demonstrations": 48,
        "baseline_new_teacher_offers": dict(baseline.new_teacher_offers),
        "baseline_offer_counter_scope": "preserved V7 session counters; not new offers",
        "baseline_raw_history_bytes": sum(len(d) for d in baseline.documents)
        + sum(sum(len(v) for v in row.values() if isinstance(v, bytes))
              for row in baseline.demonstrations),
        "reference_prediction_calls": 0, "reference_teacher_calls": 0,
        "source_ingestions": 0, "source_observations": 0, "answer_teacher_calls": 0,
        "neural_prediction_calls": 0, "execution_children": 0,
        "new_session_cursor": "caller-owned; saved V7 cursor returned as history only",
        "prior_cursor": copy.deepcopy(host["cursor"]),
        "prior_route_counters": copy.deepcopy(host["counters"]),
        "historical_metadata_authentication_passes": 2,
        "historical_frontier_authentication_passes": 2,
        "selected_payload_authentication_passes": 2,
        "unrun_v7_attempts_replayed": 0,
        "authentication_seconds": authentication_seconds,
        "decode_seconds": decode_seconds, "construction_seconds": construction_seconds,
        "final_authentication_seconds": final_authentication_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "accounting_scope": (
            "named tensors, copies, strict-load work and all evidence I/O; constructor/legacy "
            "validation internals are timed, not claimed as exhaustive scalar operation counts"),
    }
    return {"models": models, "states": states, "baseline": baseline,
            "baseline_record": baseline_record, "prior_host": host,
            "prior_frontiers": copy.deepcopy(boundary["frontiers"]), "receipt": receipt}
