"""Authenticated V5 continuation with zero recognition and lossless capacity growth.

Import performs no I/O or neuronal construction. The declared complete run owns
the single checkpoint deserialization and both independent persistent actors.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_initial_v2 import sha
from experiments.r4_integrated.persistent_language_action_initial_v5 import (
    METADATA_NAMES as V4_METADATA_NAMES,
)
from experiments.r4_integrated.persistent_language_action_initial_v5 import (
    NUMERIC_NAMES as V4_NUMERIC_NAMES,
)

BANK_SHA = "c60ca50a1800f5431e1a20c8f1aa9c6bcd6e3c82e45ebfa72977735e52728d42"
CHECKPOINT_SHA = "a141230e836d6da83e9914ae52f665d39d0d3e59089a60203b7a963ecd59ea51"
REPORT_SHA = "05307278505cbb3337c44e6bbc8d5168f898446c9f5f69c1486d233dfca8ad7c"
SOURCES_SHA = "8ae123f0e6697d80764435e8363e397e0bdb6b13f27420cc87e8e7903b899f3e"
ATTEMPTS_SHA = "9139824c7897f15e3a3aed71a7692f7a15c28ad18fa4d82d48072dee5981e638"
SOURCE_REVISION = "c2b3f92e8a702e3908deb6332c1a9c27bcc14e6f"
PREFIX = ".pytest-tmp/persistent-language-action-v5-run/"
ROUTES = ("online", "learning_disabled")
PRIOR_EXPERIENCE = 238
PRIOR_READER_EXPERIENCE = 152
PRIOR_MODEL_BYTES = 48_764_528
PRIOR_TAPE_BYTES = 1_650_696
PRIOR_PAYLOAD_BYTES = 100_830_448
NEW_MODEL_BYTES = 83_665_008
NEW_TAPE_BYTES = 3_301_384
NUMERIC_NAMES = (*V4_NUMERIC_NAMES, "reader.router.theta")
METADATA_NAMES = (*V4_METADATA_NAMES, "reader.router._extra_state")
GROWN_NAMES = ("reader.assemblies.prototypes", "reader.assemblies.outgoing")
ADAPTED_METADATA = ("_extra_state", "reader._extra_state", "reader.assemblies._extra_state")


def _read_bound(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("recognition continuation evidence identity changed")
    return json.loads(raw)


def initialize(prior_path, bank_path, *, seed=733):
    """Copy the complete online state twice; no old model or lesson is replayed."""
    started = time.perf_counter()
    prior_path, bank_path = Path(prior_path).resolve(), Path(bank_path).resolve()
    if type(seed) is not int or seed != 733:
        raise ValueError("recognition continuation retains seed733")
    report_path, sources_path = prior_path.parent / "report.json", prior_path.parent / "sources.json"
    bank = _read_bound(bank_path, BANK_SHA)
    report = _read_bound(report_path, REPORT_SHA)
    sources = _read_bound(sources_path, SOURCES_SHA)
    if (bank["status"] != "completed_and_independently_reviewed"
            or bank["source_revision"] != SOURCE_REVISION
            or any(bank["evidence"][PREFIX + name] != expected for name, expected in (
                ("final-state.pt", CHECKPOINT_SHA), ("report.json", REPORT_SHA),
                ("sources.json", SOURCES_SHA), ("attempts.jsonl", ATTEMPTS_SHA)))
            or report["schema"] != "persistent-language-action-report-v5"
            or report["status"] != "completed"
            or report["final_checkpoint"]["sha256"] != CHECKPOINT_SHA
            or report["final_checkpoint"]["tensor_bytes"] != PRIOR_PAYLOAD_BYTES
            or report["final_memory_slots"]["online"] != 864
            or report["final_assembly_occupied"]["online"] != 398):
        raise ValueError("bank/report do not describe the complete selected V5 online state")
    required_sources = (
        "experiments/r4_integrated/persistent_language_action_initial_v5.py",
        "experiments/r4_integrated/persistent_language_action_v5.py",
        "src/humanbrain/factorized_sequence_router.py",
        "src/humanbrain/factorized_sequence_memory_learner.py",
        "src/humanbrain/factorized_sequence_splice_learner.py",
    )
    historical_sources = {name.replace("\\", "/"): value for name, value in sources.items()}
    if any(name not in historical_sources for name in required_sources):
        raise ValueError("historical source closure omits a required V5 component")
    if sha(prior_path) != CHECKPOINT_SHA:
        raise ValueError("recognition continuation checkpoint identity changed")

    import torch

    from humanbrain.neuronal_sequence_memory import TapeState
    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.recognition_sequence_splice_learner import RecognitionSequenceSpliceLearner

    with prior_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA:
            raise ValueError("checkpoint changed before its single deserialization")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [tensor for item in value.values() for tensor in tensors(item)]
        return []

    def tensor_bytes(value):
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors(value))

    if (not isinstance(payload, dict) or set(payload) != {"schema", "routes"}
            or payload["schema"] != "persistent-language-action-state-v5"
            or set(payload["routes"]) != set(ROUTES)
            or len(tensors(payload)) != 40 or tensor_bytes(payload) != PRIOR_PAYLOAD_BYTES):
        raise ValueError("unexpected complete V5 checkpoint container")
    source = payload["routes"]["online"]
    del payload
    if (not isinstance(source, dict)
            or set(source) != {"model", "memory", "experience", "reader_experience"}
            or type(source["experience"]) is not int or source["experience"] != PRIOR_EXPERIENCE
            or type(source["reader_experience"]) is not int
            or source["reader_experience"] != PRIOR_READER_EXPERIENCE
            or set(source["model"]) != set(NUMERIC_NAMES) | set(METADATA_NAMES)
            or set(source["memory"]) != {"keys", "sensory", "successor", "write_index"}
            or not all(isinstance(source["model"][name], torch.Tensor) for name in NUMERIC_NAMES)
            or tensor_bytes(source["model"]) != PRIOR_MODEL_BYTES
            or tensor_bytes(source["memory"]) != PRIOR_TAPE_BYTES
            or _digest(source["model"]) != report["final_hashes"]["online"]["model"]
            or _digest(source["memory"]) != report["final_hashes"]["online"]["memory"]):
        raise ValueError("selected prior state is not the complete authenticated online state")
    source_tape = TapeState(**source["memory"])
    models, states, work_by_route, views_by_route = {}, {}, {}, {}
    model_sizes, tape_sizes, storage_by_route = {}, {}, {}
    old_numeric_hash = _digest({name: source["model"][name] for name in NUMERIC_NAMES})
    added_names = {"reader.recognition.gains", "reader.recognition._extra_state"}
    for route in ROUTES:
        model = RecognitionSequenceSpliceLearner(mode="learned", seed=seed)
        expected = model.state_dict()
        if (set(expected) != set(source["model"]) | added_names
                or any(source["model"][name] != expected[name]
                       for name in METADATA_NAMES if name not in ADAPTED_METADATA)
                or source["model"]["_extra_state"]["schema"]
                != "factorized-sequence-splice-learner-v1"
                or source["model"]["reader._extra_state"]["schema"]
                != "factorized-sequence-memory-learner-v1"
                or bool(expected["reader.recognition.gains"].ne(0).any())):
            raise ValueError("recognition successor extension or retained metadata changed")
        work = {}
        work["inherited_empty_512_bank_constructions"] = 1
        work["inherited_empty_512_bank_tensor_bytes"] = 34_892_296
        work["successor_1024_bank_constructions"] = 1
        work["successor_1024_bank_tensor_bytes"] = 69_784_584
        old_bank = {name: source["model"]["reader.assemblies." + name]
                    for name in ("prototypes", "outgoing", "used", "_extra_state")}
        grown = model.reader.assemblies.promote_512_state(old_bank, work=work)
        adapted = dict(expected)
        for name in NUMERIC_NAMES:
            adapted[name] = source["model"][name]
        for name, value in grown.items():
            adapted["reader.assemblies." + name] = value
        model.load_state_dict(adapted, work=work)
        states[route] = model.reader.memory.promote_1024_state(source_tape, work=work)
        model._experience = source["experience"]
        model.reader._active_experience = source["reader_experience"]
        actual = model.state_dict()
        for name in NUMERIC_NAMES:
            retained = actual[name][:512] if name in GROWN_NAMES else actual[name]
            if not torch.equal(retained, source["model"][name]):
                raise ValueError("recognition initialization changed a prior numerical value")
        if (any(bool(actual[name][512:].ne(0).any()) for name in GROWN_NAMES)
                or bool(actual["reader.recognition.gains"].ne(0).any())
                or tuple(actual["reader.recognition.gains"].shape) != (1024,)
                or int(model.reader.assemblies.used.item()) != 398 or states[route].used != 864
                or len(tensors(actual)) != 17 or tensor_bytes(actual) != NEW_MODEL_BYTES
                or tensor_bytes(vars(states[route])) != NEW_TAPE_BYTES):
            raise ValueError("successor growth changed occupancy, zero tail or gain initialization")
        views, wirings = model.reader._candidate_views, model.reader._self_wirings
        if (len(views) != 398 or len(wirings) != 398
                or work.get("factorized_candidate_builds") != 398
                or work.get("recognition_self_builds") != 398):
            raise ValueError("successor must build each of398 existing owner views and self wirings once")
        values = tensors(actual) + tensors(vars(states[route]))
        values += [tensor for view in views for tensor in (view.indices, view.values, view.flags)]
        values += [tensor for wiring in wirings for tensor in (wiring.indices, wiring.values)]
        storage_by_route[route] = {value.untyped_storage().data_ptr() for value in values
                                   if value.numel()}
        models[route], work_by_route[route] = model, work
        model_sizes[route], tape_sizes[route] = tensor_bytes(actual), tensor_bytes(vars(states[route]))
        views_by_route[route] = {
            "owners": len(views), "entries": sum(view.indices.numel() for view in views),
            "view_tensor_bytes": sum(view.indices.numel() * 4 + view.values.numel() * 4
                                     + view.flags.numel() for view in views),
            "self_wiring_tensor_bytes": sum(wiring.indices.numel() * 4 + wiring.values.numel() * 8
                                            for wiring in wirings),
            "views_sha256": model.reader._view_sha256,
            "self_wiring_sha256": model.reader._self_wiring_sha256,
        }
        # Adapted dictionaries borrow old tensors until strict in-place loading;
        # neither they nor extra growth scratch remain part of the live actor.
        del expected, adapted, grown, actual, values
    prior_storage = {value.untyped_storage().data_ptr() for value in tensors(source) if value.numel()}
    if (any(storage_by_route[route] & prior_storage for route in ROUTES)
            or storage_by_route["online"] & storage_by_route["learning_disabled"]):
        raise ValueError("recognition successor shares numerical or derived storage")
    model_hashes = {route: _digest(model.state_dict()) for route, model in models.items()}
    memory_hashes = {route: _digest(vars(state)) for route, state in states.items()}
    if (len(set(model_hashes.values())) != 1 or len(set(memory_hashes.values())) != 1
            or views_by_route["online"] != views_by_route["learning_disabled"]):
        raise ValueError("recognition online and disabled initial states differ")
    for path, expected_sha in ((prior_path, CHECKPOINT_SHA), (bank_path, BANK_SHA),
                               (report_path, REPORT_SHA), (sources_path, SOURCES_SHA)):
        if sha(path) != expected_sha:
            raise ValueError("authenticated initialization input changed during construction")
    return models, states, {
        "schema": "persistent-language-action-initial-custody-v6", "status": "initialized",
        "bank_path": str(bank_path), "bank_sha256": BANK_SHA,
        "prior_path": str(prior_path), "prior_sha256": CHECKPOINT_SHA,
        "prior_report_sha256": REPORT_SHA, "prior_sources_sha256": SOURCES_SHA,
        "prior_attempts_sha256": ATTEMPTS_SHA,
        "prior_source_revision": SOURCE_REVISION, "prior_selected_route": "online",
        "prior_required_source_sha256": {name: historical_sources[name] for name in required_sources},
        "loaded_payload_tensor_bytes": PRIOR_PAYLOAD_BYTES, "loaded_tensor_count": 40,
        "checkpoint_file_bytes": prior_path.stat().st_size, "torch_load_calls": 1,
        "checkpoint_hash_passes": 3, "bank_authenticated_read_passes": 2,
        "report_authenticated_read_passes": 2, "source_manifest_authenticated_read_passes": 2,
        "discarded_prior_routes": ["learning_disabled"], "old_model_constructions": 0,
        "new_combined_model_constructions": 2, "promoted_tape_states": 2,
        "restored_occupied_slots_per_route": 864, "preserved_assembly_owners_per_route": 398,
        "initial_experience_per_route": PRIOR_EXPERIENCE,
        "initial_reader_experience_per_route": PRIOR_READER_EXPERIENCE,
        "reader_experience_source": "complete V5 checkpoint; no journal reconstruction",
        "initial_model_tensor_bytes": model_sizes, "initial_tape_tensor_bytes": tape_sizes,
        "initial_model_tensor_counts": dict.fromkeys(ROUTES, 17),
        "old_numeric_tensor_sha256": old_numeric_hash,
        "new_numeric_tensor_names": ["reader.recognition.gains"],
        "metadata_adapters": list(ADAPTED_METADATA),
        "initial_model_sha256": model_hashes, "initial_memory_sha256": memory_hashes,
        "derived_views": views_by_route, "initialization_work": work_by_route,
        "initialization_seconds": time.perf_counter() - started,
        "ingest_calls": 0, "checkpoint_free_replay_calls": 0, "teacher_replay_calls": 0,
        "scope": "all V5 numeric values/512 event rows/1024 tape rows; independent zero tails",
        "free_policy": "frozen prior plus local input recognition; no old-output parity claim",
        "allocation_scope": "retained state and counted growth/wiring; constructors included in run peak",
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES, True),
    }
