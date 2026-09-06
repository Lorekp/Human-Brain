"""Continue actual neuronal state with changed local output-credit ownership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_initial_v2 import sha

BANK_SHA = "de1eac1cc27a961ba1525c71ee90ff599eb5dff31b25ff8e4f1260bcd09304c8"
CHECKPOINT_SHA = "bdc2db89692fee5d49b305935020199c06e8cf0282324e5c1a15bfe6fd9db860"
PREFIX = ".pytest-tmp/persistent-language-action-v2-run/"
ROUTES = ("online", "learning_disabled")


def initialize(prior_path, bank_path, *, seed=733):
    """Authenticate one saved container and preserve all numerical state in both copies."""
    prior_path, bank_path = Path(prior_path).resolve(), Path(bank_path).resolve()
    if seed != 733 or sha(bank_path) != BANK_SHA or sha(prior_path) != CHECKPOINT_SHA:
        raise ValueError("active-path continuation prior identity changed")
    bank = json.loads(bank_path.read_bytes())
    if (bank["status"] != "completed_and_independently_reviewed"
            or bank["evidence"][PREFIX + "final-state.pt"] != CHECKPOINT_SHA):
        raise ValueError("prior bank does not bind the completed continuation")
    report_bytes = (prior_path.parent / "report.json").read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != bank["evidence"][PREFIX + "report.json"]:
        raise ValueError("prior report changed")
    report = json.loads(report_bytes)

    import torch

    from humanbrain.active_path_splice_learner import ActivePathSpliceLearner
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest
    from humanbrain.request_conditioned_memory_learner import RequestConditionedMemoryLearner

    with prior_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA:
            raise ValueError("prior changed before deserialization")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if (set(payload) != {"schema", "routes"}
            or payload["schema"] != "persistent-language-action-state-v2"
            or set(payload["routes"]) != set(ROUTES)):
        raise ValueError("unexpected continuation checkpoint container")

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        return 0

    loaded_bytes = tensor_bytes(payload)
    if loaded_bytes != 4275920:
        raise ValueError("prior checkpoint geometry changed")
    source = payload["routes"]["online"]
    del payload
    if (set(source) != {"model", "memory", "experience"}
            or source["experience"] != 92
            or _digest(source["model"]) != report["final_hashes"]["online"]["model"]
            or _digest(source["memory"]) != report["final_hashes"]["online"]["memory"]):
        raise ValueError("selected prior is not the completed online model and whole memory")
    prior_numeric = {
        key: value for key, value in source["model"].items() if isinstance(value, torch.Tensor)
    }
    models, states = {}, {}
    for route in ROUTES:
        model = ActivePathSpliceLearner(mode="learned", seed=seed)
        expected = model.state_dict()
        if set(expected) != set(source["model"]):
            raise ValueError("active-path continuation must add no numerical state")
        if (source["model"]["_extra_state"] != NeuronalSpliceLearner.get_extra_state(model)
                or source["model"]["reader._extra_state"]
                != RequestConditionedMemoryLearner.get_extra_state(model.reader)):
            raise ValueError("prior reader/action metadata changed")
        adapted = dict(source["model"])
        for key in ("_extra_state", "reader._extra_state"):
            adapted[key] = expected[key]
        states[route] = model.restore_snapshot({
            "model": adapted, "memory": source["memory"], "experience": source["experience"],
        })
        actual = model.state_dict()
        if any(not torch.equal(actual[key], value) for key, value in prior_numeric.items()):
            raise ValueError("prior numerical state changed during adaptation")
        if (_digest(states[route].snapshot()) != _digest(source["memory"])
                or states[route].used != 505):
            raise ValueError("whole accumulated memory was not preserved")
        models[route] = model
    if (_digest(models["online"].state_dict()) != _digest(models["learning_disabled"].state_dict())
            or any(models["online"].state_dict()[key].data_ptr()
                   == models["learning_disabled"].state_dict()[key].data_ptr()
                   for key in prior_numeric)
            or any(getattr(states["online"], key).data_ptr()
                   == getattr(states["learning_disabled"], key).data_ptr()
                   for key, value in vars(states["online"]).items()
                   if isinstance(value, torch.Tensor))):
        raise ValueError("controls must match with independent model and memory storage")
    receipt = {
        "schema": "persistent-language-action-initial-custody-v3",
        "status": "initialized",
        "bank_path": str(bank_path), "bank_sha256": BANK_SHA,
        "prior_path": str(prior_path), "prior_sha256": CHECKPOINT_SHA,
        "prior_report_sha256": bank["evidence"][PREFIX + "report.json"],
        "prior_selected_route": "online", "prior_source_revision": bank["source_revision"],
        "loaded_payload_tensor_bytes": loaded_bytes, "torch_load_calls": 1,
        "loaded_model_state_containers": 2, "loaded_tape_state_containers": 2,
        "discarded_prior_routes": ["learning_disabled"],
        "new_combined_model_constructions": 2, "new_reader_constructions": 2,
        "new_action_circuit_constructions": 2, "restored_tape_states": 2,
        "restored_occupied_slots_per_route": 505, "initial_experience_per_route": 92,
        "old_model_constructions": 0, "retained_model_tensor_bytes": 2625216,
        "new_parameter_names": [], "old_numeric_tensor_sha256": _digest(prior_numeric),
        "initial_model_sha256": {route: _digest(m.state_dict()) for route, m in models.items()},
        "initial_memory_sha256": {route: _digest(s.snapshot()) for route, s in states.items()},
        "model_modes": dict.fromkeys(ROUTES, "learned"),
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES, True),
        "scope": "all prior numerical state and505-slot memory; new actual-response credit rule",
        "free_policy": "same numerical equations with incremental fixed activity reuse",
        "checkpoint_free_replay_calls": 0,
    }
    if sha(prior_path) != CHECKPOINT_SHA or sha(bank_path) != BANK_SHA:
        raise ValueError("continuation inputs changed during initialization")
    return models, states, receipt
