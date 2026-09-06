"""Continue the broader neuronal state while adding local contextual output cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BANK_SHA = "39eca6edf662f0e580bfbacd931d512ab1eb009f6daadd459d046e6e78037799"
CHECKPOINT_SHA = "e0c581ab907310e846a256fbd868c667e93c8d6cf96626c60b1177d3cdf3da7b"
PREFIX = ".pytest-tmp/persistent-language-action-v1-run/"
ROUTES = ("online", "learning_disabled")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(raw)
    return digest.hexdigest()


def initialize(prior_path, bank_path, *, seed=733):
    """One authenticated load; no task-selected tape or fresh old-model reset."""
    prior_path, bank_path = Path(prior_path).resolve(), Path(bank_path).resolve()
    if seed != 733 or sha(bank_path) != BANK_SHA or sha(prior_path) != CHECKPOINT_SHA:
        raise ValueError("broader continuation prior identity changed")
    bank = json.loads(bank_path.read_bytes())
    if (
        bank["status"] != "completed_and_independently_reviewed"
        or bank["evidence"][PREFIX + "final-state.pt"] != CHECKPOINT_SHA
    ):
        raise ValueError("prior result does not bind the continuation checkpoint")
    report_path = prior_path.parent / "report.json"
    if sha(report_path) != bank["evidence"][PREFIX + "report.json"]:
        raise ValueError("prior report changed")
    prior_report = json.loads(report_path.read_bytes())

    import torch

    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.persistent_coincidence_reader import PersistentCoincidenceReader512
    from humanbrain.request_conditioned_splice_learner import RequestConditionedSpliceLearner

    with prior_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA:
            raise ValueError("prior changed before deserialization")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if (
        set(payload) != {"schema", "routes"}
        or payload["schema"] != "persistent-language-action-state-v1"
        or set(payload["routes"]) != set(ROUTES)
    ):
        raise ValueError("unexpected prior continuation container")

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        return 0

    loaded_bytes = tensor_bytes(payload)
    if loaded_bytes != 2919568:
        raise ValueError("prior checkpoint payload geometry changed")
    source = payload["routes"]["online"]
    del payload
    if (
        set(source) != {"model", "memory", "experience"}
        or source["experience"] != 46
        or _digest(source["model"]) != prior_report["final_hashes"]["online"]["model"]
        or _digest(source["memory"]) != prior_report["final_hashes"]["online"]["memory"]
    ):
        raise ValueError("selected prior model/tape differs from the completed online route")
    models, states = {}, {}
    prior_numeric = {
        key: value for key, value in source["model"].items() if isinstance(value, torch.Tensor)
    }
    added_parameters = {"reader.prefix_weight", "reader.output_weight"}
    new_tensor_bytes = None
    for route in ROUTES:
        model = RequestConditionedSpliceLearner(mode="learned", seed=seed)
        expected = model.state_dict()
        fixed_new = {
            "reader." + key
            for key in model.reader.FIXED_STATE_NAMES
            if "reader." + key not in source["model"]
        }
        if set(expected) - set(source["model"]) != added_parameters | fixed_new | {
            "reader.context._extra_state"
        }:
            raise ValueError("undeclared new neuronal state")
        if source["model"]["_extra_state"] != expected["_extra_state"] or source["model"][
            "reader._extra_state"
        ] != PersistentCoincidenceReader512.get_extra_state(model.reader):
            raise ValueError("prior metadata is not the completed persistent reader/action model")
        for key in added_parameters:
            if bool(torch.count_nonzero(expected[key])):
                raise ValueError("new plastic cells must begin with zero synaptic currents")
        adapted = dict(expected)
        for key, value in source["model"].items():
            if key != "reader._extra_state":
                adapted[key] = value
        states[route] = model.restore_snapshot(
            {
                "model": adapted,
                "memory": source["memory"],
                "experience": source["experience"],
            }
        )
        actual = model.state_dict()
        if any(not torch.equal(actual[key], value) for key, value in prior_numeric.items()):
            raise ValueError("old learned numerical state was not preserved")
        if _digest(states[route].snapshot()) != _digest(source["memory"]):
            raise ValueError("accumulated neuronal memory changed during continuation")
        if states[route].used != 453:
            raise ValueError("continuation requires the whole accumulated 453-slot memory")
        models[route] = model
        new_tensor_bytes = tensor_bytes(actual)
    if _digest(models["online"].state_dict()) != _digest(
        models["learning_disabled"].state_dict()
    ) or any(
        models["online"].state_dict()[key].data_ptr()
        == models["learning_disabled"].state_dict()[key].data_ptr()
        for key, value in models["online"].state_dict().items()
        if isinstance(value, torch.Tensor)
    ):
        raise ValueError("continuation controls must match without shared parameter storage")
    receipt = {
        "schema": "persistent-language-action-initial-custody-v2",
        "status": "initialized",
        "bank_path": str(bank_path),
        "bank_sha256": BANK_SHA,
        "prior_path": str(prior_path),
        "prior_sha256": CHECKPOINT_SHA,
        "prior_report_sha256": bank["evidence"][PREFIX + "report.json"],
        "prior_selected_route": "online",
        "prior_source_revision": bank["source_revision"],
        "loaded_payload_tensor_bytes": loaded_bytes,
        "torch_load_calls": 1,
        "loaded_model_state_containers": 2,
        "loaded_tape_state_containers": 2,
        "discarded_prior_routes": ["learning_disabled"],
        "new_combined_model_constructions": 2,
        "new_reader_constructions": 2,
        "new_action_circuit_constructions": 2,
        "restored_tape_states": 2,
        "restored_occupied_slots_per_route": 453,
        "initial_experience_per_route": 46,
        "old_model_constructions": 0,
        "retained_model_tensor_bytes": 2 * new_tensor_bytes,
        "new_parameter_names": sorted(added_parameters),
        "old_numeric_tensor_sha256": _digest(prior_numeric),
        "initial_model_sha256": {
            route: _digest(model.state_dict()) for route, model in models.items()
        },
        "initial_memory_sha256": {
            route: _digest(state.snapshot()) for route, state in states.items()
        },
        "model_modes": dict.fromkeys(ROUTES, "learned"),
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES, True),
        "scope": "same prior trained tensors and accumulated memory; new cells and changed per-byte retrieval policy",
        "functional_prior_parity": False,
    }
    if sha(prior_path) != CHECKPOINT_SHA or sha(bank_path) != BANK_SHA:
        raise ValueError("continuation inputs changed during initialization")
    return models, states, receipt
