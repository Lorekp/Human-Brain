"""Continue authenticated v3 online state into the larger assembly interface.

Importing this module performs no file reads, checkpoint load or construction.
The runtime entry point loads one whole saved container and constructs only the
two requested successor learners. Old numerical coordinates remain verifiable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_initial_v2 import sha

BANK_SHA = "39e899d8ae5fa5bb741edab2681ccf214d6f1883fd258f96d043a02b3609022e"
CHECKPOINT_SHA = "692634352ea323ff7737425a4d5c4e69cd3b39f1bcb9a6c85443a6612e59d03d"
PREFIX = ".pytest-tmp/persistent-language-action-v3-run/"
ROUTES = ("online", "learning_disabled")
PRIOR_EXPERIENCE = 138
PRIOR_TENSOR_BYTES = 4275920
PRIOR_MODEL_BYTES = 1312608
PRIOR_TAPE_BYTES = 825352
ACTION_NAMES = ("start_weight", "end_weight", "operation_weight", "operation_bias")
READER_NAMES = (
    "reader.input_codes", "reader.address_weight", "reader.codec.weight", "reader.codec.bias",
    "reader.prefix_weight", "reader.output_weight",
    "reader.context.indices", "reader.context.signs",
)
NEW_ASSEMBLY_NAMES = (
    "reader.assemblies.prototypes", "reader.assemblies.outgoing", "reader.assemblies.used",
)
OLD_METADATA_NAMES = ("_extra_state", "reader._extra_state", "reader.context._extra_state")


def _legacy_coordinate_indices():
    """Independent scalar inverse coordinates, without calling the embedding helper.

    Each list is in the old flattened feature order. New request/status rows,
    file suffixes and payload suffixes are the complement and must remain zero.
    """
    query_rows, file_rows, payload_rows = 1298, 513, 129
    widths = {
        "start": query_rows * (32 + file_rows + 32) + 1,
        "end": query_rows * 32 + 1024 + 1,
        "operation": query_rows * 64 + file_rows * payload_rows + 3,
    }
    start = [query * 32 + path for query in range(32) for path in range(32)]
    start.extend(query_rows * 32 + query * file_rows + offset
                 for query in range(32) for offset in range(129))
    start.extend(query_rows * (32 + file_rows) + query * 32 + lag
                 for query in range(32) for lag in range(32))
    start.append(widths["start"] - 1)
    end = [query * 32 + lag for query in range(32) for lag in range(32)]
    end.extend(query_rows * 32 + item for item in range(1024))
    end.append(widths["end"] - 1)
    operation = [block * query_rows * 32 + query * 32 + lag
                 for block in range(2) for query in range(32) for lag in range(32)]
    operation.extend(query_rows * 64 + offset * payload_rows + payload
                     for offset in range(129) for payload in range(17))
    operation.extend(range(widths["operation"] - 3, widths["operation"]))
    indices = {"start": start, "end": end, "operation": operation}
    for head, expected in (("start", 6177), ("end", 2049), ("operation", 4244)):
        if (len(indices[head]) != expected or len(set(indices[head])) != expected
                or min(indices[head]) < 0 or max(indices[head]) >= widths[head]):
            raise ValueError("independent action coordinate specification is invalid")
    return widths, indices


def initialize(prior_path, bank_path, *, seed=733):
    """One authenticated load, two independent copies, no prediction or task input."""
    prior_path, bank_path = Path(prior_path).resolve(), Path(bank_path).resolve()
    if type(seed) is not int or seed != 733:
        raise ValueError("assembly continuation requires the preserved seed733")
    bank_bytes = bank_path.read_bytes()
    if hashlib.sha256(bank_bytes).hexdigest() != BANK_SHA:
        raise ValueError("assembly continuation bank identity changed")
    bank = json.loads(bank_bytes)
    if (bank["status"] != "completed_and_independently_reviewed"
            or bank["evidence"][PREFIX + "final-state.pt"] != CHECKPOINT_SHA):
        raise ValueError("v3 bank does not bind the completed continuation")
    report_path = prior_path.parent / "report.json"
    report_bytes = report_path.read_bytes()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    if report_sha != bank["evidence"][PREFIX + "report.json"]:
        raise ValueError("completed v3 report identity changed")
    report = json.loads(report_bytes)
    if (report["schema"] != "persistent-language-action-report-v3"
            or report["status"] != "completed"):
        raise ValueError("prior report is not a completed v3 run")
    if sha(prior_path) != CHECKPOINT_SHA:
        raise ValueError("assembly continuation checkpoint identity changed")

    import torch

    from humanbrain.active_path_memory_learner import ActivePathMemoryLearner
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest
    from humanbrain.sequence_assembly_splice_learner import SequenceAssemblySpliceLearner

    # Rehash the same open descriptor immediately before the single load. The
    # discarded disabled route is decoded and counted, never installed or queried.
    with prior_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA:
            raise ValueError("checkpoint changed before its one deserialization")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {"schema", "routes"}
            or payload["schema"] != "persistent-language-action-state-v3"
            or set(payload["routes"]) != set(ROUTES)):
        raise ValueError("unexpected v3 continuation container")

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [tensor for item in value.values() for tensor in tensors(item)]
        return []

    def tensor_bytes(value):
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors(value))

    loaded_count = len(tensors(payload))
    loaded_bytes = tensor_bytes(payload)
    if loaded_count != 32 or loaded_bytes != PRIOR_TENSOR_BYTES:
        raise ValueError("whole v3 checkpoint tensor geometry changed")
    source = payload["routes"]["online"]
    del payload
    if (not isinstance(source, dict) or set(source) != {"model", "memory", "experience"}
            or type(source["experience"]) is not int or source["experience"] != PRIOR_EXPERIENCE
            or set(source["memory"]) != {"keys", "sensory", "successor", "write_index"}
            or _digest(source["model"]) != report["final_hashes"]["online"]["model"]
            or _digest(source["memory"]) != report["final_hashes"]["online"]["memory"]):
        raise ValueError("selected state is not the final online model and whole memory")
    old_numeric = {**{name: source["model"].get(name) for name in READER_NAMES},
                   **{"actions." + name: source["model"].get("actions." + name)
                      for name in ACTION_NAMES}}
    if (set(source["model"]) != set(old_numeric) | set(OLD_METADATA_NAMES)
            or not all(isinstance(value, torch.Tensor) for value in old_numeric.values())
            or tensor_bytes(source["model"]) != PRIOR_MODEL_BYTES
            or tensor_bytes(source["memory"]) != PRIOR_TAPE_BYTES):
        raise ValueError("v3 selected model key set or payload differs from the original contract")
    old_actions = {name: source["model"]["actions." + name] for name in ACTION_NAMES}
    widths, scalar_indices = _legacy_coordinate_indices()
    index_tensors = {head: torch.tensor(values, dtype=torch.int64)
                     for head, values in scalar_indices.items()}
    mapping_receipts = {}

    def validate_embedding(actions):
        for head, index in index_tensors.items():
            actual = actions[head + "_weight"]
            old = old_actions[head + "_weight"]
            shape = (2, widths[head]) if head == "operation" else (widths[head],)
            if (tuple(actual.shape) != shape or actual.dtype != torch.float32
                    or actual.device.type != "cpu"):
                raise ValueError("new action geometry differs from independent coordinates")
            extracted = actual.index_select(-1, index)
            if not torch.equal(extracted, old):
                raise ValueError("inverse action coordinate extraction changed prior values")
            unmapped = torch.ones(widths[head], dtype=torch.bool)
            unmapped[index] = False
            if bool(actual[..., unmapped].ne(0).any()):
                raise ValueError("new action coordinates must be zero before acquisition")
            mapping_receipts[head] = {
                "old_shape": list(old.shape), "new_shape": list(actual.shape),
                "mapped_values": old.numel(), "zero_new_values": actual.numel() - old.numel(),
                "indices_sha256": hashlib.sha256(json.dumps(
                    scalar_indices[head], separators=(",", ":")).encode("ascii")).hexdigest(),
                "inverse_values_sha256": _digest({"values": extracted}),
                "prior_values_sha256": _digest({"values": old}),
            }
        if not torch.equal(actions["operation_bias"], old_actions["operation_bias"]):
            raise ValueError("legacy operation biases changed")

    models, states, new_tensor_counts, new_tensor_bytes = {}, {}, {}, {}
    for route in ROUTES:
        model = SequenceAssemblySpliceLearner(mode="learned", seed=seed)
        expected = model.state_dict()
        # The new wrapper does not subclass ActivePathSpliceLearner. Its prior
        # metadata is reconstructed without instantiating an obsolete model.
        old_root_metadata = NeuronalSpliceLearner.get_extra_state(model)
        old_root_metadata.update({
            "schema": "active-path-splice-learner-v1",
            "payload_custody": "exact pending reader response; actual action before correction",
            "stop_teacher": (
                "operation correction only; acknowledge reader with zero payload offers"),
            "teacher_free": "resolve both action and reader responses without local updates",
        })
        if (source["model"]["_extra_state"] != old_root_metadata
                or source["model"]["reader._extra_state"]
                != ActivePathMemoryLearner.get_extra_state(model.reader)
                or source["model"]["reader.context._extra_state"]
                != expected["reader.context._extra_state"]):
            raise ValueError("original v3 configuration differs from the declared prior")
        expected_added = {
            *NEW_ASSEMBLY_NAMES, "reader.assemblies._extra_state", "actions._extra_state"}
        if set(expected) != set(source["model"]) | expected_added:
            raise ValueError("assembly continuation added an undeclared state field")
        model.actions.load_legacy_state(old_actions)
        validate_embedding(model.actions.state_dict())
        adapted = dict(model.state_dict())
        for name in READER_NAMES:
            adapted[name] = source["model"][name]
        for name in NEW_ASSEMBLY_NAMES:
            if bool(adapted[name].ne(0).any()):
                raise ValueError("new assembly bank must start completely empty")
        # The strict full successor loader validates all old/new fields before
        # copying; restoring the same whole tape does not ingest or select docs.
        states[route] = model.restore_snapshot({
            "model": adapted, "memory": source["memory"], "experience": PRIOR_EXPERIENCE,
        })
        actual = model.state_dict()
        if any(not torch.equal(actual[name], source["model"][name]) for name in READER_NAMES):
            raise ValueError("inherited reader numerical state changed")
        validate_embedding(model.actions.state_dict())
        if (_digest(states[route].snapshot()) != _digest(source["memory"])
                or states[route].used != 505 or model._experience != PRIOR_EXPERIENCE):
            raise ValueError("continuation lost the whole505-slot memory or action experience")
        if any(bool(actual[name].ne(0).any()) for name in NEW_ASSEMBLY_NAMES):
            raise ValueError("assembly state changed during initial restoration")
        models[route] = model
        new_tensor_counts[route] = len(tensors(actual))
        new_tensor_bytes[route] = tensor_bytes(actual)

    online, disabled = (models[route].state_dict() for route in ROUTES)
    if _digest(online) != _digest(disabled):
        raise ValueError("online and disabled initial numerical/configuration states differ")
    for name, value in online.items():
        if isinstance(value, torch.Tensor):
            if value.untyped_storage().data_ptr() == disabled[name].untyped_storage().data_ptr():
                raise ValueError("route model tensors share storage")
            old = source["model"].get(name)
            if isinstance(old, torch.Tensor):
                for copied in (value, disabled[name]):
                    if copied.untyped_storage().data_ptr() == old.untyped_storage().data_ptr():
                        raise ValueError("successor model aliases decoded prior storage")
    for name, value in vars(states["online"]).items():
        if isinstance(value, torch.Tensor):
            other = getattr(states["learning_disabled"], name)
            if value.untyped_storage().data_ptr() == other.untyped_storage().data_ptr():
                raise ValueError("successor tape aliases another route or decoded prior")
            for copied in (value, other):
                if (copied.untyped_storage().data_ptr()
                        == source["memory"][name].untyped_storage().data_ptr()):
                    raise ValueError("successor tape aliases decoded prior")
    if (sha(prior_path) != CHECKPOINT_SHA or sha(bank_path) != BANK_SHA
            or sha(report_path) != report_sha):
        raise ValueError("continuation inputs changed during initialization")
    receipt = {
        "schema": "persistent-language-action-initial-custody-v4", "status": "initialized",
        "bank_path": str(bank_path), "bank_sha256": BANK_SHA,
        "prior_path": str(prior_path), "prior_sha256": CHECKPOINT_SHA,
        "prior_report_path": str(report_path), "prior_report_sha256": report_sha,
        "prior_selected_route": "online", "prior_source_revision": bank["source_revision"],
        "loaded_payload_tensor_bytes": loaded_bytes, "loaded_tensor_count": loaded_count,
        "torch_load_calls": 1, "checkpoint_hash_passes": 3,
        "bank_authenticated_read_passes": 2, "report_authenticated_read_passes": 2,
        "checkpoint_file_bytes": prior_path.stat().st_size,
        "loaded_model_state_containers": 2, "loaded_tape_state_containers": 2,
        "discarded_prior_routes": ["learning_disabled"],
        "new_combined_model_constructions": 2, "new_reader_constructions": 2,
        "new_action_circuit_constructions": 2, "new_assembly_bank_constructions": 2,
        "old_model_constructions": 0, "restored_tape_states": 2, "ingest_calls": 0,
        "restored_occupied_slots_per_route": 505, "initial_experience_per_route": PRIOR_EXPERIENCE,
        "prior_model_tensor_bytes_per_route": PRIOR_MODEL_BYTES,
        "prior_tape_tensor_bytes_per_route": PRIOR_TAPE_BYTES,
        "initial_model_tensor_bytes": new_tensor_bytes,
        "initial_model_tensor_counts": new_tensor_counts,
        "new_assembly_tensor_names": list(NEW_ASSEMBLY_NAMES),
        "new_assembly_tensor_bytes_per_route": models["online"].reader.assemblies.tensor_bytes,
        "old_reader_numeric_tensor_sha256": _digest({
            name: old_numeric[name] for name in READER_NAMES}),
        "old_numeric_tensor_sha256": _digest(old_numeric),
        "action_coordinate_mappings": mapping_receipts,
        "metadata_adapters": ["_extra_state", "reader._extra_state"],
        "added_metadata": ["reader.assemblies._extra_state", "actions._extra_state"],
        "initial_model_sha256": {
            route: _digest(model.state_dict()) for route, model in models.items()},
        "initial_memory_sha256": {
            route: _digest(state.snapshot()) for route, state in states.items()},
        "prior_memory_sha256": _digest(source["memory"]),
        "model_modes": dict.fromkeys(ROUTES, "learned"),
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES, True),
        "scope": (
            "all prior reader values, inverse-verified action coordinates "
            "and whole505-slot memory"),
        "free_policy": "new assembly nearest-prototype output; enlarged raw input/action geometry",
        "coordinate_parity": "old numerical values preserved; no functional output parity claim",
        "checkpoint_free_replay_calls": 0,
    }
    return models, states, receipt
