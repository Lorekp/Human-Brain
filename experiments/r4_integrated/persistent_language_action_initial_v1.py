"""Authenticate a completed numerical prior for two persistent development routes.

Importing this module constructs no models and loads no checkpoint. The broader
reviewed workload alone may call initialize. Prior tapes are counted on load,
then discarded; both routes begin with the same trained tensors and no episodes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

PRIOR_BANK_SHA256 = "807e256dbb750a41e24a00997bee89dbfb7ad4d6e2ee7ab3dc556686f93770a8"
PRIOR_RELATIVE_PATH = ".pytest-tmp/neuronal-splice-online-v3-run/endpoint/source_free.pt"
PRIOR_SHA256 = "deeb951704e59abf1e819172ffa7345558bc94151af4783970d2864016ea8de7"
PRIOR_ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled", "graded_reference")
ROUTES = ("online", "learning_disabled")
PRIOR_TENSOR_BYTES = 109896320
MODEL_TENSOR_BYTES = 634432
ADAPTED_READER_FIELDS = frozenset({
    "schema", "capacity", "memory", "address_activity", "competition_capacity",
    "query_limit_scope", "feature_block_size",
})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_prior(
    prior_path: Path, bank_path: Path, *, expected_bank_sha256: str = PRIOR_BANK_SHA256,
) -> dict:
    """Authenticate the completed bank and its direct checkpoint binding only."""
    prior_path, bank_path = Path(prior_path).resolve(), Path(bank_path).resolve()
    if expected_bank_sha256 != PRIOR_BANK_SHA256:
        raise ValueError("persistent development has one selected prior bank")
    raw = bank_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_bank_sha256:
        raise ValueError("prior bank bytes changed")
    bank = json.loads(raw)
    if (not isinstance(bank, dict)
            or bank.get("schema") != "neuronal-splice-online-result-v3"
            or bank.get("status") != "completed_and_audited"
            or bank.get("source_revision") != "4840c24d8e4265eaddeaea032231d52d2af3185d"
            or bank.get("evidence", {}).get(PRIOR_RELATIVE_PATH) != PRIOR_SHA256):
        raise ValueError("prior bank does not authenticate the selected completed checkpoint")
    if file_sha256(prior_path) != PRIOR_SHA256:
        raise ValueError("selected prior checkpoint bytes changed")
    return {
        "schema": "persistent-language-action-initial-custody-v1",
        "prior_path": str(prior_path), "prior_sha256": PRIOR_SHA256,
        "bank_path": str(bank_path), "bank_sha256": expected_bank_sha256,
        "bank_checkpoint_entry": PRIOR_RELATIVE_PATH,
        "prior_source_revision": bank["source_revision"],
        "prior_roster_sha256": bank["roster_sha256"],
        "prior_selected_arm": "learned",
        "prior_all_arm_exposure_counts": copy.deepcopy(bank["counts"]),
        "prior_exposure_scope": (
            "trained learned arm from the completed five-arm byte-coincidence stream; "
            "bank counts describe all five arms, not this one arm; that stream itself "
            "used its declared earlier trained reader prior, so this is not fresh training"
        ),
        "prior_learning_rule": bank["learning_rule"],
    }


def _payload_scope(payload, torch):
    """Check loaded container/count scope without reconstructing old models/tapes."""
    if (not isinstance(payload, dict) or set(payload) != {"schema", "arms"}
            or payload["schema"] != "neuronal-splice-endpoint-state-v3"
            or not isinstance(payload["arms"], dict) or set(payload["arms"]) != set(PRIOR_ARMS)):
        raise ValueError("selected prior checkpoint container changed")
    tensor_count = 0

    def count(value):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(count(item) for item in value)
        if value is None or type(value) in {str, int, float, bool}:
            return 0
        raise ValueError("undeclared prior checkpoint value type")

    for entry in payload["arms"].values():
        if (not isinstance(entry, dict) or set(entry) != {"weights", "memories", "experience"}
                or not isinstance(entry["weights"], dict)
                or not isinstance(entry["memories"], dict) or len(entry["memories"]) != 72
                or type(entry["experience"]) is not int or entry["experience"] != 256):
            raise ValueError("prior must contain five complete trained model/72-tape entries")
        for tape in entry["memories"].values():
            if (not isinstance(tape, dict)
                    or set(tape) != {"keys", "sensory", "successor", "write_index"}
                    or any(not isinstance(value, torch.Tensor) for value in tape.values())):
                raise ValueError("prior tape container changed")
    loaded_bytes = count(payload)
    if loaded_bytes != PRIOR_TENSOR_BYTES or tensor_count != 1480:
        raise ValueError("full prior payload scope changed")
    source = payload["arms"]["learned"]["weights"]
    return source, {
        "loaded_payload_tensor_bytes": loaded_bytes, "loaded_tensor_count": tensor_count,
        "loaded_model_state_containers": 5, "loaded_tape_state_containers": 360,
        "prior_selected_experience": payload["arms"]["learned"]["experience"],
        "discarded_model_state_containers": 4, "discarded_tape_state_containers": 360,
        "imported_tape_states": 0, "old_model_constructions": 0,
    }


def initialize(
    prior_path: Path, bank_path: Path, *, expected_bank_sha256: str = PRIOR_BANK_SHA256,
    seed: int = 733,
) -> tuple[dict, dict]:
    """Load once, discard old tapes, strictly initialize two identical online modes.

    Both routes use mode=learned and retain episodic writes. The caller must pass
    learning_enabled=False for every correction of the learning_disabled route.
    No source episode state, attempt identity or paused operation branch transfers.
    """
    if type(seed) is not int or seed != 733:
        raise ValueError("selected prior retains seed733 fixed input codes")
    receipt = authenticate_prior(prior_path, bank_path, expected_bank_sha256=expected_bank_sha256)
    import torch

    from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest
    from humanbrain.persistent_coincidence_reader import PersistentCoincidenceReader512

    # Bind the bytes read by torch to the bank, rather than trusting a path check
    # performed before imports. Exactly one weights_only deserialization follows.
    with Path(prior_path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != PRIOR_SHA256:
            raise ValueError("prior changed before load")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    source, scope = _payload_scope(payload, torch)
    del payload  # Only selected model tensors/metadata survive; all360 tapes are released.
    receipt.update(scope)
    models = {route: NeuronalSpliceLearner(
        mode="learned", seed=seed, reader=PersistentCoincidenceReader512(mode="learned", seed=seed)
    ) for route in ROUTES}
    prototype = models["online"].state_dict()
    if set(source) != set(prototype):
        raise ValueError("selected numerical model fields differ from the persistent model")
    old_reader_metadata = CoincidenceLocalMemoryLearner.get_extra_state(models["online"].reader)
    if (source["reader._extra_state"] != old_reader_metadata
            or source["_extra_state"] != prototype["_extra_state"]):
        raise ValueError("selected prior metadata is not the frozen learned coincidence model")
    expected_metadata = prototype["reader._extra_state"]
    adapted_metadata = copy.deepcopy(old_reader_metadata)
    adapted_metadata.update({key: expected_metadata[key] for key in ADAPTED_READER_FIELDS})
    if adapted_metadata != expected_metadata:
        raise ValueError("persistent reader adaptation changes undeclared fields")
    tensor_fields = {key: value for key, value in source.items() if isinstance(value, torch.Tensor)}
    if set(tensor_fields) != set(prototype) - {"_extra_state", "reader._extra_state"}:
        raise ValueError("selected prior tensor field types changed")
    selected_bytes = sum(value.numel() * value.element_size() for value in tensor_fields.values())
    if selected_bytes != MODEL_TENSOR_BYTES:
        raise ValueError("selected model numerical payload changed")
    for key, value in tensor_fields.items():
        expected = prototype[key]
        if (value.shape != expected.shape or value.dtype != expected.dtype
                or value.device != expected.device or value.requires_grad
                or value.grad_fn is not None or value.grad is not None
                or not bool(torch.isfinite(value).all())):
            raise ValueError("prior tensor geometry or finite state changed")
        if key == "reader.input_codes" and not torch.equal(value, expected):
            raise ValueError("prior fixed input codes changed")
    numeric_hash = _digest(tensor_fields)
    pointers = {key: {value.data_ptr()} for key, value in tensor_fields.items()}
    for model in models.values():
        adapted = {**source, "reader._extra_state": copy.deepcopy(adapted_metadata)}
        model.load_state_dict(adapted, strict=True)
        actual = model.state_dict()
        for key, expected in tensor_fields.items():
            if not torch.equal(actual[key], expected) or actual[key].data_ptr() in pointers[key]:
                raise ValueError("initial route differs from prior or shares parameter storage")
            pointers[key].add(actual[key].data_ptr())
        if model._experience != 0 or model._pending is not None:
            raise ValueError("persistent route inherited old action custody")
    if (file_sha256(Path(prior_path)) != PRIOR_SHA256
            or file_sha256(Path(bank_path)) != expected_bank_sha256):
        raise ValueError("prior custody changed during initialization")
    receipt.update({
        "status": "initialized", "torch_load_calls": 1,
        "new_combined_model_constructions": 2, "new_reader_constructions": 2,
        "new_action_circuit_constructions": 2, "new_tape_constructions": 0,
        "retained_model_tensor_bytes": 2 * MODEL_TENSOR_BYTES,
        "routes": list(ROUTES), "model_modes": dict.fromkeys(ROUTES, "learned"),
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES, True),
        "adapted_reader_fields": sorted(ADAPTED_READER_FIELDS),
        "original_reader_metadata_sha256": _digest(old_reader_metadata),
        "initial_reader_metadata_sha256": _digest(adapted_metadata),
        "prior_numeric_tensor_sha256": numeric_hash,
        "initial_model_sha256": {key: _digest(model.state_dict()) for key, model in models.items()},
        "initial_model_tensor_sha256": dict.fromkeys(ROUTES, numeric_hash),
        "initial_experience": dict.fromkeys(ROUTES, 0),
        "runtime_calls": {"ingest": 0, "query": 0, "teacher": 0, "action": 0},
        "scope": "all prior tensors loaded once; old tapes discarded before two new models",
    })
    del actual, adapted, tensor_fields, source, prototype
    return models, receipt
