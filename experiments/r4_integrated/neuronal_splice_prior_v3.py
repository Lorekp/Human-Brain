"""Strict original v6 tensor prior with explicit changed address-feature meaning.

Five new systems share numerical reader tensors, not old tape states or an
assertion of functional parity. No additional historical reader is constructed.
Authentication helpers perform no prediction, correction or acquisition.
"""

from __future__ import annotations

import copy
from pathlib import Path

from experiments.r4_integrated.neuronal_splice_models_v3 import (
    ARMS,
    address_frontend,
    new_model,
    reader_policy,
)
from experiments.r4_integrated.neuronal_splice_prior_v1 import (
    authenticate_prior as authenticate_original_prior,
)
from experiments.r4_integrated.neuronal_splice_prior_v1 import file_sha256, tensor_bytes

COMPETITIVE_METADATA_FIELDS = frozenset({
    "schema", "recall", "competition", "free_belief", "codec_activity", "teacher_phase",
})
COINCIDENCE_METADATA_FIELDS = frozenset({
    "schema", "query_size", "input_code", "address_rail_size", "address",
    "address_initialization", "address_activity", "query", "query_normalization",
    "memory", "query_eos_feature_row", "upstream_dtype", "retention",
})


def authenticate_prior(root: Path) -> dict:
    custody = authenticate_original_prior(root)
    custody.update({
        "schema": "neuronal-splice-prior-custody-v3",
        "metadata_adaptation": (
            "mode and enumerated competitive phase fields in all five readers; "
            "enumerated categorical feature/typed-memory fields in four readers; "
            "all original v6 learned.weights tensors identical with changed feature "
            "meaning, no score or reading parity claim; fresh seed733 action synapses"
        ),
        "reader_policy": {arm: reader_policy(arm) for arm in ARMS},
        "address_frontend": {arm: address_frontend(arm) for arm in ARMS},
    })
    return custody


def initialize_models(root: Path) -> tuple[dict, dict]:
    """Authenticate/load once, construct five models, validate then copy the prior.

    Only a reviewed custody check or admitted workload may call this function.
    Calling the fixed base metadata method on the learned reference reads only
    its mode/seed and constants; it constructs no sixth reader or tape.
    """
    custody = authenticate_prior(root)
    import torch

    from humanbrain.local_memory_learner import LocalMemoryLearner
    from humanbrain.neuronal_splice_learner import _digest

    saved = torch.load(custody["checkpoint_path"], map_location="cpu", weights_only=True)
    if (not isinstance(saved, dict)
            or set(saved) != {"learned", "address_frozen", "slow_frozen", "writes_disabled"}
            or any(not isinstance(entry, dict) or set(entry) != {"weights", "memories"}
                   for entry in saved.values())):
        raise ValueError("historical checkpoint container changed")
    loaded_bytes = tensor_bytes(saved)
    if loaded_bytes != custody["historical_payload_tensor_bytes"]:
        raise ValueError("historical checkpoint tensor scope changed")
    source = saved["learned"]["weights"]
    models = {arm: new_model(arm, seed=733) for arm in ARMS}
    reference = models["graded_reference"].reader
    prototype = reference.state_dict()
    original_metadata = LocalMemoryLearner.get_extra_state(reference)
    if (not isinstance(source, dict) or set(source) != set(prototype)
            or source["_extra_state"] != original_metadata):
        raise ValueError("original reader fields or unadapted metadata changed")
    tensor_fields = {key: value for key, value in source.items() if isinstance(value, torch.Tensor)}
    if set(tensor_fields) != set(prototype) - {"_extra_state"}:
        raise ValueError("original reader tensor field types changed")
    for key, value in tensor_fields.items():
        expected = prototype[key]
        if (value.shape != expected.shape or value.dtype != expected.dtype
                or value.device != expected.device or value.requires_grad
                or value.grad is not None or value.grad_fn is not None):
            raise ValueError("original reader tensor geometry changed")
        if key == "input_codes" and not torch.equal(value, expected):
            raise ValueError("original reader fixed input codes changed")

    # Enumerate and validate every metadata transformation before publishing any
    # source tensor. No undeclared field may be added, removed or changed.
    adaptations, metadata = {}, {}
    source_metadata = source["_extra_state"]
    for arm, model in models.items():
        expected = model.reader.get_extra_state()
        fields = COMPETITIVE_METADATA_FIELDS
        if address_frontend(arm) == "coincidence":
            fields = fields | COINCIDENCE_METADATA_FIELDS
        permitted = {**source_metadata, "mode": model.reader.mode}
        permitted.update({name: expected[name] for name in fields})
        if permitted != expected:
            raise ValueError("reader adaptation changes undeclared metadata")
        metadata[arm] = copy.deepcopy(expected)
        adaptations[arm] = sorted(name for name in set(source_metadata) | set(expected)
                                  if source_metadata.get(name) != expected.get(name))
    action_before = {arm: _digest(model.actions.state_dict()) for arm, model in models.items()}
    for arm, model in models.items():
        adapted = {**source, "_extra_state": metadata[arm]}
        model.reader.load_state_dict(adapted, strict=True)
        del adapted
    pointers = {key: {value.data_ptr()} for key, value in tensor_fields.items()}
    for arm, model in models.items():
        actual = model.reader.state_dict()
        for key, value in tensor_fields.items():
            if (not torch.equal(actual[key], value)
                    or actual[key].data_ptr() in pointers[key]):
                raise ValueError("reader prior differs or shares tensor storage")
            pointers[key].add(actual[key].data_ptr())
        if _digest(model.actions.state_dict()) != action_before[arm]:
            raise ValueError("reader prior loading changed fresh action weights")
    if len(set(action_before.values())) != 1:
        raise ValueError("fresh action initialization differs across arms")
    action_pointers = {}
    for model in models.values():
        for name, value in model.actions.state_dict().items():
            if isinstance(value, torch.Tensor):
                seen = action_pointers.setdefault(name, set())
                if value.data_ptr() in seen:
                    raise ValueError("fresh action weights share storage across arms")
                seen.add(value.data_ptr())
    custody.update({
        "actual_loaded_payload_tensor_bytes": loaded_bytes,
        "metadata_fields_changed": adaptations,
        "original_reader_metadata_sha256": _digest(original_metadata),
        "initial_reader_metadata_sha256": {arm: _digest(value) for arm, value in metadata.items()},
        "initial_reader_tensor_sha256": _digest(tensor_fields),
        "initial_action_tensor_sha256": action_before,
        "initial_model_sha256": {arm: _digest(model.state_dict()) for arm, model in models.items()},
        "new_model_tensor_bytes": sum(
            tensor_bytes(model.state_dict()) for model in models.values()),
        "source_sha256": {
            relative: file_sha256(root / relative) for relative in (
                "experiments/r4_integrated/neuronal_splice_models_v3.py",
                "experiments/r4_integrated/neuronal_splice_prior_v3.py",
                "src/humanbrain/competitive_local_memory_learner.py",
                "src/humanbrain/neuronal_address_competition.py",
                "src/humanbrain/coincidence_local_memory_learner.py",
                "src/humanbrain/coincidence_sequence_memory.py",
                "src/humanbrain/neuronal_byte_coincidence.py",
            )
        },
    })
    if custody["new_model_tensor_bytes"] != 3172160:
        raise ValueError("five-arm combined learner storage changed")
    del tensor_fields, source, saved, actual, prototype
    return models, custody
