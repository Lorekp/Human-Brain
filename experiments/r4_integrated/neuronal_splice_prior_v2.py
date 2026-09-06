"""Authenticated original reader tensors with explicitly changed free metadata.

No acquisition or prediction occurs here. The original compliant v6 lineage is
reused through its held authentication helper. Frozen splice-v1 final weights
and historical tapes never initialize this comparison.
"""

from __future__ import annotations

import copy
from pathlib import Path

from experiments.r4_integrated.neuronal_splice_models_v2 import ARMS, new_model, reader_policy
from experiments.r4_integrated.neuronal_splice_prior_v1 import (
    authenticate_prior as authenticate_original_prior,
)
from experiments.r4_integrated.neuronal_splice_prior_v1 import file_sha256, tensor_bytes

# These are precisely the free-policy/phase metadata overrides in the selected
# CompetitiveLocalMemoryLearner. Query, codec and local teaching rules retain
# their inherited metadata; mode is an independently declared control gate.
COMPETITIVE_METADATA_FIELDS = frozenset({
    "schema", "recall", "competition", "free_belief", "codec_activity", "teacher_phase",
})


def authenticate_prior(root: Path) -> dict:
    custody = authenticate_original_prior(root)
    custody.update({
        "schema": "neuronal-splice-prior-custody-v2",
        "metadata_adaptation": (
            "mode plus declared competitive free/teacher phase fields; all numerical reader "
            "tensors identical to original v6 learned.weights; fresh seed733 action synapses"
        ),
        "reader_policy": {arm: reader_policy(arm) for arm in ARMS},
    })
    return custody


def initialize_models(root: Path) -> tuple[dict, dict]:
    """Authenticate once, load once, and make five strict numerical continuations.

    Call only inside a reviewed ordinary custody check or admitted cost/full
    workload. The marginal reference authenticates the original metadata before
    the four competitive copies adapt their explicitly enumerated phase fields.
    No old memory is restored and no free/teacher traversal is performed.
    """
    custody = authenticate_prior(root)
    import torch

    from humanbrain.neuronal_splice_learner import _digest

    saved = torch.load(custody["checkpoint_path"], map_location="cpu", weights_only=True)
    if (set(saved) != {"learned", "address_frozen", "slow_frozen", "writes_disabled"}
            or any(set(entry) != {"weights", "memories"} for entry in saved.values())):
        raise ValueError("historical checkpoint container changed")
    loaded_bytes = tensor_bytes(saved)
    if loaded_bytes != custody["historical_payload_tensor_bytes"]:
        raise ValueError("historical checkpoint tensor scope changed")
    source = saved["learned"]["weights"]
    models = {arm: new_model(arm, seed=733) for arm in ARMS}
    models["marginal_reference"].reader.load_state_dict(source, strict=True)
    source_metadata = source["_extra_state"]
    adaptations = {"marginal_reference": []}
    for arm in ARMS[:-1]:
        reader = models[arm].reader
        expected = reader.get_extra_state()
        permitted = {**source_metadata, "mode": reader.mode}
        permitted.update({name: expected[name] for name in COMPETITIVE_METADATA_FIELDS})
        if permitted != expected:
            raise ValueError("competitive reader adaptation changes undeclared metadata")
        adapted = dict(source)
        adapted["_extra_state"] = copy.deepcopy(expected)
        reader.load_state_dict(adapted, strict=True)
        adaptations[arm] = sorted(name for name in set(source_metadata) | set(expected)
                                  if source_metadata.get(name) != expected.get(name))
        del adapted
    tensor_fields = {key: value for key, value in source.items() if isinstance(value, torch.Tensor)}
    for model in models.values():
        actual = model.reader.state_dict()
        if any(not torch.equal(actual[key], value) for key, value in tensor_fields.items()):
            raise ValueError("initial reader tensors differ across comparison arms")
    custody.update({
        "actual_loaded_payload_tensor_bytes": loaded_bytes,
        "metadata_fields_changed": adaptations,
        "initial_reader_tensor_sha256": _digest(tensor_fields),
        "initial_model_sha256": {arm: _digest(model.state_dict()) for arm, model in models.items()},
        "new_model_tensor_bytes": sum(tensor_bytes(model.state_dict()) for model in models.values()),
        "source_sha256": {
            relative: file_sha256(root / relative) for relative in (
                "experiments/r4_integrated/neuronal_splice_models_v2.py",
                "src/humanbrain/competitive_local_memory_learner.py",
                "src/humanbrain/neuronal_address_competition.py",
            )
        },
    })
    if custody["new_model_tensor_bytes"] != 3172160:
        raise ValueError("five-arm combined learner storage changed")
    del tensor_fields, source, saved, actual
    return models, custody
