"""Authenticated v6 reader continuation; no historical learner is instantiated."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled")
PRIOR_DIRECTORY = ".pytest-tmp/local-joint-address-online-v6-run"
DECISION_PATH = "benchmarks/local_joint_address_online_v6.decision.json"
DECISION_SHA256 = "f9b3b02b3bacf9977bb990f115072d41a043adeb074283d390bf45b5a0bf7a97"
CHECKPOINT_SHA256 = "03654f61af4cccb93688a75e75a84c089800672530cdaffdc7a30f7cc5181180"
READER_SHA256 = "2a6bfe480757b248c585a00bdc9c79155af6abd3a162b3e93a098d406b73a3b8"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated_json(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"artifact identity changed: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("artifact must contain an object")
    return value


def authenticate_prior(root: Path) -> dict:
    """Verify the completed, independently audited lineage before torch loading.

    The pinned decision binds the historical audit and source manifest. We do
    not rerun its scientific audit or require all historical live files to remain
    unchanged; the reader source that will actually execute must match exactly.
    """
    root = root.resolve()
    decision = authenticated_json(root / DECISION_PATH, DECISION_SHA256)
    if not all(decision.get(key) is True for key in (
        "full_run_completed", "full_audit_passed", "compliant_executed_learning_preserved",
    )):
        raise ValueError("prior has no completed compliant learning evidence")
    bindings = decision["input_sha256"]
    names = {
        "report": f"{PRIOR_DIRECTORY}/report.json",
        "completion": f"{PRIOR_DIRECTORY}/completion.json",
        "audit": ".pytest-tmp/local-joint-address-online-v6-run-independent-audit.json",
    }
    evidence = {name: authenticated_json(root / path, bindings[path])
                for name, path in names.items()}
    report, completion, audit = (evidence[name] for name in names)
    if (report["status"] != "completed" or report["mode"] != "run"
            or completion["status"] != "passed" or completion["mode"] != "run"
            or audit["status"] != "passed"
            or report["source_revision"] != decision["source_revision"]
            or completion["source_revision"] != decision["source_revision"]
            or completion["report_sha256"] != bindings[names["report"]]
            or report["source_sha256"] != report["source_sha256_after"]):
        raise ValueError("prior completion/source/audit chain is inconsistent")
    reader_path = "src/humanbrain/local_memory_learner.py"
    if (report["source_sha256"].get(reader_path) != READER_SHA256
            or file_sha256(root / reader_path) != READER_SHA256):
        raise ValueError("prior reader source is not the held local learner")
    artifacts = completion["artifacts"]
    generated = authenticated_json(
        root / PRIOR_DIRECTORY / "source-free/generated.json",
        artifacts["source-free/generated.json"],
    )
    if (artifacts["source-free/source_free.pt"] != CHECKPOINT_SHA256
            or generated["checkpoint_sha256"] != CHECKPOINT_SHA256
            or generated["status"] != "completed"
            or audit["source_free_tensor_bytes"] != 33801664):
        raise ValueError("prior endpoint checkpoint is not the audited complete artifact")
    checkpoint = root / PRIOR_DIRECTORY / "source-free/source_free.pt"
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("prior checkpoint bytes changed")
    return {
        "schema": "neuronal-splice-prior-custody-v1",
        "decision_path": DECISION_PATH, "decision_sha256": DECISION_SHA256,
        "evidence_sha256": {path: bindings[path] for path in names.values()},
        "reader_source_sha256": READER_SHA256,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
        "historical_payload_tensor_bytes": 33801664,
        "selected_entry": "learned.weights",
        "metadata_adaptation": "mode only; all numerical tensors identical",
    }


def tensor_bytes(value) -> int:
    """Count every loaded tensor, including historical tapes not reused by us."""
    import torch

    if isinstance(value, torch.Tensor):
        if (value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None
                or value.grad is not None or not torch.isfinite(value).all()):
            raise ValueError("prior tensors must be finite CPU values without autograd")
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("unexpected checkpoint payload type")
    return 0


def initialize_models(root: Path) -> tuple[dict, dict]:
    """Continue the same reader circuit in four freshly gated action learners.

    Call only under the reviewed cost/full execution contract. Authentication
    happens before deserialization. The entire old payload remains counted until
    each strict copy is complete, then is released; no old tapes are restored.
    """
    custody = authenticate_prior(root)
    import torch

    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest

    saved = torch.load(custody["checkpoint_path"], map_location="cpu", weights_only=True)
    if (set(saved) != {"learned", "address_frozen", "slow_frozen", "writes_disabled"}
            or any(set(entry) != {"weights", "memories"} for entry in saved.values())):
        raise ValueError("historical checkpoint container changed")
    loaded_bytes = tensor_bytes(saved)
    if loaded_bytes != custody["historical_payload_tensor_bytes"]:
        raise ValueError("historical checkpoint tensor scope changed")
    source = saved["learned"]["weights"]
    models = {arm: NeuronalSpliceLearner(mode=arm, seed=733) for arm in ARMS}
    # Strict loading of the learned arm checks every tensor and all metadata,
    # including the private fixed codes. Other modes may differ ONLY by mode.
    models["learned"].reader.load_state_dict(source, strict=True)
    source_metadata = source["_extra_state"]
    for arm in ARMS[1:]:
        reader = models[arm].reader
        expected_metadata = reader.get_extra_state()
        permitted = {**source_metadata, "mode": reader.mode}
        if permitted != expected_metadata:
            raise ValueError("reader adaptation changes more than the declared mode")
        adapted = {key: value for key, value in source.items()}
        adapted["_extra_state"] = copy.deepcopy(expected_metadata)
        reader.load_state_dict(adapted, strict=True)
        del adapted
    tensor_fields = {key: value for key, value in source.items()
                     if isinstance(value, torch.Tensor)}
    for model in models.values():
        actual = model.reader.state_dict()
        if any(not torch.equal(actual[key], value) for key, value in tensor_fields.items()):
            raise ValueError("initial reader tensors differ across modes")
    custody["actual_loaded_payload_tensor_bytes"] = loaded_bytes
    custody["initial_model_sha256"] = {arm: _digest(model.state_dict())
                                        for arm, model in models.items()}
    custody["initial_reader_tensor_sha256"] = _digest(tensor_fields)
    custody["new_model_tensor_bytes"] = sum(tensor_bytes(model.state_dict())
                                            for model in models.values())
    if custody["new_model_tensor_bytes"] != 2537728:
        raise ValueError("combined learner storage changed")
    del tensor_fields, source, saved, actual
    return models, custody
