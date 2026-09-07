"""Authenticated v4 continuation with new zero routing and lossless tape growth.

Import performs no I/O or neuronal construction. initialize loads one complete
saved container, installs its online state in two independent successors, and
derives owner views from receptive bits without prediction or lesson replay.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_initial_v2 import sha

BANK_SHA = "3122aed0c040ae086597cbdec6c0e2535a268cc62f7c72da05453ab88dd35466"
CHECKPOINT_SHA = "26cabfa5aa12ce362c51fcac2f57d5fdf05be6a5041a33f00acd47c05f618817"
REPORT_SHA = "5c63ac8bbbb2992a8dfe94d078801c06748208aa96d5a0ab0cd4711a661e32aa"
SOURCES_SHA = "4c8d21ad74b4c6a44d0dec34e4c903114f13b34974d3c900629e4ea17c4f4395"
ATTEMPTS_SHA = "803bacd6949a9c695c30d30e8d3ff1f86c4203b0c019bf1df86919e91f6ee400"
SOURCE_REVISION = "07ab1047b19a3f39c82ca2e70aaa255f55a3923e"
PREFIX = ".pytest-tmp/persistent-language-action-v4-run/"
ROUTES = ("online", "learning_disabled")
PRIOR_EXPERIENCE = 186
PRIOR_READER_EXPERIENCE = 72
PRIOR_MODEL_BYTES = 40_498_096
PRIOR_TAPE_BYTES = 825_352
PRIOR_PAYLOAD_BYTES = 82_646_896
NEW_MODEL_BYTES = 48_764_528
NEW_TAPE_BYTES = 1_650_696
NUMERIC_NAMES = (
    "reader.input_codes", "reader.address_weight", "reader.codec.weight", "reader.codec.bias",
    "reader.prefix_weight", "reader.output_weight", "reader.context.indices",
    "reader.context.signs", "reader.assemblies.prototypes", "reader.assemblies.outgoing",
    "reader.assemblies.used", "actions.start_weight", "actions.end_weight",
    "actions.operation_weight", "actions.operation_bias",
)
METADATA_NAMES = (
    "_extra_state", "reader._extra_state", "reader.context._extra_state",
    "reader.assemblies._extra_state", "actions._extra_state",
)


def _read_bound(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("factorized continuation evidence identity changed")
    return json.loads(raw)


def initialize(prior_path, bank_path, *, seed=733):
    """One torch.load; two actors; counted canonical views and whole-tape promotion."""
    started = time.perf_counter()
    prior_path, bank_path = Path(prior_path).resolve(), Path(bank_path).resolve()
    if type(seed) is not int or seed != 733:
        raise ValueError("factorized continuation retains seed733")
    report_path, sources_path = (prior_path.parent / "report.json",
                                 prior_path.parent / "sources.json")
    attempts_path = prior_path.parent / "attempts.jsonl"
    bank = _read_bound(bank_path, BANK_SHA)
    report = _read_bound(report_path, REPORT_SHA)
    source_manifest = _read_bound(sources_path, SOURCES_SHA)
    if (bank["status"] != "completed_and_independently_reviewed"
            or bank["source_revision"] != SOURCE_REVISION
            or any(bank["evidence"][PREFIX + name] != expected for name, expected in (
                ("final-state.pt", CHECKPOINT_SHA), ("report.json", REPORT_SHA),
                ("sources.json", SOURCES_SHA), ("attempts.jsonl", ATTEMPTS_SHA)))
            or report["schema"] != "persistent-language-action-report-v4"
            or report["status"] != "completed"
            or report["final_checkpoint"]["sha256"] != CHECKPOINT_SHA
            or report["final_checkpoint"]["tensor_bytes"] != PRIOR_PAYLOAD_BYTES
            or report["final_memory_slots"]["online"] != 505
            or report["final_assembly_occupied"]["online"] != 235
            or bank["exposure"]["online"]["free_queries"] != PRIOR_READER_EXPERIENCE
            or bank["exposure"]["online"]["executions"] != 48):
        raise ValueError("bank/report do not describe the selected complete v4 continuation")
    if (not isinstance(source_manifest, dict) or not source_manifest
            or any(not isinstance(name, str) or not isinstance(value, str)
                   or len(value) != 64 for name, value in source_manifest.items())):
        raise ValueError("prior source manifest is invalid")
    # The historical manifest is authenticated as historical source identity.
    # Current successor files intentionally differ and get their own run closure.
    required_sources = (
        "experiments/r4_integrated/persistent_language_action_initial_v4.py",
        "experiments/r4_integrated/persistent_language_action_v4.py",
        "src/humanbrain/active_path_memory_learner.py",
        "src/humanbrain/sequence_assembly_memory_learner.py",
        "src/humanbrain/sequence_assembly_splice_learner.py",
    )
    historical_sources = {name.replace("\\", "/"): value
                          for name, value in source_manifest.items()}
    if any(name not in historical_sources for name in required_sources):
        raise ValueError("prior source lineage omits a required constructor or response path")
    if sha(prior_path) != CHECKPOINT_SHA or sha(attempts_path) != ATTEMPTS_SHA:
        raise ValueError("factorized continuation checkpoint identity changed")

    import torch

    from humanbrain.factorized_sequence_splice_learner import FactorizedSequenceSpliceLearner
    from humanbrain.neuronal_sequence_memory import TapeState
    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.sequence_assembly_memory_learner import SequenceAssemblyMemoryLearner
    from humanbrain.sequence_assembly_splice_learner import SequenceAssemblySpliceLearner

    with prior_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA:
            raise ValueError("checkpoint changed before its single deserialization")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {"schema", "routes"}
            or payload["schema"] != "persistent-language-action-state-v4"
            or set(payload["routes"]) != set(ROUTES)):
        raise ValueError("unexpected v4 checkpoint container")

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [tensor for item in value.values() for tensor in tensors(item)]
        return []

    def tensor_bytes(value):
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors(value))

    loaded_count, loaded_bytes = len(tensors(payload)), tensor_bytes(payload)
    if loaded_count != 38 or loaded_bytes != PRIOR_PAYLOAD_BYTES:
        raise ValueError("whole prior checkpoint tensor scope changed")
    source = payload["routes"]["online"]
    del payload
    if (not isinstance(source, dict) or set(source) != {"model", "memory", "experience"}
            or type(source["experience"]) is not int or source["experience"] != PRIOR_EXPERIENCE
            or not isinstance(source["model"], dict)
            or set(source["model"]) != set(NUMERIC_NAMES) | set(METADATA_NAMES)
            or not isinstance(source["memory"], dict)
            or set(source["memory"]) != {"keys", "sensory", "successor", "write_index"}
            or not all(isinstance(source["model"][name], torch.Tensor) for name in NUMERIC_NAMES)
            or tensor_bytes(source["model"]) != PRIOR_MODEL_BYTES
            or tensor_bytes(source["memory"]) != PRIOR_TAPE_BYTES
            or _digest(source["model"]) != report["final_hashes"]["online"]["model"]
            or _digest(source["memory"]) != report["final_hashes"]["online"]["memory"]):
        raise ValueError("selected checkpoint is not the complete final online state")
    source_tape = TapeState(**source["memory"])
    models, states, work_by_route, views_by_route = {}, {}, {}, {}
    numeric_hash = _digest({name: source["model"][name] for name in NUMERIC_NAMES})
    for route in ROUTES:
        model = FactorizedSequenceSpliceLearner(mode="learned", seed=seed)
        expected = model.state_dict()
        if (source["model"]["_extra_state"] != SequenceAssemblySpliceLearner.get_extra_state(model)
                or source["model"]["reader._extra_state"]
                != SequenceAssemblyMemoryLearner.get_extra_state(model.reader)
                or any(source["model"][name] != expected[name] for name in METADATA_NAMES[2:])
                or set(expected) != set(source["model"]) | {
                    "reader.router.theta", "reader.router._extra_state"}):
            raise ValueError("prior metadata or successor state extension differs from contract")
        if bool(expected["reader.router.theta"].ne(0).any()):
            raise ValueError("new routing coefficients must start at zero")
        adapted = dict(expected)
        for name in NUMERIC_NAMES:
            adapted[name] = source["model"][name]
        work = {}
        # This strict loader builds every canonical view exactly once before
        # model publication. It does not run any routing or output decision.
        model.load_state_dict(adapted, work=work)
        states[route] = model.reader.memory.promote_512_state(source_tape, work=work)
        model._experience = source["experience"]
        # v4's fresh SequenceAssembly reader constructor starts this counter at0;
        # its initializer restores only the outer counter. The bank authenticates
        # exactly72 completed reader responses. This is custody reconstruction,
        # not a value falsely attributed to the old checkpoint's numerical state.
        model.reader._active_experience = PRIOR_READER_EXPERIENCE
        actual = model.state_dict()
        if (any(not torch.equal(actual[name], source["model"][name]) for name in NUMERIC_NAMES)
                or bool(actual["reader.router.theta"].ne(0).any())
                or int(model.reader.assemblies.used.item()) != 235
                or states[route].used != 505 or tensor_bytes(actual) != NEW_MODEL_BYTES
                or len(tensors(actual)) != 16
                or tensor_bytes(vars(states[route])) != NEW_TAPE_BYTES
                or model._experience != PRIOR_EXPERIENCE
                or model.reader._active_experience != PRIOR_READER_EXPERIENCE):
            raise ValueError("successor initialization changed old values or declared new geometry")
        views = model.reader._candidate_views
        if len(views) != 235 or work.get("factorized_candidate_builds") != 235:
            raise ValueError("initialization must derive exactly all235 actual owner views")
        work.update({
            "new_model_initial_retained_tensor_bytes": NEW_MODEL_BYTES,
            "strict_model_publication_tensor_bytes": NEW_MODEL_BYTES,
            "prior_numeric_comparison_values": sum(source["model"][name].numel()
                                                    for name in NUMERIC_NAMES),
            "explicit_model_clone_calls": 0,
        })
        models[route], work_by_route[route] = model, work
        views_by_route[route] = {
            "owners": len(views), "entries": sum(view.indices.numel() for view in views),
            "tensor_bytes": sum(view.indices.numel() * 4 + view.values.numel() * 4
                                + view.flags.numel() for view in views),
            "source_sha256": model.reader._view_source_sha256,
            "views_sha256": model.reader._view_sha256,
        }

    def storage_set(values):
        return {value.untyped_storage().data_ptr() for value in values if value.numel()}

    prior_storage = storage_set(tensors(source))
    route_storage = {}
    for route in ROUTES:
        values = tensors(models[route].state_dict()) + tensors(vars(states[route]))
        values += [tensor for view in models[route].reader._candidate_views
                   for tensor in (view.indices, view.values, view.flags)]
        route_storage[route] = storage_set(values)
        if route_storage[route] & prior_storage:
            raise ValueError("successor aliases decoded prior model, tape or view storage")
    if route_storage["online"] & route_storage["learning_disabled"]:
        raise ValueError("successor routes share model, tape or derived view storage")
    model_hashes = {route: _digest(model.state_dict()) for route, model in models.items()}
    memory_hashes = {route: _digest(vars(state)) for route, state in states.items()}
    if (len(set(model_hashes.values())) != 1 or len(set(memory_hashes.values())) != 1
            or views_by_route["online"] != views_by_route["learning_disabled"]):
        raise ValueError("initial online and disabled states differ")
    for path, expected_sha in ((prior_path, CHECKPOINT_SHA), (bank_path, BANK_SHA),
                               (report_path, REPORT_SHA), (sources_path, SOURCES_SHA),
                               (attempts_path, ATTEMPTS_SHA)):
        if sha(path) != expected_sha:
            raise ValueError("authenticated initialization inputs changed during construction")
    receipt = {
        "schema": "persistent-language-action-initial-custody-v5", "status": "initialized",
        "bank_path": str(bank_path), "bank_sha256": BANK_SHA,
        "prior_path": str(prior_path), "prior_sha256": CHECKPOINT_SHA,
        "prior_report_path": str(report_path), "prior_report_sha256": REPORT_SHA,
        "prior_sources_path": str(sources_path), "prior_sources_sha256": SOURCES_SHA,
        "prior_attempts_path": str(attempts_path), "prior_attempts_sha256": ATTEMPTS_SHA,
        "prior_source_revision": SOURCE_REVISION, "prior_selected_route": "online",
        "prior_required_source_sha256": {
            name: historical_sources[name] for name in required_sources},
        "loaded_payload_tensor_bytes": loaded_bytes, "loaded_tensor_count": loaded_count,
        "checkpoint_file_bytes": prior_path.stat().st_size,
        "torch_load_calls": 1, "checkpoint_hash_passes": 3,
        "bank_authenticated_read_passes": 2, "report_authenticated_read_passes": 2,
        "source_manifest_authenticated_read_passes": 2,
        "prior_attempts_hash_passes": 2, "prior_attempt_prediction_replays": 0,
        "loaded_model_state_containers": 2, "loaded_tape_state_containers": 2,
        "discarded_prior_routes": ["learning_disabled"],
        "new_combined_model_constructions": 2, "new_reader_constructions": 2,
        "new_action_circuit_constructions": 2, "new_assembly_bank_constructions": 2,
        "new_router_constructions": 2, "old_model_constructions": 0,
        "promoted_tape_states": 2, "ingest_calls": 0, "checkpoint_free_replay_calls": 0,
        "restored_occupied_slots_per_route": 505, "preserved_assembly_owners_per_route": 235,
        "initial_experience_per_route": PRIOR_EXPERIENCE,
        "initial_reader_experience_per_route": PRIOR_READER_EXPERIENCE,
        "reader_experience_source": "v4 constructor0 + banked72 actual responses; not checkpoint",
        "prior_model_tensor_bytes_per_route": PRIOR_MODEL_BYTES,
        "prior_tape_tensor_bytes_per_route": PRIOR_TAPE_BYTES,
        "initial_model_tensor_bytes": dict.fromkeys(ROUTES, NEW_MODEL_BYTES),
        "initial_model_tensor_counts": dict.fromkeys(ROUTES, 16),
        "initial_tape_tensor_bytes": dict.fromkeys(ROUTES, NEW_TAPE_BYTES),
        "old_numeric_tensor_sha256": numeric_hash,
        "new_numeric_tensor_names": ["reader.router.theta"],
        "new_router_tensor_bytes_per_route": NEW_MODEL_BYTES - PRIOR_MODEL_BYTES,
        "metadata_adapters": ["_extra_state", "reader._extra_state"],
        "added_metadata": ["reader.router._extra_state"],
        "initial_model_sha256": model_hashes, "initial_memory_sha256": memory_hashes,
        "prior_memory_sha256": report["final_hashes"]["online"]["memory"],
        "derived_views": views_by_route, "initialization_work": work_by_route,
        "initialization_seconds": time.perf_counter() - started,
        "allocation_scope": (
            "retained model/tape bytes and counted view temporaries; inherited constructor "
            "scratch is included in measured initialization time and run peak, not a byte bound"),
        "model_modes": dict.fromkeys(ROUTES, "learned"),
        "learning_enabled": {"online": True, "learning_disabled": False},
        "episodic_writes_enabled": dict.fromkeys(ROUTES, True),
        "scope": "all v4 numerical tensors/235 owners and512 tape rows; new tail remains empty",
        "free_policy": "all-owner factorized routing; no functional v4 output parity claim",
        "reference_restoration": "separate parent-owned available-history reconstruction",
    }
    return models, states, receipt
