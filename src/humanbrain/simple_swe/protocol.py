from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA = "humanbrain.simple-swe-protocol.v1"
PARTITIONS = ("train", "dev", "eval")
FAILURE_TAXONOMY = (
    "format_error",
    "syntax_error",
    "policy_violation",
    "signature_error",
    "timeout",
    "memory_limit",
    "runtime_error",
    "regression_failure",
    "property_failure",
    "harness_error",
    "passed",
)


@dataclass(frozen=True)
class ExecutionLimits:
    wall_seconds: float
    cpu_seconds: int
    memory_mib: int
    processes: int
    output_bytes: int
    max_completion_bytes: int
    max_ast_nodes: int
    python_hash_seed: int


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    task_count: int
    family_quotas: Mapping[str, int]
    seed: int | None = None
    seed_strategy: str | None = None


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    partitions: Mapping[str, PartitionSpec]
    execution: ExecutionLimits
    regression_cases_per_task: int
    property_cases_per_task: int
    max_new_tokens: int


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _partition(name: str, value: object) -> PartitionSpec:
    if not isinstance(value, dict):
        raise ValueError(f"partition {name} must be an object")
    quotas = value.get("family_quotas")
    if not isinstance(quotas, dict) or not quotas:
        raise ValueError(f"partition {name} must define family quotas")
    parsed_quotas: dict[str, int] = {}
    for family, quota in quotas.items():
        if not isinstance(family, str) or not family.startswith(("t0_", "t1_", "t2_")):
            raise ValueError(f"invalid family id in partition {name}")
        parsed_quotas[family] = _positive_int(quota, f"{name}.{family}")
    task_count = _positive_int(value.get("tasks"), f"{name}.tasks")
    if sum(parsed_quotas.values()) != task_count:
        raise ValueError(f"partition {name} family quotas do not sum to tasks")
    seed = value.get("seed")
    seed_strategy = value.get("seed_strategy")
    if name == "eval":
        if seed is not None:
            raise ValueError("eval seed must not exist before model freeze")
        if seed_strategy != "post-freeze-256-bit-os-entropy":
            raise ValueError("eval seed strategy must be post-freeze OS entropy")
    else:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"partition {name} must have a fixed integer seed")
        if seed_strategy is not None:
            raise ValueError(f"partition {name} cannot define a seed strategy")
    return PartitionSpec(
        name=name,
        task_count=task_count,
        family_quotas=MappingProxyType(parsed_quotas),
        seed=seed,
        seed_strategy=seed_strategy,
    )


def _require_invariants(payload: dict[str, Any]) -> None:
    required: tuple[tuple[tuple[str, str], object], ...] = (
        (("metric", "name"), "pass@1"),
        (("metric", "samples_per_task"), 1),
        (("metric", "decoding"), "deterministic-cell-race"),
        (("metric", "repairs"), 0),
        (("metric", "test_feedback"), False),
        (("prompt_contract", "response"), "one-complete-function"),
        (("prompt_contract", "imports"), False),
        (("prompt_contract", "examples_in_prompt"), False),
        (
            ("hidden_tests", "storage"),
            "outside-source-and-model-visible-workspace",
        ),
        (("hidden_tests", "model_feedback"), "outcome-only-after-run"),
        (("freeze", "one_checkpoint_for_all_tasks"), True),
        (("freeze", "reload_checkpoint_before_each_task"), True),
        (("freeze", "verify_checkpoint_state_before_and_after"), True),
        (("freeze", "learning_during_dev_or_eval"), False),
        (("freeze", "task_specific_replay"), False),
        (("freeze", "require_clean_source_tree"), True),
        (("freeze", "eval_seed_created_after_freeze"), True),
        (("freeze", "sandbox_runtime_frozen_before_eval_seed"), True),
        (("freeze", "compiler_registry_frozen"), True),
        (("freeze", "canonical_runner_only"), True),
        (("freeze", "eval_seed_reserved_before_task_generation"), True),
        (("freeze", "external_anchor_before_eval_task_generation"), True),
        (
            ("reporting", "baselines"),
            "reporting-controls-after-cell-completions-are-sealed",
        ),
        (("execution", "max_completion_bytes"), 16384),
        (("execution", "max_ast_nodes"), 4096),
    )
    if payload.get("language") != "python-3.11":
        raise ValueError("protocol invariant language must remain python-3.11")
    for (section, field), expected in required:
        value = payload.get(section)
        if not isinstance(value, dict) or value.get(field) != expected:
            raise ValueError(f"protocol invariant {section}.{field} was weakened")
    reporting = payload.get("reporting")
    if not isinstance(reporting, dict) or tuple(
        reporting.get("failure_taxonomy", ())
    ) != FAILURE_TAXONOMY:
        raise ValueError("protocol failure taxonomy differs from v1")


def load_protocol(path: str | Path) -> ProtocolSpec:
    resolved = Path(path).resolve()
    payload_bytes = resolved.read_bytes()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("simple-SWE protocol is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unsupported simple-SWE protocol schema")
    _require_invariants(payload)
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol_id must be a non-empty string")
    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, dict) or set(raw_partitions) != set(PARTITIONS):
        raise ValueError("protocol must define exactly train, dev, and eval")
    partitions = {
        name: _partition(name, raw_partitions[name]) for name in PARTITIONS
    }
    family_sets = [set(partition.family_quotas) for partition in partitions.values()]
    if any(left & right for index, left in enumerate(family_sets) for right in family_sets[index + 1 :]):
        raise ValueError("task families must be disjoint across partitions")
    if partitions["train"].seed == partitions["dev"].seed:
        raise ValueError("train and dev seeds must be distinct")

    execution = payload.get("execution")
    hidden = payload.get("hidden_tests")
    prompt = payload.get("prompt_contract")
    if not isinstance(execution, dict) or not isinstance(hidden, dict) or not isinstance(prompt, dict):
        raise ValueError("protocol execution, hidden_tests, and prompt_contract are required")
    limits = ExecutionLimits(
        wall_seconds=float(execution["wall_seconds"]),
        cpu_seconds=_positive_int(execution["cpu_seconds"], "cpu_seconds"),
        memory_mib=_positive_int(execution["memory_mib"], "memory_mib"),
        processes=_positive_int(execution["processes"], "processes"),
        output_bytes=_positive_int(execution["output_bytes"], "output_bytes"),
        max_completion_bytes=_positive_int(
            execution["max_completion_bytes"], "max_completion_bytes"
        ),
        max_ast_nodes=_positive_int(execution["max_ast_nodes"], "max_ast_nodes"),
        python_hash_seed=int(execution["python_hash_seed"]),
    )
    if limits.wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    if limits.processes != 1:
        raise ValueError("simple-SWE v1 requires one sandbox process")
    return ProtocolSpec(
        protocol_id=protocol_id,
        path=resolved,
        sha256=sha256(payload_bytes).hexdigest(),
        raw=MappingProxyType(payload),
        partitions=MappingProxyType(partitions),
        execution=limits,
        regression_cases_per_task=_positive_int(
            hidden["fixed_regression_cases_per_task"],
            "fixed_regression_cases_per_task",
        ),
        property_cases_per_task=_positive_int(
            hidden["property_cases_per_task"], "property_cases_per_task"
        ),
        max_new_tokens=_positive_int(prompt["max_new_tokens"], "max_new_tokens"),
    )
