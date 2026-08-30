"""One-shot six-family public-TRAIN gate for typed semantic synthesis.

The shared frozen synthesizer receives only a typed signature and the four
visible regression input/output rows.  Prompt text, task ids, family labels,
and the 32 property rows never enter model selection.  A selected completion
is scored once by the canonical Simple-SWE sandbox over the regressions and
property cases.  Preflight does not generate tasks or inspect a desired row;
execution requires an exclusive durable spend marker.  This is public TRAIN
development evidence, not prompt-only, M0, DEV/eval, protected, held-out,
Path-2, or C1 evidence.
"""

# ruff: noqa: E402 -- direct-script execution installs repo import roots first.

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain.simple_swe.artifacts import atomic_create_bytes
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.sandbox import (
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "public-typed-semantic-six-family-train-gate-v1"
PASS_VERDICT = "PASS_PUBLIC_TYPED_SEMANTIC_SIX_FAMILY_TRAIN_GATE_V1"
STOP_VERDICT = "STOP_PUBLIC_TYPED_SEMANTIC_SIX_FAMILY_TRAIN_GATE_V1"
INVALID_VERDICT = "INVALID_PUBLIC_TYPED_SEMANTIC_SIX_FAMILY_TRAIN_GATE_V1"
PREFLIGHT_VERDICT = "PREFLIGHT_READY_NOT_RUN"
BLOCKED_PREFLIGHT_VERDICT = "PREFLIGHT_BLOCKED_ALREADY_SPENT"

PUBLIC_FAMILIES = (
    "t0_scalar_clamp",
    "t0_scalar_nonnegative",
    "t1_string_filter_class",
    "t2_clamp_then_sum",
    "t2_map_then_count",
    "t2_filter_then_reverse",
)
PUBLIC_FAMILY_TIERS = (0, 0, 1, 2, 2, 2)
FAMILY_QUOTA = 10
FAMILY_PASS_THRESHOLD = 8
FAMILY_FAILURE_STOP = FAMILY_QUOTA - FAMILY_PASS_THRESHOLD + 1
VISIBLE_REGRESSION_CASES = 4
UNEXPOSED_PROPERTY_CASES = 32
PUBLIC_LITERAL_BANK = tuple(range(-9, 10))
MAX_AST_COST = 9
_EXPECTED_WORK_BOUND_LIMITS = {
    "retained-representatives-per-bucket": 256,
    "declared-root-candidates": 6_000,
    "operator-predictions": 30_000,
    "physical-races": 80_000,
}
DECLARED_TASK_IDS = tuple(
    f"simple-swe-v1/train/tier-{tier}/{family}/{ordinal:03d}"
    for ordinal in range(FAMILY_QUOTA)
    for family, tier in zip(PUBLIC_FAMILIES, PUBLIC_FAMILY_TIERS, strict=True)
)

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_STRING = ProgramType.String()
_PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
_EXPECTED_PROTOCOL_SHA256 = (
    "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
)
_EXPECTED_FAMILY_SIGNATURE_SHAPES = {
    "t0_scalar_clamp": ("n", _INT, _INT),
    "t0_scalar_nonnegative": ("n", _INT, _INT),
    "t1_string_filter_class": ("text", _STRING, _STRING),
    "t2_clamp_then_sum": ("values", _LIST_INT, _INT),
    "t2_map_then_count": ("values", _LIST_INT, _INT),
    "t2_filter_then_reverse": ("text", _STRING, _STRING),
}
_ARTIFACT_PATH = Path(__file__).with_suffix(".json")
_SPEND_MARKER_PATH = Path(__file__).with_suffix(".spent.json")
_EXPOSURE_REGISTRY_PATH = Path(__file__).with_name(
    "public_typed_semantic_six_family_train_exposure_registry_v1.json"
)
_PRIOR_BRIDGE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_semantic_simple_swe_train_bridge_v1.json"
)
_REQUIRED_PRIOR_BRIDGE_ARTIFACT_SHA256 = (
    "09fbaef134d2ba18a68b513a89d7df5ab8708c8be84940e71a29b710f7ff9767"
)
_COMPACT_ARTIFACT_PATH = Path(__file__).with_name(
    "compact_typed_output_compact_v1.json"
)
_REQUIRED_COMPACT_ARTIFACT_SHA256 = (
    "f503820711ebc597aeac6b96e6d89dce17c48be7b756b182619a7238b531e179"
)
_REQUIRED_COMPACT_PAYLOAD_SHA256 = (
    "beab356461d443fc5ac5f5cb5198e755e3e6927b48a8aa860bc8fdbd68b3604e"
)
_EXPECTED_SEMANTIC_CHECKPOINT_SHA256 = (
    "af3eb36d62a5419bdd7855576e76e4fde832459b882bd35c39aa39dd58c25427"
)
_EXPECTED_OPERATOR_CHECKPOINT_SHA256 = (
    "92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e"
)
_EXPECTED_TASK_GENERATOR_SHA256 = (
    "b0edfc2eedc25335bcb2318a7b13026f6ab7affc29643af76e16f4c6b57240c3"
)
_EXPECTED_EXPOSURE_REGISTRY_PAYLOAD_SHA256 = (
    "91a4b2db928cbc894dabc65355d77a0fc778ab28221994980406356598e68efc"
)
_EXPECTED_HOST_RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.11.15",
    "python_sys_version": (
        "3.11.15 (main, Jul 23 2026, 14:42:43) [MSC v.1944 64 bit (AMD64)]"
    ),
    "python_executable_sha256": (
        "89e640c427d53939a746fd172f3f3071bfe45c2bdb171706655c8674a9d709ac"
    ),
    "python_base_executable_sha256": (
        "6765c6b1685c86877fabe14d82240f3fab2913a617a85e8d09e61dc40798013b"
    ),
    "python_runtime_dll_sha256": (
        "71ec92fb9efdf0bf9f8cc1780cf8be4ef7ba38bc1ff6d1d32ab13f24f2ded648"
    ),
    "numpy_version": "2.4.6",
    "numpy_core_binary_sha256": (
        "4fb4c5d62a6bd766eea716350eaf5396580e33cf7dc159e305488d1b7d72dad2"
    ),
    "numpy_build_config_sha256": (
        "7181b6ea1cca52555011ad8d54643db629cc48f6cbcf99bc50235b8d8a163cac"
    ),
    "numpy_record_sha256": (
        "79967f2393e5ba758534f4b99a5b5a4b72cfded61d04840a982afb5ea2676723"
    ),
}
_FREEZE_TAG = "public-typed-semantic-six-family-train-gate-v1-freeze"
_MECHANISM_SOURCE_PATHS = (
    "experiments/r3_pilot/public_typed_semantic_six_family_train_gate_v1.py",
    "experiments/r3_pilot/public_typed_semantic_synthesizer.py",
    "experiments/r3_pilot/public_typed_semantic_vector_race.py",
    "experiments/r3_pilot/public_typed_operator_successor.py",
    "experiments/r3_pilot/public_typed_output_cells.py",
    "src/humanbrain/typed_semantic_synthesis.py",
    "src/humanbrain/typed_program.py",
    "src/humanbrain/typed_program_types.py",
    "src/humanbrain/resting_glif_race.py",
    "src/humanbrain/resting_glif_tier_beam.py",
    "src/humanbrain/simple_swe/artifacts.py",
    "src/humanbrain/simple_swe/protocol.py",
    "src/humanbrain/simple_swe/sandbox.py",
    "src/humanbrain/simple_swe/tasks.py",
    "src/humanbrain/simple_swe/_worker.py",
    "benchmarks/simple_swe_protocol_v1.json",
    "docs/research/public-typed-semantic-six-family-train-gate-v1.md",
    "experiments/r3_pilot/public_typed_semantic_six_family_train_exposure_registry_v1.json",
)


@dataclass(frozen=True, slots=True)
class SandboxRuntime:
    executable: Path
    identity: dict[str, str]
    worker_source: bytes
    worker_sha256: str


class _CohortExecutionError(RuntimeError):
    """Carry a row-free completed audit prefix across terminal failure."""

    def __init__(
        self,
        cause: Exception,
        *,
        completed_results: Sequence[dict[str, object]],
        attempted_tasks: int,
        failing_task_id: str | None,
    ) -> None:
        super().__init__("cohort execution failed after the spend marker")
        self.cause_type = type(cause).__name__
        self.cause_detail = str(cause)
        self.completed_results = tuple(completed_results)
        self.attempted_tasks = attempted_tasks
        self.failing_task_id = failing_task_id


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _artifact_bytes(payload: dict[str, object]) -> bytes:
    without_hash = dict(payload)
    without_hash.pop("payload_sha256", None)
    final = dict(without_hash)
    final["payload_sha256"] = sha256(_canonical_json(without_hash)).hexdigest()
    return _json_bytes(final)


def _write_new_bytes(path: Path, payload: dict[str, object]) -> bytes:
    path = path.resolve()
    raw = _artifact_bytes(payload)
    try:
        atomic_create_bytes(path, raw)
    except FileExistsError as error:
        raise RuntimeError(f"exclusive artifact already exists: {path.name}") from error
    return raw


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _host_runtime_identity() -> dict[str, str]:
    """Authenticate the Python/NumPy build that performs physical synthesis."""

    executable = Path(sys.executable).resolve()
    base_executable = Path(sys._base_executable).resolve()
    python_runtime = base_executable.parent / "python311.dll"
    numpy_core = Path(np._core._multiarray_umath.__file__).resolve()
    numpy_record = Path(distribution("numpy")._path) / "RECORD"
    identity = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_sys_version": sys.version,
        "python_executable": str(executable),
        "python_executable_sha256": sha256(executable.read_bytes()).hexdigest(),
        "python_base_executable": str(base_executable),
        "python_base_executable_sha256": sha256(
            base_executable.read_bytes()
        ).hexdigest(),
        "python_runtime_dll": str(python_runtime.resolve()),
        "python_runtime_dll_sha256": sha256(python_runtime.read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "numpy_core_binary": numpy_core.name,
        "numpy_core_binary_sha256": sha256(numpy_core.read_bytes()).hexdigest(),
        "numpy_build_config_sha256": sha256(
            _canonical_json(np.__config__.CONFIG)
        ).hexdigest(),
        "numpy_record": str(numpy_record.resolve()),
        "numpy_record_sha256": sha256(numpy_record.read_bytes()).hexdigest(),
        "platform": platform.platform(),
    }
    if any(identity.get(key) != value for key, value in _EXPECTED_HOST_RUNTIME.items()):
        raise RuntimeError(
            "host synthesis runtime differs from the preregistered Python/NumPy build"
        )
    return identity


def _snapshot(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


def _visible_problem(
    task: GeneratedTask,
) -> tuple[TypedSignature, tuple[tuple[object, ...], ...], tuple[object, ...]]:
    regressions = task.private.regression_cases
    if len(regressions) != VISIBLE_REGRESSION_CASES:
        raise RuntimeError("Simple-SWE regression-case count changed")
    arguments: list[tuple[object, ...]] = []
    desired: list[object] = []
    for case in regressions:
        row = _snapshot(case.args)
        if not isinstance(row, tuple):
            raise RuntimeError("Simple-SWE case arguments are not a tuple")
        arguments.append(row)
        desired.append(_snapshot(case.expected))
    return task.public.signature, tuple(arguments), tuple(desired)


def _encode_visible(
    value: object,
    declared_type: ProgramType,
) -> output_cells.TypedOutputEncoding:
    if declared_type == _INT:
        output_type = output_cells.OutputType.INT
    elif declared_type == _LIST_INT:
        output_type = output_cells.OutputType.LIST_INT
        if isinstance(value, tuple):
            value = list(value)
    elif declared_type == _STRING:
        output_type = output_cells.OutputType.STRING
    else:
        raise RuntimeError("six-family gate supports only Int, List[Int], and String")
    return output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_type,
    )


def require_public_visible_support(
    tasks: Sequence[GeneratedTask],
) -> dict[str, object]:
    """Fail before model setup if any visible semantic row is unrepresentable."""

    selected = tuple(tasks)
    failures: list[dict[str, object]] = []
    encoded_values = 0
    for task in selected:
        signature, arguments, desired = _visible_problem(task)
        for case_index, (row, expected) in enumerate(
            zip(arguments, desired, strict=True)
        ):
            if len(row) != len(signature.parameters):
                failures.append(
                    {
                        "task_id": task.public.task_id,
                        "case_index": case_index,
                        "role": "arguments",
                        "status": "ARITY_MISMATCH",
                    }
                )
                continue
            for parameter, value in zip(signature.parameters, row, strict=True):
                encoded_values += 1
                if not _encode_visible(value, parameter.type).supported:
                    failures.append(
                        {
                            "task_id": task.public.task_id,
                            "case_index": case_index,
                            "role": parameter.name,
                            "status": "OUTPUT_UNKNOWN",
                        }
                    )
            encoded_values += 1
            if not _encode_visible(expected, signature.result).supported:
                failures.append(
                    {
                        "task_id": task.public.task_id,
                        "case_index": case_index,
                        "role": "desired",
                        "status": "OUTPUT_UNKNOWN",
                    }
                )
    if failures:
        raise RuntimeError(
            "public TRAIN visible values exceed the physical output bank: "
            + json.dumps(failures, sort_keys=True, allow_nan=False)
        )
    return {
        "tasks": len(selected),
        "cases": sum(len(_visible_problem(task)[1]) for task in selected),
        "encoded_values": encoded_values,
        "int_min": output_cells.INT_MIN,
        "int_max": output_cells.INT_MAX,
        "list_capacity": output_cells.LIST_CAPACITY,
    }


def select_public_train_tasks(
    protocol: ProtocolSpec,
) -> tuple[GeneratedTask, ...]:
    """Spend the sealed rows by selecting all 60 tasks in fixed round-robin order."""

    if protocol.path != _PROTOCOL_PATH.resolve():
        raise RuntimeError("gate requires the canonical Simple-SWE v1 protocol")
    if protocol.regression_cases_per_task != VISIBLE_REGRESSION_CASES:
        raise RuntimeError("canonical regression-case count changed")
    if protocol.property_cases_per_task != UNEXPOSED_PROPERTY_CASES:
        raise RuntimeError("canonical property-case count changed")
    train = protocol.partitions["train"]
    if any(
        train.family_quotas.get(family) != FAMILY_QUOTA
        for family in PUBLIC_FAMILIES
    ):
        raise RuntimeError("selected public TRAIN family quota changed")

    generated = generate_visible_partition(protocol, "train")
    indexed = {
        (task.public.family, task.public.ordinal): task
        for task in generated
        if task.public.family in PUBLIC_FAMILIES
    }
    expected_keys = {
        (family, ordinal)
        for family in PUBLIC_FAMILIES
        for ordinal in range(FAMILY_QUOTA)
    }
    if set(indexed) != expected_keys:
        raise RuntimeError("public TRAIN selection omitted or duplicated an ordinal")
    selected = tuple(
        indexed[(family, ordinal)]
        for ordinal in range(FAMILY_QUOTA)
        for family in PUBLIC_FAMILIES
    )
    if tuple(task.public.task_id for task in selected) != DECLARED_TASK_IDS:
        raise RuntimeError("generated public TRAIN ids differ from the frozen roster")
    if len({task.public.task_id for task in selected}) != len(selected):
        raise RuntimeError("public TRAIN gate contains duplicate task ids")
    for task in selected:
        if task.public.partition != "train":
            raise RuntimeError("gate may use public TRAIN tasks only")
        if len(task.private.regression_cases) != VISIBLE_REGRESSION_CASES:
            raise RuntimeError("selected task regression-case count changed")
        if len(task.private.property_cases) != UNEXPOSED_PROPERTY_CASES:
            raise RuntimeError("selected task property-case count changed")
        parameter_name, parameter_type, result_type = (
            _EXPECTED_FAMILY_SIGNATURE_SHAPES[task.public.family]
        )
        if (
            len(task.public.signature.parameters) != 1
            or task.public.signature.parameters[0].name != parameter_name
            or task.public.signature.parameters[0].type != parameter_type
            or task.public.signature.result != result_type
        ):
            raise RuntimeError("selected task callable type differs from the frozen family shape")
    return selected


def _require_prior_bridge_boundary(
    path: Path = _PRIOR_BRIDGE_ARTIFACT_PATH,
    compact_path: Path = _COMPACT_ARTIFACT_PATH,
) -> dict[str, object]:
    """Authenticate the exact prior capability and compact-engineering gates."""

    try:
        prior_bytes = path.read_bytes()
        prior = json.loads(prior_bytes)
        compact_bytes = compact_path.read_bytes()
        compact = json.loads(compact_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "exact prerequisite artifact is absent; gate stays unrun"
        ) from error
    prior_sha256 = sha256(prior_bytes).hexdigest()
    compact_sha256 = sha256(compact_bytes).hexdigest()
    if (
        prior_sha256 != _REQUIRED_PRIOR_BRIDGE_ARTIFACT_SHA256
        or not isinstance(prior, dict)
        or prior.get("schema")
        != "public-typed-semantic-simple-swe-train-bridge-v1"
        or prior.get("verdict")
        != "PASS_PUBLIC_TYPED_SEMANTIC_SIMPLE_SWE_TRAIN_BRIDGE_V1"
        or prior.get("declared_tasks") != 30
        or prior.get("attempted_tasks") != 30
        or prior.get("passed_tasks") != 30
        or prior.get("stop_family") is not None
        or prior.get("stop_reason") is not None
    ):
        raise RuntimeError("prior three-family public-TRAIN PASS is not exact")

    if not isinstance(compact, dict):
        raise RuntimeError("compact prerequisite is not one JSON object")
    compact_without_hash = dict(compact)
    compact_payload_sha256 = compact_without_hash.pop("payload_sha256", None)
    if (
        compact_sha256 != _REQUIRED_COMPACT_ARTIFACT_SHA256
        or compact_payload_sha256 != _REQUIRED_COMPACT_PAYLOAD_SHA256
        or compact_payload_sha256
        != sha256(_canonical_json(compact_without_hash)).hexdigest()
        or compact.get("phase") != "compact"
        or compact.get("comparison", {}).get("verdict")
        != "PASS_COMPACT_TYPED_OUTPUT_ENCODING"
        or compact.get("comparison", {}).get("passed") is not True
        or compact.get("publication", {}).get("status")
        != "PUBLISHED_COMPACT_PASS"
        or compact.get("production_audit_materialization_calls") != 0
        or compact.get("scope_guard", {}).get("external_task_or_corpus_inputs")
        != 0
    ):
        raise RuntimeError("compact typed-output prerequisite is not exact")
    return {
        "three_family_gate": {
            "schema": prior["schema"],
            "verdict": prior["verdict"],
            "artifact_sha256": prior_sha256,
            "declared_tasks": 30,
            "attempted_tasks": 30,
            "passed_tasks": 30,
        },
        "compact_gate": {
            "schema": compact["schema"],
            "verdict": compact["comparison"]["verdict"],
            "artifact_sha256": compact_sha256,
            "payload_sha256": compact_payload_sha256,
            "source_commit": compact["provenance"]["git_commit"],
            "comparison_gates": len(compact["comparison"]["gates"]),
        },
    }


def _mechanism_source_freeze() -> dict[str, object]:
    files: dict[str, str] = {}
    for relative in _MECHANISM_SOURCE_PATHS:
        path = REPO_ROOT / relative
        try:
            files[relative] = sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(f"required semantic source is absent: {relative}") from error
    if files["src/humanbrain/simple_swe/tasks.py"] != _EXPECTED_TASK_GENERATOR_SHA256:
        raise RuntimeError("canonical TRAIN task generator source changed")
    return {
        "files": files,
        "bundle_sha256": sha256(_canonical_json(files)).hexdigest(),
    }


def _load_exposure_registry(
    path: Path = _EXPOSURE_REGISTRY_PATH,
) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("frozen TRAIN exposure registry is absent") from error
    if not isinstance(payload, dict):
        raise RuntimeError("TRAIN exposure registry is not one JSON object")
    without_hash = dict(payload)
    expected_payload_hash = without_hash.pop("payload_sha256", None)
    prior_presence = payload.get("prior_r3_artifact_presence")
    if (
        payload.get("schema")
        != "public-typed-semantic-six-family-train-exposure-registry-v1"
        or payload.get("protocol_id") != "human-cell-simple-swe-v1"
        or payload.get("protocol_sha256") != _EXPECTED_PROTOCOL_SHA256
        or payload.get("partition") != "train"
        or tuple(payload.get("families", ())) != PUBLIC_FAMILIES
        or tuple(payload.get("declared_task_ids", ())) != DECLARED_TASK_IDS
        or tuple(payload.get("prompt_previously_public_task_ids", ()))
        != DECLARED_TASK_IDS
        or not isinstance(prior_presence, dict)
        or len(prior_presence) != 24
        or not set(prior_presence).issubset(DECLARED_TASK_IDS)
        or any(
            not isinstance(paths, list)
            or not paths
            or any(type(item) is not str for item in paths)
            for paths in prior_presence.values()
        )
        or type(expected_payload_hash) is not str
        or expected_payload_hash != _EXPECTED_EXPOSURE_REGISTRY_PAYLOAD_SHA256
        or expected_payload_hash != sha256(_canonical_json(without_hash)).hexdigest()
    ):
        raise RuntimeError("TRAIN exposure registry failed canonical validation")
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "raw_sha256": sha256(raw).hexdigest(),
        "payload_sha256": expected_payload_hash,
        "prompt_exposed_tasks": len(DECLARED_TASK_IDS),
        "prior_r3_artifact_presence_tasks": len(prior_presence),
        "prior_r3_artifact_presence_by_family": {
            family: sum(f"/{family}/" in task_id for task_id in prior_presence)
            for family in PUBLIC_FAMILIES
        },
    }


def _git_identity() -> dict[str, object]:
    def invoke(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git", "-C", str(REPO_ROOT), *arguments),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("cannot authenticate the gate source commit") from error
        return completed.stdout.strip()

    dirty = invoke("status", "--porcelain=v1", "--untracked-files=no")
    if dirty:
        raise RuntimeError("tracked source tree must be clean before preflight")
    tracked = invoke("ls-files", "--error-unmatch", "--", *_MECHANISM_SOURCE_PATHS)
    if len(tracked.splitlines()) != len(_MECHANISM_SOURCE_PATHS):
        raise RuntimeError("every mechanism source must be committed before preflight")
    commit = invoke("rev-parse", "HEAD")
    freeze_commit = invoke("rev-parse", f"refs/tags/{_FREEZE_TAG}^{{commit}}")
    if freeze_commit != commit:
        raise RuntimeError("HEAD does not match the external frozen gate tag")
    return {
        "commit": commit,
        "tree": invoke("rev-parse", "HEAD^{tree}"),
        "branch": invoke("rev-parse", "--abbrev-ref", "HEAD"),
        "freeze_tag": _FREEZE_TAG,
        "freeze_commit": freeze_commit,
        "tracked_worktree_clean": True,
    }


def _protocol_preflight(protocol: ProtocolSpec) -> dict[str, object]:
    if protocol.path != _PROTOCOL_PATH.resolve():
        raise RuntimeError("gate requires the canonical Simple-SWE v1 protocol")
    if protocol.sha256 != _EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("canonical Simple-SWE v1 protocol bytes changed")
    if protocol.regression_cases_per_task != VISIBLE_REGRESSION_CASES:
        raise RuntimeError("canonical regression-case count changed")
    if protocol.property_cases_per_task != UNEXPOSED_PROPERTY_CASES:
        raise RuntimeError("canonical property-case count changed")
    train = protocol.partitions["train"]
    if train.seed != 1300781 or train.task_count != 90:
        raise RuntimeError("canonical public TRAIN partition changed")
    if any(train.family_quotas.get(family) != FAMILY_QUOTA for family in PUBLIC_FAMILIES):
        raise RuntimeError("selected public TRAIN family quota changed")
    return {
        "verdict": PREFLIGHT_VERDICT,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "partition": "train",
        "partition_seed": train.seed,
        "families": PUBLIC_FAMILIES,
        "family_quota": FAMILY_QUOTA,
        "family_pass_threshold": FAMILY_PASS_THRESHOLD,
        "family_failure_stop": FAMILY_FAILURE_STOP,
        "declared_tasks": len(DECLARED_TASK_IDS),
        "declared_task_ids": DECLARED_TASK_IDS,
        "declared_task_ids_sha256": sha256(
            _canonical_json(DECLARED_TASK_IDS)
        ).hexdigest(),
        "task_order": "ordinal-major-round-robin-in-protocol-family-order",
        "sealed_rows_materialized": 0,
    }


def _claim_spend(
    *,
    preflight: dict[str, object],
    prerequisites: dict[str, object],
    source_freeze: dict[str, object],
    git_identity: dict[str, object],
    exposure_registry: dict[str, object],
    runtime: SandboxRuntime,
    host_runtime: dict[str, str],
    model: semantic.FrozenTypedSemanticSynthesizer,
    exclusive_host_attested: bool,
) -> dict[str, object]:
    if not exclusive_host_attested:
        raise RuntimeError("--attest-exclusive-host is required for the one-shot spend")
    if _ARTIFACT_PATH.exists():
        raise RuntimeError("canonical six-family result artifact already exists")
    if source_freeze != _mechanism_source_freeze():
        raise RuntimeError("mechanism source changed after preflight")
    if git_identity != _git_identity():
        raise RuntimeError("Git tag/commit/tree identity changed after preflight")
    if (
        model.checkpoint_sha256 != _EXPECTED_SEMANTIC_CHECKPOINT_SHA256
        or model.live_checkpoint_sha256 != _EXPECTED_SEMANTIC_CHECKPOINT_SHA256
    ):
        raise RuntimeError("semantic checkpoint differs from the preregistered identity")
    if model.operator.checkpoint_sha256 != _EXPECTED_OPERATOR_CHECKPOINT_SHA256:
        raise RuntimeError("operator checkpoint differs from the preregistered identity")
    if host_runtime != _host_runtime_identity():
        raise RuntimeError("host synthesis runtime changed after preflight")
    marker_payload: dict[str, object] = {
        "schema": "public-typed-semantic-six-family-train-spend-v1",
        "status": "CLAIMED_BEFORE_TASK_MATERIALIZATION",
        "exclusive_host_attested": True,
        "preflight": preflight,
        "prerequisites": prerequisites,
        "source_freeze": source_freeze,
        "git": git_identity,
        "exposure_registry": exposure_registry,
        "synthesizer": {
            "literal_bank": PUBLIC_LITERAL_BANK,
            "max_ast_cost": MAX_AST_COST,
            "checkpoint_sha256": model.checkpoint_sha256,
            "operator_checkpoint_sha256": model.operator.checkpoint_sha256,
        },
        "sandbox": {
            "python": runtime.identity,
            "worker_sha256": runtime.worker_sha256,
        },
        "host_synthesis_runtime": host_runtime,
        "requested_artifact": str(_ARTIFACT_PATH.relative_to(REPO_ROOT)).replace(
            "\\", "/"
        ),
    }
    raw = _write_new_bytes(_SPEND_MARKER_PATH, marker_payload)
    written = json.loads(raw)
    return {
        "path": str(_SPEND_MARKER_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "raw_sha256": sha256(raw).hexdigest(),
        "payload_sha256": written["payload_sha256"],
        "status": marker_payload["status"],
    }


def _prepare_sandbox(
    protocol: ProtocolSpec,
    python_executable: Path | None,
) -> SandboxRuntime:
    executable = resolve_python_executable(python_executable)
    language = protocol.raw.get("language")
    identity = python_runtime_identity(executable, language)
    worker_path = SRC_ROOT / "humanbrain/simple_swe/_worker.py"
    worker_sha256 = sha256(worker_path.read_bytes()).hexdigest()
    return SandboxRuntime(
        executable=executable,
        identity=identity,
        worker_source=load_worker_snapshot(worker_sha256),
        worker_sha256=worker_sha256,
    )


def _tier_record(
    attempt: semantic.SemanticSynthesisTierAttempt,
) -> dict[str, object]:
    race = attempt.race
    race_record: dict[str, object] | None = None
    if race is not None:
        race_record = {
            "status": race.status,
            "candidate_keys": race.candidate_keys,
            "candidate_vector_sha256s": race.candidate_vector_sha256s,
            "desired_vector_sha256": race.desired_vector_sha256,
            "detector_sha256s": race.detector_sha256s,
            "projection_sha256": race.projection_sha256,
            "currents_sha256": race.currents_sha256,
            "tier_sha256": race.tier_sha256,
            "exact_winner_keys": race.exact_winner_keys,
            "winner_key": race.winner_key,
            "sha256": race.sha256,
        }
    return {
        "ast_cost": attempt.ast_cost,
        "evaluated_candidates": attempt.evaluated_candidates,
        "representative_candidate_ids": attempt.representative_candidate_ids,
        "race": race_record,
    }


def _validate_work_bound_decision(
    decision: semantic.TypedSemanticSynthesisDecision,
) -> None:
    audit = decision.work_bound
    if decision.status != semantic.STATUS_WORK_BOUND_STOP:
        if audit is not None:
            raise RuntimeError("non-work-bound decision carried a work-bound audit")
        return
    if audit is None:
        raise RuntimeError("work-bound STOP omitted its completed-prefix audit")
    expected_limit = _EXPECTED_WORK_BOUND_LIMITS.get(audit.bound_name)
    if expected_limit is None or audit.limit != expected_limit:
        raise RuntimeError("work-bound STOP named an unfrozen bound or limit")
    if audit.completed > audit.limit or audit.attempted <= audit.limit:
        raise RuntimeError("work-bound STOP did not cross its exact frozen limit")
    if (
        decision.operator_predictions != audit.completed_operator_predictions
        or decision.operator_races != audit.completed_physical_races
        or decision.operator_races_sha256
        != audit.completed_operator_races_sha256
    ):
        raise RuntimeError("work-bound STOP counters differ from its completed prefix")
    completed_by_bound = {
        "declared-root-candidates": audit.completed_declared_candidates,
        "operator-predictions": audit.completed_operator_predictions,
        "physical-races": audit.completed_physical_races,
    }
    if (
        audit.bound_name in completed_by_bound
        and audit.completed != completed_by_bound[audit.bound_name]
    ):
        raise RuntimeError("work-bound STOP completed count is internally inconsistent")
    if (
        len(audit.completed_operator_races_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in audit.completed_operator_races_sha256
        )
    ):
        raise RuntimeError("work-bound STOP omitted an exact completed-prefix digest")


def _task_record(
    model: semantic.FrozenTypedSemanticSynthesizer,
    task: GeneratedTask,
    protocol: ProtocolSpec,
    runtime: SandboxRuntime,
) -> dict[str, object]:
    started = perf_counter()
    signature, arguments, desired = _visible_problem(task)
    visible_problem_sha256 = sha256(
        _canonical_json(
            {
                "signature": signature.to_record(),
                "arguments": arguments,
                "desired_outputs": desired,
            }
        )
    ).hexdigest()
    initial_checkpoint = model.live_checkpoint_sha256
    if initial_checkpoint != model.checkpoint_sha256:
        raise RuntimeError("shared semantic synthesizer is not frozen")

    # This is the complete model-selection seam.  No task metadata or prompt is
    # supplied to the synthesizer.
    decision = model.synthesize(signature, arguments, desired)
    if (
        model.live_checkpoint_sha256 != initial_checkpoint
        or decision.checkpoint_before_sha256 != initial_checkpoint
        or decision.checkpoint_after_sha256 != initial_checkpoint
    ):
        raise RuntimeError("shared semantic synthesizer changed across a TRAIN task")
    _validate_work_bound_decision(decision)

    sandbox_record: dict[str, object] | None = None
    sandbox_integrity_error = False
    passed = False
    program_sha256: str | None = None
    source_sha256: str | None = None
    if decision.status == semantic.STATUS_DONE:
        if decision.program is None or decision.source is None:
            raise RuntimeError("DONE semantic decision omitted its compiled program")
        program_sha256 = decision.program.sha256
        source_sha256 = sha256(decision.source.encode("utf-8")).hexdigest()
        sandbox_result = execute_completion(
            decision.source,
            task.public,
            task.private,
            protocol.execution,
            python_executable=runtime.executable,
            python_executable_sha256=runtime.identity["executable_sha256"],
            worker_source=runtime.worker_source,
            worker_sha256=runtime.worker_sha256,
        )
        sandbox_record = asdict(sandbox_result)
        sandbox_integrity_error = sandbox_result.outcome == "harness_error"
        passed = sandbox_result.passed
    elif decision.status not in {
        semantic.STATUS_STOP,
        semantic.STATUS_WORK_BOUND_STOP,
    }:
        raise RuntimeError("semantic synthesizer returned an unknown terminal status")

    return {
        "task_id": task.public.task_id,
        "family": task.public.family,
        "ordinal": task.public.ordinal,
        "task_sha256": task.public.task_sha256,
        "hidden_test_sha256": task.public.hidden_test_sha256,
        "private_test_sha256": task.private.test_sha256,
        "signature": signature.to_record(),
        "visible_problem_sha256": visible_problem_sha256,
        "semantic_status": decision.status,
        "stop_reason": decision.stop_reason,
        "candidate_count": decision.candidate_count,
        "observed_semantic_classes": decision.observed_semantic_classes,
        "tier_attempts": tuple(_tier_record(item) for item in decision.tier_attempts),
        "selected_candidate_id": decision.selected_candidate_id,
        "terminal_program_sha256": program_sha256,
        "terminal_source_sha256": source_sha256,
        "operator_predictions": decision.operator_predictions,
        "operator_races": decision.operator_races,
        "operator_races_sha256": decision.operator_races_sha256,
        "work_bound": None if decision.work_bound is None else asdict(decision.work_bound),
        "checkpoint_immutable": True,
        "visible_regression_cases": VISIBLE_REGRESSION_CASES,
        "property_cases_exposed_to_synthesizer": 0,
        "property_cases_scored_in_sandbox": UNEXPOSED_PROPERTY_CASES,
        "sandbox": sandbox_record,
        "sandbox_integrity_error": sandbox_integrity_error,
        "passed": passed,
        "wall_time_seconds": perf_counter() - started,
    }


def run_public_train_cohort(
    model: semantic.FrozenTypedSemanticSynthesizer,
    training_summary: object,
    runtime: SandboxRuntime,
    protocol: ProtocolSpec,
    tasks: Sequence[GeneratedTask],
    prerequisite: dict[str, object],
    source_freeze: dict[str, object],
    preflight: dict[str, object],
    git_identity: dict[str, object],
    exposure_registry: dict[str, object],
    spend_marker: dict[str, object],
    host_runtime: dict[str, str],
) -> dict[str, object]:
    """Run the fixed cohort with one synthesizer and a decisive family gate."""

    selected = tuple(tasks)
    if tuple(task.public.task_id for task in selected) != DECLARED_TASK_IDS:
        raise RuntimeError("runner task order differs from the complete public cohort")
    try:
        marker_raw = _SPEND_MARKER_PATH.read_bytes()
    except OSError as error:
        raise RuntimeError("durable spend marker disappeared before the cohort") from error
    if (
        spend_marker.get("status") != "CLAIMED_BEFORE_TASK_MATERIALIZATION"
        or sha256(marker_raw).hexdigest() != spend_marker.get("raw_sha256")
    ):
        raise RuntimeError("durable spend marker identity changed before the cohort")
    prior_gate = prerequisite.get("three_family_gate")
    compact_gate = prerequisite.get("compact_gate")
    if not isinstance(prior_gate, dict) or not isinstance(compact_gate, dict):
        raise RuntimeError("exact nested prerequisite record is required")
    if (
        prior_gate.get("verdict")
        != "PASS_PUBLIC_TYPED_SEMANTIC_SIMPLE_SWE_TRAIN_BRIDGE_V1"
        or prior_gate.get("artifact_sha256")
        != _REQUIRED_PRIOR_BRIDGE_ARTIFACT_SHA256
        or compact_gate.get("verdict") != "PASS_COMPACT_TYPED_OUTPUT_ENCODING"
        or compact_gate.get("artifact_sha256") != _REQUIRED_COMPACT_ARTIFACT_SHA256
        or compact_gate.get("payload_sha256") != _REQUIRED_COMPACT_PAYLOAD_SHA256
    ):
        raise RuntimeError("exact prior capability and compact gates are required")
    if source_freeze != _mechanism_source_freeze():
        raise RuntimeError("semantic mechanism source changed before the cohort")
    if host_runtime != _host_runtime_identity():
        raise RuntimeError("host synthesis runtime changed before the cohort")
    if (
        model.live_checkpoint_sha256 != _EXPECTED_SEMANTIC_CHECKPOINT_SHA256
        or model.checkpoint_sha256 != _EXPECTED_SEMANTIC_CHECKPOINT_SHA256
        or model.operator.checkpoint_sha256 != _EXPECTED_OPERATOR_CHECKPOINT_SHA256
    ):
        raise RuntimeError("shared semantic synthesizer is not frozen before the cohort")

    started = perf_counter()
    results: list[dict[str, object]] = []
    attempted = Counter[str]()
    passed = Counter[str]()
    stop_family: str | None = None
    for task in selected:
        try:
            result = _task_record(model, task, protocol, runtime)
        except Exception as error:
            raise _CohortExecutionError(
                error,
                completed_results=results,
                attempted_tasks=len(results) + 1,
                failing_task_id=task.public.task_id,
            ) from error
        results.append(result)
        if result["sandbox_integrity_error"]:
            raise _CohortExecutionError(
                RuntimeError("canonical sandbox reported an integrity/harness error"),
                completed_results=results,
                attempted_tasks=len(results),
                failing_task_id=task.public.task_id,
            )
        family = task.public.family
        attempted[family] += 1
        passed[family] += int(bool(result["passed"]))
        if attempted[family] - passed[family] >= FAMILY_FAILURE_STOP:
            stop_family = family
            break

    try:
        after_sources = _mechanism_source_freeze()
        if after_sources != source_freeze:
            raise RuntimeError("semantic mechanism source changed during the cohort")
        if _git_identity() != git_identity:
            raise RuntimeError("Git commit/tree identity changed during the cohort")
        if _host_runtime_identity() != host_runtime:
            raise RuntimeError("host synthesis runtime changed during the cohort")
        if (
            sha256(_SPEND_MARKER_PATH.read_bytes()).hexdigest()
            != spend_marker["raw_sha256"]
        ):
            raise RuntimeError("exclusive spend marker changed during the cohort")
    except Exception as error:
        raise _CohortExecutionError(
            error,
            completed_results=results,
            attempted_tasks=len(results),
            failing_task_id=None,
        ) from error
    checkpoint_immutable = model.live_checkpoint_sha256 == model.checkpoint_sha256
    summaries = {
        family: {
            "passed": passed[family],
            "failed": attempted[family] - passed[family],
            "attempted": attempted[family],
            "not_run": FAMILY_QUOTA - attempted[family],
            "declared": FAMILY_QUOTA,
            "gate": FAMILY_PASS_THRESHOLD,
        }
        for family in PUBLIC_FAMILIES
    }
    cohort_passed = (
        len(results) == len(selected)
        and all(passed[family] >= FAMILY_PASS_THRESHOLD for family in PUBLIC_FAMILIES)
        and checkpoint_immutable
    )
    return {
        "schema": SCHEMA,
        "verdict": PASS_VERDICT if cohort_passed else STOP_VERDICT,
        "outcome_class": "PASS" if cohort_passed else "STOP_CAPABILITY",
        "stop_reason": None if cohort_passed else "FAMILY_FAILURE_LIMIT",
        "stop_family": stop_family,
        "declared_tasks": len(selected),
        "attempted_tasks": len(results),
        "passed_tasks": sum(passed.values()),
        "task_order": tuple(task.public.task_id for task in selected),
        "attempted_task_order": tuple(result["task_id"] for result in results),
        "family_results": summaries,
        "results": tuple(results),
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "sha256": protocol.sha256,
            "partition": "train",
            "families": PUBLIC_FAMILIES,
            "declared_task_ids_sha256": preflight["declared_task_ids_sha256"],
        },
        "visibility_contract": {
            "typed_signature_visible": True,
            "regression_arguments_visible_per_task": VISIBLE_REGRESSION_CASES,
            "regression_outputs_visible_per_task": VISIBLE_REGRESSION_CASES,
            "prompt_visible_to_synthesizer": False,
            "task_or_family_identity_visible_to_synthesizer": False,
            "property_cases_visible_per_task": 0,
            "property_cases_sandboxed_per_task": UNEXPOSED_PROPERTY_CASES,
        },
        "synthesizer": {
            "builds": 1,
            "literal_bank": PUBLIC_LITERAL_BANK,
            "max_ast_cost": MAX_AST_COST,
            "checkpoint_sha256": model.checkpoint_sha256,
            "operator_checkpoint_sha256": model.operator.checkpoint_sha256,
            "checkpoint_immutable": checkpoint_immutable,
            "training_summary": asdict(training_summary),
            "source_freeze": source_freeze,
        },
        "prerequisite": prerequisite,
        "preflight": preflight,
        "spend_marker": spend_marker,
        "provenance": git_identity,
        "exposure_registry": exposure_registry,
        "sandbox": {
            "python": runtime.identity,
            "worker_sha256": runtime.worker_sha256,
            "limits": asdict(protocol.execution),
        },
        "host_synthesis_runtime": host_runtime,
        "wall_time_seconds": perf_counter() - started,
        "claim_boundary": (
            "All 60 tasks from six canonical public TRAIN families; typed "
            "signature plus four regression input/output rows are visible to "
            "the shared frozen bottom-up synthesizer, while prompt/task/family "
            "identity and 32 property cases are excluded from model selection. "
            "All 60 prompts were historically public and some task ids appeared "
            "in earlier R3 artifacts, so this is not unseen-instance or "
            "contamination-clean evidence. This is not prompt-only, family-"
            "disjoint, M0, DEV/eval, protected, held-out, Path-2, C1, SWE-bench, "
            "Gemma, or Opus evidence."
        ),
    }


def write_public_artifact(payload: dict[str, object], path: Path) -> None:
    _write_new_bytes(path, payload)


def _execute_once(
    *,
    protocol: ProtocolSpec,
    preflight: dict[str, object],
    prerequisite: dict[str, object],
    exposure_registry: dict[str, object],
    source_freeze: dict[str, object],
    git_identity: dict[str, object],
    runtime: SandboxRuntime,
    host_runtime: dict[str, str],
    model: semantic.FrozenTypedSemanticSynthesizer,
    training_summary: object,
    exclusive_host_attested: bool,
) -> tuple[dict[str, object], int]:
    """Claim the one shot before materializing rows, then publish one result."""

    spend_marker = _claim_spend(
        preflight=preflight,
        prerequisites=prerequisite,
        source_freeze=source_freeze,
        git_identity=git_identity,
        exposure_registry=exposure_registry,
        runtime=runtime,
        host_runtime=host_runtime,
        model=model,
        exclusive_host_attested=exclusive_host_attested,
    )

    # This is the first operation that generates or reads any of the 60 sealed
    # regression/property rows.  The exclusive file-fsynced marker already exists.
    failure_stage = "task_selection"
    visible_support: dict[str, object] | None = None
    try:
        tasks = select_public_train_tasks(protocol)
        failure_stage = "visible_output_support"
        visible_support = require_public_visible_support(tasks)
        failure_stage = "cohort"
        payload = run_public_train_cohort(
            model,
            training_summary,
            runtime,
            protocol,
            tasks,
            prerequisite,
            source_freeze,
            preflight,
            git_identity,
            exposure_registry,
            spend_marker,
            host_runtime,
        )
        payload["visible_support"] = visible_support
        exit_code = 0 if payload["verdict"] == PASS_VERDICT else 1
    except Exception as error:
        cohort_error = error if isinstance(error, _CohortExecutionError) else None
        error_type = (
            cohort_error.cause_type if cohort_error is not None else type(error).__name__
        )
        error_detail = (
            cohort_error.cause_detail if cohort_error is not None else str(error)
        )
        completed_results = (
            cohort_error.completed_results if cohort_error is not None else ()
        )
        attempted_tasks = (
            cohort_error.attempted_tasks if cohort_error is not None else 0
        )
        failing_task_id = (
            cohort_error.failing_task_id if cohort_error is not None else None
        )
        capacity_stop = error_detail.startswith(
            "public TRAIN visible values exceed the physical output bank"
        )
        payload = {
            "schema": SCHEMA,
            "verdict": STOP_VERDICT if capacity_stop else INVALID_VERDICT,
            "outcome_class": (
                "STOP_CAPABILITY" if capacity_stop else "INVALID_INTEGRITY_OR_HARNESS"
            ),
            "stop_reason": (
                "VISIBLE_PHYSICAL_OUTPUT_UNSUPPORTED"
                if capacity_stop
                else "POST_MARKER_INTEGRITY_OR_HARNESS_ERROR"
            ),
            "failure_stage": failure_stage,
            "error_type": error_type,
            "error_detail_sha256": sha256(error_detail.encode("utf-8")).hexdigest(),
            "error_detail_redacted": True,
            "declared_tasks": len(DECLARED_TASK_IDS),
            "attempted_tasks": attempted_tasks,
            "completed_tasks": len(completed_results),
            "completed_task_records": completed_results,
            "attempted_task_order": tuple(
                record["task_id"] for record in completed_results
            ),
            "failing_task_id": failing_task_id,
            "partial_progress_available": bool(completed_results),
            "visible_support": visible_support,
            "preflight": preflight,
            "spend_marker": spend_marker,
            "prerequisite": prerequisite,
            "provenance": git_identity,
            "exposure_registry": exposure_registry,
            "source_freeze": source_freeze,
            "host_synthesis_runtime": host_runtime,
            "sandbox": {
                "python": runtime.identity,
                "worker_sha256": runtime.worker_sha256,
                "limits": asdict(protocol.execution),
            },
            "claim_boundary": (
                "Terminal one-shot diagnostic after the exclusive marker; safe "
                "completed-prefix records are retained, raw exception detail is "
                "redacted, and no retry, replacement cohort, literal/bound change, "
                "or family patch is licensed."
            ),
        }
        exit_code = 1 if capacity_stop else 2
    payload["publication"] = {
        "status": "PUBLISHED_TERMINAL_RESULT",
        "artifact": str(_ARTIFACT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    write_public_artifact(payload, _ARTIFACT_PATH)
    return payload, exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--attest-exclusive-host", action="store_true")
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.execute and (_SPEND_MARKER_PATH.exists() or _ARTIFACT_PATH.exists()):
        print(
            json.dumps(
                {
                    "verdict": BLOCKED_PREFLIGHT_VERDICT,
                    "declared_tasks": len(DECLARED_TASK_IDS),
                    "sealed_rows_materialized": 0,
                    "spend_marker_exists": _SPEND_MARKER_PATH.exists(),
                    "result_artifact_exists": _ARTIFACT_PATH.exists(),
                },
                sort_keys=True,
            )
        )
        return 0
    protocol = load_protocol(_PROTOCOL_PATH)
    preflight = _protocol_preflight(protocol)
    prerequisite = _require_prior_bridge_boundary()
    exposure_registry = _load_exposure_registry()
    source_freeze = _mechanism_source_freeze()
    git_identity = _git_identity()
    runtime = _prepare_sandbox(protocol, args.python_executable)
    host_runtime = _host_runtime_identity()
    model, training_summary = semantic.build_public_typed_semantic_synthesizer(
        literals=PUBLIC_LITERAL_BANK,
        max_ast_cost=MAX_AST_COST,
    )
    if (
        model.checkpoint_sha256 != _EXPECTED_SEMANTIC_CHECKPOINT_SHA256
        or model.live_checkpoint_sha256 != _EXPECTED_SEMANTIC_CHECKPOINT_SHA256
        or model.operator.checkpoint_sha256 != _EXPECTED_OPERATOR_CHECKPOINT_SHA256
    ):
        raise RuntimeError("task-independent synthesizer checkpoint drifted")

    if not args.execute:
        marker_exists = _SPEND_MARKER_PATH.exists()
        result_exists = _ARTIFACT_PATH.exists()
        print(
            json.dumps(
                {
                    **preflight,
                    "verdict": (
                        BLOCKED_PREFLIGHT_VERDICT
                        if marker_exists or result_exists
                        else PREFLIGHT_VERDICT
                    ),
                    "git": git_identity,
                    "source_freeze": source_freeze,
                    "prerequisites": prerequisite,
                    "exposure_registry": exposure_registry,
                    "host_synthesis_runtime": host_runtime,
                    "semantic_checkpoint_sha256": model.checkpoint_sha256,
                    "operator_checkpoint_sha256": model.operator.checkpoint_sha256,
                    "spend_marker_exists": marker_exists,
                    "result_artifact_exists": result_exists,
                },
                sort_keys=True,
            )
        )
        return 0

    payload, exit_code = _execute_once(
        protocol=protocol,
        preflight=preflight,
        prerequisite=prerequisite,
        exposure_registry=exposure_registry,
        source_freeze=source_freeze,
        git_identity=git_identity,
        runtime=runtime,
        host_runtime=host_runtime,
        model=model,
        training_summary=training_summary,
        exclusive_host_attested=args.attest_exclusive_host,
    )
    print(json.dumps({"artifact": str(_ARTIFACT_PATH), "verdict": payload["verdict"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAMILY_FAILURE_STOP",
    "FAMILY_PASS_THRESHOLD",
    "FAMILY_QUOTA",
    "DECLARED_TASK_IDS",
    "BLOCKED_PREFLIGHT_VERDICT",
    "MAX_AST_COST",
    "INVALID_VERDICT",
    "PASS_VERDICT",
    "PREFLIGHT_VERDICT",
    "PUBLIC_FAMILIES",
    "PUBLIC_FAMILY_TIERS",
    "PUBLIC_LITERAL_BANK",
    "SCHEMA",
    "STOP_VERDICT",
    "UNEXPOSED_PROPERTY_CASES",
    "VISIBLE_REGRESSION_CASES",
    "require_public_visible_support",
    "run_public_train_cohort",
    "select_public_train_tasks",
    "write_public_artifact",
]
