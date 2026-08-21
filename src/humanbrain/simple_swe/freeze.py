from __future__ import annotations

import importlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from humanbrain.simple_swe.artifacts import atomic_create_bytes
from humanbrain.simple_swe.compiler import (
    COMPLETION_ADAPTER_ID,
    CompilerIdentity,
    RawPythonCompiler,
)
from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.sandbox import (
    SANDBOX_PYTHON_FLAGS,
    SandboxConfigurationError,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import canonical_json_bytes, canonical_sha256

SCHEMA = "humanbrain.simple-swe-freeze.v3"
CANONICAL_RUNNER_ID = "humanbrain-simple-swe-sealed-runner-v1"
SANDBOX_EXECUTOR_ID = "humanbrain-simple-swe-job-sandbox-v1"
_EXECUTION_COMMITMENT_FIELDS = (
    "evaluation_root",
    "canonical_runner_id",
    "canonical_runner_sha256",
    "sandbox_executor_id",
    "sandbox_executor_sha256",
    "sandbox_worker_sha256",
    "sandbox_python_executable",
    "sandbox_python_implementation",
    "sandbox_python_version",
    "sandbox_python_executable_sha256",
    "sandbox_python_flags",
    "simulation_backend",
    "simulation_device",
    "simulation_device_policy",
    "simulation_source_digest",
    "simulation_artifact_path",
    "simulation_artifact_sha256",
    "simulation_architecture",
)
ALLOWED_CHECKPOINT_FORMATS = {
    "human-cell-brain-checkpoint-v1",
    "recurrent-human-cell-checkpoint-v3",
}
_EXECUTION_MODULES = (
    "humanbrain",
    "humanbrain.artifacts",
    "humanbrain.cells",
    "humanbrain.cortex",
    "humanbrain.cuda",
    "humanbrain.language",
    "humanbrain.native",
    "humanbrain.readout",
    "humanbrain.recurrent",
    "humanbrain.recurrent_artifacts",
    "humanbrain.recurrent_language",
    "humanbrain.tokenizer",
    "humanbrain.topology",
    "humanbrain.simple_swe",
    "humanbrain.simple_swe._worker",
    "humanbrain.simple_swe.adapter",
    "humanbrain.simple_swe.artifacts",
    "humanbrain.simple_swe.compiler",
    "humanbrain.simple_swe.freeze",
    "humanbrain.simple_swe.protocol",
    "humanbrain.simple_swe.reveal",
    "humanbrain.simple_swe.runner",
    "humanbrain.simple_swe.sandbox",
    "humanbrain.simple_swe.tasks",
)


class FreezeIntegrityError(RuntimeError):
    """The source or learned cell state differs from its freeze commitment."""


@dataclass(frozen=True)
class CheckpointHashes:
    checkpoint_format: str
    checkpoint_sha256: str
    checkpoint_state_sha256: str
    model_config_sha256: str
    tokenizer_sha256: str
    checkpoint_bytes: int


@dataclass(frozen=True)
class SimulationBackendIdentity:
    backend: str
    device: str
    device_policy: str
    source_digest: str | None
    artifact_path: str | None
    artifact_sha256: str | None
    architecture: str | None


@dataclass(frozen=True)
class FreezeManifest:
    schema: str
    protocol_id: str
    protocol_sha256: str
    checkpoint_format: str
    checkpoint_sha256: str
    checkpoint_state_sha256: str
    model_config_sha256: str
    tokenizer_sha256: str
    checkpoint_bytes: int
    evaluation_root: str
    completion_adapter_id: str
    completion_adapter_sha256: str
    compiler_id: str
    compiler_sha256: str
    compiler_mode: str
    compiler_algorithm_source: str
    canonical_runner_id: str
    canonical_runner_sha256: str
    sandbox_executor_id: str
    sandbox_executor_sha256: str
    sandbox_worker_sha256: str
    sandbox_python_executable: str
    sandbox_python_implementation: str
    sandbox_python_version: str
    sandbox_python_executable_sha256: str
    sandbox_python_flags: tuple[str, ...]
    simulation_backend: str
    simulation_device: str
    simulation_device_policy: str
    simulation_source_digest: str | None
    simulation_artifact_path: str | None
    simulation_artifact_sha256: str | None
    simulation_architecture: str | None
    inference_config_sha256: str
    source_head: str
    source_tree_sha256: str
    created_utc: str
    one_checkpoint_for_all_tasks: bool
    reload_before_each_task: bool
    verify_state_before_and_after: bool
    no_learning_after_freeze: bool
    no_task_specific_replay: bool
    one_canonical_evidence_root: bool
    external_anchor_before_task_generation: bool


def _normalized_array_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    array = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return len(header).to_bytes(8, "big") + header + array.tobytes(order="C")


def _checkpoint_metadata(state: Any) -> dict[str, Any]:
    try:
        metadata = json.loads(str(state["metadata"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise FreezeIntegrityError("checkpoint metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise FreezeIntegrityError("checkpoint metadata must be an object")
    checkpoint_format = metadata.get("format")
    if checkpoint_format not in ALLOWED_CHECKPOINT_FORMATS:
        raise FreezeIntegrityError("checkpoint is not a supported human-cell model")
    if checkpoint_format == "human-cell-brain-checkpoint-v1":
        if "allen_config_sha256" not in metadata:
            raise FreezeIntegrityError("checkpoint lacks its human-cell provenance")
    elif "allen_glif5_config_sha256" not in metadata:
        raise FreezeIntegrityError("checkpoint lacks its adaptive human-cell provenance")
    return metadata


def checkpoint_hashes(path: str | Path) -> CheckpointHashes:
    checkpoint = Path(path).resolve()
    raw = checkpoint.read_bytes()
    try:
        with np.load(checkpoint, allow_pickle=False) as state:
            metadata = _checkpoint_metadata(state)
            if "phrase_offsets" not in state.files or "phrase_values" not in state.files:
                raise FreezeIntegrityError("checkpoint lacks its embedded tokenizer")
            state_digest = sha256()
            for name in sorted(state.files):
                encoded_name = name.encode("utf-8")
                state_digest.update(len(encoded_name).to_bytes(4, "big"))
                state_digest.update(encoded_name)
                if name == "metadata":
                    value = canonical_json_bytes(metadata)
                else:
                    value = _normalized_array_bytes(state[name])
                state_digest.update(len(value).to_bytes(8, "big"))
                state_digest.update(value)
            tokenizer_record = {
                "phrase_offsets": sha256(
                    _normalized_array_bytes(state["phrase_offsets"])
                ).hexdigest(),
                "phrase_values": sha256(
                    _normalized_array_bytes(state["phrase_values"])
                ).hexdigest(),
            }
    except (OSError, ValueError) as error:
        raise FreezeIntegrityError(
            "checkpoint cannot be read as an immutable cell state"
        ) from error

    volatile_metadata = {
        "training_digest",
        "grown_transitions",
        "growth_examples",
        "local_updates",
        "runtime_active_reservoir_cells",
    }
    config_record = {
        key: value for key, value in metadata.items() if key not in volatile_metadata
    }
    return CheckpointHashes(
        checkpoint_format=str(metadata["format"]),
        checkpoint_sha256=sha256(raw).hexdigest(),
        checkpoint_state_sha256=state_digest.hexdigest(),
        model_config_sha256=canonical_sha256(config_record),
        tokenizer_sha256=canonical_sha256(tokenizer_record),
        checkpoint_bytes=len(raw),
    )


def _git(source_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-c", "core.excludesFile=", *arguments],
            cwd=source_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeIntegrityError("source root is not a readable Git worktree") from error
    return result.stdout


def _installed_execution_source_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in _EXECUTION_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            raise FreezeIntegrityError(
                f"installed Human Brain execution source cannot load: {name}"
            ) from error
        source = getattr(module, "__file__", None)
        if not source:
            raise FreezeIntegrityError(
                f"installed Human Brain execution source has no file: {name}"
            )
        paths.append(Path(source).resolve())
    return tuple(paths)


def _require_execution_sources_in_tree(source_root: Path) -> None:
    root = source_root.resolve()
    for source_path in _installed_execution_source_paths():
        try:
            relative = source_path.relative_to(root)
        except ValueError as error:
            raise FreezeIntegrityError(
                "installed Human Brain execution source is outside source_root: "
                f"{source_path}"
            ) from error
        try:
            _git(root, "ls-files", "--error-unmatch", "--", relative.as_posix())
        except FreezeIntegrityError as error:
            raise FreezeIntegrityError(
                "installed Human Brain execution source is not Git-tracked: "
                f"{relative.as_posix()}"
            ) from error


def _clean_source_snapshot(source_root: str | Path) -> tuple[str, str]:
    root = Path(source_root).resolve()
    _require_execution_sources_in_tree(root)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise FreezeIntegrityError("source tree is dirty; commit or remove every change")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree_listing = _git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    return head, sha256(tree_listing).hexdigest()


def _atomic_create_json(path: Path, payload: object) -> None:
    atomic_create_bytes(path, canonical_json_bytes(payload))


def _outside_source(output_path: Path, source_root: Path) -> None:
    try:
        output_path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return
    raise FreezeIntegrityError(
        "freeze artifacts must remain outside the model-visible source tree"
    )


def _module_source_sha256(module_name: str) -> str:
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if not source:
        raise FreezeIntegrityError(f"execution module has no source: {module_name}")
    return sha256(Path(source).read_bytes()).hexdigest()


def _completion_identities() -> tuple[CompilerIdentity, str]:
    identity = RawPythonCompiler().identity
    if not isinstance(identity, CompilerIdentity):
        raise FreezeIntegrityError("compiler identity has the wrong type")
    return identity, _module_source_sha256("humanbrain.simple_swe.adapter")


def _simulation_backend_identity(
    checkpoint_path: str | Path, checkpoint_format: str
) -> SimulationBackendIdentity:
    policy = os.environ.get("HUMANBRAIN_DEVICE", "auto").strip().lower()
    if checkpoint_format == "human-cell-brain-checkpoint-v1":
        return SimulationBackendIdentity(
            backend="python",
            device="cpu",
            device_policy=policy,
            source_digest=None,
            artifact_path=None,
            artifact_sha256=None,
            architecture=None,
        )
    if checkpoint_format != "recurrent-human-cell-checkpoint-v3":
        raise FreezeIntegrityError("unsupported cell checkpoint format")
    from humanbrain.recurrent_language import RecurrentCellLanguageModel

    try:
        model = RecurrentCellLanguageModel.load(checkpoint_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise FreezeIntegrityError("simulation backend cannot load checkpoint") from error
    backend = model._circuit.simulation_backend
    device = model._circuit.simulation_device
    if backend == "cpu":
        from humanbrain.native import loaded_build_info

        build = loaded_build_info()
        artifact = build.library_path.resolve()
        architecture = None
        source_digest = build.source_digest
    elif backend == "cuda":
        from humanbrain.cuda import build_cuda

        build = build_cuda()
        artifact = build.cubin_path.resolve()
        architecture = build.architecture
        source_digest = build.source_digest
    else:
        raise FreezeIntegrityError(f"unknown recurrent simulation backend: {backend}")
    if not artifact.is_file():
        raise FreezeIntegrityError("simulation backend artifact is missing")
    return SimulationBackendIdentity(
        backend=backend,
        device=device,
        device_policy=policy,
        source_digest=source_digest,
        artifact_path=str(artifact),
        artifact_sha256=sha256(artifact.read_bytes()).hexdigest(),
        architecture=architecture,
    )


def _verify_simulation_backend_artifact(identity: SimulationBackendIdentity) -> None:
    policy = os.environ.get("HUMANBRAIN_DEVICE", "auto").strip().lower()
    if policy != identity.device_policy:
        raise FreezeIntegrityError("simulation device policy drift after freeze")
    if identity.backend == "python":
        if any(
            value is not None
            for value in (
                identity.source_digest,
                identity.artifact_path,
                identity.artifact_sha256,
                identity.architecture,
            )
        ):
            raise FreezeIntegrityError("Python backend has an unexpected native artifact")
        return
    if identity.backend == "cpu":
        from humanbrain.native import loaded_build_info

        build = loaded_build_info()
        artifact = build.library_path.resolve()
        current_source_digest = build.source_digest
        current_architecture = None
    elif identity.backend == "cuda":
        from humanbrain.cuda import build_cuda

        build = build_cuda()
        artifact = build.cubin_path.resolve()
        current_source_digest = build.source_digest
        current_architecture = build.architecture
    else:
        raise FreezeIntegrityError("unknown frozen simulation backend")
    if (
        str(artifact) != identity.artifact_path
        or current_source_digest != identity.source_digest
        or current_architecture != identity.architecture
        or not artifact.is_file()
        or sha256(artifact.read_bytes()).hexdigest() != identity.artifact_sha256
    ):
        raise FreezeIntegrityError("simulation backend artifact drift after freeze")


def _inference_config_sha256(
    protocol: ProtocolSpec,
    checkpoint: CheckpointHashes,
    compiler_identity: CompilerIdentity,
    adapter_sha256: str,
    commitments: dict[str, object],
) -> str:
    if not isinstance(compiler_identity, CompilerIdentity):
        raise FreezeIntegrityError("compiler identity has the wrong type")
    return canonical_sha256(
        {
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "max_new_tokens": protocol.max_new_tokens,
            "checkpoint_format": checkpoint.checkpoint_format,
            "model_config_sha256": checkpoint.model_config_sha256,
            "tokenizer_sha256": checkpoint.tokenizer_sha256,
            "completion_adapter_id": COMPLETION_ADAPTER_ID,
            "completion_adapter_sha256": adapter_sha256,
            "compiler_id": compiler_identity.compiler_id,
            "compiler_sha256": compiler_identity.compiler_sha256,
            "compiler_mode": compiler_identity.mode,
            "compiler_algorithm_source": compiler_identity.algorithm_source,
            **commitments,
        }
    )


def _execution_components(
    *,
    protocol: ProtocolSpec,
    checkpoint_path: str | Path,
    checkpoint: CheckpointHashes,
    evaluation_root: Path,
    python_executable: str | Path | None,
    backend: SimulationBackendIdentity | None = None,
) -> tuple[CompilerIdentity, str, dict[str, object]]:
    identity, adapter_sha256 = _completion_identities()
    if not isinstance(identity, CompilerIdentity):
        raise FreezeIntegrityError("compiler identity has the wrong type")
    executable = resolve_python_executable(python_executable)
    try:
        runtime = python_runtime_identity(executable, protocol.raw.get("language"))
    except SandboxConfigurationError as error:
        raise FreezeIntegrityError(str(error)) from error
    selected_backend = backend or _simulation_backend_identity(
        checkpoint_path, checkpoint.checkpoint_format
    )
    commitments: dict[str, object] = {
        "evaluation_root": str(evaluation_root),
        "canonical_runner_id": CANONICAL_RUNNER_ID,
        "canonical_runner_sha256": _module_source_sha256(
            "humanbrain.simple_swe.runner"
        ),
        "sandbox_executor_id": SANDBOX_EXECUTOR_ID,
        "sandbox_executor_sha256": _module_source_sha256(
            "humanbrain.simple_swe.sandbox"
        ),
        "sandbox_worker_sha256": _module_source_sha256(
            "humanbrain.simple_swe._worker"
        ),
        "sandbox_python_executable": str(executable),
        "sandbox_python_implementation": runtime["implementation"],
        "sandbox_python_version": runtime["python_version"],
        "sandbox_python_executable_sha256": runtime["executable_sha256"],
        "sandbox_python_flags": list(SANDBOX_PYTHON_FLAGS),
        "simulation_backend": selected_backend.backend,
        "simulation_device": selected_backend.device,
        "simulation_device_policy": selected_backend.device_policy,
        "simulation_source_digest": selected_backend.source_digest,
        "simulation_artifact_path": selected_backend.artifact_path,
        "simulation_artifact_sha256": selected_backend.artifact_sha256,
        "simulation_architecture": selected_backend.architecture,
    }
    return identity, adapter_sha256, commitments


def create_freeze_manifest(
    *,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
    output_path: str | Path,
    python_executable: str | Path | None = None,
) -> FreezeManifest:
    protocol = load_protocol(protocol_path)
    root = Path(source_root).resolve()
    output = Path(output_path).resolve()
    if output.name != "freeze.json":
        raise FreezeIntegrityError("canonical freeze manifest must be named freeze.json")
    _outside_source(output, root)
    head, tree_sha256 = _clean_source_snapshot(root)
    checkpoint = checkpoint_hashes(checkpoint_path)
    identity, adapter_sha256, commitments = _execution_components(
        protocol=protocol,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        evaluation_root=output.parent,
        python_executable=python_executable,
    )
    if not isinstance(identity, CompilerIdentity):
        raise FreezeIntegrityError("compiler identity has the wrong type")
    manifest = FreezeManifest(
        schema=SCHEMA,
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol.sha256,
        checkpoint_format=checkpoint.checkpoint_format,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        checkpoint_state_sha256=checkpoint.checkpoint_state_sha256,
        model_config_sha256=checkpoint.model_config_sha256,
        tokenizer_sha256=checkpoint.tokenizer_sha256,
        checkpoint_bytes=checkpoint.checkpoint_bytes,
        evaluation_root=str(output.parent),
        completion_adapter_id=COMPLETION_ADAPTER_ID,
        completion_adapter_sha256=adapter_sha256,
        compiler_id=identity.compiler_id,
        compiler_sha256=identity.compiler_sha256,
        compiler_mode=identity.mode,
        compiler_algorithm_source=identity.algorithm_source,
        canonical_runner_id=str(commitments["canonical_runner_id"]),
        canonical_runner_sha256=str(commitments["canonical_runner_sha256"]),
        sandbox_executor_id=str(commitments["sandbox_executor_id"]),
        sandbox_executor_sha256=str(commitments["sandbox_executor_sha256"]),
        sandbox_worker_sha256=str(commitments["sandbox_worker_sha256"]),
        sandbox_python_executable=str(commitments["sandbox_python_executable"]),
        sandbox_python_implementation=str(
            commitments["sandbox_python_implementation"]
        ),
        sandbox_python_version=str(commitments["sandbox_python_version"]),
        sandbox_python_executable_sha256=str(
            commitments["sandbox_python_executable_sha256"]
        ),
        sandbox_python_flags=SANDBOX_PYTHON_FLAGS,
        simulation_backend=str(commitments["simulation_backend"]),
        simulation_device=str(commitments["simulation_device"]),
        simulation_device_policy=str(commitments["simulation_device_policy"]),
        simulation_source_digest=commitments["simulation_source_digest"],
        simulation_artifact_path=commitments["simulation_artifact_path"],
        simulation_artifact_sha256=commitments["simulation_artifact_sha256"],
        simulation_architecture=commitments["simulation_architecture"],
        inference_config_sha256=_inference_config_sha256(
            protocol, checkpoint, identity, adapter_sha256, commitments
        ),
        source_head=head,
        source_tree_sha256=tree_sha256,
        created_utc=datetime.now(UTC).isoformat(),
        one_checkpoint_for_all_tasks=True,
        reload_before_each_task=True,
        verify_state_before_and_after=True,
        no_learning_after_freeze=True,
        no_task_specific_replay=True,
        one_canonical_evidence_root=True,
        external_anchor_before_task_generation=True,
    )
    _atomic_create_json(output, asdict(manifest))
    return manifest


def _load_manifest(path: str | Path) -> FreezeManifest:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
        if raw != canonical_json_bytes(value):
            raise FreezeIntegrityError("freeze manifest is not canonical JSON")
        value["sandbox_python_flags"] = tuple(value["sandbox_python_flags"])
        manifest = FreezeManifest(**value)
    except FreezeIntegrityError:
        raise
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise FreezeIntegrityError("freeze manifest is invalid") from error
    if manifest.schema != SCHEMA:
        raise FreezeIntegrityError("unsupported freeze manifest schema")
    if not all(
        (
            manifest.one_checkpoint_for_all_tasks,
            manifest.reload_before_each_task,
            manifest.verify_state_before_and_after,
            manifest.no_learning_after_freeze,
            manifest.no_task_specific_replay,
            manifest.one_canonical_evidence_root,
            manifest.external_anchor_before_task_generation,
        )
    ):
        raise FreezeIntegrityError("freeze manifest weakens a required invariant")
    if manifest.sandbox_python_flags != SANDBOX_PYTHON_FLAGS:
        raise FreezeIntegrityError("freeze manifest changed sandbox Python flags")
    if (
        manifest.canonical_runner_id != CANONICAL_RUNNER_ID
        or manifest.sandbox_executor_id != SANDBOX_EXECUTOR_ID
    ):
        raise FreezeIntegrityError("freeze manifest changed production execution mode")
    return manifest


def verify_freeze_manifest(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
) -> FreezeManifest:
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = _load_manifest(resolved_manifest_path)
    canonical_manifest_path = Path(manifest.evaluation_root).resolve() / "freeze.json"
    if resolved_manifest_path != canonical_manifest_path:
        raise FreezeIntegrityError("freeze manifest moved from its canonical evidence root")
    protocol: ProtocolSpec = load_protocol(protocol_path)
    if (
        protocol.protocol_id != manifest.protocol_id
        or protocol.sha256 != manifest.protocol_sha256
    ):
        raise FreezeIntegrityError("protocol hash drift after model freeze")
    head, source_tree_sha256 = _clean_source_snapshot(source_root)
    if head != manifest.source_head:
        raise FreezeIntegrityError("source commit drift after model freeze")
    if source_tree_sha256 != manifest.source_tree_sha256:
        raise FreezeIntegrityError("source tree hash drift after model freeze")
    current = checkpoint_hashes(checkpoint_path)
    if current.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise FreezeIntegrityError("checkpoint hash drift after model freeze")
    if current.checkpoint_state_sha256 != manifest.checkpoint_state_sha256:
        raise FreezeIntegrityError("checkpoint state hash drift after model freeze")
    if current.model_config_sha256 != manifest.model_config_sha256:
        raise FreezeIntegrityError("model config hash drift after model freeze")
    if current.tokenizer_sha256 != manifest.tokenizer_sha256:
        raise FreezeIntegrityError("tokenizer hash drift after model freeze")
    if current.checkpoint_format != manifest.checkpoint_format:
        raise FreezeIntegrityError("checkpoint format drift after model freeze")
    backend = SimulationBackendIdentity(
        backend=manifest.simulation_backend,
        device=manifest.simulation_device,
        device_policy=manifest.simulation_device_policy,
        source_digest=manifest.simulation_source_digest,
        artifact_path=manifest.simulation_artifact_path,
        artifact_sha256=manifest.simulation_artifact_sha256,
        architecture=manifest.simulation_architecture,
    )
    _verify_simulation_backend_artifact(backend)
    identity, adapter_sha256, commitments = _execution_components(
        protocol=protocol,
        checkpoint_path=checkpoint_path,
        checkpoint=current,
        evaluation_root=canonical_manifest_path.parent,
        python_executable=manifest.sandbox_python_executable,
        backend=backend,
    )
    if not isinstance(identity, CompilerIdentity):
        raise FreezeIntegrityError("compiler identity has the wrong type")
    if (
        manifest.completion_adapter_id != COMPLETION_ADAPTER_ID
        or manifest.completion_adapter_sha256 != adapter_sha256
    ):
        raise FreezeIntegrityError("completion adapter identity drift after model freeze")
    if (
        manifest.compiler_id != identity.compiler_id
        or manifest.compiler_sha256 != identity.compiler_sha256
        or manifest.compiler_mode != identity.mode
        or manifest.compiler_algorithm_source != identity.algorithm_source
    ):
        raise FreezeIntegrityError("source compiler identity drift after model freeze")
    frozen_commitments = {
        name: (
            list(manifest.sandbox_python_flags)
            if name == "sandbox_python_flags"
            else getattr(manifest, name)
        )
        for name in _EXECUTION_COMMITMENT_FIELDS
    }
    if commitments != frozen_commitments:
        raise FreezeIntegrityError("frozen execution/runtime commitment drift")
    inference_sha256 = _inference_config_sha256(
        protocol, current, identity, adapter_sha256, commitments
    )
    if inference_sha256 != manifest.inference_config_sha256:
        raise FreezeIntegrityError("inference config hash drift after model freeze")
    return manifest


__all__ = [
    "CheckpointHashes",
    "FreezeIntegrityError",
    "FreezeManifest",
    "checkpoint_hashes",
    "create_freeze_manifest",
    "verify_freeze_manifest",
]
