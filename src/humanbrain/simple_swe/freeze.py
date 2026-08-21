from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from humanbrain.simple_swe.protocol import ProtocolSpec, load_protocol
from humanbrain.simple_swe.tasks import canonical_json_bytes, canonical_sha256


SCHEMA = "humanbrain.simple-swe-freeze.v1"
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
    "humanbrain.simple_swe.reveal",
    "humanbrain.simple_swe.runner",
    "humanbrain.simple_swe.sandbox",
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
    completion_adapter_id: str
    completion_adapter_sha256: str
    compiler_id: str
    compiler_sha256: str
    compiler_mode: str
    compiler_algorithm_source: str
    inference_config_sha256: str
    source_head: str
    source_tree_sha256: str
    created_utc: str
    one_checkpoint_for_all_tasks: bool
    reload_before_each_task: bool
    verify_state_before_and_after: bool
    no_learning_after_freeze: bool
    no_task_specific_replay: bool


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
        raise FreezeIntegrityError("checkpoint cannot be read as an immutable cell state") from error

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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _outside_source(output_path: Path, source_root: Path) -> None:
    try:
        output_path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return
    raise FreezeIntegrityError(
        "freeze artifacts must remain outside the model-visible source tree"
    )


def _completion_identities(compiler_identity: object | None) -> tuple[object, str]:
    from humanbrain.simple_swe.compiler import (
        COMPLETION_ADAPTER_ID,
        CompilerIdentity,
        RawPythonCompiler,
    )

    identity = compiler_identity or RawPythonCompiler().identity
    if not isinstance(identity, CompilerIdentity):
        raise FreezeIntegrityError("compiler identity has the wrong type")
    adapter_path = Path(__file__).with_name("adapter.py")
    if not adapter_path.is_file():
        raise FreezeIntegrityError("completion adapter source is missing")
    adapter_sha256 = sha256(adapter_path.read_bytes()).hexdigest()
    return identity, adapter_sha256


def _inference_config_sha256(
    protocol: ProtocolSpec,
    checkpoint: CheckpointHashes,
    compiler_identity: object,
    adapter_sha256: str,
) -> str:
    from humanbrain.simple_swe.compiler import COMPLETION_ADAPTER_ID, CompilerIdentity

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
        }
    )


def create_freeze_manifest(
    *,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
    output_path: str | Path,
    compiler_identity: object | None = None,
) -> FreezeManifest:
    protocol = load_protocol(protocol_path)
    root = Path(source_root).resolve()
    output = Path(output_path).resolve()
    _outside_source(output, root)
    head, tree_sha256 = _clean_source_snapshot(root)
    checkpoint = checkpoint_hashes(checkpoint_path)
    identity, adapter_sha256 = _completion_identities(compiler_identity)
    from humanbrain.simple_swe.compiler import COMPLETION_ADAPTER_ID, CompilerIdentity

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
        completion_adapter_id=COMPLETION_ADAPTER_ID,
        completion_adapter_sha256=adapter_sha256,
        compiler_id=identity.compiler_id,
        compiler_sha256=identity.compiler_sha256,
        compiler_mode=identity.mode,
        compiler_algorithm_source=identity.algorithm_source,
        inference_config_sha256=_inference_config_sha256(
            protocol, checkpoint, identity, adapter_sha256
        ),
        source_head=head,
        source_tree_sha256=tree_sha256,
        created_utc=datetime.now(UTC).isoformat(),
        one_checkpoint_for_all_tasks=True,
        reload_before_each_task=True,
        verify_state_before_and_after=True,
        no_learning_after_freeze=True,
        no_task_specific_replay=True,
    )
    _atomic_create_json(output, asdict(manifest))
    return manifest


def _load_manifest(path: str | Path) -> FreezeManifest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = FreezeManifest(**value)
    except (OSError, TypeError, json.JSONDecodeError) as error:
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
        )
    ):
        raise FreezeIntegrityError("freeze manifest weakens a required invariant")
    return manifest


def verify_freeze_manifest(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    checkpoint_path: str | Path,
    source_root: str | Path,
    compiler_identity: object | None = None,
) -> FreezeManifest:
    manifest = _load_manifest(manifest_path)
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
    identity, adapter_sha256 = _completion_identities(compiler_identity)
    from humanbrain.simple_swe.compiler import COMPLETION_ADAPTER_ID, CompilerIdentity

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
    inference_sha256 = _inference_config_sha256(
        protocol, current, identity, adapter_sha256
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
