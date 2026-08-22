from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from humanbrain.language import CellLanguageModel
from humanbrain.recurrent_language import RecurrentCellLanguageModel
from humanbrain.simple_swe.compiler import (
    COMPLETION_ADAPTER_ID,
    CompileRequest,
    RawPythonCompiler,
)
from humanbrain.simple_swe.freeze import (
    FreezeIntegrityError,
    FreezeManifest,
    checkpoint_hashes,
    verify_freeze_manifest,
)
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import canonical_sha256
from humanbrain.typed_program_types import TypedSignature


@dataclass(frozen=True)
class CompletionRequest:
    """The only public task information available during direct synthesis."""

    prompt: str
    signature: TypedSignature

    def __post_init__(self) -> None:
        if not isinstance(self.signature, TypedSignature):
            raise TypeError("completion request signature must be typed")


@dataclass(frozen=True)
class CompletionAudit:
    adapter_id: str
    adapter_sha256: str
    compiler_id: str
    compiler_sha256: str
    compiler_mode: str
    compiler_algorithm_source: str
    checkpoint_sha256: str
    learned_checkpoint_state_before: str
    learned_checkpoint_state_after: str
    prompt_sha256: str
    emitted_token_ids: tuple[int, ...]
    emitted_token_count: int
    emitted_token_sha256: str
    emitted_text_sha256: str
    source_sha256: str
    direct_synthesis_no_tools: bool


@dataclass(frozen=True)
class SynthesizedCompletion:
    source: str
    emitted_text: str
    audit: CompletionAudit


def completion_adapter_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _load_cell_model(
    checkpoint_path: Path, checkpoint_format: str
) -> CellLanguageModel | RecurrentCellLanguageModel:
    if checkpoint_format == "human-cell-brain-checkpoint-v1":
        return CellLanguageModel.load(checkpoint_path)
    if checkpoint_format == "recurrent-human-cell-checkpoint-v3":
        model = RecurrentCellLanguageModel.load(checkpoint_path)
        if not model._circuit.learning_frozen:
            raise FreezeIntegrityError("recurrent checkpoint learning is not frozen")
        return model
    raise FreezeIntegrityError("unsupported cell checkpoint format")


def _verify_loaded_backend(
    model: CellLanguageModel | RecurrentCellLanguageModel,
    manifest: FreezeManifest,
) -> None:
    if isinstance(model, CellLanguageModel):
        backend, device = "python", "cpu"
    else:
        backend = model._circuit.simulation_backend
        device = model._circuit.simulation_device
    if backend != manifest.simulation_backend or device != manifest.simulation_device:
        raise FreezeIntegrityError("loaded cell simulation backend differs from freeze")


def _emit_cell_tokens(
    model: CellLanguageModel | RecurrentCellLanguageModel,
    prompt: str,
    max_new_tokens: int,
) -> tuple[tuple[int, ...], str]:
    if isinstance(model, CellLanguageModel):
        tokenizer = model._tokenizer
        context = tokenizer.encode(prompt)
        if not context:
            raise ValueError("completion prompt cannot be empty")
        generated: list[int] = []
        for _ in range(max_new_tokens):
            race = model._cortex.race(context)
            context.append(race.winner)
            generated.append(race.winner)
        return tuple(generated), tokenizer.decode(generated)

    tokenizer = model._tokenizer
    context = tokenizer.encode(prompt)
    if not context:
        raise ValueError("completion prompt cannot be empty")
    circuit = model._circuit
    circuit.reset_sequence()
    for token_id in context:
        circuit.observe(token_id)
    generated = []
    for _ in range(max_new_tokens):
        race = circuit.race()
        generated.append(race.winner)
        circuit.observe(race.winner)
    return tuple(generated), tokenizer.decode(generated)


class CellCompletionAdapter:
    """Restore, run, and audit direct cell-only synthesis for one public prompt."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        protocol_path: str | Path,
        checkpoint_path: str | Path,
        source_root: str | Path,
    ) -> None:
        self._manifest_path = Path(manifest_path).resolve()
        self._protocol_path = Path(protocol_path).resolve()
        self._checkpoint_path = Path(checkpoint_path).resolve()
        self._source_root = Path(source_root).resolve()
        self._compiler = RawPythonCompiler()
        self._protocol = load_protocol(self._protocol_path)
        self._manifest = verify_freeze_manifest(
            manifest_path=self._manifest_path,
            protocol_path=self._protocol_path,
            checkpoint_path=self._checkpoint_path,
            source_root=self._source_root,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._manifest.checkpoint_sha256

    def synthesize(self, request: CompletionRequest) -> SynthesizedCompletion:
        manifest = verify_freeze_manifest(
            manifest_path=self._manifest_path,
            protocol_path=self._protocol_path,
            checkpoint_path=self._checkpoint_path,
            source_root=self._source_root,
        )
        before = checkpoint_hashes(self._checkpoint_path)
        if before.checkpoint_state_sha256 != manifest.checkpoint_state_sha256:
            raise FreezeIntegrityError("checkpoint state before task differs from freeze")
        model = _load_cell_model(self._checkpoint_path, manifest.checkpoint_format)
        _verify_loaded_backend(model, manifest)
        token_ids, emitted_text = _emit_cell_tokens(
            model, request.prompt, self._protocol.max_new_tokens
        )
        compile_request = CompileRequest(
            emitted_text=emitted_text,
            emitted_token_ids=token_ids,
            entry_point=request.signature.entry_point,
            parameter_names=tuple(
                parameter.name for parameter in request.signature.parameters
            ),
        )
        source = self._compiler.compile(compile_request)
        if not isinstance(source, str):
            raise TypeError("deterministic source compiler must return Python text")
        with TemporaryDirectory(prefix="humanbrain-cell-state-audit-") as directory:
            after_path = Path(directory) / "after.npz"
            model.save(after_path)
            after = checkpoint_hashes(after_path)
        # Checkpoints persist learned cell/synapse structure, configuration, and
        # tokenizer state. Runtime membrane and eligibility traces are deliberately
        # absent from both checkpoint formats, so recurrent inference may evolve
        # those transient traces without weakening this learned-state comparison.
        if after.checkpoint_state_sha256 != before.checkpoint_state_sha256:
            raise FreezeIntegrityError("persistent cell/synapse state changed during synthesis")
        verify_freeze_manifest(
            manifest_path=self._manifest_path,
            protocol_path=self._protocol_path,
            checkpoint_path=self._checkpoint_path,
            source_root=self._source_root,
        )
        identity = self._compiler.identity
        audit = CompletionAudit(
            adapter_id=COMPLETION_ADAPTER_ID,
            adapter_sha256=completion_adapter_sha256(),
            compiler_id=identity.compiler_id,
            compiler_sha256=identity.compiler_sha256,
            compiler_mode=identity.mode,
            compiler_algorithm_source=identity.algorithm_source,
            checkpoint_sha256=before.checkpoint_sha256,
            learned_checkpoint_state_before=before.checkpoint_state_sha256,
            learned_checkpoint_state_after=after.checkpoint_state_sha256,
            prompt_sha256=sha256(request.prompt.encode("utf-8")).hexdigest(),
            emitted_token_ids=token_ids,
            emitted_token_count=len(token_ids),
            emitted_token_sha256=canonical_sha256(list(token_ids)),
            emitted_text_sha256=sha256(emitted_text.encode("utf-8")).hexdigest(),
            source_sha256=sha256(source.encode("utf-8")).hexdigest(),
            direct_synthesis_no_tools=True,
        )
        return SynthesizedCompletion(source=source, emitted_text=emitted_text, audit=audit)


__all__ = [
    "CellCompletionAdapter",
    "CompletionAudit",
    "CompletionRequest",
    "SynthesizedCompletion",
    "completion_adapter_sha256",
]
