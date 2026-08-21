from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol, runtime_checkable

COMPLETION_ADAPTER_ID = "humanbrain-cell-completion-adapter-v1"
COMPILER_FACTORY_ID = "humanbrain-frozen-compiler-registry-v1"


@dataclass(frozen=True)
class CompilerIdentity:
    compiler_id: str
    compiler_sha256: str
    mode: str
    algorithm_source: str

    def __post_init__(self) -> None:
        if not self.compiler_id or not self.mode:
            raise ValueError("compiler identity and mode must be non-empty")
        if (
            len(self.compiler_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.compiler_sha256)
        ):
            raise ValueError("compiler_sha256 must be lowercase SHA-256 hex")
        if self.algorithm_source != "emitted-tokens-only":
            raise ValueError("compiler may derive algorithms only from emitted tokens")


@dataclass(frozen=True)
class CompileRequest:
    emitted_text: str
    emitted_token_ids: tuple[int, ...]
    entry_point: str
    parameter_names: tuple[str, ...]


@runtime_checkable
class SourceCompiler(Protocol):
    @property
    def identity(self) -> CompilerIdentity: ...

    def compile(self, request: CompileRequest) -> str: ...


class RawPythonCompiler:
    """Identity compiler: every algorithmic byte is emitted by cell races."""

    @property
    def identity(self) -> CompilerIdentity:
        return CompilerIdentity(
            compiler_id="raw-python-cell-emissions-v1",
            compiler_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            mode="raw-python",
            algorithm_source="emitted-tokens-only",
        )

    def compile(self, request: CompileRequest) -> str:
        return request.emitted_text


def compiler_factory_sha256() -> str:
    """Commit the tracked registry and every currently selectable compiler."""
    return sha256(Path(__file__).read_bytes()).hexdigest()


def create_source_compiler() -> RawPythonCompiler:
    """Construct the sole compiler registered by this frozen protocol version."""
    return RawPythonCompiler()


__all__ = [
    "COMPILER_FACTORY_ID",
    "COMPLETION_ADAPTER_ID",
    "CompileRequest",
    "CompilerIdentity",
    "RawPythonCompiler",
    "SourceCompiler",
    "compiler_factory_sha256",
    "create_source_compiler",
]
