"""Authenticated source-only Rust action registry.

This module authorizes two unlabeled Rust sources only where their code-derived
identifier factors exactly overlap authenticated consolidated knowledge factors.
It does not compile or execute Rust, and it releases only the action selected by
an authenticated physical knowledge-action recall produced through this registry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import TypeVar

from experiments.r3_pilot import common_knowledge_action_memory_v1 as action_memory

from humanbrain import consolidated_knowledge_cell_projection as consolidated_cells
from humanbrain.knowledge_consolidation import (
    LEXICAL_FACTOR_SCHEMA,
    KnowledgeConsolidationV1,
    lexical_factors_from_text_v1,
)
from humanbrain.knowledge_corpus import KnowledgeCorpus

SCHEMA = "humanbrain.rust-action-registry.v1"
BOUND_MEMORY_SCHEMA = "humanbrain.bound-rust-action-memory.v1"
CLAIM_BOUNDARY = (
    "Authenticated task-independent lexical authorization of two opaque Rust actions; "
    "trusted-host source custody only; not hostile Python-object isolation, Rust "
    "correctness, semantic understanding, coding, or transfer evidence."
)

ACTION_COUNT = 2
MAX_SOURCE_BYTES = 64 * 1024
MAX_SOURCE_FACTORS = 4_096
MAX_ISSUED_RECALLS = 64
RUST_EDITION = "2021"
RUST_STYLE_EDITION = "2021"

_RUSTFMT_ARGS = (
    "--emit",
    "stdout",
    "--edition",
    RUST_EDITION,
    "--style-edition",
    RUST_STYLE_EDITION,
    "--color",
    "never",
)
_RUSTFMT_POLICY_ARGS = (
    *_RUSTFMT_ARGS,
    "--config-path",
    "<isolated-rustfmt.toml>",
)
_RUSTFMT_CONFIG_BYTES = b'edition = "2021"\nstyle_edition = "2021"\nnewline_style = "Unix"\n'

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ACTION_HANDLE = re.compile(r"ra_[0-9a-f]{64}")
_FACTOR = re.compile(r"tok:[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*")
_NUMBER_LITERAL = re.compile(
    r"(?:"
    r"0[bB][01_]+|"
    r"0[oO][0-7_]+|"
    r"0[xX][0-9a-fA-F_]+|"
    r"[0-9][0-9_]*\.(?![._A-Za-z0-9])|"
    r"[0-9][0-9_]*(?:\.[0-9][0-9_]*)?(?:[eE][+-]?[0-9_]+)?"
    r")(?:[A-Za-z][A-Za-z0-9_]*)?"
)

_TOOL_NAMESPACE = b"rust-action-registry-rustfmt-tool-v1"
_POLICY_NAMESPACE = b"rust-action-registry-policy-v1"
_STRUCTURAL_NAMESPACE = b"rust-action-registry-structural-v1"
_FACTOR_SET_NAMESPACE = b"rust-action-registry-source-factors-v1"
_HANDLE_NAMESPACE = b"rust-action-registry-handle-v1"
_SOURCE_PROJECTION_NAMESPACE = b"rust-action-registry-source-projection-v1"
_AUTHORIZATION_NAMESPACE = b"rust-action-registry-authorization-v1"
_BINDING_NAMESPACE = b"rust-action-registry-factor-binding-v1"
_SELECTOR_NAMESPACE = b"rust-action-registry-selector-v1"
_RECEIPT_NAMESPACE = b"rust-action-registry-receipt-v1"
_BOUND_COGNITION_NAMESPACE = b"rust-action-registry-bound-cognition-v1"
_BOUND_AUTHORIZATION_NAMESPACE = b"rust-action-registry-bound-authorization-v1"
_BOUND_RECEIPT_NAMESPACE = b"rust-action-registry-bound-receipt-v1"
_MATERIALIZATION_NAMESPACE = b"rust-action-registry-materialization-v1"

_T = TypeVar("_T")
_CONSTRUCTION_TOKEN = object()


class RustActionRegistryError(ValueError):
    """Registry source, authorization, selection, or integrity failed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _basic_process_environment() -> dict[str, str]:
    allowed = (
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
    )
    return {name: os.environ[name] for name in allowed if name in os.environ}


def _rustup_environment() -> dict[str, str]:
    result = _basic_process_environment()
    for name in ("HOME", "USERPROFILE", "RUSTUP_HOME", "CARGO_HOME"):
        if name in os.environ:
            result[name] = os.environ[name]
    return result


@dataclass(frozen=True, slots=True)
class _RustfmtIdentityV1:
    path: str
    version: str
    executable_sha256: str
    sha256: str


def _tool_payload(identity: _RustfmtIdentityV1) -> dict[str, object]:
    return {
        "path": identity.path,
        "version": identity.version,
        "executable_sha256": identity.executable_sha256,
    }


def _read_rustfmt_identity(path: Path) -> _RustfmtIdentityV1:
    try:
        resolved = path.resolve(strict=True)
        executable_sha256 = sha256(resolved.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="humanbrain-rustfmt-version-") as root:
            completed = subprocess.run(
                (str(resolved), "--version"),
                cwd=root,
                env=_basic_process_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=10,
            )
        version = completed.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RustActionRegistryError("rustfmt identity could not be authenticated") from exc
    if completed.returncode != 0 or not version or completed.stderr:
        raise RustActionRegistryError("rustfmt identity could not be authenticated")
    draft = _RustfmtIdentityV1(
        path=str(resolved),
        version=version,
        executable_sha256=executable_sha256,
        sha256="",
    )
    return _RustfmtIdentityV1(
        **{
            **_tool_payload(draft),
            "sha256": _digest(_TOOL_NAMESPACE, _tool_payload(draft)),
        }
    )


def _discover_rustfmt() -> _RustfmtIdentityV1:
    rustup = shutil.which("rustup")
    rustfmt = shutil.which("rustfmt")
    if rustfmt is None:
        raise RustActionRegistryError("one installed rustfmt executable is required")
    concrete = Path(rustfmt)
    if rustup is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="humanbrain-rustfmt-discovery-") as root:
                completed = subprocess.run(
                    (rustup, "which", "rustfmt"),
                    cwd=root,
                    env=_rustup_environment(),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            output = completed.stdout.decode("utf-8", errors="strict").strip()
            if completed.returncode != 0 or not output or completed.stderr:
                raise RustActionRegistryError("concrete rustfmt executable could not be resolved")
            concrete = Path(output)
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise RustActionRegistryError(
                "concrete rustfmt executable could not be resolved"
            ) from exc
    return _read_rustfmt_identity(concrete)


def _policy_sha256(tool: _RustfmtIdentityV1) -> str:
    return _digest(
        _POLICY_NAMESPACE,
        {
            "action_count": ACTION_COUNT,
            "authorization": "exact factor string intersection only",
            "chapter_path_and_declared_symbols": "audit only",
            "lexical_factor_schema": LEXICAL_FACTOR_SCHEMA,
            "literal_policy": "mask every literal including numeric suffixes",
            "max_source_bytes": MAX_SOURCE_BYTES,
            "max_source_factors": MAX_SOURCE_FACTORS,
            "rust_source_encoding": "ASCII v1",
            "shebang_policy": "only byte-zero hash-bang-slash; reject ambiguous trivia",
            "rustfmt_args": _RUSTFMT_POLICY_ARGS,
            "rustfmt_config_sha256": sha256(_RUSTFMT_CONFIG_BYTES).hexdigest(),
            "rustfmt_identity_sha256": tool.sha256,
            "source_noncode_mask": "nested-comments-and-rust-literals-v1",
        },
    )


def _run_rustfmt(source: str, tool: _RustfmtIdentityV1) -> str:
    try:
        source_bytes = source.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise RustActionRegistryError("Rust action source must be ASCII in v1") from exc
    if not source_bytes or len(source_bytes) > MAX_SOURCE_BYTES or b"\0" in source_bytes:
        raise RustActionRegistryError("Rust action source failed exact byte constraints")
    directive_projection = "".join(_mask_rust_noncode_v1(source).split())
    if "rustfmt::skip" in directive_projection or "rustfmt_skip" in directive_projection:
        raise RustActionRegistryError("rustfmt skip directives are not permitted")
    try:
        with tempfile.TemporaryDirectory(prefix="humanbrain-rustfmt-config-") as root:
            config_path = Path(root) / "rustfmt.toml"
            config_path.write_bytes(_RUSTFMT_CONFIG_BYTES)
            completed = subprocess.run(
                (
                    tool.path,
                    *_RUSTFMT_ARGS,
                    "--config-path",
                    str(config_path),
                ),
                cwd=root,
                env=_basic_process_environment(),
                input=source_bytes,
                capture_output=True,
                check=False,
                timeout=15,
            )
        formatted = completed.stdout.decode("ascii", errors="strict")
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RustActionRegistryError("Rust action source projection failed") from exc
    if (
        completed.returncode != 0
        or completed.stderr
        or not formatted
        or len(formatted.encode("ascii")) > MAX_SOURCE_BYTES
    ):
        raise RustActionRegistryError("Rust action source projection failed")
    return formatted


def _blank(chars: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        if chars[index] not in "\r\n":
            chars[index] = " "


def _quoted_end(source: str, start: int, quote: str) -> int:
    index = start + 1
    escaped = False
    while index < len(source):
        value = source[index]
        if escaped:
            escaped = False
        elif value == "\\":
            escaped = True
        elif value == quote:
            return index + 1
        index += 1
    raise RustActionRegistryError("unterminated Rust literal projection")


def _raw_literal_end(source: str, start: int) -> int | None:
    prefix_length = 0
    for prefix in ("br", "cr", "r"):
        if source.startswith(prefix, start):
            prefix_length = len(prefix)
            break
    if prefix_length == 0:
        return None
    index = start + prefix_length
    hash_count = 0
    while index < len(source) and source[index] == "#":
        hash_count += 1
        index += 1
    if hash_count > 255:
        raise RustActionRegistryError("Rust raw literal exceeds the frozen hash cap")
    if index >= len(source) or source[index] != '"':
        return None
    closing = '"' + ("#" * hash_count)
    end = source.find(closing, index + 1)
    if end < 0:
        raise RustActionRegistryError("unterminated Rust raw literal projection")
    return end + len(closing)


def _literal_suffix_end(source: str, end: int) -> int:
    if end >= len(source) or not (source[end].isalpha() or source[end] == "_"):
        return end
    index = end + 1
    while index < len(source) and (source[index].isalnum() or source[index] == "_"):
        index += 1
    return index


def _char_literal_end(source: str, start: int) -> int | None:
    if start + 1 >= len(source):
        return None
    if source[start + 1] == "\\":
        return _quoted_end(source, start, "'")
    if start + 2 < len(source) and source[start + 2] == "'":
        return start + 3
    if source[start + 1] in " \t\r\n":
        return _quoted_end(source, start, "'")
    return None


def _identifier_end(source: str, start: int) -> int:
    index = start + 1
    while index < len(source) and (source[index].isalnum() or source[index] == "_"):
        index += 1
    return index


def _mask_rust_noncode_v1(source: str) -> str:
    """Blank comments and literal tokens while retaining code identifiers."""

    chars = list(source)
    index = 0
    while index < len(source):
        if index == 0 and source.startswith("#!", index):
            if source.startswith("#![", index):
                pass
            elif source.startswith("#!/", index) and not source.startswith(("#!/*", "#!//"), index):
                end = source.find("\n", index + 2)
                end = len(source) if end < 0 else end
                _blank(chars, index, end)
                index = end
                continue
            else:
                raise RustActionRegistryError(
                    "ambiguous Rust shebang or inner attribute syntax is not permitted"
                )
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = len(source) if end < 0 else end
            _blank(chars, index, end)
            index = end
            continue
        if source.startswith("/*", index):
            depth = 1
            end = index + 2
            while end < len(source) and depth:
                if source.startswith("/*", end):
                    depth += 1
                    end += 2
                elif source.startswith("*/", end):
                    depth -= 1
                    end += 2
                else:
                    end += 1
            if depth:
                raise RustActionRegistryError("unterminated Rust comment projection")
            _blank(chars, index, end)
            index = end
            continue

        raw_end = _raw_literal_end(source, index)
        if raw_end is not None:
            raw_end = _literal_suffix_end(source, raw_end)
            _blank(chars, index, raw_end)
            index = raw_end
            continue

        if source.startswith(('b"', 'c"'), index):
            end = _literal_suffix_end(source, _quoted_end(source, index + 1, '"'))
            _blank(chars, index, end)
            index = end
            continue
        if source.startswith("b'", index):
            end = _literal_suffix_end(source, _quoted_end(source, index + 1, "'"))
            _blank(chars, index, end)
            index = end
            continue
        if source[index] == '"':
            end = _literal_suffix_end(source, _quoted_end(source, index, '"'))
            _blank(chars, index, end)
            index = end
            continue
        if source[index] == "'":
            end = _char_literal_end(source, index)
            if end is not None:
                end = _literal_suffix_end(source, end)
                _blank(chars, index, end)
                index = end
                continue
            if (
                source.startswith("'r#", index)
                and index + 3 < len(source)
                and (source[index + 3].isalpha() or source[index + 3] == "_")
            ):
                end = _identifier_end(source, index + 3)
                _blank(chars, index + 1, index + 3)
                index = end
                continue
            if index + 1 < len(source) and (
                source[index + 1].isalpha() or source[index + 1] == "_"
            ):
                index = _identifier_end(source, index + 1)
                continue
        if (
            source.startswith("r#", index)
            and index + 2 < len(source)
            and (source[index + 2].isalpha() or source[index + 2] == "_")
        ):
            end = _identifier_end(source, index + 2)
            _blank(chars, index, index + 2)
            index = end
            continue
        if source[index].isalpha() or source[index] == "_":
            end = _identifier_end(source, index)
            if source[index:end] in {"false", "true"}:
                _blank(chars, index, end)
            index = end
            continue
        if source[index].isdigit():
            number = _NUMBER_LITERAL.match(source, index)
            if number is None:
                raise RustActionRegistryError("Rust numeric literal projection failed")
            _blank(chars, index, number.end())
            index = number.end()
            continue
        index += 1
    return "".join(chars)


@dataclass(frozen=True, slots=True)
class _RustSourceProjectionV1:
    action_handle: str
    source_sha256: str
    formatted_sha256: str
    structural_sha256: str
    source_factors: tuple[str, ...]
    source_factors_sha256: str
    rustfmt_identity_sha256: str
    sha256: str


def _project_source(
    source: str,
    *,
    tool: _RustfmtIdentityV1,
    policy_sha256: str,
) -> _RustSourceProjectionV1:
    formatted = _run_rustfmt(source, tool)
    if _run_rustfmt(formatted, tool) != formatted:
        raise RustActionRegistryError("rustfmt projection is not idempotent")
    masked = _mask_rust_noncode_v1(formatted)
    structural_code = " ".join(masked.split())
    factors = lexical_factors_from_text_v1(masked)
    if not factors or len(factors) > MAX_SOURCE_FACTORS:
        raise RustActionRegistryError("Rust source factor capacity failed")
    source_sha256 = sha256(source.encode("ascii")).hexdigest()
    formatted_sha256 = sha256(formatted.encode("ascii")).hexdigest()
    source_factors_sha256 = _digest(_FACTOR_SET_NAMESPACE, factors)
    structural_sha256 = _digest(
        _STRUCTURAL_NAMESPACE,
        {
            "policy_sha256": policy_sha256,
            "rustfmt_identity_sha256": tool.sha256,
            "masked_structure_sha256": sha256(structural_code.encode("ascii")).hexdigest(),
            "source_factors_sha256": source_factors_sha256,
        },
    )
    action_handle = "ra_" + _digest(
        _HANDLE_NAMESPACE,
        {
            "source_sha256": source_sha256,
            "structural_sha256": structural_sha256,
        },
    )
    payload = {
        "action_handle": action_handle,
        "source_sha256": source_sha256,
        "formatted_sha256": formatted_sha256,
        "structural_sha256": structural_sha256,
        "source_factors": factors,
        "source_factors_sha256": source_factors_sha256,
        "rustfmt_identity_sha256": tool.sha256,
    }
    return _RustSourceProjectionV1(
        **payload,
        sha256=_digest(_SOURCE_PROJECTION_NAMESPACE, payload),
    )


@dataclass(frozen=True, slots=True)
class _RustActionAuthorizationV1:
    policy_sha256: str
    action_handle: str
    factor: str
    source_projection_sha256: str
    evidence_sha256s: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class RustActionFactorBindingV1:
    policy_sha256: str
    factor: str
    action_handle: str
    opaque_symbol: str
    sha256: str

    def authenticate(self) -> None:
        payload = {
            "policy_sha256": self.policy_sha256,
            "factor": self.factor,
            "action_handle": self.action_handle,
            "opaque_symbol": self.opaque_symbol,
        }
        if (
            not _valid_sha256(self.policy_sha256)
            or type(self.factor) is not str
            or _FACTOR.fullmatch(self.factor) is None
            or type(self.action_handle) is not str
            or _ACTION_HANDLE.fullmatch(self.action_handle) is None
            or self.opaque_symbol != f"rust_action.{self.action_handle}"
            or self.sha256 != _digest(_BINDING_NAMESPACE, payload)
        ):
            raise RustActionRegistryError("Rust action binding authentication failed")


@dataclass(frozen=True, slots=True)
class RustActionSelectorViewV1:
    policy_sha256: str
    candidate_projections: tuple[action_memory.KnowledgeActionCandidateProjectionV1, ...]
    factor_bindings: tuple[RustActionFactorBindingV1, ...]
    candidate_set_sha256: str
    sha256: str

    def authenticate(self) -> None:
        if (
            not _valid_sha256(self.policy_sha256)
            or type(self.candidate_projections) is not tuple
            or len(self.candidate_projections) != ACTION_COUNT
            or type(self.factor_bindings) is not tuple
            or not self.factor_bindings
        ):
            raise RustActionRegistryError("Rust action selector authentication failed")
        try:
            for candidate in self.candidate_projections:
                if type(candidate) is not action_memory.KnowledgeActionCandidateProjectionV1:
                    raise RustActionRegistryError("Rust action selector authentication failed")
                candidate.authenticate()
            for binding in self.factor_bindings:
                if type(binding) is not RustActionFactorBindingV1:
                    raise RustActionRegistryError("Rust action selector authentication failed")
                binding.authenticate()
        except action_memory.KnowledgeActionMemoryError as exc:
            raise RustActionRegistryError("Rust action selector authentication failed") from exc
        candidates = tuple(
            sorted(
                self.candidate_projections,
                key=lambda value: (value.symbol, value.candidate_id),
            )
        )
        bindings = tuple(
            sorted(
                self.factor_bindings,
                key=lambda value: (value.factor, value.action_handle),
            )
        )
        candidate_by_handle = {value.candidate_id: value for value in self.candidate_projections}
        handles_with_bindings = {value.action_handle for value in self.factor_bindings}
        expected_set = action_memory.knowledge_action_candidate_set_sha256_v1(
            self.candidate_projections
        )
        payload = {
            "policy_sha256": self.policy_sha256,
            "candidate_projection_sha256s": tuple(
                value.sha256 for value in self.candidate_projections
            ),
            "factor_binding_sha256s": tuple(value.sha256 for value in self.factor_bindings),
            "candidate_set_sha256": self.candidate_set_sha256,
        }
        if (
            self.candidate_projections != candidates
            or len(candidate_by_handle) != ACTION_COUNT
            or self.factor_bindings != bindings
            or len({value.sha256 for value in bindings}) != len(bindings)
            or handles_with_bindings != set(candidate_by_handle)
            or any(
                binding.policy_sha256 != self.policy_sha256
                or candidate_by_handle[binding.action_handle].symbol != binding.opaque_symbol
                for binding in bindings
            )
            or self.candidate_set_sha256 != expected_set
            or self.sha256 != _digest(_SELECTOR_NAMESPACE, payload)
        ):
            raise RustActionRegistryError("Rust action selector authentication failed")


@dataclass(frozen=True, slots=True)
class RustActionRegistryReceiptV1:
    schema: str
    claim_boundary: str
    policy_sha256: str
    input_records_sha256: str
    consolidation_corpus_sha256: str
    consolidation_producer_sha256: str
    rustfmt_path: str
    rustfmt_version: str
    rustfmt_executable_sha256: str
    rustfmt_identity_sha256: str
    source_projection_sha256s: tuple[str, str]
    authorization_sha256s: tuple[str, ...]
    selector_view_sha256: str
    action_count: int
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        tool_payload = {
            "path": self.rustfmt_path,
            "version": self.rustfmt_version,
            "executable_sha256": self.rustfmt_executable_sha256,
        }
        if (
            self.schema != SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or not _valid_sha256(self.policy_sha256)
            or not _valid_sha256(self.input_records_sha256)
            or not _valid_sha256(self.consolidation_corpus_sha256)
            or not _valid_sha256(self.consolidation_producer_sha256)
            or type(self.rustfmt_path) is not str
            or not Path(self.rustfmt_path).is_absolute()
            or type(self.rustfmt_version) is not str
            or not self.rustfmt_version
            or not _valid_sha256(self.rustfmt_executable_sha256)
            or self.rustfmt_identity_sha256 != _digest(_TOOL_NAMESPACE, tool_payload)
            or type(self.source_projection_sha256s) is not tuple
            or len(self.source_projection_sha256s) != ACTION_COUNT
            or len(set(self.source_projection_sha256s)) != ACTION_COUNT
            or any(not _valid_sha256(value) for value in self.source_projection_sha256s)
            or type(self.authorization_sha256s) is not tuple
            or not self.authorization_sha256s
            or self.authorization_sha256s != tuple(sorted(set(self.authorization_sha256s)))
            or any(not _valid_sha256(value) for value in self.authorization_sha256s)
            or not _valid_sha256(self.selector_view_sha256)
            or self.action_count != ACTION_COUNT
            or self.sha256 != _digest(_RECEIPT_NAMESPACE, payload)
        ):
            raise RustActionRegistryError("Rust action receipt authentication failed")


@dataclass(frozen=True, slots=True)
class BoundRustActionMemoryReceiptV1:
    """Auditable commitment for one exact registry/knowledge memory fit."""

    schema: str
    claim_boundary: str
    registry_receipt_sha256: str
    selector_view_sha256: str
    registry_authorization_sha256s: tuple[str, ...]
    candidate_set_sha256: str
    knowledge_projection_sha256: str
    knowledge_provenance_sha256: str
    binder_cognition_sha256: str
    binder_authorization_sha256: str
    association_projection_sha256: str
    factor_count: int
    symbol_count: int
    association_count: int
    plasticity_gate: bool
    checkpoint_sha256: str
    memory_provenance_roots: tuple[str, ...]
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        scalar_roots = (
            self.registry_receipt_sha256,
            self.selector_view_sha256,
            self.candidate_set_sha256,
            self.knowledge_projection_sha256,
            self.knowledge_provenance_sha256,
            self.binder_cognition_sha256,
            self.binder_authorization_sha256,
            self.association_projection_sha256,
            self.checkpoint_sha256,
        )
        required_provenance = {
            self.registry_receipt_sha256,
            self.selector_view_sha256,
            *self.registry_authorization_sha256s,
            self.candidate_set_sha256,
            self.knowledge_projection_sha256,
            self.knowledge_provenance_sha256,
            self.binder_cognition_sha256,
            self.binder_authorization_sha256,
            self.association_projection_sha256,
        }
        if (
            self.schema != BOUND_MEMORY_SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or any(not _valid_sha256(value) for value in scalar_roots)
            or type(self.registry_authorization_sha256s) is not tuple
            or not self.registry_authorization_sha256s
            or self.registry_authorization_sha256s
            != tuple(sorted(set(self.registry_authorization_sha256s)))
            or any(not _valid_sha256(value) for value in self.registry_authorization_sha256s)
            or type(self.factor_count) is not int
            or self.factor_count < 1
            or self.symbol_count != ACTION_COUNT
            or type(self.association_count) is not int
            or not ACTION_COUNT <= self.association_count <= self.factor_count * ACTION_COUNT
            or type(self.plasticity_gate) is not bool
            or type(self.memory_provenance_roots) is not tuple
            or self.memory_provenance_roots != tuple(sorted(set(self.memory_provenance_roots)))
            or any(not _valid_sha256(value) for value in self.memory_provenance_roots)
            or not required_provenance.issubset(self.memory_provenance_roots)
            or self.sha256 != _digest(_BOUND_RECEIPT_NAMESPACE, payload)
        ):
            raise RustActionRegistryError("bound Rust action memory receipt failed")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class BoundRustActionMemoryV1:
    """Builder-owned frozen memory authorized by one registry and all-card projection."""

    _knowledge_projection: consolidated_cells.ConsolidatedKnowledgeCellProjectionV1
    _association_projection: action_memory.KnowledgeActionAssociationProjectionV1
    _memory: action_memory.FrozenCommonKnowledgeActionMemoryV1
    _receipt: BoundRustActionMemoryReceiptV1

    def __init__(
        self,
        *,
        knowledge_projection: consolidated_cells.ConsolidatedKnowledgeCellProjectionV1,
        association_projection: action_memory.KnowledgeActionAssociationProjectionV1,
        memory: action_memory.FrozenCommonKnowledgeActionMemoryV1,
        receipt: BoundRustActionMemoryReceiptV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("bound Rust action memories are registry-builder-owned")
        object.__setattr__(self, "_knowledge_projection", knowledge_projection)
        object.__setattr__(self, "_association_projection", association_projection)
        object.__setattr__(self, "_memory", memory)
        object.__setattr__(self, "_receipt", receipt)
        self.authenticate_structure()

    @property
    def receipt(self) -> BoundRustActionMemoryReceiptV1:
        self.authenticate_structure()
        return self._receipt

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate_structure()
        return self._memory.checkpoint_sha256

    @property
    def training_summary(self) -> action_memory.KnowledgeActionTrainingSummaryV1:
        self.authenticate_structure()
        return self._memory.training_summary

    @property
    def provenance_roots(self) -> tuple[str, ...]:
        self.authenticate_structure()
        return self._memory.provenance_roots

    def authenticate_structure(self) -> None:
        if (
            type(self._knowledge_projection)
            is not consolidated_cells.ConsolidatedKnowledgeCellProjectionV1
            or type(self._association_projection)
            is not action_memory.KnowledgeActionAssociationProjectionV1
            or type(self._memory) is not action_memory.FrozenCommonKnowledgeActionMemoryV1
            or type(self._receipt) is not BoundRustActionMemoryReceiptV1
        ):
            raise RustActionRegistryError("bound Rust action memory shape changed")
        try:
            self._knowledge_projection.authenticate_structure()
            self._association_projection.authenticate()
            self._memory.authenticate()
            self._receipt.authenticate_structure()
        except (
            action_memory.KnowledgeActionMemoryError,
            consolidated_cells.ConsolidatedKnowledgeCellProjectionError,
        ) as exc:
            raise RustActionRegistryError("bound Rust action memory authentication failed") from exc
        summary = self._memory.training_summary
        association = self._association_projection
        receipt = self._receipt
        if (
            receipt.knowledge_projection_sha256 != self._knowledge_projection.sha256
            or receipt.knowledge_provenance_sha256 != self._knowledge_projection.provenance_sha256
            or receipt.association_projection_sha256 != association.sha256
            or receipt.factor_count != len(association.factor_vocabulary)
            or receipt.symbol_count != len(association.symbol_vocabulary)
            or receipt.association_count != len(association.associations)
            or association.cognition_sha256 != receipt.binder_cognition_sha256
            or association.authorization_roots
            != tuple(
                sorted(
                    {
                        receipt.binder_authorization_sha256,
                        receipt.registry_receipt_sha256,
                        receipt.selector_view_sha256,
                        *receipt.registry_authorization_sha256s,
                    }
                )
            )
            or receipt.checkpoint_sha256 != self._memory.checkpoint_sha256
            or receipt.memory_provenance_roots != self._memory.provenance_roots
            or receipt.plasticity_gate is not summary.plasticity_gate
            or summary.records_sha256 != association.source_sha256
            or summary.graph_sha256 != association.graph_sha256
            or summary.producer_receipt_sha256 != association.producer_receipt_sha256
            or summary.projection_sha256 != association.sha256
            or self._memory.symbol_vocabulary != association.symbol_vocabulary
        ):
            raise RustActionRegistryError("bound Rust action memory derivation changed")


@dataclass(frozen=True, slots=True)
class RustActionMaterializationV1:
    registry_sha256: str
    recall_sha256: str
    action_handle: str
    source_sha256: str
    structural_sha256: str
    source: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        try:
            source_sha256 = sha256(self.source.encode("ascii", errors="strict")).hexdigest()
        except (AttributeError, UnicodeEncodeError) as exc:
            raise RustActionRegistryError(
                "Rust action materialization authentication failed"
            ) from exc
        if (
            not _valid_sha256(self.registry_sha256)
            or not _valid_sha256(self.recall_sha256)
            or type(self.action_handle) is not str
            or _ACTION_HANDLE.fullmatch(self.action_handle) is None
            or self.source_sha256 != source_sha256
            or not _valid_sha256(self.structural_sha256)
            or self.sha256 != _digest(_MATERIALIZATION_NAMESPACE, payload)
        ):
            raise RustActionRegistryError("Rust action materialization authentication failed")


@dataclass(frozen=True, slots=True)
class _RustActionEntryV1:
    source: str
    projection: _RustSourceProjectionV1
    candidate: action_memory.KnowledgeActionCandidateProjectionV1
    authorizations: tuple[_RustActionAuthorizationV1, ...]


@dataclass(frozen=True, slots=True)
class _BoundAssociationDerivationV1:
    factor_vocabulary: tuple[str, ...]
    symbol_vocabulary: tuple[str, ...]
    associations: tuple[tuple[str, str], ...]
    cognition_sha256: str
    authorization_sha256: str
    projection: action_memory.KnowledgeActionAssociationProjectionV1


def _projection_provenance_roots(
    knowledge: consolidated_cells.ConsolidatedKnowledgeCellProjectionV1,
    registry_receipt: RustActionRegistryReceiptV1,
    selector: RustActionSelectorViewV1,
    *,
    binder_cognition_sha256: str,
    binder_authorization_sha256: str,
) -> tuple[str, ...]:
    roots = {
        knowledge.input_records_sha256,
        knowledge.policy_sha256,
        knowledge.consolidation_corpus_sha256,
        knowledge.consolidation_producer_sha256,
        knowledge.cognition_corpus_sha256,
        knowledge.provenance_sha256,
        knowledge.graph.sha256,
        knowledge.producer_receipt.source_sha256,
        knowledge.producer_receipt.graph_sha256,
        knowledge.producer_receipt.producer_schema_sha256,
        knowledge.producer_receipt.producer_attestation_sha256,
        knowledge.producer_receipt.sha256,
        knowledge.sha256,
        registry_receipt.policy_sha256,
        registry_receipt.input_records_sha256,
        registry_receipt.consolidation_corpus_sha256,
        registry_receipt.consolidation_producer_sha256,
        registry_receipt.rustfmt_executable_sha256,
        registry_receipt.rustfmt_identity_sha256,
        *registry_receipt.source_projection_sha256s,
        *registry_receipt.authorization_sha256s,
        registry_receipt.sha256,
        selector.candidate_set_sha256,
        selector.sha256,
        binder_cognition_sha256,
        binder_authorization_sha256,
    }
    for concept in knowledge.concepts:
        roots.add(concept.cognition_sha256)
    for receipt in knowledge.concept_receipts:
        roots.update((receipt.cognition_sha256, receipt.card_sha256, receipt.sha256))
        roots.update(receipt.provenance_roots)
    for candidate in selector.candidate_projections:
        roots.add(candidate.sha256)
    for binding in selector.factor_bindings:
        roots.add(binding.sha256)
    if any(not _valid_sha256(value) for value in roots):
        raise RustActionRegistryError("bound memory provenance root changed")
    return tuple(sorted(roots))


def _derive_bound_association(
    knowledge: consolidated_cells.ConsolidatedKnowledgeCellProjectionV1,
    registry_receipt: RustActionRegistryReceiptV1,
    selector: RustActionSelectorViewV1,
) -> _BoundAssociationDerivationV1:
    factor_vocabulary = tuple(
        sorted({factor for concept in knowledge.concepts for factor in concept.content_factors})
    )
    symbol_vocabulary = tuple(candidate.symbol for candidate in selector.candidate_projections)
    associations = tuple(
        sorted((binding.factor, binding.opaque_symbol) for binding in selector.factor_bindings)
    )
    if (
        not factor_vocabulary
        or not set(factor for factor, _symbol in associations).issubset(factor_vocabulary)
        or {symbol for _factor, symbol in associations} != set(symbol_vocabulary)
    ):
        raise RustActionRegistryError(
            "registry factors are not authorized by the all-card knowledge projection"
        )
    cognition_payload = {
        "factor_vocabulary": factor_vocabulary,
        "symbol_vocabulary": symbol_vocabulary,
        "associations": associations,
        "candidate_set_sha256": selector.candidate_set_sha256,
    }
    cognition_sha256 = _digest(_BOUND_COGNITION_NAMESPACE, cognition_payload)
    authorization_payload = {
        "knowledge_projection_sha256": knowledge.sha256,
        "knowledge_provenance_sha256": knowledge.provenance_sha256,
        "registry_receipt_sha256": registry_receipt.sha256,
        "selector_view_sha256": selector.sha256,
        "registry_authorization_sha256s": registry_receipt.authorization_sha256s,
        "candidate_set_sha256": selector.candidate_set_sha256,
        "binder_cognition_sha256": cognition_sha256,
    }
    authorization_sha256 = _digest(
        _BOUND_AUTHORIZATION_NAMESPACE,
        authorization_payload,
    )
    provenance_roots = _projection_provenance_roots(
        knowledge,
        registry_receipt,
        selector,
        binder_cognition_sha256=cognition_sha256,
        binder_authorization_sha256=authorization_sha256,
    )
    try:
        projection = action_memory.make_knowledge_action_association_projection_v1(
            source_sha256=knowledge.input_records_sha256,
            graph_sha256=knowledge.graph.sha256,
            producer_receipt_sha256=knowledge.producer_receipt.sha256,
            cognition_sha256=cognition_sha256,
            authorization_roots=tuple(
                sorted(
                    {
                        authorization_sha256,
                        registry_receipt.sha256,
                        selector.sha256,
                        *registry_receipt.authorization_sha256s,
                    }
                )
            ),
            factor_vocabulary=factor_vocabulary,
            symbol_vocabulary=symbol_vocabulary,
            associations=associations,
            source_projection_cell_steps=knowledge.graph.projection_cell_steps,
            provenance_roots=provenance_roots,
        )
    except action_memory.KnowledgeActionMemoryError as exc:
        raise RustActionRegistryError("bound association projection failed") from exc
    return _BoundAssociationDerivationV1(
        factor_vocabulary=factor_vocabulary,
        symbol_vocabulary=symbol_vocabulary,
        associations=associations,
        cognition_sha256=cognition_sha256,
        authorization_sha256=authorization_sha256,
        projection=projection,
    )


def _make_bound_receipt(
    *,
    knowledge: consolidated_cells.ConsolidatedKnowledgeCellProjectionV1,
    registry_receipt: RustActionRegistryReceiptV1,
    selector: RustActionSelectorViewV1,
    derivation: _BoundAssociationDerivationV1,
    memory: action_memory.FrozenCommonKnowledgeActionMemoryV1,
    plasticity_gate: bool,
) -> BoundRustActionMemoryReceiptV1:
    payload = {
        "schema": BOUND_MEMORY_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "registry_receipt_sha256": registry_receipt.sha256,
        "selector_view_sha256": selector.sha256,
        "registry_authorization_sha256s": registry_receipt.authorization_sha256s,
        "candidate_set_sha256": selector.candidate_set_sha256,
        "knowledge_projection_sha256": knowledge.sha256,
        "knowledge_provenance_sha256": knowledge.provenance_sha256,
        "binder_cognition_sha256": derivation.cognition_sha256,
        "binder_authorization_sha256": derivation.authorization_sha256,
        "association_projection_sha256": derivation.projection.sha256,
        "factor_count": len(derivation.factor_vocabulary),
        "symbol_count": len(derivation.symbol_vocabulary),
        "association_count": len(derivation.associations),
        "plasticity_gate": plasticity_gate,
        "checkpoint_sha256": memory.checkpoint_sha256,
        "memory_provenance_roots": memory.provenance_roots,
    }
    result = BoundRustActionMemoryReceiptV1(
        **payload,
        sha256=_digest(_BOUND_RECEIPT_NAMESPACE, payload),
    )
    result.authenticate_structure()
    return result


def _make_authorization(
    *,
    policy_sha256: str,
    projection: _RustSourceProjectionV1,
    factor: str,
    evidence: tuple[tuple[str, str], ...],
) -> _RustActionAuthorizationV1:
    payload = {
        "policy_sha256": policy_sha256,
        "action_handle": projection.action_handle,
        "factor": factor,
        "source_projection_sha256": projection.sha256,
        "evidence_sha256s": evidence,
    }
    return _RustActionAuthorizationV1(
        **payload,
        sha256=_digest(_AUTHORIZATION_NAMESPACE, payload),
    )


def _assemble(
    corpus: KnowledgeCorpus,
    consolidation: KnowledgeConsolidationV1,
    sources: tuple[str, str],
    tool: _RustfmtIdentityV1,
) -> tuple[
    tuple[_RustActionEntryV1, _RustActionEntryV1],
    RustActionSelectorViewV1,
    RustActionRegistryReceiptV1,
]:
    if type(corpus) is not KnowledgeCorpus:
        raise TypeError("exact KnowledgeCorpus required")
    if type(consolidation) is not KnowledgeConsolidationV1:
        raise TypeError("exact KnowledgeConsolidationV1 required")
    if (
        type(sources) is not tuple
        or len(sources) != ACTION_COUNT
        or any(type(source) is not str for source in sources)
    ):
        raise TypeError("sources must be one exact pair of unlabeled strings")
    consolidation.authenticate(corpus)
    if consolidation.split != "acquire":
        raise RustActionRegistryError("Rust action registry requires acquire knowledge")

    policy_sha256 = _policy_sha256(tool)
    projections = tuple(
        _project_source(source, tool=tool, policy_sha256=policy_sha256) for source in sources
    )
    if (
        len({value.source_sha256 for value in projections}) != ACTION_COUNT
        or len({value.structural_sha256 for value in projections}) != ACTION_COUNT
        or len({value.action_handle for value in projections}) != ACTION_COUNT
    ):
        raise RustActionRegistryError(
            "Rust actions require distinct exact and structural source roots"
        )

    evidence_by_factor: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for card in consolidation.cards:
        for factor in card.factors:
            evidence_by_factor[factor.factor].append((card.sha256, factor.sha256))

    entries: list[_RustActionEntryV1] = []
    for source, projection in zip(sources, projections, strict=True):
        overlap = tuple(sorted(set(projection.source_factors).intersection(evidence_by_factor)))
        if not overlap:
            raise RustActionRegistryError(
                "each Rust action requires exact authenticated factor overlap"
            )
        authorizations = tuple(
            _make_authorization(
                policy_sha256=policy_sha256,
                projection=projection,
                factor=factor,
                evidence=tuple(sorted(set(evidence_by_factor[factor]))),
            )
            for factor in overlap
        )
        handle = projection.action_handle
        candidate = action_memory.project_knowledge_action_candidate_v1(
            handle,
            f"rust_action.{handle}()",
        )
        entries.append(
            _RustActionEntryV1(
                source=source,
                projection=projection,
                candidate=candidate,
                authorizations=authorizations,
            )
        )
    entries.sort(key=lambda value: value.projection.action_handle)
    first_factors = {value.factor for value in entries[0].authorizations}
    second_factors = {value.factor for value in entries[1].authorizations}
    if not first_factors - second_factors or not second_factors - first_factors:
        raise RustActionRegistryError(
            "each Rust action requires one distinguishing authenticated factor"
        )

    bindings = tuple(
        sorted(
            (
                RustActionFactorBindingV1(
                    policy_sha256=policy_sha256,
                    factor=authorization.factor,
                    action_handle=entry.projection.action_handle,
                    opaque_symbol=entry.candidate.symbol,
                    sha256=_digest(
                        _BINDING_NAMESPACE,
                        {
                            "policy_sha256": policy_sha256,
                            "factor": authorization.factor,
                            "action_handle": entry.projection.action_handle,
                            "opaque_symbol": entry.candidate.symbol,
                        },
                    ),
                )
                for entry in entries
                for authorization in entry.authorizations
            ),
            key=lambda value: (value.factor, value.action_handle),
        )
    )
    candidates = tuple(
        sorted(
            (entry.candidate for entry in entries),
            key=lambda value: (value.symbol, value.candidate_id),
        )
    )
    candidate_set_sha256 = action_memory.knowledge_action_candidate_set_sha256_v1(candidates)
    selector_payload = {
        "policy_sha256": policy_sha256,
        "candidate_projection_sha256s": tuple(value.sha256 for value in candidates),
        "factor_binding_sha256s": tuple(value.sha256 for value in bindings),
        "candidate_set_sha256": candidate_set_sha256,
    }
    selector = RustActionSelectorViewV1(
        policy_sha256=policy_sha256,
        candidate_projections=candidates,
        factor_bindings=bindings,
        candidate_set_sha256=candidate_set_sha256,
        sha256=_digest(_SELECTOR_NAMESPACE, selector_payload),
    )
    selector.authenticate()

    receipt_payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "policy_sha256": policy_sha256,
        "input_records_sha256": consolidation.input_records_sha256,
        "consolidation_corpus_sha256": consolidation.corpus_sha256,
        "consolidation_producer_sha256": consolidation.producer_sha256,
        "rustfmt_path": tool.path,
        "rustfmt_version": tool.version,
        "rustfmt_executable_sha256": tool.executable_sha256,
        "rustfmt_identity_sha256": tool.sha256,
        "source_projection_sha256s": tuple(entry.projection.sha256 for entry in entries),
        "authorization_sha256s": tuple(
            sorted(
                authorization.sha256 for entry in entries for authorization in entry.authorizations
            )
        ),
        "selector_view_sha256": selector.sha256,
        "action_count": ACTION_COUNT,
    }
    receipt = RustActionRegistryReceiptV1(
        **receipt_payload,
        sha256=_digest(_RECEIPT_NAMESPACE, receipt_payload),
    )
    receipt.authenticate_structure()
    return (entries[0], entries[1]), selector, receipt


class PreparedRustActionRegistryV1:
    """Trusted-host registry; learner code receives only its source-free view."""

    __slots__ = (
        "_consolidation",
        "_corpus",
        "_entries",
        "_issued_recalls",
        "_materialized",
        "_receipt",
        "_selector_view",
        "_sources",
        "_state_lock",
        "_tool",
    )

    def __init__(
        self,
        *,
        corpus: KnowledgeCorpus,
        consolidation: KnowledgeConsolidationV1,
        sources: tuple[str, str],
        tool: _RustfmtIdentityV1,
        entries: tuple[_RustActionEntryV1, _RustActionEntryV1],
        selector_view: RustActionSelectorViewV1,
        receipt: RustActionRegistryReceiptV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("Rust action registries are builder-owned")
        self._corpus = corpus
        self._consolidation = consolidation
        self._sources = sources
        self._tool = tool
        self._entries = entries
        self._selector_view = selector_view
        self._receipt = receipt
        self._issued_recalls: dict[str, tuple[str, str, str]] = {}
        self._materialized = False
        self._state_lock = Lock()
        self._authenticate_current()

    def _authenticate_current(self) -> None:
        current_tool = _read_rustfmt_identity(Path(self._tool.path))
        if current_tool != self._tool:
            raise RustActionRegistryError("rustfmt identity changed")
        entries, selector, receipt = _assemble(
            self._corpus,
            self._consolidation,
            self._sources,
            current_tool,
        )
        if entries != self._entries or selector != self._selector_view or receipt != self._receipt:
            raise RustActionRegistryError("Rust action registry changed")

    @property
    def selector_view(self) -> RustActionSelectorViewV1:
        self._authenticate_current()
        return self._selector_view

    @property
    def receipt(self) -> RustActionRegistryReceiptV1:
        self._authenticate_current()
        return self._receipt

    def fit_memory(
        self,
        knowledge: consolidated_cells.ConsolidatedKnowledgeCellProjectionV1,
        *,
        plasticity_gate: bool = True,
    ) -> BoundRustActionMemoryV1:
        """Fit exact all-card factor-to-opaque-action associations for this registry."""

        self._authenticate_current()
        if type(knowledge) is not consolidated_cells.ConsolidatedKnowledgeCellProjectionV1:
            raise TypeError("exact consolidated knowledge cell projection required")
        if type(plasticity_gate) is not bool:
            raise TypeError("plasticity_gate must be one exact boolean")
        try:
            knowledge.authenticate(self._consolidation, corpus=self._corpus)
        except consolidated_cells.ConsolidatedKnowledgeCellProjectionError as exc:
            raise RustActionRegistryError(
                "all-card knowledge projection is not bound to this registry"
            ) from exc
        derivation = _derive_bound_association(
            knowledge,
            self._receipt,
            self._selector_view,
        )
        try:
            memory = action_memory.fit_common_knowledge_action_associations_v1(
                derivation.projection,
                plasticity_gate=plasticity_gate,
            )
        except action_memory.KnowledgeActionMemoryError as exc:
            raise RustActionRegistryError("bound Rust action memory fit failed") from exc
        receipt = _make_bound_receipt(
            knowledge=knowledge,
            registry_receipt=self._receipt,
            selector=self._selector_view,
            derivation=derivation,
            memory=memory,
            plasticity_gate=plasticity_gate,
        )
        result = BoundRustActionMemoryV1(
            knowledge_projection=knowledge,
            association_projection=derivation.projection,
            memory=memory,
            receipt=receipt,
            construction_token=_CONSTRUCTION_TOKEN,
        )
        self._authenticate_current()
        result.authenticate_structure()
        return result

    def _authenticate_bound_memory(self, bound: BoundRustActionMemoryV1) -> None:
        if type(bound) is not BoundRustActionMemoryV1:
            raise TypeError("exact bound Rust action memory required")
        bound.authenticate_structure()
        try:
            bound._knowledge_projection.authenticate(
                self._consolidation,
                corpus=self._corpus,
            )
        except consolidated_cells.ConsolidatedKnowledgeCellProjectionError as exc:
            raise RustActionRegistryError(
                "bound memory knowledge projection is not from this registry corpus"
            ) from exc
        derivation = _derive_bound_association(
            bound._knowledge_projection,
            self._receipt,
            self._selector_view,
        )
        if bound._association_projection != derivation.projection:
            raise RustActionRegistryError("bound association projection changed")
        try:
            expected_memory = action_memory.fit_common_knowledge_action_associations_v1(
                derivation.projection,
                plasticity_gate=bound._receipt.plasticity_gate,
            )
        except action_memory.KnowledgeActionMemoryError as exc:
            raise RustActionRegistryError("bound Rust action memory derivation failed") from exc
        memory_identity = (
            bound._memory.checkpoint_sha256,
            bound._memory.training_summary,
            bound._memory.symbol_vocabulary,
            bound._memory.provenance_roots,
            bound._memory.learned_state_shapes,
        )
        expected_identity = (
            expected_memory.checkpoint_sha256,
            expected_memory.training_summary,
            expected_memory.symbol_vocabulary,
            expected_memory.provenance_roots,
            expected_memory.learned_state_shapes,
        )
        expected_receipt = _make_bound_receipt(
            knowledge=bound._knowledge_projection,
            registry_receipt=self._receipt,
            selector=self._selector_view,
            derivation=derivation,
            memory=expected_memory,
            plasticity_gate=bound._receipt.plasticity_gate,
        )
        if memory_identity != expected_identity or bound._receipt != expected_receipt:
            raise RustActionRegistryError("bound Rust action memory changed")

    def recall(
        self,
        memory: BoundRustActionMemoryV1,
        task: action_memory.KnowledgeActionTaskProjectionV1,
        *,
        cell_gate: bool = True,
    ) -> action_memory.KnowledgeActionRecallV1:
        self._authenticate_current()
        if type(memory) is not BoundRustActionMemoryV1:
            raise TypeError("exact bound Rust action memory required")
        if type(task) is not action_memory.KnowledgeActionTaskProjectionV1:
            raise TypeError("exact knowledge-action task projection required")
        if type(cell_gate) is not bool:
            raise TypeError("cell_gate must be one exact boolean")
        self._authenticate_bound_memory(memory)
        task.authenticate()
        result = memory._memory.recall(
            task,
            self._selector_view.candidate_projections,
            cell_gate=cell_gate,
        )
        result.authenticate()
        self._authenticate_current()
        memory.authenticate_structure()
        with self._state_lock:
            if (
                result.sha256 not in self._issued_recalls
                and len(self._issued_recalls) >= MAX_ISSUED_RECALLS
            ):
                raise RustActionRegistryError("issued recall capacity exceeded")
            self._issued_recalls[result.sha256] = (
                self._receipt.sha256,
                memory._receipt.sha256,
                memory.checkpoint_sha256,
            )
        return result

    def materialize_selected(
        self,
        recall: action_memory.KnowledgeActionRecallV1,
    ) -> RustActionMaterializationV1:
        self._authenticate_current()
        if type(recall) is not action_memory.KnowledgeActionRecallV1:
            raise TypeError("exact knowledge-action recall required")
        try:
            recall.authenticate()
        except action_memory.KnowledgeActionMemoryError as exc:
            raise RustActionRegistryError("selected recall authentication failed") from exc
        matches = tuple(
            entry
            for entry in self._entries
            if entry.projection.action_handle == recall.selected_candidate_id
            and entry.candidate.symbol == recall.selected_symbol
            and entry.candidate.sha256 == recall.selected_candidate_sha256
        )
        if len(matches) != 1:
            raise RustActionRegistryError("selected Rust action is not authenticated")
        entry = matches[0]
        with self._state_lock:
            issued = self._issued_recalls.get(recall.sha256)
            if (
                issued is None
                or issued[0] != self._receipt.sha256
                or not _valid_sha256(issued[1])
                or issued[2] != recall.checkpoint_sha256
                or recall.candidate_set_sha256 != self._selector_view.candidate_set_sha256
                or recall.status != action_memory.RECALLED
                or recall.selected_candidate_id is None
                or recall.selected_symbol is None
                or recall.selected_candidate_sha256 is None
                or self._materialized
            ):
                raise RustActionRegistryError("recall cannot materialize a Rust action")
            # Consume before blocking projection or host callback. Failures remain
            # consumed, so no concurrent or retry path can release another source.
            self._materialized = True
        reprojected = _project_source(
            entry.source,
            tool=self._tool,
            policy_sha256=self._receipt.policy_sha256,
        )
        if reprojected != entry.projection:
            raise RustActionRegistryError("selected Rust source changed")
        payload = {
            "registry_sha256": self._receipt.sha256,
            "recall_sha256": recall.sha256,
            "action_handle": entry.projection.action_handle,
            "source_sha256": entry.projection.source_sha256,
            "structural_sha256": entry.projection.structural_sha256,
            "source": entry.source,
        }
        result = RustActionMaterializationV1(
            **payload,
            sha256=_digest(_MATERIALIZATION_NAMESPACE, payload),
        )
        result.authenticate_structure()
        return result

    def execute_selected(
        self,
        recall: action_memory.KnowledgeActionRecallV1,
        execute: Callable[[RustActionMaterializationV1], _T],
    ) -> _T:
        if not callable(execute):
            raise TypeError("execute must be one callable selected-action adapter")
        materialization = self.materialize_selected(recall)
        return execute(materialization)


def prepare_rust_action_registry_v1(
    corpus: KnowledgeCorpus,
    consolidation: KnowledgeConsolidationV1,
    *,
    sources: tuple[str, str],
) -> PreparedRustActionRegistryV1:
    """Prepare exactly two task-independent, unlabeled Rust action sources."""

    tool = _discover_rustfmt()
    entries, selector, receipt = _assemble(corpus, consolidation, sources, tool)
    return PreparedRustActionRegistryV1(
        corpus=corpus,
        consolidation=consolidation,
        sources=sources,
        tool=tool,
        entries=entries,
        selector_view=selector,
        receipt=receipt,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "ACTION_COUNT",
    "BOUND_MEMORY_SCHEMA",
    "CLAIM_BOUNDARY",
    "SCHEMA",
    "BoundRustActionMemoryReceiptV1",
    "BoundRustActionMemoryV1",
    "PreparedRustActionRegistryV1",
    "RustActionFactorBindingV1",
    "RustActionMaterializationV1",
    "RustActionRegistryError",
    "RustActionRegistryReceiptV1",
    "RustActionSelectorViewV1",
    "prepare_rust_action_registry_v1",
]
