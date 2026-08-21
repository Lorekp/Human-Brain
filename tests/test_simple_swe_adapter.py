from __future__ import annotations

import subprocess
from hashlib import sha256
from pathlib import Path

from humanbrain.language import CellLanguageModel
from humanbrain.simple_swe.adapter import CellCompletionAdapter
from humanbrain.simple_swe.compiler import (
    CompileRequest,
    CompilerIdentity,
    RawPythonCompiler,
)
from humanbrain.simple_swe.freeze import create_freeze_manifest
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import PublicTask, canonical_sha256


PROTOCOL_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "core.excludesFile=", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def _source_repository(root: Path) -> Path:
    repository = root / "source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "simple-swe@example.invalid")
    _git(repository, "config", "user.name", "Simple SWE Test")
    (repository / "brain.py").write_text("CELL_ONLY = True\n", encoding="utf-8")
    _git(repository, "add", "brain.py")
    _git(repository, "commit", "-q", "-m", "frozen source")
    return repository


def _checkpoint(root: Path) -> Path:
    model = CellLanguageModel.grow(
        [
            "def solve(value):\n    return value + 1\n",
            "def keep(text):\n    return text\n",
        ]
    )
    checkpoint = root / "generic-cell-checkpoint.npz"
    model.save(checkpoint)
    return checkpoint


def _task() -> PublicTask:
    return PublicTask(
        task_id="fixture/direct-synthesis",
        partition="fixture",
        tier=0,
        family="fixture_direct",
        ordinal=0,
        entry_point="solve_opaque",
        parameter_names=("value",),
        prompt="Return one complete function ending in task-specific semantics.",
        hidden_test_sha256="f" * 64,
        task_sha256="e" * 64,
    )


def test_adapter_reloads_one_cell_checkpoint_and_audits_exact_emitted_tokens(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL_PATH,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=freeze_path,
    )
    adapter = CellCompletionAdapter(
        manifest_path=freeze_path,
        protocol_path=PROTOCOL_PATH,
        checkpoint_path=checkpoint,
        source_root=source,
        compiler=RawPythonCompiler(),
    )

    first = adapter.synthesize(_task())
    second = adapter.synthesize(_task())

    assert first.source == first.emitted_text
    expected_tokens = load_protocol(PROTOCOL_PATH).max_new_tokens
    assert first.audit.emitted_token_count == expected_tokens
    assert len(first.audit.emitted_token_ids) == expected_tokens
    assert first.audit.emitted_token_sha256 == canonical_sha256(
        list(first.audit.emitted_token_ids)
    )
    assert first.audit.checkpoint_state_before == first.audit.checkpoint_state_after
    assert second.audit.checkpoint_state_before == first.audit.checkpoint_state_before
    assert second.audit.checkpoint_state_after == first.audit.checkpoint_state_after
    assert second.audit.emitted_token_ids == first.audit.emitted_token_ids
    assert first.audit.direct_synthesis_no_tools is True
    assert first.audit.compiler_algorithm_source == "emitted-tokens-only"
    assert sha256(checkpoint.read_bytes()).hexdigest() == first.audit.checkpoint_sha256


class RecordingCompiler:
    def __init__(self) -> None:
        self.identity = CompilerIdentity(
            compiler_id="fixture-deterministic-prefix-compiler-v1",
            compiler_sha256="a" * 64,
            mode="deterministic-prefix-dsl",
            algorithm_source="emitted-tokens-only",
        )
        self.requests: list[CompileRequest] = []

    def compile(self, request: CompileRequest) -> str:
        self.requests.append(request)
        parameters = ", ".join(request.parameter_names)
        return f"def {request.entry_point}({parameters}):\n    return 0\n"


def test_compiler_receives_emissions_and_signature_but_no_prompt_tests_or_tools(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    compiler = RecordingCompiler()
    create_freeze_manifest(
        protocol_path=PROTOCOL_PATH,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=freeze_path,
        compiler_identity=compiler.identity,
    )
    adapter = CellCompletionAdapter(
        manifest_path=freeze_path,
        protocol_path=PROTOCOL_PATH,
        checkpoint_path=checkpoint,
        source_root=source,
        compiler=compiler,
    )

    completion = adapter.synthesize(_task())
    request = compiler.requests[0]

    assert completion.source == "def solve_opaque(value):\n    return 0\n"
    assert set(vars(request)) == {
        "emitted_text",
        "emitted_token_ids",
        "entry_point",
        "parameter_names",
    }
    assert request.entry_point == "solve_opaque"
    assert len(request.emitted_token_ids) == load_protocol(PROTOCOL_PATH).max_new_tokens
    assert not hasattr(request, "prompt")
    assert not hasattr(request, "private_tests")
    assert not hasattr(request, "filesystem")
    assert not hasattr(request, "tools")
    assert completion.audit.compiler_sha256 == compiler.identity.compiler_sha256


def test_protocol_max_tokens_is_the_adapter_default(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL_PATH,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=freeze_path,
    )
    adapter = CellCompletionAdapter(
        manifest_path=freeze_path,
        protocol_path=PROTOCOL_PATH,
        checkpoint_path=checkpoint,
        source_root=source,
    )

    completion = adapter.synthesize(_task())

    assert completion.audit.emitted_token_count == load_protocol(
        PROTOCOL_PATH
    ).max_new_tokens
