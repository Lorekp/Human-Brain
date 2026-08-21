from __future__ import annotations

import inspect
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
from humanbrain.language import CellLanguageModel
from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig
from humanbrain.simple_swe import freeze
from humanbrain.simple_swe.adapter import CellCompletionAdapter, CompletionRequest
from humanbrain.simple_swe import compiler as compiler_module
from humanbrain.simple_swe.compiler import CompileRequest, RawPythonCompiler
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


def _source_repository(root: Path, monkeypatch) -> Path:
    repository = root / "source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "simple-swe@example.invalid")
    _git(repository, "config", "user.name", "Simple SWE Test")
    (repository / "brain.py").write_text("CELL_ONLY = True\n", encoding="utf-8")
    installed_source = repository / "src" / "humanbrain" / "execution.py"
    installed_source.parent.mkdir(parents=True)
    installed_source.write_text("CELL_EXECUTION = True\n", encoding="utf-8")
    _git(repository, "add", "brain.py", "src/humanbrain/execution.py")
    _git(repository, "commit", "-q", "-m", "frozen source")
    monkeypatch.setattr(
        freeze,
        "_installed_execution_source_paths",
        lambda: (installed_source.resolve(),),
    )
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


def _request() -> CompletionRequest:
    task = _task()
    return CompletionRequest(
        prompt=task.prompt,
        entry_point=task.entry_point,
        parameter_names=tuple(task.parameter_names),
    )


def test_completion_request_exposes_only_prompt_and_signature() -> None:
    request = _request()

    assert set(vars(request)) == {"prompt", "entry_point", "parameter_names"}
    assert not hasattr(request, "family")
    assert not hasattr(request, "tier")
    assert not hasattr(request, "ordinal")
    assert not hasattr(request, "task_id")
    assert not hasattr(request, "hidden_test_sha256")


def test_adapter_reloads_one_cell_checkpoint_and_audits_exact_emitted_tokens(
    tmp_path: Path, monkeypatch,
) -> None:
    source = _source_repository(tmp_path, monkeypatch)
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

    first = adapter.synthesize(_request())
    second = adapter.synthesize(_request())

    assert first.source == first.emitted_text
    expected_tokens = load_protocol(PROTOCOL_PATH).max_new_tokens
    assert first.audit.emitted_token_count == expected_tokens
    assert len(first.audit.emitted_token_ids) == expected_tokens
    assert first.audit.emitted_token_sha256 == canonical_sha256(
        list(first.audit.emitted_token_ids)
    )
    assert (
        first.audit.learned_checkpoint_state_before
        == first.audit.learned_checkpoint_state_after
    )
    assert (
        second.audit.learned_checkpoint_state_before
        == first.audit.learned_checkpoint_state_before
    )
    assert (
        second.audit.learned_checkpoint_state_after
        == first.audit.learned_checkpoint_state_after
    )
    assert second.audit.emitted_token_ids == first.audit.emitted_token_ids
    assert first.audit.direct_synthesis_no_tools is True
    assert first.audit.compiler_algorithm_source == "emitted-tokens-only"
    assert sha256(checkpoint.read_bytes()).hexdigest() == first.audit.checkpoint_sha256


def test_concrete_raw_python_compiler_receives_only_emissions_and_signature(
) -> None:
    compiler = RawPythonCompiler()
    request = CompileRequest(
        emitted_text="def solve_opaque(value):\n    return value\n",
        emitted_token_ids=tuple(range(load_protocol(PROTOCOL_PATH).max_new_tokens)),
        entry_point="solve_opaque",
        parameter_names=("value",),
    )

    assert type(compiler) is RawPythonCompiler
    assert compiler.compile(request) == request.emitted_text
    assert compiler.identity.compiler_sha256 == sha256(
        Path(compiler_module.__file__).read_bytes()
    ).hexdigest()
    assert not hasattr(compiler_module, "SourceCompiler")
    assert not hasattr(compiler_module, "create_source_compiler")
    assert not hasattr(compiler_module, "compiler_factory_sha256")
    assert "compiler" not in inspect.signature(CellCompletionAdapter).parameters
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


def test_protocol_max_tokens_is_the_adapter_default(tmp_path: Path, monkeypatch) -> None:
    source = _source_repository(tmp_path, monkeypatch)
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

    completion = adapter.synthesize(_request())

    assert completion.audit.emitted_token_count == load_protocol(
        PROTOCOL_PATH
    ).max_new_tokens


def test_recurrent_adapter_allows_transient_traces_but_preserves_learned_state(
    tmp_path: Path, monkeypatch,
) -> None:
    source = _source_repository(tmp_path, monkeypatch)
    model = RecurrentCellLanguageModel.grow(
        ["def solve(value):\n    return value + 1\n"],
        RecurrentModelConfig(
            circuit=RecurrentCircuitConfig(
                reservoir_cells=8,
                input_fanout=4,
                recurrent_fanout=2,
                token_interval_ms=1.0,
                input_pulse_ms=0.5,
            )
        ),
    )
    assert model._circuit.learning_frozen is True
    checkpoint = tmp_path / "generic-recurrent-cell-checkpoint.npz"
    model.save(checkpoint)
    checkpoint_bytes_before = checkpoint.read_bytes()
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

    completion = adapter.synthesize(_request())

    assert checkpoint.read_bytes() == checkpoint_bytes_before
    assert (
        completion.audit.learned_checkpoint_state_before
        == completion.audit.learned_checkpoint_state_after
    )
    restored = RecurrentCellLanguageModel.load(checkpoint)
    assert restored._circuit.learning_frozen is True


def test_sparse_three_factor_adapter_preserves_checkpointed_synapse_structure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_repository(tmp_path, monkeypatch)
    model = RecurrentCellLanguageModel.grow(
        ["def solve(value):\n    return value + 1\n"],
        RecurrentModelConfig(
            growth_epochs=2,
            circuit=RecurrentCircuitConfig(
                reservoir_cells=8,
                input_fanout=4,
                recurrent_fanout=2,
                readout_learning_rule="sparse_three_factor",
                readout_fan_in=2,
                readout_structural_interval=2,
                readout_structural_grace=2,
                readout_turnover_probability=1.0,
            ),
        ),
    )
    checkpoint = tmp_path / "generic-sparse-recurrent-cell-checkpoint.npz"
    model.save(checkpoint)
    checkpoint_bytes_before = checkpoint.read_bytes()
    with np.load(checkpoint, allow_pickle=False) as state:
        assert "readout_synapses_pa" not in state.files
        assert "readout_contact_sources" in state.files
        assert "readout_contact_weights_pa" in state.files
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

    completion = adapter.synthesize(_request())

    assert checkpoint.read_bytes() == checkpoint_bytes_before
    assert (
        completion.audit.learned_checkpoint_state_before
        == completion.audit.learned_checkpoint_state_after
    )
