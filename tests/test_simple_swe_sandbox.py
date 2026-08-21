from __future__ import annotations

import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from humanbrain.simple_swe import sandbox
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.sandbox import execute_completion
from humanbrain.simple_swe.tasks import HiddenCase, PrivateTask, PublicTask

PROTOCOL = load_protocol(
    Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
)
WORKER_SOURCE = Path(sandbox.__file__).with_name("_worker.py").read_bytes()
WORKER_SHA256 = sha256(WORKER_SOURCE).hexdigest()
PYTHON_EXECUTABLE = sandbox.resolve_python_executable()
PYTHON_EXECUTABLE_SHA256 = sha256(PYTHON_EXECUTABLE.read_bytes()).hexdigest()


def _execute(source, public, private, limits):
    return execute_completion(
        source,
        public,
        private,
        limits,
        python_executable=PYTHON_EXECUTABLE,
        python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
        worker_source=WORKER_SOURCE,
        worker_sha256=WORKER_SHA256,
    )


@pytest.fixture
def increment_task() -> tuple[PublicTask, PrivateTask]:
    public = PublicTask(
        task_id="fixture/increment",
        partition="fixture",
        tier=0,
        family="fixture_increment",
        ordinal=0,
        entry_point="solve_increment",
        parameter_names=("value",),
        prompt="Return value plus one.",
        hidden_test_sha256="fixture-test-hash",
        task_sha256="fixture-task-hash",
    )
    private = PrivateTask(
        task_id=public.task_id,
        public_task_sha256=public.task_sha256,
        regression_cases=(
            HiddenCase(args=(0,), expected=1, distribution="fixed-regression"),
            HiddenCase(args=(1,), expected=2, distribution="fixed-regression"),
        ),
        property_cases=(
            HiddenCase(args=(-4,), expected=-3, distribution="edge-distribution"),
            HiddenCase(args=(9,), expected=10, distribution="random-small"),
        ),
        test_sha256="fixture-test-hash",
    )
    return public, private


def test_correct_function_passes_in_a_bounded_child_process(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task

    result = _execute(
        "def solve_increment(value):\n    return value + 1\n",
        public,
        private,
        PROTOCOL.execution,
    )

    assert result.outcome == "passed"
    assert result.passed is True
    assert result.wall_seconds <= PROTOCOL.execution.wall_seconds
    assert result.sandboxed is True


def test_worker_snapshot_must_match_the_frozen_hash(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task

    result = execute_completion(
        "def solve_increment(value):\n    return value + 1\n",
        public,
        private,
        PROTOCOL.execution,
        python_executable=PYTHON_EXECUTABLE,
        python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
        worker_source=b"tampered worker",
        worker_sha256=WORKER_SHA256,
    )

    assert result.outcome == "harness_error"
    assert result.sandboxed is False


def test_replaced_runtime_fails_closed_after_the_completion_seal(
    tmp_path: Path,
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    runtime = tmp_path / PYTHON_EXECUTABLE.name
    shutil.copyfile(PYTHON_EXECUTABLE, runtime)
    frozen_sha256 = sha256(runtime.read_bytes()).hexdigest()
    runtime.write_bytes(b"replacement after completion seal")

    result = execute_completion(
        "def solve_increment(value):\n    return value + 1\n",
        public,
        private,
        PROTOCOL.execution,
        python_executable=runtime,
        python_executable_sha256=frozen_sha256,
        worker_source=WORKER_SOURCE,
        worker_sha256=WORKER_SHA256,
    )

    assert result.outcome == "harness_error"
    assert result.detail == "sandbox Python executable differs from its freeze"
    assert result.sandboxed is False


def test_generator_frame_escape_is_rejected_before_hidden_execution(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    exploit = """def solve_increment(value):
    g = (g.gi_frame.f_back.f_back.f_locals for _ in [0])
    worker_locals = list(g)[0]
    payload = worker_locals["payload"]
    for case in payload["regression_cases"] + payload["property_cases"]:
        if case["args"][0] == value:
            return case["expected"]
    return 0
"""

    result = _execute(exploit, public, private, PROTOCOL.execution)

    assert result.outcome == "policy_violation"
    assert any(
        attribute in result.detail
        for attribute in ("gi_frame", "f_back", "f_locals")
    )
    assert result.sandboxed is False


def test_string_format_attribute_lookup_is_not_allowlisted(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task

    result = _execute(
        'def solve_increment(value):\n    return "{0.__class__}".format(value)\n',
        public,
        private,
        PROTOCOL.execution,
    )

    assert result.outcome == "policy_violation"
    assert "format" in result.detail
    assert result.sandboxed is False


@pytest.mark.parametrize(
    "source",
    [
        """def solve_increment(value):
    try:
        return 1 // 0
    except Exception as error:
        return error.__traceback__.tb_frame.f_back.f_locals["payload"]
""",
        """def solve_increment(value):
    method = [].append
    return method.__self__
""",
        """def solve_increment(value):
    generator = (value for _ in [0])
    return f"{generator.gi_frame}"
""",
        """def solve_increment(value):
    return Exception.mro()
""",
    ],
)
def test_traceback_bound_method_fstring_and_type_reflection_are_denied(
    increment_task: tuple[PublicTask, PrivateTask], source: str
) -> None:
    public, private = increment_task

    result = _execute(source, public, private, PROTOCOL.execution)

    assert result.outcome == "policy_violation"
    assert result.sandboxed is False


def test_lambda_is_rejected_as_an_anonymous_nested_function(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    source = """def solve_increment(value):
    add_one = lambda item: item + 1
    return add_one(value)
"""

    result = _execute(source, public, private, PROTOCOL.execution)

    assert result.outcome == "policy_violation"
    assert "Lambda" in result.detail
    assert result.sandboxed is False


def test_task_required_string_and_list_attributes_remain_allowlisted(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    source = """def solve_increment(value):
    parts = "a b".split()
    parts.append("")
    return value + len("".join(parts)) - 1
"""

    result = _execute(source, public, private, PROTOCOL.execution)

    assert result.outcome == "passed"


@pytest.mark.parametrize(
    ("source", "outcome"),
    [
        ("```python\ndef solve_increment(value):\n    return value + 1\n```", "format_error"),
        ("def solve_increment(value)\n    return value + 1", "syntax_error"),
        ("import os\ndef solve_increment(value):\n    return value + 1", "policy_violation"),
        ("def wrong_name(value):\n    return value + 1", "signature_error"),
        ("def solve_increment(value):\n    return open('secret.txt').read()", "policy_violation"),
    ],
)
def test_completion_is_rejected_before_execution(
    increment_task: tuple[PublicTask, PrivateTask], source: str, outcome: str
) -> None:
    public, private = increment_task

    result = _execute(source, public, private, PROTOCOL.execution)

    assert result.outcome == outcome
    assert result.passed is False
    assert result.sandboxed is False


def test_hidden_failures_have_separate_regression_and_property_taxonomy(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task

    regression = _execute(
        "def solve_increment(value):\n    return value\n",
        public,
        private,
        PROTOCOL.execution,
    )
    property_only = _execute(
        "def solve_increment(value):\n    return value + 1 if value in (0, 1) else value\n",
        public,
        private,
        PROTOCOL.execution,
    )

    assert regression.outcome == "regression_failure"
    assert property_only.outcome == "property_failure"
    assert "-4" not in property_only.detail
    assert "-3" not in property_only.detail


def test_runtime_memory_and_wall_limits_have_stable_taxonomy(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    short_limits = replace(PROTOCOL.execution, wall_seconds=0.25, cpu_seconds=2)

    runtime = _execute(
        "def solve_increment(value):\n    return 1 // 0\n",
        public,
        private,
        short_limits,
    )
    memory = _execute(
        "def solve_increment(value):\n    raise MemoryError\n",
        public,
        private,
        short_limits,
    )
    timeout = _execute(
        "def solve_increment(value):\n    while True:\n        pass\n",
        public,
        private,
        short_limits,
    )

    assert runtime.outcome == "runtime_error"
    assert memory.outcome == "memory_limit"
    assert timeout.outcome == "timeout"


def test_real_allocation_over_process_limit_is_memory_failure(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    memory_limits = replace(
        PROTOCOL.execution, wall_seconds=2.0, cpu_seconds=2, memory_mib=96
    )

    result = _execute(
        "def solve_increment(value):\n    return [0] * 50000000\n",
        public,
        private,
        memory_limits,
    )

    assert result.outcome == "memory_limit"


def test_public_and_private_task_commitments_cannot_be_mixed(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    mismatched = replace(private, test_sha256="different-test-hash")

    result = _execute(
        "def solve_increment(value):\n    return value + 1\n",
        public,
        mismatched,
        PROTOCOL.execution,
    )

    assert result.outcome == "harness_error"
    assert result.sandboxed is False


def test_parent_bounds_completion_bytes_and_ast_nodes_before_child_execution(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task
    byte_limits = replace(PROTOCOL.execution, max_completion_bytes=32)
    ast_limits = replace(
        PROTOCOL.execution, max_completion_bytes=4096, max_ast_nodes=8
    )

    too_many_bytes = _execute(
        "def solve_increment(value):\n    return value + 1\n",
        public,
        private,
        byte_limits,
    )
    too_many_nodes = _execute(
        "def solve_increment(value):\n    return value + 1 + 2 + 3 + 4\n",
        public,
        private,
        ast_limits,
    )

    assert too_many_bytes.outcome == "format_error"
    assert too_many_bytes.sandboxed is False
    assert too_many_nodes.outcome == "format_error"
    assert too_many_nodes.sandboxed is False


def test_runtime_identity_queries_and_hashes_the_selected_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "selected-python.exe"
    executable.write_bytes(b"fixture-python-binary")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(
            stdout='{"implementation":"cpython","version":[3,11,8]}\n'
        )

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)

    identity = sandbox.python_runtime_identity(executable, "python-3.11")

    assert commands[0][0] == str(executable)
    assert commands[0][1:3] == ["-I", "-S"]
    assert identity == {
        "implementation": "cpython",
        "python_version": "3.11.8",
        "executable_sha256": sha256(b"fixture-python-binary").hexdigest(),
    }


def test_runtime_identity_rejects_a_different_python_minor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "selected-python.exe"
    executable.write_bytes(b"fixture-python-binary")
    monkeypatch.setattr(
        sandbox,
        "_probe_python_runtime",
        lambda _executable: {"implementation": "cpython", "version": [3, 12, 1]},
    )

    with pytest.raises(
        sandbox.SandboxConfigurationError, match=r"python-3.11.*3.12"
    ):
        sandbox.python_runtime_identity(executable, "python-3.11")
