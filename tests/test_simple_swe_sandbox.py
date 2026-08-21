from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.sandbox import execute_completion
from humanbrain.simple_swe.tasks import HiddenCase, PrivateTask, PublicTask


PROTOCOL = load_protocol(
    Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
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

    result = execute_completion(
        "def solve_increment(value):\n    return value + 1\n",
        public,
        private,
        PROTOCOL.execution,
    )

    assert result.outcome == "passed"
    assert result.passed is True
    assert result.wall_seconds <= PROTOCOL.execution.wall_seconds
    assert result.sandboxed is True


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

    result = execute_completion(source, public, private, PROTOCOL.execution)

    assert result.outcome == outcome
    assert result.passed is False
    assert result.sandboxed is False


def test_hidden_failures_have_separate_regression_and_property_taxonomy(
    increment_task: tuple[PublicTask, PrivateTask],
) -> None:
    public, private = increment_task

    regression = execute_completion(
        "def solve_increment(value):\n    return value\n",
        public,
        private,
        PROTOCOL.execution,
    )
    property_only = execute_completion(
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

    runtime = execute_completion(
        "def solve_increment(value):\n    return 1 // 0\n",
        public,
        private,
        short_limits,
    )
    memory = execute_completion(
        "def solve_increment(value):\n    raise MemoryError\n",
        public,
        private,
        short_limits,
    )
    timeout = execute_completion(
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

    result = execute_completion(
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

    result = execute_completion(
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

    too_many_bytes = execute_completion(
        "def solve_increment(value):\n    return value + 1\n",
        public,
        private,
        byte_limits,
    )
    too_many_nodes = execute_completion(
        "def solve_increment(value):\n    return value + 1 + 2 + 3 + 4\n",
        public,
        private,
        ast_limits,
    )

    assert too_many_bytes.outcome == "format_error"
    assert too_many_bytes.sandboxed is False
    assert too_many_nodes.outcome == "format_error"
    assert too_many_nodes.sandboxed is False
