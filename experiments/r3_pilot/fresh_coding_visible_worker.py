"""Isolated visible-case executor for the fresh coding challenge."""

from __future__ import annotations

import builtins
import copy
import json
import os
import sys


def _apply_posix_limits(limits: dict[str, int | float]) -> None:
    if os.name == "nt":
        return
    import resource

    cpu = int(limits["cpu_seconds"])
    memory = int(limits["memory_bytes"])
    output = int(limits["output_bytes"])
    process_count = int(limits["processes"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    if hasattr(resource, "RLIMIT_DATA"):
        resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output, output))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (process_count, process_count))
    if hasattr(resource, "RLIMIT_NOFILE"):
        resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def _same_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _same_value(left, right)
            for left, right in zip(actual, expected)  # type: ignore[arg-type]
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(  # type: ignore[union-attr]
            _same_value(actual[key], expected[key])  # type: ignore[index]
            for key in expected
        )
    return actual == expected


def _safe_builtins() -> dict[str, object]:
    names = (
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "ord",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "MemoryError",
        "ValueError",
    )
    return {name: getattr(builtins, name) for name in names}


def _format_exception(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _case_exception(
    case: dict[str, object], error: BaseException
) -> dict[str, object]:
    return {
        "args": case["args"],
        "expected": case["expected"],
        "exception": _format_exception(error),
        "passed": False,
    }


def _emit(result: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            result,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    sys.stdout.flush()


def _load_function(source: str, entry_point: str) -> object:
    namespace: dict[str, object] = {"__builtins__": _safe_builtins()}
    compiled = compile(source, "<cell-completion>", "exec")
    exec(compiled, namespace)
    function = namespace[entry_point]
    if not callable(function):
        raise TypeError("entry point is not callable")
    return function


def main() -> int:
    payload = json.loads(sys.stdin.buffer.read())
    cases = payload["cases"]

    try:
        _apply_posix_limits(payload["limits"])
        function = _load_function(payload["source"], payload["entry_point"])
    except BaseException as error:
        results = [_case_exception(case, error) for case in cases]
        _emit({"cases": results, "passed": False})
        return 0

    results: list[dict[str, object]] = []
    for case in cases:
        try:
            actual = function(*copy.deepcopy(case["args"]))  # type: ignore[operator]
            # Validate before adding it to the envelope. This turns values such as
            # sets, generators, and non-finite floats into an ordinary case error.
            json.dumps(
                actual,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except BaseException as error:
            results.append(_case_exception(case, error))
            continue
        passed = _same_value(actual, case["expected"])
        results.append(
            {
                "args": case["args"],
                "expected": case["expected"],
                "actual": actual,
                "passed": passed,
            }
        )

    passed = bool(results) and all(case["passed"] for case in results)
    _emit({"cases": results, "passed": passed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
