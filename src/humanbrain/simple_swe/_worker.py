from __future__ import annotations

import builtins
import json
import os
import sys


def _emit(outcome: str, detail: str = "") -> None:
    sys.stdout.write(
        json.dumps(
            {"outcome": outcome, "detail": detail},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    sys.stdout.flush()


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
            _same_value(left, right) for left, right in zip(actual, expected)  # type: ignore[arg-type]
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(  # type: ignore[union-attr]
            _same_value(actual[key], expected[key]) for key in expected  # type: ignore[index]
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


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read())
        _apply_posix_limits(payload["limits"])
    except BaseException as error:
        _emit("harness_error", type(error).__name__)
        return 2

    namespace: dict[str, object] = {"__builtins__": _safe_builtins()}
    try:
        compiled = compile(payload["source"], "<cell-completion>", "exec")
        exec(compiled, namespace)
        function = namespace[payload["entry_point"]]
        if not callable(function):
            _emit("signature_error", "entry point is not callable")
            return 0
        for case in payload["regression_cases"]:
            actual = function(*case["args"])
            if not _same_value(actual, case["expected"]):
                _emit("regression_failure", "fixed regression case failed")
                return 0
        for case in payload["property_cases"]:
            actual = function(*case["args"])
            if not _same_value(actual, case["expected"]):
                _emit("property_failure", "property case failed")
                return 0
    except MemoryError:
        _emit("memory_limit", "MemoryError")
        return 0
    except BaseException as error:
        _emit("runtime_error", type(error).__name__)
        return 0
    _emit("passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
