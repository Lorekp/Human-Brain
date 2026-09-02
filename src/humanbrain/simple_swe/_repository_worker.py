from __future__ import annotations

import ast
import builtins
import json
import os
import sys

SCHEMA = "humanbrain.simple-swe-repository-worker.v1"
RESULT_SCHEMA = "humanbrain.simple-swe-repository-result.v1"
OUTCOMES = (
    "policy_violation",
    "runtime_error",
    "case_failure",
    "harness_error",
    "passed",
)

_DENIED_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.ClassDef,
    ast.Global,
    ast.Import,
    ast.Lambda,
    ast.Nonlocal,
    ast.With,
)
_DENIED_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "print",
    "setattr",
    "vars",
    "__import__",
}
_ALLOWED_ATTRIBUTES = frozenset(
    {
        "append",
        "count",
        "endswith",
        "find",
        "get",
        "index",
        "isalnum",
        "isalpha",
        "isdecimal",
        "isdigit",
        "islower",
        "isspace",
        "isupper",
        "items",
        "join",
        "keys",
        "lower",
        "lstrip",
        "pop",
        "replace",
        "reverse",
        "rfind",
        "rstrip",
        "split",
        "startswith",
        "strip",
        "upper",
        "values",
    }
)


def _emit(
    task_commitment_sha256: str,
    action_sha256: str,
    evaluator_sha256: str,
    outcome: str,
    completed_case_count: int,
    expected_case_count: int,
    detail: str = "",
) -> None:
    if outcome not in OUTCOMES:
        raise ValueError("unknown repository worker outcome")
    sys.stdout.write(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "task_commitment_sha256": task_commitment_sha256,
                "action_sha256": action_sha256,
                "evaluator_sha256": evaluator_sha256,
                "outcome": outcome,
                "completed_case_count": completed_case_count,
                "expected_case_count": expected_case_count,
                "detail": detail,
            },
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


def _same_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _same_value(left, right)
            for left, right in zip(actual, expected, strict=True)  # type: ignore[arg-type]
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(  # type: ignore[union-attr]
            _same_value(actual[key], expected[key])
            for key in expected  # type: ignore[index]
        )
    return actual == expected


def _module_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and value
        and all(part.isidentifier() and not part.startswith("__") for part in value.split("."))
    )


def _validated_trees(records: object, max_ast_nodes: int) -> dict[str, ast.Module]:
    if not isinstance(records, list) or not records:
        raise ValueError("modules must be a non-empty list")
    raw_sources: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "source"}:
            raise ValueError("module record is invalid")
        name, source = record["name"], record["source"]
        if not _module_name(name) or not isinstance(source, str) or name in raw_sources:
            raise ValueError("module identity is invalid")
        raw_sources[name] = source
    trees: dict[str, ast.Module] = {}
    seen_nodes = 0
    for name, source in raw_sources.items():
        tree = ast.parse(source, filename=f"<{name}>", mode="exec")
        for node in ast.walk(tree):
            seen_nodes += 1
            if seen_nodes > max_ast_nodes:
                raise ValueError("repository AST-node limit exceeded")
            if isinstance(node, _DENIED_NODES):
                raise PermissionError(f"{type(node).__name__} is forbidden")
            if isinstance(node, ast.ImportFrom) and (
                node.level != 0
                or node.module not in raw_sources
                or any(alias.name == "*" or alias.asname is not None for alias in node.names)
            ):
                raise PermissionError("only exact internal imports are allowed")
            if isinstance(node, ast.Attribute) and node.attr not in _ALLOWED_ATTRIBUTES:
                raise PermissionError(f"attribute {node.attr!r} is forbidden")
            if isinstance(node, ast.Name) and (
                node.id in _DENIED_NAMES or node.id.startswith("__")
            ):
                raise PermissionError(f"name {node.id!r} is forbidden")
        trees[name] = tree
    return trees


def _load_modules(trees: dict[str, ast.Module]) -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    loading: set[str] = set()

    def load(name: str) -> dict[str, object]:
        if name in loaded:
            return loaded[name]
        if name in loading:
            raise ValueError("cyclic internal import")
        loading.add(name)
        tree = trees[name]
        namespace: dict[str, object] = {"__builtins__": _safe_builtins()}
        body: list[ast.stmt] = []
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                dependency = load(statement.module or "")
                for alias in statement.names:
                    if alias.name not in dependency:
                        raise ValueError("internal import name is unavailable")
                    namespace[alias.name] = dependency[alias.name]
            else:
                body.append(statement)
        compiled = compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            f"<{name}>",
            "exec",
        )
        exec(compiled, namespace)
        loading.remove(name)
        loaded[name] = namespace
        return namespace

    for module_name in trees:
        load(module_name)
    return loaded


def main() -> int:
    task_commitment_sha256 = ""
    action_sha256 = ""
    evaluator_sha256 = ""
    expected_case_count = 0
    try:
        payload = json.loads(sys.stdin.buffer.read())
        _apply_posix_limits(payload["limits"])
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("worker payload schema is invalid")
        task_commitment_sha256 = payload["task_commitment_sha256"]
        action_sha256 = payload["action_sha256"]
        evaluator_sha256 = payload["evaluator_sha256"]
        expected_case_count = payload["expected_case_count"]
        cases = payload["cases"]
        if (
            not all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in (
                    task_commitment_sha256,
                    action_sha256,
                    evaluator_sha256,
                )
            )
            or not isinstance(expected_case_count, int)
            or isinstance(expected_case_count, bool)
            or expected_case_count < 1
            or not isinstance(cases, list)
            or len(cases) != expected_case_count
        ):
            raise ValueError("worker evaluator commitment is invalid")
        trees = _validated_trees(payload["modules"], int(payload["max_ast_nodes"]))
        entry_module = payload["entry_module"]
        entry_point = payload["entry_point"]
        if entry_module not in trees or not isinstance(entry_point, str):
            raise ValueError("worker entry point is invalid")
    except PermissionError as error:
        _emit(
            task_commitment_sha256,
            action_sha256,
            evaluator_sha256,
            "policy_violation",
            0,
            expected_case_count,
            str(error),
        )
        return 0
    except BaseException as error:
        _emit(
            task_commitment_sha256,
            action_sha256,
            evaluator_sha256,
            "harness_error",
            0,
            expected_case_count,
            type(error).__name__,
        )
        return 2

    completed = 0
    try:
        modules = _load_modules(trees)
        function = modules[entry_module][entry_point]
        if not callable(function):
            raise TypeError("entry point is not callable")
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"request", "expected"}:
                raise ValueError("case record is invalid")
            request = case["request"]
            if not isinstance(request, dict) or not all(
                isinstance(name, str) and name.isidentifier() for name in request
            ):
                raise ValueError("case request is invalid")
            actual = function(**request)
            if not _same_value(actual, case["expected"]):
                _emit(
                    task_commitment_sha256,
                    action_sha256,
                    evaluator_sha256,
                    "case_failure",
                    completed,
                    expected_case_count,
                    "hidden case failed",
                )
                return 0
            completed += 1
    except MemoryError:
        _emit(
            task_commitment_sha256,
            action_sha256,
            evaluator_sha256,
            "runtime_error",
            completed,
            expected_case_count,
            "MemoryError",
        )
        return 0
    except BaseException as error:
        _emit(
            task_commitment_sha256,
            action_sha256,
            evaluator_sha256,
            "runtime_error",
            completed,
            expected_case_count,
            type(error).__name__,
        )
        return 0
    _emit(
        task_commitment_sha256,
        action_sha256,
        evaluator_sha256,
        "passed",
        completed,
        expected_case_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
