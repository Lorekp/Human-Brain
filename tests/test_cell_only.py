from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "humanbrain"
FORBIDDEN = frozenset({"torch", "transformers", "jax", "tensorflow", "sklearn"})


def test_inference_package_has_no_ann_or_llm_dependency() -> None:
    imports: list[tuple[str, str]] = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((path.name, alias.name.split(".")[0]) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((path.name, node.module.split(".")[0]))
    assert [(path, name) for path, name in imports if name in FORBIDDEN] == []


def test_language_interface_delegates_selection_to_cell_race() -> None:
    source = (PACKAGE / "language.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"argmax", "argmin", "max", "min"}
    direct = []
    inference_methods = {"predict_next", "_prediction", "complete"}
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        if function.name not in inference_methods:
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in forbidden_calls:
                    direct.append(name)
    assert direct == []
