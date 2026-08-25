"""Command-line entry point for the native experiment diagnostics workbench."""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import load_catalog


def _default_experiments_root() -> Path:
    return Path(__file__).resolve().parents[3] / "experiments"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explore Human Brain experiments in a native diagnostics workbench."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=_default_experiments_root(),
        help="experiments root containing <experiment>/<arm>.json",
    )
    parser.add_argument(
        "--experiment",
        help="experiment to select initially (default: visualizer_demo or first available)",
    )
    parser.add_argument(
        "--activity-stream",
        type=Path,
        help=(
            "attach the Activity view to a live .observatory JSONL stream; "
            "the file may appear after the visualizer starts"
        ),
    )
    parser.add_argument(
        "--structure-manifest",
        type=Path,
        help=(
            "inspect an exact procedural topology manifest independently of "
            "the selected result (cell_count, fanout, topology, seed, protocol)"
        ),
    )
    arguments = parser.parse_args(argv)
    try:
        catalog = load_catalog(arguments.path)
    except FileNotFoundError as error:
        parser.error(str(error))
    if not catalog.runs:
        parser.error(f"no ladder experiment records found under {catalog.root}")
    if (
        arguments.experiment is not None
        and arguments.experiment not in catalog.experiment_ids
    ):
        parser.error(f"unknown experiment: {arguments.experiment}")

    try:
        from .app import run_workbench
        from .structure import StructureUnavailable, load_structure_manifest
    except ImportError as error:
        if error.name and error.name.startswith("dearpygui"):
            parser.error(
                "the visualizer extra is not installed; run "
                "'uv --cache-dir .uv-cache pip install --python "
                ".venv\\Scripts\\python.exe -e \".[visualizer]\"'"
            )
        raise
    try:
        structure_override = (
            load_structure_manifest(arguments.structure_manifest)
            if arguments.structure_manifest is not None
            else None
        )
    except StructureUnavailable as error:
        parser.error(str(error))
    run_workbench(
        catalog,
        initial_experiment=arguments.experiment,
        activity_stream=arguments.activity_stream,
        structure_override=structure_override,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
