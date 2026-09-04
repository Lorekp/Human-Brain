"""Checkout-local developer checks; never selects scientific evaluations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Reviewed public fixtures only. Keep explicit selections rather than inferring
# safety from names/markers or collecting the entire experiments tree.
QA = {
    "smoke": [
        "tests/test_tokenizer.py",
        "tests/test_cell_program.py::test_scalar_program_grammar_exposes_only_prefixes_that_can_finish",
        "tests/test_cell_program.py::test_program_compiler_literally_translates_one_complete_cell_program",
        "tests/test_windowed_probe.py",
    ],
    "native": [
        "tests/test_native_build_identity.py::test_source_addressed_release_exports_temporal_checkpoint_capabilities",
        "tests/test_native_temporal_runtime.py::test_actual_native_fresh_checkpoint_import_round_trip",
        "tests/test_native_temporal_runtime.py::test_actual_native_immutable_bytes_publication_is_exclusive_and_bit_exact",
        "tests/test_recurrent_observation_sink.py",
    ],
    "integration": ["tests/test_observatory.py", "tests/test_experiment_visualizer.py"],
    "tooling": ["tests/test_dev_workflow.py"],
}


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8"
    ).strip()


def doctor() -> dict:
    """Verify imports before pytest's conftest can mask a wrong installation."""
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise RuntimeError("The developer tool is outside the intended Git checkout.")
    if Path(sys.prefix).resolve() != ROOT / ".venv":
        raise RuntimeError("Use this checkout's .venv interpreter; run tools/setup.ps1.")
    import humanbrain

    package = Path(humanbrain.__file__).resolve()
    if package != ROOT / "src" / "humanbrain" / "__init__.py":
        raise RuntimeError(f"Wrong editable installation: {package}")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pytest", "ruff")}
    return {
        "checkout": str(ROOT),
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD", "--binary"])
        ).hexdigest(),
        "status": git("status", "--short", "--untracked-files=normal"),
        "python": sys.version,
        "executable": sys.executable,
        "package": str(package),
        "platform": platform.platform(),
        "versions": versions,
        "device": os.environ.get("HUMANBRAIN_DEVICE", "auto"),
        "native_cache": os.environ.get("HUMANBRAIN_NATIVE_CACHE"),
    }


def native_preflight() -> dict:
    # Explicitly builds AND loads through the existing attested API.
    from humanbrain.native import loaded_build_info, loaded_library_identity

    build = loaded_build_info()
    return {
        **loaded_library_identity().to_record(),
        "library": str(build.library_path),
        "toolchain_sha256": build.build_plan.toolchain_identity_sha256,
        "sdk_sha256": build.build_plan.sdk_identity_sha256,
        "target": build.build_plan.target_triple,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Verify environment without compiling native code")
    sub.add_parser("native", help="Build/load the attested native kernel and record identity")
    qa = sub.add_parser("qa", help="Run an explicit public fixture selection with receipts")
    qa.add_argument("tier", choices=QA)
    args = parser.parse_args()
    # One scratch convention; respect explicit supported device/cache overrides.
    os.environ.setdefault("HUMANBRAIN_NATIVE_CACHE", str(ROOT / ".native-cache"))
    os.environ.setdefault("HUMANBRAIN_DEVICE", "cpu")
    os.environ["RUFF_CACHE_DIR"] = str(ROOT / ".ruff_cache")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    scratch = ROOT / ".pytest-tmp" / "dev"
    scratch.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{args.command}-", dir=scratch))
    receipt = {"command": sys.argv[1:], "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()
    result = 1
    try:
        receipt["environment"] = doctor()
        if args.command == "native" or (args.command == "qa" and args.tier in ("native", "integration")):
            receipt["native"] = native_preflight()
        if args.command == "qa":
            command = [
                sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider",
                "-ra", "--durations=10", "--basetemp", str(run / "tmp"),
                f"--junitxml={run / 'junit.xml'}", *QA[args.tier],
            ]
            receipt["pytest_command"] = command
            print(subprocess.list2cmdline(command), flush=True)
            with (run / "output.log").open("w", encoding="utf-8") as output:
                with subprocess.Popen(
                    command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                ) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        output.write(line)
                        output.flush()
                    result = process.wait()
        else:
            result = 0
            print(json.dumps(receipt, indent=2))
    except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        receipt["error"] = str(error)
        print(f"Preflight failed: {error}", file=sys.stderr)
    finally:
        receipt["exit_code"] = result
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Receipt: {run / 'receipt.json'}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
