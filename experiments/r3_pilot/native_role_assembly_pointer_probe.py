"""Run the isolated native recurrent assembly-pointer prototype.

The mechanism lives in the adjacent standalone Rust source. It is compiled
directly into a temporary directory so it cannot become a Cargo auto-binary or
widen the production native ABI. No target, held-out, M0, calibration, or C1
surface is opened by this development probe.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
RUST_SOURCE: Final = Path(__file__).with_suffix(".rs")
PASS_OUTCOME: Final = "PASS_GENUINE_RECURRENT_K11_POINTER"


def run_development_probe() -> dict[str, object]:
    """Compile and run only the synthetic recurrent-mechanism isolate."""

    rustc = shutil.which("rustc")
    if rustc is None:
        raise RuntimeError("rustc is required for the native mechanism probe")
    with tempfile.TemporaryDirectory(prefix="humanbrain-role-pointer-") as directory:
        executable = Path(directory) / (
            "native_role_assembly_pointer_probe.exe"
            if os.name == "nt"
            else "native_role_assembly_pointer_probe"
        )
        compile_result = subprocess.run(
            [
                rustc,
                "--edition=2021",
                "-C",
                "opt-level=2",
                str(RUST_SOURCE),
                "-o",
                str(executable),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                "standalone native probe did not compile:\n" + compile_result.stderr
            )
        run_result = subprocess.run(
            [str(executable)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if run_result.returncode != 0:
            raise RuntimeError(
                "standalone native probe stopped:\n"
                + run_result.stdout
                + run_result.stderr
            )
    report = json.loads(run_result.stdout)
    if report.get("outcome") != PASS_OUTCOME:
        raise RuntimeError(f"unexpected native probe outcome: {report!r}")
    return report


def main() -> int:
    print(json.dumps(run_development_probe(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
