"""Compact versus varied teaching with common read-only retention probes.

Set CPU and the checkout native-cache environment before launching Python.
Output guards run before importing the shared harness or its native dependencies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEACHING_PATHS = {
    name: HERE / f"contextual_curriculum_{name.lower()}_train.json"
    for name in ("COMPACT", "VARIED")
}
EXTRA_SOURCE_PATHS = (
    "experiments/r4_integrated/contextual_curriculum_python_session.py",
    "experiments/r4_integrated/contextual_curriculum_protocol.md",
    "experiments/r4_integrated/contextual_curriculum_cases.json",
    "experiments/r4_integrated/contextual_curriculum_compact_train.json",
    "experiments/r4_integrated/contextual_curriculum_varied_train.json",
    "experiments/r4_integrated/contextual_reserved_feedback_protocol.md",
)


def _fresh_output(output: Path | None) -> Path:
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "benchmarks" / f"contextual_curriculum_{stamp}_{uuid4().hex}.json"
    output = output.resolve()
    if output.suffix != ".json":
        raise ValueError("comparison output must have a .json suffix")
    if any(output.with_suffix(suffix).exists() for suffix in (".json", ".npz", ".failure.json")):
        raise FileExistsError("choose a fresh output path; existing evidence is immutable")
    return output


def run(output: Path | None = None, model_dir: Path | None = None) -> dict:
    output = _fresh_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = HERE / "contextual_curriculum_cases.json"
    source_revision = fixture_sha256 = None
    try:
        source_revision = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
        fixture_sha256 = sha256(fixture_path.read_bytes()).hexdigest()
        from experiments.r4_integrated import contextual_binding_python_session as binding

        return binding.run(
            output,
            ROOT / ".models/all-MiniLM-L6-v2" if model_dir is None else model_dir,
            fixture_path=fixture_path,
            source_paths=(*binding.SOURCE_PATHS, *EXTRA_SOURCE_PATHS),
            arm_specs=(("COMPACT", True, "fixed"), ("VARIED", True, "fixed")),
            teaching_fixture_paths=TEACHING_PATHS,
            allocation_policies={
                "COMPACT": "reserve_future_corrections",
                "VARIED": "reserve_future_corrections",
            },
            protection_policies={"COMPACT": "none", "VARIED": "none"},
            primary_policy="respond_affine_interpolated",
            include_affine=True,
            exploration_policy="raw_candidate",
            include_rank_diagnostics=True,
            track_learning_diagnostics=True,
            track_source_competition=True,
            report_labels={
                "kind": "public development; compact versus varied teaching "
                "with fixed RESERVED learning",
                "teaching_exposure": "matched sixteen facts and 48 encounters; "
                "only teaching source/request "
                "wording differs; common compact retention probes; no oracle weights",
                "feedback_contract": "affine primary, raw candidate fallback after STOP; "
                "actual FAIL/teacher PASS pairs only; reserved 150-offer allocation in both arms; "
                "fixed rate-one learning and identical rules; no contrast protection in either arm",
            },
        )
    except Exception as error:
        # The shared harness preserves its full partial report in finally. This
        # small sidecar also survives failures before that report can be written.
        failure = {
            "terminal": "error",
            "source_revision": source_revision,
            "fixture_path": str(fixture_path),
            "fixture_sha256": fixture_sha256,
            "output": str(output),
            "error": traceback.format_exc(),
            "report_exists": output.exists(),
            "checkpoint_exists": output.with_suffix(".npz").exists(),
        }
        try:
            with output.with_suffix(".failure.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
        except OSError as recording_error:
            error.add_note(f"Could not preserve failure sidecar: {recording_error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    output = _fresh_output(args.output)
    result = run(output, args.model_dir)
    print(json.dumps({"terminal": result["terminal"], "output": str(output)}))
