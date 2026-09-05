"""Projected bilinear associations versus the adopted AROW evidence learner.

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
TEACHING_PATHS = dict.fromkeys(
    ("BASELINE", "PROJECTED"), HERE / "contextual_projected_binding_train.json"
)
EXTRA_SOURCE_PATHS = (
    "experiments/r4_integrated/contextual_projected_binding_python_session.py",
    "experiments/r4_integrated/contextual_projected_binding_protocol.md",
    "experiments/r4_integrated/contextual_projected_binding_cases.json",
    "experiments/r4_integrated/contextual_projected_binding_train.json",
    "experiments/r4_integrated/contextual_projected_binding_common_train.json",
    "experiments/r4_integrated/contextual_covariance_protocol.md",
    "experiments/r4_integrated/contextual_reserved_feedback_protocol.md",
    "src/humanbrain/_native_src/src/lib.rs",
)


def _fresh_output(output: Path | None) -> Path:
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = (
            ROOT / "benchmarks" / f"contextual_projected_binding_{stamp}_{uuid4().hex}.json"
        )
    output = output.resolve()
    if output.suffix != ".json":
        raise ValueError("comparison output must have a .json suffix")
    if any(output.with_suffix(suffix).exists() for suffix in (".json", ".npz", ".failure.json")):
        raise FileExistsError("choose a fresh output path; existing evidence is immutable")
    return output


def run(output: Path | None = None, model_dir: Path | None = None) -> dict:
    output = _fresh_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = HERE / "contextual_projected_binding_cases.json"
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
            arm_specs=(("BASELINE", True, "arow"), ("PROJECTED", True, "arow")),
            teaching_fixture_paths=TEACHING_PATHS,
            projected_bilinear_policies={"BASELINE": False, "PROJECTED": True},
            allocation_policies={
                "BASELINE": "reserve_future_corrections",
                "PROJECTED": "reserve_future_corrections",
            },
            protection_policies={"BASELINE": "none", "PROJECTED": "none"},
            primary_policy="respond_affine_interpolated",
            include_affine=True,
            exploration_policy="raw_candidate",
            include_rank_diagnostics=True,
            track_learning_diagnostics=True,
            track_source_competition=True,
            report_labels={
                "kind": "public development; projected bilinear associations under AROW",
                "teaching_exposure": "identical prospective mixed-length sixteen-fact TRAIN; "
                "48 encounters and teacher executions per arm; distinct common old probes "
                "and a new shared fresh cohort",
                "feedback_contract": "affine primary, raw candidate fallback after STOP; "
                "actual FAIL/teacher PASS pairs; RESERVED 150-offer limit in both arms; "
                "AROW with identity covariance and r=1; one fixed rank32 projection recipe, "
                "no parameter, wording or teacher-dose search",
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
