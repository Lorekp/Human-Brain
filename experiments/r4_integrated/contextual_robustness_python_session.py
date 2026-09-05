"""One unchanged-policy comparison on a predeclared robustness stream.

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
STREAMS = ("wording", "questions", "competition")
EXTRA_SOURCE_PATHS = (
    "experiments/r4_integrated/contextual_robustness_python_session.py",
    "experiments/r4_integrated/contextual_robustness_protocol.md",
    "experiments/r4_integrated/contextual_protected_contrast_protocol.md",
    "experiments/r4_integrated/contextual_reserved_feedback_protocol.md",
    *(f"experiments/r4_integrated/contextual_robustness_{stream}_cases.json" for stream in STREAMS),
)


def _fresh_output(stream: str, output: Path | None) -> Path:
    if stream not in STREAMS:
        raise ValueError("unknown robustness stream")
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "benchmarks" / f"contextual_robustness_{stream}_{stamp}_{uuid4().hex}.json"
    output = output.resolve()
    if output.suffix != ".json":
        raise ValueError("comparison output must have a .json suffix")
    if any(output.with_suffix(suffix).exists() for suffix in (".json", ".npz", ".failure.json")):
        raise FileExistsError("choose a fresh output path; existing evidence is immutable")
    return output


def run(stream: str, output: Path | None = None, model_dir: Path | None = None) -> dict:
    output = _fresh_output(stream, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture_path = HERE / f"contextual_robustness_{stream}_cases.json"
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
            arm_specs=(("RESERVED", True, "fixed"), ("PROTECTED", True, "fixed")),
            allocation_policies={
                "RESERVED": "reserve_future_corrections",
                "PROTECTED": "reserve_future_corrections",
            },
            protection_policies={"RESERVED": "none", "PROTECTED": "orthogonal_certified"},
            primary_policy="respond_affine_interpolated",
            include_affine=True,
            exploration_policy="raw_candidate",
            include_rank_diagnostics=True,
            track_learning_diagnostics=True,
            track_source_competition=True,
            report_labels={
                "kind": f"public development; unchanged-policy robustness stream: {stream}",
                "teaching_exposure": "same linked sixteen teaching requests and 48 encounters; "
                "three fresh streams frozen before outputs; no oracle weights",
                "feedback_contract": "affine primary, raw candidate fallback after STOP; "
                "actual FAIL/teacher PASS pairs only; reserved 150-offer allocation in both arms; "
                "projection and learned basis are conventional global training arithmetic",
            },
        )
    except Exception as error:
        # The shared harness preserves its full partial report in finally. This
        # small sidecar also survives failures before that report can be written.
        failure = {
            "terminal": "error",
            "stream": stream,
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
    parser.add_argument("--stream", choices=STREAMS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    output = _fresh_output(args.stream, args.output)
    result = run(args.stream, output, args.model_dir)
    print(
        json.dumps({"terminal": result["terminal"], "stream": args.stream, "output": str(output)})
    )
