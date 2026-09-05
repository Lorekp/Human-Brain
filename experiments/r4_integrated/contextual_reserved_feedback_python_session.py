"""Reserve correction offers for later actual errors within the same 150-offer cap."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.r4_integrated import contextual_binding_python_session as binding

SOURCE_PATHS = (
    *binding.SOURCE_PATHS,
    "experiments/r4_integrated/contextual_reserved_feedback_python_session.py",
    "experiments/r4_integrated/contextual_reserved_feedback_cases.json",
    "experiments/r4_integrated/contextual_reserved_feedback_protocol.md",
)


def run(output: Path, model_dir: Path) -> dict:
    return binding.run(
        output,
        model_dir,
        fixture_path=binding.HERE / "contextual_reserved_feedback_cases.json",
        source_paths=SOURCE_PATHS,
        arm_specs=(("BASELINE", True, "fixed"), ("RESERVED", True, "fixed")),
        allocation_policies={"BASELINE": "baseline", "RESERVED": "reserve_future_corrections"},
        primary_policy="respond_affine_interpolated",
        include_affine=True,
        exploration_policy="raw_candidate",
        include_rank_diagnostics=True,
        track_learning_diagnostics=True,
        report_labels={
            "kind": "public development; fixed BINDING baseline versus reserved future corrections",
            "teaching_exposure": "same linked sixteen teaching requests and 48 encounters; "
            "fresh transfer substitutions declared before outputs; no oracle weights",
            "feedback_contract": "affine primary, raw candidate fallback after STOP; "
            "actual FAIL/teacher PASS pairs only; 150 maximum total pair offers; "
            "baseline 32-new cap versus one offer reserved for each remaining encounter",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=binding.ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(binding._json({"terminal": result["terminal"], "output": str(args.output)}))
