"""Reserved feedback with or without orthogonal protection of certified contrasts.

The projection and learned basis are a disclosed global training operator.
No oracle weights, candidate filters or extra feedback enter either arm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.r4_integrated import contextual_binding_python_session as binding

SOURCE_PATHS = (
    *binding.SOURCE_PATHS,
    "experiments/r4_integrated/contextual_protected_contrast_python_session.py",
    "experiments/r4_integrated/contextual_protected_contrast_cases.json",
    "experiments/r4_integrated/contextual_protected_contrast_protocol.md",
)


def run(output: Path, model_dir: Path) -> dict:
    return binding.run(
        output,
        model_dir,
        fixture_path=binding.HERE / "contextual_protected_contrast_cases.json",
        source_paths=SOURCE_PATHS,
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
            "kind": "public development; reserved feedback versus certified-contrast protection",
            "teaching_exposure": "same linked sixteen teaching requests and 48 encounters; "
            "fresh transfer substitutions declared before outputs; no oracle weights",
            "feedback_contract": "affine primary, raw candidate fallback after STOP; "
            "actual FAIL/teacher PASS pairs only; reserved 150-offer allocation in both arms; "
            "projection and learned basis are conventional global training arithmetic",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=binding.ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(binding._json({"terminal": result["terminal"], "output": str(args.output)}))
