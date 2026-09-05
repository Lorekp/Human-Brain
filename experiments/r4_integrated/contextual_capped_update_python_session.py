"""Fixed versus unit-margin-capped BINDING updates in the affine action loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.r4_integrated import contextual_binding_python_session as binding

SOURCE_PATHS = (
    *binding.SOURCE_PATHS,
    "experiments/r4_integrated/contextual_capped_update_python_session.py",
    "experiments/r4_integrated/contextual_capped_update_cases.json",
    "experiments/r4_integrated/contextual_capped_update_protocol.md",
)


def run(output: Path, model_dir: Path) -> dict:
    return binding.run(
        output,
        model_dir,
        fixture_path=binding.HERE / "contextual_capped_update_cases.json",
        source_paths=SOURCE_PATHS,
        arm_specs=(("FIXED", True, "fixed"), ("CAPPED", True, "unit_margin_capped")),
        primary_policy="respond_affine_interpolated",
        include_affine=True,
        exploration_policy="raw_candidate",
        include_rank_diagnostics=True,
        track_learning_diagnostics=True,
        report_labels={
            "kind": "public development; fixed versus unit-margin-capped BINDING updates",
            "teaching_exposure": "same linked sixteen teaching requests and 48 encounters; "
            "fresh transfer substitutions declared before outputs",
            "feedback_contract": "affine neuronal primary; after STOP execute unique raw "
            "candidate, otherwise external cosine; actual FAIL/teacher PASS pairs and "
            "certified replay only; eta0=1, same source exposure and candidate family",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=binding.ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(binding._json({"terminal": result["terminal"], "output": str(args.output)}))
