"""One fixed multiscale writer versus its immediate homogeneous predecessor."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated import local_memory_six_arm_harness as harness  # noqa: E402
from experiments.r4_integrated import local_memory_value_credit_v1 as previous  # noqa: E402

ARMS = previous.ARMS
file_hash, write_json = previous.file_hash, previous.write_json
duration_clock, validate_cost_launch = previous.duration_clock, previous.validate_cost_launch
roster, teaching_schedule = previous.roster, previous.teaching_schedule
project_cost, qualifies = previous.project_cost, previous.qualifies
detached_state, digest = previous.detached_state, previous.digest


@dataclass(frozen=True)
class MultiscaleSpec(previous.ValueCreditSpec):
    """Only the reviewed fixed bank is selectable by this experiment."""

    @property
    def config(self):
        config = super().config
        del config["reference_only_buffer"]
        return {
            **config,
            "value_credit": dict.fromkeys(ARMS, "direct_read_neuron_error"),
            "current_reference_revision": "86cc7232320cf4cef9337cf4f3b79a017bb55171",
            "candidate_only_buffer": "memory.controller.update_fraction",
            "writer_update_fraction": {
                arm: 0.5 if arm == "current_tagged"
                else [0.5] * 32 + [0.125] * 16 + [0.03125] * 16 for arm in ARMS
            },
            "writer_timescale": {
                arm: "scalar_half" if arm == "current_tagged" else "fixed_multiscale_32_16_16"
                for arm in ARMS
            },
        }


SPEC = MultiscaleSpec(
    schema="local-memory-multiscale-v1",
    module="experiments.r4_integrated.local_memory_multiscale_v1",
    entrypoint=Path(__file__).resolve(),
    recipe=Path(__file__).with_suffix(".md").resolve(),
)


def sources(spec=SPEC):
    paths = (
        ROOT / "experiments/r4_integrated/local_memory_value_credit_v1.py",
        ROOT / "experiments/r4_integrated/local_memory_value_credit_v1.md",
        ROOT / "experiments/r4_integrated/local_memory_multiscale_reference.py",
        ROOT / "docs/research/local-multiscale-writer-review.md",
        ROOT / "docs/research/local-multiscale-writer-sources.md",
    )
    return {**previous.sources(spec),
            **{p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}}


def new_model(spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown multiscale experiment arm")
    if arm == "current_tagged":
        from experiments.r4_integrated.local_memory_multiscale_reference import LocalMemoryLearner
    else:
        from humanbrain.local_memory_learner import LocalMemoryLearner
    config = spec.config
    return LocalMemoryLearner(
        mode="tagged" if arm == "current_tagged" else arm,
        seed=config["seed"], input_code_size=config["input_code_size"],
        hidden_size=config["hidden_size"], key_size=config["key_size"],
        value_size=config["value_size"],
    ).eval()


EXPERIMENT = harness.Experiment(spec=SPEC, new_model=new_model, sources=sources)


if __name__ == "__main__":
    harness.main(EXPERIMENT)
