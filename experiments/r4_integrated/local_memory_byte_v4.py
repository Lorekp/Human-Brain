"""Precise duration measurement for the unchanged local memory/byte workload."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated.local_memory_byte_harness import (  # noqa: E402
    ExperimentSpec,
    main,
)

SPEC = ExperimentSpec(
    schema="local-memory-byte-v4",
    module="experiments.r4_integrated.local_memory_byte_v4",
    entrypoint=Path(__file__).resolve(),
    recipe=Path(__file__).with_suffix(".md").resolve(),
)


if __name__ == "__main__":
    main(SPEC)
