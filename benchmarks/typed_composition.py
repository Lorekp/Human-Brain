"""Plan or explicitly execute the frozen typed-composition A0/A1/A2 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from humanbrain.typed_composition_runner import (  # noqa: E402
    build_execution_plan,
    execute_protocol,
)

PROTOCOL = ROOT / "benchmarks" / "typed_composition_protocol_v1.json"
EVIDENCE_ROOT = ROOT / "artifacts" / "typed_composition_a012_v1" / "evidence"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-protocol-sha256")
    parser.add_argument("--confirm-source-bundle-sha256")
    parser.add_argument("--run-id")
    arguments = parser.parse_args(argv)
    plan = build_execution_plan(ROOT, PROTOCOL)
    if arguments.execute:
        if arguments.run_id is None:
            parser.error("--execute requires a unique --run-id")
        evaluation = execute_protocol(
            ROOT,
            PROTOCOL,
            evidence_root=EVIDENCE_ROOT,
            run_id=arguments.run_id,
            confirmation=arguments.confirm_protocol_sha256,
            source_bundle_confirmation=arguments.confirm_source_bundle_sha256,
        )
        print(json.dumps(evaluation.record, sort_keys=True, indent=2, allow_nan=False))
        return 0 if evaluation.a2_eligible else 2
    print(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
