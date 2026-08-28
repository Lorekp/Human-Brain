"""Development-only q255 falsifier for the native two-tier repair.

This is deliberately separate from the committed pre-repair q255 artifact.
It uses the existing physical q255 helpers, runs production-overlap before the
target-disjoint control for every seed, and stops the arm ladder after the
first candidate passes both regimes at all four development seeds.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.r3_pilot import calibration_v3_authority as authority_contract
from experiments.r3_pilot import calibration_v3_definitions as definitions
from experiments.r3_pilot import calibration_v3_protocol as governing
from experiments.r3_pilot import run_calibration_v3_fixtures as fixtures
from humanbrain.native import loaded_library_identity
import humanbrain.native_temporal_runtime as runtime_adapter


OUTPUT_PATH = Path(__file__).with_name("q255_two_tier_postrepair_smoke.json")
ARMS = ((16, 18), (18, 20), (20, 20))
DEVELOPMENT_SEEDS = (3, 7, 17, 29)
SYNTHETIC_SOURCE_IDENTITY = "22" * 32


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _authority() -> authority_contract.FrozenPlanAuthority:
    protocol = governing.build_protocol()
    digest = hashlib.sha256(governing.canonical_json_bytes(protocol)).hexdigest()
    return authority_contract.FrozenPlanAuthority(
        repository_root=REPO_ROOT,
        protocol=protocol,
        protocol_sha256=digest,
        source_bundle_identity_sha256=SYNTHETIC_SOURCE_IDENTITY,
    )


@contextlib.contextmanager
def _synthetic_development_authority(
    authority: authority_contract.FrozenPlanAuthority,
) -> Iterator[None]:
    """Bypass only the deliberately non-runnable official producer hold."""

    original = runtime_adapter._fresh_authority
    native_authority = runtime_adapter._FreshAuthority(
        bytes.fromhex(authority.protocol_sha256),
        bytes.fromhex(authority.source_bundle_identity_sha256),
    )
    runtime_adapter._fresh_authority = lambda: native_authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _run_seed(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    arm: tuple[int, int],
    seed: int,
) -> dict[str, object]:
    protocol = authority.protocol
    production_definition = definitions.build_full_cohort_q255_production_definition(
        protocol,
        arm,
        seed,
    )
    production_identity = production_definition.circuit_identity_sha256()
    try:
        production = fixtures._run_source_bound_q255_production_slice(
            authority,
            arm=arm,
            seed=seed,
        )
    except fixtures.FixtureRunnerStop as exc:
        return {
            "seed": seed,
            "production_overlap": {
                "status": "STOP",
                "circuit_identity_sha256": production_identity,
                "reason": str(exc),
            },
            "target_disjoint": "NOT_RUN_FAIL_FAST",
            "status": "STOP_PRODUCTION_OVERLAP",
        }

    disjoint_definition = definitions.build_full_cohort_q255_disjoint_definition(
        protocol,
        arm,
        seed,
    )
    disjoint_identity = disjoint_definition.circuit_identity_sha256()
    try:
        disjoint = fixtures._run_source_bound_q255_disjoint_slice(
            authority,
            arm=arm,
            seed=seed,
        )
    except fixtures.FixtureRunnerStop as exc:
        return {
            "seed": seed,
            "production_overlap": {"status": "PASS", **production},
            "target_disjoint": {
                "status": "STOP",
                "circuit_identity_sha256": disjoint_identity,
                "reason": str(exc),
            },
            "status": "STOP_TARGET_DISJOINT",
        }

    return {
        "seed": seed,
        "production_overlap": {"status": "PASS", **production},
        "target_disjoint": {"status": "PASS", **disjoint},
        "status": "PASS_BOTH_REGIMES",
    }


def main() -> int:
    args = _parse_args()
    authority = _authority()
    arm_results: list[dict[str, object]] = []
    selected_arm: list[int] | None = None

    with _synthetic_development_authority(authority):
        for arm in ARMS:
            seeds = [
                _run_seed(authority, arm=arm, seed=seed)
                for seed in DEVELOPMENT_SEEDS
            ]
            passed = all(seed["status"] == "PASS_BOTH_REGIMES" for seed in seeds)
            arm_results.append(
                {
                    "theta_match": arm[0],
                    "theta_active": arm[1],
                    "status": (
                        "PASS_ALL_DEVELOPMENT_SEEDS"
                        if passed
                        else "STOP_DEVELOPMENT_SEED_SET"
                    ),
                    "seeds": seeds,
                }
            )
            if passed:
                selected_arm = list(arm)
                break
        native_identity = loaded_library_identity()

    for arm in ARMS[len(arm_results) :]:
        arm_results.append(
            {
                "theta_match": arm[0],
                "theta_active": arm[1],
                "status": "NOT_RUN_FIRST_POSTREPAIR_PASS_FOUND",
            }
        )

    payload = {
        "schema_version": 1,
        "artifact_id": "r3-q255-two-tier-postrepair-smoke",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "PASS_FIRST_POSTREPAIR_DEVELOPMENT_ARM"
            if selected_arm is not None
            else "STOP_NO_POSTREPAIR_DEVELOPMENT_ARM"
        ),
        "claim_boundary": {
            "scope": "post-repair q255 physical development falsifier only",
            "pre_repair_artifact_overwritten": False,
            "official_threshold_selection": False,
            "official_source_binding": False,
            "held_out_data_accessed": False,
            "c1_launched": False,
            "r3_f0_executed": False,
            "capability_evidence": False,
        },
        "authority": {
            "kind": "synthetic_test_authority",
            "protocol_runnable": False,
            "protocol_sha256": authority.protocol_sha256,
            "source_bundle_identity_sha256": authority.source_bundle_identity_sha256,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
            "native_binary_size": native_identity.binary_size,
        },
        "candidate_order": [list(arm) for arm in ARMS],
        "candidate_order_rationale": (
            "(16,18) is replayed first because it is the exact arm of the separate "
            "Genesis matching-but-not-active allocation STOP; stricter arms retain "
            "their frozen order."
        ),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "pre_repair_artifact": "experiments/r3_pilot/q255_physical_development_smoke.json",
        "arms": arm_results,
        "diagnostic_verdict": {
            "first_postrepair_q255_pass": selected_arm,
            "official_threshold_selection": False,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "first_postrepair_q255_pass": selected_arm,
                "native_source_digest": native_identity.source_digest,
                "native_binary_sha256": native_identity.binary_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if selected_arm is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
