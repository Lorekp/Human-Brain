"""Flatten established rustc cue extraction into one opaque unordered cue set.

The existing rustc adapter remains the environment extractor for error-code and
relative-source-half roots.  This adapter deliberately discards that two-axis
partition before the observation reaches cognition.  It does not parse rustc,
assign cue roles, or expose the active hypothesis bank.
"""

from __future__ import annotations

from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as ordered
from experiments.r3_pilot.common_unordered_cue_factor_memory_v1 import (
    UnorderedCueFactorObservationV1,
    seal_unordered_cue_factor_observation_v1,
)

SCHEMA = "COMMON_RUSTC_UNORDERED_CUE_FACTOR_ADAPTER_V1"
CLAIM_BOUNDARY = (
    "established-caller-attested-rustc-code-and-relative-source-half-extraction-"
    "flattened-into-one-opaque-unordered-cue-set-not-learned-cue-discovery-"
    "localization-diagnosis-edit-semantics-source-synthesis-swe-or-intelligence"
)


def project_rust_unordered_cue_factor_observation_v1(
    attempt: ordered.RustcDiagnosticCompileAttemptV1,
    *,
    selection_context_sha256: str,
    active_hypothesis_sha256s: tuple[str, ...],
) -> UnorderedCueFactorObservationV1:
    """Extract rustc cues, then irreversibly discard their supplied axis roles."""

    extracted = ordered.project_rust_factorized_feedback_v1(
        attempt,
        selection_context_sha256=selection_context_sha256,
        active_hypothesis_sha256s=active_hypothesis_sha256s,
    )
    cue_sha256s = tuple(
        sorted(cue_sha256 for axis in extracted.axis_cue_sha256s for cue_sha256 in axis)
    )
    return seal_unordered_cue_factor_observation_v1(
        selection_context_sha256=extracted.selection_context_sha256,
        cue_sha256s=cue_sha256s,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA",
    "project_rust_unordered_cue_factor_observation_v1",
]
