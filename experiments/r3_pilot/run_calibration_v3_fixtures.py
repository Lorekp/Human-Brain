"""Mechanically disabled official runner for the frozen calibration-v3 plan.

Importing this module is transport-only: it does not import native execution,
corpus, tokenizer, or historical C1 runner code.  The sole execution entry
point first obtains a source-current runnable authority.  The governing
protocol is deliberately non-runnable today, so no execution or artifact seam
can be reached.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

if __package__:
    from . import calibration_v3_authority as authority_contract
else:
    import calibration_v3_authority as authority_contract

EXIT_OK = 0
EXIT_STOP = 2


class FixtureRunnerStop(RuntimeError):
    """A fail-closed pre-outcome or protocol STOP."""


class _FailClosedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise FixtureRunnerStop(f"invalid command: {message}")


def _governing_protocol() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.calibration_v3_protocol")
    return importlib.import_module("calibration_v3_protocol")


def _gate3_authority_module() -> ModuleType:
    try:
        if __package__:
            return importlib.import_module(f"{__package__}.calibration_v3_gate3_authority")
        return importlib.import_module("calibration_v3_gate3_authority")
    except ImportError as exc:
        raise FixtureRunnerStop("governing calibration-v3 Gate3 authority is unavailable") from exc


def _source_bundle_identity(repository_root: Path) -> str:
    """Compute the bounded source closure only after protocol validation."""

    native = importlib.import_module("humanbrain.native")
    try:
        return native.procedural_runtime_source_bundle_sha256(repository_root)
    except native.NativeBuildError as exc:
        raise FixtureRunnerStop(
            f"procedural runtime source identity is unavailable: {exc}"
        ) from exc


def _source_manifest(
    repository_root: Path,
) -> tuple[dict[str, str | int], ...]:
    """Compute the bounded live source manifest after Gate3 is parsed."""

    native = importlib.import_module("humanbrain.native")
    try:
        return native.procedural_runtime_source_manifest(repository_root)
    except native.NativeBuildError as exc:
        raise FixtureRunnerStop(
            f"procedural runtime source manifest is unavailable: {exc}"
        ) from exc


def _exact_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FixtureRunnerStop(f"{label} is not a lowercase SHA-256 identity")
    return value


def _authorize_frozen_plan(
    *,
    governing: ModuleType | None = None,
    source_manifest: Callable[[Path], tuple[dict[str, str | int], ...]] | None = None,
    source_bundle_identity: Callable[[Path], str] | None = None,
) -> authority_contract.FrozenPlanAuthority:
    """Run the only permitted authority sequence, in its frozen order."""

    protocol_module = _governing_protocol() if governing is None else governing
    bundle_identity = (
        _source_bundle_identity if source_bundle_identity is None else source_bundle_identity
    )
    manifest = _source_manifest if source_manifest is None else source_manifest

    try:
        protocol = protocol_module.build_protocol()
        protocol_module.validate_protocol(protocol, require_runnable=True)
        binding = protocol["source_binding"]
        protocol_module.validate_committed_source_blobs(protocol_module.REPO_ROOT, binding)
        protocol_module.validate_source_bytes(protocol_module.REPO_ROOT, binding)
        protocol_digest = _exact_sha256("protocol identity", protocol_module.protocol_sha256())
        gate3 = _gate3_authority_module()
        try:
            source_authority = gate3.authorize_gate3_source(
                protocol_module.REPO_ROOT,
                binding,
                live_manifest=manifest,
                live_bundle_sha256=bundle_identity,
            )
        except gate3.Gate3AuthorityError as exc:
            raise FixtureRunnerStop(str(exc)) from exc
        source_digest = source_authority.source_bundle_sha256
    except FixtureRunnerStop:
        raise
    except Exception as exc:
        protocol_error = getattr(protocol_module, "ProtocolError", ())
        if isinstance(exc, protocol_error):
            raise FixtureRunnerStop(str(exc)) from exc
        raise

    try:
        return authority_contract.FrozenPlanAuthority(
            repository_root=protocol_module.REPO_ROOT,
            protocol=protocol,
            protocol_sha256=protocol_digest,
            source_bundle_identity_sha256=source_digest,
        )
    except authority_contract.FrozenPlanAuthorityError as exc:
        raise FixtureRunnerStop(str(exc)) from exc


def _electrical_v3_module() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.calibrate_minicolumn_electrical")
    return importlib.import_module("calibrate_minicolumn_electrical")


def _definition_module() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.calibration_v3_definitions")
    return importlib.import_module("calibration_v3_definitions")


def _run_electrical_v3_prerequisite() -> None:
    """Run the existing GLIF3 coupled-race sweep before any q=4 work."""

    electrical = _electrical_v3_module()
    try:
        result = electrical.run_replay()
    except electrical.CalibrationError as exc:
        raise FixtureRunnerStop(f"electrical-v3 STOP: {exc}") from exc
    if type(result) is not dict or result.get("verdict") != (
        "PASS_EXHAUSTIVE_COUPLED_SCREEN"
    ):
        raise FixtureRunnerStop("electrical-v3 STOP: coupled-race verdict did not pass")


def _run_source_bound_q4_vertical_slice(
    authority: authority_contract.FrozenPlanAuthority,
) -> dict[str, int | str]:
    """Run one real native q=4 arm after checking its frozen circuit digest."""

    protocol = authority.protocol
    screen = protocol["screens"]["compressed_reserve_q4"]
    source_binding = screen["native_vertical_slice"]
    arm = (source_binding["theta_match"], source_binding["theta_active"])
    seed = source_binding["construction_seed"]
    expected_digest = _exact_sha256(
        "q4 native circuit identity",
        source_binding["circuit_identity_sha256"],
    )
    definitions = _definition_module()
    native = importlib.import_module("humanbrain.native_temporal_runtime")
    definition = definitions.build_compressed_reserve_q4_definition(
        protocol,
        arm,
        seed,
    )
    observed_digest = definition.circuit_identity_sha256()
    if observed_digest != expected_digest:
        raise FixtureRunnerStop(
            "q4 STOP: native circuit definition identity differs from frozen authority"
        )

    milestones = {
        row["exposure"]: row for row in screen["transition_schedule"]
    }
    final_identity = None
    try:
        with native.NativeTemporalCircuit.fresh(definition) as circuit:
            for exposure, context_token in enumerate(screen["context_ids"], 1):
                context_steps = circuit.advance_block(
                    [context_token],
                    mode=native.TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                target_steps = circuit.advance_block(
                    [screen["target_token"]],
                    mode=native.TemporalStepMode.LEARN,
                    reset_before=False,
                )
                if len(context_steps) != 1 or len(target_steps) != 1:
                    raise FixtureRunnerStop(
                        f"q4 STOP at exposure {exposure}: native episode length drifted"
                    )
                target = target_steps[0]
                if (
                    target.allocated_segments != 11
                    or target.predicted_columns != 0
                    or target.punished_segments != 0
                ):
                    raise FixtureRunnerStop(
                        f"q4 STOP at exposure {exposure}: novel allocation gate failed"
                    )
                milestone = milestones.get(exposure)
                if milestone is not None:
                    identity = circuit.checkpoint_identity()
                    expected_cohorts = (
                        milestone["target_column_depth"] // 8 - 1
                    ) * 11
                    if (
                        identity.segment_count != exposure * 11
                        or identity.cohort_count != expected_cohorts
                        or identity.next_reserve_cell != milestone["reserve_cursor"]
                    ):
                        raise FixtureRunnerStop(
                            f"q4 STOP at exposure {exposure}: reserve transition drifted"
                        )
                    final_identity = identity
    except FixtureRunnerStop:
        raise
    except Exception as exc:
        raise FixtureRunnerStop(f"q4 native execution STOP: {exc}") from exc

    if (
        final_identity is None
        or final_identity.segment_count != screen["expected_target_segments"]
    ):
        raise FixtureRunnerStop("q4 STOP: final target segment count drifted")
    return {
        "circuit_identity_sha256": observed_digest,
        "cohort_count": final_identity.cohort_count,
        "next_reserve_cell": final_identity.next_reserve_cell,
        "segment_count": final_identity.segment_count,
    }


def _execute_frozen_plan(authority: authority_contract.FrozenPlanAuthority) -> int:
    """Execute the first real vertical slice, then stop before later screens."""

    _run_electrical_v3_prerequisite()
    _run_source_bound_q4_vertical_slice(authority)
    raise FixtureRunnerStop(
        "electrical-to-q4 vertical slice passed; remaining calibration-v3 execution "
        "remains disabled pre-outcome"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _FailClosedArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-frozen-plan",
        action="store_true",
        required=True,
        help="run the complete source-bound plan once it is explicitly enabled",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if not args.run_frozen_plan:  # Defensive against parser substitution.
            raise FixtureRunnerStop("the frozen-plan execution flag is required")
        authority = _authorize_frozen_plan()
        return _execute_frozen_plan(authority)
    except FixtureRunnerStop as exc:
        print(f"calibration-v3 fixture runner: STOP: {exc}", file=sys.stderr)
        return EXIT_STOP


if __name__ == "__main__":
    raise SystemExit(main())
