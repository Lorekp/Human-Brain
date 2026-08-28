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
    if type(result) is not dict or result.get("verdict") != ("PASS_RESERVE_VALID_ELECTRICAL_V3"):
        raise FixtureRunnerStop("electrical-v3 STOP: reserve-valid verdict did not pass")


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

    milestones = {row["exposure"]: row for row in screen["transition_schedule"]}
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
                    expected_cohorts = (milestone["target_column_depth"] // 8 - 1) * 11
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

    if final_identity is None or final_identity.segment_count != screen["expected_target_segments"]:
        raise FixtureRunnerStop("q4 STOP: final target segment count drifted")
    return {
        "circuit_identity_sha256": observed_digest,
        "cohort_count": final_identity.cohort_count,
        "next_reserve_cell": final_identity.next_reserve_cell,
        "segment_count": final_identity.segment_count,
    }


def _q255_step_cells(step: object, field: str) -> tuple[int, ...]:
    return tuple(spike.cell_id for spike in getattr(step, field))


def _q255_overlap_gate(
    *,
    regime: str,
    source_winners: Sequence[tuple[int, ...]],
    source_contexts: Sequence[tuple[int, ...]],
    theta_match: int,
) -> int:
    """Validate the frozen winner/relay collision bound."""

    if len(set(source_contexts)) != len(source_contexts):
        raise FixtureRunnerStop(f"q255 {regime} STOP: source contexts are not unique")
    winner_sets = tuple(frozenset(winners) for winners in source_winners)
    context_sets = tuple(frozenset(context) for context in source_contexts)
    maximum_winner_overlap = 0
    for right_index, right_winners in enumerate(winner_sets):
        right_context = context_sets[right_index]
        for left_index in range(right_index):
            winner_overlap = len(right_winners.intersection(winner_sets[left_index]))
            physical_overlap = len(right_context.intersection(context_sets[left_index]))
            if physical_overlap != 2 * winner_overlap:
                raise FixtureRunnerStop(
                    f"q255 {regime} STOP: winner/relay overlap is not exactly doubled"
                )
            maximum_winner_overlap = max(maximum_winner_overlap, winner_overlap)
    if maximum_winner_overlap >= (theta_match + 1) // 2:
        raise FixtureRunnerStop(
            f"q255 {regime} STOP: winner overlap reaches the frozen rejection bound"
        )
    return maximum_winner_overlap


def _run_full_cohort_q255_regime(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    regime: str,
    definition: object,
    sources: Sequence[tuple[int, ...]],
    novel_sources: Sequence[tuple[int, ...]],
    target_token: int,
) -> dict[str, int | str]:
    """Execute one bounded physical q=255 regime without publishing artifacts."""

    protocol = authority.protocol
    screen = protocol["screens"]["full_cohort_q255"]
    expected_exposures = screen["target_exposures"]
    if len(sources) != expected_exposures or len(set(sources)) != expected_exposures:
        raise FixtureRunnerStop(f"q255 {regime} STOP: source schedule is not exact and unique")

    native = importlib.import_module("humanbrain.native_temporal_runtime")
    observed_digest = definition.circuit_identity_sha256()
    source_winners: list[tuple[int, ...]] = []
    source_contexts: list[tuple[int, ...]] = []
    target_winners: list[tuple[int, ...]] = []
    base_full_identity = None
    final_identity = None
    try:
        with native.NativeTemporalCircuit.fresh(definition) as circuit:
            for exposure, source in enumerate(sources, 1):
                source_steps = circuit.advance_block(
                    source,
                    mode=native.TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                if len(source_steps) != len(source):
                    raise FixtureRunnerStop(
                        f"q255 {regime} STOP at exposure {exposure}: source length drifted"
                    )
                source_step = source_steps[-1]
                target_steps = circuit.advance_block(
                    [target_token],
                    mode=native.TemporalStepMode.LEARN,
                    reset_before=False,
                )
                if len(target_steps) != 1:
                    raise FixtureRunnerStop(
                        f"q255 {regime} STOP at exposure {exposure}: target length drifted"
                    )
                target = target_steps[0]
                if (
                    target.matching_segments != 0
                    or target.active_segments != 0
                    or target.predicted_columns != 0
                    or target.allocated_segments != 11
                    or target.punished_segments != 0
                ):
                    raise FixtureRunnerStop(
                        f"q255 {regime} STOP at exposure {exposure}: novel allocation gate "
                        f"failed (matching={target.matching_segments}, "
                        f"active={target.active_segments}, "
                        f"predicted_columns={target.predicted_columns}, "
                        f"allocated={target.allocated_segments}, "
                        f"punished={target.punished_segments})"
                    )

                winners = _q255_step_cells(source_step, "winner_spikes")
                relays = _q255_step_cells(source_step, "winner_relay_spikes")
                context = tuple(source_step.context)
                emitted = _q255_step_cells(target, "winner_spikes")
                if (
                    len(winners) != 11
                    or len(relays) != 11
                    or len(context) != 22
                    or len(emitted) != 11
                    or set(winners).intersection(relays)
                    or set(context) != set(winners).union(relays)
                    or context != tuple(source_step.winner_context)
                ):
                    raise FixtureRunnerStop(
                        f"q255 {regime} STOP at exposure {exposure}: physical context drifted"
                    )
                source_winners.append(winners)
                source_contexts.append(context)
                target_winners.append(emitted)

                if exposure == screen["base_full_exposure"]:
                    base_full_identity = circuit.checkpoint_identity()

            final_identity = circuit.checkpoint_identity()
            if len(set(target_winners)) != expected_exposures:
                raise FixtureRunnerStop(f"q255 {regime} STOP: target winner tuples are not unique")
            maximum_winner_overlap = _q255_overlap_gate(
                regime=regime,
                source_winners=source_winners,
                source_contexts=source_contexts,
                theta_match=definition.config.theta_match,
            )

            for recurrence, (source, expected_winners) in enumerate(
                zip(sources, target_winners, strict=True),
                1,
            ):
                replay_steps = circuit.advance_block(
                    (*source, target_token),
                    mode=native.TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                replay = replay_steps[-1]
                if (
                    _q255_step_cells(replay, "winner_spikes") != expected_winners
                    or replay.matching_segments != 11
                    or replay.active_segments != 11
                    or replay.predicted_columns != 11
                    or replay.allocated_segments != 0
                    or replay.punished_segments != 0
                ):
                    raise FixtureRunnerStop(
                        f"q255 {regime} STOP at recurrence {recurrence}: stored owners "
                        f"drifted (winners_equal="
                        f"{_q255_step_cells(replay, 'winner_spikes') == expected_winners}, "
                        f"matching={replay.matching_segments}, "
                        f"active={replay.active_segments}, "
                        f"predicted_columns={replay.predicted_columns})"
                    )

            before_novel = circuit.checkpoint_identity()
            for probe, source in enumerate(novel_sources, 1):
                novel = circuit.advance_block(
                    (*source, target_token),
                    mode=native.TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )[-1]
                if (
                    novel.matching_segments != 0
                    or novel.active_segments != 0
                    or novel.predicted_columns != 0
                    or novel.allocated_segments != 0
                    or novel.punished_segments != 0
                ):
                    raise FixtureRunnerStop(
                        f"q255 {regime} STOP at novel probe {probe}: false match or prediction"
                    )
            after_novel = circuit.checkpoint_identity()
            if (
                after_novel.segment_count != before_novel.segment_count
                or after_novel.semantic_index_sha256 != before_novel.semantic_index_sha256
            ):
                raise FixtureRunnerStop(f"q255 {regime} STOP: read-only probes mutated learned state")
    except FixtureRunnerStop:
        raise
    except Exception as exc:
        raise FixtureRunnerStop(f"q255 {regime} native execution STOP: {exc}") from exc

    base_cells = definition.config.column_count * definition.config.base_cells_per_column
    if (
        base_full_identity is None
        or base_full_identity.segment_count != screen["base_full_exposure"] * 11
        or base_full_identity.cohort_count != 0
        or base_full_identity.next_reserve_cell != base_cells
    ):
        raise FixtureRunnerStop(f"q255 {regime} STOP: exposure 2040 base-load gate failed")
    if (
        final_identity is None
        or final_identity.segment_count != screen["expected_target_segments"]
        or final_identity.cohort_count != 11
        or final_identity.next_reserve_cell != screen["expected_reserve_cursor"]
        or final_identity.poison_code != 0
    ):
        raise FixtureRunnerStop(f"q255 {regime} STOP: reserve transition drifted")
    return {
        "circuit_identity_sha256": observed_digest,
        "learned_recurrences": len(sources),
        "maximum_winner_overlap": maximum_winner_overlap,
        "next_reserve_cell": final_identity.next_reserve_cell,
        "novel_probes": len(novel_sources),
        "segment_count": final_identity.segment_count,
        "unique_source_contexts": len(set(source_contexts)),
        "unique_target_winner_tuples": len(set(target_winners)),
    }


def _run_source_bound_q255_production_slice(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    arm: tuple[int, int],
    seed: int,
) -> dict[str, int | str]:
    """Run one explicit q=255 production-overlap development candidate."""

    protocol = authority.protocol
    definitions = _definition_module()
    governing = _governing_protocol()
    production_sources = tuple(tuple(pair) for pair in governing.production_overlap_pairs())
    return _run_full_cohort_q255_regime(
        authority,
        regime="production-overlap",
        definition=definitions.build_full_cohort_q255_production_definition(
            protocol,
            arm,
            seed,
        ),
        sources=production_sources,
        novel_sources=(),
        target_token=governing.PRODUCTION_TARGET_TOKEN,
    )


def _run_source_bound_q255_disjoint_slice(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    arm: tuple[int, int],
    seed: int,
) -> dict[str, int | str]:
    """Run the target-disjoint control for one production-passing candidate."""

    protocol = authority.protocol
    definitions = _definition_module()
    governing = _governing_protocol()
    learned, novel = governing.target_disjoint_context_ids()
    return _run_full_cohort_q255_regime(
        authority,
        regime="target-disjoint",
        definition=definitions.build_full_cohort_q255_disjoint_definition(
            protocol,
            arm,
            seed,
        ),
        sources=tuple((token_id,) for token_id in learned),
        novel_sources=tuple((token_id,) for token_id in novel),
        target_token=governing.DISJOINT_TARGET_TOKEN,
    )


def _run_source_bound_q255_vertical_slices(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    arm: tuple[int, int],
    seed: int,
) -> dict[str, dict[str, int | str]]:
    """Run both q=255 regimes for one explicit development candidate."""

    production = _run_source_bound_q255_production_slice(
        authority,
        arm=arm,
        seed=seed,
    )
    disjoint = _run_source_bound_q255_disjoint_slice(
        authority,
        arm=arm,
        seed=seed,
    )
    return {"production_overlap": production, "target_disjoint": disjoint}


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
