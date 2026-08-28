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
import json
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
_TRAIN_CHUNK_TOKENS = 16_384


class FixtureRunnerStop(RuntimeError):
    """A fail-closed pre-outcome or protocol STOP."""

    def __init__(
        self,
        message: str,
        *,
        result: dict[str, object] | None = None,
    ) -> None:
        self.result = result
        super().__init__(message)


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


def _transaction_module() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.calibration_v3_transaction")
    return importlib.import_module("calibration_v3_transaction")


def _persistence_module() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.calibration_v3_persistence")
    return importlib.import_module("calibration_v3_persistence")


def _train_input_module() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.r3_f0_train_input")
    return importlib.import_module("r3_f0_train_input")


def _train_scoring_module() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.calibration_v3_train_scoring")
    return importlib.import_module("calibration_v3_train_scoring")


def _native_temporal_module() -> ModuleType:
    return importlib.import_module("humanbrain.native_temporal_runtime")


def _open_development_artifact_store(
    authority: authority_contract.FrozenPlanAuthority,
) -> tuple[object, ModuleType]:
    """Open and durably prepare the full frozen namespace before execution."""

    persistence = _persistence_module()
    native = _native_temporal_module()
    transaction = _transaction_module()
    try:
        store = persistence.CalibrationV3ArtifactStore.open(
            repository_root=authority.repository_root,
            validated_protocol=authority.protocol,
            expected_source_bundle_identity_sha256=(
                authority.source_bundle_identity_sha256
            ),
        )
        for repository_root, relative_directory in store.durable_directory_requests():
            preparation = native.prepare_durable_directory(
                repository_root,
                Path(relative_directory),
            )
            transaction._require_durable_directory_preparation(preparation, native)
    except Exception as exc:
        raise FixtureRunnerStop(f"development artifact namespace STOP: {exc}") from exc
    return store, native


def _development_result_payload(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    artifact_path: str,
    construction_seeds: Sequence[int],
    result: dict[str, object],
    screen_id: str,
    thresholds: tuple[int, int] | None,
) -> dict[str, object]:
    protocol = authority.protocol
    return {
        "artifact_path": artifact_path,
        "claim_boundary": protocol["claim_boundary"],
        "construction_seeds": list(construction_seeds),
        "phase": "development",
        "protocol_sha256": authority.protocol_sha256,
        "result": result,
        "screen_id": screen_id,
        "source_binding": protocol["source_binding"],
        "source_bundle_identity_sha256": (
            authority.source_bundle_identity_sha256
        ),
        "thresholds": (
            {}
            if thresholds is None
            else {
                "theta_active": thresholds[1],
                "theta_match": thresholds[0],
            }
        ),
    }


def _publish_development_result(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    store: object,
    native: ModuleType,
    artifact_path: str,
    construction_seeds: Sequence[int],
    result: dict[str, object],
    screen_id: str,
    thresholds: tuple[int, int] | None,
) -> dict[str, str]:
    """Exclusively publish and reopen one exact DEVELOPMENT result envelope."""

    payload = _development_result_payload(
        authority,
        artifact_path=artifact_path,
        construction_seeds=construction_seeds,
        result=result,
        screen_id=screen_id,
        thresholds=thresholds,
    )
    try:
        encoded = store.encode_result(payload)
        destination = store.prepare_absent_result(artifact_path)
        publication = native.publish_immutable_bytes(destination, encoded)
        _transaction_module()._validate_manifest_publication(
            publication,
            encoded,
            native,
        )
        confirmed = store.confirm_exact(artifact_path, encoded)
        reopened = store.read(artifact_path)
    except Exception as exc:
        raise FixtureRunnerStop(
            f"development artifact publication STOP at {artifact_path}: {exc}"
        ) from exc
    if confirmed.payload_sha256 != reopened.payload_sha256:
        raise FixtureRunnerStop(
            f"development artifact payload SHA drift after reopen at {artifact_path}"
        )
    return {
        "artifact_path": artifact_path,
        "payload_sha256": reopened.payload_sha256,
    }


def _publish_development_electrical_results(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    store: object,
    native: ModuleType,
    replay: dict[str, object],
) -> list[dict[str, str]]:
    """Publish only DEVELOPMENT seed rows from the all-seed electrical replay."""

    protocol = authority.protocol
    seeds = tuple(protocol["threshold_selection"]["development_seeds"])
    paths = protocol["artifact_contract"]["electrical_v3"]["development"]
    rows = replay.get("seeds")
    if type(paths) is not list or len(paths) != len(seeds) or type(rows) is not dict:
        raise FixtureRunnerStop("electrical-v3 development artifact grid is incomplete")
    receipts: list[dict[str, str]] = []
    for seed, artifact_path in zip(seeds, paths, strict=True):
        row = rows.get(str(seed))
        if type(row) is not dict or row.get("seed") != seed:
            raise FixtureRunnerStop(
                f"electrical-v3 development seed {seed} result is not exact"
            )
        receipts.append(
            _publish_development_result(
                authority,
                store=store,
                native=native,
                artifact_path=artifact_path,
                construction_seeds=(seed,),
                result=row,
                screen_id="electrical_v3",
                thresholds=None,
            )
        )
    return receipts


def _run_electrical_v3_prerequisite() -> dict[str, object]:
    """Run the existing GLIF3 coupled-race sweep before any q=4 work."""

    electrical = _electrical_v3_module()
    try:
        result = electrical.run_replay()
    except electrical.CalibrationError as exc:
        raise FixtureRunnerStop(f"electrical-v3 STOP: {exc}") from exc
    if type(result) is not dict or result.get("verdict") not in {
        "PASS_RESERVE_VALID_ELECTRICAL_V3",
        "REJECT_RESERVE_VALID_ELECTRICAL_V3",
    }:
        raise FixtureRunnerStop("electrical-v3 STOP: reserve-valid verdict is not exact")
    return result


def _run_source_bound_q4_development_slice(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    arm: tuple[int, int],
    seed: int,
) -> dict[str, int | str]:
    """Run one q=4 development coordinate against its frozen native identity."""

    protocol = authority.protocol
    screen = protocol["screens"]["compressed_reserve_q4"]
    arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
    seed_slug = f"construction-seed-{seed:010d}"
    try:
        frozen_identity = screen["native_circuit_identity_sha256_by_arm_and_seed"][
            arm_slug
        ][seed_slug]
    except (KeyError, TypeError) as exc:
        raise FixtureRunnerStop(
            f"q4 STOP: no frozen circuit identity for {arm_slug}/{seed_slug}"
        ) from exc
    expected_digest = _exact_sha256(
        "q4 native circuit identity",
        frozen_identity,
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


def _run_source_bound_q4_vertical_slice(
    authority: authority_contract.FrozenPlanAuthority,
) -> dict[str, int | str]:
    """Retain the original one-point q=4 transport sentinel."""

    source_binding = authority.protocol["screens"]["compressed_reserve_q4"][
        "native_vertical_slice"
    ]
    result = _run_source_bound_q4_development_slice(
        authority,
        arm=(source_binding["theta_match"], source_binding["theta_active"]),
        seed=source_binding["construction_seed"],
    )
    if result["circuit_identity_sha256"] != _exact_sha256(
        "q4 vertical-slice circuit identity",
        source_binding["circuit_identity_sha256"],
    ):
        raise FixtureRunnerStop("q4 STOP: one-point vertical-slice authority drifted")
    return result


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
                raise FixtureRunnerStop(
                    f"q255 {regime} STOP: read-only probes mutated learned state"
                )
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


def _run_development_q4_screens(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    store: object,
    native: ModuleType,
) -> dict[str, object]:
    """Run every frozen q=4 development coordinate before q=255."""

    selection = authority.protocol["threshold_selection"]
    candidate_arms = tuple(tuple(row) for row in selection["candidate_arms"])
    development_seeds = tuple(selection["development_seeds"])
    arm_results: list[dict[str, object]] = []

    for arm in candidate_arms:
        arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
        artifact_paths = authority.protocol["artifact_contract"][
            "compressed_reserve_q4"
        ]["development"][arm_slug]
        if type(artifact_paths) is not list or len(artifact_paths) != len(
            development_seeds
        ):
            raise FixtureRunnerStop(f"q4 artifact path grid is incomplete for {arm_slug}")
        seed_results: list[dict[str, object]] = []
        for seed, artifact_path in zip(
            development_seeds,
            artifact_paths,
            strict=True,
        ):
            try:
                result = _run_source_bound_q4_development_slice(
                    authority,
                    arm=arm,
                    seed=seed,
                )
            except FixtureRunnerStop as exc:
                fixture_result: dict[str, object] = {
                    "construction_seed": seed,
                    "reason": str(exc),
                    "status": "STOP",
                }
                _publish_development_result(
                    authority,
                    store=store,
                    native=native,
                    artifact_path=artifact_path,
                    construction_seeds=(seed,),
                    result=fixture_result,
                    screen_id="compressed_reserve_q4",
                    thresholds=arm,
                )
                seed_results.append(fixture_result)
                continue
            fixture_result = {
                "construction_seed": seed,
                "status": "PASS",
                **result,
            }
            _publish_development_result(
                authority,
                store=store,
                native=native,
                artifact_path=artifact_path,
                construction_seeds=(seed,),
                result=fixture_result,
                screen_id="compressed_reserve_q4",
                thresholds=arm,
            )
            seed_results.append(fixture_result)

        arm_passed = all(result["status"] == "PASS" for result in seed_results)
        arm_results.append(
            {
                "arm_slug": arm_slug,
                "seeds": seed_results,
                "status": (
                    "PASS_ALL_DEVELOPMENT_SEEDS"
                    if arm_passed
                    else "STOP_DEVELOPMENT_SEED_SET"
                ),
                "theta_active": arm[1],
                "theta_match": arm[0],
            }
        )

    return {
        "arms": arm_results,
        "candidate_order": [list(arm) for arm in candidate_arms],
        "construction_seed_order": list(development_seeds),
        "phase": "development",
        "screen_id": "compressed_reserve_q4",
        "status": (
            "PASS_ALL_CANDIDATES"
            if all(
                arm["status"] == "PASS_ALL_DEVELOPMENT_SEEDS"
                for arm in arm_results
            )
            else "COMPLETE_WITH_REJECTED_CANDIDATES"
        ),
    }


def _run_development_q255_screens(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    q4: dict[str, object],
    store: object,
    native: ModuleType,
) -> dict[str, object]:
    """Run q=255 only for arms passing their entire q=4 seed set."""

    selection = authority.protocol["threshold_selection"]
    candidate_arms = tuple(tuple(row) for row in selection["candidate_arms"])
    development_seeds = tuple(selection["development_seeds"])
    q4_passing_arm_slugs = {
        row["arm_slug"]
        for row in q4["arms"]
        if row["status"] == "PASS_ALL_DEVELOPMENT_SEEDS"
    }
    arm_results: list[dict[str, object]] = []

    for arm in candidate_arms:
        arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
        if arm_slug not in q4_passing_arm_slugs:
            arm_results.append(
                {
                    "arm_slug": arm_slug,
                    "seeds": [],
                    "status": "NOT_RUN_Q4_DEVELOPMENT_STOP",
                    "theta_active": arm[1],
                    "theta_match": arm[0],
                }
            )
            continue

        production_paths = authority.protocol["artifact_contract"][
            "full_cohort_q255"
        ]["development"]["production-overlap-ordered-pairs"][arm_slug]
        disjoint_paths = authority.protocol["artifact_contract"][
            "full_cohort_q255"
        ]["development"]["target-disjoint-isolation-control"][arm_slug]
        if (
            type(production_paths) is not list
            or type(disjoint_paths) is not list
            or len(production_paths) != len(development_seeds)
            or len(disjoint_paths) != len(development_seeds)
        ):
            raise FixtureRunnerStop(
                f"q255 artifact path grid is incomplete for {arm_slug}"
            )
        seed_results: list[dict[str, object]] = []
        for seed, production_path, disjoint_path in zip(
            development_seeds,
            production_paths,
            disjoint_paths,
            strict=True,
        ):
            try:
                production = _run_source_bound_q255_production_slice(
                    authority,
                    arm=arm,
                    seed=seed,
                )
            except FixtureRunnerStop as exc:
                production_result: dict[str, object] = {
                    "reason": str(exc),
                    "status": "STOP",
                }
                _publish_development_result(
                    authority,
                    store=store,
                    native=native,
                    artifact_path=production_path,
                    construction_seeds=(seed,),
                    result=production_result,
                    screen_id="full_cohort_q255",
                    thresholds=arm,
                )
                seed_results.append(
                    {
                        "construction_seed": seed,
                        "production_overlap": production_result,
                        "status": "STOP_PRODUCTION_OVERLAP",
                        "target_disjoint": {"status": "NOT_RUN_FAIL_FAST"},
                    }
                )
                continue

            production_result = {"status": "PASS", **production}
            _publish_development_result(
                authority,
                store=store,
                native=native,
                artifact_path=production_path,
                construction_seeds=(seed,),
                result=production_result,
                screen_id="full_cohort_q255",
                thresholds=arm,
            )
            try:
                disjoint = _run_source_bound_q255_disjoint_slice(
                    authority,
                    arm=arm,
                    seed=seed,
                )
            except FixtureRunnerStop as exc:
                disjoint_result: dict[str, object] = {
                    "reason": str(exc),
                    "status": "STOP",
                }
                _publish_development_result(
                    authority,
                    store=store,
                    native=native,
                    artifact_path=disjoint_path,
                    construction_seeds=(seed,),
                    result=disjoint_result,
                    screen_id="full_cohort_q255",
                    thresholds=arm,
                )
                seed_results.append(
                    {
                        "construction_seed": seed,
                        "production_overlap": production_result,
                        "status": "STOP_TARGET_DISJOINT",
                        "target_disjoint": disjoint_result,
                    }
                )
                continue

            disjoint_result = {"status": "PASS", **disjoint}
            _publish_development_result(
                authority,
                store=store,
                native=native,
                artifact_path=disjoint_path,
                construction_seeds=(seed,),
                result=disjoint_result,
                screen_id="full_cohort_q255",
                thresholds=arm,
            )
            seed_results.append(
                {
                    "construction_seed": seed,
                    "production_overlap": production_result,
                    "status": "PASS_BOTH_REGIMES",
                    "target_disjoint": disjoint_result,
                }
            )

        arm_passed = all(
            result["status"] == "PASS_BOTH_REGIMES" for result in seed_results
        )
        arm_results.append(
            {
                "arm_slug": arm_slug,
                "seeds": seed_results,
                "status": (
                    "PASS_ALL_DEVELOPMENT_SEEDS"
                    if arm_passed
                    else "STOP_DEVELOPMENT_SEED_SET"
                ),
                "theta_active": arm[1],
                "theta_match": arm[0],
            }
        )

    eligible_arms = [
        arm
        for arm in arm_results
        if arm["status"] != "NOT_RUN_Q4_DEVELOPMENT_STOP"
    ]
    return {
        "arms": arm_results,
        "candidate_order": [list(arm) for arm in candidate_arms],
        "construction_seed_order": list(development_seeds),
        "eligible_arm_order": [
            [arm["theta_match"], arm["theta_active"]] for arm in eligible_arms
        ],
        "phase": "development",
        "screen_id": "full_cohort_q255",
        "status": (
            "STOP_NO_Q4_ELIGIBLE_CANDIDATES"
            if not eligible_arms
            else "PASS_ALL_Q4_ELIGIBLE_CANDIDATES"
            if all(
                arm["status"] == "PASS_ALL_DEVELOPMENT_SEEDS"
                for arm in eligible_arms
            )
            else "COMPLETE_WITH_REJECTED_Q4_ELIGIBLE_CANDIDATES"
        ),
    }


def _close_active_r3_session(session: object, transaction: ModuleType) -> None:
    """End an uncommitted scientific branch through its transaction capability."""

    try:
        session.close()
    except transaction.CalibrationV3TransactionStop:
        # Production close intentionally terminalizes and raises its immutable receipt.
        return
    except Exception as exc:
        raise FixtureRunnerStop(f"R3-F0 active-session close failed: {exc}") from exc


def _r3_result_path(
    authority: authority_contract.FrozenPlanAuthority,
    arm_slug: str,
) -> str:
    try:
        path = authority.protocol["artifact_contract"]["r3_f0_train"]["development"][
            arm_slug
        ]
    except (KeyError, TypeError) as exc:
        raise FixtureRunnerStop("R3-F0 development result path authority is incomplete") from exc
    if type(path) is not str or not path:
        raise FixtureRunnerStop("R3-F0 development result path is not exact")
    return path


def _run_source_bound_r3_f0_development(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    arm: tuple[int, int],
) -> dict[str, object]:
    """Run and transactionally commit one seed-17 TRAIN structural screen."""

    transaction = _transaction_module()
    arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
    result_path = _r3_result_path(authority, arm_slug)
    definition_box: list[object] = []

    def definition_factory() -> object:
        definition = _definition_module().build_r3_f0_definition(
            authority.protocol,
            arm,
            17,
        )
        definition_box.append(definition)
        return definition

    try:
        coordinator = transaction.CalibrationV3Transaction.from_authority(authority)
        session = coordinator.start_fresh(result_path, definition_factory)
    except transaction.CalibrationV3TransactionStop as exc:
        raise FixtureRunnerStop(f"R3-F0 {arm_slug} transaction start STOP: {exc}") from exc

    train_input: ModuleType | None = None
    try:
        if len(definition_box) != 1:
            raise FixtureRunnerStop("R3-F0 definition factory did not run exactly once")
        definition = definition_box[0]
        scoring = _train_scoring_module()
        scorer = scoring.R3F0TrainScorer(
            config=definition.config,
            token_columns=definition.token_codes,
        )

        # TRAIN authentication and every decoder/execution dependency remain inside
        # the active interval established above.
        train_input = _train_input_module()
        verified = train_input.load_verified_train_input()
        np = importlib.import_module("numpy")
        native = _native_temporal_module()
        session.check_cap()

        book_count = 0
        observed_token_count = 0
        for book in verified.iter_books():
            tokens = np.frombuffer(book.token_bytes_le_i32, dtype="<i4")
            if (
                tokens.ndim != 1
                or int(tokens.size) != book.token_count
                or int(tokens.nbytes) != len(book.token_bytes_le_i32)
                or not tokens.size
            ):
                raise FixtureRunnerStop(
                    f"R3-F0 {arm_slug} TRAIN book {book.ordinal} <i4 decode drifted"
                )
            book_count += 1
            observed_token_count += int(tokens.size)
            for start in range(0, int(tokens.size), _TRAIN_CHUNK_TOKENS):
                stop = min(start + _TRAIN_CHUNK_TOKENS, int(tokens.size))
                reset_before = start == 0
                token_chunk = tuple(int(value) for value in tokens[start:stop])
                frames = session.advance_block(
                    token_chunk,
                    mode=native.TemporalStepMode.LEARN,
                    reset_before=reset_before,
                )
                if type(frames) is not tuple or len(frames) != stop - start:
                    raise FixtureRunnerStop(
                        f"R3-F0 {arm_slug} transaction frame count drifted"
                    )
                for frame_index, frame in enumerate(frames):
                    scientific_stop = scorer.observe(
                        reset_before and frame_index == 0,
                        frame,
                    )
                    if scientific_stop is not None:
                        raise FixtureRunnerStop(
                            f"R3-F0 {arm_slug} scorer STOP before terminal identity",
                            result=scientific_stop,
                        )
                session.check_cap()

        terminal_identity = session.terminal_identity()
        structural = scorer.finalize(terminal_identity)
        if structural.get("status") != "PASS_R3_F0_TRAIN_STRUCTURE":
            raise FixtureRunnerStop(
                f"R3-F0 {arm_slug} scorer STOP at terminal identity",
                result=structural,
            )
        progress = session.progress
        if progress is None:
            raise FixtureRunnerStop("R3-F0 terminal transaction progress is absent")
        checkpoint_identity = dict(terminal_identity)
        structural_metrics = dict(structural)
        structural_metrics.pop("status")
        result = {
            "arm_slug": arm_slug,
            "comparisons": {},
            "kind": "r3_f0_train",
            "reference": {
                "checkpoint_identity": checkpoint_identity,
                "step_chain": {
                    "count": progress.step_chain.count,
                    "sha256": progress.step_chain.hexdigest,
                },
                "winner_chain": {
                    "count": progress.winner_chain.count,
                    "sha256": progress.winner_chain.hexdigest,
                },
            },
            "resumes": [],
            "structural_metrics": structural_metrics,
            "train_input": {
                "book_count": book_count,
                "held_out_access": "forbidden",
                "token_count": observed_token_count,
                "tokenizer_sha256": verified.tokenizer_sha256,
                "train_input_authority_sha256": (
                    verified.train_input_authority_sha256
                ),
                "train_stream_sha256": verified.train_stream_sha256,
            },
            "verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
        }
        receipt = coordinator.commit_r3_result(session, result)
    except FixtureRunnerStop:
        _close_active_r3_session(session, transaction)
        raise
    except transaction.CalibrationV3TransactionStop as exc:
        raise FixtureRunnerStop(f"R3-F0 {arm_slug} transaction STOP: {exc}") from exc
    except Exception as exc:
        _close_active_r3_session(session, transaction)
        train_error = () if train_input is None else getattr(train_input, "TrainInputError", ())
        if isinstance(exc, train_error):
            raise FixtureRunnerStop(f"R3-F0 {arm_slug} TRAIN authority STOP: {exc}") from exc
        raise FixtureRunnerStop(f"R3-F0 {arm_slug} execution STOP: {exc}") from exc

    return {
        "arm_slug": arm_slug,
        "artifact_path": receipt.artifact_path,
        "construction_seed": 17,
        "payload_sha256": receipt.payload_sha256,
        "status": "PASS",
        "verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
    }


def _run_development_r3_f0_screens(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    q255: dict[str, object],
) -> dict[str, object]:
    """Run seed-17 TRAIN only for exact q255 development-pass arms."""

    candidate_arms = tuple(
        tuple(row) for row in authority.protocol["threshold_selection"]["candidate_arms"]
    )
    q255_rows = q255.get("arms")
    if type(q255_rows) is not list or len(q255_rows) != len(candidate_arms):
        raise FixtureRunnerStop("q255 eligibility rows do not match frozen arm order")
    arm_results: list[dict[str, object]] = []
    eligible_order: list[list[int]] = []
    for arm, q255_row in zip(candidate_arms, q255_rows, strict=True):
        arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
        if (
            type(q255_row) is not dict
            or q255_row.get("arm_slug") != arm_slug
            or q255_row.get("theta_match") != arm[0]
            or q255_row.get("theta_active") != arm[1]
        ):
            raise FixtureRunnerStop("q255 eligibility row differs from frozen arm order")
        if q255_row.get("status") != "PASS_ALL_DEVELOPMENT_SEEDS":
            arm_results.append(
                {
                    "arm_slug": arm_slug,
                    "status": "NOT_RUN_Q255_DEVELOPMENT_STOP",
                    "theta_active": arm[1],
                    "theta_match": arm[0],
                }
            )
            continue
        eligible_order.append([arm[0], arm[1]])
        try:
            result = _run_source_bound_r3_f0_development(authority, arm=arm)
        except FixtureRunnerStop as exc:
            stopped: dict[str, object] = {
                "arm_slug": arm_slug,
                "reason": str(exc),
                "status": "STOP",
                "theta_active": arm[1],
                "theta_match": arm[0],
            }
            if exc.result is not None:
                stopped["scientific_stop"] = exc.result
            arm_results.append(stopped)
            continue
        arm_results.append(
            {
                **result,
                "theta_active": arm[1],
                "theta_match": arm[0],
            }
        )

    eligible_results = [
        row for row in arm_results if row["status"] != "NOT_RUN_Q255_DEVELOPMENT_STOP"
    ]
    return {
        "arms": arm_results,
        "candidate_order": [list(arm) for arm in candidate_arms],
        "construction_seed": 17,
        "eligible_arm_order": eligible_order,
        "phase": "development",
        "screen_id": "r3_f0_train",
        "status": (
            "STOP_NO_Q255_ELIGIBLE_CANDIDATES"
            if not eligible_results
            else "PASS_ALL_Q255_ELIGIBLE_CANDIDATES"
            if all(row["status"] == "PASS" for row in eligible_results)
            else "COMPLETE_WITH_REJECTED_Q255_ELIGIBLE_CANDIDATES"
        ),
    }


def _publish_development_selection(
    authority: authority_contract.FrozenPlanAuthority,
    *,
    store: object,
    native: ModuleType,
    q255: dict[str, object],
    r3_f0: dict[str, object],
) -> dict[str, object]:
    """Publish the first exact q255+R3 development pass, if one exists."""

    protocol = authority.protocol
    candidate_arms = tuple(
        tuple(row) for row in protocol["threshold_selection"]["candidate_arms"]
    )
    q255_rows = q255.get("arms")
    r3_rows = r3_f0.get("arms")
    if (
        type(q255_rows) is not list
        or type(r3_rows) is not list
        or len(q255_rows) != len(candidate_arms)
        or len(r3_rows) != len(candidate_arms)
    ):
        raise FixtureRunnerStop("development selection rows do not cover frozen candidates")

    verdicts: dict[str, bool] = {}
    for arm, q255_row, r3_row in zip(
        candidate_arms,
        q255_rows,
        r3_rows,
        strict=True,
    ):
        arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
        for label, row in (("q255", q255_row), ("R3-F0", r3_row)):
            if (
                type(row) is not dict
                or row.get("arm_slug") != arm_slug
                or row.get("theta_match") != arm[0]
                or row.get("theta_active") != arm[1]
            ):
                raise FixtureRunnerStop(
                    f"development selection {label} row differs from frozen arm order"
                )
        verdicts[arm_slug] = bool(
            q255_row.get("status") == "PASS_ALL_DEVELOPMENT_SEEDS"
            and r3_row.get("status") == "PASS"
            and r3_row.get("verdict") == "PASS_R3_F0_TRAIN_STRUCTURE"
        )

    selected = next(
        (
            (arm_slug, arm)
            for arm_slug, arm in zip(verdicts, candidate_arms, strict=True)
            if verdicts[arm_slug]
        ),
        None,
    )
    artifact_path = protocol["artifact_contract"]["selection"]["development"]
    if selected is None:
        return {
            "artifact_path": artifact_path,
            "selected_arm_slug": None,
            "status": "STOP_NO_PASSING_DEVELOPMENT_CANDIDATE",
            "verdicts": verdicts,
        }

    selected_arm_slug, selected_arm = selected
    receipt = _publish_development_result(
        authority,
        store=store,
        native=native,
        artifact_path=artifact_path,
        construction_seeds=tuple(protocol["threshold_selection"]["development_seeds"]),
        result={
            "selected_arm_slug": selected_arm_slug,
            "verdicts": verdicts,
        },
        screen_id="threshold_selection",
        thresholds=selected_arm,
    )
    return {
        **receipt,
        "selected_arm_slug": selected_arm_slug,
        "status": "PASS",
        "verdicts": verdicts,
    }


def _execute_frozen_plan(authority: authority_contract.FrozenPlanAuthority) -> int:
    """Execute every DEVELOPMENT screen, then stop before confirmation."""

    store, native = _open_development_artifact_store(authority)
    electrical = _run_electrical_v3_prerequisite()
    electrical_artifacts = _publish_development_electrical_results(
        authority,
        store=store,
        native=native,
        replay=electrical,
    )
    electrical_summary = {
        "development_artifacts": electrical_artifacts,
        "status": electrical["verdict"],
    }
    if electrical["verdict"] != "PASS_RESERVE_VALID_ELECTRICAL_V3":
        raise FixtureRunnerStop(
            "electrical-v3 development STOP after durable fixture publication",
            result={
                "electrical_v3": electrical_summary,
                "next_screen": "final_rejection",
                "status": "STOP_ELECTRICAL_V3_DEVELOPMENT",
            },
        )
    q4 = _run_development_q4_screens(
        authority,
        store=store,
        native=native,
    )
    q255 = _run_development_q255_screens(
        authority,
        q4=q4,
        store=store,
        native=native,
    )
    r3_f0 = _run_development_r3_f0_screens(authority, q255=q255)
    selection = _publish_development_selection(
        authority,
        store=store,
        native=native,
        q255=q255,
        r3_f0=r3_f0,
    )
    common_result = {
        "compressed_reserve_q4": q4,
        "development_selection": selection,
        "electrical_v3": electrical_summary,
        "full_cohort_q255": q255,
        "r3_f0_train": r3_f0,
    }
    if selection["status"] != "PASS":
        raise FixtureRunnerStop(
            "development selection STOP: no candidate passed q255 plus R3-F0",
            result={
                **common_result,
                "next_screen": "final_rejection",
                "status": "STOP_DEVELOPMENT_SELECTION",
            },
        )
    raise FixtureRunnerStop(
        "development screens completed; confirmation remains disabled",
        result={
            **common_result,
            "next_screen": "independent_confirmation",
            "status": "STOP_BEFORE_CONFIRMATION",
        },
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
        if exc.result is not None:
            print(json.dumps(exc.result, sort_keys=True, separators=(",", ":")))
        print(f"calibration-v3 fixture runner: STOP: {exc}", file=sys.stderr)
        return EXIT_STOP


if __name__ == "__main__":
    raise SystemExit(main())
