from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from inspect import signature

import humanbrain.variability_multiscale_source_motor_online_growth as online
import pytest
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.variability_multiscale_source_motor_controller import (
    SuccessfulVariabilityMultiscaleSourceMotorExperience,
    VariabilityMultiscaleSourceMotorAblation,
    VariabilityMultiscaleSourceMotorController,
)
from humanbrain.variability_multiscale_source_motor_online_growth import (
    AppendOnlyVariabilityMultiscaleSourceMotorStack,
    MultiscaleSourceMotorExecutionAuthority,
    MultiscaleSourceMotorSuccessEvidence,
    OnlineMultiscaleGrowthAblation,
)


def _transition(
    directory: str,
    stem: str,
    paths: tuple[str, str, str],
) -> tuple[
    bytes,
    tuple[OpaqueWorkspaceFile, ...],
    tuple[OpaqueWorkspaceFile, ...],
]:
    changed_path = f"{directory}/{stem}.py"
    if changed_path not in paths:
        raise ValueError("changed path is absent")
    retained = f"{stem}_retained\n".encode("ascii")
    before = tuple(
        OpaqueWorkspaceFile(
            path,
            (
                retained + f"{stem}_discarded\n".encode("ascii")
                if path == changed_path
                else f"opaque_{path.replace('/', '_').replace('.', '_')}\n".encode(
                    "ascii"
                )
            ),
        )
        for path in paths
    )
    after = tuple(
        OpaqueWorkspaceFile(
            item.path,
            retained if item.path == changed_path else item.content,
        )
        for item in before
    )
    query = b"retain " + stem.encode("ascii") + b" from the complete stream"
    return query, before, after


def _experience(
    provenance_id: str,
    directory: str,
    stem: str,
    paths: tuple[str, str, str],
) -> SuccessfulVariabilityMultiscaleSourceMotorExperience:
    query, before, after = _transition(directory, stem, paths)
    return SuccessfulVariabilityMultiscaleSourceMotorExperience(
        query=query,
        before_files=before,
        after_files=after,
        provenance_id=provenance_id,
    )


def _stack() -> AppendOnlyVariabilityMultiscaleSourceMotorStack:
    controller = VariabilityMultiscaleSourceMotorController.grow(
        (
            _experience(
                "seed-cedar",
                "amber",
                "cedar_entry",
                (
                    "amber/cedar_entry.py",
                    "moss/brass_slot.py",
                    "quartz/umber_slot.py",
                ),
            ),
            _experience(
                "seed-lilac",
                "harbor",
                "lilac_entry",
                (
                    "brass/ochre_slot.py",
                    "harbor/lilac_entry.py",
                    "umber/violet_slot.py",
                ),
            ),
        )
    )
    return AppendOnlyVariabilityMultiscaleSourceMotorStack.seed(controller)


def _deep_rows() -> tuple[
    tuple[bytes, tuple[OpaqueWorkspaceFile, ...], tuple[OpaqueWorkspaceFile, ...]],
    ...,
]:
    return (
        _transition(
            "amber/copper",
            "cedar_deep",
            (
                "amber/copper/cedar_deep.py",
                "moss/brass_slot.py",
                "quartz/umber_slot.py",
            ),
        ),
        _transition(
            "harbor/indigo",
            "lilac_deep",
            (
                "brass/ochre_slot.py",
                "umber/violet_slot.py",
                "harbor/indigo/lilac_deep.py",
            ),
        ),
    )


def _evidence(
    stack: AppendOnlyVariabilityMultiscaleSourceMotorStack,
) -> tuple[MultiscaleSourceMotorSuccessEvidence, ...]:
    authority = MultiscaleSourceMotorExecutionAuthority(
        evaluator=lambda **_observation: True,
        evaluator_sha256="a" * 64,
    )
    rows = _deep_rows()
    sealed = []
    for ordinal, (query, before, after) in enumerate(rows):
        receipt = authority.execute_and_issue(
            learner_checkpoint_sha256=stack.checkpoint_sha256,
            query=query,
            before_files=before,
            after_files=after,
        )
        sealed.append(
            MultiscaleSourceMotorSuccessEvidence.seal(
                query=query,
                before_files=before,
                after_files=after,
                execution_receipt=receipt,
                provenance_id=f"current-deep-{ordinal}",
            )
        )
    return tuple(sealed)


def _assert_materializes(
    query: bytes,
    before: tuple[OpaqueWorkspaceFile, ...],
    after: tuple[OpaqueWorkspaceFile, ...],
    emissions: tuple[bytes, ...],
) -> None:
    observation = MultiscaleOpaqueEventTape.observe(query, before)
    assert observation.materialize_motor_emission(emissions) == tuple(
        sorted(after, key=lambda item: item.path)
    )


def test_seed_transfers_and_unseen_deeper_structure_physically_stops() -> None:
    stack = _stack()
    checkpoint = stack.checkpoint_sha256
    query, before, after = _transition(
        "onyx",
        "vireo_entry",
        (
            "birch/cobalt_slot.py",
            "onyx/vireo_entry.py",
            "willow/yarrow_slot.py",
        ),
    )

    prior = stack.respond(query, before)
    assert prior.decision == "EMIT"
    assert prior.winning_shard_ordinal == 0
    assert prior.race.shard_scores == (2,)
    _assert_materializes(query, before, after, prior.emissions)

    deep_query, deep_before, _ = _transition(
        "outer/inner",
        "prism_entry",
        (
            "brass/copper_slot.py",
            "outer/inner/prism_entry.py",
            "sable/tawny_slot.py",
        ),
    )
    unsupported = stack.respond(deep_query, deep_before)
    assert unsupported.decision == "STOP"
    assert unsupported.emissions == ()
    assert unsupported.race.shard_scores == (0,)
    assert unsupported.race.stop_reason == "tonic_stop"
    assert unsupported.race.winner_cell_id == 0
    assert stack.checkpoint_sha256 == checkpoint

    assert tuple(
        signature(AppendOnlyVariabilityMultiscaleSourceMotorStack.respond).parameters
    ) == ("self", "query", "files", "ablation")


def test_two_current_successes_append_deeper_transfer_without_parent_replay() -> None:
    parent = _stack()
    parent_checkpoint = parent.checkpoint_sha256
    evidence = _evidence(parent)

    with pytest.raises(ValueError, match="at least two"):
        parent.extend_successes(evidence[:1])

    no_update = parent.extend_successes(
        evidence,
        ablation=OnlineMultiscaleGrowthAblation.NO_UPDATE,
    )
    assert no_update.successor is parent
    assert no_update.receipt.status == "NO_UPDATE"
    assert no_update.receipt.parent_checkpoint_sha256 == parent_checkpoint
    assert no_update.receipt.successor_checkpoint_sha256 == parent_checkpoint
    assert no_update.receipt.current_success_count == 2
    assert no_update.receipt.parent_experience_replay_count == 0
    assert no_update.receipt.raw_training_sequences_retained == 0

    grown = parent.extend_successes(evidence)
    successor = grown.successor
    assert grown.receipt.status == "APPENDED"
    assert grown.receipt.appended_shard_ordinal == 1
    assert grown.receipt.current_experience_training_rows == 2
    assert grown.receipt.parent_experience_replay_count == 0
    assert grown.receipt.raw_training_sequences_retained == 0
    assert successor.checkpoint_sha256 != parent_checkpoint
    assert parent.checkpoint_sha256 == parent_checkpoint
    assert successor.training_summary.shard_count == 2
    assert successor.training_summary.committed_current_success_count == 2
    assert successor.training_summary.parent_experience_replay_count == 0
    assert successor.training_summary.raw_training_sequences_retained == 0

    deep_query, deep_before, deep_after = _transition(
        "silver/graphite",
        "maple_fresh",
        (
            "cobalt/topaz_slot.py",
            "silver/graphite/maple_fresh.py",
            "umber/yarrow_slot.py",
        ),
    )
    deep = successor.respond(deep_query, deep_before)
    assert deep.decision == "EMIT"
    assert deep.winning_shard_ordinal == 1
    assert deep.race.shard_scores == (0, 2)
    _assert_materializes(deep_query, deep_before, deep_after, deep.emissions)

    prior_query, prior_before, prior_after = _transition(
        "zircon",
        "lumen_fresh",
        (
            "citrine/ivory_slot.py",
            "sienna/russet_slot.py",
            "zircon/lumen_fresh.py",
        ),
    )
    retained = successor.respond(prior_query, prior_before)
    assert retained.decision == "EMIT"
    assert retained.winning_shard_ordinal == 0
    assert retained.race.shard_scores == (2, 0)
    _assert_materializes(prior_query, prior_before, prior_after, retained.emissions)

    for ablation in (
        VariabilityMultiscaleSourceMotorAblation.EQUALITY_OFF,
        VariabilityMultiscaleSourceMotorAblation.PROPAGATION_OFF,
        VariabilityMultiscaleSourceMotorAblation.BOUNDARY_OFF,
    ):
        stopped = successor.respond(deep_query, deep_before, ablation=ablation)
        assert stopped.decision == "STOP"
        assert stopped.race.shard_scores == (0, 0)
        assert stopped.race.stop_reason == "tonic_stop"


def test_multiple_emitting_shards_collision_stop_without_ordinal_choice() -> None:
    seed = _stack()
    first = seed._shards[0]
    duplicate = replace(first, ordinal=1, selector_cell_id=2)
    ambiguous = AppendOnlyVariabilityMultiscaleSourceMotorStack(
        online._CONSTRUCTION_TOKEN,
        shards=(first, duplicate),
        provenance_sha256=seed.training_summary.provenance_sha256,
    )
    query, before, _ = _transition(
        "onyx",
        "vireo_entry",
        (
            "birch/cobalt_slot.py",
            "onyx/vireo_entry.py",
            "willow/yarrow_slot.py",
        ),
    )

    decision = ambiguous.respond(query, before)
    assert decision.decision == "STOP"
    assert decision.emissions == ()
    assert decision.winning_shard_ordinal is None
    assert decision.winning_selector_cell_id == 0
    assert decision.race.shard_scores == (2, 2)
    assert decision.race.stop_reason == "collision"
    assert decision.race.simultaneous_spike_cells == 2
    assert decision.race.physical_cell_steps == 4 * 600

    with pytest.raises(FrozenInstanceError):
        ambiguous._checkpoint_sha256 = "0" * 64  # type: ignore[misc]
