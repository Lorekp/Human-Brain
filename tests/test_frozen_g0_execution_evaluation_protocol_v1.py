from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import frozen_g0_execution_evaluation_protocol_v1 as protocol
from experiments.g0 import frozen_regulatory_adapter_v1 as regulatory
from experiments.g0 import spatial_candidate_adapter_v1 as spatial


def test_frozen_execution_evaluation_protocol_is_raw_deterministic_and_unspent() -> None:
    assert protocol.ZERO_EXECUTION_RECORD == {
        "generator": 0,
        "cohort_generation": 0,
        "adjudicative_seed": 0,
        "learner": 0,
        "evaluator": 0,
        "bootstrap": 0,
        "outcome": 0,
        "dev": 0,
        "m0": 0,
        "protected": 0,
    }
    assert protocol.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA == 0
    assert sha256(Path(protocol.__file__).read_bytes()).hexdigest() == protocol.SOURCE_SHA256
    assert len(protocol.CONFIG_SHA256) == len(protocol.PROTOCOL_ROOT_SHA256) == 64

    noop = regulatory.ActionEvent(regulatory.ACTION_NOOP)
    decoded_noop = protocol.decode_physical_action(
        protocol.encode_regulatory_action(noop), expected_domain="regulatory"
    )
    assert decoded_noop.regulatory == noop
    clamp = regulatory.ActionEvent(regulatory.ACTION_CLAMP, 0x0102_0304, 1)
    decoded_clamp = protocol.decode_physical_action(
        protocol.encode_regulatory_action(clamp), expected_domain="regulatory"
    )
    assert decoded_clamp.regulatory == clamp
    decoded_spatial = protocol.decode_physical_action(
        protocol.encode_spatial_action(5), expected_domain="spatial"
    )
    assert decoded_spatial.spatial_action_id == 5
    with pytest.raises(
        protocol.FrozenProtocolStop,
        match="STOP_G0_PHYSICAL_ACTION_SEQUENCE_INCOMPLETE",
    ):
        protocol.decode_physical_action(b"\x05", expected_domain="spatial")
    with pytest.raises(
        protocol.FrozenProtocolStop,
        match="STOP_G0_PHYSICAL_ACTION_DOMAIN_MISMATCH",
    ):
        protocol.decode_physical_action(
            protocol.encode_spatial_action(5), expected_domain="regulatory"
        )

    query_seed = protocol.derive_query_seed(
        partition="adjudicative",
        learner_seed=17,
        system_index=3,
        query_index=4,
    )
    assert query_seed == protocol.derive_query_seed(
        partition="adjudicative",
        learner_seed=17,
        system_index=3,
        query_index=4,
    )
    assert query_seed != protocol.derive_query_seed(
        partition="adjudicative",
        learner_seed=17,
        system_index=3,
        query_index=4,
        renamed=True,
    )
    assert protocol.derive_episode_seed(
        partition="learning", learner_seed=29, layout_index=2, episode_index=7
    ) != protocol.derive_rollout_seed(learner_seed=29, layout_index=2, rollout_index=7)

    def prefix(first: int, target_value: int) -> tuple[regulatory.RegulatoryEvent, ...]:
        states = tuple(
            (10 + index, target_value if index == 0 else first % 2) for index in range(8)
        )
        return (
            regulatory.ObservationEvent(states, first),
            regulatory.QueryEvent(10, regulatory.ANSWER_SLOT_BINARY),
        )

    support = (
        protocol.RegulatorySupportExample(prefix(0, 0), 0),
        protocol.RegulatorySupportExample(prefix(1, 1), 1),
        protocol.RegulatorySupportExample(prefix(2, 1), 1),
    )
    query = protocol.RegulatoryQuery(prefix(3, 1))
    predictions = {
        name: protocol.predict_regulatory_baseline(name, support, query, seed=7)
        for name in protocol.REGULATORY_BASELINES
    }
    assert predictions["target-state-persistence"] == 1
    assert predictions["support-marginal-majority"] == 1
    assert set(predictions.values()) <= {0, 1}

    wall_image = bytearray(spatial.IMAGE_BYTE_COUNT)
    wall_image[((5 * 7) + 3) * 3] = 2
    wall_observation = protocol.SpatialObservation(bytes(wall_image), 0)
    support_step = spatial.SpatialStep(
        image=bytes(wall_image),
        direction=0,
        action_id=2,
        reward=1.0,
        terminated=True,
        truncated=False,
    )
    policies = {
        name: protocol.fit_spatial_baseline(name, ((support_step,),), seed=11, transition_budget=1)
        for name in protocol.SPATIAL_BASELINES
    }
    assert policies["fixed-wall-follower"].act(wall_observation, 0) == 0
    assert policies["nearest-observation-action-memorizer"].act(wall_observation, 0) == 2
    assert policies["memoryless-feed-forward-policy-matched-transition-budget"].update_count == 1
    assert all(
        policy.act(wall_observation, 1) in spatial.RAW_ACTION_IDS for policy in policies.values()
    )

    @dataclass
    class Runtime:
        learning: bool = True
        cells: bool = True
        resets: int = 0
        permutation: tuple[int, ...] | None = None

        def set_learning_enabled(self, enabled: bool) -> None:
            self.learning = enabled

        def set_cell_and_output_populations_enabled(self, enabled: bool) -> None:
            self.cells = enabled

        def reset_working_and_persistent_memory(self) -> None:
            self.resets += 1

        def set_representation_output_derangement(
            self, permutation: tuple[int, ...] | None
        ) -> None:
            self.permutation = permutation

    runtime = Runtime()
    memory_ablation = protocol.focused_ablation(runtime, protocol.ABLATION_MEMORY_RESET)
    memory_ablation.prepare_from_same_checkpoint()
    memory_ablation.before_observation()
    assert runtime.resets == 1
    deranged = Runtime()
    derangement = protocol.focused_ablation(deranged, protocol.ABLATION_DERANGEMENT)
    derangement.prepare_from_same_checkpoint()
    assert deranged.permutation == protocol.FIXED_DERANGEMENT
    assert all(index != value for index, value in enumerate(protocol.FIXED_DERANGEMENT))

    candidate = tuple(
        protocol.ScoreRecord(
            domain="regulatory",
            learner_seed=seed,
            unit_id=f"unit-{unit}",
            item_id="item-0",
            strata=("next-state-prediction",),
            success=int(unit == 0),
        )
        for seed in (17, 29)
        for unit in range(2)
    )
    baseline = tuple(
        protocol.ScoreRecord(
            domain=row.domain,
            learner_seed=row.learner_seed,
            unit_id=row.unit_id,
            item_id=row.item_id,
            strata=row.strata,
            success=0,
        )
        for row in candidate
    )
    first = protocol.stratified_bootstrap(
        candidate,
        "all-heldout-queries",
        paired_records=baseline,
        resamples=32,
    )
    second = protocol.stratified_bootstrap(
        candidate,
        "all-heldout-queries",
        paired_records=baseline,
        resamples=32,
    )
    assert first == second
    assert first.point_estimate == 0.5
    assert protocol.synthetic_self_check() == {
        "synthetic_record_count": 4,
        "point_estimate": 0.5,
        "resamples": 32,
        "adjudicative_seed_execution_count": 0,
        "evaluator_execution_count": 0,
        "bootstrap_execution_count": 0,
    }
