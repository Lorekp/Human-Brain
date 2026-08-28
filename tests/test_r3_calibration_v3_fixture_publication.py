from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from experiments.r3_pilot import calibration_v3_authority as authority_contract
from experiments.r3_pilot import calibration_v3_persistence as persistence
from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot import run_calibration_v3_fixtures as runner
from humanbrain import native_temporal_runtime as native

PASS_ELECTRICAL = "PASS_RESERVE_VALID_ELECTRICAL_V3"


def _authority(tmp_path: Path) -> authority_contract.FrozenPlanAuthority:
    protocol = protocol_authority.build_protocol()
    protocol["source_binding"]["runnable"] = True
    protocol_sha256 = hashlib.sha256(
        protocol_authority.canonical_json_bytes(protocol)
    ).hexdigest()
    return authority_contract.FrozenPlanAuthority(
        repository_root=tmp_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        source_bundle_identity_sha256="a" * 64,
    )


def _install_native_publication(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    def prepare(root: Path, relative: Path) -> object:
        events.append(f"prepare:{relative.as_posix()}")
        (root / relative).mkdir(parents=True, exist_ok=True)
        count = len(relative.parts)
        return native.TemporalDurableDirectoryPreparation(
            outcome=native.TemporalDurabilityOutcome.DURABLE,
            primary_error=None,
            cleanup_errors=(),
            required_component_count=count,
            confirmed_component_count=count,
            created_component_count=count,
            failed_component_index=None,
        )

    def publish(destination: Path, payload: bytes) -> object:
        events.append(f"publish:{destination.as_posix()}")
        with destination.open("xb") as handle:
            handle.write(payload)
        return native.TemporalImmutableBytesPublication(
            outcome=native.TemporalPublicationOutcome.DURABLE_PUBLISHED,
            temp_cleanup=native.TemporalPublicationCleanup.CLEAN,
            primary_error=None,
            cleanup_errors=(),
            owned_temp_pid=1,
            owned_temp_nonce=1,
            payload_bytes=len(payload),
            written_bytes=len(payload),
        )

    monkeypatch.setattr(native, "prepare_durable_directory", prepare)
    monkeypatch.setattr(native, "publish_immutable_bytes", publish)
    monkeypatch.setattr(runner, "_native_temporal_module", lambda: native)


def _electrical_row(seed: int) -> dict[str, object]:
    return {
        "boundaries": {},
        "fixture": {},
        "monotonicity": {},
        "proximal_channel": {},
        "seed": seed,
        "verdict": PASS_ELECTRICAL,
        "winner_relay": {},
    }


def _q4_pass(seed: int) -> dict[str, object]:
    return {
        "circuit_identity_sha256": f"{seed % 10}" * 64,
        "cohort_count": 176,
        "next_reserve_cell": 34_184,
        "segment_count": 1_419,
    }


def _q255_pass(seed: int) -> dict[str, object]:
    return {
        "circuit_identity_sha256": f"{seed % 10}" * 64,
        "learned_recurrences": 2_041,
        "maximum_winner_overlap": 3,
        "next_reserve_cell": 32_856,
        "novel_probes": 0,
        "segment_count": 22_451,
        "unique_source_contexts": 2_041,
        "unique_target_winner_tuples": 2_041,
    }


def test_development_coordinates_publish_once_with_exact_absence_and_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path)
    protocol = authority.protocol
    arms = tuple(tuple(row) for row in protocol["threshold_selection"]["candidate_arms"])
    seeds = tuple(protocol["threshold_selection"]["development_seeds"])
    events: list[str] = []
    _install_native_publication(monkeypatch, events)

    all_electrical_seeds = tuple(
        protocol["screens"]["electrical_v3"]["construction_seeds"]
    )
    monkeypatch.setattr(
        runner,
        "_run_electrical_v3_prerequisite",
        lambda: events.append("electrical")
        or {
            "seeds": {
                str(seed): _electrical_row(seed) for seed in all_electrical_seeds
            },
            "verdict": PASS_ELECTRICAL,
        },
    )

    def q4(
        _authority: authority_contract.FrozenPlanAuthority,
        *,
        arm: tuple[int, int],
        seed: int,
    ) -> dict[str, object]:
        if arm == arms[0] and seed == seeds[0]:
            raise runner.FixtureRunnerStop("synthetic q4 STOP")
        return _q4_pass(seed)

    def production(
        _authority: authority_contract.FrozenPlanAuthority,
        *,
        arm: tuple[int, int],
        seed: int,
    ) -> dict[str, object]:
        if arm == arms[1] and seed == seeds[0]:
            raise runner.FixtureRunnerStop("synthetic production STOP")
        return _q255_pass(seed)

    def disjoint(
        _authority: authority_contract.FrozenPlanAuthority,
        *,
        arm: tuple[int, int],
        seed: int,
    ) -> dict[str, object]:
        if arm == arms[2] and seed == seeds[0]:
            raise runner.FixtureRunnerStop("synthetic disjoint STOP")
        return _q255_pass(seed)

    monkeypatch.setattr(runner, "_run_source_bound_q4_development_slice", q4)
    monkeypatch.setattr(runner, "_run_source_bound_q255_production_slice", production)
    monkeypatch.setattr(runner, "_run_source_bound_q255_disjoint_slice", disjoint)

    def r3(
        _authority: authority_contract.FrozenPlanAuthority,
        *,
        q255: dict[str, object],
    ) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for index, arm in enumerate(arms):
            arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
            if index == len(arms) - 1:
                rows.append(
                    {
                        "arm_slug": arm_slug,
                        "status": "PASS",
                        "theta_active": arm[1],
                        "theta_match": arm[0],
                        "verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
                    }
                )
            else:
                rows.append(
                    {
                        "arm_slug": arm_slug,
                        "status": "NOT_RUN_Q255_DEVELOPMENT_STOP",
                        "theta_active": arm[1],
                        "theta_match": arm[0],
                    }
                )
        assert q255["arms"][-1]["status"] == "PASS_ALL_DEVELOPMENT_SEEDS"
        return {"arms": rows, "status": "SYNTHETIC_R3_COMPLETE"}

    monkeypatch.setattr(runner, "_run_development_r3_f0_screens", r3)
    monkeypatch.setattr(
        runner,
        "_run_independent_confirmation",
        lambda _authority, **_kwargs: {
            "status": "PASS_INDEPENDENT_CONFIRMATION"
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_deterministic_confirmation",
        lambda _authority, **_kwargs: {
            "status": "PASS_DETERMINISTIC_CONFIRMATION"
        },
    )

    assert runner._execute_frozen_plan(authority) == runner.EXIT_OK
    selected_slug = f"theta-m-{arms[-1][0]}_theta-a-{arms[-1][1]}"
    assert max(index for index, event in enumerate(events) if event.startswith("prepare:")) < (
        events.index("electrical")
    )

    contract = protocol["artifact_contract"]
    for path in contract["electrical_v3"]["development"]:
        assert (tmp_path / path).is_file()
    for path in contract["electrical_v3"]["independent_confirmation"]:
        assert not (tmp_path / path).exists()
    for arm in arms:
        arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
        for path in contract["compressed_reserve_q4"]["development"][arm_slug]:
            assert (tmp_path / path).is_file()

    q255_contract = contract["full_cohort_q255"]["development"]
    ineligible_slug = f"theta-m-{arms[0][0]}_theta-a-{arms[0][1]}"
    for regime in q255_contract.values():
        for path in regime[ineligible_slug]:
            assert not (tmp_path / path).exists()

    production_stopped_slug = f"theta-m-{arms[1][0]}_theta-a-{arms[1][1]}"
    production_paths = q255_contract["production-overlap-ordered-pairs"][
        production_stopped_slug
    ]
    disjoint_paths = q255_contract["target-disjoint-isolation-control"][
        production_stopped_slug
    ]
    assert all((tmp_path / path).is_file() for path in production_paths)
    assert not (tmp_path / disjoint_paths[0]).exists()
    assert all((tmp_path / path).is_file() for path in disjoint_paths[1:])

    disjoint_stopped_slug = f"theta-m-{arms[2][0]}_theta-a-{arms[2][1]}"
    assert all(
        (tmp_path / path).is_file()
        for regime in q255_contract.values()
        for path in regime[disjoint_stopped_slug]
    )
    assert (tmp_path / contract["selection"]["development"]).is_file()
    assert (tmp_path / contract["selection"]["final"]).is_file()
    assert len([event for event in events if event.startswith("publish:")]) == 45

    store = persistence.CalibrationV3ArtifactStore.open(
        repository_root=tmp_path,
        validated_protocol=protocol,
        expected_source_bundle_identity_sha256=authority.source_bundle_identity_sha256,
    )
    q4_stop = store.read(
        contract["compressed_reserve_q4"]["development"][ineligible_slug][0]
    )
    assert q4_stop.payload["result"]["status"] == "STOP"
    production_stop = store.read(production_paths[0])
    assert production_stop.payload["result"]["status"] == "STOP"
    disjoint_stop = store.read(
        q255_contract["target-disjoint-isolation-control"][disjoint_stopped_slug][0]
    )
    assert disjoint_stop.payload["result"]["status"] == "STOP"
    selection = store.read(contract["selection"]["development"])
    assert selection.payload["result"]["selected_arm_slug"] == selected_slug
    final = store.read(contract["selection"]["final"])
    assert final.payload["result"] == {
        "first_failed_gate": None,
        "selected_arm_slug": selected_slug,
        "verdict": "PASS_R3_CALIBRATION_V3",
    }


def test_no_q255_plus_r3_pass_leaves_development_selection_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path)
    protocol = authority.protocol
    arms = tuple(tuple(row) for row in protocol["threshold_selection"]["candidate_arms"])
    events: list[str] = []
    _install_native_publication(monkeypatch, events)
    store, native_module = runner._open_development_artifact_store(authority)

    q255_rows: list[dict[str, object]] = []
    r3_rows: list[dict[str, object]] = []
    for index, arm in enumerate(arms):
        arm_slug = f"theta-m-{arm[0]}_theta-a-{arm[1]}"
        q255_rows.append(
            {
                "arm_slug": arm_slug,
                "status": (
                    "PASS_ALL_DEVELOPMENT_SEEDS"
                    if index == 0
                    else "STOP_DEVELOPMENT_SEED_SET"
                ),
                "theta_active": arm[1],
                "theta_match": arm[0],
            }
        )
        r3_rows.append(
            {
                "arm_slug": arm_slug,
                "status": "STOP" if index == 0 else "PASS",
                "theta_active": arm[1],
                "theta_match": arm[0],
                "verdict": (
                    "STOP_R3_F0_TRAIN_STRUCTURE"
                    if index == 0
                    else "PASS_R3_F0_TRAIN_STRUCTURE"
                ),
            }
        )

    selection = runner._publish_development_selection(
        authority,
        store=store,
        native=native_module,
        q255={"arms": q255_rows},
        r3_f0={"arms": r3_rows},
    )

    assert selection["status"] == "STOP_NO_PASSING_DEVELOPMENT_CANDIDATE"
    assert selection["verdicts"] == {
        f"theta-m-{arm[0]}_theta-a-{arm[1]}": False for arm in arms
    }
    selection_path = protocol["artifact_contract"]["selection"]["development"]
    assert not (tmp_path / selection_path).exists()
    assert not any(event.startswith("publish:") for event in events)


@pytest.mark.parametrize(
    ("subdirectory", "selected", "gate"),
    (
        ("pre-selection", None, "development_selection"),
        ("post-selection", (18, 20), "independent_confirmation"),
    ),
)
def test_final_reject_codec_preserves_pre_and_post_selection_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subdirectory: str,
    selected: tuple[int, int] | None,
    gate: str,
) -> None:
    repository_root = tmp_path / subdirectory
    repository_root.mkdir()
    authority = _authority(repository_root)
    events: list[str] = []
    _install_native_publication(monkeypatch, events)
    store, native_module = runner._open_development_artifact_store(authority)

    receipt = runner._publish_final_verdict(
        authority,
        store=store,
        native=native_module,
        selected_arm=selected,
        first_failed_gate=gate,
        verdict="REJECT",
    )

    final = store.read(authority.protocol["artifact_contract"]["selection"]["final"])
    selected_slug = (
        None
        if selected is None
        else f"theta-m-{selected[0]}_theta-a-{selected[1]}"
    )
    assert receipt["verdict"] == "REJECT_R3_CALIBRATION_V3"
    assert final.payload["result"] == {
        "first_failed_gate": gate,
        "selected_arm_slug": selected_slug,
        "verdict": "REJECT_R3_CALIBRATION_V3",
    }
    assert final.payload["thresholds"] == (
        None
        if selected is None
        else {"theta_active": selected[1], "theta_match": selected[0]}
    )
