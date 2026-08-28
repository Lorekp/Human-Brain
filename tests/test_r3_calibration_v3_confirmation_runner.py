from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.r3_pilot import calibration_v3_authority as authority_contract
from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot import run_calibration_v3_fixtures as runner

ROOT = Path(__file__).resolve().parents[1]
PASS_ELECTRICAL = "PASS_RESERVE_VALID_ELECTRICAL_V3"


def _authority(protocol: dict[str, object]) -> authority_contract.FrozenPlanAuthority:
    return authority_contract.FrozenPlanAuthority(
        repository_root=ROOT,
        protocol=protocol,
        protocol_sha256=hashlib.sha256(
            protocol_authority.canonical_json_bytes(protocol)
        ).hexdigest(),
        source_bundle_identity_sha256="b" * 64,
    )


def test_independent_confirmation_runs_only_the_selected_arm_and_keeps_fail_fast_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = protocol_authority.build_protocol()
    authority = _authority(protocol)
    selected_arm = (18, 20)
    selected_slug = "theta-m-18_theta-a-20"
    seeds = tuple(protocol["threshold_selection"]["independent_confirmation_seeds"])
    published: list[str] = []
    calls: list[tuple[str, tuple[int, int], int]] = []

    monkeypatch.setattr(
        runner,
        "_publish_independent_confirmation_result",
        lambda _authority, **kwargs: (
            published.append(kwargs["artifact_path"])
            or {"artifact_path": kwargs["artifact_path"], "payload_sha256": "c" * 64}
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_source_bound_q4_confirmation_slice",
        lambda _authority, *, arm, seed: (
            calls.append(("q4", arm, seed)) or {"marker": seed}
        ),
    )

    def production(
        _authority: authority_contract.FrozenPlanAuthority,
        *,
        arm: tuple[int, int],
        seed: int,
    ) -> dict[str, int]:
        calls.append(("production", arm, seed))
        if seed == seeds[1]:
            raise runner.FixtureRunnerStop("synthetic production STOP")
        return {"marker": seed}

    monkeypatch.setattr(
        runner,
        "_run_source_bound_q255_confirmation_production_slice",
        production,
    )
    monkeypatch.setattr(
        runner,
        "_run_source_bound_q255_confirmation_disjoint_slice",
        lambda _authority, *, arm, seed: (
            calls.append(("disjoint", arm, seed)) or {"marker": seed}
        ),
    )
    replay = {
        "seeds": {
            str(seed): {"seed": seed, "verdict": PASS_ELECTRICAL}
            for seed in seeds
        }
    }

    result = runner._run_independent_confirmation(
        authority,
        store=object(),
        native=SimpleNamespace(),
        electrical_replay=replay,
        selected_arm=selected_arm,
    )

    assert result["status"] == "STOP_Q255_INDEPENDENT_CONFIRMATION"
    assert all(arm == selected_arm for _screen, arm, _seed in calls)
    assert [call for call in calls if call[0] == "q4"] == [
        ("q4", selected_arm, seed) for seed in seeds
    ]
    assert [call for call in calls if call[0] == "production"] == [
        ("production", selected_arm, seed) for seed in seeds
    ]
    assert [call for call in calls if call[0] == "disjoint"] == [
        ("disjoint", selected_arm, seed)
        for seed in seeds
        if seed != seeds[1]
    ]

    contract = protocol["artifact_contract"]
    omitted_disjoint = contract["full_cohort_q255"]["independent_confirmation"][
        "target-disjoint-isolation-control"
    ][selected_slug][1]
    assert omitted_disjoint not in published
    assert len(published) == 15
    other_arm_slugs = {
        f"theta-m-{arm[0]}_theta-a-{arm[1]}"
        for arm in protocol_authority.THRESHOLD_ARMS
        if arm != selected_arm
    }
    assert not any(slug in path for slug in other_arm_slugs for path in published)


def _install_deterministic_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop_on_first_observation: bool,
) -> tuple[authority_contract.FrozenPlanAuthority, list[str], list[int]]:
    protocol = protocol_authority.build_protocol()
    arm_slug = "theta-m-18_theta-a-20"
    root = protocol["artifact_contract"]["root"]
    milestone_events = (2, 4, 6)
    protocol["screens"]["r3_f0_train"]["determinism"]["resume_milestones"] = [
        {
            "after_transition_event": event,
            "book_name": "tiny",
            "book_ordinal_zero_based": 0,
            "next_token_index_zero_based": next_token,
        }
        for event, next_token in zip(milestone_events, (3, 5, 7), strict=True)
    ]
    protocol["artifact_contract"]["r3_f0_train"][
        "deterministic_confirmation_resumes"
    ] = [
        "/".join(
            (
                root,
                "r3-f0",
                "deterministic-confirmation",
                arm_slug,
                "resume",
                f"transition-{event:010d}.resume.json",
            )
        )
        for event in milestone_events
    ]
    authority = _authority(protocol)
    events: list[str] = []
    observed: list[int] = []

    class TransactionStop(Exception):
        pass

    class Chain:
        def __init__(self) -> None:
            self.count = 0
            self.hexdigest = "d" * 64

    class Session:
        def __init__(self, label: str) -> None:
            self.label = label
            self.progress = SimpleNamespace(winner_chain=Chain(), step_chain=Chain())

        def advance_block(
            self,
            tokens: tuple[int, ...],
            *,
            mode: object,
            reset_before: bool,
        ) -> tuple[int, ...]:
            events.append(f"advance:{self.label}:{','.join(map(str, tokens))}")
            increment = len(tokens) - int(reset_before)
            self.progress.winner_chain.count += increment
            self.progress.step_chain.count += len(tokens)
            return tokens

        def check_cap(self) -> None:
            return None

        def terminal_identity(self) -> dict[str, int]:
            events.append("terminal_identity")
            return {"segment_count": 1}

        def close(self) -> None:
            events.append(f"close:{self.label}")

    class Coordinator:
        def start_fresh(self, result_path: str, definition_factory: object) -> Session:
            events.append(f"start:{result_path}")
            definition_factory()
            return Session("fresh")

        def open_resume(self, manifest_path: str, definition: object) -> Session:
            events.append(f"open:{Path(manifest_path).name}")
            return Session(Path(manifest_path).stem)

        def commit_resume_checkpoint(self, session: Session) -> object:
            count = session.progress.winner_chain.count
            events.append(f"commit:{count}")
            return SimpleNamespace(session=session, count=count, close=lambda: None)

        def continue_after(self, commit: object) -> Session:
            events.append(f"continue:{commit.count}")
            return commit.session

        def finish_resumed_branch(self, session: Session) -> object:
            events.append(f"finish:{session.label}")
            return SimpleNamespace(label=session.label)

        def commit_r3_result(
            self,
            session: Session,
            result: dict[str, object],
            *,
            resumed_branches: tuple[object, ...],
        ) -> object:
            events.append(f"commit_result:{len(resumed_branches)}")
            return SimpleNamespace(
                artifact_path=protocol["artifact_contract"]["r3_f0_train"][
                    "deterministic_confirmation"
                ][arm_slug],
                payload_sha256="e" * 64,
            )

    class TransactionFactory:
        @staticmethod
        def from_authority(_authority: object) -> Coordinator:
            return Coordinator()

    transaction_module = SimpleNamespace(
        CalibrationV3Transaction=TransactionFactory,
        CalibrationV3TransactionStop=TransactionStop,
    )
    monkeypatch.setattr(runner, "_transaction_module", lambda: transaction_module)
    monkeypatch.setattr(
        runner,
        "_definition_module",
        lambda: SimpleNamespace(
            build_r3_f0_definition=lambda _protocol, _arm, _seed: SimpleNamespace(
                config=object(),
                token_codes=((0, (0,)),),
            )
        ),
    )

    class Scorer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def observe(self, _reset_before: bool, frame: int) -> dict[str, str] | None:
            observed.append(frame)
            if stop_on_first_observation:
                return {"status": "STOP_SYNTHETIC"}
            return None

        def finalize(self, _identity: object) -> dict[str, object]:
            return {"status": "PASS_R3_F0_TRAIN_STRUCTURE", "metric": 1}

    monkeypatch.setattr(
        runner,
        "_train_scoring_module",
        lambda: SimpleNamespace(R3F0TrainScorer=Scorer),
    )
    token_bytes = np.asarray(range(8), dtype="<i4").tobytes()
    book = SimpleNamespace(
        ordinal=0,
        short_name="tiny",
        token_bytes_le_i32=token_bytes,
        token_count=8,
    )

    class Verified:
        tokenizer_sha256 = "1" * 64
        train_input_authority_sha256 = "2" * 64
        train_stream_sha256 = "3" * 64

        @staticmethod
        def iter_books() -> object:
            return iter((book,))

    def load_train() -> Verified:
        events.append("load_train")
        return Verified()

    monkeypatch.setattr(
        runner,
        "_train_input_module",
        lambda: SimpleNamespace(
            TrainInputError=RuntimeError,
            load_verified_train_input=load_train,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_native_temporal_module",
        lambda: SimpleNamespace(TemporalStepMode=SimpleNamespace(LEARN="learn")),
    )
    return authority, events, observed


def test_deterministic_confirmation_interleaves_three_resumes_then_commits_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, events, observed = _install_deterministic_fakes(
        monkeypatch,
        stop_on_first_observation=False,
    )

    result = runner._run_deterministic_confirmation(
        authority,
        selected_arm=(18, 20),
    )

    assert result["resume_origins"] == [2, 4, 6]
    assert result["status"] == "PASS_DETERMINISTIC_CONFIRMATION"
    assert observed == list(range(8))
    for event, next_token in zip((2, 4, 6), (3, 5, 7), strict=True):
        commit = events.index(f"commit:{event}")
        opened = events.index(f"open:transition-{event:010d}.resume.json")
        resumed = events.index(
            next(
                value
                for value in events
                if value.startswith(f"advance:transition-{event:010d}.resume:")
            )
        )
        continued = events.index(f"continue:{event}")
        assert commit < opened < resumed < continued
        assert str(next_token) in events[resumed]
    assert events[-1] == "commit_result:3"


def test_deterministic_preterminal_scorer_stop_never_commits_an_r3_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, events, observed = _install_deterministic_fakes(
        monkeypatch,
        stop_on_first_observation=True,
    )

    with pytest.raises(runner.FixtureRunnerStop, match="before terminal identity"):
        runner._run_deterministic_confirmation(authority, selected_arm=(18, 20))

    assert observed == [0]
    assert "close:fresh" in events
    assert not any(event.startswith("commit_result:") for event in events)
