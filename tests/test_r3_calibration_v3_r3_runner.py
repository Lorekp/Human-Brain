from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.r3_pilot import calibration_v3_authority as authority_contract
from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from experiments.r3_pilot import run_calibration_v3_fixtures as runner

ARM = (16, 18)
ARM_SLUG = "theta-m-16_theta-a-18"
RESULT_PATH = (
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/development/"
    "theta-m-16_theta-a-18/construction-seed-0000000017.json"
)


class _TransactionStop(RuntimeError):
    pass


class _Chain:
    def __init__(self, count: int, digest: str) -> None:
        self.count = count
        self.hexdigest = digest


class _Session:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.advance_calls: list[tuple[int, bool, object]] = []
        self.scorer_resets: list[bool] = []
        self.cap_checks = 0
        self.close_calls = 0
        self.terminal_calls = 0
        self.progress = SimpleNamespace(
            step_chain=_Chain(16_387, "7" * 64),
            winner_chain=_Chain(16_385, "8" * 64),
        )

    def advance_block(
        self,
        token_ids: object,
        *,
        mode: object,
        reset_before: bool,
    ) -> tuple[object, ...]:
        values = tuple(int(value) for value in token_ids)
        self.advance_calls.append((len(values), reset_before, mode))
        return tuple(SimpleNamespace(token_id=value) for value in values)

    def check_cap(self) -> str:
        self.cap_checks += 1
        return "WITHIN_CAP"

    def terminal_identity(self) -> tuple[tuple[str, object], ...]:
        self.terminal_calls += 1
        return (
            ("body_sha256", "1" * 64),
            ("learning_step", 16_387),
            ("segment_count", 11),
        )

    def close(self) -> None:
        self.close_calls += 1


class _Coordinator:
    def __init__(self, events: list[str], session: _Session) -> None:
        self.events = events
        self.session = session
        self.committed_result: dict[str, object] | None = None
        self.start_path: str | None = None

    def start_fresh(self, path: str, factory: object) -> _Session:
        self.events.append("start_fresh")
        self.start_path = path
        factory()
        self.events.append("definition_returned")
        return self.session

    def commit_r3_result(
        self,
        session: _Session,
        result: dict[str, object],
    ) -> object:
        assert session is self.session
        self.events.append("commit")
        self.committed_result = result
        return SimpleNamespace(artifact_path=RESULT_PATH, payload_sha256="9" * 64)


def _authority() -> authority_contract.FrozenPlanAuthority:
    protocol = {
        "artifact_contract": {
            "r3_f0_train": {"development": {ARM_SLUG: RESULT_PATH}},
        },
        "source_binding": {"runnable": True},
        "threshold_selection": {"candidate_arms": [list(ARM)]},
    }
    return authority_contract.FrozenPlanAuthority(
        repository_root=Path.cwd(),
        protocol=protocol,
        protocol_sha256=hashlib.sha256(
            protocol_authority.canonical_json_bytes(protocol)
        ).hexdigest(),
        source_bundle_identity_sha256="a" * 64,
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop_on_first_step: bool,
) -> tuple[list[str], _Session, _Coordinator, type]:
    events: list[str] = []
    session = _Session(events)
    coordinator = _Coordinator(events, session)

    class Transaction:
        @classmethod
        def from_authority(cls, authority: object) -> _Coordinator:
            assert authority is not None
            events.append("transaction")
            return coordinator

    transaction_module = SimpleNamespace(
        CalibrationV3Transaction=Transaction,
        CalibrationV3TransactionStop=_TransactionStop,
    )
    definition = SimpleNamespace(
        config=SimpleNamespace(name="config"),
        token_codes=((0, tuple(range(11))),),
    )

    def build_definition(protocol: object, arm: tuple[int, int], seed: int) -> object:
        assert protocol is not None
        assert arm == ARM
        assert seed == 17
        events.append("definition")
        return definition

    first = np.arange(16_385, dtype="<i4")
    second = np.array([3, 4], dtype="<i4")
    books = (
        SimpleNamespace(
            ordinal=0,
            token_count=int(first.size),
            token_bytes_le_i32=first.tobytes(),
        ),
        SimpleNamespace(
            ordinal=1,
            token_count=int(second.size),
            token_bytes_le_i32=second.tobytes(),
        ),
    )
    verified = SimpleNamespace(
        iter_books=lambda: iter(books),
        token_count=16_387,
        tokenizer_sha256="b" * 64,
        train_input_authority_sha256="c" * 64,
        train_stream_sha256="d" * 64,
    )

    class TrainInputError(RuntimeError):
        pass

    def load_train() -> object:
        events.append("train_load")
        return verified

    train_module = SimpleNamespace(
        TrainInputError=TrainInputError,
        load_verified_train_input=load_train,
    )

    class Scorer:
        def __init__(self, *, config: object, token_columns: object) -> None:
            assert config is definition.config
            assert token_columns is definition.token_codes
            events.append("scorer")
            self.calls = 0

        def observe(self, reset_before: bool, frame: object) -> object | None:
            session.scorer_resets.append(reset_before)
            self.calls += 1
            if stop_on_first_step and self.calls == 1:
                return {"status": "STOP", "stop_kind": "SYNTHETIC_STOP"}
            return None

        def finalize(self, identity: object) -> dict[str, object]:
            events.append("finalize")
            assert identity == session.terminal_identity()
            return {
                "status": "PASS_R3_F0_TRAIN_STRUCTURE",
                "calls": self.calls,
                "winner_stream_sha256": "e" * 64,
            }

    monkeypatch.setattr(runner, "_transaction_module", lambda: transaction_module)
    monkeypatch.setattr(
        runner,
        "_definition_module",
        lambda: SimpleNamespace(build_r3_f0_definition=build_definition),
    )
    monkeypatch.setattr(runner, "_train_input_module", lambda: train_module)
    monkeypatch.setattr(
        runner,
        "_train_scoring_module",
        lambda: SimpleNamespace(R3F0TrainScorer=Scorer),
    )
    monkeypatch.setattr(
        runner,
        "_native_temporal_module",
        lambda: SimpleNamespace(TemporalStepMode=SimpleNamespace(LEARN="LEARN")),
    )
    return events, session, coordinator, Scorer


def test_r3_development_starts_clocked_factory_before_train_and_chunks_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, session, coordinator, _scorer = _install_fakes(
        monkeypatch,
        stop_on_first_step=False,
    )

    result = runner._run_source_bound_r3_f0_development(_authority(), arm=ARM)

    assert events[:5] == [
        "transaction",
        "start_fresh",
        "definition",
        "definition_returned",
        "scorer",
    ]
    assert events.index("train_load") > events.index("definition_returned")
    assert session.advance_calls == [
        (16_384, True, "LEARN"),
        (1, False, "LEARN"),
        (2, True, "LEARN"),
    ]
    assert session.cap_checks == 4
    assert [index for index, flag in enumerate(session.scorer_resets) if flag] == [0, 16_385]
    assert session.terminal_calls == 2  # runner snapshot plus fake-scorer assertion
    assert session.close_calls == 0
    assert coordinator.start_path == RESULT_PATH
    assert coordinator.committed_result is not None
    committed = coordinator.committed_result
    assert set(committed) == {
        "arm_slug",
        "comparisons",
        "kind",
        "reference",
        "resumes",
        "structural_metrics",
        "train_input",
        "verdict",
    }
    assert "active_wall_timing" not in committed["structural_metrics"]
    assert committed["train_input"] == {
        "book_count": 2,
        "held_out_access": "forbidden",
        "token_count": 16_387,
        "tokenizer_sha256": "b" * 64,
        "train_input_authority_sha256": "c" * 64,
        "train_stream_sha256": "d" * 64,
    }
    assert result == {
        "arm_slug": ARM_SLUG,
        "artifact_path": RESULT_PATH,
        "construction_seed": 17,
        "payload_sha256": "9" * 64,
        "status": "PASS",
        "verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
    }


def test_preterminal_scientific_stop_closes_without_identity_or_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _events, session, coordinator, _scorer = _install_fakes(
        monkeypatch,
        stop_on_first_step=True,
    )

    with pytest.raises(runner.FixtureRunnerStop) as stopped:
        runner._run_source_bound_r3_f0_development(_authority(), arm=ARM)

    assert stopped.value.result == {
        "status": "STOP",
        "stop_kind": "SYNTHETIC_STOP",
    }
    assert session.close_calls == 1
    assert session.terminal_calls == 0
    assert coordinator.committed_result is None


def test_r3_grid_uses_exact_q255_pass_status_in_frozen_arm_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arms = ((10, 13), (16, 18), (18, 20))
    protocol = {
        "source_binding": {"runnable": True},
        "threshold_selection": {"candidate_arms": [list(arm) for arm in arms]},
    }
    authority = authority_contract.FrozenPlanAuthority(
        repository_root=Path.cwd(),
        protocol=protocol,
        protocol_sha256=hashlib.sha256(
            protocol_authority.canonical_json_bytes(protocol)
        ).hexdigest(),
        source_bundle_identity_sha256="a" * 64,
    )
    q255_rows = [
        {
            "arm_slug": f"theta-m-{arm[0]}_theta-a-{arm[1]}",
            "status": status,
            "theta_active": arm[1],
            "theta_match": arm[0],
        }
        for arm, status in zip(
            arms,
            (
                "PASS_ALL_DEVELOPMENT_SEEDS",
                "STOP_DEVELOPMENT_SEED_SET",
                "PASS_ALL_DEVELOPMENT_SEEDS",
            ),
            strict=True,
        )
    ]
    calls: list[tuple[int, int]] = []

    def run(_authority: object, *, arm: tuple[int, int]) -> dict[str, object]:
        calls.append(arm)
        return {
            "arm_slug": f"theta-m-{arm[0]}_theta-a-{arm[1]}",
            "artifact_path": f"{arm[0]}-{arm[1]}.json",
            "construction_seed": 17,
            "payload_sha256": "9" * 64,
            "status": "PASS",
            "verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
        }

    monkeypatch.setattr(runner, "_run_source_bound_r3_f0_development", run)

    result = runner._run_development_r3_f0_screens(
        authority,
        q255={"arms": q255_rows},
    )

    assert calls == [(10, 13), (18, 20)]
    assert result["candidate_order"] == [list(arm) for arm in arms]
    assert result["eligible_arm_order"] == [[10, 13], [18, 20]]
    assert result["arms"][1]["status"] == "NOT_RUN_Q255_DEVELOPMENT_STOP"
    assert result["status"] == "PASS_ALL_Q255_ELIGIBLE_CANDIDATES"
