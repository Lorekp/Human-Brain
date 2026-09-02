from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

import pytest
from experiments.r4_integrated import continual_coding_session_v8 as session
from humanbrain.persistent_tool_learning import ModelEmission


def _contract() -> dict[str, object]:
    return {
        "external_mutating_affordances": [
            {
                "name": "submit_patch",
                "payload": {
                    "additionalProperties": False,
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                    "type": "object",
                },
            }
        ],
        "instructions": "public prose is not cognitive routing metadata",
    }


def _tool_output(patch: object) -> str:
    return (
        "<tool>"
        + session._json({"arguments": {"patch": patch}, "name": "submit_patch"})
        + "</tool>"
    )


class _FakeModel:
    def __init__(self, outputs: list[str], *, plasticity: bool = True) -> None:
        self.outputs = list(outputs)
        self.plasticity = plasticity
        self.training_digest = "initial-model-digest"
        self.prompts: list[str] = []
        self.budgets: list[int] = []
        self.last_trace = None
        self.last_update = None

    def infer(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> ModelEmission:
        self.prompts.append(prompt)
        self.budgets.append(max_new_tokens)
        output = self.outputs.pop(0)
        return ModelEmission.seal(
            prompt,
            self.training_digest,
            output,
            selected_stop=False,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )

    def _resolve(self, query_sha256: str, polarity: str) -> None:
        if self.plasticity and polarity != "neutral":
            self.training_digest = sha256(
                f"{self.training_digest}:{query_sha256}:{polarity}".encode()
            ).hexdigest()

    def resolve_success(
        self,
        query_sha256: str,
        consequence_text: str,
        strength: float,
    ) -> None:
        assert consequence_text
        assert strength == 1.0
        self._resolve(query_sha256, "success")

    def resolve_failure(self, query_sha256: str, consequence_text: str) -> None:
        assert consequence_text
        self._resolve(query_sha256, "failure")

    def resolve_neutral(self, query_sha256: str) -> None:
        self._resolve(query_sha256, "neutral")


@dataclass(frozen=True)
class _FakeFixtureReceipt:
    task_commitment_sha256: str
    sequence: int
    prior_receipt_sha256: str
    action: str
    action_sha256: str
    evaluator_sha256: str
    conclusion: str
    feedback: str
    receipt_sha256: str

    def authenticate(self) -> None:
        assert self.action == "submit_patch"
        assert self.conclusion in {"success", "failure", "action-required"}

    def record(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_sha256": self.action_sha256,
            "conclusion": self.conclusion,
            "evaluator_sha256": self.evaluator_sha256,
            "feedback": self.feedback,
            "prior_receipt_sha256": self.prior_receipt_sha256,
            "schema": "fake-action-receipt-v8",
            "sequence": self.sequence,
            "task_commitment_sha256": self.task_commitment_sha256,
        }


class _FakeSubmit:
    def __init__(self, conclusions: list[str]) -> None:
        self.conclusions = list(conclusions)
        self.calls: list[tuple[str, dict[str, object], object | None]] = []

    def __call__(
        self,
        task_commitment_sha256: str,
        payload: dict[str, object],
        prior_receipt: _FakeFixtureReceipt | None = None,
    ) -> _FakeFixtureReceipt:
        self.calls.append((task_commitment_sha256, payload, prior_receipt))
        sequence = 1 if prior_receipt is None else prior_receipt.sequence + 1
        prior_sha = "0" * 64 if prior_receipt is None else prior_receipt.receipt_sha256
        conclusion = self.conclusions.pop(0)
        patch = payload["patch"]
        action_sha = sha256(str(patch).encode()).hexdigest()
        return _FakeFixtureReceipt(
            task_commitment_sha256=task_commitment_sha256,
            sequence=sequence,
            prior_receipt_sha256=prior_sha,
            action="submit_patch",
            action_sha256=action_sha,
            evaluator_sha256="e" * 64,
            conclusion=conclusion,
            feedback=(
                "Reviewer requests a revision." if conclusion == "action-required" else conclusion
            ),
            receipt_sha256=sha256(f"receipt:{sequence}:{action_sha}".encode()).hexdigest(),
        )


class _FakeFixtureBaseError(Exception):
    pass


class _FakeCandidateRejected(_FakeFixtureBaseError):
    pass


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        instance_id="public-instance",
        issue="Repair behavior using the repository evidence.",
        task_commitment_sha256="a" * 64,
    )


def _workspace(_: str) -> dict[str, bytes]:
    return {
        "pkg/worker.py": b"def work(value):\n    return value\n",
        "README.md": b"Public contract.\n",
        "tests/test_public.py": b"def test_public():\n    assert True\n",
    }


def test_frame_contains_issue_every_file_and_only_a_prior_public_receipt() -> None:
    affordance = session._submit_patch_affordance(_contract())
    prior = {
        "conclusion": "action-required",
        "feedback": "Preserve the stable branch too.",
        "receipt_sha256": "b" * 64,
    }
    frame = session._frame(
        issue="Update the package behavior.",
        workspace_files={"z.py": b"Z = 1\n", "pkg/a.py": b"A = 1\n"},
        affordance=affordance,
        prior_public_receipt=prior,
    )

    assert [block.kind for block in frame.blocks] == [
        "issue",
        "workspace-file",
        "workspace-file",
        "prior-public-receipt",
    ]
    file_records = [
        json.loads(block.text) for block in frame.blocks if block.kind == "workspace-file"
    ]
    assert [record["path"] for record in file_records] == ["pkg/a.py", "z.py"]
    assert json.loads(frame.blocks[-1].text) == prior
    assert [item.name for item in frame.affordances] == ["submit_patch"]
    assert not {
        "starter-source",
        "target-path",
        "typed-signature",
        "entry-point",
        "compatibility",
    }.intersection(block.kind for block in frame.blocks)


def test_execution_substrate_preflight_requires_all_three_frozen_hashes() -> None:
    expected = {
        "python_executable_sha256": "a" * 64,
        "repository_worker_sha256": "b" * 64,
        "simple_swe_sandbox_sha256": "c" * 64,
    }

    assert session._authenticate_execution_substrate_record(expected, expected) == expected
    with pytest.raises(RuntimeError, match="substrate custody"):
        session._authenticate_execution_substrate_record(
            {**expected, "repository_worker_sha256": "d" * 64},
            expected,
        )
    with pytest.raises(RuntimeError, match="substrate custody"):
        session._authenticate_execution_substrate_record(
            {"repository_worker_sha256": "b" * 64},
            expected,
        )


def test_failed_public_preflight_precedes_growth_and_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fixture = types.ModuleType("experiments.r4_integrated.continual_toy_swe_tasks_v8")
    fixture.CandidatePatchRejectedV8 = type("CandidatePatchRejectedV8", (Exception,), {})
    fixture.learner_workspace_files_v8 = lambda commitment: {}
    fixture.submit_patch_v8 = lambda commitment, payload, prior_receipt=None: None
    monkeypatch.setitem(
        sys.modules,
        "experiments.r4_integrated.continual_toy_swe_tasks_v8",
        fixture,
    )
    reached = {"growth": False, "reservation": False}

    def fail_preflight(output_path) -> object:
        raise RuntimeError("V8 execution substrate custody changed")

    def forbidden_growth(*args: object, **kwargs: object) -> object:
        reached["growth"] = True
        raise AssertionError("model growth crossed failed preflight")

    def forbidden_reservation(*args: object, **kwargs: object) -> None:
        reached["reservation"] = True
        raise AssertionError("reservation crossed failed preflight")

    monkeypatch.setattr(session, "_load_public_setup", fail_preflight)
    monkeypatch.setattr(session, "_grow_arms", forbidden_growth)
    monkeypatch.setattr(session, "_write_exclusive", forbidden_reservation)

    with pytest.raises(RuntimeError, match="substrate custody"):
        session.run_session(tmp_path / "must-not-exist.json")
    assert reached == {"growth": False, "reservation": False}
    assert not (tmp_path / "must-not-exist.json").exists()


def test_public_prior_becomes_eight_generic_one_file_patch_examples() -> None:
    @dataclass(frozen=True)
    class Example:
        prompt: str
        output: str
        provenance_id: str

    examples = session._bootstrap_examples(
        Example,
        session._submit_patch_affordance(_contract()),
    )

    assert len(examples) == 8
    for example in examples:
        assert example.output.startswith("<tool>") and example.output.endswith("</tool>")
        action = json.loads(example.output.removeprefix("<tool>").removesuffix("</tool>"))
        assert set(action) == {"arguments", "name"}
        assert action["name"] == "submit_patch"
        assert set(action["arguments"]) == {"patch"}
        patch = action["arguments"]["patch"]
        assert patch.startswith("diff --git a/solution.py b/solution.py\n")
        assert "--- a/solution.py\n+++ b/solution.py\n" in patch
        assert '"kind":"issue"' in example.prompt
        assert '"kind":"workspace-file"' in example.prompt
        assert "typed-signature" not in example.prompt
        assert "target-path" not in example.prompt


def test_action_required_is_successful_learning_and_binds_the_revision_frame() -> None:
    first_patch = "diff --git a/pkg/worker.py b/pkg/worker.py\nfirst\n"
    second_patch = "diff --git a/pkg/worker.py b/pkg/worker.py\nsecond\n"
    model = _FakeModel([_tool_output(first_patch), _tool_output(second_patch)])
    submit = _FakeSubmit(["action-required", "success"])

    run = session._run_task(
        arm="treatment",
        task=_task(),
        model=model,
        affordance=session._submit_patch_affordance(_contract()),
        max_patch_submissions=3,
        workspace_files=_workspace,
        submit_patch=submit,
        candidate_rejection=_FakeCandidateRejected,
    )

    assert run.terminal_conclusion == "success"
    assert len(run.attempts) == 2
    assert [row["resolution"] for row in run.attempts] == ["success", "success"]
    assert len(run.fixture_receipts) == 2
    assert submit.calls[0][2] is None
    assert submit.calls[1][2] is run.fixture_receipts[0]
    assert "prior-public-receipt" not in model.prompts[0]
    assert "prior-public-receipt" in model.prompts[1]
    assert model.budgets == [8192, 8192]
    serialized = json.dumps(run.public_record(), sort_keys=True)
    assert first_patch not in serialized and second_patch not in serialized


def test_neutral_stop_never_calls_the_executor() -> None:
    model = _FakeModel(["<stop>insufficient public evidence</stop>"])
    submit = _FakeSubmit([])

    run = session._run_task(
        arm="treatment",
        task=_task(),
        model=model,
        affordance=session._submit_patch_affordance(_contract()),
        max_patch_submissions=3,
        workspace_files=_workspace,
        submit_patch=submit,
        candidate_rejection=_FakeCandidateRejected,
    )

    assert run.terminal_conclusion == "stop"
    assert len(run.attempts) == 1
    assert run.attempts[0]["resolution"] == "neutral"
    assert submit.calls == []
    assert run.fixture_receipts == []


def test_malformed_generations_retry_without_becoming_false_stop_actions() -> None:
    model = _FakeModel(["not-an-action", "still-not-an-action", "bad"])
    submit = _FakeSubmit([])

    run = session._run_task(
        arm="treatment",
        task=_task(),
        model=model,
        affordance=session._submit_patch_affordance(_contract()),
        max_patch_submissions=3,
        workspace_files=_workspace,
        submit_patch=submit,
        candidate_rejection=_FakeCandidateRejected,
    )

    assert len(run.attempts) == 3
    assert run.budget_exhausted is True
    assert run.terminal_conclusion.startswith("generation-failure:")
    assert all(row["resolution"] == "failure" for row in run.attempts)
    assert submit.calls == []
    assert run.fixture_receipts == []


def test_only_typed_candidate_rejection_becomes_negative_neuronal_evidence() -> None:
    patch = "diff --git a/pkg/worker.py b/pkg/worker.py\nmalformed\n"

    def candidate_rejected(*args: object, **kwargs: object) -> object:
        raise _FakeCandidateRejected("candidate detail must not reach the learner")

    run = session._run_task(
        arm="treatment",
        task=_task(),
        model=_FakeModel([_tool_output(patch)]),
        affordance=session._submit_patch_affordance(_contract()),
        max_patch_submissions=1,
        workspace_files=_workspace,
        submit_patch=candidate_rejected,
        candidate_rejection=_FakeCandidateRejected,
    )

    assert run.terminal_conclusion == "failure"
    assert run.attempts[0]["resolution"] == "failure"
    assert run.episode_receipts == [None]

    def substrate_failed(*args: object, **kwargs: object) -> object:
        raise _FakeFixtureBaseError("custody failure")

    with pytest.raises(_FakeFixtureBaseError, match="custody failure"):
        session._run_task(
            arm="treatment",
            task=_task(),
            model=_FakeModel([_tool_output(patch)]),
            affordance=session._submit_patch_affordance(_contract()),
            max_patch_submissions=1,
            workspace_files=_workspace,
            submit_patch=substrate_failed,
            candidate_rejection=_FakeCandidateRejected,
        )


def test_public_prior_retention_is_read_only_and_hash_only() -> None:
    examples = (
        SimpleNamespace(prompt="prompt one", output="answer one", provenance_id="one"),
        SimpleNamespace(prompt="prompt two", output="answer two", provenance_id="two"),
    )
    model = _FakeModel(["answer one", "answer two"])

    record = session._public_prior_retention_probe(model, examples)

    assert record["passed"] == 2
    assert record["pass_rate"] == 1.0
    assert model.training_digest == "initial-model-digest"
    serialized = json.dumps(record, sort_keys=True)
    assert "answer one" not in serialized and "answer two" not in serialized


def _unscored_run(arm: str) -> session._TaskRun:
    return session._TaskRun(
        arm=arm,
        task_commitment_sha256="a" * 64,
        instance_id="public-instance",
        issue_sha256="b" * 64,
        attempts=[
            {
                "attempt": 1,
                "decision": {"action_kind": "invalid-generation", "kind": "stop"},
            }
        ],
        submitted_patches=[],
        fixture_receipts=[],
        episode_receipts=[None],
        terminal_action="<generation-failure>malformed</generation-failure>",
        terminal_conclusion="generation-failure:malformed",
        budget_exhausted=True,
    )


def test_posthoc_scoring_requires_all_arms_and_handles_no_external_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="before every arm"):
        session._posthoc_score([_unscored_run("treatment")])

    calls: list[tuple[str, tuple[object, ...], bool]] = []
    fixture = types.ModuleType("experiments.r4_integrated.continual_toy_swe_tasks_v8")
    fixture.INVALID_ACTION = "invalid"
    fixture.STOP = "<stop>"
    fixture.SUBMIT_PATCH = "submit_patch"

    def seal_action(
        commitment: str,
        attempt: int,
        action: str,
        receipt: object | None = None,
    ) -> tuple[object, ...]:
        return commitment, attempt, action, receipt

    fixture.seal_action_summary_v8 = seal_action

    def score(
        commitment: str,
        completed_actions: tuple[object, ...],
        *,
        all_arm_streams_finished: bool,
    ) -> dict[str, object]:
        calls.append((commitment, completed_actions, all_arm_streams_finished))
        return {
            "expected_action_matches": (False,),
            "stage": "repository-family-acquisition",
            "terminal_pass": False,
            "winning_attempt": None,
        }

    fixture.posthoc_score_terminal_actions_v8 = score
    monkeypatch.setitem(
        sys.modules,
        "experiments.r4_integrated.continual_toy_swe_tasks_v8",
        fixture,
    )
    runs = [_unscored_run(policy.name) for policy in session._ARM_POLICIES]

    session._posthoc_score(runs)

    assert len(calls) == 3
    assert all(
        len(actions) == 1 and actions[0][2] == "invalid" and finished is True
        for _, actions, finished in calls
    )
    assert all(run.posthoc_passed is False for run in runs)
    assert all(run.posthoc_stage == "repository-family-acquisition" for run in runs)


def test_exclusive_artifact_creation_refuses_an_existing_result(tmp_path) -> None:
    path = tmp_path / "v8.json"
    session._write_exclusive(path, {"status": "RUNNING"})

    with pytest.raises(FileExistsError):
        session._write_exclusive(path, {"status": "RUNNING-again"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "RUNNING"}
