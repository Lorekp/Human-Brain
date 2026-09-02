from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

import pytest
from experiments.r4_integrated import continual_coding_session_v10 as session
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
        self.failure_consequences: list[str] = []
        self.last_trace = None
        self.last_update = None
        self.last_resolution_used_generic_feedback = False

    def infer(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> ModelEmission:
        self.prompts.append(prompt)
        self.budgets.append(max_new_tokens)
        self.last_resolution_used_generic_feedback = False
        output = self.outputs.pop(0)
        return ModelEmission.seal(
            prompt,
            self.training_digest,
            output,
            selected_stop=False,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )

    def _resolve(
        self,
        query_sha256: str,
        polarity: str,
        consequence_text: str | None = None,
    ) -> None:
        prior = self.training_digest
        if self.plasticity and polarity != "neutral":
            self.training_digest = sha256(
                f"{self.training_digest}:{query_sha256}:{polarity}".encode()
            ).hexdigest()
        self.last_update = SimpleNamespace(
            consequence_sha256=(
                None if consequence_text is None else sha256(consequence_text.encode()).hexdigest()
            ),
            polarity=polarity,
            prior_training_digest=prior,
            provenance_sha256=sha256(f"update:{query_sha256}:{polarity}".encode()).hexdigest(),
            query_sha256=query_sha256,
            training_digest=self.training_digest,
        )

    def resolve_success(
        self,
        query_sha256: str,
        consequence_text: str,
        strength: float,
    ) -> None:
        assert consequence_text
        assert strength == 1.0
        self._resolve(query_sha256, "success", consequence_text)

    def resolve_failure(self, query_sha256: str, consequence_text: str) -> None:
        assert consequence_text
        if session._is_generation_failure_consequence(consequence_text):
            consequence_text = session.GENERATION_FAILURE_FEEDBACK
            self.last_resolution_used_generic_feedback = True
        self.failure_consequences.append(consequence_text)
        self._resolve(query_sha256, "failure", consequence_text)

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
    sandbox_evaluator_reached: bool
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
            "sandbox_evaluator_reached": self.sandbox_evaluator_reached,
            "schema": "fake-action-receipt-v10",
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
            sandbox_evaluator_reached=True,
            receipt_sha256=sha256(f"receipt:{sequence}:{action_sha}".encode()).hexdigest(),
        )


class _FakeFixtureBaseError(Exception):
    pass


class _FakeCandidateRejected(_FakeFixtureBaseError):
    pass


class _FakeGenerationError(ValueError):
    pass


class _FakeTypedFaultModel:
    def __init__(self, *, plasticity: bool) -> None:
        self.plasticity = plasticity
        self.training_digest = "typed-fault-initial"
        self.last_trace = None
        self.last_update = None
        self.pending_query: str | None = None
        self.consequences: list[str] = []
        self.prompts: list[str] = []

    def infer(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> ModelEmission:
        if self.pending_query is not None:
            raise RuntimeError("inner emission remains unresolved")
        self.prompts.append(prompt)
        emission = ModelEmission.seal(
            prompt,
            self.training_digest,
            "unmaterializable-neuronal-output",
            selected_stop=False,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )
        self.pending_query = emission.query_sha256
        self.last_update = None
        self.last_trace = SimpleNamespace(
            inventory_cardinality=17,
            inventory_sha256="i" * 64,
            materialization_error="invalid_pointer_output",
            materialized_output_sha256=None,
            pointer_output_sha256="p" * 64,
            query_sha256=emission.query_sha256,
            selected_stop=False,
            sha256=sha256(f"trace:{emission.query_sha256}".encode()).hexdigest(),
        )
        raise _FakeGenerationError("candidate pointer detail")

    def resolve_failure(self, query_sha256: str, consequence_text: str) -> None:
        self._resolve(query_sha256, "failure", consequence_text)

    def resolve_neutral(self, query_sha256: str) -> None:
        self._resolve(query_sha256, "neutral", None)

    def resolve_success(
        self,
        query_sha256: str,
        consequence_text: str,
        strength: float,
    ) -> None:
        raise AssertionError("typed generation fault cannot resolve as success")

    def _resolve(
        self,
        query_sha256: str,
        polarity: str,
        consequence_text: str | None,
    ) -> None:
        if query_sha256 != self.pending_query:
            raise RuntimeError("typed fault query was not pending")
        prior = self.training_digest
        if polarity == "failure":
            assert consequence_text == session.GENERATION_FAILURE_FEEDBACK
            self.consequences.append(consequence_text)
            if self.plasticity:
                self.training_digest = sha256(
                    f"{prior}:{query_sha256}:failure".encode()
                ).hexdigest()
        self.pending_query = None
        self.last_update = SimpleNamespace(
            consequence_sha256=(
                None if consequence_text is None else sha256(consequence_text.encode()).hexdigest()
            ),
            polarity=polarity,
            prior_training_digest=prior,
            provenance_sha256=sha256(f"typed:{query_sha256}:{polarity}".encode()).hexdigest(),
            query_sha256=query_sha256,
            training_digest=self.training_digest,
        )


class _FakeReturnedTransportFaultModel(_FakeTypedFaultModel):
    def infer(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> ModelEmission:
        if self.pending_query is not None:
            raise RuntimeError("inner emission remains unresolved")
        self.prompts.append(prompt)
        emission = ModelEmission.seal(
            prompt,
            self.training_digest,
            "returned-invalid-action-transport",
            selected_stop=False,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )
        self.pending_query = emission.query_sha256
        self.last_update = None
        self.last_trace = SimpleNamespace(
            inventory_cardinality=17,
            inventory_sha256="i" * 64,
            materialization_error="invalid_action_transport",
            materialized_output_sha256=None,
            pointer_output_sha256="p" * 64,
            query_sha256=emission.query_sha256,
            selected_stop=False,
            sha256=sha256(f"transport:{emission.query_sha256}".encode()).hexdigest(),
        )
        return emission


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


def test_frame_contains_files_public_receipt_and_only_generic_generation_feedback() -> None:
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
        generation_failure_feedback=True,
    )

    assert [block.kind for block in frame.blocks] == [
        "issue",
        "workspace-file",
        "workspace-file",
        "prior-public-receipt",
        "generation-feedback",
    ]
    file_records = [
        json.loads(block.text) for block in frame.blocks if block.kind == "workspace-file"
    ]
    assert [record["path"] for record in file_records] == ["pkg/a.py", "z.py"]
    assert json.loads(frame.blocks[-2].text) == prior
    assert frame.blocks[-1].text == session.GENERATION_FAILURE_FEEDBACK
    assert "invalid_pointer_output" not in frame.blocks[-1].text
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
    fixture = types.ModuleType("experiments.r4_integrated.continual_toy_swe_tasks_v10")
    fixture.CandidatePatchRejectedV10 = type("CandidatePatchRejectedV10", (Exception,), {})
    fixture.learner_workspace_files_v10 = lambda commitment: {}
    fixture.submit_patch_v10 = lambda commitment, payload, prior_receipt=None: None
    monkeypatch.setitem(
        sys.modules,
        "experiments.r4_integrated.continual_toy_swe_tasks_v10",
        fixture,
    )
    reached = {"growth": False, "reservation": False}

    def fail_preflight(output_path) -> object:
        raise RuntimeError("V10 execution substrate custody changed")

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
    assert "sandbox_evaluator_reached" not in model.prompts[1]
    assert "evaluator_sha256" not in model.prompts[1]
    assert model.budgets == [8192, 8192]
    assert [row["capability"]["sandbox_evaluator_reached"] for row in run.attempts] == [
        True,
        True,
    ]
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
    assert run.attempts[0]["decision"]["reason_utf8_bytes"] == len("insufficient public evidence")
    assert "reason" not in run.attempts[0]["decision"]
    assert "insufficient public evidence" not in json.dumps(run.public_record())
    assert submit.calls == []
    assert run.fixture_receipts == []


def test_malformed_generations_retry_without_becoming_false_stop_actions() -> None:
    raw_faults = ["not-an-action", "still-not-an-action", "bad"]
    model = _FakeModel(raw_faults)
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
    assert run.terminal_conclusion == "generation-failure"
    assert all(row["resolution"] == "failure" for row in run.attempts)
    assert all(row["decision"]["fault_subtype"] == "malformed_model_output" for row in run.attempts)
    assert all(row["digest_changed"] is True for row in run.attempts)
    assert all(
        row["model_update"]["polarity"] == "failure"
        and row["model_update"]["query_sha256"] == row["generation_trace"]["model_query_sha256"]
        for row in run.attempts
    )
    assert all(
        not row["capability"][key]
        for row in run.attempts
        for key in (
            "canonical_tool_action_admitted",
            "execution_adapter_contacted",
            "fixture_receipt_reached",
            "patch_validator_reached",
            "sandbox_evaluator_reached",
            "valid_action_transport",
        )
    )
    assert model.failure_consequences == [session.GENERATION_FAILURE_FEEDBACK] * 3
    assert "generation-feedback" not in model.prompts[0]
    assert all("generation-feedback" in prompt for prompt in model.prompts[1:])
    assert all(output not in json.dumps(run.public_record()) for output in raw_faults)
    assert submit.calls == []
    assert run.fixture_receipts == []


@pytest.mark.parametrize(("plasticity", "digest_changed"), [(True, True), (False, False)])
def test_authenticated_typed_generation_fault_is_an_ordinary_retryable_attempt(
    plasticity: bool,
    digest_changed: bool,
) -> None:
    inner = _FakeTypedFaultModel(plasticity=plasticity)
    model = session._GenerationFaultNormalizingModel(inner, _FakeGenerationError)
    submit = _FakeSubmit([])

    run = session._run_task(
        arm="treatment" if plasticity else "no_update",
        task=_task(),
        model=model,
        affordance=session._submit_patch_affordance(_contract()),
        max_patch_submissions=2,
        workspace_files=_workspace,
        submit_patch=submit,
        candidate_rejection=_FakeCandidateRejected,
    )

    assert len(run.attempts) == 2
    assert run.terminal_conclusion == "generation-failure"
    assert inner.consequences == [session.GENERATION_FAILURE_FEEDBACK] * 2
    assert submit.calls == []
    assert "generation-feedback" not in inner.prompts[0]
    assert "generation-feedback" in inner.prompts[1]
    assert "invalid_pointer_output" not in inner.prompts[1]
    for attempt in run.attempts:
        assert attempt["decision"]["fault_subtype"] == "invalid_pointer_output"
        assert attempt["decision"]["exception_type"] == "_FakeGenerationError"
        assert attempt["digest_changed"] is digest_changed
        assert attempt["model_update"]["polarity"] == "failure"
        assert (
            attempt["generation_trace"]["model_query_sha256"]
            == attempt["model_trace"]["query_sha256"]
            == attempt["model_update"]["query_sha256"]
        )
        assert not any(
            attempt["capability"][key]
            for key in (
                "canonical_tool_action_admitted",
                "execution_adapter_contacted",
                "fixture_receipt_reached",
                "patch_validator_reached",
                "pointer_materialized",
                "sandbox_evaluator_reached",
                "valid_action_transport",
            )
        )


def test_returned_action_transport_fault_is_not_caught_as_a_typed_exception() -> None:
    inner = _FakeReturnedTransportFaultModel(plasticity=False)
    model = session._GenerationFaultNormalizingModel(inner, _FakeGenerationError)

    run = session._run_task(
        arm="no_update",
        task=_task(),
        model=model,
        affordance=session._submit_patch_affordance(_contract()),
        max_patch_submissions=1,
        workspace_files=_workspace,
        submit_patch=_FakeSubmit([]),
        candidate_rejection=_FakeCandidateRejected,
    )

    attempt = run.attempts[0]
    assert attempt["decision"]["fault_subtype"] == "invalid_action_transport"
    assert attempt["decision"]["exception_type"] is None
    assert (
        attempt["generation_trace"]["raw_output_sha256"]
        == sha256(b"returned-invalid-action-transport").hexdigest()
    )
    assert attempt["digest_changed"] is False
    assert attempt["capability"]["execution_adapter_contacted"] is False
    assert inner.consequences == [session.GENERATION_FAILURE_FEEDBACK]


def test_fault_normalizer_is_transparent_for_an_ordinary_emission() -> None:
    inner = _FakeModel(["<stop>ordinary</stop>"], plasticity=False)
    model = session._GenerationFaultNormalizingModel(inner, _FakeGenerationError)

    emission = model.infer(
        "ordinary prompt",
        max_new_tokens=session.MAX_NEW_TOKENS,
        stop_text=session.MODEL_STOP_TEXT,
    )

    assert emission.raw_output == "<stop>ordinary</stop>"
    assert model.last_generation_fault_subtype is None
    model.resolve_neutral(emission.query_sha256)
    assert model.training_digest == "initial-model-digest"


def test_unresolved_normalized_generation_remains_fatal() -> None:
    model = session._GenerationFaultNormalizingModel(
        _FakeTypedFaultModel(plasticity=False),
        _FakeGenerationError,
    )
    emission = model.infer(
        "prompt",
        max_new_tokens=session.MAX_NEW_TOKENS,
        stop_text=session.MODEL_STOP_TEXT,
    )

    with pytest.raises(RuntimeError, match="resolve the pending"):
        model.infer(
            "prompt",
            max_new_tokens=session.MAX_NEW_TOKENS,
            stop_text=session.MODEL_STOP_TEXT,
        )
    model.resolve_neutral(emission.query_sha256)


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
    assert run.attempts[0]["capability"]["patch_validator_reached"] is True
    assert run.attempts[0]["capability"]["fixture_receipt_reached"] is False
    assert run.attempts[0]["capability"]["sandbox_evaluator_reached"] is False
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


def test_post_stream_retention_records_authenticated_generation_fault_as_a_miss() -> None:
    inner = _FakeTypedFaultModel(plasticity=True)
    model = session._GenerationFaultNormalizingModel(inner, _FakeGenerationError)
    example = SimpleNamespace(prompt="retention prompt", output="known output")

    record = session._public_prior_retention_probe(model, (example,))

    assert record["passed"] == 0
    assert record["pass_rate"] == 0.0
    assert record["checkpoint_digest"] == "typed-fault-initial"
    assert model.training_digest == "typed-fault-initial"
    assert record["probes"][0]["matched"] is False
    assert record["probes"][0]["trace"]["materialization_error"] == "invalid_pointer_output"


def _capability_row(**overrides: bool) -> dict[str, bool]:
    row = {key: False for key in session._CAPABILITY_KEYS}
    row.update(overrides)
    return row


def test_capability_curve_counts_real_boundaries_and_changed_action_recovery() -> None:
    submit = _FakeSubmit(["action-required", "success"])
    first = submit("a" * 64, {"patch": "first"})
    second = submit("a" * 64, {"patch": "second"}, first)
    revision = session._TaskRun(
        arm="treatment",
        task_commitment_sha256="a" * 64,
        instance_id="revision",
        issue_sha256="b" * 64,
        attempts=[
            {
                "attempt": 1,
                "capability": _capability_row(
                    physical_non_stop=True,
                    pointer_materialized=True,
                    valid_action_transport=True,
                    canonical_tool_action_admitted=True,
                    execution_adapter_contacted=True,
                    patch_validator_reached=True,
                    fixture_receipt_reached=True,
                    sandbox_evaluator_reached=True,
                ),
                "decision": {
                    "action_kind": "submit_patch",
                    "arguments_sha256": "1" * 64,
                },
            },
            {
                "attempt": 2,
                "capability": _capability_row(
                    physical_non_stop=True,
                    pointer_materialized=True,
                    valid_action_transport=True,
                    canonical_tool_action_admitted=True,
                    execution_adapter_contacted=True,
                    patch_validator_reached=True,
                    fixture_receipt_reached=True,
                    sandbox_evaluator_reached=True,
                ),
                "decision": {
                    "action_kind": "submit_patch",
                    "arguments_sha256": "2" * 64,
                },
            },
        ],
        submitted_patches=[],
        fixture_receipts=[first, second],
        episode_receipts=[first, second],
        terminal_action="second",
        terminal_conclusion="success",
        budget_exhausted=False,
        posthoc_stage="consequence-guided-revision",
        posthoc_passed=True,
        posthoc_expected_action_matches=(True, True),
        posthoc_winning_attempt=2,
    )
    stop = session._TaskRun(
        arm="treatment",
        task_commitment_sha256="c" * 64,
        instance_id="stop",
        issue_sha256="d" * 64,
        attempts=[
            {
                "attempt": 1,
                "capability": _capability_row(),
                "decision": {"action_kind": "neutral-stop"},
            }
        ],
        submitted_patches=[],
        fixture_receipts=[],
        episode_receipts=[None],
        terminal_action="<stop>",
        terminal_conclusion="stop",
        budget_exhausted=False,
        posthoc_stage="genuinely-ambiguous-stop",
        posthoc_passed=True,
        posthoc_expected_action_matches=(True,),
        posthoc_winning_attempt=1,
    )

    counts = session._capability_counts([revision, stop])

    assert counts["physical_non_stop"] == 2
    assert counts["pointer_materialized"] == 2
    assert counts["valid_action_transport"] == 2
    assert counts["canonical_tool_action_admitted"] == 2
    assert counts["execution_adapter_contacted"] == 2
    assert counts["patch_validator_reached"] == 2
    assert counts["fixture_receipt_reached"] == 2
    assert counts["sandbox_evaluator_reached"] == 2
    assert counts["semantic_pass"] == 1
    assert counts["retry"] == {
        "changed_action_recovery": 1,
        "repeated_action_attempts": 0,
        "repeated_action_recovery": 0,
        "successful_retry": 1,
        "tasks_retried": 1,
        "tasks_with_repeated_action": 0,
    }
    assert counts["stop_accuracy"] == {
        "accuracy": 1.0,
        "correct": 2,
        "expected_stop": 1,
        "false_negative": 0,
        "false_positive": 0,
        "selected_stop": 1,
        "tasks": 2,
    }

    revision.attempts[1]["decision"]["arguments_sha256"] = "1" * 64
    assert session._repeated_action_metrics([revision]) == {
        "changed_action_recovery": 0,
        "repeated_action_attempts": 1,
        "repeated_action_recovery": 1,
        "successful_retry": 1,
        "tasks_retried": 1,
        "tasks_with_repeated_action": 1,
    }


def test_retention_summary_records_expected_pre_post_counts_without_forcing_post() -> None:
    before = {"pass_rate": 1.0, "passed": 8, "total": 8}
    after = {"pass_rate": 0.75, "passed": 6, "total": 8}

    assert session._retention_summary(before, after) == {
        "expected_exact": 8,
        "post": {"pass_rate": 0.75, "passed": 6, "total": 8},
        "post_minus_pre": -2,
        "pre": {"pass_rate": 1.0, "passed": 8, "total": 8},
    }


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
        terminal_conclusion="generation-failure",
        budget_exhausted=True,
    )


def test_anchor_conditional_transfer_uses_only_preregistered_posthoc_ordinals() -> None:
    passes = {
        0: True,
        1: False,
        3: True,
        5: True,
        6: False,
        8: True,
        9: False,
        11: False,
    }
    runs = []
    for ordinal, stage in enumerate(session._EXPECTED_POSTHOC_STAGE_SEQUENCE):
        run = _unscored_run("treatment")
        run.posthoc_stage = stage
        run.posthoc_passed = passes.get(ordinal, False)
        runs.append(run)

    assert session._anchor_conditional_transfer(runs) == {
        "anchor_passed": 2,
        "pairs": 4,
        "rows": [
            {
                "anchor_ordinal": 0,
                "anchor_passed": True,
                "transfer_ordinal": 5,
                "transfer_passed": True,
            },
            {
                "anchor_ordinal": 1,
                "anchor_passed": False,
                "transfer_ordinal": 8,
                "transfer_passed": True,
            },
            {
                "anchor_ordinal": 3,
                "anchor_passed": True,
                "transfer_ordinal": 9,
                "transfer_passed": False,
            },
            {
                "anchor_ordinal": 6,
                "anchor_passed": False,
                "transfer_ordinal": 11,
                "transfer_passed": False,
            },
        ],
        "transfer_pass_rate_given_anchor_fail": 0.5,
        "transfer_pass_rate_given_anchor_pass": 0.5,
        "transfer_passed": 2,
    }


def test_posthoc_scoring_requires_all_arms_and_handles_no_external_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="before every arm"):
        session._posthoc_score([_unscored_run("treatment")])

    calls: list[tuple[str, tuple[object, ...], bool]] = []
    fixture = types.ModuleType("experiments.r4_integrated.continual_toy_swe_tasks_v10")
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

    fixture.seal_action_summary_v10 = seal_action

    def score(
        commitment: str,
        completed_actions: tuple[object, ...],
        *,
        all_arm_streams_finished: bool,
    ) -> dict[str, object]:
        calls.append((commitment, completed_actions, all_arm_streams_finished))
        return {
            "expected_action_matches": (False,),
            "stage": "direct-repository-relation",
            "terminal_pass": False,
            "winning_attempt": None,
        }

    fixture.posthoc_score_terminal_actions_v10 = score
    monkeypatch.setitem(
        sys.modules,
        "experiments.r4_integrated.continual_toy_swe_tasks_v10",
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
    assert all(run.posthoc_stage == "direct-repository-relation" for run in runs)


def test_exclusive_artifact_creation_refuses_an_existing_result(tmp_path) -> None:
    path = tmp_path / "v10.json"
    session._write_exclusive(path, {"status": "RUNNING"})

    with pytest.raises(FileExistsError):
        session._write_exclusive(path, {"status": "RUNNING-again"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "RUNNING"}
