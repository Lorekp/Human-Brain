from __future__ import annotations

import ast
import inspect
import json
from dataclasses import FrozenInstanceError, fields, replace
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path

import experiments.r4_integrated.continual_toy_swe_tasks_v8 as v8
import pytest
from experiments.r4_integrated.continual_toy_swe_tasks_v8 import (
    GLOBAL_ACTIONS,
    INVALID_ACTION,
    MAX_PATCH_SUBMISSIONS,
    PYTHON_EXECUTABLE_SHA256,
    REPOSITORY_WORKER_SHA256,
    SIMPLE_SWE_SANDBOX_SHA256,
    SOURCE_PRIOR_CASE_COUNT,
    SOURCE_PRIOR_CORPUS_SHA256,
    STOP,
    SUBMIT_PATCH,
    TASK_COUNT,
    TASK_SOURCE_SHA256,
    TOOL_CONTRACT_SHA256,
    TRAIN_STREAM_CUSTODY_SHA256,
    TRAIN_STREAM_SHA256,
    CandidatePatchRejectedV8,
    ContinualToySweTaskV8Error,
    LearnerVisibleToySweObservationV8,
    global_tool_contract_v8,
    learner_workspace_files_v8,
    posthoc_score_terminal_actions_v8,
    preflight_execution_substrate_v8,
    public_train_tasks_v8,
    runner_mount_allowlist_v8,
    runner_writable_source_allowlist_v8,
    seal_action_summary_v8,
    submit_patch_v8,
    train_stream_custody_v8,
    train_stream_sha256_v8,
)
from humanbrain.simple_swe import sandbox
from humanbrain.simple_swe.sandbox import FrozenWorkerExecution

SOURCE_PATH = Path(v8.__file__)
MANIFEST_PATH = SOURCE_PATH.with_name(v8.MANIFEST_FILENAME)
WORKER_PATH = Path(v8._REPOSITORY_WORKER_PATH)
SANDBOX_PATH = Path(v8._SANDBOX_SOURCE_PATH)
EXPECTED_RAW_SOURCE_SHA256 = "1ed195eb65abe58e632acc5732e0a7853de86625ae7d9c654105ea1d91dcea1d"
EXPECTED_RAW_MANIFEST_SHA256 = "413ccd5a7a0d46fa74ae7b487b4cdd2da9be4c65b4e4c21bc3f17097b524bdf2"


def _workspace_text(commitment: str) -> str:
    return "\n".join(
        content.decode("utf-8") for content in learner_workspace_files_v8(commitment).values()
    )


def _public_information(ordinal: int, *, include_history: bool) -> str:
    tasks = (
        v8._EVALUATOR_STREAM[: ordinal + 1] if include_history else (v8._EVALUATOR_STREAM[ordinal],)
    )
    sections = []
    for task in tasks:
        sections.append(task.public.issue)
        sections.append(_workspace_text(task.public.task_commitment_sha256))
    return "\n".join(sections)


def _replacement_patch(ordinal: int, path: str, old: str, new: str) -> str:
    commitment = v8._EVALUATOR_STREAM[ordinal].public.task_commitment_sha256
    before = learner_workspace_files_v8(commitment)[path].decode("utf-8")
    assert before.count(old) == 1
    after = before.replace(old, new)
    body = "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


INITIAL_REPLACEMENTS: dict[int, tuple[str, str, str]] = {
    0: ("catalog/keys.py", "return value.strip()", "return value.strip().lower()"),
    1: (
        "routing/registry.py",
        """    decoded = [decode(row) for row in rows]
    if decoded:
        return decoded[0][1]
    return fallback""",
        """    for candidate, destination in map(decode, rows):
        if candidate == route_id:
            return destination
    return fallback""",
    ),
    2: (
        "telemetry/service.py",
        "return calibrate(distance(sample, reference), gain, bias)",
        "return distance(calibrate(sample, gain, bias), reference)",
    ),
    3: (
        "catalog/keys.py",
        "return value.strip().lower()",
        """trimmed = value.strip()
    pairs = {(\"[\", \"]\"), (\"<\", \">\")}
    if len(trimmed) >= 2 and (trimmed[0], trimmed[-1]) in pairs:
        trimmed = trimmed[1:-1]
    return trimmed.strip().lower()""",
    ),
    4: (
        "routing/service.py",
        "return shared(without_boundaries(enabled), without_boundaries(healthy))",
        "return without_boundaries(shared(enabled, healthy))",
    ),
    5: (
        "telemetry/summary.py",
        "for value in calibrate(even(readings), gain, bias):",
        "for value in even(calibrate(readings, gain, bias)):",
    ),
    6: ("catalog/identity.py", "return label", "return label.strip()"),
    7: (
        "routing/service.py",
        "return REGION_ROUTE.get(region, DEFAULT_ROUTE)",
        "return clean_route(explicit) or REGION_ROUTE.get(region, DEFAULT_ROUTE)",
    ),
    8: (
        "signals/report.py",
        "return apply_profile(separation(captured, baseline), scale, offset)",
        "return separation(apply_profile(captured, scale, offset), baseline)",
    ),
    9: (
        "directory/identity.py",
        "return name.strip()",
        "return name.strip().lower()",
    ),
}

REVISED_REPLACEMENTS: dict[int, tuple[str, str, str]] = {
    6: ("catalog/identity.py", "return label", "return label.strip().lower()"),
    7: (
        "routing/service.py",
        "return REGION_ROUTE.get(region, DEFAULT_ROUTE)",
        "return REGION_ROUTE.get(region) or clean_route(explicit) or DEFAULT_ROUTE",
    ),
}


def _solution_patch(ordinal: int, *, revised: bool = False) -> str:
    replacements = REVISED_REPLACEMENTS if revised else INITIAL_REPLACEMENTS
    return _replacement_patch(ordinal, *replacements[ordinal])


def test_raw_source_external_manifest_and_source_prior_custody_are_frozen() -> None:
    assert sha256(SOURCE_PATH.read_bytes()).hexdigest() == EXPECTED_RAW_SOURCE_SHA256
    assert sha256(MANIFEST_PATH.read_bytes()).hexdigest() == EXPECTED_RAW_MANIFEST_SHA256
    assert TASK_SOURCE_SHA256 == EXPECTED_RAW_SOURCE_SHA256
    assert train_stream_sha256_v8() == TRAIN_STREAM_SHA256

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == v8._computed_manifest_record()
    assert manifest["raw_source_sha256"] == EXPECTED_RAW_SOURCE_SHA256
    assert "raw_manifest_sha256" not in manifest
    custody = train_stream_custody_v8()
    assert custody["source_prior_corpus_sha256"] == SOURCE_PRIOR_CORPUS_SHA256
    assert custody["source_prior_case_count"] == SOURCE_PRIOR_CASE_COUNT == 8
    assert SOURCE_PRIOR_CORPUS_SHA256 == (
        "81a60454b9af6b8592ea8b422c25bdd98115c101e5580c8cd571555fdf0abfb1"
    )
    assert "prompt_transform_sha256" not in custody
    assert sha256(WORKER_PATH.read_bytes()).hexdigest() == REPOSITORY_WORKER_SHA256
    assert sha256(SANDBOX_PATH.read_bytes()).hexdigest() == SIMPLE_SWE_SANDBOX_SHA256
    executable = sandbox.resolve_python_executable()
    assert sha256(executable.read_bytes()).hexdigest() == PYTHON_EXECUTABLE_SHA256
    assert custody["repository_worker_sha256"] == REPOSITORY_WORKER_SHA256
    assert custody["simple_swe_sandbox_sha256"] == SIMPLE_SWE_SANDBOX_SHA256
    assert custody["python_executable_sha256"] == PYTHON_EXECUTABLE_SHA256
    assert custody["train_stream_custody_sha256"] == TRAIN_STREAM_CUSTODY_SHA256
    assert preflight_execution_substrate_v8() == {
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }

    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "checks_passed" not in source
    assert "SOURCE_HASH_MASK" not in source
    assert "source-hash-omit" not in source
    syntax = ast.parse(source)
    imports = {
        node.module
        for node in syntax.body
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    } | {alias.name for node in syntax.body if isinstance(node, ast.Import) for alias in node.names}
    assert imports == {
        "collections",
        "dataclasses",
        "hashlib",
        "humanbrain.simple_swe.protocol",
        "humanbrain.simple_swe.sandbox",
        "json",
        "pathlib",
        "subprocess",
        "tempfile",
        "textwrap",
    }


def test_public_projection_is_minimal_unique_and_evaluator_free() -> None:
    stream = public_train_tasks_v8()
    forbidden = {
        "attempt_budget",
        "candidate_patch",
        "evaluator",
        "expected_action",
        "family_id",
        "gold_patch",
        "retention_anchor_ordinal",
        "signature",
        "stage",
        "stub",
        "target_path",
        "target_span",
        "witnesses",
    }
    assert len(stream) == TASK_COUNT == 12
    assert all(type(task) is LearnerVisibleToySweObservationV8 for task in stream)
    assert tuple(field.name for field in fields(LearnerVisibleToySweObservationV8)) == (
        "task_commitment_sha256",
        "instance_id",
        "issue",
        "repository_snapshot_sha256",
        "tool_contract_sha256",
        "observation_sha256",
    )
    assert len({task.task_commitment_sha256 for task in stream}) == TASK_COUNT
    assert len({task.repository_snapshot_sha256 for task in stream}) == TASK_COUNT
    for task in stream:
        record = task.record()
        assert set(record).isdisjoint(forbidden)
        assert set(record) == {
            "instance_id",
            "issue",
            "partition",
            "repository_snapshot_sha256",
            "schema",
            "task_commitment_sha256",
            "tool_contract_sha256",
        }
        assert task.tool_contract_sha256 == TOOL_CONTRACT_SHA256
        task.authenticate()

    serialized = json.dumps([task.record() for task in stream], sort_keys=True)
    for task in v8._EVALUATOR_STREAM:
        evaluator = task.private.evaluator
        if evaluator is not None:
            workspace_text = _workspace_text(task.public.task_commitment_sha256)
            for case in evaluator.initial_cases:
                assert case.case_id not in serialized
                assert case.case_id not in workspace_text
                assert case.request_json.decode("ascii") not in serialized
                assert case.request_json.decode("ascii") not in workspace_text


def test_one_pointer_free_patch_affordance_and_distinct_mount_allowlists() -> None:
    contract = global_tool_contract_v8()
    affordances = contract["external_mutating_affordances"]
    assert len(affordances) == 1
    assert affordances[0] == {
        "name": SUBMIT_PATCH,
        "payload": {
            "additionalProperties": False,
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
            "type": "object",
        },
    }
    assert GLOBAL_ACTIONS == (SUBMIT_PATCH,)
    assert contract["terminal_neutral_choice"] == STOP == "<stop>"
    assert contract["max_patch_submissions"] == MAX_PATCH_SUBMISSIONS == 3
    assert "runner computes sha256" in contract["patch_hashing"]
    assert "public tests as read-only" in contract["learner_readable_files"]

    for task in public_train_tasks_v8():
        commitment = task.task_commitment_sha256
        readable = runner_mount_allowlist_v8(commitment)
        writable = runner_writable_source_allowlist_v8(commitment)
        workspace = learner_workspace_files_v8(commitment)
        assert tuple(workspace) == readable
        assert {"README.md", "pyproject.toml", "tests/test_public.py"} <= set(readable)
        assert set(writable) < set(readable)
        assert all(path.endswith(".py") and not path.startswith("tests/") for path in writable)
        assert "tests/test_public.py" not in writable
        assert "pyproject.toml" not in writable
        package_modules = [path for path in writable if not path.endswith("/__init__.py")]
        assert len(package_modules) == 3
        for path, content in workspace.items():
            if path.endswith(".py"):
                ast.parse(content.decode("utf-8"), filename=path)


def test_family_schedule_and_history_dependent_retention_information() -> None:
    stream = v8._EVALUATOR_STREAM
    assert [task.private.family_id for task in stream] == [
        "catalog-normalization",
        "route-registry",
        "telemetry-calibration",
    ] * 4
    assert [task.private.stage for task in stream] == [
        *("repository-family-acquisition" for _ in range(3)),
        *("cross-module-composition" for _ in range(3)),
        *("action-bound-review-revision" for _ in range(2)),
        *("delayed-structural-transfer" for _ in range(2)),
        *("evidence-limited-stop" for _ in range(2)),
    ]
    assert {
        8: stream[8].private.retention_anchor_ordinal,
        9: stream[9].private.retention_anchor_ordinal,
    } == {
        8: 2,
        9: 0,
    }
    for ordinal, anchor in ((8, 2), (9, 0)):
        assert (
            stream[ordinal].private.transfer_relation_sha256
            == stream[anchor].private.transfer_relation_sha256
        )

    telemetry_current = _public_information(8, include_history=False)
    telemetry_history = _public_information(8, include_history=True)
    assert "Beacon Transform Q4" in telemetry_current
    assert "profiles the captured sample first" not in telemetry_current
    assert "profiles the captured sample first" in telemetry_history

    catalog_current = _public_information(9, include_history=False)
    catalog_history = _public_information(9, include_history=True)
    assert "Archive Key R17" in catalog_current
    assert "ignoring letter case" not in catalog_current
    assert "ignoring letter case" in catalog_history

    telemetry_public_test = learner_workspace_files_v8(stream[8].public.task_commitment_sha256)[
        "tests/test_public.py"
    ].decode("utf-8")
    catalog_public_test = learner_workspace_files_v8(stream[9].public.task_commitment_sha256)[
        "tests/test_public.py"
    ].decode("utf-8")
    assert "offset_report(2, 5, 1, 0)" in telemetry_public_test
    assert 'identifier_for(["u-7;  Ada Lovelace "], " Ada Lovelace ")' in catalog_public_test

    ordinal_nine = v8._materialized_witnesses_v8(stream[9].public.task_commitment_sha256)
    assert ordinal_nine[0]["request"]["lines"] == ["d-1;  South Hall  "]
    assert ordinal_nine[1]["request"]["lines"] == ["d-2;Atrium"]


@pytest.mark.parametrize("ordinal", range(10))
def test_frozen_evaluator_executes_solution_and_review_revision(ordinal: int) -> None:
    task = v8._EVALUATOR_STREAM[ordinal]
    evaluator = task.private.evaluator
    assert evaluator is not None
    evaluator.authenticate()
    signature = inspect.signature(submit_patch_v8)
    assert "checks_passed" not in signature.parameters
    assert "action_sha256" not in signature.parameters

    initial_patch = _solution_patch(ordinal)
    initial = submit_patch_v8(
        task.public.task_commitment_sha256,
        {"patch": initial_patch},
    )
    assert initial.action_sha256 == sha256(initial_patch.encode("utf-8")).hexdigest()
    assert initial.evaluator_sha256 == evaluator.phase_sha256(revised=False)
    initial.authenticate()

    if ordinal not in REVISED_REPLACEMENTS:
        assert initial.conclusion == "success"
        return

    assert initial.conclusion == "action-required"
    assert task.private.review_plan is not None
    assert initial.feedback == task.private.review_plan.review_text
    revised_patch = _solution_patch(ordinal, revised=True)
    final = submit_patch_v8(
        task.public.task_commitment_sha256,
        {"patch": revised_patch},
        initial,
    )
    assert final.sequence == 2
    assert final.prior_receipt_sha256 == initial.receipt_sha256
    assert final.action_sha256 != initial.action_sha256
    assert evaluator.revised_cases is not None
    assert final.evaluator_sha256 == evaluator.phase_sha256(revised=True)
    assert final.conclusion == "success"
    final.authenticate()


def test_failed_patch_is_computed_by_evaluator_and_receipt_is_hash_bound() -> None:
    task = v8._EVALUATOR_STREAM[0]
    commitment = task.public.task_commitment_sha256
    ineffective = _replacement_patch(
        0,
        "catalog/codec.py",
        "return sku, label",
        "return (sku, label)",
    )
    receipt = submit_patch_v8(commitment, {"patch": ineffective})
    assert receipt.conclusion == "failure"
    assert receipt.action_sha256 == sha256(ineffective.encode("utf-8")).hexdigest()
    with pytest.raises(ContinualToySweTaskV8Error, match="receipt changed"):
        replace(receipt, feedback="changed").authenticate()


def test_candidate_malformed_diff_and_nonapplying_hunk_are_typed_rejections() -> None:
    task = v8._EVALUATOR_STREAM[0]
    commitment = task.public.task_commitment_sha256
    with pytest.raises(CandidatePatchRejectedV8, match="unified Git diff"):
        submit_patch_v8(commitment, {"patch": "not a diff\n"})
    invalid_hunk = """diff --git a/catalog/keys.py b/catalog/keys.py
--- a/catalog/keys.py
+++ b/catalog/keys.py
@@ -99,1 +99,1 @@
-missing context
+replacement
"""
    with pytest.raises(CandidatePatchRejectedV8, match="does not apply cleanly"):
        submit_patch_v8(commitment, {"patch": invalid_hunk})


def test_candidate_exit_zero_is_not_an_authenticated_evaluator_completion() -> None:
    task = v8._EVALUATOR_STREAM[0]
    escape = _replacement_patch(
        0,
        "catalog/keys.py",
        "return value.strip()",
        "import os\n    os._exit(0)",
    )
    receipt = submit_patch_v8(task.public.task_commitment_sha256, {"patch": escape})
    assert receipt.conclusion == "failure"


@pytest.mark.parametrize(
    "replacement",
    (
        'return open("private_acceptance.json").read()',
        "import socket\n    return socket.socket()",
        "return ().__class__.__base__.__subclasses__()",
    ),
)
def test_candidate_has_no_hidden_file_or_network_authority(replacement: str) -> None:
    task = v8._EVALUATOR_STREAM[0]
    patch = _replacement_patch(
        0,
        "catalog/keys.py",
        "return value.strip()",
        replacement,
    )
    receipt = submit_patch_v8(task.public.task_commitment_sha256, {"patch": patch})
    assert receipt.conclusion == "failure"
    workspace = learner_workspace_files_v8(task.public.task_commitment_sha256)
    assert all("private" not in path.lower() for path in workspace)
    workspace_text = b"\n".join(workspace.values()).decode("utf-8")
    assert all(case.case_id not in workspace_text for case in task.private.witnesses)


def test_worker_environment_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V8_HOST_SECRET", "must-not-cross")
    monkeypatch.setenv("PYTHONPATH", "host-only")
    environment = sandbox._sanitized_environment(Path.cwd(), v8._EXECUTION_LIMITS)
    assert "V8_HOST_SECRET" not in environment
    assert "PYTHONPATH" not in environment
    assert "PATH" not in environment
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    mismatch = sandbox.execute_frozen_worker(
        b"{}",
        v8._EXECUTION_LIMITS,
        python_executable=sandbox.resolve_python_executable(),
        python_executable_sha256="0" * 64,
        worker_source=WORKER_PATH.read_bytes(),
        worker_sha256=REPOSITORY_WORKER_SHA256,
    )
    assert mismatch.sandboxed is False
    assert "Python executable differs" in mismatch.detail


def test_evaluator_requires_authenticated_full_case_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = v8._EVALUATOR_STREAM[0]
    patch = _solution_patch(0)

    def incomplete_worker(
        input_bytes: bytes, *_args: object, **_kwargs: object
    ) -> FrozenWorkerExecution:
        payload = json.loads(input_bytes)
        result = {
            "schema": v8._REPOSITORY_RESULT_SCHEMA,
            "task_commitment_sha256": payload["task_commitment_sha256"],
            "action_sha256": payload["action_sha256"],
            "evaluator_sha256": payload["evaluator_sha256"],
            "outcome": "passed",
            "completed_case_count": payload["expected_case_count"] - 1,
            "expected_case_count": payload["expected_case_count"],
            "detail": "",
        }
        return FrozenWorkerExecution(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode("ascii"),
            b"",
            0,
            False,
            0.01,
            True,
            "",
        )

    monkeypatch.setattr(v8, "execute_frozen_worker", incomplete_worker)
    receipt = submit_patch_v8(task.public.task_commitment_sha256, {"patch": patch})
    assert receipt.conclusion == "failure"


def test_stop_is_neutral_and_never_crosses_patch_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_executor(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("STOP crossed the executor")

    monkeypatch.setattr(v8, "_run_private_evaluator", forbidden_executor)
    stop_patches = {
        10: _replacement_patch(
            10,
            "routing/policies.py",
            "return list(routes)",
            "return routes[:]",
        ),
        11: _replacement_patch(
            11,
            "telemetry/profiles.py",
            "return sample * gain + bias",
            "return (sample * gain) + bias",
        ),
    }
    for task in v8._EVALUATOR_STREAM[10:12]:
        assert task.private.expected_action == STOP
        assert task.private.evaluator is None
        certificate = task.private.ambiguity_certificate
        assert certificate is not None
        certificate.authenticate()
        assert len(set(certificate.compatible_outcome_jsons)) == 2
        receipt = submit_patch_v8(
            task.public.task_commitment_sha256,
            {"patch": stop_patches[task.ordinal]},
        )
        assert receipt.conclusion == "failure"
        assert receipt.feedback == "Patch rejected by the committed acceptance protocol."

    assert all(
        receipt_name not in v8.__all__ for receipt_name in ("stop_v8", "issue_stop_receipt_v8")
    )


def test_posthoc_scoring_uses_one_sealed_summary_per_episode() -> None:
    coding = v8._EVALUATOR_STREAM[1]
    coding_receipt = submit_patch_v8(
        coding.public.task_commitment_sha256,
        {"patch": _solution_patch(1)},
    )
    coding_summaries = (
        seal_action_summary_v8(
            coding.public.task_commitment_sha256,
            1,
            INVALID_ACTION,
        ),
        seal_action_summary_v8(
            coding.public.task_commitment_sha256,
            2,
            SUBMIT_PATCH,
            coding_receipt,
        ),
    )
    with pytest.raises(ContinualToySweTaskV8Error, match="arm completion"):
        posthoc_score_terminal_actions_v8(
            coding.public.task_commitment_sha256,
            coding_summaries,
            all_arm_streams_finished=False,
        )
    assert posthoc_score_terminal_actions_v8(
        coding.public.task_commitment_sha256,
        coding_summaries,
        all_arm_streams_finished=True,
    ) == {
        "stage": "repository-family-acquisition",
        "expected_action_matches": (False, True),
        "terminal_pass": True,
        "winning_attempt": 2,
    }

    stop_task = v8._EVALUATOR_STREAM[10]
    stop_summary = seal_action_summary_v8(
        stop_task.public.task_commitment_sha256,
        1,
        STOP,
    )
    assert (
        posthoc_score_terminal_actions_v8(
            stop_task.public.task_commitment_sha256,
            (stop_summary,),
            all_arm_streams_finished=True,
        )["terminal_pass"]
        is True
    )
    assert (
        posthoc_score_terminal_actions_v8(
            stop_task.public.task_commitment_sha256,
            (),
            all_arm_streams_finished=True,
        )["expected_action_matches"]
        == ()
    )


def test_patch_scope_and_deep_immutability_reject_tampering() -> None:
    task = v8._EVALUATOR_STREAM[0]
    commitment = task.public.task_commitment_sha256
    with pytest.raises(ContinualToySweTaskV8Error, match="exactly"):
        submit_patch_v8(commitment, {"patch": "x", "path": "catalog/keys.py"})  # type: ignore[dict-item]

    forbidden_test_patch = _replacement_patch(
        0,
        "tests/test_public.py",
        "def test_padding_is_ignored()",
        "def test_padding_is_still_ignored()",
    )
    with pytest.raises(ContinualToySweTaskV8Error, match="writable allowlist"):
        submit_patch_v8(commitment, {"patch": forbidden_test_patch})

    public = public_train_tasks_v8()[0]
    with pytest.raises(FrozenInstanceError):
        public.issue = "changed"  # type: ignore[misc]
    evaluator = task.private.evaluator
    assert evaluator is not None
    with pytest.raises(TypeError):
        evaluator.initial_cases[0].request_json[0] = 0  # type: ignore[index]
    workspace = learner_workspace_files_v8(commitment)
    readme = workspace["README.md"]
    workspace["README.md"] = b"changed"
    assert learner_workspace_files_v8(commitment)["README.md"] == readme

    changed_public = replace(public, issue=public.issue + " changed")
    with pytest.raises(ContinualToySweTaskV8Error, match="observation changed"):
        changed_public.authenticate()
    with pytest.raises(ContinualToySweTaskV8Error, match="public stream order changed"):
        v8._stream_root(tuple(reversed(public_train_tasks_v8())))
