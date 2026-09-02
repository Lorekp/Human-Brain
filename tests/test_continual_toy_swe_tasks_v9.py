from __future__ import annotations

import ast
import inspect
import json
import re
from collections import Counter
from dataclasses import FrozenInstanceError, fields, replace
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path

import experiments.r4_integrated.continual_toy_swe_tasks_v9 as v9
import pytest
from experiments.r4_integrated.continual_toy_swe_tasks_v9 import (
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
    CandidatePatchRejectedV9,
    ContinualToySweTaskV9Error,
    LearnerVisibleToySweObservationV9,
    global_tool_contract_v9,
    learner_workspace_files_v9,
    posthoc_score_terminal_actions_v9,
    preflight_execution_substrate_v9,
    public_train_tasks_v9,
    runner_mount_allowlist_v9,
    runner_writable_source_allowlist_v9,
    seal_action_summary_v9,
    submit_patch_v9,
    train_stream_custody_v9,
    train_stream_sha256_v9,
)
from humanbrain.neuronal_tool_model import (
    NeuronalToolModelError,
    _decode_action_transport,
    _encode_action_transport,
    _lexical_inventory,
)
from humanbrain.persistent_tool_learning import (
    ObservationBlock,
    ObservationFrame,
    ToolAffordance,
    prompt_for_frame,
)
from humanbrain.pointer_binding import materialize, pointerize
from humanbrain.simple_swe import sandbox

SOURCE_PATH = Path(v9.__file__)
MANIFEST_PATH = SOURCE_PATH.with_name(v9.MANIFEST_FILENAME)
WORKER_PATH = Path(v9._REPOSITORY_WORKER_PATH)
SANDBOX_PATH = Path(v9._SANDBOX_SOURCE_PATH)
EXPECTED_RAW_SOURCE_SHA256 = "2ff864ec3819e79dca5ef11dbd7ebe2a0b1dee2fe50bc47c65d424d3c8b5ef5d"
EXPECTED_RAW_MANIFEST_SHA256 = "b01d00a2ba26c3245ca39251a024db6764679533da15c8ab3bc00d17d8a06ddc"


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _workspace_text(commitment: str) -> str:
    return "\n".join(
        _json({"content_utf8": content.decode("utf-8"), "path": path})
        for path, content in sorted(learner_workspace_files_v9(commitment).items())
    )


def _nonimplementation_public_evidence(ordinal: int) -> str:
    task = v9._EVALUATOR_STREAM[ordinal]
    workspace = learner_workspace_files_v9(task.public.task_commitment_sha256)
    evidence_paths = tuple(
        path
        for path in workspace
        if path == "README.md" or path.startswith("docs/") or path.startswith("tests/")
    )
    return "\n".join(
        (task.public.issue, *(workspace[path].decode("utf-8") for path in evidence_paths))
    )


def _canonical_prompt(ordinal: int) -> str:
    task = v9._EVALUATOR_STREAM[ordinal]
    contract = global_tool_contract_v9()
    external = contract["external_mutating_affordances"]
    assert isinstance(external, list) and len(external) == 1
    action = external[0]
    assert isinstance(action, dict)
    affordance = ToolAffordance.seal(SUBMIT_PATCH, action["payload"])
    blocks = [ObservationBlock.seal("issue", task.public.issue)]
    blocks.extend(
        ObservationBlock.seal("workspace-file", row)
        for row in _workspace_text(task.public.task_commitment_sha256).splitlines()
    )
    return prompt_for_frame(ObservationFrame.seal(blocks, (affordance,)))


Replacement = tuple[str, str, str]


INITIAL_REPLACEMENTS: dict[int, tuple[Replacement, ...]] = {
    0: (
        (
            "quayledger/service.py",
            "return amount + voucher_credit(voucher)",
            "return amount - voucher_credit(voucher)",
        ),
    ),
    1: (
        (
            "folioleaf/caption.py",
            "cleaned = [clean(item) for item in stable_unique(labels)]",
            "cleaned = stable_unique(clean(item) for item in labels)",
        ),
    ),
    2: (
        (
            "fenwatch/bounds.py",
            "return first_day <= day < final_day",
            "return first_day <= day <= final_day",
        ),
    ),
    3: (
        (
            "swiftpick/ranking.py",
            "return max(choices, key=rank)",
            "return min(choices, key=rank)",
        ),
        (
            "swiftpick/report.py",
            "return f\"{selected['latency']}@{selected['code']}\"",
            "return f\"{selected['code']}@{selected['latency']}\"",
        ),
    ),
    4: (
        (
            "loammark/bands.py",
            'return "dry" if moisture <= 2 else "ready"',
            'return "dry" if moisture <= 3 else "ready"',
        ),
    ),
    5: (
        (
            "reedpack/batches.py",
            "    return groups",
            "    if current:\n        groups.append(current)\n    return groups",
        ),
    ),
    6: (
        (
            "canalsettle/formula.py",
            "return amount + max(rebate, 0)",
            "return amount - max(rebate, 0)",
        ),
    ),
    8: (
        (
            "ridgepermit/span.py",
            "below_ceiling = level < ceiling",
            "below_ceiling = level <= ceiling",
        ),
    ),
    9: (
        (
            "prismdesk/rule.py",
            'return "expedite" if age > 7 else "standard"',
            'return "expedite" if age >= 7 else "standard"',
        ),
    ),
    10: (
        (
            "quiltlegend/legend.py",
            "    unique = first_seen(swatches)\n"
            "    cleaned = [canonical(value) for value in unique]",
            "    cleaned = first_seen(canonical(value) for value in swatches)",
        ),
    ),
}

REVISED_REPLACEMENTS: dict[int, tuple[Replacement, ...]] = {
    4: (
        (
            "loammark/bands.py",
            'return "dry" if moisture <= 2 else "ready"',
            'return "dry" if moisture <= 4 else "ready"',
        ),
    ),
    9: (
        (
            "prismdesk/rule.py",
            'return "expedite" if age > 7 else "standard"',
            'return "expedite" if age >= 9 else "standard"',
        ),
    ),
}


def _replacement_patch(ordinal: int, replacements: tuple[Replacement, ...]) -> str:
    commitment = v9._EVALUATOR_STREAM[ordinal].public.task_commitment_sha256
    workspace = learner_workspace_files_v9(commitment)
    blocks = []
    for path, old, new in replacements:
        before = workspace[path].decode("utf-8")
        assert before.count(old) == 1
        after = before.replace(old, new)
        body = "".join(
            unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="\n",
            )
        )
        assert body.startswith(f"--- a/{path}\n+++ b/{path}\n")
        blocks.append(f"diff --git a/{path} b/{path}\n{body}")
    return "".join(blocks)


def _solution_patch(ordinal: int, *, revised: bool = False) -> str:
    replacements = REVISED_REPLACEMENTS if revised else INITIAL_REPLACEMENTS
    return _replacement_patch(ordinal, replacements[ordinal])


def _tool_action(patch: str) -> str:
    return f"<tool>{_json({'arguments': {'patch': patch}, 'name': SUBMIT_PATCH})}</tool>"


def test_raw_source_manifest_prior_and_substrate_custody_are_frozen() -> None:
    assert sha256(SOURCE_PATH.read_bytes()).hexdigest() == EXPECTED_RAW_SOURCE_SHA256
    assert sha256(MANIFEST_PATH.read_bytes()).hexdigest() == EXPECTED_RAW_MANIFEST_SHA256
    assert TASK_SOURCE_SHA256 == EXPECTED_RAW_SOURCE_SHA256
    assert train_stream_sha256_v9() == TRAIN_STREAM_SHA256
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == v9._computed_manifest_record()
    assert manifest["raw_source_sha256"] == EXPECTED_RAW_SOURCE_SHA256
    assert "raw_manifest_sha256" not in manifest

    custody = train_stream_custody_v9()
    assert custody["source_prior_corpus_sha256"] == SOURCE_PRIOR_CORPUS_SHA256
    assert custody["source_prior_case_count"] == SOURCE_PRIOR_CASE_COUNT == 8
    assert custody["train_stream_custody_sha256"] == TRAIN_STREAM_CUSTODY_SHA256
    assert sha256(WORKER_PATH.read_bytes()).hexdigest() == REPOSITORY_WORKER_SHA256
    assert sha256(SANDBOX_PATH.read_bytes()).hexdigest() == SIMPLE_SWE_SANDBOX_SHA256
    executable = sandbox.resolve_python_executable()
    assert sha256(executable.read_bytes()).hexdigest() == PYTHON_EXECUTABLE_SHA256
    assert preflight_execution_substrate_v9() == {
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }

    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert not re.search(r"(?i)\bv8\b|catalog|routing|telemetry", source)
    assert "checks_passed" not in source
    assert "SOURCE_HASH_MASK" not in source
    ast.parse(source)


def test_public_projection_is_minimal_unique_and_evaluator_free() -> None:
    stream = public_train_tasks_v9()
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
        "target_path",
        "target_span",
        "witnesses",
    }
    assert len(stream) == TASK_COUNT == 12
    assert all(type(task) is LearnerVisibleToySweObservationV9 for task in stream)
    assert tuple(field.name for field in fields(LearnerVisibleToySweObservationV9)) == (
        "task_commitment_sha256",
        "instance_id",
        "issue",
        "repository_snapshot_sha256",
        "tool_contract_sha256",
        "observation_sha256",
    )
    assert len({task.task_commitment_sha256 for task in stream}) == TASK_COUNT
    assert len({task.repository_snapshot_sha256 for task in stream}) == TASK_COUNT
    serialized = _json([task.record() for task in stream])
    for public, task in zip(stream, v9._EVALUATOR_STREAM, strict=True):
        assert set(public.record()).isdisjoint(forbidden)
        assert public.tool_contract_sha256 == TOOL_CONTRACT_SHA256
        public.authenticate()
        workspace_text = _workspace_text(public.task_commitment_sha256)
        assert all(
            "private" not in path.lower()
            for path in learner_workspace_files_v9(public.task_commitment_sha256)
        )
        for witness in task.private.witnesses:
            assert witness.case_id not in serialized
            assert witness.case_id not in workspace_text
            assert witness.request_json.decode("ascii") not in serialized
            assert witness.request_json.decode("ascii") not in workspace_text


def test_exact_mix_delayed_transfer_and_variable_repository_pressure() -> None:
    stream = v9._EVALUATOR_STREAM
    assert Counter(task.private.stage for task in stream) == {
        "direct-repository-relation": 4,
        "issue-doc-test-implementation-composition": 4,
        "consequence-guided-revision": 2,
        "genuinely-ambiguous-stop": 2,
    }
    assert [task.private.stage for task in stream] == [
        "direct-repository-relation",
        "issue-doc-test-implementation-composition",
        "direct-repository-relation",
        "issue-doc-test-implementation-composition",
        "consequence-guided-revision",
        "direct-repository-relation",
        "issue-doc-test-implementation-composition",
        "genuinely-ambiguous-stop",
        "direct-repository-relation",
        "consequence-guided-revision",
        "issue-doc-test-implementation-composition",
        "genuinely-ambiguous-stop",
    ]
    assert {
        task.ordinal: task.private.retention_anchor_ordinal
        for task in stream
        if task.private.retention_anchor_ordinal is not None
    } == {6: 0, 8: 2, 10: 1}
    assert stream[6].private.stage == "issue-doc-test-implementation-composition"
    assert stream[8].private.stage == "direct-repository-relation"
    assert stream[10].private.stage == "issue-doc-test-implementation-composition"
    for ordinal, anchor in ((6, 0), (8, 2), (10, 1)):
        assert (
            stream[ordinal].private.transfer_relation_sha256
            == stream[anchor].private.transfer_relation_sha256
        )
        current = _workspace_text(stream[ordinal].public.task_commitment_sha256)
        history = "\n".join(
            task.public.issue + "\n" + _workspace_text(task.public.task_commitment_sha256)
            for task in stream[: ordinal + 1]
        )
        anchor_issue_phrase = stream[anchor].public.issue.split(". ", maxsplit=1)[0]
        assert anchor_issue_phrase not in current
        assert anchor_issue_phrase in history

    canal_evidence = _nonimplementation_public_evidence(6)
    assert "earlier Quay voucher settlement relation" in canal_evidence
    assert "finalize(3, 10, 0, 2) == 32" in canal_evidence
    assert not re.search(r"(?i)subtract|deduct|rebate reduces|rebate is a credit", canal_evidence)

    ridge_evidence = _nonimplementation_public_evidence(8)
    assert "earlier Fen observation-window endpoint convention" in ridge_evidence
    assert not re.search(r"(?i)inclusive|include(?:s|d)? the ceiling|exact ceiling", ridge_evidence)

    quilt_evidence = _nonimplementation_public_evidence(10)
    assert "earlier Folio label identity and ordering convention" in quilt_evidence
    assert '["teal", "ochre"]' in quilt_evidence
    assert not re.search(
        r"(?i)case-?fold|lowercase|duplicate|canonicaliz|before (?:removing|deduplicating)",
        quilt_evidence,
    )

    file_counts = set()
    patch_file_counts = set()
    patch_hunk_counts = set()
    for task in stream:
        commitment = task.public.task_commitment_sha256
        readable = runner_mount_allowlist_v9(commitment)
        writable = runner_writable_source_allowlist_v9(commitment)
        workspace = learner_workspace_files_v9(commitment)
        assert tuple(workspace) == readable
        assert {"README.md", "pyproject.toml"} <= set(readable)
        assert len([path for path in readable if path.startswith("tests/")]) == 1
        assert len(writable) >= 3
        assert len(readable) >= 7
        file_counts.add(len(readable))
        for path, content in workspace.items():
            if path.endswith(".py"):
                ast.parse(content.decode("utf-8"), filename=path)
        if task.private.stage == "issue-doc-test-implementation-composition":
            assert any(path.startswith("docs/") for path in readable)
        if task.ordinal in INITIAL_REPLACEMENTS:
            patch = _solution_patch(task.ordinal)
            patch_file_counts.add(patch.count("diff --git "))
            patch_hunk_counts.add(patch.count("@@ "))
    assert len(file_counts) >= 3
    assert patch_file_counts == {1, 2}
    assert patch_hunk_counts == {1, 2}


def test_one_generic_patch_affordance_and_no_task_specific_routing() -> None:
    contract = global_tool_contract_v9()
    assert GLOBAL_ACTIONS == (SUBMIT_PATCH,)
    assert contract["terminal_neutral_choice"] == STOP == "<stop>"
    assert contract["max_patch_submissions"] == MAX_PATCH_SUBMISSIONS == 3
    assert contract["external_mutating_affordances"] == [
        {
            "name": SUBMIT_PATCH,
            "payload": {
                "additionalProperties": False,
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "type": "object",
            },
        }
    ]
    signature = inspect.signature(submit_patch_v9)
    assert set(signature.parameters) == {
        "task_commitment_sha256",
        "payload",
        "prior_receipt",
    }
    assert set(v9.__all__).isdisjoint(
        {"target_path_v9", "gold_patch_v9", "candidate_actions_v9", "private_tasks_v9"}
    )


def test_expected_actions_are_unique_pointer_outputs_and_do_not_cross_materialize() -> None:
    actions: list[str] = []
    pointers: list[str] = []
    inventories = []
    first_pointer_ordinals = set()
    for ordinal in sorted(INITIAL_REPLACEMENTS):
        action = _tool_action(_solution_patch(ordinal))
        prompt = _canonical_prompt(ordinal)
        inventory = _lexical_inventory(prompt, 4096)
        inner = _encode_action_transport(action)
        pointer = pointerize(inner, inventory)
        assert materialize(pointer, inventory) == inner
        assert _decode_action_transport(inner) == action
        referenced = [int(value) for value in re.findall(r"<@(\d+)>", pointer)]
        assert referenced
        first_pointer_ordinals.add(referenced[0])
        actions.append(action)
        pointers.append(pointer)
        inventories.append(inventory)

    assert len(actions) == len(set(actions)) == 10
    assert len(pointers) == len(set(pointers)) == 10
    assert len(first_pointer_ordinals) >= 4
    expected_actions = set(actions)
    for source_index, pointer in enumerate(pointers):
        for target_index, inventory in enumerate(inventories):
            if source_index == target_index:
                continue
            try:
                crossed = _decode_action_transport(materialize(pointer, inventory))
            except (NeuronalToolModelError, UnicodeDecodeError, ValueError):
                continue
            assert crossed not in expected_actions


@pytest.mark.parametrize("ordinal", sorted(INITIAL_REPLACEMENTS))
def test_frozen_evaluator_executes_solution_and_review_revision(ordinal: int) -> None:
    task = v9._EVALUATOR_STREAM[ordinal]
    evaluator = task.private.evaluator
    assert evaluator is not None
    initial_patch = _solution_patch(ordinal)
    initial = submit_patch_v9(task.public.task_commitment_sha256, {"patch": initial_patch})
    assert initial.action_sha256 == sha256(initial_patch.encode("utf-8")).hexdigest()
    assert initial.evaluator_sha256 == evaluator.phase_sha256(revised=False)
    assert initial.sandbox_evaluator_reached is True
    initial.authenticate()
    if ordinal not in REVISED_REPLACEMENTS:
        assert initial.conclusion == "success"
        return

    assert initial.conclusion == "action-required"
    assert task.private.review_plan is not None
    assert initial.feedback == task.private.review_plan.review_text
    revised_patch = _solution_patch(ordinal, revised=True)
    final = submit_patch_v9(
        task.public.task_commitment_sha256,
        {"patch": revised_patch},
        initial,
    )
    assert final.sequence == 2
    assert final.prior_receipt_sha256 == initial.receipt_sha256
    assert final.action_sha256 != initial.action_sha256
    assert final.sandbox_evaluator_reached is True
    assert final.conclusion == "success"
    final.authenticate()


def test_failure_and_sandbox_reached_telemetry_are_hash_bound() -> None:
    task = v9._EVALUATOR_STREAM[0]
    ineffective = _replacement_patch(
        0,
        (
            (
                "quayledger/forecast.py",
                "return units * unit_price + drift",
                "return (units * unit_price) + drift",
            ),
        ),
    )
    receipt = submit_patch_v9(task.public.task_commitment_sha256, {"patch": ineffective})
    assert receipt.conclusion == "failure"
    assert receipt.sandbox_evaluator_reached is True
    with pytest.raises(ContinualToySweTaskV9Error, match="receipt changed"):
        replace(receipt, sandbox_evaluator_reached=False).authenticate()


def test_malformed_and_nonapplying_candidate_patches_are_typed_rejections() -> None:
    commitment = v9._EVALUATOR_STREAM[0].public.task_commitment_sha256
    with pytest.raises(CandidatePatchRejectedV9, match="unified Git diff"):
        submit_patch_v9(commitment, {"patch": "not a diff\n"})
    invalid_hunk = """diff --git a/quayledger/service.py b/quayledger/service.py
--- a/quayledger/service.py
+++ b/quayledger/service.py
@@ -99,1 +99,1 @@
-missing context
+replacement
"""
    with pytest.raises(CandidatePatchRejectedV9, match="does not apply cleanly"):
        submit_patch_v9(commitment, {"patch": invalid_hunk})


@pytest.mark.parametrize(
    "replacement",
    (
        "import os\n    os._exit(0)",
        "return ().__class__.__base__.__subclasses__()",
    ),
)
def test_candidate_exit_and_reflection_cannot_false_pass(replacement: str) -> None:
    task = v9._EVALUATOR_STREAM[0]
    patch = _replacement_patch(
        0,
        (("quayledger/service.py", "return amount + voucher_credit(voucher)", replacement),),
    )
    receipt = submit_patch_v9(task.public.task_commitment_sha256, {"patch": patch})
    assert receipt.conclusion == "failure"
    assert receipt.sandbox_evaluator_reached is True


def test_stop_is_neutral_and_patch_attempt_never_reaches_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_executor(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("ambiguous patch crossed the sandbox evaluator")

    monkeypatch.setattr(v9, "_run_private_evaluator", forbidden_executor)
    stop_patches = {
        7: _replacement_patch(
            7,
            (("cloudvote/scores.py", "return max(value, 0)", "return max(0, value)"),),
        ),
        11: _replacement_patch(
            11,
            (
                (
                    "seedreserve/readiness.py",
                    "return first_ready != second_ready",
                    "return second_ready != first_ready",
                ),
            ),
        ),
    }
    for ordinal, patch in stop_patches.items():
        task = v9._EVALUATOR_STREAM[ordinal]
        assert task.private.expected_action == STOP
        assert task.private.evaluator is None
        certificate = task.private.ambiguity_certificate
        assert certificate is not None
        certificate.authenticate()
        assert len(set(certificate.compatible_outcome_jsons)) == 2
        receipt = submit_patch_v9(task.public.task_commitment_sha256, {"patch": patch})
        assert receipt.conclusion == "failure"
        assert receipt.sandbox_evaluator_reached is False
        receipt.authenticate()


def test_posthoc_scoring_accepts_only_sealed_terminal_summaries() -> None:
    coding = v9._EVALUATOR_STREAM[1]
    receipt = submit_patch_v9(
        coding.public.task_commitment_sha256,
        {"patch": _solution_patch(1)},
    )
    summaries = (
        seal_action_summary_v9(coding.public.task_commitment_sha256, 1, INVALID_ACTION),
        seal_action_summary_v9(coding.public.task_commitment_sha256, 2, SUBMIT_PATCH, receipt),
    )
    with pytest.raises(ContinualToySweTaskV9Error, match="arm completion"):
        posthoc_score_terminal_actions_v9(
            coding.public.task_commitment_sha256,
            summaries,
            all_arm_streams_finished=False,
        )
    assert posthoc_score_terminal_actions_v9(
        coding.public.task_commitment_sha256,
        summaries,
        all_arm_streams_finished=True,
    ) == {
        "stage": "issue-doc-test-implementation-composition",
        "expected_action_matches": (False, True),
        "terminal_pass": True,
        "winning_attempt": 2,
    }

    stop_task = v9._EVALUATOR_STREAM[7]
    stop_summary = seal_action_summary_v9(stop_task.public.task_commitment_sha256, 1, STOP)
    assert (
        posthoc_score_terminal_actions_v9(
            stop_task.public.task_commitment_sha256,
            (stop_summary,),
            all_arm_streams_finished=True,
        )["terminal_pass"]
        is True
    )


def test_patch_scope_and_deep_immutability_reject_tampering() -> None:
    task = v9._EVALUATOR_STREAM[0]
    commitment = task.public.task_commitment_sha256
    test_path = next(
        path for path in runner_mount_allowlist_v9(commitment) if path.startswith("tests/")
    )
    forbidden_test_patch = _replacement_patch(
        0,
        (
            (
                test_path,
                "def test_voucher_reduces_the_amount()",
                "def test_voucher_still_reduces_the_amount()",
            ),
        ),
    )
    with pytest.raises(ContinualToySweTaskV9Error, match="writable allowlist"):
        submit_patch_v9(commitment, {"patch": forbidden_test_patch})

    public = public_train_tasks_v9()[0]
    with pytest.raises(FrozenInstanceError):
        public.issue = "changed"  # type: ignore[misc]
    changed = replace(public, issue=public.issue + " changed")
    with pytest.raises(ContinualToySweTaskV9Error, match="observation changed"):
        changed.authenticate()
    workspace = learner_workspace_files_v9(commitment)
    original = workspace["README.md"]
    workspace["README.md"] = b"changed"
    assert learner_workspace_files_v9(commitment)["README.md"] == original
