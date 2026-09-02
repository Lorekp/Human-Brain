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

import experiments.r4_integrated.continual_toy_swe_tasks_v10 as v10
import pytest
from experiments.r4_integrated.continual_toy_swe_tasks_v10 import (
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
    CandidatePatchRejectedV10,
    ContinualToySweTaskV10Error,
    LearnerVisibleToySweObservationV10,
    global_tool_contract_v10,
    learner_workspace_files_v10,
    posthoc_score_terminal_actions_v10,
    preflight_execution_substrate_v10,
    public_train_tasks_v10,
    runner_mount_allowlist_v10,
    runner_writable_source_allowlist_v10,
    seal_action_summary_v10,
    submit_patch_v10,
    train_stream_custody_v10,
    train_stream_sha256_v10,
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

SOURCE_PATH = Path(v10.__file__)
MANIFEST_PATH = SOURCE_PATH.with_name(v10.MANIFEST_FILENAME)
WORKER_PATH = Path(v10._REPOSITORY_WORKER_PATH)
SANDBOX_PATH = Path(v10._SANDBOX_SOURCE_PATH)
EXPECTED_RAW_SOURCE_SHA256 = "b35b43d67dfc1eb329160b47185573e5230de3eaa41105c13e28af7c87590e89"
EXPECTED_RAW_MANIFEST_SHA256 = "343f7252340f9932dcc6b8996d4267bde9912cc77ccd71f2a1406bbf3a86660c"


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _workspace_rows(commitment: str) -> tuple[str, ...]:
    return tuple(
        _json({"content_utf8": content.decode("utf-8"), "path": path})
        for path, content in sorted(learner_workspace_files_v10(commitment).items())
    )


def _workspace_text(commitment: str) -> str:
    return "\n".join(_workspace_rows(commitment))


def _nonimplementation_public_evidence(ordinal: int) -> str:
    task = v10._EVALUATOR_STREAM[ordinal]
    workspace = learner_workspace_files_v10(task.public.task_commitment_sha256)
    paths = tuple(
        path
        for path in workspace
        if path == "README.md" or path.startswith("docs/") or path.startswith("tests/")
    )
    return "\n".join((task.public.issue, *(workspace[path].decode("utf-8") for path in paths)))


def _canonical_prompt(ordinal: int) -> str:
    task = v10._EVALUATOR_STREAM[ordinal]
    contract = global_tool_contract_v10()
    external = contract["external_mutating_affordances"]
    assert isinstance(external, list) and len(external) == 1
    action = external[0]
    assert isinstance(action, dict)
    affordance = ToolAffordance.seal(SUBMIT_PATCH, action["payload"])
    blocks = [ObservationBlock.seal("issue", task.public.issue)]
    blocks.extend(
        ObservationBlock.seal("workspace-file", row)
        for row in _workspace_rows(task.public.task_commitment_sha256)
    )
    return prompt_for_frame(ObservationFrame.seal(blocks, (affordance,)))


Replacement = tuple[str, str, str]


_SCAN_BODY = """    result: list[str] = []
    current = ""
    escaped = False
    for character in text:
        if escaped:
            current += character
            escaped = False
        elif character == escape:
            escaped = True
        elif character == separator:
            result.append(current)
            current = ""
        else:
            current += character
    if escaped:
        current += escape
    result.append(current)
    return result"""

_FRAGMENT_BODY = """    result: list[str] = []
    current = ""
    quoted = False
    for character in text:
        if quoted:
            current += character
            quoted = False
        elif character == quote:
            quoted = True
        elif character == divider:
            result.append(current)
            current = ""
        else:
            current += character
    if quoted:
        current += quote
    result.append(current)
    return result"""

_TREE_BODY = """    rows: list[list[object]] = []

    def visit(current: dict[str, object], path: list[list[object]]) -> None:
        kind = current["kind"]
        if kind == "scalar":
            rows.append([path, current["value"]])
        elif kind == "map":
            children = current["children"]
            for key in sorted(children):
                visit(children[key], path + [["key", key]])
        elif kind == "list":
            for index, child in enumerate(current["children"]):
                visit(child, path + [["index", index]])

    visit(node, [])
    return rows"""

_SAMPLE_TREE_BODY = """    rows: list[list[object]] = []

    def visit(current: dict[str, object], path: list[list[object]]) -> None:
        kind = current["type"]
        if kind == "reading":
            rows.append([path, current["reading"]])
        elif kind == "group":
            branches = current["branches"]
            for name in sorted(branches):
                visit(branches[name], path + [["key", name]])
        elif kind == "series":
            for index, sample in enumerate(current["samples"]):
                visit(sample, path + [["index", index]])

    visit(specimen, [])
    return rows"""

_ROUND_ROBIN_BODY = """    tenants: list[str] = []
    buckets: dict[str, list[dict[str, object]]] = {}
    for job in jobs:
        tenant = job["tenant"]
        if tenant not in buckets:
            tenants.append(tenant)
            buckets[tenant] = []
        buckets[tenant].append(job)
    ordered: list[dict[str, object]] = []
    index = 0
    while True:
        emitted = False
        for tenant in tenants:
            bucket = buckets[tenant]
            if index < len(bucket):
                ordered.append(bucket[index])
                emitted = True
        if not emitted:
            return ordered
        index += 1"""

INITIAL_REPLACEMENTS: dict[int, tuple[Replacement, ...]] = {
    0: (
        (
            "emberjournal/replay.py",
            'state[entry["key"]] = entry["value"]',
            'staged[entry["key"]] = entry["value"]',
        ),
        ("emberjournal/counts.py", "return len(entries)", "return count"),
    ),
    1: (
        (
            "cairnroute/traverse.py",
            """        if node != target:
            ordered.append(node)
        for dependency in sorted(graph.get(node, [])):
            visit(dependency)""",
            """        for dependency in sorted(graph.get(node, [])):
            visit(dependency)
        if node != target:
            ordered.append(node)""",
        ),
    ),
    2: (
        (
            "mosaicoverlay/merge.py",
            """    for key, value in sparse.items():
        if value is not None:
            merged[key] = value""",
            """    for key, value in sparse.items():
        merged[key] = value""",
        ),
    ),
    3: (("glyphfields/scan.py", "    return text.split(separator)", _SCAN_BODY),),
    5: (
        (
            "pressbatch/publish.py",
            'published[record["slug"]] = record["body"]',
            'pending[record["slug"]] = record["body"]',
        ),
        ("pressbatch/accepted.py", "return len(records)", "return accepted"),
    ),
    6: (
        (
            "nestedscalar/walk.py",
            """    if node["kind"] == "scalar":
        return [[[], node["value"]]]
    return []""",
            _TREE_BODY,
        ),
    ),
    7: (("turnscheduler/dispatch.py", "    return list(reversed(jobs))", "    return list(jobs)"),),
    8: (
        (
            "labweave/sequence.py",
            """        if protocol != assay:
            plan.append(protocol)
        for prerequisite in sorted(links.get(protocol, [])):
            collect(prerequisite)""",
            """        for prerequisite in sorted(links.get(protocol, [])):
            collect(prerequisite)
        if protocol != assay:
            plan.append(protocol)""",
        ),
    ),
    9: (("commandfragments/fragments.py", "    return text.split(divider)", _FRAGMENT_BODY),),
    11: (
        (
            "samplecanopy/measure.py",
            """    if specimen["type"] == "reading":
        return [[[], specimen["reading"]]]
    return []""",
            _SAMPLE_TREE_BODY,
        ),
    ),
}

REVISED_REPLACEMENTS: dict[int, tuple[Replacement, ...]] = {
    2: (
        (
            "mosaicoverlay/merge.py",
            """    for key, value in sparse.items():
        if value is not None:
            merged[key] = value""",
            """    for key, value in sparse.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value""",
        ),
    ),
    7: (("turnscheduler/dispatch.py", "    return list(reversed(jobs))", _ROUND_ROBIN_BODY),),
}


def _replacement_patch(ordinal: int, replacements: tuple[Replacement, ...]) -> str:
    commitment = v10._EVALUATOR_STREAM[ordinal].public.task_commitment_sha256
    workspace = learner_workspace_files_v10(commitment)
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
    assert train_stream_sha256_v10() == TRAIN_STREAM_SHA256
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest == v10._computed_manifest_record()
    assert manifest["raw_source_sha256"] == EXPECTED_RAW_SOURCE_SHA256
    assert "raw_manifest_sha256" not in manifest
    custody = train_stream_custody_v10()
    assert custody["source_prior_corpus_sha256"] == SOURCE_PRIOR_CORPUS_SHA256
    assert custody["source_prior_case_count"] == SOURCE_PRIOR_CASE_COUNT == 8
    assert custody["train_stream_custody_sha256"] == TRAIN_STREAM_CUSTODY_SHA256
    assert sha256(WORKER_PATH.read_bytes()).hexdigest() == REPOSITORY_WORKER_SHA256
    assert sha256(SANDBOX_PATH.read_bytes()).hexdigest() == SIMPLE_SWE_SANDBOX_SHA256
    executable = sandbox.resolve_python_executable()
    assert sha256(executable.read_bytes()).hexdigest() == PYTHON_EXECUTABLE_SHA256
    assert preflight_execution_substrate_v10() == {
        "repository_worker_sha256": REPOSITORY_WORKER_SHA256,
        "simple_swe_sandbox_sha256": SIMPLE_SWE_SANDBOX_SHA256,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
    }
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert not re.search(
        r"(?i)\bv8\b|\bv9\b|catalog|routing|telemetry|quay|folio|fen|swift|loam|reed|canal|cloud|ridge|prism|quilt",
        source,
    )
    assert "checks_passed" not in source
    assert "SOURCE_HASH_MASK" not in source
    ast.parse(source)


def test_public_projection_is_minimal_unique_and_evaluator_free() -> None:
    stream = public_train_tasks_v10()
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
        "transfer_relation_sha256",
        "witnesses",
    }
    assert len(stream) == TASK_COUNT == 12
    assert all(type(task) is LearnerVisibleToySweObservationV10 for task in stream)
    assert tuple(field.name for field in fields(LearnerVisibleToySweObservationV10)) == (
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
    for public, task in zip(stream, v10._EVALUATOR_STREAM, strict=True):
        assert set(public.record()).isdisjoint(forbidden)
        assert public.tool_contract_sha256 == TOOL_CONTRACT_SHA256
        public.authenticate()
        workspace_text = _workspace_text(public.task_commitment_sha256)
        assert all(
            "private" not in path.lower()
            for path in learner_workspace_files_v10(public.task_commitment_sha256)
        )
        for witness in task.private.witnesses:
            assert witness.case_id not in serialized
            assert witness.case_id not in workspace_text
            assert witness.request_json.decode("ascii") not in serialized
            assert witness.request_json.decode("ascii") not in workspace_text


def test_exact_schedule_anchors_and_history_required_public_evidence() -> None:
    stream = v10._EVALUATOR_STREAM
    assert Counter(task.private.stage for task in stream) == {
        "direct-repository-relation": 4,
        "history-required-transfer": 4,
        "consequence-guided-revision": 2,
        "genuinely-ambiguous-stop": 2,
    }
    assert [task.private.stage for task in stream] == [
        "direct-repository-relation",
        "direct-repository-relation",
        "consequence-guided-revision",
        "direct-repository-relation",
        "genuinely-ambiguous-stop",
        "history-required-transfer",
        "direct-repository-relation",
        "consequence-guided-revision",
        "history-required-transfer",
        "history-required-transfer",
        "genuinely-ambiguous-stop",
        "history-required-transfer",
    ]
    anchors = {
        task.ordinal: task.private.retention_anchor_ordinal
        for task in stream
        if task.private.retention_anchor_ordinal is not None
    }
    assert anchors == {5: 0, 8: 1, 9: 3, 11: 6}
    for ordinal, anchor in anchors.items():
        assert (
            stream[ordinal].private.transfer_relation_sha256
            == stream[anchor].private.transfer_relation_sha256
        )

    transfer_evidence = {
        5: (_nonimplementation_public_evidence(5), "earlier Ember mutation-journal"),
        8: (_nonimplementation_public_evidence(8), "earlier Cairn prerequisite-route"),
        9: (_nonimplementation_public_evidence(9), "earlier Glyph field-scanner"),
        11: (_nonimplementation_public_evidence(11), "earlier Nested scalar-path"),
    }
    for evidence, cue in transfer_evidence.values():
        assert cue in evidence
    assert not re.search(
        r"(?i)provisional|until committed|abort discards|write to pending", transfer_evidence[5][0]
    )
    assert not re.search(
        r"(?i)postorder|before each consumer|sorted dependencies|cycle", transfer_evidence[8][0]
    )
    assert not re.search(
        r"(?i)escape consumes|terminal escape|empty fields|odd|even|parity", transfer_evidence[9][0]
    )
    assert not re.search(
        r"(?i)sort map|list index order|recursive|\[key,|\[index,", transfer_evidence[11][0]
    )
    assert "publication_result([])" in transfer_evidence[5][0]
    assert '"assay": ["wash"]' in transfer_evidence[8][0]
    assert 'fragments("north:south", ":", "~")' in transfer_evidence[9][0]
    assert '"type": "reading"' in transfer_evidence[11][0]


def test_repository_pressure_and_one_generic_patch_affordance() -> None:
    contract = global_tool_contract_v10()
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
    assert set(inspect.signature(submit_patch_v10).parameters) == {
        "task_commitment_sha256",
        "payload",
        "prior_receipt",
    }
    file_counts = set()
    for task in v10._EVALUATOR_STREAM:
        commitment = task.public.task_commitment_sha256
        readable = runner_mount_allowlist_v10(commitment)
        writable = runner_writable_source_allowlist_v10(commitment)
        workspace = learner_workspace_files_v10(commitment)
        assert tuple(workspace) == readable
        assert {"README.md", "pyproject.toml"} <= set(readable)
        assert len([path for path in readable if path.startswith("tests/")]) == 1
        assert len(writable) >= 4
        assert 7 <= len(readable) <= 11
        file_counts.add(len(readable))
        for path, content in workspace.items():
            if path.endswith(".py"):
                ast.parse(content.decode("utf-8"), filename=path)
    assert len(file_counts) >= 2


def test_expected_actions_are_unique_and_do_not_cross_materialize() -> None:
    actions: list[str] = []
    pointers: list[str] = []
    inventories = []
    patch_cardinalities = set()
    pointer_positions = set()
    for ordinal in sorted(INITIAL_REPLACEMENTS):
        patch = _solution_patch(ordinal)
        patch_cardinalities.add(patch.count("diff --git "))
        action = _tool_action(patch)
        inventory = _lexical_inventory(_canonical_prompt(ordinal), 4096)
        inner = _encode_action_transport(action)
        pointer = pointerize(inner, inventory)
        assert materialize(pointer, inventory) == inner
        assert _decode_action_transport(inner) == action
        referenced = [int(value) for value in re.findall(r"<@(\d+)>", pointer)]
        assert referenced
        pointer_positions.add(referenced[0])
        actions.append(action)
        pointers.append(pointer)
        inventories.append(inventory)
    assert len(actions) == len(set(actions)) == 10
    assert len(pointers) == len(set(pointers)) == 10
    assert patch_cardinalities == {1, 2}
    assert len(pointer_positions) >= 4
    expected = set(actions)
    for source_index, pointer in enumerate(pointers):
        for target_index, inventory in enumerate(inventories):
            if source_index == target_index:
                continue
            try:
                crossed = _decode_action_transport(materialize(pointer, inventory))
            except (NeuronalToolModelError, UnicodeDecodeError, ValueError):
                continue
            assert crossed not in expected


@pytest.mark.parametrize("ordinal", sorted(INITIAL_REPLACEMENTS))
def test_frozen_evaluator_executes_solution_and_review_revision(ordinal: int) -> None:
    task = v10._EVALUATOR_STREAM[ordinal]
    evaluator = task.private.evaluator
    assert evaluator is not None
    initial_patch = _solution_patch(ordinal)
    initial = submit_patch_v10(task.public.task_commitment_sha256, {"patch": initial_patch})
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
    final = submit_patch_v10(
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


def test_failure_and_evaluator_reach_are_action_and_hash_bound() -> None:
    task = v10._EVALUATOR_STREAM[1]
    ineffective = _replacement_patch(
        1,
        (("cairnroute/render.py", 'return " > ".join(nodes)', 'return (" > ").join(nodes)'),),
    )
    receipt = submit_patch_v10(task.public.task_commitment_sha256, {"patch": ineffective})
    assert receipt.conclusion == "failure"
    assert receipt.sandbox_evaluator_reached is True
    with pytest.raises(ContinualToySweTaskV10Error, match="receipt changed"):
        replace(receipt, sandbox_evaluator_reached=False).authenticate()


def test_candidate_rejection_is_typed_and_security_failures_cannot_pass() -> None:
    task = v10._EVALUATOR_STREAM[0]
    commitment = task.public.task_commitment_sha256
    with pytest.raises(CandidatePatchRejectedV10, match="unified Git diff"):
        submit_patch_v10(commitment, {"patch": "not a diff\n"})
    invalid_hunk = """diff --git a/emberjournal/replay.py b/emberjournal/replay.py
--- a/emberjournal/replay.py
+++ b/emberjournal/replay.py
@@ -99,1 +99,1 @@
-missing context
+replacement
"""
    with pytest.raises(CandidatePatchRejectedV10, match="does not apply cleanly"):
        submit_patch_v10(commitment, {"patch": invalid_hunk})

    for replacement in (
        "import os\n            os._exit(0)",
        'state[entry["key"]] = ().__class__.__base__.__subclasses__()',
    ):
        patch = _replacement_patch(
            0,
            (("emberjournal/replay.py", 'state[entry["key"]] = entry["value"]', replacement),),
        )
        receipt = submit_patch_v10(commitment, {"patch": patch})
        assert receipt.conclusion == "failure"
        assert receipt.sandbox_evaluator_reached is True


def test_stop_is_neutral_and_patch_attempt_does_not_reach_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_executor(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("ambiguous patch crossed the private evaluator")

    monkeypatch.setattr(v10, "_run_private_evaluator", forbidden_executor)
    stop_patches = {
        4: _replacement_patch(
            4,
            (
                (
                    "packetmeasure/display.py",
                    'return f"{first}:{second}"',
                    'return f"{first}" + ":" + f"{second}"',
                ),
            ),
        ),
        10: _replacement_patch(
            10,
            (
                (
                    "localclock/hours.py",
                    "return 0 <= value < 24",
                    "return value >= 0 and value < 24",
                ),
            ),
        ),
    }
    for ordinal, patch in stop_patches.items():
        task = v10._EVALUATOR_STREAM[ordinal]
        assert task.private.expected_action == STOP
        certificate = task.private.ambiguity_certificate
        assert certificate is not None
        certificate.authenticate()
        assert len(set(certificate.compatible_outcome_jsons)) == 2
        receipt = submit_patch_v10(task.public.task_commitment_sha256, {"patch": patch})
        assert receipt.conclusion == "failure"
        assert receipt.sandbox_evaluator_reached is False
        receipt.authenticate()


def test_posthoc_scoring_accepts_only_one_sealed_summary_per_attempt() -> None:
    coding = v10._EVALUATOR_STREAM[1]
    receipt = submit_patch_v10(
        coding.public.task_commitment_sha256,
        {"patch": _solution_patch(1)},
    )
    summaries = (
        seal_action_summary_v10(coding.public.task_commitment_sha256, 1, INVALID_ACTION),
        seal_action_summary_v10(coding.public.task_commitment_sha256, 2, SUBMIT_PATCH, receipt),
    )
    with pytest.raises(ContinualToySweTaskV10Error, match="arm completion"):
        posthoc_score_terminal_actions_v10(
            coding.public.task_commitment_sha256,
            summaries,
            all_arm_streams_finished=False,
        )
    assert posthoc_score_terminal_actions_v10(
        coding.public.task_commitment_sha256,
        summaries,
        all_arm_streams_finished=True,
    ) == {
        "stage": "direct-repository-relation",
        "expected_action_matches": (False, True),
        "terminal_pass": True,
        "winning_attempt": 2,
    }
    stop_task = v10._EVALUATOR_STREAM[4]
    stop = seal_action_summary_v10(stop_task.public.task_commitment_sha256, 1, STOP)
    assert (
        posthoc_score_terminal_actions_v10(
            stop_task.public.task_commitment_sha256,
            (stop,),
            all_arm_streams_finished=True,
        )["terminal_pass"]
        is True
    )


def test_patch_scope_and_deep_immutability_reject_tampering() -> None:
    task = v10._EVALUATOR_STREAM[0]
    commitment = task.public.task_commitment_sha256
    test_path = next(
        path for path in runner_mount_allowlist_v10(commitment) if path.startswith("tests/")
    )
    forbidden = _replacement_patch(
        0,
        (
            (
                test_path,
                "def test_aborted_write_is_not_materialized()",
                "def test_aborted_write_remains_unmaterialized()",
            ),
        ),
    )
    with pytest.raises(ContinualToySweTaskV10Error, match="writable allowlist"):
        submit_patch_v10(commitment, {"patch": forbidden})
    public = public_train_tasks_v10()[0]
    with pytest.raises(FrozenInstanceError):
        public.issue = "changed"  # type: ignore[misc]
    with pytest.raises(ContinualToySweTaskV10Error, match="observation changed"):
        replace(public, issue=public.issue + " changed").authenticate()
    workspace = learner_workspace_files_v10(commitment)
    original = workspace["README.md"]
    workspace["README.md"] = b"changed"
    assert learner_workspace_files_v10(commitment)["README.md"] == original
