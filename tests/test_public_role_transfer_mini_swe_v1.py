from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import causal_microrepository_repair_loop_v2 as string_repair
from experiments.r3_pilot import causal_predicate_gated_fold_v1 as gated
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import causal_task_observation_candidate_plan_bank_v1 as plan_bank
from experiments.r3_pilot import common_microrepository_autonomous_correction_v1 as correction
from experiments.r3_pilot import common_repository_action_memory_v2 as memory
from experiments.r3_pilot import common_repository_source_views_v1 as source_views
from experiments.r3_pilot import common_rooted_structured_event_graph_v1 as structured_graph
from experiments.r3_pilot import common_structured_role_path_projection_v1 as role_projection
from experiments.r3_pilot import public_role_transfer_mini_swe_v1 as public_transfer
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import python_entry_return_provenance_v1 as python_provenance
from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from humanbrain.simple_swe.artifacts import ArtifactLedger
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _predicate_checkpoint() -> gated.FrozenPredicateGateV1:
    return gated.fit_causal_predicate_gate(
        (
            gated.observe_predicate_gate_lesson(
                accumulator_before=11,
                item=-10,
                threshold=3,
                observed_after=1,
            ),
            gated.observe_predicate_gate_lesson(
                accumulator_before=-5,
                item=7,
                threshold=-1,
                observed_after=-5,
            ),
            gated.observe_predicate_gate_lesson(
                accumulator_before=6,
                item=2,
                threshold=8,
                observed_after=8,
            ),
        ),
        feature_cell_ids=(911, 307),
        action_cell_ids=(701, 113),
    )


def _encoded_ints(
    values: Sequence[int],
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in values
    )


def _fit_recurrent_policy_once(
) -> recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
    signature = absolute.list_int_to_int_signature()
    publication = plan_bank.CausalTaskObservationCandidatePlanBankV1.build(
        _predicate_checkpoint(),
        signature,
        (([64],), ([-2],), ([4],), ([5],)),
    )
    library = fragments.fit_reusable_program_fragment_library_v1(
        publication,
        (
            _encoded_ints((64, 2, 4, 5)),
            _encoded_ints((0, -2, 4, 0)),
        ),
    )
    cases = (
        ([-8, 7, -1],),
        ([6, 5, 4, -3],),
        ([-2, 3, 9],),
        ([12, -11, 0, 5],),
        ([-9, 2, 6],),
        ([5, -4, 3],),
        ([8, 7],),
        ([-1, 0, 12],),
    )
    frontier = library.build(signature, cases)
    return recurrent.fit_causal_recurrent_fragment_relation_policy_v2(
        library,
        (
            recurrent.FragmentRelationLesson(
                frontier,
                frontier.select(_encoded_ints((-1, 1, 1, 0, 2, -1, 0, -1))),
            ),
            recurrent.FragmentRelationLesson(
                frontier,
                frontier.select(_encoded_ints((9, 7, 5, 11, 11, 7, 0, 1))),
            ),
        ),
    )


def _string_frontier() -> tuple[object, object]:
    recurrent_policy = _fit_recurrent_policy_once()
    string = ProgramType.String()
    signature = TypedSignature(
        entry_point="clean_text",
        parameters=(ProgramParameter("payload", string),),
        result=string,
    )
    training = string_repair.solve_string_relation_training_v2(
        recurrent_policy,
        signature,
        (("A1e2Z9",), ("u3B4i8",), ("O5x6a0",), ("e7Y8U9",)),
    )
    policy = string_repair.fit_string_relation_policy_v2(
        recurrent_policy,
        training,
    )
    inputs = ("a1b2c1", "x9y8z9", "zero00", "no-digits")
    expected = ("121", "989", "00", "")
    frontier = policy.build(signature, tuple((value,) for value in inputs))
    encoded = frontier.encode_visible_expected(list(expected))
    assert all(
        record.physical_outputs == encoded
        for record in frontier.repository_action_bank.records
    )
    return frontier, training.identity_program


def _observation(source: bytes) -> memory.RepositoryActionObservationV2:
    provenance = python_provenance.observe_python_entry_return_provenance_v1(
        source,
        entry_point="clean_text",
    )
    provenance.authenticate()
    receipt = structured_graph.make_structured_event_graph_producer_receipt_v1(
        source_sha256=provenance.source_sha256,
        graph_sha256=provenance.graph.sha256,
        producer_schema_sha256=provenance.schema_sha256,
        producer_attestation_sha256=provenance.sha256,
    )
    return memory.make_repository_action_observation_v2(
        source_views.project_repository_source_views_v1(source),
        role_projection.project_structured_role_paths_v1(provenance.graph),
        producer_receipt=receipt,
    )


def _committed_lessons(
    root: Path,
    source: bytes,
    *,
    frontier: object,
    root_program: object,
    hidden_expected: str,
) -> tuple[object, ...]:
    root.mkdir()
    (root / "solution.py").write_bytes(source)
    (root / "visible.json").write_text(
        json.dumps({"expected": ["121", "989", "00", ""]}),
        encoding="utf-8",
    )
    (root / "hidden.case.json").write_text(
        json.dumps({"input": "A1b2!", "expected": hidden_expected}),
        encoding="utf-8",
    )
    (root / "check_hidden.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "root=Path(__file__).parent\n"
        "ns={}; p=root/'solution.py'\n"
        "exec(compile(p.read_text(encoding='utf-8'),str(p),'exec'),ns)\n"
        "r=json.loads((root/'hidden.case.json').read_text(encoding='utf-8'))\n"
        "raise SystemExit(0 if ns['clean_text'](r['input'])==r['expected'] else 3)\n",
        encoding="utf-8",
    )
    spec = correction.AutonomousCorrectionSpecV1(
        repo_root=root,
        source_relative_path="solution.py",
        expected_manifest_relative_path="visible.json",
        test_command=(sys.executable, "-B", "check_hidden.py"),
        root_program_sha256=root_program.sha256,
        root_source_sha256=sha256(source).hexdigest(),
    )
    result = correction.prepare_autonomous_correction_v1(frontier, spec).repair()
    result.authenticate()
    assert result.status == correction.COMMITTED
    return result.repository_actions


def _frozen_memory(
    root: Path,
) -> tuple[object, object, memory.FrozenRepositoryActionMemoryV2]:
    root.mkdir()
    frontier, root_program = _string_frontier()
    sources = (
        b"def clean_text(alpha):\n"
        b"    r0=alpha[::-1]\n"
        b"    i0=alpha[:]\n"
        b"    r1=alpha[::-1]\n"
        b"    i1=alpha[:]\n"
        b"    return r0\n",
        b"def clean_text(beta):\n"
        b"    j1=beta[:]\n"
        b"    q1=beta[::-1]\n"
        b"    j0=beta[:]\n"
        b"    q0=beta[::-1]\n"
        b"    return q0\n",
        b"def clean_text(gamma):\n"
        b"    r0=gamma[::-1]\n"
        b"    i0=gamma[:]\n"
        b"    r1=gamma[::-1]\n"
        b"    i1=gamma[:]\n"
        b"    return i0\n",
        b"def clean_text(delta):\n"
        b"    j1=delta[:]\n"
        b"    q1=delta[::-1]\n"
        b"    j0=delta[:]\n"
        b"    q0=delta[::-1]\n"
        b"    return j0\n",
    )
    expected_source_sha256s = (
        "a06bc80efb3c562ac072aa75b4213224c8d76d65b32033441ab0e49ef1278d67",
        "eb0b71163326c73644f0f4cd967b205779f10a317355ea73c3d0e2d135d1683a",
        "5c09e598efecec9ac000228de8d6df0c14db74f24bcd52ed0e434fe2ecbc6182",
        "9e311942b36c5b6b72601236351908306526a00e67b54174e9050464d6d2593f",
    )
    assert tuple(sha256(source).hexdigest() for source in sources) == (
        expected_source_sha256s
    )
    state: memory.FrozenRepositoryActionMemoryV2 | None = None
    for index, (source, hidden_expected) in enumerate(
        zip(sources, ("21", "21", "12", "12"), strict=True)
    ):
        observation = _observation(source)
        experience = memory.make_repository_action_experience_v2(
            observation,
            _committed_lessons(
                root / f"train-{index}",
                source,
                frontier=frontier,
                root_program=root_program,
                hidden_expected=hidden_expected,
            ),
        )
        state = (
            memory.start_repository_action_memory_v2(experience)
            if state is None
            else memory.advance_repository_action_memory_v2(state, experience)
        )
    assert state is not None
    state.authenticate()
    return frontier, root_program, state


def test_public_role_transfer_mini_swe_v1(tmp_path: Path) -> None:
    frontier, root_program, state = _frozen_memory(tmp_path / "training")
    bank = frontier.repository_action_bank
    frozen_memory_sha256 = state.checkpoint_sha256
    frozen_bank_sha256 = bank.sha256

    run_root = tmp_path / "public-role-transfer"
    events = public_transfer.run_public_role_transfer_mini_swe_v1(
        run_root,
        state=state,
        frontier=frontier,
        root_program_sha256=root_program.sha256,
    )
    assert events == ArtifactLedger(
        run_root / "ledger",
        protocol_sha256=public_transfer.PROTOCOL_SHA256,
    ).verify()

    assert [event.kind for event in events] == (
        ["protocol"]
        + ["positive-case"] * 12
        + ["control-case"] * 5
        + ["summary"]
    )
    protocol = events[0].body
    assert protocol["frozen_implementation_commit"] == (
        public_transfer.FROZEN_IMPLEMENTATION_COMMIT
    )
    assert protocol["case_manifest_sha256"] == public_transfer.CASE_MANIFEST_SHA256
    assert public_transfer.CASE_MANIFEST_SHA256 == (
        "f528a0d77aa173ec24d7fca650bbd3dc099b9f6513d2496dcad1f4c5b18b8198"
    )

    positives = [event.body for event in events if event.kind == "positive-case"]
    assert [body["case_id"] for body in positives] == [
        case.case_id for case in public_transfer.PUBLIC_CASES
    ]
    assert [body["family"] for body in positives[:6]] == [
        body["family"] for body in positives[6:]
    ]
    assert all(
        left["evaluator_stratum"] != right["evaluator_stratum"]
        for left, right in zip(positives[:6], positives[6:], strict=True)
    )
    reversed_rows = [body for body in positives if body["reverse_candidate_order"]]
    assert len(reversed_rows) == 6
    assert Counter(body["evaluator_stratum"] for body in reversed_rows) == {
        "R": 3,
        "I": 3,
    }
    selected_occurrences = {
        stratum: Counter(
            body["selected_occurrence"]
            for body in positives
            if body["evaluator_stratum"] == stratum
        )
        for stratum in ("R", "I")
    }
    assert selected_occurrences["R"] == selected_occurrences["I"] == Counter(
        {0: 2, 1: 2, 2: 1, 3: 1}
    )

    stratum_actions: dict[str, set[str]] = {"R": set(), "I": set()}
    selected_actions: list[str] = []
    for body in positives:
        recall = body["recall"]
        controller = body["controller"]
        assert body["source_novel"]
        assert body["frozen_roots_unchanged"]
        assert recall["status"] == memory.RECALLED
        assert recall["tier"] == memory.STRUCTURAL_ROLE
        assert recall["work"]["role_factor_match_cell_steps"] > 0
        assert recall["work"]["lexical_factor_match_cell_steps"] == 0
        action = recall["selected_action_sha256"]
        assert isinstance(action, str)
        stratum_actions[body["evaluator_stratum"]].add(action)
        selected_actions.append(action)

        assert controller["status"] == correction.COMMITTED
        assert controller["selected_action_sha256s"] == [action]
        assert controller["source_before_sha256"] == body["source_sha256"]
        assert controller["test_executed"]
        assert len(controller["repository_action_sha256s"]) == 1
        assert len(controller["attempts"]) == 1
        attempt = controller["attempts"][0]
        assert attempt["attempt"] == 1
        assert attempt["repository_test_exit_code"] == 0
        assert not attempt["rolled_back"]
        assert attempt["committed"]
        work = controller["work"]
        assert work["candidate_races"] == 1
        assert work["source_materializations"] == 1
        assert work["unselected_source_materializations"] == 0
        assert work["selected_patch_sources"] == 1
        assert work["patch_writes"] == 1
        assert work["repository_tests"] == 1
        assert work["unselected_repository_tests"] == 0
        assert work["revisions"] == 0
        assert work["replan_updates"] == 0
        assert work["reverts"] == 0
        assert work["rollback_verifications"] == 0
        assert work["commits"] == 1

    assert len(stratum_actions["R"]) == len(stratum_actions["I"]) == 1
    assert stratum_actions["R"] != stratum_actions["I"]
    assert sorted(Counter(selected_actions).values()) == [6, 6]

    controls = {
        event.body["control_id"]: event.body
        for event in events
        if event.kind == "control-case"
    }
    assert set(controls) == {
        "unknown",
        "conflict",
        "structural-role-gate",
        "producer-receipt-splice",
        "candidate-order",
    }
    for control_id, expected_status, expected_tier in (
        ("unknown", memory.STOP_LEXICAL_FACTOR_UNKNOWN, memory.LEXICAL_FACTOR),
        (
            "conflict",
            memory.STOP_STRUCTURAL_ROLE_CONFLICT,
            memory.STRUCTURAL_ROLE,
        ),
        (
            "structural-role-gate",
            memory.STOP_STRUCTURAL_ROLE_GATE,
            memory.STRUCTURAL_ROLE,
        ),
    ):
        body = controls[control_id]
        assert body["recall"]["status"] == expected_status
        assert body["recall"]["tier"] == expected_tier
        assert body["controller"]["status"] == correction.STOP_LEARNED_RECALL
        assert body["controller"]["attempts"] == []
        assert body["controller"]["work"]["repository_tests"] == 0
        assert not body["controller"]["test_executed"]
        assert body["frozen_roots_unchanged"]
    assert controls["unknown"]["recall"]["work"][
        "role_factor_match_cell_steps"
    ] == 0
    assert controls["conflict"]["recall"]["work"][
        "lexical_factor_match_cell_steps"
    ] == 0
    assert controls["producer-receipt-splice"]["rejected_before_recall"]
    assert controls["candidate-order"]["same_selected_action"]

    summary = events[-1].body
    assert summary["positive_count"] == 12
    assert summary["control_count"] == 5
    assert summary["committed_count"] == 12
    assert summary["one_attempt_count"] == 12
    assert summary["structural_role_recall_count"] == 12
    assert summary["reversed_candidate_count"] == 6
    assert sorted(summary["selected_action_counts"].values()) == [6, 6]
    assert summary["frozen_roots_unchanged"]
    state.authenticate()
    bank.authenticate()
    assert state.checkpoint_sha256 == frozen_memory_sha256
    assert bank.sha256 == frozen_bank_sha256
