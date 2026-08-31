from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from experiments.r3_pilot import common_microrepository_autonomous_correction_v1 as correction
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from experiments.r3_pilot import common_repository_action_memory_v2 as memory_v2
from experiments.r3_pilot import common_repository_source_views_v1 as source_views
from experiments.r3_pilot import common_rooted_structured_event_graph_v1 as structured_graph
from experiments.r3_pilot import common_structured_role_path_projection_v1 as role_projection
from experiments.r3_pilot import python_entry_return_provenance_v1 as python_provenance
from humanbrain.simple_swe.artifacts import ArtifactEvent, ArtifactLedger
from humanbrain.simple_swe.tasks import canonical_sha256


FROZEN_IMPLEMENTATION_COMMIT = "ad39f18c93c337a3866790763df25647b2336057"
PROTOCOL_SHA256 = sha256(
    b"humanbrain.public-role-transfer-mini-swe.v1"
).hexdigest()
VISIBLE_MANIFEST = b'{"expected":["121","989","00",""]}'


@dataclass(frozen=True, slots=True)
class PublicRoleTransferCaseV1:
    case_id: str
    family: str
    evaluator_stratum: str
    selected_occurrence: int
    reverse_candidate_order: bool
    source: bytes
    source_sha256: str
    hidden_input: str
    hidden_expected: str


PUBLIC_CASES = (
    PublicRoleTransferCaseV1(
        "E01",
        "surface",
        "R",
        1,
        True,
        b"def clean_text(text):\n"
        b"    keepA = text[:]\n"
        b"    turnA = text[::-1]\n"
        b"    keepB = text[::1]\n"
        b"    turnB = text[::-1]\n"
        b"    return turnA\n",
        "1d5fd2a82d46c64ac4346b24c934013f2ea863002f4ac25b2961d7db093beeba",
        "M4n7!",
        "74",
    ),
    PublicRoleTransferCaseV1(
        "E02",
        "syntax_packaging",
        "I",
        1,
        True,
        b"def clean_text(content):\n"
        b"    spun=\"\".join(reversed(content))\n"
        b"    chosen=content[:]\n"
        b"    return chosen\n",
        "dbaf164af4ae30ebb5c806cd156971491204fb55fabb24ef69e4079cf65952ee",
        "q8-R3",
        "83",
    ),
    PublicRoleTransferCaseV1(
        "E03",
        "parity_probe",
        "R",
        0,
        True,
        b"def clean_text(payload):\n"
        b"    chosen=payload[::-1] if len(payload)%2 else payload[-1::-1]\n"
        b"    return chosen\n",
        "d77bfafeda3df5d2158bef3c06a8a6c5e3e91a4b35e341a0fd69e75bcd882cd8",
        "c3D4e5",
        "543",
    ),
    PublicRoleTransferCaseV1(
        "E04",
        "graph_isomorphic",
        "I",
        2,
        True,
        b"def clean_text(streamlet):\n"
        b"    spare0=streamlet[:]\n"
        b"    spare1=streamlet[::-1]\n"
        b"    chosen=streamlet[:]\n"
        b"    spare2=streamlet[::-1]\n"
        b"    return chosen\n",
        "2f462f195645eedbf55b06c3a13c8552dfb02d5b5d4441e0dce9a96b16514ba3",
        "Z6y2x9",
        "629",
    ),
    PublicRoleTransferCaseV1(
        "E05",
        "decoy_context",
        "R",
        2,
        True,
        b"def clean_text(sample):\n"
        b"    shift=sample[1:]\n"
        b"    size=len(sample)\n"
        b"    chosen=sample[::-1]\n"
        b"    copy=sample[:]\n"
        b"    return chosen\n",
        "b9a2ea2ce6c48f990d62214d158fc55710eb8e55dbc3d3583c8d9a25194e13f3",
        "5a0B4",
        "405",
    ),
    PublicRoleTransferCaseV1(
        "E06",
        "role_counterfactual",
        "I",
        1,
        True,
        b"def clean_text(unit):\n"
        b"    first=unit[::-1]\n"
        b"    second=unit[:]\n"
        b"    third=unit[:]\n"
        b"    fourth=unit[::-1]\n"
        b"    return second\n",
        "ff8140b8044cb93da6e40e644991f62f53932734dee65611cadad334880bd129",
        "p1Q7r3",
        "173",
    ),
    PublicRoleTransferCaseV1(
        "E07",
        "surface",
        "I",
        0,
        False,
        b"def clean_text(data):\n"
        b"    result=data[:]\n"
        b"    backward=data[::-1]\n"
        b"    spare=data[::-1]\n"
        b"    copy=data[::1]\n"
        b"    return result\n",
        "8fbe809777ed2642b1d3da39cc3af125df7d12667cc5a3f25755fff4ef17bc77",
        "M4n7!",
        "47",
    ),
    PublicRoleTransferCaseV1(
        "E08",
        "syntax_packaging",
        "R",
        0,
        False,
        b"def clean_text(sequence):\n"
        b"    chosen=\"\".join(reversed(sequence))\n"
        b"    plain=sequence[:]\n"
        b"    return chosen\n",
        "e6782a1eec3d3dc05faf6cefa589a54b5cd4545fad2df85f16d5f4f6e0d1b343",
        "q8-R3",
        "38",
    ),
    PublicRoleTransferCaseV1(
        "E09",
        "parity_probe",
        "I",
        0,
        False,
        b"def clean_text(payload):\n"
        b"    chosen=payload[:] if len(payload)%2 else payload[::1]\n"
        b"    return chosen\n",
        "ae64793022f962fd3be8e35086c68e96c8e24f9b72a28e5ba81093437cefcba9",
        "c3D4e5",
        "345",
    ),
    PublicRoleTransferCaseV1(
        "E10",
        "graph_isomorphic",
        "R",
        1,
        False,
        b"def clean_text(stream):\n"
        b"    spare0=stream[:]\n"
        b"    chosen=stream[::-1]\n"
        b"    spare1=stream[:]\n"
        b"    spare2=stream[::-1]\n"
        b"    return chosen\n",
        "6b9f5b622a409091e981435d646bbcfe5c3caa20360e47d59e8afffb07aca0a3",
        "Z6y2x9",
        "926",
    ),
    PublicRoleTransferCaseV1(
        "E11",
        "decoy_context",
        "I",
        3,
        False,
        b"def clean_text(value):\n"
        b"    shift=value[:-1]\n"
        b"    size=len(value)\n"
        b"    rev=value[::-1]\n"
        b"    chosen=value[:]\n"
        b"    return chosen\n",
        "c9a59fac2a4ce64276dd3748500304ec2d72963dd1b2b04ef46051b9fc75186f",
        "5a0B4",
        "504",
    ),
    PublicRoleTransferCaseV1(
        "E12",
        "role_counterfactual",
        "R",
        3,
        False,
        b"def clean_text(unit):\n"
        b"    first=unit[::-1]\n"
        b"    second=unit[:]\n"
        b"    third=unit[:]\n"
        b"    fourth=unit[::-1]\n"
        b"    return fourth\n",
        "9e5f12cd7122170f292bab5c31fecd5ce1bd3e50280f6a2fc61b2d067cb842ba",
        "p1Q7r3",
        "371",
    ),
)

UNKNOWN_SOURCE = (
    b"def clean_text(omega):\n"
    b"    chosen=omega[1:]\n"
    b"    reverse=omega[::-1]\n"
    b"    identity=omega[:]\n"
    b"    tail=omega[:-1]\n"
    b"    return chosen\n"
)
UNKNOWN_SOURCE_SHA256 = (
    "12ba1d26e4c08195266830962530d3d97916d9545c9acd19ab77a6fe4a867fe4"
)
CONFLICT_SOURCE = (
    b"def clean_text(sigma):\n"
    b"    chosen=sigma[::-1] if len(sigma)%2 else sigma[:]\n"
    b"    reverse=sigma[::-1]\n"
    b"    identity=sigma[:]\n"
    b"    return chosen\n"
)
CONFLICT_SOURCE_SHA256 = (
    "64264b6503b5e10f464932d1c3dfe2e975fb8f15741d903d5525e30edae17a7e"
)

_CHECK_HIDDEN = (
    "import json\n"
    "from pathlib import Path\n"
    "root=Path(__file__).parent\n"
    "(root/'test-executed.marker').write_text('1',encoding='utf-8')\n"
    "ns={}\n"
    "source=root/'solution.py'\n"
    "exec(compile(source.read_bytes(),str(source),'exec'),ns)\n"
    "case=json.loads((root/'hidden.case.json').read_text(encoding='utf-8'))\n"
    "ok=ns['clean_text'](case['input'])==case['expected']\n"
    "raise SystemExit(0 if ok else 3)\n"
)


def _case_manifest() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "case_id": case.case_id,
            "family": case.family,
            "evaluator_stratum": case.evaluator_stratum,
            "source_sha256": case.source_sha256,
            "selected_occurrence": case.selected_occurrence,
            "reverse_candidate_order": case.reverse_candidate_order,
            "oracle_sha256": canonical_sha256(
                {"input": case.hidden_input, "expected": case.hidden_expected}
            ),
        }
        for case in PUBLIC_CASES
    )


CASE_MANIFEST_SHA256 = (
    "f528a0d77aa173ec24d7fca650bbd3dc099b9f6513d2496dcad1f4c5b18b8198"
)


def _observe(source: bytes) -> memory_v2.RepositoryActionObservationV2:
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
    roles = role_projection.project_structured_role_paths_v1(provenance.graph)
    views = source_views.project_repository_source_views_v1(source)
    observation = memory_v2.make_repository_action_observation_v2(
        views,
        roles,
        producer_receipt=receipt,
    )
    observation.authenticate()
    return observation


def _frozen_roots(
    state: memory_v2.FrozenRepositoryActionMemoryV2,
    bank: action_bank.FrozenRepositoryActionBankV1,
) -> dict[str, object]:
    state.authenticate()
    bank.authenticate()
    return {
        "memory_checkpoint_sha256": state.checkpoint_sha256,
        "bank_sha256": bank.sha256,
        "policy_checkpoint_sha256": bank.policy_checkpoint_sha256,
        "frontier_sha256": bank.frontier_sha256,
        "root_program_sha256": bank.root_program_sha256,
        "action_sha256s": list(bank.action_sha256s),
    }


def _stage_repository(
    root: Path,
    source: bytes,
    *,
    hidden_input: str,
    hidden_expected: str,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / "solution.py").write_bytes(source)
    (root / "visible.json").write_bytes(VISIBLE_MANIFEST)
    (root / "hidden.case.json").write_text(
        json.dumps(
            {"input": hidden_input, "expected": hidden_expected},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (root / "check_hidden.py").write_text(_CHECK_HIDDEN, encoding="utf-8")


def _spec(
    root: Path,
    source: bytes,
    *,
    root_program_sha256: str,
) -> correction.AutonomousCorrectionSpecV1:
    return correction.AutonomousCorrectionSpecV1(
        repo_root=root,
        source_relative_path="solution.py",
        expected_manifest_relative_path="visible.json",
        test_command=(sys.executable, "-B", "check_hidden.py"),
        root_program_sha256=root_program_sha256,
        root_source_sha256=sha256(source).hexdigest(),
    )


def _recall_body(recall: memory_v2.RepositoryActionRecallV2) -> dict[str, object]:
    return {
        "status": recall.status,
        "tier": recall.tier,
        "sha256": recall.sha256,
        "selected_action_sha256": recall.selected_candidate_action_sha256,
        "candidate_sha256s": [item.sha256 for item in recall.candidates],
        "work": dict(recall.work._asdict()),
        "total_cell_steps": recall.work.total_cell_steps,
        "total_reads": recall.work.total_reads,
    }


def _controller_body(
    result: correction.AutonomousCorrectionResultV1,
    bank: action_bank.FrozenRepositoryActionBankV1,
    *,
    test_executed: bool,
) -> dict[str, object]:
    action_by_candidate = {
        record.sha256: bank.action_sha256_for(record) for record in bank.records
    }
    return {
        "status": result.status,
        "sha256": result.sha256,
        "selected_action_sha256s": [
            action_by_candidate[attempt.candidate_sha256]
            for attempt in result.attempts
        ],
        "attempts": [dict(attempt._asdict()) for attempt in result.attempts],
        "repository_action_sha256s": [
            lesson.sha256 for lesson in result.repository_actions
        ],
        "source_before_sha256": result.source_before_sha256,
        "source_after_sha256": result.source_after_sha256,
        "test_executed": test_executed,
        "work": dict(result.work._asdict()),
    }


def _run_stop_control(
    root: Path,
    *,
    control_id: str,
    source: bytes,
    observation: memory_v2.RepositoryActionObservationV2,
    state: memory_v2.FrozenRepositoryActionMemoryV2,
    frontier: action_bank.RepositoryActionFrontierV1,
    root_program_sha256: str,
    structural_role_gate: bool = True,
) -> dict[str, object]:
    bank = frontier.repository_action_bank
    recall = state.recall_attested_observation(
        observation,
        bank,
        structural_role_gate=structural_role_gate,
    )
    recall.authenticate()
    repository = root / "controls" / control_id
    _stage_repository(
        repository,
        source,
        hidden_input="sentinel9",
        hidden_expected="__must_not_execute__",
    )
    result = correction.prepare_autonomous_correction_v1(
        frontier,
        _spec(repository, source, root_program_sha256=root_program_sha256),
        action_memory=state,
        action_observation=observation,
        memory_structural_role_gate=structural_role_gate,
    ).repair()
    result.authenticate()
    return {
        "control_id": control_id,
        "source_sha256": sha256(source).hexdigest(),
        "recall": _recall_body(recall),
        "controller": _controller_body(
            result,
            bank,
            test_executed=(repository / "test-executed.marker").exists(),
        ),
    }


def run_public_role_transfer_mini_swe_v1(
    root: Path,
    *,
    state: memory_v2.FrozenRepositoryActionMemoryV2,
    frontier: action_bank.RepositoryActionFrontierV1,
    root_program_sha256: str,
) -> tuple[ArtifactEvent, ...]:
    """Run the answer-blind public transfer cohort against one frozen V2 memory."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    bank = frontier.repository_action_bank
    frozen_before = _frozen_roots(state, bank)
    if root_program_sha256 != bank.root_program_sha256:
        raise ValueError("root program differs from the frozen repository action bank")
    if canonical_sha256(_case_manifest()) != CASE_MANIFEST_SHA256:
        raise RuntimeError("public role-transfer manifest commitment changed")
    if len(PUBLIC_CASES) != 12 or len({case.source_sha256 for case in PUBLIC_CASES}) != 12:
        raise RuntimeError("public role-transfer cohort shape changed")
    for case in PUBLIC_CASES:
        if sha256(case.source).hexdigest() != case.source_sha256:
            raise RuntimeError(f"{case.case_id} source commitment changed")
    if sha256(UNKNOWN_SOURCE).hexdigest() != UNKNOWN_SOURCE_SHA256:
        raise RuntimeError("unknown control source commitment changed")
    if sha256(CONFLICT_SOURCE).hexdigest() != CONFLICT_SOURCE_SHA256:
        raise RuntimeError("conflict control source commitment changed")

    # All cognitive observations are complete before an evaluator oracle is staged.
    observations = {case.case_id: _observe(case.source) for case in PUBLIC_CASES}
    unknown_observation = _observe(UNKNOWN_SOURCE)
    conflict_observation = _observe(CONFLICT_SOURCE)

    ledger = ArtifactLedger(root / "ledger", protocol_sha256=PROTOCOL_SHA256)
    ledger.append(
        "protocol",
        {
            "frozen_implementation_commit": FROZEN_IMPLEMENTATION_COMMIT,
            "case_manifest_sha256": CASE_MANIFEST_SHA256,
            "case_count": len(PUBLIC_CASES),
            "frozen_roots": frozen_before,
        },
    )

    positive_events: list[ArtifactEvent] = []
    known_sources = {item.source_sha256 for item in state.source_provenance}
    for case in PUBLIC_CASES:
        observation = observations[case.case_id]
        records = (
            tuple(reversed(bank.records))
            if case.reverse_candidate_order
            else bank.records
        )
        recall = state.recall_attested_observation(
            observation,
            bank,
            candidate_records=records,
        )
        recall.authenticate()
        repository = root / "repositories" / case.case_id
        _stage_repository(
            repository,
            case.source,
            hidden_input=case.hidden_input,
            hidden_expected=case.hidden_expected,
        )
        result = correction.prepare_autonomous_correction_v1(
            frontier,
            _spec(
                repository,
                case.source,
                root_program_sha256=root_program_sha256,
            ),
            candidate_records=records,
            action_memory=state,
            action_observation=observation,
        ).repair()
        result.authenticate()
        event = ledger.append(
            "positive-case",
            {
                "case_id": case.case_id,
                "family": case.family,
                "evaluator_stratum": case.evaluator_stratum,
                "selected_occurrence": case.selected_occurrence,
                "reverse_candidate_order": case.reverse_candidate_order,
                "source_sha256": case.source_sha256,
                "source_novel": case.source_sha256 not in known_sources,
                "observation_sha256": observation.sha256,
                "graph_sha256": observation.graph_sha256,
                "producer_receipt_sha256": observation.producer_receipt_sha256,
                "source_views_sha256": observation.source_views.sha256,
                "structural_roles_sha256": observation.structural_roles.sha256,
                "oracle_sha256": canonical_sha256(
                    {
                        "input": case.hidden_input,
                        "expected": case.hidden_expected,
                    }
                ),
                "recall": _recall_body(recall),
                "controller": _controller_body(
                    result,
                    bank,
                    test_executed=(repository / "test-executed.marker").exists(),
                ),
                "frozen_roots_unchanged": _frozen_roots(state, bank)
                == frozen_before,
            },
        )
        positive_events.append(event)

    control_events: list[ArtifactEvent] = []
    for body in (
        _run_stop_control(
            root,
            control_id="unknown",
            source=UNKNOWN_SOURCE,
            observation=unknown_observation,
            state=state,
            frontier=frontier,
            root_program_sha256=root_program_sha256,
        ),
        _run_stop_control(
            root,
            control_id="conflict",
            source=CONFLICT_SOURCE,
            observation=conflict_observation,
            state=state,
            frontier=frontier,
            root_program_sha256=root_program_sha256,
        ),
        _run_stop_control(
            root,
            control_id="structural-role-gate",
            source=PUBLIC_CASES[0].source,
            observation=observations[PUBLIC_CASES[0].case_id],
            state=state,
            frontier=frontier,
            root_program_sha256=root_program_sha256,
            structural_role_gate=False,
        ),
    ):
        body["frozen_roots_unchanged"] = _frozen_roots(state, bank) == frozen_before
        control_events.append(ledger.append("control-case", body))

    splice_rejected = False
    try:
        first = observations[PUBLIC_CASES[0].case_id]
        second = observations[PUBLIC_CASES[1].case_id]
        memory_v2.make_repository_action_observation_v2(
            first.source_views,
            first.structural_roles,
            producer_receipt=second.producer_receipt,
        )
    except ValueError:
        splice_rejected = True
    control_events.append(
        ledger.append(
            "control-case",
            {
                "control_id": "producer-receipt-splice",
                "rejected_before_recall": splice_rejected,
                "frozen_roots_unchanged": _frozen_roots(state, bank)
                == frozen_before,
            },
        )
    )

    order_case = PUBLIC_CASES[0]
    canonical_order = state.recall_attested_observation(
        observations[order_case.case_id],
        bank,
        candidate_records=bank.records,
    )
    canonical_order.authenticate()
    reversed_selected = positive_events[0].body["recall"][
        "selected_action_sha256"
    ]
    control_events.append(
        ledger.append(
            "control-case",
            {
                "control_id": "candidate-order",
                "case_id": order_case.case_id,
                "canonical_recall": _recall_body(canonical_order),
                "reversed_selected_action_sha256": reversed_selected,
                "same_selected_action": canonical_order.selected_candidate_action_sha256
                == reversed_selected,
                "frozen_roots_unchanged": _frozen_roots(state, bank)
                == frozen_before,
            },
        )
    )

    action_counts = Counter(
        event.body["recall"]["selected_action_sha256"]
        for event in positive_events
        if event.body["recall"]["selected_action_sha256"] is not None
    )
    ledger.append(
        "summary",
        {
            "positive_count": len(positive_events),
            "control_count": len(control_events),
            "committed_count": sum(
                event.body["controller"]["status"] == correction.COMMITTED
                for event in positive_events
            ),
            "one_attempt_count": sum(
                len(event.body["controller"]["attempts"]) == 1
                for event in positive_events
            ),
            "structural_role_recall_count": sum(
                event.body["recall"]["status"] == memory_v2.RECALLED
                and event.body["recall"]["tier"] == memory_v2.STRUCTURAL_ROLE
                for event in positive_events
            ),
            "reversed_candidate_count": sum(
                bool(event.body["reverse_candidate_order"])
                for event in positive_events
            ),
            "selected_action_counts": dict(sorted(action_counts.items())),
            "frozen_roots_unchanged": _frozen_roots(state, bank) == frozen_before,
        },
    )
    return ledger.verify()
