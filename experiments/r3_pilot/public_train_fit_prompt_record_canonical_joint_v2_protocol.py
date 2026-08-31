"""Fresh data-only public-TRAIN prompt/I-O joint preregistration.

V1 is terminally spent and remains immutable.  V2 first chooses two different
public-TRAIN appearances by pinned source order using only a predeclared,
target-free eligibility rule.  Only after those appearances are fixed does it
open their visible regression rows and atomically bind prompt records,
signatures, cases, literals, and desired outputs.

The public runtime seam is deliberately small: authenticate the exact opaque
roster, then open one desired-free build payload.  This module exposes no
desired-opening operation and runs no producer selector, occurrence decoder,
packed frontier, proof search, electrical simulation, or result writer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as producer,
)
from experiments.r3_pilot import (
    public_train_list_int_to_int_prompt_input_capsule_v1 as input_capsule,
)
from experiments.r3_pilot import public_train_prompt_semantic_bridge as prompt_observer
from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_PROTOCOL"
PASS = "PASS_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_PROTOCOL"
ONE_SHOT_PASS = "PASS_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2"
ONE_SHOT_STOP = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2"
STOP_ROW_CERTIFICATE = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_ROW_CERTIFICATE"
)
STOP_PROMPT_IO_PAIRING_MISMATCH = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_"
    "PROMPT_IO_PAIRING_MISMATCH"
)

V1_PROTOCOL_BANK_COMMIT = "c9ae0fdad7534a5b9ed44e6cab8d057afcd1f7f0"
V1_RUNNER_BANK_COMMIT = "156dc24ef54c2a2844988f977e64e0042f22ca2c"
V1_RESULT_BANK_COMMIT = "7f4d5763607495c87a729ef4deeb93b729902760"
PRODUCER_BANK_COMMIT = "1232d2ddc14501557ac8d85e8c0846a7c4a7cee2"
STATIC_BANK_COMMIT = "d2376c7dc75314ae285ffb98edeae3e704b63d49"
CONVERTER_BANK_COMMIT = "41ad28ca7f136077a1e677029864178837b7d34d"
FRONTIER_BANK_COMMIT = "69677f7d59e8c43951002602421a1272a34c0462"
INPUT_BANK_COMMIT = "eae4944c22a748300c1210ce4b4201d21cdb0d55"

V1_PROTOCOL_SOURCE_SHA256 = (
    "a149b3468a980517c30d7322a4247b9f7506306fc80f2ced4d0ff9ab479a97a0"
)
V1_RUNNER_SOURCE_SHA256 = (
    "160e824cae03f24c5d78c0057f7567736c2cd787cac85978377b3f3291bd946e"
)
V1_RESULT_ARTIFACT_SHA256 = (
    "67c261f31fa26aac6d021501878d2fd15bc1c6ea297bba3c31accd78453d843a"
)
V1_SPENT_ARTIFACT_SHA256 = (
    "5c45278c3f8aaa6cbe2857e95d855799d933fccae84255077bcc31a9c6f33c6b"
)
V1_PROTOCOL_ROOT_SHA256 = (
    "524947ef3a29dbb4a7292c4e386d69090d9027c72fccb9559409aff1cf9cc6d0"
)
V1_PREPARED_SHA256 = (
    "d02abb1e2068c69d68bf6ee22218a42531e14d08aac4b9bc2b9eb514797cd423"
)
V1_SPENT_APPEARANCE_INDICES = (61, 71)

PRODUCER_ARTIFACT_SHA256 = (
    "6945aae4edb456462163c5428fd303a461a9bfde5abafedac0c47c6d073e868c"
)
PRODUCER_CONTENT_ROOT = (
    "a604e316ebd59a6c813408cfaa8245666d5d3387083e0c337142f7ffc45c0caf"
)
PRODUCER_PROTOCOL_SHA256 = (
    "023a2ab777abbd7159de9fa7c54cc29d31d53f2e2e1e9e4d0e36579a8c81ba8c"
)
PRODUCER_CHECKPOINT_SHA256 = (
    "de6192dc2de627c19d9078a5c887cf84cf692aa17537539cdb36944d1ab9797e"
)
INPUT_ARTIFACT_SHA256 = (
    "0e90318d1c1d3bd365f9812fb3c43aa5f0806dbfa5a1e81cb527c8e7f2c20dcf"
)
INPUT_CONTENT_ROOT = (
    "a7ff53818cf4e83c823e6b23f5547e0e7ee1fd30ed73dd136e3672063ea7aca8"
)

ROW_COUNT = 2
VISIBLE_CASE_COUNT = 4
MAX_COST = 9
SUPPORTED_SOURCE_EVENT_COUNTS = (8, 9)
TREATMENT_COUNT = 5
FITTED_GLIF_STEPS_PER_QUERY = 30
PLAN_WORK_CAPS = (30_000, 30_000, 80_000)
AGGREGATE_PLAN_WORK_CAPS = (120_000, 120_000, 320_000)

LEARNER_EXECUTION_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
PROOF_MATERIALIZATION_COUNT = 0
RUNTIME_DESIRED_READ_COUNT = 0
OLD_V1_OUTCOME_RECLASSIFICATION_COUNT = 0
RESULT_ARTIFACT_COUNT = 0
SPEND_MARKER_COUNT = 0
DEV_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
PROGRAM_INPUT_COUNT = 0

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_SOURCE_PATH = REPO_ROOT / (
    "experiments/r3_pilot/public_train_prompt_signature_capsule_v1.json"
)
SIMPLE_SWE_PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
V1_RESULT_PATH = REPO_ROOT / (
    "experiments/r3_pilot/"
    "public_train_fit_prompt_record_canonical_joint_v1_runner.result.json"
)
V1_SPENT_PATH = REPO_ROOT / (
    "experiments/r3_pilot/"
    "public_train_fit_prompt_record_canonical_joint_v1_runner.spent.json"
)
FUTURE_RUNNER_PATH = (
    "experiments/r3_pilot/"
    "public_train_fit_prompt_record_canonical_joint_v2_runner.py"
)
FUTURE_RESULT_PATH = (
    "experiments/r3_pilot/"
    "public_train_fit_prompt_record_canonical_joint_v2_runner.result.json"
)
FUTURE_SPENT_PATH = (
    "experiments/r3_pilot/"
    "public_train_fit_prompt_record_canonical_joint_v2_runner.spent.json"
)

_NAMESPACE = b"public-train-fit-prompt-record-canonical-joint-v2\0"
_COLLISION_NAMESPACE = (
    b"public-train-fit-prompt-record-canonical-joint-v2-collisions\0"
)
_SPAN_NAMESPACE = b"public-train-prompt-selection-v1-span-events\0"
_LITERAL_NAMESPACE = b"public-train-prompt-selection-v1-literal-events\0"
_ROLE_NAMESPACE = b"public-train-prompt-selection-v1-signature-role\0"
_CAUSAL_NAMESPACE = b"public-train-prompt-selection-v1-causal-key\0"

_DEPENDENCY_FILE_SHA256S = MappingProxyType(
    {
        "experiments/r3_pilot/public_train_fit_prompt_record_source_event_producer_v1.py": (
            "5e195bccf8c8ebbe59bfc6d89f6f27b9fc15ee6fe8a6a62cbe96f57650d04784"
        ),
        "experiments/r3_pilot/public_train_fit_prompt_record_source_event_producer_v1.npz": (
            PRODUCER_ARTIFACT_SHA256
        ),
        "experiments/r3_pilot/public_train_fit_occurrence_factor_static_checkpoint_capsule_v1.py": (
            "0af68609e46afd65e8dc50c656f7ad22919a08329045d3068f8a457813870c89"
        ),
        (
            "experiments/r3_pilot/public_train_fit_occurrence_factor_"
            "static_checkpoint_capsule_v1.npz"
        ): (
            "42ca5251b7a045ca98da4e2b008a82d8c89d42f229f6b3eb670ade7a30919711"
        ),
        "experiments/r3_pilot/public_train_prompt_semantic_bridge.py": (
            "74a123d88707525b63ff50021b3c7e5638f93ed8ed3dd3b82d20724bb3f1d662"
        ),
        "experiments/r3_pilot/public_train_list_int_to_int_prompt_input_capsule_v1.py": (
            "cef67a95765e268425e23cc9c270f8cdc161723584135c7b29e208b8ec9a4433"
        ),
        "experiments/r3_pilot/public_train_list_int_to_int_prompt_input_capsule_v1.json": (
            INPUT_ARTIFACT_SHA256
        ),
        "experiments/r3_pilot/public_train_prompt_signature_capsule_v1.json": (
            "0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289"
        ),
        "experiments/r3_pilot/public_typed_output_cells.py": (
            "ee4e33e1a4616c586b91f5c95ae8be0a9ec8e56edaa904f813851806ceb7c976"
        ),
        "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v1_protocol.py": (
            V1_PROTOCOL_SOURCE_SHA256
        ),
        "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v1_runner.py": (
            V1_RUNNER_SOURCE_SHA256
        ),
        (
            "experiments/r3_pilot/public_train_fit_prompt_record_"
            "canonical_joint_v1_runner.result.json"
        ): (
            V1_RESULT_ARTIFACT_SHA256
        ),
        (
            "experiments/r3_pilot/public_train_fit_prompt_record_"
            "canonical_joint_v1_runner.spent.json"
        ): (
            V1_SPENT_ARTIFACT_SHA256
        ),
        "src/humanbrain/simple_swe/tasks.py": (
            "b0edfc2eedc25335bcb2318a7b13026f6ab7affc29643af76e16f4c6b57240c3"
        ),
        "src/humanbrain/simple_swe/protocol.py": (
            "22993997004a156b7d94b8d79198deae3e2f71f04d1aebc9740b342c0aa94849"
        ),
        "src/humanbrain/typed_program_types.py": (
            "5f06a43bc771a8d61881d7ebf7c7decefcadbe245d77ce10640dbfff5174d099"
        ),
        "benchmarks/simple_swe_protocol_v1.json": (
            "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
        ),
    }
)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    if isinstance(value, TypedSignature):
        return value.to_record()
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    result = sha256()
    result.update(namespace)
    result.update(_canonical_json(value))
    return result.hexdigest()


def _snapshot(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class JointProtocolStop(RuntimeError):
    """Fail-closed V2 protocol error."""

    def __init__(self, status: str, reason: str) -> None:
        self.status = status
        super().__init__(reason)


_INTEGER = ProgramType.Int()
_LIST_INTEGER = ProgramType.List(_INTEGER)
_NEUTRAL_SIGNATURE = TypedSignature(
    "joint_entry",
    (ProgramParameter("arg0", _LIST_INTEGER),),
    _INTEGER,
)


@dataclass(frozen=True, slots=True)
class _FrozenRowSpec:
    appearance_index: int
    task_id: str
    family: str
    expected_public_task_sha256: str
    expected_prompt_record_sha256: str
    expected_prompt_sha256: str
    expected_full_signature: TypedSignature
    expected_full_signature_sha256: str
    expected_visible_problem_sha256: str
    expected_roots: tuple[str, str, str, str]
    expected_multiplicity: int
    expected_literal_cells: tuple[tuple[int, ...], ...]
    expected_cases: tuple[tuple[object, ...], ...]
    expected_literals: tuple[int, ...]
    expected_desired: tuple[int, ...]
    expected_geometry: tuple[int, int, int, int, int, int, str]


_FROZEN_ROW_SPECS = (
    _FrozenRowSpec(
        60,
        "simple-swe-v1/train/tier-2/t2_clamp_then_sum/000",
        "t2_clamp_then_sum",
        "07d5bd2a1cf1ecb0c49a83874df00582699587a6789b5c02493ed7dce0b64727",
        "1d3c4595cbfa41b98b562b4cb2db87bb17ac93d31a8ef370a1d700c18fb84e15",
        "eb9e8109792ba9fc9ca3db6210d7a39a1416be795981ddebc7feaff05b872c28",
        TypedSignature(
            "solve_6011ce2b72dee089",
            (ProgramParameter("values", _LIST_INTEGER),),
            _INTEGER,
        ),
        "64b077b91b4a16a194265859bae099a9ee3ce863b944704d0f9060d9c48e22c8",
        "bd996fbb2efa2e11f3b6eed433ed52b274140fe36fc56f685b952caae59bd01f",
        (
            "73bc5fede7406b0071c128b96be17ad027844dbc43b5f098d38588b2a84b3188",
            "512a68fca0469c17dc8d64590ccdfb4f9297146bc2de0d0b35fc512ad38c55a5",
            "945c6e4b1a8c5a77d711060711bbb6f412935d3d6cd48e7a9c5b2dc7300c63e4",
            "d56f3f0d2f5b5f26ddd5061927f750ad071fa60fef4174cb0fd10828c08686e2",
        ),
        3,
        ((0, 62), (0, 76)),
        (((),), ((0,),), ((-8, 8),), ((1, 2, 3),)),
        (-7, 0, 7),
        (0, 0, 0, 6),
        (
            70,
            69,
            2,
            552,
            2,
            554,
            "81348e943876c13d60113170e61643b2b505d625842735c4484aea1567fcb90f",
        ),
    ),
    _FrozenRowSpec(
        62,
        "simple-swe-v1/train/tier-2/t2_clamp_then_sum/002",
        "t2_clamp_then_sum",
        "90c8dbb8627b6167c441440812327140f512061278fae5e386c901b6c75fd004",
        "b218e14566cdf4176ce9d025730fde6f58c2deb88dc3cb7a3c4172997440994a",
        "e6b84dfaef34d704cf74e9649e8192379fa04745d4020f5202a381d6cf1730d4",
        TypedSignature(
            "solve_7994ea445e2aa4d7",
            (ProgramParameter("values", _LIST_INTEGER),),
            _INTEGER,
        ),
        "9ed2a3dc71e6c06e26d93a84d4e567e5aed4ffd97bb93c0e382eb8969e308d8b",
        "f9d0c16e395a95ccfde8bfa442613435873cb05b70b219bd1d902bffe3609ae6",
        (
            "73bc5fede7406b0071c128b96be17ad027844dbc43b5f098d38588b2a84b3188",
            "b0beb20f3f2e51fbb096e27b79e210f9c0a48fb3793176db03847e2ff409e03e",
            "945c6e4b1a8c5a77d711060711bbb6f412935d3d6cd48e7a9c5b2dc7300c63e4",
            "fb63e9413feeb3d64774d9175668fbc40ce3fe664021e70c6e9288d191e11330",
        ),
        1,
        ((0, 65), (0, 73)),
        (((),), ((0,),), ((-5, 5),), ((1, 2, 3),)),
        (-4, 0, 4),
        (0, 0, 0, 6),
        (
            70,
            69,
            2,
            552,
            2,
            554,
            "81348e943876c13d60113170e61643b2b505d625842735c4484aea1567fcb90f",
        ),
    ),
)

NEUTRAL_ELIGIBILITY_RULE = MappingProxyType(
    {
        "source": "pinned-public-train-prompt-signature-source-order",
        "excluded_exact_spent_v1_appearances": V1_SPENT_APPEARANCE_INDICES,
        "signature_shape": "one-List[Int]-argument-to-Int; spellings ignored",
        "physical_requirement": (
            "exact re-observation authenticates a member of the pinned "
            "target-free input capsule"
        ),
        "candidate_requirement": (
            "pinned producer normalization and candidate construction succeed"
        ),
        "task_family_name_program_root_desired_authority": False,
        "visible_io_opened_during_selection": False,
        "take": "first-two-eligible",
    }
)

V1_IMMUTABLE_SPENT_COHORT = MappingProxyType(
    {
        "protocol_bank_commit": V1_PROTOCOL_BANK_COMMIT,
        "runner_bank_commit": V1_RUNNER_BANK_COMMIT,
        "result_bank_commit": V1_RESULT_BANK_COMMIT,
        "protocol_root_sha256": V1_PROTOCOL_ROOT_SHA256,
        "prepared_sha256": V1_PREPARED_SHA256,
        "result_artifact_sha256": V1_RESULT_ARTIFACT_SHA256,
        "spent_artifact_sha256": V1_SPENT_ARTIFACT_SHA256,
        "appearance_indices": V1_SPENT_APPEARANCE_INDICES,
        "verdict": "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1",
        "reason": "producer candidate count changed",
        "actual_first_candidate_count": 554,
        "actual_first_fitted_glif_steps": 16_620,
        "terminal_and_never_reclassified": True,
    }
)


class CandidateGeometryReceipt(NamedTuple):
    appearance_index: int
    span_record_count: int
    unique_assembly_count: int
    unique_literal_count: int
    raw_candidate_count: int
    literal_candidate_count: int
    total_candidate_count: int
    collision_root_sha256: str
    candidate_events_sha256: str
    sha256: str


class EligibilityReceipt(NamedTuple):
    source_appearance_count: int
    scanned_appearance_count: int
    selected_appearance_indices: tuple[int, ...]
    excluded_spent_appearance_indices: tuple[int, ...]
    visible_io_read_count: int
    task_family_name_program_desired_authority_count: int
    sha256: str


class _ObservedAppearance(NamedTuple):
    appearance_index: int
    prompt_record: Mapping[str, object]
    full_signature: TypedSignature
    span_records: tuple[Mapping[str, object], ...]
    literal_records: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]
    roots: tuple[str, str, str, str]
    input_multiplicity: int
    geometry: CandidateGeometryReceipt


class _FrozenBuildRecord(NamedTuple):
    span_records: tuple[Mapping[str, object], ...]
    literal_records: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]
    signature: TypedSignature
    cases: tuple[tuple[object, ...], ...]
    literals: tuple[int, ...]
    max_cost: int


class _FrozenSelectionRecord(NamedTuple):
    desired: tuple[int, ...]


class JointRowProvenanceReceipt(NamedTuple):
    appearance_index: int
    task_id: str
    family: str
    public_task_sha256: str
    prompt_record_sha256: str
    prompt_sha256: str
    full_signature_sha256: str
    visible_problem_sha256: str
    input_root_sha256: str
    span_events_sha256: str
    literal_events_sha256: str
    signature_role_events_sha256: str
    input_multiplicity: int
    candidate_geometry_sha256: str
    build_payload_sha256: str
    selection_payload_sha256: str
    pairing_sha256: str
    sha256: str


class _JointRowReceipt(NamedTuple):
    build: _FrozenBuildRecord
    selection: _FrozenSelectionRecord
    geometry: CandidateGeometryReceipt
    provenance_sha256: str
    build_sha256: str
    selection_sha256: str
    pairing_sha256: str


class _RowDraft(NamedTuple):
    receipt: _JointRowReceipt
    provenance: JointRowProvenanceReceipt


class FrozenPublicTrainJointRow:
    """Opaque exact-object row; content stays in the V2 registry."""

    __slots__ = ()

    def __new__(cls) -> FrozenPublicTrainJointRow:
        raise TypeError("public-TRAIN V2 rows require private protocol authority")


class _PreflightReceipt(NamedTuple):
    status: str
    rows: tuple[FrozenPublicTrainJointRow, ...]
    roster_sha256: str | None
    stop_reason: str | None
    electrical_execution_count: int
    proof_materialization_count: int
    runtime_desired_read_count: int
    sha256: str


class FrozenPairingPreflight:
    """Exact-object atomic roster preflight with no runtime evidence reads."""

    __slots__ = ()

    def __new__(cls) -> FrozenPairingPreflight:
        raise TypeError("V2 pairing preflights require private protocol authority")

    @property
    def status(self) -> str:
        return _lookup_preflight(self).status

    @property
    def roster_sha256(self) -> str | None:
        return _lookup_preflight(self).roster_sha256

    @property
    def stop_reason(self) -> str | None:
        return _lookup_preflight(self).stop_reason

    @property
    def electrical_execution_count(self) -> int:
        return _lookup_preflight(self).electrical_execution_count

    @property
    def proof_materialization_count(self) -> int:
        return _lookup_preflight(self).proof_materialization_count

    @property
    def runtime_desired_read_count(self) -> int:
        return _lookup_preflight(self).runtime_desired_read_count

    @property
    def sha256(self) -> str:
        return _lookup_preflight(self).sha256


class _BuildAccessReceipt(NamedTuple):
    preflight: FrozenPairingPreflight
    row: FrozenPublicTrainJointRow
    frozen: _FrozenBuildRecord
    build_sha256: str
    sha256: str


class FrozenJointBuildPayload:
    """Desired/name/root/program-free build payload after exact preflight."""

    __slots__ = ()

    def __new__(cls) -> FrozenJointBuildPayload:
        raise TypeError("V2 build payloads require private protocol authority")

    @property
    def span_records(self) -> tuple[Mapping[str, object], ...]:
        return _lookup_build_access(self).frozen.span_records

    @property
    def literal_records(self) -> tuple[Mapping[str, object], ...]:
        return _lookup_build_access(self).frozen.literal_records

    @property
    def signature_role_record(self) -> Mapping[str, object]:
        return _lookup_build_access(self).frozen.signature_role_record

    @property
    def signature(self) -> TypedSignature:
        return _lookup_build_access(self).frozen.signature

    @property
    def cases(self) -> tuple[tuple[object, ...], ...]:
        return tuple((list(row[0]),) for row in _lookup_build_access(self).frozen.cases)

    @property
    def literals(self) -> tuple[int, ...]:
        return _lookup_build_access(self).frozen.literals

    @property
    def max_cost(self) -> int:
        return _lookup_build_access(self).frozen.max_cost

    @property
    def sha256(self) -> str:
        return _lookup_build_access(self).build_sha256


class CausalControlContract(NamedTuple):
    name: str
    rows: int
    expected_statuses: tuple[str, ...]
    expected_zero_proof: bool
    no_retry: bool


class JointWorkFormula(NamedTuple):
    per_row_candidate_counts: tuple[int, ...]
    treatment_count: int
    producer_query_cap: int
    supported_source_event_counts: tuple[int, ...]
    per_row_static_query_formula: str
    static_query_cap: int
    fitted_glif_steps_per_query: int
    fitted_glif_step_cap: int
    per_plan_build: tuple[int, int, int]
    aggregate_plan_builds: tuple[int, int, int]
    plan_build_count: int


class PublicTrainCanonicalJointProtocol(NamedTuple):
    status: str
    rows: tuple[FrozenPublicTrainJointRow, ...]
    provenance: tuple[JointRowProvenanceReceipt, ...]
    candidate_geometries: tuple[CandidateGeometryReceipt, ...]
    eligibility: EligibilityReceipt
    work_formula: JointWorkFormula
    dependency_sha256: str
    v1_immutable_sha256: str
    roster_sha256: str
    controls_sha256: str
    outcome_contract_sha256: str
    work_formula_sha256: str
    one_shot_order_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str


CONTROL_CONTRACTS = (
    CausalControlContract(
        "producer-off", ROW_COUNT, ("ZERO_SELECTED_SOURCE_EVENTS",), True, True
    ),
    CausalControlContract("static-cells-off", ROW_COUNT, ("STOP_ZERO",), True, True),
    CausalControlContract(
        "converter-authority-off",
        ROW_COUNT,
        (
            "STOP_PACKED_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_AUTHORIZED_"
            "PLAN_SLICE_V1_SOURCE_CERTIFICATE",
        ),
        True,
        True,
    ),
    CausalControlContract(
        "desired-permutation",
        ROW_COUNT,
        (
            "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_NO_EXACT",
            "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_AUTHORIZED_ECLASS_MISMATCH",
        ),
        True,
        True,
    ),
)

PAIRING_MISMATCH_GATE = MappingProxyType(
    {
        "delegated_status": (
            "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_AUTHORIZED_ECLASS_MISMATCH"
        ),
        "public_status": STOP_PROMPT_IO_PAIRING_MISMATCH,
        "required_exact_proof_count": 0,
        "required_witness_count": 0,
        "fallback_allowed": False,
    }
)

OUTCOME_CONTRACT = MappingProxyType(
    {
        "one_shot_pass_status": ONE_SHOT_PASS,
        "one_shot_stop_status": ONE_SHOT_STOP,
        "required_control_contracts": tuple(item.name for item in CONTROL_CONTRACTS),
        "desired_opening_interface_present": False,
        "future_private_exact_payload_to_frontier_binding_required": True,
        "shape_only_binding_allowed": False,
        "normal_rows_execute_in_frozen_order_even_after_scientific_stop": True,
        "retry_search_substitution_cap_raise_allowed": False,
        "publish_result_from_protocol_module": False,
    }
)

ONE_SHOT_ORDER = (
    "authenticate-v1-immutability-dependencies-and-v2-atomic-roster",
    "v2-row-certificate-forgery-preflight",
    "v2-producer-off-row-0",
    "v2-producer-off-row-1",
    "v2-static-cells-off-row-0",
    "v2-static-cells-off-row-1",
    "v2-converter-authority-off-row-0",
    "v2-converter-authority-off-row-1",
    "v2-desired-permutation-row-0",
    "v2-desired-permutation-row-1",
    "v2-normal-row-0",
    "v2-normal-row-1",
    "freeze-one-v2-result-and-create-v2-spent-marker",
)

SURFACE_BOUNDARY = MappingProxyType(
    {
        "scope": "fresh-two-row-public-train-same-fit-data-only-preregistration",
        "same_fit_finite_cue_replay_only": True,
        "unseen_prompt_or_family_transfer": False,
        "language_understanding": False,
        "coding_capability": False,
        "learner_execution_count": LEARNER_EXECUTION_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "proof_materialization_count": PROOF_MATERIALIZATION_COUNT,
        "runtime_desired_read_count": RUNTIME_DESIRED_READ_COUNT,
        "old_v1_outcome_reclassification_count": (
            OLD_V1_OUTCOME_RECLASSIFICATION_COUNT
        ),
        "result_artifact_count": RESULT_ARTIFACT_COUNT,
        "spend_marker_count": SPEND_MARKER_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "heldout_input_count": HELDOUT_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "program_input_count": PROGRAM_INPUT_COUNT,
    }
)

_ISSUED_ROWS: dict[int, tuple[FrozenPublicTrainJointRow, _JointRowReceipt]] = {}
_ISSUED_PREFLIGHTS: dict[int, tuple[FrozenPairingPreflight, _PreflightReceipt]] = {}
_ISSUED_BUILD_PAYLOADS: dict[
    int, tuple[FrozenJointBuildPayload, _BuildAccessReceipt]
] = {}


def _lookup_row(row: object) -> _JointRowReceipt:
    if type(row) is not FrozenPublicTrainJointRow:
        raise JointProtocolStop(STOP_ROW_CERTIFICATE, "V2 row has the wrong type")
    issued = _ISSUED_ROWS.get(id(row))
    if issued is None or issued[0] is not row:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE, "V2 row is not an issued exact object"
        )
    return issued[1]


def _lookup_preflight(preflight: object) -> _PreflightReceipt:
    if type(preflight) is not FrozenPairingPreflight:
        raise JointProtocolStop(STOP_ROW_CERTIFICATE, "V2 preflight has the wrong type")
    issued = _ISSUED_PREFLIGHTS.get(id(preflight))
    if issued is None or issued[0] is not preflight:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE, "V2 preflight is not an issued exact object"
        )
    return issued[1]


def _lookup_build_access(payload: object) -> _BuildAccessReceipt:
    if type(payload) is not FrozenJointBuildPayload:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE, "V2 build payload has the wrong type"
        )
    issued = _ISSUED_BUILD_PAYLOADS.get(id(payload))
    if issued is None or issued[0] is not payload:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE, "V2 build payload is not an issued exact object"
        )
    return issued[1]


def _issue_row(receipt: _JointRowReceipt) -> FrozenPublicTrainJointRow:
    row = object.__new__(FrozenPublicTrainJointRow)
    _ISSUED_ROWS[id(row)] = (row, receipt)
    return row


def _issue_preflight(
    status: str,
    rows: tuple[FrozenPublicTrainJointRow, ...],
    roster_sha256: str | None,
    stop_reason: str | None,
) -> FrozenPairingPreflight:
    record = (
        status,
        len(rows),
        roster_sha256,
        stop_reason,
        ELECTRICAL_EXECUTION_COUNT,
        PROOF_MATERIALIZATION_COUNT,
        RUNTIME_DESIRED_READ_COUNT,
    )
    receipt = _PreflightReceipt(
        status,
        rows,
        roster_sha256,
        stop_reason,
        0,
        0,
        0,
        _digest("pairing-preflight", record),
    )
    preflight = object.__new__(FrozenPairingPreflight)
    _ISSUED_PREFLIGHTS[id(preflight)] = (preflight, receipt)
    return preflight


def _authenticate_dependencies() -> tuple[str, str]:
    for relative, expected in _DEPENDENCY_FILE_SHA256S.items():
        actual = sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"banked V2 protocol dependency changed: {relative}")
    if (
        producer.ARTIFACT_SHA256 != PRODUCER_ARTIFACT_SHA256
        or producer.ARTIFACT_CONTENT_ROOT != PRODUCER_CONTENT_ROOT
        or producer.PROTOCOL_SHA256 != PRODUCER_PROTOCOL_SHA256
        or producer.CHECKPOINT_SHA256 != PRODUCER_CHECKPOINT_SHA256
        or producer.occurrence_static.OUTPUT_STEPS != FITTED_GLIF_STEPS_PER_QUERY
        or input_capsule.ARTIFACT_SHA256 != INPUT_ARTIFACT_SHA256
        or input_capsule.CONTENT_ROOT != INPUT_CONTENT_ROOT
        or output_cells.INT_MIN != -64
        or output_cells.INT_MAX != 64
        or output_cells.TYPE_CELL_COUNT != 5
        or output_cells.INT_OFFSET != 5
    ):
        raise RuntimeError("banked target-free prompt/producer geometry changed")

    result = json.loads(V1_RESULT_PATH.read_bytes())
    spent = json.loads(V1_SPENT_PATH.read_bytes())
    if (
        result.get("verdict") != V1_IMMUTABLE_SPENT_COHORT["verdict"]
        or result.get("stop_reason") != V1_IMMUTABLE_SPENT_COHORT["reason"]
        or result.get("prepared_sha256") != V1_PREPARED_SHA256
        or result.get("protocol", {}).get("protocol_sha256")
        != V1_PROTOCOL_ROOT_SHA256
        or spent.get("state") != "PUBLIC_TRAIN_JOINT_ATTEMPT_SPENT"
        or spent.get("no_retry") is not True
        or spent.get("prepared_sha256") != V1_PREPARED_SHA256
        or spent.get("protocol_root_sha256") != V1_PROTOCOL_ROOT_SHA256
    ):
        raise RuntimeError("terminal V1 cohort was changed or reclassified")
    v1_immutable_sha = _digest("terminal-v1-immutable", V1_IMMUTABLE_SPENT_COHORT)
    dependency_sha = _digest(
        "dependencies",
        {
            "commits": {
                "v1_protocol": V1_PROTOCOL_BANK_COMMIT,
                "v1_runner": V1_RUNNER_BANK_COMMIT,
                "v1_result": V1_RESULT_BANK_COMMIT,
                "producer": PRODUCER_BANK_COMMIT,
                "static": STATIC_BANK_COMMIT,
                "converter": CONVERTER_BANK_COMMIT,
                "frontier": FRONTIER_BANK_COMMIT,
                "input": INPUT_BANK_COMMIT,
            },
            "files": dict(_DEPENDENCY_FILE_SHA256S),
            "producer_roots": (
                PRODUCER_CONTENT_ROOT,
                PRODUCER_PROTOCOL_SHA256,
                PRODUCER_CHECKPOINT_SHA256,
            ),
            "input_content_root": INPUT_CONTENT_ROOT,
            "v1_immutable_sha256": v1_immutable_sha,
        },
    )
    return dependency_sha, v1_immutable_sha


def _strict_prompt_source() -> tuple[Mapping[str, object], ...]:
    source = json.loads(PROMPT_SOURCE_PATH.read_bytes())
    if (
        not isinstance(source, Mapping)
        or set(source) != {"records", "schema", "scope"}
        or source["schema"] != "public-train-prompt-signature-capsule-v1"
        or source["scope"] != "PUBLIC_TRAIN_PROMPTS_AND_SIGNATURES_ONLY"
        or not isinstance(source["records"], list)
        or len(source["records"]) != 90
        or any(not isinstance(row, Mapping) for row in source["records"])
    ):
        raise RuntimeError("public-TRAIN prompt/signature source changed")
    return tuple(_freeze(row) for row in source["records"])


def _signature_role_record(signature: TypedSignature) -> dict[str, object]:
    count = len(signature.parameters)
    return {
        "parameters": [
            {
                "role": f"ARG{count - index - 1}",
                "type": parameter.type.to_record(),
            }
            for index, parameter in enumerate(signature.parameters)
        ],
        "result": signature.result.to_record(),
        "role_binding": "reverse-declaration-depth",
        "entrypoint_spelling_present": False,
        "parameter_spelling_present": False,
    }


def _observe_prompt_records(
    prompt: str,
    signature: TypedSignature,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
    tuple[str, str, str, str],
]:
    observed = prompt_observer.observe_public_train_prompt(prompt, signature)
    spans = tuple(
        {
            "start_token": span.start_token,
            "stop_token": span.stop_token,
            "first_spike_step": span.first_spike_step,
            "assembly_cells": tuple(span.assembly_cells),
            "spike_cells": tuple(span.spike_cells),
        }
        for span in observed.spans
    )
    literals = tuple(
        {
            "canonical_token_index": literal.canonical_token_index,
            "typed_output_active_cells": tuple(literal.typed_output_active_cells),
        }
        for literal in observed.signed_literals
    )
    role = _signature_role_record(signature)
    span_root = _namespaced_sha256(_SPAN_NAMESPACE, spans)
    literal_root = _namespaced_sha256(_LITERAL_NAMESPACE, literals)
    role_root = _namespaced_sha256(_ROLE_NAMESPACE, role)
    input_root = _namespaced_sha256(
        _CAUSAL_NAMESPACE,
        {
            "physical_span_events_sha256": span_root,
            "typed_literal_events_sha256": literal_root,
            "signature_role_events_sha256": role_root,
            "causal_sources": (
                "ordered-physical-span-events",
                "ordered-typed-literal-cell-events",
                "ordered-name-free-signature-role-events",
            ),
        },
    )
    return (
        tuple(_freeze(row) for row in spans),
        tuple(_freeze(row) for row in literals),
        _freeze(role),
        (span_root, literal_root, role_root, input_root),
    )


def _candidate_geometry(
    appearance_index: int,
    spans: tuple[Mapping[str, object], ...],
    literals: tuple[Mapping[str, object], ...],
    role: Mapping[str, object],
) -> CandidateGeometryReceipt:
    assemblies, literal_cells = producer._normalize_prompt_records(
        spans, literals, role
    )
    candidates = producer._candidate_events(assemblies, literal_cells)
    raw_count = sum(type(item) is producer.RawPhysicalSourceEvent for item in candidates)
    literal_count = sum(
        type(item) is producer.LiteralPhysicalSourceEvent for item in candidates
    )
    if raw_count + literal_count != len(candidates):
        raise RuntimeError("pinned producer emitted a new candidate event type")

    positions: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, span in enumerate(spans):
        positions[tuple(span["assembly_cells"])].append(index)
    collisions = tuple(
        {
            "assembly_cells": assembly,
            "span_indices": tuple(indices),
            "multiplicity": len(indices),
        }
        for assembly, indices in sorted(positions.items())
        if len(indices) > 1
    )
    collision_root = _namespaced_sha256(_COLLISION_NAMESPACE, collisions)
    candidate_events_sha = _digest(
        "producer-normalized-candidate-events",
        tuple(item.neural_record() for item in candidates),
    )
    record = (
        appearance_index,
        len(spans),
        len(assemblies),
        len(literal_cells),
        raw_count,
        literal_count,
        len(candidates),
        collision_root,
        candidate_events_sha,
    )
    return CandidateGeometryReceipt(*record, _digest("candidate-geometry", record))


def _select_target_free_appearances(
    source_rows: tuple[Mapping[str, object], ...],
    frozen_inputs: input_capsule.PromptInputCapsule,
) -> tuple[tuple[_ObservedAppearance, ...], EligibilityReceipt]:
    selected: list[_ObservedAppearance] = []
    scanned = 0
    for appearance_index, prompt_record in enumerate(source_rows):
        scanned = appearance_index + 1
        if appearance_index in V1_SPENT_APPEARANCE_INDICES:
            continue
        if set(prompt_record) != {"prompt", "signature"}:
            raise RuntimeError("prompt source row schema changed")
        prompt = prompt_record["prompt"]
        if type(prompt) is not str:
            raise RuntimeError("prompt source text changed type")
        full_signature = TypedSignature.from_record(prompt_record["signature"])
        if (
            len(full_signature.parameters) != 1
            or full_signature.parameters[0].type != _LIST_INTEGER
            or full_signature.result != _INTEGER
        ):
            continue
        spans, literals, role, roots = _observe_prompt_records(prompt, full_signature)
        matches = tuple(
            item
            for item in frozen_inputs.iter_inputs()
            if _plain((spans, literals, role)) == _plain(item.neural_records())
        )
        if not matches:
            continue
        if len(matches) != 1 or matches[0].input_root != roots[3]:
            raise RuntimeError("target-free physical input authentication changed")
        frozen_input = matches[0]
        geometry = _candidate_geometry(appearance_index, spans, literals, role)
        selected.append(
            _ObservedAppearance(
                appearance_index,
                prompt_record,
                full_signature,
                spans,
                literals,
                role,
                roots,
                frozen_input.multiplicity,
                geometry,
            )
        )
        if len(selected) == ROW_COUNT:
            break
    if len(selected) != ROW_COUNT:
        raise RuntimeError("neutral source-order eligibility found fewer than two rows")
    indices = tuple(item.appearance_index for item in selected)
    record = {
        "neutral_rule": NEUTRAL_ELIGIBILITY_RULE,
        "source_appearance_count": len(source_rows),
        "scanned_appearance_count": scanned,
        "selected_appearance_indices": indices,
        "excluded_spent_appearance_indices": V1_SPENT_APPEARANCE_INDICES,
        "visible_io_read_count": 0,
        "task_family_name_program_desired_authority_count": 0,
    }
    receipt = EligibilityReceipt(
        len(source_rows),
        scanned,
        indices,
        V1_SPENT_APPEARANCE_INDICES,
        0,
        0,
        _digest("neutral-eligibility", record),
    )
    return tuple(selected), receipt


def _decode_literal_cells(
    literal_records: Sequence[Mapping[str, object]],
) -> tuple[int, ...]:
    values: set[int] = {0}
    for record in literal_records:
        cells = tuple(record["typed_output_active_cells"])
        if (
            len(cells) != 2
            or cells[0] != output_cells.TYPE_OFFSET
            or type(cells[1]) is not int
        ):
            raise JointProtocolStop(
                STOP_PROMPT_IO_PAIRING_MISMATCH,
                "typed literal is not one physical Int event",
            )
        winner = cells[1] - output_cells.INT_OFFSET
        if not 0 <= winner < output_cells.INT_KNOWN_CELL_COUNT:
            raise JointProtocolStop(
                STOP_PROMPT_IO_PAIRING_MISMATCH,
                "typed literal Int event is out of range",
            )
        values.add(output_cells.INT_MIN + winner)
    return tuple(sorted(values))


def _visible_train_tasks() -> tuple[GeneratedTask, ...]:
    protocol = load_protocol(SIMPLE_SWE_PROTOCOL_PATH)
    if protocol.path != SIMPLE_SWE_PROTOCOL_PATH.resolve():
        raise RuntimeError("public-TRAIN protocol path changed")
    tasks = generate_visible_partition(protocol, "train")
    if len(tasks) != 90:
        raise RuntimeError("public-TRAIN appearance count changed")
    return tasks


def _build_draft(
    spec: _FrozenRowSpec,
    observed: _ObservedAppearance,
    task: GeneratedTask,
) -> _RowDraft:
    prompt = observed.prompt_record["prompt"]
    cases = tuple(_snapshot(case.args) for case in task.private.regression_cases)
    desired = tuple(_snapshot(case.expected) for case in task.private.regression_cases)
    visible_problem = {
        "signature": observed.full_signature.to_record(),
        "arguments": cases,
        "desired": desired,
    }
    prompt_record_sha = sha256(_canonical_json(observed.prompt_record)).hexdigest()
    prompt_sha = sha256(prompt.encode("utf-8")).hexdigest()
    signature_sha = sha256(
        _canonical_json(observed.full_signature.to_record())
    ).hexdigest()
    visible_sha = sha256(_canonical_json(visible_problem)).hexdigest()
    public = task.public
    literal_cells = tuple(
        tuple(record["typed_output_active_cells"])
        for record in observed.literal_records
    )
    geometry_fields = (
        observed.geometry.span_record_count,
        observed.geometry.unique_assembly_count,
        observed.geometry.unique_literal_count,
        observed.geometry.raw_candidate_count,
        observed.geometry.literal_candidate_count,
        observed.geometry.total_candidate_count,
        observed.geometry.collision_root_sha256,
    )
    if (
        observed.appearance_index != spec.appearance_index
        or public.partition != "train"
        or public.prompt != prompt
        or public.signature != observed.full_signature
        or public.task_id != spec.task_id
        or public.family != spec.family
        or public.task_sha256 != spec.expected_public_task_sha256
        or prompt_record_sha != spec.expected_prompt_record_sha256
        or prompt_sha != spec.expected_prompt_sha256
        or observed.full_signature != spec.expected_full_signature
        or signature_sha != spec.expected_full_signature_sha256
        or visible_sha != spec.expected_visible_problem_sha256
        or observed.roots != spec.expected_roots
        or observed.input_multiplicity != spec.expected_multiplicity
        or literal_cells != spec.expected_literal_cells
        or cases != spec.expected_cases
        or desired != spec.expected_desired
        or len(cases) != VISIBLE_CASE_COUNT
        or _decode_literal_cells(observed.literal_records) != spec.expected_literals
        or geometry_fields != spec.expected_geometry
    ):
        raise JointProtocolStop(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            "selected prompt appearance and public-TRAIN visible I/O changed",
        )

    frozen_build = _FrozenBuildRecord(
        observed.span_records,
        observed.literal_records,
        observed.signature_role_record,
        _NEUTRAL_SIGNATURE,
        cases,
        spec.expected_literals,
        MAX_COST,
    )
    frozen_selection = _FrozenSelectionRecord(spec.expected_desired)
    build_record = {
        "span_records": frozen_build.span_records,
        "literal_records": frozen_build.literal_records,
        "signature_role_record": frozen_build.signature_role_record,
        "signature": frozen_build.signature.to_record(),
        "cases": frozen_build.cases,
        "literals": frozen_build.literals,
        "max_cost": frozen_build.max_cost,
        "desired_present": False,
        "task_family_name_root_program_appearance_fields_present": False,
    }
    build_sha = _digest("build-payload", build_record)
    selection_sha = _digest("selection-payload", {"desired": spec.expected_desired})
    provenance_record = {
        "appearance_index": observed.appearance_index,
        "task_id": public.task_id,
        "family": public.family,
        "public_task_sha256": public.task_sha256,
        "prompt_record_sha256": prompt_record_sha,
        "prompt_sha256": prompt_sha,
        "full_signature_sha256": signature_sha,
        "visible_problem_sha256": visible_sha,
        "input_root_sha256": observed.roots[3],
        "span_events_sha256": observed.roots[0],
        "literal_events_sha256": observed.roots[1],
        "signature_role_events_sha256": observed.roots[2],
        "input_multiplicity": observed.input_multiplicity,
        "candidate_geometry_sha256": observed.geometry.sha256,
        "build_payload_sha256": build_sha,
        "selection_payload_sha256": selection_sha,
    }
    provenance_sha = _digest("row-provenance", provenance_record)
    pairing_sha = _digest(
        "atomic-row-pairing",
        {
            "build_payload_sha256": build_sha,
            "selection_payload_sha256": selection_sha,
            "provenance_sha256": provenance_sha,
            "candidate_geometry_sha256": observed.geometry.sha256,
            "atomic": True,
        },
    )
    provenance = JointRowProvenanceReceipt(
        observed.appearance_index,
        public.task_id,
        public.family,
        public.task_sha256,
        prompt_record_sha,
        prompt_sha,
        signature_sha,
        visible_sha,
        observed.roots[3],
        observed.roots[0],
        observed.roots[1],
        observed.roots[2],
        observed.input_multiplicity,
        observed.geometry.sha256,
        build_sha,
        selection_sha,
        pairing_sha,
        provenance_sha,
    )
    receipt = _JointRowReceipt(
        frozen_build,
        frozen_selection,
        observed.geometry,
        provenance_sha,
        build_sha,
        selection_sha,
        pairing_sha,
    )
    return _RowDraft(receipt, provenance)


def _static_query_bound(source_count: int) -> int:
    return source_count + 6 * (source_count * source_count + 18 * source_count)


def _build_work_formula(
    geometries: tuple[CandidateGeometryReceipt, ...],
) -> JointWorkFormula:
    candidate_counts = tuple(row.total_candidate_count for row in geometries)
    producer_cap = TREATMENT_COUNT * sum(candidate_counts)
    maximum_source_count = max(SUPPORTED_SOURCE_EVENT_COUNTS)
    static_cap = len(geometries) * _static_query_bound(maximum_source_count)
    fitted_cap = (
        producer_cap + static_cap
    ) * FITTED_GLIF_STEPS_PER_QUERY
    return JointWorkFormula(
        candidate_counts,
        TREATMENT_COUNT,
        producer_cap,
        SUPPORTED_SOURCE_EVENT_COUNTS,
        "for each row n in {8,9}: n + 6*(n*n + 18*n)",
        static_cap,
        FITTED_GLIF_STEPS_PER_QUERY,
        fitted_cap,
        PLAN_WORK_CAPS,
        AGGREGATE_PLAN_WORK_CAPS,
        4,
    )


def _build_protocol() -> PublicTrainCanonicalJointProtocol:
    dependency_sha, v1_immutable_sha = _authenticate_dependencies()
    source_rows = _strict_prompt_source()
    frozen_inputs = input_capsule.load_input_capsule()
    selected, eligibility = _select_target_free_appearances(source_rows, frozen_inputs)
    selected_indices = tuple(row.appearance_index for row in selected)
    expected_indices = tuple(spec.appearance_index for spec in _FROZEN_ROW_SPECS)
    if selected_indices != expected_indices:
        raise JointProtocolStop(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            "neutral source-order roster changed before visible I/O was opened",
        )

    tasks = _visible_train_tasks()
    drafts = tuple(
        _build_draft(spec, observed, tasks[observed.appearance_index])
        for spec, observed in zip(_FROZEN_ROW_SPECS, selected, strict=True)
    )
    roster_sha = _digest(
        "roster", tuple(draft.receipt.pairing_sha256 for draft in drafts)
    )
    rows = tuple(_issue_row(draft.receipt) for draft in drafts)
    provenance = tuple(draft.provenance for draft in drafts)
    geometries = tuple(observed.geometry for observed in selected)
    work_formula = _build_work_formula(geometries)
    controls_sha = _digest(
        "controls",
        {"controls": CONTROL_CONTRACTS, "pairing_gate": PAIRING_MISMATCH_GATE},
    )
    outcome_sha = _digest("outcome-contract", OUTCOME_CONTRACT)
    work_sha = _digest("work-formula", work_formula)
    order_sha = _digest(
        "one-shot-order",
        {
            "order": ONE_SHOT_ORDER,
            "future_runner_path": FUTURE_RUNNER_PATH,
            "future_result_path": FUTURE_RESULT_PATH,
            "future_spent_path": FUTURE_SPENT_PATH,
        },
    )
    surface_sha = _digest("surface-boundary", SURFACE_BOUNDARY)
    protocol_record = {
        "schema": SCHEMA,
        "status": PASS,
        "dependency_sha256": dependency_sha,
        "v1_immutable_sha256": v1_immutable_sha,
        "eligibility_sha256": eligibility.sha256,
        "roster_sha256": roster_sha,
        "provenance_sha256s": tuple(row.sha256 for row in provenance),
        "candidate_geometry_sha256s": tuple(row.sha256 for row in geometries),
        "controls_sha256": controls_sha,
        "outcome_contract_sha256": outcome_sha,
        "work_formula_sha256": work_sha,
        "one_shot_order_sha256": order_sha,
        "surface_boundary_sha256": surface_sha,
        "row_count": len(rows),
        "electrical_execution_count": 0,
        "proof_materialization_count": 0,
        "runtime_desired_read_count": 0,
        "result_artifact_count": 0,
        "spend_marker_count": 0,
    }
    return PublicTrainCanonicalJointProtocol(
        PASS,
        rows,
        provenance,
        geometries,
        eligibility,
        work_formula,
        dependency_sha,
        v1_immutable_sha,
        roster_sha,
        controls_sha,
        outcome_sha,
        work_sha,
        order_sha,
        surface_sha,
        _digest("protocol", protocol_record),
    )


PROTOCOL = _build_protocol()
ROWS = PROTOCOL.rows
PROVENANCE_RECEIPTS = PROTOCOL.provenance
CANDIDATE_GEOMETRY_RECEIPTS = PROTOCOL.candidate_geometries
ELIGIBILITY_RECEIPT = PROTOCOL.eligibility
WORK_FORMULA = PROTOCOL.work_formula


def preflight_pairing(
    rows: Sequence[FrozenPublicTrainJointRow],
) -> FrozenPairingPreflight:
    """Authenticate the complete exact V2 roster before opening payloads."""

    supplied = tuple(rows)
    try:
        receipts = tuple(_lookup_row(row) for row in supplied)
    except JointProtocolStop as error:
        return _issue_preflight(error.status, (), None, str(error))
    exact_objects = len(supplied) == len(ROWS) and all(
        actual is expected for actual, expected in zip(supplied, ROWS, strict=False)
    )
    roster_sha = _digest(
        "roster", tuple(receipt.pairing_sha256 for receipt in receipts)
    )
    if not exact_objects or roster_sha != PROTOCOL.roster_sha256:
        return _issue_preflight(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            supplied,
            roster_sha,
            "supplied rows are not the exact fresh atomic V2 roster",
        )
    return _issue_preflight(PASS, supplied, roster_sha, None)


def require_build_payload(
    preflight: FrozenPairingPreflight,
    row: FrozenPublicTrainJointRow,
) -> FrozenJointBuildPayload:
    """Open one exact desired-free payload after the atomic roster preflight."""

    preflight_receipt = _lookup_preflight(preflight)
    if preflight_receipt.status != PASS:
        raise JointProtocolStop(
            preflight_receipt.status,
            preflight_receipt.stop_reason or "V2 pairing preflight stopped",
        )
    row_receipt = _lookup_row(row)
    if not any(row is item for item in preflight_receipt.rows):
        raise JointProtocolStop(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            "V2 row is not bound to this exact preflight",
        )
    build_sha = _digest(
        "build-payload",
        {
            "span_records": row_receipt.build.span_records,
            "literal_records": row_receipt.build.literal_records,
            "signature_role_record": row_receipt.build.signature_role_record,
            "signature": row_receipt.build.signature.to_record(),
            "cases": row_receipt.build.cases,
            "literals": row_receipt.build.literals,
            "max_cost": row_receipt.build.max_cost,
            "desired_present": False,
            "task_family_name_root_program_appearance_fields_present": False,
        },
    )
    if build_sha != row_receipt.build_sha256:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE, "V2 build payload changed after preregistration"
        )
    access = _BuildAccessReceipt(
        preflight,
        row,
        row_receipt.build,
        build_sha,
        _digest(
            "build-access",
            {
                "preflight_sha256": preflight_receipt.sha256,
                "build_payload_sha256": build_sha,
                "row_pairing_sha256": row_receipt.pairing_sha256,
            },
        ),
    )
    payload = object.__new__(FrozenJointBuildPayload)
    _ISSUED_BUILD_PAYLOADS[id(payload)] = (payload, access)
    return payload


CLAIM_BOUNDARY = (
    "data-only atomic preregistration of the first two fresh target-free-eligible "
    "public-TRAIN appearances after excluding the terminal V1 rows; exact "
    "per-row producer candidate geometry and same-FIT finite-cue protocol only, "
    "with no learner, electrical run, desired opening, proof, result, spend, "
    "unseen/family transfer, heldout, DEV, M0, C1, G0, protected, or coding-"
    "capability evidence"
)


__all__ = [
    "AGGREGATE_PLAN_WORK_CAPS",
    "CANDIDATE_GEOMETRY_RECEIPTS",
    "CLAIM_BOUNDARY",
    "CONTROL_CONTRACTS",
    "ELECTRICAL_EXECUTION_COUNT",
    "ELIGIBILITY_RECEIPT",
    "FUTURE_RESULT_PATH",
    "FUTURE_RUNNER_PATH",
    "FUTURE_SPENT_PATH",
    "LEARNER_EXECUTION_COUNT",
    "NEUTRAL_ELIGIBILITY_RULE",
    "ONE_SHOT_ORDER",
    "ONE_SHOT_PASS",
    "ONE_SHOT_STOP",
    "OUTCOME_CONTRACT",
    "PAIRING_MISMATCH_GATE",
    "PASS",
    "PLAN_WORK_CAPS",
    "PROOF_MATERIALIZATION_COUNT",
    "PROTOCOL",
    "PROVENANCE_RECEIPTS",
    "RESULT_ARTIFACT_COUNT",
    "ROWS",
    "RUNTIME_DESIRED_READ_COUNT",
    "SCHEMA",
    "SPEND_MARKER_COUNT",
    "STOP_PROMPT_IO_PAIRING_MISMATCH",
    "STOP_ROW_CERTIFICATE",
    "SURFACE_BOUNDARY",
    "V1_IMMUTABLE_SPENT_COHORT",
    "V1_SPENT_APPEARANCE_INDICES",
    "WORK_FORMULA",
    "CandidateGeometryReceipt",
    "EligibilityReceipt",
    "FrozenJointBuildPayload",
    "FrozenPairingPreflight",
    "FrozenPublicTrainJointRow",
    "JointProtocolStop",
    "JointRowProvenanceReceipt",
    "JointWorkFormula",
    "PublicTrainCanonicalJointProtocol",
    "preflight_pairing",
    "require_build_payload",
]
