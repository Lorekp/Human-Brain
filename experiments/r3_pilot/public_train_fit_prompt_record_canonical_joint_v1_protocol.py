"""Data-only two-row public-TRAIN prompt/I-O joint preregistration.

This module freezes two actual public-TRAIN ``List[Int] -> Int`` prompt
appearances together with their own visible regression rows.  It re-observes
the pinned prompt text into target-free physical records, authenticates those
records against the frozen input capsule, and only then issues opaque exact
row objects.  Runtime build payloads contain no source ordinal, task/family
label, original callable spelling, input root, expected program, or desired
output.  This data-only protocol exposes no desired-opening interface; a
future runner must supply a private exact build-binding receipt rather than a
frontier with merely matching shape.

The module is deliberately inert beyond data authentication.  It runs no
prompt producer, occurrence learner, packed frontier, selection, proof, old
panel, or result writer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

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

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1_PROTOCOL"
PASS = "PASS_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1_PROTOCOL"
ONE_SHOT_PASS = "PASS_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1"
ONE_SHOT_STOP = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1"
STOP_ROW_CERTIFICATE = "STOP_PUBLIC_TRAIN_JOINT_ROW_CERTIFICATE"
STOP_PROMPT_IO_PAIRING_MISMATCH = (
    "STOP_PUBLIC_TRAIN_JOINT_PROMPT_IO_PAIRING_MISMATCH"
)
FRONTIER_READY_STATUS = "PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_READY"
SELECTION_SELECTED_STATUS = "PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_SELECTED"

PRODUCER_BANK_COMMIT = "1232d2ddc14501557ac8d85e8c0846a7c4a7cee2"
STATIC_BANK_COMMIT = "d2376c7dc75314ae285ffb98edeae3e704b63d49"
CONVERTER_BANK_COMMIT = "41ad28ca7f136077a1e677029864178837b7d34d"
FRONTIER_BANK_COMMIT = "69677f7d59e8c43951002602421a1272a34c0462"
INPUT_BANK_COMMIT = "eae4944c22a748300c1210ce4b4201d21cdb0d55"

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
STATIC_ARTIFACT_SHA256 = (
    "42ca5251b7a045ca98da4e2b008a82d8c89d42f229f6b3eb670ade7a30919711"
)
STATIC_CONTENT_ROOT = (
    "9dcd2022ad8550bf597ee3614f3052699216e2fe63c3a57e67e3bafb0b1eeba8"
)
STATIC_PROTOCOL_SHA256 = (
    "792b517f364f075c46fef51671ea26b8117eb0f78d37d3b1d0cdee6867c26e61"
)
STATIC_CHECKPOINT_SHA256 = (
    "aee5d7da81f11a2d71bd1a6cb37383f8364a5e5427fe5055ef15c11790dba290"
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
PLAN_WORK_CAPS = (30_000, 30_000, 80_000)
AGGREGATE_PLAN_WORK_CAPS = (120_000, 120_000, 320_000)
PRODUCER_CANDIDATES_PER_ACTIVATION = 562
PRODUCER_ACTIVATION_COUNT = 10
PRODUCER_QUERY_CAP = 5_620
STATIC_QUERY_CAP = 2_723
FITTED_GLIF_STEP_CAP = 250_290

LEARNER_EXECUTION_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
PROOF_MATERIALIZATION_COUNT = 0
RUNTIME_DESIRED_READ_COUNT = 0
OLD_PANEL_OUTCOME_READ_COUNT = 0
RESULT_ARTIFACT_COUNT = 0
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

_NAMESPACE = b"public-train-fit-prompt-record-canonical-joint-v1\0"
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
            STATIC_ARTIFACT_SHA256
        ),
        (
            "experiments/r3_pilot/typed_semantic_packed_public_train_fit_"
            "occurrence_factor_authorized_plan_slice_v1.py"
        ): (
            "daa8a119caec7c2700c46fadb569676fed7742413899856de6d6e2d12bf5a90c"
        ),
        (
            "experiments/r3_pilot/typed_semantic_packed_public_train_fit_"
            "prompt_record_authorized_plan_frontier_v1.py"
        ): (
            "e16e589a8fabf4f63ec8540ff8670f87bbc5b7f041277c48df67003da99ad5f4"
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
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


class JointProtocolStop(RuntimeError):
    """Fail-closed protocol error with one frozen public status."""

    def __init__(self, status: str, reason: str) -> None:
        self.status = status
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class _RowSpec:
    appearance_index: int
    task_id: str
    family: str
    expected_public_task_sha256: str
    expected_prompt_record_sha256: str
    expected_prompt_sha256: str
    expected_signature_sha256: str
    expected_visible_problem_sha256: str
    expected_input_root: str
    expected_span_root: str
    expected_literal_root: str
    expected_role_root: str
    expected_multiplicity: int
    expected_literal_cells: tuple[tuple[int, ...], ...]
    expected_cases: tuple[tuple[object, ...], ...]
    expected_literals: tuple[int, ...]
    expected_desired: tuple[int, ...]
    expected_source_event_count: int
    expected_occurrence_count: int


_ROW_SPECS = (
    _RowSpec(
        appearance_index=61,
        task_id="simple-swe-v1/train/tier-2/t2_clamp_then_sum/001",
        family="t2_clamp_then_sum",
        expected_public_task_sha256=(
            "15826daacf818e45fb2c86c3a6d15dfea76520d6d74d2e2681e65b6e615aec87"
        ),
        expected_prompt_record_sha256=(
            "03b50b5698b79293200379a8ac8ccb6d2c7af3adc28312a4e7d143f804a22479"
        ),
        expected_prompt_sha256=(
            "5f97cf065c72a970d4d80c070150b33bd17892eb8526c6ea9ddab8a97b238b4d"
        ),
        expected_signature_sha256=(
            "1a0056668600b40b6cebeb9388b5c1bb92abe1e31042a74bf46fe1036d0b347d"
        ),
        expected_visible_problem_sha256=(
            "b257767f9d8886fda9bc8da70a3794441242220e178f5f2b47e339333a850a5d"
        ),
        expected_input_root=(
            "781be7355d7b577006361245a80d38817bde4b71dd970236124d15378421f577"
        ),
        expected_span_root=(
            "73bc5fede7406b0071c128b96be17ad027844dbc43b5f098d38588b2a84b3188"
        ),
        expected_literal_root=(
            "14fe0583ee6933db287935a572729ef0fd930497cc54ff6fa0204635277700ed"
        ),
        expected_role_root=(
            "945c6e4b1a8c5a77d711060711bbb6f412935d3d6cd48e7a9c5b2dc7300c63e4"
        ),
        expected_multiplicity=1,
        expected_literal_cells=((0, 59), (0, 79)),
        expected_cases=(((),), ((0,),), ((-11, 11),), ((1, 2, 3),)),
        expected_literals=(-10, 0, 10),
        expected_desired=(0, 0, 0, 6),
        expected_source_event_count=9,
        expected_occurrence_count=9,
    ),
    _RowSpec(
        appearance_index=71,
        task_id="simple-swe-v1/train/tier-2/t2_map_then_count/001",
        family="t2_map_then_count",
        expected_public_task_sha256=(
            "ca7f28a37cfa83e6df6e2bc7ca42d471e507fa980faecb46fc7069e5420d2fda"
        ),
        expected_prompt_record_sha256=(
            "658827cb3d547c52865431150858781139bdb1518c2fd7e85b3758ceb864d249"
        ),
        expected_prompt_sha256=(
            "b775aa7e1f57dbb2c0ed3064555e7a00c5ba76bd64886086cbe13aa0771d670b"
        ),
        expected_signature_sha256=(
            "266f46cda74070c57b0414ae8df8c63816b420934dfce1a2a7c326828a1e3bb9"
        ),
        expected_visible_problem_sha256=(
            "b36829df368459722f58a4702b86094bf5c55f188a838d262785439d0fb13118"
        ),
        expected_input_root=(
            "6ce57bcff03faaee3204c0806de890bdb37d87dc80ebce31dbf069c2ceabb8d6"
        ),
        expected_span_root=(
            "0a8ced7eb5330ba32526885f2bcd3aa005c1021b6e87d9553e1bc6b3d1740793"
        ),
        expected_literal_root=(
            "a4e5dfc895d91453cef44e691acf9bd0d06deb53ca1605fbb52eb1a7e52df3c0"
        ),
        expected_role_root=(
            "945c6e4b1a8c5a77d711060711bbb6f412935d3d6cd48e7a9c5b2dc7300c63e4"
        ),
        expected_multiplicity=2,
        expected_literal_cells=((0, 73), (0, 71)),
        expected_cases=(((),), ((-2,),), ((-1,),), ((-9, 0, 9),)),
        expected_literals=(0, 2, 4),
        expected_desired=(0, 0, 1, 2),
        expected_source_event_count=8,
        expected_occurrence_count=8,
    ),
)

_INTEGER = ProgramType.Int()
_LIST_INTEGER = ProgramType.List(_INTEGER)
_NEUTRAL_SIGNATURE = TypedSignature(
    "joint_entry",
    (ProgramParameter("arg0", _LIST_INTEGER),),
    _INTEGER,
)


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
    build_payload_sha256: str
    selection_payload_sha256: str
    pairing_sha256: str
    sha256: str


class _JointRowReceipt(NamedTuple):
    build: _FrozenBuildRecord
    selection: _FrozenSelectionRecord
    provenance_sha256: str
    build_sha256: str
    selection_sha256: str
    pairing_sha256: str
    source_event_count: int
    occurrence_count: int


class _RowDraft(NamedTuple):
    receipt: _JointRowReceipt
    provenance: JointRowProvenanceReceipt


class FrozenPublicTrainJointRow:
    """Opaque exact-object row; all content stays in the module registry."""

    __slots__ = ()

    def __new__(cls) -> FrozenPublicTrainJointRow:
        raise TypeError("public-TRAIN joint rows require private protocol authority")


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
        raise TypeError("pairing preflights require private protocol authority")

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
    """Desired-free exact build payload opened after atomic preflight."""

    __slots__ = ()

    def __new__(cls) -> FrozenJointBuildPayload:
        raise TypeError("joint build payloads require private protocol authority")

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
        frozen = _lookup_build_access(self).frozen.cases
        return tuple((list(row[0]),) for row in frozen)

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


class JointWorkCaps(NamedTuple):
    producer_candidates_per_activation: int
    producer_activation_count: int
    producer_query_cap: int
    static_query_cap: int
    fitted_glif_step_cap: int
    per_plan_build: tuple[int, int, int]
    aggregate_plan_builds: tuple[int, int, int]
    plan_build_count: int


class PublicTrainCanonicalJointProtocol(NamedTuple):
    status: str
    rows: tuple[FrozenPublicTrainJointRow, ...]
    provenance: tuple[JointRowProvenanceReceipt, ...]
    dependency_sha256: str
    roster_sha256: str
    controls_sha256: str
    outcome_contract_sha256: str
    work_caps_sha256: str
    one_shot_order_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str


CONTROL_CONTRACTS = (
    CausalControlContract(
        "producer-off",
        ROW_COUNT,
        ("ZERO_SELECTED_SOURCE_EVENTS",),
        True,
        True,
    ),
    CausalControlContract(
        "static-cells-off",
        ROW_COUNT,
        ("STOP_ZERO",),
        True,
        True,
    ),
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
        "required_selected_proof_forest": None,
        "fallback_allowed": False,
    }
)

OUTCOME_CONTRACT = MappingProxyType(
    {
        "one_shot_pass_status": ONE_SHOT_PASS,
        "one_shot_stop_status": ONE_SHOT_STOP,
        "required_build_statuses": (FRONTIER_READY_STATUS,) * ROW_COUNT,
        "required_selection_statuses": (SELECTION_SELECTED_STATUS,) * ROW_COUNT,
        "required_exact_proof_counts": (1,) * ROW_COUNT,
        "required_minimum_witness_counts": (1,) * ROW_COUNT,
        "required_preselect_desired_reads": (0,) * ROW_COUNT,
        "required_preselect_proof_witnesses": (0,) * ROW_COUNT,
        "required_control_contracts": tuple(item.name for item in CONTROL_CONTRACTS),
        "desired_opening_interface_present": False,
        "future_private_exact_build_binding_required": True,
        "shape_only_frontier_binding_allowed": False,
        "any_other_outcome_is_terminal_stop": True,
        "retry_allowed": False,
        "fallback_allowed": False,
        "row_substitution_allowed": False,
        "cap_change_allowed": False,
        "publish_result_from_protocol_module": False,
    }
)

WORK_CAPS = JointWorkCaps(
    producer_candidates_per_activation=PRODUCER_CANDIDATES_PER_ACTIVATION,
    producer_activation_count=PRODUCER_ACTIVATION_COUNT,
    producer_query_cap=PRODUCER_QUERY_CAP,
    static_query_cap=STATIC_QUERY_CAP,
    fitted_glif_step_cap=FITTED_GLIF_STEP_CAP,
    per_plan_build=PLAN_WORK_CAPS,
    aggregate_plan_builds=AGGREGATE_PLAN_WORK_CAPS,
    plan_build_count=4,
)

ONE_SHOT_ORDER = (
    "authenticate-dependencies-and-atomic-roster",
    "row-certificate-forgery-preflight",
    "producer-off-row-0",
    "producer-off-row-1",
    "static-cells-off-row-0",
    "static-cells-off-row-1",
    "converter-authority-off-row-0",
    "converter-authority-off-row-1",
    "desired-permutation-row-0",
    "desired-permutation-row-1",
    "normal-row-0",
    "normal-row-1",
    "freeze-one-result-and-mark-spent",
)

SURFACE_BOUNDARY = MappingProxyType(
    {
        "scope": "two-preregistered-public-train-fit-appearances-only",
        "same_fit_finite_cue_replay": True,
        "unseen_prompt_transfer": False,
        "family_transfer": False,
        "language_understanding": False,
        "code_emission_or_execution": False,
        "learner_execution_count": LEARNER_EXECUTION_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "proof_materialization_count": PROOF_MATERIALIZATION_COUNT,
        "runtime_desired_read_count": RUNTIME_DESIRED_READ_COUNT,
        "old_panel_outcome_read_count": OLD_PANEL_OUTCOME_READ_COUNT,
        "result_artifact_count": RESULT_ARTIFACT_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "heldout_input_count": HELDOUT_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "program_input_count": PROGRAM_INPUT_COUNT,
    }
)

_ISSUED_ROWS: dict[
    int, tuple[FrozenPublicTrainJointRow, _JointRowReceipt]
] = {}
_ISSUED_PREFLIGHTS: dict[int, tuple[FrozenPairingPreflight, _PreflightReceipt]] = {}
_ISSUED_BUILD_PAYLOADS: dict[
    int, tuple[FrozenJointBuildPayload, _BuildAccessReceipt]
] = {}


def _lookup_row(row: object) -> _JointRowReceipt:
    if type(row) is not FrozenPublicTrainJointRow:
        raise JointProtocolStop(STOP_ROW_CERTIFICATE, "joint row has the wrong type")
    issued = _ISSUED_ROWS.get(id(row))
    if issued is None or issued[0] is not row:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE,
            "joint row is not an issued exact object",
        )
    return issued[1]


def _lookup_preflight(preflight: object) -> _PreflightReceipt:
    if type(preflight) is not FrozenPairingPreflight:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE,
            "pairing preflight has the wrong type",
        )
    issued = _ISSUED_PREFLIGHTS.get(id(preflight))
    if issued is None or issued[0] is not preflight:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE,
            "pairing preflight is not an issued exact object",
        )
    return issued[1]


def _lookup_build_access(payload: object) -> _BuildAccessReceipt:
    if type(payload) is not FrozenJointBuildPayload:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE,
            "joint build payload has the wrong type",
        )
    issued = _ISSUED_BUILD_PAYLOADS.get(id(payload))
    if issued is None or issued[0] is not payload:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE,
            "joint build payload is not an issued exact object",
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
    record = {
        "status": status,
        "row_count": len(rows),
        "roster_sha256": roster_sha256,
        "stop_reason": stop_reason,
        "electrical_execution_count": 0,
        "proof_materialization_count": 0,
        "runtime_desired_read_count": 0,
    }
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


def _authenticate_dependencies() -> str:
    for relative, expected in _DEPENDENCY_FILE_SHA256S.items():
        actual = sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"banked joint dependency changed: {relative}")
    if (
        input_capsule.ARTIFACT_SHA256 != INPUT_ARTIFACT_SHA256
        or input_capsule.CONTENT_ROOT != INPUT_CONTENT_ROOT
        or prompt_observer.TOKENIZER_SHA256
        != "73777898bca9c1b009a3fb652de1993464f2bc6b0e14dafca5d706a0519fe62e"
        or prompt_observer.ASSEMBLY_PROJECTION_SHA256
        != "05181981735d4dbe6f7e609e26b26bc105bc289d8e77a9a108bbd50f17fcc89c"
        or output_cells.INT_MIN != -64
        or output_cells.INT_MAX != 64
        or output_cells.TYPE_CELL_COUNT != 5
        or output_cells.INT_OFFSET != 5
    ):
        raise RuntimeError("banked prompt input geometry changed")
    return _digest(
        "dependencies",
        {
            "commits": {
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
            "static_roots": (
                STATIC_CONTENT_ROOT,
                STATIC_PROTOCOL_SHA256,
                STATIC_CHECKPOINT_SHA256,
            ),
            "input_content_root": INPUT_CONTENT_ROOT,
            "plan_work_caps": PLAN_WORK_CAPS,
        },
    )


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
    return tuple(source["records"])


def _task_index() -> Mapping[str, GeneratedTask]:
    protocol = load_protocol(SIMPLE_SWE_PROTOCOL_PATH)
    if protocol.path != SIMPLE_SWE_PROTOCOL_PATH.resolve():
        raise RuntimeError("public-TRAIN protocol path changed")
    wanted = {spec.task_id for spec in _ROW_SPECS}
    rows = {
        task.public.task_id: task
        for task in generate_visible_partition(protocol, "train")
        if task.public.task_id in wanted
    }
    if set(rows) != wanted:
        raise RuntimeError("two-row public-TRAIN roster changed")
    return MappingProxyType(rows)


def _signature_role_record(signature: TypedSignature) -> dict[str, object]:
    parameter_count = len(signature.parameters)
    return {
        "parameters": [
            {
                "role": f"ARG{parameter_count - index - 1}",
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


def _decode_literal_cells(literal_records: Sequence[Mapping[str, object]]) -> tuple[int, ...]:
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


def _build_draft(
    spec: _RowSpec,
    prompt_record: Mapping[str, object],
    task: GeneratedTask,
    frozen_inputs: input_capsule.PromptInputCapsule,
) -> _RowDraft:
    if set(prompt_record) != {"prompt", "signature"}:
        raise RuntimeError("prompt source row schema changed")
    prompt = prompt_record["prompt"]
    if type(prompt) is not str:
        raise RuntimeError("prompt source text changed type")
    full_signature = TypedSignature.from_record(prompt_record["signature"])
    cases = tuple(_snapshot(case.args) for case in task.private.regression_cases)
    desired = tuple(_snapshot(case.expected) for case in task.private.regression_cases)
    visible_problem = {
        "signature": full_signature.to_record(),
        "arguments": cases,
        "desired": desired,
    }
    prompt_record_sha = sha256(_canonical_json(prompt_record)).hexdigest()
    prompt_sha = sha256(prompt.encode("utf-8")).hexdigest()
    signature_sha = sha256(_canonical_json(full_signature.to_record())).hexdigest()
    visible_sha = sha256(_canonical_json(visible_problem)).hexdigest()
    public = task.public
    if (
        public.partition != "train"
        or public.tier != 2
        or public.ordinal != 1
        or public.family != spec.family
        or public.task_id != spec.task_id
        or public.task_sha256 != spec.expected_public_task_sha256
        or public.prompt != prompt
        or public.signature != full_signature
        or prompt_record_sha != spec.expected_prompt_record_sha256
        or prompt_sha != spec.expected_prompt_sha256
        or signature_sha != spec.expected_signature_sha256
        or visible_sha != spec.expected_visible_problem_sha256
        or cases != spec.expected_cases
        or desired != spec.expected_desired
        or len(cases) != VISIBLE_CASE_COUNT
    ):
        raise JointProtocolStop(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            "prompt appearance and public-TRAIN visible I/O are not one exact row",
        )

    spans, literal_records, role, roots = _observe_prompt_records(prompt, full_signature)
    span_root, literal_root, role_root, input_root = roots
    frozen_input = frozen_inputs.lookup(spec.expected_input_root)
    literal_cells = tuple(
        tuple(record["typed_output_active_cells"]) for record in literal_records
    )
    if (
        roots
        != (
            spec.expected_span_root,
            spec.expected_literal_root,
            spec.expected_role_root,
            spec.expected_input_root,
        )
        or input_root != frozen_input.input_root
        or frozen_input.multiplicity != spec.expected_multiplicity
        or len(spans) != 70
        or len(literal_records) != 2
        or literal_cells != spec.expected_literal_cells
        or _plain((spans, literal_records, role))
        != _plain(frozen_input.neural_records())
        or _decode_literal_cells(literal_records) != spec.expected_literals
    ):
        raise JointProtocolStop(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            "re-observed physical prompt records do not match the frozen row",
        )

    frozen_build = _FrozenBuildRecord(
        spans,
        literal_records,
        role,
        _NEUTRAL_SIGNATURE,
        cases,
        spec.expected_literals,
        MAX_COST,
    )
    frozen_selection = _FrozenSelectionRecord(spec.expected_desired)
    build_record = {
        "span_records": spans,
        "literal_records": literal_records,
        "signature_role_record": role,
        "signature": _NEUTRAL_SIGNATURE.to_record(),
        "cases": cases,
        "literals": spec.expected_literals,
        "max_cost": MAX_COST,
        "desired_present": False,
        "task_family_name_root_program_fields_present": False,
    }
    build_sha = _digest("build-payload", build_record)
    selection_sha = _digest(
        "selection-payload",
        {"desired": spec.expected_desired},
    )
    provenance_record = {
        "appearance_index": spec.appearance_index,
        "task_id": spec.task_id,
        "family": spec.family,
        "public_task_sha256": public.task_sha256,
        "prompt_record_sha256": prompt_record_sha,
        "prompt_sha256": prompt_sha,
        "full_signature_sha256": signature_sha,
        "visible_problem_sha256": visible_sha,
        "input_root_sha256": input_root,
        "span_events_sha256": span_root,
        "literal_events_sha256": literal_root,
        "signature_role_events_sha256": role_root,
        "input_multiplicity": frozen_input.multiplicity,
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
            "atomic": True,
        },
    )
    provenance = JointRowProvenanceReceipt(
        spec.appearance_index,
        spec.task_id,
        spec.family,
        public.task_sha256,
        prompt_record_sha,
        prompt_sha,
        signature_sha,
        visible_sha,
        input_root,
        span_root,
        literal_root,
        role_root,
        frozen_input.multiplicity,
        build_sha,
        selection_sha,
        pairing_sha,
        provenance_sha,
    )
    receipt = _JointRowReceipt(
        frozen_build,
        frozen_selection,
        provenance_sha,
        build_sha,
        selection_sha,
        pairing_sha,
        spec.expected_source_event_count,
        spec.expected_occurrence_count,
    )
    return _RowDraft(receipt, provenance)


def _build_protocol() -> PublicTrainCanonicalJointProtocol:
    dependency_sha = _authenticate_dependencies()
    source_rows = _strict_prompt_source()
    tasks = _task_index()
    frozen_inputs = input_capsule.load_input_capsule()
    drafts = tuple(
        _build_draft(
            spec,
            source_rows[spec.appearance_index],
            tasks[spec.task_id],
            frozen_inputs,
        )
        for spec in _ROW_SPECS
    )
    if len(drafts) != ROW_COUNT:
        raise RuntimeError("atomic joint roster is incomplete")
    roster_sha = _digest(
        "roster",
        tuple(draft.receipt.pairing_sha256 for draft in drafts),
    )
    rows = tuple(_issue_row(draft.receipt) for draft in drafts)
    provenance = tuple(draft.provenance for draft in drafts)
    controls_sha = _digest(
        "controls",
        {
            "controls": CONTROL_CONTRACTS,
            "pairing_mismatch_gate": PAIRING_MISMATCH_GATE,
        },
    )
    outcome_sha = _digest("outcome-contract", OUTCOME_CONTRACT)
    work_sha = _digest("work-caps", WORK_CAPS)
    order_sha = _digest("one-shot-order", ONE_SHOT_ORDER)
    surface_sha = _digest("surface-boundary", SURFACE_BOUNDARY)
    protocol_record = {
        "schema": SCHEMA,
        "status": PASS,
        "dependency_sha256": dependency_sha,
        "roster_sha256": roster_sha,
        "provenance_sha256s": tuple(row.sha256 for row in provenance),
        "controls_sha256": controls_sha,
        "outcome_contract_sha256": outcome_sha,
        "work_caps_sha256": work_sha,
        "one_shot_order_sha256": order_sha,
        "surface_boundary_sha256": surface_sha,
        "row_count": len(rows),
        "electrical_execution_count": 0,
        "proof_materialization_count": 0,
        "runtime_desired_read_count": 0,
    }
    return PublicTrainCanonicalJointProtocol(
        PASS,
        rows,
        provenance,
        dependency_sha,
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


def preflight_pairing(
    rows: Sequence[FrozenPublicTrainJointRow],
) -> FrozenPairingPreflight:
    """Authenticate the complete row order before any runtime payload is opened."""

    supplied = tuple(rows)
    try:
        receipts = tuple(_lookup_row(row) for row in supplied)
    except JointProtocolStop as error:
        return _issue_preflight(error.status, (), None, str(error))
    exact_objects = len(supplied) == len(ROWS) and all(
        actual is expected for actual, expected in zip(supplied, ROWS, strict=False)
    )
    roster_sha = _digest(
        "roster",
        tuple(receipt.pairing_sha256 for receipt in receipts),
    )
    if not exact_objects or roster_sha != PROTOCOL.roster_sha256:
        return _issue_preflight(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            supplied,
            roster_sha,
            "the supplied rows are not the exact atomic preregistered roster",
        )
    return _issue_preflight(PASS, supplied, roster_sha, None)


def require_build_payload(
    preflight: FrozenPairingPreflight,
    row: FrozenPublicTrainJointRow,
) -> FrozenJointBuildPayload:
    """Open one desired-free build payload after an exact atomic preflight."""

    preflight_receipt = _lookup_preflight(preflight)
    if preflight_receipt.status != PASS:
        raise JointProtocolStop(
            preflight_receipt.status,
            preflight_receipt.stop_reason or "joint pairing preflight stopped",
        )
    row_receipt = _lookup_row(row)
    if not any(row is item for item in preflight_receipt.rows):
        raise JointProtocolStop(
            STOP_PROMPT_IO_PAIRING_MISMATCH,
            "joint row is not bound to this atomic preflight",
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
            "task_family_name_root_program_fields_present": False,
        },
    )
    if build_sha != row_receipt.build_sha256:
        raise JointProtocolStop(
            STOP_ROW_CERTIFICATE,
            "joint build payload changed after preregistration",
        )
    access_sha = _digest(
        "build-access",
        {
            "preflight_sha256": preflight_receipt.sha256,
            "build_payload_sha256": build_sha,
        },
    )
    access = _BuildAccessReceipt(
        preflight,
        row,
        row_receipt.build,
        build_sha,
        access_sha,
    )
    payload = object.__new__(FrozenJointBuildPayload)
    _ISSUED_BUILD_PAYLOADS[id(payload)] = (payload, access)
    return payload


CLAIM_BOUNDARY = (
    "data-only atomic preregistration of two known public-TRAIN FIT prompt/I-O "
    "pairs; no learner or panel execution, no proof or result, and no unseen "
    "prompt, family, heldout, DEV, G0, protected, or coding-capability evidence"
)


__all__ = [
    "AGGREGATE_PLAN_WORK_CAPS",
    "CLAIM_BOUNDARY",
    "CONTROL_CONTRACTS",
    "ELECTRICAL_EXECUTION_COUNT",
    "FITTED_GLIF_STEP_CAP",
    "LEARNER_EXECUTION_COUNT",
    "ONE_SHOT_ORDER",
    "ONE_SHOT_PASS",
    "ONE_SHOT_STOP",
    "OUTCOME_CONTRACT",
    "PAIRING_MISMATCH_GATE",
    "PASS",
    "PLAN_WORK_CAPS",
    "PROGRAM_INPUT_COUNT",
    "PROOF_MATERIALIZATION_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PROTOCOL",
    "PROVENANCE_RECEIPTS",
    "RESULT_ARTIFACT_COUNT",
    "ROWS",
    "RUNTIME_DESIRED_READ_COUNT",
    "SCHEMA",
    "STOP_PROMPT_IO_PAIRING_MISMATCH",
    "STOP_ROW_CERTIFICATE",
    "SURFACE_BOUNDARY",
    "WORK_CAPS",
    "CausalControlContract",
    "FrozenJointBuildPayload",
    "FrozenPairingPreflight",
    "FrozenPublicTrainJointRow",
    "JointProtocolStop",
    "JointRowProvenanceReceipt",
    "JointWorkCaps",
    "PublicTrainCanonicalJointProtocol",
    "preflight_pairing",
    "require_build_payload",
]
