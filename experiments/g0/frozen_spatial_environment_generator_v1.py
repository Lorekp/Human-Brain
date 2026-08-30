"""Frozen MiniGrid source for the G0 spatial domain.

This deep module owns environment construction, deterministic source-validation
rollouts, full-state layout identities, and raw token renaming.  Its candidate
seam is still the banked six-field byte adapter: 7x7x3 opaque image bytes,
direction, one of seven raw actions, reward, terminated, and truncated.

Only declared quarantined non-cohort seeds are executable.  The preregistered
smoke, learning, adjudicative, and rename seeds remain fail-closed until a
separate nine-artifact common-architecture binding exists.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

import minigrid
import minigrid.core.roomgrid as minigrid_roomgrid
import minigrid.core.world_object as minigrid_world_object
import minigrid.envs.doorkey as minigrid_doorkey
import minigrid.envs.keycorridor as minigrid_keycorridor
import minigrid.envs.memory as minigrid_memory
import minigrid.minigrid_env as minigrid_env_source
import numpy as np
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg
from experiments.g0 import spatial_candidate_adapter_v1 as adapter
from experiments.g0 import spatial_fixture_generator_v1 as fixture_preflight
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Floor, Goal, Key
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.envs.keycorridor import KeyCorridorEnv
from minigrid.envs.memory import MemoryEnv
from minigrid.minigrid_env import MiniGridEnv

GENERATOR_SCOPE = "FROZEN_G0_SPATIAL_ENVIRONMENT_GENERATOR_V1"
GENERATOR_STATUS = "SOURCE_FROZEN_COHORT_GENERATION_FAIL_CLOSED"
STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED = "STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED"

MINIGRID_VERSION = "3.1.0"
GYMNASIUM_VERSION = "1.3.0"
EXPECTED_PREREGISTRATION_SHA256 = "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"
EXPECTED_SPATIAL_CONFIGURATION_SHA256 = (
    "8705597cd610e00390a73bcd926fc8b3bcbfbd8be07b08646214972ad8402100"
)

_ROOT = Path(__file__).resolve().parents[2]
_PINNED_PROJECT_SOURCES = MappingProxyType(
    {
        "pyproject.toml": "70fd2232f335929385f4b61e482aa714bc4baf8a706aaa825fa126427f23f77e",
        "experiments/g0/frozen_cross_domain_g0_preregistration_v1.py": (
            "e027e598aa926b108e90208e1586a93c0da35716e165a746bb2732d980774395"
        ),
        "experiments/g0/spatial_candidate_adapter_v1.py": (
            "ac2c9307d45a48d6cdba72c5e4d0502c8e9a60af799783caa9c0fb41ad028fe1"
        ),
        "experiments/g0/spatial_fixture_generator_v1.py": (
            "a28bc1c2fd46daccd88084e49403a1b639f9ef6d8f50856c8e68b20aec00f0b3"
        ),
    }
)
_MINIGRID_MODULES = MappingProxyType(
    {
        "minigrid.envs.memory": minigrid_memory,
        "minigrid.envs.doorkey": minigrid_doorkey,
        "minigrid.envs.keycorridor": minigrid_keycorridor,
        "minigrid.minigrid_env": minigrid_env_source,
        "minigrid.core.roomgrid": minigrid_roomgrid,
        "minigrid.core.world_object": minigrid_world_object,
    }
)
MINIGRID_SOURCE_SHA256S = MappingProxyType(
    {
        "minigrid.envs.memory": (
            "d208b6b0cfb20a8ed00d55a0ea6012697f2a0da0e54e4da7d8378eb1da52df8c"
        ),
        "minigrid.envs.doorkey": (
            "b2802d8e98e9c99d54a6bbf4c04d7047cb8e0c4a3150b186bba8f4defb20114c"
        ),
        "minigrid.envs.keycorridor": (
            "9542b175a59921dcff0e4a8a3c38cfad86c1a217965f78c270c8e758df4c692e"
        ),
        "minigrid.minigrid_env": (
            "23490887fbaadd8f4973b4375cd40a23b7636e7aedf59561f767315c36cd0371"
        ),
        "minigrid.core.roomgrid": (
            "a6cebd1c2cec45ee61a1de21f53474b8660d33a5f642aa6916e8f522af30a3cf"
        ),
        "minigrid.core.world_object": (
            "f60793fd1fcaa58f373a7a8773261821b0fe0a86461412abbedd2a64ea65757d"
        ),
    }
)

FROZEN_COHORT_SURFACES = ("smoke", "learning", "adjudicative", "rename")
FROZEN_COHORT_SEEDS = (
    prereg.SPATIAL_SPLIT.smoke_seed,
    prereg.SPATIAL_SPLIT.learning_seed,
    prereg.SPATIAL_SPLIT.adjudicative_seed,
    prereg.SPATIAL_SPLIT.rename_seed,
)

ARCHITECTURE_BINDING_COUNT = 0
FROZEN_COHORT_EXECUTION_COUNT = 0
FROZEN_LAYOUT_GENERATION_COUNT = 0
LEARNER_EXECUTION_COUNT = 0
EVALUATOR_EXECUTION_COUNT = 0
OUTCOME_COUNT = 0

_DEFINITION_NAMESPACE = b"frozen-g0-spatial-definition-v1\0"
_CONFIGURATION_NAMESPACE = b"frozen-g0-spatial-configuration-v1\0"
_DEPENDENCY_NAMESPACE = b"frozen-g0-spatial-dependency-v1\0"
_RENAME_NAMESPACE = b"frozen-g0-spatial-raw-token-rename-v1\0"
_LAYOUT_NAMESPACE = b"frozen-g0-spatial-layout-identity-v1\0"
_VALIDATION_NAMESPACE = b"frozen-g0-spatial-quarantined-validation-v1\0"
_COMPOSITION_NAMESPACE = b"frozen-g0-spatial-cue-key-door-audit-v1\0"
_CUSTOM_LAYOUT_NAMESPACE = b"frozen-g0-spatial-cue-key-door-layout-family-v1\0"
_ROOT_NAMESPACE = b"frozen-g0-spatial-generator-root-v1\0"


def _plain_record(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain_record(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_record(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain_record(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


CUSTOM_LAYOUT_COUNT = 64
CUSTOM_LAYOUT_CODE_BITS = 6
CUSTOM_LAYOUT_MARKER_POSITIONS = ((5, 4), (8, 4), (11, 4), (14, 4), (17, 4), (20, 4))
CUSTOM_LAYOUT_MARKER_COLORS = ("grey", "yellow")
QUARANTINED_CUSTOM_LAYOUT_INDEX = 37


def _custom_layout_marker_pattern(layout_index: int) -> tuple[tuple[int, int, str], ...]:
    if type(layout_index) is not int or not 0 <= layout_index < CUSTOM_LAYOUT_COUNT:
        raise ValueError("custom cue-key-door layout index is outside the frozen family")
    return tuple(
        (x, y, CUSTOM_LAYOUT_MARKER_COLORS[(layout_index >> bit) & 1])
        for bit, (x, y) in enumerate(CUSTOM_LAYOUT_MARKER_POSITIONS)
    )


CUSTOM_LAYOUT_MARKER_PATTERNS = tuple(
    _custom_layout_marker_pattern(layout_index) for layout_index in range(CUSTOM_LAYOUT_COUNT)
)
if (
    CUSTOM_LAYOUT_COUNT != 2**CUSTOM_LAYOUT_CODE_BITS
    or len(CUSTOM_LAYOUT_MARKER_POSITIONS) != CUSTOM_LAYOUT_CODE_BITS
    or len(set(CUSTOM_LAYOUT_MARKER_PATTERNS)) != CUSTOM_LAYOUT_COUNT
    or set(CUSTOM_LAYOUT_MARKER_COLORS) & {"red", "blue"}
):
    raise RuntimeError("custom cue-key-door layout family is not injective and cue-neutral")
CUSTOM_LAYOUT_FAMILY_RECORD: Mapping[str, object] = MappingProxyType(
    {
        "layout_count": CUSTOM_LAYOUT_COUNT,
        "layout_code_bits": CUSTOM_LAYOUT_CODE_BITS,
        "marker_positions": CUSTOM_LAYOUT_MARKER_POSITIONS,
        "marker_colors": CUSTOM_LAYOUT_MARKER_COLORS,
        "marker_semantics": "neutral-traversable-floor",
        "pairwise_distinct_marker_patterns": True,
        "cue_answer_encoded": False,
        "layout_index_candidate_field_count": 0,
        "full_state_layout_hash_uniqueness_required_at_cohort_seal": True,
    }
)
CUSTOM_LAYOUT_FAMILY_SHA256 = _digest(
    _CUSTOM_LAYOUT_NAMESPACE,
    {
        "contract": CUSTOM_LAYOUT_FAMILY_RECORD,
        "patterns": CUSTOM_LAYOUT_MARKER_PATTERNS,
    },
)


@dataclass(frozen=True, slots=True)
class SpatialEnvironmentDefinition:
    definition_id: str
    cohort_surface: str
    preregistered_partition: str
    preregistered_layout_count: int
    factory: str
    configuration: tuple[tuple[str, int | bool], ...]
    side_length_for_timeout: int
    max_steps: int
    definition_sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "definition_id": self.definition_id,
            "cohort_surface": self.cohort_surface,
            "preregistered_partition": self.preregistered_partition,
            "preregistered_layout_count": self.preregistered_layout_count,
            "factory": self.factory,
            "configuration": self.configuration,
            "side_length_for_timeout": self.side_length_for_timeout,
            "max_steps": self.max_steps,
            "candidate_observation_shape": adapter.IMAGE_SHAPE,
            "raw_action_ids": adapter.RAW_ACTION_IDS,
            "mission_discarded": True,
        }
        if include_sha256:
            record["definition_sha256"] = self.definition_sha256
        return record


def _definition(
    definition_id: str,
    cohort_surface: str,
    partition: str,
    count: int,
    factory: str,
    configuration: tuple[tuple[str, int | bool], ...],
    side_length: int,
) -> SpatialEnvironmentDefinition:
    max_steps = 4 * side_length**2
    record = {
        "definition_id": definition_id,
        "cohort_surface": cohort_surface,
        "preregistered_partition": partition,
        "preregistered_layout_count": count,
        "factory": factory,
        "configuration": (*configuration, ("max_steps", max_steps)),
        "side_length_for_timeout": side_length,
        "max_steps": max_steps,
        "candidate_observation_shape": adapter.IMAGE_SHAPE,
        "raw_action_ids": adapter.RAW_ACTION_IDS,
        "mission_discarded": True,
    }
    return SpatialEnvironmentDefinition(
        definition_id=definition_id,
        cohort_surface=cohort_surface,
        preregistered_partition=partition,
        preregistered_layout_count=count,
        factory=factory,
        configuration=record["configuration"],
        side_length_for_timeout=side_length,
        max_steps=max_steps,
        definition_sha256=_digest(_DEFINITION_NAMESPACE, record),
    )


LEARNING_DEFINITIONS = (
    _definition(
        "learning-memory-s11-random",
        "learning",
        "memory-s11-random-style",
        192,
        "MemoryEnv",
        (("size", 11), ("random_length", True)),
        11,
    ),
    _definition(
        "learning-door-key-8x8",
        "learning",
        "door-key-8x8-style",
        160,
        "DoorKeyEnv",
        (("size", 8),),
        8,
    ),
    _definition(
        "learning-key-corridor-s3r2",
        "learning",
        "key-corridor-s3r2-style",
        160,
        "KeyCorridorEnv",
        (("room_size", 3), ("num_rows", 2)),
        7,
    ),
)

ADJUDICATION_DEFINITIONS = (
    _definition(
        "adjudicative-memory-s13-random",
        "adjudicative",
        "memory-s13-random-style",
        32,
        "MemoryEnv",
        (("size", 13), ("random_length", True)),
        13,
    ),
    _definition(
        "adjudicative-door-key-8x8",
        "adjudicative",
        "door-key",
        16,
        "DoorKeyEnv",
        (("size", 8),),
        8,
    ),
    _definition(
        "adjudicative-key-corridor-s3r2",
        "adjudicative",
        "key-corridor",
        16,
        "KeyCorridorEnv",
        (("room_size", 3), ("num_rows", 2)),
        7,
    ),
    _definition(
        "adjudicative-cue-key-door-composition",
        "adjudicative",
        "cue-key-door-composition",
        64,
        "CueKeyDoorEnv",
        (
            ("width", 27),
            ("height", 9),
            ("minimum_cue_gap", 16),
            ("layout_count", CUSTOM_LAYOUT_COUNT),
            ("layout_code_bits", CUSTOM_LAYOUT_CODE_BITS),
        ),
        27,
    ),
)

ALL_DEFINITIONS = (*LEARNING_DEFINITIONS, *ADJUDICATION_DEFINITIONS)
DEFINITION_BY_ID = MappingProxyType({row.definition_id: row for row in ALL_DEFINITIONS})


@dataclass(frozen=True, slots=True)
class RawTokenRename:
    name: str
    object_permutation: tuple[int, ...]
    color_permutation: tuple[int, ...]
    rename_sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "object_token_count": len(self.object_permutation),
            "color_token_count": len(self.color_permutation),
            "object_permutation": self.object_permutation,
            "color_permutation": self.color_permutation,
            "inverse_exposed_to_candidate": False,
        }
        if include_sha256:
            record["rename_sha256"] = self.rename_sha256
        return record


def _rename(
    name: str,
    object_permutation: tuple[int, ...],
    color_permutation: tuple[int, ...],
) -> RawTokenRename:
    if sorted(object_permutation) != list(range(len(OBJECT_TO_IDX))):
        raise RuntimeError("object rename is not a full MiniGrid token bijection")
    if sorted(color_permutation) != list(range(len(COLOR_TO_IDX))):
        raise RuntimeError("color rename is not a full MiniGrid token bijection")
    record = {
        "name": name,
        "object_permutation": object_permutation,
        "color_permutation": color_permutation,
        "inverse_exposed_to_candidate": False,
    }
    return RawTokenRename(
        name=name,
        object_permutation=object_permutation,
        color_permutation=color_permutation,
        rename_sha256=_digest(_RENAME_NAMESPACE, record),
    )


IDENTITY_RENAME = _rename(
    "identity",
    tuple(range(len(OBJECT_TO_IDX))),
    tuple(range(len(COLOR_TO_IDX))),
)
CYCLIC_TOKEN_RENAME = _rename(
    "cyclic-object-color",
    tuple((value + 3) % len(OBJECT_TO_IDX) for value in range(len(OBJECT_TO_IDX))),
    tuple((value + 2) % len(COLOR_TO_IDX) for value in range(len(COLOR_TO_IDX))),
)
RAW_TOKEN_RENAMES = MappingProxyType(
    {
        IDENTITY_RENAME.name: IDENTITY_RENAME,
        CYCLIC_TOKEN_RENAME.name: CYCLIC_TOKEN_RENAME,
    }
)

QUARANTINED_VALIDATION_SEEDS = MappingProxyType(
    {
        definition.definition_id: 9_270_001 + index * 18
        for index, definition in enumerate(ALL_DEFINITIONS)
    }
)
DEFAULT_VALIDATION_ACTIONS = (*adapter.RAW_ACTION_IDS, 2, 2, 5)

SURFACE_BOUNDARY_RECORD: Mapping[str, object] = MappingProxyType(
    {
        "common_architecture_binding_count": ARCHITECTURE_BINDING_COUNT,
        "frozen_cohort_execution_count": FROZEN_COHORT_EXECUTION_COUNT,
        "frozen_layout_generation_count": FROZEN_LAYOUT_GENERATION_COUNT,
        "learner_execution_count": LEARNER_EXECUTION_COUNT,
        "evaluator_execution_count": EVALUATOR_EXECUTION_COUNT,
        "outcome_count": OUTCOME_COUNT,
        "dev_input_count": 0,
        "m0_input_count": 0,
        "protected_input_count": 0,
        "preregistered_seed_execution_count": 0,
        "quarantined_source_validation_authorized": True,
        "frozen_cohort_generation_authorized": False,
        "nine_artifact_binding_required": True,
        "candidate_fields": adapter.CANDIDATE_FIELDS,
        "mission_discarded": True,
        "full_grid_candidate_exposure_count": 0,
        "position_candidate_exposure_count": 0,
        "object_identity_candidate_exposure_count": 0,
        "path_or_solver_candidate_exposure_count": 0,
        "codebook_inverse_candidate_exposure_count": 0,
        "layout_identity_candidate_exposure_count": 0,
        "cue_or_door_color_candidate_field_count": 0,
        "cue_or_door_color_in_mission": False,
        "cue_or_door_color_in_definition_id": False,
        "cue_or_door_color_in_audit_record": False,
        "rename_forward_mapping_candidate_exposure_count": 0,
        "rename_inverse_candidate_exposure_count": 0,
        "custom_layout_index_candidate_exposure_count": 0,
        "candidate_payload": "adapter-transcript-bytes-only",
    }
)


class G0CommonArchitectureBindingRequired(RuntimeError):
    """Frozen spatial cohort generation has no authenticated authority."""


class QuarantinedSpatialValidationError(ValueError):
    """A request was outside the exact source-validation fixtures."""


class _CueKeyDoorEnv(MiniGridEnv):
    """Private memory composition: cue, long delay, irreversible key choice."""

    WIDTH = 27
    HEIGHT = 9
    CUE_POSITION = (3, 4)
    NORTH_KEY_POSITION = (22, 3)
    SOUTH_KEY_POSITION = (22, 5)
    DOOR_POSITION = (24, 4)
    GOAL_POSITION = (25, 4)
    MINIMUM_CUE_GAP = 16

    def __init__(self, *, max_steps: int, layout_index: int) -> None:
        if type(layout_index) is not int or not 0 <= layout_index < CUSTOM_LAYOUT_COUNT:
            raise ValueError("custom cue-key-door layout index changed")
        self._layout_index = layout_index
        mission_space = MissionSpace(
            mission_func=lambda: "remember the cue, choose a key, and reach the goal"
        )
        super().__init__(
            mission_space=mission_space,
            width=self.WIDTH,
            height=self.HEIGHT,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=7,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        if (width, height) != (self.WIDTH, self.HEIGHT):
            raise RuntimeError("cue-key-door dimensions changed")
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.horz_wall(1, 3, 20)
        self.grid.horz_wall(1, 5, 20)
        self.grid.vert_wall(self.DOOR_POSITION[0], 1, height - 2)

        for x, y, marker_color in _custom_layout_marker_pattern(self._layout_index):
            self.grid.set(x, y, Floor(marker_color))

        cue_color = ("red", "blue")[self._rand_int(0, 2)]
        self.grid.set(*self.CUE_POSITION, Floor(cue_color))
        self.grid.set(*self.NORTH_KEY_POSITION, Key("red"))
        self.grid.set(*self.SOUTH_KEY_POSITION, Key("blue"))
        door = Door(cue_color, is_locked=True)
        self.grid.set(*self.DOOR_POSITION, door)
        self.grid.set(*self.GOAL_POSITION, Goal())

        self.agent_pos = (2, 4)
        self.agent_dir = 0
        self.mission = "remember the cue, choose a key, and reach the goal"

        self._cue_color = cue_color
        self._door = door
        self._key_positions = {
            "red": self.NORTH_KEY_POSITION,
            "blue": self.SOUTH_KEY_POSITION,
        }
        self._cue_removed_step: int | None = None
        self._choice_color: str | None = None
        self._choice_correct: bool | None = None
        self._cue_absent_actions_before_choice: int | None = None
        self._unchosen_key_removed = False

    def step(self, action: int):  # type: ignore[no-untyped-def]
        previous_carrying = self.carrying
        observation, reward, terminated, truncated, info = super().step(action)

        if self.step_count == 1:
            cue = self.grid.get(*self.CUE_POSITION)
            if not isinstance(cue, Floor) or cue.color != self._cue_color:
                raise RuntimeError("initial cue identity changed")
            self.grid.set(*self.CUE_POSITION, None)
            self._cue_removed_step = self.step_count
            observation = self.gen_obs()

        if (
            previous_carrying is None
            and isinstance(self.carrying, Key)
            and self._choice_color is None
        ):
            if self._cue_removed_step is None:
                raise RuntimeError("key choice occurred before cue removal")
            self._choice_color = self.carrying.color
            self._choice_correct = self._choice_color == self._cue_color
            self._cue_absent_actions_before_choice = self.step_count - self._cue_removed_step
            if self._cue_absent_actions_before_choice < self.MINIMUM_CUE_GAP:
                raise RuntimeError("cue-key-door delay invariant changed")
            other_color = "blue" if self._choice_color == "red" else "red"
            other_position = self._key_positions[other_color]
            if not isinstance(self.grid.get(*other_position), Key):
                raise RuntimeError("unchosen key identity changed")
            self.grid.set(*other_position, None)
            self._unchosen_key_removed = True
            observation = self.gen_obs()

        return observation, reward, terminated, truncated, info


def _apply_rename(image: np.ndarray, rename: RawTokenRename) -> np.ndarray:
    if image.shape[-1] != 3 or image.dtype != np.uint8:
        raise RuntimeError("MiniGrid observation encoding changed")
    result = np.array(image, dtype=np.uint8, copy=True, order="C")
    object_lut = np.asarray(rename.object_permutation, dtype=np.uint8)
    color_lut = np.asarray(rename.color_permutation, dtype=np.uint8)
    if int(result[..., 0].max()) >= len(object_lut) or int(result[..., 1].max()) >= len(color_lut):
        raise RuntimeError("MiniGrid token domain changed")
    result[..., 0] = object_lut[result[..., 0]]
    result[..., 1] = color_lut[result[..., 1]]
    return result


def _layout_identity(environment: MiniGridEnv, rename: RawTokenRename) -> str:
    encoded = _apply_rename(environment.grid.encode(), rename)
    agent_position = tuple(int(value) for value in environment.agent_pos)
    header = _canonical_json(
        {
            "width": environment.width,
            "height": environment.height,
            "agent_position": agent_position,
            "agent_direction": int(environment.agent_dir),
            "carrying": None,
        }
    )
    return sha256(
        _LAYOUT_NAMESPACE + len(header).to_bytes(8, "little") + header + encoded.tobytes(order="C")
    ).hexdigest()


def _candidate_step(
    observation: Mapping[str, object],
    action_id: int,
    reward: object,
    terminated: object,
    truncated: object,
    rename: RawTokenRename,
) -> adapter.SpatialStep:
    image = observation.get("image")
    direction = observation.get("direction")
    if (
        type(image) is not np.ndarray
        or image.shape != adapter.IMAGE_SHAPE
        or image.dtype != np.uint8
        or isinstance(direction, bool)
        or not isinstance(direction, (int, np.integer))
    ):
        raise RuntimeError("MiniGrid candidate observation contract changed")
    renamed = _apply_rename(image, rename)
    return adapter.SpatialStep(
        image=renamed.tobytes(order="C"),
        direction=int(direction),
        action_id=action_id,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


@dataclass(frozen=True, slots=True)
class QuarantinedEnvironmentValidation:
    definition_id: str
    definition_sha256: str
    fixture_seed: int
    rename_sha256: str
    candidate_transcript: bytes
    candidate_transcript_sha256: str
    layout_identity_sha256: str
    renamed_layout_identity_sha256: str
    candidate_field_audit_sha256: str
    raw_action_ids: tuple[int, ...]
    step_count: int
    total_reward: float
    terminated: bool
    truncated: bool
    validation_sha256: str

    def root_record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": GENERATOR_SCOPE,
            "definition_id": self.definition_id,
            "definition_sha256": self.definition_sha256,
            "fixture_seed": self.fixture_seed,
            "quarantined": True,
            "is_frozen_cohort": False,
            "rename_sha256": self.rename_sha256,
            "candidate_transcript_sha256": self.candidate_transcript_sha256,
            "layout_identity_sha256": self.layout_identity_sha256,
            "renamed_layout_identity_sha256": self.renamed_layout_identity_sha256,
            "candidate_field_audit_sha256": self.candidate_field_audit_sha256,
            "raw_action_ids": self.raw_action_ids,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "candidate_fields": adapter.CANDIDATE_FIELDS,
            "mission_discarded": True,
            "generator_only_layout_hashes_exposed_to_candidate": False,
            "frozen_cohort_generation_count": 0,
        }
        if include_sha256:
            record["validation_sha256"] = self.validation_sha256
        return record


def _make_environment(
    definition: SpatialEnvironmentDefinition,
    *,
    custom_layout_index: int = 0,
) -> MiniGridEnv:
    configuration = dict(definition.configuration)
    if definition.factory == "MemoryEnv":
        return MemoryEnv(**configuration)
    if definition.factory == "DoorKeyEnv":
        return DoorKeyEnv(**configuration)
    if definition.factory == "KeyCorridorEnv":
        return KeyCorridorEnv(**configuration)
    if definition.factory == "CueKeyDoorEnv":
        return _CueKeyDoorEnv(
            max_steps=definition.max_steps,
            layout_index=custom_layout_index,
        )
    raise RuntimeError("unknown frozen spatial environment factory")


def _rollout_validation(
    definition: SpatialEnvironmentDefinition,
    fixture_seed: int,
    rename: RawTokenRename,
    actions: tuple[int, ...],
    *,
    custom_layout_index: int = 0,
) -> tuple[QuarantinedEnvironmentValidation, MiniGridEnv, Mapping[str, object]]:
    environment = _make_environment(
        definition,
        custom_layout_index=custom_layout_index,
    )
    observation, _info = environment.reset(seed=fixture_seed)
    if not isinstance(observation, Mapping):
        raise RuntimeError("MiniGrid reset observation contract changed")
    layout_sha256 = _layout_identity(environment, IDENTITY_RENAME)
    renamed_layout_sha256 = _layout_identity(environment, rename)
    current_observation = observation
    steps: list[adapter.SpatialStep] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    for action_id in actions:
        if type(action_id) is not int or action_id not in adapter.RAW_ACTION_IDS:
            raise QuarantinedSpatialValidationError("STOP_G0_SPATIAL_VALIDATION_RAW_ACTION")
        following, reward, terminated, truncated, _info = environment.step(action_id)
        steps.append(
            _candidate_step(
                current_observation,
                action_id,
                reward,
                terminated,
                truncated,
                rename,
            )
        )
        total_reward += float(reward)
        current_observation = following
        if terminated or truncated:
            break
    if not steps:
        raise RuntimeError("quarantined spatial validation produced no steps")
    transcript = adapter.encode_transcript(steps)
    transcript_sha256 = adapter.transcript_sha256(transcript)
    field_audit = adapter.audit_candidate_record(steps[0].record())
    root_without_sha = {
        "scope": GENERATOR_SCOPE,
        "definition_id": definition.definition_id,
        "definition_sha256": definition.definition_sha256,
        "fixture_seed": fixture_seed,
        "quarantined": True,
        "is_frozen_cohort": False,
        "rename_sha256": rename.rename_sha256,
        "candidate_transcript_sha256": transcript_sha256,
        "layout_identity_sha256": layout_sha256,
        "renamed_layout_identity_sha256": renamed_layout_sha256,
        "candidate_field_audit_sha256": field_audit.sha256,
        "raw_action_ids": tuple(row.action_id for row in steps),
        "step_count": len(steps),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "candidate_fields": adapter.CANDIDATE_FIELDS,
        "mission_discarded": True,
        "generator_only_layout_hashes_exposed_to_candidate": False,
        "frozen_cohort_generation_count": 0,
    }
    return (
        QuarantinedEnvironmentValidation(
            definition_id=definition.definition_id,
            definition_sha256=definition.definition_sha256,
            fixture_seed=fixture_seed,
            rename_sha256=rename.rename_sha256,
            candidate_transcript=transcript,
            candidate_transcript_sha256=transcript_sha256,
            layout_identity_sha256=layout_sha256,
            renamed_layout_identity_sha256=renamed_layout_sha256,
            candidate_field_audit_sha256=field_audit.sha256,
            raw_action_ids=root_without_sha["raw_action_ids"],  # type: ignore[arg-type]
            step_count=len(steps),
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            validation_sha256=_digest(_VALIDATION_NAMESPACE, root_without_sha),
        ),
        environment,
        current_observation,
    )


def validate_quarantined_environment(
    definition_id: str,
    fixture_seed: int,
    *,
    rename_name: str = "identity",
) -> QuarantinedEnvironmentValidation:
    """Run one declared non-cohort source-validation transcript."""

    if type(definition_id) is not str or type(fixture_seed) is not int:
        raise QuarantinedSpatialValidationError("STOP_G0_SPATIAL_VALIDATION_NOT_DECLARED")
    expected_seed = QUARANTINED_VALIDATION_SEEDS.get(definition_id)
    definition = DEFINITION_BY_ID.get(definition_id)
    rename = RAW_TOKEN_RENAMES.get(rename_name)
    if (
        expected_seed is None
        or definition is None
        or rename is None
        or fixture_seed != expected_seed
        or fixture_seed in FROZEN_COHORT_SEEDS
    ):
        raise QuarantinedSpatialValidationError("STOP_G0_SPATIAL_VALIDATION_NOT_DECLARED")
    validation, _environment, _observation = _rollout_validation(
        definition,
        fixture_seed,
        rename,
        DEFAULT_VALIDATION_ACTIONS,
        custom_layout_index=(
            QUARANTINED_CUSTOM_LAYOUT_INDEX if definition.factory == "CueKeyDoorEnv" else 0
        ),
    )
    return validation


def _composition_actions(environment: _CueKeyDoorEnv, *, choose_matching: bool) -> tuple[int, ...]:
    chosen_color = environment._cue_color
    if not choose_matching:
        chosen_color = "blue" if chosen_color == "red" else "red"
    turn_to_key = int(
        environment.actions.left if chosen_color == "red" else environment.actions.right
    )
    turn_back = int(
        environment.actions.right if chosen_color == "red" else environment.actions.left
    )
    return (
        *(int(environment.actions.forward) for _index in range(20)),
        turn_to_key,
        int(environment.actions.pickup),
        turn_back,
        int(environment.actions.forward),
        int(environment.actions.toggle),
        int(environment.actions.forward),
        int(environment.actions.forward),
    )


@dataclass(frozen=True, slots=True)
class CueKeyDoorInvariantAudit:
    fixture_seed: int
    initial_cue_visible: bool
    cue_selects_one_of_two_keys: bool
    cue_removed_step: int
    irreversible_choice_step: int
    cue_absent_actions_before_choice: int
    minimum_cue_absence_required: int
    unchosen_key_removed: bool
    matching_key_unlocked_door: bool
    nonmatching_key_left_door_locked: bool
    locked_door_gates_goal: bool
    matching_rollout_terminated_at_goal: bool
    nonmatching_rollout_terminated: bool
    matching_candidate_transcript_sha256: str
    nonmatching_candidate_transcript_sha256: str
    candidate_field_audit_sha256: str
    audit_sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = dataclasses.asdict(self)
        if not include_sha256:
            record.pop("audit_sha256")
        return record


def validate_quarantined_cue_key_door(
    fixture_seed: int,
) -> CueKeyDoorInvariantAudit:
    """Validate the custom composition with no candidate-visible solver fields."""

    definition = DEFINITION_BY_ID["adjudicative-cue-key-door-composition"]
    expected_seed = QUARANTINED_VALIDATION_SEEDS[definition.definition_id]
    if (
        type(fixture_seed) is not int
        or fixture_seed != expected_seed
        or fixture_seed in FROZEN_COHORT_SEEDS
    ):
        raise QuarantinedSpatialValidationError("STOP_G0_SPATIAL_VALIDATION_NOT_DECLARED")

    probe = _make_environment(
        definition,
        custom_layout_index=QUARANTINED_CUSTOM_LAYOUT_INDEX,
    )
    if type(probe) is not _CueKeyDoorEnv:
        raise RuntimeError("cue-key-door factory changed")
    initial_observation, _info = probe.reset(seed=fixture_seed)
    cue_encoding = np.asarray(Floor(probe._cue_color).encode(), dtype=np.uint8)
    initial_image = initial_observation["image"]
    initial_cue_visible = bool(np.any(np.all(initial_image.reshape(-1, 3) == cue_encoding, axis=1)))
    matching_actions = _composition_actions(probe, choose_matching=True)
    matching, matching_environment, _matching_observation = _rollout_validation(
        definition,
        fixture_seed,
        IDENTITY_RENAME,
        matching_actions,
        custom_layout_index=QUARANTINED_CUSTOM_LAYOUT_INDEX,
    )
    if type(matching_environment) is not _CueKeyDoorEnv:
        raise RuntimeError("cue-key-door matching rollout changed")

    nonmatching_probe = _make_environment(
        definition,
        custom_layout_index=QUARANTINED_CUSTOM_LAYOUT_INDEX,
    )
    if type(nonmatching_probe) is not _CueKeyDoorEnv:
        raise RuntimeError("cue-key-door nonmatching factory changed")
    nonmatching_probe.reset(seed=fixture_seed)
    nonmatching_actions = _composition_actions(
        nonmatching_probe,
        choose_matching=False,
    )
    nonmatching, nonmatching_environment, _nonmatching_observation = _rollout_validation(
        definition,
        fixture_seed,
        IDENTITY_RENAME,
        nonmatching_actions,
        custom_layout_index=QUARANTINED_CUSTOM_LAYOUT_INDEX,
    )
    if type(nonmatching_environment) is not _CueKeyDoorEnv:
        raise RuntimeError("cue-key-door nonmatching rollout changed")

    gap = matching_environment._cue_absent_actions_before_choice
    cue_removed_step = matching_environment._cue_removed_step
    if gap is None or cue_removed_step is None:
        raise RuntimeError("cue-key-door choice audit was not reached")
    choice_step = cue_removed_step + gap
    candidate_field_audit = adapter.audit_candidate_record(
        adapter.decode_transcript(matching.candidate_transcript)[0].record()
    )
    audit_without_sha = {
        "fixture_seed": fixture_seed,
        "initial_cue_visible": initial_cue_visible,
        "cue_selects_one_of_two_keys": matching_environment._cue_color
        in matching_environment._key_positions,
        "cue_removed_step": cue_removed_step,
        "irreversible_choice_step": choice_step,
        "cue_absent_actions_before_choice": gap,
        "minimum_cue_absence_required": _CueKeyDoorEnv.MINIMUM_CUE_GAP,
        "unchosen_key_removed": matching_environment._unchosen_key_removed,
        "matching_key_unlocked_door": not matching_environment._door.is_locked,
        "nonmatching_key_left_door_locked": nonmatching_environment._door.is_locked,
        "locked_door_gates_goal": matching.terminated and not nonmatching.terminated,
        "matching_rollout_terminated_at_goal": matching.terminated and matching.total_reward > 0.0,
        "nonmatching_rollout_terminated": nonmatching.terminated,
        "matching_candidate_transcript_sha256": matching.candidate_transcript_sha256,
        "nonmatching_candidate_transcript_sha256": (nonmatching.candidate_transcript_sha256),
        "candidate_field_audit_sha256": candidate_field_audit.sha256,
    }
    return CueKeyDoorInvariantAudit(
        **audit_without_sha,
        audit_sha256=_digest(_COMPOSITION_NAMESPACE, audit_without_sha),
    )


def generate_frozen_spatial_cohort(
    *_args: object,
    **_kwargs: object,
) -> NoReturn:
    """Stop before inspecting a surface, seed, or purported binding."""

    raise G0CommonArchitectureBindingRequired(STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED)


def _authenticate_dependencies() -> None:
    import gymnasium

    if (
        minigrid.__version__ != MINIGRID_VERSION
        or importlib.metadata.version("minigrid") != MINIGRID_VERSION
        or gymnasium.__version__ != GYMNASIUM_VERSION
        or prereg.PROTOCOL.protocol_sha256 != EXPECTED_PREREGISTRATION_SHA256
        or prereg.PROTOCOL.spatial_sha256 != EXPECTED_SPATIAL_CONFIGURATION_SHA256
        or prereg.SPATIAL_SPECIFICATION["partial_observation_shape"] != adapter.IMAGE_SHAPE
        or prereg.SPATIAL_SPECIFICATION["discrete_action_count"] != len(adapter.RAW_ACTION_IDS)
        or tuple(FROZEN_COHORT_SEEDS) != (470_100, 470_101, 470_102, 470_103)
        or set(QUARANTINED_VALIDATION_SEEDS.values()) & set(FROZEN_COHORT_SEEDS)
        or fixture_preflight.GENERATOR_SCOPE != "G0_SPATIAL_FIXTURE_GENERATOR_V1"
        or len(fixture_preflight.REQUIRED_BINDING_ARTIFACTS) != 9
        or fixture_preflight.SURFACE_BOUNDARY_RECORD["frozen_cohort_generation_authorized"]
        is not False
        or adapter.IMAGE_SHAPE != (7, 7, 3)
        or adapter.CANDIDATE_FIELDS
        != ("image", "direction", "action_id", "reward", "terminated", "truncated")
        or tuple(range(7)) != adapter.RAW_ACTION_IDS
    ):
        raise RuntimeError("STOP_G0_SPATIAL_FROZEN_DEPENDENCY_MISMATCH")
    for relative_path, expected_sha256 in _PINNED_PROJECT_SOURCES.items():
        if sha256((_ROOT / relative_path).read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError("STOP_G0_SPATIAL_PROJECT_SOURCE_MISMATCH")
    for name, module in _MINIGRID_MODULES.items():
        path = Path(inspect.getfile(module))
        if sha256(path.read_bytes()).hexdigest() != MINIGRID_SOURCE_SHA256S[name]:
            raise RuntimeError("STOP_G0_SPATIAL_MINIGRID_SOURCE_MISMATCH")


_authenticate_dependencies()

SOURCE_CONFIGURATION_RECORD: Mapping[str, object] = MappingProxyType(
    {
        "scope": GENERATOR_SCOPE,
        "status": GENERATOR_STATUS,
        "definitions": tuple(row.record() for row in ALL_DEFINITIONS),
        "raw_token_renames": tuple(row.record() for row in RAW_TOKEN_RENAMES.values()),
        "custom_layout_family": CUSTOM_LAYOUT_FAMILY_RECORD,
        "custom_layout_family_sha256": CUSTOM_LAYOUT_FAMILY_SHA256,
        "frozen_cohort_seeds": FROZEN_COHORT_SEEDS,
        "quarantined_validation_seeds": tuple(QUARANTINED_VALIDATION_SEEDS.items()),
        "candidate_fields": adapter.CANDIDATE_FIELDS,
        "forbidden_candidate_fields": adapter.SPATIAL_FORBIDDEN_FIELDS,
        "custom_composition": {
            "initial_cue": "one-observation-colored-floor",
            "key_colors": ("red", "blue"),
            "first-key-pickup-removes-other-key": True,
            "matching-key-required-by-MiniGrid-Door.toggle": True,
            "locked-door-gates-goal": True,
            "minimum-cue-absent-actions": 16,
            "layout-count": CUSTOM_LAYOUT_COUNT,
            "layout-code-bits": CUSTOM_LAYOUT_CODE_BITS,
            "layout-markers-carry-cue-answer": False,
            "mission-carries-cue-or-door-color": False,
            "definition-id-carries-cue-or-door-color": False,
            "invariant-audit-carries-cue-or-door-color": False,
        },
        "frozen_cohort_generation_authorized": False,
    }
)
SOURCE_CONFIGURATION_SHA256 = _digest(
    _CONFIGURATION_NAMESPACE,
    SOURCE_CONFIGURATION_RECORD,
)
DEPENDENCY_RECORD: Mapping[str, object] = MappingProxyType(
    {
        "minigrid_version": MINIGRID_VERSION,
        "gymnasium_version": GYMNASIUM_VERSION,
        "minigrid_source_sha256s": tuple(MINIGRID_SOURCE_SHA256S.items()),
        "project_source_sha256s": tuple(_PINNED_PROJECT_SOURCES.items()),
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "spatial_configuration_sha256": EXPECTED_SPATIAL_CONFIGURATION_SHA256,
        "adapter_scope": adapter.ADAPTER_SCOPE,
        "fixture_preflight_scope": fixture_preflight.GENERATOR_SCOPE,
        "required_binding_artifact_count": len(fixture_preflight.REQUIRED_BINDING_ARTIFACTS),
    }
)
DEPENDENCY_SHA256 = _digest(_DEPENDENCY_NAMESPACE, DEPENDENCY_RECORD)
GENERATOR_ROOT_SHA256 = _digest(
    _ROOT_NAMESPACE,
    {
        "dependency_sha256": DEPENDENCY_SHA256,
        "source_configuration_sha256": SOURCE_CONFIGURATION_SHA256,
        "surface_boundary": dict(SURFACE_BOUNDARY_RECORD),
    },
)


__all__ = [
    "ADJUDICATION_DEFINITIONS",
    "ALL_DEFINITIONS",
    "ARCHITECTURE_BINDING_COUNT",
    "CUSTOM_LAYOUT_CODE_BITS",
    "CUSTOM_LAYOUT_COUNT",
    "CUSTOM_LAYOUT_FAMILY_RECORD",
    "CUSTOM_LAYOUT_FAMILY_SHA256",
    "CUSTOM_LAYOUT_MARKER_COLORS",
    "CUSTOM_LAYOUT_MARKER_PATTERNS",
    "CUSTOM_LAYOUT_MARKER_POSITIONS",
    "CYCLIC_TOKEN_RENAME",
    "DEFAULT_VALIDATION_ACTIONS",
    "DEFINITION_BY_ID",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "EVALUATOR_EXECUTION_COUNT",
    "EXPECTED_PREREGISTRATION_SHA256",
    "EXPECTED_SPATIAL_CONFIGURATION_SHA256",
    "FROZEN_COHORT_EXECUTION_COUNT",
    "FROZEN_COHORT_SEEDS",
    "FROZEN_COHORT_SURFACES",
    "FROZEN_LAYOUT_GENERATION_COUNT",
    "GENERATOR_ROOT_SHA256",
    "GENERATOR_SCOPE",
    "GENERATOR_STATUS",
    "GYMNASIUM_VERSION",
    "IDENTITY_RENAME",
    "LEARNER_EXECUTION_COUNT",
    "LEARNING_DEFINITIONS",
    "MINIGRID_SOURCE_SHA256S",
    "MINIGRID_VERSION",
    "OUTCOME_COUNT",
    "QUARANTINED_CUSTOM_LAYOUT_INDEX",
    "QUARANTINED_VALIDATION_SEEDS",
    "RAW_TOKEN_RENAMES",
    "SOURCE_CONFIGURATION_RECORD",
    "SOURCE_CONFIGURATION_SHA256",
    "STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED",
    "SURFACE_BOUNDARY_RECORD",
    "CueKeyDoorInvariantAudit",
    "G0CommonArchitectureBindingRequired",
    "QuarantinedEnvironmentValidation",
    "QuarantinedSpatialValidationError",
    "RawTokenRename",
    "SpatialEnvironmentDefinition",
    "generate_frozen_spatial_cohort",
    "validate_quarantined_cue_key_door",
    "validate_quarantined_environment",
]
