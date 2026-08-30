"""Frozen execution and evaluation source for the unspent G0 gate.

This module closes source-definition gaps only.  It does not import a domain
generator, construct a cohort, execute a learner, inspect an adjudicative seed,
or run the 50,000-resample terminal evaluator.  Candidate-visible computation
is limited to the two already-frozen raw adapters.
"""

from __future__ import annotations

import json
import math
import struct
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as prereg
from experiments.g0 import frozen_regulatory_adapter_v1 as regulatory_adapter
from experiments.g0 import spatial_candidate_adapter_v1 as spatial_adapter

SCOPE = "FROZEN_G0_EXECUTION_EVALUATION_PROTOCOL_V1"
STATUS = "SOURCE_FROZEN_G0_UNSPENT"
CLAIM_BOUNDARY = "execution-evaluation-source-only-no-cohort-learner-bootstrap-or-g0-outcome"

EXPECTED_PREREGISTRATION_SHA256 = "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"

GENERATOR_EXECUTION_COUNT = 0
COHORT_GENERATION_COUNT = 0
ADJUDICATIVE_SEED_EXECUTION_COUNT = 0
LEARNER_EXECUTION_COUNT = 0
EVALUATOR_EXECUTION_COUNT = 0
BOOTSTRAP_EXECUTION_COUNT = 0
OUTCOME_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_SOURCE_NAMESPACE = b"frozen-g0-execution-evaluation-source-v1\0"
_CONFIG_NAMESPACE = b"frozen-g0-execution-evaluation-config-v1\0"
_ROOT_NAMESPACE = b"frozen-g0-execution-evaluation-root-v1\0"
_SEED_NAMESPACE = b"frozen-g0-seed-derivation-v1\0"
_BASELINE_NAMESPACE = b"frozen-g0-simple-baseline-v1\0"
_BOOTSTRAP_NAMESPACE = b"frozen-g0-paired-bootstrap-v1\0"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _source_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class FrozenProtocolStop(RuntimeError):
    """A frozen execution/evaluation precondition was not satisfied."""


# ---------------------------------------------------------------------------
# Common physical output -> raw action framing (binding class 4)
# ---------------------------------------------------------------------------

ACTION_FRAME_MAGIC = b"G0A1"
ACTION_FRAME_VERSION = 1
REGULATORY_DOMAIN_TOKEN = 0
SPATIAL_DOMAIN_TOKEN = 1
_ACTION_HEADER = struct.Struct(">4sBBB")


@dataclass(frozen=True, slots=True)
class DecodedPhysicalAction:
    domain: str
    regulatory: regulatory_adapter.ActionEvent | None
    spatial_action_id: int | None


def _encode_action_payload(domain_token: int, payload: bytes) -> bytes:
    if not payload or len(payload) > 0xFF:
        raise FrozenProtocolStop("STOP_G0_ACTION_FRAME_PAYLOAD_LENGTH")
    return (
        _ACTION_HEADER.pack(
            ACTION_FRAME_MAGIC,
            ACTION_FRAME_VERSION,
            domain_token,
            len(payload),
        )
        + payload
    )


def encode_regulatory_action(action: regulatory_adapter.ActionEvent) -> bytes:
    """Reference encoder; the physical system must emit these exact bytes."""

    if type(action) is not regulatory_adapter.ActionEvent:
        raise FrozenProtocolStop("STOP_G0_REGULATORY_ACTION_TYPE")
    if action.action_id == regulatory_adapter.ACTION_NOOP:
        payload = bytes((regulatory_adapter.ACTION_NOOP,))
    else:
        if action.entity_token is None or action.value is None:
            raise FrozenProtocolStop("STOP_G0_REGULATORY_COMPOSITE_ACTION_OPERANDS")
        payload = struct.pack(">BIB", action.action_id, action.entity_token, action.value)
    return _encode_action_payload(REGULATORY_DOMAIN_TOKEN, payload)


def encode_spatial_action(action_id: int) -> bytes:
    """Reference encoder for one of the seven already-frozen raw action IDs."""

    if type(action_id) is not int or action_id not in spatial_adapter.RAW_ACTION_IDS:
        raise FrozenProtocolStop("STOP_G0_SPATIAL_RAW_ACTION_ID")
    return _encode_action_payload(SPATIAL_DOMAIN_TOKEN, bytes((action_id,)))


def decode_physical_action(output_bytes: bytes, *, expected_domain: str) -> DecodedPhysicalAction:
    """Decode an exact physical byte sequence without choosing missing operands.

    A current one-byte output is deliberately insufficient even for spatial
    actions: it does not authenticate the frame, version, or domain.  The host
    may never supply a missing entity, value, action, or domain default.
    """

    if type(output_bytes) is not bytes or len(output_bytes) < _ACTION_HEADER.size + 1:
        raise FrozenProtocolStop("STOP_G0_PHYSICAL_ACTION_SEQUENCE_INCOMPLETE")
    magic, version, domain_token, payload_size = _ACTION_HEADER.unpack_from(output_bytes)
    payload = output_bytes[_ACTION_HEADER.size :]
    if (
        magic != ACTION_FRAME_MAGIC
        or version != ACTION_FRAME_VERSION
        or len(payload) != payload_size
    ):
        raise FrozenProtocolStop("STOP_G0_PHYSICAL_ACTION_FRAME_INVALID")

    if domain_token == REGULATORY_DOMAIN_TOKEN:
        if expected_domain != "regulatory":
            raise FrozenProtocolStop("STOP_G0_PHYSICAL_ACTION_DOMAIN_MISMATCH")
        if payload == bytes((regulatory_adapter.ACTION_NOOP,)):
            action = regulatory_adapter.ActionEvent(regulatory_adapter.ACTION_NOOP)
        elif len(payload) == 6:
            action_id, entity_token, value = struct.unpack(">BIB", payload)
            action = regulatory_adapter.ActionEvent(action_id, entity_token, value)
        else:
            raise FrozenProtocolStop("STOP_G0_REGULATORY_COMPOSITE_ACTION_INCOMPLETE")
        return DecodedPhysicalAction("regulatory", action, None)

    if domain_token == SPATIAL_DOMAIN_TOKEN:
        if expected_domain != "spatial":
            raise FrozenProtocolStop("STOP_G0_PHYSICAL_ACTION_DOMAIN_MISMATCH")
        if len(payload) != 1 or payload[0] not in spatial_adapter.RAW_ACTION_IDS:
            raise FrozenProtocolStop("STOP_G0_SPATIAL_RAW_ACTION_INVALID")
        return DecodedPhysicalAction("spatial", None, payload[0])

    raise FrozenProtocolStop("STOP_G0_PHYSICAL_ACTION_DOMAIN_TOKEN")


# ---------------------------------------------------------------------------
# Deterministic seed tree (binding class 6)
# ---------------------------------------------------------------------------

_DOMAIN_PROTOCOLS = MappingProxyType({"regulatory": prereg.REGULATORY, "spatial": prereg.SPATIAL})


def _partition_seed(domain: str, partition: str) -> int:
    try:
        split = _DOMAIN_PROTOCOLS[domain].split
    except KeyError as error:
        raise FrozenProtocolStop("STOP_G0_SEED_DOMAIN") from error
    seeds = {
        "smoke": split.smoke_seed,
        "learning": split.learning_seed,
        "adjudicative": split.adjudicative_seed,
    }
    try:
        return seeds[partition]
    except KeyError as error:
        raise FrozenProtocolStop("STOP_G0_SEED_PARTITION") from error


def _derived_seed(kind: str, record: object) -> int:
    digest = sha256(
        _SEED_NAMESPACE + kind.encode("ascii") + b"\0" + _canonical_json(record)
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_system_seed(domain: str, partition: str, system_or_layout_index: int) -> int:
    if type(system_or_layout_index) is not int or system_or_layout_index < 0:
        raise FrozenProtocolStop("STOP_G0_SEED_UNIT_INDEX")
    protocol = _DOMAIN_PROTOCOLS.get(domain)
    if protocol is None:
        raise FrozenProtocolStop("STOP_G0_SEED_DOMAIN")
    return _derived_seed(
        "system-or-layout",
        {
            "protocol_sha256": prereg.PROTOCOL.protocol_sha256,
            "domain_sha256": protocol.sha256,
            "partition": partition,
            "generator_seed_root": _partition_seed(domain, partition),
            "system_or_layout_index": system_or_layout_index,
        },
    )


def _learner_seed_record(
    domain: str,
    partition: str,
    learner_seed: int,
    system_or_layout_index: int,
    *,
    renamed: bool,
) -> dict[str, object]:
    if learner_seed not in prereg.LEARNER_SEEDS:
        raise FrozenProtocolStop("STOP_G0_UNREGISTERED_LEARNER_SEED")
    protocol = _DOMAIN_PROTOCOLS.get(domain)
    if protocol is None:
        raise FrozenProtocolStop("STOP_G0_SEED_DOMAIN")
    return {
        "protocol_sha256": prereg.PROTOCOL.protocol_sha256,
        "domain_sha256": protocol.sha256,
        "partition": partition,
        "generator_seed_root": _partition_seed(domain, partition),
        "rename_seed_root": protocol.split.rename_seed if renamed else None,
        "renamed": renamed,
        "learner_seed_root": learner_seed,
        "system_seed": derive_system_seed(domain, partition, system_or_layout_index),
    }


def derive_query_seed(
    *,
    partition: str,
    learner_seed: int,
    system_index: int,
    query_index: int,
    renamed: bool = False,
) -> int:
    if type(query_index) is not int or query_index < 0:
        raise FrozenProtocolStop("STOP_G0_SEED_QUERY_INDEX")
    record = _learner_seed_record(
        "regulatory", partition, learner_seed, system_index, renamed=renamed
    )
    record["query_index"] = query_index
    return _derived_seed("regulatory-query", record)


def derive_episode_seed(
    *,
    partition: str,
    learner_seed: int,
    layout_index: int,
    episode_index: int,
    renamed: bool = False,
) -> int:
    if type(episode_index) is not int or episode_index < 0:
        raise FrozenProtocolStop("STOP_G0_SEED_EPISODE_INDEX")
    record = _learner_seed_record("spatial", partition, learner_seed, layout_index, renamed=renamed)
    record["episode_index"] = episode_index
    return _derived_seed("spatial-episode", record)


def derive_rollout_seed(
    *,
    learner_seed: int,
    layout_index: int,
    rollout_index: int,
    renamed: bool = False,
) -> int:
    if type(rollout_index) is not int or rollout_index < 0:
        raise FrozenProtocolStop("STOP_G0_SEED_ROLLOUT_INDEX")
    record = _learner_seed_record(
        "spatial", "adjudicative", learner_seed, layout_index, renamed=renamed
    )
    record["rollout_index"] = rollout_index
    return _derived_seed("spatial-rollout", record)


# ---------------------------------------------------------------------------
# Exact fixed simple baselines (binding class 5)
# ---------------------------------------------------------------------------

REGULATORY_BASELINES = prereg.REGULATORY.baselines
SPATIAL_BASELINES = prereg.SPATIAL.baselines
MISSING_BASELINE_STOPS: tuple[str, ...] = ()


def _seeded_choice(seed: int, label: str, choices: Sequence[int]) -> int:
    if not choices:
        raise FrozenProtocolStop("STOP_G0_BASELINE_EMPTY_CHOICE")
    index = _derived_seed(
        "baseline-choice",
        {"seed": seed, "label": label, "choices": tuple(choices)},
    ) % len(choices)
    return int(choices[index])


@dataclass(frozen=True, slots=True)
class RegulatorySupportExample:
    prefix: tuple[regulatory_adapter.RegulatoryEvent, ...]
    answer_bit: int

    def __post_init__(self) -> None:
        if self.answer_bit not in (0, 1):
            raise FrozenProtocolStop("STOP_G0_REGULATORY_SUPPORT_ANSWER")
        _regulatory_view(self.prefix)


@dataclass(frozen=True, slots=True)
class RegulatoryQuery:
    prefix: tuple[regulatory_adapter.RegulatoryEvent, ...]

    def __post_init__(self) -> None:
        _regulatory_view(self.prefix)


def _regulatory_view(
    events: tuple[regulatory_adapter.RegulatoryEvent, ...],
) -> tuple[int, dict[int, int]]:
    target: int | None = None
    state: dict[int, int] | None = None
    for event in events:
        if type(event) is regulatory_adapter.QueryEvent:
            target = event.target_token
        elif type(event) in (
            regulatory_adapter.ObservationEvent,
            regulatory_adapter.OutcomeEvent,
        ):
            state = dict(event.entity_states)
    if target is None or state is None or target not in state:
        raise FrozenProtocolStop("STOP_G0_REGULATORY_RAW_QUERY_PREFIX")
    return target, state


def _majority(bits: Iterable[int], *, seed: int, label: str) -> int:
    rows = tuple(bits)
    ones = sum(rows)
    zeros = len(rows) - ones
    if ones == zeros:
        return _seeded_choice(seed, label, (0, 1))
    return int(ones > zeros)


def _prefix_distance(left: bytes, right: bytes) -> int:
    shared = min(len(left), len(right))
    return abs(len(left) - len(right)) + sum(left[index] != right[index] for index in range(shared))


def predict_regulatory_baseline(
    name: str,
    support: Sequence[RegulatorySupportExample],
    query: RegulatoryQuery,
    *,
    seed: int,
) -> int:
    """Predict one binary answer from raw adapter events only."""

    if name not in REGULATORY_BASELINES:
        raise FrozenProtocolStop("STOP_G0_REGULATORY_BASELINE_NAME")
    target, query_state = _regulatory_view(query.prefix)
    if name == "balanced-coin":
        return _seeded_choice(seed, name, (0, 1))
    if name == "target-state-persistence":
        return query_state[target]

    matching = tuple(row for row in support if _regulatory_view(row.prefix)[0] == target)
    if name == "support-marginal-majority":
        if not matching:
            return _seeded_choice(seed, name, (0, 1))
        return _majority((row.answer_bit for row in matching), seed=seed, label=f"{name}:{target}")

    if name == "serialized-support-prefix-nearest-neighbor":
        if not support:
            return _seeded_choice(seed, name, (0, 1))
        query_bytes = regulatory_adapter.serialize_transcript(query.prefix)
        distances = tuple(
            _prefix_distance(query_bytes, regulatory_adapter.serialize_transcript(row.prefix))
            for row in support
        )
        nearest = min(distances)
        return _majority(
            (
                row.answer_bit
                for row, distance in zip(support, distances, strict=True)
                if distance == nearest
            ),
            seed=seed,
            label=name,
        )

    # Observational correlation deliberately ignores every ActionEvent.  It
    # sums raw signed feature/answer covariation and never receives an
    # intervention flag, graph edge, equation, or causal relation.
    if not matching:
        return _seeded_choice(seed, name, (0, 1))
    bias = sum(2 * row.answer_bit - 1 for row in matching)
    score = bias
    for entity_token, query_value in sorted(query_state.items()):
        covariance = 0
        for row in matching:
            _, state = _regulatory_view(row.prefix)
            covariance += (2 * state[entity_token] - 1) * (2 * row.answer_bit - 1)
        score += covariance * (2 * query_value - 1)
    if score == 0:
        return _seeded_choice(seed, name, (0, 1))
    return int(score > 0)


@dataclass(frozen=True, slots=True)
class SpatialObservation:
    image: bytes
    direction: int

    def __post_init__(self) -> None:
        if type(self.image) is not bytes or len(self.image) != spatial_adapter.IMAGE_BYTE_COUNT:
            raise FrozenProtocolStop("STOP_G0_SPATIAL_BASELINE_IMAGE")
        if self.direction not in spatial_adapter.DIRECTION_IDS:
            raise FrozenProtocolStop("STOP_G0_SPATIAL_BASELINE_DIRECTION")


@dataclass(frozen=True, slots=True)
class SpatialBaselinePolicy:
    name: str
    act: Callable[[SpatialObservation, int], int]
    raw_transition_count: int
    update_count: int


def _spatial_rows(
    support_episodes: Sequence[Sequence[spatial_adapter.SpatialStep]],
) -> tuple[tuple[SpatialObservation, int, float], ...]:
    rows: list[tuple[SpatialObservation, int, float]] = []
    for episode in support_episodes:
        future_return = 0.0
        reversed_rows: list[tuple[SpatialObservation, int, float]] = []
        for step in reversed(tuple(episode)):
            if type(step) is not spatial_adapter.SpatialStep:
                raise FrozenProtocolStop("STOP_G0_SPATIAL_BASELINE_RAW_STEP")
            future_return += step.reward
            reversed_rows.append(
                (SpatialObservation(step.image, step.direction), step.action_id, future_return)
            )
        rows.extend(reversed(reversed_rows))
    return tuple(rows)


def _observation_distance(left: SpatialObservation, right: SpatialObservation) -> int:
    return sum(a != b for a, b in zip(left.image, right.image, strict=True)) + (
        spatial_adapter.IMAGE_BYTE_COUNT * int(left.direction != right.direction)
    )


def _fixed_wall_follower(observation: SpatialObservation, _step: int) -> int:
    # Public MiniGrid raw assignments, not hidden environment state.  The
    # adapter image is agent-centric: front is (3,5), object channel 0.
    wall_object_token = 2
    turn_left_action = 0
    forward_action = 2
    front_offset = ((5 * 7) + 3) * 3
    return (
        turn_left_action if observation.image[front_offset] == wall_object_token else forward_action
    )


def _input_vector(observation: SpatialObservation) -> np.ndarray:
    pixels = np.frombuffer(observation.image, dtype=np.uint8).astype(np.float64) / 255.0
    direction = np.zeros(len(spatial_adapter.DIRECTION_IDS), dtype=np.float64)
    direction[observation.direction] = 1.0
    return np.concatenate((pixels, direction))


def fit_spatial_baseline(
    name: str,
    support_episodes: Sequence[Sequence[spatial_adapter.SpatialStep]],
    *,
    seed: int,
    transition_budget: int,
) -> SpatialBaselinePolicy:
    """Fit a named policy from raw partial observations/actions/outcomes only."""

    if name not in SPATIAL_BASELINES:
        raise FrozenProtocolStop("STOP_G0_SPATIAL_BASELINE_NAME")
    if type(transition_budget) is not int or transition_budget < 0:
        raise FrozenProtocolStop("STOP_G0_SPATIAL_BASELINE_TRANSITION_BUDGET")
    rows = _spatial_rows(support_episodes)
    if transition_budget > len(rows):
        raise FrozenProtocolStop("STOP_G0_SPATIAL_BASELINE_BUDGET_EXCEEDS_RAW_TRANSITIONS")

    if name == "uniform-random-action":

        def uniform(observation: SpatialObservation, step: int) -> int:
            del observation
            return _seeded_choice(seed, f"uniform:{step}", spatial_adapter.RAW_ACTION_IDS)

        return SpatialBaselinePolicy(name, uniform, len(rows), 0)

    if name == "fixed-wall-follower":
        return SpatialBaselinePolicy(name, _fixed_wall_follower, len(rows), 0)

    if name == "nearest-observation-action-memorizer":
        frozen_rows = rows[:transition_budget]

        def memorizer(observation: SpatialObservation, step: int) -> int:
            if not frozen_rows:
                return _seeded_choice(
                    seed, f"memorizer-empty:{step}", spatial_adapter.RAW_ACTION_IDS
                )
            distances = tuple(_observation_distance(observation, row[0]) for row in frozen_rows)
            nearest = min(distances)
            action_returns: dict[int, list[float]] = defaultdict(list)
            for row, distance in zip(frozen_rows, distances, strict=True):
                if distance == nearest:
                    action_returns[row[1]].append(row[2])
            means = {
                action: math.fsum(values) / len(values) for action, values in action_returns.items()
            }
            best = max(means.values())
            choices = tuple(sorted(action for action, value in means.items() if value == best))
            return _seeded_choice(seed, f"memorizer:{step}", choices)

        return SpatialBaselinePolicy(name, memorizer, len(rows), 0)

    # A fixed 32-hidden-unit memoryless MLP.  It receives only the current raw
    # image/direction and one successful-return behavior target per consumed
    # environment transition.  There is exactly one SGD opportunity per raw
    # transition and no replay, early stopping, privileged state, or oracle.
    rng = np.random.Generator(np.random.PCG64(seed))
    input_width = spatial_adapter.IMAGE_BYTE_COUNT + len(spatial_adapter.DIRECTION_IDS)
    hidden_width = 32
    output_width = len(spatial_adapter.RAW_ACTION_IDS)
    w1 = rng.normal(0.0, 0.02, size=(input_width, hidden_width))
    b1 = np.zeros(hidden_width, dtype=np.float64)
    w2 = rng.normal(0.0, 0.02, size=(hidden_width, output_width))
    b2 = np.zeros(output_width, dtype=np.float64)
    learning_rate = 0.01
    update_count = 0
    for observation, action, future_return in rows[:transition_budget]:
        x = _input_vector(observation)
        hidden = np.tanh(x @ w1 + b1)
        logits = hidden @ w2 + b2
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        weight = max(0.0, min(1.0, future_return))
        gradient_logits = probabilities
        gradient_logits[action] -= 1.0
        gradient_logits *= weight
        gradient_w2 = np.outer(hidden, gradient_logits)
        gradient_b2 = gradient_logits
        gradient_hidden = (w2 @ gradient_logits) * (1.0 - hidden * hidden)
        w2 -= learning_rate * gradient_w2
        b2 -= learning_rate * gradient_b2
        w1 -= learning_rate * np.outer(x, gradient_hidden)
        b1 -= learning_rate * gradient_hidden
        update_count += 1

    def feed_forward(observation: SpatialObservation, step: int) -> int:
        hidden = np.tanh(_input_vector(observation) @ w1 + b1)
        logits = hidden @ w2 + b2
        maximum = float(np.max(logits))
        choices = tuple(int(index) for index in np.flatnonzero(logits == maximum))
        return _seeded_choice(seed, f"feed-forward:{step}", choices)

    return SpatialBaselinePolicy(name, feed_forward, len(rows), update_count)


# ---------------------------------------------------------------------------
# Focused causal ablation wrappers (binding class 5)
# ---------------------------------------------------------------------------

ABLATION_LEARNING_OFF = "learning-off"
ABLATION_CELL_OUTPUT_OFF = "cell-or-output-population-off"
ABLATION_MEMORY_RESET = "working-and-persistent-memory-reset-each-observation"
ABLATION_DERANGEMENT = "fixed-representation-consequence-action-derangement"
ABLATION_NAMES = (
    ABLATION_LEARNING_OFF,
    ABLATION_CELL_OUTPUT_OFF,
    ABLATION_MEMORY_RESET,
    ABLATION_DERANGEMENT,
)
FIXED_DERANGEMENT = tuple((index + 1) % 256 for index in range(256))


class AblatableRuntime(Protocol):
    def set_learning_enabled(self, enabled: bool) -> None: ...

    def set_cell_and_output_populations_enabled(self, enabled: bool) -> None: ...

    def reset_working_and_persistent_memory(self) -> None: ...

    def set_representation_output_derangement(
        self, permutation: tuple[int, ...] | None
    ) -> None: ...


@dataclass(slots=True)
class FocusedAblationWrapper:
    runtime: AblatableRuntime
    name: str
    preparation_count: int = 0
    observation_reset_count: int = 0

    def prepare_from_same_checkpoint(self) -> None:
        if self.preparation_count:
            raise FrozenProtocolStop("STOP_G0_ABLATION_PREPARED_TWICE")
        self.runtime.set_learning_enabled(self.name != ABLATION_LEARNING_OFF)
        self.runtime.set_cell_and_output_populations_enabled(self.name != ABLATION_CELL_OUTPUT_OFF)
        self.runtime.set_representation_output_derangement(
            FIXED_DERANGEMENT if self.name == ABLATION_DERANGEMENT else None
        )
        self.preparation_count = 1

    def before_observation(self) -> None:
        if self.preparation_count != 1:
            raise FrozenProtocolStop("STOP_G0_ABLATION_NOT_PREPARED")
        if self.name == ABLATION_MEMORY_RESET:
            self.runtime.reset_working_and_persistent_memory()
            self.observation_reset_count += 1


def focused_ablation(runtime: AblatableRuntime, name: str) -> FocusedAblationWrapper:
    if name not in ABLATION_NAMES:
        raise FrozenProtocolStop("STOP_G0_ABLATION_NAME")
    return FocusedAblationWrapper(runtime, name)


# ---------------------------------------------------------------------------
# Exact evaluator, paired bootstrap, and terminal gate (binding class 8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    domain: str
    learner_seed: int
    unit_id: str
    item_id: str
    strata: tuple[str, ...]
    success: int

    def __post_init__(self) -> None:
        if self.domain not in _DOMAIN_PROTOCOLS:
            raise FrozenProtocolStop("STOP_G0_SCORE_DOMAIN")
        if self.learner_seed not in prereg.LEARNER_SEEDS:
            raise FrozenProtocolStop("STOP_G0_SCORE_LEARNER_SEED")
        if not self.unit_id or not self.item_id or self.success not in (0, 1):
            raise FrozenProtocolStop("STOP_G0_SCORE_RECORD")
        if len(set(self.strata)) != len(self.strata):
            raise FrozenProtocolStop("STOP_G0_SCORE_DUPLICATE_STRATUM")

    @property
    def key(self) -> tuple[str, int, str, str]:
        return self.domain, self.learner_seed, self.unit_id, self.item_id


@dataclass(frozen=True, slots=True)
class FrozenArmResults:
    name: str
    records: tuple[ScoreRecord, ...]
    architecture_root: str
    learning_mechanism_root: str
    adapter_audit_clean: bool
    transcript_hashes_clean: bool
    domain_specific_primitive_delta: int
    protected_input_count: int


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    point_estimate: float
    lower: float
    upper: float
    interquartile_mean: float
    performance_profile: tuple[tuple[float, float], ...]
    resamples: int


@dataclass(frozen=True, slots=True)
class MetricGate:
    metric: str
    estimate: BootstrapEstimate
    point_minimum: float
    lower_minimum: float
    passed: bool


@dataclass(frozen=True, slots=True)
class DomainGate:
    domain: str
    passed: bool
    metric_gates: tuple[MetricGate, ...]
    strongest_baseline: str
    strongest_baseline_score: float
    baseline_gap: BootstrapEstimate
    rename_accuracy_change: float
    rename_pair_agreement: float
    ablation_checks: tuple[tuple[str, bool, float], ...]
    stop_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CrossDomainGate:
    status: str
    passed: bool
    regulatory: DomainGate
    spatial: DomainGate
    stop_reasons: tuple[str, ...]
    evaluator_execution_count: int
    bootstrap_resamples: int


def _selected_records(records: Sequence[ScoreRecord], metric: str) -> tuple[ScoreRecord, ...]:
    overall = metric in ("all-heldout-queries", "all-heldout-layout-success")
    selected = tuple(row for row in records if overall or metric in row.strata)
    if not selected:
        raise FrozenProtocolStop(f"STOP_G0_EMPTY_METRIC:{metric}")
    return selected


def _cell_matrix(
    records: Sequence[ScoreRecord], metric: str
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    selected = _selected_records(records, metric)
    seeds = tuple(sorted({row.learner_seed for row in selected}))
    units = tuple(sorted({row.unit_id for row in selected}))
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    seen: set[tuple[str, int, str, str]] = set()
    for row in selected:
        if row.key in seen:
            raise FrozenProtocolStop("STOP_G0_DUPLICATE_SCORE_KEY")
        seen.add(row.key)
        grouped[(row.learner_seed, row.unit_id)].append(row.success)
    if any((seed, unit) not in grouped for seed in seeds for unit in units):
        raise FrozenProtocolStop("STOP_G0_NONRECTANGULAR_SEED_UNIT_MATRIX")
    matrix = np.asarray(
        [
            [math.fsum(grouped[(seed, unit)]) / len(grouped[(seed, unit)]) for unit in units]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    return matrix, seeds, units


def _paired_matrix(
    left: Sequence[ScoreRecord], right: Sequence[ScoreRecord], metric: str
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    left_rows = _selected_records(left, metric)
    right_rows = _selected_records(right, metric)
    left_map = {row.key: row for row in left_rows}
    right_map = {row.key: row for row in right_rows}
    if len(left_map) != len(left_rows) or len(right_map) != len(right_rows):
        raise FrozenProtocolStop("STOP_G0_DUPLICATE_PAIRED_SCORE_KEY")
    if set(left_map) != set(right_map):
        raise FrozenProtocolStop("STOP_G0_PAIRED_SCORE_KEYS")
    for key, row in left_map.items():
        other = right_map[key]
        if row.strata != other.strata:
            raise FrozenProtocolStop("STOP_G0_PAIRED_SCORE_STRATA")
    # ScoreRecord intentionally requires binary values; paired gaps are built
    # directly to keep that invariant on externally supplied outcome rows.
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for key, row in left_map.items():
        grouped[(row.learner_seed, row.unit_id)].append(row.success - right_map[key].success)
    seeds = tuple(sorted({key[0] for key in grouped}))
    units = tuple(sorted({key[1] for key in grouped}))
    if any((seed, unit) not in grouped for seed in seeds for unit in units):
        raise FrozenProtocolStop("STOP_G0_NONRECTANGULAR_PAIRED_MATRIX")
    matrix = np.asarray(
        [
            [math.fsum(grouped[(seed, unit)]) / len(grouped[(seed, unit)]) for unit in units]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    return matrix, seeds, units


def _interquartile_mean(values: np.ndarray) -> float:
    ordered = np.sort(values.reshape(-1))
    start = math.floor(0.25 * len(ordered))
    stop = math.ceil(0.75 * len(ordered))
    return float(np.mean(ordered[start:stop]))


def _performance_profile(values: np.ndarray) -> tuple[tuple[float, float], ...]:
    flattened = values.reshape(-1)
    return tuple(
        (threshold / 20.0, float(np.mean(flattened >= threshold / 20.0))) for threshold in range(21)
    )


def stratified_bootstrap(
    records: Sequence[ScoreRecord],
    metric: str,
    *,
    paired_records: Sequence[ScoreRecord] | None = None,
    resamples: int = prereg.BOOTSTRAP_RESAMPLES,
    confidence: float = prereg.BOOTSTRAP_CONFIDENCE,
) -> BootstrapEstimate:
    """Resample learner-seed and whole-system/layout axes, never timesteps.

    For paired gaps the two arms are first subtracted at identical item keys,
    then the same sampled seed and unit indices are applied to the differences.
    """

    if type(resamples) is not int or resamples <= 0 or not 0.0 < confidence < 1.0:
        raise FrozenProtocolStop("STOP_G0_BOOTSTRAP_CONFIGURATION")
    if paired_records is None:
        matrix, seeds, units = _cell_matrix(records, metric)
        paired_root = None
    else:
        matrix, seeds, units = _paired_matrix(records, paired_records, metric)
        paired_root = _digest(
            _BOOTSTRAP_NAMESPACE + b"paired-root\0",
            tuple(sorted(row.key for row in paired_records)),
        )
    seed = _derived_seed(
        "bootstrap",
        {
            "protocol_sha256": prereg.PROTOCOL.protocol_sha256,
            "metric": metric,
            "record_keys": tuple(sorted(row.key for row in records)),
            "paired_root": paired_root,
            "resamples": resamples,
            "confidence": confidence,
        },
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        seed_indices = generator.integers(0, len(seeds), size=len(seeds))
        unit_indices = generator.integers(0, len(units), size=len(units))
        estimates[index] = float(matrix[np.ix_(seed_indices, unit_indices)].mean())
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = min(resamples - 1, math.floor(tail * resamples))
    upper_index = min(resamples - 1, math.ceil((1.0 - tail) * resamples) - 1)
    return BootstrapEstimate(
        point_estimate=float(matrix.mean()),
        lower=float(estimates[lower_index]),
        upper=float(estimates[upper_index]),
        interquartile_mean=_interquartile_mean(matrix),
        performance_profile=_performance_profile(matrix),
        resamples=resamples,
    )


def _validate_arm(domain: str, arm: FrozenArmResults) -> None:
    if not arm.records or any(row.domain != domain for row in arm.records):
        raise FrozenProtocolStop("STOP_G0_ARM_DOMAIN_OR_EMPTY")
    if not arm.architecture_root or not arm.learning_mechanism_root:
        raise FrozenProtocolStop("STOP_G0_ARM_IDENTITY")


def _validate_terminal_roster(
    domain: str,
    candidate: FrozenArmResults,
    paired_arms: Sequence[FrozenArmResults],
) -> None:
    expected_unit_count = _DOMAIN_PROTOCOLS[domain].split.adjudicative_count
    expected_items_per_cell = (
        prereg.REGULATORY_SCHEDULE["queries_per_system"]
        if domain == "regulatory"
        else prereg.SPATIAL_SCHEDULE["adjudicative_rollouts_per_layout"]
    )
    expected_record_count = (
        len(prereg.LEARNER_SEEDS) * expected_unit_count * expected_items_per_cell
    )
    candidate_keys = {row.key for row in candidate.records}
    candidate_seeds = {row.learner_seed for row in candidate.records}
    candidate_units = {row.unit_id for row in candidate.records}
    if (
        len(candidate.records) != expected_record_count
        or len(candidate_keys) != expected_record_count
        or candidate_seeds != set(prereg.LEARNER_SEEDS)
        or len(candidate_units) != expected_unit_count
    ):
        raise FrozenProtocolStop(f"STOP_G0_{domain.upper()}_TERMINAL_ROSTER")
    by_cell_stratum: dict[tuple[int, str, str], int] = defaultdict(int)
    if domain == "regulatory":
        expected_strata = prereg.REGULATORY_SCHEDULE["query_strata_per_system"]
        allowed_strata = set(expected_strata)
        for row in candidate.records:
            if len(row.strata) != 1 or row.strata[0] not in allowed_strata:
                raise FrozenProtocolStop("STOP_G0_REGULATORY_STRATUM_ROSTER")
            by_cell_stratum[(row.learner_seed, row.unit_id, row.strata[0])] += 1
        if any(
            by_cell_stratum[(seed, unit, stratum)] != count
            for seed in prereg.LEARNER_SEEDS
            for unit in candidate_units
            for stratum, count in expected_strata.items()
        ):
            raise FrozenProtocolStop("STOP_G0_REGULATORY_STRATUM_COUNTS")
    else:
        allowed_strata = {row.metric for row in prereg.SPATIAL_THRESHOLDS[1:]}
        primary_strata = {
            "fresh-memory-layouts": 32,
            "fresh-key-door-layouts": 32,
            "heldout-compositions": 64,
        }
        unit_primary: dict[str, str] = {}
        for row in candidate.records:
            if not set(row.strata) <= allowed_strata:
                raise FrozenProtocolStop("STOP_G0_SPATIAL_STRATUM_ROSTER")
            primary = tuple(stratum for stratum in row.strata if stratum in primary_strata)
            if len(primary) != 1:
                raise FrozenProtocolStop("STOP_G0_SPATIAL_PRIMARY_STRATUM")
            previous = unit_primary.setdefault(row.unit_id, primary[0])
            if previous != primary[0]:
                raise FrozenProtocolStop("STOP_G0_SPATIAL_UNIT_STRATUM_CHANGED")
        if any(
            sum(value == stratum for value in unit_primary.values()) != count
            for stratum, count in primary_strata.items()
        ):
            raise FrozenProtocolStop("STOP_G0_SPATIAL_STRATUM_COUNTS")
    for arm in paired_arms:
        arm_map = {row.key: row for row in arm.records}
        if (
            len(arm.records) != expected_record_count
            or set(arm_map) != candidate_keys
            or any(arm_map[row.key].strata != row.strata for row in candidate.records)
        ):
            raise FrozenProtocolStop(f"STOP_G0_{domain.upper()}_PAIRED_ARM_ROSTER")


def _evaluate_domain(
    domain: str,
    candidate: FrozenArmResults,
    renamed: FrozenArmResults,
    baselines: Mapping[str, FrozenArmResults],
    ablations: Mapping[str, FrozenArmResults],
    *,
    resamples: int,
) -> DomainGate:
    _validate_arm(domain, candidate)
    _validate_arm(domain, renamed)
    protocol = _DOMAIN_PROTOCOLS[domain]
    expected_baselines = set(protocol.baselines)
    if set(baselines) != expected_baselines or set(ablations) != set(ABLATION_NAMES):
        raise FrozenProtocolStop("STOP_G0_BASELINE_OR_ABLATION_ROSTER")
    for arm in (*baselines.values(), *ablations.values()):
        _validate_arm(domain, arm)

    metric_gates: list[MetricGate] = []
    stop_reasons: list[str] = []
    for threshold in protocol.thresholds:
        estimate = stratified_bootstrap(candidate.records, threshold.metric, resamples=resamples)
        passed = (
            estimate.point_estimate >= threshold.point_estimate_minimum
            and estimate.lower >= threshold.lower_confidence_bound_minimum
        )
        metric_gates.append(
            MetricGate(
                threshold.metric,
                estimate,
                threshold.point_estimate_minimum,
                threshold.lower_confidence_bound_minimum,
                passed,
            )
        )
        if not passed:
            stop_reasons.append(f"STOP_G0_{domain.upper()}_METRIC:{threshold.metric}")

    overall_metric = protocol.thresholds[0].metric
    baseline_scores = {
        name: stratified_bootstrap(arm.records, overall_metric, resamples=resamples).point_estimate
        for name, arm in baselines.items()
    }
    strongest_score = max(baseline_scores.values())
    tied_strongest = tuple(
        name for name in sorted(baseline_scores) if baseline_scores[name] == strongest_score
    )
    tied_gaps = {
        name: stratified_bootstrap(
            candidate.records,
            overall_metric,
            paired_records=baselines[name].records,
            resamples=resamples,
        )
        for name in tied_strongest
    }
    strongest_name = min(
        tied_strongest,
        key=lambda name: (
            tied_gaps[name].lower,
            tied_gaps[name].point_estimate,
            name,
        ),
    )
    gap = tied_gaps[strongest_name]
    if gap.point_estimate < 0.10 or gap.lower < 0.05:
        stop_reasons.append(f"STOP_G0_{domain.upper()}_STRONGEST_BASELINE_GAP")

    candidate_map = {row.key: row for row in candidate.records}
    rename_map = {row.key: row for row in renamed.records}
    if set(candidate_map) != set(rename_map):
        raise FrozenProtocolStop("STOP_G0_RENAME_PAIRED_SCORE_KEYS")
    rename_accuracy = abs(
        math.fsum(row.success for row in candidate_map.values()) / len(candidate_map)
        - math.fsum(row.success for row in rename_map.values()) / len(rename_map)
    )
    rename_agreement = math.fsum(
        candidate_map[key].success == rename_map[key].success for key in candidate_map
    ) / len(candidate_map)
    rename_requirements = protocol.rename_requirements
    accuracy_limit = float(
        rename_requirements.get(
            "absolute-accuracy-change-maximum",
            rename_requirements.get("absolute-success-change-maximum"),
        )
    )
    agreement_minimum = float(
        rename_requirements.get(
            "mapped-query-agreement-minimum",
            rename_requirements.get("paired-success-failure-agreement-minimum"),
        )
    )
    if rename_accuracy > accuracy_limit or rename_agreement < agreement_minimum:
        stop_reasons.append(f"STOP_G0_{domain.upper()}_RENAME_INVARIANCE")

    candidate_overall = metric_gates[0].estimate.point_estimate
    ablation_checks: list[tuple[str, bool, float]] = []
    learning_arm = ablations[ABLATION_LEARNING_OFF]
    learning_score = stratified_bootstrap(
        learning_arm.records, overall_metric, resamples=resamples
    ).point_estimate
    learning_metric_passes = []
    for threshold in protocol.thresholds:
        estimate = stratified_bootstrap(learning_arm.records, threshold.metric, resamples=resamples)
        learning_metric_passes.append(
            estimate.point_estimate >= threshold.point_estimate_minimum
            and estimate.lower >= threshold.lower_confidence_bound_minimum
        )
    learning_drop = candidate_overall - learning_score
    learning_ok = learning_drop >= 0.15 and not all(learning_metric_passes)
    ablation_checks.append((ABLATION_LEARNING_OFF, learning_ok, learning_drop))

    cell_score = stratified_bootstrap(
        ablations[ABLATION_CELL_OUTPUT_OFF].records,
        overall_metric,
        resamples=resamples,
    ).point_estimate
    cell_margin = cell_score - baseline_scores[strongest_name]
    cell_ok = cell_margin <= 0.02
    ablation_checks.append((ABLATION_CELL_OUTPUT_OFF, cell_ok, cell_margin))

    delayed_metric = "delayed-memory" if domain == "regulatory" else "delayed-cue-subset"
    candidate_delayed = stratified_bootstrap(
        candidate.records, delayed_metric, resamples=resamples
    ).point_estimate
    reset_delayed = stratified_bootstrap(
        ablations[ABLATION_MEMORY_RESET].records,
        delayed_metric,
        resamples=resamples,
    ).point_estimate
    memory_drop = candidate_delayed - reset_delayed
    memory_ok = memory_drop >= 0.20
    ablation_checks.append((ABLATION_MEMORY_RESET, memory_ok, memory_drop))

    causal_metrics = (
        ("single-intervention", "paired-counterfactual", "relation-composition")
        if domain == "regulatory"
        else ("heldout-compositions", "multi-step-intervention-subset")
    )
    derangement_drops = tuple(
        stratified_bootstrap(candidate.records, metric, resamples=resamples).point_estimate
        - stratified_bootstrap(
            ablations[ABLATION_DERANGEMENT].records,
            metric,
            resamples=resamples,
        ).point_estimate
        for metric in causal_metrics
    )
    # The minimum prevents one intervention/composition slice from averaging
    # over a failed causal ablation in another slice.
    derangement_drop = min(derangement_drops)
    derangement_ok = derangement_drop >= 0.15
    ablation_checks.append((ABLATION_DERANGEMENT, derangement_ok, derangement_drop))
    for name, passed, _value in ablation_checks:
        if not passed:
            stop_reasons.append(f"STOP_G0_{domain.upper()}_ABLATION:{name}")

    return DomainGate(
        domain=domain,
        passed=not stop_reasons,
        metric_gates=tuple(metric_gates),
        strongest_baseline=strongest_name,
        strongest_baseline_score=baseline_scores[strongest_name],
        baseline_gap=gap,
        rename_accuracy_change=rename_accuracy,
        rename_pair_agreement=rename_agreement,
        ablation_checks=tuple(ablation_checks),
        stop_reasons=tuple(stop_reasons),
    )


def evaluate_cross_domain_g0(
    *,
    regulatory_candidate: FrozenArmResults,
    regulatory_renamed: FrozenArmResults,
    regulatory_baselines: Mapping[str, FrozenArmResults],
    regulatory_ablations: Mapping[str, FrozenArmResults],
    spatial_candidate: FrozenArmResults,
    spatial_renamed: FrozenArmResults,
    spatial_baselines: Mapping[str, FrozenArmResults],
    spatial_ablations: Mapping[str, FrozenArmResults],
    bootstrap_resamples: int = prereg.BOOTSTRAP_RESAMPLES,
) -> CrossDomainGate:
    """Run the terminal no-domain-averaging gate on already-recorded outcomes."""

    if bootstrap_resamples != prereg.BOOTSTRAP_RESAMPLES:
        raise FrozenProtocolStop("STOP_G0_TERMINAL_BOOTSTRAP_MUST_BE_50000")
    _validate_terminal_roster(
        "regulatory",
        regulatory_candidate,
        (
            regulatory_renamed,
            *regulatory_baselines.values(),
            *regulatory_ablations.values(),
        ),
    )
    _validate_terminal_roster(
        "spatial",
        spatial_candidate,
        (
            spatial_renamed,
            *spatial_baselines.values(),
            *spatial_ablations.values(),
        ),
    )
    arms = (
        regulatory_candidate,
        regulatory_renamed,
        *regulatory_ablations.values(),
        spatial_candidate,
        spatial_renamed,
        *spatial_ablations.values(),
    )
    architecture_roots = {arm.architecture_root for arm in arms}
    learning_roots = {arm.learning_mechanism_root for arm in arms}
    stop_reasons: list[str] = []
    if len(architecture_roots) != 1 or len(learning_roots) != 1:
        stop_reasons.append("STOP_G0_COMMON_ARCHITECTURE_IDENTITY")
    if any(
        arm.domain_specific_primitive_delta != 0
        or arm.protected_input_count != 0
        or not arm.adapter_audit_clean
        or not arm.transcript_hashes_clean
        for arm in arms
    ):
        stop_reasons.append("STOP_G0_SURFACE_OR_GENERALITY_BOUNDARY")

    regulatory = _evaluate_domain(
        "regulatory",
        regulatory_candidate,
        regulatory_renamed,
        regulatory_baselines,
        regulatory_ablations,
        resamples=bootstrap_resamples,
    )
    spatial = _evaluate_domain(
        "spatial",
        spatial_candidate,
        spatial_renamed,
        spatial_baselines,
        spatial_ablations,
        resamples=bootstrap_resamples,
    )
    stop_reasons.extend(regulatory.stop_reasons)
    stop_reasons.extend(spatial.stop_reasons)
    passed = regulatory.passed and spatial.passed and not stop_reasons
    return CrossDomainGate(
        status=(
            "PASS_FROZEN_ARCH_CROSS_DOMAIN_G0" if passed else "STOP_FROZEN_ARCH_CROSS_DOMAIN_G0"
        ),
        passed=passed,
        regulatory=regulatory,
        spatial=spatial,
        stop_reasons=tuple(stop_reasons),
        evaluator_execution_count=1,
        bootstrap_resamples=bootstrap_resamples,
    )


def synthetic_self_check() -> dict[str, object]:
    """Tiny non-adjudicative arithmetic check; never a G0 evaluator run."""

    rows = tuple(
        ScoreRecord(
            domain="regulatory",
            learner_seed=seed,
            unit_id=f"synthetic-{unit}",
            item_id="item-0",
            strata=("next-state-prediction",),
            success=int(unit == 0),
        )
        for seed in prereg.LEARNER_SEEDS[:2]
        for unit in range(2)
    )
    estimate = stratified_bootstrap(rows, "all-heldout-queries", resamples=32)
    return {
        "synthetic_record_count": len(rows),
        "point_estimate": estimate.point_estimate,
        "resamples": estimate.resamples,
        "adjudicative_seed_execution_count": 0,
        "evaluator_execution_count": 0,
        "bootstrap_execution_count": 0,
    }


def _authenticate_preregistration() -> None:
    if prereg.PROTOCOL.protocol_sha256 != EXPECTED_PREREGISTRATION_SHA256:
        raise FrozenProtocolStop("STOP_G0_PREREGISTRATION_ROOT_MISMATCH")
    if prereg.BOOTSTRAP_RESAMPLES != 50_000 or prereg.BOOTSTRAP_CONFIDENCE != 0.95:
        raise FrozenProtocolStop("STOP_G0_STATISTICS_CONFIG_MISMATCH")
    if tuple(prereg.LEARNER_SEEDS) != (17, 29, 43, 71, 101):
        raise FrozenProtocolStop("STOP_G0_LEARNER_SEED_ROOT_MISMATCH")
    if tuple(REGULATORY_BASELINES) != (
        "balanced-coin",
        "target-state-persistence",
        "support-marginal-majority",
        "serialized-support-prefix-nearest-neighbor",
        "observational-correlation-without-intervention-semantics",
    ) or tuple(SPATIAL_BASELINES) != (
        "uniform-random-action",
        "fixed-wall-follower",
        "nearest-observation-action-memorizer",
        "memoryless-feed-forward-policy-matched-transition-budget",
    ):
        raise FrozenProtocolStop("STOP_G0_SIMPLE_BASELINE_ROSTER_MISMATCH")


_authenticate_preregistration()

ACTION_ASSIGNMENT_RECORD = {
    "magic_hex": ACTION_FRAME_MAGIC.hex(),
    "version": ACTION_FRAME_VERSION,
    "domain_tokens": {
        "regulatory": REGULATORY_DOMAIN_TOKEN,
        "spatial": SPATIAL_DOMAIN_TOKEN,
    },
    "regulatory_payloads": (
        "noop:action-id",
        "clamp:action-id+uint32-entity-token+binary-value",
    ),
    "spatial_payload": "one-raw-action-id-from-0-through-6",
    "one_byte_physical_output_is_sufficient": False,
    "host_operand_or_action_default_count": 0,
}
SEED_DERIVATION_RECORD = {
    "namespace_hex": _SEED_NAMESPACE.hex(),
    "root_inputs": (
        "preregistration-protocol-sha256",
        "domain-protocol-sha256",
        "partition-generator-seed-root",
        "rename-seed-root-if-renamed",
        "learner-seed-root",
    ),
    "derived_kinds": (
        "system-or-layout",
        "regulatory-query",
        "spatial-episode",
        "spatial-rollout",
    ),
    "output": "first-64-big-endian-bits-of-namespaced-sha256",
}
BASELINE_RECORD = {
    "regulatory": REGULATORY_BASELINES,
    "spatial": SPATIAL_BASELINES,
    "missing_baseline_stops": MISSING_BASELINE_STOPS,
    "candidate_dependency_count": 0,
    "generator_dependency_count": 0,
    "privileged_state_field_count": 0,
}
ABLATION_RECORD = {
    "names": ABLATION_NAMES,
    "derangement_sha256": _digest(_BASELINE_NAMESPACE, FIXED_DERANGEMENT),
    "same_checkpoint_required": True,
}
EVALUATOR_RECORD = {
    "bootstrap_resamples": prereg.BOOTSTRAP_RESAMPLES,
    "confidence": prereg.BOOTSTRAP_CONFIDENCE,
    "bootstrap_axes": ("learner-seed", "whole-system-or-layout"),
    "paired_item_identity_preserved": True,
    "interval": "sorted-order-statistic-two-sided-95-percent",
    "iqm": "middle-50-percent-of-seed-unit-cell-scores",
    "performance_profile_thresholds": tuple(index / 20.0 for index in range(21)),
    "terminal_gate": prereg.TERMINAL_GATE_REQUIREMENTS,
    "failed_domain_or_stratum_rescue_by_aggregate": False,
}
ZERO_EXECUTION_RECORD = {
    "generator": GENERATOR_EXECUTION_COUNT,
    "cohort_generation": COHORT_GENERATION_COUNT,
    "adjudicative_seed": ADJUDICATIVE_SEED_EXECUTION_COUNT,
    "learner": LEARNER_EXECUTION_COUNT,
    "evaluator": EVALUATOR_EXECUTION_COUNT,
    "bootstrap": BOOTSTRAP_EXECUTION_COUNT,
    "outcome": OUTCOME_COUNT,
    "dev": DEV_INPUT_COUNT,
    "m0": M0_INPUT_COUNT,
    "protected": PROTECTED_INPUT_COUNT,
}
CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "preregistration_sha256": prereg.PROTOCOL.protocol_sha256,
    "regulatory_adapter_sha256": regulatory_adapter.ADAPTER_SHA256,
    "spatial_adapter_source_sha256": _source_sha256(Path(spatial_adapter.__file__)),
    "action_assignment": ACTION_ASSIGNMENT_RECORD,
    "seed_derivation": SEED_DERIVATION_RECORD,
    "baselines": BASELINE_RECORD,
    "ablations": ABLATION_RECORD,
    "evaluator": EVALUATOR_RECORD,
    "domain_specific_computational_primitive_delta": (
        DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA
    ),
    "zero_execution": ZERO_EXECUTION_RECORD,
}
SOURCE_SHA256 = _source_sha256(_SOURCE_PATH)
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)
PROTOCOL_ROOT_SHA256 = _digest(
    _ROOT_NAMESPACE,
    {
        "source_sha256": SOURCE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "preregistration_sha256": prereg.PROTOCOL.protocol_sha256,
    },
)


__all__ = [
    "ABLATION_CELL_OUTPUT_OFF",
    "ABLATION_DERANGEMENT",
    "ABLATION_LEARNING_OFF",
    "ABLATION_MEMORY_RESET",
    "ABLATION_NAMES",
    "ACTION_ASSIGNMENT_RECORD",
    "BASELINE_RECORD",
    "BOOTSTRAP_EXECUTION_COUNT",
    "CLAIM_BOUNDARY",
    "COHORT_GENERATION_COUNT",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "EVALUATOR_EXECUTION_COUNT",
    "EVALUATOR_RECORD",
    "FIXED_DERANGEMENT",
    "GENERATOR_EXECUTION_COUNT",
    "LEARNER_EXECUTION_COUNT",
    "MISSING_BASELINE_STOPS",
    "OUTCOME_COUNT",
    "PROTOCOL_ROOT_SHA256",
    "REGULATORY_BASELINES",
    "SCOPE",
    "SEED_DERIVATION_RECORD",
    "SOURCE_SHA256",
    "SPATIAL_BASELINES",
    "STATUS",
    "ZERO_EXECUTION_RECORD",
    "CrossDomainGate",
    "DecodedPhysicalAction",
    "DomainGate",
    "FocusedAblationWrapper",
    "FrozenArmResults",
    "FrozenProtocolStop",
    "MetricGate",
    "RegulatoryQuery",
    "RegulatorySupportExample",
    "ScoreRecord",
    "SpatialBaselinePolicy",
    "SpatialObservation",
    "decode_physical_action",
    "derive_episode_seed",
    "derive_query_seed",
    "derive_rollout_seed",
    "derive_system_seed",
    "encode_regulatory_action",
    "encode_spatial_action",
    "evaluate_cross_domain_g0",
    "fit_spatial_baseline",
    "focused_ablation",
    "predict_regulatory_baseline",
    "stratified_bootstrap",
    "synthetic_self_check",
]
