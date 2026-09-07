"""Input-defined rank-one recognition beside a frozen factorized router.

No byte, task label or desired owner enters this module. The owning reader binds
the prepared index to its input-only assembly circuit before any current target.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from humanbrain.factorized_sequence_features import (
    FEATURE_WIDTH,
    LEXICAL_WIDTH,
    CandidateView,
    FactorizedContext,
    _half_normalize,
    activity,
    check_candidate_view,
    validate_candidate_view,
    validate_context,
)
from humanbrain.factorized_sequence_router import (
    _finite,
    _tensor,
    compete_float64,
    compete_float64_1024,
)
from humanbrain.neuronal_address_competition import AddressCompetition

CAPACITY = 1024
WORK_KEYS = (
    "recognition_self_builds", "recognition_self_cast_values",
    "recognition_self_products", "recognition_self_tensor_bytes",
    "recognition_self_validations", "recognition_validation_values",
    "recognition_custody_bytes_hashed", "recognition_evaluations",
    "recognition_candidates", "recognition_projection_products",
    "recognition_projection_reductions", "recognition_gain_products",
    "recognition_score_additions", "recognition_prepared_contexts",
    "recognition_offers", "recognition_skipped_offers", "recognition_local_errors",
    "recognition_local_divisions", "recognition_local_additions",
    "recognition_staged_updates", "recognition_stage_copy_bytes",
    "recognition_publications", "recognition_publication_copy_bytes",
    "recognition_named_allocated_bytes",
)


def _add(work, name, value):
    if work is not None:
        work[name] = work.get(name, 0) + value


def _hash(*values, work=None):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, Tensor):
            raw = value.contiguous().numpy().tobytes()
            digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
        else:
            raw = repr(value).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        _add(work, "recognition_custody_bytes_hashed", len(raw))
    return digest.hexdigest()


def _view_hash(view, work=None):
    return _hash(view.indices, view.values, view.flags, view.position,
                 view.raw_context_sha256, work=work)


@dataclass(frozen=True)
class SelfActivity:
    indices: Tensor
    values: Tensor
    context_sha256: str
    view_sha256: str
    seal: str


def _self_seal(wiring, work=None):
    return _hash(wiring.indices, wiring.values, wiring.context_sha256,
                 wiring.view_sha256, work=work)


def _self_from_validated_view(view, work=None):
    """Derive wiring immediately after an owner's fresh canonical view build.

    The synchronous caller owns the authoritative pattern and must not let the
    view change between construction and this call. No raw pattern is replayed.
    Public build/installation APIs retain independent canonical validation.
    """
    count = view.indices.numel()
    values64 = view.values.to(torch.float64)
    lexical = values64 * values64
    _half_normalize(lexical, work)
    indices = torch.cat((view.indices, torch.arange(
        LEXICAL_WIDTH, FEATURE_WIDTH, dtype=torch.int32, device="cpu")))
    values = torch.cat((lexical, torch.zeros(14, dtype=torch.float64, device="cpu")))
    wiring = SelfActivity(indices, values, view.raw_context_sha256,
                          _view_hash(view, work), "")
    object.__setattr__(wiring, "seal", _self_seal(wiring, work))
    _add(work, "recognition_self_builds", 1)
    _add(work, "recognition_self_cast_values", count)
    _add(work, "recognition_self_products", count)
    _add(work, "recognition_self_tensor_bytes", 12 * (count + 14))
    # Explicit casts/products/tail/concatenations; normalization scratch is
    # separately counted by the reused fixed-population helper.
    _add(work, "recognition_named_allocated_bytes", 28 * count + 336)
    return wiring


@torch.no_grad()
def build_self_activity(view: CandidateView, pattern: Tensor, work=None) -> SelfActivity:
    """Build fixed wiring only from an authenticated canonical receptive view."""
    validate_candidate_view(view, pattern, work=work)
    return _self_from_validated_view(view, work)


def check_self_activity(wiring: SelfActivity, view: CandidateView, work=None):
    """Runtime seal check, without replaying raw feature construction."""
    check_candidate_view(view, work)
    if not isinstance(wiring, SelfActivity):
        raise ValueError("recognition wiring must be SelfActivity")
    count = view.indices.numel() + 14
    _tensor(wiring.indices, (count,), torch.int32, "recognition columns")
    _tensor(wiring.values, (count,), torch.float64, "recognition self activity")
    _finite(wiring.values, "recognition self activity")
    if (wiring.context_sha256 != view.raw_context_sha256
            or wiring.view_sha256 != _view_hash(view, work)
            or wiring.seal != _self_seal(wiring, work)):
        raise ValueError("recognition self activity or its canonical view changed")
    _add(work, "recognition_self_validations", 1)
    _add(work, "recognition_validation_values", 2 * count)


@torch.no_grad()
def validate_self_activity(wiring, view, pattern, work=None):
    """Installation/load validation includes authoritative raw receptive input."""
    check_self_activity(wiring, view, work)
    expected = build_self_activity(view, pattern, work)
    if (not torch.equal(wiring.indices, expected.indices)
            or not torch.equal(wiring.values, expected.values)):
        raise ValueError("recognition wiring is not its exact canonical self activity")


@dataclass
class RecognitionStage:
    gains: Tensor
    source_sha256: str
    revision: int = 0
    _owner: object = field(default=None, repr=False)


@dataclass(frozen=True)
class _OwnedRecognition:
    gains: Tensor
    gain_version: int
    gains_sha256: str
    theta: Tensor
    theta_version: int
    theta_sha256: str
    owner: object


@dataclass(frozen=True)
class RecognitionDecision:
    scores: Tensor
    prior_scores: Tensor
    kappas: Tensor
    selected_index: int | None
    selected_score: float
    selected_baseline: float
    selected_kappa: float
    energy: float
    competition: AddressCompetition | None
    context_sha256: str


@dataclass(frozen=True)
class RecognitionContext:
    selected_index: int
    context_sha256: str
    prior_score: float
    kappa: float
    gain_before: float
    score_before: float
    stage_revision: int
    stage_sha256: str
    theta_sha256: str
    _theta: Tensor = field(repr=False)
    _stage: RecognitionStage = field(repr=False)
    _owner: object = field(repr=False)
    _seal: str = field(repr=False)


@dataclass(frozen=True)
class RecognitionUpdate:
    selected_index: int
    context_sha256: str
    kappa: float
    prior_score: float
    score_before: float
    score_after: float
    gain_before: float
    gain_after: float
    error: float
    delta_gain: float
    staged: bool
    skipped: bool


class InputDefinedRecognition(nn.Module):
    """One private gain per event; theta is borrowed, never owned or updated."""

    def __init__(self, *, capacity=CAPACITY):
        super().__init__()
        if type(capacity) is not int or capacity not in (512, 1024):
            raise ValueError("recognition capacity must be512 or1024")
        self.capacity = capacity
        self.register_buffer("gains", torch.zeros(capacity, dtype=torch.float64, device="cpu"))
        self._owner = object()

    @property
    def tensor_bytes(self):
        return self.capacity * 8

    def get_extra_state(self):
        return {"schema": "input-defined-recognition-v1", "capacity": self.capacity,
                "feature_width": FEATURE_WIDTH, "dtype": "float64",
                "target": "input-activated current one before current byte",
                "prior_theta": "held", "learning": "rank-one exact-context local current"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("recognition metadata mismatch")

    def _check_gains(self, gains, work=None):
        _tensor(gains, (self.capacity,), torch.float64, "recognition gains")
        _finite(gains, "recognition gains")
        _add(work, "recognition_validation_values", self.capacity)

    def check_parameters(self, work=None):
        self._check_gains(self.gains, work)

    def validate_state(self, state: Mapping, prefix="", work=None):
        self.check_parameters(work)
        if not isinstance(state, Mapping):
            raise ValueError("recognition state must be a mapping")
        self._check_gains(state[prefix + "gains"], work)
        if state[prefix + "_extra_state"] != self.get_extra_state():
            raise ValueError("recognition state metadata mismatch")

    def load_state_dict(self, state_dict: Mapping, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError("recognition loading requires strict copy semantics")
        if not isinstance(state_dict, Mapping) or set(state_dict) != {"gains", "_extra_state"}:
            raise ValueError("recognition state keys mismatch")
        self.validate_state(state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def snapshot(self):
        self.check_parameters()
        return {"gains": self.gains.clone(), "_extra_state": self.get_extra_state()}

    def begin_stage(self, work=None):
        self.check_parameters(work)
        _add(work, "recognition_stage_copy_bytes", self.tensor_bytes)
        return RecognitionStage(
            self.gains.clone(), _hash(self.gains, work=work), _owner=self._owner)

    def validate_stage(self, stage, work=None):
        self.check_parameters(work)
        if (not isinstance(stage, RecognitionStage) or stage._owner is not self._owner
                or type(stage.revision) is not int or stage.revision < 0):
            raise ValueError("recognition stage owner/revision mismatch")
        self._check_gains(stage.gains, work)
        if stage.gains.untyped_storage().data_ptr() == self.gains.untyped_storage().data_ptr():
            raise ValueError("recognition stage aliases live gains")
        if stage.source_sha256 != _hash(self.gains, work=work):
            raise ValueError("live recognition gains changed after staging")

    @torch.no_grad()
    def commit_validated(self, stage):
        """Copy-only, immediately after the owner prevalidates all joint stages."""
        self.gains.copy_(stage.gains)
        stage._owner = None

    def _check_theta(self, theta, work=None):
        _tensor(theta, (FEATURE_WIDTH,), torch.float64, "held prior theta")
        _finite(theta, "held prior theta")
        _add(work, "recognition_validation_values", FEATURE_WIDTH)

    def _begin_validated_free_response(self, theta, work=None):
        """Borrow already validated state; reader seals all wiring at both ends."""
        return _OwnedRecognition(self.gains, self.gains._version, _hash(self.gains, work=work),
                                 theta, theta._version, _hash(theta, work=work), self._owner)

    def _check_owned_free_response(self, owned):
        if (not isinstance(owned, _OwnedRecognition) or owned.owner is not self._owner
                or owned.gains is not self.gains or owned.gain_version != self.gains._version
                or owned.theta_version != owned.theta._version):
            raise RuntimeError("owned recognition gains or theta changed")

    def _score(self, current, view, wiring, theta, gain, work, *, checked_wiring=False):
        if not checked_wiring:
            check_self_activity(wiring, view, work)
        row = activity(current, view, work=work)
        if not torch.equal(row.indices, wiring.indices):
            raise ValueError("recognition activity columns differ from self wiring")
        indices = row.indices.to(torch.int64)
        prior = float((theta.index_select(0, indices) * row.values).sum().item()) + row.baseline
        kappa = float((wiring.values * row.values).sum().item())
        score = prior + gain * kappa
        if not all(math.isfinite(value) for value in (prior, kappa, score)):
            raise ValueError("recognition current must be finite")
        if kappa < 0 or kappa > 0.5 + 1e-12:
            raise ValueError("recognition projection exceeds self half-energy bound")
        count = indices.numel()
        for key, value in {
            "router_index_cast_values": count, "router_weight_products": count,
            "router_score_reductions": count - 1, "router_baseline_additions": 1,
            "recognition_projection_products": count,
            "recognition_projection_reductions": count - 1,
            "recognition_gain_products": 1, "recognition_score_additions": 1,
            "recognition_named_allocated_bytes": 32 * count,
        }.items():
            _add(work, key, value)
        return prior, kappa, score, row

    @torch.no_grad()
    def evaluate(self, current, views, wirings, theta, *, stage=None, work=None):
        validate_context(current, work=work)
        self._check_theta(theta, work)
        if stage is None:
            self.check_parameters(work)
            gains = self.gains
        else:
            self.validate_stage(stage, work)
            gains = stage.gains
        return self._evaluate_validated(current, views, wirings, theta, gains, work, False)

    @torch.no_grad()
    def _evaluate_owned_free(self, current, views, wirings, owned, *, work=None):
        self._check_owned_free_response(owned)
        return self._evaluate_validated(
            current, views, wirings, owned.theta, owned.gains, work, True)

    def _evaluate_validated(self, current, views, wirings, theta, gains, work, owned):
        if (not isinstance(views, (tuple, list)) or not isinstance(wirings, (tuple, list))
                or len(views) != len(wirings) or len(views) > self.capacity):
            raise ValueError("recognition needs every ordered owner and its self wiring")
        count = len(views)
        scores = torch.empty(count, dtype=torch.float64)
        priors, kappas = torch.empty_like(scores), torch.empty_like(scores)
        for index, (view, wiring) in enumerate(zip(views, wirings, strict=True)):
            prior, kappa, score, _ = self._score(
                current, view, wiring, theta, float(gains[index].item()), work,
                checked_wiring=owned)
            priors[index], kappas[index], scores[index] = prior, kappa, score
        compete = compete_float64 if self.capacity == 512 else compete_float64_1024
        competition = compete(scores) if count else None
        selected = competition.winner_index if competition is not None else None
        selected_row = activity(current, views[selected], work=work) if count else None
        if competition is not None:
            for key, value in competition.work.items():
                _add(work, "router_" + key, value)
        _add(work, "recognition_evaluations", 1)
        _add(work, "recognition_candidates", count)
        _add(work, "recognition_named_allocated_bytes", 24 * count)
        return RecognitionDecision(
            scores, priors, kappas, selected, float(scores[selected].item()) if count else 0.0,
            selected_row.baseline if count else 0.0,
            float(kappas[selected].item()) if count else 0.0,
            selected_row.energy if count else 0.0, competition, current.raw_context_sha256)

    def _context_seal(self, prepared, work=None):
        return _hash(prepared.selected_index, prepared.context_sha256, prepared.prior_score,
                     prepared.kappa, prepared.gain_before, prepared.score_before,
                     prepared.stage_revision, prepared.stage_sha256, prepared.theta_sha256,
                     work=work)

    @torch.no_grad()
    def prepare_context(self, current: FactorizedContext, view, wiring, index, theta, *,
                        stage, work=None):
        validate_context(current, work=work)
        self.validate_stage(stage, work)
        self._check_theta(theta, work)
        if type(index) is not int or not 0 <= index < self.capacity:
            raise ValueError("recognition prepared owner index outside capacity")
        if current.raw_context_sha256 != view.raw_context_sha256:
            raise ValueError("recognition requires the input-defined owner's exact context")
        gain = float(stage.gains[index].item())
        prior, kappa, score, row = self._score(current, view, wiring, theta, gain, work)
        if not torch.equal(row.values, wiring.values):
            raise ValueError("recognition offered context is not the actual self activity")
        prepared = RecognitionContext(
            index, current.raw_context_sha256, prior, kappa, gain, score,
            stage.revision, _hash(stage.gains, work=work), _hash(theta, work=work),
            theta, stage, self._owner, "")
        object.__setattr__(prepared, "_seal", self._context_seal(prepared, work))
        _add(work, "recognition_prepared_contexts", 1)
        return prepared

    def validate_context(self, prepared, *, stage, work=None):
        self.validate_stage(stage, work)
        if (not isinstance(prepared, RecognitionContext) or prepared._owner is not self._owner
                or prepared._stage is not stage):
            raise ValueError("recognition context owner or stage mismatch")
        self._check_theta(prepared._theta, work)
        if (prepared.stage_revision != stage.revision
                or prepared.stage_sha256 != _hash(stage.gains, work=work)
                or prepared.theta_sha256 != _hash(prepared._theta, work=work)
                or prepared._seal != self._context_seal(prepared, work)):
            raise ValueError("recognition context, gains or prior changed after preparation")

    @torch.no_grad()
    def prepare_recognition(self, prepared, *, stage, learning_enabled=True, work=None):
        self.validate_context(prepared, stage=stage, work=work)
        if type(learning_enabled) is not bool:
            raise ValueError("recognition enable must be Boolean")
        _add(work, "recognition_offers", 1)
        error = 1.0 - prepared.score_before
        skipped = prepared.kappa == 0
        delta = 0.0 if skipped else error / prepared.kappa
        gain_after = prepared.gain_before + delta
        score_after = prepared.prior_score + gain_after * prepared.kappa
        _add(work, "recognition_local_errors", 1)
        _add(work, "recognition_gain_products", 1)
        _add(work, "recognition_score_additions", 1)
        if skipped:
            _add(work, "recognition_skipped_offers", 1)
        else:
            _add(work, "recognition_local_divisions", 1)
            _add(work, "recognition_local_additions", 1)
        if not all(math.isfinite(value) for value in (error, delta, gain_after, score_after)):
            raise ValueError("recognition local proposal must be finite")
        result = RecognitionUpdate(
            prepared.selected_index, prepared.context_sha256, prepared.kappa,
            prepared.prior_score, prepared.score_before, score_after,
            prepared.gain_before, gain_after, error, delta,
            learning_enabled and not skipped, skipped)
        if result.staged:
            stage.gains[prepared.selected_index] = gain_after
            stage.revision += 1
            _add(work, "recognition_staged_updates", 1)
        return result
