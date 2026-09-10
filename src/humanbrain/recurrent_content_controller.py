"""Sparse stochastic cells and causal observed-symbol credit for one acting reader.

The caller owns causal input framing, h/cursor relays and actual execution.
This module owns all random choices, their direct afferent eligibility and the
atomic controller publication. No downstream or recurrent derivative is taken.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

INPUT_WIDTH = 69954
EMISSION_WIDTH = 1072
POPULATIONS = ("context", "gate", "mode", "literal")
SHAPES = {
    "context": (32, INPUT_WIDTH),
    "gate": (8, INPUT_WIDTH),
    "mode": (4, EMISSION_WIDTH),
    "literal": (257, EMISSION_WIDTH),
}
CATEGORIES = ("scan", "free", "teacher")
WEIGHT_NAMES = tuple(name + "_weight" for name in POPULATIONS)
TENSOR_NAMES = (*WEIGHT_NAMES, "baseline_mean", "baseline_count", "rng_state")
WEIGHT_BYTES = 24623616
EMISSION_PHASE_START = 811
EVENT_NORM_FACTORS = {"scan": math.sqrt(71), "teacher": math.sqrt(74)}
UPDATE_NORM_TOLERANCE = 1e-10
WORK_KEYS = tuple(
    "controller_" + name
    for name in (
        "parameter_validation_values",
        "sparse_validation_values",
        "sparse_norm_products",
        "sparse_norm_reductions",
        "identity_checks",
        "custody_bytes_hashed",
        "stages",
        "forks",
        "eligibility_zero_values",
        "stage_copy_bytes",
        "rng_generators",
        "rng_state_bytes",
        "rng_uniform_draws",
        "score_gathers",
        "weight_products",
        "score_reductions",
        "softmax_max_comparisons",
        "softmax_subtractions",
        "softmax_exponentials",
        "softmax_sum_additions",
        "softmax_divisions",
        "cdf_additions",
        "cdf_comparisons",
        "cdf_endpoint_closures",
        "draws",
        "eligibility_errors",
        "eligibility_products",
        "eligibility_gathers",
        "eligibility_additions",
        "eligibility_copy_values",
        "causal_zero_values",
        "causal_products",
        "causal_gathers",
        "causal_additions",
        "causal_copy_values",
        "credit_validation_values",
        "literal_pending_copy_bytes",
        "literal_teaching_offers",
        "literal_teaching_errors",
        "literal_teaching_products",
        "literal_teaching_gathers",
        "literal_teaching_additions",
        "literal_teaching_copy_values",
        "literal_logarithms",
        "literal_log_probability_subtractions",
        "literal_loss_negations",
        "conditional_reward_products",
        "conditional_reward_subtractions",
        "brier_norm_products",
        "brier_norm_reductions",
        "brier_norm_retained_bytes",
        "mode_pending_copy_bytes",
        "mode_association_products",
        "mode_association_values",
        "prefix_credit_divisions",
        "phase_geometry_values",
        "phase_contacts",
        "phase_pre_additions",
        "phase_post_products",
        "phase_post_additions",
        "phase_alpha_additions",
        "phase_center_divisions",
        "phase_center_products",
        "phase_center_subtractions",
        "phase_restore_additions",
        "signal_additions",
        "baseline_subtractions",
        "update_divisions",
        "update_subtractions",
        "ticks",
        "offers",
        "outcome_comparisons",
        "invalid_register_emissions",
        "absent_prior_emissions",
        "finish_calls",
        "category_terms",
        "update_products",
        "update_additions",
        "update_norm_products",
        "update_norm_reductions",
        "proposed_weight_bytes",
        "baseline_updates",
        "proposal_validations",
        "publications",
        "weight_publications",
        "rng_publications",
        "publication_copy_bytes",
        "record_copy_bytes",
        "named_allocated_bytes",
    )
)


def _add(work, name, count):
    if work is not None:
        key = "controller_" + name
        work[key] = work.get(key, 0) + count


def _tensor(value, shape, dtype, name):
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device.type != "cpu"
        or value.requires_grad
        or value.grad_fn is not None
        or value.grad is not None
        or not value.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous detached CPU {dtype} {shape}")


def _finite(value, name):
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


def _phase_alpha(indices, values, phase, work=None):
    """Validate the existing frame geometry used by the scan-covariance proof."""
    phase_mask = (indices >= EMISSION_PHASE_START) & (indices < EMISSION_PHASE_START + 3)
    if (not 6 <= indices.numel() <= 9 or int(phase_mask.sum()) != 1
            or int(indices[phase_mask][0]) != EMISSION_PHASE_START + phase):
        raise ValueError("observed emission needs its existing source/public/output phase")
    alpha = values[phase_mask][0]
    if float(alpha) <= 0 or not torch.equal(values, alpha.expand_as(values)):
        raise ValueError("observed emission needs equal positive normalized binary contacts")
    _add(work, "phase_geometry_values", 3 * indices.numel())
    _add(work, "named_allocated_bytes", 3 * indices.numel() + 16)
    return float(alpha)


def _digest(values, work=None):
    digest = hashlib.sha256(b"humanbrain.recurrent-controller.v1\0")
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        if isinstance(value, Tensor):
            digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
            raw = value.detach().contiguous().numpy().tobytes()
        else:
            raw = repr(value).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        _add(work, "custody_bytes_hashed", len(raw))
    return digest.hexdigest()


def _identity(value):
    return id(value), value.data_ptr(), value._version, value.shape, value.dtype, value.device


@dataclass(frozen=True)
class SparseActivity:
    indices: Tensor
    values: Tensor
    width: int


def validate_activity(activity, width, work=None):
    if not isinstance(activity, SparseActivity) or type(activity.width) is not int:
        raise ValueError("activity must be an explicit SparseActivity")
    if (
        activity.width != width
        or not isinstance(activity.indices, Tensor)
        or activity.indices.ndim != 1
    ):
        raise ValueError("activity width/rank mismatch")
    count = activity.indices.numel()
    if not 1 <= count <= width:
        raise ValueError("activity must have one or more unique active afferents")
    _tensor(activity.indices, (count,), torch.int64, "activity indices")
    _tensor(activity.values, (count,), torch.float64, "activity values")
    _finite(activity.values, "activity values")
    if (
        int(activity.indices[0]) < 0
        or int(activity.indices[-1]) >= width
        or (count > 1 and not bool((activity.indices[1:] > activity.indices[:-1]).all()))
        or bool((activity.values == 0).any())
    ):
        raise ValueError("sparse afferents must be sorted, unique, nonzero and in bounds")
    energy = float((activity.values * activity.values).sum().item())
    if not math.isclose(energy, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("activity must have unit energy")
    _add(work, "sparse_validation_values", 2 * count)
    _add(work, "sparse_norm_products", count)
    _add(work, "sparse_norm_reductions", count - 1)


@dataclass(frozen=True)
class ControllerDraw:
    population: str
    category: str
    index: int
    logits: Tensor
    probabilities: Tensor
    uniform: float
    activity_sha256: str
    rng_before_sha256: str
    rng_after_sha256: str
    ordinal: int

    def record(self, work=None):
        _add(work, "record_copy_bytes", 16 * self.logits.numel())
        return {
            "schema": "recurrent-content-draw-v1",
            "population": self.population,
            "category": self.category,
            "index": self.index,
            "logits": self.logits.clone(),
            "probabilities": self.probabilities.clone(),
            "uniform": self.uniform,
            "activity_sha256": self.activity_sha256,
            "rng_before_sha256": self.rng_before_sha256,
            "rng_after_sha256": self.rng_after_sha256,
            "ordinal": self.ordinal,
        }


@dataclass(eq=False)
class ControllerStage:
    """Private synchronous transaction. Mutating its fields invalidates ownership."""

    kind: str
    scan_offers: int
    max_bytes: int
    source_sha256: str
    _root: object = field(repr=False)
    _token: object = field(default_factory=object, repr=False)
    _generator: object = field(default=None, repr=False)
    _eligibility: dict = field(default_factory=dict, repr=False)
    _causal_prefix: dict = field(default_factory=dict, repr=False)
    _literal_direct: dict = field(default_factory=dict, repr=False)
    _mode_direct: dict = field(default_factory=dict, repr=False)
    _scan_moments: dict = field(default_factory=dict, repr=False)
    _scan_contacts: dict = field(default_factory=lambda: {"mode": 0, "literal": 0})
    _mode_pending: dict | None = field(default=None, repr=False)
    _mode_sha256: str | None = None
    _literal_pending: dict | None = field(default=None, repr=False)
    _literal_sha256: str | None = None
    _pending_mode: int | None = None
    _baselines: tuple = ()
    _signal_sums: dict = field(default_factory=lambda: {"scan": 0.0, "teacher": 0.0})
    _causal_totals: dict = field(default_factory=lambda: {"scan": 0.0, "teacher": 0.0})
    _prefix_totals: dict = field(default_factory=lambda: {"scan": 0.0, "teacher": 0.0})
    _literal_offers: dict = field(default_factory=lambda: {"scan": 0, "teacher": 0})
    _last_observation: tuple = ()
    _phase: str = "scan"
    _teacher_offers: int | None = None
    _next: str = "context"
    _choices: dict = field(default_factory=dict, repr=False)
    _pending_token: int | None = None
    _counts: dict = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    _reward_sums: dict = field(default_factory=lambda: {"scan": 0, "teacher": 0})
    _ticks: dict = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    _draws: int = 0
    _free_ended: bool = False
    _finished: bool = False
    _expected: tuple = field(default=(), repr=False)
    _tensor_ids: dict = field(default_factory=dict, repr=False)
    _rng_sha256: str = ""


@dataclass(frozen=True)
class ControllerProposal:
    state: Mapping
    source_sha256: str
    state_sha256: str
    learning_enabled: bool
    rewards: Mapping
    category_count: int
    normalization_ticks: int
    delta_norm: float
    draws: int
    ticks: Mapping
    _stage: ControllerStage = field(repr=False, compare=False)
    _stage_sha256: str = field(repr=False, compare=False)
    _seal: str = field(repr=False, compare=False)

    def record(self):
        return {
            "schema": "recurrent-content-update-v3",
            "local_rule": "integrated-local-learning-revision-v1",
            "source_sha256": self.source_sha256,
            "state_sha256": self.state_sha256,
            "stage_sha256": self._stage_sha256,
            "learning_enabled": self.learning_enabled,
            "rewards": dict(self.rewards),
            "category_count": self.category_count,
            "normalization_ticks": self.normalization_ticks,
            "delta_norm": self.delta_norm,
            "delta_norm_bound": self.rewards["norm_bound"],
            "draws": self.draws,
            "ticks": dict(self.ticks),
            "rng_published": True,
        }


class RecurrentContentController(nn.Module):
    """Held-weight local control; live RNG advances only at validated commit."""

    TENSOR_NAMES = TENSOR_NAMES

    def __init__(self, *, seed=733):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("controller seed must be an integer in[0,2**63)")
        self.seed = seed
        for name, shape in SHAPES.items():
            self.register_buffer(name + "_weight", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("baseline_mean", torch.zeros(3, dtype=torch.float64))
        self.register_buffer("baseline_count", torch.zeros(3, dtype=torch.int64))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("rng_state", generator.get_state())
        self._rng_shape = tuple(self.rng_state.shape)
        self._root = None
        self._root_stage = None
        self._issued = {}
        self._source_ids = {}
        self._revision = 0
        self._validated = None

    def get_extra_state(self):
        return {
            "schema": "recurrent-content-controller-v3",
            "seed": self.seed,
            "shapes": dict(SHAPES),
            "rng": "torch.Generator/cpu",
            "torch_version": str(torch.__version__),
            "rng_shape": self._rng_shape,
            "sampling": "f64-softmax-ascending-cdf-final1-strict-greater",
            "local_rule": "integrated-local-learning-revision-v1",
            "upstream_objective": "selected-emitter-Brier; context/gate-causal-prefix-means",
            "mode_objective": "immediate-own-score-Brier; scan-phase-covariance",
            "literal_objective": "actual-contact-CE; scan-phase-covariance; uncentered-teacher",
            "scan_association": "alpha-weighted-own-contact-covariance; phase-only-mean",
            "public_consequence": "unchanged-actual-scan-and-free-eligibility-over-fixed-L",
            "update_norm_bound": "eta*(sqrt71*S+sqrt74*T+abs(r)*sqrt8*(S+F)/(K*L))",
            "update_norm_tolerance": UPDATE_NORM_TOLERANCE,
            "learning_rate": 0.01,
            "free_normalization_ticks": 129,
            "baseline_order": ("source", "observation", "teacher"),
            "baseline_history": "retained numerical priors; subsequent means use Brier utility",
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("controller metadata differs")

    @property
    def pending_stage(self):
        return self._root_stage

    @property
    def tensor_bytes(self):
        return sum(
            getattr(self, key).numel() * getattr(self, key).element_size() for key in TENSOR_NAMES
        )

    def _state(self):
        return {
            **{key: getattr(self, key) for key in TENSOR_NAMES},
            "_extra_state": self.get_extra_state(),
        }

    def validate_state(self, state, prefix="", work=None):
        if not isinstance(state, Mapping):
            raise ValueError("controller state must be a mapping")
        expected = {*TENSOR_NAMES, "_extra_state"}
        relevant = {key[len(prefix) :] for key in state if key.startswith(prefix)}
        if relevant != expected:
            raise ValueError("controller strict state keys differ")
        if state[prefix + "_extra_state"] != self.get_extra_state():
            raise ValueError("controller strict metadata differs")
        for name, shape in SHAPES.items():
            value = state[prefix + name + "_weight"]
            _tensor(value, shape, torch.float64, name)
            _finite(value, name)
            _add(work, "parameter_validation_values", value.numel())
        means, counts = state[prefix + "baseline_mean"], state[prefix + "baseline_count"]
        _tensor(means, (3,), torch.float64, "baseline means")
        _tensor(counts, (3,), torch.int64, "baseline counts")
        _finite(means, "baseline means")
        if bool((counts < 0).any()) or bool((means.abs() > 1).any()):
            raise ValueError("controller baseline outside its range")
        if bool(((counts == 0) & (means != 0)).any()):
            raise ValueError("unobserved baseline must be zero")
        rng = state[prefix + "rng_state"]
        _tensor(rng, self._rng_shape, torch.uint8, "RNG state")
        torch.Generator(device="cpu").set_state(rng)
        _add(work, "rng_generators", 1)
        _add(work, "rng_state_bytes", rng.numel())
        _add(work, "parameter_validation_values", 6 + rng.numel())

    def check_parameters(self, work=None):
        self.validate_state(self._state(), work=work)

    def invalidate_stages_after_load(self):
        self._revision += 1
        self._root = None
        self._root_stage = None
        self._issued.clear()
        self._source_ids.clear()
        self._validated = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or assign or self._root is not None:
            raise ValueError("controller load requires strict, copy-only, idle state")
        self.validate_state(state_dict)
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self.invalidate_stages_after_load()
        return result

    @staticmethod
    def _status(stage):
        return (
            stage.kind,
            stage.scan_offers,
            stage.max_bytes,
            stage.source_sha256,
            stage._phase,
            stage._teacher_offers,
            stage._next,
            tuple(sorted(stage._choices.items())),
            stage._pending_token,
            stage._pending_mode,
            stage._literal_sha256,
            stage._mode_sha256,
            stage._baselines,
            tuple(stage._signal_sums.items()),
            tuple(stage._causal_totals.items()),
            tuple(stage._prefix_totals.items()),
            tuple(stage._scan_contacts.items()),
            tuple(stage._literal_offers.items()),
            stage._last_observation,
            tuple(stage._counts.items()),
            tuple(stage._reward_sums.items()),
            tuple(stage._ticks.items()),
            stage._draws,
            stage._free_ended,
            stage._finished,
            stage._rng_sha256,
        )

    @staticmethod
    def _private_tensors(stage):
        values = {
            f"eligibility/{c}/{n}": v
            for c, bank in stage._eligibility.items() for n, v in bank.items()
        }
        values.update({f"causal_prefix/{c}/{n}": v
                       for c, bank in stage._causal_prefix.items() for n, v in bank.items()})
        values.update({f"literal_direct/{c}": v for c, v in stage._literal_direct.items()})
        values.update({f"mode_direct/{c}": v for c, v in stage._mode_direct.items()})
        values.update({f"scan_moments/{n}/{k}": v
                       for n, bank in stage._scan_moments.items() for k, v in bank.items()})
        if stage._mode_pending is not None:
            values.update({f"mode_pending/{n}": v for n, v in stage._mode_pending.items()})
        if stage._literal_pending is not None:
            values.update({f"literal_pending/{n}": v for n, v in stage._literal_pending.items()})
        return values

    def _remember(self, stage):
        stage._expected = self._status(stage)
        stage._tensor_ids = {
            key: _identity(value) for key, value in self._private_tensors(stage).items()
        }

    def _guard(self, stage, work=None):
        if (
            not isinstance(stage, ControllerStage)
            or self._root is None
            or stage._root is not self._root
            or self._issued.get(stage._token, lambda: None)() is not stage
            or stage._expected != self._status(stage)
        ):
            raise ValueError("stale, foreign or modified controller stage")
        actual = {key: _identity(getattr(self, key)) for key in TENSOR_NAMES}
        eligible = {
            key: _identity(value) for key, value in self._private_tensors(stage).items()
        }
        if actual != self._source_ids or eligible != stage._tensor_ids:
            raise ValueError("held weights or private eligibility changed outside controller")
        _add(work, "identity_checks", len(actual) + len(eligible))

    def _rng(self, stage, work=None):
        value = stage._generator.get_state()
        _add(work, "rng_state_bytes", value.numel())
        digest = _digest({"rng": value}, work)
        if digest != stage._rng_sha256:
            raise ValueError("private RNG changed outside controller")
        return value, digest

    def check_stage(self, stage, work=None):
        """Check synchronous ownership without rescanning dense banks per tick."""
        self._guard(stage, work)
        self._rng(stage, work)

    def stage_digest(self, stage, work=None):
        """Seal a pending boundary, including mutations through NumPy aliases.

        The caller retains this digest and compares it before resolving the
        actual response. Dense eligibility is read here, never per cell draw.
        """
        self._guard(stage, work)
        rng, _ = self._rng(stage, work)
        values = self._private_tensors(stage)
        values.update(
            {
                "status": self._status(stage),
                "rng": rng,
                "live_state_sha256": _digest(self._state(), work),
            }
        )
        return _digest(values, work)

    def discard_stage(self, stage):
        """Abandon an owned failed branch without publishing RNG or learning.

        This deliberately accepts a damaged private stage: a failure may occur
        after a draw but before its bookkeeping is sealed. Foreign stages are
        still rejected. Discarding a fork leaves its sealed actual root intact.
        """
        if (
            not isinstance(stage, ControllerStage)
            or self._root is None
            or self._issued.get(stage._token, lambda: None)() is not stage
        ):
            raise ValueError("cannot discard a stale or foreign controller stage")
        if stage is self._root_stage:
            self.invalidate_stages_after_load()
        else:
            del self._issued[stage._token]
            if self._validated is not None and self._validated._stage is stage:
                self._validated = None

    @torch.no_grad()
    def begin_stage(self, *, kind, scan_offers, max_bytes=128, work=None):
        if (
            self._root is not None
            or kind not in ("source", "reading", "edit")
            or type(scan_offers) is not int
            or not 1 <= scan_offers <= 3485
            or type(max_bytes) is not int
            or not 0 <= max_bytes <= 128
        ):
            raise ValueError("begin needs an idle controller and explicit external bounds")
        self.check_parameters(work)
        root = object()
        stage = ControllerStage(kind, scan_offers, max_bytes, _digest(self._state(), work), root)
        stage._generator = torch.Generator(device="cpu").set_state(self.rng_state)
        stage._rng_sha256 = _digest({"rng": self.rng_state}, work)
        stage._eligibility = {
            category: {
                name: torch.zeros(shape, dtype=torch.float64) for name, shape in SHAPES.items()
            }
            for category in CATEGORIES
        }
        stage._causal_prefix = {
            c: {n: torch.zeros(SHAPES[n], dtype=torch.float64) for n in ("context", "gate")}
            for c in ("scan", "teacher")
        }
        stage._literal_direct = {
            c: torch.zeros(SHAPES["literal"], dtype=torch.float64) for c in ("scan", "teacher")
        }
        stage._mode_direct = {
            c: torch.zeros(SHAPES["mode"], dtype=torch.float64) for c in ("scan", "teacher")
        }
        stage._scan_moments = {
            name: {"pre": torch.zeros(EMISSION_WIDTH, dtype=torch.float64),
                   "post": torch.zeros(SHAPES[name][0], dtype=torch.float64),
                   "alpha_sum": torch.zeros((), dtype=torch.float64)}
            for name in ("mode", "literal")
        }
        stage._baselines = (float(self.baseline_mean[0 if kind == "source" else 1]),
                            float(self.baseline_mean[2]))
        _add(work, "eligibility_zero_values", 3 * WEIGHT_BYTES // 8)
        allocated = sum(v.numel() * v.element_size() for v in self._private_tensors(stage).values())
        _add(work, "causal_zero_values", (allocated - 3 * WEIGHT_BYTES) // 8)
        _add(work, "named_allocated_bytes", allocated)
        _add(work, "rng_generators", 1)
        _add(work, "rng_state_bytes", self.rng_state.numel())
        _add(work, "stages", 1)
        self._root = root
        self._root_stage = stage
        self._issued[stage._token] = weakref.ref(stage)
        self._source_ids = {key: _identity(getattr(self, key)) for key in TENSOR_NAMES}
        self._remember(stage)
        return stage

    @torch.no_grad()
    def draw(self, stage, population, activity, *, category, work=None):
        self._guard(stage, work)
        if (
            stage._finished
            or category not in CATEGORIES
            or category != stage._phase
            or population != stage._next
            or stage._pending_token is not None
        ):
            raise ValueError("draw is outside its causal population/category order")
        if category == "free" and stage._free_ended:
            raise ValueError("free response already reached EOS or its output limit")
        offered = stage.scan_offers if category == "scan" else stage._teacher_offers
        if category != "free" and stage._counts[category] >= offered:
            raise ValueError("draw exceeds externally declared offers")
        rows, width = SHAPES[population]
        validate_activity(activity, width, work)
        if category != "free" and population in ("mode", "literal"):
            phase = 2 if category == "teacher" else int(stage.kind != "source")
            _phase_alpha(activity.indices, activity.values, phase, work)
        count = activity.indices.numel()
        weights = getattr(self, population + "_weight")
        gathered = weights.index_select(1, activity.indices)
        products = gathered * activity.values
        logits = products.sum(1)
        _finite(logits, "controller currents")
        shifted = logits - logits.max()
        _finite(shifted, "controller shifted currents")
        exponentials = shifted.exp()
        total = exponentials.sum()
        probabilities = exponentials / total
        _finite(probabilities, "controller probabilities")
        if not math.isfinite(float(total)) or float(total) <= 0:
            raise ValueError("controller softmax cannot normalize")
        # Preserve finite participation. Underflow must not silently mask a cell.
        if bool((probabilities <= 0).any()):
            raise ValueError("controller probability underflow")
        _before, before_sha = self._rng(stage, work)
        uniform = float(torch.rand((), dtype=torch.float64, generator=stage._generator).item())
        after = stage._generator.get_state()
        after_sha = _digest({"rng": after}, work)
        cumulative, selected = 0.0, None
        for index, probability in enumerate(probabilities.tolist()):
            cumulative += probability
            _add(work, "cdf_additions", 1)
            if index == rows - 1:
                cumulative = 1.0
                _add(work, "cdf_endpoint_closures", 1)
            _add(work, "cdf_comparisons", 1)
            if selected is None and cumulative > uniform:
                selected = index
        if selected is None:
            raise RuntimeError("normalized CDF has no sampled cell")
        error = -probabilities
        error[selected] += 1.0
        update = error[:, None] * activity.values[None, :]
        eligible = stage._eligibility[category][population]
        columns = eligible.index_select(1, activity.indices)
        columns += update
        _finite(columns, "controller eligibility")
        eligible.index_copy_(1, activity.indices, columns)
        if category != "free" and population in ("context", "gate"):
            prefix_bank = stage._causal_prefix[category][population]
            prefix_columns = prefix_bank.index_select(1, activity.indices)
            prefix_columns += update * stage._prefix_totals[category]
            _finite(prefix_columns, "causal prefix eligibility")
            _add(work, "credit_validation_values", prefix_columns.numel())
            prefix_bank.index_copy_(1, activity.indices, prefix_columns)
            for key in ("causal_gathers", "causal_products", "causal_additions",
                        "causal_copy_values"):
                _add(work, key, rows * count)
            _add(work, "named_allocated_bytes", 16 * rows * count)
        if category != "free" and population == "mode":
            stage._mode_pending = {"probabilities": probabilities.clone(),
                                   "indices": activity.indices.clone(),
                                   "values": activity.values.clone()}
            stage._mode_sha256 = _digest(stage._mode_pending, work)
            copied = 8 * rows + 16 * count
            _add(work, "mode_pending_copy_bytes", copied)
            _add(work, "named_allocated_bytes", copied)
        if category != "free" and population == "literal":
            # Own immutable copies outlive the returned draw and caller activity.
            # Only this small pre-target bank is byte-checked at observe.
            log_probabilities = shifted - math.log(float(total))
            _finite(log_probabilities, "controller literal log probabilities")
            stage._literal_pending = {"probabilities": probabilities.clone(),
                                      "log_probabilities": log_probabilities,
                                      "squared_probability_norm": probabilities.square().sum(),
                                      "indices": activity.indices.clone(),
                                      "values": activity.values.clone()}
            stage._literal_sha256 = _digest(stage._literal_pending, work)
            # Copy accounting covers cloned p/indices/values; log-p and the
            # squared-probability norm are newly computed retained values.
            copied = 8 * rows + 16 * count
            _add(work, "literal_pending_copy_bytes", copied)
            _add(work, "brier_norm_products", rows)
            _add(work, "brier_norm_reductions", rows - 1)
            _add(work, "brier_norm_retained_bytes", 8)
            _add(work, "literal_logarithms", 1)
            _add(work, "literal_log_probability_subtractions", rows)
            _add(work, "credit_validation_values", rows)
            _add(work, "named_allocated_bytes", copied + 16 * rows + 8)
        stage._choices[population] = selected
        stage._next = {"context": "gate", "gate": "mode", "literal": "complete"}.get(
            population, "literal" if selected == 3 else "complete"
        )
        stage._rng_sha256 = after_sha
        stage._draws += 1
        self._remember(stage)
        _add(work, "rng_state_bytes", after.numel())
        _add(work, "rng_uniform_draws", 1)
        _add(work, "draws", 1)
        for key in (
            "score_gathers",
            "weight_products",
            "eligibility_products",
            "eligibility_gathers",
            "eligibility_additions",
            "eligibility_copy_values",
        ):
            _add(work, key, rows * count)
        _add(work, "score_reductions", rows * (count - 1))
        _add(work, "softmax_max_comparisons", rows - 1)
        _add(work, "softmax_subtractions", rows)
        _add(work, "softmax_exponentials", rows)
        _add(work, "softmax_sum_additions", rows - 1)
        _add(work, "softmax_divisions", rows)
        _add(work, "eligibility_errors", rows + 1)
        _add(work, "named_allocated_bytes", 32 * rows * count + 40 * rows + 8)
        return ControllerDraw(
            population,
            category,
            selected,
            logits,
            probabilities,
            uniform,
            _digest({"indices": activity.indices, "values": activity.values, "width": width}, work),
            before_sha,
            after_sha,
            stage._draws,
        )

    def complete_tick(self, stage, *, prior_token=None, register_tokens=(None, None), work=None):
        self._guard(stage, work)
        if stage._next != "complete" or len(register_tokens) != 2:
            raise ValueError("complete_tick needs all actual population choices")
        if prior_token is not None and (
            type(prior_token) is not int or not 0 <= prior_token <= 256
        ):
            raise ValueError("prior token must be an actual byte/EOS or absent")
        if any(
            t is not None and (type(t) is not int or not 0 <= t <= 258) for t in register_tokens
        ):
            raise ValueError("register token must be a raw symbol or absent")
        if stage._phase in ("free", "teacher") and prior_token is None:
            raise ValueError("output requires its actual causal frozen prior")
        if stage._phase == "scan" and prior_token is not None:
            raise ValueError("predictive scan cannot receive an output prior")
        mode = stage._choices["mode"]
        if mode == 0:
            token = 258 if prior_token is None else prior_token
            _add(work, "absent_prior_emissions", int(prior_token is None))
        elif mode in (1, 2):
            read = register_tokens[mode - 1]
            invalid = read is None or read > 256
            token = 256 if invalid else read
            _add(work, "invalid_register_emissions", int(invalid))
        else:
            token = stage._choices["literal"]
        stage._ticks[stage._phase] += 1
        if stage._phase == "free":
            stage._free_ended = token == 256 or stage._ticks["free"] == stage.max_bytes + 1
        else:
            stage._pending_token = token
            stage._pending_mode = mode
        stage._choices.clear()
        stage._next = "context"
        self._remember(stage)
        _add(work, "ticks", 1)
        return token

    @staticmethod
    def _associate(stage, population, category, error, held, work=None):
        """Own pre/post association, with private scan moments on actual contacts."""
        indices, values = held["indices"], held["values"]
        bank = (stage._literal_direct if population == "literal" else stage._mode_direct)[category]
        increment = error[:, None] * values[None, :]
        columns = bank.index_select(1, indices)
        columns += increment
        _finite(columns, "observed emission association")
        bank.index_copy_(1, indices, columns)
        size = increment.numel()
        if population == "literal":
            for key in ("literal_teaching_products", "literal_teaching_gathers",
                        "literal_teaching_additions", "literal_teaching_copy_values"):
                _add(work, key, size)
        else:
            _add(work, "mode_association_products", size)
            _add(work, "mode_association_values", 3 * size)
        _add(work, "credit_validation_values", size)
        _add(work, "named_allocated_bytes", 16 * size)
        if category == "scan":
            alpha = _phase_alpha(indices, values, int(stage.kind != "source"), work)
            moments = stage._scan_moments[population]
            moments["pre"].index_add_(0, indices, values)
            post = error * alpha
            moments["post"].add_(post)
            moments["alpha_sum"].add_(alpha)
            stage._scan_contacts[population] += 1
            _add(work, "phase_contacts", 1)
            _add(work, "phase_pre_additions", indices.numel())
            _add(work, "phase_post_products", error.numel())
            _add(work, "phase_post_additions", error.numel())
            _add(work, "phase_alpha_additions", 1)
            _add(work, "named_allocated_bytes", 8 * error.numel())

    @torch.no_grad()
    def observe(self, stage, target, *, category, work=None):
        self._guard(stage, work)
        if (
            category not in ("scan", "teacher")
            or stage._phase != category
            or stage._pending_token is None
            or type(target) is not int
            or not 0 <= target <= 256
        ):
            raise ValueError("observation needs one completed pre-arrival prediction")
        reward = 1 if stage._pending_token == target else -1
        probability = loss = activity_sha = None
        squared_probability_norm = 1.0
        if stage._mode_pending is None or _digest(stage._mode_pending, work) != stage._mode_sha256:
            raise ValueError("pre-target mode distribution or activity was modified")
        if (stage._pending_mode == 3) != (stage._literal_pending is not None):
            raise ValueError("observed literal branch lost its pre-target distribution")
        if stage._literal_pending is not None:
            held = stage._literal_pending
            if _digest(held, work) != stage._literal_sha256:
                raise ValueError("pre-target literal distribution or activity was modified")
            probability = float(held["probabilities"][target])
            squared_probability_norm = float(held["squared_probability_norm"])
            signal = 2.0 * probability - squared_probability_norm
            loss = -float(held["log_probabilities"][target])
            error = -held["probabilities"]
            error[target] += 1.0
            activity_sha = _digest({"indices": held["indices"], "values": held["values"],
                                    "width": EMISSION_WIDTH}, work)
            self._associate(stage, "literal", category, error, held, work)
            stage._literal_offers[category] += 1
            _add(work, "literal_teaching_errors", 258)
            _add(work, "literal_loss_negations", 1)
            _add(work, "literal_teaching_offers", 1)
            _add(work, "conditional_reward_products", 1)
            _add(work, "conditional_reward_subtractions", 1)
            _add(work, "named_allocated_bytes", 8 * 257)
        else:
            signal = float(reward)
        baseline = stage._baselines[int(category == "teacher")]
        advantage = signal - baseline
        mode_error = -stage._mode_pending["probabilities"]
        mode_error[stage._pending_mode] += 1.0
        mode_error *= advantage
        self._associate(stage, "mode", category, mode_error, stage._mode_pending, work)
        _add(work, "mode_association_products", 4)
        # Four own-error negations and one target-free sampled-choice addition.
        _add(work, "mode_association_values", 5)
        _add(work, "named_allocated_bytes", 32)
        before = stage._causal_totals[category]
        after = before + advantage
        prefix_before = stage._prefix_totals[category]
        prefix_after = prefix_before + advantage / (stage._counts[category] + 1)
        stage._last_observation = tuple({
            "schema": "integrated-local-learning-offer-v1", "category": category,
            "observed_token": target, "mode": stage._pending_mode,
            "sampled_reward": reward, "signal": signal, "baseline_before": baseline,
            "causal_total_before": before, "causal_total_after": after,
            "prefix_total_before": prefix_before, "prefix_total_after": prefix_after,
            "squared_probability_norm": squared_probability_norm,
            "signal_objective": "selected-emitter-Brier",
            "literal_taught": probability is not None, "target_probability": probability,
            "literal_loss": loss, "literal_activity_sha256": activity_sha,
        }.items())
        stage._counts[category] += 1
        stage._reward_sums[category] += reward
        stage._signal_sums[category] += signal
        stage._causal_totals[category] = after
        stage._prefix_totals[category] = prefix_after
        stage._literal_pending = None
        stage._literal_sha256 = None
        stage._mode_pending = None
        stage._mode_sha256 = None
        stage._pending_mode = None
        stage._pending_token = None
        self._remember(stage)
        _add(work, "offers", 1)
        _add(work, "outcome_comparisons", 1)
        _add(work, "baseline_subtractions", 1)
        _add(work, "signal_additions", 4)
        _add(work, "prefix_credit_divisions", 1)
        return reward

    def observation_record(self, stage, work=None):
        """Return scalar actual-symbol credit without exposing private tensors."""
        self._guard(stage, work)
        if not stage._last_observation or stage._pending_token is not None:
            raise ValueError("credit receipt requires an already consumed observation")
        return dict(stage._last_observation)

    def seal_scan(self, stage, work=None):
        self._guard(stage, work)
        if (
            stage._phase != "scan"
            or stage._counts["scan"] != stage.scan_offers
            or stage._pending_token is not None
            or stage._next != "context"
        ):
            raise ValueError("scan must consume exactly its declared symbols before sealing")
        stage._phase = "source_done" if stage.kind == "source" else "free"
        self._remember(stage)

    def seal_free(self, stage, work=None):
        self._guard(stage, work)
        if (
            stage.kind == "source"
            or stage._phase != "free"
            or not stage._free_ended
            or stage._next != "context"
        ):
            raise ValueError("free response must reach actual EOS or its fixed limit")
        stage._phase = "resolved_free"
        self._remember(stage)

    def declare_teacher(self, stage, offers, work=None):
        self._guard(stage, work)
        if (
            stage._phase != "resolved_free"
            or stage._teacher_offers is not None
            or type(offers) is not int
            or not 1 <= offers <= stage.max_bytes + 1
        ):
            raise ValueError("teacher declaration must follow actual free output once")
        stage._teacher_offers = offers
        stage._phase = "teacher"
        self._remember(stage)

    @torch.no_grad()
    def fork_stage(self, stage, work=None):
        self._guard(stage, work)
        if stage._phase != "resolved_free" or stage._finished:
            raise ValueError("only sealed actual free state may fork for resolution")
        rng, _ = self._rng(stage, work)
        fork = ControllerStage(
            stage.kind, stage.scan_offers, stage.max_bytes, stage.source_sha256, stage._root
        )
        fork._generator = torch.Generator(device="cpu").set_state(rng)
        fork._eligibility = {
            c: {n: v.clone() for n, v in bank.items()} for c, bank in stage._eligibility.items()
        }
        fork._causal_prefix = {
            c: {n: v.clone() for n, v in bank.items()} for c, bank in stage._causal_prefix.items()
        }
        fork._literal_direct = {c: v.clone() for c, v in stage._literal_direct.items()}
        fork._mode_direct = {c: v.clone() for c, v in stage._mode_direct.items()}
        fork._scan_moments = {
            n: {k: v.clone() for k, v in bank.items()} for n, bank in stage._scan_moments.items()
        }
        fork._scan_contacts = dict(stage._scan_contacts)
        fork._baselines = stage._baselines
        fork._signal_sums = dict(stage._signal_sums)
        fork._causal_totals = dict(stage._causal_totals)
        fork._prefix_totals = dict(stage._prefix_totals)
        fork._literal_offers = dict(stage._literal_offers)
        fork._last_observation = stage._last_observation
        fork._phase = stage._phase
        fork._counts = dict(stage._counts)
        fork._reward_sums = dict(stage._reward_sums)
        fork._ticks = dict(stage._ticks)
        fork._draws = stage._draws
        fork._free_ended = stage._free_ended
        fork._rng_sha256 = stage._rng_sha256
        self._issued[fork._token] = weakref.ref(fork)
        self._remember(fork)
        _add(work, "forks", 1)
        copied = sum(v.numel() * v.element_size() for v in self._private_tensors(fork).values())
        _add(work, "stage_copy_bytes", copied)
        _add(work, "named_allocated_bytes", copied)
        _add(work, "rng_generators", 1)
        _add(work, "rng_state_bytes", rng.numel())
        return fork

    @staticmethod
    def _center_scan(stage, population, work=None):
        moments, contacts = stage._scan_moments[population], stage._scan_contacts[population]
        direct = (stage._literal_direct if population == "literal" else stage._mode_direct)["scan"]
        if contacts == 0:
            if (float(moments["alpha_sum"]) != 0 or bool(moments["pre"].any())
                    or bool(moments["post"].any()) or bool(direct.any())):
                raise ValueError("empty scan contact set has nonzero association")
            _add(work, "named_allocated_bytes", 8 * direct.numel())
            return direct.clone()
        if contacts < 0 or float(moments["alpha_sum"]) <= 0:
            raise ValueError("scan covariance needs a positive actual contact weight")
        mean = moments["pre"] / moments["alpha_sum"]
        phase_column = EMISSION_PHASE_START + int(stage.kind != "source")
        if (float(mean[phase_column]) != 1.0 or bool((mean < 0).any())
                or bool((mean > 1.0 + 1e-12).any())):
            raise ValueError("scan mean differs from its weighted binary contact geometry")
        result = direct - moments["post"][:, None] * mean[None, :]
        result[:, phase_column] += moments["post"]
        _add(work, "phase_center_divisions", EMISSION_WIDTH)
        _add(work, "phase_center_products", direct.numel())
        _add(work, "phase_center_subtractions", direct.numel())
        _add(work, "phase_restore_additions", moments["post"].numel())
        _add(work, "named_allocated_bytes", 8 * EMISSION_WIDTH + 16 * direct.numel())
        return result

    @torch.no_grad()
    def finish(self, stage, *, learning_enabled, public_reward=None, work=None):
        self._guard(stage, work)
        if type(learning_enabled) is not bool or stage._finished:
            raise ValueError("finish needs a live stage and Boolean publication gate")
        phase = (
            "source_done"
            if stage.kind == "source"
            else ("resolved_free" if stage._teacher_offers is None else "teacher")
        )
        if (
            stage._phase != phase
            or stage._next != "context"
            or stage._pending_token is not None
            or stage._literal_pending is not None
            or stage._mode_pending is not None
            or stage._pending_mode is not None
            or stage._counts["teacher"] != (stage._teacher_offers or 0)
        ):
            raise ValueError("finish cannot omit outstanding draws or external offers")
        if stage.kind == "edit":
            if type(public_reward) is not int or public_reward not in (-1, 0):
                raise ValueError("scheduled edit retains its public category including zero STOP")
        elif public_reward is not None:
            raise ValueError("source/reading have no public execution category")
        rng, _ = self._rng(stage, work)
        self.check_parameters(work)
        if _digest(self._state(), work) != stage.source_sha256:
            raise ValueError("controller source changed before publication")
        for value in self._private_tensors(stage).values():
            _finite(value, "whole local credit bank")
            _add(work, "parameter_validation_values", value.numel())
        source_index = 0 if stage.kind == "source" else 1
        scan_reward = stage._signal_sums["scan"] / stage.scan_offers
        teacher_count = stage._teacher_offers or 0
        teacher_reward = (stage._signal_sums["teacher"] / teacher_count) if teacher_count else None
        public = 0.0 if public_reward is None else float(public_reward)
        count = 1 + int(teacher_reward is not None) + int(stage.kind == "edit")
        length = stage.scan_offers + (0 if stage.kind == "source" else teacher_count + 129)
        scale = 0.01 / (count * length)
        teacher_error = stage._causal_totals["teacher"] / stage.scan_offers
        contacts = {"mode": stage.scan_offers, "literal": stage._literal_offers["scan"]}
        if stage._scan_contacts != contacts:
            raise ValueError("scan covariance contact counts differ from actual offers")
        norm_bound = 0.01 * (
            EVENT_NORM_FACTORS["scan"] * stage.scan_offers
            + EVENT_NORM_FACTORS["teacher"] * teacher_count
            + abs(public) * math.sqrt(8)
            * (stage.scan_offers + stage._ticks["free"]) / (count * length)
        )
        coefficients = {
            "prefix_scan": 0.01,
            "prefix_teacher": 0.01 if teacher_count else 0.0,
            "common_scan_teacher": 0.01 * teacher_error,
            "public": scale * public,
            "scan_covariance": 0.01,
            "mode_teacher": 0.01 if teacher_count else 0.0,
            "literal_teacher": 0.01 if teacher_count else 0.0,
        }
        state, norm_squared = {}, 0.0
        for name in POPULATIONS:
            n = math.prod(SHAPES[name])
            scan = stage._eligibility["scan"][name]
            if name in ("mode", "literal"):
                direct = stage._literal_direct if name == "literal" else stage._mode_direct
                delta = self._center_scan(stage, name, work)
                if teacher_count:
                    delta += direct["teacher"]
                delta *= 0.01
                delta += (scan + stage._eligibility["free"][name]) * (scale * public)
                _add(work, "update_products", 2 * n)
                _add(work, "update_additions", (2 + int(bool(teacher_count))) * n)
                _add(work, "named_allocated_bytes", 16 * n)
            else:
                # A_final E-P is a forward sparse accumulation, evaluated once here.
                delta = stage._prefix_totals["scan"] * scan - stage._causal_prefix["scan"][name]
                if teacher_count:
                    teacher = (stage._prefix_totals["teacher"] * stage._eligibility["teacher"][name]
                               - stage._causal_prefix["teacher"][name])
                    delta += teacher_error * scan
                    delta += teacher
                delta *= 0.01
                delta += (scan + stage._eligibility["free"][name]) * (scale * public)
                _add(work, "update_products", (3 + 2 * int(bool(teacher_count))) * n)
                _add(work, "update_subtractions", (1 + int(bool(teacher_count))) * n)
                _add(work, "update_additions", (3 + 2 * int(bool(teacher_count))) * n)
                _add(work, "named_allocated_bytes", (48 + 24 * int(bool(teacher_count))) * n)
            _finite(delta, "controller proposed delta")
            squared = float((delta * delta).sum().item())
            norm_squared += squared
            weight = getattr(self, name + "_weight")
            candidate = weight + delta
            _finite(candidate, "controller proposed weights")
            state[name + "_weight"] = candidate if learning_enabled else weight
            _add(work, "update_norm_products", n)
            _add(work, "update_norm_reductions", n - 1)
            _add(work, "proposed_weight_bytes", 8 * n)
        norm = math.sqrt(norm_squared)
        if not math.isfinite(norm) or norm > norm_bound + UPDATE_NORM_TOLERANCE:
            raise ValueError("controller local update exceeds declared norm bound")
        means, counts = self.baseline_mean.clone(), self.baseline_count.clone()
        for index, value in ((source_index, scan_reward), (2, teacher_reward)):
            if value is None:
                continue
            old_count = int(counts[index])
            if old_count == torch.iinfo(torch.int64).max:
                raise ValueError("controller baseline count overflow")
            counts[index] += 1
            means[index] += (value - means[index]) / (old_count + 1)
            _add(work, "baseline_updates", 1)
        state.update(
            {
                "baseline_mean": means if learning_enabled else self.baseline_mean,
                "baseline_count": counts if learning_enabled else self.baseline_count,
                "rng_state": rng,
                "_extra_state": self.get_extra_state(),
            }
        )
        self.validate_state(state, work=work)
        rewards = {
            "scan": scan_reward,
            "teacher": teacher_reward,
            "sampled_scan": stage._reward_sums["scan"] / stage.scan_offers,
            "sampled_teacher": (stage._reward_sums["teacher"] / teacher_count
                                if teacher_count else None),
            "public": public_reward,
            "scan_baseline_before": float(self.baseline_mean[source_index]),
            "teacher_baseline_before": float(self.baseline_mean[2]),
            "coefficients": tuple(coefficients.items()),
            "causal_totals": tuple(stage._causal_totals.items()),
            "prefix_totals": tuple(stage._prefix_totals.items()),
            "literal_offers": tuple(stage._literal_offers.items()),
            "external_offers": (("scan", stage.scan_offers), ("teacher", teacher_count)),
            "scan_contacts": tuple(stage._scan_contacts.items()),
            "scan_alpha_sums": tuple((n, float(m["alpha_sum"]))
                                     for n, m in stage._scan_moments.items()),
            "scan_phase_column": EMISSION_PHASE_START + int(stage.kind != "source"),
            "norm_bound": norm_bound,
            "norm_bound_formula": "eta*(sqrt71*S+sqrt74*T+abs(r)*sqrt8*(S+F)/(K*L))",
            "objectives": ("Brier-prefix-context/gate; immediate-mode; contact-literal-CE; "
                           "scan-covariance; actual-public"),
        }
        state_hash = _digest(state, work)
        stage._finished = True
        self._remember(stage)
        stage_hash = self.stage_digest(stage, work)
        seal = _digest(
            {
                "state": state_hash,
                "stage": stage_hash,
                "source": stage.source_sha256,
                "enabled": learning_enabled,
                "rewards": rewards,
                "categories": count,
                "length": length,
                "norm": norm,
                "draws": stage._draws,
                "ticks": tuple(stage._ticks.items()),
            },
            work,
        )
        proposal = ControllerProposal(
            state,
            stage.source_sha256,
            state_hash,
            learning_enabled,
            rewards,
            count,
            length,
            norm,
            stage._draws,
            dict(stage._ticks),
            stage,
            stage_hash,
            seal,
        )
        _add(work, "finish_calls", 1)
        _add(work, "category_terms", count)
        return proposal

    def validate_stage(self, proposal, work=None):
        if not isinstance(proposal, ControllerProposal):
            raise ValueError("validate_stage requires a prepared controller proposal")
        self._guard(proposal._stage, work)
        stage_hash = self.stage_digest(proposal._stage, work)
        self.validate_state(proposal.state, work=work)
        state_hash = _digest(proposal.state, work)
        seal = _digest(
            {
                "state": state_hash,
                "stage": stage_hash,
                "source": proposal.source_sha256,
                "enabled": proposal.learning_enabled,
                "rewards": proposal.rewards,
                "categories": proposal.category_count,
                "length": proposal.normalization_ticks,
                "norm": proposal.delta_norm,
                "draws": proposal.draws,
                "ticks": tuple(proposal.ticks.items()),
            },
            work,
        )
        if (state_hash != proposal.state_sha256 or stage_hash != proposal._stage_sha256
                or seal != proposal._seal):
            raise ValueError("prepared controller proposal was modified")
        if _digest(self._state(), work) != proposal.source_sha256:
            raise ValueError("controller changed since proposal preparation")
        self._validated = proposal
        _add(work, "proposal_validations", 1)

    @torch.no_grad()
    def commit_validated(self, proposal, work=None):
        """Caller prevalidates every joint proposal; only copies follow this guard."""
        if self._validated is not proposal:
            raise ValueError("controller publication requires its validated proposal")
        names = TENSOR_NAMES if proposal.learning_enabled else ("rng_state",)
        for name in names:
            getattr(self, name).copy_(proposal.state[name])
            _add(
                work,
                "publication_copy_bytes",
                getattr(self, name).numel() * getattr(self, name).element_size(),
            )
        _add(work, "publications", 1)
        _add(work, "weight_publications", int(proposal.learning_enabled))
        _add(work, "rng_publications", 1)
        self.invalidate_stages_after_load()
