"""Race/output-pivot red tests T-P1..T-P8 (protocol d52c9d2, docs/research/race-pivot-protocol.md §7).

PREREGISTERED RED-TEST HARNESS. Written against current HEAD BEFORE any race-pivot
implementation exists, so that when R2 green-lights the pivot, implementation starts
from a ready harness instead of a cold start. No test here consumes the cohort.

Marker convention (task preregistration)
----------------------------------------
* ``@pytest.mark.xfail(strict=True, reason="... unimplemented ...")``
  → **RED now, expected**: the test exercises race-pivot behavior that does not exist
  at this HEAD (``complete_broadcast``, settle records, margin logging/calibration,
  abstention, sink reentrancy guard). ``strict=True`` means an unexpected XPASS fails
  the suite: when an implementation lands and the test turns green, the marker MUST be
  removed deliberately in the same change.
* No marker → **must pass always**: an invariant of existing behavior. These pin the
  seams the pivot depends on (non-mutating probes, tie-to-lowest-id, observation-sink
  purity, flag-off determinism). If one of these fails, the protocol's mechanism
  premises are broken — stop and re-examine before implementing.

Expected split at the writing HEAD (f086684; ``git diff 47e77bd..HEAD --
src/humanbrain/recurrent.py`` is empty, so race/observe/complete dynamics are
byte-identical to the design anchor 47e77bd and the goldens below are anchor-lineage
goldens)::

    GREEN now (invariants, 9):
      test_tp2_constructed_case_premises_hold_on_live_cells
      test_tp3_reference_ps_b1_bit_identity_with_complete
      test_tp3_reference_ps_b1_zero_extra_observe_events
      test_tp3_production_ps_b1_parity_with_complete            <- NO: xfail (RED)
      test_tp4_flagoff_golden_head_bit_identity
      test_tp4_flagoff_event_accounting_exact
      test_tp5_recomputed_margins_sign_convention_and_tie_semantics
      test_tp5_singleton_candidate_set_yields_explicit_zero_margin
      test_tp6_margin_pooling_reproducible_across_identical_grows
      test_tp8_sink_disabled_zero_calls_and_results_identical
    RED now (xfail strict, 12): every other test in this file.

Synthetic fixtures only (protocol §7: test_recurrent_observation_sink.py pattern,
token_cells=16, reservoir_cells=32, seed 7). All margins/stability numbers produced
here come from public primitives (``race``, ``observe``, ``output_current_pa``,
``observation_sink``); the B=1 P-S degenerate loop used by the parity tests is the
frozen §3.1 pseudocode transcribed host-side and is NOT a production implementation.

Single retargeting point: ``_require_broadcast_driver`` /
``_require_calibration_entrypoint`` / ``_require_abstention_scorer`` locate the future
production APIs. When R2 lands, update those three adapters (and only those) if the
real signatures differ, remove the xfail markers that turn green, and leave every
invariant test untouched.
"""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import humanbrain

# Guard: these tests must exercise THIS worktree's HEAD, never a stray install.
assert Path(humanbrain.__file__).resolve().is_relative_to(
    Path(__file__).resolve().parents[1]
), "humanbrain imported from outside the worktree; check sys.path/venv"

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import (
    RecurrentCellLanguageModel,
    RecurrentModelConfig,
)

MAX_SETTLE_ROUNDS = 4  # frozen, protocol §3.1

# ---------------------------------------------------------------------------
# T-P4 goldens: recorded by executing the flag-off path at the writing HEAD
# (see module docstring). Fixture: token_cells=16, reservoir_cells=32,
# input_fanout=4, recurrent_fanout=4, random_seed=7, ridge, spike_counts,
# grow_sequence(TRAIN_IDS, epochs=3), freeze_learning, prompt [2, 7, 13],
# 8 free-run positions. Regenerate only via a committed-tool change, never silently.
# ---------------------------------------------------------------------------
_TRAIN_IDS = [
    2, 2, 12, 7, 9, 9, 11, 0, 7, 2, 6, 14, 8, 1, 8, 2, 12, 15, 15, 9,
    13, 5, 2, 8, 7, 10, 15, 4,
]
_PROMPT_IDS = [2, 7, 13]
_FREE_RUN_TOKENS = 8
_GOLDEN_WINNERS = [2, 12, 15, 15, 15, 15, 2, 6]
_GOLDEN_WINNERS_SHA256 = (
    "b67b73f407b16d4a60f0297f8309ae05f45aa8171ff8055c3ecd056684a2f753"
)
# RE-PINNED (stale pin, not a regression): the original pin e2a3700525dc73ed
# 4ad95263cb2a9cb79ad1c227e0992f13c9d8b682c0cfb820 was recorded on a different
# numeric stack. Evidence chain (verified empirically before this re-pin):
#   1. `git diff 47e77bd..HEAD -- src/humanbrain/recurrent.py` is EMPTY — the
#      entire dynamics/race/observe path is byte-identical to the design anchor
#      47e77bd, and no native-kernel source changed on that range either.
#   2. The winner-sequence golden and the final feature-vector golden still
#      match their anchor-lineage pins exactly; ONLY the digest over the raw
#      float64 `output_current_pa()` bytes drifted.
#   3. On this machine the currents blob is bit-deterministic across repeated
#      runs (dc7743b0... twice), so the drift is not flakiness: it is a
#      last-ULP difference in the host BLAS matmul inside output_current_pa()
#      (base + pacemaker + heterogeneity + W @ features), which changes the
#      sha256 of raw float64 bytes without moving any argmax (margins are O(100)
#      pA; ULP noise is O(1e-13)). A sha256 over raw float64 matmul output is
#      therefore machine-portable only in its winner/feature projections, which
#      are pinned separately and still match.
_GOLDEN_CURRENTS_SHA256 = (
    "dc7743b0d7ecdd2369cb684e1350dd68030b8a553fbd34ee842bbe26694a5aa9"
)
_GOLDEN_FEATURES_SHA256 = (
    "6a802cde95066a6805a464ef00bde71958fea3083d98cb64446ffc0c1b9d2b6e"
)


# ---------------------------------------------------------------------------
# Fixtures (synthetic small circuits, protocol §7 pattern)
# ---------------------------------------------------------------------------
def _fixture_config() -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
        reservoir_cells=32,
        input_fanout=4,
        recurrent_fanout=4,
        random_seed=7,
        readout_learning_rule="ridge",
        feature_family="spike_counts",
    )


def _fixture_circuit() -> RecurrentCellCircuit:
    return RecurrentCellCircuit(token_cells=16, config=_fixture_config())


def _trained_circuit(train_ids: list[int], epochs: int = 3) -> RecurrentCellCircuit:
    circuit = _fixture_circuit()
    circuit.grow_sequence(train_ids, epochs=epochs)
    circuit.freeze_learning()
    return circuit


def _grow_model(text: str = "abcabcabcabc") -> RecurrentCellLanguageModel:
    return RecurrentCellLanguageModel.grow(
        [text],
        RecurrentModelConfig(
            growth_epochs=8,
            circuit=_fixture_config(),
        ),
    )


class _CountingSink:
    """Observation-sink recorder: counts events, captures post-boundary reads.

    Purely passive: it never drives the circuit (contract, recurrent.py
    observation_sink docstring). ``log`` rows are ``(token_id, currents_sha256)``.
    """

    def __init__(self) -> None:
        self.log: list[tuple[int, str]] = []

    def __call__(self, circuit: RecurrentCellCircuit, step) -> None:
        currents = np.ascontiguousarray(circuit.output_current_pa()).tobytes()
        self.log.append((int(step.token_id), sha256(currents).hexdigest()))

    @property
    def calls(self) -> int:
        return len(self.log)


# ---------------------------------------------------------------------------
# Executable spec helpers (host-side transcription of frozen protocol text)
# ---------------------------------------------------------------------------
def _reference_margin(
    currents: np.ndarray,
    winner: int,
    allowed: list[int] | tuple[int, ...] | None = None,
) -> float:
    """Frozen margin formula (§3.2, generalizing benchmarks/ladder_experiments.py:797).

    m = currents[winner] - max(currents[c] for c in allowed_set - {winner}).
    A single-candidate allowed set yields m = 0.0 explicitly — never a max over an
    empty set (this REFINES the ladder collector, which skips such positions).
    """
    current = np.asarray(currents, dtype=np.float64)
    candidates = range(len(current)) if allowed is None else list(allowed)
    competitors = [float(current[c]) for c in candidates if c != int(winner)]
    if not competitors:
        return 0.0
    return float(current[int(winner)]) - max(competitors)


def _ps_b1_reference_run(
    circuit: RecurrentCellCircuit,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> dict[str, object]:
    """Degenerate P-S/B=1 broadcast position loop, transcribed from §3.1.

    B=1 performs ZERO extra injections: injection #1 is the ordinary commit
    ``observe`` of current ``complete()``; the trailing ``race()`` is a pure,
    non-mutating stability probe. Emitted token is always the initial winner w0.
    """
    circuit.reset_sequence()
    for token_id in prompt_ids:
        circuit.observe(token_id)
    stream: list[int] = []
    step_shas: list[str] = []
    stabilities: list[bool] = []
    for _ in range(max_new_tokens):
        w0 = int(circuit.race().winner)
        stream.append(w0)
        circuit.observe(w0)  # injection #1 == the normal commit observe
        step_shas.append(
            sha256(np.ascontiguousarray(circuit.output_current_pa()).tobytes()).hexdigest()
        )
        probe = int(circuit.race().winner)  # non-mutating stability probe
        stabilities.append(probe == w0)
    return {
        "stream": stream,
        "step_shas": step_shas,
        "stabilities": stabilities,
        "final_features": np.ascontiguousarray(
            circuit.feature_state_vector()
        ).tobytes(),
    }


def _sha256_ids(stream: list[int]) -> str:
    return sha256(np.asarray(list(stream), dtype="<i8").tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Production-API adapters — THE single retargeting point when R2 lands
# ---------------------------------------------------------------------------
def _not_implemented(where: str, names: tuple[str, ...]) -> NotImplementedError:
    return NotImplementedError(
        "race-pivot feature unimplemented at this HEAD "
        f"(looked for {' / '.join(names)} on {where}); expected RED until the "
        "pivot is implemented per docs/research/race-pivot-protocol.md"
    )


def _require_broadcast_driver(holder: object):
    """Locate the future ``complete_broadcast`` entry point (§3.1).

    Expected call shape (adapt here if the real API differs):
    ``driver(prompt_ids, max_new_tokens, broadcast_b=B, variant="P-S"|"P-R",
    chase_to_settle=bool, max_settle_rounds=MAX_SETTLE_ROUNDS)`` returning the
    emitted token stream plus per-position settle records.
    """
    for obj, label in ((holder, "holder"), (getattr(holder, "_circuit", None), "circuit")):
        if obj is None:
            continue
        for name in ("complete_broadcast", "broadcast_complete", "run_broadcast"):
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
    raise _not_implemented(label, ("complete_broadcast", "broadcast_complete", "run_broadcast"))


def _coerce_stream(result: object) -> list[int]:
    for container in (result, getattr(result, "record", None)):
        if isinstance(container, dict):
            for key in ("tokens", "token_ids", "stream", "generated"):
                if key in container:
                    return [int(t) for t in container[key]]
        for name in ("tokens", "token_ids", "stream", "generated"):
            value = getattr(container, name, None)
            if value is not None:
                return [int(t) for t in value]
    raise _not_implemented("broadcast result", ("tokens", "token_ids", "stream", "generated"))


def _coerce_settle_records(result: object) -> list[object]:
    for container in (result, getattr(result, "record", None)):
        if isinstance(container, dict):
            for key in ("settle_records", "settle_profile", "positions", "records"):
                if key in container:
                    return list(container[key])
        for name in ("settle_records", "settle_profile", "positions", "records"):
            value = getattr(container, name, None)
            if value is not None:
                return list(value)
    raise _not_implemented(
        "broadcast result", ("settle_records", "settle_profile", "positions", "records")
    )


def _record_field(record: object, aliases: tuple[str, ...]):
    if isinstance(record, dict):
        for alias in aliases:
            if alias in record:
                return record[alias]
    for alias in aliases:
        value = getattr(record, alias, None)
        if value is not None:
            return value
    raise _not_implemented(f"settle record {record!r}", aliases)


def _settle_profile_hash(records: list[object]) -> str:
    """Canonical digest of settle profiles for cross-run comparison."""
    import json

    normalized = []
    for record in records:
        if isinstance(record, dict):
            normalized.append({k: repr(v) for k, v in sorted(record.items())})
        else:
            normalized.append(repr(record))
    blob = json.dumps(normalized, sort_keys=True).encode("utf-8")
    return sha256(blob).hexdigest()


def _require_calibration_entrypoint():
    """Locate the future margin-calibration / tau-freeze API (§3.2)."""
    candidates: list[tuple[object, str]] = []
    for module_name in ("humanbrain.race_pivot", "humanbrain.race_pivot_gate"):
        module = sys.modules.get(module_name)
        if module is None:
            try:
                import importlib

                module = importlib.import_module(module_name)
            except ImportError:
                continue
        candidates.append((module, module_name))
    module_names = tuple(name for _, name in candidates) or ("<absent modules>",)
    for module, _ in candidates:
        for name in ("freeze_tau", "calibrate_margins", "compute_tau"):
            fn = getattr(module, name, None)
            if callable(fn):
                return fn
    raise _not_implemented(
        ", ".join(module_names),
        ("freeze_tau", "calibrate_margins", "compute_tau"),
    )


def _require_abstention_scorer():
    """Locate the future P-M1 abstention labeling/scoring API (§3.2)."""
    for module_name in ("humanbrain.race_pivot", "humanbrain.race_pivot_gate"):
        module = sys.modules.get(module_name)
        if module is None:
            try:
                import importlib

                module = importlib.import_module(module_name)
            except ImportError:
                continue
        for name in ("selective_accuracy", "apply_abstention", "score_with_abstention"):
            fn = getattr(module, name, None)
            if callable(fn):
                return fn
    raise _not_implemented(
        "humanbrain.race_pivot[_gate]",
        ("selective_accuracy", "apply_abstention", "score_with_abstention"),
    )


# ===========================================================================
# T-P1 — broadcast determinism (protocol §7)
# ===========================================================================
def test_tp1_b2_two_runs_bit_identical_streams_settle_hashes_features() -> None:
    """Identical seed/prompt/config with B=2 → identical token streams, settle-profile
    hashes, bit-identical final feature vectors across two independent runs."""
    streams = []
    profile_hashes = []
    feature_blobs = []
    for _ in range(2):
        circuit = _trained_circuit(_TRAIN_IDS)
        driver = _require_broadcast_driver(circuit)
        result = driver(
            list(_PROMPT_IDS),
            _FREE_RUN_TOKENS,
            broadcast_b=2,
            variant="P-S",
            max_settle_rounds=MAX_SETTLE_ROUNDS,
        )
        streams.append(_coerce_stream(result))
        profile_hashes.append(_settle_profile_hash(_coerce_settle_records(result)))
        feature_blobs.append(
            np.ascontiguousarray(circuit.feature_state_vector()).tobytes()
        )
    assert streams[0] == streams[1]
    assert profile_hashes[0] == profile_hashes[1]
    assert feature_blobs[0] == feature_blobs[1]


# ===========================================================================
# T-P2 — stability convergence bounds (protocol §7, §3.1 control flow)
# ===========================================================================
def test_tp2_fixed_point_case_stable_within_max_settle_rounds() -> None:
    """Constructed fixed-point case (corpus 'aaaa…'): every position terminates STABLE
    within MAX_SETTLE_ROUNDS with correct settle counters; emitted == initial."""
    circuit = _trained_circuit([4] * 12)
    driver = _require_broadcast_driver(circuit)
    result = driver(
        [4], 3, broadcast_b=2, variant="P-S", max_settle_rounds=MAX_SETTLE_ROUNDS
    )
    records = _coerce_settle_records(result)
    assert len(records) == 3
    for record in records:
        assert _record_field(record, ("outcome", "status")) == "STABLE"
        rounds = _record_field(record, ("settle_rounds", "rounds", "probe_rounds"))
        assert isinstance(rounds, int) and 0 <= rounds <= MAX_SETTLE_ROUNDS
        assert _record_field(record, ("emitted_token", "emitted")) == 4
        assert _record_field(record, ("initial_winner", "initial", "w0")) == 4


def test_tp2_oscillating_case_unconverged_exactly_at_max_settle_rounds_and_returns() -> None:
    """Constructed oscillating case (corpus 'abab…', 2-cycle 3<->5 under chase): the
    position flags UNCONVERGED at exactly MAX_SETTLE_ROUNDS and the call RETURNS
    (never hangs)."""
    circuit = _trained_circuit([3, 5] * 10)
    driver = _require_broadcast_driver(circuit)
    result = driver(
        [3],
        2,
        broadcast_b=2,
        variant="P-R",
        chase_to_settle=True,
        max_settle_rounds=MAX_SETTLE_ROUNDS,
    )
    records = _coerce_settle_records(result)
    unconverged = [
        r
        for r in records
        if _record_field(r, ("outcome", "status")) == "UNCONVERGED"
    ]
    assert unconverged, "constructed 2-cycle must fail to converge under chase"
    for record in unconverged:
        rounds = _record_field(record, ("settle_rounds", "rounds", "probe_rounds"))
        assert rounds == MAX_SETTLE_ROUNDS


def test_tp2_pr_flip_bookkeeping_emitted_eq_settled_initial_recorded_separately() -> None:
    """P-R: emitted token is the SETTLED winner; the INITIAL race winner is recorded
    separately; flips are counted consistently (settled != initial ⇔ flipped)."""
    circuit = _trained_circuit([3, 5] * 10)
    driver = _require_broadcast_driver(circuit)
    result = driver(
        [3],
        4,
        broadcast_b=4,
        variant="P-R",
        chase_to_settle=True,
        max_settle_rounds=MAX_SETTLE_ROUNDS,
    )
    stream = _coerce_stream(result)
    records = _coerce_settle_records(result)
    flips = 0
    for emitted, record in zip(stream, records):
        settled = _record_field(record, ("settled_winner", "settled", "final_winner"))
        initial = _record_field(record, ("initial_winner", "initial", "w0"))
        assert _record_field(record, ("emitted_token", "emitted")) == settled
        assert emitted == settled
        flipped = _record_field(record, ("flipped", "flip"))
        assert bool(flipped) == (settled != initial)
        flips += int(settled != initial)
    reported_rate = _record_field(
        records[0], ("sequence_flip_rate",)
    ) if hasattr(records[0], "__getitem__") else None
    if reported_rate is not None:
        assert abs(float(reported_rate) * len(records) - flips) < 1e-9


def test_tp2_constructed_case_premises_hold_on_live_cells() -> None:
    """MUST PASS ALWAYS (invariant). The two T-P2 constructions exist on real fitted
    cells today: an all-same-token circuit is a race fixed point, and an 'abab'
    circuit produces a genuine 2-cycle winner alternation — so the convergence-bound
    cases above are constructible without any new machinery."""
    fixed = _trained_circuit([4] * 12)
    fixed.reset_sequence()
    fixed.observe(4)
    winner_a = int(fixed.race().winner)
    fixed.observe(winner_a)
    winner_b = int(fixed.race().winner)
    assert winner_a == 4 and winner_b == 4, "expected a race fixed point at token 4"

    osc = _trained_circuit([3, 5] * 10)
    osc.reset_sequence()
    trace: list[tuple[int, int]] = []
    token = 3
    for _ in range(2 * MAX_SETTLE_ROUNDS):
        osc.observe(token)
        nxt = int(osc.race().winner)
        trace.append((token, nxt))
        token = nxt
    pairs = set(trace)
    assert (3, 5) in pairs and (5, 3) in pairs, (
        f"expected a 3<->5 two-cycle for the oscillation construction, got {trace}"
    )
    assert all(nxt != tok for tok, nxt in trace), "cycle must actually alternate"


# ===========================================================================
# T-P3 — B=1 parity with current behavior (protocol §7; MUST hold already)
# ===========================================================================
def test_tp3_reference_ps_b1_bit_identity_with_complete() -> None:
    """MUST PASS ALWAYS (seam invariant). The §3.1 B=1 P-S degenerate loop — built ONLY
    from public primitives race()/observe() — reproduces ``complete()``'s token stream,
    per-step post-commit ``output_current_pa()`` sha256s, and final feature vector
    BIT-IDENTICALLY. This proves the pivot's core seam assumptions today: probes are
    pure and the afferent commit path is exactly what complete() exercises."""
    model = _grow_model()
    prompt_text = "abc"
    k = 8
    prompt_ids = list(model._tokenizer.encode(prompt_text))

    sink = _CountingSink()
    model._circuit.observation_sink = sink
    completed = model.complete(prompt_text, max_new_tokens=k)
    complete_events = list(sink.log)
    complete_features = np.ascontiguousarray(model._circuit.feature_state_vector()).tobytes()

    assert len(complete_events) == len(prompt_ids) + k, (
        "complete() must fire exactly len(prompt)+k observe events"
    )
    complete_generated_ids = [token for token, _ in complete_events[-k:]]
    complete_step_shas = [digest for _, digest in complete_events[-k:]]
    # Sanity: decoded stream must round-trip through the tokenizer for this check.
    assert list(model._tokenizer.encode(completed)) == complete_generated_ids

    sink.log.clear()
    reference = _ps_b1_reference_run(model._circuit, prompt_ids, k)

    assert reference["stream"] == complete_generated_ids
    assert reference["step_shas"] == complete_step_shas
    assert reference["final_features"] == complete_features
    # And the B=1 loop added zero extra observe events.
    assert len(sink.log) == len(complete_events)
    model._circuit.observation_sink = None


def test_tp3_reference_ps_b1_zero_extra_observe_events() -> None:
    """MUST PASS ALWAYS (injection-count accounting). B=1 P-S performs ZERO extra
    injections beyond complete()'s own commit observes: sink-call count AND
    ``observed_tokens`` match exactly, on twin circuits."""
    bare = _grow_model()
    instrumented = _grow_model()
    prompt_ids = list(bare._tokenizer.encode("abc"))

    bare.complete("abc", max_new_tokens=8)
    sink = _CountingSink()
    instrumented._circuit.observation_sink = sink
    instrumented.complete("abc", max_new_tokens=8)
    with_sink_calls = sink.calls
    sink.log.clear()

    _ps_b1_reference_run(instrumented._circuit, prompt_ids, 8)

    expected = len(prompt_ids) + 8
    assert with_sink_calls == expected
    assert sink.calls == expected, "B=1 P-S must add zero extra observe events"
    assert instrumented._circuit.observed_tokens == expected
    assert bare._circuit.observed_tokens == expected
    instrumented._circuit.observation_sink = None


def test_tp3_production_ps_b1_parity_with_complete() -> None:
    """Production P-S/B=1 must be BIT-IDENTICAL to current ``complete()``: token
    stream, per-step ``output_current_pa()`` sha256s, final feature vectors, and
    event-count equality via a counting sink (zero extra observe events)."""
    model = _grow_model()
    k = 8
    prompt_ids = list(model._tokenizer.encode("abc"))

    sink = _CountingSink()
    model._circuit.observation_sink = sink
    driver = _require_broadcast_driver(model)
    result = driver(
        prompt_ids,
        k,
        broadcast_b=1,
        variant="P-S",
        max_settle_rounds=MAX_SETTLE_ROUNDS,
    )
    b1_events = list(sink.log)
    b1_features = np.ascontiguousarray(model._circuit.feature_state_vector()).tobytes()
    sink.log.clear()

    completed = model.complete("abc", max_new_tokens=k)
    complete_events = list(sink.log)
    complete_features = np.ascontiguousarray(model._circuit.feature_state_vector()).tobytes()

    assert _coerce_stream(result) == [t for t, _ in complete_events[-k:]]
    assert [d for _, d in b1_events[-k:]] == [d for _, d in complete_events[-k:]]
    assert b1_features == complete_features
    assert len(b1_events) == len(complete_events) == len(prompt_ids) + k
    assert list(model._tokenizer.encode(completed)) == [t for t, _ in complete_events[-k:]]


# ===========================================================================
# T-P4 — flag-off bit-identity to HEAD behavior (protocol §7)
# ===========================================================================
def test_tp4_flagoff_golden_head_bit_identity() -> None:
    """MUST PASS ALWAYS (flag-off discipline). With broadcast disabled there is no
    race-pivot code on the path at all; the free-run trajectory must stay BYTE-IDENTICAL
    to the HEAD-recorded goldens (anchor-lineage: recurrent.py unchanged since 47e77bd).
    Any drift here means the flag-off contract of the future flag is already broken."""
    circuit = _trained_circuit(_TRAIN_IDS)
    currents_blob = bytearray()
    winners: list[int] = []
    circuit.reset_sequence()
    for token_id in _PROMPT_IDS:
        circuit.observe(token_id)
    for _ in range(_FREE_RUN_TOKENS):
        race = circuit.race()  # flag-off path: plain argmax-current race, no broadcast
        winners.append(int(race.winner))
        currents_blob += np.ascontiguousarray(circuit.output_current_pa()).tobytes()
        circuit.observe(race.winner)

    assert winners == _GOLDEN_WINNERS
    assert _sha256_ids(winners) == _GOLDEN_WINNERS_SHA256
    assert sha256(bytes(currents_blob)).hexdigest() == _GOLDEN_CURRENTS_SHA256
    assert (
        sha256(np.ascontiguousarray(circuit.feature_state_vector()).tobytes()).hexdigest()
        == _GOLDEN_FEATURES_SHA256
    )


def test_tp4_flagoff_event_accounting_exact() -> None:
    """MUST PASS ALWAYS (injection-count accounting). Flag-off generation injects
    EXACTLY len(prompt) + max_new_tokens afferent events — no more, no fewer — and the
    circuit's own ``observed_tokens`` ledger agrees with the counting sink."""
    circuit = _trained_circuit(_TRAIN_IDS)
    sink = _CountingSink()
    circuit.observation_sink = sink
    circuit.reset_sequence()
    for token_id in _PROMPT_IDS:
        circuit.observe(token_id)
    for _ in range(_FREE_RUN_TOKENS):
        circuit.observe(int(circuit.race().winner))
    expected = len(_PROMPT_IDS) + _FREE_RUN_TOKENS
    assert sink.calls == expected
    assert circuit.observed_tokens == expected
    assert [token for token, _ in sink.log[: len(_PROMPT_IDS)]] == _PROMPT_IDS
    circuit.observation_sink = None


# ===========================================================================
# T-P5 — margin capture correctness (protocol §7, §3.2 formula)
# ===========================================================================
def test_tp5_recomputed_margins_sign_convention_and_tie_semantics() -> None:
    """MUST PASS ALWAYS (invariant). Over a live free-run the recomputed margin
    m = currents[winner] − max(currents[competitors]) equals a brute-force recompute to
    0.0 tolerance, is non-negative (the race winner is the first-occurrence argmax), and
    an exact tie resolves to the LOWEST id with margin exactly 0.0 (public
    ``race_output_current_pa`` seam)."""
    circuit = _trained_circuit([3, 5] * 10)
    circuit.reset_sequence()
    token = 3
    checked = 0
    for _ in range(6):
        circuit.observe(token)
        currents = np.ascontiguousarray(circuit.output_current_pa())
        winner = int(circuit.race().winner)
        margin = _reference_margin(currents, winner)
        # Brute-force independent recompute, exact tolerance.
        others = [float(currents[i]) for i in range(len(currents)) if i != winner]
        assert margin == float(currents[winner]) - max(others)
        assert margin >= 0.0, "argmax-current race winner can never trail a competitor"
        assert winner == int(np.argmax(currents)), "tie-to-lowest-id == first occurrence"
        checked += 1
        token = winner
    assert checked == 6

    # Crafted exact tie: cells 6 and 9 share the top current → lowest id wins, margin 0.
    rng = np.random.default_rng(5)
    vec = rng.normal(size=16) * 1000.0
    vec[6] = abs(vec).max() + 500.0
    vec[9] = vec[6]
    tied = circuit.race_output_current_pa(vec)
    assert tied.winner == 6
    assert _reference_margin(vec, tied.winner) == 0.0


def test_tp5_singleton_candidate_set_yields_explicit_zero_margin() -> None:
    """MUST PASS ALWAYS (invariant + executable spec). A single-candidate allowed set
    yields m = 0 explicitly — never a max over an empty set (refines the ladder
    collector, which skips such positions). The live singleton race itself is reachable
    through the public seam when the candidate is competitive."""
    circuit = _trained_circuit([3, 5] * 10)
    circuit.reset_sequence()
    circuit.observe(3)
    rng = np.random.default_rng(5)
    vec = rng.normal(size=16) * 1000.0
    vec[11] = abs(vec).max() + 2000.0  # craft the singleton clearly competitive
    race = circuit.race_output_current_pa(vec, allowed_token_ids=[11])
    assert race.winner == 11
    assert _reference_margin(vec, 11, allowed=[11]) == 0.0
    # Spec edge exercised directly as well: empty competitor set, no exception.
    assert _reference_margin(np.arange(4, dtype=float), 2, allowed=[2]) == 0.0


def test_tp5_logged_margins_equal_independent_recompute_zero_tolerance() -> None:
    """Logged margins must equal independently recomputed values from the public
    ``output_current_pa()`` to 0.0 tolerance, carry the target−best-competitor sign
    convention, and handle single-candidate sets as m = 0."""
    circuit = _trained_circuit([3, 5] * 10)
    driver = _require_broadcast_driver(circuit)
    result = driver(
        [3], 4, broadcast_b=1, variant="P-S", record_settle_profile=True
    )
    records = _coerce_settle_records(result)
    for record in records:
        logged = float(_record_field(record, ("margin_pa", "margin")))
        winner = int(_record_field(record, ("committed_winner", "emitted_token")))
        currents = np.ascontiguousarray(circuit.output_current_pa())
        assert logged == _reference_margin(currents, winner)
        assert logged >= 0.0


# ===========================================================================
# T-P6 — margin calibration reproducibility (protocol §7, §3.2 freeze)
# ===========================================================================
def test_tp6_margin_pooling_reproducible_across_identical_grows() -> None:
    """MUST PASS ALWAYS (premise of tau reproducibility). Two identically-configured
    fresh builds pool BYTE-IDENTICAL margin distributions over the same sweep, so
    freezing τ twice from the same manifest can only disagree if the calibration
    machinery itself is nondeterministic."""
    def pooled_margins() -> tuple[float, ...]:
        model = _grow_model("ababababab")
        circuit = model._circuit
        ids = list(model._tokenizer.encode("abab"))
        circuit.reset_sequence()
        circuit.observe(ids[0])
        margins: list[float] = []
        for target in ids[1:]:
            currents = np.ascontiguousarray(circuit.output_current_pa())
            winner = int(circuit.race().winner)
            margins.append(_reference_margin(currents, winner))
            circuit.observe(target)
        return tuple(margins)

    first = pooled_margins()
    second = pooled_margins()
    assert first == second  # bitwise: machine precision, identical paths
    assert len(first) == 3


def test_tp6_tau_freeze_twice_machine_precision_and_hash_sealed() -> None:
    """τ computed twice from the frozen TRAIN pool agrees to machine precision, and the
    τ block is hash-sealed into the gate JSON's pre-registration block."""
    freeze = _require_calibration_entrypoint()
    pool = [1500.0 + float(i) for i in range(100)]
    block_a = freeze(pool, tau_low_percentile=5.0, tau_high_percentile=95.0)
    block_b = freeze(pool, tau_low_percentile=5.0, tau_high_percentile=95.0)
    tau_low_a = _record_field(block_a, ("tau_low",))
    tau_low_b = _record_field(block_b, ("tau_low",))
    tau_high_a = _record_field(block_a, ("tau_high",))
    tau_high_b = _record_field(block_b, ("tau_high",))
    assert tau_low_a == tau_low_b  # machine precision
    assert tau_high_a == tau_high_b
    assert _record_field(block_a, ("block_sha256", "hash", "seal")) == _record_field(
        block_b, ("block_sha256", "hash", "seal")
    )


def test_tp6_preflight_rejects_post_freeze_calibration_mutation() -> None:
    """Any post-freeze mutation of the calibration inputs must FAIL the anchor check
    (hash mismatch ⇒ ABORT, cohort not spent — simulated in-test)."""
    freeze = _require_calibration_entrypoint()
    pool = [1500.0 + float(i) for i in range(100)]
    sealed = freeze(pool, tau_low_percentile=5.0, tau_high_percentile=95.0)
    mutated = list(pool)
    mutated[0] += 1.0  # post-freeze recalibration attempt
    with pytest.raises(Exception) as excinfo:  # noqa: B017 - contract is "must raise"
        freeze(mutated, tau_low_percentile=5.0, tau_high_percentile=95.0, sealed_block=sealed)
    assert "anchor" in str(excinfo.value).lower() or "hash" in str(excinfo.value).lower()


# ===========================================================================
# T-P7 — abstention correctness (protocol §7, §3.2 P-M1)
# ===========================================================================
def test_tp7_abstention_labels_straddle_tau_low() -> None:
    """Crafted margin vectors straddling τ_low produce exactly the specified
    abstain/answer labels; high-confidence flags fire strictly above τ_high."""
    apply_abstention = _require_abstention_scorer()
    tau_low, tau_high = 100.0, 2000.0
    decisions = [
        {"position": 0, "margin_pa": 50.0, "would_be_correct": True},   # < τ_low → ABSTAIN
        {"position": 1, "margin_pa": 99.999, "would_be_correct": True},  # < τ_low → ABSTAIN
        {"position": 2, "margin_pa": 100.0, "would_be_correct": False},  # == τ_low → answer
        {"position": 3, "margin_pa": 1500.0, "would_be_correct": False},  # mid → answer
        {"position": 4, "margin_pa": 2500.0, "would_be_correct": True},   # > τ_high → flag
    ]
    labeled = apply_abstention(decisions, tau_low=tau_low, tau_high=tau_high)
    labels = [_record_field(r, ("label", "decision")) for r in labeled]
    assert labels[0] == "ABSTAIN" and labels[1] == "ABSTAIN"
    assert labels[2] != "ABSTAIN" and labels[3] != "ABSTAIN" and labels[4] != "ABSTAIN"
    assert bool(_record_field(labeled[4], ("high_confidence", "flag"))) is True
    assert not any(
        _record_field(r, ("high_confidence", "flag")) for r in labeled[:4]
    )


def test_tp7_coverage_denominator_and_abstained_never_correct() -> None:
    """Coverage denominators verified on synthetic records; a position engineered to be
    CORRECT but abstained never counts toward selective accuracy (denominator and
    numerator both exclude it)."""
    score = _require_abstention_scorer()
    tau_low, tau_high = 100.0, 2000.0
    decisions = [
        {"position": 0, "margin_pa": 50.0, "would_be_correct": True},    # abstained!
        {"position": 1, "margin_pa": 99.0, "would_be_correct": False},   # abstained!
        {"position": 2, "margin_pa": 800.0, "would_be_correct": True},   # answered, correct
        {"position": 3, "margin_pa": 900.0, "would_be_correct": False},  # answered, wrong
    ]
    summary = score(decisions, tau_low=tau_low, tau_high=tau_high)
    coverage = _record_field(summary, ("coverage",))
    selective = _record_field(summary, ("selective_accuracy", "selective_span_accuracy"))
    assert coverage == pytest.approx(2 / 4)
    assert selective == pytest.approx(1 / 2)  # the abstained-correct position is excluded


# ===========================================================================
# T-P8 — logging discipline through the observation_sink contract (§7)
# ===========================================================================
def test_tp8_sink_disabled_zero_calls_and_results_identical() -> None:
    """MUST PASS ALWAYS (existing contract). Logging disabled (sink None) → zero sink
    calls and results identical to a sink-attached run (observers are read-only)."""
    bare = _grow_model()
    instrumented = _grow_model()
    bare_out = bare.complete("abc", max_new_tokens=8)
    sink = _CountingSink()
    instrumented._circuit.observation_sink = sink
    instrumented_out = instrumented.complete("abc", max_new_tokens=8)
    assert sink.calls > 0  # attached → fired per event
    assert bare_out == instrumented_out
    assert bare._circuit.observed_tokens == instrumented._circuit.observed_tokens

    silent = _grow_model()
    assert silent._circuit.observation_sink is None
    before = silent.complete("abc", max_new_tokens=8)
    assert silent._circuit.observed_tokens == len(list(silent._tokenizer.encode("abc"))) + 8
    assert before == bare_out  # logging fully off changes nothing


def test_tp8_sink_that_drives_circuit_raises() -> None:
    """A callback that calls observe/race/reset must RAISE with a deliberate guard
    error immediately. Currently only docstring discipline exists: a driving sink runs
    away into RecursionError (a RuntimeError subclass) — asserted NOT acceptable."""
    circuit = _trained_circuit([3, 5] * 10)
    circuit.reset_sequence()

    def driving_sink(observed_circuit: RecurrentCellCircuit, token_id: int) -> None:
        observed_circuit.observe((token_id + 1) % observed_circuit.token_cells)

    circuit.observation_sink = driving_sink
    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        circuit.observe(3)
    assert not isinstance(
        excinfo.value, RecursionError
    ), "runaway recursion is not a reentrancy guard; expected a deliberate guard error"
    circuit.observation_sink = None


def test_tp8_settle_profiles_and_margins_flow_through_observation_sink() -> None:
    """Settle profiles and margins reach logs THROUGH the observation_sink contract:
    post-boundary reads only, and the sink-visible state equals the caller-visible
    state after each injection (no pre-boundary leakage)."""
    circuit = _trained_circuit([3, 5] * 10)
    captured: list[dict] = []

    def logging_sink(observed_circuit: RecurrentCellCircuit, token_id: int) -> None:
        captured.append(
            {
                "token": token_id,
                "features": np.ascontiguousarray(
                    observed_circuit.feature_state_vector()
                ).tobytes(),
                "settle": getattr(observed_circuit, "last_settle_record", None),
                "margin": getattr(observed_circuit, "last_margin_pa", None),
            }
        )

    circuit.observation_sink = logging_sink
    driver = _require_broadcast_driver(circuit)
    result = driver([3], 3, broadcast_b=2, variant="P-S")
    records = _coerce_settle_records(result)
    assert captured, "logging sink must observe every broadcast injection"
    for row, record in zip(captured, records):
        assert row["settle"] is not None
        assert row["margin"] is not None
        assert _record_field(record, ("emitted_token", "emitted")) == row["token"]
