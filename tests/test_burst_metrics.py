"""Burst-fraction tracking (DD-HTM TM surprise metric) over observation-sink events.

Coverage:
- hand-computed burst classification, fractions, windows, per-cell counts on
  synthetic traces;
- predicted-hit vs burst discrimination and inhibitory (negative-current)
  exclusion from the predictive set;
- sequence-reset handling (stale prediction state must not survive a reset);
- ActivityRecorder flag-off parity: default construction keeps the activity
  stream schema byte-for-byte legacy (no ``burst_metrics`` keys anywhere);
- ActivityRecorder flag-on: footer carries the burst-metrics snapshot;
- live-circuit non-invasiveness: wiring the tracker (or a tracked recorder)
  into the observation sink leaves dynamics bit-identical.

Burst semantics (docs/research/deep-dive-htm-tm.md §1.4 / adopt table): a cell
"bursts" when it fires during a token although NO predictive depolarization
(context drive) was present prior to input arrival. The tracker consumes the
same ``(circuit, step)`` events the observation sink sees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from humanbrain.burst_metrics import BURST_METRICS_FORMAT, BurstFractionTracker
from humanbrain.observatory import ACTIVITY_STREAM_FORMAT, ActivityRecorder
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


class _ScriptedCircuit:
    """Minimal duck-typed circuit feeding scripted kernel views to the tracker."""

    def __init__(
        self,
        spike_counts: list[np.ndarray],
        context_current: list[np.ndarray],
    ) -> None:
        assert len(spike_counts) == len(context_current)
        self._spike_counts = iter(spike_counts)
        self._context_current = iter(context_current)
        self.observed_tokens = 0

    @property
    def token_spike_counts(self) -> np.ndarray:
        """Post-token per-cell spike accumulator view (live u32 semantics)."""
        return next(self._spike_counts)

    @property
    def context_current_pa(self) -> np.ndarray:
        """Post-token recurrent context drive; predicts the NEXT arrival."""
        return next(self._context_current)


def _u32(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.uint32)


def _f64(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Hand-computed metric correctness
# ---------------------------------------------------------------------------


class TestHandComputedMetrics:
    def test_full_trace_classification_windows_and_per_cell_counts(self) -> None:
        # Four cells, three scripted tokens, window_tokens=2. The context
        # sampled AFTER event k is the predictive set for arrival k+1.
        #
        # step 1: cells {0} fire;   post-token context {1:+2}
        #         predicted(t1) = {}      burst(t1) = {0}
        # step 2: cells {1} fire;   post-token context {0:+3, 1:-1, 3:+5}
        #         predicted(t2) = {1}     burst(t2) = {}            (hit)
        # step 3: cells {0,1,2} fire
        #         predicted(t3) = {0,3}   burst(t3) = {1,2}  (cell 1's old
        #         +2 context was consumed at t2; it now sits at -1, so its
        #         t3 spike is unpredicted again)
        #
        # Totals: active 5, bursts 3 -> fraction 3/5.
        circuit = _ScriptedCircuit(
            spike_counts=[
                _u32([1, 0, 0, 0]),
                _u32([0, 1, 0, 0]),
                _u32([1, 1, 1, 0]),
            ],
            context_current=[
                _f64([0.0, 2.0, 0.0, 0.0]),
                _f64([3.0, -1.0, 0.0, 5.0]),
                _f64([0.0, 0.0, 0.0, 0.0]),
            ],
        )
        tracker = BurstFractionTracker(window_tokens=2)
        for _ in range(3):
            circuit.observed_tokens += 1
            tracker.record(circuit, None)

        snapshot = tracker.snapshot()
        assert snapshot["format"] == BURST_METRICS_FORMAT
        assert snapshot["window_tokens"] == 2
        assert snapshot["steps_observed"] == 3
        assert snapshot["active_cells_total"] == 5
        assert snapshot["burst_cells_total"] == 3
        assert snapshot["burst_fraction"] == pytest.approx(3.0 / 5.0)
        # Window 1 = steps {1,2}: 2 active (cells 0 and 1), 1 burst (cell 0).
        assert snapshot["completed_window_burst_fractions"] == [0.5]
        assert snapshot["open_window"]["tokens"] == 1
        assert snapshot["open_window"]["active_cells"] == 3
        assert snapshot["open_window"]["burst_cells"] == 2
        assert snapshot["open_window"]["burst_fraction"] == pytest.approx(2.0 / 3.0)
        assert snapshot["per_cell_burst_counts"] == [[0, 1], [1, 1], [2, 1]]

    def test_first_observed_token_always_bursts(self) -> None:
        circuit = _ScriptedCircuit(
            spike_counts=[_u32([1, 1, 0, 0])],
            context_current=[_f64([0.0, 0.0, 9.0, 9.0])],
        )
        tracker = BurstFractionTracker()
        circuit.observed_tokens = 1
        tracker.record(circuit, None)
        snapshot = tracker.snapshot()
        assert snapshot["active_cells_total"] == 2
        assert snapshot["burst_cells_total"] == 2
        assert snapshot["burst_fraction"] == 1.0

    def test_empty_trace_yields_null_fractions(self) -> None:
        snapshot = BurstFractionTracker().snapshot()
        assert snapshot["steps_observed"] == 0
        assert snapshot["burst_fraction"] is None
        assert snapshot["completed_window_burst_fractions"] == []
        assert snapshot["open_window"]["burst_fraction"] is None
        assert json.dumps(snapshot, allow_nan=False)  # JSON-safe


# ---------------------------------------------------------------------------
# Predicted-hit vs burst discrimination
# ---------------------------------------------------------------------------


class TestDiscrimination:
    def test_same_cell_predicted_hit_and_burst_are_separated(self) -> None:
        # Cell 7 fires four times. Context sampled after event k predicts
        # arrival k+1: with contexts [predict, none, predict, none] the
        # sequence is burst (first-ever), HIT, burst (context withdrawn),
        # HIT. Active contribution 4, burst count 2.
        fire7 = _u32([0, 0, 0, 0, 0, 0, 0, 1])
        current_none = _f64([0.0] * 8)
        current_predict_7 = _f64([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0])
        circuit = _ScriptedCircuit(
            spike_counts=[fire7, fire7, fire7, fire7],
            context_current=[
                current_predict_7,
                current_none,
                current_predict_7,
                current_none,
            ],
        )
        tracker = BurstFractionTracker(window_tokens=10)
        for _ in range(4):
            circuit.observed_tokens += 1
            tracker.record(circuit, None)

        snapshot = tracker.snapshot()
        assert snapshot["steps_observed"] == 4
        assert snapshot["active_cells_total"] == 4
        assert snapshot["burst_cells_total"] == 2
        assert snapshot["burst_fraction"] == pytest.approx(0.5)
        assert snapshot["per_cell_burst_counts"] == [[7, 2]]

    def test_inhibitory_context_drive_is_not_prediction(self) -> None:
        # Cell 1 bursts unpredicted at t1 and leaves +4 on cell 0 and -50
        # (inhibition) on cell 2. At t2 cells 0 and 2 both fire: cell 0 was
        # depolarized-predictive (hit), but negative drive is NOT predictive
        # state, so cell 2's spike counts as a burst.
        circuit = _ScriptedCircuit(
            spike_counts=[_u32([0, 1, 0]), _u32([1, 0, 1])],
            context_current=[
                _f64([4.0, 0.0, -50.0]),
                _f64([0.0, 0.0, 0.0]),
            ],
        )
        tracker = BurstFractionTracker()
        for _ in range(2):
            circuit.observed_tokens += 1
            tracker.record(circuit, None)
        snapshot = tracker.snapshot()
        assert snapshot["steps_observed"] == 2
        assert snapshot["active_cells_total"] == 3
        # Bursts: cell 1 (t1, first-ever) and cell 2 (t2, inhibited is not
        # predicted). Cell 0's t2 spike was a predicted hit, not a burst.
        assert snapshot["burst_cells_total"] == 2
        assert snapshot["per_cell_burst_counts"] == [[1, 1], [2, 1]]

    def test_sequence_reset_discards_stale_prediction(self) -> None:
        # Token 1 leaves positive context on cell 0. A reset happens (the
        # observed-token counter goes backwards); token 2 fires cell 0 with
        # NO fresh context: it must count as a burst, not a predicted hit.
        circuit = _ScriptedCircuit(
            spike_counts=[_u32([0, 1]), _u32([1, 0])],
            context_current=[_f64([6.0, 0.0]), _f64([0.0, 0.0])],
        )
        tracker = BurstFractionTracker()
        circuit.observed_tokens = 1
        tracker.record(circuit, None)
        circuit.observed_tokens = 1  # reset_sequence() rewound the counter
        tracker.record(circuit, None)
        snapshot = tracker.snapshot()
        assert snapshot["steps_observed"] == 2
        assert snapshot["burst_cells_total"] == 2
        assert snapshot["per_cell_burst_counts"] == [[0, 1], [1, 1]]


# ---------------------------------------------------------------------------
# Window accounting
# ---------------------------------------------------------------------------


class TestWindows:
    def test_completed_windows_close_exactly_on_the_boundary(self) -> None:
        steps = 5
        circuit = _ScriptedCircuit(
            spike_counts=[_u32([1, 0])] * steps,
            context_current=[_f64([0.0, 0.0])] * steps,
        )
        tracker = BurstFractionTracker(window_tokens=2)
        for _ in range(steps):
            circuit.observed_tokens += 1
            tracker.record(circuit, None)
        snapshot = tracker.snapshot()
        # Every token bursts (no context ever): windows of 2 active/2 bursts.
        assert snapshot["completed_window_burst_fractions"] == [1.0, 1.0]
        assert snapshot["open_window"] == {
            "tokens": 1,
            "active_cells": 1,
            "burst_cells": 1,
            "burst_fraction": 1.0,
        }
        assert snapshot["mean_completed_window_burst_fraction"] == 1.0


# ---------------------------------------------------------------------------
# Live-circuit integration (real GLIF kernel, tiny population)
# ---------------------------------------------------------------------------


def _mini_circuit(seed: int = 11) -> RecurrentCellCircuit:
    return RecurrentCellCircuit(
        token_cells=8,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=4,
            recurrent_fanout=4,
            random_seed=seed,
        ),
    )


class TestLiveCircuitNonInvasiveness:
    def test_wired_tracker_leaves_dynamics_bit_identical(self) -> None:
        # Drive two same-seed circuits step-by-step; one carries the tracker
        # on its observation sink. Every spike total and every sampled state
        # must match bit-for-bit: instrumentation reads, never writes.
        plain = _mini_circuit(seed=5)
        tracked = _mini_circuit(seed=5)
        tracker = BurstFractionTracker(window_tokens=8)
        tracked.observation_sink = tracker.record
        gids = tuple(range(plain.active_reservoir_cells))
        for token in (2, 5, 2, 5, 3, 1, 4, 6, 2, 5, 2):
            for circuit in (plain, tracked):
                circuit.reset_sequence()
            for repeat in range(3):
                token_id = (token + repeat) % plain.token_cells
                step_plain = plain.observe(token_id)
                step_tracked = tracked.observe(token_id)
                assert step_plain.reservoir_spikes == step_tracked.reservoir_spikes
                left = plain.capture_dynamics(gids)
                right = tracked.capture_dynamics(gids)
                assert left.voltage_relative_mv == right.voltage_relative_mv
                assert left.synaptic_current_pa == right.synaptic_current_pa
                assert left.eligibility == right.eligibility
        # The tracker actually observed the same trajectory.
        assert tracker.snapshot()["steps_observed"] == 33

    def test_tracker_metrics_reflect_a_real_sequence(self) -> None:
        circuit = _mini_circuit(seed=5)
        tracker = BurstFractionTracker(window_tokens=1000)
        circuit.observation_sink = tracker.record
        for token in (2, 5, 2, 5, 3, 1, 4, 6):
            circuit.reset_sequence()
            for repeat in range(4):
                circuit.observe((token + repeat) % circuit.token_cells)
        snapshot = tracker.snapshot()
        assert snapshot["steps_observed"] == 32
        assert 0.0 <= snapshot["burst_fraction"] <= 1.0
        assert snapshot["active_cells_total"] > 0
        # Sorted by gid, counts positive.
        pairs = snapshot["per_cell_burst_counts"]
        assert pairs == sorted(pairs)
        assert all(count > 0 for _, count in pairs)


# ---------------------------------------------------------------------------
# Observation-sink pipeline hook (flag off = bit-exact legacy schema)
# ---------------------------------------------------------------------------

_FRAME_KEYS_BASE = {
    "type",
    "index",
    "phase",
    "token_id",
    "biological_time_ms",
    "spike_count",
    "population_firing_rate_hz",
    "neuron_step_activity_density",
}
_FRAME_KEYS_WITH_STATE = _FRAME_KEYS_BASE | {
    "voltage_relative_mv",
    "synaptic_current_pa",
    "eligibility",
}


def _read_stream(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class TestActivityRecorderHook:
    def _record_stream(
        self,
        tmp_path: Path,
        name: str,
        *,
        burst_tracker: BurstFractionTracker | None,
    ) -> tuple[Path, RecurrentCellCircuit]:
        path = tmp_path / name
        recorder = ActivityRecorder(
            path,
            run_id=f"live/{name}",
            sample_count=4,
            state_stride=2,
            burst_tracker=burst_tracker,
        )
        circuit = _mini_circuit(seed=7)
        circuit.observation_sink = recorder.record
        recorder.set_phase("training")
        circuit.reset_sequence()
        for token in (2, 5, 2, 5, 3, 1):
            circuit.observe(token)
        recorder.close()
        return path, circuit

    def test_flag_off_keeps_legacy_schema_without_burst_keys(
        self, tmp_path: Path
    ) -> None:
        path, _ = self._record_stream(tmp_path, "off.jsonl", burst_tracker=None)
        records = _read_stream(path)
        assert records[0]["type"] == "header"
        assert records[0]["format"] == ACTIVITY_STREAM_FORMAT
        frames = [r for r in records if r["type"] == "frame"]
        assert len(frames) == 6
        footer = records[-1]
        assert footer["type"] == "footer"
        assert footer["complete"] is True
        assert footer["frame_count"] == 6
        assert "burst_metrics" not in footer
        for index, frame in enumerate(frames):
            expected = (
                _FRAME_KEYS_WITH_STATE if index % 2 == 0 else _FRAME_KEYS_BASE
            )
            assert set(frame) == expected, f"schema drift at frame {index}"

    def test_flag_on_footer_carries_burst_snapshot(self, tmp_path: Path) -> None:
        tracker = BurstFractionTracker(window_tokens=4)
        path, _ = self._record_stream(
            tmp_path, "on.jsonl", burst_tracker=tracker
        )
        records = _read_stream(path)
        footer = records[-1]
        assert footer["type"] == "footer"
        metrics = footer["burst_metrics"]
        assert metrics["format"] == BURST_METRICS_FORMAT
        assert metrics["steps_observed"] == 6
        assert metrics["window_tokens"] == 4
        assert 0.0 <= metrics["burst_fraction"] <= 1.0
        assert metrics["open_window"]["tokens"] == 2
        # Frames stay untouched when tracking is enabled.
        frames = [r for r in records if r["type"] == "frame"]
        for index, frame in enumerate(frames):
            expected = (
                _FRAME_KEYS_WITH_STATE if index % 2 == 0 else _FRAME_KEYS_BASE
            )
            assert set(frame) == expected

    def test_default_recorder_constructs_without_tracker(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "default.jsonl"
        recorder = ActivityRecorder(path, run_id="live/default")
        assert recorder._burst_tracker is None
