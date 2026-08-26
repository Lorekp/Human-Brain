"""C1 diagnostic scaffold - r3 segment-growth pilot (prereg §4-§8).

HARD NO-GO (2026-08-26): the long cohort entry point is mechanically blocked
until the neuron-only substrate and scientific-control failures in
`docs/research/r3-c1-prelaunch-audit.md` are repaired. `--dry-run` remains
available for reproducing and localizing the documented wiring defects; its
success is not launch-readiness evidence.

Runs the complete C1 leg of ``docs/research/r3-segment-growth-preregistration.md``
on branch ``codex/path2-pilot``: corpus assert-or-STOP, arms A1/A2/A3 at
N = 131,072 cells via the frozen glue stack
(``humanbrain.r3_pilot_glue`` pool mapping + observation-sink replay driver +
``humanbrain.segment_growth.SegmentGrowthTracker`` mechanics), the epoch-0
structural sweep, E = 3 ordered exposure passes over TRAIN, ONE final
evaluation sweep over TRAIN u HELD-OUT, then gates

    G-S1  engagement : TRAIN f_pred(final) - 0           >= 0.05, Wilson LB > 0
    G-S2  transfer   : HELD-OUT f_pred(final) - baseline >= 0.05, Wilson LB > 0
                       (baseline_burst = A2 HELD-OUT f_pred, asserted 0)
    G-S3  attribution: A1 HELD-OUT - A3 HELD-OUT          >= 0.05, Newcombe > 0

with stop-on-failure between gates and the mechanical verdict mapping of
prereg §7.  Resume-safe checkpointing per arm at sweep boundaries; hard
wall-clock cap (default 15 h, prereg §8); telemetry via the burst tracker
(per-sweep/per-book burst fractions, SinkSpikeRecorder health pre-flight).

------------------------------------------------------------------------------
W_{t-1} DECISION (recorded before launch, per dispatch requirement)
------------------------------------------------------------------------------
Prereg §2 defines W_{t-1} as "elected winners of step t-1 for bursting pools;
fired (depolarized) cells themselves for pools that took the predicted path" -
i.e. tracker-internal election.  The G-S0 manifest instead records
``winner_set_source = "observation-sink spike stream"``: the driver feeds the
LANE's spiking set of the previous token as W_{t-1}.  C1 must pick ONE.

**C1 RE-AFFIRMS THE LANE-SPIKE-STREAM DEFINITION** (gs0_health_run.json,
``winner_set_source`` field).  Reasons, all mechanical:

1. Tracker-internal election is arithmetically sterile at the frozen
   constants.  One token pool arrives per step, so a tracker-internal
   W_{t-1} has exactly ONE element on every burst step (the single elected
   winner).  Candidacy requires >= theta_m = 10 potential-active synapses
   against W_{t-1}; any segment can hold at most |W_{t-1}| = 1 contact inside
   W, so no candidate can EVER exist; every burst then allocates a brand-new
   least-used-cell segment whose contacts enter at initialPermanence = 0.21
   < connectedPermanence = 0.50 and can never be reinforced (reinforcement
   requires candidacy).  f_pred is structurally 0 and G-S1 fails by
   construction - the cohort would spend itself proving arithmetic.
2. The frozen G-S0 fixtures are unsatisfiable under tracker-internal W:
   fixture (i) "a repeated bigram becomes predicted after one exposure"
   PASSED at G-S0 only under the lane-spike-stream definition - the wiring
   actually validated there (bit-identical reruns, A2 structural-zero
   tripwire, spike-rate pre-flight all green).
3. Under the re-affirmed definition the A3 control stays meaningful: A1 grows
   presynaptic contacts toward the winning lane representation of step t-1;
   A3 samples an equal mass of targets from the active-cell population,
   destroying persistent-context binding while keeping generic co-occurrence
   growth - exactly the contrast G-S3 scores.

A3 OPERATIONALIZATION (disclosed deviation, letter vs purpose): with W_{t-1}
:= the lane-spike stream, prereg §4's literal A3 sentence ("sampled uniformly
from the active cells of step t-1 instead of the elected winner set")
degenerates - under this definition those are the same set and A3 would equal
A1, making G-S3 vacuous.  A3 therefore implements the stated PURPOSE:
presynaptic growth targets are drawn uniformly WITHOUT replacement from the
cumulative union of lane-active cells observed so far (size matched through
the same min(budget, available) rule), seeded by the construction seed with
the tracker's own rng key pattern ``(construction_seed, step_index,
cell_id)``.  Recorded here, before any C1 token is replayed, per the repo's
open-deviation discipline.

MEASURED PERFORMANCE ENVELOPE (2026-08-26, see C1_READINESS.md): the native
kernel selects its CUDA backend at N = 131,072 (RTX 4070 SUPER, ~193 lane
steps/s), but (a) the FROZEN default lane constants go silent at that
population size - spike-rate pre-flight FAIL, mean winner-set size 0 - and
(b) the reference tracker's global recognition scan costs ~30 ms/step at just
1,000 live segments, growing linearly.  This runner therefore enforces the
prereg §6 spike-rate pre-flight as a HARD STOP before any long leg, anchors
its replay loop to the glue driver bit-for-bit on a prefix, and projects
remaining wall-clock against the 15 h cap continuously.  As measured, the C1
leg is BLOCKED; read C1_READINESS.md before launching.

No forbidden trainable components anywhere (prereg §1 tripwire);
determinism: same (seed, corpus, constants) => bit-identical arrays/logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for _p in (str(SRC), str(REPO_ROOT / "experiments" / "r3_pilot")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from humanbrain.r3_pilot_glue import (  # noqa: E402
    CONCEPT_ID_CEILING,
    CONSTRUCTION_SEED,
    SinkSpikeRecorder,
    assert_feasibility,
    pool_cells_for_id,
    required_pool_cells,
    run_arm,
)
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig  # noqa: E402
from humanbrain.segment_growth import SegmentGrowthTracker, _Segment  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen literals (prereg + tokenizer freeze; never tune)
# ---------------------------------------------------------------------------

C1_LANE_CELLS = 131_072           # "C1 @ 128k" (R2 128k manifest convention)
LANE_CELLS_FIXTURE = 96           # N=96 fixture leg (G-S0 convention) for --dry-run
TOKEN_CELLS = int(CONCEPT_ID_CEILING)  # concept band ceiling, ids [1024, 1536)
EXPOSURE_PASSES = 3               # prereg §5: E = 3 ordered exposure passes
DELTA = 0.05                      # absolute margin on every gate
WILSON_Z = 1.959964               # prereg §6 literal z
WALL_CLOCK_CAP_HOURS = 15.0       # prereg §8 hard cap
N_TRAIN_BOOKS = 49
ARMS = ("A1", "A2", "A3")         # prereg §4: no other arms, no post-hoc additions
SPIKE_RATE_FLOOR_FRACTION = 0.001  # prereg §6 pre-flight floor (analytic ceiling fraction)
WINDOW_ROWS = 4096                # rolling telemetry window (O(WINDOW_ROWS), not O(steps))
PREFLIGHT_PREFIX_TOKENS = 4096    # prereg §6 spike-rate pre-flight evaluated after this many pass-1 steps
DRY_RUN_PREFIX_TOKENS = 256       # --dry-run replay length per arm
HELD_OUT_BOOKS = (
    "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
    "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
)
CONCEPT_BAND_MIN = 1024

PHASE_EPOCH0 = "epoch0"
PHASE_EXPOSURE = "exposure"
PHASE_FINAL = "final"
PHASE_DONE = "done"

EXIT_OK = 0
EXIT_STOP = 2                     # protocol STOP (corpus drift, gate failure w/ verdict, ...)
EXIT_WALLCLOCK_ABORT = 3          # cap exceeded; partial logs reported
EXIT_RESUME_INVALID = 4

C1_LAUNCH_BLOCK_REASON = (
    "C1 launch blocked by docs/research/r3-c1-prelaunch-audit.md: the current "
    "GLIF5+Python-tracker scaffold is not the frozen GLIF3 plateau/race "
    "experiment and learns during held-out scoring"
)

TOKENIZER_TAG = (
    "BytePhraseTokenizer(max_phrases=0,min_frequency=3)"
    "+ConceptTokenizer(freq>=50,MI>=5,cap512)"
)


class CohortStop(RuntimeError):
    """Mechanical STOP (assert-or-STOP discipline)."""

    def __init__(self, reason: str, kind: str = "STOP") -> None:
        super().__init__(reason)
        self.kind = kind


# ---------------------------------------------------------------------------
# Corpus layer
# ---------------------------------------------------------------------------


def segment_verses(book_text: str) -> list[str]:
    """Verse segmentation identical to the validated G-S0 leg."""
    import run_gs0_health as gs0

    return gs0.segment_verses(book_text)


def load_corpus() -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    """Assert-or-STOP the pinned corpus and PRIMARY gating split.

    Returns (books, provenance); books = [(short_name, role, text)] canon order.
    """
    freeze_dir = Path(os.environ.get("TEMP", os.environ.get("LOCALAPPDATA", ""))) / "hb_bible_freeze"
    freeze = json.loads((freeze_dir / "FREEZE-CANDIDATE-v3-BIBLE.json").read_text(encoding="utf-8"))
    spans = json.loads((freeze_dir / "evidence" / "pg10_book_spans_v3.json").read_text(encoding="utf-8"))
    trimmed = (freeze_dir / "evidence_pg10.trimmed.bin").read_bytes()

    digest = hashlib.sha256(trimmed).hexdigest()
    expected = freeze["corpus"]["sha256_trimmed"]
    if digest != expected:
        raise CohortStop(f"corpus drift: sha256 {digest} != frozen {expected}")

    books: list[tuple[str, str, str]] = []
    for book in spans["books"]:
        start, end = book["span_offsets"]
        books.append((book["short_name"], book["role"], trimmed[start:end].decode("utf-8")))

    train = [b for b in books if b[1] == "train"]
    held = [b for b in books if b[1] != "train"]
    if len(books) != 66:
        raise CohortStop(f"split drift: {len(books)} books != 66")
    if len(train) != N_TRAIN_BOOKS:
        raise CohortStop(f"split drift: {len(train)} TRAIN books != {N_TRAIN_BOOKS}")
    if tuple(b[0] for b in held) != HELD_OUT_BOOKS:
        raise CohortStop(
            f"split drift: held-out {tuple(b[0] for b in held)} != frozen {HELD_OUT_BOOKS}"
        )

    prov = {
        "candidate_id": freeze["candidate_id"],
        "trimmed_sha256": digest,
        "sha256_match": True,
        "train_books": len(train),
        "held_out_books": len(held),
        "held_out_names": [b[0] for b in held],
    }
    return books, prov


def build_concept_stack(train_texts: list[str]):
    """Frozen tokenizer stack, byte-identical recipe to the G-S0 leg."""
    from humanbrain.concept_tokens import ConceptConfig, ConceptTokenizer, mine_concepts_detailed
    from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

    base = BytePhraseTokenizer.fit(train_texts, TokenizerConfig(max_phrases=0, min_frequency=3))
    if base.vocab_size != 256:
        raise CohortStop(f"base vocab {base.vocab_size} != 256")
    candidates = mine_concepts_detailed(train_texts, base, ConceptConfig())
    stack = ConceptTokenizer(base, tuple(c.payload for c in candidates))
    return stack, {"base_vocab": int(base.vocab_size), "concepts": len(candidates),
                   "tokenizer": TOKENIZER_TAG}


def encode_side(
    books: Sequence[tuple[str, str, str]], side: str, stack
) -> tuple[list[int], list[dict[str, Any]], str]:
    """Encode every verse of one side, canon order.

    Returns (tokens, per-book offset records, chunked sha256 over the
    comma-joined id string - the canonical form the G-S0 slice hashed).
    """
    h = hashlib.sha256()
    tokens: list[int] = []
    records: list[dict[str, Any]] = []
    buf: list[str] = []
    for name, role, text in books:
        if role != side:
            continue
        start = len(tokens)
        ids: list[int] = []
        for verse in segment_verses(text):
            ids.extend(stack.encode(verse))
        tokens.extend(ids)
        records.append({"book": name, "role": role, "start": start, "end": len(tokens)})
        buf.extend(str(i) for i in ids)
        buf.append(",")
        if len(buf) >= (1 << 20):
            h.update("".join(buf).encode("utf-8"))
            buf.clear()
    if buf:
        h.update("".join(buf).encode("utf-8"))
    return tokens, records, h.hexdigest()


def build_token_caches(output_dir: Path) -> dict[str, Any]:
    """Load or create the full-side concept-token streams under output_dir."""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "streams.json"

    books, prov = load_corpus()
    train_books = [b for b in books if b[1] == "train"]
    stack_info: dict[str, Any] = {}
    streams: dict[str, Any] = {}

    cached = None
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("trimmed_sha256") != prov["trimmed_sha256"]:
            cached = None  # corpus changed underneath => re-derive

    tokens_by_side: dict[str, np.ndarray] = {}
    for side in ("train", "val"):
        npy = cache_dir / f"{side}_tokens.npy"
        rec = (cached or {}).get("sides", {}).get(side) if cached else None
        if rec and npy.exists():
            arr = np.load(npy)
            if hashlib.sha256(arr.tobytes()).hexdigest() == rec["npy_sha256"]:
                tokens_by_side[side] = arr
                streams[side] = rec
                continue
        if not stack_info:
            stack, stack_info = build_concept_stack([t for _, _, t in train_books])
        tokens, book_records, stream_sha = encode_side(books, side, stack)
        arr = np.asarray(tokens, dtype=np.int32)
        np.save(npy, arr)
        streams[side] = {
            "tokens": int(arr.size),
            "verses": sum(1 for r in book_records),
            "books": len(book_records),
            "distinct_ids": int(np.unique(arr).size),
            "max_token_id": int(arr.max()),
            "concept_band_ids": int((np.unique(arr) >= CONCEPT_BAND_MIN).sum()),
            "stream_sha256_comma_joined": stream_sha,
            "npy_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "book_records": book_records,
            **stack_info,
        }
        tokens_by_side[side] = arr

    out = {
        "candidate_id": prov["candidate_id"],
        "trimmed_sha256": prov["trimmed_sha256"],
        "sha256_match": prov["sha256_match"],
        "train_books": prov["train_books"],
        "held_out_books": prov["held_out_books"],
        "held_out_names": prov["held_out_names"],
        "sides": streams,
        "feasibility": feasibility_report(),
    }
    manifest_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    max_id = max(int(s["max_token_id"]) for s in streams.values())
    configured = C1_LANE_CELLS + required_pool_cells(max_id)
    try:
        assert_feasibility(max_id, configured)
    except RuntimeError as exc:
        raise CohortStop(str(exc)) from exc
    return out


def feasibility_report() -> dict[str, Any]:
    ceiling = TOKEN_CELLS - 1
    pool_cells = required_pool_cells(ceiling)
    configured = C1_LANE_CELLS + pool_cells
    return {
        "lane_cells": C1_LANE_CELLS,
        "pool_cells": int(pool_cells),
        "configured_cells": int(configured),
        "literal_(max_id+1)*k": int((ceiling + 1) * K_POOL_CONST()),
        "assert_passed_at_build": True,
    }


def K_POOL_CONST() -> int:
    return SegmentGrowthTracker.CELLS_PER_POOL


# ---------------------------------------------------------------------------
# Replay engine (mirrors glue.run_arm step semantics exactly)
# ---------------------------------------------------------------------------


class RollingSinkRecorder:
    """Observation-sink callback keeping ONLY rolling statistics.

    Mirrors the glue SinkSpikeRecorder API used by the loop
    (``winner_sets[-1]``, ``spike_rate_health``, ``mean_activity_density``)
    without per-step history: at C1 stream lengths (millions of steps x
    large winner sets) storing history cannot fit in memory.  Density
    accounting identical: len(winners) / (cells * substeps).
    """

    def __init__(self, lane_cells: int) -> None:
        self.lane_cells = int(lane_cells)
        self.last_winners: tuple[int, ...] = ()
        self._density_sum = 0.0
        self._steps = 0
        self._empty_w_steps = 0
        self._winner_mass = 0

    def __call__(self, circuit: RecurrentCellCircuit, step) -> None:
        counts = np.asarray(circuit._token_spike_counts, dtype=np.uint32)
        winners = tuple(int(g) for g in np.flatnonzero(counts))
        density = len(winners) / float(counts.size * max(int(step.simulated_steps), 1))
        self.last_winners = winners
        self._density_sum += density
        self._steps += 1
        self._winner_mass += len(winners)
        if not winners:
            self._empty_w_steps += 1

    @property
    def winner_sets(self) -> list[tuple[int, ...]]:
        """Compatibility shim: callers index [-1] only."""
        return [self.last_winners]

    def mean_winner_set_size(self) -> float:
        if self._steps == 0:
            return 0.0
        return self._winner_mass / self._steps

    def empty_winner_fraction(self) -> float:
        if self._steps == 0:
            return 0.0
        return self._empty_w_steps / self._steps

    def mean_activity_density(self) -> float:
        if self._steps == 0:
            raise ValueError("no tokens observed")
        return self._density_sum / self._steps

    def spike_rate_health(self, floor_fraction: float = 0.001) -> dict[str, Any]:
        """Prereg §6 pre-flight: population rate >= 0.1% of analytic ceiling."""
        if self._steps == 0:
            return {"passed": False, "reason": "no tokens observed"}
        mean_density = self.mean_activity_density()
        return {
            "mean_activity_density": mean_density,
            "floor_fraction": floor_fraction,
            "mean_winner_set_size": self.mean_winner_set_size(),
            "empty_winner_fraction": self.empty_winner_fraction(),
            "passed": bool(mean_density >= floor_fraction),
        }


class ShuffledWinnerTracker(SegmentGrowthTracker):
    """A3 control (see module docstring disclosure): presynaptic growth
    targets sampled uniformly WITHOUT replacement from the cumulative union
    of lane-active cells instead of the previous step's winner set."""

    def __init__(self, pools, *, lane_cells: int, **kwargs: Any) -> None:
        super().__init__(pools, **kwargs)
        self.lane_cells = int(lane_cells)
        self._active_union = np.zeros(self.lane_cells, dtype=bool)

    def note_active(self, winner_set: Sequence[int]) -> None:
        if winner_set:
            idx = np.fromiter((int(x) for x in winner_set), dtype=np.int64)
            idx = idx[idx < self.lane_cells]
            self._active_union[idx] = True

    def union_size(self) -> int:
        return int(np.count_nonzero(self._active_union))

    def _fill_toward(self, cell: int, seg: Any, w: np.ndarray) -> None:  # type: ignore[override]
        budget = self._max_synapses_per_segment - len(seg.presyn)
        if budget <= 0:
            return
        universe = np.flatnonzero(self._active_union)
        if universe.size == 0:
            return
        existing = set(seg.presyn)
        if existing:
            available = universe[~np.isin(universe, np.asarray(sorted(existing), dtype=np.int64))]
        else:
            available = universe
        if available.size == 0:
            return
        n_new = min(budget, int(available.size))
        rng = np.random.default_rng([self._construction_seed, self._step_index, int(cell)])
        chosen = np.sort(rng.choice(available, size=n_new, replace=False))
        seg.presyn.extend(int(c) for c in chosen)
        seg.perm.extend([self._initial_permanence] * n_new)


def make_tracker(arm: str, pools, lane_cells: int) -> SegmentGrowthTracker:
    if arm == "A3":
        return ShuffledWinnerTracker(
            pools, lane_cells=lane_cells, growth_on=True,
            construction_seed=CONSTRUCTION_SEED,
        )
    return SegmentGrowthTracker(
        pools, growth_on=(arm != "A2"), construction_seed=CONSTRUCTION_SEED,
    )


def telemetry_sha256(rows: Sequence[Sequence[Any]]) -> str:
    """Bit-exact replica of the glue driver's canonical row hashing."""
    payload = [
        [
            int(r[0]), int(r[1]), bool(r[2]),
            (int(r[3]) if r[3] is not None else None),
            [int(x) for x in r[4]],
            [[int(c), int(s)] for c, s in r[5]],
            bool(r[6]), int(r[7]), int(r[8]),
        ]
        for r in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def permanences_sha256(tracker: SegmentGrowthTracker) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(tracker.all_permanences()).tobytes()
    ).hexdigest()


class ArmEngine:
    """One arm's circuit + tracker + rolling telemetry, glue-semantics-faithful."""

    def __init__(self, arm: str, distinct_ids: Sequence[int], lane_cells: int) -> None:
        self.arm = arm
        self.lane_cells = int(lane_cells)
        offset = lane_cells  # pools live above lane gids; namespaces disjoint
        pools = {int(t): pool_cells_for_id(int(t), cell_offset=offset) for t in distinct_ids}
        self.tracker = make_tracker(arm, pools, lane_cells)
        self.circuit = RecurrentCellCircuit(
            TOKEN_CELLS, config=RecurrentCircuitConfig(reservoir_cells=lane_cells)
        )
        self.recorder = RollingSinkRecorder(lane_cells)
        # R0-c Thread-1 seam: attach via the property setter (see glue docstring).
        self.circuit.observation_sink = self.recorder
        self.w_prev: tuple[int, ...] = ()
        # Burst-tracker telemetry aggregates (prereg §5/§7 secondary instrumentation)
        self.sweeps: list[dict[str, Any]] = []
        self.book_rows: list[dict[str, Any]] = []
        self.predicted_total = 0
        self.burst_total = 0
        self.allocations_concept_band = 0
        self.allocations_base_band = 0
        self.punished_events = 0
        self.removed_contacts = 0
        self.elections = 0
        self.steps_total = 0
        self.window: deque[tuple[Any, ...]] = deque(maxlen=WINDOW_ROWS)
        # Per-sweep packed predicted-path counters (exact gate scoring at
        # O(1 byte)/step); the orchestrator opens one per protocol sweep.
        self.counters: dict[str, SweepCounter] = {}
        self.active_label: str | None = None
        self.stream_hash = hashlib.sha256()

    def open_counter(self, label: str) -> None:
        self.counters[label] = SweepCounter()
        self.active_label = label

    @staticmethod
    def counter_key(label: str) -> str:
        return "c_" + label.replace("::", "_").replace("-", "_")

    def counter_arrays(self) -> dict[str, np.ndarray]:
        return {self.counter_key(k): v.finalize_array() for k, v in self.counters.items()}

    def load_counters(self, labels: Sequence[str], arrays: dict[str, np.ndarray],
                      active_label: str | None) -> None:
        self.counters = {}
        for label in labels:
            key = self.counter_key(label)
            if key not in arrays:
                raise CohortStop(
                    f"checkpoint counters missing array {key!r} => resume invalid",
                    kind="RESUME_INVALID",
                )
            counter = SweepCounter()
            counter.restore_array(arrays[key])
            self.counters[label] = counter
        self.active_label = active_label

    # -- core step (order of operations identical to glue.run_arm) ----------
    def step(self, token_id: int, pass_label: str) -> dict[str, Any]:
        tok = int(token_id)
        self.circuit.observe(tok)
        result = self.tracker.step(tok, self.w_prev)
        new_w = self.recorder.last_winners
        if isinstance(self.tracker, ShuffledWinnerTracker):
            self.tracker.note_active(new_w)
        predicted = bool(result.fired_cells)
        burst = bool(result.burst)
        self.predicted_total += predicted
        self.burst_total += burst
        self.elections += 1 if result.elected_winner is not None else 0
        self.punished_events += len(result.punished_segments)
        self.removed_contacts += int(result.removed_contacts)
        if result.allocated_segment:
            if tok >= CONCEPT_BAND_MIN:
                self.allocations_concept_band += 1
            else:
                self.allocations_base_band += 1
        if self.active_label is not None:
            self.counters[self.active_label].add(predicted)
        row = (
            result.step_index, tok, burst,
            (int(result.elected_winner) if result.elected_winner is not None else None),
            tuple(int(c) for c in result.fired_cells),
            tuple((int(c), int(s)) for c, s in result.punished_segments),
            bool(result.allocated_segment), int(result.removed_contacts),
            len(self.w_prev),
        )
        self.window.append(row)
        self.last_row = row
        self.stream_hash.update(_row_bytes(row))
        self.steps_total += 1
        self.w_prev = new_w
        return {"predicted": predicted, "burst": burst}

    # -- sweep driver --------------------------------------------------------
    def run_stream(
        self,
        tokens: np.ndarray,
        book_records: Sequence[dict[str, Any]],
        *,
        sweep_label: str,
        deadline: float | None,
        on_book_done=None,
        start_book: int = 0,
        health_gate=None,
        health_gate_at_step: int | None = None,
    ) -> int:
        """Replay ``tokens`` (canon order) honoring book-boundary checkpoints.

        ``health_gate(engine)`` fires exactly once, mid-book, once
        ``steps_total`` reaches ``health_gate_at_step`` - the prereg §6
        spike-rate hard stop early in exposure pass 1.  It only reads rolling
        telemetry and may raise; replay semantics are untouched either way.
        Each book's telemetry rows are hashed into a per-book row digest so
        determinism stays auditable across resume boundaries.
        """
        gate_armed = health_gate is not None
        for bi in range(start_book, len(book_records)):
            rec = book_records[bi]
            lo, hi = rec["start"], rec["end"]
            t0 = time.time()
            pred = burst = 0
            book_hash = hashlib.sha256()
            for i in range(lo, hi):
                out = self.step(int(tokens[i]), sweep_label)
                pred += out["predicted"]
                burst += out["burst"]
                book_hash.update(_row_bytes(self.last_row))
                if gate_armed and self.steps_total >= (health_gate_at_step or 0):
                    gate_armed = False
                    health_gate(self)
            self.book_rows.append({
                "sweep": sweep_label, "book": rec["book"], "role": rec["role"],
                "steps": hi - lo, "predicted": pred, "burst": burst,
                "burst_fraction": burst / max(hi - lo, 1),
                "row_digest": book_hash.hexdigest(),
                "wall_s": round(time.time() - t0, 2),
            })
            if on_book_done is not None:
                on_book_done(bi)
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"wall-clock cap exceeded during {sweep_label} ({rec['book']})")
        return len(tokens)

    def sweep_summary(self, label: str, side_masks: dict[str, np.ndarray]) -> dict[str, Any]:
        """Aggregate one completed sweep using its per-step packed counter.

        ``side_masks`` maps 'train'/'heldout' to boolean masks over that
        sweep's step axis (the orchestrator builds them from the sweep's book
        records), generalizing contiguous slices: canon order interleaves the
        two roles, so each side of the final evaluation sweep is a union of
        step ranges.
        """
        if label not in self.counters:
            raise KeyError(f"no packed counter opened for sweep {label!r}")
        arr = self.counters[label].finalize_array()
        out: dict[str, Any] = {
            "label": label,
            "steps": int(arr.size),
            "predicted": int(arr.sum()),
            "f_pred": float(arr.mean()) if arr.size else 0.0,
        }
        for side in sorted(side_masks):
            mask = np.asarray(side_masks[side], dtype=bool)
            if mask.size != arr.size:
                raise CohortStop(
                    f"sweep {label!r}: mask {side!r} covers {mask.size} steps "
                    f"but the sweep recorded {arr.size}"
                )
            seg = arr[mask]
            n, x = int(seg.size), int(seg.sum())
            _, lb, ub = wilson_interval(x, n)
            out[side] = {
                "steps": n,
                "predicted": x,
                "f_pred": (x / n) if n else 0.0,
                "burst": n - x,
                "burst_fraction": ((n - x) / n) if n else 0.0,
                "wilson_lb": lb,
                "wilson_ub": ub,
            }
        return out


def _row_bytes(row: Sequence[Any]) -> bytes:
    parts = [str(int(row[0])), str(int(row[1])), "1" if row[2] else "0",
             "-" if row[3] is None else str(int(row[3])),
             "|".join(str(int(x)) for x in row[4]),
             "|".join(f"{int(c)}.{int(s)}" for c, s in row[5]),
             "1" if row[6] else "0", str(int(row[7])), str(int(row[8]))]
    return (",".join(parts) + "\n").encode("ascii")


class SweepCounter:
    """Per-sweep packed counters: O(steps) memory, exact gate scoring.

    One byte per replay step (bytearray, ~1 MB per million steps) recording
    whether the arriving token's pool took the predicted path.  Survives
    checkpoint round-trips via :meth:`restore_array` so gates score exactly
    the same bits an uninterrupted run would have scored.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def add(self, flag: bool) -> None:
        self._buf.append(1 if flag else 0)

    def __len__(self) -> int:
        return len(self._buf)

    def finalize_array(self) -> np.ndarray:
        return np.frombuffer(bytes(self._buf), dtype=np.uint8)

    def restore_array(self, arr: np.ndarray) -> None:
        self._buf = bytearray(np.asarray(arr, dtype=np.uint8).tobytes())


# ---------------------------------------------------------------------------
# Checkpointing (resume-safe, per arm, sweep granularity)
# ---------------------------------------------------------------------------


class ArmCheckpointer:
    """Serializes one arm's replay state at sweep boundaries.

    Layout under <output_dir>/checkpoints/<arm>/:
      tracker.npz  flattened segments (cells, counts, lengths, presyn, perm),
                   wins, step_index, active-union bits (A3)
      lane.npz     enveloped lane state (capture_cell_synapse_state payload)
      meta.json    cursor, digests, summaries, w_prev, sweep counters file refs
    Writes are atomic (tmp + rename).
    """

    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- flatten / restore ---------------------------------------------------
    @staticmethod
    def flatten_tracker(tracker: SegmentGrowthTracker) -> dict[str, np.ndarray]:
        cells: list[int] = []
        counts: list[int] = []
        lengths: list[int] = []
        presyn: list[int] = []
        perm: list[float] = []
        for cell in sorted(tracker._segments):  # package-internal seam, stable order
            segs = tracker._segments[cell]
            cells.append(int(cell))
            counts.append(len(segs))
            for seg in segs:
                lengths.append(len(seg.presyn))
                presyn.extend(seg.presyn)
                perm.extend(seg.perm)
        win_cells = sorted(tracker._wins)
        out = {
            "seg_cells": np.asarray(cells, dtype=np.int64),
            "seg_counts": np.asarray(counts, dtype=np.int64),
            "seg_lengths": np.asarray(lengths, dtype=np.int64),
            "seg_presyn": np.asarray(presyn, dtype=np.int64),
            "seg_perm": np.asarray(perm, dtype=np.float64),
            "win_cells": np.asarray(win_cells, dtype=np.int64),
            "win_counts": np.asarray([tracker._wins[c] for c in win_cells], dtype=np.int64),
            "step_index": np.asarray(tracker.num_steps, dtype=np.int64),
        }
        if isinstance(tracker, ShuffledWinnerTracker):
            out["union_bits"] = np.packbits(tracker._active_union)
        return out

    @staticmethod
    def restore_tracker(arm: str, arrays: dict[str, np.ndarray], pools,
                        lane_cells: int = C1_LANE_CELLS) -> SegmentGrowthTracker:
        if "union_bits" in arrays:
            tracker: SegmentGrowthTracker = ShuffledWinnerTracker(
                pools, lane_cells=lane_cells, growth_on=(arm != "A2"),
                construction_seed=CONSTRUCTION_SEED,
            )
            bits = arrays["union_bits"]
            tracker._active_union = np.unpackbits(bits, count=lane_cells).astype(bool)
        else:
            tracker = SegmentGrowthTracker(
                pools, growth_on=(arm != "A2"), construction_seed=CONSTRUCTION_SEED
            )
        # flatten_tracker wrote lengths/presyn/perm as one flat stream ordered by
        # (cell asc, segment order within cell); consume with a GLOBAL segment
        # ordinal - a per-cell index would re-read early entries and corrupt
        # every cell after the first.
        segments: dict[int, list[_Segment]] = {}
        pos = 0
        seg_ordinal = 0
        for cell, count in zip(arrays["seg_cells"].tolist(), arrays["seg_counts"].tolist()):
            seg_list: list[_Segment] = []
            for _ in range(int(count)):
                length = int(arrays["seg_lengths"][seg_ordinal])
                seg_ordinal += 1
                seg = _Segment()
                seg.presyn = [int(x) for x in arrays["seg_presyn"][pos:pos + length]]
                seg.perm = [float(x) for x in arrays["seg_perm"][pos:pos + length]]
                pos += length
                seg_list.append(seg)
            segments[int(cell)] = seg_list
        tracker._segments = segments
        wins = {}
        for c, n in zip(arrays["win_cells"].tolist(), arrays["win_counts"].tolist()):
            wins[int(c)] = int(n)
        tracker._wins = wins
        tracker._step_index = int(arrays["step_index"])
        return tracker

    def save(self, arm: str, engine: ArmEngine, cursor: dict[str, Any],
             summaries: dict[str, Any], digests: dict[str, str],
             counter_arrays: dict[str, np.ndarray],
             context: dict[str, Any] | None = None,
             counter_labels: Sequence[str] = ()) -> str:
        arm_dir = self.dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)

        # C1 never applies readout plasticity (no strengthen/grow_sequence
        # calls anywhere in this runner), but capture_cell_synapse_state gates
        # on learning_frozen. freeze_learning() itself refuses zero-growth
        # circuits ("cannot freeze readout before growth"), which C1 always is,
        # so set the flag through the same internal seam that
        # RecurrentCellCircuit.restore_cell_synapse_state uses on rebuild.
        if not engine.circuit.learning_frozen:
            engine.circuit._learning_frozen = True

        arrays = self.flatten_tracker(engine.tracker)
        # np.savez_compressed APPENDS ".npz" to string paths, so pass file
        # objects for the .tmp names; the lane envelope API demands a ".npz"
        # suffix itself, hence "lane.tmp.npz" -> rename to "lane.npz".
        with open(arm_dir / "tracker.tmp.npz", "wb") as fh:
            np.savez_compressed(fh, **arrays)
        (arm_dir / "tracker.tmp.npz").replace(arm_dir / "tracker.npz")

        state = engine.circuit.capture_cell_synapse_state(include_ridge_statistics=False)
        state.save_enveloped_checkpoint(
            arm_dir / "lane.tmp.npz", envelope={"arm": arm, "saved_by": "run_c1_cohort.py"}
        )
        (arm_dir / "lane.tmp.npz").replace(arm_dir / "lane.npz")

        counters_path = arm_dir / "counters.npz"
        with open(counters_path.with_suffix(".tmp.npz"), "wb") as fh:
            np.savez_compressed(fh, **counter_arrays)
        counters_path.with_suffix(".tmp.npz").replace(counters_path)

        meta = {
            "arm": arm,
            "cursor": cursor,
            "w_prev": [int(x) for x in engine.w_prev],
            "steps_total": int(engine.steps_total),
            "active_counter_label": engine.active_label,
            "counter_labels": list(counter_labels),
            "summaries": summaries,
            "digests": digests,
            "context": dict(context or {}),
            "construction_seed": CONSTRUCTION_SEED,
            "saved_unix": time.time(),
        }
        tmp_json = arm_dir / "meta.json.tmp"
        tmp_json.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        tmp_json.replace(arm_dir / "meta.json")
        return sha256_file(arm_dir / "meta.json")

    def load_meta(self, arm: str) -> dict[str, Any]:
        return json.loads((self.dir / arm / "meta.json").read_text(encoding="utf-8"))

    def has_checkpoint(self, arm: str) -> bool:
        return (self.dir / arm / "meta.json").exists()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Gate statistics (prereg §6: Wilson z = 1.959964; Newcombe MOVER-D)
# ---------------------------------------------------------------------------


def wilson_interval(x: int, n: int, z: float = WILSON_Z) -> tuple[float, float, float]:
    """Wilson score interval (point, lower, upper) for x successes in n."""
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def newcombe_difference(x1: int, n1: int, x2: int, n2: int,
                        z: float = WILSON_Z) -> tuple[float, float, float]:
    """Newcombe MOVER-D interval for p1 - p2 built from the two Wilsons."""
    p1, l1, u1 = wilson_interval(x1, n1, z)
    p2, l2, u2 = wilson_interval(x2, n2, z)
    d = p1 - p2
    lb = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    ub = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lb, ub


def evaluate_gates(final_by_arm: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Score G-S1/G-S2/G-S3 on the final sweep exactly as prereg §6 freezes.

    ``final_by_arm`` maps arm -> {"train": ..., "heldout": ...} sweep summaries
    (ArmEngine.sweep_summary output).  Verdict mapping is §7 verbatim; note the
    S1-AND-S2-but-not-S3 cell has NO §7 terminal label (§7's four labels cover
    C2 outcomes plus the two anticipated C1-fail modes), so the ladder halts
    mechanically with C1_FAIL_ATTRIBUTION_NO_C2 - reported once, never retried.
    """
    a1_tr = final_by_arm["A1"]["train"]
    a1_ho = final_by_arm["A1"]["heldout"]
    a2_ho = final_by_arm["A2"]["heldout"]
    a3_ho = final_by_arm["A3"]["heldout"]

    baseline_burst = a2_ho["f_pred"]
    if a2_ho["predicted"] != 0:
        raise CohortStop(
            "G-S2 precondition broken: A2 HELD-OUT predicted-path steps "
            f"{a2_ho['predicted']} != 0 => leakage/bug (prereg §4 structural zero)"
        )

    _, s1_lb, _ = wilson_interval(a1_tr["predicted"], a1_tr["steps"])
    g_s1_point = a1_tr["f_pred"]
    g_s1 = bool(g_s1_point >= DELTA and s1_lb > 0.0)

    _, s2_lb, _ = wilson_interval(a1_ho["predicted"], a1_ho["steps"])
    g_s2_point = a1_ho["f_pred"] - baseline_burst
    g_s2 = bool(g_s2_point >= DELTA and s2_lb > 0.0)

    nd, nd_lb, nd_ub = newcombe_difference(
        a1_ho["predicted"], a1_ho["steps"], a3_ho["predicted"], a3_ho["steps"]
    )
    g_s3_point = a1_ho["f_pred"] - a3_ho["f_pred"]
    g_s3 = bool(g_s3_point >= DELTA and nd_lb > 0.0)

    c1_pass = bool(g_s1 and g_s2 and g_s3)
    if c1_pass:
        verdict = "C1_PASS"
    elif not g_s1:
        verdict = "NEGATIVE_NO_ENGAGEMENT"
    elif not g_s2:
        verdict = "KILL_PILOT_MEMORIZATION_ONLY"
    else:
        verdict = "C1_FAIL_ATTRIBUTION_NO_C2"

    return {
        "baseline_burst_A2_heldout": baseline_burst,
        "G-S1": {
            "point": g_s1_point, "wilson_lb": s1_lb, "passed": g_s1,
            "definition": "TRAIN f_pred(final) - 0 >= 0.05 AND Wilson LB > 0",
        },
        "G-S2": {
            "point": g_s2_point, "wilson_lb": s2_lb, "passed": g_s2,
            "definition": "HELD-OUT f_pred(final) - baseline_burst >= 0.05 AND Wilson LB > 0",
        },
        "G-S3": {
            "point": g_s3_point, "newcombe_lb": nd_lb, "newcombe_ub": nd_ub,
            "passed": g_s3,
            "definition": "A1 HELD-OUT - A3 HELD-OUT >= 0.05 AND Newcombe excludes 0",
        },
        "c1_pass": c1_pass,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Wall-clock governor (prereg §8 hard cap + continuous projection)
# ---------------------------------------------------------------------------


class WallClockGovernor:

    def __init__(self, cap_hours: float) -> None:
        self.cap_seconds = float(cap_hours) * 3600.0
        self.t0 = time.time()
        self.deadline = self.t0 + self.cap_seconds

    def elapsed_s(self) -> float:
        return time.time() - self.t0

    def remaining_s(self) -> float:
        return self.deadline - time.time()

    def projection(self, steps_done: int, steps_total: int) -> dict[str, Any]:
        rate = steps_done / self.elapsed_s() if self.elapsed_s() > 0 else 0.0
        remaining_steps = max(steps_total - steps_done, 0)
        eta_s = remaining_steps / rate if rate > 0 else None
        return {
            "steps_done": steps_done,
            "steps_total": steps_total,
            "steps_per_s": round(rate, 4),
            "eta_hours": round(eta_s / 3600.0, 3) if eta_s is not None else None,
            "projected_total_hours":
                round((self.elapsed_s() + eta_s) / 3600.0, 3) if eta_s is not None else None,
            "cap_hours": round(self.cap_seconds / 3600.0, 3),
        }

    def check(self, where: str) -> None:
        if time.time() > self.deadline:
            raise TimeoutError(f"wall-clock cap exceeded at {where}")


def spike_preflight_gate(engine: ArmEngine) -> None:
    """Prereg §6 HARD STOP before any long leg: population rate floor.

    As measured on 2026-08-26 the FROZEN default lane constants are silent at
    N = 131,072 (mean winner-set size 0); this gate converts that physics into
    a clean STOP instead of a silent null result.  See C1_READINESS.md.
    """
    health = engine.recorder.spike_rate_health(SPIKE_RATE_FLOOR_FRACTION)
    health["mean_winner_set_size"] = engine.recorder.mean_winner_set_size()
    health["empty_winner_fraction"] = engine.recorder.empty_winner_fraction()
    health["lane_cells"] = engine.lane_cells
    health["steps_evaluated"] = engine.steps_total
    if not health["passed"] or health["mean_winner_set_size"] <= 0.0:
        raise CohortStop(
            "BLOCKED: prereg §6 spike-rate pre-flight FAILED at N="
            f"{engine.lane_cells} (mean_activity_density="
            f"{health.get('mean_activity_density')}, mean_winner_set_size="
            f"{health['mean_winner_set_size']}, empty_winner_fraction="
            f"{health['empty_winner_fraction']}). The frozen default lane "
            "constants are silent at this population size; a drive-calibration "
            "decision is required BEFORE any C1 leg runs (C1_READINESS.md).",
            kind="BLOCKED",
        )


# ---------------------------------------------------------------------------
# Cohort orchestration (epoch-0 -> E exposure passes -> final sweep -> gates)
# ---------------------------------------------------------------------------


def _canon_final_stream(
    books: Sequence[tuple[str, str, str]], manifest: dict[str, Any], cache_dir: Path
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """TRAIN u HELD-OUT token stream in canon book order + book records.

    The final evaluation sweep replays every book of the PRIMARY gating split
    in canon order; side masks for gate scoring come from the records' roles.
    """
    arrays = {
        "train": np.load(cache_dir / "train_tokens.npy"),
        "val": np.load(cache_dir / "val_tokens.npy"),
    }
    tokens: list[int] = []
    records: list[dict[str, Any]] = []
    for short, role, _text in books:
        side = "train" if role == "train" else "val"
        rec = next(
            r for r in manifest["sides"][side]["book_records"] if r["book"] == short
        )
        lo, hi = rec["start"], rec["end"]
        records.append({
            "book": short, "role": role,
            "start": len(tokens), "end": len(tokens) + (hi - lo),
        })
        tokens.extend(arrays[side][lo:hi].tolist())
    return np.asarray(tokens, dtype=np.int32), records


def _side_masks(records: Sequence[dict[str, Any]], total_steps: int) -> dict[str, np.ndarray]:
    masks = {
        "train": np.zeros(total_steps, dtype=bool),
        "heldout": np.zeros(total_steps, dtype=bool),
    }
    for rec in records:
        side = "train" if rec["role"] == "train" else "heldout"
        masks[side][rec["start"]:rec["end"]] = True
    return masks


def arm_summary_block(engine: ArmEngine) -> dict[str, Any]:
    tracker = engine.tracker
    return {
        "steps_total": engine.steps_total,
        "f_pred_overall": (engine.predicted_total / engine.steps_total) if engine.steps_total else 0.0,
        "burst_fraction_overall": (engine.burst_total / engine.steps_total) if engine.steps_total else 0.0,
        "elections": engine.elections,
        "allocated_segments": engine.allocations_concept_band + engine.allocations_base_band,
        "allocations_concept_band": engine.allocations_concept_band,
        "allocations_base_band": engine.allocations_base_band,
        "punished_events": engine.punished_events,
        "removed_contacts": engine.removed_contacts,
        "total_segments": tracker.total_segments(),
        "total_contacts": tracker.total_contacts(),
        "permanences_sha256": permanences_sha256(tracker),
        "spike_health": {
            **engine.recorder.spike_rate_health(SPIKE_RATE_FLOOR_FRACTION),
            "mean_winner_set_size": engine.recorder.mean_winner_set_size(),
            "empty_winner_fraction": engine.recorder.empty_winner_fraction(),
        },
    }


def _resume_context(args, manifest: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match between a checkpoint and this launch."""
    return {
        "trimmed_sha256": manifest["trimmed_sha256"],
        "lane_cells": int(args.lane_cells),
        "exposure_passes": int(args.exposure_passes),
        "arms": ",".join(args.arms),
        "tokenizer": TOKENIZER_TAG,
        "construction_seed": CONSTRUCTION_SEED,
    }


def _validate_resume_meta(meta: dict[str, Any], context: dict[str, Any]) -> None:
    saved = meta.get("context", {})
    for key, expected in context.items():
        got = saved.get(key)
        if key == "arms":
            got = ",".join(a for a in (got or "").split(",") if a)
        if got != expected:
            raise CohortStop(
                f"resume invalid: checkpoint {key}={got!r} != launch {expected!r}",
                kind="RESUME_INVALID",
            )


def _build_engine_from_checkpoint(
    arm: str, ctx: dict[str, Any], checkpointer: ArmCheckpointer, meta: dict[str, Any]
) -> ArmEngine:
    """Rebuild one arm's engine from its checkpoint (tracker + lane + counters)."""
    arm_dir = checkpointer.dir / arm
    lane_path = arm_dir / "lane.npz"
    if not (arm_dir / "tracker.npz").exists() or not lane_path.exists():
        raise CohortStop(
            f"checkpoint for {arm} missing tracker.npz/lane.npz => resume invalid",
            kind="RESUME_INVALID",
        )

    engine = ArmEngine(arm, ctx["distinct_ids"], ctx["lane_cells"])
    arrays = dict(np.load(arm_dir / "tracker.npz"))
    engine.tracker = ArmCheckpointer.restore_tracker(
        arm, arrays, {int(t): pool_cells_for_id(int(t), cell_offset=ctx["lane_cells"])
                      for t in ctx["distinct_ids"]},
        lane_cells=ctx["lane_cells"],
    )

    from humanbrain.recurrent_state import RecurrentCellSynapseState

    _envelope, state = RecurrentCellSynapseState.load_enveloped_checkpoint(lane_path)
    engine.circuit = RecurrentCellCircuit.restore_cell_synapse_state(state)
    # R0-c Thread-1 seam: re-attach the sink through the property setter -
    # restore_cell_synapse_state builds a fresh circuit without one.
    engine.circuit.observation_sink = engine.recorder

    counters_arrays = dict(np.load(arm_dir / "counters.npz"))
    engine.load_counters(
        meta.get("counter_labels", []), counters_arrays, meta.get("active_counter_label")
    )
    engine.w_prev = tuple(int(x) for x in meta.get("w_prev", []))
    engine.steps_total = int(meta.get("steps_total", 0))
    return engine


def run_single_arm(
    arm: str,
    *,
    ctx: dict[str, Any],
    args,
    checkpointer: ArmCheckpointer,
    governor: WallClockGovernor,
) -> dict[str, Any]:
    """Epoch-0 -> E exposure passes -> final eval sweep for ONE arm.

    Resume-safe at book boundaries: cursor phases epoch0/exposure/final/done;
    checkpoints carry tracker + lane envelope + packed counters + w_prev.
    """
    train_tokens = ctx["train_tokens"]
    train_records = ctx["train_records"]
    context = ctx["resume_context"]

    cursor = {"phase": PHASE_EPOCH0, "pass_index": 0, "book_index": 0}
    summaries: dict[str, Any] = {}
    resumed = False
    if checkpointer.has_checkpoint(arm) and not getattr(args, "fresh", False):
        meta = checkpointer.load_meta(arm)
        _validate_resume_meta(meta, context)
        cursor = meta["cursor"]
        summaries = meta.get("summaries", {})
        if cursor["phase"] != PHASE_DONE:
            print(f"      [{arm}] resuming at {cursor}", flush=True)
            engine = _build_engine_from_checkpoint(arm, ctx, checkpointer, meta)
            resumed = True
        else:
            print(f"      [{arm}] checkpoint says DONE; reloading summary only", flush=True)
            return summaries

    if not resumed:
        t0 = time.time()
        engine = ArmEngine(arm, ctx["distinct_ids"], ctx["lane_cells"])
        print(
            f"      [{arm}] wired: {len(ctx['distinct_ids'])} pools x "
            f"{K_POOL_CONST()} cells, growth_on={engine.tracker.growth_on}, "
            f"N={ctx['lane_cells']} ({time.time() - t0:.1f}s)",
            flush=True,
        )

    def snapshot(cur: dict[str, Any]) -> None:
        checkpointer.save(
            arm, engine, cur, summaries,
            {"book_row_digests": [
                {"sweep": r["sweep"], "book": r["book"], "row_digest": r.get("row_digest", "")}
                for r in engine.book_rows
            ]},
            engine.counter_arrays(),
            context=context,
            counter_labels=list(engine.counters.keys()),
        )

    try:
        # -- phase: epoch-0 structural sweep ---------------------------------
        if cursor["phase"] == PHASE_EPOCH0:
            engine.open_counter("epoch0")
            print(f"      [{arm}] epoch-0 structural sweep...", flush=True)
            engine.run_stream(
                train_tokens, train_records, sweep_label=PHASE_EPOCH0,
                deadline=governor.deadline, start_book=int(cursor["book_index"]),
                on_book_done=lambda bi: snapshot(
                    {"phase": PHASE_EPOCH0, "pass_index": 0, "book_index": bi + 1}),
            )
            arr = engine.counters["epoch0"].finalize_array()
            predicted_steps = int(arr.sum())
            if predicted_steps != 0:
                raise CohortStop(
                    f"epoch-0 structural violation in {arm}: f_pred = "
                    f"{predicted_steps}/{arr.size} != 0 => STOP (prereg §5)"
                )
            summaries["epoch0"] = {"steps": int(arr.size), "predicted": 0, "f_pred": 0.0}
            cursor = {"phase": PHASE_EXPOSURE, "pass_index": 0, "book_index": 0}
            snapshot(cursor)

        # -- phase: E ordered exposure passes over TRAIN ---------------------
        start_pass = int(cursor["pass_index"]) if cursor["phase"] == PHASE_EXPOSURE else 0
        for p in range(start_pass, ctx["exposure_passes"]):
            label = f"exposure::{p}"
            if label not in engine.counters:
                engine.open_counter(label)
            start_book = int(cursor["book_index"]) if (
                cursor["phase"] == PHASE_EXPOSURE and cursor["pass_index"] == p
            ) else 0
            # Prereg §6 hard stop early in pass 1 only (rolling telemetry must
            # have accumulated PREFLIGHT_PREFIX_TOKENS steps to be meaningful).
            attach_gate = (p == 0 and engine.steps_total < PREFLIGHT_PREFIX_TOKENS)
            print(f"      [{arm}] exposure pass {p + 1}/{ctx['exposure_passes']}...", flush=True)
            engine.run_stream(
                train_tokens, train_records, sweep_label=label,
                deadline=governor.deadline, start_book=start_book,
                health_gate=spike_preflight_gate if attach_gate else None,
                health_gate_at_step=PREFLIGHT_PREFIX_TOKENS if attach_gate else None,
                on_book_done=lambda bi, _p=p: snapshot(
                    {"phase": PHASE_EXPOSURE, "pass_index": _p, "book_index": bi + 1}),
            )
            arr = engine.counters[label].finalize_array()
            summaries[label] = {
                "steps": int(arr.size),
                "predicted": int(arr.sum()),
                "f_pred": float(arr.mean()) if arr.size else 0.0,
                "burst_fraction_by_book": [
                    {"book": r["book"], "burst_fraction": r["burst_fraction"]}
                    for r in engine.book_rows if r["sweep"] == label
                ],
            }
            cursor = {"phase": PHASE_EXPOSURE, "pass_index": p + 1, "book_index": 0}
            snapshot(cursor)

        # -- phase: ONE final evaluation sweep over TRAIN u HELD-OUT ---------
        if cursor["phase"] in (PHASE_EXPOSURE, PHASE_FINAL):
            engine.open_counter(PHASE_FINAL)
            start_book = int(cursor["book_index"]) if cursor["phase"] == PHASE_FINAL else 0
            print(f"      [{arm}] final evaluation sweep (TRAIN u HELD-OUT)...", flush=True)
            engine.run_stream(
                ctx["final_tokens"], ctx["final_records"], sweep_label=PHASE_FINAL,
                deadline=governor.deadline, start_book=start_book,
                on_book_done=lambda bi: snapshot(
                    {"phase": PHASE_FINAL, "pass_index": cursor["pass_index"],
                     "book_index": bi + 1}),
            )
            masks = _side_masks(ctx["final_records"], len(engine.counters[PHASE_FINAL]))
            final_summary = engine.sweep_summary(PHASE_FINAL, masks)
            summaries[PHASE_FINAL] = final_summary
            summaries["arm"] = arm_summary_block(engine)
            if arm == "A2":
                # Structural-zero tripwire over the WHOLE arm history (§4):
                # growth-off must never predict and never grow state.
                if (final_summary["predicted"] > 0
                        or summaries["epoch0"]["predicted"] > 0
                        or any(summaries[k].get("predicted", 0) > 0
                               for k in summaries if k.startswith("exposure::"))
                        or engine.tracker.total_segments()
                        or engine.tracker.total_contacts()):
                    raise CohortStop(
                        "STOP: A2 growth-off produced nonzero predicted-path "
                        "state => leakage/bug (prereg §4)"
                    )
            cursor = {"phase": PHASE_DONE, "pass_index": ctx["exposure_passes"],
                      "book_index": 0}
            snapshot(cursor)
    finally:
        # Free the big per-arm state promptly; the next arm rebuilds fresh.
        del engine.circuit, engine.tracker
    return summaries


def orchestrate(args) -> tuple[int, dict[str, Any]]:
    """Full C1 cohort: corpus pins -> per-arm pipeline -> gates -> manifest."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    governor = WallClockGovernor(args.wall_clock_cap_hours)

    print(f"[1/3] corpus pins + frozen tokenizer stack -> {out_dir / 'cache'}", flush=True)
    manifest = build_token_caches(out_dir)
    books, _prov = load_corpus()
    train_tokens = np.load(out_dir / "cache" / "train_tokens.npy")
    final_tokens, final_records = _canon_final_stream(books, manifest, out_dir / "cache")
    distinct_ids = sorted(
        set(np.unique(train_tokens).tolist())
        | set(np.unique(final_tokens).tolist())
    )
    feasibility = manifest["feasibility"]
    print(
        f"      TRAIN {manifest['sides']['train']['tokens']:,} tokens | "
        f"HELD-OUT {manifest['sides']['val']['tokens']:,} tokens | "
        f"{len(distinct_ids)} distinct ids | configured cells "
        f"{feasibility['configured_cells']:,}",
        flush=True,
    )

    steps_budget = (
        len(train_tokens) * (1 + args.exposure_passes) + len(final_tokens)
    )
    print(
        f"[2/3] cohort schedule: epoch-0 + {args.exposure_passes} exposure passes "
        f"+ final sweep = {steps_budget:,} steps/arm x {len(args.arms)} arms; "
        f"wall-clock cap {args.wall_clock_cap_hours:g} h", flush=True,
    )

    ctx = {
        "train_tokens": train_tokens,
        "train_records": [r for r in manifest["sides"]["train"]["book_records"]],
        "final_tokens": final_tokens,
        "final_records": final_records,
        "distinct_ids": distinct_ids,
        "lane_cells": int(args.lane_cells),
        "exposure_passes": int(args.exposure_passes),
        "resume_context": _resume_context(args, manifest),
    }
    checkpointer = ArmCheckpointer(out_dir)

    summaries: dict[str, Any] = {}
    status = "completed"
    error: str | None = None
    stop_kind: str | None = None
    try:
        for arm in args.arms:
            print(f"[arm {arm}] pipeline starting...", flush=True)
            summaries[arm] = run_single_arm(
                arm, ctx=ctx, args=args, checkpointer=checkpointer, governor=governor
            )
            print(f"[arm {arm}] done.", flush=True)
            print("      projection: " + json.dumps(governor.projection(
                sum(s.get("steps_total", 0) for s in summaries.values() if isinstance(s, dict)),
                steps_budget * len(args.arms))), flush=True)
    except CohortStop as exc:
        status, error, stop_kind = exc.kind.lower(), str(exc), exc.kind
    except TimeoutError as exc:
        status, error = "aborted_wallclock", str(exc)

    gates: dict[str, Any] | None = None
    verdict: str | None = None
    if all(a in summaries and PHASE_FINAL in summaries[a] for a in ("A1", "A2", "A3")):
        try:
            gates = evaluate_gates({a: summaries[a][PHASE_FINAL] for a in ("A1", "A2", "A3")})
            verdict = gates["verdict"]
        except CohortStop as exc:
            status, error, stop_kind = exc.kind.lower(), str(exc), exc.kind

    run_manifest = {
        "experiment": "r3-c1-cohort-segment-growth",
        "prereg": "docs/research/r3-segment-growth-preregistration.md §4-§8",
        "runner": "experiments/r3_pilot/run_c1_cohort.py",
        "status": status,
        "error": error,
        "verdict": verdict,
        "gates": gates,
        "corpus": {
            "candidate_id": manifest["candidate_id"],
            "trimmed_sha256": manifest["trimmed_sha256"],
            "train_books": manifest["train_books"],
            "held_out_books": manifest["held_out_books"],
            "held_out_names": manifest["held_out_names"],
            "sides": {
                side: {k: v for k, v in rec.items() if k != "book_records"}
                for side, rec in manifest["sides"].items()
            },
        },
        "feasibility": feasibility,
        "lane_cells": int(args.lane_cells),
        "configured_cells": feasibility["configured_cells"],
        "exposure_passes": int(args.exposure_passes),
        "construction_seed": CONSTRUCTION_SEED,
        "winner_set_source": "observation-sink spike stream (re-affirmed; see module docstring)",
        "a3_operationalization": "cumulative-union shuffled targets (disclosed deviation; see module docstring)",
        "spike_rate_preflight": {
            "floor_fraction": SPIKE_RATE_FLOOR_FRACTION,
            "enforced_at_step": PREFLIGHT_PREFIX_TOKENS,
            "kind": stop_kind,
        },
        "steps_budget_per_arm": steps_budget,
        "wall_clock": governor.projection(
            sum(s.get("steps_total", 0) for s in summaries.values() if isinstance(s, dict)),
            steps_budget * len(args.arms),
        ),
        "arms_summaries": summaries,
    }
    out_path = out_dir / "c1_run.json"
    out_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[3/3] status={status}" + (f" verdict={verdict}" if verdict else ""), flush=True)
    if gates:
        for name in ("G-S1", "G-S2", "G-S3"):
            g = gates[name]
            key = "wilson_lb" if "wilson_lb" in g else "newcombe_lb"
            print(f"      {name}: point={g['point']:.4f} {key}={g[key]:.4f} "
                  f"passed={g['passed']}", flush=True)
    if error:
        print(f"      {status.upper()}: {error}", flush=True)
    print(f"      manifest: {out_path}", flush=True)

    exit_code = EXIT_OK
    if status == "aborted_wallclock":
        exit_code = EXIT_WALLCLOCK_ABORT
    elif status != "completed":
        exit_code = EXIT_RESUME_INVALID if stop_kind == "RESUME_INVALID" else EXIT_STOP
    return exit_code, run_manifest


# ---------------------------------------------------------------------------
# Dry run: corpus pins + three-arm wiring end-to-end at fixture scale
# ---------------------------------------------------------------------------


def run_dry_run(args) -> tuple[int, dict[str, Any]]:
    """--dry-run: exercise every wiring seam WITHOUT spending cohort compute.

    Verifies, at --lane-cells (default the N=96 G-S0 fixture leg):
      1. corpus pins: sha assert, 66/49/17 split, held-out roster;
      2. frozen tokenizer stack + BOTH side streams (cached after first build);
      3. build-time feasibility assertion at the concept-band ceiling;
      4. all three arms wired (pool mapping + tracker + circuit + sink) and
         replaying a TRAIN prefix with glue-faithful step semantics - the A1/A2
         telemetry hashes must equal the reference glue driver bit-for-bit;
      5. A2 structural-zero tripwire on the prefix;
      6. spike-rate pre-flight health REPORT (not gating here - the C1-scale
         silence is a drive-calibration decision, not a dry-run failure);
      7. checkpoint save -> restore round-trip equality.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "experiment": "r3-c1-dry-run",
        "prereg": "docs/research/r3-segment-growth-preregistration.md §4-§8",
        "runner": "experiments/r3_pilot/run_c1_cohort.py",
        "lane_cells": int(args.lane_cells),
        "prefix_tokens": int(args.prefix_tokens),
        "construction_seed": CONSTRUCTION_SEED,
        "winner_set_source": "observation-sink spike stream (re-affirmed; see module docstring)",
    }

    print("[1/5] corpus pins (assert-or-STOP)...", flush=True)
    books, prov = load_corpus()
    report["corpus"] = prov
    print(
        f"      sha256 OK ({prov['trimmed_sha256'][:16]}...) | 66 books | "
        f"{prov['train_books']} TRAIN | {prov['held_out_books']} HELD-OUT",
        flush=True,
    )

    print("[2/5] frozen concept-token caches (both sides)...", flush=True)
    manifest = build_token_caches(out_dir)
    report["streams"] = {
        side: {k: v for k, v in rec.items() if k != "book_records"}
        for side, rec in manifest["sides"].items()
    }
    report["feasibility"] = manifest["feasibility"]
    train_arr = np.load(out_dir / "cache" / "train_tokens.npy")
    val_arr = np.load(out_dir / "cache" / "val_tokens.npy")
    distinct_ids = sorted(set(np.unique(train_arr).tolist()) | set(np.unique(val_arr).tolist()))
    report["distinct_ids_union"] = len(distinct_ids)
    for side in ("train", "val"):
        s = report["streams"][side]
        print(
            f"      {side}: {s['tokens']:,} tokens, {s['books']} books, "
            f"{s['verses']} verses, max id {s['max_token_id']}, "
            f"concept-band ids {s['concept_band_ids']}",
            flush=True,
        )
    print(
        f"      feasibility: lane {report['feasibility']['lane_cells']:,} + pools "
        f"{report['feasibility']['pool_cells']:,} >= "
        f"(max_id+1)*k = {report['feasibility']['literal_(max_id+1)*k']:,} OK",
        flush=True,
    )

    prefix = train_arr[: int(args.prefix_tokens)]
    engines: dict[str, ArmEngine] = {}
    failures: list[str] = []
    print(f"[3/5] wiring A1/A2/A3 at N={args.lane_cells} + prefix replay...", flush=True)
    t0 = time.time()
    for arm in ARMS:
        engine = ArmEngine(arm, distinct_ids, int(args.lane_cells))
        engine.open_counter("dryrun")
        engine.run_stream(prefix, [{"book": "DRYRUN-PREFIX", "role": "train",
                                    "start": 0, "end": len(prefix)}],
                          sweep_label="dryrun", deadline=None)
        engines[arm] = engine
        summary = arm_summary_block(engine)
        report.setdefault("arms", {})[arm] = summary
        print(
            f"      [{arm}] {summary['steps_total']} steps, elections "
            f"{summary['elections']}, segments {summary['total_segments']}, "
            f"contacts {summary['total_contacts']}, mean|W| "
            f"{round(summary['spike_health']['mean_winner_set_size'], 2)}",
            flush=True,
        )
    report["wiring_wall_s"] = round(time.time() - t0, 1)

    a2 = engines["A2"]
    a2_zero = (
        a2.predicted_total == 0
        and a2.tracker.total_segments() == 0
        and a2.tracker.total_contacts() == 0
        and all(row[2] for row in list(a2.window))  # row[2] = burst flag
    )
    report["a2_structural_zero_prefix"] = bool(a2_zero)
    if not a2_zero:
        failures.append("A2 structural-zero tripwire failed on dry-run prefix")
    print(f"      A2 structural zero on prefix: {bool(a2_zero)}", flush=True)

    print("[4/5] glue anchor (bit-for-bit vs reference driver) + pre-flight report...",
          flush=True)
    anchors: dict[str, bool] = {}
    for arm in ("A1", "A2"):
        glue_run, glue_recorder, _diag = run_arm(
            [int(t) for t in prefix], arm=arm, lane_cells=int(args.lane_cells), passes=1
        )
        engine_hash = telemetry_sha256(list(engines[arm].window))
        anchors[arm] = engine_hash == glue_run.telemetry_sha256
        if not anchors[arm]:
            failures.append(f"glue anchor mismatch for {arm}")
    report["glue_anchor_bit_identical"] = anchors
    print(f"      A1 {anchors['A1']} | A2 {anchors['A2']} (telemetry sha equality)", flush=True)

    preflight: dict[str, Any] = {}
    for arm, engine in engines.items():
        health = engine.recorder.spike_rate_health(SPIKE_RATE_FLOOR_FRACTION)
        health["mean_winner_set_size"] = engine.recorder.mean_winner_set_size()
        health["empty_winner_fraction"] = engine.recorder.empty_winner_fraction()
        health["steps_evaluated"] = engine.steps_total
        preflight[arm] = health
    report["spike_rate_preflight_report"] = preflight
    print(
        "      pre-flight (report-only): "
        + ", ".join(f"{a}: density={h['mean_activity_density']:.5f} passed={h['passed']}"
                    for a, h in preflight.items()),
        flush=True,
    )

    print("[5/5] checkpoint round-trip (save -> restore -> array equality)...", flush=True)
    checkpointer = ArmCheckpointer(out_dir / "dryrun_ckpt")
    roundtrip: dict[str, bool] = {}
    for arm, engine in engines.items():
        # Compare CANONICAL flattened state: all_permanences() walks
        # _segments.values() in insertion order, which a restore legitimately
        # reorders to sorted-cell order - the flatten arrays are the contract.
        before = ArmCheckpointer.flatten_tracker(engine.tracker)
        union_before = (
            engine.tracker.union_size() if isinstance(engine.tracker, ShuffledWinnerTracker) else None
        )
        checkpointer.save(
            arm, engine, {"phase": PHASE_EPOCH0, "pass_index": 0, "book_index": 0},
            {}, {}, engine.counter_arrays(), context={"probe": True}, counter_labels=["dryrun"],
        )
        meta = checkpointer.load_meta(arm)
        arrays = dict(np.load(checkpointer.dir / arm / "tracker.npz"))
        restored = ArmCheckpointer.restore_tracker(
            arm, arrays,
            {int(t): pool_cells_for_id(int(t), cell_offset=int(args.lane_cells))
             for t in distinct_ids},
            lane_cells=int(args.lane_cells),
        )
        after = ArmCheckpointer.flatten_tracker(restored)
        ok = set(before) == set(after) and all(
            np.array_equal(before[k], after[k]) for k in before
        )
        if union_before is not None:
            ok = ok and restored.union_size() == union_before
        ok = ok and restored.num_steps == engine.tracker.num_steps
        roundtrip[arm] = bool(ok)
        del restored
    report["checkpoint_round_trip"] = roundtrip
    print(f"      A1 {roundtrip['A1']} | A2 {roundtrip['A2']} | A3 {roundtrip['A3']}", flush=True)
    if not all(roundtrip.values()):
        failures.append("checkpoint round-trip inequality")

    report["failures"] = failures
    report["ok"] = not failures
    out_path = out_dir / "c1_dry_run.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"dry-run: {'OK' if report['ok'] else 'FAIL'} -> {out_path}", flush=True)
    return (EXIT_OK if report["ok"] else EXIT_STOP), report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="C1 cohort runner - r3 segment-growth pilot (see module docstring).",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise corpus pins + arm wiring end-to-end; no long legs")
    ap.add_argument("--output-dir", default=None,
                    help="cohort output directory (default experiments/r3_pilot/c1_run,"
                         " c1_dry_run under --dry-run)")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of A1,A2,A3 (default all)")
    ap.add_argument("--lane-cells", type=int, default=None,
                    help=f"population size (default {C1_LANE_CELLS}; dry-run default "
                         f"{LANE_CELLS_FIXTURE})")
    ap.add_argument("--exposure-passes", type=int, default=EXPOSURE_PASSES,
                    help=f"E ordered exposure passes over TRAIN (default {EXPOSURE_PASSES})")
    ap.add_argument("--wall-clock-cap-hours", type=float, default=WALL_CLOCK_CAP_HOURS,
                    help=f"hard cap (default {WALL_CLOCK_CAP_HOURS:g} h)")
    ap.add_argument("--prefix-tokens", type=int, default=DRY_RUN_PREFIX_TOKENS,
                    help=f"--dry-run replay length per arm (default {DRY_RUN_PREFIX_TOKENS})")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and start the cohort from scratch")
    args = ap.parse_args(argv)

    args.arms = tuple(a.strip().upper() for a in args.arms.split(",") if a.strip())
    bad = [a for a in args.arms if a not in ARMS]
    if bad:
        print(f"unknown arms {bad}; legal: {ARMS}", file=sys.stderr)
        return EXIT_STOP
    if args.lane_cells is None:
        args.lane_cells = LANE_CELLS_FIXTURE if args.dry_run else C1_LANE_CELLS
    if args.output_dir is None:
        args.output_dir = str(
            REPO_ROOT / "experiments" / "r3_pilot" / ("c1_dry_run" if args.dry_run else "c1_run")
        )

    if not args.dry_run:
        print(f"HARD NO-GO: {C1_LAUNCH_BLOCK_REASON}", file=sys.stderr)
        return EXIT_STOP

    code, _manifest = run_dry_run(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
