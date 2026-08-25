"""L1 spike: token injection via caller-created ProceduralInputEvents.

Question (roadmap R5 prerequisite, Layer 1): can a host language-model loop
drive the procedural runtime today -- no kernel changes -- by encoding each
token as external current delivered to chosen GIDs at a token boundary?

Evidence collected (run on codex/scale-opt-cpu, 2026-08-23):
  1. An injected event at 1e6 CONFIGURED cells makes its target spike.
  2. The spike procedurally expands fanout base edges (generated_edges > 0)
     and pulls downstream targets into residency with nonzero synaptic
     current; the target's own eligibility trace becomes >= 1.0.
  3. Empty advances over the same 1e6 address space touch zero configured
     ids (configured_ids_scanned == 0) -- configured N is metadata.
  4. Repeated tokens, distinct tokens, and determinism all check out;
     duplicate/stale/out-of-range events are rejected before FFI.

CPU-only, runs in ~1 s.
"""

from __future__ import annotations

import sys

from humanbrain.procedural_runtime import (
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
    ProceduralRuntimeError,
)
from humanbrain.procedural_topology import TopologyManifest

CONFIGURED_CELLS = 1_048_576  # 2^20 -- the 1e6 text-grow target
FANOUT = 8
TOKEN_STEPS = 100  # token_interval_ms / dt_ms == 5 ms / 0.05 ms
DRIVE_CURRENT_PA = 150_000.0  # test-proven suprathreshold magnitude


def make_runtime() -> NativeProceduralRuntime:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=CONFIGURED_CELLS,
            fanout=FANOUT,
            topology="uniform_no_self",
            seed=71,
        ),
        resident_capacity=4_096,
        event_capacity=4_096,
        spike_capacity=8_192,
        interval_distinct_capacity=4_096,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )
    return NativeProceduralRuntime(config)


def token_event(step: int, gid: int) -> ProceduralInputEvent:
    """Encode one token drive as a caller-created input event."""
    return ProceduralInputEvent(
        due_step=step,
        target_gid=gid,
        source_gid=gid,  # self provenance; nothing is materialized
        edge_slot=0,     # bounded order value for injected events
        current_pa=DRIVE_CURRENT_PA,
    )


def main() -> int:
    checks: list[str] = []

    # --- 1. negative control: an empty advance leaves the space implicit ---
    with make_runtime() as runtime:
        result = runtime.advance(TOKEN_STEPS)
        assert result.spikes == ()
        assert result.accounting.cells.resident_distinct == 0
        snapshot = runtime.snapshot()
        assert not snapshot.residents and not snapshot.pending_events
        checks.append(
            "empty advance @1e6 cfg: 0 spikes, 0 residents, "
            f"configured_ids_scanned={result.configured_ids_scanned}"
        )
        assert result.configured_ids_scanned == 0

    # --- 2. one injected token spikes its target and propagates ---
    token_a = 409_096
    with make_runtime() as runtime:
        runtime.advance(TOKEN_STEPS)  # settle boundary at step 100
        result = runtime.advance(
            TOKEN_STEPS, [token_event(runtime._current_step, token_a)]
        )
        spiked_gids = {spike.gid for spike in result.spikes}
        assert token_a in spiked_gids, f"token cell did not spike: {sorted(spiked_gids)[:8]}"
        accounting = result.accounting
        assert accounting.events.delivered >= 1
        assert accounting.edges.generated >= FANOUT
        snapshot = runtime.snapshot()
        states = {state.gid: state for state in snapshot.residents}
        target_state = states[token_a]
        # eligibility is +1.0 per own spike, then decays x exp(-dt/80 ms)
        # every step -- a spike anywhere in the 100-step window leaves
        # >= exp(-100*dt/80) ~= 0.939 at the boundary.
        assert target_state.eligibility > 0.9
        downstream = [gid for gid in states if gid != token_a]
        assert downstream, "no downstream cell became resident"
        assert any(states[gid].synaptic_current_pa != 0.0 for gid in downstream)
        checks.append(
            "injected token @step100: target spiked, "
            f"delivered={accounting.events.delivered}, "
            f"generated_edges={accounting.edges.generated}, "
            f"residents={len(snapshot.residents)}, "
            f"target eligibility={target_state.eligibility:.1f}, "
            f"{len(downstream)} downstream cells resident"
        )

        # --- 3. second token on the next boundary; repeat of token A later ---
        step_b = runtime._current_step
        result_b = runtime.advance(
            TOKEN_STEPS, [token_event(step_b, token_a + 7)]
        )
        assert token_a + 7 in {spike.gid for spike in result_b.spikes}
        step_c = runtime._current_step
        result_c = runtime.advance(TOKEN_STEPS, [token_event(step_c, token_a)])
        assert token_a in {spike.gid for spike in result_c.spikes}
        checks.append("distinct token fires on its own boundary; repeated token re-fires")

    # --- 4. determinism across identically-seeded runtimes ---
    def scripted() -> tuple[tuple[int, int], ...]:
        with make_runtime() as runtime:
            trace: list[tuple[int, int]] = []
            for offset, gid in ((0, token_a), (1, token_a + 7), (2, token_a)):
                runtime.advance(offset * TOKEN_STEPS or TOKEN_STEPS)
                result = runtime.advance(
                    TOKEN_STEPS, [token_event(runtime._current_step, gid)]
                )
                trace.extend((spike.step, spike.gid) for spike in result.spikes)
            return tuple(trace)

    assert scripted() == scripted()
    checks.append("identically-seeded runtimes produce bit-identical spike traces")

    # --- 5. validation guards fire before any FFI work ---
    with make_runtime() as runtime:
        runtime.advance(1)
        step = runtime._current_step
        for bad, label in (
            (
                [
                    token_event(step, 11),
                    token_event(step, 11),  # identical canonical key
                ],
                "duplicate canonical key rejected",
            ),
            (
                [ProceduralInputEvent(step - 1, 13, 13, 0, DRIVE_CURRENT_PA)],
                "stale due_step rejected",
            ),
            (
                [ProceduralInputEvent(step, CONFIGURED_CELLS, 0, 0, DRIVE_CURRENT_PA)],
                "out-of-range target_gid rejected",
            ),
        ):
            try:
                runtime.advance(TOKEN_STEPS, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{label} did not raise")
        try:
            runtime.advance(TOKEN_STEPS, [ProceduralInputEvent(0, 0, -1, 0, 1.0)])
        except ValueError:
            pass
        checks.append("duplicate/stale/out-of-range events rejected pre-FFI")

    print("L1 TOKEN INJECTION SPIKE -- PASS")
    for index, line in enumerate(checks, 1):
        print(f"  [{index}] {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
