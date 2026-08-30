"""One-time engineering reproduction of the banked prompt V4 checkpoint.

This module is inert when imported.  Only an explicit script invocation imports
the spent executable learner, authenticates its already-banked result, and
materializes the static inference artifact.  Normal runtime and tests must not
execute :func:`reproduce_artifact`.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from experiments.r3_pilot import (
    public_train_prompt_selection_v4_static_checkpoint_capsule_v1 as capsule,
)


def _reduced_span_records(row: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "start_token": int(record["start_token"]),
            "stop_token": int(record["stop_token"]),
            "first_spike_step": int(record["first_spike_step"]),
            "assembly_cells": tuple(int(cell) for cell in record["assembly_cells"]),
            "spike_cells": tuple(int(cell) for cell in record["spike_cells"]),
        }
        for record in row.span_events
    )


def _reduced_literal_records(row: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_token_index": int(record["canonical_token_index"]),
            "typed_output_active_cells": tuple(
                int(cell) for cell in record["typed_output_active_cells"]
            ),
        }
        for record in row.literal_events
    )


def reproduce_artifact(path: Path = capsule.ARTIFACT_PATH) -> tuple[str, str, str]:
    """Reproduce exact bytes once; this is not a new learner or scientific run."""

    from experiments.r3_pilot import public_train_prompt_selection_v4 as banked

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("static prompt V4 capsule already exists")
    if (
        capsule.sha256(capsule.BANKED_SOURCE_PATH.read_bytes()).hexdigest()
        != capsule.BANKED_SOURCE_SHA256
        or capsule.sha256(capsule.BANKED_TEST_PATH.read_bytes()).hexdigest()
        != capsule.BANKED_TEST_SHA256
        or banked.STATUS != banked.PASS
        or banked.RESULT.result_sha256 != capsule.BANKED_RESULT_SHA256
        or banked.WORKSPACE.checkpoint_sha256 != capsule.BANKED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("banked prompt V4 reproduction authority changed")

    excitatory = banked.WORKSPACE.excitatory_ticks
    inhibitory = banked.WORKSPACE.inhibitory_ticks
    checkpoint_header = banked._canonical_json(banked.WORKSPACE._metadata)
    output_cells = tuple(row.record() for row in banked.OUTPUT_CELLS)
    channel_branch = banked.CHANNEL_BRANCH.record()
    output_branches = tuple(row.record() for row in banked.OUTPUT_BRANCHES)
    branch_features = tuple(
        {
            "branch_id": branch.branch_id,
            "relation_cells": banked.protocol_v4._assembly_cells("relation", branch.channel),
            "role_cells": banked.protocol_v4._assembly_cells(
                "role", f"{branch.channel}|{branch.branch_role}"
            ),
        }
        for branch in banked.OUTPUT_BRANCHES
    )

    physical_inputs = banked._load_fit_physical_inputs()
    replay = banked.FIT_RESULT.replays[0]
    physical = physical_inputs[0]
    meter = banked._WorkMeter()
    trace = banked._encode_prompt_trace(physical, meter)
    if trace.sha256 != replay.trace_sha256:
        raise RuntimeError("banked reproduction input/replay ordering changed")
    state = banked._initial_occurrence_state(trace)
    actual_occurrences: list[dict[str, object]] = []
    for occurrence in replay.occurrence_replays:
        if occurrence.state_sha256 != state.sha256:
            raise RuntimeError("banked reproduction occurrence state changed")
        physical_winners = [int(occurrence.channel_decision.selected_cell_id)]
        for index, decision in enumerate(occurrence.branch_decisions):
            physical_winners.append(
                decision.selected_cell_id
                if decision.selected_cell_id is not None
                else banked._STOP_STATE_BASE + index
            )
        actual_occurrences.append(
            {
                "sequence_index": occurrence.sequence_index,
                "state_sha256": occurrence.state_sha256,
                "channel_decision": occurrence.channel_decision.record(),
                "selected_channel": occurrence.selected_channel,
                "branch_decisions": tuple(row.record() for row in occurrence.branch_decisions),
                "emitted_by_branch": occurrence.emitted_by_branch,
                "occurrence_binding": (
                    occurrence.occurrence_binding.record()
                    if occurrence.occurrence_binding is not None
                    else None
                ),
            }
        )
        state = banked._advance_occurrence_state(state, tuple(physical_winners))
    terminal = banked._decide_branch(
        banked.WORKSPACE,
        banked.CHANNEL_BRANCH,
        banked._canonical_channel_feature(trace, state, meter),
        meter,
        occurrence_index=len(replay.occurrence_replays),
        masked_cell_ids=set(),
    )
    if replay.termination != capsule.STOP or terminal.decision != capsule.STOP:
        raise RuntimeError("banked reproduction lost physical STOP")
    actual_record = {
        "termination": capsule.STOP,
        "occurrences": tuple(actual_occurrences),
        "terminal_decision": terminal.record(),
    }
    actual_winner_sha256 = capsule._actual_winner_sha256(actual_record)
    signature_role = capsule._plain(physical.signature_role_record)
    reproduction_fixture = {
        "span_records": _reduced_span_records(physical),
        "literal_records": _reduced_literal_records(physical),
        "signature_role_record": signature_role,
        "banked_actual_winner_sha256": actual_winner_sha256,
    }

    metadata: dict[str, object] = {
        "schema": capsule.SCHEMA,
        "banked_commit": capsule.BANKED_COMMIT,
        "banked_source_sha256": capsule.BANKED_SOURCE_SHA256,
        "banked_test_sha256": capsule.BANKED_TEST_SHA256,
        "banked_result_sha256": capsule.BANKED_RESULT_SHA256,
        "banked_checkpoint_sha256": capsule.BANKED_CHECKPOINT_SHA256,
        "banked_dependency_sha256": banked.DEPENDENCY_SHA256,
        "output_registry_sha256": banked.OUTPUT_REGISTRY_SHA256,
        "branch_registry_sha256": banked.BRANCH_REGISTRY_SHA256,
        "checkpoint_header_sha256": capsule.sha256(checkpoint_header).hexdigest(),
        "array_shape": excitatory.shape,
        "array_dtype": excitatory.dtype.str,
        "excitatory_sha256": capsule.sha256(excitatory.tobytes(order="C")).hexdigest(),
        "inhibitory_sha256": capsule.sha256(inhibitory.tobytes(order="C")).hexdigest(),
        "tick_cap": banked.TICK_CAP,
        "stop_score": banked.STOP_SCORE,
        "race_steps": banked.RACE_STEPS,
        "latency_current_pa_hex": tuple(
            row.current_pa.hex() for row in banked.race_spec.PROTOCOL.latency_table
        ),
        "latency_table_sha256": banked.race_spec.PROTOCOL.latency_table_sha256,
        "prompt_lane_count": banked.PROMPT_LANE_COUNT,
        "prompt_cells_per_lane": banked.PROMPT_CELLS_PER_LANE,
        "prompt_feature_offset": banked.PROMPT_FEATURE_OFFSET,
        "input_cell_count": banked.INPUT_CELL_COUNT,
        "occurrence_cap": banked.OCCURRENCE_CAP,
        "prompt_multipliers": banked._PROMPT_MULTIPLIERS,
        "prompt_position_gains": banked._PROMPT_POSITION_GAINS,
        "prompt_biases": banked._PROMPT_BIASES,
        "stop_state_base": banked._STOP_STATE_BASE,
        "output_cells": output_cells,
        "channel_branch": channel_branch,
        "output_branches": output_branches,
        "branch_features": branch_features,
        "reproduction_fixture": reproduction_fixture,
    }
    metadata["artifact_content_root"] = capsule._namespaced_sha256(
        capsule._ARTIFACT_CONTENT_NAMESPACE, metadata
    )
    metadata_text = capsule._canonical_json(metadata).decode("ascii")
    stream = BytesIO()
    np.savez_compressed(
        stream,
        metadata=np.asarray(metadata_text),
        checkpoint_header=np.frombuffer(checkpoint_header, dtype=np.uint8).copy(),
        excitatory=np.ascontiguousarray(excitatory, dtype=np.uint8),
        inhibitory=np.ascontiguousarray(inhibitory, dtype=np.uint8),
    )
    payload = stream.getvalue()
    path.write_bytes(payload)
    return (
        capsule.sha256(payload).hexdigest(),
        str(metadata["artifact_content_root"]),
        actual_winner_sha256,
    )


if __name__ == "__main__":
    print(reproduce_artifact())
