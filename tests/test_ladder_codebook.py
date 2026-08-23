"""Red tests for the a013 three-arm codebook knob (representation choice).

Covers the preregistered red-test list of
``docs/research/a013-representation-choice.md``:

- codebook determinism under seed (and frozen similarity-projection pinning);
- assembly-size bound and collision ceiling;
- same-token-same-assembly across grow AND completion (one shared mapping);
- framing-order preservation under re-encoding;
- CLI gate behavior for ``--codebook`` / ``--assembly-k``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "ladder_experiments.py"
SPEC = importlib.util.spec_from_file_location("ladder_experiments", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
# Register before exec: Python 3.14 dataclasses resolve annotations through
# sys.modules, which importlib file-location modules do not populate.
sys.modules.setdefault("ladder_experiments", HARNESS)
SPEC.loader.exec_module(HARNESS)

N_E = 128
ASSEMBLY_K = 16
TOKEN_COUNT = HARNESS._TOKEN_CELLS


# ---------------------------------------------------------------------------
# 1. Determinism under seed + frozen projection hash pinning
# ---------------------------------------------------------------------------


def test_codebook_determinism_under_seed() -> None:
    random_a = HARNESS.build_codebook(
        "random", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    random_b = HARNESS.build_codebook(
        "random", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    assert random_a.sha256 == random_b.sha256
    assert np.array_equal(random_a.assemblies, random_b.assemblies)

    random_other_seed = HARNESS.build_codebook(
        "random", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=8
    )
    assert random_a.sha256 != random_other_seed.sha256
    assert not np.array_equal(random_a.assemblies, random_other_seed.assemblies)

    # The similarity arm is a FROZEN deterministic projection: identical for
    # every run seed (the seed must not perturb it).
    sim_a = HARNESS.build_codebook(
        "similarity", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    sim_b = HARNESS.build_codebook(
        "similarity", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=8
    )
    assert sim_a.sha256 == sim_b.sha256
    assert np.array_equal(sim_a.assemblies, sim_b.assemblies)

    # Explicit is the untouched status quo: no replacement table at all.
    explicit = HARNESS.build_codebook(
        "explicit", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    assert explicit.mode == "explicit"
    assert explicit.assemblies is None
    assert explicit.sha256 == HARNESS.build_codebook(
        "explicit", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    ).sha256


def test_similarity_projection_hash_is_pinned() -> None:
    """Frozen v1 pins: any drift in the documented hash function fails here."""
    # Golden cells computed from the exact v1 definition (character-trigram
    # unit-count feature hashing into R^N with per-(name, cell) tie-break).
    assert HARNESS._similarity_assembly_cells("variable-0", N_E, ASSEMBLY_K) == (
        0,
        2,
        9,
        35,
        36,
        43,
        46,
        61,
        85,
        87,
        89,
        100,
        107,
        108,
        122,
        125,
    )
    assert HARNESS._similarity_assembly_cells("add", N_E, ASSEMBLY_K) == (
        4,
        39,
        40,
        50,
        51,
        53,
        54,
        58,
        59,
        67,
        69,
        71,
        88,
        98,
        99,
        125,
    )

    def overlap(name_a: str, name_b: str) -> int:
        cells_a = set(
            HARNESS._similarity_assembly_cells(name_a, N_E, ASSEMBLY_K)
        )
        cells_b = set(
            HARNESS._similarity_assembly_cells(name_b, N_E, ASSEMBLY_K)
        )
        return len(cells_a & cells_b)

    # Overlap must be monotone in shared character structure.
    similar = overlap("shared-prefix-alpha", "shared-prefix-beta")
    dissimilar = overlap("shared-prefix-alpha", "zeta-omega-qux")
    assert similar >= 8
    assert dissimilar <= 4
    assert similar > dissimilar

    # Real Layer-1 vocabulary: the two delimiter tokens share the
    # "-delimiter" body, while an unrelated primitive pair stays near the
    # hypergeometric baseline (~2/16 expected overlap).
    delimiter_overlap = overlap("context-delimiter", "input-delimiter")
    unrelated_overlap = overlap("variable-0", "subtract")
    assert delimiter_overlap >= 5
    assert unrelated_overlap <= 6
    assert delimiter_overlap > unrelated_overlap


# ---------------------------------------------------------------------------
# 2. Assembly-size bound + collision ceiling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["random", "similarity"])
def test_assembly_size_bound_and_collision_ceiling(mode: str) -> None:
    codebook = HARNESS.build_codebook(
        mode, token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    assemblies = codebook.assemblies
    assert assemblies is not None
    assert assemblies.shape == (TOKEN_COUNT, ASSEMBLY_K)
    assert assemblies.dtype == np.int32
    # Every assembly is exactly k DISTINCT reservoir cells inside [0, N_E),
    # stored in canonical sorted order.
    assert np.all(np.diff(assemblies.astype(np.int64), axis=1) > 0)
    assert int(assemblies.min()) >= 0
    assert int(assemblies.max()) < N_E

    # Collision bound: no single assembly may be shared by too many tokens.
    _, counts = np.unique(assemblies, axis=0, return_counts=True)
    max_shared = int(counts.max())
    if mode == "random":
        assert max_shared == 1  # distinct random draws; a tie would be a bug
    else:
        # Deterministic hash duplicates are possible but tightly bounded;
        # pinned to the observed frozen-projection value of 2. Exactly the
        # byte-char/primitive name aliases ('+', '-', '0'..'9') share an
        # assembly — same symbol string, hence same projected assembly.
        assert max_shared <= 2


def test_build_codebook_rejects_impossible_geometry() -> None:
    with pytest.raises(ValueError):
        HARNESS.build_codebook(
            "random", token_count=TOKEN_COUNT, n_e=N_E, k=N_E + 1, seed=7
        )
    with pytest.raises(ValueError):
        HARNESS.build_codebook(
            "random", token_count=TOKEN_COUNT, n_e=N_E, k=0, seed=7
        )
    with pytest.raises(ValueError):
        HARNESS.build_codebook(
            "bogus", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
        )


# ---------------------------------------------------------------------------
# 3. Same token -> same assembly across grow AND completion
# ---------------------------------------------------------------------------


def _grow_once(mode: str):
    curriculum = HARNESS.build_curriculum(7, 21)
    codebook = HARNESS.build_codebook(
        mode, token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    circuit = HARNESS._grow_circuit(
        curriculum,
        n_cells=N_E,
        seed=7,
        readout_rule="ridge",
        schedule="spaced",
        supervision="program_only",
        codebook=codebook,
    )
    return curriculum, codebook, circuit


def test_same_token_same_assembly_across_grow_and_completion() -> None:
    curriculum, codebook, circuit = _grow_once("random")
    assert codebook.assemblies is not None
    # Grow installed exactly the codebook's assemblies.
    assert np.array_equal(circuit._afferent_targets, codebook.assemblies)

    # Completion (typed-hole evaluation on the SAME circuit object) must keep
    # consulting that one shared mapping — nothing may reinstall identity or
    # rebuilt wiring between grow and completion.
    train_result = HARNESS._evaluate_cohort(
        circuit, curriculum["train_tasks"], collect_margins=True
    )
    calibration_result = HARNESS._evaluate_cohort(
        circuit, curriculum["calibration_tasks"], collect_margins=False
    )
    assert isinstance(train_result["replayed"], int)
    assert train_result["cases_total"] > 0
    assert calibration_result["cases_total"] > 0
    assert np.array_equal(circuit._afferent_targets, codebook.assemblies)


def test_explicit_mode_leaves_status_quo_wiring_untouched() -> None:
    curriculum, codebook, circuit = _grow_once("explicit")
    reference_config = HARNESS.RecurrentCircuitConfig(
        reservoir_cells=N_E,
        input_fanout=HARNESS._INPUT_FANOUT,
        recurrent_fanout=HARNESS._RECURRENT_FANOUT,
        random_seed=7,
        readout_learning_rule="ridge",
    )
    reference = HARNESS.RecurrentCellCircuit(
        HARNESS._TOKEN_CELLS, config=reference_config
    )
    assert codebook.assemblies is None
    assert np.array_equal(
        circuit._afferent_targets, reference._afferent_targets
    )


# ---------------------------------------------------------------------------
# 4. Framing-order preservation under re-encoding
# ---------------------------------------------------------------------------


def test_reencoding_preserves_framing_order_token_sequence() -> None:
    from humanbrain import typed_program

    curriculum = HARNESS.build_curriculum(7, 21)
    tasks = curriculum["train_tasks"]
    framing_order = typed_program.Layer1Protocol.for_lane("A2").framing_order

    # Independent recomposition of the A2 framing order per task.
    context_delimiter = typed_program._CONTEXT_DELIMITER_ID
    input_delimiter = typed_program._INPUT_DELIMITER_ID
    prepared = HARNESS._prepare_sequences(tasks)
    assert len(prepared) == len(tasks)
    for task, (input_ids, teacher_ids) in zip(tasks, prepared):
        signature_bytes = typed_program._signature_bytes(task.signature)
        expected_input = (
            *signature_bytes,
            context_delimiter,
            *task.prompt.encode("utf-8"),
            input_delimiter,
        )
        assert input_ids == expected_input
        assert teacher_ids == typed_program._program_token_ids(
            task.teacher_tokens
        )

    # The codebook maps ids elementwise: driving one full completion through
    # the grown circuit must inject exactly the framing-order id sequence as
    # its prefix — no reordering, no deduplication, same mapping throughout.
    codebook = HARNESS.build_codebook(
        "random", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    circuit = HARNESS._grow_circuit(
        curriculum,
        n_cells=N_E,
        seed=7,
        readout_rule="ridge",
        schedule="spaced",
        supervision="program_only",
        codebook=codebook,
    )
    task = tasks[0]
    input_ids = prepared[0][0]
    observed: list[int] = []
    kernel = circuit._native_kernel
    original_observe = kernel.observe

    def spying_observe(token_id: int) -> int:
        observed.append(int(token_id))
        return original_observe(token_id)

    kernel.observe = spying_observe  # type: ignore[method-assign]
    try:
        HARNESS._complete_program(circuit, input_ids, task.signature, None)
    finally:
        del kernel.observe  # restore the bound method
    assert observed[: len(input_ids)] == list(input_ids)
    assert len(observed) >= len(input_ids)
    # And the injected cell populations followed the codebook rows for those
    # exact ids, in the same order (elementwise mapping, sequence preserved).
    assert codebook.assemblies is not None
    expected_rows = codebook.assemblies[np.asarray(input_ids, dtype=np.int64)]
    installed = circuit._afferent_targets[np.asarray(input_ids, dtype=np.int64)]
    assert np.array_equal(expected_rows, installed)


# ---------------------------------------------------------------------------
# 5. CLI gates
# ---------------------------------------------------------------------------


_BASE_ARGS = [
    "--experiment",
    "codebook-gate",
    "--arm",
    "gate",
    "--N",
    str(N_E),
    "--seed",
    "7",
]


def test_cli_rejects_unknown_codebook(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        HARNESS.main([*_BASE_ARGS, "--out", str(tmp_path), "--codebook", "bogus"])


def test_cli_rejects_assembly_k_larger_than_population(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        HARNESS.main(
            [
                *_BASE_ARGS,
                "--out",
                str(tmp_path),
                "--codebook",
                "random",
                "--assembly-k",
                str(N_E + 1),
            ]
        )


def test_cli_rejects_custom_assembly_k_for_explicit_arm(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        HARNESS.main(
            [
                *_BASE_ARGS,
                "--out",
                str(tmp_path),
                "--codebook",
                "explicit",
                "--assembly-k",
                "8",
            ]
        )


def test_run_experiment_records_codebook_provenance(tmp_path: Path) -> None:
    """config_dump carries the effective codebook knobs for the metrics schema."""
    record = HARNESS.run_experiment(
        experiment_id="codebook-gate",
        arm="random-arm",
        n_cells=N_E,
        seed=7,
        readout_rule="ridge",
        schedule="spaced",
        data_scale=21,
        supervision="program_only",
        codebook="random",
        assembly_k=ASSEMBLY_K,
    )
    dump = record["config_dump"]
    assert dump["codebook"] == "random"
    assert dump["assembly_k"] == ASSEMBLY_K
    expected = HARNESS.build_codebook(
        "random", token_count=TOKEN_COUNT, n_e=N_E, k=ASSEMBLY_K, seed=7
    )
    assert dump["codebook_sha256"] == expected.sha256
    assert record["readout_rule"] == "ridge"
