"""Post-hoc degeneracy probe for codebook_compare (read-only reuse of harness).

For every arm config, regrows the circuit from the recorded arguments and
completes all 18 CALIBRATION tasks teacher-free, then measures how many
DISTINCT programs each arm emits and how much of its case-passing comes from
the modal (degenerate) program. Mirrors the post-hoc probe documented in
experiments/readout_compare/FINDINGS.md.

Run from the repo root of the pinned worktree:
    python experiments/codebook_compare/_probe_degeneracy.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import ladder_experiments as lx  # noqa: E402

ARMS = ("explicit", "random", "similarity")
NS = (128, 512, 1024)
SEEDS = (7, 8, 9)


def probe_arm(codebook: str, n_cells: int, seed: int) -> dict:
    curriculum = lx.build_curriculum(seed, 42)
    codebook_arm = lx.build_codebook(
        codebook,
        token_count=lx._TOKEN_CELLS,
        n_e=n_cells,
        k=lx._DEFAULT_ASSEMBLY_K,
        seed=seed,
    )
    circuit = lx._grow_circuit(
        curriculum,
        n_cells=n_cells,
        seed=seed,
        readout_rule="ridge",
        schedule="massed",
        supervision="all_transition",
        codebook=codebook_arm,
    )
    protocol = lx.typed_program.Layer1Protocol.for_lane("A2")
    programs: list[tuple[tuple[str, ...], int, int]] = []
    for task in curriculum["calibration_tasks"]:
        input_ids = lx.typed_program._model_input(
            task.prompt, task.signature, framing_order=protocol.framing_order
        )
        try:
            tokens, _, _ = lx._complete_program(circuit, input_ids, task.signature, None)
        except Exception:
            tokens = ("<ERROR>",)
        passed, total = lx._case_results(task, tokens) if tokens != ("<ERROR>",) else (0, len(task.cases))
        programs.append((tokens, passed, total))

    counter = Counter(tokens for tokens, _, _ in programs)
    modal_tokens, modal_count = counter.most_common(1)[0]
    total_cases = sum(t for _, _, t in programs)
    passed_cases = sum(p for _, p, _ in programs)
    modal_passed = sum(p for tokens, p, _ in programs if tokens == modal_tokens)
    return {
        "arm": f"{codebook}_N{n_cells}_s{seed}",
        "distinct_programs": len(counter),
        "modal_program": list(modal_tokens)[:6],
        "modal_program_len": len(modal_tokens),
        "modal_task_share": round(modal_count / len(programs), 4),
        "cases_passed_total": passed_cases,
        "cases_passed_by_modal": modal_passed,
        "cases_passed_non_modal": passed_cases - modal_passed,
        "cases_total": total_cases,
    }


def main() -> int:
    results = []
    for codebook in ARMS:
        for n_cells in NS:
            for seed in SEEDS:
                results.append(probe_arm(codebook, n_cells, seed))
                print(json.dumps(results[-1]), flush=True)
    out = Path(__file__).parent / "_probe_degeneracy_results.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    print("\n## summary: mean over seeds")
    print("| arm | N | distinct programs (mean) | modal share | calib passed | by modal | non-modal |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for codebook in ARMS:
        for n_cells in NS:
            rows = [r for r in results if r["arm"].startswith(f"{codebook}_N{n_cells}_")]
            dp = sum(r["distinct_programs"] for r in rows) / len(rows)
            ms = sum(r["modal_task_share"] for r in rows) / len(rows)
            cp = sum(r["cases_passed_total"] for r in rows) / len(rows)
            bm = sum(r["cases_passed_by_modal"] for r in rows) / len(rows)
            nm = sum(r["cases_passed_non_modal"] for r in rows) / len(rows)
            print(
                f"| {codebook} | {n_cells} | {dp:.2f} | {ms:.3f} | {cp:.2f} | {bm:.2f} | {nm:.2f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
