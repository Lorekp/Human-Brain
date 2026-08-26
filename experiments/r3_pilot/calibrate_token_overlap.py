#!/usr/bin/env python3
"""Build the pre-outcome R3 token-overlap calibration artifact.

The tokenizer freeze section 5 asks for a deterministic hypergeometric
reference for the probability that at most 20 grown contacts overlap a
20-id active context at theta_m=10 or theta_a=13.  This harness uses only
the frozen mechanism constants and the already-committed concept-composition
statistics.  It never reads C1 logs, checkpoints, readiness files, or verdicts.

The primary law is exact, not Monte Carlo:

    X ~ Hypergeometric(N=distinct ids, K=20 active ids, n=contacts)

The 20 active concept ids are identified by results_v1.json's
top20_concepts backbone.  Their mined n-gram counts are preserved as
provenance but are not treated as emitted-token probabilities: mined n-grams
overlap, so normalizing those counts across the 439-id inventory would invent
an unsupported sampling distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanbrain.concept_tokens import CONCEPT_ID_BASE
from humanbrain.segment_growth import SegmentGrowthTracker


COMPOSITION_RESULTS = (
    REPO_ROOT / "experiments" / "concept_sdr_composition" / "results_v1.json"
)
TOKENIZER_FREEZE = REPO_ROOT / "docs" / "research" / "r3-tokenizer-freeze.md"
SEGMENT_PREREG = (
    REPO_ROOT / "docs" / "research" / "r3-segment-growth-preregistration.md"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "token_overlap_calibration.json"

ARTIFACT_ID = "r3-token-overlap-calibration-v1"
ACTIVE_CONTEXT_IDS = 20
BYTE_UNIVERSE = 74
CONCEPT_UNIVERSE = 439
CONCEPT_BAND_IDS = 367


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hypergeometric_pmf(
    population: int,
    active: int,
    draws: int,
    overlap: int,
) -> Fraction:
    """Return the exact hypergeometric probability P(X=overlap)."""
    if not 0 <= active <= population:
        raise ValueError("active must be in [0, population]")
    if not 0 <= draws <= population:
        raise ValueError("draws must be in [0, population]")
    low = max(0, draws - (population - active))
    high = min(active, draws)
    if not low <= overlap <= high:
        return Fraction(0, 1)
    return Fraction(
        math.comb(active, overlap)
        * math.comb(population - active, draws - overlap),
        math.comb(population, draws),
    )


def hypergeometric_tail(
    population: int,
    active: int,
    draws: int,
    threshold: int,
) -> Fraction:
    """Return exact P(X>=threshold) for the hypergeometric law."""
    return sum(
        (
            hypergeometric_pmf(population, active, draws, overlap)
            for overlap in range(threshold, min(active, draws) + 1)
        ),
        Fraction(0, 1),
    )


def _number(value: Fraction) -> dict[str, Any]:
    return {
        "decimal": float(value),
        "exact_fraction": f"{value.numerator}/{value.denominator}",
    }


def _contact_row(
    population: int,
    contacts: int,
    theta_match: int,
    theta_active: int,
) -> dict[str, Any]:
    expected = Fraction(contacts * ACTIVE_CONTEXT_IDS, population)
    return {
        "contacts": contacts,
        "expected_active_overlap": _number(expected),
        "p_ge_theta_m_10": _number(
            hypergeometric_tail(
                population, ACTIVE_CONTEXT_IDS, contacts, theta_match
            )
        ),
        "p_ge_theta_a_13": _number(
            hypergeometric_tail(
                population, ACTIVE_CONTEXT_IDS, contacts, theta_active
            )
        ),
    }


def _regime(
    *,
    name: str,
    population: int,
    basis: str,
    theta_match: int,
    theta_active: int,
    max_contacts: int,
) -> dict[str, Any]:
    rows = [
        _contact_row(population, contacts, theta_match, theta_active)
        for contacts in range(max_contacts + 1)
    ]
    pmf = [
        {
            "active_overlap": overlap,
            "probability": _number(
                hypergeometric_pmf(
                    population, ACTIVE_CONTEXT_IDS, max_contacts, overlap
                )
            ),
        }
        for overlap in range(max_contacts + 1)
    ]
    return {
        "name": name,
        "distinct_id_universe": population,
        "basis": basis,
        "active_context_ids": ACTIVE_CONTEXT_IDS,
        "contact_curve_0_to_20": rows,
        "overlap_pmf_at_20_contacts": pmf,
        "at_20_contacts": rows[-1],
    }


def _validate_regime(
    regime: dict[str, Any],
    theta_match: int,
    theta_active: int,
) -> dict[str, Any]:
    population = int(regime["distinct_id_universe"])
    previous_match = Fraction(0, 1)
    previous_active = Fraction(0, 1)
    max_float_sum_error = 0.0
    for row in regime["contact_curve_0_to_20"]:
        contacts = int(row["contacts"])
        probabilities = [
            hypergeometric_pmf(
                population, ACTIVE_CONTEXT_IDS, contacts, overlap
            )
            for overlap in range(contacts + 1)
        ]
        _require(sum(probabilities, Fraction(0, 1)) == 1, "PMF must sum to one")
        exact_mean = sum(
            (overlap * probability for overlap, probability in enumerate(probabilities)),
            Fraction(0, 1),
        )
        expected = Fraction(contacts * ACTIVE_CONTEXT_IDS, population)
        _require(exact_mean == expected, "PMF mean must equal n*K/N")

        p_match = hypergeometric_tail(
            population, ACTIVE_CONTEXT_IDS, contacts, theta_match
        )
        p_active = hypergeometric_tail(
            population, ACTIVE_CONTEXT_IDS, contacts, theta_active
        )
        _require(p_match >= previous_match, "theta_m tail must be monotone")
        _require(p_active >= previous_active, "theta_a tail must be monotone")
        _require(p_active <= p_match, "theta_a tail cannot exceed theta_m tail")
        if contacts < theta_match:
            _require(p_match == 0, "theta_m tail must be zero below threshold")
        if contacts < theta_active:
            _require(p_active == 0, "theta_a tail must be zero below threshold")
        previous_match = p_match
        previous_active = p_active
        max_float_sum_error = max(
            max_float_sum_error,
            abs(sum(float(probability) for probability in probabilities) - 1.0),
        )
    return {
        "all_exact_pmf_sums_equal_one": True,
        "all_exact_means_equal_n_times_k_over_N": True,
        "tails_monotone_in_contact_count": True,
        "theta_a_tail_never_exceeds_theta_m_tail": True,
        "tails_zero_below_their_thresholds": True,
        "max_float_pmf_sum_error": max_float_sum_error,
    }


def build_artifact() -> dict[str, Any]:
    results = json.loads(COMPOSITION_RESULTS.read_text(encoding="utf-8"))
    _require(
        results.get("study") == "concept-sdr-composition-v1",
        "unexpected composition-study id",
    )
    distinct = results["token_stats"]["distinct_ids_used"]
    _require(distinct["byte_full"] == BYTE_UNIVERSE, "byte id universe drift")
    _require(distinct["concept"] == CONCEPT_UNIVERSE, "concept id universe drift")
    _require(
        distinct["concept_band_ids_used"] == CONCEPT_BAND_IDS,
        "concept-band id count drift",
    )
    _require(CONCEPT_ID_BASE == 1024, "concept id-band base drift")
    top20 = results["top20_concepts"]
    _require(len(top20) == ACTIVE_CONTEXT_IDS, "top20 backbone must contain 20 ids")
    _require(
        len({row["text"] for row in top20}) == ACTIVE_CONTEXT_IDS,
        "top20 backbone texts must be unique",
    )

    theta_active = int(SegmentGrowthTracker.THETA_ACTIVE)
    theta_match = int(SegmentGrowthTracker.THETA_MATCH)
    max_contacts = int(SegmentGrowthTracker.MAX_SYNAPSES_PER_SEGMENT)
    _require(theta_active == 13, "theta_a drift")
    _require(theta_match == 10, "theta_m drift")
    _require(max_contacts == 20, "contact-budget drift")

    counts = [int(row["count"]) for row in top20]
    count_sum = sum(counts)
    count_square_sum = sum(count * count for count in counts)
    effective_backbone = Fraction(count_sum * count_sum, count_square_sum)
    backbone_items = [
        {
            "rank": rank,
            "concept_token_id": CONCEPT_ID_BASE + rank - 1,
            "text": row["text"],
            "mined_ngram_count": int(row["count"]),
            "mi_bits": float(row["mi_bits"]),
            "share_within_top20_counts": _number(
                Fraction(int(row["count"]), count_sum)
            ),
        }
        for rank, row in enumerate(top20, start=1)
    ]

    regimes = [
        _regime(
            name="byte_74_ids",
            population=BYTE_UNIVERSE,
            basis="results_v1 token_stats.distinct_ids_used.byte_full",
            theta_match=theta_match,
            theta_active=theta_active,
            max_contacts=max_contacts,
        ),
        _regime(
            name="concept_stack_439_ids",
            population=CONCEPT_UNIVERSE,
            basis=(
                "results_v1 token_stats.distinct_ids_used.concept; active "
                "20-id backbone identified by top20_concepts"
            ),
            theta_match=theta_match,
            theta_active=theta_active,
            max_contacts=max_contacts,
        ),
    ]
    validations = {
        regime["name"]: _validate_regime(regime, theta_match, theta_active)
        for regime in regimes
    }

    byte_at_20 = regimes[0]["at_20_contacts"]
    concept_at_20 = regimes[1]["at_20_contacts"]
    byte_match = hypergeometric_tail(
        BYTE_UNIVERSE, ACTIVE_CONTEXT_IDS, max_contacts, theta_match
    )
    concept_match = hypergeometric_tail(
        CONCEPT_UNIVERSE, ACTIVE_CONTEXT_IDS, max_contacts, theta_match
    )
    byte_active = hypergeometric_tail(
        BYTE_UNIVERSE, ACTIVE_CONTEXT_IDS, max_contacts, theta_active
    )
    concept_active = hypergeometric_tail(
        CONCEPT_UNIVERSE, ACTIVE_CONTEXT_IDS, max_contacts, theta_active
    )

    return {
        "artifact_id": ARTIFACT_ID,
        "status": "pre-replay diagnostic; non-gating",
        "pilot_outcome_data_used": False,
        "inputs_limited_to_frozen_pre_pilot_sources": True,
        "generated_by": "experiments/r3_pilot/calibrate_token_overlap.py",
        "sources": {
            "tokenizer_freeze": {
                "path": "docs/research/r3-tokenizer-freeze.md",
                "sha256": _sha256(TOKENIZER_FREEZE),
            },
            "segment_growth_preregistration": {
                "path": "docs/research/r3-segment-growth-preregistration.md",
                "sha256": _sha256(SEGMENT_PREREG),
            },
            "composition_study": {
                "path": "experiments/concept_sdr_composition/results_v1.json",
                "sha256": _sha256(COMPOSITION_RESULTS),
                "study": results["study"],
                "corpus_sha256": results["provenance"]["trimmed_sha256"],
                "corpus_sha256_match": results["provenance"]["sha256_match"],
            },
        },
        "model": {
            "method": "exact hypergeometric enumeration; no RNG or Monte Carlo error",
            "law": "X ~ Hypergeometric(N=distinct ids, K=20 active ids, n=contacts)",
            "pmf": "C(K,x)*C(N-K,n-x)/C(N,n)",
            "active_context_ids_K": ACTIVE_CONTEXT_IDS,
            "contact_counts_n": list(range(max_contacts + 1)),
            "theta_m_potential_active": theta_match,
            "theta_a_connected_active": theta_active,
            "contact_sampling": "uniform without replacement, as frozen in section 3",
            "unit_scope": (
                "Token-id overlap abstraction required by the tokenizer freeze; "
                "not a cell-level model of the native lane's full winner set."
            ),
            "theta_a_scope": (
                "P(active overlap >= 13) is an upper bound on depolarization: "
                "the hypergeometric law does not assume contacts are connected."
            ),
            "recurrence_scope": (
                "Independent-context overlap reference only. Exact recurrence "
                "of the segment's own context is intentionally not averaged in."
            ),
        },
        "concept_backbone": {
            "size": ACTIVE_CONTEXT_IDS,
            "source_field": "results_v1.top20_concepts",
            "mined_ngram_count_sum": count_sum,
            "effective_ids_within_top20_counts": _number(effective_backbone),
            "top1_share_within_top20_counts": _number(
                Fraction(counts[0], count_sum)
            ),
            "top5_share_within_top20_counts": _number(
                Fraction(sum(counts[:5]), count_sum)
            ),
            "count_semantics": (
                "Counts identify and describe the frequent backbone. They are "
                "overlapping mined n-gram occurrences, not emitted-token "
                "probabilities, so they do not weight the hypergeometric law."
            ),
            "items": backbone_items,
        },
        "regimes": regimes,
        "comparison_at_20_contacts": {
            "byte": byte_at_20,
            "concept": concept_at_20,
            "byte_over_concept_probability_ratio": {
                "theta_m_10": _number(byte_match / concept_match),
                "theta_a_13": _number(byte_active / concept_active),
            },
            "interpretation": (
                "The concept stack sharply lowers accidental independent-context "
                "threshold crossings. This is an expected-engagement reference, "
                "not a gate and not a prediction for exact context recurrence."
            ),
        },
        "self_validation": validations,
    }


def canonical_json(artifact: dict[str, Any]) -> str:
    return json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that OUTPUT is byte-identical to a fresh source-only build",
    )
    args = parser.parse_args(argv)

    rendered = canonical_json(build_artifact())
    rendered_bytes = rendered.encode("utf-8")
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"missing calibration artifact: {args.output}")
        committed = args.output.read_bytes()
        if committed != rendered_bytes:
            raise SystemExit(
                f"calibration artifact drift: rerun {Path(__file__).name}"
            )
        print(f"OK: {args.output} is deterministic and source-current")
        return 0

    args.output.write_bytes(rendered_bytes)
    digest = hashlib.sha256(rendered_bytes).hexdigest()
    print(f"wrote {args.output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
