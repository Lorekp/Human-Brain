"""Machine-local timing for exact occurrence-neighborhood scoring."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from collections.abc import Callable, Sequence, Set
from random import Random
from statistics import median
from time import perf_counter

from humanbrain.occurrence_neighborhood_scorer import OccurrenceNeighborhoodScorer


def _scalar_score(
    source: Sequence[bytes],
    radius: int,
    learned_features: Set[tuple[int, bytes]],
    excluded_center_events: Set[bytes],
) -> tuple[int, ...]:
    return tuple(
        0
        if event in excluded_center_events
        else len(
            learned_features
            & frozenset(
                (offset, source[neighbor])
                for offset in range(-radius, radius + 1)
                if offset != 0
                if 0 <= (neighbor := center + offset) < len(source)
            )
        )
        for center, event in enumerate(source)
    )


def _workload(
    token_count: int,
    role_count: int,
    radius: int,
    seed: int,
) -> tuple[
    tuple[bytes, ...],
    tuple[frozenset[tuple[int, bytes]], ...],
    tuple[frozenset[bytes], ...],
]:
    generator = Random(seed + token_count * 101 + role_count)
    vocabulary_size = min(512, max(16, token_count // 8))
    vocabulary = tuple(f"opaque-event-{index}".encode() for index in range(vocabulary_size))
    source = tuple(generator.choice(vocabulary) for _ in range(token_count))
    offsets = tuple(offset for offset in range(-radius, radius + 1) if offset)
    roles = tuple(
        frozenset((offset, generator.choice(source)) for offset in offsets)
        for _ in range(role_count)
    )
    exclusions = tuple(
        frozenset({source[(role * 193) % token_count]}) for role in range(role_count)
    )
    return source, roles, exclusions


def _median_call_seconds(
    function: Callable[[], tuple[tuple[int, ...], ...]],
    *,
    repeats: int,
    iterations: int,
) -> float:
    function()
    samples: list[float] = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = perf_counter()
            for _ in range(iterations):
                function()
            samples.append((perf_counter() - started) / iterations)
    finally:
        if gc_enabled:
            gc.enable()
    return median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="8,16,32,64,128,256,512,1024,4096")
    parser.add_argument("--roles", type=int, default=16)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-token-roles", type=int, default=262_144)
    args = parser.parse_args()
    token_counts = tuple(int(value) for value in args.tokens.split(","))
    if not token_counts or any(value < 1 for value in token_counts):
        raise ValueError("tokens must contain positive integers")
    if args.roles < 1 or args.radius < 1 or args.repeats < 1:
        raise ValueError("roles, radius, and repeats must be positive")
    if args.target_token_roles < 1:
        raise ValueError("target-token-roles must be positive")

    measurements: list[dict[str, float | int | bool]] = []
    for token_count in token_counts:
        source, roles, exclusions = _workload(
            token_count,
            args.roles,
            args.radius,
            seed=1_947,
        )
        scorer = OccurrenceNeighborhoodScorer(source, args.radius)

        def scalar(
            current_source: tuple[bytes, ...] = source,
            current_roles: tuple[frozenset[tuple[int, bytes]], ...] = roles,
            current_exclusions: tuple[frozenset[bytes], ...] = exclusions,
            radius: int = args.radius,
        ) -> tuple[tuple[int, ...], ...]:
            return tuple(
                _scalar_score(current_source, radius, features, excluded)
                for features, excluded in zip(
                    current_roles,
                    current_exclusions,
                    strict=True,
                )
            )

        def indexed_reused(
            current_scorer: OccurrenceNeighborhoodScorer[bytes] = scorer,
            current_roles: tuple[frozenset[tuple[int, bytes]], ...] = roles,
            current_exclusions: tuple[frozenset[bytes], ...] = exclusions,
        ) -> tuple[tuple[int, ...], ...]:
            return tuple(
                current_scorer.score(features, excluded_center_events=excluded)
                for features, excluded in zip(
                    current_roles,
                    current_exclusions,
                    strict=True,
                )
            )

        def indexed_with_build(
            current_source: tuple[bytes, ...] = source,
            current_roles: tuple[frozenset[tuple[int, bytes]], ...] = roles,
            current_exclusions: tuple[frozenset[bytes], ...] = exclusions,
            radius: int = args.radius,
        ) -> tuple[tuple[int, ...], ...]:
            fresh = OccurrenceNeighborhoodScorer(current_source, radius)
            return tuple(
                fresh.score(features, excluded_center_events=excluded)
                for features, excluded in zip(
                    current_roles,
                    current_exclusions,
                    strict=True,
                )
            )

        expected = scalar()
        if indexed_reused() != expected or indexed_with_build() != expected:
            raise RuntimeError(f"indexed result diverged at {token_count} tokens")
        iterations = max(1, args.target_token_roles // (token_count * args.roles))
        scalar_seconds = _median_call_seconds(
            scalar,
            repeats=args.repeats,
            iterations=iterations,
        )
        reused_seconds = _median_call_seconds(
            indexed_reused,
            repeats=args.repeats,
            iterations=iterations,
        )
        with_build_seconds = _median_call_seconds(
            indexed_with_build,
            repeats=args.repeats,
            iterations=iterations,
        )
        measurements.append(
            {
                "tokens": token_count,
                "roles": args.roles,
                "learned_synapses_per_role": 2 * args.radius,
                "iterations_per_sample": iterations,
                "scalar_seconds": scalar_seconds,
                "indexed_reused_seconds": reused_seconds,
                "indexed_with_build_seconds": with_build_seconds,
                "speedup_reused_index": scalar_seconds / reused_seconds,
                "speedup_including_index_build": scalar_seconds / with_build_seconds,
                "scalar_token_role_scores_per_second": (token_count * args.roles / scalar_seconds),
                "indexed_token_role_scores_per_second_including_build": (
                    token_count * args.roles / with_build_seconds
                ),
                "exact_integer_parity": True,
            }
        )

    crossover_with_build = next(
        (
            int(item["tokens"])
            for item in measurements
            if item["speedup_including_index_build"] > 1.0
        ),
        None,
    )
    print(
        json.dumps(
            {
                "schema": "occurrence-neighborhood-performance-v1",
                "claim_boundary": "machine-local score-kernel timing; no capability evidence",
                "python": sys.version,
                "platform": platform.platform(),
                "protocol": {
                    "event_vocabulary_cap": 512,
                    "index_build_accounting": (
                        "reported both once-per-source construction and reused-index timing"
                    ),
                    "neighborhood_radius": args.radius,
                    "repeats": args.repeats,
                    "roles": args.roles,
                    "seed": 1_947,
                    "target_token_roles_per_sample": args.target_token_roles,
                    "timing_statistic": "median",
                },
                "crossover_tokens_including_index_build": crossover_with_build,
                "measurements": measurements,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
