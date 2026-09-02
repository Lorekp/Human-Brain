"""Frozen opaque fixture for recurrent assembly-stack composition.

The first four public experiences form one parity subset of three binary opaque
factors.  Every row carries a different payload token, and that same token
occurs in the demonstrated response.  The fifth experience is a direct
one-token response route.  The learner receives only complete observation and
response sequences: no spans, groups, roles, stack operations, or
answer-bearing metadata are supplied.

Fresh scoring observations expose only their complete token sequence.  Expected
responses remain scoring-only until the runner has frozen treatment, causal
ablation, direct-route, and unknown-cue decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

SCHEMA = "recurrent-assembly-stack-fixture-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _validate_tokens(name: str, tokens: tuple[str, ...]) -> None:
    if not tokens:
        raise ValueError(f"{name} must not be empty")
    if any(
        not isinstance(token, str) or not token or token.strip() != token
        for token in tokens
    ):
        raise ValueError(f"{name} must contain non-empty trimmed strings")


@dataclass(frozen=True, slots=True)
class PublicAssemblyExperienceV1:
    """One complete public observation/success sequence."""

    experience_id: str
    observation_tokens: tuple[str, ...]
    response_tokens: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "experience_id": self.experience_id,
            "observation_tokens": self.observation_tokens,
            "response_tokens": self.response_tokens,
        }

    def authenticate(self) -> None:
        if not self.experience_id:
            raise ValueError("public experience id must not be empty")
        _validate_tokens("observation_tokens", self.observation_tokens)
        _validate_tokens("response_tokens", self.response_tokens)


@dataclass(frozen=True, slots=True)
class FreshAssemblyCaseV1:
    """One fresh observation with scoring-only response custody."""

    case_id: str
    observation_tokens: tuple[str, ...]
    scoring_only_expected_tokens: tuple[str, ...]

    def _public_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "observation_tokens": self.observation_tokens,
            "schema": SCHEMA,
        }

    def public_observation(self) -> dict[str, object]:
        record = self._public_record()
        return {**record, "observation_sha256": _digest(record)}

    def accepts(self, tokens: tuple[str, ...]) -> bool:
        return tuple(tokens) == self.scoring_only_expected_tokens

    def scoring_record(self) -> dict[str, object]:
        return {
            **self.public_observation(),
            "scoring_only_expected_tokens": self.scoring_only_expected_tokens,
        }

    def authenticate(self) -> None:
        if not self.case_id:
            raise ValueError("fresh case id must not be empty")
        _validate_tokens("observation_tokens", self.observation_tokens)
        _validate_tokens(
            "scoring_only_expected_tokens",
            self.scoring_only_expected_tokens,
        )


@dataclass(frozen=True, slots=True)
class AssemblyStopProbeV1:
    """One unsupported opaque observation; only explicit STOP is accepted."""

    probe_id: str
    observation_tokens: tuple[str, ...]

    def _public_record(self) -> dict[str, object]:
        return {
            "observation_tokens": self.observation_tokens,
            "probe_id": self.probe_id,
            "schema": SCHEMA,
        }

    def public_observation(self) -> dict[str, object]:
        record = self._public_record()
        return {**record, "observation_sha256": _digest(record)}

    def scoring_record(self) -> dict[str, object]:
        return {
            **self.public_observation(),
            "scoring_only_expected_behavior": "physical_explicit_stop",
        }

    def authenticate(self) -> None:
        if not self.probe_id:
            raise ValueError("stop probe id must not be empty")
        _validate_tokens("observation_tokens", self.observation_tokens)


PUBLIC_EXPERIENCES = (
    PublicAssemblyExperienceV1(
        experience_id="experience-00",
        observation_tokens=("nivo", "bexi", "cora", "payload-amber-17"),
        response_tokens=(
            "qev",
            "dax",
            "rax",
            "payload-amber-17",
            "tib",
            "pif",
            "mur",
        ),
    ),
    PublicAssemblyExperienceV1(
        experience_id="experience-01",
        observation_tokens=("nivo", "toma", "lune", "payload-cobalt-29"),
        response_tokens=(
            "qev",
            "kud",
            "wex",
            "payload-cobalt-29",
            "yop",
            "zan",
            "mur",
        ),
    ),
    PublicAssemblyExperienceV1(
        experience_id="experience-02",
        observation_tokens=("raku", "bexi", "lune", "payload-ivory-43"),
        response_tokens=(
            "sot",
            "dax",
            "wex",
            "payload-ivory-43",
            "yop",
            "pif",
            "vel",
        ),
    ),
    PublicAssemblyExperienceV1(
        experience_id="experience-03",
        observation_tokens=("raku", "toma", "cora", "payload-lilac-61"),
        response_tokens=(
            "sot",
            "kud",
            "rax",
            "payload-lilac-61",
            "tib",
            "zan",
            "vel",
        ),
    ),
    PublicAssemblyExperienceV1(
        experience_id="experience-04",
        observation_tokens=("ATOM", "payload-saffron-73"),
        response_tokens=("payload-saffron-73",),
    ),
)

FRESH_NESTED_CASES = (
    FreshAssemblyCaseV1(
        case_id="fresh-nested-00",
        observation_tokens=("nivo", "bexi", "lune", "payload-quartz-83"),
        scoring_only_expected_tokens=(
            "qev",
            "dax",
            "wex",
            "payload-quartz-83",
            "yop",
            "pif",
            "mur",
        ),
    ),
    FreshAssemblyCaseV1(
        case_id="fresh-nested-01",
        observation_tokens=("raku", "toma", "lune", "payload-thistle-97"),
        scoring_only_expected_tokens=(
            "sot",
            "kud",
            "wex",
            "payload-thistle-97",
            "yop",
            "zan",
            "vel",
        ),
    ),
)

FRESH_ATOM_CASE = FreshAssemblyCaseV1(
    case_id="fresh-atom-00",
    observation_tokens=("ATOM", "payload-umber-101"),
    scoring_only_expected_tokens=("payload-umber-101",),
)

UNKNOWN_CUE_PROBE = AssemblyStopProbeV1(
    probe_id="unknown-cue-00",
    observation_tokens=("ulvek", "sorin", "drav", "payload-xanthic-109"),
)


def _authenticate_fixture() -> None:
    if len(PUBLIC_EXPERIENCES) != 5:
        raise ValueError("fixture requires four nested and one direct experience")
    for experience in PUBLIC_EXPERIENCES:
        experience.authenticate()
    for case in (*FRESH_NESTED_CASES, FRESH_ATOM_CASE):
        case.authenticate()
    UNKNOWN_CUE_PROBE.authenticate()

    nested = PUBLIC_EXPERIENCES[:4]
    training_factor_triples = {
        experience.observation_tokens[:3] for experience in nested
    }
    if training_factor_triples != {
        ("nivo", "bexi", "cora"),
        ("nivo", "toma", "lune"),
        ("raku", "bexi", "lune"),
        ("raku", "toma", "cora"),
    }:
        raise ValueError("nested public experiences lost the frozen parity subset")
    if any(len(experience.response_tokens) != 7 for experience in nested):
        raise ValueError("nested public response width changed")
    interval_tokens = {
        0: {
            "nivo": ("qev", "mur"),
            "raku": ("sot", "vel"),
        },
        1: {
            "bexi": ("dax", "pif"),
            "toma": ("kud", "zan"),
        },
        2: {
            "cora": ("rax", "tib"),
            "lune": ("wex", "yop"),
        },
    }
    for factor_index, variants in interval_tokens.items():
        support = tuple(
            experience.observation_tokens[factor_index]
            for experience in nested
        )
        if set(support) != set(variants) or any(
            support.count(variant) != 2 for variant in variants
        ):
            raise ValueError("each nested factor variant needs repeated support")
    for experience in nested:
        cue = experience.observation_tokens
        response = experience.response_tokens
        expected = (
            interval_tokens[0][cue[0]][0],
            interval_tokens[1][cue[1]][0],
            interval_tokens[2][cue[2]][0],
            cue[3],
            interval_tokens[2][cue[2]][1],
            interval_tokens[1][cue[1]][1],
            interval_tokens[0][cue[0]][1],
        )
        if response != expected:
            raise ValueError("nested interval/copy structure changed")
    if PUBLIC_EXPERIENCES[4].observation_tokens[0] != "ATOM":
        raise ValueError("direct public route lost its opaque ATOM cue")
    if len(PUBLIC_EXPERIENCES[4].response_tokens) != 1:
        raise ValueError("direct public response must contain exactly one token")

    training_payloads = tuple(
        experience.observation_tokens[-1] for experience in PUBLIC_EXPERIENCES
    )
    if len(set(training_payloads)) != len(training_payloads):
        raise ValueError("every public experience must use a unique payload")
    if any(
        experience.response_tokens.count(payload) != 1
        for experience, payload in zip(
            PUBLIC_EXPERIENCES,
            training_payloads,
            strict=True,
        )
    ):
        raise ValueError("public payload equality/copy evidence changed")

    scoring_observations = (
        *(case.observation_tokens for case in FRESH_NESTED_CASES),
        FRESH_ATOM_CASE.observation_tokens,
        UNKNOWN_CUE_PROBE.observation_tokens,
    )
    scoring_payloads = tuple(tokens[-1] for tokens in scoring_observations)
    if set(training_payloads) & set(scoring_payloads):
        raise ValueError("fresh payload leaked into public experiences")
    if len(set(scoring_payloads)) != len(scoring_payloads):
        raise ValueError("fresh payloads must be pairwise distinct")
    if set(UNKNOWN_CUE_PROBE.observation_tokens[:-1]) & {
        token
        for experience in PUBLIC_EXPERIENCES
        for token in experience.observation_tokens[:-1]
    }:
        raise ValueError("unknown probe reuses a trained cue token")
    fresh_factor_triples = {
        case.observation_tokens[:3] for case in FRESH_NESTED_CASES
    }
    if fresh_factor_triples != {
        ("nivo", "bexi", "lune"),
        ("raku", "toma", "lune"),
    }:
        raise ValueError("fresh nested factor triples changed")
    if training_factor_triples & fresh_factor_triples:
        raise ValueError("fresh factor triple leaked into public experiences")
    if any(
        {case.observation_tokens[index] for case in FRESH_NESTED_CASES}
        - {experience.observation_tokens[index] for experience in nested}
        for index in range(3)
    ):
        raise ValueError("fresh nested case contains an unlearned factor token")
    for case in FRESH_NESTED_CASES:
        cue = case.observation_tokens
        expected = (
            interval_tokens[0][cue[0]][0],
            interval_tokens[1][cue[1]][0],
            interval_tokens[2][cue[2]][0],
            cue[3],
            interval_tokens[2][cue[2]][1],
            interval_tokens[1][cue[1]][1],
            interval_tokens[0][cue[0]][1],
        )
        if case.scoring_only_expected_tokens != expected:
            raise ValueError("fresh scoring target lost nested interval custody")
    if FRESH_ATOM_CASE.scoring_only_expected_tokens != (
        FRESH_ATOM_CASE.observation_tokens[-1],
    ):
        raise ValueError("fresh direct scoring target lost payload copy custody")


_authenticate_fixture()

_CUSTODY = {
    "fresh_atom_case": FRESH_ATOM_CASE.scoring_record(),
    "fresh_nested_cases": tuple(
        case.scoring_record() for case in FRESH_NESTED_CASES
    ),
    "public_experiences": tuple(
        experience.record() for experience in PUBLIC_EXPERIENCES
    ),
    "schema": SCHEMA,
    "unknown_cue_probe": UNKNOWN_CUE_PROBE.scoring_record(),
}
RECURRENT_ASSEMBLY_STACK_FIXTURE_SHA256 = _digest(_CUSTODY)


def authenticate_fixture() -> dict[str, object]:
    """Revalidate the frozen fixture and return non-model custody metadata."""

    _authenticate_fixture()
    return {
        "fixture_sha256": RECURRENT_ASSEMBLY_STACK_FIXTURE_SHA256,
        "fresh_atom_case_count": 1,
        "fresh_nested_case_count": len(FRESH_NESTED_CASES),
        "public_experience_count": len(PUBLIC_EXPERIENCES),
        "schema": SCHEMA,
        "unknown_cue_probe_count": 1,
    }


__all__ = [
    "FRESH_ATOM_CASE",
    "FRESH_NESTED_CASES",
    "PUBLIC_EXPERIENCES",
    "RECURRENT_ASSEMBLY_STACK_FIXTURE_SHA256",
    "SCHEMA",
    "UNKNOWN_CUE_PROBE",
    "AssemblyStopProbeV1",
    "FreshAssemblyCaseV1",
    "PublicAssemblyExperienceV1",
    "authenticate_fixture",
]
