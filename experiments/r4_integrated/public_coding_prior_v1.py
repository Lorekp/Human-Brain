"""Small public supervised prior for generic Python coding form."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot.fresh_coding_checkpoint import (
    CORPUS_PATH,
    MANIFEST_PATH,
    load_corpus,
)

SCHEMA = "public-coding-prior-v1"


@dataclass(frozen=True, slots=True)
class PublicCodingPriorCaseV1:
    """One fully public prompt-to-successful-source supervision case."""

    public_prompt: str
    entrypoint: str
    starter_source: str
    successful_source: str


def _case(
    public_prompt: str,
    entrypoint: str,
    starter_source: str,
    successful_source: str,
) -> PublicCodingPriorCaseV1:
    definition_prefix = f"def {entrypoint}("
    if (
        not public_prompt
        or not entrypoint.isidentifier()
        or not starter_source.startswith(definition_prefix)
        or not successful_source.startswith(definition_prefix)
    ):
        raise ValueError("public coding prior case is not exact")
    return PublicCodingPriorCaseV1(
        public_prompt=public_prompt,
        entrypoint=entrypoint,
        starter_source=starter_source,
        successful_source=successful_source,
    )


_SOURCE_CORPUS = load_corpus()
_SOURCE_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
SOURCE_CORPUS_SHA256 = sha256(CORPUS_PATH.read_bytes()).hexdigest()
SOURCE_MANIFEST_SHA256 = sha256(MANIFEST_PATH.read_bytes()).hexdigest()
SOURCE_FREEZE_COMMIT = _SOURCE_MANIFEST["freeze_source_commit"]
if (
    _SOURCE_MANIFEST["corpus"]["sha256"] != SOURCE_CORPUS_SHA256
    or _SOURCE_MANIFEST["scope"]["target_tasks_present_at_freeze"] is not False
):
    raise ValueError("frozen public coding corpus provenance failed")

_PUBLIC_CASES = tuple(
    _case(
        public_prompt=example["task"],
        entrypoint=_SOURCE_CORPUS["protocol"]["entrypoint"],
        starter_source=example["initial_source"],
        successful_source=example["revision_source"],
    )
    for example in _SOURCE_CORPUS["examples"]
)


def _record(case: PublicCodingPriorCaseV1) -> dict[str, str]:
    return {
        "entrypoint": case.entrypoint,
        "public_prompt": case.public_prompt,
        "starter_source": case.starter_source,
        "successful_source": case.successful_source,
    }


def _corpus_sha256() -> str:
    payload = json.dumps(
        {"cases": [_record(case) for case in _PUBLIC_CASES], "schema": SCHEMA},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(b"humanbrain-public-coding-prior-v1\0" + payload).hexdigest()


PUBLIC_CODING_PRIOR_V1_SHA256 = _corpus_sha256()


def public_coding_prior_v1() -> tuple[PublicCodingPriorCaseV1, ...]:
    """Return the immutable, deterministic public supervision corpus."""

    return _PUBLIC_CASES
