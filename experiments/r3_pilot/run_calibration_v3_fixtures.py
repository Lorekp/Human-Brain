"""Mechanically disabled official runner for the frozen calibration-v3 plan.

Importing this module is transport-only: it does not import native execution,
corpus, tokenizer, or historical C1 runner code.  The sole execution entry
point first obtains a source-current runnable authority.  The governing
protocol is deliberately non-runnable today, so no execution or artifact seam
can be reached.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence


EXIT_OK = 0
EXIT_STOP = 2


class FixtureRunnerStop(RuntimeError):
    """A fail-closed pre-outcome or protocol STOP."""


@dataclass(frozen=True, slots=True)
class FrozenPlanAuthority:
    """Authority handed to the future outcome implementation exactly once.

    ``protocol`` is the canonical object that passed all three governing
    validations.  The executor must treat it as immutable; the two retained
    identities bind result and checkpoint envelopes without rebuilding an
    authority from unvalidated digest strings.
    """

    repository_root: Path
    protocol: dict[str, Any]
    protocol_sha256: str
    source_bundle_identity_sha256: str


class _FailClosedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise FixtureRunnerStop(f"invalid command: {message}")


def _governing_protocol() -> ModuleType:
    if __package__:
        return importlib.import_module(f"{__package__}.calibration_v3_protocol")
    return importlib.import_module("calibration_v3_protocol")


def _source_bundle_identity(repository_root: Path) -> str:
    """Compute the bounded source closure only after protocol validation."""

    native = importlib.import_module("humanbrain.native")
    try:
        return native.procedural_runtime_source_bundle_sha256(repository_root)
    except native.NativeBuildError as exc:
        raise FixtureRunnerStop(
            f"procedural runtime source identity is unavailable: {exc}"
        ) from exc


def _exact_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FixtureRunnerStop(f"{label} is not a lowercase SHA-256 identity")
    return value


def _authorize_frozen_plan(
    *,
    governing: ModuleType | None = None,
    source_bundle_identity: Callable[[Path], str] | None = None,
) -> FrozenPlanAuthority:
    """Run the only permitted authority sequence, in its frozen order."""

    protocol_module = _governing_protocol() if governing is None else governing
    bundle_identity = (
        _source_bundle_identity
        if source_bundle_identity is None
        else source_bundle_identity
    )

    try:
        protocol = protocol_module.build_protocol()
        protocol_module.validate_protocol(protocol, require_runnable=True)
        binding = protocol["source_binding"]
        protocol_module.validate_committed_source_blobs(
            protocol_module.REPO_ROOT, binding
        )
        protocol_module.validate_source_bytes(protocol_module.REPO_ROOT, binding)
        protocol_digest = _exact_sha256(
            "protocol identity", protocol_module.protocol_sha256()
        )
        source_digest = _exact_sha256(
            "source-bundle identity", bundle_identity(protocol_module.REPO_ROOT)
        )
    except FixtureRunnerStop:
        raise
    except Exception as exc:
        protocol_error = getattr(protocol_module, "ProtocolError", ())
        if isinstance(exc, protocol_error):
            raise FixtureRunnerStop(str(exc)) from exc
        raise

    return FrozenPlanAuthority(
        repository_root=protocol_module.REPO_ROOT,
        protocol=protocol,
        protocol_sha256=protocol_digest,
        source_bundle_identity_sha256=source_digest,
    )


def _execute_frozen_plan(authority: FrozenPlanAuthority) -> int:
    """Future execution seam; unreachable until a later source-bound change."""

    del authority
    raise FixtureRunnerStop(
        "outcome-bearing calibration-v3 execution remains disabled pre-outcome"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _FailClosedArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-frozen-plan",
        action="store_true",
        required=True,
        help="run the complete source-bound plan once it is explicitly enabled",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if not args.run_frozen_plan:  # Defensive against parser substitution.
            raise FixtureRunnerStop("the frozen-plan execution flag is required")
        authority = _authorize_frozen_plan()
        return _execute_frozen_plan(authority)
    except FixtureRunnerStop as exc:
        print(f"calibration-v3 fixture runner: STOP: {exc}", file=sys.stderr)
        return EXIT_STOP


if __name__ == "__main__":
    raise SystemExit(main())
