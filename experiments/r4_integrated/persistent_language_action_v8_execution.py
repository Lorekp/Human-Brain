"""Public immutable-driver proposal policy over the qualified V2 executor.

The host rejects direct driver edits before materialization. This is not an OS
sandbox: code in other files can still modify the driver during execution.
Actual after-files and that separately reported integrity outcome are retained.
No request interpretation, expected output or private grade enters execution.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from experiments.r4_integrated import persistent_language_action_execution_v2 as execution
from experiments.r4_integrated.persistent_language_action_execution_v2 import (
    feedback_from_receipt as feedback_from_receipt,
)
from experiments.r4_integrated.persistent_language_action_execution_v2 import (
    qualified_launch as qualified_launch,
)
from humanbrain.sequence_assembly_observation import SpliceProposal, SpliceWorkspace

POLICY = "persistent-language-action-public-driver-v1"
PROTECTED_PATH = "main.py"
POLICY_REJECTION = "ValueError: public driver main.py cannot be a SPLICE target"


def _seal(record: dict) -> str:
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(b"humanbrain.public-driver-execution.v1\0" + raw).hexdigest()


def _driver(records: list[dict]) -> dict | None:
    """Describe the public driver bytes, without inspecting their program meaning."""
    matches = [row for row in records if row["path"].casefold() == PROTECTED_PATH]
    if len(matches) != 1:
        return None
    row = matches[0]
    raw = bytes.fromhex(row["content_hex"])
    return {"path": row["path"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def execute_proposal(
    workspace: SpliceWorkspace | list[dict],
    proposal: SpliceProposal | None,
    output: Path,
    *,
    launch: execution.runtime.LaunchRecord,
    deadline: float,
) -> tuple[list[dict], dict]:
    """Apply once or verify unchanged once, with the same public policy per route.

    A protected SPLICE delegates V2's existing no-proposal verification path.
    The final receipt binds the submitted proposal while separately identifying
    the delegated no-proposal call and its original rejection. The child,
    runtime, deadline, streams, snapshots and all actual outcomes are V2's.
    Stable execution-v2 receipt semantics remain compatible with the acting
    learner: rejected application means proposed_files equals before_files.
    """
    started = time.perf_counter()
    # Do not convert unsupported workspace/proposal combinations into an allowed
    # no-proposal call. V2 retains every original type/path/span/custody check.
    blocked = (
        isinstance(workspace, SpliceWorkspace)
        and isinstance(proposal, SpliceProposal)
        and proposal.operation == "SPLICE"
        and proposal.path.casefold() == PROTECTED_PATH
    )
    delegated = None if blocked else proposal
    after, receipt = execution.execute_proposal(
        workspace, delegated, output, launch=launch, deadline=deadline)
    delegated_sha = receipt["proposal_sha256"]
    delegated_rejection = receipt["rejection"]
    delegated_seconds = receipt["elapsed_seconds"]
    if blocked:
        # No replacement bytes were applied or written by the delegate. Preserve
        # its actual unchanged verification, including any child-side changes.
        if receipt["accepted"] or receipt["proposed_files"] != receipt["before_files"]:
            raise RuntimeError("unchanged verification violated the protected-driver contract")
        receipt["proposal_sha256"] = proposal.sha256
        receipt["rejection"] = POLICY_REJECTION
    before_driver = _driver(receipt["before_files"])
    proposed_driver = _driver(receipt["proposed_files"])
    after_driver = _driver(receipt["after_files"])
    policy = {
        "schema": POLICY,
        "protected_path": PROTECTED_PATH,
        "path_comparison": "casefold exact relative path",
        "proposal_rejected": blocked,
        "submitted_proposal_sha256": proposal.sha256 if proposal is not None else None,
        "delegated_proposal_sha256": delegated_sha,
        "delegated_rejection": delegated_rejection,
        "before_driver": before_driver,
        "proposed_driver": proposed_driver,
        "after_driver": after_driver,
        "driver_integrity": (
            receipt["files_complete"] and before_driver is not None
            and proposed_driver == before_driver and after_driver == before_driver
        ),
        "scope": "proposal path rejection and captured public byte integrity; no OS sandbox",
    }
    receipt["public_driver_policy"] = policy
    receipt["public_driver_policy_sha256"] = _seal(policy)
    receipt["delegated_execution_seconds"] = delegated_seconds
    receipt["elapsed_seconds"] = time.perf_counter() - started
    return after, receipt
