"""One-shot paired k=3 boundary for the banked successor-planner PASS."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import (  # noqa: E402
    public_typed_horizon2_successor_planner as planner,
)

_VISIBLE_PATH = Path(planner.__file__).with_name(
    "public_typed_horizon2_successor_visible_probe.json"
)
_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_horizon2_successor_fresh_k3_probe.json"
)
_SPEND_MARKER_PATH = Path(__file__).with_name(
    "public_typed_horizon2_successor_fresh_k3.spend.json"
)
_VISIBLE_SHA256 = "d72e0b98de6d01f6d6901efbd82fcbba9e4f4db0d5f253d45fbf4db8b3105904"
_SUCCESSOR_SHA256 = "20cfa5885998a08583e9367b9f97bcbcc3de87e39f61f531a82195eee6987a52"
_PAIR_BANK_SHA256 = "0eee88c5c2f23d9249b131574648f5c363f1c5510810beb89aa6aa51f1229cec"
_SCORING_SHA256 = "53fef1753f8bd0fdce79123e89c2985a0d20de57229b271b1bbc856f85ab966b"
_TRAINING_SHA256 = "da48bd2d134466c63dd6bc9f29e11d52379b61df7a72552c0f3a27f8f8e618cf"
_VALUE_PROJECTION_SHA256 = (
    "a4486ac15e4fbcbc88d82a5ef0bfcfc7e8833e9b20aa5f2fff78c12bb3c298f0"
)
_PLANNER_SOURCE_SHA256 = "5a68250e3d0a0be12e9ead28273e387aa9f29b7f0ddd38cef8ecfcb731347f1b"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _authenticate_visible() -> dict[str, object]:
    _require(_VISIBLE_PATH.exists(), "visible successor-planner artifact is missing")
    raw = _VISIBLE_PATH.read_bytes()
    _require(sha256(raw).hexdigest() == _VISIBLE_SHA256, "visible artifact SHA changed")
    visible = json.loads(raw)
    training = visible.get("successor_training", {})
    tasks = visible.get("tasks", [])
    _require(visible.get("schema") == "public-typed-horizon2-successor-visible-v1", "wrong visible schema")
    _require(visible.get("verdict") == "PASS_HORIZON2_SUCCESSOR_VISIBLE", "visible matrix did not pass")
    _require(visible.get("passed_tasks") == visible.get("total_tasks") == 24, "visible matrix is not 24/24")
    _require(visible.get("successor_checkpoint_sha256") == _SUCCESSOR_SHA256, "successor checkpoint changed")
    _require(visible.get("pair_bank_sha256") == _PAIR_BANK_SHA256, "pair bank changed")
    _require(visible.get("scoring_sha256") == _SCORING_SHA256, "scoring projection changed")
    _require(training.get("training_digest") == _TRAINING_SHA256, "primitive training digest changed")
    _require(training.get("exact_value_projection_sha256") == _VALUE_PROJECTION_SHA256, "value projection changed")
    _require(training.get("observations") == training.get("local_updates") == training.get("learned_synapses") == 410, "visible successor counts changed")
    _require(len(tasks) == 24 and all(task.get("passed") for task in tasks), "visible task pass ledger changed")
    _require(all(task.get("unique_races") for task in tasks), "visible matrix contains a non-unique race")
    _require(all(task.get("successor_checkpoint_immutable") for task in tasks), "visible checkpoint mutated")
    return visible


def _acquire_spend_marker(commitment: dict[str, object]) -> bool:
    marker = {
        "schema": "public-typed-horizon2-successor-fresh-k3-spend-v1",
        "visible_artifact_sha256": _VISIBLE_SHA256,
        "successor_checkpoint_sha256": _SUCCESSOR_SHA256,
        "pair_bank_sha256": _PAIR_BANK_SHA256,
        "scoring_sha256": _SCORING_SHA256,
        "planner_source_sha256": _PLANNER_SOURCE_SHA256,
        "fresh_commitment": commitment,
        "fresh_commitment_sha256": sha256(
            planner._canonical_json(commitment)
        ).hexdigest(),
        "disposition": (
            "Canonical fresh boundary spent once marker; retain after PASS, STOP, "
            "exception, interruption, or crash."
        ),
    }
    try:
        with _SPEND_MARKER_PATH.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return False
    return True


def _fresh_specs() -> tuple[planner.TaskSpec, ...]:
    # Constructed only after persisted visible authentication and successor
    # reconstruction in main().  The paired order prevents an ABS-first or
    # SUB-first heuristic from passing both arms.
    return (
        planner.TaskSpec(
            "abs(x)-3",
            ("WRAP_ABS", "WRAP_SUB[3]"),
            ("WRAP_ABS", "WRAP_SUB[3]", "DONE"),
        ),
        planner.TaskSpec(
            "abs(x-3)",
            ("WRAP_SUB[3]", "WRAP_ABS"),
            ("WRAP_SUB[3]", "WRAP_ABS", "DONE"),
        ),
    )


def _fresh_commitment(
    arguments: tuple[int, ...],
    specs: tuple[planner.TaskSpec, ...],
) -> dict[str, object]:
    return {
        "arguments": arguments,
        "tasks": tuple(
            {
                "name": spec.name,
                "target_actions": spec.target_actions,
                "expected_actions": spec.expected_actions,
            }
            for spec in specs
        ),
        "visible_artifact_sha256": _VISIBLE_SHA256,
        "successor_checkpoint_sha256": _SUCCESSOR_SHA256,
        "pair_bank_sha256": _PAIR_BANK_SHA256,
        "scoring_sha256": _SCORING_SHA256,
        "planner_source_sha256": _PLANNER_SOURCE_SHA256,
    }


def _run_fresh(
    successor: planner.FrozenSuccessorModel,
    training: planner.SuccessorTrainingSummary,
    visible: dict[str, object],
    specs: tuple[planner.TaskSpec, ...],
    arguments: tuple[int, ...],
    commitment: dict[str, object],
) -> dict[str, object]:
    _require(sha256(_VISIBLE_PATH.read_bytes()).hexdigest() == _VISIBLE_SHA256, "visible artifact changed after authentication")
    _require(sha256(Path(planner.__file__).read_bytes()).hexdigest() == _PLANNER_SOURCE_SHA256, "planner source changed after authentication")
    _require(successor.live_checkpoint_sha256 == _SUCCESSOR_SHA256, "reconstructed successor hash changed")
    _require(training.checkpoint_sha256 == _SUCCESSOR_SHA256, "reconstructed training hash changed")
    _require(
        planner._canonical_json(commitment)
        == planner._canonical_json(_fresh_commitment(arguments, specs)),
        "fresh run differs from the durable spend commitment",
    )
    result = planner._run_matrix(
        successor,
        training,
        specs,
        arguments,
        schema="public-typed-horizon2-successor-fresh-k3-v1",
        pass_verdict="PASS_HORIZON2_SUCCESSOR_FRESH_K3",
        stop_verdict="STOP_HORIZON2_SUCCESSOR_FRESH_K3",
    )
    _require(result.pair_bank_sha256 == _PAIR_BANK_SHA256, "fresh pair bank differs from visible PASS")
    _require(result.scoring_sha256 == _SCORING_SHA256, "fresh scoring projection differs from visible PASS")
    payload = asdict(result)
    payload["visible_artifact_sha256"] = _VISIBLE_SHA256
    payload["visible_verdict"] = visible["verdict"]
    payload["planner_source_sha256"] = _PLANNER_SOURCE_SHA256
    payload["fresh_commitment_sha256"] = sha256(
        planner._canonical_json(commitment)
    ).hexdigest()
    payload["retry_policy"] = (
        "One paired frozen-checkpoint k=3 attempt; no retry, seed, successor, "
        "score, horizon, grid, pair-bank, action-bank, or order sweep."
    )
    payload["claim_boundary"] = (
        "Fresh public paired scalar order-transfer evidence from a target-free "
        "learned cell successor and physical value/pair/motor races; not Simple-SWE, "
        "M0, held-out, C1, or broad coding capability."
    )
    return payload


def main() -> int:
    if _ARTIFACT_PATH.exists():
        prior = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "verdict": prior.get("verdict"),
                    "guard": "existing artifact; fresh boundary not rerun",
                },
                sort_keys=True,
            )
        )
        return 0
    if _SPEND_MARKER_PATH.exists():
        print(
            json.dumps(
                {
                    "artifact": str(_ARTIFACT_PATH),
                    "spend_marker": str(_SPEND_MARKER_PATH),
                    "guard": "fresh boundary already spent; no rerun",
                },
                sort_keys=True,
            )
        )
        return 2
    visible = _authenticate_visible()
    _require(
        sha256(Path(planner.__file__).read_bytes()).hexdigest()
        == _PLANNER_SOURCE_SHA256,
        "current planner source differs from the visible PASS implementation",
    )
    successor, training = planner.train_public_successor()
    pair_sha, scoring_sha = planner._verify_visible_readiness(successor, training)
    _require(pair_sha == _PAIR_BANK_SHA256, "current pair bank differs from visible PASS")
    _require(
        scoring_sha == _SCORING_SHA256,
        "current scoring projection differs from visible PASS",
    )
    _require(
        planner._canonical_json(asdict(training))
        == planner._canonical_json(visible["successor_training"]),
        "reconstructed training summary differs from visible PASS",
    )
    arguments = (-14, -8, -2, 1, 6, 11)
    specs = _fresh_specs()
    commitment = _fresh_commitment(arguments, specs)
    _require(
        _acquire_spend_marker(commitment),
        "fresh k=3 spend marker already exists",
    )
    payload = _run_fresh(
        successor,
        training,
        visible,
        specs,
        arguments,
        commitment,
    )
    _ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(_ARTIFACT_PATH),
                "verdict": payload["verdict"],
                "passed_tasks": payload["passed_tasks"],
                "total_tasks": payload["total_tasks"],
                "fresh_commitment_sha256": payload["fresh_commitment_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["verdict"] == "PASS_HORIZON2_SUCCESSOR_FRESH_K3" else 1


if __name__ == "__main__":
    raise SystemExit(main())
