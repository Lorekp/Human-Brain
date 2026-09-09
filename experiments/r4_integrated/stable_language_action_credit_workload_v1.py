"""Pure saved-JSON transformation for one fixed exposed credit comparison."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_MANIFEST_SHA = "66b0bd96be3b9e97c4eb5f9812c2b5977f294fc5900d55109445c0a303e522d0"
SCHEMA = "stable-language-action-credit-stream-v1"
CHECKPOINTS = (0, 16, 32, 48)
PROBE_GROUPS = (
    ("acquisition", ("A00-R", "A00-E")),
    ("historical_retention", ("R00-R", "R01-E", "R02-R", "R03-E", "R04-R", "R05-E",
                              "R06-R", "R07-E", "R08-R", "R09-E", "R10-R", "R11-E",
                              "R12-R", "R13-E")),
    ("changed_wording", ("A01-R", "A01-E")),
    ("changed_values_context", ("F03-R", "F02-E")),
    ("structural_combination", ("F05-R", "F07-E")),
    ("recovery", ("C00-E", "C01-E")),
)
ROSTER = (
    "A00-R A00-E C00-E F00-R F00-E R00-R R01-E F01-R "
    "A01-R A01-E C01-E F01-E F02-R R02-R R03-E C02-E "
    "F02-E F03-R C03-E F03-E R04-R R05-E F04-R F04-E "
    "F05-R F05-E C04-E F06-R F06-E R06-R R07-E C05-E "
    "F07-R F07-E C06-E F08-R F08-E R08-R R09-E C07-E "
    "F09-R F09-E R10-R R11-E R12-R R13-E A02-R A02-E"
).split()


def build_manifest(base_path=None):
    """Authenticate and transform saved data only; no builder/model imports."""
    path = Path(base_path) if base_path is not None else (
        ROOT / ".pytest-tmp/persistent-language-action-v8-manifest.json")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_MANIFEST_SHA:
        raise ValueError("stable comparison requires the exact saved V8 manifest")
    result = json.loads(raw)
    if [row["episode_id"] for row in result["episodes"]] != ROSTER:
        raise ValueError("stable comparison episode order changed")
    result["schema"] = SCHEMA
    for document in result["documents"]:
        document["neural_tape_admission"] = False
    result["continuation"] = {
        "initial_memory_slots": 1982,
        "baseline_prior_documents": 60,
        "baseline_prior_edit_demonstrations": 49,
        "baseline_prior_source_symbols": 11498,
        "selected_neural_route": "completed-V8-online",
        "payload_name": "route-online-000000262.pt",
        "payload_sha256": "f49a6496f91bca6dc431ae221034dfda9e897d0f2476ae18d2bb4b65f8617592",
        "source_learning": "one transient reoffer of each exposed source; no factual append",
        "baseline_source_retention": "existing append behavior; repeated source offers counted",
        "control_policy": "matched retained numeric state; disabled slow state; actual RNG advances",
    }
    result["counts"] = {
        "episodes_per_route": 48, "attempts_per_route": 64, "executions_per_route": 44,
        "sources_per_route": 24, "source_raw_bytes_per_route": 9860,
        "source_eos_per_route": 24, "teacher_occasions_per_neural_route": 2,
        "teacher_raw_bytes_per_neural_route": 33, "teacher_eos_per_neural_route": 2,
        "action_labels_per_neural_route": 3, "main_boundaries_per_triad": 265,
        "probe_attempts_per_checkpoint_route": 28, "probe_executions_per_checkpoint_route": 17,
        "all_attempts_two_triads": 1056, "all_executions_two_triads": 672,
    }
    result["fresh_split"] = {
        "scope": "all V8 sources/tasks exposed; checkpoint probes are repeated development",
        "changed_value_context_probes_also_change_wording": True,
    }
    result["persistence"] = {
        "main": "every resolved route transaction retains original full safe-boundary protocol",
        "probes": "disposable exact-checkpoint copies; no live state or RNG publication",
        "owned_append_journals": True,
    }
    result["credit_comparison"] = {
        "base_manifest_sha256": BASE_MANIFEST_SHA,
        "checkpoint_episodes": list(CHECKPOINTS),
        "probe_groups": [{"name": name, "episode_ids": list(ids)} for name, ids in PROBE_GROUPS],
        "probe_teacher_offers": 0,
        "probe_source_offers": 0,
        "probe_learning_enabled": False,
        "probe_writes_enabled": False,
        "probe_rng": "exact own scheduled checkpoint RNG; no evaluator reseeding",
        "main_teaching_episode_ids": ["A00-R", "A00-E"],
        "main_teacher_budget_reduction": False,
        "main_difficulty_escalation": False,
        "new_artifact_bytes": 46 * 1024**3,
        "free_reserve_bytes": 8 * 1024**3,
        "pending_write_reserve_bytes": 2 * 1024**3,
        "complete_seconds": 10800,
        "runner_seconds": 10200,
        "audit_seconds": 600,
        "stop_reserve_seconds": 300,
        "peak_mib": 4096,
    }
    result["provenance"] = {
        "kind": "exact saved V8 exposed roster with explicit source-admission transformation",
        "base_manifest_sha256": BASE_MANIFEST_SHA,
        "new_tasks": 0,
        "new_source_bytes": 0,
        "model_or_child_calls_in_builder": 0,
    }
    return result


def check_saved_manifest(path, expected_sha256, *, base_path=None):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("stable comparison manifest bytes changed")
    result = json.loads(raw)
    if result != build_manifest(base_path):
        raise ValueError("stable manifest differs from its complete declared transformation")
    return result


def probe_episodes(manifest):
    """Return independent fixed cases, removing only main-source/teacher offers."""
    by_id = {row["episode_id"]: row for row in manifest["episodes"]}
    result = []
    for group, ids in PROBE_GROUPS:
        for episode_id in ids:
            episode = copy.deepcopy(by_id[episode_id])
            if episode.get("predecessor_episode_id") is not None:
                raise ValueError("probe cannot fabricate an actual project predecessor")
            episode["append_document_ids"] = []
            episode["teacher"] = None
            episode["probe_group"] = group
            result.append(episode)
    return result


def write_manifest(output, *, base_path=None):
    """Explicit pure artifact build; callers bind the returned hash before execution."""
    raw = (json.dumps(build_manifest(base_path), sort_keys=True, indent=2) + "\n").encode()
    with Path(output).open("xb") as handle:
        handle.write(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
