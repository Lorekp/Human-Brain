"""Select exact predecessor/current learning sources before neuronal imports.

The predecessor is the completed repeated-teaching learner. Only its controller
and learner modules are source-isolated; all other neuronal dependencies remain
the authenticated shared implementation. Importing this helper performs no source
preparation, neuronal import, checkpoint decoding or model construction.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.r4_integrated.persistent_language_action_credit_runtime import (
    _relative,
    _sha,
    check_origins,
)

ROOT = Path(__file__).resolve().parents[2]
REVISION = "7271b9caf8c043277503abd3a74f7fd593f00563"
PARENT_REVISION = "f577413dd9bac32302c3c84d5e09b950f8c8473c"
BANK = ROOT / "benchmarks/repeated_teaching_acquisition_v1.run.json"
BANK_SHA = "3783e751a40c0150da8adc091a976d91ab70c879dc1b2a5c6ffc020c27e8e262"
PARENT = ROOT / ".pytest-tmp/repeated-teaching-v1-run"
CONFIG = PARENT / "checkpoints/config.json"
CONFIG_SHA = "cd15cf70fd3aac81bc10068c00e7343cae62d2af3280d0274a41f4b0c579af1e"
BOUNDARY_SHA = "3408eecadf057ce322f5063a879584e82ea993c433e74d1de290e5114e1ae677"
CONTROLLER_SHA = "9b6568d315c2069db250f342c67dfe8d9946ec963f668c263cdcd8bdbbc0f870"
LEARNER_SHA = "3a7d85612a8313bc9da427525520e294af05d191e3d5b84c1e0b9330f94ded5a"
MODULES = {
    "humanbrain.recurrent_content_controller": CONTROLLER_SHA,
    "humanbrain.recurrent_content_learner": LEARNER_SHA,
}
READ_LIMIT = 16 * 1024**2


def _read(path, expected=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(READ_LIMIT + 1)
    if len(raw) > READ_LIMIT:
        raise ValueError("learning source or binding exceeds its explicit read bound")
    if expected is not None and _sha(raw) != expected:
        raise ValueError("learning source or parent binding changed: " + str(path))
    return raw


def predecessor_sources():
    """Authenticate the completed parent bank and its exact source configuration."""
    bank = json.loads(_read(BANK, BANK_SHA))
    if (
        bank["status"] != "completed"
        or bank["restore"]["status"] != "passed"
        or bank["final"]["boundary_sha256"] != BOUNDARY_SHA
        or bank["restore"]["boundary_sha256"] != BOUNDARY_SHA
        or bank["final"]["config_sha256"] != CONFIG_SHA
    ):
        raise ValueError("learning predecessor is not the completed restored parent")
    config = json.loads(_read(CONFIG, CONFIG_SHA))
    if (
        config["schema"] != "repeated-teaching-checkpoint-config-v1"
        or config["source_revision"] != PARENT_REVISION
        or Path(config["source_root"]).resolve() != ROOT
        or Path(config["evidence_root"]).resolve() != PARENT
        or config["manifest_sha256"] != bank["manifest_sha256"]
    ):
        raise ValueError("learning predecessor source configuration differs")
    inventory = config["source_sha256"]
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("learning predecessor source inventory is missing")
    for name, identity in MODULES.items():
        if inventory.get(_relative(name)) != identity:
            raise ValueError("predecessor module differs from its banked source identity")
    return inventory


def prepare(destination):
    """Write the two pinned Git blobs once, without executing either module."""
    destination = Path(destination).resolve()
    if destination == ROOT or not destination.is_relative_to(ROOT):
        raise ValueError("predecessor copies need a distinct checkout-local directory")
    if destination.exists():
        raise FileExistsError(destination)
    predecessor_sources()
    blobs = {}
    for name, identity in MODULES.items():
        raw = subprocess.run(
            ["git", "show", REVISION + ":" + _relative(name)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        if len(raw) > READ_LIMIT or _sha(raw) != identity:
            raise ValueError("predecessor Git blob differs from the completed parent source")
        blobs[name] = raw
    # Validate both blobs before creating output. Interrupted writes remain
    # visible and cannot be adopted by another prepare invocation.
    destination.mkdir(parents=True, exist_ok=False)
    bindings = {}
    for name, raw in blobs.items():
        path = destination / (name.rsplit(".", 1)[1] + ".py")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _read(path, MODULES[name])
        bindings[path.relative_to(ROOT).as_posix()] = MODULES[name]
    return {
        "schema": "integrated-local-learning-source-preparation-v1",
        "revision": REVISION,
        "parent_source_revision": PARENT_REVISION,
        "parent_bank_sha256": BANK_SHA,
        "parent_config_sha256": CONFIG_SHA,
        "source_sha256": bindings,
        "source_bytes_written": sum(map(len, blobs.values())),
        "git_blob_reads": 2,
        "fsync_calls": 2,
        "model_calls": 0,
        "payload_decodes": 0,
    }


def select(rule, snapshot=None):
    """Select predecessor/current once in a fresh worker, before neuronal imports.

    The current files are bound by their actual authenticated execution bytes and
    the returned receipt. The caller freezes that receipt with its worker source
    inventory and calls check_origins again after subsequent neuronal imports.
    """
    if rule not in {"predecessor", "current"}:
        raise ValueError("learning rule must be predecessor or current")
    if any(name.startswith("humanbrain.") for name in sys.modules):
        raise ValueError("learning runtime must be selected before neuronal imports")
    inventory = predecessor_sources()
    overridden = {_relative(name) for name in MODULES}
    bindings = {}
    for relative, identity in inventory.items():
        if relative.startswith("src/humanbrain/") and relative not in overridden:
            _read(ROOT / relative, identity)
            bindings[relative] = identity
    if rule == "predecessor":
        if snapshot is None:
            raise ValueError("predecessor selection requires its prepared source directory")
        snapshot = Path(snapshot).resolve()
        if snapshot == ROOT or not snapshot.is_relative_to(ROOT):
            raise ValueError("predecessor source directory escaped the checkout")
        if {path.name for path in snapshot.iterdir()} != {
            name.rsplit(".", 1)[1] + ".py" for name in MODULES
        }:
            raise ValueError("predecessor source directory needs exactly its two modules")
    selected = {}
    for name, identity in MODULES.items():
        path = (
            snapshot / (name.rsplit(".", 1)[1] + ".py")
            if rule == "predecessor"
            else ROOT / _relative(name)
        )
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("selected learning source escaped its declared local binding")
        selected[name] = path.resolve(), _read(path, identity if rule == "predecessor" else None)
    package = importlib.import_module("humanbrain")
    if Path(package.__file__).resolve() != ROOT / "src/humanbrain/__init__.py":
        raise ValueError("worker imported another humanbrain package")
    origins = {}
    for name, (path, raw) in selected.items():
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Execute the same bytes whose hash enters the receipt, avoiding a
        # second mutable source read or a cached bytecode substitution.
        exec(compile(raw, str(path), "exec"), module.__dict__)
        setattr(package, name.rsplit(".", 1)[1], module)
        origins[name] = str(path)
        bindings[path.relative_to(ROOT).as_posix()] = _sha(raw)
    receipt = {
        "schema": "integrated-local-learning-source-selection-v1",
        "rule": rule,
        "predecessor_source_revision": REVISION,
        "parent_source_revision": PARENT_REVISION,
        "parent_bank_sha256": BANK_SHA,
        "parent_config_sha256": CONFIG_SHA,
        "module_origins": origins,
        "source_sha256": bindings,
        "model_calls": 0,
        "payload_decodes": 0,
    }
    receipt["imported_neuronal_origins"] = check_origins(receipt)
    return receipt
