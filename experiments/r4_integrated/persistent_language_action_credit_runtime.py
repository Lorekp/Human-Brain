"""Source-bound old/new credit selection before importing the acting system.

No model is constructed here. Old credit uses the two exact V8 source files;
every other banked neuronal Python dependency must still match V8. This is an
experiment worker boundary, not a compatibility mode in the live learner.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "8e8df91d7e7c6889dff15f97d9fec379ebff71bb"
INVENTORY_SHA = "ae23542c22d89a1e1bbbd4babb3739928a550a60da89b6bc50fbef06f92148ac"
INVENTORY = ROOT / ".pytest-tmp/persistent-language-action-v8-corrected-run/sources.json"
MODULES = {
    "humanbrain.recurrent_content_controller":
        "e2fb738b86d820db85f3e8b6190a93a9c85fbc704bbfe897d8ac7c5a335f56b5",
    "humanbrain.recurrent_content_learner":
        "e89eb5ccfa44eda3b9a8306bc8603b2aeab11d78630e47de4cc2d8f1e22c9ece",
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _relative(name):
    return "src/" + name.replace(".", "/") + ".py"


def historical_sources():
    raw = INVENTORY.read_bytes()
    if _sha(raw) != INVENTORY_SHA:
        raise ValueError("banked V8 source inventory changed")
    inventory = json.loads(raw)
    for name, identity in MODULES.items():
        if inventory.get(_relative(name)) != identity:
            raise ValueError("old credit module differs from the banked V8 inventory")
    return inventory


def prepare(destination):
    """Copy exact Git blobs once; no imports of models, fit, probes or decoding."""
    destination = Path(destination).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("comparison source copies must remain inside the checkout")
    historical_sources()
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, identity in MODULES.items():
        raw = subprocess.run(
            ["git", "show", REVISION + ":" + _relative(name)], cwd=ROOT,
            check=True, capture_output=True).stdout
        if _sha(raw) != identity:
            raise ValueError("Git source blob differs from authenticated historical source")
        path = destination / (name.rsplit(".", 1)[1] + ".py")
        with path.open("xb") as stream:
            stream.write(raw)
        if _sha(path.read_bytes()) != identity:
            raise ValueError("old-rule source copy changed on publication")
        result[path.relative_to(ROOT).as_posix()] = identity
    return {"revision": REVISION, "source_sha256": result,
            "historical_inventory_sha256": INVENTORY_SHA, "model_calls": 0}


def select(rule, snapshot):
    """Call once in a fresh worker before any humanbrain submodule is imported."""
    if rule not in {"old", "causal"}:
        raise ValueError("comparison rule must be old or causal")
    if any(name.startswith("humanbrain.") for name in sys.modules):
        raise ValueError("credit runtime must be selected before neuronal imports")
    inventory = historical_sources()
    shared = {}
    overridden = {_relative(name) for name in MODULES}
    for relative, identity in inventory.items():
        if relative.startswith("src/humanbrain/") and relative not in overridden:
            if _sha((ROOT / relative).read_bytes()) != identity:
                raise ValueError("shared neuronal dependency changed since V8: " + relative)
            shared[relative] = identity
    package = importlib.import_module("humanbrain")
    if Path(package.__file__).resolve() != ROOT / "src/humanbrain/__init__.py":
        raise ValueError("worker imported another humanbrain package")
    origins, bindings = {}, dict(shared)
    for name, old_identity in MODULES.items():
        path = ((Path(snapshot).resolve() / (name.rsplit(".", 1)[1] + ".py"))
                if rule == "old" else ROOT / _relative(name))
        if not path.is_relative_to(ROOT) or path.is_symlink():
            raise ValueError("credit source escaped its declared local snapshot")
        raw = path.read_bytes()
        if rule == "old" and _sha(raw) != old_identity:
            raise ValueError("old credit source identity changed")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Execute the authenticated bytes, without a mutable reopen or pyc cache.
        exec(compile(raw, str(path), "exec"), module.__dict__)
        setattr(package, name.rsplit(".", 1)[1], module)
        origins[name] = str(path)
        bindings[path.relative_to(ROOT).as_posix()] = _sha(raw)
    receipt = {"rule": rule, "old_source_revision": REVISION,
               "historical_inventory_sha256": INVENTORY_SHA,
               "module_origins": origins, "source_sha256": bindings}
    receipt["imported_neuronal_origins"] = check_origins(receipt)
    return receipt


def check_origins(receipt):
    """Attest every imported neuronal module, including transitive dependencies."""
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != "humanbrain" and not name.startswith("humanbrain."):
            continue
        path = Path(module.__file__).resolve()
        expected = receipt["module_origins"].get(name)
        if expected is not None and path != Path(expected):
            raise ValueError("selected credit module was replaced")
        if not path.is_relative_to(ROOT):
            raise ValueError("neuronal import escaped the source-bound checkout")
        relative = path.relative_to(ROOT).as_posix()
        if receipt["source_sha256"].get(relative) != _sha(path.read_bytes()):
            raise ValueError("unbound or changed neuronal import: " + name)
        origins[name] = str(path)
    return origins
