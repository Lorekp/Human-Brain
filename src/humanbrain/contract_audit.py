"""Contract tripwires for scored evaluation runs.

Guards the frozen-evaluation contract: a *scored* run must never load
PyTorch/Autograd machinery and must not mutate model weights while producing
numbers.  Stdlib-only so the audit itself runs where torch is absent.

Public API
----------
- ``assert_no_forbidden_trainable_components()`` -- fail fast when torch or
  autograd modules are present in ``sys.modules``.
- ``weight_snapshot_compare(before, after)`` -- compare two state dicts via
  content hashes; raises :class:`ContractViolation` on drift.
- ``state_dict_digest(state)`` -- stable SHA-256 digest of a state dict.
- ``audit(...)`` -- decorator wrapping a scored entrypoint with both checks.
"""

from __future__ import annotations

import functools
import hashlib
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ContractViolation",
    "DEFAULT_FORBIDDEN_PREFIXES",
    "assert_no_forbidden_trainable_components",
    "state_dict_digest",
    "weight_snapshot_compare",
    "audit",
]

DEFAULT_FORBIDDEN_PREFIXES: Tuple[str, ...] = ("torch", "autograd")


class ContractViolation(AssertionError):
    """A scored run broke the no-training contract."""


def _is_forbidden(module_name: str, prefixes: Sequence[str]) -> bool:
    return any(module_name == p or module_name.startswith(p + ".") for p in prefixes)


def assert_no_forbidden_trainable_components(
    forbidden: Sequence[str] = DEFAULT_FORBIDDEN_PREFIXES,
) -> None:
    """Raise :class:`ContractViolation` if torch/autograd got imported."""
    offenders = sorted(
        name for name in sys.modules if _is_forbidden(name, forbidden)
    )
    if offenders:
        shown = ", ".join(offenders[:12]) + (" ..." if len(offenders) > 12 else "")
        raise ContractViolation(
            f"scored run loaded forbidden trainable machinery: {shown}"
        )
    return None


def _digest_value(h: "hashlib._Hash", value: Any) -> None:
    """Feed ``value`` into hash ``h`` deterministically (structural)."""
    if value is None:
        h.update(b"n:")
    elif isinstance(value, bool):
        h.update(b"z:%d" % value)
    elif isinstance(value, int):
        h.update(b"i:%d" % value)
    elif isinstance(value, float):
        h.update(b"f:" + repr(value).encode("ascii"))
    elif isinstance(value, str):
        h.update(b"s:" + value.encode("utf-8"))
    elif isinstance(value, bytes):
        h.update(b"b:" + value)
    elif isinstance(value, Mapping):
        h.update(b"M[%d]:" % len(value))
        for key in sorted(value, key=repr):
            _digest_value(h, key)
            _digest_value(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(b"L[%d]:" % len(value))
        for item in value:
            _digest_value(h, item)
    else:
        # Tensors / ndarrays / array-like: prefer raw bytes plus shape.
        shape = getattr(value, "shape", None)
        if shape is not None:
            h.update(b"h:" + repr(tuple(shape)).encode("ascii") + b"|")
        data = None
        for grab in (
            lambda v: v.detach().cpu().contiguous().numpy().tobytes(),  # torch.Tensor
            lambda v: v.tobytes(),  # numpy array / buffer
            lambda v: memoryview(v).tobytes(),
        ):
            try:
                data = grab(value)
                break
            except Exception:
                continue
        h.update(b"a:" + (data if data is not None else repr(value).encode("utf-8")))


def state_dict_digest(state: Mapping[str, Any]) -> str:
    """Stable SHA-256 hex digest of a state dict (key order independent)."""
    h = hashlib.sha256()
    h.update(b"contract-audit-v1|%d|" % len(state))
    for key in sorted(state, key=str):
        h.update(b"k:" + str(key).encode("utf-8") + b"|")
        _digest_value(h, state[key])
        h.update(b"|")
    return h.hexdigest()


def _per_key_digests(state: Mapping[str, Any]) -> Dict[str, str]:
    return {str(k): state_dict_digest({k: state[k]}) for k in state}


def weight_snapshot_compare(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare two state-dict snapshots; raise :class:`ContractViolation` on drift.

    Returns a report dict with ``identical``, ``added``, ``removed``, and
    ``changed`` key lists plus both whole-snapshot digests.
    """
    before_keys, after_keys = set(map(str, before)), set(map(str, after))
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    before_digests = _per_key_digests(before)
    after_digests = _per_key_digests(after)
    changed = sorted(k for k in before_keys & after_keys
                     if before_digests[k] != after_digests[k])
    report = {
        "identical": not (added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": changed,
        "before_digest": state_dict_digest(before),
        "after_digest": state_dict_digest(after),
    }
    if not report["identical"]:
        raise ContractViolation(f"weight drift detected: {report}")
    return report


def audit(fn: Optional[Callable] = None,
          *,
          weight_source: Optional[Callable[[], Mapping[str, Any]]] = None) -> Callable:
    """Decorator enforcing the no-training contract around a scored run.

    Usage::

        @audit
        def score(model): ...

        @audit(weight_source=lambda: model.state_dict())
        def score(): ...

    Checks that no torch/autograd module is loaded immediately before and
    after the call; when ``weight_source`` is given, also asserts per-key
    digests of the returned state dict are unchanged across the call.
    """
    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            assert_no_forbidden_trainable_components()
            snapshot_before: Optional[List[Tuple[str, str]]] = (
                sorted(_per_key_digests(weight_source()).items())
                if weight_source is not None else None
            )
            result = func(*args, **kwargs)
            assert_no_forbidden_trainable_components()
            if weight_source is not None:
                snapshot_after = sorted(_per_key_digests(weight_source()).items())
                if snapshot_after != snapshot_before:
                    drift = {
                        k for k, v in dict(snapshot_after).items()
                        if dict(snapshot_before or {}).get(k) != v
                    }
                    raise ContractViolation(
                        f"weights mutated during scored run: keys={sorted(drift)}"
                    )
            return result
        return wrapper

    return decorate(fn) if fn is not None else decorate
