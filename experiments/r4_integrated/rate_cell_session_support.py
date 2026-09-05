"""Data-only retained Python/NumPy storage accounting for reading sessions.

This counts reachable owned buffers and Python objects, including instances
defined by ``python -m`` as ``__main__``. It is not an allocator/RSS measurement.
Code, weak references, and opaque native handles are outside this traversal;
unsupported objects raise instead of silently contributing only their header.
Encoder/native allocations must be reported separately with process memory.
"""

from __future__ import annotations

import gc
import sys
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from types import (
    BuiltinFunctionType,
    CodeType,
    CoroutineType,
    FunctionType,
    GeneratorType,
    MappingProxyType,
    MemberDescriptorType,
    MethodType,
    ModuleType,
)

import numpy as np

_LEAVES = (type(None), bool, int, float, complex, str, bytes, bytearray)
_CODE = (
    type,
    ModuleType,
    FunctionType,
    MethodType,
    BuiltinFunctionType,
    CodeType,
    GeneratorType,
    CoroutineType,
)


def retained_size(*roots: object) -> int:
    """Count each reachable data object/allocation once across all roots.

    NumPy's own-buffer size is included by ``getsizeof``; views instead lead to
    their owner, so overlapping views and shared bytes count once. A memoryview
    likewise leads to its owner. Object-dtype arrays also own references to their
    elements. Python instance dictionaries and every inherited slot are visited,
    independent of module name. The result includes Python container overhead.

    Pass only data roots, never an encoder/session/module or a callable closure.
    Generic instances expose their Python attributes; any opaque extension
    allocation behind such attributes remains outside this API and requires RSS
    accounting. Unsupported leaf handles raise ``TypeError``. No properties,
    generators, code globals, or weak-reference targets are traversed.
    """
    total, seen, pending = 0, set(), list(roots)
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, _CODE):
            raise TypeError(f"code object is outside retained data scope: {type(value).__name__}")
        total += sys.getsizeof(value)
        if type(value) in _LEAVES:
            continue
        if isinstance(value, np.ndarray):
            if type(value) is not np.ndarray:
                raise TypeError("NumPy subclasses require explicit allocation accounting")
            if value.base is not None:
                pending.append(value.base)
            if value.dtype.hasobject:
                pending.extend(value.flat)
            continue
        if isinstance(value, np.generic):
            if value.dtype.hasobject:
                raise TypeError(
                    "object-bearing NumPy scalars require an explicit data representation"
                )
            continue
        if isinstance(value, memoryview):
            pending.append(value.obj)
            continue
        if isinstance(value, MappingProxyType):
            owners = gc.get_referents(value)
            if len(owners) != 1 or not isinstance(owners[0], Mapping):
                raise TypeError("mapping proxy owner cannot be accounted explicitly")
            pending.append(owners[0])
            continue
        container = True
        if isinstance(value, Mapping):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset, deque)):
            pending.extend(value)
        elif isinstance(value, range):
            pending.extend((value.start, value.stop, value.step))
        else:
            container = False
        try:
            attributes = object.__getattribute__(value, "__dict__")
        except AttributeError:
            attributes = None
        if attributes is not None:
            pending.append(attributes)
        slots = [
            descriptor
            for cls in type(value).__mro__
            for name, descriptor in vars(cls).items()
            if isinstance(descriptor, MemberDescriptorType)
            and name not in ("__dict__", "__weakref__")
        ]
        for descriptor in slots:
            # An uninitialized declared slot retains no object.
            with suppress(AttributeError):
                pending.append(descriptor.__get__(value, type(value)))
        if attributes is None and not slots and not container:
            raise TypeError(
                "opaque object needs separate allocation accounting: "
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
    return total
