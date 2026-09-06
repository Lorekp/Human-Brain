"""Teacher-free bounded JSON receipts and synchronous worker transport."""

import hashlib
import json
import math
import sys


def host_tree_bytes(value):
    """Conservative no-alias-dedup CPython receipt payload, never tensor/model data."""
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(host_tree_bytes(k) + host_tree_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return size + sum(host_tree_bytes(item) for item in value)
    if type(value) not in (str, int, float, bool, type(None)):
        raise TypeError("journal rows must contain JSON primitives only")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("journal scalar must be finite")
    return size


class JsonlJournal:
    """Bounded one-row append; owns only scalar/hash summaries after return."""

    def __init__(self, path, spec):
        self.path, self.limits = path, spec.config["receipt_limits"]
        self.stream = path.open("xb")
        self.hasher = hashlib.sha256()
        self.rows = self.bytes = self.max_line = self.max_tree = 0

    def append(self, row):
        tree = host_tree_bytes(row)
        if tree > self.limits["max_row_tree_bytes"]:
            raise MemoryError("receipt exceeds host-tree limit")
        encoded = (
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if len(encoded) > self.limits["max_line_bytes"]:
            raise MemoryError("receipt exceeds line limit")
        self.stream.write(encoded)
        self.stream.flush()
        self.hasher.update(encoded)
        self.rows += 1
        self.bytes += len(encoded)
        self.max_line, self.max_tree = max(self.max_line, len(encoded)), max(self.max_tree, tree)
        return hashlib.sha256(encoded).hexdigest()

    def close(self):
        self.stream.close()
        return dict(
            path=self.path.name,
            rows=self.rows,
            bytes=self.bytes,
            sha256=self.hasher.hexdigest(),
            max_line_bytes=self.max_line,
            max_row_tree_bytes=self.max_tree,
        )


def read_row(stream, spec):
    line = stream.readline(spec.config["receipt_limits"]["max_line_bytes"] + 1)
    if not line:
        return None
    if len(line) > spec.config["receipt_limits"]["max_line_bytes"] or not line.endswith(b"\n"):
        raise ValueError("oversized or truncated receipt line")
    row = json.loads(line)
    if host_tree_bytes(row) > spec.config["receipt_limits"]["max_row_tree_bytes"]:
        raise MemoryError("decoded receipt exceeds host-tree limit")
    return row


def write_message(stream, row, spec):
    """Write one guarded binary protocol line; keep no queue or receipt history."""
    if host_tree_bytes(row) > spec.config["receipt_limits"]["max_row_tree_bytes"]:
        raise MemoryError("message exceeds host-tree limit")
    encoded = (json.dumps(row, ensure_ascii=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    if len(encoded) > spec.config["receipt_limits"]["max_line_bytes"]:
        raise MemoryError("message exceeds line limit")
    stream.write(encoded)
    stream.flush()
    return hashlib.sha256(encoded).hexdigest()


def process_memory():
    """Read this Windows process's counters without launching another process."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    ("peak_ws", ctypes.c_size_t), ("ws", ctypes.c_size_t),
                    ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                    ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
                    ("private", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    value = Counters()
    value.cb = ctypes.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"working_set_mib": value.ws / 2**20, "peak_working_set_mib": value.peak_ws / 2**20,
            "private_memory_mib": value.private / 2**20}


def process_identity(pid=None):
    """Bind a Windows PID to creation time expressed as .NET UTC ticks."""
    import ctypes
    import os
    from ctypes import wintypes

    selected = os.getpid() if pid is None else pid
    if type(selected) is not int or selected <= 0:
        raise ValueError("process identity requires a positive PID")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, selected)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(system), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        file_ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)
    return {"pid": selected, "creation_ticks": file_ticks + 504911232000000000}
