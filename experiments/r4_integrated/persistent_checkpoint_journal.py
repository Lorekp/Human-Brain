"""Owned append journals; no native calls or files are opened at import.

Local NTFS: CREATE_NEW, GENERIC_READ|FILE_APPEND_DATA, FILE_SHARE_READ,
FILE_FLAG_WRITE_THROUGH. No FILE_WRITE_DATA/GENERIC_WRITE/delete rights.
Write-through is the native durability mechanism; we do not claim fsync or
FlushFileBuffers (the latter documents GENERIC_WRITE as required).
Microsoft documents local append-only access, NTFS write-through data/metadata,
and GENERIC_WRITE for writable mappings:
https://learn.microsoft.com/en-us/windows/win32/fileio/file-access-rights-constants
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew
https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-createfilemappingw
https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers

No handle is public or inherited. This is source-bound parent ownership, not a
sandbox against hostile code inside that process. Unsupported filesystems use
ordinary xb/fsync and NEVER authorize cached prefix authentication.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from experiments.r4_integrated import persistent_checkpoint_payload_lease as lease
from experiments.r4_integrated.persistent_language_action_v1 import encode

ROW_LIMIT = 64 * 1024 * 1024
BLOCK = 1024 * 1024


def _add(work, name, amount=1):
    if work is not None:
        key = "checkpoint_journal_" + name
        work[key] = work.get(key, 0) + amount


def _lease_call(function, *args, work):
    counters = {}
    try:
        return function(*args, counters)
    finally:
        for key, value in counters.items():
            _add(work, key.removeprefix("checkpoint_payload_"), value)


@lru_cache(maxsize=1)
def _windows():
    api = lease._windows()
    handle, pointer, dword = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
    api.WriteFile.argtypes = [handle, pointer, dword, ctypes.POINTER(dword), pointer]
    api.WriteFile.restype = ctypes.c_int
    api.SetFilePointerEx.argtypes = [handle, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), dword]
    api.SetFilePointerEx.restype = ctypes.c_int
    return api


class OwnedJournal:
    """One new outcome journal, one writer, two authenticated frontier values.

    append returns the existing row SHA. freeze excludes append/close across a
    checkpoint's capture and publication. Failed physical IO poisons the owner;
    closing preserves every byte and does not repair a failed transaction.
    """

    def __init__(self, path, work=None):
        self._path = Path(path).absolute()
        self.work = work if work is not None else {}
        self._lock = threading.Lock()
        self._frozen_thread = None
        self._closed = False
        self._poisoned = False
        self._handle = None
        self._owned_handle = None
        self._file = None
        self._api = None
        self._identity = None
        self._hash = hashlib.sha256()
        self._current = (0, 0, self._hash.hexdigest())
        self._committed = self._current
        self.rows = self.bytes_written = self.max_row_bytes = 0
        self.append_seconds = 0.0
        self._native = _lease_call(lease.supported, self.path.parent, work=self.work)
        try:
            if self._native:
                self._api = _windows()
                _add(self.work, "writer_open_attempts")
                handle = self._api.CreateFileW(
                    str(self.path), 0x80000004, 1, None, 1, 0x80200000, None)
                if handle is None or handle == lease._INVALID_HANDLE:
                    raise lease._error(self.path)
                self._handle = self._owned_handle = handle
                _add(self.work, "writer_opens")
                self._identity, size = self._metadata(handle)
                if size:
                    raise ValueError("new owned journal is not empty")
            else:
                self._file = self.path.open("xb")
                stat = os.fstat(self._file.fileno())
                self._identity = stat.st_dev, stat.st_ino
                _add(self.work, "fallback_opens")
            self._check()
        except BaseException:
            self.close()
            raise

    @property
    def path(self):
        return self._path

    @property
    def closed(self):
        return self._closed

    @property
    def cached_prefixes(self):
        return self._native

    def _metadata(self, handle):
        return _lease_call(lease._metadata, self._api, handle, self.path, work=self.work)

    def _check(self):
        try:
            self._check_owned()
        except BaseException:
            self._poisoned = True
            raise

    def _check_owned(self):
        if self._closed or self._poisoned:
            raise ValueError("owned journal is closed or poisoned")
        if self._native:
            if self._handle != self._owned_handle:
                raise ValueError("owned journal handle was rebound")
            identity, size = self._metadata(self._handle)
            # A read-only identity handle must share the existing append access.
            # The original writer still denies every other write/delete open.
            _add(self.work, "identity_open_attempts")
            other = self._api.CreateFileW(str(self.path), 0x80000000, 3, None, 3, 0x00200000, None)
            if other is None or other == lease._INVALID_HANDLE:
                raise lease._error(self.path)
            _add(self.work, "identity_opens")
            try:
                path_identity, path_size = self._metadata(other)
            finally:
                if not self._api.CloseHandle(other):
                    raise lease._error(self.path)
                _add(self.work, "identity_closes")
            if path_identity != identity or path_size != size:
                raise ValueError("owned journal path changed")
        else:
            stat, named = os.fstat(self._file.fileno()), self.path.stat()
            identity, size = (stat.st_dev, stat.st_ino), stat.st_size
            if (named.st_dev, named.st_ino, named.st_size) != (*identity, size):
                raise ValueError("journal path changed")
        _add(self.work, "identity_checks")
        if identity != self._identity or size != self._current[0]:
            raise ValueError("owned journal identity or EOF changed")

    @contextmanager
    def freeze(self):
        if not self._lock.acquire(blocking=False):
            raise ValueError("journal writer or checkpoint already active")
        try:
            self._check()
            self._frozen_thread = threading.get_ident()
            _add(self.work, "freeze_scopes")
            yield self
            self._check()
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._frozen_thread = None
            self._lock.release()

    def require_frozen(self):
        if self._frozen_thread != threading.get_ident():
            raise ValueError("owned journal needs the checkpoint freeze scope")
        self._check()

    def poison(self):
        """An interrupted publication cannot reuse this owner."""
        self._poisoned = True

    def frontier(self, *, prefix=None):
        self.require_frozen()
        if not self.cached_prefixes:
            raise ValueError("unsupported journal backend requires full prefix verification")
        selected = self._current if prefix is None else next(
            (row for row in (self._current, self._committed) if row[0] == prefix), None)
        if selected is None:
            raise ValueError("journal prefix is neither current nor committed")
        return {"bytes": selected[0], "rows": selected[1], "sha256": selected[2], "complete": True}

    def prepare_commit(self, row):
        self.require_frozen()
        expected = dict(zip(("bytes", "rows", "sha256"), self._current))
        if any(row[key] != expected[key] for key in ("bytes", "rows", "sha256")):
            raise ValueError("checkpoint journal frontier differs from actual owned bytes")
        return self._current

    def commit_validated(self, prepared):
        # Called only after all prepublication checks and successful HEAD replace.
        self._committed = prepared

    def _read_tail(self, offset, raw):
        result, row_hash, seen = self._hash.copy(), hashlib.sha256(), 0
        if self._native:
            if not self._api.SetFilePointerEx(self._handle, offset, None, 0):
                raise lease._error(self.path)
            buffer = ctypes.create_string_buffer(min(BLOCK, len(raw)))
            _add(self.work, "read_buffer_bytes", len(buffer))
            while seen < len(raw):
                count = ctypes.c_uint32()
                wanted = min(len(buffer), len(raw) - seen)
                _add(self.work, "tail_read_calls")
                if not self._api.ReadFile(self._handle, buffer, wanted, ctypes.byref(count), None):
                    raise lease._error(self.path)
                _add(self.work, "tail_bytes_read", count.value)
                if count.value != wanted:
                    raise ValueError("owned journal tail read shortened")
                chunk = memoryview(buffer).cast("B")[:count.value]
                result.update(chunk)
                row_hash.update(chunk)
                seen += count.value
        else:
            with self.path.open("rb") as source:
                source.seek(offset)
                while seen < len(raw):
                    chunk = source.read(min(BLOCK, len(raw) - seen))
                    _add(self.work, "tail_read_calls")
                    if not chunk:
                        raise ValueError("journal tail read shortened")
                    result.update(chunk)
                    row_hash.update(chunk)
                    seen += len(chunk)
                    _add(self.work, "tail_bytes_read", len(chunk))
        if row_hash.hexdigest() != hashlib.sha256(raw).hexdigest():
            raise ValueError("journal written tail differs from intended bytes")
        return result, row_hash.hexdigest()

    def append(self, row):
        tick = time.perf_counter()
        if not self._lock.acquire(blocking=False):
            raise ValueError("journal append forbidden during checkpoint freeze")
        started_write = False
        try:
            self._check()
            raw = encode(row)
            if not raw or len(raw) > ROW_LIMIT or raw.count(b"\n") != 1 or raw[-1:] != b"\n":
                raise ValueError("journal requires one bounded complete encoded record")
            started_write = True
            if self._native:
                buffer, count = ctypes.create_string_buffer(raw), ctypes.c_uint32()
                _add(self.work, "write_buffer_bytes", len(buffer))
                _add(self.work, "write_calls")
                if not self._api.WriteFile(self._handle, buffer, len(raw), ctypes.byref(count), None):
                    raise lease._error(self.path)
                _add(self.work, "bytes_written", count.value)
                if count.value != len(raw):
                    raise ValueError("owned journal write shortened")
                _add(self.work, "write_through_completions")
            else:
                count = self._file.write(raw)
                _add(self.work, "write_calls")
                _add(self.work, "bytes_written", count or 0)
                if count != len(raw):
                    raise ValueError("journal write shortened")
                self._file.flush()
                os.fsync(self._file.fileno())
                _add(self.work, "fsync_calls")
            result, digest = self._read_tail(self._current[0], raw)
            proposed = self._current[0] + len(raw), self._current[1] + 1, result.hexdigest()
            old = self._current
            self._current = proposed
            try:
                self._check()
            except BaseException:
                self._current = old
                raise
            self._hash = result
            self.rows, self.bytes_written = proposed[1], proposed[0]
            self.max_row_bytes = max(self.max_row_bytes, len(raw))
            _add(self.work, "authenticated_rows")
            return digest
        except BaseException:
            if started_write:
                self._poisoned = True
            raise
        finally:
            self.append_seconds += time.perf_counter() - tick
            self._lock.release()

    def metrics(self):
        return {"rows": self.rows, "bytes_written": self.bytes_written,
                "max_row_bytes": self.max_row_bytes, "append_seconds": self.append_seconds,
                "scope": "encode, append, native write-through or fallback fsync, actual tail authentication"}

    def close(self):
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise ValueError("cannot close journal during append/checkpoint")
        try:
            self._closed = True
            if self._owned_handle is not None:
                handle = self._owned_handle
                self._handle = self._owned_handle = None
                if not self._api.CloseHandle(handle):
                    raise lease._error(self.path)
                _add(self.work, "writer_closes")
            elif self._file is not None:
                self._file.close()
                _add(self.work, "fallback_closes")
        finally:
            self._lock.release()
