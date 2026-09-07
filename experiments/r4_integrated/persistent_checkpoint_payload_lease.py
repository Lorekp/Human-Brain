"""Read-only Windows ownership of newly published immutable payload bytes.

No native call runs at import. Unsupported platforms retain the checkpoint's
full-hash path. A successful lease never permits a writable/delete-sharing
handle, and loss of ownership is an error rather than a cache miss.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path

_BLOCK = 1024 * 1024
_INVALID_HANDLE = ctypes.c_void_p(-1).value


class _FileId(ctypes.Structure):
    _fields_ = [("volume", ctypes.c_uint64), ("identifier", ctypes.c_ubyte * 16)]


class _StandardInfo(ctypes.Structure):
    _fields_ = [
        ("allocation", ctypes.c_int64), ("size", ctypes.c_int64),
        ("links", ctypes.c_uint32), ("delete_pending", ctypes.c_ubyte),
        ("directory", ctypes.c_ubyte),
    ]


class _AttributeInfo(ctypes.Structure):
    _fields_ = [("attributes", ctypes.c_uint32), ("reparse_tag", ctypes.c_uint32)]


def _add(work, name, value=1):
    if work is not None:
        work[name] = work.get(name, 0) + value


@lru_cache(maxsize=1)
def _windows():
    if os.name != "nt":
        raise NotImplementedError("payload leases require Windows")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, pointer, dword = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
    signatures = {
        "CreateFileW": ([ctypes.c_wchar_p, dword, dword, pointer, dword, dword, handle], handle),
        "CloseHandle": ([handle], ctypes.c_int),
        "GetFileType": ([handle], dword),
        "GetFileInformationByHandleEx": ([handle, ctypes.c_int, pointer, dword], ctypes.c_int),
        "ReadFile": ([handle, pointer, dword, ctypes.POINTER(dword), pointer], ctypes.c_int),
        "GetVolumePathNameW": ([ctypes.c_wchar_p, ctypes.c_wchar_p, dword], ctypes.c_int),
        "GetDriveTypeW": ([ctypes.c_wchar_p], dword),
        "GetVolumeInformationW": (
            [ctypes.c_wchar_p, ctypes.c_wchar_p, dword, pointer, pointer, pointer,
             ctypes.c_wchar_p, dword], ctypes.c_int,
        ),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api


def _error(path):
    error = ctypes.WinError(ctypes.get_last_error())
    error.filename = str(path)
    return error


def supported(path, work=None):
    """Select only fixed local NTFS; query errors are not silent fallbacks."""
    _add(work, "checkpoint_payload_capability_checks")
    if os.name != "nt":
        return False
    try:
        api = _windows()
    except AttributeError:  # Required API absent: keep the original verifier.
        return False
    name = os.path.abspath(os.fspath(path))
    if name.startswith("\\\\"):
        return False
    root = ctypes.create_unicode_buffer(32768)
    _add(work, "checkpoint_payload_volume_queries")
    if not api.GetVolumePathNameW(name, root, len(root)):
        raise _error(name)
    _add(work, "checkpoint_payload_volume_queries")
    if api.GetDriveTypeW(root.value) != 3:  # DRIVE_FIXED
        return False
    filesystem = ctypes.create_unicode_buffer(32)
    _add(work, "checkpoint_payload_volume_queries")
    if not api.GetVolumeInformationW(root.value, None, 0, None, None, None,
                                     filesystem, len(filesystem)):
        raise _error(name)
    return filesystem.value == "NTFS"


def _open(api, path):
    # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, OPEN_REPARSE_POINT.
    # Null SECURITY_ATTRIBUTES makes the handle non-inheritable.
    handle = api.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x00200000, None)
    if handle == _INVALID_HANDLE or handle is None:
        raise _error(path)
    return handle


def _metadata(api, handle, path, work):
    _add(work, "checkpoint_payload_metadata_queries")
    if api.GetFileType(handle) != 1:  # FILE_TYPE_DISK
        raise ValueError("payload lease requires an ordinary disk file")
    identity, standard, attributes = _FileId(), _StandardInfo(), _AttributeInfo()
    for kind, value in ((18, identity), (1, standard), (9, attributes)):
        _add(work, "checkpoint_payload_metadata_queries")
        if not api.GetFileInformationByHandleEx(handle, kind, ctypes.byref(value),
                                               ctypes.sizeof(value)):
            raise _error(path)
    if (standard.directory or standard.delete_pending or standard.size < 0
            or attributes.attributes & (0x10 | 0x400) or attributes.reparse_tag):
        raise ValueError("payload lease forbids directories, reparses and pending deletion")
    return (identity.volume, bytes(identity.identifier)), standard.size


class PayloadLease:
    """One non-inherited read handle authenticated once, with explicit lifetime."""

    __slots__ = (
        "_api", "_handle", "_owned_handle", "_path", "_identity", "_bytes", "_sha256", "_work",
        "_counted_live",
    )

    @classmethod
    def open(cls, path, expected_sha256, expected_bytes, work=None):
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_sha256)
                or type(expected_bytes) is not int or expected_bytes < 0):
            raise ValueError("payload lease requires an exact digest and byte length")
        path = Path(os.path.abspath(os.fspath(path)))
        if not supported(path, work):
            raise NotImplementedError("payload lease requires fixed local NTFS")
        obj = cls.__new__(cls)
        obj._api, obj._work, obj._path = _windows(), work, path
        obj._counted_live = False
        obj._handle = obj._owned_handle = None
        try:
            _add(work, "checkpoint_payload_lease_open_attempts")
            obj._owned_handle = _open(obj._api, path)
            obj._handle = obj._owned_handle
            _add(work, "checkpoint_payload_lease_acquisitions")
            _add(work, "checkpoint_payload_live_handles")
            obj._counted_live = work is not None
            if work is not None:
                work["checkpoint_payload_peak_handles"] = max(
                    work.get("checkpoint_payload_peak_handles", 0),
                    work["checkpoint_payload_live_handles"],
                )
            obj._identity, obj._bytes = _metadata(obj._api, obj._handle, path, work)
            if obj._bytes != expected_bytes:
                raise ValueError("payload length changed before lease authentication")
            digest, size = hashlib.sha256(), 0
            buffer = ctypes.create_string_buffer(_BLOCK)
            _add(work, "checkpoint_payload_hash_buffer_bytes", _BLOCK)
            _add(work, "checkpoint_file_hash_passes")
            while True:
                count = ctypes.c_uint32()
                wanted = min(_BLOCK, expected_bytes - size + 1)
                _add(work, "checkpoint_payload_read_calls")
                if not obj._api.ReadFile(obj._handle, buffer, wanted, ctypes.byref(count), None):
                    raise _error(path)
                _add(work, "checkpoint_hash_bytes_read", count.value)
                size += count.value
                if size > expected_bytes:
                    raise ValueError("payload grew during lease authentication")
                if not count.value:
                    break
                digest.update(memoryview(buffer).cast("B")[:count.value])
            obj._sha256 = digest.hexdigest()
            if size != expected_bytes or obj._sha256 != expected_sha256:
                raise ValueError("payload bytes changed during publication handoff")
            obj.verify_path(path, work)
            _add(work, "checkpoint_payload_authenticated_versions")
            return obj
        except BaseException:
            obj.close()
            raise

    @property
    def closed(self):
        return self._owned_handle is None

    def verify_path(self, path, work=None):
        """Verify ownership and current name without rereading leased bytes."""
        if self.closed:
            raise ValueError("payload lease is closed")
        if self._handle != self._owned_handle:
            raise ValueError("payload lease handle was rebound")
        if work is None:
            work = self._work
        path = Path(os.path.abspath(os.fspath(path)))
        if path != self._path:
            raise ValueError("payload lease path was rebound")
        _add(work, "checkpoint_payload_path_checks")
        expected = self._identity, self._bytes
        if _metadata(self._api, self._handle, path, work) != expected:
            raise ValueError("held payload identity or length changed")
        _add(work, "checkpoint_payload_identity_open_attempts")
        handle = _open(self._api, path)
        try:
            _add(work, "checkpoint_payload_identity_opens")
            if _metadata(self._api, handle, path, work) != expected:
                raise ValueError("payload path resolves to a different file")
        finally:
            try:
                _add(work, "checkpoint_payload_identity_close_calls")
            finally:
                if not self._api.CloseHandle(handle):
                    raise _error(path)
        return {"bytes": self._bytes, "sha256": self._sha256}

    def close(self):
        if not self.closed:
            handle = self._owned_handle
            rebound = self._handle != handle
            self._handle = self._owned_handle = None
            try:
                _add(self._work, "checkpoint_payload_lease_close_calls")
            finally:
                if not self._api.CloseHandle(handle):
                    raise _error(self._path)
            _add(self._work, "checkpoint_payload_lease_releases")
            if self._counted_live:
                _add(self._work, "checkpoint_payload_live_handles", -1)
            if rebound:
                raise ValueError("payload lease handle was rebound before close")

    def __enter__(self):
        if self.closed:
            raise ValueError("payload lease is closed")
        return self

    def __exit__(self, *_exception):
        self.close()
