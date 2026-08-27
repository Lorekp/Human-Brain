use std::{
    ffi::c_void,
    fs::File,
    mem::{size_of, MaybeUninit},
    os::windows::io::{AsRawHandle, FromRawHandle, IntoRawHandle, RawHandle},
    ptr,
};

use windows_sys::Win32::{
    Foundation::{
        CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, ERROR_FILE_EXISTS, ERROR_FILE_NOT_FOUND,
        ERROR_PATH_NOT_FOUND, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    },
    Storage::FileSystem::{
        CreateFileW, FileAttributeTagInfo, FileDispositionInfo, FileIdInfo, FlushFileBuffers,
        GetDriveTypeW, GetFileInformationByHandleEx, GetFinalPathNameByHandleW,
        GetVolumeInformationByHandleW, MoveFileExW, SetFileInformationByHandle, CREATE_NEW, DELETE,
        FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_REPARSE_POINT,
        FILE_ATTRIBUTE_TAG_INFO, FILE_DISPOSITION_INFO, FILE_FLAG_BACKUP_SEMANTICS,
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_ID_INFO, FILE_NAME_NORMALIZED, FILE_READ_ATTRIBUTES,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, MOVEFILE_WRITE_THROUGH,
        OPEN_EXISTING, VOLUME_NAME_DOS,
    },
};

use super::{
    CleanupErrors, PublicationCompletion, PublicationError, TempName,
    ValidatedCheckpointDestination, ERROR_STAGE_EXCLUSIVE_PUBLISH, ERROR_STAGE_FILESYSTEM_POLICY,
    ERROR_STAGE_FILE_CLOSE, ERROR_STAGE_FILE_SYNC, ERROR_STAGE_PARENT_IDENTITY,
    ERROR_STAGE_PARENT_OPEN, ERROR_STAGE_TEMP_CLEANUP, ERROR_STAGE_TEMP_CREATE,
    ERROR_STAGE_TEMP_IDENTITY, OUTCOME_COLLISION, OUTCOME_DURABLE_PUBLISHED, OUTCOME_NOT_PUBLISHED,
    OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE, POLICY_ALLOCATION_FAILED, POLICY_INVALID_PARENT,
    POLICY_PARENT_IS_REPARSE, POLICY_PATH_IDENTITY_CHANGED, POLICY_REMOTE_OR_UNKNOWN_FILESYSTEM,
    POLICY_TEMP_IDENTITY_CHANGED, POLICY_TEMP_NAME_EXHAUSTED, POLICY_UNSUPPORTED_FILESYSTEM,
};
use crate::temporal_runtime::checkpoint::PreparedCheckpointV1Export;

const DRIVE_FIXED: u32 = 3;
const WINDOWS_PATH_CAPACITY: usize = 32 * 1024;

#[derive(Debug)]
struct OwnedHandle {
    raw: HANDLE,
}

impl OwnedHandle {
    fn new(raw: HANDLE) -> Self {
        Self { raw }
    }

    fn take(&mut self) -> HANDLE {
        let raw = self.raw;
        self.raw = INVALID_HANDLE_VALUE;
        raw
    }
}

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        if self.raw != INVALID_HANDLE_VALUE {
            unsafe {
                CloseHandle(self.raw);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    volume_serial: u64,
    file_id: [u8; 16],
}

#[derive(Debug)]
struct WindowsParent {
    _handle: OwnedHandle,
    identity: FileIdentity,
    canonical: Vec<u16>,
    maximum_component_length: usize,
}

enum BackendFailure {
    Error(PublicationError),
    Allocation,
}

struct CreatedTemp {
    handle: OwnedHandle,
    name: TempName,
    path: Vec<u16>,
    identity: FileIdentity,
}

struct UnidentifiedOwnedTemp {
    name: TempName,
    primary: PublicationError,
    cleanup: CleanupErrors,
}

enum TempCreateFailure {
    BeforeTemp(BackendFailure),
    Identity(UnidentifiedOwnedTemp),
}

struct VerifiedTempFailure {
    primary: PublicationError,
    close_error: Option<PublicationError>,
}

pub(super) fn publish(
    prepared: &PreparedCheckpointV1Export<'_>,
    destination: &ValidatedCheckpointDestination,
) -> PublicationCompletion {
    let checkpoint_bytes = prepared.encoded_len();
    if !valid_windows_basename(destination.final_basename_utf8()) {
        return PublicationCompletion::without_temp(
            OUTCOME_NOT_PUBLISHED,
            PublicationError::policy(ERROR_STAGE_PARENT_IDENTITY, POLICY_INVALID_PARENT),
            checkpoint_bytes,
        );
    }
    let parent = match open_validated_parent(destination.parent_utf8()) {
        Ok(parent) => parent,
        Err(BackendFailure::Error(error)) => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                error,
                checkpoint_bytes,
            )
        }
        Err(BackendFailure::Allocation) => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                PublicationError::policy(ERROR_STAGE_PARENT_OPEN, POLICY_ALLOCATION_FAILED),
                checkpoint_bytes,
            )
        }
    };
    let basename_wide = match utf8_to_wide(destination.final_basename_utf8()) {
        Ok(name) => name,
        Err(()) => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                PublicationError::policy(ERROR_STAGE_PARENT_IDENTITY, POLICY_ALLOCATION_FAILED),
                checkpoint_bytes,
            )
        }
    };
    if basename_wide.len().saturating_sub(1) > parent.maximum_component_length {
        return PublicationCompletion::without_temp(
            OUTCOME_NOT_PUBLISHED,
            PublicationError::policy(ERROR_STAGE_FILESYSTEM_POLICY, POLICY_UNSUPPORTED_FILESYSTEM),
            checkpoint_bytes,
        );
    }
    let final_path =
        match join_parent_name(&parent.canonical, &basename_wide[..basename_wide.len() - 1]) {
            Ok(path) => path,
            Err(()) => {
                return PublicationCompletion::without_temp(
                    OUTCOME_NOT_PUBLISHED,
                    PublicationError::policy(ERROR_STAGE_PARENT_IDENTITY, POLICY_ALLOCATION_FAILED),
                    checkpoint_bytes,
                )
            }
        };

    let CreatedTemp {
        mut handle,
        name: temp_name,
        path: temp_path,
        identity,
    } = match create_owned_temp(
        &parent.canonical,
        destination.final_basename_utf8(),
        parent.maximum_component_length,
    ) {
        Ok(created) => created,
        Err(TempCreateFailure::BeforeTemp(BackendFailure::Error(error))) => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                error,
                checkpoint_bytes,
            )
        }
        Err(TempCreateFailure::BeforeTemp(BackendFailure::Allocation)) => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                PublicationError::policy(ERROR_STAGE_TEMP_CREATE, POLICY_ALLOCATION_FAILED),
                checkpoint_bytes,
            )
        }
        Err(TempCreateFailure::Identity(failure)) => {
            return PublicationCompletion::terminal(
                OUTCOME_NOT_PUBLISHED,
                failure.primary,
                failure.cleanup,
                failure.name.pid,
                failure.name.nonce,
                checkpoint_bytes,
                0,
            )
        }
    };
    let raw = handle.take();
    let mut file = unsafe { File::from_raw_handle(raw as RawHandle) };
    if let Err(error) = prepared.encode_into(&mut file) {
        let encoded_bytes = observed_file_length(&file, checkpoint_bytes);
        let mut cleanup = CleanupErrors::new();
        close_file_for_cleanup(file, &mut cleanup);
        cleanup_owned_temp(&temp_path, identity, &mut cleanup);
        return PublicationCompletion::encode_failure(
            error,
            cleanup,
            temp_name.pid,
            temp_name.nonce,
            checkpoint_bytes,
            encoded_bytes,
        );
    }
    if unsafe { FlushFileBuffers(file.as_raw_handle() as HANDLE) } == 0 {
        let primary = last_win32(ERROR_STAGE_FILE_SYNC);
        let mut cleanup = CleanupErrors::new();
        close_file_for_cleanup(file, &mut cleanup);
        cleanup_owned_temp(&temp_path, identity, &mut cleanup);
        return PublicationCompletion::terminal(
            OUTCOME_NOT_PUBLISHED,
            primary,
            cleanup,
            temp_name.pid,
            temp_name.nonce,
            checkpoint_bytes,
            checkpoint_bytes,
        );
    }
    if let Err(primary) = close_file_checked(file) {
        let mut cleanup = CleanupErrors::new();
        cleanup_owned_temp(&temp_path, identity, &mut cleanup);
        return PublicationCompletion::terminal(
            OUTCOME_NOT_PUBLISHED,
            primary,
            cleanup,
            temp_name.pid,
            temp_name.nonce,
            checkpoint_bytes,
            checkpoint_bytes,
        );
    }
    if let Err(primary) = verify_parent_identity(&parent) {
        let mut cleanup = CleanupErrors::new();
        cleanup_owned_temp(&temp_path, identity, &mut cleanup);
        return PublicationCompletion::terminal(
            OUTCOME_NOT_PUBLISHED,
            primary,
            cleanup,
            temp_name.pid,
            temp_name.nonce,
            checkpoint_bytes,
            checkpoint_bytes,
        );
    }
    let verified_temp = match open_verified_temp(&temp_path, identity) {
        Ok(handle) => handle,
        Err(failure) => {
            let mut cleanup = CleanupErrors::new();
            cleanup.push(failure.primary);
            if let Some(close_error) = failure.close_error {
                cleanup.push(close_error);
            }
            return PublicationCompletion::terminal(
                OUTCOME_NOT_PUBLISHED,
                PublicationError::policy(ERROR_STAGE_TEMP_IDENTITY, POLICY_TEMP_IDENTITY_CHANGED),
                cleanup,
                temp_name.pid,
                temp_name.nonce,
                checkpoint_bytes,
                checkpoint_bytes,
            );
        }
    };
    if let Err(primary) = close_owned_handle_checked(verified_temp) {
        let mut cleanup = CleanupErrors::new();
        cleanup_owned_temp(&temp_path, identity, &mut cleanup);
        return PublicationCompletion::terminal(
            OUTCOME_NOT_PUBLISHED,
            primary,
            cleanup,
            temp_name.pid,
            temp_name.nonce,
            checkpoint_bytes,
            checkpoint_bytes,
        );
    }

    debug_assert_eq!(MOVEFILE_WRITE_THROUGH, 0x0000_0008);
    if unsafe { MoveFileExW(temp_path.as_ptr(), final_path.as_ptr(), 0x0000_0008) } != 0 {
        return PublicationCompletion::terminal(
            OUTCOME_DURABLE_PUBLISHED,
            PublicationError::ZERO,
            CleanupErrors::new(),
            temp_name.pid,
            temp_name.nonce,
            checkpoint_bytes,
            checkpoint_bytes,
        );
    }
    let move_error = unsafe { GetLastError() };
    let mut cleanup = CleanupErrors::new();
    cleanup_owned_temp(&temp_path, identity, &mut cleanup);
    let (outcome, primary) = classify_move_failure(move_error);
    PublicationCompletion::terminal(
        outcome,
        primary,
        cleanup,
        temp_name.pid,
        temp_name.nonce,
        checkpoint_bytes,
        checkpoint_bytes,
    )
}

fn classify_move_failure(code: u32) -> (u16, PublicationError) {
    if code == ERROR_FILE_EXISTS || code == ERROR_ALREADY_EXISTS {
        (OUTCOME_COLLISION, PublicationError::ZERO)
    } else {
        (
            OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
            PublicationError::win32(ERROR_STAGE_EXCLUSIVE_PUBLISH, code),
        )
    }
}

fn open_validated_parent(parent_utf8: &[u8]) -> Result<WindowsParent, BackendFailure> {
    let parent = std::str::from_utf8(parent_utf8)
        .map_err(|_| BackendFailure::Error(invalid_parent(ERROR_STAGE_PARENT_OPEN)))?;
    let normalized = normalized_drive_path(parent)?;
    let root = [normalized[0], b':' as u16, b'\\' as u16, 0];
    if unsafe { GetDriveTypeW(root.as_ptr()) } != DRIVE_FIXED {
        return Err(BackendFailure::Error(PublicationError::policy(
            ERROR_STAGE_FILESYSTEM_POLICY,
            POLICY_REMOTE_OR_UNKNOWN_FILESYSTEM,
        )));
    }

    // OPEN_REPARSE_POINT applies to the requested final component.  Resolve
    // any intermediate alias once, retain the final parent handle, and accept
    // only the canonical local DOS/NTFS target obtained from that handle.
    // Publication thereafter uses only that canonical path and rechecks the
    // final parent's identity immediately before MoveFileExW.
    let current = open_directory(normalized.as_slice())?;
    validate_directory_handle(current.raw)?;

    let identity =
        handle_identity(current.raw, ERROR_STAGE_PARENT_IDENTITY).map_err(BackendFailure::Error)?;
    let (maximum_component_length, is_ntfs) = volume_information(current.raw)?;
    if !is_ntfs {
        return Err(BackendFailure::Error(PublicationError::policy(
            ERROR_STAGE_FILESYSTEM_POLICY,
            POLICY_UNSUPPORTED_FILESYSTEM,
        )));
    }
    let canonical = canonical_handle_path(current.raw, normalized[0])?;
    Ok(WindowsParent {
        _handle: current,
        identity,
        canonical,
        maximum_component_length,
    })
}

fn normalized_drive_path(parent: &str) -> Result<Vec<u16>, BackendFailure> {
    if parent.starts_with("\\\\")
        || parent.starts_with("//")
        || parent.starts_with("\\\\?\\")
        || parent.starts_with("\\\\.\\")
    {
        return Err(BackendFailure::Error(invalid_parent(
            ERROR_STAGE_PARENT_OPEN,
        )));
    }
    let mut normalized = Vec::new();
    normalized
        .try_reserve_exact(parent.len())
        .map_err(|_| BackendFailure::Allocation)?;
    normalized.extend(parent.encode_utf16().map(|unit| {
        if unit == b'/' as u16 {
            b'\\' as u16
        } else {
            unit
        }
    }));
    let drive_is_ascii = matches!(normalized.first(), Some(unit) if (*unit >= b'A' as u16 && *unit <= b'Z' as u16) || (*unit >= b'a' as u16 && *unit <= b'z' as u16));
    if normalized.len() < 3
        || !drive_is_ascii
        || normalized[1] != b':' as u16
        || normalized[2] != b'\\' as u16
        || normalized.len() > WINDOWS_PATH_CAPACITY
        || (normalized.last().is_some_and(|unit| *unit == b'\\' as u16) && normalized.len() > 3)
    {
        return Err(BackendFailure::Error(invalid_parent(
            ERROR_STAGE_PARENT_OPEN,
        )));
    }
    if normalized.len() > 3 {
        let mut component_start = 3;
        for index in 3..=normalized.len() {
            if index == normalized.len() || normalized[index] == b'\\' as u16 {
                let component = &normalized[component_start..index];
                if component.is_empty()
                    || component == [b'.' as u16]
                    || component == [b'.' as u16, b'.' as u16]
                    || component
                        .last()
                        .is_some_and(|unit| *unit == b'.' as u16 || *unit == b' ' as u16)
                    || component.iter().any(|unit| {
                        *unit == b':' as u16
                            || *unit == b'"' as u16
                            || *unit == b'<' as u16
                            || *unit == b'>' as u16
                            || *unit == b'|' as u16
                            || *unit == b'?' as u16
                            || *unit == b'*' as u16
                    })
                {
                    return Err(BackendFailure::Error(invalid_parent(
                        ERROR_STAGE_PARENT_OPEN,
                    )));
                }
                component_start = index + 1;
            }
        }
    }
    Ok(normalized)
}

fn open_directory(path_without_nul: &[u16]) -> Result<OwnedHandle, BackendFailure> {
    let path = nul_terminated_copy(path_without_nul).map_err(|_| BackendFailure::Allocation)?;
    let raw = unsafe {
        CreateFileW(
            path.as_ptr(),
            FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
            ptr::null_mut(),
        )
    };
    if raw == INVALID_HANDLE_VALUE {
        Err(BackendFailure::Error(last_win32(ERROR_STAGE_PARENT_OPEN)))
    } else {
        Ok(OwnedHandle::new(raw))
    }
}

fn validate_directory_handle(handle: HANDLE) -> Result<(), BackendFailure> {
    let attributes =
        file_attribute_tag(handle, ERROR_STAGE_PARENT_IDENTITY).map_err(BackendFailure::Error)?;
    if attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(BackendFailure::Error(PublicationError::policy(
            ERROR_STAGE_PARENT_IDENTITY,
            POLICY_PARENT_IS_REPARSE,
        )));
    }
    if attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY == 0 {
        return Err(BackendFailure::Error(invalid_parent(
            ERROR_STAGE_PARENT_IDENTITY,
        )));
    }
    Ok(())
}

fn file_attribute_tag(
    handle: HANDLE,
    stage: u16,
) -> Result<FILE_ATTRIBUTE_TAG_INFO, PublicationError> {
    let mut info = MaybeUninit::<FILE_ATTRIBUTE_TAG_INFO>::uninit();
    if unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileAttributeTagInfo,
            info.as_mut_ptr().cast::<c_void>(),
            size_of::<FILE_ATTRIBUTE_TAG_INFO>() as u32,
        )
    } == 0
    {
        Err(last_win32(stage))
    } else {
        Ok(unsafe { info.assume_init() })
    }
}

fn handle_identity(handle: HANDLE, stage: u16) -> Result<FileIdentity, PublicationError> {
    let mut info = MaybeUninit::<FILE_ID_INFO>::uninit();
    if unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            info.as_mut_ptr().cast::<c_void>(),
            size_of::<FILE_ID_INFO>() as u32,
        )
    } == 0
    {
        return Err(last_win32(stage));
    }
    let info = unsafe { info.assume_init() };
    Ok(FileIdentity {
        volume_serial: info.VolumeSerialNumber,
        file_id: info.FileId.Identifier,
    })
}

fn volume_information(handle: HANDLE) -> Result<(usize, bool), BackendFailure> {
    let mut maximum_component_length = 0u32;
    let mut filesystem = [0u16; 32];
    if unsafe {
        GetVolumeInformationByHandleW(
            handle,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            &mut maximum_component_length,
            ptr::null_mut(),
            filesystem.as_mut_ptr(),
            filesystem.len() as u32,
        )
    } == 0
    {
        return Err(BackendFailure::Error(last_win32(
            ERROR_STAGE_FILESYSTEM_POLICY,
        )));
    }
    let length = filesystem
        .iter()
        .position(|unit| *unit == 0)
        .unwrap_or(filesystem.len());
    let expected = [b'N' as u16, b'T' as u16, b'F' as u16, b'S' as u16];
    let is_ntfs = filesystem[..length]
        .iter()
        .zip(expected)
        .all(|(observed, expected)| ascii_upper_wide(*observed) == expected)
        && length == expected.len();
    Ok((maximum_component_length as usize, is_ntfs))
}

fn canonical_handle_path(handle: HANDLE, expected_drive: u16) -> Result<Vec<u16>, BackendFailure> {
    let flags = FILE_NAME_NORMALIZED | VOLUME_NAME_DOS;
    let required = unsafe { GetFinalPathNameByHandleW(handle, ptr::null_mut(), 0, flags) };
    if required == 0 {
        return Err(BackendFailure::Error(last_win32(
            ERROR_STAGE_PARENT_IDENTITY,
        )));
    }
    if required as usize > WINDOWS_PATH_CAPACITY {
        return Err(BackendFailure::Error(PublicationError::policy(
            ERROR_STAGE_PARENT_IDENTITY,
            POLICY_INVALID_PARENT,
        )));
    }
    let Some(capacity) = (required as usize).checked_add(1) else {
        return Err(BackendFailure::Allocation);
    };
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(capacity)
        .map_err(|_| BackendFailure::Allocation)?;
    buffer.resize(capacity, 0);
    let written = unsafe {
        GetFinalPathNameByHandleW(handle, buffer.as_mut_ptr(), buffer.len() as u32, flags)
    };
    if written == 0 || written as usize >= buffer.len() {
        return Err(BackendFailure::Error(last_win32(
            ERROR_STAGE_PARENT_IDENTITY,
        )));
    }
    buffer.truncate(written as usize);
    if !is_canonical_local_dos_path(&buffer, expected_drive) {
        return Err(BackendFailure::Error(PublicationError::policy(
            ERROR_STAGE_PARENT_IDENTITY,
            POLICY_REMOTE_OR_UNKNOWN_FILESYSTEM,
        )));
    }
    Ok(buffer)
}

fn is_canonical_local_dos_path(path: &[u16], expected_drive: u16) -> bool {
    path.len() >= 7
        && path[..4] == [b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16]
        && ascii_upper_wide(path[4]) == ascii_upper_wide(expected_drive)
        && path[5] == b':' as u16
        && path[6] == b'\\' as u16
}

fn create_owned_temp(
    canonical_parent: &[u16],
    final_basename: &[u8],
    maximum_component_length: usize,
) -> Result<CreatedTemp, TempCreateFailure> {
    for _ in 0..TempName::max_attempts() {
        let temp_name = TempName::next(final_basename);
        if temp_name.as_bytes().len() > maximum_component_length {
            return Err(TempCreateFailure::BeforeTemp(BackendFailure::Error(
                PublicationError::policy(ERROR_STAGE_TEMP_CREATE, POLICY_UNSUPPORTED_FILESYSTEM),
            )));
        }
        let name_wide = ascii_to_wide(temp_name.as_bytes())
            .map_err(|()| TempCreateFailure::BeforeTemp(BackendFailure::Allocation))?;
        let path = join_parent_name(canonical_parent, &name_wide)
            .map_err(|_| TempCreateFailure::BeforeTemp(BackendFailure::Allocation))?;
        let raw = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_WRITE | FILE_READ_ATTRIBUTES | DELETE,
                0,
                ptr::null(),
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
                ptr::null_mut(),
            )
        };
        if raw != INVALID_HANDLE_VALUE {
            let owned = OwnedHandle::new(raw);
            let identity = match handle_identity(raw, ERROR_STAGE_TEMP_IDENTITY) {
                Ok(identity) => identity,
                Err(primary) => {
                    let mut cleanup = CleanupErrors::new();
                    cleanup.push(primary);
                    close_owned_handle_for_cleanup(owned, &mut cleanup);
                    return Err(TempCreateFailure::Identity(UnidentifiedOwnedTemp {
                        name: temp_name,
                        primary,
                        cleanup,
                    }));
                }
            };
            return Ok(CreatedTemp {
                handle: owned,
                name: temp_name,
                path,
                identity,
            });
        }
        let code = unsafe { GetLastError() };
        if code != ERROR_FILE_EXISTS && code != ERROR_ALREADY_EXISTS {
            return Err(TempCreateFailure::BeforeTemp(BackendFailure::Error(
                PublicationError::win32(ERROR_STAGE_TEMP_CREATE, code),
            )));
        }
    }
    Err(TempCreateFailure::BeforeTemp(BackendFailure::Error(
        PublicationError::policy(ERROR_STAGE_TEMP_CREATE, POLICY_TEMP_NAME_EXHAUSTED),
    )))
}

fn verify_parent_identity(parent: &WindowsParent) -> Result<(), PublicationError> {
    let reopened = open_directory(&parent.canonical[..]).map_err(|failure| match failure {
        BackendFailure::Error(error) => error,
        BackendFailure::Allocation => {
            PublicationError::policy(ERROR_STAGE_PARENT_IDENTITY, POLICY_ALLOCATION_FAILED)
        }
    })?;
    validate_directory_handle(reopened.raw).map_err(|failure| match failure {
        BackendFailure::Error(error) => error,
        BackendFailure::Allocation => {
            PublicationError::policy(ERROR_STAGE_PARENT_IDENTITY, POLICY_ALLOCATION_FAILED)
        }
    })?;
    let observed = handle_identity(reopened.raw, ERROR_STAGE_PARENT_IDENTITY)?;
    if observed != parent.identity {
        Err(PublicationError::policy(
            ERROR_STAGE_PARENT_IDENTITY,
            POLICY_PATH_IDENTITY_CHANGED,
        ))
    } else {
        Ok(())
    }
}

fn open_verified_temp(
    path: &[u16],
    expected: FileIdentity,
) -> Result<OwnedHandle, VerifiedTempFailure> {
    let raw = unsafe {
        CreateFileW(
            path.as_ptr(),
            FILE_READ_ATTRIBUTES | DELETE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT,
            ptr::null_mut(),
        )
    };
    if raw == INVALID_HANDLE_VALUE {
        return Err(VerifiedTempFailure {
            primary: last_win32(ERROR_STAGE_TEMP_IDENTITY),
            close_error: None,
        });
    }
    let handle = OwnedHandle::new(raw);
    let attributes = match file_attribute_tag(raw, ERROR_STAGE_TEMP_IDENTITY) {
        Ok(attributes) => attributes,
        Err(primary) => {
            return Err(VerifiedTempFailure {
                primary,
                close_error: close_owned_handle_checked(handle).err(),
            });
        }
    };
    if attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(VerifiedTempFailure {
            primary: PublicationError::policy(
                ERROR_STAGE_TEMP_IDENTITY,
                POLICY_TEMP_IDENTITY_CHANGED,
            ),
            close_error: close_owned_handle_checked(handle).err(),
        });
    }
    let observed = match handle_identity(handle.raw, ERROR_STAGE_TEMP_IDENTITY) {
        Ok(observed) => observed,
        Err(primary) => {
            return Err(VerifiedTempFailure {
                primary,
                close_error: close_owned_handle_checked(handle).err(),
            });
        }
    };
    if observed != expected {
        Err(VerifiedTempFailure {
            primary: PublicationError::policy(
                ERROR_STAGE_TEMP_IDENTITY,
                POLICY_TEMP_IDENTITY_CHANGED,
            ),
            close_error: close_owned_handle_checked(handle).err(),
        })
    } else {
        Ok(handle)
    }
}

fn cleanup_owned_temp(path: &[u16], expected: FileIdentity, cleanup: &mut CleanupErrors) {
    match open_verified_temp(path, expected) {
        Ok(handle) => {
            let disposition = FILE_DISPOSITION_INFO { DeleteFile: true };
            if unsafe {
                SetFileInformationByHandle(
                    handle.raw,
                    FileDispositionInfo,
                    (&disposition as *const FILE_DISPOSITION_INFO).cast::<c_void>(),
                    size_of::<FILE_DISPOSITION_INFO>() as u32,
                )
            } == 0
            {
                cleanup.push(last_win32(ERROR_STAGE_TEMP_CLEANUP));
            }
            close_owned_handle_for_cleanup(handle, cleanup);
        }
        Err(failure)
            if failure.primary.domain == super::ERROR_DOMAIN_WIN32
                && (failure.primary.code == ERROR_FILE_NOT_FOUND
                    || failure.primary.code == ERROR_PATH_NOT_FOUND) =>
        {
            if let Some(close_error) = failure.close_error {
                cleanup.push(close_error);
            }
        }
        Err(failure) => {
            cleanup.push(failure.primary);
            if let Some(close_error) = failure.close_error {
                cleanup.push(close_error);
            }
        }
    }
}

fn close_owned_handle_checked(mut handle: OwnedHandle) -> Result<(), PublicationError> {
    let raw = handle.take();
    if unsafe { CloseHandle(raw) } != 0 {
        Ok(())
    } else {
        Err(last_win32(ERROR_STAGE_FILE_CLOSE))
    }
}

fn close_owned_handle_for_cleanup(handle: OwnedHandle, cleanup: &mut CleanupErrors) {
    if let Err(error) = close_owned_handle_checked(handle) {
        cleanup.push(error);
    }
}

fn close_file_checked(file: File) -> Result<(), PublicationError> {
    let raw = file.into_raw_handle() as HANDLE;
    if unsafe { CloseHandle(raw) } != 0 {
        Ok(())
    } else {
        Err(last_win32(ERROR_STAGE_FILE_CLOSE))
    }
}

fn close_file_for_cleanup(file: File, cleanup: &mut CleanupErrors) {
    if let Err(error) = close_file_checked(file) {
        cleanup.push(error);
    }
}

fn valid_windows_basename(value: &[u8]) -> bool {
    if std::str::from_utf8(value).is_err() {
        return false;
    }
    if value.ends_with(b".")
        || value.ends_with(b" ")
        || value.iter().any(|byte| {
            *byte <= 0x1f
                || matches!(
                    *byte,
                    b'<' | b'>' | b':' | b'"' | b'/' | b'\\' | b'|' | b'?' | b'*'
                )
        })
    {
        return false;
    }
    let stem = value.split(|byte| *byte == b'.').next().unwrap_or(value);
    if stem.ends_with(b" ") {
        return false;
    }
    ![
        b"CON".as_slice(),
        b"PRN",
        b"AUX",
        b"NUL",
        b"CONIN$",
        b"CONOUT$",
        b"CLOCK$",
    ]
    .iter()
    .any(|reserved| stem.eq_ignore_ascii_case(reserved))
        && !is_reserved_numbered_device(stem)
}

fn is_reserved_numbered_device(stem: &[u8]) -> bool {
    let numbered_prefix = stem.len() >= 3
        && (stem[..3].eq_ignore_ascii_case(b"COM") || stem[..3].eq_ignore_ascii_case(b"LPT"));
    numbered_prefix
        && ((stem.len() == 4 && matches!(stem[3], b'1'..=b'9'))
            || (stem.len() == 5 && stem[3] == 0xc2 && matches!(stem[4], 0xb9 | 0xb2 | 0xb3)))
}

fn utf8_to_wide(value: &[u8]) -> Result<Vec<u16>, ()> {
    let value = std::str::from_utf8(value).map_err(|_| ())?;
    let units = value.encode_utf16().count();
    let capacity = units.checked_add(1).ok_or(())?;
    let mut wide = Vec::new();
    wide.try_reserve_exact(capacity).map_err(|_| ())?;
    wide.extend(value.encode_utf16());
    wide.push(0);
    Ok(wide)
}

fn ascii_to_wide(value: &[u8]) -> Result<Vec<u16>, ()> {
    let mut wide = Vec::new();
    wide.try_reserve_exact(value.len()).map_err(|_| ())?;
    wide.extend(value.iter().map(|byte| u16::from(*byte)));
    Ok(wide)
}

fn nul_terminated_copy(value: &[u16]) -> Result<Vec<u16>, ()> {
    let value = value.strip_suffix(&[0]).unwrap_or(value);
    let capacity = value.len().checked_add(1).ok_or(())?;
    let mut copied = Vec::new();
    copied.try_reserve_exact(capacity).map_err(|_| ())?;
    copied.extend_from_slice(value);
    copied.push(0);
    Ok(copied)
}

fn join_parent_name(parent: &[u16], name_without_nul: &[u16]) -> Result<Vec<u16>, ()> {
    let parent = parent.strip_suffix(&[0]).unwrap_or(parent);
    let needs_separator = parent.last().is_some_and(|unit| *unit != b'\\' as u16);
    let capacity = parent
        .len()
        .checked_add(usize::from(needs_separator))
        .and_then(|length| length.checked_add(name_without_nul.len()))
        .and_then(|length| length.checked_add(1))
        .ok_or(())?;
    let mut joined = Vec::new();
    joined.try_reserve_exact(capacity).map_err(|_| ())?;
    joined.extend_from_slice(parent);
    if needs_separator {
        joined.push(b'\\' as u16);
    }
    joined.extend_from_slice(name_without_nul);
    joined.push(0);
    Ok(joined)
}

fn invalid_parent(stage: u16) -> PublicationError {
    PublicationError::policy(stage, POLICY_INVALID_PARENT)
}

fn ascii_upper_wide(unit: u16) -> u16 {
    if unit >= b'a' as u16 && unit <= b'z' as u16 {
        unit - (b'a' - b'A') as u16
    } else {
        unit
    }
}

fn observed_file_length(file: &File, checkpoint_bytes: u64) -> u64 {
    file.metadata()
        .map(|metadata| metadata.len().min(checkpoint_bytes))
        .unwrap_or(0)
}

fn last_win32(stage: u16) -> PublicationError {
    PublicationError::win32(stage, unsafe { GetLastError() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basename_grammar_rejects_windows_aliases() {
        for invalid in [
            b"CON".as_slice(),
            b"con.txt".as_slice(),
            b"CONIN$".as_slice(),
            b"conout$.log".as_slice(),
            b"LPT9.bin".as_slice(),
            "COM¹.txt".as_bytes(),
            "lpt³".as_bytes(),
            b"CON .txt".as_slice(),
            b"bad:name".as_slice(),
            b"trailing.".as_slice(),
            b"trailing ".as_slice(),
        ] {
            assert!(!valid_windows_basename(invalid), "{invalid:?}");
        }
        assert!(valid_windows_basename(b"checkpoint-v1.bin"));
    }

    #[test]
    fn joined_path_has_one_separator_and_one_nul() {
        let joined =
            join_parent_name(&[b'C' as u16, b':' as u16, b'\\' as u16], &[b'x' as u16]).unwrap();
        assert_eq!(
            joined,
            [b'C' as u16, b':' as u16, b'\\' as u16, b'x' as u16, 0]
        );
    }

    #[test]
    fn windows_move_flags_are_exactly_write_through_only() {
        assert_eq!(MOVEFILE_WRITE_THROUGH, 0x0000_0008);
    }

    #[test]
    fn move_failure_classification_has_only_two_collision_codes() {
        for code in [ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS] {
            assert_eq!(
                classify_move_failure(code),
                (OUTCOME_COLLISION, PublicationError::ZERO)
            );
        }
        for code in [0, 2, 5, 17, 32, u32::MAX] {
            let (outcome, error) = classify_move_failure(code);
            assert_eq!(outcome, OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE);
            assert_eq!(error.stage, ERROR_STAGE_EXCLUSIVE_PUBLISH);
            assert_eq!(error.domain, super::super::ERROR_DOMAIN_WIN32);
            assert_eq!(error.code, code);
        }
    }

    #[test]
    fn canonical_parent_form_rejects_unc_and_drive_mismatch() {
        let local = [
            b'\\' as u16,
            b'\\' as u16,
            b'?' as u16,
            b'\\' as u16,
            b'C' as u16,
            b':' as u16,
            b'\\' as u16,
        ];
        assert!(is_canonical_local_dos_path(&local, b'c' as u16));
        assert!(!is_canonical_local_dos_path(&local, b'D' as u16));
        let unc = [
            b'\\' as u16,
            b'\\' as u16,
            b'?' as u16,
            b'\\' as u16,
            b'U' as u16,
            b'N' as u16,
            b'C' as u16,
        ];
        assert!(!is_canonical_local_dos_path(&unc, b'U' as u16));
    }

    #[test]
    fn requested_parent_grammar_rejects_aliasing_and_dos_ambiguity() {
        for invalid in [
            r"relative\path",
            r"C:\one\..\two",
            r"C:\one\.\two",
            r"C:\one\\two",
            r"C:\trailing.\two",
            r"C:\alternate:stream",
            r"\\server\share\path",
            r"\\?\C:\extended",
        ] {
            assert!(normalized_drive_path(invalid).is_err(), "{invalid}");
        }
        assert!(normalized_drive_path(r"C:\plain parent\child").is_ok());
    }
}
