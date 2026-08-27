//! Exclusive, no-overwrite publication of prepared immutable payloads.
//!
//! The checkpoint codec and bounded byte capability own their payload bytes.
//! This module owns only the destination namespace, durability boundary, and
//! the four publication outcomes. It deliberately exposes no replace, retry,
//! or fallback option.

use std::{
    fs::File,
    io,
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(test)]
use std::cell::Cell;

use super::checkpoint::{CheckpointV1EncodeError, PreparedCheckpointV1Export};

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

pub(crate) const OUTCOME_NONE: u16 = 0;
pub(crate) const OUTCOME_DURABLE_PUBLISHED: u16 = 1;
pub(crate) const OUTCOME_COLLISION: u16 = 2;
pub(crate) const OUTCOME_NOT_PUBLISHED: u16 = 3;
pub(crate) const OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE: u16 = 4;

pub(crate) const CLEANUP_NOT_APPLICABLE: u8 = 0;
pub(crate) const CLEANUP_CLEAN: u8 = 1;
pub(crate) const CLEANUP_ORPHANED: u8 = 2;

pub(crate) const ERROR_DOMAIN_NONE: u16 = 0;
pub(crate) const ERROR_DOMAIN_POSIX: u16 = 1;
pub(crate) const ERROR_DOMAIN_WIN32: u16 = 2;
pub(crate) const ERROR_DOMAIN_POLICY: u16 = 3;
pub(crate) const ERROR_DOMAIN_ENCODER: u16 = 4;
pub(crate) const ERROR_DOMAIN_IO_WITHOUT_OS_CODE: u16 = 5;

pub(crate) const ERROR_STAGE_NONE: u16 = 0;
pub(crate) const ERROR_STAGE_PARENT_OPEN: u16 = 1;
pub(crate) const ERROR_STAGE_PARENT_IDENTITY: u16 = 2;
pub(crate) const ERROR_STAGE_FILESYSTEM_POLICY: u16 = 3;
pub(crate) const ERROR_STAGE_TEMP_CREATE: u16 = 4;
pub(crate) const ERROR_STAGE_ENCODE: u16 = 5;
pub(crate) const ERROR_STAGE_FILE_SYNC: u16 = 6;
pub(crate) const ERROR_STAGE_FILE_CLOSE: u16 = 7;
pub(crate) const ERROR_STAGE_TEMP_IDENTITY: u16 = 8;
pub(crate) const ERROR_STAGE_EXCLUSIVE_PUBLISH: u16 = 9;
pub(crate) const ERROR_STAGE_TEMP_CLEANUP: u16 = 10;
pub(crate) const ERROR_STAGE_PARENT_SYNC: u16 = 11;

pub(crate) const DURABILITY_OUTCOME_NONE: u16 = 0;
pub(crate) const DURABILITY_OUTCOME_DURABLE: u16 = 1;
pub(crate) const DURABILITY_OUTCOME_NOT_DURABLE: u16 = 2;
pub(crate) const DURABILITY_NO_FAILED_COMPONENT: u32 = u32::MAX;

pub(crate) const DURABILITY_STAGE_NONE: u16 = 0;
pub(crate) const DURABILITY_STAGE_PARENT_OPEN: u16 = 1;
pub(crate) const DURABILITY_STAGE_PARENT_IDENTITY: u16 = 2;
pub(crate) const DURABILITY_STAGE_FILESYSTEM_POLICY: u16 = 3;
pub(crate) const DURABILITY_STAGE_FINAL_OPEN: u16 = 4;
pub(crate) const DURABILITY_STAGE_FINAL_IDENTITY: u16 = 5;
pub(crate) const DURABILITY_STAGE_FINAL_READ: u16 = 6;
pub(crate) const DURABILITY_STAGE_PAYLOAD_MISMATCH: u16 = 7;
pub(crate) const DURABILITY_STAGE_FILE_SYNC: u16 = 8;
pub(crate) const DURABILITY_STAGE_DIRECTORY_CREATE: u16 = 9;
pub(crate) const DURABILITY_STAGE_DIRECTORY_OPEN: u16 = 10;
pub(crate) const DURABILITY_STAGE_DIRECTORY_IDENTITY: u16 = 11;
pub(crate) const DURABILITY_STAGE_DIRECTORY_SYNC: u16 = 12;
pub(crate) const DURABILITY_STAGE_FILE_CLOSE: u16 = 13;
pub(crate) const DURABILITY_STAGE_PATH_IDENTITY: u16 = 14;

pub(crate) const POLICY_INVALID_PARENT: u32 = 1;
pub(crate) const POLICY_PARENT_IS_REPARSE: u32 = 2;
pub(crate) const POLICY_REMOTE_OR_UNKNOWN_FILESYSTEM: u32 = 3;
pub(crate) const POLICY_UNSUPPORTED_FILESYSTEM: u32 = 4;
pub(crate) const POLICY_TEMP_IDENTITY_CHANGED: u32 = 6;
pub(crate) const POLICY_TEMP_NAME_EXHAUSTED: u32 = 7;
pub(crate) const POLICY_PATH_IDENTITY_CHANGED: u32 = 8;
pub(crate) const POLICY_ALLOCATION_FAILED: u32 = 9;
pub(crate) const POLICY_NOT_REGULAR_FILE: u32 = 10;
pub(crate) const POLICY_BYTE_MISMATCH: u32 = 11;
pub(crate) const POLICY_SIZE_MISMATCH: u32 = 12;
pub(crate) const POLICY_NOT_DIRECTORY: u32 = 13;

const MAX_PATH_BYTES: usize = 32 * 1024;
const MAX_BASENAME_BYTES: usize = 255;
const MAX_TEMP_ATTEMPTS: usize = 256;
const CHECKPOINT_TEMP_PREFIX: &[u8] = b".humanbrain-ckpt-v1-";
const IMMUTABLE_BYTES_TEMP_PREFIX: &[u8] = b".humanbrain-bytes-v1-";
const TEMP_SUFFIX: &[u8] = b".tmp";
const TEMP_NAME_CAPACITY: usize = 64;
const CLEANUP_ERROR_CAPACITY: usize = 4;
const DURABILITY_CLEANUP_ERROR_CAPACITY: usize = 2;
pub(crate) const MAX_IMMUTABLE_PAYLOAD_BYTES: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DurabilityCleanupErrors {
    errors: [PublicationError; DURABILITY_CLEANUP_ERROR_CAPACITY],
    len: u8,
}

impl DurabilityCleanupErrors {
    pub(crate) const fn new() -> Self {
        Self {
            errors: [PublicationError::ZERO; DURABILITY_CLEANUP_ERROR_CAPACITY],
            len: 0,
        }
    }

    pub(crate) fn push(&mut self, error: PublicationError) {
        let index = usize::from(self.len);
        if let Some(slot) = self.errors.get_mut(index) {
            *slot = error;
            self.len += 1;
        }
    }

    pub(crate) const fn len(&self) -> u8 {
        self.len
    }

    pub(crate) const fn errors(&self) -> &[PublicationError; DURABILITY_CLEANUP_ERROR_CAPACITY] {
        &self.errors
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PathDurabilityCompletion {
    pub(crate) outcome: u16,
    pub(crate) primary_error: PublicationError,
    pub(crate) cleanup_errors: DurabilityCleanupErrors,
    pub(crate) component_count: u32,
    pub(crate) completed_component_count: u32,
    pub(crate) created_component_count: u32,
    pub(crate) failed_component_index: u32,
    pub(crate) expected_bytes: u64,
    pub(crate) verified_bytes: u64,
}

impl PathDurabilityCompletion {
    pub(crate) const fn reconciled(payload_bytes: u64) -> Self {
        Self {
            outcome: DURABILITY_OUTCOME_DURABLE,
            primary_error: PublicationError::ZERO,
            cleanup_errors: DurabilityCleanupErrors::new(),
            component_count: 0,
            completed_component_count: 0,
            created_component_count: 0,
            failed_component_index: DURABILITY_NO_FAILED_COMPONENT,
            expected_bytes: payload_bytes,
            verified_bytes: payload_bytes,
        }
    }

    pub(crate) const fn reconciliation_failed(
        error: PublicationError,
        payload_bytes: u64,
        verified_bytes: u64,
    ) -> Self {
        Self {
            outcome: DURABILITY_OUTCOME_NOT_DURABLE,
            primary_error: error,
            cleanup_errors: DurabilityCleanupErrors::new(),
            component_count: 0,
            completed_component_count: 0,
            created_component_count: 0,
            failed_component_index: DURABILITY_NO_FAILED_COMPONENT,
            expected_bytes: payload_bytes,
            verified_bytes,
        }
    }

    pub(crate) const fn directory_prepared(
        component_count: u32,
        created_component_count: u32,
    ) -> Self {
        Self {
            outcome: DURABILITY_OUTCOME_DURABLE,
            primary_error: PublicationError::ZERO,
            cleanup_errors: DurabilityCleanupErrors::new(),
            component_count,
            completed_component_count: component_count,
            created_component_count,
            failed_component_index: DURABILITY_NO_FAILED_COMPONENT,
            expected_bytes: 0,
            verified_bytes: 0,
        }
    }

    pub(crate) const fn directory_failed(
        error: PublicationError,
        component_count: u32,
        completed_component_count: u32,
        created_component_count: u32,
        failed_component_index: u32,
    ) -> Self {
        Self {
            outcome: DURABILITY_OUTCOME_NOT_DURABLE,
            primary_error: error,
            cleanup_errors: DurabilityCleanupErrors::new(),
            component_count,
            completed_component_count,
            created_component_count,
            failed_component_index,
            expected_bytes: 0,
            verified_bytes: 0,
        }
    }

    pub(crate) fn push_cleanup(&mut self, error: PublicationError) {
        self.cleanup_errors.push(error);
    }
}

static NEXT_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PublicationError {
    pub(crate) stage: u16,
    pub(crate) domain: u16,
    pub(crate) code: u32,
}

impl PublicationError {
    pub(crate) const ZERO: Self = Self {
        stage: ERROR_STAGE_NONE,
        domain: ERROR_DOMAIN_NONE,
        code: 0,
    };

    pub(crate) const fn policy(stage: u16, code: u32) -> Self {
        Self {
            stage,
            domain: ERROR_DOMAIN_POLICY,
            code,
        }
    }

    #[cfg(unix)]
    pub(crate) fn last_posix(stage: u16) -> Self {
        Self {
            stage,
            domain: ERROR_DOMAIN_POSIX,
            code: io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32,
        }
    }

    #[cfg(windows)]
    pub(crate) const fn win32(stage: u16, code: u32) -> Self {
        Self {
            stage,
            domain: ERROR_DOMAIN_WIN32,
            code,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CleanupErrors {
    errors: [PublicationError; CLEANUP_ERROR_CAPACITY],
    len: u8,
}

impl CleanupErrors {
    pub(crate) const fn new() -> Self {
        Self {
            errors: [PublicationError::ZERO; CLEANUP_ERROR_CAPACITY],
            len: 0,
        }
    }

    pub(crate) fn push(&mut self, error: PublicationError) {
        let index = usize::from(self.len);
        debug_assert!(
            index < self.errors.len(),
            "cleanup error bound is structural"
        );
        if let Some(slot) = self.errors.get_mut(index) {
            *slot = error;
            self.len += 1;
        }
    }

    pub(crate) const fn tag(&self) -> u8 {
        if self.len == 0 {
            CLEANUP_CLEAN
        } else {
            CLEANUP_ORPHANED
        }
    }

    pub(crate) const fn len(&self) -> u8 {
        self.len
    }

    pub(crate) const fn errors(&self) -> &[PublicationError; CLEANUP_ERROR_CAPACITY] {
        &self.errors
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PublicationCompletion {
    pub(crate) outcome: u16,
    pub(crate) temp_cleanup: u8,
    pub(crate) primary_error: PublicationError,
    pub(crate) cleanup_errors: CleanupErrors,
    pub(crate) owned_temp_pid: u32,
    pub(crate) owned_temp_nonce: u64,
    pub(crate) payload_bytes: u64,
    // Best-effort temporary-file length, capped at payload_bytes. Zero
    // means no trustworthy length was observable; it never attests that a
    // partial write is canonical.
    pub(crate) written_bytes: u64,
}

impl PublicationCompletion {
    pub(crate) const fn terminal(
        outcome: u16,
        primary_error: PublicationError,
        cleanup_errors: CleanupErrors,
        owned_temp_pid: u32,
        owned_temp_nonce: u64,
        payload_bytes: u64,
        written_bytes: u64,
    ) -> Self {
        Self {
            outcome,
            temp_cleanup: cleanup_errors.tag(),
            primary_error,
            cleanup_errors,
            owned_temp_pid,
            owned_temp_nonce,
            payload_bytes,
            written_bytes,
        }
    }

    pub(crate) const fn without_temp(
        outcome: u16,
        error: PublicationError,
        payload_bytes: u64,
    ) -> Self {
        let mut completion =
            Self::terminal(outcome, error, CleanupErrors::new(), 0, 0, payload_bytes, 0);
        completion.temp_cleanup = CLEANUP_NOT_APPLICABLE;
        completion
    }

    pub(crate) fn write_failure(
        error: PublicationWriteError,
        cleanup_errors: CleanupErrors,
        pid: u32,
        nonce: u64,
        payload_bytes: u64,
        written_bytes: u64,
    ) -> Self {
        Self::terminal(
            OUTCOME_NOT_PUBLISHED,
            publication_write_error(&error),
            cleanup_errors,
            pid,
            nonce,
            payload_bytes,
            written_bytes,
        )
    }
}

pub(crate) enum PublicationWriteError {
    Checkpoint(CheckpointV1EncodeError),
    Bytes(io::Error),
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ImmutablePayloadInputError {
    Invalid,
    Allocation,
}

pub(crate) struct PreparedImmutableBytes {
    bytes: Vec<u8>,
}

impl PreparedImmutableBytes {
    fn payload_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn write_into(&self, file: &mut File) -> Result<(), PublicationWriteError> {
        file.write_all(&self.bytes)
            .and_then(|()| file.flush())
            .map_err(PublicationWriteError::Bytes)
    }

    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub(crate) fn prepare_immutable_bytes_v1(
    payload: &[u8],
) -> Result<PreparedImmutableBytes, ImmutablePayloadInputError> {
    if payload.is_empty() || payload.len() > MAX_IMMUTABLE_PAYLOAD_BYTES {
        return Err(ImmutablePayloadInputError::Invalid);
    }
    #[cfg(test)]
    if FAIL_NEXT_IMMUTABLE_PAYLOAD_ALLOCATION.with(|fail| fail.replace(false)) {
        return Err(ImmutablePayloadInputError::Allocation);
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(payload.len())
        .map_err(|_| ImmutablePayloadInputError::Allocation)?;
    bytes.extend_from_slice(payload);
    Ok(PreparedImmutableBytes { bytes })
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum DestinationInputError {
    Invalid,
    Allocation,
}

#[derive(Debug)]
pub(crate) struct ValidatedCheckpointDestination {
    parent_utf8: Vec<u8>,
    final_basename_utf8: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct ValidatedDurableDirectory {
    root_utf8: Vec<u8>,
    relative_directory_utf8: Vec<u8>,
    component_count: u32,
}

impl ValidatedDurableDirectory {
    pub(crate) fn root_utf8(&self) -> &[u8] {
        &self.root_utf8
    }

    pub(crate) fn relative_directory_utf8(&self) -> &[u8] {
        &self.relative_directory_utf8
    }

    pub(crate) const fn component_count(&self) -> u32 {
        self.component_count
    }
}

impl ValidatedCheckpointDestination {
    pub(crate) fn parent_utf8(&self) -> &[u8] {
        &self.parent_utf8
    }

    pub(crate) fn final_basename_utf8(&self) -> &[u8] {
        &self.final_basename_utf8
    }
}

pub(crate) fn validate_destination_v1(
    parent_utf8: &[u8],
    final_basename_utf8: &[u8],
) -> Result<ValidatedCheckpointDestination, DestinationInputError> {
    if parent_utf8.is_empty()
        || parent_utf8.len() > MAX_PATH_BYTES
        || final_basename_utf8.is_empty()
        || final_basename_utf8.len() > MAX_BASENAME_BYTES
        || parent_utf8.contains(&0)
        || final_basename_utf8.contains(&0)
        || final_basename_utf8 == b"."
        || final_basename_utf8 == b".."
        || final_basename_utf8.contains(&b'/')
        || final_basename_utf8.contains(&b'\\')
        || std::str::from_utf8(parent_utf8).is_err()
        || std::str::from_utf8(final_basename_utf8).is_err()
    {
        return Err(DestinationInputError::Invalid);
    }

    Ok(ValidatedCheckpointDestination {
        parent_utf8: try_copy_bytes(parent_utf8)?,
        final_basename_utf8: try_copy_bytes(final_basename_utf8)?,
    })
}

pub(crate) fn validate_durable_directory_v1(
    root_utf8: &[u8],
    relative_directory_utf8: &[u8],
) -> Result<ValidatedDurableDirectory, DestinationInputError> {
    if root_utf8.is_empty()
        || root_utf8.len() > MAX_PATH_BYTES
        || relative_directory_utf8.is_empty()
        || relative_directory_utf8.len() > MAX_PATH_BYTES
        || root_utf8.contains(&0)
        || relative_directory_utf8.contains(&0)
        || relative_directory_utf8.starts_with(b"/")
        || relative_directory_utf8.ends_with(b"/")
        || relative_directory_utf8.contains(&b'\\')
        || std::str::from_utf8(root_utf8).is_err()
        || std::str::from_utf8(relative_directory_utf8).is_err()
    {
        return Err(DestinationInputError::Invalid);
    }
    let mut component_count = 0u32;
    for component in relative_directory_utf8.split(|byte| *byte == b'/') {
        if component.is_empty()
            || component == b"."
            || component == b".."
            || component.len() > MAX_BASENAME_BYTES
        {
            return Err(DestinationInputError::Invalid);
        }
        component_count = component_count
            .checked_add(1)
            .ok_or(DestinationInputError::Invalid)?;
    }
    if component_count == 0 {
        return Err(DestinationInputError::Invalid);
    }
    Ok(ValidatedDurableDirectory {
        root_utf8: try_copy_bytes(root_utf8)?,
        relative_directory_utf8: try_copy_bytes(relative_directory_utf8)?,
        component_count,
    })
}

fn try_copy_bytes(source: &[u8]) -> Result<Vec<u8>, DestinationInputError> {
    #[cfg(test)]
    if FAIL_NEXT_DESTINATION_ALLOCATION.with(|fail| fail.replace(false)) {
        return Err(DestinationInputError::Allocation);
    }
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(source.len())
        .map_err(|_| DestinationInputError::Allocation)?;
    copied.extend_from_slice(source);
    Ok(copied)
}

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_DESTINATION_ALLOCATION: Cell<bool> = const { Cell::new(false) };
    static FAIL_NEXT_IMMUTABLE_PAYLOAD_ALLOCATION: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn fail_next_destination_allocation() {
    FAIL_NEXT_DESTINATION_ALLOCATION.with(|fail| fail.set(true));
}

#[cfg(test)]
pub(crate) fn fail_next_immutable_payload_allocation() {
    FAIL_NEXT_IMMUTABLE_PAYLOAD_ALLOCATION.with(|fail| fail.set(true));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TempName {
    bytes: [u8; TEMP_NAME_CAPACITY],
    len: u8,
    pub(crate) pid: u32,
    pub(crate) nonce: u64,
}

impl TempName {
    pub(crate) fn next(final_basename: &[u8], temp_prefix: &[u8]) -> Self {
        loop {
            let nonce = NEXT_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let mut bytes = [0; TEMP_NAME_CAPACITY];
            let mut cursor = 0;
            debug_assert!(temp_prefix.len() + 8 + 1 + 16 + TEMP_SUFFIX.len() <= bytes.len());
            bytes[..temp_prefix.len()].copy_from_slice(temp_prefix);
            cursor += temp_prefix.len();
            write_hex(&mut bytes[cursor..cursor + 8], u64::from(pid), 8);
            cursor += 8;
            bytes[cursor] = b'-';
            cursor += 1;
            write_hex(&mut bytes[cursor..cursor + 16], nonce, 16);
            cursor += 16;
            bytes[cursor..cursor + TEMP_SUFFIX.len()].copy_from_slice(TEMP_SUFFIX);
            cursor += TEMP_SUFFIX.len();
            let name = Self {
                bytes,
                len: cursor as u8,
                pid,
                nonce,
            };
            if name.as_bytes() != final_basename {
                return name;
            }
        }
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }

    pub(crate) const fn max_attempts() -> usize {
        MAX_TEMP_ATTEMPTS
    }
}

fn write_hex(output: &mut [u8], value: u64, digits: usize) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (index, byte) in output.iter_mut().enumerate().take(digits) {
        let shift = 4 * (digits - index - 1);
        *byte = HEX[((value >> shift) & 0xf) as usize];
    }
}

pub(crate) fn publish_prepared_checkpoint_v1(
    prepared: &PreparedCheckpointV1Export<'_>,
    destination: &ValidatedCheckpointDestination,
) -> PublicationCompletion {
    let checkpoint_bytes = prepared.encoded_len();
    #[cfg(unix)]
    {
        unix::publish(
            checkpoint_bytes,
            CHECKPOINT_TEMP_PREFIX,
            destination,
            |file| {
                prepared
                    .encode_into(file)
                    .map_err(PublicationWriteError::Checkpoint)
            },
        )
    }
    #[cfg(windows)]
    {
        windows::publish(
            checkpoint_bytes,
            CHECKPOINT_TEMP_PREFIX,
            destination,
            |file| {
                prepared
                    .encode_into(file)
                    .map_err(PublicationWriteError::Checkpoint)
            },
        )
    }
}

pub(crate) fn publish_prepared_immutable_bytes_v1(
    prepared: &PreparedImmutableBytes,
    destination: &ValidatedCheckpointDestination,
) -> PublicationCompletion {
    let payload_bytes = prepared.payload_len();
    #[cfg(unix)]
    {
        unix::publish(
            payload_bytes,
            IMMUTABLE_BYTES_TEMP_PREFIX,
            destination,
            |file| prepared.write_into(file),
        )
    }
    #[cfg(windows)]
    {
        windows::publish(
            payload_bytes,
            IMMUTABLE_BYTES_TEMP_PREFIX,
            destination,
            |file| prepared.write_into(file),
        )
    }
}

pub(crate) fn reconcile_existing_immutable_bytes_v1(
    prepared: &PreparedImmutableBytes,
    destination: &ValidatedCheckpointDestination,
) -> PathDurabilityCompletion {
    #[cfg(unix)]
    {
        unix::reconcile_existing_immutable_bytes(prepared.bytes(), destination)
    }
    #[cfg(windows)]
    {
        windows::reconcile_existing_immutable_bytes(prepared.bytes(), destination)
    }
}

pub(crate) fn prepare_durable_directory_v1(
    directory: &ValidatedDurableDirectory,
) -> PathDurabilityCompletion {
    #[cfg(unix)]
    {
        unix::prepare_durable_directory(directory)
    }
    #[cfg(windows)]
    {
        windows::prepare_durable_directory(directory)
    }
}

fn publication_write_error(error: &PublicationWriteError) -> PublicationError {
    match error {
        PublicationWriteError::Checkpoint(error) => encode_publication_error(error),
        PublicationWriteError::Bytes(source) => io_publication_error(source),
    }
}

fn io_publication_error(source: &io::Error) -> PublicationError {
    if let Some(raw) = source.raw_os_error() {
        return PublicationError {
            stage: ERROR_STAGE_ENCODE,
            domain: if cfg!(windows) {
                ERROR_DOMAIN_WIN32
            } else {
                ERROR_DOMAIN_POSIX
            },
            code: raw as u32,
        };
    }
    PublicationError {
        stage: ERROR_STAGE_ENCODE,
        domain: ERROR_DOMAIN_IO_WITHOUT_OS_CODE,
        code: io_error_kind_code(source.kind()),
    }
}

fn encode_publication_error(error: &CheckpointV1EncodeError) -> PublicationError {
    let source = match error {
        CheckpointV1EncodeError::Seek { source, .. }
        | CheckpointV1EncodeError::Write { source, .. }
        | CheckpointV1EncodeError::Flush { source } => Some(source),
        _ => None,
    };
    if let Some(source) = source {
        if let Some(raw) = source.raw_os_error() {
            return PublicationError {
                stage: ERROR_STAGE_ENCODE,
                domain: if cfg!(windows) {
                    ERROR_DOMAIN_WIN32
                } else {
                    ERROR_DOMAIN_POSIX
                },
                code: raw as u32,
            };
        }
        return PublicationError {
            stage: ERROR_STAGE_ENCODE,
            domain: ERROR_DOMAIN_IO_WITHOUT_OS_CODE,
            code: io_error_kind_code(source.kind()),
        };
    }
    PublicationError {
        stage: ERROR_STAGE_ENCODE,
        domain: ERROR_DOMAIN_ENCODER,
        code: encoder_error_code(error),
    }
}

fn io_error_kind_code(kind: io::ErrorKind) -> u32 {
    match kind {
        io::ErrorKind::NotFound => 1,
        io::ErrorKind::PermissionDenied => 2,
        io::ErrorKind::AlreadyExists => 3,
        io::ErrorKind::InvalidInput => 4,
        io::ErrorKind::InvalidData => 5,
        io::ErrorKind::WriteZero => 6,
        io::ErrorKind::Interrupted => 7,
        io::ErrorKind::UnexpectedEof => 8,
        io::ErrorKind::OutOfMemory => 9,
        _ => 255,
    }
}

fn encoder_error_code(error: &CheckpointV1EncodeError) -> u32 {
    match error {
        CheckpointV1EncodeError::PreparedCodec(_) => 1,
        CheckpointV1EncodeError::PreparedScalarsMismatch => 2,
        CheckpointV1EncodeError::PreparedPairCountMismatch { .. } => 3,
        CheckpointV1EncodeError::BodySectionCount { .. } => 4,
        CheckpointV1EncodeError::BodyByteCount { .. } => 5,
        CheckpointV1EncodeError::TotalByteCount { .. } => 6,
        CheckpointV1EncodeError::BodyBufferInvariant(_) => 7,
        CheckpointV1EncodeError::SinkNotEmpty { .. } => 8,
        CheckpointV1EncodeError::SeekPosition { .. } => 9,
        CheckpointV1EncodeError::WriteCount { .. } => 10,
        CheckpointV1EncodeError::HeaderCodec(_) => 11,
        CheckpointV1EncodeError::Seek { .. }
        | CheckpointV1EncodeError::Write { .. }
        | CheckpointV1EncodeError::Flush { .. } => 12,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destination_validation_is_strict_and_owned() {
        let parent = b"C:\\checkpoint".to_vec();
        let basename = b"state.chk".to_vec();
        let validated = validate_destination_v1(&parent, &basename).unwrap();
        assert_eq!(validated.parent_utf8(), parent);
        assert_eq!(validated.final_basename_utf8(), basename);
        assert!(matches!(
            validate_destination_v1(b"", basename.as_slice()),
            Err(DestinationInputError::Invalid)
        ));
        assert!(matches!(
            validate_destination_v1(&parent, b"../state.chk"),
            Err(DestinationInputError::Invalid)
        ));
        assert!(matches!(
            validate_destination_v1(&parent, b"bad\0name"),
            Err(DestinationInputError::Invalid)
        ));
    }

    #[test]
    fn generated_temp_names_are_fixed_ascii_unique_and_not_the_destination() {
        let first = TempName::next(b"state.chk", CHECKPOINT_TEMP_PREFIX);
        let second = TempName::next(first.as_bytes(), CHECKPOINT_TEMP_PREFIX);
        assert_ne!(first.as_bytes(), second.as_bytes());
        assert!(first.as_bytes().starts_with(CHECKPOINT_TEMP_PREFIX));
        assert!(first.as_bytes().ends_with(TEMP_SUFFIX));
        assert_eq!(
            first.as_bytes().len(),
            CHECKPOINT_TEMP_PREFIX.len() + 8 + 1 + 16 + 4
        );
        assert!(first.as_bytes().iter().all(u8::is_ascii));
    }

    #[test]
    fn immutable_payload_is_bounded_and_owned_before_publication() {
        assert!(matches!(
            prepare_immutable_bytes_v1(b""),
            Err(ImmutablePayloadInputError::Invalid)
        ));
        assert!(prepare_immutable_bytes_v1(&vec![0; MAX_IMMUTABLE_PAYLOAD_BYTES]).is_ok());
        assert!(matches!(
            prepare_immutable_bytes_v1(&vec![0; MAX_IMMUTABLE_PAYLOAD_BYTES + 1]),
            Err(ImmutablePayloadInputError::Invalid)
        ));

        let mut foreign = b"original\0\xff".to_vec();
        let prepared = prepare_immutable_bytes_v1(&foreign).unwrap();
        foreign.fill(b'x');
        assert_eq!(prepared.bytes, b"original\0\xff");
    }

    #[test]
    fn byte_writer_errors_use_the_existing_write_stage_taxonomy() {
        let error = io_publication_error(&io::Error::from(io::ErrorKind::WriteZero));
        assert_eq!(error.stage, ERROR_STAGE_ENCODE);
        assert_eq!(error.domain, ERROR_DOMAIN_IO_WITHOUT_OS_CODE);
        assert_eq!(error.code, 6);
    }

    #[test]
    fn cleanup_errors_preserve_order_and_tag() {
        let mut errors = CleanupErrors::new();
        assert_eq!(errors.tag(), CLEANUP_CLEAN);
        errors.push(PublicationError::policy(ERROR_STAGE_FILE_CLOSE, 10));
        errors.push(PublicationError::policy(ERROR_STAGE_TEMP_CLEANUP, 11));
        errors.push(PublicationError::policy(ERROR_STAGE_PARENT_SYNC, 12));
        assert_eq!(errors.tag(), CLEANUP_ORPHANED);
        assert_eq!(errors.len(), 3);
        assert_eq!(errors.errors()[0].code, 10);
        assert_eq!(errors.errors()[1].code, 11);
        assert_eq!(errors.errors()[2].code, 12);
    }

    #[test]
    fn no_temp_completion_is_not_applicable_not_clean() {
        let completion = PublicationCompletion::without_temp(
            OUTCOME_NOT_PUBLISHED,
            PublicationError::policy(ERROR_STAGE_PARENT_OPEN, POLICY_INVALID_PARENT),
            256,
        );
        assert_eq!(completion.temp_cleanup, CLEANUP_NOT_APPLICABLE);
        assert_eq!(completion.owned_temp_pid, 0);
        assert_eq!(completion.owned_temp_nonce, 0);
    }
}
