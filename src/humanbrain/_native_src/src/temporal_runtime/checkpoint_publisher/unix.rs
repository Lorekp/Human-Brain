use std::{
    fs::File,
    mem::MaybeUninit,
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, RawFd},
};

use super::{
    CleanupErrors, PublicationCompletion, PublicationError, PublicationWriteError, TempName,
    ValidatedCheckpointDestination, ERROR_STAGE_EXCLUSIVE_PUBLISH, ERROR_STAGE_FILESYSTEM_POLICY,
    ERROR_STAGE_FILE_CLOSE, ERROR_STAGE_FILE_SYNC, ERROR_STAGE_PARENT_IDENTITY,
    ERROR_STAGE_PARENT_OPEN, ERROR_STAGE_PARENT_SYNC, ERROR_STAGE_TEMP_CLEANUP,
    ERROR_STAGE_TEMP_CREATE, ERROR_STAGE_TEMP_IDENTITY, OUTCOME_COLLISION,
    OUTCOME_DURABLE_PUBLISHED, OUTCOME_NOT_PUBLISHED, OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
    POLICY_INVALID_PARENT, POLICY_TEMP_IDENTITY_CHANGED, POLICY_TEMP_NAME_EXHAUSTED,
    POLICY_UNSUPPORTED_FILESYSTEM,
};

const EXT4_SUPER_MAGIC: u64 = 0x0000_0000_0000_ef53;
const XFS_SUPER_MAGIC: u64 = 0x0000_0000_5846_5342;

#[derive(Debug)]
struct OwnedFd {
    raw: RawFd,
}

impl OwnedFd {
    fn new(raw: RawFd) -> Self {
        Self { raw }
    }

    fn take(&mut self) -> RawFd {
        let raw = self.raw;
        self.raw = -1;
        raw
    }
}

impl Drop for OwnedFd {
    fn drop(&mut self) {
        if self.raw >= 0 {
            // There is no remaining semantic operation whose outcome could be
            // changed by closing a read-only parent descriptor.
            unsafe {
                libc::close(self.raw);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: libc::dev_t,
    inode: libc::ino_t,
}

struct CName {
    bytes: [libc::c_char; 256],
}

impl CName {
    fn new(value: &[u8]) -> Option<Self> {
        if value.is_empty() || value.len() >= 256 || value.contains(&0) {
            return None;
        }
        let mut bytes = [0; 256];
        for (destination, source) in bytes.iter_mut().zip(value) {
            *destination = *source as libc::c_char;
        }
        Some(Self { bytes })
    }

    fn as_ptr(&self) -> *const libc::c_char {
        self.bytes.as_ptr()
    }
}

pub(super) fn publish<F>(
    checkpoint_bytes: u64,
    temp_prefix: &[u8],
    destination: &ValidatedCheckpointDestination,
    write_payload: F,
) -> PublicationCompletion
where
    F: FnOnce(&mut File) -> Result<(), PublicationWriteError>,
{
    let parent = match open_stable_parent(destination.parent_utf8()) {
        Ok(parent) => parent,
        Err(error) => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                error,
                checkpoint_bytes,
            )
        }
    };
    if let Err(error) = validate_local_filesystem(parent.raw) {
        return PublicationCompletion::without_temp(OUTCOME_NOT_PUBLISHED, error, checkpoint_bytes);
    }
    if unsafe { libc::fsync(parent.raw) } != 0 {
        return PublicationCompletion::without_temp(
            OUTCOME_NOT_PUBLISHED,
            PublicationError::last_posix(ERROR_STAGE_PARENT_SYNC),
            checkpoint_bytes,
        );
    }

    let final_name = match CName::new(destination.final_basename_utf8()) {
        Some(name) => name,
        None => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                PublicationError::policy(ERROR_STAGE_PARENT_IDENTITY, POLICY_INVALID_PARENT),
                checkpoint_bytes,
            )
        }
    };
    let name_max = unsafe { libc::fpathconf(parent.raw, libc::_PC_NAME_MAX) };
    if name_max < 0 || destination.final_basename_utf8().len() > name_max as usize {
        return PublicationCompletion::without_temp(
            OUTCOME_NOT_PUBLISHED,
            PublicationError::policy(ERROR_STAGE_FILESYSTEM_POLICY, POLICY_UNSUPPORTED_FILESYSTEM),
            checkpoint_bytes,
        );
    }

    let (mut temp_fd, temp_name, temp_c_name) = match create_owned_temp(
        parent.raw,
        destination.final_basename_utf8(),
        temp_prefix,
        name_max as usize,
    ) {
        Ok(created) => created,
        Err(error) => {
            return PublicationCompletion::without_temp(
                OUTCOME_NOT_PUBLISHED,
                error,
                checkpoint_bytes,
            )
        }
    };
    let identity = match file_identity(temp_fd.raw) {
        Ok(identity) => identity,
        Err(primary) => {
            let mut cleanup = CleanupErrors::new();
            // Ownership may not be inferred from a generated name alone.  If
            // the opened handle cannot establish identity, leave the name for
            // operator reconciliation instead of risking deletion of a
            // replacement object.
            cleanup.push(primary);
            close_fd_for_cleanup(&mut temp_fd, &mut cleanup);
            if let Some(sync_error) = sync_parent(parent.raw) {
                cleanup.push(sync_error);
            }
            return PublicationCompletion::terminal(
                OUTCOME_NOT_PUBLISHED,
                primary,
                cleanup,
                temp_name.pid,
                temp_name.nonce,
                checkpoint_bytes,
                0,
            );
        }
    };

    let raw = temp_fd.take();
    let mut file = unsafe { File::from_raw_fd(raw) };
    if let Err(error) = write_payload(&mut file) {
        let encoded_bytes = observed_file_length(&file, checkpoint_bytes);
        let mut cleanup = CleanupErrors::new();
        close_file_for_cleanup(file, &mut cleanup);
        if let Some(sync_error) =
            cleanup_owned_temp(parent.raw, &temp_c_name, identity, &mut cleanup)
        {
            cleanup.push(sync_error);
        }
        return PublicationCompletion::write_failure(
            error,
            cleanup,
            temp_name.pid,
            temp_name.nonce,
            checkpoint_bytes,
            encoded_bytes,
        );
    }
    if unsafe { libc::fsync(file.as_raw_fd()) } != 0 {
        let primary = PublicationError::last_posix(ERROR_STAGE_FILE_SYNC);
        let mut cleanup = CleanupErrors::new();
        close_file_for_cleanup(file, &mut cleanup);
        if let Some(sync_error) =
            cleanup_owned_temp(parent.raw, &temp_c_name, identity, &mut cleanup)
        {
            cleanup.push(sync_error);
        }
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
        if let Some(sync_error) =
            cleanup_owned_temp(parent.raw, &temp_c_name, identity, &mut cleanup)
        {
            cleanup.push(sync_error);
        }
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
    if let Err(primary) = verify_named_identity(parent.raw, &temp_c_name, identity) {
        let mut cleanup = CleanupErrors::new();
        cleanup.push(primary);
        if let Some(sync_error) = sync_parent(parent.raw) {
            cleanup.push(sync_error);
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

    let link_result = unsafe {
        libc::linkat(
            parent.raw,
            temp_c_name.as_ptr(),
            parent.raw,
            final_name.as_ptr(),
            0,
        )
    };
    let link_error = if link_result == 0 {
        None
    } else {
        Some(PublicationError::last_posix(ERROR_STAGE_EXCLUSIVE_PUBLISH))
    };
    let mut cleanup = CleanupErrors::new();
    let parent_sync_error = cleanup_owned_temp(parent.raw, &temp_c_name, identity, &mut cleanup);

    let (outcome, primary, cleanup_sync_error) =
        classify_link_attempt(link_error, parent_sync_error);
    if let Some(sync_error) = cleanup_sync_error {
        cleanup.push(sync_error);
    }
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

fn classify_link_attempt(
    link_error: Option<PublicationError>,
    parent_sync_error: Option<PublicationError>,
) -> (u16, PublicationError, Option<PublicationError>) {
    match (link_error, parent_sync_error) {
        (None, None) => (OUTCOME_DURABLE_PUBLISHED, PublicationError::ZERO, None),
        (None, Some(sync_error)) => (
            OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
            sync_error,
            None,
        ),
        (Some(link_error), sync_error) if link_error.code == libc::EEXIST as u32 => {
            (OUTCOME_COLLISION, PublicationError::ZERO, sync_error)
        }
        (Some(link_error), sync_error) => (OUTCOME_NOT_PUBLISHED, link_error, sync_error),
    }
}

fn open_stable_parent(parent_utf8: &[u8]) -> Result<OwnedFd, PublicationError> {
    if !parent_utf8.starts_with(b"/") || parent_utf8.len() > 1 && parent_utf8.ends_with(b"/") {
        return Err(PublicationError::policy(
            ERROR_STAGE_PARENT_OPEN,
            POLICY_INVALID_PARENT,
        ));
    }
    let root = unsafe {
        libc::open(
            c"/".as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    };
    if root < 0 {
        return Err(PublicationError::last_posix(ERROR_STAGE_PARENT_OPEN));
    }
    let mut current = OwnedFd::new(root);
    for component in parent_utf8[1..]
        .split(|byte| *byte == b'/')
        .filter(|component| !component.is_empty() || parent_utf8.len() > 1)
    {
        if component.is_empty() || component == b"." || component == b".." {
            return Err(PublicationError::policy(
                ERROR_STAGE_PARENT_OPEN,
                POLICY_INVALID_PARENT,
            ));
        }
        let name = CName::new(component).ok_or_else(|| {
            PublicationError::policy(ERROR_STAGE_PARENT_OPEN, POLICY_INVALID_PARENT)
        })?;
        let next = unsafe {
            libc::openat(
                current.raw,
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if next < 0 {
            return Err(PublicationError::last_posix(ERROR_STAGE_PARENT_OPEN));
        }
        current = OwnedFd::new(next);
    }
    let mut metadata = MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(current.raw, metadata.as_mut_ptr()) } != 0 {
        return Err(PublicationError::last_posix(ERROR_STAGE_PARENT_IDENTITY));
    }
    let metadata = unsafe { metadata.assume_init() };
    if metadata.st_mode & libc::S_IFMT != libc::S_IFDIR {
        return Err(PublicationError::policy(
            ERROR_STAGE_PARENT_IDENTITY,
            POLICY_INVALID_PARENT,
        ));
    }
    Ok(current)
}

fn validate_local_filesystem(parent_fd: RawFd) -> Result<(), PublicationError> {
    #[cfg(target_os = "linux")]
    {
        let mut metadata = MaybeUninit::<libc::statfs>::uninit();
        if unsafe { libc::fstatfs(parent_fd, metadata.as_mut_ptr()) } != 0 {
            return Err(PublicationError::last_posix(ERROR_STAGE_FILESYSTEM_POLICY));
        }
        let kind = unsafe { metadata.assume_init() }.f_type as u64;
        if kind == EXT4_SUPER_MAGIC || kind == XFS_SUPER_MAGIC {
            return Ok(());
        }
        return Err(PublicationError::policy(
            ERROR_STAGE_FILESYSTEM_POLICY,
            POLICY_UNSUPPORTED_FILESYSTEM,
        ));
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = parent_fd;
        Err(PublicationError::policy(
            ERROR_STAGE_FILESYSTEM_POLICY,
            POLICY_REMOTE_OR_UNKNOWN_FILESYSTEM,
        ))
    }
}

fn create_owned_temp(
    parent_fd: RawFd,
    final_basename: &[u8],
    temp_prefix: &[u8],
    name_max: usize,
) -> Result<(OwnedFd, TempName, CName), PublicationError> {
    for _ in 0..TempName::max_attempts() {
        let temp_name = TempName::next(final_basename, temp_prefix);
        if temp_name.as_bytes().len() > name_max {
            return Err(PublicationError::policy(
                ERROR_STAGE_TEMP_CREATE,
                POLICY_UNSUPPORTED_FILESYSTEM,
            ));
        }
        let Some(c_name) = CName::new(temp_name.as_bytes()) else {
            return Err(PublicationError::policy(
                ERROR_STAGE_TEMP_CREATE,
                POLICY_TEMP_NAME_EXHAUSTED,
            ));
        };
        let raw = unsafe {
            libc::openat(
                parent_fd,
                c_name.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if raw >= 0 {
            return Ok((OwnedFd::new(raw), temp_name, c_name));
        }
        let error = PublicationError::last_posix(ERROR_STAGE_TEMP_CREATE);
        if error.code != libc::EEXIST as u32 {
            return Err(error);
        }
    }
    Err(PublicationError::policy(
        ERROR_STAGE_TEMP_CREATE,
        POLICY_TEMP_NAME_EXHAUSTED,
    ))
}

fn file_identity(raw: RawFd) -> Result<FileIdentity, PublicationError> {
    let mut metadata = MaybeUninit::<libc::stat>::uninit();
    if unsafe { libc::fstat(raw, metadata.as_mut_ptr()) } != 0 {
        return Err(PublicationError::last_posix(ERROR_STAGE_TEMP_IDENTITY));
    }
    let metadata = unsafe { metadata.assume_init() };
    if metadata.st_mode & libc::S_IFMT != libc::S_IFREG {
        return Err(PublicationError::policy(
            ERROR_STAGE_TEMP_IDENTITY,
            POLICY_TEMP_IDENTITY_CHANGED,
        ));
    }
    Ok(FileIdentity {
        device: metadata.st_dev,
        inode: metadata.st_ino,
    })
}

fn verify_named_identity(
    parent_fd: RawFd,
    name: &CName,
    expected: FileIdentity,
) -> Result<(), PublicationError> {
    let mut metadata = MaybeUninit::<libc::stat>::uninit();
    if unsafe {
        libc::fstatat(
            parent_fd,
            name.as_ptr(),
            metadata.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(PublicationError::last_posix(ERROR_STAGE_TEMP_IDENTITY));
    }
    let metadata = unsafe { metadata.assume_init() };
    if metadata.st_dev != expected.device || metadata.st_ino != expected.inode {
        return Err(PublicationError::policy(
            ERROR_STAGE_TEMP_IDENTITY,
            POLICY_TEMP_IDENTITY_CHANGED,
        ));
    }
    Ok(())
}

fn close_file_checked(file: File) -> Result<(), PublicationError> {
    let raw = file.into_raw_fd();
    if unsafe { libc::close(raw) } == 0 {
        Ok(())
    } else {
        Err(PublicationError::last_posix(ERROR_STAGE_FILE_CLOSE))
    }
}

fn close_file_for_cleanup(file: File, cleanup: &mut CleanupErrors) {
    if let Err(error) = close_file_checked(file) {
        cleanup.push(error);
    }
}

fn close_fd_for_cleanup(fd: &mut OwnedFd, cleanup: &mut CleanupErrors) {
    let raw = fd.take();
    if raw >= 0 && unsafe { libc::close(raw) } != 0 {
        cleanup.push(PublicationError::last_posix(ERROR_STAGE_FILE_CLOSE));
    }
}

fn cleanup_owned_temp(
    parent_fd: RawFd,
    name: &CName,
    expected: FileIdentity,
    cleanup: &mut CleanupErrors,
) -> Option<PublicationError> {
    let may_unlink = match verify_named_identity(parent_fd, name, expected) {
        Ok(()) => true,
        Err(error)
            if error.domain == super::ERROR_DOMAIN_POSIX && error.code == libc::ENOENT as u32 =>
        {
            false
        }
        Err(error) => {
            cleanup.push(error);
            false
        }
    };
    if may_unlink && unsafe { libc::unlinkat(parent_fd, name.as_ptr(), 0) } != 0 {
        cleanup.push(PublicationError::last_posix(ERROR_STAGE_TEMP_CLEANUP));
    }
    sync_parent(parent_fd)
}

fn sync_parent(parent_fd: RawFd) -> Option<PublicationError> {
    (unsafe { libc::fsync(parent_fd) } != 0)
        .then(|| PublicationError::last_posix(ERROR_STAGE_PARENT_SYNC))
}

fn observed_file_length(file: &File, checkpoint_bytes: u64) -> u64 {
    file.metadata()
        .map(|metadata| metadata.len().min(checkpoint_bytes))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_name_rejects_non_basename_geometry() {
        assert!(CName::new(b"").is_none());
        assert!(CName::new(&[b'x'; 255]).is_some());
        assert!(CName::new(&[b'x'; 256]).is_none());
        assert!(CName::new(b"bad\0name").is_none());
    }

    #[test]
    fn link_and_parent_sync_matrix_preserves_primary_vs_cleanup_errors() {
        let link_collision = PublicationError {
            stage: ERROR_STAGE_EXCLUSIVE_PUBLISH,
            domain: super::super::ERROR_DOMAIN_POSIX,
            code: libc::EEXIST as u32,
        };
        let link_other = PublicationError {
            code: libc::EPERM as u32,
            ..link_collision
        };
        let sync = PublicationError {
            stage: ERROR_STAGE_PARENT_SYNC,
            domain: super::super::ERROR_DOMAIN_POSIX,
            code: libc::EIO as u32,
        };
        assert_eq!(
            classify_link_attempt(None, None),
            (OUTCOME_DURABLE_PUBLISHED, PublicationError::ZERO, None)
        );
        assert_eq!(
            classify_link_attempt(None, Some(sync)),
            (OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE, sync, None)
        );
        assert_eq!(
            classify_link_attempt(Some(link_collision), Some(sync)),
            (OUTCOME_COLLISION, PublicationError::ZERO, Some(sync))
        );
        assert_eq!(
            classify_link_attempt(Some(link_other), Some(sync)),
            (OUTCOME_NOT_PUBLISHED, link_other, Some(sync))
        );
    }
}
