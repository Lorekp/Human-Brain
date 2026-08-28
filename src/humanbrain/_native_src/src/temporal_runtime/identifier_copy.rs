//! Transient physical role/place binding for one bounded identifier.
//!
//! The tape stores no token or name identity.  Each occupied place owns eleven
//! sparse binding segments whose owners are the eleven content columns that
//! fired while that place assembly was active.  Recall therefore reconstructs
//! a distributed column assembly and lets the ordinary fitted output-cell race
//! decide which token it denotes.

use crate::random123::sample_without_replacement;

use super::{TokenCode, ACTIVE_COLUMNS, NONE, PAIRS_PER_SEGMENT};

pub(crate) const IDENTIFIER_START_TOKEN_ID: u32 = u32::MAX - 4;
pub(crate) const IDENTIFIER_END_TOKEN_ID: u32 = u32::MAX - 3;
pub(crate) const COPY_ID_TOKEN_ID: u32 = u32::MAX - 2;
pub(crate) const COPY_END_TOKEN_ID: u32 = u32::MAX - 1;

pub(crate) const MAX_IDENTIFIER_TOKENS: usize = 8;
const BINDING_ROWS: usize = MAX_IDENTIFIER_TOKENS + 1;
const PLACE_CELL_COUNT: usize = 128;
const ROLE_CELL_COUNT: usize = 64;
const PLACE_DOMAIN: u64 = 0x5233_504C_4143_4531;
const ROLE_DOMAIN: u64 = 0x5233_524F_4C45_5331;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CopyRole {
    Idle,
    Capture,
    Ready,
    Recall,
}

#[derive(Clone, Copy)]
struct BindingRow {
    segment_sources: [u32; PAIRS_PER_SEGMENT],
    owner_columns: [u32; ACTIVE_COLUMNS],
    live: bool,
}

const EMPTY_BINDING: BindingRow = BindingRow {
    segment_sources: [NONE; PAIRS_PER_SEGMENT],
    owner_columns: [NONE; ACTIVE_COLUMNS],
    live: false,
};

pub(crate) struct IdentifierCopyState {
    enabled: bool,
    construction_seed: u32,
    copy_column_offset: u32,
    place_offset: u32,
    role_offset: u32,
    role: CopyRole,
    role_cells: [u32; ACTIVE_COLUMNS],
    place_cells: [u32; ACTIVE_COLUMNS],
    bindings: [BindingRow; BINDING_ROWS],
    identifier_len: u8,
    recall_position: u8,
    copy_end_columns: [u32; ACTIVE_COLUMNS],
}

impl IdentifierCopyState {
    pub(crate) fn new(
        construction_seed: u32,
        column_count: u32,
        base_physical_cell_count: u32,
        token_codebook: &[TokenCode],
    ) -> Result<Self, &'static str> {
        let role_ids = [
            IDENTIFIER_START_TOKEN_ID,
            IDENTIFIER_END_TOKEN_ID,
            COPY_ID_TOKEN_ID,
            COPY_END_TOKEN_ID,
        ];
        let role_count = role_ids
            .iter()
            .filter(|&&token_id| token_code(token_codebook, token_id).is_some())
            .count();
        if role_count != 0 && role_count != role_ids.len() {
            return Err("identifier-copy role tokens must be all present or all absent");
        }
        let enabled = role_count == role_ids.len();
        let token_count = u32::try_from(token_codebook.len())
            .map_err(|_| "identifier-copy token population exceeds uint32")?;
        let output_basket = base_physical_cell_count
            .checked_add(token_count)
            .ok_or("identifier-copy physical offsets overflow uint32")?;
        let copy_column_offset = output_basket
            .checked_add(1)
            .ok_or("identifier-copy physical offsets overflow uint32")?;
        let place_offset = copy_column_offset
            .checked_add(column_count)
            .ok_or("identifier-copy physical offsets overflow uint32")?;
        let role_offset = place_offset
            .checked_add(PLACE_CELL_COUNT as u32)
            .ok_or("identifier-copy physical offsets overflow uint32")?;
        role_offset
            .checked_add(ROLE_CELL_COUNT as u32)
            .ok_or("identifier-copy physical offsets overflow uint32")?;

        let copy_end_columns = if enabled {
            token_code(token_codebook, COPY_END_TOKEN_ID)
                .expect("all role codes were counted")
                .columns
        } else {
            [NONE; ACTIVE_COLUMNS]
        };
        Ok(Self {
            enabled,
            construction_seed,
            copy_column_offset,
            place_offset,
            role_offset,
            role: CopyRole::Idle,
            role_cells: [NONE; ACTIVE_COLUMNS],
            place_cells: [NONE; ACTIVE_COLUMNS],
            bindings: [EMPTY_BINDING; BINDING_ROWS],
            identifier_len: 0,
            recall_position: 0,
            copy_end_columns,
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn reset(&mut self) {
        self.role = CopyRole::Idle;
        self.role_cells = [NONE; ACTIVE_COLUMNS];
        self.place_cells = [NONE; ACTIVE_COLUMNS];
        self.bindings = [EMPTY_BINDING; BINDING_ROWS];
        self.identifier_len = 0;
        self.recall_position = 0;
    }

    pub(crate) fn is_capturing(&self) -> bool {
        self.enabled && self.role == CopyRole::Capture
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.enabled && self.role == CopyRole::Ready
    }

    pub(crate) fn is_recalling(&self) -> bool {
        self.enabled && self.role == CopyRole::Recall
    }

    pub(crate) fn preflight_arrival(&self, token_id: u32) -> Result<(), &'static str> {
        if !self.enabled {
            return Ok(());
        }
        if token_id == IDENTIFIER_START_TOKEN_ID {
            return if self.role == CopyRole::Capture || self.role == CopyRole::Recall {
                Err("identifier start arrived while a copy role was active")
            } else {
                Ok(())
            };
        }
        if token_id == IDENTIFIER_END_TOKEN_ID {
            return if self.role != CopyRole::Capture {
                Err("identifier end arrived without an active capture role")
            } else if self.identifier_len == 0 {
                Err("identifier capture cannot be empty")
            } else {
                Ok(())
            };
        }
        if self.role == CopyRole::Capture {
            if token_id == COPY_ID_TOKEN_ID || token_id == COPY_END_TOKEN_ID {
                return Err("copy control token arrived inside identifier capture");
            }
            if usize::from(self.identifier_len) >= MAX_IDENTIFIER_TOKENS {
                return Err("identifier exceeds the bounded copy tape");
            }
        }
        Ok(())
    }

    pub(crate) fn observe_arrival(
        &mut self,
        token_id: u32,
        columns: [u32; ACTIVE_COLUMNS],
    ) {
        if !self.enabled {
            return;
        }
        if token_id == IDENTIFIER_START_TOKEN_ID {
            self.reset();
            self.role = CopyRole::Capture;
            self.activate_role(0);
            self.activate_place(0);
            return;
        }
        if token_id == IDENTIFIER_END_TOKEN_ID {
            debug_assert_eq!(self.role, CopyRole::Capture);
            let end_position = usize::from(self.identifier_len);
            self.bindings[end_position] = self.binding_row(self.copy_end_columns);
            self.role = CopyRole::Ready;
            self.activate_role(1);
            self.place_cells = [NONE; ACTIVE_COLUMNS];
            return;
        }
        if self.role == CopyRole::Capture {
            let position = usize::from(self.identifier_len);
            self.bindings[position] = self.binding_row(columns);
            self.identifier_len += 1;
            self.activate_place(usize::from(self.identifier_len));
        }
    }

    pub(crate) fn begin_recall(&mut self) {
        debug_assert!(self.is_ready());
        self.role = CopyRole::Recall;
        self.recall_position = 0;
        self.activate_role(2);
        self.activate_place(0);
    }

    pub(crate) fn binding_columns_at(
        &self,
        position: u8,
    ) -> Result<[u32; ACTIVE_COLUMNS], &'static str> {
        if !(self.is_recalling() || (self.is_ready() && position == 0)) {
            return Err("copy binding was requested outside a recall-capable role");
        }
        let place_cells = sparse_code(
            self.construction_seed,
            PLACE_DOMAIN,
            u64::from(position),
            self.place_offset,
            PLACE_CELL_COUNT,
        );
        let Some(row) = self.bindings.get(usize::from(position)).copied() else {
            return Err("copy place escaped its bounded population");
        };
        if !row.live {
            return Err("copy place has no live binding row");
        }
        let connected_pairs = row
            .segment_sources
            .iter()
            .filter(|source| place_cells.contains(source))
            .count();
        if connected_pairs < 9 {
            return Err("copy place failed the physical active threshold");
        }
        Ok(row.owner_columns)
    }

    pub(crate) fn advance_recall(&mut self) {
        debug_assert!(self.is_recalling() && self.recall_position < self.identifier_len);
        self.recall_position += 1;
        self.activate_place(usize::from(self.recall_position));
    }

    pub(crate) fn finish_recall(&mut self) {
        debug_assert!(self.is_recalling() && self.recall_position == self.identifier_len);
        self.role = CopyRole::Idle;
        self.role_cells = [NONE; ACTIVE_COLUMNS];
        self.place_cells = [NONE; ACTIVE_COLUMNS];
    }

    pub(crate) fn recall_position(&self) -> u8 {
        self.recall_position
    }

    pub(crate) fn recall_is_at_end(&self) -> bool {
        self.is_recalling() && self.recall_position == self.identifier_len
    }

    pub(crate) fn copy_column_cell(&self, column: u32) -> Result<u32, &'static str> {
        self.copy_column_offset
            .checked_add(column)
            .ok_or("copy-column physical identity overflowed uint32")
    }

    fn binding_row(&self, owner_columns: [u32; ACTIVE_COLUMNS]) -> BindingRow {
        let mut sources = [NONE; PAIRS_PER_SEGMENT];
        sources.copy_from_slice(&self.place_cells[..PAIRS_PER_SEGMENT]);
        sources.sort_unstable();
        BindingRow {
            segment_sources: sources,
            owner_columns,
            live: true,
        }
    }

    fn activate_place(&mut self, position: usize) {
        self.place_cells = sparse_code(
            self.construction_seed,
            PLACE_DOMAIN,
            position as u64,
            self.place_offset,
            PLACE_CELL_COUNT,
        );
    }

    fn activate_role(&mut self, role: u64) {
        self.role_cells = sparse_code(
            self.construction_seed,
            ROLE_DOMAIN,
            role,
            self.role_offset,
            ROLE_CELL_COUNT,
        );
    }
}

fn token_code(codebook: &[TokenCode], token_id: u32) -> Option<&TokenCode> {
    codebook
        .binary_search_by_key(&token_id, |code| code.token_id)
        .ok()
        .map(|index| &codebook[index])
}

fn sparse_code(
    construction_seed: u32,
    domain: u64,
    item: u64,
    gid_offset: u32,
    population: usize,
) -> [u32; ACTIVE_COLUMNS] {
    let mut scratch = [0u64; PLACE_CELL_COUNT];
    let selected = sample_without_replacement(
        population,
        ACTIVE_COLUMNS,
        u64::from(construction_seed) ^ domain,
        item,
        &mut scratch,
    )
    .expect("fixed role/place population can sample eleven cells");
    let mut code = std::array::from_fn(|index| gid_offset + selected[index] as u32);
    code.sort_unstable();
    code
}
