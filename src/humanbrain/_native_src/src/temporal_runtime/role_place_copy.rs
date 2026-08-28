//! Transient physical role/place binding for a bounded ordinal slot bank.
//!
//! A slot stores neither token IDs nor decoded strings. Each occupied place
//! owns eleven distributed content columns behind a ten-contact binding row.
//! Five contacts come from a slot-role assembly and five from a shared place
//! assembly, so equal positions overlap physically while ordinal slot identity
//! remains physical. Recall reconstructs the content assembly and lets the
//! fitted output-cell race choose every surfaced token.

use crate::random123::sample_without_replacement;

use super::{TokenCode, ACTIVE_COLUMNS, NONE, PAIRS_PER_SEGMENT};

pub(crate) const SLOT_COUNT: usize = 3;
pub(crate) const MAX_SLOT_TOKENS: usize = 22;
pub(crate) const SLOT_START_TOKEN_BASE: u32 = u32::MAX - 10;
pub(crate) const SLOT_END_TOKEN_BASE: u32 = u32::MAX - 7;
pub(crate) const COPY_SLOT_TOKEN_BASE: u32 = u32::MAX - 4;
pub(crate) const COPY_END_TOKEN_ID: u32 = u32::MAX - 1;

const BINDING_ROWS: usize = MAX_SLOT_TOKENS + 1;
const ROLE_CONTACTS: usize = PAIRS_PER_SEGMENT / 2;
const PLACE_CONTACTS: usize = PAIRS_PER_SEGMENT - ROLE_CONTACTS;
const PLACE_CELL_COUNT: usize = 128;
const ROLE_CELL_COUNT: usize = 64;
const PLACE_DOMAIN: u64 = 0x5233_504C_4143_4533;
const ROLE_DOMAIN: u64 = 0x5233_524F_4C45_5333;

pub(crate) const fn slot_start_token_id(slot_ordinal: u8) -> Option<u32> {
    if (slot_ordinal as usize) < SLOT_COUNT {
        Some(SLOT_START_TOKEN_BASE + slot_ordinal as u32)
    } else {
        None
    }
}

pub(crate) const fn slot_end_token_id(slot_ordinal: u8) -> Option<u32> {
    if (slot_ordinal as usize) < SLOT_COUNT {
        Some(SLOT_END_TOKEN_BASE + slot_ordinal as u32)
    } else {
        None
    }
}

pub(crate) const fn copy_slot_token_id(slot_ordinal: u8) -> Option<u32> {
    if (slot_ordinal as usize) < SLOT_COUNT {
        Some(COPY_SLOT_TOKEN_BASE + slot_ordinal as u32)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SlotOrdinal(u8);

impl SlotOrdinal {
    pub(crate) const fn new(value: u8) -> Option<Self> {
        if (value as usize) < SLOT_COUNT {
            Some(Self(value))
        } else {
            None
        }
    }

    const fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) const fn value(self) -> u8 {
        self.0
    }

    pub(crate) const fn copy_token_id(self) -> u32 {
        COPY_SLOT_TOKEN_BASE + self.0 as u32
    }

    fn from_start_token(token_id: u32) -> Option<Self> {
        token_id
            .checked_sub(SLOT_START_TOKEN_BASE)
            .and_then(|value| u8::try_from(value).ok())
            .and_then(Self::new)
    }

    fn from_end_token(token_id: u32) -> Option<Self> {
        token_id
            .checked_sub(SLOT_END_TOKEN_BASE)
            .and_then(|value| u8::try_from(value).ok())
            .and_then(Self::new)
    }

    pub(crate) fn from_copy_token(token_id: u32) -> Option<Self> {
        token_id
            .checked_sub(COPY_SLOT_TOKEN_BASE)
            .and_then(|value| u8::try_from(value).ok())
            .and_then(Self::new)
    }
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

#[derive(Clone, Copy)]
struct SlotTape {
    bindings: [BindingRow; BINDING_ROWS],
    len: u8,
    ready: bool,
}

const EMPTY_TAPE: SlotTape = SlotTape {
    bindings: [EMPTY_BINDING; BINDING_ROWS],
    len: 0,
    ready: false,
};

pub(crate) struct RolePlaceCopyState {
    enabled: bool,
    construction_seed: u32,
    copy_column_offset: u32,
    place_offset: u32,
    role_offset: u32,
    capture_slot: Option<SlotOrdinal>,
    recall_slot: Option<SlotOrdinal>,
    role_cells: [u32; ACTIVE_COLUMNS],
    place_cells: [u32; ACTIVE_COLUMNS],
    slots: [SlotTape; SLOT_COUNT],
    recall_position: u8,
    copy_end_columns: [u32; ACTIVE_COLUMNS],
}

impl RolePlaceCopyState {
    pub(crate) fn new(
        construction_seed: u32,
        column_count: u32,
        base_physical_cell_count: u32,
        token_codebook: &[TokenCode],
    ) -> Result<Self, &'static str> {
        let role_count = token_codebook
            .iter()
            .filter(|code| (SLOT_START_TOKEN_BASE..=COPY_END_TOKEN_ID).contains(&code.token_id))
            .count();
        let required_role_count = SLOT_COUNT * 3 + 1;
        if role_count != 0 && role_count != required_role_count {
            return Err("ordinal slot-bank control tokens must be all present or all absent");
        }
        let enabled = role_count == required_role_count;
        let token_count = u32::try_from(token_codebook.len())
            .map_err(|_| "ordinal slot-bank token population exceeds uint32")?;
        let output_basket = base_physical_cell_count
            .checked_add(token_count)
            .ok_or("ordinal slot-bank physical offsets overflow uint32")?;
        let copy_column_offset = output_basket
            .checked_add(1)
            .ok_or("ordinal slot-bank physical offsets overflow uint32")?;
        let place_offset = copy_column_offset
            .checked_add(column_count)
            .ok_or("ordinal slot-bank physical offsets overflow uint32")?;
        let role_offset = place_offset
            .checked_add(PLACE_CELL_COUNT as u32)
            .ok_or("ordinal slot-bank physical offsets overflow uint32")?;
        role_offset
            .checked_add(ROLE_CELL_COUNT as u32)
            .ok_or("ordinal slot-bank physical offsets overflow uint32")?;

        let copy_end_columns = if enabled {
            token_code(token_codebook, COPY_END_TOKEN_ID)
                .expect("all slot-bank role codes were counted")
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
            capture_slot: None,
            recall_slot: None,
            role_cells: [NONE; ACTIVE_COLUMNS],
            place_cells: [NONE; ACTIVE_COLUMNS],
            slots: [EMPTY_TAPE; SLOT_COUNT],
            recall_position: 0,
            copy_end_columns,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.capture_slot = None;
        self.recall_slot = None;
        self.role_cells = [NONE; ACTIVE_COLUMNS];
        self.place_cells = [NONE; ACTIVE_COLUMNS];
        self.slots = [EMPTY_TAPE; SLOT_COUNT];
        self.recall_position = 0;
    }

    pub(crate) fn normalizes_temporal_context(&self, token_id: u32) -> bool {
        SlotOrdinal::from_end_token(token_id).is_some_and(|slot| self.capture_slot == Some(slot))
    }

    pub(crate) fn is_end_boundary(&self, token_id: u32) -> bool {
        self.enabled && SlotOrdinal::from_end_token(token_id).is_some()
    }

    pub(crate) fn is_ready(&self, slot: SlotOrdinal) -> bool {
        self.enabled && self.slots[slot.index()].ready
    }

    pub(crate) fn has_ready_slot(&self) -> bool {
        self.enabled && self.slots.iter().any(|slot| slot.ready)
    }

    pub(crate) fn is_recalling(&self) -> bool {
        self.enabled && self.recall_slot.is_some()
    }

    pub(crate) fn preflight_arrival(&self, token_id: u32) -> Result<(), &'static str> {
        if !self.enabled {
            return Ok(());
        }
        if self.recall_slot.is_some() {
            return Err("arriving token cannot interrupt an active slot recall");
        }
        if SlotOrdinal::from_start_token(token_id).is_some() {
            return if self.capture_slot.is_some() {
                Err("slot start arrived while another capture was active")
            } else {
                Ok(())
            };
        }
        if let Some(slot) = SlotOrdinal::from_end_token(token_id) {
            return if self.capture_slot != Some(slot) {
                Err("slot end arrived without its matching active capture")
            } else if self.slots[slot.index()].len == 0 {
                Err("slot capture cannot be empty")
            } else {
                Ok(())
            };
        }
        if let Some(slot) = self.capture_slot {
            if is_internal_role_token(token_id) {
                return Err("slot-bank control token arrived inside slot capture");
            }
            if usize::from(self.slots[slot.index()].len) >= MAX_SLOT_TOKENS {
                return Err("captured value exceeds the bounded slot tape");
            }
        }
        Ok(())
    }

    pub(crate) fn observe_arrival(&mut self, token_id: u32, columns: [u32; ACTIVE_COLUMNS]) {
        if !self.enabled {
            return;
        }
        if let Some(slot) = SlotOrdinal::from_start_token(token_id) {
            self.slots[slot.index()] = EMPTY_TAPE;
            self.capture_slot = Some(slot);
            self.activate_role(slot);
            self.activate_place(0);
            return;
        }
        if let Some(slot) = SlotOrdinal::from_end_token(token_id) {
            debug_assert_eq!(self.capture_slot, Some(slot));
            let end_position = usize::from(self.slots[slot.index()].len);
            self.slots[slot.index()].bindings[end_position] =
                self.binding_row(slot, self.copy_end_columns);
            self.slots[slot.index()].ready = true;
            self.capture_slot = None;
            self.activate_role(slot);
            self.place_cells = [NONE; ACTIVE_COLUMNS];
            return;
        }
        if let Some(slot) = self.capture_slot {
            let position = usize::from(self.slots[slot.index()].len);
            self.slots[slot.index()].bindings[position] = self.binding_row(slot, columns);
            self.slots[slot.index()].len += 1;
            self.activate_place(usize::from(self.slots[slot.index()].len));
        }
    }

    pub(crate) fn begin_recall(&mut self, slot: SlotOrdinal) {
        debug_assert!(self.is_ready(slot));
        debug_assert!(self.capture_slot.is_none() && self.recall_slot.is_none());
        self.recall_slot = Some(slot);
        self.recall_position = 0;
        self.activate_role(slot);
        self.activate_place(0);
    }

    pub(crate) fn binding_columns_at(
        &self,
        slot: SlotOrdinal,
        position: u8,
    ) -> Result<[u32; ACTIVE_COLUMNS], &'static str> {
        let preview = self.recall_slot.is_none() && self.is_ready(slot) && position == 0;
        if !(self.recall_slot == Some(slot) || preview) {
            return Err("copy binding was requested outside its slot recall");
        }
        let contacts = self.binding_contacts(slot, position);
        let Some(row) = self.slots[slot.index()]
            .bindings
            .get(usize::from(position))
            .copied()
        else {
            return Err("copy place escaped its bounded population");
        };
        if !row.live {
            return Err("copy place has no live binding row");
        }
        let connected_pairs = row
            .segment_sources
            .iter()
            .filter(|source| contacts.contains(source))
            .count();
        if connected_pairs < 9 {
            return Err("copy role/place binding failed the physical active threshold");
        }
        Ok(row.owner_columns)
    }

    pub(crate) fn advance_recall(&mut self) {
        let slot = self.recall_slot.expect("recall advance requires one slot");
        debug_assert!(self.recall_position < self.slots[slot.index()].len);
        self.recall_position += 1;
        self.activate_place(usize::from(self.recall_position));
    }

    pub(crate) fn finish_recall(&mut self) {
        let slot = self.recall_slot.expect("recall finish requires one slot");
        debug_assert_eq!(self.recall_position, self.slots[slot.index()].len);
        self.recall_slot = None;
        self.role_cells = [NONE; ACTIVE_COLUMNS];
        self.place_cells = [NONE; ACTIVE_COLUMNS];
    }

    pub(crate) fn recall_slot(&self) -> Option<SlotOrdinal> {
        self.recall_slot
    }

    pub(crate) fn recall_position(&self) -> u8 {
        self.recall_position
    }

    pub(crate) fn recall_is_at_end(&self) -> bool {
        self.recall_slot
            .is_some_and(|slot| self.recall_position == self.slots[slot.index()].len)
    }

    pub(crate) fn copy_column_cell(&self, column: u32) -> Result<u32, &'static str> {
        self.copy_column_offset
            .checked_add(column)
            .ok_or("copy-column physical identity overflowed uint32")
    }

    fn binding_row(&self, slot: SlotOrdinal, owner_columns: [u32; ACTIVE_COLUMNS]) -> BindingRow {
        let mut sources = self.binding_contacts(slot, self.slots[slot.index()].len);
        sources.sort_unstable();
        BindingRow {
            segment_sources: sources,
            owner_columns,
            live: true,
        }
    }

    fn binding_contacts(&self, slot: SlotOrdinal, position: u8) -> [u32; PAIRS_PER_SEGMENT] {
        let role_cells = sparse_code(
            self.construction_seed,
            ROLE_DOMAIN,
            u64::from(slot.value()),
            self.role_offset,
            ROLE_CELL_COUNT,
        );
        let place_cells = sparse_code(
            self.construction_seed,
            PLACE_DOMAIN,
            u64::from(position),
            self.place_offset,
            PLACE_CELL_COUNT,
        );
        let mut contacts = [NONE; PAIRS_PER_SEGMENT];
        contacts[..ROLE_CONTACTS].copy_from_slice(&role_cells[..ROLE_CONTACTS]);
        contacts[ROLE_CONTACTS..].copy_from_slice(&place_cells[..PLACE_CONTACTS]);
        contacts
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

    fn activate_role(&mut self, slot: SlotOrdinal) {
        self.role_cells = sparse_code(
            self.construction_seed,
            ROLE_DOMAIN,
            u64::from(slot.value()),
            self.role_offset,
            ROLE_CELL_COUNT,
        );
    }
}

pub(crate) fn is_internal_role_token(token_id: u32) -> bool {
    (SLOT_START_TOKEN_BASE..=COPY_END_TOKEN_ID).contains(&token_id)
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
