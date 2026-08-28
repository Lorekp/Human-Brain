//! Fresh-circuit typed workspace prototype.
//!
//! The authoritative temporal store still owns every learned association. This
//! state retains one fitted plan assembly, canonical owner fingerprints for
//! slot-presence codes, and a bounded role-pointer stack, then reconstructs the
//! exact BoundWinners source/relay context used at a structural hole.

use super::{
    role_place_copy::{SlotOrdinal, FIRST_BOUND_SLOT},
    typed_hole_state::{
        SurfaceKind, TypedHoleAddress, TypedHoleCue, TypedHoleRole, TypedHoleState, TypedValue,
        MAX_HOLE_ADDRESS_DEPTH,
    },
    TemporalCircuitStop, TemporalStepMode, TemporalTransientState, ACTIVE_COLUMNS, CONTEXT_CELLS,
    NONE,
};

pub(super) const PLAN_WINNERS_IN_MIX: usize = 5;
// Nested exact fusion is 3 pointer + 3 projected plan + 3 semantic + 2
// relative-address winners. At theta_active=18 (nine logical pairs), the role
// pointer is the guaranteed three-source plan boundary; the projected plan
// adds local operation/signature specificity within that role. Transfer stays
// on its established pointer-free 5 plan + 5 semantic + 1 address mix so a new
// enclosing program can still compose a learned subtree.
const POINTER_WINNERS_IN_NESTED_EXACT: usize = 3;
const PROJECTED_PLAN_WINNERS_IN_POINTER_MIX: usize = 3;
const NESTED_POINTER_PLAN_WINNERS_IN_MIX: usize =
    POINTER_WINNERS_IN_NESTED_EXACT + PROJECTED_PLAN_WINNERS_IN_POINTER_MIX;
const NESTED_POINTER_SEMANTIC_WINNERS_IN_MIX: usize = 3;
const NESTED_POINTER_ADDRESS_WINNERS_IN_MIX: usize = 2;
const SIGNATURE_SEMANTIC_WINNERS_IN_MIX: usize = 3;
const SIGNATURE_ADDRESS_WINNERS_IN_MIX: usize = 3;
// A parented pointer cue is 3 parent + 3 surface + 2 semantic + 3 relative
// address. Parent, constructor surface, and position therefore each cross the
// three-mismatch deactivation boundary independently. Two semantic winners are
// sufficient here because, under an identical parent/surface/relative address,
// the typed hole cue is structurally determined; a semantic change necessarily
// changes the parent identity or child address too. Downstream nested exact
// fusion deliberately remains 3 pointer + 3 plan + 3 semantic + 2 address.
const PARENT_POINTER_WINNERS_IN_POINTER_CUE: usize = 3;
const SURFACE_WINNERS_WITH_PARENT_POINTER: usize = 3;
const SEMANTIC_WINNERS_WITH_PARENT_POINTER: usize = 2;
const ADDRESS_WINNERS_WITH_PARENT_POINTER: usize = 3;
// A bound-sensitive root or Head operation has no enclosing role identity. It
// binds the elected constructor to both bound-slot plan components directly:
// 3 surface + 3 lower-slot + 3 upper-slot + 2 semantic winners. Structural
// constructors retain the plan-neutral root cue so their pointers transfer
// across lower/upper presence; parented pointers remain keyed by their parent.
const SURFACE_WINNERS_WITHOUT_PARENT_POINTER: usize = 3;
const SLOT_FINGERPRINT_WINNERS: usize = 3;
const SEMANTIC_WINNERS_WITHOUT_PARENT_POINTER: usize = 2;
// Private component token codes are globally disjoint. The neutral code uses
// separate bands for absent lower/upper slots so one cue remains injective;
// present slots use the first band of their own private component code.
const LOWER_ABSENT_FINGERPRINT_SOURCES: [usize; SLOT_FINGERPRINT_WINNERS] = [3, 4, 5];
const UPPER_ABSENT_FINGERPRINT_SOURCES: [usize; SLOT_FINGERPRINT_WINNERS] = [6, 7, 8];
const BOUND_FINGERPRINT_SOURCES: [usize; SLOT_FINGERPRINT_WINNERS] = [0, 1, 2];
const ROLE_POINTER_STACK_CAPACITY: usize = MAX_HOLE_ADDRESS_DEPTH as usize + 1;
const TRANSFER_SEMANTIC_WINNERS_IN_MIX: usize = 5;
const TRANSFER_ADDRESS_WINNERS_IN_MIX: usize = 1;
const SCAFFOLD_HEAD_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [0, 1, 2, 3, 4];
// Scaffold backoff preserves the parameter signature and only whether any
// bound exists; the rebased semantic/address context carries subtree identity.
const PARAMETER_BOUND_PRESENCE_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [0, 1, 2, 5, 6];
const ROOT_OPERATION_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [0, 1, 2, 9, 10];
const NESTED_EXACT_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [0, 1, 8, 9, 10];
const TRANSFER_OPERATION_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [5, 6, 7, 9, 10];
const PARAMETER_SCAFFOLD_DESTINATIONS: [usize; 3] = [0, 1, 2];
const PARAMETER_SCAFFOLD_SOURCES: [usize; 3] = [3, 4, 5];
const EOS_PARAMETER_SOURCES: [usize; 2] = [6, 7];
const RESULT_SCAFFOLD_DESTINATIONS: [usize; 2] = [3, 4];
const RESULT_SCAFFOLD_SOURCES: [usize; 2] = [6, 7];
const ANY_BOUND_DESTINATIONS: [usize; 4] = [5, 6, 7, 8];
const ANY_BOUND_SOURCES: [usize; 4] = [5, 6, 7, 8];
const LOWER_BOUND_WINNER_INDEX: usize = 9;
const UPPER_BOUND_WINNER_INDEX: usize = 10;
pub(crate) const TYPED_CUE_TOKEN_COUNT: usize = 26;
pub(crate) const TYPED_SURFACE_TOKEN_COUNT: usize = 23;
pub(super) const ROLE_POINTER_SURFACE_ORDINALS: [usize; 8] = [0, 1, 2, 3, 5, 20, 21, 22];
pub(super) const ROLE_POINTER_SURFACE_SAMPLE_COUNT: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TypedPlanProjection {
    RootExact,
    NestedExact,
    OperationTransfer,
    ScaffoldTransfer,
}

#[derive(Clone, Copy)]
enum TypedPlanWinnerProjection {
    Retained(&'static [usize; PLAN_WINNERS_IN_MIX]),
    EosParameter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TypedWorkspaceVocabulary {
    pub(crate) plan_commit_token_id: u32,
    pub(crate) address_token_base: u32,
    pub(crate) cue_token_ids: [u32; TYPED_CUE_TOKEN_COUNT],
    pub(crate) surface_token_ids: [u32; TYPED_SURFACE_TOKEN_COUNT],
}

impl TypedWorkspaceVocabulary {
    pub(crate) fn validate(self) -> Result<(), &'static str> {
        let mut ids = [0u32; 2 + TYPED_CUE_TOKEN_COUNT + TYPED_SURFACE_TOKEN_COUNT];
        ids[0] = self.plan_commit_token_id;
        ids[1] = self.address_token_base;
        ids[2..2 + TYPED_CUE_TOKEN_COUNT].copy_from_slice(&self.cue_token_ids);
        ids[2 + TYPED_CUE_TOKEN_COUNT..].copy_from_slice(&self.surface_token_ids);
        if ids.contains(&NONE) {
            return Err("typed vocabulary contains the reserved absent token ID");
        }
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("typed vocabulary token IDs must be distinct");
        }
        Ok(())
    }

    pub(crate) fn cue_token(self, cue: TypedHoleCue, role: TypedHoleRole) -> Option<u32> {
        let role_offset = match role {
            TypedHoleRole::Head => 0,
            TypedHoleRole::Child0 => 1,
            TypedHoleRole::Child1 => 2,
            TypedHoleRole::Child2 => 3,
        };
        let index = match cue {
            TypedHoleCue::Expression(TypedValue::Int) => role_offset,
            TypedHoleCue::Expression(TypedValue::Bool) => 4 + role_offset,
            TypedHoleCue::Expression(TypedValue::ListInt) => 8 + role_offset,
            TypedHoleCue::Expression(TypedValue::IntToInt) => 12 + role_offset,
            TypedHoleCue::LiteralSign => 16,
            TypedHoleCue::LiteralFirstDigit => 17,
            TypedHoleCue::LiteralFirstNonzeroDigit => 18,
            TypedHoleCue::LiteralDigitOrEnd => 19,
            TypedHoleCue::LiteralEnd => 20,
            TypedHoleCue::Eos => 21,
            TypedHoleCue::Terminated => return None,
        };
        Some(self.cue_token_ids[index])
    }

    pub(crate) const fn neutral_operation_component_token(self) -> u32 {
        self.cue_token_ids[22]
    }

    pub(crate) const fn any_bound_operation_component_token(self) -> u32 {
        self.cue_token_ids[25]
    }

    pub(crate) fn bound_operation_component_token(
        self,
        slot: SlotOrdinal,
    ) -> Result<u32, &'static str> {
        let role_offset = slot
            .value()
            .checked_sub(FIRST_BOUND_SLOT)
            .ok_or("typed operation component requires a bound slot role")?;
        let index = 23usize
            .checked_add(usize::from(role_offset))
            .filter(|index| *index < TYPED_CUE_TOKEN_COUNT)
            .ok_or("typed operation component exceeds the bound slot role bank")?;
        Ok(self.cue_token_ids[index])
    }

    pub(crate) fn address_token(self, address: TypedHoleAddress) -> Result<u32, &'static str> {
        let mut level_start = 0u32;
        let mut level_width = 1u32;
        let mut rank = 0u32;
        for index in 0..address.depth() {
            level_start = level_start
                .checked_add(level_width)
                .ok_or("typed structural address offset overflowed uint32")?;
            level_width = level_width
                .checked_mul(3)
                .ok_or("typed structural address width overflowed uint32")?;
            let ordinal = address
                .child_ordinal(index)
                .ok_or("typed structural address lost an ordinal")?;
            if ordinal > 2 {
                return Err("typed structural address contains an unsupported child ordinal");
            }
            rank = rank
                .checked_mul(3)
                .and_then(|value| value.checked_add(u32::from(ordinal)))
                .ok_or("typed structural address rank overflowed uint32")?;
        }
        let offset = level_start
            .checked_add(rank)
            .ok_or("typed structural address offset overflowed uint32")?;
        let token_id = self
            .address_token_base
            .checked_add(offset)
            .ok_or("typed structural address token ID overflowed uint32")?;
        if token_id == self.plan_commit_token_id
            || self.cue_token_ids.contains(&token_id)
            || self.surface_token_ids.contains(&token_id)
        {
            return Err("typed structural address token collides with the installed vocabulary");
        }
        Ok(token_id)
    }

    pub(crate) fn surface_for_token(self, token_id: u32) -> Option<SurfaceKind> {
        let index = self
            .surface_token_ids
            .iter()
            .position(|&candidate| candidate == token_id)?;
        Some(match index {
            0 => SurfaceKind::Add,
            1 => SurfaceKind::Absolute,
            2 => SurfaceKind::Map,
            3 => SurfaceKind::Bind,
            4 => SurfaceKind::Variable0,
            5 => SurfaceKind::Integer,
            6 => SurfaceKind::Plus,
            7 => SurfaceKind::Minus,
            8..=17 => SurfaceKind::Digit((index - 8) as u8),
            18 => SurfaceKind::End,
            19 => SurfaceKind::Eos,
            20 => SurfaceKind::LessThan,
            21 => SurfaceKind::If,
            22 => SurfaceKind::Sum,
            _ => unreachable!("fixed typed surface vocabulary is exhaustive"),
        })
    }

    pub(crate) fn admitted_tokens(
        self,
        holes: &TypedHoleState,
    ) -> ([u32; TYPED_SURFACE_TOKEN_COUNT], usize) {
        let mut admitted = [NONE; TYPED_SURFACE_TOKEN_COUNT];
        let mut count = 0usize;
        for &token_id in &self.surface_token_ids {
            let surface = self
                .surface_for_token(token_id)
                .expect("surface table maps every own token");
            if holes.admits(surface) {
                admitted[count] = token_id;
                count += 1;
            }
        }
        (admitted, count)
    }
}

struct TypedSessionState {
    vocabulary: TypedWorkspaceVocabulary,
    holes: TypedHoleState,
    mode: TemporalStepMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ActiveRolePointer {
    pub(super) owners: [u32; ACTIVE_COLUMNS],
    pub(super) anchor: TypedHoleAddress,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RolePointerBinding {
    Bound(ActiveRolePointer),
    Unbound(TypedHoleAddress),
}

impl RolePointerBinding {
    const fn anchor(self) -> TypedHoleAddress {
        match self {
            Self::Bound(pointer) => pointer.anchor,
            Self::Unbound(anchor) => anchor,
        }
    }
}

pub(super) struct TypedWorkspaceState {
    plan_winners: [u32; ACTIVE_COLUMNS],
    eos_parameter_winners: [u32; 2],
    lower_slot_pointer_winners: [u32; SLOT_FINGERPRINT_WINNERS],
    upper_slot_pointer_winners: [u32; SLOT_FINGERPRINT_WINNERS],
    plan_ready: bool,
    role_pointer_bindings: [Option<RolePointerBinding>; ROLE_POINTER_STACK_CAPACITY],
    session: Option<TypedSessionState>,
}

impl TypedWorkspaceState {
    pub(super) const EMPTY: Self = Self {
        plan_winners: [0; ACTIVE_COLUMNS],
        eos_parameter_winners: [0; 2],
        lower_slot_pointer_winners: [0; SLOT_FINGERPRINT_WINNERS],
        upper_slot_pointer_winners: [0; SLOT_FINGERPRINT_WINNERS],
        plan_ready: false,
        role_pointer_bindings: [None; ROLE_POINTER_STACK_CAPACITY],
        session: None,
    };

    pub(super) fn clear(&mut self) {
        *self = Self::EMPTY;
    }

    pub(super) fn retain_plan_components(
        &mut self,
        parameter_scaffold_context: [u32; CONTEXT_CELLS],
        result_scaffold_context: [u32; CONTEXT_CELLS],
        neutral_operation_context: [u32; CONTEXT_CELLS],
        any_bound_operation_context: Option<[u32; CONTEXT_CELLS]>,
        lower_bound_context: Option<[u32; CONTEXT_CELLS]>,
        upper_bound_context: Option<[u32; CONTEXT_CELLS]>,
        neutral_operation_fingerprint: [u32; ACTIVE_COLUMNS],
        lower_bound_fingerprint: Option<[u32; ACTIVE_COLUMNS]>,
        upper_bound_fingerprint: Option<[u32; ACTIVE_COLUMNS]>,
    ) {
        for (destination, source) in PARAMETER_SCAFFOLD_DESTINATIONS
            .into_iter()
            .zip(PARAMETER_SCAFFOLD_SOURCES)
        {
            self.plan_winners[destination] = parameter_scaffold_context[source];
        }
        for (destination, source) in self
            .eos_parameter_winners
            .iter_mut()
            .zip(EOS_PARAMETER_SOURCES)
        {
            *destination = parameter_scaffold_context[source];
        }
        for (destination, source) in RESULT_SCAFFOLD_DESTINATIONS
            .into_iter()
            .zip(RESULT_SCAFFOLD_SOURCES)
        {
            self.plan_winners[destination] = result_scaffold_context[source];
        }
        let any_bound_operation_context =
            any_bound_operation_context.unwrap_or(neutral_operation_context);
        for (destination, source) in ANY_BOUND_DESTINATIONS.into_iter().zip(ANY_BOUND_SOURCES) {
            self.plan_winners[destination] = any_bound_operation_context[source];
        }
        let lower_fingerprint_sources = if lower_bound_fingerprint.is_some() {
            BOUND_FINGERPRINT_SOURCES
        } else {
            LOWER_ABSENT_FINGERPRINT_SOURCES
        };
        let lower_pointer_context =
            lower_bound_fingerprint.unwrap_or(neutral_operation_fingerprint);
        let lower_bound_context = lower_bound_context.unwrap_or(neutral_operation_context);
        self.plan_winners[LOWER_BOUND_WINNER_INDEX] = lower_bound_context[LOWER_BOUND_WINNER_INDEX];
        for (destination, source) in self
            .lower_slot_pointer_winners
            .iter_mut()
            .zip(lower_fingerprint_sources)
        {
            *destination = lower_pointer_context[source];
        }
        let upper_fingerprint_sources = if upper_bound_fingerprint.is_some() {
            BOUND_FINGERPRINT_SOURCES
        } else {
            UPPER_ABSENT_FINGERPRINT_SOURCES
        };
        let upper_pointer_context =
            upper_bound_fingerprint.unwrap_or(neutral_operation_fingerprint);
        let upper_bound_context = upper_bound_context.unwrap_or(neutral_operation_context);
        self.plan_winners[UPPER_BOUND_WINNER_INDEX] = upper_bound_context[UPPER_BOUND_WINNER_INDEX];
        for (destination, source) in self
            .upper_slot_pointer_winners
            .iter_mut()
            .zip(upper_fingerprint_sources)
        {
            *destination = upper_pointer_context[source];
        }
        self.plan_ready = true;
    }

    pub(super) const fn is_ready(&self) -> bool {
        self.plan_ready
    }

    pub(super) fn begin_session(
        &mut self,
        vocabulary: TypedWorkspaceVocabulary,
        parameter: TypedValue,
        result: TypedValue,
        mode: TemporalStepMode,
    ) -> Result<(), TemporalCircuitStop> {
        if !self.plan_ready {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed plan assembly is not committed",
            ));
        }
        vocabulary
            .validate()
            .map_err(TemporalCircuitStop::MalformedContextOrPolicy)?;
        let holes = TypedHoleState::unary(parameter, result).map_err(|_| {
            TemporalCircuitStop::MalformedContextOrPolicy(
                "typed workspace received an unsupported unary signature",
            )
        })?;
        self.session = Some(TypedSessionState {
            vocabulary,
            holes,
            mode,
        });
        self.role_pointer_bindings = [None; ROLE_POINTER_STACK_CAPACITY];
        Ok(())
    }

    pub(super) fn session_mode(&self) -> Option<TemporalStepMode> {
        self.session.as_ref().map(|session| session.mode)
    }

    pub(super) fn current_context_tokens(
        &self,
        projection: TypedPlanProjection,
    ) -> Result<(Option<u32>, u32), TemporalCircuitStop> {
        let session =
            self.session
                .as_ref()
                .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed workspace session is not initialized",
                ))?;
        let mut role = session.holes.role();
        let mut address = session.holes.address();
        let relative_pointer_address = (projection == TypedPlanProjection::NestedExact)
            .then(|| self.role_local_pointer_for_current_hole())
            .flatten()
            .map(|(_, relative)| relative);
        // A scaffold teacher stores its backoff trace one generic enclosing
        // edge closer to root. Read-only fallback keeps the target's current
        // path, so a learned [0]/[0,0]/[0,1] subtree is queried as
        // []/[0]/[1] without changing TypedHoleState itself.
        if let Some(relative) = relative_pointer_address {
            address = relative;
        } else if projection == TypedPlanProjection::ScaffoldTransfer
            && session.mode == TemporalStepMode::Learn
        {
            address = address.without_leading_ordinal();
            if address.depth() == 0 {
                role = TypedHoleRole::Head;
            }
        }
        let cue_token_id = session.vocabulary.cue_token(session.holes.cue(), role);
        let address_token_id = session
            .vocabulary
            .address_token(address)
            .map_err(TemporalCircuitStop::MalformedContextOrPolicy)?;
        Ok((cue_token_id, address_token_id))
    }

    /// Return the native private-token cue used to recruit or recall the role
    /// pointer for the surface currently occupying this hole.  A nested
    /// constructor uses its address relative to the active enclosing pointer;
    /// a true root or fresh `Head` starts at the constructor-neutral root.
    pub(super) fn current_role_pointer_tokens(
        &self,
    ) -> Result<(u32, u32, u32, TypedHoleAddress), TemporalCircuitStop> {
        let session =
            self.session
                .as_ref()
                .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed workspace session is not initialized",
                ))?;
        let cue_token_id = session
            .vocabulary
            .cue_token(session.holes.cue(), session.holes.role())
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "terminated typed state cannot recruit a role pointer",
            ))?;
        let anchor = session.holes.address();
        let relative_address = self
            .role_local_pointer_for_current_hole()
            .map_or_else(TypedHoleAddress::root, |(_, relative)| relative);
        let address_token_id = session
            .vocabulary
            .address_token(relative_address)
            .map_err(TemporalCircuitStop::MalformedContextOrPolicy)?;
        Ok((
            session.vocabulary.plan_commit_token_id,
            cue_token_id,
            address_token_id,
            anchor,
        ))
    }

    pub(super) fn current_address_is_root(&self) -> Result<bool, TemporalCircuitStop> {
        self.session
            .as_ref()
            .map(|session| session.holes.address().depth() == 0)
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed workspace session is not initialized",
            ))
    }

    pub(super) fn current_cue(&self) -> Result<TypedHoleCue, TemporalCircuitStop> {
        self.session
            .as_ref()
            .map(|session| session.holes.cue())
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed workspace session is not initialized",
            ))
    }

    pub(super) fn current_role(&self) -> Result<TypedHoleRole, TemporalCircuitStop> {
        self.session
            .as_ref()
            .map(|session| session.holes.role())
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed workspace session is not initialized",
            ))
    }

    pub(super) fn admitted_tokens(
        &self,
    ) -> Result<([u32; TYPED_SURFACE_TOKEN_COUNT], usize), TemporalCircuitStop> {
        let session =
            self.session
                .as_ref()
                .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed workspace session is not initialized",
                ))?;
        Ok(session.vocabulary.admitted_tokens(&session.holes))
    }

    pub(super) fn advance_token(&mut self, token_id: u32) -> Result<(), TemporalCircuitStop> {
        {
            let session =
                self.session
                    .as_mut()
                    .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                        "typed workspace session is not initialized",
                    ))?;
            let surface = session.vocabulary.surface_for_token(token_id).ok_or(
                TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed generation elected a token outside its structural vocabulary",
                ),
            )?;
            session.holes.advance(surface).map_err(|_| {
                TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed generation elected a structurally inadmissible token",
                )
            })?;
        }
        let Some(current_address) = self.session.as_ref().map(|session| session.holes.address())
        else {
            self.role_pointer_bindings = [None; ROLE_POINTER_STACK_CAPACITY];
            return Ok(());
        };
        let mut retained = 0usize;
        for index in 0..ROLE_POINTER_STACK_CAPACITY {
            let Some(binding) = self.role_pointer_bindings[index] else {
                continue;
            };
            if current_address.relative_to(binding.anchor()).is_some() {
                self.role_pointer_bindings[retained] = Some(binding);
                retained += 1;
            }
        }
        for binding in &mut self.role_pointer_bindings[retained..] {
            *binding = None;
        }
        Ok(())
    }

    /// Role pointers bind constructors which open a descendant assembly and
    /// Integer leaves which open a literal phase at the same structural
    /// address. A variable, sign, digit, end marker, or EOS does not open a new
    /// pointer scope; the enclosing constructor or Integer pointer remains on
    /// the bounded stack.
    pub(super) fn token_opens_role_pointer_scope(&self, token_id: u32) -> bool {
        self.session.as_ref().is_some_and(|session| {
            session
                .vocabulary
                .surface_token_ids
                .iter()
                .position(|&candidate| candidate == token_id)
                .is_some_and(|ordinal| ROLE_POINTER_SURFACE_ORDINALS.contains(&ordinal))
        })
    }

    pub(super) fn active_role_pointer(&self) -> Option<ActiveRolePointer> {
        self.role_local_pointer_for_current_hole()
            .map(|(pointer, _)| pointer)
    }

    /// A binder body is explicitly a fresh `Head`, so an enclosing Bind
    /// pointer must not decide the body's operation. Ordinary constructor
    /// children retain the deepest ancestor pointer; a nested constructor then
    /// pushes its own role-local assembly after winning its surface race.
    fn role_local_pointer_for_current_hole(&self) -> Option<(ActiveRolePointer, TypedHoleAddress)> {
        let session = self.session.as_ref()?;
        let current_address = session.holes.address();
        let binding = self
            .role_pointer_bindings
            .iter()
            .rev()
            .flatten()
            .copied()
            .find(|binding| current_address.relative_to(binding.anchor()).is_some())?;
        let pointer = match binding {
            RolePointerBinding::Bound(pointer) => pointer,
            RolePointerBinding::Unbound(_) => return None,
        };
        let relative = current_address.relative_to(pointer.anchor)?;
        if relative.depth() != 0 && session.holes.role() == TypedHoleRole::Head {
            return None;
        }
        Some((pointer, relative))
    }

    pub(super) fn activate_role_pointer(
        &mut self,
        owners: [u32; ACTIVE_COLUMNS],
        anchor: TypedHoleAddress,
    ) -> Result<(), TemporalCircuitStop> {
        let current_address = self
            .session
            .as_ref()
            .map(|session| session.holes.address())
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed workspace session is not initialized",
            ))?;
        if current_address != anchor {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed role pointer anchor changed before activation",
            ));
        }
        let mut sorted = owners;
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed role pointer owners are not injective",
            ));
        }
        self.install_role_pointer_binding(
            RolePointerBinding::Bound(ActiveRolePointer { owners, anchor }),
            anchor,
        )
    }

    pub(super) fn mark_role_pointer_unbound(
        &mut self,
        anchor: TypedHoleAddress,
    ) -> Result<(), TemporalCircuitStop> {
        self.install_role_pointer_binding(RolePointerBinding::Unbound(anchor), anchor)
    }

    fn install_role_pointer_binding(
        &mut self,
        binding: RolePointerBinding,
        anchor: TypedHoleAddress,
    ) -> Result<(), TemporalCircuitStop> {
        let current_address = self
            .session
            .as_ref()
            .map(|session| session.holes.address())
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed workspace session is not initialized",
            ))?;
        if current_address != anchor || binding.anchor() != anchor {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed role pointer anchor changed before activation",
            ));
        }
        for index in 0..ROLE_POINTER_STACK_CAPACITY {
            match self.role_pointer_bindings[index] {
                Some(existing) if existing.anchor() == anchor => {
                    self.role_pointer_bindings[index] = Some(binding);
                    for descendant in &mut self.role_pointer_bindings[index + 1..] {
                        *descendant = None;
                    }
                    return Ok(());
                }
                Some(_) => {}
                None => {
                    self.role_pointer_bindings[index] = Some(binding);
                    return Ok(());
                }
            }
        }
        return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
            "typed role pointer stack exceeds the structural address depth",
        ));
    }

    pub(super) fn role_pointer_cue_transient(
        &self,
        surface_token_id: u32,
        surface_winner_context: [u32; CONTEXT_CELLS],
        cue_winner_context: [u32; CONTEXT_CELLS],
        address_winner_context: [u32; CONTEXT_CELLS],
        relay_offset: u32,
        event_count: u64,
    ) -> Result<TemporalTransientState, TemporalCircuitStop> {
        let session =
            self.session
                .as_ref()
                .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed workspace session is not initialized",
                ))?;
        let surface = session
            .vocabulary
            .surface_for_token(surface_token_id)
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed role pointer received an unknown surface token",
            ))?;
        let bound_sensitive = match surface {
            SurfaceKind::Add | SurfaceKind::Integer | SurfaceKind::LessThan | SurfaceKind::If => {
                true
            }
            SurfaceKind::Absolute | SurfaceKind::Map | SurfaceKind::Bind | SurfaceKind::Sum => {
                false
            }
            _ => {
                return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed surface does not open a role-pointer scope",
                ));
            }
        };
        let mut winners = [0u32; ACTIVE_COLUMNS];
        if let Some((parent, _)) = self.role_local_pointer_for_current_hole() {
            let surface_start = PARENT_POINTER_WINNERS_IN_POINTER_CUE;
            let semantic_start = surface_start + SURFACE_WINNERS_WITH_PARENT_POINTER;
            let address_start = semantic_start + SEMANTIC_WINNERS_WITH_PARENT_POINTER;
            winners[..surface_start]
                .copy_from_slice(&parent.owners[..PARENT_POINTER_WINNERS_IN_POINTER_CUE]);
            winners[surface_start..semantic_start]
                .copy_from_slice(&surface_winner_context[..SURFACE_WINNERS_WITH_PARENT_POINTER]);
            winners[semantic_start..address_start]
                .copy_from_slice(&cue_winner_context[..SEMANTIC_WINNERS_WITH_PARENT_POINTER]);
            winners[address_start..]
                .copy_from_slice(&address_winner_context[..ADDRESS_WINNERS_WITH_PARENT_POINTER]);
        } else if bound_sensitive {
            let lower_start = SURFACE_WINNERS_WITHOUT_PARENT_POINTER;
            let upper_start = lower_start + SLOT_FINGERPRINT_WINNERS;
            let semantic_start = upper_start + SLOT_FINGERPRINT_WINNERS;
            winners[..lower_start]
                .copy_from_slice(&surface_winner_context[..SURFACE_WINNERS_WITHOUT_PARENT_POINTER]);
            winners[lower_start..upper_start].copy_from_slice(&self.lower_slot_pointer_winners);
            winners[upper_start..semantic_start].copy_from_slice(&self.upper_slot_pointer_winners);
            winners[semantic_start..]
                .copy_from_slice(&cue_winner_context[..SEMANTIC_WINNERS_WITHOUT_PARENT_POINTER]);
        } else {
            winners[..PLAN_WINNERS_IN_MIX]
                .copy_from_slice(&surface_winner_context[..PLAN_WINNERS_IN_MIX]);
            winners[PLAN_WINNERS_IN_MIX..PLAN_WINNERS_IN_MIX + SIGNATURE_SEMANTIC_WINNERS_IN_MIX]
                .copy_from_slice(&cue_winner_context[..SIGNATURE_SEMANTIC_WINNERS_IN_MIX]);
            winners[PLAN_WINNERS_IN_MIX + SIGNATURE_SEMANTIC_WINNERS_IN_MIX..]
                .copy_from_slice(&address_winner_context[..SIGNATURE_ADDRESS_WINNERS_IN_MIX]);
        }
        let mut sorted = winners;
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed surface/cue role-pointer mix is not injective",
            ));
        }
        context_from_winners(winners, relay_offset, event_count)
    }

    pub(super) const fn plan_winners(&self) -> Option<[u32; ACTIVE_COLUMNS]> {
        if self.plan_ready {
            Some(self.plan_winners)
        } else {
            None
        }
    }

    pub(super) fn mixed_transient(
        &self,
        cue: TypedHoleCue,
        projection: TypedPlanProjection,
        cue_winner_context: [u32; CONTEXT_CELLS],
        address_winner_context: [u32; CONTEXT_CELLS],
        relay_offset: u32,
        event_count: u64,
    ) -> Result<TemporalTransientState, TemporalCircuitStop> {
        if !self.plan_ready {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed plan assembly is not committed",
            ));
        }
        let mut winners = [0u32; ACTIVE_COLUMNS];
        let plan_projection = plan_projection(cue, projection)?;
        let mut projected_plan = [0u32; PLAN_WINNERS_IN_MIX];
        match plan_projection {
            TypedPlanWinnerProjection::Retained(plan_indices) => {
                for (mixed_index, &plan_index) in plan_indices.iter().enumerate() {
                    projected_plan[mixed_index] = self.plan_winners[plan_index];
                }
            }
            TypedPlanWinnerProjection::EosParameter => {
                projected_plan[..3].copy_from_slice(&self.plan_winners[..3]);
                projected_plan[3..].copy_from_slice(&self.eos_parameter_winners);
            }
        }
        let nested_pointer = (projection == TypedPlanProjection::NestedExact)
            .then(|| self.role_local_pointer_for_current_hole())
            .flatten()
            .map(|(pointer, _)| pointer);
        if let Some(pointer) = nested_pointer {
            winners[..POINTER_WINNERS_IN_NESTED_EXACT]
                .copy_from_slice(&pointer.owners[..POINTER_WINNERS_IN_NESTED_EXACT]);
            winners[POINTER_WINNERS_IN_NESTED_EXACT..NESTED_POINTER_PLAN_WINNERS_IN_MIX]
                .copy_from_slice(
                    &projected_plan[PLAN_WINNERS_IN_MIX - PROJECTED_PLAN_WINNERS_IN_POINTER_MIX..],
                );
            for mixed_index in 0..NESTED_POINTER_SEMANTIC_WINNERS_IN_MIX {
                winners[NESTED_POINTER_PLAN_WINNERS_IN_MIX + mixed_index] =
                    cue_winner_context[mixed_index];
            }
            for mixed_index in 0..NESTED_POINTER_ADDRESS_WINNERS_IN_MIX {
                winners[NESTED_POINTER_PLAN_WINNERS_IN_MIX
                    + NESTED_POINTER_SEMANTIC_WINNERS_IN_MIX
                    + mixed_index] = address_winner_context[mixed_index];
            }
        } else {
            winners[..PLAN_WINNERS_IN_MIX].copy_from_slice(&projected_plan);
            let (semantic_winner_count, address_winner_count) =
                if uses_transfer_projection(cue, projection) {
                    (
                        TRANSFER_SEMANTIC_WINNERS_IN_MIX,
                        TRANSFER_ADDRESS_WINNERS_IN_MIX,
                    )
                } else {
                    (
                        SIGNATURE_SEMANTIC_WINNERS_IN_MIX,
                        SIGNATURE_ADDRESS_WINNERS_IN_MIX,
                    )
                };
            for mixed_index in 0..semantic_winner_count {
                winners[PLAN_WINNERS_IN_MIX + mixed_index] = cue_winner_context[mixed_index];
            }
            for mixed_index in 0..address_winner_count {
                winners[PLAN_WINNERS_IN_MIX + semantic_winner_count + mixed_index] =
                    address_winner_context[mixed_index];
            }
        }
        let mut sorted = winners;
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed plan/cue winner mix is not injective",
            ));
        }
        context_from_winners(winners, relay_offset, event_count)
    }
}

fn uses_transfer_projection(cue: TypedHoleCue, projection: TypedPlanProjection) -> bool {
    matches!(
        (cue, projection),
        (
            TypedHoleCue::Expression(TypedValue::Int | TypedValue::Bool)
                | TypedHoleCue::LiteralSign
                | TypedHoleCue::LiteralFirstDigit
                | TypedHoleCue::LiteralFirstNonzeroDigit
                | TypedHoleCue::LiteralDigitOrEnd
                | TypedHoleCue::LiteralEnd,
            TypedPlanProjection::OperationTransfer,
        ) | (
            TypedHoleCue::Expression(TypedValue::ListInt | TypedValue::IntToInt),
            TypedPlanProjection::ScaffoldTransfer,
        )
    )
}

fn plan_projection(
    cue: TypedHoleCue,
    projection: TypedPlanProjection,
) -> Result<TypedPlanWinnerProjection, TemporalCircuitStop> {
    match cue {
        TypedHoleCue::Expression(TypedValue::ListInt)
        | TypedHoleCue::Expression(TypedValue::IntToInt) => match projection {
            TypedPlanProjection::RootExact | TypedPlanProjection::NestedExact => Ok(
                TypedPlanWinnerProjection::Retained(&SCAFFOLD_HEAD_PLAN_INDICES),
            ),
            TypedPlanProjection::ScaffoldTransfer => Ok(TypedPlanWinnerProjection::Retained(
                &PARAMETER_BOUND_PRESENCE_PLAN_INDICES,
            )),
            TypedPlanProjection::OperationTransfer => {
                Err(TemporalCircuitStop::MalformedContextOrPolicy(
                    "operation transfer requested for a scaffold typed hole",
                ))
            }
        },
        TypedHoleCue::Expression(TypedValue::Int | TypedValue::Bool)
        | TypedHoleCue::LiteralSign
        | TypedHoleCue::LiteralFirstDigit
        | TypedHoleCue::LiteralFirstNonzeroDigit
        | TypedHoleCue::LiteralDigitOrEnd
        | TypedHoleCue::LiteralEnd => match projection {
            TypedPlanProjection::RootExact => Ok(TypedPlanWinnerProjection::Retained(
                &ROOT_OPERATION_PLAN_INDICES,
            )),
            TypedPlanProjection::NestedExact => Ok(TypedPlanWinnerProjection::Retained(
                &NESTED_EXACT_PLAN_INDICES,
            )),
            TypedPlanProjection::OperationTransfer => Ok(TypedPlanWinnerProjection::Retained(
                &TRANSFER_OPERATION_PLAN_INDICES,
            )),
            TypedPlanProjection::ScaffoldTransfer => {
                Err(TemporalCircuitStop::MalformedContextOrPolicy(
                    "scaffold transfer requested for an operation typed hole",
                ))
            }
        },
        TypedHoleCue::Eos => match projection {
            TypedPlanProjection::RootExact | TypedPlanProjection::NestedExact => {
                Ok(TypedPlanWinnerProjection::EosParameter)
            }
            TypedPlanProjection::OperationTransfer | TypedPlanProjection::ScaffoldTransfer => {
                Err(TemporalCircuitStop::MalformedContextOrPolicy(
                    "EOS does not admit a typed transfer projection",
                ))
            }
        },
        TypedHoleCue::Terminated => Err(TemporalCircuitStop::MalformedContextOrPolicy(
            "terminated typed state has no structural plan projection",
        )),
    }
}

pub(super) fn context_from_winners(
    winners: [u32; ACTIVE_COLUMNS],
    relay_offset: u32,
    event_count: u64,
) -> Result<TemporalTransientState, TemporalCircuitStop> {
    let mut context = [0u32; CONTEXT_CELLS];
    context[..ACTIVE_COLUMNS].copy_from_slice(&winners);
    for (index, winner) in winners.into_iter().enumerate() {
        context[ACTIVE_COLUMNS + index] = relay_offset.checked_add(winner).ok_or(
            TemporalCircuitStop::MalformedContextOrPolicy(
                "typed workspace relay identity overflows uint32",
            ),
        )?;
    }
    Ok(TemporalTransientState {
        previous_context: context,
        previous_context_len: CONTEXT_CELLS as u8,
        previous_winner_context: context,
        previous_winner_context_len: CONTEXT_CELLS as u8,
        event_count,
    })
}
