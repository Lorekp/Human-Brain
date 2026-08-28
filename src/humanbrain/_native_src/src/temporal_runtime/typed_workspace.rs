//! Fresh-circuit typed workspace prototype.
//!
//! The authoritative temporal store still owns every learned association. This
//! state only retains one fitted plan assembly and reconstructs the exact
//! BoundWinners source/relay context used at a structural hole.

use super::{
    typed_hole_state::{
        SurfaceKind, TypedHoleAddress, TypedHoleCue, TypedHoleRole, TypedHoleState, TypedValue,
    },
    TemporalCircuitStop, TemporalStepMode, TemporalTransientState, ACTIVE_COLUMNS, CONTEXT_CELLS,
    NONE,
};

pub(super) const PLAN_WINNERS_IN_MIX: usize = 5;
pub(super) const SEMANTIC_CUE_WINNERS_IN_MIX: usize = 3;
pub(super) const ADDRESS_CUE_WINNERS_IN_MIX: usize = 3;
const SCAFFOLD_HEAD_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [0, 1, 2, 3, 4];
const OPERATION_HEAD_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [5, 6, 7, 8, 9];
pub(crate) const TYPED_CUE_TOKEN_COUNT: usize = 25;
pub(crate) const TYPED_SURFACE_TOKEN_COUNT: usize = 22;

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

    pub(crate) fn operation_component_token(
        self,
        ready_bound_slots: u8,
    ) -> Result<u32, &'static str> {
        let index = 22usize
            .checked_add(usize::from(ready_bound_slots))
            .filter(|index| *index < TYPED_CUE_TOKEN_COUNT)
            .ok_or("typed operation plan supports zero, one, or two bound slots")?;
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

pub(super) struct TypedWorkspaceState {
    plan_winners: [u32; ACTIVE_COLUMNS],
    plan_ready: bool,
    session: Option<TypedSessionState>,
}

impl TypedWorkspaceState {
    pub(super) const EMPTY: Self = Self {
        plan_winners: [0; ACTIVE_COLUMNS],
        plan_ready: false,
        session: None,
    };

    pub(super) fn clear(&mut self) {
        *self = Self::EMPTY;
    }

    pub(super) fn retain_plan_components(
        &mut self,
        scaffold_context: [u32; CONTEXT_CELLS],
        operation_context: [u32; CONTEXT_CELLS],
    ) {
        self.plan_winners[..PLAN_WINNERS_IN_MIX]
            .copy_from_slice(&scaffold_context[..PLAN_WINNERS_IN_MIX]);
        self.plan_winners[PLAN_WINNERS_IN_MIX..]
            .copy_from_slice(&operation_context[PLAN_WINNERS_IN_MIX..ACTIVE_COLUMNS]);
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
        Ok(())
    }

    pub(super) fn session_mode(&self) -> Option<TemporalStepMode> {
        self.session.as_ref().map(|session| session.mode)
    }

    pub(super) fn current_cue_token(&self) -> Result<Option<u32>, TemporalCircuitStop> {
        let session =
            self.session
                .as_ref()
                .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed workspace session is not initialized",
                ))?;
        Ok(session
            .vocabulary
            .cue_token(session.holes.cue(), session.holes.role()))
    }

    pub(super) fn current_address_token(&self) -> Result<u32, TemporalCircuitStop> {
        let session =
            self.session
                .as_ref()
                .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                    "typed workspace session is not initialized",
                ))?;
        session
            .vocabulary
            .address_token(session.holes.address())
            .map_err(TemporalCircuitStop::MalformedContextOrPolicy)
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
        })
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
        let plan_indices = plan_projection(cue)?;
        for (mixed_index, &plan_index) in plan_indices.iter().enumerate() {
            winners[mixed_index] = self.plan_winners[plan_index];
        }
        for mixed_index in 0..SEMANTIC_CUE_WINNERS_IN_MIX {
            winners[PLAN_WINNERS_IN_MIX + mixed_index] = cue_winner_context[mixed_index];
        }
        for mixed_index in 0..ADDRESS_CUE_WINNERS_IN_MIX {
            winners[PLAN_WINNERS_IN_MIX + SEMANTIC_CUE_WINNERS_IN_MIX + mixed_index] =
                address_winner_context[mixed_index];
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

fn plan_projection(
    cue: TypedHoleCue,
) -> Result<&'static [usize; PLAN_WINNERS_IN_MIX], TemporalCircuitStop> {
    match cue {
        TypedHoleCue::Expression(TypedValue::ListInt)
        | TypedHoleCue::Expression(TypedValue::IntToInt)
        | TypedHoleCue::Eos => Ok(&SCAFFOLD_HEAD_PLAN_INDICES),
        TypedHoleCue::Expression(TypedValue::Int | TypedValue::Bool)
        | TypedHoleCue::LiteralSign
        | TypedHoleCue::LiteralFirstDigit
        | TypedHoleCue::LiteralFirstNonzeroDigit
        | TypedHoleCue::LiteralDigitOrEnd
        | TypedHoleCue::LiteralEnd => Ok(&OPERATION_HEAD_PLAN_INDICES),
        TypedHoleCue::Terminated => Err(TemporalCircuitStop::MalformedContextOrPolicy(
            "terminated typed state has no structural plan projection",
        )),
    }
}

fn context_from_winners(
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
