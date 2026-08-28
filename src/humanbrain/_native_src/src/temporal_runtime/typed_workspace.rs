//! Fresh-circuit typed workspace prototype.
//!
//! The authoritative temporal store still owns every learned association. This
//! state only retains one fitted plan assembly and reconstructs the exact
//! BoundWinners source/relay context used at a structural hole.

use super::{
    typed_hole_state::{SurfaceKind, TypedHoleCue, TypedHoleRole, TypedHoleState, TypedValue},
    TemporalCircuitStop, TemporalStepMode, TemporalTransientState, ACTIVE_COLUMNS, CONTEXT_CELLS,
    NONE,
};

pub(super) const PLAN_WINNERS_IN_MIX: usize = 5;
pub(super) const CUE_WINNERS_IN_MIX: usize = ACTIVE_COLUMNS - PLAN_WINNERS_IN_MIX;
const SCAFFOLD_HEAD_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [0, 1, 2, 3, 4];
const OPERATION_HEAD_PLAN_INDICES: [usize; PLAN_WINNERS_IN_MIX] = [5, 6, 7, 8, 9];
// Local support for the three current literal roles, not a general depth/path
// address. The projected cue sets keep each pair below the physical active
// threshold while preserving the same transferable operation-plan component.
const LITERAL_HEAD_CUE_INDICES: [usize; CUE_WINNERS_IN_MIX] = [0, 2, 4, 6, 8, 10];
const LITERAL_CHILD0_CUE_INDICES: [usize; CUE_WINNERS_IN_MIX] = [0, 1, 2, 3, 4, 5];
const LITERAL_CHILD1_CUE_INDICES: [usize; CUE_WINNERS_IN_MIX] = [5, 6, 7, 8, 9, 10];
pub(crate) const TYPED_CUE_TOKEN_COUNT: usize = 17;
pub(crate) const TYPED_SURFACE_TOKEN_COUNT: usize = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TypedWorkspaceVocabulary {
    pub(crate) plan_commit_token_id: u32,
    pub(crate) cue_token_ids: [u32; TYPED_CUE_TOKEN_COUNT],
    pub(crate) surface_token_ids: [u32; TYPED_SURFACE_TOKEN_COUNT],
}

impl TypedWorkspaceVocabulary {
    pub(crate) fn validate(self) -> Result<(), &'static str> {
        let mut ids = [0u32; 1 + TYPED_CUE_TOKEN_COUNT + TYPED_SURFACE_TOKEN_COUNT];
        ids[0] = self.plan_commit_token_id;
        ids[1..1 + TYPED_CUE_TOKEN_COUNT].copy_from_slice(&self.cue_token_ids);
        ids[1 + TYPED_CUE_TOKEN_COUNT..].copy_from_slice(&self.surface_token_ids);
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
        };
        let index = match cue {
            TypedHoleCue::Expression(TypedValue::Int) => role_offset,
            TypedHoleCue::Expression(TypedValue::ListInt) => 3 + role_offset,
            TypedHoleCue::Expression(TypedValue::IntToInt) => 6 + role_offset,
            TypedHoleCue::LiteralSign => 9,
            TypedHoleCue::LiteralFirstDigit => 10,
            TypedHoleCue::LiteralFirstNonzeroDigit => 11,
            TypedHoleCue::LiteralDigitOrEnd => 12,
            TypedHoleCue::LiteralEnd => 13,
            TypedHoleCue::Eos => 14,
            TypedHoleCue::Terminated => return None,
        };
        Some(self.cue_token_ids[index])
    }

    pub(crate) fn operation_component_token(self, slot_ready: bool) -> u32 {
        self.cue_token_ids[if slot_ready { 16 } else { 15 }]
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
        role: TypedHoleRole,
        cue_winner_context: [u32; CONTEXT_CELLS],
        relay_offset: u32,
        event_count: u64,
    ) -> Result<TemporalTransientState, TemporalCircuitStop> {
        if !self.plan_ready {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed plan assembly is not committed",
            ));
        }
        let mut winners = [0u32; ACTIVE_COLUMNS];
        let plan_indices = plan_projection(cue, role)?;
        for (mixed_index, &plan_index) in plan_indices.iter().enumerate() {
            winners[mixed_index] = self.plan_winners[plan_index];
        }
        for (mixed_index, &cue_index) in cue_projection(cue, role).iter().enumerate() {
            winners[PLAN_WINNERS_IN_MIX + mixed_index] = cue_winner_context[cue_index];
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
    _role: TypedHoleRole,
) -> Result<&'static [usize; PLAN_WINNERS_IN_MIX], TemporalCircuitStop> {
    match cue {
        TypedHoleCue::Expression(TypedValue::ListInt)
        | TypedHoleCue::Expression(TypedValue::IntToInt)
        | TypedHoleCue::Eos => Ok(&SCAFFOLD_HEAD_PLAN_INDICES),
        TypedHoleCue::Expression(TypedValue::Int)
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

fn cue_projection(cue: TypedHoleCue, role: TypedHoleRole) -> &'static [usize; CUE_WINNERS_IN_MIX] {
    if cue != TypedHoleCue::LiteralSign {
        return &LITERAL_CHILD0_CUE_INDICES;
    }
    match role {
        TypedHoleRole::Head => &LITERAL_HEAD_CUE_INDICES,
        TypedHoleRole::Child0 => &LITERAL_CHILD0_CUE_INDICES,
        TypedHoleRole::Child1 => &LITERAL_CHILD1_CUE_INDICES,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_fixed_operation_projection_aliases_root_and_both_add_operand_holes() {
        let vocabulary = TypedWorkspaceVocabulary {
            plan_commit_token_id: 1,
            cue_token_ids: std::array::from_fn(|index| 10 + index as u32),
            surface_token_ids: std::array::from_fn(|index| 20 + index as u32),
        };
        let mut workspace = TypedWorkspaceState::EMPTY;
        let mut plan_context = [0u32; CONTEXT_CELLS];
        plan_context[..ACTIVE_COLUMNS].copy_from_slice(
            &std::array::from_fn::<_, ACTIVE_COLUMNS, _>(|index| 100 + index as u32),
        );
        workspace.retain_plan_components(plan_context, plan_context);
        workspace
            .begin_session(
                vocabulary,
                TypedValue::Int,
                TypedValue::Int,
                TemporalStepMode::Learn,
            )
            .unwrap();

        let mut cue_context = [0u32; CONTEXT_CELLS];
        cue_context[..ACTIVE_COLUMNS].copy_from_slice(
            &std::array::from_fn::<_, ACTIVE_COLUMNS, _>(|index| 200 + index as u32),
        );

        let root_cue = workspace.current_cue().unwrap();
        let root_role = workspace.current_role().unwrap();
        let rejected_root = workspace
            .mixed_transient(root_cue, TypedHoleRole::Head, cue_context, 1_000, 1)
            .unwrap();
        let (root_admitted, root_count) = workspace.admitted_tokens().unwrap();
        assert!(root_admitted[..root_count].contains(&vocabulary.surface_token_ids[0]));
        assert!(root_admitted[..root_count].contains(&vocabulary.surface_token_ids[4]));
        assert!(root_admitted[..root_count].contains(&vocabulary.surface_token_ids[5]));

        workspace
            .advance_token(vocabulary.surface_token_ids[0])
            .unwrap();
        let first_operand_cue = workspace.current_cue().unwrap();
        let first_operand_role = workspace.current_role().unwrap();
        let rejected_first_operand = workspace
            .mixed_transient(
                first_operand_cue,
                TypedHoleRole::Head,
                cue_context,
                1_000,
                1,
            )
            .unwrap();

        workspace
            .advance_token(vocabulary.surface_token_ids[4])
            .unwrap();
        let second_operand_cue = workspace.current_cue().unwrap();
        let second_operand_role = workspace.current_role().unwrap();
        let rejected_second_operand = workspace
            .mixed_transient(
                second_operand_cue,
                TypedHoleRole::Head,
                cue_context,
                1_000,
                1,
            )
            .unwrap();

        assert_eq!(root_cue, TypedHoleCue::Expression(TypedValue::Int));
        assert_eq!(first_operand_cue, root_cue);
        assert_eq!(second_operand_cue, root_cue);
        assert_eq!(root_role, TypedHoleRole::Head);
        assert_eq!(first_operand_role, TypedHoleRole::Child0);
        assert_eq!(second_operand_role, TypedHoleRole::Child1);
        assert_eq!(
            rejected_first_operand.previous_context,
            rejected_root.previous_context
        );
        assert_eq!(
            rejected_second_operand.previous_context,
            rejected_root.previous_context
        );
        assert_ne!(
            vocabulary.cue_token(root_cue, root_role),
            vocabulary.cue_token(first_operand_cue, first_operand_role)
        );
        assert_ne!(
            vocabulary.cue_token(first_operand_cue, first_operand_role),
            vocabulary.cue_token(second_operand_cue, second_operand_role)
        );
    }
}
