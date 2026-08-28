//! Fresh-circuit typed workspace prototype.
//!
//! The authoritative temporal store still owns every learned association. This
//! state only retains one fitted plan assembly and reconstructs the exact
//! BoundWinners source/relay context used at a structural hole.

use super::{TemporalCircuitStop, TemporalTransientState, ACTIVE_COLUMNS, CONTEXT_CELLS};

pub(super) const PLAN_WINNERS_IN_MIX: usize = 5;
pub(super) const CUE_WINNERS_IN_MIX: usize = ACTIVE_COLUMNS - PLAN_WINNERS_IN_MIX;

#[derive(Clone, Copy)]
pub(super) struct TypedWorkspaceState {
    plan_winners: [u32; ACTIVE_COLUMNS],
    plan_ready: bool,
}

impl TypedWorkspaceState {
    pub(super) const EMPTY: Self = Self {
        plan_winners: [0; ACTIVE_COLUMNS],
        plan_ready: false,
    };

    pub(super) fn clear(&mut self) {
        *self = Self::EMPTY;
    }

    pub(super) fn retain_plan(&mut self, winner_context: [u32; CONTEXT_CELLS]) {
        self.plan_winners
            .copy_from_slice(&winner_context[..ACTIVE_COLUMNS]);
        self.plan_ready = true;
    }

    pub(super) const fn is_ready(&self) -> bool {
        self.plan_ready
    }

    pub(super) const fn plan_winners(&self) -> Option<[u32; ACTIVE_COLUMNS]> {
        if self.plan_ready {
            Some(self.plan_winners)
        } else {
            None
        }
    }

    pub(super) fn plan_transient(
        &self,
        relay_offset: u32,
        event_count: u64,
    ) -> Result<TemporalTransientState, TemporalCircuitStop> {
        if !self.plan_ready {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "typed plan assembly is not committed",
            ));
        }
        context_from_winners(self.plan_winners, relay_offset, event_count)
    }

    pub(super) fn mixed_transient(
        &self,
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
        winners[..PLAN_WINNERS_IN_MIX]
            .copy_from_slice(&self.plan_winners[..PLAN_WINNERS_IN_MIX]);
        winners[PLAN_WINNERS_IN_MIX..].copy_from_slice(
            &cue_winner_context[..CUE_WINNERS_IN_MIX],
        );
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
