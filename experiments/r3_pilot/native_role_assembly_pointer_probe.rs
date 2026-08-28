//! PROTOTYPE ONLY: native recurrent role-addressed assembly pointer.
//!
//! This file is intentionally outside Cargo auto-discovery.  It directly
//! includes the native store and fitted-race modules so the experiment can use
//! crate-private construction seams without widening the production ABI.

#[path = "../../src/humanbrain/_native_src/src/temporal.rs"]
mod temporal;
#[path = "../../src/humanbrain/_native_src/src/temporal_physics.rs"]
mod temporal_physics;

use std::fmt::Write as _;

use temporal::{TemporalOwnerDiagnostic, TemporalSegmentRow, TemporalStore, TemporalStoreConfig};
use temporal_physics::{
    coupled_first_spike_race, BasketDrive, CandidateClass, CandidateDrive, CoupledRaceScratch,
    TEMPORAL_PHYSICS_IDENTITY,
};

const K: usize = 11;
const CELLS_PER_COLUMN: u32 = 8;
const OP_FIRST_COLUMN: u32 = 0;
const ROLE_FIRST_COLUMN: u32 = 11;
const POINTER_FIRST_COLUMN: u32 = 22;
const COLUMN_COUNT: u32 = 33;
const OWNER_COUNT: usize = (COLUMN_COUNT * CELLS_PER_COLUMN) as usize;
const NONE: u32 = u32::MAX;
const CONNECTED_TICKS: u16 = 250;
const LEARNED_TICKS: u16 = 255;
const THETA_MATCH: u8 = 16;
const THETA_ACTIVE: u8 = 18;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WinnerSet {
    owners: [u32; K],
    ticks: [u16; K],
}

struct TrainedProbe {
    store: TemporalStore,
    operation: WinnerSet,
    role: WinnerSet,
    pointer: WinnerSet,
    pointer_replay: WinnerSet,
    cue_repeat_segment_count_before: u32,
    cue_repeat_segment_count_after: u32,
}

fn empty_diagnostic() -> TemporalOwnerDiagnostic {
    TemporalOwnerDiagnostic {
        owner: NONE,
        best_matching_segment: NONE,
        best_active_segment: NONE,
        matching_potential_contacts: 0,
        matching_connected_contacts: 0,
        active_potential_contacts: 0,
        active_connected_contacts: 0,
    }
}

fn store() -> TemporalStore {
    TemporalStore::new_runtime_owned(TemporalStoreConfig {
        source_count: OWNER_COUNT,
        owner_count: OWNER_COUNT,
        segment_capacity: 33,
        source_gid_base: 0,
        connected_permanence_ticks: CONNECTED_TICKS,
        // Reuse the already-established typed-workspace probe arm.  With two
        // physical contacts per logical pair, nine of ten recurrent peers are
        // sufficient to keep a clamped partial assembly active.
        theta_match: THETA_MATCH,
        theta_active: THETA_ACTIVE,
        padding_zero: 0,
    })
    .expect("bounded native temporal store construction")
}

fn owner(column: u32, slot: u32) -> u32 {
    column * CELLS_PER_COLUMN + slot
}

fn diagnostic_for(
    diagnostics: &[TemporalOwnerDiagnostic],
    candidate_owner: u32,
) -> Option<&TemporalOwnerDiagnostic> {
    diagnostics
        .binary_search_by_key(&candidate_owner, |diagnostic| diagnostic.owner)
        .ok()
        .map(|index| &diagnostics[index])
}

/// Fit one owner per sensory column through the production native GLIF race.
///
/// The probe freezes homogeneous threshold offsets at zero; it never changes
/// thresholds, currents, quorums, or backoff after observing an outcome.
fn fit_token(store: &mut TemporalStore, context: &[u32], first_column: u32) -> WinnerSet {
    let mut diagnostics = vec![empty_diagnostic(); OWNER_COUNT];
    let summary = store
        .recognize_runtime_context(context, &mut diagnostics)
        .expect("native contextual recognition");
    let diagnostics = &diagnostics[..summary.owner_result_count];
    let mut scratch = CoupledRaceScratch::try_new(CELLS_PER_COLUMN as usize)
        .expect("fixed eight-cell race scratch");
    let mut winners = WinnerSet {
        owners: [NONE; K],
        ticks: [0; K],
    };

    for index in 0..K {
        let column = first_column + index as u32;
        let mut candidates = Vec::with_capacity(CELLS_PER_COLUMN as usize);
        let mut categorized = [false; CELLS_PER_COLUMN as usize];
        let mut categorized_count = 0;
        for slot in 0..CELLS_PER_COLUMN {
            let candidate_owner = owner(column, slot);
            let load = store
                .runtime_owner_load(candidate_owner)
                .expect("candidate owner is inside the native store");
            let current_pa = match diagnostic_for(diagnostics, candidate_owner) {
                Some(diagnostic) if diagnostic.best_active_segment != NONE => {
                    categorized[slot as usize] = true;
                    categorized_count += 1;
                    CandidateClass::Active { load }.current_pa()
                }
                Some(diagnostic) if diagnostic.best_matching_segment != NONE => {
                    categorized[slot as usize] = true;
                    categorized_count += 1;
                    CandidateClass::Matching { load }.current_pa()
                }
                _ => 0.0,
            };
            candidates.push(CandidateDrive {
                cell_id: candidate_owner,
                current_pa,
                threshold_offset_mv: 0.0,
            });
        }

        // Mirror the runtime's crucial load-first eligibility rule before
        // applying any novel rank.  This makes the recurrence-off ablation an
        // honest election, rather than a hand-picked competing owner.
        let novel_limit = CELLS_PER_COLUMN as usize - categorized_count;
        let min_load = (0..CELLS_PER_COLUMN as usize)
            .filter(|slot| !categorized[*slot])
            .map(|slot| {
                store
                    .runtime_owner_load(candidates[slot].cell_id)
                    .expect("candidate owner load")
            })
            .min();
        let mut novel_rank = 0u8;
        if let Some(min_load) = min_load {
            for slot in 0..CELLS_PER_COLUMN as usize {
                if novel_rank as usize == novel_limit {
                    break;
                }
                let candidate_owner = candidates[slot].cell_id;
                let load = store
                    .runtime_owner_load(candidate_owner)
                    .expect("candidate owner load");
                if !categorized[slot] && load == min_load {
                    candidates[slot].current_pa = CandidateClass::Novel {
                        rank: novel_rank,
                        load,
                    }
                    .current_pa();
                    novel_rank += 1;
                }
            }
        }
        let race = coupled_first_spike_race(
            &candidates,
            BasketDrive {
                cell_id: OWNER_COUNT as u32 + column,
                current_pa: TEMPORAL_PHYSICS_IDENTITY.basket_current_pa,
                threshold_offset_mv: 0.0,
                inhibition_pa: TEMPORAL_PHYSICS_IDENTITY.basket_inhibition_pa,
            },
            &mut scratch,
        )
        .expect("one fitted pyramidal winner and causal basket inhibition");
        winners.owners[index] = race.winner.cell_id;
        winners.ticks[index] = race.winner.tick;
    }
    winners
}

fn canonical_row(mut sources: [u32; K - 1]) -> TemporalSegmentRow {
    sources.sort_unstable();
    assert!(sources.windows(2).all(|pair| pair[0] < pair[1]));
    TemporalSegmentRow {
        normalized_sources: sources,
        permanence_ticks: [LEARNED_TICKS; K - 1],
        live_pair_count: (K - 1) as u8,
    }
}

fn first_ten(values: &[u32; K]) -> [u32; K - 1] {
    let mut result = [NONE; K - 1];
    result.copy_from_slice(&values[..K - 1]);
    result
}

fn append_rows(
    store: &mut TemporalStore,
    owners: &[u32; K],
    rows: &[TemporalSegmentRow; K],
    max_segments_per_cell: u8,
) {
    let receipt = store
        .preflight_runtime_mutation_backing(owners, max_segments_per_cell)
        .expect("aggregate native append preflight");
    assert_eq!(receipt.prospective_append_count(), K as u32);
    let first = receipt.first_append_segment_id();
    for (&owner, row) in owners.iter().zip(rows) {
        store
            .preflight_runtime_canonical_append(owner, row)
            .expect("canonical native segment row");
    }
    for (index, (&owner, row)) in owners.iter().zip(rows).enumerate() {
        assert_eq!(
            store.commit_prevalidated_runtime_canonical_append(owner, row),
            first + index as u32
        );
    }
}

fn repeated_rows(sources: [u32; K - 1]) -> [TemporalSegmentRow; K] {
    [canonical_row(sources); K]
}

fn recurrent_rows(pointer: &[u32; K]) -> [TemporalSegmentRow; K] {
    std::array::from_fn(|excluded| {
        let mut sources = [NONE; K - 1];
        let mut cursor = 0;
        for (index, source) in pointer.iter().copied().enumerate() {
            if index != excluded {
                sources[cursor] = source;
                cursor += 1;
            }
        }
        assert_eq!(cursor, K - 1);
        canonical_row(sources)
    })
}

fn train(include_recurrence: bool) -> TrainedProbe {
    let mut store = store();

    let operation = fit_token(&mut store, &[], OP_FIRST_COLUMN);
    let expected_operation: [u32; K] =
        std::array::from_fn(|index| owner(OP_FIRST_COLUMN + index as u32, 0));
    assert_eq!(operation.owners, expected_operation);

    // The first contextual role cue recruits fitted winners.  Only after that
    // race do we append the rows which make this exact owner identity durable.
    let role = fit_token(&mut store, &operation.owners, ROLE_FIRST_COLUMN);
    let role_owners = role.owners;
    let role_rows = repeated_rows(first_ten(&operation.owners));
    append_rows(&mut store, &role_owners, &role_rows, 1);
    let role_replay = fit_token(&mut store, &operation.owners, ROLE_FIRST_COLUMN);
    assert_eq!(role_replay.owners, role.owners);

    // Likewise, the operation+relative-role context recruits the pointer
    // owners before any pointer row exists.  Replaying the cue below must use
    // the learned native rows and must not grow the store.
    let pointer = fit_token(&mut store, &role.owners, POINTER_FIRST_COLUMN);
    let pointer_owners = pointer.owners;
    let pointer_rows = repeated_rows(first_ten(&role.owners));
    append_rows(&mut store, &pointer_owners, &pointer_rows, 1);
    let cue_repeat_segment_count_before = store.runtime_segment_count();
    let pointer_replay = fit_token(&mut store, &role.owners, POINTER_FIRST_COLUMN);
    let cue_repeat_segment_count_after = store.runtime_segment_count();
    assert_eq!(pointer_replay.owners, pointer.owners);
    assert_eq!(
        cue_repeat_segment_count_before,
        cue_repeat_segment_count_after
    );

    if include_recurrence {
        let rows = recurrent_rows(&pointer.owners);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row.live_pair_count, (K - 1) as u8);
            assert!(!row.normalized_sources.contains(&pointer.owners[index]));
        }
        append_rows(&mut store, &pointer.owners, &rows, 2);
        assert_eq!(store.runtime_segment_count(), 33);
    } else {
        assert_eq!(store.runtime_segment_count(), 22);
    }

    TrainedProbe {
        store,
        operation,
        role,
        pointer,
        pointer_replay,
        cue_repeat_segment_count_before,
        cue_repeat_segment_count_after,
    }
}

fn json_array(values: &[u32]) -> String {
    let mut output = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(&mut output, "{value}").expect("write into String");
    }
    output.push(']');
    output
}

fn main() {
    let mut recurrent = train(true);
    let mut ablated = train(false);
    assert_eq!(recurrent.operation, ablated.operation);
    assert_eq!(recurrent.role, ablated.role);
    assert_eq!(recurrent.pointer, ablated.pointer);

    // Deliberately omit owner 10 and replace it with the unused slot-one cell
    // in that same pointer column. The cue therefore contains exactly ten of
    // the eleven learned pointer owners.
    let mut partial = recurrent.pointer.owners;
    partial[K - 1] = owner(POINTER_FIRST_COLUMN + (K - 1) as u32, 1);
    assert_eq!(
        partial
            .iter()
            .filter(|source| recurrent.pointer.owners.contains(source))
            .count(),
        K - 1
    );

    let segment_count_before = recurrent.store.runtime_segment_count();
    let completed = fit_token(&mut recurrent.store, &partial, POINTER_FIRST_COLUMN);
    let settled = fit_token(
        &mut recurrent.store,
        &completed.owners,
        POINTER_FIRST_COLUMN,
    );
    let repeated = fit_token(&mut recurrent.store, &partial, POINTER_FIRST_COLUMN);
    let segment_count_after = recurrent.store.runtime_segment_count();
    assert_eq!(completed.owners, recurrent.pointer.owners);
    assert_eq!(settled.owners, recurrent.pointer.owners);
    assert_eq!(repeated.owners, recurrent.pointer.owners);
    assert_eq!(segment_count_before, segment_count_after);

    let ablated_before = ablated.store.runtime_segment_count();
    let ablated_completion = fit_token(&mut ablated.store, &partial, POINTER_FIRST_COLUMN);
    let ablated_after = ablated.store.runtime_segment_count();
    let default_novel: [u32; K] =
        std::array::from_fn(|index| owner(POINTER_FIRST_COLUMN + index as u32, 1));
    assert_eq!(ablated_completion.owners, default_novel);
    assert_ne!(ablated_completion.owners, recurrent.pointer.owners);
    assert_eq!(ablated_before, ablated_after);

    println!(
        concat!(
            "{{",
            "\"experiment\":\"native-recurrent-role-assembly-pointer-prototype\",",
            "\"outcome\":\"PASS_GENUINE_RECURRENT_K11_POINTER\",",
            "\"assembly_k\":{k},",
            "\"theta_match\":{theta_match},",
            "\"theta_active\":{theta_active},",
            "\"threshold_offset_mv\":0.0,",
            "\"operation_owners\":{operation},",
            "\"role_owners\":{role},",
            "\"pointer_owners\":{pointer},",
            "\"pointer_cue_replay_owners\":{pointer_replay},",
            "\"cue_repeat_segment_count_before\":{cue_before},",
            "\"cue_repeat_segment_count_after\":{cue_after},",
            "\"partial_pointer_cue\":{partial},",
            "\"partial_overlap\":{overlap},",
            "\"completed_owners\":{completed},",
            "\"settled_owners\":{settled},",
            "\"repeated_partial_owners\":{repeated},",
            "\"ablated_completed_owners\":{ablated},",
            "\"recurrent_rows\":{recurrent_rows},",
            "\"recurrent_live_pairs_per_row\":{live_pairs},",
            "\"recurrent_self_edges\":0,",
            "\"segment_count_before_recall\":{before},",
            "\"segment_count_after_recall\":{after},",
            "\"ablated_segment_count_before_recall\":{ablated_before},",
            "\"ablated_segment_count_after_recall\":{ablated_after},",
            "\"recurrence_changes_outcome\":true,",
            "\"target_attempt_count\":0,",
            "\"claim_boundary\":\"native store recognition plus fitted GLIF races prove recurrent partial completion only; no TypedWorkspace injection, token generation, target, held-out, M0, or C1\"",
            "}}"
        ),
        k = K,
        theta_match = THETA_MATCH,
        theta_active = THETA_ACTIVE,
        operation = json_array(&recurrent.operation.owners),
        role = json_array(&recurrent.role.owners),
        pointer = json_array(&recurrent.pointer.owners),
        pointer_replay = json_array(&recurrent.pointer_replay.owners),
        cue_before = recurrent.cue_repeat_segment_count_before,
        cue_after = recurrent.cue_repeat_segment_count_after,
        partial = json_array(&partial),
        overlap = K - 1,
        completed = json_array(&completed.owners),
        settled = json_array(&settled.owners),
        repeated = json_array(&repeated.owners),
        ablated = json_array(&ablated_completion.owners),
        recurrent_rows = K,
        live_pairs = K - 1,
        before = segment_count_before,
        after = segment_count_after,
        ablated_before = ablated_before,
        ablated_after = ablated_after,
    );
}
