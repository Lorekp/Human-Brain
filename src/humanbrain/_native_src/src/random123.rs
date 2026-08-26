//! Shared portable Random123 primitives for native deterministic addressing.

const PHILOX_M2X64_0: u64 = 0xD2B7_4407_B1CE_6E93;
pub(crate) const PHILOX_W64_0: u64 = 0x9E37_79B9_7F4A_7C15;
const PHILOX_ROUNDS: usize = 10;

/// Return Random123 Philox-2x64-10 words in numeric `v[0], v[1]` order.
///
/// Numeric word order is independent of host byte order and matches the
/// project's Python oracle and the Random123 known-answer vectors.
#[inline]
pub(crate) fn philox2x64_10(counter_0: u64, counter_1: u64, key: u64) -> (u64, u64) {
    let mut left = counter_0;
    let mut right = counter_1;
    let mut round_key = key;
    for round_index in 0..PHILOX_ROUNDS {
        let product = (PHILOX_M2X64_0 as u128) * (left as u128);
        let high = (product >> 64) as u64;
        let low = product as u64;
        left = high ^ round_key ^ right;
        right = low;
        if round_index != PHILOX_ROUNDS - 1 {
            round_key = round_key.wrapping_add(PHILOX_W64_0);
        }
    }
    (left, right)
}

/// Map one uniform 64-bit word into `0..bound` without modulo bias.
///
/// `None` identifies the exact Lemire low-tail rejection region; callers must
/// draw another independent word. A zero `bound` is invalid and returns `None`.
#[inline]
pub(crate) fn lemire_candidate(word: u64, bound: u64) -> Option<u64> {
    if bound == 0 {
        return None;
    }
    let rejection_threshold = bound.wrapping_neg() % bound;
    let product = (word as u128) * (bound as u128);
    ((product as u64) >= rejection_threshold).then_some((product >> 64) as u64)
}

fn bounded_index_with_words<F>(bound: u64, mut word_for_attempt: F) -> Option<u64>
where
    F: FnMut(u32) -> u64,
{
    if bound == 0 {
        return None;
    }
    for attempt in 0..=u32::MAX {
        if let Some(index) = lemire_candidate(word_for_attempt(attempt), bound) {
            return Some(index);
        }
    }
    None
}

pub(crate) fn bounded_philox_index(
    bound: u64,
    key: u64,
    learning_step: u64,
    draw: u32,
) -> Option<u64> {
    bounded_index_with_words(bound, |attempt| {
        let counter_1 = ((attempt as u64) << 32) | draw as u64;
        philox2x64_10(learning_step, counter_1, key).0
    })
}

/// Sample sorted positions without replacement using the frozen contact scheme.
///
/// The caller supplies reusable scratch covering the population so the future
/// temporal mutation path can remain allocation-free. `None` denotes invalid
/// dimensions or exhaustion of the exact 32-bit rejection-attempt address.
pub(crate) fn sample_without_replacement(
    population_size: usize,
    sample_size: usize,
    key: u64,
    learning_step: u64,
    scratch: &mut [u64],
) -> Option<&[u64]> {
    if sample_size > population_size
        || population_size > scratch.len()
        || (sample_size as u64) >= (1u64 << 32)
    {
        return None;
    }
    for (position, value) in scratch.iter_mut().take(population_size).enumerate() {
        *value = position as u64;
    }
    for draw in 0..sample_size {
        let remaining = (population_size - draw) as u64;
        let offset = bounded_philox_index(remaining, key, learning_step, draw as u32)? as usize;
        scratch.swap(draw, draw + offset);
    }
    scratch[..sample_size].sort_unstable();
    Some(&scratch[..sample_size])
}

#[cfg(test)]
mod tests {
    use super::{bounded_index_with_words, lemire_candidate, philox2x64_10};

    #[test]
    fn philox2x64_10_matches_random123_known_answers() {
        assert_eq!(
            philox2x64_10(0, 0, 0),
            (0xCA00_A045_9843_D731, 0x66C2_4222_C9A8_45B5),
        );
        assert_eq!(
            philox2x64_10(
                0x243F_6A88_85A3_08D3,
                0x1319_8A2E_0370_7344,
                0xA409_3822_299F_31D0,
            ),
            (0x0A5E_742C_2997_341C, 0xB0F8_83D3_8000_DE5D),
        );
    }

    #[test]
    fn lemire_mapping_rejects_the_biased_low_tail_exactly() {
        assert_eq!(lemire_candidate(u64::MAX, 0), None);
        assert_eq!(lemire_candidate(0, 3), None);
        assert_eq!(lemire_candidate(u64::MAX, 3), Some(2));
    }

    #[test]
    fn bounded_mapping_advances_the_exact_rejection_attempt() {
        let mut attempts = Vec::new();
        let words = [0, u64::MAX];
        let selected = bounded_index_with_words(3, |attempt| {
            attempts.push(attempt);
            words[attempt as usize]
        });

        assert_eq!(selected, Some(2));
        assert_eq!(attempts, [0, 1]);
    }
}
