// STAGE B DRAFT (applied procedurally via patch after baseline golden capture)
// Replaces: HashMap interval tracking -> sorted IntervalLedger; per-event
// O(R) sorted inserts -> per-step staged buffer + fused update/compact pass;
// unconditional retain -> streaming eviction.

// ---------- IntervalLedger (replaces HashMap<u64,u8> interval_ids) ----------
struct IntervalLedger {
    gids: Vec<u64>, // ascending unique
    flags: Vec<u8>, // parallel accumulated flag bits
}

impl IntervalLedger {
    fn try_new(capacity: usize) -> Result<Self, u32> {
        let mut gids = Vec::new();
        gids.try_reserve_exact(capacity).map_err(|_| PROCEDURAL_ALLOCATION)?;
        let mut flags = Vec::new();
        flags.try_reserve_exact(capacity).map_err(|_| PROCEDURAL_ALLOCATION)?;
        Ok(Self { gids, flags })
    }

    fn clear(&mut self) {
        self.gids.clear();
        self.flags.clear();
    }

    /// Exact single-GID mark preserving the previous fail-stop timing: the
    /// first new GID beyond the logical capacity raises immediately.
    fn mark_single(&mut self, gid: u64, flag: u8, capacity: usize) -> Result<(), u32> {
        match self.gids.binary_search(&gid) {
            Ok(index) => {
                self.flags[index] |= flag;
                Ok(())
            }
            Err(insertion) => {
                if self.gids.len() >= capacity || self.gids.len() >= self.gids.capacity() {
                    return Err(PROCEDURAL_DISTINCT_CAPACITY);
                }
                self.gids.insert(insertion, gid);
                self.flags.insert(insertion, flag);
                Ok(())
            }
        }
    }

    /// True when an ascending unique run certainly fits without new growth
    /// beyond capacity (cheap upper bound first, exact walk only if needed).
    fn can_fit_run(&self, run: &[u64], capacity: usize) -> bool {
        let room = capacity.saturating_sub(self.gids.len());
        let reserved_room = self.gids.capacity().saturating_sub(self.gids.len());
        let room = room.min(reserved_room);
        if run.len() <= room {
            return true;
        }
        let mut new_count = 0usize;
        let mut cursor = 0usize;
        for gid in run {
            cursor += self.gids[cursor..].partition_point(|existing| *existing < *gid);
            if cursor == self.gids.len() || self.gids[cursor] != *gid {
                new_count += 1;
                if new_count > room {
                    return false;
                }
            }
        }
        true
    }

    /// Union-merge an ascending unique run, OR-ing `flag` into members.
    /// `scratch` must be a distinct ledger with equal reservation.
    fn merge_run_in_place(&mut self, run: &[u64], flag: u8, scratch: &mut Self) {
        scratch.clear();
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.gids.len() && j < run.len() {
            match self.gids[i].cmp(&run[j]) {
                Ordering::Less => {
                    scratch.gids.push(self.gids[i]);
                    scratch.flags.push(self.flags[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    scratch.gids.push(run[j]);
                    scratch.flags.push(flag);
                    j += 1;
                }
                Ordering::Equal => {
                    scratch.gids.push(self.gids[i]);
                    scratch.flags.push(self.flags[i] | flag);
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < self.gids.len() {
            scratch.gids.push(self.gids[i]);
            scratch.flags.push(self.flags[i]);
            i += 1;
        }
        while j < run.len() {
            scratch.gids.push(run[j]);
            scratch.flags.push(flag);
            j += 1;
        }
        std::mem::swap(self, scratch);
    }

    fn count_flags(&self, mask: u8) -> usize {
        self.flags.iter().filter(|flags| **flags & mask != 0).count()
    }
}

// ---------- RuntimeScratch changes ----------
struct RuntimeScratch {
    submitted_events: Vec<ProceduralEventRecord>,
    event_work: Vec<ProceduralEventRecord>,
    step_spikes: Vec<u64>,
    row_targets: Vec<u64>,
    row_used: HashSet<u64>,
    interval_ledger: IntervalLedger,
    ledger_scratch: IntervalLedger,
    gid_run: Vec<u64>,
}

fn try_new(config) {
    ...
    interval_ledger: IntervalLedger::try_new(config.interval_distinct_capacity)?,
    ledger_scratch: IntervalLedger::try_new(config.interval_distinct_capacity)?,
    gid_run: reserved_vec(config.resident_capacity)?,
    ...
}

fn payload_bytes(&self) -> usize {
    ... existing terms ...
    + self.interval_ledger.gids.capacity() * size_of::<u64>()
    + self.interval_ledger.flags.capacity() * size_of::<u8>()
    + self.ledger_scratch.gids.capacity() * size_of::<u64>()
    + self.ledger_scratch.flags.capacity() * size_of::<u8>()
    + self.gid_run.capacity() * size_of::<u64>()
}
// mark_interval helper removed; callers use ledger.mark_single / runs.

// ---------- ProceduralRuntime ----------
// new field: staged: Vec<CellState>  (reserved min(event_capacity, resident_capacity))
// helper:
fn total_residents(&self) -> usize { self.residents.len() + self.staged.len() }

// apply_event: dual lookup (main binary search, staged linear), inserts go to
// staged; capacity checks + mark_single(SEEN_RESIDENT) keep original order.

// advance():
//   entry marks: clear ledger; if can_fit_run(main gids) merge_run_in_place
//   else legacy per-cell mark_single loop.
//   delivery: resident_peak.max(total_residents())
//   cell_steps uses total_residents()
//   stepping fast path (pre-checked): fused dual-cursor update+compact pass,
//   then merge UPDATED run and SPIKING run into ledger.
//   stepping fallback (capacity would trip): exact legacy per-cell loop.
