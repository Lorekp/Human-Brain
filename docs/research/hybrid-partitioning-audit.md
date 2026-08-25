# Hybrid CPU/GPU partitioning — feasibility audit (decision: DO NOT BUILD NOW)

Status: decision-grade audit adopted 2026-08-24. Full report in delegation
transcript deleg_c7185fce; this doc carries the operative content.

## Verdict

Cell-level hybrid is feasible but NARROW, and the narrow window is not our
mission regime. For the text-like sparse regime the CPU event-proportional
path alone wins or ties at every size. Hybrid should NOT be built before
the two named CPU seams land (`read_cells` targeted read; indexed staged
membership) — those seams are prerequisites for hybrid anyway and may erase
its advantage entirely.

## Key findings

- Sync floor: delay=1 coupling forces 100 exchanges/token; latency-dominated
  at 1.0-3.0 ms/token => hard ceiling ~330-1,000 tok/s for ANY tightly-
  coupled hybrid, regardless of network size.
- GPU ingestion path does not exist today (only token afferent mask); new
  kernel surface required (external-event buffer + canonical merge pass).
- Migration asymmetry is decisive: procedural side migrates state-only
  (topology regenerable from seed+gid); CUDA side needs reverse-CSR rebuild,
  which today is O(N_cfg) per migration => epoch-static only.
- Readback seam missing on procedural (snapshot() O(all residents), measured
  161 ms/token @100k residents).
- Bandwidth never binds (hundreds of bytes/step at text rates); latency and
  engineering do.

## Recommendation matrix (size x activity)

| Size / Activity | Sparse (<~3%) | Mid-density (3-20%) | High (>20%) |
|---|---|---|---|
| <=200k | CPU frontier (843 M) | GPU-only | GPU-only |
| 200k-4M | CPU EP (363 M @1M) | ONLY genuine hybrid window (projected 100-300) | CPU EP after injection-seam fix |
| >=10M | CPU EP (only headroom option) | CPU EP unless persistent dense hotspot | Known wall |

Honest lose conditions: hybrid loses whenever the working set fits one
device and that device clears the gate — i.e., all of <=200k and the whole
text-like regime. If read_cells + indexed staged membership land, CPU-alone
EP plausibly reaches the upper band at 1e6-1e7 and closes the mid-density
window too.

## Before ever revisiting: two microbenchmarks

(a) small-message PCIe round-trip latency on the actual card (cost model
pivots on unmeasured 10-30 us assumption); (b) GPU external-event-ingestion
prototype cost (<50 us/step at text rates or hybrid is dead).

## Sequencing adopted

Land `read_cells` + indexed staged membership first (serve CPU-alone EP),
re-run L3 activity matrix post-fix, revisit hybrid only if the 3-20% band at
>=1M remains real AND both microbenchmarks clear. Phase-level hybrid (CUDA
reservoir + host sparse readout) already ships and is correct for <=200k.
