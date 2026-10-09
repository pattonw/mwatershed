# Profiling Summary

Profiled on 2026-04-03 using a release build (`cargo run --example profile --release`).
Random normal-distributed affinities, 2D with offsets `[0,1]` and `[1,0]`, all unique seeds (worst case for mutex tracking).

## Phase Breakdown

| Phase | Description | 100x100 (10k voxels) | 200x200 (40k voxels) |
|-------|-------------|---------------------|---------------------|
| 0 | Relabel seeds | 0.6ms (0.0%) | 1.4ms (0.0%) |
| 1 | Edge extraction + sort | 2.8ms (0.0%) | 10.9ms (0.0%) |
| **2** | **`process_edges`** | **9.0s (100.0%)** | **217.5s (100.0%)** |
| 3 | Relabel output | 0.2ms (0.0%) | 14.4ms (0.0%) |
| | **Total** | **9.0s** | **217.5s** |

The 500x500 run (250k voxels) was OOM-killed before completing.

## Key Findings

1. **`process_edges` accounts for 100% of runtime.** Edge extraction, sorting, and relabeling are negligible at all tested sizes. Parallelizing those phases would have no measurable impact.

2. **Super-linear scaling.** 4x the voxels produced ~24x the runtime. This points to the `Negatives` mutex `HashSet` growing large as clusters merge — `Negatives::merge()` iterates all mutex edges of the absorbed node and transfers them to the representative node.

3. **Memory blowup.** The OOM kill at 500x500 confirms the per-node `HashSet<usize>` in `Negatives` is consuming excessive memory as mutex sets accumulate through merges.

## Implications for Parallelization

- Parallelizing edge extraction or sorting is not worthwhile — they contribute <0.1% of runtime.
- `process_edges` is inherently sequential: each edge decision depends on the current Union-Find and mutex state modified by all prior edges.
- The only viable multi-core strategy is **spatial block decomposition**: partition the volume into tiles, run the algorithm independently per tile in parallel, then stitch boundary edges.
- The super-linear scaling and OOM behavior also suggest that optimizing `Negatives` (e.g., more compact data structures, lazy mutex propagation) could yield large constant-factor improvements even on a single core.
