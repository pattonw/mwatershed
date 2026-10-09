use ndarray::{Array, Axis};
use ndarray::IxDyn;
use ordered_float::NotNan;
use rayon::prelude::*;
use std::collections::HashSet;
use std::convert::TryInto;

use ndarray_rand::rand;

use crate::clustering::{AgglomEdge, Clustering};

// ── Coordinate helpers ──────────────────────────────────────────────

fn flat_to_nd<const D: usize>(flat_idx: usize, shape: &[usize; D]) -> [usize; D] {
    let mut coords = [0usize; D];
    let mut remaining = flat_idx;
    for i in (0..D).rev() {
        coords[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    coords
}

fn nd_to_flat<const D: usize>(coords: &[usize; D], shape: &[usize; D]) -> usize {
    let mut flat = 0;
    for i in 0..D {
        flat = flat * shape[i] + coords[i];
    }
    flat
}

fn compute_grid<const D: usize>(
    shape: &[usize; D],
    block_shape: &[usize; D],
) -> ([usize; D], usize) {
    let mut grid_shape = [0usize; D];
    let mut num_blocks = 1;
    for i in 0..D {
        grid_shape[i] = (shape[i] + block_shape[i] - 1) / block_shape[i];
        num_blocks *= grid_shape[i];
    }
    (grid_shape, num_blocks)
}

fn block_bounds<const D: usize>(
    block_idx: usize,
    grid_shape: &[usize; D],
    block_shape: &[usize; D],
    shape: &[usize; D],
) -> ([usize; D], [usize; D]) {
    let block_coords = flat_to_nd(block_idx, grid_shape);
    let mut start = [0usize; D];
    let mut end = [0usize; D];
    for i in 0..D {
        start[i] = block_coords[i] * block_shape[i];
        end[i] = ((block_coords[i] + 1) * block_shape[i]).min(shape[i]);
    }
    (start, end)
}

fn in_bounds<const D: usize>(coords: &[usize; D], start: &[usize; D], end: &[usize; D]) -> bool {
    (0..D).all(|i| coords[i] >= start[i] && coords[i] < end[i])
}

fn block_dims<const D: usize>(bstart: &[usize; D], bend: &[usize; D]) -> [usize; D] {
    let mut dims = [0usize; D];
    for i in 0..D {
        dims[i] = bend[i] - bstart[i];
    }
    dims
}

fn coords_to_local<const D: usize>(
    coords: &[usize; D],
    bstart: &[usize; D],
    bdims: &[usize; D],
) -> usize {
    let mut local = 0;
    for i in 0..D {
        local = local * bdims[i] + (coords[i] - bstart[i]);
    }
    local
}

fn local_to_global<const D: usize>(
    local_id: usize,
    bstart: &[usize; D],
    bdims: &[usize; D],
    shape: &[usize; D],
) -> usize {
    let local_coords = flat_to_nd(local_id, bdims);
    let mut global_coords = [0usize; D];
    for i in 0..D {
        global_coords[i] = bstart[i] + local_coords[i];
    }
    nd_to_flat(&global_coords, shape)
}

/// Map a global node ID to a block index given the block grid.
fn node_to_block<const D: usize>(
    node_global: usize,
    spatial_shape: &[usize; D],
    block_shape: &[usize; D],
    grid_shape: &[usize; D],
) -> usize {
    let coords = flat_to_nd(node_global, spatial_shape);
    let mut block_id = 0;
    for i in 0..D {
        block_id = block_id * grid_shape[i] + coords[i] / block_shape[i];
    }
    block_id
}

// ── Edge types ──────────────────────────────────────────────────────

enum RawEdge {
    Interior {
        abs_aff: NotNan<f64>,
        positive: bool,
        u_local: usize,
        v_local: usize,
        u_global: usize,
        v_global: usize,
    },
    CrossBlock {
        abs_aff: NotNan<f64>,
        positive: bool,
        local_endpoint: usize,
        u_global: usize,
        v_global: usize,
    },
}

impl RawEdge {
    fn abs_aff(&self) -> NotNan<f64> {
        match self {
            RawEdge::Interior { abs_aff, .. } => *abs_aff,
            RawEdge::CrossBlock { abs_aff, .. } => *abs_aff,
        }
    }
}

/// Result of processing one block (either from affinities or from deferred edges).
struct BlockResult {
    deferred: Vec<(NotNan<f64>, AgglomEdge)>,
    merges: Vec<(usize, usize)>,
    mutex_pairs: Vec<(usize, usize)>,
    positive_nodes: HashSet<usize>,
}

// ── Level 0: Extract from affinities ────────────────────────────────

fn process_block_from_affinities<const D: usize>(
    block_idx: usize,
    affinities: &Array<f64, IxDyn>,
    offsets: &[[isize; D]],
    seeds: &Array<usize, IxDyn>,
    strides: &[Vec<usize>],
    randomized_strides: bool,
    shape: &[usize; D],
    block_shape: &[usize; D],
    grid_shape: &[usize; D],
    context: &[usize; D],
    lossy: bool,
) -> BlockResult {
    let (core_start, core_end) = block_bounds(block_idx, grid_shape, block_shape, shape);
    // Expand block by context, clamped to volume bounds
    let mut bstart = [0usize; D];
    let mut bend = [0usize; D];
    for i in 0..D {
        bstart[i] = core_start[i].saturating_sub(context[i]);
        bend[i] = (core_end[i] + context[i]).min(shape[i]);
    }
    let bdims = block_dims(&bstart, &bend);
    let n_local = bdims.iter().product::<usize>();

    let has_context = context.iter().any(|&c| c > 0);
    let mut raw_edges: Vec<RawEdge> = Vec::new();
    let mut positive_nodes: HashSet<usize> = HashSet::new();
    // In lossy mode with context, cross-core edges are processed locally
    // (as Interior) for better clustering state, but also deferred so the
    // serial pass can reconcile conflicting decisions from adjacent blocks.
    let mut cross_core_deferred: Vec<(NotNan<f64>, AgglomEdge)> = Vec::new();

    for (offset_index, offset) in offsets.iter().enumerate() {
        let stride = &strides[offset_index];
        let all_offset_affs = affinities.index_axis(Axis(0), offset_index);

        let mut iter_start = [0usize; D];
        let mut iter_end = [0usize; D];
        for i in 0..D {
            if offset[i] > 0 {
                iter_start[i] = bstart[i].saturating_sub(offset[i] as usize);
                iter_end[i] = bend[i];
            } else if offset[i] < 0 {
                iter_start[i] = bstart[i];
                iter_end[i] = (bend[i] + (-offset[i]) as usize).min(shape[i]);
            } else {
                iter_start[i] = bstart[i];
                iter_end[i] = bend[i];
            }
        }

        let mut u_coords = iter_start;
        loop {
            let mut v_valid = true;
            let mut v_coords = [0usize; D];
            for i in 0..D {
                let vi = u_coords[i] as isize + offset[i];
                if vi < 0 || vi >= shape[i] as isize {
                    v_valid = false;
                    break;
                }
                v_coords[i] = vi as usize;
            }

            if v_valid {
                let u_in = in_bounds(&u_coords, &bstart, &bend);
                let v_in = in_bounds(&v_coords, &bstart, &bend);

                if u_in || v_in {
                    let mut use_edge = true;
                    if !randomized_strides {
                        for i in 0..D {
                            let slice_start =
                                if offset[i] < 0 { (-offset[i]) as usize } else { 0 };
                            if u_coords[i] >= slice_start
                                && (u_coords[i] - slice_start) % stride[i] != 0
                            {
                                use_edge = false;
                                break;
                            }
                        }
                    } else {
                        use_edge = rand::random::<f32>()
                            < 1.0 / stride.iter().product::<usize>() as f32;
                    }

                    if use_edge {
                        let aff_val = all_offset_affs[u_coords.as_slice()];
                        let abs_aff = NotNan::new(aff_val.abs())
                            .expect("Cannot handle `nan` affinities");
                        let is_positive = aff_val > 0.0;
                        let is_cross = !(u_in && v_in);

                        let u_node = seeds[u_coords.as_slice()];
                        let v_node = seeds[v_coords.as_slice()];

                        if is_positive {
                            positive_nodes.insert(u_node);
                            positive_nodes.insert(v_node);
                        }

                        if is_cross {
                            let local_endpoint = if u_in {
                                coords_to_local(&u_coords, &bstart, &bdims)
                            } else {
                                coords_to_local(&v_coords, &bstart, &bdims)
                            };
                            raw_edges.push(RawEdge::CrossBlock {
                                abs_aff,
                                positive: is_positive,
                                local_endpoint,
                                u_global: u_node,
                                v_global: v_node,
                            });
                        } else {
                            let u_local = coords_to_local(&u_coords, &bstart, &bdims);
                            let v_local = coords_to_local(&v_coords, &bstart, &bdims);
                            raw_edges.push(RawEdge::Interior {
                                abs_aff,
                                positive: is_positive,
                                u_local,
                                v_local,
                                u_global: u_node,
                                v_global: v_node,
                            });
                            // Track edges that cross the core boundary for deferral
                            if has_context && lossy {
                                let u_in_core = in_bounds(&u_coords, &core_start, &core_end);
                                let v_in_core = in_bounds(&v_coords, &core_start, &core_end);
                                if u_in_core != v_in_core {
                                    cross_core_deferred.push((
                                        abs_aff,
                                        AgglomEdge(is_positive, u_node, v_node),
                                    ));
                                }
                            }
                        }
                    }
                }
            }

            let mut dim = D - 1;
            loop {
                u_coords[dim] += 1;
                if u_coords[dim] < iter_end[dim] {
                    break;
                }
                u_coords[dim] = iter_start[dim];
                if dim == 0 {
                    u_coords = iter_end;
                    break;
                }
                dim -= 1;
            }
            if u_coords == iter_end {
                break;
            }
        }
    }

    // Filter interior edges where both endpoints are negative-only
    raw_edges.retain(|edge| match edge {
        RawEdge::Interior {
            u_global, v_global, ..
        } => positive_nodes.contains(u_global) || positive_nodes.contains(v_global),
        RawEdge::CrossBlock { .. } => true,
    });

    let mut result = process_raw_edges(raw_edges, n_local, positive_nodes, shape, &bstart, &bdims, seeds, lossy);

    // Add cross-core edges that were processed locally but also need deferral
    result.deferred.extend(cross_core_deferred);

    // With context, filter outputs to only include edges/merges/mutexes
    // where at least one endpoint is in the core block. Halo-only items
    // will be handled by the adjacent block that owns those nodes as core.
    if has_context {
        let has_core_endpoint = |u: usize, v: usize| -> bool {
            let u_coords = flat_to_nd(u, shape);
            let v_coords = flat_to_nd(v, shape);
            in_bounds(&u_coords, &core_start, &core_end)
                || in_bounds(&v_coords, &core_start, &core_end)
        };
        result.deferred.retain(|(_, AgglomEdge(_, u, v))| has_core_endpoint(*u, *v));
        result.merges.retain(|(u, v)| has_core_endpoint(*u, *v));
        result.mutex_pairs.retain(|(u, v)| has_core_endpoint(*u, *v));
    }

    result
}

// ── Shared freeze-process logic ─────────────────────────────────────

fn process_raw_edges<const D: usize>(
    mut raw_edges: Vec<RawEdge>,
    n_local: usize,
    positive_nodes: HashSet<usize>,
    shape: &[usize; D],
    bstart: &[usize; D],
    bdims: &[usize; D],
    seeds: &Array<usize, IxDyn>,
    lossy: bool,
) -> BlockResult {
    if raw_edges.is_empty() {
        return BlockResult {
            deferred: Vec::new(),
            merges: Vec::new(),
            mutex_pairs: Vec::new(),
            positive_nodes,
        };
    }

    raw_edges.sort_unstable_by(|a, b| b.abs_aff().cmp(&a.abs_aff()));

    let mut clustering = Clustering::new(n_local);
    let mut frozen = vec![false; n_local];
    let mut deferred: Vec<(NotNan<f64>, AgglomEdge)> = Vec::new();
    let mut merges: Vec<(usize, usize)> = Vec::new();

    for edge in raw_edges {
        match edge {
            RawEdge::CrossBlock {
                abs_aff,
                positive,
                local_endpoint,
                u_global,
                v_global,
            } => {
                if !lossy {
                    let rep = clustering.positives.clusters.find(local_endpoint);
                    frozen[rep] = true;
                }
                deferred.push((abs_aff, AgglomEdge(positive, u_global, v_global)));
            }
            RawEdge::Interior {
                abs_aff,
                positive,
                u_local,
                v_local,
                u_global,
                v_global,
            } => {
                if clustering.positives.clusters.equiv(u_local, v_local) {
                    continue;
                }

                let u_rep = clustering.positives.clusters.find(u_local);
                let v_rep = clustering.positives.clusters.find(v_local);

                if !lossy && (frozen[u_rep] || frozen[v_rep]) {
                    frozen[u_rep] = true;
                    frozen[v_rep] = true;
                    deferred.push((abs_aff, AgglomEdge(positive, u_global, v_global)));
                    continue;
                }

                if !clustering.negatives.mutex(u_rep, v_rep) {
                    let (new_id, old_id) = if u_rep < v_rep {
                        (u_rep, v_rep)
                    } else {
                        (v_rep, u_rep)
                    };
                    if positive {
                        clustering.merge(new_id, old_id);
                        merges.push((u_global, v_global));
                        if !lossy {
                            let rep = clustering.positives.clusters.find(new_id);
                            if frozen[new_id] || frozen[old_id] {
                                frozen[rep] = true;
                            }
                        }
                    } else {
                        clustering.split(new_id, old_id);
                    }
                }
            }
        }
    }

    let mutex_pairs: Vec<(usize, usize)> = clustering
        .negatives
        .iter_pairs()
        .map(|(a, b)| {
            let ga = local_to_global(a, bstart, bdims, shape);
            let gb = local_to_global(b, bstart, bdims, shape);
            (
                seeds[flat_to_nd(ga, shape).as_slice()],
                seeds[flat_to_nd(gb, shape).as_slice()],
            )
        })
        .collect();

    BlockResult {
        deferred,
        merges,
        mutex_pairs,
        positive_nodes,
    }
}

// ── Level N: Process deferred edges in larger blocks ────────────────

/// Process a list of pre-sorted deferred edges blockwise.
/// Edges are in global node IDs. They get partitioned into blocks based on
/// min(u,v) spatial position, with cross-block edges triggering freeze.
fn process_deferred_level<const D: usize>(
    edges: Vec<(NotNan<f64>, AgglomEdge)>,
    spatial_shape: &[usize; D],
    block_shape: &[usize; D],
    seeds: &Array<usize, IxDyn>,
    global: &Clustering,
    lossy: bool,
) -> Vec<BlockResult> {
    let (grid_shape, num_blocks) = compute_grid(spatial_shape, block_shape);

    // Build a precomputed node-to-spatial-index lookup for the seed array.
    // Since seeds maps node_id = seeds[spatial_coords], and after relabeling
    // node IDs are flat indices, we can reverse this. But with seeds, node IDs
    // might not be flat indices (they're the relabeled values). We need to map
    // node_id -> spatial coords. For no-seeds case, node_id == flat_index.
    // For seeds case, we need the reverse lookup.
    //
    // Actually, the edges already have global node IDs. To determine which
    // block a node belongs to, we need its spatial position. Since node IDs
    // after relabeling are the flat indices (for unseeded voxels) or the first
    // occurrence index (for seeded), we can use node_id as the spatial flat index
    // directly — seeds[flat_index] == node_id after relabeling, and for unseeded
    // voxels node_id == flat_index. For seeded voxels sharing a label, they all
    // map to the same node_id (the first occurrence), so block assignment uses
    // that first occurrence's position.

    // Partition edges into blocks
    let mut block_edges: Vec<Vec<RawEdge>> = (0..num_blocks).map(|_| Vec::new()).collect();
    // Track block dims for local ID computation
    let block_info: Vec<([usize; D], [usize; D], [usize; D])> = (0..num_blocks)
        .map(|bi| {
            let (bs, be) = block_bounds(bi, &grid_shape, block_shape, spatial_shape);
            let bd = block_dims(&bs, &be);
            (bs, be, bd)
        })
        .collect();

    for (abs_aff, AgglomEdge(positive, u, v)) in edges {
        // Find reps in the current global clustering state
        let u_rep = global.positives.clusters.find(u);
        let v_rep = global.positives.clusters.find(v);
        if u_rep == v_rep {
            continue; // already merged globally
        }
        // Skip edges already blocked by a global mutex
        if global.negatives.mutex(u_rep, v_rep) {
            continue;
        }

        // Use the node ID as spatial flat index to determine block
        let block_u = node_to_block(u, spatial_shape, block_shape, &grid_shape);
        let block_v = node_to_block(v, spatial_shape, block_shape, &grid_shape);

        if block_u == block_v {
            let (ref bs, _, ref bd) = block_info[block_u];
            let u_coords = flat_to_nd(u, spatial_shape);
            let v_coords = flat_to_nd(v, spatial_shape);
            let u_local = coords_to_local(&u_coords, bs, bd);
            let v_local = coords_to_local(&v_coords, bs, bd);
            block_edges[block_u].push(RawEdge::Interior {
                abs_aff,
                positive,
                u_local,
                v_local,
                u_global: u,
                v_global: v,
            });
        } else {
            // Cross-block: add to the block of the lower node
            let primary_block = block_u.min(block_v);
            let (ref bs, ref be, ref bd) = block_info[primary_block];
            let u_coords = flat_to_nd(u, spatial_shape);
            let v_coords = flat_to_nd(v, spatial_shape);
            let local_endpoint = if in_bounds(&u_coords, bs, be) {
                coords_to_local(&u_coords, bs, bd)
            } else if in_bounds(&v_coords, bs, be) {
                coords_to_local(&v_coords, bs, bd)
            } else {
                // Neither endpoint is in the primary block — this can happen when
                // node IDs don't correspond to spatial positions within the block
                // (seeded voxels). Defer directly.
                // Add to the block that contains u
                let (ref bs2, _, ref bd2) = block_info[block_u];
                let u_local = coords_to_local(&flat_to_nd(u, spatial_shape), bs2, bd2);
                block_edges[block_u].push(RawEdge::CrossBlock {
                    abs_aff,
                    positive,
                    local_endpoint: u_local,
                    u_global: u,
                    v_global: v,
                });
                continue;
            };
            block_edges[primary_block].push(RawEdge::CrossBlock {
                abs_aff,
                positive,
                local_endpoint,
                u_global: u,
                v_global: v,
            });
        }
    }

    // Process each block in parallel
    block_edges
        .into_par_iter()
        .enumerate()
        .map(|(bi, edges)| {
            let (ref bs, _, ref bd) = block_info[bi];
            let n_local = bd.iter().product::<usize>();
            process_raw_edges(edges, n_local, HashSet::new(), spatial_shape, bs, bd, seeds, lossy)
        })
        .collect()
}

// ── Replay helper ───────────────────────────────────────────────────

fn replay_results(global: &mut Clustering, results: &[BlockResult]) {
    // Two-pass replay: merges first, then mutex pairs.
    // During the merge pass, no mutex edges exist yet, so negatives.merge()
    // operates on empty sets (O(1)). We bypass Clustering::merge and call
    // positives.merge directly to avoid that overhead entirely.
    for result in results {
        for &(a, b) in &result.merges {
            if !global.positives.clusters.equiv(a, b) {
                global.positives.merge(a, b);
            }
        }
    }

    // Now add all mutex pairs using the final merge representatives.
    for result in results {
        for &(a, b) in &result.mutex_pairs {
            let rep_a = global.positives.clusters.find(a);
            let rep_b = global.positives.clusters.find(b);
            if rep_a != rep_b {
                global.split(rep_a, rep_b);
            }
        }
    }
}

/// Remap deferred edges to use current cluster representatives.
/// Filters out edges that are already equiv or already mutexed.
fn remap_deferred(
    edges: &[(NotNan<f64>, AgglomEdge)],
    global: &Clustering,
) -> Vec<(NotNan<f64>, AgglomEdge)> {
    let mut result = Vec::with_capacity(edges.len());
    for &(aff, AgglomEdge(pos, u, v)) in edges {
        let u_rep = global.positives.clusters.find(u);
        let v_rep = global.positives.clusters.find(v);
        if u_rep == v_rep {
            continue; // already merged
        }
        if global.negatives.mutex(u_rep, v_rep) {
            continue; // already mutexed
        }
        result.push((aff, AgglomEdge(pos, u_rep, v_rep)));
    }
    result
}

fn collect_deferred(results: &[BlockResult]) -> Vec<(NotNan<f64>, AgglomEdge)> {
    let total_len: usize = results.iter().map(|r| r.deferred.len()).sum();
    let mut all = Vec::with_capacity(total_len);
    for result in results {
        all.extend(result.deferred.iter().cloned());
    }
    all.par_sort_unstable_by(|a, b| b.0.cmp(&a.0));
    all
}

// ── Public API ──────────────────────────────────────────────────────

/// Run blockwise parallel mutex watershed with a pyramid of block sizes.
///
/// Level 0 extracts edges from affinities using `block_shapes[0]`.
/// Each subsequent level processes deferred edges from the previous level
/// using progressively larger blocks. The final remaining deferred edges
/// are processed sequentially.
pub fn process_blockwise<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    seed_mutex_edges: Vec<AgglomEdge>,
    seeds: &Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
    block_shapes: &[[usize; D]],
    num_nodes: usize,
    context: Option<[usize; D]>,
    lossy: bool,
) -> (Clustering, HashSet<usize>) {
    use std::time::Instant;

    assert!(!block_shapes.is_empty(), "Need at least one block shape");

    let spatial_shape: [usize; D] = seeds
        .shape()
        .try_into()
        .expect("seeds shape does not match D");

    let offsets: Vec<[isize; D]> = offsets
        .into_iter()
        .map(|o| o.try_into().unwrap())
        .collect();

    let strides = strides.unwrap_or_else(|| {
        (0..offsets.len())
            .map(|_| (0..D).map(|_| 1).collect())
            .collect()
    });

    let ctx = context.unwrap_or([0; D]);

    // Level 0: extract from affinities (PARALLEL)
    let block_shape = &block_shapes[0];
    let (grid_shape, num_blocks) = compute_grid(&spatial_shape, block_shape);

    let t = Instant::now();
    let level0_results: Vec<BlockResult> = (0..num_blocks)
        .into_par_iter()
        .map(|block_idx| {
            process_block_from_affinities(
                block_idx, affinities, &offsets, seeds, &strides, randomized_strides,
                &spatial_shape, block_shape, &grid_shape, &ctx, lossy,
            )
        })
        .collect();
    let t_level0_par = t.elapsed();
    eprintln!("  [blockwise] Level 0 parallel (extract+sort+process {} blocks): {:.1}ms",
        num_blocks, t_level0_par.as_secs_f64() * 1000.0);

    // Compute to_filter from level 0 (SERIAL)
    let t = Instant::now();
    let mut is_positive = vec![false; num_nodes];
    for result in &level0_results {
        for &n in &result.positive_nodes {
            is_positive[n] = true;
        }
    }
    let mut to_filter: HashSet<usize> = HashSet::new();
    for &node in seeds.iter() {
        if !is_positive[node] {
            to_filter.insert(node);
        }
    }
    let t_filter = t.elapsed();
    eprintln!("  [blockwise] Compute to_filter (serial): {:.1}ms",
        t_filter.as_secs_f64() * 1000.0);

    // Build global clustering, replay level 0 (SERIAL)
    let t = Instant::now();
    let mut global = Clustering::new(num_nodes);
    replay_results(&mut global, &level0_results);
    let t_replay0 = t.elapsed();
    eprintln!("  [blockwise] Replay level 0 (serial): {:.1}ms",
        t_replay0.as_secs_f64() * 1000.0);

    let t = Instant::now();
    let mut deferred = collect_deferred(&level0_results);
    deferred.retain(|(_, AgglomEdge(_, u, v))| !to_filter.contains(u) && !to_filter.contains(v));
    let t_collect0 = t.elapsed();
    let n_deferred_0 = deferred.len();
    eprintln!("  [blockwise] Collect+filter deferred (serial): {:.1}ms ({} deferred edges)",
        t_collect0.as_secs_f64() * 1000.0, n_deferred_0);

    // Levels 1..N: process deferred edges with larger blocks
    for level in 1..block_shapes.len() {
        if deferred.is_empty() {
            break;
        }

        let block_shape = &block_shapes[level];
        let covers_all = (0..D).all(|i| block_shape[i] >= spatial_shape[i]);
        if covers_all {
            break;
        }

        let t = Instant::now();
        deferred = remap_deferred(&deferred, &global);
        let t_remap = t.elapsed();

        let t = Instant::now();
        let level_results = process_deferred_level(
            deferred,
            &spatial_shape,
            block_shape,
            seeds,
            &global,
            lossy,
        );
        let t_level = t.elapsed();

        let t = Instant::now();
        replay_results(&mut global, &level_results);
        let t_replay = t.elapsed();

        let t = Instant::now();
        deferred = collect_deferred(&level_results);
        let t_collect = t.elapsed();

        eprintln!("  [blockwise] Level {} (block {:?}): remap {:.1}ms, parallel {:.1}ms, replay {:.1}ms, collect {:.1}ms ({} deferred)",
            level, block_shape,
            t_remap.as_secs_f64() * 1000.0,
            t_level.as_secs_f64() * 1000.0,
            t_replay.as_secs_f64() * 1000.0,
            t_collect.as_secs_f64() * 1000.0,
            deferred.len());
    }

    // Final: process remaining deferred edges sequentially
    let t = Instant::now();
    let mut final_edges: Vec<AgglomEdge> = seed_mutex_edges;
    final_edges.extend(deferred.into_iter().map(|(_, e)| e));
    let n_final = final_edges.len();
    global.process_edges(final_edges);
    let t_final = t.elapsed();
    eprintln!("  [blockwise] Final sequential process_edges: {:.1}ms ({} edges)",
        t_final.as_secs_f64() * 1000.0, n_final);

    (global, to_filter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_to_nd() {
        assert_eq!(flat_to_nd(5, &[3, 4]), [1, 1]);
        assert_eq!(flat_to_nd(0, &[3, 4]), [0, 0]);
        assert_eq!(flat_to_nd(11, &[3, 4]), [2, 3]);
    }

    #[test]
    fn test_coords_to_local() {
        assert_eq!(coords_to_local(&[3, 5], &[2, 3], &[4, 5]), 7);
        assert_eq!(coords_to_local(&[2, 3], &[2, 3], &[4, 5]), 0);
    }

    #[test]
    fn test_local_to_global_roundtrip() {
        let shape = [10, 12];
        let bstart = [2, 4];
        let bdims = [3, 5];
        for local in 0..15 {
            let global = local_to_global(local, &bstart, &bdims, &shape);
            let global_coords = flat_to_nd(global, &shape);
            let back = coords_to_local(&global_coords, &bstart, &bdims);
            assert_eq!(back, local, "roundtrip failed for local={}", local);
        }
    }

    #[test]
    fn test_block_bounds() {
        let shape = [10, 10];
        let block_shape = [4, 4];
        let grid_shape = [3, 3];

        let (s, e) = block_bounds(0, &grid_shape, &block_shape, &shape);
        assert_eq!(s, [0, 0]);
        assert_eq!(e, [4, 4]);

        let (s, e) = block_bounds(8, &grid_shape, &block_shape, &shape);
        assert_eq!(s, [8, 8]);
        assert_eq!(e, [10, 10]);
    }

    /// Adding context should never increase the total number of deferred edges.
    #[test]
    fn test_context_reduces_deferred_edges() {
        // Build a 20x20 volume with deterministic affinities
        let size = 20usize;
        let n = size * size;
        let mut state: u64 = 99999;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
        };
        let mut aff_data = Vec::with_capacity(2 * n);
        for _ in 0..(2 * n) {
            aff_data.push(next_f64());
        }
        let affinities = Array::from_shape_vec(vec![2, size, size], aff_data).unwrap().into_dyn();
        let mut seeds = Array::from_elem(vec![size, size], 0usize).into_dyn();
        seeds.iter_mut().enumerate().for_each(|(i, x)| *x = i);
        let offsets: Vec<[isize; 2]> = vec![[0, 1], [1, 0]];
        let strides: Vec<Vec<usize>> = vec![vec![1, 1], vec![1, 1]];

        let block_shape = [5, 5];
        let (grid_shape, num_blocks) = compute_grid(&[size, size], &block_shape);

        // Count total deferred edges across all blocks for each context level
        for lossy in [false, true] {
            let mut prev_deferred = usize::MAX;

            for ctx_size in [0, 1, 2, 3, 5] {
                let context = [ctx_size, ctx_size];
                let total_deferred: usize = (0..num_blocks)
                    .map(|block_idx| {
                        let result = process_block_from_affinities(
                            block_idx, &affinities, &offsets, &seeds, &strides,
                            false, &[size, size], &block_shape, &grid_shape,
                            &context, lossy,
                        );
                        result.deferred.len()
                    })
                    .sum();

                assert!(
                    total_deferred <= prev_deferred,
                    "lossy={}, context={}: deferred {} > previous {} (context should only reduce deferred edges)",
                    lossy, ctx_size, total_deferred, prev_deferred,
                );
                prev_deferred = total_deferred;
            }
        }
    }
}
