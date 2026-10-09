use ndarray::Array;
use std::collections::HashMap;
use std::time::Instant;

use mwatershed::clustering::{AgglomEdge, Clustering};
use mwatershed::get_edges;

/// Simple LCG random number generator for deterministic data generation.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.max(1e-15).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn profile_size(size: usize, num_segments: usize) {
    let num_voxels = size * size;
    println!(
        "=== {}x{} ({} voxels, ~{} target segments) ===",
        size, size, num_voxels, num_segments
    );

    let mut rng = Rng::new(42);

    // Generate a grid of ground-truth regions, then create affinities:
    // positive for same-region neighbors, negative for boundary neighbors,
    // with noise. No seeds — let the algorithm cluster purely from affinities.
    // Generate affinities: uniform random in [-1, 1], biased so that
    // split_fraction of edges are negative (split) and the rest are positive
    // (merge). This creates complex, irregular boundaries with realistic
    // mutex contention.
    let split_fraction = num_segments as f64 / num_voxels as f64;
    // bias shifts the distribution so ~split_fraction of values end up < 0
    let bias = 1.0 - 2.0 * split_fraction;
    let mut aff_data = Vec::with_capacity(2 * num_voxels);
    for _ in 0..2 * num_voxels {
        // uniform in [0, 1], then shift to [-1+bias, 1+bias] range
        let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        aff_data.push(u * 2.0 - 1.0 + bias);
    }
    let affinities = Array::from_shape_vec(vec![2, size, size], aff_data).unwrap();

    // No seeds — all zeros
    let seeds_arr = Array::from_elem(vec![size, size], 0usize);
    let offsets = vec![vec![0isize, 1], vec![1, 0]];

    println!("  Seeds: 0 (unsupervised)");

    // --- Phase 0: Relabel seeds ---
    let t0 = Instant::now();
    let mut seeds_work = seeds_arr.clone();
    let mut lookup = HashMap::new();
    seeds_work
        .iter_mut()
        .enumerate()
        .for_each(|(ind, x)| match x {
            0 => *x = ind,
            k => {
                let id = *lookup.get(k).unwrap_or(&ind);
                lookup.insert(*k, id);
                *k = id;
            }
        });
    let phase0 = t0.elapsed();

    // --- Phase 1: Edge extraction + sort ---
    let t1 = Instant::now();
    let (sorted_edges, filtered_background) =
        get_edges::<2>(&affinities, offsets.clone(), &seeds_work, None, false);
    let phase1 = t1.elapsed();
    let num_edges = sorted_edges.len();

    // --- Phase 2: process_edges ---
    let mut all_edges: Vec<AgglomEdge> = Vec::new();
    let unique_seeds: Vec<&usize> = lookup.values().collect();
    for id1 in 0..unique_seeds.len() {
        for id2 in (id1 + 1)..unique_seeds.len() {
            all_edges.push(AgglomEdge(false, *unique_seeds[id1], *unique_seeds[id2]));
        }
    }
    all_edges.extend(sorted_edges);

    let t2 = Instant::now();
    let mut clustering = Clustering::new(seeds_work.len());
    clustering.process_edges(all_edges);
    let phase2 = t2.elapsed();

    // --- Phase 3: Relabel output ---
    let t3 = Instant::now();
    clustering.filter_map(&mut seeds_work, filtered_background);
    let mut rev_lookup = HashMap::with_capacity(lookup.len());
    lookup.iter().for_each(|(seed_val, id)| {
        let rep_id = clustering.positives.clusters.find(*id);
        if *seed_val != rep_id {
            rev_lookup.insert(rep_id, *seed_val);
        }
    });
    seeds_work.iter_mut().for_each(|x| {
        *x = *rev_lookup.get(x).unwrap_or(x);
    });
    let phase3 = t3.elapsed();

    let total = phase0 + phase1 + phase2 + phase3;
    let unique_labels: std::collections::HashSet<usize> = seeds_work.iter().copied().collect();

    println!("  Edges processed: {}", num_edges);
    println!("  Output clusters: {}", unique_labels.len());
    println!(
        "  Phase 0 (relabel seeds):      {:>10.3?}  ({:.1}%)",
        phase0,
        phase0.as_secs_f64() / total.as_secs_f64() * 100.0
    );
    println!(
        "  Phase 1 (edge extract+sort):  {:>10.3?}  ({:.1}%)",
        phase1,
        phase1.as_secs_f64() / total.as_secs_f64() * 100.0
    );
    println!(
        "  Phase 2 (process_edges):      {:>10.3?}  ({:.1}%)",
        phase2,
        phase2.as_secs_f64() / total.as_secs_f64() * 100.0
    );
    println!(
        "  Phase 3 (relabel output):     {:>10.3?}  ({:.1}%)",
        phase3,
        phase3.as_secs_f64() / total.as_secs_f64() * 100.0
    );
    println!("  TOTAL:                        {:>10.3?}", total);
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        // Default: test various sizes with ~100 segments
        for &size in &[100, 200, 500] {
            profile_size(size, 100);
        }
    } else {
        // Usage: profile <size> [num_segments]
        let size: usize = args[0].parse().expect("First arg: size");
        let segments: usize = args
            .get(1)
            .map(|s| s.parse().expect("Second arg: num_segments"))
            .unwrap_or(100);
        profile_size(size, segments);
    }
}
