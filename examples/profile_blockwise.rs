use ndarray::Array;
use std::collections::HashSet;
use std::time::Instant;

use mwatershed::{agglomerate, agglomerate_blockwise};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn gen_small_objects_2d(size: usize, tile_size: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let n = size * size;
    let tile_id = |r: usize, c: usize| (r / tile_size) * ((size + tile_size - 1) / tile_size) + c / tile_size;
    let mut data = Vec::with_capacity(2 * n);
    for r in 0..size {
        for c in 0..size {
            let noise = 0.1 * rng.normal();
            if c + 1 < size && tile_id(r, c) == tile_id(r, c + 1) {
                data.push(0.8 + noise);
            } else { data.push(-0.8 + noise); }
        }
    }
    for r in 0..size {
        for c in 0..size {
            let noise = 0.1 * rng.normal();
            if r + 1 < size && tile_id(r, c) == tile_id(r + 1, c) {
                data.push(0.8 + noise);
            } else { data.push(-0.8 + noise); }
        }
    }
    data
}

fn gen_random_2d(size: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let n = size * size;
    (0..2 * n).map(|_| rng.normal() * 0.5).collect()
}

fn profile_scenario(name: &str, size: usize, block_size: usize, aff_data: Vec<f64>) {
    let affinities = Array::from_shape_vec(vec![2, size, size], aff_data).unwrap();
    let seeds = Array::from_elem(vec![size, size], 0usize);
    let offsets = vec![vec![0isize, 1], vec![1, 0]];

    println!("\n=== {} ({}x{}, block={}) ===", name, size, size, block_size);

    // Sequential baseline
    let t = Instant::now();
    let seq = agglomerate::<2>(&affinities, offsets.clone(), vec![], seeds.clone(), None, false);
    let seq_time = t.elapsed();
    let seq_labels: HashSet<usize> = seq.iter().copied().collect();
    println!("Sequential: {:.1}ms, {} labels", seq_time.as_secs_f64() * 1000.0, seq_labels.len());

    // Blockwise with timing instrumentation via environment variable
    // We can't easily instrument inside the library, so time the whole call
    // and compare with sequential to infer where time is spent.
    // Instead, let's time multiple runs with different block sizes to see scaling.

    // Time the full blockwise call
    let t = Instant::now();
    let bw = agglomerate_blockwise::<2>(
        &affinities, offsets.clone(), vec![], seeds.clone(), None, false, [block_size, block_size], None, false
    );
    let bw_time = t.elapsed();
    let bw_labels: HashSet<usize> = bw.iter().copied().collect();
    println!("Blockwise:  {:.1}ms, {} labels, speedup {:.2}x",
        bw_time.as_secs_f64() * 1000.0, bw_labels.len(),
        seq_time.as_secs_f64() / bw_time.as_secs_f64());

    // Now profile sub-phases by calling the internal function directly
    // We need to replicate what agglomerate_blockwise does but with timing

    // Phase 0: seed relabeling
    let t0 = Instant::now();
    let mut seeds_work = seeds.clone();
    let mut lookup = std::collections::HashMap::new();
    seeds_work.iter_mut().enumerate().for_each(|(ind, x)| match x {
        0 => *x = ind,
        k => {
            let id = *lookup.get(k).unwrap_or(&ind);
            lookup.insert(*k, id);
            *k = id;
        }
    });
    let unique_seeds: Vec<&usize> = lookup.values().collect();
    let mut seed_mutex_edges = Vec::new();
    (0..unique_seeds.len()).for_each(|id1| {
        ((id1 + 1)..unique_seeds.len()).for_each(|id2| {
            seed_mutex_edges.push(mwatershed::clustering::AgglomEdge(
                false, *unique_seeds[id1], *unique_seeds[id2]
            ));
        })
    });
    let phase0 = t0.elapsed();

    // Phase 1-5: blockwise processing (this is the meat)
    let t1 = Instant::now();
    let (clustering, mut filtered_background) = mwatershed::blockwise::process_blockwise::<2>(
        &affinities,
        offsets.clone(),
        seed_mutex_edges,
        &seeds_work,
        None,
        false,
        &[[block_size, block_size]],
        seeds_work.len(),
        None,
        false,
    );
    let phase_blockwise = t1.elapsed();

    lookup.values().for_each(|node_id| {
        filtered_background.remove(node_id);
    });

    // Phase 6: relabeling
    let t6 = Instant::now();
    let mut seeds_out = seeds_work.clone();
    clustering.filter_map(&mut seeds_out, filtered_background);
    let mut rev_lookup = std::collections::HashMap::with_capacity(lookup.len());
    rev_lookup.insert(seeds_out.len(), 0);
    lookup.iter().for_each(|(seed, id)| {
        let rep_id = clustering.positives.clusters.find(*id);
        if *seed != rep_id {
            rev_lookup.insert(rep_id, *seed);
        }
    });
    seeds_out.iter_mut().for_each(|x| {
        *x = *rev_lookup.get(x).unwrap_or(x);
    });
    let phase6 = t6.elapsed();

    let total = phase0 + phase_blockwise + phase6;
    println!("\nPhase breakdown:");
    println!("  Phase 0 (seed relabel):      {:>8.1}ms  ({:.1}%)",
        phase0.as_secs_f64() * 1000.0, phase0.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Phase 1-5 (blockwise core):  {:>8.1}ms  ({:.1}%)",
        phase_blockwise.as_secs_f64() * 1000.0, phase_blockwise.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Phase 6 (relabel output):    {:>8.1}ms  ({:.1}%)",
        phase6.as_secs_f64() * 1000.0, phase6.as_secs_f64() / total.as_secs_f64() * 100.0);
    println!("  Total:                       {:>8.1}ms", total.as_secs_f64() * 1000.0);
}

fn main() {
    let size = 512;

    // Best block size (64) vs worst block sizes (2, 4)
    for &block in &[2, 4, 8, 32, 64] {
        let data = gen_small_objects_2d(size, 8, 42);
        profile_scenario("Small objects", size, block, data);
    }

    for &block in &[2, 4, 8, 32, 64] {
        let data = gen_random_2d(size, 42);
        profile_scenario("Random", size, block, data);
    }
}
