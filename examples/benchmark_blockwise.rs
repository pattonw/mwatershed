use ndarray::Array;
use std::collections::HashSet;
use std::time::Instant;

use mwatershed::agglomerate;

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self { SimpleRng(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn main() {
    let size = 400usize;
    let frag_size = 10usize;
    let num_voxels = size * size * size;
    let frags_per_dim = size / frag_size;
    let num_frags = frags_per_dim * frags_per_dim * frags_per_dim;
    let num_gt_segments = 200;
    let noise = 0.3;

    let mut rng = SimpleRng::new(42);
    let frag_to_segment: Vec<usize> = (0..num_frags)
        .map(|_| (rng.next_u64() as usize) % num_gt_segments)
        .collect();

    let segment_of = |z: usize, y: usize, x: usize| -> usize {
        frag_to_segment[(z / frag_size) * frags_per_dim * frags_per_dim
            + (y / frag_size) * frags_per_dim
            + x / frag_size]
    };

    println!("Generating {}x{}x{} volume ({} voxels)...", size, size, size, num_voxels);

    let mut aff_data = Vec::with_capacity(3 * num_voxels);
    for z in 0..size { for y in 0..size { for x in 0..size {
        if x + 1 < size && segment_of(z, y, x) == segment_of(z, y, x + 1) {
            aff_data.push(0.5 + noise * rng.normal());
        } else {
            aff_data.push(-0.5 + noise * rng.normal());
        }
    }}}
    for z in 0..size { for y in 0..size { for x in 0..size {
        if y + 1 < size && segment_of(z, y, x) == segment_of(z, y + 1, x) {
            aff_data.push(0.5 + noise * rng.normal());
        } else {
            aff_data.push(-0.5 + noise * rng.normal());
        }
    }}}
    for z in 0..size { for y in 0..size { for x in 0..size {
        if z + 1 < size && segment_of(z, y, x) == segment_of(z + 1, y, x) {
            aff_data.push(0.5 + noise * rng.normal());
        } else {
            aff_data.push(-0.5 + noise * rng.normal());
        }
    }}}

    let affinities = Array::from_shape_vec(vec![3, size, size, size], aff_data).unwrap();
    let seeds = Array::from_elem(vec![size, size, size], 0usize);
    let offsets = vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]];

    let bsize = 67;
    println!("\nBlockwise ({}³ blocks)...", bsize);

    // Phase 0: seed relabel
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

    // Phases 1-5: blockwise core
    let t1 = Instant::now();
    let (clustering, mut filtered_background) = mwatershed::blockwise::process_blockwise::<3>(
        &affinities,
        offsets,
        seed_mutex_edges,
        &seeds_work,
        None,
        false,
        &[[bsize, bsize, bsize]],
        seeds_work.len(),
        None,
        false,
    );
    let phase_core = t1.elapsed();

    lookup.values().for_each(|node_id| {
        filtered_background.remove(node_id);
    });

    // Phase 6: relabel
    let t6 = Instant::now();
    let mut seeds_out = seeds_work;
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

    let bw_time = phase0 + phase_core + phase6;
    let bw_labels: HashSet<usize> = seeds_out.iter().copied().collect();

    println!("  Phase 0 (seed relabel):     {:>10.3?}", phase0);
    println!("  Phase 1-5 (blockwise core): {:>10.3?}", phase_core);
    println!("  Phase 6 (relabel output):   {:>10.3?}", phase6);
    println!("  Total:  {:>10.3?}", bw_time);
    println!("  Labels: {}", bw_labels.len());
    println!("  Rayon threads: {}", rayon::current_num_threads());
}
