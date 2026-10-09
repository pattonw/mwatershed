use ndarray::Array;
use std::collections::HashMap;
use std::time::Instant;

use mwatershed::{agglomerate, agglomerate_blockwise};

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

/// Compute F1 score (harmonic mean of precision and recall) between two
/// segmentations, averaged over segments. Uses the "best matching" variant:
/// for each segment in `reference`, find the segment in `test` with highest
/// overlap, compute per-segment F1, then macro-average.
fn f1_score(reference: &Array<usize, ndarray::IxDyn>, test: &Array<usize, ndarray::IxDyn>) -> f64 {
    // Build label→voxel-indices maps
    let mut ref_segments: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut test_segments: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, (&r, &t)) in reference.iter().zip(test.iter()).enumerate() {
        ref_segments.entry(r).or_default().push(i);
        test_segments.entry(t).or_default().push(i);
    }

    // For each voxel, record which test segment it belongs to
    let n = reference.len();
    let mut voxel_to_test = vec![0usize; n];
    for (i, &t) in test.iter().enumerate() {
        voxel_to_test[i] = t;
    }

    let mut total_f1 = 0.0;
    let mut count = 0;

    for (_ref_label, ref_voxels) in &ref_segments {
        // Count overlaps with each test segment
        let mut overlap_counts: HashMap<usize, usize> = HashMap::new();
        for &idx in ref_voxels {
            *overlap_counts.entry(voxel_to_test[idx]).or_insert(0) += 1;
        }

        // Find best matching test segment
        let mut best_f1 = 0.0f64;
        for (&test_label, &overlap) in &overlap_counts {
            let test_size = test_segments.get(&test_label).map_or(0, |v| v.len());
            let precision = overlap as f64 / test_size as f64;
            let recall = overlap as f64 / ref_voxels.len() as f64;
            if precision + recall > 0.0 {
                let f1 = 2.0 * precision * recall / (precision + recall);
                best_f1 = best_f1.max(f1);
            }
        }
        total_f1 += best_f1;
        count += 1;
    }

    if count > 0 { total_f1 / count as f64 } else { 0.0 }
}

fn main() {
    let size = 200usize;
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

    // Run serial baseline
    println!("\nSerial baseline...");
    let t = Instant::now();
    let serial_result = agglomerate::<3>(
        &affinities,
        offsets.clone(),
        vec![],
        seeds.clone(),
        None,
        false,
    );
    let serial_time = t.elapsed();
    let serial_labels: std::collections::HashSet<usize> = serial_result.iter().copied().collect();
    println!("  Serial:  {:.3?}  {} labels\n", serial_time, serial_labels.len());

    // Run blockwise modes
    let bsize = 50;
    println!("{}x{}x{} volume, {}³ blocks, {} threads", size, size, size, bsize, rayon::current_num_threads());
    println!("{:<40} {:>8} {:>8} {:>8} {:>10}", "Mode", "Time", "Speedup", "Labels", "F1 score");
    println!("{}", "-".repeat(78));

    let modes: Vec<(&str, Option<[usize; 3]>, bool)> = vec![
        ("lossy=false, context=None", None, false),
        ("lossy=true,  context=None", None, true),
        ("lossy=false, context=1",    Some([1,1,1]), false),
        ("lossy=true,  context=1",    Some([1,1,1]), true),
        ("lossy=false, context=5",    Some([5,5,5]), false),
        ("lossy=true,  context=5",    Some([5,5,5]), true),
    ];

    for (label, context, lossy) in &modes {
        let t = Instant::now();
        let result = agglomerate_blockwise::<3>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
            [bsize, bsize, bsize],
            *context,
            *lossy,
        );
        let elapsed = t.elapsed();
        let labels: std::collections::HashSet<usize> = result.iter().copied().collect();
        let speedup = serial_time.as_secs_f64() / elapsed.as_secs_f64();
        let f1 = f1_score(&serial_result, &result);

        println!("{:<40} {:>7.2}s {:>7.2}x {:>8} {:>10.6}",
            label, elapsed.as_secs_f64(), speedup, labels.len(), f1);
    }
}
