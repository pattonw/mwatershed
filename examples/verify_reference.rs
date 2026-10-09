use ndarray::Array;
use std::fs;

use mwatershed::agglomerate;

/// Simple LCG random number generator — must match save_reference.rs exactly.
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
}

fn verify(size: usize, seed: u64, filename: &str) {
    let mut rng = Rng::new(seed);
    let num_voxels = size * size;

    let aff_data: Vec<f64> = (0..2 * num_voxels).map(|_| rng.next_f64()).collect();
    let affinities = Array::from_shape_vec(vec![2, size, size], aff_data).unwrap();

    let seeds = Array::from_iter(0..num_voxels)
        .into_shape_with_order(vec![size, size])
        .unwrap();

    let offsets = vec![vec![0isize, 1], vec![1, 0]];
    let result = agglomerate::<2>(&affinities, offsets, vec![], seeds, None, false);

    let reference = fs::read_to_string(filename).expect("Failed to read reference file");
    let reference: Vec<usize> = reference
        .trim()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();

    let actual: Vec<usize> = result.iter().copied().collect();

    assert_eq!(
        actual.len(),
        reference.len(),
        "Length mismatch for {}",
        filename
    );

    let mismatches: Vec<usize> = actual
        .iter()
        .zip(reference.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();

    if mismatches.is_empty() {
        println!("{}: PASS (all {} values match)", filename, actual.len());
    } else {
        println!(
            "{}: FAIL ({} mismatches out of {})",
            filename,
            mismatches.len(),
            actual.len()
        );
        for &i in mismatches.iter().take(10) {
            println!("  [{}] expected={} got={}", i, reference[i], actual[i]);
        }
        std::process::exit(1);
    }
}

fn main() {
    verify(50, 42, "reference_50x50.csv");
    verify(100, 42, "reference_100x100.csv");
    println!("All reference checks passed.");
}
