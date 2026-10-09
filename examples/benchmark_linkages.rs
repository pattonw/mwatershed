//! Speed evaluation of the GASP linkage criteria.
//!
//! Runs every linkage on random and structured affinities in 2D and 3D and
//! prints a markdown report (wall time, number of segments, speed relative to
//! the mutex watershed). Run with:
//!
//! ```sh
//! cargo run --release --example benchmark_linkages > LINKAGE_REPORT.md
//! ```

use mwatershed::{agglomerate_with, GaspOptions, Linkage};
use ndarray::{Array, IxDyn};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;
use std::collections::HashSet;
use std::time::{Duration, Instant};

struct Scenario {
    name: &'static str,
    offsets: Vec<Vec<isize>>,
    shape: Vec<usize>,
    affinities: Array<f64, IxDyn>,
}

fn random_normal(offsets: &[Vec<isize>], shape: &[usize], seed: u64) -> Array<f64, IxDyn> {
    let mut rng = Isaac64Rng::seed_from_u64(seed);
    let mut full = vec![offsets.len()];
    full.extend(shape);
    Array::random_using(IxDyn(&full), Normal::new(0.0, 1.0).unwrap(), &mut rng)
}

/// Blobs of `cell` voxels with attractive interiors and repulsive borders,
/// plus noise, which resembles boundary predictions from a network.
fn blobs(
    offsets: &[Vec<isize>],
    shape: &[usize],
    cell: usize,
    noise: f64,
    seed: u64,
) -> Array<f64, IxDyn> {
    let mut rng = Isaac64Rng::seed_from_u64(seed);
    let mut full = vec![offsets.len()];
    full.extend(shape);
    let mut affs =
        Array::random_using(IxDyn(&full), Uniform::new(-noise, noise).unwrap(), &mut rng);
    let dims = shape.len();
    for (index, value) in affs.indexed_iter_mut() {
        let offset = &offsets[index[0]];
        let mut same_blob = true;
        for d in 0..dims {
            let u = index[d + 1] as isize;
            let v = u + offset[d];
            if v < 0 || v >= shape[d] as isize {
                continue;
            }
            if (u as usize) / cell != (v as usize) / cell {
                same_blob = false;
            }
        }
        *value += if same_blob { 0.6 } else { -0.6 };
        // occasional label noise
        if rng.random::<f64>() < 0.02 {
            *value = -*value;
        }
    }
    affs
}

fn scenarios() -> Vec<Scenario> {
    let offsets_2d = vec![vec![0, 1], vec![1, 0], vec![0, 4], vec![4, 0]];
    let offsets_3d = vec![
        vec![0, 0, 1],
        vec![0, 1, 0],
        vec![1, 0, 0],
        vec![0, 0, 3],
        vec![0, 3, 0],
        vec![3, 0, 0],
    ];
    let mut out = Vec::new();
    for size in [128usize, 256] {
        let shape = vec![size, size];
        out.push(Scenario {
            name: "2D random normal",
            offsets: offsets_2d.clone(),
            affinities: random_normal(&offsets_2d, &shape, 1),
            shape: shape.clone(),
        });
        out.push(Scenario {
            name: "2D blobs (16px)",
            offsets: offsets_2d.clone(),
            affinities: blobs(&offsets_2d, &shape, 16, 0.5, 2),
            shape,
        });
    }
    for size in [32usize, 48] {
        let shape = vec![size, size, size];
        out.push(Scenario {
            name: "3D random normal",
            offsets: offsets_3d.clone(),
            affinities: random_normal(&offsets_3d, &shape, 3),
            shape: shape.clone(),
        });
        out.push(Scenario {
            name: "3D blobs (8vx)",
            offsets: offsets_3d.clone(),
            affinities: blobs(&offsets_3d, &shape, 8, 0.5, 4),
            shape,
        });
    }
    out
}

fn run(scenario: &Scenario, options: GaspOptions) -> (Duration, usize) {
    let seeds: Array<usize, IxDyn> = Array::zeros(IxDyn(&scenario.shape));
    let start = Instant::now();
    let labels = match scenario.shape.len() {
        2 => agglomerate_with::<2>(
            &scenario.affinities,
            scenario.offsets.clone(),
            vec![],
            seeds,
            None,
            false,
            options,
        ),
        3 => agglomerate_with::<3>(
            &scenario.affinities,
            scenario.offsets.clone(),
            vec![],
            seeds,
            None,
            false,
            options,
        ),
        _ => unreachable!(),
    };
    let elapsed = start.elapsed();
    let segments: HashSet<usize> = labels.iter().copied().filter(|&x| x != 0).collect();
    (elapsed, segments.len())
}

fn main() {
    let repeats = 3;
    println!("# Linkage criteria benchmark\n");
    println!("Best of {repeats} runs per cell, release build, single thread.");
    println!("`rel` is wall time relative to `abs_max` (mutex watershed) on the same input;");
    println!("`segments` is the number of non-background labels. `_nocl` disables cannot-link constraints.\n");
    println!("| Scenario | Shape | Linkage | Time (ms) | rel | Segments |");
    println!("|----------|-------|---------|----------:|----:|---------:|");
    for scenario in scenarios() {
        let shape = scenario
            .shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let mut baseline: Option<f64> = None;
        for linkage in Linkage::ALL {
            for cannot_link in [true, false] {
                if linkage == Linkage::AbsMax && !cannot_link {
                    continue;
                }
                let options = GaspOptions::new(linkage, cannot_link);
                let mut best = Duration::MAX;
                let mut segments = 0;
                for _ in 0..repeats {
                    let (elapsed, n) = run(&scenario, options);
                    best = best.min(elapsed);
                    segments = n;
                }
                let ms = best.as_secs_f64() * 1e3;
                let base = *baseline.get_or_insert(ms);
                let label = format!("{linkage}{}", if cannot_link { "" } else { "_nocl" });
                println!(
                    "| {} | {} | {} | {:.1} | {:.2}x | {} |",
                    scenario.name,
                    shape,
                    label,
                    ms,
                    ms / base,
                    segments
                );
            }
        }
    }
}
