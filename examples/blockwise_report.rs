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

// ── Scenario generators ─────────────────────────────────────────────

/// Worst case: everything splits. Random affinities centered around 0.
fn gen_random_2d(size: usize, seed: u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>) {
    let mut rng = Rng::new(seed);
    let n = size * size;
    let data: Vec<f64> = (0..2 * n).map(|_| rng.normal() * 0.5).collect();
    let affs = Array::from_shape_vec(vec![2, size, size], data).unwrap();
    (affs, vec![vec![0, 1], vec![1, 0]])
}

/// Best case: everything merges. All affinities strongly positive.
fn gen_all_merge_2d(size: usize, seed: u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>) {
    let mut rng = Rng::new(seed);
    let n = size * size;
    let data: Vec<f64> = (0..2 * n).map(|_| 0.8 + 0.1 * rng.uniform()).collect();
    let affs = Array::from_shape_vec(vec![2, size, size], data).unwrap();
    (affs, vec![vec![0, 1], vec![1, 0]])
}

/// Best case for blockwise: small objects with clear boundaries.
/// Grid of small tiles (tile_size × tile_size) with +1 inside, -1 between.
/// Small noise so objects are well-defined. Freeze can't propagate far.
fn gen_small_objects_2d(size: usize, tile_size: usize, seed: u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>) {
    let mut rng = Rng::new(seed);
    let n = size * size;
    let tile_id = |r: usize, c: usize| (r / tile_size) * ((size + tile_size - 1) / tile_size) + c / tile_size;
    let mut data = Vec::with_capacity(2 * n);
    // offset [0,1]
    for r in 0..size {
        for c in 0..size {
            let noise = 0.1 * rng.normal();
            if c + 1 < size && tile_id(r, c) == tile_id(r, c + 1) {
                data.push(0.8 + noise);
            } else {
                data.push(-0.8 + noise);
            }
        }
    }
    // offset [1,0]
    for r in 0..size {
        for c in 0..size {
            let noise = 0.1 * rng.normal();
            if r + 1 < size && tile_id(r, c) == tile_id(r + 1, c) {
                data.push(0.8 + noise);
            } else {
                data.push(-0.8 + noise);
            }
        }
    }
    (Array::from_shape_vec(vec![2, size, size], data).unwrap(), vec![vec![0, 1], vec![1, 0]])
}

/// Worst case for blockwise: large diffuse objects with noisy boundaries.
/// Few large regions with high noise so clusters propagate across block boundaries.
fn gen_large_objects_2d(size: usize, seed: u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>) {
    let mut rng = Rng::new(seed);
    let n = size * size;
    // ~4 large regions
    let tile_size = size / 2;
    let tile_id = |r: usize, c: usize| (r / tile_size) * 2 + c / tile_size;
    let mut data = Vec::with_capacity(2 * n);
    let noise_scale = 0.4;
    for r in 0..size {
        for c in 0..size {
            let noise = noise_scale * rng.normal();
            if c + 1 < size && tile_id(r, c) == tile_id(r, c + 1) {
                data.push(0.3 + noise);
            } else {
                data.push(-0.3 + noise);
            }
        }
    }
    for r in 0..size {
        for c in 0..size {
            let noise = noise_scale * rng.normal();
            if r + 1 < size && tile_id(r, c) == tile_id(r + 1, c) {
                data.push(0.3 + noise);
            } else {
                data.push(-0.3 + noise);
            }
        }
    }
    (Array::from_shape_vec(vec![2, size, size], data).unwrap(), vec![vec![0, 1], vec![1, 0]])
}

// ── 3D variants ─────────────────────────────────────────────────────

fn gen_random_3d(size: usize, seed: u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>) {
    let mut rng = Rng::new(seed);
    let n = size * size * size;
    let data: Vec<f64> = (0..3 * n).map(|_| rng.normal() * 0.5).collect();
    let affs = Array::from_shape_vec(vec![3, size, size, size], data).unwrap();
    (affs, vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]])
}

fn gen_small_objects_3d(size: usize, tile_size: usize, seed: u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>) {
    let mut rng = Rng::new(seed);
    let n = size * size * size;
    let gpd = (size + tile_size - 1) / tile_size;
    let tile_id = |z: usize, y: usize, x: usize|
        (z / tile_size) * gpd * gpd + (y / tile_size) * gpd + x / tile_size;
    let mut data = Vec::with_capacity(3 * n);
    // [0,0,1]
    for z in 0..size { for y in 0..size { for x in 0..size {
        let noise = 0.1 * rng.normal();
        if x + 1 < size && tile_id(z, y, x) == tile_id(z, y, x + 1) {
            data.push(0.8 + noise);
        } else { data.push(-0.8 + noise); }
    }}}
    // [0,1,0]
    for z in 0..size { for y in 0..size { for x in 0..size {
        let noise = 0.1 * rng.normal();
        if y + 1 < size && tile_id(z, y, x) == tile_id(z, y + 1, x) {
            data.push(0.8 + noise);
        } else { data.push(-0.8 + noise); }
    }}}
    // [1,0,0]
    for z in 0..size { for y in 0..size { for x in 0..size {
        let noise = 0.1 * rng.normal();
        if z + 1 < size && tile_id(z, y, x) == tile_id(z + 1, y, x) {
            data.push(0.8 + noise);
        } else { data.push(-0.8 + noise); }
    }}}
    (Array::from_shape_vec(vec![3, size, size, size], data).unwrap(),
     vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]])
}

fn gen_large_objects_3d(size: usize, seed: u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>) {
    let mut rng = Rng::new(seed);
    let n = size * size * size;
    let tile_size = size / 2;
    let tile_id = |z: usize, y: usize, x: usize|
        (z / tile_size) * 4 + (y / tile_size) * 2 + x / tile_size;
    let noise_scale = 0.4;
    let mut data = Vec::with_capacity(3 * n);
    for z in 0..size { for y in 0..size { for x in 0..size {
        let noise = noise_scale * rng.normal();
        if x + 1 < size && tile_id(z, y, x) == tile_id(z, y, x + 1) {
            data.push(0.3 + noise);
        } else { data.push(-0.3 + noise); }
    }}}
    for z in 0..size { for y in 0..size { for x in 0..size {
        let noise = noise_scale * rng.normal();
        if y + 1 < size && tile_id(z, y, x) == tile_id(z, y + 1, x) {
            data.push(0.3 + noise);
        } else { data.push(-0.3 + noise); }
    }}}
    for z in 0..size { for y in 0..size { for x in 0..size {
        let noise = noise_scale * rng.normal();
        if z + 1 < size && tile_id(z, y, x) == tile_id(z + 1, y, x) {
            data.push(0.3 + noise);
        } else { data.push(-0.3 + noise); }
    }}}
    (Array::from_shape_vec(vec![3, size, size, size], data).unwrap(),
     vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]])
}

// ── Benchmarking helpers ────────────────────────────────────────────

struct BenchResult {
    scenario: String,
    dim: usize,
    size: usize,
    block_size: usize,
    seq_ms: f64,
    bw_ms: f64,
    seq_labels: usize,
    bw_labels: usize,
    speedup: f64,
    label_match: bool,
}

fn bench_2d(
    scenario: &str,
    affs: &Array<f64, ndarray::IxDyn>,
    offsets: &[Vec<isize>],
    size: usize,
    block_sizes: &[usize],
) -> Vec<BenchResult> {
    let seeds = Array::from_elem(vec![size, size], 0usize);

    let t = Instant::now();
    let seq = agglomerate::<2>(affs, offsets.to_vec(), vec![], seeds.clone(), None, false);
    let seq_ms = t.elapsed().as_secs_f64() * 1000.0;
    let seq_labels: HashSet<usize> = seq.iter().copied().collect();

    block_sizes.iter().map(|&bs| {
        let bs = bs.min(size);
        let t = Instant::now();
        let bw = agglomerate_blockwise::<2>(affs, offsets.to_vec(), vec![], seeds.clone(), None, false, [bs, bs], None, false);
        let bw_ms = t.elapsed().as_secs_f64() * 1000.0;
        let bw_labels: HashSet<usize> = bw.iter().copied().collect();
        BenchResult {
            scenario: scenario.to_string(),
            dim: 2, size, block_size: bs,
            seq_ms, bw_ms,
            seq_labels: seq_labels.len(), bw_labels: bw_labels.len(),
            speedup: seq_ms / bw_ms,
            label_match: seq_labels.len() == bw_labels.len(),
        }
    }).collect()
}

fn bench_3d(
    scenario: &str,
    affs: &Array<f64, ndarray::IxDyn>,
    offsets: &[Vec<isize>],
    size: usize,
    block_sizes: &[usize],
) -> Vec<BenchResult> {
    let seeds = Array::from_elem(vec![size, size, size], 0usize);

    let t = Instant::now();
    let seq = agglomerate::<3>(affs, offsets.to_vec(), vec![], seeds.clone(), None, false);
    let seq_ms = t.elapsed().as_secs_f64() * 1000.0;
    let seq_labels: HashSet<usize> = seq.iter().copied().collect();

    block_sizes.iter().map(|&bs| {
        let bs = bs.min(size);
        let t = Instant::now();
        let bw = agglomerate_blockwise::<3>(affs, offsets.to_vec(), vec![], seeds.clone(), None, false, [bs, bs, bs], None, false);
        let bw_ms = t.elapsed().as_secs_f64() * 1000.0;
        let bw_labels: HashSet<usize> = bw.iter().copied().collect();
        BenchResult {
            scenario: scenario.to_string(),
            dim: 3, size, block_size: bs,
            seq_ms, bw_ms,
            seq_labels: seq_labels.len(), bw_labels: bw_labels.len(),
            speedup: seq_ms / bw_ms,
            label_match: seq_labels.len() == bw_labels.len(),
        }
    }).collect()
}

// ── Sizing calibration ──────────────────────────────────────────────

/// Find the largest size where sequential takes < max_ms
fn calibrate_2d<F: Fn(usize, u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>)>(
    gen: &F, max_ms: f64, seed: u64,
) -> usize {
    let mut size = 32;
    loop {
        let (affs, offsets) = gen(size, seed);
        let seeds = Array::from_elem(vec![size, size], 0usize);
        let t = Instant::now();
        let _ = agglomerate::<2>(&affs, offsets, vec![], seeds, None, false);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  calibrate 2D size={}: {:.1}ms", size, ms);
        if ms > max_ms { return size / 2; }
        if size >= 4096 { return size; }
        size *= 2;
    }
}

fn calibrate_3d<F: Fn(usize, u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>)>(
    gen: &F, max_ms: f64, seed: u64,
) -> usize {
    let mut size = 16;
    loop {
        let (affs, offsets) = gen(size, seed);
        let seeds = Array::from_elem(vec![size, size, size], 0usize);
        let t = Instant::now();
        let _ = agglomerate::<3>(&affs, offsets, vec![], seeds, None, false);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  calibrate 3D size={}: {:.1}ms", size, ms);
        if ms > max_ms { return size / 2; }
        if size >= 512 { return size; }
        size = (size as f64 * 1.5) as usize;
    }
}

fn main() {
    let max_ms = 250.0;
    let block_sizes = vec![2, 4, 8, 16, 32, 64, 128, 256, 512];

    println!("# Blockwise Mutex Watershed Benchmark Report\n");
    println!("Max sequential time per scenario: {:.0}ms", max_ms);
    println!("Available parallelism: {}\n", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    let mut all_results: Vec<BenchResult> = Vec::new();

    // ── 2D scenarios ──
    let scenarios_2d: Vec<(&str, Box<dyn Fn(usize, u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>)>)> = vec![
        ("2D random (worst: many splits)", Box::new(|s, seed| gen_random_2d(s, seed))),
        ("2D all-merge (best: one cluster)", Box::new(|s, seed| gen_all_merge_2d(s, seed))),
        ("2D small-objects (best for blockwise)", Box::new(|s, seed| gen_small_objects_2d(s, 8, seed))),
        ("2D large-objects (worst for blockwise)", Box::new(|s, seed| gen_large_objects_2d(s, seed))),
    ];

    for (name, gen) in &scenarios_2d {
        eprintln!("Calibrating {}...", name);
        let size = calibrate_2d(gen, max_ms, 42);
        eprintln!("  -> size={}\n", size);

        let (affs, offsets) = gen(size, 42);
        let valid_blocks: Vec<usize> = block_sizes.iter().copied()
            .filter(|&b| b <= size)
            .collect();
        let results = bench_2d(name, &affs, &offsets, size, &valid_blocks);
        all_results.extend(results);
    }

    // ── 3D scenarios ──
    let scenarios_3d: Vec<(&str, Box<dyn Fn(usize, u64) -> (Array<f64, ndarray::IxDyn>, Vec<Vec<isize>>)>)> = vec![
        ("3D random (worst: many splits)", Box::new(|s, seed| gen_random_3d(s, seed))),
        ("3D small-objects (best for blockwise)", Box::new(|s, seed| gen_small_objects_3d(s, 8, seed))),
        ("3D large-objects (worst for blockwise)", Box::new(|s, seed| gen_large_objects_3d(s, seed))),
    ];

    for (name, gen) in &scenarios_3d {
        eprintln!("Calibrating {}...", name);
        let size = calibrate_3d(gen, max_ms, 42);
        eprintln!("  -> size={}\n", size);

        let (affs, offsets) = gen(size, 42);
        let valid_blocks: Vec<usize> = block_sizes.iter().copied()
            .filter(|&b| b <= size)
            .collect();
        let results = bench_3d(name, &affs, &offsets, size, &valid_blocks);
        all_results.extend(results);
    }

    // ── Output table ──
    println!("## Results\n");
    println!("| Scenario | Dim | Size | Block | Seq(ms) | BW(ms) | Speedup | SeqLabels | BWLabels | Match |");
    println!("|----------|-----|------|-------|---------|--------|---------|-----------|----------|-------|");
    for r in &all_results {
        println!("| {} | {}D | {} | {} | {:.1} | {:.1} | {:.2}x | {} | {} | {} |",
            r.scenario, r.dim, r.size, r.block_size,
            r.seq_ms, r.bw_ms, r.speedup,
            r.seq_labels, r.bw_labels,
            if r.label_match { "✓" } else { "✗" });
    }

    // ── ASCII sparkline graphs ──
    println!("\n## Speedup by Block Size\n");

    let scenarios: Vec<String> = all_results.iter().map(|r| r.scenario.clone()).collect::<HashSet<_>>().into_iter().collect();
    let mut scenarios: Vec<String> = scenarios;
    scenarios.sort();

    for scenario in &scenarios {
        let results: Vec<&BenchResult> = all_results.iter().filter(|r| &r.scenario == scenario).collect();
        if results.is_empty() { continue; }

        println!("### {}\n", scenario);
        println!("  Size={}{}  Sequential={:.1}ms  Labels: seq={} ",
            results[0].size,
            if results[0].dim == 3 { "³" } else { "²" },
            results[0].seq_ms,
            results[0].seq_labels);
        println!();

        let max_speedup = results.iter().map(|r| r.speedup).fold(0.0f64, f64::max).max(1.5);
        let bar_width = 40;

        for r in &results {
            let bar_len = ((r.speedup / max_speedup) * bar_width as f64) as usize;
            let bar: String = "█".repeat(bar_len);
            let match_ch = if r.label_match { "✓" } else {
                &format!("✗({})", r.bw_labels)
            };
            println!("  bs={:>4} | {:<40} {:.2}x  {}", r.block_size, bar, r.speedup, match_ch);
        }
        println!();
    }
}
