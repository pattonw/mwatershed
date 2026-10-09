use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array, IxDyn};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

use mwatershed::{agglomerate, agglomerate_with, GaspOptions, Linkage};

const BENCH_SIZE: usize = 64;

fn random_affinities(shape: &[usize], seed: u64) -> Array<f64, IxDyn> {
    let mut rng = Isaac64Rng::seed_from_u64(seed);
    Array::random_using(IxDyn(shape), Normal::new(0., 1.0).unwrap(), &mut rng)
}

fn bench_agglom(c: &mut Criterion) {
    let affinities = random_affinities(&[2, BENCH_SIZE, BENCH_SIZE], 42);
    let seeds: Array<usize, IxDyn> = Array::zeros(IxDyn(&[BENCH_SIZE, BENCH_SIZE]));

    c.bench_function("agglomerate", |b| {
        b.iter(|| {
            agglomerate::<2>(
                black_box(&affinities),
                black_box(vec![vec![0, 1], vec![1, 0]]),
                black_box(vec![]),
                black_box(seeds.clone()),
                None,
                false,
            )
        })
    });
}

/// Compare every linkage criterion on the same random 2D and 3D affinities.
fn bench_linkages(c: &mut Criterion) {
    let cases: Vec<(&str, Vec<Vec<isize>>, Vec<usize>)> = vec![
        (
            "2d_64",
            vec![vec![0, 1], vec![1, 0], vec![0, 4], vec![4, 0]],
            vec![BENCH_SIZE, BENCH_SIZE],
        ),
        (
            "3d_24",
            vec![
                vec![0, 0, 1],
                vec![0, 1, 0],
                vec![1, 0, 0],
                vec![0, 0, 3],
                vec![0, 3, 0],
                vec![3, 0, 0],
            ],
            vec![24, 24, 24],
        ),
    ];

    let mut group = c.benchmark_group("linkage");
    for (name, offsets, shape) in cases {
        let mut full_shape = vec![offsets.len()];
        full_shape.extend(&shape);
        let affinities = random_affinities(&full_shape, 7);
        let seeds: Array<usize, IxDyn> = Array::zeros(IxDyn(&shape));
        for linkage in Linkage::ALL {
            for cannot_link in [true, false] {
                if linkage == Linkage::AbsMax && !cannot_link {
                    continue; // identical path and result
                }
                let id = format!("{name}/{linkage}{}", if cannot_link { "" } else { "_nocl" });
                let options = GaspOptions::new(linkage, cannot_link);
                group.bench_with_input(BenchmarkId::from_parameter(id), &options, |b, options| {
                    b.iter(|| match shape.len() {
                        2 => agglomerate_with::<2>(
                            black_box(&affinities),
                            black_box(offsets.clone()),
                            vec![],
                            seeds.clone(),
                            None,
                            false,
                            *options,
                        ),
                        3 => agglomerate_with::<3>(
                            black_box(&affinities),
                            black_box(offsets.clone()),
                            vec![],
                            seeds.clone(),
                            None,
                            false,
                            *options,
                        ),
                        _ => unreachable!(),
                    })
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_agglom, bench_linkages);
criterion_main!(benches);
