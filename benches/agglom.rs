use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

use mwatershed::agglomerate; // replace with your actual crate name and function

const BENCH_SIZE: usize = 20;

fn bench_agglom(c: &mut Criterion) {
    // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
    let seed = 42;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    // Generate a random array using `rng`
    let affinities = Array::random_using(
        (2, BENCH_SIZE, BENCH_SIZE),
        Normal::new(0., 1.0).unwrap(),
        &mut rng,
    )
    .into_dyn();

    c.bench_function("agglomerate", |b| {
        b.iter(|| {
            agglomerate::<2>(
                black_box(&affinities),
                black_box(vec![vec![0, 1], vec![1, 0]]),
                black_box(vec![]),
                black_box(
                    Array::from_iter(0..BENCH_SIZE.pow(2))
                        .into_shape((BENCH_SIZE, BENCH_SIZE))
                        .unwrap()
                        .into_dyn(),
                ),
            )
        })
    });
}

criterion_group!(benches, bench_agglom);
criterion_main!(benches);
