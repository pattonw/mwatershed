#![feature(test)]
#![feature(binary_heap_drain_sorted)]
#![feature(extend_one)]

extern crate test;
mod lib;
use crate::lib::*;
use itertools::Itertools;
use ndarray::array;
use ndarray::Array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

static BENCH_SIZE: usize = 500;

fn main() {
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

    agglomerate::<2>(
        &affinities,
        vec![vec![0, 1], vec![1, 0]],
        vec![],
        Array::from_iter(0..BENCH_SIZE.pow(2))
            .into_shape((BENCH_SIZE, BENCH_SIZE))
            .unwrap()
            .into_dyn(),
    );
}
