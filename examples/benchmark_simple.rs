use ndarray::Array;
use std::collections::HashSet;
use std::time::Instant;

use mwatershed::{agglomerate, agglomerate_blockwise};

fn main() {
    let size = 100usize;
    let tile = 10usize;
    let num_voxels = size * size;

    let tile_id = |r: usize, c: usize| -> usize { (r / tile) * (size / tile) + (c / tile) };

    let mut aff_data = Vec::with_capacity(2 * num_voxels);
    for r in 0..size {
        for c in 0..size {
            if c + 1 < size && tile_id(r, c) == tile_id(r, c + 1) {
                aff_data.push(1.0);
            } else {
                aff_data.push(-1.0);
            }
        }
    }
    for r in 0..size {
        for c in 0..size {
            if r + 1 < size && tile_id(r, c) == tile_id(r + 1, c) {
                aff_data.push(1.0);
            } else {
                aff_data.push(-1.0);
            }
        }
    }

    let affinities = Array::from_shape_vec(vec![2, size, size], aff_data).unwrap();
    let seeds = Array::from_elem(vec![size, size], 0usize);
    let offsets = vec![vec![0isize, 1], vec![1, 0]];

    let result_seq = agglomerate::<2>(&affinities, offsets.clone(), vec![], seeds.clone(), None, false);
    let unique_seq: HashSet<usize> = result_seq.iter().copied().collect();
    println!("Sequential: {} unique labels", unique_seq.len());

    let result_bw = agglomerate_blockwise::<2>(&affinities, offsets.clone(), vec![], seeds.clone(), None, false, [25, 25], None, false);
    let unique_bw: HashSet<usize> = result_bw.iter().copied().collect();
    println!("Blockwise:  {} unique labels", unique_bw.len());
}
