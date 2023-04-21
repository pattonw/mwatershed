#![feature(test)]
#![feature(binary_heap_drain_sorted)]

extern crate test;

use ndarray::{Array, Dim, IxDynImpl};
use ndarray::{Dimension, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

use ordered_float::NotNan;
use std;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::convert::TryInto;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct AgglomEdge(NotNan<f64>, bool, usize, usize);

fn get_dims<const D: usize>(dim: Dim<IxDynImpl>, skip: usize) -> (Vec<usize>, [usize; D]) {
    (
        (0..skip).map(|i| dim[i]).collect(),
        (skip..D + skip)
            .map(|i| dim[i])
            .collect::<Vec<usize>>()
            .try_into()
            .expect("Wrong number of dimensions"),
    )
}

fn agglomerate<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<usize>>,
    edges: Vec<AgglomEdge>,
    mut seeds: Array<usize, IxDyn>,
) -> Array<usize, IxDyn> {
    let (_, array_shape) = get_dims::<D>(seeds.dim(), 0);
    let mut sorted_edges = BinaryHeap::new();
    let offsets: Vec<[usize; D]> = offsets
        .into_iter()
        .map(|offset| offset.try_into().unwrap())
        .collect();

    edges.into_iter().for_each(|edge| {
        sorted_edges.push(edge);
    });

    let num_nodes = seeds.iter().max().unwrap();

    affinities.indexed_iter().for_each(|(ind, aff)| {
        let (offset_indices, u_index) = get_dims::<D>(ind, 1);
        let offset_ind = *offset_indices.get(0).unwrap();

        let offset = offsets.get(offset_ind).unwrap();
        let v_index: [usize; D] = u_index
            .iter()
            .zip(offset.iter())
            .map(|(u_ind, o)| u_ind + o)
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap();
        let oob = v_index
            .iter()
            .zip(array_shape.iter())
            .map(|(a, b)| a >= b)
            .any(|x| x);
        match oob {
            false => sorted_edges.push(AgglomEdge(
                NotNan::new(aff.abs()).expect("Cannot handle `nan` affinities"),
                aff > &0.0,
                seeds[IxDyn(&u_index)],
                seeds[IxDyn(&v_index)],
            )),
            true => (),
        };
    });

    let mut node_to_cluster: Vec<usize> = (0..(num_nodes + 1)).into_iter().collect();
    let mut cluster_to_nodes: Vec<Vec<usize>> = (0..(num_nodes + 1))
        .into_iter()
        .map(|indx| vec![indx])
        .collect();
    let mut mutexes: Vec<HashSet<usize>> = (0..(num_nodes + 1))
        .into_iter()
        .map(|_| HashSet::new())
        .collect();

    sorted_edges.drain_sorted().for_each(|edge| {
        let AgglomEdge(_aff, pos, u, v) = edge;
        let u_cluster_id = *node_to_cluster.get(u).unwrap_or(&u);
        let v_cluster_id = *node_to_cluster.get(v).unwrap_or(&v);

        // println!("pos: {}, u: {}, u_cluster_id: {}, v: {}, v_cluster_id: {}, node_to_cluster: {:?}, cluster_to_nodes: {:?}", pos, u, u_cluster_id, v, v_cluster_id, node_to_cluster, cluster_to_nodes);

        let (new_id, old_id) = match u_cluster_id < v_cluster_id {
            true => (u_cluster_id, v_cluster_id),
            false => (v_cluster_id, u_cluster_id),
        };

        if new_id != old_id {
            let mutex_nodes = &mutexes[new_id];
            if !(mutex_nodes.contains(&old_id)) {
                match pos {
                    true => {
                        // update cluster ids
                        let old_cluster_nodes =
                            std::mem::replace(&mut cluster_to_nodes[old_id], vec![]);
                        old_cluster_nodes.iter().for_each(|node| {
                            node_to_cluster[*node] = new_id;
                        });
                        let new_cluster_nodes = cluster_to_nodes.get_mut(new_id).unwrap();
                        new_cluster_nodes.extend(old_cluster_nodes);

                        // merge_mutexes
                        let old_mutex_nodes =
                            std::mem::replace(&mut mutexes[old_id], HashSet::new());
                        for old_mutex_node in old_mutex_nodes.iter() {
                            mutexes[*old_mutex_node].remove(&old_id);
                            mutexes[*old_mutex_node].insert(new_id);
                        }
                        mutexes[new_id].extend(old_mutex_nodes);
                    }
                    false => {
                        mutexes[new_id].insert(old_id);
                        mutexes[old_id].insert(new_id);
                    }
                }
            }
        }
    });

    seeds.iter_mut().for_each(|x| {
        *x = *node_to_cluster.get(*x).unwrap();
    });

    return seeds;
}

/// agglomerate nodes given an array of affinities and optional additional edges
#[pyfunction()]
fn agglom<'py>(
    _py: Python<'py>,
    affinities: &PyArrayDyn<f64>,
    offsets: Vec<Vec<usize>>,
    seeds: Option<&PyArrayDyn<usize>>,
    edges: Option<Vec<(usize, usize, f64)>>,
) -> PyResult<&'py PyArrayDyn<usize>> {
    let affinities = unsafe { affinities.as_array() }.to_owned();
    let seeds = unsafe { seeds.expect("Seeds not provided!").as_array() }.to_owned();
    let dim = seeds.dim().ndim();
    let edges: Vec<AgglomEdge> = edges
        .unwrap_or(vec![])
        .into_iter()
        .map(|(u, v, aff)| AgglomEdge(NotNan::new(aff.abs()).unwrap(), aff > 0.0, u, v))
        .collect();
    let result = match dim {
        1 => agglomerate::<1>(&affinities, offsets, edges, seeds),
        2 => agglomerate::<2>(&affinities, offsets, edges, seeds),
        3 => agglomerate::<3>(&affinities, offsets, edges, seeds),
        4 => agglomerate::<4>(&affinities, offsets, edges, seeds),
        5 => agglomerate::<5>(&affinities, offsets, edges, seeds),
        6 => agglomerate::<6>(&affinities, offsets, edges, seeds),
        _ => panic!["Only 1-6 dimensional arrays supported"],
    };
    Ok(result.into_pyarray(_py))
}

#[pymodule]
fn mwatershed(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(agglom))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use ndarray::array;
    use ndarray::Array;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use rand_isaac::isaac64::Isaac64Rng;
    use test::Bencher;

    static BENCH_SIZE: usize = 200;

    #[test]
    fn test_agglom() {
        let affinities = array![
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn();
        let offsets = vec![vec![0, 1], vec![1, 0]];
        let components = agglomerate::<2>(&affinities, offsets, vec![], seeds);
        println!("{:?}", components);
        assert!(components.into_iter().unique().collect::<Vec<usize>>() == vec![1, 2, 4, 5]);
    }
    #[bench]
    fn bench_agglom(b: &mut Bencher) {
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

        b.iter(|| {
            agglomerate::<2>(
                &affinities,
                vec![vec![0, 1], vec![1, 0]],
                vec![],
                Array::from_iter(0..BENCH_SIZE.pow(2))
                    .into_shape((BENCH_SIZE, BENCH_SIZE))
                    .unwrap()
                    .into_dyn(),
            )
        });
    }
}
