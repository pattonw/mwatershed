#![feature(test)]
#![feature(binary_heap_drain_sorted)]

use itertools::{sorted, Itertools};
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
pub struct AgglomEdge(NotNan<f64>, bool, usize, usize);

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

pub fn get_edges<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<usize>>,
    edges: Vec<AgglomEdge>,
    seeds: &Array<usize, IxDyn>,
) -> Vec<AgglomEdge> {
    let (_, array_shape) = get_dims::<D>(seeds.dim(), 0);
    let mut sorted_edges = Vec::with_capacity(affinities.len() + edges.len());
    let offsets: Vec<[usize; D]> = offsets
        .into_iter()
        .map(|offset| offset.try_into().unwrap())
        .collect();

    edges.into_iter().for_each(|edge| {
        sorted_edges.push(edge);
    });

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
    return sorted_edges.into_iter().sorted_unstable().collect();
}

struct Clustering {
    node_to_cluster: Vec<usize>,
    cluster_to_nodes: Vec<Vec<usize>>,
    mutexes: Vec<HashSet<usize>>,
}

impl Clustering {
    fn new(num_nodes: usize) -> Clustering {
        return Clustering {
            node_to_cluster: (0..(num_nodes + 1)).collect(),
            cluster_to_nodes: (0..(num_nodes + 1)).map(|indx| vec![indx]).collect(),
            mutexes: (0..(num_nodes + 1)).map(|_| HashSet::new()).collect(),
        };
    }

    fn cluster_id(&self, node: usize) -> usize {
        *self.node_to_cluster.get(node).unwrap()
    }

    fn mutex(&self, a: usize, b: usize) -> bool {
        self.mutexes[a].contains(&b)
    }

    fn merge_clusters(&mut self, a: usize, b: usize) {
        // replace b cluster with empty vector
        let b_nodes = std::mem::replace(&mut self.cluster_to_nodes[b], vec![]);
        // update every b node to point to a cluster id
        b_nodes.iter().for_each(|node| {
            self.node_to_cluster[*node] = a;
        });
        // update a cluster to contain b cluster nodes
        self.cluster_to_nodes.get_mut(a).unwrap().extend(b_nodes);
    }

    fn merge_mutex(&mut self, a: usize, b: usize) {
        // empty b mutex edges
        let b_mutexes = std::mem::replace(&mut self.mutexes[b], HashSet::new());
        // for each b-x mutex edge, replace with a-x edge
        for mutex_node in b_mutexes.iter() {
            self.mutexes[*mutex_node].remove(&b);
            self.mutexes[*mutex_node].insert(a);
        }
        // update a mutexes to include all b mutexes
        self.mutexes[a].extend(b_mutexes);
    }

    fn insert_mutex(&mut self, a: usize, b: usize) {
        self.mutexes[a].insert(b);
        self.mutexes[b].insert(a);
    }

    fn map(&self, x: usize) -> usize {
        self.node_to_cluster[x]
    }
}

pub fn agglomerate<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<usize>>,
    edges: Vec<AgglomEdge>,
    mut seeds: Array<usize, IxDyn>,
) -> Array<usize, IxDyn> {
    let num_nodes = seeds.len() + 1;

    let sorted_edges = get_edges::<D>(affinities, offsets, edges, &seeds);

    let mut clustering = Clustering::new(num_nodes);

    sorted_edges.into_iter().for_each(|edge| {
        let AgglomEdge(_aff, pos, u, v) = edge;

        let u_cluster_id = clustering.cluster_id(u);
        let v_cluster_id = clustering.cluster_id(v);

        let (new_id, old_id) = match u_cluster_id < v_cluster_id {
            true => (u_cluster_id, v_cluster_id),
            false => (v_cluster_id, u_cluster_id),
        };

        if new_id != old_id {
            if !(clustering.mutex(new_id, old_id)) {
                match pos {
                    true => {
                        clustering.merge_clusters(new_id, old_id);
                        clustering.merge_mutex(new_id, old_id);
                    }
                    false => {
                        clustering.insert_mutex(new_id, old_id);
                    }
                }
            }
        }
    });

    seeds.iter_mut().for_each(|x| {
        *x = clustering.map(*x);
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

    static BENCH_SIZE: usize = 50;

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
