#![feature(test)]
#![feature(binary_heap_drain_sorted)]
#![feature(extend_one)]

use itertools::{sorted, Itertools};
use ndarray::{Array, Axis, Dim, IxDynImpl, Slice};
use ndarray::{Dimension, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

use ordered_float::NotNan;
use std;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::convert::TryInto;
use std::ops::Index;

#[derive(Debug)]
pub struct AgglomEdge(bool, usize, usize);

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
    mut edges: Vec<AgglomEdge>,
    seeds: &Array<usize, IxDyn>,
) -> Vec<AgglomEdge> {
    let (_, array_shape) = get_dims::<D>(seeds.dim(), 0);
    let offsets: Vec<[usize; D]> = offsets
        .into_iter()
        .map(|offset| offset.try_into().unwrap())
        .collect();

    edges.extend_reserve(affinities.len());
    let mut affs = Vec::with_capacity(edges.capacity());

    offsets
        .iter()
        .enumerate()
        .for_each(|(offset_index, offset)| {
            let all_offset_affs = affinities.index_axis(Axis(0), offset_index);
            let offset_affs = all_offset_affs
                .slice_each_axis(|ax| Slice::from(0..(ax.len - offset[ax.axis.index()])));
            let u_seeds =
                seeds.slice_each_axis(|ax| Slice::from(0..(ax.len - offset[ax.axis.index()])));
            let v_seeds = seeds.slice_each_axis(|ax| Slice::from(offset[ax.axis.index()]..));
            offset_affs.indexed_iter().for_each(|(index, aff)| {
                let u = u_seeds[&index];
                let v = v_seeds[&index];
                affs.push(NotNan::new(aff.abs()).expect("Cannot handle `nan` affinities"));
                edges.push(AgglomEdge(aff > &0.0, u, v))
            });
        });
    affs.into_iter()
        .zip(edges.into_iter())
        .sorted_unstable_by(|a, b| Ord::cmp(&a.0, &b.0))
        .map(|(aff, edge)| edge)
        .collect()
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
        // for each x-b mutex edge, replace with x-a edge
        for mutex_node in b_mutexes.iter() {
            self.mutexes[*mutex_node].remove(&b);
            let inserted = self.mutexes[*mutex_node].insert(a);
            if inserted {
                self.mutexes[a].insert(*mutex_node);
            }
        }
    }

    fn insert_mutex(&mut self, a: usize, b: usize) {
        self.mutexes[a].insert(b);
        self.mutexes[b].insert(a);
    }

    fn process_edges(&mut self, sorted_edges: Vec<AgglomEdge>) {
        sorted_edges.into_iter().for_each(|edge| {
            let AgglomEdge(pos, u, v) = edge;

            let u_cluster_id = self.cluster_id(u);
            let v_cluster_id = self.cluster_id(v);

            let (new_id, old_id) = match self.cluster_to_nodes[u_cluster_id].len()
                > self.cluster_to_nodes[v_cluster_id].len()
            {
                true => (u_cluster_id, v_cluster_id),
                false => (v_cluster_id, u_cluster_id),
            };

            if new_id != old_id {
                if !(self.mutex(new_id, old_id)) {
                    match pos {
                        true => {
                            self.merge_clusters(new_id, old_id);
                            self.merge_mutex(new_id, old_id);
                        }
                        false => {
                            self.insert_mutex(new_id, old_id);
                        }
                    }
                }
            }
        });
    }

    fn map(&self, seeds: &mut Array<usize, IxDyn>) {
        seeds.iter_mut().for_each(|x| {
            *x = self.node_to_cluster[*x];
        });
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

    clustering.process_edges(sorted_edges);

    clustering.map(&mut seeds);

    return seeds;
}

/// agglomerate nodes given an array of affinities and optional additional edges
#[pyfunction()]
fn agglom<'py>(
    _py: Python<'py>,
    affinities: &PyArrayDyn<f64>,
    offsets: Vec<Vec<usize>>,
    seeds: Option<&PyArrayDyn<usize>>,
    edges: Option<Vec<(bool, usize, usize)>>,
) -> PyResult<&'py PyArrayDyn<usize>> {
    let affinities = unsafe { affinities.as_array() }.to_owned();
    let seeds = unsafe { seeds.expect("Seeds not provided!").as_array() }.to_owned();
    let dim = seeds.dim().ndim();
    let edges: Vec<AgglomEdge> = edges
        .unwrap_or(vec![])
        .into_iter()
        .map(|(pos, u, v)| AgglomEdge(pos, u, v))
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
    // use test::Bencher;

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
    // #[bench]
    // fn bench_agglom(b: &mut Bencher) {
    //     // Get a seeded random number generator for reproducibility (Isaac64 algorithm)
    //     let seed = 42;
    //     let mut rng = Isaac64Rng::seed_from_u64(seed);

    //     // Generate a random array using `rng`
    //     let affinities = Array::random_using(
    //         (2, BENCH_SIZE, BENCH_SIZE),
    //         Normal::new(0., 1.0).unwrap(),
    //         &mut rng,
    //     )
    //     .into_dyn();

    //     b.iter(|| {
    //         agglomerate::<2>(
    //             &affinities,
    //             vec![vec![0, 1], vec![1, 0]],
    //             vec![],
    //             Array::from_iter(0..BENCH_SIZE.pow(2))
    //                 .into_shape((BENCH_SIZE, BENCH_SIZE))
    //                 .unwrap()
    //                 .into_dyn(),
    //         )
    //     });
    // }
}
