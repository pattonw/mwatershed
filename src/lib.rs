#![feature(test)]
#![feature(binary_heap_drain_sorted)]
#![feature(extend_one)]

use disjoint_sets::UnionFind;

use itertools::Itertools;
use ndarray::{Array, Axis, Slice};
use ndarray::{Dimension, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

use ordered_float::NotNan;
use std;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

#[derive(Debug)]
pub struct AgglomEdge(bool, usize, usize);

pub fn get_edges<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<usize>>,
    seeds: &Array<usize, IxDyn>,
) -> Vec<AgglomEdge> {
    // let (_, array_shape) = get_dims::<D>(seeds.dim(), 0);
    let offsets: Vec<[usize; D]> = offsets
        .into_iter()
        .map(|offset| offset.try_into().unwrap())
        .collect();

    let mut edges = Vec::with_capacity(affinities.len());
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
        .sorted_unstable_by(|a, b| Ord::cmp(&b.0, &a.0))
        .map(|(_aff, edge)| edge)
        .collect()
}

struct Negatives {
    mutexes: Vec<HashSet<usize>>,
}
impl Negatives {
    fn new(num_nodes: usize) -> Negatives {
        Negatives {
            mutexes: (0..(num_nodes)).map(|_| HashSet::new()).collect(),
        }
    }

    fn mutex(&self, a: usize, b: usize) -> bool {
        self.mutexes[a].contains(&b)
    }

    fn merge(&mut self, a: usize, b: usize) {
        // empty b mutex edges
        let b_mutexes = std::mem::replace(&mut self.mutexes[b], HashSet::with_capacity(0));
        // for each x-b mutex edge, replace with x-a edge
        for mutex_node in b_mutexes.iter() {
            self.mutexes[*mutex_node].remove(&b);
            let inserted = self.mutexes[*mutex_node].insert(a);
            if inserted {
                self.mutexes[a].insert(*mutex_node);
            }
        }
    }

    fn insert(&mut self, a: usize, b: usize) {
        self.mutexes[a].insert(b);
        self.mutexes[b].insert(a);
    }
}
struct Positives {
    clusters: UnionFind<usize>,
}
impl Positives {
    fn new(num_nodes: usize) -> Positives {
        Positives {
            clusters: UnionFind::new(num_nodes),
        }
    }

    fn merge(&mut self, a: usize, b: usize) {
        self.clusters.union(a, b);
    }
}

struct Clustering {
    positives: Positives,
    negatives: Negatives,
}

impl Clustering {
    fn new(num_nodes: usize) -> Clustering {
        return Clustering {
            positives: Positives::new(num_nodes),
            negatives: Negatives::new(num_nodes),
        };
    }

    fn merge(&mut self, a: usize, b: usize) {
        self.positives.merge(a, b);
        let rep = self.positives.clusters.find(a);
        match a == rep {
            true => self.negatives.merge(a, b),
            false => self.negatives.merge(b, a),
        }
    }

    fn process_edges(&mut self, sorted_edges: Vec<AgglomEdge>) {
        sorted_edges.into_iter().for_each(|edge| {
            let AgglomEdge(pos, u, v) = edge;
            if !self.positives.clusters.equiv(u, v) {
                let u_rep = self.positives.clusters.find(u);
                let v_rep = self.positives.clusters.find(v);

                if !self.negatives.mutex(u_rep, v_rep) {
                    let (new_id, old_id) = match u_rep < v_rep {
                        true => (u_rep, v_rep),
                        false => (v_rep, u_rep),
                    };
                    match pos {
                        true => self.merge(new_id, old_id),
                        false => {
                            self.negatives.insert(new_id, old_id);
                        }
                    }
                }
            }
        });
    }

    fn map(&self, seeds: &mut Array<usize, IxDyn>) {
        seeds.iter_mut().for_each(|x| {
            *x = self.positives.clusters.find(*x);
        });
    }
}

pub fn agglomerate<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<usize>>,
    mut edges: Vec<AgglomEdge>,
    mut seeds: Array<usize, IxDyn>,
) -> Array<usize, IxDyn> {
    // relabel to consecutive ids
    let mut lookup = HashMap::new();
    seeds.iter_mut().enumerate().for_each(|(ind, x)| match x {
        0 => *x = ind,
        k => {
            let id = *lookup.get(k).unwrap_or(&ind);
            lookup.insert(*k, id);
            *k = id;
        }
    });

    let unique_seeds: Vec<&usize> = lookup.values().collect();
    (0..unique_seeds.len()).for_each(|id1| {
        ((id1 + 1)..unique_seeds.len()).for_each(|id2| {
            edges.push(AgglomEdge(false, *unique_seeds[id1], *unique_seeds[id2]));
        })
    });

    // main algorithm
    let sorted_edges = get_edges::<D>(affinities, offsets, &seeds);
    edges.extend(sorted_edges);

    let mut clustering = Clustering::new(seeds.len());

    clustering.process_edges(edges);
    clustering.map(&mut seeds);

    // TODO: Fix seed handling
    // now we have to remap seeded entries back onto the original ids
    let mut rev_lookup = HashMap::with_capacity(lookup.len());
    //rev_lookup.insert(0, seeds.len());
    lookup.iter().for_each(|(seed, id)| {
        let rep_id = clustering.positives.clusters.find(*id);
        if *seed != rep_id {
            rev_lookup.insert(rep_id, *seed);
        }
    });

    seeds.iter_mut().for_each(|x| {
        *x = *rev_lookup.get(x).unwrap_or(x);
    });

    return seeds;
}

pub fn cluster_edges(mut sorted_edges: Vec<AgglomEdge>) -> Vec<(usize, usize)> {
    // relabel to consecutive ids
    let mut lookup = HashMap::new();
    let mut next_id = 0..;
    sorted_edges.iter_mut().for_each(|edge| {
        let AgglomEdge(_pos, u, v) = edge;
        let serialized_u = match lookup.get(u) {
            Some(&serialized_u) => serialized_u,
            None => next_id.next().unwrap(),
        };
        lookup.insert(*u, serialized_u);
        let serialized_v = match lookup.get(v) {
            Some(&serialized_v) => serialized_v,
            None => next_id.next().unwrap(),
        };
        lookup.insert(*v, serialized_v);
        *u = serialized_u;
        *v = serialized_v;
    });

    // main algorithm
    let mut clustering = Clustering::new(lookup.len());

    clustering.process_edges(sorted_edges);

    let mut rev_lookup = HashMap::with_capacity(lookup.len());
    lookup.iter().for_each(|(seed, id)| {
        let rep_id = clustering.positives.clusters.find(*id);
        rev_lookup.insert(rep_id, *seed);
    });

    lookup
        .into_iter()
        .map(|(og, serialized)| {
            (
                og,
                *rev_lookup
                    .get(&clustering.positives.clusters.find(serialized))
                    .unwrap(),
            )
        })
        .collect()
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
    let seeds = match seeds {
        Some(seeds) => unsafe { seeds.as_array() }.to_owned(),
        None => Array::zeros(&affinities.shape()[1..]),
    };
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

/// agglomerate nodes given an array of affinities and optional additional edges
#[pyfunction()]
fn cluster(edges: Vec<(bool, usize, usize)>) -> PyResult<Vec<(usize, usize)>> {
    let edges: Vec<AgglomEdge> = edges
        .into_iter()
        .map(|(pos, u, v)| AgglomEdge(pos, u, v))
        .collect();
    let result = cluster_edges(edges);
    Ok(result)
}

#[pymodule]
fn mwatershed(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(agglom))?;
    m.add_wrapped(wrap_pyfunction!(cluster))?;

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
        let seeds = array![[1, 2, 0], [4, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, 1], vec![1, 0]];
        let components = agglomerate::<2>(&affinities, offsets, vec![], seeds);
        let ids = components
            .clone()
            .into_iter()
            .unique()
            .collect::<Vec<usize>>();
        for id in [1, 2, 4].iter() {
            assert!(ids.contains(id));
        }
        assert!(ids.len() == 4);
    }
    #[test]
    fn test_cluster() {
        let edges = vec![
            AgglomEdge(false, 1, 2),
            AgglomEdge(true, 1, 3),
            AgglomEdge(true, 2, 3),
            AgglomEdge(false, 1, 3),
        ];
        let matching = cluster_edges(edges);
        panic!["{:?}", matching];
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
