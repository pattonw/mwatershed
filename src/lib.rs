#![feature(test)]
#![feature(binary_heap_drain_sorted)]

use ndarray::{Array, Dim, IxDynImpl};
use ndarray::{Dimension, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn};

use itertools::Unique;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

use std;

use std::convert::TryInto;

use std::collections::{BinaryHeap, HashMap, HashSet};

use ordered_float::NotNan;

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

    let mut node_to_cluster = HashMap::new();
    let mut cluster_to_nodes = HashMap::new();
    let mut unique_clusters = HashSet::new();

    sorted_edges.drain_sorted().for_each(|edge| {
        let AgglomEdge(_aff, pos, u, v) = edge;
        let u_cluster_id = *node_to_cluster.get(&u).unwrap_or(&u);
        let v_cluster_id = *node_to_cluster.get(&v).unwrap_or(&v);

        match pos {
            true => {
                if !(unique_clusters.contains(&u_cluster_id)
                    && unique_clusters.contains(&v_cluster_id))
                {
                    let new_id = *vec![u, v, u_cluster_id, v_cluster_id].iter().min().unwrap();
                    let mut u_cluster_nodes = cluster_to_nodes
                        .remove(&u_cluster_id)
                        .unwrap_or(HashSet::from([u_cluster_id]));
                    let v_cluster_nodes = cluster_to_nodes
                        .remove(&v_cluster_id)
                        .unwrap_or(HashSet::from([v_cluster_id]));
                    if !(u == new_id) {
                        u_cluster_nodes.iter().for_each(|node| {
                            node_to_cluster.insert(*node, new_id);
                        })
                    }
                    if !(v == new_id) {
                        v_cluster_nodes.iter().for_each(|node| {
                            node_to_cluster.insert(*node, new_id);
                        })
                    }
                    u_cluster_nodes.extend(&v_cluster_nodes);
                    cluster_to_nodes.insert(new_id, u_cluster_nodes);
                }
            }
            false => {
                unique_clusters.insert(u_cluster_id);
                unique_clusters.insert(v_cluster_id);
            }
        }
    });

    seeds.iter_mut().for_each(|x| {
        *x = *node_to_cluster.get(&x).unwrap_or(&x);
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
        assert!(components.into_iter().unique().collect::<Vec<usize>>() == vec![1, 2, 4, 5]);
    }
}
