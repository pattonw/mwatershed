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

fn get_dims<const D: usize>(dim: Dim<IxDynImpl>) -> [usize; D] {
    (0..D)
        .map(|i| dim[i])
        .collect::<Vec<usize>>()
        .try_into()
        .expect("Wrong number of dimensions")
}

fn agglomerate<const D: usize, const T: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<usize>>,
    edges: Vec<AgglomEdge>,
    mut seeds: Array<usize, IxDyn>,
    bias: f64,
) -> Array<usize, IxDyn> {
    let array_shape = get_dims::<D>(seeds.dim());
    let mut sorted_edges = BinaryHeap::new();

    edges.into_iter().for_each(|edge| {
        sorted_edges.push(edge);
    });

    affinities.indexed_iter().for_each(|(ind, aff)| {
        let aff_index = get_dims::<T>(ind);
        let offset_ind = aff_index[0];
        let u_index: [usize; D] = aff_index[1..T].try_into().expect(&format!(
            "Affinities does not have an dimension {:?} + 1",
            D
        ));
        let offset: [usize; D] = offsets[offset_ind]
            .clone()
            .try_into()
            .expect("Wrong number of dimensions!");
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
                NotNan::new((aff - bias).abs()).expect("Cannot handle `nan` affinities"),
                aff > &bias,
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
    bias: Option<f64>,
    seeds: Option<&PyArrayDyn<usize>>,
    edges: Option<Vec<(usize, usize, f64)>>,
) -> PyResult<&'py PyArrayDyn<usize>> {
    let affinities = unsafe { affinities.as_array() }.to_owned();
    let seeds = unsafe { seeds.expect("Seeds not provided!").as_array() }.to_owned();
    let dim = seeds.dim().ndim();
    let bias = bias.unwrap_or(0.5);
    let edges: Vec<AgglomEdge> = edges
        .unwrap_or(vec![])
        .into_iter()
        .map(|(u, v, aff)| {
            let biased_aff = aff - bias;
            AgglomEdge(
                NotNan::new(biased_aff.abs()).unwrap(),
                biased_aff > 0.0,
                u,
                v,
            )
        })
        .collect();
    let result = match dim {
        1 => agglomerate::<1, 2>(&affinities, offsets, edges, seeds, bias),
        2 => agglomerate::<2, 3>(&affinities, offsets, edges, seeds, bias),
        3 => agglomerate::<3, 4>(&affinities, offsets, edges, seeds, bias),
        _ => panic!["Only 1-3 dimensional arrays supported"],
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
        .into_dyn();
        let seeds = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn();
        let offsets = vec![vec![0, 1], vec![1, 0]];
        let components = agglomerate::<2, 3>(&affinities, offsets, vec![], seeds, 0.5);
        assert!(components.into_iter().unique().collect::<Vec<usize>>() == vec![1, 2, 4, 5]);
    }
}
