#![feature(test)]
#![feature(binary_heap_drain_sorted)]
#![feature(extend_one)]

use itertools::Itertools;
use ndarray::{Array, Axis, Slice};
use ndarray::{Dimension, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

use ndarray_rand::rand;

use ordered_float::NotNan;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::iter::FromIterator;

pub mod blockwise;
pub mod clustering;
mod labels;

use clustering::{AgglomEdge, Clustering};

pub fn get_edges<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    seeds: &Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
) -> (Vec<AgglomEdge>, HashSet<usize>) {
    // let (_, array_shape) = get_dims::<D>(seeds.dim(), 0);
    let offsets: Vec<[isize; D]> = offsets
        .into_iter()
        .map(|offset| offset.try_into().unwrap())
        .collect();

    let mut edges = Vec::with_capacity(affinities.len());
    edges.extend_reserve(affinities.len());
    let mut affs = Vec::with_capacity(edges.capacity());

    let mut to_filter: HashSet<usize> = HashSet::from_iter(seeds.iter().copied());

    let strides = strides.unwrap_or_else(|| {
        (0..offsets.len())
            .map(|_| (0..D).map(|_| 1).collect())
            .collect::<Vec<Vec<usize>>>()
    });

    offsets
        .iter()
        .zip(strides.iter())
        .enumerate()
        .for_each(|(offset_index, (offset, stride))| {
            let all_offset_affs = affinities.index_axis(Axis(0), offset_index);
            let offset_affs = all_offset_affs.slice_each_axis(|ax| {
                let step = if randomized_strides {
                    1
                } else {
                    stride[ax.axis.index()].try_into().unwrap()
                };
                Slice::new(
                    std::cmp::max(0, -offset[ax.axis.index()]),
                    Some(std::cmp::min(
                        ax.len as isize,
                        (ax.len as isize) - offset[ax.axis.index()],
                    )),
                    step,
                )
            });
            let u_seeds = seeds.slice_each_axis(|ax| {
                let step = if randomized_strides {
                    1
                } else {
                    stride[ax.axis.index()].try_into().unwrap()
                };
                Slice::new(
                    std::cmp::max(0, -offset[ax.axis.index()]),
                    Some(std::cmp::min(
                        ax.len as isize,
                        (ax.len as isize) - offset[ax.axis.index()],
                    )),
                    step,
                )
            });
            let v_seeds = seeds.slice_each_axis(|ax| {
                let step = if randomized_strides {
                    1
                } else {
                    stride[ax.axis.index()].try_into().unwrap()
                };
                Slice::new(
                    std::cmp::max(0, offset[ax.axis.index()]),
                    Some(std::cmp::min(
                        ax.len as isize,
                        (ax.len as isize) + offset[ax.axis.index()],
                    )),
                    step,
                )
            });
            offset_affs.indexed_iter().for_each(|(index, aff)| {
                if !randomized_strides
                    || rand::random::<f32>() < 1.0 / stride.iter().product::<usize>() as f32
                {
                    let u = u_seeds[&index];
                    let v = v_seeds[&index];
                    affs.push(NotNan::new(aff.abs()).expect("Cannot handle `nan` affinities"));
                    edges.push(AgglomEdge(aff > &0.0, u, v))
                }
            });
        });
    let agglom_edges: Vec<AgglomEdge> = affs
        .into_iter()
        .zip(edges)
        .sorted_unstable_by(|a, b| Ord::cmp(&b.0, &a.0))
        .map(|(_aff, edge)| edge)
        .collect();
    agglom_edges.iter().for_each(|edge| {
        let AgglomEdge(pos, u, v) = edge;
        if *pos {
            to_filter.remove(u);
            to_filter.remove(v);
        }
    });
    (
        agglom_edges
            .into_iter()
            .filter(|edge| !(to_filter.contains(&edge.1) || to_filter.contains(&edge.2)))
            .collect(),
        to_filter,
    )
}

pub fn agglomerate<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    mut edges: Vec<AgglomEdge>,
    mut seeds: Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
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

    // update edges from seed ids to node ids:
    edges.iter_mut().for_each(|AgglomEdge(_pos, u, v)| {
        *u = lookup[u];
        *v = lookup[v];
    });

    let unique_seeds: Vec<&usize> = lookup.values().collect();
    (0..unique_seeds.len()).for_each(|id1| {
        ((id1 + 1)..unique_seeds.len()).for_each(|id2| {
            edges.push(AgglomEdge(false, *unique_seeds[id1], *unique_seeds[id2]));
        })
    });

    // main algorithm
    let (sorted_edges, mut filtered_background) =
        get_edges::<D>(affinities, offsets, &seeds, strides, randomized_strides);
    edges.extend(sorted_edges);
    lookup.values().for_each(|node_id| {
        filtered_background.remove(node_id);
    });

    let mut clustering = Clustering::new(seeds.len());

    clustering.process_edges(edges);
    clustering.filter_map(&mut seeds, filtered_background);

    // TODO: Fix seed handling
    // now we have to remap seeded entries back onto the original ids
    let mut rev_lookup = HashMap::with_capacity(lookup.len());
    rev_lookup.insert(seeds.len(), 0);
    lookup.iter().for_each(|(seed, id)| {
        let rep_id = clustering.positives.clusters.find(*id);
        if *seed != rep_id {
            rev_lookup.insert(rep_id, *seed);
        }
    });

    seeds.iter_mut().for_each(|x| {
        *x = *rev_lookup.get(x).unwrap_or(x);
    });

    seeds
}

pub fn agglomerate_blockwise<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    edges: Vec<AgglomEdge>,
    seeds: Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
    block_shape: [usize; D],
    context: Option<[usize; D]>,
    lossy: bool,
) -> Array<usize, IxDyn> {
    agglomerate_blockwise_pyramid(affinities, offsets, edges, seeds, strides, randomized_strides, vec![block_shape], context, lossy)
}

pub fn agglomerate_blockwise_pyramid<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    mut edges: Vec<AgglomEdge>,
    mut seeds: Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
    block_shapes: Vec<[usize; D]>,
    context: Option<[usize; D]>,
    lossy: bool,
) -> Array<usize, IxDyn> {
    // Phase 0: relabel seeds (identical to sequential)
    let mut lookup = HashMap::new();
    seeds.iter_mut().enumerate().for_each(|(ind, x)| match x {
        0 => *x = ind,
        k => {
            let id = *lookup.get(k).unwrap_or(&ind);
            lookup.insert(*k, id);
            *k = id;
        }
    });

    // update user-provided edges from seed ids to node ids
    edges.iter_mut().for_each(|AgglomEdge(_pos, u, v)| {
        *u = lookup[u];
        *v = lookup[v];
    });

    // seed-pair mutex edges go to deferred queue
    let unique_seeds: Vec<&usize> = lookup.values().collect();
    let mut seed_mutex_edges = Vec::new();
    (0..unique_seeds.len()).for_each(|id1| {
        ((id1 + 1)..unique_seeds.len()).for_each(|id2| {
            seed_mutex_edges.push(AgglomEdge(false, *unique_seeds[id1], *unique_seeds[id2]));
        })
    });
    // user-provided edges also go to deferred (they reference arbitrary node IDs)
    seed_mutex_edges.extend(edges);

    // Phases 1-5: extract edges into blocks, process in parallel, merge
    let (clustering, mut filtered_background) = blockwise::process_blockwise::<D>(
        affinities,
        offsets,
        seed_mutex_edges,
        &seeds,
        strides,
        randomized_strides,
        &block_shapes,
        seeds.len(),
        context,
        lossy,
    );
    lookup.values().for_each(|node_id| {
        filtered_background.remove(node_id);
    });

    // Phase 6: relabel output (identical to sequential)
    clustering.filter_map(&mut seeds, filtered_background);

    let mut rev_lookup = HashMap::with_capacity(lookup.len());
    rev_lookup.insert(seeds.len(), 0);
    lookup.iter().for_each(|(seed, id)| {
        let rep_id = clustering.positives.clusters.find(*id);
        if *seed != rep_id {
            rev_lookup.insert(rep_id, *seed);
        }
    });

    seeds.iter_mut().for_each(|x| {
        *x = *rev_lookup.get(x).unwrap_or(x);
    });

    seeds
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
#[pyfunction]
fn agglom_rs<'py>(
    py: Python<'py>,
    affinities: &Bound<'py, PyArrayDyn<f64>>,
    offsets: Vec<Vec<isize>>,
    seeds: Option<&Bound<'py, PyArrayDyn<usize>>>,
    edges: Option<Vec<(bool, usize, usize)>>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: Option<bool>,
) -> Bound<'py, PyArrayDyn<usize>> {
    let affinities = unsafe { affinities.as_array() }.to_owned();
    let seeds = match seeds {
        Some(seeds) => unsafe { seeds.as_array() }.to_owned(),
        None => Array::zeros(&affinities.shape()[1..]),
    };
    let dim = seeds.dim().ndim();
    let edges: Vec<AgglomEdge> = edges
        .unwrap_or_default()
        .into_iter()
        .map(|(pos, u, v)| AgglomEdge(pos, u, v))
        .collect();
    let result = match dim {
        1 => agglomerate::<1>(
            &affinities,
            offsets,
            edges,
            seeds,
            strides,
            randomized_strides.unwrap_or(false),
        ),
        2 => agglomerate::<2>(
            &affinities,
            offsets,
            edges,
            seeds,
            strides,
            randomized_strides.unwrap_or(false),
        ),
        3 => agglomerate::<3>(
            &affinities,
            offsets,
            edges,
            seeds,
            strides,
            randomized_strides.unwrap_or(false),
        ),
        4 => agglomerate::<4>(
            &affinities,
            offsets,
            edges,
            seeds,
            strides,
            randomized_strides.unwrap_or(false),
        ),
        5 => agglomerate::<5>(
            &affinities,
            offsets,
            edges,
            seeds,
            strides,
            randomized_strides.unwrap_or(false),
        ),
        6 => agglomerate::<6>(
            &affinities,
            offsets,
            edges,
            seeds,
            strides,
            randomized_strides.unwrap_or(false),
        ),
        _ => panic!["Only 1-6 dimensional arrays supported"],
    };
    result.into_pyarray(py)
}

/// Cluster edges into a set of connected components using mutex watershed.
/// It is assumed that the edges have been sorted in descending order of affinity
/// magnitude and thresholded at zero to get a list of [pos:bool, u:int, v:int] tuples.
///
/// :param edges: A list of edges, each represented as a tuple (pos: bool, u: int, v: int).
/// :return: A list of tuples representing the connected components, where each tuple
///         contains two integers (fragment: int, segment: int).
#[pyfunction]
fn cluster(edges: Vec<(bool, usize, usize)>) -> PyResult<Vec<(usize, usize)>> {
    let edges: Vec<AgglomEdge> = edges
        .into_iter()
        .map(|(pos, u, v)| AgglomEdge(pos, u, v))
        .collect();
    let result = cluster_edges(edges);
    Ok(result)
}

/// Blockwise parallel agglomeration using mutex watershed.
#[pyfunction]
fn agglom_blockwise_rs<'py>(
    py: Python<'py>,
    affinities: &Bound<'py, PyArrayDyn<f64>>,
    offsets: Vec<Vec<isize>>,
    block_shape: Vec<usize>,
    seeds: Option<&Bound<'py, PyArrayDyn<usize>>>,
    edges: Option<Vec<(bool, usize, usize)>>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: Option<bool>,
    context: Option<Vec<usize>>,
    lossy: Option<bool>,
) -> Bound<'py, PyArrayDyn<usize>> {
    let affinities = unsafe { affinities.as_array() }.to_owned();
    let seeds = match seeds {
        Some(seeds) => unsafe { seeds.as_array() }.to_owned(),
        None => Array::zeros(&affinities.shape()[1..]),
    };
    let dim = seeds.dim().ndim();
    let edges: Vec<AgglomEdge> = edges
        .unwrap_or_default()
        .into_iter()
        .map(|(pos, u, v)| AgglomEdge(pos, u, v))
        .collect();
    let lossy = lossy.unwrap_or(false);
    let result = match dim {
        1 => {
            let bs: [usize; 1] = block_shape.try_into().expect("block_shape length must match dimensions");
            let ctx: Option<[usize; 1]> = context.map(|c| c.try_into().expect("context length must match dimensions"));
            agglomerate_blockwise::<1>(&affinities, offsets, edges, seeds, strides, randomized_strides.unwrap_or(false), bs, ctx, lossy)
        }
        2 => {
            let bs: [usize; 2] = block_shape.try_into().expect("block_shape length must match dimensions");
            let ctx: Option<[usize; 2]> = context.map(|c| c.try_into().expect("context length must match dimensions"));
            agglomerate_blockwise::<2>(&affinities, offsets, edges, seeds, strides, randomized_strides.unwrap_or(false), bs, ctx, lossy)
        }
        3 => {
            let bs: [usize; 3] = block_shape.try_into().expect("block_shape length must match dimensions");
            let ctx: Option<[usize; 3]> = context.map(|c| c.try_into().expect("context length must match dimensions"));
            agglomerate_blockwise::<3>(&affinities, offsets, edges, seeds, strides, randomized_strides.unwrap_or(false), bs, ctx, lossy)
        }
        4 => {
            let bs: [usize; 4] = block_shape.try_into().expect("block_shape length must match dimensions");
            let ctx: Option<[usize; 4]> = context.map(|c| c.try_into().expect("context length must match dimensions"));
            agglomerate_blockwise::<4>(&affinities, offsets, edges, seeds, strides, randomized_strides.unwrap_or(false), bs, ctx, lossy)
        }
        5 => {
            let bs: [usize; 5] = block_shape.try_into().expect("block_shape length must match dimensions");
            let ctx: Option<[usize; 5]> = context.map(|c| c.try_into().expect("context length must match dimensions"));
            agglomerate_blockwise::<5>(&affinities, offsets, edges, seeds, strides, randomized_strides.unwrap_or(false), bs, ctx, lossy)
        }
        6 => {
            let bs: [usize; 6] = block_shape.try_into().expect("block_shape length must match dimensions");
            let ctx: Option<[usize; 6]> = context.map(|c| c.try_into().expect("context length must match dimensions"));
            agglomerate_blockwise::<6>(&affinities, offsets, edges, seeds, strides, randomized_strides.unwrap_or(false), bs, ctx, lossy)
        }
        _ => panic!["Only 1-6 dimensional arrays supported"],
    };
    result.into_pyarray(py)
}

#[pymodule(name = "mwatershed")]
fn mwatershed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(agglom_rs, m)?)?;
    m.add_function(wrap_pyfunction!(agglom_blockwise_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use itertools::Itertools;
    use ndarray::array;
    extern crate test;

    /// Seeds
    /// 1 2 0
    /// 4 0 0
    /// 0 0 0
    ///
    /// Affs
    /// offset [0, 1]
    /// 0 1 0
    /// 0 1 0
    /// 0 1 0
    ///
    /// offset [1, 0]
    /// 0 0 0
    /// 1 1 1
    /// 0 0 0
    ///
    /// Expected Components
    /// 1 2 2
    /// 4 x x
    /// 4 x x
    ///
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
        let components = agglomerate::<2>(&affinities, offsets, vec![], seeds, None, false);
        let ids = components
            .clone()
            .into_iter()
            .unique()
            .collect::<Vec<usize>>();
        for id in [1, 2, 4].iter() {
            assert!(ids.contains(id), "{:?}", components);
        }
        assert!(!ids.contains(&0), "{:?}", components);
        assert!(ids.len() == 4, "{:?}", components);
    }
    /// Seeds
    /// 1 2 0
    /// 4 0 0
    /// 0 0 0
    ///
    /// Affs
    /// offset [0, 1]
    /// 0 1 0
    /// 0 1 0
    /// 0 1 0
    ///
    /// offset [1, 0]
    /// 0 0 0
    /// 1 1 1
    /// 0 0 0
    ///
    /// Expected Components
    /// 1 2 2
    /// 4 0 x
    /// 4 x x
    ///
    #[test]
    fn test_agglom_with_strides() {
        let affinities = array![
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[1, 2, 0], [4, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, 1], vec![1, 0]];
        let strides = vec![vec![2, 1], vec![1, 2]];
        let components =
            agglomerate::<2>(&affinities, offsets, vec![], seeds, Some(strides), false);
        let ids = components
            .clone()
            .into_iter()
            .unique()
            .collect::<Vec<usize>>();
        for id in [1, 2, 4].iter() {
            assert!(ids.contains(id), "{:?}", components);
        }
        assert!(ids.contains(&0), "{:?}", components);
        assert!(ids.len() == 5, "{:?}", components);
    }

    /// Seeds
    /// 1 2 0
    /// 4 0 0
    /// 0 0 0
    ///
    /// Affs
    /// offset [0, -1]
    /// 0 0 1
    /// 0 0 1
    /// 0 0 1
    ///
    /// offset [-1, 0]
    /// 0 0 0
    /// 0 0 0
    /// 1 1 1
    ///
    /// Expected Components
    /// 1 2 2
    /// 4 x x
    /// 4 x x
    ///
    #[test]
    fn test_agglom_negative_offsets() {
        let affinities = array![
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[1, 2, 0], [4, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, -1], vec![-1, 0]];
        let components = agglomerate::<2>(&affinities, offsets, vec![], seeds, None, false);
        let ids = components
            .clone()
            .into_iter()
            .unique()
            .collect::<Vec<usize>>();
        for id in [1, 2, 4].iter() {
            assert!(ids.contains(id), "{:?}", components);
        }
        assert!(!ids.contains(&0), "{:?}", components);
        assert!(ids.len() == 4, "{:?}", components);
    }

    /// Seeds
    /// 1 2 0
    /// 4 0 0
    /// 0 0 0
    ///
    /// Affs
    /// offset [0, -1]
    /// 0 0 0
    /// 0 0 0
    /// 0 0 0
    ///
    /// offset [-1, 0]
    /// 0 0 0
    /// 0 0 0
    /// 0 0 0
    ///
    /// Expected Components
    /// 1 2 0
    /// 4 0 0
    /// 0 0 0
    ///
    #[test]
    fn test_filtered_background() {
        let affinities = array![
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[1, 2, 0], [4, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, -1], vec![-1, 0]];
        let components = agglomerate::<2>(&affinities, offsets, vec![], seeds, None, false);
        let ids = components
            .clone()
            .into_iter()
            .unique()
            .collect::<Vec<usize>>();
        for id in [1, 2, 4].iter() {
            assert!(ids.contains(id), "{:?}", components);
        }
        assert!(ids.contains(&0), "{:?}", components);
        assert!(ids.len() == 4, "{:?}", components);
        assert!(
            components.iter().counts().get(&0).unwrap() == &6,
            "{:?}",
            components
        );
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
    }

    /// Check that two label arrays have the same grouping (structurally equivalent).
    fn assert_structurally_equivalent(
        a: &Array<usize, IxDyn>,
        b: &Array<usize, IxDyn>,
        msg: &str,
    ) {
        use std::collections::HashMap;
        assert_eq!(a.shape(), b.shape(), "{}: shape mismatch", msg);
        let mut map: HashMap<usize, usize> = HashMap::new();
        let mut rev: HashMap<usize, usize> = HashMap::new();
        for (va, vb) in a.iter().zip(b.iter()) {
            match map.get(va) {
                Some(&expected) => assert_eq!(
                    expected, *vb,
                    "{}: label {} maps to {} but found {}",
                    msg, va, expected, vb
                ),
                None => {
                    if let Some(&existing_a) = rev.get(vb) {
                        panic!(
                            "{}: blockwise label {} already mapped from seq label {}, now also from {}",
                            msg, vb, existing_a, va
                        );
                    }
                    map.insert(*va, *vb);
                    rev.insert(*vb, *va);
                }
            }
        }
        // Same number of unique labels
        assert_eq!(
            map.len(),
            rev.len(),
            "{}: different number of unique labels",
            msg
        );
    }

    /// Test that blockwise produces structurally equivalent output to sequential.
    #[test]
    fn test_blockwise_matches_sequential() {
        let affinities = array![
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[1, 2, 0], [4, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, 1], vec![1, 0]];

        let sequential = agglomerate::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
        );

        // Block size = full array (degenerate case, should match exactly)
        let blockwise_full = agglomerate_blockwise::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
            [3, 3],
            None,
            false,
        );
        assert_structurally_equivalent(&sequential, &blockwise_full, "full block size");

        // Block size = 2x2
        let blockwise_2x2 = agglomerate_blockwise::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
            [2, 2],
            None,
            false,
        );
        assert_structurally_equivalent(&sequential, &blockwise_2x2, "2x2 block size");

        // Block size = 1x1 (maximally deferred)
        let blockwise_1x1 = agglomerate_blockwise::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
            [1, 1],
            None,
            false,
        );
        assert_structurally_equivalent(&sequential, &blockwise_1x1, "1x1 block size");
    }

    /// Test blockwise with no seeds (unsupervised clustering).
    #[test]
    fn test_blockwise_no_seeds() {
        let affinities = array![
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[0, 0, 0], [0, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, 1], vec![1, 0]];

        let sequential = agglomerate::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
        );

        let blockwise = agglomerate_blockwise::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
            [2, 2],
            None,
            false,
        );
        assert_structurally_equivalent(&sequential, &blockwise, "unsupervised blockwise");
    }

    /// Test blockwise with negative offsets.
    #[test]
    fn test_blockwise_negative_offsets() {
        let affinities = array![
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[1, 2, 0], [4, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, -1], vec![-1, 0]];

        let sequential = agglomerate::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
        );

        let blockwise = agglomerate_blockwise::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
            [2, 2],
            None,
            false,
        );
        assert_structurally_equivalent(&sequential, &blockwise, "negative offsets blockwise");
    }

    /// Larger randomized test to verify blockwise matches sequential.
    #[test]
    fn test_blockwise_matches_sequential_large() {
        // 50x50 with deterministic pseudo-random affinities
        let size = 50usize;
        let n = size * size;

        // Simple LCG for deterministic values
        let mut state: u64 = 12345;
        let mut next_f64 = || -> f64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0
        };

        let mut aff_data = Vec::with_capacity(2 * n);
        for _ in 0..2 {
            for _ in 0..n {
                aff_data.push(next_f64());
            }
        }

        let affinities = Array::from_shape_vec(vec![2, size, size], aff_data).unwrap();
        let seeds = Array::from_elem(vec![size, size], 0usize);
        let offsets = vec![vec![0, 1], vec![1, 0]];

        let sequential = agglomerate::<2>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
        );

        for block_size in &[5, 10, 17, 25, 50] {
            let blockwise = agglomerate_blockwise::<2>(
                &affinities,
                offsets.clone(),
                vec![],
                seeds.clone(),
                None,
                false,
                [*block_size, *block_size],
                None,
                false,
            );
            assert_structurally_equivalent(
                &sequential,
                &blockwise,
                &format!("{}x{} blocks on {}x{}", block_size, block_size, size, size),
            );
        }
    }
}
