#![feature(test)]
#![feature(binary_heap_drain_sorted)]
#![feature(extend_one)]

use itertools::Itertools;
use ndarray::{Array, Axis, Slice};
use ndarray::{Dimension, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyResult;

use ndarray_rand::rand;

use ordered_float::NotNan;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::iter::FromIterator;

mod clustering;
pub mod gasp;
mod labels;

pub use clustering::{AgglomEdge, Clustering};
pub use gasp::{GaspOptions, Linkage};

/// A weighted edge `(weight, u, v)`.
pub type WeightedEdge = (f64, usize, usize);

/// Extract weighted edges from an affinity array.
///
/// Returns the edges sorted by descending `|weight|` together with the set of
/// node ids that have no attractive edge at all (background, to be filtered).
pub fn get_weighted_edges<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    seeds: &Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
) -> (Vec<WeightedEdge>, HashSet<usize>) {
    let offsets: Vec<[isize; D]> = offsets
        .into_iter()
        .map(|offset| offset.try_into().unwrap())
        .collect();

    let mut edges: Vec<(NotNan<f64>, WeightedEdge)> = Vec::with_capacity(affinities.len());

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
                    let magnitude = NotNan::new(aff.abs()).expect("Cannot handle `nan` affinities");
                    edges.push((magnitude, (*aff, u, v)));
                }
            });
        });
    let sorted_edges: Vec<WeightedEdge> = edges
        .into_iter()
        .sorted_unstable_by(|a, b| Ord::cmp(&b.0, &a.0))
        .map(|(_magnitude, edge)| edge)
        .collect();
    sorted_edges.iter().for_each(|(w, u, v)| {
        if *w > 0.0 {
            to_filter.remove(u);
            to_filter.remove(v);
        }
    });
    (
        sorted_edges
            .into_iter()
            .filter(|(_, u, v)| !(to_filter.contains(u) || to_filter.contains(v)))
            .collect(),
        to_filter,
    )
}

/// Extract edges from an affinity array thresholded at zero, sorted by
/// descending affinity magnitude (the mutex watershed processing order).
pub fn get_edges<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    seeds: &Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
) -> (Vec<AgglomEdge>, HashSet<usize>) {
    let (weighted, to_filter) =
        get_weighted_edges::<D>(affinities, offsets, seeds, strides, randomized_strides);
    (
        weighted
            .into_iter()
            .map(|(w, u, v)| AgglomEdge(w > 0.0, u, v))
            .collect(),
        to_filter,
    )
}

/// Mutex watershed agglomeration (GASP with `AbsMax` linkage).
pub fn agglomerate<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    edges: Vec<AgglomEdge>,
    seeds: Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
) -> Array<usize, IxDyn> {
    agglomerate_with::<D>(
        affinities,
        offsets,
        edges,
        seeds,
        strides,
        randomized_strides,
        GaspOptions::default(),
    )
}

/// Agglomeration with a configurable linkage criterion.
///
/// `AbsMax` linkage uses the dedicated mutex watershed implementation; every
/// other linkage runs the generic GASP engine in [`gasp`].
pub fn agglomerate_with<const D: usize>(
    affinities: &Array<f64, IxDyn>,
    offsets: Vec<Vec<isize>>,
    mut edges: Vec<AgglomEdge>,
    mut seeds: Array<usize, IxDyn>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: bool,
    options: GaspOptions,
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
        get_weighted_edges::<D>(affinities, offsets, &seeds, strides, randomized_strides);
    lookup.values().for_each(|node_id| {
        filtered_background.remove(node_id);
    });

    let num_nodes = seeds.len();
    let representatives: Vec<usize> = match options.linkage {
        Linkage::AbsMax => {
            edges.extend(
                sorted_edges
                    .into_iter()
                    .map(|(w, u, v)| AgglomEdge(w > 0.0, u, v)),
            );
            let mut clustering = Clustering::new(num_nodes);
            clustering.process_edges(edges);
            (0..num_nodes)
                .map(|i| clustering.positives.clusters.find(i))
                .collect()
        }
        _ => gasp::cluster(num_nodes, &edges, &sorted_edges, options),
    };

    relabel_output(seeds, &representatives, &filtered_background, &lookup)
}

/// Map node ids to cluster representatives, zero out filtered background and
/// restore the original seed ids.
///
/// Non-seeded clusters are labelled by their representative node id. A
/// representative id that collides with the background label (0) or with a
/// seed value is replaced by a fresh id above both the node count and the
/// largest seed value, so output labels are always unambiguous.
fn relabel_output(
    mut seeds: Array<usize, IxDyn>,
    representatives: &[usize],
    filtered_background: &HashSet<usize>,
    lookup: &HashMap<usize, usize>,
) -> Array<usize, IxDyn> {
    let background = seeds.len();

    // representative -> original seed value
    let mut rev_lookup: HashMap<usize, usize> = HashMap::with_capacity(lookup.len() + 1);
    rev_lookup.insert(background, 0);
    lookup.iter().for_each(|(seed, id)| {
        rev_lookup.insert(representatives[*id], *seed);
    });

    let reserved: HashSet<usize> = lookup.keys().copied().chain(std::iter::once(0)).collect();
    let mut next_fresh = lookup.keys().copied().max().unwrap_or(0).max(background) + 1;

    seeds.iter_mut().for_each(|x| {
        let rep = if filtered_background.contains(x) {
            background
        } else {
            representatives[*x]
        };
        *x = match rev_lookup.get(&rep) {
            Some(&label) => label,
            None if reserved.contains(&rep) => {
                let fresh = next_fresh;
                next_fresh += 1;
                rev_lookup.insert(rep, fresh);
                fresh
            }
            None => rep,
        };
    });

    seeds
}

/// Relabel arbitrary node ids to consecutive ids starting at zero.
fn serialize_ids<T>(
    edges: &mut [T],
    endpoints: impl Fn(&mut T) -> (&mut usize, &mut usize),
) -> HashMap<usize, usize> {
    let mut lookup = HashMap::new();
    let mut next_id = 0..;
    edges.iter_mut().for_each(|edge| {
        let (u, v) = endpoints(edge);
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
    lookup
}

fn fragment_to_segment(
    lookup: HashMap<usize, usize>,
    representatives: &[usize],
) -> Vec<(usize, usize)> {
    // Each segment is labelled with the smallest original fragment id it
    // contains, which keeps the output deterministic.
    let mut rev_lookup: HashMap<usize, usize> = HashMap::with_capacity(lookup.len());
    lookup.iter().for_each(|(original, id)| {
        rev_lookup
            .entry(representatives[*id])
            .and_modify(|label| *label = (*label).min(*original))
            .or_insert(*original);
    });
    lookup
        .into_iter()
        .map(|(original, serialized)| (original, rev_lookup[&representatives[serialized]]))
        .collect()
}

/// Mutex watershed on a list of pre-sorted, thresholded edges.
pub fn cluster_edges(mut sorted_edges: Vec<AgglomEdge>) -> Vec<(usize, usize)> {
    let lookup = serialize_ids(&mut sorted_edges, |AgglomEdge(_, u, v)| (u, v));

    let mut clustering = Clustering::new(lookup.len());
    clustering.process_edges(sorted_edges);
    let representatives: Vec<usize> = (0..lookup.len())
        .map(|i| clustering.positives.clusters.find(i))
        .collect();

    fragment_to_segment(lookup, &representatives)
}

/// Cluster weighted `(weight, u, v)` edges with the given linkage criterion.
/// The edges do not need to be sorted.
pub fn cluster_weighted_edges(
    mut edges: Vec<WeightedEdge>,
    options: GaspOptions,
) -> Vec<(usize, usize)> {
    gasp::sort_edges(&mut edges);
    let lookup = serialize_ids(&mut edges, |(_, u, v)| (u, v));

    let representatives = match options.linkage {
        Linkage::AbsMax => {
            let mut clustering = Clustering::new(lookup.len());
            clustering.process_edges(
                edges
                    .into_iter()
                    .map(|(w, u, v)| AgglomEdge(w > 0.0, u, v))
                    .collect(),
            );
            (0..lookup.len())
                .map(|i| clustering.positives.clusters.find(i))
                .collect()
        }
        _ => gasp::cluster(lookup.len(), &[], &edges, options),
    };

    fragment_to_segment(lookup, &representatives)
}

fn parse_options(linkage: Option<&str>, cannot_link: Option<bool>) -> PyResult<GaspOptions> {
    let linkage = match linkage {
        None => Linkage::AbsMax,
        Some(name) => Linkage::parse(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown linkage {name:?}; expected one of abs_max, sum, mean, max, min"
            ))
        })?,
    };
    Ok(GaspOptions::new(linkage, cannot_link.unwrap_or(true)))
}

/// agglomerate nodes given an array of affinities and optional additional edges
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (affinities, offsets, seeds=None, edges=None, strides=None, randomized_strides=None, linkage=None, cannot_link=None))]
fn agglom_rs<'py>(
    py: Python<'py>,
    affinities: &Bound<'py, PyArrayDyn<f64>>,
    offsets: Vec<Vec<isize>>,
    seeds: Option<&Bound<'py, PyArrayDyn<usize>>>,
    edges: Option<Vec<(bool, usize, usize)>>,
    strides: Option<Vec<Vec<usize>>>,
    randomized_strides: Option<bool>,
    linkage: Option<&str>,
    cannot_link: Option<bool>,
) -> PyResult<Bound<'py, PyArrayDyn<usize>>> {
    let options = parse_options(linkage, cannot_link)?;
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
    let randomized_strides = randomized_strides.unwrap_or(false);
    macro_rules! run {
        ($d:literal) => {
            agglomerate_with::<$d>(
                &affinities,
                offsets,
                edges,
                seeds,
                strides,
                randomized_strides,
                options,
            )
        };
    }
    let result = match dim {
        1 => run!(1),
        2 => run!(2),
        3 => run!(3),
        4 => run!(4),
        5 => run!(5),
        6 => run!(6),
        _ => {
            return Err(PyValueError::new_err(
                "Only 1-6 dimensional arrays supported",
            ))
        }
    };
    Ok(result.into_pyarray(py))
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

/// Cluster weighted edges with a configurable linkage criterion.
///
/// :param edges: A list of (weight: float, u: int, v: int) tuples, in any order.
/// :param linkage: One of "abs_max" (default), "sum", "mean", "max", "min".
/// :param cannot_link: Whether to add cannot-link constraints (default True).
/// :return: A list of (fragment: int, segment: int) tuples.
#[pyfunction]
#[pyo3(signature = (edges, linkage=None, cannot_link=None))]
fn cluster_weighted(
    edges: Vec<(f64, usize, usize)>,
    linkage: Option<&str>,
    cannot_link: Option<bool>,
) -> PyResult<Vec<(usize, usize)>> {
    let options = parse_options(linkage, cannot_link)?;
    Ok(cluster_weighted_edges(edges, options))
}

#[pymodule(name = "mwatershed")]
fn mwatershed(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(agglom_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_weighted, m)?)?;

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

    // ── Linkage / GASP tests ───────────────────────────────────────

    /// Assert two label arrays describe the same partition.
    fn assert_same_partition(a: &[usize], b: &[usize], msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        let mut fwd: HashMap<usize, usize> = HashMap::new();
        let mut rev: HashMap<usize, usize> = HashMap::new();
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            match fwd.get(&x) {
                Some(&expected) => assert_eq!(expected, y, "{msg}: node {i}"),
                None => {
                    fwd.insert(x, y);
                }
            }
            match rev.get(&y) {
                Some(&expected) => assert_eq!(expected, x, "{msg}: node {i}"),
                None => {
                    rev.insert(y, x);
                }
            }
        }
    }

    fn random_affinities(shape: &[usize], seed: u64) -> Array<f64, IxDyn> {
        use ndarray_rand::rand::SeedableRng;
        use ndarray_rand::rand_distr::Normal;
        use ndarray_rand::RandomExt;
        let mut rng = ndarray_rand::rand::rngs::StdRng::seed_from_u64(seed);
        Array::random_using(IxDyn(shape), Normal::new(0.0, 1.0).unwrap(), &mut rng)
    }

    /// The generic GASP engine with AbsMax linkage must reproduce the
    /// dedicated mutex watershed implementation exactly, including with
    /// seeds, hard edges, long range offsets and strides.
    #[test]
    fn test_absmax_engine_matches_mutex_watershed() {
        let offsets = vec![vec![0, 1], vec![1, 0], vec![0, 3], vec![3, 0], vec![-2, 2]];
        for seed in 0..5u64 {
            let affinities = random_affinities(&[5, 24, 24], seed);
            let mut seeds: Array<usize, IxDyn> = Array::zeros(IxDyn(&[24, 24]));
            seeds[[0, 0]] = 100;
            seeds[[0, 1]] = 100;
            seeds[[10, 10]] = 7;
            seeds[[23, 23]] = 9;
            let strides = Some(vec![
                vec![1, 1],
                vec![1, 1],
                vec![2, 2],
                vec![2, 2],
                vec![1, 1],
            ]);
            let hard = vec![AgglomEdge(true, 7, 9)];

            let mws = agglomerate_with::<2>(
                &affinities,
                offsets.clone(),
                hard.clone(),
                seeds.clone(),
                strides.clone(),
                false,
                GaspOptions::new(Linkage::AbsMax, true),
            );

            // Re-run through the generic engine by replicating the pipeline.
            let mut relabeled = seeds.clone();
            let mut lookup = HashMap::new();
            relabeled
                .iter_mut()
                .enumerate()
                .for_each(|(ind, x)| match x {
                    0 => *x = ind,
                    k => {
                        let id = *lookup.get(k).unwrap_or(&ind);
                        lookup.insert(*k, id);
                        *k = id;
                    }
                });
            let mut hard_edges: Vec<AgglomEdge> = hard
                .iter()
                .map(|AgglomEdge(p, u, v)| AgglomEdge(*p, lookup[u], lookup[v]))
                .collect();
            let unique: Vec<usize> = lookup.values().copied().collect();
            for i in 0..unique.len() {
                for j in (i + 1)..unique.len() {
                    hard_edges.push(AgglomEdge(false, unique[i], unique[j]));
                }
            }
            let (weighted, mut filtered) = get_weighted_edges::<2>(
                &affinities,
                offsets.clone(),
                &relabeled,
                strides.clone(),
                false,
            );
            lookup.values().for_each(|id| {
                filtered.remove(id);
            });
            for cannot_link in [true, false] {
                let reps = gasp::cluster(
                    relabeled.len(),
                    &hard_edges,
                    &weighted,
                    GaspOptions::new(Linkage::AbsMax, cannot_link),
                );
                let engine = relabel_output(relabeled.clone(), &reps, &filtered, &lookup);
                assert_same_partition(
                    mws.as_slice().unwrap(),
                    engine.as_slice().unwrap(),
                    &format!("seed {seed} cannot_link {cannot_link}"),
                );
                // seed ids must be preserved
                assert_eq!(engine[[0, 0]], 100);
                assert_eq!(engine[[10, 10]], engine[[23, 23]]);
                assert_eq!(mws[[0, 0]], 100);
            }
        }
    }

    /// 1D chain with offsets [1] and [2]:
    ///   0-1: +0.6, 1-2: +0.6, 2-3: +0.6 (offset 1)
    ///   0-2: -0.7, 1-3: +0.5             (offset 2)
    ///
    /// AbsMax: 0-2 becomes a mutex, so {0,1} and {2,3} never merge.
    /// Min (complete): the -0.7 edge dominates, two clusters.
    /// Sum: {0,1}-{2,3} interaction is -0.7 + 0.6 + 0.5 = 0.4, merged in
    ///      phase two (or directly without constraints).
    /// Mean: 0.4 / 3 > 0, merged.
    /// Max (single): 0.6 > 0, merged.
    #[test]
    fn test_linkages_differ_on_chain() {
        let affinities = array![[0.6, 0.6, 0.6, 0.0], [-0.7, 0.5, 0.0, 0.0]].into_dyn();
        let seeds: Array<usize, IxDyn> = Array::zeros(IxDyn(&[4]));
        let offsets = vec![vec![1], vec![2]];
        let expected = [
            (Linkage::AbsMax, 2),
            (Linkage::Min, 2),
            (Linkage::Sum, 1),
            (Linkage::Mean, 1),
            (Linkage::Max, 1),
        ];
        for (linkage, clusters) in expected {
            for cannot_link in [true, false] {
                let result = agglomerate_with::<1>(
                    &affinities,
                    offsets.clone(),
                    vec![],
                    seeds.clone(),
                    None,
                    false,
                    GaspOptions::new(linkage, cannot_link),
                );
                let n = result.iter().unique().count();
                assert_eq!(
                    n, clusters,
                    "{linkage} cannot_link={cannot_link}: {result:?}"
                );
                if clusters == 2 {
                    assert_eq!(result[0], result[1], "{linkage}");
                    assert_eq!(result[2], result[3], "{linkage}");
                }
            }
        }
        // Phase one only ("GreedyFixation") keeps the 0-2 constraint for Sum.
        let result = agglomerate_with::<1>(
            &affinities,
            offsets.clone(),
            vec![],
            seeds.clone(),
            None,
            false,
            GaspOptions {
                linkage: Linkage::Sum,
                cannot_link: true,
                merge_constrained: false,
            },
        );
        assert_eq!(result.iter().unique().count(), 2, "{result:?}");
    }

    /// Seeds are hard constraints for every linkage criterion.
    #[test]
    fn test_seeds_with_all_linkages() {
        let affinities = array![
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ]
        .into_dyn()
            - 0.5;
        let seeds = array![[1, 2, 0], [4, 0, 0], [0, 0, 0]].into_dyn();
        let offsets = vec![vec![0, 1], vec![1, 0]];
        for linkage in Linkage::ALL {
            let components = agglomerate_with::<2>(
                &affinities,
                offsets.clone(),
                vec![],
                seeds.clone(),
                None,
                false,
                GaspOptions::new(linkage, true),
            );
            let ids = components.iter().copied().unique().collect::<Vec<usize>>();
            for id in [1, 2, 4].iter() {
                assert!(ids.contains(id), "{linkage}: {components:?}");
            }
            assert!(!ids.contains(&0), "{linkage}: {components:?}");
            assert_eq!(ids.len(), 4, "{linkage}: {components:?}");
            assert_eq!(components[[0, 0]], 1);
            assert_eq!(components[[0, 1]], 2);
            assert_eq!(components[[0, 2]], 2);
            assert_eq!(components[[1, 0]], 4);
            assert_eq!(components[[2, 0]], 4);
        }
    }

    /// Fully attractive affinities give one cluster, fully repulsive give
    /// all background, for every linkage.
    #[test]
    fn test_linkage_extremes() {
        let offsets = vec![vec![0, 1], vec![1, 0]];
        let seeds: Array<usize, IxDyn> = Array::zeros(IxDyn(&[6, 6]));
        let attractive: Array<f64, IxDyn> = Array::from_elem(IxDyn(&[2, 6, 6]), 0.8);
        let repulsive: Array<f64, IxDyn> = Array::from_elem(IxDyn(&[2, 6, 6]), -0.8);
        for linkage in Linkage::ALL {
            let one = agglomerate_with::<2>(
                &attractive,
                offsets.clone(),
                vec![],
                seeds.clone(),
                None,
                false,
                GaspOptions::new(linkage, true),
            );
            assert_eq!(one.iter().unique().count(), 1, "{linkage}");
            assert!(one.iter().all(|&x| x != 0), "{linkage}");

            let none = agglomerate_with::<2>(
                &repulsive,
                offsets.clone(),
                vec![],
                seeds.clone(),
                None,
                false,
                GaspOptions::new(linkage, true),
            );
            assert!(none.iter().all(|&x| x == 0), "{linkage}: {none:?}");
        }
    }

    #[test]
    fn test_cluster_weighted_edges() {
        let edges = vec![
            (0.6, 10, 11),
            (0.6, 11, 12),
            (0.6, 12, 13),
            (-0.7, 10, 12),
            (0.5, 11, 13),
        ];
        let segments = |linkage| {
            let mapping = cluster_weighted_edges(edges.clone(), GaspOptions::new(linkage, true));
            let lookup: HashMap<usize, usize> = mapping.into_iter().collect();
            assert_eq!(lookup.len(), 4);
            lookup
        };
        let absmax = segments(Linkage::AbsMax);
        assert_eq!(absmax[&10], absmax[&11]);
        assert_eq!(absmax[&12], absmax[&13]);
        assert_ne!(absmax[&10], absmax[&12]);
        // segment ids are drawn from the original fragment ids
        assert!(absmax.values().all(|v| (10..=13).contains(v)));
        let sum = segments(Linkage::Sum);
        assert!(sum.values().all_equal());

        // AbsMax through the weighted path agrees with the boolean path.
        let mut sorted = edges.clone();
        gasp::sort_edges(&mut sorted);
        let boolean: HashMap<usize, usize> = cluster_edges(
            sorted
                .into_iter()
                .map(|(w, u, v)| AgglomEdge(w > 0.0, u, v))
                .collect(),
        )
        .into_iter()
        .collect();
        assert_eq!(boolean, absmax);
    }

    /// Random 3D volumes: every linkage runs, is deterministic, keeps seeds,
    /// and never merges seeded nodes with each other.
    #[test]
    fn test_random_3d_all_linkages() {
        let offsets = vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]];
        let affinities = random_affinities(&[3, 8, 8, 8], 3);
        let mut seeds: Array<usize, IxDyn> = Array::zeros(IxDyn(&[8, 8, 8]));
        seeds[[0, 0, 0]] = 11;
        seeds[[7, 7, 7]] = 12;
        seeds[[3, 4, 5]] = 13;
        for linkage in Linkage::ALL {
            for cannot_link in [true, false] {
                let options = GaspOptions::new(linkage, cannot_link);
                let a = agglomerate_with::<3>(
                    &affinities,
                    offsets.clone(),
                    vec![],
                    seeds.clone(),
                    None,
                    false,
                    options,
                );
                let b = agglomerate_with::<3>(
                    &affinities,
                    offsets.clone(),
                    vec![],
                    seeds.clone(),
                    None,
                    false,
                    options,
                );
                assert_eq!(
                    a, b,
                    "{linkage} cannot_link={cannot_link} not deterministic"
                );
                assert_eq!(a[[0, 0, 0]], 11, "{linkage}");
                assert_eq!(a[[7, 7, 7]], 12, "{linkage}");
                assert_eq!(a[[3, 4, 5]], 13, "{linkage}");
            }
        }
    }
}
