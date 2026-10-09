//! Generalized Agglomerative Signed graph Partitioning (GASP).
//!
//! Implements Algorithm 1 of Bailoni et al., "GASP, a generalized framework
//! for agglomerative clustering of signed graphs and its application to
//! instance segmentation" (CVPR 2022). The interaction between two clusters is
//! computed from the edges connecting them using a *linkage criterion*:
//!
//! | Linkage  | Interaction `W(S, S')`                        | Known algorithm            |
//! |----------|-----------------------------------------------|----------------------------|
//! | AbsMax   | weight of the edge with largest magnitude     | Mutex watershed            |
//! | Sum      | sum of edge weights                           | GAEC / HCC-Sum             |
//! | Mean     | mean of edge weights                          | HC-Avg / HCC-Avg           |
//! | Max      | maximum edge weight                           | single linkage             |
//! | Min      | minimum edge weight                           | complete linkage           |
//!
//! Phase one pops cluster pairs by decreasing `|W|`. Attractive pairs
//! (`W > 0`) are merged unless constrained; repulsive pairs (`W <= 0`) are
//! constrained if `cannot_link` is enabled. Phase two drops the soft
//! constraints and keeps merging the most attractive pair until none remains.
//!
//! With `AbsMax` linkage this reproduces the mutex watershed exactly (edges are
//! processed in the same order and tie-broken identically), which is verified
//! by the tests in `lib.rs`.

use ordered_float::NotNan;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use crate::clustering::AgglomEdge;

/// Linkage criterion used to compute the interaction between two clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linkage {
    /// Edge with the largest absolute weight (mutex watershed).
    AbsMax,
    /// Sum of all connecting edge weights (GAEC).
    Sum,
    /// Mean of all connecting edge weights (average linkage).
    Mean,
    /// Maximum connecting edge weight (single linkage).
    Max,
    /// Minimum connecting edge weight (complete linkage).
    Min,
}

impl Linkage {
    /// All linkage criteria, in a stable order (useful for benchmarks).
    pub const ALL: [Linkage; 5] = [
        Linkage::AbsMax,
        Linkage::Sum,
        Linkage::Mean,
        Linkage::Max,
        Linkage::Min,
    ];

    /// Parse a linkage name. Accepts the canonical names (`abs_max`, `sum`,
    /// `mean`, `max`, `min`) plus common aliases (`absmax`, `mutex`, `average`,
    /// `avg`, `single`, `complete`). Case insensitive.
    pub fn parse(name: &str) -> Option<Linkage> {
        match name.to_ascii_lowercase().replace('-', "_").as_str() {
            "abs_max" | "absmax" | "mutex" | "mws" => Some(Linkage::AbsMax),
            "sum" => Some(Linkage::Sum),
            "mean" | "average" | "avg" => Some(Linkage::Mean),
            "max" | "single" => Some(Linkage::Max),
            "min" | "complete" => Some(Linkage::Min),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Linkage::AbsMax => "abs_max",
            Linkage::Sum => "sum",
            Linkage::Mean => "mean",
            Linkage::Max => "max",
            Linkage::Min => "min",
        }
    }
}

impl std::fmt::Display for Linkage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Options controlling the agglomeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaspOptions {
    /// Linkage criterion. Defaults to `AbsMax` (mutex watershed).
    pub linkage: Linkage,
    /// Add cannot-link constraints when a repulsive interaction is popped in
    /// phase one. Defaults to `true`. Has no effect on the result for `AbsMax`.
    pub cannot_link: bool,
    /// Run phase two: drop the soft cannot-link constraints and merge all
    /// remaining attractive interactions. Defaults to `true`. Setting it to
    /// `false` yields the "GreedyFixation" style phase-one-only variant.
    pub merge_constrained: bool,
}

impl Default for GaspOptions {
    fn default() -> Self {
        GaspOptions {
            linkage: Linkage::AbsMax,
            cannot_link: true,
            merge_constrained: true,
        }
    }
}

impl GaspOptions {
    pub fn new(linkage: Linkage, cannot_link: bool) -> Self {
        GaspOptions {
            linkage,
            cannot_link,
            merge_constrained: true,
        }
    }
}

// ── Linkage statistics ─────────────────────────────────────────────

/// Per cluster-pair statistics from which the interaction is derived.
///
/// `combine` is given the key (position in the globally sorted edge order,
/// smaller = larger magnitude) of both operands so that tie-breaking is
/// deterministic.
pub trait LinkageStats: Copy {
    fn new(w: f64) -> Self;
    fn combine(a: Self, key_a: u32, b: Self, key_b: u32) -> Self;
    fn value(&self) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct AbsMaxStats(f64);
impl LinkageStats for AbsMaxStats {
    fn new(w: f64) -> Self {
        AbsMaxStats(w)
    }
    fn combine(a: Self, key_a: u32, b: Self, key_b: u32) -> Self {
        // Edges are keyed by descending magnitude, so the smaller key is the
        // edge with the largest |w| (ties broken by sort position).
        if key_a <= key_b {
            a
        } else {
            b
        }
    }
    fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SumStats(f64);
impl LinkageStats for SumStats {
    fn new(w: f64) -> Self {
        SumStats(w)
    }
    fn combine(a: Self, _: u32, b: Self, _: u32) -> Self {
        SumStats(a.0 + b.0)
    }
    fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeanStats {
    sum: f64,
    n: u32,
}
impl LinkageStats for MeanStats {
    fn new(w: f64) -> Self {
        MeanStats { sum: w, n: 1 }
    }
    fn combine(a: Self, _: u32, b: Self, _: u32) -> Self {
        MeanStats {
            sum: a.sum + b.sum,
            n: a.n + b.n,
        }
    }
    fn value(&self) -> f64 {
        self.sum / self.n as f64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaxStats(f64);
impl LinkageStats for MaxStats {
    fn new(w: f64) -> Self {
        MaxStats(w)
    }
    fn combine(a: Self, _: u32, b: Self, _: u32) -> Self {
        MaxStats(a.0.max(b.0))
    }
    fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MinStats(f64);
impl LinkageStats for MinStats {
    fn new(w: f64) -> Self {
        MinStats(w)
    }
    fn combine(a: Self, _: u32, b: Self, _: u32) -> Self {
        MinStats(a.0.min(b.0))
    }
    fn value(&self) -> f64 {
        self.0
    }
}

// ── Internal data structures ───────────────────────────────────────

/// Fast non-cryptographic hasher for `usize` keys (FxHash).
#[derive(Default, Clone, Copy)]
struct FxHasher(u64);

const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(FX_SEED);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FX_SEED);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(FX_SEED);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxMap<V> = HashMap<usize, V, BuildHasherDefault<FxHasher>>;

/// The interaction between two adjacent clusters.
#[derive(Clone, Copy, Debug)]
struct Link<S> {
    /// Aggregated edge statistics; `None` if the two clusters are only linked
    /// by a constraint and share no edges.
    stats: Option<S>,
    /// Smallest edge key among the connecting edges (deterministic tie-break).
    key: u32,
    /// Bumped on every update so stale heap entries can be detected.
    version: u64,
    /// Soft cannot-link constraint added during phase one.
    soft: bool,
    /// Hard constraint (seeds / user supplied edges) that is never lifted.
    hard: bool,
}

impl<S: LinkageStats> Link<S> {
    fn constrained(&self) -> bool {
        self.soft || self.hard
    }

    fn combine(a: Self, b: Self, version: u64) -> Self {
        let stats = match (a.stats, b.stats) {
            (Some(x), Some(y)) => Some(S::combine(x, a.key, y, b.key)),
            (Some(x), None) => Some(x),
            (None, y) => y,
        };
        Link {
            stats,
            key: a.key.min(b.key),
            version,
            soft: a.soft || b.soft,
            hard: a.hard || b.hard,
        }
    }
}

/// Priority queue entry. Ordered by priority (descending), then by key
/// (ascending, i.e. the edge that comes first in sorted order wins ties).
#[derive(Clone, Copy, Debug)]
struct Entry {
    prio: NotNan<f64>,
    key: u32,
    u: usize,
    v: usize,
    version: u64,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.prio
            .cmp(&other.prio)
            .then_with(|| other.key.cmp(&self.key))
            .then_with(|| other.u.cmp(&self.u))
            .then_with(|| other.v.cmp(&self.v))
    }
}

// ── The algorithm ──────────────────────────────────────────────────

/// GASP agglomeration state, generic over the linkage statistics.
pub struct Gasp<S: LinkageStats> {
    parent: Vec<usize>,
    adjacency: Vec<FxMap<Link<S>>>,
    heap: BinaryHeap<Entry>,
    version: u64,
    options: GaspOptions,
}

impl<S: LinkageStats> Gasp<S> {
    pub fn new(num_nodes: usize, options: GaspOptions) -> Self {
        Gasp {
            parent: (0..num_nodes).collect(),
            adjacency: (0..num_nodes).map(|_| FxMap::default()).collect(),
            heap: BinaryHeap::new(),
            version: 0,
            options,
        }
    }

    /// Find the representative of `x` (path halving).
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let grandparent = self.parent[self.parent[x]];
            self.parent[x] = grandparent;
            x = grandparent;
        }
        x
    }

    #[inline]
    fn is_rep(&self, x: usize) -> bool {
        self.parent[x] == x
    }

    #[inline]
    fn next_version(&mut self) -> u64 {
        self.version += 1;
        self.version
    }

    #[inline]
    fn set_link(&mut self, u: usize, v: usize, link: Link<S>) {
        self.adjacency[u].insert(v, link);
        self.adjacency[v].insert(u, link);
    }

    /// Insert or combine a link between representatives `u != v`.
    fn upsert_link(&mut self, u: usize, v: usize, link: Link<S>) -> Link<S> {
        debug_assert!(u != v);
        let merged = match self.adjacency[u].get(&v) {
            Some(existing) => Link::combine(*existing, link, link.version),
            None => link,
        };
        self.set_link(u, v, merged);
        merged
    }

    /// Apply a hard edge: `true` merges the two nodes unconditionally (unless
    /// they are hard-constrained), `false` adds a permanent cannot-link
    /// constraint. Must be called before `add_edges`.
    pub fn add_hard_edge(&mut self, positive: bool, u: usize, v: usize) {
        let ru = self.find(u);
        let rv = self.find(v);
        if ru == rv {
            return;
        }
        if positive {
            let constrained = self.adjacency[ru].get(&rv).map(|l| l.hard).unwrap_or(false);
            if !constrained {
                self.merge(ru, rv, false);
            }
        } else {
            let version = self.next_version();
            self.upsert_link(
                ru,
                rv,
                Link {
                    stats: None,
                    key: u32::MAX,
                    version,
                    soft: false,
                    hard: true,
                },
            );
        }
    }

    /// Add weighted edges. `edges` must be sorted by descending `|w|`; the
    /// position in this list is used as the tie-breaking key.
    pub fn add_edges(&mut self, edges: &[(f64, usize, usize)]) {
        assert!(
            edges.len() < u32::MAX as usize,
            "too many edges for 32 bit edge keys"
        );
        for (key, &(w, u, v)) in edges.iter().enumerate() {
            let ru = self.find(u);
            let rv = self.find(v);
            if ru == rv {
                continue;
            }
            self.upsert_link(
                ru,
                rv,
                Link {
                    stats: Some(S::new(w)),
                    key: key as u32,
                    version: 0,
                    soft: false,
                    hard: false,
                },
            );
        }
    }

    /// Heap entry for a link, or `None` if it should not be queued.
    ///
    /// Phase one queues every unconstrained interaction by magnitude; phase
    /// two queues attractive interactions by signed value, ignoring soft
    /// constraints but still honouring hard ones.
    #[inline]
    fn entry(u: usize, v: usize, link: &Link<S>, phase_two: bool) -> Option<Entry> {
        let stats = link.stats?;
        let value = stats.value();
        let prio = if phase_two {
            if link.hard || value <= 0.0 {
                return None;
            }
            value
        } else {
            if link.constrained() {
                return None;
            }
            value.abs()
        };
        Some(Entry {
            prio: NotNan::new(prio).expect("nan interaction"),
            key: link.key,
            u,
            v,
            version: link.version,
        })
    }

    fn push_phase_one(&mut self, u: usize, v: usize, link: &Link<S>) {
        if let Some(entry) = Self::entry(u, v, link, false) {
            self.heap.push(entry);
        }
    }

    fn push_phase_two(&mut self, u: usize, v: usize, link: &Link<S>) {
        if let Some(entry) = Self::entry(u, v, link, true) {
            self.heap.push(entry);
        }
    }

    /// Merge representatives `u` and `v`, updating the interactions of the
    /// merged cluster with its neighbours and pushing them onto the heap.
    fn merge(&mut self, u: usize, v: usize, phase_two: bool) {
        debug_assert!(self.is_rep(u) && self.is_rep(v) && u != v);
        // Keep the node with the larger adjacency as the representative so
        // every edge is moved O(log n) times at most.
        let (big, small) = if self.adjacency[u].len() >= self.adjacency[v].len() {
            (u, v)
        } else {
            (v, u)
        };
        self.parent[small] = big;
        let small_adjacency = std::mem::take(&mut self.adjacency[small]);
        self.adjacency[big].remove(&small);

        for (t, link) in small_adjacency {
            if t == big {
                continue;
            }
            self.adjacency[t].remove(&small);
            match self.adjacency[big].get(&t).copied() {
                Some(existing) => {
                    // Both clusters were adjacent to `t`: the interaction
                    // changes, so give it a new version and re-queue it.
                    let version = self.next_version();
                    let merged = Link::combine(existing, link, version);
                    self.set_link(big, t, merged);
                    if phase_two {
                        self.push_phase_two(big, t, &merged);
                    } else {
                        self.push_phase_one(big, t, &merged);
                    }
                }
                None => {
                    // Only `small` was adjacent to `t`: the interaction is
                    // unchanged. Keep its version so the heap entry that
                    // still names `small` resolves to it through `find`.
                    self.set_link(big, t, link);
                }
            }
        }
    }

    /// Pop the next heap entry that still refers to a live interaction.
    fn pop_live(&mut self) -> Option<(usize, usize, Link<S>)> {
        while let Some(entry) = self.heap.pop() {
            let Entry { u, v, version, .. } = entry;
            let ru = self.find(u);
            let rv = self.find(v);
            if ru == rv {
                continue;
            }
            match self.adjacency[ru].get(&rv) {
                Some(link) if link.version == version => return Some((ru, rv, *link)),
                _ => continue,
            }
        }
        None
    }

    /// Build the heap from every live interaction in one heapify pass.
    fn seed_heap(&mut self, phase_two: bool) {
        let mut entries = Vec::new();
        for (u, neighbours) in self.adjacency.iter().enumerate() {
            for (&v, link) in neighbours.iter() {
                if u < v {
                    if let Some(entry) = Self::entry(u, v, link, phase_two) {
                        entries.push(entry);
                    }
                }
            }
        }
        self.heap = BinaryHeap::from(entries);
    }

    fn phase_one(&mut self) {
        self.seed_heap(false);

        while let Some((u, v, link)) = self.pop_live() {
            if link.constrained() {
                continue;
            }
            let value = link.stats.map(|s| s.value()).unwrap_or(0.0);
            if value > 0.0 {
                self.merge(u, v, false);
            } else if self.options.cannot_link {
                let constrained = Link { soft: true, ..link };
                self.set_link(u, v, constrained);
            }
        }
    }

    fn phase_two(&mut self) {
        self.seed_heap(true);

        while let Some((u, v, link)) = self.pop_live() {
            if link.hard {
                continue;
            }
            let value = link.stats.map(|s| s.value()).unwrap_or(0.0);
            if value > 0.0 {
                self.merge(u, v, true);
            }
        }
    }

    /// Run the agglomeration.
    pub fn run(&mut self) {
        self.phase_one();
        // Phase two can only merge pairs that were soft-constrained in phase
        // one. Without constraints there is nothing left to do, and with
        // AbsMax linkage a constrained pair can never become attractive.
        if self.options.cannot_link
            && self.options.merge_constrained
            && self.options.linkage != Linkage::AbsMax
        {
            self.phase_two();
        }
        self.heap = BinaryHeap::new();
    }

    /// Representative of every node.
    pub fn representatives(&mut self) -> Vec<usize> {
        (0..self.parent.len()).map(|i| self.find(i)).collect()
    }

    pub fn num_nodes(&self) -> usize {
        self.parent.len()
    }
}

fn run_with<S: LinkageStats>(
    num_nodes: usize,
    hard_edges: &[AgglomEdge],
    edges: &[(f64, usize, usize)],
    options: GaspOptions,
) -> Vec<usize> {
    let mut gasp = Gasp::<S>::new(num_nodes, options);
    for AgglomEdge(positive, u, v) in hard_edges {
        gasp.add_hard_edge(*positive, *u, *v);
    }
    gasp.add_edges(edges);
    gasp.run();
    gasp.representatives()
}

/// Cluster `num_nodes` nodes.
///
/// * `hard_edges` are applied first, in order: positive edges merge nodes,
///   negative edges add permanent cannot-link constraints.
/// * `edges` are `(weight, u, v)` triples sorted by descending `|weight|`.
///
/// Returns the representative node id of every node.
pub fn cluster(
    num_nodes: usize,
    hard_edges: &[AgglomEdge],
    edges: &[(f64, usize, usize)],
    options: GaspOptions,
) -> Vec<usize> {
    match options.linkage {
        Linkage::AbsMax => run_with::<AbsMaxStats>(num_nodes, hard_edges, edges, options),
        Linkage::Sum => run_with::<SumStats>(num_nodes, hard_edges, edges, options),
        Linkage::Mean => run_with::<MeanStats>(num_nodes, hard_edges, edges, options),
        Linkage::Max => run_with::<MaxStats>(num_nodes, hard_edges, edges, options),
        Linkage::Min => run_with::<MinStats>(num_nodes, hard_edges, edges, options),
    }
}

/// Sort `(weight, u, v)` edges by descending magnitude, the order expected by
/// [`cluster`] and [`Gasp::add_edges`].
pub fn sort_edges(edges: &mut [(f64, usize, usize)]) {
    edges.sort_by(|a, b| {
        NotNan::new(b.0.abs())
            .expect("Cannot handle `nan` affinities")
            .cmp(&NotNan::new(a.0.abs()).expect("Cannot handle `nan` affinities"))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        n: usize,
        mut edges: Vec<(f64, usize, usize)>,
        linkage: Linkage,
        cannot_link: bool,
    ) -> Vec<usize> {
        sort_edges(&mut edges);
        cluster(n, &[], &edges, GaspOptions::new(linkage, cannot_link))
    }

    fn same(reps: &[usize], a: usize, b: usize) -> bool {
        reps[a] == reps[b]
    }

    fn num_clusters(reps: &[usize]) -> usize {
        let mut r = reps.to_vec();
        r.sort_unstable();
        r.dedup();
        r.len()
    }

    #[test]
    fn parse_names() {
        assert_eq!(Linkage::parse("abs_max"), Some(Linkage::AbsMax));
        assert_eq!(Linkage::parse("AbsMax"), Some(Linkage::AbsMax));
        assert_eq!(Linkage::parse("mutex"), Some(Linkage::AbsMax));
        assert_eq!(Linkage::parse("sum"), Some(Linkage::Sum));
        assert_eq!(Linkage::parse("average"), Some(Linkage::Mean));
        assert_eq!(Linkage::parse("mean"), Some(Linkage::Mean));
        assert_eq!(Linkage::parse("single"), Some(Linkage::Max));
        assert_eq!(Linkage::parse("complete"), Some(Linkage::Min));
        assert_eq!(Linkage::parse("median"), None);
    }

    /// Triangle: a-b strongly repulsive, a-c and b-c attractive.
    ///
    /// Sum linkage merges a-c first, then (ac)-b has interaction 2 - 3 = -1,
    /// so b stays separate. Max (single) linkage ignores the repulsion once
    /// c bridges the two, so everything merges.
    #[test]
    fn sum_vs_single_linkage() {
        let edges = vec![(-3.0, 0, 1), (2.0, 0, 2), (1.5, 1, 2)];
        let sum = run(3, edges.clone(), Linkage::Sum, true);
        assert!(same(&sum, 0, 2));
        assert!(!same(&sum, 0, 1));

        let single = run(3, edges.clone(), Linkage::Max, true);
        assert_eq!(num_clusters(&single), 1);

        // Complete linkage: (ac)-b = min(-3, 1.5) = -3, never merges.
        let complete = run(3, edges.clone(), Linkage::Min, true);
        assert!(same(&complete, 0, 2));
        assert!(!same(&complete, 0, 1));

        // Mean: (ac)-b = (-3 + 1.5) / 2 < 0, never merges.
        let mean = run(3, edges.clone(), Linkage::Mean, true);
        assert!(same(&mean, 0, 2));
        assert!(!same(&mean, 0, 1));

        // AbsMax: a-b constrained first, then a-c merged, (ac)-b = -3.
        let absmax = run(3, edges, Linkage::AbsMax, true);
        assert!(same(&absmax, 0, 2));
        assert!(!same(&absmax, 0, 1));
    }

    /// Sum linkage where many weak attractions outweigh one repulsion.
    #[test]
    fn sum_accumulates() {
        // a-b: -1 ; a-c: 0.6 ; b-c: 0.6 ; a-d: 0.5 ; b-d: 0.5
        let edges = vec![
            (-1.0, 0, 1),
            (0.6, 0, 2),
            (0.6, 1, 2),
            (0.5, 0, 3),
            (0.5, 1, 3),
        ];
        // Without constraints (GAEC): a-c merged, then (ac)-b = -0.4, (ac)-d =
        // 0.5 merged, then (acd)-b = -1 + 0.6 + 0.5 = 0.1 > 0: merged.
        let gaec = run(4, edges.clone(), Linkage::Sum, false);
        assert_eq!(num_clusters(&gaec), 1);

        // With constraints, phase one constrains a-b first (|W| = 1 is the
        // largest), so phase one ends with {a,c,d} and {b}; phase two lifts
        // the constraint and merges because the interaction is +0.1.
        let hcc = run(4, edges.clone(), Linkage::Sum, true);
        assert_eq!(num_clusters(&hcc), 1);

        // Phase one only ("GreedyFixation") keeps the constraint.
        let mut sorted = edges.clone();
        sort_edges(&mut sorted);
        let fixation = cluster(
            4,
            &[],
            &sorted,
            GaspOptions {
                linkage: Linkage::Sum,
                cannot_link: true,
                merge_constrained: false,
            },
        );
        assert_eq!(num_clusters(&fixation), 2);
        assert!(same(&fixation, 0, 2) && same(&fixation, 0, 3));
        assert!(!same(&fixation, 0, 1));
    }

    #[test]
    fn mean_linkage_dilutes_repulsion() {
        // Two clusters connected by three edges: -2, +1, +1 -> mean 0, not merged.
        // Add a fourth +1 edge -> mean 0.25, merged.
        let base = vec![
            (5.0, 0, 1),
            (5.0, 2, 3),
            (-2.0, 0, 2),
            (1.0, 0, 3),
            (1.0, 1, 2),
        ];
        let reps = run(4, base.clone(), Linkage::Mean, true);
        assert_eq!(num_clusters(&reps), 2);

        let mut more = base.clone();
        more.push((1.0, 1, 3));
        let reps = run(4, more, Linkage::Mean, true);
        assert_eq!(num_clusters(&reps), 1);
    }

    #[test]
    fn hard_constraints_survive_phase_two() {
        let mut edges = vec![(1.0, 0, 1), (1.0, 1, 2)];
        sort_edges(&mut edges);
        let hard = vec![AgglomEdge(false, 0, 2)];
        for linkage in Linkage::ALL {
            let reps = cluster(3, &hard, &edges, GaspOptions::new(linkage, true));
            // 0 and 2 can never be in the same cluster; whichever of 0 or 2 is
            // merged with 1, the other stays separate.
            assert!(!same(&reps, 0, 2), "{}", linkage);
            assert_eq!(num_clusters(&reps), 2, "{linkage}");
        }
    }

    #[test]
    fn hard_merges_pre_cluster() {
        // Hard merge 0-1 then a repulsive edge between them is ignored and
        // their combined attraction to 2 pulls it in under sum linkage.
        let mut edges = vec![(-10.0, 0, 1), (0.4, 0, 2), (0.4, 1, 2), (-0.5, 0, 2)];
        sort_edges(&mut edges);
        let hard = vec![AgglomEdge(true, 0, 1)];
        let reps = cluster(3, &hard, &edges, GaspOptions::new(Linkage::Sum, true));
        assert_eq!(num_clusters(&reps), 1);
        // AbsMax: the strongest 0-2 edge is -0.5, so 2 stays out.
        let reps = cluster(3, &hard, &edges, GaspOptions::new(Linkage::AbsMax, true));
        assert_eq!(num_clusters(&reps), 2);
        assert!(same(&reps, 0, 1));
    }

    #[test]
    fn isolated_and_zero_weight_edges() {
        // Zero weight edges are repulsive (W <= 0 adds a constraint).
        let edges = vec![(0.0, 0, 1), (1.0, 1, 2)];
        for linkage in Linkage::ALL {
            let reps = run(4, edges.clone(), linkage, true);
            assert!(same(&reps, 1, 2), "{}", linkage);
            assert!(!same(&reps, 0, 1), "{}", linkage);
            assert_eq!(reps[3], 3, "{linkage}");
        }
    }

    #[test]
    fn deterministic() {
        use ndarray_rand::rand::{Rng, SeedableRng};
        let mut rng = ndarray_rand::rand::rngs::StdRng::seed_from_u64(7);
        let n = 200;
        let edges: Vec<(f64, usize, usize)> = (0..2000)
            .map(|_| {
                (
                    rng.random_range(-1.0..1.0),
                    rng.random_range(0..n),
                    rng.random_range(0..n),
                )
            })
            .filter(|(_, u, v)| u != v)
            .collect();
        for linkage in Linkage::ALL {
            for cannot_link in [true, false] {
                let a = run(n, edges.clone(), linkage, cannot_link);
                let b = run(n, edges.clone(), linkage, cannot_link);
                assert_eq!(a, b, "{linkage} cannot_link={cannot_link}");
            }
        }
    }
}
