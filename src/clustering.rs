use disjoint_sets::UnionFind;

use ndarray::Array;
use ndarray::IxDyn;
use std::collections::HashSet;

/// A struct to represent edges in the mutex merge graph.
#[derive(Debug)]
pub struct AgglomEdge(pub bool, pub usize, pub usize);

/// A struct to keep track of the mutually exclusive sets of nodes.
pub struct Negatives {
    mutexes: Vec<HashSet<usize>>,
}
impl Negatives {
    /// Create a new Negatives struct with `num_nodes` nodes.
    fn new(num_nodes: usize) -> Negatives {
        Negatives {
            mutexes: (0..(num_nodes)).map(|_| HashSet::new()).collect(),
        }
    }

    /// Check if two nodes are mutually exclusive.
    fn mutex(&self, a: usize, b: usize) -> bool {
        self.mutexes[a].contains(&b)
    }

    /// Merge two nodes, replacing all edges to b with edges to a
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

    /// Insert a mutex edge between two nodes.
    fn insert(&mut self, a: usize, b: usize) {
        self.mutexes[a].insert(b);
        self.mutexes[b].insert(a);
    }
}

/// A simple wrapper around a UnionFind struct to keep track of the positively
/// merged nodes.
pub struct Positives {
    pub clusters: UnionFind<usize>,
}
impl Positives {
    /// create a new Positives struct with `num_nodes` nodes.
    pub fn new(num_nodes: usize) -> Positives {
        Positives {
            clusters: UnionFind::new(num_nodes),
        }
    }

    /// merge two nodes
    pub fn merge(&mut self, a: usize, b: usize) {
        self.clusters.union(a, b);
    }
}

/// A struct to keep track of the clustering with both positive and negative
/// edges.
pub struct Clustering {
    pub positives: Positives,
    pub negatives: Negatives,
}

impl Clustering {
    /// Create a new Clustering struct with `num_nodes` nodes.
    pub fn new(num_nodes: usize) -> Clustering {
        Clustering {
            positives: Positives::new(num_nodes),
            negatives: Negatives::new(num_nodes),
        }
    }

    /// Merge two clusters, replacing all edges to cluster a and cluster b with edges
    /// to the representative node of the merged clusters.
    pub fn merge(&mut self, a: usize, b: usize) {
        self.positives.merge(a, b);
        let rep = self.positives.clusters.find(a);
        match a == rep {
            true => self.negatives.merge(a, b),
            false => self.negatives.merge(b, a),
        }
    }

    /// Split two clusters
    pub fn split(&mut self, a: usize, b: usize) {
        self.negatives.insert(a, b);
    }

    /// Process a vector of edges, merging and splitting clusters as necessary.
    pub fn process_edges(&mut self, sorted_edges: Vec<AgglomEdge>) {
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
                        false => self.split(new_id, old_id),
                    }
                }
            }
        });
    }

    /// Map a vector of node ids to their representative node ids.
    pub fn map(&self, seeds: &mut Array<usize, IxDyn>) {
        seeds.iter_mut().for_each(|x| {
            *x = self.positives.clusters.find(*x);
        });
    }

    /// Map a vector of node ids to their representative node ids, replacing
    /// nodes that have been filtered out with 0.
    pub fn filter_map(&self, seeds: &mut Array<usize, IxDyn>, filtered_nodes: HashSet<usize>) {
        seeds
            .iter_mut()
            .for_each(|x| match filtered_nodes.contains(x) {
                false => *x = self.positives.clusters.find(*x),
                true => *x = self.positives.clusters.len(),
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clustering_merge() {
        let mut clustering = Clustering::new(5);
        clustering.split(0, 1);
        clustering.merge(1, 2);
        clustering.merge(3, 4);
        clustering.merge(2, 3);

        assert_ne!(
            clustering.positives.clusters.find(0),
            clustering.positives.clusters.find(1)
        );
        assert_eq!(
            clustering.positives.clusters.find(1),
            clustering.positives.clusters.find(2)
        );
        assert_eq!(
            clustering.positives.clusters.find(2),
            clustering.positives.clusters.find(3)
        );
        assert_eq!(
            clustering.positives.clusters.find(3),
            clustering.positives.clusters.find(4)
        );
    }

    #[test]
    fn test_clustering_process_edges() {
        let mut clustering = Clustering::new(5);
        let edges = vec![
            AgglomEdge(true, 1, 2),
            AgglomEdge(true, 3, 4),
            AgglomEdge(false, 2, 3),
            AgglomEdge(true, 0, 3),
            AgglomEdge(true, 0, 2),
        ];
        clustering.process_edges(edges);

        assert_eq!(
            clustering.positives.clusters.find(1),
            clustering.positives.clusters.find(2)
        );
        assert_ne!(
            clustering.positives.clusters.find(2),
            clustering.positives.clusters.find(3)
        );
        assert_eq!(
            clustering.positives.clusters.find(3),
            clustering.positives.clusters.find(4)
        );
        assert_eq!(
            clustering.positives.clusters.find(0),
            clustering.positives.clusters.find(4),
        );
    }

    #[test]
    fn test_clustering_map() {
        let mut clustering = Clustering::new(5);
        clustering.merge(1, 2);
        let mut seeds = Array::from(vec![0, 0, 1, 1, 2, 2, 4, 4, 3, 3]).into_dyn();
        clustering.map(&mut seeds);

        assert_eq!(
            seeds,
            Array::from(vec![0, 0, 2, 2, 2, 2, 4, 4, 3, 3]).into_dyn()
        );
    }

    #[test]
    fn test_positives_new() {
        let positives = Positives::new(5);
        assert_eq!(positives.clusters.len(), 5);
    }

    #[test]
    fn test_positives_merge() {
        let mut positives = Positives::new(5);
        positives.merge(1, 2);
        positives.merge(3, 4);
        positives.merge(2, 3);

        assert_ne!(positives.clusters.find(0), positives.clusters.find(1));
        assert_eq!(positives.clusters.find(1), positives.clusters.find(2));
        assert_eq!(positives.clusters.find(2), positives.clusters.find(3));
        assert_eq!(positives.clusters.find(3), positives.clusters.find(4));
    }

    #[test]
    fn test_negatives_new() {
        let negatives = Negatives::new(5);
        assert_eq!(negatives.mutexes.len(), 5);
    }

    #[test]
    fn test_negatives_insert() {
        let mut negatives = Negatives::new(5);
        negatives.insert(1, 2);
        negatives.insert(3, 4);
        negatives.insert(2, 3);

        assert!(negatives.mutexes[1].contains(&2));
        assert!(negatives.mutexes[2].contains(&1));
        assert!(negatives.mutexes[3].contains(&4));
        assert!(negatives.mutexes[4].contains(&3));
        assert!(negatives.mutexes[2].contains(&3));
        assert!(negatives.mutexes[3].contains(&2));
    }
}
