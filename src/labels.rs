use itertools::Itertools;
use ndarray::{Array, IxDyn};
use std::{
    sync::{Arc, Mutex},
    vec,
};

use crate::clustering::Positives;

struct LabelWrapper {
    labels: Array<usize, IxDyn>,
    label_to_node_id: Arc<Mutex<Vec<Option<usize>>>>,
    node_id_to_label: Arc<Mutex<Vec<Option<usize>>>>,
}

impl LabelWrapper {
    fn new(labels: Array<usize, IxDyn>) -> Self {
        let unique_labels: Vec<usize> = labels
            .iter()
            .unique()
            .filter(|x| **x > 0)
            .copied()
            .collect();
        let max_label = *unique_labels.iter().max().unwrap();
        let label_to_node_id = Arc::new(Mutex::new(vec![None; max_label + 1]));
        let node_id_to_label = Arc::new(Mutex::new(vec![None; unique_labels.len()]));

        for (node_id, &label) in labels.iter().enumerate() {
            label_to_node_id.lock().unwrap()[label] = Some(node_id);
            node_id_to_label.lock().unwrap()[node_id] = Some(label);
        }

        LabelWrapper {
            labels,
            label_to_node_id,
            node_id_to_label,
        }
    }

    fn filter(&mut self, node_ids: &[usize]) {
        let mut label_to_node_id = self.label_to_node_id.lock().unwrap();
        let mut node_id_to_label = self.node_id_to_label.lock().unwrap();

        for &node_id in node_ids {
            if let Some(label) = node_id_to_label[node_id] {
                label_to_node_id[label] = None;
                node_id_to_label[node_id] = None;
                self.labels[node_id] = 0;
            }
        }

        // Reset the mapping to have incrementing node_ids
        let mut next_node_id = 0;
        for (label, node_id) in label_to_node_id.iter_mut().enumerate() {
            if node_id.is_some() {
                *node_id = Some(next_node_id);
                node_id_to_label[next_node_id] = Some(label);
                next_node_id += 1;
            }
        }
    }

    fn merge(&mut self, clusters: &Positives) {
        let mut label_to_node_id = self.label_to_node_id.lock().unwrap();
        let mut node_id_to_label = self.node_id_to_label.lock().unwrap();

        (0..label_to_node_id.len()).for_each(|old_label: usize| {
            if let Some(old_node_id) = label_to_node_id[old_label] {
                // get new id and label
                let new_node_id = clusters.clusters.find(old_node_id);
                if new_node_id != old_node_id {
                    // this is a non-representative label, pointing to a non-representative node id
                    // we need to update the label to point to the representative node id
                    // and update the old node id to point to None
                    label_to_node_id[old_label] = Some(new_node_id);
                    node_id_to_label[old_node_id] = None;
                }
            }
        });

        let labels = &mut self.labels;
        let data = labels.as_slice_memory_order_mut().unwrap();

        data.iter_mut().for_each(|label| {
            *label = node_id_to_label[label_to_node_id[*label].unwrap()].unwrap()
        });
    }

    fn get_node_id(&self, label: usize) -> Option<usize> {
        self.label_to_node_id.lock().unwrap()[label]
    }

    fn get_label(&self, node_id: usize) -> Option<usize> {
        self.node_id_to_label.lock().unwrap()[node_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_wrapper() {
        let data: Vec<usize> = vec![1, 2, 3, 4, 5];
        let labels = Array::from_shape_vec(IxDyn(&[5]), data).unwrap();
        let mut wrapper = LabelWrapper::new(labels.clone());

        // Test initial mapping
        assert_eq!(wrapper.get_node_id(1), Some(0));
        assert_eq!(wrapper.get_node_id(2), Some(1));
        assert_eq!(wrapper.get_node_id(3), Some(2));
        assert_eq!(wrapper.get_node_id(4), Some(3));
        assert_eq!(wrapper.get_node_id(5), Some(4));

        // Test initial node_id_to_label mapping
        assert_eq!(wrapper.get_label(0), Some(1));
        assert_eq!(wrapper.get_label(1), Some(2));
        assert_eq!(wrapper.get_label(2), Some(3));
        assert_eq!(wrapper.get_label(3), Some(4));
        assert_eq!(wrapper.get_label(4), Some(5));

        // Test filter method
        wrapper.filter(&[1, 3]);

        // Test updated mapping
        assert_eq!(wrapper.get_node_id(1), Some(0));
        assert_eq!(wrapper.get_node_id(2), None);
        assert_eq!(wrapper.get_node_id(3), Some(1));
        assert_eq!(wrapper.get_node_id(4), None);
        assert_eq!(wrapper.get_node_id(5), Some(2));

        // Test updated node_id_to_label mapping
        assert_eq!(wrapper.get_label(0), Some(1),);
        assert_eq!(wrapper.get_label(1), Some(3));
        assert_eq!(wrapper.get_label(2), Some(5));

        // Test updated labels
        assert_eq!(
            wrapper.labels,
            Array::from_shape_vec(IxDyn(&[5]), vec![1, 0, 3, 0, 5]).unwrap()
        );
    }

    #[test]
    fn test_merge() {
        let data: Vec<usize> = vec![1, 2, 3, 4, 5];
        let labels = Array::from_shape_vec(IxDyn(&[5]), data).unwrap();
        let mut wrapper = LabelWrapper::new(labels.clone());

        // Test initial mapping
        assert_eq!(wrapper.get_node_id(1), Some(0));
        assert_eq!(wrapper.get_node_id(2), Some(1));
        assert_eq!(wrapper.get_node_id(3), Some(2));
        assert_eq!(wrapper.get_node_id(4), Some(3));
        assert_eq!(wrapper.get_node_id(5), Some(4));

        // Test initial node_id_to_label mapping
        assert_eq!(wrapper.get_label(0), Some(1));
        assert_eq!(wrapper.get_label(1), Some(2));
        assert_eq!(wrapper.get_label(2), Some(3));
        assert_eq!(wrapper.get_label(3), Some(4));
        assert_eq!(wrapper.get_label(4), Some(5));

        // Create a Positives struct with cluster mappings
        let mut clusters = Positives::new(5);
        clusters.clusters.union(1, 3); // Merge clusters 1 and 3

        // Call the merge function
        wrapper.merge(&clusters);

        // Test clustered mapping
        assert_eq!(wrapper.get_node_id(1), Some(0));
        assert_eq!(wrapper.get_node_id(2), wrapper.get_node_id(4));
        assert_eq!(wrapper.get_node_id(3), Some(2));
        assert_eq!(wrapper.get_node_id(5), Some(4));

        // Test clustered node_id_to_label mapping
        assert_eq!(wrapper.get_label(0), Some(1));
        assert_eq!(wrapper.get_label(1), None);
        assert_eq!(wrapper.get_label(2), Some(3));
        assert_eq!(wrapper.get_label(3), Some(4));
        assert_eq!(wrapper.get_label(4), Some(5));

        // Test updated labels
        assert_eq!(
            wrapper.labels,
            Array::from_shape_vec(IxDyn(&[5]), vec![1, 4, 3, 4, 5]).unwrap()
        );
    }
}
