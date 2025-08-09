use std::collections::{HashMap, HashSet};

use crate::graph::ir::{
    node::{AnnotatedNode, GraphIRNode},
    BackendMarker, GraphIR, GraphIRError,
};

impl<B: BackendMarker> GraphIR<B> {
    pub fn root(&self) -> Result<AnnotatedNode, GraphIRError> {
        let roots = self.nodes.values().filter(|node| node.num_children == 0).count();

        if roots != 1 {
            return Err(GraphIRError::MultipleRoots);
        }

        let idx = *self.topo_order()?.last().unwrap();
        let data = self.get(idx)?;

        Ok(AnnotatedNode { idx, shape: data.info.shape })
    }

    pub fn is_valid(&self) -> Result<(), GraphIRError> {
        let mut children_count: HashMap<_, _> = self.nodes.keys().map(|&idx| (idx, 0)).collect();

        for node in self.topo_order()? {
            if let Ok(data) = self.get(node) {
                if data.idx != node {
                    return Err(GraphIRError::NodeDataDoesNotMatchExpected);
                }

                self.is_data_valid(data)?;

                if let Some(op) = &data.parent_operation {
                    for parent in op.nodes() {
                        *children_count.get_mut(&parent.idx).unwrap() += 1;

                        let actual_parent = self.nodes.get(&parent.idx).ok_or(GraphIRError::NodeDoesNotExist)?;

                        if parent.idx != actual_parent.idx || parent.shape.size() != actual_parent.info.shape.size() {
                            return Err(GraphIRError::NodeDataDoesNotMatchExpected);
                        }
                    }
                }
            }
        }

        for idx in self.nodes.keys() {
            if children_count.get(idx).copied() != self.nodes.get(idx).map(|x| x.num_children) {
                return Err(GraphIRError::NodeHasInvalidNumberOfChildren);
            }
        }

        Ok(())
    }

    pub fn is_data_valid(&self, data: &GraphIRNode<B>) -> Result<(), GraphIRError> {
        if let Some(op) = &data.parent_operation {
            let shape = op.output_shape(self)?;
            let batched = op.output_batched(self)?;
            let requires_grad = op.output_requires_grad(self)?;

            if data.info.shape != shape || data.info.batched != batched || data.info.requires_grad != requires_grad {
                return Err(GraphIRError::NodeDataDoesNotMatchExpected);
            }
        }

        Ok(())
    }

    pub fn topo_order(&self) -> Result<Vec<usize>, GraphIRError> {
        let mut edges: HashMap<usize, HashSet<usize>> = self.nodes.keys().map(|idx| (*idx, HashSet::new())).collect();
        let mut edgest: HashMap<usize, HashSet<usize>> = self.nodes.keys().map(|idx| (*idx, HashSet::new())).collect();

        for (&idx, data) in self.nodes.iter() {
            assert_eq!(idx, data.idx);

            if let Some(op) = &data.parent_operation {
                for node in op.nodes() {
                    edges.get_mut(&node.idx).unwrap().insert(idx);
                    edgest.get_mut(&idx).unwrap().insert(node.idx);
                }
            }
        }

        let mut leafs: HashSet<usize> = self.leafs.clone();

        let mut topo = Vec::new();

        loop {
            if leafs.is_empty() {
                break;
            }

            let n = *leafs.iter().next().unwrap();
            leafs.remove(&n);
            topo.push(n);

            let children = edges.get(&n).unwrap().clone();
            for child in children {
                edges.get_mut(&n).unwrap().remove(&child);

                let parents = edgest.get_mut(&child).unwrap();
                parents.remove(&n);
                if parents.is_empty() {
                    leafs.insert(child);
                }
            }
        }

        if edges.values().all(HashSet::is_empty) && edgest.values().all(HashSet::is_empty) {
            Ok(topo)
        } else {
            Err(GraphIRError::CannotBeTopologicallyOrdered)
        }
    }
}
