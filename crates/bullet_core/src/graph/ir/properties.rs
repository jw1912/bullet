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
        let edges_rev = self
            .nodes
            .iter()
            .map(|(&idx, data)| {
                let op = data.parent_operation.as_ref();
                let set = op.map(|x| x.nodes()).unwrap_or(Vec::new()).iter().map(|x| x.idx).collect();
                (idx, set)
            })
            .collect();

        topo_order(edges_rev).ok_or(GraphIRError::CannotBeTopologicallyOrdered)
    }
}

pub fn topo_order(mut edges_rev: HashMap<usize, HashSet<usize>>) -> Option<Vec<usize>> {
    let mut edges: HashMap<usize, HashSet<usize>> = edges_rev.keys().map(|idx| (*idx, HashSet::new())).collect();

    for (&idx, parents) in edges_rev.iter() {
        for parent in parents {
            edges.get_mut(parent).unwrap().insert(idx);
        }
    }

    let mut leafs: HashSet<usize> =
        edges_rev.iter().filter_map(|(&idx, parents)| parents.is_empty().then_some(idx)).collect();

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

            let parents = edges_rev.get_mut(&child).unwrap();
            parents.remove(&n);
            if parents.is_empty() {
                leafs.insert(child);
            }
        }
    }

    (edges.values().all(HashSet::is_empty) && edges_rev.values().all(HashSet::is_empty)).then_some(topo)
}
