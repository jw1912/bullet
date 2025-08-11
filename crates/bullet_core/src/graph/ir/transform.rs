use std::collections::{HashMap, HashSet};

use crate::graph::ir::{node::GraphIRNode, properties::topo_order, BackendMarker, GraphIR, GraphIRError};

pub struct GraphIRTransform<B: BackendMarker> {
    pub delete: Vec<usize>,
    pub create: Vec<GraphIRNode<B>>,
}

impl<B: BackendMarker> GraphIRTransform<B> {
    pub fn new(
        delete: impl Into<Vec<usize>>,
        create: impl Into<Vec<GraphIRNode<B>>>,
    ) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        Ok(Some(GraphIRTransform { delete: delete.into(), create: create.into() }))
    }
}

impl<B: BackendMarker> GraphIR<B> {
    pub fn delete_node(&mut self, node: usize) -> Result<(), GraphIRError> {
        if let Some(Some(op)) = self.nodes.get(&node).map(|x| &x.parent_operation) {
            for parent in op.nodes() {
                self.get_mut(parent.idx)?.num_children -= 1;
            }
        }

        if self.nodes.remove(&node).is_none() {
            return Err(GraphIRError::NodeDoesNotExist);
        }

        Ok(())
    }

    pub fn insert_node(&mut self, data: GraphIRNode<B>) -> Result<(), GraphIRError> {
        if let Some(id) = data.id.as_ref() {
            if self.ids.contains(id) {
                return Err(GraphIRError::NodeWithIdAlreadyExists(id.clone()));
            }

            self.ids.insert(id.to_string());
        }

        if let Some(op) = data.parent_operation.as_ref() {
            for parent in op.nodes() {
                self.get_mut(parent.idx)?.num_children += 1;
            }
        }

        if self.nodes.insert(data.idx, data).is_some() {
            return Err(GraphIRError::NodeAlreadyExists);
        }

        Ok(())
    }

    pub fn apply_transform(&mut self, desc: GraphIRTransform<B>) -> Result<(), GraphIRError> {
        let GraphIRTransform { delete, create } = desc;

        let eliminated = delete.into_iter().collect::<HashSet<_>>();

        let subgraph = eliminated
            .iter()
            .map(|&idx| {
                let op = self.get(idx).unwrap().parent_operation.as_ref();
                let set =
                    op.map(|x| x.nodes().iter().filter_map(|x| eliminated.contains(&x.idx).then_some(x.idx)).collect());
                let set = set.unwrap_or_default();
                (idx, set)
            })
            .collect();

        let eliminated = topo_order(subgraph).ok_or(GraphIRError::CannotBeTopologicallyOrdered)?;

        for dead in eliminated.into_iter().rev() {
            self.delete_node(dead)?;
        }

        let mut map: HashMap<_, _> = create.into_iter().map(|data| (data.idx, data)).collect();

        let subgraph = map
            .iter()
            .map(|(&idx, data)| {
                let op = data.parent_operation.as_ref();
                let set =
                    op.map(|x| x.nodes().iter().filter_map(|x| map.contains_key(&x.idx).then_some(x.idx)).collect());
                let set = set.unwrap_or_default();
                (idx, set)
            })
            .collect();

        let order = topo_order(subgraph).ok_or(GraphIRError::CannotBeTopologicallyOrdered)?;

        for idx in order {
            self.insert_node(map.remove(&idx).unwrap())?;
        }

        Ok(())
    }
}
