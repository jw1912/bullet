use crate::graph::ir::{node::GraphIRNode, BackendMarker, GraphIR, GraphIRError};

pub struct GraphIRTransform<B: BackendMarker> {
    pub eliminated: Vec<usize>,
    pub new: Vec<GraphIRNode<B>>,
}

impl<B: BackendMarker> GraphIRTransform<B> {
    pub fn new(
        eliminated: &[usize],
        new: impl Into<Vec<GraphIRNode<B>>>,
    ) -> Result<Option<GraphIRTransform<B>>, GraphIRError> {
        Ok(Some(GraphIRTransform { eliminated: eliminated.to_vec(), new: new.into() }))
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
        if let Some(op) = data.parent_operation.as_ref() {
            for parent in op.nodes() {
                self.get_mut(parent.idx)?.num_children += 1;
            }
        } else if !self.leafs.insert(data.idx) {
            return Err(GraphIRError::NodeAlreadyExists);
        }

        if self.nodes.insert(data.idx, data).is_some() {
            return Err(GraphIRError::NodeAlreadyExists);
        }

        Ok(())
    }

    pub fn apply_transform(&mut self, desc: GraphIRTransform<B>) -> Result<(), GraphIRError> {
        let GraphIRTransform { mut eliminated, mut new } = desc;

        eliminated.sort();
        for dead in eliminated.into_iter().rev() {
            self.delete_node(dead)?;
        }

        new.sort_by_key(|x| x.idx);
        for new_data in new {
            self.insert_node(new_data)?;
        }

        Ok(())
    }
}
