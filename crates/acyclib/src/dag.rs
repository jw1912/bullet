pub mod format;
pub mod manager;
pub mod node;
pub mod topo;

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

pub use node::{Node, NodeId};

#[derive(Clone, Debug)]
pub enum DAGraphError {
    NodeDoesNotExist,
    NodeWithIdAlreadyExists,
    NodeIsNotRoot,
    FailedTypeCheck,
    Cyclic,
    Message(String),
}

pub trait Operation<Ty: Clone + PartialEq>: Clone {
    fn parents(&self) -> HashSet<NodeId>;

    fn out_type(&self, graph: &DAGraph<Ty, Self>) -> Result<Ty, DAGraphError>;
}

#[derive(Clone)]
pub struct DAGraph<Ty: Clone + PartialEq, Op: Operation<Ty>> {
    nodes: HashMap<NodeId, Node<Ty, Op>>,
}

impl<Ty: Clone + PartialEq, Op: Operation<Ty>> Default for DAGraph<Ty, Op> {
    fn default() -> Self {
        Self { nodes: HashMap::new() }
    }
}

impl<Ty: Clone + PartialEq, Op: Operation<Ty>> DAGraph<Ty, Op> {
    pub fn get(&self, id: NodeId) -> Result<&Node<Ty, Op>, DAGraphError> {
        self.nodes.get(&id).ok_or(DAGraphError::NodeDoesNotExist)
    }

    fn get_mut(&mut self, id: NodeId) -> Result<&mut Node<Ty, Op>, DAGraphError> {
        self.nodes.get_mut(&id).ok_or(DAGraphError::NodeDoesNotExist)
    }

    pub fn roots(&self) -> HashSet<NodeId> {
        self.nodes.values().filter_map(|node| (node.children == 0).then_some(node.id)).collect()
    }

    pub fn topo_order(&self) -> Result<Vec<NodeId>, DAGraphError> {
        let edges_rev =
            self.nodes.iter().map(|(&idx, data)| (idx.0, data.op.parents().iter().map(|x| x.0).collect())).collect();

        topo::topo_order(edges_rev).ok_or(DAGraphError::Cyclic).map(|x| x.into_iter().map(NodeId).collect())
    }

    pub fn add_node(&mut self, op: impl Into<Op>) -> Result<NodeId, DAGraphError> {
        let op = op.into();
        let ty = op.out_type(self)?;
        let node = Node::<Ty, Op>::new(ty, op);
        let id = node.id;

        for parent in node.op.parents() {
            self.get_mut(parent)?.children += 1;
        }

        self.nodes.insert(id, node).map_or(Ok(()), |_| Err(DAGraphError::NodeWithIdAlreadyExists))?;

        Ok(id)
    }

    pub fn remove(&mut self, id: NodeId) -> Result<(), DAGraphError> {
        let node = self.get(id)?;

        if node.children > 0 {
            return Err(DAGraphError::NodeIsNotRoot);
        }

        for parent in node.op.parents() {
            self.get_mut(parent)?.children -= 1;
        }

        self.nodes.remove(&id).expect("Already verified node is present!");

        Ok(())
    }

    pub fn replace_op(&mut self, id: NodeId, new_op: impl Into<Op>) -> Result<(), DAGraphError> {
        let mut new_op = new_op.into();

        if self.get(id)?.ty != new_op.out_type(self)? {
            return Err(DAGraphError::FailedTypeCheck);
        }

        for parent in new_op.parents() {
            self.get_mut(parent)?.children += 1;
        }

        let node = self.get_mut(id)?;

        std::mem::swap(&mut new_op, &mut node.op);

        for parent in new_op.parents() {
            self.get_mut(parent)?.children -= 1;
        }

        // replacing op is an opportunity to introduce cycles...
        self.topo_order()?;

        Ok(())
    }

    pub fn eliminate_dead_nodes(&mut self, required: HashSet<NodeId>) -> Result<(), DAGraphError> {
        for id in self.topo_order()?.into_iter().rev() {
            if !required.contains(&id) && self.get(id)?.children == 0 {
                self.remove(id)?;
            }
        }

        Ok(())
    }
}
