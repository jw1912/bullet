mod error;

pub use error::DAGraphManagerError;

use std::{backtrace::Backtrace, collections::HashSet, fmt};

use crate::graph::{DAGraph, DAGraphError, Node, NodeId, Operation, format::FormattedGraph};

pub trait DAGraphType {
    type Type: Clone + PartialEq + fmt::Debug;
    type Operation: Operation<Self::Type> + fmt::Debug;
}

pub struct DAGraphManager<T: DAGraphType> {
    history: Vec<DAGraph<T::Type, T::Operation>>,
}

impl<T: DAGraphType> Default for DAGraphManager<T> {
    fn default() -> Self {
        Self { history: vec![DAGraph::default()] }
    }
}

impl<T: DAGraphType> DAGraphManager<T> {
    pub fn current(&self) -> &DAGraph<T::Type, T::Operation> {
        self.history.last().as_ref().unwrap()
    }

    pub fn roots(&self) -> HashSet<NodeId> {
        self.current().roots()
    }

    pub fn capture_error<U>(&self, result: Result<U, DAGraphError>) -> Result<U, DAGraphManagerError<T>> {
        result.map_err(|error| DAGraphManagerError {
            graph: self.current().clone(),
            trace: Backtrace::force_capture(),
            error,
        })
    }

    pub fn formatted(&self) -> Result<FormattedGraph, DAGraphError> {
        self.current().formatted()
    }

    pub fn modify<U, F>(&mut self, f: F) -> Result<U, DAGraphManagerError<T>>
    where
        F: FnOnce(&mut DAGraph<T::Type, T::Operation>) -> Result<U, DAGraphError>,
    {
        let mut new = self.current().clone();

        let res = self.capture_error(f(&mut new))?;

        self.history.push(new);

        Ok(res)
    }

    pub fn get(&self, node: NodeId) -> Result<&Node<T::Type, T::Operation>, DAGraphManagerError<T>> {
        self.capture_error(self.current().get(node))
    }

    pub fn add_node(&mut self, op: impl Into<T::Operation>) -> Result<NodeId, DAGraphManagerError<T>> {
        self.modify(|graph| graph.add_node(op))
    }

    pub fn replace_op(&mut self, node: NodeId, op: impl Into<T::Operation>) -> Result<(), DAGraphManagerError<T>> {
        self.modify(|graph| graph.replace_op(node, op))
    }

    pub fn eliminate_dead_nodes(&mut self, required: HashSet<NodeId>) -> Result<(), DAGraphManagerError<T>> {
        self.modify(|graph| graph.eliminate_dead_nodes(required))
    }

    pub fn topo_order(&self) -> Result<Vec<NodeId>, DAGraphManagerError<T>> {
        self.capture_error(self.current().topo_order())
    }
}
