mod error;

pub use error::GraphManagerError;

use std::{backtrace::Backtrace, collections::HashSet, fmt};

use crate::graph::{Graph, GraphError, Node, NodeId, Operation, format::FormattedGraph};

pub trait GraphType {
    type Type: Clone + PartialEq + fmt::Debug;
    type Operation: Operation<Self::Type> + fmt::Debug;
}

pub struct GraphManager<T: GraphType> {
    history: Vec<Graph<T::Type, T::Operation>>,
}

impl<T: GraphType> Default for GraphManager<T> {
    fn default() -> Self {
        Self { history: vec![Graph::default()] }
    }
}

impl<T: GraphType> GraphManager<T> {
    pub fn current(&self) -> &Graph<T::Type, T::Operation> {
        self.history.last().as_ref().unwrap()
    }

    pub fn roots(&self) -> HashSet<NodeId> {
        self.current().roots()
    }

    pub fn capture_error<U>(&self, result: Result<U, GraphError>) -> Result<U, GraphManagerError<T>> {
        result.map_err(|error| GraphManagerError {
            graph: self.current().clone(),
            trace: Backtrace::force_capture(),
            error,
        })
    }

    pub fn formatted(&self) -> Result<FormattedGraph, GraphError> {
        self.current().formatted()
    }

    pub fn modify<U, F>(&mut self, f: F) -> Result<U, GraphManagerError<T>>
    where
        F: FnOnce(&mut Graph<T::Type, T::Operation>) -> Result<U, GraphError>,
    {
        let mut new = self.current().clone();

        let res = self.capture_error(f(&mut new))?;

        self.history.push(new);

        Ok(res)
    }

    pub fn get(&self, node: NodeId) -> Result<&Node<T::Type, T::Operation>, GraphManagerError<T>> {
        self.capture_error(self.current().get(node))
    }

    pub fn add_node(&mut self, op: impl Into<T::Operation>) -> Result<NodeId, GraphManagerError<T>> {
        self.modify(|graph| graph.add_node(op))
    }

    pub fn replace_op(&mut self, node: NodeId, op: impl Into<T::Operation>) -> Result<(), GraphManagerError<T>> {
        self.modify(|graph| graph.replace_op(node, op))
    }

    pub fn eliminate_dead_nodes(&mut self, required: HashSet<NodeId>) -> Result<(), GraphManagerError<T>> {
        self.modify(|graph| graph.eliminate_dead_nodes(required))
    }

    pub fn topo_order(&self) -> Result<Vec<NodeId>, GraphManagerError<T>> {
        self.capture_error(self.current().topo_order())
    }
}
