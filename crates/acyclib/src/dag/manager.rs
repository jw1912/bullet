use std::{backtrace::Backtrace, collections::HashSet, error::Error, fmt};

use super::{DAGraph, DAGraphError, Node, NodeId, Operation, format::FormattedGraph};

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

pub struct DAGraphManagerError<T: DAGraphType> {
    pub trace: Backtrace,
    pub graph: DAGraph<T::Type, T::Operation>,
    pub error: DAGraphError,
}

impl<T: DAGraphType> fmt::Debug for DAGraphManagerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl<T: DAGraphType> fmt::Display for DAGraphManagerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "## Error Occurred ##")?;
        writeln!(f, "{:?}", self.error)?;
        writeln!(f, "## Graph State")?;
        writeln!(f, "{}", self.graph.formatted().unwrap())?;
        write!(f, "## Backtrace")?;
        let trace = self.trace.to_string();

        let mut count = 0;

        for line in trace.lines() {
            let split = line.split_whitespace().collect::<Vec<_>>();

            if !["rustc", "std", " core::", "toolchains"].iter().any(|x| split[1].contains(x)) {
                writeln!(f)?;

                if split[0] != "at" {
                    count += 1;
                    write!(f, "{count: >4}: {}", split[1])?;
                } else {
                    write!(f, "{line}")?;
                }
            }

            if split[1] == "main" {
                break;
            }
        }

        Ok(())
    }
}

impl<T: DAGraphType> Error for DAGraphManagerError<T> {}
