use std::{backtrace::Backtrace, error::Error, fmt};

use crate::{
    common::{DType, Size},
    ir::{
        IrError, IrGraph,
        node::{IrNode, IrNodeId, IrType},
        ops::IrOperation,
    },
};

pub struct IrManager {
    history: Vec<IrGraph>,
}

impl Default for IrManager {
    fn default() -> Self {
        Self { history: vec![IrGraph::default()] }
    }
}

impl IrManager {
    pub fn current(&self) -> &IrGraph {
        self.history.last().as_ref().unwrap()
    }

    pub fn capture_error<U>(&self, result: Result<U, IrError>) -> Result<U, IrManagerError> {
        result.map_err(|error| IrManagerError {
            graph: Box::new(self.current().clone()),
            trace: Backtrace::force_capture(),
            error,
        })
    }

    pub fn modify<U, F>(&mut self, f: F) -> Result<U, IrManagerError>
    where
        F: FnOnce(&mut IrGraph) -> Result<U, IrError>,
    {
        let mut new = self.current().clone();

        let res = self.capture_error(f(&mut new))?;

        self.history.push(new);

        Ok(res)
    }

    pub fn get(&self, node: IrNodeId) -> Result<&IrNode, IrManagerError> {
        self.capture_error(self.current().get_node(node))
    }

    pub fn add_leaf(&mut self, size: impl Into<Size>, dtype: DType) -> Result<IrNodeId, IrManagerError> {
        self.modify(|graph| Ok(graph.add_leaf(IrType::new(size, dtype))))
    }

    pub fn add_op(&mut self, op: impl IrOperation) -> Result<Vec<IrNodeId>, IrManagerError> {
        self.modify(|graph| graph.add_op(op))
    }

    pub fn register_output(&mut self, node: IrNodeId) -> Result<(), IrManagerError> {
        self.modify(|graph| {
            graph.register_output(node);
            Ok(())
        })
    }

    pub fn eliminate_dead_ops(&mut self) -> Result<(), IrManagerError> {
        self.modify(|graph| graph.eliminate_dead_ops())
    }
}

impl fmt::Display for IrManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.current())
    }
}

pub struct IrManagerError {
    pub trace: Backtrace,
    pub graph: Box<IrGraph>,
    pub error: IrError,
}

impl fmt::Debug for IrManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for IrManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "## Error Occurred ##")?;
        writeln!(f, "{:?}", self.error)?;
        writeln!(f, "## Graph State")?;
        writeln!(f, "{}", self.graph)?;
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

impl Error for IrManagerError {}
