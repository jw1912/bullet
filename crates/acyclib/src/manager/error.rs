use std::{backtrace::Backtrace, error::Error, fmt};

use crate::{
    graph::{Graph, GraphError},
    manager::GraphType,
};

pub struct GraphManagerError<T: GraphType> {
    pub trace: Backtrace,
    pub graph: Graph<T::Type, T::Operation>,
    pub error: GraphError,
}

impl<T: GraphType> fmt::Debug for GraphManagerError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl<T: GraphType> fmt::Display for GraphManagerError<T> {
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

impl<T: GraphType> Error for GraphManagerError<T> {}
