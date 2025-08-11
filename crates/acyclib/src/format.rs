use std::fmt;

use super::*;

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl<Ty: Type + Debug, Op: Operation + Debug> fmt::Display for Graph<Ty, Op> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(res: Result<T, GraphError>) -> Result<T, fmt::Error> {
            res.map_err(|_| fmt::Error)
        }

        let order = map(self.topo_order())?;

        write!(f, "## Graph ##")?;

        for id in order {
            writeln!(f)?;

            let node = map(self.get(id))?;
            let Node { id, ty, src, .. } = node;
            write!(f, "{id:?}: {ty:?}")?;

            if let Some(op) = src {
                write!(f, " = [{op:?}] {:?}", op.parents())?;
            }

            write!(f, ";")?;
        }

        Ok(())
    }
}
