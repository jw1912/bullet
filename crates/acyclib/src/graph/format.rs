use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::graph::{DAGraph, DAGraphError, NodeId, Operation};

impl<Ty: Clone + PartialEq + fmt::Debug, Op: Operation<Ty> + fmt::Debug> DAGraph<Ty, Op> {
    pub fn formatted(&self) -> Result<FormattedGraph, DAGraphError> {
        let mut graph = DAGraph::default();

        let mut node_map = HashMap::new();

        for id in self.topo_order()? {
            let node = self.get(id)?;

            let parents = node.op().parents().iter().map(|p| *node_map.get(p).unwrap()).collect();
            let out_type = format!("{:?}", node.ty());
            let new_id = graph.add_node(StringOperation { parents, op: format!("{:?}", node.op()), out_type })?;

            assert!(node_map.insert(node.id(), new_id).is_none());
        }

        Ok(graph)
    }
}

#[derive(Clone)]
pub struct StringOperation {
    parents: HashSet<NodeId>,
    op: String,
    out_type: String,
}

impl Operation<String> for StringOperation {
    fn parents(&self) -> HashSet<NodeId> {
        self.parents.clone()
    }

    fn out_type(&self, _graph: &DAGraph<String, Self>) -> Result<String, DAGraphError> {
        Ok(self.out_type.clone())
    }
}

pub type FormattedGraph = DAGraph<String, StringOperation>;

impl fmt::Display for FormattedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(res: Result<T, DAGraphError>) -> Result<T, fmt::Error> {
            res.map_err(|_| fmt::Error)
        }

        let order = map(self.topo_order())?;
        let nd = |ln: usize| format!("%{ln}");

        let mut ids = HashMap::new();
        let mut idl = 0;
        let mut tyl = 4;
        let mut prl = 0;
        let mut opl = 9;
        let mut lines = Vec::new();

        for (line, id) in order.into_iter().enumerate() {
            ids.insert(id, line);

            let node = map(self.get(id))?;

            let id = nd(line);
            let ty = node.ty().to_string();
            let op = node.op();

            idl = idl.max(id.len());
            tyl = tyl.max(ty.len());

            let pr = (!op.parents.is_empty()).then(|| {
                let mut parents: Vec<_> = op.parents.iter().filter_map(|p| ids.get(p).cloned()).collect();
                parents.sort();

                let mut parents = parents.into_iter();
                let first = parents.next().unwrap();

                let parents = format!("{{{}}}", parents.fold(nd(first), |x, y| format!("{x}, {}", nd(y))));
                prl = prl.max(parents.len());

                parents
            });

            opl = opl.max(op.op.len());
            lines.push((id, ty, pr, op.op.clone()));
        }

        let mlen = tyl + prl + opl;
        let bar = format!("+{}+", "-".repeat(idl + mlen + 12));
        let s = |num| " ".repeat(num);

        writeln!(f, "{bar}")?;

        writeln!(f, "| Flow{} | Type{} | Operation{} |", s(idl + prl), s(tyl - 4), s(opl - 9))?;

        writeln!(f, "{bar}")?;

        for (id, ty, pr, op) in lines {
            if let Some(pr) = pr {
                write!(f, "| {id}{} <- {pr}{}", s(idl - id.len()), s(prl - pr.len()))?;
            } else {
                write!(f, "| {id}{}", s(idl - id.len() + prl + 4))?;
            }

            writeln!(f, " | {ty}{} | {op}{} |", s(tyl - ty.len()), s(opl - op.len()))?;
        }

        write!(f, "{bar}")?;

        Ok(())
    }
}
