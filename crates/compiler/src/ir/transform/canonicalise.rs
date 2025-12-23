use std::collections::HashSet;

use crate::ir::{IrError, IrGraph};

impl IrGraph {
    pub fn canonicalise(&mut self) -> Result<(), IrError> {
        for op_id in self.topo_order_ops()? {
            let op = self.get_op(op_id)?;
            let groups = op.op().commutating_groups();
            let inputs = op.inputs().to_vec();

            for group_i in &groups {
                for group_j in &groups {
                    if group_i.intersection(group_j).next().is_some() {
                        return Err("IrGraph::canonicalise: distinct commutating groups intersect!".into());
                    }
                }
            }

            for group in groups {
                let mut group = group.into_iter().collect::<Vec<_>>();
                let mut nodes = group.iter().map(|&i| inputs[i]).collect::<Vec<_>>();

                if nodes.iter().map(|&id| self.get_node_type(id)).collect::<Result<HashSet<_>, _>>()?.len() > 1 {
                    return Err("IrGraph::canonicalise: inputs within commutating group have differing types!".into());
                }

                group.sort();
                nodes.sort();

                for (idx, id) in group.into_iter().zip(nodes) {
                    self.get_op_mut(op_id)?.set_input(idx, id);
                }
            }
        }

        Ok(())
    }
}
