use std::collections::{HashMap, HashSet};

use crate::{
    common::topo_order,
    ir::{
        IrError, IrGraph,
        operation::{IrOperation, IrOperationId},
    },
};

impl IrGraph {
    pub fn topo_order_ops(&self) -> Result<Vec<IrOperationId>, IrError> {
        let edges_rev = self
            .ops
            .iter()
            .map(|(&idx, data)| {
                data.inputs()
                    .iter()
                    .map(|&x| self.get_parent_op(x).map(|x| x.inner()))
                    .collect::<Result<_, _>>()
                    .map(|x| (idx.inner(), x))
            })
            .collect::<Result<_, _>>()?;

        topo_order(edges_rev)
            .ok_or("IrGraph::topo_order_ops: cycle found!".into())
            .map(|x| x.into_iter().map(IrOperationId::from_inner).collect())
    }

    pub fn check_valid(&self) -> Result<(), IrError> {
        let mut registered_outputs = HashSet::new();
        let mut expected_child_count = HashMap::new();
        let mut actual_child_count: HashMap<_, _> = self.nodes.keys().map(|x| (x, 0)).collect();

        fn check<T: Into<String>>(cond: bool, msg: T) -> Result<(), IrError> {
            cond.then_some(()).ok_or(format!("IrGraph::check_valid: {}!", msg.into()).into())
        }

        for op_id in self.topo_order_ops()? {
            let op = self.get_op(op_id)?;

            IrOperation::check(
                op.inputs().iter().map(|&x| self.get_node(x)).collect::<Result<Vec<_>, _>>()?,
                op.outputs().iter().map(|&x| self.get_node(x)).collect::<Result<Vec<_>, _>>()?,
                op.op().as_ref(),
            )?;

            for input in op.inputs() {
                *actual_child_count.get_mut(input).ok_or("IrGraph::check_valid: unexpected input node!")? += 1;
            }

            let output_types = op.op().outputs();

            check(
                op.outputs().len() == output_types.len(),
                format!(
                    "length of operation outputs ({}) does not match expected ({})",
                    op.outputs().len(),
                    output_types.len()
                ),
            )?;

            for (&output, ty) in op.outputs().iter().zip(output_types) {
                check(registered_outputs.insert(output), "output already registered")?;

                let node = self.get_node(output)?;
                check(node.ty() == ty, format!("output type ({:?}) does not match expected ({ty:?})", node.ty()))?;
                check(
                    node.id() == output,
                    format!("output id ({:?}) does not match expected ({output:?})", node.id()),
                )?;
                check(
                    expected_child_count.insert(output, node.children()).is_none(),
                    "expected child count already present",
                )?;
            }
        }

        for (id, count) in expected_child_count {
            let actual = *actual_child_count.get(&id).ok_or("IrGraph::check_valid: node does not exist!")?;
            check(count == actual, format!("actual child count ({actual}) does not match expected ({count})"))?;
        }

        Ok(())
    }
}
