use crate::ir::{IrError, IrGraph};

impl IrGraph {
    pub fn eliminate_unused_ops(&mut self) -> Result<(), IrError> {
        for op_id in self.topo_order_ops()?.into_iter().rev() {
            let unused_op = self.get_op(op_id)?.outputs().iter().all(|output| {
                let node = self.get_node(*output).unwrap();
                node.children() == 0 && !self.outputs.contains(output)
            });

            if unused_op {
                self.remove_op(op_id)?;
            }
        }

        Ok(())
    }
}
