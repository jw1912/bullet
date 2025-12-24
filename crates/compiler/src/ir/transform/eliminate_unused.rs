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

#[cfg(test)]
mod tests {
    use crate::{
        common::{Binary, DTypeTensor},
        ir::{IrError, IrGraph},
    };

    #[test]
    fn eliminate_unused_ops() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Sub)?;

        ir.register_output(z);

        ir.eliminate_unused_ops()?;

        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());
        assert!(ir.get_node(z).is_ok());
        assert!(ir.get_node(w).is_err());
        assert!(ir.get_node(t).is_err());

        ir.check_valid()
    }
}
