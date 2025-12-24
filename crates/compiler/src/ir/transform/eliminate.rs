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

    pub fn eliminate_common_subexprs(&mut self) -> Result<(), IrError> {
        while self.eliminate_single_common_subexpr()? {}
        Ok(())
    }

    fn eliminate_single_common_subexpr(&mut self) -> Result<bool, IrError> {
        let ops = self.topo_order_ops()?;

        for (i, &op_id_i) in ops.iter().enumerate() {
            for &op_id_j in ops.iter().skip(i + 1) {
                let op_i = self.get_op(op_id_i)?.clone();
                let op_j = self.get_op(op_id_j)?.clone();

                if op_i.inputs() == op_j.inputs() && op_i.op().equals(op_j.op()) {
                    for (&out_i, &out_j) in op_i.outputs().iter().zip(op_j.outputs()) {
                        self.replace_input(out_i, out_j)?;

                        if self.is_output(out_j) {
                            self.unregister_output(out_j);
                            self.register_output(out_i);
                        }
                    }

                    self.eliminate_unused_ops()?;

                    return Ok(true);
                }
            }
        }

        Ok(false)
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

    #[test]
    fn eliminate_common_subexprs() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));

        let z1 = ir.add_binary(x, y, Binary::Add)?;
        let w1 = ir.add_binary(z1, y, Binary::Add)?;
        let t1 = ir.add_binary(w1, y, Binary::Sub)?;

        let z2 = ir.add_binary(x, y, Binary::Add)?;
        let w2 = ir.add_binary(z2, y, Binary::Add)?;
        let t2 = ir.add_binary(w2, y, Binary::Sub)?;

        ir.register_output(t1);
        ir.register_output(t2);
        ir.eliminate_common_subexprs()?;

        assert!(ir.get_node(x).is_ok());
        assert!(ir.get_node(y).is_ok());

        ir.check_valid()
    }
}
