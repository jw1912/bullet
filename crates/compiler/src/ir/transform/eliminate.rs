use crate::ir::{
    IrError, IrGraph,
    operation::{IrCopy, IrOperation},
};

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

    pub fn eliminate_copies(&mut self) -> Result<(), IrError> {
        for op_id in self.topo_order_ops()?.into_iter().rev() {
            let op = self.get_op(op_id)?;

            if IrOperation::downcast::<IrCopy>(op.op()).is_some()
                && let [input] = op.inputs()[..]
                && let [output] = op.outputs()[..]
            {
                if !self.is_output(output) {
                    self.replace_input(input, output)?;
                    self.remove_op(op_id)?;
                    continue;
                }

                if !self.is_output(input) {
                    self.replace_input_no_topo_check(output, input)?;
                    self.swap_outputs(input, output)?;
                    self.remove_op(op_id)?;
                }
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
                            let new_out = self.copy(out_i)?;
                            self.swap_outputs(new_out, out_j)?;
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
        common::{Binary, DType, DTypeTensor},
        ir::{IrError, IrGraph, node::IrType},
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
    fn eliminate_copies() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_leaf(IrType::new(1, DType::F32));
        let y = ir.copy(x)?;
        let z = ir.copy(y)?;
        let w = ir.copy(z)?;

        let a = ir.add_binary(x, y, Binary::Add)?;
        let b = ir.add_binary(z, w, Binary::Mul)?;
        let c = ir.add_binary(a, b, Binary::Add)?;

        ir.register_output(a);
        ir.register_output(b);
        ir.register_output(c);
        ir.eliminate_copies()?;

        assert_eq!(ir.num_ops(), 4);

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
