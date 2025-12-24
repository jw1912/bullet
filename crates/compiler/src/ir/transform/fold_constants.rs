use crate::{
    common::DTypeTensor,
    ir::{
        IrError, IrGraph,
        operation::{Constant, IrOperation},
    },
};

impl IrGraph {
    pub fn fold_constants(&mut self) -> Result<(), IrError> {
        while self.fold_single_constant()? {}
        Ok(())
    }

    fn fold_single_constant(&mut self) -> Result<bool, IrError> {
        'op_loop: for op in self.ops.values() {
            let inputs = op.inputs();

            if !inputs.is_empty() {
                let mut consts = Vec::new();

                for &input in inputs {
                    let parent = self.get_op(self.get_parent_op(input)?)?;

                    if let Some(Constant(value)) = IrOperation::downcast(parent.op()) {
                        consts.push(value);
                    } else {
                        continue 'op_loop;
                    }
                }

                let output_ids = op.outputs().to_vec();

                let mut tensors = Vec::new();
                for &output in &output_ids {
                    let ty = self.get_node_type(output)?;

                    if let Some(size) = ty.size().evaluate_constant() {
                        tensors.push(DTypeTensor::new(ty.dtype(), size));
                    } else {
                        continue 'op_loop;
                    }
                }

                let mut outputs = tensors.iter_mut().collect::<Vec<_>>();

                op.op().evaluate(&consts, &mut outputs);

                for (old_out, value) in output_ids.into_iter().zip(tensors) {
                    let new_out = self.add_const(value);
                    self.swap_outputs(new_out, old_out)?;
                }

                self.eliminate_unused_ops()?;

                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use crate::common::Binary;

    use super::*;

    #[test]
    fn fold_constants() -> Result<(), IrError> {
        let mut ir = IrGraph::default();

        let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Sub)?;

        ir.register_output(t);
        ir.fold_constants()?;
        ir.eliminate_unused_ops()?;

        assert_eq!(ir.num_ops(), 1);
        assert_eq!(ir.num_nodes(), 1);

        for node in [x, y, z, w] {
            assert!(ir.get_node(node).is_err());
        }

        assert!(ir.get_node(t).is_ok());

        let t_op = ir.get_op(ir.get_parent_op(t)?)?;
        assert_eq!(t_op.inputs(), &[]);
        assert_eq!(t_op.outputs(), &[t]);
        assert_eq!(IrOperation::downcast(t_op.op()), Some(&Constant(DTypeTensor::F32(vec![2.0; 8]))));

        ir.check_valid()
    }
}
