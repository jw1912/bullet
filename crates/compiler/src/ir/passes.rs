use crate::{
    common::DTypeTensor,
    ir::{
        IrError, IrGraph,
        operation::{Constant, IrOperation},
    },
};

impl IrGraph {
    pub fn eliminate_dead_ops(&mut self) -> Result<(), IrError> {
        for op_id in self.topo_order_ops()?.into_iter().rev() {
            let dead_op = self.get_op(op_id)?.outputs().iter().all(|output| {
                let node = self.get_node(*output).unwrap();
                node.children() == 0 && !self.outputs.contains(output)
            });

            if dead_op {
                self.remove_op(op_id)?;
            }
        }

        Ok(())
    }

    pub fn propagate_constants(&mut self) -> Result<(), IrError> {
        while self.propagate_single_constant()? {}
        Ok(())
    }

    fn propagate_single_constant(&mut self) -> Result<bool, IrError> {
        'op_loop: for &op_id in self.ops.keys() {
            let op = self.get_op(op_id)?;
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

                self.eliminate_dead_ops()?;

                return Ok(true);
            }
        }

        Ok(false)
    }
}
