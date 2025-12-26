use crate::{
    core::DTypeTensor,
    ir::{
        IR, IRTrace,
        graph::operation::{Constant, IrOperation},
        transform::{EliminateUnusedOperations, IrTransform},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldConstants;

impl IrTransform for FoldConstants {
    /// Constant evaluate all operations that have constant inputs:
    /// ```text
    /// irgraph() {
    ///     %0 = constant<f32[1]>(1.0)
    ///     %1 = constant<f32[1]>(2.0)
    ///     %2 = %0 + %1
    ///     %3 = %2 * %2
    ///     return %3
    /// }
    /// ```text
    /// Will be rewritten to
    /// irgraph() {
    ///     %3 = constant<f32[1]>(9.0)
    ///     return %3
    /// }
    /// ```
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        while fold_single_constant(ir)? {}
        Ok(())
    }
}

fn fold_single_constant(ir: &mut IR) -> Result<bool, IRTrace> {
    'op_loop: for op in ir.operations() {
        let inputs = op.inputs();

        if !inputs.is_empty() {
            let mut consts = Vec::new();

            for &input in inputs {
                let parent = ir.get_op(ir.get_parent_op(input)?)?;

                if let Some(Constant(value)) = IrOperation::downcast(parent.op()) {
                    consts.push(value);
                } else {
                    continue 'op_loop;
                }
            }

            let output_ids = op.outputs().to_vec();

            let mut tensors = Vec::new();
            for &output in &output_ids {
                let ty = ir.get_node(output)?.ty();

                if let Some(size) = ty.size().evaluate_constant() {
                    tensors.push(DTypeTensor::new(ty.dtype(), size));
                } else {
                    continue 'op_loop;
                }
            }

            let mut outputs = tensors.iter_mut().collect::<Vec<_>>();

            op.op().evaluate(&consts, &mut outputs);

            for (old_out, value) in output_ids.into_iter().zip(tensors) {
                let new_out = ir.add_const(value);
                ir.swap_outputs(new_out, old_out)?;
            }

            ir.transform(EliminateUnusedOperations)?;

            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use crate::core::Binary;

    use super::*;

    #[test]
    fn fold_constants() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
        let z = ir.add_binary(x, y, Binary::Add)?;
        let w = ir.add_binary(z, y, Binary::Add)?;
        let t = ir.add_binary(w, y, Binary::Sub)?;

        ir.register_output(t);
        ir.transform(FoldConstants)?;

        assert_eq!(ir.num_ops(), 1);
        assert_eq!(ir.num_nodes(), 1);
        assert_eq!(ir.is_child_of(t)?, Some(&Constant(DTypeTensor::F32(vec![2.0; 8]))));

        ir.check_valid()
    }
}
