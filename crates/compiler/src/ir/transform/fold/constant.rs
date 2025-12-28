use std::rc::Rc;

use crate::{
    core::DTypeTensor,
    ir::{
        IR, IRTrace,
        graph::{
            IrNodeId, IrOperation, IrOperationType,
            operation::{Constant, ScalarConstant},
        },
    },
};

pub fn fold_constant_expression(
    ir: &IR,
    inputs: &[IrNodeId],
    op: &Rc<dyn IrOperationType>,
) -> Result<Option<Vec<Constant>>, IRTrace> {
    if !inputs.is_empty() {
        let mut consts = Vec::new();

        for &input in inputs {
            let parent = ir.get_op(ir.get_parent_op(input)?)?;

            if let Some(Constant(value)) = IrOperation::downcast(parent.op()).cloned() {
                consts.push(value);
            } else if let Some(Some(scalar)) = IrOperation::downcast(parent.op()).map(ScalarConstant::to_tensor) {
                consts.push(scalar);
            } else {
                return Ok(None);
            }
        }

        let mut tensors = Vec::new();
        for &ty in &op.outputs() {
            if let Some(size) = ty.size().evaluate_constant() {
                tensors.push(DTypeTensor::new(ty.dtype(), size));
            } else {
                return Ok(None);
            }
        }

        let inputs = consts.iter().collect::<Vec<_>>();
        let mut outputs = tensors.iter_mut().collect::<Vec<_>>();

        op.evaluate(&inputs, &mut outputs);

        return Ok(Some(tensors.into_iter().map(Constant).collect()));
    }

    Ok(None)
}
