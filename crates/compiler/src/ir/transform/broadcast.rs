use crate::ir::{IrError, IrGraph, operation::{BroadcastAcrossDimension, IrOperation, IrUnary}};



impl IrGraph {
    fn fold_single_broadcast(&mut self) -> Result<bool, IrError> {
        for op in self.ops.values() {
            if let Some(unary) = IrOperation::downcast::<IrUnary>(op.op()) {
                let [output] = op.outputs()[..] else { panic!() };
                let [parent] = op.inputs()[..] else { panic!() };

                let parent_op = self.get_op(self.get_parent_op(parent)?)?;
                if let Some(broadcast) = IrOperation::downcast::<BroadcastAcrossDimension>(parent_op.op()).cloned() {
                    assert_eq!(parent_op.outputs()[..], [parent]);
                    let [grandparent] = parent_op.inputs()[..] else { panic!() };
                    let new_input = self.add_unary(grandparent, unary.op())?;
                    let new_dtype = self.get_node_type(new_input)?.dtype();
                    let new_output = self.add_op([new_input], broadcast.with_new_dtype(new_dtype))?;
                    self.swap_outputs(new_output[0], output)?;

                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}