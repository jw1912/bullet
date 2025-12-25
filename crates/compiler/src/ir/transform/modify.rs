use std::rc::Rc;

use crate::ir::{
    IR, IRTrace,
    graph::{IrNodeId, IrOperationId, IrOperationType},
    transform::IrTransform,
};

#[derive(Clone, Debug)]
pub struct AddOperation(pub Vec<IrNodeId>, pub Result<Rc<dyn IrOperationType>, IRTrace>);

impl IrTransform for AddOperation {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        let output = ir.graph.add_op_dyn(&self.0, self.1.clone()?)?;
        ir.most_recent = output;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoveOperation(pub IrOperationId);

impl IrTransform for RemoveOperation {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        ir.graph.remove_op(self.0).map_err(IRTrace::Root)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapOutputs(pub IrNodeId, pub IrNodeId);

impl IrTransform for SwapOutputs {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        ir.graph.swap_outputs_unchecked(self.0, self.1)?;
        ir.check_valid()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplaceInput {
    pub old: IrNodeId,
    pub new: IrNodeId,
}

impl IrTransform for ReplaceInput {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        ir.graph.replace_input_unchecked(self.new, self.old)?;
        ir.check_valid()
    }
}
