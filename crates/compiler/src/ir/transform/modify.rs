use std::{fmt, rc::Rc};

use crate::ir::{
    IR, IRTrace,
    graph::{IrNodeId, IrOperationId, IrOperationType},
    transform::IrTransform,
};

#[derive(Clone)]
pub struct AddOperation(pub Vec<IrNodeId>, pub Result<Rc<dyn IrOperationType>, IRTrace>);

impl fmt::Debug for AddOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let frame = self.1.as_ref().map(|frame| format!("{frame:?}")).unwrap_or_else(|err| {
            let mut s = String::new();
            err.frame(&mut s).unwrap();
            s
        });
        write!(f, "AddOperation({:?}, {frame})", self.0)
    }
}

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

#[derive(Clone, Debug)]
pub struct ReplaceOperation(pub IrOperationId, pub AddOperation);

impl IrTransform for ReplaceOperation {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        ir.transform(self.1.clone())?;
        let new_outputs = ir.most_recent.clone();
        let old = ir.get_op_mut(self.0)?;

        for (new, old) in new_outputs.into_iter().zip(old.outputs().to_vec()) {
            ir.swap_outputs(new, old)?;
        }

        ir.remove_op(self.0)
    }
}
