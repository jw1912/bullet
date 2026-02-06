use std::{cell::RefCell, fmt, rc::Rc};

use crate::{
    IR, IRTrace,
    graph::{NodeId, OpId, OpType},
    transform::IRTransform,
};

#[derive(Clone)]
pub struct AddOperation {
    inputs: Vec<NodeId>,
    operation: Result<Rc<dyn OpType>, IRTrace>,
    outputs: Rc<RefCell<Vec<NodeId>>>,
}

impl AddOperation {
    pub fn new(inputs: impl Into<Vec<NodeId>>, operation: Result<Rc<dyn OpType>, IRTrace>) -> Self {
        Self { inputs: inputs.into(), operation, outputs: Rc::default() }
    }

    pub fn outputs(&self) -> Vec<NodeId> {
        self.outputs.borrow().clone()
    }
}

impl fmt::Debug for AddOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let frame = self.operation.as_ref().map(|frame| format!("{frame:?}")).unwrap_or_else(|err| {
            let mut s = String::new();
            err.frame(&mut s).unwrap();
            s
        });
        write!(f, "AddOperation({:?}, {frame})", self.inputs)
    }
}

impl IRTransform for AddOperation {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        *self.outputs.borrow_mut() = ir.graph.add_op_dyn(&self.inputs, self.operation.clone()?)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoveOperation(pub OpId);

impl IRTransform for RemoveOperation {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        ir.graph.remove_op(self.0).map_err(IRTrace::Root)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapOutputs(pub NodeId, pub NodeId);

impl IRTransform for SwapOutputs {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        if self.0 == self.1 {
            return Ok(());
        }

        ir.graph.swap_outputs_unchecked(self.0, self.1)?;
        ir.check_valid()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplaceInput {
    pub old: NodeId,
    pub new: NodeId,
}

impl IRTransform for ReplaceInput {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        if self.new == self.old {
            return Ok(());
        }

        ir.graph.replace_input_unchecked(self.new, self.old)?;
        ir.check_valid()
    }
}

#[derive(Clone, Debug)]
pub struct ReplaceOperation(pub OpId, pub AddOperation);

impl IRTransform for ReplaceOperation {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        ir.transform(self.1.clone())?;
        let new_outputs = self.1.outputs();
        let old = ir.get_op_mut(self.0)?;

        for (new, old) in new_outputs.into_iter().zip(old.outputs().to_vec()) {
            ir.swap_outputs(new, old)?;
        }

        ir.remove_op(self.0)
    }
}
