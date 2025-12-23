//mod broadcast;
mod elementwise;
mod reduce;

use std::{
    fmt::Debug,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

//pub use broadcast::Broadcast;
pub use elementwise::IrElementwise;
pub use reduce::{ReduceAcrossDimension, Reduction};

use crate::{
    common::DTypeTensor,
    ir::{IrError, IrNodeId, IrType, node::IrNode},
};

pub trait IrOperationType: Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<IrType>;

    fn outputs(&self) -> Vec<IrType>;

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]);

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool;
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct IrOperationId(usize);

impl Default for IrOperationId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl IrOperationId {
    pub(super) fn from_inner(id: usize) -> Self {
        Self(id)
    }

    pub fn inner(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct IrOperation {
    id: IrOperationId,
    inputs: Vec<IrNodeId>,
    outputs: Vec<IrNodeId>,
    op: Rc<dyn IrOperationType>,
}

impl IrOperation {
    pub fn new(inputs: Vec<&IrNode>, outputs: Vec<&IrNode>, op: impl IrOperationType) -> Result<Self, IrError> {
        let id = IrOperationId::default();
        let op = Rc::new(op);

        if op.inputs() != inputs.iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err(IrError::InvalidOperationInputs);
        }

        if op.outputs() != outputs.iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err(IrError::InvalidOperationOutputs);
        }

        let inputs = inputs.iter().map(|&i| i.id()).collect();
        let outputs = outputs.iter().map(|&i| i.id()).collect();

        Ok(Self { id, op, inputs, outputs })
    }

    pub fn id(&self) -> IrOperationId {
        self.id
    }

    pub fn inputs(&self) -> &[IrNodeId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[IrNodeId] {
        &self.outputs
    }

    pub fn swap_input_with(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IrError> {
        let mut found = false;

        for id in &mut self.inputs {
            if *id == old {
                *id = new;
                found = true;
            }
        }

        found.then_some(()).ok_or(IrError::NodeDoesNotExist)
    }

    pub fn swap_output_with(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IrError> {
        let mut found = false;

        for id in &mut self.outputs {
            if *id == old {
                if found {
                    return Err(IrError::InvalidOperationOutputs);
                }

                *id = new;
                found = true;
            }
        }

        found.then_some(()).ok_or(IrError::NodeDoesNotExist)
    }

    pub fn op(&self) -> &Rc<dyn IrOperationType> {
        &self.op
    }

    pub fn downcast<T: IrOperationType + 'static>(input: &Rc<dyn IrOperationType>) -> Option<&T> {
        let op: &dyn std::any::Any = input;
        op.downcast_ref().cloned()
    }
}

#[derive(Debug)]
pub struct Leaf(pub IrType);

impl IrOperationType for Leaf {
    fn opname(&self) -> String {
        format!("leaf<{:?}>", self.0)
    }

    fn inputs(&self) -> Vec<IrType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.0]
    }

    fn evaluate(&self, _: &[&DTypeTensor], _: &mut [&mut DTypeTensor]) {}

    fn equals(&self, _: &Rc<dyn IrOperationType>) -> bool {
        false
    }
}
