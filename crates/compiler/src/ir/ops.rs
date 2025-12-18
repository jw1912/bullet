mod broadcast;
mod elementwise;
mod reduce;

use std::{
    fmt::Debug,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

pub use broadcast::Broadcast;
pub use elementwise::IrElementwise;
pub use reduce::{Reduce, ReduceOp};

use crate::ir::{IrError, IrGraph, IrNodeId, IrType};

pub trait IrOperation: Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<IrNodeId>;

    fn output_types(&self, ir: &IrGraph) -> Result<Vec<IrType>, IrError>;
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct IrOpId(usize);

impl Default for IrOpId {
    fn default() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl IrOpId {
    pub(super) fn from_inner(id: usize) -> Self {
        Self(id)
    }

    pub fn inner(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct IrOp {
    id: IrOpId,
    inputs: Vec<IrNodeId>,
    outputs: Vec<IrNodeId>,
    op: Rc<dyn IrOperation>,
}

impl IrOp {
    pub fn new(op: impl IrOperation, ir: &IrGraph) -> Result<Self, IrError> {
        let id = IrOpId::default();
        let op = Rc::new(op);
        let inputs = op.inputs();
        let output_tys = op.output_types(ir)?;
        let outputs = (0..output_tys.len()).map(|_| IrNodeId::default()).collect();

        Ok(Self { id, op, outputs, inputs })
    }

    pub fn id(&self) -> IrOpId {
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

    pub fn op(&self) -> &Rc<dyn IrOperation> {
        &self.op
    }
}

#[derive(Debug)]
pub struct Leaf(pub IrType);

impl IrOperation for Leaf {
    fn opname(&self) -> String {
        format!("leaf<{:?}>", self.0)
    }

    fn inputs(&self) -> Vec<IrNodeId> {
        Vec::new()
    }

    fn output_types(&self, _ir: &IrGraph) -> Result<Vec<IrType>, IrError> {
        Ok(vec![self.0])
    }
}
