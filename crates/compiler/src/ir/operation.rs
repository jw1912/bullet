mod binary;
mod broadcast;
mod elementwise;
mod reduce;
mod unary;

use std::{
    collections::HashSet,
    fmt::Debug,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

pub use binary::IrBinary;
pub use broadcast::BroadcastAcrossDimension;
pub use elementwise::IrElementwise;
pub use reduce::{ReduceAcrossDimension, Reduction};
pub use unary::IrUnary;

use crate::{
    common::DTypeTensor,
    ir::{IrError, IrNodeId, IrType, node::IrNode},
};

pub trait IrOperationType: std::any::Any + Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<IrType>;

    fn outputs(&self) -> Vec<IrType>;

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]);

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool;

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
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

        Self::check(&inputs, &outputs, op.as_ref())?;

        let inputs = inputs.iter().map(|&i| i.id()).collect();
        let outputs = outputs.iter().map(|&i| i.id()).collect();

        Ok(Self { id, op, inputs, outputs })
    }

    pub fn check<'a>(
        inputs: impl AsRef<[&'a IrNode]>,
        outputs: impl AsRef<[&'a IrNode]>,
        op: &'a dyn IrOperationType,
    ) -> Result<(), IrError> {
        if op.inputs() != inputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("IrOperation::new: inputs don't match expected!".into());
        }

        if op.outputs() != outputs.as_ref().iter().map(|&i| i.ty()).collect::<Vec<_>>() {
            return Err("IrOperation::new: outputs don't match expected!".into());
        }

        Ok(())
    }

    pub fn id(&self) -> IrOperationId {
        self.id
    }

    pub fn inputs(&self) -> &[IrNodeId] {
        &self.inputs
    }

    pub fn set_input(&mut self, idx: usize, node: IrNodeId) {
        self.inputs[idx] = node;
    }

    pub fn outputs(&self) -> &[IrNodeId] {
        &self.outputs
    }

    pub fn swap_output_with(&mut self, new: IrNodeId, old: IrNodeId) -> Result<(), IrError> {
        let mut found = false;

        for id in &mut self.outputs {
            if *id == old {
                if found {
                    panic!("This cannot happen!");
                }

                *id = new;
                found = true;
            }
        }

        found.then_some(()).ok_or(format!("IrOperation::swap_output_with: {old:?} not found!").into())
    }

    pub fn op(&self) -> &Rc<dyn IrOperationType> {
        &self.op
    }

    pub fn downcast<T: IrOperationType + 'static>(input: &Rc<dyn IrOperationType>) -> Option<&T> {
        let op: &dyn std::any::Any = input.as_ref();
        op.downcast_ref::<T>()
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

#[derive(Clone, Debug, PartialEq)]
pub struct Constant(pub DTypeTensor);

impl Constant {
    pub fn ty(&self) -> IrType {
        IrType::new(self.0.size(), self.0.dtype())
    }
}

impl IrOperationType for Constant {
    fn opname(&self) -> String {
        format!("constant<{:?}>", self.ty())
    }

    fn inputs(&self) -> Vec<IrType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.ty()]
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        assert_eq!(inputs.len(), 0);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].size(), self.0.size());
        assert_eq!(outputs[0].dtype(), self.0.dtype());

        *outputs[0] = self.0.clone();
    }

    fn equals(&self, _: &Rc<dyn IrOperationType>) -> bool {
        false
    }
}
