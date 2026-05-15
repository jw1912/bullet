mod dfo;
mod passthrough;
mod qat;
mod softmax;

use std::{fmt, rc::Rc};

use crate::tensor::{IRBuilder, IRTrace, OpType, TNode, TType, TValue, TensorOp, operation::SubGraph};

pub use dfo::{CReLU, DiffableFromOutput, DiffableFromOutputOp, ReLU, SCReLU, Sigmoid, SqrReLU};
pub use passthrough::PassThrough;
pub use qat::FauxQuantise;
pub use softmax::SoftmaxCrossEntropyLoss;

pub trait CustomAutograd: std::any::Any + fmt::Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<TType>;

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace>;

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace>;

    fn equals(&self, other: &CustomAutogradOp) -> bool;
}

#[derive(Clone, Debug)]
pub struct CustomAutogradOp {
    pub(crate) op: Rc<dyn CustomAutograd>,
    pub(crate) forward: SubGraph,
}

impl CustomAutogradOp {
    pub fn into_forward(self) -> SubGraph {
        self.forward
    }

    pub fn downcast_rc<T: CustomAutograd>(input: &Rc<dyn CustomAutograd>) -> Option<&T> {
        let op: &dyn std::any::Any = input.as_ref();
        op.downcast_ref::<T>()
    }

    pub fn downcast<T: CustomAutograd>(&self) -> Option<&T> {
        Self::downcast_rc::<T>(&self.op)
    }

    pub fn new(op: impl CustomAutograd + 'static) -> Result<Self, IRTrace> {
        let op_inputs = op.inputs();

        let builder = IRBuilder::default();
        let inputs = op_inputs.iter().map(|i| builder.add_input(i.size(), i.dtype())).collect::<Vec<_>>();
        let outputs = op.forward(inputs.clone())?;
        let forward = builder.build(&outputs);
        let inputs = inputs.iter().map(TNode::node).collect();
        let outputs = outputs.iter().map(TNode::node).collect();
        let forward = SubGraph::new(forward, inputs, outputs)?;

        Ok(Self { op: Rc::new(op), forward })
    }
}

impl OpType for CustomAutogradOp {
    fn opname(&self) -> String {
        format!("autograd.{}", self.op.opname())
    }

    fn inputs(&self) -> Vec<TType> {
        self.op.inputs()
    }

    fn outputs(&self) -> Vec<TType> {
        self.forward.outputs()
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(op) = other.downcast::<CustomAutogradOp>() { self.op.equals(op) } else { false }
    }

    fn evaluate(&self, inputs: Vec<&TValue>, outputs: Vec<&mut TValue>) -> bool {
        self.forward.evaluate(inputs, outputs)
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        self.op.backward(inputs, output_grads)
    }
}
