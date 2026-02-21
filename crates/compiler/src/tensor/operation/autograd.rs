mod broadcast;
mod dfo;
mod linear;
mod pointwise;
mod qat;
mod reduce;

use std::{fmt, rc::Rc};

use crate::tensor::{IRBuilder, IRNode, IRTrace, OpType, TType, TValue, TensorOp, operation::SubGraph};

pub use dfo::{CReLU, DiffableFromOutput, DiffableFromOutputOp, ReLU, SCReLU, Sigmoid};
pub use qat::FauxQuantise;

pub trait Autograd: std::any::Any + fmt::Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<TType>;

    fn forward<'a>(&self, inputs: Vec<IRNode<'a>>) -> Result<Vec<IRNode<'a>>, IRTrace>;

    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace>;

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool;
}

pub trait AutogradOnCoreOp: Clone + OpType + PartialEq {
    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace>;
}

impl<T: AutogradOnCoreOp> Autograd for T {
    fn opname(&self) -> String {
        <T as OpType>::opname(self)
    }

    fn inputs(&self) -> Vec<TType> {
        <T as OpType>::inputs(self)
    }

    fn forward<'a>(&self, inputs: Vec<IRNode<'a>>) -> Result<Vec<IRNode<'a>>, IRTrace> {
        inputs[0].builder().add_op(inputs, self.clone())
    }

    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        <T as AutogradOnCoreOp>::backward(self, inputs, output_grads)
    }

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool {
        if let Some(other) = AutogradOp::downcast_rc(other) { self == other } else { false }
    }
}

#[derive(Clone, Debug)]
pub struct AutogradOp {
    pub(crate) op: Rc<dyn Autograd>,
    pub(crate) forward: SubGraph,
}

impl AutogradOp {
    pub fn into_forward(self) -> SubGraph {
        self.forward
    }

    pub fn downcast_rc<T: Autograd>(input: &Rc<dyn Autograd>) -> Option<&T> {
        let op: &dyn std::any::Any = input.as_ref();
        op.downcast_ref::<T>()
    }

    pub fn downcast<T: Autograd>(&self) -> Option<&T> {
        Self::downcast_rc::<T>(&self.op)
    }

    pub fn new(op: impl Autograd + 'static) -> Result<Self, IRTrace> {
        let op_inputs = op.inputs();

        let builder = IRBuilder::default();
        let inputs = op_inputs.iter().map(|i| builder.add_input(i.size(), i.dtype())).collect::<Vec<_>>();
        let outputs = op.forward(inputs.clone())?;
        let forward = builder.build(&outputs);
        let inputs = inputs.iter().map(IRNode::node).collect();
        let outputs = outputs.iter().map(IRNode::node).collect();
        let forward = SubGraph::new(forward, inputs, outputs)?;

        Ok(Self { op: Rc::new(op), forward })
    }
}

impl OpType for AutogradOp {
    fn opname(&self) -> String {
        format!("autograd.{}", self.op.opname())
    }

    fn inputs(&self) -> Vec<TType> {
        self.op.inputs()
    }

    fn outputs(&self) -> Vec<TType> {
        self.forward.outputs()
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(AutogradOp { op, .. }) = TensorOp::downcast_rc(other) {
            self.op.equals(op)
        } else {
            false
        }
    }

    fn evaluate(&self, inputs: Vec<&TValue>, outputs: Vec<&mut TValue>) -> bool {
        self.forward.evaluate(inputs, outputs)
    }
}
