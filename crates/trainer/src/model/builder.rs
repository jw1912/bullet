use bullet_compiler::ir::{
    frontend::{IRBuilder, IRNode},
    graph::{NodeId, TType},
    operation::{Unary, UnaryOp},
};

use super::autograd::{Autograd, AutogradOp};

#[derive(Default)]
pub struct ModelBuilder {
    ir: IRBuilder,
}

impl ModelBuilder {
    pub fn add_op<'a>(&'a self, inputs: impl AsRef<[ModelNode<'a>]>, op: impl Autograd) -> Vec<ModelNode<'a>> {
        let inputs = inputs.as_ref().iter().map(ModelNode::detach).collect::<Vec<_>>();
        let op = self.ir.add_op(inputs, AutogradOp::new(op).unwrap()).unwrap();
        op.iter().map(|&node| ModelNode { builder: self, node: node.node() }).collect()
    }
}

#[derive(Clone, Copy)]
pub struct ModelNode<'a> {
    builder: &'a ModelBuilder,
    node: NodeId,
}

impl<'a> ModelNode<'a> {
    pub fn detach(&self) -> IRNode<'a> {
        IRNode::new(&self.builder.ir, self.node)
    }

    pub fn ty(&self) -> TType {
        self.detach().ty()
    }

    pub fn abs(&self) -> Self {
        self.builder.add_op([*self], UnaryOp::new(self.ty(), Unary::Abs).unwrap())[0]
    }

    pub fn sin(&self) -> Self {
        self.builder.add_op([*self], UnaryOp::new(self.ty(), Unary::Sin).unwrap())[0]
    }

    pub fn cos(&self) -> Self {
        self.builder.add_op([*self], UnaryOp::new(self.ty(), Unary::Cos).unwrap())[0]
    }

    pub fn tan(&self) -> Self {
        self.builder.add_op([*self], UnaryOp::new(self.ty(), Unary::Tan).unwrap())[0]
    }

    pub fn sinh(&self) -> Self {
        self.builder.add_op([*self], UnaryOp::new(self.ty(), Unary::Sinh).unwrap())[0]
    }

    pub fn cosh(&self) -> Self {
        self.builder.add_op([*self], UnaryOp::new(self.ty(), Unary::Cosh).unwrap())[0]
    }

    pub fn tanh(&self) -> Self {
        self.builder.add_op([*self], UnaryOp::new(self.ty(), Unary::Tanh).unwrap())[0]
    }
}
