use std::rc::Rc;

use bullet_compiler::{
    graph::{OpType, TType},
    operation::{CABinary, CABinaryOp},
    prelude::ProgramNode,
};

use crate::autograd::{Autograd, AutogradOp};

impl Autograd for CABinaryOp {
    fn opname(&self) -> String {
        <Self as OpType>::opname(self)
    }

    fn inputs(&self) -> Vec<TType> {
        <Self as OpType>::inputs(self)
    }

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool {
        if let Some(other) = AutogradOp::downcast_rc(other) { self == other } else { false }
    }

    fn forward<'a>(&self, inputs: &[ProgramNode<'a>]) -> Vec<ProgramNode<'a>> {
        vec![inputs[0].binary(inputs[1], self.op())]
    }

    fn backward<'a>(
        &self,
        inputs: &[ProgramNode<'a>],
        output_grads: &[ProgramNode<'a>],
    ) -> Vec<Option<ProgramNode<'a>>> {
        let [grad] = output_grads[..] else { panic!() };

        let (glhs, grhs) = match self.op() {
            CABinary::Add => (grad, grad),
            CABinary::Mul => (grad * inputs[1], grad * inputs[0]),
            _ => unimplemented!(),
        };

        vec![Some(glhs), Some(grhs)]
    }
}
