use std::rc::Rc;

use bullet_compiler::{
    graph::{OpType, TType},
    operation::BroadcastAcrossDimension,
    prelude::IRNode,
};

use crate::autograd::{Autograd, AutogradOp};

impl Autograd for BroadcastAcrossDimension {
    fn opname(&self) -> String {
        <Self as OpType>::opname(self)
    }

    fn inputs(&self) -> Vec<TType> {
        <Self as OpType>::inputs(self)
    }

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool {
        if let Some(other) = AutogradOp::downcast_rc(other) { self == other } else { false }
    }

    fn forward<'a>(&self, inputs: Vec<IRNode<'a>>) -> Vec<IRNode<'a>> {
        vec![inputs[0].builder().add_op([inputs[0]], *self)[0]]
    }

    fn backward<'a>(&self, _inputs: Vec<IRNode<'a>>, output_grads: Vec<IRNode<'a>>) -> Vec<Option<IRNode<'a>>> {
        let op = self.invert().unwrap();
        vec![Some(output_grads[0].builder().add_op([output_grads[0]], op)[0])]
    }
}
