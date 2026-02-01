use std::rc::Rc;

use bullet_compiler::{
    graph::{DValue, OpType, TType},
    operation::{Unary, UnaryOp},
    prelude::IRNode,
};

use crate::autograd::{Autograd, AutogradOp};

impl Autograd for UnaryOp {
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
        vec![inputs[0].unary(self.op())]
    }

    fn backward<'a>(&self, inputs: Vec<IRNode<'a>>, output_grads: Vec<IRNode<'a>>) -> Vec<Option<IRNode<'a>>> {
        let grad = output_grads[0];
        let input = inputs[0];

        let g = match self.op() {
            Unary::Abs => input.sgn(),
            Unary::Cos => -input.sin(),
            Unary::Sin => input.cos(),
            Unary::Exp => input.exp(),
            Unary::Reciprocal => {
                let x = input.unary(Unary::Reciprocal);
                -(x * x)
            }
            Unary::Log => input.unary(Unary::Reciprocal),
            Unary::Sgn => {
                let zero = DValue::zero(input.ty().dtype());
                input.builder().scalar(zero, input.ty().size())
            }
            Unary::Cast(_) => grad,
            _ => unimplemented!(),
        };

        if let Unary::Cast(_) = self.op() {
            vec![Some(grad.unary(Unary::Cast(self.input_type().dtype())))]
        } else {
            vec![Some(grad * g)]
        }
    }
}
