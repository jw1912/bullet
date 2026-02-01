use bullet_compiler::{
    graph::DValue,
    operation::{BroadcastAcrossDimension, CABinary, CABinaryOp, ReduceAcrossDimension, Unary, UnaryOp},
    prelude::IRNode,
};

use super::AutogradOnCoreOp;

impl AutogradOnCoreOp for CABinaryOp {
    fn backward<'a>(&self, inputs: Vec<IRNode<'a>>, output_grads: Vec<IRNode<'a>>) -> Vec<Option<IRNode<'a>>> {
        let [grad] = output_grads[..] else { panic!() };

        let (glhs, grhs) = match self.op() {
            CABinary::Add => (grad, grad),
            CABinary::Mul => (grad * inputs[1], grad * inputs[0]),
            _ => unimplemented!(),
        };

        vec![Some(glhs), Some(grhs)]
    }
}

impl AutogradOnCoreOp for BroadcastAcrossDimension {
    fn backward<'a>(&self, _inputs: Vec<IRNode<'a>>, output_grads: Vec<IRNode<'a>>) -> Vec<Option<IRNode<'a>>> {
        let op = self.invert().unwrap();
        vec![Some(output_grads[0].builder().add_op([output_grads[0]], op)[0])]
    }
}

impl AutogradOnCoreOp for ReduceAcrossDimension {
    fn backward<'a>(&self, _inputs: Vec<IRNode<'a>>, output_grads: Vec<IRNode<'a>>) -> Vec<Option<IRNode<'a>>> {
        let op = self.invert().unwrap().expect("Reduction backprop only implemented for Sum!");
        vec![Some(output_grads[0].builder().add_op([output_grads[0]], op)[0])]
    }
}

impl AutogradOnCoreOp for UnaryOp {
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
