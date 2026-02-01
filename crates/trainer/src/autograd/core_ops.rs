use bullet_compiler::{
    IRTrace,
    graph::DValue,
    operation::{BroadcastAcrossDimension, CABinary, CABinaryOp, ReduceAcrossDimension, Unary, UnaryOp},
    prelude::IRNode,
};

use super::AutogradOnCoreOp;

impl AutogradOnCoreOp for CABinaryOp {
    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let [grad] = output_grads[..] else { panic!() };

        let (glhs, grhs) = match self.op() {
            CABinary::Add => (grad, grad),
            CABinary::Mul => ((grad * inputs[1])?, (grad * inputs[0])?),
            _ => unimplemented!(),
        };

        Ok(vec![Some(glhs), Some(grhs)])
    }
}

impl AutogradOnCoreOp for BroadcastAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap();
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}

impl AutogradOnCoreOp for ReduceAcrossDimension {
    fn backward<'a>(
        &self,
        _inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let op = self.invert().unwrap().expect("Reduction backprop only implemented for Sum!");
        output_grads[0].builder().add_op([output_grads[0]], op).map(|x| vec![Some(x[0])])
    }
}

impl AutogradOnCoreOp for UnaryOp {
    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let grad = output_grads[0];
        let input = inputs[0];

        let g = match self.op() {
            Unary::Abs => input.sgn(),
            Unary::Cos => -input.sin()?,
            Unary::Sin => input.cos(),
            Unary::Exp => input.exp(),
            Unary::Reciprocal => {
                let x = input.unary(Unary::Reciprocal)?;
                -(x * x)?
            }
            Unary::Log => input.unary(Unary::Reciprocal),
            Unary::Sgn => {
                let zero = DValue::zero(input.ty().dtype());
                Ok(input.builder().scalar(zero, input.ty().size()))
            }
            Unary::Cast(_) => Ok(grad),
            _ => unimplemented!(),
        }?;

        if let Unary::Cast(_) = self.op() {
            grad.unary(Unary::Cast(self.input_type().dtype()))
        } else {
            grad * g
        }
        .map(|x| vec![Some(x)])
    }
}
