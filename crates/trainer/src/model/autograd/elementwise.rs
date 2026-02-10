use bullet_compiler::ir::{
    frontend::{DValue, IRNode, IRTrace},
    operation::{CABinary, CABinaryOp, Unary, UnaryOp},
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
            CABinary::Max => {
                let diff = (inputs[0] - inputs[1])?;
                let lgrad = diff.unary(Unary::IsPositive)?;
                let rgrad = (-diff)?.unary(Unary::IsPositive)?;
                ((grad * lgrad)?, (grad * rgrad)?)
            },
            CABinary::Min => {
                let diff = (inputs[0] - inputs[1])?;
                let lgrad = (-diff)?.unary(Unary::IsPositive)?;
                let rgrad = diff.unary(Unary::IsPositive)?;
                ((grad * lgrad)?, (grad * rgrad)?)
            },
            _ => unimplemented!(),
        };

        Ok(vec![Some(glhs), Some(grhs)])
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
            Unary::Sgn | Unary::IsPositive | Unary::IsZero => {
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
