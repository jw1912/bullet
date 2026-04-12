use std::rc::Rc;

use crate::tensor::{
    DValue, IRTrace, TNode, TType,
    operation::{
        CABinary, CABinaryOp, Power, Unary, UnaryOp,
        autograd::{Autograd, AutogradOp},
    },
};

use super::AutogradOnCoreOp;

impl AutogradOnCoreOp for CABinaryOp {
    fn backward<'a>(
        &self,
        inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let [lhs, rhs] = inputs[..] else { panic!() };
        let [grad] = output_grads[..] else { panic!() };

        let (glhs, grhs) = match self.op() {
            CABinary::Add => (grad, grad),
            CABinary::Mul => ((grad * rhs)?, (grad * lhs)?),
            CABinary::Max => {
                let diff = (lhs - rhs)?;
                let lgrad = diff.unary(Unary::IsPositive)?;
                let rgrad = (-diff)?.unary(Unary::IsPositive)?;
                ((grad * lgrad)?, (grad * rgrad)?)
            }
            CABinary::Min => {
                let diff = (lhs - rhs)?;
                let lgrad = (-diff)?.unary(Unary::IsPositive)?;
                let rgrad = diff.unary(Unary::IsPositive)?;
                ((grad * lgrad)?, (grad * rgrad)?)
            }
        };

        Ok(vec![Some(glhs), Some(grhs)])
    }
}

impl AutogradOnCoreOp for UnaryOp {
    fn backward<'a>(
        &self,
        inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
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
            Unary::Sgn | Unary::IsPositive | Unary::IsZero | Unary::IsNonNegative => {
                let zero = DValue::zero(input.ty().dtype());
                Ok(input.builder().scalar(zero, input.ty().size()))
            }
            Unary::Cast(_) => Ok(grad),
            Unary::Sinh | Unary::Cosh | Unary::Tanh | Unary::Tan | Unary::Truncate | Unary::Round | Unary::Sqrt => {
                unimplemented!()
            }
        }?;

        if let Unary::Cast(_) = self.op() {
            grad.unary(Unary::Cast(self.input_type().dtype()))
        } else {
            grad * g
        }
        .map(|x| vec![Some(x)])
    }
}

impl AutogradOnCoreOp for Power {
    fn backward<'a>(
        &self,
        inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let grad = output_grads[0];
        let base_grad = (inputs[1] * inputs[0].pow((inputs[1] - 1.0)?)?)?;
        let powf_grad = (inputs[0].log()? * inputs[0].pow(inputs[1])?)?;
        Ok(vec![Some((grad * base_grad)?), Some((grad * powf_grad)?)])
    }
}

#[derive(Debug, PartialEq)]
pub struct FauxQuantise(pub TType, pub DValue, pub bool);

impl Autograd for FauxQuantise {
    fn opname(&self) -> String {
        format!("diffable-from-output.{:?}", self.0).to_lowercase()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.0.size(), self.0.dtype())]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let [input] = inputs[..] else { return Err("Invalid number of inputs!".into()) };

        let op = if self.2 { Unary::Round } else { Unary::Truncate };
        let scalar = input.builder().scalar(self.1, input.ty().size());
        ((scalar * input)?.unary(op)? * scalar.unary(Unary::Reciprocal)?).map(|x| vec![x])
    }

    fn backward<'a>(
        &self,
        inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        if inputs.len() != 1 {
            return Err("Invalid number of inputs!".into());
        }

        if output_grads.len() != 1 {
            return Err("Invalid number of output grads!".into());
        }

        Ok(vec![Some(output_grads[0])])
    }

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool {
        if let Some(other) = AutogradOp::downcast_rc(other) { self == other } else { false }
    }
}
