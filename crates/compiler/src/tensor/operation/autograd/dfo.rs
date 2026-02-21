use std::{fmt, rc::Rc};

use crate::tensor::{DType, DValue, IRNode, IRTrace, Size, TType, operation::Unary};

use super::{Autograd, AutogradOp};

pub trait DiffableFromOutput: fmt::Debug + PartialEq + 'static {
    fn forward<'a>(&self, input: IRNode<'a>) -> Result<IRNode<'a>, IRTrace>;

    fn backward<'a>(&self, output: IRNode<'a>) -> Result<IRNode<'a>, IRTrace>;
}

#[derive(Debug, PartialEq)]
pub struct DiffableFromOutputOp<T: DiffableFromOutput>(pub T, pub DType, pub Size);

impl<T: DiffableFromOutput> Autograd for DiffableFromOutputOp<T> {
    fn opname(&self) -> String {
        format!("diffable-from-output.{:?}", self.0).to_lowercase()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.2, self.1)]
    }

    fn forward<'a>(&self, inputs: Vec<IRNode<'a>>) -> Result<Vec<IRNode<'a>>, IRTrace> {
        let [input] = inputs[..] else { return Err("Invalid number of inputs!".into()) };
        self.0.forward(input).map(|x| vec![x])
    }

    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
        let [input] = inputs[..] else { return Err("Invalid number of inputs!".into()) };
        let [grad] = output_grads[..] else { return Err("Invalid number of output grads!".into()) };

        let output = self.0.forward(input)?;

        match self.0.backward(output) {
            Ok(x) => Ok(vec![Some((grad * x)?)]),
            Err(e) => Err(e),
        }
    }

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool {
        if let Some(other) = AutogradOp::downcast_rc(other) { self == other } else { false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReLU;
impl DiffableFromOutput for ReLU {
    fn forward<'a>(&self, input: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        let zero = DValue::zero(input.ty().dtype());
        let zero = input.builder().scalar(zero, input.ty().size());
        input.max(zero)
    }

    fn backward<'a>(&self, output: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        output.unary(Unary::IsPositive)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CReLU;
impl DiffableFromOutput for CReLU {
    fn forward<'a>(&self, input: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        let one = DValue::one(input.ty().dtype());
        let one = input.builder().scalar(one, input.ty().size());
        ReLU.forward(input)?.min(one)
    }

    fn backward<'a>(&self, output: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        let one = DValue::one(output.ty().dtype());
        let one = output.builder().scalar(one, output.ty().size());
        ReLU.backward(output)? * (one - output)?.unary(Unary::IsPositive)?
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SCReLU;
impl DiffableFromOutput for SCReLU {
    fn forward<'a>(&self, input: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        let crelu = CReLU.forward(input)?;
        crelu * crelu
    }

    fn backward<'a>(&self, output: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        let half = (CReLU.backward(output)? * output.sqrt()?)?;
        half + half
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sigmoid;
impl DiffableFromOutput for Sigmoid {
    fn forward<'a>(&self, input: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        let one = DValue::one(input.ty().dtype());
        let one = input.builder().scalar(one, input.ty().size());
        (one + (-input)?.exp()?)?.unary(Unary::Reciprocal)
    }

    fn backward<'a>(&self, output: IRNode<'a>) -> Result<IRNode<'a>, IRTrace> {
        let one = DValue::one(output.ty().dtype());
        let one = output.builder().scalar(one, output.ty().size());
        output * (one - output)?
    }
}
