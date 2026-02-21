use std::rc::Rc;

use crate::tensor::{
    DValue, IRNode, IRTrace, TType,
    operation::{
        Unary,
        autograd::{Autograd, AutogradOp},
    },
};

#[derive(Debug, PartialEq)]
pub struct FauxQuantise(pub TType, pub DValue, pub bool);

impl Autograd for FauxQuantise {
    fn opname(&self) -> String {
        format!("diffable-from-output.{:?}", self.0).to_lowercase()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.0.size(), self.0.dtype())]
    }

    fn forward<'a>(&self, inputs: Vec<IRNode<'a>>) -> Result<Vec<IRNode<'a>>, IRTrace> {
        let [input] = inputs[..] else { return Err("Invalid number of inputs!".into()) };

        let op = if self.2 { Unary::Round } else { Unary::Truncate };
        let scalar = input.builder().scalar(self.1, input.ty().size());
        ((scalar * input)?.unary(op)? * scalar.unary(Unary::Reciprocal)?).map(|x| vec![x])
    }

    fn backward<'a>(
        &self,
        inputs: Vec<IRNode<'a>>,
        output_grads: Vec<IRNode<'a>>,
    ) -> Result<Vec<Option<IRNode<'a>>>, IRTrace> {
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
