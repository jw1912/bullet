use crate::tensor::{
    DValue, IRTrace, TNode, TType,
    operation::{
        Unary,
        autograd::{CustomAutograd, CustomAutogradOp},
    },
};

#[derive(Debug, PartialEq)]
pub struct FauxQuantise(pub TType, pub DValue, pub bool);

impl CustomAutograd for FauxQuantise {
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

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        if inputs.len() != 1 {
            return Err("Invalid number of inputs!".into());
        }

        if output_grads.len() != 1 {
            return Err("Invalid number of output grads!".into());
        }

        Ok(output_grads)
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        if let Some(other) = other.downcast() { self == other } else { false }
    }
}
