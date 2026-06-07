use std::fmt;

use crate::tensor::{
    IRTrace, TNode, TType,
    operation::autograd::{CustomAutograd, CustomAutogradOp},
};

#[allow(clippy::type_complexity)]
pub struct PassThrough(pub TType, pub Box<dyn for<'a> Fn(TNode<'a>) -> Result<TNode<'a>, IRTrace>>);

impl fmt::Debug for PassThrough {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PassThrough")
    }
}

impl CustomAutograd for PassThrough {
    fn opname(&self) -> String {
        "pass-through".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        self.1(inputs[0]).map(|x| vec![x])
    }

    fn backward<'a>(&self, _inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        Ok(output_grads)
    }

    fn equals(&self, _other: &CustomAutogradOp) -> bool {
        false
    }
}
