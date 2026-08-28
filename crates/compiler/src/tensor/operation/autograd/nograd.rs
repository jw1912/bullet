use crate::tensor::{
    IRTrace, TNode, TType,
    operation::autograd::{CustomAutograd, CustomAutogradOp},
};

#[derive(Debug, PartialEq)]
pub struct CopyNoGrad(pub TType);

impl CustomAutograd for CopyNoGrad {
    fn opname(&self) -> String {
        "copy-no-grad".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![TType::new(self.0.size(), self.0.dtype())]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        Ok(vec![inputs[0].copy()?])
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, _: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        Ok(vec![inputs[0].zeros_like()])
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        if let Some(other) = other.downcast() { self == other } else { false }
    }
}
