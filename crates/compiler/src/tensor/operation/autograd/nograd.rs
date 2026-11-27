use crate::tensor::{
    DValue, IRTrace, TNode, TType,
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
        let ty = inputs[0].ty();
        Ok(vec![inputs[0].builder().scalar(DValue::zero(ty.dtype().grad()), ty.size())])
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        if let Some(other) = other.downcast() { self == other } else { false }
    }
}
