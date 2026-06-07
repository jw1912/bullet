use crate::tensor::{
    DType, IRTrace, Size, TNode, TType,
    operation::autograd::{CustomAutograd, CustomAutogradOp},
};

#[derive(Debug, PartialEq)]
pub struct SoftmaxCrossEntropyLoss {
    pub batch_size: Size,
    pub axis_size: usize,
}

impl SoftmaxCrossEntropyLoss {
    pub fn ttype(&self) -> TType {
        TType::new(self.batch_size * self.axis_size, DType::F32)
    }
}

impl CustomAutograd for SoftmaxCrossEntropyLoss {
    fn opname(&self) -> String {
        "softmax-cross-entropy-loss".into()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.ttype(), self.ttype()]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let [input, target] = inputs[..] else { return Err("Invalid number of inputs!".into()) };
        (-(target * input.softmax(self.axis_size)?.log()?)?).map(|x| vec![x])
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let [input, target] = inputs[..] else { return Err("Invalid number of inputs!".into()) };
        let [grad] = output_grads[..] else { return Err("Invalid number of output grads!".into()) };
        let igrad = ((input.softmax(self.axis_size)? - target)? * grad)?;
        Ok(vec![igrad, target.zeros_like()])
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        if let Some(other) = other.downcast() { self == other } else { false }
    }
}
