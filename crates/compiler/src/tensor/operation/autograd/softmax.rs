use std::rc::Rc;

use crate::tensor::{
    DType, IRTrace, Size, TNode, TType,
    operation::autograd::{Autograd, AutogradOp},
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

impl Autograd for SoftmaxCrossEntropyLoss {
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

    fn backward<'a>(
        &self,
        inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        let [input, target] = inputs[..] else { return Err("Invalid number of inputs!".into()) };
        let [grad] = output_grads[..] else { return Err("Invalid number of output grads!".into()) };
        let igrad = ((input.softmax(self.axis_size)? - target)? * grad)?;
        Ok(vec![Some(igrad), None])
    }

    fn equals(&self, other: &Rc<dyn Autograd>) -> bool {
        if let Some(other) = AutogradOp::downcast_rc(other) { self == other } else { false }
    }
}
