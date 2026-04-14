use std::{fmt, rc::Rc};

use crate::tensor::{IRTrace, TNode, TType, operation::autograd::Autograd};

#[allow(clippy::type_complexity)]
pub struct PassThrough(pub TType, pub Box<dyn for<'a> Fn(TNode<'a>) -> Result<TNode<'a>, IRTrace>>);

impl fmt::Debug for PassThrough {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PassThrough")
    }
}

impl Autograd for PassThrough {
    fn opname(&self) -> String {
        "pass-through".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        self.1(inputs[0]).map(|x| vec![x])
    }

    fn backward<'a>(
        &self,
        _inputs: Vec<TNode<'a>>,
        output_grads: Vec<TNode<'a>>,
    ) -> Result<Vec<Option<TNode<'a>>>, IRTrace> {
        Ok(vec![Some(output_grads[0])])
    }

    fn equals(&self, _other: &Rc<dyn Autograd>) -> bool {
        false
    }
}
