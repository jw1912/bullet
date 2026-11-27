use crate::tensor::{
    DValue, IRTrace, TNode, TType,
    operation::{
        Unary,
        autograd::{CustomAutograd, CustomAutogradOp},
    },
};

#[derive(Debug, PartialEq)]
pub struct FauxQuantise {
    pub ty: TType,
    pub scale: DValue,
    pub zero_point: DValue,
    pub round: bool,
}

impl FauxQuantise {
    pub fn new(ty: TType, scale: DValue, zero_point: DValue, round: bool) -> Self {
        Self { ty, scale, zero_point, round }
    }
}

impl CustomAutograd for FauxQuantise {
    fn opname(&self) -> String {
        "faux-quantise".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.ty]
    }

    fn forward<'a>(&self, inputs: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let [input] = inputs[..] else { return Err("Invalid number of inputs!".into()) };

        let builder = input.builder();
        let size = input.ty().size();
        let scale = builder.scalar(self.scale, size);
        let zp = builder.scalar(self.zero_point, size);
        let op = if self.round { Unary::Round } else { Unary::Truncate };

        let centered = (input - zp)?;
        let quantized = ((scale * centered)?.unary(op)? * scale.unary(Unary::Reciprocal)?)?;
        let out = (quantized + zp)?;

        Ok(vec![out])
    }

    fn backward<'a>(&self, _inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        if output_grads.len() != 1 {
            return Err("Invalid number of output grads!".into());
        }

        Ok(output_grads)
    }

    fn equals(&self, other: &CustomAutogradOp) -> bool {
        if let Some(other) = other.downcast::<FauxQuantise>() { self == other } else { false }
    }
}
