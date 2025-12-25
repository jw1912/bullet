use std::rc::Rc;

use crate::{
    common::{DTypeTensor, Unary},
    ir::graph::{IrOperation, IrOperationType, IrType},
};

#[derive(Debug, PartialEq)]
pub struct IrUnary {
    ty: IrType,
    op: Unary,
}

impl IrUnary {
    pub fn new(ty: IrType, op: Unary) -> Self {
        Self { ty, op }
    }

    pub fn op(&self) -> Unary {
        self.op
    }
}

impl IrOperationType for IrUnary {
    fn opname(&self) -> String {
        format!("unary.{:?}", self.op).to_lowercase()
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![self.ty]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.ty.size(), self.op.dtype(self.ty.dtype()))]
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, self.op.evaluate(inputs[0].read(idx)).unwrap());
        }
    }

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool {
        if let Some(other) = IrOperation::downcast::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::{DType, Size};

    use super::*;

    #[test]
    fn evaluate() {
        let ty = IrType::new(Size::variable(), DType::F32);

        let binary = IrUnary::new(ty, Unary::Cos);

        let a = DTypeTensor::F32(vec![0.0; 4]);
        let mut b = DTypeTensor::F32(vec![0.0; 4]);

        binary.evaluate(&[&a], &mut [&mut b]);

        assert_eq!(b, DTypeTensor::F32(vec![1.0; 4]));
    }
}
