use std::rc::Rc;

use crate::{
    core::{DTypeTensor, Unary},
    ir::graph::{IrError, IrOperation, IrOperationType, IrType},
};

#[derive(Debug, PartialEq)]
pub struct UnaryOp {
    ty: IrType,
    op: Unary,
}

impl UnaryOp {
    pub fn new(ty: IrType, op: Unary) -> Result<Self, IrError> {
        if op.dtype(ty.dtype()).is_none() {
            return Err("UnaryOp failed type check!".into());
        }

        Ok(Self { ty, op })
    }

    pub fn input_type(&self) -> IrType {
        self.ty
    }

    pub fn output_type(&self) -> IrType {
        IrType::new(self.ty.size(), self.op.dtype(self.ty.dtype()).unwrap())
    }

    pub fn op(&self) -> Unary {
        self.op
    }
}

impl IrOperationType for UnaryOp {
    fn opname(&self) -> String {
        format!("unary.{:?}", self.op).to_lowercase()
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![self.input_type()]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.output_type()]
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
        if let Some(other) = IrOperation::downcast_rc::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{DType, Size};

    use super::*;

    #[test]
    fn evaluate() {
        let ty = IrType::new(Size::variable(), DType::F32);

        let binary = UnaryOp::new(ty, Unary::Cos).unwrap();

        let a = DTypeTensor::F32(vec![0.0; 4]);
        let mut b = DTypeTensor::F32(vec![0.0; 4]);

        binary.evaluate(&[&a], &mut [&mut b]);

        assert_eq!(b, DTypeTensor::F32(vec![1.0; 4]));
    }
}
