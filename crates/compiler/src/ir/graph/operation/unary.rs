use std::rc::Rc;

use crate::{
    core::{DTypeTensor, Unary},
    ir::graph::{IrError, IrOperation, IrOperationType, IrType},
};

#[derive(Debug, PartialEq)]
pub struct IrUnary {
    ty: IrType,
    op: Unary,
}

impl IrUnary {
    pub fn new(ty: IrType, mut op: Unary) -> Result<Self, IrError> {
        if op.dtype(ty.dtype()).is_none() {
            return Err("Failed type check!".into());
        }

        if let Unary::BinaryWithConst { op, lhs, .. } = &mut op {
            if op.is_commutative() {
                *lhs = true;
            }
        }

        Ok(Self { ty, op })
    }

    pub fn op(&self) -> Unary {
        self.op
    }
}

impl IrOperationType for IrUnary {
    fn opname(&self) -> String {
        if let Unary::BinaryWithConst { op, val, lhs } = self.op {
            format!("binary.{op:?}.withconst<{val:?}, {lhs}>").to_lowercase()
        } else {
            format!("unary.{:?}", self.op).to_lowercase()
        }
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![self.ty]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.ty.size(), self.op.dtype(self.ty.dtype()).unwrap())]
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

        let binary = IrUnary::new(ty, Unary::Cos).unwrap();

        let a = DTypeTensor::F32(vec![0.0; 4]);
        let mut b = DTypeTensor::F32(vec![0.0; 4]);

        binary.evaluate(&[&a], &mut [&mut b]);

        assert_eq!(b, DTypeTensor::F32(vec![1.0; 4]));
    }
}
