use std::{collections::HashSet, rc::Rc};

use crate::{
    common::{Binary, DType, DTypeTensor, Size},
    ir::{
        IrError,
        node::IrType,
        operation::{IrOperation, IrOperationType},
    },
};

#[derive(Debug, PartialEq)]
pub struct IrBinary {
    size: Size,
    lhs: DType,
    rhs: DType,
    op: Binary,
}

impl IrBinary {
    pub fn new(lhs: IrType, rhs: IrType, op: Binary) -> Result<Self, IrError> {
        let size = lhs.size();

        if size != rhs.size() {
            return Err(format!("IrBinary::new: mismatched sizes {size:?} != {:?}!", rhs.size()).into());
        }

        let lhs = lhs.dtype();
        let rhs = rhs.dtype();

        if op.dtype(lhs, rhs).is_none() {
            return Err(format!("IrBinary::new: invalid dtype combo {op:?}({lhs:?}, {rhs:?})!").into());
        }

        Ok(Self { size, lhs, rhs, op })
    }

    pub fn op(&self) -> Binary {
        self.op
    }
}

impl IrOperationType for IrBinary {
    fn opname(&self) -> String {
        format!("binary.{:?}", self.op).to_lowercase()
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.size, self.lhs), IrType::new(self.size, self.rhs)]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![IrType::new(self.size, self.op.dtype(self.lhs, self.rhs).unwrap())]
    }

    fn evaluate(&self, inputs: &[&DTypeTensor], outputs: &mut [&mut DTypeTensor]) {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, inputs[1].size());
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, self.op.evaluate(inputs[0].read(idx), inputs[1].read(idx)).unwrap());
        }
    }

    fn equals(&self, other: &Rc<dyn IrOperationType>) -> bool {
        if let Some(other) = IrOperation::downcast::<Self>(other) { self == other } else { false }
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        match self.op {
            Binary::AbsPow | Binary::Div | Binary::Sub => Vec::new(),
            Binary::Add | Binary::Max | Binary::Min | Binary::Mul => vec![[0, 1].into()],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::DType;

    use super::*;

    #[test]
    fn evaluate() {
        let ty = IrType::new(Size::variable(), DType::F32);

        let binary = IrBinary::new(ty, ty, Binary::Add).unwrap();

        let a = DTypeTensor::F32(vec![2.0; 4]);
        let b = DTypeTensor::F32(vec![1.0; 4]);
        let mut c = DTypeTensor::F32(vec![0.0; 4]);

        binary.evaluate(&[&a, &b], &mut [&mut c]);

        assert_eq!(c, DTypeTensor::F32(vec![3.0; 4]));
    }
}
