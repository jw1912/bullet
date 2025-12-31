use std::{collections::HashSet, rc::Rc};

use crate::{
    core::{CABinary, DTypeTensor},
    ir::graph::{IrError, IrOperation, IrOperationType, IrType},
};

#[derive(Debug, PartialEq)]
pub struct CABinaryOp {
    ty: IrType,
    op: CABinary,
}

impl CABinaryOp {
    pub fn new(ty: IrType, op: CABinary) -> Result<Self, IrError> {
        Ok(Self { ty, op })
    }

    pub fn ty(&self) -> IrType {
        self.ty
    }

    pub fn op(&self) -> CABinary {
        self.op
    }
}

impl IrOperationType for CABinaryOp {
    fn opname(&self) -> String {
        format!("binary.{:?}", self.op).to_lowercase()
    }

    fn inputs(&self) -> Vec<IrType> {
        vec![self.ty, self.ty]
    }

    fn outputs(&self) -> Vec<IrType> {
        vec![self.ty]
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
        if let Some(other) = IrOperation::downcast_rc::<Self>(other) { self == other } else { false }
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        vec![[0, 1].into()]
    }
}

#[cfg(test)]
mod tests {
    use crate::core::{DType, Size};

    use super::*;

    #[test]
    fn evaluate() {
        let ty = IrType::new(Size::variable(), DType::F32);

        let binary = CABinaryOp::new(ty, CABinary::Add).unwrap();

        let a = DTypeTensor::F32(vec![2.0; 4]);
        let b = DTypeTensor::F32(vec![1.0; 4]);
        let mut c = DTypeTensor::F32(vec![0.0; 4]);

        binary.evaluate(&[&a, &b], &mut [&mut c]);

        assert_eq!(c, DTypeTensor::F32(vec![3.0; 4]));
    }
}
