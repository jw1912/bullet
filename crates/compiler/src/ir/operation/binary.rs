use std::{collections::HashSet, rc::Rc};

use crate::ir::graph::{DValue, Op, OpType, TType, TValue};

/// Commutative & associative binary operations.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CABinary {
    Add,
    Mul,
    Min,
    Max,
}

impl CABinary {
    pub fn evaluate(self, lhs: DValue, rhs: DValue) -> Option<DValue> {
        match (lhs, rhs) {
            (DValue::F32(x), DValue::F32(y)) => Some(DValue::F32(self.evaluate_f32(x, y))),
            (DValue::I32(x), DValue::I32(y)) => self.evaluate_i32(x, y).map(DValue::I32),
            _ => None,
        }
    }

    pub fn evaluate_f32(self, lhs: f32, rhs: f32) -> f32 {
        match self {
            Self::Add => lhs + rhs,
            Self::Mul => lhs * rhs,
            Self::Min => lhs.min(rhs),
            Self::Max => lhs.max(rhs),
        }
    }

    pub fn evaluate_i32(self, lhs: i32, rhs: i32) -> Option<i32> {
        Some(match self {
            Self::Add => lhs + rhs,
            Self::Mul => lhs * rhs,
            Self::Min => lhs.min(rhs),
            Self::Max => lhs.max(rhs),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CABinaryOp {
    ty: TType,
    op: CABinary,
}

impl CABinaryOp {
    pub fn new(ty: TType, op: CABinary) -> Self {
        Self { ty, op }
    }

    pub fn ty(&self) -> TType {
        self.ty
    }

    pub fn op(&self) -> CABinary {
        self.op
    }
}

impl OpType for CABinaryOp {
    fn opname(&self) -> String {
        format!("binary.{:?}", self.op).to_lowercase()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.ty, self.ty]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.ty]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, inputs[1].size());
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, self.op.evaluate(inputs[0].read(idx), inputs[1].read(idx)).unwrap());
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        vec![[0, 1].into()]
    }
}

/// Commutative & associative binary operations.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NCABinary {
    Pow,
}

impl NCABinary {
    pub fn evaluate(self, lhs: DValue, rhs: DValue) -> Option<DValue> {
        match (lhs, rhs) {
            (DValue::F32(x), DValue::F32(y)) => Some(DValue::F32(self.evaluate_f32(x, y))),
            (DValue::I32(x), DValue::I32(y)) => self.evaluate_i32(x, y).map(DValue::I32),
            _ => None,
        }
    }

    pub fn evaluate_f32(self, lhs: f32, rhs: f32) -> f32 {
        match self {
            Self::Pow => lhs.powf(rhs),
        }
    }

    pub fn evaluate_i32(self, lhs: i32, rhs: i32) -> Option<i32> {
        Some(match self {
            Self::Pow => lhs.pow(rhs as u32),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NCABinaryOp {
    ty: TType,
    op: NCABinary,
}

impl NCABinaryOp {
    pub fn new(ty: TType, op: NCABinary) -> Self {
        Self { ty, op }
    }

    pub fn ty(&self) -> TType {
        self.ty
    }

    pub fn op(&self) -> NCABinary {
        self.op
    }
}

impl OpType for NCABinaryOp {
    fn opname(&self) -> String {
        format!("binary.{:?}", self.op).to_lowercase()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.ty, self.ty]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.ty]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, inputs[1].size());
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, self.op.evaluate(inputs[0].read(idx), inputs[1].read(idx)).unwrap());
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        vec![[0, 1].into()]
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::graph::{DType, Size};

    use super::*;

    #[test]
    fn evaluate_ca() {
        let ty = TType::new(Size::variable(), DType::F32);

        let binary = CABinaryOp::new(ty, CABinary::Add);

        let a = TValue::F32(vec![2.0; 4]);
        let b = TValue::F32(vec![1.0; 4]);
        let mut c = TValue::F32(vec![0.0; 4]);

        binary.evaluate(vec![&a, &b], vec![&mut c]);

        assert_eq!(c, TValue::F32(vec![3.0; 4]));
    }

    #[test]
    fn evaluate_nca() {
        let ty = TType::new(Size::variable(), DType::F32);

        let binary = NCABinaryOp::new(ty, NCABinary::Pow);

        let a = TValue::F32(vec![2.0; 4]);
        let b = TValue::F32(vec![3.0; 4]);
        let mut c = TValue::F32(vec![0.0; 4]);

        binary.evaluate(vec![&a, &b], vec![&mut c]);

        assert_eq!(c, TValue::F32(vec![8.0; 4]));
    }
}
