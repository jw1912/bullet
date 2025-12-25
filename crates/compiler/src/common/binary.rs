use crate::common::{DType, DTypeValue};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Mul,
    Sub,
    Div,
    Min,
    Max,
    AbsPow,
}

impl Binary {
    pub fn dtype(&self, lhs: DType, rhs: DType) -> Option<DType> {
        (lhs == rhs).then_some(lhs)
    }

    pub fn evaluate(self, lhs: DTypeValue, rhs: DTypeValue) -> Option<DTypeValue> {
        match (lhs, rhs) {
            (DTypeValue::F32(x), DTypeValue::F32(y)) => Some(DTypeValue::F32(self.evaluate_f32(x, y))),
            (DTypeValue::I32(x), DTypeValue::I32(y)) => self.evaluate_i32(x, y).map(DTypeValue::I32),
            _ => None,
        }
    }

    pub fn evaluate_f32(self, lhs: f32, rhs: f32) -> f32 {
        match self {
            Self::Add => lhs + rhs,
            Self::Mul => lhs * rhs,
            Self::AbsPow => lhs.abs().powf(rhs),
            Self::Div => lhs / rhs,
            Self::Sub => lhs - rhs,
            Self::Min => lhs.min(rhs),
            Self::Max => lhs.max(rhs),
        }
    }

    pub fn evaluate_i32(self, lhs: i32, rhs: i32) -> Option<i32> {
        Some(match self {
            Self::Add => lhs + rhs,
            Self::Mul => lhs * rhs,
            Self::AbsPow => rhs.try_into().ok().map(|r| lhs.abs().pow(r))?,
            Self::Div => lhs / rhs,
            Self::Sub => lhs - rhs,
            Self::Min => lhs.min(rhs),
            Self::Max => lhs.max(rhs),
        })
    }

    pub fn is_commutative(self) -> bool {
        match self {
            Binary::AbsPow | Binary::Div | Binary::Sub => false,
            Binary::Add | Binary::Max | Binary::Min | Binary::Mul => true,
        }
    }
}
