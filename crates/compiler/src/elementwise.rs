pub(crate) mod builder;
pub(crate) mod description;
pub(crate) mod kernel;

use crate::common::{DType, DTypeValue};

pub use builder::{ElementwiseBuilder, ElementwiseNode};
pub use description::{ElementwiseDescription, ElementwiseId, Operation};
pub use kernel::{ElementwiseKernel, ElementwiseKernelBuilder, ElementwiseMut, ElementwiseRef};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Log1pAbs,
    Sgn,
    Abs,
    Cast(DType),
}

impl Unary {
    pub fn dtype(self, input: DType) -> DType {
        if let Self::Cast(ty) = self { ty } else { input }
    }

    pub fn evaluate(self, input: DTypeValue) -> Option<DTypeValue> {
        let fp = |f: fn(f32) -> f32| input.f32().map(|x| DTypeValue::F32(f(x)));

        Some(match self {
            Self::Sin => fp(f32::sin)?,
            Self::Cos => fp(f32::cos)?,
            Self::Tan => fp(f32::tan)?,
            Self::Sinh => fp(f32::sinh)?,
            Self::Cosh => fp(f32::cosh)?,
            Self::Tanh => fp(f32::tanh)?,
            Self::Exp => fp(f32::exp)?,
            Self::Log1pAbs => fp(|x| x.abs().ln_1p())?,
            Self::Sgn => match input {
                DTypeValue::F32(x) => DTypeValue::F32(x.signum()),
                DTypeValue::I32(x) => DTypeValue::I32(x.signum()),
            },
            Self::Abs => match input {
                DTypeValue::F32(x) => DTypeValue::F32(x.abs()),
                DTypeValue::I32(x) => DTypeValue::I32(x.abs()),
            },
            Self::Cast(ty) => match (input, ty) {
                (DTypeValue::F32(x), DType::I32) => DTypeValue::I32(x as i32),
                (DTypeValue::I32(x), DType::F32) => DTypeValue::F32(x as f32),
                _ => input,
            },
        })
    }
}

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
}
