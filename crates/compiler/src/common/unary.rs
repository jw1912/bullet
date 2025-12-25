use crate::common::{Binary, DType, DTypeValue};

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
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
    BinaryWithConst { op: Binary, val: DTypeValue, lhs: bool },
}

impl Unary {
    pub fn dtype(self, input: DType) -> Option<DType> {
        match self {
            Self::Cast(ty) => Some(ty),
            Self::BinaryWithConst { op, val, lhs } => {
                let val = val.dtype();
                let (a, b) = if lhs { (input, val) } else { (val, input) };
                op.dtype(a, b)
            }
            Self::Sgn | Self::Abs => Some(input),
            _ => (input != DType::I32).then_some(input),
        }
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
            Self::BinaryWithConst { op, val, lhs } => {
                let (a, b) = if lhs { (input, val) } else { (val, input) };
                op.evaluate(a, b)?
            }
        })
    }
}
