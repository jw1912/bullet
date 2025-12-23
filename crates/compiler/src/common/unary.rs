use crate::common::{DType, DTypeValue};

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
