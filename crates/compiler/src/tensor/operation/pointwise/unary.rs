use crate::{
    ir::IRError,
    tensor::{DType, DValue, IRTrace, OpType, TNode, TType, TValue, TensorOp, bf16, f16},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unary {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Log,
    Sgn,
    Abs,
    Sqrt,
    Reciprocal,
    Cast(DType),
    IsPositive,
    IsZero,
    IsNonNegative,
    Round,
    Truncate,
    Identity,
}

impl Unary {
    pub fn dtype(self, input: DType) -> Option<DType> {
        let is_half = |ty| matches!(ty, DType::F16 | DType::Bf16);

        match self {
            // half precision types only support conversion to/from f32
            Self::Cast(ty) if is_half(input) || is_half(ty) => {
                (input == ty || input == DType::F32 || ty == DType::F32).then_some(ty)
            }
            Self::Cast(ty) => Some(ty),
            Self::Identity => Some(input),
            _ if is_half(input) => None,
            Self::Sgn | Self::Abs | Self::IsNonNegative | Self::IsPositive | Self::IsZero => Some(input),
            _ => (input != DType::I32).then_some(input),
        }
    }

    /// Whether this is a cast between integer and half precision types,
    /// which is not supported
    pub fn is_integer_half_cast(self, input: DType) -> bool {
        let is_half = |ty| matches!(ty, DType::F16 | DType::Bf16);
        matches!(self, Self::Cast(ty) if (input == DType::I32 && is_half(ty)) || (is_half(input) && ty == DType::I32))
    }

    pub fn evaluate(self, input: DValue) -> Option<DValue> {
        let fp = |f: fn(f32) -> f32| input.f32().map(|x| DValue::F32(f(x)));

        Some(match self {
            Self::Sin => fp(f32::sin)?,
            Self::Cos => fp(f32::cos)?,
            Self::Tan => fp(f32::tan)?,
            Self::Sinh => fp(f32::sinh)?,
            Self::Cosh => fp(f32::cosh)?,
            Self::Tanh => fp(f32::tanh)?,
            Self::Exp => fp(f32::exp)?,
            Self::Reciprocal => fp(|x| 1.0 / x)?,
            Self::Log => fp(|x| x.ln())?,
            Self::Sqrt => fp(|x| x.sqrt())?,
            Self::Round => fp(|x| x.round())?,
            Self::Truncate => fp(|x| x.trunc())?,
            Self::Identity => input,
            Self::Sgn => match input {
                DValue::F32(x) => DValue::F32(x.signum()),
                DValue::I32(x) => DValue::I32(x.signum()),
                DValue::F16(_) | DValue::Bf16(_) => return None,
            },
            Self::Abs => match input {
                DValue::F32(x) => DValue::F32(x.abs()),
                DValue::I32(x) => DValue::I32(x.abs()),
                DValue::F16(_) | DValue::Bf16(_) => return None,
            },
            Self::Cast(ty) => match (input, ty) {
                (DValue::F32(x), DType::I32) => DValue::I32(x as i32),
                (DValue::I32(x), DType::F32) => DValue::F32(x as f32),
                (DValue::F32(x), DType::F16) => DValue::F16(f16::from_f32(x)),
                (DValue::F16(x), DType::F32) => DValue::F32(x.to_f32()),
                (DValue::F32(x), DType::Bf16) => DValue::Bf16(bf16::from_f32(x)),
                (DValue::Bf16(x), DType::F32) => DValue::F32(x.to_f32()),
                (x, ty) if x.dtype() == ty => input,
                _ => return None,
            },
            Self::IsPositive => match input {
                DValue::F32(x) => DValue::F32(f32::from(x > 0.0)),
                DValue::I32(x) => DValue::I32(i32::from(x > 0)),
                DValue::F16(_) | DValue::Bf16(_) => return None,
            },
            Self::IsZero => match input {
                DValue::F32(x) => DValue::F32(f32::from(x == 0.0)),
                DValue::I32(x) => DValue::I32(i32::from(x == 0)),
                DValue::F16(_) | DValue::Bf16(_) => return None,
            },
            Self::IsNonNegative => match input {
                DValue::F32(x) => DValue::F32(f32::from(x >= 0.0)),
                DValue::I32(x) => DValue::I32(i32::from(x >= 0)),
                DValue::F16(_) | DValue::Bf16(_) => return None,
            },
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnaryOp {
    ty: TType,
    op: Unary,
}

impl UnaryOp {
    pub fn new(ty: TType, op: Unary) -> Result<Self, IRError> {
        if op.is_integer_half_cast(ty.dtype()) {
            return Err("Cannot cast between integer and half precision types!".into());
        }

        if op.dtype(ty.dtype()).is_none() {
            return Err("UnaryOp failed type check!".into());
        }

        Ok(Self { ty, op })
    }

    pub fn input_type(&self) -> TType {
        self.ty
    }

    pub fn output_type(&self) -> TType {
        TType::new(self.ty.size(), self.op.dtype(self.ty.dtype()).unwrap())
    }

    pub fn op(&self) -> Unary {
        self.op
    }
}

impl OpType for UnaryOp {
    fn opname(&self) -> String {
        format!("unary.{:?}", self.op).to_lowercase()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.input_type()]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.output_type()]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, self.op.evaluate(inputs[0].read(idx)).unwrap());
        }

        true
    }

    fn equals(&self, other: &TensorOp) -> bool {
        if let Some(other) = other.downcast::<Self>() { self == other } else { false }
    }

    fn backward<'a>(&self, inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        let grad = output_grads[0];
        let input = inputs[0];

        let g = match self.op() {
            Unary::Abs => input.sgn(),
            Unary::Cos => -input.sin()?,
            Unary::Sin => input.cos(),
            Unary::Exp => input.exp(),
            Unary::Reciprocal => {
                let x = input.unary(Unary::Reciprocal)?;
                -(x * x)?
            }
            Unary::Log => input.unary(Unary::Reciprocal),
            Unary::Sinh => input.cosh(),
            Unary::Cosh => input.sinh(),
            Unary::Tanh => {
                let t = input.tanh()?;
                1.0 - (t * t)?
            }
            Unary::Tan => {
                let t = input.tan()?;
                1.0 + (t * t)?
            }
            Unary::Sqrt => 0.5 * input.unary(Unary::Sqrt)?.unary(Unary::Reciprocal)?,
            Unary::Sgn | Unary::IsPositive | Unary::IsZero | Unary::IsNonNegative | Unary::Round | Unary::Truncate => {
                let zero = DValue::zero(input.ty().dtype());
                Ok(input.builder().scalar(zero, input.ty().size()))
            }
            Unary::Cast(_) | Unary::Identity => Ok(grad),
        }?;

        if let Unary::Cast(_) = self.op() {
            grad.unary(Unary::Cast(self.input_type().dtype()))
        } else if let Unary::Identity = self.op() {
            Ok(grad)
        } else {
            grad * g
        }
        .map(|x| vec![x])
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::DType;

    use super::*;

    #[test]
    fn evaluate() {
        let ty = TType::new(4, DType::F32);

        let binary = UnaryOp::new(ty, Unary::Cos).unwrap();

        let a = TValue::F32(vec![0.0; 4]);
        let mut b = TValue::F32(vec![0.0; 4]);

        binary.evaluate(vec![&a], vec![&mut b]);

        assert_eq!(b, TValue::F32(vec![1.0; 4]));
    }

    #[test]
    fn half_casts() {
        for half in [DType::F16, DType::Bf16] {
            assert_eq!(Unary::Cast(half).dtype(DType::F32), Some(half));
            assert_eq!(Unary::Cast(DType::F32).dtype(half), Some(DType::F32));
            assert_eq!(Unary::Cast(half).dtype(DType::I32), None);
            assert_eq!(Unary::Cast(DType::I32).dtype(half), None);
            assert_eq!(Unary::Exp.dtype(half), None);
            assert_eq!(Unary::Abs.dtype(half), None);
        }

        assert_eq!(Unary::Cast(DType::Bf16).dtype(DType::F16), None);

        for half in [DType::F16, DType::Bf16] {
            assert!(UnaryOp::new(TType::new(4, DType::I32), Unary::Cast(half)).is_err());
            assert!(UnaryOp::new(TType::new(4, half), Unary::Cast(DType::I32)).is_err());
            assert_eq!(Unary::Cast(half).evaluate(DValue::I32(1)), None);
            assert_eq!(Unary::Cast(DType::I32).evaluate(DValue::zero(half)), None);
        }

        let a = TValue::F32(vec![1.0, -2.5, 1.0e-3, 65520.0]);
        let mut h = TValue::zeros(DType::F16, 4);
        let mut b = TValue::zeros(DType::Bf16, 4);

        UnaryOp::new(TType::new(4, DType::F32), Unary::Cast(DType::F16)).unwrap().evaluate(vec![&a], vec![&mut h]);
        UnaryOp::new(TType::new(4, DType::F32), Unary::Cast(DType::Bf16)).unwrap().evaluate(vec![&a], vec![&mut b]);

        let TValue::F32(a) = a else { panic!() };
        assert_eq!(h, TValue::F16(a.iter().map(|&x| f16::from_f32(x)).collect()));
        assert_eq!(b, TValue::Bf16(a.iter().map(|&x| bf16::from_f32(x)).collect()));

        let mut back = TValue::zeros(DType::F32, 4);
        UnaryOp::new(TType::new(4, DType::F16), Unary::Cast(DType::F32)).unwrap().evaluate(vec![&h], vec![&mut back]);
        assert_eq!(back, TValue::F32(vec![1.0, -2.5, f16::from_f32(1.0e-3).to_f32(), f32::INFINITY]));
    }
}
