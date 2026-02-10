use std::rc::Rc;

use crate::ir::graph::{DType, DValue, GraphError, Op, OpType, TType, TValue};

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
    Log,
    Sgn,
    Abs,
    Reciprocal,
    Cast(DType),
    IsPositive,
    IsZero,
}

impl Unary {
    pub fn dtype(self, input: DType) -> Option<DType> {
        match self {
            Self::Cast(ty) => Some(ty),
            Self::Sgn | Self::Abs => Some(input),
            _ => (input != DType::I32).then_some(input),
        }
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
            Self::Sgn => match input {
                DValue::F32(x) => DValue::F32(x.signum()),
                DValue::I32(x) => DValue::I32(x.signum()),
            },
            Self::Abs => match input {
                DValue::F32(x) => DValue::F32(x.abs()),
                DValue::I32(x) => DValue::I32(x.abs()),
            },
            Self::Cast(ty) => match (input, ty) {
                (DValue::F32(x), DType::I32) => DValue::I32(x as i32),
                (DValue::I32(x), DType::F32) => DValue::F32(x as f32),
                _ => input,
            },
            Self::IsPositive => match input {
                DValue::F32(x) => DValue::F32(f32::from(x > 0.0)),
                DValue::I32(x) => DValue::I32(i32::from(x > 0)),
            },
            Self::IsZero => match input {
                DValue::F32(x) => DValue::F32(f32::from(x == 0.0)),
                DValue::I32(x) => DValue::I32(i32::from(x == 0)),
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
    pub fn new(ty: TType, op: Unary) -> Result<Self, GraphError> {
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

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let size = inputs[0].size();
        assert_eq!(size, outputs[0].size());

        for idx in 0..size {
            outputs[0].write(idx, self.op.evaluate(inputs[0].read(idx)).unwrap());
        }
    }

    fn equals(&self, other: &Rc<dyn OpType>) -> bool {
        if let Some(other) = Op::downcast_rc::<Self>(other) { self == other } else { false }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::graph::{DType, Size};

    use super::*;

    #[test]
    fn evaluate() {
        let ty = TType::new(Size::variable(), DType::F32);

        let binary = UnaryOp::new(ty, Unary::Cos).unwrap();

        let a = TValue::F32(vec![0.0; 4]);
        let mut b = TValue::F32(vec![0.0; 4]);

        binary.evaluate(vec![&a], vec![&mut b]);

        assert_eq!(b, TValue::F32(vec![1.0; 4]));
    }
}
