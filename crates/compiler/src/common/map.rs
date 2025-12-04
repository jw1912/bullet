use crate::common::DTypeValue;

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapOp<T> {
    Unary { inp: T, op: UnaryOp },
    Binary { lhs: T, rhs: T, op: BinaryOp },
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
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
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Min,
    Max,
    AbsPow,
}

impl<T: Copy> MapOp<T> {
    pub fn mapname(&self) -> String {
        format!(
            "map.{}",
            match *self {
                MapOp::Unary { op, .. } => format!("{op:?}"),
                MapOp::Binary { op, .. } => format!("{op:?}"),
            }
            .to_lowercase()
        )
    }

    pub fn args(&self) -> Vec<T> {
        match *self {
            MapOp::Binary { lhs, rhs, .. } => vec![lhs, rhs],
            MapOp::Unary { inp, .. } => vec![inp],
        }
    }

    pub fn to<U>(&self, f: impl Fn(&T) -> U) -> MapOp<U> {
        match self {
            Self::Unary { inp, op } => MapOp::Unary { inp: f(inp), op: *op },
            Self::Binary { lhs, rhs, op } => MapOp::Binary { lhs: f(lhs), rhs: f(rhs), op: *op },
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Self::Unary { .. } => 1,
            Self::Binary { .. } => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MapNode<T> {
    Value(T),
    Constant(DTypeValue),
}
