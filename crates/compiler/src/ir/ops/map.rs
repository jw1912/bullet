use std::fmt::Debug;

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
    Log,
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
    pub fn opname(&self) -> String {
        format!(
            "map.{}",
            match *self {
                MapOp::Unary { op, .. } => format!("{op:?}"),
                MapOp::Binary { op, .. } => format!("{op:?}"),
            }
            .to_lowercase()
        )
    }

    pub fn inputs(&self) -> Vec<T> {
        match *self {
            MapOp::Binary { lhs, rhs, .. } => vec![lhs, rhs],
            MapOp::Unary { inp, .. } => vec![inp],
        }
    }
}
