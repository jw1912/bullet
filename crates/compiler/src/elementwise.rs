pub(crate) mod builder;
pub(crate) mod description;
pub(crate) mod kernel;

use crate::DType;

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
}
