pub mod ir;
pub mod tensor;

pub mod frontend {
    pub use crate::tensor::{
        DType, DValue, IRTrace, Shape, Size, TType, TValue, TensorIR,
        builder::{IRBuilder, IRNode},
    };
}
