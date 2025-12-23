pub mod common;
pub mod elementwise;
pub mod frontend;
pub mod ir;

pub mod prelude {
    pub use crate::{
        common::{DType, DTypeTensor, Shape, Size},
        frontend::{ProgramBuilder, ProgramNode},
    };
}
