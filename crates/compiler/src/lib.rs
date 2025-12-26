pub mod core;
pub mod frontend;
pub mod ir;
pub mod utils;

pub mod prelude {
    pub use crate::{
        core::{DType, DTypeTensor, Shape, Size},
        frontend::{ProgramBuilder, ProgramNode},
    };
}
