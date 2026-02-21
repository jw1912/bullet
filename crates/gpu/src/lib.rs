//! Crate for compiling and executing tensor DAGs from `bullet-compiler` on CUDA/ROCm devices.

pub mod buffer;
pub mod function;
pub mod kernel;
pub mod runtime;
