#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(not(feature = "cuda"))]
mod cpu;

#[cfg(not(feature = "cuda"))]
pub use cpu::*;
