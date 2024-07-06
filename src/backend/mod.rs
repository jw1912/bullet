#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "hip")]
mod hip;

#[cfg(feature = "hip")]
pub use hip::*;

#[cfg(not(any(feature = "cuda", feature = "hip")))]
mod cpu;

#[cfg(not(any(feature = "cuda", feature = "hip")))]
pub use cpu::*;
