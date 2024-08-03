#[cfg(not(any(feature = "cuda", feature = "metal", feature = "hip")))]
pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "hip")]
pub use hip::*;
#[cfg(feature = "metal")]
pub use metal::*;

pub mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "hip")]
mod hip;

#[cfg(feature = "metal")]
mod metal;
