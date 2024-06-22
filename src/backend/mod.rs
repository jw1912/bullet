#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "metal")]
pub use metal::*;

pub mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "metal")]
mod metal;
