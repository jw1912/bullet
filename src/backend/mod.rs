#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "hip")]
mod hip;

#[cfg(feature = "hip")]
pub use hip::*;

#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "metal")]
pub use metal::*;

// feature = "metal" depends on the cpu module.
#[cfg(not(any(feature = "cuda", feature = "hip")))]
pub mod cpu;

#[cfg(not(any(feature = "cuda", feature = "metal", feature = "hip")))]
pub use cpu::*;
