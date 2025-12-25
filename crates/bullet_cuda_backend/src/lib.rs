#[cfg(feature = "build")]
pub mod device;
#[cfg(feature = "build")]
pub mod kernel;
#[cfg(feature = "build")]
pub mod ops;

#[cfg(feature = "build")]
#[cfg(feature = "nccl")]
pub mod multi;

#[cfg(feature = "build")]
pub use cudarc;

#[cfg(feature = "build")]
pub use device::{CudaBuffer, CudaDevice, CudaError, CudaMarker, convert_gemm_config};
