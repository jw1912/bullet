#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "hip")]
pub mod hip;

use std::{
    ffi::{c_int, c_uint, c_void},
    fmt,
};

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// This is a private trait, so nobody outside the crate can access these methods
/// and instead must go through the `GpuDevice` and `GpuStream` structs defined in
/// `crate::gpu::device`
#[allow(clippy::missing_safety_doc)]
pub trait GpuBindings {
    type E: Copy + fmt::Debug + Eq;
    type S: Copy + fmt::Debug + Eq;

    unsafe fn device_init(device: i32) -> Result<(), Self::E>;

    unsafe fn device_set(device: i32) -> Result<(), Self::E>;

    unsafe fn stream_create(stream: *mut Self::S) -> Result<(), Self::E>;

    unsafe fn stream_destroy(stream: Self::S) -> Result<(), Self::E>;

    unsafe fn stream_sync(stream: Self::S) -> Result<(), Self::E>;

    unsafe fn stream_malloc(stream: Self::S, dev_ptr: *mut *mut c_void, bytes: usize) -> Result<(), Self::E>;

    unsafe fn stream_free(stream: Self::S, dev_ptr: *mut c_void) -> Result<(), Self::E>;

    unsafe fn stream_memcpy(
        stream: Self::S,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: MemcpyKind,
    ) -> Result<(), Self::E>;

    unsafe fn stream_launch_kernel(
        stream: Self::S,
        func: *const c_void,
        drid_dim: Dim3,
        block_dim: Dim3,
        args: *mut *mut c_void,
        smem: usize,
    ) -> Result<(), Self::E>;
}

const _C_INT_IS_I32: () = assert!(std::mem::size_of::<i32>() == std::mem::size_of::<c_int>());
const _C_UINT_IS_U32: () = assert!(std::mem::size_of::<u32>() == std::mem::size_of::<c_uint>());
