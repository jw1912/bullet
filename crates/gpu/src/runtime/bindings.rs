use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    fmt,
    hash::Hash,
};

/// Kernel grid or block dimensions
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// This is a private trait, so nobody outside the crate can access these methods
/// and instead must go through the `Device` and `Stream` structs defined in
/// `crate::gpu::device`
#[allow(clippy::missing_safety_doc)]
pub trait GpuBindings {
    type Err: fmt::Debug + Eq + From<String>;
    type Dev: Copy;
    type Ptr: Copy + Default + Eq + Hash;
    type Ctx: Copy;
    type Stream: Copy;
    type BlasHandle: Copy;
    type Kernel: Copy;
    type Module: Copy;

    unsafe fn driver_init() -> Result<(), Self::Err>;

    unsafe fn device_get(ordinal: c_int) -> Result<Self::Dev, Self::Err>;

    unsafe fn context_create(device: Self::Dev) -> Result<Self::Ctx, Self::Err>;

    unsafe fn context_destroy(device: Self::Dev) -> Result<(), Self::Err>;

    unsafe fn context_set(ctx: Self::Ctx) -> Result<(), Self::Err>;

    unsafe fn stream_create() -> Result<Self::Stream, Self::Err>;

    unsafe fn stream_destroy(stream: Self::Stream) -> Result<(), Self::Err>;

    unsafe fn stream_sync(stream: Self::Stream) -> Result<(), Self::Err>;

    unsafe fn stream_malloc(stream: Self::Stream, bytes: usize) -> Result<Self::Ptr, Self::Err>;

    unsafe fn stream_free(stream: Self::Stream, dev_ptr: Self::Ptr) -> Result<(), Self::Err>;

    unsafe fn stream_memset(stream: Self::Stream, dev_ptr: Self::Ptr, bytes: usize, value: u8)
    -> Result<(), Self::Err>;

    unsafe fn stream_memcpy_d2h(
        stream: Self::Stream,
        dst: *mut c_void,
        src: Self::Ptr,
        bytes: usize,
    ) -> Result<(), Self::Err>;

    unsafe fn stream_memcpy_h2d(
        stream: Self::Stream,
        dst: Self::Ptr,
        src: *const c_void,
        bytes: usize,
    ) -> Result<(), Self::Err>;

    unsafe fn kernel_load(func: Self::Kernel) -> Result<(), Self::Err>;

    unsafe fn kernel_launch(
        func: Self::Kernel,
        stream: Self::Stream,
        grid_dim: Dim3,
        block_dim: Dim3,
        args: *mut *mut c_void,
        smem: c_uint,
    ) -> Result<(), Self::Err>;

    unsafe fn module_create(code: *const c_void) -> Result<Self::Module, Self::Err>;

    unsafe fn module_destroy(module: Self::Module) -> Result<(), Self::Err>;

    unsafe fn module_get_kernel(module: Self::Module, kernel_name: &CStr) -> Result<Self::Kernel, Self::Err>;

    unsafe fn program_compile(
        source_code: &CStr,
        num_options: c_int,
        options: *const *const c_char,
    ) -> Result<Vec<c_char>, Self::Err>;
}

const _C_INT_IS_I32: () = assert!(std::mem::size_of::<i32>() == std::mem::size_of::<c_int>());
const _C_UINT_IS_U32: () = assert!(std::mem::size_of::<u32>() == std::mem::size_of::<c_uint>());
