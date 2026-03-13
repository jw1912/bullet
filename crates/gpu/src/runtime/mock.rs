use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use crate::runtime::{
    Dim3,
    bindings::{GemmConfig, GpuBindings},
};

/// Used to type check code without requiring CUDA/ROCm
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MockGpu;

type MockResult = Result<(), String>;

#[allow(unused)]
impl GpuBindings for MockGpu {
    type Err = String;
    type Dev = ();
    type Ptr = ();
    type Ctx = ();
    type Stream = ();
    type BlasHandle = ();
    type Kernel = ();
    type Module = ();

    unsafe fn driver_init() -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn device_get(ordinal: c_int) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn context_create(device: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn context_destroy(device: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn context_set(ctx: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_create() -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_destroy(stream: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_sync(stream: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_malloc(stream: (), bytes: usize) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_free(stream: (), dev_ptr: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_memset(stream: (), dev_ptr: (), bytes: usize, value: u8) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_memcpy_d2h(stream: (), dst: *mut c_void, src: (), bytes: usize) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn stream_memcpy_h2d(stream: (), dst: (), src: *const c_void, bytes: usize) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn kernel_load(kernel: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn kernel_launch(
        func: (),
        stream: (),
        gdim: Dim3,
        bdim: Dim3,
        args: *mut *mut c_void,
        smem: c_uint,
    ) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn module_create(code: *const c_void) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn module_destroy(module: ()) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn module_get_kernel(module: (), kernel_name: &CStr) -> MockResult {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn program_compile(
        source_code: &CStr,
        num_options: c_int,
        options: *const *const c_char,
    ) -> Result<Vec<c_char>, String> {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn blas_create() -> Result<Self::BlasHandle, Self::Err> {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn blas_destroy(handle: Self::BlasHandle) -> Result<(), Self::Err> {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn blas_set_stream(handle: Self::BlasHandle, stream: Self::Stream) -> Result<(), Self::Err> {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn blas_gemm(
        handle: Self::BlasHandle,
        config: GemmConfig,
        a: Self::Ptr,
        b: Self::Ptr,
        c: Self::Ptr,
    ) -> Result<(), Self::Err> {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }

    unsafe fn blas_gemm_batched(
        handle: Self::BlasHandle,
        batch_size: c_int,
        config: GemmConfig,
        a: Self::Ptr,
        b: Self::Ptr,
        c: Self::Ptr,
    ) -> Result<(), Self::Err> {
        Err("This is a mock runtime! It can't actually do anything!".into())
    }
}
