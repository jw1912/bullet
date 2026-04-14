use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use crate::runtime::{
    Dim3,
    bindings::{DeviceProps, GemmConfig, GpuBindings},
};

/// Used to type check code without requiring CUDA/ROCm
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MockGpu;

type MockResult = Result<(), String>;

static MSG: &str =
    "This is a mock runtime! It can't actually do anything! You need to enable either the `cuda` or `rocm` features!";

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
        Err(MSG.into())
    }

    unsafe fn device_get(ordinal: c_int) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn device_props(device: Self::Dev) -> Result<DeviceProps, Self::Err> {
        Err(MSG.into())
    }

    unsafe fn context_create(device: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn context_destroy(device: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn context_set(ctx: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn context_sync() -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn context_malloc(bytes: usize) -> Result<Self::Ptr, Self::Err> {
        Err(MSG.into())
    }

    unsafe fn context_free(dev_ptr: Self::Ptr) -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn context_memset(dev_ptr: Self::Ptr, bytes: usize, value: u8) -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn context_memcpy_d2h(dst: *mut c_void, src: Self::Ptr, bytes: usize) -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn context_memcpy_h2d(dst: Self::Ptr, src: *const c_void, bytes: usize) -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn stream_create() -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_destroy(stream: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_sync(stream: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_malloc(stream: (), bytes: usize) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_free(stream: (), dev_ptr: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_memset(stream: (), dev_ptr: (), bytes: usize, value: u8) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_memcpy_d2h(stream: (), dst: *mut c_void, src: (), bytes: usize) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_memcpy_h2d(stream: (), dst: (), src: *const c_void, bytes: usize) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn kernel_load(kernel: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn kernel_launch(
        func: (),
        stream: (),
        gdim: Dim3,
        bdim: Dim3,
        args: *mut *mut c_void,
        smem: c_uint,
    ) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn module_create(code: *const c_void) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn module_destroy(module: ()) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn module_get_kernel(module: (), kernel_name: &CStr) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn program_compile(
        source_code: &CStr,
        num_options: c_int,
        options: *const *const c_char,
    ) -> Result<Vec<c_char>, String> {
        Err(MSG.into())
    }

    unsafe fn blas_create() -> Result<Self::BlasHandle, Self::Err> {
        Err(MSG.into())
    }

    unsafe fn blas_destroy(handle: Self::BlasHandle) -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn blas_set_stream(handle: Self::BlasHandle, stream: Self::Stream) -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn blas_gemm(
        handle: Self::BlasHandle,
        config: GemmConfig,
        a: Self::Ptr,
        b: Self::Ptr,
        c: Self::Ptr,
    ) -> Result<(), Self::Err> {
        Err(MSG.into())
    }

    unsafe fn blas_gemm_batched(
        handle: Self::BlasHandle,
        batch_size: c_int,
        config: GemmConfig,
        a: Self::Ptr,
        b: Self::Ptr,
        c: Self::Ptr,
    ) -> Result<(), Self::Err> {
        Err(MSG.into())
    }
}
