use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

use crate::runtime::{Dim3, bindings::GpuBindings};

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
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn device_get(ordinal: c_int) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn context_create(device: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn context_destroy(device: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn context_set(ctx: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_create() -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_destroy(stream: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_sync(stream: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_malloc(stream: (), bytes: usize) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_free(stream: (), dev_ptr: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_memset(stream: (), dev_ptr: (), bytes: usize, value: u8) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_memcpy_d2h(stream: (), dst: *mut c_void, src: (), bytes: usize) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn stream_memcpy_h2d(stream: (), dst: (), src: *const c_void, bytes: usize) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn kernel_load(kernel: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn kernel_launch(
        func: (),
        stream: (),
        gdim: Dim3,
        bdim: Dim3,
        args: *mut *mut c_void,
        smem: c_uint,
    ) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn module_create(code: *const c_void) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn module_destroy(module: ()) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn module_get_kernel(module: (), kernel_name: &CStr) -> MockResult {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }

    unsafe fn program_compile(
        source_code: &CStr,
        num_options: c_int,
        options: *const *const c_char,
    ) -> Result<Vec<c_char>, String> {
        unimplemented!("This is a mock runtime! It can't actually do anything!")
    }
}
