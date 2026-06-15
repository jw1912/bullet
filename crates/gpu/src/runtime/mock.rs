use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    ffi::{CStr, c_char, c_int, c_uint, c_void},
};

use crate::runtime::{
    Dialect, Dim3,
    bindings::{DeviceProps, GemmConfig, GpuBindings},
};

/// Used to type check code without requiring CUDA/ROCm
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MockGpu;

type MockResult = Result<(), String>;

static MSG: &str =
    "This is a mock runtime! It can't actually do anything! You need to enable either the `cuda` or `rocm` features!";

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MockPtr {
    ptr: *mut u8,
    bytes: usize,
}

#[allow(unused)]
impl GpuBindings for MockGpu {
    type Err = String;
    type Dev = ();
    type Ptr = MockPtr;
    type Ctx = ();
    type Stream = ();
    type BlasHandle = ();
    type Kernel = ();
    type Module = ();

    unsafe fn driver_init() -> MockResult {
        Ok(())
    }

    unsafe fn device_get(ordinal: c_int) -> MockResult {
        Ok(())
    }

    unsafe fn device_props(device: Self::Dev) -> Result<DeviceProps, Self::Err> {
        Ok(DeviceProps {
            name: "MockGPU".into(),
            warp_size: Some(32),
            stream_mem_alloc: false,
            vec_atomics: false,
            arch: None,
            dialect: Dialect::CudaHip,
        })
    }

    unsafe fn context_create(device: ()) -> MockResult {
        Ok(())
    }

    unsafe fn context_destroy(device: ()) -> MockResult {
        Ok(())
    }

    unsafe fn context_set(ctx: ()) -> MockResult {
        Ok(())
    }

    unsafe fn context_sync() -> MockResult {
        Ok(())
    }

    unsafe fn context_malloc(bytes: usize) -> Result<MockPtr, String> {
        let layout = Layout::array::<u8>(bytes).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Ok(MockPtr { ptr, bytes })
    }

    unsafe fn context_free(dev_ptr: MockPtr) -> MockResult {
        let layout = Layout::array::<u8>(dev_ptr.bytes).unwrap();
        unsafe { dealloc(dev_ptr.ptr, layout) };
        Ok(())
    }

    unsafe fn context_memset(dev_ptr: MockPtr, bytes: usize, value: u8) -> MockResult {
        assert!(bytes <= dev_ptr.bytes);

        for entry in unsafe { std::slice::from_raw_parts_mut(dev_ptr.ptr, bytes) } {
            *entry = value;
        }

        Ok(())
    }

    unsafe fn context_memcpy_d2h(dst: *mut c_void, src: MockPtr, bytes: usize) -> MockResult {
        assert!(bytes <= src.bytes);

        let dst = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), bytes) };
        let src = unsafe { std::slice::from_raw_parts(src.ptr, bytes) };

        for (i, j) in dst.iter_mut().zip(src) {
            *i = *j;
        }

        Ok(())
    }

    unsafe fn context_memcpy_h2d(dst: MockPtr, src: *const c_void, bytes: usize) -> MockResult {
        assert!(bytes <= dst.bytes);

        let dst = unsafe { std::slice::from_raw_parts_mut(dst.ptr, bytes) };
        let src = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), bytes) };

        for (i, j) in dst.iter_mut().zip(src) {
            *i = *j;
        }

        Ok(())
    }

    unsafe fn stream_create() -> MockResult {
        Ok(())
    }

    unsafe fn stream_destroy(stream: ()) -> MockResult {
        Ok(())
    }

    unsafe fn stream_sync(stream: ()) -> MockResult {
        Ok(())
    }

    unsafe fn stream_malloc(stream: (), bytes: usize) -> Result<MockPtr, String> {
        Err(MSG.into())
    }

    unsafe fn stream_free(stream: (), dev_ptr: MockPtr) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn stream_memset(stream: (), dev_ptr: MockPtr, bytes: usize, value: u8) -> MockResult {
        unsafe { Self::context_memset(dev_ptr, bytes, value) }
    }

    unsafe fn stream_memcpy_d2h(stream: (), dst: *mut c_void, src: MockPtr, bytes: usize) -> MockResult {
        unsafe { Self::context_memcpy_d2h(dst, src, bytes) }
    }

    unsafe fn stream_memcpy_h2d(stream: (), dst: MockPtr, src: *const c_void, bytes: usize) -> MockResult {
        unsafe { Self::context_memcpy_h2d(dst, src, bytes) }
    }

    unsafe fn kernel_load(kernel: ()) -> MockResult {
        Ok(())
    }

    unsafe fn kernel_destroy(_kernel: ()) -> MockResult {
        Ok(())
    }

    unsafe fn kernel_launch(
        func: (),
        stream: (),
        gdim: Dim3,
        bdim: Dim3,
        args: &mut [*mut c_void],
        smem: c_uint,
    ) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn module_create(code: *const c_void) -> MockResult {
        Ok(())
    }

    unsafe fn module_destroy(module: ()) -> MockResult {
        Ok(())
    }

    unsafe fn module_get_kernel(module: (), kernel_name: &CStr) -> MockResult {
        Ok(())
    }

    unsafe fn program_compile(
        source_code: &CStr,
        num_options: c_int,
        options: *const *const c_char,
    ) -> Result<Vec<c_char>, String> {
        Ok(Vec::new())
    }

    unsafe fn blas_create() -> Result<Self::BlasHandle, Self::Err> {
        Ok(())
    }

    unsafe fn blas_destroy(handle: Self::BlasHandle) -> MockResult {
        Ok(())
    }

    unsafe fn blas_set_stream(handle: Self::BlasHandle, stream: Self::Stream) -> MockResult {
        Ok(())
    }

    unsafe fn blas_gemm(
        handle: Self::BlasHandle,
        config: GemmConfig,
        a: MockPtr,
        b: MockPtr,
        c: MockPtr,
    ) -> MockResult {
        Err(MSG.into())
    }

    unsafe fn blas_gemm_batched(
        handle: Self::BlasHandle,
        batch_size: c_int,
        config: GemmConfig,
        a: MockPtr,
        b: MockPtr,
        c: MockPtr,
    ) -> MockResult {
        Err(MSG.into())
    }
}
