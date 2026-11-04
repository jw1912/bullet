use std::{
    alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error},
    cell::Cell,
    collections::BTreeSet,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
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

const DEVICE_COUNT: c_int = 4;
const NO_PEER_DEVICE: c_int = 3;

thread_local! {
    static CURRENT: Cell<Option<c_int>> = const { Cell::new(None) };
}

static NEXT_EVENT: AtomicUsize = AtomicUsize::new(0);
static RECORDED: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MockPtr {
    ptr: *mut u8,
    bytes: usize,
    device: c_int,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MockStream {
    device: c_int,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MockEvent {
    id: usize,
    device: c_int,
}

fn current() -> Result<c_int, String> {
    CURRENT.get().ok_or_else(|| "No device set on this thread!".to_string())
}

fn on_current(device: c_int) -> MockResult {
    let curr = current()?;

    if device != curr {
        return Err(format!("GPU{device} handle used while GPU{curr} is current!"));
    }

    Ok(())
}

fn check_peer(peer: c_int) -> MockResult {
    let curr = current()?;

    if !unsafe { MockGpu::device_can_access_peer(curr, peer)? } {
        return Err(format!("GPU{curr} cannot access GPU{peer}!"));
    }

    Ok(())
}

fn copy(dst: MockPtr, dst_offset: usize, src: MockPtr, src_offset: usize, bytes: usize) -> MockResult {
    for (ptr, offset) in [(dst, dst_offset), (src, src_offset)] {
        if offset.checked_add(bytes).is_none_or(|end| end > ptr.bytes) {
            return Err("Copy range out of bounds!".to_string());
        }
    }

    unsafe { std::ptr::copy(src.ptr.add(src_offset), dst.ptr.add(dst_offset), bytes) };

    Ok(())
}

/// Page-aligned (and never zero-sized) layout for allocations
fn layout(bytes: usize) -> Layout {
    Layout::from_size_align(bytes.max(1), 4096).unwrap()
}

#[allow(unused)]
impl GpuBindings for MockGpu {
    type Err = String;
    type Dev = c_int;
    type Ptr = MockPtr;
    type Ctx = c_int;
    type Stream = MockStream;
    type Event = MockEvent;
    type BlasHandle = ();
    type Kernel = ();
    type Module = ();

    unsafe fn driver_init() -> MockResult {
        Ok(())
    }

    unsafe fn device_get(ordinal: c_int) -> Result<c_int, String> {
        if !(0..DEVICE_COUNT).contains(&ordinal) {
            return Err(format!("GPU{ordinal} does not exist!"));
        }

        Ok(ordinal)
    }

    unsafe fn device_count() -> Result<c_int, String> {
        Ok(DEVICE_COUNT)
    }

    unsafe fn device_props(device: Self::Dev) -> Result<DeviceProps, Self::Err> {
        Ok(DeviceProps {
            name: "MockGPU".into(),
            warp_size: Some(32),
            stream_mem_alloc: false,
            vec_atomics: false,
            arch: None,
            dialect: Dialect::CudaHip,
            is_rocm: false,
        })
    }

    unsafe fn device_can_access_peer(device: c_int, peer: c_int) -> Result<bool, String> {
        Ok(device != peer && device != NO_PEER_DEVICE && peer != NO_PEER_DEVICE)
    }

    unsafe fn context_create(device: c_int) -> Result<c_int, String> {
        Ok(device)
    }

    unsafe fn context_destroy(device: c_int) -> MockResult {
        Ok(())
    }

    unsafe fn context_set(ctx: c_int) -> MockResult {
        CURRENT.set(Some(ctx));
        Ok(())
    }

    unsafe fn context_sync() -> MockResult {
        Ok(())
    }

    unsafe fn context_enable_peer_access(peer: c_int) -> MockResult {
        check_peer(peer)
    }

    unsafe fn context_disable_peer_access(peer: c_int) -> MockResult {
        check_peer(peer)
    }

    unsafe fn context_malloc(bytes: usize) -> Result<MockPtr, String> {
        let device = current()?;
        let ptr = unsafe { Self::host_malloc(bytes)? };
        Ok(MockPtr { ptr: ptr.cast(), bytes, device })
    }

    unsafe fn context_free(dev_ptr: MockPtr) -> MockResult {
        on_current(dev_ptr.device)?;
        unsafe { Self::host_free(dev_ptr.ptr.cast(), dev_ptr.bytes) }
    }

    unsafe fn context_memset(dev_ptr: MockPtr, bytes: usize, value: u8) -> MockResult {
        on_current(dev_ptr.device)?;
        assert!(bytes <= dev_ptr.bytes);

        for entry in unsafe { std::slice::from_raw_parts_mut(dev_ptr.ptr, bytes) } {
            *entry = value;
        }

        Ok(())
    }

    unsafe fn context_memcpy_d2h(dst: *mut c_void, src: MockPtr, bytes: usize) -> MockResult {
        on_current(src.device)?;
        assert!(bytes <= src.bytes);

        let dst = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), bytes) };
        let src = unsafe { std::slice::from_raw_parts(src.ptr, bytes) };

        for (i, j) in dst.iter_mut().zip(src) {
            *i = *j;
        }

        Ok(())
    }

    unsafe fn context_memcpy_h2d(dst: MockPtr, src: *const c_void, bytes: usize) -> MockResult {
        on_current(dst.device)?;
        assert!(bytes <= dst.bytes);

        let dst = unsafe { std::slice::from_raw_parts_mut(dst.ptr, bytes) };
        let src = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), bytes) };

        for (i, j) in dst.iter_mut().zip(src) {
            *i = *j;
        }

        Ok(())
    }

    unsafe fn host_malloc(bytes: usize) -> Result<*mut c_void, String> {
        let layout = layout(bytes);
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Ok(ptr.cast())
    }

    unsafe fn host_free(ptr: *mut c_void, bytes: usize) -> MockResult {
        unsafe { dealloc(ptr.cast(), layout(bytes)) };
        Ok(())
    }

    unsafe fn stream_create() -> Result<MockStream, String> {
        Ok(MockStream { device: current()? })
    }

    unsafe fn stream_destroy(stream: MockStream) -> MockResult {
        on_current(stream.device)
    }

    unsafe fn stream_sync(stream: MockStream) -> MockResult {
        on_current(stream.device)
    }

    unsafe fn stream_malloc(stream: MockStream, bytes: usize) -> Result<MockPtr, String> {
        on_current(stream.device)?;
        Err(MSG.into())
    }

    unsafe fn stream_free(stream: MockStream, dev_ptr: MockPtr) -> MockResult {
        on_current(stream.device)?;
        on_current(dev_ptr.device)?;
        Err(MSG.into())
    }

    unsafe fn stream_memset(stream: MockStream, dev_ptr: MockPtr, bytes: usize, value: u8) -> MockResult {
        on_current(stream.device)?;
        unsafe { Self::context_memset(dev_ptr, bytes, value) }
    }

    unsafe fn stream_memcpy_d2h(stream: MockStream, dst: *mut c_void, src: MockPtr, bytes: usize) -> MockResult {
        on_current(stream.device)?;
        unsafe { Self::context_memcpy_d2h(dst, src, bytes) }
    }

    unsafe fn stream_memcpy_h2d(stream: MockStream, dst: MockPtr, src: *const c_void, bytes: usize) -> MockResult {
        on_current(stream.device)?;
        unsafe { Self::context_memcpy_h2d(dst, src, bytes) }
    }

    unsafe fn stream_memcpy_d2d(
        stream: MockStream,
        dst: MockPtr,
        dst_offset: usize,
        src: MockPtr,
        src_offset: usize,
        bytes: usize,
    ) -> MockResult {
        on_current(stream.device)?;
        on_current(dst.device)?;
        on_current(src.device)?;
        copy(dst, dst_offset, src, src_offset, bytes)
    }

    unsafe fn stream_memcpy_peer(
        stream: MockStream,
        dst: MockPtr,
        dst_ctx: c_int,
        dst_offset: usize,
        src: MockPtr,
        src_ctx: c_int,
        src_offset: usize,
        bytes: usize,
    ) -> MockResult {
        on_current(stream.device)?;

        if stream.device != dst_ctx && stream.device != src_ctx {
            return Err(format!("GPU{} stream cannot copy GPU{src_ctx} -> GPU{dst_ctx}!", stream.device));
        }

        if dst.device != dst_ctx || src.device != src_ctx {
            return Err("Peer copy pointers do not match their contexts!".to_string());
        }

        copy(dst, dst_offset, src, src_offset, bytes)
    }

    unsafe fn event_create() -> Result<MockEvent, String> {
        Ok(MockEvent { id: NEXT_EVENT.fetch_add(1, Ordering::SeqCst), device: current()? })
    }

    unsafe fn event_destroy(event: MockEvent) -> MockResult {
        on_current(event.device)?;
        RECORDED.lock().unwrap().remove(&event.id);
        Ok(())
    }

    unsafe fn event_record(event: MockEvent, stream: MockStream) -> MockResult {
        on_current(event.device)?;
        on_current(stream.device)?;
        RECORDED.lock().unwrap().insert(event.id);
        Ok(())
    }

    unsafe fn stream_wait_event(stream: MockStream, event: MockEvent) -> MockResult {
        on_current(stream.device)?;

        if !RECORDED.lock().unwrap().contains(&event.id) {
            return Err("Waited on an event that was never recorded!".to_string());
        }

        Ok(())
    }

    unsafe fn kernel_load(kernel: ()) -> MockResult {
        Ok(())
    }

    unsafe fn kernel_destroy(_kernel: ()) -> MockResult {
        Ok(())
    }

    unsafe fn kernel_launch(
        func: (),
        stream: MockStream,
        gdim: Dim3,
        bdim: Dim3,
        args: &mut [*mut c_void],
        smem: c_uint,
    ) -> MockResult {
        on_current(stream.device)?;
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
        on_current(stream.device)
    }

    unsafe fn blas_gemm(
        handle: Self::BlasHandle,
        config: GemmConfig,
        a: MockPtr,
        b: MockPtr,
        c: MockPtr,
    ) -> MockResult {
        on_current(a.device)?;
        on_current(b.device)?;
        on_current(c.device)?;
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
        on_current(a.device)?;
        on_current(b.device)?;
        on_current(c.device)?;
        Err(MSG.into())
    }
}
