use std::ffi::c_void;
use crate::util;
use super::bindings::{
    cudaDeviceSynchronize, cudaError, cudaFree, cudaGetLastError,
    cudaMalloc, cudaMemcpy, cudaMemcpyKind, cudaMemset, cudaGetDeviceCount, cudaGetDeviceProperties_v2,
};

#[macro_export]
macro_rules! catch {
    ($func:expr, $caller:expr) => {
        let err = unsafe { $func };
        if err != cudaError::cudaSuccess {
            panic!("{}: {:?}", $caller, err);
        }
    };
    ($func:expr) => {
        catch!($func, "synchronise")
    };
}

pub fn device_name() -> String {
    use std::ffi::CStr;
    let mut num = 0;
    catch!(cudaGetDeviceCount(&mut num));
    assert!(num >= 1);
    let mut props = util::boxed_and_zeroed();
    catch!(cudaGetDeviceProperties_v2(&mut *props, 0));

    let mut buf = [0u8; 256];

    for (val, &ch) in buf.iter_mut().zip(props.name.iter()) {
        *val = ch as u8;
    }

    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
    let my_str = cstr.to_str().unwrap();
    my_str.to_string()
}

pub fn device_synchronise() {
    catch!(cudaDeviceSynchronize());
}

pub fn panic_if_device_error(msg: &str) {
    catch!(cudaGetLastError(), msg);
}

pub fn malloc<T>(num: usize) -> *mut T {
    let size = num * std::mem::size_of::<T>();
    let mut grad = std::ptr::null_mut::<T>();
    let grad_ptr = (&mut grad) as *mut *mut T;

    assert!(!grad_ptr.is_null(), "null pointer");

    catch!(cudaMalloc(grad_ptr.cast(), size), "malloc");
    catch!(cudaDeviceSynchronize());

    grad
}

/// # Safety
/// Need to make sure not to double free.
pub unsafe fn free(ptr: *mut f32, _: usize) {
    catch!(cudaFree(ptr.cast()));
}

/// # Safety
/// Need to make sure not to double free.
pub unsafe fn free_raw_bytes(ptr: *mut u8, _: usize) {
    catch!(cudaFree(ptr.cast()));
}

pub fn calloc<T>(num: usize) -> *mut T {
    let size = num * std::mem::size_of::<T>();
    let grad = malloc(num);
    catch!(cudaMemset(grad as *mut c_void, 0, size), "memset");
    catch!(cudaDeviceSynchronize());

    grad
}

pub fn set_zero<T>(ptr: *mut T, num: usize) {
    catch!(
        cudaMemset(ptr.cast(), 0, num * std::mem::size_of::<T>()),
        "memset"
    );
    catch!(cudaDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_to_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        cudaMemcpy(
            dest.cast(),
            src.cast(),
            amt * std::mem::size_of::<T>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice
        ),
        "memcpy"
    );
    catch!(cudaDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_from_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        cudaMemcpy(
            dest.cast(),
            src.cast(),
            amt * std::mem::size_of::<T>(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost
        ),
        "memcpy"
    );
    catch!(cudaDeviceSynchronize());
}
