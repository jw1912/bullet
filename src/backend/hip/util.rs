use super::bindings::{
    hipDeviceSynchronize, hipError_t, hipFree, hipGetDeviceCount, hipGetDeviceProperties, hipGetLastError,
    hipMalloc, hipMemcpy, hipMemcpyKind, hipMemset,
};
use crate::util;
use std::ffi::c_void;

#[macro_export]
macro_rules! catch {
    ($func:expr, $caller:expr) => {
        let err = unsafe { $func };
        if err != hipError_t::hipSuccess {
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
    catch!(hipGetDeviceCount(&mut num));
    assert!(num >= 1);
    let mut props = util::boxed_and_zeroed();
    catch!(hipGetDeviceProperties(&mut *props, 0));

    let mut buf = [0u8; 256];

    for (val, &ch) in buf.iter_mut().zip(props.name.iter()) {
        *val = ch as u8;
    }

    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
    let my_str = cstr.to_str().unwrap();
    my_str.to_string()
}

pub fn device_synchronise() {
    catch!(hipDeviceSynchronize());
}

pub fn panic_if_device_error(msg: &str) {
    catch!(hipGetLastError(), msg);
}

pub fn malloc<T>(num: usize) -> *mut T {
    let size = num * std::mem::size_of::<T>();
    let mut grad = std::ptr::null_mut::<T>();
    let grad_ptr = (&mut grad) as *mut *mut T;

    assert!(!grad_ptr.is_null(), "null pointer");

    catch!(hipMalloc(grad_ptr.cast(), size), "malloc");
    catch!(hipDeviceSynchronize());

    grad
}

/// # Safety
/// Need to make sure not to double free.
pub unsafe fn free<T>(ptr: *mut T, _: usize) {
    catch!(hipFree(ptr.cast()));
}

pub fn calloc<T>(num: usize) -> *mut T {
    let size = num * std::mem::size_of::<T>();
    let grad = malloc(num);
    catch!(hipMemset(grad as *mut c_void, 0, size), "memset");
    catch!(hipDeviceSynchronize());

    grad
}

pub fn set_zero<T>(ptr: *mut T, num: usize) {
    catch!(hipMemset(ptr.cast(), 0, num * std::mem::size_of::<T>()), "memset");
    catch!(hipDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_to_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        hipMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), hipMemcpyKind::hipMemcpyHostToDevice),
        "memcpy"
    );
    catch!(hipDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_from_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        hipMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), hipMemcpyKind::hipMemcpyDeviceToHost),
        "memcpy"
    );
    catch!(hipDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_on_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        hipMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), hipMemcpyKind::hipMemcpyDeviceToDevice),
        "memcpy"
    );
    catch!(hipDeviceSynchronize());
}
