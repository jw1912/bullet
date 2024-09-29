use super::bindings;
use std::ffi::c_void;

#[macro_export]
macro_rules! catch {
    ($func:expr, $caller:expr) => {
        let err = $func;
        if err != bindings::cudaError::cudaSuccess {
            panic!("{}: {:?}", $caller, err);
        }
    };
    ($func:expr) => {
        catch!($func, "synchronise")
    };
}

pub unsafe fn device_name() -> String {
    use std::ffi::CStr;
    let mut num = 0;
    catch!(bindings::cudaGetDeviceCount(&mut num));
    assert!(num >= 1);
    let mut props = crate::util::boxed_and_zeroed();
    catch!(bindings::cudaGetDeviceProperties_v2(&mut *props, 0));

    let mut buf = [0u8; 256];

    for (val, &ch) in buf.iter_mut().zip(props.name.iter()) {
        *val = ch as u8;
    }

    let cstr = CStr::from_bytes_until_nul(&buf).unwrap();
    let my_str = cstr.to_str().unwrap();
    my_str.to_string()
}

pub fn device_synchronise() {
    // # Safety
    // This function cannot fail.
    unsafe {
        catch!(bindings::cudaDeviceSynchronize());
    }
}

pub fn panic_if_device_error(msg: &str) {
    // # Safety
    // This function cannot fail.
    unsafe {
        catch!(bindings::cudaGetLastError(), msg);
    }
}

fn malloc<T>(num: usize) -> *mut T {
    let size = num * std::mem::size_of::<T>();
    let mut grad = std::ptr::null_mut::<T>();
    let grad_ptr = (&mut grad) as *mut *mut T;

    assert!(!grad_ptr.is_null(), "null pointer");

    unsafe {
        catch!(bindings::cudaMalloc(grad_ptr.cast(), size), "malloc");
        catch!(bindings::cudaDeviceSynchronize());
    }

    grad
}

/// ### Safety
/// Need to make sure not to double free.
pub unsafe fn free<T>(ptr: *mut T, _: usize) {
    catch!(bindings::cudaFree(ptr.cast()));
}

/// ### Safety
/// Type needs to be zeroable.
pub unsafe fn calloc<T>(num: usize) -> *mut T {
    let size = num * std::mem::size_of::<T>();
    let grad = malloc(num);
    catch!(bindings::cudaMemset(grad as *mut c_void, 0, size), "memset");
    catch!(bindings::cudaDeviceSynchronize());

    grad
}

/// ### Safety
/// Type needs to be zeroable.
pub unsafe fn set_zero<T>(ptr: *mut T, num: usize) {
    catch!(bindings::cudaMemset(ptr.cast(), 0, num * std::mem::size_of::<T>()), "memset");
    catch!(bindings::cudaDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_to_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::cudaMemcpyKind::cudaMemcpyHostToDevice),
        "memcpy"
    );
    catch!(bindings::cudaDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_from_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::cudaMemcpyKind::cudaMemcpyDeviceToHost),
        "memcpy"
    );
    catch!(bindings::cudaDeviceSynchronize());
}

/// # Safety
/// Pointers need to be valid and `amt` need to be valid.
pub unsafe fn copy_on_device<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(
        bindings::cudaMemcpy(dest.cast(), src.cast(), amt * std::mem::size_of::<T>(), bindings::cudaMemcpyKind::cudaMemcpyDeviceToDevice),
        "memcpy"
    );
    catch!(bindings::cudaDeviceSynchronize());
}
