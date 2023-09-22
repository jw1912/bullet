use std::ffi::{c_float, c_void};

use crate::bindings::{
    cudaDeviceSynchronize,
    cudaError,
    cudaMalloc,
    cudaMemcpy,
    cudaMemcpyKind,
    cudaMemset,
};

#[macro_export]
macro_rules! catch {
    ($func:expr, $caller:expr) => {
        let err = unsafe{ $func };
        if err != cudaError::cudaSuccess {
            panic!("{}: {:?}", $caller, err);
        }
    };
    ($func:expr) => {
        catch!($func, "synchronise")
    }
}

pub fn cuda_malloc<T>(size: usize) -> *mut T {
    let mut grad = std::ptr::null_mut::<T>();

    let grad_ptr = (&mut grad) as *mut *mut T;
    assert!(!grad_ptr.is_null(), "null pointer");
    catch!(cudaMalloc(grad_ptr.cast(), size), "malloc");
    catch!(cudaDeviceSynchronize());

    grad
}

pub fn cuda_calloc(size: usize) -> *mut c_float {
    let mut grad = std::ptr::null_mut::<c_float>();

    let grad_ptr = (&mut grad) as *mut *mut c_float;
    catch!(cudaMalloc(grad_ptr.cast(), size), "malloc");
    catch!(cudaDeviceSynchronize());
    catch!(cudaMemset(grad as *mut c_void, 0, size), "memset");
    catch!(cudaDeviceSynchronize());

    grad
}

pub fn cuda_copy_to_gpu<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(cudaMemcpy(
        dest.cast(),
        src.cast(),
        amt * std::mem::size_of::<T>(),
        cudaMemcpyKind::cudaMemcpyHostToDevice
    ), "memcpy");
    catch!(cudaDeviceSynchronize());
}

pub fn cuda_copy_from_gpu<T>(dest: *mut T, src: *const T, amt: usize) {
    catch!(cudaMemcpy(
        dest.cast(),
        src.cast(),
        amt * std::mem::size_of::<T>(),
        cudaMemcpyKind::cudaMemcpyDeviceToHost
    ), "memcpy");
    catch!(cudaDeviceSynchronize());
}