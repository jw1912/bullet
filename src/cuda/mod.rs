pub mod bindings;

use std::ffi::{c_float, c_void};

use bindings::{cudaDeviceSynchronize, cudaMalloc, cudaMemcpy, cudaMemcpyKind, cudaMemset};

pub fn cuda_malloc(size: usize) -> *mut c_float {
    let mut grad = std::ptr::null_mut::<c_float>();

    unsafe {
        let grad_ptr = (&mut grad) as *mut *mut c_float;
        let _ = cudaMalloc(grad_ptr as *mut *mut c_void, size);
        let _ = cudaDeviceSynchronize();
    }

    grad
}

pub fn cuda_calloc<const SIZE: usize>() -> *mut c_float {
    let mut grad = std::ptr::null_mut::<c_float>();

    unsafe {
        let grad_ptr = (&mut grad) as *mut *mut c_float;
        let _ = cudaMalloc(grad_ptr as *mut *mut c_void, SIZE);
        let _ = cudaDeviceSynchronize();
        let _ = cudaMemset(grad as *mut c_void, 0, SIZE);
        let _ = cudaDeviceSynchronize();
    }

    grad
}

pub fn cuda_copy_to_gpu<T>(dest: *mut c_float, src: *const T, amt: usize) {
    unsafe {
        let _ = cudaMemcpy(
            dest as *mut c_void,
            src as *const c_void,
            amt * std::mem::size_of::<T>(),
            cudaMemcpyKind::cudaMemcpyHostToDevice
        );
        let _ = cudaDeviceSynchronize();
    }
}