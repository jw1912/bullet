pub mod bindings;

use std::ffi::{c_float, c_void};

use bindings::{cudaMalloc, cudaMemset};

pub unsafe fn cuda_calloc<const SIZE: usize>() -> *mut c_float {
    let mut grad = std::ptr::null_mut::<c_float>();
    let grad_ptr = (&mut grad) as *mut *mut c_float;
    cudaMalloc(grad_ptr as *mut *mut c_void, SIZE);
    cudaMemset(grad as *mut c_void, 0, SIZE);
    grad
}