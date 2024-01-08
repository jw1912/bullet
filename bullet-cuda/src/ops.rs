use crate::{
    bindings::{self, cublasOperation_t},
    DeviceHandles,
};

use std::ffi::c_int;

pub use bindings::sigmoidMSE as sigmoid_mse;
pub use bindings::sparseAffineBackward as sparse_affine_backward;
pub use bindings::sparseAffineForward as sparse_affine_forward;
pub use bindings::splatAdd as splat_add;
pub use bindings::updateWeights as update_weights;

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn splat_mul_matrix_vector(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    unsafe {
        bindings::cublasSgemm_v2(
            *handle,
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            n,
            batch_size,
            m,
            &alpha,
            a_ptr,
            n,
            x_ptr,
            m,
            &beta,
            y_ptr,
            n,
        );
    }
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn splat_mul_matrixt_vector(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    y_ptr: *const f32,
    x_ptr: *mut f32,
    batch_size: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    unsafe {
        bindings::cublasSgemm_v2(
            *handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            batch_size,
            n,
            &alpha,
            a_ptr,
            n,
            y_ptr,
            n,
            &beta,
            x_ptr,
            m,
        );
    }
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn reduce_add_mul_vector_vectort(
    handle: DeviceHandles,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = m as c_int;
    let n = n as c_int;
    let batch_size = batch_size as c_int;

    unsafe {
        bindings::cublasSgemm_v2(
            *handle,
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_T,
            n,
            m,
            batch_size,
            &alpha,
            y_ptr,
            n,
            x_ptr,
            m,
            &beta,
            a_ptr,
            n,
        );
    }
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn reduce_add(
    handle: DeviceHandles,
    ones: *const f32,
    batch_size: usize,
    out_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let m = batch_size as c_int;
    let n = out_size as c_int;

    bindings::cublasSgemv_v2(
        *handle,
        cublasOperation_t::CUBLAS_OP_N,
        n,
        m,
        &alpha,
        inp,
        n,
        ones,
        0,
        &beta,
        out,
        1,
    );
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_relu(size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateReLU(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_crelu(size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateCReLU(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn activate_screlu(size: usize, inp: *const f32, out: *mut f32) {
    bindings::activateSCReLU(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_relu(size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropReLU(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_crelu(size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropCReLU(size, inp, out);
}

/// # Safety
/// Pass valid pointers and sizes.
pub unsafe fn backprop_screlu(size: usize, inp: *const f32, out: *mut f32) {
    bindings::backpropSCReLU(size, inp, out);
}
