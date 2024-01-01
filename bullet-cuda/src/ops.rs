use crate::{bindings::{self, cublasOperation_t}, CublasHandle};

use std::ffi::c_int;

pub use bindings::splatAdd as splat_add;
pub use bindings::sigmoidMSE as sigmoid_mse;
pub use bindings::sparseAffineForward as sparse_affine_forward;
pub use bindings::sparseAffineBackward as sparse_affine_backward;
pub use bindings::updateWeights as update_weights;
pub use bindings::activateReLU as activate_relu;
pub use bindings::activateCReLU as activate_crelu;
pub use bindings::activateSCReLU as activate_screlu;
pub use bindings::backpropReLU as backprop_relu;
pub use bindings::backpropCReLU as backprop_crelu;
pub use bindings::backpropSCReLU as backprop_screlu;

#[allow(clippy::too_many_arguments)]
/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn mul_matrix_vector<const TRANSA: bool>(
    handle: CublasHandle,
    m: c_int,
    n: c_int,
    a_ptr: *const f32,
    a_str: c_int,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: c_int,
) {
    let alpha = 1.0;
    let beta = 0.0;

    let (transa, x_ld, y_ld) = if TRANSA {
        (cublasOperation_t::CUBLAS_OP_T, n, m)
    } else {
        (cublasOperation_t::CUBLAS_OP_N, m, n)
    };

    unsafe {
        bindings::cublasSgemvStridedBatched(
            *handle,
            transa,
            n,
            m,
            &alpha,
            a_ptr,
            n,
            a_str.into(),
            x_ptr,
            1,
            x_ld.into(),
            &beta,
            y_ptr,
            1,
            y_ld.into(),
            batch_size,
        );
    }
}

#[allow(clippy::too_many_arguments)]
/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn reduce_add_mul_vector_vectort(
    handle: CublasHandle,
    m: c_int,
    n: c_int,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: c_int,
) {
    let alpha = 1.0;
    let beta = 0.0;

    unsafe {
        bindings::cublasSgemm_v2(
            *handle,
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_T,
            n, m, batch_size,
            &alpha,
            y_ptr, n,
            x_ptr, m,
            &beta,
            a_ptr, n,
        );
    }
}

/// # Safety
/// This should only be used and exposed internally.
pub unsafe fn reduce_add(
    handle: CublasHandle,
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
