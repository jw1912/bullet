use bullet_core::backend::{error::OperationError, DeviceBuffer};

use crate::{
    backend::{blas, ops, util::catch_cublas},
    Buffer, OperationResult,
};

pub fn linear_comb_single(
    size: usize,
    alpha: f32,
    input_a: Option<&Buffer<f32>>,
    beta: f32,
    input_b: Option<&Buffer<f32>>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    // cublas scale is super slow for some reason
    if let (None, None) = (input_a, input_b) {
        return scale(size, output, alpha);
    }

    let aptr = input_a.map(|a| {
        assert!(size <= a.size());
        a.ptr()
    });

    let bptr = if let Some(b) = input_b {
        assert!(size <= b.size());
        b.ptr()
    } else {
        std::ptr::null()
    };

    unsafe {
        let err =
            blas::linear_comb_matrices(output.device().as_ref(), size, 1, alpha, aptr, beta, bptr, output.mut_ptr());
        Ok(catch_cublas(err)?)
    }
}

pub fn add_assign_single_to_batched_scaled(
    single_size: usize,
    batch_size: usize,
    ones: &Buffer<f32>,
    alpha: f32,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    if single_size > input.size() || single_size * batch_size > output.size() || batch_size > ones.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        let err = blas::add_vector_to_matrix_columns(
            output.device().as_ref(),
            single_size,
            batch_size,
            alpha,
            ones.ptr(),
            input.ptr(),
            output.mut_ptr(),
        );

        Ok(catch_cublas(err)?)
    }
}

fn scale(size: usize, params: &mut Buffer<f32>, alpha: f32) -> OperationResult {
    if size > params.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::scale(size, params.mut_ptr(), alpha);
    }

    Ok(())
}
