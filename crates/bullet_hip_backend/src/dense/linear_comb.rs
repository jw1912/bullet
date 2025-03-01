use bullet_core::backend::{DeviceBuffer, OperationError};

use crate::{
    backend::{blas, util::catch_cublas},
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

pub fn reduce_add(
    ones: &Buffer<f32>,
    size: usize,
    batch_size: usize,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    if size * batch_size > input.size() || size > output.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        let err = blas::reduce_add_cols(
            input.device().as_ref(),
            size,
            batch_size,
            ones.ptr(),
            input.ptr(),
            output.mut_ptr(),
            1.0,
            false,
        );

        Ok(catch_cublas(err)?)
    }
}
