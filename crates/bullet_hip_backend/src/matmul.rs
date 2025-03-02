use bullet_core::backend::{error::OperationError, shape::Shape, DeviceBuffer};

use crate::{
    backend::{blas, util::catch_cublas, Buffer},
    OperationResult,
};

#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    alpha: f32,
    input_a: &Buffer<f32>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &Buffer<f32>,
    shape_b: Shape,
    trans_b: bool,
    beta: f32,
    output: &mut Buffer<f32>,
) -> OperationResult {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    if shape_a.size() > input_a.size() || shape_b.size() > input_b.size() || shape_o.size() > output.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        let err = blas::sgemm(
            input_a.device().as_ref(),
            alpha,
            input_a.ptr(),
            shape_a.rows(),
            shape_a.cols(),
            trans_a,
            input_b.ptr(),
            shape_b.rows(),
            shape_b.cols(),
            trans_b,
            beta,
            output.mut_ptr(),
            shape_o.rows(),
            shape_o.cols(),
        );

        Ok(catch_cublas(err)?)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sgemm_batched(
    batch_size: usize,
    alpha: f32,
    input_a: &Buffer<f32>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &Buffer<f32>,
    shape_b: Shape,
    trans_b: bool,
    beta: f32,
    output: &mut Buffer<f32>,
) -> OperationResult {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    if batch_size * shape_a.size() > input_a.size()
        || batch_size * shape_b.size() > input_b.size()
        || batch_size * shape_o.size() > output.size()
    {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        let err = blas::batched_sgemm(
            input_a.device().as_ref(),
            batch_size,
            alpha,
            input_a.ptr(),
            shape_a.rows(),
            shape_a.cols(),
            trans_a,
            input_b.ptr(),
            shape_b.rows(),
            shape_b.cols(),
            trans_b,
            beta,
            output.mut_ptr(),
            shape_o.rows(),
            shape_o.cols(),
        );

        Ok(catch_cublas(err)?)
    }
}
