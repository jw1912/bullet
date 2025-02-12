use bullet_core::{device::DeviceBuffer, shape::Shape};

use crate::backend::{blas, Buffer};

#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    input_a: &Buffer<f32>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &Buffer<f32>,
    shape_b: Shape,
    trans_b: bool,
    output: &mut Buffer<f32>,
    increment: bool,
) {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    assert!(shape_a.size() <= input_a.size());
    assert!(shape_b.size() <= input_b.size());
    assert!(shape_o.size() <= output.size());

    unsafe {
        blas::sgemm(
            input_a.device().as_ref(),
            input_a.ptr(),
            shape_a.rows(),
            shape_a.cols(),
            trans_a,
            input_b.ptr(),
            shape_b.rows(),
            shape_b.cols(),
            trans_b,
            output.mut_ptr(),
            shape_o.rows(),
            shape_o.cols(),
            increment,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sgemm_batched(
    batch_size: usize,
    input_a: &Buffer<f32>,
    shape_a: Shape,
    trans_a: bool,
    input_b: &Buffer<f32>,
    shape_b: Shape,
    trans_b: bool,
    output: &mut Buffer<f32>,
    increment: bool,
) {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    assert!(batch_size * shape_a.size() <= input_a.size());
    assert!(batch_size * shape_b.size() <= input_b.size());
    assert!(batch_size * shape_o.size() <= output.size());

    unsafe {
        blas::batched_sgemm(
            input_a.device().as_ref(),
            batch_size,
            input_a.ptr(),
            shape_a.rows(),
            shape_a.cols(),
            trans_a,
            input_b.ptr(),
            shape_b.rows(),
            shape_b.cols(),
            trans_b,
            output.mut_ptr(),
            shape_o.rows(),
            shape_o.cols(),
            increment,
        );
    }
}
