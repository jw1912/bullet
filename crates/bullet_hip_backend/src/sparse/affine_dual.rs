use crate::{
    backend::{ops, Buffer},
    OperationResult,
};
use bullet_core::{
    device::{DeviceBuffer, OperationError},
    graph::operation::Activation,
    shape::Shape,
};

#[allow(clippy::too_many_arguments)]
pub fn sparse_affine_dual_activate(
    batch_size: usize,
    input_a: &Buffer<f32>,
    shape_a: Shape,
    input_b1: &Buffer<i32>,
    input_b2: &Buffer<i32>,
    shape_b: Shape,
    nnz: usize,
    input_c: &Buffer<f32>,
    output: &mut Buffer<f32>,
    activation: Activation,
) -> OperationResult {
    let mut output_shape = shape_a * shape_b;
    if output_shape.size() > input_c.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    output_shape = Shape::new(output_shape.rows() * 2, output_shape.cols());

    if shape_a.size() > input_a.size()
        || batch_size * nnz > input_b1.size()
        || batch_size * nnz > input_b2.size()
        || batch_size * output_shape.size() > output.size()
    {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::sparseAffineDualForward(
            batch_size,
            nnz,
            shape_a.rows(),
            input_a.ptr(),
            input_c.ptr(),
            input_b1.ptr(),
            input_b2.ptr(),
            output.mut_ptr(),
            activation as i32,
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_sparse_affine_dual_activate(
    batch_size: usize,
    input_a: &Buffer<f32>,
    input_a_grad: &mut Buffer<f32>,
    shape_a: Shape,
    input_b1: &Buffer<i32>,
    input_b2: &Buffer<i32>,
    shape_b: Shape,
    nnz: usize,
    input_c: &Buffer<f32>,
    input_c_grad: &mut Buffer<f32>,
    outputs: &Buffer<f32>,
    output_grad: &Buffer<f32>,
    activation: Activation,
) -> OperationResult {
    let mut output_shape = shape_a * shape_b;
    if output_shape.size() > input_c.size() || output_shape.size() > input_c_grad.size() {
        return Err(OperationError::IndexOutOfBounds);
    }

    output_shape = Shape::new(output_shape.rows() * 2, output_shape.cols());

    if shape_a.size() > input_a.size()
        || shape_a.size() > input_a_grad.size()
        || batch_size * nnz > input_b1.size()
        || batch_size * nnz > input_b2.size()
        || batch_size * output_shape.size() > outputs.size()
        || batch_size * output_shape.size() > output_grad.size()
    {
        return Err(OperationError::IndexOutOfBounds);
    }

    unsafe {
        ops::sparseAffineDualBackward(
            batch_size,
            nnz,
            shape_a.rows(),
            input_a_grad.mut_ptr(),
            input_c_grad.mut_ptr(),
            input_b1.ptr(),
            input_b2.ptr(),
            outputs.ptr(),
            output_grad.ptr(),
            activation as i32,
        );
    }

    Ok(())
}
