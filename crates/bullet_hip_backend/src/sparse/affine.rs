use bullet_core::{
    device::{DeviceBuffer, OperationError},
    graph::operation::Activation,
    shape::Shape,
};

use crate::{
    backend::{ops, Buffer},
    OperationResult,
};

#[allow(clippy::too_many_arguments)]
pub fn sparse_affine(
    batch_size: usize,
    stride: Option<bool>,
    activation: Activation,
    input_a: &Buffer<f32>,
    shape_a: Shape,
    input_b: &Buffer<i32>,
    shape_b: Shape,
    nnz: usize,
    input_c: Option<&Buffer<f32>>,
    input_c_buckets: Option<(&Buffer<i32>, usize)>,
    output: &mut Buffer<f32>,
) -> OperationResult {
    let shape_o = shape_a * shape_b;

    let (stride, offset) = if let Some(b) = stride { (2, if b { shape_a.rows() } else { 0 }) } else { (1, 0) };

    if shape_a.size() > input_a.size()
        || batch_size * nnz > input_b.size()
        || batch_size * shape_o.size() * stride > output.size()
    {
        return Err(OperationError::IndexOutOfBounds);
    }

    let (c_ptr, s_ptr) = if let Some(c) = input_c {
        let s_ptr = if let Some((buckets, num_buckets)) = input_c_buckets {
            if num_buckets * shape_o.size() > c.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            if batch_size > buckets.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            buckets.ptr()
        } else {
            if shape_o.size() > c.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            std::ptr::null()
        };

        (c.ptr(), s_ptr)
    } else {
        (std::ptr::null(), std::ptr::null())
    };

    unsafe {
        ops::sparse_affine(
            activation as i32,
            stride,
            nnz,
            shape_a.rows(),
            shape_a.cols(),
            batch_size,
            input_a.ptr(),
            input_b.ptr(),
            s_ptr,
            c_ptr,
            output.mut_ptr().add(offset),
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_sparse_affine(
    batch_size: usize,
    stride: Option<bool>,
    activation: Activation,
    input_a: &Buffer<f32>,
    input_a_grad: &mut Buffer<f32>,
    shape_a: Shape,
    input_b: &Buffer<i32>,
    shape_b: Shape,
    nnz: usize,
    _input_c: Option<&Buffer<f32>>,
    input_c_grad: Option<&mut Buffer<f32>>,
    input_c_buckets: Option<(&Buffer<i32>, usize)>,
    outputs: &Buffer<f32>,
    output_grad: &Buffer<f32>,
) -> OperationResult {
    let shape_o = shape_a * shape_b;

    let (stride, offset) = if let Some(b) = stride { (2, if b { shape_a.rows() } else { 0 }) } else { (1, 0) };

    assert_eq!(shape_b.cols(), 1);
    assert_eq!(shape_o.cols(), 1);
    if shape_a.size() > input_a.size()
        || shape_a.size() > input_a_grad.size()
        || batch_size * nnz > input_b.size()
        || batch_size * shape_o.size() > outputs.size()
        || batch_size * shape_o.size() * stride > output_grad.size()
    {
        return Err(OperationError::IndexOutOfBounds);
    }

    let (c_ptr, s_ptr) = if let Some(grad) = input_c_grad {
        let s_ptr = if let Some((buckets, num_buckets)) = input_c_buckets {
            if num_buckets * shape_o.size() > grad.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            if batch_size > buckets.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            buckets.ptr()
        } else {
            if shape_o.size() > grad.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            std::ptr::null()
        };

        (grad.mut_ptr(), s_ptr)
    } else {
        (std::ptr::null_mut(), std::ptr::null())
    };

    unsafe {
        ops::sparse_affine_backward(
            activation as i32,
            stride,
            nnz,
            shape_a.rows(),
            shape_a.cols(),
            batch_size,
            input_b.ptr(),
            s_ptr,
            outputs.ptr().add(offset),
            output_grad.ptr().add(offset),
            input_a_grad.mut_ptr(),
            c_ptr,
        );
    }

    Ok(())
}
