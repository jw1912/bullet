use crate::{backend::{ops, ExecutionContext}, Tensor};

use super::{RawTensor, RawTensorValues};

impl RawTensor {
    /// Apply a linear transform to a batch of vectors.
    pub fn linear_transform_forward_batched(
        ctx: &ExecutionContext,
        input_transform: &Self,
        input_batch: &Self,
        output_batch: &mut Self,
    ) {
        let output_dims = input_transform.shape * input_batch.shape;

        assert_eq!(output_dims, output_batch.shape);
        output_batch.resize_if_necessary(input_batch.len);
        output_batch.len = input_batch.len;

        assert_eq!(
            input_transform.len,
            1,
            "Must contain only 1 transform!"
        );

        unsafe {
            splat_mul_matrix_vector(
                ctx,
                input_transform,
                input_batch,
                output_batch,
            );
        }
    }

    /// Apply the backwards step of a linear transform to a batch.
    pub fn linear_transform_backward_batched(
        ctx: &ExecutionContext,
        inputs: &mut [&mut Tensor],
        output_batch_grad: &Self,
    ) {
        let input_batch_dims = inputs[1].values.shape;
        let output_dims = inputs[0].values.shape * input_batch_dims;

        assert_eq!(input_batch_dims.cols(), 1);
        assert_eq!(output_dims, output_batch_grad.shape);

        unsafe {
            reduce_add_mul_vector_vectort(
                ctx,
                output_batch_grad,
                &inputs[1].values,
                inputs[0].gradients.as_mut().unwrap(),
            );
        }

        if let Some(grd) = inputs[1].gradients.as_mut() {
            grd.resize_if_necessary(output_batch_grad.len);
            grd.len = output_batch_grad.len;

            unsafe {
                splat_mul_matrixt_vector(
                    ctx,
                    &inputs[0].values,
                    output_batch_grad,
                    grd,
                );
            }

        }
    }
}

unsafe fn splat_mul_matrix_vector(
    ctx: &ExecutionContext,
    input_a: &RawTensor,
    input_b: &RawTensor,
    dest: &mut RawTensor,
) {
    let batch_size = input_b.len;

    if input_b.shape.cols() != 1 {
        unimplemented!("Currently support only vectors on the RHS of matmuls!")
    }

    if let RawTensorValues::DenseFloats(dst) = &mut dest.values {
        match (&input_a.values, &input_b.values) {
            (RawTensorValues::DenseFloats(src_a), RawTensorValues::DenseFloats(src_b)) => {
                ops::splat_mul_matrix_vector(
                    ctx,
                    input_a.shape.cols(),
                    input_a.shape.rows(),
                    src_a.ptr(),
                    src_b.ptr(),
                    dst.ptr(),
                    batch_size,
                    false,
                );
            }
            _ => unimplemented!(),
        }
    }
}

unsafe fn splat_mul_matrixt_vector(
    ctx: &ExecutionContext,
    input_a: &RawTensor,
    input_b: &RawTensor,
    dest: &mut RawTensor,
) {
    let batch_size = input_b.len;

    if input_b.shape.cols() != 1 {
        unimplemented!("Currently support only vectors on the RHS of matmuls!")
    }

    if let RawTensorValues::DenseFloats(dst) = &mut dest.values {
        match (&input_a.values, &input_b.values) {
            (RawTensorValues::DenseFloats(src_a), RawTensorValues::DenseFloats(src_b)) => {
                ops::splat_mul_matrixt_vector(
                    ctx,
                    input_a.shape.cols(),
                    input_a.shape.rows(),
                    src_a.ptr(),
                    src_b.ptr(),
                    dst.ptr(),
                    batch_size,
                    true,
                );
            }
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot backprop gradient into non-dense tensors!")
    }
}

unsafe fn reduce_add_mul_vector_vectort(
    ctx: &ExecutionContext,
    input_a: &RawTensor,
    input_b: &RawTensor,
    dest: &mut RawTensor,
) {
    let batch_size = input_a.len;
    assert_eq!(batch_size, input_b.len);

    if let RawTensorValues::DenseFloats(dst) = &mut dest.values {
        match (&input_a.values, &input_b.values) {
            (RawTensorValues::DenseFloats(src_a), RawTensorValues::DenseFloats(src_b)) => {
                ops::reduce_add_mul_vector_vectort(
                    ctx,
                    dest.shape.cols(),
                    dest.shape.rows(),
                    src_a.ptr(),
                    src_b.ptr(),
                    dst.ptr(),
                    batch_size,
                    true,
                );
            }
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot backprop gradient into non-dense tensors!")
    }
}
