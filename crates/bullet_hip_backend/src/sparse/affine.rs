use bullet_core::{device::DeviceBuffer, shape::Shape};

use crate::backend::{ops, Buffer};

#[allow(clippy::too_many_arguments)]
pub fn sparse_affine(
    batch_size: usize,
    input_a: &Buffer<f32>,
    shape_a: Shape,
    input_b: &Buffer<i32>,
    shape_b: Shape,
    nnz: usize,
    input_c: Option<&Buffer<f32>>,
    output: &mut Buffer<f32>,
) {
    let shape_o = shape_a * shape_b;

    assert!(shape_a.size() <= input_a.size());
    assert!(batch_size * nnz <= input_b.size());
    assert!(batch_size * shape_o.size() <= output.size());

    if let Some(c) = input_c {
        assert!(shape_o.size() <= c.size());
    }

    unsafe {
        ops::sparseAffineForward(
            batch_size,
            nnz,
            shape_o.rows(),
            input_a.ptr(),
            input_c.map(|c| c.ptr()).unwrap_or(std::ptr::null()),
            input_b.ptr(),
            output.mut_ptr(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn backprop_sparse_affine(
    batch_size: usize,
    input_a: &Buffer<f32>,
    input_a_grad: &mut Buffer<f32>,
    shape_a: Shape,
    input_b: &Buffer<i32>,
    shape_b: Shape,
    nnz: usize,
    _input_c: Option<&Buffer<f32>>,
    input_c_grad: Option<&mut Buffer<f32>>,
    outputs: &Buffer<f32>,
    output_grad: &Buffer<f32>,
) {
    let shape_o = shape_a * shape_b;

    assert_eq!(shape_b.cols(), 1);
    assert_eq!(shape_o.cols(), 1);
    assert!(shape_a.size() <= input_a.size());
    assert!(shape_a.size() <= input_a_grad.size());
    assert!(batch_size * nnz <= input_b.size());
    assert!(batch_size * shape_o.size() <= outputs.size());
    assert!(batch_size * shape_o.size() <= output_grad.size());

    let c_ptr = if let Some(grad) = input_c_grad { grad.mut_ptr() } else { std::ptr::null_mut() };

    unsafe {
        ops::sparseAffineBackward(
            batch_size,
            nnz,
            shape_o.rows(),
            input_a_grad.mut_ptr(),
            c_ptr,
            input_b.ptr(),
            outputs.ptr(),
            output_grad.ptr(),
        );
    }
}
