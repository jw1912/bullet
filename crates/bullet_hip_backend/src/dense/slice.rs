use bullet_core::device::DeviceBuffer;

use crate::backend::{blas, Buffer};

#[allow(clippy::too_many_arguments)]
pub fn copy_or_add_strided(
    rows: usize,
    cols: usize,
    input: &Buffer<f32>,
    input_offset: usize,
    input_stride: usize,
    output: &mut Buffer<f32>,
    output_offset: usize,
    output_stride: usize,
    add: bool,
) {
    assert!(cols > 0);
    assert!(rows > 0);
    assert!(input_offset <= input_stride);
    assert!(output_offset <= output_stride);
    assert!(cols * input_stride <= input.size());
    assert!(cols * output_stride <= output.size());
    assert!(rows <= output_stride);

    unsafe {
        blas::copy_strided(
            input.device().as_ref(),
            rows,
            cols,
            input_stride,
            input.ptr().add(input_offset),
            output_stride,
            output.mut_ptr().add(output_offset),
            add,
        );
    }
}
