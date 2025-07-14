use bullet_core::device::DeviceBuffer;

use crate::{
    backend::{blas, util::catch_cublas, Buffer},
    DeviceError,
};

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
) -> Result<(), DeviceError> {
    assert!(cols > 0);
    assert!(rows > 0);

    if input_offset > input_stride
        || output_offset > output_stride
        || cols * input_stride > input.size()
        || cols * output_stride > output.size()
        || rows > output_stride
    {
        return Err(DeviceError::ExpectedIllegalAddressAccess);
    }

    unsafe {
        let err = blas::copy_strided(
            input.device().as_ref(),
            rows,
            cols,
            input_stride,
            input.ptr().add(input_offset),
            output_stride,
            output.mut_ptr().add(output_offset),
            add,
        );

        catch_cublas(err)
    }
}
