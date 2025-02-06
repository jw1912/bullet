use crate::{backend::ops, DenseMatrix};

pub fn abs_power_error(power: f32, input_a: &DenseMatrix, input_b: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(input_a.shape, input_b.shape);
    output.reshape_if_needed(input_a.shape);

    unsafe {
        ops::powerError(input_a.shape.size(), input_a.buf.ptr(), input_b.buf.ptr(), output.buf.mut_ptr(), power);
    }
}

pub fn backprop_abs_power_error_single(
    power: f32,
    input_a: &DenseMatrix,
    input_b: &DenseMatrix,
    output_grad: &DenseMatrix,
    input_a_grad: &mut DenseMatrix,
) {
    assert_eq!(input_a.shape, input_b.shape);
    assert_eq!(output_grad.shape, input_a.shape);
    input_a_grad.reshape_if_needed(input_a.shape);

    unsafe {
        ops::backpropPowerError(
            input_a.shape.size(),
            input_a.buf.ptr(),
            input_b.buf.ptr(),
            output_grad.buf.ptr(),
            input_a_grad.buf.mut_ptr(),
            power,
        );
    }
}
