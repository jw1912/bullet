use crate::{backend::ops, DenseMatrix};

pub fn abs_power_error(power: f32, input_a: &DenseMatrix, input_b: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(input_a.shape, input_b.shape);
    output.reshape_if_needed(input_a.shape);

    unsafe {
        ops::powerError(input_a.shape.size(), input_a.buf.ptr(), input_b.buf.ptr(), output.buf.mut_ptr(), power);
    }
}

pub fn backprop_abs_power_error(
    power: f32,
    input_a: &DenseMatrix,
    input_a_grad: Option<&mut DenseMatrix>,
    input_b: &DenseMatrix,
    input_b_grad: Option<&mut DenseMatrix>,
    output_grad: &DenseMatrix,
) {
    if let Some(grd) = input_a_grad {
        backprop_abs_power_error_single(power, input_a, input_b, output_grad, grd);
    }

    if let Some(grd) = input_b_grad {
        backprop_abs_power_error_single(power, input_b, input_a, output_grad, grd);
    }
}

fn backprop_abs_power_error_single(
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    #[test]
    fn abs_power_error_basic() {
        abs_power_error_custom([-1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 2.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]);
    }

    #[test]
    fn abs_power_error_rev() {
        abs_power_error_custom([1.0, 2.0, 3.0], [-1.0, 4.0, 2.0], [2.0, 2.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]);
    }

    fn abs_power_error_custom(
        input_a: [f32; 3],
        input_b: [f32; 3],
        outputs: [f32; 3],
        grad_a: [f32; 3],
        grad_b: [f32; 3],
    ) {
        let device = Arc::new(ExecutionContext::default());
        let shape = Shape::new(3, 1);

        let mut input1 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2 = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input1_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input2_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_from_slice(shape, &input_a);

            input2.load_from_slice(shape, &input_b);

            assert_eq!(input1.shape(), shape);
            assert_eq!(input2.shape(), shape);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // power error
        {
            abs_power_error(1.0, &input1, &input2, &mut output);

            util::panic_if_device_error("Failed to add matrices!");

            assert_eq!(output.shape(), shape);

            let mut buf = [0.0; 3];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, outputs);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop add
        {
            output.load_from_slice(shape, &[1.0, 1.0, 1.0]);

            util::panic_if_device_error("Failed to load data from CPU!");

            backprop_abs_power_error(1.0, &input1, Some(&mut input1_grad), &input2, Some(&mut input2_grad), &output);

            util::panic_if_device_error("Failed to backprop addition!");

            assert_eq!(input1_grad.shape(), shape);
            assert_eq!(input2_grad.shape(), shape);

            let mut grad1 = [0.0; 3];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, grad_a);

            let mut grad2 = [0.0; 3];
            input2_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, grad_b);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
