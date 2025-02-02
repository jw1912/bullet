use crate::{backend::ops, DenseMatrix, Shape};

pub fn pairwise(input: &DenseMatrix, output: &mut DenseMatrix, post_concat: bool) {
    assert_eq!(input.shape.cols(), 1);
    let mut rows = input.shape.rows();
    let mut batch_size = input.shape.batch_size().unwrap_or(1);

    let shape = Shape::new_batched(rows / 2, 1, batch_size);

    if post_concat {
        rows /= 2;
        batch_size *= 2;
    }

    assert_eq!(rows % 2, 0);
    output.reshape_if_needed(shape);

    unsafe {
        ops::pairwiseMul(batch_size, rows / 2, input.buf.ptr(), output.buf.mut_ptr());
    }
}

pub fn backprop_pairwise(
    input: &DenseMatrix,
    output_grad: &DenseMatrix,
    input_grad: &mut DenseMatrix,
    post_concat: bool,
) {
    assert_eq!(input.shape.cols(), 1);
    let mut rows = input.shape.rows();
    let mut batch_size = input.shape.batch_size().unwrap_or(1);

    if post_concat {
        rows /= 2;
        batch_size *= 2;
    }

    input_grad.reshape_if_needed(input.shape);

    unsafe {
        ops::backpropPairwiseMul(
            batch_size,
            rows / 2,
            input.buf.ptr(),
            output_grad.buf.ptr(),
            input_grad.buf.mut_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    #[test]
    fn pairwise_no_concat() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(4, 1, 2);
        let shape2 = Shape::new_batched(2, 1, 2);

        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape1, &[-1.0, 0.5, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0]);
        assert_eq!(input.shape(), shape1);

        util::panic_if_device_error("Failed to load data from CPU!");

        pairwise(&input, &mut output, false);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape2);

        let mut buf = [0.0; 4];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [-2.0, 1.0, 4.0, -4.0]);

        util::panic_if_device_error("Failed to write data to CPU!");

        backprop_pairwise(&input, &output, &mut input_grad, false);

        util::panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input_grad.shape, input.shape);

        let mut buf = [0.0; 8];
        input_grad.write_to_slice(&mut buf);
        assert_eq!(buf, [-4.0, 2.0, 2.0, 0.5, 8.0, -8.0, 8.0, 8.0]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn pairwise_post_concat() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(4, 1, 2);
        let shape2 = Shape::new_batched(2, 1, 2);

        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape1, &[-1.0, 0.5, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0]);
        assert_eq!(input.shape(), shape1);

        util::panic_if_device_error("Failed to load data from CPU!");

        pairwise(&input, &mut output, true);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape2);

        let mut buf = [0.0; 4];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [-0.5, 4.0, -4.0, 4.0]);

        util::panic_if_device_error("Failed to write data to CPU!");

        backprop_pairwise(&input, &output, &mut input_grad, true);

        util::panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input_grad.shape, input.shape);

        let mut buf = [0.0; 8];
        input_grad.write_to_slice(&mut buf);
        assert_eq!(buf, [-0.25, 0.5, 8.0, 8.0, 8.0, -8.0, 8.0, 8.0]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }
}
