use bullet_core::device::DeviceBuffer;

use crate::{
    backend::{blas, ops},
    Buffer, DenseMatrix, Shape,
};

fn softmax_across_batch(input: &DenseMatrix, output: &mut DenseMatrix) {
    assert!(input.shape.size() > 0);

    output.reshape_if_needed(input.shape);

    unsafe {
        ops::softmax_across_columns(
            input.shape.without_batch_size().size(),
            input.shape.batch_size().unwrap_or(1),
            input.buf.ptr(),
            output.buf.mut_ptr(),
        );
    }
}

fn crossentropy(pred: &DenseMatrix, target: &DenseMatrix, output: &mut DenseMatrix) {
    assert_eq!(pred.shape, target.shape);

    output.reshape_if_needed(pred.shape);

    unsafe {
        ops::crossentropy(pred.shape.size(), pred.buf.ptr(), target.buf.ptr(), output.buf.mut_ptr());
    }
}

pub fn softmax_crossentropy_loss(
    ones: &Buffer<f32>,
    input: &DenseMatrix,
    target: &DenseMatrix,
    output: &mut DenseMatrix,
    softmaxed: &mut DenseMatrix,
    individual_losses: &mut DenseMatrix,
) {
    assert_eq!(input.shape, target.shape);

    assert!(input.shape.size() <= ones.size());

    softmax_across_batch(input, softmaxed);

    crossentropy(softmaxed, target, individual_losses);

    output.reshape_if_needed(Shape::new(1, 1));

    unsafe {
        blas::reduce_add_cols(
            input.buf.device().as_ref(),
            1,
            input.shape.size(),
            ones.ptr(),
            individual_losses.buf.ptr(),
            output.buf.mut_ptr(),
            1.0,
            false,
        );
    }
}

pub fn backprop_softmax_crossentropy_loss(
    softmaxed: &DenseMatrix,
    target: &DenseMatrix,
    output_grad: &DenseMatrix,
    input_grad: &mut DenseMatrix,
) {
    assert_eq!(softmaxed.shape, target.shape);
    assert_eq!(output_grad.shape, Shape::new(1, 1));

    input_grad.reshape_if_needed(softmaxed.shape);

    unsafe {
        ops::backprop_softmax_cross_entropy(
            softmaxed.shape.size(),
            softmaxed.buf.ptr(),
            target.buf.ptr(),
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
    fn softmax() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(2, 2, 3);

        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape, &[2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(input.shape(), shape);

        util::panic_if_device_error("Failed to load data from CPU!");

        softmax_across_batch(&input, &mut output);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape);

        let mut buf = [0.0; 12];
        output.write_to_slice(&mut buf);

        assert_eq!(buf, [0.25; 12]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn softmax_crossentropy() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(2, 2, 3);

        let mut ones = Buffer::new(device.clone(), shape.size());
        ones.load_from_slice(&vec![1.0; shape.size()]);

        let mut pred = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut target = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut softmaxed = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut individual_losses = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        pred.load_from_slice(shape, &[1.0, 2.0, 1.0, 2.0, -4.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(pred.shape(), shape);

        target.load_from_slice(shape, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(target.shape(), shape);

        util::panic_if_device_error("Failed to load data from CPU!");

        softmax_crossentropy_loss(&ones, &pred, &target, &mut output, &mut softmaxed, &mut individual_losses);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, Shape::new(1, 1));

        let mut buf = [0.0];
        output.write_to_slice(&mut buf);

        assert!((buf[0] - 3.865).abs() < 0.001);

        util::panic_if_device_error("Failed to load data from CPU!");

        pred.set_zero();
        output.load_from_slice(Shape::new(1, 1), &[1.0]);

        backprop_softmax_crossentropy_loss(&softmaxed, &target, &output, &mut pred);

        util::panic_if_device_error("Failed to calculate activation!");

        let mut buf = [0.0; 12];
        pred.write_to_slice(&mut buf);

        let expected =
            [-0.8655, 0.3655, 0.1345, 0.3655, 0.0163, -0.6721, 0.3279, 0.3279, 0.1749, 0.1749, -0.5246, 0.1749];

        let mut total = 0.0;
        for (p, e) in buf.iter().zip(expected.iter()) {
            total += (p - e).abs();
        }

        assert!(total < 0.01);

        util::panic_if_device_error("Failed to write data to CPU!");
    }
}
