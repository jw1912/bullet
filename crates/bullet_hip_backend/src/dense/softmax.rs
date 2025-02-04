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
    use crate::ExecutionContext;
    use bullet_core::device::Device;

    #[test]
    fn softmax() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(2, 2, 3);

        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        device.panic_if_device_error("Failed to initialise matrices!");

        input.load_from_slice(shape, &[2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(input.shape(), shape);

        device.panic_if_device_error("Failed to load data from CPU!");

        softmax_across_batch(&input, &mut output);

        device.panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape(), shape);

        let mut buf = [0.0; 12];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [0.25; 12]);

        device.panic_if_device_error("Failed to write data to CPU!");
    }
}
