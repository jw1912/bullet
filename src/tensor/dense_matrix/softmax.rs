use crate::{backend::ops, ExecutionContext, Shape};

use super::DenseMatrix;

impl DenseMatrix {
    fn softmax_across_columns(input: &Self, output: &mut Self) {
        assert!(input.shape.size() > 0);

        output.reshape_if_needed(input.shape);

        unsafe {
            ops::softmax_across_columns(input.shape.rows(), input.shape.cols(), input.buf.ptr(), output.buf.mut_ptr());
        }
    }

    fn crossentropy(pred: &Self, target: &Self, output: &mut Self) {
        assert_eq!(pred.shape, target.shape);

        output.reshape_if_needed(pred.shape);

        unsafe {
            ops::crossentropy(pred.shape.size(), pred.buf.ptr(), target.buf.ptr(), output.buf.mut_ptr());
        }
    }

    pub fn softmax_crossentropy_loss(
        ctx: &mut ExecutionContext,
        input: &Self,
        target: &Self,
        output: &mut Self,
        softmaxed: &mut Self,
        individual_losses: &mut Self,
    ) {
        assert_eq!(input.shape, target.shape);

        Self::softmax_across_columns(input, softmaxed);

        Self::crossentropy(softmaxed, target, individual_losses);

        output.reshape_if_needed(Shape::new(1, 1));

        unsafe {
            ops::reduce_add_cols(ctx, 1, input.shape.size(), individual_losses.buf.ptr(), output.buf.mut_ptr(), false);
        }
    }

    pub fn backprop_softmax_crossentropy_loss(
        softmaxed: &Self,
        target: &Self,
        output_grad: &Self,
        input_grad: &mut Self,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::util, tensor::Shape};

    #[test]
    fn softmax() {
        let shape = Shape::new(4, 3);

        let mut input = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape, &[2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(input.shape(), shape);

        util::panic_if_device_error("Failed to load data from CPU!");

        DenseMatrix::softmax_across_columns(&input, &mut output);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape);

        let mut buf = [0.0; 12];
        output.write_to_slice(&mut buf);

        assert_eq!(buf, [0.25; 12]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn softmax_crossentropy() {
        let mut ctx = ExecutionContext::default();

        let shape = Shape::new(4, 3);

        let mut pred = DenseMatrix::default();
        let mut target = DenseMatrix::default();
        let mut softmaxed = DenseMatrix::default();
        let mut individual_losses = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        pred.load_from_slice(shape, &[1.0, 2.0, 1.0, 2.0, -4.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(pred.shape(), shape);

        target.load_from_slice(shape, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(target.shape(), shape);

        util::panic_if_device_error("Failed to load data from CPU!");

        DenseMatrix::softmax_crossentropy_loss(
            &mut ctx,
            &pred,
            &target,
            &mut output,
            &mut softmaxed,
            &mut individual_losses,
        );

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, Shape::new(1, 1));

        let mut buf = [0.0];
        output.write_to_slice(&mut buf);

        assert!((buf[0] - 3.865).abs() < 0.001);

        util::panic_if_device_error("Failed to load data from CPU!");

        pred.set_zero();
        output.load_from_slice(Shape::new(1, 1), &[1.0]);

        DenseMatrix::backprop_softmax_crossentropy_loss(&softmaxed, &target, &output, &mut pred);

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
