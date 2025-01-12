use crate::tensor::{backend::ops, Shape};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn pairwise(input: &Self, output: &mut Self, post_concat: bool) {
        let mut rows = input.shape.rows();
        let mut cols = input.shape.cols();

        if post_concat {
            rows /= 2;
            cols *= 2;
        }

        assert_eq!(rows % 2, 0);

        let shape = Shape::new(input.shape.rows() / 2, input.shape.cols());
        output.reshape_if_needed(shape);

        unsafe {
            ops::pairwiseMul(cols, rows / 2, input.buf.ptr(), output.buf.mut_ptr());
        }
    }

    pub fn backprop_pairwise(input: &Self, output_grad: &Self, input_grad: &mut Self, post_concat: bool) {
        let mut rows = input.shape.rows();
        let mut cols = input.shape.cols();

        if post_concat {
            rows /= 2;
            cols *= 2;
        }

        input_grad.reshape_if_needed(input.shape);

        unsafe {
            ops::backpropPairwiseMul(cols, rows / 2, input.buf.ptr(), output_grad.buf.ptr(), input_grad.buf.mut_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::util, Shape};

    #[test]
    fn pairwise() {
        let shape1 = Shape::new(4, 2);
        let shape2 = Shape::new(2, 2);

        let mut input = DenseMatrix::default();
        let mut input_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape1, &[-1.0, 0.5, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0]);
        assert_eq!(input.shape(), shape1);

        util::panic_if_device_error("Failed to load data from CPU!");

        DenseMatrix::pairwise(&input, &mut output, false);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape2);

        let mut buf = [0.0; 4];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [-2.0, 1.0, 4.0, -4.0]);

        util::panic_if_device_error("Failed to write data to CPU!");

        DenseMatrix::backprop_pairwise(&input, &output, &mut input_grad, false);

        util::panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input_grad.shape, input.shape);

        let mut buf = [0.0; 8];
        input_grad.write_to_slice(&mut buf);
        assert_eq!(buf, [-4.0, 2.0, 2.0, 0.5, 8.0, -8.0, 8.0, 8.0]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn pairwise_post_concat() {
        let shape1 = Shape::new(4, 2);
        let shape2 = Shape::new(2, 2);

        let mut input = DenseMatrix::default();
        let mut input_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape1, &[-1.0, 0.5, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0]);
        assert_eq!(input.shape(), shape1);

        util::panic_if_device_error("Failed to load data from CPU!");

        DenseMatrix::pairwise(&input, &mut output, true);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape2);

        let mut buf = [0.0; 4];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [-0.5, 4.0, -4.0, 4.0]);

        util::panic_if_device_error("Failed to write data to CPU!");

        DenseMatrix::backprop_pairwise(&input, &output, &mut input_grad, true);

        util::panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input_grad.shape, input.shape);

        let mut buf = [0.0; 8];
        input_grad.write_to_slice(&mut buf);
        assert_eq!(buf, [-0.25, 0.5, 8.0, 8.0, 8.0, -8.0, 8.0, 8.0]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }
}
