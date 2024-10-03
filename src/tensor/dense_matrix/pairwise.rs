use crate::{backend::ops, tensor::Shape};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn pairwise(input: &Self, output: &mut Self) {
        assert_eq!(input.shape.rows() % 2, 0);

        let shape = Shape::new(input.shape.rows() / 2, input.shape.cols());
        output.reshape_if_needed(shape);

        unsafe {
            ops::pairwiseMul(
                shape.cols(),
                shape.rows(),
                input.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }

    pub fn backprop_pairwise(input: &Self, output_grad: &Self, input_grad: &mut Self) {
        let shape = Shape::new(output_grad.shape.rows() * 2, output_grad.shape.cols());
        assert_eq!(shape, input.shape);
        input_grad.reshape_if_needed(shape);

        unsafe {
            ops::backpropPairwiseMul(
                shape.cols(),
                output_grad.shape.rows(),
                input.buf.ptr(),
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

        DenseMatrix::pairwise(&input, &mut output);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, shape2);

        let mut buf = [0.0; 4];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [-2.0, 1.0, 4.0, -4.0]);

        util::panic_if_device_error("Failed to write data to CPU!");

        DenseMatrix::backprop_pairwise(&input, &output, &mut input_grad);

        util::panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input_grad.shape, input.shape);

        let mut buf = [0.0; 8];
        input_grad.write_to_slice(&mut buf);
        assert_eq!(buf, [-4.0, 2.0, 2.0, 0.5, 8.0, -8.0, 8.0, 8.0]);

        util::panic_if_device_error("Failed to write data to CPU!");
    }
}
