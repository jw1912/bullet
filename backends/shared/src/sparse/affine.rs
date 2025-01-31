use crate::tensor::{backend::ops, DenseMatrix};

use super::SparseMatrix;

impl SparseMatrix {
    pub fn affine(input_a: &DenseMatrix, input_b: &Self, input_c: Option<&DenseMatrix>, output: &mut DenseMatrix) {
        let output_shape = input_a.shape * input_b.shape;
        output.reshape_if_needed(output_shape);

        if let Some(c) = input_c {
            assert_eq!(c.shape.rows(), output_shape.rows());
            assert_eq!(c.shape.cols(), 1);
        }

        unsafe {
            ops::sparseAffineForward(
                input_b.shape.cols(),
                input_b.max_active,
                output.shape().rows(),
                input_a.buf.ptr(),
                input_c.map(|c| c.buf.ptr()).unwrap_or(std::ptr::null()),
                input_b.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }

    pub fn backprop_affine(
        input_a: &DenseMatrix,
        input_a_grad: &mut DenseMatrix,
        input_b: &Self,
        input_c: Option<&DenseMatrix>,
        input_c_grad: Option<&mut DenseMatrix>,
        outputs: &DenseMatrix,
        output_grad: &DenseMatrix,
    ) {
        input_a_grad.reshape_if_needed(input_a.shape());

        let c_ptr = if let Some(grad) = input_c_grad {
            grad.reshape_if_needed(input_c.unwrap().shape);
            grad.buf.mut_ptr()
        } else {
            std::ptr::null_mut()
        };

        unsafe {
            ops::sparseAffineBackward(
                input_b.shape.cols(),
                input_b.max_active,
                output_grad.shape.rows(),
                input_a_grad.buf.mut_ptr(),
                c_ptr,
                input_b.buf.ptr(),
                outputs.buf.ptr(),
                output_grad.buf.ptr(),
            );
        }
    }

    pub fn linear(input_a: &DenseMatrix, input_b: &Self, output: &mut DenseMatrix) {
        Self::affine(input_a, input_b, None, output);
    }

    pub fn backprop_linear(
        input_a: &DenseMatrix,
        input_a_grad: &mut DenseMatrix,
        input_b: &Self,
        outputs: &DenseMatrix,
        output_grad: &DenseMatrix,
    ) {
        Self::backprop_affine(input_a, input_a_grad, input_b, None, None, outputs, output_grad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::util, Shape};

    #[test]
    fn linear() {
        let shape1 = Shape::new(2, 3);
        let shape2 = Shape::new(3, 3);

        let mut input1 = DenseMatrix::default();
        let mut input2 = SparseMatrix::default();
        let mut input1_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_from_slice(
                shape1,
                // [ -1.0,  2.0,  0.0 ]
                // [  4.0, -2.0, -3.0 ]
                &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0],
            );

            unsafe {
                input2.load_from_slice(shape2, 2, &[0, -1, 1, 2, -1, -1]);
            }

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // sparse linear
        {
            SparseMatrix::linear(&input1, &input2, &mut output);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new(2, 3));

            let mut buf = [0.0; 6];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [-1.0, 4.0, 2.0, -5.0, 0.0, 0.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop sparse linear
        {
            SparseMatrix::backprop_linear(&input1, &mut input1_grad, &input2, &output, &output);

            util::panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1_grad.shape(), shape1);

            let mut grad1 = [0.0; 6];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, [-1.0, 4.0, 2.0, -5.0, 2.0, -5.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }

    #[test]
    fn aligned_matches_unaligned() {
        let input_shape = Shape::new(4, 3);
        let shape1 = Shape::new(256, 4);
        let mut inputs1 = SparseMatrix::default();
        let mut weights1 = DenseMatrix::default();

        println!("wha");

        unsafe {
            inputs1.load_from_slice(input_shape, 2, &[0, -1, 1, 2, -1, -1]);
        }
        weights1.load_from_slice(shape1, &vec![1.0; 256 * 4]);

        let shape2 = Shape::new(255, 4);
        let mut inputs2 = SparseMatrix::default();
        let mut weights2 = DenseMatrix::default();

        println!("yeah");

        unsafe {
            inputs2.load_from_slice(input_shape, 2, &[0, -1, 1, 2, -1, -1]);
        }
        weights2.load_from_slice(shape2, &[1.0; 255 * 4]);

        let mut output = DenseMatrix::default();

        println!("loaded");

        util::panic_if_device_error("Failed to initialise matrices!");

        let mut buf2 = vec![0.0; 255 * 3];
        SparseMatrix::linear(&weights2, &inputs2, &mut output);
        output.write_to_slice(&mut buf2);

        let mut buf = vec![0.0; 256 * 3];
        SparseMatrix::linear(&weights1, &inputs1, &mut output);
        output.write_to_slice(&mut buf);

        for i in 0..3 {
            for j in 0..255 {
                assert_eq!(buf[256 * i + j], buf2[255 * i + j]);
            }
        }
    }
}
