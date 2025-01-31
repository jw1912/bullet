use crate::{
    tensor::{backend::ops, DenseMatrix},
    Shape,
};

use super::SparseMatrix;

impl SparseMatrix {
    pub fn select(input: &DenseMatrix, indices: &Self, output: &mut DenseMatrix) {
        let rows = input.shape.rows();
        let cols = input.shape.cols();
        let buckets = indices.shape.rows();

        assert_eq!(cols, indices.shape.cols());
        assert_eq!(indices.max_active, 1);
        assert_eq!(rows % buckets, 0, "Cannot divide vector evenly among buckets!");

        let output_rows = rows / buckets;
        let shape = Shape::new(output_rows, cols);
        output.reshape_if_needed(shape);

        unsafe {
            ops::selectForward(cols, rows, output_rows, indices.buf.ptr(), input.buf.ptr(), output.buf.mut_ptr());
        }
    }

    pub fn select_backprop(
        input: &DenseMatrix,
        indices: &Self,
        output_grad: &DenseMatrix,
        input_grad: &mut DenseMatrix,
    ) {
        let rows = input.shape.rows();
        let cols = input.shape.cols();
        let buckets = indices.shape.rows();

        assert_eq!(cols, indices.shape.cols());
        assert_eq!(cols, input.shape.cols());
        assert_eq!(indices.max_active, 1);
        assert_eq!(rows % buckets, 0, "Cannot divide vector evenly among buckets!");
        assert_eq!(rows / buckets, output_grad.shape.rows());

        input_grad.reshape_if_needed(input.shape);

        unsafe {
            ops::selectBackprop(
                cols,
                rows,
                rows / buckets,
                indices.buf.ptr(),
                output_grad.buf.ptr(),
                input_grad.buf.mut_ptr(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::util, Shape};

    #[test]
    fn select() {
        let shape1 = Shape::new(8, 3);
        let shape2 = Shape::new(4, 3);

        let mut input1 = DenseMatrix::default();
        let mut input2 = SparseMatrix::default();
        let mut input1_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_from_slice(
                shape1,
                &[
                    -1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 0.0, -3.0, -1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 0.0, -3.0, -1.0, 4.0,
                    2.0, -2.0, 0.0, -3.0, 0.0, -3.0,
                ],
            );

            unsafe {
                input2.load_from_slice(shape2, 1, &[0, 1, 2]);
            }

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // sparse linear
        {
            SparseMatrix::select(&input1, &input2, &mut output);

            util::panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::new(2, 3));

            let mut buf = [0.0; 6];
            output.write_to_slice(&mut buf);
            assert_eq!(buf, [-1.0, 4.0, 2.0, -2.0, 0.0, -3.0]);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop sparse linear
        {
            SparseMatrix::select_backprop(&input1, &input2, &output, &mut input1_grad);

            util::panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1_grad.shape(), shape1);

            let mut grad1 = [0.0; 24];
            input1_grad.write_to_slice(&mut grad1);
            assert_eq!(
                grad1,
                [
                    -1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, -3.0, 0.0, 0.0,
                ],
            );

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
