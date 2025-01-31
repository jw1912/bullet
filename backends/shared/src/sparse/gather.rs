use crate::{
    tensor::{backend::ops, DenseMatrix},
    Shape,
};

use super::SparseMatrix;

impl SparseMatrix {
    pub fn gather(inputs: &DenseMatrix, indices: &Self, outputs: &mut DenseMatrix) {
        assert_eq!(indices.shape.cols(), 1);
        assert_eq!(indices.shape.rows(), indices.max_active);

        outputs.reshape_if_needed(Shape::new(indices.shape.rows(), inputs.shape.cols()));
        outputs.set_zero();

        unsafe {
            ops::gather(
                inputs.shape.rows(),
                outputs.shape.rows(),
                outputs.shape.cols(),
                inputs.buf.ptr(),
                indices.buf.ptr(),
                outputs.buf.mut_ptr(),
            );
        }
    }

    pub fn backprop_gather(
        output_grads: &DenseMatrix,
        indices: &Self,
        inputs: &DenseMatrix,
        input_grads: &mut DenseMatrix,
    ) {
        assert_eq!(indices.shape.cols(), 1);
        assert_eq!(output_grads.shape.cols(), inputs.shape.cols());
        assert_eq!(output_grads.shape.rows(), indices.shape.rows());

        input_grads.reshape_if_needed(inputs.shape);

        unsafe {
            ops::gather_backprop(
                inputs.shape.rows(),
                output_grads.shape.rows(),
                output_grads.shape.cols(),
                output_grads.buf.ptr(),
                indices.buf.ptr(),
                input_grads.buf.mut_ptr(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn gather() {
        let shape1 = Shape::new(3, 3);

        let mut inputs = DenseMatrix::default();
        let mut output = DenseMatrix::default();
        let mut indices = SparseMatrix::default();
        let mut input_grads = DenseMatrix::default();

        inputs.load_from_slice(shape1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        unsafe {
            indices.load_from_slice(Shape::new(5, 1), 5, &[-1, 0, 2, 1, 2]);
        }

        SparseMatrix::gather(&inputs, &indices, &mut output);

        let mut buf = [0.0; 15];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [0.0, 1.0, 3.0, 2.0, 3.0, 0.0, 4.0, 6.0, 5.0, 6.0, 0.0, 7.0, 9.0, 8.0, 9.0]);

        SparseMatrix::backprop_gather(&output, &indices, &inputs, &mut input_grads);

        let mut buf = [0.0; 9];
        input_grads.write_to_slice(&mut buf);
        assert_eq!(buf, [1.0, 2.0, 6.0, 4.0, 5.0, 12.0, 7.0, 8.0, 18.0,]);
    }
}
