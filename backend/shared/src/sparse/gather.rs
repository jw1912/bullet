use crate::{backend::ops, DenseMatrix, Shape, SparseMatrix};

pub fn gather(inputs: &DenseMatrix, indices: &SparseMatrix, outputs: &mut DenseMatrix) {
    assert!(indices.shape.batch_size().is_none());
    assert_eq!(indices.shape.cols(), 1);
    assert_eq!(indices.shape.rows(), indices.nnz);
    assert_eq!(inputs.shape.cols(), 1);

    let output_shape = Shape::from_raw(indices.shape.rows(), 1, inputs.shape.batch_size());
    outputs.reshape_if_needed(output_shape);
    outputs.set_zero();

    unsafe {
        ops::gather(
            inputs.shape.rows(),
            output_shape.rows(),
            output_shape.batch_size().unwrap_or(1),
            inputs.buf.ptr(),
            indices.buf.ptr(),
            outputs.buf.mut_ptr(),
        );
    }
}

pub fn backprop_gather(
    output_grads: &DenseMatrix,
    indices: &SparseMatrix,
    inputs: &DenseMatrix,
    input_grads: &mut DenseMatrix,
) {
    assert!(indices.shape.batch_size().is_none());
    assert_eq!(indices.shape.cols(), 1);
    assert_eq!(indices.shape.rows(), indices.nnz);

    assert_eq!(inputs.shape.cols(), 1);
    assert_eq!(output_grads.shape.cols(), 1);
    assert_eq!(output_grads.shape.batch_size(), inputs.shape.batch_size());
    assert_eq!(output_grads.shape.rows(), indices.shape.rows());

    input_grads.reshape_if_needed(inputs.shape);

    unsafe {
        ops::gather_backprop(
            inputs.shape.rows(),
            output_grads.shape.rows(),
            output_grads.shape.batch_size().unwrap_or(1),
            output_grads.buf.ptr(),
            indices.buf.ptr(),
            input_grads.buf.mut_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{ExecutionContext, Shape};

    #[test]
    fn test_gather() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(3, 1, 3);

        let mut inputs = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut indices = SparseMatrix::zeroed(device.clone(), Shape::new(1, 1), 1);
        let mut input_grads = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        inputs.load_from_slice(shape1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        unsafe {
            indices.load_from_slice(Shape::new(5, 1), 5, &[-1, 0, 2, 1, 2]);
        }

        gather(&inputs, &indices, &mut output);

        let mut buf = [0.0; 15];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, [0.0, 1.0, 3.0, 2.0, 3.0, 0.0, 4.0, 6.0, 5.0, 6.0, 0.0, 7.0, 9.0, 8.0, 9.0]);

        backprop_gather(&output, &indices, &inputs, &mut input_grads);

        let mut buf = [0.0; 9];
        input_grads.write_to_slice(&mut buf);
        assert_eq!(buf, [1.0, 2.0, 6.0, 4.0, 5.0, 12.0, 7.0, 8.0, 18.0,]);
    }
}
