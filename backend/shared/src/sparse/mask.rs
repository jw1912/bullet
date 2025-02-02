use crate::{backend::ops, DenseMatrix, SparseMatrix};

pub fn mask(inputs: &DenseMatrix, masks: &SparseMatrix, outputs: &mut DenseMatrix) {
    let shape = inputs.shape;
    assert_eq!(shape, masks.shape);
    assert_eq!(shape.cols(), 1);
    assert!(masks.nnz <= shape.rows());

    outputs.reshape_if_needed(shape);
    outputs.set_zero();

    unsafe {
        ops::sparse_mask(
            shape.rows(),
            shape.batch_size().unwrap_or(1),
            masks.nnz,
            inputs.buf.ptr(),
            masks.buf.ptr(),
            outputs.buf.mut_ptr(),
        );
    }
}

pub fn backprop_mask(output_grads: &DenseMatrix, masks: &SparseMatrix, input_grads: &mut DenseMatrix) {
    let shape = output_grads.shape;
    assert_eq!(shape, masks.shape);
    assert_eq!(shape.cols(), 1);
    assert!(masks.nnz <= shape.rows());

    input_grads.reshape_if_needed(shape);

    unsafe {
        ops::sparse_mask_backprop(
            shape.rows(),
            shape.batch_size().unwrap_or(1),
            masks.nnz,
            output_grads.buf.ptr(),
            masks.buf.ptr(),
            input_grads.buf.mut_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    #[test]
    fn test_mask() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(3, 1, 4);

        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let mask_vals = [0, 1, 2, -1, 0, 2, -1, -1];

        let masked_vals = [1.0, 2.0, 0.0, 0.0, 0.0, 6.0, 7.0, 0.0, 9.0, 0.0, 0.0, 0.0];

        let mut inputs = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut masks = SparseMatrix::zeroed(device.clone(), Shape::new(1, 1), 1);
        let mut outputs = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        inputs.load_from_slice(shape, &vals);
        unsafe {
            masks.load_from_slice(shape, 2, &mask_vals);
        }

        util::panic_if_device_error("Failed to initialise matrices!");

        mask(&inputs, &masks, &mut outputs);

        util::panic_if_device_error("Failed to compute mask!");

        assert_eq!(outputs.shape, shape);
        let mut buf = [0.0; 12];
        outputs.write_to_slice(&mut buf);
        assert_eq!(buf, masked_vals);

        backprop_mask(&outputs, &masks, &mut inputs);

        util::panic_if_device_error("Failed to mask_backprop!");

        assert_eq!(inputs.shape, shape);
        inputs.write_to_slice(&mut buf);

        let mut bprop = vals;
        for (a, b) in bprop.iter_mut().zip(masked_vals.iter()) {
            *a += *b;
        }
        assert_eq!(buf, bprop);
    }
}
