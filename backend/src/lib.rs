pub mod operations;

pub use bullet_shared_backend as backend;

pub use backend::{
    dense, sparse, Activation, ConvolutionDescription, DenseMatrix, ExecutionContext, Matrix, SparseMatrix, Tensor,
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bullet_core::{device::Device, shape::Shape};

    use super::*;

    #[test]
    fn sparse_to_dense() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(3, 1, 3);

        let mut input = SparseMatrix::zeroed(device.clone(), Shape::new(1, 1), 1);
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        device.panic_if_device_error("Failed to initialise matrices!");

        unsafe {
            input.load_from_slice(shape, 2, &[0, -1, 1, 2, 1, -1]);
        }

        sparse::copy_into_dense(&input, &mut output);

        let mut buf = [0.0; 9];
        output.write_to_slice(&mut buf);

        assert_eq!(buf, [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn read_write_dense_matrix() {
        let device = Arc::new(ExecutionContext::default());
        let mut matrix = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        matrix.load_from_slice(Shape::new(3, 3), &values);

        let bytes = matrix.write_to_byte_buffer("matrix").unwrap();

        println!("{bytes:?}");

        matrix.set_zero();

        let (id, bytes_read) = matrix.read_from_byte_buffer(&bytes);

        assert_eq!(id.as_str(), "matrix");
        assert_eq!(bytes_read, 59);

        let mut buf = [0.0; 9];

        matrix.write_to_slice(&mut buf);

        assert_eq!(buf, values);
    }

    #[test]
    fn attempt_invalid_writes() {
        let device = Arc::new(ExecutionContext::default());
        let matrix = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        assert!(matrix.write_to_byte_buffer("matrix\n").is_err());
        assert!(matrix.write_to_byte_buffer("màtrix\n").is_err());
    }
}
