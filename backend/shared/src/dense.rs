mod activate;
mod adamw;
mod concat;
mod conv;
mod linear_comb;
mod matmul;
mod pairwise;
mod power_error;
mod slice;
mod softmax;
mod submatrix_product;

pub use activate::*;
pub use adamw::*;
pub use concat::*;
pub use conv::*;
pub use linear_comb::*;
pub use matmul::*;
pub use pairwise::*;
pub use power_error::*;
pub use slice::*;
pub use softmax::*;
pub use submatrix_product::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bullet_core::{shape::Shape, tensor::DenseMatrix};

    use crate::ExecutionContext;

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
