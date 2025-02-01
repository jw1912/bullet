pub mod operations;

pub use bullet_shared_backend as backend;

pub use backend::{dense, sparse, Activation, DenseMatrix, SparseMatrix, Tensor, ExecutionContext, Matrix, ConvolutionDescription};
