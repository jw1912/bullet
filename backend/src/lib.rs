pub mod operations;

pub use bullet_shared_backend as backend;

pub use backend::{
    dense, sparse, Activation, ConvolutionDescription, DenseMatrix, ExecutionContext, Matrix, SparseMatrix, Tensor,
};
