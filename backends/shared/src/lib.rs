mod backend;
pub mod dense;
mod operations;
mod sparse;

use backend::util;
pub use dense::Activation;
pub use backend::{ExecutionContext, Buffer};

use bullet_core::{device::{Device, ValidType}, graph::Operation, shape::Shape, tensor};

impl Device for ExecutionContext {
    type Buffer<T: ValidType> = Buffer<T>;
    type IdType = ();
    type ReduceAcrossBatch = ReduceAcrossBatch;
    
    fn new(id: Self::IdType) -> Self {
        Self::default()    
    }

    fn synchronise(&self) {
        util::device_synchronise();
    }
}

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;

#[derive(Debug, Default)]
pub struct ReduceAcrossBatch;

impl Operation<ExecutionContext> for ReduceAcrossBatch {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 1 && inputs[0] == Shape::new(1, 1) {
            Ok(Shape::new(1, 1))
        } else {
            Err("Must be single scalar input!".to_string())
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        let input = inputs[0].values.dense();

        DenseMatrix::reduce_add_cols(ctx, input, output.values.dense_mut());
    }

    fn backward(&self, output_grad: &Tensor, inputs: &mut [&mut Tensor]) {
        if let Some(grad) = &mut inputs[0].gradients {
            grad.reshape_if_needed(inputs[0].values.shape());
            DenseMatrix::add_assign_vector_to_matrix_columns_scaled(
                ctx,
                1.0,
                output_grad.gradients.as_ref().unwrap(),
                grad,
            );
        }
    }
}
