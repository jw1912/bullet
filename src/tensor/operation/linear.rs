use crate::{
    backend::ExecutionContext,
    tensor::{DenseMatrix, Matrix, Shape, SparseMatrix, Tensor},
};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 2 {
        Ok(inputs[0] * inputs[1])
    } else {
        Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
    }
}

pub fn forward(ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
    match &inputs[1].values {
        Matrix::Dense(dense) => {
            DenseMatrix::matmul(ctx, inputs[0].values.dense(), false, dense, false, output.values.dense_mut());
        }
        Matrix::Sparse(sparse) => {
            SparseMatrix::linear(inputs[0].values.dense(), sparse, output.values.dense_mut());
        }
    }
}

pub fn backprop(ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, input2) = inputs.split_at_mut(1);

    match &input2[0].values {
        Matrix::Dense(dense) => {
            DenseMatrix::backprop_matmul(
                ctx,
                input1[0].values.dense(),
                input1[0].gradients.as_mut(),
                false,
                dense,
                input2[0].gradients.as_mut(),
                false,
                output.gradients.as_ref().unwrap(),
            );
        }
        Matrix::Sparse(sparse) => {
            assert!(input2[0].gradients.as_ref().is_none());

            if let Some(grad) = input1[0].gradients.as_mut() {
                SparseMatrix::backprop_linear(
                    input1[0].values.dense(),
                    grad,
                    sparse,
                    output.values.dense(),
                    output.gradients.as_ref().unwrap(),
                );
            }
        }
    }
}
