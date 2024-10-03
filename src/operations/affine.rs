use crate::{
    backend::ExecutionContext,
    tensor::{DenseMatrix, Matrix, Shape, SparseMatrix},
    Tensor,
};

pub fn output_tensor(inputs: &[Shape]) -> Result<Shape, String> {
    if inputs.len() == 3 && inputs[0] * inputs[1] == inputs[2] {
        Ok(inputs[2])
    } else {
        Err(String::from("Invalid number of inputs!"))
    }
}

pub fn forward(ctx: &mut ExecutionContext, inputs: &[&Tensor], output: &mut Tensor) {
    let weights = inputs[0].values.dense();
    let biases = inputs[2].values.dense();
    let out = output.values.dense_mut();

    match &inputs[1].values {
        Matrix::Sparse(sparse) => {
            SparseMatrix::affine(weights, sparse, Some(biases), out)
        }
        Matrix::Dense(dense) => {
            DenseMatrix::matmul(ctx, weights, false, dense, false, out);
            DenseMatrix::add_assign_vector_to_matrix_columns(ctx, biases, out);
        }
    }
}

pub fn backprop(ctx: &mut ExecutionContext, output: &Tensor, inputs: &mut [&mut Tensor]) {
    let (input1, inputs2) = inputs.split_at_mut(1);
    let (input2, input3) = inputs2.split_at_mut(1);
    let out = output.gradients.as_ref().unwrap();

    match &input2[0].values {
        Matrix::Sparse(sparse) => {
            SparseMatrix::backprop_affine(
                input1[0].values.dense(),
                input1[0].gradients.as_mut().unwrap(),
                sparse,
                input3[0].gradients.as_mut(),
                out,
            );
        }
        Matrix::Dense(_) => {
            super::linear::backprop(ctx, output, &mut [&mut input1[0], &mut input2[0]]);
            if let Some(grad) = &mut input3[0].gradients {
                DenseMatrix::backprop_add_single(ctx, input3[0].values.dense(), grad, out);
            }
        }
    }
}
