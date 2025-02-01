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
