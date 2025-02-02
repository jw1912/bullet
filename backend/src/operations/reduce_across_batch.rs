use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

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
        super::setup_ones(output, inputs[0].shape().batch_size().unwrap_or(1));
        let ones = output.internal.get("ones").unwrap().borrow();

        dense::reduce_add_batch(&ones.buf, input, output.values.dense_mut());
    }

    fn backward(&self, output_grad: &Tensor, inputs: &mut [&mut Tensor]) {
        let ones = output_grad.internal.get("ones").unwrap().borrow();
        assert!(ones.shape.size() >= inputs[0].shape().cols());

        if let Some(grad) = &mut inputs[0].gradients {
            grad.reshape_if_needed(inputs[0].values.shape());
            dense::add_assign_single_to_batched_scaled(&ones.buf, 1.0, output_grad.gradients.as_ref().unwrap(), grad);
        }
    }
}
