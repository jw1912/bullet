use std::num::NonZeroUsize;

use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct LinearCombination(pub f32, pub f32);

impl Operation<ExecutionContext> for LinearCombination {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 && inputs[0] == inputs[1] {
            Ok(inputs[0])
        } else {
            Err(format!("Invalid number of inputs in add! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        let batch_size = Shape::get_batch_size(&inputs[0].shape(), &inputs[1].shape());
        let batch_size = batch_size.map(NonZeroUsize::get).unwrap_or(1);
        super::setup_ones(output, batch_size);
        let ones = output.internal.get("ones").unwrap().borrow();

        dense::linear_comb(
            &ones.buf,
            self.0,
            inputs[0].values.dense(),
            self.1,
            inputs[1].values.dense(),
            output.values.dense_mut(),
        );
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let batch_size = inputs[0].shape().cols().max(inputs[1].shape().cols());
        let (input1, input2) = inputs.split_at_mut(1);
        let ones = output.internal.get("ones").unwrap().borrow();
        assert!(ones.shape.size() >= batch_size);

        dense::linear_comb_backward(
            &ones.buf,
            self.0,
            input1[0].values.dense(),
            input1[0].gradients.as_mut(),
            self.1,
            input2[0].values.dense(),
            input2[0].gradients.as_mut(),
            output.gradients.as_ref().unwrap(),
        );
    }
}
