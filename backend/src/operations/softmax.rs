use std::ops::{Deref, DerefMut};

use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct SoftmaxCrossEntropyLoss;

impl Operation<ExecutionContext> for SoftmaxCrossEntropyLoss {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 && inputs[0] == inputs[1] {
            Ok(Shape::new(1, 1))
        } else {
            Err(format!("Invalid number of inputs in power error! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        super::setup_softmax(output);
        super::setup_ones(output, inputs[0].shape().batch_size().unwrap_or(1));

        let ones = output.internal.get("ones").unwrap().borrow();
        let mut smax = output.internal.get("softmaxed").unwrap().borrow_mut();
        let mut indv = output.internal.get("individual_losses").unwrap().borrow_mut();

        dense::softmax_crossentropy_loss(
            &ones.buf,
            inputs[0].values.dense(),
            inputs[1].values.dense(),
            output.values.dense_mut(),
            smax.deref_mut(),
            indv.deref_mut(),
        );
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        let softmaxed = output.internal.get("softmaxed").unwrap().borrow();

        if let Some(grad) = &mut input1[0].gradients {
            dense::backprop_softmax_crossentropy_loss(
                softmaxed.deref(),
                input2[0].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }

        if let Some(grad) = &mut input2[0].gradients {
            dense::backprop_softmax_crossentropy_loss(
                softmaxed.deref(),
                input1[0].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }
    }
}
