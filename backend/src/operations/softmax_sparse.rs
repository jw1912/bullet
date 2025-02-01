use std::ops::{Deref, DerefMut};

use crate::backend::{sparse, ExecutionContext, Matrix, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct SparseSoftmaxCrossEntropyLoss;

impl Operation<ExecutionContext> for SparseSoftmaxCrossEntropyLoss {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 3 && inputs[0] == inputs[1] && inputs[1].cols() == inputs[2].cols() {
            Ok(Shape::new(1, 1))
        } else {
            Err(format!("Invalid shape in sparse softmax: {inputs:?}"))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        super::setup_softmax(output);

        let mask = match &inputs[0].values {
            Matrix::Sparse(sparse) => sparse,
            Matrix::Dense(_) => panic!("Dense mask not supported!"),
        };

        let mut smax = output.internal.get("softmaxed").unwrap().borrow_mut();
        let mut indv = output.internal.get("individual_losses").unwrap().borrow_mut();

        sparse::softmax_crossentropy_loss_masked(
            mask,
            inputs[1].values.dense(),
            inputs[2].values.dense(),
            output.values.dense_mut(),
            smax.deref_mut(),
            indv.deref_mut(),
        );
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        let softmaxed = output.internal.get("softmaxed").unwrap().borrow();
        assert_eq!(softmaxed.shape, input1[0].shape());

        let mask = match &input1[0].values {
            Matrix::Sparse(sparse) => sparse,
            Matrix::Dense(_) => panic!("Dense mask not supported!"),
        };

        if let Some(grad) = &mut input2[0].gradients {
            sparse::backprop_softmax_crossentropy_loss_masked(
                mask,
                softmaxed.deref(),
                input2[1].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }

        if let Some(grad) = &mut input2[1].gradients {
            sparse::backprop_softmax_crossentropy_loss_masked(
                mask,
                softmaxed.deref(),
                input2[0].values.dense(),
                output.gradients.as_ref().unwrap(),
                grad,
            );
        }
    }
}
