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
        super::setup_ones(output, inputs[0].shape().size());

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use bullet_core::device::Device;

    #[test]
    fn softmax_crossentropy() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(2, 2, 3);

        let mut pred = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut target = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        pred.load_dense_from_slice(shape, &[1.0, 2.0, 1.0, 2.0, -4.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(pred.shape(), shape);

        target.load_dense_from_slice(shape, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(target.shape(), shape);

        device.panic_if_device_error("Failed to load data from CPU!");

        SoftmaxCrossEntropyLoss.forward(&[&pred, &target], &mut output);

        device.panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape(), Shape::new(1, 1));

        let buf = output.get_dense_vals().unwrap();
        assert!((buf[0] - 3.865).abs() < 0.001);

        device.panic_if_device_error("Failed to load data from CPU!");

        output.gradients.as_mut().unwrap().load_from_slice(Shape::new(1, 1), &[1.0]);

        SoftmaxCrossEntropyLoss.backward(&output, &mut [&mut pred, &mut target]);

        device.panic_if_device_error("Failed to calculate activation!");

        let mut buf = [0.0; 12];
        pred.gradients.as_ref().unwrap().write_to_slice(&mut buf);
        let expected =
            [-0.8655, 0.3655, 0.1345, 0.3655, 0.0163, -0.6721, 0.3279, 0.3279, 0.1749, 0.1749, -0.5246, 0.1749];

        let mut total = 0.0;
        for (p, e) in buf.iter().zip(expected.iter()) {
            total += (p - e).abs();
        }

        assert!(total < 0.01);

        device.panic_if_device_error("Failed to write data to CPU!");
    }
}
