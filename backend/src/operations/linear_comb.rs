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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::Matrix;
    use bullet_core::device::Device;

    #[test]
    fn linear_comb() {
        let device = Arc::new(ExecutionContext::default());

        let comb = LinearCombination(2.0, -1.0);

        let shape1 = Shape::new_batched(3, 1, 3);
        let shape2 = Shape::new(3, 1);

        let mut input1 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input2 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_dense_from_slice(shape1, &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

            input2.load_dense_from_slice(shape2, &[1.0, 2.0, 3.0]);

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            device.panic_if_device_error("Failed to load data from CPU!");
        }

        let expected_fwd =
            [[-3.0, 6.0, 1.0, -5.0, -2.0, -9.0, 1.0, 0.0, -1.0], [3.0, 0.0, 4.0, 4.0, 4.0, 9.0, 1.0, 3.0, 5.0]];

        let expected_bwd1 = [
            [-6.0, 12.0, 2.0, -10.0, -4.0, -18.0, 2.0, 0.0, -2.0],
            [-3.0, 0.0, -4.0, -4.0, -4.0, -9.0, -1.0, -3.0, -5.0],
        ];

        let expected_bwd2 = [[7.0, -4.0, 9.0], [16.0, 14.0, 36.0]];

        let mut test_linear_comb = |i: &mut Tensor, j: &mut Tensor, num: usize| {
            comb.forward(&[i, j], &mut output);

            device.panic_if_device_error("Failed to add matrices!");

            assert_eq!(output.shape(), Shape::new_batched(3, 1, 3));

            let buf = output.get_dense_vals().unwrap();
            assert_eq!(buf, expected_fwd[num], "{num}");

            device.panic_if_device_error("Failed to write data to CPU!");

            i.gradients.as_mut().unwrap().set_zero();
            j.gradients.as_mut().unwrap().set_zero();

            if let Matrix::Dense(vals) = &output.values {
                vals.copy_into(output.gradients.as_mut().unwrap());
            }

            comb.backward(&output, &mut [i, j]);

            device.panic_if_device_error("Failed to backprop addition!");

            let mut grads = [vec![0.0; 9], vec![0.0; 3]];
            i.gradients.as_ref().unwrap().write_to_slice(&mut grads[num]);
            j.gradients.as_ref().unwrap().write_to_slice(&mut grads[1 - num]);

            assert_eq!(&grads[0], &expected_bwd1[num], "{num}");
            assert_eq!(&grads[1], &expected_bwd2[num], "{num}");

            device.panic_if_device_error("Failed to write data to CPU!");
        };

        test_linear_comb(&mut input1, &mut input2, 0);
        test_linear_comb(&mut input2, &mut input1, 1);
    }
}
