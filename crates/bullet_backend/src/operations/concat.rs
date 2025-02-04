use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Concat;

impl Operation<ExecutionContext> for Concat {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 && inputs[0].cols() == inputs[1].cols() {
            Ok(Shape::new(inputs[0].rows() + inputs[1].rows(), inputs[0].cols()))
        } else {
            Err(format!("Invalid number of inputs in concat! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::concat(inputs[0].values.dense(), inputs[1].values.dense(), output.values.dense_mut());
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        dense::backprop_concat(
            input1[0].values.dense(),
            input1[0].gradients.as_mut(),
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
    fn concat() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new_batched(3, 1, 3);
        let shape2 = Shape::new_batched(1, 1, 3);

        let mut input1 = Tensor::new(device.clone(), shape1, true);
        let mut input2 = Tensor::new(device.clone(), shape2, true);
        let mut output = Tensor::new(device.clone(), shape2, true);

        device.panic_if_device_error("Failed to initialise matrices!");

        input1.load_dense_from_slice(shape1, &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

        input2.load_dense_from_slice(shape2, &[1.0, 2.0, 3.0]);

        assert_eq!(input1.shape(), shape1);
        assert_eq!(input2.shape(), shape2);

        device.panic_if_device_error("Failed to load data from CPU!");

        Concat.forward(&[&input1, &input2], &mut output);

        device.panic_if_device_error("Failed to concat matrices!");

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap())
        }

        assert_eq!(output.shape(), Shape::new_batched(4, 1, 3));

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[-1.0, 4.0, 2.0, 1.0, -2.0, 0.0, -3.0, 2.0, 1.0, 1.0, 1.0, 3.0]);

        device.panic_if_device_error("Failed to write data to CPU!");

        input1.gradients.as_mut().unwrap().load_from_slice(shape1, &[1.0; 9]);

        Concat.backward(&output, &mut [&mut input1, &mut input2]);

        device.panic_if_device_error("Failed to de-concat!");

        assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape1);
        assert_eq!(input2.gradients.as_ref().unwrap().shape(), shape2);

        let mut grad1 = [0.0; 9];
        input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
        assert_eq!(grad1, [0.0, 5.0, 3.0, -1.0, 1.0, -2.0, 2.0, 2.0, 2.0]);

        let mut grad2 = [0.0; 3];
        input2.gradients.as_ref().unwrap().write_to_slice(&mut grad2);
        assert_eq!(grad2, [1.0, 2.0, 3.0]);

        device.panic_if_device_error("Failed to write data to CPU!");
    }
}
