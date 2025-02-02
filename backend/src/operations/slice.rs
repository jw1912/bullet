use crate::backend::{dense, ExecutionContext, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct SliceRows(pub usize, pub usize);

impl Operation<ExecutionContext> for SliceRows {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 1 {
            if self.1 > self.0 {
                if self.1 <= inputs[0].rows() {
                    if inputs[0].cols() == 1 {
                        Ok(Shape::new(self.1 - self.0, 1))
                    } else {
                        unimplemented!("Cannot slice matrices yet!")
                    }
                } else {
                    Err(format!("Invalid slice indices! end = {} > rows = {}", self.1, inputs[0].rows()))
                }
            } else {
                Err(format!("Invalid slice indices! start = {} >= end = {}", self.0, self.1))
            }
        } else {
            Err(format!("Invalid number of inputs in slice! Expected 1, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        dense::slice_vector_batched(inputs[0].values.dense(), self.0, self.1, output.values.dense_mut());
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        dense::backprop_slice_vector_batched(
            inputs[0].values.dense(),
            inputs[0].gradients.as_mut(),
            self.0,
            self.1,
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
    fn slice() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(3, 1, 3);

        let mut input = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        input.load_dense_from_slice(shape, &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0]);

        assert_eq!(input.shape(), shape);

        device.panic_if_device_error("Failed to load data from CPU!");

        SliceRows(0, 2).forward(&[&input], &mut output);

        device.panic_if_device_error("Failed to concat matrices!");

        assert_eq!(output.shape(), Shape::new_batched(2, 1, 3));

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[-1.0, 4.0, -2.0, 0.0, 1.0, 1.0]);

        device.panic_if_device_error("Failed to write data to CPU!");

        input.gradients.as_mut().unwrap().load_from_slice(shape, &[1.0; 9]);

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        SliceRows(0, 2).backward(&output, &mut [&mut input]);

        device.panic_if_device_error("Failed to backprop slice!");

        assert_eq!(input.gradients.as_ref().unwrap().shape(), shape);

        let mut grad = [0.0; 9];
        input.gradients.as_ref().unwrap().write_to_slice(&mut grad);
        assert_eq!(grad, [0.0, 5.0, 1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 1.0]);

        device.panic_if_device_error("Failed to write data to CPU!");
    }
}
