use crate::backend::{sparse, ExecutionContext, Matrix, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Mask;

impl Operation<ExecutionContext> for Mask {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 {
            if inputs[0] == inputs[1] {
                Ok(inputs[0])
            } else {
                Err(format!("Input and mask must have the same shape: {} != {}", inputs[0], inputs[1]))
            }
        } else {
            Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        match &inputs[1].values {
            Matrix::Dense(_) => unimplemented!("Masking with dense masks is not supported!"),
            Matrix::Sparse(sparse) => {
                sparse::mask(inputs[0].values.dense(), sparse, output.values.dense_mut());
            }
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        match &input2[0].values {
            Matrix::Dense(_) => unimplemented!("Masking with dense masks is not supported!"),
            Matrix::Sparse(sparse) => {
                assert!(input2[0].gradients.as_ref().is_none());

                if let Some(grad) = input1[0].gradients.as_mut() {
                    sparse::backprop_mask(output.gradients.as_ref().unwrap(), sparse, grad);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::Matrix;
    use bullet_core::device::Device;

    #[test]
    fn test_mask() {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new_batched(3, 1, 4);

        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let mask_vals = [0, 1, 2, -1, 0, 2, -1, -1];

        let masked_vals = [1.0, 2.0, 0.0, 0.0, 0.0, 6.0, 7.0, 0.0, 9.0, 0.0, 0.0, 0.0];

        let mut inputs = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut masks = Tensor::new(device.clone(), Shape::new(1, 1), false);
        let mut outputs = Tensor::new(device.clone(), Shape::new(1, 1), true);

        inputs.load_dense_from_slice(shape, &vals);
        unsafe {
            masks.load_sparse_from_slice(shape, 2, &mask_vals);
        }

        device.panic_if_device_error("Failed to initialise matrices!");

        Mask.forward(&[&inputs, &masks], &mut outputs);

        device.panic_if_device_error("Failed to compute mask!");

        assert_eq!(outputs.shape(), shape);
        let buf = outputs.get_dense_vals().unwrap();
        assert_eq!(&buf, &masked_vals);

        if let Matrix::Dense(vals) = &outputs.values {
            vals.copy_into(outputs.gradients.as_mut().unwrap());
        }

        if let Matrix::Dense(vals) = &inputs.values {
            vals.copy_into(inputs.gradients.as_mut().unwrap());
        }

        Mask.backward(&outputs, &mut [&mut inputs, &mut masks]);

        device.panic_if_device_error("Failed to mask_backprop!");

        assert_eq!(inputs.shape(), shape);
        let mut buf = [0.0; 12];
        inputs.gradients.as_ref().unwrap().write_to_slice(&mut buf);

        let mut bprop = vals;
        for (a, b) in bprop.iter_mut().zip(masked_vals.iter()) {
            *a += *b;
        }
        assert_eq!(buf, bprop);
    }
}
