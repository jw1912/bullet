use bullet_core::{graph::Operation, shape::Shape};

use crate::backend::{sparse, Activation, ExecutionContext, Matrix, Tensor};

#[derive(Debug)]
pub struct AffineDualActivate(pub Activation);

impl Operation<ExecutionContext> for AffineDualActivate {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 4 {
            if inputs[0] * inputs[1] == inputs[3] && inputs[1] == inputs[2] {
                Ok(Shape::new(inputs[3].rows() * 2, inputs[3].cols()))
            } else {
                Err(String::from("Incompatible dims in sparse affine dual!"))
            }
        } else {
            Err(format!("Invalid number of inputs in sparse affine dual! Expected 3, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        let weights = inputs[0].values.dense();
        let biases = inputs[3].values.dense();
        let out = output.values.dense_mut();

        if let (Matrix::Sparse(stm), Matrix::Sparse(ntm)) = (&inputs[1].values, &inputs[2].values) {
            sparse::affine_dual(weights, stm, ntm, biases, out, self.0);
        } else {
            panic!("Inputs must be sparse!");
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, inputs2) = inputs.split_at_mut(1);
        let (input2, inputs3) = inputs2.split_at_mut(1);
        let (input3, input4) = inputs3.split_at_mut(1);

        if let (Matrix::Sparse(stm), Matrix::Sparse(ntm)) = (&input2[0].values, &input3[0].values) {
            sparse::backprop_affine_dual(
                input1[0].values.dense(),
                input1[0].gradients.as_mut().unwrap(),
                stm,
                ntm,
                input4[0].values.dense(),
                input4[0].gradients.as_mut().unwrap(),
                output.values.dense(),
                output.gradients.as_ref().unwrap(),
                self.0,
            );
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
    fn test_affine_dual() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new(2, 3);
        let shape2 = Shape::new_batched(3, 1, 3);

        let mut input1 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input2 = Tensor::new(device.clone(), Shape::new(1, 1), false);
        let mut input3 = Tensor::new(device.clone(), Shape::new(1, 1), false);
        let mut input4 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        input1.load_dense_from_slice(
            shape1,
            // [ -1.0,  2.0,  0.0 ]
            // [  4.0, -2.0, -3.0 ]
            &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0],
        );

        unsafe {
            input2.load_sparse_from_slice(shape2, 2, &[0, -1, 1, 2, -1, -1]);

            input3.load_sparse_from_slice(shape2, 2, &[0, -1, 1, 1, -1, -1]);
        }

        input4.load_dense_from_slice(Shape::new(2, 1), &[0.0, 0.0]);

        assert_eq!(input1.shape(), shape1);
        assert_eq!(input2.shape(), shape2);

        device.panic_if_device_error("Failed to load data from CPU!");

        AffineDualActivate(Activation::Identity).forward(&[&input1, &input2, &input3, &input4], &mut output);

        device.panic_if_device_error("Failed to calculate matmul!");

        assert_eq!(output.shape(), Shape::new_batched(4, 1, 3));

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[-1.0, 4.0, -1.0, 4.0, 2.0, -5.0, 4.0, -4.0, 0.0, 0.0, 0.0, 0.0]);

        device.panic_if_device_error("Failed to write data to CPU!");

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        AffineDualActivate(Activation::Identity)
            .backward(&output, &mut [&mut input1, &mut input2, &mut input3, &mut input4]);

        device.panic_if_device_error("Failed to backprop matmul!");

        assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape1);

        let mut grad1 = [0.0; 6];
        input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
        assert_eq!(grad1, [-2.0, 8.0, 10.0, -13.0, 2.0, -5.0]);

        device.panic_if_device_error("Failed to write data to CPU!");
    }
}
