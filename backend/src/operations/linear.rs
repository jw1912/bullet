use crate::backend::{dense, sparse, ExecutionContext, Matrix, Tensor};
use bullet_core::{graph::Operation, shape::Shape};

#[derive(Debug)]
pub struct Linear;

impl Operation<ExecutionContext> for Linear {
    fn output_tensor(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.len() == 2 {
            Ok(inputs[0] * inputs[1])
        } else {
            Err(format!("Invalid number of inputs in linear! Expected 2, got {}", inputs.len()))
        }
    }

    fn forward(&self, inputs: &[&Tensor], output: &mut Tensor) {
        match &inputs[1].values {
            Matrix::Dense(dense) => {
                dense::matmul(inputs[0].values.dense(), false, dense, false, output.values.dense_mut());
            }
            Matrix::Sparse(sparse) => {
                sparse::linear(inputs[0].values.dense(), sparse, output.values.dense_mut());
            }
        }
    }

    fn backward(&self, output: &Tensor, inputs: &mut [&mut Tensor]) {
        let (input1, input2) = inputs.split_at_mut(1);

        match &input2[0].values {
            Matrix::Dense(dense) => {
                dense::backprop_matmul(
                    input1[0].values.dense(),
                    input1[0].gradients.as_mut(),
                    false,
                    dense,
                    input2[0].gradients.as_mut(),
                    false,
                    output.gradients.as_ref().unwrap(),
                );
            }
            Matrix::Sparse(sparse) => {
                assert!(input2[0].gradients.as_ref().is_none());

                if let Some(grad) = input1[0].gradients.as_mut() {
                    sparse::backprop_linear(
                        input1[0].values.dense(),
                        grad,
                        sparse,
                        output.values.dense(),
                        output.gradients.as_ref().unwrap(),
                    );
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
    fn single_matmul() {
        test_matmul_with_shape(Shape::new(3, 2));
    }

    #[test]
    fn batched_matmul() {
        test_matmul_with_shape(Shape::new_batched(3, 1, 2));
    }

    fn test_matmul_with_shape(shape2: Shape) {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new(2, 3);

        let mut input1 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input2 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        device.panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            input1.load_dense_from_slice(
                shape1,
                // [ -1.0,  2.0,  0.0 ]
                // [  4.0, -2.0, -3.0 ]
                &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0],
            );

            input2.load_dense_from_slice(
                shape2,
                // [ 1.0 ]
                // [ 2.0 ]
                // [ 3.0 ]
                &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            );

            assert_eq!(input1.shape(), shape1);
            assert_eq!(input2.shape(), shape2);

            device.panic_if_device_error("Failed to load data from CPU!");
        }

        // normal matmul
        {
            Linear.forward(&[&input1, &input2], &mut output);

            device.panic_if_device_error("Failed to calculate matmul!");

            assert_eq!(output.shape(), Shape::from_raw(2, shape2.cols(), shape2.batch_size()));

            let buf = output.get_dense_vals().unwrap();
            assert_eq!(&buf, &[3.0, -9.0, 3.0, -9.0]);

            device.panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop normal matmul
        {
            if let Matrix::Dense(vals) = &output.values {
                vals.copy_into(output.gradients.as_mut().unwrap());
            }

            Linear.backward(&output, &mut [&mut input1, &mut input2]);

            device.panic_if_device_error("Failed to backprop matmul!");

            assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape1);
            assert_eq!(input2.gradients.as_ref().unwrap().shape(), shape2);

            let mut grad1 = [0.0; 6];
            input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
            assert_eq!(grad1, [6.0, -18.0, 12.0, -36.0, 18.0, -54.0]);

            let mut grad2 = [0.0; 6];
            input2.gradients.as_ref().unwrap().write_to_slice(&mut grad2);
            assert_eq!(grad2, [-39.0, 24.0, 27.0, -39.0, 24.0, 27.0]);

            device.panic_if_device_error("Failed to write data to CPU!");
        }

        if shape2.batch_size().is_none() {
            // transposed matmul
            {
                dense::matmul(input2.values.dense(), true, input1.values.dense(), true, output.values.dense_mut());

                device.panic_if_device_error("Failed to calculate transposed matmul!");

                assert_eq!(output.shape(), Shape::new(2, 2));

                let buf = output.get_dense_vals().unwrap();
                assert_eq!(&buf, &[3.0, 3.0, -9.0, -9.0]);

                device.panic_if_device_error("Failed to write data to CPU!");
            }

            // backprop transposed matmul
            {
                input1.gradients.as_mut().unwrap().set_zero();
                input2.gradients.as_mut().unwrap().set_zero();

                dense::backprop_matmul(
                    input2.values.dense(),
                    input2.gradients.as_mut(),
                    true,
                    input1.values.dense(),
                    input1.gradients.as_mut(),
                    true,
                    output.values.dense(),
                );

                device.panic_if_device_error("Failed to backprop transposed matmul!");

                assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape1);
                assert_eq!(input2.gradients.as_ref().unwrap().shape(), shape2);

                let mut grad1 = [0.0; 6];
                input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
                assert_eq!(grad1, [6.0, -18.0, 12.0, -36.0, 18.0, -54.0]);

                let mut grad2 = [0.0; 6];
                input2.gradients.as_ref().unwrap().write_to_slice(&mut grad2);
                assert_eq!(grad2, [-39.0, 24.0, 27.0, -39.0, 24.0, 27.0]);

                device.panic_if_device_error("Failed to write data to CPU!");
            }
        }
    }

    #[test]
    fn sparse_matmul() {
        let device = Arc::new(ExecutionContext::default());

        let shape1 = Shape::new(2, 3);
        let shape2 = Shape::new_batched(3, 1, 3);

        let mut input1 = Tensor::new(device.clone(), Shape::new(1, 1), true);
        let mut input2 = Tensor::new(device.clone(), Shape::new(1, 1), false);
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
        }

        assert_eq!(input1.shape(), shape1);
        assert_eq!(input2.shape(), shape2);

        device.panic_if_device_error("Failed to load data from CPU!");

        Linear.forward(&[&input1, &input2], &mut output);

        device.panic_if_device_error("Failed to calculate matmul!");

        assert_eq!(output.shape(), Shape::new_batched(2, 1, 3));

        let buf = output.get_dense_vals().unwrap();
        assert_eq!(&buf, &[-1.0, 4.0, 2.0, -5.0, 0.0, 0.0]);

        device.panic_if_device_error("Failed to write data to CPU!");

        if let Matrix::Dense(vals) = &output.values {
            vals.copy_into(output.gradients.as_mut().unwrap());
        }

        Linear.backward(&output, &mut [&mut input1, &mut input2]);

        device.panic_if_device_error("Failed to backprop matmul!");

        assert_eq!(input1.gradients.as_ref().unwrap().shape(), shape1);

        let mut grad1 = [0.0; 6];
        input1.gradients.as_ref().unwrap().write_to_slice(&mut grad1);
        assert_eq!(grad1, [-1.0, 4.0, 2.0, -5.0, 2.0, -5.0]);

        device.panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn aligned_matches_unaligned() {
        let device = Arc::new(ExecutionContext::default());

        let input_shape = Shape::new_batched(4, 1, 3);
        let shape1 = Shape::new(256, 4);
        let mut inputs1 = Tensor::new(device.clone(), Shape::new(1, 1), false);
        let mut weights1 = Tensor::new(device.clone(), Shape::new(1, 1), true);

        println!("wha");

        unsafe {
            inputs1.load_sparse_from_slice(input_shape, 2, &[0, -1, 1, 2, -1, -1]);
        }
        weights1.load_dense_from_slice(shape1, &vec![1.0; 256 * 4]);

        let shape2 = Shape::new(255, 4);
        let mut inputs2 = Tensor::new(device.clone(), Shape::new(1, 1), false);
        let mut weights2 = Tensor::new(device.clone(), Shape::new(1, 1), true);

        println!("yeah");

        unsafe {
            inputs2.load_sparse_from_slice(input_shape, 2, &[0, -1, 1, 2, -1, -1]);
        }
        weights2.load_dense_from_slice(shape2, &[1.0; 255 * 4]);

        let mut output = Tensor::new(device.clone(), Shape::new(1, 1), true);

        println!("loaded");

        device.panic_if_device_error("Failed to initialise matrices!");

        let mut buf2 = vec![0.0; 255 * 3];
        Linear.forward(&[&weights2, &inputs2], &mut output);
        output.values.dense().write_to_slice(&mut buf2);

        let mut buf = vec![0.0; 256 * 3];
        Linear.forward(&[&weights1, &inputs1], &mut output);
        output.values.dense().write_to_slice(&mut buf);

        for i in 0..3 {
            for j in 0..255 {
                assert_eq!(buf[256 * i + j], buf2[255 * i + j]);
            }
        }
    }
}
