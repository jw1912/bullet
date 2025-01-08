use crate::{backend::ops, tensor::Shape};

use super::DenseMatrix;

impl DenseMatrix {
    pub fn gaussian_error(input_a: &Self, input_b: &Self, input_c: &Self, output: &mut Self) {
        assert_eq!(input_a.shape, input_b.shape);
        assert_eq!(input_a.shape, input_c.shape);
        output.reshape_if_needed(Shape::new(1, 1));
        output.set_zero();

        unsafe {
            ops::gaussianError(
                input_a.shape.size(),
                input_a.buf.ptr(),
                input_b.buf.ptr(),
                input_c.buf.ptr(),
                output.buf.mut_ptr(),
            );
        }
    }

    pub fn backprop_gaussian_error(
        input_a: &Self,
        input_a_grad: Option<&mut Self>,
        input_b: &Self,
        input_b_grad: Option<&mut Self>,
        input_c: &Self,
        input_c_grad: Option<&mut Self>,
        output_grad: &Self,
    ) {
        // what the hell is this doing?

        if let Some(grd) = input_a_grad {
            backprop_gaussian_error_single(input_a, input_b, input_c, output_grad, grd);
        }

        if let Some(grd) = input_b_grad {
            backprop_gaussian_error_single(input_b, input_a, input_c, output_grad, grd);
        }

        if let Some(grd) = input_c_grad {
            backprop_gaussian_error_single(input_c, input_a, input_b, output_grad, grd);
        }
    }
}

fn backprop_gaussian_error_single(
    input_a: &DenseMatrix,
    input_b: &DenseMatrix,
    input_c: &DenseMatrix,
    output_grad: &DenseMatrix,
    input_a_grad: &mut DenseMatrix,
) {
    assert_eq!(input_a.shape, input_b.shape);
    assert_eq!(input_a.shape, input_c.shape);
    assert_eq!(output_grad.shape, Shape::new(1, 1));
    input_a_grad.reshape_if_needed(input_a.shape);

    unsafe {
        ops::backpropGaussianError(
            input_a.shape.size(),
            input_a.buf.ptr(),
            input_b.buf.ptr(),
            input_c.buf.ptr(),
            output_grad.buf.ptr(),
            input_a_grad.buf.mut_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::util, tensor::Shape};

    #[test]
    fn gaussian_error_simple() {
        let means = [-1.0];
        let variances = [1.0];
        let target = [-1.0];
        let output = 0.9189;
        let mean_grad = [0.0];
        let var_grad = [0.5];
        let target_grad = [1.0];
        gaussian_error_custom(means, variances, target, output, mean_grad, var_grad, target_grad);
    }

    #[test]
    fn gaussian_error() {
        let means = [-1.0, 5.0, 2.0, 1.0, 2.0, 3.0];
        let variances = [1.0, 2.0, 3.0, 1.1, 4.1, 2.1];
        let target = [-1.0, 4.0, 2.5, 3.3, 2.0, 2.9];
        let output = 1.0;
        let mean_grad = [0.0, 0.5, -0.1667, -2.0909, 0.0, 0.0476];
        let var_grad = [0.5, 0.125, 0.1528, 1.7314, -0.1220, -0.2370];
        let target_grad = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        gaussian_error_custom(means, variances, target, output, mean_grad, var_grad, target_grad);
    }

    fn gaussian_error_custom<const N: usize>(
        means: [f32; N],
        vars: [f32; N],
        targets: [f32; N],
        output_val: f32,
        grad_mean: [f32; N],
        grad_var: [f32; N],
        grad_target: [f32; N],
    ) {
        let shape = Shape::new(N, 1);

        let mut means_mat = DenseMatrix::default();
        let mut vars_mat = DenseMatrix::default();
        let mut target_mat = DenseMatrix::default();
        let mut means_grad = DenseMatrix::default();
        let mut vars_grad = DenseMatrix::default();
        let mut target_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrices from CPU
        {
            means_mat.load_from_slice(shape, &means);

            vars_mat.load_from_slice(shape, &vars);

            target_mat.load_from_slice(shape, &targets);

            assert_eq!(means_mat.shape(), shape);
            assert_eq!(vars_mat.shape(), shape);
            assert_eq!(target_mat.shape(), shape);

            util::panic_if_device_error("Failed to load data from CPU!");
        }

        // NLL error
        {
            DenseMatrix::gaussian_error(&means_mat, &vars_mat, &target_mat, &mut output);

            util::panic_if_device_error("Failed to add matrices!");

            assert_eq!(output.shape(), Shape::new(1, 1));

            let mut buf = [0.0];
            output.write_to_slice(&mut buf);
            assert_eq!(buf[0], output_val);

            util::panic_if_device_error("Failed to write data to CPU!");
        }

        // backprop add
        {
            output.load_from_slice(shape, &[1.0; N]);

            util::panic_if_device_error("Failed to load data from CPU!");

            output.load_from_slice(Shape::new(1, 1), &[1.0]);

            DenseMatrix::backprop_gaussian_error(
                &means_mat,
                Some(&mut means_grad),
                &vars_mat,
                Some(&mut vars_grad),
                &target_mat,
                Some(&mut target_grad),
                &output,
            );

            util::panic_if_device_error("Failed to backprop addition!");

            assert_eq!(means_grad.shape(), shape);
            assert_eq!(vars_grad.shape(), shape);
            assert_eq!(target_grad.shape(), shape);

            let mut grad1 = [0.0; N];
            means_grad.write_to_slice(&mut grad1);
            assert_eq!(grad1, grad_mean);

            let mut grad2 = [0.0; N];
            vars_grad.write_to_slice(&mut grad2);
            assert_eq!(grad2, grad_var);

            let mut grad3 = [0.0; N];
            target_grad.write_to_slice(&mut grad3);
            assert_eq!(grad3, grad_target);

            util::panic_if_device_error("Failed to write data to CPU!");
        }
    }
}
