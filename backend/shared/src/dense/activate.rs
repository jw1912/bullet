use crate::{backend::ops, DenseMatrix};

/// List of supported activation functions.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity = 0,
    ReLU = 1,
    CReLU = 2,
    SCReLU = 3,
    SqrReLU = 4,
    Sigmoid = 5,
}

macro_rules! define_activation {
    (
        $fwd:ident,
        $bwd:ident,
        $fwd_kernel:ident,
        $bwd_kernel:ident
    ) => {
        pub fn $fwd(input: &DenseMatrix, output: &mut DenseMatrix) {
            output.reshape_if_needed(input.shape);
            unsafe {
                ops::$fwd_kernel(output.shape.size(), input.buf.ptr(), output.buf.mut_ptr());
            }
        }

        pub fn $bwd(input: &DenseMatrix, input_grad: &mut DenseMatrix, output_grad: &DenseMatrix) {
            assert_eq!(input.shape, output_grad.shape);
            input_grad.reshape_if_needed(input.shape);
            unsafe {
                ops::$bwd_kernel(input.shape.size(), input.buf.ptr(), output_grad.buf.ptr(), input_grad.buf.mut_ptr());
            }
        }
    };
}

define_activation!(relu, relu_backward, activateReLU, backpropReLU);
define_activation!(crelu, crelu_backward, activateCReLU, backpropCReLU);
define_activation!(screlu, screlu_backward, activateSCReLU, backpropSCReLU);
define_activation!(sqrrelu, sqrrelu_backward, activateSqrReLU, backpropSqrReLU);
define_activation!(sigmoid, sigmoid_backward, activateSigmoid, backpropSigmoid);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{backend::util, ExecutionContext, Shape};

    fn activation_test(
        fwd: fn(&DenseMatrix, &mut DenseMatrix),
        bwd: fn(&DenseMatrix, &mut DenseMatrix, &DenseMatrix),
        fwd_expected: [f32; 4],
        bwd_expected: [f32; 4],
    ) {
        let device = Arc::new(ExecutionContext::default());

        let shape = Shape::new(2, 2);
        let mut input = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut input_grad = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));
        let mut output = DenseMatrix::zeroed(device.clone(), Shape::new(1, 1));

        util::panic_if_device_error("Failed to initialise matrices!");

        // load matrix from CPU
        input.load_from_slice(shape, &[-1.0, 0.5, 2.0, -2.0]);
        assert_eq!(input.shape(), shape);

        util::panic_if_device_error("Failed to load data from CPU!");

        fwd(&input, &mut output);

        util::panic_if_device_error("Failed to calculate activation!");

        assert_eq!(output.shape, input.shape);

        let mut buf = [0.0; 4];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, fwd_expected);

        util::panic_if_device_error("Failed to write data to CPU!");

        bwd(&input, &mut input_grad, &output);

        util::panic_if_device_error("Failed to backprop activation!");

        assert_eq!(input_grad.shape, input.shape);

        input_grad.write_to_slice(&mut buf);
        assert_eq!(buf, bwd_expected);

        util::panic_if_device_error("Failed to write data to CPU!");
    }

    #[test]
    fn test_relu() {
        activation_test(relu, relu_backward, [0.0, 0.5, 2.0, 0.0], [0.0, 0.5, 2.0, 0.0]);
    }

    #[test]
    fn test_crelu() {
        activation_test(crelu, crelu_backward, [0.0, 0.5, 1.0, 0.0], [0.0, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn test_screlu() {
        activation_test(screlu, screlu_backward, [0.0, 0.25, 1.0, 0.0], [0.0, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn test_sqrrelu() {
        activation_test(sqrrelu, sqrrelu_backward, [0.0, 0.25, 4.0, 0.0], [0.0, 0.25, 16.0, 0.0]);
    }
}
