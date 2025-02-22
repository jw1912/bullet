use bullet_core::device::{DeviceBuffer, OperationError};

use crate::{backend::ops, Buffer, OperationResult};

macro_rules! define_activation {
    (
        $fwd:ident,
        $bwd:ident,
        $fwd_kernel:ident,
        $bwd_kernel:ident
    ) => {
        pub fn $fwd(size: usize, input: &Buffer<f32>, output: &mut Buffer<f32>) -> OperationResult {
            if size > input.size() || size > output.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            unsafe {
                ops::$fwd_kernel(size, input.ptr(), output.mut_ptr());
            }

            Ok(())
        }

        pub fn $bwd(
            size: usize,
            input: &Buffer<f32>,
            input_grad: &mut Buffer<f32>,
            output_grad: &Buffer<f32>,
        ) -> OperationResult {
            if size > input.size() || size > input_grad.size() || size > output_grad.size() {
                return Err(OperationError::IndexOutOfBounds);
            }

            unsafe {
                ops::$bwd_kernel(size, input.ptr(), output_grad.ptr(), input_grad.mut_ptr());
            }

            Ok(())
        }
    };
}

define_activation!(relu, relu_backward, activateReLU, backpropReLU);
define_activation!(crelu, crelu_backward, activateCReLU, backpropCReLU);
define_activation!(screlu, screlu_backward, activateSCReLU, backpropSCReLU);
define_activation!(sqrrelu, sqrrelu_backward, activateSqrReLU, backpropSqrReLU);
define_activation!(sigmoid, sigmoid_backward, activateSigmoid, backpropSigmoid);
define_activation!(square, square_backward, activateSquare, backpropSquare);
