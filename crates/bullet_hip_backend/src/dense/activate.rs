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
