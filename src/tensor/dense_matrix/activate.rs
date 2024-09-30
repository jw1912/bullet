use crate::backend::ops;

use super::DenseMatrix;

macro_rules! define_activation {
    (
        $fwd:ident,
        $bwd:ident,
        $fwd_kernel:ident,
        $bwd_kernel:ident
    ) => {
        impl DenseMatrix {
            pub fn $fwd(input: &Self, output: &mut Self) {
                output.reshape_if_needed(input.shape);
                unsafe {
                    ops::$fwd_kernel(output.shape.size(), input.buf.ptr(), output.buf.ptr());
                }
            }

            pub fn $bwd(input: &Self, input_grad: &mut Self, output_grad: &Self) {
                assert_eq!(input.shape, output_grad.shape);
                input_grad.reshape_if_needed(input.shape);
                unsafe {
                    ops::$bwd_kernel(input.shape.size(), input.buf.ptr(), output_grad.buf.ptr(), input_grad.buf.ptr());
                }
            }
        }
    };
}

define_activation!(relu, relu_backward, activateReLU, backpropReLU);
define_activation!(crelu, crelu_backward, activateCReLU, backpropCReLU);
define_activation!(screlu, screlu_backward, activateSCReLU, backpropSCReLU);
define_activation!(sqrrelu, sqrrelu_backward, activateSqrReLU, backpropSqrReLU);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    fn activation_test(
        fwd: fn(&DenseMatrix, &mut DenseMatrix),
        bwd: fn(&DenseMatrix, &mut DenseMatrix, &DenseMatrix),
        fwd_expected: [f32; 4],
        bwd_expected: [f32; 4],
    ) {
        let shape = Shape::new(2, 2);
        let mut input = DenseMatrix::default();
        let mut input_grad = DenseMatrix::default();
        let mut output = DenseMatrix::default();

        // load matrix from CPU
        input.load_from_slice(shape, &[-1.0, 0.5, 2.0, -2.0]);
        assert_eq!(input.shape(), shape);

        fwd(&input, &mut output);

        assert_eq!(output.shape, input.shape);

        let mut buf = [0.0; 4];
        output.write_to_slice(&mut buf);
        assert_eq!(buf, fwd_expected);

        bwd(&input, &mut input_grad, &output);

        assert_eq!(input_grad.shape, input.shape);

        input_grad.write_to_slice(&mut buf);
        assert_eq!(buf, bwd_expected);
    }

    #[test]
    fn relu() {
        activation_test(DenseMatrix::relu, DenseMatrix::relu_backward, [0.0, 0.5, 2.0, 0.0], [0.0, 0.5, 2.0, 0.0]);
    }

    #[test]
    fn crelu() {
        activation_test(DenseMatrix::crelu, DenseMatrix::crelu_backward, [0.0, 0.5, 1.0, 0.0], [0.0, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn screlu() {
        activation_test(
            DenseMatrix::screlu,
            DenseMatrix::screlu_backward,
            [0.0, 0.25, 1.0, 0.0],
            [0.0, 0.25, 0.0, 0.0],
        );
    }

    #[test]
    fn sqrrelu() {
        activation_test(
            DenseMatrix::sqrrelu,
            DenseMatrix::sqrrelu_backward,
            [0.0, 0.25, 4.0, 0.0],
            [0.0, 0.25, 16.0, 0.0],
        );
    }
}
