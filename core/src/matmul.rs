use crate::{device::Device, shape::Shape, tensor::DenseMatrix};

pub trait Matmul: Device {
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        input_a: &DenseMatrix<Self>,
        shape_a: Shape,
        trans_a: bool,
        input_b: &DenseMatrix<Self>,
        shape_b: Shape,
        trans_b: bool,
        output: &mut DenseMatrix<Self>,
        output_shape: Shape,
        increment: bool,
    );

    fn sgemm_batched(
        input_a: &DenseMatrix<Self>,
        trans_a: bool,
        input_b: &DenseMatrix<Self>,
        trans_b: bool,
        output: &mut DenseMatrix<Self>,
        increment: bool,
    );

    fn matmul(input_a: &DenseMatrix<Self>, trans_a: bool, input_b: &DenseMatrix<Self>, trans_b: bool, output: &mut DenseMatrix<Self>) {
        let output_shape = input_a.shape.maybe_transpose(trans_a) * input_b.shape.maybe_transpose(trans_b);

        match (input_a.shape.batch_size(), input_b.shape.batch_size()) {
            (Some(_), Some(_)) => Self::sgemm_batched(input_a, trans_a, input_b, trans_b, output, false),
            (None, None) => {
                Self::sgemm(input_a, input_a.shape, trans_a, input_b, input_b.shape, trans_b, output, output_shape, false);
            }
            (None, Some(x)) => {
                let shape_b = Shape::new(input_b.shape.rows(), x);
                if trans_b || input_b.shape.cols() > 1 {
                    unimplemented!()
                }

                Self::sgemm(input_a, input_a.shape, trans_a, input_b, shape_b, trans_b, output, output_shape, false);
            }
            (Some(_), None) => unimplemented!(),
        }
    }

    fn backprop_matmul(
        input_a: &DenseMatrix<Self>,
        input_a_grad: Option<&mut DenseMatrix<Self>>,
        trans_a: bool,
        input_b: &DenseMatrix<Self>,
        input_b_grad: Option<&mut DenseMatrix<Self>>,
        trans_b: bool,
        output_grad: &DenseMatrix<Self>,
    ) {
        match (input_a.shape.batch_size(), input_b.shape.batch_size()) {
            (Some(_), Some(_)) => {
                backprop_batched_matmul(input_a, input_a_grad, trans_a, input_b, input_b_grad, trans_b, output_grad);
            }
            (None, None) => {
                backprop_single_matmul(
                    input_a,
                    input_a.shape,
                    input_a_grad,
                    trans_a,
                    input_b,
                    input_b.shape,
                    input_b_grad,
                    trans_b,
                    output_grad,
                );
            }
            (None, Some(x)) => {
                let shape_b = Shape::new(input_b.shape.rows(), x);
                if trans_b || input_b.shape.cols() > 1 {
                    unimplemented!()
                }

                backprop_single_matmul(
                    input_a,
                    input_a.shape,
                    input_a_grad,
                    trans_a,
                    input_b,
                    shape_b,
                    input_b_grad,
                    trans_b,
                    output_grad,
                );
            }
            (Some(_), None) => unimplemented!(),
        }
    }
}


#[allow(clippy::too_many_arguments)]
fn backprop_single_matmul<D: Matmul>(
    input_a: &DenseMatrix<D>,
    shape_a: Shape,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    shape_b: Shape,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    trans_b: bool,
    output_grad: &DenseMatrix<D>,
) {
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    if let Some(grad_a) = input_a_grad {
        if trans_a {
            D::sgemm(input_b, shape_b, trans_b, output_grad, shape_o, true, grad_a, input_a.shape, true);
        } else {
            D::sgemm(output_grad, shape_o, false, input_b, shape_b, !trans_b, grad_a, input_a.shape, true);
        }
    }

    if let Some(grad_b) = input_b_grad {
        if trans_b {
            D::sgemm(output_grad, shape_o, true, input_a, shape_a, trans_a, grad_b, input_b.shape, true);
        } else {
            D::sgemm(input_a, shape_a, !trans_a, output_grad, shape_o, false, grad_b, input_b.shape, true);
        }
    }
}

fn backprop_batched_matmul<D: Matmul>(
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    trans_b: bool,
    output_grad: &DenseMatrix<D>,
) {
    if let Some(grad_a) = input_a_grad {
        if trans_a {
            D::sgemm_batched(input_b, trans_b, output_grad, true, grad_a, true);
        } else {
            D::sgemm_batched(output_grad, false, input_b, !trans_b, grad_a, true);
        }
    }

    if let Some(grad_b) = input_b_grad {
        if trans_b {
            D::sgemm_batched(output_grad, true, input_a, trans_a, grad_b, true);
        } else {
            D::sgemm_batched(input_a, !trans_a, output_grad, false, grad_b, true);
        }
    }
}
