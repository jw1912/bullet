use crate::{
    device::Device,
    shape::Shape,
    tensor::{DenseMatrix, Matrix, Tensor},
};

pub fn affine<D: Device>(
    a: &DenseMatrix<D>,
    b: &Tensor<D>,
    c: Option<(&DenseMatrix<D>, &D::BufferF32)>,
    out: &mut DenseMatrix<D>,
) {
    match &b.values {
        Matrix::Dense(dense) => {
            matmul(a, false, dense, false, out);
            if let Some((c, ones)) = c {
                D::add_assign_single_to_batched_scaled(ones, 1.0, c, out);
            }
        }
        Matrix::Sparse(sparse) => D::sparse_affine(a, sparse, c.map(|x| x.0), out),
    }
}

pub fn backprop_affine<D: Device>(
    a: &mut Tensor<D>,
    b: &mut Tensor<D>,
    c: Option<(&mut Tensor<D>, &D::BufferF32)>,
    output: &DenseMatrix<D>,
    output_grad: &DenseMatrix<D>,
) {
    match &b.values {
        Matrix::Dense(dense) => {
            backprop_matmul(
                a.values.dense(),
                a.gradients.as_mut(),
                false,
                dense,
                b.gradients.as_mut(),
                false,
                output_grad,
            );

            if let Some((c, ones)) = c {
                if let Some(grad) = c.gradients.as_mut() {
                    D::backprop_add_single_scaled(ones, 1.0, c.values.dense(), grad, output_grad);
                }
            }
        }
        Matrix::Sparse(sparse) => {
            assert!(b.gradients.is_none());

            if let Some(agrd) = a.gradients.as_mut() {
                let (c, cgrd) = c.map(|x| (Some(x.0.values.dense()), x.0.gradients.as_mut())).unwrap_or((None, None));
                D::backprop_sparse_affine(a.values.dense(), agrd, sparse, c, cgrd, output, output_grad)
            } else if let Some((c, ones)) = c {
                if let Some(grad) = c.gradients.as_mut() {
                    D::backprop_add_single_scaled(ones, 1.0, c.values.dense(), grad, output_grad);
                }
            }
        }
    }
}

pub fn matmul<D: Device>(
    input_a: &DenseMatrix<D>,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    trans_b: bool,
    output: &mut DenseMatrix<D>,
) {
    let output_shape = input_a.shape.maybe_transpose(trans_a) * input_b.shape.maybe_transpose(trans_b);

    match (input_a.shape.batch_size(), input_b.shape.batch_size()) {
        (Some(_), Some(_)) => D::sgemm_batched(input_a, trans_a, input_b, trans_b, output, false),
        (None, None) => {
            D::sgemm(input_a, input_a.shape, trans_a, input_b, input_b.shape, trans_b, output, output_shape, false);
        }
        (None, Some(x)) => {
            let shape_b = Shape::new(input_b.shape.rows(), x);
            if trans_b || input_b.shape.cols() > 1 {
                unimplemented!()
            }

            D::sgemm(input_a, input_a.shape, trans_a, input_b, shape_b, trans_b, output, output_shape, false);
        }
        (Some(_), None) => unimplemented!(),
    }
}

pub fn backprop_matmul<D: Device>(
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    trans_a: bool,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    trans_b: bool,
    output_grad: &DenseMatrix<D>,
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

#[allow(clippy::too_many_arguments)]
fn backprop_single_matmul<D: Device>(
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

fn backprop_batched_matmul<D: Device>(
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

pub fn submatrix_product<D: Device>(
    key_size: usize,
    input_a: &DenseMatrix<D>,
    input_b: &DenseMatrix<D>,
    output: &mut DenseMatrix<D>,
) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(input_a.shape.rows() % key_size, 0);
    assert_eq!(input_b.shape.rows() % key_size, 0);

    let batch_size = input_a.shape.batch_size();
    assert_eq!(batch_size, input_b.shape.batch_size());

    let shape_a = Shape::from_raw(key_size, input_a.shape.rows() / key_size, batch_size);
    let shape_b = Shape::from_raw(key_size, input_b.shape.rows() / key_size, batch_size);

    let output_size = shape_a.cols() * shape_b.cols();
    let output_shape = Shape::from_raw(output_size, 1, batch_size);
    output.reshape_if_needed(output_shape);
    D::sgemm_batched_reshaped(input_a, shape_a, true, input_b, shape_b, false, output, false);
}

pub fn backprop_submatrix_product<D: Device>(
    key_size: usize,
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    output_grad: &DenseMatrix<D>,
) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(input_a.shape.rows() % key_size, 0);
    assert_eq!(input_b.shape.rows() % key_size, 0);

    let batch_size = input_a.shape.batch_size();
    assert_eq!(batch_size, input_b.shape.batch_size());
    assert_eq!(batch_size, output_grad.shape.batch_size());

    let shape_a = Shape::from_raw(key_size, input_a.shape.rows() / key_size, batch_size);
    let shape_b = Shape::from_raw(key_size, input_b.shape.rows() / key_size, batch_size);
    let output_shape = shape_a.transpose() * shape_b;

    assert_eq!(output_grad.shape.rows(), output_shape.without_batch_size().size());

    if let Some(grad_a) = input_a_grad {
        grad_a.reshape_if_needed(input_a.shape);
        D::sgemm_batched_reshaped(input_b, shape_b, false, output_grad, output_shape, true, grad_a, true);
    }

    if let Some(grad_b) = input_b_grad {
        grad_b.reshape_if_needed(input_b.shape);
        D::sgemm_batched_reshaped(input_a, shape_a, false, output_grad, output_shape, false, grad_b, true);
    }
}
