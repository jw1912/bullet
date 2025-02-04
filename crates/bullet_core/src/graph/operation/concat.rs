use crate::{device::Device, shape::Shape, tensor::DenseMatrix};

pub fn concat<D: Device>(input_a: &DenseMatrix<D>, input_b: &DenseMatrix<D>, output: &mut DenseMatrix<D>) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(input_a.shape.batch_size(), input_b.shape.batch_size());

    let output_rows = input_a.shape.rows() + input_b.shape.rows();
    let output_shape = Shape::from_raw(output_rows, 1, input_a.shape.batch_size());
    output.reshape_if_needed(output_shape);

    D::copy_or_add_strided(
        input_a.shape.rows(),
        input_a.shape.batch_size().unwrap_or(1),
        &input_a.buf,
        0,
        input_a.shape.rows(),
        &mut output.buf,
        0,
        output_shape.rows(),
        false,
    );

    D::copy_or_add_strided(
        input_b.shape.rows(),
        input_b.shape.batch_size().unwrap_or(1),
        &input_b.buf,
        0,
        input_b.shape.rows(),
        &mut output.buf,
        input_a.shape.rows(),
        output_shape.rows(),
        false,
    );
}

pub fn backprop_concat<D: Device>(
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    output_grad: &DenseMatrix<D>,
) {
    assert_eq!(input_a.shape.cols(), 1);
    assert_eq!(input_b.shape.cols(), 1);
    assert_eq!(output_grad.shape.cols(), 1);
    assert_eq!(input_a.shape.batch_size(), input_b.shape.batch_size());
    assert_eq!(input_a.shape.batch_size(), output_grad.shape.batch_size());

    if let Some(grad) = input_a_grad {
        grad.reshape_if_needed(input_a.shape);

        D::copy_or_add_strided(
            grad.shape.rows(),
            grad.shape.batch_size().unwrap_or(1),
            &output_grad.buf,
            0,
            output_grad.shape.rows(),
            &mut grad.buf,
            0,
            grad.shape.rows(),
            true,
        );
    }

    if let Some(grad) = input_b_grad {
        grad.reshape_if_needed(input_b.shape);

        D::copy_or_add_strided(
            grad.shape.rows(),
            grad.shape.batch_size().unwrap_or(1),
            &output_grad.buf,
            input_a.shape.rows(),
            output_grad.shape.rows(),
            &mut grad.buf,
            0,
            grad.shape.rows(),
            true,
        );
    }
}
