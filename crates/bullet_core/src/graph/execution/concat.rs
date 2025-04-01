use crate::{
    backend::{
        device::{base::BaseOperations, Device, OperationError},
        tensor::DenseMatrix,
    },
    graph::ir::shape::Shape,
};

pub fn concat<D: Device>(
    input_a: &DenseMatrix<D>,
    shape_a: Shape,
    input_b: &DenseMatrix<D>,
    shape_b: Shape,
    output: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(shape_a.cols(), 1);
    assert_eq!(shape_b.cols(), 1);
    assert_eq!(input_a.batch_size(), input_b.batch_size());

    let shape_o = Shape::new(shape_a.rows() + shape_b.rows(), 1);

    assert_eq!(shape_a.size(), input_a.single_size());
    assert_eq!(shape_b.size(), input_b.single_size());
    assert_eq!(shape_o.size(), output.single_size());

    output.set_batch_size(input_a.batch_size())?;

    output.buf.copy_or_add_strided(
        false,
        shape_a.rows(),
        input_a.batch_size().unwrap_or(1),
        0,
        shape_o.rows(),
        &input_a.buf,
        0,
        shape_a.rows(),
    )?;

    output.buf.copy_or_add_strided(
        false,
        shape_b.rows(),
        input_b.batch_size().unwrap_or(1),
        shape_a.rows(),
        shape_o.rows(),
        &input_b.buf,
        0,
        shape_b.rows(),
    )?;

    Ok(())
}

pub fn backprop_concat<D: Device>(
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    shape_a: Shape,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    shape_b: Shape,
    output_grad: &DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(shape_a.cols(), 1);
    assert_eq!(shape_b.cols(), 1);
    assert_eq!(input_a.batch_size(), input_b.batch_size());
    assert_eq!(input_a.batch_size(), output_grad.batch_size());

    let shape_o = Shape::new(shape_a.rows() + shape_b.rows(), 1);

    assert_eq!(shape_a.size(), input_a.single_size());
    assert_eq!(shape_b.size(), input_b.single_size());
    assert_eq!(shape_o.size(), output_grad.single_size());

    if let Some(grad) = input_a_grad {
        assert_eq!(grad.single_size(), input_a.single_size());
        grad.set_batch_size(input_a.batch_size())?;

        grad.buf.copy_or_add_strided(
            true,
            shape_a.rows(),
            grad.batch_size().unwrap_or(1),
            0,
            shape_a.rows(),
            &output_grad.buf,
            0,
            shape_o.rows(),
        )?;
    }

    if let Some(grad) = input_b_grad {
        assert_eq!(grad.single_size(), input_b.single_size());
        grad.set_batch_size(input_b.batch_size())?;

        grad.buf.copy_or_add_strided(
            true,
            shape_b.rows(),
            grad.batch_size().unwrap_or(1),
            0,
            shape_b.rows(),
            &output_grad.buf,
            shape_a.rows(),
            shape_o.rows(),
        )?;
    }

    Ok(())
}
