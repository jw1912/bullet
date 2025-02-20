use crate::{
    device::{Device, OperationError},
    shape::Shape,
    tensor::DenseMatrix,
};

pub fn slice_vector_batched<D: Device>(
    shape: Shape,
    input: &DenseMatrix<D>,
    start: usize,
    end: usize,
    output: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(shape.cols(), 1);
    assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");
    assert!(
        end <= shape.rows(),
        "Slice index out of bounds! Number of rows is {} but slice endpoint is {end}!",
        shape.rows()
    );

    let output_shape = Shape::new(end - start, 1);
    output.set_batch_size(input.batch_size())?;

    D::copy_or_add_strided(
        output_shape.rows(),
        input.batch_size().unwrap_or(1),
        &input.buf,
        start,
        shape.rows(),
        &mut output.buf,
        0,
        output_shape.rows(),
        false,
    )
}

pub fn backprop_slice_vector_batched<D: Device>(
    shape: Shape,
    input: &DenseMatrix<D>,
    input_grad: &mut DenseMatrix<D>,
    start: usize,
    end: usize,
    output_grad: &DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(shape.cols(), 1);
    assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");
    assert!(
        end <= shape.rows(),
        "Slice index out of bounds! Number of rows is {} but slice endpoint is {end}!",
        shape.rows()
    );

    let output_shape = Shape::new(end - start, 1);

    assert_eq!(input.single_size, shape.size());
    assert_eq!(input.single_size, input_grad.single_size);
    assert_eq!(input.batch_size, output_grad.batch_size);
    assert_eq!(output_grad.single_size, output_shape.size());

    input_grad.set_batch_size(input.batch_size())?;

    D::copy_or_add_strided(
        output_shape.rows(),
        input.batch_size().unwrap_or(1),
        &output_grad.buf,
        0,
        output_shape.rows(),
        &mut input_grad.buf,
        start,
        shape.rows(),
        true,
    )
}
