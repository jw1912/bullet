use crate::{device::Device, shape::Shape, tensor::DenseMatrix};

pub fn slice_vector_batched<D: Device>(input: &DenseMatrix<D>, start: usize, end: usize, output: &mut DenseMatrix<D>) {
    assert_eq!(input.shape.cols(), 1);
    assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");
    assert!(
        end <= input.shape.rows(),
        "Slice index out of bounds! Number of rows is {} but slice endpoint is {end}!",
        input.shape.rows()
    );

    let output_shape = Shape::from_raw(end - start, 1, input.shape.batch_size());
    output.reshape_if_needed(output_shape);

    D::copy_or_add_strided(
        output_shape.rows(),
        input.shape.batch_size().unwrap_or(1),
        &input.buf,
        start,
        input.shape.rows(),
        &mut output.buf,
        0,
        output_shape.rows(),
        false,
    );
}

pub fn backprop_slice_vector_batched<D: Device>(
    input: &DenseMatrix<D>,
    input_grad: &mut DenseMatrix<D>,
    start: usize,
    end: usize,
    output_grad: &DenseMatrix<D>,
) {
    assert_eq!(input.shape.cols(), 1);
    assert!(end > start, "Invalid slice indices! end = {end} > start = {start}");
    assert!(
        end <= input.shape.rows(),
        "Slice index out of bounds! Number of rows is {} but slice endpoint is {end}!",
        input.shape.rows()
    );
    let output_shape = Shape::from_raw(end - start, 1, input.shape.batch_size());
    assert_eq!(output_shape, output_grad.shape);

    input_grad.reshape_if_needed(input.shape);

    D::copy_or_add_strided(
        output_shape.rows(),
        output_shape.batch_size().unwrap_or(1),
        &output_grad.buf,
        0,
        output_shape.rows(),
        &mut input_grad.buf,
        start,
        input_grad.shape.rows(),
        true,
    );
}
