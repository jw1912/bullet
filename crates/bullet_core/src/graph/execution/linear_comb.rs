use crate::{
    backend::{
        device::{
            blas::{BlasOperations, GemmConfig},
            Device, DeviceBuffer, OperationError, OperationResult,
        },
        tensor::DenseMatrix,
    },
    graph::ir::shape::Shape,
};

#[allow(clippy::too_many_arguments)]
pub fn linear_comb<D: Device>(
    ones: &D::BufferF32,
    alpha: f32,
    input_a: &DenseMatrix<D>,
    shape_a: Shape,
    beta: f32,
    input_b: &DenseMatrix<D>,
    shape_b: Shape,
    output: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(shape_a, shape_b);
    let size = shape_a.size();

    match (input_a.batch_size(), input_b.batch_size()) {
        (Some(x), Some(y)) => {
            assert_eq!(x, y, "Batch sizes do not match: {x} != {y}");
            output.set_batch_size(Some(x))?;
            output.buf.geam(input_a.size(), alpha, Some(&input_a.buf), beta, Some(&input_b.buf))?;
        }
        (None, Some(bs)) => {
            output.set_batch_size(Some(bs))?;
            copy_into_scaled(beta, input_b, output)?;
            add_assign_single_to_batched_scaled::<D>(size, bs, ones, alpha, &input_a.buf, &mut output.buf)?;
        }
        (_, None) => {
            output.set_batch_size(input_a.batch_size())?;
            let bs = input_a.batch_size().unwrap_or(1);
            copy_into_scaled(alpha, input_a, output)?;
            add_assign_single_to_batched_scaled::<D>(size, bs, ones, beta, &input_b.buf, &mut output.buf)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn linear_comb_backward<D: Device>(
    ones: &D::BufferF32,
    alpha: f32,
    input_a: &DenseMatrix<D>,
    input_a_grad: Option<&mut DenseMatrix<D>>,
    beta: f32,
    input_b: &DenseMatrix<D>,
    input_b_grad: Option<&mut DenseMatrix<D>>,
    output_grad: &DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    if let Some(grd) = input_a_grad {
        backprop_add_single_scaled(ones, alpha, input_a, grd, output_grad)?;
    }

    if let Some(grd) = input_b_grad {
        backprop_add_single_scaled(ones, beta, input_b, grd, output_grad)?;
    }

    Ok(())
}

fn copy_into_scaled<D: Device>(
    alpha: f32,
    input: &DenseMatrix<D>,
    output: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(input.single_size(), output.single_size());
    output.set_batch_size(input.batch_size())?;
    output.buf.geam(input.size(), alpha, Some(&input.buf), 0.0, None)?;
    Ok(())
}

fn add_assign_scaled<D: Device>(
    alpha: f32,
    input: &DenseMatrix<D>,
    output: &mut DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(input.single_size(), output.single_size());
    output.set_batch_size(input.batch_size())?;
    output.buf.geam(input.size(), 1.0, None, alpha, Some(&input.buf))?;
    Ok(())
}

pub fn backprop_add_single_scaled<D: Device>(
    ones: &D::BufferF32,
    alpha: f32,
    input: &DenseMatrix<D>,
    input_grad: &mut DenseMatrix<D>,
    output_grad: &DenseMatrix<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    assert_eq!(input.single_size(), output_grad.single_size());
    assert_eq!(input.single_size(), input_grad.single_size());
    input_grad.set_batch_size(input.batch_size())?;

    match (input.batch_size(), output_grad.batch_size()) {
        (Some(_), Some(_)) | (None, None) => add_assign_scaled(alpha, output_grad, input_grad),
        (None, Some(x)) => {
            assert!(output_grad.batch_size().unwrap_or(1) <= ones.size());
            reduce_add::<D>(ones, input.single_size(), x, &output_grad.buf, &mut input_grad.buf, true)
        }
        (Some(_), None) => Err(OperationError::UnsupportedOperation),
    }
}

pub fn reduce_add<D: Device>(
    ones: &D::BufferF32,
    size: usize,
    batch_size: usize,
    input: &D::BufferF32,
    output: &mut D::BufferF32,
    onto: bool,
) -> OperationResult<D::DeviceError> {
    let cfg =
        GemmConfig::new(1.0, f32::from(onto), Shape::new(size, batch_size), false, Shape::new(batch_size, 1), false);
    output.gemm(&cfg, input, ones)?;
    Ok(())
}

pub fn add_assign_single_to_batched_scaled<D: Device>(
    single_size: usize,
    batch_size: usize,
    ones: &D::BufferF32,
    alpha: f32,
    input: &D::BufferF32,
    output: &mut D::BufferF32,
) -> OperationResult<D::DeviceError> {
    let cfg = GemmConfig::new(alpha, 1.0, Shape::new(single_size, 1), false, Shape::new(1, batch_size), false);
    output.gemm(&cfg, input, ones)?;
    Ok(())
}
