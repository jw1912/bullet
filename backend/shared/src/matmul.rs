use bullet_core::{device::DeviceBuffer, matmul::Matmul, shape::Shape};

use crate::{backend::blas, DenseMatrix, ExecutionContext};

impl Matmul for ExecutionContext {
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        input_a: &DenseMatrix,
        shape_a: Shape,
        trans_a: bool,
        input_b: &DenseMatrix,
        shape_b: Shape,
        trans_b: bool,
        output: &mut DenseMatrix,
        output_shape: Shape,
        increment: bool,
    ) {
        assert!(shape_a.batch_size().is_none());
        assert!(shape_b.batch_size().is_none());

        let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);
        output.reshape_if_needed(output_shape);
        assert_eq!(output_shape.size(), shape_o.size());

        unsafe {
            blas::sgemm(
                input_a.buf.device().as_ref(),
                input_a.buf.ptr(),
                shape_a.rows(),
                shape_a.cols(),
                trans_a,
                input_b.buf.ptr(),
                shape_b.rows(),
                shape_b.cols(),
                trans_b,
                output.buf.mut_ptr(),
                shape_o.rows(),
                shape_o.cols(),
                increment,
            );
        }
    }

    fn sgemm_batched(
        input_a: &DenseMatrix,
        trans_a: bool,
        input_b: &DenseMatrix,
        trans_b: bool,
        output: &mut DenseMatrix,
        increment: bool,
    ) {
        assert_eq!(input_a.shape.batch_size(), input_b.shape.batch_size());

        let output_shape = input_a.shape.maybe_transpose(trans_a) * input_b.shape.maybe_transpose(trans_b);
        let batch_size = input_a.shape.batch_size().unwrap_or(1);
        output.reshape_if_needed(output_shape);

        unsafe {
            blas::batched_sgemm(
                input_a.buf.device().as_ref(),
                batch_size,
                input_a.buf.ptr(),
                input_a.shape.rows(),
                input_a.shape.cols(),
                trans_a,
                input_b.buf.ptr(),
                input_b.shape.rows(),
                input_b.shape.cols(),
                trans_b,
                output.buf.mut_ptr(),
                output_shape.rows(),
                output_shape.cols(),
                increment,
            );
        }
    }
}
