use crate::{
    backend::device::{
        blas::{BlasOperations, GemmConfig},
        Device, OperationError,
    },
    graph::{builder::Shape, Graph, NodeId},
};

use super::GraphInstruction;

#[derive(Clone, Copy)]
pub struct Matmul {
    pub alpha: f32,
    pub beta: f32,
    pub input_a: NodeId,
    pub shape_a: Shape,
    pub trans_a: bool,
    pub input_b: NodeId,
    pub shape_b: Shape,
    pub trans_b: bool,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for Matmul {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let Matmul { alpha, beta, input_a, shape_a, trans_a, input_b, shape_b, trans_b, output } = *self;

        let input_a = graph.get(input_a)?;
        let input_a = input_a.dense()?;
        let input_b = graph.get(input_b)?;
        let input_b = input_b.dense()?;
        let mut output = graph.get_mut(output)?;
        let output = output.dense_mut()?;

        let output_shape = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

        if input_a.single_size() != shape_a.size()
            || input_b.single_size() != shape_b.size()
            || output.single_size() != output_shape.size()
        {
            return Err(OperationError::InvalidTensorFormat);
        }

        match (input_a.batch_size(), input_b.batch_size()) {
            (Some(x), Some(y)) => {
                if x != y {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                let cfg = GemmConfig::new(alpha, beta, shape_a, trans_a, shape_b, trans_b);
                output.set_batch_size(Some(x))?;
                output.buf.gebmm(&cfg, x, &input_a.buf, &input_b.buf)?;
            }
            (None, None) => {
                let cfg = GemmConfig::new(alpha, beta, shape_a, trans_a, shape_b, trans_b);
                output.set_batch_size(None)?;
                output.buf.gemm(&cfg, &input_a.buf, &input_b.buf)?;
            }
            (None, Some(x)) => {
                if trans_b {
                    return Err(OperationError::UnsupportedOperation);
                }

                let shape_b = Shape::new(shape_b.rows(), x * shape_b.cols());
                let cfg = GemmConfig::new(alpha, beta, shape_a, trans_a, shape_b, trans_b);
                output.set_batch_size(Some(x))?;
                output.buf.gemm(&cfg, &input_a.buf, &input_b.buf)?;
            }
            (Some(_), None) => return Err(OperationError::UnsupportedOperation),
        }

        Ok(())
    }
}
