use crate::{
    device::{
        blas::{BlasOperations, GemmConfig},
        Device, OperationError,
    },
    graph::{builder::Shape, Graph, NodeId},
};

use super::GraphInstruction;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulType {
    BatBat,
    NobBat,
    NobNob,
    BatBatRed,
}

#[derive(Clone, Copy, Debug)]
pub struct Matmul {
    pub cfg: GemmConfig,
    pub input_a: NodeId,
    pub input_b: NodeId,
    pub output: NodeId,
    pub ty: MatmulType,
}

impl<D: Device> GraphInstruction<D> for Matmul {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let Matmul { cfg, input_a, input_b, output, ty } = *self;

        let input_a = graph.get(input_a)?;
        let input_a = input_a.dense()?;
        let input_b = graph.get(input_b)?;
        let input_b = input_b.dense()?;
        let mut output = graph.get_mut(output)?;
        let output = output.dense_mut()?;

        if input_a.single_size() != cfg.shape_a.size() || input_b.single_size() != cfg.shape_b.size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        match ty {
            MatmulType::BatBat => {
                let bs = input_a.batch_size();

                if bs != input_b.batch_size() || bs != output.batch_size() {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                output.buf.gebmm(&cfg, bs.unwrap_or(1), &input_a.buf, &input_b.buf)?;
            }
            MatmulType::NobNob => {
                if input_a.batch_size().is_some() || input_b.batch_size().is_some() || output.batch_size().is_some() {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                output.buf.gemm(&cfg, &input_a.buf, &input_b.buf)?;
            }
            MatmulType::NobBat => {
                if input_a.batch_size().is_some()
                    || input_b.batch_size().is_none()
                    || input_b.batch_size() != output.batch_size()
                {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                if cfg.trans_b {
                    return Err(OperationError::UnsupportedOperation);
                }

                let bs = input_b.batch_size().unwrap();

                let shape_b = Shape::new(cfg.shape_b.rows(), bs * cfg.shape_b.cols());
                let cfg = GemmConfig { shape_b, ..cfg };
                output.buf.gemm(&cfg, &input_a.buf, &input_b.buf)?;
            }
            MatmulType::BatBatRed => {
                if input_a.batch_size() != input_b.batch_size() || output.batch_size().is_some() {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                if !cfg.trans_b {
                    return Err(OperationError::UnsupportedOperation);
                }

                let bs = input_a.batch_size().unwrap_or(1);

                let cfg = GemmConfig {
                    shape_a: Shape::new(self.cfg.shape_a.rows(), bs * self.cfg.shape_a.cols()),
                    shape_b: Shape::new(self.cfg.shape_b.rows(), bs * self.cfg.shape_b.cols()),
                    ..self.cfg
                };

                output.buf.gemm(&cfg, &input_a.buf, &input_b.buf)?;
            }
        }

        Ok(())
    }
}
