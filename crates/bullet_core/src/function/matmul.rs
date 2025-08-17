use crate::{
    device::{
        Device, OperationError,
        blas::{BlasOperations, GemmConfig},
    },
    function::DeviceOperation,
    graph::ir::shape::Shape,
    tensor::TensorRef,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulType {
    BatBat,
    NobBat,
    NobNob,
    BatBatRed,
}

#[derive(Clone)]
pub struct Matmul<D: Device> {
    pub cfg: GemmConfig,
    pub input_a: TensorRef<D>,
    pub input_b: TensorRef<D>,
    pub output: TensorRef<D>,
    pub ty: MatmulType,
}

impl<D: Device> DeviceOperation<D> for Matmul<D> {
    fn opname(&self) -> String {
        format!("Matmul({:?})", self.ty)
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let Matmul { cfg, input_a, input_b, output, ty } = self;

        let input_a = input_a.borrow();
        let input_a = input_a.dense()?;
        let input_b = input_b.borrow();
        let input_b = input_b.dense()?;
        let mut output = output.borrow_mut();
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

                output.buf.gebmm(cfg, bs.unwrap_or(1), &input_a.buf, &input_b.buf)?;
            }
            MatmulType::NobNob => {
                if input_a.batch_size().is_some() || input_b.batch_size().is_some() || output.batch_size().is_some() {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                output.buf.gemm(cfg, &input_a.buf, &input_b.buf)?;
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
                let cfg = GemmConfig { shape_b, ..*cfg };
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
