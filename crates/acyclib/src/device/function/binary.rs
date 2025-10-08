use crate::device::{
    Device, OperationError,
    function::{DeviceOperation, UnaryOp},
    operation::{BaseOperations, CoreDeviceOps},
    tensor::TensorRef,
};

#[derive(Clone)]
pub struct AbsPowerError<D: Device> {
    pub a: TensorRef<D>,
    pub b: TensorRef<D>,
    pub power: f32,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for AbsPowerError<D> {
    fn opname(&self) -> String {
        format!("AbsPowerError({})", self.power)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let a = self.a.dense();
        let b = self.b.dense();
        let mut output = self.output.dense_mut();

        if a.batch_size() != b.batch_size() || a.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if a.single_size() != b.single_size() || a.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        output.buf.power_error_fwd(self.power, a.size(), &a.buf, &b.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct UnaryBackward<D: Device> {
    pub input: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub input_grad: TensorRef<D>,
    pub op: UnaryOp,
}

impl<D: Device> DeviceOperation<D> for UnaryBackward<D> {
    fn opname(&self) -> String {
        format!("UnaryBackward({:?})", self.op)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let mut input_grad = self.input_grad.dense_mut();
        let output_grad = self.output_grad.dense();
        let input = self.input.dense();

        if input.batch_size() != input_grad.batch_size() || input.batch_size() != output_grad.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != input_grad.single_size() || input.single_size() != output_grad.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let size = output_grad.size();
        let igrd = &mut input_grad.buf;
        let ogrd = &output_grad.buf;

        match self.op {
            UnaryOp::AbsPow(p) => igrd.abs_pow_scalar_backward(size, p, &input.buf, ogrd)?,
            UnaryOp::Add(_) => igrd.linear_comb(size, 1.0, 1.0, ogrd)?,
            UnaryOp::Mul(x) => igrd.linear_comb(size, 1.0, x, ogrd)?,
            UnaryOp::DiffableFromOutput(act) => igrd.diffable_from_output_bwd(size, &input.buf, ogrd, act)?,
        }

        Ok(())
    }
}

pub struct PairwiseMulBackward<D: Device> {
    pub offset: usize,
    pub values: TensorRef<D>,
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for PairwiseMulBackward<D> {
    fn opname(&self) -> String {
        "PairwiseMulBackward".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let values = self.values.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() || input.batch_size() != values.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if output.single_size() > 2 * input.single_size() || output.single_size() != values.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let single_size = output.single_size();
        let stride = input.single_size();
        let batch_size = input.batch_size().unwrap_or(1);

        output.buf.pairwise_bwd(self.offset, stride, single_size, batch_size, &values.buf, &input.buf)?;

        Ok(())
    }
}

pub struct Select<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub buckets: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for Select<D> {
    fn opname(&self) -> String {
        "Select".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();
        let buckets = self.buckets.sparse();

        if output.batch_size() != buckets.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let input_size = input.single_size();
        let output_size = output.single_size();

        if input_size != buckets.single_size() * output_size || buckets.nnz() != 1 {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::select(
            output.batch_size().unwrap_or(1),
            input.batch_size().is_some(),
            input_size,
            output_size,
            &input.buf,
            &buckets.buf,
            &mut output.buf,
        )
    }
}

pub struct SelectBackprop<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub buckets: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for SelectBackprop<D> {
    fn opname(&self) -> String {
        "SelectBackprop".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();
        let buckets = self.buckets.sparse();

        if input.batch_size() != buckets.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let input_size = input.single_size();
        let output_size = output.single_size();

        if output_size != buckets.single_size() * input_size || buckets.nnz() != 1 {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::select_backprop(
            input.batch_size().unwrap_or(1),
            output.batch_size().is_some(),
            output_size,
            input_size,
            &buckets.buf,
            &input.buf,
            &mut output.buf,
        )
    }
}

pub struct CrossEntropy<D: Device> {
    pub a: TensorRef<D>,
    pub b: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for CrossEntropy<D> {
    fn opname(&self) -> String {
        "CrossEntropy".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let a = self.a.dense();
        let b = self.b.dense();
        let mut output = self.output.dense_mut();

        if a.batch_size() != b.batch_size() || a.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if a.single_size() != b.single_size() || a.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::crossentropy(a.size(), &a.buf, &b.buf, &mut output.buf)
    }
}
