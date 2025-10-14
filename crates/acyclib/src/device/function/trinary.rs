use crate::device::{
    Device, OperationError,
    function::DeviceOperation,
    operation::{BaseOperations, CoreDeviceOps},
    tensor::TensorRef,
};

#[derive(Debug)]
pub struct AbsPowerErrorBackward<D: Device> {
    pub a: TensorRef<D>,
    pub b: TensorRef<D>,
    pub c: TensorRef<D>,
    pub output: TensorRef<D>,
    pub power: f32,
}

impl<D: Device> DeviceOperation<D> for AbsPowerErrorBackward<D> {
    fn opname(&self) -> String {
        format!("AbsPowerErrorBackward({:?})", self.power)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let a = self.a.dense();
        let b = self.b.dense();
        let c = self.c.dense();
        let mut output = self.output.dense_mut();

        if a.batch_size() != b.batch_size() || a.batch_size() != c.batch_size() || a.batch_size() != output.batch_size()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if a.single_size() != b.single_size()
            || a.single_size() != c.single_size()
            || a.single_size() != output.single_size()
        {
            return Err(OperationError::InvalidTensorFormat);
        }

        output.buf.power_error_bwd(self.power, a.size(), &a.buf, &b.buf, &c.buf)?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct SoftmaxCrossEntropyBackward<D: Device> {
    pub softmax: TensorRef<D>,
    pub targets: TensorRef<D>,
    pub output_grads: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for SoftmaxCrossEntropyBackward<D> {
    fn opname(&self) -> String {
        "SoftmaxCrossEntropyBackward".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let softmax = self.softmax.dense();
        let targets = self.targets.dense();
        let output_grads = self.output_grads.dense();
        let mut output = self.output.dense_mut();

        if softmax.batch_size() != targets.batch_size()
            || softmax.batch_size() != output_grads.batch_size()
            || softmax.batch_size() != output.batch_size()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if softmax.single_size() != targets.single_size()
            || softmax.single_size() != output_grads.single_size()
            || softmax.single_size() != output.single_size()
        {
            return Err(OperationError::InvalidTensorFormat);
        }

        let size = softmax.size();
        D::backprop_softmax_crossentropy(size, &softmax.buf, &targets.buf, &output_grads.buf, &mut output.buf)
    }
}

#[derive(Debug)]
pub struct BCELogitLossBackward<D: Device> {
    pub input: TensorRef<D>,
    pub target: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for BCELogitLossBackward<D> {
    fn opname(&self) -> String {
        "BCELogitLossBackward".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let target = self.target.dense();
        let output_grad = self.output_grad.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != target.batch_size()
            || input.batch_size() != output_grad.batch_size()
            || input.batch_size() != output.batch_size()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != target.single_size()
            || input.single_size() != output_grad.single_size()
            || input.single_size() != output.single_size()
        {
            return Err(OperationError::InvalidTensorFormat);
        }

        output.buf.bce_logit_loss_bwd(input.size(), &input.buf, &target.buf, &output_grad.buf)?;

        Ok(())
    }
}
