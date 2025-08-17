use crate::{
    device::{Device, OperationError, base::BaseOperations},
    function::DeviceOperation,
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
        let a = self.a.borrow();
        let a = a.dense()?;
        let b = self.b.borrow();
        let b = b.dense()?;
        let c = self.c.borrow();
        let c = c.dense()?;

        let mut output = self.output.borrow_mut();
        let output = output.dense_mut()?;

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

impl<D: Device> DeviceOperation<D> for SoftmaxCrossEntropyBackward<D> {
    fn opname(&self) -> String {
        "SoftmaxCrossEntropyBackward".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let softmax = self.softmax.borrow();
        let softmax = softmax.dense()?;
        let targets = self.targets.borrow();
        let targets = targets.dense()?;
        let output_grads = self.output_grads.borrow();
        let output_grads = output_grads.dense()?;
        let mut output = self.output.borrow_mut();
        let output = output.dense_mut()?;

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
