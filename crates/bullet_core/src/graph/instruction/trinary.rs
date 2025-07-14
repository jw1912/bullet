use crate::{
    device::{base::BaseOperations, Device, OperationError},
    graph::{instruction::GraphInstruction, Graph, NodeId},
};

#[derive(Debug)]
pub struct AbsPowerErrorBackward {
    pub a: NodeId,
    pub b: NodeId,
    pub c: NodeId,
    pub output: NodeId,
    pub power: f32,
}

impl<D: Device> GraphInstruction<D> for AbsPowerErrorBackward {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let a = graph.get(self.a)?;
        let a = a.dense()?;

        let b = graph.get(self.b)?;
        let b = b.dense()?;

        let c = graph.get(self.c)?;
        let c = c.dense()?;

        let mut output = graph.get_mut(self.output)?;
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
pub struct SoftmaxCrossEntropyBackward {
    pub softmax: NodeId,
    pub targets: NodeId,
    pub output_grads: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for SoftmaxCrossEntropyBackward {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let softmax = graph.get(self.softmax)?;
        let softmax = softmax.dense()?;

        let targets = graph.get(self.targets)?;
        let targets = targets.dense()?;

        let output_grads = graph.get(self.output_grads)?;
        let output_grads = output_grads.dense()?;

        let mut output = graph.get_mut(self.output)?;
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
