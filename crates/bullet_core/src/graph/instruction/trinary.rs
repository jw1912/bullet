use crate::{
    backend::device::{base::BaseOperations, Device, OperationError},
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
