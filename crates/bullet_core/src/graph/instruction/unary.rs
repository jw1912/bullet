use crate::{
    backend::device::{base::BaseOperations, Device, OperationError},
    graph::{instruction::GraphInstruction, Graph, NodeId},
};

pub struct SparseToDense {
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for SparseToDense {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.sparse()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let batch_size = input.batch_size();
        output.set_batch_size(batch_size)?;
        D::sparse_to_dense(batch_size.unwrap_or(1), input.single_size, input.nnz, &input.buf, &mut output.buf)
    }
}

pub struct PairwiseMul {
    pub post_concat: bool,
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for PairwiseMul {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let batch_size = input.batch_size();
        output.set_batch_size(batch_size)?;

        let mut single_size = input.single_size();
        let mut batch_size = batch_size.unwrap_or(1);

        if self.post_concat {
            single_size /= 2;
            batch_size *= 2;
        }

        output.buf.pairwise_fwd(single_size, batch_size, &input.buf)?;

        Ok(())
    }
}
