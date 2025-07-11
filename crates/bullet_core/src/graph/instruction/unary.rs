use crate::{
    backend::device::{base::BaseOperations, Device, OperationError},
    graph::{instruction::GraphInstruction, ir::operation::unary::UnaryOp, Graph, NodeId},
};

pub struct SetBatchSize {
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for SetBatchSize {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if output.batch_size() != input.batch_size() {
            output.set_batch_size(input.batch_size())?;
        }

        Ok(())
    }
}

pub struct LinearCombination {
    pub input_mul: f32,
    pub output_mul: f32,
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for LinearCombination {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        output.buf.linear_comb(input.size(), self.output_mul, self.input_mul, &input.buf)?;

        Ok(())
    }
}

pub struct LinearCombinationSplat {
    pub input_mul: f32,
    pub output_mul: f32,
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for LinearCombinationSplat {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if input.batch_size().is_some() || output.batch_size().is_none() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = output.batch_size().unwrap_or(1);
        output.buf.linear_comb_splat(input.size(), bs, self.output_mul, self.input_mul, &input.buf)?;

        Ok(())
    }
}

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

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::sparse_to_dense(input.batch_size().unwrap_or(1), input.single_size, input.nnz, &input.buf, &mut output.buf)
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

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != 2 * output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let mut single_size = input.single_size();
        let mut batch_size = input.batch_size().unwrap_or(1);

        if self.post_concat {
            single_size /= 2;
            batch_size *= 2;
        }

        output.buf.pairwise_fwd(single_size, batch_size, &input.buf)?;

        Ok(())
    }
}

pub struct Unary {
    pub input: NodeId,
    pub output: NodeId,
    pub op: UnaryOp,
}

impl<D: Device> GraphInstruction<D> for Unary {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let size = input.size();

        match self.op {
            UnaryOp::AbsPow(p) => output.buf.abs_pow_scalar(size, p, &input.buf)?,
            UnaryOp::Add(x) => output.buf.add_scalar(size, x, &input.buf)?,
            UnaryOp::Mul(x) => output.buf.linear_comb(size, 0.0, x, &input.buf)?,
            UnaryOp::DiffableFromOutput(act) => output.buf.diffable_from_output_fwd(size, &input.buf, act)?,
        }

        Ok(())
    }
}
