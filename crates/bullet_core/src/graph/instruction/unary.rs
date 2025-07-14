use crate::{
    device::{base::BaseOperations, Device, OperationError},
    graph::{
        instruction::GraphInstruction,
        ir::operation::unary::{Reduce, UnaryOp},
        Graph, NodeId,
    },
};

#[derive(Debug)]
pub struct MaybeUpdateBatchSize {
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for MaybeUpdateBatchSize {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if output.batch_size() != input.values.batch_size() {
            output.set_batch_size(input.values.batch_size())?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct ReduceAcrossBatch {
    pub input: NodeId,
    pub output: NodeId,
    pub input_mul: f32,
    pub output_mul: f32,
    pub reduction: Reduce,
}

impl<D: Device> GraphInstruction<D> for ReduceAcrossBatch {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if input.batch_size().is_none() || output.batch_size().is_some() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = input.batch_size().unwrap_or(1);

        let scale = match self.reduction {
            Reduce::Avg => 1.0 / bs as f32,
            Reduce::Sum => 1.0,
        };

        output.buf.reduce_across_batch(input.single_size(), bs, self.output_mul, self.input_mul * scale, &input.buf)?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct SplatAcrossBatch {
    pub input: NodeId,
    pub output: NodeId,
    pub input_mul: f32,
    pub output_mul: f32,
    pub reduction: Reduce,
}

impl<D: Device> GraphInstruction<D> for SplatAcrossBatch {
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

        let scale = match self.reduction {
            Reduce::Avg => 1.0 / bs as f32,
            Reduce::Sum => 1.0,
        };

        output.buf.linear_comb_splat(input.single_size(), bs, self.output_mul, self.input_mul * scale, &input.buf)?;

        Ok(())
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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
            println!("{:?} {:?}", input.batch_size(), output.batch_size());
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
pub struct CopyOrAddStrided {
    pub input: NodeId,
    pub output: NodeId,
    pub input_offset: usize,
    pub output_offset: usize,
    pub add: bool,
    pub len_is_out: bool,
}

impl<D: Device> GraphInstruction<D> for CopyOrAddStrided {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let rows = if self.len_is_out { output.single_size() } else { input.single_size() };

        output.buf.copy_or_add_strided(
            self.add,
            rows,
            input.batch_size().unwrap_or(1),
            self.output_offset,
            output.single_size(),
            &input.buf,
            self.input_offset,
            input.single_size(),
        )?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct Softmax {
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for Softmax {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let batch_size = input.batch_size();
        let single_size = input.single_size();

        if batch_size != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if single_size != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::softmax_across_batch(batch_size.unwrap_or(1), single_size, &input.buf, &mut output.buf)
    }
}
