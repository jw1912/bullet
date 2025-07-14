use crate::{
    device::{base::BaseOperations, Device, OperationError},
    graph::{instruction::GraphInstruction, ir::operation::unary::UnaryOp, Graph, NodeId},
};

#[derive(Debug)]
pub struct AbsPowerError {
    pub a: NodeId,
    pub b: NodeId,
    pub power: f32,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for AbsPowerError {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let a = graph.get(self.a)?;
        let a = a.dense()?;

        let b = graph.get(self.b)?;
        let b = b.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

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

#[derive(Debug)]
pub struct UnaryBackward {
    pub input: NodeId,
    pub output_grad: NodeId,
    pub input_grad: NodeId,
    pub op: UnaryOp,
}

impl<D: Device> GraphInstruction<D> for UnaryBackward {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let mut input_grad = graph.get_mut(self.input_grad)?;
        let input_grad = input_grad.dense_mut()?;
        let output_grad = graph.get(self.output_grad)?;
        let output_grad = output_grad.dense()?;
        let input = graph.get(self.input)?;
        let input = input.dense()?;

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

#[derive(Debug)]
pub struct PairwiseMulBackward {
    pub post_concat: bool,
    pub values: NodeId,
    pub input: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for PairwiseMulBackward {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;
        let values = graph.get(self.values)?;
        let values = values.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if input.batch_size() != output.batch_size() || input.batch_size() != values.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if output.single_size() != 2 * input.single_size() || output.single_size() != values.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let mut single_size = output.single_size();
        let mut batch_size = input.batch_size().unwrap_or(1);

        if self.post_concat {
            single_size /= 2;
            batch_size *= 2;
        }

        output.buf.pairwise_bwd(single_size, batch_size, &values.buf, &input.buf)?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct Select {
    pub input: NodeId,
    pub output: NodeId,
    pub buckets: NodeId,
}

impl<D: Device> GraphInstruction<D> for Select {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let buckets = graph.get(self.buckets)?;
        let buckets = buckets.sparse()?;

        if input.batch_size() != output.batch_size() || input.batch_size() != buckets.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let input_size = input.single_size();
        let output_size = output.single_size();

        if input_size != buckets.single_size() * output_size || buckets.nnz != 1 {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::select(
            input.batch_size().unwrap_or(1),
            input.single_size(),
            output.single_size(),
            &input.buf,
            &buckets.buf,
            &mut output.buf,
        )
    }
}

#[derive(Debug)]
pub struct SelectBackprop {
    pub input: NodeId,
    pub output: NodeId,
    pub buckets: NodeId,
}

impl<D: Device> GraphInstruction<D> for SelectBackprop {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let buckets = graph.get(self.buckets)?;
        let buckets = buckets.sparse()?;

        if input.batch_size() != output.batch_size() || input.batch_size() != buckets.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let input_size = input.single_size();
        let output_size = output.single_size();

        if output_size != buckets.single_size() * input_size || buckets.nnz != 1 {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::select_backprop(
            input.batch_size().unwrap_or(1),
            output.single_size(),
            input.single_size(),
            &buckets.buf,
            &input.buf,
            &mut output.buf,
        )
    }
}

#[derive(Debug)]
pub struct CrossEntropy {
    pub a: NodeId,
    pub b: NodeId,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for CrossEntropy {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let a = graph.get(self.a)?;
        let a = a.dense()?;

        let b = graph.get(self.b)?;
        let b = b.dense()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        if a.batch_size() != b.batch_size() || a.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if a.single_size() != b.single_size() || a.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::crossentropy(a.size(), &a.buf, &b.buf, &mut output.buf)
    }
}
