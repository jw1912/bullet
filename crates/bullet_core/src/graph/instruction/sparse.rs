use crate::{
    device::{Device, OperationError},
    graph::{builder::Shape, instruction::GraphInstruction, ir::operation::unary::DiffableFromOutput, Graph, NodeId},
};

#[derive(Clone, Copy, Debug)]
pub struct SparseAffineActivateStrided {
    pub weights: NodeId,
    pub weights_shape: Shape,
    pub biases: Option<NodeId>,
    pub input_shape: Shape,
    pub indices: NodeId,
    pub values: Option<NodeId>,
    pub stride: Option<bool>,
    pub activation: DiffableFromOutput,
    pub output: NodeId,
}

impl<D: Device> GraphInstruction<D> for SparseAffineActivateStrided {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<D::DeviceError>> {
        let SparseAffineActivateStrided {
            weights,
            weights_shape,
            biases,
            input_shape,
            indices,
            values,
            stride,
            activation,
            output,
        } = *self;

        let weights = graph.get(weights)?;
        let weights = weights.dense()?;
        let indices = graph.get(indices)?;
        let indices = indices.sparse()?;
        let mut output = graph.get_mut(output)?;
        let output = output.dense_mut()?;

        let biases = if let Some(b) = biases { Some(graph.get(b)?) } else { None };
        let biases = if let Some(b) = &biases { Some(b.dense()?) } else { None };

        let values = if let Some(v) = values { Some(graph.get(v)?) } else { None };
        let values = if let Some(v) = &values { Some(v.dense()?) } else { None };

        let batch_size = indices.batch_size();

        if batch_size != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        D::sparse_affine_activate(
            batch_size.unwrap_or(1),
            stride,
            activation,
            &weights.buf,
            weights_shape,
            &indices.buf,
            values.map(|v| &v.buf),
            input_shape,
            indices.nnz,
            biases.map(|b| &b.buf),
            biases.map(|b| b.batch_size.is_some()).unwrap_or(false),
            &mut output.buf,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BackpropSparseAffineActivateStrided {
    pub weights_grads: NodeId,
    pub weights_shape: Shape,
    pub biases_grads: Option<NodeId>,
    pub input_shape: Shape,
    pub indices: NodeId,
    pub values: Option<NodeId>,
    pub stride: Option<bool>,
    pub activation: DiffableFromOutput,
    pub output: NodeId,
    pub output_grads: NodeId,
}

impl<D: Device> GraphInstruction<D> for BackpropSparseAffineActivateStrided {
    fn execute(&self, graph: &Graph<D>) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let BackpropSparseAffineActivateStrided {
            weights_grads,
            weights_shape,
            biases_grads,
            input_shape,
            indices,
            values,
            stride,
            activation,
            output,
            output_grads,
        } = *self;

        let mut weights_grads = graph.get_mut(weights_grads)?;
        let weights_grads = weights_grads.dense_mut()?;
        let indices = graph.get(indices)?;
        let indices = indices.sparse()?;
        let output = graph.get(output)?;
        let output = output.dense()?;
        let output_grads = graph.get(output_grads)?;
        let output_grads = output_grads.dense()?;

        let mut biases_grads = if let Some(b) = biases_grads { Some(graph.get_mut(b)?) } else { None };
        let biases_grads = if let Some(ref mut b) = &mut biases_grads { Some(b.dense_mut()?) } else { None };

        let values = if let Some(v) = values { Some(graph.get(v)?) } else { None };
        let values = if let Some(v) = &values { Some(v.dense()?) } else { None };

        let biases_batched = biases_grads.as_ref().map(|b| b.batch_size.is_some()).unwrap_or(false);

        let batch_size = indices.batch_size();

        if batch_size != output.batch_size() || batch_size != output_grads.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        D::backprop_sparse_affine_activate(
            batch_size.unwrap_or(1),
            stride,
            activation,
            &mut weights_grads.buf,
            weights_shape,
            &indices.buf,
            values.map(|v| &v.buf),
            input_shape,
            indices.nnz,
            biases_grads.map(|b| &mut b.buf),
            biases_batched,
            &output.buf,
            &output_grads.buf,
        )
    }
}
