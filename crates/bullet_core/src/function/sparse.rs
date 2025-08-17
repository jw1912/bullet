use crate::{
    device::{Device, OperationError},
    function::DeviceOperation,
    graph::ir::{operation::unary::DiffableFromOutput, shape::Shape},
    tensor::TensorRef,
};

#[derive(Clone)]
pub struct SparseAffineActivate<D: Device> {
    pub weights: TensorRef<D>,
    pub weights_shape: Shape,
    pub biases: Option<TensorRef<D>>,
    pub input_shape: Shape,
    pub indices: TensorRef<D>,
    pub values: Option<TensorRef<D>>,
    pub activation: DiffableFromOutput,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for SparseAffineActivate<D> {
    fn opname(&self) -> String {
        "SparseAffineActivate".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let SparseAffineActivate { weights, weights_shape, biases, input_shape, indices, values, activation, output } =
            self;

        let weights = weights.borrow();
        let weights = weights.dense()?;
        let indices = indices.borrow();
        let indices = indices.sparse()?;
        let mut output = output.borrow_mut();
        let output = output.dense_mut()?;

        let biases = biases.as_ref().map(|b| b.borrow());
        let biases = if let Some(b) = &biases { Some(b.dense()?) } else { None };
        let values = values.as_ref().map(|v| v.borrow());
        let values = if let Some(v) = &values { Some(v.dense()?) } else { None };

        let batch_size = indices.batch_size();

        if batch_size != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        D::sparse_affine_activate(
            batch_size.unwrap_or(1),
            *activation,
            &weights.buf,
            *weights_shape,
            &indices.buf,
            values.map(|v| &v.buf),
            *input_shape,
            indices.nnz,
            biases.map(|b| &b.buf),
            biases.map(|b| b.batch_size.is_some()).unwrap_or(false),
            &mut output.buf,
        )
    }
}

#[derive(Clone)]
pub struct BackpropSparseAffineActivate<D: Device> {
    pub weights_grads: TensorRef<D>,
    pub weights_shape: Shape,
    pub biases_grads: Option<TensorRef<D>>,
    pub input_shape: Shape,
    pub indices: TensorRef<D>,
    pub values: Option<TensorRef<D>>,
    pub activation: DiffableFromOutput,
    pub output: TensorRef<D>,
    pub output_grads: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for BackpropSparseAffineActivate<D> {
    fn opname(&self) -> String {
        "BackpropSparseAffineActivate".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let BackpropSparseAffineActivate {
            weights_grads,
            weights_shape,
            biases_grads,
            input_shape,
            indices,
            values,
            activation,
            output,
            output_grads,
        } = self;

        let mut weights_grads = weights_grads.borrow_mut();
        let weights_grads = weights_grads.dense_mut()?;
        let indices = indices.borrow();
        let indices = indices.sparse()?;
        let output = output.borrow();
        let output = output.dense()?;
        let output_grads = output_grads.borrow();
        let output_grads = output_grads.dense()?;

        let mut biases_grads = biases_grads.as_ref().map(|b| b.borrow_mut());
        let biases_grads = if let Some(b) = &mut biases_grads { Some(b.dense_mut()?) } else { None };
        let values = values.as_ref().map(|v| v.borrow());
        let values = if let Some(v) = &values { Some(v.dense()?) } else { None };

        let biases_batched = biases_grads.as_ref().map(|b| b.batch_size.is_some()).unwrap_or(false);

        let batch_size = indices.batch_size();

        if batch_size != output.batch_size() || batch_size != output_grads.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        D::backprop_sparse_affine_activate(
            batch_size.unwrap_or(1),
            *activation,
            &mut weights_grads.buf,
            *weights_shape,
            &indices.buf,
            values.map(|v| &v.buf),
            *input_shape,
            indices.nnz,
            biases_grads.map(|b| &mut b.buf),
            biases_batched,
            &output.buf,
            &output_grads.buf,
        )
    }
}
